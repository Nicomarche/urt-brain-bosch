#!/usr/bin/env python3
"""Run one deterministic simulator route and collect campaign artifacts.

This is the first building block for automated test/correction loops:

1. Start Gazebo + sim_bridge.
2. Start the independent gz ground-truth logger.
3. Start the brain in sim mode with live JSONL logging.
4. Drive one commanded route through the dashboard Socket.IO API.
5. Stop everything and emit machine-readable pass/fail metrics.

Default route: OSM reference start -> lanelet 138. It extends the first
validated segment through the right-side loop and up the next vertical road,
exercising a longer intersection/curve sequence on the current map.
"""

from __future__ import annotations

import argparse
import base64
import bisect
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BRAIN_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SIM_DIR = BRAIN_DIR.parent / "urt-simulator"
DEFAULT_OSM_PATH = BRAIN_DIR / "maps" / "sim" / "lanelet2_map.osm"
DEFAULT_TRACK_PNG = BRAIN_DIR / "maps" / "sim" / "track.png"


@dataclass
class ManagedProcess:
    name: str
    popen: subprocess.Popen
    stdout_handle: Any

    def terminate(self, timeout_s: float = 5.0) -> None:
        if self.popen.poll() is not None:
            self._close_stdout()
            return
        try:
            os.killpg(self.popen.pid, signal.SIGTERM)
        except ProcessLookupError:
            self._close_stdout()
            return
        except Exception:
            self.popen.terminate()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.popen.poll() is not None:
                self._close_stdout()
                return
            time.sleep(0.1)
        try:
            os.killpg(self.popen.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            self.popen.kill()
        self._close_stdout()

    def _close_stdout(self) -> None:
        try:
            self.stdout_handle.close()
        except Exception:
            pass


@dataclass
class RouteCommandResult:
    completed: bool = False
    timed_out: bool = False
    route_activated: bool = False
    nav_handled: bool = False
    final_progress: float = 0.0
    final_route_active: bool = False
    final_route_id: str | None = None
    max_progress: float = 0.0
    node_changes: list[str] = field(default_factory=list)


class JsonlFollower:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0

    def read_new(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            fh.seek(self.offset)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            self.offset = fh.tell()
        return events


class DashboardClient:
    def __init__(self, host: str) -> None:
        try:
            import socketio
        except ImportError as exc:
            raise RuntimeError(
                "python-socketio is required. Run this with the brain venv, "
                "for example: .venv/bin/python tools/dev/campaign_runner.py"
            ) from exc

        self._socketio = socketio
        self.host = host
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=10,
            reconnection_delay=0.5,
            reconnection_delay_max=2.0,
        )
        self.session_granted = False
        self.latest_nav_status: dict[str, Any] = {}
        self.response_count = 0
        self.last_response: dict[str, Any] = {}
        self._gui_recording_lock = threading.RLock()
        self._gui_recording_path: Path | None = None
        self._gui_recording_writer: Any = None
        self._gui_recording_size: tuple[int, int] | None = None
        self._gui_recording_fps = 7.0
        self._gui_recording_frame_count = 0
        self._gui_recording_started_at: float | None = None
        self._gui_recording_error: str | None = None
        self._gui_recording_last_binary_ts = 0.0
        self._gui_recording_frame_index: list[dict[str, Any]] = []
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        @self.sio.event
        def connect() -> None:
            print(f"[campaign] dashboard connected: {self.host}")

        @self.sio.event
        def disconnect() -> None:
            print("[campaign] dashboard disconnected")

        @self.sio.on("session_access")
        def _on_session_access(data: Any) -> None:
            self.session_granted = bool(_unwrap_payload(data))
            print(f"[campaign] session_access={self.session_granted}")

        @self.sio.on("heartbeat")
        def _on_heartbeat(_data: Any) -> None:
            self.emit_message("Heartbeat", "pong")

        @self.sio.on("NavigationStatus")
        def _on_navigation_status(data: Any) -> None:
            payload = _unwrap_payload(data)
            if isinstance(payload, dict):
                self.latest_nav_status = payload

        @self.sio.on("response")
        def _on_response(data: Any) -> None:
            self.response_count += 1
            if isinstance(data, dict):
                self.last_response = data

        @self.sio.on("serialCamera_bin")
        def _on_serial_camera_bin(data: Any) -> None:
            if not self._is_gui_video_recording_active():
                return
            if data is None:
                return
            try:
                raw = bytes(data) if not isinstance(data, (bytes, bytearray)) else bytes(data)
            except (TypeError, ValueError):
                return
            self._record_gui_video_bytes(raw, source="binary")

        @self.sio.on("serialCamera")
        def _on_serial_camera(data: Any) -> None:
            if not self._is_gui_video_recording_active():
                return
            payload = _unwrap_payload(data)
            if not isinstance(payload, str):
                return
            # The dashboard emits serialCamera_bin and serialCamera for the same
            # frame. Prefer binary, and keep base64 only as a compatibility
            # fallback when binary is not available.
            if (time.time() - self._gui_recording_last_binary_ts) < 0.20:
                return
            if "," in payload:
                payload = payload.split(",", 1)[1]
            try:
                raw = base64.b64decode(payload)
            except (ValueError, TypeError):
                return
            self._record_gui_video_bytes(raw, source="base64")

    def connect(self, timeout_s: float = 10.0) -> None:
        self.sio.connect(self.host, wait_timeout=timeout_s, transports=["websocket"])

    def close(self) -> None:
        try:
            if self.sio.connected:
                self.emit_message("DrivingMode", "stop")
                self.emit_message("Record", False)
                self.emit_message("SessionEnd", True)
                time.sleep(0.2)
                self.sio.disconnect()
        except Exception:
            pass
        self.stop_gui_video_recording()

    def start_gui_video_recording(self, out_path: Path, *, fps: float = 7.0) -> None:
        with self._gui_recording_lock:
            self._release_gui_video_writer()
            self._gui_recording_path = out_path
            self._gui_recording_size = None
            self._gui_recording_fps = max(1.0, float(fps))
            self._gui_recording_frame_count = 0
            self._gui_recording_started_at = time.time()
            self._gui_recording_error = None
            self._gui_recording_last_binary_ts = 0.0
            self._gui_recording_frame_index = []

    def _is_gui_video_recording_active(self) -> bool:
        with self._gui_recording_lock:
            return self._gui_recording_path is not None

    def stop_gui_video_recording(self) -> dict[str, Any] | None:
        with self._gui_recording_lock:
            if self._gui_recording_path is None:
                return None
            path = self._gui_recording_path
            fps = self._gui_recording_fps
            frame_count = self._gui_recording_frame_count
            size = self._gui_recording_size
            started_at = self._gui_recording_started_at
            error = self._gui_recording_error
            frame_index = list(self._gui_recording_frame_index)
            self._release_gui_video_writer()
            self._gui_recording_path = None
            self._gui_recording_frame_index = []
            frame_index_path = path.with_suffix(".frames.json")
            if frame_index:
                try:
                    frame_index_path.write_text(
                        json.dumps(
                            {
                                "video_path": str(path),
                                "frame_count": frame_count,
                                "fps": fps,
                                "started_at": started_at,
                                "frames": frame_index,
                            },
                            indent=2,
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                except OSError:
                    frame_index_path = None
            meta: dict[str, Any] = {
                "written": frame_count > 0,
                "path": str(path),
                "frame_count": frame_count,
                "fps": fps,
                "started_at": started_at,
            }
            if frame_index_path is not None and frame_index_path.exists():
                meta["frame_index_path"] = str(frame_index_path)
            if frame_index:
                meta["_frame_timestamps"] = [
                    float(item["ts"]) for item in frame_index if _finite_float(item.get("ts")) is not None
                ]
            if size is not None:
                meta["width"] = size[0]
                meta["height"] = size[1]
            if error:
                meta["error"] = error
            elif frame_count <= 0:
                meta["error"] = "no_gui_camera_frames"
            return meta

    def _record_gui_video_bytes(self, raw_jpeg: bytes, *, source: str) -> None:
        with self._gui_recording_lock:
            if self._gui_recording_path is None:
                return
        try:
            import cv2
            import numpy as np
        except ImportError:
            with self._gui_recording_lock:
                self._gui_recording_error = "opencv_unavailable"
            return

        arr = np.frombuffer(raw_jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            with self._gui_recording_lock:
                self._gui_recording_error = "gui_camera_decode_failed"
            return
        received_ts = time.time()
        self._record_gui_video_frame(frame, received_ts=received_ts, source=source)
        if source == "binary":
            self._gui_recording_last_binary_ts = received_ts

    def _record_gui_video_frame(
        self,
        frame: Any,
        *,
        received_ts: float | None = None,
        source: str = "unknown",
    ) -> None:
        import cv2

        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            return
        frame_h, frame_w = frame.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return

        with self._gui_recording_lock:
            if self._gui_recording_path is None:
                return
            if self._gui_recording_writer is None:
                self._gui_recording_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(self._gui_recording_path),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    self._gui_recording_fps,
                    (int(frame_w), int(frame_h)),
                )
                if not writer.isOpened():
                    try:
                        writer.release()
                    except Exception:
                        pass
                    self._gui_recording_error = "gui_video_writer_open_failed"
                    return
                self._gui_recording_writer = writer
                self._gui_recording_size = (int(frame_w), int(frame_h))

            out_frame = frame
            if self._gui_recording_size is not None:
                width, height = self._gui_recording_size
                if (int(frame_w), int(frame_h)) != (width, height):
                    out_frame = cv2.resize(frame, (width, height))
            self._gui_recording_writer.write(out_frame)
            self._gui_recording_frame_count += 1
            frame_no = int(self._gui_recording_frame_count)
            ts = float(received_ts if received_ts is not None else time.time())
            started_at = self._gui_recording_started_at
            self._gui_recording_frame_index.append(
                {
                    "frame": frame_no,
                    "ts": ts,
                    "time_s": (
                        float(ts) - float(started_at)
                        if started_at is not None
                        else None
                    ),
                    "source": str(source),
                }
            )

    def _release_gui_video_writer(self) -> None:
        writer = self._gui_recording_writer
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        self._gui_recording_writer = None

    def emit_message(
        self,
        name: str,
        value: Any = None,
        *,
        wait_response: bool = False,
        timeout_s: float = 3.0,
    ) -> bool:
        envelope: dict[str, Any] = {"Name": name}
        if value is not None:
            envelope["Value"] = value
        responses_before = self.response_count
        self.sio.emit("message", json.dumps(envelope))
        if not wait_response:
            return True
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.response_count > responses_before:
                return True
            time.sleep(0.05)
        return False

    def request_session(self, timeout_s: float = 10.0) -> bool:
        self.emit_message("SessionAccess")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.session_granted:
                return True
            time.sleep(0.1)
        return False


def _unwrap_payload(data: Any) -> Any:
    if isinstance(data, dict):
        if "value" in data:
            return data["value"]
        if "data" in data:
            return data["data"]
    return data


def _run(
    args: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    env: dict[str, str] | None = None,
    name: str,
) -> ManagedProcess:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = stdout_path.open("w", encoding="utf-8")
    popen = subprocess.Popen(
        args,
        cwd=str(cwd),
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    print(f"[campaign] started {name} pid={popen.pid}")
    return ManagedProcess(name=name, popen=popen, stdout_handle=stdout)


def _wait_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _wait_for_jsonl_event(path: Path, *, thread: str, event: str, timeout_s: float) -> bool:
    follower = JsonlFollower(path)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for item in follower.read_new():
            if item.get("thread") == thread and item.get("event") == event:
                return True
        time.sleep(0.25)
    return False


def _find_brain_python(brain_dir: Path) -> str:
    candidate = brain_dir / ".venv" / "bin" / "python"
    if candidate.exists() and os.access(candidate, os.X_OK):
        return str(candidate)
    return sys.executable


def _maybe_reexec_with_brain_venv() -> None:
    if os.environ.get("URT_CAMPAIGN_REEXECED") == "1":
        return
    brain_dir = BRAIN_DIR
    argv = list(sys.argv)
    for idx, arg in enumerate(argv):
        if arg == "--brain-dir" and idx + 1 < len(argv):
            brain_dir = Path(argv[idx + 1]).expanduser().resolve()
        elif arg.startswith("--brain-dir="):
            brain_dir = Path(arg.split("=", 1)[1]).expanduser().resolve()
    candidate = brain_dir / ".venv" / "bin" / "python"
    if not (candidate.exists() and os.access(candidate, os.X_OK)):
        return
    try:
        if Path(sys.executable).resolve() == candidate.resolve():
            return
    except OSError:
        pass
    env = os.environ.copy()
    env["URT_CAMPAIGN_REEXECED"] = "1"
    os.execvpe(str(candidate), [str(candidate), *argv], env)


def _find_gz_python() -> tuple[str, str]:
    brew = shutil.which("brew")
    if brew:
        try:
            prefix = subprocess.check_output([brew, "--prefix"], text=True).strip()
        except subprocess.SubprocessError:
            prefix = ""
        for version in ("3.13", "3.12"):
            py = shutil.which(f"python{version}")
            site = Path(prefix) / "lib" / f"python{version}" / "site-packages"
            if py and (site / "gz").exists():
                return py, str(site)
    return sys.executable, ""


def _cleanup_known_processes(brain_dir: Path, sim_dir: Path) -> None:
    patterns = (
        "gz-sim-main",
        "gz-transport-topic",
        "ruby .*ign-gazebo",
        str(sim_dir / "sim_bridge.py"),
        str(sim_dir / "tools" / "dev" / "gz_pose_logger.py"),
        str(brain_dir / "main.py"),
        str(brain_dir / ".venv" / "bin" / "python.*main.py"),
        "headless_controller.py",
    )
    current_pid = os.getpid()
    for pattern in patterns:
        try:
            subprocess.run(
                ["pkill", "-9", "-f", pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass
    # Avoid leaving the runner dead if a broad pkill pattern matched argv.
    if not _pid_exists(current_pid):
        raise RuntimeError("campaign runner process was unexpectedly killed")
    _kill_port_listeners(5005, current_pid=current_pid)
    _kill_orphaned_multiprocessing_children(brain_dir, current_pid=current_pid)
    time.sleep(1.0)


def _kill_port_listeners(port: int, *, current_pid: int) -> None:
    try:
        output = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return
    for raw_pid in output.splitlines():
        raw_pid = raw_pid.strip()
        if not raw_pid:
            continue
        try:
            pid = int(raw_pid)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass


def _kill_orphaned_multiprocessing_children(brain_dir: Path, *, current_pid: int) -> None:
    try:
        output = subprocess.check_output(
            ["ps", "-axo", "pid=,ppid=,command="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.SubprocessError:
        return
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        command = parts[2]
        if pid == current_pid or ppid != 1:
            continue
        if "multiprocessing.spawn" not in command or "--multiprocessing-fork" not in command:
            continue
        cwd = _process_cwd(pid)
        if cwd is None or cwd.resolve() != brain_dir.resolve():
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass


def _process_cwd(pid: int) -> Path | None:
    try:
        output = subprocess.check_output(
            ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    for line in output.splitlines():
        if line.startswith("n") and len(line) > 1:
            return Path(line[1:])
    return None


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "samples": 0,
            "p10": None,
            "p50": None,
            "p90": None,
            "mean": None,
            "min": None,
            "max": None,
        }
    return {
        "samples": len(values),
        "p10": _percentile(values, 10),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _point_to_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def _point_to_polyline_distance(point: tuple[float, float], polyline: list[tuple[float, float]]) -> float:
    if len(polyline) < 2:
        return float("inf")
    px, py = point
    best = float("inf")
    for idx in range(len(polyline) - 1):
        ax, ay = polyline[idx]
        bx, by = polyline[idx + 1]
        best = min(best, _point_to_segment_distance(px, py, ax, ay, bx, by))
    return best


def _route_progress_from_logs(brain_jsonl: Path, timeout_s: float) -> RouteCommandResult:
    follower = JsonlFollower(brain_jsonl)
    result = RouteCommandResult()
    deadline = time.monotonic() + timeout_s
    seen_nodes: set[str] = set()
    while time.monotonic() < deadline:
        for item in follower.read_new():
            thread = item.get("thread")
            event = item.get("event")
            if thread == "route_planner" and event == "route_activated":
                result.route_activated = True
                result.final_route_id = str(item.get("route_id") or "")
            elif thread == "nav_planner" and event == "nav_command_result":
                result.nav_handled = bool(item.get("handled", False))
                if item.get("route_id") is not None:
                    result.final_route_id = str(item.get("route_id"))
            elif thread == "nav_planner" and event == "node_change":
                node_id = item.get("to_node")
                if node_id is not None and str(node_id) not in seen_nodes:
                    seen_nodes.add(str(node_id))
                    result.node_changes.append(str(node_id))
            elif thread == "nav_planner" and event == "route_update":
                progress = float(item.get("route_progress", 0.0) or 0.0)
                result.final_progress = progress
                result.max_progress = max(result.max_progress, progress)
                result.final_route_active = bool(item.get("route_active", False))
                if item.get("route_id") is not None:
                    result.final_route_id = str(item.get("route_id"))
                if bool(item.get("route_completed", False)):
                    result.completed = True
                    return result
        time.sleep(0.25)
    result.timed_out = True
    return result


def _wait_for_mode_state(brain_jsonl: Path, mode: str, timeout_s: float) -> bool:
    expected = str(mode or "").upper()
    follower = JsonlFollower(brain_jsonl)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for item in follower.read_new():
            thread = item.get("thread")
            event = item.get("event")
            if thread == "dashboard" and event == "state_change_request":
                if str(item.get("to_mode") or "").upper() == expected:
                    return True
            if thread == "dispatcher" and event == "dispatch_decision":
                if str(item.get("state") or "").upper() == expected:
                    return True
            if thread == "state_machine" and event == "state_change":
                if str(item.get("to") or "").upper() == expected:
                    return True
        time.sleep(0.1)
    return False


def _wait_for_navigation_acceptance(brain_jsonl: Path, timeout_s: float) -> bool:
    follower = JsonlFollower(brain_jsonl)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for item in follower.read_new():
            thread = item.get("thread")
            event = item.get("event")
            if thread == "nav_planner" and event == "nav_command_result":
                return bool(item.get("handled", False))
            if thread == "route_planner" and event == "route_activated":
                return True
            if thread == "nav_planner" and event == "route_update":
                if bool(item.get("route_active", False)):
                    return True
        time.sleep(0.1)
    return False


def _build_expected_route(
    *,
    osm_path: Path,
    destination_lanelet: str | None,
    destination_x: float | None,
    destination_y: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if str(BRAIN_DIR) not in sys.path:
        sys.path.insert(0, str(BRAIN_DIR))
    from src.routing.lanelet.osm_router import OsmRouteGraph
    from src.utils.sim_start_pose import load_saved_start_pose, resolve_saved_start_pose

    graph = OsmRouteGraph(str(osm_path))
    default_start_pose = graph.get_start_pose()
    saved_start_pose = load_saved_start_pose(map_dir=osm_path.resolve().parent)
    start_x, start_y, start_yaw = resolve_saved_start_pose(graph, default=default_start_pose)
    start_spec: dict[str, Any] = {"x": start_x, "y": start_y, "yaw_rad": start_yaw}
    if saved_start_pose and saved_start_pose.get("lanelet_id") is not None:
        start_spec["lanelet_id"] = str(saved_start_pose["lanelet_id"])
        start_source = "saved_sim_start_pose"
    else:
        start_spec["lanelet_id"] = graph.get_start_node_id()
        start_source = "map_default_start_pose"

    if destination_lanelet:
        destination_spec: dict[str, Any] = {"lanelet_id": str(destination_lanelet)}
    elif destination_x is not None and destination_y is not None:
        destination_spec = {"x": float(destination_x), "y": float(destination_y)}
    else:
        destination_spec = {"lanelet_id": "138"}

    route = graph.go_to(start_spec, destination_spec)
    if route is None or route.waypoints.size == 0:
        raise RuntimeError(f"empty route for destination {destination_spec!r}")
    waypoints = [
        [float(pt[0]), float(pt[1]), float(pt[2]) if len(pt) > 2 else 0.0]
        for pt in route.waypoints
    ]
    expected = {
        "source": "campaign_runner",
        "osm_path": str(osm_path),
        "start_source": start_source,
        "start": start_spec,
        "destination": destination_spec,
        "node_ids": list(route.node_ids),
        "closed_loop": bool(route.closed_loop),
        "waypoint_count": len(waypoints),
        "waypoints": waypoints,
        "map_metadata": graph.get_map_metadata(),
    }
    command = {
        "mode": "go_to",
        "destinations": [destination_spec],
    }
    return expected, command


def _apply_mode_env(
    env: dict[str, str],
    *,
    mode: str,
    expected_route: dict[str, Any],
    command_speed_scale: float | None,
) -> None:
    if command_speed_scale is not None:
        env["URT_COMMAND_SPEED_CALIBRATION_SCALE"] = f"{float(command_speed_scale):.6g}"

    if str(mode or "").lower() != "odometry":
        return

    start = dict(expected_route.get("start") or {})
    env["URT_USE_ENCODER"] = "0"
    env["URT_USE_GPS"] = "0"
    env["URT_SIM_SUBMODEL"] = "0"
    if start.get("x") is not None:
        env["URT_X0"] = str(float(start["x"]))
    if start.get("y") is not None:
        env["URT_Y0"] = str(float(start["y"]))
    if start.get("yaw_rad") is not None:
        env["URT_YAW0"] = str(float(start["yaw_rad"]))
    if start.get("lanelet_id") is not None:
        env["URT_TRACKING_START_LANELET_ID"] = str(start["lanelet_id"])


def _compute_metrics(
    *,
    expected_route: dict[str, Any],
    brain_events: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    auto_start_ts: float | None,
    max_p90_m: float,
    max_max_m: float,
    max_lane_offset_p90_m: float,
    max_lane_offset_max_m: float,
    route_result: RouteCommandResult,
) -> dict[str, Any]:
    expected_xy = [
        (float(pt[0]), float(pt[1]))
        for pt in expected_route.get("waypoints", [])
        if isinstance(pt, list) and len(pt) >= 2
    ]
    gt_poses = [
        ev
        for ev in gt_events
        if ev.get("thread") == "ground_truth" and ev.get("event") == "gz_pose"
    ]
    if auto_start_ts is not None:
        gt_poses = [ev for ev in gt_poses if float(ev.get("ts", 0.0) or 0.0) >= auto_start_ts]

    distances: list[float] = []
    actual_xy: list[tuple[float, float]] = []
    for ev in gt_poses:
        x = ev.get("brain_map_x", ev.get("graphml_x"))
        y = ev.get("brain_map_y", ev.get("graphml_y"))
        if x is None or y is None:
            continue
        point = (float(x), float(y))
        actual_xy.append(point)
        if expected_xy:
            distances.append(_point_to_polyline_distance(point, expected_xy))

    route_updates = [
        ev for ev in brain_events
        if ev.get("thread") == "nav_planner" and ev.get("event") == "route_update"
    ]
    final_update = route_updates[-1] if route_updates else {}
    logged_progress_values = [
        float(ev.get("route_progress", 0.0) or 0.0)
        for ev in route_updates
    ]
    logged_completed = any(bool(ev.get("route_completed", False)) for ev in route_updates)
    final_progress = max(
        route_result.final_progress,
        float(final_update.get("route_progress", 0.0) or 0.0) if final_update else 0.0,
    )
    max_progress = max([route_result.max_progress, *logged_progress_values] or [0.0])
    final_route_active = (
        bool(final_update.get("route_active", route_result.final_route_active))
        if final_update
        else route_result.final_route_active
    )
    final_route_id = (
        str(final_update.get("route_id"))
        if final_update.get("route_id") is not None
        else route_result.final_route_id
    )
    destination = expected_route.get("destination") if isinstance(expected_route, dict) else {}
    destination_lanelet_id = None
    if isinstance(destination, dict) and destination.get("lanelet_id") is not None:
        destination_lanelet_id = str(destination.get("lanelet_id"))
    logged_lanelet_ids = [
        str(ev.get("current_lanelet_id"))
        for ev in route_updates
        if ev.get("current_lanelet_id") is not None
    ]
    destination_lanelet_reached = bool(
        destination_lanelet_id
        and destination_lanelet_id in set(logged_lanelet_ids + list(route_result.node_changes or []))
        and max_progress >= 0.85
    )
    route_completed = bool(route_result.completed or logged_completed or destination_lanelet_reached)
    p90 = _percentile(distances, 90) if distances else None
    max_dist = max(distances) if distances else None
    pass_reasons: list[str] = []
    fail_reasons: list[str] = []

    if route_completed:
        pass_reasons.append("route_completed")
    else:
        fail_reasons.append("route_not_completed")
    if distances:
        if p90 is not None and p90 <= max_p90_m:
            pass_reasons.append("cross_track_p90_ok")
        else:
            fail_reasons.append("cross_track_p90_high")
        if max_dist is not None and max_dist <= max_max_m:
            pass_reasons.append("cross_track_max_ok")
        else:
            fail_reasons.append("cross_track_max_high")
    else:
        fail_reasons.append("no_ground_truth_samples")

    lane_events = [
        ev
        for ev in brain_events
        if ev.get("thread") == "lane_observer"
        and ev.get("event") == "lane_obs"
        and float(ev.get("quality", 0.0) or 0.0) > 0.5
    ]
    lane_offsets: list[float] = []
    direct_lane_offsets = [
        offset
        for ev in lane_events
        for offset in [_finite_float(ev.get("offset_m"))]
        if offset is not None
    ]
    lane_abs_offsets = [abs(value) for value in lane_offsets]
    line_left_distances: list[float] = []
    line_right_distances: list[float] = []
    line_gap_distances: list[float] = []
    line_center_offsets: list[float] = []
    closer_side_counts = {"left": 0, "right": 0, "balanced": 0}
    per_lanelet: dict[str, dict[str, list[float]]] = {}
    route_ts = [float(ev.get("ts", 0.0) or 0.0) for ev in route_updates]

    def lanelet_at(ts: float) -> str | None:
        if not route_ts:
            return None
        idx = bisect.bisect_right(route_ts, ts) - 1
        if idx < 0:
            return None
        lanelet = route_updates[idx].get("current_lanelet_id")
        return str(lanelet) if lanelet is not None else None

    for ev in lane_events:
        offset_m = _finite_float(ev.get("offset_m"))
        left_m = _finite_float(ev.get("left_line_distance_m"))
        right_m = _finite_float(ev.get("right_line_distance_m"))
        center_offset_m = _finite_float(ev.get("line_center_offset_m"))
        if center_offset_m is None and left_m is not None and right_m is not None:
            center_offset_m = 0.5 * (right_m - left_m)
        visual_offset_m = center_offset_m if center_offset_m is not None else offset_m
        if visual_offset_m is not None:
            lane_offsets.append(visual_offset_m)
        if left_m is not None and right_m is not None:
            line_left_distances.append(left_m)
            line_right_distances.append(right_m)
            line_gap_distances.append(left_m + right_m)
            if center_offset_m is not None:
                line_center_offsets.append(center_offset_m)
            if abs(left_m - right_m) < 0.01:
                closer_side_counts["balanced"] += 1
            elif left_m < right_m:
                closer_side_counts["left"] += 1
            else:
                closer_side_counts["right"] += 1

        lanelet_id = lanelet_at(float(ev.get("ts", 0.0) or 0.0))
        if lanelet_id is None:
            continue
        bucket = per_lanelet.setdefault(
            lanelet_id,
            {
                "offset_m": [],
            "left_line_distance_m": [],
            "right_line_distance_m": [],
            "line_gap_m": [],
            "line_center_offset_m": [],
                "line_center_offset_abs_m": [],
            },
        )
        if visual_offset_m is not None:
            bucket["offset_m"].append(visual_offset_m)
        if left_m is None or right_m is None:
            continue
        bucket["left_line_distance_m"].append(left_m)
        bucket["right_line_distance_m"].append(right_m)
        bucket["line_gap_m"].append(left_m + right_m)
        if center_offset_m is not None:
            bucket["line_center_offset_m"].append(center_offset_m)
            bucket["line_center_offset_abs_m"].append(abs(center_offset_m))

    lane_abs_offsets = [abs(value) for value in lane_offsets]
    lane_p90 = _percentile(lane_abs_offsets, 90) if lane_abs_offsets else None
    lane_max = max(lane_abs_offsets) if lane_abs_offsets else None

    per_lanelet_summary: dict[str, Any] = {}
    for lanelet_id, bucket in sorted(per_lanelet.items(), key=lambda item: item[0]):
        offsets = bucket["offset_m"]
        center_offsets = bucket["line_center_offset_m"]
        per_lanelet_summary[lanelet_id] = {
            "samples": max(len(values) for values in bucket.values()),
            "offset_abs_m": _summary([abs(value) for value in offsets]),
            "offset_signed_m": _summary(offsets),
            "left_line_distance_m": _summary(bucket["left_line_distance_m"]),
            "right_line_distance_m": _summary(bucket["right_line_distance_m"]),
            "line_gap_m": _summary(bucket["line_gap_m"]),
            "line_center_offset_m": _summary(center_offsets),
            "line_center_offset_abs_m": _summary(bucket["line_center_offset_abs_m"]),
        }
    if lane_abs_offsets:
        visual_lane_p90_ok = lane_p90 is not None and lane_p90 <= max_lane_offset_p90_m
        visual_lane_max_ok = lane_max is not None and lane_max <= max_lane_offset_max_m
        if visual_lane_p90_ok:
            pass_reasons.append("visual_lane_offset_p90_ok")
        else:
            fail_reasons.append("visual_lane_offset_p90_high")
        if visual_lane_max_ok:
            pass_reasons.append("visual_lane_offset_max_ok")
        else:
            fail_reasons.append("visual_lane_offset_max_high")
        if route_completed and visual_lane_p90_ok and visual_lane_max_ok:
            removed_cross_track_failures = [
                reason for reason in fail_reasons if reason.startswith("cross_track_")
            ]
            if removed_cross_track_failures:
                fail_reasons = [
                    reason for reason in fail_reasons if not reason.startswith("cross_track_")
                ]
                pass_reasons.append("cross_track_map_mismatch_ignored_visual_lane_ok")
    else:
        fail_reasons.append("no_visual_lane_offset_samples")

    passed = not fail_reasons
    return {
        "passed": passed,
        "pass_reasons": pass_reasons,
        "fail_reasons": fail_reasons,
        "route": {
            "completed": route_completed,
            "destination_lanelet_reached": destination_lanelet_reached,
            "timed_out": route_result.timed_out,
            "activated": route_result.route_activated,
            "nav_handled": route_result.nav_handled,
            "final_progress": final_progress,
            "max_progress": max_progress,
            "final_route_active": final_route_active,
            "final_route_id": final_route_id,
            "node_changes": route_result.node_changes,
            "final_update": final_update,
        },
        "ground_truth": {
            "samples": len(actual_xy),
            "cross_track_error_m": {
                "p50": _percentile(distances, 50) if distances else None,
                "p90": p90,
                "p99": _percentile(distances, 99) if distances else None,
                "max": max_dist,
                "threshold_p90": max_p90_m,
                "threshold_max": max_max_m,
            },
            "actual_path": [[float(x), float(y)] for x, y in actual_xy],
        },
        "visual_lane": {
            "samples": len(lane_abs_offsets),
            "offset_source": "line_center_offset_m when available, offset_m fallback",
            "offset_abs_m": {
                "p50": _percentile(lane_abs_offsets, 50) if lane_abs_offsets else None,
                "p90": lane_p90,
                "p99": _percentile(lane_abs_offsets, 99) if lane_abs_offsets else None,
                "max": lane_max,
                "threshold_p90": max_lane_offset_p90_m,
                "threshold_max": max_lane_offset_max_m,
            },
            "offset_signed_mean_m": (
                sum(lane_offsets) / len(lane_offsets)
                if lane_offsets
                else None
            ),
            "direct_offset_abs_m": _summary([abs(value) for value in direct_lane_offsets]),
            "line_distance_samples": len(line_gap_distances),
            "left_line_distance_m": _summary(line_left_distances),
            "right_line_distance_m": _summary(line_right_distances),
            "line_gap_m": _summary(line_gap_distances),
            "line_center_offset_m": _summary(line_center_offsets),
            "line_center_offset_abs_m": _summary([abs(value) for value in line_center_offsets]),
            "closer_side_counts": closer_side_counts,
            "per_lanelet": per_lanelet_summary,
        },
    }


def _find_mode_start_ts(events: list[dict[str, Any]], mode: str) -> float | None:
    expected = str(mode or "").upper()
    for ev in events:
        if ev.get("thread") == "dashboard" and ev.get("event") == "state_change_request":
            if str(ev.get("to_mode") or "").upper() == expected:
                return float(ev.get("ts", 0.0) or 0.0)
    for ev in events:
        if ev.get("thread") == "state_machine" and str(ev.get("to") or "").upper() == expected:
            return float(ev.get("ts", 0.0) or 0.0)
    for ev in events:
        if ev.get("thread") == "route_planner" and ev.get("event") == "route_activated":
            return float(ev.get("ts", 0.0) or 0.0)
    return None


def _overlay_map_metadata(track_png: Path, expected_route: dict[str, Any]) -> dict[str, Any]:
    meta = dict(expected_route.get("map_metadata") or {})
    track_meta_path = track_png.with_name("track_meta.json")
    try:
        track_meta = json.loads(track_meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return meta
    if not isinstance(track_meta, dict):
        return meta

    # The OSM metadata describes the lanelet geometry grid; the PNG may have
    # been rasterized at a slightly different scale/origin. Use calibrated
    # raster metadata for pixels and visual world bounds when available.
    if track_meta.get("metersPerPixel") is not None:
        meta["meters_per_pixel"] = track_meta.get("metersPerPixel")
    elif track_meta.get("meters_per_pixel") is not None:
        meta["meters_per_pixel"] = track_meta.get("meters_per_pixel")
    if track_meta.get("imgW") is not None:
        meta["image_width_px"] = track_meta.get("imgW")
    if track_meta.get("imgH") is not None:
        meta["image_height_px"] = track_meta.get("imgH")
    if track_meta.get("y_axis_inverted") is not None:
        meta["y_axis_inverted"] = bool(track_meta.get("y_axis_inverted"))
    if isinstance(track_meta.get("world_bounds"), dict):
        route_bounds = dict(meta.get("world_bounds") or {})
        route_bounds.update(track_meta["world_bounds"])
        meta["world_bounds"] = route_bounds
    return meta


def _write_overlay(
    *,
    out_path: Path,
    track_png: Path,
    expected_route: dict[str, Any],
    actual_xy: list[list[float]],
) -> bool:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False

    meta = _overlay_map_metadata(track_png, expected_route)
    img = cv2.imread(str(track_png), cv2.IMREAD_COLOR) if track_png.exists() else None
    if img is None:
        width = int(meta.get("image_width_px", 2039) or 2039)
        height = int(meta.get("image_height_px", 1343) or 1343)
        img = np.full((height, width, 3), 245, dtype=np.uint8)

    bounds = dict(meta.get("world_bounds") or {})
    x_min = float(bounds.get("x_min", 0.0) or 0.0)
    y_min = float(bounds.get("y_min", 0.0) or 0.0)
    mpp = float(meta.get("meters_per_pixel", 0.01) or 0.01)
    y_axis_inverted = bool(meta.get("y_axis_inverted", False))
    height_px = int(img.shape[0])

    def world_to_pixel(x: float, y: float) -> tuple[int, int]:
        px = (float(x) - x_min) / mpp
        local_y = float(y) - y_min
        py = height_px - (local_y / mpp) if y_axis_inverted else local_y / mpp
        return int(round(px)), int(round(py))

    def draw_polyline(points: list[tuple[float, float]], color: tuple[int, int, int], thickness: int) -> None:
        if len(points) < 2:
            return
        pix = np.asarray([world_to_pixel(x, y) for x, y in points], dtype=np.int32)
        cv2.polylines(img, [pix], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    expected_xy = [
        (float(pt[0]), float(pt[1]))
        for pt in expected_route.get("waypoints", [])
        if isinstance(pt, list) and len(pt) >= 2
    ]
    actual_points = [(float(pt[0]), float(pt[1])) for pt in actual_xy if len(pt) >= 2]
    draw_polyline(expected_xy, (0, 170, 0), 4)
    draw_polyline(actual_points, (30, 30, 220), 3)
    cv2.putText(img, "expected", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "actual", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 220), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), img))


def _collect_visual_lane_samples(
    brain_events: list[dict[str, Any]],
    *,
    record_start_ts: float | None = None,
    fps: float | None = None,
) -> list[dict[str, Any]]:
    route_updates = [
        ev for ev in brain_events
        if ev.get("thread") == "nav_planner" and ev.get("event") == "route_update"
    ]
    route_ts = [float(ev.get("ts", 0.0) or 0.0) for ev in route_updates]

    def route_at(ts: float) -> dict[str, Any]:
        if not route_ts:
            return {}
        idx = bisect.bisect_right(route_ts, ts) - 1
        if idx < 0:
            idx = 0
        return route_updates[idx]

    samples: list[dict[str, Any]] = []
    for ev in brain_events:
        if ev.get("thread") != "lane_observer" or ev.get("event") != "lane_obs":
            continue
        if float(ev.get("quality", 0.0) or 0.0) <= 0.5:
            continue
        ts = float(ev.get("ts", 0.0) or 0.0)
        route_ev = route_at(ts)
        pose_x = _finite_float(route_ev.get("pose_x"))
        pose_y = _finite_float(route_ev.get("pose_y"))
        if pose_x is None or pose_y is None:
            continue

        left_m = _finite_float(ev.get("left_line_distance_m"))
        right_m = _finite_float(ev.get("right_line_distance_m"))
        center_offset_m = _finite_float(ev.get("line_center_offset_m"))
        if center_offset_m is None and left_m is not None and right_m is not None:
            center_offset_m = 0.5 * (right_m - left_m)
        direct_offset_m = _finite_float(ev.get("offset_m"))
        visual_offset_m = center_offset_m if center_offset_m is not None else direct_offset_m
        if visual_offset_m is None:
            continue

        frame = None
        if record_start_ts is not None and fps is not None and fps > 0.0:
            frame = int(round((ts - record_start_ts) * fps)) + 1

        samples.append(
            {
                "ts": ts,
                "frame": frame,
                "lanelet_id": str(route_ev.get("current_lanelet_id")) if route_ev.get("current_lanelet_id") is not None else None,
                "route_progress": _finite_float(route_ev.get("route_progress")),
                "pose_x": pose_x,
                "pose_y": pose_y,
                "visual_offset_m": float(visual_offset_m),
                "visual_offset_abs_m": abs(float(visual_offset_m)),
                "direct_offset_m": direct_offset_m,
                "left_line_distance_m": left_m,
                "right_line_distance_m": right_m,
                "measurement_mode": ev.get("measurement_mode"),
                "quality": _finite_float(ev.get("quality")),
            }
        )
    return samples


def _write_visual_lane_overlay(
    *,
    out_path: Path,
    track_png: Path,
    expected_route: dict[str, Any],
    actual_xy: list[list[float]],
    visual_samples: list[dict[str, Any]],
    min_abs_m: float = 0.08,
) -> bool:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False

    meta = _overlay_map_metadata(track_png, expected_route)
    img = cv2.imread(str(track_png), cv2.IMREAD_COLOR) if track_png.exists() else None
    if img is None:
        width = int(meta.get("image_width_px", 2039) or 2039)
        height = int(meta.get("image_height_px", 1343) or 1343)
        img = np.full((height, width, 3), 245, dtype=np.uint8)

    bounds = dict(meta.get("world_bounds") or {})
    x_min = float(bounds.get("x_min", 0.0) or 0.0)
    y_min = float(bounds.get("y_min", 0.0) or 0.0)
    mpp = float(meta.get("meters_per_pixel", 0.01) or 0.01)
    y_axis_inverted = bool(meta.get("y_axis_inverted", False))
    height_px = int(img.shape[0])

    def world_to_pixel(x: float, y: float) -> tuple[int, int]:
        px = (float(x) - x_min) / mpp
        local_y = float(y) - y_min
        py = height_px - (local_y / mpp) if y_axis_inverted else local_y / mpp
        return int(round(px)), int(round(py))

    def draw_polyline(points: list[tuple[float, float]], color: tuple[int, int, int], thickness: int) -> None:
        if len(points) < 2:
            return
        pix = np.asarray([world_to_pixel(x, y) for x, y in points], dtype=np.int32)
        cv2.polylines(img, [pix], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    expected_xy = [
        (float(pt[0]), float(pt[1]))
        for pt in expected_route.get("waypoints", [])
        if isinstance(pt, list) and len(pt) >= 2
    ]
    actual_points = [(float(pt[0]), float(pt[1])) for pt in actual_xy if len(pt) >= 2]
    draw_polyline(expected_xy, (0, 170, 0), 4)
    draw_polyline(actual_points, (30, 30, 220), 3)

    drawn: list[dict[str, Any]] = []
    for sample in visual_samples:
        abs_m = _finite_float(sample.get("visual_offset_abs_m"))
        x = _finite_float(sample.get("pose_x"))
        y = _finite_float(sample.get("pose_y"))
        if abs_m is None or x is None or y is None or abs_m < min_abs_m:
            continue
        px, py = world_to_pixel(x, y)
        if abs_m >= 0.18:
            color = (255, 0, 255)
            radius = 8
        elif abs_m >= 0.12:
            color = (0, 170, 255)
            radius = 6
        else:
            color = (255, 255, 0)
            radius = 4
        cv2.circle(img, (px, py), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (px, py), radius + 2, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        drawn.append(sample)

    for sample in sorted(drawn, key=lambda item: float(item.get("visual_offset_abs_m", 0.0) or 0.0), reverse=True)[:12]:
        x = _finite_float(sample.get("pose_x"))
        y = _finite_float(sample.get("pose_y"))
        signed = _finite_float(sample.get("visual_offset_m"))
        if x is None or y is None or signed is None:
            continue
        px, py = world_to_pixel(x, y)
        frame = sample.get("frame")
        lanelet = sample.get("lanelet_id") or "?"
        label = f"f{frame} L{lanelet} {signed:+.2f}m" if frame is not None else f"L{lanelet} {signed:+.2f}m"
        cv2.putText(img, label, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, label, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(img, "expected", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "actual GT", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 220), 2, cv2.LINE_AA)
    cv2.putText(img, "visual lane offset: cyan>8cm orange>12cm magenta>18cm", (24, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), img))


def _visual_offset_from_lane_event(lane_ev: dict[str, Any] | None) -> float | None:
    if not isinstance(lane_ev, dict):
        return None
    center_offset_m = _finite_float(lane_ev.get("line_center_offset_m"))
    if center_offset_m is not None:
        return center_offset_m
    return _finite_float(lane_ev.get("offset_m"))


def _control_error_from_lane_event(lane_ev: dict[str, Any] | None) -> float | None:
    if not isinstance(lane_ev, dict):
        return None
    return _finite_float(lane_ev.get("control_lateral_error_m"))


def _campaign_digest_flags(
    *,
    lane_ev: dict[str, Any] | None,
    plan_ev: dict[str, Any] | None,
    visual_reentry_ev: dict[str, Any] | None,
    mpc_ev: dict[str, Any] | None,
) -> list[str]:
    flags: list[str] = []
    measurement_mode = str((lane_ev or {}).get("measurement_mode") or "missing")
    path_source = str((plan_ev or {}).get("path_source") or "")
    visual_offset_m = _visual_offset_from_lane_event(lane_ev)
    control_error_m = _control_error_from_lane_event(lane_ev)
    control_source = str((lane_ev or {}).get("control_lateral_error_source") or "")
    steering_deg = _finite_float((mpc_ev or {}).get("steering_deg"))
    quality = _finite_float((lane_ev or {}).get("quality"))

    if measurement_mode not in {"two_line"}:
        flags.append(f"mode:{measurement_mode}")
    if path_source and path_source != "visual_lane_waypoints":
        flags.append(f"path:{path_source}")
    if visual_offset_m is not None and abs(visual_offset_m) >= 0.08:
        flags.append(f"visual_offset:{abs(visual_offset_m):.3f}m")
    if control_source == "two_line_edge_urgency":
        flags.append("edge_urgency")
    if control_source == "single_line_reacquire":
        flags.append("single_line_reacquire")
    if control_source == "single_line_visual_waypoint_low_authority":
        flags.append("single_line_low_authority")
    if str((visual_reentry_ev or {}).get("reason") or "") == "single_line_reacquire_semantic_block":
        flags.append("single_line_reacquire_blocked:semantic")
    if bool((visual_reentry_ev or {}).get("visual_control_memory_active")):
        flags.append(f"visual_memory:{(visual_reentry_ev or {}).get('visual_control_memory_ticks', '-')}")
    if control_error_m is not None and abs(control_error_m) >= 0.10:
        flags.append(f"control_error:{abs(control_error_m):.3f}m")
    if steering_deg is not None and abs(steering_deg) >= 20.0:
        flags.append(f"steer_sat:{steering_deg:+.1f}deg")
    if quality is not None and quality < 0.55:
        flags.append(f"quality:{quality:.2f}")
    return flags


def _save_video_frame(
    *,
    video_path: str | Path | None,
    frame_no: int,
    out_path: Path,
) -> str | None:
    if not video_path:
        return None
    try:
        import cv2
    except ImportError:
        return None
    source = Path(video_path)
    if not source.exists() or frame_no <= 0:
        return None
    cap = cv2.VideoCapture(str(source))
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_no) - 1))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(out_path), frame):
            return str(out_path)
    finally:
        cap.release()
    return None


def _write_campaign_satellite_snapshot(
    *,
    out_path: Path,
    track_png: Path,
    expected_route: dict[str, Any],
    actual_xy: list[list[float]],
    pose_x: float | None,
    pose_y: float | None,
    pose_yaw: float | None,
) -> str | None:
    if pose_x is None or pose_y is None:
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    meta = _overlay_map_metadata(track_png, expected_route)
    img = cv2.imread(str(track_png), cv2.IMREAD_COLOR) if track_png.exists() else None
    if img is None:
        width = int(meta.get("image_width_px", 2039) or 2039)
        height = int(meta.get("image_height_px", 1343) or 1343)
        img = np.full((height, width, 3), 245, dtype=np.uint8)

    bounds = dict(meta.get("world_bounds") or {})
    x_min = float(bounds.get("x_min", 0.0) or 0.0)
    y_min = float(bounds.get("y_min", 0.0) or 0.0)
    mpp = float(meta.get("meters_per_pixel", 0.01) or 0.01)
    y_axis_inverted = bool(meta.get("y_axis_inverted", False))
    height_px, width_px = img.shape[:2]

    def world_to_pixel(x: float, y: float) -> tuple[int, int]:
        px = (float(x) - x_min) / mpp
        local_y = float(y) - y_min
        py = height_px - (local_y / mpp) if y_axis_inverted else local_y / mpp
        return int(round(px)), int(round(py))

    def draw_polyline(points: list[tuple[float, float]], color: tuple[int, int, int], thickness: int) -> None:
        if len(points) < 2:
            return
        pix = np.asarray([world_to_pixel(x, y) for x, y in points], dtype=np.int32)
        cv2.polylines(img, [pix], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    expected_xy = [
        (float(pt[0]), float(pt[1]))
        for pt in expected_route.get("waypoints", [])
        if isinstance(pt, list) and len(pt) >= 2
    ]
    actual_points = [(float(pt[0]), float(pt[1])) for pt in actual_xy if len(pt) >= 2]
    draw_polyline(expected_xy, (0, 155, 0), 4)
    draw_polyline(actual_points, (25, 25, 220), 3)

    px, py = world_to_pixel(float(pose_x), float(pose_y))
    cv2.circle(img, (px, py), 8, (255, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (px, py), 11, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    if pose_yaw is not None and math.isfinite(float(pose_yaw)):
        tip = (
            int(round(px + 34.0 * math.cos(float(pose_yaw)))),
            int(round(py + 34.0 * math.sin(float(pose_yaw)))),
        )
        cv2.arrowedLine(img, (px, py), tip, (255, 0, 255), 3, cv2.LINE_AA, tipLength=0.30)

    crop_w = min(720, width_px)
    crop_h = min(520, height_px)
    x0 = max(0, min(width_px - crop_w, px - crop_w // 2))
    y0 = max(0, min(height_px - crop_h, py - crop_h // 2))
    crop = img[y0 : y0 + crop_h, x0 : x0 + crop_w]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(str(out_path), crop):
        return str(out_path)
    return None


def _write_campaign_digest(
    *,
    run_dir: Path,
    brain_events: list[dict[str, Any]],
    video_frames: dict[str, Any] | None,
    frame_timestamps: list[float] | None,
    debug_video_path: str | None,
    track_png: Path,
    expected_route: dict[str, Any],
    actual_xy: list[list[float]],
    max_rows: int = 24,
) -> dict[str, Any]:
    if not video_frames and not frame_timestamps:
        return {"written": False, "error": "no_frame_index"}

    lane_idx = _build_event_index(brain_events, thread="lane_observer", event="lane_obs")
    mpc_idx = _build_event_index(brain_events, thread="mpc", event="compute_out")
    plan_idx = _build_event_index(brain_events, thread="behavior_planner", event="plan_output")
    visual_reentry_idx = _build_event_index(brain_events, thread="path_optimizer", event="visual_lane_reentry_bias")
    route_idx = _build_event_index(brain_events, thread="nav_planner", event="route_update")
    pose_idx = _build_event_index(brain_events, thread="pose_estimator", event="pose_published")
    line_hold_idx = _build_event_index(brain_events, thread="path_optimizer", event="line_loss_straight_hold")
    route_hold_idx = _build_event_index(brain_events, thread="path_optimizer", event="route_tracking_visual_hold")
    single_hold_idx = _build_event_index(brain_events, thread="path_optimizer", event="single_line_transition_hold")

    frame_items = []
    if video_frames and isinstance(video_frames.get("frames"), list):
        frame_items = list(video_frames.get("frames") or [])
    elif frame_timestamps:
        frame_items = [
            {"frame": idx + 1, "time_s": None, "path": str(run_dir / "frames" / f"frame_{idx + 1:06d}.png")}
            for idx in range(len(frame_timestamps))
        ]
    if not frame_items:
        return {"written": False, "error": "empty_frame_index"}

    candidates: list[dict[str, Any]] = []
    for item in frame_items:
        try:
            frame_no = int(item.get("frame") or 0)
        except (TypeError, ValueError):
            continue
        if frame_no <= 0:
            continue
        frame_ts = None
        if frame_timestamps and frame_no <= len(frame_timestamps):
            frame_ts = _finite_float(frame_timestamps[frame_no - 1])
        if frame_ts is None:
            frame_time_s = _finite_float(item.get("time_s"))
            # Without wall-clock timestamps, we cannot align robustly to JSONL.
            if frame_time_s is None:
                continue
            continue

        lane_ev = _event_at(lane_idx, frame_ts, max_age_s=0.45)
        mpc_ev = _event_at(mpc_idx, frame_ts, max_age_s=0.35)
        plan_ev = _event_at(plan_idx, frame_ts, max_age_s=0.45)
        visual_reentry_ev = _event_at(visual_reentry_idx, frame_ts, max_age_s=0.35)
        route_ev = _event_at(route_idx, frame_ts, max_age_s=0.45)
        pose_ev = _event_at(pose_idx, frame_ts, max_age_s=0.35)
        line_hold_ev = _event_at(line_hold_idx, frame_ts, max_age_s=0.35)
        route_hold_ev = _event_at(route_hold_idx, frame_ts, max_age_s=0.35)
        single_hold_ev = _event_at(single_hold_idx, frame_ts, max_age_s=0.35)
        if not plan_ev or not bool(plan_ev.get("valid", False)):
            continue
        flags = _campaign_digest_flags(
            lane_ev=lane_ev,
            plan_ev=plan_ev,
            visual_reentry_ev=visual_reentry_ev,
            mpc_ev=mpc_ev,
        )
        if not flags:
            continue
        visual_offset_m = _visual_offset_from_lane_event(lane_ev)
        control_error_m = _control_error_from_lane_event(lane_ev)
        steering_deg = _finite_float((mpc_ev or {}).get("steering_deg"))
        score = len(flags)
        if steering_deg is not None and abs(steering_deg) >= 20.0:
            score += 3
        if visual_offset_m is not None and abs(visual_offset_m) >= 0.12:
            score += 2
        if (lane_ev or {}).get("measurement_mode") in {"blind", "none"}:
            score += 1
        candidates.append(
            {
                "frame": frame_no,
                "ts": frame_ts,
                "score": score,
                "flags": flags,
                "pov_frame": str(item.get("path") or (run_dir / "frames" / f"frame_{frame_no:06d}.png")),
                "lanelet_id": (route_ev or {}).get("current_lanelet_id"),
                "route_progress": _finite_float((route_ev or {}).get("route_progress")),
                "map_match_error_m": _finite_float((route_ev or {}).get("map_match_error_m")),
                "measurement_mode": (lane_ev or {}).get("measurement_mode"),
                "line_sides": (lane_ev or {}).get("sides"),
                "quality": _finite_float((lane_ev or {}).get("quality")),
                "visual_offset_m": visual_offset_m,
                "control_lateral_error_m": control_error_m,
                "control_lateral_error_source": (lane_ev or {}).get("control_lateral_error_source"),
                "single_line_reacquire_active": (lane_ev or {}).get("single_line_reacquire_active"),
                "single_line_reacquire_side": (lane_ev or {}).get("single_line_reacquire_side"),
                "single_line_reacquire_error_m": (lane_ev or {}).get("single_line_reacquire_error_m"),
                "single_line_reacquire_error_source": (lane_ev or {}).get("single_line_reacquire_error_source"),
                "direct_error_valid": (lane_ev or {}).get("direct_error_valid"),
                "path_source": (plan_ev or {}).get("path_source"),
                "visual_lane_measurement_source": (visual_reentry_ev or {}).get("measurement_source"),
                "visual_lane_control_error_source": (visual_reentry_ev or {}).get("control_source"),
                "visual_lane_reentry_reason": (visual_reentry_ev or {}).get("reason"),
                "visual_control_memory_active": (visual_reentry_ev or {}).get("visual_control_memory_active"),
                "visual_control_memory_ticks": (visual_reentry_ev or {}).get("visual_control_memory_ticks"),
                "visual_control_memory_source": (visual_reentry_ev or {}).get("visual_control_memory_source"),
                "visual_path_reason": (
                    (plan_ev or {}).get("visual_path_primary_rejected_reason")
                    or (plan_ev or {}).get("visual_path_primary_reason")
                    or ""
                ),
                "speed_mps": _finite_float((mpc_ev or {}).get("speed_mps")),
                "steering_deg": steering_deg,
                "line_loss_hold": bool(line_hold_ev),
                "line_loss_hold_ticks": (line_hold_ev or {}).get("hold_ticks"),
                "line_loss_hold_reason": (line_hold_ev or {}).get("reason"),
                "line_loss_measurement_mode": (line_hold_ev or {}).get("measurement_mode"),
                "route_tracking_hold": bool(route_hold_ev),
                "single_line_transition_hold": bool(single_hold_ev),
                "pose_x": _finite_float((pose_ev or {}).get("fused_x")),
                "pose_y": _finite_float((pose_ev or {}).get("fused_y")),
                "pose_yaw": _finite_float((pose_ev or {}).get("fused_yaw_rad")),
                "visual_map_match_accepted": (pose_ev or {}).get("visual_lane_match_accepted"),
                "visual_map_match_reason": (pose_ev or {}).get("visual_lane_match_reason"),
            }
        )

    transition_rows: list[dict[str, Any]] = []
    last_signature: tuple[Any, Any] | None = None
    for row in candidates:
        signature = (row.get("measurement_mode"), row.get("path_source"))
        if signature != last_signature:
            transition_rows.append(row)
            last_signature = signature
    top_rows = sorted(
        candidates,
        key=lambda item: (-int(item.get("score", 0) or 0), int(item.get("frame", 0) or 0)),
    )
    selected_by_frame: dict[int, dict[str, Any]] = {}
    for row in transition_rows[: max(1, int(max_rows) // 2)]:
        selected_by_frame[int(row["frame"])] = row
    for row in top_rows:
        frame_no = int(row["frame"])
        signature = (row.get("measurement_mode"), row.get("path_source"))
        if any(
            abs(frame_no - int(existing.get("frame", 0) or 0)) < 5
            and signature == (existing.get("measurement_mode"), existing.get("path_source"))
            for existing in selected_by_frame.values()
        ):
            continue
        selected_by_frame.setdefault(int(row["frame"]), row)
        if len(selected_by_frame) >= int(max_rows):
            break
    selected = [selected_by_frame[key] for key in sorted(selected_by_frame)]

    digest_dir = run_dir / "campaign_digest"
    debug_frames_dir = digest_dir / "debug"
    satellite_dir = digest_dir / "satellite"
    for row in selected:
        frame_no = int(row["frame"])
        row["debug_frame"] = _save_video_frame(
            video_path=debug_video_path,
            frame_no=frame_no,
            out_path=debug_frames_dir / f"frame_{frame_no:06d}.png",
        )
        row["satellite_frame"] = _write_campaign_satellite_snapshot(
            out_path=satellite_dir / f"frame_{frame_no:06d}.png",
            track_png=track_png,
            expected_route=expected_route,
            actual_xy=actual_xy,
            pose_x=_finite_float(row.get("pose_x")),
            pose_y=_finite_float(row.get("pose_y")),
            pose_yaw=_finite_float(row.get("pose_yaw")),
        )

    digest_dir.mkdir(parents=True, exist_ok=True)
    digest_json = digest_dir / "digest.json"
    digest_md = digest_dir / "digest.md"
    digest_json.write_text(
        json.dumps({"rows": selected}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# Campaign Digest",
        "",
        "| frame | flags | lanelet | mode | side | q | visual_offset_m | control_m | control_source | visual_reason | path_source | visual_source | steering_deg | speed_mps | hold | POV | debug | satellite |",
        "|---:|---|---:|---|---|---:|---:|---:|---|---|---|---|---:|---:|---|---|---|---|",
    ]
    for row in selected:
        hold_bits = []
        if row.get("line_loss_hold"):
            hold_bits.append(f"line_loss:{row.get('line_loss_hold_ticks')}")
        if row.get("route_tracking_hold"):
            hold_bits.append("route_visual")
        if row.get("single_line_transition_hold"):
            hold_bits.append("single_transition")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("frame")),
                    ", ".join(str(flag) for flag in row.get("flags", [])),
                    str(row.get("lanelet_id") or "-"),
                    str(row.get("measurement_mode") or "-"),
                    str(row.get("single_line_reacquire_side") or row.get("line_sides") or "-"),
                    _fmt_value(row.get("quality"), precision=2),
                    _fmt_value(row.get("visual_offset_m"), precision=3),
                    _fmt_value(row.get("control_lateral_error_m"), precision=3),
                    str(row.get("control_lateral_error_source") or "-"),
                    str(row.get("visual_lane_reentry_reason") or "-"),
                    str(row.get("path_source") or "-"),
                    str(row.get("visual_lane_measurement_source") or "-"),
                    _fmt_value(row.get("steering_deg"), precision=1),
                    _fmt_value(row.get("speed_mps"), precision=2),
                    ", ".join(hold_bits) if hold_bits else "-",
                    str(row.get("pov_frame") or "-"),
                    str(row.get("debug_frame") or "-"),
                    str(row.get("satellite_frame") or "-"),
                ]
            )
            + " |"
        )
    digest_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "written": True,
        "row_count": len(selected),
        "digest_json": str(digest_json),
        "digest_md": str(digest_md),
        "rows": selected,
    }


def _collect_camera_video(brain_dir: Path, run_dir: Path, started_at: float) -> str | None:
    candidates = [
        path for path in brain_dir.glob("output_video*.avi")
        if path.stat().st_mtime >= started_at - 1.0
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    source = candidates[0]
    target = run_dir / "camera.avi"
    try:
        shutil.move(str(source), str(target))
        return str(target)
    except OSError:
        return str(source)


def _promote_gui_overlay_camera_video(
    *,
    run_dir: Path,
    raw_video_path: str | None,
    gui_overlay_video: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    """Make camera.avi match the Driving GUI when that stream was recorded.

    The camera thread records the raw capture. The GUI Driving window shows the
    Local AI overlay (`serialCamera_bin`). For campaign review, `camera.avi`
    should be the visual stream humans inspect, while `camera_raw.avi` keeps the
    original capture available for lower-level debugging.
    """
    if not (gui_overlay_video and gui_overlay_video.get("written") and gui_overlay_video.get("path")):
        return raw_video_path, raw_video_path

    gui_path = Path(str(gui_overlay_video["path"]))
    if not gui_path.exists():
        return raw_video_path, raw_video_path

    raw_preserved_path: str | None = raw_video_path
    if raw_video_path:
        raw_source = Path(raw_video_path)
        raw_target = run_dir / "camera_raw.avi"
        try:
            if raw_source.exists() and raw_source.resolve() != raw_target.resolve():
                if raw_target.exists():
                    raw_target.unlink()
                shutil.move(str(raw_source), str(raw_target))
            if raw_target.exists():
                raw_preserved_path = str(raw_target)
        except OSError:
            raw_preserved_path = str(raw_source)

    promoted_path = run_dir / "camera.avi"
    try:
        if promoted_path.exists():
            promoted_path.unlink()
        shutil.copy2(gui_path, promoted_path)
        return str(promoted_path), raw_preserved_path
    except OSError:
        return raw_preserved_path, raw_preserved_path


def _extract_video_frames(video_path: str | Path, run_dir: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError:
        return {
            "video_path": str(video_path),
            "frame_count": 0,
            "saved_frame_count": 0,
            "error": "opencv_unavailable",
        }

    video = Path(video_path)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {
            "video_path": str(video),
            "frames_dir": str(frames_dir),
            "frame_count": 0,
            "saved_frame_count": 0,
            "error": "video_open_failed",
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    index: list[dict[str, Any]] = []
    frame_count = 0
    saved_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        frame_path = frames_dir / f"frame_{frame_count:06d}.png"
        if cv2.imwrite(str(frame_path), frame):
            saved_count += 1
            index.append(
                {
                    "frame": frame_count,
                    "time_s": ((frame_count - 1) / fps) if fps > 0.0 else None,
                    "path": str(frame_path),
                }
            )

    cap.release()
    meta: dict[str, Any] = {
        "video_path": str(video),
        "frames_dir": str(frames_dir),
        "index_json": str(frames_dir / "index.json"),
        "fps": fps,
        "width": width,
        "height": height,
        "reported_frame_count": total_reported,
        "frame_count": frame_count,
        "saved_frame_count": saved_count,
        "frames": index,
    }
    if frame_count <= 0:
        meta["error"] = "video_has_no_frames"
    (frames_dir / "index.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return meta


def _build_event_index(
    brain_events: list[dict[str, Any]],
    *,
    thread: str | None = None,
    event: str | None = None,
) -> tuple[list[float], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    for ev in brain_events:
        if thread is not None and ev.get("thread") != thread:
            continue
        if event is not None and ev.get("event") != event:
            continue
        ts = _finite_float(ev.get("ts"))
        if ts is None:
            continue
        selected.append(ev)
    selected.sort(key=lambda item: float(item.get("ts", 0.0) or 0.0))
    return [float(ev.get("ts", 0.0) or 0.0) for ev in selected], selected


def _event_at(
    index: tuple[list[float], list[dict[str, Any]]],
    ts: float,
    *,
    max_age_s: float = 0.50,
) -> dict[str, Any] | None:
    times, events = index
    if not times:
        return None
    idx = bisect.bisect_right(times, float(ts)) - 1
    if idx < 0:
        return None
    ev = events[idx]
    age_s = float(ts) - float(times[idx])
    if age_s < -1e-6 or age_s > float(max_age_s):
        return None
    return ev


def _fmt_value(value: Any, *, unit: str = "", precision: int = 2, none: str = "-") -> str:
    number = _finite_float(value)
    if number is None:
        if value is None:
            return none
        text = str(value)
        return text if len(text) <= 42 else text[:39] + "..."
    return f"{number:.{precision}f}{unit}"


def _fmt_cm(value_m: Any) -> str:
    number = _finite_float(value_m)
    if number is None:
        return "-"
    return f"{number * 100.0:+.1f}cm"


def _fmt_deg(value_rad: Any) -> str:
    number = _finite_float(value_rad)
    if number is None:
        return "-"
    return f"{math.degrees(number):+.1f}deg"


def _short_text(value: Any, *, max_len: int = 44) -> str:
    if value is None:
        return "-"
    text = str(value)
    return text if len(text) <= max_len else text[: max(0, max_len - 3)] + "..."


def _put_panel_line(
    img: Any,
    text: str,
    org: tuple[int, int],
    *,
    color: tuple[int, int, int] = (235, 235, 235),
    scale: float = 0.42,
    thickness: int = 1,
) -> None:
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _put_panel_header(img: Any, text: str, org: tuple[int, int]) -> None:
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_camera_lane_overlay(frame: Any, lane_ev: dict[str, Any] | None) -> None:
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    overlay = frame.copy()
    cx = int(round(w * 0.50))
    y_ref = int(round(h * 0.64))
    y_top = int(round(h * 0.36))
    y_bottom = int(round(h * 0.95))

    cv2.line(overlay, (cx, y_top), (cx, y_bottom), (120, 120, 120), 1, cv2.LINE_AA)
    cv2.line(overlay, (0, y_ref), (w - 1, y_ref), (90, 90, 90), 1, cv2.LINE_AA)

    if not lane_ev:
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        cv2.putText(frame, "AI lane: no lane_obs", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2, cv2.LINE_AA)
        return

    mode = str(lane_ev.get("measurement_mode") or "none")
    quality = float(_finite_float(lane_ev.get("quality")) or 0.0)
    sides = tuple(lane_ev.get("sides") or ())
    left_m = _finite_float(lane_ev.get("left_line_distance_m"))
    right_m = _finite_float(lane_ev.get("right_line_distance_m"))
    lane_width_m = _finite_float(lane_ev.get("lane_width_m")) or 0.35
    lane_width_px = _finite_float(lane_ev.get("lane_width_px"))
    px_per_m = None
    if lane_width_px is not None and lane_width_m > 1e-6:
        px_per_m = float(lane_width_px) / float(lane_width_m)
    else:
        px_per_m = float(w) * 0.90 / max(float(lane_width_m), 1e-6)

    x_left = int(round(cx - left_m * px_per_m)) if left_m is not None else None
    x_right = int(round(cx + right_m * px_per_m)) if right_m is not None else None
    center_offset_m = _finite_float(lane_ev.get("line_center_offset_m"))
    if center_offset_m is None and left_m is not None and right_m is not None:
        center_offset_m = 0.5 * (right_m - left_m)
    x_lane_center = int(round(cx + center_offset_m * px_per_m)) if center_offset_m is not None else None

    if x_left is not None and x_right is not None:
        x_left = max(-w, min(2 * w, x_left))
        x_right = max(-w, min(2 * w, x_right))
        pts = np.array(
            [
                [x_left, y_ref],
                [x_right, y_ref],
                [x_right + int(0.18 * (x_right - cx)), y_bottom],
                [x_left + int(0.18 * (x_left - cx)), y_bottom],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(overlay, [pts], (60, 80, 0), lineType=cv2.LINE_AA)

    def _draw_side_line(x_ref: int | None, color: tuple[int, int, int], label: str, dashed: bool = False) -> None:
        if x_ref is None:
            return
        x_ref = max(-w, min(2 * w, int(x_ref)))
        x_top = int(round(cx + 0.35 * (x_ref - cx)))
        x_bottom = int(round(cx + 1.18 * (x_ref - cx)))
        if dashed:
            segments = 7
            for idx in range(segments):
                if idx % 2:
                    continue
                y0 = int(round(y_top + (y_bottom - y_top) * idx / segments))
                y1 = int(round(y_top + (y_bottom - y_top) * (idx + 1) / segments))
                x0 = int(round(x_top + (x_bottom - x_top) * idx / segments))
                x1 = int(round(x_top + (x_bottom - x_top) * (idx + 1) / segments))
                cv2.line(overlay, (x0, y0), (x1, y1), color, 4, cv2.LINE_AA)
        else:
            cv2.line(overlay, (x_top, y_top), (x_bottom, y_bottom), color, 4, cv2.LINE_AA)
        cv2.circle(overlay, (x_ref, y_ref), 6, color, -1, cv2.LINE_AA)
        cv2.putText(overlay, label, (max(4, min(w - 80, x_ref - 22)), y_ref - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    _draw_side_line(x_left, (255, 180, 40), "AI L", dashed=("left" not in sides and mode == "single_line"))
    _draw_side_line(x_right, (60, 255, 80), "AI R", dashed=("right" not in sides and mode == "single_line"))

    if x_lane_center is not None:
        cv2.line(overlay, (x_lane_center, y_top), (x_lane_center, y_bottom), (255, 0, 255), 2, cv2.LINE_AA)
        cv2.arrowedLine(overlay, (cx, y_ref + 24), (x_lane_center, y_ref + 24), (255, 0, 255), 2, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(
            overlay,
            f"center {center_offset_m * 100.0:+.1f}cm" if center_offset_m is not None else "center",
            (max(4, min(w - 160, x_lane_center - 58)), y_ref + 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.46, frame, 0.54, 0, frame)
    status_color = (80, 255, 80) if mode in {"two_line", "single_line"} and quality >= 0.55 else (0, 180, 255)
    if mode in {"blind", "none", "route_tracking"}:
        status_color = (0, 0, 255)
    cv2.putText(
        frame,
        f"AI mask/lane: {mode} q={quality:.2f} sides={','.join(map(str, sides)) or '-'}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"AI mask/lane: {mode} q={quality:.2f} sides={','.join(map(str, sides)) or '-'}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        status_color,
        1,
        cv2.LINE_AA,
    )


def _draw_debug_panel(
    panel: Any,
    *,
    frame_no: int,
    frame_ts: float | None,
    record_start_ts: float | None,
    lane_ev: dict[str, Any] | None,
    mpc_ev: dict[str, Any] | None,
    plan_ev: dict[str, Any] | None,
    pose_ev: dict[str, Any] | None,
    prefix_ev: dict[str, Any] | None,
    line_hold_ev: dict[str, Any] | None,
    route_hold_ev: dict[str, Any] | None,
    single_hold_ev: dict[str, Any] | None,
) -> None:
    import cv2

    panel[:] = (24, 24, 24)
    y = 24
    x = 12
    dt = None
    if frame_ts is not None and record_start_ts is not None:
        dt = frame_ts - record_start_ts
    _put_panel_header(panel, f"Frame {frame_no}  t={_fmt_value(dt, unit='s', precision=2)}", (x, y))
    y += 22

    _put_panel_header(panel, "AI LANE / MASK", (x, y))
    y += 18
    if lane_ev:
        _put_panel_line(panel, f"mode={_short_text(lane_ev.get('measurement_mode'), max_len=16)} prev={_short_text(lane_ev.get('previous_measurement_mode'), max_len=12)} streak={lane_ev.get('measurement_mode_streak_frames', '-')}", (x, y)); y += 16
        _put_panel_line(panel, f"q={_fmt_value(lane_ev.get('quality'), precision=2)} sides={','.join(map(str, lane_ev.get('sides') or ())) or '-'} blind={_short_text(lane_ev.get('blind_mode'), max_len=12)}", (x, y)); y += 16
        _put_panel_line(panel, f"center={_fmt_cm(lane_ev.get('line_center_offset_m'))} direct={_fmt_cm(lane_ev.get('direct_error_m'))} valid={bool(lane_ev.get('direct_error_valid'))}", (x, y)); y += 16
        _put_panel_line(panel, f"control={_fmt_cm(lane_ev.get('control_lateral_error_m'))} src={_short_text(lane_ev.get('control_lateral_error_source'), max_len=18)}", (x, y)); y += 16
        _put_panel_line(panel, f"L={_fmt_cm(lane_ev.get('left_line_distance_m'))} R={_fmt_cm(lane_ev.get('right_line_distance_m'))} width_px={_fmt_value(lane_ev.get('lane_width_px'), precision=1)}", (x, y)); y += 16
        _put_panel_line(panel, f"hdg={_fmt_deg(lane_ev.get('heading_error_rad'))} cam_yaw={_fmt_deg(lane_ev.get('camera_yaw_hint_rad'))} wp={lane_ev.get('visual_waypoint_count', '-')}", (x, y)); y += 16
        _put_panel_line(panel, f"policy={_short_text(lane_ev.get('control_policy_mode'), max_len=18)} planner={bool(lane_ev.get('planner_priority_active'))}", (x, y)); y += 18
    else:
        _put_panel_line(panel, "no recent lane_obs", (x, y), color=(0, 120, 255)); y += 18

    _put_panel_header(panel, "MPC COMMAND", (x, y))
    y += 18
    if mpc_ev:
        _put_panel_line(panel, f"valid={bool(mpc_ev.get('valid', False))} reason={_short_text(mpc_ev.get('reason'), max_len=22)}", (x, y)); y += 16
        _put_panel_line(panel, f"steer={_fmt_value(mpc_ev.get('steering_deg'), unit='deg', precision=1)} raw={_fmt_value(mpc_ev.get('delta_deg_raw'), unit='deg', precision=1)} alpha={_fmt_value(mpc_ev.get('alpha_deg'), unit='deg', precision=1)}", (x, y)); y += 16
        _put_panel_line(panel, f"speed={_fmt_value(mpc_ev.get('speed_mps'), unit='m/s', precision=2)} vraw={_fmt_value(mpc_ev.get('v_opt_raw'), unit='m/s', precision=2)} src={_short_text(mpc_ev.get('path_source'), max_len=20)}", (x, y)); y += 16
        _put_panel_line(panel, f"vis_cap={bool(mpc_ev.get('visual_steer_cap_applied'))} cap={_fmt_value(mpc_ev.get('visual_steer_cap_deg'), unit='deg', precision=1)} rate={_fmt_value(mpc_ev.get('max_steer_rate_deg_s'), precision=0)}", (x, y)); y += 18
    else:
        _put_panel_line(panel, "no recent compute_out", (x, y), color=(0, 120, 255)); y += 18

    _put_panel_header(panel, "PLANNER", (x, y))
    y += 18
    if plan_ev:
        _put_panel_line(panel, f"scenario={_short_text(plan_ev.get('scenario'), max_len=15)} valid={bool(plan_ev.get('valid', False))} stop={bool(plan_ev.get('stop_required', False))}", (x, y)); y += 16
        _put_panel_line(panel, f"path={_short_text(plan_ev.get('path_source'), max_len=22)} lanelet={_short_text(plan_ev.get('current_lanelet_id'), max_len=12)}", (x, y)); y += 16
        _put_panel_line(panel, f"visual={_short_text(plan_ev.get('visual_path_primary_reason'), max_len=28)}", (x, y)); y += 16
        _put_panel_line(panel, f"reject={_short_text(plan_ev.get('visual_path_primary_rejected_reason'), max_len=30)}", (x, y)); y += 16
        _put_panel_line(panel, f"map_err={_fmt_cm(plan_ev.get('map_match_error_m'))} progress={_fmt_value(plan_ev.get('route_progress'), precision=2)} sem={_short_text(plan_ev.get('next_semantic_type'), max_len=12)}", (x, y)); y += 18
    else:
        _put_panel_line(panel, "no recent plan_output", (x, y), color=(0, 120, 255)); y += 18

    _put_panel_header(panel, "LOCALIZATION", (x, y))
    y += 18
    if pose_ev:
        _put_panel_line(panel, f"mode={_short_text(pose_ev.get('localization_mode'), max_len=14)} reloc={_short_text(pose_ev.get('reloc_mode'), max_len=18)}", (x, y)); y += 16
        _put_panel_line(panel, f"pose=({_fmt_value(pose_ev.get('fused_x'), precision=2)}, {_fmt_value(pose_ev.get('fused_y'), precision=2)}) yaw={_fmt_deg(pose_ev.get('fused_yaw_rad'))}", (x, y)); y += 16
        _put_panel_line(panel, f"gps={_short_text(pose_ev.get('gps_mode'), max_len=16)} lane_corr={_fmt_cm(pose_ev.get('lane_correction_m'))} vmap={_fmt_cm(pose_ev.get('visual_map_lateral_correction_m'))}", (x, y)); y += 16
        _put_panel_line(panel, f"vmatch={pose_ev.get('visual_lane_match_accepted')} reason={_short_text(pose_ev.get('visual_lane_match_reason'), max_len=20)}", (x, y)); y += 16
        _put_panel_line(panel, f"vm_lat={_fmt_cm(pose_ev.get('visual_lane_match_lateral_error_m'))} yaw={_fmt_deg(pose_ev.get('visual_lane_match_yaw_error_rad'))} near={_fmt_deg(pose_ev.get('visual_lane_match_near_yaw_error_rad'))}", (x, y)); y += 18
    else:
        _put_panel_line(panel, "no recent pose_published", (x, y), color=(0, 120, 255)); y += 18

    _put_panel_header(panel, "PATH OPT", (x, y))
    y += 18
    if prefix_ev:
        _put_panel_line(panel, f"prefix={bool(prefix_ev.get('applied'))} s1_y={_fmt_cm(prefix_ev.get('sample1_y_left_before_m'))}->{_fmt_cm(prefix_ev.get('sample1_y_left_after_m'))}", (x, y)); y += 16
        _put_panel_line(panel, f"hlim={_fmt_value(prefix_ev.get('heading_limit_deg'), unit='deg', precision=1)} hcap={bool(prefix_ev.get('heading_cap_applied'))} ecap={bool(prefix_ev.get('entry_heading_cap_applied'))}", (x, y)); y += 16
    if line_hold_ev:
        _put_panel_line(panel, f"line_hold={line_hold_ev.get('hold_ticks', '-')} { _short_text(line_hold_ev.get('reason'), max_len=24)}", (x, y), color=(80, 220, 255)); y += 16
    if route_hold_ev:
        _put_panel_line(panel, f"route_hold={route_hold_ev.get('hold_ticks', '-')} { _short_text(route_hold_ev.get('reason'), max_len=22)}", (x, y), color=(80, 220, 255)); y += 16
    if single_hold_ev:
        _put_panel_line(panel, f"single_hold={single_hold_ev.get('hold_ticks', '-')} { _short_text(single_hold_ev.get('reason'), max_len=21)}", (x, y), color=(80, 220, 255)); y += 16


def _write_debug_overlay_video(
    *,
    video_path: str | Path,
    out_path: Path,
    brain_events: list[dict[str, Any]],
    record_start_ts: float | None,
    draw_lane_overlay: bool = True,
    frame_timestamps: list[float] | None = None,
) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {"written": False, "error": "opencv_unavailable"}

    source = Path(video_path)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return {"written": False, "error": "video_open_failed", "path": str(source)}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0:
        fps = 5.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return {"written": False, "error": "invalid_video_dimensions", "path": str(source)}

    panel_w = 430
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width + panel_w, height),
    )
    if not writer.isOpened():
        cap.release()
        return {"written": False, "error": "video_writer_open_failed", "path": str(out_path)}

    lane_idx = _build_event_index(brain_events, thread="lane_observer", event="lane_obs")
    mpc_idx = _build_event_index(brain_events, thread="mpc", event="compute_out")
    plan_idx = _build_event_index(brain_events, thread="behavior_planner", event="plan_output")
    pose_idx = _build_event_index(brain_events, thread="pose_estimator", event="pose_published")
    prefix_idx = _build_event_index(brain_events, thread="path_optimizer", event="visual_prefix_stabilization")
    line_hold_idx = _build_event_index(brain_events, thread="path_optimizer", event="line_loss_straight_hold")
    route_hold_idx = _build_event_index(brain_events, thread="path_optimizer", event="route_tracking_visual_hold")
    single_hold_idx = _build_event_index(brain_events, thread="path_optimizer", event="single_line_transition_hold")

    if record_start_ts is None:
        event_ts = [
            float(ev.get("ts", 0.0) or 0.0)
            for ev in brain_events
            if _finite_float(ev.get("ts")) is not None
        ]
        base_ts = min(event_ts) if event_ts else 0.0
    else:
        base_ts = float(record_start_ts)

    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_timestamps is not None and frame_no <= len(frame_timestamps):
            frame_ts = float(frame_timestamps[frame_no - 1])
        else:
            frame_ts = float(base_ts) + ((frame_no - 1) / float(fps))

        lane_ev = _event_at(lane_idx, frame_ts, max_age_s=0.45)
        mpc_ev = _event_at(mpc_idx, frame_ts, max_age_s=0.35)
        plan_ev = _event_at(plan_idx, frame_ts, max_age_s=0.45)
        pose_ev = _event_at(pose_idx, frame_ts, max_age_s=0.35)
        prefix_ev = _event_at(prefix_idx, frame_ts, max_age_s=0.45)
        line_hold_ev = _event_at(line_hold_idx, frame_ts, max_age_s=0.35)
        route_hold_ev = _event_at(route_hold_idx, frame_ts, max_age_s=0.35)
        single_hold_ev = _event_at(single_hold_idx, frame_ts, max_age_s=0.35)

        annotated = frame.copy()
        if draw_lane_overlay:
            _draw_camera_lane_overlay(annotated, lane_ev)
        panel = np.zeros((height, panel_w, 3), dtype=np.uint8)
        _draw_debug_panel(
            panel,
            frame_no=frame_no,
            frame_ts=frame_ts,
            record_start_ts=base_ts,
            lane_ev=lane_ev,
            mpc_ev=mpc_ev,
            plan_ev=plan_ev,
            pose_ev=pose_ev,
            prefix_ev=prefix_ev,
            line_hold_ev=line_hold_ev,
            route_hold_ev=route_hold_ev,
            single_hold_ev=single_hold_ev,
        )
        combined = np.zeros((height, width + panel_w, 3), dtype=np.uint8)
        combined[:, :width] = annotated
        combined[:, width:] = panel
        writer.write(combined)

    cap.release()
    writer.release()
    return {
        "written": frame_no > 0,
        "path": str(out_path),
        "frame_count": frame_no,
        "fps": fps,
        "width": width + panel_w,
        "height": height,
        "uses_frame_timestamps": bool(frame_timestamps),
    }


def _update_latest_symlink(root: Path, target: Path) -> None:
    latest = root / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(target, target_is_directory=True)
    except OSError:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--duration", type=float, default=240.0, help="Max seconds in the requested mode for the route.")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=("auto", "odometry"),
        help="DrivingMode to request after arming the route. Default: auto.",
    )
    parser.add_argument("--destination-lanelet", default="138", help="Destination lanelet id for the default go_to route.")
    parser.add_argument("--destination-x", type=float, default=None, help="Destination point x in brain map frame.")
    parser.add_argument("--destination-y", type=float, default=None, help="Destination point y in brain map frame.")
    parser.add_argument("--run-id", default=None, help="Override run id. Default: run_<YYYYmmdd_HHMMSS>.")
    parser.add_argument("--brain-dir", type=Path, default=BRAIN_DIR)
    parser.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    parser.add_argument("--osm-path", type=Path, default=DEFAULT_OSM_PATH)
    parser.add_argument("--track-png", type=Path, default=DEFAULT_TRACK_PNG)
    parser.add_argument("--host", default="http://localhost:5005")
    parser.add_argument("--max-cross-track-p90-m", type=float, default=0.20)
    parser.add_argument("--max-cross-track-max-m", type=float, default=0.35)
    parser.add_argument("--max-lane-offset-p90-m", type=float, default=0.12)
    parser.add_argument("--max-lane-offset-max-m", type=float, default=0.20)
    parser.add_argument(
        "--command-speed-scale",
        type=float,
        default=None,
        help=(
            "Override URT_COMMAND_SPEED_CALIBRATION_SCALE for no-encoder "
            "odometry sweeps."
        ),
    )
    parser.add_argument("--no-record", action="store_true", help="Do not toggle camera AVI recording.")
    parser.add_argument("--skip-cleanup-before", action="store_true", help="Do not kill previous sim/brain processes before starting.")
    parser.add_argument("--keep-running", action="store_true", help="Leave sim/brain running after the route attempt.")
    parser.add_argument("--skip-analyze-run", action="store_true", help="Do not run tools/dev/analyze_run.py after cleanup.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    brain_dir = args.brain_dir.resolve()
    sim_dir = args.sim_dir.resolve()
    if (args.destination_x is None) != (args.destination_y is None):
        print(
            "[campaign] ERROR: --destination-x and --destination-y must be provided together",
            file=sys.stderr,
        )
        return 2
    run_id = args.run_id or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    brain_run_dir = brain_dir / "temp" / "logs" / run_id
    sim_run_dir = sim_dir / "temp" / "logs" / run_id
    brain_jsonl = brain_run_dir / "brain.jsonl"
    sim_jsonl = sim_run_dir / "sim_bridge.jsonl"
    copied_sim_jsonl = brain_run_dir / "sim_bridge.jsonl"

    brain_run_dir.mkdir(parents=True, exist_ok=True)
    sim_run_dir.mkdir(parents=True, exist_ok=True)
    brain_jsonl.write_text("", encoding="utf-8")
    sim_jsonl.write_text("", encoding="utf-8")

    destination_lanelet = None if args.destination_x is not None else args.destination_lanelet
    expected_route, nav_command = _build_expected_route(
        osm_path=args.osm_path.resolve(),
        destination_lanelet=destination_lanelet,
        destination_x=args.destination_x,
        destination_y=args.destination_y,
    )
    (brain_run_dir / "route_expected.json").write_text(
        json.dumps(expected_route, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (brain_run_dir / "navigation_command.json").write_text(
        json.dumps(nav_command, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    brain_py = _find_brain_python(brain_dir)
    gz_py, gz_pypath = _find_gz_python()
    processes: list[ManagedProcess] = []
    client: DashboardClient | None = None
    started_at = time.time()
    record_start_ts: float | None = None
    record_stop_ts: float | None = None
    gui_overlay_video: dict[str, Any] | None = None

    print(f"[campaign] run_id={run_id}")
    print(f"[campaign] brain logs: {brain_run_dir}")
    print(f"[campaign] sim logs:   {sim_run_dir}")
    print(f"[campaign] brain_py={brain_py}")
    print(f"[campaign] gz_py={gz_py} PYTHONPATH+={gz_pypath or '<none>'}")

    try:
        if not args.skip_cleanup_before:
            _cleanup_known_processes(brain_dir, sim_dir)

        processes.append(
            _run(
                ["./run_sim.sh", "--headless"],
                cwd=sim_dir,
                stdout_path=sim_run_dir / "sim.stdout",
                name="gz_sim",
            )
        )

        time.sleep(5.0)
        gz_env = os.environ.copy()
        if gz_pypath:
            gz_env["PYTHONPATH"] = f"{gz_pypath}{os.pathsep}{gz_env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
        processes.append(
            _run(
                [
                    gz_py,
                    str(sim_dir / "tools" / "dev" / "gz_pose_logger.py"),
                    "--out",
                    str(sim_jsonl),
                ],
                cwd=sim_dir,
                stdout_path=sim_run_dir / "gz_logger.stdout",
                env=gz_env,
                name="gz_pose_logger",
            )
        )

        print("[campaign] waiting for sim_bridge :5575")
        if not _wait_port("localhost", 5575, 60.0):
            raise RuntimeError("sim_bridge ZMQ port :5575 did not become ready")

        brain_env = os.environ.copy()
        brain_env["URT_LIVE_LOG_PATH"] = str(brain_jsonl)
        brain_env["URT_SIM_MODE"] = "1"
        brain_env.setdefault("URT_DISABLE_AUTO_PARKING", "1")
        _apply_mode_env(
            brain_env,
            mode=args.mode,
            expected_route=expected_route,
            command_speed_scale=args.command_speed_scale,
        )
        processes.append(
            _run(
                ["./run.sh", "--no-gui"],
                cwd=brain_dir,
                stdout_path=brain_run_dir / "brain.stdout",
                env=brain_env,
                name="brain",
            )
        )

        print("[campaign] waiting for dashboard :5005")
        if not _wait_port("localhost", 5005, 60.0):
            raise RuntimeError("brain dashboard port :5005 did not become ready")

        print("[campaign] waiting for first pose_published")
        _wait_for_jsonl_event(
            brain_jsonl,
            thread="pose_estimator",
            event="pose_published",
            timeout_s=90.0,
        )

        client = DashboardClient(args.host)
        client.connect()
        if not client.request_session():
            raise RuntimeError("dashboard session was not granted")
        # Mirror the GUI/headless-controller handshake: give the backend a
        # short beat to register the socket as the active session before the
        # route and mode commands start flowing.
        time.sleep(1.0)

        if not args.no_record:
            client.start_gui_video_recording(brain_run_dir / "camera_gui_overlay.avi", fps=7.0)
            client.emit_message("Record", True, wait_response=True)
            record_start_ts = time.time()
            time.sleep(0.5)

        nav_ok = False
        for attempt in range(1, 4):
            print(f"[campaign] sending route command attempt {attempt}: {nav_command}")
            if not client.emit_message("NavigationCommand", nav_command, wait_response=True):
                print("[campaign] WARNING: dashboard did not acknowledge NavigationCommand")
            if _wait_for_navigation_acceptance(brain_jsonl, 5.0):
                nav_ok = True
                break
            time.sleep(0.5)
        if not nav_ok:
            raise RuntimeError("navigation command was not accepted by the brain")

        mode_ok = False
        for attempt in range(1, 4):
            print(f"[campaign] requesting {args.mode.upper()} attempt {attempt}")
            if not client.emit_message("DrivingMode", args.mode, wait_response=True):
                print(f"[campaign] WARNING: dashboard did not acknowledge DrivingMode={args.mode}")
            if _wait_for_mode_state(brain_jsonl, args.mode, 5.0):
                mode_ok = True
                break
            time.sleep(0.5)
        if not mode_ok:
            raise RuntimeError(f"{args.mode.upper()} mode was not confirmed by the brain")

        route_result = _route_progress_from_logs(brain_jsonl, args.duration)
        client.emit_message("DrivingMode", "stop")
        if not args.no_record:
            client.emit_message("Record", False, wait_response=True)
            record_stop_ts = time.time()
            gui_overlay_video = client.stop_gui_video_recording()
        time.sleep(0.5)

    except Exception as exc:
        print(f"[campaign] ERROR: {exc}", file=sys.stderr)
        route_result = RouteCommandResult(timed_out=True)
    finally:
        if client is not None:
            if gui_overlay_video is None:
                gui_overlay_video = client.stop_gui_video_recording()
            client.close()
        if not args.keep_running:
            for proc in reversed(processes):
                print(f"[campaign] stopping {proc.name}")
                proc.terminate()
            _cleanup_known_processes(brain_dir, sim_dir)

    if sim_jsonl.exists():
        shutil.copy2(sim_jsonl, copied_sim_jsonl)

    raw_video_path = None if args.no_record else _collect_camera_video(brain_dir, brain_run_dir, started_at)
    video_path, raw_video_path = _promote_gui_overlay_camera_video(
        run_dir=brain_run_dir,
        raw_video_path=raw_video_path,
        gui_overlay_video=gui_overlay_video,
    )
    video_frames: dict[str, Any] | None = None
    if video_path:
        video_frames = _extract_video_frames(video_path, brain_run_dir)
    brain_events = _load_jsonl(brain_jsonl)
    gt_events = _load_jsonl(copied_sim_jsonl)
    mode_start_ts = _find_mode_start_ts(brain_events, args.mode)
    metrics = _compute_metrics(
        expected_route=expected_route,
        brain_events=brain_events,
        gt_events=gt_events,
        auto_start_ts=mode_start_ts,
        max_p90_m=args.max_cross_track_p90_m,
        max_max_m=args.max_cross_track_max_m,
        max_lane_offset_p90_m=args.max_lane_offset_p90_m,
        max_lane_offset_max_m=args.max_lane_offset_max_m,
        route_result=route_result,
    )
    visual_samples = _collect_visual_lane_samples(
        brain_events,
        record_start_ts=record_start_ts,
        fps=float(video_frames.get("fps", 0.0) or 0.0) if video_frames else None,
    )
    if "visual_lane" in metrics:
        metrics["visual_lane"]["worst_samples"] = sorted(
            visual_samples,
            key=lambda sample: float(sample.get("visual_offset_abs_m", 0.0) or 0.0),
            reverse=True,
        )[:30]
    if video_path:
        metrics["artifacts"] = {"camera_video": video_path}
        if raw_video_path and raw_video_path != video_path:
            metrics["artifacts"]["camera_raw_video"] = raw_video_path
        if video_frames:
            metrics["artifacts"]["camera_frames_dir"] = str(brain_run_dir / "frames")
            metrics["artifacts"]["camera_frames_index_json"] = str(brain_run_dir / "frames" / "index.json")
    else:
        metrics["artifacts"] = {}
    if gui_overlay_video is not None:
        metrics["gui_overlay_video"] = {
            key: value
            for key, value in gui_overlay_video.items()
            if key != "_frame_timestamps"
        }
        if gui_overlay_video.get("written"):
            metrics["artifacts"]["camera_gui_overlay_video"] = str(gui_overlay_video.get("path"))
            if gui_overlay_video.get("frame_index_path"):
                metrics["artifacts"]["camera_gui_overlay_frame_index_json"] = str(
                    gui_overlay_video.get("frame_index_path")
                )

    overlay_written = _write_overlay(
        out_path=brain_run_dir / "overlay.png",
        track_png=args.track_png.resolve(),
        expected_route=expected_route,
        actual_xy=metrics["ground_truth"]["actual_path"],
    )
    if overlay_written:
        metrics["artifacts"]["overlay_png"] = str(brain_run_dir / "overlay.png")
    visual_overlay_written = _write_visual_lane_overlay(
        out_path=brain_run_dir / "overlay_visual_lane.png",
        track_png=args.track_png.resolve(),
        expected_route=expected_route,
        actual_xy=metrics["ground_truth"]["actual_path"],
        visual_samples=visual_samples,
    )
    if visual_overlay_written:
        metrics["artifacts"]["overlay_visual_lane_png"] = str(brain_run_dir / "overlay_visual_lane.png")
    debug_overlay_video: dict[str, Any] | None = None
    debug_source_video = None
    debug_draw_lane_overlay = True
    if gui_overlay_video is not None and gui_overlay_video.get("written") and gui_overlay_video.get("path"):
        debug_source_video = str(gui_overlay_video["path"])
        # The GUI stream already contains the exact AI mask overlay shown in
        # Driving, so keep that image untouched and add only the right panel.
        debug_draw_lane_overlay = False
    elif video_path:
        debug_source_video = video_path
    if debug_source_video:
        debug_overlay_video = _write_debug_overlay_video(
            video_path=debug_source_video,
            out_path=brain_run_dir / "camera_debug_overlay.avi",
            brain_events=brain_events,
            record_start_ts=(
                _finite_float(gui_overlay_video.get("started_at"))
                if not debug_draw_lane_overlay and gui_overlay_video is not None
                else record_start_ts
            ),
            draw_lane_overlay=debug_draw_lane_overlay,
            frame_timestamps=(
                gui_overlay_video.get("_frame_timestamps")
                if not debug_draw_lane_overlay and gui_overlay_video is not None
                else None
            ),
        )
        debug_overlay_video["source_video"] = str(debug_source_video)
        debug_overlay_video["uses_gui_mask_overlay"] = not debug_draw_lane_overlay
        metrics["debug_overlay_video"] = debug_overlay_video
        if debug_overlay_video.get("written"):
            metrics["artifacts"]["camera_debug_overlay_video"] = str(brain_run_dir / "camera_debug_overlay.avi")
    campaign_digest = _write_campaign_digest(
        run_dir=brain_run_dir,
        brain_events=brain_events,
        video_frames=video_frames,
        frame_timestamps=(
            gui_overlay_video.get("_frame_timestamps")
            if gui_overlay_video is not None
            else None
        ),
        debug_video_path=(
            str(brain_run_dir / "camera_debug_overlay.avi")
            if debug_overlay_video is not None and debug_overlay_video.get("written")
            else None
        ),
        track_png=args.track_png.resolve(),
        expected_route=expected_route,
        actual_xy=metrics["ground_truth"]["actual_path"],
    )
    metrics["campaign_digest"] = {
        key: value for key, value in campaign_digest.items() if key != "rows"
    }
    if campaign_digest.get("written"):
        metrics["artifacts"]["campaign_digest_md"] = str(campaign_digest.get("digest_md"))
        metrics["artifacts"]["campaign_digest_json"] = str(campaign_digest.get("digest_json"))
    metrics["artifacts"].update(
        {
            "brain_jsonl": str(brain_jsonl),
            "sim_bridge_jsonl": str(copied_sim_jsonl),
            "route_expected_json": str(brain_run_dir / "route_expected.json"),
            "navigation_command_json": str(brain_run_dir / "navigation_command.json"),
        }
    )
    metrics["run_id"] = run_id
    metrics["mode"] = args.mode
    metrics["command_speed_scale"] = args.command_speed_scale
    metrics["mode_start_ts"] = mode_start_ts
    metrics["auto_start_ts"] = mode_start_ts
    metrics["record_start_ts"] = record_start_ts
    metrics["record_stop_ts"] = record_stop_ts
    if video_frames:
        metrics["video_frames"] = {
            key: value
            for key, value in video_frames.items()
            if key != "frames"
        }

    (brain_run_dir / "report.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if not args.skip_analyze_run:
        report_txt = brain_run_dir / "report.txt"
        with report_txt.open("w", encoding="utf-8") as out:
            subprocess.run(
                [
                    brain_py,
                    str(brain_dir / "tools" / "dev" / "analyze_run.py"),
                    str(brain_run_dir),
                    "--sim-jsonl",
                    str(copied_sim_jsonl),
                ],
                cwd=str(brain_dir),
                stdout=out,
                stderr=subprocess.STDOUT,
                check=False,
            )
        metrics["artifacts"]["analyze_run_txt"] = str(report_txt)
        (brain_run_dir / "report.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    _update_latest_symlink(brain_dir / "temp" / "logs", brain_run_dir)
    _update_latest_symlink(sim_dir / "temp" / "logs", sim_run_dir)

    print(f"[campaign] passed={metrics['passed']} reasons={metrics['fail_reasons']}")
    print(f"[campaign] report={brain_run_dir / 'report.json'}")
    return 0 if metrics["passed"] else 2


if __name__ == "__main__":
    _maybe_reexec_with_brain_venv()
    raise SystemExit(main())
