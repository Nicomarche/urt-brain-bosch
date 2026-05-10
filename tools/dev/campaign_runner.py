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
import bisect
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
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


def _wait_for_auto_state(brain_jsonl: Path, timeout_s: float) -> bool:
    follower = JsonlFollower(brain_jsonl)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for item in follower.read_new():
            thread = item.get("thread")
            event = item.get("event")
            if thread == "dashboard" and event == "state_change_request":
                if str(item.get("to_mode") or "").upper() == "AUTO":
                    return True
            if thread == "dispatcher" and event == "dispatch_decision":
                if str(item.get("state") or "").upper() == "AUTO":
                    return True
            if thread == "state_machine" and event == "state_change":
                if str(item.get("to") or "").upper() == "AUTO":
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


def _compute_metrics(
    *,
    expected_route: dict[str, Any],
    brain_events: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    auto_start_ts: float | None,
    trace_start_ts: float | None,
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
    run_start_ts = trace_start_ts if trace_start_ts is not None else auto_start_ts
    if run_start_ts is not None:
        gt_poses = [ev for ev in gt_poses if float(ev.get("ts", 0.0) or 0.0) >= run_start_ts]

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
    route_completed = bool(route_result.completed or logged_completed)
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
        if lane_p90 is not None and lane_p90 <= max_lane_offset_p90_m:
            pass_reasons.append("visual_lane_offset_p90_ok")
        else:
            fail_reasons.append("visual_lane_offset_p90_high")
        if lane_max is not None and lane_max <= max_lane_offset_max_m:
            pass_reasons.append("visual_lane_offset_max_ok")
        else:
            fail_reasons.append("visual_lane_offset_max_high")
    else:
        fail_reasons.append("no_visual_lane_offset_samples")

    passed = not fail_reasons
    return {
        "passed": passed,
        "pass_reasons": pass_reasons,
        "fail_reasons": fail_reasons,
        "route": {
            "completed": route_completed,
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
            "trace_start_ts": run_start_ts,
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


def _find_auto_start_ts(events: list[dict[str, Any]]) -> float | None:
    for ev in events:
        if ev.get("thread") == "dashboard" and ev.get("event") == "state_change_request":
            if str(ev.get("to_mode") or "").upper() == "AUTO":
                return float(ev.get("ts", 0.0) or 0.0)
    for ev in events:
        if ev.get("thread") == "state_machine" and str(ev.get("to") or "").upper() == "AUTO":
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


def _split_polyline_on_jumps(
    points: list[tuple[float, float]],
    *,
    max_jump_m: float = 0.60,
) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for point in points:
        if current:
            last_x, last_y = current[-1]
            if math.hypot(float(point[0]) - last_x, float(point[1]) - last_y) > max_jump_m:
                if len(current) >= 2:
                    segments.append(current)
                current = []
        current.append(point)
    if len(current) >= 2:
        segments.append(current)
    return segments


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
    for segment in _split_polyline_on_jumps(actual_points):
        draw_polyline(segment, (30, 30, 220), 3)
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
    for segment in _split_polyline_on_jumps(actual_points):
        draw_polyline(segment, (30, 30, 220), 3)

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


def _extract_video_frames(video_path: str | Path, run_dir: Path, frames_subdir: str = "frames") -> dict[str, Any]:
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
    frames_dir = run_dir / frames_subdir
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


def _index_image_frames(frames_dir: Path, *, fps: float | None = None, video_path: Path | None = None) -> dict[str, Any]:
    images = sorted(
        [
            path
            for pattern in ("frame_*.jpg", "frame_*.jpeg", "frame_*.png")
            for path in frames_dir.glob(pattern)
        ]
    )
    frame_records: list[dict[str, Any]] = []
    index_jsonl = frames_dir / "index.jsonl"
    if index_jsonl.exists():
        for line in index_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                frame_records.append(record)
    if frame_records:
        first_wall_time = float(frame_records[0].get("wall_time", 0.0) or 0.0)
        index = []
        for idx, record in enumerate(frame_records, start=1):
            wall_time = float(record.get("wall_time", 0.0) or 0.0)
            item = dict(record)
            item.setdefault("frame", idx)
            item["time_s"] = (wall_time - first_wall_time) if wall_time and first_wall_time else (
                ((idx - 1) / fps) if fps and fps > 0.0 else None
            )
            index.append(item)
    else:
        index = [
            {
                "frame": idx,
                "time_s": ((idx - 1) / fps) if fps and fps > 0.0 else None,
                "path": str(path),
            }
            for idx, path in enumerate(images, start=1)
        ]
    meta: dict[str, Any] = {
        "video_path": str(video_path) if video_path is not None else None,
        "frames_dir": str(frames_dir),
        "index_json": str(frames_dir / "index.json"),
        "index_jsonl": str(index_jsonl) if index_jsonl.exists() else None,
        "fps": float(fps or 0.0),
        "frame_count": len(images),
        "saved_frame_count": len(images),
        "frames": index,
    }
    frames_dir.mkdir(parents=True, exist_ok=True)
    (frames_dir / "index.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return meta


def _write_video_from_image_frames(frames_dir: Path, out_path: Path, *, fps: float = 10.0) -> dict[str, Any]:
    try:
        import cv2
    except ImportError:
        return {"video_path": str(out_path), "error": "opencv_unavailable"}

    images = sorted(
        [
            path
            for pattern in ("frame_*.jpg", "frame_*.jpeg", "frame_*.png")
            for path in frames_dir.glob(pattern)
        ]
    )
    if not images:
        return {"video_path": str(out_path), "error": "no_image_frames"}

    first = cv2.imread(str(images[0]))
    if first is None:
        return {"video_path": str(out_path), "error": "first_frame_read_failed"}
    height, width = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"XVID"),
        max(1.0, float(fps or 10.0)),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        return {"video_path": str(out_path), "error": "video_writer_open_failed"}

    written = 0
    try:
        for path in images:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    meta: dict[str, Any] = {
        "video_path": str(out_path),
        "frames_dir": str(frames_dir),
        "fps": float(fps or 10.0),
        "width": int(width),
        "height": int(height),
        "frame_count": len(images),
        "written_frame_count": written,
    }
    if written <= 0:
        meta["error"] = "no_frames_written"
    return meta


def _write_satellite_artifacts(
    *,
    brain_run_dir: Path,
    brain_jsonl: Path,
    sim_jsonl: Path,
    sim_dir: Path,
    record_start_ts: float | None,
    stride: int,
    width_px: int,
) -> dict[str, Any]:
    try:
        from tools.dev.render_satellite_frames import render_satellite_frames
    except Exception as exc:
        return {"error": f"import_failed:{exc}"}

    args = argparse.Namespace(
        run_dir=str(brain_run_dir),
        sim_jsonl=str(sim_jsonl),
        brain_jsonl=str(brain_jsonl),
        world=str(sim_dir / "Simulator/src/sim_pkg/worlds/world_with_separators.world"),
        texture=str(sim_dir / "Simulator/src/models_pkg/track/materials/textures/new_Small.png"),
        out_dir=None,
        width_px=int(width_px),
        stride=max(1, int(stride)),
        max_frames=0,
        record_start_ts=record_start_ts,
        video=True,
    )
    try:
        return render_satellite_frames(args)
    except Exception as exc:
        return {"error": str(exc)}


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
    parser.add_argument("--duration", type=float, default=240.0, help="Max seconds in AUTO for the route.")
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
    parser.add_argument("--no-record", action="store_true", help="Do not toggle camera AVI recording.")
    parser.add_argument("--no-ai-debug-video", action="store_true", help="Do not record the AI/dashboard debug preview.")
    parser.add_argument("--no-satellite", action="store_true", help="Do not render Gazebo-texture top-down frames.")
    parser.add_argument("--satellite-stride", type=int, default=1, help="Render every Nth camera frame in the top-down view.")
    parser.add_argument("--satellite-width-px", type=int, default=1280, help="Width of rendered top-down frames.")
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
    ai_debug_live_video = brain_run_dir / "ai_debug_live.avi"
    ai_debug_video = brain_run_dir / "ai_debug.avi"
    ai_debug_frames_dir = brain_run_dir / "ai_debug_frames"

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
        if not args.no_ai_debug_video:
            brain_env["URT_RECORD_AI_DEBUG_VIDEO"] = "1"
            brain_env["URT_AI_DEBUG_VIDEO_PATH"] = str(ai_debug_live_video)
            brain_env["URT_AI_DEBUG_FRAMES_DIR"] = str(ai_debug_frames_dir)
            brain_env.setdefault("URT_AI_DEBUG_VIDEO_FPS", "10")
            print(f"[campaign] AI debug preview: {ai_debug_frames_dir}")
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

        auto_ok = False
        for attempt in range(1, 4):
            print(f"[campaign] requesting AUTO attempt {attempt}")
            if not client.emit_message("DrivingMode", "auto", wait_response=True):
                print("[campaign] WARNING: dashboard did not acknowledge DrivingMode=auto")
            if _wait_for_auto_state(brain_jsonl, 5.0):
                auto_ok = True
                break
            time.sleep(0.5)
        if not auto_ok:
            raise RuntimeError("AUTO mode was not confirmed by the brain")

        route_result = _route_progress_from_logs(brain_jsonl, args.duration)
        client.emit_message("DrivingMode", "stop")
        if not args.no_record:
            client.emit_message("Record", False, wait_response=True)
            record_stop_ts = time.time()
        time.sleep(0.5)

    except Exception as exc:
        print(f"[campaign] ERROR: {exc}", file=sys.stderr)
        route_result = RouteCommandResult(timed_out=True)
    finally:
        if client is not None:
            client.close()
        if not args.keep_running:
            for proc in reversed(processes):
                print(f"[campaign] stopping {proc.name}")
                proc.terminate()
            _cleanup_known_processes(brain_dir, sim_dir)

    if sim_jsonl.exists():
        shutil.copy2(sim_jsonl, copied_sim_jsonl)

    video_path = None if args.no_record else _collect_camera_video(brain_dir, brain_run_dir, started_at)
    video_frames: dict[str, Any] | None = None
    if video_path:
        video_frames = _extract_video_frames(video_path, brain_run_dir)
    ai_debug_meta: dict[str, Any] | None = None
    ai_debug_video_meta: dict[str, Any] | None = None
    if not args.no_ai_debug_video:
        if ai_debug_frames_dir.exists():
            ai_debug_video_meta = _write_video_from_image_frames(
                ai_debug_frames_dir,
                ai_debug_video,
                fps=float(os.environ.get("URT_AI_DEBUG_VIDEO_FPS", "10") or 10.0),
            )
            ai_debug_meta = _index_image_frames(
                ai_debug_frames_dir,
                fps=float(ai_debug_video_meta.get("fps", 10.0) if ai_debug_video_meta else 10.0),
                video_path=ai_debug_video if ai_debug_video.exists() else ai_debug_live_video,
            )
        elif ai_debug_live_video.exists():
            ai_debug_meta = _extract_video_frames(ai_debug_live_video, brain_run_dir, frames_subdir="ai_debug_frames")
            if ai_debug_live_video != ai_debug_video:
                try:
                    shutil.copy2(ai_debug_live_video, ai_debug_video)
                except OSError:
                    pass
    brain_events = _load_jsonl(brain_jsonl)
    gt_events = _load_jsonl(copied_sim_jsonl)
    auto_start_ts = _find_auto_start_ts(brain_events)
    trace_start_ts = record_start_ts if record_start_ts is not None else auto_start_ts
    metrics = _compute_metrics(
        expected_route=expected_route,
        brain_events=brain_events,
        gt_events=gt_events,
        auto_start_ts=auto_start_ts,
        trace_start_ts=trace_start_ts,
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
        if video_frames:
            metrics["artifacts"]["camera_frames_dir"] = str(brain_run_dir / "frames")
            metrics["artifacts"]["camera_frames_index_json"] = str(brain_run_dir / "frames" / "index.json")
    else:
        metrics["artifacts"] = {}
    if ai_debug_live_video.exists():
        metrics["artifacts"]["ai_debug_live_video"] = str(ai_debug_live_video)
    if ai_debug_video.exists():
        metrics["artifacts"]["ai_debug_video"] = str(ai_debug_video)
    if ai_debug_meta:
        metrics["artifacts"]["ai_debug_frames_dir"] = str(ai_debug_frames_dir)
        metrics["artifacts"]["ai_debug_frames_index_json"] = str(ai_debug_frames_dir / "index.json")
        if (ai_debug_frames_dir / "index.jsonl").exists():
            metrics["artifacts"]["ai_debug_frames_index_jsonl"] = str(ai_debug_frames_dir / "index.jsonl")
    if ai_debug_video_meta and ai_debug_video_meta.get("error"):
        metrics["artifacts"]["ai_debug_video_error"] = ai_debug_video_meta.get("error")

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
    if not args.no_satellite and video_frames and copied_sim_jsonl.exists():
        satellite_meta = _write_satellite_artifacts(
            brain_run_dir=brain_run_dir,
            brain_jsonl=brain_jsonl,
            sim_jsonl=copied_sim_jsonl,
            sim_dir=sim_dir,
            record_start_ts=record_start_ts,
            stride=args.satellite_stride,
            width_px=args.satellite_width_px,
        )
        if satellite_meta.get("error"):
            metrics["artifacts"]["satellite_error"] = satellite_meta["error"]
        else:
            metrics["artifacts"]["satellite_frames_dir"] = satellite_meta.get("frames_dir")
            metrics["artifacts"]["satellite_frames_index_json"] = str(
                Path(str(satellite_meta.get("frames_dir"))) / "index.json"
            )
            if satellite_meta.get("video_path"):
                metrics["artifacts"]["satellite_video"] = satellite_meta.get("video_path")
    metrics["artifacts"].update(
        {
            "brain_jsonl": str(brain_jsonl),
            "sim_bridge_jsonl": str(copied_sim_jsonl),
            "route_expected_json": str(brain_run_dir / "route_expected.json"),
            "navigation_command_json": str(brain_run_dir / "navigation_command.json"),
        }
    )
    metrics["run_id"] = run_id
    metrics["auto_start_ts"] = auto_start_ts
    metrics["trace_start_ts"] = trace_start_ts
    metrics["record_start_ts"] = record_start_ts
    metrics["record_stop_ts"] = record_stop_ts
    if video_frames:
        metrics["video_frames"] = {
            key: value
            for key, value in video_frames.items()
            if key != "frames"
        }
    if ai_debug_meta:
        metrics["ai_debug_video_frames"] = {
            key: value
            for key, value in ai_debug_meta.items()
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
