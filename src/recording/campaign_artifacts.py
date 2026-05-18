"""Artefactos compatibles con ``tools/dev/campaign_runner.py``.

El formato esperado por los runs de campaign es:

* ``camera.avi``
* ``frames/frame_000001.png`` + ``frames/index.json``
* ``overlay.png``
* ``overlay_visual_lane.png``

Este módulo genera los mismos nombres dentro de una sesión manual y crea
symlinks en el ``run_dir`` padre para que abrir ``temp/logs/run_*`` muestre
los artefactos esperados inmediatamente.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any


_BRAIN_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_TRACK_PNG = _BRAIN_DIR / "maps" / "sim" / "track.png"


def write_campaign_artifacts(session_dir: Path, *, run_dir: Path | None = None) -> dict[str, Any]:
    """Genera artefactos estilo campaign para una sesión manual.

    Args:
        session_dir: carpeta ``manual_<ts>``.
        run_dir: carpeta ``run_<ts>``. Si no se pasa, usa el padre de la sesión.
    """
    run_dir = run_dir or session_dir.parent
    artifacts: dict[str, Any] = {}

    camera_path = _ensure_camera_video_name(session_dir)
    if camera_path is not None:
        artifacts["camera_video"] = str(camera_path)
        frame_meta = _extract_video_frames(camera_path, session_dir)
        artifacts["camera_frames_dir"] = str(session_dir / "frames")
        artifacts["camera_frames_index_json"] = str(session_dir / "frames" / "index.json")
        artifacts["video_frames"] = {k: v for k, v in frame_meta.items() if k != "frames"}

    brain_jsonl = run_dir / "brain.jsonl"
    brain_events = _load_jsonl(brain_jsonl)
    expected_route = _load_expected_route(run_dir)
    start_ts, end_ts = _session_time_window(session_dir)
    actual_segments = _actual_path_segments_from_brain_events(
        brain_events,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    overlay_path = session_dir / "overlay.png"
    if _write_overlay(
        out_path=overlay_path,
        track_png=_DEFAULT_TRACK_PNG,
        expected_route=expected_route,
        actual_xy=actual_segments,
    ):
        artifacts["overlay_png"] = str(overlay_path)
        artifacts["actual_path_segments"] = len(actual_segments)
        artifacts["actual_path_points"] = sum(len(segment) for segment in actual_segments)

    fps = None
    if isinstance(artifacts.get("video_frames"), dict):
        fps = _finite_float(artifacts["video_frames"].get("fps"))
    visual_samples = _collect_visual_lane_samples(
        brain_events,
        record_start_ts=start_ts,
        record_end_ts=end_ts,
        fps=fps,
    )
    visual_overlay_path = session_dir / "overlay_visual_lane.png"
    if _write_visual_lane_overlay(
        out_path=visual_overlay_path,
        track_png=_DEFAULT_TRACK_PNG,
        expected_route=expected_route,
        actual_xy=actual_segments,
        visual_samples=visual_samples,
    ):
        artifacts["overlay_visual_lane_png"] = str(visual_overlay_path)

    (session_dir / "artifacts.json").write_text(
        json.dumps(artifacts, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    _mirror_campaign_artifacts(session_dir, run_dir)
    return artifacts


def _ensure_camera_video_name(session_dir: Path) -> Path | None:
    camera = session_dir / "camera.avi"
    if camera.exists():
        return camera
    raw = session_dir / "raw_video.avi"
    if raw.exists():
        try:
            raw.rename(camera)
            return camera
        except OSError:
            try:
                shutil.copy2(raw, camera)
                return camera
            except OSError:
                return raw
    return None


def _extract_video_frames(video_path: str | Path, run_dir: Path) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
    return events


def _load_expected_route(run_dir: Path) -> dict[str, Any]:
    route_path = run_dir / "route_expected.json"
    if route_path.exists():
        try:
            payload = json.loads(route_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "map_metadata": _track_metadata(_DEFAULT_TRACK_PNG),
        "waypoints": [],
    }


def _track_metadata(track_png: Path) -> dict[str, Any]:
    meta_path = track_png.with_name("track_meta.json")
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    meta: dict[str, Any] = {}
    if payload.get("metersPerPixel") is not None:
        meta["meters_per_pixel"] = payload.get("metersPerPixel")
    elif payload.get("meters_per_pixel") is not None:
        meta["meters_per_pixel"] = payload.get("meters_per_pixel")
    if payload.get("imgW") is not None:
        meta["image_width_px"] = payload.get("imgW")
    if payload.get("imgH") is not None:
        meta["image_height_px"] = payload.get("imgH")
    if payload.get("y_axis_inverted") is not None:
        meta["y_axis_inverted"] = bool(payload.get("y_axis_inverted"))
    if isinstance(payload.get("world_bounds"), dict):
        meta["world_bounds"] = dict(payload["world_bounds"])
    return meta


def _actual_path_from_brain_events(events: list[dict[str, Any]]) -> list[list[float]]:
    """Backward-compatible flattened driven path for tests/tools."""
    return [
        point
        for segment in _actual_path_segments_from_brain_events(events)
        for point in segment
    ]


def _actual_path_segments_from_brain_events(
    events: list[dict[str, Any]],
    *,
    start_ts: float | None = None,
    end_ts: float | None = None,
    max_jump_m: float = 0.08,
    max_speed_mps: float = 1.0,
) -> list[list[list[float]]]:
    """Return actual driven pose segments, excluding setup/relocalization jumps.

    Campaign overlays are meant to show what the car physically drove during
    this session. Manual map_view relocalization and start calibration are
    pose resets, not travelled distance, so they must not be connected into
    the red "actual" trajectory.
    """
    segments: list[list[list[float]]] = []
    current_segment: list[list[float]] = []
    last_xy_ts: tuple[float, float, float] | None = None

    def finish_segment() -> None:
        nonlocal current_segment, last_xy_ts
        if len(current_segment) >= 2:
            segments.append(current_segment)
        current_segment = []
        last_xy_ts = None

    for ev in events:
        ts = _finite_float(ev.get("ts"))
        if ts is None:
            continue
        if start_ts is not None and ts < float(start_ts):
            continue
        if end_ts is not None and ts > float(end_ts):
            continue
        if ev.get("thread") != "pose_estimator" or ev.get("event") != "pose_published":
            continue

        x = _finite_float(ev.get("fused_x"))
        y = _finite_float(ev.get("fused_y"))
        if x is None or y is None:
            continue

        if _pose_event_is_external_relocalization(ev):
            finish_segment()
            current_segment.append([float(x), float(y)])
            last_xy_ts = (float(ts), float(x), float(y))
            continue

        if last_xy_ts is not None:
            prev_ts, prev_x, prev_y = last_xy_ts
            dt_s = max(0.0, float(ts) - float(prev_ts))
            distance_m = math.hypot(float(x) - float(prev_x), float(y) - float(prev_y))
            allowed_step_m = max(float(max_jump_m), float(max_speed_mps) * dt_s)
            if distance_m > allowed_step_m:
                finish_segment()

        current = (round(float(x), 4), round(float(y), 4))
        if current_segment:
            previous = current_segment[-1]
            if current == (round(float(previous[0]), 4), round(float(previous[1]), 4)):
                last_xy_ts = (float(ts), float(x), float(y))
                continue
        current_segment.append([float(x), float(y)])
        last_xy_ts = (float(ts), float(x), float(y))

    finish_segment()
    return segments


def _pose_event_is_external_relocalization(ev: dict[str, Any]) -> bool:
    mode = str(ev.get("reloc_mode") or ev.get("relocalization_mode") or "").strip().lower()
    source = str(ev.get("reloc_source") or ev.get("last_relocalization_source") or "").strip().lower()
    external_modes = {
        "auto_gps_disabled",
        "auto_gps_entry_pending",
        "gps_disabled_start",
        "gps_fix",
        "start_gps_calibration",
        "start_gps_calibration_failed",
        "start_gps_calibration_pending",
    }
    if mode in external_modes:
        return True
    return "manual_dashboard" in source or "start_calibration" in source


def _overlay_map_metadata(track_png: Path, expected_route: dict[str, Any]) -> dict[str, Any]:
    meta = dict(expected_route.get("map_metadata") or {})
    meta.update({k: v for k, v in _track_metadata(track_png).items() if v is not None})
    return meta


def _write_overlay(
    *,
    out_path: Path,
    track_png: Path,
    expected_route: dict[str, Any],
    actual_xy: list[list[float]] | list[list[list[float]]],
) -> bool:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return False

    meta = _overlay_map_metadata(track_png, expected_route)
    img = cv2.imread(str(track_png), cv2.IMREAD_COLOR) if track_png.exists() else None
    if img is None:
        width = int(meta.get("image_width_px", 2039) or 2039)
        height = int(meta.get("image_height_px", 1343) or 1343)
        img = np.full((height, width, 3), 245, dtype=np.uint8)

    world_to_pixel = _world_to_pixel_factory(img, meta)

    def draw_polyline(points: list[tuple[float, float]], color: tuple[int, int, int], thickness: int) -> None:
        if len(points) < 2:
            return
        pix = np.asarray([world_to_pixel(x, y) for x, y in points], dtype=np.int32)
        cv2.polylines(img, [pix], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    def draw_segments(segments: list[list[tuple[float, float]]], color: tuple[int, int, int], thickness: int) -> None:
        for segment in segments:
            draw_polyline(segment, color, thickness)

    expected_xy = [
        (float(pt[0]), float(pt[1]))
        for pt in expected_route.get("waypoints", [])
        if isinstance(pt, list) and len(pt) >= 2
    ]
    actual_segments = _normalize_path_segments(actual_xy)
    draw_polyline(expected_xy, (0, 170, 0), 4)
    draw_segments(actual_segments, (30, 30, 220), 3)
    cv2.putText(img, "expected", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "actual driven", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 220), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), img))


def _collect_visual_lane_samples(
    brain_events: list[dict[str, Any]],
    *,
    record_start_ts: float | None = None,
    record_end_ts: float | None = None,
    fps: float | None = None,
) -> list[dict[str, Any]]:
    pose_events = [
        ev for ev in brain_events
        if ev.get("thread") in {"nav_planner", "pose_estimator"}
        and ev.get("event") in {"route_update", "pose_published"}
        and _event_in_window(ev, start_ts=record_start_ts, end_ts=record_end_ts)
    ]
    pose_ts = [float(ev.get("ts", 0.0) or 0.0) for ev in pose_events]

    def pose_at(ts: float) -> dict[str, Any]:
        if not pose_ts:
            return {}
        idx = bisect.bisect_right(pose_ts, ts) - 1
        if idx < 0:
            idx = 0
        return pose_events[idx]

    samples: list[dict[str, Any]] = []
    for ev in brain_events:
        if ev.get("thread") != "lane_observer" or ev.get("event") != "lane_obs":
            continue
        if not _event_in_window(ev, start_ts=record_start_ts, end_ts=record_end_ts):
            continue
        if float(ev.get("quality", 0.0) or 0.0) <= 0.5:
            continue
        ts = float(ev.get("ts", 0.0) or 0.0)
        pose_ev = pose_at(ts)
        pose_x = _finite_float(pose_ev.get("pose_x", pose_ev.get("fused_x")))
        pose_y = _finite_float(pose_ev.get("pose_y", pose_ev.get("fused_y")))
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
                "lanelet_id": str(pose_ev.get("current_lanelet_id")) if pose_ev.get("current_lanelet_id") is not None else None,
                "route_progress": _finite_float(pose_ev.get("route_progress")),
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
    actual_xy: list[list[float]] | list[list[list[float]]],
    visual_samples: list[dict[str, Any]],
    min_abs_m: float = 0.08,
) -> bool:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return False

    meta = _overlay_map_metadata(track_png, expected_route)
    img = cv2.imread(str(track_png), cv2.IMREAD_COLOR) if track_png.exists() else None
    if img is None:
        width = int(meta.get("image_width_px", 2039) or 2039)
        height = int(meta.get("image_height_px", 1343) or 1343)
        img = np.full((height, width, 3), 245, dtype=np.uint8)

    world_to_pixel = _world_to_pixel_factory(img, meta)

    def draw_polyline(points: list[tuple[float, float]], color: tuple[int, int, int], thickness: int) -> None:
        if len(points) < 2:
            return
        pix = np.asarray([world_to_pixel(x, y) for x, y in points], dtype=np.int32)
        cv2.polylines(img, [pix], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    def draw_segments(segments: list[list[tuple[float, float]]], color: tuple[int, int, int], thickness: int) -> None:
        for segment in segments:
            draw_polyline(segment, color, thickness)

    expected_xy = [
        (float(pt[0]), float(pt[1]))
        for pt in expected_route.get("waypoints", [])
        if isinstance(pt, list) and len(pt) >= 2
    ]
    actual_segments = _normalize_path_segments(actual_xy)
    draw_polyline(expected_xy, (0, 170, 0), 4)
    draw_segments(actual_segments, (30, 30, 220), 3)

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
    cv2.putText(img, "actual driven", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 220), 2, cv2.LINE_AA)
    cv2.putText(img, "visual lane offset: cyan>8cm orange>12cm magenta>18cm", (24, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), img))


def _world_to_pixel_factory(img, meta: dict[str, Any]):
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

    return world_to_pixel


def _mirror_campaign_artifacts(session_dir: Path, run_dir: Path) -> None:
    for name in ("camera.avi", "frames", "overlay.png", "overlay_visual_lane.png", "artifacts.json"):
        source = session_dir / name
        if not source.exists():
            continue
        target = run_dir / name
        try:
            if target.is_symlink() or target.exists():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            target.symlink_to(source.relative_to(run_dir), target_is_directory=source.is_dir())
        except OSError:
            try:
                if source.is_dir():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(source, target)
                else:
                    shutil.copy2(source, target)
            except OSError:
                pass


def _session_started_at(session_dir: Path) -> float | None:
    start_ts, _ = _session_time_window(session_dir)
    return start_ts


def _session_time_window(session_dir: Path) -> tuple[float | None, float | None]:
    manifest_path = session_dir / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    return _finite_float(payload.get("started_at")), _finite_float(payload.get("end_ts"))


def _event_in_window(
    ev: dict[str, Any],
    *,
    start_ts: float | None,
    end_ts: float | None,
) -> bool:
    ts = _finite_float(ev.get("ts"))
    if ts is None:
        return False
    if start_ts is not None and ts < float(start_ts):
        return False
    if end_ts is not None and ts > float(end_ts):
        return False
    return True


def _normalize_path_segments(
    path_or_segments: list[list[float]] | list[list[list[float]]],
) -> list[list[tuple[float, float]]]:
    if not path_or_segments:
        return []
    first = path_or_segments[0]
    if isinstance(first, list) and first and isinstance(first[0], list):
        raw_segments = path_or_segments  # type: ignore[assignment]
    else:
        raw_segments = [path_or_segments]  # type: ignore[list-item]

    segments: list[list[tuple[float, float]]] = []
    for raw_segment in raw_segments:
        segment: list[tuple[float, float]] = []
        for pt in raw_segment:
            try:
                if len(pt) < 2:  # type: ignore[arg-type]
                    continue
                segment.append((float(pt[0]), float(pt[1])))  # type: ignore[index]
            except (TypeError, ValueError, IndexError):
                continue
        if len(segment) >= 2:
            segments.append(segment)
    return segments


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


__all__ = ["write_campaign_artifacts"]
