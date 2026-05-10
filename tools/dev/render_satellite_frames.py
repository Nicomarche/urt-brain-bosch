#!/usr/bin/env python3
"""Render synchronized top-down diagnostic frames for a campaign run.

The background is the same track texture used by Gazebo, and the vehicle pose
comes from the gz ground-truth logger. This keeps the artifact independent from
the lanelet/SVG geometry used by the planner.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - dev tool
    raise SystemExit(f"opencv/numpy unavailable: {exc}") from exc


DEFAULT_SIM_DIR = Path("/Users/luciogarcia/urt-simulator")
DEFAULT_WORLD = DEFAULT_SIM_DIR / "Simulator/src/sim_pkg/worlds/world_with_separators.world"
DEFAULT_TEXTURE = (
    DEFAULT_SIM_DIR
    / "Simulator/src/models_pkg/track/materials/textures/new_Small.png"
)


def _load_jsonl(path: Path) -> list[dict]:
    events: list[dict] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8", errors="ignore") as fd:
        for line in fd:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _parse_float_pair(text: str | None, fallback: tuple[float, float]) -> tuple[float, float]:
    if not text:
        return fallback
    values = [part for part in text.strip().split() if part]
    if len(values) < 2:
        return fallback
    try:
        return float(values[0]), float(values[1])
    except ValueError:
        return fallback


def _track_plane_from_world(world_path: Path) -> tuple[float, float, float, float]:
    fallback = (7.34, -7.495, 20.64785, 13.79795)
    try:
        root = ET.parse(world_path).getroot()
    except (OSError, ET.ParseError):
        return fallback
    model = root.find(".//model[@name='track']")
    if model is None:
        return fallback
    pose_text = model.findtext("pose")
    pose_values = [part for part in (pose_text or "").strip().split() if part]
    try:
        center_x = float(pose_values[0])
        center_y = float(pose_values[1])
    except (IndexError, ValueError):
        center_x, center_y = fallback[:2]
    size_text = model.findtext(".//visual/geometry/plane/size")
    size_x, size_y = _parse_float_pair(size_text, fallback[2:])
    return center_x, center_y, size_x, size_y


def _world_to_pixel(
    x: float,
    y: float,
    *,
    plane: tuple[float, float, float, float],
    width_px: int,
    height_px: int,
) -> tuple[int, int]:
    center_x, center_y, size_x, size_y = plane
    x_min = center_x - size_x * 0.5
    y_min = center_y - size_y * 0.5
    u = (float(x) - x_min) / max(size_x, 1e-9)
    v = (float(y) - y_min) / max(size_y, 1e-9)
    px = int(round(u * float(width_px - 1)))
    py = int(round((1.0 - v) * float(height_px - 1)))
    return px, py


def _pose_forward_angle_gz(pose: dict) -> float:
    # The current car model's body forward axis is -Y in the model frame.
    yaw = float(pose.get("gz_yaw_rad", 0.0) or 0.0)
    return yaw - math.pi * 0.5


def _draw_vehicle(
    image: np.ndarray,
    pose: dict,
    *,
    plane: tuple[float, float, float, float],
) -> None:
    h, w = image.shape[:2]
    x = float(pose.get("gz_x", 0.0) or 0.0)
    y = float(pose.get("gz_y", 0.0) or 0.0)
    heading = _pose_forward_angle_gz(pose)
    length_m = 0.42
    width_m = 0.22
    forward = np.array([math.cos(heading), math.sin(heading)], dtype=float)
    left = np.array([-forward[1], forward[0]], dtype=float)
    center = np.array([x, y], dtype=float)
    corners = np.array(
        [
            center + forward * (length_m * 0.5) + left * (width_m * 0.5),
            center + forward * (length_m * 0.5) - left * (width_m * 0.5),
            center - forward * (length_m * 0.5) - left * (width_m * 0.5),
            center - forward * (length_m * 0.5) + left * (width_m * 0.5),
        ],
        dtype=float,
    )
    pix = np.asarray(
        [
            _world_to_pixel(float(px), float(py), plane=plane, width_px=w, height_px=h)
            for px, py in corners
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, pix, (40, 40, 230), lineType=cv2.LINE_AA)
    cv2.polylines(image, [pix], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    nose = center + forward * (length_m * 0.72)
    cpx, cpy = _world_to_pixel(x, y, plane=plane, width_px=w, height_px=h)
    npx, npy = _world_to_pixel(float(nose[0]), float(nose[1]), plane=plane, width_px=w, height_px=h)
    cv2.arrowedLine(image, (cpx, cpy), (npx, npy), (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)


def _route_updates_by_ts(brain_events: list[dict]) -> tuple[list[float], list[dict]]:
    route = [
        ev
        for ev in brain_events
        if ev.get("thread") == "nav_planner" and ev.get("event") == "route_update"
    ]
    return [float(ev.get("ts", 0.0) or 0.0) for ev in route], route


def _nearest_pose(poses: list[dict], pose_ts: list[float], ts: float) -> tuple[int, dict] | None:
    if not poses:
        return None
    idx = bisect.bisect_left(pose_ts, ts)
    if idx <= 0:
        return 0, poses[0]
    if idx >= len(poses):
        return len(poses) - 1, poses[-1]
    before = poses[idx - 1]
    after = poses[idx]
    if abs(float(after.get("ts", 0.0) or 0.0) - ts) < abs(ts - float(before.get("ts", 0.0) or 0.0)):
        return idx, after
    return idx - 1, before


def _lanelet_at(route_ts: list[float], route_events: list[dict], ts: float) -> str:
    idx = bisect.bisect_right(route_ts, ts) - 1
    if idx < 0 or idx >= len(route_events):
        return "?"
    lanelet_id = route_events[idx].get("current_lanelet_id")
    return str(lanelet_id) if lanelet_id is not None else "?"


def _draw_trail(
    image: np.ndarray,
    poses: list[dict],
    *,
    end_idx: int,
    plane: tuple[float, float, float, float],
    stride: int = 2,
) -> None:
    h, w = image.shape[:2]
    segments: list[list[tuple[int, int]]] = []
    points: list[tuple[int, int]] = []
    prev_xy: tuple[float, float] | None = None
    for pose in poses[: end_idx + 1 : max(1, stride)]:
        x = pose.get("gz_x")
        y = pose.get("gz_y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        xy = (float(x), float(y))
        if prev_xy is not None and math.hypot(xy[0] - prev_xy[0], xy[1] - prev_xy[1]) > 0.60:
            if len(points) >= 2:
                segments.append(points)
            points = []
        points.append(_world_to_pixel(xy[0], xy[1], plane=plane, width_px=w, height_px=h))
        prev_xy = xy
    if len(points) >= 2:
        segments.append(points)
    for segment in segments:
        cv2.polylines(image, [np.asarray(segment, dtype=np.int32)], False, (0, 0, 255), 3, cv2.LINE_AA)


def _read_camera_frames(run_dir: Path) -> tuple[list[dict], float | None]:
    index_path = run_dir / "frames" / "index.json"
    try:
        meta = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], None
    frames = list(meta.get("frames") or [])
    fps = meta.get("fps")
    return frames, float(fps) if isinstance(fps, (int, float)) and fps > 0 else None


def render_satellite_frames(args: argparse.Namespace) -> dict:
    run_dir = Path(args.run_dir).resolve()
    sim_jsonl = Path(args.sim_jsonl).resolve() if args.sim_jsonl else run_dir / "sim_bridge.jsonl"
    brain_jsonl = Path(args.brain_jsonl).resolve() if args.brain_jsonl else run_dir / "brain.jsonl"
    report_path = run_dir / "report.json"
    record_start_ts = getattr(args, "record_start_ts", None)
    if record_start_ts is None:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {}
        record_start_ts = report.get("record_start_ts")
    if not isinstance(record_start_ts, (int, float)):
        raise SystemExit("report.json missing numeric record_start_ts; cannot sync camera frames")

    poses = [
        ev
        for ev in _load_jsonl(sim_jsonl)
        if ev.get("thread") == "ground_truth"
        and ev.get("event") == "gz_pose"
        and isinstance(ev.get("gz_x"), (int, float))
        and isinstance(ev.get("gz_y"), (int, float))
    ]
    if not poses:
        raise SystemExit(f"no gz_pose samples found in {sim_jsonl}")
    poses.sort(key=lambda item: float(item.get("ts", 0.0) or 0.0))
    poses = [
        pose
        for pose in poses
        if float(pose.get("ts", 0.0) or 0.0) >= float(record_start_ts) - 0.5
    ]
    if not poses:
        raise SystemExit(f"no gz_pose samples overlap record_start_ts={record_start_ts}")
    pose_ts = [float(item.get("ts", 0.0) or 0.0) for item in poses]

    camera_frames, camera_fps = _read_camera_frames(run_dir)
    if not camera_frames:
        raise SystemExit(f"no extracted camera frame index found under {run_dir / 'frames'}")
    brain_events = _load_jsonl(brain_jsonl)
    route_ts, route_events = _route_updates_by_ts(brain_events)

    texture = cv2.imread(str(Path(args.texture)), cv2.IMREAD_COLOR)
    if texture is None:
        raise SystemExit(f"failed to read texture {args.texture}")
    target_width = int(args.width_px)
    scale = target_width / float(texture.shape[1])
    target_height = max(1, int(round(texture.shape[0] * scale)))
    background = cv2.resize(texture, (target_width, target_height), interpolation=cv2.INTER_AREA)
    plane = _track_plane_from_world(Path(args.world))

    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "satellite_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_stride = max(1, int(args.stride))
    max_frames = int(args.max_frames) if args.max_frames else None

    written: list[dict] = []
    video_writer = None
    video_path = out_dir.with_suffix(".avi")
    if args.video:
        fps = camera_fps if camera_fps and camera_fps > 0 else 5.0
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps / frame_stride, (target_width, target_height))

    selected_frames = camera_frames[::frame_stride]
    if max_frames is not None:
        selected_frames = selected_frames[:max_frames]
    for frame in selected_frames:
        frame_no = int(frame.get("frame", len(written) + 1) or len(written) + 1)
        time_s = float(frame.get("time_s", 0.0) or 0.0)
        ts = float(record_start_ts) + time_s
        nearest = _nearest_pose(poses, pose_ts, ts)
        if nearest is None:
            continue
        pose_idx, pose = nearest
        image = np.array(background, copy=True)
        _draw_trail(image, poses, end_idx=pose_idx, plane=plane)
        _draw_vehicle(image, pose, plane=plane)
        lanelet_id = _lanelet_at(route_ts, route_events, ts)
        label = f"frame {frame_no:06d}  t={time_s:5.1f}s  lanelet={lanelet_id}"
        cv2.rectangle(image, (0, 0), (560, 34), (0, 0, 0), -1)
        cv2.putText(image, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)
        out_path = out_dir / f"satellite_{frame_no:06d}.png"
        if cv2.imwrite(str(out_path), image):
            written.append(
                {
                    "frame": frame_no,
                    "time_s": time_s,
                    "path": str(out_path),
                    "pose_ts": float(pose.get("ts", 0.0) or 0.0),
                    "lanelet_id": lanelet_id,
                }
            )
            if video_writer is not None:
                video_writer.write(image)

    if video_writer is not None:
        video_writer.release()

    meta = {
        "frames_dir": str(out_dir),
        "frame_count": len(written),
        "source_camera_frame_count": len(camera_frames),
        "stride": frame_stride,
        "texture": str(Path(args.texture).resolve()),
        "world": str(Path(args.world).resolve()),
        "sim_jsonl": str(sim_jsonl),
        "brain_jsonl": str(brain_jsonl),
        "record_start_ts": float(record_start_ts),
        "width": target_width,
        "height": target_height,
        "video_path": str(video_path) if args.video else None,
        "frames": written,
    }
    (out_dir / "index.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Campaign run directory under temp/logs")
    parser.add_argument("--sim-jsonl", default=None)
    parser.add_argument("--brain-jsonl", default=None)
    parser.add_argument("--world", default=str(DEFAULT_WORLD))
    parser.add_argument("--texture", default=str(DEFAULT_TEXTURE))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--width-px", type=int, default=1280)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--record-start-ts", type=float, default=None)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args(argv)

    meta = render_satellite_frames(args)
    print(json.dumps({key: meta[key] for key in ("frames_dir", "frame_count", "video_path")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
