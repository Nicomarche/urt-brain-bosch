from __future__ import annotations

import math
from statistics import median
from typing import Iterable

import numpy as np

from src.core.types.perception import LaneObservation, lane_observation_has_visual_path
from src.core.types.pose import Pose2D, VisualLaneMatch
from src.core.types.routing import RouteContext


def _wrap_angle(angle_rad: float) -> float:
    angle = float(angle_rad)
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _body_waypoints_to_map(
    waypoints: Iterable[tuple[float, float, float]],
    ego_pose: Pose2D,
) -> np.ndarray:
    pose_x = float(ego_pose.x)
    pose_y = float(ego_pose.y)
    pose_yaw = float(ego_pose.yaw)
    cos_y = math.cos(pose_yaw)
    sin_y = math.sin(pose_yaw)
    pts: list[tuple[float, float]] = []
    for item in waypoints or ():
        if item is None or len(item) < 2:
            continue
        try:
            x_fwd = float(item[0])
            y_left = float(item[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x_fwd) and math.isfinite(y_left)):
            continue
        pts.append((
            pose_x + cos_y * x_fwd + sin_y * y_left,
            pose_y + sin_y * x_fwd - cos_y * y_left,
        ))
    return np.asarray(pts, dtype=float)


def _coerce_route_xy(route_waypoints, matched_idx: int, *, max_points: int = 60) -> np.ndarray:
    try:
        route = np.asarray(route_waypoints or (), dtype=float)
    except (TypeError, ValueError):
        return np.zeros((0, 2), dtype=float)
    if route.ndim != 2 or route.shape[0] < 2 or route.shape[1] < 2:
        return np.zeros((0, 2), dtype=float)
    idx0 = max(0, min(int(matched_idx or 0), route.shape[0] - 1))
    tail = route[idx0 : idx0 + max(2, int(max_points)), :2]
    if tail.shape[0] < 2 and route.shape[0] >= 2:
        tail = route[max(0, route.shape[0] - 2) :, :2]
    return np.asarray(tail, dtype=float)


def _project_to_polyline(point: np.ndarray, polyline: np.ndarray) -> tuple[float, float]:
    best_dist = math.inf
    best_offset = 0.0
    best_heading = 0.0
    px, py = float(point[0]), float(point[1])
    for idx in range(polyline.shape[0] - 1):
        ax, ay = float(polyline[idx, 0]), float(polyline[idx, 1])
        bx, by = float(polyline[idx + 1, 0]), float(polyline[idx + 1, 1])
        seg_dx = bx - ax
        seg_dy = by - ay
        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len_sq <= 1e-12:
            continue
        seg_len = math.sqrt(seg_len_sq)
        t = ((px - ax) * seg_dx + (py - ay) * seg_dy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        qx = ax + t * seg_dx
        qy = ay + t * seg_dy
        dist = math.hypot(px - qx, py - qy)
        if dist < best_dist:
            best_dist = dist
            best_offset = (seg_dx * (py - ay) - seg_dy * (px - ax)) / seg_len
            best_heading = math.atan2(seg_dy, seg_dx)
    return float(best_offset), float(best_heading)


def _sample_indices(count: int) -> tuple[int, ...]:
    if count <= 0:
        return ()
    if count <= 3:
        return tuple(range(count))
    return tuple(dict.fromkeys((0, count // 3, (2 * count) // 3, count - 1)))


def match_visual_lane_to_route(
    route_context: RouteContext | None,
    lane_observation: LaneObservation | None,
    ego_pose: Pose2D,
    *,
    enabled: bool = True,
    min_quality: float = 0.55,
    min_points: int = 8,
    lane_half_width_m: float = 0.175,
    max_lateral_error_m: float = 0.175,
    max_yaw_error_rad: float = math.radians(6.0),
    max_sample_lateral_error_m: float | None = None,
    max_near_yaw_error_rad: float | None = None,
    max_sample_yaw_error_rad: float | None = None,
    min_confidence: float = 0.45,
) -> VisualLaneMatch | None:
    if not enabled or route_context is None or lane_observation is None:
        return None
    if not bool(route_context.route_active):
        return None
    if not lane_observation_has_visual_path(
        lane_observation,
        min_quality=float(min_quality),
        min_points=int(min_points),
    ):
        return None

    route_xy = _coerce_route_xy(route_context.route_waypoints, int(route_context.matched_idx or 0))
    if route_xy.shape[0] < 2:
        matched = route_context.matched_pose
        yaw = float(route_context.path_psi or matched.yaw or ego_pose.yaw)
        route_xy = np.asarray(
            [
                [float(matched.x), float(matched.y)],
                [float(matched.x) + math.cos(yaw), float(matched.y) + math.sin(yaw)],
            ],
            dtype=float,
        )

    visual_xy = _body_waypoints_to_map(lane_observation.center_waypoints_body, ego_pose)
    if visual_xy.shape[0] < 2:
        return None

    visual_headings = np.zeros((visual_xy.shape[0],), dtype=float)
    diffs = np.diff(visual_xy, axis=0)
    segment_headings = np.asarray([math.atan2(float(dy), float(dx)) for dx, dy in diffs], dtype=float)
    if segment_headings.size:
        visual_headings[:-1] = segment_headings
        visual_headings[-1] = segment_headings[-1]

    lateral_errors: list[float] = []
    yaw_errors: list[float] = []
    for idx in _sample_indices(visual_xy.shape[0]):
        lateral_m, route_heading = _project_to_polyline(visual_xy[idx], route_xy)
        lateral_errors.append(float(lateral_m))
        yaw_errors.append(_wrap_angle(float(visual_headings[idx]) - float(route_heading)))
    if not lateral_errors:
        return None

    lateral_error_m = float(median(lateral_errors))
    yaw_error_rad = float(median(yaw_errors)) if yaw_errors else 0.0
    near_lateral_error_m = float(lateral_errors[0])
    near_yaw_error_rad = float(yaw_errors[0]) if yaw_errors else 0.0
    max_abs_lateral_error_m = float(max(abs(v) for v in lateral_errors))
    max_abs_yaw_error_rad = float(max(abs(v) for v in yaw_errors)) if yaw_errors else 0.0
    sample_lateral_limit_m = (
        float(max_sample_lateral_error_m)
        if max_sample_lateral_error_m is not None
        else float(max_lateral_error_m) * 1.35
    )
    near_yaw_limit_rad = (
        float(max_near_yaw_error_rad)
        if max_near_yaw_error_rad is not None
        else float(max_yaw_error_rad)
    )
    sample_yaw_limit_rad = (
        float(max_sample_yaw_error_rad)
        if max_sample_yaw_error_rad is not None
        else float(max_yaw_error_rad) * 1.35
    )
    lateral_norm = abs(lateral_error_m) / max(float(lane_half_width_m), 1e-6)
    yaw_norm = abs(yaw_error_rad) / max(float(max_yaw_error_rad), 1e-6)
    score = min(1.0, (0.65 * lateral_norm) + (0.35 * yaw_norm))
    confidence = max(0.0, min(1.0, float(lane_observation.quality or 0.0) * (1.0 - score)))
    reason = "accepted"
    if abs(lateral_error_m) > float(max_lateral_error_m):
        reason = "median_lateral_error"
    elif abs(yaw_error_rad) > float(max_yaw_error_rad):
        reason = "median_yaw_error"
    elif abs(near_lateral_error_m) > float(max_lateral_error_m):
        reason = "near_lateral_error"
    elif max_abs_lateral_error_m > float(sample_lateral_limit_m):
        reason = "sample_lateral_error"
    elif abs(near_yaw_error_rad) > float(near_yaw_limit_rad):
        reason = "near_yaw_error"
    elif max_abs_yaw_error_rad > float(sample_yaw_limit_rad):
        reason = "sample_yaw_error"
    elif confidence < float(min_confidence):
        reason = "low_confidence"
    accepted = (
        reason == "accepted"
    )
    return VisualLaneMatch(
        lanelet_id=route_context.current_lanelet_id,
        lateral_error_m=lateral_error_m,
        yaw_error_rad=yaw_error_rad,
        near_lateral_error_m=near_lateral_error_m,
        near_yaw_error_rad=near_yaw_error_rad,
        max_abs_lateral_error_m=max_abs_lateral_error_m,
        max_abs_yaw_error_rad=max_abs_yaw_error_rad,
        sample_count=len(lateral_errors),
        score=float(score),
        confidence=float(confidence),
        accepted=bool(accepted),
        reason=str(reason),
    )
