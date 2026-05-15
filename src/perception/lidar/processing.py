from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import replace
from typing import Iterable

import config
from src.core.types.lidar import LidarObstacle, LidarPoint, LidarScan
from src.core.types.pose import PoseEstimate


TAU = 2.0 * math.pi


def normalize_angle_rad(angle_rad: float) -> float:
    """Normalize to [0, 2*pi)."""

    return float(angle_rad) % TAU


def signed_angle_rad(angle_rad: float) -> float:
    """Normalize to [-pi, pi)."""

    return ((float(angle_rad) + math.pi) % TAU) - math.pi


def raw_points_to_body_points(
    points: Iterable[LidarPoint],
    *,
    yaw_offset_deg: float = 0.0,
    clockwise: bool = False,
) -> tuple[LidarPoint, ...]:
    """Apply sensor yaw/handedness and return body-frame polar points."""

    yaw_offset = math.radians(float(yaw_offset_deg))
    out: list[LidarPoint] = []
    for point in points:
        angle = float(point.angle_rad)
        if clockwise:
            angle = -angle
        angle = signed_angle_rad(angle + yaw_offset)
        out.append(
            LidarPoint(
                angle_rad=angle,
                distance_m=float(point.distance_m),
                intensity=float(point.intensity),
            )
        )
    return tuple(out)


def scan_from_points(
    points: Iterable[LidarPoint],
    *,
    timestamp: float | None = None,
    frame_id: str = "lidar",
    bins: int | None = None,
    range_min_m: float | None = None,
    range_max_m: float | None = None,
    source: str = "serial",
) -> LidarScan:
    """Rasterize polar points into a dense scan using nearest return per bin."""

    n = max(1, int(bins if bins is not None else getattr(config, "LIDAR_BINS", 720)))
    rmin = float(range_min_m if range_min_m is not None else getattr(config, "LIDAR_RANGE_MIN_M", 0.05))
    rmax = float(range_max_m if range_max_m is not None else getattr(config, "LIDAR_RANGE_MAX_M", 4.0))
    ranges = [math.inf] * n
    intensities = [0.0] * n
    step = TAU / float(n)

    for point in points:
        dist = float(point.distance_m)
        if not math.isfinite(dist) or dist < rmin or dist > rmax:
            continue
        angle = signed_angle_rad(float(point.angle_rad))
        idx = int(round((angle + math.pi) / step)) % n
        if dist < ranges[idx]:
            ranges[idx] = dist
            intensities[idx] = float(point.intensity)

    return LidarScan(
        timestamp=time.time() if timestamp is None else float(timestamp),
        frame_id=str(frame_id),
        angle_min_rad=-math.pi,
        angle_max_rad=math.pi,
        angle_step_rad=step,
        range_min_m=rmin,
        range_max_m=rmax,
        ranges_m=tuple(ranges),
        intensities=tuple(intensities),
        source=str(source),
    )


class RollingLidarBuffer:
    """Small TTL buffer for serial packets before rasterization."""

    def __init__(self, *, ttl_s: float = 0.5) -> None:
        self.ttl_s = float(ttl_s)
        self._points: deque[tuple[float, LidarPoint]] = deque()

    def add_points(self, points: Iterable[LidarPoint], *, timestamp: float | None = None) -> None:
        ts = time.time() if timestamp is None else float(timestamp)
        for point in points:
            self._points.append((ts, point))
        self.prune(now=ts)

    def prune(self, *, now: float | None = None) -> None:
        t_now = time.time() if now is None else float(now)
        cutoff = t_now - self.ttl_s
        while self._points and self._points[0][0] < cutoff:
            self._points.popleft()

    def points(self, *, now: float | None = None) -> tuple[LidarPoint, ...]:
        self.prune(now=now)
        return tuple(point for _, point in self._points)

    def to_scan(self, **kwargs) -> LidarScan:
        return scan_from_points(self.points(), **kwargs)


def distance_in_sector(
    scan: LidarScan | None,
    *,
    center_angle_rad: float,
    half_width_rad: float,
) -> float | None:
    """Nearest valid range inside an angular sector."""

    if scan is None:
        return None
    ranges = tuple(scan.ranges_m or ())
    if not ranges:
        return None
    intensities = tuple(scan.intensities or ())
    rmin = float(scan.range_min_m)
    rmax = float(scan.range_max_m)
    center = signed_angle_rad(center_angle_rad)
    half = max(0.0, float(half_width_rad))
    best: float | None = None

    for idx, raw_dist in enumerate(ranges):
        try:
            dist = float(raw_dist)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(dist) or dist < rmin or dist > rmax:
            continue
        if intensities and idx < len(intensities):
            try:
                intensity = float(intensities[idx])
            except (TypeError, ValueError):
                intensity = 0.0
            if intensity <= 0.0 and any(float(v or 0.0) > 0.0 for v in intensities):
                continue
        angle = _scan_angle(scan, idx)
        if abs(signed_angle_rad(angle - center)) <= half:
            best = dist if best is None else min(best, dist)
    return best


def cluster_obstacles(
    scan: LidarScan | None,
    *,
    max_range_m: float | None = None,
    min_intensity: float | None = None,
    max_gap_deg: float | None = None,
    min_points: int | None = None,
    keep_single_below_m: float | None = None,
    min_forward_x_m: float | None = None,
) -> tuple[LidarObstacle, ...]:
    """Cluster nearby valid scan bins by angular continuity."""

    if scan is None:
        return ()
    ranges = tuple(scan.ranges_m or ())
    if not ranges:
        return ()
    intensities = tuple(scan.intensities or ())
    max_range = float(max_range_m if max_range_m is not None else getattr(config, "LIDAR_OBSTACLE_CLUSTER_RANGE_M", 1.2))
    min_i = float(min_intensity if min_intensity is not None else getattr(config, "LIDAR_CLUSTER_MIN_INTENSITY", 5.0))
    max_gap = math.radians(float(max_gap_deg if max_gap_deg is not None else getattr(config, "LIDAR_CLUSTER_MAX_GAP_DEG", 5.0)))
    min_pts = int(min_points if min_points is not None else getattr(config, "LIDAR_CLUSTER_MIN_POINTS", 3))
    keep_single = float(keep_single_below_m if keep_single_below_m is not None else getattr(config, "LIDAR_CLUSTER_KEEP_SINGLE_BELOW_M", 1.2))
    min_forward_x = float(min_forward_x_m if min_forward_x_m is not None else getattr(config, "LIDAR_OBSTACLE_MIN_FORWARD_X_M", 0.0))

    valid: list[tuple[float, float, float]] = []
    any_positive_intensity = bool(intensities and any(float(v or 0.0) > 0.0 for v in intensities))
    for idx, raw_dist in enumerate(ranges):
        try:
            dist = float(raw_dist)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(dist) or dist < float(scan.range_min_m) or dist > min(float(scan.range_max_m), max_range):
            continue
        angle = _scan_angle(scan, idx)
        if dist * math.cos(angle) < min_forward_x:
            continue
        intensity = 0.0
        if idx < len(intensities):
            try:
                intensity = float(intensities[idx])
            except (TypeError, ValueError):
                intensity = 0.0
        if any_positive_intensity and intensity < min_i:
            continue
        valid.append((angle, dist, intensity))

    if not valid:
        return ()

    valid.sort(key=lambda item: normalize_angle_rad(item[0]))
    groups: list[list[tuple[float, float, float]]] = [[valid[0]]]
    for prev, item in zip(valid, valid[1:]):
        gap = normalize_angle_rad(item[0]) - normalize_angle_rad(prev[0])
        if gap < 0.0:
            gap += TAU
        if gap <= max_gap:
            groups[-1].append(item)
        else:
            groups.append([item])

    if len(groups) > 1:
        first = groups[0][0]
        last = groups[-1][-1]
        wrap_gap = (normalize_angle_rad(first[0]) + TAU) - normalize_angle_rad(last[0])
        if wrap_gap <= max_gap:
            groups[0] = groups[-1] + groups[0]
            groups.pop()

    obstacles: list[LidarObstacle] = []
    for group in groups:
        if len(group) < min_pts and not any(d <= keep_single for _, d, _ in group):
            continue
        obstacles.append(_group_to_obstacle(group, timestamp=scan.timestamp))

    obstacles.sort(key=lambda obs: obs.distance_min_m)
    return tuple(obstacles)


def transform_obstacles_to_world(
    obstacles: Iterable[LidarObstacle],
    pose: PoseEstimate | None,
) -> tuple[LidarObstacle, ...]:
    if pose is None or not isinstance(pose, PoseEstimate):
        return tuple(obstacles)
    yaw = float(pose.fused_pose.yaw)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    ox = float(pose.fused_pose.x)
    oy = float(pose.fused_pose.y)
    out: list[LidarObstacle] = []
    for obs in obstacles:
        wx = ox + obs.x_m * cos_y - obs.y_m * sin_y
        wy = oy + obs.x_m * sin_y + obs.y_m * cos_y
        out.append(replace(obs, world_xy=(float(wx), float(wy))))
    return tuple(out)


def _scan_angle(scan: LidarScan, idx: int) -> float:
    return signed_angle_rad(float(scan.angle_min_rad) + float(idx) * float(scan.angle_step_rad))


def _group_to_obstacle(
    group: list[tuple[float, float, float]],
    *,
    timestamp: float,
) -> LidarObstacle:
    min_item = min(group, key=lambda item: item[1])
    min_dist = float(min_item[1])
    xs = [dist * math.cos(angle) for angle, dist, _ in group]
    ys = [dist * math.sin(angle) for angle, dist, _ in group]
    x = sum(xs) / float(len(xs))
    y = sum(ys) / float(len(ys))
    center_angle = math.atan2(y, x)
    angles = [signed_angle_rad(angle - center_angle) for angle, _, _ in group]
    width = max(angles) - min(angles) if angles else 0.0
    intensities = [max(0.0, float(i)) for _, _, i in group]
    conf = min(1.0, 0.35 + 0.08 * len(group) + 0.01 * (sum(intensities) / max(1, len(intensities))))
    return LidarObstacle(
        timestamp=float(timestamp),
        distance_min_m=min_dist,
        angle_center_rad=signed_angle_rad(center_angle),
        x_m=float(x),
        y_m=float(y),
        width_rad=max(0.0, float(width)),
        point_count=int(len(group)),
        confidence=float(conf),
        source="cluster",
    )
