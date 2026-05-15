from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LidarPoint:
    """Single LiDAR return in the vehicle body frame.

    Convention: x forward, y left, angle_rad=0 at the front, positive to the left.
    """

    angle_rad: float = 0.0
    distance_m: float = 0.0
    intensity: float = 0.0


@dataclass(frozen=True)
class LidarScan:
    """Dense polar scan shared by physical LD19 and Gazebo ZMQ backends."""

    timestamp: float = 0.0
    frame_id: str = "lidar"
    angle_min_rad: float = 0.0
    angle_max_rad: float = 0.0
    angle_step_rad: float = 0.0
    range_min_m: float = 0.05
    range_max_m: float = 4.0
    ranges_m: tuple[float, ...] = field(default_factory=tuple)
    intensities: tuple[float, ...] = field(default_factory=tuple)
    source: str = "unknown"


@dataclass(frozen=True)
class LidarObstacle:
    """Cluster or compact obstacle extracted from a LiDAR scan."""

    timestamp: float = 0.0
    distance_min_m: float = 0.0
    angle_center_rad: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0
    width_rad: float = 0.0
    point_count: int = 0
    confidence: float = 0.0
    world_xy: tuple[float, float] | None = None
    source: str = "cluster"
    class_name: str = "obstacle"


@dataclass(frozen=True)
class LidarHealth:
    """Snapshot for dashboard/debug; control uses typed buffers directly."""

    timestamp: float = 0.0
    enabled: bool = False
    backend: str = "none"
    status: str = "disabled"
    detail: str = ""
    scan_hz: float = 0.0
    points: int = 0
    obstacles: int = 0
    age_s: float | None = None
    crc_errors: int = 0
    resyncs: int = 0
    source: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)
