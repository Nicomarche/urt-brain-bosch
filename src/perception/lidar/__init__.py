"""2D LiDAR support shared by physical LD19 and Gazebo ZMQ backends."""

from src.perception.lidar.processing import (
    RollingLidarBuffer,
    cluster_obstacles,
    distance_in_sector,
    normalize_angle_rad,
    raw_points_to_body_points,
    scan_from_points,
    signed_angle_rad,
    transform_obstacles_to_world,
)
from src.perception.lidar.protocol import LD19Packet, parse_ld19_packet

__all__ = [
    "LD19Packet",
    "parse_ld19_packet",
    "normalize_angle_rad",
    "signed_angle_rad",
    "raw_points_to_body_points",
    "scan_from_points",
    "RollingLidarBuffer",
    "distance_in_sector",
    "cluster_obstacles",
    "transform_obstacles_to_world",
]
