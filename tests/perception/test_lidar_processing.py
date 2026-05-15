from __future__ import annotations

import math

import pytest

from src.core.types.lidar import LidarPoint
from src.core.types.pose import Pose2D, PoseEstimate
from src.perception.lidar.processing import (
    cluster_obstacles,
    distance_in_sector,
    scan_from_points,
    transform_obstacles_to_world,
)
from src.perception.lidar.thread_lidar import _scan_from_zmq_payload


def test_distance_in_sector_uses_nearest_valid() -> None:
    scan = scan_from_points(
        [
            LidarPoint(angle_rad=0.0, distance_m=0.80, intensity=20.0),
            LidarPoint(angle_rad=math.radians(2.0), distance_m=0.50, intensity=20.0),
        ],
        bins=720,
        range_min_m=0.05,
        range_max_m=4.0,
    )
    assert distance_in_sector(scan, center_angle_rad=0.0, half_width_rad=math.radians(5.0)) == pytest.approx(0.50)


def test_distance_in_sector_handles_zero_wrap() -> None:
    scan = scan_from_points(
        [LidarPoint(angle_rad=math.radians(179.0), distance_m=0.60, intensity=20.0)],
        bins=720,
        range_min_m=0.05,
        range_max_m=4.0,
    )
    assert distance_in_sector(scan, center_angle_rad=math.radians(-181.0), half_width_rad=math.radians(3.0)) == pytest.approx(0.60)


def test_cluster_obstacles_groups_front_points() -> None:
    points = [
        LidarPoint(angle_rad=math.radians(a), distance_m=0.55, intensity=20.0)
        for a in (-2.0, 0.0, 2.0)
    ]
    scan = scan_from_points(points, bins=720, range_min_m=0.05, range_max_m=4.0)
    obstacles = cluster_obstacles(scan, min_points=3, max_gap_deg=5.0)
    assert len(obstacles) == 1
    assert obstacles[0].distance_min_m == pytest.approx(0.55)
    assert abs(obstacles[0].angle_center_rad) < math.radians(1.0)


def test_cluster_obstacles_ignores_rear_points() -> None:
    points = [
        LidarPoint(angle_rad=math.radians(178.0), distance_m=0.08, intensity=20.0),
        LidarPoint(angle_rad=math.radians(180.0), distance_m=0.08, intensity=20.0),
        LidarPoint(angle_rad=math.radians(-178.0), distance_m=0.08, intensity=20.0),
        LidarPoint(angle_rad=math.radians(-3.0), distance_m=0.45, intensity=20.0),
        LidarPoint(angle_rad=math.radians(0.0), distance_m=0.43, intensity=20.0),
        LidarPoint(angle_rad=math.radians(3.0), distance_m=0.45, intensity=20.0),
    ]
    scan = scan_from_points(points, bins=720, range_min_m=0.05, range_max_m=4.0)

    obstacles = cluster_obstacles(scan, min_points=3)

    assert len(obstacles) == 1
    assert obstacles[0].x_m > 0.0
    assert abs(math.degrees(obstacles[0].angle_center_rad)) < 5.0


def test_cluster_keep_single_below_threshold() -> None:
    scan = scan_from_points(
        [LidarPoint(angle_rad=0.0, distance_m=0.45, intensity=20.0)],
        bins=720,
        range_min_m=0.05,
        range_max_m=4.0,
    )
    obstacles = cluster_obstacles(scan, min_points=3, keep_single_below_m=1.20)
    assert len(obstacles) == 1
    assert obstacles[0].point_count == 1


def test_transform_obstacles_to_world() -> None:
    scan = scan_from_points(
        [LidarPoint(angle_rad=0.0, distance_m=1.0, intensity=20.0)],
        bins=720,
        range_min_m=0.05,
        range_max_m=4.0,
    )
    obs = cluster_obstacles(scan, min_points=1)[0]
    pose = PoseEstimate(fused_pose=Pose2D(x=2.0, y=3.0, yaw=math.pi / 2.0))
    world = transform_obstacles_to_world((obs,), pose)[0]
    assert world.world_xy == pytest.approx((2.0, 4.0))


def test_zmq_payload_normalizes_sim_timestamp_and_zero_intensities() -> None:
    scan = _scan_from_zmq_payload(
        {
            "timestamp": 12.5,
            "angle_min": -math.pi,
            "angle_max": math.pi,
            "angle_step": (2.0 * math.pi) / 720.0,
            "range_min": 0.05,
            "range_max": 4.0,
            "ranges": [1.0] * 720,
            "intensities": [0.0] * 720,
            "frame_id": "lidar",
        }
    )
    assert scan.timestamp > 1_000_000_000.0
    assert all(v == pytest.approx(20.0) for v in scan.intensities)
