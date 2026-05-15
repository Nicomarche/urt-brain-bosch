from __future__ import annotations

import math

import pytest

from src.core.types.lidar import LidarScan
from src.core.types.pose import Pose2D, PoseEstimate
from src.hardware.camera.threads.threadLocalPerception import threadLocalPerception


class _RecordingSender:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, message: str) -> None:
        self.messages.append(message)


class _Buffer:
    def __init__(self, value=None) -> None:
        self.value = value
        self.writes = []

    def read_latest(self):
        return self.value

    def write(self, value, *, timestamp=None):
        self.value = value
        self.writes.append((value, timestamp))
        return value


def _scan_at_angle(angle_deg: float, distance_m: float) -> LidarScan:
    ranges = [math.inf] * 720
    intensities = [20.0] * 720
    step = (2.0 * math.pi) / 720.0
    idx = int(round((math.radians(angle_deg) + math.pi) / step)) % 720
    ranges[idx] = distance_m
    return LidarScan(
        timestamp=100.0,
        frame_id="lidar",
        angle_min_rad=-math.pi,
        angle_max_rad=math.pi,
        angle_step_rad=step,
        range_min_m=0.05,
        range_max_m=4.0,
        ranges_m=tuple(ranges),
        intensities=tuple(intensities),
        source="test",
    )


def _scan_at_front(distance_m: float) -> LidarScan:
    return _scan_at_angle(0.0, distance_m)


def _local_thread_for_distance(distance_m: float | None) -> threadLocalPerception:
    thread = threadLocalPerception.__new__(threadLocalPerception)
    thread.enable_sign_detection = True
    thread.sign_min_confidence = 0.5
    thread.sign_min_box_area = 0.01
    thread.sign_min_box_area_per_sign = {}
    thread.detection_count = 0
    thread.last_sign_name = ""
    thread.enable_actions = False
    thread.is_sign_actions_active = True
    thread._current_mode = "auto"
    thread.signDetectedSender = _RecordingSender()
    thread.detectedObjectsSender = _RecordingSender()
    thread.sign_hints_buffer = _Buffer()
    thread.detection_buffer = _Buffer()
    thread.lidar_scan_buffer = _Buffer(_scan_at_front(distance_m)) if distance_m is not None else _Buffer(None)
    thread.pose_estimate_buffer = _Buffer(
        PoseEstimate(fused_pose=Pose2D(x=1.0, y=2.0, yaw=0.0))
    )
    return thread


def test_walk_area_crossed_does_not_request_parking_mode() -> None:
    thread = threadLocalPerception.__new__(threadLocalPerception)
    thread._current_mode = "auto"
    thread._walk_area_stop_duration = 3.0
    thread._walk_area_min_box_area = 0.04
    thread._walk_area_active = False
    thread._walk_area_no_obstacle_since = None
    thread._walk_area_cooldown = 10.0
    thread._walk_area_last_cleared = 0.0
    thread.stateChangeSender = _RecordingSender()

    detections = [{"class": "walk_area", "box": [0.0, 0.0, 0.5, 0.5]}]

    thread._handle_walk_area(detections, now=20.0)
    assert thread._walk_area_active is True
    assert thread.stateChangeSender.messages == []

    thread._handle_walk_area(detections, now=23.5)

    assert thread._walk_area_active is False
    assert thread._walk_area_no_obstacle_since is None
    assert thread._walk_area_last_cleared == 23.5
    assert thread.stateChangeSender.messages == []


def test_sign_distance_source_prefers_lidar_when_sector_has_point() -> None:
    thread = _local_thread_for_distance(0.70)
    thread._publish_sign(
        [{"class": "stop", "confidence": 0.9, "box": [0.40, 0.45, 0.60, 0.55]}],
        now=10.0,
        img_shape=(480, 640),
    )
    msg = thread.signDetectedSender.messages[-1]
    assert msg["distance_source"] == "lidar"
    assert msg["distance_cm"] == pytest.approx(70.0)
    assert thread.sign_hints_buffer.value[0]["distance_source"] == "lidar"


def test_sign_distance_falls_back_to_camera() -> None:
    thread = _local_thread_for_distance(None)
    thread._publish_sign(
        [{"class": "stop", "confidence": 0.9, "box": [0.40, 0.45, 0.60, 0.55]}],
        now=10.0,
        img_shape=(480, 640),
    )
    msg = thread.signDetectedSender.messages[-1]
    assert msg["distance_source"] == "camera"
    assert msg["distance_cm"] is not None


def test_detected_object_gets_world_position_from_lidar_and_pose() -> None:
    thread = _local_thread_for_distance(0.50)
    thread._publish_detected_objects(
        [{"class": "obstacle", "confidence": 0.7, "box": [0.40, 0.45, 0.60, 0.55]}],
        now=20.0,
        img_shape=(480, 640),
    )
    detections = thread.detection_buffer.value
    assert len(detections) == 1
    assert detections[0].position_world_xy == pytest.approx((1.5, 2.0))
    msg = thread.detectedObjectsSender.messages[-1]
    assert msg["objects"][0]["distance_source"] == "lidar"


def test_detected_object_uses_broader_lidar_sector_for_camera_lidar_offset() -> None:
    thread = _local_thread_for_distance(None)
    thread.lidar_scan_buffer = _Buffer(_scan_at_angle(22.0, 0.37))

    thread._publish_detected_objects(
        [{"class": "obstacle", "confidence": 0.86, "box": [0.35, 0.35, 0.72, 0.44]}],
        now=30.0,
        img_shape=(480, 640),
    )

    msg = thread.detectedObjectsSender.messages[-1]
    assert msg["objects"][0]["distance_source"] == "lidar"
    assert msg["objects"][0]["distance_m"] == pytest.approx(0.37)
