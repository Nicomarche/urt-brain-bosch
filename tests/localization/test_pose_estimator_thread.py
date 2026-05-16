from __future__ import annotations

import math
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.types.perception import LaneObservation
from src.core.types.pose import Pose2D
from src.core.types.routing import RouteContext
from src.localization.pose_estimator_thread import threadPoseEstimator


class _FakeDR:
    def __init__(self, *, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)
        self.corrections: list[tuple[float, float]] = []

    def correct_lateral(self, lateral_error_m: float, path_psi: float) -> None:
        self.corrections.append((float(lateral_error_m), float(path_psi)))
        dx = float(lateral_error_m) * math.cos(float(path_psi) + math.pi / 2.0)
        dy = float(lateral_error_m) * math.sin(float(path_psi) + math.pi / 2.0)
        self.x -= dx
        self.y -= dy

    def correct_yaw(self, yaw_correction_rad: float) -> None:
        self.yaw += float(yaw_correction_rad)

    def get_state(self) -> tuple[float, float, float]:
        return self.x, self.y, self.yaw

    def reset(self, x: float, y: float, yaw: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)


class _FakeLanelet:
    lanelet_id = "lane-a"
    length_m = 2.0
    centerline = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)

    def project_arclength(self, x: float, y: float) -> tuple[float, float]:
        return max(0.0, min(2.0, float(x))), float(y)


class _FakeLaneletMap:
    def __init__(self, *, match: bool = True) -> None:
        self.match = bool(match)
        self.lanelet = _FakeLanelet()
        self.last_at_pose_args = None

    def at_pose(self, *args, **kwargs):
        self.last_at_pose_args = (args, kwargs)
        return "lane-a" if self.match else None

    def get_lanelet(self, lanelet_id):
        return self.lanelet if lanelet_id == "lane-a" else None


def _make_route_context() -> RouteContext:
    return RouteContext(
        route_active=True,
        waypoint_mode_active=False,
        matched_pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
    )


def _make_estimator(*, speed_mps: float = 0.20, dr_y: float = 0.05) -> threadPoseEstimator:
    estimator = threadPoseEstimator.__new__(threadPoseEstimator)
    estimator._dr = _FakeDR(y=dr_y)
    estimator._last_speed = float(speed_mps)
    estimator._last_camera_lateral_correction_monotonic = 0.0
    estimator._last_absolute_yaw_fix_monotonic = 0.0
    estimator._last_yaw_rad = 0.0
    estimator.tracking_state = SimpleNamespace(last_yaw_correction_deg=0.0)
    return estimator


def test_apply_lane_observation_skips_invalid_single_line_measurement() -> None:
    estimator = _make_estimator()
    lane_observation = LaneObservation(
        detected_sides=("left",),
        lateral_offset_m=None,
        direct_error_m=None,
        quality=0.65,
        measurement_mode="single_line",
        direct_error_valid=False,
        control_policy_mode="ROUTE_TRACKING",
    )

    new_x, new_y, new_yaw, correction_m, reliable = estimator._apply_lane_observation(
        _make_route_context(),
        lane_observation,
        now=10.0,
        raw_x=0.0,
        raw_y=0.05,
        raw_yaw=0.0,
    )

    assert new_x == 0.0
    assert new_y == 0.05
    assert new_yaw == 0.0
    assert correction_m == 0.0
    assert reliable is False
    assert estimator._dr.corrections == []


def test_apply_lane_observation_still_skips_valid_single_line_measurement() -> None:
    estimator = _make_estimator()
    lane_observation = LaneObservation(
        detected_sides=("right",),
        lateral_offset_m=-0.12,
        direct_error_m=-0.12,
        quality=0.65,
        measurement_mode="single_line",
        direct_error_valid=True,
        control_policy_mode="ROUTE_TRACKING",
        planner_priority_active=True,
    )

    new_x, new_y, new_yaw, correction_m, reliable = estimator._apply_lane_observation(
        _make_route_context(),
        lane_observation,
        now=10.0,
        raw_x=0.0,
        raw_y=0.05,
        raw_yaw=0.0,
    )

    assert new_x == 0.0
    assert new_y == 0.05
    assert new_yaw == 0.0
    assert correction_m == 0.0
    assert reliable is False
    assert estimator._dr.corrections == []


def test_apply_lane_observation_accepts_valid_two_line_measurement() -> None:
    estimator = _make_estimator()
    lane_observation = LaneObservation(
        detected_sides=("left", "right"),
        lateral_offset_m=0.10,
        direct_error_m=0.10,
        quality=1.0,
        measurement_mode="two_line",
        direct_error_valid=True,
        control_policy_mode="VISUAL_ASSIST",
    )

    new_x, new_y, new_yaw, correction_m, reliable = estimator._apply_lane_observation(
        _make_route_context(),
        lane_observation,
        now=10.0,
        raw_x=0.0,
        raw_y=0.05,
        raw_yaw=0.0,
    )

    assert reliable is True
    assert correction_m > 0.0
    assert new_x == pytest.approx(0.0, abs=1e-9)
    assert new_y < 0.05
    assert new_yaw == pytest.approx(0.0, abs=1e-9)
    assert len(estimator._dr.corrections) == 1


def test_apply_camera_yaw_hint_blocks_conflicting_single_line_turn_on_straight_route() -> None:
    estimator = _make_estimator(speed_mps=0.20)
    lane_observation = LaneObservation(
        detected_sides=("right",),
        quality=0.65,
        measurement_mode="single_line",
        direct_error_valid=True,
        heading_error_rad=math.radians(20.0),
        camera_yaw_hint_rad=math.radians(20.0),
        camera_yaw_hint_confidence=0.9,
    )
    route_context = RouteContext(
        route_active=True,
        matched_pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
        path_psi=0.0,
        path_heading_change_rad=0.0,
    )

    correction = estimator._apply_camera_yaw_hint(10.0, 0.0, route_context, lane_observation)

    assert correction == 0.0
    assert estimator._dr.yaw == pytest.approx(0.0, abs=1e-9)
    assert estimator.tracking_state.last_yaw_correction_deg == pytest.approx(0.0, abs=1e-9)


def test_apply_camera_yaw_hint_allows_single_line_when_route_turn_matches() -> None:
    estimator = _make_estimator(speed_mps=0.20)
    lane_observation = LaneObservation(
        detected_sides=("right",),
        quality=0.65,
        measurement_mode="single_line",
        direct_error_valid=True,
        heading_error_rad=math.radians(12.0),
        camera_yaw_hint_rad=math.radians(14.0),
        camera_yaw_hint_confidence=0.9,
    )
    route_context = RouteContext(
        route_active=True,
        matched_pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
        path_psi=math.radians(12.0),
        path_heading_change_rad=math.radians(18.0),
    )

    correction = estimator._apply_camera_yaw_hint(10.0, 0.0, route_context, lane_observation)

    assert correction > 0.0
    assert estimator._dr.yaw > 0.0


def test_apply_lane_observation_accepts_valid_two_line_in_waypoint_mode() -> None:
    estimator = _make_estimator(speed_mps=0.10)
    lane_observation = LaneObservation(
        detected_sides=("left", "right"),
        lateral_offset_m=-0.20,
        direct_error_m=-0.20,
        quality=1.0,
        measurement_mode="two_line",
        direct_error_valid=True,
        control_policy_mode="ROUTE_TRACKING",
        planner_priority_active=True,
    )
    route_context = RouteContext(
        route_active=True,
        waypoint_mode_active=True,
        matched_pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
    )

    new_x, new_y, new_yaw, correction_m, reliable = estimator._apply_lane_observation(
        route_context,
        lane_observation,
        now=10.0,
        raw_x=0.0,
        raw_y=0.05,
        raw_yaw=0.0,
    )

    assert reliable is True
    assert correction_m < 0.0
    assert new_x == pytest.approx(0.0, abs=1e-9)
    assert new_y > 0.05
    assert new_yaw == pytest.approx(0.0, abs=1e-9)
    assert len(estimator._dr.corrections) == 1


def test_apply_lane_observation_accepts_valid_two_line_at_low_route_speed() -> None:
    estimator = _make_estimator(speed_mps=0.04)
    lane_observation = LaneObservation(
        detected_sides=("left", "right"),
        lateral_offset_m=-0.20,
        direct_error_m=-0.20,
        quality=1.0,
        measurement_mode="two_line",
        direct_error_valid=True,
        control_policy_mode="VISUAL_ASSIST",
    )
    route_context = RouteContext(
        route_active=True,
        waypoint_mode_active=True,
        matched_pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
    )

    new_x, new_y, new_yaw, correction_m, reliable = estimator._apply_lane_observation(
        route_context,
        lane_observation,
        now=10.0,
        raw_x=0.0,
        raw_y=0.05,
        raw_yaw=0.0,
    )

    assert reliable is True
    assert correction_m < 0.0
    assert new_x == pytest.approx(0.0, abs=1e-9)
    assert new_y > 0.05
    assert new_yaw == pytest.approx(0.0, abs=1e-9)
    assert len(estimator._dr.corrections) == 1


def test_apply_localisation_fix_manual_pose_teleports_sim(monkeypatch) -> None:
    sent_messages: list[dict] = []

    class _FakeSender:
        def __init__(self, _queues_list, _message) -> None:
            pass

        def send(self, value) -> None:
            sent_messages.append(dict(value))

    import config as cfg

    monkeypatch.setattr(cfg, "MOTOR_OUTPUT", "zmq", raising=False)
    monkeypatch.setattr(cfg, "GZ_SPAWN_Z", 0.123, raising=False)
    monkeypatch.setattr(
        "src.localization.pose_estimator_thread.messageHandlerSender",
        _FakeSender,
    )

    estimator = threadPoseEstimator.__new__(threadPoseEstimator)
    estimator._dr = _FakeDR(x=0.0, y=0.0, yaw=0.0)
    estimator._graph = SimpleNamespace(
        localisation_to_world_pose=lambda payload, default_yaw=0.0: (
            float(payload["world_x"]),
            float(payload["world_y"]),
            math.radians(float(payload["yaw_deg"])),
        ),
        resolve_node_id=lambda _value: None,
    )
    estimator._localisation_fix_sub = SimpleNamespace(
        receive=lambda: {
            "world_x": 1.5,
            "world_y": -2.0,
            "yaw_deg": 90.0,
            "meta": {"manual": True, "source": "manual_dashboard"},
        }
    )
    estimator._last_absolute_yaw_fix_monotonic = 0.0
    estimator._last_absolute_yaw_fix_source = None
    estimator._last_yaw_rad = 0.0
    estimator._yaw_ekf_p = 0.0
    estimator.tracking_state = SimpleNamespace(
        set_lane_measurement_state=lambda *_args, **_kwargs: None
    )
    estimator.queuesList = {"General": object()}

    applied, info = estimator._apply_localisation_fix(current_yaw=0.0)

    assert applied is True
    assert info is not None
    assert info["source"] == "manual_dashboard"
    assert estimator._dr.get_state() == pytest.approx(
        (1.5, -2.0, math.pi / 2.0),
        abs=1e-9,
    )
    assert sent_messages == [
        {
            "world_x": 1.5,
            "world_y": -2.0,
            "yaw_rad": math.pi / 2.0,
            "z": 0.123,
        }
    ]


def test_auto_entry_starts_gps_window_and_sends_odometer_reset() -> None:
    sent: list[str] = []
    emptied = {"value": False}
    estimator = threadPoseEstimator.__new__(threadPoseEstimator)
    estimator._gps_fix_history = deque([{"x": 9.0, "y": 9.0}], maxlen=10)
    estimator._localisation_fix_sub = SimpleNamespace(empty=lambda: emptied.__setitem__("value", True))
    estimator._odo_reset_sender = SimpleNamespace(send=lambda value: sent.append(value))
    estimator.logging = None

    estimator._begin_auto_gps_entry()

    assert sent == ["1"]
    assert emptied["value"] is True
    assert len(estimator._gps_fix_history) == 0
    assert estimator._auto_gps_pending is True


def _make_auto_gps_estimator(*, lanelet_match: bool = True) -> threadPoseEstimator:
    estimator = threadPoseEstimator.__new__(threadPoseEstimator)
    estimator._dr = _FakeDR(x=0.0, y=0.30, yaw=0.0)
    estimator._graph = SimpleNamespace(lanelet_map=_FakeLaneletMap(match=lanelet_match))
    estimator._gps_fix_history = deque(maxlen=10)
    estimator._auto_gps_pending = True
    estimator._auto_gps_deadline_monotonic = 20.0
    estimator._auto_gps_last_mode = None
    estimator._auto_gps_last_source = None
    estimator._auto_gps_last_error_m = 0.0
    estimator._auto_gps_last_mode_monotonic = 0.0
    estimator._last_yaw_rad = 0.0
    estimator._yaw_ekf_p = 0.0
    estimator._last_absolute_yaw_fix_monotonic = 0.0
    estimator._last_absolute_yaw_fix_source = None
    estimator._pending_yaw_offset_target_rad = None
    estimator._last_raw_imu = None
    estimator._last_imu_t = None
    estimator._yaw_offset_calibrated = False
    estimator.tracking_state = SimpleNamespace(
        set_lane_measurement_state=lambda *_args, **_kwargs: None
    )
    estimator._send_sim_relocalize_pose = lambda *_args, **_kwargs: None
    return estimator


def test_auto_gps_relocalization_averages_three_reliable_fixes_and_snaps_to_lanelet() -> None:
    estimator = _make_auto_gps_estimator()
    now = 10.0
    for x, y in ((0.98, 0.02), (1.00, 0.01), (1.02, -0.01)):
        estimator._remember_auto_gps_fix(
            {"world_x": x, "world_y": y, "timestamp": time.time()},
            {"source": "gps_localisation", "gps_frame_in_bounds": True},
            now,
        )

    applied, info = estimator._try_apply_auto_gps_relocalization(current_yaw=0.0, now=now)

    assert applied is True
    assert info is not None
    assert info["mode"] == "auto_gps_entry"
    assert estimator._auto_gps_pending is False
    assert estimator._dr.get_state() == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    lanelet_map = estimator._graph.lanelet_map
    assert lanelet_map.last_at_pose_args[1]["yaw_rad"] == pytest.approx(0.0)


def test_auto_gps_relocalization_rejects_dispersed_fixes_and_keeps_pose() -> None:
    estimator = _make_auto_gps_estimator()
    now = 10.0
    estimator._auto_gps_deadline_monotonic = 9.0
    for x, y in ((0.0, 0.0), (0.8, 0.0), (1.6, 0.0)):
        estimator._remember_auto_gps_fix(
            {"world_x": x, "world_y": y, "timestamp": time.time()},
            {"source": "gps_localisation", "gps_frame_in_bounds": True},
            now,
        )

    applied, info = estimator._try_apply_auto_gps_relocalization(current_yaw=0.0, now=now)

    assert applied is False
    assert info is None
    assert estimator._auto_gps_pending is False
    assert estimator._auto_gps_last_mode == "auto_gps_unavailable"
    assert estimator._dr.get_state() == pytest.approx((0.0, 0.30, 0.0), abs=1e-9)
