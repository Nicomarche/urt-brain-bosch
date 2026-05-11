from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from src.core.types.perception import LaneObservation
from src.core.types.pose import Pose2D, VisualLaneMatch
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


class _FakeSub:
    def __init__(self, values) -> None:
        self._values = list(values)

    def receive(self):
        if not self._values:
            return None
        return self._values.pop(0)


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
    estimator._last_visual_map_match_correction_monotonic = 0.0
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


def test_apply_visual_lane_match_update_applies_small_gated_correction() -> None:
    estimator = _make_estimator(dr_y=0.10)
    route_context = _make_route_context()
    match = VisualLaneMatch(
        lanelet_id="n1->n2",
        lateral_error_m=0.12,
        yaw_error_rad=math.radians(8.0),
        score=0.2,
        confidence=0.8,
        accepted=True,
    )

    new_x, new_y, new_yaw, lateral_m, yaw_rad, available = estimator._apply_visual_lane_match_update(
        route_context,
        match,
        now=10.0,
    )

    assert available is True
    assert lateral_m == pytest.approx(0.008, abs=1e-9)
    assert yaw_rad == pytest.approx(math.radians(0.5), abs=1e-9)
    assert new_x == pytest.approx(0.0, abs=1e-9)
    assert new_y < 0.10
    assert new_yaw > 0.0
    assert len(estimator._dr.corrections) == 1
    assert estimator._dr.corrections[0][0] == pytest.approx(0.008, abs=1e-9)
    assert estimator._dr.corrections[0][1] == pytest.approx(0.0, abs=1e-9)


def test_apply_visual_lane_match_update_ignores_rejected_match() -> None:
    estimator = _make_estimator(dr_y=0.10)
    match = VisualLaneMatch(
        lanelet_id="n1->n2",
        lateral_error_m=0.12,
        yaw_error_rad=math.radians(8.0),
        score=0.9,
        confidence=0.1,
        accepted=False,
    )

    new_x, new_y, new_yaw, lateral_m, yaw_rad, available = estimator._apply_visual_lane_match_update(
        _make_route_context(),
        match,
        now=10.0,
    )

    assert available is False
    assert lateral_m == 0.0
    assert yaw_rad == 0.0
    assert new_x == pytest.approx(0.0, abs=1e-9)
    assert new_y == pytest.approx(0.10, abs=1e-9)
    assert new_yaw == pytest.approx(0.0, abs=1e-9)
    assert estimator._dr.corrections == []


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


def _make_localisation_estimator(payloads, *, dr_x: float = 1.0, dr_y: float = 1.0):
    estimator = threadPoseEstimator.__new__(threadPoseEstimator)
    estimator._dr = _FakeDR(x=dr_x, y=dr_y, yaw=0.0)
    estimator._graph = SimpleNamespace(
        localisation_to_world_pose=lambda payload, default_yaw=0.0: (
            float(payload["world_x"]),
            float(payload["world_y"]),
            float(payload.get("yaw_rad", default_yaw)),
        ),
        resolve_node_id=lambda _value: None,
    )
    estimator._localisation_fix_sub = _FakeSub(payloads)
    estimator._last_absolute_yaw_fix_monotonic = 0.0
    estimator._last_absolute_yaw_fix_source = None
    estimator._last_yaw_rad = 0.0
    estimator._yaw_ekf_p = 0.0
    estimator._gps_fix_samples = []
    estimator._last_accepted_gps_pose = None
    estimator._last_gps_fix_quality = 0.0
    estimator.tracking_state = SimpleNamespace(
        set_lane_measurement_state=lambda *_args, **_kwargs: None
    )
    estimator.queuesList = {"General": object()}
    return estimator


def test_apply_localisation_fix_ignores_gps_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr("src.localization.relocalization_thread._USE_GPS", False)
    estimator = _make_localisation_estimator(
        [
            {
                "world_x": 1.02,
                "world_y": 1.01,
                "meta": {"source": "gps_localisation", "manual": False},
            }
        ]
    )

    applied, info = estimator._apply_localisation_fix(current_yaw=0.0)

    assert applied is False
    assert info is None
    assert estimator._dr.get_state() == pytest.approx((1.0, 1.0, 0.0), abs=1e-9)


def test_apply_localisation_fix_accepts_validated_gps_after_samples(monkeypatch) -> None:
    monkeypatch.setattr("src.localization.relocalization_thread._USE_GPS", True)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_MIN_SAMPLES", 3)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_SAMPLE_WINDOW", 5)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_OUTLIER_DISTANCE_M", 0.20)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_MAX_EXPECTED_ERROR_M", 0.75)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_MAX_JUMP_M", 1.00)
    monkeypatch.setattr("src.localization.pose_estimator_thread._LOCALIZATION_GPS_AUTHORITY", "hard")
    estimator = _make_localisation_estimator(
        [
            {"world_x": 1.02, "world_y": 1.00, "meta": {"source": "gps_localisation"}},
            {"world_x": 1.04, "world_y": 1.01, "meta": {"source": "gps_localisation"}},
            {"world_x": 1.03, "world_y": 0.99, "meta": {"source": "gps_localisation"}},
        ]
    )

    applied_1, _ = estimator._apply_localisation_fix(current_yaw=0.0)
    applied_2, _ = estimator._apply_localisation_fix(current_yaw=0.0)
    applied_3, info = estimator._apply_localisation_fix(current_yaw=0.0)

    assert applied_1 is False
    assert applied_2 is False
    assert applied_3 is True
    assert info is not None
    assert info["source"] == "gps_localisation"
    assert info["hard_reset"] is True
    x, y, yaw = estimator._dr.get_state()
    assert x == pytest.approx((1.02 + 1.04 + 1.03) / 3.0, abs=1e-9)
    assert y == pytest.approx((1.00 + 1.01 + 0.99) / 3.0, abs=1e-9)
    assert yaw == pytest.approx(0.0, abs=1e-9)


def test_apply_localisation_fix_softly_nudges_toward_validated_gps(monkeypatch) -> None:
    monkeypatch.setattr("src.localization.relocalization_thread._USE_GPS", True)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_MIN_SAMPLES", 3)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_SAMPLE_WINDOW", 5)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_OUTLIER_DISTANCE_M", 0.20)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_MAX_EXPECTED_ERROR_M", 0.75)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_MAX_JUMP_M", 1.00)
    monkeypatch.setattr("src.localization.pose_estimator_thread._LOCALIZATION_GPS_AUTHORITY", "init_recovery_soft")
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_SOFT_GAIN", 1.0)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_SOFT_MAX_STEP_M", 0.02)
    estimator = _make_localisation_estimator(
        [
            {"world_x": 1.02, "world_y": 1.00, "meta": {"source": "gps_localisation"}},
            {"world_x": 1.04, "world_y": 1.01, "meta": {"source": "gps_localisation"}},
            {"world_x": 1.03, "world_y": 0.99, "meta": {"source": "gps_localisation"}},
        ]
    )

    applied_1, _ = estimator._apply_localisation_fix(current_yaw=0.0)
    applied_2, _ = estimator._apply_localisation_fix(current_yaw=0.0)
    applied_3, info = estimator._apply_localisation_fix(current_yaw=0.0)

    assert applied_1 is False
    assert applied_2 is False
    assert applied_3 is True
    assert info is not None
    assert info["mode"] == "gps_soft"
    assert info["hard_reset"] is False
    x, y, yaw = estimator._dr.get_state()
    assert x == pytest.approx(1.02, abs=1e-9)
    assert y == pytest.approx(1.0, abs=1e-9)
    assert yaw == pytest.approx(0.0, abs=1e-9)


def test_apply_localisation_fix_does_not_apply_lateral_gps_when_visual_lane_is_reliable(
    monkeypatch,
) -> None:
    monkeypatch.setattr("src.localization.relocalization_thread._USE_GPS", True)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_MIN_SAMPLES", 1)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_SAMPLE_WINDOW", 1)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_OUTLIER_DISTANCE_M", 0.50)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_MAX_EXPECTED_ERROR_M", 0.75)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_MAX_JUMP_M", 1.00)
    monkeypatch.setattr("src.localization.pose_estimator_thread._LOCALIZATION_GPS_AUTHORITY", "init_recovery_soft")
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_SOFT_GAIN", 1.0)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_SOFT_MAX_STEP_M", 0.20)
    monkeypatch.setattr("src.localization.pose_estimator_thread._GPS_VISUAL_LATERAL_BLOCK_M", 0.05)
    estimator = _make_localisation_estimator(
        [
            {"world_x": 1.0, "world_y": 1.18, "meta": {"source": "gps_localisation"}},
        ]
    )
    lane_observation = LaneObservation(
        detected_sides=("left", "right"),
        lateral_offset_m=0.0,
        direct_error_m=0.0,
        quality=0.95,
        measurement_mode="two_line",
        direct_error_valid=True,
        center_waypoints_body=tuple((0.05 * i, 0.0, 0.0) for i in range(12)),
    )

    applied, info = estimator._apply_localisation_fix(
        current_yaw=0.0,
        route_context=_make_route_context(),
        lane_observation=lane_observation,
    )

    assert applied is True
    assert info is not None
    assert info["visual_lateral_blocked"] is True
    assert info["hard_reset"] is False
    assert estimator._dr.get_state() == pytest.approx((1.0, 1.0, 0.0), abs=1e-9)


def test_apply_localisation_fix_rejects_zero_gps(monkeypatch) -> None:
    monkeypatch.setattr("src.localization.relocalization_thread._USE_GPS", True)
    estimator = _make_localisation_estimator(
        [
            {"world_x": 0.0, "world_y": 0.0, "meta": {"source": "gps_localisation"}},
        ]
    )

    applied, info = estimator._apply_localisation_fix(current_yaw=0.0)

    assert applied is False
    assert info is None
    assert estimator._dr.get_state() == pytest.approx((1.0, 1.0, 0.0), abs=1e-9)
