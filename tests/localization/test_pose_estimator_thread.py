from __future__ import annotations

import math
from types import SimpleNamespace

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


# ──────────────────────────────────────────────────────────────────────────
#   _apply_corridor_yaw_reset (modo sin GPS: cross-check mapa × visión)
# ──────────────────────────────────────────────────────────────────────────


def _make_corridor_yaw_estimator(
    *,
    speed_mps: float = 0.30,
    dr_yaw: float = 0.0,
    last_gps_t: float = 0.0,
    last_corridor_t: float = 0.0,
) -> threadPoseEstimator:
    """Estimator con TODOS los atributos que usa `_apply_corridor_yaw_reset`."""
    estimator = threadPoseEstimator.__new__(threadPoseEstimator)
    estimator._dr = _FakeDR(yaw=dr_yaw)
    estimator._last_speed = float(speed_mps)
    estimator._last_yaw_rad = float(dr_yaw)
    estimator._last_absolute_yaw_fix_monotonic = float(last_gps_t)
    estimator._last_corridor_yaw_reset_monotonic = float(last_corridor_t)
    estimator._yaw_ekf_p = 0.5
    estimator.tracking_state = SimpleNamespace(last_yaw_correction_deg=0.0)
    return estimator


def _aligned_two_line_observation(
    *,
    heading_error_rad: float = 0.0,
    quality: float = 0.9,
) -> LaneObservation:
    return LaneObservation(
        detected_sides=("left", "right"),
        heading_error_rad=float(heading_error_rad),
        quality=float(quality),
        measurement_mode="two_line",
        direct_error_valid=True,
        direct_error_m=0.0,
        control_policy_mode="ROUTE_TRACKING",
        planner_priority_active=True,
    )


def _route_with_path_psi(path_psi: float, *, map_match_error_m: float = 0.05) -> RouteContext:
    return RouteContext(
        route_active=True,
        path_psi=float(path_psi),
        map_match_error_m=float(map_match_error_m),
        matched_pose=Pose2D(x=0.0, y=0.0, yaw=float(path_psi)),
    )


def test_corridor_yaw_reset_applies_when_all_conditions_met(monkeypatch) -> None:
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_ENABLED", True)
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_MIN_GPS_AGE_S", 0.0)
    # Drift acumulado: el DR cree que mira a 0.10 rad, la visión + mapa
    # dicen que en realidad el yaw real es 0.0 (path_psi=0, heading_error=0).
    estimator = _make_corridor_yaw_estimator(dr_yaw=0.10)
    route = _route_with_path_psi(0.0)
    lane = _aligned_two_line_observation(heading_error_rad=0.0, quality=0.9)

    delta, info = estimator._apply_corridor_yaw_reset(
        now=10.0, raw_yaw=0.10, route_context=route, lane_observation=lane
    )

    assert info is not None and info["applied"] is True
    # Reset target = path_psi + heading_error = 0; raw_yaw = 0.10 → delta = -0.10
    assert delta == pytest.approx(-0.10, abs=1e-6)
    # El DR aplicó la corrección.
    assert estimator._dr.yaw == pytest.approx(0.0, abs=1e-6)


def test_corridor_yaw_reset_target_uses_heading_error(monkeypatch) -> None:
    """target_yaw = path_psi + heading_error_rad. Verificar la suma."""
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_ENABLED", True)
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_MIN_GPS_AGE_S", 0.0)
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_MAX_HEADING_ERROR_DEG", 15.0)
    # El target será 35° = path_psi(30°) + heading_error(5°), arrancando desde
    # raw_yaw=0° → delta = 35°. Subimos el safety guard para no chocar con él
    # en este test (queremos validar el cálculo, no el guard).
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_MAX_DELTA_DEG", 60.0)
    estimator = _make_corridor_yaw_estimator(dr_yaw=0.0)
    route = _route_with_path_psi(math.radians(30.0))  # corredor apunta a 30°
    lane = _aligned_two_line_observation(heading_error_rad=math.radians(5.0))

    delta, info = estimator._apply_corridor_yaw_reset(
        now=10.0, raw_yaw=0.0, route_context=route, lane_observation=lane
    )

    assert info is not None and info["applied"] is True
    expected_target = math.radians(35.0)
    assert info["target_yaw_rad"] == pytest.approx(expected_target, abs=1e-6)
    assert delta == pytest.approx(expected_target, abs=1e-6)


def test_corridor_yaw_reset_skipped_with_fresh_gps(monkeypatch) -> None:
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_ENABLED", True)
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_MIN_GPS_AGE_S", 2.0)
    # GPS fresco hace 0.5s → ventana de gracia activa, no aplicar.
    estimator = _make_corridor_yaw_estimator(dr_yaw=0.10, last_gps_t=9.5)
    route = _route_with_path_psi(0.0)
    lane = _aligned_two_line_observation()

    delta, info = estimator._apply_corridor_yaw_reset(
        now=10.0, raw_yaw=0.10, route_context=route, lane_observation=lane
    )

    assert delta == 0.0
    assert info is None
    assert estimator._dr.yaw == pytest.approx(0.10, abs=1e-9)  # no cambió


def test_corridor_yaw_reset_skipped_with_single_line(monkeypatch) -> None:
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_ENABLED", True)
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_MIN_GPS_AGE_S", 0.0)
    estimator = _make_corridor_yaw_estimator(dr_yaw=0.10)
    route = _route_with_path_psi(0.0)
    lane = LaneObservation(
        detected_sides=("left",),
        heading_error_rad=0.0,
        quality=0.9,
        measurement_mode="single_line",
        direct_error_valid=True,
        control_policy_mode="ROUTE_TRACKING",
    )

    delta, info = estimator._apply_corridor_yaw_reset(
        now=10.0, raw_yaw=0.10, route_context=route, lane_observation=lane
    )

    assert delta == 0.0
    assert info is None


def test_corridor_yaw_reset_skipped_with_low_quality(monkeypatch) -> None:
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_ENABLED", True)
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_MIN_GPS_AGE_S", 0.0)
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_MIN_QUALITY", 0.75)
    estimator = _make_corridor_yaw_estimator(dr_yaw=0.10)
    route = _route_with_path_psi(0.0)
    lane = _aligned_two_line_observation(quality=0.50)

    delta, info = estimator._apply_corridor_yaw_reset(
        now=10.0, raw_yaw=0.10, route_context=route, lane_observation=lane
    )

    assert delta == 0.0
    assert info is None


def test_corridor_yaw_reset_skipped_with_high_map_match_error(monkeypatch) -> None:
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_ENABLED", True)
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_MIN_GPS_AGE_S", 0.0)
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_MAX_MAP_ERROR_M", 0.15)
    estimator = _make_corridor_yaw_estimator(dr_yaw=0.10)
    route = _route_with_path_psi(0.0, map_match_error_m=0.50)  # mal matcheado
    lane = _aligned_two_line_observation()

    delta, info = estimator._apply_corridor_yaw_reset(
        now=10.0, raw_yaw=0.10, route_context=route, lane_observation=lane
    )

    assert delta == 0.0
    assert info is None


def test_corridor_yaw_reset_skipped_with_high_heading_error(monkeypatch) -> None:
    """Si el coche NO está alineado con el corredor, asumir lane_tangent≈path_psi
    es falso y el reset inyectaría sesgo."""
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_ENABLED", True)
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_MIN_GPS_AGE_S", 0.0)
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_MAX_HEADING_ERROR_DEG", 10.0)
    estimator = _make_corridor_yaw_estimator(dr_yaw=0.10)
    route = _route_with_path_psi(0.0)
    lane = _aligned_two_line_observation(heading_error_rad=math.radians(20.0))  # >10°

    delta, info = estimator._apply_corridor_yaw_reset(
        now=10.0, raw_yaw=0.10, route_context=route, lane_observation=lane
    )

    assert delta == 0.0
    assert info is None


def test_corridor_yaw_reset_skipped_with_low_speed(monkeypatch) -> None:
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_ENABLED", True)
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_MIN_GPS_AGE_S", 0.0)
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_MIN_SPEED_MPS", 0.10)
    estimator = _make_corridor_yaw_estimator(dr_yaw=0.10, speed_mps=0.03)
    route = _route_with_path_psi(0.0)
    lane = _aligned_two_line_observation()

    delta, info = estimator._apply_corridor_yaw_reset(
        now=10.0, raw_yaw=0.10, route_context=route, lane_observation=lane
    )

    assert delta == 0.0
    assert info is None


def test_corridor_yaw_reset_safety_guard_aborts_on_huge_delta(monkeypatch) -> None:
    """Si la corrección requerida es >max_delta_deg, abortar (probable bug)."""
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_ENABLED", True)
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_MIN_GPS_AGE_S", 0.0)
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_MAX_DELTA_DEG", 30.0)
    # raw_yaw = 0.0 pero el target manda a 60° → delta = 60° > 30° → abort.
    estimator = _make_corridor_yaw_estimator(dr_yaw=0.0)
    route = _route_with_path_psi(math.radians(60.0))
    lane = _aligned_two_line_observation(heading_error_rad=0.0)

    delta, info = estimator._apply_corridor_yaw_reset(
        now=10.0, raw_yaw=0.0, route_context=route, lane_observation=lane
    )

    assert delta == 0.0
    assert info is not None
    assert info["applied"] is False
    assert info["skipped_reason"] == "delta_too_large"
    assert estimator._dr.yaw == 0.0  # no cambió


def test_corridor_yaw_reset_disabled_by_flag(monkeypatch) -> None:
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_ENABLED", False)
    estimator = _make_corridor_yaw_estimator(dr_yaw=0.10)
    route = _route_with_path_psi(0.0)
    lane = _aligned_two_line_observation()

    delta, info = estimator._apply_corridor_yaw_reset(
        now=10.0, raw_yaw=0.10, route_context=route, lane_observation=lane
    )

    assert delta == 0.0
    assert info is None


def test_corridor_yaw_reset_respects_cooldown(monkeypatch) -> None:
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_ENABLED", True)
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_MIN_GPS_AGE_S", 0.0)
    monkeypatch.setattr("config.POSE_CORRIDOR_YAW_FIX_COOLDOWN_S", 3.0)
    estimator = _make_corridor_yaw_estimator(dr_yaw=0.10, last_corridor_t=9.0)
    route = _route_with_path_psi(0.0)
    lane = _aligned_two_line_observation()

    # 0.5s después del último reset, dentro del cooldown → no aplica.
    delta, info = estimator._apply_corridor_yaw_reset(
        now=9.5, raw_yaw=0.10, route_context=route, lane_observation=lane
    )
    assert delta == 0.0
    assert info is None

    # 4s después → ya pasó el cooldown → aplica.
    delta, info = estimator._apply_corridor_yaw_reset(
        now=13.0, raw_yaw=0.10, route_context=route, lane_observation=lane
    )
    assert info is not None and info["applied"] is True
    assert delta == pytest.approx(-0.10, abs=1e-6)
