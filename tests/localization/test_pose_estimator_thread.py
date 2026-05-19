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
from src.localization.dead_reckoning import DeadReckoning
from src.localization.chamfer_map_matcher import ChamferMatchResult
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

    def update_from_yaw(
        self,
        speed_mps: float,
        yaw_start_rad: float,
        yaw_end_rad: float,
        dt: float,
        **_,
    ) -> None:
        self.x += float(speed_mps) * float(dt) * math.cos(float(yaw_end_rad))
        self.y += float(speed_mps) * float(dt) * math.sin(float(yaw_end_rad))
        self.yaw = float(yaw_end_rad)

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


class _CaptureBuffer:
    def __init__(self, value=None) -> None:
        self.value = value
        self.writes = []

    def read_latest(self, *, with_metadata: bool = False):
        if with_metadata:
            return self.value, None, None
        return self.value

    def write(self, value, timestamp=None) -> None:
        self.writes.append((value, timestamp))


class _OneShotSub:
    def __init__(self, value) -> None:
        self.value = value

    def receive(self):
        value = self.value
        self.value = None
        return value


class _SequenceSub:
    def __init__(self, values) -> None:
        self.values = deque(values)

    def receive(self):
        if not self.values:
            return None
        return self.values.popleft()


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


def _make_thread_work_estimator(*, speed_mps: float, steer_deg: float, imu_yaw_deg: float) -> threadPoseEstimator:
    estimator = threadPoseEstimator.__new__(threadPoseEstimator)
    estimator._dr = DeadReckoning(0.0, 0.0, 0.0)
    estimator._last_t = time.monotonic() - 0.10
    estimator._last_speed = float(speed_mps)
    estimator._last_speed_t = None
    estimator._last_speed_source = "encoder"
    estimator._last_cmd_speed_t = None
    estimator._last_steer_rad = 0.0
    estimator._steer_filtered_rad = 0.0
    estimator._last_yaw_fusion_steer_rad = 0.0
    estimator._imu_yaw_inhibit_until = 0.0
    estimator._last_yaw_rad = 0.0
    estimator._start_yaw_rad = 0.0
    estimator._yaw_offset = 0.0
    estimator._yaw_offset_calibrated = True
    estimator._pending_yaw_offset_target_rad = None
    estimator._yaw_ekf_p = 0.5
    estimator._imu_received = False
    estimator._last_raw_imu = None
    estimator._last_imu_t = time.monotonic() - 0.10
    estimator._last_absolute_yaw_fix_monotonic = 0.0
    estimator._last_absolute_yaw_fix_source = None

    estimator._consume_state_change = lambda: None
    estimator._resolve_speed_mps = lambda _now: estimator._last_speed
    estimator._resolve_steer_rad = lambda _now: math.radians(float(steer_deg))
    estimator._imu_sub = _OneShotSub(str({"yaw": float(imu_yaw_deg), "roll": 0.0, "pitch": 0.0}))
    estimator.route_context_buffer = _CaptureBuffer(None)
    estimator.lane_observation_buffer = _CaptureBuffer(None)
    estimator.stopline_observation_buffer = _CaptureBuffer(None)
    estimator.pose_estimate_buffer = _CaptureBuffer(None)
    estimator._apply_localisation_fix = lambda _current_yaw, _now: (False, None)
    estimator._consume_sign_observation = lambda _now: None
    estimator._auto_gps_recent_status = lambda _now: None
    estimator._publish_location_from_pose_estimate = lambda _pose, _route: None
    estimator.tracking_state = SimpleNamespace(
        update_from_pose_estimate=lambda _pose: None,
        last_yaw_correction_deg=0.0,
    )
    estimator._debug_log_enabled = False
    estimator._debug_log_path = None
    estimator._frame_idx = 0
    estimator._log_every = 1
    return estimator


def test_thread_work_uses_trusted_imu_yaw_when_steering_is_quiet() -> None:
    estimator = _make_thread_work_estimator(speed_mps=0.20, steer_deg=0.0, imu_yaw_deg=10.0)

    estimator.thread_work()

    x, y, yaw = estimator._dr.get_state()
    assert x > 0.0
    assert y > 0.0
    assert yaw == pytest.approx(math.radians(10.0), abs=math.radians(0.2))
    pose_estimate, _ = estimator.pose_estimate_buffer.writes[-1]
    assert pose_estimate.fused_pose.yaw == pytest.approx(math.radians(10.0), abs=math.radians(0.2))
    assert pose_estimate.steer_rad == pytest.approx(0.0, abs=1e-9)
    assert estimator._last_yaw_rejected is False


def test_imu_yaw_gets_extra_authority_when_two_line_visual_is_lost() -> None:
    estimator = _make_thread_work_estimator(speed_mps=0.20, steer_deg=12.0, imu_yaw_deg=10.0)
    estimator._last_steer_rad = math.radians(12.0)

    yaw = estimator._fuse_yaw_for_dead_reckoning(
        now=100.0,
        yaw_start_rad=0.0,
        dt=0.10,
        imu_dict={"yaw": 10.0},
        previous_imu_t=99.9,
        lane_observation=None,
    )

    assert yaw > 0.0
    assert estimator._last_imu_yaw_gain >= 0.35
    assert estimator._last_imu_visual_lost_correction_active is True
    assert estimator._last_visual_two_line_reliable is False


def test_reliable_two_line_visual_keeps_imu_yaw_on_normal_steer_gate() -> None:
    estimator = _make_thread_work_estimator(speed_mps=0.20, steer_deg=12.0, imu_yaw_deg=10.0)
    estimator._last_steer_rad = math.radians(12.0)
    lane_observation = LaneObservation(
        detected_sides=("left", "right"),
        lateral_offset_m=0.0,
        direct_error_m=0.0,
        quality=1.0,
        measurement_mode="two_line",
        direct_error_valid=True,
    )

    yaw = estimator._fuse_yaw_for_dead_reckoning(
        now=100.0,
        yaw_start_rad=0.0,
        dt=0.10,
        imu_dict={"yaw": 10.0},
        previous_imu_t=99.9,
        lane_observation=lane_observation,
    )

    assert yaw < 0.0
    assert estimator._last_imu_visual_lost_correction_active is False
    assert estimator._last_visual_two_line_reliable is True


def test_pose_estimator_reuses_fresh_imu_between_messages() -> None:
    estimator = _make_thread_work_estimator(speed_mps=0.20, steer_deg=0.0, imu_yaw_deg=12.0)
    now = 100.0

    imu_dict, previous_imu_t = estimator._read_imu_sample_for_fusion(now)
    assert imu_dict is not None and float(imu_dict["yaw"]) == pytest.approx(12.0)
    assert previous_imu_t is not None
    assert estimator._last_imu_sample_reused is False

    imu_dict, previous_imu_t = estimator._read_imu_sample_for_fusion(now + 0.10)
    assert imu_dict is not None and float(imu_dict["yaw"]) == pytest.approx(12.0)
    assert previous_imu_t == pytest.approx(now)
    assert estimator._last_imu_sample_reused is True
    assert estimator._last_imu_sample_age_s == pytest.approx(0.10)

    imu_dict, _ = estimator._read_imu_sample_for_fusion(now + 1.00)
    assert imu_dict is None
    assert estimator._last_imu_sample_reused is False


def test_resolve_speed_uses_odometer_distance_when_speed_feedback_is_zero() -> None:
    estimator = threadPoseEstimator.__new__(threadPoseEstimator)
    estimator._speed_sub = _SequenceSub([0.0, 0.0, 0.0])
    estimator._speed_cmd_sub = _SequenceSub([None, None, None])
    estimator._distance_sub = _SequenceSub([1000, 1000, 1050])
    estimator._current_state_message = "AUTO"
    estimator._last_speed = 0.0
    estimator._last_speed_source = "none"
    estimator._last_speed_t = None
    estimator._last_cmd_speed_raw = 0.0
    estimator._last_cmd_speed_t = None
    estimator._cmd_speed_history = deque(maxlen=5)
    estimator._last_encoder_filter_reason = "uninitialized"
    estimator._last_encoder_raw_mps = 0.0
    estimator._last_distance_m = None
    estimator._last_distance_t = None
    estimator._last_distance_speed_mps = 0.0
    estimator._last_distance_speed_t = None
    estimator._last_raw_distance = None

    assert estimator._resolve_speed_mps(100.0) == pytest.approx(0.0)
    assert estimator._last_speed_source == "encoder_zero"

    assert estimator._resolve_speed_mps(100.1) == pytest.approx(0.0)
    assert estimator._last_speed_source == "encoder_zero"

    speed = estimator._resolve_speed_mps(100.5)

    assert speed == pytest.approx(0.10)
    assert estimator._last_speed_source == "encoder_distance"
    assert estimator._last_distance_speed_mps == pytest.approx(0.10)


def test_thread_work_rejects_imu_yaw_spike_when_steering_at_zero_speed() -> None:
    estimator = _make_thread_work_estimator(speed_mps=0.0, steer_deg=25.0, imu_yaw_deg=45.0)

    estimator.thread_work()

    x, y, yaw = estimator._dr.get_state()
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert yaw == pytest.approx(0.0, abs=1e-9)
    assert estimator._last_yaw_rejected is True
    assert estimator._last_imu_yaw_gain == pytest.approx(0.0, abs=1e-9)


def test_thread_work_uses_bicycle_yaw_when_imu_is_inhibited_in_turn() -> None:
    estimator = _make_thread_work_estimator(speed_mps=0.20, steer_deg=25.0, imu_yaw_deg=45.0)

    estimator.thread_work()

    x, y, yaw = estimator._dr.get_state()
    assert x > 0.0
    assert y < 0.0
    assert -0.08 < yaw < -0.01
    assert estimator._last_yaw_rejected is True
    assert estimator._last_imu_yaw_gain == pytest.approx(0.0, abs=1e-9)


def test_lane_yaw_reset_uses_straight_two_line_camera_yaw() -> None:
    estimator = _make_estimator(speed_mps=0.20, dr_y=0.0)
    estimator._lane_yaw_reset_count = 1
    estimator._lane_yaw_reset_next_t = 0.0
    estimator._calibrate_imu_yaw_offset_to = lambda _yaw: (True, 0.0)
    lane_observation = LaneObservation(
        detected_sides=("left", "right"),
        quality=0.95,
        curve_hint="STRAIGHT",
        measurement_mode="two_line",
        heading_error_rad=math.radians(0.5),
        camera_yaw_hint_rad=math.radians(2.0),
        camera_yaw_hint_confidence=0.95,
    )

    correction_rad = estimator._try_lane_yaw_reset(
        now=10.0,
        raw_yaw=0.0,
        route_context=_make_route_context(),
        lane_observation=lane_observation,
    )

    assert correction_rad == pytest.approx(math.radians(2.0), abs=1e-9)
    assert estimator._dr.get_state()[2] == pytest.approx(math.radians(2.0), abs=1e-9)
    assert estimator._lane_yaw_reset_next_t > 10.0


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


def test_apply_lane_observation_aligns_map_pose_to_two_line_visual_offset() -> None:
    estimator = _make_estimator(speed_mps=0.0, dr_y=0.22)
    estimator.tracking_state.current_scenario = "lane_keep"
    lane_observation = LaneObservation(
        detected_sides=("left", "right"),
        lateral_offset_m=0.02,
        direct_error_m=0.02,
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
        raw_y=0.22,
        raw_yaw=0.0,
        raw_lateral_error_m=0.22,
    )

    assert reliable is True
    assert correction_m == pytest.approx(0.08, abs=1e-9)
    assert new_x == pytest.approx(0.0, abs=1e-9)
    assert new_y == pytest.approx(0.14, abs=1e-9)
    assert new_yaw == pytest.approx(0.0, abs=1e-9)
    assert estimator._last_lane_relocalization_kind == "visual_lane_map_alignment"
    assert estimator._dr.corrections == []


def test_apply_lane_observation_blocks_strong_visual_map_alignment_near_intersection() -> None:
    estimator = _make_estimator(speed_mps=0.0, dr_y=0.22)
    estimator.tracking_state.current_scenario = "lane_keep"
    lane_observation = LaneObservation(
        detected_sides=("left", "right"),
        lateral_offset_m=0.02,
        direct_error_m=0.02,
        quality=1.0,
        measurement_mode="two_line",
        direct_error_valid=True,
        control_policy_mode="VISUAL_ASSIST",
    )
    route_context = RouteContext(
        route_active=True,
        matched_pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
        next_semantic_type="intersection",
        next_semantic_distance_m=0.10,
    )

    new_x, new_y, new_yaw, correction_m, reliable = estimator._apply_lane_observation(
        route_context,
        lane_observation,
        now=10.0,
        raw_x=0.0,
        raw_y=0.22,
        raw_yaw=0.0,
        raw_lateral_error_m=0.22,
    )

    assert reliable is False
    assert correction_m == 0.0
    assert (new_x, new_y, new_yaw) == pytest.approx((0.0, 0.22, 0.0), abs=1e-9)
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


def test_apply_chamfer_map_alignment_reanchors_pose_from_visual_lines() -> None:
    class _FakeMatcher:
        def match_body_points(self, *args, **kwargs):
            return ChamferMatchResult(
                applied=True,
                reason="matched",
                dx_m=0.02,
                dy_m=-0.03,
                best_cost_px=1.0,
                baseline_cost_px=5.0,
                improvement_px=4.0,
                point_count=40,
                valid_fraction=1.0,
            )

    estimator = _make_estimator(speed_mps=0.0, dr_y=0.10)
    estimator.tracking_state.current_scenario = "lane_keep"
    estimator._chamfer_matcher = _FakeMatcher()
    estimator._chamfer_matcher_failed = False
    estimator._last_chamfer_alignment_monotonic = 0.0
    lane_observation = LaneObservation(
        detected_sides=("left", "right"),
        quality=1.0,
        measurement_mode="two_line",
        line_points_body=tuple((0.04 * i, 0.175) for i in range(40)),
    )

    new_x, new_y, new_yaw, correction_m, applied, result = estimator._apply_chamfer_map_alignment(
        RouteContext(route_active=True, matched_pose=Pose2D(x=0.0, y=0.0, yaw=0.0)),
        lane_observation,
        now=10.0,
        raw_x=0.0,
        raw_y=0.10,
        raw_yaw=0.0,
    )

    assert applied is True
    assert result is not None and result.reason == "matched"
    assert correction_m == pytest.approx(math.hypot(0.02, -0.03), abs=1e-9)
    assert (new_x, new_y, new_yaw) == pytest.approx((0.02, 0.07, 0.0), abs=1e-9)
    assert estimator._dr.get_state() == pytest.approx((0.02, 0.07, 0.0), abs=1e-9)


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
    estimator._last_raw_imu = None
    estimator._last_imu_t = None
    estimator._pending_yaw_offset_target_rad = None
    estimator._yaw_offset_calibrated = False
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


def test_manual_dashboard_relocates_even_without_recent_gps(monkeypatch) -> None:
    import config as cfg

    monkeypatch.setattr(cfg, "MOTOR_OUTPUT", "serial", raising=False)

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
            "world_x": 2.5,
            "world_y": -1.25,
            "yaw_deg": 45.0,
            "meta": {
                "manual": True,
                "source": "manual_dashboard",
                "gps_calibration": True,
            },
        }
    )
    estimator._last_gps_raw_xy = None
    estimator._last_gps_raw_monotonic = 0.0
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
    assert info["mode"] == "gps_fix"
    assert estimator._dr.get_state() == pytest.approx(
        (2.5, -1.25, math.radians(45.0)),
        abs=1e-9,
    )
    assert estimator._pending_yaw_offset_target_rad == pytest.approx(math.radians(45.0))


def test_manual_dashboard_calibrates_gps_when_recent_fix_exists(monkeypatch) -> None:
    import config as cfg

    monkeypatch.setattr(cfg, "MOTOR_OUTPUT", "serial", raising=False)

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
            "world_x": 2.5,
            "world_y": -1.25,
            "yaw_deg": 45.0,
            "meta": {
                "manual": True,
                "source": "manual_dashboard",
                "gps_calibration": True,
            },
        }
    )
    estimator._last_gps_raw_xy = (2.0, -2.0)
    estimator._last_gps_raw_monotonic = time.monotonic()
    estimator._last_raw_imu = None
    estimator._last_absolute_yaw_fix_monotonic = 0.0
    estimator._last_absolute_yaw_fix_source = None
    estimator._last_yaw_rad = 0.0
    estimator._yaw_ekf_p = 0.0
    estimator._yaw_offset_calibrated = False
    estimator._pending_yaw_offset_target_rad = None
    estimator.tracking_state = SimpleNamespace(
        set_lane_measurement_state=lambda *_args, **_kwargs: None
    )

    applied, info = estimator._apply_localisation_fix(current_yaw=0.0)

    assert applied is True
    assert info is not None
    assert info["mode"] == "gps_calibration"
    assert info["source"] == "manual_dashboard:gps_calibration"
    assert estimator._gps_calibration_offset_xy == pytest.approx((0.5, 0.75))
    assert estimator._dr.get_state() == pytest.approx(
        (2.5, -1.25, math.radians(45.0)),
        abs=1e-9,
    )


def test_initial_world_pose_uses_saved_start_for_real_and_sim(monkeypatch) -> None:
    saved_pose = (4.0, -2.0, math.radians(30.0))
    default_pose = (0.0, 0.0, 0.0)
    estimator = threadPoseEstimator.__new__(threadPoseEstimator)
    estimator._graph = SimpleNamespace(get_start_pose=lambda: default_pose)

    monkeypatch.setattr(
        "src.localization.pose_estimator_thread.resolve_saved_start_pose",
        lambda graph, default=None: saved_pose,
    )

    assert estimator._resolve_initial_world_pose() == saved_pose


def _start_calibration_estimator() -> threadPoseEstimator:
    estimator = threadPoseEstimator.__new__(threadPoseEstimator)
    estimator._dr = _FakeDR(x=0.0, y=0.0, yaw=0.0)
    estimator._graph = SimpleNamespace(
        localisation_to_world_pose=lambda payload, default_yaw=0.0: (
            float(payload["world_x"]),
            float(payload["world_y"]),
            math.radians(float(payload["yaw_deg"]))
            if payload.get("yaw_deg") is not None
            else float(payload.get("yaw_rad", default_yaw)),
        ),
        resolve_node_id=lambda _value: None,
    )
    estimator._gps_calibration_offset_xy = None
    estimator._last_gps_raw_xy = None
    estimator._last_gps_raw_monotonic = 0.0
    estimator._gps_disabled = False
    estimator._last_raw_imu = None
    estimator._last_imu_t = None
    estimator._pending_yaw_offset_target_rad = None
    estimator._yaw_offset_calibrated = False
    estimator._last_absolute_yaw_fix_monotonic = 0.0
    estimator._last_absolute_yaw_fix_source = None
    estimator._last_yaw_rad = 0.0
    estimator._yaw_ekf_p = 0.0
    estimator._gps_fix_history = deque(maxlen=10)
    estimator._auto_gps_pending = False
    estimator._startup_gps_calibration_pending = False
    estimator.tracking_state = SimpleNamespace(
        set_lane_measurement_state=lambda *_args, **_kwargs: None
    )
    estimator.queuesList = {"General": object()}
    return estimator


def _gps_payload(x: float, y: float) -> dict:
    return {
        "world_x": float(x),
        "world_y": float(y),
        "timestamp": time.time(),
        "meta": {"source": "gps_localisation", "gps_frame_in_bounds": True},
    }


def test_start_calibration_waits_for_five_fresh_gps_samples(monkeypatch) -> None:
    import config as cfg

    monkeypatch.setattr(cfg, "MOTOR_OUTPUT", "serial", raising=False)

    estimator = _start_calibration_estimator()
    estimator._localisation_fix_sub = _OneShotSub(
        {
            "world_x": 5.0,
            "world_y": 7.5,
            "yaw_deg": 90.0,
            "meta": {
                "manual": True,
                "source": "manual_dashboard_start_calibration",
                "gps_calibration": True,
                "start_calibration": True,
            },
        }
    )

    applied, info = estimator._apply_localisation_fix(current_yaw=0.0, now=10.0)

    assert applied is False
    assert info is None
    assert estimator._start_gps_calibration_pending is True
    assert estimator._dr.get_state() == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)

    fixes = [(4.50, 8.00), (4.52, 8.02), (4.48, 7.98), (4.51, 8.01), (4.49, 7.99)]
    for index, (x, y) in enumerate(fixes):
        estimator._localisation_fix_sub = _OneShotSub(_gps_payload(x, y))
        applied, info = estimator._apply_localisation_fix(current_yaw=0.0, now=10.1 + index * 0.1)

    assert applied is True
    assert info is not None
    assert info["mode"] == "start_gps_calibration"
    assert info["gps_sample_count"] == 5
    assert estimator._start_gps_calibration_pending is False
    assert estimator._gps_calibration_offset_xy == pytest.approx((0.5, -0.5), abs=1e-9)
    assert estimator._dr.get_state() == pytest.approx(
        (5.0, 7.5, math.radians(90.0)),
        abs=1e-9,
    )
    assert estimator._pending_yaw_offset_target_rad == pytest.approx(math.radians(90.0))


def test_start_calibration_rejects_dispersed_gps_samples(monkeypatch) -> None:
    import config as cfg

    monkeypatch.setattr(cfg, "MOTOR_OUTPUT", "serial", raising=False)

    estimator = _start_calibration_estimator()
    estimator._begin_start_gps_calibration_for_pose(
        (5.0, 7.5, math.radians(90.0)),
        10.0,
        source="manual_dashboard_start_calibration",
    )
    for index, (x, y) in enumerate(((4.0, 8.0), (4.1, 8.0), (4.2, 8.0), (4.3, 8.0), (5.2, 8.0))):
        estimator._remember_start_gps_calibration_fix(
            _gps_payload(x, y),
            {"source": "gps_localisation", "gps_frame_in_bounds": True},
            10.1 + index * 0.1,
        )

    applied, info = estimator._try_apply_start_gps_calibration(current_yaw=0.0, now=10.6)

    assert applied is False
    assert info is None
    assert estimator._start_gps_calibration_pending is False
    assert estimator._start_gps_calibration_last_mode == "start_gps_calibration_failed"
    assert estimator._start_gps_calibration_last_source == "gps_spread_too_large"
    assert estimator._gps_calibration_offset_xy is None
    assert estimator._dr.get_state() == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_start_calibration_times_out_without_gps(monkeypatch) -> None:
    import config as cfg

    monkeypatch.setattr(cfg, "MOTOR_OUTPUT", "serial", raising=False)

    estimator = _start_calibration_estimator()
    estimator._begin_start_gps_calibration_for_pose(
        (5.0, 7.5, math.radians(90.0)),
        10.0,
        source="manual_dashboard_start_calibration",
    )

    applied, info = estimator._try_apply_start_gps_calibration(current_yaw=0.0, now=20.0)

    assert applied is False
    assert info is None
    assert estimator._start_gps_calibration_pending is False
    assert estimator._start_gps_calibration_last_mode == "start_gps_calibration_failed"
    assert estimator._start_gps_calibration_last_source == "not_enough_fresh_gps_samples"
    assert estimator._gps_calibration_offset_xy is None
    assert estimator._dr.get_state() == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_startup_gps_calibration_uses_sample_window(monkeypatch) -> None:
    import config as cfg

    monkeypatch.setattr(cfg, "MOTOR_OUTPUT", "serial", raising=False)

    estimator = _start_calibration_estimator()
    estimator._startup_world_pose = (5.0, 7.5, math.radians(90.0))
    estimator._startup_gps_calibration_pending = True

    started = estimator._apply_startup_gps_calibration_if_pending(
        _gps_payload(4.5, 8.0),
        now=10.0,
    )

    assert started is True
    assert estimator._startup_gps_calibration_pending is False
    assert estimator._start_gps_calibration_pending is True
    assert estimator._gps_calibration_offset_xy is None

    for index, (x, y) in enumerate(((4.50, 8.00), (4.52, 8.02), (4.48, 7.98), (4.51, 8.01), (4.49, 7.99))):
        estimator._remember_start_gps_calibration_fix(
            _gps_payload(x, y),
            {"source": "gps_localisation", "gps_frame_in_bounds": True},
            10.1 + index * 0.1,
        )

    calibrated, info = estimator._try_apply_start_gps_calibration(current_yaw=0.0, now=10.7)

    assert calibrated is True
    assert info is not None
    assert info["source"] == "startup_start_calibration"
    assert estimator._gps_calibration_offset_xy == pytest.approx((0.5, -0.5), abs=1e-9)
    assert estimator._dr.get_state() == pytest.approx(
        (5.0, 7.5, math.radians(90.0)),
        abs=1e-9,
    )


def test_no_gps_start_mode_sets_start_and_ignores_later_gps(monkeypatch) -> None:
    import config as cfg
    import src.localization.pose_estimator_thread as pet

    gps_mode = {"disabled": False, "source": None}

    def fake_save_gps_disabled(disabled: bool, *, source: str = "test"):
        gps_mode["disabled"] = bool(disabled)
        gps_mode["source"] = str(source)
        return dict(gps_mode)

    monkeypatch.setattr(cfg, "MOTOR_OUTPUT", "serial", raising=False)
    monkeypatch.setattr(pet, "is_gps_disabled", lambda: bool(gps_mode["disabled"]))
    monkeypatch.setattr(pet, "save_gps_disabled", fake_save_gps_disabled)

    estimator = _start_calibration_estimator()
    estimator._localisation_fix_sub = _OneShotSub(
        {
            "world_x": 5.0,
            "world_y": 7.5,
            "yaw_deg": 90.0,
            "meta": {
                "manual": True,
                "source": "manual_dashboard_no_gps_start",
                "start_calibration": True,
                "no_gps_mode": True,
                "gps_disabled": True,
            },
        }
    )

    applied, info = estimator._apply_localisation_fix(current_yaw=0.0, now=10.0)

    assert applied is True
    assert info is not None
    assert info["mode"] == "gps_disabled_start"
    assert gps_mode == {
        "disabled": True,
        "source": "manual_dashboard_no_gps_start",
    }
    assert estimator._gps_disabled is True
    assert estimator._dr.get_state() == pytest.approx(
        (5.0, 7.5, math.radians(90.0)),
        abs=1e-9,
    )

    estimator._localisation_fix_sub = _OneShotSub(_gps_payload(30.0, 40.0))
    applied, info = estimator._apply_localisation_fix(current_yaw=math.radians(90.0), now=10.2)

    assert applied is False
    assert info is None
    assert estimator._last_gps_raw_xy is None
    assert estimator._dr.get_state() == pytest.approx(
        (5.0, 7.5, math.radians(90.0)),
        abs=1e-9,
    )


def test_receive_localisation_fix_prefers_manual_over_newer_gps() -> None:
    class _FakeSubscriber:
        def __init__(self) -> None:
            self.payloads = [
                {
                    "world_x": 1.0,
                    "world_y": 2.0,
                    "meta": {"manual": True, "source": "manual_dashboard"},
                },
                {
                    "world_x": 9.0,
                    "world_y": 9.0,
                    "meta": {"source": "gps_localisation"},
                },
            ]

        def receive(self):
            if not self.payloads:
                return None
            return self.payloads.pop(0)

        def is_data_in_pipe(self) -> bool:
            return bool(self.payloads)

    estimator = threadPoseEstimator.__new__(threadPoseEstimator)
    estimator._localisation_fix_sub = _FakeSubscriber()

    payload = estimator._receive_localisation_fix_payload()

    assert payload["world_x"] == pytest.approx(1.0)
    assert payload["world_y"] == pytest.approx(2.0)
    assert payload["meta"]["source"] == "manual_dashboard"


def test_receive_localisation_fix_uses_latest_gps_when_no_manual() -> None:
    class _FakeSubscriber:
        def __init__(self) -> None:
            self.payloads = [
                {
                    "world_x": 1.0,
                    "world_y": 2.0,
                    "meta": {"source": "gps_localisation"},
                },
                {
                    "world_x": 3.0,
                    "world_y": 4.0,
                    "meta": {"source": "gps_localisation"},
                },
            ]

        def receive(self):
            if not self.payloads:
                return None
            return self.payloads.pop(0)

        def is_data_in_pipe(self) -> bool:
            return bool(self.payloads)

    estimator = threadPoseEstimator.__new__(threadPoseEstimator)
    estimator._localisation_fix_sub = _FakeSubscriber()

    payload = estimator._receive_localisation_fix_payload()

    assert payload["world_x"] == pytest.approx(3.0)
    assert payload["world_y"] == pytest.approx(4.0)
    assert payload["meta"]["source"] == "gps_localisation"


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
