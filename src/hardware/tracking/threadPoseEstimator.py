from __future__ import annotations

import ast
import math
import time
from types import SimpleNamespace

from src.hardware.pipeline.sharedTypes import LaneObservation, Pose2D, PoseEstimate, RouteContext, StoplineObservation
from src.hardware.tracking.threadTracking import (
    _CAMERA_LATERAL_CORRECTION_COOLDOWN_S,
    _CAMERA_LATERAL_CORRECTION_GAIN,
    _CAMERA_LATERAL_CORRECTION_MAX_M,
    _CAMERA_LATERAL_CORRECTION_STEP_MAX_M,
    _IMU_STEER_INHIBIT_DEG,
    _MAX_INTEGRATION_DT,
    _MAX_PHYSICAL_YAW_RATE_RADS,
    _SEMANTIC_RELOCALIZATION_COOLDOWN_S,
    _SEMANTIC_RELOCALIZATION_DISTANCE_TOLERANCE_M,
    _SEMANTIC_RELOCALIZATION_MAX_DISTANCE_M,
    _SEMANTIC_RELOCALIZATION_MAX_MAP_ERROR_M,
    _STEER_GAIN_DR,
    _STEER_LAG_ALPHA,
    _VISUAL_LANE_RELOCALIZATION_SPEED_MIN_MPS,
    _VISUAL_STOPLINE_EVENT_MAX_AGE_S,
    _VISUAL_STOPLINE_MAX_MAP_ERROR_M,
    _VISUAL_STOPLINE_RELOCALIZATION_COOLDOWN_S,
    _WHEELBASE_M,
    _YAW_EKF_P_INIT,
    _YAW_EKF_Q,
    _YAW_EKF_R_STRAIGHT,
    _YAW_EKF_R_STEER_K,
    threadTracking,
)


class threadPoseEstimator(threadTracking):
    """Dead-reckoning pose estimator with visual and semantic corrections."""

    def __init__(
        self,
        queuesList,
        tracking_state,
        lane_observation_buffer,
        stopline_observation_buffer,
        pose_estimate_buffer,
        route_context_buffer,
        *,
        logging=None,
        debugging: bool = False,
    ):
        super().__init__(queuesList, tracking_state, logging=logging, debugging=debugging, visualizer=None)
        self.lane_observation_buffer = lane_observation_buffer
        self.stopline_observation_buffer = stopline_observation_buffer
        self.pose_estimate_buffer = pose_estimate_buffer
        self.route_context_buffer = route_context_buffer
        self._last_camera_lateral_correction_monotonic = 0.0

    @staticmethod
    def _to_path_update(route_context: RouteContext | None):
        if route_context is None:
            return None
        return SimpleNamespace(
            route_active=route_context.route_active,
            matched_x=route_context.matched_pose.x,
            matched_y=route_context.matched_pose.y,
            matched_yaw=route_context.matched_pose.yaw,
            matched_idx=route_context.matched_idx,
            target_idx=route_context.target_idx,
            waypoint_mode_active=route_context.waypoint_mode_active,
            map_match_error_m=route_context.map_match_error_m,
            current_node_attr=route_context.current_node_attr,
            upcoming_node_attr=route_context.upcoming_node_attr,
            next_semantic_type=route_context.next_semantic_type,
            next_semantic_distance_m=route_context.next_semantic_distance_m,
            expected_control_type=route_context.expected_control_type,
            error_m=route_context.error_m,
            heading_rad=route_context.heading_rad,
            path_psi=route_context.path_psi,
            path_kappa=route_context.path_kappa,
            path_heading_change_rad=route_context.path_heading_change_rad,
        )

    def _apply_camera_yaw_hint(self, raw_yaw: float, lane_observation: LaneObservation | None) -> float:
        if lane_observation is None:
            return 0.0
        cam_yaw = lane_observation.camera_yaw_hint_rad
        cam_conf = float(lane_observation.camera_yaw_hint_confidence or 0.0)
        if cam_yaw is None or cam_conf <= 0.3:
            return 0.0
        alpha = 0.08 * cam_conf
        delta = float(cam_yaw) - float(raw_yaw)
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        yaw_correction = alpha * delta
        self._last_yaw_rad += yaw_correction
        self._dr.correct_yaw(yaw_correction)
        self.tracking_state.last_yaw_correction_deg = math.degrees(yaw_correction)
        return float(yaw_correction)

    def _apply_lane_observation(
        self,
        route_context: RouteContext | None,
        lane_observation: LaneObservation | None,
        now: float,
        raw_x: float,
        raw_y: float,
        raw_yaw: float,
    ) -> tuple[float, float, float, float, bool]:
        if self._dr is None or route_context is None:
            return raw_x, raw_y, raw_yaw, 0.0, False
        if lane_observation is None:
            return raw_x, raw_y, raw_yaw, 0.0, False
        if route_context.waypoint_mode_active:
            return raw_x, raw_y, raw_yaw, 0.0, False
        if abs(self._last_speed) < float(_VISUAL_LANE_RELOCALIZATION_SPEED_MIN_MPS):
            return raw_x, raw_y, raw_yaw, 0.0, False

        measurement = lane_observation.direct_error_m
        if measurement is None:
            measurement = lane_observation.lateral_offset_m
        if measurement is None:
            return raw_x, raw_y, raw_yaw, 0.0, False
        if float(lane_observation.quality or 0.0) < 0.35:
            return raw_x, raw_y, raw_yaw, 0.0, False

        if (now - self._last_camera_lateral_correction_monotonic) < float(
            _CAMERA_LATERAL_CORRECTION_COOLDOWN_S
        ):
            return raw_x, raw_y, raw_yaw, 0.0, True

        correction_m = float(measurement) * float(_CAMERA_LATERAL_CORRECTION_GAIN)
        max_corr_m = float(_CAMERA_LATERAL_CORRECTION_MAX_M)
        if max_corr_m > 0.0:
            correction_m = max(-max_corr_m, min(max_corr_m, correction_m))
        max_step_m = float(_CAMERA_LATERAL_CORRECTION_STEP_MAX_M)
        if max_step_m > 0.0:
            correction_m = max(-max_step_m, min(max_step_m, correction_m))

        if abs(correction_m) < 1e-9:
            return raw_x, raw_y, raw_yaw, 0.0, True

        self._dr.correct_lateral(correction_m, float(route_context.matched_pose.yaw))
        self._last_camera_lateral_correction_monotonic = now
        new_x, new_y, new_yaw = self._dr.get_state()
        return new_x, new_y, new_yaw, float(correction_m), True

    def _apply_semantic_reset(self, route_context: RouteContext | None, now: float) -> tuple[bool, tuple[str, float] | None]:
        if route_context is None or self._dr is None:
            return False, None
        path_update = self._to_path_update(route_context)
        semantic_match = self._sign_matches_expected_semantic(path_update)
        if semantic_match is None:
            return False, None
        if (float(now) - float(self._last_semantic_relocalization_t)) < float(_SEMANTIC_RELOCALIZATION_COOLDOWN_S):
            return False, semantic_match
        if route_context.next_semantic_distance_m is not None and float(route_context.next_semantic_distance_m) > float(
            _SEMANTIC_RELOCALIZATION_MAX_DISTANCE_M
        ):
            return False, semantic_match
        if float(route_context.map_match_error_m or 0.0) > float(_SEMANTIC_RELOCALIZATION_MAX_MAP_ERROR_M):
            return False, semantic_match
        observation = self._last_sign_observation if isinstance(self._last_sign_observation, dict) else {}
        observed_distance_m = observation.get("distance_m")
        if (
            observed_distance_m is not None
            and route_context.next_semantic_distance_m is not None
            and abs(float(observed_distance_m) - float(route_context.next_semantic_distance_m))
            > float(_SEMANTIC_RELOCALIZATION_DISTANCE_TOLERANCE_M)
        ):
            return False, semantic_match

        self._dr.reset(
            float(route_context.matched_pose.x),
            float(route_context.matched_pose.y),
            float(route_context.matched_pose.yaw),
        )
        self._last_semantic_relocalization_t = float(now)
        self._last_yaw_rad = float(route_context.matched_pose.yaw)
        return True, semantic_match

    def _apply_stopline_reset(
        self,
        route_context: RouteContext | None,
        stopline_observation: StoplineObservation | None,
        now: float,
    ) -> tuple[bool, tuple[str, float] | None]:
        if route_context is None or stopline_observation is None or self._dr is None:
            return False, None
        if not isinstance(stopline_observation.pass_event, dict):
            return False, None
        event_age = max(0.0, float(now) - float(stopline_observation.pass_event.get("observed_at_monotonic", now) or now))
        if event_age > float(_VISUAL_STOPLINE_EVENT_MAX_AGE_S):
            return False, None
        if (float(now) - float(self._last_visual_stopline_relocalization_t)) < float(
            _VISUAL_STOPLINE_RELOCALIZATION_COOLDOWN_S
        ):
            return False, None
        if float(route_context.map_match_error_m or 0.0) > float(_VISUAL_STOPLINE_MAX_MAP_ERROR_M):
            return False, None
        stopline_context = (
            route_context.next_semantic_type == "stopline"
            or route_context.current_node_attr == int(stopline_observation.expected_node_attr or 0)
            or route_context.upcoming_node_attr == int(stopline_observation.expected_node_attr or 0)
        )
        if not stopline_context:
            return False, None

        old_x, old_y, _ = self._dr.get_state()
        self._dr.reset(
            float(route_context.matched_pose.x),
            float(route_context.matched_pose.y),
            float(route_context.matched_pose.yaw),
        )
        self._last_visual_stopline_relocalization_t = float(now)
        self._last_yaw_rad = float(route_context.matched_pose.yaw)
        correction_m = math.hypot(float(route_context.matched_pose.x) - float(old_x), float(route_context.matched_pose.y) - float(old_y))
        source = f"stopline_visual:{stopline_observation.expected_node_id or 'matched_pose'}"
        return True, (source, float(correction_m))

    def _build_pose_estimate(
        self,
        now: float,
        raw_pose: Pose2D,
        fused_pose: Pose2D,
        raw_lateral_error_m: float,
        lane_measurement_reliable: bool,
        camera_lateral_correction_m: float,
        relocalization_mode: str,
        relocalization_source: str,
        relocalization_error_m: float,
        route_context: RouteContext | None,
    ) -> PoseEstimate:
        map_match_error_m = float(route_context.map_match_error_m or 0.0) if route_context is not None else 0.5
        route_conf = max(0.0, min(1.0, 1.0 - (map_match_error_m / 0.5)))
        lane_bonus = 0.2 if lane_measurement_reliable else 0.0
        localization_confidence = max(0.0, min(1.0, route_conf + lane_bonus))
        speed_feedback_age_s = (now - self._last_speed_t) if self._last_speed_t is not None else None
        speed_command_age_s = (now - self._last_cmd_speed_t) if self._last_cmd_speed_t is not None else None
        return PoseEstimate(
            timestamp=float(now),
            raw_pose=raw_pose,
            fused_pose=fused_pose,
            speed_mps=float(self._last_speed or 0.0),
            yaw_rad=float(fused_pose.yaw),
            steer_rad=float(self._last_steer_rad or 0.0),
            speed_source=str(self._last_speed_source or "none"),
            speed_feedback_age_s=speed_feedback_age_s,
            speed_command_age_s=speed_command_age_s,
            localization_confidence=localization_confidence,
            relocalization_mode=str(relocalization_mode or "dead_reckoning"),
            last_relocalization_source=str(relocalization_source or "none"),
            last_relocalization_error_m=float(relocalization_error_m or 0.0),
            raw_lateral_error_m=float(raw_lateral_error_m or 0.0),
            lane_measurement_reliable=bool(lane_measurement_reliable),
            camera_lateral_correction_m=float(camera_lateral_correction_m or 0.0),
            imu_received=bool(self._imu_received),
        )

    def thread_work(self):
        now = time.monotonic()
        dt = now - self._last_t
        self._last_t = now
        self._consume_state_change()
        self._resolve_speed_mps(now)
        self._last_steer_rad = self._resolve_steer_rad(now)

        imu_raw = self._imu_sub.receive()
        if imu_raw is not None:
            try:
                imu_dict = ast.literal_eval(str(imu_raw))
                self._last_raw_imu = imu_dict
                prev_imu_t = self._last_imu_t
                self._last_imu_t = now
                yaw_deg = float(imu_dict.get("yaw", math.degrees(self._last_yaw_rad)))
                yaw_raw_rad = -math.radians(yaw_deg)
                if not self._yaw_offset_calibrated:
                    self._yaw_offset = self._start_yaw_rad - yaw_raw_rad
                    self._yaw_offset_calibrated = True
                else:
                    # EKF correction: fuse IMU absolute heading with kinematic prediction.
                    # R scales with steer²: large steer → servo EMI biases magnetometer
                    # → kinematic model trusted more. Smooth transition, no hard cutoff.
                    yaw_imu = yaw_raw_rad + self._yaw_offset
                    dt_imu = (now - prev_imu_t) if prev_imu_t is not None else 0.05
                    innov = yaw_imu - self._last_yaw_rad
                    while innov > math.pi:
                        innov -= 2.0 * math.pi
                    while innov < -math.pi:
                        innov += 2.0 * math.pi
                    # Rate-limit innovation to reject servo-EMI step spikes
                    max_innov = _MAX_PHYSICAL_YAW_RATE_RADS * max(dt_imu, 0.02)
                    innov = max(-max_innov, min(max_innov, innov))
                    # Kalman gain: R grows with steer² → IMU less reliable when turning
                    R_imu = _YAW_EKF_R_STRAIGHT + _YAW_EKF_R_STEER_K * (self._steer_filtered_rad ** 2)
                    K = self._yaw_ekf_p / (self._yaw_ekf_p + R_imu)
                    self._last_yaw_rad += K * innov
                    self._yaw_ekf_p = (1.0 - K) * self._yaw_ekf_p
                self._imu_received = True
            except Exception:
                pass

        if self._dr is None:
            return

        eff_steer_raw = self._last_steer_rad * _STEER_GAIN_DR
        self._steer_filtered_rad = (
            self._steer_filtered_rad
            + _STEER_LAG_ALPHA * (eff_steer_raw - self._steer_filtered_rad)
        )
        eff_steer_rad = self._steer_filtered_rad
        dr_dt = min(dt, _MAX_INTEGRATION_DT)
        self._dr.update(
            self._last_speed,
            self._last_yaw_rad,
            dr_dt,
            steer_rad=eff_steer_rad,
            wheelbase_m=_WHEELBASE_M,
        )
        raw_x, raw_y, raw_yaw = self._dr.get_state()
        self._last_yaw_rad = float(raw_yaw)
        # EKF process noise: covariance grows over time as kinematic drift accumulates
        self._yaw_ekf_p = min(self._yaw_ekf_p + _YAW_EKF_Q * dr_dt, 1.0)
        raw_pose = Pose2D(float(raw_x), float(raw_y), float(raw_yaw))

        route_context, _, _ = self.route_context_buffer.read_latest(with_metadata=True)
        if not isinstance(route_context, RouteContext):
            route_context = None

        lane_observation, _, _ = self.lane_observation_buffer.read_latest(with_metadata=True)
        if not isinstance(lane_observation, LaneObservation):
            lane_observation = None
        stopline_observation, _, _ = self.stopline_observation_buffer.read_latest(with_metadata=True)
        if not isinstance(stopline_observation, StoplineObservation):
            stopline_observation = None

        yaw_correction_rad = self._apply_camera_yaw_hint(raw_yaw, lane_observation)
        if abs(yaw_correction_rad) > 1e-9:
            raw_x, raw_y, raw_yaw = self._dr.get_state()

        self._consume_sign_observation(now)

        raw_lateral_error_m = 0.0
        if route_context is not None:
            raw_lateral_error_m = self._signed_lateral_error_to_path(
                raw_x,
                raw_y,
                route_context.matched_pose.x,
                route_context.matched_pose.y,
                route_context.matched_pose.yaw,
            )

        raw_x, raw_y, raw_yaw, lane_relocalization_m, lane_measurement_reliable = self._apply_lane_observation(
            route_context,
            lane_observation,
            now,
            raw_x,
            raw_y,
            raw_yaw,
        )

        semantic_relocalized, semantic_match = self._apply_semantic_reset(route_context, now)
        if semantic_relocalized:
            raw_x, raw_y, raw_yaw = self._dr.get_state()

        stopline_relocalized, stopline_match = self._apply_stopline_reset(route_context, stopline_observation, now)
        if stopline_relocalized:
            raw_x, raw_y, raw_yaw = self._dr.get_state()

        relocalization_mode = "dead_reckoning"
        relocalization_source = "dead_reckoning"
        relocalization_error_m = 0.0
        if stopline_relocalized and stopline_match is not None:
            relocalization_mode = "visual_stopline"
            relocalization_source, relocalization_error_m = stopline_match
        elif semantic_relocalized and semantic_match is not None:
            relocalization_mode = "semantic"
            relocalization_source, relocalization_error_m = semantic_match
        elif abs(yaw_correction_rad) > math.radians(0.25):
            relocalization_mode = "lane_yaw_reset"
            relocalization_source = "camera_yaw_hint"
            relocalization_error_m = abs(math.degrees(yaw_correction_rad)) / 180.0
        elif abs(lane_relocalization_m) > 1e-9:
            relocalization_mode = "lane_relocalization"
            relocalization_source = "lane_center"
            relocalization_error_m = abs(float(lane_relocalization_m))

        fused_pose = Pose2D(float(raw_x), float(raw_y), float(raw_yaw))
        pose_estimate = self._build_pose_estimate(
            now,
            raw_pose=raw_pose,
            fused_pose=fused_pose,
            raw_lateral_error_m=raw_lateral_error_m,
            lane_measurement_reliable=lane_measurement_reliable,
            camera_lateral_correction_m=lane_relocalization_m,
            relocalization_mode=relocalization_mode,
            relocalization_source=relocalization_source,
            relocalization_error_m=relocalization_error_m,
            route_context=route_context,
        )
        self.pose_estimate_buffer.write(pose_estimate, timestamp=pose_estimate.timestamp)
        if self.tracking_state is not None and hasattr(self.tracking_state, "update_from_pose_estimate"):
            self.tracking_state.update_from_pose_estimate(pose_estimate)

        # Debug log (same file as parent, same format — only minimal fields available here)
        self._frame_idx += 1
        if self._debug_log_enabled and self._debug_log_path and \
                (self._frame_idx % self._log_every == 0):
            matched_x = float(raw_x)
            matched_y = float(raw_y)
            matched_yaw = float(raw_yaw)
            if route_context is not None:
                matched_x = route_context.matched_pose.x
                matched_y = route_context.matched_pose.y
                matched_yaw = route_context.matched_pose.yaw
            self._write_tracking_log(
                raw_x, raw_y, raw_yaw,
                matched_x, matched_y, matched_yaw,
                0.0, 0.0,
                0, 0, 0, False, 0, dt,
            )
