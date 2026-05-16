from __future__ import annotations

import ast
import math
import time
from types import SimpleNamespace

from src.core.types import LaneObservation, Pose2D, PoseEstimate, RouteContext, StoplineObservation
from src.core.types.perception import lane_observation_supports_lateral_relocalization
from src.utils.live_log import live_log
from src.utils.sim_start_pose import resolve_saved_start_pose
from src.localization.relocalization_thread import (
    _CAMERA_LATERAL_CORRECTION_COOLDOWN_S,
    _CAMERA_LATERAL_CORRECTION_GAIN,
    _CAMERA_LATERAL_CORRECTION_MAX_M,
    _CAMERA_LATERAL_CORRECTION_STEP_MAX_M,
    _IMU_STEER_INHIBIT_DEG,
    _IMU_YAW_SIGN,
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
from src.core.messaging.allMessages import Localisation
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.core.messaging.messageHandlerSender import messageHandlerSender

_WAYPOINT_TWO_LINE_LATERAL_RELOCALIZATION_SPEED_MIN_MPS = 0.02
_SINGLE_LINE_ROUTE_STRAIGHT_TOL_RAD = math.radians(6.0)
_SINGLE_LINE_ROUTE_DIRECTION_MARGIN_RAD = math.radians(10.0)
_SINGLE_LINE_ROUTE_MAX_YAW_MISMATCH_RAD = math.radians(18.0)


def _wrap_angle(angle_rad: float) -> float:
    angle = float(angle_rad)
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _single_line_direction_conflicts_with_route(
    route_context: RouteContext | None,
    lane_observation: LaneObservation | None,
    *,
    reference_yaw: float,
) -> bool:
    if route_context is None or lane_observation is None:
        return False
    if str(getattr(lane_observation, "measurement_mode", "none") or "none") != "single_line":
        return False

    route_path_yaw = float(
        getattr(route_context, "path_psi", getattr(route_context.matched_pose, "yaw", reference_yaw)) or reference_yaw
    )
    route_delta = _wrap_angle(route_path_yaw - float(reference_yaw))

    heading_hint = None
    cam_yaw = getattr(lane_observation, "camera_yaw_hint_rad", None)
    cam_conf = float(getattr(lane_observation, "camera_yaw_hint_confidence", 0.0) or 0.0)
    if cam_yaw is not None and cam_conf > 0.3:
        heading_hint = _wrap_angle(float(cam_yaw) - float(reference_yaw))
    else:
        try:
            heading_hint = float(getattr(lane_observation, "heading_error_rad", 0.0) or 0.0)
        except (TypeError, ValueError):
            heading_hint = None

    if heading_hint is None:
        return False

    if abs(route_delta) <= _SINGLE_LINE_ROUTE_STRAIGHT_TOL_RAD:
        return abs(float(heading_hint)) > _SINGLE_LINE_ROUTE_DIRECTION_MARGIN_RAD

    if (
        abs(float(heading_hint)) > _SINGLE_LINE_ROUTE_STRAIGHT_TOL_RAD
        and math.copysign(1.0, float(heading_hint)) != math.copysign(1.0, float(route_delta))
    ):
        return True

    return abs(_wrap_angle(float(heading_hint) - float(route_delta))) > _SINGLE_LINE_ROUTE_MAX_YAW_MISMATCH_RAD


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
        self._last_absolute_yaw_fix_monotonic = 0.0
        self._last_absolute_yaw_fix_source = None
        self._localisation_fix_sub = messageHandlerSubscriber(
            queuesList, Localisation, "lastOnly", subscribe=True
        )
        self._last_gps_raw_xy: tuple[float, float] | None = None
        self._last_gps_raw_monotonic = 0.0
        self._gps_calibration_offset_xy: tuple[float, float] | None = None
        self._startup_world_pose = self._resolve_initial_world_pose()
        self._apply_start_pose_override()
        self._send_sim_relocalize()

    @staticmethod
    def _sim_start_pose_enabled() -> bool:
        try:
            from config import MOTOR_OUTPUT
        except ImportError:
            return False
        return MOTOR_OUTPUT == "zmq"

    @staticmethod
    def _gps_dashboard_calibration_enabled() -> bool:
        try:
            from config import MOTOR_OUTPUT
        except ImportError:
            return True
        return MOTOR_OUTPUT != "zmq"

    @staticmethod
    def _payload_xy(payload: dict) -> tuple[float, float] | None:
        for x_key, y_key in (("world_x", "world_y"), ("x", "y"), ("posA", "posB")):
            if payload.get(x_key) is None or payload.get(y_key) is None:
                continue
            try:
                return float(payload[x_key]), float(payload[y_key])
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _raw_gps_xy_from_calibration_payload(payload: dict) -> tuple[float, float] | None:
        for x_key, y_key in (
            ("gps_raw_world_x", "gps_raw_world_y"),
            ("raw_gps_world_x", "raw_gps_world_y"),
            ("gps_raw_posA", "gps_raw_posB"),
        ):
            if payload.get(x_key) is None or payload.get(y_key) is None:
                continue
            try:
                return float(payload[x_key]), float(payload[y_key])
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _localisation_source(payload: dict, meta: dict, default_source: str) -> str:
        return str(meta.get("source") or payload.get("source") or default_source)

    def _remember_gps_raw_fix(self, payload: dict) -> None:
        xy = self._payload_xy(payload)
        if xy is None:
            return
        self._last_gps_raw_xy = xy
        self._last_gps_raw_monotonic = time.monotonic()

    def _latest_gps_raw_xy_for_calibration(self, payload: dict) -> tuple[float, float] | None:
        xy = self._raw_gps_xy_from_calibration_payload(payload)
        if xy is not None:
            return xy
        if self._last_gps_raw_xy is None:
            return None
        if time.monotonic() - self._last_gps_raw_monotonic > 3.0:
            return None
        return self._last_gps_raw_xy

    def _should_calibrate_gps_from_dashboard(self, payload: dict, meta: dict) -> bool:
        if not self._gps_dashboard_calibration_enabled():
            return False
        if not bool(meta.get("manual")):
            return False
        source = self._localisation_source(payload, meta, "manual_localisation").strip().lower()
        return bool(meta.get("gps_calibration")) or source.startswith("manual_dashboard")

    def _apply_gps_dashboard_calibration(self, payload: dict, meta: dict, current_yaw: float):
        raw_xy = self._latest_gps_raw_xy_for_calibration(payload)
        if raw_xy is None:
            print(
                "\033[1;97m[ PoseEstimator ] :\033[0m \033[1;93mWARNING\033[0m"
                " - GPS calibration ignored: no recent raw LoCSys fix available"
            )
            return False, None

        old_x, old_y, _ = self._dr.get_state()
        pose = self._graph.localisation_to_world_pose(payload, default_yaw=current_yaw)
        if pose is None:
            return False, None

        target_x, target_y, yaw = pose
        raw_x, raw_y = raw_xy
        offset_x = float(target_x) - float(raw_x)
        offset_y = float(target_y) - float(raw_y)
        self._gps_calibration_offset_xy = (offset_x, offset_y)

        self._dr.reset(float(target_x), float(target_y), float(yaw))
        self._last_yaw_rad = float(yaw)
        self._yaw_ekf_p = _YAW_EKF_P_INIT
        self._last_absolute_yaw_fix_monotonic = time.monotonic()
        self._last_absolute_yaw_fix_source = "gps_dashboard_calibration"
        if self.tracking_state is not None and hasattr(self.tracking_state, "set_lane_measurement_state"):
            self.tracking_state.set_lane_measurement_state(False, 0.0)

        print(
            "\033[1;97m[ PoseEstimator ] :\033[0m \033[1;92mINFO\033[0m"
            f" - GPS calibrated: raw ({raw_x:.3f}, {raw_y:.3f}) -> "
            f"map ({float(target_x):.3f}, {float(target_y):.3f}); "
            f"offset=({offset_x:.3f}, {offset_y:.3f})"
        )
        return True, {
            "mode": "gps_calibration",
            "source": "manual_dashboard:gps_calibration",
            "error_m": float(math.hypot(float(target_x) - float(old_x), float(target_y) - float(old_y))),
        }

    def _apply_gps_calibration_offset(self, payload: dict, meta: dict) -> tuple[dict, dict]:
        if self._gps_calibration_offset_xy is None or not self._gps_dashboard_calibration_enabled():
            return payload, meta
        xy = self._payload_xy(payload)
        if xy is None:
            return payload, meta
        raw_x, raw_y = xy
        offset_x, offset_y = self._gps_calibration_offset_xy
        adjusted_x = float(raw_x) + float(offset_x)
        adjusted_y = float(raw_y) + float(offset_y)

        adjusted_payload = dict(payload)
        adjusted_meta = dict(meta)
        adjusted_payload["world_x"] = adjusted_x
        adjusted_payload["world_y"] = adjusted_y
        adjusted_payload["posA"] = adjusted_x
        adjusted_payload["posB"] = adjusted_y
        adjusted_payload["gps_raw_world_x"] = float(raw_x)
        adjusted_payload["gps_raw_world_y"] = float(raw_y)
        adjusted_meta["gps_calibrated"] = True
        adjusted_meta["gps_calibration_offset_x"] = float(offset_x)
        adjusted_meta["gps_calibration_offset_y"] = float(offset_y)
        adjusted_payload["meta"] = adjusted_meta
        return adjusted_payload, adjusted_meta

    def _resolve_initial_world_pose(self) -> tuple[float, float, float] | None:
        if self._graph is None:
            return None
        default_pose = self._graph.get_start_pose()
        if not self._sim_start_pose_enabled():
            return default_pose
        return resolve_saved_start_pose(self._graph, default=default_pose)

    def _apply_start_pose_override(self) -> None:
        if self._dr is None or self._startup_world_pose is None:
            return
        x0, y0, yaw0 = self._startup_world_pose
        self._dr.reset(float(x0), float(y0), float(yaw0))
        self._start_yaw_rad = float(yaw0)
        self._last_yaw_rad = float(yaw0)
        self._yaw_offset = 0.0
        self._yaw_offset_calibrated = False

    def _send_sim_relocalize_pose(
        self,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        *,
        source: str,
    ) -> None:
        try:
            from config import (
                MOTOR_OUTPUT,
                GZ_SPAWN_Z,
            )
        except ImportError:
            return
        if MOTOR_OUTPUT != "zmq":
            return
        from src.core.messaging.allMessages import SimRelocalize
        messageHandlerSender(self.queuesList, SimRelocalize).send({
            "world_x": float(x_m),
            "world_y": float(y_m),
            "yaw_rad": float(yaw_rad),
            "z": float(GZ_SPAWN_Z),
        })
        print(
            f"\033[1;97m[ PoseEstimator ] :\033[0m \033[1;92mINFO\033[0m"
            f" - SIM: {source} → ({float(x_m):.3f}, {float(y_m):.3f})"
            f" yaw={math.degrees(float(yaw_rad)):.1f}°"
        )

    def _has_fresh_absolute_yaw_fix(self, now: float, freshness_s: float = 0.50) -> bool:
        return (
            self._last_absolute_yaw_fix_monotonic > 0.0
            and (now - self._last_absolute_yaw_fix_monotonic) < float(freshness_s)
        )

    def _send_sim_relocalize(self) -> None:
        """In sim mode, teleport the Gazebo car to the startup pose."""
        if self._graph is None:
            return
        pose = self._startup_world_pose or self._graph.get_start_pose()
        x0, y0, yaw0 = pose
        self._send_sim_relocalize_pose(
            x0,
            y0,
            yaw0,
            source="startup pose",
        )

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

    def _apply_camera_yaw_hint(
        self,
        now: float,
        raw_yaw: float,
        route_context: RouteContext | None,
        lane_observation: LaneObservation | None,
    ) -> float:
        if lane_observation is None:
            return 0.0
        # Si acabamos de recibir una fijación absoluta con yaw explícito
        # (sim bridge / GPS), priorizamos esa orientación durante una pequeña
        # ventana. La cámara da una buena tangente de carril, pero en curvas o
        # con un solo borde visible puede sesgar 10–20° el heading y desalinear
        # el auto del simulador aunque el fix absoluto sea correcto.
        if self._has_fresh_absolute_yaw_fix(now):
            return 0.0
        # Con poca velocidad, el hint de cámara tiende a reflejar el borde
        # visible o el último frame "usable" y puede rotar el auto aunque la
        # pose absoluta esté quieta. También exigimos una calidad similar a la
        # del lane relocalization lateral para no aplicar yaw sobre fallback
        # visual débil.
        if abs(float(self._last_speed or 0.0)) < float(_VISUAL_LANE_RELOCALIZATION_SPEED_MIN_MPS):
            return 0.0
        if float(lane_observation.quality or 0.0) < 0.35:
            return 0.0
        cam_yaw = lane_observation.camera_yaw_hint_rad
        cam_conf = float(lane_observation.camera_yaw_hint_confidence or 0.0)
        if cam_yaw is None or cam_conf <= 0.3:
            return 0.0
        if _single_line_direction_conflicts_with_route(
            route_context,
            lane_observation,
            reference_yaw=raw_yaw,
        ):
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
        supports_lateral_relocalization = lane_observation_supports_lateral_relocalization(
            lane_observation
        )
        if not supports_lateral_relocalization:
            return raw_x, raw_y, raw_yaw, 0.0, False
        min_speed_mps = float(_VISUAL_LANE_RELOCALIZATION_SPEED_MIN_MPS)
        if bool(route_context.waypoint_mode_active):
            min_speed_mps = min(
                min_speed_mps,
                float(_WAYPOINT_TWO_LINE_LATERAL_RELOCALIZATION_SPEED_MIN_MPS),
            )
        if abs(self._last_speed) < min_speed_mps:
            return raw_x, raw_y, raw_yaw, 0.0, False

        measurement = lane_observation.direct_error_m
        if measurement is None:
            measurement = lane_observation.lateral_offset_m
        if measurement is None:
            return raw_x, raw_y, raw_yaw, 0.0, False

        # NOTA: probamos modo CONFIDENT (snap rápido cuando quality≥0.7)
        # y empeoraba el tracking — el lane visual, cuando hay desalineamiento
        # textura↔mapa, metía la pose al carril visual a costa del centerline
        # OSM. Con el frame OSM unificado, el centerline debería coincidir con
        # el carril visual, así que el lane visual solo tiene que reforzar la
        # pose. Pero los saltos de 5cm/tick eran demasiado abruptos.
        # Mantener solo el modo cautious (controlado por GAIN/MAX/STEP/COOLDOWN
        # de config) parece el mejor compromiso por ahora.

        # Modo CAUTIOUS (single-side visible, quality 0.35–0.7): la medida
        # es ruidosa, aplicamos correcciones lentas con cooldown como antes.
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

        old_x, old_y, current_yaw = self._dr.get_state()
        self._dr.reset(
            float(route_context.matched_pose.x),
            float(route_context.matched_pose.y),
            current_yaw,
        )
        self._last_visual_stopline_relocalization_t = float(now)
        correction_m = math.hypot(float(route_context.matched_pose.x) - float(old_x), float(route_context.matched_pose.y) - float(old_y))
        source = f"stopline_visual:{stopline_observation.expected_node_id or 'matched_pose'}"
        return True, (source, float(correction_m))

    def _apply_localisation_fix(self, current_yaw: float):
        if self._dr is None or self._graph is None:
            return False, None
        payload = self._localisation_fix_sub.receive()
        if not isinstance(payload, dict):
            return False, None

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            meta = {}

        default_source = "manual_localisation" if bool(meta.get("manual")) else "gps_localisation"
        source = self._localisation_source(payload, meta, default_source)
        if source.strip().lower() == "gps_localisation":
            self._remember_gps_raw_fix(payload)
            payload, meta = self._apply_gps_calibration_offset(payload, meta)
        elif self._should_calibrate_gps_from_dashboard(payload, meta):
            return self._apply_gps_dashboard_calibration(payload, meta, current_yaw)

        old_x, old_y, _ = self._dr.get_state()
        pose = self._graph.localisation_to_world_pose(payload, default_yaw=current_yaw)
        if pose is None:
            return False, None

        x, y, yaw = pose
        self._dr.reset(float(x), float(y), float(yaw))
        self._last_yaw_rad = float(yaw)
        self._yaw_ekf_p = _YAW_EKF_P_INIT
        if payload.get("yaw_rad") is not None or payload.get("yaw_deg") is not None:
            self._last_absolute_yaw_fix_monotonic = time.monotonic()
            self._last_absolute_yaw_fix_source = str(meta.get("source") or payload.get("source") or "gps_localisation")
        if self.tracking_state is not None and hasattr(self.tracking_state, "set_lane_measurement_state"):
            self.tracking_state.set_lane_measurement_state(False, 0.0)

        source = self._localisation_source(payload, meta, default_source)
        if bool(meta.get("gps_calibrated")):
            source = f"{source}:calibrated"
        resolved_node_id = self._graph.resolve_node_id(meta.get("node_id") or payload.get("node_id"))
        if resolved_node_id is not None:
            source = f"{source}:{resolved_node_id}"
        if bool(meta.get("manual")):
            self._send_sim_relocalize_pose(float(x), float(y), float(yaw), source=source)
        error_m = math.hypot(float(x) - float(old_x), float(y) - float(old_y))
        return True, {
            "mode": "gps_fix",
            "source": source,
            "error_m": float(error_m),
        }

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
        wall_timestamp: float | None = None,
    ) -> PoseEstimate:
        # ``now`` is monotonic and used internally for *age* deltas (must be
        # consistent with the rest of this thread's monotonic counters).
        # ``wall_timestamp`` is wall-clock (``time.time()``) and used for the
        # *published* PoseEstimate.timestamp — that field crosses the thread
        # boundary into safety_gate / motor_command_dispatcher, both of which
        # compute ``time.time() - timestamp`` and would otherwise see an
        # ``age ≈ 1.78e9 s`` (wall - monotonic) and trigger a permanent
        # ``pose_stale`` fallback. See safety_gate.py:135.
        map_match_error_m = float(route_context.map_match_error_m or 0.0) if route_context is not None else 0.5
        route_conf = max(0.0, min(1.0, 1.0 - (map_match_error_m / 0.5)))
        lane_bonus = 0.2 if lane_measurement_reliable else 0.0
        localization_confidence = max(0.0, min(1.0, route_conf + lane_bonus))
        speed_feedback_age_s = (now - self._last_speed_t) if self._last_speed_t is not None else None
        speed_command_age_s = (now - self._last_cmd_speed_t) if self._last_cmd_speed_t is not None else None
        published_ts = float(wall_timestamp) if wall_timestamp is not None else time.time()
        return PoseEstimate(
            timestamp=published_ts,
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

    def _publish_location_from_pose_estimate(
        self,
        pose_estimate: PoseEstimate,
        route_context: RouteContext | None,
    ) -> None:
        fused = pose_estimate.fused_pose
        raw = pose_estimate.raw_pose
        matched = route_context.matched_pose if route_context is not None else fused
        try:
            self._loc_sender.send({
                "x": round(float(fused.x), 4),
                "y": round(float(fused.y), 4),
                "yaw": round(math.degrees(float(fused.yaw)), 2),
                "yaw_rad": float(fused.yaw),
                "raw_x": round(float(raw.x), 4),
                "raw_y": round(float(raw.y), 4),
                "raw_yaw": round(math.degrees(float(raw.yaw)), 2),
                "matched_x": round(float(matched.x), 4),
                "matched_y": round(float(matched.y), 4),
                "matched_yaw": round(math.degrees(float(matched.yaw)), 2),
                "meta": {
                    "source": "ego_pose_pose_estimator",
                    "frame": "osm_map",
                    "pose_source": "PoseEstimate.fused_pose",
                    "relocalization_mode": pose_estimate.relocalization_mode,
                    "last_relocalization_source": pose_estimate.last_relocalization_source,
                    "last_relocalization_error_m": float(pose_estimate.last_relocalization_error_m),
                    "localization_confidence": float(pose_estimate.localization_confidence),
                },
            })
        except Exception:
            pass

    def thread_work(self):
        now = time.monotonic()
        # Wall-clock companion: we keep ``now`` in monotonic land for all
        # internal age/cooldown deltas (resilient to NTP step changes), but
        # we capture the wall clock once per tick to stamp PoseEstimate so
        # other threads (safety_gate, dispatcher) — which compare against
        # ``time.time()`` — read a sane age. Single sample per tick keeps
        # the timestamp consistent across all downstream re-stamps in the
        # same iteration.
        wall_now = time.time()
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
                yaw_raw_rad = _IMU_YAW_SIGN * math.radians(yaw_deg)
                if not self._yaw_offset_calibrated:
                    self._yaw_offset = self._start_yaw_rad - yaw_raw_rad
                    self._yaw_offset_calibrated = True
                elif not self._has_fresh_absolute_yaw_fix(now):
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

        gps_relocalized, gps_match = self._apply_localisation_fix(raw_yaw)
        if gps_relocalized:
            raw_x, raw_y, raw_yaw = self._dr.get_state()
            raw_pose = Pose2D(float(raw_x), float(raw_y), float(raw_yaw))
            yaw_correction_rad = 0.0
            raw_lateral_error_m = 0.0
            lane_relocalization_m = 0.0
            lane_measurement_reliable = False
            semantic_relocalized = False
            semantic_match = None
            stopline_relocalized = False
            stopline_match = None
        else:
            yaw_correction_rad = self._apply_camera_yaw_hint(now, raw_yaw, route_context, lane_observation)
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
        if gps_relocalized and isinstance(gps_match, dict):
            relocalization_mode = str(gps_match.get("mode") or "gps_fix")
            relocalization_source = str(gps_match.get("source") or "gps_localisation")
            relocalization_error_m = float(gps_match.get("error_m") or 0.0)
        elif stopline_relocalized and stopline_match is not None:
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
            wall_timestamp=wall_now,
        )
        self.pose_estimate_buffer.write(pose_estimate, timestamp=pose_estimate.timestamp)
        if self.tracking_state is not None and hasattr(self.tracking_state, "update_from_pose_estimate"):
            self.tracking_state.update_from_pose_estimate(pose_estimate)
        self._publish_location_from_pose_estimate(pose_estimate, route_context)

        live_log(
            "pose_estimator", event="pose_published",
            fused_x=float(fused_pose.x), fused_y=float(fused_pose.y),
            fused_yaw_rad=float(fused_pose.yaw),
            raw_x=float(raw_pose.x), raw_y=float(raw_pose.y),
            raw_yaw_rad=float(raw_pose.yaw),
            speed_mps=float(pose_estimate.speed_mps),
            steer_rad=float(pose_estimate.steer_rad),
            reloc_mode=relocalization_mode,
            reloc_source=relocalization_source,
            reloc_error_m=float(relocalization_error_m or 0.0),
            lane_correction_m=float(lane_relocalization_m or 0.0),
            lane_reliable=bool(lane_measurement_reliable),
            lane_measurement_mode=(
                str(lane_observation.measurement_mode or "none")
                if lane_observation is not None
                else "none"
            ),
            lane_direct_error_valid=bool(
                lane_observation.direct_error_valid if lane_observation is not None else False
            ),
            lane_control_policy_mode=(
                lane_observation.control_policy_mode
                if lane_observation is not None
                else None
            ),
            raw_lateral_error_m=float(raw_lateral_error_m or 0.0),
            imu_received=bool(getattr(self, "_imu_received", False)),
            ts_pose=float(pose_estimate.timestamp),
        )

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
