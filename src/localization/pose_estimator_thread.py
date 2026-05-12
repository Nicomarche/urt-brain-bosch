from __future__ import annotations

import ast
import math
import time
from types import SimpleNamespace

from src.core.types import LaneObservation, Pose2D, PoseEstimate, RouteContext, StoplineObservation
from src.core.types.perception import (
    lane_observation_has_visual_path,
    lane_observation_supports_lateral_relocalization,
)
from src.localization.visual_lane_matcher import match_visual_lane_to_route
from src.utils.live_log import live_log
from src.utils.sim_start_pose import resolve_saved_start_pose
from src.localization.relocalization_thread import (
    _CAMERA_LATERAL_CORRECTION_COOLDOWN_S,
    _CAMERA_LATERAL_CORRECTION_GAIN,
    _CAMERA_LATERAL_CORRECTION_MAX_M,
    _CAMERA_LATERAL_CORRECTION_STEP_MAX_M,
    _GPS_MAX_EXPECTED_ERROR_M,
    _GPS_MAX_JUMP_M,
    _GPS_MIN_SAMPLES,
    _GPS_OUTLIER_DISTANCE_M,
    _GPS_SAMPLE_WINDOW,
    _GPS_VALIDATION_ENABLED,
    _GPS_ZERO_EPS_M,
    _DR_SPEED_SCALE,
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
# sim_bridge keeps the previous IMU payload fresh for up to one second; after
# teleporting we wait past that window before using yaw for a new offset.
_SIM_RELOCALIZE_IMU_SETTLE_S = 1.10

try:
    import config as _config
except Exception:  # pragma: no cover - test/import fallback
    _config = SimpleNamespace()

_LANE_WIDTH_M = max(0.01, float(getattr(_config, "LANE_WIDTH_CM", 35.0) or 35.0) / 100.0)
_LANE_HALF_WIDTH_M = 0.5 * _LANE_WIDTH_M
_LOCALIZATION_GPS_AUTHORITY = str(
    getattr(_config, "LOCALIZATION_GPS_AUTHORITY", "init_recovery_soft") or "init_recovery_soft"
).strip().lower()
_GPS_SOFT_GAIN = float(getattr(_config, "TRACKING_GPS_SOFT_GAIN", 0.25) or 0.25)
_GPS_SOFT_MAX_STEP_M = float(getattr(_config, "TRACKING_GPS_SOFT_MAX_STEP_M", 0.020) or 0.020)
_GPS_RECOVERY_MAX_STEP_M = float(getattr(_config, "TRACKING_GPS_RECOVERY_MAX_STEP_M", 0.050) or 0.050)
_GPS_RECOVERY_ERROR_M = float(getattr(_config, "TRACKING_GPS_RECOVERY_ERROR_M", 0.30) or 0.30)
_GPS_SOFT_LATERAL_GAIN = float(getattr(_config, "TRACKING_GPS_SOFT_LATERAL_GAIN", 0.20) or 0.20)
_GPS_VISUAL_LATERAL_BLOCK_M = float(getattr(_config, "TRACKING_GPS_VISUAL_LATERAL_BLOCK_M", 0.05) or 0.05)
_GPS_VISUAL_PROTECT_QUALITY = float(getattr(_config, "TRACKING_GPS_VISUAL_PROTECT_QUALITY", 0.55) or 0.55)
_VISUAL_MAP_MATCH_ENABLED = bool(getattr(_config, "VISUAL_MAP_MATCH_ENABLED", True))
_VISUAL_MAP_MATCH_MIN_CONFIDENCE = float(
    getattr(_config, "VISUAL_MAP_MATCH_MIN_CONFIDENCE", 0.45) or 0.45
)
_VISUAL_MAP_MATCH_MAX_LATERAL_ERROR_M = float(
    getattr(_config, "VISUAL_MAP_MATCH_MAX_LATERAL_ERROR_M", _LANE_HALF_WIDTH_M) or _LANE_HALF_WIDTH_M
)
_VISUAL_MAP_MATCH_MAX_YAW_ERROR_RAD = math.radians(
    float(getattr(_config, "VISUAL_MAP_MATCH_MAX_YAW_ERROR_DEG", 6.0) or 6.0)
)
_VISUAL_MAP_MATCH_MAX_SAMPLE_YAW_ERROR_RAD = math.radians(
    float(getattr(_config, "VISUAL_MAP_MATCH_MAX_SAMPLE_YAW_ERROR_DEG", 8.0) or 8.0)
)
_VISUAL_MAP_MATCH_CORRECTION_GAIN = float(
    getattr(_config, "VISUAL_MAP_MATCH_CORRECTION_GAIN", 0.20) or 0.20
)
_VISUAL_MAP_MATCH_CORRECTION_STEP_MAX_M = float(
    getattr(_config, "VISUAL_MAP_MATCH_CORRECTION_STEP_MAX_M", 0.008) or 0.008
)
_VISUAL_MAP_MATCH_CORRECTION_COOLDOWN_S = float(
    getattr(_config, "VISUAL_MAP_MATCH_CORRECTION_COOLDOWN_S", 0.10) or 0.10
)
_VISUAL_MAP_MATCH_YAW_CORRECTION_GAIN = float(
    getattr(_config, "VISUAL_MAP_MATCH_YAW_CORRECTION_GAIN", 0.10) or 0.10
)
_VISUAL_MAP_MATCH_YAW_CORRECTION_STEP_MAX_RAD = math.radians(
    float(getattr(_config, "VISUAL_MAP_MATCH_YAW_CORRECTION_STEP_MAX_DEG", 0.50) or 0.50)
)


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


def _lane_observation_has_local_authority(lane_observation: LaneObservation | None) -> bool:
    if lane_observation is None:
        return False
    if float(getattr(lane_observation, "quality", 0.0) or 0.0) < float(_GPS_VISUAL_PROTECT_QUALITY):
        return False
    if lane_observation_supports_lateral_relocalization(lane_observation, min_quality=0.70):
        return True
    return lane_observation_has_visual_path(
        lane_observation,
        min_quality=float(_GPS_VISUAL_PROTECT_QUALITY),
        min_points=int(getattr(_config, "LANE_VISUAL_MIN_POLY_POINTS", 8) or 8),
    )


def _route_reference_yaw(route_context: RouteContext | None, fallback_yaw: float) -> float:
    if route_context is None:
        return float(fallback_yaw)
    for value in (
        getattr(route_context, "path_psi", None),
        getattr(getattr(route_context, "matched_pose", None), "yaw", None),
    ):
        try:
            if value is not None and math.isfinite(float(value)):
                return float(value)
        except (TypeError, ValueError):
            continue
    return float(fallback_yaw)


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
        self._last_visual_map_match_correction_monotonic = 0.0
        self._last_absolute_yaw_fix_monotonic = 0.0
        self._last_absolute_yaw_fix_source = None
        self._ignore_imu_until_monotonic = 0.0
        self._gps_fix_samples: list[tuple[float, float]] = []
        self._last_accepted_gps_pose: tuple[float, float] | None = None
        self._last_gps_fix_quality = 0.0
        # GPS-derived heading: estima yaw a partir de fixes consecutivos.
        # Cuando el coche se mueve >0.3m entre 2 fixes, atan2(dy,dx) es el
        # yaw real del coche (independiente del drift IMU/dead reckoning).
        self._gps_heading_anchor: tuple[float, float] | None = None  # (x, y) del último anchor
        self._gps_heading_last_yaw: float | None = None
        self._localisation_fix_sub = messageHandlerSubscriber(
            queuesList, Localisation, "lastOnly", subscribe=True
        )
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

    def _resolve_initial_world_pose(self) -> tuple[float, float, float] | None:
        if self._graph is None:
            return None
        default_pose = self._graph.get_start_pose()
        configured_pose = self._resolve_configured_start_pose(default_pose)
        if configured_pose is not None:
            return configured_pose
        if not self._sim_start_pose_enabled():
            return default_pose
        return resolve_saved_start_pose(self._graph, default=default_pose)

    @staticmethod
    def _resolve_configured_start_pose(
        default_pose: tuple[float, float, float] | None,
    ) -> tuple[float, float, float] | None:
        try:
            import config as cfg
        except ImportError:
            return None

        x0 = getattr(cfg, "TRACKING_START_X", None)
        y0 = getattr(cfg, "TRACKING_START_Y", None)
        yaw0 = getattr(cfg, "TRACKING_START_YAW_RAD", None)
        yaw0_deg = getattr(cfg, "TRACKING_START_YAW_DEG", None)
        if x0 is None or y0 is None:
            return None
        if yaw0 is None and yaw0_deg is not None:
            yaw0 = math.radians(float(yaw0_deg))
        if yaw0 is None and default_pose is not None:
            yaw0 = float(default_pose[2])
        if yaw0 is None:
            yaw0 = 0.0
        try:
            return float(x0), float(y0), float(yaw0)
        except (TypeError, ValueError):
            return None

    def _apply_start_pose_override(self) -> None:
        if self._dr is None or self._startup_world_pose is None:
            return
        x0, y0, yaw0 = self._startup_world_pose
        self._dr.reset(float(x0), float(y0), float(yaw0))
        self._reset_imu_yaw_alignment(
            float(yaw0),
            source="startup_pose",
            settle_s=0.0,
        )

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
        self._reset_imu_yaw_alignment(
            float(yaw_rad),
            source=f"sim_relocalize:{source}",
            settle_s=_SIM_RELOCALIZE_IMU_SETTLE_S,
        )
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

    def _reset_imu_yaw_alignment(
        self,
        yaw_rad: float,
        *,
        source: str,
        settle_s: float = 0.15,
    ) -> None:
        """Force the next fresh IMU sample to define the map-frame yaw offset."""
        now = time.monotonic()
        self._start_yaw_rad = float(yaw_rad)
        self._last_yaw_rad = float(yaw_rad)
        self._yaw_offset = 0.0
        self._yaw_offset_calibrated = False
        self._last_imu_t = None
        self._last_absolute_yaw_fix_monotonic = now
        self._last_absolute_yaw_fix_source = str(source)
        if float(settle_s) > 0.0:
            self._ignore_imu_until_monotonic = max(
                float(getattr(self, "_ignore_imu_until_monotonic", 0.0) or 0.0),
                now + float(settle_s),
            )
        live_log(
            "pose_estimator",
            event="imu_yaw_alignment_reset",
            source=str(source),
            yaw_rad=float(yaw_rad),
            yaw_deg=float(math.degrees(float(yaw_rad))),
            ignore_imu_until_mono=float(getattr(self, "_ignore_imu_until_monotonic", 0.0) or 0.0),
        )

    def _send_sim_relocalize(self) -> None:
        """In sim mode, teleport the Gazebo car to the startup pose.

        Repetimos el envío persistentemente porque ZMQ PUB/SUB pierde mensajes
        si el SUB no está listo cuando se publica. El sim_bridge tarda hasta
        ~12s en estar listo (timeout del GPS gate). Enviamos cada 0.5s durante
        20s, deteniéndonos en cuanto detectemos que la pose GT del sim
        coincide con el target (= set_pose fue aplicado).
        """
        if self._graph is None:
            return
        pose = self._startup_world_pose or self._graph.get_start_pose()
        x0, y0, yaw0 = pose

        def _resend_loop():
            import time as _time
            # Loop persistente: cada 0.5s durante 20s o hasta que la pose
            # fusionada esté cerca del target (el sim aplicó el teleport).
            for tick in range(40):  # 40 * 0.5s = 20s max
                try:
                    self._send_sim_relocalize_pose(
                        x0,
                        y0,
                        yaw0,
                        source=f"startup pose (retry#{tick})" if tick > 0 else "startup pose",
                    )
                except Exception:
                    pass
                _time.sleep(0.5)
                # Stop si la pose del brain ya está sincronizada con el target.
                # `dr` actualiza después del primer set_pose exitoso.
                dr = getattr(self, "_dr", None)
                if dr is not None and tick >= 4:
                    try:
                        cur_x, cur_y, _ = dr.snapshot()
                        if abs(float(cur_x) - float(x0)) < 0.5 and abs(float(cur_y) - float(y0)) < 0.5:
                            # Pose ya está cerca del target — el sim aplicó.
                            break
                    except Exception:
                        pass

        import threading as _threading
        _threading.Thread(target=_resend_loop, daemon=True).start()

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
        if (
            bool(_VISUAL_MAP_MATCH_ENABLED)
            and bool(getattr(route_context, "route_active", False))
            and lane_observation_has_visual_path(
                lane_observation,
                min_quality=float(getattr(_config, "LANE_VISUAL_MIN_QUALITY_FOR_PRIMARY_PATH", 0.55) or 0.55),
                min_points=int(getattr(_config, "LANE_VISUAL_MIN_POLY_POINTS", 8) or 8),
            )
        ):
            return raw_x, raw_y, raw_yaw, 0.0, True
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

    def _apply_visual_lane_match_update(
        self,
        route_context: RouteContext | None,
        visual_lane_match,
        now: float,
    ) -> tuple[float, float, float, float, float, bool]:
        if self._dr is None or route_context is None or visual_lane_match is None:
            raw_x, raw_y, raw_yaw = self._dr.get_state() if self._dr is not None else (0.0, 0.0, 0.0)
            return raw_x, raw_y, raw_yaw, 0.0, 0.0, False
        if not bool(getattr(visual_lane_match, "accepted", False)):
            raw_x, raw_y, raw_yaw = self._dr.get_state()
            return raw_x, raw_y, raw_yaw, 0.0, 0.0, False
        if (now - getattr(self, "_last_visual_map_match_correction_monotonic", 0.0)) < float(
            _VISUAL_MAP_MATCH_CORRECTION_COOLDOWN_S
        ):
            raw_x, raw_y, raw_yaw = self._dr.get_state()
            return raw_x, raw_y, raw_yaw, 0.0, 0.0, True

        lateral_error_m = float(getattr(visual_lane_match, "lateral_error_m", 0.0) or 0.0)
        yaw_error_rad = float(getattr(visual_lane_match, "yaw_error_rad", 0.0) or 0.0)
        lateral_correction_m = lateral_error_m * float(_VISUAL_MAP_MATCH_CORRECTION_GAIN)
        max_step_m = abs(float(_VISUAL_MAP_MATCH_CORRECTION_STEP_MAX_M))
        if max_step_m > 0.0:
            lateral_correction_m = max(-max_step_m, min(max_step_m, lateral_correction_m))
        yaw_correction_rad = yaw_error_rad * float(_VISUAL_MAP_MATCH_YAW_CORRECTION_GAIN)
        max_yaw_step = abs(float(_VISUAL_MAP_MATCH_YAW_CORRECTION_STEP_MAX_RAD))
        if max_yaw_step > 0.0:
            yaw_correction_rad = max(-max_yaw_step, min(max_yaw_step, yaw_correction_rad))

        applied = False
        if abs(lateral_correction_m) > 1e-9:
            self._dr.correct_lateral(lateral_correction_m, float(route_context.matched_pose.yaw))
            applied = True
        if abs(yaw_correction_rad) > 1e-9:
            self._dr.correct_yaw(yaw_correction_rad)
            applied = True
        if applied:
            self._last_visual_map_match_correction_monotonic = float(now)
        raw_x, raw_y, raw_yaw = self._dr.get_state()
        return (
            raw_x,
            raw_y,
            raw_yaw,
            float(lateral_correction_m if applied else 0.0),
            float(yaw_correction_rad if applied else 0.0),
            True,
        )

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

    @staticmethod
    def _localisation_is_manual(payload: dict, meta: dict) -> bool:
        source = str(meta.get("source") or payload.get("source") or "").lower()
        return bool(meta.get("manual")) or source.startswith("manual")

    @staticmethod
    def _median(values: list[float]) -> float:
        ordered = sorted(float(v) for v in values)
        n = len(ordered)
        mid = n // 2
        if n % 2:
            return float(ordered[mid])
        return 0.5 * (float(ordered[mid - 1]) + float(ordered[mid]))

    def _validate_gps_fix(
        self,
        *,
        x: float,
        y: float,
        expected_x: float,
        expected_y: float,
    ) -> tuple[bool, str, float, float]:
        if not _GPS_VALIDATION_ENABLED:
            return True, "validation_disabled", float(x), float(y)

        x = float(x)
        y = float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            return False, "non_finite", x, y
        if math.hypot(x, y) <= float(_GPS_ZERO_EPS_M):
            return False, "zero_fix", x, y

        samples = list(getattr(self, "_gps_fix_samples", []) or [])
        samples.append((x, y))
        max_window = max(1, int(_GPS_SAMPLE_WINDOW))
        samples = samples[-max_window:]
        self._gps_fix_samples = samples

        min_samples = max(1, int(_GPS_MIN_SAMPLES))
        if len(samples) < min_samples:
            return False, "collecting_samples", x, y

        median_x = self._median([sx for sx, _ in samples])
        median_y = self._median([sy for _, sy in samples])
        inliers = [
            (sx, sy)
            for sx, sy in samples
            if math.hypot(float(sx) - median_x, float(sy) - median_y)
            <= float(_GPS_OUTLIER_DISTANCE_M)
        ]
        if len(inliers) < min_samples:
            return False, "outlier_window", x, y

        filtered_x = sum(sx for sx, _ in inliers) / float(len(inliers))
        filtered_y = sum(sy for _, sy in inliers) / float(len(inliers))

        last_accepted = getattr(self, "_last_accepted_gps_pose", None)
        if last_accepted is not None:
            jump_m = math.hypot(
                filtered_x - float(last_accepted[0]),
                filtered_y - float(last_accepted[1]),
            )
            if jump_m > float(_GPS_MAX_JUMP_M):
                return False, "large_jump", filtered_x, filtered_y

        expected_error_m = math.hypot(filtered_x - float(expected_x), filtered_y - float(expected_y))
        if (
            float(_GPS_MAX_EXPECTED_ERROR_M) > 0.0
            and expected_error_m > float(_GPS_MAX_EXPECTED_ERROR_M)
        ):
            self._last_gps_fix_quality = 0.0
            return False, "expected_pose_mismatch", filtered_x, filtered_y

        self._last_accepted_gps_pose = (float(filtered_x), float(filtered_y))
        quality_den = max(float(_GPS_MAX_EXPECTED_ERROR_M), 1e-6)
        self._last_gps_fix_quality = max(0.0, min(1.0, 1.0 - (expected_error_m / quality_den)))
        return True, "accepted", float(filtered_x), float(filtered_y)

    @staticmethod
    def _localisation_has_explicit_yaw(payload: dict) -> bool:
        return payload.get("yaw_rad") is not None or payload.get("yaw_deg") is not None

    def _apply_gps_soft_update(
        self,
        *,
        gps_x: float,
        gps_y: float,
        gps_yaw: float,
        current_yaw: float,
        route_context: RouteContext | None,
        lane_observation: LaneObservation | None,
        source: str,
        explicit_yaw: bool,
    ) -> tuple[bool, dict]:
        old_x, old_y, old_yaw = self._dr.get_state()
        dx = float(gps_x) - float(old_x)
        dy = float(gps_y) - float(old_y)
        raw_error_m = math.hypot(dx, dy)
        ref_yaw = _route_reference_yaw(route_context, current_yaw)
        tx = math.cos(ref_yaw)
        ty = math.sin(ref_yaw)
        nx = math.cos(ref_yaw + math.pi / 2.0)
        ny = math.sin(ref_yaw + math.pi / 2.0)
        longitudinal_m = (dx * tx) + (dy * ty)
        lateral_m = (dx * nx) + (dy * ny)

        visual_protected = _lane_observation_has_local_authority(lane_observation)
        lateral_gain = float(_GPS_SOFT_LATERAL_GAIN)
        lateral_blocked = False
        if visual_protected and abs(lateral_m) >= float(_GPS_VISUAL_LATERAL_BLOCK_M):
            lateral_update_m = 0.0
            lateral_blocked = True
        else:
            lateral_update_m = lateral_m * lateral_gain
        longitudinal_update_m = longitudinal_m

        gain = max(0.0, min(1.0, float(_GPS_SOFT_GAIN)))
        step_dx = gain * ((longitudinal_update_m * tx) + (lateral_update_m * nx))
        step_dy = gain * ((longitudinal_update_m * ty) + (lateral_update_m * ny))
        max_step_m = float(_GPS_RECOVERY_MAX_STEP_M if raw_error_m >= float(_GPS_RECOVERY_ERROR_M) else _GPS_SOFT_MAX_STEP_M)
        step_norm = math.hypot(step_dx, step_dy)
        if max_step_m > 0.0 and step_norm > max_step_m:
            scale = max_step_m / max(step_norm, 1e-9)
            step_dx *= scale
            step_dy *= scale
            step_norm = max_step_m

        new_yaw = float(old_yaw)
        yaw_update_rad = 0.0
        # GPS-derived heading: si tengo un anchor previo y la distancia
        # recorrida es >0.3m (señal>noise=15cm × 2), atan2(dy_gps, dx_gps)
        # es el yaw real del coche. Eso corrige el drift del IMU.
        gps_heading_anchor = getattr(self, "_gps_heading_anchor", None)
        if gps_heading_anchor is not None:
            ax, ay = gps_heading_anchor
            anchor_dx = float(gps_x) - float(ax)
            anchor_dy = float(gps_y) - float(ay)
            anchor_dist = math.hypot(anchor_dx, anchor_dy)
            if anchor_dist >= 0.30:  # 2× GPS noise (15cm)
                gps_heading = math.atan2(anchor_dy, anchor_dx)
                yaw_delta = _wrap_angle(gps_heading - float(old_yaw))
                yaw_update_rad = max(
                    -math.radians(15.0),
                    min(math.radians(15.0), 0.6 * yaw_delta),
                )
                new_yaw = _wrap_angle(float(old_yaw) + yaw_update_rad)
                self._gps_heading_anchor = (float(gps_x), float(gps_y))
                self._gps_heading_last_yaw = gps_heading
        else:
            # Primera vez: set anchor inicial
            self._gps_heading_anchor = (float(gps_x), float(gps_y))

        if explicit_yaw and not visual_protected and abs(yaw_update_rad) < 1e-9:
            yaw_delta = _wrap_angle(float(gps_yaw) - float(old_yaw))
            yaw_update_rad = max(
                -math.radians(2.0),
                min(math.radians(2.0), 0.15 * yaw_delta),
            )
            new_yaw = _wrap_angle(float(old_yaw) + yaw_update_rad)

        if step_norm > 1e-9 or abs(yaw_update_rad) > 1e-9:
            self._dr.reset(float(old_x) + step_dx, float(old_y) + step_dy, new_yaw)
            self._last_yaw_rad = new_yaw

        mode = "gps_recovery_soft" if raw_error_m >= float(_GPS_RECOVERY_ERROR_M) else "gps_soft"
        if lateral_blocked:
            mode = f"{mode}_visual_lateral_blocked"
        return True, {
            "mode": mode,
            "source": source,
            "error_m": float(raw_error_m),
            "applied_step_m": float(step_norm),
            "lateral_error_m": float(lateral_m),
            "longitudinal_error_m": float(longitudinal_m),
            "visual_lateral_blocked": bool(lateral_blocked),
            "hard_reset": False,
        }

    def _apply_localisation_fix(
        self,
        current_yaw: float,
        route_context: RouteContext | None = None,
        lane_observation: LaneObservation | None = None,
    ):
        if self._dr is None or self._graph is None:
            return False, None
        payload = self._localisation_fix_sub.receive()
        if not isinstance(payload, dict):
            return False, None

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        is_manual = self._localisation_is_manual(payload, meta)
        if not is_manual and not self._use_gps_for_localization():
            live_log(
                "pose_estimator",
                event="gps_fix_rejected",
                reason="gps_disabled",
            )
            return False, None

        old_x, old_y, _ = self._dr.get_state()
        pose = self._graph.localisation_to_world_pose(payload, default_yaw=current_yaw)
        if pose is None:
            return False, None

        x, y, yaw = pose
        if not is_manual:
            accepted, reason, filtered_x, filtered_y = self._validate_gps_fix(
                x=float(x),
                y=float(y),
                expected_x=float(old_x),
                expected_y=float(old_y),
            )
            if not accepted:
                live_log(
                    "pose_estimator",
                    event="gps_fix_rejected",
                    reason=str(reason),
                    gps_x=float(filtered_x),
                    gps_y=float(filtered_y),
                    expected_x=float(old_x),
                    expected_y=float(old_y),
                    error_m=float(math.hypot(float(filtered_x) - float(old_x), float(filtered_y) - float(old_y))),
                )
                return False, None
            x, y = float(filtered_x), float(filtered_y)

        default_source = "manual_localisation" if bool(meta.get("manual")) else "gps_localisation"
        source = str(meta.get("source") or payload.get("source") or default_source)
        resolved_node_id = self._graph.resolve_node_id(meta.get("node_id") or payload.get("node_id"))
        if resolved_node_id is not None:
            source = f"{source}:{resolved_node_id}"
        if bool(meta.get("manual")):
            self._send_sim_relocalize_pose(float(x), float(y), float(yaw), source=source)

        authority = str(_LOCALIZATION_GPS_AUTHORITY or "init_recovery_soft").strip().lower()
        if not is_manual and authority not in {"hard", "reset", "authoritative"}:
            return self._apply_gps_soft_update(
                gps_x=float(x),
                gps_y=float(y),
                gps_yaw=float(yaw),
                current_yaw=float(current_yaw),
                route_context=route_context,
                lane_observation=lane_observation,
                source=source,
                explicit_yaw=self._localisation_has_explicit_yaw(payload),
            )

        self._dr.reset(float(x), float(y), float(yaw))
        self._yaw_ekf_p = _YAW_EKF_P_INIT
        if payload.get("yaw_rad") is not None or payload.get("yaw_deg") is not None:
            self._reset_imu_yaw_alignment(float(yaw), source=source, settle_s=0.20)
        else:
            self._last_yaw_rad = float(yaw)
        if self.tracking_state is not None and hasattr(self.tracking_state, "set_lane_measurement_state"):
            self.tracking_state.set_lane_measurement_state(False, 0.0)
        error_m = math.hypot(float(x) - float(old_x), float(y) - float(old_y))
        return True, {
            "mode": "manual_fix" if is_manual else "gps_fix",
            "source": source,
            "error_m": float(error_m),
            "hard_reset": True,
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
        lane_observation: LaneObservation | None = None,
        gps_match: dict | None = None,
        visual_lane_match=None,
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
        visual_has_path = lane_observation_has_visual_path(
            lane_observation,
            min_quality=float(getattr(_config, "LANE_VISUAL_MIN_QUALITY_FOR_PRIMARY_PATH", 0.55) or 0.55),
            min_points=int(getattr(_config, "LANE_VISUAL_MIN_POLY_POINTS", 8) or 8),
        )
        lane_bonus = 0.35 if visual_has_path else (0.2 if lane_measurement_reliable else 0.0)
        localization_confidence = max(0.0, min(1.0, route_conf + lane_bonus))
        gps_applied = isinstance(gps_match, dict) and str(gps_match.get("mode") or "").startswith("gps")
        if visual_has_path:
            localization_mode = "VISUAL_PRIMARY"
        elif gps_applied and float(gps_match.get("error_m") or 0.0) >= float(_GPS_RECOVERY_ERROR_M):
            localization_mode = "GPS_RECOVERY"
        elif localization_confidence < 0.35:
            localization_mode = "DEGRADED"
        else:
            localization_mode = "OK"
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
            localization_mode=str(localization_mode),
            gps_fix_quality=float(getattr(self, "_last_gps_fix_quality", 0.0) or 0.0),
            visual_lane_match=visual_lane_match,
        )

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
                yaw_deg = float(imu_dict.get("yaw", math.degrees(self._last_yaw_rad)))
                yaw_raw_rad = _IMU_YAW_SIGN * math.radians(yaw_deg)
                ignore_until = float(getattr(self, "_ignore_imu_until_monotonic", 0.0) or 0.0)
                if now < ignore_until:
                    pass
                elif not self._yaw_offset_calibrated:
                    self._yaw_offset = self._start_yaw_rad - yaw_raw_rad
                    self._yaw_offset_calibrated = True
                    self._last_imu_t = now
                    live_log(
                        "pose_estimator",
                        event="imu_yaw_calibrated",
                        raw_yaw_deg=float(yaw_deg),
                        start_yaw_deg=float(math.degrees(self._start_yaw_rad)),
                        offset_deg=float(math.degrees(self._yaw_offset)),
                    )
                elif not self._has_fresh_absolute_yaw_fix(now):
                    prev_imu_t = self._last_imu_t
                    self._last_imu_t = now
                    # EKF correction: fuse IMU absolute heading with kinematic prediction.
                    # R scales with steer²: large steer → servo EMI biases magnetometer
                    # → kinematic model trusted more. Smooth transition, no hard cutoff.
                    yaw_imu = yaw_raw_rad + self._yaw_offset
                    use_encoder = self._use_encoder_for_localization()
                    if not use_encoder:
                        self._last_yaw_rad = yaw_imu
                        self._yaw_ekf_p = _YAW_EKF_P_INIT
                    else:
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
        use_encoder = self._use_encoder_for_localization()
        speed_for_dr = (
            float(self._last_speed) * float(_DR_SPEED_SCALE)
            if use_encoder else float(self._last_speed)
        )
        steer_for_dr = float(eff_steer_rad) if use_encoder else 0.0
        self._dr.update(
            speed_for_dr,
            self._last_yaw_rad,
            dr_dt,
            steer_rad=steer_for_dr,
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

        gps_relocalized, gps_match = self._apply_localisation_fix(
            raw_yaw,
            route_context=route_context,
            lane_observation=lane_observation,
        )
        gps_hard_reset = bool(isinstance(gps_match, dict) and gps_match.get("hard_reset"))
        if gps_hard_reset:
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
            if gps_relocalized:
                raw_x, raw_y, raw_yaw = self._dr.get_state()
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

        visual_lane_match = match_visual_lane_to_route(
            route_context,
            lane_observation,
            Pose2D(float(raw_x), float(raw_y), float(raw_yaw)),
            enabled=bool(_VISUAL_MAP_MATCH_ENABLED),
            min_quality=float(getattr(_config, "LANE_VISUAL_MIN_QUALITY_FOR_PRIMARY_PATH", 0.55) or 0.55),
            min_points=int(getattr(_config, "LANE_VISUAL_MIN_POLY_POINTS", 8) or 8),
            lane_half_width_m=float(_LANE_HALF_WIDTH_M),
            max_lateral_error_m=float(_VISUAL_MAP_MATCH_MAX_LATERAL_ERROR_M),
            max_yaw_error_rad=float(_VISUAL_MAP_MATCH_MAX_YAW_ERROR_RAD),
            max_sample_yaw_error_rad=float(_VISUAL_MAP_MATCH_MAX_SAMPLE_YAW_ERROR_RAD),
            min_confidence=float(_VISUAL_MAP_MATCH_MIN_CONFIDENCE),
        )
        (
            raw_x,
            raw_y,
            raw_yaw,
            visual_map_lateral_correction_m,
            visual_map_yaw_correction_rad,
            visual_map_match_update_available,
        ) = self._apply_visual_lane_match_update(route_context, visual_lane_match, now)
        if visual_map_match_update_available:
            fused_pose_for_match = Pose2D(float(raw_x), float(raw_y), float(raw_yaw))
            visual_lane_match = match_visual_lane_to_route(
                route_context,
                lane_observation,
                fused_pose_for_match,
                enabled=bool(_VISUAL_MAP_MATCH_ENABLED),
                min_quality=float(getattr(_config, "LANE_VISUAL_MIN_QUALITY_FOR_PRIMARY_PATH", 0.55) or 0.55),
                min_points=int(getattr(_config, "LANE_VISUAL_MIN_POLY_POINTS", 8) or 8),
                lane_half_width_m=float(_LANE_HALF_WIDTH_M),
                max_lateral_error_m=float(_VISUAL_MAP_MATCH_MAX_LATERAL_ERROR_M),
                max_yaw_error_rad=float(_VISUAL_MAP_MATCH_MAX_YAW_ERROR_RAD),
                max_sample_yaw_error_rad=float(_VISUAL_MAP_MATCH_MAX_SAMPLE_YAW_ERROR_RAD),
                min_confidence=float(_VISUAL_MAP_MATCH_MIN_CONFIDENCE),
            )
        raw_pose = Pose2D(float(raw_x), float(raw_y), float(raw_yaw))

        relocalization_mode = "dead_reckoning"
        relocalization_source = "dead_reckoning"
        relocalization_error_m = 0.0
        if gps_hard_reset and isinstance(gps_match, dict):
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
        elif abs(visual_map_lateral_correction_m) > 1e-9 or abs(visual_map_yaw_correction_rad) > 1e-9:
            relocalization_mode = "visual_lane_match"
            relocalization_source = "mini_yabloc_2d"
            relocalization_error_m = max(
                abs(float(visual_map_lateral_correction_m)),
                abs(float(visual_map_yaw_correction_rad)),
            )
        elif gps_relocalized and isinstance(gps_match, dict):
            relocalization_mode = str(gps_match.get("mode") or "gps_soft")
            relocalization_source = str(gps_match.get("source") or "gps_localisation")
            relocalization_error_m = float(gps_match.get("error_m") or 0.0)

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
            lane_observation=lane_observation,
            gps_match=gps_match if isinstance(gps_match, dict) else None,
            visual_lane_match=visual_lane_match,
            wall_timestamp=wall_now,
        )
        self.pose_estimate_buffer.write(pose_estimate, timestamp=pose_estimate.timestamp)
        if self.tracking_state is not None and hasattr(self.tracking_state, "update_from_pose_estimate"):
            self.tracking_state.update_from_pose_estimate(pose_estimate)

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
            localization_mode=str(pose_estimate.localization_mode),
            gps_fix_quality=float(pose_estimate.gps_fix_quality or 0.0),
            gps_mode=str(gps_match.get("mode", "none") if isinstance(gps_match, dict) else "none"),
            lane_correction_m=float(lane_relocalization_m or 0.0),
            lane_reliable=bool(lane_measurement_reliable),
            visual_lane_match_confidence=(
                float(visual_lane_match.confidence)
                if visual_lane_match is not None
                else None
            ),
            visual_lane_match_accepted=(
                bool(visual_lane_match.accepted)
                if visual_lane_match is not None
                else None
            ),
            visual_lane_match_lateral_error_m=(
                float(visual_lane_match.lateral_error_m)
                if visual_lane_match is not None
                else None
            ),
            visual_lane_match_yaw_error_rad=(
                float(visual_lane_match.yaw_error_rad)
                if visual_lane_match is not None
                else None
            ),
            visual_lane_match_near_yaw_error_rad=(
                float(visual_lane_match.near_yaw_error_rad)
                if visual_lane_match is not None
                else None
            ),
            visual_lane_match_max_abs_yaw_error_rad=(
                float(visual_lane_match.max_abs_yaw_error_rad)
                if visual_lane_match is not None
                else None
            ),
            visual_lane_match_reason=(
                str(visual_lane_match.reason)
                if visual_lane_match is not None
                else "none"
            ),
            visual_map_lateral_correction_m=float(visual_map_lateral_correction_m or 0.0),
            visual_map_yaw_correction_rad=float(visual_map_yaw_correction_rad or 0.0),
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
