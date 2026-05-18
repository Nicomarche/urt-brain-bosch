from __future__ import annotations

import ast
import math
import time
from collections import deque
from types import SimpleNamespace

from src.core.types import LaneObservation, Pose2D, PoseEstimate, RouteContext, StoplineObservation
from src.core.types.perception import lane_observation_supports_lateral_relocalization
from src.utils.live_log import live_log
from src.utils.gps_mode import gps_mode_path, is_gps_disabled, save_gps_disabled
from src.utils.sim_start_pose import resolve_saved_start_pose
from src.localization.relocalization_thread import (
    _CAMERA_LATERAL_CORRECTION_COOLDOWN_S,
    _IMU_YAW_SIGN,
    _ENCODER_CURVE_SPEED_COMPENSATION,
    _MAX_INTEGRATION_DT,
    _MAX_PHYSICAL_YAW_RATE_RADS,
    _REAR_AXLE_TO_CG_M,
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
    _YAW_EKF_Q,
    _YAW_EKF_P_INIT,
    _YAW_EKF_R_STEER_K,
    _YAW_EKF_R_STRAIGHT,
    threadTracking,
)
from src.localization.dead_reckoning import curve_speed_compensated
from src.localization.lane_correction import compute_lateral_correction
from src.localization.visual_correction_profile import (
    resolve_visual_correction_profile,
)
from src.core.bus.topics import LOCALISATION, ODO_RESET
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.core.messaging.header import SeqCounter, stamp as _stamp_header

_WAYPOINT_TWO_LINE_LATERAL_RELOCALIZATION_SPEED_MIN_MPS = 0.02
_SINGLE_LINE_ROUTE_STRAIGHT_TOL_RAD = math.radians(6.0)
_SINGLE_LINE_ROUTE_DIRECTION_MARGIN_RAD = math.radians(10.0)
_SINGLE_LINE_ROUTE_MAX_YAW_MISMATCH_RAD = math.radians(18.0)

# Política de single_line — alineada con urt-ref:
# El observador en modo single_line extrapola la línea faltante asumiendo
# ``lane_width=0.35 m``. Si el ancho real difiere o hay perspective drift,
# ``direct_error_m`` puede saturar en valores grandes y consistentes (28 cm
# observado en run_20260518_025658), arrastrando al ego fuera del carril.
#
# El repo de referencia (urt-ref/.../control/scripts/control_node2.py)
# directamente NO consume la cámara para mover el lazo de control
# (``subLane=False``); el MPC sigue waypoints del mapa. Adoptamos la misma
# postura para la CORRECCIÓN AL POSE: en single_line no corregimos. El
# observador sigue publicando la observación para el dashboard, pero el
# pose_estimator la descarta.
#
# Two_line sigue corrigiendo con la autoridad del perfil activo (lane_keep
# gain=0.30 / max=0.04). Cuando hay dos líneas, la medida es robusta y no
# depende del ``lane_width`` asumido.

try:
    import config as _cfg_auto
except Exception:  # pragma: no cover - isolated imports in tests
    _cfg_auto = None

_AUTO_GPS_SAMPLE_COUNT = max(1, int(getattr(_cfg_auto, "AUTO_GPS_SAMPLE_COUNT", 3)))
_AUTO_GPS_COLLECTION_TIMEOUT_S = float(getattr(_cfg_auto, "AUTO_GPS_COLLECTION_TIMEOUT_S", 2.0))
_AUTO_GPS_MAX_FIX_AGE_S = float(getattr(_cfg_auto, "AUTO_GPS_MAX_FIX_AGE_S", 1.0))
_AUTO_GPS_MAX_SPREAD_M = float(getattr(_cfg_auto, "AUTO_GPS_MAX_SPREAD_M", 0.35))
_AUTO_GPS_MAX_LANELET_DISTANCE_M = float(getattr(_cfg_auto, "AUTO_GPS_MAX_LANELET_DISTANCE_M", 0.50))
_AUTO_GPS_MAX_YAW_DIFF_RAD = float(getattr(_cfg_auto, "AUTO_GPS_MAX_YAW_DIFF_RAD", math.pi / 2.0))
_AUTO_GPS_STATUS_HOLD_S = 0.25
_START_GPS_CALIBRATION_SAMPLE_COUNT = max(1, int(getattr(_cfg_auto, "START_GPS_CALIBRATION_SAMPLE_COUNT", 5)))
_START_GPS_CALIBRATION_TIMEOUT_S = float(getattr(_cfg_auto, "START_GPS_CALIBRATION_TIMEOUT_S", 4.0))
_START_GPS_CALIBRATION_MAX_FIX_AGE_S = float(getattr(_cfg_auto, "START_GPS_CALIBRATION_MAX_FIX_AGE_S", 1.5))
_START_GPS_CALIBRATION_MAX_SPREAD_M = float(getattr(_cfg_auto, "START_GPS_CALIBRATION_MAX_SPREAD_M", _AUTO_GPS_MAX_SPREAD_M))
_START_GPS_CALIBRATION_STATUS_HOLD_S = float(getattr(_cfg_auto, "START_GPS_CALIBRATION_STATUS_HOLD_S", 4.0))
_IMU_STEER_TRUST_FULL_DEG = float(getattr(_cfg_auto, "TRACKING_IMU_STEER_TRUST_FULL_DEG", 3.0))
_IMU_STEER_TRUST_ZERO_DEG = float(getattr(_cfg_auto, "TRACKING_IMU_STEER_TRUST_ZERO_DEG", 8.0))
_IMU_STEER_RATE_TRUST_FULL_DEGS = float(getattr(_cfg_auto, "TRACKING_IMU_STEER_RATE_TRUST_FULL_DEGS", 20.0))
_IMU_STEER_RATE_TRUST_ZERO_DEGS = float(getattr(_cfg_auto, "TRACKING_IMU_STEER_RATE_TRUST_ZERO_DEGS", 90.0))
_IMU_STEER_INHIBIT_HOLD_S = float(getattr(_cfg_auto, "TRACKING_IMU_STEER_INHIBIT_HOLD_S", 0.35))
_LANE_YAW_RESET_ENABLED = bool(getattr(_cfg_auto, "TRACKING_LANE_YAW_RESET_ENABLED", True))
_LANE_YAW_RESET_CONSECUTIVE = max(1, int(getattr(_cfg_auto, "TRACKING_LANE_YAW_RESET_CONSECUTIVE", 2)))
_LANE_YAW_RESET_QUALITY_MIN = float(getattr(_cfg_auto, "TRACKING_LANE_YAW_RESET_QUALITY_MIN", 0.80))
_LANE_YAW_RESET_STRAIGHT_THRESH_RAD = math.radians(
    float(getattr(_cfg_auto, "TRACKING_LANE_YAW_RESET_STRAIGHT_THRESH_DEG", 1.1))
)
_LANE_YAW_RESET_MAX_ERROR_RAD = math.radians(
    float(getattr(_cfg_auto, "TRACKING_LANE_YAW_RESET_MAX_ERROR_DEG", 3.75))
)
_LANE_YAW_RESET_COOLDOWN_S = float(getattr(_cfg_auto, "TRACKING_LANE_YAW_RESET_COOLDOWN_S", 30.0))


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
        # LOCALISATION multiplexes dashboard manual fixes and LoCSys GPS fixes.
        # Keep FIFO here and drain/prioritize in _receive_localisation_fix_payload();
        # LastOnly can drop a manual relocate if a GPS fix arrives right after it.
        self._localisation_fix_sub = messageHandlerSubscriber(
            queuesList, LOCALISATION, "fifo", subscribe=True
        )
        self._odo_reset_sender = messageHandlerSender(queuesList, ODO_RESET)
        # Contador monotónico para detección de gaps en LOCATION (D2).
        self._seq_location = SeqCounter()
        print(
            f"[GPS-DBG] PoseEstimatorThread.__init__: _localisation_fix_sub "
            f"registered (topic={LOCALISATION.name})",
            flush=True,
        )
        self._last_gps_raw_xy: tuple[float, float] | None = None
        self._last_gps_raw_monotonic = 0.0
        self._gps_disabled = is_gps_disabled()
        self._gps_calibration_offset_xy: tuple[float, float] | None = None
        self._startup_gps_calibration_pending = False
        self._gps_fix_history = deque(maxlen=max(10, _AUTO_GPS_SAMPLE_COUNT * 3))
        self._auto_gps_pending = False
        self._auto_gps_started_monotonic = 0.0
        self._auto_gps_deadline_monotonic = 0.0
        self._auto_gps_last_mode: str | None = None
        self._auto_gps_last_source: str | None = None
        self._auto_gps_last_error_m = 0.0
        self._auto_gps_last_mode_monotonic = 0.0
        self._pending_yaw_offset_target_rad: float | None = None
        self._start_gps_calibration_pending = False
        self._start_gps_calibration_target_pose: tuple[float, float, float] | None = None
        self._start_gps_calibration_samples = deque(
            maxlen=max(_START_GPS_CALIBRATION_SAMPLE_COUNT * 2, _START_GPS_CALIBRATION_SAMPLE_COUNT)
        )
        self._start_gps_calibration_deadline_monotonic = 0.0
        self._start_gps_calibration_source: str | None = None
        self._start_gps_calibration_last_mode: str | None = None
        self._start_gps_calibration_last_source: str | None = None
        self._start_gps_calibration_last_error_m = 0.0
        self._start_gps_calibration_last_sample_count = 0
        self._start_gps_calibration_last_mode_monotonic = 0.0
        self._startup_world_pose = self._resolve_initial_world_pose()
        self._startup_gps_calibration_pending = (
            self._startup_world_pose is not None
            and self._gps_dashboard_calibration_enabled()
            and not self._gps_runtime_disabled()
        )
        self._last_yaw_fusion_steer_rad = 0.0
        self._imu_yaw_inhibit_until = 0.0
        self._last_imu_yaw_raw_deg = None
        self._last_yaw_used_rad = float(getattr(self, "_last_yaw_rad", 0.0) or 0.0)
        self._last_yaw_kinematic_rad = self._last_yaw_used_rad
        self._last_imu_yaw_limited_rad = None
        self._last_imu_yaw_gain = 0.0
        self._last_yaw_rejected = False
        self._lane_yaw_reset_count = 0
        self._lane_yaw_reset_next_t = 0.0
        self._last_lane_yaw_reset_deg = 0.0
        self._apply_start_pose_override()
        self._send_sim_relocalize()

    @staticmethod
    def _gps_dashboard_calibration_enabled() -> bool:
        try:
            from config import MOTOR_OUTPUT
        except ImportError:
            return True
        return MOTOR_OUTPUT != "zmq"

    def _gps_runtime_disabled(self) -> bool:
        runtime_disabled = bool(getattr(self, "_gps_disabled", False))
        try:
            path = gps_mode_path()
            if path.exists():
                disabled = bool(is_gps_disabled())
                self._gps_disabled = disabled
                return self._gps_disabled
        except Exception:
            pass
        self._gps_disabled = runtime_disabled
        return self._gps_disabled

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

    @staticmethod
    def _is_manual_localisation_payload(payload: dict) -> bool:
        if not isinstance(payload, dict):
            return False
        meta = payload.get("meta")
        if isinstance(meta, dict):
            if bool(meta.get("manual")):
                return True
            source = str(meta.get("source") or "").strip().lower()
            if source.startswith("manual"):
                return True
        source = str(payload.get("source") or "").strip().lower()
        return source.startswith("manual")

    def _receive_localisation_fix_payload(self):
        subscriber = getattr(self, "_localisation_fix_sub", None)
        if subscriber is None:
            return None
        payload = subscriber.receive()
        if payload is None:
            return None

        latest_payload = payload
        latest_manual_payload = (
            payload if self._is_manual_localisation_payload(payload) else None
        )

        has_more = getattr(subscriber, "is_data_in_pipe", None)
        if not callable(has_more):
            return latest_manual_payload or latest_payload

        for _ in range(50):
            try:
                if not bool(has_more()):
                    break
            except Exception:
                break
            next_payload = subscriber.receive()
            if next_payload is None:
                break
            latest_payload = next_payload
            if self._is_manual_localisation_payload(next_payload):
                latest_manual_payload = next_payload

        return latest_manual_payload or latest_payload

    def _remember_gps_raw_fix(self, payload: dict) -> None:
        xy = self._payload_xy(payload)
        if xy is None:
            return
        self._last_gps_raw_xy = xy
        self._last_gps_raw_monotonic = time.monotonic()

    def _consume_state_change(self) -> tuple[str, str] | None:
        transition = super()._consume_state_change()
        if transition is None:
            return None
        previous_state, state_name = transition
        if state_name == "AUTO" and previous_state != "AUTO":
            self._begin_auto_gps_entry()
        return transition

    def _begin_auto_gps_entry(self) -> None:
        now = time.monotonic()
        gps_disabled = self._gps_runtime_disabled()
        self._auto_gps_pending = not gps_disabled
        self._auto_gps_started_monotonic = now
        self._auto_gps_deadline_monotonic = now + max(0.0, _AUTO_GPS_COLLECTION_TIMEOUT_S)
        self._auto_gps_last_mode = None
        self._auto_gps_last_source = None
        self._auto_gps_last_error_m = 0.0
        self._auto_gps_last_mode_monotonic = 0.0
        try:
            self._gps_fix_history.clear()
        except Exception:
            self._gps_fix_history = deque(maxlen=max(10, _AUTO_GPS_SAMPLE_COUNT * 3))
        try:
            self._localisation_fix_sub.empty()
        except Exception:
            pass
        try:
            self._odo_reset_sender.send("1")
        except Exception:
            if self.logging is not None:
                self.logging.exception("failed to send ODO_RESET on AUTO entry")
        if gps_disabled:
            self._finish_auto_gps_entry(mode="auto_gps_disabled", source="no_gps_mode")
            live_log(
                "pose_estimator",
                event="auto_gps_entry_skipped",
                reason="no_gps_mode",
            )
            return
        live_log(
            "pose_estimator",
            event="auto_gps_entry_started",
            sample_count=int(_AUTO_GPS_SAMPLE_COUNT),
            timeout_s=float(_AUTO_GPS_COLLECTION_TIMEOUT_S),
        )

    def _finish_auto_gps_entry(self, *, mode: str, source: str, error_m: float = 0.0) -> None:
        self._auto_gps_pending = False
        self._auto_gps_last_mode = str(mode)
        self._auto_gps_last_source = str(source)
        self._auto_gps_last_error_m = float(error_m or 0.0)
        self._auto_gps_last_mode_monotonic = time.monotonic()
        live_log(
            "pose_estimator",
            event="auto_gps_entry_finished",
            mode=str(mode),
            source=str(source),
            error_m=float(error_m or 0.0),
        )

    def _auto_gps_recent_status(self, now: float) -> tuple[str, str, float] | None:
        if self._auto_gps_pending:
            return "auto_gps_entry_pending", "gps_waiting_for_samples", 0.0
        last_t = float(getattr(self, "_auto_gps_last_mode_monotonic", 0.0) or 0.0)
        mode = getattr(self, "_auto_gps_last_mode", None)
        if mode and now - last_t <= _AUTO_GPS_STATUS_HOLD_S:
            return (
                str(mode),
                str(getattr(self, "_auto_gps_last_source", None) or "auto_gps_entry"),
                float(getattr(self, "_auto_gps_last_error_m", 0.0) or 0.0),
            )
        return None

    @staticmethod
    def _meta_marks_gps_out_of_bounds(payload: dict, meta: dict) -> bool:
        for source in (payload, meta):
            if not isinstance(source, dict):
                continue
            value = source.get("gps_frame_in_bounds")
            if value is False:
                return True
        return False

    def _remember_auto_gps_fix(self, payload: dict, meta: dict, now: float) -> None:
        xy = self._payload_xy(payload)
        if xy is None:
            return
        x, y = xy
        if not (math.isfinite(float(x)) and math.isfinite(float(y))):
            return
        if self._meta_marks_gps_out_of_bounds(payload, meta):
            return
        ts = payload.get("timestamp")
        if ts is not None:
            try:
                ts_f = float(ts)
            except (TypeError, ValueError):
                ts_f = 0.0
            if ts_f > 0.0 and (time.time() - ts_f) > _AUTO_GPS_MAX_FIX_AGE_S:
                return
        history = getattr(self, "_gps_fix_history", None)
        if history is None:
            self._gps_fix_history = deque(maxlen=max(10, _AUTO_GPS_SAMPLE_COUNT * 3))
            history = self._gps_fix_history
        history.append({
            "x": float(x),
            "y": float(y),
            "received_monotonic": float(now),
            "payload": dict(payload),
            "meta": dict(meta),
        })

    def _recent_auto_gps_samples(self, now: float) -> list[dict]:
        history = list(getattr(self, "_gps_fix_history", []) or [])
        fresh = [
            sample
            for sample in history
            if now - float(sample.get("received_monotonic", 0.0)) <= _AUTO_GPS_MAX_FIX_AGE_S
        ]
        return fresh[-_AUTO_GPS_SAMPLE_COUNT:]

    @staticmethod
    def _gps_sample_spread(samples: list[dict]) -> float:
        if len(samples) <= 1:
            return 0.0
        max_spread = 0.0
        for i, first in enumerate(samples):
            for second in samples[i + 1:]:
                max_spread = max(
                    max_spread,
                    math.hypot(float(first["x"]) - float(second["x"]), float(first["y"]) - float(second["y"])),
                )
        return float(max_spread)

    @staticmethod
    def _lanelet_point_and_yaw(lanelet, arc_m: float) -> tuple[float, float, float] | None:
        cl = getattr(lanelet, "centerline", None)
        if cl is None or getattr(cl, "shape", (0,))[0] < 1:
            return None
        if cl.shape[0] == 1:
            return float(cl[0, 0]), float(cl[0, 1]), 0.0
        target_arc = max(0.0, min(float(arc_m), float(getattr(lanelet, "length_m", arc_m) or arc_m)))
        cumulative = 0.0
        for idx in range(cl.shape[0] - 1):
            ax, ay = float(cl[idx, 0]), float(cl[idx, 1])
            bx, by = float(cl[idx + 1, 0]), float(cl[idx + 1, 1])
            seg_len = math.hypot(bx - ax, by - ay)
            if seg_len <= 1e-9:
                continue
            if cumulative + seg_len >= target_arc:
                ratio = (target_arc - cumulative) / seg_len
                x = ax + ratio * (bx - ax)
                y = ay + ratio * (by - ay)
                return float(x), float(y), math.atan2(by - ay, bx - ax)
            cumulative += seg_len
        ax, ay = float(cl[-2, 0]), float(cl[-2, 1])
        bx, by = float(cl[-1, 0]), float(cl[-1, 1])
        return bx, by, math.atan2(by - ay, bx - ax)

    def _try_apply_auto_gps_relocalization(self, current_yaw: float, now: float):
        if not bool(getattr(self, "_auto_gps_pending", False)):
            return False, None
        samples = self._recent_auto_gps_samples(now)
        deadline_expired = now > float(getattr(self, "_auto_gps_deadline_monotonic", 0.0) or 0.0)
        if len(samples) < _AUTO_GPS_SAMPLE_COUNT:
            if deadline_expired:
                self._finish_auto_gps_entry(mode="auto_gps_unavailable", source="not_enough_fresh_gps_samples")
            return False, None

        spread_m = self._gps_sample_spread(samples)
        if spread_m > _AUTO_GPS_MAX_SPREAD_M:
            if deadline_expired:
                self._finish_auto_gps_entry(mode="auto_gps_unavailable", source="gps_spread_too_large", error_m=spread_m)
            return False, None

        avg_x = sum(float(sample["x"]) for sample in samples) / float(len(samples))
        avg_y = sum(float(sample["y"]) for sample in samples) / float(len(samples))
        lanelet_map = getattr(getattr(self, "_graph", None), "lanelet_map", None)
        lanelet_id = None
        if lanelet_map is not None and hasattr(lanelet_map, "at_pose"):
            lanelet_id = lanelet_map.at_pose(
                avg_x,
                avg_y,
                max_distance_m=_AUTO_GPS_MAX_LANELET_DISTANCE_M,
                yaw_rad=float(current_yaw),
                max_yaw_diff_rad=_AUTO_GPS_MAX_YAW_DIFF_RAD,
            )
        lanelet = lanelet_map.get_lanelet(lanelet_id) if lanelet_map is not None and lanelet_id is not None else None
        if lanelet is None:
            self._finish_auto_gps_entry(mode="auto_gps_lanelet_unmatched", source="no_yaw_compatible_lanelet")
            return False, None

        old_x, old_y, _ = self._dr.get_state()
        arc_m, lateral_m = lanelet.project_arclength(avg_x, avg_y)
        snapped = self._lanelet_point_and_yaw(lanelet, arc_m)
        if snapped is None:
            self._finish_auto_gps_entry(mode="auto_gps_lanelet_unmatched", source="lanelet_projection_failed")
            return False, None
        snap_x, snap_y, snap_yaw = snapped
        self._dr.reset(float(snap_x), float(snap_y), float(snap_yaw))
        self._last_yaw_rad = float(snap_yaw)
        self._yaw_ekf_p = _YAW_EKF_P_INIT
        self._last_absolute_yaw_fix_monotonic = float(now)
        self._last_absolute_yaw_fix_source = "auto_gps_entry"
        self._calibrate_imu_yaw_offset_to(float(snap_yaw))
        if self.tracking_state is not None and hasattr(self.tracking_state, "set_lane_measurement_state"):
            self.tracking_state.set_lane_measurement_state(False, 0.0)
        source = f"auto_gps_entry:{lanelet_id}"
        error_m = math.hypot(float(snap_x) - float(old_x), float(snap_y) - float(old_y))
        self._send_sim_relocalize_pose(float(snap_x), float(snap_y), float(snap_yaw), source=source)
        self._finish_auto_gps_entry(mode="auto_gps_entry", source=source, error_m=error_m)
        return True, {
            "mode": "auto_gps_entry",
            "source": source,
            "error_m": float(error_m),
            "gps_spread_m": float(spread_m),
            "gps_lateral_m": float(lateral_m),
        }

    def _ensure_start_gps_calibration_state(self) -> None:
        if not hasattr(self, "_start_gps_calibration_samples"):
            self._start_gps_calibration_samples = deque(
                maxlen=max(
                    _START_GPS_CALIBRATION_SAMPLE_COUNT * 2,
                    _START_GPS_CALIBRATION_SAMPLE_COUNT,
                )
            )
        if not hasattr(self, "_start_gps_calibration_pending"):
            self._start_gps_calibration_pending = False
        if not hasattr(self, "_start_gps_calibration_target_pose"):
            self._start_gps_calibration_target_pose = None
        if not hasattr(self, "_start_gps_calibration_deadline_monotonic"):
            self._start_gps_calibration_deadline_monotonic = 0.0
        if not hasattr(self, "_start_gps_calibration_source"):
            self._start_gps_calibration_source = None
        if not hasattr(self, "_start_gps_calibration_last_mode"):
            self._start_gps_calibration_last_mode = None
            self._start_gps_calibration_last_source = None
            self._start_gps_calibration_last_error_m = 0.0
            self._start_gps_calibration_last_sample_count = 0
            self._start_gps_calibration_last_mode_monotonic = 0.0

    @staticmethod
    def _is_start_calibration_request(payload: dict, meta: dict) -> bool:
        return bool(meta.get("start_calibration") or payload.get("start_calibration"))

    @staticmethod
    def _is_no_gps_start_request(payload: dict, meta: dict) -> bool:
        return bool(
            meta.get("no_gps_mode")
            or meta.get("gps_disabled")
            or payload.get("no_gps_mode")
            or payload.get("gps_disabled")
        )

    def _cancel_start_gps_calibration(self) -> None:
        self._ensure_start_gps_calibration_state()
        self._start_gps_calibration_pending = False
        self._start_gps_calibration_target_pose = None
        self._start_gps_calibration_source = None
        self._start_gps_calibration_samples.clear()

    def _apply_no_gps_start_mode(
        self,
        payload: dict,
        meta: dict,
        current_yaw: float,
        now: float,
    ) -> tuple[bool, dict | None]:
        self._ensure_start_gps_calibration_state()
        source = self._localisation_source(
            payload,
            meta,
            "manual_dashboard_no_gps_start",
        )
        pose = self._graph.localisation_to_world_pose(payload, default_yaw=current_yaw)
        if pose is None:
            self._finish_start_gps_calibration(
                mode="start_gps_calibration_failed",
                source="invalid_no_gps_start_pose",
                error_m=0.0,
                sample_count=0,
            )
            return False, None

        try:
            save_gps_disabled(True, source=source)
        except Exception as exc:
            print(
                "\033[1;97m[ PoseEstimator ] :\033[0m \033[1;93mWARNING\033[0m"
                f" - Could not persist no-GPS mode: {exc}"
            )

        self._gps_disabled = True
        self._startup_gps_calibration_pending = False
        self._auto_gps_pending = False
        self._last_gps_raw_xy = None
        self._last_gps_raw_monotonic = 0.0
        self._gps_calibration_offset_xy = None
        try:
            self._gps_fix_history.clear()
        except Exception:
            self._gps_fix_history = deque(maxlen=max(10, _AUTO_GPS_SAMPLE_COUNT * 3))
        self._cancel_start_gps_calibration()

        target_x, target_y, target_yaw = pose
        old_x, old_y, _ = self._dr.get_state()
        self._dr.reset(float(target_x), float(target_y), float(target_yaw))
        self._last_yaw_rad = float(target_yaw)
        self._yaw_ekf_p = _YAW_EKF_P_INIT
        self._last_absolute_yaw_fix_monotonic = float(now)
        self._last_absolute_yaw_fix_source = "no_gps_start"
        self._calibrate_imu_yaw_offset_to(float(target_yaw))
        if self.tracking_state is not None and hasattr(self.tracking_state, "set_lane_measurement_state"):
            self.tracking_state.set_lane_measurement_state(False, 0.0)
        error_m = math.hypot(float(target_x) - float(old_x), float(target_y) - float(old_y))
        self._send_sim_relocalize_pose(
            float(target_x),
            float(target_y),
            float(target_yaw),
            source="no_gps_start",
        )
        self._finish_start_gps_calibration(
            mode="gps_disabled_start",
            source=source,
            error_m=error_m,
            sample_count=0,
        )
        print(
            "\033[1;97m[ PoseEstimator ] :\033[0m \033[1;92mINFO\033[0m"
            f" - No-GPS start mode active: pose set to "
            f"({float(target_x):.3f}, {float(target_y):.3f}) "
            f"yaw={math.degrees(float(target_yaw)):.1f}°; GPS fixes ignored"
        )
        return True, {
            "mode": "gps_disabled_start",
            "source": source,
            "error_m": float(error_m),
            "gps_disabled": True,
        }

    def _begin_start_gps_calibration_for_pose(
        self,
        target_pose: tuple[float, float, float],
        now: float,
        *,
        source: str,
    ) -> None:
        self._ensure_start_gps_calibration_state()
        target_x, target_y, target_yaw = target_pose
        self._start_gps_calibration_pending = True
        self._start_gps_calibration_target_pose = (
            float(target_x),
            float(target_y),
            float(target_yaw),
        )
        self._start_gps_calibration_source = str(source)
        self._start_gps_calibration_deadline_monotonic = (
            float(now) + max(0.0, _START_GPS_CALIBRATION_TIMEOUT_S)
        )
        self._start_gps_calibration_samples.clear()
        self._start_gps_calibration_last_mode = None
        self._start_gps_calibration_last_source = None
        self._start_gps_calibration_last_error_m = 0.0
        self._start_gps_calibration_last_sample_count = 0
        self._start_gps_calibration_last_mode_monotonic = 0.0
        live_log(
            "pose_estimator",
            event="start_gps_calibration_started",
            source=str(source),
            target_x=float(target_x),
            target_y=float(target_y),
            sample_count=int(_START_GPS_CALIBRATION_SAMPLE_COUNT),
            timeout_s=float(_START_GPS_CALIBRATION_TIMEOUT_S),
        )
        print(
            "\033[1;97m[ PoseEstimator ] :\033[0m \033[1;96mINFO\033[0m"
            f" - Start GPS calibration waiting for {_START_GPS_CALIBRATION_SAMPLE_COUNT} "
            f"fresh GPS samples at start ({float(target_x):.3f}, {float(target_y):.3f})"
        )

    def _begin_start_gps_calibration(
        self,
        payload: dict,
        meta: dict,
        current_yaw: float,
        now: float,
        *,
        source: str,
    ) -> tuple[bool, dict | None]:
        self._ensure_start_gps_calibration_state()
        pose = self._graph.localisation_to_world_pose(payload, default_yaw=current_yaw)
        if pose is None:
            self._finish_start_gps_calibration(
                mode="start_gps_calibration_failed",
                source="invalid_start_pose",
                error_m=0.0,
                sample_count=0,
            )
            return False, None

        self._begin_start_gps_calibration_for_pose(pose, now, source=source)
        return False, None

    def _finish_start_gps_calibration(
        self,
        *,
        mode: str,
        source: str,
        error_m: float = 0.0,
        sample_count: int | None = None,
    ) -> None:
        self._ensure_start_gps_calibration_state()
        self._start_gps_calibration_pending = False
        self._start_gps_calibration_target_pose = None
        self._start_gps_calibration_source = None
        if sample_count is None:
            sample_count = len(self._recent_start_gps_calibration_samples(time.monotonic()))
        self._start_gps_calibration_last_mode = str(mode)
        self._start_gps_calibration_last_source = str(source)
        self._start_gps_calibration_last_error_m = float(error_m or 0.0)
        self._start_gps_calibration_last_sample_count = int(sample_count or 0)
        self._start_gps_calibration_last_mode_monotonic = time.monotonic()
        self._start_gps_calibration_samples.clear()
        live_log(
            "pose_estimator",
            event="start_gps_calibration_finished",
            mode=str(mode),
            source=str(source),
            error_m=float(error_m or 0.0),
            sample_count=int(sample_count or 0),
        )

    def _start_gps_calibration_recent_status(self, now: float) -> tuple[str, str, float] | None:
        self._ensure_start_gps_calibration_state()
        if self._start_gps_calibration_pending:
            sample_count = len(self._recent_start_gps_calibration_samples(now))
            return (
                "start_gps_calibration_pending",
                f"gps_waiting_for_samples:{sample_count}/{_START_GPS_CALIBRATION_SAMPLE_COUNT}",
                0.0,
            )
        last_t = float(getattr(self, "_start_gps_calibration_last_mode_monotonic", 0.0) or 0.0)
        mode = getattr(self, "_start_gps_calibration_last_mode", None)
        if mode and now - last_t <= _START_GPS_CALIBRATION_STATUS_HOLD_S:
            source = str(getattr(self, "_start_gps_calibration_last_source", None) or "start_gps_calibration")
            samples = int(getattr(self, "_start_gps_calibration_last_sample_count", 0) or 0)
            return (
                str(mode),
                f"{source}:{samples}/{_START_GPS_CALIBRATION_SAMPLE_COUNT}",
                float(getattr(self, "_start_gps_calibration_last_error_m", 0.0) or 0.0),
            )
        return None

    def _remember_start_gps_calibration_fix(self, payload: dict, meta: dict, now: float) -> None:
        self._ensure_start_gps_calibration_state()
        if not bool(getattr(self, "_start_gps_calibration_pending", False)):
            return
        xy = self._payload_xy(payload)
        if xy is None:
            return
        x, y = xy
        if not (math.isfinite(float(x)) and math.isfinite(float(y))):
            return
        if self._meta_marks_gps_out_of_bounds(payload, meta):
            return
        ts = payload.get("timestamp")
        if ts is not None:
            try:
                ts_f = float(ts)
            except (TypeError, ValueError):
                ts_f = 0.0
            if ts_f > 0.0 and (time.time() - ts_f) > _START_GPS_CALIBRATION_MAX_FIX_AGE_S:
                return
        self._start_gps_calibration_samples.append({
            "x": float(x),
            "y": float(y),
            "received_monotonic": float(now),
            "payload": dict(payload),
            "meta": dict(meta),
        })

    def _recent_start_gps_calibration_samples(self, now: float) -> list[dict]:
        self._ensure_start_gps_calibration_state()
        history = list(getattr(self, "_start_gps_calibration_samples", []) or [])
        fresh = [
            sample
            for sample in history
            if now - float(sample.get("received_monotonic", 0.0)) <= _START_GPS_CALIBRATION_MAX_FIX_AGE_S
        ]
        return fresh[-_START_GPS_CALIBRATION_SAMPLE_COUNT:]

    def _try_apply_start_gps_calibration(self, current_yaw: float, now: float):
        self._ensure_start_gps_calibration_state()
        if not bool(getattr(self, "_start_gps_calibration_pending", False)):
            return False, None

        samples = self._recent_start_gps_calibration_samples(now)
        deadline_expired = now > float(getattr(self, "_start_gps_calibration_deadline_monotonic", 0.0) or 0.0)
        if len(samples) < _START_GPS_CALIBRATION_SAMPLE_COUNT:
            if deadline_expired:
                self._finish_start_gps_calibration(
                    mode="start_gps_calibration_failed",
                    source="not_enough_fresh_gps_samples",
                    error_m=0.0,
                    sample_count=len(samples),
                )
                print(
                    "\033[1;97m[ PoseEstimator ] :\033[0m \033[1;93mWARNING\033[0m"
                    f" - Start GPS calibration failed: only {len(samples)}/"
                    f"{_START_GPS_CALIBRATION_SAMPLE_COUNT} fresh GPS samples"
                )
            return False, None

        spread_m = self._gps_sample_spread(samples)
        if spread_m > _START_GPS_CALIBRATION_MAX_SPREAD_M:
            self._finish_start_gps_calibration(
                mode="start_gps_calibration_failed",
                source="gps_spread_too_large",
                error_m=spread_m,
                sample_count=len(samples),
            )
            print(
                "\033[1;97m[ PoseEstimator ] :\033[0m \033[1;93mWARNING\033[0m"
                f" - Start GPS calibration failed: GPS sample spread {spread_m:.3f} m "
                f"> {_START_GPS_CALIBRATION_MAX_SPREAD_M:.3f} m"
            )
            return False, None

        target_pose = getattr(self, "_start_gps_calibration_target_pose", None)
        if target_pose is None:
            self._finish_start_gps_calibration(
                mode="start_gps_calibration_failed",
                source="missing_start_pose",
                error_m=0.0,
                sample_count=len(samples),
            )
            return False, None

        target_x, target_y, target_yaw = target_pose
        avg_x = sum(float(sample["x"]) for sample in samples) / float(len(samples))
        avg_y = sum(float(sample["y"]) for sample in samples) / float(len(samples))
        offset_x = float(target_x) - float(avg_x)
        offset_y = float(target_y) - float(avg_y)
        old_x, old_y, _ = self._dr.get_state()
        self._gps_calibration_offset_xy = (offset_x, offset_y)
        self._dr.reset(float(target_x), float(target_y), float(target_yaw))
        self._last_yaw_rad = float(target_yaw)
        self._yaw_ekf_p = _YAW_EKF_P_INIT
        self._last_absolute_yaw_fix_monotonic = time.monotonic()
        self._last_absolute_yaw_fix_source = "start_gps_calibration"
        self._calibrate_imu_yaw_offset_to(float(target_yaw))
        if self.tracking_state is not None and hasattr(self.tracking_state, "set_lane_measurement_state"):
            self.tracking_state.set_lane_measurement_state(False, 0.0)
        error_m = math.hypot(float(target_x) - float(old_x), float(target_y) - float(old_y))
        calibration_source = str(
            getattr(self, "_start_gps_calibration_source", None)
            or "manual_dashboard_start_calibration"
        )
        self._send_sim_relocalize_pose(
            float(target_x),
            float(target_y),
            float(target_yaw),
            source="start_gps_calibration",
        )
        self._finish_start_gps_calibration(
            mode="start_gps_calibration",
            source=calibration_source,
            error_m=error_m,
            sample_count=len(samples),
        )
        print(
            "\033[1;97m[ PoseEstimator ] :\033[0m \033[1;92mINFO\033[0m"
            f" - Start GPS calibrated from {len(samples)} samples: "
            f"avg raw ({avg_x:.3f}, {avg_y:.3f}) -> "
            f"start ({float(target_x):.3f}, {float(target_y):.3f}); "
            f"offset=({offset_x:.3f}, {offset_y:.3f}); spread={spread_m:.3f} m"
        )
        return True, {
            "mode": "start_gps_calibration",
            "source": calibration_source,
            "error_m": float(error_m),
            "gps_spread_m": float(spread_m),
            "gps_sample_count": int(len(samples)),
        }

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
        if self._gps_runtime_disabled():
            return False
        if not bool(meta.get("manual")):
            return False
        source = self._localisation_source(payload, meta, "manual_localisation").strip().lower()
        return bool(meta.get("gps_calibration")) or source.startswith("manual_dashboard")

    def _calibrate_imu_yaw_offset_to(self, target_yaw_rad: float) -> tuple[bool, float | None]:
        self._pending_yaw_offset_target_rad = float(target_yaw_rad)
        imu = getattr(self, "_last_raw_imu", None)
        if not isinstance(imu, dict):
            self._yaw_offset_calibrated = False
            return False, None
        last_imu_t = getattr(self, "_last_imu_t", None)
        if last_imu_t is None or (time.monotonic() - float(last_imu_t)) > 1.0:
            self._yaw_offset_calibrated = False
            return False, None
        try:
            raw_yaw_deg = float(imu.get("yaw"))
        except (TypeError, ValueError):
            self._yaw_offset_calibrated = False
            return False, None
        yaw_raw_rad = _IMU_YAW_SIGN * math.radians(raw_yaw_deg)
        self._yaw_offset = float(target_yaw_rad) - float(yaw_raw_rad)
        self._yaw_offset_calibrated = True
        self._pending_yaw_offset_target_rad = None
        return True, raw_yaw_deg

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
        imu_calibrated, raw_yaw_deg = self._calibrate_imu_yaw_offset_to(float(yaw))
        if self.tracking_state is not None and hasattr(self.tracking_state, "set_lane_measurement_state"):
            self.tracking_state.set_lane_measurement_state(False, 0.0)

        imu_text = (
            f"; imu_raw={float(raw_yaw_deg):.1f}° "
            f"imu_offset={math.degrees(float(self._yaw_offset)):.1f}°"
            if imu_calibrated
            else "; imu offset pending next IMU"
        )
        print(
            "\033[1;97m[ PoseEstimator ] :\033[0m \033[1;92mINFO\033[0m"
            f" - GPS calibrated: raw ({raw_x:.3f}, {raw_y:.3f}) -> "
            f"map ({float(target_x):.3f}, {float(target_y):.3f}); "
            f"offset=({offset_x:.3f}, {offset_y:.3f})"
            f"{imu_text}"
        )
        return True, {
            "mode": "gps_calibration",
            "source": "manual_dashboard:gps_calibration",
            "error_m": float(math.hypot(float(target_x) - float(old_x), float(target_y) - float(old_y))),
        }

    def _apply_gps_calibration_offset(self, payload: dict, meta: dict) -> tuple[dict, dict]:
        if (
            self._gps_calibration_offset_xy is None
            or not self._gps_dashboard_calibration_enabled()
            or self._gps_runtime_disabled()
        ):
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

    def _apply_startup_gps_calibration_if_pending(self, payload: dict, now: float | None = None) -> bool:
        if not bool(getattr(self, "_startup_gps_calibration_pending", False)):
            return False
        if self._gps_runtime_disabled():
            self._startup_gps_calibration_pending = False
            return False
        if not self._gps_dashboard_calibration_enabled():
            self._startup_gps_calibration_pending = False
            return False
        pose = getattr(self, "_startup_world_pose", None)
        if pose is None:
            self._startup_gps_calibration_pending = False
            return False
        self._startup_gps_calibration_pending = False
        self._begin_start_gps_calibration_for_pose(
            (
                float(pose[0]),
                float(pose[1]),
                float(pose[2]),
            ),
            time.monotonic() if now is None else float(now),
            source="startup_start_calibration",
        )
        return True

    def _resolve_initial_world_pose(self) -> tuple[float, float, float] | None:
        if self._graph is None:
            return None
        default_pose = self._graph.get_start_pose()
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
        from src.core.bus.topics import SIM_RELOCALIZE
        messageHandlerSender(self.queuesList, SIM_RELOCALIZE).send({
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

    @staticmethod
    def _confidence_fade(value: float, full_value: float, zero_value: float) -> float:
        value = abs(float(value))
        full_value = abs(float(full_value))
        zero_value = abs(float(zero_value))
        if zero_value <= full_value:
            return 0.0 if value >= zero_value else 1.0
        if value <= full_value:
            return 1.0
        if value >= zero_value:
            return 0.0
        return (zero_value - value) / (zero_value - full_value)

    def _filtered_steer_for_yaw(self, dt: float) -> tuple[float, float]:
        """Return servo-lagged wheel angle and its rate for body-yaw prediction."""
        previous = float(getattr(self, "_steer_filtered_rad", 0.0) or 0.0)
        commanded = float(getattr(self, "_last_steer_rad", 0.0) or 0.0) * float(_STEER_GAIN_DR)
        alpha = max(0.0, min(1.0, float(_STEER_LAG_ALPHA)))
        filtered = previous + alpha * (commanded - previous)
        self._steer_filtered_rad = float(filtered)

        last_for_fusion = float(getattr(self, "_last_yaw_fusion_steer_rad", previous) or 0.0)
        steer_rate = (filtered - last_for_fusion) / max(float(dt), 0.02)
        self._last_yaw_fusion_steer_rad = float(filtered)
        return float(filtered), float(steer_rate)

    def _speed_for_dead_reckoning(self, speed_mps: float, steer_rad: float) -> float:
        speed = float(speed_mps)
        if (
            bool(_ENCODER_CURVE_SPEED_COMPENSATION)
            and str(getattr(self, "_last_speed_source", "none") or "none").startswith("encoder")
        ):
            speed = curve_speed_compensated(
                speed,
                float(steer_rad),
                wheelbase_m=float(_WHEELBASE_M),
                rear_axle_to_cg_m=float(_REAR_AXLE_TO_CG_M),
            )
        self._last_dr_speed_integrated_mps = float(speed)
        return float(speed)

    @staticmethod
    def _kinematic_yaw_end(yaw_start: float, speed_mps: float, steer_rad: float, dt: float) -> float:
        wheelbase_m = max(1e-6, float(_WHEELBASE_M))
        beta = math.atan((float(_REAR_AXLE_TO_CG_M) / wheelbase_m) * math.tan(float(steer_rad)))
        yaw_rate = -(float(speed_mps) * math.cos(beta) / wheelbase_m) * math.tan(float(steer_rad))
        return float(yaw_start) + yaw_rate * float(dt)

    def _fuse_yaw_for_dead_reckoning(
        self,
        *,
        now: float,
        yaw_start_rad: float,
        dt: float,
        imu_dict: dict | None,
        previous_imu_t: float | None,
    ) -> float:
        """Fuse IMU yaw with a delayed bicycle-model body-yaw prediction.

        The BNO055 yaw can jump when the steering servo moves.  The safe path is
        to predict body yaw from speed, wheelbase and lagged wheel angle, then
        use IMU yaw only as a rate-limited correction when steering is quiet.
        """
        dt = max(0.0, min(float(dt), float(_MAX_INTEGRATION_DT)))
        steer_rad, steer_rate_rad_s = self._filtered_steer_for_yaw(dt)
        speed_for_dr = self._speed_for_dead_reckoning(
            float(getattr(self, "_last_speed", 0.0) or 0.0),
            steer_rad,
        )
        yaw_kinematic = self._kinematic_yaw_end(
            float(yaw_start_rad),
            speed_for_dr,
            steer_rad,
            dt,
        )
        wheelbase_m = max(1e-6, float(_WHEELBASE_M))
        self._last_dr_beta_rad = float(
            math.atan((float(_REAR_AXLE_TO_CG_M) / wheelbase_m) * math.tan(float(steer_rad)))
        )

        self._last_yaw_kinematic_rad = float(yaw_kinematic)
        self._last_yaw_used_rad = float(yaw_kinematic)
        self._last_imu_yaw_limited_rad = None
        self._last_imu_yaw_gain = 0.0
        self._last_yaw_rejected = imu_dict is not None

        p_prior = min(
            float(getattr(self, "_yaw_ekf_p", _YAW_EKF_P_INIT) or 0.0) + float(_YAW_EKF_Q) * dt,
            1.0,
        )

        if imu_dict is None:
            self._last_imu_yaw_raw_deg = None
            self._yaw_ekf_p = p_prior
            return float(yaw_kinematic)

        yaw_deg = float(imu_dict.get("yaw", math.degrees(float(yaw_start_rad))))
        self._last_imu_yaw_raw_deg = float(yaw_deg)
        yaw_raw_rad = _IMU_YAW_SIGN * math.radians(yaw_deg)

        if not self._yaw_offset_calibrated:
            target_yaw = (
                float(self._pending_yaw_offset_target_rad)
                if self._pending_yaw_offset_target_rad is not None
                else float(self._start_yaw_rad)
            )
            self._yaw_offset = target_yaw - yaw_raw_rad
            self._yaw_offset_calibrated = True
            self._pending_yaw_offset_target_rad = None

        yaw_imu = yaw_raw_rad + self._yaw_offset
        dt_imu = (
            float(now) - float(previous_imu_t)
            if previous_imu_t is not None
            else max(float(dt), 0.02)
        )
        max_delta = float(_MAX_PHYSICAL_YAW_RATE_RADS) * max(dt_imu, 0.02)
        imu_delta = _wrap_angle(float(yaw_imu) - float(yaw_start_rad))
        limited_delta = max(-max_delta, min(max_delta, imu_delta))
        yaw_imu_limited = float(yaw_start_rad) + limited_delta
        clipped = abs(imu_delta - limited_delta) > math.radians(0.1)
        self._last_imu_yaw_limited_rad = float(yaw_imu_limited)

        steer_abs_deg = abs(math.degrees(float(steer_rad)))
        steer_rate_deg_s = abs(math.degrees(float(steer_rate_rad_s)))
        steer_conf = self._confidence_fade(
            steer_abs_deg,
            _IMU_STEER_TRUST_FULL_DEG,
            _IMU_STEER_TRUST_ZERO_DEG,
        )
        steer_rate_conf = self._confidence_fade(
            steer_rate_deg_s,
            _IMU_STEER_RATE_TRUST_FULL_DEGS,
            _IMU_STEER_RATE_TRUST_ZERO_DEGS,
        )

        if steer_conf <= 0.0 or steer_rate_conf <= 0.0:
            self._imu_yaw_inhibit_until = max(
                float(getattr(self, "_imu_yaw_inhibit_until", 0.0) or 0.0),
                float(now) + max(0.0, float(_IMU_STEER_INHIBIT_HOLD_S)),
            )

        confidence = min(steer_conf, steer_rate_conf)
        if float(now) < float(getattr(self, "_imu_yaw_inhibit_until", 0.0) or 0.0):
            confidence = 0.0

        r_imu = float(_YAW_EKF_R_STRAIGHT) + float(_YAW_EKF_R_STEER_K) * (float(steer_rad) ** 2)
        base_gain = p_prior / (p_prior + r_imu) if (p_prior + r_imu) > 1e-12 else 0.0
        imu_gain = max(0.0, min(1.0, base_gain * confidence))
        innovation = _wrap_angle(float(yaw_imu_limited) - float(yaw_kinematic))
        yaw_used = float(yaw_kinematic) + imu_gain * innovation

        self._yaw_ekf_p = (1.0 - imu_gain) * p_prior
        self._last_imu_yaw_gain = float(imu_gain)
        self._last_yaw_rejected = bool(clipped or imu_gain < 0.05)
        self._last_yaw_used_rad = float(yaw_used)
        return float(yaw_used)

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

    def _try_lane_yaw_reset(
        self,
        now: float,
        raw_yaw: float,
        route_context: RouteContext | None,
        lane_observation: LaneObservation | None,
    ) -> float:
        if not bool(_LANE_YAW_RESET_ENABLED):
            return 0.0
        if self._dr is None or lane_observation is None or route_context is None:
            self._lane_yaw_reset_count = 0
            return 0.0
        if float(now) < float(getattr(self, "_lane_yaw_reset_next_t", 0.0) or 0.0):
            return 0.0
        if self._has_fresh_absolute_yaw_fix(now):
            self._lane_yaw_reset_count = 0
            return 0.0
        if abs(float(self._last_speed or 0.0)) < float(_VISUAL_LANE_RELOCALIZATION_SPEED_MIN_MPS):
            self._lane_yaw_reset_count = 0
            return 0.0

        detected_sides = {str(side).lower() for side in (lane_observation.detected_sides or ())}
        if (
            str(lane_observation.measurement_mode or "none") != "two_line"
            or "left" not in detected_sides
            or "right" not in detected_sides
            or float(lane_observation.quality or 0.0) < float(_LANE_YAW_RESET_QUALITY_MIN)
            or str(lane_observation.curve_hint or "UNKNOWN").upper() != "STRAIGHT"
        ):
            self._lane_yaw_reset_count = 0
            return 0.0

        heading_error = float(lane_observation.heading_error_rad or 0.0)
        if abs(heading_error) > float(_LANE_YAW_RESET_STRAIGHT_THRESH_RAD):
            self._lane_yaw_reset_count = 0
            return 0.0

        cam_yaw = lane_observation.camera_yaw_hint_rad
        cam_conf = float(lane_observation.camera_yaw_hint_confidence or 0.0)
        if cam_yaw is None or cam_conf < 0.6:
            self._lane_yaw_reset_count = 0
            return 0.0

        yaw_error = _wrap_angle(float(cam_yaw) - float(raw_yaw))
        if abs(yaw_error) > float(_LANE_YAW_RESET_MAX_ERROR_RAD):
            self._lane_yaw_reset_count = 0
            return 0.0

        self._lane_yaw_reset_count = int(getattr(self, "_lane_yaw_reset_count", 0) or 0) + 1
        if self._lane_yaw_reset_count < int(_LANE_YAW_RESET_CONSECUTIVE):
            return 0.0

        self._dr.correct_yaw(float(yaw_error))
        self._last_yaw_rad = float(raw_yaw) + float(yaw_error)
        self._calibrate_imu_yaw_offset_to(float(cam_yaw))
        self._lane_yaw_reset_next_t = float(now) + max(0.0, float(_LANE_YAW_RESET_COOLDOWN_S))
        self._lane_yaw_reset_count = 0
        self._last_lane_yaw_reset_deg = math.degrees(float(yaw_error))
        if self.tracking_state is not None:
            self.tracking_state.last_yaw_correction_deg = math.degrees(float(yaw_error))
        live_log(
            "pose_estimator",
            event="lane_yaw_reset",
            yaw_correction_deg=math.degrees(float(yaw_error)),
            cam_yaw_deg=math.degrees(float(cam_yaw)),
            raw_yaw_deg=math.degrees(float(raw_yaw)),
            heading_error_deg=math.degrees(float(heading_error)),
            confidence=float(cam_conf),
        )
        return float(yaw_error)

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

        # Plan B3.2: la gain/max/step de la corrección visual depende del
        # scenario activo. El BehaviorPlanner publica el scenario_name en
        # TrackingState; resolvemos el perfil (lane_keep agresivo, parking
        # apagado, intersection débil, ...).
        scenario_name = None
        if self.tracking_state is not None:
            try:
                scenario_name = self.tracking_state.current_scenario
            except Exception:
                scenario_name = None
        profile = resolve_visual_correction_profile(scenario_name)
        if profile.is_disabled:
            return raw_x, raw_y, raw_yaw, 0.0, False

        # Política single_line (post-mortem run_20260518_025658):
        # No aplicamos corrección al pose cuando solo hay una línea visible.
        # La medición es indirecta (extrapola la línea faltante con
        # lane_width asumido) y produjo saturación sesgada de 28 cm en el
        # run pasado, arrastrando el ego fuera del carril. El REF tampoco
        # consume la cámara para el control (subLane=False); adoptamos la
        # misma postura para la corrección al pose. Two_line sigue activo.
        measurement_mode = str(
            getattr(lane_observation, "measurement_mode", "") or ""
        ).strip().lower()
        if measurement_mode == "single_line":
            live_log(
                "pose_estimator",
                event="lane_correction_single_line_ignored",
                measurement_m=float(measurement),
                scenario=scenario_name,
            )
            return raw_x, raw_y, raw_yaw, 0.0, False

        # Cooldown: lo gestiona el caller (la función pura compute_lateral_*
        # no lo conoce). Mantenemos la constante del config — no es parte
        # del perfil porque define la cadencia de muestreo, no la autoridad.
        if (now - self._last_camera_lateral_correction_monotonic) < float(
            _CAMERA_LATERAL_CORRECTION_COOLDOWN_S
        ):
            return raw_x, raw_y, raw_yaw, 0.0, True

        result = compute_lateral_correction(
            lateral_error_m=float(measurement),
            speed_mps=abs(float(self._last_speed)),
            now_s=float(now),
            last_correction_time_s=float(self._last_camera_lateral_correction_monotonic),
            profile=profile,
            min_speed_mps=float(min_speed_mps),
        )
        if not result.applied or abs(result.correction_m) < 1e-9:
            return raw_x, raw_y, raw_yaw, 0.0, result.applied
        correction_m = float(result.correction_m)

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

    def _apply_localisation_fix(self, current_yaw: float, now: float | None = None):
        if self._dr is None or self._graph is None:
            return False, None
        now_mono = time.monotonic() if now is None else float(now)
        payload = self._receive_localisation_fix_payload()
        if not isinstance(payload, dict):
            self._try_apply_start_gps_calibration(current_yaw, now_mono)
            self._try_apply_auto_gps_relocalization(current_yaw, now_mono)
            return False, None

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            meta = {}

        print(
            f"[GPS-DBG] PoseEstimatorThread._apply_localisation_fix: GOT fix "
            f"world_x={payload.get('world_x')} world_y={payload.get('world_y')} "
            f"meta.source={meta.get('source')!r} manual={meta.get('manual')}",
            flush=True,
        )

        default_source = "manual_localisation" if bool(meta.get("manual")) else "gps_localisation"
        source = self._localisation_source(payload, meta, default_source)
        if self._is_no_gps_start_request(payload, meta) and bool(meta.get("manual")):
            return self._apply_no_gps_start_mode(
                payload,
                meta,
                current_yaw,
                now_mono,
            )
        if source.strip().lower() == "gps_localisation":
            if self._gps_runtime_disabled():
                print(
                    "[GPS-DBG] PoseEstimatorThread._apply_localisation_fix: "
                    "GPS fix ignored because no-GPS mode is active",
                    flush=True,
                )
                return False, None
            self._remember_gps_raw_fix(payload)
            self._apply_startup_gps_calibration_if_pending(payload, now_mono)
            self._remember_start_gps_calibration_fix(payload, meta, now_mono)
            start_calibrated, start_info = self._try_apply_start_gps_calibration(
                current_yaw,
                now_mono,
            )
            if start_calibrated:
                return True, start_info
            payload, meta = self._apply_gps_calibration_offset(payload, meta)
            self._remember_auto_gps_fix(payload, meta, now_mono)
            return self._try_apply_auto_gps_relocalization(current_yaw, now_mono)
        elif (
            self._gps_dashboard_calibration_enabled()
            and self._is_start_calibration_request(payload, meta)
            and bool(meta.get("manual"))
        ):
            return self._begin_start_gps_calibration(
                payload,
                meta,
                current_yaw,
                now_mono,
                source=self._localisation_source(
                    payload,
                    meta,
                    "manual_dashboard_start_calibration",
                ),
            )
        elif self._should_calibrate_gps_from_dashboard(payload, meta):
            calibrated, calibration_info = self._apply_gps_dashboard_calibration(
                payload,
                meta,
                current_yaw,
            )
            if calibrated:
                return True, calibration_info
            # Manual relocate must still work when LoCSys/TrafficCommunication is
            # disconnected or stale. GPS calibration is an optional side effect.

        old_x, old_y, _ = self._dr.get_state()
        pose = self._graph.localisation_to_world_pose(payload, default_yaw=current_yaw)
        if pose is None:
            print(
                f"[GPS-DBG] PoseEstimatorThread._apply_localisation_fix: "
                f"localisation_to_world_pose returned None for payload "
                f"world_x={payload.get('world_x')} world_y={payload.get('world_y')}",
                flush=True,
            )
            return False, None

        x, y, yaw = pose
        print(
            f"[GPS-DBG] PoseEstimatorThread._apply_localisation_fix: "
            f"world_pose=({float(x):.3f}, {float(y):.3f}, {math.degrees(float(yaw)):+.1f}deg) "
            f"applied to DR; prev=({old_x:.3f}, {old_y:.3f}) "
            f"jump={math.hypot(float(x)-float(old_x), float(y)-float(old_y)):.3f} m",
            flush=True,
        )
        self._dr.reset(float(x), float(y), float(yaw))
        self._last_yaw_rad = float(yaw)
        self._yaw_ekf_p = _YAW_EKF_P_INIT
        if bool(meta.get("manual")):
            self._calibrate_imu_yaw_offset_to(float(yaw))
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
        # D2: Adoptamos MessageHeader para trazabilidad cross-thread.
        header = _stamp_header(
            "pose_estimator",
            self._seq_location,
            timestamp=float(pose_estimate.timestamp) if pose_estimate.timestamp else None,
        )
        try:
            self._loc_sender.send({
                "header": header.to_dict(),
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

        yaw_start_rad = float(self._last_yaw_rad)
        dr_dt = min(dt, _MAX_INTEGRATION_DT)

        imu_dict = None
        previous_imu_t = getattr(self, "_last_imu_t", None)
        imu_raw = self._imu_sub.receive()
        if imu_raw is not None:
            try:
                imu_dict = ast.literal_eval(str(imu_raw))
                self._last_raw_imu = imu_dict
                self._last_imu_t = now
                self._imu_received = True
            except Exception:
                imu_dict = None

        if self._dr is None:
            return

        yaw_end_rad = self._fuse_yaw_for_dead_reckoning(
            now=now,
            yaw_start_rad=yaw_start_rad,
            dt=dr_dt,
            imu_dict=imu_dict,
            previous_imu_t=previous_imu_t,
        )
        self._dr.update_from_yaw(
            float(getattr(self, "_last_dr_speed_integrated_mps", self._last_speed)),
            yaw_start_rad,
            yaw_end_rad,
            dr_dt,
            steer_rad=float(getattr(self, "_steer_filtered_rad", self._last_steer_rad) or 0.0),
            wheelbase_m=float(_WHEELBASE_M),
            rear_axle_to_cg_m=float(_REAR_AXLE_TO_CG_M),
        )
        raw_x, raw_y, raw_yaw = self._dr.get_state()
        self._last_yaw_rad = float(raw_yaw)
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

        gps_relocalized, gps_match = self._apply_localisation_fix(raw_yaw, now)
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
            yaw_correction_rad = self._try_lane_yaw_reset(now, raw_yaw, route_context, lane_observation)
            if abs(yaw_correction_rad) <= 1e-9:
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
        start_gps_status = self._start_gps_calibration_recent_status(now)
        auto_gps_status = self._auto_gps_recent_status(now)
        if gps_relocalized and isinstance(gps_match, dict):
            relocalization_mode = str(gps_match.get("mode") or "gps_fix")
            relocalization_source = str(gps_match.get("source") or "gps_localisation")
            relocalization_error_m = float(gps_match.get("error_m") or 0.0)
        elif start_gps_status is not None:
            relocalization_mode, relocalization_source, relocalization_error_m = start_gps_status
        elif auto_gps_status is not None:
            relocalization_mode, relocalization_source, relocalization_error_m = auto_gps_status
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
            speed_source=str(pose_estimate.speed_source),
            speed_integrated_mps=float(
                getattr(self, "_last_dr_speed_integrated_mps", pose_estimate.speed_mps)
            ),
            steer_rad=float(pose_estimate.steer_rad),
            beta_rad=float(getattr(self, "_last_dr_beta_rad", 0.0) or 0.0),
            imu_yaw_raw=(
                None
                if getattr(self, "_last_imu_yaw_raw_deg", None) is None
                else float(self._last_imu_yaw_raw_deg)
            ),
            yaw_used=float(getattr(self, "_last_yaw_used_rad", fused_pose.yaw)),
            yaw_kinematic=float(getattr(self, "_last_yaw_kinematic_rad", fused_pose.yaw)),
            yaw_rejected=bool(getattr(self, "_last_yaw_rejected", False)),
            imu_yaw_gain=float(getattr(self, "_last_imu_yaw_gain", 0.0) or 0.0),
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
