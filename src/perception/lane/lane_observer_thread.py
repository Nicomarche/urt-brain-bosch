from __future__ import annotations

import math
import time

import config as _config
from src.core.types import LaneObservation, StoplineObservation, VisualStateSnapshot
from src.templates.threadwithstop import ThreadWithStop
from src.utils.live_log import live_log

_VISUAL_PATH_QUALITY_BUMP = 0.85
_VISUAL_PATH_MIN_POINTS = max(2, int(getattr(_config, "LANE_VISUAL_MIN_POLY_POINTS", 8)))
_LANE_WIDTH_M = max(0.01, float(getattr(_config, "LANE_WIDTH_CM", 35.0) or 35.0) / 100.0)
_LINE_DISTANCE_MARGIN_M = max(0.04, _LANE_WIDTH_M * 0.18)
_SINGLE_LINE_TRANSITION_MAX_ERROR_DELTA_M = max(
    0.02,
    float(
        getattr(
            _config,
            "LANE_VISUAL_SINGLE_LINE_TRANSITION_MAX_ERROR_DELTA_M",
            max(0.035, _LANE_WIDTH_M * 0.14),
        )
    ),
)
_TWO_LINE_ERROR_MEMORY_MAX_AGE_S = max(
    0.05,
    float(getattr(_config, "LANE_VISUAL_TWO_LINE_ERROR_MEMORY_MAX_AGE_S", 0.75)),
)
_SINGLE_LINE_SIDE_MEMORY_MAX_AGE_S = max(
    0.05,
    float(getattr(_config, "LANE_VISUAL_SINGLE_LINE_SIDE_MEMORY_MAX_AGE_S", 0.85)),
)
_SINGLE_LINE_MAX_PREFIX_ABS_Y_M = max(
    _LANE_WIDTH_M * 0.5,
    float(
        getattr(
            _config,
            "LANE_VISUAL_SINGLE_LINE_MAX_PREFIX_ABS_Y_M",
            (_LANE_WIDTH_M * 0.5) + 0.08,
        )
    ),
)
_SINGLE_LINE_MAX_NEAR_YAW_RAD = math.radians(
    max(5.0, float(getattr(_config, "LANE_VISUAL_SINGLE_LINE_MAX_NEAR_YAW_DEG", 32.0)))
)


class threadLaneObserver(ThreadWithStop):
    """Translate visual-controller snapshots into typed lane/stopline observations."""

    def __init__(
        self,
        queuesList,
        visual_state_buffer,
        lane_observation_buffer,
        stopline_observation_buffer,
    ):
        super().__init__(pause=0.05)
        self.queuesList = queuesList
        self.visual_state_buffer = visual_state_buffer
        self.lane_observation_buffer = lane_observation_buffer
        self.stopline_observation_buffer = stopline_observation_buffer
        self._last_sequence = 0
        self._last_measurement_mode: str | None = None
        self._measurement_mode_streak_frames = 0
        self._last_reliable_two_line_error_m: float | None = None
        self._last_reliable_two_line_timestamp: float | None = None
        self._last_reliable_two_line_error_source: str | None = None
        self._last_reliable_single_line_side: str | None = None
        self._last_reliable_single_line_timestamp: float | None = None

    @staticmethod
    def _line_side_from_screen_position(snapshot: VisualStateSnapshot) -> tuple[str, ...]:
        local_payload = snapshot.local_lane_payload or {}
        lines = dict(local_payload.get("lane_side_lines") or {})
        try:
            img_w = float(local_payload.get("frame_width") or 0.0)
        except (TypeError, ValueError):
            img_w = 0.0
        if img_w <= 1.0:
            return tuple()

        present_lines: list[tuple[str, tuple[float, float, float, float]]] = []
        for raw_side in ("left", "right"):
            raw_line = lines.get(raw_side)
            if not isinstance(raw_line, (list, tuple)) or len(raw_line) < 4:
                continue
            try:
                x1, y1, x2, y2 = (float(raw_line[0]), float(raw_line[1]), float(raw_line[2]), float(raw_line[3]))
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                continue
            present_lines.append((raw_side, (x1, y1, x2, y2)))

        if not present_lines:
            return tuple()
        if len(present_lines) >= 2:
            return ("left", "right")

        _raw_side, (x1, y1, x2, y2) = present_lines[0]
        # Usamos el extremo más cercano a la parte baja de la imagen, que es el
        # punto más estable para decidir si la línea visible cae a la izquierda
        # o a la derecha del auto en pantalla.
        line_x = x1 if y1 >= y2 else x2
        return ("left",) if line_x < (img_w / 2.0) else ("right",)

    @staticmethod
    def _detected_sides(snapshot: VisualStateSnapshot) -> tuple[str, ...]:
        frame_trace = snapshot.frame_trace or {}
        lane_observation = frame_trace.get("lane_observation") or {}
        visible_side = str(lane_observation.get("visible_side", "") or "")
        if visible_side == "both":
            return ("left", "right")
        if visible_side in {"left", "right"}:
            return (visible_side,)

        debug = frame_trace.get("debug") or {}
        resolved_side = str(debug.get("single_line_resolved_side", "") or "")
        if resolved_side in {"left", "right"}:
            return (resolved_side,)

        line_sides = threadLaneObserver._line_side_from_screen_position(snapshot)
        if line_sides:
            return line_sides

        local_payload = snapshot.local_lane_payload or {}
        point_counts = dict(local_payload.get("lane_side_point_counts") or {})
        sides = []
        if int(point_counts.get("left", 0) or 0) > 0:
            sides.append("left")
        if int(point_counts.get("right", 0) or 0) > 0:
            sides.append("right")
        if sides:
            return tuple(sides)

        if frame_trace.get("avg_left_line") is not None and frame_trace.get("avg_right_line") is not None:
            return ("left", "right")
        if frame_trace.get("avg_left_line") is not None:
            return ("left",)
        if frame_trace.get("avg_right_line") is not None:
            return ("right",)
        return tuple()

    @staticmethod
    def _lane_width_px(snapshot: VisualStateSnapshot) -> float | None:
        frame_trace = snapshot.frame_trace or {}
        debug = frame_trace.get("debug") or {}
        local_mask_guidance = debug.get("local_mask_guidance")
        if isinstance(local_mask_guidance, dict):
            value = local_mask_guidance.get("lane_width_px")
            if value is not None:
                return float(value)
        local_payload = snapshot.local_lane_payload or {}
        lane_count = int(local_payload.get("lane_count", 0) or 0)
        if lane_count >= 2:
            lines = dict(local_payload.get("lane_side_lines") or {})
            left_line = lines.get("left") or []
            right_line = lines.get("right") or []
            if len(left_line) >= 4 and len(right_line) >= 4:
                return abs(float(right_line[0]) - float(left_line[0]))
        return None

    @staticmethod
    def _line_distance_metrics(snapshot: VisualStateSnapshot) -> tuple[float | None, float | None, float | None]:
        frame_trace = snapshot.frame_trace or {}
        debug = frame_trace.get("debug") or {}
        if not isinstance(debug, dict):
            return None, None, None

        for prefix in ("two_line", "sl"):
            left_cm = debug.get(f"{prefix}_D_left_cm")
            right_cm = debug.get(f"{prefix}_D_right_cm")
            if left_cm is None or right_cm is None:
                continue
            try:
                left_m = float(left_cm) / 100.0
                right_m = float(right_cm) / 100.0
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(left_m) and math.isfinite(right_m)):
                continue
            # Positivo significa que la distancia derecha es mayor que la
            # izquierda, o sea: el auto está desplazado hacia la línea izquierda.
            center_offset_m = 0.5 * (right_m - left_m)
            return left_m, right_m, center_offset_m
        return None, None, None

    @staticmethod
    def _line_geometry_valid(
        left_m: float | None,
        right_m: float | None,
        lane_width_m: float | None = None,
        *,
        measurement_mode: str = "none",
    ) -> bool:
        if left_m is None or right_m is None:
            return True
        width = lane_width_m if lane_width_m is not None and lane_width_m > 0.0 else _LANE_WIDTH_M
        if str(measurement_mode or "none") == "two_line":
            # With both lane lines visible, distances are measured from the
            # vehicle centre to each boundary. A negative distance means the
            # detector paired the wrong visual line or the car centre has
            # already crossed a boundary; that is not a high-authority
            # centering observation for a 35 cm lane.
            negative_slack = min(0.008, width * 0.025)
            upper_slack = max(0.025, width * 0.08)
            width_slack = max(0.025, width * 0.10)
            return (
                left_m >= -negative_slack
                and right_m >= -negative_slack
                and left_m <= width + upper_slack
                and right_m <= width + upper_slack
                and abs((left_m + right_m) - width) <= width_slack
            )
        margin = max(_LINE_DISTANCE_MARGIN_M, width * 0.18)
        lower = -margin
        upper = width + margin
        return lower <= left_m <= upper and lower <= right_m <= upper

    @staticmethod
    def _line_geometry_allows_visual_path(
        left_m: float | None,
        right_m: float | None,
        lane_width_m: float | None = None,
    ) -> bool:
        if left_m is None or right_m is None:
            return True
        width = lane_width_m if lane_width_m is not None and lane_width_m > 0.0 else _LANE_WIDTH_M
        negative_slack = max(0.018, width * 0.05)
        distance_upper = width + max(0.18, width * 0.50)
        width_upper = width + max(0.30, width * 0.85)
        return (
            left_m >= -negative_slack
            and right_m >= -negative_slack
            and left_m <= distance_upper
            and right_m <= distance_upper
            and (left_m + right_m) <= width_upper
        )

    @staticmethod
    def _direct_error_m(snapshot: VisualStateSnapshot) -> float | None:
        frame_trace = snapshot.frame_trace or {}
        debug = frame_trace.get("debug") or {}
        for key in ("two_line_direct_error_m", "sl_direct_error_m"):
            value = debug.get(key)
            if value is not None:
                return float(value)
        local_mask_guidance = debug.get("local_mask_guidance")
        if isinstance(local_mask_guidance, dict):
            guidance_mode = str(local_mask_guidance.get("guidance_mode", "") or "")
            if guidance_mode == "single_line_physical":
                error_cm = local_mask_guidance.get("error_cm")
                try:
                    error_m = float(error_cm) / 100.0 if error_cm is not None else None
                except (TypeError, ValueError):
                    error_m = None
                if error_m is not None and math.isfinite(error_m):
                    return error_m
        error_m = frame_trace.get("error_m")
        return float(error_m) if error_m is not None else None

    @staticmethod
    def _finite_float(value) -> float | None:
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    @staticmethod
    def _measurement_mode(
        snapshot: VisualStateSnapshot,
        detected_sides: tuple[str, ...],
        blind_mode: str | None,
    ) -> str:
        frame_trace = snapshot.frame_trace or {}
        debug = frame_trace.get("debug") or {}
        debug_mode = str(debug.get("measurement_mode", "") or "")
        if debug_mode in {"two_line", "single_line", "route_tracking", "blind", "none"}:
            return debug_mode
        if len(detected_sides) >= 2:
            return "two_line"
        if len(detected_sides) == 1:
            return "single_line"
        if blind_mode == "route_tracking":
            return "route_tracking"
        if blind_mode:
            return "blind"
        return "none"

    @staticmethod
    def _control_policy_mode(snapshot: VisualStateSnapshot) -> str | None:
        frame_trace = snapshot.frame_trace or {}
        debug = frame_trace.get("debug") or {}
        value = debug.get("control_policy_mode")
        return str(value) if value else None

    @staticmethod
    def _planner_priority_active(snapshot: VisualStateSnapshot) -> bool:
        frame_trace = snapshot.frame_trace or {}
        debug = frame_trace.get("debug") or {}
        if "planner_priority_active" in debug:
            return bool(debug.get("planner_priority_active"))
        return bool(debug.get("planner_priority"))

    @staticmethod
    def _direct_error_valid(
        snapshot: VisualStateSnapshot,
        *,
        measurement_mode: str,
        direct_error_m: float | None,
    ) -> bool:
        frame_trace = snapshot.frame_trace or {}
        debug = frame_trace.get("debug") or {}
        if "direct_error_valid" in debug:
            return bool(debug.get("direct_error_valid"))
        if measurement_mode == "two_line":
            return direct_error_m is not None
        if measurement_mode == "single_line":
            if debug.get("sl_direct_error_m") is not None:
                return True
            local_mask_guidance = debug.get("local_mask_guidance")
            if isinstance(local_mask_guidance, dict):
                guidance_mode = str(local_mask_guidance.get("guidance_mode", "") or "")
                if guidance_mode == "single_line_physical":
                    error_cm = local_mask_guidance.get("error_cm")
                    try:
                        error_m = float(error_cm) / 100.0 if error_cm is not None else None
                    except (TypeError, ValueError):
                        error_m = None
                    return error_m is not None and math.isfinite(error_m)
            return False
        return False

    @staticmethod
    def _quality_from_sides(detected_sides: tuple[str, ...], blind_mode: str | None) -> float:
        if blind_mode == "route_tracking":
            return 0.1
        if blind_mode == "visual_fallback":
            return 0.2
        if len(detected_sides) >= 2:
            return 1.0
        if len(detected_sides) == 1:
            return 0.65
        return 0.0

    @staticmethod
    def _coerce_waypoints(raw) -> tuple[tuple[float, float, float], ...]:
        if not raw:
            return ()
        out: list[tuple[float, float, float]] = []
        for item in raw:
            if item is None:
                continue
            try:
                values = tuple(float(v) for v in item)
            except (TypeError, ValueError):
                continue
            if len(values) < 3:
                continue
            x, y, psi = values[0], values[1], values[2]
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(psi)):
                continue
            out.append((x, y, psi))
        return tuple(out)

    @staticmethod
    def _coerce_poly(raw) -> tuple[float, ...] | None:
        if raw is None:
            return None
        try:
            coeffs = tuple(float(v) for v in raw)
        except (TypeError, ValueError):
            return None
        if not coeffs or not all(math.isfinite(c) for c in coeffs):
            return None
        return coeffs

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        out = float(angle_rad)
        while out > math.pi:
            out -= 2.0 * math.pi
        while out < -math.pi:
            out += 2.0 * math.pi
        return out

    @staticmethod
    def _visual_waypoint_geometry_reject_reason(
        *,
        measurement_mode: str,
        center_waypoints_body: tuple[tuple[float, float, float], ...],
        lane_width_m: float | None,
    ) -> str | None:
        if str(measurement_mode or "none") != "single_line":
            return None
        if len(center_waypoints_body) < 2:
            return None

        width = lane_width_m if lane_width_m is not None and lane_width_m > 0.0 else _LANE_WIDTH_M
        lateral_limit_m = max(float(_SINGLE_LINE_MAX_PREFIX_ABS_Y_M), (float(width) * 0.5) + 0.05)
        prefix = center_waypoints_body[: min(len(center_waypoints_body), 8)]
        xs = [float(item[0]) for item in prefix]
        ys = [float(item[1]) for item in prefix]
        psis = [float(item[2]) for item in prefix]

        if any(x < -0.02 for x in xs):
            return "single_line_waypoint_behind_ego"
        if len(xs) >= 3:
            diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
            if sum(1 for diff in diffs if diff < -0.005) > 0:
                return "single_line_waypoint_non_monotonic"
            if (xs[-1] - xs[0]) < 0.04:
                return "single_line_waypoint_low_forward_span"
        if max(abs(y) for y in ys) > lateral_limit_m:
            return "single_line_waypoint_lateral_out_of_lane"
        near_yaw = max(abs(threadLaneObserver._wrap_angle(psi)) for psi in psis[: min(4, len(psis))])
        if near_yaw > float(_SINGLE_LINE_MAX_NEAR_YAW_RAD):
            return "single_line_waypoint_yaw_outlier"
        return None

    def _build_lane_observation(self, snapshot: VisualStateSnapshot) -> LaneObservation:
        frame_trace = snapshot.frame_trace or {}
        blind_mode = None
        debug = frame_trace.get("debug") or {}
        if isinstance(debug, dict):
            blind_mode = debug.get("blind_control_mode")
        detected_sides = self._detected_sides(snapshot)
        raw_direct_error_m = self._direct_error_m(snapshot)
        measurement_mode = self._measurement_mode(snapshot, detected_sides, blind_mode)
        previous_measurement_mode = getattr(self, "_last_measurement_mode", None)
        current_streak = int(getattr(self, "_measurement_mode_streak_frames", 0) or 0)
        if measurement_mode == previous_measurement_mode:
            self._measurement_mode_streak_frames = current_streak + 1
        else:
            self._measurement_mode_streak_frames = 1
        self._last_measurement_mode = measurement_mode
        direct_error_valid = self._direct_error_valid(
            snapshot,
            measurement_mode=measurement_mode,
            direct_error_m=raw_direct_error_m,
        )
        direct_error_m = raw_direct_error_m if direct_error_valid else None
        left_line_distance_m, right_line_distance_m, line_center_offset_m = self._line_distance_metrics(snapshot)
        explicit_direct_error_invalid = (
            isinstance(debug, dict)
            and "direct_error_valid" in debug
            and not bool(debug.get("direct_error_valid"))
        )

        visual_payload = frame_trace.get("visual_lane_waypoints") if isinstance(frame_trace, dict) else None
        center_waypoints_body: tuple[tuple[float, float, float], ...] = ()
        left_poly_coeffs: tuple[float, ...] | None = None
        right_poly_coeffs: tuple[float, ...] | None = None
        lane_width_m: float | None = None
        extrapolated_side: str | None = None
        if isinstance(visual_payload, dict):
            center_waypoints_body = self._coerce_waypoints(visual_payload.get("center_waypoints_body"))
            left_poly_coeffs = self._coerce_poly(visual_payload.get("left_poly_coeffs"))
            right_poly_coeffs = self._coerce_poly(visual_payload.get("right_poly_coeffs"))
            lw = visual_payload.get("lane_width_m")
            try:
                lane_width_m = float(lw) if lw is not None else None
            except (TypeError, ValueError):
                lane_width_m = None
            side_value = visual_payload.get("extrapolated_side")
            if side_value in ("left", "right"):
                extrapolated_side = side_value

        visual_waypoint_geometry_reject_reason = self._visual_waypoint_geometry_reject_reason(
            measurement_mode=measurement_mode,
            center_waypoints_body=center_waypoints_body,
            lane_width_m=lane_width_m,
        )
        if visual_waypoint_geometry_reject_reason is not None:
            center_waypoints_body = ()
            left_poly_coeffs = None
            right_poly_coeffs = None

        derived_error_m = None
        if center_waypoints_body:
            # y_left positivo = vehículo desplazado a la izquierda del centro
            # de carril, por eso el error directo usa el signo opuesto.
            candidate_derived_error = -float(center_waypoints_body[0][1])
            if math.isfinite(candidate_derived_error):
                derived_error_m = candidate_derived_error

        raw_left_line_distance_m = left_line_distance_m
        raw_right_line_distance_m = right_line_distance_m
        raw_line_center_offset_m = line_center_offset_m
        line_geometry_valid = self._line_geometry_valid(
            left_line_distance_m,
            right_line_distance_m,
            lane_width_m,
            measurement_mode=measurement_mode,
        )
        if not line_geometry_valid:
            direct_error_m = None
            direct_error_valid = False
            left_line_distance_m = None
            right_line_distance_m = None
            line_center_offset_m = None

        if measurement_mode == "two_line" and line_geometry_valid and not explicit_direct_error_invalid:
            center_error_m = self._finite_float(line_center_offset_m)
            if center_error_m is not None:
                direct_error_m = float(center_error_m)
                direct_error_valid = True

        transition_reference_error_m = None
        transition_error_delta_m = None
        transition_error_coherent = True
        transition_selected_error_source = None
        transition_selected_error_m = None
        single_line_side_reject_reason = None
        if measurement_mode == "single_line" and line_geometry_valid:
            last_two_line_error_m = self._finite_float(
                getattr(self, "_last_reliable_two_line_error_m", None)
            )
            last_two_line_timestamp = self._finite_float(
                getattr(self, "_last_reliable_two_line_timestamp", None)
            )
            snapshot_timestamp = self._finite_float(snapshot.timestamp)
            memory_age_s = None
            if last_two_line_timestamp is not None and snapshot_timestamp is not None:
                memory_age_s = max(0.0, float(snapshot_timestamp) - float(last_two_line_timestamp))
            memory_fresh = (
                last_two_line_error_m is not None
                and (
                    memory_age_s is None
                    or memory_age_s <= float(_TWO_LINE_ERROR_MEMORY_MAX_AGE_S)
                )
            )
            if memory_fresh and last_two_line_error_m is not None:
                transition_reference_error_m = float(last_two_line_error_m)
                candidates: list[tuple[str, float]] = []
                for source, value in (
                    ("line_center_offset_m", line_center_offset_m),
                    ("direct_error_m", direct_error_m),
                    ("visual_waypoint_error_m", derived_error_m),
                ):
                    candidate = self._finite_float(value)
                    if candidate is not None:
                        candidates.append((source, candidate))
                if candidates:
                    transition_selected_error_source, transition_selected_error_m = min(
                        candidates,
                        key=lambda item: abs(float(item[1]) - float(last_two_line_error_m)),
                    )
                    transition_error_delta_m = abs(
                        float(transition_selected_error_m) - float(last_two_line_error_m)
                    )
                    transition_error_coherent = (
                        transition_error_delta_m
                        <= float(_SINGLE_LINE_TRANSITION_MAX_ERROR_DELTA_M)
                    )
                    if transition_error_coherent:
                        direct_error_m = float(transition_selected_error_m)
                        direct_error_valid = True
                    else:
                        direct_error_m = None
                        direct_error_valid = False
                        explicit_direct_error_invalid = True

        current_single_line_side = None
        if measurement_mode == "single_line" and len(detected_sides) == 1:
            side_candidate = str(detected_sides[0])
            if side_candidate in {"left", "right"}:
                current_single_line_side = side_candidate
        if current_single_line_side is not None:
            last_side = getattr(self, "_last_reliable_single_line_side", None)
            last_side_ts = self._finite_float(
                getattr(self, "_last_reliable_single_line_timestamp", None)
            )
            snapshot_timestamp = self._finite_float(snapshot.timestamp)
            side_memory_age_s = None
            if last_side_ts is not None and snapshot_timestamp is not None:
                side_memory_age_s = max(0.0, float(snapshot_timestamp) - float(last_side_ts))
            side_memory_fresh = (
                last_side in {"left", "right"}
                and (
                    side_memory_age_s is None
                    or side_memory_age_s <= float(_SINGLE_LINE_SIDE_MEMORY_MAX_AGE_S)
                )
            )
            if (
                side_memory_fresh
                and last_side != current_single_line_side
                and str(previous_measurement_mode or "none") != "two_line"
            ):
                single_line_side_reject_reason = "side_flip_without_two_line_confirmation"
                transition_error_coherent = False
                direct_error_m = None
                direct_error_valid = False
                explicit_direct_error_invalid = True
                center_waypoints_body = ()
                left_poly_coeffs = None
                right_poly_coeffs = None

        base_quality = self._quality_from_sides(detected_sides, blind_mode)
        visual_path_available = len(center_waypoints_body) >= _VISUAL_PATH_MIN_POINTS
        visual_path_geometry_allowed = (
            visual_waypoint_geometry_reject_reason is None
            and (
                measurement_mode != "two_line"
                or line_geometry_valid
                or self._line_geometry_allows_visual_path(
                    raw_left_line_distance_m,
                    raw_right_line_distance_m,
                    lane_width_m,
                )
            )
        )

        if not line_geometry_valid:
            quality = min(base_quality, 0.2)
            if (
                measurement_mode == "two_line"
                and visual_path_available
                and visual_path_geometry_allowed
            ):
                quality = max(float(quality), _VISUAL_PATH_QUALITY_BUMP)
        elif visual_path_available:
            quality = max(base_quality, _VISUAL_PATH_QUALITY_BUMP)
            if direct_error_m is None and derived_error_m is not None and not explicit_direct_error_invalid:
                direct_error_m = derived_error_m
                direct_error_valid = True
        else:
            quality = base_quality

        if not transition_error_coherent:
            quality = min(float(quality), 0.45)
        if single_line_side_reject_reason is not None or visual_waypoint_geometry_reject_reason is not None:
            quality = min(float(quality), 0.35)

        if measurement_mode == "two_line" and direct_error_valid and line_geometry_valid:
            memory_error_m = self._finite_float(line_center_offset_m)
            memory_source = "line_center_offset_m"
            if memory_error_m is None:
                memory_error_m = self._finite_float(direct_error_m)
                memory_source = "direct_error_m"
            if memory_error_m is not None and float(quality) >= 0.8:
                self._last_reliable_two_line_error_m = float(memory_error_m)
                self._last_reliable_two_line_timestamp = float(snapshot.timestamp)
                self._last_reliable_two_line_error_source = memory_source
                self._last_reliable_single_line_side = None
                self._last_reliable_single_line_timestamp = None
        elif (
            current_single_line_side is not None
            and single_line_side_reject_reason is None
            and visual_waypoint_geometry_reject_reason is None
            and float(quality) >= 0.6
        ):
            self._last_reliable_single_line_side = str(current_single_line_side)
            self._last_reliable_single_line_timestamp = float(snapshot.timestamp)

        observation_debug = dict(debug)
        observation_debug["raw_direct_error_m"] = raw_direct_error_m
        observation_debug["line_geometry_valid"] = bool(line_geometry_valid)
        observation_debug["previous_measurement_mode"] = previous_measurement_mode
        observation_debug["measurement_mode_streak_frames"] = int(self._measurement_mode_streak_frames)
        if measurement_mode == "two_line" and line_center_offset_m is not None:
            observation_debug["two_line_center_error_m"] = float(line_center_offset_m)
            observation_debug["two_line_direct_error_for_localization_m"] = float(direct_error_m or 0.0)
        observation_debug["transition_error_coherent"] = bool(transition_error_coherent)
        observation_debug["visual_waypoint_geometry_valid"] = visual_waypoint_geometry_reject_reason is None
        if transition_reference_error_m is not None:
            observation_debug["transition_reference_error_m"] = float(transition_reference_error_m)
        if transition_error_delta_m is not None:
            observation_debug["transition_error_delta_m"] = float(transition_error_delta_m)
        if transition_selected_error_source:
            observation_debug["transition_selected_error_source"] = str(transition_selected_error_source)
            observation_debug["transition_selected_error_m"] = float(transition_selected_error_m)
        if measurement_mode == "two_line":
            observation_debug["last_reliable_two_line_error_m"] = getattr(
                self,
                "_last_reliable_two_line_error_m",
                None,
            )
            observation_debug["last_reliable_two_line_error_source"] = getattr(
                self,
                "_last_reliable_two_line_error_source",
                None,
            )
        if not line_geometry_valid:
            observation_debug["line_geometry_reject_reason"] = "line_distance_out_of_lane_bounds"
            observation_debug["raw_left_line_distance_m"] = raw_left_line_distance_m
            observation_debug["raw_right_line_distance_m"] = raw_right_line_distance_m
            observation_debug["raw_line_center_offset_m"] = raw_line_center_offset_m
            observation_debug["line_geometry_visual_path_allowed"] = bool(visual_path_geometry_allowed)
        elif not transition_error_coherent:
            observation_debug["single_line_transition_reject_reason"] = (
                single_line_side_reject_reason or "error_jump_from_last_two_line"
            )
            observation_debug["single_line_transition_max_error_delta_m"] = float(
                _SINGLE_LINE_TRANSITION_MAX_ERROR_DELTA_M
            )
        if visual_waypoint_geometry_reject_reason is not None:
            observation_debug["visual_waypoint_reject_reason"] = str(
                visual_waypoint_geometry_reject_reason
            )
            observation_debug["single_line_transition_reject_reason"] = str(
                visual_waypoint_geometry_reject_reason
            )
        if single_line_side_reject_reason is not None:
            observation_debug["single_line_side_reject_reason"] = str(single_line_side_reject_reason)
            observation_debug["single_line_previous_reliable_side"] = str(
                getattr(self, "_last_reliable_single_line_side", "") or ""
            )
        if center_waypoints_body:
            observation_debug["visual_waypoint_count"] = len(center_waypoints_body)
            if extrapolated_side:
                observation_debug["visual_extrapolated_side"] = extrapolated_side
        return LaneObservation(
            timestamp=float(snapshot.timestamp),
            source_mode=str(snapshot.detection_mode or "unknown"),
            detected_sides=detected_sides,
            lateral_offset_m=direct_error_m,
            heading_error_rad=float(snapshot.heading_error_rad or 0.0),
            direct_error_m=direct_error_m,
            lane_width_px=self._lane_width_px(snapshot),
            left_line_distance_m=left_line_distance_m,
            right_line_distance_m=right_line_distance_m,
            line_center_offset_m=line_center_offset_m,
            quality=quality,
            curve_hint=str(snapshot.curve_state or "STRAIGHT"),
            camera_yaw_hint_rad=snapshot.camera_yaw_hint_rad,
            camera_yaw_hint_confidence=float(snapshot.camera_yaw_hint_confidence or 0.0),
            measurement_mode=measurement_mode,
            measurement_mode_streak_frames=int(self._measurement_mode_streak_frames),
            previous_measurement_mode=previous_measurement_mode,
            transition_reference_error_m=transition_reference_error_m,
            transition_error_delta_m=transition_error_delta_m,
            transition_error_coherent=bool(transition_error_coherent),
            direct_error_valid=direct_error_valid,
            control_policy_mode=self._control_policy_mode(snapshot),
            planner_priority_active=self._planner_priority_active(snapshot),
            blind_mode=str(blind_mode) if blind_mode else None,
            center_waypoints_body=center_waypoints_body,
            left_poly_coeffs=left_poly_coeffs,
            right_poly_coeffs=right_poly_coeffs,
            lane_width_m=lane_width_m,
            extrapolated_side=extrapolated_side,
            debug=observation_debug,
        )

    def _build_stopline_observation(self, snapshot: VisualStateSnapshot) -> StoplineObservation:
        stopline_debug = dict(snapshot.stopline_debug or {})
        pass_event = stopline_debug.get("stopline_pass_event_payload")
        return StoplineObservation(
            timestamp=float(snapshot.timestamp),
            visible=bool(stopline_debug.get("stopline_visible_candidate", False)),
            stable=bool(stopline_debug.get("stopline_stable_visible", False)),
            distance_m=(
                float(stopline_debug.get("stopline_distance_m"))
                if stopline_debug.get("stopline_distance_m") is not None
                else None
            ),
            confidence=float(stopline_debug.get("stopline_confidence", 0.0) or 0.0),
            pass_event=dict(pass_event) if isinstance(pass_event, dict) else None,
            expected_node_id=stopline_debug.get("stopline_expected_node_id"),
            expected_node_attr=int(stopline_debug.get("stopline_expected_node_attr", 0) or 0),
            source=str(stopline_debug.get("stopline_source", "none") or "none"),
            debug=stopline_debug,
        )

    def thread_work(self):
        snapshot, _, sequence = self.visual_state_buffer.read_latest(with_metadata=True)
        if not isinstance(snapshot, VisualStateSnapshot):
            return
        if int(sequence or 0) == self._last_sequence:
            return
        self._last_sequence = int(sequence or 0)

        lane_observation = self._build_lane_observation(snapshot)
        stopline_observation = self._build_stopline_observation(snapshot)
        self.lane_observation_buffer.write(lane_observation, timestamp=lane_observation.timestamp or time.time())
        self.stopline_observation_buffer.write(
            stopline_observation,
            timestamp=stopline_observation.timestamp or time.time(),
        )

        live_log(
            "lane_observer", event="lane_obs",
            offset_m=lane_observation.lateral_offset_m,
            direct_error_m=lane_observation.direct_error_m,
            heading_error_rad=float(lane_observation.heading_error_rad or 0.0),
            camera_yaw_hint_rad=lane_observation.camera_yaw_hint_rad,
            camera_yaw_hint_confidence=float(lane_observation.camera_yaw_hint_confidence or 0.0),
            quality=float(lane_observation.quality or 0.0),
            sides=list(lane_observation.detected_sides or ()),
            curve_hint=lane_observation.curve_hint,
            blind_mode=lane_observation.blind_mode,
            source_mode=lane_observation.source_mode,
            lane_width_px=lane_observation.lane_width_px,
            left_line_distance_m=lane_observation.left_line_distance_m,
            right_line_distance_m=lane_observation.right_line_distance_m,
            line_center_offset_m=lane_observation.line_center_offset_m,
            measurement_mode=lane_observation.measurement_mode,
            measurement_mode_streak_frames=int(lane_observation.measurement_mode_streak_frames),
            previous_measurement_mode=lane_observation.previous_measurement_mode,
            transition_error_coherent=bool(lane_observation.transition_error_coherent),
            transition_reference_error_m=lane_observation.transition_reference_error_m,
            transition_error_delta_m=lane_observation.transition_error_delta_m,
            direct_error_valid=bool(lane_observation.direct_error_valid),
            control_policy_mode=lane_observation.control_policy_mode,
            planner_priority_active=bool(lane_observation.planner_priority_active),
            visual_waypoint_count=len(lane_observation.center_waypoints_body or ()),
            extrapolated_side=lane_observation.extrapolated_side,
            lane_width_m=lane_observation.lane_width_m,
        )

        if stopline_observation.visible or stopline_observation.pass_event is not None:
            live_log(
                "lane_observer", event="stopline_obs",
                visible=bool(stopline_observation.visible),
                stable=bool(stopline_observation.stable),
                distance_m=stopline_observation.distance_m,
                confidence=float(stopline_observation.confidence or 0.0),
                expected_node_id=stopline_observation.expected_node_id,
                pass_event=bool(stopline_observation.pass_event is not None),
                source=stopline_observation.source,
            )
