from __future__ import annotations

import math
import time

import config as _config
from src.core.types import LaneObservation, StoplineObservation, VisualStateSnapshot
from src.templates.threadwithstop import ThreadWithStop
from src.utils.live_log import live_log

_VISUAL_PATH_QUALITY_BUMP = 0.85
_VISUAL_PATH_MIN_POINTS = max(2, int(getattr(_config, "LANE_VISUAL_MIN_POLY_POINTS", 8)))


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

    def _build_lane_observation(self, snapshot: VisualStateSnapshot) -> LaneObservation:
        frame_trace = snapshot.frame_trace or {}
        blind_mode = None
        debug = frame_trace.get("debug") or {}
        if isinstance(debug, dict):
            blind_mode = debug.get("blind_control_mode")
        detected_sides = self._detected_sides(snapshot)
        raw_direct_error_m = self._direct_error_m(snapshot)
        measurement_mode = self._measurement_mode(snapshot, detected_sides, blind_mode)
        direct_error_valid = self._direct_error_valid(
            snapshot,
            measurement_mode=measurement_mode,
            direct_error_m=raw_direct_error_m,
        )
        direct_error_m = raw_direct_error_m if direct_error_valid else None

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

        base_quality = self._quality_from_sides(detected_sides, blind_mode)
        if len(center_waypoints_body) >= _VISUAL_PATH_MIN_POINTS:
            quality = max(base_quality, _VISUAL_PATH_QUALITY_BUMP)
            # Derivar `direct_error_m` desde el primer waypoint cuando hay
            # waypoints visuales. y_left positivo = vehículo desplazado a la
            # izquierda del centro de carril → direct_error es el negativo.
            derived_error = -float(center_waypoints_body[0][1])
            if direct_error_m is None and math.isfinite(derived_error):
                direct_error_m = derived_error
                direct_error_valid = True
        else:
            quality = base_quality

        observation_debug = dict(debug)
        observation_debug["raw_direct_error_m"] = raw_direct_error_m
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
            quality=quality,
            curve_hint=str(snapshot.curve_state or "STRAIGHT"),
            camera_yaw_hint_rad=snapshot.camera_yaw_hint_rad,
            camera_yaw_hint_confidence=float(snapshot.camera_yaw_hint_confidence or 0.0),
            measurement_mode=measurement_mode,
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
            measurement_mode=lane_observation.measurement_mode,
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
