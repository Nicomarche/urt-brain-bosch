from __future__ import annotations

import time

from src.core.types import LaneObservation, StoplineObservation, VisualStateSnapshot
from src.templates.threadwithstop import ThreadWithStop
from src.utils.live_log import live_log


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
    def _detected_sides(snapshot: VisualStateSnapshot) -> tuple[str, ...]:
        local_payload = snapshot.local_lane_payload or {}
        point_counts = dict(local_payload.get("lane_side_point_counts") or {})
        sides = []
        if int(point_counts.get("left", 0) or 0) > 0:
            sides.append("left")
        if int(point_counts.get("right", 0) or 0) > 0:
            sides.append("right")
        if sides:
            return tuple(sides)

        frame_trace = snapshot.frame_trace or {}
        lane_observation = frame_trace.get("lane_observation") or {}
        visible_side = str(lane_observation.get("visible_side", "") or "")
        if visible_side == "both":
            return ("left", "right")
        if visible_side in {"left", "right"}:
            return (visible_side,)

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
        error_m = frame_trace.get("error_m")
        return float(error_m) if error_m is not None else None

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

    def _build_lane_observation(self, snapshot: VisualStateSnapshot) -> LaneObservation:
        frame_trace = snapshot.frame_trace or {}
        blind_mode = None
        debug = frame_trace.get("debug") or {}
        if isinstance(debug, dict):
            blind_mode = debug.get("blind_control_mode")
        detected_sides = self._detected_sides(snapshot)
        direct_error_m = self._direct_error_m(snapshot)
        return LaneObservation(
            timestamp=float(snapshot.timestamp),
            source_mode=str(snapshot.detection_mode or "unknown"),
            detected_sides=detected_sides,
            lateral_offset_m=direct_error_m,
            heading_error_rad=float(snapshot.heading_error_rad or 0.0),
            direct_error_m=direct_error_m,
            lane_width_px=self._lane_width_px(snapshot),
            quality=self._quality_from_sides(detected_sides, blind_mode),
            curve_hint=str(snapshot.curve_state or "STRAIGHT"),
            camera_yaw_hint_rad=snapshot.camera_yaw_hint_rad,
            camera_yaw_hint_confidence=float(snapshot.camera_yaw_hint_confidence or 0.0),
            blind_mode=str(blind_mode) if blind_mode else None,
            debug=dict(debug),
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
