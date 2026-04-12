from __future__ import annotations

import time

from src.hardware.pipeline.sharedTypes import ControlDecision, VisualControlCandidate


class NavigationControlPolicy:
    """Resolve the final command authority from route context and visual control."""

    def decide(
        self,
        candidate: VisualControlCandidate | None,
        tracking_state=None,
        *,
        sign_action_active: bool = False,
        steer_override_active: bool = False,
    ) -> ControlDecision:
        now = time.time()
        if candidate is None or candidate.steering_deg is None:
            return ControlDecision(
                timestamp=now,
                steering_deg=0.0,
                speed_cmd=0.0,
                authority="safety",
                safety_override=True,
                reason="missing_candidate",
                speed_sent=False,
                steer_sent=False,
            )

        debug = dict(candidate.debug or {})
        route_active = bool(getattr(tracking_state, "route_active", False)) if tracking_state is not None else False
        waypoint_mode_active = bool(getattr(tracking_state, "waypoint_mode_active", False)) if tracking_state is not None else False
        planner_priority = route_active and (
            waypoint_mode_active
            or str(getattr(tracking_state, "maneuver_type", "none") or "none")
            in {"stopline", "turn_left", "turn_right", "intersection_straight", "roundabout", "traffic_light"}
        )

        authority = "visual"
        reason = "visual_assist"
        if waypoint_mode_active:
            authority = "planner"
            reason = "waypoint_mode"
        elif candidate.blind_mode == "route_tracking":
            authority = "planner"
            reason = "blind_route_tracking"
        elif candidate.blind_mode == "visual_fallback":
            authority = "visual_fallback"
            reason = "visual_hold"
        elif planner_priority:
            authority = "planner"
            reason = "planner_priority"
        elif route_active:
            authority = "planner_visual_assist"
            reason = "route_visual_assist"

        if sign_action_active and steer_override_active:
            return ControlDecision(
                timestamp=now,
                steering_deg=None,
                speed_cmd=None,
                authority="sign_override",
                safety_override=False,
                reason="sign_controls_speed_and_steer",
                speed_sent=False,
                steer_sent=False,
                source_candidate=str(candidate.source or "none"),
                debug=debug,
            )

        if sign_action_active:
            return ControlDecision(
                timestamp=now,
                steering_deg=float(candidate.steering_deg),
                speed_cmd=float(candidate.speed_cmd or 0.0),
                authority="sign_align",
                safety_override=False,
                reason="sign_controls_speed",
                speed_sent=False,
                steer_sent=True,
                source_candidate=str(candidate.source or "none"),
                debug=debug,
            )

        safety_override = bool(debug.get("safety_stop")) or float(candidate.speed_cmd or 0.0) <= 0.0
        return ControlDecision(
            timestamp=now,
            steering_deg=float(candidate.steering_deg),
            speed_cmd=float(candidate.speed_cmd or 0.0),
            authority=authority,
            safety_override=safety_override,
            reason=reason,
            speed_sent=True,
            steer_sent=True,
            source_candidate=str(candidate.source or "none"),
            debug=debug,
        )
