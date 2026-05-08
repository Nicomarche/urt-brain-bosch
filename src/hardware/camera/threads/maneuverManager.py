from __future__ import annotations

import time
from dataclasses import dataclass

import config

from src.perception.signs.sign_classifier import normalize_sign_name

# ----------------------------------------------------------------
# Defaults locales (reemplazan a las constantes que vivían en SignActions
# antes de que ese módulo fuera borrado en Phase 6). Cada uno se sobre-
# escribe con `getattr(config, ...)` en `__init__`, así que tunearlos
# desde `config.py` sigue funcionando igual que antes.
# ----------------------------------------------------------------
_DEFAULT_BASE_SPEED = 5
_DEFAULT_LOW_SPEED = 3
_DEFAULT_SPEED_20 = 3
_DEFAULT_SPEED_30 = 5
_DEFAULT_STOP_DURATION = 3.0
_DEFAULT_CROSSWALK_DURATION = 3.0


@dataclass
class ManeuverDecision:
    control_mode: str
    active_maneuver: str
    mission_state: str = "LANE_FOLLOW"
    speed_override: float | None = None
    steer_override: float | None = None
    speed_cap: float | None = None
    planner_priority: bool = False
    safety_stop: bool = False
    notes: str = ""


class ManeuverManager:
    """Centralized maneuver and sign-observation policy."""

    def __init__(self, state_change_sender=None, highway_mode_event=None):
        self.state_change_sender = state_change_sender
        self.highway_mode_event = highway_mode_event

        self.base_speed = float(getattr(config, "SIGN_BASE_SPEED", _DEFAULT_BASE_SPEED))
        self.low_speed = float(getattr(config, "SIGN_LOW_SPEED", _DEFAULT_LOW_SPEED))
        self.speed_20 = float(getattr(config, "SIGN_SPEED_20", _DEFAULT_SPEED_20))
        self.speed_30 = float(getattr(config, "SIGN_SPEED_30", _DEFAULT_SPEED_30))
        self.stop_duration = float(getattr(config, "SIGN_STOP_DURATION", _DEFAULT_STOP_DURATION))
        self.crosswalk_duration = float(
            getattr(config, "SIGN_CROSSWALK_DURATION", _DEFAULT_CROSSWALK_DURATION)
        )
        self.red_light_max_wait = float(getattr(config, "SIGN_RED_LIGHT_MAX_WAIT", 10.0))
        self.yellow_light_duration = float(getattr(config, "SIGN_YELLOW_LIGHT_DURATION", 2.0))
        self.detection_cooldown = float(getattr(config, "SIGN_ACTION_COOLDOWN", 2.0))

        self._stop_until = 0.0
        self._crosswalk_until = 0.0
        self._yellow_until = 0.0
        self._traffic_light_blocked = False
        self._traffic_light_until = 0.0
        self._speed_limit = None
        self._last_sign_seen: dict[str, float] = {}
        self._active_maneuver = "none"

    @staticmethod
    def _tracking_value(tracking_state, name: str, default=None):
        if tracking_state is None:
            return default
        return getattr(tracking_state, name, default)

    def _sign_matches_route_context(self, sign_name: str, tracking_state) -> bool:
        if tracking_state is None or not bool(self._tracking_value(tracking_state, "route_active", False)):
            return True

        next_semantic_type = str(self._tracking_value(tracking_state, "next_semantic_type", "") or "")
        expected_control_type = str(self._tracking_value(tracking_state, "expected_control_type", "") or "")
        next_semantic_distance_m = self._tracking_value(tracking_state, "next_semantic_distance_m", None)
        zone_types = {
            str(item or "")
            for item in list(self._tracking_value(tracking_state, "current_zone_types", []) or [])
        }

        try:
            if next_semantic_distance_m is not None and float(next_semantic_distance_m) > 1.4:
                if sign_name in {"stop", "no_entry", "red_light", "yellow_light", "green_light", "crosswalk"}:
                    return False
        except (TypeError, ValueError):
            pass

        if sign_name in {"red_light", "yellow_light", "green_light"}:
            return expected_control_type == "traffic_light"
        if sign_name in {"stop", "no_entry"}:
            return expected_control_type == "stop" or next_semantic_type in {"stopline", "intersection"}
        if sign_name == "crosswalk":
            return next_semantic_type == "crosswalk" or "crosswalk" in zone_types
        if sign_name == "parking":
            return next_semantic_type == "parking_spot" or "parking" in zone_types
        return True

    def _planner_mission_state(self, tracking_state) -> str:
        next_semantic_type = str(self._tracking_value(tracking_state, "next_semantic_type", "") or "")
        expected_control_type = str(self._tracking_value(tracking_state, "expected_control_type", "") or "")
        maneuver_type = str(self._tracking_value(tracking_state, "maneuver_type", "none") or "none")

        if next_semantic_type == "parking_spot" or maneuver_type == "parking_search":
            return "PARKING_SEARCH"
        if expected_control_type == "traffic_light":
            return "APPROACH_INTERSECTION"
        if expected_control_type == "stop":
            return "APPROACH_INTERSECTION"
        if next_semantic_type == "crosswalk" or maneuver_type == "crosswalk":
            return "CROSSWALK_SLOW"
        if maneuver_type in {"turn_left", "turn_right", "intersection_straight", "roundabout"}:
            return "APPROACH_INTERSECTION"
        return "LANE_FOLLOW"

    def _mark_seen(self, sign_name: str, now: float) -> bool:
        last_seen = self._last_sign_seen.get(sign_name, 0.0)
        if (now - last_seen) < self.detection_cooldown:
            return False
        self._last_sign_seen[sign_name] = now
        return True

    def _request_parking_mode(self, current_mode: str) -> None:
        if current_mode != "auto" or self.state_change_sender is None:
            return
        # Test/debug toggle: durante runs automatizados no queremos que un
        # parking sign visto por el detector cambie el state_machine —
        # eso corta los tests de lane following después de pocos segundos.
        # Set URT_DISABLE_AUTO_PARKING=1 (run_test.sh lo setea) para
        # ignorar el sign de parking pero seguir respetando otros signs
        # (stop, traffic light, crosswalk).
        import os
        if os.environ.get("URT_DISABLE_AUTO_PARKING", "0") == "1":
            try:
                from src.utils.live_log import live_log
                live_log("maneuver_manager", event="parking_request_ignored",
                         reason="URT_DISABLE_AUTO_PARKING=1")
            except Exception:
                pass
            return
        try:
            self.state_change_sender.send("PARKING")
            try:
                from src.utils.live_log import live_log
                live_log("maneuver_manager", event="parking_request_sent",
                         current_mode=current_mode)
            except Exception:
                pass
        except Exception:
            pass

    def observe_sign(self, payload: dict | None, current_mode: str = "", tracking_state=None) -> None:
        if not isinstance(payload, dict):
            return
        sign_name = normalize_sign_name(payload.get("sign"))
        if not sign_name:
            return
        if not self._sign_matches_route_context(sign_name, tracking_state):
            return

        now = float(payload.get("timestamp", time.time()) or time.time())
        if not self._mark_seen(sign_name, now):
            return

        if sign_name in {"stop", "no_entry"}:
            self._stop_until = max(self._stop_until, now + self.stop_duration)
            self._active_maneuver = sign_name
            return

        if sign_name == "crosswalk":
            self._crosswalk_until = max(self._crosswalk_until, now + self.crosswalk_duration)
            self._active_maneuver = sign_name
            return

        if sign_name == "red_light":
            self._traffic_light_blocked = True
            self._traffic_light_until = max(self._traffic_light_until, now + self.red_light_max_wait)
            self._active_maneuver = sign_name
            return

        if sign_name == "green_light":
            self._traffic_light_blocked = False
            self._traffic_light_until = 0.0
            if self._active_maneuver == "red_light":
                self._active_maneuver = "none"
            return

        if sign_name == "yellow_light":
            self._yellow_until = max(self._yellow_until, now + self.yellow_light_duration)
            self._active_maneuver = sign_name
            return

        if sign_name == "speed_20":
            self._speed_limit = self.speed_20
            return

        if sign_name == "speed_30":
            self._speed_limit = self.speed_30
            return

        if sign_name == "highway_entrance":
            if self.highway_mode_event is not None:
                self.highway_mode_event.set()
            return

        if sign_name == "highway_exit":
            if self.highway_mode_event is not None:
                self.highway_mode_event.clear()
            return

        if sign_name == "parking":
            self._request_parking_mode(str(current_mode or "").lower())

    def _resolve_speed_cap(self):
        if self._speed_limit is None:
            return None
        return float(self._speed_limit)

    def decide(self, now: float, tracking_state=None) -> ManeuverDecision:
        maneuver_type = str(getattr(tracking_state, "maneuver_type", "none") or "none")
        route_active = bool(getattr(tracking_state, "route_active", False))
        waypoint_mode_active = bool(getattr(tracking_state, "waypoint_mode_active", False))
        mission_state = self._planner_mission_state(tracking_state)

        planner_priority = route_active and (
            waypoint_mode_active
            or maneuver_type in {
                "stopline",
                "turn_left",
                "turn_right",
                "intersection_straight",
                "roundabout",
                "crosswalk",
                "traffic_light",
                "parking_search",
            }
        )

        if self._traffic_light_blocked:
            if now <= self._traffic_light_until:
                return ManeuverDecision(
                    control_mode="MANEUVER_CONTROL",
                    active_maneuver="red_light",
                    mission_state="WAIT_TRAFFIC_LIGHT",
                    speed_override=0.0,
                    planner_priority=True,
                    notes="traffic_light_hold",
                )
            self._traffic_light_blocked = False

        if now <= self._stop_until:
            return ManeuverDecision(
                control_mode="MANEUVER_CONTROL",
                active_maneuver=self._active_maneuver or "stop",
                mission_state="WAIT_STOP_SIGN",
                speed_override=0.0,
                planner_priority=True,
                notes="stop_hold",
            )

        if now <= self._crosswalk_until:
            return ManeuverDecision(
                control_mode="MANEUVER_CONTROL",
                active_maneuver="crosswalk",
                mission_state="CROSSWALK_SLOW",
                speed_override=self.low_speed,
                planner_priority=True,
                speed_cap=self.low_speed,
                notes="crosswalk_slow",
            )

        if now <= self._yellow_until:
            return ManeuverDecision(
                control_mode="MANEUVER_CONTROL",
                active_maneuver="yellow_light",
                mission_state="WAIT_TRAFFIC_LIGHT",
                speed_override=self.low_speed,
                planner_priority=True,
                speed_cap=self.low_speed,
                notes="yellow_light_slow",
            )

        return ManeuverDecision(
            control_mode="ROUTE_TRACKING" if planner_priority else "VISUAL_ASSIST",
            active_maneuver="none",
            mission_state=mission_state,
            speed_cap=self._resolve_speed_cap(),
            planner_priority=planner_priority,
        )
