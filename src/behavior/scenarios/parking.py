# src/behavior/scenarios/parking.py
#
# `Parking` — escenario para maniobra de estacionamiento. Activación
# se basa en `sign_hints` ("parking" sign reciente) o en una flag
# explícita en RouteContext (`maneuver_type == "parking"`).
#
# IMPLEMENTACIÓN EN FASE 4 — DELIBERADAMENTE BÁSICA:
#   - Si está activo, baja velocidad a `_PARKING_APPROACH_SPEED_MPS`.
#   - El plan sigue el centerline normal (no genera trayectoria reverse
#     ni S-curve aún).
#
# El refinamiento (cálculo de pose final del bay, trayectoria con
# reverse, detección "spot ocupado") es trabajo de iteración. La
# arquitectura permite agregarlo subclaseando o reescribiendo `plan()`
# sin tocar el resto del planner.

from __future__ import annotations

import time

from src.behavior.context import PlanningContext
from src.behavior.scenarios.base import BaseScenario
from src.behavior.sign_utils import nearest_sign_hint
from src.core.types.behavior import BehaviorOutput, ScenarioName


try:
    import config as _cfg
except Exception:  # pragma: no cover
    _cfg = None

_PARKING_APPROACH_SPEED_MPS = float(
    getattr(_cfg, "TRAFFIC_SIGN_LOW_SPEED_MPS", 0.10) if _cfg is not None else 0.10
)
_PARKING_TRIGGER_DISTANCE_M = 0.01 * float(
    getattr(_cfg, "PARKING_TRIGGER_DISTANCE_CM", 100.0) if _cfg is not None else 100.0
)
_PARKING_SPOT_MISS_TICKS = int(
    getattr(_cfg, "PARKING_SPOT_MISS_THRESHOLD", 8) if _cfg is not None else 8
)
_WAIT_STEER_S = float(getattr(_cfg, "PARKING_T_WAIT_STEER", 1.0) if _cfg is not None else 1.0)
_T_FORWARD = float(getattr(_cfg, "PARKING_T_FORWARD", 1.5) if _cfg is not None else 1.5)
_T_REV_ENTRY = float(getattr(_cfg, "PARKING_T_REVERSING_ENTRY", 3.0) if _cfg is not None else 3.0)
_T_REV_ALIGN = float(getattr(_cfg, "PARKING_T_REVERSING_ALIGN", 1.5) if _cfg is not None else 1.5)
_T_FORWARD_CORR = float(getattr(_cfg, "PARKING_T_FORWARD_CORR", 1.5) if _cfg is not None else 1.5)
_ENTRY_STEER_LEGACY_DEG = float(getattr(_cfg, "PARKING_ENTRY_STEER", 25.0) if _cfg is not None else 25.0)
_ALIGN_STEER_LEGACY_DEG = float(getattr(_cfg, "PARKING_ALIGN_STEER", -25.0) if _cfg is not None else -25.0)


class Parking(BaseScenario):
    """Parking search + direct maneuver commands through MotionController."""

    name = ScenarioName.PARKING.value
    priority = 80  # alta — supera intersection/highway si el sign apareció

    def __init__(self) -> None:
        self._active = False
        self._state = "search"
        self._state_start_s = 0.0
        self._miss_ticks = 0

    def is_active(self, ctx: PlanningContext) -> bool:
        # Activación 1: maneuver_type explícito en la ruta.
        if ctx.route.maneuver_type == "parking":
            self._active = True
            return True
        # Activación 2: sign hint reciente con kind == "parking".
        if nearest_sign_hint(ctx, {"parking"}, max_distance_m=5.0) is not None:
            self._active = True
            return True
        return bool(self._active and self._state != "done")

    def plan(self, ctx: PlanningContext) -> BehaviorOutput:
        sign = nearest_sign_hint(ctx, {"parking"}, max_distance_m=5.0)
        spot_visible = sign is not None
        spot_distance_m = _hint_distance_m(sign)
        spot_occupied = self._spot_occupied(ctx)
        notes = {
            "reason": "parking_approach",
            "parking_state": self._state,
            "spot_visible": bool(spot_visible),
            "spot_distance_m": spot_distance_m,
            "min_moving_speed_mps": float(_PARKING_APPROACH_SPEED_MPS),
        }
        if spot_occupied:
            notes["spot_occupied"] = True

        if self._state in {"search", "spot_tracked"}:
            self._update_search_state(
                spot_visible=spot_visible,
                spot_distance_m=spot_distance_m,
                spot_occupied=spot_occupied,
                now_s=float(ctx.now_s or time.time()),
            )
            notes["parking_state"] = self._state
        else:
            direct = self._maneuver_direct_command(float(ctx.now_s or time.time()))
            if direct is not None:
                notes["direct_motor_command"] = direct
                notes["parking_state"] = self._state

        return self._build_constant_speed_plan(
            ctx=ctx,
            target_speed_mps=_PARKING_APPROACH_SPEED_MPS,
            scenario_name=self.name,
            notes=notes,
        )

    def _update_search_state(
        self,
        *,
        spot_visible: bool,
        spot_distance_m: float | None,
        spot_occupied: bool,
        now_s: float,
    ) -> None:
        if spot_occupied:
            self._state = "search"
            self._miss_ticks = 0
            return
        if spot_visible:
            if spot_distance_m is None or spot_distance_m <= _PARKING_TRIGGER_DISTANCE_M:
                self._state = "spot_tracked"
            self._miss_ticks = 0
            return
        if self._state == "spot_tracked":
            self._miss_ticks += 1
            if self._miss_ticks >= _PARKING_SPOT_MISS_TICKS:
                self._advance("forward_past_spot", now_s)

    def _advance(self, state: str, now_s: float) -> None:
        self._state = state
        self._state_start_s = float(now_s)
        self._miss_ticks = 0

    def _maneuver_direct_command(self, now_s: float) -> dict | None:
        elapsed = max(0.0, float(now_s) - float(self._state_start_s))
        if self._state == "forward_past_spot":
            if elapsed >= _T_FORWARD:
                self._advance("wait_steer_1", now_s)
            return _direct(0.0, _PARKING_APPROACH_SPEED_MPS, "parking_forward_past_spot")
        if self._state == "wait_steer_1":
            if elapsed >= _WAIT_STEER_S:
                self._advance("reversing_entry", now_s)
            return _direct(_legacy_to_iso_steer(_ENTRY_STEER_LEGACY_DEG), 0.0, "parking_wait_steer_1")
        if self._state == "reversing_entry":
            if elapsed >= _T_REV_ENTRY:
                self._advance("wait_steer_2", now_s)
            return _direct(_legacy_to_iso_steer(_ENTRY_STEER_LEGACY_DEG), -_PARKING_APPROACH_SPEED_MPS, "parking_reversing_entry", allow_reverse=True)
        if self._state == "wait_steer_2":
            if elapsed >= _WAIT_STEER_S:
                self._advance("reversing_align", now_s)
            return _direct(_legacy_to_iso_steer(_ALIGN_STEER_LEGACY_DEG), 0.0, "parking_wait_steer_2")
        if self._state == "reversing_align":
            if elapsed >= _T_REV_ALIGN:
                self._advance("wait_steer_3", now_s)
            return _direct(_legacy_to_iso_steer(_ALIGN_STEER_LEGACY_DEG), -_PARKING_APPROACH_SPEED_MPS, "parking_reversing_align", allow_reverse=True)
        if self._state == "wait_steer_3":
            if elapsed >= _WAIT_STEER_S:
                self._advance("forward_correction", now_s)
            return _direct(_legacy_to_iso_steer(_ENTRY_STEER_LEGACY_DEG), 0.0, "parking_wait_steer_3")
        if self._state == "forward_correction":
            if elapsed >= _T_FORWARD_CORR:
                self._advance("parked", now_s)
            return _direct(_legacy_to_iso_steer(_ENTRY_STEER_LEGACY_DEG), _PARKING_APPROACH_SPEED_MPS, "parking_forward_correction")
        if self._state == "parked":
            self._active = False
            return _direct(0.0, 0.0, "parking_parked")
        return None

    def _spot_occupied(self, ctx: PlanningContext) -> bool:
        for obs in ctx.lidar_obstacles:
            if 0.15 <= float(obs.x_m) <= 0.90 and abs(float(obs.y_m)) <= 0.35:
                return True
        for track in ctx.tracked_objects:
            if str(track.class_name).lower() not in {"car", "vehicle", "obstacle"}:
                continue
            if track.age_frames >= 2:
                return True
        return False


def _hint_distance_m(hint: dict | None) -> float | None:
    if not isinstance(hint, dict):
        return None
    value = hint.get("distance_m")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _legacy_to_iso_steer(legacy_deg: float) -> float:
    # Legacy parking constants use +right; MotorCommand uses +left.
    return -float(legacy_deg)


def _direct(
    steering_deg: float,
    speed_mps: float,
    reason: str,
    *,
    allow_reverse: bool = False,
) -> dict:
    return {
        "steering_deg": float(steering_deg),
        "speed_mps": float(speed_mps),
        "allow_reverse": bool(allow_reverse),
        "reason": str(reason),
    }
