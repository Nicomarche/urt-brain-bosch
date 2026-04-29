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

from src.behavior.context import PlanningContext
from src.behavior.scenarios.base import BaseScenario
from src.core.types.behavior import BehaviorOutput, ScenarioName


_PARKING_APPROACH_SPEED_MPS = 0.25


class Parking(BaseScenario):
    """Maniobra de estacionamiento — Phase 4: solo desaceleración."""

    name = ScenarioName.PARKING.value
    priority = 80  # alta — supera intersection/highway si el sign apareció

    def is_active(self, ctx: PlanningContext) -> bool:
        # Activación 1: maneuver_type explícito en la ruta.
        if ctx.route.maneuver_type == "parking":
            return True
        # Activación 2: sign hint reciente con kind == "parking".
        for hint in ctx.sign_hints:
            kind = str(hint.get("kind", "")).lower()
            distance = float(hint.get("distance_m", 999.0))
            if kind == "parking" and distance < 5.0:
                return True
        return False

    def plan(self, ctx: PlanningContext) -> BehaviorOutput:
        return self._build_constant_speed_plan(
            ctx=ctx,
            target_speed_mps=_PARKING_APPROACH_SPEED_MPS,
            scenario_name=self.name,
            notes={"reason": "parking_approach"},
        )
