# src/behavior/scenarios/lane_keep.py
#
# `LaneKeep` — el escenario default. Activo siempre que ningún otro
# scenario más específico lo esté. Implementa "seguí el carril a
# velocidad nominal" — el comportamiento al 95% del tiempo en pista.
#
# `priority = 0` — se evalúa último; cualquier scenario con priority
# positivo gana. Se podría modelar como "implicit fallback" en el
# planner, pero tener un scenario explícito hace que la cadena de
# decisiones sea uniforme y testeable.

from __future__ import annotations

from src.behavior.context import PlanningContext
from src.behavior.scenarios.base import BaseScenario
from src.core.types.behavior import BehaviorOutput, ScenarioName


class LaneKeep(BaseScenario):
    """Sigue el centerline a velocidad nominal."""

    name = ScenarioName.LANE_KEEP.value
    priority = 0  # último en evaluarse

    def is_active(self, ctx: PlanningContext) -> bool:
        # LaneKeep es el comportamiento default: siempre podemos intentar
        # seguir lane si hay lanelet asignada. Si no, devolvemos True
        # igual y `plan(ctx)` produce el fallback (el controller verá
        # invalid y aplicará safety_gate).
        return True

    def plan(self, ctx: PlanningContext) -> BehaviorOutput:
        import logging as _log
        _log.getLogger(__name__).debug(
            "[ LaneKeep ] plan: nominal=%.3f lanelet=%r",
            ctx.nominal_speed_mps,
            ctx.route.current_lanelet_id,
        )
        return self._build_constant_speed_plan(
            ctx=ctx,
            target_speed_mps=ctx.nominal_speed_mps,
            scenario_name=self.name,
            notes={"reason": "lane_keep_default"},
        )
