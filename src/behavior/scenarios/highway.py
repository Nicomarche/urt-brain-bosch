# src/behavior/scenarios/highway.py
#
# `Highway` — sube velocidad nominal cuando la lanelet actual es
# ATTR_HIGHWAY_LEFT (4) o ATTR_HIGHWAY_RIGHT (5). Es el ÚNICO scenario
# que puede aumentar la velocidad sobre `nominal_speed_mps`. El cap
# absoluto sigue siendo `max_speed_mps` (lo aplica el velocity_overlay).
#
# Histeresis: activo cuando ATTR es highway; desactivo en el primer
# step donde la lanelet sea otra cosa. Sin lookahead — el cambio de
# atributo en BFMC es muy localizado y no genera oscilación.

from __future__ import annotations

from src.behavior.context import PlanningContext
from src.behavior.scenarios.base import BaseScenario
from src.core.types.behavior import BehaviorOutput, ScenarioName
from src.routing.lanelet.from_graphml import (
    ATTR_HIGHWAY_LEFT,
    ATTR_HIGHWAY_RIGHT,
)


_HIGHWAY_SPEED_MPS = 0.80


class Highway(BaseScenario):
    """Sube velocidad en tramos highway (ATTR_HIGHWAY_LEFT/RIGHT)."""

    name = ScenarioName.HIGHWAY.value
    priority = 30  # menor que intersection/crosswalk; mayor que lane_keep

    def is_active(self, ctx: PlanningContext) -> bool:
        if ctx.lanelet_map is None or not ctx.route.current_lanelet_id:
            return False
        ll = ctx.lanelet_map.get_lanelet(ctx.route.current_lanelet_id)
        if ll is None:
            return False
        return int(ll.attribute) in (ATTR_HIGHWAY_LEFT, ATTR_HIGHWAY_RIGHT)

    def plan(self, ctx: PlanningContext) -> BehaviorOutput:
        # La velocidad highway puede ser mayor que la nominal pero
        # nunca más alta que max_speed_mps (el velocity_overlay capea).
        target = min(_HIGHWAY_SPEED_MPS, ctx.max_speed_mps)
        return self._build_constant_speed_plan(
            ctx=ctx,
            target_speed_mps=target,
            scenario_name=self.name,
            notes={"reason": "highway_segment"},
        )
