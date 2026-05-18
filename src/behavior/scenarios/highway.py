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

from dataclasses import replace

from src.behavior.context import PlanningContext
from src.behavior.scenarios.base import BaseScenario
from src.behavior.sign_utils import nearest_sign_hint
from src.core.types.behavior import BehaviorOutput, MotionState, ScenarioName
from src.routing.lanelet.attributes import (
    ATTR_HIGHWAY_LEFT,
    ATTR_HIGHWAY_RIGHT,
)


try:
    from config import (
        BEHAVIOR_HIGHWAY_MIN_SPEED_MPS as _HIGHWAY_MIN_SPEED_MPS,
        BEHAVIOR_HIGHWAY_SPEED_MPS as _HIGHWAY_SPEED_MPS,
    )
except Exception:
    _HIGHWAY_SPEED_MPS = 0.80
    _HIGHWAY_MIN_SPEED_MPS = 0.40


class Highway(BaseScenario):
    """Sube velocidad en tramos highway (ATTR_HIGHWAY_LEFT/RIGHT)."""

    name = ScenarioName.HIGHWAY.value
    priority = 30  # menor que intersection/crosswalk; mayor que lane_keep

    def __init__(self) -> None:
        self._highway_active = False

    def is_active(self, ctx: PlanningContext) -> bool:
        if nearest_sign_hint(ctx, {"highway_exit", "highway_end"}, max_distance_m=6.0) is not None:
            self._highway_active = False
            return False
        if nearest_sign_hint(ctx, {"highway_entrance", "highway_entry"}, max_distance_m=6.0) is not None:
            self._highway_active = True
            return True
        if ctx.lanelet_map is not None and ctx.route.current_lanelet_id:
            ll = ctx.lanelet_map.get_lanelet(ctx.route.current_lanelet_id)
            if ll is not None and int(ll.attribute) in (ATTR_HIGHWAY_LEFT, ATTR_HIGHWAY_RIGHT):
                self._highway_active = True
                return True
        return self._highway_active

    def plan(self, ctx: PlanningContext) -> BehaviorOutput:
        # La velocidad highway puede ser mayor que la nominal pero
        # nunca más alta que max_speed_mps (el velocity_overlay capea).
        target = min(max(_HIGHWAY_SPEED_MPS, float(_HIGHWAY_MIN_SPEED_MPS)), ctx.max_speed_mps)
        plan = self._build_constant_speed_plan(
            ctx=ctx,
            target_speed_mps=target,
            scenario_name=self.name,
            notes={
                "reason": "highway_segment",
                "highway_active": True,
                "min_moving_speed_mps": float(_HIGHWAY_MIN_SPEED_MPS),
            },
        )
        return replace(plan, motion_state=MotionState.HIGHWAY_CRUISE.value)
