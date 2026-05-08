# src/behavior/scenarios/roundabout.py
#
# `Roundabout` — desacelera al aproximarse a una rotonda; yield al
# tráfico en la rotonda. Detección via `ATTR_ROUNDABOUT` (6) en la
# lanelet actual o sucesoras inmediatas.
#
# Implementación en Phase 4: lógica básica equivalente a Intersection,
# con velocidad reducida y yield a TrackedObject que estén ya dentro
# del círculo. El refinamiento (gap acceptance, geometría circular) es
# trabajo de iteración posterior.

from __future__ import annotations

import math
from dataclasses import replace

from src.behavior.context import PlanningContext
from src.behavior.scenarios.base import BaseScenario
from src.core.types.behavior import BehaviorOutput, ScenarioName
from src.routing.lanelet.attributes import ATTR_ROUNDABOUT


_ROUNDABOUT_SPEED_MPS = 0.35
_YIELD_DISTANCE_M = 3.5


class Roundabout(BaseScenario):
    """Desacelera + yield para entrar a una rotonda."""

    name = ScenarioName.ROUNDABOUT.value
    priority = 50  # menor que crosswalk/intersection

    def is_active(self, ctx: PlanningContext) -> bool:
        if ctx.lanelet_map is None or not ctx.route.current_lanelet_id:
            return False
        ll = ctx.lanelet_map.get_lanelet(ctx.route.current_lanelet_id)
        if ll is None:
            return False
        if int(ll.attribute) == ATTR_ROUNDABOUT:
            return True
        # ¿Sucesora inmediata es rotonda?
        for succ_id in ll.successor_ids[:2]:
            succ = ctx.lanelet_map.get_lanelet(succ_id)
            if succ and int(succ.attribute) == ATTR_ROUNDABOUT:
                return True
        return False

    def plan(self, ctx: PlanningContext) -> BehaviorOutput:
        notes: dict = {"reason": "roundabout_active"}
        if self._yield_required(ctx):
            notes["yield"] = True
            plan = self._build_constant_speed_plan(
                ctx=ctx,
                target_speed_mps=0.0,
                scenario_name=self.name,
                notes=notes,
            )
            return replace(plan, stop_required=True)
        return self._build_constant_speed_plan(
            ctx=ctx,
            target_speed_mps=_ROUNDABOUT_SPEED_MPS,
            scenario_name=self.name,
            notes=notes,
        )

    def _yield_required(self, ctx: PlanningContext) -> bool:
        """Yield si hay vehículo trackeado dentro del círculo de rotonda
        y a < _YIELD_DISTANCE_M del ego."""
        ego_x, ego_y = ctx.pose.fused_pose.x, ctx.pose.fused_pose.y
        for track in ctx.tracked_objects:
            if track.class_name.lower() not in ("car", "vehicle", "motorcycle"):
                continue
            if track.age_frames < 3:
                continue
            tx, ty = track.position_world_xy
            if math.hypot(tx - ego_x, ty - ego_y) <= _YIELD_DISTANCE_M:
                return True
        return False
