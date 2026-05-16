# src/behavior/scenarios/crosswalk.py
#
# `Crosswalk` — desacelera al aproximarse a un crosswalk; frena si hay
# peatón cerca. Se distingue de `Intersection` en que:
#   - No requiere yielding a tráfico cruzado, sólo a peatones.
#   - Si no hay peatón, sólo bajamos velocidad (no paramos).
#
# Histeresis: activo desde `_CROSSWALK_ENTER_RANGE_M` antes hasta
# `_CROSSWALK_EXIT_RANGE_M` después. La diferencia entre enter y exit
# evita que el scenario thrasheee al cruzar.

from __future__ import annotations

import math

from dataclasses import replace

from src.behavior.context import PlanningContext
from src.behavior.scenarios.base import BaseScenario
from src.behavior.sign_utils import nearest_sign_hint
from src.core.types.behavior import BehaviorOutput, ScenarioName
from src.routing.lanelet.attributes import ATTR_CROSSWALK


_CROSSWALK_ENTER_RANGE_M = 4.0
_CROSSWALK_EXIT_RANGE_M = 1.5
try:
    from config import TRAFFIC_SIGN_LOW_SPEED_MPS as _CROSSWALK_SPEED_MPS
except Exception:
    _CROSSWALK_SPEED_MPS = 0.10
_PEDESTRIAN_BLOCK_DISTANCE_M = 2.5


class Crosswalk(BaseScenario):
    """Slowdown en zona crosswalk; stop si peatón cerca."""

    name = ScenarioName.CROSSWALK.value
    priority = 70  # mayor que Intersection — peatones tienen prioridad absoluta

    def __init__(self) -> None:
        self._active = False

    def is_active(self, ctx: PlanningContext) -> bool:
        sign_hint = nearest_sign_hint(ctx, {"crosswalk"}, max_distance_m=_CROSSWALK_ENTER_RANGE_M)
        if sign_hint is not None:
            self._active = True
            return True

        if ctx.lanelet_map is None or not ctx.route.current_lanelet_id:
            if self._active:
                self._active = False
            return False

        current = ctx.lanelet_map.get_lanelet(ctx.route.current_lanelet_id)
        in_crosswalk_lanelet = current is not None and int(current.attribute) == ATTR_CROSSWALK

        # Buscamos crosswalk en regulatory_ahead.
        nearest_crosswalk_dist = math.inf
        for reg in ctx.route.regulatory_ahead:
            if reg.kind.lower() == "crosswalk":
                d = float(reg.data.get("distance_m", math.inf)) if reg.data else math.inf
                nearest_crosswalk_dist = min(nearest_crosswalk_dist, d)

        if self._active:
            # Permanecer activo hasta que crosswalk_dist supere exit_range.
            if (
                not in_crosswalk_lanelet
                and nearest_crosswalk_dist > _CROSSWALK_EXIT_RANGE_M
            ):
                self._active = False
        else:
            if in_crosswalk_lanelet or nearest_crosswalk_dist <= _CROSSWALK_ENTER_RANGE_M:
                self._active = True

        return self._active

    def plan(self, ctx: PlanningContext) -> BehaviorOutput:
        notes: dict = {"reason": "crosswalk_active"}

        if self._pedestrian_blocking(ctx):
            notes["pedestrian_blocking"] = True
            plan = self._build_constant_speed_plan(
                ctx=ctx,
                target_speed_mps=0.0,
                scenario_name=self.name,
                notes=notes,
            )
            return replace(plan, stop_required=True)

        return self._build_constant_speed_plan(
            ctx=ctx,
            target_speed_mps=_CROSSWALK_SPEED_MPS,
            scenario_name=self.name,
            notes={**notes, "min_moving_speed_mps": float(_CROSSWALK_SPEED_MPS)},
        )

    def _pedestrian_blocking(self, ctx: PlanningContext) -> bool:
        """¿Algún tracked_object con class_name == 'pedestrian' está
        dentro de _PEDESTRIAN_BLOCK_DISTANCE_M y delante del ego?
        """
        ego_x, ego_y = ctx.pose.fused_pose.x, ctx.pose.fused_pose.y
        ego_psi = ctx.pose.fused_pose.yaw
        cos_psi = math.cos(ego_psi)
        sin_psi = math.sin(ego_psi)

        for track in ctx.tracked_objects:
            if track.class_name.lower() not in {"pedestrian", "person"}:
                continue
            if track.age_frames < 2:
                continue
            tx, ty = track.position_world_xy
            dx, dy = tx - ego_x, ty - ego_y
            forward = dx * cos_psi + dy * sin_psi
            if forward < -0.5:  # bien atrás del ego
                continue
            distance = math.hypot(dx, dy)
            if distance <= _PEDESTRIAN_BLOCK_DISTANCE_M:
                return True
        return False
