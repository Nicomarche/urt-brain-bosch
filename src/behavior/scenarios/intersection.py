# src/behavior/scenarios/intersection.py
#
# `Intersection` — el escenario que se activa cuando el ego está
# atravesando o por entrar a una intersección. Comportamiento:
#   - Reduce velocidad a `_INTERSECTION_SPEED_MPS`.
#   - Si hay objeto trackeado cruzando (vector de velocidad atravesando
#     la trayectoria del ego dentro de los próximos `_TTC_HORIZON_S`),
#     marca `stop_required = True`.
#   - El BehaviorPlanner aplicará después el `velocity_overlay`, que
#     puede agregar caps adicionales por crosswalks que coincidan con
#     la intersección.
#
# Detección de "estoy en intersección":
#   1. La lanelet actual tiene `attribute == ATTR_INTERSECTION` (2),
#      o sus sucesoras inmediatas lo tienen (entrando).
#   2. RouteContext lo marca con `next_semantic_type == "intersection"`
#      o `expected_control_type == "intersection"` (legacy).
#
# Histeresis: una vez activo, sigue activo hasta que la lanelet actual
# Y todas las sucesoras dentro de 3 m sean ATTR_NORMAL otra vez.

from __future__ import annotations

import math

from src.behavior.context import PlanningContext
from src.behavior.scenarios.base import BaseScenario
from src.core.types.behavior import BehaviorOutput, ScenarioName
from src.routing.lanelet.attributes import (
    ATTR_INTERSECTION,
    ATTR_INTERSECTION_EXIT,
    ATTR_STOPLINE,
)


_INTERSECTION_SPEED_MPS = 0.40
_INTERSECTION_LOOKAHEAD_M = 3.0
_TTC_HORIZON_S = 2.5


class Intersection(BaseScenario):
    """Travesía de intersección — desacelerar, vigilar cross-traffic."""

    name = ScenarioName.INTERSECTION.value
    priority = 60

    def __init__(self) -> None:
        # Estado interno para histeresis. False = no estamos en
        # intersección; True = estamos. Las transiciones se evalúan en
        # `is_active`.
        self._was_active = False

    def is_active(self, ctx: PlanningContext) -> bool:
        if ctx.lanelet_map is None or not ctx.route.current_lanelet_id:
            return self._was_active  # holdover si bootstrap

        # Atributo de la lanelet actual.
        current = ctx.lanelet_map.get_lanelet(ctx.route.current_lanelet_id)
        attr_now = int(current.attribute) if current else 0

        # ¿Hay intersection ahead dentro de _INTERSECTION_LOOKAHEAD_M?
        approaching = False
        if attr_now != ATTR_INTERSECTION:
            for reg in ctx.route.regulatory_ahead:
                kind = reg.kind.lower()
                dist = float(reg.data.get("distance_m", 0.0)) if reg.data else 0.0
                if kind == "intersection" and dist <= _INTERSECTION_LOOKAHEAD_M:
                    approaching = True
                    break

        is_in_intersection = attr_now in (ATTR_INTERSECTION, ATTR_INTERSECTION_EXIT)
        active_now = is_in_intersection or approaching

        # Histeresis simple: salimos sólo cuando dejamos de estar EN
        # intersección Y no hay otra inminente.
        prev = self._was_active
        if self._was_active:
            if not is_in_intersection and not approaching:
                self._was_active = False
        else:
            self._was_active = active_now

        if self._was_active != prev or self._was_active:
            import logging as _log
            _log.getLogger(__name__).info(
                "[ Intersection ] is_active=%s attr_now=%d lanelet=%r "
                "in_inter=%s approaching=%s",
                self._was_active,
                attr_now,
                ctx.route.current_lanelet_id,
                is_in_intersection,
                approaching,
            )
        return self._was_active

    def plan(self, ctx: PlanningContext) -> BehaviorOutput:
        notes: dict = {"reason": "intersection_active"}

        # Cross-traffic: si algún tracked_object cruza nuestra
        # trayectoria dentro del TTC, paramos.
        if self._cross_traffic_blocking(ctx):
            notes["cross_traffic"] = True
            plan = self._build_constant_speed_plan(
                ctx=ctx,
                target_speed_mps=0.0,  # parar
                scenario_name=self.name,
                notes=notes,
            )
            # Aseguramos stop_required=True (build_constant_speed_plan no lo levanta).
            from dataclasses import replace
            return replace(plan, stop_required=True)

        # Si la lanelet actual es STOPLINE explícita, frenar a 0 también:
        # el velocity_overlay hará la rampa, pero acá emitimos baseline 0
        # para que el rampleo sea desde velocidad reducida.
        current = ctx.lanelet_map.get_lanelet(ctx.route.current_lanelet_id) if ctx.lanelet_map else None
        if current and int(current.attribute) == ATTR_STOPLINE:
            notes["stopline_lanelet"] = True
            return self._build_constant_speed_plan(
                ctx=ctx,
                target_speed_mps=0.0,
                scenario_name=self.name,
                notes=notes,
            )

        return self._build_constant_speed_plan(
            ctx=ctx,
            target_speed_mps=_INTERSECTION_SPEED_MPS,
            scenario_name=self.name,
            notes=notes,
        )

    # --------------------------------------------------------------
    # Internals
    # --------------------------------------------------------------
    def _cross_traffic_blocking(self, ctx: PlanningContext) -> bool:
        """¿Hay objeto trackeado con TTC < _TTC_HORIZON_S sobre el path?

        Heurística simple: para cada track, calcular la distancia del
        track al ego y la velocidad de aproximación. Si TTC < horizonte
        y track está delante (en el sentido de avance), bloquear.

        Evita falsos positivos descartando tracks con `age_frames < 3`
        (recién detectados, posibles transitorios).
        """
        ego_x, ego_y = ctx.pose.fused_pose.x, ctx.pose.fused_pose.y
        ego_psi = ctx.pose.fused_pose.yaw
        cos_psi = math.cos(ego_psi)
        sin_psi = math.sin(ego_psi)

        for track in ctx.tracked_objects:
            if track.age_frames < 3:
                continue
            tx, ty = track.position_world_xy
            dx, dy = tx - ego_x, ty - ego_y
            # Vector ego→track en coordenadas del cuerpo (eje x adelante).
            forward = dx * cos_psi + dy * sin_psi
            if forward <= 0.0:
                continue  # objeto detrás
            distance = math.hypot(dx, dy)
            # Velocidad relativa: track alejándose ⇒ no bloquea.
            tvx, tvy = track.velocity_world_xy
            ego_vx = ctx.pose.speed_mps * cos_psi
            ego_vy = ctx.pose.speed_mps * sin_psi
            closing = -(tvx - ego_vx) * (dx / max(distance, 1e-3)) - (tvy - ego_vy) * (
                dy / max(distance, 1e-3)
            )
            if closing <= 0.0:
                continue
            ttc = distance / max(closing, 1e-3)
            if ttc < _TTC_HORIZON_S:
                return True
        return False
