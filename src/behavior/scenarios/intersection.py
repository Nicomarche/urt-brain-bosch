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
from src.core.types.behavior import BehaviorOutput, MotionState, ScenarioName
from src.routing.lanelet.attributes import (
    ATTR_INTERSECTION,
    ATTR_INTERSECTION_EXIT,
    ATTR_STOPLINE,
)


_INTERSECTION_SPEED_MPS = 0.40
_INTERSECTION_LOOKAHEAD_M = 3.0
_TTC_HORIZON_S = 2.5
_INTERSECTION_STOPLINE_ARM_M = 0.40  # plan TANDA 1.1: arm visual stopline


class Intersection(BaseScenario):
    """Travesía de intersección — desacelerar, vigilar cross-traffic."""

    name = ScenarioName.INTERSECTION.value
    priority = 60

    def __init__(self) -> None:
        # Estado interno para histeresis. False = no estamos en
        # intersección; True = estamos. Las transiciones se evalúan en
        # `is_active`.
        self._was_active = False
        # Snapshots de la última pasada de is_active() — usados por plan()
        # para clasificar motion_state (APPROACHING / IN / EXITING).
        self._last_is_in_intersection: bool = False
        self._last_attr_now: int = 0
        self._last_approaching: bool = False

    def is_active(self, ctx: PlanningContext) -> bool:
        # Plan TANDA 1.1: la stopline visual estable y cercana también
        # dispara la intersección, aún sin mapa o sin atributos.
        # Razón: en pista nueva el mapa OSM puede no estar re-anotado;
        # la línea blanca en el piso es la señal autoritativa de
        # "frená ya". El umbral 0.40 m es coherente con
        # ``TRAFFIC_SIGN_STOP_COMMAND_DISTANCE_M`` y con el arm_distance
        # ya usado en el visual_state_machine.
        stopline_visual_trigger = self._stopline_visual_trigger(ctx)

        if ctx.lanelet_map is None or not ctx.route.current_lanelet_id:
            # Sin mapa todavía mantenemos histeresis; pero si la
            # stopline visual aparece, activamos igual.
            if stopline_visual_trigger:
                self._was_active = True
                import logging as _log
                _log.getLogger(__name__).info(
                    "[ Intersection ] activación SÓLO por stopline visual "
                    "(bootstrap sin mapa)"
                )
                return True
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
        active_now = is_in_intersection or approaching or stopline_visual_trigger

        # Cache para que plan() pueda clasificar motion_state sin recomputar.
        self._last_attr_now = attr_now
        self._last_is_in_intersection = bool(is_in_intersection)
        self._last_approaching = bool(approaching or stopline_visual_trigger)

        # Histeresis simple: salimos sólo cuando dejamos de estar EN
        # intersección Y no hay otra inminente Y no vemos stopline.
        prev = self._was_active
        if self._was_active:
            if not is_in_intersection and not approaching and not stopline_visual_trigger:
                self._was_active = False
        else:
            self._was_active = active_now

        if self._was_active != prev or self._was_active:
            import logging as _log
            _log.getLogger(__name__).info(
                "[ Intersection ] is_active=%s attr_now=%d lanelet=%r "
                "in_inter=%s approaching=%s stopline_visual=%s",
                self._was_active,
                attr_now,
                ctx.route.current_lanelet_id,
                is_in_intersection,
                approaching,
                stopline_visual_trigger,
            )
        return self._was_active

    @staticmethod
    def _stopline_visual_trigger(ctx: PlanningContext) -> bool:
        """¿Hay stopline visual estable a ≤ _INTERSECTION_STOPLINE_ARM_M?

        El observador de stopline (``lane_observer_thread``) emite la
        observación con ``stable=True`` cuando la línea blanca cae en una
        ventana temporal coherente. Distancia 0 < d < 0.40 m: arm de
        freno. Más cerca de 0 ya es "estamos sobre la línea" y se mantiene
        activo por histeresis. Más lejos de 0.40 m: aún no es accionable
        para evitar falsos activos por líneas de marca pintadas lejanas.
        """
        obs = getattr(ctx, "stopline_observation", None)
        if obs is None or not getattr(obs, "stable", False):
            return False
        distance_m = getattr(obs, "distance_m", None)
        if distance_m is None:
            return False
        try:
            d = float(distance_m)
        except (TypeError, ValueError):
            return False
        return 0.0 < d < _INTERSECTION_STOPLINE_ARM_M

    def plan(self, ctx: PlanningContext) -> BehaviorOutput:
        from dataclasses import replace

        notes: dict = {"reason": "intersection_active"}
        motion_state = self._classify_motion_state()

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
            return replace(plan, stop_required=True, motion_state=motion_state)

        # Si la lanelet actual es STOPLINE explícita, frenar a 0 también:
        # el velocity_overlay hará la rampa, pero acá emitimos baseline 0
        # para que el rampleo sea desde velocidad reducida.
        current = ctx.lanelet_map.get_lanelet(ctx.route.current_lanelet_id) if ctx.lanelet_map else None
        if current and int(current.attribute) == ATTR_STOPLINE:
            notes["stopline_lanelet"] = True
            plan = self._build_constant_speed_plan(
                ctx=ctx,
                target_speed_mps=0.0,
                scenario_name=self.name,
                notes=notes,
            )
            return replace(plan, motion_state=motion_state)

        plan = self._build_constant_speed_plan(
            ctx=ctx,
            target_speed_mps=_INTERSECTION_SPEED_MPS,
            scenario_name=self.name,
            notes=notes,
        )
        return replace(plan, motion_state=motion_state)

    def _classify_motion_state(self) -> str:
        """Map attr_now → APPROACHING / IN / EXITING.

        ATTR_INTERSECTION (2) = adentro del centro de la intersección.
        ATTR_INTERSECTION_EXIT = saliendo (rama post-intersección con histeresis).
        approaching (sin estar adentro) = pre-frenado, stopline visible o regulator.
        """
        if self._last_attr_now == ATTR_INTERSECTION_EXIT:
            return MotionState.EXITING_INTERSECTION.value
        if self._last_is_in_intersection:
            return MotionState.IN_INTERSECTION.value
        if self._last_approaching:
            return MotionState.APPROACHING_INTERSECTION.value
        return MotionState.MOVING.value

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
