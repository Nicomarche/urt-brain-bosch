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

from dataclasses import replace

from src.behavior.context import PlanningContext
from src.behavior.scenarios.base import BaseScenario
from src.core.types.behavior import BehaviorOutput, MotionState, ScenarioName


# Plan B3 / T2.2: hysteresis de detección de curva. Umbral 0.3 rad/m
# (~3.3m de radio) — más conservador que tight track (que tiene
# curvas <1m radio), pero suficiente para que el dashboard chip y la
# selección de perfil visual sepan que estamos en curva. Pedimos N
# ticks de evidencia consistente antes de transicionar para evitar
# flapping CRUISING ↔ IN_CURVE_* cuando el robot atraviesa puntos de
# inflexión del path.
_CURVE_KAPPA_THRESHOLD_RAD_PER_M = 0.30
_CURVE_HYSTERESIS_TICKS = 3


class LaneKeep(BaseScenario):
    """Sigue el centerline a velocidad nominal."""

    name = ScenarioName.LANE_KEEP.value
    priority = 0  # último en evaluarse

    def __init__(self) -> None:
        # Contadores de hysteresis para clasificación CRUISING vs IN_CURVE_*.
        # Se incrementan tick a tick mientras kappa cumpla el criterio.
        # Se resetean cuando deja de cumplirlo.
        self._left_curve_ticks: int = 0
        self._right_curve_ticks: int = 0

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
        plan = self._build_constant_speed_plan(
            ctx=ctx,
            target_speed_mps=ctx.nominal_speed_mps,
            scenario_name=self.name,
            notes={"reason": "lane_keep_default"},
        )
        motion_state = self._classify_motion_state(ctx)
        return replace(plan, motion_state=motion_state)

    def _classify_motion_state(self, ctx: PlanningContext) -> str:
        """Devuelve el motion_state correcto según curvatura del path.

        Lógica con hysteresis: necesitamos ``_CURVE_HYSTERESIS_TICKS``
        ticks consecutivos con ``|kappa| > threshold`` y signo consistente
        antes de transicionar a IN_CURVE_*. Un solo tick bajo umbral
        resetea el contador.
        """
        kappa = float(getattr(ctx.route, "path_kappa", 0.0) or 0.0)
        if kappa > _CURVE_KAPPA_THRESHOLD_RAD_PER_M:
            self._left_curve_ticks += 1
            self._right_curve_ticks = 0
        elif kappa < -_CURVE_KAPPA_THRESHOLD_RAD_PER_M:
            self._right_curve_ticks += 1
            self._left_curve_ticks = 0
        else:
            self._left_curve_ticks = 0
            self._right_curve_ticks = 0

        if self._left_curve_ticks >= _CURVE_HYSTERESIS_TICKS:
            return MotionState.IN_CURVE_LEFT.value
        if self._right_curve_ticks >= _CURVE_HYSTERESIS_TICKS:
            return MotionState.IN_CURVE_RIGHT.value
        return MotionState.CRUISING.value
