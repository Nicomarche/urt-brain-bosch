# src/behavior/scenarios/base.py
#
# `BaseScenario` — clase abstracta común. NO es la `IScenario` Protocol
# de `core/interfaces/planner.py`; es una implementación parcial que
# centraliza utilities compartidas (construcción de speed_profile uniforme,
# manejo de fallback BehaviorOutput, helpers de histeresis).
#
# Las subclases concretas (LaneKeep, Intersection, etc.) heredan de
# `BaseScenario` Y satisfacen el Protocol `IScenario` (estructural).
# Esto da SRP (BaseScenario = utilities) + DIP (planner depende de
# IScenario, no de BaseScenario).

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.behavior.context import PlanningContext
from src.behavior.trajectory_builder import build_target_path
from src.core.types.behavior import BehaviorOutput, ScenarioName


@dataclass
class HysteresisGate:
    """Gate con histeresis para activación/desactivación estable.

    Uso:
        gate = HysteresisGate(enter=0.4, exit=0.6)
        gate.update(measure)  # recibe valor a comparar
        if gate.active: ...    # estado actual con histeresis aplicada

    Convención: `enter` ≤ `exit`. El gate se activa cuando el valor
    cae bajo `enter`, se desactiva cuando supera `exit`. Útil para
    señales tipo "distancia al stopline" donde queremos activar al
    acercarnos y desactivar cuando ya pasamos.
    """

    enter: float
    exit: float
    active: bool = False

    def update(self, value: float) -> bool:
        if self.active and value > self.exit:
            self.active = False
        elif (not self.active) and value < self.enter:
            self.active = True
        return self.active


class BaseScenario(ABC):
    """Esqueleto común para escenarios de comportamiento.

    Subclases DEBEN definir:
      - `name: str` (tipicamente `ScenarioName.X.value`)
      - `priority: int` (mayor = se evalúa primero)
      - `is_active(ctx) -> bool`
      - `plan(ctx) -> BehaviorOutput`

    Helpers provistos:
      - `_build_constant_speed_plan(...)`: caso común — construir un
        BehaviorOutput a velocidad constante siguiendo la lanelet actual.
      - `_fallback_plan(...)`: BehaviorOutput inválido para situaciones
        en las que `is_active` dice True pero `plan` no puede producir
        un resultado utilizable.
    """

    name: str = "base"
    priority: int = 0

    @abstractmethod
    def is_active(self, ctx: PlanningContext) -> bool: ...

    @abstractmethod
    def plan(self, ctx: PlanningContext) -> BehaviorOutput: ...

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
    def _build_constant_speed_plan(
        self,
        ctx: PlanningContext,
        target_speed_mps: float,
        scenario_name: str,
        notes: dict | None = None,
    ) -> BehaviorOutput:
        """Construye un plan que sigue la lanelet actual a velocidad constante.

        Los detalles que comparten todos los scenarios "siga el lane":
          - Resolver lanelet actual desde `ctx.route.current_lanelet_id`
            (poblado por el RoutePlanner cuando hay LaneletMap).
          - Si no hay LaneletMap o no hay lanelet — fallback inválido.
          - Llamar a `build_target_path` para resolver Frenet.
          - speed_profile uniforme.
        """
        if ctx.lanelet_map is None or not ctx.route.current_lanelet_id:
            return self._fallback_plan(ctx, reason="no_lanelet_map_or_id")

        target_path = build_target_path(
            lanelet_map=ctx.lanelet_map,
            start_lanelet_id=ctx.route.current_lanelet_id,
            start_xy=(ctx.pose.fused_pose.x, ctx.pose.fused_pose.y),
            target_speed_mps=target_speed_mps,
            horizon_n=ctx.horizon_n,
            dt=ctx.dt,
            next_lanelet_hint_ids=ctx.route.next_lanelet_ids,
        )
        speed_profile = np.full(ctx.horizon_n, float(target_speed_mps), dtype=float)

        return BehaviorOutput(
            timestamp=ctx.now_s,
            dt=ctx.dt,
            target_path=target_path,
            speed_profile=speed_profile,
            scenario_name=scenario_name,
            valid=True,
            stop_required=False,
            notes=notes or {},
        )

    def _fallback_plan(self, ctx: PlanningContext, reason: str) -> BehaviorOutput:
        """BehaviorOutput inválido — el `safety_gate` toma control.

        Devuelve speed_profile = ceros y `valid=False`. El controller
        debe llamar `safety_gate.fallback()` cuando vea esto.
        """
        return BehaviorOutput(
            timestamp=ctx.now_s if ctx is not None else time.time(),
            dt=ctx.dt if ctx is not None else 0.05,
            target_path=np.zeros((1, 3)),
            speed_profile=np.zeros(0),
            scenario_name=ScenarioName.FALLBACK.value,
            valid=False,
            stop_required=True,
            notes={"reason": reason},
        )
