# src/behavior/planner.py
#
# `BehaviorPlanner` — orquestador de scenarios. Es la **única fuente de
# verdad de velocidad** del sistema: nadie más decide cuánto va a ir el
# auto. Es lo que el plan global llama "single source of truth".
#
# Responsabilidades:
#   1. Mantener una lista de scenarios (cada uno implementa IScenario).
#   2. Por tick, iterar scenarios por prioridad descendente.
#   3. El primero que `is_active(ctx) == True` produce el plan
#      `scenario.plan(ctx) -> BehaviorOutput`.
#   4. Aplicar `velocity_overlay` (caps regulatorios) sobre ese plan.
#   5. Devolver el `BehaviorOutput` final.
#
# NO responsabilidades:
#   - Decidir lógica concreta de scenarios (vive en cada scenario).
#   - Construir target_path (vive en `trajectory_builder.build_target_path`).
#   - Aplicar caps regulatorios (vive en `velocity_overlay.apply_overlay`).
#   - Subscribirse a topics o publicar mensajes (vive en `planner_thread.py`).
#
# Esta separación garantiza:
#   - SRP: el planner sólo orquesta.
#   - OCP: agregar un scenario nuevo es subclasear IScenario y agregarlo
#     a la lista en el constructor o vía `add_scenario()`.
#   - DIP: el planner depende de IScenario (Protocol), no de clases
#     concretas. Se puede testear con mocks.

from __future__ import annotations

import logging

from src.behavior.context import PlanningContext
from src.behavior.velocity_overlay import apply_overlay
from src.core.interfaces.planner import IScenario
from src.core.types.behavior import BehaviorOutput, ScenarioName

_log = logging.getLogger(__name__)


class BehaviorPlanner:
    """Orquesta una lista de IScenario para producir BehaviorOutput por tick.

    El planner es **stateless en términos de decisión**: el estado
    (qué scenario estuvo activo el tick pasado, histeresis, etc.) lo
    mantienen los scenarios. Esto hace el planner trivialmente testeable.
    """

    def __init__(self, scenarios: list[IScenario] | None = None) -> None:
        self._scenarios: list[IScenario] = []
        if scenarios:
            for s in scenarios:
                self.add_scenario(s)

    # --------------------------------------------------------------
    # Registro de scenarios
    # --------------------------------------------------------------
    def add_scenario(self, scenario: IScenario) -> None:
        """Agrega un scenario y reordena por prioridad descendente.

        Idempotente: agregar dos veces el mismo objeto NO duplica.
        """
        if scenario in self._scenarios:
            return
        self._scenarios.append(scenario)
        # Mayor prioridad primero. Si dos comparten priority, el orden de
        # inserción decide (estable).
        self._scenarios.sort(key=lambda s: -getattr(s, "priority", 0))

    @property
    def scenarios(self) -> tuple[IScenario, ...]:
        return tuple(self._scenarios)

    # --------------------------------------------------------------
    # Plan
    # --------------------------------------------------------------
    def plan(self, ctx: PlanningContext) -> BehaviorOutput:
        """Itera scenarios por prioridad. Primero activo gana, luego overlay."""
        chosen: IScenario | None = None
        for scenario in self._scenarios:
            try:
                if scenario.is_active(ctx):
                    chosen = scenario
                    break
            except Exception as exc:
                # Un scenario que crashea NO debe romper el planner —
                # logueamos y seguimos con el siguiente.
                _log.exception(
                    "scenario %s.is_active() lanzó excepción: %s",
                    getattr(scenario, "name", "?"),
                    exc,
                )

        if chosen is None:
            # Ningún scenario activo — fallback estructurado.
            return _empty_fallback(ctx)

        try:
            raw_plan = chosen.plan(ctx)
        except Exception as exc:
            _log.exception(
                "scenario %s.plan() lanzó excepción: %s",
                getattr(chosen, "name", "?"),
                exc,
            )
            return _empty_fallback(ctx, reason=f"scenario_crash:{getattr(chosen, 'name', '?')}")

        if not isinstance(raw_plan, BehaviorOutput):
            _log.error(
                "scenario %s devolvió tipo inesperado %s",
                getattr(chosen, "name", "?"),
                type(raw_plan).__name__,
            )
            return _empty_fallback(ctx, reason="bad_scenario_output_type")

        # Aplicar caps regulatorios.
        return apply_overlay(plan=raw_plan, route=ctx.route, max_speed_mps=ctx.max_speed_mps)


def _empty_fallback(ctx: PlanningContext, reason: str = "no_scenario_active") -> BehaviorOutput:
    """BehaviorOutput inválido para que el `safety_gate` tome control."""
    import numpy as np

    return BehaviorOutput(
        timestamp=ctx.now_s,
        dt=ctx.dt,
        target_path=np.zeros((1, 3)),
        speed_profile=np.zeros(0),
        scenario_name=ScenarioName.FALLBACK.value,
        valid=False,
        stop_required=True,
        notes={"reason": reason},
    )
