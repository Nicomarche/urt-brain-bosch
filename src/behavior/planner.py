# src/behavior/planner.py
#
# `BehaviorPlanner` — capa pre-MPC inspirada en Autoware.
#
# Responsabilidades:
#   1. `PlannerManager`: elegir el scene module activo.
#   2. `PathOptimizer`: transformar el path candidato en una trayectoria
#      suave y discretizada para el horizonte del MPC.
#   3. `BehaviorVelocityPlanner`: generar el perfil de velocidad final
#      según reglas regulatorias y restricciones geométricas.
#   4. Entregar un `BehaviorOutput` final al controller/MPC.
#
# NO responsabilidades:
#   - Resolver la ruta global/misión (eso pertenece al mission planner /
#     route handler).
#   - Seguir la trayectoria (eso pertenece al MPC/controller).
#   - Subscribirse a topics o publicar mensajes (vive en `planner_thread.py`).
#
# Esta separación garantiza:
#   - un pipeline casi 1:1 con Autoware:
#       scene modules -> behavior path -> path optimizer -> velocity planner -> MPC
#   - desacoplar el planner de la representación concreta del mapa
#   - mantener el contrato actual del controller (`BehaviorOutput`) mientras
#     migramos el resto del stack.

from __future__ import annotations

import logging

from src.behavior.context import PlanningContext
from src.behavior.path_optimizer import PathOptimizer
from src.behavior.planner_manager import PlannerManager
from src.behavior.velocity_planner import BehaviorVelocityPlanner
from src.core.interfaces.planner import IScenario
from src.core.types.behavior import BehaviorOutput, ScenarioName

_log = logging.getLogger(__name__)


class BehaviorPlanner:
    """Pipeline de planning pre-MPC con manager/path/velocity separados."""

    def __init__(self, scenarios: list[IScenario] | None = None) -> None:
        self._manager = PlannerManager(scenarios)
        self._path_optimizer = PathOptimizer()
        self._velocity_planner = BehaviorVelocityPlanner()

    # --------------------------------------------------------------
    # Registro de scenarios
    # --------------------------------------------------------------
    def add_scenario(self, scenario: IScenario) -> None:
        self._manager.add_scenario(scenario)

    @property
    def scenarios(self) -> tuple[IScenario, ...]:
        return self._manager.scenarios

    # --------------------------------------------------------------
    # Plan
    # --------------------------------------------------------------
    def plan(self, ctx: PlanningContext) -> BehaviorOutput:
        """Pipeline pre-MPC: scene selection -> path optimization -> velocity."""
        path_plan = self._manager.plan_path(ctx)
        if not bool(path_plan.valid):
            return _empty_fallback(
                ctx,
                reason=str(path_plan.notes.get("reason", "invalid_path_plan")),
            )

        optimized = self._path_optimizer.optimize(path_plan, ctx)
        return self._velocity_planner.build_output(
            path_plan=path_plan,
            target_path=optimized.target_path,
            drivable_left_bound=optimized.drivable_left_bound,
            drivable_right_bound=optimized.drivable_right_bound,
            ctx=ctx,
        )


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
