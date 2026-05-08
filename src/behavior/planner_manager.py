from __future__ import annotations

import logging
import math

import numpy as np

from src.behavior.context import PlanningContext
from src.core.interfaces.planner import IScenario
from src.core.types.behavior import (
    BehaviorOutput,
    BehaviorPathPlan,
    ScenarioName,
    TurnSignalCommand,
)

_log = logging.getLogger(__name__)


class PlannerManager:
    """Autoware-inspired manager de scene modules.

    Responsabilidades:
      - mantener el registro ordenado por prioridad de los módulos;
      - preguntar qué módulos están activos en el tick actual;
      - aprobar el módulo ganador y adaptar su salida al contrato interno
        `BehaviorPathPlan`.

    En Autoware, el manager también resuelve concurrencia entre módulos y
    mergea múltiples paths. En esta fase mantenemos el modelo
    "single-approved-module" para no romper el controller, pero dejamos toda
    la metadata de transición para poder evolucionarlo luego.
    """

    def __init__(self, scenarios: list[IScenario] | None = None) -> None:
        self._scenarios: list[IScenario] = []
        if scenarios:
            for scenario in scenarios:
                self.add_scenario(scenario)

    def add_scenario(self, scenario: IScenario) -> None:
        if scenario in self._scenarios:
            return
        self._scenarios.append(scenario)
        self._scenarios.sort(key=lambda s: -getattr(s, "priority", 0))

    @property
    def scenarios(self) -> tuple[IScenario, ...]:
        return tuple(self._scenarios)

    def plan_path(self, ctx: PlanningContext) -> BehaviorPathPlan:
        active: list[IScenario] = []
        for scenario in self._scenarios:
            try:
                if bool(scenario.is_active(ctx)):
                    active.append(scenario)
            except Exception as exc:
                _log.exception(
                    "scene module %s.is_active() lanzó excepción: %s",
                    getattr(scenario, "name", "?"),
                    exc,
                )

        manager_notes = {
            "registered_modules": [getattr(s, "name", "?") for s in self._scenarios],
            "candidate_modules": [getattr(s, "name", "?") for s in active],
            "candidate_priorities": {
                str(getattr(s, "name", "?")): int(getattr(s, "priority", 0))
                for s in active
            },
            "manager_mode": "single_approved_module",
        }

        if not active:
            return self._fallback_plan(
                ctx,
                reason="no_scene_module_active",
                manager_notes=manager_notes,
            )

        approved = active[0]
        manager_notes["approved_module"] = getattr(approved, "name", "?")

        try:
            raw_plan = self._invoke_scene_module(approved, ctx)
        except Exception as exc:
            _log.exception(
                "scene module %s.plan() lanzó excepción: %s",
                getattr(approved, "name", "?"),
                exc,
            )
            return self._fallback_plan(
                ctx,
                reason=f"scenario_crash:{getattr(approved, 'name', '?')}",
                manager_notes=manager_notes,
            )

        if not isinstance(raw_plan, BehaviorPathPlan):
            return self._fallback_plan(
                ctx,
                reason="bad_scene_module_output_type",
                manager_notes=manager_notes,
            )

        notes = dict(raw_plan.notes)
        notes.update(manager_notes)
        if "turn_signal" not in notes:
            notes["turn_signal"] = str(raw_plan.turn_signal)
        return BehaviorPathPlan(
            timestamp=float(raw_plan.timestamp or ctx.now_s),
            raw_path=np.asarray(raw_plan.raw_path, dtype=float),
            base_speed_profile=np.asarray(raw_plan.base_speed_profile, dtype=float),
            scenario_name=str(raw_plan.scenario_name or ScenarioName.FALLBACK.value),
            valid=bool(raw_plan.valid),
            stop_required=bool(raw_plan.stop_required),
            turn_signal=str(raw_plan.turn_signal),
            drivable_left_bound=np.asarray(raw_plan.drivable_left_bound, dtype=float),
            drivable_right_bound=np.asarray(raw_plan.drivable_right_bound, dtype=float),
            notes=notes,
        )

    def _invoke_scene_module(
        self,
        scenario: IScenario,
        ctx: PlanningContext,
    ) -> BehaviorPathPlan:
        plan_path_fn = getattr(scenario, "plan_path", None)
        if callable(plan_path_fn):
            out = plan_path_fn(ctx)
            if isinstance(out, BehaviorPathPlan):
                return out
            raise TypeError(
                f"{getattr(scenario, 'name', '?')}.plan_path devolvió {type(out).__name__}"
            )

        out = scenario.plan(ctx)
        if not isinstance(out, BehaviorOutput):
            raise TypeError(
                f"{getattr(scenario, 'name', '?')}.plan devolvió {type(out).__name__}"
            )
        return self._adapt_behavior_output(ctx, out)

    def _adapt_behavior_output(
        self,
        ctx: PlanningContext,
        plan: BehaviorOutput,
    ) -> BehaviorPathPlan:
        raw_path = np.asarray(plan.target_path, dtype=float)
        if raw_path.ndim != 2 or raw_path.shape[1] < 3 or raw_path.shape[0] == 0:
            raw_path = np.array(
                [[ctx.pose.fused_pose.x, ctx.pose.fused_pose.y, ctx.pose.fused_pose.yaw]],
                dtype=float,
            )

        speed = np.asarray(plan.speed_profile, dtype=float)
        if speed.ndim != 1:
            speed = speed.reshape(-1)

        notes = dict(plan.notes)
        turn_signal = str(notes.get("turn_signal") or self._infer_turn_signal(ctx, plan.scenario_name))

        return BehaviorPathPlan(
            timestamp=float(plan.timestamp or ctx.now_s),
            raw_path=raw_path,
            base_speed_profile=speed,
            scenario_name=str(plan.scenario_name),
            valid=bool(plan.valid),
            stop_required=bool(plan.stop_required),
            turn_signal=turn_signal,
            notes=notes,
        )

    def _fallback_plan(
        self,
        ctx: PlanningContext,
        *,
        reason: str,
        manager_notes: dict,
    ) -> BehaviorPathPlan:
        return BehaviorPathPlan(
            timestamp=float(ctx.now_s),
            raw_path=np.array(
                [[ctx.pose.fused_pose.x, ctx.pose.fused_pose.y, ctx.pose.fused_pose.yaw]],
                dtype=float,
            ),
            base_speed_profile=np.zeros(0, dtype=float),
            scenario_name=ScenarioName.FALLBACK.value,
            valid=False,
            stop_required=True,
            turn_signal=TurnSignalCommand.HAZARD.value,
            notes={"reason": reason, **manager_notes},
        )

    @staticmethod
    def _infer_turn_signal(ctx: PlanningContext, scenario_name: str) -> str:
        if scenario_name == ScenarioName.PARKING.value:
            return TurnSignalCommand.HAZARD.value
        heading_change = float(getattr(ctx.route, "path_heading_change_rad", 0.0) or 0.0)
        if abs(heading_change) < math.radians(20.0):
            return TurnSignalCommand.NONE.value
        return (
            TurnSignalCommand.LEFT.value
            if heading_change > 0.0
            else TurnSignalCommand.RIGHT.value
        )
