from __future__ import annotations

import numpy as np

from src.behavior.planner_manager import PlannerManager
from src.behavior.scenarios.base import BaseScenario
from src.core.types.behavior import BehaviorOutput
from tests.behavior.conftest import make_context


class _FakeScenario(BaseScenario):
    def __init__(self, name: str, priority: int, active: bool, speed: float = 0.5) -> None:
        self.name = name
        self.priority = priority
        self._active = active
        self._speed = speed

    def is_active(self, ctx) -> bool:
        return self._active

    def plan(self, ctx) -> BehaviorOutput:
        n = ctx.horizon_n
        return BehaviorOutput(
            timestamp=ctx.now_s,
            dt=ctx.dt,
            target_path=np.zeros((n + 1, 3), dtype=float),
            speed_profile=np.full(n, self._speed, dtype=float),
            scenario_name=self.name,
            valid=True,
        )


def test_manager_selects_highest_priority_scene_module() -> None:
    low = _FakeScenario("low", priority=1, active=True, speed=0.2)
    high = _FakeScenario("high", priority=10, active=True, speed=0.9)
    mgr = PlannerManager([low, high])

    out = mgr.plan_path(make_context(horizon_n=6))

    assert out.scenario_name == "high"
    assert out.base_speed_profile[0] == 0.9
    assert out.notes.get("approved_module") == "high"
    assert out.notes.get("candidate_modules") == ["high", "low"]
