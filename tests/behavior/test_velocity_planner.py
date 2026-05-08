from __future__ import annotations

import numpy as np
import pytest

from src.behavior.velocity_planner import BehaviorVelocityPlanner
from src.core.types.behavior import BehaviorPathPlan, ScenarioName
from src.core.types.routing import RegulatoryElement
from tests.behavior.conftest import make_context


def test_velocity_planner_stops_for_stopline() -> None:
    reg = RegulatoryElement(
        element_id="stop-1",
        kind="stopline",
        position_xy=(0.0, 0.0),
        data={"distance_m": 1.0},
    )
    ctx = make_context(
        regulatory_ahead=(reg,),
        horizon_n=20,
        dt=0.1,
        nominal_speed_mps=1.0,
        max_speed_mps=2.0,
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=np.column_stack([np.linspace(0.0, 4.0, 21), np.zeros(21), np.zeros(21)]),
        base_speed_profile=np.full(ctx.horizon_n, 1.0, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )
    target_path = np.column_stack([np.linspace(0.0, 4.0, 21), np.zeros(21), np.zeros(21)])

    out = BehaviorVelocityPlanner().build_output(
        path_plan=plan,
        target_path=target_path,
        drivable_left_bound=np.zeros((21, 2), dtype=float),
        drivable_right_bound=np.zeros((21, 2), dtype=float),
        ctx=ctx,
    )

    assert out.stop_required is True
    assert out.speed_profile[-1] == pytest.approx(0.0, abs=1e-6)


def test_velocity_planner_caps_speed_on_curvature() -> None:
    theta = np.linspace(0.0, np.pi / 2.0, 21)
    radius = 0.30
    target_path = np.column_stack(
        [
            radius * np.cos(theta),
            radius * np.sin(theta),
            theta + np.pi / 2.0,
        ]
    )
    ctx = make_context(horizon_n=20, dt=0.1, nominal_speed_mps=1.5, max_speed_mps=2.0)
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=target_path,
        base_speed_profile=np.full(ctx.horizon_n, 1.5, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )

    out = BehaviorVelocityPlanner().build_output(
        path_plan=plan,
        target_path=target_path,
        drivable_left_bound=np.zeros((21, 2), dtype=float),
        drivable_right_bound=np.zeros((21, 2), dtype=float),
        ctx=ctx,
    )

    assert np.min(out.speed_profile) < 1.5
    assert any(note.get("kind") == "curvature_constraint" for note in out.notes.get("velocity_modules", []))
