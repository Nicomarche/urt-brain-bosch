from __future__ import annotations

from dataclasses import replace

import pytest

from src.behavior.scenarios.base import BaseScenario
from src.core.types.behavior import BehaviorOutput
from src.core.types.perception import LaneObservation
from src.routing.lanelet.lanelet_map import from_track_graph
from tests.behavior.conftest import _build_track_graph, make_context


class _ConstantSpeedScenario(BaseScenario):
    name = "test_constant_speed"
    priority = 0

    def is_active(self, ctx) -> bool:
        return True

    def plan(self, ctx) -> BehaviorOutput:
        return self._build_constant_speed_plan(
            ctx,
            target_speed_mps=0.4,
            scenario_name=self.name,
        )


def test_constant_speed_plan_falls_back_to_lanelet_centerline_when_route_alignment_degrades(
) -> None:
    scenario = _ConstantSpeedScenario()
    route_waypoints = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    lanelet_map = from_track_graph(
        _build_track_graph(
            [("n1", 0.0, 0.0, 0), ("n2", 1.0, 0.0, 0), ("n3", 2.0, 0.0, 0)]
        ),
        step_m=0.20,
    )
    ctx = make_context(
        pose_x=0.1,
        pose_y=0.15,
        pose_yaw=0.0,
        current_lanelet_id="n1->n2",
        next_lanelet_ids=("n2->n3",),
        lanelet_map=lanelet_map,
        route_waypoints=route_waypoints,
        matched_pose=(0.0, 0.0, 0.0),
        map_match_error_m=0.25,
    )

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "lanelet_centerline"
    assert plan.notes["route_alignment_fallback"] is True
    assert plan.notes["route_alignment_error_m"] == 0.25
    assert plan.target_path.shape[1] == 3
    assert plan.target_path[0, 1] == pytest.approx(0.0, abs=1e-6)


def test_constant_speed_plan_does_not_cap_or_stop_for_visual_lane_drift_inputs() -> None:
    scenario = _ConstantSpeedScenario()
    ctx = replace(
        make_context(
            pose_x=0.02,
            pose_y=0.05,
            pose_yaw=0.0,
            current_lanelet_id="n1->n2",
            route_waypoints=[
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [1.2, 0.0, 0.0],
            ],
            matched_pose=(0.0, 0.0, 0.0),
            map_match_error_m=0.05,
        ),
        lane_observation=LaneObservation(
            detected_sides=("left",),
            quality=0.65,
            measurement_mode="single_line",
            direct_error_valid=False,
            control_policy_mode="ROUTE_TRACKING",
        ),
    )

    plan = scenario.plan(ctx)

    assert plan.stop_required is False
    assert "visual_lane_drift_guard" not in plan.notes
    assert plan.speed_profile[0] == pytest.approx(0.4, abs=1e-6)
