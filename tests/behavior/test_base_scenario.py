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


def _straight_visual_waypoints(n: int = 20, density: float = 0.05):
    return tuple((density * i, 0.0, 0.0) for i in range(n))


def _lookahead_visual_waypoints(n: int = 20, start_m: float = 0.20, density: float = 0.05):
    return tuple((start_m + (density * i), 0.0, 0.0) for i in range(n))


def test_constant_speed_plan_keeps_route_waypoints_when_alignment_degrades_but_route_is_active(
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
    assert plan.notes["path_source"] == "route_waypoints"
    assert plan.notes["recovery_source"] == "route_waypoints_reentry"
    assert "route_alignment_fallback" not in plan.notes
    assert plan.target_path.shape[1] == 3
    assert plan.target_path[0, 1] == pytest.approx(0.15, abs=1e-6)
    assert plan.target_path[1, 1] < plan.target_path[0, 1]


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


def test_constant_speed_plan_keeps_single_line_visual_primary_path() -> None:
    scenario = _ConstantSpeedScenario()
    pose_x = 0.02
    pose_y = 0.05
    ctx = replace(
        make_context(
            pose_x=pose_x,
            pose_y=pose_y,
            pose_yaw=0.0,
            current_lanelet_id="n1->n2",
            route_waypoints=[
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [1.2, 0.0, 0.0],
            ],
            matched_pose=(0.0, 0.0, 0.0),
            map_match_error_m=0.35,
        ),
        lane_observation=LaneObservation(
            detected_sides=("left",),
            quality=0.85,
            measurement_mode="single_line",
            direct_error_valid=True,
            direct_error_m=0.05,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
            center_waypoints_body=_lookahead_visual_waypoints(),
            extrapolated_side="right",
            lane_width_m=0.35,
        ),
    )

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["visual_path_primary_reason"] == "single_line_primary"
    assert plan.notes["visual_path_connected_from_ego_pose"] is False
    assert plan.target_path[0, 0] > pose_x + 0.15
    assert plan.target_path[0, 1] == pytest.approx(pose_y, abs=1e-6)


def test_constant_speed_plan_keeps_two_line_visual_primary_path() -> None:
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
            map_match_error_m=0.20,
        ),
        lane_observation=LaneObservation(
            detected_sides=("left", "right"),
            quality=1.0,
            measurement_mode="two_line",
            direct_error_valid=True,
            direct_error_m=0.10,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
            center_waypoints_body=_straight_visual_waypoints(),
            lane_width_m=0.35,
        ),
    )

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["visual_path_primary_reason"] == "two_line_primary"
