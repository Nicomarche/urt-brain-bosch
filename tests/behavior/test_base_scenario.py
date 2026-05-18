from __future__ import annotations

from dataclasses import replace

import pytest

from src.behavior.scenarios.base import BaseScenario
from src.core.types.behavior import BehaviorOutput, ScenarioName
from src.core.types.perception import LaneObservation
from src.routing.lanelet.attributes import ATTR_INTERSECTION
from src.routing.lanelet.lanelet_map import from_track_graph
from tests.behavior.conftest import _build_track_graph, make_context


class _ConstantSpeedScenario(BaseScenario):
    name = ScenarioName.LANE_KEEP.value
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


def _transverse_visual_waypoints(n: int = 20, x_m: float = 0.20, density: float = 0.05):
    return tuple((x_m, density * i, 0.0) for i in range(n))


def _plan_after_visual_hysteresis(scenario: _ConstantSpeedScenario, ctx):
    plan = scenario.plan(ctx)
    plan = scenario.plan(ctx)
    return scenario.plan(ctx)


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


def test_constant_speed_plan_uses_single_line_visual_primary_on_normal_route() -> None:
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

    plan = _plan_after_visual_hysteresis(scenario, ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["path_authority"] == "visual"
    assert plan.notes["mpc_weight_profile"] == "lane_keep_visual"
    assert plan.notes["steer_rate_limit_deg_s"] == pytest.approx(180.0)
    assert plan.notes["visual_path_primary_reason"] == "single_line_primary"
    assert plan.notes["visual_path_connected_from_ego_pose"] is False


def test_constant_speed_plan_uses_two_line_visual_primary_on_normal_route() -> None:
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

    plan = _plan_after_visual_hysteresis(scenario, ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["path_authority"] == "visual"
    assert plan.notes["visual_path_primary_reason"] == "two_line_primary"


def test_constant_speed_plan_keeps_visual_primary_path_without_route_corridor() -> None:
    scenario = _ConstantSpeedScenario()
    pose_x = 0.02
    pose_y = 0.05
    ctx = replace(
        make_context(
            pose_x=pose_x,
            pose_y=pose_y,
            pose_yaw=0.0,
            current_lanelet_id="n1->n2",
            route_waypoints=[],
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

    plan = _plan_after_visual_hysteresis(scenario, ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["visual_path_primary_reason"] == "single_line_primary"
    assert plan.notes["visual_path_connected_from_ego_pose"] is False


def test_constant_speed_plan_keeps_route_primary_when_visual_quality_is_low() -> None:
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
        ),
        lane_observation=LaneObservation(
            detected_sides=("left", "right"),
            quality=0.50,
            measurement_mode="two_line",
            direct_error_valid=True,
            direct_error_m=0.05,
            center_waypoints_body=_straight_visual_waypoints(),
            lane_width_m=0.35,
        ),
    )

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "route_waypoints"
    assert plan.notes["path_authority"] == "route"
    assert plan.notes["visual_path_primary_rejected_reason"] == "low_quality:two_line"


def test_constant_speed_plan_keeps_map_primary_near_intersection_semantic() -> None:
    scenario = _ConstantSpeedScenario()
    base_ctx = make_context(
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
    )
    ctx = replace(
        base_ctx,
        route=replace(
            base_ctx.route,
            next_semantic_type="intersection",
            next_semantic_distance_m=0.40,
        ),
        lane_observation=LaneObservation(
            detected_sides=("left", "right"),
            quality=1.0,
            measurement_mode="two_line",
            direct_error_valid=True,
            direct_error_m=0.05,
            center_waypoints_body=_straight_visual_waypoints(),
            lane_width_m=0.35,
        ),
    )

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "route_waypoints"
    assert plan.notes["path_authority"] == "map"
    assert plan.notes["mpc_weight_profile"] == "map_turn_authority"
    assert plan.notes["steer_rate_limit_deg_s"] == pytest.approx(160.0)
    assert plan.notes["visual_path_primary_rejected_reason"].startswith("map_authority:")


def test_constant_speed_plan_keeps_map_primary_on_turn_direction_attr() -> None:
    scenario = _ConstantSpeedScenario()
    base_ctx = make_context(
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
    )
    ctx = replace(
        base_ctx,
        route=replace(base_ctx.route, current_node_attr=ATTR_INTERSECTION),
        lane_observation=LaneObservation(
            detected_sides=("left", "right"),
            quality=1.0,
            measurement_mode="two_line",
            direct_error_valid=True,
            direct_error_m=0.05,
            center_waypoints_body=_straight_visual_waypoints(),
            lane_width_m=0.35,
        ),
    )

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "route_waypoints"
    assert plan.notes["path_authority"] == "map"
    assert "current_node_attr" in plan.notes["visual_path_primary_rejected_reason"]


def test_constant_speed_plan_rejects_transverse_visual_path() -> None:
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
        ),
        lane_observation=LaneObservation(
            detected_sides=("left", "right"),
            quality=1.0,
            measurement_mode="two_line",
            direct_error_valid=True,
            direct_error_m=0.05,
            center_waypoints_body=_transverse_visual_waypoints(),
            lane_width_m=0.35,
        ),
    )

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "route_waypoints"
    assert plan.notes["visual_path_primary_rejected_reason"] == "visual_path_insufficient_forward_span"


def test_constant_speed_plan_uses_hysteresis_for_soft_visual_rejections() -> None:
    scenario = _ConstantSpeedScenario()
    good_ctx = replace(
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
        ),
        lane_observation=LaneObservation(
            detected_sides=("left", "right"),
            quality=1.0,
            measurement_mode="two_line",
            direct_error_valid=True,
            direct_error_m=0.05,
            center_waypoints_body=_straight_visual_waypoints(),
            lane_width_m=0.35,
        ),
    )
    low_quality_ctx = replace(
        good_ctx,
        lane_observation=replace(good_ctx.lane_observation, quality=0.50),
    )

    assert scenario.plan(good_ctx).notes["path_source"] == "route_waypoints"
    assert scenario.plan(good_ctx).notes["path_source"] == "route_waypoints"
    assert scenario.plan(good_ctx).notes["path_source"] == "visual_lane_waypoints"
    assert scenario.plan(low_quality_ctx).notes["path_source"] == "visual_lane_waypoints"
    plan = scenario.plan(low_quality_ctx)

    assert plan.notes["path_source"] == "route_waypoints"
    assert plan.notes["visual_path_primary_rejected_reason"] == "low_quality:two_line"
