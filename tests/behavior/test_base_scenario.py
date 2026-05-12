from __future__ import annotations

import math
from dataclasses import replace

import pytest

from src.behavior.scenarios.base import BaseScenario
from src.core.types.behavior import BehaviorOutput, ScenarioName
from src.core.types.perception import LaneObservation
from src.core.types.pose import VisualLaneMatch
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


def _with_visual_match(
    ctx,
    *,
    yaw_deg: float = 0.0,
    accepted: bool = True,
    lateral_error_m: float = 0.0,
    reason: str | None = None,
):
    yaw_rad = math.radians(float(yaw_deg))
    match_reason = reason or ("accepted" if accepted else "median_yaw_error")
    return replace(
        ctx,
        pose=replace(
            ctx.pose,
            visual_lane_match=VisualLaneMatch(
                lanelet_id=ctx.route.current_lanelet_id,
                lateral_error_m=float(lateral_error_m),
                yaw_error_rad=yaw_rad,
                near_lateral_error_m=float(lateral_error_m),
                near_yaw_error_rad=yaw_rad,
                max_abs_lateral_error_m=abs(float(lateral_error_m)),
                max_abs_yaw_error_rad=abs(yaw_rad),
                sample_count=4,
                score=0.0,
                confidence=0.95,
                accepted=bool(accepted),
                reason=match_reason,
            ),
        ),
    )


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
    ctx = _with_visual_match(ctx)

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
    ctx = _with_visual_match(ctx)

    plan = scenario.plan(ctx)

    assert plan.stop_required is False
    assert "visual_lane_drift_guard" not in plan.notes
    assert plan.speed_profile[0] == pytest.approx(0.4, abs=1e-6)


def test_constant_speed_plan_prefers_reliable_single_line_visual_path_over_active_route() -> None:
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
            measurement_mode_streak_frames=8,
            direct_error_valid=True,
            direct_error_m=0.05,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
            center_waypoints_body=_lookahead_visual_waypoints(),
            extrapolated_side="right",
            lane_width_m=0.35,
        ),
    )
    ctx = _with_visual_match(ctx)

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["visual_path_primary_reason"] == "single_line_visual_primary_over_route"
    assert plan.notes["visual_path_connected_from_ego_pose"] is False
    # Opción C: path en frame body — primer waypoint con x>=0, y~0 (no
    # `pose_y`). El motion_controller también arma `x_current=(0,0,0)`
    # cuando ve `visual_path_frame_body=True`.
    # Path en frame world por default (Opción C deshabilitada). El flag
    # `visual_path_frame_body` queda en notes para auditoría.
    assert "visual_path_frame_body" in plan.notes
    assert plan.target_path[0, 0] >= 0.0


def test_constant_speed_plan_takes_fresh_single_line_as_primary_path() -> None:
    # Estilo urt-ref: el streak frames y `transition_error_coherent` ya no son
    # gates. Una transición recién entrada (streak=2) con calidad ≥0.6 entra
    # como primary path sin necesidad de quedarse en route hasta confirmar.
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
            detected_sides=("left",),
            quality=0.85,
            measurement_mode="single_line",
            measurement_mode_streak_frames=2,
            previous_measurement_mode="two_line",
            direct_error_valid=True,
            direct_error_m=0.05,
            transition_error_coherent=True,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
            center_waypoints_body=_lookahead_visual_waypoints(),
            extrapolated_side="right",
            lane_width_m=0.35,
        ),
    )
    ctx = _with_visual_match(ctx)

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["visual_path_primary_reason"] == "single_line_visual_primary_over_route"


def test_constant_speed_plan_accepts_single_line_even_when_legacy_transition_flag_is_false() -> None:
    # El gate `transition_error_coherent` se eliminó: la curvatura del path
    # visual la pone el polinomio del lado visible, no la coherencia entre
    # frames con el último two_line.
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
            detected_sides=("left",),
            quality=0.85,
            measurement_mode="single_line",
            measurement_mode_streak_frames=8,
            direct_error_valid=True,
            direct_error_m=-0.18,
            transition_error_coherent=False,
            transition_reference_error_m=-0.04,
            transition_error_delta_m=0.14,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
            center_waypoints_body=_lookahead_visual_waypoints(),
            extrapolated_side="right",
            lane_width_m=0.35,
        ),
    )
    ctx = _with_visual_match(ctx)

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["visual_path_primary_reason"] == "single_line_visual_primary_over_route"


def test_constant_speed_plan_promotes_single_line_visual_assist_to_primary_path() -> None:
    # `assist_primary` con threshold 0.82 ya no existe — basta con quality ≥ 0.6.
    # Visual gana incluso sin direct_error_valid mientras los waypoints sean válidos.
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
            map_match_error_m=0.20,
        ),
        lane_observation=LaneObservation(
            detected_sides=("left",),
            quality=0.85,
            measurement_mode="single_line",
            measurement_mode_streak_frames=8,
            direct_error_valid=False,
            control_policy_mode="VISUAL_ASSIST",
            planner_priority_active=False,
            center_waypoints_body=_lookahead_visual_waypoints(),
            extrapolated_side="right",
            lane_width_m=0.35,
        ),
    )
    ctx = _with_visual_match(ctx)

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["visual_path_primary_reason"] == "single_line_visual_primary_over_route"
    # Path en frame world por default (Opción C deshabilitada).
    assert "visual_path_frame_body" in plan.notes


def test_constant_speed_plan_rejects_single_line_visual_path_below_min_quality() -> None:
    # Quality 0.55 (debajo del mínimo 0.6 estilo ref) sí rechaza visual y
    # cae a route. Es el único gate que queda en single_line.
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
            detected_sides=("left",),
            quality=0.55,
            measurement_mode="single_line",
            measurement_mode_streak_frames=8,
            direct_error_valid=False,
            control_policy_mode="VISUAL_ASSIST",
            planner_priority_active=False,
            center_waypoints_body=_lookahead_visual_waypoints(),
            extrapolated_side="right",
            lane_width_m=0.35,
        ),
    )
    ctx = _with_visual_match(ctx)

    plan = scenario.plan(ctx)

    assert plan.valid is True
    # quality 0.55 < `lane_observation_has_visual_path` min 0.55 → primero
    # rechaza el gate de waypoints; con 0.55 estricto puede pasar y caer al
    # gate de single_line min 0.6 → mismo resultado: route.
    assert plan.notes["path_source"] == "route_waypoints"


def test_constant_speed_plan_uses_single_line_visual_path_when_waypoints_are_reliable() -> None:
    # Single-line con direct_error_valid=False pero waypoints sólidos: visual
    # gana. Antes había un reason específico `_path_only`; ahora es el reason
    # único `single_line_visual_primary_over_route`.
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
            detected_sides=("left",),
            quality=0.85,
            measurement_mode="single_line",
            measurement_mode_streak_frames=8,
            direct_error_valid=False,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
            center_waypoints_body=_lookahead_visual_waypoints(),
            extrapolated_side="right",
            lane_width_m=0.35,
        ),
    )
    ctx = _with_visual_match(ctx)

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["visual_path_primary_reason"] == "single_line_visual_primary_over_route"


def test_constant_speed_plan_keeps_single_line_visual_when_map_match_rejects() -> None:
    # `_visual_map_match_primary_gate` se eliminó del single_line: el path
    # visual gana aunque el matcher contra el mapa diga "lateral mismatch".
    # El mapa ya no es prerequisito para el modo visual primario.
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
            detected_sides=("right",),
            quality=0.85,
            measurement_mode="single_line",
            measurement_mode_streak_frames=8,
            direct_error_valid=True,
            direct_error_m=0.03,
            transition_error_coherent=True,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
            center_waypoints_body=_lookahead_visual_waypoints(),
            extrapolated_side="left",
            lane_width_m=0.35,
        ),
    )
    ctx = _with_visual_match(
        ctx,
        accepted=False,
        lateral_error_m=0.42,
        reason="median_lateral_error",
    )

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["visual_path_primary_reason"] == "single_line_visual_primary_over_route"


def test_constant_speed_plan_prefers_reliable_two_line_visual_path_over_active_route() -> None:
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
    ctx = _with_visual_match(ctx)

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["visual_path_primary_reason"] == "two_line_visual_primary_over_route"
    assert plan.notes["visual_path_connected_from_ego_pose"] is False


def test_constant_speed_plan_keeps_sparse_two_line_visual_path_over_active_route() -> None:
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
            detected_sides=("left", "right"),
            quality=1.0,
            measurement_mode="two_line",
            direct_error_valid=True,
            direct_error_m=-0.01,
            control_policy_mode="VISUAL_ASSIST",
            planner_priority_active=True,
            center_waypoints_body=_straight_visual_waypoints(n=4),
            lane_width_m=0.35,
        ),
    )
    ctx = _with_visual_match(ctx)

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["visual_path_primary_reason"] == "two_line_visual_primary_over_route"
    assert plan.notes["visual_lane_waypoint_count"] == 4


def test_constant_speed_plan_allows_visual_primary_during_waypoint_mode_hint() -> None:
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
        map_match_error_m=0.20,
    )
    ctx = replace(
        base_ctx,
        route=replace(
            base_ctx.route,
            waypoint_mode_active=True,
            next_semantic_type="intersection",
            next_semantic_distance_m=0.15,
        ),
        lane_observation=LaneObservation(
            detected_sides=("left", "right"),
            quality=1.0,
            measurement_mode="two_line",
            direct_error_valid=True,
            direct_error_m=0.04,
            center_waypoints_body=_straight_visual_waypoints(),
            lane_width_m=0.35,
        ),
    )
    ctx = _with_visual_match(ctx)

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["visual_path_primary_reason"] == "two_line_visual_primary_over_route"


def test_constant_speed_plan_allows_visual_primary_in_intersection_scenario_when_not_precision_gated() -> None:
    scenario = _ConstantSpeedScenario()
    scenario_name = ScenarioName.INTERSECTION.value
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
            direct_error_m=0.04,
            center_waypoints_body=_straight_visual_waypoints(),
            lane_width_m=0.35,
        ),
    )
    ctx = _with_visual_match(ctx)

    plan = scenario._build_constant_speed_plan(
        ctx,
        target_speed_mps=0.4,
        scenario_name=scenario_name,
    )

    assert plan.valid is True
    assert plan.scenario_name == scenario_name
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["visual_path_primary_reason"] == "two_line_visual_primary_over_route"


def test_constant_speed_plan_keeps_two_line_visual_path_when_map_yaw_disagrees() -> None:
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
            line_center_offset_m=0.10,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
            center_waypoints_body=_straight_visual_waypoints(),
            lane_width_m=0.35,
        ),
    )
    ctx = _with_visual_match(ctx, yaw_deg=8.5, accepted=False)

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["visual_path_primary_reason"] == "two_line_visual_primary_over_route"


def test_constant_speed_plan_keeps_two_line_visual_path_on_large_local_yaw_disagreement() -> None:
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
            line_center_offset_m=0.10,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
            center_waypoints_body=_straight_visual_waypoints(),
            lane_width_m=0.35,
        ),
    )
    ctx = _with_visual_match(ctx, yaw_deg=18.0, accepted=False)

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["visual_path_primary_reason"] == "two_line_visual_primary_over_route"


def test_constant_speed_plan_keeps_route_primary_in_precision_context() -> None:
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
        map_match_error_m=0.20,
    )
    ctx = replace(
        base_ctx,
        route=replace(base_ctx.route, expected_control_type="intersection"),
        lane_observation=LaneObservation(
            detected_sides=("left", "right"),
            quality=1.0,
            measurement_mode="two_line",
            direct_error_valid=True,
            direct_error_m=0.10,
            center_waypoints_body=_straight_visual_waypoints(),
            lane_width_m=0.35,
        ),
    )

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "route_waypoints"
    assert plan.notes["visual_path_primary_rejected_reason"] == "route_precision_context:intersection"


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

    plan = scenario.plan(ctx)

    assert plan.valid is True
    assert plan.notes["path_source"] == "visual_lane_waypoints"
    assert plan.notes["visual_path_primary_reason"] == "single_line_primary"
    assert plan.notes["visual_path_connected_from_ego_pose"] is False
