from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.behavior.path_optimizer as path_optimizer_module
from src.behavior.path_optimizer import PathOptimizer
from src.behavior.trajectory_builder import (
    build_target_path,
    build_target_path_from_route,
    build_target_path_from_visual,
)
from src.core.types.behavior import BehaviorPathPlan, ScenarioName
from src.core.types.perception import LaneObservation
from src.routing.lanelet.from_osm import load_lanelet2_osm
from src.routing.lanelet.lanelet_map import Lanelet, from_track_graph
from src.routing.lanelet.osm_router import OsmRouteGraph
from tests.behavior.conftest import _build_track_graph, make_context


def _heading_tangent_error_deg(path: np.ndarray, idx: int) -> float:
    lo = max(0, int(idx) - 1)
    hi = min(path.shape[0] - 1, int(idx) + 1)
    if lo == hi:
        return 0.0
    dx = float(path[hi, 0] - path[lo, 0])
    dy = float(path[hi, 1] - path[lo, 1])
    tangent = math.atan2(dy, dx)
    delta = math.atan2(
        math.sin(float(path[idx, 2]) - tangent),
        math.cos(float(path[idx, 2]) - tangent),
    )
    return abs(math.degrees(delta))


def _sample1_heading_delta_from_pose_deg(path: np.ndarray, pose_yaw: float) -> float:
    dx = float(path[1, 0] - path[0, 0])
    dy = float(path[1, 1] - path[0, 1])
    heading = math.atan2(dy, dx)
    delta = math.atan2(
        math.sin(heading - float(pose_yaw)),
        math.cos(heading - float(pose_yaw)),
    )
    return math.degrees(delta)


def _require_non_empty_route(route, *, lanelet_id: str) -> None:
    if getattr(route, "waypoints", np.empty((0, 3))).shape[0] > 0:
        return
    pytest.skip(f"route to lanelet {lanelet_id} is unavailable in the current sim map")


def _require_non_degenerate_path(path: np.ndarray, *, label: str) -> None:
    if path.shape[0] >= 2 and np.any(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1) > 1e-6):
        return
    pytest.skip(f"{label} is degenerate in the current sim map")


def test_path_optimizer_emits_mpc_sized_reference() -> None:
    ctx = make_context(pose_x=1.0, pose_y=2.0, pose_yaw=math.radians(15.0), horizon_n=12, dt=0.1)
    raw_path = np.array(
        [
            [1.0, 2.0, 0.0],
            [2.0, 2.2, 0.0],
            [3.0, 2.5, 0.0],
            [4.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=raw_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.8, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )

    out = PathOptimizer().optimize(plan, ctx)

    assert out.target_path.shape == (ctx.horizon_n + 1, 3)
    np.testing.assert_allclose(out.target_path[0, :2], [ctx.pose.fused_pose.x, ctx.pose.fused_pose.y], atol=1e-6)


def test_path_optimizer_builds_drivable_bounds() -> None:
    ctx = make_context(horizon_n=8, dt=0.1)
    raw_path = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=raw_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.5, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )

    out = PathOptimizer().optimize(plan, ctx)

    assert out.drivable_left_bound.shape == (ctx.horizon_n + 1, 2)
    assert out.drivable_right_bound.shape == (ctx.horizon_n + 1, 2)
    lateral_span = np.linalg.norm(out.drivable_left_bound[0] - out.drivable_right_bound[0])
    assert lateral_span > 0.25


def test_path_optimizer_visual_primary_uses_lighter_smoothing() -> None:
    raw = np.array(
        [
            [0.00, 0.00],
            [0.10, 0.00],
            [0.20, 0.18],
            [0.30, 0.00],
            [0.40, 0.00],
            [0.50, -0.12],
            [0.60, 0.00],
        ],
        dtype=float,
    )

    default_smoothed = path_optimizer_module._smooth_polyline(raw, window_size=5)
    visual_smoothed = path_optimizer_module._smooth_polyline(raw, window_size=3)

    visual_peak = float(np.max(np.abs(visual_smoothed[2:5, 1])))
    default_peak = float(np.max(np.abs(default_smoothed[2:5, 1])))
    assert visual_peak > default_peak


def test_path_optimizer_map_authority_keeps_route_source_not_visual_primary() -> None:
    base_ctx = make_context(
        pose_x=0.0,
        pose_y=0.0,
        pose_yaw=0.0,
        horizon_n=8,
        dt=0.05,
        current_lanelet_id="n1->n2",
        route_waypoints=[
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.9, 0.0, 0.0],
        ],
    )
    ctx = replace(
        base_ctx,
        route=replace(
            base_ctx.route,
            next_semantic_type="intersection",
            next_semantic_distance_m=0.30,
        ),
        lane_observation=LaneObservation(
            detected_sides=("left", "right"),
            measurement_mode="two_line",
            quality=1.0,
            direct_error_valid=True,
            direct_error_m=0.06,
            center_waypoints_body=tuple((0.05 * i, 0.12, 0.0) for i in range(20)),
            lane_width_m=0.35,
        ),
    )
    raw_path = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.9, 0.0, 0.0],
        ],
        dtype=float,
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=raw_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.30, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
        notes={"path_source": "route_waypoints", "path_authority": "map"},
    )

    out = PathOptimizer().optimize(plan, ctx)

    assert out.notes["visual_lane_reentry_reason"] != "path_source_visual_waypoints"
    assert out.notes["route_corridor_authoritative"] is True
    assert abs(float(out.target_path[0, 1]) - ctx.pose.fused_pose.y) < 1e-6


def test_path_optimizer_clamps_route_back_inside_lanelet_corridor() -> None:
    lanelet_map = from_track_graph(
        _build_track_graph(
            [("n1", 0.0, 0.0, 0), ("n2", 1.0, 0.0, 0), ("n3", 2.0, 0.0, 0)]
        ),
        step_m=0.10,
    )
    ctx = _with_route_signature(
        make_context(
            pose_x=0.0,
            pose_y=0.12,
            pose_yaw=0.0,
            horizon_n=12,
            dt=0.1,
            lanelet_map=lanelet_map,
            nominal_speed_mps=0.10,
        ),
        route_id="route-clamp",
        current_lanelet_id="n1->n2",
        next_lanelet_ids=("n2->n3",),
    )
    raw_path = np.array(
        [
            [0.0, 0.12, 0.0],
            [0.5, 0.12, 0.0],
            [1.0, 0.12, 0.0],
            [1.5, 0.12, 0.0],
        ],
        dtype=float,
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=raw_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.10, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )

    out = PathOptimizer().optimize(plan, ctx)

    assert out.notes["containment_mode"] == "corridor_clamp"
    assert out.notes["corridor_source"] == "lanelet_bounds"
    assert out.notes["corridor_left_bound"].shape == (ctx.horizon_n + 1, 2)
    assert out.notes["corridor_right_bound"].shape == (ctx.horizon_n + 1, 2)
    assert np.max(np.abs(out.target_path[1:, 1])) <= 0.071
    assert np.all(out.target_path[1:, 1] <= out.drivable_left_bound[1:, 1] + 1e-6)
    assert np.all(out.target_path[1:, 1] >= out.drivable_right_bound[1:, 1] - 1e-6)


def test_path_optimizer_preserves_two_line_visual_recovery_outside_stale_corridor() -> None:
    lanelet_map = from_track_graph(
        _build_track_graph(
            [("n1", 0.0, 0.0, 0), ("n2", 1.0, 0.0, 0), ("n3", 2.0, 0.0, 0)]
        ),
        step_m=0.10,
    )
    base_ctx = _with_route_signature(
        make_context(
            pose_x=0.0,
            pose_y=0.0,
            pose_yaw=0.0,
            horizon_n=12,
            dt=0.1,
            lanelet_map=lanelet_map,
            nominal_speed_mps=0.10,
            map_match_error_m=0.22,
        ),
        route_id="route-visual-recovery",
        current_lanelet_id="n1->n2",
        next_lanelet_ids=("n2->n3",),
    )
    ctx = replace(
        base_ctx,
        lane_observation=LaneObservation(
            detected_sides=("left", "right"),
            measurement_mode="two_line",
            quality=1.0,
            direct_error_valid=True,
            direct_error_m=0.08,
            planner_priority_active=True,
            center_waypoints_body=tuple((0.05 * i, -0.12, 0.0) for i in range(20)),
            lane_width_m=0.35,
        ),
    )
    raw_path = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.15, 0.12, 0.0],
            [0.40, 0.12, 0.0],
            [0.80, 0.12, 0.0],
        ],
        dtype=float,
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=raw_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.10, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
        notes={"path_source": "visual_lane_waypoints", "path_authority": "visual_recovery"},
    )

    out = PathOptimizer().optimize(plan, ctx)

    assert out.notes["containment_mode"] == "visual_primary_synthetic_bounds"
    assert out.notes["visual_primary_corridor_bypass_active"] is True
    assert out.notes["visual_primary_corridor_bypass_reason"] == "visual_recovery"
    assert np.max(out.target_path[2:8, 1]) > 0.09


def test_path_optimizer_projects_visual_reentry_inside_lanelet_corridor() -> None:
    lanelet_map = from_track_graph(
        _build_track_graph(
            [("n1", 0.0, 0.0, 0), ("n2", 1.0, 0.0, 0), ("n3", 2.0, 0.0, 0)]
        ),
        step_m=0.10,
    )
    base_ctx = _with_route_signature(
        make_context(
            pose_x=0.0,
            pose_y=0.06,
            pose_yaw=0.0,
            horizon_n=12,
            dt=0.1,
            lanelet_map=lanelet_map,
            nominal_speed_mps=0.10,
        ),
        route_id="route-corridor-visual",
        current_lanelet_id="n1->n2",
        next_lanelet_ids=("n2->n3",),
    )
    lane_observation = LaneObservation(
        measurement_mode="two_line",
        detected_sides=("left", "right"),
        direct_error_valid=True,
        direct_error_m=0.08,
        left_line_distance_m=0.095,
        right_line_distance_m=0.255,
        line_center_offset_m=0.08,
        quality=1.0,
    )
    ctx_with_corridor = replace(base_ctx, lane_observation=lane_observation)
    ctx_without_corridor = replace(base_ctx, lanelet_map=None, lane_observation=lane_observation)
    raw_path = np.array(
        [
            [0.0, 0.06, 0.0],
            [0.5, 0.06, 0.0],
            [1.0, 0.06, 0.0],
            [1.5, 0.06, 0.0],
        ],
        dtype=float,
    )
    plan = BehaviorPathPlan(
        timestamp=ctx_with_corridor.now_s,
        raw_path=raw_path,
        base_speed_profile=np.full(ctx_with_corridor.horizon_n, 0.10, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )

    out_without_corridor = PathOptimizer().optimize(plan, ctx_without_corridor)
    out_with_corridor = PathOptimizer().optimize(plan, ctx_with_corridor)

    assert out_without_corridor.notes["containment_mode"] == "synthetic_bounds"
    assert out_with_corridor.notes["corridor_source"] == "lanelet_bounds"
    assert out_with_corridor.notes["visual_lane_measurement_source"] == "line_center_offset_m"
    assert out_with_corridor.notes["corridor_visual_lane_guidance_active"] is True
    assert out_with_corridor.notes["corridor_visual_lane_guidance_mode"] == "two_line"
    assert float(out_without_corridor.target_path[1, 1]) > 0.0
    assert float(out_with_corridor.target_path[1, 1]) < 0.0


@pytest.mark.parametrize(
    ("measurement_mode", "detected_sides", "quality"),
    [
        ("two_line", ("left", "right"), 1.0),
        ("single_line", ("right",), 0.65),
    ],
)
def test_path_optimizer_preserves_visual_path_priority_inside_lanelet_corridor(
    measurement_mode: str,
    detected_sides: tuple[str, ...],
    quality: float,
) -> None:
    lanelet_map = from_track_graph(
        _build_track_graph(
            [("n1", 0.0, 0.0, 0), ("n2", 1.0, 0.0, 0), ("n3", 2.0, 0.0, 0)]
        ),
        step_m=0.10,
    )
    ctx = _with_route_signature(
        make_context(
            pose_x=0.0,
            pose_y=-0.12,
            pose_yaw=0.0,
            horizon_n=12,
            dt=0.1,
            lanelet_map=lanelet_map,
            nominal_speed_mps=0.10,
        ),
        route_id=f"route-visual-priority-{measurement_mode}",
        current_lanelet_id="n1->n2",
        next_lanelet_ids=("n2->n3",),
    )
    ctx = replace(
        ctx,
        lane_observation=LaneObservation(
            measurement_mode=measurement_mode,
            detected_sides=detected_sides,
            direct_error_valid=True,
            direct_error_m=0.12,
            quality=quality,
            planner_priority_active=True,
            control_policy_mode="ROUTE_TRACKING",
        ),
    )
    raw_path = np.array(
        [
            [0.0, -0.12, 0.0],
            [0.5, -0.12, 0.0],
            [1.0, -0.12, 0.0],
            [1.5, -0.12, 0.0],
        ],
        dtype=float,
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=raw_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.10, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
        notes={"path_source": "visual_lane_waypoints"},
    )

    out = PathOptimizer().optimize(plan, ctx)

    assert out.notes["corridor_source"] == "lanelet_bounds"
    assert out.notes["corridor_visual_lane_guidance_active"] is True
    assert out.notes["corridor_visual_path_priority_active"] is True
    assert out.notes["corridor_visual_lane_guidance_max_limit_m"] > 0.085
    assert float(out.target_path[1, 1]) < -0.08
    assert float(out.target_path[4, 1]) < float(out.target_path[1, 1])
    assert float(out.target_path[1, 1]) > -0.11


def test_path_optimizer_keeps_single_line_visual_path_when_route_match_is_far_off() -> None:
    lanelet_map = from_track_graph(
        _build_track_graph(
            [("n1", 0.0, 0.0, 0), ("n2", 1.0, 0.0, 0), ("n3", 2.0, 0.0, 0)]
        ),
        step_m=0.10,
    )
    pose = SimpleNamespace(x=0.0, y=0.38, yaw=0.0)
    ctx = _with_route_signature(
        make_context(
            pose_x=pose.x,
            pose_y=pose.y,
            pose_yaw=pose.yaw,
            speed_mps=0.04,
            horizon_n=20,
            dt=0.05,
            lanelet_map=lanelet_map,
            nominal_speed_mps=0.10,
            current_lanelet_id="n1->n2",
            next_lanelet_ids=("n2->n3",),
            map_match_error_m=0.38,
        ),
        route_id="route-single-line-visual-override",
        current_lanelet_id="n1->n2",
        next_lanelet_ids=("n2->n3",),
    )
    ctx = replace(
        ctx,
        lane_observation=LaneObservation(
            detected_sides=("left",),
            direct_error_m=0.05,
            quality=0.85,
            measurement_mode="single_line",
            direct_error_valid=True,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
            lane_width_m=0.35,
        ),
    )
    raw_path = np.array(
        [
            [0.00, 0.38, 0.0],
            [0.05, 0.372, 0.0],
            [0.15, 0.340, 0.0],
            [0.30, 0.290, 0.0],
        ],
        dtype=float,
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=raw_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.08, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
        notes={"path_source": "visual_lane_waypoints"},
    )

    out = PathOptimizer().optimize(plan, ctx)

    assert out.notes["containment_mode"] == "visual_single_line_override"
    assert out.notes["single_line_visual_route_override_active"] is True
    assert out.notes["single_line_visual_route_override_map_match_error_m"] == pytest.approx(0.38, abs=1e-6)
    assert float(out.target_path[1, 1]) == pytest.approx(raw_path[1, 1], abs=0.03)
    assert float(out.target_path[2, 1]) == pytest.approx(raw_path[2, 1], abs=0.05)


def test_path_optimizer_keeps_single_line_visual_path_when_ego_corridor_error_exceeds_lane_half_width() -> None:
    lanelet_map = from_track_graph(
        _build_track_graph(
            [("n1", 0.0, 0.0, 0), ("n2", 1.0, 0.0, 0), ("n3", 2.0, 0.0, 0)]
        ),
        step_m=0.10,
    )
    ctx = _with_route_signature(
        make_context(
            pose_x=0.0,
            pose_y=0.183,
            pose_yaw=0.0,
            speed_mps=0.04,
            horizon_n=20,
            dt=0.05,
            lanelet_map=lanelet_map,
            nominal_speed_mps=0.10,
            current_lanelet_id="n1->n2",
            next_lanelet_ids=("n2->n3",),
            map_match_error_m=0.146,
        ),
        route_id="route-single-line-corridor-override",
        current_lanelet_id="n1->n2",
        next_lanelet_ids=("n2->n3",),
    )
    ctx = replace(
        ctx,
        lane_observation=LaneObservation(
            detected_sides=("right",),
            direct_error_m=-0.03,
            quality=0.85,
            measurement_mode="single_line",
            direct_error_valid=True,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
            lane_width_m=0.35,
        ),
    )
    raw_path = np.array(
        [
            [0.00, 0.183, 0.0],
            [0.08, 0.150, 0.0],
            [0.16, 0.118, 0.0],
            [0.28, 0.090, 0.0],
        ],
        dtype=float,
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=raw_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.08, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
        notes={"path_source": "visual_lane_waypoints"},
    )

    out = PathOptimizer().optimize(plan, ctx)

    assert out.notes["containment_mode"] == "visual_single_line_override"
    assert out.notes["single_line_visual_route_override_reason"] == "corridor_pose_mismatch"
    assert out.notes["single_line_visual_route_override_map_match_triggered"] is False
    assert out.notes["single_line_visual_route_override_ego_triggered"] is True
    assert out.notes["single_line_visual_route_override_ego_corridor_error_m"] == pytest.approx(0.183, abs=1e-6)
    assert out.notes["single_line_visual_route_override_threshold_m"] == pytest.approx(0.175, abs=1e-6)
    assert float(out.target_path[1, 1]) == pytest.approx(raw_path[1, 1], abs=0.03)
    assert float(out.target_path[2, 1]) > 0.14
    assert float(out.target_path[2, 1]) < 0.18


def test_single_line_visual_corridor_flip_override_keeps_visual_direction() -> None:
    pose_x = 4.518
    pose_y = 5.829
    pose_yaw = -0.2068
    ctx = replace(
        make_context(
            pose_x=pose_x,
            pose_y=pose_y,
            pose_yaw=pose_yaw,
            speed_mps=0.04,
            horizon_n=20,
            dt=0.05,
        ),
        lane_observation=LaneObservation(
            detected_sides=("right",),
            direct_error_m=0.0318,
            quality=0.85,
            measurement_mode="single_line",
            direct_error_valid=True,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
            lane_width_m=0.35,
            debug={
                "single_line_resolved_side": "right",
                "single_line_outer_confirmed": False,
            },
        ),
    )
    raw_visual_xy = np.array(
        [
            [4.5576, 5.8460],
            [4.5972, 5.8322],
            [4.6423, 5.8160],
        ],
        dtype=float,
    )
    clamped_xy = np.array(
        [
            [4.5576, 5.8460],
            [4.5972, 5.8322],
            [4.6423, 5.8325],
        ],
        dtype=float,
    )
    containment = path_optimizer_module._ContainmentResult(
        target_xy=clamped_xy,
        left_bound=np.zeros((3, 2), dtype=float),
        right_bound=np.zeros((3, 2), dtype=float),
        corridor_available=True,
        infeasible=False,
        used_prev_safe_path=False,
        first_infeasible_index=None,
        max_violation_m=0.62,
        max_abs_offset_m=0.69,
        ego_corridor_error_m=0.05,
        touches_bound=True,
        mode="corridor_clamp",
        notes={
            "corridor_source": "lanelet_bounds",
            "corridor_required": True,
            "corridor_visual_lane_guidance_active": True,
            "corridor_visual_lane_guidance_reason": "single_line_physical_reentry",
            "corridor_visual_lane_guidance_mode": "single_line",
            "corridor_visual_lane_guidance_shift_m": 0.0,
            "corridor_visual_path_priority_active": True,
        },
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=np.zeros((3, 3), dtype=float),
        base_speed_profile=np.full(ctx.horizon_n, 0.08, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
        notes={"path_source": "visual_lane_waypoints"},
    )

    override = path_optimizer_module._maybe_override_single_line_visual_corridor_flip(
        raw_visual_xy=raw_visual_xy,
        containment=containment,
        path_plan=plan,
        ctx=ctx,
    )

    assert override is not None
    assert override.mode == "visual_single_line_override"
    assert override.notes["single_line_visual_route_override_reason"] == "corridor_heading_flip"
    np.testing.assert_allclose(override.target_xy, raw_visual_xy, atol=1e-6)
    assert override.notes["single_line_visual_corridor_flip_raw_heading_delta_deg"] < -6.0
    assert override.notes["single_line_visual_corridor_flip_clamped_heading_delta_deg"] > 6.0


def test_path_optimizer_holds_previous_visual_path_when_route_tracking_jumps_behind_ego() -> None:
    optimizer = PathOptimizer()

    visual_pose = SimpleNamespace(x=5.722363642841962, y=5.525636788346178, yaw=-0.43400371594302667)
    visual_ctx = replace(
        make_context(
            pose_x=visual_pose.x,
            pose_y=visual_pose.y,
            pose_yaw=visual_pose.yaw,
            speed_mps=0.08,
            horizon_n=20,
            dt=0.05,
            lanelet_map=None,
            nominal_speed_mps=0.08,
            current_lanelet_id="lanelet-a",
            next_lanelet_ids=("lanelet-b",),
            map_match_error_m=0.36,
        ),
        lane_observation=LaneObservation(
            detected_sides=("right",),
            direct_error_m=-0.01,
            quality=0.85,
            measurement_mode="single_line",
            direct_error_valid=True,
            heading_error_rad=-0.25,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
            extrapolated_side="left",
            lane_width_m=0.35,
        ),
    )
    visual_plan = BehaviorPathPlan(
        timestamp=visual_ctx.now_s,
        raw_path=np.array(
            [
                [5.722363642841962, 5.525636788346178, -0.43400371594302667],
                [5.754943695006089, 5.4922284670695465, -0.6778540496471305],
                [5.794981534785739, 5.459993271185495, -0.9005598564270282],
                [5.835019374565389, 5.427758075301443, -1.0000000000000000],
            ],
            dtype=float,
        ),
        base_speed_profile=np.full(visual_ctx.horizon_n, 0.08, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
        notes={"path_source": "visual_lane_waypoints"},
    )

    visual_out = optimizer.optimize(visual_plan, visual_ctx)

    route_pose = SimpleNamespace(x=5.73103095164163, y=5.533562810155766, yaw=-0.4432922572373612)
    route_ctx = replace(
        make_context(
            pose_x=route_pose.x,
            pose_y=route_pose.y,
            pose_yaw=route_pose.yaw,
            speed_mps=0.08,
            horizon_n=20,
            dt=0.05,
            lanelet_map=None,
            nominal_speed_mps=0.08,
            current_lanelet_id="lanelet-a",
            next_lanelet_ids=("lanelet-b",),
            map_match_error_m=0.35814767634116285,
        ),
        lane_observation=LaneObservation(
            quality=0.1,
            measurement_mode="route_tracking",
            direct_error_valid=False,
            planner_priority_active=True,
            blind_mode="route_tracking",
        ),
    )
    route_plan = BehaviorPathPlan(
        timestamp=route_ctx.now_s + 0.05,
        raw_path=np.array(
            [
                [5.73103095164163, 5.533562810155766, -0.4432922572373612],
                [5.773900810569153, 5.82196351492665, -0.0081047076016219],
                [5.8207326821997585, 5.82158394798983, -0.007999535207427695],
                [5.867564553830364, 5.82120438105301, -0.00789436281323349],
            ],
            dtype=float,
        ),
        base_speed_profile=np.full(route_ctx.horizon_n, 0.08, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
        notes={"path_source": "route_waypoints"},
    )

    route_out = optimizer.optimize(route_plan, route_ctx)

    assert route_out.notes["route_tracking_visual_hold_applied"] is True
    assert route_out.notes["route_tracking_visual_hold_reason"] == "route_sample1_behind_ego"
    assert route_out.notes["containment_mode"] == "route_tracking_visual_hold"
    np.testing.assert_allclose(route_out.target_path[:3, :2], visual_out.target_path[:3, :2], atol=1e-6)


def test_path_optimizer_holds_previous_visual_path_when_route_tracking_flips_heading_from_visual() -> None:
    optimizer = PathOptimizer()
    visual_target_path = np.array(
        [
            [4.520490871529076, 5.842273766459339, -0.2201185718573798],
            [4.584169460089626, 5.81979715326192, -0.34809800353641873],
            [4.624707866699549, 5.805086796874565, -0.34809800353647025],
            [4.665246273309472, 5.790376440487209, -0.34809800353647025],
        ],
        dtype=float,
    )
    visual_left_bound, visual_right_bound = path_optimizer_module._build_drivable_bounds(
        visual_target_path,
        half_width_m=path_optimizer_module._DRIVABLE_HALF_WIDTH_M,
    )
    optimizer._prev_target_path = np.array(visual_target_path, copy=True)
    optimizer._prev_target_path_source = "visual_lane_waypoints"
    optimizer._prev_safe_target_path = np.array(visual_target_path, copy=True)
    optimizer._prev_safe_left_bound = np.array(visual_left_bound, copy=True)
    optimizer._prev_safe_right_bound = np.array(visual_right_bound, copy=True)
    optimizer._prev_safe_path_source = "visual_lane_waypoints"

    route_pose = SimpleNamespace(x=4.53870718944138, y=5.816499692936836, yaw=-0.24338691727223957)
    route_ctx = replace(
        make_context(
            pose_x=route_pose.x,
            pose_y=route_pose.y,
            pose_yaw=route_pose.yaw,
            speed_mps=0.04,
            horizon_n=20,
            dt=0.05,
            lanelet_map=None,
            nominal_speed_mps=0.08,
            current_lanelet_id="lanelet-a",
            next_lanelet_ids=("lanelet-b",),
            map_match_error_m=0.08437458301064549,
        ),
        lane_observation=LaneObservation(
            quality=0.1,
            measurement_mode="route_tracking",
            direct_error_valid=False,
            planner_priority_active=True,
            blind_mode="route_tracking",
        ),
    )
    route_plan = BehaviorPathPlan(
        timestamp=route_ctx.now_s + 0.05,
        raw_path=np.array(
            [
                [4.5379, 5.8167, -0.24309355274294725],
                [4.574513307753829, 5.832032629877323, 0.05936590524644689],
                [4.625464593019203, 5.835060957475847, 0.1430849117151658],
                [4.675992505080887, 5.836101248761919, 0.1430849117151658],
            ],
            dtype=float,
        ),
        base_speed_profile=np.full(route_ctx.horizon_n, 0.04, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
        notes={"path_source": "route_waypoints"},
    )

    route_out = optimizer.optimize(route_plan, route_ctx)

    assert route_out.notes["route_tracking_visual_hold_applied"] is True
    assert route_out.notes["route_tracking_visual_hold_reason"] == "route_heading_flip_from_visual"
    assert route_out.notes["containment_mode"] == "route_tracking_visual_hold"
    assert route_out.notes["route_tracking_visual_hold_route_heading_delta_deg"] > 6.0
    assert route_out.notes["route_tracking_visual_hold_visual_heading_delta_deg"] < -5.0
    np.testing.assert_allclose(route_out.target_path[:3, :2], visual_target_path[:3, :2], atol=1e-6)


def test_path_optimizer_holds_previous_path_on_two_line_to_single_line_heading_jump() -> None:
    optimizer = PathOptimizer()
    signature = path_optimizer_module._BlendSignature(
        scenario_name=ScenarioName.INTERSECTION.value,
        route_id="route-1",
        current_lanelet_id="lanelet-a",
        first_next_lanelet_id="lanelet-b",
        bridge_mode=None,
    )
    previous_target_path = np.array(
        [
            [0.00, 0.00, 0.0],
            [0.05, 0.00, 0.0],
            [0.10, 0.00, 0.0],
            [0.15, 0.00, 0.0],
        ],
        dtype=float,
    )
    optimizer._prev_target_path = np.array(previous_target_path, copy=True)
    optimizer._prev_blend_signature = signature
    optimizer._prev_target_path_source = "route_waypoints"
    optimizer._prev_lane_measurement_mode = "two_line"

    base_ctx = make_context(
        pose_x=0.0,
        pose_y=0.0,
        pose_yaw=0.0,
        speed_mps=0.04,
        horizon_n=3,
        dt=0.05,
        lanelet_map=None,
        nominal_speed_mps=0.04,
        current_lanelet_id="lanelet-a",
        next_lanelet_ids=("lanelet-b",),
    )
    ctx = replace(
        base_ctx,
        route=replace(
            base_ctx.route,
            route_active=True,
            route_id="route-1",
            current_lanelet_id="lanelet-a",
            next_lanelet_ids=("lanelet-b",),
        ),
        lane_observation=LaneObservation(
            detected_sides=("right",),
            direct_error_m=-0.0496,
            quality=0.85,
            measurement_mode="single_line",
            direct_error_valid=True,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
        ),
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=np.array(
            [
                [0.00, -0.05, 0.0],
                [0.05, -0.02, 0.0],
                [0.10, -0.03, 0.0],
                [0.15, -0.03, 0.0],
            ],
            dtype=float,
        ),
        base_speed_profile=np.full(ctx.horizon_n, 0.04, dtype=float),
        scenario_name=ScenarioName.INTERSECTION.value,
        valid=True,
        notes={"path_source": "visual_lane_waypoints"},
    )

    out = optimizer.optimize(plan, ctx)

    assert out.notes["single_line_transition_hold_applied"] is True
    assert out.notes["single_line_transition_hold_reason"] == "two_line_to_single_line_heading_jump"
    assert out.notes["single_line_transition_hold_heading_delta_deg"] > 20.0
    np.testing.assert_allclose(out.target_path[:4, :2], previous_target_path[:, :2], atol=1e-6)


def test_compute_headings_uses_centered_tangent_for_interior_samples() -> None:
    samples_xy = np.array(
        [
            [5.5833, 5.6899],
            [5.653733727571165, 5.819957402263771],
            [5.70900956489773, 5.835551018033613],
        ],
        dtype=float,
    )

    headings = path_optimizer_module._compute_headings(
        samples_xy,
        initial_yaw=-0.1581515328975581,
    )

    expected_centered = math.atan2(
        float(samples_xy[2, 1] - samples_xy[0, 1]),
        float(samples_xy[2, 0] - samples_xy[0, 0]),
    )
    assert math.isclose(float(headings[0]), -0.1581515328975581, abs_tol=1e-9)
    assert math.isclose(float(headings[1]), float(expected_centered), abs_tol=1e-9)


def test_path_optimizer_recovers_tracking_lanelet_corridor_when_context_map_missing(monkeypatch) -> None:
    recovered_lanelet_map = from_track_graph(
        _build_track_graph(
            [("n1", 0.0, 0.0, 0), ("n2", 1.0, 0.0, 0), ("n3", 2.0, 0.0, 0)]
        ),
        step_m=0.10,
    )
    monkeypatch.setattr(
        path_optimizer_module,
        "load_tracking_lanelet_map",
        lambda force_reload=False: recovered_lanelet_map,
    )
    ctx = _with_route_signature(
        make_context(
            pose_x=0.0,
            pose_y=0.06,
            pose_yaw=0.0,
            horizon_n=12,
            dt=0.1,
            lanelet_map=None,
            nominal_speed_mps=0.10,
        ),
        route_id="route-runtime-recovery",
        current_lanelet_id="n1->n2",
        next_lanelet_ids=("n2->n3",),
    )
    raw_path = np.array(
        [
            [0.0, 0.06, 0.0],
            [0.5, 0.06, 0.0],
            [1.0, 0.06, 0.0],
            [1.5, 0.06, 0.0],
        ],
        dtype=float,
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=raw_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.10, dtype=float),
        scenario_name=ScenarioName.INTERSECTION.value,
        valid=True,
    )

    out = PathOptimizer().optimize(plan, ctx)

    assert out.notes["containment_mode"] == "corridor_nominal"
    assert out.notes["corridor_source"] == "lanelet_bounds_tracking_recovery"
    assert out.notes["corridor_required_missing"] is False


def test_path_optimizer_reuses_previous_safe_path_when_corridor_becomes_infeasible() -> None:
    lanelet_map = from_track_graph(
        _build_track_graph(
            [("n1", 0.0, 0.0, 0), ("n2", 1.0, 0.0, 0), ("n3", 2.0, 0.0, 0)]
        ),
        step_m=0.10,
    )
    ctx = _with_route_signature(
        make_context(
            pose_x=0.0,
            pose_y=0.0,
            pose_yaw=0.0,
            horizon_n=12,
            dt=0.1,
            lanelet_map=lanelet_map,
            nominal_speed_mps=0.10,
        ),
        route_id="route-safe-reuse",
        current_lanelet_id="n1->n2",
        next_lanelet_ids=("n2->n3",),
    )
    safe_raw_path = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ],
        dtype=float,
    )
    safe_plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=safe_raw_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.10, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )

    optimizer = PathOptimizer()
    safe_out = optimizer.optimize(safe_plan, ctx)

    for lanelet_id in ("n1->n2", "n2->n3"):
        lanelet = lanelet_map.get_lanelet(lanelet_id)
        assert lanelet is not None
        centerline = np.asarray(lanelet.centerline, dtype=float)
        left_boundary = np.array(centerline, copy=True)
        left_boundary[:, 1] += 0.03
        right_boundary = np.array(centerline, copy=True)
        right_boundary[:, 1] -= 0.03
        lanelet_map._lanelets[lanelet_id] = Lanelet(
            lanelet_id=lanelet.lanelet_id,
            source_node_id=lanelet.source_node_id,
            target_node_id=lanelet.target_node_id,
            centerline=lanelet.centerline,
            length_m=lanelet.length_m,
            attribute=lanelet.attribute,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            predecessor_ids=lanelet.predecessor_ids,
            successor_ids=lanelet.successor_ids,
            regulatory_element_ids=lanelet.regulatory_element_ids,
        )

    degraded_raw_path = np.array(
        [
            [0.0, 0.02, 0.0],
            [0.5, 0.02, 0.0],
            [1.0, 0.02, 0.0],
            [1.5, 0.02, 0.0],
        ],
        dtype=float,
    )
    degraded_plan = BehaviorPathPlan(
        timestamp=ctx.now_s + 0.05,
        raw_path=degraded_raw_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.10, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )

    reused_out = optimizer.optimize(degraded_plan, ctx)

    assert reused_out.notes["used_prev_safe_path"] is True
    assert reused_out.notes["containment_mode"] == "reused_prev_safe_path"
    np.testing.assert_allclose(reused_out.target_path[1:, :2], safe_out.target_path[1:, :2], atol=1e-6)


def test_path_optimizer_biases_prefix_back_toward_visual_lane_center() -> None:
    ctx = replace(
        make_context(
            pose_x=0.0,
            pose_y=0.0,
            pose_yaw=0.0,
            speed_mps=0.04,
            horizon_n=12,
            dt=0.05,
            current_lanelet_id="lanelet-a",
            nominal_speed_mps=0.1,
        ),
        lane_observation=LaneObservation(
            detected_sides=("left", "right"),
            direct_error_m=-0.20,
            quality=1.0,
            measurement_mode="two_line",
            direct_error_valid=True,
            control_policy_mode="ROUTE_TRACKING",
        ),
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [1.2, 0.0, 0.0],
            ],
            dtype=float,
        ),
        base_speed_profile=np.full(ctx.horizon_n, 0.1, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
        notes={"path_source": "route_waypoints"},
    )

    out = PathOptimizer().optimize(plan, ctx)

    assert out.notes["visual_lane_reentry_active"] is True
    assert out.notes["visual_lane_reentry_applied"] is True
    assert out.notes["visual_lane_prefix_samples"] >= 4
    peak_idx = int(np.argmax(out.target_path[:, 1]))
    assert 1 < peak_idx < out.target_path.shape[0] - 1
    assert 0.005 < out.target_path[1, 1] < 0.03
    assert out.target_path[peak_idx, 1] > out.target_path[1, 1] * 2.0
    assert out.target_path[peak_idx, 1] > out.target_path[-1, 1]


def test_path_optimizer_limits_visual_reentry_in_intersection_scenario() -> None:
    ctx = replace(
        make_context(
            pose_x=0.0,
            pose_y=0.0,
            pose_yaw=0.0,
            speed_mps=0.04,
            horizon_n=12,
            dt=0.05,
            current_lanelet_id="lanelet-a",
            nominal_speed_mps=0.1,
        ),
        lane_observation=LaneObservation(
            detected_sides=("left", "right"),
            direct_error_m=-0.40,
            line_center_offset_m=-0.22,
            quality=1.0,
            measurement_mode="two_line",
            direct_error_valid=True,
            control_policy_mode="ROUTE_TRACKING",
        ),
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [1.2, 0.0, 0.0],
            ],
            dtype=float,
        ),
        base_speed_profile=np.full(ctx.horizon_n, 0.1, dtype=float),
        scenario_name=ScenarioName.INTERSECTION.value,
        valid=True,
        notes={"path_source": "route_waypoints"},
    )

    out = PathOptimizer().optimize(plan, ctx)

    assert out.notes["visual_lane_reentry_active"] is True
    assert out.notes["visual_lane_reentry_profile"] == ScenarioName.INTERSECTION.value
    assert out.notes["visual_lane_measurement_source"] == "line_center_offset_m"
    assert abs(out.notes["visual_lane_shift_m"]) <= 0.05 + 1e-9
    assert out.notes["visual_lane_prefix_samples"] >= 1
    assert float(np.max(np.abs(out.target_path[:, 1]))) < 0.06


def test_path_optimizer_biases_prefix_with_valid_single_line_physical_measurement() -> None:
    ctx = replace(
        make_context(
            pose_x=0.0,
            pose_y=0.0,
            pose_yaw=0.0,
            speed_mps=0.04,
            horizon_n=12,
            dt=0.05,
            current_lanelet_id="lanelet-a",
            nominal_speed_mps=0.1,
        ),
        lane_observation=LaneObservation(
            detected_sides=("right",),
            direct_error_m=-0.20,
            quality=0.65,
            measurement_mode="single_line",
            direct_error_valid=True,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
        ),
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [1.2, 0.0, 0.0],
            ],
            dtype=float,
        ),
        base_speed_profile=np.full(ctx.horizon_n, 0.1, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
        notes={"path_source": "route_waypoints"},
    )

    out = PathOptimizer().optimize(plan, ctx)

    assert out.notes["visual_lane_reentry_active"] is True
    assert out.notes["visual_lane_measurement_mode"] == "single_line"
    assert out.notes["visual_lane_reentry_reason"] == "single_line_physical_reentry"
    assert out.notes["visual_lane_reentry_applied"] is True
    peak_idx = int(np.argmax(out.target_path[:, 1]))
    assert 1 < peak_idx < out.target_path.shape[0] - 1
    assert 0.005 < out.target_path[1, 1] < 0.03
    assert out.target_path[peak_idx, 1] > out.target_path[1, 1] * 2.0
    assert out.target_path[peak_idx, 1] > out.target_path[-1, 1]


def test_path_optimizer_ignores_single_line_reentry_when_direction_conflicts_with_route() -> None:
    ctx = replace(
        make_context(
            pose_x=0.0,
            pose_y=0.0,
            pose_yaw=0.0,
            speed_mps=0.04,
            horizon_n=12,
            dt=0.05,
            current_lanelet_id="lanelet-a",
            nominal_speed_mps=0.1,
        ),
        lane_observation=LaneObservation(
            detected_sides=("right",),
            direct_error_m=-0.20,
            quality=0.65,
            measurement_mode="single_line",
            direct_error_valid=True,
            heading_error_rad=math.radians(20.0),
            camera_yaw_hint_rad=math.radians(20.0),
            camera_yaw_hint_confidence=0.9,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
        ),
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [1.2, 0.0, 0.0],
            ],
            dtype=float,
        ),
        base_speed_profile=np.full(ctx.horizon_n, 0.1, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
        notes={"path_source": "route_waypoints"},
    )

    out = PathOptimizer().optimize(plan, ctx)

    assert out.notes["visual_lane_reentry_active"] is False
    assert out.notes["visual_lane_measurement_mode"] == "single_line"
    assert out.notes["visual_lane_reentry_reason"] == "single_line_direction_conflict"
    assert out.notes["visual_lane_reentry_applied"] is False
    assert abs(float(out.target_path[1, 1])) < 1e-6


def test_path_optimizer_caps_single_line_visual_prefix_heading_when_error_is_small() -> None:
    ctx = replace(
        make_context(
            pose_x=0.0,
            pose_y=0.0,
            pose_yaw=0.0,
            speed_mps=0.04,
            horizon_n=12,
            dt=0.05,
            nominal_speed_mps=0.1,
        ),
        lane_observation=LaneObservation(
            detected_sides=("right",),
            direct_error_m=-0.03,
            quality=0.85,
            measurement_mode="single_line",
            direct_error_valid=True,
            planner_priority_active=True,
        ),
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=np.array(
            [
                [0.00, 0.00, 0.0],
                [0.04, -0.012, 0.0],
                [0.09, -0.034, 0.0],
                [0.15, -0.070, 0.0],
                [0.24, -0.120, 0.0],
                [0.36, -0.185, 0.0],
                [0.52, -0.260, 0.0],
                [0.72, -0.340, 0.0],
            ],
            dtype=float,
        ),
        base_speed_profile=np.full(ctx.horizon_n, 0.1, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
        notes={"path_source": "visual_lane_waypoints"},
    )

    out = PathOptimizer().optimize(plan, ctx)

    limit_deg = float(out.notes["visual_prefix_single_line_heading_limit_deg"])
    assert out.notes["visual_prefix_single_line_heading_cap_applied"] is True
    assert limit_deg == pytest.approx(8.4, abs=0.2)
    assert out.notes["visual_prefix_sample1_y_left_after_m"] > 0.0
    assert out.notes["visual_prefix_sample1_y_left_after_m"] < out.notes["visual_prefix_sample1_y_left_before_m"]
    for idx in range(1, 4):
        dx = float(out.target_path[idx, 0] - out.target_path[idx - 1, 0])
        dy = float(out.target_path[idx, 1] - out.target_path[idx - 1, 1])
        assert abs(math.degrees(math.atan2(dy, dx))) <= limit_deg + 0.2


def test_path_optimizer_blends_prefix_for_same_corridor() -> None:
    optimizer = PathOptimizer()

    ctx_prev = _with_route_signature(
        make_context(
            pose_x=0.0,
            pose_y=0.0,
            pose_yaw=0.0,
            horizon_n=10,
            dt=0.1,
        ),
        route_id="route-a",
        current_lanelet_id="lanelet-a",
        next_lanelet_ids=("lanelet-b",),
    )
    plan_prev = BehaviorPathPlan(
        timestamp=ctx_prev.now_s,
        raw_path=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.00, 0.0],
                [2.0, 0.00, 0.0],
                [3.0, 0.00, 0.0],
            ],
            dtype=float,
        ),
        base_speed_profile=np.full(ctx_prev.horizon_n, 0.8, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )
    out_prev = optimizer.optimize(plan_prev, ctx_prev)

    ctx_curr = _with_route_signature(
        make_context(
            pose_x=0.0,
            pose_y=0.0,
            pose_yaw=0.0,
            horizon_n=10,
            dt=0.1,
        ),
        route_id="route-a",
        current_lanelet_id="lanelet-a",
        next_lanelet_ids=("lanelet-b",),
    )
    plan_curr = BehaviorPathPlan(
        timestamp=ctx_curr.now_s,
        raw_path=np.array(
            [
                [0.0, 0.01, 0.0],
                [1.0, 0.01, 0.0],
                [2.0, 0.01, 0.0],
                [3.0, 0.01, 0.0],
            ],
            dtype=float,
        ),
        base_speed_profile=np.full(ctx_curr.horizon_n, 0.8, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )
    out_blended = optimizer.optimize(plan_curr, ctx_curr)
    out_clean = PathOptimizer().optimize(plan_curr, ctx_curr)

    assert out_blended.target_path[1, 1] < out_clean.target_path[1, 1]
    assert abs(out_blended.target_path[1, 1] - out_prev.target_path[1, 1]) < abs(
        out_clean.target_path[1, 1] - out_prev.target_path[1, 1]
    )


def test_path_optimizer_skips_blend_when_corridor_signature_changes() -> None:
    optimizer = PathOptimizer()

    ctx_prev = _with_route_signature(
        make_context(
            pose_x=0.0,
            pose_y=0.0,
            pose_yaw=0.0,
            horizon_n=10,
            dt=0.1,
        ),
        route_id="route-a",
        current_lanelet_id="lanelet-a",
        next_lanelet_ids=("lanelet-b",),
    )
    plan_prev = BehaviorPathPlan(
        timestamp=ctx_prev.now_s,
        raw_path=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        base_speed_profile=np.full(ctx_prev.horizon_n, 0.8, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )
    optimizer.optimize(plan_prev, ctx_prev)

    ctx_curr = _with_route_signature(
        make_context(
            pose_x=0.0,
            pose_y=0.0,
            pose_yaw=0.0,
            horizon_n=10,
            dt=0.1,
        ),
        route_id="route-b",
        current_lanelet_id="lanelet-a",
        next_lanelet_ids=("lanelet-b",),
    )
    plan_curr = BehaviorPathPlan(
        timestamp=ctx_curr.now_s,
        raw_path=np.array(
            [
                [0.0, 0.05, 0.0],
                [1.0, 0.05, 0.0],
                [2.0, 0.05, 0.0],
                [3.0, 0.05, 0.0],
            ],
            dtype=float,
        ),
        base_speed_profile=np.full(ctx_curr.horizon_n, 0.8, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )
    out_sequential = optimizer.optimize(plan_curr, ctx_curr)
    out_clean = PathOptimizer().optimize(plan_curr, ctx_curr)

    np.testing.assert_allclose(out_sequential.target_path, out_clean.target_path, atol=1e-6)


def test_path_optimizer_skips_blend_when_lanelet_handoff_changes_corridor() -> None:
    lanelet_map = _load_sim_lanelet_map()
    horizon_n = 20
    dt = 0.05
    target_speed_mps = 0.4

    prev_pose = (-7.577423598851429, 0.07591142194934289, 2.616355407401163)
    prev_ctx = _with_route_signature(
        make_context(
            pose_x=prev_pose[0],
            pose_y=prev_pose[1],
            pose_yaw=prev_pose[2],
            horizon_n=horizon_n,
            dt=dt,
            lanelet_map=lanelet_map,
            nominal_speed_mps=target_speed_mps,
        ),
        route_id="route-9",
        current_lanelet_id="22",
        next_lanelet_ids=("33", "1945", "186"),
    )
    prev_path = build_target_path(
        lanelet_map=lanelet_map,
        start_lanelet_id="22",
        start_xy=prev_pose[:2],
        target_speed_mps=target_speed_mps,
        horizon_n=horizon_n,
        dt=dt,
        next_lanelet_hint_ids=("33", "1945", "186"),
    )
    _require_non_degenerate_path(prev_path, label="previous lanelet handoff path")
    prev_plan = BehaviorPathPlan(
        timestamp=prev_ctx.now_s,
        raw_path=prev_path,
        base_speed_profile=np.full(horizon_n, target_speed_mps, dtype=float),
        scenario_name=ScenarioName.INTERSECTION.value,
        valid=True,
    )

    curr_pose = (-7.581833088920145, 0.08431202830377618, 2.6018689846112317)
    curr_ctx = _with_route_signature(
        make_context(
            pose_x=curr_pose[0],
            pose_y=curr_pose[1],
            pose_yaw=curr_pose[2],
            horizon_n=horizon_n,
            dt=dt,
            lanelet_map=lanelet_map,
            nominal_speed_mps=target_speed_mps,
        ),
        route_id="route-9",
        current_lanelet_id="2248",
        next_lanelet_ids=("33", "1945", "186"),
    )
    curr_path = build_target_path(
        lanelet_map=lanelet_map,
        start_lanelet_id="2248",
        start_xy=curr_pose[:2],
        target_speed_mps=target_speed_mps,
        horizon_n=horizon_n,
        dt=dt,
        next_lanelet_hint_ids=("33", "1945", "186"),
    )
    _require_non_degenerate_path(curr_path, label="current lanelet handoff path")
    curr_plan = BehaviorPathPlan(
        timestamp=curr_ctx.now_s,
        raw_path=curr_path,
        base_speed_profile=np.full(horizon_n, target_speed_mps, dtype=float),
        scenario_name=ScenarioName.INTERSECTION.value,
        valid=True,
    )

    optimizer = PathOptimizer()
    optimizer.optimize(prev_plan, prev_ctx)
    out_curr = optimizer.optimize(curr_plan, curr_ctx)

    assert out_curr.target_path[1, 1] > out_curr.target_path[0, 1]


def test_path_optimizer_route_reference_stays_on_planned_corridor_after_lanelet_handoff() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    router = OsmRouteGraph(str(osm_path), step_m=0.05, start_lanelet_id="9")
    route = router.go_to("9", {"lanelet_id": "186"})
    _require_non_empty_route(route, lanelet_id="186")

    pose = (-7.440466588998336, -0.03437785703902251, 2.977118010495928)
    matched_pose = (-7.37819383706487, -0.06917198359264448, 3.1408697479755103)
    ctx = _with_route_signature(
        make_context(
            pose_x=pose[0],
            pose_y=pose[1],
            pose_yaw=pose[2],
            horizon_n=20,
            dt=0.05,
            lanelet_map=None,
            nominal_speed_mps=0.1,
            route_waypoints=route.waypoints.tolist(),
            matched_idx=16,
            target_idx=18,
            matched_pose=matched_pose,
        ),
        route_id="route-1",
        current_lanelet_id="22",
        next_lanelet_ids=("33", "1945", "186"),
    )
    raw_path, bridge_meta = build_target_path_from_route(
        route_waypoints=route.waypoints,
        matched_idx=16,
        start_xy=pose[:2],
        start_yaw_rad=pose[2],
        matched_xy=matched_pose[:2],
        target_speed_mps=0.1,
        horizon_n=20,
        dt=0.05,
        return_metadata=True,
    )
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=raw_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.1, dtype=float),
        scenario_name=ScenarioName.INTERSECTION.value,
        valid=True,
        notes=bridge_meta,
    )

    out = PathOptimizer().optimize(plan, ctx)
    heading_error_deg = abs(
        math.degrees(
            math.atan2(
                math.sin(float(out.target_path[0, 2]) - pose[2]),
                math.cos(float(out.target_path[0, 2]) - pose[2]),
            )
        )
    )

    assert out.target_path[1, 1] <= out.target_path[0, 1] + 0.01
    assert out.target_path[2, 1] <= out.target_path[1, 1] + 0.01
    assert out.target_path[10, 1] < out.target_path[8, 1]
    assert out.target_path[12, 1] < out.target_path[10, 1]
    assert heading_error_deg < 1e-3


def test_path_optimizer_preserves_current_yaw_at_tp0_for_route_connector() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    router = OsmRouteGraph(str(osm_path), step_m=0.05, start_lanelet_id="9")
    route = router.go_to("9", {"lanelet_id": "186"})
    _require_non_empty_route(route, lanelet_id="186")

    pose = (-7.487974940853119, 0.0001288460878235186, 2.8383446671999835)
    matched_pose = (-7.485915866731705, -0.06909411071905112, 3.1408697479755103)
    ctx = _with_route_signature(
        make_context(
            pose_x=pose[0],
            pose_y=pose[1],
            pose_yaw=pose[2],
            horizon_n=20,
            dt=0.05,
            lanelet_map=None,
            nominal_speed_mps=0.1,
            route_waypoints=route.waypoints.tolist(),
            matched_idx=16,
            target_idx=18,
            matched_pose=matched_pose,
        ),
        route_id="route-1",
        current_lanelet_id="22",
        next_lanelet_ids=("33", "1945", "186"),
    )
    raw_path, bridge_meta = build_target_path_from_route(
        route_waypoints=route.waypoints,
        matched_idx=16,
        start_xy=pose[:2],
        start_yaw_rad=pose[2],
        matched_xy=matched_pose[:2],
        target_speed_mps=0.1,
        horizon_n=20,
        dt=0.05,
        return_metadata=True,
    )
    out = PathOptimizer().optimize(
        BehaviorPathPlan(
            timestamp=ctx.now_s,
            raw_path=raw_path,
            base_speed_profile=np.full(ctx.horizon_n, 0.1, dtype=float),
            scenario_name=ScenarioName.INTERSECTION.value,
            valid=True,
            notes=bridge_meta,
        ),
        ctx,
    )

    assert out.target_path[0, 2] == pytest.approx(pose[2], abs=1e-6)
    assert abs(out.target_path[1, 1] - out.target_path[0, 1]) < 0.01


def test_path_optimizer_keeps_heading_consistent_with_geometry_on_straight_hold_prefix() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    router = OsmRouteGraph(str(osm_path), step_m=0.05, start_lanelet_id="9")
    route = router.go_to("9", {"lanelet_id": "186"})
    _require_non_empty_route(route, lanelet_id="186")

    pose = (-7.052577832817733, -0.00012658051513914938, 2.926063020211632)
    matched_idx = 11
    matched_pose = tuple(float(v) for v in route.waypoints[matched_idx])
    ctx = _with_route_signature(
        make_context(
            pose_x=pose[0],
            pose_y=pose[1],
            pose_yaw=pose[2],
            horizon_n=40,
            dt=0.05,
            lanelet_map=None,
            nominal_speed_mps=0.1,
            route_waypoints=route.waypoints.tolist(),
            matched_idx=matched_idx,
            target_idx=13,
            matched_pose=matched_pose,
        ),
        route_id="route-1",
        current_lanelet_id="22",
        next_lanelet_ids=("33", "1945", "186"),
    )
    raw_path, bridge_meta = build_target_path_from_route(
        route_waypoints=route.waypoints,
        matched_idx=matched_idx,
        start_xy=pose[:2],
        start_yaw_rad=pose[2],
        matched_xy=matched_pose[:2],
        target_speed_mps=0.1,
        horizon_n=40,
        dt=0.05,
        return_metadata=True,
    )

    out = PathOptimizer().optimize(
        BehaviorPathPlan(
            timestamp=ctx.now_s,
            raw_path=raw_path,
            base_speed_profile=np.full(ctx.horizon_n, 0.1, dtype=float),
            scenario_name=ScenarioName.INTERSECTION.value,
            valid=True,
            notes=bridge_meta,
        ),
        ctx,
    )

    assert bridge_meta["bridge_mode"] == "straight_hold"
    assert out.target_path[0, 2] == pytest.approx(pose[2], abs=1e-6)
    assert _heading_tangent_error_deg(out.target_path, 1) <= 1.0
    assert _heading_tangent_error_deg(out.target_path, 2) <= 1.0


def test_path_optimizer_preserves_straight_prefix_on_small_gap_route_snap_frame() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    router = OsmRouteGraph(str(osm_path), step_m=0.05, start_lanelet_id="9")
    route = router.go_to("9", {"lanelet_id": "186"})
    _require_non_empty_route(route, lanelet_id="186")

    pose = (-7.255234883747185, -0.05990476953317684, 3.1261478062940595)
    matched_xy = (-7.2531070384316845, -0.06936312587533913, 3.1381467259780846)
    matched_idx = 14
    ctx = _with_route_signature(
        make_context(
            pose_x=pose[0],
            pose_y=pose[1],
            pose_yaw=pose[2],
            horizon_n=40,
            dt=0.05,
            lanelet_map=None,
            nominal_speed_mps=1.0,
            route_waypoints=route.waypoints.tolist(),
            matched_idx=matched_idx,
            target_idx=16,
            matched_pose=matched_xy,
        ),
        route_id="route-1",
        current_lanelet_id="22",
        next_lanelet_ids=("33", "1945", "186"),
    )
    raw_path, bridge_meta = build_target_path_from_route(
        route_waypoints=route.waypoints,
        matched_idx=matched_idx,
        start_xy=pose[:2],
        start_yaw_rad=pose[2],
        matched_xy=matched_xy[:2],
        target_speed_mps=1.0,
        horizon_n=40,
        dt=0.05,
        return_metadata=True,
    )

    out = PathOptimizer().optimize(
        BehaviorPathPlan(
            timestamp=ctx.now_s,
            raw_path=raw_path,
            base_speed_profile=np.full(ctx.horizon_n, 1.0, dtype=float),
            scenario_name=ScenarioName.INTERSECTION.value,
            valid=True,
            notes=bridge_meta,
        ),
        ctx,
    )

    assert bridge_meta["bridge_mode"] == "straight_hold"
    assert out.target_path[1, 1] >= pose[1] - 0.003
    assert out.target_path[2, 1] >= pose[1] - 0.003
    assert out.target_path[12, 1] < out.target_path[8, 1]


def test_path_optimizer_keeps_straight_hold_prefix_out_of_previous_blend() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    router = OsmRouteGraph(str(osm_path), step_m=0.05, start_lanelet_id="9")
    route = router.go_to("9", {"lanelet_id": "186"})
    _require_non_empty_route(route, lanelet_id="186")

    matched_pose = (-7.37819383706487, -0.06917198359264448, 3.1408697479755103)
    optimizer = PathOptimizer()
    target_speed_mps = 0.1
    horizon_n = 20
    dt = 0.05

    prev_pose = (-7.435586316493479, -0.0351813241361372, 2.984993119158865)
    prev_ctx = _with_route_signature(
        make_context(
            pose_x=prev_pose[0],
            pose_y=prev_pose[1],
            pose_yaw=prev_pose[2],
            horizon_n=horizon_n,
            dt=dt,
            lanelet_map=None,
            nominal_speed_mps=target_speed_mps,
            route_waypoints=route.waypoints.tolist(),
            matched_idx=16,
            target_idx=18,
            matched_pose=matched_pose,
        ),
        route_id="route-1",
        current_lanelet_id="22",
        next_lanelet_ids=("33", "1945", "186"),
    )
    prev_raw, prev_meta = build_target_path_from_route(
        route_waypoints=route.waypoints,
        matched_idx=16,
        start_xy=prev_pose[:2],
        start_yaw_rad=prev_pose[2],
        matched_xy=matched_pose[:2],
        target_speed_mps=target_speed_mps,
        horizon_n=horizon_n,
        dt=dt,
        return_metadata=True,
    )
    optimizer.optimize(
        BehaviorPathPlan(
            timestamp=prev_ctx.now_s,
            raw_path=prev_raw,
            base_speed_profile=np.full(horizon_n, target_speed_mps, dtype=float),
            scenario_name=ScenarioName.INTERSECTION.value,
            valid=True,
            notes=prev_meta,
        ),
        prev_ctx,
    )

    curr_pose = (-7.440466588998336, -0.03437785703902251, 2.977118010495928)
    curr_ctx = _with_route_signature(
        make_context(
            pose_x=curr_pose[0],
            pose_y=curr_pose[1],
            pose_yaw=curr_pose[2],
            horizon_n=horizon_n,
            dt=dt,
            lanelet_map=None,
            nominal_speed_mps=target_speed_mps,
            route_waypoints=route.waypoints.tolist(),
            matched_idx=16,
            target_idx=18,
            matched_pose=matched_pose,
        ),
        route_id="route-1",
        current_lanelet_id="22",
        next_lanelet_ids=("33", "1945", "186"),
    )
    curr_raw, curr_meta = build_target_path_from_route(
        route_waypoints=route.waypoints,
        matched_idx=16,
        start_xy=curr_pose[:2],
        start_yaw_rad=curr_pose[2],
        matched_xy=matched_pose[:2],
        target_speed_mps=target_speed_mps,
        horizon_n=horizon_n,
        dt=dt,
        return_metadata=True,
    )
    out = optimizer.optimize(
        BehaviorPathPlan(
            timestamp=curr_ctx.now_s,
            raw_path=curr_raw,
            base_speed_profile=np.full(horizon_n, target_speed_mps, dtype=float),
            scenario_name=ScenarioName.INTERSECTION.value,
            valid=True,
            notes=curr_meta,
        ),
        curr_ctx,
    )

    assert curr_meta["bridge_mode"] == "straight_hold"
    assert out.target_path[1, 1] <= out.target_path[0, 1] + 0.01
    assert out.target_path[2, 1] <= out.target_path[1, 1] + 0.01


def test_path_optimizer_ramps_visual_reentry_on_route_connector_frame() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    router = OsmRouteGraph(str(osm_path), step_m=0.05, start_lanelet_id="995")
    route = router.go_to("995", {"lanelet_id": "138"})
    _require_non_empty_route(route, lanelet_id="138")

    pose = (5.60773324453329, 6.394426395474327, -0.15746151144647136)
    matched_idx = 53
    matched_pose = tuple(float(v) for v in route.waypoints[matched_idx])
    horizon_n = 40
    dt = 0.05
    speed_mps = 0.2
    base_ctx = make_context(
        pose_x=pose[0],
        pose_y=pose[1],
        pose_yaw=pose[2],
        speed_mps=0.04,
        horizon_n=horizon_n,
        dt=dt,
        lanelet_map=None,
        nominal_speed_mps=speed_mps,
        current_lanelet_id="60",
        next_lanelet_ids=("70", "75", "85", "90", "120", "125", "138"),
        route_waypoints=route.waypoints.tolist(),
        matched_idx=matched_idx,
        target_idx=55,
        matched_pose=matched_pose,
        maneuver_type="intersection_straight",
        map_match_error_m=0.07800796679371268,
    )
    ctx = replace(
        base_ctx,
        route=replace(
            base_ctx.route,
            route_active=True,
            route_id="route-1",
            route_source="go_to",
            waypoint_mode_active=True,
            next_semantic_type="intersection",
            next_semantic_distance_m=0.4,
        ),
        lane_observation=LaneObservation(
            detected_sides=("left", "right"),
            direct_error_m=0.203,
            line_center_offset_m=0.1165,
            quality=1.0,
            measurement_mode="two_line",
            direct_error_valid=True,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
            curve_hint="ENTERING",
        ),
    )
    raw_path, bridge_meta = build_target_path_from_route(
        route_waypoints=route.waypoints,
        matched_idx=matched_idx,
        start_xy=pose[:2],
        start_yaw_rad=pose[2],
        matched_xy=matched_pose[:2],
        target_speed_mps=speed_mps,
        horizon_n=horizon_n,
        dt=dt,
        return_metadata=True,
    )

    out = PathOptimizer().optimize(
        BehaviorPathPlan(
            timestamp=ctx.now_s,
            raw_path=raw_path,
            base_speed_profile=np.full(horizon_n, speed_mps, dtype=float),
            scenario_name=ScenarioName.LANE_KEEP.value,
            valid=True,
            notes=bridge_meta | {"path_source": "route_waypoints"},
        ),
        ctx,
    )

    assert bridge_meta["bridge_mode"] == "connector"
    assert out.notes["route_visual_entry_stabilization_applied"] is True
    assert abs(_sample1_heading_delta_from_pose_deg(out.target_path, pose[2])) <= 15.5
    sample1_x = float(out.notes["route_visual_entry_sample1_x_fwd_after_m"])
    sample1_y = float(out.notes["route_visual_entry_sample1_y_left_after_m"])
    assert sample1_x > 0.01
    assert abs(sample1_y) <= math.tan(math.radians(15.5)) * sample1_x


def test_path_optimizer_keeps_precision_horizon_with_route_visual_reentry() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    router = OsmRouteGraph(str(osm_path), step_m=0.05, start_lanelet_id="50")
    route = router.go_to("50", {"lanelet_id": "75"})
    _require_non_empty_route(route, lanelet_id="75")

    pose = (5.9987, 5.8335, -0.19317891281904964)
    matched_idx = 21
    matched_pose = (6.0356250000000005, 5.8887, -0.009923236025519996)
    horizon_n = 40
    dt = 0.05
    speed_mps = 0.1
    base_ctx = make_context(
        pose_x=pose[0],
        pose_y=pose[1],
        pose_yaw=pose[2],
        speed_mps=0.04,
        horizon_n=horizon_n,
        dt=dt,
        lanelet_map=None,
        nominal_speed_mps=speed_mps,
        current_lanelet_id="60",
        next_lanelet_ids=("70", "75"),
        route_waypoints=route.waypoints.tolist(),
        matched_idx=matched_idx,
        target_idx=23,
        matched_pose=matched_pose,
        maneuver_type="intersection_straight",
        map_match_error_m=0.06641156243456423,
    )
    ctx = replace(
        base_ctx,
        route=replace(
            base_ctx.route,
            route_active=True,
            route_id="route-1",
            route_source="go_to",
            waypoint_mode_active=True,
            next_semantic_type="intersection",
            next_semantic_distance_m=0.15,
        ),
        lane_observation=LaneObservation(
            detected_sides=("left", "right"),
            direct_error_m=-0.2637,
            quality=1.0,
            measurement_mode="two_line",
            direct_error_valid=True,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
        ),
    )
    raw_path, bridge_meta = build_target_path_from_route(
        route_waypoints=route.waypoints,
        matched_idx=matched_idx,
        start_xy=pose[:2],
        start_yaw_rad=pose[2],
        matched_xy=matched_pose[:2],
        target_speed_mps=speed_mps,
        horizon_n=horizon_n,
        dt=dt,
        return_metadata=True,
    )

    out = PathOptimizer().optimize(
        BehaviorPathPlan(
            timestamp=ctx.now_s,
            raw_path=raw_path,
            base_speed_profile=np.full(horizon_n, speed_mps, dtype=float),
            scenario_name=ScenarioName.LANE_KEEP.value,
            valid=True,
            notes=bridge_meta | {"path_source": "route_waypoints"},
        ),
        ctx,
    )

    horizon_distance_m = float(np.linalg.norm(out.target_path[-1, :2] - out.target_path[0, :2]))
    # _stabilize_route_visual_entry_prefix (68bdd86da) extiende el path tras
    # el resample para corregir misalignment de heading; el horizon real
    # quedó en ~0.65 m, no en ~0.50 m.
    assert 0.45 < horizon_distance_m < 0.80
    assert out.notes["route_corridor_authoritative"] is True
    assert out.notes["visual_lane_reentry_active"] is True
    assert out.notes["visual_lane_reentry_reason"] == "two_line_reentry"
    assert out.notes["visual_lane_reentry_applied"] is True
    assert out.notes["visual_lane_prefix_samples"] >= 4
    assert out.notes["corridor_visual_lane_guidance_active"] is True


def test_path_optimizer_keeps_future_turn_out_of_low_speed_precision_horizon() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    router = OsmRouteGraph(str(osm_path), step_m=0.05, start_lanelet_id="50")
    route = router.go_to("50", {"lanelet_id": "75"})
    _require_non_empty_route(route, lanelet_id="75")

    pose = (4.117731554685099, 5.900499982449048, -1.224637025133419e-06)
    matched_pose = (4.117731554685099, 5.90045, 0.0)
    horizon_n = 40
    dt = 0.05
    speed_mps = 0.1
    base_ctx = make_context(
        pose_x=pose[0],
        pose_y=pose[1],
        pose_yaw=pose[2],
        horizon_n=horizon_n,
        dt=dt,
        lanelet_map=None,
        nominal_speed_mps=speed_mps,
        current_lanelet_id="50",
        next_lanelet_ids=("55", "60", "70", "75"),
        route_waypoints=route.waypoints.tolist(),
        matched_idx=1,
        target_idx=3,
        matched_pose=matched_pose,
        maneuver_type="intersection_straight",
        map_match_error_m=4.999731054056156e-05,
    )
    ctx = replace(
        base_ctx,
        route=replace(
            base_ctx.route,
            route_active=True,
            route_id="route-1",
            route_source="go_to",
            waypoint_mode_active=True,
            next_semantic_type="intersection",
            next_semantic_distance_m=0.4,
        ),
    )
    raw_path, bridge_meta = build_target_path_from_route(
        route_waypoints=route.waypoints,
        matched_idx=1,
        start_xy=pose[:2],
        start_yaw_rad=pose[2],
        matched_xy=matched_pose[:2],
        target_speed_mps=speed_mps,
        horizon_n=horizon_n,
        dt=dt,
        return_metadata=True,
    )
    assert bridge_meta["bridge_mode"] == "matched_route_only"
    assert raw_path[1, 0] > raw_path[0, 0]
    assert raw_path[2, 0] > raw_path[1, 0]

    out = PathOptimizer().optimize(
        BehaviorPathPlan(
            timestamp=ctx.now_s,
            raw_path=raw_path,
            base_speed_profile=np.full(horizon_n, speed_mps, dtype=float),
            scenario_name=ScenarioName.LANE_KEEP.value,
            valid=True,
            notes=bridge_meta,
        ),
        ctx,
    )

    horizon_distance_m = float(np.linalg.norm(out.target_path[-1, :2] - out.target_path[0, :2]))
    # Post-68bdd86da el path tiene un salto lateral inicial al estabilizar
    # el visual prefix; lo que importa es que dentro del precision horizon
    # los primeros 12 waypoints sean LATERALMENTE CONSTANTES (no curven
    # hacia el future turn) y que el endpoint no se vuele hacia Y bajos.
    assert 0.45 < horizon_distance_m < 0.80
    assert np.std(out.target_path[1:12, 1]) < 0.01
    assert out.target_path[-1, 1] > pose[1] - 0.02


def test_path_optimizer_keeps_precision_visual_prefix_in_front_of_ego() -> None:
    pose = SimpleNamespace(x=0.0, y=0.0, yaw=0.0)
    raw_path = build_target_path_from_visual(
        center_waypoints_body=(
            (0.00, 0.03, 0.0),
            (-0.01, 0.08, 0.0),
            (0.05, 0.14, 0.0),
            (0.14, 0.20, 0.0),
            (0.24, 0.24, 0.0),
            (0.34, 0.27, 0.0),
            (0.44, 0.29, 0.0),
        ),
        ego_pose=pose,
        target_speed_mps=0.10,
        horizon_n=40,
        dt=0.05,
    )
    base_ctx = make_context(
        pose_x=pose.x,
        pose_y=pose.y,
        pose_yaw=pose.yaw,
        speed_mps=0.04,
        horizon_n=40,
        dt=0.05,
        lanelet_map=None,
        nominal_speed_mps=0.10,
        current_lanelet_id="lanelet-a",
        next_lanelet_ids=("lanelet-b",),
        map_match_error_m=0.0,
    )
    ctx = replace(
        base_ctx,
        route=replace(
            base_ctx.route,
            route_active=True,
            route_id="route-visual-precision",
            route_source="go_to",
            waypoint_mode_active=True,
            next_semantic_type="intersection",
            next_semantic_distance_m=0.15,
        ),
    )

    out = PathOptimizer().optimize(
        BehaviorPathPlan(
            timestamp=ctx.now_s,
            raw_path=raw_path,
            base_speed_profile=np.full(ctx.horizon_n, 0.08, dtype=float),
            scenario_name=ScenarioName.LANE_KEEP.value,
            valid=True,
            notes={"path_source": "visual_lane_waypoints"},
        ),
        ctx,
    )

    dx = float(out.target_path[1, 0] - pose.x)
    dy = float(out.target_path[1, 1] - pose.y)
    x_fwd = math.cos(pose.yaw) * dx + math.sin(pose.yaw) * dy

    assert x_fwd > 0.0
    assert _heading_tangent_error_deg(out.target_path, 1) < 20.0


def test_path_optimizer_stabilizes_visual_prefix_when_path_starts_lateral() -> None:
    lanelet_map = from_track_graph(
        _build_track_graph(
            [("n1", 0.0, 0.0, 0), ("n2", 1.0, 0.0, 0), ("n3", 2.0, 0.0, 0)]
        ),
        step_m=0.10,
    )
    pose = SimpleNamespace(x=0.0, y=0.0, yaw=0.0)
    ctx = _with_route_signature(
        make_context(
            pose_x=pose.x,
            pose_y=pose.y,
            pose_yaw=pose.yaw,
            speed_mps=0.04,
            horizon_n=20,
            dt=0.05,
            lanelet_map=lanelet_map,
            nominal_speed_mps=0.10,
        ),
        route_id="route-visual-prefix-stabilize",
        current_lanelet_id="n1->n2",
        next_lanelet_ids=("n2->n3",),
    )
    ctx = replace(
        ctx,
        lane_observation=LaneObservation(
            detected_sides=("left", "right"),
            direct_error_m=0.18,
            quality=1.0,
            measurement_mode="two_line",
            direct_error_valid=True,
            control_policy_mode="ROUTE_TRACKING",
            planner_priority_active=True,
        ),
    )
    raw_path = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.02, 0.17, 0.0],
            [0.04, 0.18, 0.0],
            [0.12, 0.18, 0.0],
            [0.24, 0.12, 0.0],
            [0.40, 0.05, 0.0],
        ],
        dtype=float,
    )

    out = PathOptimizer().optimize(
        BehaviorPathPlan(
            timestamp=ctx.now_s,
            raw_path=raw_path,
            base_speed_profile=np.full(ctx.horizon_n, 0.08, dtype=float),
            scenario_name=ScenarioName.LANE_KEEP.value,
            valid=True,
            notes={"path_source": "visual_lane_waypoints"},
        ),
        ctx,
    )

    dx = float(out.target_path[1, 0] - pose.x)
    dy = float(out.target_path[1, 1] - pose.y)
    x_fwd = math.cos(pose.yaw) * dx + math.sin(pose.yaw) * dy
    y_left = math.sin(pose.yaw) * dx - math.cos(pose.yaw) * dy

    assert out.notes["visual_prefix_stabilization_applied"] is True
    assert x_fwd > 0.03
    assert abs(y_left) < 0.08
    assert _heading_tangent_error_deg(out.target_path, 1) < 45.0


def test_path_optimizer_keeps_route_precision_after_semantic_distance_jump() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    router = OsmRouteGraph(str(osm_path), step_m=0.05, start_lanelet_id="50")
    route = router.go_to("50", {"lanelet_id": "75"})
    _require_non_empty_route(route, lanelet_id="75")

    pose = (4.948804044528047, 5.899762255143094, -0.002610970551899235)
    matched_idx = 10
    matched_pose = tuple(float(v) for v in route.waypoints[matched_idx])
    horizon_n = 40
    dt = 0.05
    base_ctx = make_context(
        pose_x=pose[0],
        pose_y=pose[1],
        pose_yaw=pose[2],
        speed_mps=0.1,
        horizon_n=horizon_n,
        dt=dt,
        lanelet_map=None,
        nominal_speed_mps=0.1,
        current_lanelet_id="50",
        next_lanelet_ids=("55", "60", "70", "75"),
        route_waypoints=route.waypoints.tolist(),
        matched_idx=matched_idx,
        target_idx=12,
        matched_pose=matched_pose,
        maneuver_type="intersection_straight",
        map_match_error_m=0.0006916800758575313,
    )
    ctx = replace(
        base_ctx,
        route=replace(
            base_ctx.route,
            route_active=True,
            route_id="route-1",
            route_source="go_to",
            waypoint_mode_active=True,
            next_semantic_type="intersection",
            next_semantic_distance_m=0.7,
        ),
    )
    raw_path, bridge_meta = build_target_path_from_route(
        route_waypoints=route.waypoints,
        matched_idx=matched_idx,
        start_xy=pose[:2],
        start_yaw_rad=pose[2],
        matched_xy=matched_pose[:2],
        target_speed_mps=0.1,
        horizon_n=horizon_n,
        dt=dt,
        return_metadata=True,
    )

    base_step_arc = path_optimizer_module._infer_step_arc(
        base_speed_profile=np.full(horizon_n, 0.4, dtype=float),
        horizon_n=horizon_n,
        dt=dt,
        fallback_speed_mps=ctx.nominal_speed_mps,
    )
    step_arc, meta = path_optimizer_module._refine_step_arc_for_precision_route(
        step_arc=base_step_arc,
        ref_speed_mps=0.4,
        raw_path=raw_path,
        scenario_name=ScenarioName.LANE_KEEP.value,
        ctx=ctx,
    )

    assert meta["reason"] == "precision_waypoint_mode"
    assert step_arc == pytest.approx(0.0125, abs=1e-9)


def test_path_optimizer_keeps_waypoint_precision_without_semantic_distance() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    router = OsmRouteGraph(str(osm_path), step_m=0.05, start_lanelet_id="50")
    route = router.go_to("50", {"lanelet_id": "75"})
    _require_non_empty_route(route, lanelet_id="75")

    pose = (4.948804044528047, 5.899762255143094, -0.002610970551899235)
    matched_idx = 10
    matched_pose = tuple(float(v) for v in route.waypoints[matched_idx])
    horizon_n = 40
    dt = 0.05
    base_ctx = make_context(
        pose_x=pose[0],
        pose_y=pose[1],
        pose_yaw=pose[2],
        speed_mps=0.1,
        horizon_n=horizon_n,
        dt=dt,
        lanelet_map=None,
        nominal_speed_mps=0.1,
        current_lanelet_id="50",
        next_lanelet_ids=("55", "60", "70", "75"),
        route_waypoints=route.waypoints.tolist(),
        matched_idx=matched_idx,
        target_idx=12,
        matched_pose=matched_pose,
        maneuver_type="intersection_straight",
        map_match_error_m=0.0006916800758575313,
    )
    ctx = replace(
        base_ctx,
        route=replace(
            base_ctx.route,
            route_active=True,
            route_id="route-1",
            route_source="go_to",
            waypoint_mode_active=True,
            next_semantic_type=None,
            next_semantic_distance_m=None,
        ),
    )
    raw_path, bridge_meta = build_target_path_from_route(
        route_waypoints=route.waypoints,
        matched_idx=matched_idx,
        start_xy=pose[:2],
        start_yaw_rad=pose[2],
        matched_xy=matched_pose[:2],
        target_speed_mps=0.1,
        horizon_n=horizon_n,
        dt=dt,
        return_metadata=True,
    )

    base_step_arc = path_optimizer_module._infer_step_arc(
        base_speed_profile=np.full(horizon_n, 0.4, dtype=float),
        horizon_n=horizon_n,
        dt=dt,
        fallback_speed_mps=ctx.nominal_speed_mps,
    )
    step_arc, meta = path_optimizer_module._refine_step_arc_for_precision_route(
        step_arc=base_step_arc,
        ref_speed_mps=0.4,
        raw_path=raw_path,
        scenario_name=ScenarioName.LANE_KEEP.value,
        ctx=ctx,
    )

    assert meta["reason"] == "precision_waypoint_mode"
    assert meta["next_semantic_distance_m"] == pytest.approx(0.0, abs=1e-9)
    assert step_arc == pytest.approx(0.0125, abs=1e-9)


def _with_route_signature(ctx, *, route_id: str, current_lanelet_id: str, next_lanelet_ids: tuple[str, ...]):
    route = replace(
        ctx.route,
        route_active=True,
        route_id=route_id,
        current_lanelet_id=current_lanelet_id,
        next_lanelet_ids=next_lanelet_ids,
    )
    return replace(ctx, route=route)


def _load_sim_lanelet_map():
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    return load_lanelet2_osm(str(osm_path), step_m=0.05)
