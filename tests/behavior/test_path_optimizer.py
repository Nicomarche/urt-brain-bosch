from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest

from src.behavior.path_optimizer import PathOptimizer
from src.behavior.trajectory_builder import build_target_path, build_target_path_from_route
from src.core.types.behavior import BehaviorPathPlan, ScenarioName
from src.routing.lanelet.from_osm import load_lanelet2_osm
from src.routing.lanelet.osm_router import OsmRouteGraph
from tests.behavior.conftest import make_context


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
    assert horizon_distance_m < 0.25
    assert np.max(np.abs(out.target_path[:12, 1] - out.target_path[0, 1])) < 0.01
    assert out.target_path[-1, 1] > pose[1] - 0.02


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

    out = PathOptimizer().optimize(
        BehaviorPathPlan(
            timestamp=ctx.now_s,
            raw_path=raw_path,
            base_speed_profile=np.full(horizon_n, 0.4, dtype=float),
            scenario_name=ScenarioName.LANE_KEEP.value,
            valid=True,
            notes=bridge_meta,
        ),
        ctx,
    )

    horizon_distance_m = float(np.linalg.norm(out.target_path[-1, :2] - out.target_path[0, :2]))
    assert horizon_distance_m < 0.25
    assert abs(float(out.target_path[-1, 2])) < 0.05
    assert np.max(np.abs(out.target_path[:12, 1] - out.target_path[0, 1])) < 0.02


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
