from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.behavior.velocity_planner import (
    BehaviorVelocityPlanner,
    _BEHAVIOR_CONTAINMENT_CRAWL_SPEED_MPS,
    _BEHAVIOR_CONTAINMENT_RECOVERY_SPEED_MPS,
    _BEHAVIOR_MIN_SPEED_MPS,
)
from src.core.types.behavior import BehaviorPathPlan, ScenarioName
from src.core.types.lidar import LidarObstacle
from src.core.types.perception import LaneObservation, TrackedObject
from src.core.types.routing import RegulatoryElement
from tests.behavior.conftest import make_context


def test_velocity_planner_ignores_stopline_for_stop_behavior() -> None:
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

    assert out.stop_required is False
    assert out.speed_profile[-1] == pytest.approx(1.0, abs=1e-6)


def test_velocity_planner_allows_low_speed_for_crosswalk() -> None:
    ctx = make_context(horizon_n=10, dt=0.1, nominal_speed_mps=0.10, max_speed_mps=0.40)
    target_path = np.column_stack([np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)])
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=target_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.10, dtype=float),
        scenario_name=ScenarioName.CROSSWALK.value,
        valid=True,
        notes={"min_moving_speed_mps": 0.10},
    )

    out = BehaviorVelocityPlanner().build_output(
        path_plan=plan,
        target_path=target_path,
        drivable_left_bound=np.zeros((11, 2), dtype=float),
        drivable_right_bound=np.zeros((11, 2), dtype=float),
        ctx=ctx,
    )

    assert np.max(out.speed_profile) == pytest.approx(0.10, abs=1e-6)
    assert out.min_moving_speed_mps == pytest.approx(0.10)


def test_velocity_planner_raises_highway_to_highway_minimum() -> None:
    ctx = make_context(horizon_n=10, dt=0.1, nominal_speed_mps=0.30, max_speed_mps=1.00)
    target_path = np.column_stack([np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)])
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=target_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.30, dtype=float),
        scenario_name=ScenarioName.HIGHWAY.value,
        valid=True,
        notes={"min_moving_speed_mps": 0.40},
    )

    out = BehaviorVelocityPlanner().build_output(
        path_plan=plan,
        target_path=target_path,
        drivable_left_bound=np.zeros((11, 2), dtype=float),
        drivable_right_bound=np.zeros((11, 2), dtype=float),
        ctx=ctx,
    )

    assert np.min(out.speed_profile) == pytest.approx(0.40, abs=1e-6)
    assert out.min_moving_speed_mps == pytest.approx(0.40)


def test_lidar_front_obstacle_slowdown_overrides_highway_minimum_without_mpc_myopia() -> None:
    ctx = make_context(
        horizon_n=10,
        dt=0.1,
        nominal_speed_mps=0.40,
        max_speed_mps=0.40,
        lidar_obstacles=(LidarObstacle(distance_min_m=0.24, x_m=0.26, y_m=0.01, point_count=69),),
    )
    target_path = np.column_stack([np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)])
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=target_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.40, dtype=float),
        scenario_name=ScenarioName.HIGHWAY.value,
        valid=True,
        notes={"min_moving_speed_mps": 0.40},
    )

    out = BehaviorVelocityPlanner().build_output(
        path_plan=plan,
        target_path=target_path,
        drivable_left_bound=np.zeros((11, 2), dtype=float),
        drivable_right_bound=np.zeros((11, 2), dtype=float),
        ctx=ctx,
    )

    assert np.max(out.speed_profile) == pytest.approx(0.20, abs=1e-6)
    assert out.min_moving_speed_mps == pytest.approx(0.20)
    assert any(
        note.get("mode") == "lidar_obstacle_close_slow"
        and note.get("requested_cap_mps") == pytest.approx(0.20)
        and note.get("motion_floor_mps") == pytest.approx(0.20)
        for note in out.notes["velocity_modules"]
    )


def test_lidar_self_return_does_not_force_close_slow() -> None:
    ctx = make_context(
        horizon_n=10,
        dt=0.1,
        nominal_speed_mps=0.40,
        max_speed_mps=0.40,
        lidar_obstacles=(LidarObstacle(distance_min_m=0.054, x_m=0.054, y_m=0.014, point_count=40),),
    )
    target_path = np.column_stack([np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)])
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=target_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.40, dtype=float),
        scenario_name=ScenarioName.HIGHWAY.value,
        valid=True,
        notes={"min_moving_speed_mps": 0.40},
    )

    out = BehaviorVelocityPlanner().build_output(
        path_plan=plan,
        target_path=target_path,
        drivable_left_bound=np.zeros((11, 2), dtype=float),
        drivable_right_bound=np.zeros((11, 2), dtype=float),
        ctx=ctx,
    )

    assert np.max(out.speed_profile) == pytest.approx(0.40, abs=1e-6)
    assert not any(
        note.get("kind") == "lidar_obstacle"
        for note in out.notes["velocity_modules"]
    )


def test_velocity_planner_slows_instead_of_stopping_for_lidar_obstacle_in_corridor() -> None:
    ctx = make_context(
        horizon_n=10,
        dt=0.1,
        nominal_speed_mps=0.4,
        max_speed_mps=1.0,
        lidar_obstacles=(LidarObstacle(distance_min_m=0.50, x_m=0.50, y_m=0.0, point_count=4),),
    )
    target_path = np.column_stack([np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)])
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=target_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.4, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )
    out = BehaviorVelocityPlanner().build_output(
        path_plan=plan,
        target_path=target_path,
        drivable_left_bound=np.zeros((11, 2), dtype=float),
        drivable_right_bound=np.zeros((11, 2), dtype=float),
        ctx=ctx,
    )
    assert out.stop_required is False
    assert not np.allclose(out.speed_profile, 0.0)
    assert np.max(out.speed_profile) <= 0.20
    assert any(note.get("mode") == "lidar_obstacle_slow" for note in out.notes["velocity_modules"])


def test_velocity_planner_slows_for_lidar_obstacle() -> None:
    ctx = make_context(
        horizon_n=10,
        dt=0.1,
        nominal_speed_mps=0.6,
        max_speed_mps=1.0,
        lidar_obstacles=(LidarObstacle(distance_min_m=0.70, x_m=0.70, y_m=0.02, point_count=4),),
    )
    target_path = np.column_stack([np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)])
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=target_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.6, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )
    out = BehaviorVelocityPlanner().build_output(
        path_plan=plan,
        target_path=target_path,
        drivable_left_bound=np.zeros((11, 2), dtype=float),
        drivable_right_bound=np.zeros((11, 2), dtype=float),
        ctx=ctx,
    )
    assert out.stop_required is False
    assert np.max(out.speed_profile) <= 0.20
    assert any(note.get("mode") == "lidar_obstacle_slow" for note in out.notes["velocity_modules"])


def test_velocity_planner_acc_with_track_ahead() -> None:
    track = TrackedObject(
        track_id=7,
        class_name="vehicle",
        confidence=0.9,
        position_world_xy=(0.75, 0.0),
        velocity_world_xy=(0.15, 0.0),
        age_frames=4,
    )
    ctx = make_context(horizon_n=10, dt=0.1, nominal_speed_mps=0.6, max_speed_mps=1.0, tracked_objects=(track,))
    target_path = np.column_stack([np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)])
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=target_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.6, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )
    out = BehaviorVelocityPlanner().build_output(
        path_plan=plan,
        target_path=target_path,
        drivable_left_bound=np.zeros((11, 2), dtype=float),
        drivable_right_bound=np.zeros((11, 2), dtype=float),
        ctx=ctx,
    )
    assert np.max(out.speed_profile) <= 0.20
    assert any(note.get("mode") == "acc_target" and note.get("track_id") == 7.0 for note in out.notes["velocity_modules"])


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


def test_velocity_planner_slows_nominal_bfmc_speed_on_tight_curve() -> None:
    theta = np.linspace(0.0, np.pi / 2.0, 21)
    radius = 0.30
    target_path = np.column_stack(
        [
            radius * np.cos(theta),
            radius * np.sin(theta),
            theta + np.pi / 2.0,
        ]
    )
    ctx = make_context(horizon_n=20, dt=0.1, nominal_speed_mps=0.25, max_speed_mps=0.40)
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=target_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.25, dtype=float),
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

    assert np.min(out.speed_profile) == pytest.approx(0.20, abs=1e-6)
    assert any(note.get("kind") == "curvature_constraint" for note in out.notes.get("velocity_modules", []))


def test_velocity_planner_caps_speed_for_lane_containment_warning() -> None:
    ctx = make_context(horizon_n=20, dt=0.1, nominal_speed_mps=0.25, max_speed_mps=0.40)
    ctx = replace(
        ctx,
        lane_observation=LaneObservation(
            detected_sides=("left", "right"),
            quality=0.92,
            measurement_mode="two_line",
            direct_error_valid=True,
            direct_error_m=0.055,
            control_policy_mode="MAP_ONLY",
        ),
    )
    target_path = np.column_stack([np.linspace(0.0, 1.0, 21), np.zeros(21), np.zeros(21)])
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=target_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.25, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )

    out = BehaviorVelocityPlanner().build_output(
        path_plan=plan,
        target_path=target_path,
        drivable_left_bound=np.zeros((21, 2), dtype=float),
        drivable_right_bound=np.zeros((21, 2), dtype=float),
        optimizer_notes={"ego_corridor_error_m": 0.03},
        ctx=ctx,
    )

    assert np.max(out.speed_profile) == pytest.approx(float(_BEHAVIOR_MIN_SPEED_MPS), abs=1e-6)
    assert any(
        note.get("kind") == "lane_containment" and note.get("mode") == "warn"
        for note in out.notes.get("velocity_modules", [])
    )


def test_velocity_planner_crawls_when_corridor_touches_bound() -> None:
    ctx = make_context(horizon_n=20, dt=0.1, nominal_speed_mps=0.25, max_speed_mps=0.40)
    target_path = np.column_stack([np.linspace(0.0, 1.0, 21), np.zeros(21), np.zeros(21)])
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=target_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.25, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )

    out = BehaviorVelocityPlanner().build_output(
        path_plan=plan,
        target_path=target_path,
        drivable_left_bound=np.zeros((21, 2), dtype=float),
        drivable_right_bound=np.zeros((21, 2), dtype=float),
        optimizer_notes={
            "ego_corridor_error_m": 0.03,
            "corridor_touches_bound": True,
        },
        ctx=ctx,
    )

    expected_crawl = max(float(_BEHAVIOR_CONTAINMENT_CRAWL_SPEED_MPS), float(_BEHAVIOR_MIN_SPEED_MPS))
    assert np.max(out.speed_profile) == pytest.approx(expected_crawl, abs=1e-6)
    assert any(
        note.get("kind") == "lane_containment" and note.get("mode") == "crawl"
        for note in out.notes.get("velocity_modules", [])
    )


def test_velocity_planner_keeps_steering_for_future_infeasible_containment() -> None:
    ctx = make_context(horizon_n=20, dt=0.1, nominal_speed_mps=0.25, max_speed_mps=0.40)
    target_path = np.column_stack([np.linspace(0.0, 1.0, 21), np.zeros(21), np.zeros(21)])
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=target_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.25, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )

    out = BehaviorVelocityPlanner().build_output(
        path_plan=plan,
        target_path=target_path,
        drivable_left_bound=np.zeros((21, 2), dtype=float),
        drivable_right_bound=np.zeros((21, 2), dtype=float),
        optimizer_notes={
            "ego_corridor_error_m": 0.09,
            "used_prev_safe_path": True,
            "containment_infeasible_ticks": 4,
            "containment_stop_after_ticks": 4,
            "first_infeasible_index": 3,
        },
        ctx=ctx,
    )

    assert out.stop_required is False
    assert not np.allclose(out.speed_profile, 0.0)
    assert any(
        note.get("kind") == "lane_containment" and note.get("mode") == "future_horizon_crawl"
        for note in out.notes.get("velocity_modules", [])
    )


def test_velocity_planner_keeps_steering_for_near_infeasible_when_ego_aligned() -> None:
    ctx = make_context(horizon_n=20, dt=0.1, nominal_speed_mps=0.25, max_speed_mps=0.40)
    target_path = np.column_stack([np.linspace(0.0, 1.0, 21), np.zeros(21), np.zeros(21)])
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=target_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.25, dtype=float),
        scenario_name=ScenarioName.INTERSECTION.value,
        valid=True,
    )

    out = BehaviorVelocityPlanner().build_output(
        path_plan=plan,
        target_path=target_path,
        drivable_left_bound=np.zeros((21, 2), dtype=float),
        drivable_right_bound=np.zeros((21, 2), dtype=float),
        optimizer_notes={
            "ego_corridor_error_m": 0.003,
            "corridor_touches_bound": True,
            "containment_infeasible_ticks": 1300,
            "containment_stop_after_ticks": 4,
            "first_infeasible_index": 1,
        },
        ctx=ctx,
    )

    assert out.stop_required is False
    assert not np.allclose(out.speed_profile, 0.0)
    assert any(
        note.get("kind") == "lane_containment" and note.get("mode") == "near_horizon_crawl"
        for note in out.notes.get("velocity_modules", [])
    )


def _build_containment_plan(ctx):
    target_path = np.column_stack([np.linspace(0.0, 1.0, 21), np.zeros(21), np.zeros(21)])
    plan = BehaviorPathPlan(
        timestamp=ctx.now_s,
        raw_path=target_path,
        base_speed_profile=np.full(ctx.horizon_n, 0.25, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )
    return plan, target_path


def test_velocity_planner_escapes_crawl_after_stuck_ticks() -> None:
    # Reproduce el patrón del run_20260508_214446: el robot queda atascado en
    # crawl con error lateral grande sin decrecer. Tras N ticks sin recuperar
    # el cap sube a recovery_speed para que el robot pueda maniobrar.
    from src.behavior.velocity_planner import _BEHAVIOR_CONTAINMENT_STUCK_TICKS
    ctx = make_context(horizon_n=20, dt=0.1, nominal_speed_mps=0.25, max_speed_mps=0.40)
    plan, target_path = _build_containment_plan(ctx)
    planner = BehaviorVelocityPlanner()
    optimizer_notes = {
        "ego_corridor_error_m": 0.36,
        "corridor_touches_bound": True,
    }

    last_out = None
    for _ in range(int(_BEHAVIOR_CONTAINMENT_STUCK_TICKS) + 1):
        last_out = planner.build_output(
            path_plan=plan,
            target_path=target_path,
            drivable_left_bound=np.zeros((21, 2), dtype=float),
            drivable_right_bound=np.zeros((21, 2), dtype=float),
            optimizer_notes=optimizer_notes,
            ctx=ctx,
        )

    expected_recovery = max(float(_BEHAVIOR_CONTAINMENT_RECOVERY_SPEED_MPS), float(_BEHAVIOR_MIN_SPEED_MPS))
    assert np.max(last_out.speed_profile) == pytest.approx(expected_recovery, abs=1e-6)
    assert any(
        note.get("kind") == "lane_containment" and note.get("mode") == "stuck_recovery"
        for note in last_out.notes.get("velocity_modules", [])
    )


def test_velocity_planner_resets_stuck_when_error_decreases() -> None:
    # Si el error decrece al menos 5 mm/tick, no se considera atascado y
    # el cap se mantiene en crawl normal (4 cm/s) — no escala a recovery.
    from src.behavior.velocity_planner import _BEHAVIOR_CONTAINMENT_STUCK_TICKS
    ctx = make_context(horizon_n=20, dt=0.1, nominal_speed_mps=0.25, max_speed_mps=0.40)
    plan, target_path = _build_containment_plan(ctx)
    planner = BehaviorVelocityPlanner()

    # Error decrece monótonamente (1 cm por tick).
    last_out = None
    error = 0.36
    for _ in range(int(_BEHAVIOR_CONTAINMENT_STUCK_TICKS) + 5):
        last_out = planner.build_output(
            path_plan=plan,
            target_path=target_path,
            drivable_left_bound=np.zeros((21, 2), dtype=float),
            drivable_right_bound=np.zeros((21, 2), dtype=float),
            optimizer_notes={
                "ego_corridor_error_m": error,
                "corridor_touches_bound": True,
            },
            ctx=ctx,
        )
        error = max(error - 0.01, 0.08)

    expected_crawl = max(float(_BEHAVIOR_CONTAINMENT_CRAWL_SPEED_MPS), float(_BEHAVIOR_MIN_SPEED_MPS))
    assert np.max(last_out.speed_profile) == pytest.approx(expected_crawl, abs=1e-6)
    assert any(
        note.get("kind") == "lane_containment" and note.get("mode") == "crawl"
        for note in last_out.notes.get("velocity_modules", [])
    )


def test_velocity_planner_exits_recovery_when_back_in_corridor() -> None:
    # Tras escalar a recovery, una vez que el error baja del threshold de
    # crawl (0.07 m) el rule vuelve a IDLE: ya no aparece la nota.
    from src.behavior.velocity_planner import _BEHAVIOR_CONTAINMENT_STUCK_TICKS
    ctx = make_context(horizon_n=20, dt=0.1, nominal_speed_mps=0.25, max_speed_mps=0.40)
    plan, target_path = _build_containment_plan(ctx)
    planner = BehaviorVelocityPlanner()

    # Atascar primero.
    for _ in range(int(_BEHAVIOR_CONTAINMENT_STUCK_TICKS) + 1):
        planner.build_output(
            path_plan=plan,
            target_path=target_path,
            drivable_left_bound=np.zeros((21, 2), dtype=float),
            drivable_right_bound=np.zeros((21, 2), dtype=float),
            optimizer_notes={
                "ego_corridor_error_m": 0.36,
                "corridor_touches_bound": True,
            },
            ctx=ctx,
        )

    # Ahora el robot ya bajó del threshold y no toca el corridor.
    out = planner.build_output(
        path_plan=plan,
        target_path=target_path,
        drivable_left_bound=np.zeros((21, 2), dtype=float),
        drivable_right_bound=np.zeros((21, 2), dtype=float),
        optimizer_notes={"ego_corridor_error_m": 0.02},
        ctx=ctx,
    )

    assert np.max(out.speed_profile) == pytest.approx(0.25, abs=1e-6)
    assert not any(
        note.get("kind") == "lane_containment"
        for note in out.notes.get("velocity_modules", [])
    )
