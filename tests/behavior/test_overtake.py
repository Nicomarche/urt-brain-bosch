from __future__ import annotations

import math

import numpy as np
import pytest

import config
from src.behavior.planner import BehaviorPlanner
from src.behavior.scenarios.crosswalk import Crosswalk
from src.behavior.scenarios.intersection import Intersection
from src.behavior.scenarios.lane_keep import LaneKeep
from src.behavior.scenarios.overtake import Overtake
from src.behavior.scenarios.stop_sign import StopSign
from src.core.types.behavior import ScenarioName
from src.core.types.lidar import LidarObstacle
from src.core.types.perception import TrackedObject
from src.routing.lanelet.attributes import ATTR_NORMAL
from src.routing.lanelet.lanelet_map import from_track_graph
from tests.behavior.conftest import _build_track_graph, make_context


def _set_overtake_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OVERTAKE_LATERAL_OFFSET_M", 0.16, raising=False)
    monkeypatch.setattr(config, "OVERTAKE_MIN_OBSTACLE_CLEARANCE_M", 0.04, raising=False)
    monkeypatch.setattr(config, "OVERTAKE_MAX_LATERAL_OFFSET_M", 0.28, raising=False)
    monkeypatch.setattr(config, "OVERTAKE_SIDE_CLEAR_FORWARD_M", 1.25, raising=False)
    monkeypatch.setattr(config, "OVERTAKE_SIDE_CLEAR_HALF_WIDTH_M", 0.12, raising=False)
    monkeypatch.setattr(config, "OVERTAKE_RAW_LIDAR_FALLBACK_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "OVERTAKE_RAW_LIDAR_REQUIRES_HIGHWAY", True, raising=False)
    monkeypatch.setattr(config, "OVERTAKE_RAW_LIDAR_FORWARD_MAX_M", 0.70, raising=False)
    monkeypatch.setattr(config, "OVERTAKE_RAW_LIDAR_CONFIRM_TICKS", 2, raising=False)
    monkeypatch.setattr(config, "OVERTAKE_RAW_LIDAR_MIN_POINTS", 12, raising=False)
    monkeypatch.setattr(config, "OVERTAKE_RAW_LIDAR_MIN_WIDTH_DEG", 8.0, raising=False)
    monkeypatch.setattr(config, "OVERTAKE_RAW_LIDAR_MIN_CONFIDENCE", 0.55, raising=False)
    monkeypatch.setattr(config, "BEHAVIOR_VEHICLE_WIDTH_M", 0.19, raising=False)


def _straight_lanelet_map():
    return from_track_graph(
        _build_track_graph(
            [
                ("n1", 0.0, 0.0, ATTR_NORMAL),
                ("n2", 1.0, 0.0, ATTR_NORMAL),
                ("n3", 2.0, 0.0, ATTR_NORMAL),
                ("n4", 3.0, 0.0, ATTR_NORMAL),
            ]
        ),
        step_m=0.20,
    )


def _route_waypoints() -> tuple[tuple[float, float, float], ...]:
    return tuple((idx * 0.10, 0.0, 0.0) for idx in range(31))


def _overtake_context(*, lidar_obstacles=(), tracked_objects=(), sign_hints=()):
    lanelet_map = _straight_lanelet_map()
    return make_context(
        current_lanelet_id="n1->n2",
        next_lanelet_ids=("n2->n3", "n3->n4"),
        lanelet_map=lanelet_map,
        route_waypoints=_route_waypoints(),
        lidar_obstacles=tuple(lidar_obstacles),
        sign_hints=tuple(sign_hints),
        tracked_objects=tuple(tracked_objects),
        horizon_n=20,
        dt=0.05,
        nominal_speed_mps=0.25,
        max_speed_mps=0.40,
    )


def test_overtake_escape_keeps_shift_outside_lanelet_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_overtake_defaults(monkeypatch)
    front_obstacle = LidarObstacle(
        distance_min_m=0.55,
        x_m=0.55,
        y_m=0.0,
        point_count=5,
        source="cluster+ai",
        class_name="vehicle",
    )
    ctx = _overtake_context(lidar_obstacles=(front_obstacle,))

    out = BehaviorPlanner([Overtake(), LaneKeep()]).plan(ctx)

    assert out.scenario_name == ScenarioName.OVERTAKE.value
    assert out.notes["containment_mode"] == "overtake_escape"
    assert out.notes["overtake_corridor_escape"] is True
    assert out.notes["overtake_effective_lateral_offset_m"] == pytest.approx(0.23)
    assert np.max(out.target_path[1:, 1]) == pytest.approx(0.23, abs=0.02)
    assert out.notes["corridor_touches_bound"] is False


def test_overtake_priority_can_beat_intersection_but_not_hard_stops() -> None:
    assert Overtake.priority > Intersection.priority
    assert Overtake.priority < Crosswalk.priority
    assert Overtake.priority < StopSign.priority


def test_overtake_blocks_when_left_lidar_corridor_is_occupied(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_overtake_defaults(monkeypatch)
    front_obstacle = LidarObstacle(
        distance_min_m=0.55,
        x_m=0.55,
        y_m=0.0,
        point_count=5,
        source="cluster+ai",
        class_name="vehicle",
    )
    left_obstacle = LidarObstacle(distance_min_m=0.70, x_m=0.70, y_m=0.12, point_count=5)
    ctx = _overtake_context(lidar_obstacles=(front_obstacle, left_obstacle))

    assert Overtake().is_active(ctx) is False


def test_overtake_blocks_when_tracked_object_occupies_left_corridor(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_overtake_defaults(monkeypatch)
    front_obstacle = LidarObstacle(
        distance_min_m=0.55,
        x_m=0.55,
        y_m=0.0,
        point_count=5,
        source="cluster+ai",
        class_name="vehicle",
    )
    tracked_left_vehicle = TrackedObject(
        track_id=42,
        class_name="vehicle",
        confidence=0.9,
        position_world_xy=(0.70, 0.12),
        velocity_world_xy=(0.0, 0.0),
        age_frames=4,
    )
    ctx = _overtake_context(
        lidar_obstacles=(front_obstacle,),
        tracked_objects=(tracked_left_vehicle,),
    )

    assert Overtake().is_active(ctx) is False


def test_overtake_ignores_raw_lidar_front_cluster_without_semantic_object(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_overtake_defaults(monkeypatch)
    raw_cluster = LidarObstacle(
        distance_min_m=0.55,
        x_m=0.55,
        y_m=0.0,
        point_count=5,
        source="cluster",
        class_name="obstacle",
    )
    ctx = _overtake_context(lidar_obstacles=(raw_cluster,))

    assert Overtake().is_active(ctx) is False


def test_overtake_activates_for_tracked_vehicle_ahead(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_overtake_defaults(monkeypatch)
    tracked_vehicle = TrackedObject(
        track_id=7,
        class_name="vehicle",
        confidence=0.9,
        position_world_xy=(0.55, 0.0),
        velocity_world_xy=(0.0, 0.0),
        age_frames=4,
    )
    ctx = _overtake_context(tracked_objects=(tracked_vehicle,))

    assert Overtake().is_active(ctx) is True


def test_overtake_raw_lidar_fallback_requires_confirmed_highway_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_overtake_defaults(monkeypatch)
    raw_cluster = LidarObstacle(
        distance_min_m=0.24,
        x_m=0.26,
        y_m=0.01,
        width_rad=math.radians(34.5),
        point_count=69,
        confidence=1.0,
        source="cluster",
        class_name="obstacle",
    )
    ctx = _overtake_context(
        lidar_obstacles=(raw_cluster,),
        sign_hints=({"kind": "highway_entrance", "distance_m": 1.0},),
    )
    overtake = Overtake()

    assert overtake.is_active(ctx) is False
    assert overtake.is_active(ctx) is True


def test_overtake_ignores_lidar_self_return_even_on_highway(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_overtake_defaults(monkeypatch)
    self_return = LidarObstacle(
        distance_min_m=0.054,
        x_m=0.054,
        y_m=0.014,
        width_rad=math.radians(20.0),
        point_count=40,
        confidence=1.0,
        source="cluster",
        class_name="obstacle",
    )
    ctx = _overtake_context(
        lidar_obstacles=(self_return,),
        sign_hints=({"kind": "highway_entrance", "distance_m": 1.0},),
    )
    overtake = Overtake()

    assert overtake.is_active(ctx) is False
    assert overtake.is_active(ctx) is False
