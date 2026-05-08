from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from src.core.types.pose import Pose2D
from src.routing.lanelet.attributes import ATTR_NORMAL
from src.routing.lanelet.lanelet_map import from_track_graph
from src.routing.navigation_planner_thread import threadNavigationPlanner
from src.routing.route_planner import PathUpdate


@dataclass(frozen=True)
class TrackNode:
    node_id: str
    x: float
    y: float
    attribute: int = ATTR_NORMAL
    is_start: bool = False


def _merge_lanelet_map():
    nodes = {
        "pose_start": TrackNode("pose_start", -1.0, 0.2),
        "route_start": TrackNode("route_start", -1.0, -0.2),
        "merge": TrackNode("merge", 0.0, 0.0),
        "shared": TrackNode("shared", 1.0, 0.0),
        "tail": TrackNode("tail", 2.0, 0.0),
    }
    graph = SimpleNamespace(
        step_m=0.05,
        nodes=nodes,
        adj=defaultdict(
            list,
            {
                "pose_start": ["merge"],
                "route_start": ["merge"],
                "merge": ["shared"],
                "shared": ["tail"],
            },
        ),
        edge_lengths={},
        edge_ids={},
        reference_node_ids=[],
        reference_path=None,
        ordered_nodes=list(nodes.values()),
        waypoints=np.empty((0, 3)),
        wp_node_attrs=np.empty(0, dtype=int),
        semantics=SimpleNamespace(
            loaded=False,
            controls=[],
            intersections=[],
            crosswalks=[],
            parking_spots=[],
        ),
        map_metadata={},
    )
    return from_track_graph(graph, step_m=0.05)


def _make_thread(*, lanelet_map, route_lanelet_ids: list[str], pose: Pose2D):
    thread = threadNavigationPlanner.__new__(threadNavigationPlanner)
    thread._lanelet_map = lanelet_map
    thread._last_pose = pose
    thread._last_lanelet_id = None
    thread.logging = None
    thread._path_manager = SimpleNamespace(
        active_route=SimpleNamespace(
            waypoints=np.zeros((len(route_lanelet_ids), 3), dtype=float),
            wp_node_ids=list(route_lanelet_ids),
            closed_loop=False,
        )
    )
    return thread


def _make_path_update(
    *,
    matched_x: float,
    matched_y: float,
    matched_yaw: float,
    matched_idx: int,
    target_idx: int,
    current_node_id: str,
    upcoming_node_id: str,
    map_match_error_m: float = 0.10,
) -> PathUpdate:
    return PathUpdate(
        route_active=True,
        route_id="route-1",
        route_points=[],
        route_waypoints=[],
        destination_node_id="tail",
        destination_label="Tail",
        route_queue=[],
        current_node_id=current_node_id,
        current_node_attr=ATTR_NORMAL,
        upcoming_node_id=upcoming_node_id,
        upcoming_node_attr=ATTR_NORMAL,
        maneuver_type="none",
        route_progress=0.0,
        route_completed=False,
        waypoint_mode_active=False,
        matched_idx=matched_idx,
        target_idx=target_idx,
        matched_x=matched_x,
        matched_y=matched_y,
        matched_yaw=matched_yaw,
        path_psi=matched_yaw,
        path_kappa=0.0,
        path_heading_change_rad=0.0,
        error_m=0.0,
        heading_rad=0.0,
        map_match_error_m=map_match_error_m,
        remaining_distance_m=1.0,
        replans=1,
        route_source="go_to",
        destination_point={"x": 2.0, "y": 0.0},
        next_semantic_id=None,
        next_semantic_type=None,
        next_semantic_label=None,
        next_semantic_distance_m=None,
        expected_control_type=None,
        current_zone_ids=[],
        current_zone_types=[],
        map_metadata={},
        available_destinations=[],
    )


def test_resolve_lanelet_context_holds_pose_lanelet_when_route_and_pose_share_successor() -> None:
    lanelet_map = _merge_lanelet_map()
    pose_yaw = -math.atan2(0.2, 1.0)
    route_yaw = math.atan2(0.2, 1.0)
    thread = _make_thread(
        lanelet_map=lanelet_map,
        route_lanelet_ids=["route_start->merge", "merge->shared", "shared->tail"],
        pose=Pose2D(x=-0.6, y=0.12, yaw=pose_yaw),
    )
    path_update = _make_path_update(
        matched_x=-0.6,
        matched_y=-0.12,
        matched_yaw=route_yaw,
        matched_idx=0,
        target_idx=1,
        current_node_id="route_start->merge",
        upcoming_node_id="merge->shared",
    )

    current_lanelet_id, next_lanelet_ids, _regulatory = thread._resolve_lanelet_context(path_update)

    assert current_lanelet_id == "pose_start->merge"
    assert next_lanelet_ids == ("merge->shared", "shared->tail")


def test_resolve_lanelet_context_keeps_predecessor_pose_lanelet_before_route_lanelet() -> None:
    lanelet_map = _merge_lanelet_map()
    pose_yaw = -math.atan2(0.2, 1.0)
    thread = _make_thread(
        lanelet_map=lanelet_map,
        route_lanelet_ids=["merge->shared", "shared->tail"],
        pose=Pose2D(x=-0.2, y=0.04, yaw=pose_yaw),
    )
    path_update = _make_path_update(
        matched_x=0.2,
        matched_y=0.0,
        matched_yaw=0.0,
        matched_idx=0,
        target_idx=1,
        current_node_id="merge->shared",
        upcoming_node_id="shared->tail",
        map_match_error_m=0.04,
    )

    current_lanelet_id, next_lanelet_ids, _regulatory = thread._resolve_lanelet_context(path_update)

    assert current_lanelet_id == "pose_start->merge"
    assert next_lanelet_ids[0] == "merge->shared"
