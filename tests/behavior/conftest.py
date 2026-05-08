# tests/behavior/conftest.py
#
# Helpers compartidos entre tests de behavior. Incluye:
#   - `straight_lanelet_map`: LaneletMap de 4 lanelets en línea recta
#     con un STOPLINE en el medio. Útil para tests de overlay y
#     scenarios.
#   - `make_context`: factory para PlanningContext con defaults
#     razonables.

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from src.behavior.context import PlanningContext
from src.core.types.perception import LaneObservation, StoplineObservation, TrackedObject
from src.core.types.pose import Pose2D, PoseEstimate
from src.core.types.routing import RegulatoryElement, RouteContext
from src.routing.lanelet.attributes import (
    ATTR_CROSSWALK,
    ATTR_HIGHWAY_LEFT,
    ATTR_INTERSECTION,
    ATTR_NORMAL,
    ATTR_ROUNDABOUT,
    ATTR_STOPLINE,
)
from src.routing.lanelet.lanelet_map import from_track_graph


@dataclass(frozen=True)
class TrackNode:
    node_id: str
    x: float
    y: float
    attribute: int = ATTR_NORMAL
    is_start: bool = False


def _build_track_graph(node_specs: list[tuple[str, float, float, int]]):
    """Construye un backend mínimo en memoria compatible con `from_track_graph`.

    La topología es una cadena simple (1->2->3...). Sólo poblamos los campos
    que consume la factory legacy de LaneletMap.
    """
    return SimpleNamespace(
        step_m=0.10,
        nodes={
        nid: TrackNode(nid, x, y, attr, False) for (nid, x, y, attr) in node_specs
        },
        adj=defaultdict(
            list,
            {
                node_specs[i][0]: [node_specs[i + 1][0]]
                for i in range(len(node_specs) - 1)
            },
        ),
        edge_lengths={},
        edge_ids={},
        reference_node_ids=[],
        reference_path=None,
        ordered_nodes=[],
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


@pytest.fixture
def straight_lanelet_map():
    """4 nodos en línea recta. Lanelet `n3->n4` tiene atributo STOPLINE."""
    g = _build_track_graph([
        ("n1", 0.0, 0.0, ATTR_NORMAL),
        ("n2", 1.0, 0.0, ATTR_NORMAL),
        ("n3", 2.0, 0.0, ATTR_NORMAL),
        ("n4", 3.0, 0.0, ATTR_STOPLINE),
        ("n5", 4.0, 0.0, ATTR_NORMAL),
    ])
    return from_track_graph(g, step_m=0.20)


@pytest.fixture
def intersection_lanelet_map():
    """5 nodos con `n3->n4` siendo INTERSECTION."""
    g = _build_track_graph([
        ("a1", 0.0, 0.0, ATTR_NORMAL),
        ("a2", 1.0, 0.0, ATTR_NORMAL),
        ("a3", 2.0, 0.0, ATTR_INTERSECTION),
        ("a4", 3.0, 0.0, ATTR_NORMAL),
    ])
    return from_track_graph(g, step_m=0.20)


@pytest.fixture
def crosswalk_lanelet_map():
    g = _build_track_graph([
        ("c1", 0.0, 0.0, ATTR_NORMAL),
        ("c2", 1.0, 0.0, ATTR_CROSSWALK),
        ("c3", 2.0, 0.0, ATTR_NORMAL),
    ])
    return from_track_graph(g, step_m=0.20)


@pytest.fixture
def highway_lanelet_map():
    g = _build_track_graph([
        ("h1", 0.0, 0.0, ATTR_NORMAL),
        ("h2", 1.0, 0.0, ATTR_HIGHWAY_LEFT),
        ("h3", 2.0, 0.0, ATTR_HIGHWAY_LEFT),
        ("h4", 3.0, 0.0, ATTR_NORMAL),
    ])
    return from_track_graph(g, step_m=0.20)


@pytest.fixture
def roundabout_lanelet_map():
    g = _build_track_graph([
        ("r1", 0.0, 0.0, ATTR_NORMAL),
        ("r2", 1.0, 0.0, ATTR_ROUNDABOUT),
        ("r3", 2.0, 0.0, ATTR_ROUNDABOUT),
        ("r4", 3.0, 0.0, ATTR_NORMAL),
    ])
    return from_track_graph(g, step_m=0.20)


def make_context(
    *,
    pose_x: float = 0.0,
    pose_y: float = 0.0,
    pose_yaw: float = 0.0,
    speed_mps: float = 0.0,
    current_lanelet_id: str | None = None,
    next_lanelet_ids: tuple[str, ...] = (),
    lanelet_map=None,
    route_waypoints: list[list[float]] | tuple[tuple[float, float, float], ...] = (),
    route_points: list[dict[str, float]] | tuple[dict[str, float], ...] = (),
    regulatory_ahead: tuple[RegulatoryElement, ...] = (),
    tracked_objects: tuple[TrackedObject, ...] = (),
    sign_hints: tuple[dict, ...] = (),
    nominal_speed_mps: float = 0.50,
    max_speed_mps: float = 1.00,
    horizon_n: int = 10,
    dt: float = 0.05,
    maneuver_type: str = "none",
    matched_idx: int = 0,
    target_idx: int = 0,
    matched_pose: tuple[float, float, float] | None = None,
    map_match_error_m: float = 0.0,
) -> PlanningContext:
    """Factory para PlanningContext con defaults razonables."""
    pose = PoseEstimate(
        timestamp=0.0,
        fused_pose=Pose2D(x=pose_x, y=pose_y, yaw=pose_yaw),
        raw_pose=Pose2D(x=pose_x, y=pose_y, yaw=pose_yaw),
        speed_mps=speed_mps,
        yaw_rad=pose_yaw,
        localization_mode="ekf7",
    )
    route = RouteContext(
        timestamp=0.0,
        route_active=current_lanelet_id is not None,
        route_points=list(route_points or []),
        route_waypoints=[list(item) for item in (route_waypoints or ())],
        current_lanelet_id=current_lanelet_id,
        next_lanelet_ids=next_lanelet_ids,
        regulatory_ahead=regulatory_ahead,
        maneuver_type=maneuver_type,
        matched_idx=int(matched_idx),
        target_idx=int(target_idx),
        map_match_error_m=float(map_match_error_m),
        matched_pose=Pose2D(
            x=float((matched_pose or (pose_x, pose_y, pose_yaw))[0]),
            y=float((matched_pose or (pose_x, pose_y, pose_yaw))[1]),
            yaw=float((matched_pose or (pose_x, pose_y, pose_yaw))[2]),
        ),
    )
    return PlanningContext(
        now_s=0.0,
        dt=dt,
        horizon_n=horizon_n,
        nominal_speed_mps=nominal_speed_mps,
        max_speed_mps=max_speed_mps,
        pose=pose,
        route=route,
        lane_observation=LaneObservation(),
        stopline_observation=StoplineObservation(),
        tracked_objects=tracked_objects,
        lanelet_map=lanelet_map,
        sign_hints=sign_hints,
    )
