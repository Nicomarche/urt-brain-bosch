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

import numpy as np
import pytest

from src.behavior.context import PlanningContext
from src.core.types.perception import LaneObservation, StoplineObservation, TrackedObject
from src.core.types.pose import Pose2D, PoseEstimate
from src.core.types.routing import RegulatoryElement, RouteContext
from src.routing.lanelet.from_graphml import (
    ATTR_CROSSWALK,
    ATTR_HIGHWAY_LEFT,
    ATTR_INTERSECTION,
    ATTR_NORMAL,
    ATTR_ROUNDABOUT,
    ATTR_STOPLINE,
    TrackGraph,
    TrackNode,
)
from src.routing.lanelet.lanelet_map import from_track_graph
from src.routing.lanelet.semantics import TrackSemantics


def _build_track_graph(node_specs: list[tuple[str, float, float, int]]) -> TrackGraph:
    """Construye un TrackGraph mínimo en memoria desde una lista de
    (node_id, x, y, attribute). Los edges van consecutivos (1->2->3...).
    """
    g = TrackGraph.__new__(TrackGraph)
    g.step_m = 0.10
    g.nodes = {
        nid: TrackNode(nid, x, y, attr, False) for (nid, x, y, attr) in node_specs
    }
    g.adj = defaultdict(list)
    for i in range(len(node_specs) - 1):
        g.adj[node_specs[i][0]].append(node_specs[i + 1][0])
    g.edge_lengths = {}
    g.edge_ids = {}
    g.reference_node_ids = []
    g.reference_path = None
    g.ordered_nodes = list(g.nodes.values())
    g.waypoints = np.empty((0, 3))
    g.wp_node_attrs = np.empty(0, dtype=int)
    g.semantics = TrackSemantics(None)
    g.map_metadata = {}
    return g


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
    regulatory_ahead: tuple[RegulatoryElement, ...] = (),
    tracked_objects: tuple[TrackedObject, ...] = (),
    sign_hints: tuple[dict, ...] = (),
    nominal_speed_mps: float = 0.50,
    max_speed_mps: float = 1.00,
    horizon_n: int = 10,
    dt: float = 0.05,
    maneuver_type: str = "none",
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
        current_lanelet_id=current_lanelet_id,
        next_lanelet_ids=next_lanelet_ids,
        regulatory_ahead=regulatory_ahead,
        maneuver_type=maneuver_type,
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
