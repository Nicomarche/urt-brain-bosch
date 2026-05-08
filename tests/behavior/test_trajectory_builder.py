# tests/behavior/test_trajectory_builder.py
#
# Tests del trajectory_builder. Validamos:
#   1. La salida tiene shape (N+1, 3) según el contrato.
#   2. El primer waypoint queda cerca del ego (start_xy), no del nodo
#      upstream de la lanelet.
#   3. El espaciado entre waypoints corresponde al `target_speed * dt`.
#   4. Las headings (psi) son consistentes con la dirección de avance
#      (atan2 del segmento adyacente).
#   5. Cuando el ego está al final de un grafo terminal, los waypoints
#      sobrantes se replican (no crashea).

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.behavior.trajectory_builder import build_target_path, build_target_path_from_route
from src.routing.lanelet.attributes import ATTR_NORMAL
from src.routing.lanelet.lanelet_map import from_track_graph
from src.routing.lanelet.osm_router import OsmRouteGraph


@dataclass(frozen=True)
class TrackNode:
    node_id: str
    x: float
    y: float
    attribute: int = ATTR_NORMAL
    is_start: bool = False


def _branched_lanelet_map():
    g = SimpleNamespace()
    g.step_m = 0.10
    g.nodes = {
        "s": TrackNode("s", 0.0, 0.0, 0, False),
        "a": TrackNode("a", 1.0, 0.0, 0, False),
        "b": TrackNode("b", 2.0, 0.0, 0, False),
        "c": TrackNode("c", 3.0, 1.0, 0, False),
        "d": TrackNode("d", 3.0, -1.0, 0, False),
        "e": TrackNode("e", 4.0, 1.0, 0, False),
        "f": TrackNode("f", 4.0, -1.0, 0, False),
    }
    g.adj = defaultdict(list, {
        "s": ["a"],
        "a": ["b"],
        # Ordenamos `c` antes que `d` para que el fallback "primera sucesora"
        # tome la rama equivocada si se pierde el orden de hints.
        "b": ["c", "d"],
        "c": ["e"],
        "d": ["f"],
    })
    g.edge_lengths = {}
    g.edge_ids = {}
    g.reference_node_ids = []
    g.reference_path = None
    g.ordered_nodes = list(g.nodes.values())
    g.waypoints = np.empty((0, 3))
    g.wp_node_attrs = np.empty(0, dtype=int)
    g.semantics = SimpleNamespace(
        loaded=False,
        controls=[],
        intersections=[],
        crosswalks=[],
        parking_spots=[],
    )
    g.map_metadata = {}
    return from_track_graph(g, step_m=0.20)


def _converging_lanelet_map():
    g = SimpleNamespace()
    g.step_m = 0.05
    g.nodes = {
        "pose_start": TrackNode("pose_start", -1.0, 0.2, 0, False),
        "route_start": TrackNode("route_start", -1.0, -0.2, 0, False),
        "merge": TrackNode("merge", 0.0, 0.0, 0, False),
        "shared": TrackNode("shared", 1.0, 0.0, 0, False),
        "tail": TrackNode("tail", 2.0, 0.0, 0, False),
    }
    g.adj = defaultdict(list, {
        "pose_start": ["merge"],
        "route_start": ["merge"],
        "merge": ["shared"],
        "shared": ["tail"],
    })
    g.edge_lengths = {}
    g.edge_ids = {}
    g.reference_node_ids = []
    g.reference_path = None
    g.ordered_nodes = list(g.nodes.values())
    g.waypoints = np.empty((0, 3))
    g.wp_node_attrs = np.empty(0, dtype=int)
    g.semantics = SimpleNamespace(
        loaded=False,
        controls=[],
        intersections=[],
        crosswalks=[],
        parking_spots=[],
    )
    g.map_metadata = {}
    return from_track_graph(g, step_m=0.05)


def test_path_has_correct_shape(straight_lanelet_map) -> None:
    """build_target_path devuelve (N+1, 3)."""
    path = build_target_path(
        lanelet_map=straight_lanelet_map,
        start_lanelet_id="n1->n2",
        start_xy=(0.0, 0.0),
        target_speed_mps=0.5,
        horizon_n=10,
        dt=0.05,
    )
    assert path.shape == (11, 3)


def test_path_starts_near_ego(straight_lanelet_map) -> None:
    """El primer waypoint debería estar cerca del ego (proyección)."""
    ego = (0.5, 0.0)
    path = build_target_path(
        lanelet_map=straight_lanelet_map,
        start_lanelet_id="n1->n2",
        start_xy=ego,
        target_speed_mps=0.5,
        horizon_n=8,
        dt=0.05,
    )
    # Distancia entre ego y first waypoint < step_arc.
    step_arc = 0.5 * 0.05
    dist = math.hypot(path[0, 0] - ego[0], path[0, 1] - ego[1])
    assert dist <= step_arc + 0.10  # tolerance for densification artifacts


def test_path_projects_onto_current_segment_before_successor(straight_lanelet_map) -> None:
    """Cerca del final de una lanelet, no debe saltar al nodo terminal.

    Si el ego viene lateralmente desplazado cerca del último punto de la
    centerline, el builder debe arrancar en la proyección sobre el segmento
    actual y no en el vértice final. Ese salto prematuro fuerza un giro
    demasiado brusco al entrar en la sucesora.
    """
    ego = (0.91, 0.21)
    path = build_target_path(
        lanelet_map=straight_lanelet_map,
        start_lanelet_id="n1->n2",
        start_xy=ego,
        target_speed_mps=0.5,
        horizon_n=8,
        dt=0.05,
    )
    assert path[0, 0] == pytest.approx(0.91, abs=0.03)
    assert path[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert path[0, 0] < 0.98
    assert path[1, 0] > path[0, 0]


def test_path_spacing_matches_target_speed(straight_lanelet_map) -> None:
    """Distance entre samples ≈ max(0.10, target_speed * dt).

    El builder aplica un piso de 0.10 m al step para evitar muestreos
    degenerados a velocidades bajas. El test usa parámetros tales que
    `target_speed * dt > 0.10` para verificar la fórmula directamente.
    """
    target_speed = 2.0
    dt = 0.1
    path = build_target_path(
        lanelet_map=straight_lanelet_map,
        start_lanelet_id="n1->n2",
        start_xy=(0.0, 0.0),
        target_speed_mps=target_speed,
        horizon_n=10,
        dt=dt,
    )
    diffs = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    expected_step = max(0.10, target_speed * dt)
    valid_diffs = diffs[diffs > 1e-6]
    assert valid_diffs.size > 3
    assert np.median(valid_diffs) == pytest.approx(expected_step, abs=expected_step * 0.5)


def test_path_headings_match_direction(straight_lanelet_map) -> None:
    """Para una lanelet horizontal +x, todas las headings ≈ 0."""
    path = build_target_path(
        lanelet_map=straight_lanelet_map,
        start_lanelet_id="n1->n2",
        start_xy=(0.0, 0.0),
        target_speed_mps=0.5,
        horizon_n=10,
        dt=0.05,
    )
    # Todos los psi cerca de 0.
    np.testing.assert_allclose(path[:, 2], 0.0, atol=1e-3)


def test_path_handles_zero_speed_safely(straight_lanelet_map) -> None:
    """target_speed=0 ⇒ usa step mínimo de 0.10 m, no crashea."""
    path = build_target_path(
        lanelet_map=straight_lanelet_map,
        start_lanelet_id="n1->n2",
        start_xy=(0.0, 0.0),
        target_speed_mps=0.0,
        horizon_n=5,
        dt=0.05,
    )
    assert path.shape == (6, 3)


def test_path_replicates_end_when_grid_short(straight_lanelet_map) -> None:
    """Horizon larger than available path ⇒ último waypoint se replica."""
    # El grafo tiene 4 edges de 1 m c/u. Con speed 5 m/s y dt 0.05 s,
    # step_arc = 0.25 m, horizonte 100 ⇒ requeriría 25 m. El path se
    # estira pero los últimos samples coinciden con el endpoint.
    path = build_target_path(
        lanelet_map=straight_lanelet_map,
        start_lanelet_id="n1->n2",
        start_xy=(0.0, 0.0),
        target_speed_mps=5.0,
        horizon_n=100,
        dt=0.05,
    )
    assert path.shape == (101, 3)
    # Los últimos pares deberían tener distancia 0 (replicación).
    last_diff = np.linalg.norm(path[-1, :2] - path[-2, :2])
    assert last_diff < 1e-6


def test_path_follows_successor_chain(straight_lanelet_map) -> None:
    """Con horizon largo, el path debe atravesar las sucesoras."""
    path = build_target_path(
        lanelet_map=straight_lanelet_map,
        start_lanelet_id="n1->n2",
        start_xy=(0.0, 0.0),
        target_speed_mps=0.5,
        horizon_n=80,
        dt=0.05,
    )
    # Debería llegar al final del grafo (~4 m en eje x).
    final_x = path[-1, 0]
    assert final_x > 1.5  # cruzó al menos a la 2da lanelet


def test_path_zero_horizon_returns_minimal(straight_lanelet_map) -> None:
    """horizon_n=0 ⇒ devuelve un path mínimo de 1 punto."""
    path = build_target_path(
        lanelet_map=straight_lanelet_map,
        start_lanelet_id="n1->n2",
        start_xy=(0.0, 0.0),
        target_speed_mps=0.5,
        horizon_n=0,
        dt=0.05,
    )
    assert path.shape == (1, 3)


def test_path_prefers_first_ordered_hint_when_multiple_successors_reappear() -> None:
    """Si más de una rama aparece en el hint, manda el orden topológico.

    El bug real acá era tratar `next_lanelet_hint_ids` como set: con dos ramas
    presentes, el builder elegía la primera sucesora del grafo en vez de la
    primera futura de la ruta planificada.
    """
    lanelet_map = _branched_lanelet_map()
    path = build_target_path(
        lanelet_map=lanelet_map,
        start_lanelet_id="s->a",
        start_xy=(0.0, 0.0),
        target_speed_mps=1.0,
        horizon_n=40,
        dt=0.10,
        next_lanelet_hint_ids=("a->b", "b->d", "d->f", "b->c"),
    )
    # La ruta ordenada debe bajar por `b->d`, no subir por `b->c`.
    assert path[-1, 1] < -0.20


def test_path_stays_on_pose_lanelet_before_entering_shared_successor() -> None:
    lanelet_map = _converging_lanelet_map()
    ego = (-0.6, 0.12)
    path = build_target_path(
        lanelet_map=lanelet_map,
        start_lanelet_id="pose_start->merge",
        start_xy=ego,
        target_speed_mps=0.8,
        horizon_n=8,
        dt=0.10,
        next_lanelet_hint_ids=("merge->shared", "shared->tail"),
    )

    assert path[0, 1] == pytest.approx(0.12, abs=0.03)
    assert path[1, 1] > 0.05


def test_route_target_path_builds_straight_hold_prefix_before_rejoining_route() -> None:
    route = np.array([[0.10 * idx, 0.0, 0.0] for idx in range(30)], dtype=float)
    pose = (0.0, 0.03, 0.05)
    path, bridge_meta = build_target_path_from_route(
        route_waypoints=route,
        matched_idx=1,
        start_xy=pose[:2],
        start_yaw_rad=pose[2],
        matched_xy=(0.1, 0.0),
        target_speed_mps=0.1,
        horizon_n=20,
        dt=0.05,
        return_metadata=True,
    )

    assert path.shape[0] >= 6
    assert path[0, 0] == pytest.approx(pose[0], abs=1e-6)
    assert path[0, 1] == pytest.approx(pose[1], abs=1e-6)
    assert bridge_meta["bridge_mode"] == "straight_hold"
    assert bridge_meta["protected_prefix_m"] == pytest.approx(0.45, abs=1e-6)
    assert path[1, 1] <= path[0, 1] + 0.01
    assert path[2, 1] <= path[1, 1] + 0.01
    assert path[6, 1] < path[5, 1]
    assert path[-1, 1] <= 0.001


def test_route_target_path_keeps_straight_prefix_when_small_gap_would_snap_to_route() -> None:
    route = np.array([[0.10 * idx, 0.0, 0.0] for idx in range(30)], dtype=float)
    pose = (0.05, 0.015, 0.05)
    matched_xy = (0.1, 0.0)
    path, bridge_meta = build_target_path_from_route(
        route_waypoints=route,
        matched_idx=1,
        start_xy=pose[:2],
        start_yaw_rad=pose[2],
        matched_xy=matched_xy,
        target_speed_mps=1.0,
        horizon_n=20,
        dt=0.05,
        return_metadata=True,
    )

    assert bridge_meta["bridge_mode"] == "straight_hold"
    assert path[0, 0] == pytest.approx(pose[0], abs=1e-6)
    assert path[0, 1] == pytest.approx(pose[1], abs=1e-6)
    assert path[1, 0] > path[0, 0] + 0.03
    assert path[1, 1] >= pose[1] - 0.003
    assert path[2, 1] >= pose[1] - 0.003
    assert path[8, 1] < path[6, 1]


def test_route_target_path_does_not_insert_backward_micro_segment_at_matched_join() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    router = OsmRouteGraph(str(osm_path), step_m=0.05, start_lanelet_id="50")
    route = router.go_to("50", {"lanelet_id": "75"})

    pose = (4.117731554685099, 5.900499982449048, -1.224637025133419e-06)
    matched_xy = (4.117731554685099, 5.90045)
    path, bridge_meta = build_target_path_from_route(
        route_waypoints=route.waypoints,
        matched_idx=1,
        start_xy=pose[:2],
        start_yaw_rad=pose[2],
        matched_xy=matched_xy,
        target_speed_mps=0.1,
        horizon_n=20,
        dt=0.05,
        return_metadata=True,
    )

    assert bridge_meta["bridge_mode"] == "matched_route_only"
    assert path[0, 0] == pytest.approx(pose[0], abs=1e-6)
    assert path[1, 0] > path[0, 0]
    assert path[2, 0] > path[1, 0]
    assert np.max(np.abs(path[:6, 1] - path[0, 1])) < 1e-3
    assert abs(path[1, 2]) < 0.05


def test_route_target_path_uses_local_recenter_without_dragging_future_turn_branch() -> None:
    osm_path = Path(__file__).resolve().parents[2] / "maps" / "sim" / "lanelet2_map.osm"
    router = OsmRouteGraph(str(osm_path), step_m=0.05, start_lanelet_id="9")
    route = router.go_to(
        {"x": 4.0125, "y": 5.9005, "yaw_rad": 0.0},
        {"lanelet_id": "75", "x": 6.72, "y": 4.066},
    )

    pose = (4.492239130919617, 5.952497235675633, 0.2097604743164211)
    matched_xy = tuple(route.waypoints[5][:2])
    path, bridge_meta = build_target_path_from_route(
        route_waypoints=route.waypoints,
        matched_idx=5,
        start_xy=pose[:2],
        start_yaw_rad=pose[2],
        matched_xy=matched_xy,
        target_speed_mps=0.1,
        horizon_n=20,
        dt=0.05,
        return_metadata=True,
    )

    assert bridge_meta["bridge_mode"] == "connector"
    assert path[0, 0] == pytest.approx(pose[0], abs=1e-6)
    assert path[0, 1] == pytest.approx(pose[1], abs=1e-6)
    assert path[3, 1] < path[1, 1]
    assert path[5, 1] < path[3, 1]
    assert path[-1, 1] > 5.88
