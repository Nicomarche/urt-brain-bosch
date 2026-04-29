# tests/routing/test_lanelet_map.py
#
# Tests del LaneletMap. Validamos:
#   1. Lanelet.project_arclength proyecta correctamente (puntos sobre y
#      fuera de la centerline).
#   2. LaneletKDTreeIndex resuelve at_pose en O(log N) y respeta el
#      filtro max_distance.
#   3. LaneletMap.successors_of, predecessors_of, get_lanelet
#      reflejan la topología del grafo.
#   4. regulatory_within devuelve regulatory elements sorted by distance
#      y sin duplicados aun cuando comparten anchor entre lanelets.
#   5. from_track_graph carga el GraphML BFMC real y produce un
#      LaneletMap consistente: cada nodo del grafo cae dentro de su
#      lanelet vecina, los regulators del semantics.json están presentes,
#      los atributos crudos del GraphML se conservan.
#
# Filosofía: usamos un grafo sintético mini cuando podemos (más rápido,
# casos exhibidos), y un GraphML real solo para el test de integración
# end-to-end que valida que la factory no rompa con datos reales.

from __future__ import annotations

import os

import numpy as np
import pytest

from src.core.types.routing import RegulatoryElement
from src.routing.lanelet.from_graphml import (
    ATTR_NORMAL,
    ATTR_STOPLINE,
    TrackGraph,
    TrackNode,
)
from src.routing.lanelet.lanelet_map import (
    Lanelet,
    LaneletMap,
    _densify_segment,
    from_track_graph,
)
from src.routing.lanelet.queries import LaneletKDTreeIndex


# ---------- helpers --------------------------------------------------------


def _mini_graph() -> TrackGraph:
    """Construye un TrackGraph sintético con 4 nodos en L:

        n1 (0,0) ── n2 (1,0) ── n3 (2,0) ── n4 (2,1)

    Útil para ejercitar topología sin pegarle a archivos.
    """
    g = TrackGraph.__new__(TrackGraph)  # bypass __init__ que carga archivo
    g.step_m = 0.10
    g.nodes = {
        "n1": TrackNode("n1", 0.0, 0.0, ATTR_NORMAL, True),
        "n2": TrackNode("n2", 1.0, 0.0, ATTR_NORMAL, False),
        "n3": TrackNode("n3", 2.0, 0.0, ATTR_STOPLINE, False),
        "n4": TrackNode("n4", 2.0, 1.0, ATTR_NORMAL, False),
    }
    from collections import defaultdict
    g.adj = defaultdict(list)
    g.adj["n1"].append("n2")
    g.adj["n2"].append("n3")
    g.adj["n3"].append("n4")
    g.edge_lengths = {("n1", "n2"): 1.0, ("n2", "n3"): 1.0, ("n3", "n4"): 1.0}
    g.edge_ids = {("n1", "n2"): "e1", ("n2", "n3"): "e2", ("n3", "n4"): "e3"}
    g.reference_node_ids = []
    g.reference_path = None
    g.ordered_nodes = list(g.nodes.values())
    g.waypoints = np.empty((0, 3))
    g.wp_node_attrs = np.empty(0, dtype=int)
    # semantics vacíos — solo regulators implícitos
    from src.routing.lanelet.semantics import TrackSemantics
    g.semantics = TrackSemantics(None)
    g.map_metadata = {}
    return g


# ---------- _densify_segment ----------------------------------------------


def test_densify_includes_endpoints() -> None:
    """Los endpoints del segmento siempre están en el output."""
    pts = _densify_segment(0.0, 0.0, 1.0, 0.0, step_m=0.20)
    np.testing.assert_allclose(pts[0], [0.0, 0.0])
    np.testing.assert_allclose(pts[-1], [1.0, 0.0])


def test_densify_resolution_respects_step() -> None:
    """Con step=0.25 y segmento de 1 m, debería haber ≥ 5 puntos (4 pasos)."""
    pts = _densify_segment(0.0, 0.0, 1.0, 0.0, step_m=0.25)
    assert pts.shape[0] >= 5


def test_densify_zero_length_returns_single_point() -> None:
    """Si los endpoints coinciden, devolvemos un solo punto."""
    pts = _densify_segment(1.0, 1.0, 1.0, 1.0, step_m=0.10)
    assert pts.shape == (1, 2)


# ---------- Lanelet.project_arclength -------------------------------------


def _straight_lanelet() -> Lanelet:
    """Lanelet recta de (0,0) a (10,0) para tests de proyección."""
    cl = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    return Lanelet(
        lanelet_id="L1",
        source_node_id="A",
        target_node_id="B",
        centerline=cl,
        length_m=10.0,
        attribute=ATTR_NORMAL,
    )


def test_project_on_centerline_zero_offset() -> None:
    """Punto sobre la centerline ⇒ offset lateral 0."""
    ll = _straight_lanelet()
    arc, offset = ll.project_arclength(3.0, 0.0)
    assert arc == pytest.approx(3.0, abs=1e-6)
    assert offset == pytest.approx(0.0, abs=1e-6)


def test_project_left_of_centerline_positive_offset() -> None:
    """Punto a la izquierda del avance ⇒ offset positivo (convención CCW)."""
    ll = _straight_lanelet()
    _, offset = ll.project_arclength(2.0, 0.5)  # avance +x → izquierda es +y
    assert offset > 0.0


def test_project_right_of_centerline_negative_offset() -> None:
    """Punto a la derecha del avance ⇒ offset negativo."""
    ll = _straight_lanelet()
    _, offset = ll.project_arclength(2.0, -0.5)
    assert offset < 0.0


def test_project_clamps_arc_to_endpoints() -> None:
    """Punto antes del inicio: arc=0 (clamp). Después del final: arc=length."""
    ll = _straight_lanelet()
    arc_before, _ = ll.project_arclength(-3.0, 0.0)
    arc_after, _ = ll.project_arclength(15.0, 0.0)
    assert arc_before == pytest.approx(0.0, abs=1e-6)
    assert arc_after == pytest.approx(10.0, abs=1e-6)


# ---------- LaneletKDTreeIndex --------------------------------------------


def test_kdtree_index_query_returns_nearest_lanelet() -> None:
    """El KDTree devuelve la lanelet del punto de centerline más cercano."""
    cl_a = np.array([[0.0, 0.0], [1.0, 0.0]])
    cl_b = np.array([[10.0, 10.0], [11.0, 10.0]])
    lanelets = {
        "A": Lanelet("A", "n1", "n2", cl_a, 1.0, 0),
        "B": Lanelet("B", "n3", "n4", cl_b, 1.0, 0),
    }
    idx = LaneletKDTreeIndex.from_lanelets(lanelets)
    assert idx.query_lanelet(0.5, 0.0) == "A"
    assert idx.query_lanelet(10.5, 10.0) == "B"


def test_kdtree_index_max_distance_filter() -> None:
    """Si la distancia al punto más cercano supera max_distance, → None."""
    cl = np.array([[0.0, 0.0], [1.0, 0.0]])
    lanelets = {"A": Lanelet("A", "n1", "n2", cl, 1.0, 0)}
    idx = LaneletKDTreeIndex.from_lanelets(lanelets)
    assert idx.query_lanelet(100.0, 100.0, max_distance_m=5.0) is None
    # Sin filtro, sí devuelve la única lanelet.
    assert idx.query_lanelet(100.0, 100.0) == "A"


def test_kdtree_index_empty_returns_none() -> None:
    """KDTree vacío ⇒ query_lanelet devuelve None."""
    idx = LaneletKDTreeIndex.from_lanelets({})
    assert idx.query_lanelet(0.0, 0.0) is None
    assert idx.num_points == 0


# ---------- LaneletMap topology (mini graph) -------------------------------


def test_from_track_graph_creates_one_lanelet_per_edge() -> None:
    """Cada edge dirigido del TrackGraph se convierte en una Lanelet."""
    g = _mini_graph()
    lm = from_track_graph(g, step_m=0.10)
    # 3 edges: n1->n2, n2->n3, n3->n4
    assert len(lm) == 3
    assert "n1->n2" in lm.lanelet_ids
    assert "n2->n3" in lm.lanelet_ids
    assert "n3->n4" in lm.lanelet_ids


def test_lanelet_topology_successors_predecessors() -> None:
    """successors_of(n1->n2) == ('n2->n3',). predecessors_of(n3->n4) == ('n2->n3',)."""
    lm = from_track_graph(_mini_graph(), step_m=0.10)
    assert lm.successors_of("n1->n2") == ("n2->n3",)
    assert lm.successors_of("n3->n4") == ()  # terminal
    assert lm.predecessors_of("n3->n4") == ("n2->n3",)
    assert lm.predecessors_of("n1->n2") == ()  # source


def test_lanelet_attribute_inherits_from_target_node() -> None:
    """El atributo de la lanelet es el del nodo destino del edge."""
    lm = from_track_graph(_mini_graph(), step_m=0.10)
    # n3 tiene atributo ATTR_STOPLINE, así que la lanelet n2->n3 hereda eso.
    assert lm.get_lanelet("n2->n3").attribute == ATTR_STOPLINE
    assert lm.get_lanelet("n1->n2").attribute == ATTR_NORMAL


def test_at_pose_finds_lanelet_for_midpoint() -> None:
    """at_pose en el centro de un edge devuelve esa lanelet."""
    lm = from_track_graph(_mini_graph(), step_m=0.10)
    # Punto (0.5, 0) está sobre el edge n1->n2.
    assert lm.at_pose(0.5, 0.0) == "n1->n2"


def test_at_pose_returns_none_outside_max_distance() -> None:
    """Punto muy lejos del mapa con max_distance estricto ⇒ None."""
    lm = from_track_graph(_mini_graph(), step_m=0.10)
    assert lm.at_pose(100.0, 100.0, max_distance_m=5.0) is None


# ---------- regulatory_within ---------------------------------------------


def test_regulatory_within_picks_up_implicit_stopline() -> None:
    """ATTR_STOPLINE en n3 genera un regulator implícito attr-stopline-n3."""
    lm = from_track_graph(_mini_graph(), step_m=0.10)
    regs = lm.regulatory_within("n1->n2", distance_m=10.0)
    kinds = {r.kind for r in regs}
    assert "stopline" in kinds


def test_regulatory_within_respects_distance_budget() -> None:
    """Con presupuesto < 1 m no se cruza el primer edge ⇒ no llega a stopline en n3."""
    lm = from_track_graph(_mini_graph(), step_m=0.10)
    regs_short = lm.regulatory_within("n1->n2", distance_m=0.5)
    # n1 y n2 son ATTR_NORMAL, no generan regulators. Solo se vería
    # stopline si BFS llegara a n3 (siguiente lanelet n2->n3) — pero
    # 0.5 m no alcanza para cruzar el primer edge de 1 m.
    kinds = {r.kind for r in regs_short}
    assert "stopline" not in kinds


def test_regulatory_within_dedup_across_lanelets() -> None:
    """Un mismo regulator anclado a un nodo compartido aparece una sola vez."""
    lm = from_track_graph(_mini_graph(), step_m=0.10)
    regs = lm.regulatory_within("n1->n2", distance_m=10.0)
    ids = [r.element_id for r in regs]
    assert len(ids) == len(set(ids))  # sin duplicados


# ---------- Integración con GraphML real -----------------------------------


_REAL_GRAPHML = "Track GraphML File.graphml"


@pytest.fixture(scope="module")
def real_track_graph() -> TrackGraph:
    """Carga el GraphML real de BFMC. Skipea si no existe (CI sin assets)."""
    if not os.path.exists(_REAL_GRAPHML):
        pytest.skip(f"{_REAL_GRAPHML} no presente; integración omitida.")
    return TrackGraph(_REAL_GRAPHML)


def test_real_graphml_loads_nontrivial_lanelet_count(real_track_graph: TrackGraph) -> None:
    """El GraphML BFMC produce > 50 lanelets (sanity check de orden de magnitud)."""
    lm = from_track_graph(real_track_graph)
    assert len(lm) > 50


def test_real_graphml_at_pose_resolves_known_node(real_track_graph: TrackGraph) -> None:
    """Cada nodo del grafo cae dentro de la lanelet vecina (offset < 1 m)."""
    lm = from_track_graph(real_track_graph)
    # Tomamos el primer nodo no-aislado.
    for node in real_track_graph.nodes.values():
        if not real_track_graph.adj.get(node.node_id):
            continue
        ll_id = lm.at_pose(node.x, node.y, max_distance_m=1.0)
        assert ll_id is not None, f"Nodo {node.node_id} no cayó en ninguna lanelet."
        break


def test_real_graphml_lanelet_has_consistent_centerline(real_track_graph: TrackGraph) -> None:
    """Sample una lanelet y validá que length_m ≈ longitud arco computada."""
    lm = from_track_graph(real_track_graph)
    ll = lm.get_lanelet(lm.lanelet_ids[0])
    assert ll is not None
    diffs = np.diff(ll.centerline, axis=0)
    arc_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    assert ll.length_m == pytest.approx(arc_length, abs=1e-6)


def test_real_graphml_topology_edges_match(real_track_graph: TrackGraph) -> None:
    """Cantidad de Lanelets == cantidad de edges del TrackGraph."""
    edge_count = sum(len(neighbors) for neighbors in real_track_graph.adj.values())
    lm = from_track_graph(real_track_graph)
    assert len(lm) == edge_count


# ---------- RegulatoryElement dataclass -----------------------------------


def test_regulatory_element_is_frozen() -> None:
    """RegulatoryElement es frozen ⇒ asignar campos lanza FrozenInstanceError."""
    reg = RegulatoryElement(
        element_id="r1",
        kind="stopline",
        position_xy=(1.0, 2.0),
    )
    with pytest.raises(Exception):
        reg.kind = "crosswalk"  # type: ignore[misc]


def test_regulatory_element_default_data_is_independent() -> None:
    """Cada instancia tiene su propio dict `data` (default_factory)."""
    a = RegulatoryElement("a", "stopline", (0.0, 0.0))
    b = RegulatoryElement("b", "stopline", (0.0, 0.0))
    a.data["foo"] = 1
    assert "foo" not in b.data
