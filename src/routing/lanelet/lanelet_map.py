# src/routing/lanelet/lanelet_map.py
#
# `LaneletMap` — capa "HD map" de BFMC. Inspirada en
# `lanelet2::Map` de Autoware (`autoware_lanelet2_extension`), pero
# reimplementada en Python idiomático: NO usamos los bindings C++ de
# `lanelet2` porque no compilan en Jetson Nano.
#
# ¿Qué resuelve?
#   El `TrackGraph` nativo da edges + nodes + atributos. El planner de
#   comportamiento necesita queries más ricos:
#     - "¿en qué lanelet estoy?"  → at_pose(x, y)
#     - "¿qué viene después?"    → successors_of(lanelet_id)
#     - "¿qué stoplines/crosswalks hay en los próximos N metros?"
#                                  → regulatory_within(lanelet_id, distance_m)
#
#   El `BehaviorPlanner` consume estas tres APIs y compone su decisión.
#   Si mañana cambiamos el formato del mapa (ej. lanelet2 OSM), basta con
#   reemplazar `from_track_graph()` por otro constructor; el contrato
#   `LaneletMap` se mantiene.
#
# Convenciones:
#   - `Lanelet.lanelet_id` se construye como `"{src_node_id}->{tgt_node_id}"`
#     desde el GraphML. Permite reconstrucción inversa para debug.
#   - Las centerlines están en el mismo frame que el GraphML (metros, ENU).
#     `from_track_graph()` densifica con `step_m` para que `at_pose` tenga
#     resolución sub-metro.
#   - Los `RegulatoryElement` se derivan de DOS fuentes:
#       1. Atributos de nodo (ATTR_STOPLINE, ATTR_CROSSWALK, ATTR_INTERSECTION)
#       2. Records de `track_semantics.json` (controls/intersections/...)
#     Si una stopline aparece en ambos, se prefiere el record de semantics
#     porque trae `control_type` y `data` adicionales.
#
# Pitfall conocido — tramos highway:
#   El GraphML BFMC tiene UNA sola línea por tramo highway con atributo
#   `ATTR_HIGHWAY_LEFT` (4) o `ATTR_HIGHWAY_RIGHT` (5). El plan original
#   habla de generar lanelets desplazadas con `_LANE_HALF_WIDTH_M`; en
#   esta implementación no las desplazamos lateralmente — el atributo
#   queda en `Lanelet.attribute` para que el `BehaviorPlanner` aplique
#   semántica de carril (ej. "estoy en HIGHWAY_LEFT, no cambies a
#   HIGHWAY_RIGHT salvo por permiso explícito"). Si la pista BFMC 2026
#   pide carriles geométricamente separados, se agrega el offset acá
#   sin tocar el contrato público.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from src.core.types.routing import RegulatoryElement
from src.routing.lanelet.from_graphml import (
    ATTR_CROSSWALK,
    ATTR_INTERSECTION,
    ATTR_STOPLINE,
)

if TYPE_CHECKING:
    from src.routing.lanelet.from_graphml import TrackGraph
    from src.routing.lanelet.queries import LaneletKDTreeIndex


# Mapa atributo de nodo → kind del RegulatoryElement implícito.
# Solo entran acá los atributos que el `BehaviorPlanner` debe tratar
# como evento regulatorio independiente. Ej. `ATTR_HIGHWAY_LEFT` NO
# está acá: pertenece al atributo de la lanelet, no a un regulator.
_NODE_ATTR_TO_REGULATOR_KIND: dict[int, str] = {
    ATTR_STOPLINE: "stopline",
    ATTR_CROSSWALK: "crosswalk",
    ATTR_INTERSECTION: "intersection",
}


@dataclass(frozen=True)
class Lanelet:
    """Segmento de carril consultable. Equivalente liviano a `lanelet2::Lanelet`.

    Una lanelet representa el arco entre dos nodos del GraphML
    (eje dirigido). Su `centerline` está densificada para que las
    queries espaciales (`at_pose`) tengan resolución < 1 m.

    Invariantes:
      - `centerline.shape == (N, 2)` con N >= 2 y todos los puntos en
        coordenadas planares ENU (mismas que el GraphML).
      - `length_m == longitud arco de centerline`. Cacheado para evitar
        recomputarlo en cada query.
      - `successor_ids` y `predecessor_ids` son tuplas (frozen) para que
        el dataclass siga siendo hashable. La lista vacía representa
        nodos sumidero / fuente.
      - `regulatory_element_ids` referencia al diccionario global de
        regulators del `LaneletMap` (no copia el objeto). Permite que
        múltiples lanelets compartan un mismo stopline.
      - `attribute` es el código entero del GraphML (ATTR_*). Se conserva
        crudo — la traducción semántica vive en el consumidor.

    Diseño: frozen por la misma razón que `RegulatoryElement` —
    inmutabilidad cross-thread sin locks.
    """

    lanelet_id: str
    source_node_id: str
    target_node_id: str
    centerline: np.ndarray  # shape (N, 2)
    length_m: float
    attribute: int
    predecessor_ids: tuple[str, ...] = ()
    successor_ids: tuple[str, ...] = ()
    regulatory_element_ids: tuple[str, ...] = ()

    def project_arclength(self, x: float, y: float) -> tuple[float, float]:
        """Proyecta (x, y) sobre la centerline.

        Devuelve `(arclength_m, lateral_offset_m)`:
          - `arclength_m` ∈ [0, length_m] — distancia desde el inicio de
            la centerline al punto proyectado más cercano.
          - `lateral_offset_m` — distancia perpendicular (signo positivo
            si el punto está a la izquierda del sentido de avance).

        Implementación: O(N) — escanea todos los segmentos. Para N
        chico (< 50 puntos por lanelet) es más rápido que el overhead
        de un KDTree por-lanelet.
        """
        cl = self.centerline
        if cl.shape[0] < 2:
            return 0.0, math.hypot(x - cl[0, 0], y - cl[0, 1])

        # Trackeamos el segmento ganador por distancia mínima al punto
        # PROYECTADO Y CLAMPEADO. Esto difiere de la distancia
        # perpendicular a la línea infinita: cuando el punto cae fuera
        # del segmento, la distancia clampeada es la que importa para
        # decidir qué segmento es "el más cercano".
        best_arc = 0.0
        best_offset = 0.0
        best_dist = math.inf
        cumulative = 0.0
        px, py = float(x), float(y)
        for i in range(cl.shape[0] - 1):
            ax, ay = float(cl[i, 0]), float(cl[i, 1])
            bx, by = float(cl[i + 1, 0]), float(cl[i + 1, 1])
            seg_dx = bx - ax
            seg_dy = by - ay
            seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
            if seg_len_sq <= 1e-12:
                continue
            seg_len = math.sqrt(seg_len_sq)
            t = ((px - ax) * seg_dx + (py - ay) * seg_dy) / seg_len_sq
            t_clamped = max(0.0, min(1.0, t))
            qx = ax + t_clamped * seg_dx
            qy = ay + t_clamped * seg_dy
            dist = math.hypot(px - qx, py - qy)
            if dist < best_dist:
                best_dist = dist
                # Signo del offset = lado del punto respecto al sentido
                # del segmento (cross product 2D normalizado).
                signed = (seg_dx * (py - ay) - seg_dy * (px - ax)) / seg_len
                best_offset = signed
                best_arc = cumulative + t_clamped * seg_len
            cumulative += seg_len
        return best_arc, float(best_offset)


class LaneletMap:
    """HD map consultable. Composición liviana de Lanelets + Regulators.

    Responsabilidades:
      - Almacenar el catálogo de lanelets y regulatory elements.
      - Resolver queries espaciales (`at_pose`) vía un KDTree externo.
      - Resolver queries topológicas (`successors_of`, `predecessors_of`).
      - Listar regulators "ahead" desde una lanelet en una distancia dada.

    NO responsabilidades (delegadas):
      - Construcción a partir del GraphML — la hace `from_track_graph()`
        (factory en este módulo).
      - Indexado espacial — lo hace `LaneletKDTreeIndex` en `queries.py`.
      - Routing global / Dijkstra — lo hace `route_planner.py`.

    El `BehaviorPlanner` consume `LaneletMap` por la interfaz pública
    de abajo. NO depende de `from_track_graph` ni de detalles del
    formato GraphML.
    """

    def __init__(
        self,
        lanelets: dict[str, Lanelet],
        regulators: dict[str, RegulatoryElement],
        kdtree_index: "LaneletKDTreeIndex | None",
    ) -> None:
        self._lanelets: dict[str, Lanelet] = dict(lanelets)
        self._regulators: dict[str, RegulatoryElement] = dict(regulators)
        self._kdtree: "LaneletKDTreeIndex | None" = kdtree_index

    # --------------------------------------------------------------
    # Catálogo
    # --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._lanelets)

    @property
    def lanelet_ids(self) -> tuple[str, ...]:
        return tuple(self._lanelets.keys())

    @property
    def regulator_ids(self) -> tuple[str, ...]:
        return tuple(self._regulators.keys())

    def get_lanelet(self, lanelet_id: str) -> Lanelet | None:
        return self._lanelets.get(lanelet_id)

    def get_regulator(self, regulator_id: str) -> RegulatoryElement | None:
        return self._regulators.get(regulator_id)

    # --------------------------------------------------------------
    # Queries espaciales
    # --------------------------------------------------------------
    def at_pose(self, x: float, y: float, max_distance_m: float | None = None) -> str | None:
        """Devuelve la lanelet más cercana al punto (x, y).

        - `max_distance_m`: si se especifica y la nearest lanelet está más
          lejos, devuelve None (vehículo fuera del mapa). Útil para que el
          BehaviorPlanner detecte "estoy fuera de pista" y pida fallback.
        - Implementación: el `KDTreeIndex` resuelve el punto-más-cercano
          en O(log N) sobre TODOS los puntos de centerline; el `Lanelet`
          asociado al hit es el ganador.
        """
        if self._kdtree is None:
            return None
        return self._kdtree.query_lanelet(x, y, max_distance_m=max_distance_m)

    # --------------------------------------------------------------
    # Queries topológicas
    # --------------------------------------------------------------
    def successors_of(self, lanelet_id: str) -> tuple[str, ...]:
        """IDs de las lanelets que continúan después de `lanelet_id`.

        Vacío si la lanelet no existe o si es un sumidero terminal.
        """
        ll = self._lanelets.get(lanelet_id)
        if ll is None:
            return ()
        return ll.successor_ids

    def predecessors_of(self, lanelet_id: str) -> tuple[str, ...]:
        ll = self._lanelets.get(lanelet_id)
        if ll is None:
            return ()
        return ll.predecessor_ids

    def regulatory_within(
        self,
        lanelet_id: str,
        distance_m: float,
    ) -> tuple[RegulatoryElement, ...]:
        """RegulatoryElements alcanzables desde el inicio de `lanelet_id`
        siguiendo sucesoras hasta cubrir `distance_m`.

        Recorrido BFS — si el grafo se ramifica, todas las ramas se
        exploran hasta agotar la distancia. Devuelve una tupla **única**
        (sin duplicados) ordenada por distancia creciente desde el inicio
        de la lanelet de partida.

        Diseño: el BehaviorPlanner usa esto para responder "¿hay
        stopline en los próximos N metros?". Si N es muy grande la
        ramificación explota; en BFMC con N ≤ 30 m el cost es trivial.
        """
        if distance_m <= 0:
            return ()
        start = self._lanelets.get(lanelet_id)
        if start is None:
            return ()

        # BFS con cola priorizada por distancia acumulada para
        # respetar orden creciente al construir el resultado.
        visited: set[str] = set()
        ordered: list[tuple[float, str]] = []  # (distancia, regulator_id)
        # Stack de (lanelet_id, distancia acumulada al inicio de esa lanelet).
        frontier: list[tuple[str, float]] = [(lanelet_id, 0.0)]

        while frontier:
            current_id, accumulated = frontier.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            ll = self._lanelets.get(current_id)
            if ll is None:
                continue
            # Distancia al final de esta lanelet.
            end_distance = accumulated + ll.length_m
            # Cada regulator se reporta a la distancia donde aparece
            # (proxy: inicio de la lanelet que lo contiene).
            for reg_id in ll.regulatory_element_ids:
                if accumulated > distance_m:
                    continue
                ordered.append((accumulated, reg_id))
            if end_distance >= distance_m:
                continue
            for nxt in ll.successor_ids:
                if nxt not in visited:
                    frontier.append((nxt, end_distance))

        # Dedup preservando orden de primera aparición; ordenar por dist.
        ordered.sort(key=lambda pair: pair[0])
        seen: set[str] = set()
        result: list[RegulatoryElement] = []
        for _, reg_id in ordered:
            if reg_id in seen:
                continue
            seen.add(reg_id)
            reg = self._regulators.get(reg_id)
            if reg is not None:
                result.append(reg)
        return tuple(result)


# ---------------------------------------------------------------------
# Factory: TrackGraph → LaneletMap
# ---------------------------------------------------------------------


def _densify_segment(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    step_m: float,
) -> np.ndarray:
    """Densifica el segmento (a, b) con resolución `step_m`.

    Devuelve un ndarray (N, 2). N >= 2 (siempre incluye los endpoints).
    Si la longitud del segmento es < step_m, devuelve solo los endpoints.
    """
    length = math.hypot(bx - ax, by - ay)
    if length <= 1e-9:
        return np.array([[ax, ay]], dtype=float)
    n_steps = max(1, int(math.ceil(length / max(step_m, 1e-3))))
    ts = np.linspace(0.0, 1.0, n_steps + 1)
    xs = ax + ts * (bx - ax)
    ys = ay + ts * (by - ay)
    return np.stack([xs, ys], axis=1)


def _build_regulators_from_semantics(track_graph: "TrackGraph") -> dict[str, RegulatoryElement]:
    """Carga RegulatoryElements desde `track_semantics.json`.

    El loader actual (`TrackSemantics`) ya parsea controls/intersections/
    crosswalks/parking_spots. Acá los convertimos al tipo público.
    Cada record genera un RegulatoryElement anclado a su `anchor_node_id`.
    """
    result: dict[str, RegulatoryElement] = {}
    semantics = track_graph.semantics
    if not semantics.loaded:
        return result

    def _node_xy(node_id: str | None) -> tuple[float, float] | None:
        if node_id is None:
            return None
        node = track_graph.nodes.get(str(node_id))
        if node is None:
            return None
        return float(node.x), float(node.y)

    # Controls (stop signs, traffic lights, etc.).
    for control in semantics.controls:
        ctrl_id = str(control.get("id") or "").strip()
        anchor = control.get("anchor_node_id")
        xy = _node_xy(anchor)
        if not ctrl_id or xy is None:
            continue
        kind = str(control.get("type") or "control").strip().lower()
        result[ctrl_id] = RegulatoryElement(
            element_id=ctrl_id,
            kind=kind,
            position_xy=xy,
            anchor_node_id=str(anchor) if anchor is not None else None,
            label=str(control.get("label") or ctrl_id),
            data={k: v for k, v in control.items() if k not in ("id", "anchor_node_id", "type", "label")},
        )

    # Intersections.
    for inter in semantics.intersections:
        inter_id = str(inter.get("id") or "").strip()
        anchor = inter.get("anchor_node_id")
        xy = _node_xy(anchor)
        if not inter_id or xy is None:
            continue
        if inter_id in result:
            continue
        result[inter_id] = RegulatoryElement(
            element_id=inter_id,
            kind="intersection",
            position_xy=xy,
            anchor_node_id=str(anchor) if anchor is not None else None,
            label=str(inter.get("label") or inter_id),
            data={k: v for k, v in inter.items() if k not in ("id", "anchor_node_id", "label")},
        )

    # Crosswalks.
    for cw in semantics.crosswalks:
        cw_id = str(cw.get("id") or "").strip()
        anchor = cw.get("anchor_node_id") or cw.get("stopline_node_id")
        xy = _node_xy(anchor)
        if not cw_id or xy is None:
            continue
        if cw_id in result:
            continue
        result[cw_id] = RegulatoryElement(
            element_id=cw_id,
            kind="crosswalk",
            position_xy=xy,
            anchor_node_id=str(anchor) if anchor is not None else None,
            label=str(cw.get("label") or cw_id),
            data={k: v for k, v in cw.items() if k not in ("id", "anchor_node_id", "label")},
        )

    # Parking spots.
    for ps in semantics.parking_spots:
        ps_id = str(ps.get("id") or "").strip()
        anchor = ps.get("anchor_node_id") or ps.get("entry_node_id")
        xy = _node_xy(anchor)
        if not ps_id or xy is None:
            continue
        if ps_id in result:
            continue
        result[ps_id] = RegulatoryElement(
            element_id=ps_id,
            kind="parking",
            position_xy=xy,
            anchor_node_id=str(anchor) if anchor is not None else None,
            label=str(ps.get("label") or ps_id),
            data={k: v for k, v in ps.items() if k not in ("id", "anchor_node_id", "label")},
        )

    return result


def _build_implicit_regulators(track_graph: "TrackGraph") -> dict[str, RegulatoryElement]:
    """Genera RegulatoryElements implícitos a partir de atributos de nodo.

    Algunos atributos del GraphML implican regulatory elements aunque no
    estén en `track_semantics.json` (ej. ATTR_STOPLINE en un nodo no
    asociado a un control explícito). Acá los emitimos con IDs
    sintéticos `attr-{kind}-{node_id}` para que el LaneletMap quede
    auto-suficiente sin depender exclusivamente de semantics.json.
    """
    result: dict[str, RegulatoryElement] = {}
    for node_id, node in track_graph.nodes.items():
        kind = _NODE_ATTR_TO_REGULATOR_KIND.get(node.attribute)
        if kind is None:
            continue
        synth_id = f"attr-{kind}-{node_id}"
        result[synth_id] = RegulatoryElement(
            element_id=synth_id,
            kind=kind,
            position_xy=(float(node.x), float(node.y)),
            anchor_node_id=str(node_id),
            label=kind.capitalize(),
            data={"source": "node_attribute", "attribute_code": int(node.attribute)},
        )
    return result


def from_track_graph(track_graph: "TrackGraph", step_m: float = 0.20) -> LaneletMap:
    """Factory: construye `LaneletMap` desde un `TrackGraph` cargado.

    Cada edge dirigido del GraphML se convierte en una `Lanelet` con:
      - `lanelet_id = "{src}->{tgt}"`
      - `centerline` densificada a paso `step_m` entre los dos nodos
      - `attribute` = atributo del nodo destino (la regla de BFMC es que
        el atributo del eje pertenece al nodo entrante)
      - `successor_ids` derivados de la adjacencia salida del nodo target
      - `predecessor_ids` derivados de la adjacencia entrada del nodo source
      - `regulatory_element_ids` = unión de regulators en src y tgt nodos.

    El `LaneletKDTreeIndex` se construye al final con TODOS los puntos
    de centerline para que `at_pose` resuelva en O(log N).

    Args:
      track_graph: Instancia de `TrackGraph` ya cargada.
      step_m: Paso de densificación de centerlines (m). Default 0.20 m
        balancea memoria (≈5 puntos por edge típico de 1 m) y resolución
        (`at_pose` resuelve dentro de 10 cm).
    """
    # Importación local para romper ciclo: queries.py importa Lanelet.
    from src.routing.lanelet.queries import LaneletKDTreeIndex

    # 1. Regulators: primero los explícitos (semantics), luego los
    #    implícitos por atributo de nodo. El primero gana en colisión
    #    porque tiene más data (label, control_type, etc.).
    regulators = _build_regulators_from_semantics(track_graph)
    for synth_id, reg in _build_implicit_regulators(track_graph).items():
        regulators.setdefault(synth_id, reg)

    # 2. Index inverso: node_id → regulator_ids para asociar a lanelets.
    regulators_by_node: dict[str, list[str]] = {}
    for reg_id, reg in regulators.items():
        if reg.anchor_node_id is not None:
            regulators_by_node.setdefault(reg.anchor_node_id, []).append(reg_id)

    # 3. Lanelets (una por edge).
    lanelets: dict[str, Lanelet] = {}
    for src_id, neighbors in track_graph.adj.items():
        src_node = track_graph.nodes.get(src_id)
        if src_node is None:
            continue
        for tgt_id in neighbors:
            tgt_node = track_graph.nodes.get(tgt_id)
            if tgt_node is None:
                continue
            centerline = _densify_segment(
                src_node.x, src_node.y, tgt_node.x, tgt_node.y, step_m=step_m
            )
            length = float(np.sum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))) if centerline.shape[0] >= 2 else 0.0
            reg_ids = tuple(
                regulators_by_node.get(src_id, []) + regulators_by_node.get(tgt_id, [])
            )
            ll_id = f"{src_id}->{tgt_id}"
            lanelets[ll_id] = Lanelet(
                lanelet_id=ll_id,
                source_node_id=str(src_id),
                target_node_id=str(tgt_id),
                centerline=centerline,
                length_m=length,
                attribute=int(tgt_node.attribute),
                regulatory_element_ids=reg_ids,
            )

    # 4. Conexiones predecessor/successor entre lanelets.
    #    successors_of(L) = todas las lanelets cuyo source == L.target.
    #    predecessors_of(L) = todas las lanelets cuyo target == L.source.
    lanelets_by_source: dict[str, list[str]] = {}
    lanelets_by_target: dict[str, list[str]] = {}
    for ll_id, ll in lanelets.items():
        lanelets_by_source.setdefault(ll.source_node_id, []).append(ll_id)
        lanelets_by_target.setdefault(ll.target_node_id, []).append(ll_id)

    enriched: dict[str, Lanelet] = {}
    for ll_id, ll in lanelets.items():
        successors = tuple(lanelets_by_source.get(ll.target_node_id, ()))
        predecessors = tuple(lanelets_by_target.get(ll.source_node_id, ()))
        enriched[ll_id] = Lanelet(
            lanelet_id=ll.lanelet_id,
            source_node_id=ll.source_node_id,
            target_node_id=ll.target_node_id,
            centerline=ll.centerline,
            length_m=ll.length_m,
            attribute=ll.attribute,
            predecessor_ids=predecessors,
            successor_ids=successors,
            regulatory_element_ids=ll.regulatory_element_ids,
        )

    # 5. KDTree index.
    kdtree = LaneletKDTreeIndex.from_lanelets(enriched)
    return LaneletMap(lanelets=enriched, regulators=regulators, kdtree_index=kdtree)
