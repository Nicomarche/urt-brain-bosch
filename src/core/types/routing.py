# src/core/types/routing.py
#
# Tipos del subsistema de ruteo. La pieza pública principal es
# `RouteContext`: representa "dónde está el vehículo en el mapa" y "qué
# viene a continuación" en términos del HD map. Es el contrato de
# comunicación entre el `RoutePlanner` (productor) y los consumidores
# (BehaviorPlanner, dashboard).
#
# Diseño:
#   - `RouteContext` es **denso a propósito** — mete tanto info crítica
#     para planning (current_lanelet_id, regulatory_ahead) como info de UI
#     (route_points). Si crece más, se splittea en RouteCoreContext +
#     RouteDebugContext sin romper consumidores actuales (siguen leyendo
#     un solo dataclass).
#   - `RegulatoryElement` vive en `core/types/` (no en `routing/lanelet/`)
#     porque `RouteContext` lo referencia, y `core/types/` no puede
#     depender de paquetes concretos (DIP — la abstracción no depende
#     del detalle).
#
# Convenciones de unidades: igual que `pose.py` (metros, radianes, segs).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.types.pose import Pose2D


@dataclass(frozen=True)
class RegulatoryElement:
    """Elemento regulatorio del HD map — stopline, crosswalk, yield, etc.

    Inspirado en `lanelet2::RegulatoryElement` de Autoware. En BFMC los
    regulatory elements se derivan de:
      - Atributos de nodo del GraphML (`ATTR_STOPLINE=7`,
        `ATTR_CROSSWALK=1`, `ATTR_INTERSECTION=2`).
      - Records de `track_semantics.json` (controls, intersections,
        crosswalks).

    Invariantes:
      - `kind ∈ {"stopline", "crosswalk", "intersection", "yield",
        "speed_limit", "parking", "highway_zone", "roundabout"}`.
      - `position_xy` está en el mismo frame que las lanelets (ENU del
        mapa BFMC, coincide con coordenadas del GraphML).
      - `data` es un dict libre con datos específicos del tipo
        (ej. para "speed_limit", `{"speed_mps": 0.6}`; para
        "crosswalk", `{"width_m": 1.5}`).

    Diseño: frozen para que se puedan compartir entre threads sin
    locks. Si necesitás mutar uno, construí otro con `dataclasses.replace`.
    """

    element_id: str
    kind: str
    position_xy: tuple[float, float]
    anchor_node_id: str | None = None
    label: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteContext:
    """Estado actual de la ruta del vehículo.

    Los campos `current_*`, `upcoming_*`, `next_semantic_*` se actualizan
    cada vez que el `route_planner` recibe una nueva `PoseEstimate`. Son
    la fuente de verdad de "dónde estoy" en términos de mapa.

    Invariantes:
      - `route_progress ∈ [0, 1]` cuando `route_active`. 0 si no hay ruta.
      - `matched_idx ≤ target_idx` siempre (target adelantado para lookahead).
      - `path_kappa` en 1/m. Positivo = curvatura izquierda, negativo derecha.
      - `expected_control_type ∈ {None, "intersection", "stopline",
        "crosswalk", "parking", "highway", "roundabout"}` — pista para que
        el `BehaviorPlanner` active el escenario correcto.
      - **Fase 3:** `current_lanelet_id` es el ID de la lanelet en la que
        está el ego según el `LaneletMap.at_pose(...)`. None si fuera
        del mapa o si el LaneletMap no fue inyectado al planner.
      - **Fase 3:** `next_lanelet_ids` es la cadena de sucesoras a lo
        largo de la ruta activa (orden topológico). Vacío si no hay ruta.
      - **Fase 3:** `regulatory_ahead` lista (en orden de distancia
        creciente) los regulatory elements dentro de
        `LANELET_LOOKAHEAD_M` desde la pose actual. El `BehaviorPlanner`
        consume esto para decidir stopline / crosswalk / yield ahead.
    """

    timestamp: float = 0.0
    route_active: bool = False
    route_id: str | None = None
    route_points: list[dict[str, float]] = field(default_factory=list)
    destination_node_id: str | None = None
    destination_label: str | None = None
    route_queue: list[dict[str, Any]] = field(default_factory=list)
    current_node_id: str | None = None
    current_node_attr: int = 0
    upcoming_node_id: str | None = None
    upcoming_node_attr: int = 0
    maneuver_type: str = "none"
    route_progress: float = 0.0
    route_completed: bool = False
    waypoint_mode_active: bool = False
    matched_idx: int = 0
    target_idx: int = 0
    matched_pose: Pose2D = field(default_factory=Pose2D)
    path_psi: float = 0.0
    path_kappa: float = 0.0
    path_heading_change_rad: float = 0.0
    error_m: float = 0.0
    heading_rad: float = 0.0
    map_match_error_m: float = 0.0
    remaining_distance_m: float = 0.0
    replans: int = 0
    route_source: str = "none"
    destination_point: dict[str, float] | None = None
    next_semantic_id: str | None = None
    next_semantic_type: str | None = None
    next_semantic_label: str | None = None
    next_semantic_distance_m: float | None = None
    expected_control_type: str | None = None
    current_zone_ids: list[str] = field(default_factory=list)
    current_zone_types: list[str] = field(default_factory=list)
    map_metadata: dict[str, Any] = field(default_factory=dict)
    available_destinations: list[dict[str, Any]] = field(default_factory=list)

    # Fase 3 (LaneletMap) — enriquecimiento del contexto de ruta.
    current_lanelet_id: str | None = None
    next_lanelet_ids: tuple[str, ...] = ()
    regulatory_ahead: tuple[RegulatoryElement, ...] = ()
