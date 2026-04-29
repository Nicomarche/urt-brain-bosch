# src/core/types/routing.py
#
# Tipos que describen el contexto de ruta del vehículo. Hoy `RouteContext` es
# el output de `pathManager` (Dijkstra sobre GraphML) — en Fase 3 se
# enriquece con `current_lanelet_id`, `next_lanelet_ids` y `regulatory_ahead`
# cuando el GraphML pase a estar envuelto por `LaneletMap`.
#
# Diseño: RouteContext es "denso" a propósito — incluye lo que necesita el
# BehaviorPlanner (zona actual, semánticos próximos) y lo que necesita el
# dashboard (route_points para dibujar). Si esto crece más, partir en
# RouteCoreContext (lo crítico para planning) + RouteDebugContext (UI).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.types.pose import Pose2D


@dataclass(frozen=True)
class RouteContext:
    """Estado actual de la ruta del vehículo.

    Los campos `current_*`, `upcoming_*`, `next_semantic_*` se actualizan
    cada vez que el `pathManager` recibe una nueva PoseEstimate. Son la
    fuente de verdad de "dónde estoy" en términos de mapa.

    Invariantes:
      - `route_progress ∈ [0, 1]` cuando `route_active`. 0 si no hay ruta.
      - `matched_idx ≤ target_idx` siempre (target adelantado para lookahead).
      - `path_kappa` en 1/m. Positivo = curvatura izquierda, negativo derecha.
      - `expected_control_type ∈ {None, "intersection", "stopline",
        "crosswalk", "parking", "highway", "roundabout"}` — pista para que
        el BehaviorPlanner active el escenario correcto.
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
