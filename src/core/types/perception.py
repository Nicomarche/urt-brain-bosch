# src/core/types/perception.py
#
# Tipos producidos por la capa de percepción visual. La cadena es:
#
#   frame BGR ──► YOLO TensorRT ──► LaneObservation
#                                ──► StoplineObservation
#                                ──► DetectedObject (raw bbox)
#                                ──► TrackedObject (con ID estable + velocidad)
#
# LaneObservation y StoplineObservation son las "mediciones" que consume el
# pipeline de localización (relocalization cascade en EKF7) y el
# BehaviorPlanner (caps de velocidad por crosswalk/intersection).
#
# DetectedObject es la salida cruda por frame del YOLO para clases
# no-sign (peatones, otros autos, semáforos como objeto). TrackedObject
# es lo que produce el `MOTTracker` (Fase 5) tras asociar detecciones
# frame-a-frame, con ID persistente y estimación de velocidad. El
# BehaviorPlanner consume `TrackedObject` para slowdowns / yields.
#
# VisualControlCandidate y VisualStateSnapshot son tipos legacy de la era
# Stanley/PID. Sobreviven en Fase 1 porque el thread de control coordinator
# todavía los lee; desaparecen cuando MotorCommand sea la única salida
# (Fase 6).

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

_LATERAL_RELOCALIZATION_BLOCKED_CONTROL_MODES = frozenset({
    "MANEUVER_CONTROL",
})


@dataclass(frozen=True)
class LaneObservation:
    """Medición de carril emitida por el detector visual frame-a-frame.

    Conventions:
      - `lateral_offset_m`: positivo = vehículo a la izquierda del centro.
        None si la confianza es demasiado baja. Derivado del primer waypoint
        cuando hay `center_waypoints_body` disponibles.
      - `heading_error_rad`: error de yaw del vehículo respecto a la tangente
        del carril (radianes; positivo CCW).
      - `quality ∈ [0, 1]`: cuánto confiar; consumidores pueden ponderar
        el update del EKF con esta calidad.
      - `curve_hint ∈ {"STRAIGHT","LEFT","RIGHT","UNKNOWN"}`.
      - `blind_mode`: si el detector no ve carril, anuncia el modo en el que
        está operando ("lost", "single_line", etc.) para que el planner
        decida si frena o usa GPS+map.
      - `center_waypoints_body`: tupla de `(x_fwd_m, y_left_m, ψ_rad)` en
        frame body del vehículo (x adelante, y izquierda). Es la
        referencia primaria del MPC cuando se sigue carril por visión —
        equivalente al `flat` de `LaneDetector::get_waypoints` en el repo
        urt-ref. Vacía si no se pudo fitear polinomio.
      - `extrapolated_side`: `"left"` o `"right"` cuando esa línea fue
        sintetizada desplazando la opuesta (single_line). None si se
        vieron ambas o no hay waypoints.
    """

    timestamp: float = 0.0
    source_mode: str = "unknown"
    detected_sides: tuple[str, ...] = field(default_factory=tuple)
    lateral_offset_m: float | None = None
    heading_error_rad: float = 0.0
    direct_error_m: float | None = None
    lane_width_px: float | None = None
    quality: float = 0.0
    curve_hint: str = "STRAIGHT"
    camera_yaw_hint_rad: float | None = None
    camera_yaw_hint_confidence: float = 0.0
    measurement_mode: str = "none"
    direct_error_valid: bool = False
    control_policy_mode: str | None = None
    planner_priority_active: bool = False
    blind_mode: str | None = None
    center_waypoints_body: tuple[tuple[float, float, float], ...] = field(default_factory=tuple)
    left_poly_coeffs: tuple[float, ...] | None = None
    right_poly_coeffs: tuple[float, ...] | None = None
    lane_width_m: float | None = None
    extrapolated_side: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)


def lane_observation_has_visual_path(
    observation: LaneObservation | None,
    *,
    min_quality: float = 0.55,
    min_points: int = 8,
) -> bool:
    """True cuando la observación tiene waypoints visuales utilizables como path primario.

    El gate combina (a) calidad mínima — la default 0.55 acepta single_line con
    polinomio sólido, (b) cantidad mínima de waypoints para que el MPC tenga
    horizonte y (c) que cada waypoint sea finito. NO mira `measurement_mode`:
    en single_line con polinomio bien fitteado los waypoints son tan válidos
    como en two_line (la curvatura la pone el polinomio, no se asume paralelo).
    """
    if observation is None:
        return False
    waypoints = tuple(observation.center_waypoints_body or ())
    if len(waypoints) < int(min_points):
        return False
    if float(observation.quality or 0.0) < float(min_quality):
        return False
    for x, y, psi in waypoints:
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(psi)):
            return False
    return True


def lane_observation_supports_lateral_relocalization(
    observation: LaneObservation | None,
    *,
    min_quality: float = 0.8,
) -> bool:
    """Return True when a lane observation is safe for lateral pose recentering."""
    if observation is None:
        return False
    if not bool(observation.direct_error_valid):
        return False
    if str(observation.measurement_mode or "none") != "two_line":
        return False
    if len(tuple(observation.detected_sides or ())) != 2:
        return False
    if float(observation.quality or 0.0) < float(min_quality):
        return False
    if str(observation.control_policy_mode or "") in _LATERAL_RELOCALIZATION_BLOCKED_CONTROL_MODES:
        return False
    return (
        observation.direct_error_m is not None
        or observation.lateral_offset_m is not None
    )


@dataclass(frozen=True)
class DetectedObject:
    """Detección cruda (sin tracking) de un objeto en el frame.

    Producida por el YOLO para clases no-sign (peatón, otro vehículo,
    obstáculo). Contiene bbox en píxeles + posición proyectada al plano
    de tierra (asume cámara calibrada). Sin ID, sin velocidad: eso lo
    agrega el `MOTTracker` (Fase 5).

    Invariantes:
      - `class_name`: identificador legible ("pedestrian", "car", etc.).
      - `bbox_xyxy`: (x1, y1, x2, y2) en píxeles del frame original.
      - `position_world_xy`: (x, y) en metros, frame ENU del mapa.
        None si no hay calibración suficiente para proyectar.
      - `confidence ∈ [0, 1]`: score del detector.
    """

    timestamp: float = 0.0
    class_id: int = -1
    class_name: str = "unknown"
    confidence: float = 0.0
    bbox_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    position_world_xy: tuple[float, float] | None = None


@dataclass(frozen=True)
class TrackedObject:
    """Objeto con ID estable y velocidad estimada — output del MOTTracker.

    El `MOTTracker` asocia `DetectedObject` frame-a-frame (Hungarian +
    KF constant-velocity) y produce esto. El `BehaviorPlanner` consume
    `tuple[TrackedObject, ...]` para decidir slowdowns/yield/stop.

    Invariantes:
      - `track_id` persiste entre frames mientras la asociación se mantenga.
      - `position_world_xy`: (x, y) en metros, frame ENU del mapa.
      - `velocity_world_xy`: (vx, vy) en m/s, frame ENU. (0, 0) si el track
        es nuevo y aún no hay suficientes muestras.
      - `age_frames`: cuántos frames consecutivos lleva activo. Filtrar
        tracks con `age_frames < 3` reduce falsos positivos.
      - `class_name` se hereda de la primera detección asociada.
    """

    timestamp: float = 0.0
    track_id: int = -1
    class_id: int = -1
    class_name: str = "unknown"
    confidence: float = 0.0
    position_world_xy: tuple[float, float] = (0.0, 0.0)
    velocity_world_xy: tuple[float, float] = (0.0, 0.0)
    age_frames: int = 0
    bbox_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class StoplineObservation:
    """Detección de stopline en BEV, usada para frenar y para relocalizar.

    `pass_event`: cuando `visible` deja de ser True después de haber sido
    True por varios frames, el detector emite un único pass_event con
    metadata para que el EKF/relocalizador haga snap del nodo en la ruta.
    """

    timestamp: float = 0.0
    visible: bool = False
    stable: bool = False
    distance_m: float | None = None
    confidence: float = 0.0
    pass_event: dict[str, Any] | None = None
    expected_node_id: str | None = None
    expected_node_attr: int = 0
    source: str = "none"
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualControlCandidate:
    """LEGACY (pre-Fase 6). Candidato de control salido del lane follower.

    Se mantiene durante el período de transición porque
    `threadControlCoordinator` todavía elige entre múltiples candidates.
    En Fase 6 desaparece: la única salida válida es `MotorCommand` del
    `MotionController`.
    """

    timestamp: float = 0.0
    steering_deg: float | None = None
    speed_cmd: float | None = None
    confidence: float = 0.0
    blind_mode: str | None = None
    source: str = "none"
    direct_error_m: float | None = None
    active: bool = False
    command_source: str = "none"
    computed_steering_deg: float | None = None
    computed_speed_cmd: float | None = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualStateSnapshot:
    """LEGACY. Snapshot por frame del estado del lane follower.

    Lo consume el dashboard para overlay y `threadTracking` para encadenar
    relocalización. Se reemplaza en Fase 4 por canales granulares
    (LaneObservation + BehaviorOutput + PoseEstimate). Mientras tanto, vive.
    """

    timestamp: float = 0.0
    frame_sequence: int = 0
    detection_mode: str = "unknown"
    active: bool = False
    curve_state: str = "STRAIGHT"
    heading_error_rad: float = 0.0
    camera_yaw_hint_rad: float | None = None
    camera_yaw_hint_confidence: float = 0.0
    frame_trace: dict[str, Any] = field(default_factory=dict)
    local_lane_payload: dict[str, Any] = field(default_factory=dict)
    stopline_debug: dict[str, Any] = field(default_factory=dict)
    candidate: VisualControlCandidate | None = None
