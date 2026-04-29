# src/core/types/perception.py
#
# Tipos producidos por la capa de percepción visual. La cadena es:
#
#   frame BGR ──► YOLO TensorRT ──► LaneObservation
#                                ──► StoplineObservation
#                                ──► (Fase 5) DetectedObject ──► TrackedObject
#
# LaneObservation y StoplineObservation son las "mediciones" que consume el
# pipeline de localización (relocalization cascade en EKF7) y el
# BehaviorPlanner (caps de velocidad por crosswalk/intersection).
#
# VisualControlCandidate y VisualStateSnapshot son tipos legacy de la era
# Stanley/PID. Sobreviven en Fase 1 porque el thread de control coordinator
# todavía los lee; desaparecen cuando MotorCommand sea la única salida
# (Fase 6).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LaneObservation:
    """Medición de carril emitida por el detector visual frame-a-frame.

    Conventions:
      - `lateral_offset_m`: positivo = vehículo a la izquierda del centro.
        None si la confianza es demasiado baja.
      - `heading_error_rad`: error de yaw del vehículo respecto a la tangente
        del carril (radianes; positivo CCW).
      - `quality ∈ [0, 1]`: cuánto confiar; consumidores pueden ponderar
        el update del EKF con esta calidad.
      - `curve_hint ∈ {"STRAIGHT","LEFT","RIGHT","UNKNOWN"}`.
      - `blind_mode`: si el detector no ve carril, anuncia el modo en el que
        está operando ("lost", "single_line", etc.) para que el planner
        decida si frena o usa GPS+map.
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
    blind_mode: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)


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
