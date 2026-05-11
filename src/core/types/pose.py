# src/core/types/pose.py
#
# Tipos de pose 2D. Pose2D es la representación geométrica pura (x, y, yaw)
# usada en cualquier punto del sistema que necesita coordenadas planares.
# PoseEstimate es la salida pública del localizador (hoy EKF de yaw, pasa a
# EKF7 en Fase 2): además de la pose fusionada incluye velocidad, modo de
# relocalización y métricas de confianza para que los consumidores aguas
# abajo (BehaviorPlanner, MotionController, dashboard) puedan razonar sobre
# la calidad de la estimación.
#
# Convención de unidades:
#   - x, y           : metros, marco local del mapa Lanelet/OSM.
#                      En este proyecto se preserva la orientación visual del
#                      dashboard/simulador, así que `y` crece hacia abajo.
#   - yaw / yaw_rad  : radianes, 0 hacia +x, sentido horario (CW) en ese frame.
#   - speed_mps      : m/s; positivo hacia adelante del vehículo.
#   - steer_rad      : radianes; el ángulo de las ruedas, no del volante.
#   - timestamp      : segundos UNIX (time.time()), no monotonic.

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Pose2D:
    """Coordenadas planares puras: la primitiva geométrica del sistema."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


@dataclass(frozen=True)
class VisualLaneMatch:
    """Resultado del matcher visual 2D contra la ruta/lanelets esperados."""

    lanelet_id: str | None = None
    lateral_error_m: float = 0.0
    yaw_error_rad: float = 0.0
    near_lateral_error_m: float = 0.0
    near_yaw_error_rad: float = 0.0
    max_abs_lateral_error_m: float = 0.0
    max_abs_yaw_error_rad: float = 0.0
    sample_count: int = 0
    score: float = 1.0
    confidence: float = 0.0
    accepted: bool = False
    reason: str = "unchecked"
    source: str = "visual_lane_route_match"


@dataclass(frozen=True)
class PoseEstimate:
    """Salida pública del localizador.

    Invariantes:
      - `fused_pose` SIEMPRE es la versión "buena" para consumir; `raw_pose`
        es solo telemetría/debug del dead-reckoning antes de fusionar.
      - `localization_confidence ∈ [0, 1]`. Un BehaviorPlanner razonable
        baja velocidad si esto cae bajo 0.3 sostenidamente.
      - `relocalization_mode` toma valores como "dead_reckoning",
        "lane_visual", "semantic", "stopline_match", "ekf7" (Fase 2).
      - `speed_feedback_age_s` y `speed_command_age_s` permiten al consumidor
        descartar lecturas viejas. None = nunca recibido.
    """

    timestamp: float = 0.0
    raw_pose: Pose2D = field(default_factory=Pose2D)
    fused_pose: Pose2D = field(default_factory=Pose2D)
    speed_mps: float = 0.0
    yaw_rad: float = 0.0
    steer_rad: float = 0.0
    speed_source: str = "none"
    speed_feedback_age_s: float | None = None
    speed_command_age_s: float | None = None
    localization_confidence: float = 0.0
    relocalization_mode: str = "dead_reckoning"
    last_relocalization_source: str = "none"
    last_relocalization_error_m: float = 0.0
    raw_lateral_error_m: float = 0.0
    lane_measurement_reliable: bool = False
    camera_lateral_correction_m: float = 0.0
    imu_received: bool = False

    # Fase 2 (EKF7) — campos nuevos que enriquecen el estado del filtro:
    #   `localization_mode` se setea a "ekf7" cuando la fusión proviene
    #     del Filtro Kalman Extendido completo. Coexiste con
    #     `relocalization_mode`, que documenta el subsistema activo de
    #     reanclaje (lane_visual, semantic, stopline_match). Los
    #     consumidores que sólo necesitan saber "es estimación robusta o
    #     dead_reckoning" leen `localization_mode`.
    #   `gps_fix_quality ∈ [0, 1]` deriva del residual GPS más reciente
    #     normalizado por su sigma — 1.0 = match perfecto, 0.0 = el
    #     EKF descartó el GPS por outlier en los últimos updates.
    localization_mode: str = "dead_reckoning"
    gps_fix_quality: float = 0.0
    visual_lane_match: VisualLaneMatch | None = None
