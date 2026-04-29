# src/core/types/
#
# Dataclasses inmutables que conforman la "lingua franca" tipada del sistema.
# Cada módulo agrupa tipos por dominio:
#
#   pose.py        — Pose2D (x, y, yaw) y PoseEstimate (pose fusionada del EKF).
#   perception.py  — observaciones de cámara: lane, stopline, snapshot visual
#                    y candidate de control visual (legacy hasta Fase 6).
#   routing.py     — RouteContext con waypoints, lanelet activo y semánticos.
#   control.py     — ControlDecision (decisión final que llega al motor).
#   behavior.py    — BehaviorOutput + ScenarioName (Fase 4: BehaviorPlanner).
#
# Reglas:
#   - Todas las clases son @dataclass(frozen=True). Inmutabilidad evita
#     race conditions cuando varios threads leen el mismo objeto.
#   - default_factory para mutables (list/dict/tuple) — nunca compartir
#     defaults mutables entre instancias.
#   - Sin lógica de negocio. Solo datos + comentarios sobre invariantes.
#
# Re-exporto los símbolos públicos para que los consumidores hagan
# `from src.core.types import PoseEstimate` y no tengan que conocer el
# layout interno de archivos.

from src.core.types.pose import Pose2D, PoseEstimate
from src.core.types.perception import (
    LaneObservation,
    StoplineObservation,
    VisualControlCandidate,
    VisualStateSnapshot,
)
from src.core.types.routing import RouteContext
from src.core.types.control import ControlDecision

__all__ = [
    "Pose2D",
    "PoseEstimate",
    "LaneObservation",
    "StoplineObservation",
    "VisualControlCandidate",
    "VisualStateSnapshot",
    "RouteContext",
    "ControlDecision",
]
