"""TrackingState wrapper module — re-exporta el legacy + agrega ``snapshot()``.

Plan A4: extraemos el acceso al estado compartido entre threads del
``relocalization_thread`` legacy. Por ahora mantenemos la implementación
allí (donde tiene 100+ atributos públicos vivos) y exponemos un módulo
de fachada con un ``snapshot()`` inmutable para tests y telemetría.

Una migración futura puede mover la clase entera acá; este wrapper deja
el path importable hoy.
"""

from __future__ import annotations

from dataclasses import dataclass

# Re-export para que ``from src.localization.tracking_state import TrackingState``
# funcione antes de que se haya mudado físicamente el archivo.
from src.localization.relocalization_thread import TrackingState  # noqa: F401


@dataclass(frozen=True)
class TrackingStateSnapshot:
    """Subconjunto inmutable del TrackingState — útil para tests."""

    x: float
    y: float
    yaw: float
    speed_mps: float
    error_m: float
    heading_rad: float
    relocalization_mode: str
    localization_confidence: float
    initialized: bool


def snapshot(state) -> TrackingStateSnapshot:
    """Construye un snapshot inmutable a partir de un TrackingState live.

    Lee sin lock — el resultado es una foto en el momento; suficiente
    para inspección/test.
    """
    return TrackingStateSnapshot(
        x=float(getattr(state, "x", 0.0)),
        y=float(getattr(state, "y", 0.0)),
        yaw=float(getattr(state, "yaw", 0.0)),
        speed_mps=float(getattr(state, "speed_mps", 0.0)),
        error_m=float(getattr(state, "error_m", 0.0)),
        heading_rad=float(getattr(state, "heading_rad", 0.0)),
        relocalization_mode=str(getattr(state, "relocalization_mode", "")),
        localization_confidence=float(getattr(state, "localization_confidence", 0.0)),
        initialized=bool(getattr(state, "initialized", False)),
    )


__all__ = ["TrackingState", "TrackingStateSnapshot", "snapshot"]
