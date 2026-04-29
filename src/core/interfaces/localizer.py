# src/core/interfaces/localizer.py
#
# Contrato de un localizador. Su responsabilidad: tomar mediciones de N
# sensores (GPS, IMU, encoder de rueda, lane visual, semantic match,
# stopline match) y producir una `PoseEstimate` consistente.
#
# La implementación concreta llega en Fase 2 con `EKF7` (Extended Kalman
# Filter, 7-state planar: [px, py, ψ, vx, vy, ω, a]). Los `update_*`
# son opcionales (no todos los localizadores fusionan los mismos sensores)
# pero `predict`, `state` y `reset` son obligatorios.
#
# Diseño: usar Protocol en lugar de ABC para que los mocks de tests no
# necesiten heredar — un objeto con los métodos correctos pasa el chequeo
# estructural, lo que simplifica la testabilidad.

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.core.types.pose import Pose2D, PoseEstimate


@runtime_checkable
class ILocalizer(Protocol):
    """Localizador que fusiona sensores y publica una pose estimada."""

    def predict(self, dt: float, steer_rad: float = 0.0) -> None:
        """Avanza el estado dt segundos con el modelo de planta del vehículo.

        `steer_rad` es la entrada de control conocida (la última que mandamos
        al actuador). Permite al filtro razonar sobre la dirección esperada
        sin esperar a la siguiente medición.
        """
        ...

    def update_gps(self, xy_m: tuple[float, float], r_m: float) -> None:
        """Update con medición GPS en marco local ENU. r_m = std (no var)."""
        ...

    def update_imu(self, omega_rad_s: float, accel_long_mps2: float, r_omega: float, r_a: float) -> None:
        """Update con velocidad angular + aceleración longitudinal."""
        ...

    def update_speed(self, vx_mps: float, r_v: float) -> None:
        """Update con velocidad longitudinal del encoder de rueda."""
        ...

    def update_lane_normal(self, lateral_offset_m: float, path_psi: float, r_m: float) -> None:
        """Update con offset lateral al carril (cámara → mapa)."""
        ...

    def update_landmark(self, xy_m: tuple[float, float], r_m: float) -> None:
        """Update con match de landmark conocido (semantic sign, stopline)."""
        ...

    def state(self) -> PoseEstimate:
        """Devuelve la pose fusionada actual (sin avanzar el tiempo)."""
        ...

    def reset(self, pose: Pose2D) -> None:
        """Reinicia el filtro con una pose conocida (manual override)."""
        ...
