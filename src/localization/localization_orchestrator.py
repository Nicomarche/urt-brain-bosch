"""LocalizationOrchestrator — composición limpia del pipeline localization.

Plan A3.1: en lugar de meter más cosas al ``pose_estimator_thread``
(1893 LOC), componemos las piezas extractas (YawFuser1D, GpsIntake,
DeadReckoning, LocalizationConfig, opcionalmente EKF7) en una clase que
expone ``step(now, wall_now, dt)``.

Pipeline por step:
  1. Drenar GPS (GpsIntake.receive_latest_fix).
  2. Predict yaw con DR (YawFuser1D.predict).
  3. Si llegó IMU: YawFuser1D.update.
  4. Integrar pose 2D con DR (vehicle model).
  5. Aplicar correcciones (camera lateral, lane, semantic) si están
     disponibles — delegando al legacy mientras la migración A4 progresa.
  6. (Opcional) EKF7 si flag ``LOCALIZATION_FILTER="ekf7"``.

Esta clase es **stateful** — cada llamada avanza el estado interno.
Pensado para que un thread la llame una vez por tick.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from src.localization.config import LocalizationConfig
from src.localization.gps_intake import GpsIntake
from src.localization.yaw_fusion import YawFuser1D


@dataclass(frozen=True)
class OrchestratorTickResult:
    """Snapshot del estado tras un step()."""

    x: float
    y: float
    yaw_rad: float
    speed_mps: float
    yaw_p_cov: float
    last_filter: str  # "yaw_ekf_1d" / "ekf7" / "dead_reckoning"


def get_active_filter_name() -> str:
    """Lee ``config.LOCALIZATION_FILTER`` o env override; default = yaw_ekf_1d."""
    env_override = os.environ.get("URT_LOCALIZATION_FILTER", "").strip()
    if env_override:
        return env_override
    try:
        import config as _cfg

        return str(getattr(_cfg, "LOCALIZATION_FILTER", "yaw_ekf_1d"))
    except Exception:
        return "yaw_ekf_1d"


class LocalizationOrchestrator:
    """Coordina las piezas de localization en un único punto.

    NO publica nada al bus — el thread driver se encarga. Acá vive sólo
    la lógica de "qué corre cuando" para que sea testeable en aislamiento.
    """

    def __init__(
        self,
        *,
        config: LocalizationConfig,
        gps_intake: GpsIntake | None = None,
        yaw_fuser: YawFuser1D | None = None,
        ekf7: Any = None,
    ) -> None:
        self._config = config
        self._gps_intake = gps_intake or GpsIntake()
        self._yaw_fuser = yaw_fuser or YawFuser1D(
            wheelbase_m=config.wheelbase_m,
            max_yaw_rate_rads=config.max_physical_yaw_rate_rads,
            q=config.yaw_ekf_q,
            p_init=config.yaw_ekf_p_init,
            r_steer_k=config.yaw_ekf_r_steer_k,
            r_straight=config.yaw_ekf_r_straight,
            imu_yaw_sign=config.imu_yaw_sign,
        )
        self._ekf7 = ekf7
        self._x = 0.0
        self._y = 0.0
        self._last_filter_name = get_active_filter_name()

    # ─── State control ───────────────────────────────────────────────
    def reset_pose(self, *, x: float, y: float, yaw_rad: float) -> None:
        self._x = float(x)
        self._y = float(y)
        self._yaw_fuser.set_yaw(yaw_rad)

    def calibrate_yaw_offset(self, target_rad: float) -> None:
        self._yaw_fuser.calibrate_offset(target_rad)

    # ─── Main tick ────────────────────────────────────────────────────
    def step(
        self,
        *,
        now: float,
        wall_now: float,
        dt: float,
        v_mps: float,
        steer_rad: float,
        imu_yaw_rad: float | None = None,
    ) -> OrchestratorTickResult:
        """Avanza un tick del pipeline.

        Args:
            now: tiempo monotónico (s).
            wall_now: tiempo wall-clock (s) — usado para stamping de
                PoseEstimate publicado.
            dt: paso de integración (s).
            v_mps: velocidad lineal medida.
            steer_rad: ángulo de dirección medido.
            imu_yaw_rad: lectura cruda del IMU (rad) o ``None`` si no
                llegó este tick.

        Returns:
            ``OrchestratorTickResult`` con la pose post-step.
        """
        # 1) GPS drain — el thread legacy todavía gestiona la fusión XY.
        self._gps_intake.receive_latest_fix()

        # 2) Predict yaw vía dead reckoning.
        self._yaw_fuser.predict(dt=dt, v_mps=v_mps, steer_rad=steer_rad)

        # 3) Update yaw con IMU si disponible.
        if imu_yaw_rad is not None:
            self._yaw_fuser.update(float(imu_yaw_rad), now_s=now)

        # 4) Integrar posición XY con cinemática plana (avance v*dt en yaw).
        import math

        yaw = self._yaw_fuser.yaw
        self._x += float(v_mps) * dt * math.cos(yaw)
        self._y += float(v_mps) * dt * math.sin(yaw)

        # 5) EKF7 opcional (placeholder — el módulo state_filter está aislado)
        if self._last_filter_name == "ekf7" and self._ekf7 is not None:
            try:
                self._ekf7.predict(dt=dt)
            except Exception:
                pass

        return OrchestratorTickResult(
            x=self._x,
            y=self._y,
            yaw_rad=yaw,
            speed_mps=float(v_mps),
            yaw_p_cov=self._yaw_fuser.covariance,
            last_filter=self._last_filter_name,
        )

    # ─── Accessors ────────────────────────────────────────────────────
    @property
    def gps_intake(self) -> GpsIntake:
        return self._gps_intake

    @property
    def yaw_fuser(self) -> YawFuser1D:
        return self._yaw_fuser

    @property
    def active_filter(self) -> str:
        return self._last_filter_name


__all__ = [
    "LocalizationOrchestrator",
    "OrchestratorTickResult",
    "get_active_filter_name",
]
