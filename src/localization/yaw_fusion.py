"""YawFuser1D — fusión 1D del heading usando IMU + dead reckoning.

Plan A2.1: extracción de la lógica de fusión de yaw del
``pose_estimator_thread`` a un módulo aislado y testeable.

Modelo:
  * Predicción: dead reckoning con bicicleta cinemática
    ``ψ_{k+1} = ψ_k + (v_k / wheelbase) * tan(δ_k) * dt`` (clipeado a un
    máximo físico para inmunizar contra outliers de encoder).
  * Update: IMU yaw lo lleva como medición directa con offset calibrable.
  * R adapta: si el steer es ≈ 0 (recta), la IMU pesa más; en curva
    cerrada, el modelo cinemático manda.

API:
  * ``predict(dt, v_mps, steer_rad)`` — RK + clipping.
  * ``update(imu_yaw_rad, now_s)`` — Kalman 1D update.
  * ``calibrate_offset(target_rad)`` — guarda offset IMU.
  * ``reset(p_init)`` — re-inicializa P.
  * ``snapshot() -> YawFuserSnapshot`` — estado serializable para tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class YawFuserSnapshot:
    """Estado interno del YawFuser1D — sólo lectura, para tests/telemetría."""

    yaw_rad: float
    p_cov: float
    offset_rad: float
    last_imu_t_s: float | None


def _wrap_pi(angle_rad: float) -> float:
    """Wrap a [-pi, pi)."""
    while angle_rad >= math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


class YawFuser1D:
    """Filtro 1D de yaw con steer-adaptive R."""

    def __init__(
        self,
        *,
        wheelbase_m: float,
        max_yaw_rate_rads: float,
        q: float,
        p_init: float,
        r_steer_k: float,
        r_straight: float,
        imu_yaw_sign: int = 1,
    ) -> None:
        self._wheelbase_m = float(wheelbase_m)
        self._max_yaw_rate_rads = float(max_yaw_rate_rads)
        self._q = float(q)
        self._p_init = float(p_init)
        self._r_steer_k = float(r_steer_k)
        self._r_straight = float(r_straight)
        self._imu_yaw_sign = int(imu_yaw_sign)

        self._yaw_rad: float = 0.0
        self._p: float = float(p_init)
        self._offset_rad: float = 0.0
        self._last_imu_t_s: float | None = None
        # Steering del último predict (sirve para escalar R en update).
        self._last_steer_rad: float = 0.0

    # ─── State control ────────────────────────────────────────────────
    def reset(self, *, yaw_rad: float = 0.0, p_init: float | None = None) -> None:
        self._yaw_rad = float(yaw_rad)
        self._p = float(p_init) if p_init is not None else self._p_init
        self._last_imu_t_s = None
        self._last_steer_rad = 0.0

    def calibrate_offset(self, target_rad: float) -> None:
        """Ajusta el offset IMU para que el próximo update lea ``target_rad``.

        Útil al cuadrar yaw inicial contra el GPS o un mapa.
        """
        self._offset_rad = _wrap_pi(float(target_rad) - self._yaw_rad)

    def set_yaw(self, yaw_rad: float) -> None:
        """Override absoluto (relocalización forzada)."""
        self._yaw_rad = float(yaw_rad)
        self._p = float(self._p_init)

    # ─── Predict / update ────────────────────────────────────────────
    def predict(self, dt: float, v_mps: float, steer_rad: float) -> float:
        """Avanza el yaw con la cinemática y suma Q a P.

        Returns:
            Yaw posterior tras predict (rad).
        """
        dt = max(0.0, float(dt))
        v = float(v_mps)
        steer = float(steer_rad)
        self._last_steer_rad = steer

        if self._wheelbase_m > 1e-6 and abs(v) > 1e-6:
            raw_rate = (v / self._wheelbase_m) * math.tan(steer)
            # Clamp a la tasa máxima física (anti-spike por encoder rumiando).
            rate = max(-self._max_yaw_rate_rads, min(self._max_yaw_rate_rads, raw_rate))
        else:
            rate = 0.0
        self._yaw_rad = _wrap_pi(self._yaw_rad + rate * dt)
        # Process noise — escala con dt para que Q sea Q*dt como en CV-EKF.
        self._p = self._p + self._q * max(dt, 0.0)
        return self._yaw_rad

    def update(self, imu_yaw_rad: float, now_s: float | None = None) -> float:
        """Kalman 1D update con measurement = imu_yaw + offset.

        R adapta con steer: en recta R≈r_straight, en curva R≈r_steer_k.
        """
        measurement = self._imu_yaw_sign * float(imu_yaw_rad) + self._offset_rad
        # Innovación con wrap (evita salto 2π).
        innovation = _wrap_pi(measurement - self._yaw_rad)
        # Steer-adaptive R: |steer| pequeño → IMU sirve mucho (R bajo);
        # |steer| grande → modelo cinemático manda (R alto).
        steer_factor = abs(self._last_steer_rad) / max(1e-6, self._r_steer_k)
        r = self._r_straight + steer_factor * self._r_steer_k
        s = self._p + r
        k = self._p / s if s > 0 else 0.0
        self._yaw_rad = _wrap_pi(self._yaw_rad + k * innovation)
        self._p = (1.0 - k) * self._p
        if now_s is not None:
            self._last_imu_t_s = float(now_s)
        return self._yaw_rad

    # ─── Read-only accessors ─────────────────────────────────────────
    @property
    def yaw(self) -> float:
        return self._yaw_rad

    @property
    def covariance(self) -> float:
        return self._p

    @property
    def offset(self) -> float:
        return self._offset_rad

    def snapshot(self) -> YawFuserSnapshot:
        return YawFuserSnapshot(
            yaw_rad=self._yaw_rad,
            p_cov=self._p,
            offset_rad=self._offset_rad,
            last_imu_t_s=self._last_imu_t_s,
        )


__all__ = ["YawFuser1D", "YawFuserSnapshot"]
