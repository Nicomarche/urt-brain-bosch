# src/localization/ekf/state_filter.py
#
# EKF7 — Filtro de Kalman Extendido planar de 7 estados.
#
# El EKF es responsable de UNA sola cosa: mantener una estimación
# probabilística del estado del vehículo dada un flujo asíncrono de
# mediciones (GPS, IMU, encoder, lane visual, landmarks). Devuelve un
# `PoseEstimate` consultable por el resto del stack.
#
# Inspirado en `autoware_universe/localization/ekf_localizer` (15-state)
# pero simplificado para BFMC:
#   - Pista plana → no necesitamos pitch/roll.
#   - Vehículo lento (<5 m/s) → bias dinámicos del IMU son despreciables.
#   - Frecuencia 30 Hz → modelo cinemático constant-acceleration es suficiente.
#
# Estado:
#   x = [px, py, ψ, vx, vy, ω, a]    (índices en sensor_models.IDX_*)
#
# Modelo de planta (predict) — discreto, paso dt:
#   px ←  px + vx·cos(ψ)·dt − vy·sin(ψ)·dt
#   py ←  py + vx·sin(ψ)·dt + vy·cos(ψ)·dt
#   ψ  ←  wrap(ψ + ω·dt)
#   vx ←  vx + a·dt
#   vy ←  vy        (no hay aceleración lateral en este modelo)
#   ω  ←  ω         (constant-rate)
#   a  ←  a         (constant-acceleration)
#
# Las dinámicas constant-rate y constant-acceleration son aproximaciones;
# la incertidumbre se inyecta vía Q (ruido de proceso) que sí incluye
# términos en ω y a — así el filtro "permite" cambios pero los penaliza.
#
# Implementa el Protocol `ILocalizer` (ver
# `src/core/interfaces/localizer.py`), por lo que el thread consumidor
# (`src/localization/pose_estimator_thread.py`) puede swappearlo por un
# mock en tests sin tocar el thread.

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from src.core.types.pose import Pose2D, PoseEstimate
from src.localization.ekf import sensor_models as sm
from src.localization.ekf.sensor_models import (
    IDX_A,
    IDX_OMEGA,
    IDX_PSI,
    IDX_PX,
    IDX_PY,
    IDX_VX,
    IDX_VY,
    STATE_DIM,
    wrap_angle,
)

# Mahalanobis gate para outlier rejection. Chi² inverse al 99% para 2 DoF.
# Una medición con d²(z, h(x)) > este threshold se considera outlier y se
# descarta. Evita que un GPS fix muy malo arrastre el filtro.
_MAHALANOBIS_GATE_2DOF = 9.21
_MAHALANOBIS_GATE_1DOF = 6.63  # chi² 99% / 1 DoF


class EKF7:
    """Filtro EKF planar de 7 estados — implementa `ILocalizer`.

    Construir con un valor inicial de pose (P inicial alta = poca
    confianza) y empezar a llamar `predict(dt)` + `update_*` por cada
    medición disponible.

    Pre-condición de uso: el caller (pose_estimator_thread) garantiza
    que las mediciones llegan en orden temporal. Out-of-order updates
    requerirían el mecanismo de delay-compensation del Autoware
    original, que NO está en esta versión (decisión documentada en el
    plan: lo implementamos si el GPS BFMC nos pide y solo si).

    Concurrencia: NO es thread-safe. Cada instancia debe vivir en un
    único thread (típicamente threadPoseEstimator).
    """

    def __init__(
        self,
        *,
        initial_pose: Pose2D | None = None,
        # Sigmas iniciales = "qué tan inseguros estamos al arrancar". Como
        # típicamente el filtro arranca antes del primer fix GPS, position
        # arranca alta (10 m); el primer update colapsa P en una iteración
        # y ya queda bien anclado. Si arrancás con `reset(pose)` después de
        # tener un fix conocido, P se reinicializa a sigmas más estrechos.
        position_sigma_m: float = 10.0,
        yaw_sigma_rad: float = math.radians(45.0),
        velocity_sigma_mps: float = 1.0,
        omega_sigma_rad_s: float = math.radians(30.0),
        accel_sigma_mps2: float = 1.0,
        # Q diagonal — ruido de proceso. Más alto = más reactivo, menos suave.
        # Calibrar contra logs reales en pista; estos defaults son punto de partida.
        q_pos_mps2: float = 0.1,
        q_yaw_rad_s: float = math.radians(5.0),
        q_vel_mps2: float = 0.5,
        q_omega_rad_s2: float = math.radians(10.0),
        q_accel_mps3: float = 1.0,
    ) -> None:
        self._x = np.zeros(STATE_DIM, dtype=float)
        if initial_pose is not None:
            self._x[IDX_PX] = float(initial_pose.x)
            self._x[IDX_PY] = float(initial_pose.y)
            self._x[IDX_PSI] = wrap_angle(float(initial_pose.yaw))

        # Covarianza inicial: incertidumbre alta hasta que lleguen
        # mediciones que la achiquen.
        self._P = np.diag([
            position_sigma_m ** 2,
            position_sigma_m ** 2,
            yaw_sigma_rad ** 2,
            velocity_sigma_mps ** 2,
            velocity_sigma_mps ** 2,
            omega_sigma_rad_s ** 2,
            accel_sigma_mps2 ** 2,
        ])

        # Densidades espectrales del ruido de proceso. Q efectiva se
        # construye en `predict` escalando con dt (Q ≈ G·diag(σ²)·G.T·dt).
        # Por simplicidad y estabilidad numérica usamos una Q diagonal
        # multiplicada por dt. No es exacta para un sistema continuo →
        # discreto, pero sí suficiente a 30 Hz.
        self._q_diag = np.array([
            q_pos_mps2 ** 2,
            q_pos_mps2 ** 2,
            q_yaw_rad_s ** 2,
            q_vel_mps2 ** 2,
            q_vel_mps2 ** 2,
            q_omega_rad_s2 ** 2,
            q_accel_mps3 ** 2,
        ], dtype=float)

        # Timestamp del último predict — sirve para diagnosticar
        # dt anormales y para el campo `timestamp` del PoseEstimate.
        self._last_predict_time: Optional[float] = None
        # Métricas observables desde fuera (loggeo / dashboard).
        self._gps_outliers_rejected = 0
        self._last_gps_innovation_norm = 0.0
        self._last_gps_fix_quality = 0.0

    # ---- ILocalizer.predict --------------------------------------------
    def predict(self, dt: float, steer_rad: float = 0.0) -> None:
        """Avanza el estado dt segundos con el modelo cinemático.

        `steer_rad` se acepta para compatibilidad con `ILocalizer`, pero
        este modelo cinemático puro no lo usa: la influencia del steering
        sobre ω se mide vía el IMU (que sí entra como update). Si en el
        futuro queremos un modelo bicycle más cerrado, agregamos la
        relación δ → ω·v / wheelbase aquí.
        """
        if dt <= 0.0 or not math.isfinite(dt):
            return

        psi = self._x[IDX_PSI]
        vx = self._x[IDX_VX]
        vy = self._x[IDX_VY]
        omega = self._x[IDX_OMEGA]
        a = self._x[IDX_A]

        # Modelo de planta (estado predicho).
        cos_psi = math.cos(psi)
        sin_psi = math.sin(psi)
        x_pred = self._x.copy()
        x_pred[IDX_PX] += (vx * cos_psi - vy * sin_psi) * dt
        x_pred[IDX_PY] += (vx * sin_psi + vy * cos_psi) * dt
        x_pred[IDX_PSI] = wrap_angle(psi + omega * dt)
        x_pred[IDX_VX] += a * dt
        # vy, omega, a → constantes según modelo.

        # Jacobiano F = ∂f/∂x evaluado en x.
        # Sólo las celdas no-triviales (el resto del modelo es identidad).
        F = np.eye(STATE_DIM, dtype=float)
        F[IDX_PX, IDX_PSI] = (-vx * sin_psi - vy * cos_psi) * dt
        F[IDX_PX, IDX_VX] = cos_psi * dt
        F[IDX_PX, IDX_VY] = -sin_psi * dt
        F[IDX_PY, IDX_PSI] = (vx * cos_psi - vy * sin_psi) * dt
        F[IDX_PY, IDX_VX] = sin_psi * dt
        F[IDX_PY, IDX_VY] = cos_psi * dt
        F[IDX_PSI, IDX_OMEGA] = dt
        F[IDX_VX, IDX_A] = dt

        # Q discreto (aproximación diagonal · dt). Mejor: integrate Q
        # continuo con el método de Van Loan, pero a 30 Hz la diferencia
        # es marginal vs el ruido de medición.
        Q = np.diag(self._q_diag) * dt

        # Covarianza propagada.
        self._P = F @ self._P @ F.T + Q
        self._x = x_pred

        self._last_predict_time = time.time()

    # ---- helpers de update --------------------------------------------
    def _apply_update(
        self,
        z: np.ndarray,
        z_pred: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        wrap_indices: tuple[int, ...] = (),
        gate: float | None = None,
    ) -> bool:
        """Aplica la corrección de Kalman (Joseph form) si pasa el gate.

        - `wrap_indices` indica qué componentes del residual son ángulos
          y deben envolverse a [-π, π) antes de medir distancia.
        - `gate` es el umbral de Mahalanobis chi² al 99% (None = sin gate).
        - Devuelve True si la corrección se aplicó, False si fue rechazada.

        Joseph form: P ← (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ. Más caro
        computacionalmente pero numéricamente estable (mantiene P
        simétrica y semi-definida positiva incluso con redondeo).
        """
        # Residual (innovation) y wrap de los ángulos.
        y = z - z_pred
        for idx in wrap_indices:
            y[idx] = wrap_angle(float(y[idx]))

        # Innovation covariance.
        S = H @ self._P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False

        # Outlier gate Mahalanobis.
        if gate is not None:
            mahal_sq = float(y @ S_inv @ y)
            if mahal_sq > gate:
                return False

        # Ganancia y actualización.
        K = self._P @ H.T @ S_inv
        self._x = self._x + K @ y
        # Mantener ψ en [-π, π) tras el update.
        self._x[IDX_PSI] = wrap_angle(self._x[IDX_PSI])

        # Joseph form para P.
        I = np.eye(STATE_DIM, dtype=float)
        IKH = I - K @ H
        self._P = IKH @ self._P @ IKH.T + K @ R @ K.T
        return True

    # ---- ILocalizer.update_* -------------------------------------------
    def update_gps(self, xy_m: tuple[float, float], r_m: float) -> None:
        """Update con medición ENU (x, y) en metros."""
        z = np.array([float(xy_m[0]), float(xy_m[1])], dtype=float)
        z_pred = sm.gps_h(self._x)
        H = sm.gps_H()
        R = sm.gps_R(r_m)
        accepted = self._apply_update(z, z_pred, H, R, gate=_MAHALANOBIS_GATE_2DOF)
        innov = float(np.linalg.norm(z - z_pred))
        self._last_gps_innovation_norm = innov
        if not accepted:
            self._gps_outliers_rejected += 1
            # Calidad ↓ cuando rebotamos GPS. Se recupera con cada update aceptado.
            self._last_gps_fix_quality = max(0.0, self._last_gps_fix_quality - 0.1)
        else:
            # Calidad inversamente proporcional al residual normalizado.
            self._last_gps_fix_quality = float(
                np.clip(1.0 / (1.0 + innov / max(r_m, 1e-3)), 0.0, 1.0)
            )

    def update_imu(
        self,
        omega_rad_s: float,
        accel_long_mps2: float,
        r_omega: float,
        r_a: float,
    ) -> None:
        z = np.array([float(omega_rad_s), float(accel_long_mps2)], dtype=float)
        z_pred = sm.imu_h(self._x)
        self._apply_update(z, z_pred, sm.imu_H(), sm.imu_R(r_omega, r_a))

    def update_speed(self, vx_mps: float, r_v: float) -> None:
        z = np.array([float(vx_mps)], dtype=float)
        z_pred = sm.speed_h(self._x)
        self._apply_update(z, z_pred, sm.speed_H(), sm.speed_R(r_v))

    def update_lane_normal(
        self,
        lateral_offset_m: float,
        path_psi: float,
        r_m: float,
    ) -> None:
        # Tomamos lateral_offset signado (cámara): +derecha = vehículo
        # desplazado a derecha del lanelet ⇒ pierde proyección lateral
        # con signo negativo en el modelo (ver sensor_models).
        z = np.array([float(lateral_offset_m), 0.0], dtype=float)
        z_pred = sm.lane_normal_h(self._x, path_psi)
        H = sm.lane_normal_H(path_psi)
        # Heading_error: la cámara no lo separa explícitamente; lo dejamos
        # con r mayor para que el filtro confíe más en el offset.
        R = sm.lane_normal_R(r_m, max(r_m * 5.0, 0.05))
        self._apply_update(z, z_pred, H, R, wrap_indices=(1,))

    def update_landmark(self, xy_m: tuple[float, float], r_m: float) -> None:
        z = np.array([float(xy_m[0]), float(xy_m[1])], dtype=float)
        z_pred = sm.landmark_h(self._x)
        # Asumimos isotropía simple (sigma_long ≈ sigma_lat = r_m). Si en
        # algún sensor querés anisotropía rotada, llamás a
        # sensor_models.landmark_R(sigma_long, sigma_lat, ψ) directo y
        # construís el update sin pasar por update_landmark.
        R = sm.landmark_R(r_m, r_m, self._x[IDX_PSI])
        self._apply_update(z, z_pred, sm.landmark_H(), R, gate=_MAHALANOBIS_GATE_2DOF)

    # ---- ILocalizer.state / reset --------------------------------------
    def state(self) -> PoseEstimate:
        """Devuelve la pose fusionada actual (sin avanzar el tiempo).

        Mapea el estado interno [px, py, ψ, vx, ...] al `PoseEstimate`
        público:
          - `fused_pose` = pose 2D corregida.
          - `raw_pose`   = misma pose; el dead-reckoning ya está
                            incorporado al fusionado (no hay pose cruda
                            independiente cuando el EKF7 está activo).
          - `localization_mode` = "ekf7" (estable mientras el filtro
                                  tenga al menos un update).
          - `gps_fix_quality`   = derivada del residual GPS más reciente.
          - `localization_confidence` = inversamente proporcional a la
                                        traza de la covarianza de
                                        posición — fácil de leer desde
                                        el dashboard.
        """
        pose2d = Pose2D(
            x=float(self._x[IDX_PX]),
            y=float(self._x[IDX_PY]),
            yaw=float(self._x[IDX_PSI]),
        )
        # Confianza heurística: 1.0 cuando σ_pos ≪ 0.1 m, decae con la
        # raíz de la traza. Buena señal cualitativa para el dashboard.
        pos_var = max(self._P[IDX_PX, IDX_PX], 0.0) + max(self._P[IDX_PY, IDX_PY], 0.0)
        pos_sigma = math.sqrt(pos_var)
        confidence = float(np.clip(1.0 / (1.0 + pos_sigma * 5.0), 0.0, 1.0))
        return PoseEstimate(
            timestamp=float(self._last_predict_time or time.time()),
            raw_pose=pose2d,
            fused_pose=pose2d,
            speed_mps=float(self._x[IDX_VX]),
            yaw_rad=float(self._x[IDX_PSI]),
            speed_source="ekf7",
            localization_confidence=confidence,
            relocalization_mode="ekf7",
            localization_mode="ekf7",
            gps_fix_quality=float(self._last_gps_fix_quality),
        )

    def reset(self, pose: Pose2D) -> None:
        """Reinicia el filtro con una pose conocida (manual override)."""
        self._x = np.zeros(STATE_DIM, dtype=float)
        self._x[IDX_PX] = float(pose.x)
        self._x[IDX_PY] = float(pose.y)
        self._x[IDX_PSI] = wrap_angle(float(pose.yaw))
        self._P = np.diag([0.5 ** 2, 0.5 ** 2, math.radians(5.0) ** 2,
                           0.2 ** 2, 0.2 ** 2, math.radians(10.0) ** 2,
                           0.2 ** 2])
        self._last_predict_time = time.time()
        self._gps_outliers_rejected = 0
        self._last_gps_innovation_norm = 0.0
        self._last_gps_fix_quality = 0.0

    # ---- introspección para tests / dashboard --------------------------
    @property
    def x(self) -> np.ndarray:
        """Estado actual (copia inmutable; modificarla no afecta al filtro)."""
        return self._x.copy()

    @property
    def P(self) -> np.ndarray:
        return self._P.copy()

    @property
    def gps_outliers_rejected(self) -> int:
        return self._gps_outliers_rejected
