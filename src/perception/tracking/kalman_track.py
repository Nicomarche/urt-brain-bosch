# src/perception/tracking/kalman_track.py
#
# `TrackState` — KF de constant-velocity en plano BEV-mundo (x, y).
# Es el "ladrillo" del MOTTracker: cada track activo es un TrackState +
# metadata (track_id, class_name, age, hits, last_seen).
#
# Modelo:
#   estado:        [x, y, vx, vy]^T
#   transición F:  [[1, 0, dt, 0],
#                   [0, 1, 0, dt],
#                   [0, 0, 1, 0],
#                   [0, 0, 0, 1]]
#   medición H:    [[1, 0, 0, 0],
#                   [0, 1, 0, 0]]   ← observamos solo (x, y)
#   ruido proceso Q: identidad escalada por `process_noise_std² * dt`
#                   en componentes de velocidad — modelo CV con
#                   aceleración como ruido gaussiano.
#   ruido medición R: diag(measure_noise_std², measure_noise_std²) — la
#                   detección viene con incertidumbre ~10–30 cm a 3 m
#                   con calibración BFMC.
#
# Por qué KF y no CTRV / CTRA: a la velocidad del peatón (≤ 2 m/s) y la
# duración de los tracks (~3–10 s) la curvatura es ruido, no señal. CV es
# suficiente y muchísimo más estable.

from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter

from src.core.types.perception import DetectedObject

# Defaults razonables — los puede sobreescribir el MOTTracker desde config.
_DEFAULT_PROCESS_NOISE_STD = 0.5  # m/s² ≈ aceleración típica de peatón
_DEFAULT_MEASURE_NOISE_STD = 0.20  # m, error de proyección 2D-BEV


class TrackState:
    """KF de constant-velocity 2D + metadata del track.

    Invariantes:
      - `track_id` es asignado por el MOTTracker, único entre los activos.
      - `hits >= 1`: cada vez que un detection se asocia, `hits += 1`.
      - `time_since_update`: frames desde el último match exitoso. El
        tracker borra tracks con `time_since_update > max_age`.
      - `class_name` se hereda de la primera detección y NO cambia (los
        cambios de clase consecutivos son ruido, no realidad).
    """

    __slots__ = (
        "_kf",
        "track_id",
        "class_id",
        "class_name",
        "confidence",
        "hits",
        "time_since_update",
        "age",
        "last_timestamp",
    )

    def __init__(
        self,
        track_id: int,
        initial_xy: tuple[float, float],
        class_id: int,
        class_name: str,
        confidence: float,
        timestamp: float,
        *,
        process_noise_std: float = _DEFAULT_PROCESS_NOISE_STD,
        measure_noise_std: float = _DEFAULT_MEASURE_NOISE_STD,
    ) -> None:
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # F y H se setean per-step (F depende de dt).
        kf.F = np.eye(4)
        kf.H = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0]])

        # Estado inicial: posición observada, velocidad 0 (asumimos quieto
        # hasta probar lo contrario — el next update mide el desplazamiento).
        kf.x = np.array([initial_xy[0], initial_xy[1], 0.0, 0.0])

        # Covarianza inicial: posición razonablemente cierta, velocidad muy
        # incierta (aún no medimos).
        kf.P = np.diag([
            measure_noise_std ** 2,
            measure_noise_std ** 2,
            5.0 ** 2,  # vx
            5.0 ** 2,  # vy
        ])

        # R y Q: R fijo (calibración constante); Q se reconstruye en cada
        # predict() porque depende de dt.
        kf.R = np.eye(2) * (measure_noise_std ** 2)
        # Guardamos el std para reconstruir Q por step.
        self._kf = kf
        self._kf._process_noise_std = float(process_noise_std)  # type: ignore[attr-defined]

        self.track_id: int = int(track_id)
        self.class_id: int = int(class_id)
        self.class_name: str = str(class_name)
        self.confidence: float = float(confidence)
        # Empieza con 1: la detección que creó el track ya cuenta.
        self.hits: int = 1
        self.time_since_update: int = 0
        self.age: int = 1
        self.last_timestamp: float = float(timestamp)

    # ----------------------------------------------------------------
    # KF predict / update
    # ----------------------------------------------------------------
    def predict(self, dt: float) -> None:
        """Avanza el estado dt segundos. Llamar UNA vez por frame."""
        if dt <= 0:
            return
        # F depende de dt — modelo CV.
        self._kf.F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        sigma_a = float(getattr(self._kf, "_process_noise_std", _DEFAULT_PROCESS_NOISE_STD))
        # Q clásica de CV: σ_a² × G G^T con G = [dt²/2, dt²/2, dt, dt]^T,
        # acoplado por componente.
        q_pos = (dt ** 2 / 2.0) ** 2 * sigma_a ** 2
        q_pv = (dt ** 2 / 2.0) * dt * sigma_a ** 2
        q_v = dt ** 2 * sigma_a ** 2
        self._kf.Q = np.array([
            [q_pos, 0.0, q_pv, 0.0],
            [0.0, q_pos, 0.0, q_pv],
            [q_pv, 0.0, q_v, 0.0],
            [0.0, q_pv, 0.0, q_v],
        ])

        self._kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: DetectedObject) -> None:
        """Incorpora una detección (debe tener position_world_xy)."""
        pos = detection.position_world_xy
        if pos is None:
            return
        z = np.asarray(pos, dtype=float)
        self._kf.update(z)

        # Refresh metadata desde la detección asociada.
        self.confidence = float(detection.confidence)
        self.last_timestamp = float(detection.timestamp)
        self.hits += 1
        self.time_since_update = 0

    # ----------------------------------------------------------------
    # Accessors
    # ----------------------------------------------------------------
    def position(self) -> tuple[float, float]:
        return float(self._kf.x[0]), float(self._kf.x[1])

    def velocity(self) -> tuple[float, float]:
        return float(self._kf.x[2]), float(self._kf.x[3])

    def state(self) -> np.ndarray:
        """Estado completo [x, y, vx, vy] como copia."""
        return self._kf.x.copy()

    def covariance(self) -> np.ndarray:
        """Matriz de covarianza P (4x4) como copia."""
        return self._kf.P.copy()

    def position_covariance(self) -> np.ndarray:
        """Submatriz 2x2 de la covarianza posicional. Útil para Mahalanobis."""
        return self._kf.P[:2, :2].copy()

    def is_confirmed(self, min_hits: int = 3) -> bool:
        """Track "confirmado" cuando lleva al menos `min_hits` matches.
        Filtrar por esto reduce falsos positivos del detector."""
        return self.hits >= int(min_hits)

    def is_stale(self, max_age_frames: int) -> bool:
        """Track muerto: no se actualiza hace más de `max_age_frames` frames."""
        return self.time_since_update > int(max_age_frames)
