# src/localization/ekf/sensor_models.py
#
# Modelos de medición para el EKF7 + builders de matrices de ruido R.
# Cada función devuelve los componentes que `state_filter.py` necesita
# para una medición específica:
#
#   h(x)  → predicción de la medición dada el estado actual.
#   H     → Jacobiano (∂h/∂x) evaluado en x. Es matriz 1×7 ó 2×7
#           dependiendo de la dimensión de la medición.
#   R     → matriz de ruido de medición (1×1 ó 2×2).
#
# Por qué módulo aparte:
#   - Aísla la matemática de cada sensor de la mecánica del filtro.
#     Si mañana querés cambiar GPS por LiDAR-SLAM, tocás solo este
#     archivo; `state_filter.py` no se entera.
#   - Test unitarios más simples: cada modelo se valida con un único
#     fixture sintético sin tener que correr el filtro entero.
#
# Convención de estado (mismo en todo el paquete EKF):
#   x = [px, py, ψ, vx, vy, ω, a]
#   índices:   0   1  2   3   4  5  6
#
#   px, py    — posición ENU local (m).
#   ψ         — yaw (rad), wrapped en [-π, π).
#   vx, vy    — velocidad en marco vehículo (m/s). vx = longitudinal.
#   ω         — velocidad angular (rad/s).
#   a         — aceleración longitudinal (m/s²).

from __future__ import annotations

import math

import numpy as np

# Índices de estado — usados como constantes documentales.
IDX_PX = 0
IDX_PY = 1
IDX_PSI = 2
IDX_VX = 3
IDX_VY = 4
IDX_OMEGA = 5
IDX_A = 6

STATE_DIM = 7


def wrap_angle(angle_rad: float) -> float:
    """Envuelve un ángulo a [-π, π).

    Sólo para residuales (z - h(x)). NUNCA aplicar al estado: el
    Kalman se acopla con el wrap y diverge si lo hacés.
    """
    return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


# --- GPS update ------------------------------------------------------------
# Mide directamente la posición ENU. Llega ya proyectada por
# `coordinate_frame.GpsToEnu`. Modelo trivial: h(x) = [px, py].

def gps_h(x: np.ndarray) -> np.ndarray:
    """h(x) para GPS: extrae (px, py)."""
    return np.array([x[IDX_PX], x[IDX_PY]], dtype=float)


def gps_H() -> np.ndarray:
    """Jacobiano del GPS: 2×7 con 1.0 en (0, IDX_PX) y (1, IDX_PY)."""
    H = np.zeros((2, STATE_DIM), dtype=float)
    H[0, IDX_PX] = 1.0
    H[1, IDX_PY] = 1.0
    return H


def gps_R(sigma_m: float) -> np.ndarray:
    """R 2×2 isotrópica con desviación `sigma_m` (no varianza)."""
    var = float(sigma_m) ** 2
    return np.diag([var, var])


# --- IMU update (ω + a longitudinal) ---------------------------------------
# El IMU del vehículo BFMC publica yaw_rate y aceleración longitudinal
# (acel del vehículo en su eje X). h(x) = [ω, a].

def imu_h(x: np.ndarray) -> np.ndarray:
    return np.array([x[IDX_OMEGA], x[IDX_A]], dtype=float)


def imu_H() -> np.ndarray:
    H = np.zeros((2, STATE_DIM), dtype=float)
    H[0, IDX_OMEGA] = 1.0
    H[1, IDX_A] = 1.0
    return H


def imu_R(sigma_omega: float, sigma_a: float) -> np.ndarray:
    return np.diag([float(sigma_omega) ** 2, float(sigma_a) ** 2])


# --- Encoder de rueda (velocidad longitudinal) -----------------------------
# Mide vx (componente longitudinal de velocidad del vehículo).
# h(x) = vx. Asume que el slip lateral es despreciable — válido a baja
# velocidad en pista BFMC.

def speed_h(x: np.ndarray) -> np.ndarray:
    return np.array([x[IDX_VX]], dtype=float)


def speed_H() -> np.ndarray:
    H = np.zeros((1, STATE_DIM), dtype=float)
    H[0, IDX_VX] = 1.0
    return H


def speed_R(sigma_v: float) -> np.ndarray:
    return np.array([[float(sigma_v) ** 2]], dtype=float)


# --- Lane normal update ----------------------------------------------------
# La cámara ve el carril y proyecta lateral-offset (m) + heading_error
# (rad) al carril. Para usar como measurement de pose necesitamos saber
# el yaw del carril (`path_psi_rad`) — viene del lanelet activo.
#
# Modelo: el offset lateral es la proyección perpendicular del vehículo
# a la línea-meta:
#       lateral = -sin(path_ψ) · (px - px_path) + cos(path_ψ) · (py - py_path)
# Como assumimos que el lanelet pasa por (px_path, py_path) ≈ pose del
# vehículo proyectada al lanelet, el offset que reporta la cámara mide
# directamente el desplazamiento lateral del estado relativo a una
# línea ideal centrada en (px, py). En la práctica, el thread llama
# `update_lane_normal(offset, path_psi)` cuando el vehículo ya está
# anclado al lanelet por route_planner; el filtro corrige usando el
# heading del carril como referencia angular.
#
# Decisión: medimos el heading_error directamente como ψ - path_ψ.
# Ahí el Jacobiano es simple y el wrap se aplica al residual.

def lane_normal_h(x: np.ndarray, path_psi_rad: float) -> np.ndarray:
    """h(x) = [lateral_offset, heading_error] dado el yaw del carril.

    En este modelo el lateral_offset es 0 cuando el vehículo está
    centrado en el lanelet — la cámara reporta el offset signado, así
    que el residual = (medido - 0) lo absorbe directamente.
    """
    return np.array([0.0, x[IDX_PSI] - float(path_psi_rad)], dtype=float)


def lane_normal_H(path_psi_rad: float) -> np.ndarray:
    """Jacobiano: 2×7. Lateral depende de px,py rotados por -path_psi.
    heading_error depende sólo de ψ.
    """
    H = np.zeros((2, STATE_DIM), dtype=float)
    H[0, IDX_PX] = -math.sin(float(path_psi_rad))
    H[0, IDX_PY] = math.cos(float(path_psi_rad))
    H[1, IDX_PSI] = 1.0
    return H


def lane_normal_R(sigma_lat_m: float, sigma_psi_rad: float) -> np.ndarray:
    return np.diag([float(sigma_lat_m) ** 2, float(sigma_psi_rad) ** 2])


# --- Landmark update -------------------------------------------------------
# Cuando matcheamos un sign/stopline contra el mapa, sabemos su posición
# ENU exacta (`landmark_xy`). Si la cámara reporta dónde lo vemos en
# pixeles, lo proyectamos a metros con la calibración y obtenemos un
# pseudo-GPS — exactamente el mismo modelo de medición que GPS, pero
# con R distinta (típicamente más ruidosa lateralmente).
#
# Por simetría con GPS reusamos las funciones h/H, sólo cambia R.

def landmark_h(x: np.ndarray) -> np.ndarray:
    return gps_h(x)


def landmark_H() -> np.ndarray:
    return gps_H()


def landmark_R(sigma_long_m: float, sigma_lat_m: float, vehicle_psi: float) -> np.ndarray:
    """R 2×2 anisotrópica rotada al marco vehículo.

    Un sign visto de frente tiene poca incertidumbre longitudinal y
    mucha lateral (la cámara estima distancia con homografía: cualquier
    error en la pose afecta más Y que X). Construimos R en marco
    vehículo y la rotamos al ENU con la matriz de yaw actual.
    """
    R_local = np.diag([float(sigma_long_m) ** 2, float(sigma_lat_m) ** 2])
    c, s = math.cos(float(vehicle_psi)), math.sin(float(vehicle_psi))
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return rot @ R_local @ rot.T
