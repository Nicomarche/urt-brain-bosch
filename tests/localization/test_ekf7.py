# tests/localization/test_ekf7.py
#
# Tests del EKF7. Usamos datos sintéticos para validar:
#   1. predict() avanza el estado según el modelo cinemático.
#   2. update_gps() corrige posición sin desestabilizar yaw.
#   3. update_imu() corrige yaw_rate y aceleración.
#   4. update_speed() colapsa vx hacia el encoder.
#   5. Outliers GPS son descartados por el gate Mahalanobis.
#   6. La covarianza se mantiene simétrica + PSD tras updates.
#   7. Conversión GPS↔ENU es invertible.
#
# Filosofía: cada test exhibe UNA propiedad del filtro. Si un cambio
# futuro rompe una de ellas, el test apunta directo al modelo afectado.

from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.types.pose import Pose2D
from src.localization.ekf.coordinate_frame import GpsToEnu, LatLon
from src.localization.ekf.sensor_models import (
    IDX_A,
    IDX_OMEGA,
    IDX_PSI,
    IDX_PX,
    IDX_PY,
    IDX_VX,
    wrap_angle,
)
from src.localization.ekf.state_filter import EKF7


# ---------- helpers --------------------------------------------------------


def _new_filter(initial: Pose2D | None = None) -> EKF7:
    """Construye un EKF7 con sigmas razonables para tests."""
    return EKF7(initial_pose=initial)


# ---------- predict --------------------------------------------------------


def test_predict_zero_velocity_does_not_move() -> None:
    """Con vx=vy=ω=0, predict() no debe cambiar la pose."""
    ekf = _new_filter(Pose2D(x=10.0, y=5.0, yaw=0.5))
    ekf.predict(dt=0.1)
    assert ekf.x[IDX_PX] == pytest.approx(10.0)
    assert ekf.x[IDX_PY] == pytest.approx(5.0)
    assert ekf.x[IDX_PSI] == pytest.approx(0.5)


def test_predict_constant_velocity_advances_correctly() -> None:
    """vx=1 m/s a yaw=0 durante 1 s ⇒ +1 m en eje X."""
    ekf = _new_filter()
    # Inyectamos vx=1.0 directamente (los updates lo lograrían más lento).
    ekf._x[IDX_VX] = 1.0
    ekf.predict(dt=1.0)
    assert ekf.x[IDX_PX] == pytest.approx(1.0, abs=1e-6)
    assert ekf.x[IDX_PY] == pytest.approx(0.0, abs=1e-6)


def test_predict_constant_velocity_with_yaw() -> None:
    """vx=1, yaw=π/2 durante 1 s ⇒ se mueve +1 m en eje Y."""
    ekf = _new_filter()
    ekf._x[IDX_VX] = 1.0
    ekf._x[IDX_PSI] = math.pi / 2.0
    ekf.predict(dt=1.0)
    assert ekf.x[IDX_PX] == pytest.approx(0.0, abs=1e-6)
    assert ekf.x[IDX_PY] == pytest.approx(1.0, abs=1e-6)


def test_predict_yaw_advances_with_omega() -> None:
    """ω=π rad/s durante 0.5 s ⇒ yaw aumenta π/2."""
    ekf = _new_filter()
    ekf._x[IDX_OMEGA] = math.pi
    ekf.predict(dt=0.5)
    assert ekf.x[IDX_PSI] == pytest.approx(math.pi / 2.0, abs=1e-6)


def test_predict_yaw_wraps() -> None:
    """yaw final debe quedar en [-π, π) tras avanzar más allá."""
    ekf = _new_filter(Pose2D(x=0, y=0, yaw=3.0))  # cerca de π
    ekf._x[IDX_OMEGA] = 1.0
    ekf.predict(dt=1.0)  # yaw → 4.0, debería wrap a 4.0 - 2π
    assert -math.pi <= ekf.x[IDX_PSI] < math.pi


def test_predict_acceleration_increments_vx() -> None:
    """vx ← vx + a·dt."""
    ekf = _new_filter()
    ekf._x[IDX_A] = 0.5
    ekf.predict(dt=2.0)
    assert ekf.x[IDX_VX] == pytest.approx(1.0)


def test_predict_inflates_covariance() -> None:
    """La traza de P debe crecer tras cada predict (Q se acumula)."""
    ekf = _new_filter()
    initial_trace = float(np.trace(ekf.P))
    ekf.predict(dt=0.1)
    after_trace = float(np.trace(ekf.P))
    assert after_trace > initial_trace


# ---------- update_gps -----------------------------------------------------


def test_update_gps_corrects_position() -> None:
    """Después de varios GPS updates en (5, -3), la pose colapsa allí."""
    ekf = _new_filter()
    for _ in range(20):
        ekf.predict(dt=0.05)
        ekf.update_gps(xy_m=(5.0, -3.0), r_m=0.5)
    assert ekf.x[IDX_PX] == pytest.approx(5.0, abs=0.2)
    assert ekf.x[IDX_PY] == pytest.approx(-3.0, abs=0.2)


def test_update_gps_outlier_rejected() -> None:
    """Un GPS muy lejos del estado (|innov|² > gate) se descarta."""
    ekf = _new_filter(Pose2D(x=0, y=0, yaw=0))
    # Bajamos covarianza para que el gate sea estricto.
    ekf._P = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    before = ekf.gps_outliers_rejected
    ekf.update_gps(xy_m=(100.0, 100.0), r_m=0.1)  # 100 sigma de error
    assert ekf.gps_outliers_rejected == before + 1
    # El estado no debe haberse movido a 100, 100.
    assert ekf.x[IDX_PX] == pytest.approx(0.0, abs=1.0)
    assert ekf.x[IDX_PY] == pytest.approx(0.0, abs=1.0)


def test_update_gps_reduces_position_covariance() -> None:
    """Un GPS update reduce la sigma de posición."""
    ekf = _new_filter()
    sigma_before = math.sqrt(ekf.P[IDX_PX, IDX_PX])
    ekf.update_gps(xy_m=(0.0, 0.0), r_m=0.1)
    sigma_after = math.sqrt(ekf.P[IDX_PX, IDX_PX])
    assert sigma_after < sigma_before


# ---------- update_imu / update_speed --------------------------------------


def test_update_imu_corrects_omega() -> None:
    """Tras varios IMU updates con ω=2 rad/s, el estado converge."""
    ekf = _new_filter()
    for _ in range(20):
        ekf.predict(dt=0.05)
        ekf.update_imu(omega_rad_s=2.0, accel_long_mps2=0.0, r_omega=0.05, r_a=0.5)
    assert ekf.x[IDX_OMEGA] == pytest.approx(2.0, abs=0.1)


def test_update_speed_corrects_vx() -> None:
    """update_speed lleva vx hacia el valor del encoder."""
    ekf = _new_filter()
    for _ in range(20):
        ekf.predict(dt=0.05)
        ekf.update_speed(vx_mps=1.5, r_v=0.05)
    assert ekf.x[IDX_VX] == pytest.approx(1.5, abs=0.05)


# ---------- covarianza ----------------------------------------------------


def test_covariance_remains_symmetric_after_updates() -> None:
    """Joseph form mantiene P simétrica con redondeo finito."""
    ekf = _new_filter()
    for _ in range(50):
        ekf.predict(dt=0.05)
        ekf.update_gps(xy_m=(0.5, 0.5), r_m=0.5)
        ekf.update_imu(omega_rad_s=0.0, accel_long_mps2=0.1, r_omega=0.05, r_a=0.3)
        ekf.update_speed(vx_mps=0.5, r_v=0.1)
    P = ekf.P
    np.testing.assert_allclose(P, P.T, atol=1e-9)


def test_covariance_remains_psd_after_updates() -> None:
    """Eigenvalues de P deben ser ≥ 0 (positiva semi-definida)."""
    ekf = _new_filter()
    for _ in range(30):
        ekf.predict(dt=0.05)
        ekf.update_gps(xy_m=(1.0, 0.0), r_m=0.3)
    eigvals = np.linalg.eigvalsh(ekf.P)
    assert np.all(eigvals >= -1e-9)


# ---------- state() / reset() ----------------------------------------------


def test_state_returns_pose_estimate_with_ekf7_mode() -> None:
    """state() debe poblar localization_mode='ekf7' y los campos de pose."""
    ekf = _new_filter(Pose2D(x=2.0, y=3.0, yaw=0.5))
    ekf.predict(dt=0.05)
    pe = ekf.state()
    assert pe.localization_mode == "ekf7"
    assert pe.relocalization_mode == "ekf7"
    assert pe.fused_pose.x == pytest.approx(2.0, abs=0.1)
    assert pe.fused_pose.y == pytest.approx(3.0, abs=0.1)
    assert pe.fused_pose.yaw == pytest.approx(0.5, abs=0.1)
    assert pe.yaw_rad == pytest.approx(0.5, abs=0.1)


def test_reset_restores_initial_state() -> None:
    """reset(pose) debe replantar el filtro en esa pose."""
    ekf = _new_filter()
    ekf.predict(dt=0.1)
    ekf.update_gps(xy_m=(10.0, 10.0), r_m=0.2)
    ekf.reset(Pose2D(x=0.0, y=0.0, yaw=0.0))
    assert ekf.x[IDX_PX] == pytest.approx(0.0)
    assert ekf.x[IDX_PY] == pytest.approx(0.0)
    assert ekf.x[IDX_PSI] == pytest.approx(0.0)


# ---------- coordinate_frame -----------------------------------------------


def test_gps_to_enu_origin_maps_to_zero() -> None:
    """El propio origen GPS debe mapear a (0, 0) en ENU."""
    origin = LatLon(lat_deg=46.7682, lon_deg=23.5870)
    proj = GpsToEnu(origin)
    east, north = proj.to_enu(origin.lat_deg, origin.lon_deg)
    assert east == pytest.approx(0.0, abs=1e-6)
    assert north == pytest.approx(0.0, abs=1e-6)


def test_gps_to_enu_invertible() -> None:
    """ENU → GPS → ENU debe regresar al mismo punto (mm-level)."""
    proj = GpsToEnu(LatLon(lat_deg=46.7682, lon_deg=23.5870))
    east_in, north_in = 12.5, -7.3
    lat, lon = proj.from_enu(east_in, north_in)
    east_out, north_out = proj.to_enu(lat, lon)
    assert east_out == pytest.approx(east_in, abs=1e-3)
    assert north_out == pytest.approx(north_in, abs=1e-3)


def test_gps_to_enu_north_increases_with_latitude() -> None:
    """Subir latitud (más al norte) ⇒ northing positivo."""
    origin = LatLon(lat_deg=46.7682, lon_deg=23.5870)
    proj = GpsToEnu(origin)
    _, north = proj.to_enu(origin.lat_deg + 1e-4, origin.lon_deg)  # ~11 m al norte
    assert north > 0.0


# ---------- wrap_angle -----------------------------------------------------


@pytest.mark.parametrize(
    "angle, expected",
    [
        (0.0, 0.0),
        (math.pi, -math.pi),    # boundary collapses to -π for stability
        (-math.pi, -math.pi),
        (3.0 * math.pi, -math.pi),
        (math.pi / 2.0, math.pi / 2.0),
    ],
)
def test_wrap_angle(angle: float, expected: float) -> None:
    assert wrap_angle(angle) == pytest.approx(expected, abs=1e-9)
