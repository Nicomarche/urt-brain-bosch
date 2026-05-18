"""Tests del filtro `YawFuser1D`."""

from __future__ import annotations

import math

import pytest

from src.localization.yaw_fusion import YawFuser1D


def _fresh(**overrides) -> YawFuser1D:
    kwargs = dict(
        wheelbase_m=0.26,
        max_yaw_rate_rads=math.radians(180.0),
        q=0.01,
        p_init=0.5,
        r_steer_k=0.30,
        r_straight=0.10,
        imu_yaw_sign=1,
    )
    kwargs.update(overrides)
    return YawFuser1D(**kwargs)


def test_init_state_is_zero():
    f = _fresh()
    s = f.snapshot()
    assert s.yaw_rad == 0.0
    assert s.p_cov == 0.5
    assert s.offset_rad == 0.0


def test_predict_at_zero_steer_keeps_yaw():
    f = _fresh()
    f.predict(dt=0.05, v_mps=0.5, steer_rad=0.0)
    assert f.yaw == 0.0


def test_predict_positive_steer_increases_yaw():
    f = _fresh()
    f.predict(dt=0.10, v_mps=0.5, steer_rad=math.radians(10.0))
    assert f.yaw > 0.0


def test_predict_negative_steer_decreases_yaw():
    f = _fresh()
    f.predict(dt=0.10, v_mps=0.5, steer_rad=math.radians(-10.0))
    assert f.yaw < 0.0


def test_max_yaw_rate_clamps_huge_steer():
    """Steer absurdamente grande no produce salto > max_yaw_rate * dt."""
    f = _fresh(max_yaw_rate_rads=1.0)  # 1 rad/s
    f.predict(dt=0.5, v_mps=5.0, steer_rad=math.radians(75.0))
    # rate clipeado a 1 rad/s * 0.5 s = 0.5 rad
    assert abs(f.yaw) <= 0.5 + 1e-9


def test_update_pulls_toward_measurement():
    f = _fresh()
    # Sin predict previo (yaw=0); update con IMU=0.5 rad debe pull hacia 0.5
    f.update(imu_yaw_rad=0.5)
    assert 0.0 < f.yaw < 0.5  # K<1 ⇒ pull parcial


def test_update_reduces_covariance():
    f = _fresh()
    p0 = f.covariance
    f.update(imu_yaw_rad=0.3)
    assert f.covariance < p0


def test_calibrate_offset_stores_value():
    """calibrate_offset guarda un offset != 0 cuando target != yaw_actual."""
    f = _fresh()
    f.set_yaw(0.0)
    f.calibrate_offset(target_rad=1.0)
    assert f.offset == pytest.approx(1.0)


def test_repeated_consistent_imu_converges_to_imu():
    """Con muchos updates donde IMU=0.5 (consistente), yaw converge a ~0.5."""
    f = _fresh()
    for _ in range(50):
        f.update(imu_yaw_rad=0.5)
    assert abs(f.yaw - 0.5) < 0.01


def test_steer_adaptive_R_higher_R_in_curve():
    """En curva, R debe ser mayor → menor K → update tira menos."""
    # Sin curva
    f_straight = _fresh()
    f_straight.predict(dt=0.05, v_mps=0.5, steer_rad=0.0)
    f_straight.update(imu_yaw_rad=1.0)
    # En curva
    f_curve = _fresh()
    f_curve.predict(dt=0.05, v_mps=0.5, steer_rad=math.radians(20.0))
    f_curve.update(imu_yaw_rad=1.0)
    # f_curve confía menos en IMU ⇒ yaw queda más cerca del modelo (no del IMU)
    assert abs(f_curve.yaw - 1.0) > abs(f_straight.yaw - 1.0)


def test_imu_yaw_sign_inverts():
    f = _fresh(imu_yaw_sign=-1)
    f.update(imu_yaw_rad=0.5)
    # IMU 0.5 con sign=-1 ⇒ measurement = -0.5 ⇒ yaw negativo
    assert f.yaw < 0.0


def test_wrap_around_pi():
    """update con yaw 3π/4 e IMU = -3π/4 debe wrappear, no rotar 2π."""
    f = _fresh()
    f.set_yaw(3 * math.pi / 4)
    f.update(imu_yaw_rad=-3 * math.pi / 4)
    # innovación pequeña vía wrap (~+π/2 wrapped a -π/2)
    # el resultado debe estar cerca de ±π, no 0.
    assert abs(abs(f.yaw) - math.pi) < math.pi / 2


def test_reset_restores_state():
    f = _fresh()
    f.predict(dt=0.05, v_mps=0.5, steer_rad=math.radians(20.0))
    f.update(imu_yaw_rad=1.0)
    f.reset(yaw_rad=0.0, p_init=0.5)
    assert f.yaw == 0.0
    assert f.covariance == 0.5
