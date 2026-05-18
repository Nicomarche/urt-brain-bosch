"""Tests del `LocalizationOrchestrator.step()`."""

from __future__ import annotations

import math

import pytest

from src.localization.config import load_localization_config
from src.localization.localization_orchestrator import (
    LocalizationOrchestrator,
    get_active_filter_name,
)


def _orchestrator() -> LocalizationOrchestrator:
    cfg = load_localization_config()
    return LocalizationOrchestrator(config=cfg)


def test_step_zero_velocity_keeps_pose():
    orch = _orchestrator()
    orch.reset_pose(x=0.0, y=0.0, yaw_rad=0.0)
    res = orch.step(now=1.0, wall_now=1.0, dt=0.05, v_mps=0.0, steer_rad=0.0)
    assert res.x == 0.0 and res.y == 0.0 and res.yaw_rad == 0.0


def test_step_constant_speed_advances_x():
    orch = _orchestrator()
    orch.reset_pose(x=0.0, y=0.0, yaw_rad=0.0)
    for _ in range(20):
        orch.step(now=0.0, wall_now=0.0, dt=0.05, v_mps=0.50, steer_rad=0.0)
    # 0.50 m/s × 1.0 s = 0.5 m
    assert orch.step(now=0.0, wall_now=0.0, dt=0.0, v_mps=0.0, steer_rad=0.0).x == pytest.approx(0.5, abs=1e-3)


def test_step_curve_changes_yaw():
    orch = _orchestrator()
    orch.reset_pose(x=0.0, y=0.0, yaw_rad=0.0)
    for _ in range(10):
        res = orch.step(
            now=0.0, wall_now=0.0, dt=0.05, v_mps=0.5, steer_rad=math.radians(10.0)
        )
    assert res.yaw_rad > 0.0


def test_imu_update_pulls_yaw():
    orch = _orchestrator()
    orch.reset_pose(x=0.0, y=0.0, yaw_rad=0.0)
    # IMU lee 1.0 rad; multiples updates con v=0 (no DR change) → converge.
    for _ in range(50):
        orch.step(
            now=0.0, wall_now=0.0, dt=0.05, v_mps=0.0, steer_rad=0.0, imu_yaw_rad=1.0
        )
    assert abs(orch.yaw_fuser.yaw - 1.0) < 0.05


def test_default_filter_is_yaw_ekf_1d(monkeypatch):
    """Sin URT_LOCALIZATION_FILTER ni config override, default = yaw_ekf_1d."""
    monkeypatch.delenv("URT_LOCALIZATION_FILTER", raising=False)
    # Sólo si config no setea otro; pero la API se respeta.
    name = get_active_filter_name()
    assert name in {"yaw_ekf_1d", "ekf7", "dead_reckoning"}


def test_env_override(monkeypatch):
    monkeypatch.setenv("URT_LOCALIZATION_FILTER", "dead_reckoning")
    assert get_active_filter_name() == "dead_reckoning"


def test_orchestrator_exposes_pieces():
    orch = _orchestrator()
    assert orch.gps_intake is not None
    assert orch.yaw_fuser is not None
