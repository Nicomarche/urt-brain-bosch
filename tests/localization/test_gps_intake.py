"""Tests del `GpsIntake`."""

from __future__ import annotations

import pytest

from src.localization.gps_intake import GpsFix, GpsIntake


class _FakeSubscriber:
    def __init__(self, fixes: list[dict]) -> None:
        self._buf = list(fixes)

    def receive(self):
        if not self._buf:
            return None
        return self._buf.pop(0)


def test_no_fix_returns_none_in_pure_mode():
    intake = GpsIntake()
    assert intake.receive_latest_fix() is None


def test_push_fix_appears_in_last_fix():
    intake = GpsIntake()
    fix = GpsFix(timestamp=1.0, x=10.0, y=20.0)
    intake.push_fix(fix)
    assert intake.last_fix is fix
    assert intake.history[-1] is fix


def test_subscriber_drains_fifo():
    sub = _FakeSubscriber([
        {"x": 1.0, "y": 2.0, "timestamp": 100.0},
        {"x": 1.1, "y": 2.1, "timestamp": 101.0},
        {"x": 1.2, "y": 2.2, "timestamp": 102.0},
    ])
    intake = GpsIntake(subscriber=sub)
    latest = intake.receive_latest_fix()
    assert latest is not None
    assert latest.x == 1.2  # último drenado
    assert len(intake.history) == 3


def test_history_max_truncates():
    intake = GpsIntake(history_max=2)
    for i in range(5):
        intake.push_fix(GpsFix(timestamp=float(i), x=float(i), y=0.0))
    assert len(intake.history) == 2
    assert intake.history[-1].x == 4.0


def test_calibration_offset_round_trip():
    intake = GpsIntake()
    assert not intake.has_calibration
    result = intake.apply_calibration_offset({"offset_x": 1.5, "offset_y": -2.5})
    assert result == (1.5, -2.5)
    assert intake.has_calibration
    assert intake.calibration_offset == (1.5, -2.5)


def test_calibration_pending_flag():
    intake = GpsIntake()
    intake.try_start_calibration(yaw_rad=0.0, now_s=10.0)
    assert intake.calibration_pending
    intake.apply_calibration_offset({"offset_x": 0.0, "offset_y": 0.0})
    assert not intake.calibration_pending


def test_auto_gps_flag_lifecycle():
    intake = GpsIntake()
    assert not intake.auto_gps_pending
    intake.try_auto_gps_entry(yaw_rad=0.0, now_s=10.0)
    assert intake.auto_gps_pending
    intake.clear_auto_gps()
    assert not intake.auto_gps_pending


def test_parse_fix_rejects_invalid():
    """Payloads no dict o sin números → no fix. La convención subscriber
    es: None = no hay más mensajes (rompe el loop). Invalid dict = skip y sigue.
    """
    sub = _FakeSubscriber([
        {"x": "abc"},  # invalid
        {"x": 1.0, "y": 2.0, "timestamp": 1.0},  # valid
    ])
    intake = GpsIntake(subscriber=sub)
    last = intake.receive_latest_fix()
    assert last is not None
    assert last.x == 1.0
