# tests/control/test_safety_gate.py
#
# Tests del SafetyGate. Validamos:
#   1. Comando válido y timestamps frescos pasa sin tocarse.
#   2. Comando None → fallback "no_motor_command".
#   3. MotorCommand.valid=False → fallback con la razón del comando.
#   4. behavior_timestamp None → fallback "no_behavior_output".
#   5. behavior stale (> timeout) → fallback "behavior_stale_*".
#   6. pose_timestamp None → fallback "no_pose_estimate".
#   7. pose stale → fallback "pose_stale_*".
#   8. fallback() directo emite STOP con razón "manual".
#   9. last_reason refleja la última decisión.

from __future__ import annotations

import time

import pytest

from src.control.safety_gate import SafetyGate, SafetyGateConfig
from src.core.types.control import MotorCommand


def _good_cmd(now: float) -> MotorCommand:
    return MotorCommand(
        timestamp=now,
        steering_deg=5.0,
        speed_mps=0.3,
        valid=True,
        source="motion_controller",
        reason="",
    )


def test_passthrough_when_everything_fresh() -> None:
    gate = SafetyGate()
    now = 1000.0
    cmd = _good_cmd(now)
    out = gate.evaluate(
        motor_command=cmd,
        behavior_timestamp=now - 0.1,
        pose_timestamp=now - 0.05,
        now=now,
    )
    assert out is cmd
    assert out.source == "motion_controller"
    assert gate.last_reason == "passthrough"


def test_no_motor_command_falls_back() -> None:
    gate = SafetyGate()
    now = 1000.0
    out = gate.evaluate(
        motor_command=None,
        behavior_timestamp=now,
        pose_timestamp=now,
        now=now,
    )
    assert out.source == "safety_gate"
    assert out.reason == "no_motor_command"
    assert out.steering_deg == 0.0
    assert out.speed_mps == 0.0


def test_invalid_command_falls_back_with_command_reason() -> None:
    gate = SafetyGate()
    now = 1000.0
    cmd = MotorCommand(
        timestamp=now,
        steering_deg=0.0,
        speed_mps=0.0,
        valid=False,
        source="motion_controller",
        reason="mpc_solver_failure",
    )
    out = gate.evaluate(
        motor_command=cmd,
        behavior_timestamp=now,
        pose_timestamp=now,
        now=now,
    )
    assert out.source == "safety_gate"
    assert out.reason == "mpc_solver_failure"


def test_no_behavior_output_falls_back() -> None:
    gate = SafetyGate()
    now = 1000.0
    out = gate.evaluate(
        motor_command=_good_cmd(now),
        behavior_timestamp=None,
        pose_timestamp=now,
        now=now,
    )
    assert out.source == "safety_gate"
    assert out.reason == "no_behavior_output"


def test_behavior_stale_falls_back() -> None:
    cfg = SafetyGateConfig(behavior_stale_timeout_s=0.5, pose_stale_timeout_s=0.3)
    gate = SafetyGate(cfg)
    now = 1000.0
    out = gate.evaluate(
        motor_command=_good_cmd(now),
        behavior_timestamp=now - 1.5,  # 1.5 s ago > 0.5 s timeout
        pose_timestamp=now - 0.05,
        now=now,
    )
    assert out.source == "safety_gate"
    assert out.reason.startswith("behavior_stale_")


def test_no_pose_falls_back() -> None:
    gate = SafetyGate()
    now = 1000.0
    out = gate.evaluate(
        motor_command=_good_cmd(now),
        behavior_timestamp=now - 0.1,
        pose_timestamp=None,
        now=now,
    )
    assert out.source == "safety_gate"
    assert out.reason == "no_pose_estimate"


def test_pose_stale_falls_back() -> None:
    cfg = SafetyGateConfig(behavior_stale_timeout_s=0.5, pose_stale_timeout_s=0.3)
    gate = SafetyGate(cfg)
    now = 1000.0
    out = gate.evaluate(
        motor_command=_good_cmd(now),
        behavior_timestamp=now - 0.1,
        pose_timestamp=now - 0.5,  # 0.5 s ago > 0.3 s timeout
        now=now,
    )
    assert out.source == "safety_gate"
    assert out.reason.startswith("pose_stale_")


def test_manual_fallback_returns_stop() -> None:
    gate = SafetyGate()
    out = gate.fallback()
    assert out.source == "safety_gate"
    assert out.reason == "manual"
    assert out.steering_deg == 0.0
    assert out.speed_mps == 0.0
    assert out.valid is True  # válido para despachar (es un STOP intencional)


def test_last_reason_updates_per_call() -> None:
    gate = SafetyGate()
    now = 1000.0
    gate.evaluate(
        motor_command=_good_cmd(now),
        behavior_timestamp=now,
        pose_timestamp=now,
        now=now,
    )
    assert gate.last_reason == "passthrough"
    gate.evaluate(
        motor_command=None,
        behavior_timestamp=now,
        pose_timestamp=now,
        now=now,
    )
    assert gate.last_reason == "no_motor_command"
