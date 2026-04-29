# tests/behavior/test_planner.py
#
# Tests del BehaviorPlanner. Validamos:
#   1. Itera scenarios por prioridad descendente.
#   2. El primero `is_active(ctx) == True` gana.
#   3. add_scenario es idempotente.
#   4. Si ningún scenario está activo, devuelve fallback con valid=False.
#   5. Si un scenario crashea, el planner sigue con el siguiente
#      (no propaga la excepción).
#   6. El overlay se aplica al output del scenario elegido.

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.behavior.planner import BehaviorPlanner
from src.behavior.scenarios.base import BaseScenario, HysteresisGate
from src.core.types.behavior import BehaviorOutput, ScenarioName
from src.core.types.routing import RegulatoryElement
from tests.behavior.conftest import make_context


class _FakeScenario(BaseScenario):
    """Scenario controlable para tests."""

    def __init__(
        self,
        name: str,
        priority: int,
        active: bool,
        speed: float = 0.5,
        crash_in: str | None = None,
    ) -> None:
        self.name = name
        self.priority = priority
        self._active = active
        self._speed = speed
        self._crash_in = crash_in

    def is_active(self, ctx) -> bool:
        if self._crash_in == "is_active":
            raise RuntimeError("simulated crash")
        return self._active

    def plan(self, ctx) -> BehaviorOutput:
        if self._crash_in == "plan":
            raise RuntimeError("plan crash")
        n = ctx.horizon_n
        return BehaviorOutput(
            timestamp=ctx.now_s,
            dt=ctx.dt,
            target_path=np.zeros((n + 1, 3)),
            speed_profile=np.full(n, self._speed, dtype=float),
            scenario_name=self.name,
            valid=True,
        )


# ---------- prioridad y selección ------------------------------------------


def test_planner_orders_by_priority_descending() -> None:
    s_low = _FakeScenario("low", priority=1, active=False)
    s_high = _FakeScenario("high", priority=10, active=False)
    p = BehaviorPlanner([s_low, s_high])
    assert [s.name for s in p.scenarios] == ["high", "low"]


def test_planner_selects_highest_priority_active() -> None:
    s_low = _FakeScenario("low", priority=1, active=True, speed=0.1)
    s_high = _FakeScenario("high", priority=10, active=True, speed=0.9)
    p = BehaviorPlanner([s_low, s_high])
    out = p.plan(make_context(max_speed_mps=2.0))
    assert out.scenario_name == "high"
    assert out.speed_profile[0] == pytest.approx(0.9)


def test_planner_skips_inactive_scenarios() -> None:
    s_high = _FakeScenario("high", priority=10, active=False)
    s_med = _FakeScenario("med", priority=5, active=True, speed=0.3)
    s_low = _FakeScenario("low", priority=1, active=True, speed=0.1)
    p = BehaviorPlanner([s_high, s_med, s_low])
    out = p.plan(make_context(max_speed_mps=2.0))
    assert out.scenario_name == "med"


# ---------- idempotencia y fallback ----------------------------------------


def test_planner_add_scenario_idempotent() -> None:
    s = _FakeScenario("a", priority=1, active=True)
    p = BehaviorPlanner()
    p.add_scenario(s)
    p.add_scenario(s)
    assert len(p.scenarios) == 1


def test_planner_returns_fallback_when_no_scenario_active() -> None:
    p = BehaviorPlanner([_FakeScenario("a", priority=1, active=False)])
    out = p.plan(make_context())
    assert out.valid is False
    assert out.stop_required is True
    assert out.scenario_name == ScenarioName.FALLBACK.value


# ---------- robustez ante crashes ------------------------------------------


def test_planner_isactive_crash_falls_through() -> None:
    s_crash = _FakeScenario("crash", priority=10, active=True, crash_in="is_active")
    s_ok = _FakeScenario("ok", priority=5, active=True, speed=0.4)
    p = BehaviorPlanner([s_crash, s_ok])
    out = p.plan(make_context())
    assert out.scenario_name == "ok"


def test_planner_plan_crash_returns_fallback() -> None:
    s_crash = _FakeScenario("crash", priority=10, active=True, crash_in="plan")
    p = BehaviorPlanner([s_crash])
    out = p.plan(make_context())
    assert out.valid is False
    assert "scenario_crash" in out.notes.get("reason", "")


# ---------- overlay aplicado -----------------------------------------------


def test_planner_applies_velocity_overlay() -> None:
    """Si el scenario emite 1 m/s pero hay stopline cercano, output va a 0."""
    s = _FakeScenario("lk", priority=0, active=True, speed=1.0)
    p = BehaviorPlanner([s])
    reg = RegulatoryElement(
        element_id="r1",
        kind="stopline",
        position_xy=(0, 0),
        data={"distance_m": 1.5},
    )
    ctx = make_context(regulatory_ahead=(reg,), max_speed_mps=1.5, dt=0.1, horizon_n=20)
    out = p.plan(ctx)
    assert out.stop_required is True
    assert out.speed_profile[-1] == pytest.approx(0.0, abs=1e-6)


# ---------- HysteresisGate -------------------------------------------------


def test_hysteresis_gate_activates_below_enter() -> None:
    g = HysteresisGate(enter=1.0, exit=2.0)
    assert g.update(0.5) is True
    assert g.active is True


def test_hysteresis_gate_does_not_activate_in_band() -> None:
    g = HysteresisGate(enter=1.0, exit=2.0, active=False)
    assert g.update(1.5) is False
    assert g.active is False


def test_hysteresis_gate_deactivates_above_exit() -> None:
    g = HysteresisGate(enter=1.0, exit=2.0, active=True)
    assert g.update(2.5) is False
    assert g.active is False


def test_hysteresis_gate_remains_active_in_band() -> None:
    g = HysteresisGate(enter=1.0, exit=2.0, active=True)
    assert g.update(1.5) is True
    assert g.active is True
