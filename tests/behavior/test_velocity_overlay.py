# tests/behavior/test_velocity_overlay.py
#
# Tests de velocity_overlay. Validamos:
#   1. Cap global por max_speed_mps (siempre se aplica).
#   2. Stoplines dentro del rango bajan velocidad a 0 con rampa.
#   3. Crosswalks dentro del rango capean a CROSSWALK_SPEED.
#   4. Speed limits explícitos respetan el dato del regulator.
#   5. El overlay nunca SUBE velocidad — sólo baja.
#   6. Overlay devuelve un BehaviorOutput inmutable distinto al original.

from __future__ import annotations

import numpy as np
import pytest

from src.behavior.velocity_overlay import apply_overlay
from src.core.types.behavior import BehaviorOutput, ScenarioName
from src.core.types.routing import RegulatoryElement, RouteContext


def _plan(speed: float, n: int = 10, dt: float = 0.1) -> BehaviorOutput:
    """Helper: crea BehaviorOutput uniforme para tests."""
    return BehaviorOutput(
        timestamp=0.0,
        dt=dt,
        target_path=np.zeros((n + 1, 3)),
        speed_profile=np.full(n, speed, dtype=float),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
        stop_required=False,
    )


# ---------- max_speed cap --------------------------------------------------


def test_overlay_caps_to_max_speed() -> None:
    plan = _plan(speed=2.0)
    out = apply_overlay(plan, RouteContext(), max_speed_mps=1.0)
    assert np.all(out.speed_profile <= 1.0)
    assert out.speed_profile[0] == pytest.approx(1.0)


def test_overlay_passes_through_below_cap() -> None:
    plan = _plan(speed=0.4)
    out = apply_overlay(plan, RouteContext(), max_speed_mps=1.0)
    np.testing.assert_allclose(out.speed_profile, 0.4)


# ---------- stopline -------------------------------------------------------


def test_overlay_stopline_within_range_triggers_stop() -> None:
    plan = _plan(speed=0.5, n=20, dt=0.1)
    reg = RegulatoryElement(
        element_id="r1",
        kind="stopline",
        position_xy=(0.0, 0.0),
        data={"distance_m": 2.0},
    )
    route = RouteContext(regulatory_ahead=(reg,))
    out = apply_overlay(plan, route, max_speed_mps=1.0)
    assert out.stop_required is True
    # El final del speed_profile debe llegar a 0.
    assert out.speed_profile[-1] == pytest.approx(0.0, abs=1e-6)


def test_overlay_stopline_outside_range_no_stop() -> None:
    plan = _plan(speed=0.5)
    reg = RegulatoryElement(
        element_id="r1",
        kind="stopline",
        position_xy=(0.0, 0.0),
        data={"distance_m": 50.0},  # muy lejos
    )
    out = apply_overlay(plan, RouteContext(regulatory_ahead=(reg,)), max_speed_mps=1.0)
    assert out.stop_required is False
    np.testing.assert_allclose(out.speed_profile, 0.5)


# ---------- crosswalk ------------------------------------------------------


def test_overlay_crosswalk_caps_speed_no_stop() -> None:
    plan = _plan(speed=0.8)
    reg = RegulatoryElement(
        element_id="cw",
        kind="crosswalk",
        position_xy=(0.0, 0.0),
        data={"distance_m": 2.0},
    )
    out = apply_overlay(plan, RouteContext(regulatory_ahead=(reg,)), max_speed_mps=1.0)
    assert out.stop_required is False
    assert np.all(out.speed_profile <= 0.30 + 1e-6)


# ---------- speed_limit ----------------------------------------------------


def test_overlay_speed_limit_respects_data() -> None:
    plan = _plan(speed=0.9)
    reg = RegulatoryElement(
        element_id="lim",
        kind="speed_limit",
        position_xy=(0.0, 0.0),
        data={"speed_mps": 0.4},
    )
    out = apply_overlay(plan, RouteContext(regulatory_ahead=(reg,)), max_speed_mps=1.0)
    assert np.all(out.speed_profile <= 0.4 + 1e-6)


# ---------- monotonicity ---------------------------------------------------


def test_overlay_never_increases_speed() -> None:
    """Cualquier combinación de regulators puede solo reducir velocidad."""
    plan = _plan(speed=0.3)
    regs = (
        RegulatoryElement("r1", "intersection", (0, 0), data={"distance_m": 1.0}),
        RegulatoryElement("r2", "crosswalk", (0, 0), data={"distance_m": 2.0}),
        RegulatoryElement("r3", "speed_limit", (0, 0), data={"speed_mps": 0.5}),
    )
    out = apply_overlay(plan, RouteContext(regulatory_ahead=regs), max_speed_mps=2.0)
    assert np.all(out.speed_profile <= 0.3 + 1e-6)


# ---------- inmutabilidad --------------------------------------------------


def test_overlay_returns_new_object() -> None:
    plan = _plan(speed=2.0)
    out = apply_overlay(plan, RouteContext(), max_speed_mps=1.0)
    # Plan original no se modifica.
    assert plan.speed_profile[0] == pytest.approx(2.0)
    assert out.speed_profile[0] == pytest.approx(1.0)
    assert out is not plan
