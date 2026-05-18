"""Tests del scenario `TrafficLight` (plan C1.3)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.behavior.scenarios.traffic_light import TrafficLight
from src.core.types.behavior import MotionState, ScenarioName


def _make_ctx(*, sign_hints=None, now_s=0.0, nominal_speed_mps=0.10):
    """PlanningContext mínimo para testear TrafficLight."""
    pose = SimpleNamespace(
        fused_pose=SimpleNamespace(x=0.0, y=0.0, yaw=0.0),
        speed_mps=0.0,
        timestamp=now_s,
    )
    route = SimpleNamespace(
        current_lanelet_id=None,
        route_waypoints=None,
        route_active=False,
        matched_idx=0,
        matched_pose=SimpleNamespace(x=0.0, y=0.0, yaw=0.0),
        map_match_error_m=0.0,
        next_lanelet_ids=(),
    )
    return SimpleNamespace(
        now_s=now_s,
        dt=0.05,
        horizon_n=20,
        nominal_speed_mps=nominal_speed_mps,
        max_speed_mps=0.5,
        pose=pose,
        route=route,
        lane_observation=None,
        stopline_observation=None,
        tracked_objects=(),
        lanelet_map=None,
        sign_hints=tuple(sign_hints or []),
    )


def test_no_hints_inactive():
    scen = TrafficLight()
    ctx = _make_ctx(sign_hints=[])
    assert not scen.is_active(ctx)


def test_red_light_in_range_activates():
    scen = TrafficLight()
    hint = {"kind": "red_light", "distance_m": 2.0}
    ctx = _make_ctx(sign_hints=[hint])
    assert scen.is_active(ctx)


def test_green_light_releases():
    scen = TrafficLight()
    # Primero, activado por RED
    ctx_red = _make_ctx(sign_hints=[{"kind": "red_light", "distance_m": 2.0}], now_s=1.0)
    assert scen.is_active(ctx_red)
    # Luego GREEN
    ctx_green = _make_ctx(sign_hints=[{"kind": "green_light", "distance_m": 2.0}], now_s=2.0)
    assert not scen.is_active(ctx_green)


def test_yellow_light_activates():
    scen = TrafficLight()
    hint = {"kind": "yellow_light", "distance_m": 2.5}
    ctx = _make_ctx(sign_hints=[hint])
    assert scen.is_active(ctx)


def test_priority_is_92():
    """Plan C1.3: priority 92 (> StopSign=90, < PedestrianYield=95)."""
    assert TrafficLight().priority == 92


def test_far_light_does_not_activate():
    scen = TrafficLight()
    hint = {"kind": "red_light", "distance_m": 10.0}  # > react_range
    ctx = _make_ctx(sign_hints=[hint])
    assert not scen.is_active(ctx)
