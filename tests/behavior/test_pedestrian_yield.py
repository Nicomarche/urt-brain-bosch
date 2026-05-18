"""Tests del scenario `PedestrianYield` (plan C1.4)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from src.behavior.scenarios.pedestrian_yield import PedestrianYield
from src.perception.object_classes import TrackedObject as _TO


def _make_ctx(*, tracked_objects=None, ego_xy=(0.0, 0.0), ego_yaw=0.0, now_s=0.0):
    pose = SimpleNamespace(
        fused_pose=SimpleNamespace(x=ego_xy[0], y=ego_xy[1], yaw=ego_yaw),
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
        nominal_speed_mps=0.10,
        max_speed_mps=0.5,
        pose=pose,
        route=route,
        lane_observation=None,
        stopline_observation=None,
        tracked_objects=tuple(tracked_objects or []),
        lanelet_map=None,
        sign_hints=(),
    )


def _ped(x, y, *, conf=0.8, age=5, class_id=int(_TO.PEDESTRIAN)):
    return SimpleNamespace(
        class_id=class_id,
        class_name="pedestrian",
        confidence=conf,
        position_world_xy=(x, y),
        age_frames=age,
        velocity_world_xy=(0.0, 0.0),
        timestamp=0.0,
        track_id=1,
        bbox_xyxy=(0, 0, 1, 1),
    )


def test_no_pedestrians_inactive():
    scen = PedestrianYield()
    ctx = _make_ctx()
    assert not scen.is_active(ctx)


def test_pedestrian_in_path_activates():
    scen = PedestrianYield()
    ctx = _make_ctx(tracked_objects=[_ped(x=1.5, y=0.0)])  # 1.5m forward
    assert scen.is_active(ctx)


def test_pedestrian_lateral_outside_cone_inactive():
    """Peatón a 90° del heading ego — fuera del cono (45° default)."""
    scen = PedestrianYield()
    ctx = _make_ctx(tracked_objects=[_ped(x=0.0, y=1.5)])  # 90°
    assert not scen.is_active(ctx)


def test_pedestrian_too_far_inactive():
    scen = PedestrianYield()
    ctx = _make_ctx(tracked_objects=[_ped(x=10.0, y=0.0)])
    assert not scen.is_active(ctx)


def test_pedestrian_low_confidence_inactive():
    scen = PedestrianYield()
    ctx = _make_ctx(tracked_objects=[_ped(x=1.5, y=0.0, conf=0.3)])
    assert not scen.is_active(ctx)


def test_pedestrian_low_age_inactive():
    scen = PedestrianYield()
    ctx = _make_ctx(tracked_objects=[_ped(x=1.5, y=0.0, age=1)])
    assert not scen.is_active(ctx)


def test_priority_is_95():
    """Plan C1.4: priority 95 (más alta que TrafficLight=92 y StopSign=90)."""
    assert PedestrianYield().priority == 95


def test_release_after_grace():
    """Una vez yielding, libera tras release_grace_s sin ver al peatón."""
    scen = PedestrianYield()
    # Activar
    ctx1 = _make_ctx(tracked_objects=[_ped(x=1.5, y=0.0)], now_s=10.0)
    assert scen.is_active(ctx1)
    # Sin peatón pero antes del grace (default 0.4s)
    ctx2 = _make_ctx(tracked_objects=[], now_s=10.1)
    assert scen.is_active(ctx2)
    # Después del grace
    ctx3 = _make_ctx(tracked_objects=[], now_s=11.0)
    assert not scen.is_active(ctx3)


def test_pedestrian_without_world_xy_is_ignored():
    """Plan TANDA 0.2 — contrato explícito.

    Un TrackedObject de peatón con ``position_world_xy=None`` NO debe
    activar yield: el scenario no tiene cómo medir distancia/ángulo
    contra el ego. Antes del fix esto pasaba con cualquier peatón
    detectado sin LiDAR; ahora el pipeline llena world_xy con fallback de
    cámara, pero el contrato del scenario se mantiene defensivo.
    """
    scen = PedestrianYield()
    pedestrian_no_xy = SimpleNamespace(
        class_id=int(_TO.PEDESTRIAN),
        class_name="pedestrian",
        confidence=0.9,
        position_world_xy=None,  # ← el bug viejo dejaba esto
        age_frames=5,
        velocity_world_xy=(0.0, 0.0),
        timestamp=0.0,
        track_id=1,
        bbox_xyxy=(0, 0, 1, 1),
    )
    ctx = _make_ctx(tracked_objects=[pedestrian_no_xy])
    assert not scen.is_active(ctx)
