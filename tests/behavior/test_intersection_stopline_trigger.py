"""Tests del trigger por stopline visual en `Intersection` (plan TANDA 1.1).

Antes el scenario `Intersection` sólo se activaba con atributos del mapa
(ATTR_INTERSECTION, ATTR_INTERSECTION_EXIT) o con regulatory_ahead. En
pista nueva donde el mapa OSM no está re-anotado, la única señal de "te
estás acercando a una intersección" es la línea blanca pintada en el
piso — el observador de stopline ya la detecta, pero el scenario no la
miraba. Este test asegura que ahora sí.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.behavior.scenarios.intersection import Intersection


def _make_ctx(*, stopline_observation=None, lanelet_map=None, current_lanelet_id=None,
              regulatory_ahead=()):
    pose = SimpleNamespace(
        fused_pose=SimpleNamespace(x=0.0, y=0.0, yaw=0.0),
        speed_mps=0.1,
    )
    route = SimpleNamespace(
        current_lanelet_id=current_lanelet_id,
        route_waypoints=None,
        route_active=False,
        matched_idx=0,
        matched_pose=SimpleNamespace(x=0.0, y=0.0, yaw=0.0),
        next_lanelet_ids=(),
        regulatory_ahead=regulatory_ahead,
    )
    return SimpleNamespace(
        now_s=0.0,
        dt=0.05,
        horizon_n=20,
        nominal_speed_mps=0.10,
        max_speed_mps=0.5,
        pose=pose,
        route=route,
        lane_observation=None,
        stopline_observation=stopline_observation,
        tracked_objects=(),
        lanelet_map=lanelet_map,
        sign_hints=(),
    )


def _stopline(*, stable=True, distance_m=0.30):
    return SimpleNamespace(
        timestamp=0.0,
        visible=True,
        stable=stable,
        distance_m=distance_m,
        confidence=0.9,
        pass_event=None,
        expected_node_id=None,
        expected_node_attr=0,
        source="visual",
        debug={},
    )


def test_stopline_visible_and_near_activates_without_map():
    """Sin mapa cargado, una stopline estable a 0.30 m ya activa Intersection."""
    scen = Intersection()
    ctx = _make_ctx(stopline_observation=_stopline(stable=True, distance_m=0.30))
    assert scen.is_active(ctx)


def test_stopline_far_does_not_activate():
    """A 1.5 m la stopline es informativa pero todavía no accionable."""
    scen = Intersection()
    ctx = _make_ctx(stopline_observation=_stopline(stable=True, distance_m=1.5))
    assert not scen.is_active(ctx)


def test_stopline_unstable_ignored():
    """Una stopline parpadeante (single frame) no es señal confiable."""
    scen = Intersection()
    ctx = _make_ctx(stopline_observation=_stopline(stable=False, distance_m=0.30))
    assert not scen.is_active(ctx)


def test_stopline_none_does_not_activate():
    """Sin observación de stopline no debe activarse (caso default sin pista)."""
    scen = Intersection()
    ctx = _make_ctx(stopline_observation=None)
    assert not scen.is_active(ctx)


def test_stopline_distance_zero_does_not_activate():
    """distance_m=0 puede ser stub/uninitialized — no activamos por las dudas."""
    scen = Intersection()
    ctx = _make_ctx(stopline_observation=_stopline(stable=True, distance_m=0.0))
    assert not scen.is_active(ctx)


def test_stopline_distance_negative_invalid_does_not_activate():
    """Distancia negativa significaría línea detrás del auto — no es accionable."""
    scen = Intersection()
    ctx = _make_ctx(stopline_observation=_stopline(stable=True, distance_m=-0.20))
    assert not scen.is_active(ctx)
