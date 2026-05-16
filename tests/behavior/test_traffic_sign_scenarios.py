from __future__ import annotations

import numpy as np
import pytest

from src.behavior.scenarios.crosswalk import Crosswalk
from src.behavior.scenarios.highway import Highway
from src.behavior.scenarios.parking import Parking
from src.behavior.scenarios.stop_sign import StopSign
from src.core.types.behavior import ScenarioName
from src.core.types.perception import TrackedObject
from src.core.types.routing import RegulatoryElement
from tests.behavior.conftest import make_context


def _route_waypoints() -> list[list[float]]:
    return [[float(x), 0.0, 0.0] for x in np.linspace(0.0, 2.0, 20)]


def test_stop_sign_holds_for_three_seconds_and_consumes_hint() -> None:
    scenario = StopSign()
    hint = {
        "kind": "stop",
        "raw_class": "stop-sign",
        "distance_m": 0.05,
        "distance_source": "lidar",
        "timestamp": 10.0,
    }
    ctx = make_context(
        current_lanelet_id="lanelet-a",
        route_waypoints=_route_waypoints(),
        sign_hints=(hint,),
        speed_mps=0.01,
        nominal_speed_mps=0.20,
        now_s=10.0,
    )

    assert scenario.is_active(ctx) is True
    plan = scenario.plan(ctx)
    assert plan.scenario_name == ScenarioName.STOP.value
    assert plan.stop_required is True

    ctx_after = make_context(
        current_lanelet_id="lanelet-a",
        route_waypoints=_route_waypoints(),
        sign_hints=({**hint, "timestamp": 13.2},),
        speed_mps=0.0,
        nominal_speed_mps=0.20,
        now_s=13.2,
    )
    assert scenario.is_active(ctx_after) is False


def test_stop_sign_ignores_stopline_and_non_lidar_distance() -> None:
    scenario = StopSign()
    reg = RegulatoryElement(
        element_id="stopline-1",
        kind="stopline",
        position_xy=(0.0, 0.0),
        data={"distance_m": 0.05},
    )
    ctx = make_context(
        current_lanelet_id="lanelet-a",
        route_waypoints=_route_waypoints(),
        regulatory_ahead=(reg,),
        sign_hints=(
            {
                "kind": "stop",
                "raw_class": "stop-sign",
                "distance_m": 0.05,
                "distance_source": "camera",
                "timestamp": 10.0,
            },
        ),
        speed_mps=0.01,
        nominal_speed_mps=0.20,
        now_s=10.0,
    )

    assert scenario.is_active(ctx) is False


def test_stop_sign_commits_to_stop_when_lidar_is_close() -> None:
    scenario = StopSign()
    hint = {
        "kind": "stop",
        "raw_class": "stop-sign",
        "distance_m": 0.35,
        "distance_source": "lidar",
        "timestamp": 10.0,
    }
    ctx = make_context(
        current_lanelet_id="lanelet-a",
        route_waypoints=_route_waypoints(),
        sign_hints=(hint,),
        speed_mps=0.20,
        nominal_speed_mps=0.20,
        now_s=10.0,
    )

    assert scenario.is_active(ctx) is True
    plan = scenario.plan(ctx)

    assert plan.stop_required is True
    assert plan.notes["stop_mode"] == "stopping"
    assert np.max(plan.speed_profile) == pytest.approx(0.0)

    ctx_missing_hint = make_context(
        current_lanelet_id="lanelet-a",
        route_waypoints=_route_waypoints(),
        sign_hints=(),
        speed_mps=0.10,
        nominal_speed_mps=0.20,
        now_s=10.2,
    )
    assert scenario.is_active(ctx_missing_hint) is True


def test_stop_sign_uses_close_camera_fallback_when_lidar_is_missing() -> None:
    scenario = StopSign()
    hint = {
        "kind": "stop",
        "raw_class": "stop-sign",
        "distance_m": 0.37,
        "distance_source": "camera",
        "confidence": 0.84,
        "box_area": 0.014,
        "timestamp": 10.0,
    }
    ctx = make_context(
        current_lanelet_id="lanelet-a",
        route_waypoints=_route_waypoints(),
        sign_hints=(hint,),
        speed_mps=0.20,
        nominal_speed_mps=0.20,
        now_s=10.0,
    )

    assert scenario.is_active(ctx) is True
    plan = scenario.plan(ctx)

    assert plan.stop_required is True
    assert plan.notes["stop_source"] == "sign_hint_camera_fallback"
    assert plan.notes["distance_source"] == "camera"
    assert np.max(plan.speed_profile) == pytest.approx(0.0)


def test_stop_sign_does_not_retrigger_same_visible_sign_after_hold() -> None:
    scenario = StopSign()
    hint = {
        "kind": "stop",
        "raw_class": "stop-sign",
        "distance_m": 0.30,
        "distance_source": "lidar",
        "timestamp": 10.0,
    }
    ctx = make_context(
        current_lanelet_id="lanelet-a",
        route_waypoints=_route_waypoints(),
        sign_hints=(hint,),
        speed_mps=0.0,
        nominal_speed_mps=0.20,
        now_s=10.0,
    )
    assert scenario.is_active(ctx) is True
    assert scenario.plan(ctx).stop_required is True

    release_ctx = make_context(
        current_lanelet_id="lanelet-a",
        route_waypoints=_route_waypoints(),
        sign_hints=({**hint, "timestamp": 13.1},),
        speed_mps=0.0,
        nominal_speed_mps=0.20,
        now_s=13.1,
    )
    assert scenario.is_active(release_ctx) is False

    same_visible_sign_new_lanelet = make_context(
        current_lanelet_id="lanelet-b",
        route_waypoints=_route_waypoints(),
        sign_hints=({**hint, "timestamp": 13.2},),
        speed_mps=0.20,
        nominal_speed_mps=0.20,
        now_s=13.2,
    )
    assert scenario.is_active(same_visible_sign_new_lanelet) is False


def test_crosswalk_sign_slows_to_ten_cm_s_without_pedestrian() -> None:
    scenario = Crosswalk()
    ctx = make_context(
        current_lanelet_id="lanelet-a",
        route_waypoints=_route_waypoints(),
        sign_hints=({"kind": "crosswalk", "distance_m": 1.0, "timestamp": 1.0},),
        nominal_speed_mps=0.25,
        max_speed_mps=0.40,
    )

    assert scenario.is_active(ctx) is True
    plan = scenario.plan(ctx)
    assert plan.stop_required is False
    assert np.max(plan.speed_profile) == pytest.approx(0.10)
    assert plan.notes["min_moving_speed_mps"] == pytest.approx(0.10)


def test_crosswalk_stops_for_pedestrian() -> None:
    scenario = Crosswalk()
    track = TrackedObject(
        track_id=1,
        class_name="pedestrian",
        position_world_xy=(0.5, 0.0),
        age_frames=2,
        confidence=0.9,
    )
    ctx = make_context(
        current_lanelet_id="lanelet-a",
        route_waypoints=_route_waypoints(),
        sign_hints=({"kind": "crosswalk", "distance_m": 1.0, "timestamp": 1.0},),
        tracked_objects=(track,),
        nominal_speed_mps=0.25,
    )

    assert scenario.is_active(ctx) is True
    assert scenario.plan(ctx).stop_required is True


def test_highway_entry_and_end_signs_toggle_highway_minimum() -> None:
    scenario = Highway()
    entry_ctx = make_context(
        current_lanelet_id="lanelet-a",
        route_waypoints=_route_waypoints(),
        sign_hints=({"kind": "highway_entry", "distance_m": 1.0, "timestamp": 1.0},),
        nominal_speed_mps=0.25,
        max_speed_mps=1.0,
    )

    assert scenario.is_active(entry_ctx) is True
    plan = scenario.plan(entry_ctx)
    assert float(np.min(plan.speed_profile)) >= 0.40 - 1e-6
    assert plan.notes["min_moving_speed_mps"] == pytest.approx(0.40)

    exit_ctx = make_context(
        current_lanelet_id="lanelet-a",
        route_waypoints=_route_waypoints(),
        sign_hints=({"kind": "highway_end", "distance_m": 1.0, "timestamp": 2.0},),
        nominal_speed_mps=0.25,
        max_speed_mps=1.0,
    )
    assert scenario.is_active(exit_ctx) is False


def test_parking_search_speed_and_direct_reverse_command() -> None:
    scenario = Parking()
    ctx = make_context(
        current_lanelet_id="lanelet-a",
        route_waypoints=_route_waypoints(),
        sign_hints=({"kind": "parking", "distance_m": 0.5, "timestamp": 1.0},),
        nominal_speed_mps=0.25,
    )

    assert scenario.is_active(ctx) is True
    plan = scenario.plan(ctx)
    assert np.max(plan.speed_profile) == pytest.approx(0.10)
    assert plan.notes["parking_state"] == "spot_tracked"

    scenario._advance("reversing_entry", 1.0)
    reverse_ctx = make_context(
        current_lanelet_id="lanelet-a",
        route_waypoints=_route_waypoints(),
        sign_hints=(),
        nominal_speed_mps=0.25,
        now_s=1.2,
    )
    reverse_plan = scenario.plan(reverse_ctx)
    direct = reverse_plan.notes["direct_motor_command"]
    assert direct["allow_reverse"] is True
    assert direct["speed_mps"] == pytest.approx(-0.10)
