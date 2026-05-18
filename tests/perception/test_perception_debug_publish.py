"""Plan TANDA 2.2 / C5.1 — PERCEPTION_DEBUG topic agregado.

Verifica que el ``BehaviorPlanner`` publica un snapshot agregado por tick
con tracked objects + scenario activo + motion_state + sign hints. Esto
permite al GUI mostrar telemetría unificada sin parsear N topics.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from src.behavior.planner_thread import threadBehaviorPlanner
from src.core.bus.topics import PERCEPTION_DEBUG
from src.core.messaging.header import SeqCounter
from src.core.types.behavior import BehaviorOutput, MotionState, ScenarioName


def _make_planner_minimal():
    """Instanciamos sólo lo que toca _publish_perception_debug."""
    inst = threadBehaviorPlanner.__new__(threadBehaviorPlanner)
    inst._perception_debug_sender = MagicMock()
    inst._seq_perception_debug = SeqCounter()
    inst.logging = None
    return inst


def _make_plan(scenario=ScenarioName.LANE_KEEP.value, motion_state=MotionState.MOVING.value,
               stop_required=False):
    return BehaviorOutput(
        timestamp=1.5,
        dt=0.05,
        target_path=np.zeros((1, 3)),
        speed_profile=np.zeros(0),
        scenario_name=scenario,
        motion_state=motion_state,
        valid=True,
        stop_required=stop_required,
    )


def _make_ctx(tracked_objects=(), sign_hints=()):
    return SimpleNamespace(
        tracked_objects=tracked_objects,
        sign_hints=sign_hints,
    )


def _tracked(track_id=1, class_id=11, class_name="pedestrian", world_xy=(1.5, 0.0),
             confidence=0.8, age_frames=5):
    return SimpleNamespace(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        position_world_xy=world_xy,
        confidence=confidence,
        age_frames=age_frames,
    )


def test_publish_empty_perception_no_objects():
    planner = _make_planner_minimal()
    plan = _make_plan()
    ctx = _make_ctx()
    planner._publish_perception_debug(plan, ctx)

    planner._perception_debug_sender.send.assert_called_once()
    payload = planner._perception_debug_sender.send.call_args[0][0]
    assert payload["active_scenario"] == "lane_keep"
    assert payload["motion_state"] == "moving"
    assert payload["tracked_objects"] == []
    assert payload["sign_hints"] == []
    assert payload["stop_required"] is False
    assert payload["header"]["source"] == "behavior_planner"
    assert payload["header"]["seq"] == 0


def test_publish_includes_tracked_objects_with_world_xy():
    planner = _make_planner_minimal()
    plan = _make_plan(
        scenario=ScenarioName.PEDESTRIAN_YIELD.value,
        motion_state=MotionState.PEDESTRIAN_YIELDING.value,
        stop_required=True,
    )
    ped = _tracked(track_id=42, class_id=11, class_name="pedestrian",
                   world_xy=(1.5, -0.2), confidence=0.85, age_frames=10)
    car = _tracked(track_id=43, class_id=12, class_name="car",
                   world_xy=(3.0, 0.5), confidence=0.65, age_frames=20)
    ctx = _make_ctx(tracked_objects=(ped, car))
    planner._publish_perception_debug(plan, ctx)

    payload = planner._perception_debug_sender.send.call_args[0][0]
    assert payload["active_scenario"] == "pedestrian_yield"
    assert payload["motion_state"] == "pedestrian_yielding"
    assert payload["stop_required"] is True
    assert len(payload["tracked_objects"]) == 2

    pedestrian = payload["tracked_objects"][0]
    assert pedestrian["track_id"] == 42
    assert pedestrian["class_id"] == 11
    assert pedestrian["class_name"] == "pedestrian"
    assert pedestrian["world_xy"] == [1.5, -0.2]
    assert pedestrian["confidence"] == 0.85
    assert pedestrian["age_frames"] == 10


def test_publish_handles_tracked_object_without_world_xy():
    """Plan TANDA 0.2 + 2.2: si por algún motivo world_xy es None, el
    serializador NO debe explotar — sólo serializa None."""
    planner = _make_planner_minimal()
    plan = _make_plan()
    obj = _tracked(world_xy=None)
    ctx = _make_ctx(tracked_objects=(obj,))
    planner._publish_perception_debug(plan, ctx)

    payload = planner._perception_debug_sender.send.call_args[0][0]
    assert payload["tracked_objects"][0]["world_xy"] is None


def test_publish_seq_monotonic_across_ticks():
    """Cada tick del planner debe avanzar el seq del PERCEPTION_DEBUG
    para que analyze_run pueda detectar drops."""
    planner = _make_planner_minimal()
    plan = _make_plan()
    ctx = _make_ctx()

    for _ in range(3):
        planner._publish_perception_debug(plan, ctx)

    seqs = [
        call.args[0]["header"]["seq"]
        for call in planner._perception_debug_sender.send.call_args_list
    ]
    assert seqs == [0, 1, 2]


def test_topic_is_registered_with_stream_qos():
    """El topic debe estar en el catálogo de topics con QoS STREAM."""
    assert PERCEPTION_DEBUG.name == "/auto/perception/debug"
    assert PERCEPTION_DEBUG.qos_preset == "STREAM"
