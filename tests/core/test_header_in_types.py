"""Plan TANDA 2.1 — Header opcional en BehaviorOutput / PoseEstimate / MotorCommand.

Las tres dataclasses ganan un campo opcional ``header: MessageHeader | None``
con default ``None``. Retro-compat es mandatorio: código existente que
construye estos tipos SIN especificar ``header`` debe seguir funcionando.
Código nuevo que sí stampea el header obtiene trazabilidad por seq y
``analyze_run --check-seq-gaps`` puede auditar drops.
"""

from __future__ import annotations

import numpy as np

from src.core.messaging.header import MessageHeader, SeqCounter, stamp
from src.core.types.behavior import BehaviorOutput, ScenarioName
from src.core.types.control import MotorCommand
from src.core.types.pose import Pose2D, PoseEstimate


def test_behavior_output_default_no_header():
    """Constructor sin header debe seguir funcionando (retro-compat)."""
    bo = BehaviorOutput()
    assert bo.header is None
    assert bo.scenario_name == ScenarioName.FALLBACK.value
    assert bo.valid is False


def test_behavior_output_with_header_preserves_seq_and_source():
    seq = SeqCounter()
    hdr = stamp("behavior_planner", seq)
    bo = BehaviorOutput(
        timestamp=1.0,
        target_path=np.zeros((1, 3)),
        speed_profile=np.zeros(0),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
        header=hdr,
    )
    assert bo.header is not None
    assert bo.header.source == "behavior_planner"
    assert bo.header.seq == 0
    # Segundo stamp debe avanzar el contador.
    hdr2 = stamp("behavior_planner", seq)
    assert hdr2.seq == 1


def test_pose_estimate_default_no_header():
    pe = PoseEstimate()
    assert pe.header is None
    # Backward-compat con consumers que leen timestamp toplevel:
    assert pe.timestamp == 0.0


def test_pose_estimate_with_header():
    seq = SeqCounter()
    hdr = stamp("pose_estimator", seq, frame_id="map")
    pe = PoseEstimate(
        timestamp=42.0,
        fused_pose=Pose2D(x=1.0, y=2.0, yaw=0.5),
        header=hdr,
    )
    assert pe.header is not None
    assert pe.header.source == "pose_estimator"
    assert pe.header.frame_id == "map"


def test_motor_command_default_no_header():
    mc = MotorCommand()
    assert mc.header is None
    assert mc.valid is False
    assert mc.source == "none"


def test_motor_command_with_header():
    seq = SeqCounter()
    hdr = stamp("motion_controller", seq)
    mc = MotorCommand(
        timestamp=0.5,
        steering_deg=10.0,
        speed_mps=0.3,
        valid=True,
        source="motion_controller",
        header=hdr,
    )
    assert mc.header is not None
    assert mc.header.seq == 0
    assert mc.header.source == "motion_controller"


def test_header_seq_monotonic_per_producer():
    """Plan TANDA 2.1 — el seq por productor debe ser monotónico
    creciente. ``analyze_run --check-seq-gaps`` se apoya en esto para
    detectar drops del bus.
    """
    seq = SeqCounter()
    headers = [stamp("any_producer", seq) for _ in range(5)]
    seqs = [h.seq for h in headers]
    assert seqs == [0, 1, 2, 3, 4]


def test_header_to_dict_round_trip_in_payload():
    """Header debe serializarse para el bus (JSON-compatible dict)."""
    hdr = MessageHeader(timestamp=1.0, seq=42, source="foo", frame_id="map")
    serialized = hdr.to_dict()
    assert serialized == {
        "timestamp": 1.0,
        "seq": 42,
        "source": "foo",
        "frame_id": "map",
    }
    restored = MessageHeader.from_dict(serialized)
    assert restored == hdr
