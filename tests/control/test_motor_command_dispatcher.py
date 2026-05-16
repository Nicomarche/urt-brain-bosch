from __future__ import annotations

from pathlib import Path
from queue import Queue

import pytest

from src.control.motor_command_dispatcher import threadMotorCommandDispatcher
from src.core.messaging.allMessages import SpeedMotor, SteerMotor
from src.core.types.control import MotorCommand
from src.core.types.pose import Pose2D, PoseEstimate
from src.core.types.routing import RouteContext


def _queues():
    return {
        "Critical": Queue(),
        "Warning": Queue(),
        "General": Queue(),
        "Config": Queue(),
    }


@pytest.mark.parametrize(
    ("steering_deg", "expected_wire"),
    [
        (5.0, -50),
        (-7.5, 75),
        (0.0, 0),
    ],
)
def test_dispatcher_inverts_steering_sign_for_wire_protocol(
    steering_deg: float,
    expected_wire: int,
) -> None:
    queues = _queues()
    dispatcher = threadMotorCommandDispatcher(queues, motor_command_buffer=None)

    dispatcher._send(
        MotorCommand(
            timestamp=1.0,
            steering_deg=steering_deg,
            speed_mps=0.42,
            valid=True,
            source="test",
        )
    )

    steer_msg = queues["Critical"].get_nowait()
    speed_msg = queues["Critical"].get_nowait()

    assert steer_msg["Owner"] == SteerMotor.Owner.value
    assert steer_msg["msgID"] == SteerMotor.msgID.value
    assert steer_msg["msgValue"] == str(expected_wire)

    assert speed_msg["Owner"] == SpeedMotor.Owner.value
    assert speed_msg["msgID"] == SpeedMotor.msgID.value
    assert speed_msg["msgValue"] == "420"


def test_dispatcher_sends_negative_speed_for_parking_reverse() -> None:
    queues = _queues()
    dispatcher = threadMotorCommandDispatcher(queues, motor_command_buffer=None)

    dispatcher._send(
        MotorCommand(
            timestamp=1.0,
            steering_deg=-25.0,
            speed_mps=-0.10,
            valid=True,
            source="motion_controller",
            reason="parking_reversing_entry",
        )
    )

    _steer_msg = queues["Critical"].get_nowait()
    speed_msg = queues["Critical"].get_nowait()
    assert speed_msg["Owner"] == SpeedMotor.Owner.value
    assert speed_msg["msgValue"] == "-100"


def test_auto_launch_hold_waits_for_route_pose_and_stable_commands() -> None:
    queues = _queues()
    dispatcher = threadMotorCommandDispatcher(queues, motor_command_buffer=None)
    dispatcher._current_state = "AUTO"
    dispatcher._auto_launch_ready_ticks_required = 2
    cmd = MotorCommand(
        timestamp=100.0,
        steering_deg=0.0,
        speed_mps=0.20,
        valid=True,
        source="motion_controller",
    )
    pose = PoseEstimate(
        fused_pose=Pose2D(0.0, 0.0, 0.0),
        relocalization_mode="dead_reckoning",
    )
    route = RouteContext(route_active=True)

    assert dispatcher._auto_launch_hold_reason(cmd, pose, None, now=100.0) == "auto_launch_hold_route_pending"
    assert dispatcher._auto_launch_hold_reason(cmd, pose, route, now=100.1) == "auto_launch_hold_debounce"
    assert dispatcher._auto_launch_hold_reason(cmd, pose, route, now=100.2) is None
    assert dispatcher._auto_launch_released is True


def test_auto_launch_hold_waits_for_gps_entry_pending() -> None:
    queues = _queues()
    dispatcher = threadMotorCommandDispatcher(queues, motor_command_buffer=None)
    dispatcher._current_state = "AUTO"
    cmd = MotorCommand(
        timestamp=100.0,
        steering_deg=0.0,
        speed_mps=0.20,
        valid=True,
        source="motion_controller",
    )
    pose = PoseEstimate(
        fused_pose=Pose2D(0.0, 0.0, 0.0),
        relocalization_mode="auto_gps_entry_pending",
    )
    route = RouteContext(route_active=True)

    assert dispatcher._auto_launch_hold_reason(cmd, pose, route, now=100.0) == "auto_launch_hold_pose_pending"


def test_thread_line_following_has_no_direct_motor_writers() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "hardware"
        / "camera"
        / "threads"
        / "threadLineFollowing.py"
    ).read_text(encoding="utf-8")

    assert "messageHandlerSender(self.queuesList, SpeedMotor" not in source
    assert "messageHandlerSender(self.queuesList, SteerMotor" not in source
    assert "speedMotorSender.send" not in source
    assert "steerMotorSender.send" not in source
