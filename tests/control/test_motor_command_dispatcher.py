from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.control.motor_command_dispatcher import threadMotorCommandDispatcher
from src.core.bus.shim import messageHandlerSubscriber
from src.core.bus.topics import SPEED_MOTOR, STEER_MOTOR
from src.core.types.control import MotorCommand
from src.core.types.pose import Pose2D, PoseEstimate
from src.core.types.routing import RouteContext


def _queues():
    """Empty dict — the new bus does not consume queues. Kept only because
    ``threadMotorCommandDispatcher`` accepts it positionally for signature
    compatibility."""
    return {}


def _drain_with_retry(sub: messageHandlerSubscriber, timeout_s: float = 2.0):
    """Read one message from a shim subscriber with slow-joiner tolerance."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = sub.receive()
        if value is not None:
            return value
        time.sleep(0.01)
    return None


@pytest.mark.parametrize(
    ("steering_deg", "expected_wire"),
    [
        (5.0, -50),
        (-7.5, 75),
        (0.0, 0),
    ],
)
def test_dispatcher_inverts_steering_sign_for_wire_protocol(
    broker,
    steering_deg: float,
    expected_wire: int,
) -> None:
    queues = _queues()
    sub_steer = messageHandlerSubscriber(queues, STEER_MOTOR, "lastonly", subscribe=True)
    sub_speed = messageHandlerSubscriber(queues, SPEED_MOTOR, "lastonly", subscribe=True)
    sub_steer.subscribe()
    sub_speed.subscribe()
    time.sleep(0.1)  # slow-joiner: filters need to propagate before publish

    dispatcher = threadMotorCommandDispatcher(queues, motor_command_buffer=None)

    # Slow-joiner safety: keep publishing until both subscribers receive.
    # The same MotorCommand is idempotent — re-sending changes nothing.
    steer_value = None
    speed_value = None
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        dispatcher._send(
            MotorCommand(
                timestamp=1.0,
                steering_deg=steering_deg,
                speed_mps=0.42,
                valid=True,
                source="test",
            )
        )
        time.sleep(0.05)
        if steer_value is None:
            steer_value = sub_steer.receive()
        if speed_value is None:
            speed_value = sub_speed.receive()
        if steer_value is not None and speed_value is not None:
            break

    assert steer_value == str(expected_wire)
    assert speed_value == "420"


def test_dispatcher_sends_negative_speed_for_parking_reverse(broker) -> None:
    queues = _queues()
    sub_speed = messageHandlerSubscriber(queues, SPEED_MOTOR, "lastonly", subscribe=True)
    sub_steer = messageHandlerSubscriber(queues, STEER_MOTOR, "lastonly", subscribe=True)
    sub_speed.subscribe()
    sub_steer.subscribe()
    time.sleep(0.1)

    dispatcher = threadMotorCommandDispatcher(queues, motor_command_buffer=None)

    speed_value = None
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
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
        time.sleep(0.05)
        if speed_value is None:
            speed_value = sub_speed.receive()
        if speed_value is not None:
            break

    assert speed_value == "-100"


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

    assert "messageHandlerSender(self.queuesList, SPEED_MOTOR" not in source
    assert "messageHandlerSender(self.queuesList, STEER_MOTOR" not in source
    assert "speedMotorSender.send" not in source
    assert "steerMotorSender.send" not in source
