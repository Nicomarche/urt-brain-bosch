from __future__ import annotations

from queue import Queue

import pytest

from src.control.motor_command_dispatcher import threadMotorCommandDispatcher
from src.core.messaging.allMessages import SpeedMotor, SteerMotor
from src.core.types.control import MotorCommand


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

    steer_msg = queues["General"].get_nowait()
    speed_msg = queues["General"].get_nowait()

    assert steer_msg["Owner"] == SteerMotor.Owner.value
    assert steer_msg["msgID"] == SteerMotor.msgID.value
    assert steer_msg["msgValue"] == str(expected_wire)

    assert speed_msg["Owner"] == SpeedMotor.Owner.value
    assert speed_msg["msgID"] == SpeedMotor.msgID.value
    assert speed_msg["msgValue"] == "420"
