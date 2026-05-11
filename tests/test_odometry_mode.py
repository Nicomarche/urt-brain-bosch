from src.control.motor_command_dispatcher import threadMotorCommandDispatcher
from src.core.types.control import MotorCommand
from src.core.types.pose import Pose2D, PoseEstimate
from src.localization.relocalization_thread import threadTracking
import src.localization.relocalization_thread as relocalization
from src.statemachine.systemMode import SystemMode
from src.statemachine.transitionTable import TransitionTable


class _FakeSub:
    def __init__(self, values):
        self._values = list(values)

    def receive(self):
        if not self._values:
            return None
        return self._values.pop(0)


class _Sender:
    def __init__(self):
        self.values = []

    def send(self, value):
        self.values.append(value)


class _Buffer:
    def __init__(self, value=None, timestamp=0.0):
        self._value = value
        self._timestamp = timestamp

    def read_latest(self, with_metadata=False):
        if with_metadata:
            return self._value, self._timestamp, 1
        return self._value


def test_odometry_transition_from_stop_is_valid() -> None:
    transition = TransitionTable.get_next_mode(SystemMode.STOP, "dashboard_odometry_button")

    assert transition["transition_valid"] is True
    assert transition["next_mode"] is SystemMode.ODOMETRY


def test_auto_transition_from_stop_still_goes_to_auto() -> None:
    transition = TransitionTable.get_next_mode(SystemMode.STOP, "dashboard_auto_button")

    assert transition["transition_valid"] is True
    assert transition["next_mode"] is SystemMode.AUTO


def test_odometry_state_forces_command_speed_even_when_encoder_enabled(monkeypatch) -> None:
    old_use_encoder = relocalization._USE_ENCODER
    try:
        relocalization._USE_ENCODER = True
        tracker = threadTracking.__new__(threadTracking)
        tracker._speed_sub = _FakeSub(["500"])
        tracker._speed_cmd_sub = _FakeSub(["200"])
        tracker._last_raw_speed = None
        tracker._last_speed_t = None
        tracker._last_speed = 0.0
        tracker._last_speed_source = "none"
        tracker._last_cmd_speed_raw = 0.0
        tracker._last_cmd_speed_t = None
        tracker._current_state_message = "ODOMETRY"

        speed_mps = tracker._resolve_speed_mps(now=10.0)
    finally:
        relocalization._USE_ENCODER = old_use_encoder

    assert speed_mps == 0.200
    assert tracker._last_speed_source == "command_odometry"


def test_dispatcher_sends_motor_command_in_odometry_mode(monkeypatch) -> None:
    cmd = MotorCommand(
        timestamp=100.0,
        steering_deg=0.0,
        speed_mps=0.20,
        valid=True,
        source="test",
        reason="passthrough",
    )
    dispatcher = threadMotorCommandDispatcher.__new__(threadMotorCommandDispatcher)
    dispatcher._current_state = "ODOMETRY"
    dispatcher._stateChangeSubscriber = _FakeSub([])
    dispatcher._motor_cmd_buf = _Buffer(cmd)
    dispatcher._behavior_buf = _Buffer(object(), timestamp=100.0)
    dispatcher._pose_buf = _Buffer(
        PoseEstimate(
            timestamp=100.0,
            fused_pose=Pose2D(),
            raw_pose=Pose2D(),
        ),
        timestamp=100.0,
    )
    dispatcher._gate = type("Gate", (), {"evaluate": lambda self, **_kwargs: cmd})()
    dispatcher._motorCmdMsgSender = _Sender()
    dispatcher._speedMotorSender = _Sender()
    dispatcher._steerMotorSender = _Sender()
    dispatcher._stats_count = 0
    dispatcher._stats_last_t = 100.0
    dispatcher._stats_period_s = 999.0
    dispatcher._ipc_mode_received = True
    dispatcher._sm_state = None
    dispatcher._sm_lock = None
    monkeypatch.setattr("src.control.motor_command_dispatcher.time.time", lambda: 100.0)

    dispatcher.thread_work()

    assert dispatcher._speedMotorSender.values == ["200"]
    assert dispatcher._steerMotorSender.values == ["0"]
