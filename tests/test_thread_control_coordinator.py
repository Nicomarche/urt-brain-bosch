import time
import types
import unittest

from src.control.navigation_control_policy import NavigationControlPolicy
from src.control.motor_command_dispatcher import threadControlCoordinator
from src.core.messaging.buffers import LatestValueBuffer
from src.core.types import VisualControlCandidate


class _FakeSender:
    def __init__(self):
        self.values = []

    def send(self, value):
        self.values.append(value)


class _FakeEvent:
    def __init__(self, active=False):
        self._active = bool(active)

    def is_set(self):
        return self._active


class ThreadControlCoordinatorTests(unittest.TestCase):
    def _make_coordinator(self):
        coordinator = threadControlCoordinator.__new__(threadControlCoordinator)
        coordinator.visual_candidate_buffer = LatestValueBuffer()
        coordinator.control_decision_buffer = LatestValueBuffer()
        coordinator.policy = NavigationControlPolicy()
        coordinator.speedMotorSender = _FakeSender()
        coordinator.steerMotorSender = _FakeSender()
        coordinator.tracking_state = types.SimpleNamespace(
            route_active=False,
            waypoint_mode_active=False,
            maneuver_type="none",
            last_decision=None,
        )
        coordinator.tracking_state.update_from_control_decision = (
            lambda decision: setattr(coordinator.tracking_state, "last_decision", decision)
        )
        coordinator.sign_action_event = _FakeEvent(False)
        coordinator.highway_mode_event = _FakeEvent(False)
        coordinator.steer_override_event = _FakeEvent(False)
        coordinator._consume_state_change = lambda: None
        coordinator._current_state = "AUTO"
        coordinator._sign_action_was_active = False
        coordinator._candidate_stale_timeout_s = 0.5
        coordinator._resume_min_speed = 8.0
        coordinator._resume_highway_min_speed = 15.0
        return coordinator

    def test_thread_work_sends_single_writer_commands_and_updates_tracking_state(self):
        coordinator = self._make_coordinator()
        candidate = VisualControlCandidate(
            timestamp=time.time(),
            steering_deg=6.5,
            speed_cmd=4.0,
            confidence=1.0,
            source="ai_local",
            active=True,
        )
        coordinator.visual_candidate_buffer.write(candidate, timestamp=candidate.timestamp)

        threadControlCoordinator.thread_work(coordinator)

        self.assertEqual(coordinator.steerMotorSender.values, ["65"])
        self.assertEqual(coordinator.speedMotorSender.values, ["40"])
        decision = coordinator.control_decision_buffer.read_latest()
        self.assertEqual(decision.authority, "visual")
        self.assertEqual(coordinator.tracking_state.last_decision.authority, "visual")

    def test_thread_work_blocks_stale_candidates(self):
        coordinator = self._make_coordinator()
        stale_ts = time.time() - 5.0
        candidate = VisualControlCandidate(
            timestamp=stale_ts,
            steering_deg=5.0,
            speed_cmd=3.0,
            confidence=1.0,
            source="ai_local",
            active=True,
        )
        coordinator.visual_candidate_buffer.write(candidate, timestamp=stale_ts)

        threadControlCoordinator.thread_work(coordinator)

        self.assertEqual(coordinator.steerMotorSender.values, [])
        self.assertEqual(coordinator.speedMotorSender.values, [])
        decision = coordinator.control_decision_buffer.read_latest()
        self.assertEqual(decision.reason, "stale_candidate")
        self.assertTrue(decision.safety_override)
        self.assertFalse(decision.speed_sent)
        self.assertFalse(decision.steer_sent)


if __name__ == "__main__":
    unittest.main()
