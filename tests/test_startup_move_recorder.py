import json
import os
import tempfile
import time
import unittest

from src.hardware.camera.threads.threadLineFollowing import ParkingState, threadLineFollowing
from src.statemachine.systemMode import SystemMode


class FakeSender:
    def __init__(self):
        self.values = []

    def send(self, value):
        self.values.append(value)


class FakeSubscriber:
    def __init__(self, values=None):
        self.values = list(values or [])
        self.emptied = False

    def receive(self):
        if not self.values:
            return None
        return self.values.pop(0)

    def empty(self):
        self.values = []
        self.emptied = True


class StartupMoveRecorderTests(unittest.TestCase):
    def _make_detector(self, tmpdir):
        detector = threadLineFollowing.__new__(threadLineFollowing)
        detector.startup_move_path = os.path.join(tmpdir, "startup_manual_trajectory.json")
        detector.startup_move_max_duration_s = 20.0
        detector.startup_move_auto_replay = True
        detector._current_system_mode = SystemMode.MANUAL
        detector._startup_move_recording = False
        detector._startup_move_record_start_monotonic = 0.0
        detector._startup_move_samples = []
        detector._startup_move_loaded = None
        detector._startup_move_error = None
        detector._startup_replay_active = False
        detector._startup_replay_samples = []
        detector._startup_replay_started_at = 0.0
        detector._startup_replay_next_index = 0
        detector._startup_replay_duration_s = 0.0
        detector._startup_replay_last_sample = None
        detector._startup_replay_pending_green = False
        detector._startup_replay_paused_since = 0.0
        detector.startup_move_wait_for_green_light = True
        detector._manual_last_speed_x10 = 0
        detector._manual_last_steer_x10 = 0
        detector._startup_move_last_status_snapshot = None
        detector._startup_move_last_status_ts = 0.0
        detector.startupMoveStatusSender = FakeSender()
        detector.startupMoveSpeedSubscriber = FakeSubscriber()
        detector.startupMoveSteerSubscriber = FakeSubscriber()
        detector.steerMotorSender = FakeSender()
        detector.speedMotorSender = FakeSender()
        detector.stateChangeSubscriber = FakeSubscriber()
        detector._parking_state = ParkingState.IDLE
        detector._auto_run_log_enabled = False
        detector._auto_run_log_frame_idx = 0
        detector.auto_run_log_path = os.path.join(tmpdir, "auto_run_log.txt")
        detector.detection_mode = "ai_local"
        detector._follow_right_only = False
        detector._follow_right_miss_frames = 0
        detector._follow_right_last_right_x = None
        detector._last_inactive_log = False
        detector.is_line_following_active = False
        detector.base_speed = 15.0
        detector._current_speed = 0.0
        detector.show_debug = False
        detector._last_motor_tx_log = 0.0
        detector._last_requested_motor_command = None
        detector._last_state_change_message = None
        detector._reset_pid_state = lambda: None
        detector.traffic_light_hold_enabled = True
        detector.traffic_light_hold_timeout_s = 1.5
        detector.traffic_light_green_confirmations = 2
        detector._traffic_light_last_seen = 0.0
        detector._traffic_light_last_state = ""
        detector._traffic_light_last_color = "unknown"
        detector._traffic_light_last_reason = ""
        detector._traffic_light_last_box_area = 0.0
        detector._traffic_light_candidate_state = ""
        detector._traffic_light_candidate_count = 0
        detector._traffic_light_candidate_last_seen = 0.0
        detector._traffic_light_holding = False
        detector._traffic_light_last_log = 0.0
        return detector

    def test_start_record_stop_saves_json_trajectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = self._make_detector(tmpdir)

            self.assertTrue(detector._start_startup_move_recording())
            detector._append_startup_move_sample(speed_x10=100)
            detector._append_startup_move_sample(steer_x10=-50)
            detector._startup_move_record_start_monotonic -= 2.0
            self.assertTrue(detector._stop_startup_move_recording())

            with open(detector.startup_move_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertEqual(payload["version"], 1)
            self.assertGreater(payload["duration_s"], 1.0)
            self.assertGreaterEqual(len(payload["samples"]), 3)
            self.assertEqual(payload["samples"][-1]["speed_x10"], 100)
            self.assertEqual(payload["samples"][-1]["steer_x10"], -50)

    def test_load_startup_move_normalizes_first_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = self._make_detector(tmpdir)
            os.makedirs(os.path.dirname(detector.startup_move_path), exist_ok=True)
            with open(detector.startup_move_path, "w", encoding="utf-8") as f:
                json.dump({
                    "version": 1,
                    "created_at": 1.0,
                    "duration_s": 2.0,
                    "samples": [
                        {"t": 5.0, "speed_x10": 10, "steer_x10": 0},
                        {"t": 6.5, "speed_x10": 20, "steer_x10": 5},
                    ],
                }, f)

            trajectory = detector._load_startup_move_trajectory()

            self.assertIsNotNone(trajectory)
            self.assertEqual(trajectory["samples"][0]["t"], 0.0)
            self.assertEqual(trajectory["duration_s"], 1.5)

    def test_entering_auto_with_trajectory_and_no_traffic_light_starts_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = self._make_detector(tmpdir)
            detector._startup_move_loaded = {
                "version": 1,
                "created_at": 1.0,
                "duration_s": 0.5,
                "samples": [{"t": 0.0, "speed_x10": 100, "steer_x10": 0}],
            }
            with open(detector.startup_move_path, "w", encoding="utf-8") as f:
                json.dump(detector._startup_move_loaded, f)
            detector.stateChangeSubscriber = FakeSubscriber(["AUTO"])

            detector.check_state_change()

            self.assertTrue(detector._startup_replay_active)
            self.assertFalse(detector._startup_replay_pending_green)
            self.assertFalse(detector.is_line_following_active)

    def test_entering_auto_with_red_light_waits_for_green_before_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = self._make_detector(tmpdir)
            detector._startup_move_loaded = {
                "version": 1,
                "created_at": 1.0,
                "duration_s": 0.5,
                "samples": [{"t": 0.0, "speed_x10": 100, "steer_x10": 0}],
            }
            with open(detector.startup_move_path, "w", encoding="utf-8") as f:
                json.dump(detector._startup_move_loaded, f)
            detector.stateChangeSubscriber = FakeSubscriber(["AUTO"])
            now = time.time()
            detector._update_traffic_light_hold({
                "sign": "red_light",
                "traffic_light_state": "red_light",
                "box_area": 0.05,
            }, now=now)

            detector.check_state_change()

            self.assertFalse(detector._startup_replay_active)
            self.assertTrue(detector._startup_replay_pending_green)
            self.assertFalse(detector.is_line_following_active)

            detector._update_traffic_light_hold({
                "sign": "green_light",
                "traffic_light_state": "green_light",
                "box_area": 0.05,
            }, now=now + 0.1)
            detector._update_traffic_light_hold({
                "sign": "green_light",
                "traffic_light_state": "green_light",
                "box_area": 0.05,
            }, now=now + 0.2)
            self.assertTrue(detector._maybe_start_pending_startup_replay())
            self.assertTrue(detector._startup_replay_active)
            self.assertFalse(detector._startup_replay_pending_green)
            self.assertFalse(detector.is_line_following_active)

    def test_entering_auto_without_trajectory_goes_directly_to_line_following(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = self._make_detector(tmpdir)
            detector.stateChangeSubscriber = FakeSubscriber(["AUTO"])

            detector.check_state_change()

            self.assertFalse(detector._startup_replay_active)
            self.assertTrue(detector.is_line_following_active)

    def test_leaving_auto_cancels_replay_and_sends_neutral_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = self._make_detector(tmpdir)
            detector._current_system_mode = SystemMode.AUTO
            detector._startup_replay_active = True
            detector._startup_replay_samples = [{"t": 0.0, "speed_x10": 100, "steer_x10": 20}]
            detector._startup_replay_duration_s = 1.0
            detector.stateChangeSubscriber = FakeSubscriber(["MANUAL"])

            detector.check_state_change()

            self.assertFalse(detector._startup_replay_active)
            self.assertEqual(detector.steerMotorSender.values[-1], "0")
            self.assertEqual(detector.speedMotorSender.values[-1], "0")

    def test_startup_replay_pauses_under_red_light_hold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = self._make_detector(tmpdir)
            detector._current_system_mode = SystemMode.AUTO
            detector._startup_replay_active = True
            detector._startup_replay_samples = [{"t": 0.0, "speed_x10": 100, "steer_x10": 20}]
            detector._startup_replay_duration_s = 1.0
            detector._startup_replay_started_at = time.monotonic() - 0.1
            now = time.time()
            detector._update_traffic_light_hold({
                "sign": "red_light",
                "traffic_light_state": "red_light",
                "box_area": 0.05,
            }, now=now)

            self.assertTrue(detector._step_startup_replay())
            self.assertEqual(detector._startup_replay_next_index, 0)
            self.assertEqual(detector.speedMotorSender.values[-1], "0")

            detector._update_traffic_light_hold({
                "sign": "green_light",
                "traffic_light_state": "green_light",
                "box_area": 0.05,
            }, now=now + 0.1)
            detector._update_traffic_light_hold({
                "sign": "green_light",
                "traffic_light_state": "green_light",
                "box_area": 0.05,
            }, now=now + 0.2)

            self.assertTrue(detector._step_startup_replay())
            self.assertEqual(detector._startup_replay_next_index, 1)
            self.assertEqual(detector.steerMotorSender.values[-1], "20")
            self.assertEqual(detector.speedMotorSender.values[-1], "100")


if __name__ == "__main__":
    unittest.main()
