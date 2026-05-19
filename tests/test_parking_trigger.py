import unittest
import time

import config
from src.hardware.camera.threads.threadLineFollowing import ParkingState, threadLineFollowing
from src.hardware.camera.threads.threadLocalPerception import threadLocalPerception
from src.statemachine.systemMode import SystemMode


class FakeSender:
    def __init__(self):
        self.values = []

    def send(self, value):
        self.values.append(value)


class FakeSubscriber:
    def __init__(self, values=None):
        self.values = list(values or [])

    def receive(self):
        if not self.values:
            return None
        return self.values.pop(0)


class FakeEvent:
    def __init__(self):
        self.is_set = False

    def set(self):
        self.is_set = True

    def clear(self):
        self.is_set = False


class FakeSignActions:
    def __init__(self):
        self.sign_action_event = FakeEvent()
        self.speeds = []

    def _send_speed(self, speed):
        self.speeds.append(speed)


def make_local_perception():
    detector = threadLocalPerception.__new__(threadLocalPerception)
    detector._current_mode = "auto"
    detector.stateChangeSender = FakeSender()
    detector.signDetectedSender = FakeSender()
    detector.sign_actions = FakeSignActions()
    detector.enable_sign_detection = True
    detector.enable_actions = False
    detector.is_sign_actions_active = False
    detector.sign_min_confidence = 0.5
    detector.sign_min_box_area = 0.01
    detector.sign_min_box_area_per_sign = dict(getattr(config, "SIGN_MIN_BOX_AREA_PER_SIGN", {}))
    detector._parking_sign_cooldown = 0.0
    detector._parking_sign_last_triggered = 0.0
    detector._walk_area_min_box_area = 0.04
    detector._walk_area_slow_speed = 10.0
    detector._walk_area_clear_grace = 0.5
    detector._walk_area_active = False
    detector._walk_area_mode = None
    detector._walk_area_last_seen = 0.0
    detector.detection_count = 0
    detector.last_sign_name = ""
    detector.traffic_light_min_box_area = 0.01
    detector.traffic_light_classifier = None
    return detector


def make_line_following(payload, parking_state=ParkingState.LANE_KEEPING, parking_enabled=True):
    detector = threadLineFollowing.__new__(threadLineFollowing)
    detector.signDetectedSubscriber = FakeSubscriber([payload])
    detector._parking_state = parking_state
    detector._parking_last_spot_box = None
    detector._parking_spot_miss_frames = 3
    detector._parking_last_spot_distance_cm = None
    detector._parking_mode_started_at = 0.0
    detector._parking_spot_ignore_until = 0.0
    detector._parking_spot_length_cm = 0.0
    detector._parking_spot_length_px = 0.0
    detector._parking_spot_px_per_cm = 0.0
    detector._stanley_last_px_per_cm = 0.0
    detector._last_two_line_ref_px_per_cm = 0.0
    detector._last_px_per_cm = 0.0
    detector._parking_enabled = parking_enabled
    detector._parking_enabled_at = 0.0
    detector._parking_side = None
    detector._current_system_mode = SystemMode.AUTO
    detector.max_steering = 25
    detector._curve_state = "STRAIGHT"
    detector._measured_speed = 0.0
    detector.stanley_measured_speed_to_mps = 1.0
    detector.show_debug = False
    return detector


class ParkingTriggerTests(unittest.TestCase):
    def test_parking_sign_does_not_send_state_change(self):
        detector = make_local_perception()

        detector._handle_parking_sign([
            {
                "class": "parking_sign",
                "confidence": 0.9,
                "box": [0.1, 0.1, 0.35, 0.35],
            }
        ])

        self.assertEqual(detector.stateChangeSender.values, [])

    def test_parking_area_does_not_enter_parking_mode(self):
        detector = make_local_perception()

        detector._handle_parking_sign([
            {
                "class": "parking_area",
                "confidence": 0.95,
                "box": [0.1, 0.1, 0.5, 0.5],
            }
        ])

        self.assertEqual(detector.stateChangeSender.values, [])

    def test_parking_sign_after_parking_area_does_not_send_state_change(self):
        detector = make_local_perception()

        detector._handle_parking_sign([
            {
                "class": "parking_area",
                "confidence": 0.95,
                "box": [0.1, 0.1, 0.55, 0.55],
            },
            {
                "class": "parking_sign",
                "confidence": 0.9,
                "box": [0.1, 0.1, 0.16, 0.16],
            },
        ])

        self.assertEqual(detector.stateChangeSender.values, [])

    def test_legacy_parking_label_does_not_send_state_change(self):
        detector = make_local_perception()

        detector._handle_parking_sign([
            {
                "class": "parking",
                "confidence": 0.9,
                "box": [0.1, 0.1, 0.16, 0.16],
            }
        ])

        self.assertEqual(detector.stateChangeSender.values, [])

    def test_publish_sign_prioritizes_parking_sign_once_then_area(self):
        detector = make_local_perception()
        detector._parking_sign_cooldown = 8.0
        detections = [
            {
                "class": "parking_area",
                "confidence": 0.95,
                "box": [0.2, 0.55, 0.8, 0.95],
            },
            {
                "class": "parking_sign",
                "confidence": 0.9,
                "box": [0.1, 0.1, 0.2, 0.2],
            },
        ]

        detector._publish_sign(detections, now=10.0, img_shape=(480, 640))
        detector._publish_sign(detections, now=11.0, img_shape=(480, 640))

        self.assertEqual(detector.signDetectedSender.values[0]["sign"], "parking_sign")
        self.assertEqual(detector.signDetectedSender.values[1]["sign"], "parking_area")

    def test_empty_walk_area_slows_to_10_cm_s(self):
        detector = make_local_perception()

        detector._handle_walk_area([
            {
                "class": "walk_area",
                "confidence": 0.9,
                "box": [0.1, 0.1, 0.4, 0.4],
            }
        ], now=1.0)

        self.assertTrue(detector._walk_area_active)
        self.assertEqual(detector._walk_area_mode, "slow")
        self.assertTrue(detector.sign_actions.sign_action_event.is_set)
        self.assertEqual(detector.sign_actions.speeds, [10.0])
        self.assertEqual(detector.stateChangeSender.values, [])

    def test_walk_area_with_pedestrian_stops(self):
        detector = make_local_perception()

        detector._handle_walk_area([
            {
                "class": "walk_area",
                "confidence": 0.9,
                "box": [0.1, 0.1, 0.4, 0.4],
            },
            {
                "class": "pedestrian",
                "confidence": 0.9,
                "box": [0.1, 0.1, 0.3, 0.3],
            },
        ], now=1.0)

        self.assertTrue(detector._walk_area_active)
        self.assertEqual(detector._walk_area_mode, "stop")
        self.assertTrue(detector.sign_actions.sign_action_event.is_set)
        self.assertEqual(detector.sign_actions.speeds, [0])

    def test_walk_area_is_ignored_outside_auto(self):
        detector = make_local_perception()
        detector._current_mode = "manual"

        detector._handle_walk_area([
            {
                "class": "walk_area",
                "confidence": 0.9,
                "box": [0.1, 0.1, 0.4, 0.4],
            },
            {
                "class": "pedestrian",
                "confidence": 0.9,
                "box": [0.1, 0.1, 0.3, 0.3],
            },
        ], now=1.0)

        self.assertFalse(detector._walk_area_active)
        self.assertIsNone(detector._walk_area_mode)
        self.assertFalse(detector.sign_actions.sign_action_event.is_set)
        self.assertEqual(detector.sign_actions.speeds, [])

    def test_walk_area_control_clears_when_leaving_auto(self):
        detector = make_local_perception()
        detector._walk_area_active = True
        detector._walk_area_mode = "slow"
        detector.sign_actions.sign_action_event.set()
        detector._current_mode = "parking"

        detector._handle_walk_area([
            {
                "class": "walk_area",
                "confidence": 0.9,
                "box": [0.1, 0.1, 0.4, 0.4],
            }
        ], now=1.0)

        self.assertFalse(detector._walk_area_active)
        self.assertIsNone(detector._walk_area_mode)
        self.assertFalse(detector.sign_actions.sign_action_event.is_set)
        self.assertEqual(detector.sign_actions.speeds, [])

    def test_walk_area_passed_resumes_normal_without_parking(self):
        detector = make_local_perception()
        detector._walk_area_active = True
        detector._walk_area_mode = "slow"
        detector._walk_area_last_seen = 1.0
        detector.sign_actions.sign_action_event.set()

        detector._handle_walk_area([], now=2.0)

        self.assertFalse(detector._walk_area_active)
        self.assertIsNone(detector._walk_area_mode)
        self.assertFalse(detector.sign_actions.sign_action_event.is_set)
        self.assertEqual(detector.stateChangeSender.values, [])
        self.assertEqual(detector.sign_actions.speeds, [])

    def test_line_following_parking_sign_enables_search_without_tracking_spot(self):
        detector = make_line_following({
            "sign": "parking_sign",
            "confidence": 0.9,
            "box_area": 0.01,
            "distance_cm": 25.0,
            "box": [0.1, 0.1, 0.3, 0.3],
        }, parking_state=ParkingState.IDLE, parking_enabled=False)

        self.assertFalse(detector._poll_sign_detected())
        self.assertEqual(detector._parking_state, ParkingState.LANE_KEEPING)
        self.assertTrue(detector._parking_enabled)
        self.assertIsNone(detector._parking_last_spot_box)

    def test_line_following_legacy_parking_label_enables_search(self):
        detector = make_line_following({
            "sign": "parking",
            "confidence": 0.9,
            "box_area": 0.01,
            "distance_cm": 25.0,
            "box": [0.1, 0.1, 0.3, 0.3],
        }, parking_state=ParkingState.IDLE, parking_enabled=False)

        self.assertFalse(detector._poll_sign_detected())
        self.assertEqual(detector._parking_state, ParkingState.LANE_KEEPING)
        self.assertTrue(detector._parking_enabled)
        self.assertIsNone(detector._parking_last_spot_box)

    def test_line_following_ignores_parking_area_until_sign_enabled(self):
        detector = make_line_following({
            "sign": "parking_area",
            "distance_cm": 25.0,
            "box": [0.2, 0.55, 0.8, 0.95],
        }, parking_state=ParkingState.IDLE, parking_enabled=False)

        self.assertFalse(detector._poll_sign_detected())
        self.assertEqual(detector._parking_state, ParkingState.IDLE)
        self.assertIsNone(detector._parking_last_spot_box)

    def test_line_following_accepts_area_with_recent_sign_hint(self):
        detector = make_line_following({
            "sign": "parking_area",
            "distance_cm": 25.0,
            "timestamp": 11.0,
            "parking_sign_recent": True,
            "parking_sign_last_seen": 10.0,
            "box": [0.2, 0.55, 0.8, 0.95],
        }, parking_state=ParkingState.IDLE, parking_enabled=False)

        self.assertTrue(detector._poll_sign_detected())
        self.assertTrue(detector._parking_enabled)
        self.assertEqual(detector._parking_state, ParkingState.SPOT_TRACKED)
        self.assertEqual(detector._parking_side, "right")

    def test_line_following_ignores_small_area_like_parking_sign(self):
        detector = make_line_following({
            "sign": "parking_area",
            "distance_cm": 25.0,
            "box_area": 0.01,
            "box": [0.1, 0.1, 0.2, 0.2],
        })

        self.assertFalse(detector._poll_sign_detected())
        self.assertEqual(detector._parking_state, ParkingState.LANE_KEEPING)
        self.assertIsNone(detector._parking_last_spot_box)

    def test_line_following_tracks_parking_area_as_spot(self):
        detector = make_line_following({
            "sign": "parking_area",
            "distance_cm": 25.0,
            "box": [0.2, 0.1, 0.8, 0.5],
        })

        self.assertTrue(detector._poll_sign_detected())
        self.assertEqual(detector._parking_state, ParkingState.SPOT_TRACKED)
        self.assertEqual(detector._parking_spot_miss_frames, 0)

    def test_line_following_sets_parking_side_from_area_box(self):
        detector = make_line_following({
            "sign": "parking_area",
            "distance_cm": 25.0,
            "box": [0.2, 0.05, 0.8, 0.45],
        })

        self.assertTrue(detector._poll_sign_detected())
        self.assertEqual(detector._parking_side, "left")

    def test_left_parking_mirrors_maneuver_steering(self):
        detector = make_line_following({}, parking_state=ParkingState.WAIT_STEER_1)
        detector._parking_side = "left"
        detector._parking_phase_start_time = time.time()
        detector._parking_phase_dist_cm = 0.0
        detector._parking_last_sm_time = time.time()

        steer, speed = detector._parking_state_machine()

        self.assertLess(steer, 0)
        self.assertEqual(speed, 0)

    def test_line_following_measures_parking_area_length_from_bbox_px(self):
        detector = make_line_following({
            "sign": "parking_area",
            "distance_cm": 25.0,
            "box_px_width": 800.0,
            "box_px_height": 280.0,
            "box": [0.1, 0.1, 0.5, 0.7],
            "box_area": 0.24,
        })
        detector._stanley_last_px_per_cm = 10.0

        self.assertTrue(detector._poll_sign_detected())

        self.assertEqual(detector._parking_state, ParkingState.SPOT_TRACKED)
        self.assertAlmostEqual(detector._parking_spot_length_cm, 80.0)
        self.assertAlmostEqual(detector._parking_forward_target_cm(), 80.0)

    def test_line_following_ignores_spot_seen_before_parking_mode(self):
        detector = make_line_following({
            "sign": "parking_area",
            "distance_cm": 25.0,
            "timestamp": 9.0,
            "box": [0.2, 0.1, 0.8, 0.5],
        })
        detector._parking_mode_started_at = 10.0

        self.assertFalse(detector._poll_sign_detected())
        self.assertEqual(detector._parking_state, ParkingState.LANE_KEEPING)
        self.assertIsNone(detector._parking_last_spot_box)

    def test_line_following_ignores_initial_spot_during_parking_arm_delay(self):
        detector = make_line_following({
            "sign": "parking_area",
            "distance_cm": 25.0,
            "timestamp": time.time(),
            "box": [0.2, 0.1, 0.8, 0.5],
        })
        detector._parking_mode_started_at = time.time() - 0.1
        detector._parking_spot_ignore_until = time.time() + 10.0

        self.assertFalse(detector._poll_sign_detected())
        self.assertEqual(detector._parking_state, ParkingState.LANE_KEEPING)
        self.assertIsNone(detector._parking_last_spot_box)


if __name__ == "__main__":
    unittest.main()
