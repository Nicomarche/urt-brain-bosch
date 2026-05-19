import unittest

from src.hardware.camera.threads.threadLineFollowing import ParkingState, threadLineFollowing
from src.hardware.camera.threads.threadLocalPerception import threadLocalPerception


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
    detector.sign_actions = FakeSignActions()
    detector.sign_min_confidence = 0.5
    detector.sign_min_box_area = 0.01
    detector.sign_min_box_area_per_sign = {}
    detector._parking_sign_cooldown = 0.0
    detector._parking_sign_last_triggered = 0.0
    detector._walk_area_min_box_area = 0.04
    detector._walk_area_slow_speed = 10.0
    detector._walk_area_clear_grace = 0.5
    detector._walk_area_active = False
    detector._walk_area_mode = None
    detector._walk_area_last_seen = 0.0
    return detector


def make_line_following(payload):
    detector = threadLineFollowing.__new__(threadLineFollowing)
    detector.signDetectedSubscriber = FakeSubscriber([payload])
    detector._parking_state = ParkingState.LANE_KEEPING
    detector._parking_last_spot_box = None
    detector._parking_spot_miss_frames = 3
    detector._parking_last_spot_distance_cm = None
    detector.show_debug = False
    return detector


class ParkingTriggerTests(unittest.TestCase):
    def test_parking_sign_enters_parking_mode(self):
        detector = make_local_perception()

        detector._handle_parking_sign([
            {
                "class": "parking_sign",
                "confidence": 0.9,
                "box": [0.1, 0.1, 0.35, 0.35],
            }
        ])

        self.assertEqual(detector.stateChangeSender.values, ["PARKING"])

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

    def test_line_following_ignores_parking_sign_as_spot(self):
        detector = make_line_following({
            "sign": "parking_sign",
            "distance_cm": 25.0,
            "box": [0.1, 0.1, 0.3, 0.3],
        })

        self.assertFalse(detector._poll_sign_detected())
        self.assertEqual(detector._parking_state, ParkingState.LANE_KEEPING)
        self.assertIsNone(detector._parking_last_spot_box)

    def test_line_following_tracks_parking_area_as_spot(self):
        detector = make_line_following({
            "sign": "parking_area",
            "distance_cm": 25.0,
            "box": [0.1, 0.1, 0.3, 0.3],
        })

        self.assertTrue(detector._poll_sign_detected())
        self.assertEqual(detector._parking_state, ParkingState.SPOT_TRACKED)
        self.assertEqual(detector._parking_spot_miss_frames, 0)


if __name__ == "__main__":
    unittest.main()
