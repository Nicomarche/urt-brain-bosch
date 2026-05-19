import unittest

import cv2
import numpy as np

from src.hardware.camera.threads.threadLineFollowing import threadLineFollowing
from src.hardware.camera.threads.threadLocalPerception import threadLocalPerception
from src.hardware.camera.threads.trafficLightClassifier import TrafficLightClassifier
from src.statemachine.systemMode import SystemMode


class FakeSender:
    def __init__(self):
        self.values = []

    def send(self, value):
        self.values.append(value)


class FakeSignActions:
    def __init__(self):
        self.executed = []

    def execute(self, sign_name, curve_state=None, steering_deg=0.0):
        self.executed.append((sign_name, curve_state, steering_deg))
        return True


def draw_synthetic_light(color, frame_shape=(160, 120), box=(0.1, 0.35, 0.9, 0.65)):
    frame = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
    height, width = frame_shape
    y1, x1, y2, x2 = box
    px_y1 = int(y1 * height)
    px_x1 = int(x1 * width)
    px_y2 = int(y2 * height)
    px_x2 = int(x2 * width)
    roi_h = px_y2 - px_y1
    roi_w = px_x2 - px_x1

    circle_diameter = max(1, int(roi_h * 0.9 / 3.0))
    red_y = int(circle_diameter / 2.0 + roi_h / 32.0)
    yellow_y = int(red_y + circle_diameter + roi_h / 32.0)
    green_y = int(yellow_y + circle_diameter + roi_h / 32.0)
    centers = {
        "red": red_y,
        "yellow": yellow_y,
        "green": green_y,
    }
    bgr = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
    }
    cv2.circle(
        frame,
        (px_x1 + roi_w // 2, px_y1 + centers[color]),
        max(4, circle_diameter // 3),
        bgr[color],
        -1,
        cv2.LINE_AA,
    )
    return frame


def make_local_perception():
    detector = threadLocalPerception.__new__(threadLocalPerception)
    detector.enable_sign_detection = True
    detector.enable_actions = True
    detector.is_sign_actions_active = True
    detector.sign_min_confidence = 0.5
    detector.sign_min_box_area = 0.01
    detector.sign_min_box_area_per_sign = {}
    detector.traffic_light_opencv_enabled = True
    detector.traffic_light_min_box_area = 0.01
    detector.traffic_light_classifier = TrafficLightClassifier()
    detector._current_mode = "auto"
    detector._walk_area_active = False
    detector._lf_curve_state = "STRAIGHT"
    detector._lf_steering_deg = 0.0
    detector.detection_count = 0
    detector.last_sign_name = ""
    detector.signDetectedSender = FakeSender()
    detector.sign_actions = FakeSignActions()
    return detector


def make_line_following():
    detector = threadLineFollowing.__new__(threadLineFollowing)
    detector._current_system_mode = SystemMode.AUTO
    detector.traffic_light_hold_enabled = True
    detector.traffic_light_hold_timeout_s = 0.6
    detector.traffic_light_min_box_area = 0.01
    detector._traffic_light_last_seen = 0.0
    detector._traffic_light_last_state = ""
    detector._traffic_light_last_color = "unknown"
    detector._traffic_light_last_reason = ""
    detector._traffic_light_last_box_area = 0.0
    detector._traffic_light_holding = False
    detector._traffic_light_last_log = 0.0
    detector.show_debug = False
    return detector


class TrafficLightOpenCVTests(unittest.TestCase):
    def test_classifier_detects_red_yellow_green(self):
        classifier = TrafficLightClassifier()
        box = (0.1, 0.35, 0.9, 0.65)

        for color in ("red", "yellow", "green"):
            frame = draw_synthetic_light(color, box=box)
            result = classifier.classify(frame, box)

            self.assertEqual(result["color"], color)
            self.assertEqual(result["sign"], f"{color}_light")

    def test_classifier_returns_unknown_for_empty_crop(self):
        classifier = TrafficLightClassifier()
        frame = np.zeros((160, 120, 3), dtype=np.uint8)

        result = classifier.classify(frame, (0.1, 0.35, 0.9, 0.65))

        self.assertEqual(result["color"], "unknown")
        self.assertEqual(result["sign"], "traffic_light_unknown")

    def test_classifier_clamps_out_of_bounds_bbox(self):
        classifier = TrafficLightClassifier()
        frame = draw_synthetic_light("green", box=(0.0, 0.0, 1.0, 1.0))

        result = classifier.classify(frame, (-0.2, -0.2, 1.2, 1.2))

        self.assertEqual(result["sign"], "green_light")
        self.assertEqual(result["crop"]["box"], [0.0, 0.0, 1.0, 1.0])

    def test_local_perception_publishes_classified_green_light(self):
        detector = make_local_perception()
        box = (0.1, 0.35, 0.9, 0.65)
        frame = draw_synthetic_light("green", box=box)

        detector._publish_sign(
            [{"class": "traffic_light", "confidence": 0.9, "box": box}],
            now=1.0,
            img_shape=frame.shape[:2],
            frame=frame,
        )

        payload = detector.signDetectedSender.values[-1]
        self.assertEqual(payload["sign"], "green_light")
        self.assertEqual(payload["traffic_light_color"], "green")
        self.assertEqual(detector.sign_actions.executed, [])

    def test_local_perception_publishes_classified_red_light_without_sign_action(self):
        detector = make_local_perception()
        box = (0.1, 0.35, 0.9, 0.65)
        frame = draw_synthetic_light("red", box=box)

        detector._publish_sign(
            [{"class": "traffic_light", "confidence": 0.9, "box": box}],
            now=1.0,
            img_shape=frame.shape[:2],
            frame=frame,
        )

        payload = detector.signDetectedSender.values[-1]
        self.assertEqual(payload["sign"], "red_light")
        self.assertEqual(payload["traffic_light_state"], "red_light")
        self.assertEqual(detector.sign_actions.executed, [])

    def test_line_following_blocks_auto_speed_for_red_yellow_and_unknown(self):
        for sign in ("red_light", "yellow_light", "traffic_light_unknown"):
            detector = make_line_following()
            detector._update_traffic_light_hold(
                {
                    "sign": sign,
                    "traffic_light_state": sign,
                    "box_area": 0.05,
                },
                now=10.0,
            )

            speed, holding = detector._guard_speed_for_traffic_light(15.0, now=10.1)

            self.assertTrue(holding, sign)
            self.assertEqual(speed, 0.0)

    def test_line_following_allows_auto_speed_for_green(self):
        detector = make_line_following()
        detector._update_traffic_light_hold(
            {
                "sign": "green_light",
                "traffic_light_state": "green_light",
                "box_area": 0.05,
            },
            now=10.0,
        )

        speed, holding = detector._guard_speed_for_traffic_light(15.0, now=10.1)

        self.assertFalse(holding)
        self.assertEqual(speed, 15.0)

    def test_line_following_releases_hold_after_timeout(self):
        detector = make_line_following()
        detector._update_traffic_light_hold(
            {
                "sign": "red_light",
                "traffic_light_state": "red_light",
                "box_area": 0.05,
            },
            now=10.0,
        )

        speed, holding = detector._guard_speed_for_traffic_light(15.0, now=11.0)

        self.assertFalse(holding)
        self.assertEqual(speed, 15.0)


if __name__ == "__main__":
    unittest.main()
