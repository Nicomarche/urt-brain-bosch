import math

import cv2
import numpy as np

import config


class TrafficLightClassifier:
    """Classify the active color inside a traffic-light bounding box with OpenCV."""

    COLOR_TO_SIGN = {
        "red": "red_light",
        "yellow": "yellow_light",
        "green": "green_light",
    }
    SIGN_TO_COLOR = {
        "red_light": "red",
        "yellow_light": "yellow",
        "green_light": "green",
    }
    UNKNOWN_COLOR = "unknown"
    UNKNOWN_SIGN = "traffic_light_unknown"

    def __init__(self):
        self.red_low1, self.red_high1 = self._range_from_config(
            "TRAFFIC_LIGHT_RED_HSV_1",
            ((0, 80, 80), (12, 255, 255)),
        )
        self.red_low2, self.red_high2 = self._range_from_config(
            "TRAFFIC_LIGHT_RED_HSV_2",
            ((165, 80, 80), (180, 255, 255)),
        )
        self.yellow_low, self.yellow_high = self._range_from_config(
            "TRAFFIC_LIGHT_YELLOW_HSV",
            ((22, 150, 150), (28, 255, 255)),
        )
        self.green_low, self.green_high = self._range_from_config(
            "TRAFFIC_LIGHT_GREEN_HSV",
            ((40, 80, 80), (95, 255, 255)),
        )
        self.active_colors = self._active_colors_from_config()
        self.hsv_color_min_ratio = float(getattr(config, "TRAFFIC_LIGHT_HSV_COLOR_MIN_RATIO", 0.60))
        self.hsv_color_min_area_ratio = float(getattr(config, "TRAFFIC_LIGHT_HSV_COLOR_MIN_AREA_RATIO", 0.005))
        self.hsv_color_dominance = float(getattr(config, "TRAFFIC_LIGHT_HSV_COLOR_DOMINANCE", 1.15))
        self.adaptive_sat_min = float(getattr(config, "TRAFFIC_LIGHT_ADAPTIVE_SAT_MIN", 100.0))
        self.adaptive_sat_max = float(getattr(config, "TRAFFIC_LIGHT_ADAPTIVE_SAT_MAX", 200.0))
        self.adaptive_val_min = float(getattr(config, "TRAFFIC_LIGHT_ADAPTIVE_VAL_MIN", 50.0))
        self.adaptive_val_max = float(getattr(config, "TRAFFIC_LIGHT_ADAPTIVE_VAL_MAX", 150.0))
        self.min_circularity = float(getattr(config, "TRAFFIC_LIGHT_MIN_CIRCULARITY", 0.357))
        self.min_light_height_ratio = float(getattr(config, "TRAFFIC_LIGHT_MIN_LIGHT_HEIGHT_RATIO", 0.08))
        self.max_light_height_ratio = float(getattr(config, "TRAFFIC_LIGHT_MAX_LIGHT_HEIGHT_RATIO", 0.40))
        self.max_center_x_offset_ratio = float(getattr(config, "TRAFFIC_LIGHT_MAX_CENTER_X_OFFSET_RATIO", 0.20))
        self.brightness_threshold_scale = float(
            getattr(config, "TRAFFIC_LIGHT_BRIGHTNESS_THRESHOLD_SCALE", 0.8595)
        )
        self.brightness_threshold_min = float(
            getattr(config, "TRAFFIC_LIGHT_BRIGHTNESS_THRESHOLD_MIN", 0.20)
        )
        self.backup_color_ratio_min = float(getattr(config, "TRAFFIC_LIGHT_BACKUP_COLOR_RATIO_MIN", 0.20))

    @staticmethod
    def _range_from_config(name, default):
        raw = getattr(config, name, default)
        try:
            lower, upper = raw
            return np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)
        except Exception:
            lower, upper = default
            return np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)

    @classmethod
    def _active_colors_from_config(cls):
        raw_colors = getattr(config, "TRAFFIC_LIGHT_ACTIVE_COLORS", ("red", "green"))
        if isinstance(raw_colors, str):
            raw_colors = [raw_colors]
        try:
            iterator = iter(raw_colors)
        except TypeError:
            iterator = iter(("red", "green"))

        colors = []
        for color in iterator:
            normalized = str(color or "").strip().lower()
            if normalized in cls.COLOR_TO_SIGN and normalized not in colors:
                colors.append(normalized)
        return tuple(colors) or ("red", "green")

    @staticmethod
    def normalize_sign_name(sign_name):
        return str(sign_name or "").strip().lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def is_traffic_light_sign(cls, sign_name):
        sign_name = cls.normalize_sign_name(sign_name)
        return sign_name in {
            "traffic_light",
            "traffic_light_unknown",
            "red",
            "yellow",
            "green",
            "red_light",
            "yellow_light",
            "green_light",
        }

    @classmethod
    def sign_for_color(cls, color):
        return cls.COLOR_TO_SIGN.get(str(color or "").lower(), cls.UNKNOWN_SIGN)

    @classmethod
    def color_for_sign(cls, sign_name):
        sign_name = cls.normalize_sign_name(sign_name)
        if sign_name in cls.COLOR_TO_SIGN:
            return sign_name
        if sign_name in cls.SIGN_TO_COLOR:
            return cls.SIGN_TO_COLOR[sign_name]
        return cls.UNKNOWN_COLOR

    @classmethod
    def payload_for_known_sign(cls, sign_name, reason="model_class"):
        sign_name = cls.normalize_sign_name(sign_name)
        if sign_name in cls.COLOR_TO_SIGN:
            sign_name = cls.COLOR_TO_SIGN[sign_name]
        color = cls.color_for_sign(sign_name)
        state = cls.sign_for_color(color)
        return {
            "sign": state,
            "color": color,
            "state": state,
            "reason": reason,
            "scores": {},
        }

    def classify(self, frame, box):
        roi, crop_info = self.crop_normalized_box(frame, box)
        if roi is None:
            return self._unknown(crop_info.get("reason", "invalid_roi"), crop_info=crop_info)
        result = self.classify_crop(roi)
        result["crop"] = crop_info
        return result

    def crop_normalized_box(self, frame, box):
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            return None, {"reason": "empty_frame"}
        if box is None or len(box) != 4:
            return None, {"reason": "invalid_box"}

        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            return None, {"reason": "empty_frame"}

        try:
            y1, x1, y2, x2 = [float(v) for v in box]
        except (TypeError, ValueError):
            return None, {"reason": "invalid_box"}

        values = (y1, x1, y2, x2)
        if any(math.isnan(v) or math.isinf(v) for v in values):
            return None, {"reason": "invalid_box"}

        y1 = max(0.0, min(1.0, y1))
        x1 = max(0.0, min(1.0, x1))
        y2 = max(0.0, min(1.0, y2))
        x2 = max(0.0, min(1.0, x2))
        if y2 <= y1 or x2 <= x1:
            return None, {
                "reason": "empty_box",
                "box": [round(y1, 5), round(x1, 5), round(y2, 5), round(x2, 5)],
            }

        px_y1 = max(0, min(height - 1, int(math.floor(y1 * height))))
        px_x1 = max(0, min(width - 1, int(math.floor(x1 * width))))
        px_y2 = max(px_y1 + 1, min(height, int(math.ceil(y2 * height))))
        px_x2 = max(px_x1 + 1, min(width, int(math.ceil(x2 * width))))

        roi = frame[px_y1:px_y2, px_x1:px_x2]
        if roi.size == 0:
            return None, {"reason": "empty_roi"}

        return roi, {
            "reason": "ok",
            "box": [round(y1, 5), round(x1, 5), round(y2, 5), round(x2, 5)],
            "pixel_box": [px_y1, px_x1, px_y2, px_x2],
        }

    def classify_crop(self, detected_light):
        if detected_light is None or detected_light.size == 0:
            return self._unknown("empty_crop")

        hsv = cv2.cvtColor(detected_light, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        value_channel = hsv[:, :, 2]
        baseline_brightness = float(cv2.mean(value_channel)[0]) / 255.0

        _, bright_mask = cv2.threshold(
            value_channel,
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        hsv_color, hsv_scores = self._classify_by_hsv_color(hsv)
        if hsv_color is not None:
            result = self._known(hsv_color, "hsv_color", baseline_brightness)
            result["scores"] = hsv_scores
            return result

        slot_geometry = self._slot_geometry(detected_light.shape[:2])
        position_color = self._classify_by_bright_contour_position(
            contours,
            detected_light.shape[:2],
            slot_geometry,
        )
        if position_color is not None:
            return self._known(position_color, "contour_position", baseline_brightness)

        brightness_color, brightness_scores = self._classify_by_slot_brightness(
            hsv,
            slot_geometry,
            baseline_brightness,
        )
        if brightness_color is not None:
            result = self._known(brightness_color, "slot_brightness", baseline_brightness)
            result["scores"] = brightness_scores
            return result

        backup_color, backup_scores = self._classify_by_hsv_backup(hsv, contours)
        if backup_color is not None:
            result = self._known(backup_color, "hsv_backup", baseline_brightness)
            result["scores"] = backup_scores
            return result

        return self._unknown(
            "undetermined",
            baseline_brightness=baseline_brightness,
            scores={**brightness_scores, **backup_scores},
        )

    def _color_masks(self, hsv):
        red_mask1 = cv2.inRange(hsv, self.red_low1, self.red_high1)
        red_mask2 = cv2.inRange(hsv, self.red_low2, self.red_high2)
        return {
            "red": cv2.bitwise_or(red_mask1, red_mask2),
            "yellow": cv2.inRange(hsv, self.yellow_low, self.yellow_high),
            "green": cv2.inRange(hsv, self.green_low, self.green_high),
        }

    def _classify_by_hsv_color(self, hsv):
        img_height, img_width = hsv.shape[:2]
        total_area = float(max(1, img_height * img_width))
        masks = self._color_masks(hsv)
        colors = tuple(self.COLOR_TO_SIGN.keys())
        counts = {
            color: float(cv2.countNonZero(masks[color]))
            for color in colors
        }
        scores = {
            f"{color}_area_ratio": counts[color] / total_area
            for color in colors
        }
        colored_pixels = sum(counts.values())
        if colored_pixels <= 0.0:
            scores["colored_pixels"] = 0.0
            return None, scores

        best_color = max(colors, key=lambda color: counts[color])
        best_count = counts[best_color]
        second_count = max(
            (count for color, count in counts.items() if color != best_color),
            default=0.0,
        )
        best_share = best_count / colored_pixels
        best_area_ratio = best_count / total_area
        scores.update({
            "colored_pixels": colored_pixels,
            "best_color": best_color,
            "best_share": best_share,
            "best_area_ratio": best_area_ratio,
        })

        if best_area_ratio < self.hsv_color_min_area_ratio:
            return None, scores
        if best_share < self.hsv_color_min_ratio:
            return None, scores
        if second_count > 0.0 and best_count < second_count * self.hsv_color_dominance:
            return None, scores
        return best_color, scores

    def _slot_geometry(self, shape):
        img_height, img_width = shape
        colors = self.active_colors
        circle_diameter = max(1, int(img_height * 0.85 / float(len(colors))))
        spacing = max(
            1.0,
            (float(img_height) - circle_diameter * len(colors)) / float(len(colors) + 1),
        )
        x_center = int(img_width / 2.0)
        radius = max(1, int(circle_diameter / 2.0))
        return {
            color: (
                x_center,
                int(spacing * (index + 1) + circle_diameter * index + circle_diameter / 2.0),
                radius,
            )
            for index, color in enumerate(colors)
        }

    def _classify_by_bright_contour_position(self, contours, shape, slot_geometry):
        img_height, img_width = shape
        if img_height <= 0 or img_width <= 0:
            return None

        for contour in contours:
            bbox = cv2.boundingRect(contour)
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue

            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            if area <= 0.0 or perimeter <= 0.0:
                continue

            circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            height_norm = math.sqrt(area) / float(img_height)
            if height_norm < self.min_light_height_ratio or height_norm > self.max_light_height_ratio:
                continue

            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-6:
                continue
            x_center = float(moments["m10"] / moments["m00"])
            y_center = float(moments["m01"] / moments["m00"])
            if abs(x_center - img_width / 2.0) > img_width * self.max_center_x_offset_ratio:
                continue

            return min(
                self.active_colors,
                key=lambda color: abs(y_center - slot_geometry[color][1]),
            )
        return None

    def _classify_by_slot_brightness(self, hsv, slot_geometry, baseline_brightness):
        img_height, img_width = hsv.shape[:2]
        scores = {}
        for color, (x_center, y_center, radius) in slot_geometry.items():
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.circle(mask, (x_center, y_center), radius, 255, -1)
            scores[f"{color}_brightness"] = float(cv2.mean(hsv, mask)[2]) / 255.0

        threshold = max(
            baseline_brightness * self.brightness_threshold_scale,
            self.brightness_threshold_min,
        )
        scores["brightness_threshold"] = threshold
        best_color = max(self.active_colors, key=lambda color: scores[f"{color}_brightness"])
        best_score = scores[f"{best_color}_brightness"]
        other_scores = [
            scores[f"{color}_brightness"]
            for color in self.active_colors
            if color != best_color
        ]
        if best_score > threshold and all(best_score > score for score in other_scores):
            return best_color, scores
        return None, scores

    def _classify_by_hsv_backup(self, hsv, contours):
        scores = {}
        sat_threshold = self._adaptive_threshold(
            float(cv2.mean(hsv[:, :, 2])[0]) / 255.0,
            self.adaptive_sat_min,
            self.adaptive_sat_max,
        )
        val_threshold = self._adaptive_threshold(
            float(cv2.mean(hsv[:, :, 2])[0]) / 255.0,
            self.adaptive_val_min,
            self.adaptive_val_max,
        )
        red_low1 = np.array([0, sat_threshold, val_threshold], dtype=np.uint8)
        red_high1 = self.red_high1
        red_low2 = np.array([165, sat_threshold, val_threshold], dtype=np.uint8)
        red_high2 = self.red_high2
        yellow_low = self.yellow_low
        yellow_high = self.yellow_high
        green_low = np.array(
            [40, min(sat_threshold, 80.0), min(val_threshold, 80.0)],
            dtype=np.uint8,
        )
        green_high = self.green_high

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width <= 0 or height <= 0:
                continue

            circle_region = hsv[y:y + height, x:x + width]
            total_area = float(circle_region.shape[0] * circle_region.shape[1])
            if total_area <= 0.0:
                continue

            red_mask1 = cv2.inRange(circle_region, red_low1, red_high1)
            red_mask2 = cv2.inRange(circle_region, red_low2, red_high2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            yellow_mask = cv2.inRange(circle_region, yellow_low, yellow_high)
            green_mask = cv2.inRange(circle_region, green_low, green_high)

            ratios = {
                "red": float(cv2.countNonZero(red_mask)) / total_area,
                "yellow": float(cv2.countNonZero(yellow_mask)) / total_area,
                "green": float(cv2.countNonZero(green_mask)) / total_area,
            }
            for color, ratio in ratios.items():
                scores[f"{color}_ratio"] = max(scores.get(f"{color}_ratio", 0.0), ratio)

            best_color = max(self.active_colors, key=lambda color: ratios[color])
            if (
                ratios[best_color] > self.backup_color_ratio_min
                and all(
                    ratios[best_color] > ratios[color]
                    for color in self.active_colors
                    if color != best_color
                )
            ):
                return best_color, scores
        return None, scores

    @staticmethod
    def _adaptive_threshold(baseline_brightness, min_val, max_val):
        return min_val + (max_val - min_val) * (1.0 - baseline_brightness)

    def _known(self, color, reason, baseline_brightness=0.0):
        state = self.sign_for_color(color)
        return {
            "sign": state,
            "color": color,
            "state": state,
            "reason": reason,
            "baseline_brightness": round(float(baseline_brightness), 4),
            "scores": {},
        }

    def _unknown(self, reason, baseline_brightness=0.0, scores=None, crop_info=None):
        result = {
            "sign": self.UNKNOWN_SIGN,
            "color": self.UNKNOWN_COLOR,
            "state": self.UNKNOWN_SIGN,
            "reason": reason,
            "baseline_brightness": round(float(baseline_brightness), 4),
            "scores": scores or {},
        }
        if crop_info is not None:
            result["crop"] = crop_info
        return result
