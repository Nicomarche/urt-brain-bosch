"""
Motor de segmentacion de carriles con YOLOv8.

Carga un modelo de segmentacion entrenado para detectar carril izquierdo y
carril derecho, y calcula el mismo payload que esperan los endpoints
/ws/inference y /ws/steering del AI Server.
"""

import os
import threading
import time

import cv2
import numpy as np

import config


class YOLOLaneSegEngine:
    """Motor de inferencia basado en YOLOv8 segmentation."""

    DEFAULT_LEFT_ALIASES = (
        "left",
        "left_lane",
        "lane_left",
        "leftline",
        "line_left",
        "left-lane",
        "izquierda",
        "carril_izquierdo",
    )
    DEFAULT_RIGHT_ALIASES = (
        "right",
        "right_lane",
        "lane_right",
        "rightline",
        "line_right",
        "right-lane",
        "derecha",
        "carril_derecho",
    )
    DEFAULT_GENERIC_LANE_ALIASES = (
        "lane",
        "lanes",
        "lane_line",
        "lane_lines",
        "line",
        "linea",
        "carril",
    )

    def __init__(self, model_path=None, min_confidence=None, imgsz=None, device=None):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics no encontrado. Instala con: pip install ultralytics"
            )

        self.model_path = model_path or getattr(
            config, "LANE_SEG_MODEL_PATH", "models/lane_segmentation/best.pt"
        )
        self.min_confidence = float(
            min_confidence
            if min_confidence is not None
            else getattr(config, "LANE_SEG_MIN_CONFIDENCE", 0.35)
        )
        self.imgsz = int(
            imgsz if imgsz is not None else getattr(config, "LANE_SEG_IMGSZ", 320)
        )
        requested_device = (
            device if device is not None else getattr(config, "LANE_SEG_DEVICE", "auto")
        )
        self.device = self._resolve_device(requested_device)
        self.fallback_center_offset = float(
            getattr(config, "LANE_SEG_FALLBACK_CENTER_OFFSET", config.LANE_WIDTH_ESTIMATE)
        )

        left_aliases = getattr(config, "LANE_SEG_LEFT_CLASS_ALIASES", None)
        right_aliases = getattr(config, "LANE_SEG_RIGHT_CLASS_ALIASES", None)
        generic_aliases = getattr(config, "LANE_SEG_GENERIC_CLASS_ALIASES", None)
        self.left_aliases = self._normalize_aliases(left_aliases or self.DEFAULT_LEFT_ALIASES)
        self.right_aliases = self._normalize_aliases(
            right_aliases or self.DEFAULT_RIGHT_ALIASES
        )
        self.generic_lane_aliases = self._normalize_aliases(
            generic_aliases or self.DEFAULT_GENERIC_LANE_ALIASES
        )
        self.min_component_area_ratio = float(
            getattr(config, "LANE_SEG_MIN_COMPONENT_AREA_RATIO", 0.0008)
        )

        model_full_path = self._resolve_model_path(self.model_path)
        if not os.path.isfile(model_full_path):
            raise FileNotFoundError(
                f"[YOLOLaneSeg] Modelo no encontrado: {model_full_path}"
            )

        self.model_format = os.path.splitext(model_full_path)[1].lower()
        self.model = YOLO(model_full_path)
        if self.device != "cpu" and self.model_format in (".pt", ".pth"):
            self.model.to(self.device)

        self.labels = getattr(self.model, "names", {})
        self.frame_count = 0
        self.previous_steering = 0.0
        self.show_visualization = getattr(config, "SHOW_VISUALIZATION", False)
        self._vis_jpeg = None
        self._vis_lock = threading.Lock()

        print(f"[YOLOLaneSeg] Cargando modelo desde: {model_full_path}")
        print(f"[YOLOLaneSeg] Device: {self.device}")
        print(f"[YOLOLaneSeg] Confianza minima: {self.min_confidence}")
        print(f"[YOLOLaneSeg] Input size: {self.imgsz}x{self.imgsz}")
        if self.labels:
            print("[YOLOLaneSeg] Clases detectadas:")
            for idx, class_name in self.labels.items():
                print(f"  [{idx}] {class_name}")
        print(
            f"[YOLOLaneSeg] Alias izquierda={sorted(self.left_aliases)} | "
            f"derecha={sorted(self.right_aliases)}"
        )
        print(f"[YOLOLaneSeg] Alias generico={sorted(self.generic_lane_aliases)}")
        print(
            f"[YOLOLaneSeg] Visualization: "
            f"{'ON (/viz/lanes)' if self.show_visualization else 'OFF'}"
        )

        self._warmup()

    @staticmethod
    def _normalize(name):
        return str(name).strip().lower().replace(" ", "_").replace("-", "_")

    def _normalize_aliases(self, aliases):
        return {self._normalize(alias) for alias in aliases if alias}

    def _resolve_model_path(self, model_path):
        if os.path.isabs(model_path):
            return model_path
        return os.path.join(os.path.dirname(__file__), model_path)

    def _resolve_device(self, requested_device):
        if requested_device != "auto":
            return requested_device
        try:
            import torch
        except ImportError:
            return "cpu"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _warmup(self):
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        start_time = time.time()
        self.model.predict(
            dummy,
            conf=self.min_confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        warmup_ms = (time.time() - start_time) * 1000
        print(f"[YOLOLaneSeg] Warm-up completado en {warmup_ms:.0f}ms")

    def _class_to_role(self, class_name):
        normalized = self._normalize(class_name)
        if normalized in self.left_aliases:
            return "left"
        if normalized in self.right_aliases:
            return "right"
        if normalized in self.generic_lane_aliases:
            return "generic"
        return None

    def _segments_to_mask(self, segments, shape):
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        if segments is None:
            return mask

        if isinstance(segments, np.ndarray):
            segment_list = [segments]
        else:
            segment_list = list(segments)

        for segment in segment_list:
            pts = np.asarray(segment, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 3:
                continue
            pts = np.round(pts).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [pts], 255)

        return mask

    def _mask_x_at_y(self, mask, y_target, search_radius=12):
        if mask is None or mask.size == 0:
            return None

        height, _ = mask.shape[:2]
        y_target = int(np.clip(y_target, 0, height - 1))

        for delta in range(search_radius + 1):
            rows = [y_target] if delta == 0 else [y_target - delta, y_target + delta]
            for row in rows:
                if row < 0 or row >= height:
                    continue
                xs = np.where(mask[row] > 0)[0]
                if xs.size > 0:
                    return float(xs.mean())

        return None

    def _extract_lane_points(self, mask, start_ratio=0.35, samples=18):
        if mask is None or mask.size == 0:
            return []

        height, _ = mask.shape[:2]
        y_positions = np.linspace(int(height * start_ratio), height - 1, samples).astype(int)

        points = []
        for y in y_positions:
            x = self._mask_x_at_y(mask, y, search_radius=4)
            if x is not None:
                points.append((int(round(x)), int(y)))

        return points

    def _combine_masks(self, masks):
        valid_masks = [mask for mask in masks if mask is not None and np.any(mask)]
        if not valid_masks:
            return None

        combined = np.zeros_like(valid_masks[0], dtype=np.uint8)
        for mask in valid_masks:
            combined = cv2.bitwise_or(combined, mask)
        return combined

    def _build_detection_payload(self, side, class_name, confidence, box, frame_shape):
        height, width = frame_shape
        x1, y1, x2, y2 = box
        return {
            "class": side,
            "source_class": class_name,
            "confidence": round(float(confidence), 4),
            "box": [
                round(float(y1) / max(height, 1), 4),
                round(float(x1) / max(width, 1), 4),
                round(float(y2) / max(height, 1), 4),
                round(float(x2) / max(width, 1), 4),
            ],
        }

    def _mask_bbox(self, mask):
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            return None
        return (
            float(xs.min()),
            float(ys.min()),
            float(xs.max()),
            float(ys.max()),
        )

    def _split_mask_components(self, mask):
        if mask is None or not np.any(mask):
            return []

        min_area_px = int(mask.shape[0] * mask.shape[1] * self.min_component_area_ratio)
        min_area_px = max(min_area_px, 20)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        components = []
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < min_area_px:
                continue

            component_mask = np.where(labels == label_idx, 255, 0).astype(np.uint8)
            bbox = self._mask_bbox(component_mask)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            components.append(
                {
                    "mask": component_mask,
                    "box": bbox,
                    "x_center": (x1 + x2) / 2.0,
                    "area": area,
                }
            )

        return sorted(components, key=lambda item: item["x_center"])

    def _split_lane_candidates(self, result, frame_shape):
        height, width = frame_shape
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return {"left": [], "right": []}, []

        masks_xy = []
        if result.masks is not None and getattr(result.masks, "xy", None) is not None:
            masks_xy = result.masks.xy

        candidate_entries = []
        for idx in range(len(boxes)):
            conf = float(boxes.conf[idx])
            cls_idx = int(boxes.cls[idx])
            class_name = str(self.labels.get(cls_idx, f"class_{cls_idx}"))
            role = self._class_to_role(class_name)
            if role is None:
                continue

            box = boxes.xyxy[idx].tolist()
            x1, y1, x2, y2 = [float(v) for v in box]
            x_center = (x1 + x2) / 2.0
            segments = masks_xy[idx] if idx < len(masks_xy) else None
            mask = self._segments_to_mask(segments, (height, width))

            if role == "generic":
                components = self._split_mask_components(mask)
                if len(components) >= 2:
                    left_component = components[0]
                    right_component = components[-1]
                    candidate_entries.append(
                        {
                            "class_name": class_name,
                            "role": "left",
                            "confidence": conf,
                            "box": left_component["box"],
                            "x_center": left_component["x_center"],
                            "mask": left_component["mask"],
                        }
                    )
                    candidate_entries.append(
                        {
                            "class_name": class_name,
                            "role": "right",
                            "confidence": conf,
                            "box": right_component["box"],
                            "x_center": right_component["x_center"],
                            "mask": right_component["mask"],
                        }
                    )
                    continue
                elif len(components) == 1:
                    mask = components[0]["mask"]
                    x1, y1, x2, y2 = components[0]["box"]
                    x_center = components[0]["x_center"]

            candidate_entries.append(
                {
                    "class_name": class_name,
                    "role": role,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                    "x_center": x_center,
                    "mask": mask,
                }
            )

        side_candidates = {"left": [], "right": []}
        unknown = []
        frame_center = width / 2.0

        for entry in candidate_entries:
            if entry["role"] == "left":
                side_candidates["left"].append(entry)
            elif entry["role"] == "right":
                side_candidates["right"].append(entry)
            elif entry["role"] == "generic":
                unknown.append(entry)

        if not side_candidates["left"] and not side_candidates["right"] and unknown:
            ordered = sorted(unknown, key=lambda item: item["x_center"])
            if len(ordered) >= 2:
                side_candidates["left"].append(ordered[0])
                side_candidates["right"].append(ordered[-1])
            else:
                side = "left" if ordered[0]["x_center"] < frame_center else "right"
                side_candidates[side].append(ordered[0])
        else:
            for entry in unknown:
                if entry["x_center"] < frame_center and not side_candidates["left"]:
                    side_candidates["left"].append(entry)
                elif entry["x_center"] >= frame_center and not side_candidates["right"]:
                    side_candidates["right"].append(entry)

        detections = []
        for side in ("left", "right"):
            for entry in side_candidates[side]:
                detections.append(
                    self._build_detection_payload(
                        side,
                        entry["class_name"],
                        entry["confidence"],
                        entry["box"],
                        frame_shape,
                    )
                )

        return side_candidates, detections

    def _build_lane_geometry(self, side_candidates, frame_shape):
        side_masks = {"left": None, "right": None}
        lane_points = []

        for side in ("left", "right"):
            merged_mask = self._combine_masks([entry["mask"] for entry in side_candidates[side]])
            side_masks[side] = merged_mask
            points = self._extract_lane_points(merged_mask)
            if points:
                lane_points.append(points)

        return side_masks, lane_points

    def _compute_steering(self, side_masks, frame_shape):
        height, width = frame_shape
        frame_center = width / 2.0
        y_target = int(height * (1.0 - config.LOOKAHEAD_RATIO))

        left_x = self._mask_x_at_y(side_masks.get("left"), y_target)
        right_x = self._mask_x_at_y(side_masks.get("right"), y_target)

        if left_x is None and right_x is None:
            return {
                "steering_angle": None,
                "lane_center_x": None,
                "confidence": 0.0,
                "error_normalized": 0.0,
            }

        lane_width_px = width * self.fallback_center_offset
        if left_x is not None and right_x is not None:
            lane_center = (left_x + right_x) / 2.0
            confidence = 1.0
        elif left_x is not None:
            lane_center = left_x + (lane_width_px / 2.0)
            confidence = 0.6
        else:
            lane_center = right_x - (lane_width_px / 2.0)
            confidence = 0.6

        error = lane_center - frame_center
        error_normalized = error / max(frame_center, 1.0)

        kp = 25.0
        kd = 3.0
        derivative = error_normalized - (
            self.previous_steering / kp if self.previous_steering else 0.0
        )
        steering_raw = kp * error_normalized + kd * derivative

        alpha = config.STEERING_SMOOTHING
        steering = alpha * steering_raw + (1.0 - alpha) * self.previous_steering
        steering = max(-25.0, min(25.0, steering))
        self.previous_steering = steering

        return {
            "steering_angle": round(steering, 2),
            "lane_center_x": round(lane_center, 1),
            "confidence": round(confidence, 2),
            "error_normalized": round(error_normalized, 4),
        }

    def _encode_mask(self, mask):
        if mask is None or not np.any(mask):
            return b""
        ok, encoded = cv2.imencode(".png", mask)
        return encoded.tobytes() if ok else b""

    def get_vis_jpeg(self):
        with self._vis_lock:
            return self._vis_jpeg

    def _visualize(self, frame, side_masks, lane_points, steering, infer_ms):
        vis = frame.copy()
        height, width = vis.shape[:2]

        overlay = np.zeros_like(vis)
        left_mask = side_masks.get("left")
        right_mask = side_masks.get("right")

        if left_mask is not None and np.any(left_mask):
            overlay[left_mask > 0] = (255, 120, 0)
        if right_mask is not None and np.any(right_mask):
            overlay[right_mask > 0] = (0, 200, 255)
        if np.any(overlay):
            vis = cv2.addWeighted(vis, 1.0, overlay, 0.35, 0)

        line_colors = [(255, 170, 0), (0, 220, 255)]
        for idx, line in enumerate(lane_points):
            if len(line) < 2:
                continue
            pts = np.array(line, dtype=np.int32).reshape((-1, 1, 2))
            color = line_colors[idx % len(line_colors)]
            cv2.polylines(vis, [pts], False, color, 3, cv2.LINE_AA)
            for point in line:
                cv2.circle(vis, point, 3, color, -1, cv2.LINE_AA)

        frame_center = width // 2
        cv2.line(vis, (frame_center, height), (frame_center, int(height * 0.45)), (200, 200, 200), 1)

        lane_center_x = steering.get("lane_center_x")
        if lane_center_x is not None:
            lane_center_x = int(round(lane_center_x))
            lookahead_y = int(height * (1.0 - config.LOOKAHEAD_RATIO))
            cv2.circle(vis, (lane_center_x, lookahead_y), 8, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.line(vis, (frame_center, lookahead_y), (lane_center_x, lookahead_y), (0, 255, 0), 2, cv2.LINE_AA)

        bar_h = 42
        cv2.rectangle(vis, (0, 0), (width, bar_h), (0, 0, 0), -1)
        steer_angle = steering.get("steering_angle")
        steer_text = "--" if steer_angle is None else f"{steer_angle:.1f} deg"
        conf_text = f"{steering.get('confidence', 0.0):.2f}"
        status = (
            f"YOLO Lanes | steer:{steer_text} | conf:{conf_text} | "
            f"{infer_ms:.0f}ms | frame {self.frame_count}"
        )
        cv2.putText(
            vis,
            status,
            (8, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        ok, jpeg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with self._vis_lock:
                self._vis_jpeg = jpeg.tobytes()

    def _run_model(self, frame):
        results = self.model.predict(
            frame,
            conf=self.min_confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return results[0] if results else None

    def _infer_impl(self, frame, include_masks):
        start_time = time.time()
        frame_shape = frame.shape[:2]

        result = self._run_model(frame)
        if result is None:
            self.frame_count += 1
            inference_ms = (time.time() - start_time) * 1000
            empty = {
                "steering": {
                    "steering_angle": None,
                    "lane_center_x": None,
                    "confidence": 0.0,
                    "error_normalized": 0.0,
                },
                "lane_points": [],
                "detections": [],
                "lane_mask": b"",
                "road_mask": b"",
                "inference_time_ms": round(inference_ms, 1),
                "frame_id": self.frame_count,
                "input_size": f"{self.imgsz}x{self.imgsz}",
            }
            return empty

        side_candidates, detections = self._split_lane_candidates(result, frame_shape)
        side_masks, lane_points = self._build_lane_geometry(side_candidates, frame_shape)
        steering = self._compute_steering(side_masks, frame_shape)
        combined_lane_mask = self._combine_masks(
            [side_masks.get("left"), side_masks.get("right")]
        )

        inference_ms = (time.time() - start_time) * 1000
        self.frame_count += 1

        payload = {
            "steering": steering,
            "lane_points": lane_points,
            "detections": detections,
            "lane_mask": self._encode_mask(combined_lane_mask) if include_masks else b"",
            "road_mask": b"",
            "inference_time_ms": round(inference_ms, 1),
            "frame_id": self.frame_count,
            "input_size": f"{self.imgsz}x{self.imgsz}",
        }
        if self.show_visualization:
            self._visualize(frame, side_masks, lane_points, steering, inference_ms)
        return payload

    def infer(self, frame):
        """Inferencia completa compatible con /ws/inference."""
        return self._infer_impl(frame, include_masks=True)

    def infer_steering_only(self, frame):
        """Inferencia ligera compatible con /ws/steering."""
        payload = self._infer_impl(frame, include_masks=False)
        return {
            "steering": payload["steering"],
            "inference_time_ms": payload["inference_time_ms"],
            "frame_id": payload["frame_id"],
            "input_size": payload["input_size"],
        }

    def shutdown(self):
        """No hay recursos externos persistentes que liberar."""

    def get_status(self):
        """Estado del motor para /status."""
        labels = [str(name) for _, name in sorted(self.labels.items())]
        return {
            "engine_type": "yolo_lane_seg",
            "model_path": self.model_path,
            "model_format": self.model_format,
            "device": self.device,
            "input_size": f"{self.imgsz}x{self.imgsz}",
            "min_confidence": self.min_confidence,
            "frames_processed": self.frame_count,
            "classes": labels,
            "left_aliases": sorted(self.left_aliases),
            "right_aliases": sorted(self.right_aliases),
            "generic_lane_aliases": sorted(self.generic_lane_aliases),
            "show_visualization": self.show_visualization,
            "supports_browser_viz": True,
        }
