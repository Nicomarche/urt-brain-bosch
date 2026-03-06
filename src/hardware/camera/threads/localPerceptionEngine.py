import os
import re
import threading
import time
import math

import cv2
import numpy as np

import config


class LocalPerceptionEngine:
    """Local YOLO-based perception engine for lane geometry and sign detections."""

    def __init__(self, model_path=None, min_confidence=None, imgsz=None, device=None):
        self.model_path = model_path or getattr(
            config, "LOCAL_AI_MODEL_PATH", "models/lane_segmentation/Best416px.engine"
        )
        self.min_confidence = float(
            min_confidence
            if min_confidence is not None
            else getattr(config, "LOCAL_AI_MIN_CONFIDENCE", 0.35)
        )
        self.imgsz = int(
            imgsz if imgsz is not None else getattr(config, "LOCAL_AI_IMGSZ", 416)
        )
        self.requested_device = (
            device if device is not None else getattr(config, "LOCAL_AI_DEVICE", "auto")
        )
        self.device = "cpu"

        self.left_aliases = self._normalize_aliases(
            getattr(config, "LOCAL_AI_LEFT_CLASS_ALIASES", [])
        )
        self.right_aliases = self._normalize_aliases(
            getattr(config, "LOCAL_AI_RIGHT_CLASS_ALIASES", [])
        )
        self.generic_lane_aliases = self._normalize_aliases(
            getattr(config, "LOCAL_AI_GENERIC_LANE_ALIASES", [])
        )
        self.sign_class_map = {
            self._normalize(key): value
            for key, value in getattr(config, "LOCAL_AI_SIGN_CLASS_MAP", {}).items()
        }
        self.nms_iou = max(0.1, min(0.9, float(getattr(config, "LOCAL_AI_NMS_IOU", 0.45))))
        self.max_detections = max(1, int(getattr(config, "LOCAL_AI_MAX_DETECTIONS", 80)))
        self.sign_min_confidence = max(
            self.min_confidence,
            min(1.0, float(getattr(config, "LOCAL_AI_SIGN_MIN_CONFIDENCE", 0.55))),
        )
        self.sign_max_detections = max(1, int(getattr(config, "LOCAL_AI_SIGN_MAX_DETECTIONS", 20)))
        self.sign_debug_max_detections = max(
            1, int(getattr(config, "LOCAL_AI_SIGN_DEBUG_MAX_DETECTIONS", 8))
        )
        self.drop_unknown_classes = bool(getattr(config, "LOCAL_AI_DROP_UNKNOWN_CLASSES", True))
        self.class_id_name_map = self._parse_class_id_name_map(
            getattr(config, "LOCAL_AI_CLASS_ID_NAME_MAP", {})
        )

        self.frame_count = 0
        self.labels = {}
        self.model = None
        self.model_ready = False
        self.init_error = None
        self._vis_lock = threading.Lock()
        self._last_debug = {}
        self._warned_fallback_labels = False

        try:
            from ultralytics import YOLO
        except ImportError:
            self.init_error = "ultralytics no instalado"
            print(
                f"\033[1;97m[ Local AI ] :\033[0m \033[1;93mWARNING\033[0m - "
                f"{self.init_error}"
            )
            return

        model_full_path = self._resolve_model_path(self.model_path)
        if not os.path.isfile(model_full_path):
            self.init_error = f"modelo no encontrado: {model_full_path}"
            print(
                f"\033[1;97m[ Local AI ] :\033[0m \033[1;93mWARNING\033[0m - "
                f"{self.init_error}"
            )
            return

        model_ext = os.path.splitext(model_full_path)[1].lower()
        self.device = self._resolve_device(self.requested_device, model_ext)
        self._sync_engine_imgsz(model_full_path, model_ext)

        try:
            # Engines TensorRT a veces no conservan metadata de task.
            # Forzamos segment para evitar fallback a detect (sin mascaras).
            self.model = YOLO(model_full_path, task="segment")
            if getattr(self.model, "task", None) != "segment":
                self.model.task = "segment"
            if self.device != "cpu" and model_ext in (".pt", ".pth"):
                self.model.to(self.device)
            self.labels = getattr(self.model, "names", {})
            self.model_ready = True
            print(f"\033[1;97m[ Local AI ] :\033[0m \033[1;92mINFO\033[0m - Modelo: {model_full_path}")
            print(f"\033[1;97m[ Local AI ] :\033[0m \033[1;92mINFO\033[0m - Device: {self.device}")
            print(f"\033[1;97m[ Local AI ] :\033[0m \033[1;92mINFO\033[0m - Input: {self.imgsz}x{self.imgsz} conf={self.min_confidence}")
            print(
                f"\033[1;97m[ Local AI ] :\033[0m \033[1;92mINFO\033[0m - "
                f"NMS iou={self.nms_iou} max_det={self.max_detections} sign_conf={self.sign_min_confidence}"
            )
            self._warn_if_fallback_labels()
            self._warmup()
        except Exception as e:
            self.model_ready = False
            self.init_error = str(e)
            self.model = None
            print(
                f"\033[1;97m[ Local AI ] :\033[0m \033[1;91mERROR\033[0m - "
                f"Fallo cargando modelo: {e}"
            )

    @staticmethod
    def _normalize(name):
        return str(name).strip().lower().replace(" ", "_").replace("-", "_")

    def _normalize_aliases(self, aliases):
        return {self._normalize(alias) for alias in aliases if alias}

    def _parse_class_id_name_map(self, raw_mapping):
        parsed = {}
        if not isinstance(raw_mapping, dict):
            return parsed
        for key, value in raw_mapping.items():
            try:
                cls_idx = int(key)
            except (TypeError, ValueError):
                continue
            if value is None:
                continue
            parsed[cls_idx] = str(value)
        return parsed

    @staticmethod
    def _is_fallback_class_name(class_name):
        return re.match(r"^class_?\d+$", str(class_name).strip().lower()) is not None

    def _label_from_index(self, cls_idx):
        if isinstance(self.labels, dict):
            return self.labels.get(cls_idx)
        if isinstance(self.labels, (list, tuple)) and 0 <= cls_idx < len(self.labels):
            return self.labels[cls_idx]
        return None

    def _resolve_class_name(self, cls_idx):
        class_name = self._label_from_index(cls_idx)
        if class_name is None:
            class_name = f"class_{cls_idx}"
        class_name = str(class_name)

        if self._is_fallback_class_name(class_name):
            mapped_name = self.class_id_name_map.get(cls_idx)
            if mapped_name:
                return mapped_name
            if self.drop_unknown_classes:
                return None
        return class_name

    def _warn_if_fallback_labels(self):
        if self._warned_fallback_labels:
            return

        samples = []
        if isinstance(self.labels, dict):
            for key in sorted(self.labels.keys())[:6]:
                samples.append(str(self.labels[key]))
        elif isinstance(self.labels, (list, tuple)):
            samples = [str(item) for item in list(self.labels)[:6]]

        if samples and all(self._is_fallback_class_name(name) for name in samples):
            self._warned_fallback_labels = True
            print(
                f"\033[1;97m[ Local AI ] :\033[0m \033[1;93mWARNING\033[0m - "
                "El engine reporta etiquetas classXX. Recomendado: exportar con "
                "Ultralytics (YOLO.export format=engine, nms=True) en la misma Jetson."
            )

    def _resolve_model_path(self, model_path):
        if os.path.isabs(model_path):
            return model_path

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        primary = os.path.join(repo_root, model_path)
        if os.path.isfile(primary):
            return primary

        legacy = os.path.join(repo_root, "aiserver", "models", "lane_segmentation", os.path.basename(model_path))
        if os.path.isfile(legacy):
            return legacy

        return primary

    def _resolve_device(self, requested_device, model_ext=None):
        normalized_device = str(requested_device).strip().lower()
        if model_ext == ".engine":
            if normalized_device in ("auto", "cpu", "mps"):
                return "cuda"
            return requested_device
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

    def _infer_engine_imgsz(self, model_full_path):
        match = re.search(r"(\d+)px\.engine$", os.path.basename(model_full_path), re.IGNORECASE)
        if match is None:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    def _sync_engine_imgsz(self, model_full_path, model_ext):
        if model_ext != ".engine":
            return

        engine_imgsz = self._infer_engine_imgsz(model_full_path)
        if engine_imgsz is None or self.imgsz == engine_imgsz:
            return

        print(
            f"\033[1;97m[ Local AI ] :\033[0m \033[1;93mWARNING\033[0m - "
            f"imgsz={self.imgsz} no coincide con el engine TensorRT; usando {engine_imgsz}"
        )
        self.imgsz = engine_imgsz

    def _warmup(self):
        if not self.model_ready or self.model is None:
            return
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            start_time = time.time()
            self.model.predict(
                dummy,
                conf=self.min_confidence,
                iou=self.nms_iou,
                imgsz=self.imgsz,
                max_det=self.max_detections,
                device=self.device,
                verbose=False,
            )
            warmup_ms = (time.time() - start_time) * 1000
            print(f"\033[1;97m[ Local AI ] :\033[0m \033[1;92mINFO\033[0m - Warm-up {warmup_ms:.0f}ms")
        except Exception as e:
            print(
                f"\033[1;97m[ Local AI ] :\033[0m \033[1;93mWARNING\033[0m - "
                f"Warm-up fallo: {e}"
            )

    def _class_to_role(self, class_name):
        normalized = self._normalize(class_name)
        if normalized in self.left_aliases:
            return "left"
        if normalized in self.right_aliases:
            return "right"
        if normalized in self.generic_lane_aliases:
            return "generic"
        return None

    def _normalize_sign_name(self, class_name):
        normalized = self._normalize(class_name)
        return self.sign_class_map.get(normalized, normalized)

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

    def _combine_masks(self, masks):
        valid_masks = [mask for mask in masks if mask is not None and np.any(mask)]
        if not valid_masks:
            return None

        combined = np.zeros_like(valid_masks[0], dtype=np.uint8)
        for mask in valid_masks:
            combined = cv2.bitwise_or(combined, mask)
        return combined

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

        min_area_px = max(int(mask.shape[0] * mask.shape[1] * 0.0008), 20)
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
                }
            )

        return sorted(components, key=lambda item: item["x_center"])

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

    def _extract_lane_line(self, mask, start_ratio=0.35):
        """Fit a representative line directly from the segmented lane mask."""
        if mask is None or mask.size == 0:
            return None

        height, width = mask.shape[:2]
        roi_top = int(height * start_ratio)

        ys, xs = np.where(mask > 0)
        if xs.size < 2 or ys.size < 2:
            return None

        roi_keep = ys >= roi_top
        if np.count_nonzero(roi_keep) >= 2:
            ys = ys[roi_keep]
            xs = xs[roi_keep]

        if xs.size < 2 or ys.size < 2:
            return None

        ys = ys.astype(np.float32)
        xs = xs.astype(np.float32)

        if np.ptp(ys) < 1.0:
            order = np.argsort(xs)
            p1_idx = int(order[0])
            p2_idx = int(order[-1])
            return [[
                int(round(np.clip(xs[p1_idx], 0, width - 1))),
                int(round(np.clip(ys[p1_idx], roi_top, height - 1))),
                int(round(np.clip(xs[p2_idx], 0, width - 1))),
                int(round(np.clip(ys[p2_idx], roi_top, height - 1))),
            ]]

        fit = np.polyfit(ys, xs, 1)
        y_bottom = float(min(np.max(ys), height - 1))
        y_top = float(max(np.min(ys), roi_top))
        if y_bottom - y_top < 1.0:
            y_top = max(float(roi_top), y_bottom - 1.0)

        x_bottom = float(np.polyval(fit, y_bottom))
        x_top = float(np.polyval(fit, y_top))

        return [[
            int(round(np.clip(x_bottom, 0, width - 1))),
            int(round(np.clip(y_bottom, 0, height - 1))),
            int(round(np.clip(x_top, 0, width - 1))),
            int(round(np.clip(y_top, 0, height - 1))),
        ]]

    def _lane_mask_overlap_ratio(self, mask_a, mask_b):
        """Return how much two lane masks overlap relative to the smaller mask."""
        if mask_a is None or mask_b is None:
            return 0.0
        if mask_a.shape[:2] != mask_b.shape[:2]:
            return 0.0

        a = mask_a > 0
        b = mask_b > 0
        a_count = int(np.count_nonzero(a))
        b_count = int(np.count_nonzero(b))
        if a_count == 0 or b_count == 0:
            return 0.0

        overlap = int(np.count_nonzero(a & b))
        if overlap == 0:
            return 0.0
        return float(overlap) / float(min(a_count, b_count))

    def _lane_mask_gap_px(self, left_mask, right_mask):
        """Estimate average horizontal separation between two lane masks."""
        if left_mask is None or right_mask is None:
            return None
        if left_mask.shape[:2] != right_mask.shape[:2]:
            return None

        height = left_mask.shape[0]
        sample_rows = [
            max(0, min(height - 1, int(round(height * ratio))))
            for ratio in (0.95, 0.75, 0.6)
        ]
        gaps = []
        for row in sample_rows:
            left_x = self._mask_x_at_y(left_mask, row, search_radius=6)
            right_x = self._mask_x_at_y(right_mask, row, search_radius=6)
            if left_x is not None and right_x is not None:
                gaps.append(abs(float(right_x) - float(left_x)))

        if not gaps:
            return None
        return sum(gaps) / len(gaps)

    @staticmethod
    def _coerce_lane_line_values(line_value):
        """Normalize a lane line payload into [x1, y1, x2, y2] floats."""
        if not isinstance(line_value, (list, tuple, np.ndarray)):
            return None

        values = line_value
        if len(values) == 1 and isinstance(values[0], (list, tuple, np.ndarray)):
            values = values[0]
        if len(values) < 4:
            return None

        try:
            return [float(values[idx]) for idx in range(4)]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _line_x_at_y(line_values, y_target):
        """Project a normalized [x1,y1,x2,y2] segment to the requested y row."""
        if line_values is None:
            return None
        x1, y1, x2, y2 = line_values
        dy = y2 - y1
        if abs(dy) < 1e-6:
            return (x1 + x2) * 0.5
        t = (float(y_target) - y1) / dy
        return x1 + t * (x2 - x1)

    def _lane_line_gap_px(self, left_line, right_line, img_h):
        """Estimate left/right separation from explicit line segments when masks are sparse."""
        left_values = self._coerce_lane_line_values(left_line)
        right_values = self._coerce_lane_line_values(right_line)
        if left_values is None or right_values is None:
            return None

        sample_rows = [
            max(0, min(img_h - 1, int(round(img_h * ratio))))
            for ratio in (0.95, 0.75, 0.6)
        ]
        gaps = []
        for row in sample_rows:
            left_x = self._line_x_at_y(left_values, row)
            right_x = self._line_x_at_y(right_values, row)
            if left_x is not None and right_x is not None:
                gaps.append(abs(float(right_x) - float(left_x)))

        if not gaps:
            return None
        return sum(gaps) / len(gaps)

    def _collapse_duplicate_lane_sides(
        self,
        side_masks,
        lane_points,
        lane_side_points,
        lane_side_lines,
        lane_side_sources,
    ):
        """Collapse overlapping left/right detections that are really one physical line."""
        left_mask = side_masks.get("left")
        right_mask = side_masks.get("right")
        if left_mask is None or right_mask is None:
            return side_masks, lane_points, lane_side_points, lane_side_lines, lane_side_sources
        if not np.any(left_mask) or not np.any(right_mask):
            return side_masks, lane_points, lane_side_points, lane_side_lines, lane_side_sources

        overlap_ratio = self._lane_mask_overlap_ratio(left_mask, right_mask)
        mean_gap_px = self._lane_mask_gap_px(left_mask, right_mask)
        line_gap_px = self._lane_line_gap_px(
            lane_side_lines.get("left"),
            lane_side_lines.get("right"),
            left_mask.shape[0],
        )
        duplicate_gap_limit = max(6.0, float(left_mask.shape[1]) * 0.04)
        duplicate_line_gap_limit = max(4.0, float(left_mask.shape[1]) * 0.02)
        is_duplicate = overlap_ratio >= 0.35
        if mean_gap_px is not None:
            is_duplicate = is_duplicate or mean_gap_px <= duplicate_gap_limit
        if line_gap_px is not None:
            is_duplicate = is_duplicate or line_gap_px <= duplicate_line_gap_limit
        if not is_duplicate:
            return side_masks, lane_points, lane_side_points, lane_side_lines, lane_side_sources

        merged_mask = self._combine_masks([left_mask, right_mask])
        if merged_mask is None or not np.any(merged_mask):
            return side_masks, lane_points, lane_side_points, lane_side_lines, lane_side_sources

        left_source = str(lane_side_sources.get("left", "none") or "none")
        right_source = str(lane_side_sources.get("right", "none") or "none")
        keep_side = None
        keep_source = "guessed_single"
        if left_source != "guessed_single" and right_source == "guessed_single":
            keep_side = "left"
            keep_source = left_source
        elif right_source != "guessed_single" and left_source == "guessed_single":
            keep_side = "right"
            keep_source = right_source

        if keep_side is None:
            bbox = self._mask_bbox(merged_mask)
            if bbox is not None:
                x1, _, x2, _ = bbox
                merged_center_x = (x1 + x2) / 2.0
            else:
                merged_center_x = float(merged_mask.shape[1]) / 2.0
            keep_side = "left" if merged_center_x < (merged_mask.shape[1] / 2.0) else "right"

        drop_side = "right" if keep_side == "left" else "left"
        merged_points = self._extract_lane_points(merged_mask)
        merged_line = self._extract_lane_line(merged_mask)

        side_masks[keep_side] = merged_mask
        side_masks[drop_side] = None
        lane_side_points[keep_side] = [[int(point[0]), int(point[1])] for point in merged_points]
        lane_side_points[drop_side] = []
        lane_side_lines[keep_side] = merged_line[0] if merged_line is not None else []
        lane_side_lines[drop_side] = []
        lane_side_sources[keep_side] = keep_source
        lane_side_sources[drop_side] = "none"

        lane_points = [merged_points] if merged_points else []
        return side_masks, lane_points, lane_side_points, lane_side_lines, lane_side_sources

    def _build_box_payload(self, x1, y1, x2, y2, frame_shape):
        height, width = frame_shape
        return [
            round(float(y1) / max(height, 1), 4),
            round(float(x1) / max(width, 1), 4),
            round(float(y2) / max(height, 1), 4),
            round(float(x2) / max(width, 1), 4),
        ]

    def _apply_classwise_nms(self, boxes):
        if boxes is None or len(boxes) == 0:
            return []

        entries = []
        for idx in range(len(boxes)):
            conf = float(boxes.conf[idx])
            if conf < self.min_confidence:
                continue
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[idx].tolist()]
            entries.append(
                {
                    "idx": idx,
                    "cls": int(boxes.cls[idx]),
                    "conf": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

        if not entries:
            return []

        by_class = {}
        for entry in entries:
            by_class.setdefault(entry["cls"], []).append(entry)

        keep_indices = []
        conf_by_index = {entry["idx"]: entry["conf"] for entry in entries}

        for class_entries in by_class.values():
            class_entries.sort(key=lambda item: item["conf"], reverse=True)
            bboxes = []
            scores = []
            for item in class_entries:
                width = max(1.0, item["x2"] - item["x1"])
                height = max(1.0, item["y2"] - item["y1"])
                bboxes.append(
                    [
                        int(round(item["x1"])),
                        int(round(item["y1"])),
                        int(round(width)),
                        int(round(height)),
                    ]
                )
                scores.append(float(item["conf"]))

            keep_local = []
            try:
                keep_local = cv2.dnn.NMSBoxes(
                    bboxes,
                    scores,
                    score_threshold=self.min_confidence,
                    nms_threshold=self.nms_iou,
                )
            except Exception:
                keep_local = []

            if keep_local is None or len(keep_local) == 0:
                keep_indices.extend([item["idx"] for item in class_entries])
                continue

            flat_local = np.array(keep_local).reshape(-1).tolist()
            for local_idx in flat_local:
                local_idx = int(local_idx)
                if 0 <= local_idx < len(class_entries):
                    keep_indices.append(class_entries[local_idx]["idx"])

        keep_indices = sorted(
            set(keep_indices),
            key=lambda idx: conf_by_index.get(idx, 0.0),
            reverse=True,
        )
        return keep_indices[: self.max_detections]

    def _split_candidates(self, result, frame_shape):
        height, width = frame_shape
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return {"left": [], "right": []}, []

        masks_xy = []
        if result.masks is not None and getattr(result.masks, "xy", None) is not None:
            masks_xy = result.masks.xy

        side_candidates = {"left": [], "right": []}
        sign_detections = []
        frame_center = width / 2.0
        keep_indices = self._apply_classwise_nms(boxes)

        for idx in keep_indices:
            conf = float(boxes.conf[idx])
            cls_idx = int(boxes.cls[idx])
            class_name = self._resolve_class_name(cls_idx)
            if class_name is None:
                continue
            role = self._class_to_role(class_name)
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[idx].tolist()]
            x_center = (x1 + x2) / 2.0
            segments = masks_xy[idx] if idx < len(masks_xy) else None
            mask = self._segments_to_mask(segments, (height, width))
            has_mask = bool(mask is not None and np.any(mask))

            # Cuando el engine viene sin nombres y devuelve classXX,
            # si trae mascara lo tratamos como carril generico.
            if role is None and self._is_fallback_class_name(class_name) and has_mask:
                role = "generic"

            if role is None:
                if conf < self.sign_min_confidence:
                    continue
                sign_detections.append(
                    {
                        "class": self._normalize_sign_name(class_name),
                        "source_class": class_name,
                        "confidence": round(conf, 4),
                        "box": self._build_box_payload(x1, y1, x2, y2, frame_shape),
                    }
                )
                continue

            if role == "generic":
                components = self._split_mask_components(mask)
                if len(components) >= 2:
                    left_component = components[0]
                    right_component = components[-1]
                    side_candidates["left"].append(
                        {
                            "mask": left_component["mask"],
                            "x_center": left_component["x_center"],
                            "confidence": conf,
                            "side_source": "split_component",
                        }
                    )
                    side_candidates["right"].append(
                        {
                            "mask": right_component["mask"],
                            "x_center": right_component["x_center"],
                            "confidence": conf,
                            "side_source": "split_component",
                        }
                    )
                    continue
                if len(components) == 1:
                    mask = components[0]["mask"]
                    x_center = components[0]["x_center"]
                    role = "left" if x_center < frame_center else "right"
                    side_source = "guessed_single"
                else:
                    role = "left" if x_center < frame_center else "right"
                    side_source = "guessed_single"
            else:
                side_source = "explicit_class"

            side_candidates[role].append(
                {
                    "mask": mask,
                    "x_center": x_center,
                    "confidence": conf,
                    "side_source": side_source,
                }
            )

        for side in ("left", "right"):
            side_candidates[side].sort(
                key=lambda item: (abs(item["x_center"] - frame_center), -item["confidence"])
            )
            if side_candidates[side]:
                side_candidates[side] = [side_candidates[side][0]]

        sign_detections.sort(key=lambda item: item["confidence"], reverse=True)
        sign_detections = sign_detections[: self.sign_max_detections]
        return side_candidates, sign_detections

    def _build_lane_geometry(self, side_candidates):
        side_masks = {"left": None, "right": None}
        lane_points = []
        lane_side_points = {"left": [], "right": []}
        lane_side_lines = {"left": [], "right": []}
        lane_side_sources = {"left": "none", "right": "none"}

        for side in ("left", "right"):
            merged_mask = self._combine_masks([entry["mask"] for entry in side_candidates[side]])
            side_masks[side] = merged_mask
            if side_candidates[side]:
                lane_side_sources[side] = str(side_candidates[side][0].get("side_source", "none") or "none")
            points = self._extract_lane_points(merged_mask)
            lane_side_points[side] = [[int(point[0]), int(point[1])] for point in points]
            line = self._extract_lane_line(merged_mask)
            lane_side_lines[side] = line[0] if line is not None else []
            if points:
                lane_points.append(points)

        side_masks, lane_points, lane_side_points, lane_side_lines, lane_side_sources = (
            self._collapse_duplicate_lane_sides(
                side_masks,
                lane_points,
                lane_side_points,
                lane_side_lines,
                lane_side_sources,
            )
        )

        lane_mask = self._combine_masks([side_masks.get("left"), side_masks.get("right")])
        return side_masks, lane_points, lane_side_points, lane_side_lines, lane_side_sources, lane_mask

    @staticmethod
    def _coerce_line_values(line_value):
        if not isinstance(line_value, (list, tuple, np.ndarray)):
            return None
        values = line_value
        if len(values) == 1 and isinstance(values[0], (list, tuple, np.ndarray)):
            values = values[0]
        if len(values) < 4:
            return None
        try:
            return [float(values[idx]) for idx in range(4)]
        except (TypeError, ValueError):
            return None

    def _line_heading_rad(self, line_value):
        values = self._coerce_line_values(line_value)
        if values is None:
            return None
        x1, y1, x2, y2 = values
        if y1 < y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx = float(x2) - float(x1)
        dy = float(y1) - float(y2)
        if dy <= 1.0:
            return None
        return math.atan2(dx, dy)

    def _estimate_heading_hint(self, lane_side_lines):
        if not isinstance(lane_side_lines, dict):
            return 0.0, 0.0, "none"

        left_heading = self._line_heading_rad(lane_side_lines.get("left"))
        right_heading = self._line_heading_rad(lane_side_lines.get("right"))

        if left_heading is not None and right_heading is not None:
            return float((left_heading + right_heading) * 0.5), 0.9, "two_line"
        if left_heading is not None:
            return float(left_heading * 0.5), 0.55, "single_left"
        if right_heading is not None:
            return float(right_heading * 0.5), 0.55, "single_right"
        return 0.0, 0.0, "none"

    @staticmethod
    def _box_area(box):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return 0.0
        try:
            y1, x1, y2, x2 = [float(v) for v in box]
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, (y2 - y1) * (x2 - x1))

    def _estimate_road_type(self, detections):
        if not isinstance(detections, list):
            return "unknown", 0.0, None

        road_type_aliases = {
            "highway": {"highway_entrance", "highway_entry", "highway_exit"},
            "roundabout": {"roundabout"},
            "urban": {
                "crosswalk",
                "traffic_light",
                "red_light",
                "yellow_light",
                "green_light",
                "stop",
                "priority",
                "no_entry",
                "one_way",
                "parking",
                "speed_20",
                "speed_30",
            },
        }

        best_by_type = {}
        for detection in detections:
            class_name = str(detection.get("class", "") or "").strip().lower()
            confidence = float(detection.get("confidence", 0.0) or 0.0)
            if confidence <= 0.0:
                continue
            for road_type, aliases in road_type_aliases.items():
                if class_name in aliases and confidence > best_by_type.get(road_type, (0.0, None))[0]:
                    best_by_type[road_type] = (confidence, class_name)

        if not best_by_type:
            return "unknown", 0.0, None

        selected_type, selected = max(best_by_type.items(), key=lambda item: item[1][0])
        return selected_type, float(selected[0]), selected[1]

    def _estimate_lead_distance(self, detections):
        """Estimate distance to the closest lead vehicle/pedestrian.

        Returns:
            (distance_class, confidence, area, class_name, box_height_norm)
            distance_class: "near" / "medium" / "far" / "none"
            box_height_norm: normalized [0-1] height of the leading object's
                bounding box, useful for metric distance estimation in the
                calling thread when px_per_cm is available.
        """
        if not isinstance(detections, list):
            return "none", 0.0, 0.0, None, 0.0

        lead_aliases = {
            "vehicle",
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
            "bike",
            "pedestrian",
        }
        best = None
        for detection in detections:
            class_name = str(detection.get("class", "") or "").strip().lower()
            if class_name not in lead_aliases:
                continue
            confidence = float(detection.get("confidence", 0.0) or 0.0)
            box = detection.get("box")
            area = self._box_area(box)
            score = area * max(confidence, 1e-3)
            # Compute normalized box height [y2 - y1] for metric distance
            box_height_norm = 0.0
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                try:
                    box_height_norm = max(0.0, float(box[2]) - float(box[0]))
                except (TypeError, ValueError):
                    box_height_norm = 0.0
            if best is None or score > best["score"]:
                best = {
                    "class_name": class_name,
                    "confidence": confidence,
                    "area": area,
                    "score": score,
                    "box_height_norm": box_height_norm,
                }

        if best is None:
            return "none", 0.0, 0.0, None, 0.0

        area = float(best["area"])
        if area >= 0.08:
            distance_class = "near"
        elif area >= 0.03:
            distance_class = "medium"
        else:
            distance_class = "far"
        return distance_class, float(best["confidence"]), area, best["class_name"], float(best["box_height_norm"])

    def _draw_debug_views(self, frame, side_masks, lane_points, detections, infer_ms):
        overlay = frame.copy()
        height, width = overlay.shape[:2]
        mask_overlay = np.zeros_like(overlay)

        left_mask = side_masks.get("left")
        right_mask = side_masks.get("right")
        if left_mask is not None and np.any(left_mask):
            mask_overlay[left_mask > 0] = (255, 120, 0)
        if right_mask is not None and np.any(right_mask):
            mask_overlay[right_mask > 0] = (0, 200, 255)
        if np.any(mask_overlay):
            overlay = cv2.addWeighted(overlay, 1.0, mask_overlay, 0.35, 0)

        colors = [(255, 170, 0), (0, 220, 255)]
        for idx, line in enumerate(lane_points):
            if len(line) < 2:
                continue
            pts = np.array(line, dtype=np.int32).reshape((-1, 1, 2))
            color = colors[idx % len(colors)]
            cv2.polylines(overlay, [pts], False, color, 3, cv2.LINE_AA)
            for point in line:
                cv2.circle(overlay, point, 3, color, -1, cv2.LINE_AA)

        frame_center = width // 2
        cv2.line(overlay, (frame_center, height), (frame_center, int(height * 0.4)), (200, 200, 200), 1)

        lookahead_y = int(height * 0.6)
        left_x = self._mask_x_at_y(left_mask, lookahead_y)
        right_x = self._mask_x_at_y(right_mask, lookahead_y)
        if left_x is not None and right_x is not None:
            lane_center_x = int(round((left_x + right_x) / 2.0))
            cv2.circle(overlay, (lane_center_x, lookahead_y), 8, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.line(overlay, (frame_center, lookahead_y), (lane_center_x, lookahead_y), (0, 255, 0), 2, cv2.LINE_AA)

        signs = frame.copy()
        rendered_detections = detections[: self.sign_debug_max_detections]
        for det_idx, det in enumerate(rendered_detections):
            y1, x1, y2, x2 = det["box"]
            pt1 = (int(round(x1 * width)), int(round(y1 * height)))
            pt2 = (int(round(x2 * width)), int(round(y2 * height)))
            color = (0, 255, 0) if det_idx == 0 else (0, 180, 255)
            cv2.rectangle(overlay, pt1, pt2, color, 2)
            cv2.rectangle(signs, pt1, pt2, color, 2)
            label = f"{det['class']} {det['confidence']:.0%}"
            cv2.putText(overlay, label, (pt1[0], max(15, pt1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(signs, label, (pt1[0], max(15, pt1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        cv2.rectangle(overlay, (0, 0), (width, 38), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            f"AI LOCAL | infer:{infer_ms:.0f}ms | lanes:{len(lane_points)} | signs:{len(detections)}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        masks_view = np.zeros_like(frame)
        if left_mask is not None and np.any(left_mask):
            masks_view[left_mask > 0] = (255, 120, 0)
        if right_mask is not None and np.any(right_mask):
            masks_view[right_mask > 0] = (0, 200, 255)
        for idx, line in enumerate(lane_points):
            if len(line) < 2:
                continue
            pts = np.array(line, dtype=np.int32).reshape((-1, 1, 2))
            color = colors[idx % len(colors)]
            cv2.polylines(masks_view, [pts], False, color, 2, cv2.LINE_AA)

        return {
            "overlay": overlay,
            "masks": masks_view,
            "signs": signs,
        }

    def infer(self, frame, build_debug=False):
        """Run local perception. Never raises; degrades to an empty payload."""
        start_time = time.time()
        self.frame_count += 1
        frame_shape = frame.shape[:2]

        if not self.model_ready or self.model is None:
            inference_ms = (time.time() - start_time) * 1000
            return {
                "lane_points": [],
                "lane_side_points": {"left": [], "right": []},
                "lane_side_lines": {"left": [], "right": []},
                "lane_side_sources": {"left": "none", "right": "none"},
                "side_masks": {"left": None, "right": None},
                "lane_mask": None,
                "detections": [],
                "lane_debug": {},
                "inference_time_ms": round(inference_ms, 1),
                "frame_id": self.frame_count,
                "input_size": f"{self.imgsz}x{self.imgsz}",
                "model_ready": False,
                "error": self.init_error,
                "heading_hint_rad": 0.0,
                "heading_hint_confidence": 0.0,
                "heading_hint_source": "none",
                "road_type_class": "unknown",
                "road_type_confidence": 0.0,
                "road_type_source": None,
                "lead_distance_class": "none",
                "lead_distance_confidence": 0.0,
                "lead_distance_area": 0.0,
                "lead_distance_source": None,
                "lead_distance_px_height": 0.0,
            }

        try:
            results = self.model.predict(
                frame,
                conf=self.min_confidence,
                iou=self.nms_iou,
                imgsz=self.imgsz,
                max_det=self.max_detections,
                device=self.device,
                verbose=False,
            )
            result = results[0] if results else None
            if result is None:
                raise RuntimeError("YOLO returned no results")

            side_candidates, detections = self._split_candidates(result, frame_shape)
            side_masks, lane_points, lane_side_points, lane_side_lines, lane_side_sources, lane_mask = self._build_lane_geometry(side_candidates)
            heading_hint_rad, heading_hint_confidence, heading_hint_source = self._estimate_heading_hint(
                lane_side_lines
            )
            road_type_class, road_type_confidence, road_type_source = self._estimate_road_type(
                detections
            )
            lead_distance_class, lead_distance_confidence, lead_distance_area, lead_distance_source, lead_box_height_norm = (
                self._estimate_lead_distance(detections)
            )
            # Convert normalized box height to pixels for metric distance estimation in the caller
            lead_distance_px_height = lead_box_height_norm * float(frame_shape[0]) if lead_box_height_norm > 0.0 else 0.0
            inference_ms = (time.time() - start_time) * 1000
            lane_debug = {}
            if build_debug:
                lane_debug = self._draw_debug_views(frame, side_masks, lane_points, detections, inference_ms)
                with self._vis_lock:
                    self._last_debug = lane_debug
            return {
                "lane_points": lane_points,
                "lane_side_points": lane_side_points,
                "lane_side_lines": lane_side_lines,
                "lane_side_sources": lane_side_sources,
                "side_masks": side_masks,
                "lane_mask": lane_mask,
                "detections": detections,
                "lane_debug": lane_debug,
                "inference_time_ms": round(inference_ms, 1),
                "frame_id": self.frame_count,
                "input_size": f"{self.imgsz}x{self.imgsz}",
                "model_ready": True,
                "error": None,
                "heading_hint_rad": float(heading_hint_rad),
                "heading_hint_confidence": float(heading_hint_confidence),
                "heading_hint_source": heading_hint_source,
                "road_type_class": road_type_class,
                "road_type_confidence": float(road_type_confidence),
                "road_type_source": road_type_source,
                "lead_distance_class": lead_distance_class,
                "lead_distance_confidence": float(lead_distance_confidence),
                "lead_distance_area": float(lead_distance_area),
                "lead_distance_source": lead_distance_source,
                "lead_distance_px_height": round(lead_distance_px_height, 1),
            }
        except Exception as e:
            self.init_error = str(e)
            inference_ms = (time.time() - start_time) * 1000
            print(
                f"\033[1;97m[ Local AI ] :\033[0m \033[1;91mERROR\033[0m - "
                f"Inferencia fallo: {e}"
            )
            return {
                "lane_points": [],
                "lane_side_points": {"left": [], "right": []},
                "lane_side_lines": {"left": [], "right": []},
                "lane_side_sources": {"left": "none", "right": "none"},
                "side_masks": {"left": None, "right": None},
                "lane_mask": None,
                "detections": [],
                "lane_debug": {},
                "inference_time_ms": round(inference_ms, 1),
                "frame_id": self.frame_count,
                "input_size": f"{self.imgsz}x{self.imgsz}",
                "model_ready": False,
                "error": self.init_error,
                "heading_hint_rad": 0.0,
                "heading_hint_confidence": 0.0,
                "heading_hint_source": "none",
                "road_type_class": "unknown",
                "road_type_confidence": 0.0,
                "road_type_source": None,
                "lead_distance_class": "none",
                "lead_distance_confidence": 0.0,
                "lead_distance_area": 0.0,
                "lead_distance_source": None,
                "lead_distance_px_height": 0.0,
            }
