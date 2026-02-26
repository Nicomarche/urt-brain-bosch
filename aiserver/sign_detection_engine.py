"""
YOLOv8 multi-model sign/element detection engine.

This engine can load one or many YOLOv8 models and merge detections in the
same output payload expected by /ws/signs.
"""

import os
import time
import threading

import cv2
import numpy as np


class SignDetectionEngine:
    """YOLOv8 sign detection engine compatible with the AI Server interface."""

    SIGN_COLORS = {
        "stop": (0, 0, 255),
        "no_entry": (0, 0, 200),
        "red_light": (0, 0, 230),
        "yellow_light": (0, 200, 255),
        "green_light": (0, 200, 0),
        "crosswalk": (0, 200, 255),
        "parking": (255, 160, 0),
        "no_parking": (80, 80, 200),
        "speed_20": (255, 100, 0),
        "speed_30": (255, 150, 0),
        "turn_right": (200, 200, 0),
        "turn_left": (200, 200, 0),
        "highway_entrance": (0, 255, 0),
        "highway_exit": (0, 180, 0),
        "pedestrian": (255, 0, 255),
        "vehicle": (255, 255, 0),
        "road_block": (0, 100, 255),
        "traffic_light": (120, 200, 255),
        "priority": (255, 255, 0),
        "roundabout": (255, 100, 255),
        "one_way": (255, 220, 80),
    }
    DEFAULT_COLOR = (0, 255, 0)

    def __init__(self, model_dir=None, min_confidence=None):
        import config as cfg

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics no encontrado. Instala con: pip install ultralytics"
            )

        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(__file__), "models", "sign_detection"
            )

        if min_confidence is None:
            min_confidence = getattr(cfg, "SIGN_MIN_CONFIDENCE", 0.50)

        self.model_dir = model_dir
        self.default_min_confidence = float(min_confidence)
        self.default_imgsz = int(getattr(cfg, "SIGN_IMGSZ", 320))
        self.show_visualization = getattr(cfg, "SHOW_VISUALIZATION", False)
        self.class_map = getattr(cfg, "SIGN_CLASS_MAP", {})
        self.sort_order = getattr(
            cfg, "SIGN_DETECTION_SORT_ORDER", {"sign": 0, "element": 1}
        )

        self.device = self._resolve_device(getattr(cfg, "SIGN_DEVICE", "auto"))
        self.models = self._load_models(cfg, YOLO)

        if not self.models:
            raise RuntimeError(
                "[SignDetection] No se cargo ningun modelo. "
                "Revisa SIGN_MODEL_FILE o SIGN_MULTI_MODELS en config.py."
            )

        self.min_confidence = min(
            model_cfg["min_confidence"] for model_cfg in self.models
        )
        self.imgsz = max(model_cfg["imgsz"] for model_cfg in self.models)

        self.frame_count = 0
        self.total_infer_ms = 0.0

        # Keep compatibility with previous status fields
        self.labels = self.models[0]["labels"]
        self.num_classes = sum(len(model_cfg["labels"]) for model_cfg in self.models)

        # Latest JPEG for MJPEG streaming
        self._vis_jpeg = None
        self._vis_lock = threading.Lock()

        print(
            f"[SignDetection] Engine ready: {len(self.models)} model(s) on {self.device} | "
            f"min_conf={self.min_confidence} | "
            f"{'visualization ON (/viz/signs)' if self.show_visualization else 'visualization OFF'}"
        )

    @staticmethod
    def _normalize(name):
        return str(name).strip().lower().replace(" ", "_")

    def _resolve_device(self, device):
        if device != "auto":
            return device
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_model_path(self, model_file):
        if os.path.isabs(model_file):
            return model_file
        return os.path.join(self.model_dir, model_file)

    def _normalize_name_set(self, values):
        if not values:
            return None
        normalized = {self._normalize(v) for v in values if v is not None}
        return normalized if normalized else None

    def _build_model_spec(self, spec, fallback_name):
        if not isinstance(spec, dict):
            raise ValueError(
                f"[SignDetection] Configuracion invalida para modelo '{fallback_name}'. "
                "Debe ser un dict."
            )

        model_file = spec.get("file") or spec.get("model_file")
        if not model_file:
            raise ValueError(
                f"[SignDetection] Configuracion invalida para modelo '{fallback_name}': falta 'file'."
            )

        return {
            "name": str(spec.get("name", fallback_name)),
            "file": model_file,
            "kind": str(spec.get("kind", "sign")),
            "imgsz": int(spec.get("imgsz", self.default_imgsz)),
            "min_confidence": float(
                spec.get("min_confidence", self.default_min_confidence)
            ),
            "class_map": spec.get("class_map", {}) or {},
            "include_classes": self._normalize_name_set(spec.get("include_classes")),
            "exclude_classes": self._normalize_name_set(spec.get("exclude_classes")),
        }

    def _iter_model_specs(self, cfg):
        # Preferred mode: select ONE model profile from config.py.
        options = getattr(cfg, "SIGN_MODEL_OPTIONS", None)
        active_key = getattr(cfg, "SIGN_ACTIVE_MODEL", None)

        if isinstance(options, dict) and len(options) > 0:
            if not active_key:
                active_key = next(iter(options.keys()))
                print(
                    f"[SignDetection] WARNING: SIGN_ACTIVE_MODEL vacio. "
                    f"Usando '{active_key}'."
                )
            if active_key not in options:
                available = ", ".join(sorted(options.keys()))
                raise ValueError(
                    f"[SignDetection] SIGN_ACTIVE_MODEL='{active_key}' no existe. "
                    f"Opciones: {available}"
                )
            yield self._build_model_spec(options[active_key], str(active_key))
            return

        # Backward-compatible fallback: a single SIGN_MODEL_FILE.
        yield self._build_model_spec(
            {
                "name": "signs_default",
                "file": getattr(cfg, "SIGN_MODEL_FILE", "trafic.pt"),
                "kind": "sign",
                "imgsz": self.default_imgsz,
                "min_confidence": self.default_min_confidence,
            },
            "signs_default",
        )

    def _load_models(self, cfg, yolo_cls):
        loaded_models = []
        print("=" * 60)
        print("[SignDetection] Loading model(s)")
        print(f"[SignDetection] Base dir: {self.model_dir}")
        print(f"[SignDetection] Device: {self.device}")

        for spec in self._iter_model_specs(cfg):
            model_path = self._resolve_model_path(spec["file"])
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"Modelo YOLOv8 no encontrado: {model_path}\n"
                    f"Revisa SIGN_ACTIVE_MODEL / SIGN_MODEL_OPTIONS en config.py"
                )

            print(
                f"[SignDetection] -> {spec['name']}: {model_path} "
                f"(kind={spec['kind']}, imgsz={spec['imgsz']}, "
                f"min_conf={spec['min_confidence']})"
            )
            model = yolo_cls(model_path)
            if self.device != "cpu":
                model.to(self.device)

            labels = model.names
            print(f"[SignDetection]    classes: {len(labels)}")
            for idx, class_name in labels.items():
                mapped = self._map_class_name(class_name, spec["class_map"])
                keep = self._class_allowed(spec, class_name, mapped)
                keep_tag = "" if keep else " (filtered)"
                map_tag = "" if mapped == class_name else f" -> {mapped}"
                print(f"[SignDetection]      [{idx}] {class_name}{map_tag}{keep_tag}")

            loaded_models.append(
                {
                    "name": spec["name"],
                    "file": spec["file"],
                    "path": model_path,
                    "kind": spec["kind"],
                    "imgsz": spec["imgsz"],
                    "min_confidence": spec["min_confidence"],
                    "class_map": spec["class_map"],
                    "include_classes": spec["include_classes"],
                    "exclude_classes": spec["exclude_classes"],
                    "model": model,
                    "labels": labels,
                    "num_classes": len(labels),
                }
            )

        self._warmup_models(loaded_models)
        print("=" * 60)
        return loaded_models

    def _warmup_models(self, model_cfgs, runs=2):
        if runs <= 0:
            return
        print(f"[SignDetection] Warmup ({runs} runs/model)")
        for model_cfg in model_cfgs:
            dummy = np.zeros((model_cfg["imgsz"], model_cfg["imgsz"], 3), dtype=np.uint8)
            for run_idx in range(runs):
                t0 = time.time()
                model_cfg["model"].predict(
                    dummy,
                    conf=model_cfg["min_confidence"],
                    verbose=False,
                    imgsz=model_cfg["imgsz"],
                    device=self.device,
                )
                elapsed_ms = (time.time() - t0) * 1000
                print(
                    f"[SignDetection]   {model_cfg['name']} warmup {run_idx + 1}/{runs}: "
                    f"{elapsed_ms:.0f}ms"
                )

    def _map_class_name(self, class_name, local_class_map=None):
        if local_class_map and class_name in local_class_map:
            return local_class_map[class_name]
        if class_name in self.class_map:
            return self.class_map[class_name]
        return self._normalize(class_name)

    def _class_allowed(self, model_cfg, raw_class_name, mapped_name):
        raw_norm = self._normalize(raw_class_name)
        mapped_norm = self._normalize(mapped_name)

        include_classes = model_cfg["include_classes"]
        exclude_classes = model_cfg["exclude_classes"]

        if include_classes and raw_norm not in include_classes and mapped_norm not in include_classes:
            return False
        if exclude_classes and (raw_norm in exclude_classes or mapped_norm in exclude_classes):
            return False
        return True

    def infer(self, frame):
        """Run sign/element detection over a BGR frame."""
        t_start = time.time()
        h, w = frame.shape[:2]
        detections = []

        for model_cfg in self.models:
            results = model_cfg["model"].predict(
                frame,
                conf=model_cfg["min_confidence"],
                verbose=False,
                imgsz=model_cfg["imgsz"],
                device=self.device,
            )

            if not results:
                continue

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes
            labels = model_cfg["labels"]
            for idx in range(len(boxes)):
                conf = float(boxes.conf[idx])
                cls_idx = int(boxes.cls[idx])
                class_name = labels.get(cls_idx, f"class_{cls_idx}")
                mapped_name = self._map_class_name(class_name, model_cfg["class_map"])

                if not self._class_allowed(model_cfg, class_name, mapped_name):
                    continue

                x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
                box_normalized = [
                    round(y1 / h, 4),  # ymin
                    round(x1 / w, 4),  # xmin
                    round(y2 / h, 4),  # ymax
                    round(x2 / w, 4),  # xmax
                ]

                detections.append(
                    {
                        "sign": mapped_name,
                        "confidence": round(conf, 3),
                        "box": box_normalized,
                        "kind": model_cfg["kind"],
                        "model": model_cfg["name"],
                        "class_raw": class_name,
                    }
                )

        detections.sort(
            key=lambda det: (
                self.sort_order.get(det.get("kind", "sign"), 999),
                -det["confidence"],
            )
        )

        infer_ms = (time.time() - t_start) * 1000
        self.frame_count += 1
        self.total_infer_ms += infer_ms

        if self.show_visualization:
            try:
                self._visualize(frame, detections, infer_ms)
            except Exception:
                self.show_visualization = False
                print("[SignDetection] Visualization disabled (OpenCV/headless error)")

        return {
            "detections": detections,
            "inference_time_ms": round(infer_ms, 1),
            "frame_id": self.frame_count,
        }

    def infer_steering_only(self, frame):
        """Alias for infer(); required by server interface."""
        return self.infer(frame)

    def get_status(self):
        """Return engine status (used by /status endpoint)."""
        avg_ms = (self.total_infer_ms / self.frame_count) if self.frame_count > 0 else 0.0
        return {
            "engine": "sign_detection_yolov8_multi" if len(self.models) > 1 else "sign_detection_yolov8",
            "device": self.device,
            "model_count": len(self.models),
            "models": [
                {
                    "name": model_cfg["name"],
                    "file": model_cfg["file"],
                    "kind": model_cfg["kind"],
                    "imgsz": model_cfg["imgsz"],
                    "min_confidence": model_cfg["min_confidence"],
                    "num_classes": model_cfg["num_classes"],
                    "labels": list(model_cfg["labels"].values()),
                }
                for model_cfg in self.models
            ],
            "class_map": self.class_map,
            "frames_processed": self.frame_count,
            "avg_inference_ms": round(avg_ms, 1),
        }

    def get_vis_jpeg(self):
        """Return latest annotated frame as JPEG bytes for MJPEG streaming."""
        with self._vis_lock:
            return self._vis_jpeg

    def _visualize(self, frame, detections, infer_ms):
        vis = frame.copy()
        h, w = vis.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        for det in detections:
            sign = det["sign"]
            conf = det["confidence"]
            box = det["box"]
            kind = det.get("kind", "sign")
            source_model = det.get("model", "model")

            color = self.SIGN_COLORS.get(sign, self.DEFAULT_COLOR)
            ymin = int(box[0] * h)
            xmin = int(box[1] * w)
            ymax = int(box[2] * h)
            xmax = int(box[3] * w)

            cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), color, 2)

            label = f"{sign} {conf:.0%} [{kind}:{source_model}]"
            (text_w, text_h), _ = cv2.getTextSize(label, font, 0.45, 1)
            label_y = max(ymin, text_h + 8)
            cv2.rectangle(
                vis,
                (xmin, label_y - text_h - 6),
                (xmin + text_w + 6, label_y + 2),
                color,
                -1,
            )
            cv2.putText(
                vis, label, (xmin + 3, label_y - 3), font, 0.45, (0, 0, 0), 1
            )

        bar_h = 36
        cv2.rectangle(vis, (0, 0), (w, bar_h), (0, 0, 0), -1)

        avg_ms = (self.total_infer_ms / self.frame_count) if self.frame_count > 0 else 0.0
        fps_est = (1000.0 / avg_ms) if avg_ms > 0 else 0.0
        status = (
            f"Dets:{len(detections)} | {infer_ms:.0f}ms ({fps_est:.0f} FPS) | "
            f"Frame {self.frame_count} | Models:{len(self.models)}"
        )
        cv2.putText(vis, status, (8, 24), font, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        _, jpeg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with self._vis_lock:
            self._vis_jpeg = jpeg.tobytes()

    def shutdown(self):
        print(f"[SignDetection] Shutdown ({self.frame_count} frames processed)")
