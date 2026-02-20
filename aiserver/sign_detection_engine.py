"""
Motor de detección de señales de tráfico usando YOLOv8 (ultralytics).

Corre en el AI Server (PC con GPU o CPU potente), recibe frames por WebSocket
y devuelve las señales detectadas con bounding boxes y confianza.

Modelo: models/sign_detection/trafic.pt (22MB, YOLOv8)
Fuente: https://huggingface.co/nezahatkorkmaz/traffic-sign-detection

Las clases del modelo se imprimen al inicio. Se puede configurar un mapeo
de nombres de clase del modelo → nombres de acción del auto en config.py
(SIGN_CLASS_MAP) para que coincidan con las acciones en SignActions.
"""

import os
import time
import threading
import cv2
import numpy as np


class SignDetectionEngine:
    """YOLOv8 engine for traffic sign detection.

    Compatible con la interfaz del AI Server (infer / infer_steering_only / get_status).
    """

    def __init__(self, model_dir=None, min_confidence=None):
        import config as cfg

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics no encontrado. Instala con: pip install ultralytics"
            )

        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "models", "sign_detection")

        if min_confidence is None:
            min_confidence = getattr(cfg, "SIGN_MIN_CONFIDENCE", 0.50)

        model_path = os.path.join(model_dir, getattr(cfg, "SIGN_MODEL_FILE", "trafic.pt"))

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Modelo YOLOv8 no encontrado: {model_path}\n"
                f"Descarga trafic.pt de HuggingFace a {model_dir}/"
            )

        self.min_confidence = min_confidence
        self.show_visualization = getattr(cfg, "SHOW_VISUALIZATION", False)
        self.imgsz = getattr(cfg, "SIGN_IMGSZ", 320)

        # Device selection: auto-detect MPS (Apple Silicon) > CUDA > CPU
        device = getattr(cfg, "SIGN_DEVICE", "auto")
        if device == "auto":
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # Class name mapping: model_class_name → our_action_name
        # If a class is not in the map, the original name is used (lowercased).
        self.class_map = getattr(cfg, "SIGN_CLASS_MAP", {})

        # Load YOLOv8 model
        print(f"[SignDetection] Loading YOLOv8 model: {model_path}")
        print(f"[SignDetection] Device: {self.device} | imgsz: {self.imgsz}")
        self.model = YOLO(model_path)

        # Move model to device
        if self.device != "cpu":
            self.model.to(self.device)
            print(f"[SignDetection] Model moved to {self.device}")

        # Extract class names from model
        self.labels = self.model.names  # dict: {0: "class0", 1: "class1", ...}
        self.num_classes = len(self.labels)

        print(f"[SignDetection] Model classes ({self.num_classes}):")
        for idx, name in self.labels.items():
            mapped = self.class_map.get(name, name.lower().replace(" ", "_"))
            suffix = f" → {mapped}" if mapped != name else ""
            print(f"  [{idx}] {name}{suffix}")

        # Warm-up (run 3 times to stabilize timings)
        print(f"[SignDetection] Warming up model (3 runs)...")
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for i in range(3):
            t0 = time.time()
            self.model.predict(
                dummy, conf=self.min_confidence, verbose=False,
                imgsz=self.imgsz, device=self.device,
            )
            ms = (time.time() - t0) * 1000
            print(f"  warmup {i+1}: {ms:.0f}ms")

        self.frame_count = 0
        self.total_infer_ms = 0

        # Visualization: store latest annotated frame as JPEG for MJPEG streaming.
        # Cross-platform (macOS/Linux/headless) — no cv2.imshow needed.
        self._vis_jpeg = None
        self._vis_lock = threading.Lock()

        print(
            f"[SignDetection] Engine ready: YOLOv8 on {self.device}, "
            f"imgsz={self.imgsz}, classes={self.num_classes}, "
            f"min_conf={self.min_confidence}"
            f"{', visualization=ON (MJPEG at /viz)' if self.show_visualization else ''}"
        )

    def _map_class_name(self, name):
        """Map model class name to our action name."""
        if name in self.class_map:
            return self.class_map[name]
        return name.lower().replace(" ", "_")

    def infer(self, frame):
        """Run full inference on a BGR frame.

        Args:
            frame: OpenCV BGR image.

        Returns:
            dict with detections, inference time, frame id.
        """
        t_start = time.time()
        h, w = frame.shape[:2]

        # Run YOLOv8 inference
        results = self.model.predict(
            frame,
            conf=self.min_confidence,
            verbose=False,
            imgsz=self.imgsz,
            device=self.device,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    cls_idx = int(boxes.cls[i])
                    cls_name = self.labels.get(cls_idx, f"class_{cls_idx}")
                    mapped_name = self._map_class_name(cls_name)

                    # Get box in xyxy pixel format and normalize to [0,1]
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                    # Convert to [ymin, xmin, ymax, xmax] normalized format
                    # (compatible with existing client protocol)
                    box_normalized = [
                        round(y1 / h, 4),  # ymin
                        round(x1 / w, 4),  # xmin
                        round(y2 / h, 4),  # ymax
                        round(x2 / w, 4),  # xmax
                    ]

                    detections.append({
                        "sign": mapped_name,
                        "confidence": round(conf, 3),
                        "box": box_normalized,
                    })

        detections.sort(key=lambda d: d["confidence"], reverse=True)

        infer_ms = (time.time() - t_start) * 1000
        self.frame_count += 1
        self.total_infer_ms += infer_ms

        # Optional visualization on the server (wrapped in try to handle headless)
        if self.show_visualization:
            try:
                self._visualize(frame, detections, infer_ms)
            except Exception:
                self.show_visualization = False
                print("[SignDetection] Visualization disabled (no display or OpenCV error)")

        return {
            "detections": detections,
            "inference_time_ms": round(infer_ms, 1),
            "frame_id": self.frame_count,
        }

    def infer_steering_only(self, frame):
        """Alias for infer() — sign detection doesn't compute steering."""
        return self.infer(frame)

    def get_status(self):
        """Return engine status (for /status endpoint)."""
        avg_ms = (self.total_infer_ms / self.frame_count) if self.frame_count > 0 else 0
        return {
            "engine": "sign_detection_yolov8",
            "model": "YOLOv8 (trafic.pt)",
            "labels": list(self.labels.values()),
            "class_map": self.class_map,
            "min_confidence": self.min_confidence,
            "num_classes": self.num_classes,
            "frames_processed": self.frame_count,
            "avg_inference_ms": round(avg_ms, 1),
        }

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
    }
    DEFAULT_COLOR = (0, 255, 0)

    def get_vis_jpeg(self):
        """Return the latest annotated frame as JPEG bytes (for MJPEG streaming)."""
        with self._vis_lock:
            return self._vis_jpeg

    def _visualize(self, frame, detections, infer_ms):
        """Render bounding boxes on frame and store as JPEG for MJPEG streaming."""
        vis = frame.copy()
        h, w = vis.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        for det in detections:
            sign = det["sign"]
            conf = det["confidence"]
            box = det["box"]
            color = self.SIGN_COLORS.get(sign, self.DEFAULT_COLOR)

            ymin = int(box[0] * h)
            xmin = int(box[1] * w)
            ymax = int(box[2] * h)
            xmax = int(box[3] * w)

            cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), color, 2)

            label = f"{sign} {conf:.0%}"
            (tw, th_t), _ = cv2.getTextSize(label, font, 0.55, 2)
            label_y = max(ymin, th_t + 8)
            cv2.rectangle(
                vis,
                (xmin, label_y - th_t - 6),
                (xmin + tw + 6, label_y + 2),
                color, -1,
            )
            cv2.putText(vis, label, (xmin + 3, label_y - 3), font, 0.55, (0, 0, 0), 2)

        # Status bar
        bar_h = 32
        overlay = vis[:bar_h, :].copy()
        cv2.rectangle(vis, (0, 0), (w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, vis[:bar_h, :], 0.7, 0, vis[:bar_h, :])

        avg_ms = (self.total_infer_ms / self.frame_count) if self.frame_count > 0 else 0
        fps_est = 1000.0 / avg_ms if avg_ms > 0 else 0
        status = (
            f"Signs: {len(detections)} | "
            f"{infer_ms:.0f}ms ({fps_est:.0f} FPS) | "
            f"Frame {self.frame_count}"
        )
        cv2.putText(vis, status, (8, 22), font, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        _, jpeg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with self._vis_lock:
            self._vis_jpeg = jpeg.tobytes()

    def shutdown(self):
        """Cleanup."""
        print(f"[SignDetection] Shutdown ({self.frame_count} frames processed)")
