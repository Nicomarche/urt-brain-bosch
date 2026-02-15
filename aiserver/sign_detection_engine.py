"""
Motor de detección de señales de tráfico usando MobilenetV2 SSD (TFLite).

Corre en el AI Server (PC con GPU o CPU potente), recibe frames por WebSocket
y devuelve las señales detectadas con bounding boxes y confianza.

Modelo: models/sign_detection/detect.tflite (11MB)
Labels: models/sign_detection/labelmap.txt (9 clases)

Clases: stop, parking, priority, crosswalk, highway_entrance,
        highway_exit, roundabout, one_way, no_entry
"""

import os
import time
import cv2
import numpy as np

# Try multiple TFLite backends
Interpreter = None
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError:
            Interpreter = None


class SignDetectionEngine:
    """TFLite MobilenetV2 SSD engine for traffic sign detection.

    Compatible con la interfaz del AI Server (infer / infer_steering_only / get_status).
    """

    def __init__(self, model_dir=None, min_confidence=None):
        import config as cfg

        if Interpreter is None:
            raise ImportError(
                "TFLite runtime no encontrado. Instala con: "
                "pip install ai-edge-litert  (o: pip install tflite-runtime)"
            )

        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "models", "sign_detection")

        if min_confidence is None:
            min_confidence = getattr(cfg, "SIGN_MIN_CONFIDENCE", 0.50)

        model_path = os.path.join(model_dir, "detect.tflite")
        labels_path = os.path.join(model_dir, "labelmap.txt")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Modelo TFLite no encontrado: {model_path}\n"
                f"Copia detect.tflite y labelmap.txt a {model_dir}/"
            )

        # Load labels
        with open(labels_path, "r") as f:
            self.labels = [line.strip() for line in f.readlines() if line.strip()]

        self.min_confidence = min_confidence
        self.input_mean = 127.5
        self.input_std = 127.5

        # Initialize interpreter
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.height = self.input_details[0]["shape"][1]
        self.width = self.input_details[0]["shape"][2]
        self.floating_model = self.input_details[0]["dtype"] == np.float32

        # Output tensor ordering (TF1 vs TF2)
        outname = self.output_details[0]["name"]
        if "StatefulPartitionedCall" in outname:
            self.boxes_idx, self.classes_idx, self.scores_idx = 1, 3, 0
        else:
            self.boxes_idx, self.classes_idx, self.scores_idx = 0, 1, 2

        self.frame_count = 0
        self.total_infer_ms = 0
        self.show_visualization = getattr(cfg, "SHOW_VISUALIZATION", False)

        # Warm-up
        print(f"[SignDetection] Warming up model ({self.width}x{self.height})...")
        dummy = np.zeros((1, self.height, self.width, 3), dtype=np.float32 if self.floating_model else np.uint8)
        self.interpreter.set_tensor(self.input_details[0]["index"], dummy)
        self.interpreter.invoke()

        print(
            f"[SignDetection] Engine ready: model={self.width}x{self.height}, "
            f"float={self.floating_model}, labels={len(self.labels)}, "
            f"min_conf={self.min_confidence}"
        )

    def infer(self, frame):
        """Run full inference on a BGR frame.

        Args:
            frame: OpenCV BGR image.

        Returns:
            dict with detections, inference time, frame id.
        """
        t_start = time.time()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[self.boxes_idx]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[self.classes_idx]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[self.scores_idx]["index"])[0]

        detections = []
        for i in range(len(scores)):
            if self.min_confidence < scores[i] <= 1.0:
                class_idx = int(classes[i])
                if 0 <= class_idx < len(self.labels):
                    detections.append({
                        "sign": self.labels[class_idx],
                        "confidence": round(float(scores[i]), 3),
                        "box": [round(float(v), 4) for v in boxes[i].tolist()],
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
                # Disable visualization if it crashes (e.g. no display)
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
            "engine": "sign_detection",
            "model": f"MobilenetV2 SSD ({self.width}x{self.height})",
            "labels": self.labels,
            "min_confidence": self.min_confidence,
            "floating_model": self.floating_model,
            "frames_processed": self.frame_count,
            "avg_inference_ms": round(avg_ms, 1),
        }

    def _visualize(self, frame, detections, infer_ms):
        """Show debug window on the server."""
        vis = frame.copy()
        h, w = vis.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        COLORS = {
            "stop": (0, 0, 255), "no_entry": (0, 0, 200),
            "parking": (255, 160, 0), "crosswalk": (0, 200, 255),
            "highway_entrance": (0, 255, 0), "highway_exit": (0, 180, 0),
            "roundabout": (255, 0, 255), "priority": (0, 255, 255),
            "one_way": (255, 255, 0),
        }

        for det in detections:
            sign = det["sign"]
            conf = det["confidence"]
            box = det["box"]
            color = COLORS.get(sign, (0, 255, 0))

            ymin, xmin, ymax, xmax = (
                int(box[0] * h), int(box[1] * w),
                int(box[2] * h), int(box[3] * w),
            )
            cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"{sign} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, font, 0.55, 2)
            cv2.rectangle(vis, (xmin, ymin - th - 6), (xmin + tw + 6, ymin), color, -1)
            cv2.putText(vis, label, (xmin + 3, ymin - 3), font, 0.55, (0, 0, 0), 2)

        # Status bar
        status = f"Signs: {len(detections)} | {infer_ms:.0f}ms | Frame {self.frame_count}"
        cv2.putText(vis, status, (10, 25), font, 0.6, (0, 255, 255), 2)

        cv2.imshow("Sign Detection Server", vis)
        cv2.waitKey(1)

    def shutdown(self):
        """Cleanup."""
        print(f"[SignDetection] Shutdown ({self.frame_count} frames processed)")
        try:
            cv2.destroyWindow("Sign Detection Server")
        except Exception:
            pass
