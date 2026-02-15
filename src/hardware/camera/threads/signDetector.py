# Traffic sign detector using MobilenetV2 SSD (TensorFlow Lite)
# Adapted from ricardolopezb/bfmc24-brain
#
# Requires:
#   pip install tflite-runtime
#   (or: pip install tensorflow — heavier, but also works)
#
# Model file: models/sign_detection/detect.tflite
# Label map:  models/sign_detection/labelmap.txt

import os
import cv2
import numpy as np

# Try multiple TFLite backends (in order of preference)
# 1. ai_edge_litert — new official name (Google LiteRT, supports Python 3.13+)
# 2. tflite_runtime — legacy lightweight package
# 3. tensorflow     — full TF (heaviest, but always has the interpreter)
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


class SignDetector:
    """MobilenetV2 SSD object detector for traffic signs using TFLite.

    Loads a .tflite model and a labelmap.txt, runs inference on a frame,
    and returns the sign with the highest confidence above the threshold.

    Args:
        model_dir: Path to directory containing detect.tflite and labelmap.txt.
                   Defaults to models/sign_detection/ (relative to project root).
        min_confidence: Minimum confidence threshold (0.0 – 1.0). Default 0.50.
    """

    def __init__(self, model_dir=None, min_confidence=0.50):
        if Interpreter is None:
            raise ImportError(
                "TFLite runtime not found. Install with: pip install tflite-runtime"
            )

        if model_dir is None:
            # Default path relative to project root
            model_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', '..', 'models', 'sign_detection'
            )
        model_dir = os.path.abspath(model_dir)

        model_path = os.path.join(model_dir, 'detect.tflite')
        labels_path = os.path.join(model_dir, 'labelmap.txt')

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"TFLite model not found at {model_path}. "
                "Download it with: python scripts/download_sign_model.py"
            )

        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines() if line.strip()]

        # Normalization constants
        self.input_mean = 127.5
        self.input_std = 127.5
        self.min_confidence = min_confidence

        # Initialize TFLite interpreter
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Model expects this input size
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        # Check if model uses float32 inputs (vs quantized uint8)
        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        # Determine output tensor ordering (TF1 vs TF2 models)
        outname = self.output_details[0]['name']
        if 'StatefulPartitionedCall' in outname:  # TF2 model
            self.boxes_idx = 1
            self.classes_idx = 3
            self.scores_idx = 0
        else:  # TF1 model
            self.boxes_idx = 0
            self.classes_idx = 1
            self.scores_idx = 2

        print(
            f"\033[1;97m[ SignDetector ] :\033[0m \033[1;92mINFO\033[0m - "
            f"Model loaded: input={self.width}x{self.height}, "
            f"float={self.floating_model}, labels={len(self.labels)}, "
            f"min_conf={self.min_confidence}"
        )

    def detect(self, frame):
        """Run inference on a BGR frame and return the highest-confidence sign.

        Args:
            frame: OpenCV BGR image (numpy array).

        Returns:
            tuple: (sign_name, confidence) or (None, 0.0) if nothing detected.
        """
        # Preprocess: BGR → RGB, resize to model input, normalize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Extract results
        boxes = self.interpreter.get_tensor(self.output_details[self.boxes_idx]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[self.classes_idx]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[self.scores_idx]['index'])[0]

        # Find the detection with the highest score above threshold
        max_score = 0.0
        max_sign = None
        max_box = None

        for i in range(len(scores)):
            if self.min_confidence < scores[i] <= 1.0:
                class_idx = int(classes[i])
                if 0 <= class_idx < len(self.labels):
                    if scores[i] > max_score:
                        max_score = float(scores[i])
                        max_sign = self.labels[class_idx]
                        max_box = boxes[i]  # [ymin, xmin, ymax, xmax] normalized

        return max_sign, max_score, max_box

    def detect_all(self, frame):
        """Run inference and return ALL detections above threshold.

        Args:
            frame: OpenCV BGR image (numpy array).

        Returns:
            list of dict: [{"sign": str, "confidence": float, "box": [y1,x1,y2,x2]}, ...]
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[self.boxes_idx]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[self.classes_idx]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[self.scores_idx]['index'])[0]

        detections = []
        for i in range(len(scores)):
            if self.min_confidence < scores[i] <= 1.0:
                class_idx = int(classes[i])
                if 0 <= class_idx < len(self.labels):
                    detections.append({
                        "sign": self.labels[class_idx],
                        "confidence": float(scores[i]),
                        "box": boxes[i].tolist(),  # [ymin, xmin, ymax, xmax]
                    })

        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections
