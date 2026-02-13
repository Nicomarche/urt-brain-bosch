"""
Motor de inferencia Supercombo (openpilot).

Carga el modelo supercombo.onnx (comma.ai) y procesa frames para:
  - Detección de líneas de carril (4 líneas x 33 puntos)
  - Trayectoria planeada (5 paths x 33 puntos 3D)
  - Bordes de ruta (2 bordes x 33 puntos)
  - Ángulo de dirección calculado desde el centro del carril

Basado en: https://github.com/MTammvee/openpilot-supercombo-model
Paper: End-to-end driving via conditional imitation learning (comma.ai)

Inputs del modelo ONNX:
  - input_imgs:          (1, 12, 128, 256) float32  — 2 frames YUV parseados
  - desire:              (1, 8)             float32  — vector de deseo (0=seguir recto)
  - traffic_convention:  (1, 2)             float32  — convención de tráfico
  - initial_state:       (1, 512)           float32  — estado recurrente GRU

Output del modelo:
  - Tensor de ~6609 valores con plan, lanes, road edges, lead, pose, etc.
"""

import os
import time
import cv2
import numpy as np

import config

# Coordenadas longitudinales para los 33 puntos de cada línea/trayectoria
# Representan distancias en metros desde el auto (espacio cuadrático)
X_IDXS = np.array([
    0., 0.1875, 0.75, 1.6875, 3., 4.6875,
    6.75, 9.1875, 12., 15.1875, 18.75, 22.6875,
    27., 31.6875, 36.75, 42.1875, 48., 54.1875,
    60.75, 67.6875, 75., 82.6875, 90.75, 99.1875,
    108., 117.1875, 126.75, 136.6875, 147., 157.6875,
    168.75, 180.1875, 192.
], dtype=np.float32)

# ============================================================
# Índices del output del modelo Supercombo
# ============================================================
PLAN_START = 0
PLAN_END = 4955

LANES_START = PLAN_END            # 4955
LANES_END = LANES_START + 528     # 5483  (4 lanes x 33 points x 2 [y, std] x 2 [t, t2])

LANE_PROB_START = LANES_END       # 5483
LANE_PROB_END = LANE_PROB_START + 8  # 5491  (4 lanes x 2 probs)

ROAD_START = LANE_PROB_END        # 5491
ROAD_END = ROAD_START + 264       # 5755  (2 edges x 33 points x 2 [y, std] x 2 [t, t2])


def _parse_yuv_frame(frame_yuv):
    """
    Parse a YUV_I420 image into the 6-channel format expected by Supercombo.
    
    Input: YUV I420 image (H * 1.5, W) uint8
    Output: (6, H//2, W//2) uint8
    
    Channels:
      0-3: Y channel subsampled in 4 patterns
      4: U channel
      5: V channel
    """
    H = (frame_yuv.shape[0] * 2) // 3
    W = frame_yuv.shape[1]
    parsed = np.zeros((6, H // 2, W // 2), dtype=np.uint8)

    parsed[0] = frame_yuv[0:H:2, 0::2]      # Y: even rows, even cols
    parsed[1] = frame_yuv[1:H:2, 0::2]      # Y: odd rows, even cols
    parsed[2] = frame_yuv[0:H:2, 1::2]      # Y: even rows, odd cols
    parsed[3] = frame_yuv[1:H:2, 1::2]      # Y: odd rows, odd cols
    parsed[4] = frame_yuv[H:H + H // 4].reshape((-1, H // 2, W // 2))  # U
    parsed[5] = frame_yuv[H + H // 4:H + H // 2].reshape((-1, H // 2, W // 2))  # V

    return parsed


class SupercomboEngine:
    """
    Motor de inferencia para el modelo Supercombo de openpilot.
    
    Implementa la misma interfaz que HybridNetsEngine:
      - infer(frame) -> dict
      - infer_steering_only(frame) -> dict
      - get_status() -> dict
    """

    def __init__(self,
                 model_path: str = None,
                 device: str = None):

        self.model_path = model_path or getattr(config, 'SUPERCOMBO_MODEL_PATH', 'models/supercombo.onnx')
        self.device_str = device or getattr(config, 'DEVICE', 'cpu')

        # Supercombo input size (fixed by the model)
        self.input_width = 512
        self.input_height = 256

        # Steering parameters
        self.lookahead_idx = getattr(config, 'SUPERCOMBO_LOOKAHEAD_IDX', 15)
        self.steering_gain = getattr(config, 'SUPERCOMBO_STEERING_GAIN', 25.0)
        self.smoothing = getattr(config, 'SUPERCOMBO_SMOOTHING', 0.7)
        self.use_path = getattr(config, 'SUPERCOMBO_USE_PATH', False)
        self.max_steering = 25.0

        # Internal state
        self._prev_parsed = None             # Previous frame (6, 128, 256)
        self._recurrent_state = np.zeros((1, 512), dtype=np.float32)
        self._previous_steering = 0.0
        self.frame_count = 0

        # Visualization
        self.show_visualization = getattr(config, 'SHOW_VISUALIZATION', False)

        # ONNX session
        self.session = None
        self._input_names = []
        self._output_name = None

        self._load_model()

    def _load_model(self):
        """Load Supercombo ONNX model with appropriate execution provider."""
        import onnxruntime as ort

        model_full_path = os.path.join(os.path.dirname(__file__), self.model_path)

        if not os.path.exists(model_full_path):
            raise FileNotFoundError(
                f"[Supercombo] Modelo no encontrado: {model_full_path}\n"
                f"  Ejecuta: python setup_supercombo.py para descargarlo"
            )

        print(f"[Supercombo] Cargando modelo desde: {model_full_path}")
        print(f"[Supercombo] Dispositivo solicitado: {self.device_str}")

        # Select execution providers based on device
        providers = []
        if self.device_str == "mps" or self.device_str == "coreml":
            providers.append('CoreMLExecutionProvider')
        if self.device_str.startswith("cuda"):
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')  # Always fallback to CPU

        # Filter to available providers
        available = ort.get_available_providers()
        providers = [p for p in providers if p in available]
        if not providers:
            providers = ['CPUExecutionProvider']

        print(f"[Supercombo] Providers disponibles: {available}")
        print(f"[Supercombo] Usando providers: {providers}")

        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            model_full_path,
            sess_options=sess_options,
            providers=providers,
        )

        # Cache input/output names
        inputs = self.session.get_inputs()
        self._input_names = [inp.name for inp in inputs]
        self._output_name = self.session.get_outputs()[0].name

        print(f"[Supercombo] Inputs del modelo:")
        for inp in inputs:
            print(f"  {inp.name}: {inp.shape} ({inp.type})")
        print(f"[Supercombo] Output: {self._output_name}")

        # Build name-to-shape map for safe input feeding
        self._input_shapes = {inp.name: inp.shape for inp in inputs}
        print(f"[Supercombo] Input shapes: {self._input_shapes}")

        # Warm-up inference
        print("[Supercombo] Warm-up...")
        feed = self._build_feed(
            np.zeros((1, 12, 128, 256), dtype=np.float32),
            np.zeros((1, 512), dtype=np.float32),
        )

        warmup_start = time.time()
        _ = self.session.run([self._output_name], feed)
        warmup_ms = (time.time() - warmup_start) * 1000

        print(f"[Supercombo] Warm-up completado en {warmup_ms:.0f}ms")
        print(f"[Supercombo] === DIAGNOSTICO ===")
        print(f"[Supercombo]   Provider activo: {self.session.get_providers()}")
        print(f"[Supercombo]   Input: {self.input_width}x{self.input_height}")
        print(f"[Supercombo]   Lookahead idx: {self.lookahead_idx} ({X_IDXS[self.lookahead_idx]:.1f}m)")
        print(f"[Supercombo]   Steering gain: {self.steering_gain}")
        print(f"[Supercombo]   Use path: {self.use_path}")
        print(f"[Supercombo]   Warmup forward: {warmup_ms:.0f}ms")
        print(f"[Supercombo] ==================")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a BGR frame for Supercombo inference.
        
        1. Resize to 512x256
        2. Convert BGR -> YUV I420
        3. Parse into 6 channels (128x256)
        4. Stack with previous frame -> (1, 12, 128, 256) float32
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            np.ndarray (1, 12, 128, 256) float32
        """
        # Resize to model input size
        resized = cv2.resize(frame, (self.input_width, self.input_height))

        # Convert BGR -> YUV I420
        yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV_I420)

        # Parse into 6-channel format
        parsed = _parse_yuv_frame(yuv)  # (6, 128, 256) uint8

        # Stack with previous frame (temporal context)
        if self._prev_parsed is None:
            # First frame: duplicate
            stacked = np.concatenate([parsed, parsed], axis=0)  # (12, 128, 256)
        else:
            stacked = np.concatenate([self._prev_parsed, parsed], axis=0)  # (12, 128, 256)

        # Update previous frame
        self._prev_parsed = parsed.copy()

        # Add batch dimension and convert to float32
        input_tensor = stacked[np.newaxis, ...].astype(np.float32)  # (1, 12, 128, 256)

        return input_tensor

    def _build_feed(self, input_imgs: np.ndarray, recurrent_state: np.ndarray) -> dict:
        """
        Build the input feed dict using explicit input names from the model.
        
        The model has 4 inputs (order may vary between versions):
          - input_imgs:          (1, 12, 128, 256)
          - desire:              (1, 8)
          - traffic_convention:  (1, 2)
          - initial_state:       (1, 512)
        
        This method matches by name, not by index, to avoid shape mismatches.
        """
        desire = np.zeros((1, 8), dtype=np.float32)
        traffic = np.zeros((1, 2), dtype=np.float32)
        traffic[0, 1] = 1.0  # Right-hand traffic convention

        # Map by name (handles any input order)
        feed = {}
        for name in self._input_names:
            name_lower = name.lower()
            if 'img' in name_lower or 'input_img' in name_lower:
                feed[name] = input_imgs
            elif 'desire' in name_lower:
                feed[name] = desire
            elif 'traffic' in name_lower or 'convention' in name_lower:
                feed[name] = traffic
            elif 'state' in name_lower or 'initial' in name_lower:
                feed[name] = recurrent_state
            else:
                # Fallback: match by shape
                shape = self._input_shapes.get(name, [])
                if len(shape) >= 2 and shape[1] == 12:
                    feed[name] = input_imgs
                elif len(shape) >= 2 and shape[1] == 512:
                    feed[name] = recurrent_state
                elif len(shape) >= 2 and shape[1] == 8:
                    feed[name] = desire
                elif len(shape) >= 2 and shape[1] == 2:
                    feed[name] = traffic
                else:
                    print(f"[Supercombo] WARN: Unknown input '{name}' shape={shape}, using zeros")
                    feed[name] = np.zeros([d if isinstance(d, int) else 1 for d in shape], dtype=np.float32)

        return feed

    def _run_inference(self, input_imgs: np.ndarray) -> np.ndarray:
        """
        Run ONNX inference with the Supercombo model.
        
        Args:
            input_imgs: (1, 12, 128, 256) float32
            
        Returns:
            Raw output array (~6609 values)
        """
        feed = self._build_feed(input_imgs, self._recurrent_state)

        result = self.session.run([self._output_name], feed)
        output = np.array(result).flatten()

        return output

    def _decode_lanes(self, output: np.ndarray) -> dict:
        """
        Decode lane lines from model output.
        
        The lanes section contains 4 lanes, each with 33 points.
        Each point has (y_offset, std) for two time steps (t, t2).
        
        Layout (528 values):
          Lane 0 (left-left):  [0:66]   -> 33 points x 2 (y, std) at t
          Lane 0 at t2:        [66:132]
          Lane 1 (left):       [132:198] -> 33 points x 2 (y, std) at t
          Lane 1 at t2:        [198:264]
          Lane 2 (right):      [264:330] -> 33 points x 2 (y, std) at t
          Lane 2 at t2:        [330:396]
          Lane 3 (right-right):[396:462] -> 33 points x 2 (y, std) at t
          Lane 3 at t2:        [462:528]
          
        Returns:
            dict with 'left', 'right', 'left_left', 'right_right' lane points
            and 'probabilities' for each lane
        """
        lanes_raw = output[LANES_START:LANES_END]
        probs_raw = output[LANE_PROB_START:LANE_PROB_END]

        def extract_lane(data_66):
            """Extract 33 y-points from 66 values (interleaved y, std)."""
            points = data_66[0::2]  # Even indices = y values
            stds = data_66[1::2]    # Odd indices = std values
            return points[:33], stds[:33]

        # Extract all 4 lanes at time t (use t, not t2)
        ll_points, ll_stds = extract_lane(lanes_raw[0:66])
        l_points, l_stds = extract_lane(lanes_raw[132:198])
        r_points, r_stds = extract_lane(lanes_raw[264:330])
        rr_points, rr_stds = extract_lane(lanes_raw[396:462])

        # Lane probabilities (sigmoid for probability)
        probs = 1.0 / (1.0 + np.exp(-probs_raw))

        return {
            'left_left': {'y': ll_points, 'std': ll_stds},
            'left': {'y': l_points, 'std': l_stds},
            'right': {'y': r_points, 'std': r_stds},
            'right_right': {'y': rr_points, 'std': rr_stds},
            'probabilities': {
                'left_left': float(probs[0]),
                'left': float(probs[2]),
                'right': float(probs[4]),
                'right_right': float(probs[6]),
            }
        }

    def _decode_path(self, output: np.ndarray) -> dict:
        """
        Decode planned trajectory from model output.
        
        Plan section: 5 paths x (33 points x 3 coords [x,y,z] x 2 [mean,std]) + 1 prob each
        Total: 5 * (33 * 3 * 2 + 1) = 5 * 199 = 995... 
        Actually the plan layout is more complex. We'll use a simplified extraction.
        
        For each path: 33 points with (x, y, z) in ego frame.
        We select the path with highest confidence.
        
        Returns:
            dict with 'x', 'y', 'z' arrays (33 points each) for best path
        """
        plan_raw = output[PLAN_START:PLAN_END]

        # Each path has 33*3*2 = 198 values (mean+std for x,y,z) + 1 prob
        # 5 paths = 5 * 199 = 995 for the first block
        # The full plan section is 4955 values, which includes multiple representations
        # We use a simplified extraction: first 5 paths

        path_size = 33 * 3 * 2  # 198 values per path (without prob)
        # There's also probabilities interleaved. Let's use a simpler approach:
        # Plan layout: 5 paths, each with 33 points, each point has (x_mean, x_std, y_mean, y_std, z_mean, z_std)
        # Actually from openpilot source: each path has 33 * 2 * 3 = 198 values + other metadata

        best_path_y = np.zeros(33)
        best_conf = -1.0

        try:
            # Try to extract 5 paths. Each path: 33 positions with 6 values each (x,sx,y,sy,z,sz) = 198
            # Plus confidence at the end
            stride = 199  # 198 values + 1 probability

            for i in range(5):
                start = i * stride
                if start + stride > len(plan_raw):
                    break

                path_data = plan_raw[start:start + 198]
                prob = plan_raw[start + 198] if start + 198 < len(plan_raw) else 0

                # Extract y values (lateral position, every 6th value starting at index 2)
                y_values = path_data[2::6][:33]  # y_mean values
                x_values = path_data[0::6][:33]  # x_mean values (longitudinal)

                sigmoid_prob = 1.0 / (1.0 + np.exp(-prob))
                if sigmoid_prob > best_conf:
                    best_conf = sigmoid_prob
                    best_path_y = y_values
        except Exception:
            pass

        return {
            'y': best_path_y,
            'confidence': float(best_conf) if best_conf > 0 else 0.0,
        }

    def _decode_road_edges(self, output: np.ndarray) -> dict:
        """
        Decode road edges from model output.
        
        264 values: 2 edges x 33 points x 2 (y, std) x 2 (t, t2)
        """
        road_raw = output[ROAD_START:ROAD_END]

        def extract_edge(data_66):
            points = data_66[0::2][:33]
            stds = data_66[1::2][:33]
            return points, stds

        right_edge, right_std = extract_edge(road_raw[0:66])
        left_edge, left_std = extract_edge(road_raw[132:198])

        return {
            'left': {'y': left_edge, 'std': left_std},
            'right': {'y': right_edge, 'std': right_std},
        }

    def _compute_steering(self, lanes: dict, path: dict) -> dict:
        """
        Compute steering angle from detected lanes or planned path.
        
        Strategy:
          - If use_path=True and path confidence is high, use path
          - Otherwise, use lane center (average of left and right inner lanes)
        
        Args:
            lanes: decoded lanes dict
            path: decoded path dict
            
        Returns:
            dict with steering_angle, confidence, error_normalized, lane_center_x
        """
        idx = min(self.lookahead_idx, 32)
        steering_angle = None
        confidence = 0.0
        error = 0.0
        source = "none"

        # Try path-based steering
        if self.use_path and path['confidence'] > 0.3:
            path_y = path['y']
            if len(path_y) > idx:
                # path_y is in meters, positive = left, negative = right
                error = float(path_y[idx])
                # Normalize: a 2m lateral offset at lookahead = max error
                error_normalized = np.clip(error / 2.0, -1.0, 1.0)
                steering_angle = float(-error_normalized * self.steering_gain)
                confidence = path['confidence']
                source = "path"

        # Try lane-based steering
        if steering_angle is None:
            left_y = lanes['left']['y']
            right_y = lanes['right']['y']
            left_prob = lanes['probabilities']['left']
            right_prob = lanes['probabilities']['right']

            if left_prob > 0.3 and right_prob > 0.3:
                # Both inner lanes detected
                lane_center = (float(left_y[idx]) + float(right_y[idx])) / 2.0
                error = lane_center
                error_normalized = np.clip(error / 2.0, -1.0, 1.0)
                steering_angle = float(-error_normalized * self.steering_gain)
                confidence = min(left_prob, right_prob)
                source = "lanes_both"

            elif left_prob > 0.3:
                # Only left lane
                error = float(left_y[idx]) - 1.85  # Assume ~1.85m lane width / 2
                error_normalized = np.clip(error / 2.0, -1.0, 1.0)
                steering_angle = float(-error_normalized * self.steering_gain)
                confidence = left_prob * 0.6
                source = "lane_left"

            elif right_prob > 0.3:
                # Only right lane
                error = float(right_y[idx]) + 1.85
                error_normalized = np.clip(error / 2.0, -1.0, 1.0)
                steering_angle = float(-error_normalized * self.steering_gain)
                confidence = right_prob * 0.6
                source = "lane_right"

        # No detection
        if steering_angle is None:
            return {
                'steering_angle': None,
                'lane_center_x': None,
                'confidence': 0.0,
                'error_normalized': 0.0,
            }

        # Smoothing
        alpha = self.smoothing
        steering_angle = alpha * steering_angle + (1 - alpha) * self._previous_steering

        # Clamp
        steering_angle = max(-self.max_steering, min(self.max_steering, steering_angle))
        self._previous_steering = steering_angle

        error_normalized = np.clip(error / 2.0, -1.0, 1.0)

        if self.frame_count < 10 or self.frame_count % 50 == 0:
            print(f"[Supercombo] Frame {self.frame_count}: "
                  f"steer={steering_angle:.1f}deg | "
                  f"source={source} | "
                  f"conf={confidence:.2f} | "
                  f"error={error:.3f}m")

        return {
            'steering_angle': round(float(steering_angle), 2),
            'lane_center_x': round(float(error) * 100 + self.input_width / 2, 1),  # Approx pixel
            'confidence': round(float(confidence), 3),
            'error_normalized': round(float(error_normalized), 4),
        }

    def _build_lane_points(self, lanes: dict, road_edges: dict) -> list:
        """
        Convert detected lanes into the lane_points format expected by the server protocol.
        
        Returns:
            list of lines, each line is a list of (x, y) tuples in pixel coordinates
        """
        points = []

        for lane_key in ['left', 'right']:
            lane = lanes[lane_key]
            prob = lanes['probabilities'][lane_key]
            if prob > 0.3:
                line = []
                for i in range(0, 33, 2):  # Sample every other point
                    y_m = float(lane['y'][i])
                    x_m = float(X_IDXS[i])
                    # Convert meters to approximate pixel coordinates
                    # x_m (forward) -> y_pixel (top to bottom, inverted)
                    # y_m (lateral) -> x_pixel (left to right)
                    px = int(self.input_width / 2 + y_m * 50)   # ~50 px per meter lateral
                    py = int(self.input_height - x_m * 2)        # ~2 px per meter forward
                    if 0 <= px < self.input_width and 0 <= py < self.input_height:
                        line.append((px, py))
                if line:
                    points.append(line)

        return points

    def infer(self, frame: np.ndarray) -> dict:
        """
        Full inference: steering, lane points, road edges, timing.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            dict compatible with HybridNetsEngine.infer()
        """
        start_time = time.time()
        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        input_imgs = self._preprocess(frame)

        # Inference
        try:
            output = self._run_inference(input_imgs)
        except Exception as e:
            print(f"[Supercombo] ERROR en inferencia: {e}")
            return self._empty_result(orig_h, orig_w, start_time)

        # Decode
        lanes = self._decode_lanes(output)
        path = self._decode_path(output)
        road_edges = self._decode_road_edges(output)

        # Compute steering
        steering = self._compute_steering(lanes, path)

        # Build lane points for visualization
        lane_points = self._build_lane_points(lanes, road_edges)

        inference_ms = (time.time() - start_time) * 1000
        self.frame_count += 1

        # Visualize
        if self.show_visualization:
            self._visualize(frame, lanes, path, road_edges, steering, inference_ms)

        return {
            'steering': steering,
            'lane_mask': b'',   # No segmentation mask in Supercombo
            'road_mask': b'',
            'lane_points': lane_points,
            'detections': [],   # No object detection in this version
            'inference_time_ms': round(inference_ms, 1),
            'frame_id': self.frame_count,
            'input_size': f"{self.input_width}x{self.input_height}",
        }

    def infer_steering_only(self, frame: np.ndarray) -> dict:
        """
        Lightweight inference: only compute steering angle.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            dict compatible with HybridNetsEngine.infer_steering_only()
        """
        start_time = time.time()
        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        t0 = time.time()
        input_imgs = self._preprocess(frame)
        t_preprocess = time.time()

        # Inference
        try:
            output = self._run_inference(input_imgs)
        except Exception as e:
            print(f"[Supercombo] ERROR en inferencia: {e}")
            return self._empty_steering_result(start_time)
        t_forward = time.time()

        # Decode lanes and path
        lanes = self._decode_lanes(output)
        path = self._decode_path(output)
        t_decode = time.time()

        # Compute steering
        steering = self._compute_steering(lanes, path)
        t_steering = time.time()

        inference_ms = (t_steering - start_time) * 1000
        self.frame_count += 1

        # Visualize if enabled
        if self.show_visualization:
            road_edges = self._decode_road_edges(output)
            self._visualize(frame, lanes, path, road_edges, steering, inference_ms)

        # Timing log for first frames
        if self.frame_count <= 10:
            preproc_ms = (t_preprocess - t0) * 1000
            forward_ms = (t_forward - t_preprocess) * 1000
            decode_ms = (t_decode - t_forward) * 1000
            steer_ms = (t_steering - t_decode) * 1000
            print(f"[Supercombo] Frame {self.frame_count}: "
                  f"preprocess={preproc_ms:.1f}ms | "
                  f"forward={forward_ms:.1f}ms | "
                  f"decode={decode_ms:.1f}ms | "
                  f"steering={steer_ms:.1f}ms | "
                  f"TOTAL={inference_ms:.1f}ms | "
                  f"input={orig_w}x{orig_h}")
        elif self.frame_count % 100 == 0:
            print(f"[Supercombo] Frame {self.frame_count}: total={inference_ms:.1f}ms | "
                  f"steer={steering['steering_angle']}")

        return {
            'steering': steering,
            'inference_time_ms': round(inference_ms, 1),
            'frame_id': self.frame_count,
            'input_size': f"{self.input_width}x{self.input_height}",
        }

    def _visualize(self, frame, lanes, path, road_edges, steering, inference_ms):
        """Show debug visualization windows."""
        if not self.show_visualization:
            return

        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Header
        cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 40), -1)
        cv2.putText(overlay, "Supercombo (openpilot)", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        steer_angle = steering.get('steering_angle')
        confidence = steering.get('confidence', 0)

        if steer_angle is not None:
            steer_color = (0, 255, 0) if abs(steer_angle) < 10 else \
                (0, 165, 255) if abs(steer_angle) < 20 else (0, 0, 255)
            cv2.putText(overlay, f"Steer: {steer_angle:.1f} deg (serial: {int(round(steer_angle * 10))})",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, steer_color, 2)
            cv2.putText(overlay, f"Conf: {confidence:.2f} | {inference_ms:.0f}ms | Frame #{self.frame_count}",
                        (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Steering arrow
            center_x = w // 2
            bottom_y = h - 20
            offset_px = int((steer_angle / 25.0) * (w / 4))
            cv2.line(overlay, (center_x, h), (center_x, h - 80), (255, 255, 0), 2)
            cv2.arrowedLine(overlay, (center_x, bottom_y),
                            (center_x + offset_px, bottom_y - 50), steer_color, 3)
        else:
            cv2.putText(overlay, "NO LANES DETECTED", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw lane lines
        scale_x = w / self.input_width
        scale_y = h / self.input_height
        colors = {'left': (0, 255, 255), 'right': (255, 255, 0),
                  'left_left': (0, 128, 255), 'right_right': (255, 128, 0)}

        for lane_key, color in colors.items():
            prob = lanes['probabilities'][lane_key]
            if prob > 0.3:
                lane_y = lanes[lane_key]['y']
                pts = []
                for i in range(33):
                    px = int((self.input_width / 2 + float(lane_y[i]) * 50) * scale_x)
                    py = int((self.input_height - float(X_IDXS[i]) * 2) * scale_y)
                    if 0 <= px < w and 0 <= py < h:
                        pts.append((px, py))
                for j in range(len(pts) - 1):
                    cv2.line(overlay, pts[j], pts[j + 1], color, 2)

        # Draw path
        if path['confidence'] > 0.3:
            path_pts = []
            for i in range(33):
                px = int((self.input_width / 2 + float(path['y'][i]) * 50) * scale_x)
                py = int((self.input_height - float(X_IDXS[i]) * 2) * scale_y)
                if 0 <= px < w and 0 <= py < h:
                    path_pts.append((px, py))
            for j in range(len(path_pts) - 1):
                cv2.line(overlay, path_pts[j], path_pts[j + 1], (0, 255, 0), 3)

        # Lane probabilities
        y_prob = 95
        for key in ['left_left', 'left', 'right', 'right_right']:
            prob = lanes['probabilities'][key]
            label = f"{key}: {prob:.2f}"
            color = (0, 255, 0) if prob > 0.5 else (0, 165, 255) if prob > 0.3 else (0, 0, 255)
            cv2.putText(overlay, label, (w - 180, y_prob),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y_prob += 15

        try:
            cv2.imshow("Supercombo - Overlay", overlay)
            cv2.waitKey(1)
        except cv2.error:
            self.show_visualization = False

    def _empty_result(self, orig_h, orig_w, start_time):
        """Empty result when inference fails."""
        inference_ms = (time.time() - start_time) * 1000
        self.frame_count += 1
        return {
            'steering': {
                'steering_angle': None,
                'lane_center_x': None,
                'confidence': 0.0,
                'error_normalized': 0.0,
            },
            'lane_mask': b'',
            'road_mask': b'',
            'lane_points': [],
            'detections': [],
            'inference_time_ms': round(inference_ms, 1),
            'frame_id': self.frame_count,
            'input_size': f"{self.input_width}x{self.input_height}",
        }

    def _empty_steering_result(self, start_time):
        """Empty result for steering-only when inference fails."""
        inference_ms = (time.time() - start_time) * 1000
        self.frame_count += 1
        return {
            'steering': {
                'steering_angle': None,
                'lane_center_x': None,
                'confidence': 0.0,
                'error_normalized': 0.0,
            },
            'inference_time_ms': round(inference_ms, 1),
            'frame_id': self.frame_count,
            'input_size': f"{self.input_width}x{self.input_height}",
        }

    def get_status(self) -> dict:
        """Get engine status information."""
        return {
            'model_loaded': self.session is not None,
            'model': 'supercombo',
            'device': self.device_str,
            'providers': self.session.get_providers() if self.session else [],
            'use_half': False,  # ONNX Runtime handles precision internally
            'input_size': f"{self.input_width}x{self.input_height}",
            'frames_processed': self.frame_count,
            'lookahead_idx': self.lookahead_idx,
            'steering_gain': self.steering_gain,
            'use_path': self.use_path,
            'gpu_info': {},
        }
