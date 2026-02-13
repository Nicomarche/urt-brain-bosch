# LSTR Lane Detection Module
# Based on: https://github.com/ibaiGorordo/ONNX-LSTR-Lane-Detection
# Paper: End-to-end Lane Shape Prediction with Transformers (WACV 2021)

import os
import cv2
import numpy as np
from enum import Enum

# Try to import onnxruntime, but don't fail if not available
try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("\033[1;97m[ LSTR ] :\033[0m \033[1;93mWARNING\033[0m - onnxruntime not installed. LSTR mode disabled.")


class LSTRModelType(Enum):
    """Available LSTR model sizes."""
    LSTR_180X320 = "lstr_180x320"      # Fastest, lowest accuracy
    LSTR_240X320 = "lstr_240x320"      # Good balance for RPi
    LSTR_360X640 = "lstr_360x640"      # Medium
    LSTR_480X640 = "lstr_480x640"      # Higher accuracy
    LSTR_720X1280 = "lstr_720x1280"    # Highest accuracy, slowest


# Lane colors for visualization (8 possible lanes)
LANE_COLORS = [
    (68, 65, 249), (44, 114, 243), (30, 150, 248), (74, 132, 249),
    (79, 199, 249), (109, 190, 144), (142, 144, 77), (161, 125, 39)
]

# Log space for smooth lane point interpolation
LOG_SPACE = np.logspace(0, 2, 50, base=1/10, endpoint=True)


class LSTRDetector:
    """
    LSTR (Lane Shape Prediction with Transformers) detector using ONNX Runtime.
    
    This is a deep learning-based lane detector that is more robust to lighting
    changes than traditional computer vision approaches. It predicts lane shape
    parameters directly using a transformer architecture.
    
    Attributes:
        model_path: Path to the ONNX model file
        is_available: Whether ONNX runtime is available and model is loaded
    """
    
    def __init__(self, model_path=None, model_type=LSTRModelType.LSTR_180X320):
        """
        Initialize LSTR detector.
        
        Args:
            model_path: Path to .onnx model file. If None, uses default location.
            model_type: LSTRModelType enum for model resolution
        """
        self.model_type = model_type
        self.is_available = False
        self.session = None
        self.lanes = []
        self.good_lanes = []
        
        # Image dimensions (will be set when processing)
        self.img_height = 0
        self.img_width = 0
        
        if not ONNX_AVAILABLE:
            print("\033[1;97m[ LSTR ] :\033[0m \033[1;91mERROR\033[0m - onnxruntime not available")
            return
        
        # Determine model path
        if model_path is None:
            # Default path relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, '..', '..', '..', '..', 'models', 'lstr', f'{model_type.value}.onnx')
            model_path = os.path.abspath(model_path)
        
        self.model_path = model_path
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"\033[1;97m[ LSTR ] :\033[0m \033[1;93mWARNING\033[0m - Model not found at {model_path}")
            print(f"\033[1;97m[ LSTR ] :\033[0m \033[1;96mINFO\033[0m - Download from: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/140_LSTR")
            return
        
        try:
            self._initialize_model()
            self.is_available = True
            print(f"\033[1;97m[ LSTR ] :\033[0m \033[1;92mINFO\033[0m - Model loaded: {model_type.value}")
        except Exception as e:
            print(f"\033[1;97m[ LSTR ] :\033[0m \033[1;91mERROR\033[0m - Failed to load model: {e}")
    
    def _initialize_model(self):
        """Initialize ONNX Runtime session."""
        # Use CPU execution provider (for Raspberry Pi compatibility)
        providers = ['CPUExecutionProvider']
        
        # Try to use OpenVINO or ARM NN if available (faster on some devices)
        available_providers = onnxruntime.get_available_providers()
        if 'OpenVINOExecutionProvider' in available_providers:
            providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        
        self.session = onnxruntime.InferenceSession(self.model_path, providers=providers)
        
        # Get input details
        self.rgb_input_name = self.session.get_inputs()[0].name
        self.mask_input_name = self.session.get_inputs()[1].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        # Get output names
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"\033[1;97m[ LSTR ] :\033[0m \033[1;96mINFO\033[0m - Input size: {self.input_width}x{self.input_height}")
    
    def detect_lanes(self, image):
        """
        Detect lanes in an image.
        
        Args:
            image: BGR image (numpy array)
            
        Returns:
            tuple: (lanes, good_lane_ids) where lanes is a list of lane points
                   and good_lane_ids are the detected lane indices
        """
        if not self.is_available:
            return [], []
        
        try:
            # Prepare inputs
            input_tensor, mask_tensor = self._prepare_inputs(image)
            
            # Run inference
            outputs = self.session.run(
                self.output_names,
                {self.rgb_input_name: input_tensor, self.mask_input_name: mask_tensor}
            )
            
            # Process outputs
            lanes, good_lanes = self._process_output(outputs)
            
            return lanes, good_lanes
            
        except Exception as e:
            print(f"\033[1;97m[ LSTR ] :\033[0m \033[1;91mERROR\033[0m - Detection failed: {e}")
            return [], []
    
    def _prepare_inputs(self, img):
        """Prepare image for model inference."""
        self.img_height, self.img_width = img.shape[:2]
        
        # Resize to model input size
        img_resized = cv2.resize(img, (self.input_width, self.input_height))
        
        # Normalize with ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_normalized = ((img_resized / 255.0 - mean) / std)
        
        # Convert to NCHW format
        img_transposed = img_normalized.transpose(2, 0, 1)
        input_tensor = img_transposed[np.newaxis, :, :, :].astype(np.float32)
        
        # Create mask tensor (required by LSTR)
        mask_tensor = np.zeros((1, 1, self.input_height, self.input_width), dtype=np.float32)
        
        return input_tensor, mask_tensor
    
    @staticmethod
    def _softmax(x):
        """Compute softmax values."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1).T
    
    def _process_output(self, outputs):
        """Process model outputs to extract lane points."""
        pred_logits = outputs[0]
        pred_curves = outputs[1]
        
        # Filter good lanes based on probability
        prob = self._softmax(pred_logits)
        good_detections = np.where(np.argmax(prob, axis=-1) == 1)
        pred_logits = pred_logits[good_detections]
        pred_curves = pred_curves[good_detections]
        
        lanes = []
        for lane_data in pred_curves:
            bounds = lane_data[:2]
            k_2, f_2, m_2, n_1, b_2, b_3 = lane_data[2:]
            
            # Calculate lane points using the polynomial parameters
            y_norm = bounds[0] + LOG_SPACE * (bounds[1] - bounds[0])
            x_norm = (k_2 / (y_norm - f_2) ** 2 + m_2 / (y_norm - f_2) + n_1 + b_2 * y_norm - b_3)
            
            # Scale to image dimensions
            lane_points = np.vstack((
                x_norm * self.img_width,
                y_norm * self.img_height
            )).astype(int)
            
            lanes.append(lane_points)
        
        self.lanes = lanes
        self.good_lanes = good_detections[1] if len(good_detections) > 1 else np.array([])
        
        return lanes, self.good_lanes
    
    def get_lane_center(self, image_width, y_position_ratio=0.7):
        """
        Calculate the center of the CURRENT lane at a given y position.
        
        When multiple lanes are visible, this method identifies which lane the vehicle
        is currently in by finding the two closest lines to the image center (one on 
        each side). This ensures proper lane keeping even when multiple lanes are visible.
        
        Args:
            image_width: Width of the image
            y_position_ratio: Vertical position ratio (0=top, 1=bottom)
            
        Returns:
            float: X coordinate of lane center, or None if not enough lanes detected
        """
        if len(self.lanes) < 1:
            return None
            
        # Calculate the x position of each lane at the target y position
        y_target = self.img_height * y_position_ratio
        image_center = image_width / 2
        
        lane_x_positions = []
        for i, lane in enumerate(self.lanes):
            # Find the x position at the target y
            idx_y = np.argmin(np.abs(lane[1] - y_target))
            x_at_target = lane[0][idx_y]
            lane_x_positions.append((i, x_at_target))
        
        if len(self.lanes) == 1:
            # If only one lane detected, estimate center
            lane_x = lane_x_positions[0][1]
            if lane_x < image_center:
                return lane_x + image_width * 0.25
            else:
                return lane_x - image_width * 0.25
        
        # Separate lanes into left (to the left of image center) and right (to the right)
        left_lanes = [(idx, x) for idx, x in lane_x_positions if x < image_center]
        right_lanes = [(idx, x) for idx, x in lane_x_positions if x >= image_center]
        
        # Find the CLOSEST lane on each side to the image center
        # This identifies the current lane we're driving in
        left_lane_idx = None
        right_lane_idx = None
        left_x = None
        right_x = None
        
        if left_lanes:
            # Get the rightmost lane on the left side (closest to center)
            closest_left = max(left_lanes, key=lambda item: item[1])
            left_lane_idx = closest_left[0]
            left_x = closest_left[1]
        
        if right_lanes:
            # Get the leftmost lane on the right side (closest to center)
            closest_right = min(right_lanes, key=lambda item: item[1])
            right_lane_idx = closest_right[0]
            right_x = closest_right[1]
        
        # Calculate lane center based on what we found
        if left_x is not None and right_x is not None:
            # Both sides detected - this is the best case
            return (left_x + right_x) / 2
        elif left_x is not None:
            # Only left lane visible - estimate right side
            return left_x + image_width * 0.25
        elif right_x is not None:
            # Only right lane visible - estimate left side
            return right_x - image_width * 0.25
        
        return None
    
    def estimate_path_curvature(self, image_width):
        """
        Estimate the curvature of the path ahead using all detected lane points.
        
        Uses the full 50-point lane data from LSTR to fit a 2nd-degree polynomial
        to the lane centerline, then computes curvature for feed-forward steering.
        
        The curvature formula (for x = a*y^2 + b*y + c):
            kappa = |2a| / (1 + (2a*y + b)^2)^(3/2)
        
        Args:
            image_width: Width of the image in pixels
            
        Returns:
            tuple: (curvature, turn_sign) where:
                - curvature: Magnitude of curvature (1/pixels). Higher = tighter curve.
                - turn_sign: +1 for right curve, -1 for left curve, 0 for straight.
                Returns (0.0, 0) if not enough data.
        """
        if len(self.lanes) < 1:
            return 0.0, 0
        
        image_center = image_width / 2
        
        # Identify left and right lane boundaries (same logic as get_lane_center)
        y_ref = self.img_height * 0.7
        
        lane_x_at_ref = []
        for i, lane in enumerate(self.lanes):
            idx_y = np.argmin(np.abs(lane[1] - y_ref))
            x_at_ref = lane[0][idx_y]
            lane_x_at_ref.append((i, x_at_ref))
        
        left_lanes = [(idx, x) for idx, x in lane_x_at_ref if x < image_center]
        right_lanes = [(idx, x) for idx, x in lane_x_at_ref if x >= image_center]
        
        left_lane = None
        right_lane = None
        
        if left_lanes:
            left_idx = max(left_lanes, key=lambda item: item[1])[0]
            left_lane = self.lanes[left_idx]
        
        if right_lanes:
            right_idx = min(right_lanes, key=lambda item: item[1])[0]
            right_lane = self.lanes[right_idx]
        
        # Compute centerline points from available lane data
        if left_lane is not None and right_lane is not None:
            # Best case: both lanes visible - compute true centerline
            center_y = left_lane[1].astype(float)
            center_x = np.zeros_like(center_y, dtype=float)
            for i, y_val in enumerate(center_y):
                left_x = float(left_lane[0][i])
                idx_right = np.argmin(np.abs(right_lane[1] - y_val))
                right_x = float(right_lane[0][idx_right])
                center_x[i] = (left_x + right_x) / 2.0
        elif left_lane is not None:
            center_y = left_lane[1].astype(float)
            center_x = left_lane[0].astype(float) + image_width * 0.25
        elif right_lane is not None:
            center_y = right_lane[1].astype(float)
            center_x = right_lane[0].astype(float) - image_width * 0.25
        else:
            return 0.0, 0
        
        # Filter valid points (within image bounds)
        valid = ((center_x > 0) & (center_x < image_width) & 
                 (center_y > 0) & (center_y < self.img_height))
        center_x = center_x[valid]
        center_y = center_y[valid]
        
        if len(center_x) < 5:
            return 0.0, 0
        
        # Fit 2nd degree polynomial: x = a*y^2 + b*y + c
        try:
            coeffs = np.polyfit(center_y, center_x, 2)
            a, b, c = coeffs
        except Exception:
            return 0.0, 0
        
        # Calculate curvature at evaluation point (middle of visible range)
        y_eval = np.mean(center_y)
        
        # kappa = |d²x/dy²| / (1 + (dx/dy)²)^(3/2)
        first_deriv = 2 * a * y_eval + b
        curvature = abs(2 * a) / (1 + first_deriv ** 2) ** 1.5
        
        # Determine turn direction from the path shape:
        # Compare centerline position at far (low y = top of image) vs near (high y = bottom)
        y_near = np.max(center_y)
        y_far = np.min(center_y)
        x_near = a * y_near**2 + b * y_near + c
        x_far = a * y_far**2 + b * y_far + c
        
        # If far point is to the right of near → right curve → positive steering
        # If far point is to the left → left curve → negative steering
        dx = x_far - x_near
        if abs(dx) < 3:  # Less than 3px difference = straight
            turn_sign = 0
        elif dx > 0:
            turn_sign = 1   # Right curve
        else:
            turn_sign = -1  # Left curve
        
        return curvature, turn_sign

    def draw_lanes(self, image, draw_center=True):
        """
        Draw detected lanes on the image, highlighting the current lane.
        
        The current lane (the one the vehicle is in) is identified by finding
        the two lines closest to the image center - one on each side.
        
        Args:
            image: BGR image to draw on
            draw_center: Whether to draw the lane center point
            
        Returns:
            numpy array: Image with drawn lanes
        """
        if not self.lanes:
            return image
        
        visualization_img = image.copy()
        img_center = image.shape[1] / 2
        img_height = image.shape[0]
        
        # Identify the current lane boundaries (closest to center on each side)
        current_left_idx = None
        current_right_idx = None
        
        if len(self.lanes) >= 1:
            # Calculate lane x positions at 70% of image height
            y_target = img_height * 0.7
            lane_x_positions = []
            for i, lane in enumerate(self.lanes):
                idx_y = np.argmin(np.abs(lane[1] - y_target))
                x_at_target = lane[0][idx_y]
                lane_x_positions.append((i, x_at_target))
            
            # Separate into left and right of center
            left_lanes = [(idx, x) for idx, x in lane_x_positions if x < img_center]
            right_lanes = [(idx, x) for idx, x in lane_x_positions if x >= img_center]
            
            if left_lanes:
                # Rightmost lane on the left side = current left boundary
                current_left_idx = max(left_lanes, key=lambda item: item[1])[0]
            
            if right_lanes:
                # Leftmost lane on the right side = current right boundary
                current_right_idx = min(right_lanes, key=lambda item: item[1])[0]
        
        # Draw the current lane area (highlighted)
        if current_left_idx is not None and current_right_idx is not None:
            left_lane = self.lanes[current_left_idx]
            right_lane = self.lanes[current_right_idx]
            
            try:
                points = np.vstack((
                    left_lane.T,
                    np.flipud(right_lane.T)
                ))
                lane_segment_img = visualization_img.copy()
                # Bright green for current lane
                cv2.fillConvexPoly(lane_segment_img, points, color=(0, 255, 100))
                visualization_img = cv2.addWeighted(visualization_img, 0.6, lane_segment_img, 0.4, 0)
            except:
                pass
        
        # Draw all lane points with different colors
        for lane_num, lane_points in enumerate(self.lanes):
            # Current lane boundaries get thicker, brighter markers
            if lane_num == current_left_idx or lane_num == current_right_idx:
                color = (0, 255, 255)  # Yellow for current lane boundaries
                radius = 5
                thickness = -1
            else:
                color = LANE_COLORS[lane_num % len(LANE_COLORS)]
                radius = 3
                thickness = -1
            
            for i in range(lane_points.shape[1]):
                x, y = lane_points[0, i], lane_points[1, i]
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(visualization_img, (x, y), radius, color, thickness)
        
        # Draw lane info text
        total_lanes = len(self.lanes)
        cv2.putText(visualization_img, f"Lanes: {total_lanes}", (10, img_height - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if total_lanes > 2:
            cv2.putText(visualization_img, "Multi-lane detected", (10, img_height - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(visualization_img, "Following current lane", (10, img_height - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
        
        # Draw lane center
        if draw_center:
            center = self.get_lane_center(image.shape[1])
            if center is not None:
                center_y = int(img_height * 0.7)
                # Draw center point
                cv2.circle(visualization_img, (int(center), center_y), 12, (255, 0, 255), -1)
                cv2.circle(visualization_img, (int(center), center_y), 14, (255, 255, 255), 2)
                # Draw center line
                cv2.line(visualization_img, (int(center), center_y - 30), 
                        (int(center), center_y + 30), (255, 0, 255), 2)
        
        return visualization_img


def download_lstr_model(model_type=LSTRModelType.LSTR_180X320, output_dir=None):
    """
    Download LSTR model from PINTO's model zoo.
    
    Args:
        model_type: LSTRModelType enum
        output_dir: Directory to save model (default: models/lstr/)
    """
    import urllib.request
    import zipfile
    import io
    
    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, '..', '..', '..', '..', 'models', 'lstr')
        output_dir = os.path.abspath(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = model_type.value
    output_path = os.path.join(output_dir, f'{model_name}.onnx')
    
    if os.path.exists(output_path):
        print(f"\033[1;97m[ LSTR ] :\033[0m \033[1;93mWARNING\033[0m - Model already exists: {output_path}")
        return output_path
    
    # PINTO model zoo URL
    # Note: You may need to update this URL based on the current model zoo structure
    base_url = "https://github.com/PINTO0309/PINTO_model_zoo/raw/main/140_LSTR"
    
    print(f"\033[1;97m[ LSTR ] :\033[0m \033[1;96mINFO\033[0m - Downloading {model_name}...")
    print(f"\033[1;97m[ LSTR ] :\033[0m \033[1;96mINFO\033[0m - This may take a while...")
    
    # Download instructions (actual download requires navigating PINTO's repo structure)
    print(f"\033[1;97m[ LSTR ] :\033[0m \033[1;93mNOTE\033[0m - Please download manually from:")
    print(f"  {base_url}")
    print(f"  Save to: {output_path}")
    
    return None


# Test the module
if __name__ == "__main__":
    print("Testing LSTR Detector...")
    
    # Create detector
    detector = LSTRDetector()
    
    if detector.is_available:
        # Test with a dummy image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        lanes, ids = detector.detect_lanes(test_img)
        print(f"Detected {len(lanes)} lanes")
    else:
        print("LSTR not available - model not found or onnxruntime not installed")
