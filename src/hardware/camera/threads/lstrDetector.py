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
        Calculate the center of the lane at a given y position.
        
        Args:
            image_width: Width of the image
            y_position_ratio: Vertical position ratio (0=top, 1=bottom)
            
        Returns:
            float: X coordinate of lane center, or None if not enough lanes detected
        """
        if len(self.lanes) < 2:
            # If only one lane detected, estimate center
            if len(self.lanes) == 1:
                lane_x = np.mean(self.lanes[0][0])
                # Assume lane is on left or right based on position
                if lane_x < image_width / 2:
                    return lane_x + image_width * 0.25
                else:
                    return lane_x - image_width * 0.25
            return None
        
        # Find leftmost and rightmost lanes
        lane_centers_x = [np.mean(lane[0]) for lane in self.lanes]
        left_idx = np.argmin(lane_centers_x)
        right_idx = np.argmax(lane_centers_x)
        
        if left_idx == right_idx:
            return None
        
        # Calculate center at the specified y position
        y_target = self.img_height * y_position_ratio
        
        left_lane = self.lanes[left_idx]
        right_lane = self.lanes[right_idx]
        
        # Find closest points to target y
        left_idx_y = np.argmin(np.abs(left_lane[1] - y_target))
        right_idx_y = np.argmin(np.abs(right_lane[1] - y_target))
        
        left_x = left_lane[0][left_idx_y]
        right_x = right_lane[0][right_idx_y]
        
        return (left_x + right_x) / 2
    
    def draw_lanes(self, image, draw_center=True):
        """
        Draw detected lanes on the image.
        
        Args:
            image: BGR image to draw on
            draw_center: Whether to draw the lane center point
            
        Returns:
            numpy array: Image with drawn lanes
        """
        if not self.lanes:
            return image
        
        visualization_img = image.copy()
        
        # Try to draw lane area if we have left and right lanes
        if len(self.lanes) >= 2:
            # Find lanes closest to center (likely the ego lanes)
            lane_centers = [np.mean(lane[0]) for lane in self.lanes]
            img_center = image.shape[1] / 2
            
            # Find left and right lanes relative to center
            left_lanes = [(i, c) for i, c in enumerate(lane_centers) if c < img_center]
            right_lanes = [(i, c) for i, c in enumerate(lane_centers) if c >= img_center]
            
            if left_lanes and right_lanes:
                # Get the rightmost left lane and leftmost right lane
                left_idx = max(left_lanes, key=lambda x: x[1])[0]
                right_idx = min(right_lanes, key=lambda x: x[1])[0]
                
                left_lane = self.lanes[left_idx]
                right_lane = self.lanes[right_idx]
                
                # Create polygon for lane area
                try:
                    points = np.vstack((
                        left_lane.T,
                        np.flipud(right_lane.T)
                    ))
                    lane_segment_img = visualization_img.copy()
                    cv2.fillConvexPoly(lane_segment_img, points, color=(0, 191, 255))
                    visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)
                except:
                    pass
        
        # Draw lane points
        for lane_num, lane_points in enumerate(self.lanes):
            color = LANE_COLORS[lane_num % len(LANE_COLORS)]
            for i in range(lane_points.shape[1]):
                x, y = lane_points[0, i], lane_points[1, i]
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(visualization_img, (x, y), 3, color, -1)
        
        # Draw lane center
        if draw_center:
            center = self.get_lane_center(image.shape[1])
            if center is not None:
                center_y = int(image.shape[0] * 0.7)
                cv2.circle(visualization_img, (int(center), center_y), 10, (255, 0, 255), -1)
        
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
