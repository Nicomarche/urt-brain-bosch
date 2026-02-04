# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.a
# Reconstructed from bytecode

import cv2
import numpy as np
import base64
import time
from enum import Enum
from src.utils.messages.allMessages import serialCamera, SpeedMotor, SteerMotor, StateChange, LineFollowingConfig, LineFollowingDebug, LineFollowingStatus
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.templates.threadwithstop import ThreadWithStop
from src.statemachine.systemMode import SystemMode

# Import LSTR detector (AI-based lane detection)
try:
    from src.hardware.camera.threads.lstrDetector import LSTRDetector, LSTRModelType
    LSTR_AVAILABLE = True
except ImportError as e:
    LSTR_AVAILABLE = False
    print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;93mWARNING\033[0m - LSTR not available: {e}")


class DetectionMode(Enum):
    """Lane detection modes."""
    OPENCV = "opencv"       # Traditional OpenCV (HSV + Hough)
    LSTR = "lstr"          # AI-based LSTR model
    HYBRID = "hybrid"       # Fusion of OpenCV + LSTR (combines both for better accuracy)


class threadLineFollowing(ThreadWithStop):
    """Thread which handles line following using OpenCV.

Args:
    queuesList (dictionar of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
    logger (logging object): Made for debugging.
    debugger (bool): A flag for debugging.
"""

    def __init__(self, queuesList, logger, debugger, show_debug=False):
        super(threadLineFollowing, self).__init__(pause=0.01)
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        self.show_debug = show_debug

        # Speed parameters
        self.base_speed = 10
        self.max_speed = 10
        self.min_speed = 5

        # Steering parameters
        self.max_steering = 25
        self.dead_zone_ratio = 0.02  # 2% of frame width - ignore small errors (prevents oscillation)

        # PID parameters - error is normalized to [-1, 1] range
        self.kp = 35.0  # Maps normalized error to steering (0.5 error -> ~17.5 deg)
        self.kd = 5.0   # Derivative gain for damping
        self.previous_error = 0
        self.smoothing_factor = 0.5  # Higher = more reactive, Lower = more smooth
        self.lookahead = 0.4

        # ROI parameters
        self.roi_height_start = 0.55
        self.roi_height_end = 0.92
        self.roi_width_margin_top = 0.35
        self.roi_width_margin_bottom = 0.15

        # HSV thresholds for white
        self.white_h_min = 81
        self.white_h_max = 180
        self.white_s_min = 0
        self.white_s_max = 98
        self.white_v_min = 200
        self.white_v_max = 255

        # HSV thresholds for yellow
        self.yellow_h_min = 173
        self.yellow_h_max = 86
        self.yellow_s_min = 100
        self.yellow_s_max = 255
        self.yellow_v_min = 100
        self.yellow_v_max = 255

        # Image processing parameters
        self.blur_kernel = 5
        self.morph_kernel = 7
        self.canny_low = 100
        self.canny_high = 200
        self.dilate_kernel = 5
        self.use_dilation = True
        self.hough_threshold = 20
        self.hough_min_line_length = 15
        self.hough_max_line_gap = 200

        # Brightness/contrast
        self.brightness = 5
        self.contrast = 0.8

        # Adaptive lighting parameters
        self.use_clahe = True
        self.clahe_clip_limit = 2.0
        self.clahe_grid_size = 8
        self.use_adaptive_white = True
        self.adaptive_white_percentile = 92
        self.adaptive_white_min_threshold = 180
        self.use_gradient_fallback = True
        self.gradient_percentile = 85

        # Detection mode parameters
        self.detection_mode = DetectionMode.OPENCV.value  # "opencv", "lstr", or "hybrid"
        self.lstr_model_size = 0  # 0=180x320, 1=240x320, 2=360x640, 3=480x640, 4=720x1280
        self.lstr_detector = None
        self.lstr_fallback_threshold = 0.01  # Switch to LSTR if detection < 1% of pixels
        self.lstr_confidence_threshold = 0.5  # Minimum confidence for LSTR detection
        self._current_lstr_model_size = -1  # Track current loaded model
        
        # Hybrid fusion parameters
        self.hybrid_opencv_weight = 0.4   # Weight for OpenCV detection (0-1)
        self.hybrid_lstr_weight = 0.6     # Weight for LSTR detection (0-1) - AI is more robust
        self.hybrid_agreement_bonus = 1.2  # Multiply confidence when both agree
        
        self._init_lstr_detector()

        # Debug streaming parameters
        self.stream_debug_view = 0  # 0=off, 1-10=different views
        self.stream_debug_fps = 5  # Max FPS for debug stream (to save bandwidth)
        self.stream_debug_quality = 50  # JPEG quality (1-100)
        self.stream_debug_scale = 0.5  # Scale factor for debug images
        self._debug_frame_counter = 0
        self._debug_images = {}  # Store debug images for streaming
        self._last_stream_time = 0

        # Sliding window parameters
        self.nwindows = 12
        self.window_margin = 60
        self.minpix = 15

        # Polynomial fit history
        self.left_fit = None
        self.right_fit = None
        self.left_fit_history = []
        self.right_fit_history = []
        self.fit_history_size = 8

        # Perspective transform
        self.perspective_M = None
        self.perspective_M_inv = None
        self.perspective_initialized = False

        self._update_hsv_arrays()

        # Line following state
        self.is_line_following_active = False
        self.last_line_position = None
        self.frames_without_line = 0
        self.max_frames_without_line = 10
        self.last_seen_side = "both"
        self.single_line_correction = 1
        self.last_turn_direction = 0

        # Message handlers
        self.speedMotorSender = messageHandlerSender(self.queuesList, SpeedMotor)
        self.steerMotorSender = messageHandlerSender(self.queuesList, SteerMotor)
        self.serialCameraSubscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "lastOnly", True)
        self.stateChangeSubscriber = messageHandlerSubscriber(self.queuesList, StateChange, "lastOnly", True)
        self.configSubscriber = messageHandlerSubscriber(self.queuesList, LineFollowingConfig, "lastOnly", True)
        
        # Debug stream senders
        self.debugStreamSender = messageHandlerSender(self.queuesList, LineFollowingDebug)
        self.statusSender = messageHandlerSender(self.queuesList, LineFollowingStatus)

        print("\033[1;97m[ Line Following ] :\033[0m \033[1;92mINFO\033[0m - Line following thread initialized")
        print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;92mINFO\033[0m - Debug mode: {self.show_debug}")
        print("\033[1;97m[ Line Following ] :\033[0m \033[1;93mWARNING\033[0m - Windows will open when camera starts streaming")

    def _update_hsv_arrays(self):
        """Update NumPy arrays for HSV thresholds."""
        self.white_lower = np.array([self.white_h_min, self.white_s_min, self.white_v_min])
        self.white_upper = np.array([self.white_h_max, self.white_s_max, self.white_v_max])
        self.yellow_lower = np.array([self.yellow_h_min, self.yellow_s_min, self.yellow_v_min])
        self.yellow_upper = np.array([self.yellow_h_max, self.yellow_s_max, self.yellow_v_max])

    def _init_lstr_detector(self):
        """Initialize LSTR detector if available."""
        if not LSTR_AVAILABLE:
            print("\033[1;97m[ Line Following ] :\033[0m \033[1;93mWARNING\033[0m - LSTR not available, using OpenCV only")
            self.lstr_detector = None
            return
        
        # Map model size index to model type
        model_types = [
            LSTRModelType.LSTR_180X320,   # 0 - Fastest
            LSTRModelType.LSTR_240X320,   # 1 - Fast
            LSTRModelType.LSTR_360X640,   # 2 - Medium
            LSTRModelType.LSTR_480X640,   # 3 - Slow
            LSTRModelType.LSTR_720X1280,  # 4 - Slowest
        ]
        
        model_idx = max(0, min(4, int(self.lstr_model_size)))
        selected_model = model_types[model_idx]
        
        # Skip if already loaded the same model
        if self._current_lstr_model_size == model_idx and self.lstr_detector is not None:
            return
        
        try:
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;96mINFO\033[0m - Loading LSTR model: {selected_model.value}")
            self.lstr_detector = LSTRDetector(model_type=selected_model)
            if self.lstr_detector.is_available:
                self._current_lstr_model_size = model_idx
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;92mINFO\033[0m - LSTR detector initialized: {selected_model.value}")
            else:
                print("\033[1;97m[ Line Following ] :\033[0m \033[1;93mWARNING\033[0m - LSTR model not found, using OpenCV only")
                self.lstr_detector = None
        except Exception as e:
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mERROR\033[0m - Failed to init LSTR: {e}")
            self.lstr_detector = None

    def _reset_pid_state(self):
        """Reset PID state when changing modes to prevent corrupted values."""
        self.previous_error = 0
        if hasattr(self, 'last_steering'):
            del self.last_steering
        self.last_line_position = None
        self.frames_without_line = 0
        self.last_turn_direction = 0
        print("\033[1;97m[ Line Following ] :\033[0m \033[1;96mPID\033[0m - State reset")

    def _store_debug_image(self, name, image):
        """Store a debug image for potential streaming."""
        if image is not None:
            self._debug_images[name] = image.copy()

    def _show_control_panel(self, steering_angle, speed):
        """Show a control panel window with current status."""
        panel_width = 350
        panel_height = 280
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        # Background gradient
        for i in range(panel_height):
            color = int(20 + (i / panel_height) * 20)
            panel[i, :] = (color, color, color + 10)
        
        # Header
        cv2.rectangle(panel, (0, 0), (panel_width, 45), (40, 40, 60), -1)
        cv2.putText(panel, "CONTROL PANEL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Status indicator
        status = "ACTIVE" if self.is_line_following_active else "STANDBY"
        status_color = (0, 255, 0) if self.is_line_following_active else (0, 165, 255)
        cv2.circle(panel, (panel_width - 25, 25), 10, status_color, -1)
        
        # Detection Mode
        y_pos = 70
        cv2.putText(panel, "Mode:", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        mode_colors = {'opencv': (0, 200, 0), 'lstr': (255, 100, 0), 'hybrid': (255, 255, 0)}
        mode_color = mode_colors.get(self.detection_mode, (200, 200, 200))
        cv2.putText(panel, self.detection_mode.upper(), (80, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        # LSTR Status
        y_pos += 30
        cv2.putText(panel, "LSTR:", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        if self.lstr_detector is not None and self.lstr_detector.is_available:
            cv2.putText(panel, f"OK ({self.lstr_detector.model_type.value})", (80, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(panel, "Not Available", (80, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Steering
        y_pos += 35
        cv2.putText(panel, "Steering:", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        if steering_angle is not None:
            steer_text = f"{steering_angle:.1f} deg"
            steer_color = (0, 255, 255) if abs(steering_angle) < 10 else (0, 165, 255) if abs(steering_angle) < 20 else (0, 0, 255)
            cv2.putText(panel, steer_text, (100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, steer_color, 2)
            
            # Steering bar
            bar_center = panel_width // 2
            bar_y = y_pos + 15
            bar_width = 150
            cv2.rectangle(panel, (bar_center - bar_width//2, bar_y), (bar_center + bar_width//2, bar_y + 10), (60, 60, 60), -1)
            steer_pos = int(bar_center + (steering_angle / self.max_steering) * (bar_width // 2))
            cv2.rectangle(panel, (bar_center - 2, bar_y), (bar_center + 2, bar_y + 10), (100, 100, 100), -1)
            cv2.circle(panel, (steer_pos, bar_y + 5), 6, steer_color, -1)
        else:
            cv2.putText(panel, "---", (100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
        
        # Speed
        y_pos += 50
        cv2.putText(panel, "Speed:", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        if speed is not None:
            speed_text = f"{speed:.0f}"
            cv2.putText(panel, speed_text, (100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Speed bar
            bar_x = 150
            bar_width = 150
            bar_y = y_pos - 12
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (60, 60, 60), -1)
            speed_width = int((speed / self.max_speed) * bar_width)
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + speed_width, bar_y + 15), (0, 200, 0), -1)
        else:
            cv2.putText(panel, "---", (100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
        
        # Frames without line
        y_pos += 35
        cv2.putText(panel, "Lost frames:", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        lost_color = (0, 255, 0) if self.frames_without_line < 3 else (0, 165, 255) if self.frames_without_line < 7 else (0, 0, 255)
        cv2.putText(panel, f"{self.frames_without_line}/{self.max_frames_without_line}", (130, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, lost_color, 1)
        
        # PID Info
        y_pos += 30
        cv2.putText(panel, f"PID: Kp={self.kp:.2f} Kd={self.kd:.2f}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        
        cv2.imshow("Control Panel", panel)

    def _send_debug_stream(self, steering_angle, speed):
        """Send selected debug view to dashboard via websocket."""
        if self.stream_debug_view == 0:
            return  # Streaming disabled
        
        # Rate limit the streaming
        current_time = time.time()
        min_interval = 1.0 / max(1, self.stream_debug_fps)
        if current_time - self._last_stream_time < min_interval:
            return
        self._last_stream_time = current_time
        
        # Debug view mapping
        view_names = {
            1: 'final',           # Final result with lane overlay
            2: 'clahe',           # CLAHE normalized
            3: 'adjusted',        # Brightness/contrast adjusted
            4: 'hsv',             # HSV color space
            5: 'white_mask',      # White line mask
            6: 'yellow_mask',     # Yellow line mask
            7: 'combined_mask',   # Combined mask
            8: 'birds_eye',       # Bird's eye view
            9: 'sliding_window',  # Sliding window visualization
            10: 'lstr',           # LSTR AI result
        }
        
        view_name = view_names.get(self.stream_debug_view, 'final')
        
        # Get the selected image
        if view_name not in self._debug_images:
            return
        
        debug_img = self._debug_images[view_name]
        
        if debug_img is None:
            return
        
        try:
            # Scale down for bandwidth
            if self.stream_debug_scale < 1.0:
                new_width = int(debug_img.shape[1] * self.stream_debug_scale)
                new_height = int(debug_img.shape[0] * self.stream_debug_scale)
                debug_img = cv2.resize(debug_img, (new_width, new_height))
            
            # Add view name overlay
            cv2.putText(debug_img, f"View: {view_name}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Encode to JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.stream_debug_quality]
            _, encoded = cv2.imencode('.jpg', debug_img, encode_params)
            
            # Convert to base64
            b64_data = base64.b64encode(encoded).decode('utf-8')
            
            # Send via message handler
            self.debugStreamSender.send(b64_data)
            
            # Send status info
            status = {
                'steering': round(steering_angle, 2) if steering_angle is not None else None,
                'speed': round(speed, 2) if speed is not None else None,
                'mode': self.detection_mode,
                'view': view_name,
                'active': self.is_line_following_active,
                'lstr_available': self.lstr_detector is not None and self.lstr_detector.is_available,
            }
            self.statusSender.send(status)
            
        except Exception as e:
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mERROR\033[0m - Debug stream failed: {e}")

    def _apply_clahe(self, frame):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to normalize lighting.
        
        This technique equalizes the histogram locally, making the image more robust
        to varying lighting conditions (shadows, bright spots, etc.)
        """
        # Convert to LAB color space (L = Lightness, A/B = color)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to the L (lightness) channel only
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_grid_size, self.clahe_grid_size)
        )
        l_channel = clahe.apply(l_channel)
        
        # Merge channels back and convert to BGR
        lab = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _adaptive_white_detection(self, frame):
        """Detect white lines using adaptive thresholding based on image statistics.
        
        Instead of fixed V threshold, calculates threshold dynamically based on
        the percentile of brightness in the current frame. This adapts to
        different lighting conditions automatically.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate adaptive threshold based on percentile
        threshold = np.percentile(gray, self.adaptive_white_percentile)
        
        # Ensure minimum threshold to avoid detecting everything as white
        threshold = max(threshold, self.adaptive_white_min_threshold)
        
        # Create binary mask for white regions
        _, white_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        return white_mask, threshold

    def _gradient_based_detection(self, frame):
        """Detect lane lines using gradient (edge) information.
        
        This method is less sensitive to color/lighting changes because it
        focuses on detecting edges (where brightness changes sharply).
        Useful as a fallback when color detection fails.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Sobel filters to detect edges in X and Y directions
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize to 0-255 range
        gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude + 1e-6))
        
        # Calculate adaptive threshold based on percentile
        threshold = np.percentile(gradient_magnitude, self.gradient_percentile)
        
        # Create binary mask for strong gradients (edges)
        _, gradient_mask = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)
        
        return gradient_mask

    def _preprocess_frame(self, frame):
        """Apply all preprocessing steps to make detection robust to lighting changes.
        
        Returns:
            preprocessed: The preprocessed frame
            debug_info: Dictionary with intermediate results for debugging
        """
        debug_info = {}
        
        # Step 1: Apply CLAHE for lighting normalization
        if self.use_clahe:
            preprocessed = self._apply_clahe(frame)
            debug_info['clahe'] = preprocessed.copy()
        else:
            preprocessed = frame.copy()
        
        # Step 2: Apply brightness/contrast adjustment
        preprocessed = cv2.convertScaleAbs(preprocessed, alpha=self.contrast, beta=self.brightness)
        debug_info['adjusted'] = preprocessed.copy()
        
        return preprocessed, debug_info

    def _create_combined_mask(self, frame, preprocessed):
        """Create a combined mask using multiple detection methods.
        
        Combines:
        1. HSV-based color detection (white and yellow)
        2. Adaptive white detection (if enabled)
        3. Gradient-based detection as fallback (if enabled)
        
        Returns:
            combined_mask: Binary mask with detected lane pixels
            debug_info: Dictionary with intermediate masks for debugging
        """
        debug_info = {}
        
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)
        debug_info['hsv'] = hsv.copy()
        
        # Standard HSV-based white detection
        white_mask_hsv = cv2.inRange(hsv, self.white_lower, self.white_upper)
        debug_info['white_hsv'] = white_mask_hsv.copy()
        
        # Yellow detection (usually more stable across lighting)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        debug_info['yellow'] = yellow_mask.copy()
        
        # Start with HSV-based masks
        combined_mask = cv2.bitwise_or(white_mask_hsv, yellow_mask)
        
        # Add adaptive white detection if enabled
        if self.use_adaptive_white:
            adaptive_white_mask, adaptive_threshold = self._adaptive_white_detection(preprocessed)
            debug_info['adaptive_white'] = adaptive_white_mask.copy()
            debug_info['adaptive_threshold'] = adaptive_threshold
            
            # Combine with existing mask (OR operation)
            combined_mask = cv2.bitwise_or(combined_mask, adaptive_white_mask)
        
        # Add gradient fallback if enabled and main detection is weak
        if self.use_gradient_fallback:
            # Check if we have enough detected pixels
            white_pixel_count = np.sum(combined_mask > 0)
            total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
            detection_ratio = white_pixel_count / total_pixels
            
            # If detection is weak (less than 1% of image), use gradient fallback
            if detection_ratio < 0.01:
                gradient_mask = self._gradient_based_detection(preprocessed)
                debug_info['gradient'] = gradient_mask.copy()
                debug_info['gradient_used'] = True
                
                # Combine gradient mask with color masks
                combined_mask = cv2.bitwise_or(combined_mask, gradient_mask)
            else:
                debug_info['gradient_used'] = False
        
        debug_info['combined_raw'] = combined_mask.copy()
        
        return combined_mask, debug_info

    def _detect_with_lstr(self, frame):
        """
        Detect lanes using LSTR (AI-based detection).
        
        Args:
            frame: BGR image
            
        Returns:
            tuple: (steering_angle, speed, lane_center, debug_frame) or (None, None, None, None) if failed
        """
        if self.lstr_detector is None or not self.lstr_detector.is_available:
            return None, None, None, None
        
        try:
            height, width = frame.shape[:2]
            
            # Run LSTR detection
            start_time = time.time()
            lanes, lane_ids = self.lstr_detector.detect_lanes(frame)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Create AI analysis window (always, even if no lanes)
            ai_analysis_frame = frame.copy()
            
            # Draw AI analysis header
            cv2.rectangle(ai_analysis_frame, (0, 0), (width, 80), (20, 20, 40), -1)
            cv2.putText(ai_analysis_frame, "LSTR AI ANALYSIS", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(ai_analysis_frame, f"Model: {self.lstr_detector.model_type.value}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(ai_analysis_frame, f"Inference: {inference_time:.1f}ms", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(ai_analysis_frame, f"Lanes: {len(lanes)}", (width - 100, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if len(lanes) > 0 else (0, 0, 255), 2)
            
            if len(lanes) == 0:
                cv2.putText(ai_analysis_frame, "NO LANES DETECTED", (width//2 - 120, height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Show AI analysis window even when no detection
                if self.show_debug:
                    cv2.imshow("AI Analysis - LSTR", ai_analysis_frame)
                    cv2.waitKey(1)
                
                self._store_debug_image('lstr', ai_analysis_frame)
                return None, None, None, None
            
            # Get lane center
            lane_center = self.lstr_detector.get_lane_center(width, y_position_ratio=1.0 - self.lookahead)
            
            if lane_center is None:
                if self.show_debug:
                    cv2.imshow("AI Analysis - LSTR", ai_analysis_frame)
                    cv2.waitKey(1)
                return None, None, None, None
            
            # Calculate steering with NORMALIZED error
            frame_center = width / 2
            error = lane_center - frame_center
            error_normalized = error / frame_center  # Range [-1, 1]
            
            # PID control with dead zone for straight line stability
            abs_error_norm = abs(error_normalized)
            if abs_error_norm < self.dead_zone_ratio:
                # Very small error - no correction
                effective_error = 0
                derivative = 0
            else:
                # Outside dead zone
                effective_error = (abs_error_norm - self.dead_zone_ratio) * (1 if error_normalized > 0 else -1)
                derivative = self.kd * (error_normalized - self.previous_error)
            
            self.previous_error = error_normalized
            
            # Determine curve intensity based on error magnitude (same logic as OpenCV)
            # Higher error = sharper curve = need more steering
            # Lower thresholds to activate multiplier more easily
            if abs_error_norm > 0.15:
                curve_intensity = 3  # Sharp curve
            elif abs_error_norm > 0.08:
                curve_intensity = 2  # Medium curve
            elif abs_error_norm > 0.04:
                curve_intensity = 1  # Slight curve
            else:
                curve_intensity = 0  # Straight
            
            curve_multipliers = (40, 40, 40, 40)
            curve_smoothing = (0.3, 0.6, 0.8, 0.15)
            curve_multiplier = curve_multipliers[curve_intensity]
            curve_smooth = curve_smoothing[curve_intensity]
            is_sharp_curve = curve_intensity >= 2
            
            proportional = self.kp * effective_error * curve_multiplier
            steering_raw = proportional + derivative
            
            if self.show_debug:
                intensity_names = ('STRAIGHT', 'SLIGHT', 'MEDIUM', 'SHARP!')
                print(f"\033[1;97m[ LSTR ] :\033[0m \033[1;96mCURVE\033[0m - error:{abs_error_norm:.2f} [{intensity_names[curve_intensity]}] mult:{curve_multiplier}x")
            
            # Apply smoothing - more smoothing for stability in straight lines
            if abs_error_norm < self.dead_zone_ratio * 3:
                smooth = 0.25  # Very smooth
            elif is_sharp_curve:
                smooth = curve_smooth
            else:
                smooth = self.smoothing_factor
            
            if hasattr(self, 'last_steering'):
                steering_angle = smooth * steering_raw + (1 - smooth) * self.last_steering
            else:
                steering_angle = steering_raw
            
            # Clamp to max steering
            steering_angle = min(max(steering_angle, -self.max_steering), self.max_steering)
            
            # Calculate speed based on steering
            abs_steering = abs(steering_angle)
            if abs_steering > 15:
                speed = self.min_speed
            elif abs_steering > 8:
                speed = (self.min_speed + self.max_speed) / 2
            else:
                speed = self.max_speed
            
            # Create debug frame with lane overlay
            debug_frame = self.lstr_detector.draw_lanes(frame)
            
            # Draw header
            cv2.rectangle(debug_frame, (0, 0), (width, 80), (20, 20, 40), -1)
            cv2.putText(debug_frame, "LSTR AI ANALYSIS", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(debug_frame, f"Model: {self.lstr_detector.model_type.value}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(debug_frame, f"Inference: {inference_time:.1f}ms ({1000/max(1,inference_time):.1f} FPS)", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Draw lane info on right side
            cv2.putText(debug_frame, f"Lanes: {len(lanes)}", (width - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Steer: {steering_angle:.1f}", (width - 130, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(debug_frame, f"Speed: {speed:.0f}", (width - 100, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw lane center indicator
            center_y = int(height * 0.7)
            cv2.circle(debug_frame, (int(lane_center), center_y), 12, (255, 0, 255), -1)
            cv2.line(debug_frame, (int(frame_center), center_y - 30), (int(frame_center), center_y + 30), (0, 255, 255), 2)
            cv2.arrowedLine(debug_frame, (int(frame_center), center_y), (int(lane_center), center_y), (255, 0, 255), 2)
            
            # Draw offset text
            offset_text = f"Offset: {error:.0f}px"
            cv2.putText(debug_frame, offset_text, (int(frame_center) - 50, center_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show AI analysis window
            if self.show_debug:
                cv2.imshow("AI Analysis - LSTR", debug_frame)
                cv2.waitKey(1)
            
            # Store LSTR result for streaming
            self._store_debug_image('lstr', debug_frame)
            self._store_debug_image('final', debug_frame)
            
            return steering_angle, speed, lane_center, debug_frame
            
        except Exception as e:
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mERROR\033[0m - LSTR detection failed: {e}")
            return None, None, None, None

    def _detect_hybrid_fusion(self, frame):
        """
        Detect lanes using FUSION of OpenCV and LSTR.
        
        Both detectors run and their results are combined:
        - If both detect: weighted average of steering angles
        - If only one detects: use that one (with reduced confidence)
        - If neither detects: return None
        
        Returns:
            tuple: (steering_angle, speed, debug_frame) or (None, speed, debug_frame)
        """
        height, width = frame.shape[:2]
        
        # Initialize perspective if needed
        if not self.perspective_initialized:
            self._init_perspective_transform(width, height)
        
        # Run both detectors
        opencv_steering = None
        opencv_center = None
        lstr_steering = None
        lstr_center = None
        
        # 1. Run OpenCV detection (simplified - just get the lane center)
        try:
            preprocessed, _ = self._preprocess_frame(frame)
            white_mask = self._threshold_white(preprocessed)
            binary_warped = self.warp_perspective(white_mask)
            leftx, lefty, rightx, righty = self.sliding_window(binary_warped)
            left_fit, right_fit, lane_center_warped, curvature = self.fit_polynomial(
                leftx, lefty, rightx, righty, binary_warped.shape
            )
            
            if lane_center_warped is not None:
                # Transform back to original coordinates
                y_eval = int(height * (1.0 - self.lookahead))
                y_eval_warped = int(binary_warped.shape[0] * (1.0 - self.lookahead))
                
                point_warped = np.array([[[lane_center_warped, y_eval_warped]]], dtype=np.float32)
                point_original = cv2.perspectiveTransform(point_warped, self.perspective_M_inv)
                opencv_center = point_original[0][0][0]
                
                # Calculate OpenCV steering
                frame_center = width / 2
                error_norm = (opencv_center - frame_center) / frame_center
                opencv_steering = self.kp * error_norm  # Simplified PID
        except Exception as e:
            if self.show_debug:
                print(f"\033[1;97m[ HYBRID ] :\033[0m \033[1;93mOpenCV failed\033[0m - {e}")
        
        # 2. Run LSTR detection
        try:
            if self.lstr_detector is not None and self.lstr_detector.is_available:
                lanes, _ = self.lstr_detector.detect_lanes(frame)
                if len(lanes) > 0:
                    lstr_center = self.lstr_detector.get_lane_center(width, y_position_ratio=1.0 - self.lookahead)
                    if lstr_center is not None:
                        frame_center = width / 2
                        error_norm = (lstr_center - frame_center) / frame_center
                        lstr_steering = self.kp * error_norm  # Simplified PID
        except Exception as e:
            if self.show_debug:
                print(f"\033[1;97m[ HYBRID ] :\033[0m \033[1;93mLSTR failed\033[0m - {e}")
        
        # 3. Combine results
        final_steering = None
        confidence = 0.0
        source = "NONE"
        
        if opencv_steering is not None and lstr_steering is not None:
            # Both detected - weighted fusion
            final_steering = (
                self.hybrid_opencv_weight * opencv_steering + 
                self.hybrid_lstr_weight * lstr_steering
            )
            
            # Check agreement - if they agree, boost confidence
            steering_diff = abs(opencv_steering - lstr_steering)
            if steering_diff < 5:  # Within 5 degrees
                confidence = 1.0 * self.hybrid_agreement_bonus
                source = "FUSION (agree)"
            else:
                confidence = 0.8
                source = f"FUSION (diff:{steering_diff:.1f}°)"
                
        elif opencv_steering is not None:
            # Only OpenCV detected
            final_steering = opencv_steering
            confidence = 0.6
            source = "OpenCV only"
            
        elif lstr_steering is not None:
            # Only LSTR detected
            final_steering = lstr_steering
            confidence = 0.8  # LSTR is more reliable
            source = "LSTR only"
        
        # Apply smoothing and clamp
        if final_steering is not None:
            # Apply dead zone
            if abs(final_steering) < self.dead_zone_ratio * self.kp:
                final_steering = 0
            
            # Smooth with previous
            if hasattr(self, 'last_steering'):
                smooth = 0.4
                final_steering = smooth * final_steering + (1 - smooth) * self.last_steering
            
            # Clamp
            final_steering = min(max(final_steering, -self.max_steering), self.max_steering)
            self.last_steering = final_steering
        
        # Calculate speed based on steering
        if final_steering is not None:
            abs_steer = abs(final_steering)
            if abs_steer > 15:
                speed = self.min_speed
            elif abs_steer > 8:
                speed = (self.min_speed + self.max_speed) / 2
            else:
                speed = self.max_speed
        else:
            speed = self.min_speed
        
        # Create debug frame
        debug_frame = frame.copy()
        
        # Draw header
        cv2.rectangle(debug_frame, (0, 0), (width, 100), (20, 20, 40), -1)
        cv2.putText(debug_frame, "HYBRID FUSION MODE", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw detection sources
        y_pos = 50
        if opencv_steering is not None:
            cv2.putText(debug_frame, f"OpenCV: {opencv_steering:.1f} deg", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if opencv_center is not None:
                cv2.circle(debug_frame, (int(opencv_center), int(height * 0.8)), 8, (0, 255, 0), -1)
        else:
            cv2.putText(debug_frame, "OpenCV: --", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        y_pos = 70
        if lstr_steering is not None:
            cv2.putText(debug_frame, f"LSTR: {lstr_steering:.1f} deg", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
            if lstr_center is not None:
                cv2.circle(debug_frame, (int(lstr_center), int(height * 0.8)), 8, (255, 100, 0), -1)
        else:
            cv2.putText(debug_frame, "LSTR: --", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        y_pos = 90
        if final_steering is not None:
            cv2.putText(debug_frame, f"FINAL: {final_steering:.1f} deg [{source}]", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            # Draw final center
            final_center = width/2 + (final_steering / self.max_steering) * (width/4)
            cv2.circle(debug_frame, (int(final_center), int(height * 0.85)), 12, (0, 255, 255), 3)
        else:
            cv2.putText(debug_frame, "FINAL: NO DETECTION", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw frame center reference
        cv2.line(debug_frame, (width//2, height - 50), (width//2, height), (255, 255, 255), 2)
        
        if self.show_debug:
            cv2.imshow("HYBRID Fusion", debug_frame)
            cv2.waitKey(1)
            if final_steering is not None:
                print(f"\033[1;97m[ HYBRID ] :\033[0m \033[1;96m{source}\033[0m - Steer: {final_steering:.1f}° (OpenCV:{opencv_steering}, LSTR:{lstr_steering})")
        
        self._store_debug_image('hybrid', debug_frame)
        self._store_debug_image('final', debug_frame)
        
        return final_steering, speed, debug_frame

    def _init_perspective_transform(self, width, height):
        """Initialize perspective transform matrices for bird's eye view."""
        # Source points (trapezoid in the original image)
        src = np.float32([
            [width * 0.1, height],           # bottom-left
            [width * 0.4, height * 0.6],     # top-left
            [width * 0.6, height * 0.6],     # top-right
            [width * 0.9, height]            # bottom-right
        ])
        
        # Destination points (rectangle in bird's eye view)
        dst = np.float32([
            [width * 0.2, height],           # bottom-left
            [width * 0.2, 0],                # top-left
            [width * 0.8, 0],                # top-right
            [width * 0.8, height]            # bottom-right
        ])
        
        self.perspective_M = cv2.getPerspectiveTransform(src, dst)
        self.perspective_M_inv = cv2.getPerspectiveTransform(dst, src)
        self.perspective_initialized = True
        print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;92mINFO\033[0m - Perspective transform initialized for {width}x{height}")

    def warp_perspective(self, img):
        """Transform image to bird's eye view."""
        return cv2.warpPerspective(img, self.perspective_M, (img.shape[1], img.shape[0]))

    def unwarp_perspective(self, img):
        """Transform image back from bird's eye view."""
        return cv2.warpPerspective(img, self.perspective_M_inv, (img.shape[1], img.shape[0]))

    def get_histogram(self, binary_img):
        """Get histogram of bottom half of binary image."""
        histogram = np.sum(binary_img[binary_img.shape[0]//2:, :], axis=0)
        return histogram

    def find_lane_pixels_sliding_window(self, binary_warped):
        """Find lane pixels using sliding window method.

Returns:
    leftx, lefty, rightx, righty: Pixel coordinates of left and right lane pixels
    out_img: Visualization image
"""
        histogram = self.get_histogram(binary_warped)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        window_height = int(binary_warped.shape[0] // self.nwindows)
        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        left_lane_inds = []
        right_lane_inds = []
        
        for window in range(self.nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            
            win_xleft_low = leftx_current - self.window_margin
            win_xleft_high = leftx_current + self.window_margin
            win_xright_low = rightx_current - self.window_margin
            win_xright_high = rightx_current + self.window_margin
            
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            pass
        
        leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        
        return leftx, lefty, rightx, righty, out_img

    def _search_around_poly(self, binary_warped, out_img, nonzerox, nonzeroy):
        """Search for lane pixels around the previous polynomial fit.
Much faster and better for curves than blind sliding window."""
        margin = self.window_margin
        
        left_lane_inds = []
        right_lane_inds = []
        
        if self.left_fit is not None:
            left_fit_x = self.left_fit[0] * nonzeroy**2 + self.left_fit[1] * nonzeroy + self.left_fit[2]
            left_lane_inds = ((nonzerox > (left_fit_x - margin)) & (nonzerox < (left_fit_x + margin))).nonzero()[0]
        
        if self.right_fit is not None:
            right_fit_x = self.right_fit[0] * nonzeroy**2 + self.right_fit[1] * nonzeroy + self.right_fit[2]
            right_lane_inds = ((nonzerox > (right_fit_x - margin)) & (nonzerox < (right_fit_x + margin))).nonzero()[0]
        
        leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        
        return leftx, lefty, rightx, righty

    def fit_polynomial(self, leftx, lefty, rightx, righty, img_shape):
        """Fit a second order polynomial to lane pixels.

Returns:
    left_fit, right_fit: Polynomial coefficients
    lane_center: Center of the lane at the bottom of the image
    curvature: Estimated curvature
"""
        left_fit = None
        right_fit = None
        lane_center = img_shape[1] // 2
        curvature = None
        
        if len(leftx) > 50:
            left_fit = np.polyfit(lefty, leftx, 2)
            self.left_fit_history.append(left_fit)
            if len(self.left_fit_history) > self.fit_history_size:
                self.left_fit_history.pop(0)
            left_fit = np.mean(self.left_fit_history, axis=0)
            self.left_fit = left_fit
        
        if len(rightx) > 50:
            right_fit = np.polyfit(righty, rightx, 2)
            self.right_fit_history.append(right_fit)
            if len(self.right_fit_history) > self.fit_history_size:
                self.right_fit_history.pop(0)
            right_fit = np.mean(self.right_fit_history, axis=0)
            self.right_fit = right_fit
        
        y_eval = img_shape[0] - 1
        
        if left_fit is not None and right_fit is not None:
            left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
            right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
            lane_center = int((left_x + right_x) / 2)
            curvature = self._calculate_curvature(left_fit, right_fit, y_eval)
        elif left_fit is not None:
            left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
            lane_center = int(left_x + img_shape[1] * 0.3)
        elif right_fit is not None:
            right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
            lane_center = int(right_x - img_shape[1] * 0.3)
        
        return left_fit, right_fit, lane_center, curvature

    def _calculate_curvature(self, left_fit, right_fit, y_eval):
        """Calculate radius of curvature in pixels."""
        if left_fit is None and right_fit is None:
            return None
        
        curvatures = []
        for fit in [left_fit, right_fit]:
            if fit is not None:
                # R = (1 + (2Ay + B)^2)^1.5 / |2A|
                A = fit[0]
                B = fit[1]
                if abs(A) > 1e-6:
                    curv = ((1 + (2*A*y_eval + B)**2)**1.5) / abs(2*A)
                    curvatures.append(curv)
        
        if curvatures:
            return np.mean(curvatures)
        return None

    def draw_lane(self, warped_blank, left_fit, right_fit, img_shape):
        """Draw the detected lane on a blank warped image."""
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
        
        if left_fit is not None:
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            cv2.polylines(warped_blank, np.int32(pts_left), False, (255, 0, 0), thickness=10)
        
        if right_fit is not None:
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
            pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
            cv2.polylines(warped_blank, np.int32(pts_right), False, (0, 0, 255), thickness=10)
        
        if left_fit is not None and right_fit is not None:
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(warped_blank, np.int32([pts]), (0, 255, 0))
        
        return warped_blank

    def _apply_config(self, config):
        """Apply configuration from dashboard sliders."""
        try:
            params = ['base_speed', 'max_speed', 'min_speed', 'kp', 'kd', 'smoothing_factor',
                     'dead_zone_ratio', 'max_steering', 'roi_height_start', 'roi_height_end',
                     'roi_width_margin_top', 'roi_width_margin_bottom',
                     'white_h_min', 'white_h_max', 'white_s_min', 'white_s_max',
                     'white_v_min', 'white_v_max', 'yellow_h_min', 'yellow_h_max',
                     'yellow_s_min', 'yellow_s_max', 'yellow_v_min', 'yellow_v_max',
                     'brightness', 'contrast', 'blur_kernel', 'morph_kernel',
                     'canny_low', 'canny_high', 'hough_threshold', 'hough_min_line_length',
                     'hough_max_line_gap',
                     # Adaptive lighting parameters
                     'use_clahe', 'clahe_clip_limit', 'clahe_grid_size',
                     'use_adaptive_white', 'adaptive_white_percentile', 'adaptive_white_min_threshold',
                     'use_gradient_fallback', 'gradient_percentile',
                     # Detection mode
                     'detection_mode', 'lstr_model_size',
                     # Debug streaming
                     'stream_debug_view', 'stream_debug_fps', 'stream_debug_quality', 'stream_debug_scale']
            
            for param in params:
                if param in config:
                    setattr(self, param, config[param])
            
            # Log mode change and reset PID state
            if 'detection_mode' in config:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;96mMODE\033[0m - Changed to: {config['detection_mode']}")
                # Reset PID state to prevent corruption between modes
                self._reset_pid_state()
            
            # Reload LSTR model if size changed
            if 'lstr_model_size' in config:
                self._init_lstr_detector()
            
            self._update_hsv_arrays()
            print("\033[1;97m[ Line Following ] :\033[0m \033[1;92mCONFIG\033[0m - Parameters updated")
        except Exception as e:
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mERROR\033[0m - Config error: {e}")

    def _check_config(self):
        """Check for configuration updates from dashboard."""
        config = self.configSubscriber.receive()
        if config is not None:
            self._apply_config(config)

    def thread_work(self):
        """Main loop for line following."""
        try:
            self.check_state_change()
            self._check_config()
            
            camera_message = self.serialCameraSubscriber.receive()
            if camera_message is None:
                time.sleep(0.05)
                return
            
            img_data = base64.b64decode(camera_message)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return
            
            steering_angle, speed, debug_frame = self.process_frame(frame)
            
            if self.is_line_following_active:
                if steering_angle is not None:
                    self.send_motor_commands(steering_angle, speed)
                    self.frames_without_line = 0
                else:
                    self.frames_without_line += 1
                    if self.show_debug and self.frames_without_line % 10 == 1:
                        print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;93mWAIT\033[0m - No steering, frame {self.frames_without_line}/{self.max_frames_without_line}")
                    if self.frames_without_line > self.max_frames_without_line:
                        self.send_motor_commands(0, self.min_speed)
            else:
                # Line following not active - need to be in AUTO mode
                if steering_angle is not None:
                    print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mBLOCKED\033[0m - Steering={steering_angle:.1f}° but is_line_following_active=False! Put system in AUTO mode.")
                if hasattr(self, '_last_inactive_log') == False:
                    print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;93mINACTIVE\033[0m - Line following not active. Detection mode: {self.detection_mode}")
                    self._last_inactive_log = True
            
            if self.show_debug:
                if debug_frame is not None:
                    status_text = "ACTIVE" if self.is_line_following_active else "INACTIVE (Debug Mode)"
                    cv2.putText(debug_frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                               (0, 255, 0) if self.is_line_following_active else (0, 0, 255), 2)
                    cv2.imshow("1. Final Result", debug_frame)
                
                # Show control panel with current status
                self._show_control_panel(steering_angle, speed)
                cv2.waitKey(1)
        except Exception as e:
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mERROR\033[0m - {e}")

    def process_frame(self, frame):
        """Process frame using selected detection mode.
        
        Supports three modes:
        - opencv: Traditional HSV + sliding window (default)
        - lstr: AI-based LSTR transformer model
        - hybrid: FUSION of OpenCV + LSTR (runs both, combines results)

Returns:
    tuple: (steering_angle, speed, debug_frame)
"""
        height, width = frame.shape[:2]
        
        if self.show_debug:
            print(f"\033[1;97m[ Line Following ] :\033[0m Frame received: {width}x{height} Mode: {self.detection_mode}")
        
        # Handle HYBRID fusion mode (runs both detectors and combines)
        if self.detection_mode == DetectionMode.HYBRID.value:
            try:
                steering, speed, debug = self._detect_hybrid_fusion(frame)
                if steering is not None:
                    self.frames_without_line = 0
                return steering, speed, debug
            except Exception as e:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mHYBRID ERROR\033[0m - {e}")
                debug_frame = frame.copy()
                cv2.putText(debug_frame, f"HYBRID Error: {str(e)[:50]}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                return None, self.min_speed, debug_frame
        
        # Handle LSTR-only mode
        if self.detection_mode == DetectionMode.LSTR.value:
            try:
                steering, speed, center, debug = self._detect_with_lstr(frame)
                if steering is not None:
                    self.last_line_position = center
                    self.last_steering = steering
                    if self.show_debug:
                        print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;92mLSTR\033[0m - Steer: {steering:.1f}° Speed: {speed:.1f}")
                    return steering, speed, debug
                else:
                    # LSTR failed, return no detection
                    if self.show_debug:
                        print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;93mLSTR\033[0m - No lanes detected")
                    debug_frame = frame.copy()
                    cv2.putText(debug_frame, "LSTR: No lanes detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    return None, self.min_speed, debug_frame
            except Exception as e:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mLSTR ERROR\033[0m - {e}")
                debug_frame = frame.copy()
                cv2.putText(debug_frame, f"LSTR Error: {str(e)[:50]}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                return None, self.min_speed, debug_frame
        
        if not self.perspective_initialized:
            self._init_perspective_transform(width, height)
        
        orig_display = frame.copy()
        
        # Apply preprocessing (CLAHE + brightness/contrast)
        preprocessed, preprocess_debug = self._preprocess_frame(frame)
        
        if self.show_debug:
            if self.use_clahe and 'clahe' in preprocess_debug:
                clahe_display = preprocess_debug['clahe'].copy()
                cv2.putText(clahe_display, f"CLAHE: clip={self.clahe_clip_limit} grid={self.clahe_grid_size}",
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("2a. CLAHE Normalized", clahe_display)
                self._store_debug_image('clahe', clahe_display)
            
            adj_display = preprocess_debug['adjusted'].copy()
            cv2.putText(adj_display, f"Brightness: {self.brightness} Contrast: {self.contrast:.2f}",
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("2b. Brightness/Contrast", adj_display)
            self._store_debug_image('adjusted', adj_display)
        
        # Create combined mask using multiple detection methods
        combined_mask, mask_debug = self._create_combined_mask(frame, preprocessed)
        
        if self.show_debug:
            cv2.imshow("3. HSV", mask_debug['hsv'])
            self._store_debug_image('hsv', mask_debug['hsv'])
            
            white_display = cv2.cvtColor(mask_debug['white_hsv'], cv2.COLOR_GRAY2BGR)
            cv2.putText(white_display, f"H:{self.white_h_min}-{self.white_h_max}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(white_display, f"S:{self.white_s_min}-{self.white_s_max}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(white_display, f"V:{self.white_v_min}-{self.white_v_max}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.imshow("4. White Mask (HSV)", white_display)
            self._store_debug_image('white_mask', white_display)
            
            yellow_display = cv2.cvtColor(mask_debug['yellow'], cv2.COLOR_GRAY2BGR)
            cv2.putText(yellow_display, f"H:{self.yellow_h_min}-{self.yellow_h_max}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(yellow_display, f"S:{self.yellow_s_min}-{self.yellow_s_max}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(yellow_display, f"V:{self.yellow_v_min}-{self.yellow_v_max}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.imshow("5. Yellow Mask", yellow_display)
            self._store_debug_image('yellow_mask', yellow_display)
            
            # Show adaptive white detection if enabled
            if self.use_adaptive_white and 'adaptive_white' in mask_debug:
                adaptive_display = cv2.cvtColor(mask_debug['adaptive_white'], cv2.COLOR_GRAY2BGR)
                threshold = mask_debug.get('adaptive_threshold', 0)
                cv2.putText(adaptive_display, f"Adaptive threshold: {threshold:.0f}", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(adaptive_display, f"Percentile: {self.adaptive_white_percentile}", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.imshow("5b. Adaptive White", adaptive_display)
            
            # Show gradient detection if it was used
            if mask_debug.get('gradient_used', False) and 'gradient' in mask_debug:
                gradient_display = cv2.cvtColor(mask_debug['gradient'], cv2.COLOR_GRAY2BGR)
                cv2.putText(gradient_display, "GRADIENT FALLBACK ACTIVE", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.imshow("5c. Gradient Fallback", gradient_display)
        
        if self.show_debug:
            cv2.imshow("6. Combined (Raw)", mask_debug['combined_raw'])
        
        # Store combined mask for streaming (always, not just when show_debug)
        combined_display = cv2.cvtColor(mask_debug['combined_raw'], cv2.COLOR_GRAY2BGR)
        self._store_debug_image('combined_mask', combined_display)
        
        # Morphological operations
        morph_k = max(1, self.morph_kernel)
        kernel = np.ones((morph_k, morph_k), np.uint8)
        
        if self.use_dilation:
            dilate_k = max(1, self.dilate_kernel)
            dilate_kernel = np.ones((dilate_k, dilate_k), np.uint8)
            combined_mask = cv2.dilate(combined_mask, dilate_kernel, iterations=1)
        
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        if self.show_debug:
            morph_display = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            dilate_str = f"Dilate: {self.dilate_kernel}" if self.use_dilation else "No dilate"
            cv2.putText(morph_display, f"Morph: {self.morph_kernel} {dilate_str}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.imshow("7. After Morphology", morph_display)
        
        # Blur
        blur_k = max(1, self.blur_kernel)
        if blur_k % 2 == 0:
            blur_k += 1
        combined_mask = cv2.GaussianBlur(combined_mask, (blur_k, blur_k), 0)
        
        if self.show_debug:
            blur_display = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(blur_display, f"Blur Kernel: {blur_k}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.imshow("8. After Blur", blur_display)
        
        # Warp to bird's eye view
        binary_warped = self.warp_perspective(combined_mask)
        
        if self.show_debug:
            cv2.imshow("9. Bird's Eye View", binary_warped)
        
        # Store birds eye for streaming
        birds_eye_display = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)
        self._store_debug_image('birds_eye', birds_eye_display)
        
        # Find lane pixels using sliding window
        leftx, lefty, rightx, righty, sliding_window_img = self.find_lane_pixels_sliding_window(binary_warped)
        
        if self.show_debug:
            cv2.imshow("10. Sliding Window", sliding_window_img)
        
        # Store sliding window for streaming
        self._store_debug_image('sliding_window', sliding_window_img)
        
        # Fit polynomial
        left_fit, right_fit, lane_center, curvature = self.fit_polynomial(leftx, lefty, rightx, righty, binary_warped.shape)
        
        # Calculate steering
        steering_angle = None
        speed = self.base_speed
        
        if left_fit is not None or right_fit is not None:
            # Calculate lane center in original image coordinates
            y_eval_warped = int(binary_warped.shape[0] * (1 - self.lookahead))
            
            if left_fit is not None and right_fit is not None:
                left_x = left_fit[0] * y_eval_warped**2 + left_fit[1] * y_eval_warped + left_fit[2]
                right_x = right_fit[0] * y_eval_warped**2 + right_fit[1] * y_eval_warped + right_fit[2]
                lane_center_warped = (left_x + right_x) / 2
            elif left_fit is not None:
                lane_center_warped = left_fit[0] * y_eval_warped**2 + left_fit[1] * y_eval_warped + left_fit[2] + width * 0.3
            else:
                lane_center_warped = right_fit[0] * y_eval_warped**2 + right_fit[1] * y_eval_warped + right_fit[2] - width * 0.3
            
            # Transform lane center back to original image
            point_warped = np.array([[[lane_center_warped, y_eval_warped]]], dtype=np.float32)
            point_original = cv2.perspectiveTransform(point_warped, self.perspective_M_inv)
            lane_center_original = point_original[0][0][0]
            
            # Also transform frame center for reference
            center_warped = np.array([[[width / 2, y_eval_warped]]], dtype=np.float32)
            center_original = cv2.perspectiveTransform(center_warped, self.perspective_M_inv)
            frame_center_at_y = center_original[0][0][0]
            
            # Calculate error
            error = lane_center_original - frame_center_at_y
            error_ratio = abs(error) / (width / 2)
            
            # Curvature-based speed adjustment
            if curvature is not None:
                if curvature < 300:
                    curve_intensity = 3  # Sharp curve
                elif curvature < 600:
                    curve_intensity = 2  # Medium curve
                elif curvature < 1000:
                    curve_intensity = 1  # Slight curve
                else:
                    curve_intensity = 0  # Straight
            else:
                curve_intensity = 0
            
            curve_multipliers = (1, 1.8, 3, 4.5)
            curve_smoothing = (0.3, 0.6, 0.8, 0.15)
            curve_multiplier = curve_multipliers[curve_intensity]
            curve_smooth = curve_smoothing[curve_intensity]
            is_sharp_curve = curve_intensity >= 2
            
            if self.show_debug:
                intensity_names = ('STRAIGHT', 'SLIGHT', 'MEDIUM', 'SHARP!')
                if curvature is not None:
                    curv_str = f"curv:{curvature:.0f}"
                else:
                    curv_str = "curv:N/A"
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;96mOFFSET\033[0m - offset:{error:.1f}px ratio:{error_ratio:.2f} {curv_str} [{intensity_names[curve_intensity]}]")
            
            # PID control with NORMALIZED error (range -1 to 1)
            half_width = width / 2
            error_normalized = error / half_width  # Now in range [-1, 1]
            
            if self.is_line_following_active:
                total_correction = self.single_line_correction if (left_fit is None or right_fit is None) else 1
            else:
                total_correction = 1
            
            # Apply dead zone - ignore small errors to prevent oscillation in straight lines
            abs_error_norm = abs(error_normalized)
            if abs_error_norm < self.dead_zone_ratio:
                # Very small error - no correction
                effective_error = 0
                derivative = 0
            else:
                # Outside dead zone - subtract dead zone for smooth transition
                effective_error = (abs_error_norm - self.dead_zone_ratio) * (1 if error_normalized > 0 else -1)
                derivative = self.kd * (error_normalized - self.previous_error)
            
            self.previous_error = error_normalized
            
            # PID calculation
            proportional = self.kp * effective_error * total_correction * curve_multiplier
            steering_raw = proportional + derivative
            
            # Apply smoothing - more smoothing in straight lines, less in curves
            if abs_error_norm < self.dead_zone_ratio * 3:  # Near straight
                smooth = 0.25  # Very smooth for stability
            elif is_sharp_curve:
                smooth = curve_smooth
            else:
                smooth = self.smoothing_factor
            
            if hasattr(self, 'last_steering'):
                steering_angle = smooth * steering_raw + (1 - smooth) * self.last_steering
            else:
                steering_angle = steering_raw
            
            # Clamp to max steering
            steering_angle = min(max(steering_angle, -self.max_steering), self.max_steering)
            self.last_steering = steering_angle
            
            # Update turn direction
            if abs(steering_angle) > 8:
                self.last_turn_direction = 1 if steering_angle > 0 else -1
            
            # Adjust speed based on steering
            abs_steering = abs(steering_angle)
            if abs_steering > 15:
                speed = self.min_speed
            elif abs_steering > 8:
                speed = (self.min_speed + self.max_speed) / 2
            else:
                speed = self.max_speed
            
            self.last_line_position = lane_center_original
            self.frames_without_line = 0
        else:
            # No lanes detected with OpenCV
            self.frames_without_line += 1
            
            if self.show_debug:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mNO LANES\033[0m - Stopped. Waiting for lines... (frame {self.frames_without_line})")
            
            if hasattr(self, 'last_steering') and self.frames_without_line < 5:
                steering_angle = self.last_steering * 0.95
            else:
                steering_angle = None
            speed = self.min_speed
        
        # Create debug frame
        debug_frame = frame.copy()
        
        if left_fit is not None or right_fit is not None:
            lane_warped = np.zeros((height, width, 3), dtype=np.uint8)
            lane_warped = self.draw_lane(lane_warped, left_fit, right_fit, (height, width))
            lane_unwarped = self.unwarp_perspective(lane_warped)
            debug_frame = cv2.addWeighted(debug_frame, 1, lane_unwarped, 0.3, 0)
            
            # Draw lane center point
            if self.last_line_position is not None:
                point_for_y = int(height * (1 - self.lookahead))
                point_y_original = int(height * 0.8)
                center_y = point_y_original
                center_x = int(self.last_line_position)
                cv2.circle(debug_frame, (center_x, center_y), 10, (255, 0, 255), -1)
                cv2.line(debug_frame, (width // 2, height), (width // 2, int(height * 0.6)), (0, 255, 255), 2)
        
        # Store final debug image and send stream
        self._store_debug_image('final', debug_frame)
        self._send_debug_stream(steering_angle, speed)
        
        return steering_angle, speed, debug_frame

    def draw_rounded_rectangle(self, img, pt1, pt2, color, thickness, radius):
        """Draw a rectangle with rounded corners.

Args:
    img: Image to draw on
    pt1: Top-left corner (x, y)
    pt2: Bottom-right corner (x, y)
    color: Rectangle color
    thickness: Line thickness
    radius: Corner radius
"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

    def calculate_line_center(self, lines, frame_width, roi_start_y, roi_end_y):
        """Calculate the center point for lane following.

Uses weighted average of all detected lines, then determines left/right lanes."""
        if lines is None or len(lines) == 0:
            return None, None, None
        
        left_lines = []
        right_lines = []
        center_x = frame_width // 2
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:
                continue
            
            avg_x = (x1 + x2) / 2
            if avg_x < center_x:
                left_lines.append((x1, y1, x2, y2, slope))
            else:
                right_lines.append((x1, y1, x2, y2, slope))
        
        left_x = None
        right_x = None
        
        if left_lines:
            left_x = np.mean([((l[0] + l[2]) / 2) for l in left_lines])
            if self.show_debug:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;93mLEFT\033[0m - lines:{len(left_lines)}")
        
        if right_lines:
            right_x = np.mean([((l[0] + l[2]) / 2) for l in right_lines])
            if self.show_debug:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;93mRIGHT\033[0m - lines:{len(right_lines)}")
        
        if left_x is not None and right_x is not None:
            lane_center = (left_x + right_x) / 2
            self.last_seen_side = "both"
            if self.show_debug:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;92mBOTH\033[0m - min:{left_x:.0f} max:{right_x:.0f}")
        elif left_x is not None:
            lane_center = left_x + frame_width * 0.25
            self.last_seen_side = "left"
        elif right_x is not None:
            lane_center = right_x - frame_width * 0.25
            self.last_seen_side = "right"
        else:
            return None, None, None
        
        return lane_center, left_x, right_x

    def send_motor_commands(self, steering_angle, speed):
        """Send steering and speed commands to the motors.
        
        Format matches frontend: SteerMotor expects string value already multiplied by 10.
        Example: steering_angle=5.0 -> sends "50" to SteerMotor
        """
        try:
            # Nucleo expects steering values multiplied by 10 (e.g. 5° becomes 50) for integer efficiency
            # This matches frontend: Math.round(this.steer * 10)
            steer_value = int(round(steering_angle * 10))
            speed_value = int(round(speed * 10))  # Speed also multiplied by 10 like frontend
            
            # Send as string - the serial handler expects str type
            self.steerMotorSender.send(str(steer_value))
            self.speedMotorSender.send(str(speed_value))
            
            # Always log motor commands so we can see they're being sent
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;92mMOTOR\033[0m - Steer: {steer_value} Speed: {speed_value}")
            
        except Exception as e:
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mERROR\033[0m - Failed to send motor commands: {e}")

    def check_state_change(self):
        """Check for state changes and enable/disable line following accordingly."""
        message = self.stateChangeSubscriber.receive()
        if message is not None:
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;96mSTATE\033[0m - Received: {message}")
            
            try:
                mode_dict = SystemMode[message].value.get("camera", {}).get("lineFollowing", {})
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;96mSTATE\033[0m - lineFollowing config: {mode_dict}")
                
                if mode_dict.get("enabled", False):
                    self.is_line_following_active = True
                    self._last_inactive_log = False  # Reset inactive log flag
                    print("\033[1;97m[ Line Following ] :\033[0m \033[1;92mACTIVATED\033[0m - Line following is now ACTIVE!")
                else:
                    self.is_line_following_active = False
                    print("\033[1;97m[ Line Following ] :\033[0m \033[1;93mDEACTIVATED\033[0m - Line following is now INACTIVE")
            except KeyError as e:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mERROR\033[0m - Unknown mode: {message} - {e}")

    def stop(self):
        """Stop the thread and cleanup."""
        if self.show_debug:
            cv2.destroyAllWindows()
        super(threadLineFollowing, self).stop()
