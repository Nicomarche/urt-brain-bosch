# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.a
# Reconstructed from bytecode

import cv2
import numpy as np
import base64
import time
import math
from collections import deque
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

# Import HybridNets remote AI client
try:
    from aiserver.client import HybridNetsClient
    HYBRIDNETS_CLIENT_AVAILABLE = True
except ImportError as e:
    HYBRIDNETS_CLIENT_AVAILABLE = False
    print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;93mWARNING\033[0m - HybridNets client not available: {e}")


class DetectionMode(Enum):
    """Lane detection modes."""
    OPENCV = "opencv"           # BFMC-style OpenCV (Threshold + Canny + Hough)
    LSTR = "lstr"              # AI-based LSTR model (local)
    HYBRID = "hybrid"           # Fusion of OpenCV + LSTR (combines both for better accuracy)
    HYBRIDNETS = "hybridnets"  # Remote HybridNets AI server (GPU offload)
    SUPERCOMBO = "supercombo"  # Remote Supercombo AI server (openpilot model, GPU offload)


class PIDController:
    """PID Controller for lane following steering.
    
    Based on bfmc24-brain implementation (ricardolopezb/bfmc24-brain).
    Uses normalized error [-1, 1] where 1.0 = max deviation from center.
    Output is steering angle in degrees, clamped to ±max_steering.
    
    Args:
        Kp: Proportional gain. Higher = more aggressive correction.
        Ki: Integral gain. Corrects persistent errors over time.
        Kd: Derivative gain. Dampens oscillations.
        max_steering: Maximum output angle in degrees (physical limit).
        tolerance: Dead zone - errors below this are ignored (prevents oscillation).
        integral_reset_interval: Reset integral every N iterations (anti-windup).
    """
    
    def __init__(self, Kp=25.0, Ki=1.0, Kd=4.0, max_steering=25, tolerance=0.02,
                 integral_reset_interval=10):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_steering = max_steering
        self.tolerance = tolerance
        self.integral_reset_interval = integral_reset_interval
        self.prev_error = 0
        self.integral = 0.0
        self.iteration_count = 0
        self._last_time = None
    
    def compute(self, error):
        """Compute PID control signal.
        
        Args:
            error: Normalized error in range [-1, 1].
                   Positive = car is to the right of center, steer left.
                   
        Returns:
            float: Steering angle in degrees, clamped to ±max_steering.
        """
        # Calculate real dt from timestamps
        current_time = time.time()
        if self._last_time is not None:
            dt = current_time - self._last_time
            dt = max(0.01, min(dt, 1.0))  # Clamp between 10ms and 1s
        else:
            dt = 0.1  # Default 100ms for first frame
        self._last_time = current_time
        
        # Dead zone: ignore small errors for straight line stability
        if abs(error) < self.tolerance:
            self.integral = 0.0  # Reset integral in dead zone
            self.prev_error = error
            return 0.0
        
        # P - Proportional: immediate response to current error
        proportional = self.Kp * error
        
        # I - Integral: accumulate error over time (corrects persistent offset)
        self.integral += error * dt
        integral = self.Ki * self.integral
        
        # D - Derivative: rate of change (dampens oscillation)
        if dt > 0:
            derivative = self.Kd * (error - self.prev_error) / dt
        else:
            derivative = 0
        
        self.prev_error = error
        self.iteration_count += 1
        
        # Reset integral periodically to prevent windup (like bfmc24-brain)
        if self.integral_reset_interval > 0 and self.iteration_count % self.integral_reset_interval == 0:
            self.integral = 0.0
        
        # Sum PID terms
        control_signal = proportional + integral + derivative
        
        # Clamp to physical steering limits
        control_signal = max(min(control_signal, self.max_steering), -self.max_steering)
        
        return control_signal
    
    def reset(self):
        """Reset all PID state (call when switching modes or restarting)."""
        self.prev_error = 0
        self.integral = 0.0
        self.iteration_count = 0
        self._last_time = None


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
        self.base_speed = 15
        self.max_speed = 25
        self.min_speed = 8
        self.speed_ramp_step = 0.5   # Max speed increase per frame (gradual acceleration)
        self._current_speed = self.base_speed  # Tracks actual speed for ramping

        # PID Controller (values from ricardolopezb/bfmc24-brain configs.py)
        # Error is fed in raw pixels (not normalized). Kp=0.075 → 293px error = 22° max steering.
        self.max_error_px = 40   # Pixel offset reference (used by LSTR normalization)
        self.kp = 0.08  # Proportional gain (bfmc24-brain: PID_KP)
        self.ki = 0.05   # Integral gain (bfmc24-brain: PID_KI)
        self.kd = 0.05   # Derivative gain (bfmc24-brain: PID_KD)
        self.max_steering = 25   # Maximum steering angle in degrees (bfmc24-brain: ±22)
        self.dead_zone_ratio = 50  # Ignore errors below 50px (bfmc24-brain: PID_TOLERANCE)
        self.integral_reset_interval = 10  # Reset integral every N iterations (anti-windup)
        self.smoothing_factor = 0.5  # Higher = more reactive, Lower = more smooth
        self.steer_history = deque(maxlen=1)  # Moving average of last 5 steering values (smooths erratic readings)
        self.lookahead = 0.4
        self.pid = PIDController(
            Kp=self.kp, Ki=self.ki, Kd=self.kd,
            max_steering=self.max_steering,
            tolerance=self.dead_zone_ratio,
            integral_reset_interval=self.integral_reset_interval
        )
        self.previous_error = 0  # Keep for compatibility

        # Feed-forward curve prediction parameters
        self.wheelbase = 0.265       # Distance between axles in meters (BFMC 1:10 car ~26.5cm)
        self.ff_weight = 0.6         # Weight of feed-forward vs PID (0=pure PID, 1=pure FF)
        self.curvature_threshold = 0.5  # Min curvature to activate feed-forward (1/pixels)

        # ROI parameters (BFMC uses roi_height_start=0.35 — top 35% masked)
        self.roi_height_start = 0.35
        self.roi_height_end = 1.0
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

        # Image processing parameters (defaults from bfmc24-brain)
        self.blur_kernel = 3           # Median blur kernel (BFMC uses 3)
        self.morph_kernel = 7
        self.canny_low = 100           # Canny lower threshold
        self.canny_high = 150          # Canny upper threshold (BFMC uses 150)
        self.dilate_kernel = 5
        self.use_dilation = True
        self.binary_threshold = 165    # BFMC grayscale threshold (main)
        self.binary_threshold_retry = 90  # Retry with lower if no lines found
        self.hough_threshold = 50      # BFMC votes threshold
        self.hough_min_line_length = 50  # BFMC min line length
        self.hough_max_line_gap = 150  # BFMC max gap between segments
        self.line_angle_filter = 30    # Min angle (degrees) to classify as lane line
        self.line_merge_distance = 175 # Max pixel distance to merge lines

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
        
        # HybridNets remote AI server parameters
        self.hybridnets_server_url = "ws://192.168.1.35:8500/ws/steering"
        self.hybridnets_jpeg_quality = 70
        self.hybridnets_timeout = 2.0  # GPU inference <100ms + network ~50ms; 2s es suficiente
        self._hybridnets_client = None
        
        # Supercombo remote AI server parameters (same protocol as HybridNets)
        self.supercombo_server_url = "ws://192.168.1.35:8500/ws/steering"
        self.supercombo_jpeg_quality = 70
        self.supercombo_timeout = 2.0
        self._supercombo_client = None
        
        self._init_lstr_detector()
        self._init_hybridnets_client()
        self._init_supercombo_client()

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
        
        # BFMC-style single-line tracking
        self.consecutive_single_left = 0
        self.consecutive_single_right = 0
        self.just_seen_two_lines = False

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

    def _init_hybridnets_client(self):
        """Initialize HybridNets remote AI client if available."""
        if not HYBRIDNETS_CLIENT_AVAILABLE:
            print("\033[1;97m[ Line Following ] :\033[0m \033[1;93mWARNING\033[0m - HybridNets client not available")
            self._hybridnets_client = None
            return
        
        try:
            self._hybridnets_client = HybridNetsClient(
                server_url=self.hybridnets_server_url,
                jpeg_quality=self.hybridnets_jpeg_quality,
                timeout=self.hybridnets_timeout,
            )
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;92mINFO\033[0m - HybridNets client created (server: {self.hybridnets_server_url})")
        except Exception as e:
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mERROR\033[0m - Failed to create HybridNets client: {e}")
            self._hybridnets_client = None

    def _start_hybridnets_client(self):
        """Start or restart the HybridNets client connection."""
        if self._hybridnets_client is None:
            self._init_hybridnets_client()
        
        if self._hybridnets_client is not None:
            # Siempre limpiar datos viejos al (re)arrancar para no procesar frames stale
            self._hybridnets_client.flush()
            
            if not self._hybridnets_client.connected:
                try:
                    self._hybridnets_client.start()
                    print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;92mINFO\033[0m - HybridNets client started, connecting to {self.hybridnets_server_url}")
                except Exception as e:
                    print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mERROR\033[0m - Failed to start HybridNets client: {e}")

    def _stop_hybridnets_client(self):
        """Stop the HybridNets client."""
        if self._hybridnets_client is not None:
            try:
                self._hybridnets_client.stop()
            except Exception:
                pass

    # ================================================================
    # Supercombo remote AI client (same protocol as HybridNets)
    # ================================================================

    def _init_supercombo_client(self):
        """Initialize Supercombo remote AI client if available."""
        if not HYBRIDNETS_CLIENT_AVAILABLE:
            print("\033[1;97m[ Line Following ] :\033[0m \033[1;93mWARNING\033[0m - Supercombo client not available (needs websockets)")
            self._supercombo_client = None
            return
        
        try:
            self._supercombo_client = HybridNetsClient(
                server_url=self.supercombo_server_url,
                jpeg_quality=self.supercombo_jpeg_quality,
                timeout=self.supercombo_timeout,
            )
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;92mINFO\033[0m - Supercombo client created (server: {self.supercombo_server_url})")
        except Exception as e:
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mERROR\033[0m - Failed to create Supercombo client: {e}")
            self._supercombo_client = None

    def _start_supercombo_client(self):
        """Start or restart the Supercombo client connection."""
        if self._supercombo_client is None:
            self._init_supercombo_client()
        
        if self._supercombo_client is not None:
            self._supercombo_client.flush()
            
            if not self._supercombo_client.connected:
                try:
                    self._supercombo_client.start()
                    print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;92mINFO\033[0m - Supercombo client started, connecting to {self.supercombo_server_url}")
                except Exception as e:
                    print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mERROR\033[0m - Failed to start Supercombo client: {e}")

    def _stop_supercombo_client(self):
        """Stop the Supercombo client."""
        if self._supercombo_client is not None:
            try:
                self._supercombo_client.stop()
            except Exception:
                pass

    def _detect_with_supercombo(self, frame):
        """
        Detect lanes using remote Supercombo AI server (openpilot model).
        
        Uses the same WebSocket protocol as HybridNets:
        sends JPEG frame, receives steering angle.
        
        Args:
            frame: BGR image
            
        Returns:
            tuple: (steering_angle, speed, debug_frame)
        """
        height, width = frame.shape[:2]
        
        # Ensure client is running
        if self._supercombo_client is None or not self._supercombo_client.connected:
            self._start_supercombo_client()
            if not self._supercombo_client or not self._supercombo_client.connected:
                debug_frame = frame.copy()
                cv2.rectangle(debug_frame, (0, 0), (width, 60), (20, 20, 40), -1)
                cv2.putText(debug_frame, "SUPERCOMBO - Connecting...", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.putText(debug_frame, f"Server: {self.supercombo_server_url}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                self._store_debug_image('final', debug_frame)
                return None, self.min_speed, debug_frame
        
        start_time = time.time()
        
        # Send frame and get result
        result = self._supercombo_client.send_frame(frame, block=True)
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Create debug frame
        debug_frame = frame.copy()
        cv2.rectangle(debug_frame, (0, 0), (width, 80), (20, 20, 40), -1)
        cv2.putText(debug_frame, "SUPERCOMBO (openpilot)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        if result is None:
            cv2.putText(debug_frame, "No response from server", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            stats = self._supercombo_client.get_stats()
            cv2.putText(debug_frame, f"Connected: {stats['connected']} | Sent: {stats['frames_sent']}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            self._store_debug_image('final', debug_frame)
            return None, self.min_speed, debug_frame
        
        # Parse result (same format as HybridNets /ws/steering endpoint)
        if 's' in result:
            steering_angle = result.get('s')
            confidence = result.get('c', 0)
            error_norm = result.get('e', 0)
            server_time = result.get('t', 0)
            frame_id = result.get('f', 0)
            roundtrip = result.get('roundtrip_ms', 0)
        else:
            steering_info = result.get('steering', {})
            steering_angle = steering_info.get('steering_angle')
            confidence = steering_info.get('confidence', 0)
            error_norm = steering_info.get('error_normalized', 0)
            server_time = result.get('inference_time_ms', 0)
            frame_id = result.get('frame_id', 0)
            roundtrip = result.get('roundtrip_ms', 0)
        
        cv2.putText(debug_frame, f"Server: {server_time:.0f}ms | Roundtrip: {roundtrip:.0f}ms", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        if steering_angle is not None:
            steering_angle = min(max(steering_angle, -self.max_steering), self.max_steering)
            self.last_steering = steering_angle
            self.previous_error = error_norm
            
            # Calculate speed based on steering magnitude
            abs_steer = abs(steering_angle)
            if abs_steer > 15:
                speed = self.min_speed
            elif abs_steer > 8:
                speed = (self.min_speed + self.max_speed) / 2
            else:
                speed = self.max_speed
            
            # Diagnostic logging
            if not hasattr(self, '_supercombo_frame_count'):
                self._supercombo_frame_count = 0
            self._supercombo_frame_count += 1
            if self._supercombo_frame_count <= 5 or self._supercombo_frame_count % 20 == 0:
                steer_serial = int(round(steering_angle * 10))
                speed_serial = int(round(speed * 10))
                print(f"\033[1;97m[ Supercombo ] :\033[0m "
                      f"steer={steering_angle:.1f}deg (serial:{steer_serial}) | "
                      f"speed={speed:.0f} (serial:{speed_serial}) | "
                      f"conf={confidence:.2f} | "
                      f"server={server_time:.0f}ms RT={roundtrip:.0f}ms | "
                      f"frame#{self._supercombo_frame_count}")
            
            # Draw steering info
            steer_color = (0, 255, 0) if abs(steering_angle) < 10 else (0, 165, 255) if abs(steering_angle) < 20 else (0, 0, 255)
            cv2.putText(debug_frame, f"Steer: {steering_angle:.1f} deg (x10={int(round(steering_angle*10))})", (width - 280, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, steer_color, 2)
            cv2.putText(debug_frame, f"Confidence: {confidence:.2f}", (width - 200, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(debug_frame, f"Speed: {speed:.0f} (x10={int(round(speed*10))})", (width - 280, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw center indicator
            center_x = width // 2
            center_y = int(height * 0.8)
            offset_px = int((steering_angle / self.max_steering) * (width / 4))
            cv2.line(debug_frame, (center_x, height), (center_x, int(height * 0.6)), (0, 255, 255), 2)
            cv2.arrowedLine(debug_frame, (center_x, center_y), (center_x + offset_px, center_y - 30),
                           steer_color, 3)
            
            self._store_debug_image('final', debug_frame)
            self._store_debug_image('supercombo', debug_frame)
            return steering_angle, speed, debug_frame
        else:
            print(f"\033[1;97m[ Supercombo ] :\033[0m \033[1;91mNO LANES\033[0m - server returned steering=None | "
                  f"server={server_time:.0f}ms RT={roundtrip:.0f}ms")
            cv2.putText(debug_frame, "No lanes detected", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            self._store_debug_image('final', debug_frame)
            return None, self.min_speed, debug_frame

    def _detect_with_hybridnets(self, frame):
        """
        Detect lanes using remote HybridNets AI server.
        
        Sends the frame to the server via WebSocket and receives
        the steering angle back.
        
        Args:
            frame: BGR image
            
        Returns:
            tuple: (steering_angle, speed, debug_frame)
        """
        height, width = frame.shape[:2]
        
        # Ensure client is running
        if self._hybridnets_client is None or not self._hybridnets_client.connected:
            self._start_hybridnets_client()
            # Give it a moment to connect on first call
            if not self._hybridnets_client or not self._hybridnets_client.connected:
                debug_frame = frame.copy()
                cv2.rectangle(debug_frame, (0, 0), (width, 60), (20, 20, 40), -1)
                cv2.putText(debug_frame, "HYBRIDNETS - Connecting...", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.putText(debug_frame, f"Server: {self.hybridnets_server_url}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                self._store_debug_image('final', debug_frame)
                return None, self.min_speed, debug_frame
        
        start_time = time.time()
        
        # Send frame and get result
        result = self._hybridnets_client.send_frame(frame, block=True)
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Create debug frame
        debug_frame = frame.copy()
        cv2.rectangle(debug_frame, (0, 0), (width, 80), (20, 20, 40), -1)
        cv2.putText(debug_frame, "HYBRIDNETS AI SERVER", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        if result is None:
            cv2.putText(debug_frame, "No response from server", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            stats = self._hybridnets_client.get_stats()
            cv2.putText(debug_frame, f"Connected: {stats['connected']} | Sent: {stats['frames_sent']}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            self._store_debug_image('final', debug_frame)
            return None, self.min_speed, debug_frame
        
        # Parse result (format from /ws/steering endpoint)
        if 's' in result:
            # Steering-only response
            steering_angle = result.get('s')
            confidence = result.get('c', 0)
            error_norm = result.get('e', 0)
            server_time = result.get('t', 0)
            frame_id = result.get('f', 0)
            roundtrip = result.get('roundtrip_ms', 0)
        else:
            # Full inference response
            steering_info = result.get('steering', {})
            steering_angle = steering_info.get('steering_angle')
            confidence = steering_info.get('confidence', 0)
            error_norm = steering_info.get('error_normalized', 0)
            server_time = result.get('inference_time_ms', 0)
            frame_id = result.get('frame_id', 0)
            roundtrip = result.get('roundtrip_ms', 0)
        
        # Draw info on debug frame
        cv2.putText(debug_frame, f"Server: {server_time:.0f}ms | Roundtrip: {roundtrip:.0f}ms", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        if steering_angle is not None:
            # El servidor ya aplica PD + smoothing, no aplicar doble PID aqui.
            # Solo clamp al rango valido y calcular velocidad.
            steering_angle = min(max(steering_angle, -self.max_steering), self.max_steering)
            self.last_steering = steering_angle
            self.previous_error = error_norm  # Mantener tracking para cambio de modo
            
            # Calculate speed based on steering magnitude
            abs_steer = abs(steering_angle)
            if abs_steer > 15:
                speed = self.min_speed
            elif abs_steer > 8:
                speed = (self.min_speed + self.max_speed) / 2
            else:
                speed = self.max_speed
            
            # Log de diagnostico: siempre mostrar los primeros 5 y luego cada 20
            if not hasattr(self, '_hybridnets_frame_count'):
                self._hybridnets_frame_count = 0
            self._hybridnets_frame_count += 1
            if self._hybridnets_frame_count <= 5 or self._hybridnets_frame_count % 20 == 0:
                steer_serial = int(round(steering_angle * 10))
                speed_serial = int(round(speed * 10))
                print(f"\033[1;97m[ HybridNets ] :\033[0m "
                      f"steer={steering_angle:.1f}deg (serial:{steer_serial}) | "
                      f"speed={speed:.0f} (serial:{speed_serial}) | "
                      f"conf={confidence:.2f} | "
                      f"server={server_time:.0f}ms RT={roundtrip:.0f}ms | "
                      f"frame#{self._hybridnets_frame_count}")
            
            # Draw steering info
            steer_color = (0, 255, 0) if abs(steering_angle) < 10 else (0, 165, 255) if abs(steering_angle) < 20 else (0, 0, 255)
            cv2.putText(debug_frame, f"Steer: {steering_angle:.1f} deg (x10={int(round(steering_angle*10))})", (width - 280, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, steer_color, 2)
            cv2.putText(debug_frame, f"Confidence: {confidence:.2f}", (width - 200, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(debug_frame, f"Speed: {speed:.0f} (x10={int(round(speed*10))})", (width - 280, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw center indicator
            center_x = width // 2
            center_y = int(height * 0.8)
            offset_px = int((steering_angle / self.max_steering) * (width / 4))
            cv2.line(debug_frame, (center_x, height), (center_x, int(height * 0.6)), (0, 255, 255), 2)
            cv2.arrowedLine(debug_frame, (center_x, center_y), (center_x + offset_px, center_y - 30),
                           steer_color, 3)
            
            self._store_debug_image('final', debug_frame)
            self._store_debug_image('hybridnets', debug_frame)
            return steering_angle, speed, debug_frame
        else:
            print(f"\033[1;97m[ HybridNets ] :\033[0m \033[1;91mNO LANES\033[0m - server returned steering=None | "
                  f"server={server_time:.0f}ms RT={roundtrip:.0f}ms")
            cv2.putText(debug_frame, "No lanes detected", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            self._store_debug_image('final', debug_frame)
            return None, self.min_speed, debug_frame

    def _reset_pid_state(self):
        """Reset PID state when changing modes to prevent corrupted values."""
        self.pid.reset()
        self.previous_error = 0
        if hasattr(self, 'last_steering'):
            del self.last_steering
        self.last_line_position = None
        self.frames_without_line = 0
        self.last_turn_direction = 0
        self._hybridnets_frame_count = 0
        print("\033[1;97m[ Line Following ] :\033[0m \033[1;96mPID\033[0m - State reset (P/I/D cleared)")

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
            
            # === CURVE PREDICTION: Estimate curvature from full lane shape ===
            curvature, turn_sign = self.lstr_detector.estimate_path_curvature(width)
            
            # Feed-forward: geometric Ackermann steering angle for detected curvature
            ff_steer = 0.0
            radius = 0.0
            if abs(curvature) > self.curvature_threshold and turn_sign != 0:
                # Convert pixel curvature to approximate real-world radius
                # The camera sees ~1m ahead in ~height pixels, so pixels_per_meter ≈ height
                pixels_per_meter = height  # Rough approximation
                real_curvature = curvature * pixels_per_meter  # 1/meters
                if real_curvature > 0.01:
                    radius = 1.0 / real_curvature  # meters
                    # Ackermann formula: steer = atan(wheelbase / radius)
                    ff_steer = math.degrees(math.atan(self.wheelbase / radius)) * turn_sign
                    # Clamp feed-forward to physical limits
                    ff_steer = max(min(ff_steer, self.max_steering), -self.max_steering)
            
            # === PID: fine corrections based on lane center error ===
            frame_center = width / 2
            error = lane_center - frame_center
            # Normalize: ±max_error_px maps to [-1, 1] (40px offset = 25° steering)
            error_normalized = max(min(error / self.max_error_px, 1.0), -1.0)
            
            pid_steer = self.pid.compute(error_normalized)
            
            # === BLEND: combine feed-forward prediction with PID correction ===
            if abs(ff_steer) > 0.1:
                # Curve detected: blend feed-forward + PID
                steering_angle = self.ff_weight * ff_steer + (1.0 - self.ff_weight) * pid_steer
            else:
                # Straight road: use pure PID
                steering_angle = pid_steer
            
            # Clamp to physical limits
            steering_angle = max(min(steering_angle, self.max_steering), -self.max_steering)
            
            # Optional smoothing for stability
            if hasattr(self, 'last_steering'):
                steering_angle = self.smoothing_factor * steering_angle + (1 - self.smoothing_factor) * self.last_steering
                steering_angle = max(min(steering_angle, self.max_steering), -self.max_steering)
            
            if self.show_debug:
                p_term = self.pid.Kp * error_normalized
                i_term = self.pid.Ki * self.pid.integral
                d_term = self.pid.Kd * (error_normalized - self.pid.prev_error) if hasattr(self.pid, 'prev_error') else 0
                if abs(ff_steer) > 0.1:
                    curve_dir = "R" if turn_sign > 0 else "L"
                    radius_str = f"R={radius:.2f}m" if radius > 0 else "R=inf"
                    print(f"\033[1;97m[ LSTR ] :\033[0m \033[1;95mCURVE {curve_dir}\033[0m - "
                          f"curv:{curvature:.4f} {radius_str} FF:{ff_steer:.1f}° PID:{pid_steer:.1f}° "
                          f"-> steer:{steering_angle:.1f}° (w={self.ff_weight:.1f})")
                else:
                    print(f"\033[1;97m[ LSTR ] :\033[0m \033[1;96mPID\033[0m - err:{error_normalized:.3f} steer:{steering_angle:.1f}° "
                          f"P={p_term:.1f} I={i_term:.1f} D={d_term:.1f} "
                          f"(Kp={self.pid.Kp:.1f} Ki={self.pid.Ki:.1f} Kd={self.pid.Kd:.1f})")
            
            # === ADAPTIVE SPEED based on curvature ===
            if abs(curvature) > 2.0:
                speed = self.min_speed  # Tight curve: minimum speed
            elif abs(curvature) > 1.0:
                speed = (self.min_speed + self.max_speed) / 2  # Moderate curve
            else:
                speed = self.max_speed  # Straight: full speed
            
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
            
            # Draw curvature / feed-forward info bar
            curv_bar_y = 85
            if abs(ff_steer) > 0.1:
                curve_dir = "R" if turn_sign > 0 else "L"
                radius_str = f"R={radius:.2f}m" if radius > 0 else ""
                cv2.rectangle(debug_frame, (0, curv_bar_y), (width, curv_bar_y + 25), (80, 0, 80), -1)
                cv2.putText(debug_frame, f"CURVE {curve_dir} {radius_str} FF:{ff_steer:.1f} PID:{pid_steer:.1f} w={self.ff_weight:.1f}",
                           (10, curv_bar_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 255), 1)
            else:
                cv2.putText(debug_frame, f"STRAIGHT  PID:{pid_steer:.1f}",
                           (10, curv_bar_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 255, 150), 1)
            
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

    # ==================================================================
    # BFMC-style lane detection helpers
    # Based on: https://github.com/ricardolopezb/bfmc24-brain
    # Pipeline: ROI → Grayscale → Threshold → MedianBlur → Canny → HoughP
    # ==================================================================

    def _bfmc_image_processing(self, image, threshold_override=None, kernel_override=None):
        """
        BFMC-style image processing pipeline.
        
        1. Apply ROI (bottom portion of image)
        2. Convert to grayscale
        3. Binary threshold
        4. Median blur for noise removal
        5. Canny edge detection
        6. HoughLinesP for line detection
        7. Classify, merge, and average lines
        
        Args:
            image: BGR frame
            threshold_override: Override binary threshold (for retry)
            kernel_override: Override median blur kernel (for retry)
            
        Returns:
            (avg_left_line, avg_right_line, height, width, canny_image, debug_info)
        """
        threshold_val = threshold_override if threshold_override is not None else self.binary_threshold
        kernel_val = kernel_override if kernel_override is not None else self.blur_kernel
        if kernel_val % 2 == 0:
            kernel_val += 1

        height, width = image.shape[:2]
        debug_info = {}

        # 1. ROI mask — keep bottom portion (roi_height_start..roi_height_end)
        y_start = int(self.roi_height_start * height)
        y_end = int(self.roi_height_end * height)
        roi_vertices = np.array([[(0, y_start), (width, y_start), (width, y_end), (0, y_end)]], dtype=np.int32)
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, roi_vertices, (255, 255, 255))
        masked_image = cv2.bitwise_and(image, mask)
        debug_info['roi'] = masked_image.copy()

        # 2. Grayscale
        grey_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        debug_info['grey'] = grey_image.copy()

        # 3. Binary threshold
        _, binary_image = cv2.threshold(grey_image, threshold_val, 255, cv2.THRESH_BINARY)
        debug_info['binary'] = binary_image.copy()

        # 4. Median blur
        noiseless = cv2.medianBlur(binary_image, kernel_val)
        debug_info['noiseless'] = noiseless.copy()

        # 5. Canny
        canny = cv2.Canny(noiseless, self.canny_low, self.canny_high)
        debug_info['canny'] = canny.copy()

        # 6. HoughLinesP
        lines = cv2.HoughLinesP(
            canny, 1, np.pi / 180,
            self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        # 7. Classify → merge → average
        left_lines, right_lines = self._bfmc_classify_lines(lines)
        merged_left = self._bfmc_merge_lines(left_lines)
        merged_right = self._bfmc_merge_lines(right_lines)
        avg_left = self._bfmc_average_lines(merged_left)
        avg_right = self._bfmc_average_lines(merged_right)

        debug_info['all_lines'] = lines
        debug_info['left_lines'] = merged_left
        debug_info['right_lines'] = merged_right
        debug_info['threshold'] = threshold_val

        return avg_left, avg_right, height, width, canny, debug_info

    def _bfmc_classify_lines(self, lines):
        """
        Classify detected lines into left and right lanes by slope.
        
        Left lane lines have negative slope (going up-left).
        Right lane lines have positive slope (going up-right).
        Lines with angle below self.line_angle_filter are horizontal → rejected.
        
        Based on MarcosLaneDetector.lines_classifier()
        """
        left_lines = []
        right_lines = []

        if lines is None:
            return left_lines, right_lines

        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            if dx == 0:
                slope = np.pi / 2
            else:
                slope = np.arctan((y2 - y1) / dx)

            angle_degrees = np.degrees(abs(slope))

            # Reject near-horizontal lines
            if angle_degrees < self.line_angle_filter:
                continue

            if slope < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

        return left_lines, right_lines

    def _bfmc_merge_lines(self, lines):
        """
        Merge nearby line segments into single lines.
        
        If the start x of one line is close to the end x of another
        (within self.line_merge_distance), merge them.
        
        Based on MarcosLaneDetector.merge_lines()
        """
        merged = []
        for line in lines:
            if len(merged) == 0:
                merged.append(line)
            else:
                merge_flag = False
                for i, mline in enumerate(merged):
                    if abs(line[0][0] - mline[0][2]) < self.line_merge_distance:
                        merged[i] = np.array([[mline[0][0], mline[0][1], line[0][2], line[0][3]]])
                        merge_flag = True
                        break
                if not merge_flag:
                    merged.append(line)
        return merged

    def _bfmc_average_lines(self, lines):
        """
        Average all lines into a single representative line.
        
        Returns:
            np.array([[x1, y1, x2, y2]]) or None if no lines
        """
        if len(lines) == 0:
            return None
        lines_array = np.array(lines)
        return np.mean(lines_array, axis=0, dtype=np.int32)

    def _bfmc_get_error(self, avg_left, avg_right, height, width):
        """
        Calculate lateral error: distance from lane midpoint to image center.
        
        When both lines are detected:
        1. Find where each line intersects the bottom of the image
        2. Compute the midpoint between those two intersections
        3. Error = midpoint_x - image_center_x (positive = offset right)
        
        Also computes the vanishing point (intersection of left and right line).
        
        Based on MarcosLaneDetector.getting_error()
        
        Returns:
            error in pixels (positive = car is right of center, steer left)
        """
        x1_l, y1_l, x2_l, y2_l = avg_left[0]
        x1_r, y1_r, x2_r, y2_r = avg_right[0]

        # Extend lines to bottom of image
        dy_l = y2_l - y1_l
        if dy_l == 0:
            dy_l = 0.01
        bottom_left_x = int(x1_l + (height - y1_l) * (x2_l - x1_l) / dy_l)

        dy_r = y2_r - y1_r
        if dy_r == 0:
            dy_r = 0.01
        bottom_right_x = int(x1_r + (height - y1_r) * (x2_r - x1_r) / dy_r)

        midpoint_x = (bottom_left_x + bottom_right_x) // 2
        bottom_center_x = width // 2
        error = midpoint_x - bottom_center_x

        return error, midpoint_x, bottom_left_x, bottom_right_x

    def _bfmc_follow_single_line(self, line, side):
        """
        Compute steering angle when only one lane line is visible.
        
        Uses the slope of the line to estimate how much to turn:
        - Steep line (close to vertical) → line is nearly parallel, go straight
        - Shallow line → line is diagonal, turn harder
        
        Based on MarcosLaneDetector.follow_left_line / follow_right_line + slope_mapper
        
        Args:
            line: np.array([[x1, y1, x2, y2]])
            side: 'left' or 'right'
            
        Returns:
            steering_angle in degrees (clamped to ±max_steering)
        """
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            # Perfectly vertical line → go straight
            return 0

        slope_deg = math.degrees(abs(dy / dx))

        # BFMC slope_mapper: maps slope angle to discrete steering values
        if slope_deg > 50:
            steering_angle = 3    # Nearly vertical → slight correction
        elif slope_deg > 40:
            steering_angle = 11   # Moderate curve
        else:
            steering_angle = 22   # Sharp curve → max steering

        # Invert for right line (mirror steering)
        if side == 'right':
            steering_angle = -steering_angle

        # Clamp to configured max
        return max(-self.max_steering, min(self.max_steering, steering_angle))

    def _bfmc_draw_debug(self, frame, avg_left, avg_right, error, midpoint_x, 
                          bottom_left_x, bottom_right_x, steering_angle, debug_info):
        """
        Draw BFMC-style debug visualization on frame.
        
        Shows: detected lines (red), midpoint (cyan), center (green),
        error line (orange), steering arrow, and info panel.
        """
        h, w = frame.shape[:2]
        debug = frame.copy()

        # Draw all raw Hough lines (faint blue)
        all_lines = debug_info.get('all_lines')
        if all_lines is not None:
            for line in all_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(debug, (x1, y1), (x2, y2), (180, 120, 60), 1)

        # Draw averaged left line (red, extended)
        if avg_left is not None:
            self._bfmc_draw_extended_line(debug, avg_left, h, (0, 0, 255), 2)

        # Draw averaged right line (red, extended)
        if avg_right is not None:
            self._bfmc_draw_extended_line(debug, avg_right, h, (0, 0, 255), 2)

        # Draw midpoint and center
        if midpoint_x is not None:
            cv2.circle(debug, (midpoint_x, h), 8, (0, 255, 255), -1)  # Cyan = midpoint
            cv2.circle(debug, (w // 2, h), 8, (0, 255, 0), -1)       # Green = center
            cv2.line(debug, (midpoint_x, h), (w // 2, h), (0, 165, 255), 2)  # Orange = error

        # Header panel
        cv2.rectangle(debug, (0, 0), (w, 55), (20, 20, 40), -1)
        cv2.putText(debug, "OpenCV - BFMC Pipeline", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        if steering_angle is not None:
            steer_color = (0, 255, 0) if abs(steering_angle) < 10 else \
                (0, 165, 255) if abs(steering_angle) < 20 else (0, 0, 255)
            cv2.putText(debug, f"Steer: {steering_angle:.1f} deg  Error: {error or 0:.0f}px",
                        (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, steer_color, 1)

            # Steering arrow at bottom
            cx = w // 2
            offset_px = int((steering_angle / self.max_steering) * (w / 4))
            cv2.arrowedLine(debug, (cx, h - 30), (cx + offset_px, h - 60), steer_color, 3, tipLength=0.3)
        else:
            cv2.putText(debug, "NO LINES DETECTED", (8, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Detection info
        th = debug_info.get('threshold', self.binary_threshold)
        l_count = len(debug_info.get('left_lines', []))
        r_count = len(debug_info.get('right_lines', []))
        cv2.putText(debug, f"Thr:{th} L:{l_count} R:{r_count}", (8, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

        return debug

    def _bfmc_draw_extended_line(self, image, line, height, color, thickness):
        """Draw a line extended to top and bottom of image."""
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        if dx == 0:
            dx = 0.001
        slope = (y2 - y1) / dx
        if slope == 0:
            slope = 0.001
        # Intersection at y=0 (top)
        x_top = int(x1 + (0 - y1) / slope)
        # Intersection at y=height (bottom)
        x_bot = int(x1 + (height - y1) / slope)
        cv2.line(image, (x_top, 0), (x_bot, height), color, thickness)

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
            params = ['base_speed', 'max_speed', 'min_speed', 'max_error_px', 'kp', 'ki', 'kd', 'smoothing_factor',
                     'dead_zone_ratio', 'max_steering', 'lookahead', 'integral_reset_interval',
                     'wheelbase', 'ff_weight', 'curvature_threshold',
                     'roi_height_start', 'roi_height_end',
                     'roi_width_margin_top', 'roi_width_margin_bottom',
                     'white_h_min', 'white_h_max', 'white_s_min', 'white_s_max',
                     'white_v_min', 'white_v_max', 'yellow_h_min', 'yellow_h_max',
                     'yellow_s_min', 'yellow_s_max', 'yellow_v_min', 'yellow_v_max',
                     'brightness', 'contrast', 'blur_kernel', 'morph_kernel',
                     'canny_low', 'canny_high', 'hough_threshold', 'hough_min_line_length',
                     'hough_max_line_gap',
                     # BFMC-style parameters
                     'binary_threshold', 'binary_threshold_retry',
                     'line_angle_filter', 'line_merge_distance',
                     # Adaptive lighting parameters
                     'use_clahe', 'clahe_clip_limit', 'clahe_grid_size',
                     'use_adaptive_white', 'adaptive_white_percentile', 'adaptive_white_min_threshold',
                     'use_gradient_fallback', 'gradient_percentile',
                     # Detection mode
                     'detection_mode', 'lstr_model_size',
                     # Debug streaming
                     'stream_debug_view', 'stream_debug_fps', 'stream_debug_quality', 'stream_debug_scale',
                     # HybridNets remote AI server
                     'hybridnets_server_url', 'hybridnets_jpeg_quality', 'hybridnets_timeout',
                     # Supercombo remote AI server
                     'supercombo_server_url', 'supercombo_jpeg_quality', 'supercombo_timeout']
            
            for param in params:
                if param in config:
                    setattr(self, param, config[param])
            
            # Sync PID controller with updated parameters
            pid_params = ['kp', 'ki', 'kd', 'max_steering', 'dead_zone_ratio', 'integral_reset_interval']
            if any(p in config for p in pid_params):
                self.pid.Kp = self.kp
                self.pid.Ki = self.ki
                self.pid.Kd = self.kd
                self.pid.max_steering = self.max_steering
                self.pid.tolerance = self.dead_zone_ratio
                self.pid.integral_reset_interval = self.integral_reset_interval
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;96mPID\033[0m - Updated: Kp={self.kp} Ki={self.ki} Kd={self.kd} max={self.max_steering}")
            
            # Log feed-forward parameter changes
            ff_params = ['wheelbase', 'ff_weight', 'curvature_threshold']
            if any(p in config for p in ff_params):
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;95mFF\033[0m - Updated: wheelbase={self.wheelbase} ff_weight={self.ff_weight} curv_thresh={self.curvature_threshold}")
            
            # Log mode change and reset PID state
            if 'detection_mode' in config:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;96mMODE\033[0m - Changed to: {config['detection_mode']}")
                # Reset PID state to prevent corruption between modes
                self._reset_pid_state()
            
            # Reload LSTR model if size changed
            if 'lstr_model_size' in config:
                self._init_lstr_detector()
            
            # Reconnect HybridNets client if server URL changed
            if 'hybridnets_server_url' in config:
                self._stop_hybridnets_client()
                self._init_hybridnets_client()
            
            # Reconnect Supercombo client if server URL changed
            if 'supercombo_server_url' in config:
                self._stop_supercombo_client()
                self._init_supercombo_client()
            
            # Start/stop remote AI clients based on mode
            if 'detection_mode' in config:
                if config['detection_mode'] == DetectionMode.HYBRIDNETS.value:
                    self._start_hybridnets_client()
                    self._stop_supercombo_client()
                elif config['detection_mode'] == DetectionMode.SUPERCOMBO.value:
                    self._start_supercombo_client()
                    self._stop_hybridnets_client()
                else:
                    self._stop_hybridnets_client()
                    self._stop_supercombo_client()
            
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
        
        # Handle SUPERCOMBO remote AI server mode
        if self.detection_mode == DetectionMode.SUPERCOMBO.value:
            try:
                steering, speed, debug = self._detect_with_supercombo(frame)
                if steering is not None:
                    self.frames_without_line = 0
                return steering, speed, debug
            except Exception as e:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mSUPERCOMBO ERROR\033[0m - {e}")
                debug_frame = frame.copy()
                cv2.putText(debug_frame, f"Supercombo Error: {str(e)[:50]}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                return None, self.min_speed, debug_frame
        
        # Handle HYBRIDNETS remote AI server mode
        if self.detection_mode == DetectionMode.HYBRIDNETS.value:
            try:
                steering, speed, debug = self._detect_with_hybridnets(frame)
                if steering is not None:
                    self.frames_without_line = 0
                return steering, speed, debug
            except Exception as e:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mHYBRIDNETS ERROR\033[0m - {e}")
                debug_frame = frame.copy()
                cv2.putText(debug_frame, f"HybridNets Error: {str(e)[:50]}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                return None, self.min_speed, debug_frame
        
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
        
        # ==============================================================
        # BFMC-style OpenCV pipeline
        # Based on: https://github.com/ricardolopezb/bfmc24-brain
        # ROI → Grayscale → Threshold → MedianBlur → Canny → HoughP
        # → Classify → Merge → Average → Error → PID
        # ==============================================================

        # First attempt with normal threshold
        avg_left, avg_right, img_h, img_w, canny, debug_info = self._bfmc_image_processing(frame)

        # BFMC retry: if no lines detected, try lower threshold (from MarcosLaneDetector)
        if avg_left is None and avg_right is None:
            avg_left, avg_right, img_h, img_w, canny, debug_info = self._bfmc_image_processing(
                frame,
                threshold_override=self.binary_threshold_retry,
                kernel_override=3
            )

        # Show debug windows
        if self.show_debug:
            if 'binary' in debug_info:
                bin_display = cv2.cvtColor(debug_info['binary'], cv2.COLOR_GRAY2BGR)
                cv2.putText(bin_display, f"Threshold: {debug_info.get('threshold', '?')}", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow("2. Binary Threshold", bin_display)
            if 'canny' in debug_info:
                cv2.imshow("3. Canny Edges", debug_info['canny'])
            self._store_debug_image('canny', cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR))

        # Calculate steering based on detected lines
        steering_angle = None
        speed = self.base_speed
        error = None
        midpoint_x = None
        bottom_left_x = None
        bottom_right_x = None

        if avg_left is not None and avg_right is not None:
            # ---- BOTH LINES DETECTED ----
            # BFMC logic: if just_seen_two_lines was False, skip this frame (stabilize)
            if self.just_seen_two_lines:
                error, midpoint_x, bottom_left_x, bottom_right_x = self._bfmc_get_error(
                    avg_left, avg_right, img_h, img_w
                )

                # PID control with raw pixel error (bfmc24-brain style)
                steering_angle = self.pid.compute(error)

                # Moving average of last 5 steering values (smooths erratic readings from lighting)
                self.steer_history.append(steering_angle)
                steering_angle = sum(self.steer_history) / len(self.steer_history)
                steering_angle = max(-self.max_steering, min(self.max_steering, steering_angle))

                self.just_seen_two_lines = False
            else:
                # First frame with two lines after single/none — use previous steering (stabilize)
                self.just_seen_two_lines = True
                steering_angle = getattr(self, 'last_steering', 0)

            # Reset single-line counters
            self.consecutive_single_left = 0
            self.consecutive_single_right = 0
            self.frames_without_line = 0
            self.last_seen_side = "both"

        elif avg_left is not None:
            # ---- ONLY LEFT LINE ----
            self.consecutive_single_left += 1
            self.consecutive_single_right = 0
            self.last_seen_side = "left"

            if self.consecutive_single_left >= 2:
                # After 2 frames with only left → max right turn
                steering_angle = self.max_steering
                speed = self.min_speed
            else:
                steering_angle = self._bfmc_follow_single_line(avg_left, 'left')

            self.frames_without_line = 0

        elif avg_right is not None:
            # ---- ONLY RIGHT LINE ----
            self.consecutive_single_right += 1
            self.consecutive_single_left = 0
            self.last_seen_side = "right"

            if self.consecutive_single_right >= 2:
                # After 2 frames with only right → max left turn
                steering_angle = -self.max_steering
                speed = self.min_speed
            else:
                steering_angle = self._bfmc_follow_single_line(avg_right, 'right')

            self.frames_without_line = 0

        else:
            # ---- NO LINES ----
            self.frames_without_line += 1

            if self.show_debug:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mNO LANES\033[0m - frame {self.frames_without_line}")

            if hasattr(self, 'last_steering') and self.frames_without_line < 5:
                steering_angle = self.last_steering * 0.95
            else:
                steering_angle = None
            speed = self.min_speed

        # Update state
        if steering_angle is not None:
            self.last_steering = steering_angle

            # Update turn direction
            if abs(steering_angle) > 8:
                self.last_turn_direction = 1 if steering_angle > 0 else -1

            # Adaptive speed based on steering magnitude (BFMC-style)
            abs_steer = abs(steering_angle)
            if abs_steer > 15:
                target_speed = self.min_speed
            elif abs_steer > 8:
                target_speed = (self.min_speed + self.max_speed) / 2
            else:
                target_speed = self.max_speed

            # Speed ramping: instant decrease, gradual increase
            if target_speed <= self._current_speed:
                # Braking: drop instantly
                self._current_speed = target_speed
            else:
                # Acceleration: ramp up gradually
                self._current_speed = min(target_speed, self._current_speed + self.speed_ramp_step)
            speed = self._current_speed

        if self.show_debug and steering_angle is not None:
            src = "both" if avg_left is not None and avg_right is not None else \
                  "left" if avg_left is not None else \
                  "right" if avg_right is not None else "none"
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;96mBFMC\033[0m - "
                  f"Steer: {steering_angle:.1f}° Error: {error or 0:.0f}px Source: {src}")

        # Create debug frame
        debug_frame = self._bfmc_draw_debug(
            frame, avg_left, avg_right, error, midpoint_x,
            bottom_left_x, bottom_right_x, steering_angle, debug_info
        )

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
        
        Protocol: Nucleo expects integer values = angle * 10 for decimal precision.
        Example: steering_angle=5.0° -> sends "50" -> Nucleo interprets as 5.0°
        This matches the frontend: Math.round(this.steer * 10)
        """
        try:
            steer_value = int(round(steering_angle * 10))
            speed_value = int(round(speed * 10))
            
            # Send as string - the serial handler expects str type
            self.steerMotorSender.send(str(steer_value))
            self.speedMotorSender.send(str(speed_value))
            
            # Log motor commands only in debug mode to avoid I/O overhead on RPi
            if self.show_debug:
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
        self._stop_hybridnets_client()
        if self.show_debug:
            cv2.destroyAllWindows()
        super(threadLineFollowing, self).stop()
