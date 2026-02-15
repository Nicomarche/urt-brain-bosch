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

        # ============= Car Physical Dimensions (cm) =============
        # Measured from the actual BFMC 1:10 scale car (Bosch)
        self.car_length = 36.5           # Total car length (cm)
        self.car_width = 19.0            # Total car width (cm)
        self.car_wheelbase_cm = 27.5     # Measured distance between axles (cm)
        self.car_front_overhang = 7.2    # Front axle to front bumper (cm)
        self.car_rear_overhang = 1.8     # Rear axle to rear of car (cm)
        self.camera_to_front_axle = 11.5 # Camera position ahead of front axle (cm)
        self.camera_to_rear_axle = 39.0  # Camera to rear axle (11.5 + 27.5) (cm)

        # Track dimensions (cm) - BFMC standard track
        self.lane_width_cm = 35.0        # Lane width between line inner edges (cm)
        self.line_width_cm = 2.0         # Width of lane line markings (cm)

        # Swept path prediction parameters
        self.use_swept_path = True       # Enable geometric curve prediction
        self.swept_path_margin_cm = 2.0  # Safety margin for clearance checks (cm)
        self.curve_offset_gain = 1.0     # Apply full optimal offset (0-1)
        self.curve_confidence_threshold = 0.3  # Min confidence to apply curve offset
        self.min_clearance_warn_cm = 3.0 # Warn if clearance below this (cm)
        self.curve_speed_reduction = True  # Reduce speed when clearance is tight
        self.single_line_radius_k = 31.0 # Calibration constant: R ≈ k / tan(slope_angle) (cm)
        self._swept_path_info = None     # Last swept path computation result (for debug)
        self._last_px_per_cm = None      # Cached px/cm ratio from last two-line detection

        # ============= HYBRID CURVE SYSTEM =============
        # Combines: BFMC known radii + steering feedback + VP estimation
        # State machine: STRAIGHT → ENTERING → IN_CURVE → EXITING → STRAIGHT

        # BFMC track known curve radii (cm, to lane center)
        self.bfmc_inner_lane_radius = 66.5   # Center of inner lane (48+85)/2
        self.bfmc_outer_lane_radius = 103.5  # Center of outer lane (85+122)/2
        self.bfmc_default_curve_radius = 66.5  # Default: assume inner lane (worst case)

        # Curve state machine
        self._curve_state = "STRAIGHT"       # STRAIGHT, ENTERING, IN_CURVE, EXITING
        self._curve_direction = 0            # -1=left, 0=none, 1=right
        self._curve_radius_estimate = 0.0    # Current best radius estimate (cm)
        self._curve_state_frames = 0         # Frames in current state
        self._curve_confidence = 0.0         # Confidence in current curve estimate

        # State transition thresholds
        self.curve_enter_frames = 1          # Frames with 1 line to start ENTERING (fast!)
        self.curve_confirm_frames = 2        # Frames with 1 line to confirm IN_CURVE
        self.curve_exit_frames = 3           # Frames with 2 lines before EXITING→STRAIGHT
        self.curve_vp_threshold = 0.10       # VP offset to detect curve early from 2 lines
        self.curve_pre_position_gain = 0.85  # Pre-position aggressively when ENTERING

        # Steering feedback estimator
        self._steering_radius_history = deque(maxlen=8)  # Last N radius estimates from steering
        self._last_steering_angle = 0.0      # Last known steering angle

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
        
        # Frame noise rejection filter (handles reflections/glare)
        self.use_noise_filter = True         # Enable noise rejection
        self.noise_max_hough_lines = 40      # Max Hough lines before flagging as noisy
        self.noise_max_error_jump_px = 80    # Max error change between frames (px)
        self.noise_max_steer_jump_deg = 15   # Max steering change between frames (deg)
        self.noise_max_reject_frames = 3     # Max consecutive frames to reject
        self._noise_reject_count = 0         # Current consecutive rejected frames
        self._last_good_error = 0.0          # Last accepted error value
        self._last_good_steering = 0.0       # Last accepted steering angle
        self._last_good_num_lines = 0        # Last accepted line count

        # Curve recovery (reverse when curve is impossible)
        self.use_curve_recovery = True       # Enable reverse recovery
        self.recovery_max_steer_frames = 8   # Frames at max steering before triggering
        self.recovery_reverse_speed = -5     # Reverse speed (negative = backward)
        self.recovery_reverse_time_min = 0.3 # Minimum reverse time (small error)
        self.recovery_reverse_time_max = 2.5 # Maximum reverse time (huge error)
        self.recovery_reverse_steer_scale = 1.5  # Amplifier for reverse steer magnitude (>1 amplifies)
        self.recovery_pre_turn_time = 0.6   # Seconds to wait for wheels to turn before reversing
        self.recovery_realign_time = 0.6     # Seconds to wait for wheels to reach curve angle
        self.recovery_error_shrink_ratio = 0.85  # If error shrinks to this ratio, car IS correcting (no recovery)
        self._recovery_state = "NONE"        # NONE, STOPPING, PRE_TURNING, REVERSING, REALIGNING, RESUMING
        self._recovery_start_time = 0.0      # When current recovery phase started
        self._recovery_actual_rev_time = 0.0 # Computed reverse time for this recovery
        self._recovery_reverse_steer_angle = 0.0  # Pre-calculated fixed steer angle for reversing
        self._max_steer_consecutive = 0      # Frames at max steering
        self._error_at_max_steer_start = 0.0 # Error when max steering streak started
        self._recovery_curve_sign = 0        # +1 for right curve, -1 for left curve

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

    def _is_frame_noisy(self, debug_info, error, steering_angle, num_lines):
        """Check if current frame detection is likely corrupted by noise/reflections.
        
        A frame is considered noisy if:
        1. Too many Hough lines detected (reflections create many spurious edges)
        2. Error/steering jumps suddenly from previous good frame
        3. Line count changes unexpectedly (had 2 lines, suddenly 0)
        
        If noisy, the frame should be rejected and previous steering maintained.
        
        Args:
            debug_info: Dict from _bfmc_image_processing with 'all_lines' etc.
            error: Current computed error in pixels (or None)
            steering_angle: Current computed steering angle (or None)
            num_lines: Number of lines detected (0, 1, or 2)
            
        Returns:
            tuple: (is_noisy: bool, reason: str)
        """
        if not self.use_noise_filter:
            return False, ""

        # Check 1: Too many Hough lines = reflections/glare
        all_lines = debug_info.get('all_lines')
        if all_lines is not None and len(all_lines) > self.noise_max_hough_lines:
            return True, f"hough_overflow({len(all_lines)}>{self.noise_max_hough_lines})"

        # Check 2: Sudden error jump (only if we have a previous reference)
        if error is not None and self._last_good_error is not None:
            error_jump = abs(error - self._last_good_error)
            if error_jump > self.noise_max_error_jump_px and self._noise_reject_count < self.noise_max_reject_frames:
                return True, f"error_jump({error_jump:.0f}px>{self.noise_max_error_jump_px})"

        # Check 3: Sudden steering jump
        if steering_angle is not None and self._last_good_steering is not None:
            steer_jump = abs(steering_angle - self._last_good_steering)
            if steer_jump > self.noise_max_steer_jump_deg and self._noise_reject_count < self.noise_max_reject_frames:
                # Exception: don't reject if we're transitioning states (curve entry/exit)
                if self._curve_state in ("ENTERING", "EXITING"):
                    return False, ""
                return True, f"steer_jump({steer_jump:.0f}°>{self.noise_max_steer_jump_deg})"

        # Check 4: Had both lines, suddenly lost both (not just one - that's normal curve entry)
        if self._last_good_num_lines == 2 and num_lines == 0:
            if self._noise_reject_count < self.noise_max_reject_frames:
                return True, "sudden_loss(2→0)"

        return False, ""

    def _accept_frame(self, error, steering_angle, num_lines):
        """Record current frame as a good (accepted) reference for noise filtering."""
        if error is not None:
            self._last_good_error = error
        if steering_angle is not None:
            self._last_good_steering = steering_angle
        self._last_good_num_lines = num_lines
        self._noise_reject_count = 0

    def _check_curve_recovery(self, steering_angle, speed, frame):
        """Check if the car is stuck at max steering and needs to reverse.
        
        Detection: If steering is at ±max for several consecutive frames
        AND the error is NOT decreasing (max steering isn't helping),
        the car can't make the curve from its current position.
        This prevents false triggers on straights where max steering
        is temporary and the error is being corrected.
        
        Reverse steering uses a FIXED angle pre-calculated at trigger time:
        - Direction: always OPPOSITE to the curve (reliable 3-point turn)
        - Magnitude: proportional to error at trigger (small error = gentle,
          big error = aggressive), scaled by recovery_reverse_steer_scale.
        
        Reverse time is variable: scaled by error magnitude so small
        deviations get a short reverse and big ones get a longer one.
        
        Recovery sequence:
        1. STOPPING:    Brake to zero (100ms)
        2. PRE_TURNING: Stop, command reverse steer angle, wait recovery_pre_turn_time
                        for wheels to physically rotate before moving
        3. REVERSING:   Go backward with fixed opposite steer (variable time)
        4. REALIGNING:  Stop, command max steering toward the curve direction,
                        wait recovery_realign_time for wheels to physically rotate
        5. RESUMING:    Brief pause, reset PID, resume forward
        
        Args:
            steering_angle: Current steering command from vision
            speed: Current speed command
            frame: Current camera frame (for debug)
            
        Returns:
            tuple: (override_steering, override_speed, is_recovering)
                   If is_recovering=True, use the override values instead.
        """
        if not self.use_curve_recovery:
            return steering_angle, speed, False

        # === HANDLE ACTIVE RECOVERY ===
        if self._recovery_state != "NONE":
            elapsed = time.time() - self._recovery_start_time

            if self._recovery_state == "STOPPING":
                # Brief stop before turning wheels (100ms)
                if elapsed > 0.1:
                    self._recovery_state = "PRE_TURNING"
                    self._recovery_start_time = time.time()
                    print(f"\033[1;97m[ Recovery ] :\033[0m \033[1;95mPRE_TURNING\033[0m "
                          f"wheels to {self._recovery_reverse_steer_angle:.0f}° "
                          f"for {self.recovery_pre_turn_time}s")
                return 0, 0, True

            elif self._recovery_state == "PRE_TURNING":
                # Car is stopped, wheels turning to reverse angle before moving
                if elapsed > self.recovery_pre_turn_time:
                    # Wheels in position → start reversing
                    self._recovery_state = "REVERSING"
                    self._recovery_start_time = time.time()
                    print(f"\033[1;97m[ Recovery ] :\033[0m \033[1;93mREVERSING\033[0m "
                          f"for {self._recovery_actual_rev_time:.2f}s at {self._recovery_reverse_steer_angle:.0f}°")
                # Send reverse steer angle with speed=0 so wheels rotate while car is still
                return self._recovery_reverse_steer_angle, 0, True

            elif self._recovery_state == "REVERSING":
                if elapsed > self._recovery_actual_rev_time:
                    # Done reversing → stop and realign wheels toward the curve
                    self._recovery_state = "REALIGNING"
                    self._recovery_start_time = time.time()
                    realign_angle = self._recovery_curve_sign * self.max_steering
                    print(f"\033[1;97m[ Recovery ] :\033[0m \033[1;96mREALIGNING\033[0m "
                          f"wheels to {realign_angle:.0f}° for {self.recovery_realign_time}s")
                else:
                    # Use pre-calculated fixed steer angle (opposite to curve, proportional to error)
                    return self._recovery_reverse_steer_angle, self.recovery_reverse_speed, True

            elif self._recovery_state == "REALIGNING":
                # Car is stopped, wheels turning to max curve angle
                realign_steer = self._recovery_curve_sign * self.max_steering
                if elapsed > self.recovery_realign_time:
                    # Wheels should be in position now → resume
                    self._recovery_state = "RESUMING"
                    self._recovery_start_time = time.time()
                    print(f"\033[1;97m[ Recovery ] :\033[0m \033[1;92mRESUMING\033[0m - Wheels aligned at {realign_steer:.0f}°")
                # Send max curve steer with speed=0 so wheels rotate while car is still
                return realign_steer, 0, True

            elif self._recovery_state == "RESUMING":
                if elapsed > 0.1:
                    # Recovery complete - reset everything
                    self._recovery_state = "NONE"
                    self._max_steer_consecutive = 0
                    self._error_at_max_steer_start = 0.0
                    self._recovery_curve_sign = 0
                    self._recovery_actual_rev_time = 0.0
                    self._recovery_reverse_steer_angle = 0.0
                    # Reset PID to prevent integral windup from recovery
                    self.pid.reset()
                    print(f"\033[1;97m[ Recovery ] :\033[0m \033[1;92mCOMPLETE\033[0m - Resuming normal operation")
                    return steering_angle, speed, False
                # Keep max curve steer while briefly stopped before resuming
                return self._recovery_curve_sign * self.max_steering, 0, True

            return 0, 0, True

        # === DETECT IF RECOVERY IS NEEDED ===
        # Guard 1: Only consider recovery when in a curve, never on straights
        if self._curve_state not in ("ENTERING", "IN_CURVE"):
            self._max_steer_consecutive = 0
            self._error_at_max_steer_start = 0.0
            return steering_angle, speed, False

        # Guard 2: Don't count frames where the noise filter just rejected data
        # (steering_angle may be stale _last_good_steering, not a real measurement)
        if self._noise_reject_count > 0:
            self._max_steer_consecutive = 0
            self._error_at_max_steer_start = 0.0
            return steering_angle, speed, False

        current_error = abs(self._last_good_error) if hasattr(self, '_last_good_error') else 0.0

        if steering_angle is not None and abs(steering_angle) >= self.max_steering * 0.9:
            # Steering at or near max
            if self._max_steer_consecutive == 0:
                # Just started hitting max steering - record the error
                self._error_at_max_steer_start = current_error

            self._max_steer_consecutive += 1

            if self._max_steer_consecutive >= self.recovery_max_steer_frames:
                # Check if error is actually decreasing → car IS correcting, no recovery needed
                # On a straight, max steering will quickly reduce the error
                # In a curve, the error stays the same or grows
                error_ratio = current_error / max(self._error_at_max_steer_start, 1.0)

                if error_ratio < self.recovery_error_shrink_ratio:
                    # Error is shrinking → max steering is working, just needs more time
                    if self._max_steer_consecutive == self.recovery_max_steer_frames:
                        print(f"\033[1;97m[ Recovery ] :\033[0m \033[1;92mSKIPPED\033[0m - "
                              f"Error shrinking ({self._error_at_max_steer_start:.0f}→{current_error:.0f}px, "
                              f"ratio={error_ratio:.2f}), car is correcting on its own")
                    return steering_angle, speed, False

                # Error NOT shrinking → truly stuck, trigger recovery
                self._recovery_state = "STOPPING"
                self._recovery_start_time = time.time()
                # Remember curve direction for REALIGNING phase: +1 right, -1 left
                self._recovery_curve_sign = 1 if steering_angle > 0 else -1

                # Variable reverse time: scale by error magnitude
                error_fraction = min(current_error / max(self.max_error_px, 1.0), 1.0)
                self._recovery_actual_rev_time = (
                    self.recovery_reverse_time_min + 
                    error_fraction * (self.recovery_reverse_time_max - self.recovery_reverse_time_min)
                )

                # Pre-calculate fixed reverse steer angle:
                # Direction: OPPOSITE to the curve (reliable 3-point turn)
                # Magnitude: proportional to error (small error = gentle, big error = aggressive)
                self._recovery_reverse_steer_angle = (
                    -self._recovery_curve_sign * error_fraction 
                    * self.max_steering * self.recovery_reverse_steer_scale
                )
                # Clamp to max steering
                self._recovery_reverse_steer_angle = max(
                    -self.max_steering, 
                    min(self.max_steering, self._recovery_reverse_steer_angle)
                )

                print(f"\033[1;97m[ Recovery ] :\033[0m \033[1;91mTRIGGERED\033[0m - "
                      f"Max steer ({steering_angle:.0f}°) for {self._max_steer_consecutive} frames. "
                      f"Error: {self._error_at_max_steer_start:.0f}→{current_error:.0f}px (ratio={error_ratio:.2f}). "
                      f"Reversing {self._recovery_actual_rev_time:.2f}s at {self._recovery_reverse_steer_angle:.0f}°, "
                      f"then realign to {self._recovery_curve_sign * self.max_steering:.0f}°")

                self._max_steer_consecutive = 0
                return 0, 0, True
        else:
            # Not at max steering - reset counter and stored error
            self._max_steer_consecutive = 0
            self._error_at_max_steer_start = 0.0

        return steering_angle, speed, False

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

        # === SWEPT PATH VISUALIZATION ===
        swept = self._swept_path_info
        if swept is not None and self.use_swept_path:
            bar_y = 58  # Below the header panel

            if swept['turn_direction'] != 0 and swept['curve_radius_cm'] > 0:
                # Curve detected - show info bar
                min_clr = swept['min_clearance_cm']
                # Color code: green (safe >5cm), yellow (tight 2-5cm), red (danger <2cm)
                if min_clr > 5:
                    bar_color = (40, 100, 40)  # Dark green bg
                    text_color = (0, 255, 0)   # Green text
                elif min_clr > 2:
                    bar_color = (20, 80, 100)  # Dark yellow bg
                    text_color = (0, 220, 255) # Yellow text
                else:
                    bar_color = (20, 20, 100)  # Dark red bg
                    text_color = (0, 80, 255)  # Red text

                cv2.rectangle(debug, (0, bar_y), (w, bar_y + 35), bar_color, -1)

                dir_str = "RIGHT" if swept['turn_direction'] > 0 else "LEFT"
                is_single = swept.get('single_line', False)
                mode_tag = "1L" if is_single else "2L"
                state_str = swept.get('curve_state', self._curve_state)
                src_str = swept.get('source', '?')
                cv2.putText(debug, f"{state_str} {dir_str} [{mode_tag}|{src_str}]  "
                            f"R={swept['curve_radius_cm']:.0f}cm  "
                            f"Clr: {min_clr:.1f}cm ({swept['critical_corner']})",
                            (8, bar_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)

                offset_cm = swept.get('offset_cm', 0)
                conf = swept.get('confidence', 0)
                fits_str = "OK" if swept.get('fits', True) else "NO FIT!"
                extra = ""
                if is_single:
                    extra = f"  Slope:{swept.get('angle_from_vertical', 0):.0f}°"
                cv2.putText(debug, f"Offset: {offset_cm:.1f}cm ({swept['offset_px']:.0f}px)  "
                            f"Conf: {conf:.0%}  {fits_str}{extra}",
                            (8, bar_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

                # Draw clearance bars for each corner
                clr = swept.get('clearances', {})
                if clr:
                    bar_x = w - 160
                    for i, (corner, val) in enumerate(clr.items()):
                        cy = bar_y + 8 + i * 8
                        # Short name
                        short = corner[:2].upper() + corner.split('_')[1][0].upper()
                        clr_color = (0, 255, 0) if val > 5 else (0, 220, 255) if val > 2 else (0, 80, 255)
                        cv2.putText(debug, f"{short}:{val:.1f}", (bar_x, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, clr_color, 1)

                # Draw the swept path offset target point (magenta)
                if is_single:
                    # Single-line mode: draw target, estimated lane boundaries
                    target_x = swept.get('target_x')
                    if target_x is not None:
                        tx = int(target_x)
                        if 0 <= tx < w:
                            cv2.circle(debug, (tx, h - 5), 6, (255, 0, 255), -1)
                            cv2.line(debug, (w // 2, h - 5), (tx, h - 5), (255, 0, 255), 1)
                    # Draw estimated invisible line (dashed, cyan)
                    est_inner = swept.get('estimated_inner_x')
                    est_outer = swept.get('estimated_outer_x')
                    if est_inner is not None:
                        eix = int(est_inner)
                        if 0 <= eix < w:
                            for y_dash in range(h - 10, int(h * 0.4), -15):
                                cv2.line(debug, (eix, y_dash), (eix, max(0, y_dash - 8)),
                                         (0, 255, 255), 1)
                    if est_outer is not None:
                        eox = int(est_outer)
                        if 0 <= eox < w:
                            for y_dash in range(h - 10, int(h * 0.4), -15):
                                cv2.line(debug, (eox, y_dash), (eox, max(0, y_dash - 8)),
                                         (0, 255, 255), 1)
                elif midpoint_x is not None and abs(swept['offset_px']) > 0.5:
                    # Two-line mode: draw offset target relative to center
                    target_x = w // 2 + int(swept['offset_px'])
                    cv2.circle(debug, (target_x, h - 5), 6, (255, 0, 255), -1)
                    cv2.line(debug, (w // 2, h - 5), (target_x, h - 5), (255, 0, 255), 1)

                # Draw vanishing point if available (two-line mode only)
                vp = swept.get('vanishing_point')
                if vp is not None:
                    vp_x, vp_y = int(vp[0]), int(vp[1])
                    if 0 <= vp_x < w and 0 <= vp_y < h:
                        cv2.drawMarker(debug, (vp_x, vp_y), (255, 255, 0),
                                       cv2.MARKER_CROSS, 15, 1)

            elif swept['turn_direction'] == 0:
                # Straight road indicator with FSM state
                cv2.rectangle(debug, (0, bar_y), (w, bar_y + 15), (40, 40, 40), -1)
                state_str = swept.get('curve_state', self._curve_state)
                cv2.putText(debug, f"{state_str} - No swept path offset",
                            (8, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 200, 150), 1)
        elif self.use_swept_path:
            # No swept info but FSM is active - show state
            bar_y = 58
            cv2.rectangle(debug, (0, bar_y), (w, bar_y + 15), (30, 30, 30), -1)
            cv2.putText(debug, f"FSM: {self._curve_state}  dir={self._curve_direction}",
                        (8, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

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

    # ================================================================
    # HYBRID CURVE SYSTEM - Context-aware curve navigation
    # ================================================================
    # Combines multiple methods for robust curve handling:
    #   STRAIGHT (2 lines):  Vanishing point estimation (works well for gentle curves)
    #   ENTERING (2→1 line): Pre-position toward inside using VP + known BFMC radii
    #   IN_CURVE (1 line):   BFMC hardcoded radius + steering feedback (most reliable)
    #   EXITING (1→2 lines): Smooth transition back to straight
    # ================================================================

    def _update_curve_state(self, num_lines, avg_left, avg_right, img_h, img_w):
        """Update the curve state machine based on current detection.
        
        State transitions:
            STRAIGHT → ENTERING:   VP shifted or 1 line starting to appear
            ENTERING → IN_CURVE:   Confirmed 1 line for several frames
            IN_CURVE → EXITING:    2 lines re-appear
            EXITING → STRAIGHT:    2 lines stable for several frames
            
        Args:
            num_lines: 0, 1, or 2 lines detected
            avg_left, avg_right: Detected lines (may be None)
            img_h, img_w: Image dimensions
        """
        prev_state = self._curve_state
        self._curve_state_frames += 1

        if self._curve_state == "STRAIGHT":
            if num_lines == 1:
                # Lost a line → entering curve
                self._curve_state = "ENTERING"
                self._curve_state_frames = 1
                self._curve_direction = 1 if avg_left is not None else -1
            elif num_lines == 2:
                # Check VP for early curve detection (pre-position)
                vp_info = self._quick_vp_check(avg_left, avg_right, img_h, img_w)
                if vp_info and abs(vp_info['offset_norm']) > self.curve_vp_threshold:
                    self._curve_state = "ENTERING"
                    self._curve_state_frames = 1
                    self._curve_direction = 1 if vp_info['offset_norm'] > 0 else -1

        elif self._curve_state == "ENTERING":
            if num_lines == 1:
                if self._curve_state_frames >= self.curve_confirm_frames:
                    self._curve_state = "IN_CURVE"
                    self._curve_state_frames = 1
                # Update direction based on which line is visible
                self._curve_direction = 1 if avg_left is not None else -1
            elif num_lines == 2:
                # Still see both lines - check if VP still indicates curve
                vp_info = self._quick_vp_check(avg_left, avg_right, img_h, img_w)
                if vp_info and abs(vp_info['offset_norm']) > self.curve_vp_threshold:
                    pass  # Stay in ENTERING
                else:
                    # False alarm - back to straight
                    self._curve_state = "STRAIGHT"
                    self._curve_state_frames = 0
                    self._curve_direction = 0
            elif num_lines == 0:
                # Lost all lines in entering - go to IN_CURVE (assume committed)
                self._curve_state = "IN_CURVE"
                self._curve_state_frames = 1

        elif self._curve_state == "IN_CURVE":
            if num_lines == 2:
                self._curve_state = "EXITING"
                self._curve_state_frames = 1
            elif num_lines == 1:
                # Update direction (might have switched which line we see)
                self._curve_direction = 1 if avg_left is not None else -1

        elif self._curve_state == "EXITING":
            if num_lines == 2:
                if self._curve_state_frames >= self.curve_exit_frames:
                    self._curve_state = "STRAIGHT"
                    self._curve_state_frames = 0
                    self._curve_direction = 0
                    self._curve_radius_estimate = 0.0
                    self._curve_confidence = 0.0
                    self._steering_radius_history.clear()
            elif num_lines <= 1:
                # Lost line again - back to IN_CURVE
                self._curve_state = "IN_CURVE"
                self._curve_state_frames = 1
                if num_lines == 1:
                    self._curve_direction = 1 if avg_left is not None else -1

        # Update radius estimate using steering feedback
        self._update_steering_radius_estimate()

        if self.show_debug and self._curve_state != prev_state:
            dir_s = {-1: "LEFT", 0: "-", 1: "RIGHT"}.get(self._curve_direction, "?")
            print(f"\033[1;97m[ Curve FSM ] :\033[0m \033[1;96m{prev_state} → {self._curve_state}\033[0m "
                  f"dir={dir_s} R={self._curve_radius_estimate:.0f}cm "
                  f"frames={self._curve_state_frames}")

    def _quick_vp_check(self, avg_left, avg_right, img_h, img_w):
        """Quick vanishing point check for early curve detection.
        
        Returns:
            dict with 'offset_norm' (-1 to 1) or None if parallel lines.
        """
        if avg_left is None or avg_right is None:
            return None
        try:
            x1_l, y1_l, x2_l, y2_l = avg_left[0]
            x1_r, y1_r, x2_r, y2_r = avg_right[0]
            dx_l, dy_l = float(x2_l - x1_l), float(y2_l - y1_l)
            dx_r, dy_r = float(x2_r - x1_r), float(y2_r - y1_r)
            det = dx_l * dy_r - dx_r * dy_l
            if abs(det) < 1e-6:
                return {'offset_norm': 0.0}
            t = ((x1_r - x1_l) * dy_r - (y1_r - y1_l) * dx_r) / det
            vp_x = x1_l + t * dx_l
            offset_norm = (vp_x - img_w / 2.0) / (img_w / 2.0)
            return {'offset_norm': max(-1.0, min(1.0, offset_norm))}
        except Exception:
            return None

    def _update_steering_radius_estimate(self):
        """Update curve radius estimate from current steering angle.
        
        Uses Ackermann: R = wheelbase / tan(steering_angle).
        Maintains a moving average for stability.
        """
        steer = self._last_steering_angle
        if abs(steer) > 2.0:  # Only meaningful for steering > 2°
            steer_rad = math.radians(abs(steer))
            if steer_rad > 0.01:
                radius = self.car_wheelbase_cm / math.tan(steer_rad)
                radius = max(20.0, min(500.0, radius))
                self._steering_radius_history.append(radius)

    def _get_best_curve_radius(self, vp_radius=None, vp_confidence=0.0):
        """Get the best curve radius estimate by fusing available sources.
        
        Priority order:
        1. IN_CURVE state → BFMC hardcoded radius (most reliable for this track)
        2. Steering feedback average (always available when turning)
        3. VP-based estimation (good for gentle curves, noisy for tight ones)
        4. Default BFMC radius (safe fallback)
        
        Returns:
            tuple: (radius_cm, confidence, source_name)
        """
        state = self._curve_state

        # Source 1: Steering feedback (moving average)
        steer_radius = None
        if len(self._steering_radius_history) >= 2:
            steer_radius = np.median(list(self._steering_radius_history))

        # IN_CURVE or EXITING: prioritize BFMC known radius, validated by steering
        if state in ("IN_CURVE", "EXITING"):
            bfmc_radius = self.bfmc_default_curve_radius

            # If steering feedback is available, use it to pick inner vs outer lane
            if steer_radius is not None:
                # Tight steering (R < 85cm) → likely inner lane
                # Wide steering (R > 85cm) → likely outer lane
                if steer_radius < 85:
                    bfmc_radius = self.bfmc_inner_lane_radius  # 66.5
                else:
                    bfmc_radius = self.bfmc_outer_lane_radius  # 103.5

            self._curve_radius_estimate = bfmc_radius
            self._curve_confidence = 0.9
            return bfmc_radius, 0.9, "BFMC+steer"

        # ENTERING: blend VP estimation with BFMC default
        if state == "ENTERING":
            if vp_radius and vp_radius > 0 and vp_confidence > 0.3:
                # Blend VP with BFMC default (VP is still somewhat reliable here)
                blended = 0.4 * vp_radius + 0.6 * self.bfmc_default_curve_radius
                blended = max(30.0, min(200.0, blended))
                self._curve_radius_estimate = blended
                self._curve_confidence = 0.7
                return blended, 0.7, "VP+BFMC"
            elif steer_radius is not None:
                self._curve_radius_estimate = steer_radius
                self._curve_confidence = 0.6
                return steer_radius, 0.6, "steer"
            else:
                self._curve_radius_estimate = self.bfmc_default_curve_radius
                self._curve_confidence = 0.5
                return self.bfmc_default_curve_radius, 0.5, "BFMC_default"

        # STRAIGHT: use VP if available, otherwise no curve
        if vp_radius and vp_radius > 0 and vp_confidence > 0.4:
            self._curve_radius_estimate = vp_radius
            self._curve_confidence = vp_confidence
            return vp_radius, vp_confidence, "VP"

        self._curve_radius_estimate = 0
        self._curve_confidence = 0
        return 0, 0, "none"

    def _calculate_swept_corners(self, rear_axle_radius_cm):
        """Calculate the turning sweep radius of each car corner.
        
        Uses Ackermann turning geometry. The rear axle center turns on a
        circle of radius R. Each corner of the car sweeps a different arc:
        - Rear inner: tightest arc (closest to turn center)
        - Front outer: widest arc (farthest from turn center)
        
        Args:
            rear_axle_radius_cm: Turn radius of rear axle center (cm).
            
        Returns:
            dict with sweep radii for each corner (cm):
                rear_inner, rear_outer, front_inner, front_outer
        """
        W2 = self.car_width / 2.0       # Half-width
        L = self.car_wheelbase_cm        # Wheelbase
        Ff = self.car_front_overhang     # Front overhang
        Fr = self.car_rear_overhang      # Rear overhang
        R = abs(rear_axle_radius_cm)

        # Each corner is offset laterally (±W/2) and longitudinally from rear axle
        # Sweep radius = distance from ICR to corner = sqrt(lateral² + longitudinal²)
        rear_inner = math.sqrt(max(0, (R - W2) ** 2 + Fr ** 2))
        rear_outer = math.sqrt((R + W2) ** 2 + Fr ** 2)
        front_inner = math.sqrt(max(0, (R - W2) ** 2 + (L + Ff) ** 2))
        front_outer = math.sqrt((R + W2) ** 2 + (L + Ff) ** 2)

        return {
            'rear_inner': rear_inner,
            'rear_outer': rear_outer,
            'front_inner': front_inner,
            'front_outer': front_outer,
        }

    def _estimate_curvature_from_lines(self, avg_left, avg_right, img_h, img_w):
        """Estimate road curvature from detected lane line geometry.
        
        Uses the vanishing point (intersection of left and right lines):
        - VP at image center → straight road
        - VP shifted left → left curve
        - VP shifted right → right curve
        - VP closer to camera → tighter curve
        
        Also calibrates px/cm using the known lane width.
        
        Args:
            avg_left: Left line np.array([[x1, y1, x2, y2]])
            avg_right: Right line np.array([[x1, y1, x2, y2]])
            img_h, img_w: Image dimensions
            
        Returns:
            dict: {
                'curve_radius_cm': float (0 = straight),
                'turn_direction': int (-1=left, 0=straight, 1=right),
                'confidence': float (0-1),
                'px_per_cm': float,
                'lane_width_px': float,
                'vanishing_point': tuple or None,
            }
        """
        x1_l, y1_l, x2_l, y2_l = avg_left[0]
        x1_r, y1_r, x2_r, y2_r = avg_right[0]

        # Direction vectors for each line
        dx_l = float(x2_l - x1_l)
        dy_l = float(y2_l - y1_l)
        dx_r = float(x2_r - x1_r)
        dy_r = float(y2_r - y1_r)

        # Lane width at bottom of image (for px/cm calibration)
        bottom_left_x = x1_l + (img_h - y1_l) * dx_l / dy_l if dy_l != 0 else float(x1_l)
        bottom_right_x = x1_r + (img_h - y1_r) * dx_r / dy_r if dy_r != 0 else float(x1_r)
        lane_width_px = abs(bottom_right_x - bottom_left_x)

        if lane_width_px < 10:
            return {
                'curve_radius_cm': 0, 'turn_direction': 0, 'confidence': 0.0,
                'px_per_cm': 1.0, 'lane_width_px': lane_width_px, 'vanishing_point': None,
            }

        px_per_cm = lane_width_px / self.lane_width_cm

        # Find vanishing point (intersection of the two lines)
        # Using parametric: P = P1 + t*(P2-P1) for each line
        det = dx_l * dy_r - dx_r * dy_l

        if abs(det) < 1e-6:
            # Lines are parallel → straight road
            return {
                'curve_radius_cm': 0, 'turn_direction': 0, 'confidence': 0.6,
                'px_per_cm': px_per_cm, 'lane_width_px': lane_width_px,
                'vanishing_point': None,
            }

        t = ((x1_r - x1_l) * dy_r - (y1_r - y1_l) * dx_r) / det
        vp_x = x1_l + t * dx_l
        vp_y = y1_l + t * dy_l

        # Vanishing point analysis
        center_x = img_w / 2.0
        vp_lateral_offset = vp_x - center_x
        vp_offset_normalized = vp_lateral_offset / (img_w / 2.0)

        # Threshold: small VP offset → essentially straight
        if abs(vp_offset_normalized) < 0.15:
            return {
                'curve_radius_cm': 0, 'turn_direction': 0, 'confidence': 0.7,
                'px_per_cm': px_per_cm, 'lane_width_px': lane_width_px,
                'vanishing_point': (vp_x, vp_y),
            }

        # Determine turn direction from VP position
        # VP shifted right → lines converge to right → right curve
        turn_direction = 1 if vp_offset_normalized > 0 else -1

        # Estimate curve radius from VP geometry
        # VP lateral offset and distance relate to arc geometry
        vp_lateral_cm = abs(vp_lateral_offset) / px_per_cm
        vp_distance_px = max(1.0, img_h - vp_y)  # VP is above, distance from bottom
        vp_distance_cm = vp_distance_px / px_per_cm

        # Approximate curve radius: R ≈ d² / (2 * lateral_offset)
        # From circular arc: for a chord of length d with sagitta h, R = d²/(2h) + h/2
        if vp_lateral_cm > 1.0:
            curve_radius_cm = (vp_distance_cm ** 2) / (2.0 * vp_lateral_cm)
            # Clamp to reasonable range for BFMC track (min ~40cm, max ~500cm)
            curve_radius_cm = max(30.0, min(500.0, curve_radius_cm))
        else:
            curve_radius_cm = 0

        # Confidence based on signal strength
        confidence = min(1.0, abs(vp_offset_normalized) / 0.5)

        # Secondary validation: check slope asymmetry
        # In a curve, inner line is more vertical than outer line
        angle_l = abs(math.degrees(math.atan2(abs(dx_l), abs(dy_l)))) if dy_l != 0 else 90
        angle_r = abs(math.degrees(math.atan2(abs(dx_r), abs(dy_r)))) if dy_r != 0 else 90
        slope_asymmetry = abs(angle_l - angle_r)

        # Strong asymmetry reinforces curve detection
        if slope_asymmetry > 15:
            confidence = min(1.0, confidence * 1.3)
        elif slope_asymmetry < 5:
            confidence *= 0.7  # Lines look parallel, less confident in curve

        return {
            'curve_radius_cm': curve_radius_cm,
            'turn_direction': turn_direction,
            'confidence': confidence,
            'px_per_cm': px_per_cm,
            'lane_width_px': lane_width_px,
            'vanishing_point': (vp_x, vp_y),
            'slope_asymmetry': slope_asymmetry,
        }

    def _calculate_curve_offset(self, curve_radius_cm, turn_direction, lane_width_cm=None):
        """Calculate optimal lateral offset to maximize clearance in a curve.
        
        Searches for the rear axle radius that gives the most clearance
        from both lane boundaries, considering all 4 corners of the car.
        
        Args:
            curve_radius_cm: Radius to LANE CENTER from turn center (cm)
            turn_direction: -1 for left turn, 1 for right turn
            lane_width_cm: Override lane width (uses default if None)
            
        Returns:
            dict: {
                'offset_cm': float - offset from lane center (negative=inside, positive=outside),
                'min_clearance_cm': float - minimum clearance at optimal position,
                'critical_corner': str - which corner is the constraint,
                'optimal_radius_cm': float - optimal rear axle radius,
                'clearances': dict - clearance for each corner,
                'fits': bool - whether the car can physically fit,
            }
        """
        if curve_radius_cm <= 0 or turn_direction == 0:
            return {
                'offset_cm': 0.0, 'min_clearance_cm': float('inf'),
                'critical_corner': 'none', 'optimal_radius_cm': 0,
                'clearances': {}, 'fits': True,
            }

        lw = lane_width_cm if lane_width_cm else self.lane_width_cm
        margin = self.swept_path_margin_cm

        # Lane boundaries as radii from the turn center
        inner_line_r = curve_radius_cm - lw / 2.0
        outer_line_r = curve_radius_cm + lw / 2.0

        if inner_line_r < 5:
            inner_line_r = 5  # Safety floor

        # Search range for rear axle position
        W2 = self.car_width / 2.0
        R_min = inner_line_r + W2 + 1.0  # Minimum viable radius
        R_max = outer_line_r - W2 - 1.0  # Maximum viable radius

        car_fits = R_min < R_max

        if not car_fits:
            # Car physically cannot fit - return center position with negative clearance
            R_center = (inner_line_r + outer_line_r) / 2.0
            corners = self._calculate_swept_corners(R_center)
            return {
                'offset_cm': 0.0,
                'min_clearance_cm': min(
                    corners['rear_inner'] - inner_line_r,
                    outer_line_r - corners['front_outer']
                ),
                'critical_corner': 'car_too_wide',
                'optimal_radius_cm': R_center,
                'clearances': {
                    'rear_inner': corners['rear_inner'] - inner_line_r,
                    'rear_outer': outer_line_r - corners['rear_outer'],
                    'front_inner': corners['front_inner'] - inner_line_r,
                    'front_outer': outer_line_r - corners['front_outer'],
                },
                'fits': False,
            }

        # Search for the optimal rear axle radius (maximize minimum clearance)
        best_R = (R_min + R_max) / 2.0
        best_min_clear = -999.0
        best_clearances = {}
        best_critical = ""

        for R_test in np.linspace(R_min, R_max, 50):
            corners = self._calculate_swept_corners(R_test)

            clr = {
                'rear_inner': corners['rear_inner'] - inner_line_r - margin,
                'rear_outer': outer_line_r - corners['rear_outer'] - margin,
                'front_inner': corners['front_inner'] - inner_line_r - margin,
                'front_outer': outer_line_r - corners['front_outer'] - margin,
            }

            min_clr = min(clr.values())

            if min_clr > best_min_clear:
                best_min_clear = min_clr
                best_R = R_test
                best_clearances = clr
                best_critical = min(clr, key=clr.get)

        # Offset from lane center (negative = shift toward inside of curve)
        lane_center_r = (inner_line_r + outer_line_r) / 2.0
        offset_cm = (best_R - lane_center_r) * self.curve_offset_gain

        return {
            'offset_cm': offset_cm,
            'min_clearance_cm': best_min_clear,
            'critical_corner': best_critical,
            'optimal_radius_cm': best_R,
            'clearances': best_clearances,
            'fits': True,
        }

    def _predict_swept_path(self, avg_left, avg_right, img_h, img_w):
        """HYBRID swept path prediction for 2-line detection.
        
        Uses the curve state machine to pick the best radius source:
        - STRAIGHT/ENTERING: VP estimation (works well when both lines visible)
        - IN_CURVE/EXITING: BFMC known radius + steering feedback
        
        Args:
            avg_left, avg_right: Both lane lines
            img_h, img_w: Image dimensions
            
        Returns:
            dict or None with offset_px and curve info.
        """
        try:
            # Step 1: VP-based estimation (always compute for data/caching)
            curve = self._estimate_curvature_from_lines(avg_left, avg_right, img_h, img_w)

            # Cache px_per_cm for single-line mode
            if curve['px_per_cm'] > 0.5:
                self._last_px_per_cm = curve['px_per_cm']

            # Step 2: Get best radius from hybrid fusion
            best_radius, confidence, source = self._get_best_curve_radius(
                vp_radius=curve['curve_radius_cm'],
                vp_confidence=curve['confidence']
            )

            # Use curve direction from VP if available, else from state machine
            turn_direction = curve['turn_direction']
            if turn_direction == 0 and self._curve_direction != 0:
                turn_direction = self._curve_direction

            if best_radius <= 0 or turn_direction == 0:
                return {
                    'offset_px': 0.0, 'curve_radius_cm': 0,
                    'turn_direction': 0, 'min_clearance_cm': float('inf'),
                    'critical_corner': 'none', 'confidence': confidence,
                    'px_per_cm': curve['px_per_cm'], 'clearances': {},
                    'fits': True, 'source': source,
                    'curve_state': self._curve_state,
                }

            if confidence < self.curve_confidence_threshold:
                return {
                    'offset_px': 0.0, 'curve_radius_cm': best_radius,
                    'turn_direction': turn_direction,
                    'min_clearance_cm': float('inf'),
                    'critical_corner': 'low_confidence',
                    'confidence': confidence,
                    'px_per_cm': curve['px_per_cm'], 'clearances': {},
                    'fits': True, 'source': source,
                    'curve_state': self._curve_state,
                }

            # Step 3: Calculate swept path offset
            offset_result = self._calculate_curve_offset(best_radius, turn_direction)

            # Step 4: Convert to pixels
            # Apply pre-position gain when ENTERING (less aggressive than full curve)
            gain = self.curve_pre_position_gain if self._curve_state == "ENTERING" else 1.0
            offset_px = -offset_result['offset_cm'] * curve['px_per_cm'] * turn_direction * gain

            if self.show_debug:
                print(f"\033[1;97m[ Hybrid 2L ] :\033[0m \033[1;96m{self._curve_state}\033[0m "
                      f"src={source} R={best_radius:.0f}cm dir={'R' if turn_direction > 0 else 'L'} "
                      f"offset={offset_result['offset_cm']:.1f}cm→{offset_px:.0f}px "
                      f"clr={offset_result['min_clearance_cm']:.1f}cm conf={confidence:.2f}")

            return {
                'offset_px': offset_px,
                'curve_radius_cm': best_radius,
                'turn_direction': turn_direction,
                'min_clearance_cm': offset_result['min_clearance_cm'],
                'critical_corner': offset_result['critical_corner'],
                'confidence': confidence,
                'px_per_cm': curve['px_per_cm'],
                'clearances': offset_result['clearances'],
                'fits': offset_result['fits'],
                'offset_cm': offset_result['offset_cm'],
                'optimal_radius_cm': offset_result['optimal_radius_cm'],
                'vanishing_point': curve.get('vanishing_point'),
                'source': source,
                'curve_state': self._curve_state,
            }

        except Exception as e:
            if self.show_debug:
                print(f"\033[1;97m[ Hybrid 2L ] :\033[0m \033[1;91mERROR\033[0m - {e}")
            return None

    def _predict_swept_path_single_line(self, line, side, img_h, img_w):
        """HYBRID swept path prediction for single-line detection (in a curve).
        
        Uses the curve state machine to get the best radius:
        - IN_CURVE: BFMC hardcoded radius (most reliable)
        - ENTERING: Blend of slope estimate + BFMC default
        - Steering feedback as validation
        
        Then reconstructs the invisible lane boundary using known lane width,
        and calculates the optimal trajectory with swept path geometry.
        
        Args:
            line: Detected line np.array([[x1, y1, x2, y2]])
            side: 'left' or 'right' - which line is visible
            img_h, img_w: Image dimensions
            
        Returns:
            dict or None with steering_error, offset info, and geometry data.
        """
        try:
            x1, y1, x2, y2 = line[0]
            dx = float(x2 - x1)
            dy = float(y2 - y1)

            if abs(dy) < 1:
                return None  # Nearly horizontal - unreliable

            turn_direction = 1 if side == 'left' else -1
            angle_from_vertical = abs(math.degrees(math.atan(dx / dy)))

            # --- HYBRID RADIUS ESTIMATION ---
            # Source 1: Slope-based estimate (original method, fallback)
            slope_radius = None
            if angle_from_vertical > 3:
                slope_radius = self.single_line_radius_k / math.tan(
                    math.radians(angle_from_vertical))
                slope_radius = max(30.0, min(300.0, slope_radius))

            # Source 2: BFMC known radius from state machine
            best_radius, confidence, source = self._get_best_curve_radius(
                vp_radius=slope_radius, vp_confidence=0.3
            )

            # If state machine didn't provide a radius, use slope estimate
            if best_radius <= 0:
                if slope_radius and slope_radius > 0:
                    best_radius = slope_radius
                    confidence = 0.4
                    source = "slope"
                else:
                    return None  # Can't estimate anything

            # --- LANE RECONSTRUCTION ---
            px_per_cm = self._last_px_per_cm
            if px_per_cm is None or px_per_cm < 0.5:
                px_per_cm = img_w / 45.0  # Fallback

            line_bottom_x = x1 + (img_h - y1) * dx / dy
            lane_width_px = self.lane_width_cm * px_per_cm

            if side == 'left':
                outer_x = line_bottom_x
                inner_x = line_bottom_x + lane_width_px
            else:
                outer_x = line_bottom_x
                inner_x = line_bottom_x - lane_width_px

            lane_center_x = (inner_x + outer_x) / 2.0

            # --- SWEPT PATH OFFSET ---
            offset_result = self._calculate_curve_offset(best_radius, turn_direction)
            offset_px = -offset_result['offset_cm'] * px_per_cm * turn_direction

            target_x = lane_center_x + offset_px
            steering_error = target_x - (img_w / 2.0)

            if self.show_debug:
                dir_str = "R" if turn_direction > 0 else "L"
                print(f"\033[1;97m[ Hybrid 1L ] :\033[0m \033[1;93m{self._curve_state} {dir_str}\033[0m "
                      f"src={source} R={best_radius:.0f}cm angle={angle_from_vertical:.1f}° "
                      f"offset={offset_result['offset_cm']:.1f}cm→{offset_px:.0f}px "
                      f"target={target_x:.0f} error={steering_error:.0f}px "
                      f"clr={offset_result['min_clearance_cm']:.1f}cm")

            return {
                'steering_error': steering_error,
                'offset_px': offset_px,
                'offset_cm': offset_result['offset_cm'],
                'curve_radius_cm': best_radius,
                'turn_direction': turn_direction,
                'min_clearance_cm': offset_result['min_clearance_cm'],
                'critical_corner': offset_result['critical_corner'],
                'confidence': confidence,
                'px_per_cm': px_per_cm,
                'clearances': offset_result.get('clearances', {}),
                'fits': offset_result.get('fits', True),
                'target_x': target_x,
                'estimated_inner_x': inner_x,
                'estimated_outer_x': outer_x,
                'angle_from_vertical': angle_from_vertical,
                'single_line': True,
                'source': source,
                'curve_state': self._curve_state,
            }

        except Exception as e:
            if self.show_debug:
                print(f"\033[1;97m[ Hybrid 1L ] :\033[0m \033[1;91mERROR\033[0m - {e}")
            return None

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
            # String parameters (URLs, mode names) - do NOT convert to number
            string_params = {'detection_mode', 'hybridnets_server_url', 'supercombo_server_url'}
            # Integer parameters - must be int for OpenCV kernels, thresholds, etc.
            int_params = {'blur_kernel', 'morph_kernel', 'canny_low', 'canny_high',
                         'hough_threshold', 'hough_min_line_length', 'hough_max_line_gap',
                         'white_h_min', 'white_h_max', 'white_s_min', 'white_s_max',
                         'white_v_min', 'white_v_max', 'yellow_h_min', 'yellow_h_max',
                         'yellow_s_min', 'yellow_s_max', 'yellow_v_min', 'yellow_v_max',
                         'binary_threshold', 'binary_threshold_retry', 'line_angle_filter',
                         'line_merge_distance', 'lstr_model_size', 'stream_debug_view',
                         'stream_debug_fps', 'stream_debug_quality', 'use_clahe',
                         'use_adaptive_white', 'use_gradient_fallback', 'clahe_grid_size',
                         'integral_reset_interval', 'hybridnets_jpeg_quality',
                         'supercombo_jpeg_quality', 'brightness',
                         'use_swept_path', 'curve_speed_reduction',
                         'curve_enter_frames', 'curve_confirm_frames', 'curve_exit_frames',
                         'use_noise_filter', 'noise_max_hough_lines',
                         'noise_max_reject_frames',
                         'use_curve_recovery', 'recovery_max_steer_frames'}
            
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
                     'supercombo_server_url', 'supercombo_jpeg_quality', 'supercombo_timeout',
                     # Swept path / car geometry parameters
                     'use_swept_path', 'car_length', 'car_width', 'car_wheelbase_cm',
                     'car_front_overhang', 'car_rear_overhang',
                     'camera_to_front_axle', 'camera_to_rear_axle',
                     'lane_width_cm', 'line_width_cm',
                     'swept_path_margin_cm', 'curve_offset_gain',
                     'curve_confidence_threshold', 'min_clearance_warn_cm',
                     'curve_speed_reduction', 'single_line_radius_k',
                     # Hybrid curve system parameters
                     'bfmc_inner_lane_radius', 'bfmc_outer_lane_radius',
                     'bfmc_default_curve_radius',
                     'curve_enter_frames', 'curve_confirm_frames', 'curve_exit_frames',
                     'curve_vp_threshold', 'curve_pre_position_gain',
                     # Noise filter parameters
                     'use_noise_filter', 'noise_max_hough_lines',
                     'noise_max_error_jump_px', 'noise_max_steer_jump_deg',
                     'noise_max_reject_frames',
                     # Curve recovery parameters
                     'use_curve_recovery', 'recovery_max_steer_frames',
                     'recovery_reverse_speed', 'recovery_reverse_time_min',
                     'recovery_reverse_time_max', 'recovery_reverse_steer_scale',
                     'recovery_pre_turn_time', 'recovery_realign_time',
                     'recovery_error_shrink_ratio']
            
            applied = []
            for param in params:
                if param in config:
                    value = config[param]
                    # Type coercion: frontend range inputs may send strings
                    if param in string_params:
                        value = str(value)
                    elif param in int_params:
                        try:
                            value = int(float(value))
                        except (ValueError, TypeError):
                            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mTYPE ERROR\033[0m - Cannot convert {param}={value!r} to int")
                            continue
                    else:
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mTYPE ERROR\033[0m - Cannot convert {param}={value!r} to float")
                            continue
                    setattr(self, param, value)
                    applied.append(param)
            
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
            
            # Log swept path / car geometry parameter changes
            swept_params = ['use_swept_path', 'car_length', 'car_width', 'car_wheelbase_cm',
                           'car_front_overhang', 'car_rear_overhang', 'lane_width_cm',
                           'swept_path_margin_cm', 'curve_offset_gain', 'curve_confidence_threshold',
                           'min_clearance_warn_cm', 'curve_speed_reduction']
            if any(p in config for p in swept_params):
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;95mSWEPT PATH\033[0m - Updated: "
                      f"enabled={self.use_swept_path} car={self.car_length}x{self.car_width}cm "
                      f"wb={self.car_wheelbase_cm}cm ovh_f={self.car_front_overhang} ovh_r={self.car_rear_overhang} "
                      f"lane={self.lane_width_cm}cm margin={self.swept_path_margin_cm}cm "
                      f"gain={self.curve_offset_gain} conf_thr={self.curve_confidence_threshold}")
            
            # Log hybrid curve system parameter changes
            hybrid_params = ['bfmc_inner_lane_radius', 'bfmc_outer_lane_radius',
                            'bfmc_default_curve_radius', 'curve_enter_frames',
                            'curve_confirm_frames', 'curve_exit_frames',
                            'curve_vp_threshold', 'curve_pre_position_gain']
            if any(p in config for p in hybrid_params):
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;95mHYBRID CURVE\033[0m - Updated: "
                      f"BFMC_R: inner={self.bfmc_inner_lane_radius} outer={self.bfmc_outer_lane_radius} "
                      f"default={self.bfmc_default_curve_radius} "
                      f"FSM: enter={self.curve_enter_frames} confirm={self.curve_confirm_frames} "
                      f"exit={self.curve_exit_frames} vp_thr={self.curve_vp_threshold} "
                      f"pre_gain={self.curve_pre_position_gain}")
            
            # Log curve recovery parameter changes
            recovery_params = ['use_curve_recovery', 'recovery_max_steer_frames',
                              'recovery_reverse_speed', 'recovery_reverse_time_min',
                              'recovery_reverse_time_max', 'recovery_reverse_steer_scale',
                              'recovery_realign_time', 'recovery_error_shrink_ratio']
            if any(p in config for p in recovery_params):
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;95mRECOVERY\033[0m - Updated: "
                      f"enabled={self.use_curve_recovery} max_frames={self.recovery_max_steer_frames} "
                      f"rev_speed={self.recovery_reverse_speed} "
                      f"rev_time={self.recovery_reverse_time_min}-{self.recovery_reverse_time_max}s "
                      f"steer_scale={self.recovery_reverse_steer_scale} realign={self.recovery_realign_time}s "
                      f"shrink_ratio={self.recovery_error_shrink_ratio}")
            
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
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;92mCONFIG\033[0m - Updated {len(applied)} params: {', '.join(applied[:8])}{'...' if len(applied) > 8 else ''}")
        except Exception as e:
            import traceback
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mERROR\033[0m - Config error: {e}")
            traceback.print_exc()

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
                # Check curve recovery FIRST - overrides normal commands if active
                rec_steer, rec_speed, is_recovering = self._check_curve_recovery(
                    steering_angle if steering_angle is not None else 0, 
                    speed, frame)
                
                if is_recovering:
                    self.send_motor_commands(rec_steer, rec_speed)
                    self.frames_without_line = 0
                    # Draw recovery status on debug frame
                    if debug_frame is not None:
                        cv2.putText(debug_frame, f"RECOVERY: {self._recovery_state}", 
                                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif steering_angle is not None:
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
        
       

        # Preprocess: apply CLAHE and brightness/contrast before detection
        preprocessed, preprocess_debug = self._preprocess_frame(frame)
        
        # Store debug images from preprocessing
        if 'clahe' in preprocess_debug:
            self._store_debug_image('clahe', preprocess_debug['clahe'])
        if 'adjusted' in preprocess_debug:
            self._store_debug_image('adjusted', preprocess_debug['adjusted'])

        # First attempt with normal threshold
        avg_left, avg_right, img_h, img_w, canny, debug_info = self._bfmc_image_processing(preprocessed)

        # BFMC retry: if no lines detected, try lower threshold (from MarcosLaneDetector)
        if avg_left is None and avg_right is None:
            avg_left, avg_right, img_h, img_w, canny, debug_info = self._bfmc_image_processing(
                preprocessed,
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

        # === PRE-CHECK: Reject obviously noisy frames (reflection/glare) ===
        num_lines = (1 if avg_left is not None else 0) + (1 if avg_right is not None else 0)
        _pre_noisy, _pre_reason = self._is_frame_noisy(debug_info, None, None, num_lines)
        if _pre_noisy:
            # Frame is noisy (e.g. too many Hough lines from reflections)
            # Use previous good steering and skip all processing
            self._noise_reject_count += 1
            if self.show_debug:
                print(f"\033[1;97m[ Noise Filter ] :\033[0m \033[1;91mREJECT\033[0m "
                      f"({self._noise_reject_count}/{self.noise_max_reject_frames}) "
                      f"reason={_pre_reason}")
            steering_angle = self._last_good_steering
            speed = self.min_speed  # Slow down during noise
            # Create debug frame showing rejection
            debug_frame = self._bfmc_draw_debug(
                frame, avg_left, avg_right, None, None, None, None, steering_angle, debug_info
            )
            cv2.rectangle(debug_frame, (0, debug_frame.shape[0] - 25),
                         (debug_frame.shape[1], debug_frame.shape[0]), (0, 0, 150), -1)
            cv2.putText(debug_frame, f"NOISE REJECT: {_pre_reason}",
                       (8, debug_frame.shape[0] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
            self._store_debug_image('final', debug_frame)
            self._send_debug_stream(steering_angle, speed)
            return steering_angle, speed, debug_frame

        # === UPDATE CURVE STATE MACHINE ===
        if self.use_swept_path:
            self._update_curve_state(num_lines, avg_left, avg_right, img_h, img_w)

        if avg_left is not None and avg_right is not None:
            # ---- BOTH LINES DETECTED ----
            # BFMC logic: if just_seen_two_lines was False, skip this frame (stabilize)
            if self.just_seen_two_lines:
                error, midpoint_x, bottom_left_x, bottom_right_x = self._bfmc_get_error(
                    avg_left, avg_right, img_h, img_w
                )

                # === SWEPT PATH CURVE PREDICTION ===
                # Use real car dimensions to predict if the car fits in the curve.
                # Adjusts the PID target laterally to maximize clearance from lane lines.
                if self.use_swept_path:
                    swept = self._predict_swept_path(avg_left, avg_right, img_h, img_w)
                    self._swept_path_info = swept
                    if swept is not None and abs(swept['offset_px']) > 0.5:
                        error += swept['offset_px']
                        if self.show_debug:
                            dir_str = "R" if swept['turn_direction'] > 0 else "L" if swept['turn_direction'] < 0 else "-"
                            print(f"\033[1;97m[ Swept Path ] :\033[0m \033[1;95mCURVE {dir_str}\033[0m - "
                                  f"R={swept['curve_radius_cm']:.0f}cm "
                                  f"offset={swept.get('offset_cm', 0):.1f}cm ({swept['offset_px']:.1f}px) "
                                  f"clearance={swept['min_clearance_cm']:.1f}cm "
                                  f"crit={swept['critical_corner']} "
                                  f"conf={swept['confidence']:.2f}")
                        # Speed reduction when clearance is tight
                        if self.curve_speed_reduction and swept['min_clearance_cm'] < self.min_clearance_warn_cm:
                            speed = self.min_speed
                else:
                    self._swept_path_info = None

                # PID control with (possibly adjusted) pixel error
                steering_angle = self.pid.compute(error)

                # Moving average of last N steering values (smooths erratic readings from lighting)
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
            # Left line visible → likely RIGHT curve (left = outer boundary)
            self.consecutive_single_left += 1
            self.consecutive_single_right = 0
            self.last_seen_side = "left"

            if self.use_swept_path:
                # Use swept path geometry: estimate curve + invisible line + optimal path
                swept = self._predict_swept_path_single_line(avg_left, 'left', img_h, img_w)
                self._swept_path_info = swept
                if swept is not None:
                    # Use PID with geometry-calculated error instead of fixed max steering
                    steering_angle = self.pid.compute(swept['steering_error'])
                    steering_angle = max(-self.max_steering, min(self.max_steering, steering_angle))
                    # Speed based on clearance
                    if self.curve_speed_reduction and swept['min_clearance_cm'] < self.min_clearance_warn_cm:
                        speed = self.min_speed
                    else:
                        speed = self.min_speed + 3  # Slightly above min for curves
                else:
                    # Swept path failed → fallback to original behavior
                    self._swept_path_info = None
                    if self.consecutive_single_left >= 2:
                        steering_angle = self.max_steering
                        speed = self.min_speed
                    else:
                        steering_angle = self._bfmc_follow_single_line(avg_left, 'left')
            else:
                # Original BFMC behavior (no swept path)
                self._swept_path_info = None
                if self.consecutive_single_left >= 2:
                    steering_angle = self.max_steering
                    speed = self.min_speed
                else:
                    steering_angle = self._bfmc_follow_single_line(avg_left, 'left')

            self.frames_without_line = 0

        elif avg_right is not None:
            # ---- ONLY RIGHT LINE ----
            # Right line visible → likely LEFT curve (right = outer boundary)
            self.consecutive_single_right += 1
            self.consecutive_single_left = 0
            self.last_seen_side = "right"

            if self.use_swept_path:
                # Use swept path geometry: estimate curve + invisible line + optimal path
                swept = self._predict_swept_path_single_line(avg_right, 'right', img_h, img_w)
                self._swept_path_info = swept
                if swept is not None:
                    steering_angle = self.pid.compute(swept['steering_error'])
                    steering_angle = max(-self.max_steering, min(self.max_steering, steering_angle))
                    if self.curve_speed_reduction and swept['min_clearance_cm'] < self.min_clearance_warn_cm:
                        speed = self.min_speed
                    else:
                        speed = self.min_speed + 3
                else:
                    self._swept_path_info = None
                    if self.consecutive_single_right >= 2:
                        steering_angle = -self.max_steering
                        speed = self.min_speed
                    else:
                        steering_angle = self._bfmc_follow_single_line(avg_right, 'right')
            else:
                self._swept_path_info = None
                if self.consecutive_single_right >= 2:
                    steering_angle = -self.max_steering
                    speed = self.min_speed
                else:
                    steering_angle = self._bfmc_follow_single_line(avg_right, 'right')

            self.frames_without_line = 0

        else:
            # ---- NO LINES ----
            self._swept_path_info = None  # Reset swept path
            self.frames_without_line += 1

            if self.show_debug:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mNO LANES\033[0m - frame {self.frames_without_line}")

            if hasattr(self, 'last_steering') and self.frames_without_line < 5:
                steering_angle = self.last_steering * 0.95
            else:
                steering_angle = None
            speed = self.min_speed

        # === POST-CHECK: Reject steering if it looks like a noise spike ===
        if steering_angle is not None and self.use_noise_filter:
            _post_noisy, _post_reason = self._is_frame_noisy(
                debug_info, error, steering_angle, num_lines
            )
            if _post_noisy:
                self._noise_reject_count += 1
                if self.show_debug:
                    print(f"\033[1;97m[ Noise Filter ] :\033[0m \033[1;93mSPIKE\033[0m "
                          f"({self._noise_reject_count}/{self.noise_max_reject_frames}) "
                          f"reason={_post_reason} steer={steering_angle:.1f}→{self._last_good_steering:.1f}")
                steering_angle = self._last_good_steering
                speed = max(speed, self.min_speed)  # Don't accelerate on rejected frame
            else:
                # Good frame - update reference
                self._accept_frame(error, steering_angle, num_lines)

        # Update state
        if steering_angle is not None:
            self.last_steering = steering_angle
            self._last_steering_angle = steering_angle  # For hybrid curve system

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
