# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.a
# Reconstructed from bytecode

import cv2
import numpy as np
import base64
import time
from src.utils.messages.allMessages import serialCamera, SpeedMotor, SteerMotor, StateChange, LineFollowingConfig
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.templates.threadwithstop import ThreadWithStop
from src.statemachine.systemMode import SystemMode


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
        self.steering_sensitivity = 3

        # PID parameters
        self.kp = 4
        self.kd = 0.05
        self.previous_error = 0
        self.smoothing_factor = 0.3
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

        print("\033[1;97m[ Line Following ] :\033[0m \033[1;92mINFO\033[0m - Line following thread initialized")
        print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;92mINFO\033[0m - Debug mode: {self.show_debug}")
        print("\033[1;97m[ Line Following ] :\033[0m \033[1;93mWARNING\033[0m - Windows will open when camera starts streaming")

    def _update_hsv_arrays(self):
        """Update NumPy arrays for HSV thresholds."""
        self.white_lower = np.array([self.white_h_min, self.white_s_min, self.white_v_min])
        self.white_upper = np.array([self.white_h_max, self.white_s_max, self.white_v_max])
        self.yellow_lower = np.array([self.yellow_h_min, self.yellow_s_min, self.yellow_v_min])
        self.yellow_upper = np.array([self.yellow_h_max, self.yellow_s_max, self.yellow_v_max])

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
                     'steering_sensitivity', 'roi_height_start', 'roi_height_end',
                     'roi_width_margin_top', 'roi_width_margin_bottom',
                     'white_h_min', 'white_h_max', 'white_s_min', 'white_s_max',
                     'white_v_min', 'white_v_max', 'yellow_h_min', 'yellow_h_max',
                     'yellow_s_min', 'yellow_s_max', 'yellow_v_min', 'yellow_v_max',
                     'brightness', 'contrast', 'blur_kernel', 'morph_kernel',
                     'canny_low', 'canny_high', 'hough_threshold', 'hough_min_line_length',
                     'hough_max_line_gap']
            
            for param in params:
                if param in config:
                    setattr(self, param, config[param])
            
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
                    if self.frames_without_line > self.max_frames_without_line:
                        self.send_motor_commands(0, self.min_speed)
            
            if self.show_debug:
                if debug_frame is not None:
                    status_text = "ACTIVE" if self.is_line_following_active else "INACTIVE (Debug Mode)"
                    cv2.putText(debug_frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                               (0, 255, 0) if self.is_line_following_active else (0, 0, 255), 2)
                    cv2.imshow("1. Final Result", debug_frame)
                    cv2.waitKey(1)
        except Exception as e:
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mERROR\033[0m - {e}")

    def process_frame(self, frame):
        """Process frame using sliding window + polynomial fit for better curve detection.

Returns:
    tuple: (steering_angle, speed, debug_frame)
"""
        height, width = frame.shape[:2]
        
        if self.show_debug:
            print(f"\033[1;97m[ Line Following ] :\033[0m Frame received: {width}x{height}")
        
        if not self.perspective_initialized:
            self._init_perspective_transform(width, height)
        
        orig_display = frame.copy()
        
        # Brightness and contrast adjustment
        adjusted = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)
        
        if self.show_debug:
            adj_display = adjusted.copy()
            cv2.putText(adj_display, f"Brightness: {self.brightness} Contrast: {self.contrast:.2f}",
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("2. Brightness/Contrast", adj_display)
        
        # Convert to HSV
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
        
        if self.show_debug:
            cv2.imshow("3. HSV", hsv)
        
        # Create masks for white and yellow
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        if self.show_debug:
            white_display = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(white_display, f"H:{self.white_h_min}-{self.white_h_max}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(white_display, f"S:{self.white_s_min}-{self.white_s_max}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(white_display, f"V:{self.white_v_min}-{self.white_v_max}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.imshow("4. White Mask", white_display)
            
            yellow_display = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(yellow_display, f"H:{self.yellow_h_min}-{self.yellow_h_max}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(yellow_display, f"S:{self.yellow_s_min}-{self.yellow_s_max}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(yellow_display, f"V:{self.yellow_v_min}-{self.yellow_v_max}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.imshow("5. Yellow Mask", yellow_display)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        if self.show_debug:
            cv2.imshow("6. Combined (Raw)", combined_mask)
        
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
        
        # Find lane pixels using sliding window
        leftx, lefty, rightx, righty, sliding_window_img = self.find_lane_pixels_sliding_window(binary_warped)
        
        if self.show_debug:
            cv2.imshow("10. Sliding Window", sliding_window_img)
        
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
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;96mERROR\033[0m - error:{error:.1f} ratio:{error_ratio:.1f} [{intensity_names[curve_intensity]}]")
            
            # PID control
            if self.is_line_following_active:
                total_correction = self.single_line_correction if (left_fit is None or right_fit is None) else 1
            else:
                total_correction = 1
            
            proportional = self.kp * error * total_correction * curve_multiplier
            derivative = self.kd * (error - self.previous_error)
            self.previous_error = error
            
            steering_raw = proportional + derivative
            
            # Apply smoothing
            smooth = curve_smooth if is_sharp_curve else self.smoothing_factor
            if hasattr(self, 'last_steering'):
                steering_angle = smooth * steering_raw + (1 - smooth) * self.last_steering
            else:
                steering_angle = steering_raw
            
            # Apply sensitivity and clamp
            steering_angle = steering_angle * self.steering_sensitivity
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
            # No lanes detected
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
        """Send steering and speed commands to the motors."""
        try:
            self.steerMotorSender.send({"Type": "Steer", "value": float(steering_angle)})
            self.speedMotorSender.send({"Type": "Speed", "value": float(speed)})
            
            if self.debugger:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;94mCMD\033[0m - Steer: {steering_angle:.1f} Speed: {speed:.1f}")
                self.logger.info(f"Line Following - Steering: {steering_angle:.1f}, Speed: {speed:.1f}")
        except Exception as e:
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mERROR\033[0m - Failed to send motor commands: {e}")

    def check_state_change(self):
        """Check for state changes and enable/disable line following accordingly."""
        message = self.stateChangeSubscriber.receive()
        if message is not None:
            mode_dict = SystemMode[message].value.get("camera", {}).get("lineFollowing", {})
            
            if mode_dict.get("enabled", False):
                if not self.is_line_following_active:
                    self.is_line_following_active = True
                    print("\033[1;97m[ Line Following ] :\033[0m \033[1;92mINFO\033[0m - Line following ACTIVATED")
            else:
                if self.is_line_following_active:
                    self.is_line_following_active = False
                    print("\033[1;97m[ Line Following ] :\033[0m \033[1;93mWARNING\033[0m - Line following DEACTIVATED")

    def stop(self):
        """Stop the thread and cleanup."""
        if self.show_debug:
            cv2.destroyAllWindows()
        super(threadLineFollowing, self).stop()
