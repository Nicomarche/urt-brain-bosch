# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.a
# Reconstructed from bytecode

import ast
import cv2
import numpy as np
import base64
import time
import math
import json
import os
from collections import deque
from enum import Enum

try:
    from scipy.optimize import minimize as _scipy_minimize
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
import config as _config
from src.utils.messages.allMessages import SpeedMotor, SteerMotor, StateChange, LineFollowingConfig, LineFollowingDebug, LineFollowingStatus, ImuData, CurrentSpeed, CurrentSteer, LocalLanePerception, LocalPerceptionStatus, ActuatorCommandStatus, SignDetected, LaneCalibMode
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.templates.threadwithstop import ThreadWithStop
from src.statemachine.systemMode import SystemMode
from src.hardware.camera.threads.maneuverManager import ManeuverManager

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
    AI_LOCAL = "ai_local"      # Local YOLO lane perception (best.pt) + local BFMC/Stanley control
    HYBRIDNETS = "hybridnets"  # Legacy/deprecated alias preserved for compatibility
    SUPERCOMBO = "supercombo"  # Legacy/deprecated alias preserved for compatibility


class ParkingState(Enum):
    """States for the autonomous parallel-reverse (de culata) parking maneuver.

    Full sequence (spot assumed to be on the RIGHT side of the lane):
      FORWARD_PAST_SPOT  → advance past spot (straight)
      WAIT_STEER_1       → stop + pre-steer wheels RIGHT (entry side) before reversing
      REVERSING_ENTRY    → reverse with max RIGHT steer (swing rear into spot)
      WAIT_STEER_2       → stop + pre-steer wheels LEFT (opposite) before 2nd reverse
      REVERSING_ALIGN    → reverse with max LEFT steer (straighten inside spot)
      WAIT_STEER_3       → stop + pre-steer wheels RIGHT again before forward correction
      FORWARD_CORRECTION → go forward with max RIGHT steer (same as entry = parallel alignment)
      PARKED             → hold still, maneuver complete
    """
    IDLE               = "idle"
    LANE_KEEPING       = "lane_keeping"
    SPOT_TRACKED       = "spot_tracked"
    FORWARD_PAST_SPOT  = "forward_past_spot"
    WAIT_STEER_1       = "wait_steer_1"       # Stopped, wheels turning to entry steer
    REVERSING_ENTRY    = "reversing_entry"    # Reverse with max entry steer
    WAIT_STEER_2       = "wait_steer_2"       # Stopped, wheels turning to opposite steer
    REVERSING_ALIGN    = "reversing_align"    # Reverse with max opposite steer
    WAIT_STEER_3       = "wait_steer_3"       # Stopped, wheels turning back to entry steer
    FORWARD_CORRECTION = "forward_correction" # Forward with max entry steer (same as initial)
    PARKED             = "parked"


# ---------------------------------------------------------------------------
# Parking maneuver constants — all values come from config.py.
# Edit config.py to tune; fallbacks here are last-resort safety defaults.
# ---------------------------------------------------------------------------

# Velocidades
PARKING_SEARCH_SPEED  = int(getattr(_config, "PARKING_SEARCH_SPEED",   13))
PARKING_FORWARD_SPEED = int(getattr(_config, "PARKING_FORWARD_SPEED",  13))
PARKING_REVERSE_SPEED = int(getattr(_config, "PARKING_REVERSE_SPEED", -13))

# Ángulos de dirección (grados)
PARKING_ENTRY_STEER = float(getattr(_config, "PARKING_ENTRY_STEER",  20.0))
PARKING_ALIGN_STEER = float(getattr(_config, "PARKING_ALIGN_STEER", -20.0))

# Detección del spot
PARKING_SPOT_MISS_THRESHOLD = int(getattr(_config,   "PARKING_SPOT_MISS_THRESHOLD",  8))
PARKING_TRIGGER_DISTANCE_CM = float(getattr(_config, "PARKING_TRIGGER_DISTANCE_CM",  100.0))

# Distancias por fase (odometría del encoder)
PARKING_D_FORWARD_CM         = float(getattr(_config, "PARKING_D_FORWARD_CM",          40.0))
PARKING_D_REVERSING_ENTRY_CM = float(getattr(_config, "PARKING_D_REVERSING_ENTRY_CM",  60.0))
PARKING_D_REVERSING_ALIGN_CM = float(getattr(_config, "PARKING_D_REVERSING_ALIGN_CM",  22.0))
PARKING_D_FORWARD_CORR_CM    = float(getattr(_config, "PARKING_D_FORWARD_CORR_CM",     20.0))

# Tiempos fallback (cuando encoder speed == 0)
PARKING_T_FORWARD         = float(getattr(_config, "PARKING_T_FORWARD",         1.5))
PARKING_T_REVERSING_ENTRY = float(getattr(_config, "PARKING_T_REVERSING_ENTRY", 3.0))
PARKING_T_REVERSING_ALIGN = float(getattr(_config, "PARKING_T_REVERSING_ALIGN", 1.5))
PARKING_T_FORWARD_CORR    = float(getattr(_config, "PARKING_T_FORWARD_CORR",    1.5))

# Tiempo de espera servo
PARKING_T_WAIT_STEER = float(getattr(_config, "PARKING_T_WAIT_STEER", 1.0))



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


class StanleyController:
    """Stanley Controller for lateral control (Hoffmann et al., Stanford 2007).

    Nonlinear control law with proven global asymptotic stability:
        δ = (ψ - ψ_ss) + arctan(k · e / (k_soft + v))
            + k_d_yaw · (r_meas - r_traj) + k_d_steer · Δδ_meas

    Unlike PID, this controller:
    - Has no integral windup (no integral term)
    - Adapts gain automatically to speed via k/(k_soft + v)
    - Provides natural steering saturation via arctan
    - Can use IMU yaw rate feedback for active damping at high speed

    Args:
        k: Crosstrack gain. Controls convergence rate toward the path.
           Higher = more aggressive correction. Units: 1/px in pixel-space mode.
        k_soft: Low-speed softening term added to speed in the denominator.
                In same units as speed (command units). This matches the
                Hoffmann paper's low-speed noise softening term.
        k_d_yaw: Yaw rate damping gain. Set >0 to use IMU feedback for stability
                 at high speed. Set 0.0 to disable (no IMU needed).
        k_d_steer: Steering servo damping gain. Applies lead-like damping based
                   on measured steering motion.
        max_steering: Maximum output angle in degrees (physical limit).
        crosstrack_deadband: Ignore very small lateral errors in the same units
            as `crosstrack_error` (meters in physical mode).
        crosstrack_deadband_px: Legacy alias for crosstrack_deadband.
        heading_deadband_rad: Ignore tiny heading offsets around zero.
        output_deadband_deg: Zero tiny output commands to avoid servo chatter.
    """

    def __init__(
        self,
        k=0.06,
        k_soft=1.0,
        k_d_yaw=0.0,
        k_d_steer=0.0,
        max_steering=25,
        crosstrack_deadband=0.0,
        crosstrack_deadband_px=None,
        heading_deadband_rad=0.0,
        output_deadband_deg=0.0,
    ):
        self.k = k
        self.k_soft = k_soft
        self.k_d_yaw = k_d_yaw
        self.k_d_steer = k_d_steer
        self.max_steering = max_steering
        if crosstrack_deadband_px is not None:
            crosstrack_deadband = crosstrack_deadband_px
        self.crosstrack_deadband = max(0.0, float(crosstrack_deadband))
        # Legacy alias preserved for backward compatibility with older callers/tests.
        self.crosstrack_deadband_px = self.crosstrack_deadband
        self.heading_deadband_rad = max(0.0, float(heading_deadband_rad))
        self.output_deadband_deg = max(0.0, float(output_deadband_deg))
        self._prev_heading = 0.0

    def compute(self, crosstrack_error, heading_error, speed,
                yaw_rate=0.0, traj_yaw_rate=0.0,
                steady_state_heading=0.0, steer_damping_delta=0.0):
        """Compute Stanley steering command.

        Args:
            crosstrack_error: Lateral error in pixels.
                Positive = lane center is right of image center (car is left of lane).
            heading_error: Angle between vehicle heading and lane direction (radians).
                Positive = car heading left of lane direction.
            speed: Vehicle speed in command units (same scale as k_soft).
            yaw_rate: Measured yaw rate from IMU (rad/s). 0.0 if unavailable.
            traj_yaw_rate: Expected yaw rate for current trajectory (rad/s).
            steady_state_heading: Desired non-zero yaw offset (rad) for curved
                steady-state motion. This is subtracted from heading_error.
            steer_damping_delta: Measured steering motion term in radians,
                typically previous_steer - current_steer.

        Returns:
            float: Steering angle in degrees, clamped to ±max_steering.
        """
        crosstrack_error = float(crosstrack_error)
        if abs(crosstrack_error) < self.crosstrack_deadband:
            crosstrack_error = 0.0

        heading_term = float(heading_error) - float(steady_state_heading)
        if abs(heading_term) < self.heading_deadband_rad:
            heading_term = 0.0

        speed_denom = max(self.k_soft + abs(float(speed)), 1e-3)
        crosstrack_term = math.atan2(self.k * crosstrack_error, speed_denom)
        yaw_damping = self.k_d_yaw * (yaw_rate - traj_yaw_rate)
        steer_damping = self.k_d_steer * steer_damping_delta

        delta_rad = heading_term + crosstrack_term + yaw_damping + steer_damping
        delta_deg = math.degrees(delta_rad)
        if abs(delta_deg) < self.output_deadband_deg:
            delta_deg = 0.0
        delta_deg = max(-self.max_steering, min(self.max_steering, delta_deg))

        self._prev_heading = heading_error
        return delta_deg

    def reset(self):
        """Reset controller state."""
        self._prev_heading = 0.0


class LateralMPC:
    """Model Predictive Controller for lateral lane tracking.

    Drop-in replacement for StanleyController using the same kinematic bicycle
    model as the reference ACADOS MPC (mpc_acados2_clean.py), but implemented
    in pure Python via scipy so it runs without ACADOS or GPS.

    Error-space bicycle kinematics (path-following formulation):
        e_{k+1}     = e_k   + dt * v * sin(psi_e_k)
        psi_e_{k+1} = psi_e_k + dt * (v/L) * tan(delta_k)

    where:
        e      — lateral offset from lane centre (m, positive = car is right of centre)
        psi_e  — heading error relative to lane tangent (rad)
        delta  — front-wheel steering angle (rad)
        v      — forward speed (m/s)
        L      — wheelbase (m)

    Quadratic cost:
        J = Σ_k [ Q_e·e_k² + Q_psi·psi_e_k² + R·δ_k² + R_rate·(δ_k−δ_{k−1})² ]
          + Q_e_N·e_N² + Q_psi_N·psi_e_N²

    Falls back to the Stanley formula if scipy is not installed.

    Interface is identical to StanleyController.compute() so either can be
    plugged in via the `use_lateral_mpc` flag on threadLineFollowing.
    """

    def __init__(
        self,
        L=0.258,
        N=10,
        dt=0.033,
        Q_e=10.0,
        Q_psi=5.0,
        R=0.5,
        R_rate=2.0,
        Q_e_N=20.0,
        Q_psi_N=10.0,
        max_steering=25,
        crosstrack_deadband=0.005,
        heading_deadband_rad=0.017,
        output_deadband_deg=1.0,
        stanley_k=0.06,
        stanley_k_soft=1.0,
    ):
        self.L = float(L)
        self.N = int(N)
        self.dt = float(dt)
        self.Q_e = float(Q_e)
        self.Q_psi = float(Q_psi)
        self.R = float(R)
        self.R_rate = float(R_rate)
        self.Q_e_N = float(Q_e_N)
        self.Q_psi_N = float(Q_psi_N)
        self.max_steering = float(max_steering)
        self.crosstrack_deadband = float(crosstrack_deadband)
        self.heading_deadband_rad = float(heading_deadband_rad)
        self.output_deadband_deg = float(output_deadband_deg)
        self.delta_max = math.radians(max_steering)
        # Stanley fallback parameters (used when scipy unavailable)
        self._fb_k = float(stanley_k)
        self._fb_k_soft = float(stanley_k_soft)
        self._prev_delta = 0.0
        self._last_debug = {}  # Stores internal state from the last compute() call

    def _rollout(self, e0, psi0, deltas, v):
        """Simulate N steps of the error-space bicycle model.

        Sign convention (matches compute() interface and Stanley):
            e     > 0  →  car is LEFT of lane centre
            psi_e > 0  →  car is heading LEFT of path tangent
            delta > 0  →  RIGHT turn (positive steering = right, as in the actuator)
        """
        e, psi = e0, psi0
        states = [(e, psi)]
        for d in deltas:
            e = e + self.dt * v * math.sin(psi)
            psi = psi - self.dt * (v / max(self.L, 0.01)) * math.tan(
                max(-self.delta_max, min(self.delta_max, d))
            )
            states.append((e, psi))
        return states

    def _cost(self, deltas, e0, psi0, v, prev_delta):
        """Compute MPC cost for a candidate control sequence."""
        states = self._rollout(e0, psi0, deltas, v)
        cost = 0.0
        for k, (e_k, psi_k) in enumerate(states[:-1]):
            cost += self.Q_e * e_k ** 2 + self.Q_psi * psi_k ** 2
            cost += self.R * deltas[k] ** 2
            prev = prev_delta if k == 0 else deltas[k - 1]
            cost += self.R_rate * (deltas[k] - prev) ** 2
        e_N, psi_N = states[-1]
        cost += self.Q_e_N * e_N ** 2 + self.Q_psi_N * psi_N ** 2
        return cost

    def compute(
        self,
        crosstrack_error,
        heading_error,
        speed,
        yaw_rate=0.0,
        traj_yaw_rate=0.0,
        steady_state_heading=0.0,
        steer_damping_delta=0.0,
    ):
        """Solve the MPC and return the optimal steering angle (degrees).

        Args match StanleyController.compute() exactly for drop-in replacement.
            crosstrack_error: Lateral error in metres.  Positive = car left of centre.
            heading_error:    Heading error in radians. Positive = car heading left.
            speed:            Forward speed in m/s.
            (other args accepted but not used by this formulation)

        Returns:
            float: Optimal steering angle in degrees, clamped to ±max_steering.
        """
        e0_raw = float(crosstrack_error)
        psi0_raw = float(heading_error) - float(steady_state_heading)
        v = max(0.01, abs(float(speed)))

        e0 = e0_raw
        psi0 = psi0_raw
        if abs(e0) < self.crosstrack_deadband:
            e0 = 0.0
        if abs(psi0) < self.heading_deadband_rad:
            psi0 = 0.0

        prev_delta_snapshot = self._prev_delta

        if not _SCIPY_AVAILABLE:
            speed_denom = max(self._fb_k_soft + v, 1e-3)
            delta_rad = psi0 + math.atan2(self._fb_k * e0, speed_denom)
            delta_deg = math.degrees(delta_rad)
            delta_deg = max(-self.max_steering, min(self.max_steering, delta_deg))
            self._prev_delta = math.radians(delta_deg)
            self._last_debug = {
                'scipy': False, 'fallback': True,
                'e0_raw_m': round(e0_raw, 5), 'psi0_raw_deg': round(math.degrees(psi0_raw), 3),
                'e0_m': round(e0, 5), 'psi0_deg': round(math.degrees(psi0), 3),
                'v_mps': round(v, 4), 'prev_delta_deg': round(math.degrees(prev_delta_snapshot), 3),
                'delta_opt_deg': round(delta_deg, 3), 'output_deg': round(delta_deg, 3),
            }
            return delta_deg

        x0 = np.full(self.N, self._prev_delta)
        bounds = [(-self.delta_max, self.delta_max)] * self.N

        solver_ok = False
        solver_msg = 'exception'
        delta_opt_raw = 0.0
        try:
            result = _scipy_minimize(
                self._cost,
                x0,
                args=(e0, psi0, v, self._prev_delta),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-5},
            )
            optimal_deltas = result.x
            solver_ok = result.success
            solver_msg = result.message if hasattr(result, 'message') else str(result.status)
            delta_opt_raw = float(optimal_deltas[0])
        except Exception as _exc:
            # On any solver failure fall back to Stanley
            speed_denom = max(self._fb_k_soft + v, 1e-3)
            delta_rad = psi0 + math.atan2(self._fb_k * e0, speed_denom)
            delta_deg = math.degrees(delta_rad)
            delta_deg = max(-self.max_steering, min(self.max_steering, delta_deg))
            self._prev_delta = math.radians(delta_deg)
            self._last_debug = {
                'scipy': True, 'fallback': True, 'solver_ok': False,
                'solver_msg': str(_exc)[:80],
                'e0_raw_m': round(e0_raw, 5), 'psi0_raw_deg': round(math.degrees(psi0_raw), 3),
                'e0_m': round(e0, 5), 'psi0_deg': round(math.degrees(psi0), 3),
                'v_mps': round(v, 4), 'prev_delta_deg': round(math.degrees(prev_delta_snapshot), 3),
                'delta_opt_deg': round(delta_deg, 3), 'output_deg': round(delta_deg, 3),
            }
            return delta_deg

        # Apply only the first control in the receding-horizon fashion.
        delta_opt = float(delta_opt_raw)
        delta_opt = max(-self.delta_max, min(self.delta_max, delta_opt))
        self._prev_delta = delta_opt

        delta_deg = math.degrees(delta_opt)
        delta_opt_deg_clamped = delta_deg
        if abs(delta_deg) < self.output_deadband_deg:
            delta_deg = 0.0
        self._last_debug = {
            'scipy': True, 'fallback': False, 'solver_ok': bool(solver_ok),
            'solver_msg': str(solver_msg)[:80],
            'e0_raw_m': round(e0_raw, 5), 'psi0_raw_deg': round(math.degrees(psi0_raw), 3),
            'e0_m': round(e0, 5), 'psi0_deg': round(math.degrees(psi0), 3),
            'v_mps': round(v, 4), 'prev_delta_deg': round(math.degrees(prev_delta_snapshot), 3),
            'delta_opt_deg': round(delta_opt_deg_clamped, 3), 'output_deg': round(delta_deg, 3),
        }
        return delta_deg

    def reset(self):
        """Reset controller state."""
        self._prev_delta = 0.0
        self._last_debug = {}


class threadLineFollowing(ThreadWithStop):
    """Thread which handles line following using OpenCV.

Args:
    queuesList (dictionar of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
    logger (logging object): Made for debugging.
    debugger (bool): A flag for debugging.
"""

    def __init__(self, queuesList, logger, debugger, frame_buffer=None,
                 show_debug=False, debug_windows=None, sign_action_event=None,
                 highway_mode_event=None, steer_override_event=None):
        super(threadLineFollowing, self).__init__(pause=0.05)  # 20Hz — camara produce ~5 FPS, no necesita polling mas rapido
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        self.frame_buffer = frame_buffer
        self.show_debug = show_debug
        self.debug_windows = debug_windows or {}
        self.sign_action_event = sign_action_event  # When set, sign action is active — don't send motor commands
        self._sign_action_was_active = False  # Track transitions to reset speed ramp
        self.highway_mode_event = highway_mode_event  # When set, car is on highway — use higher speeds
        self.steer_override_event = steer_override_event  # When set, sign action also controls steer (e.g. hardcoded turn)

        # Speed parameters
        self.base_speed        = float(getattr(_config, "LF_BASE_SPEED",        10))
        self.max_speed         = float(getattr(_config, "LF_MAX_SPEED",         13))
        self.min_speed         = float(getattr(_config, "LF_MIN_SPEED",          8))
        self.highway_max_speed = float(getattr(_config, "LF_HIGHWAY_MAX_SPEED", 25))
        self.highway_min_speed = float(getattr(_config, "LF_HIGHWAY_MIN_SPEED", 15))
        self.speed_ramp_step      = float(getattr(_config, "LF_SPEED_RAMP_STEP",      0.5))
        self.speed_ramp_down_step = float(getattr(_config, "LF_SPEED_DOWN_RAMP_STEP", 1.5))
        self._current_speed = self.base_speed  # Tracks actual speed for ramping
        self.speed_steer_factor = 0.4  # Steering attenuation at max speed (0=none, 1=full mute)
        self.max_frame_age_ms = 300.0
        self._last_frame_age_ms = None
        self._last_processed_sequence = 0
        self._last_safe_steering = 0.0
        self._last_safe_speed = self.min_speed
        self._debug_log_times = {}
        self.preview_interval = 1.0 / 5.0
        self._last_preview_time = 0.0
        self._preview_frame_due = False
        self._preview_refresh_pending = False

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
        _steer_hist_len = max(1, int(getattr(_config, 'STEER_HISTORY_LEN', 1)))
        self.steer_history = deque(maxlen=_steer_hist_len)
        self.lookahead = 0.4
        self.pid = PIDController(
            Kp=self.kp, Ki=self.ki, Kd=self.kd,
            max_steering=self.max_steering,
            tolerance=self.dead_zone_ratio,
            integral_reset_interval=self.integral_reset_interval
        )
        self.previous_error = 0  # Keep for compatibility

        # Feed-forward curve prediction parameters
        self.wheelbase = 0.260       # Distance between axles in meters (TC-04: 260mm)
        self.ff_weight = 0.6         # Weight of feed-forward vs PID (0=pure PID, 1=pure FF)
        self.curvature_threshold = 0.5  # Min curvature to activate feed-forward (1/pixels)

        # ============= Stanley Controller (Hoffmann et al., Stanford 2007) =============
        # Replaces PID for OpenCV mode with a globally stable nonlinear controller.
        # The gain k/(k_soft + v) adapts automatically to speed:
        #   Low speed  → aggressive correction (tight curves)
        #   High speed → gentle correction (highway stability)
        self.use_stanley = True          # True=Stanley, False=PID (fallback)
        # k=0.5 saturates (~25°) at e≈20cm (57% of 35cm lane) at v=0.13 m/s.
        # k=1.2 (old) saturated at only e≈8.5cm causing bang-bang oscillation.
        self.stanley_k = float(getattr(_config, "STANLEY_K", 0.5))
        # k_soft must be >= minimum operating speed to prevent gain explosion at near-zero speed.
        # Parking runs at v≈0.13 m/s; k_soft=0.20 keeps gain finite and speed-proportional.
        self.stanley_k_soft = float(getattr(_config, "STANLEY_K_SOFT", 0.20))
        self.stanley_k_d_yaw = 0.0      # Yaw rate damping from IMU (0=disabled, try 0.1 with IMU)
        self.stanley_k_d_steer = float(getattr(_config, "STANLEY_K_D_STEER", 0.10))  # from config.py
        self.stanley_k_ag = 0.0         # ψ_ss gain for curved steady-state tracking (disabled: k_ag×v²/r < 0.3° at all operating speeds)
        # Speed scale for command speed units (controller speed variable) to m/s.
        # Controller speed uses cm/s internally, so 1 cm/s = 0.01 m/s.
        self.stanley_speed_to_mps = 0.01
        # Nucleo CurrentSpeed is sent in x10 cm/s (same as speed_x10 command feedback):
        # 1 unit => 0.1 cm/s => 0.001 m/s.
        self.stanley_measured_speed_to_mps = 0.001
        self.stanley_use_measured_speed = True
        self.stanley_measured_speed_timeout = 0.4
        self.stanley_curve_min_confidence = 0.35  # Ignore weak curve estimates for ψ_ss / r_traj
        self.stanley_psi_ss_max_deg = 12.0  # Safety cap for steady-state heading bias
        self.stanley_curve_direction_guard_error_m = 0.010
        self.stanley_curve_direction_guard_heading_deg = 1.5
        self.stanley_deadband_crosstrack_m = 0.01
        self.stanley_deadband_crosstrack_px = 4.0
        self.stanley_deadband_heading_deg = 1.0
        self.stanley_deadband_output_deg = 1.2
        self.stanley = StanleyController(
            k=self.stanley_k,
            k_soft=self.stanley_k_soft,
            k_d_yaw=self.stanley_k_d_yaw,
            k_d_steer=self.stanley_k_d_steer,
            max_steering=self.max_steering,
            crosstrack_deadband=self.stanley_deadband_crosstrack_m,
            heading_deadband_rad=math.radians(self.stanley_deadband_heading_deg),
            output_deadband_deg=self.stanley_deadband_output_deg,
        )
        self._stanley_base_k = float(self.stanley_k)
        self._stanley_base_k_soft = float(self.stanley_k_soft)
        self._stanley_sched_k = float(self.stanley_k)
        self._stanley_sched_k_soft = float(self.stanley_k_soft)
        self._stanley_sched_alpha = 0.30

        # ── Lateral MPC (kinematic bicycle, receding horizon) ──────────────
        # Drop-in replacement for Stanley based on the same bicycle model as
        # mpc_acados2_clean.py from the reference repo, but running in pure
        # Python (scipy) without ACADOS or GPS.
        # Set use_lateral_mpc=True to activate; falls back to Stanley when
        # scipy is unavailable.
        self.use_lateral_mpc = bool(getattr(_config, "USE_LATERAL_MPC", True))
        self.mpc_N = int(getattr(_config, "MPC_N", 10))
        self.mpc_dt = float(getattr(_config, "MPC_DT", 0.033))
        self.mpc_L = float(getattr(_config, "MPC_WHEELBASE", 0.258))
        self.mpc_Q_e = float(getattr(_config, "MPC_Q_E", 10.0))
        self.mpc_Q_psi = float(getattr(_config, "MPC_Q_PSI", 5.0))
        self.mpc_R = float(getattr(_config, "MPC_R", 0.5))
        self.mpc_R_rate = float(getattr(_config, "MPC_R_RATE", 2.0))
        self.mpc_Q_e_N = float(getattr(_config, "MPC_Q_E_N", 20.0))
        self.mpc_Q_psi_N = float(getattr(_config, "MPC_Q_PSI_N", 10.0))
        self.mpc_output_deadband_deg = float(getattr(_config, "MPC_OUTPUT_DEADBAND_DEG", 0.5))
        self.mpc_heading_deadband_deg = float(getattr(_config, "MPC_HEADING_DEADBAND_DEG", 0.0))
        self.mpc_crosstrack_k_mult = float(getattr(_config, "MPC_CROSSTRACK_K_MULT", 1.4))
        self.lateral_mpc = LateralMPC(
            L=self.mpc_L,
            N=self.mpc_N,
            dt=self.mpc_dt,
            Q_e=self.mpc_Q_e,
            Q_psi=self.mpc_Q_psi,
            R=self.mpc_R,
            R_rate=self.mpc_R_rate,
            Q_e_N=self.mpc_Q_e_N,
            Q_psi_N=self.mpc_Q_psi_N,
            max_steering=self.max_steering,
            crosstrack_deadband=self.stanley_deadband_crosstrack_m,
            heading_deadband_rad=math.radians(self.mpc_heading_deadband_deg),
            output_deadband_deg=self.mpc_output_deadband_deg,
            stanley_k=self.stanley_k,
            stanley_k_soft=self.stanley_k_soft,
        )
        if self.use_lateral_mpc:
            if _SCIPY_AVAILABLE:
                print(
                    "\033[1;97m[ Line Following ] :\033[0m \033[1;92mINFO\033[0m - "
                    f"Lateral MPC active (N={self.mpc_N}, dt={self.mpc_dt}s, L={self.mpc_L}m)"
                )
            else:
                print(
                    "\033[1;97m[ Line Following ] :\033[0m \033[1;93mWARNING\033[0m - "
                    "USE_LATERAL_MPC=True but scipy not installed — falling back to Stanley"
                )

        # Single-line lane centering: how far from the visible line to aim.
        # 0.50 = exact center (17.5cm), 0.40 = closer to visible line (less overshoot).
        # If the car overshoots past center and hits the opposite line, lower this.
        # If the car hugs the visible line too closely, raise this.
        self.single_line_offset_factor = 0.42
        # Extra bias when only the OUTER curve line is visible.
        # Higher = stronger negative correction as the car gets closer to that line.
        self.single_line_outer_bias_gain = 0.60
        # Outer-line correction is capped to avoid sudden 1-line steering spikes.
        self.single_line_outer_bias_limit_factor = 0.18
        # A single line must look curve-like before we trust outer-line heuristics.
        self.single_line_curve_confirm_threshold = 0.35
        # Require at least this many consecutive 1-line frames for continuity-based confirmation.
        self.single_line_outer_confirm_frames = 2
        # Slope-only heading hint stays intentionally bounded in 1-line mode.
        self.single_line_curve_heading_max_deg = 10.0
        # Safety stop: when 1-line mode stays saturated for several frames, stop
        # pushing forward into a curve the chassis cannot physically close.
        self.single_line_stop_max_steer_frames = 6
        self.single_line_stop_error_scale = 1.5
        self.use_single_line_protective_stop = bool(
            getattr(_config, "USE_SINGLE_LINE_PROTECTIVE_STOP", False)
        )
        # Briefly hold the last good steering when vision drops both lines.
        self.no_lane_hold_steering_frames = 4
        # Keep full-center reconstruction only briefly after 2->1 line loss.
        self.single_line_prefer_center_frames = 2
        # Single-line heading calibration: compensates perspective bias so 1-line mode
        # stays close to 0 steering when the car is centered on straight segments.
        self.single_line_heading_ref_alpha = 0.15
        self._single_line_heading_ref_left = 0.0
        self._single_line_heading_ref_right = 0.0

        # Sensor feedback state (populated from Nucleo via message queue)
        self._measured_speed = 0.0       # Last CurrentSpeed from Nucleo (x10 cm/s)
        self._last_speed_time = None     # Timestamp of last speed reading
        self._measured_steer = 0.0       # Last steering angle feedback
        self._measured_steer_delta = 0.0 # Previous steer - current steer (deg)
        self._last_yaw = 0.0             # Last IMU yaw reading (degrees)
        self._last_imu_roll = 0.0        # Last IMU roll reading (degrees)
        self._last_imu_pitch = 0.0       # Last IMU pitch reading (degrees)
        self._yaw_rate = 0.0             # Computed yaw rate (rad/s)
        self._last_imu_time = None       # Timestamp of last IMU reading
        self._last_cmd_speed = 0.0       # Last SpeedMotor command from dashboard (raw str→float)
        self._last_cmd_steer = 0.0       # Last SteerMotor command from dashboard (raw str→float)
        self._heading_error = 0.0        # Last computed heading error (rad)
        self._stanley_last_psi_ss = 0.0  # Last steady-state heading bias (rad)
        self._stanley_last_traj_yaw_rate = 0.0  # Last reference yaw rate (rad/s)
        self._stanley_last_steer_damping = 0.0  # Last steering damping input (rad)
        self._stanley_last_speed_mps = 0.0
        self._stanley_last_speed_source = "none"
        self._stanley_last_error_m = 0.0
        self._stanley_last_px_per_cm = 0.0
        self._stanley_last_curve_guard = "none"
        # Kalman filter for crosstrack error (robust when lanes flicker due to glare/shadows)
        # Q=300, R=8 → gain ≈ 0.97 (near-passthrough): AI masks are clean and must not be
        # over-smoothed. The old Q=0.04/R=25 (ratio 1:625) was killing curve errors — a real
        # -132 px curve offset was reduced to -7.8 px, giving Stanley near-zero crosstrack input.
        self.use_kalman_filter = True
        self.kalman_process_noise = 300.0
        self.kalman_measurement_noise = 8.0
        self.kalman_max_prediction_frames = 4
        self.kalman_prediction_damping = 0.92
        self._kalman_error = cv2.KalmanFilter(2, 1)
        self._kalman_initialized = False
        self._kalman_last_ts = None
        self._kalman_prediction = 0.0
        self._kalman_age_frames = 0
        self._init_error_kalman()

        # ============= Car Physical Dimensions (cm) =============
        # Measured from the actual BFMC 1:10 scale car (Bosch)
        self.car_length = 36.5           # Total car length (cm)
        self.car_width = 19.0            # Total car width (cm)
        self.car_wheelbase_cm = float(getattr(_config, 'CAR_WHEELBASE_CM', 26.0))
        self.car_front_overhang = 7.2    # Front axle to front bumper (cm)
        self.car_rear_overhang = 1.8     # Rear axle to rear of car (cm)
        self.camera_to_front_axle = 11.5 # Camera position ahead of front axle (cm)
        self.camera_to_rear_axle = 37.5  # Camera to rear axle (11.5 + 26.0) (cm)

        # Track dimensions (cm) - BFMC standard track (configurable via config.py)
        self.lane_width_cm = float(getattr(_config, "LANE_WIDTH_CM", 35.0))
        self.line_width_cm = float(getattr(_config, "LINE_WIDTH_CM", 2.0))

        # Minimum clearance from car body edge to lane line inner edge.
        # A correction is added to the lateral error when this margin is violated,
        # proportional to how much the margin is exceeded, pushing the car back toward center.
        # Applies only in 2-line detection mode where both line positions are physically known.
        self.lane_safety_margin_cm = float(getattr(_config, 'LANE_SAFETY_MARGIN_CM', 5.0))

        # Swept path prediction parameters
        self.use_swept_path = False      # Swept path disabled: Stanley handles curves via pure crosstrack+heading error
        self.swept_path_margin_cm = 4.0  # Safety margin for clearance checks (cm) — larger = more outward in curves
        self.curve_offset_gain = 1.0     # Apply full optimal offset (0-1)
        self.curve_inner_bias_cm = 0.0   # Inner bias disabled: was pushing car toward inside of curve
        # Heading-proportional outward offset for crosstrack error (both two-line and single-line).
        # heading_rad = atan2(Δmidpoint, dy) is computed over a short baseline (≈42px), which
        # perspective-amplifies the road curvature to ~-45° in tight curves even when the car is
        # nearly centered (raw_error_px ≈ 5-13px).  Without correction this heading term drives
        # Stanley to max left steering (-25°) when only -15° to -18° is actually needed.
        # Calibrated to BFMC curve geometry (TC-04, wheelbase=26cm, lane=35cm):
        #   inner lane R=66.5cm → need 21.3°
        #   outer lane R=103.5cm → need 14.1°
        # Applied ONLY in midpoint_ref (two-line) mode — see comment below.
        # At heading=-22°/raw=-55px/lane=535px/speed=0.132 m/s (new log, outer lane):
        #   gain=0.40 → offset=84px → net=+29px → arctan=8.4° → δ=−14.2°  ✓ outer lane
        # At heading=-25°/raw=-61px (peak): gain=0.40 → net=+31px → δ=−16.8°
        # Single-line (heading=+20°/raw=-162px): NOT applied (guidance_mode != midpoint_ref)
        #   → error stays -162px → δ=-17.6° (correct, was -25° with gain applied)
        self.curve_crosstrack_heading_gain = 0.40
        self.curve_confidence_threshold = 0.15  # Min confidence to apply curve offset (lower = earlier activation)
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
        self.curve_exit_frames = 8           # Frames with 2 lines before EXITING→STRAIGHT (longer = smoother heading recovery)
        self.curve_vp_threshold = 0.20       # VP offset to detect curve early from 2 lines (0.10 was too sensitive)
        self.curve_vp_confirm_frames = 3    # Consecutive frames with VP above threshold to trigger ENTERING from 2 lines
        self._vp_above_count = 0            # Counter: consecutive frames VP exceeded threshold
        self.curve_two_line_heading_threshold_deg = float(
            getattr(_config, 'CURVE_TWO_LINE_HEADING_THRESHOLD_DEG', 24.0)
        )
        self.curve_two_line_heading_confidence = float(
            getattr(_config, 'CURVE_TWO_LINE_HEADING_CONFIDENCE', 0.55)
        )
        self.curve_two_line_confirm_frames = int(
            getattr(_config, 'CURVE_TWO_LINE_CONFIRM_FRAMES', 2)
        )
        self.curve_two_line_hold_frames = int(
            getattr(_config, 'CURVE_TWO_LINE_HOLD_FRAMES', 6)
        )
        self._two_line_curve_hint_count = 0
        self._two_line_curve_hint_direction = 0
        self._curve_enter_origin = "none"
        self._recent_curve_direction = 0
        self._recent_curve_direction_frames = 0
        self.curve_reentry_hold_frames = max(10, self.curve_exit_frames + 2)
        self.curve_reentry_min_steer_deg = 3.0
        self.curve_pre_position_gain = 0.95  # Pre-position aggressively when ENTERING (higher = more outer-line protection)
        self.curve_exit_gain = 0.08          # Tiny residual offset when EXITING; fades over curve_exit_frames
        self.curve_inner_line_steer_factor = float(getattr(_config, 'CURVE_INNER_LINE_STEER_FACTOR', 0.4))
        # Single-line fusion: blend geometric swept-path error with Stanley error in curves.
        self.single_line_swept_weight_entering = 0.40   # reduced: stronger Stanley k_soft=0.20 amplifies swept corrections
        self.single_line_swept_weight_in_curve = 0.55   # reduced from 0.85: prevents cutting inner line with new gains
        self.single_line_swept_weight_exiting = 0.45    # reduced from 0.65

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
        self.line_angle_filter = 20    # Min angle (degrees) to classify as lane line
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
        self.adaptive_white_tophat_kernel = 15
        self.adaptive_white_tophat_min_threshold = 20
        self.use_gradient_fallback = True
        self.gradient_percentile = 85
        self.use_component_filter = False
        self.component_min_area = 20
        self.component_min_height = 12
        self.component_min_aspect_ratio = 2.0
        self.component_min_angle = 25.0
        self.use_local_adaptive_threshold = False
        self.local_adaptive_block_size = 31
        self.local_adaptive_c = 7.0
        self.hough_max_lines_per_side = 12

        # Auto binary threshold: adapta a cambios de luz frame a frame.
        # Desactivado: se usa siempre binary_threshold fijo.
        self.use_auto_threshold = False
        self.auto_threshold_percentile = 85   # Más permisivo que 90 → coge más píxeles de línea
        self.auto_threshold_min = 125         # Permite bajar más en asfalto oscuro
        self.auto_threshold_max = 220         # Mantener techo para no perder líneas en brillo alto
        self._auto_threshold_val = self.binary_threshold  # Valor suavizado inicial

        # Detection mode parameters
        self.detection_mode = DetectionMode.AI_LOCAL.value # "opencv", "lstr", or "hybrid"
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
        self.hybridnets_server_url = "ws://127.0.0.1:8500/ws/steering"
        self.hybridnets_jpeg_quality = 70
        self.hybridnets_timeout = 2.0  # GPU inference <100ms + network ~50ms; 2s es suficiente
        self.hybridnets_max_result_age = 0.35  # Ignore stale AI geometry; use local recovery instead
        self.local_ai_max_result_age = 0.35
        self.local_ai_stale_grace_s = 0.20
        self.local_ai_latency_warn_ms = 450.0
        self.local_ai_latency_stop_ms = 750.0
        self.local_ai_latency_stop_frames = 3
        self._local_ai_high_latency_streak = 0
        self._local_ai_heading_hint_rad = 0.0
        self._local_ai_heading_hint_confidence = 0.0
        self._local_ai_heading_hint_source = "none"
        self._local_ai_road_type_class = "unknown"
        self._local_ai_road_type_confidence = 0.0
        self._local_ai_lead_distance_class = "none"
        self._local_ai_lead_distance_confidence = 0.0
        self._local_ai_lead_distance_area = 0.0
        self.ai_local_use_mask_geometry = True
        # AI_LOCAL uses a wider ROI for mask geometry so far-field masks (visible
        # during curve approach when near-field lines are hidden) are not discarded.
        # 0.15 = clip only top 15% of frame; standard roi_height_start=0.35 is too
        # aggressive for AI masks and causes "No lines detected" in curves.
        self.ai_local_roi_height_start = 0.15
        self._hybridnets_client = None
        self._last_local_lane_payload = None
        self._last_local_perception_status = None
        self._last_actuator_status = None
        self._last_local_ai_lane_width_px = None
        self._last_two_line_left = None
        self._last_two_line_right = None
        self._last_two_line_ts = 0.0
        self._lane_observation_history = deque(maxlen=12)
        self._last_local_ai_duplicate_collapse = None
        self._last_single_line_projection_debug = {}
        self.auto_run_log_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "..", "..",
                "temp",
                "line_following_auto_last_run.txt",
            )
        )
        self._auto_run_log_enabled = False
        self._auto_run_log_frame_idx = 0
        self._last_frame_trace = {}
        self.manual_run_log_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "..", "..",
                "temp",
                "line_following_manual_last_run.txt",
            )
        )
        self._manual_run_log_enabled = False
        self._manual_run_log_frame_idx = 0

        # Lane mask classification debug log (always-on, controlled by config)
        self._mask_debug_log_enabled = bool(getattr(_config, 'LANE_MASK_DEBUG_LOG', False))
        self._mask_debug_log_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "..", "..",
                "temp",
                "lane_mask_debug.log",
            )
        )
        self._mask_debug_frame_idx = 0
        self._last_bev_reclassify_debug = {}   # filled by _reclassify_masks_via_bev
        self._last_yolo_raw_debug = {}          # filled by _detect_with_local_ai
        if self._mask_debug_log_enabled:
            try:
                os.makedirs(os.path.dirname(self._mask_debug_log_path), exist_ok=True)
                with open(self._mask_debug_log_path, "w", encoding="utf-8") as _f:
                    import datetime
                    _f.write(
                        f"=== LANE MASK DEBUG LOG — started {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                        "Legend:\n"
                        "  raw      : masks from YOLO (before any classification fix)\n"
                        "  prep     : after _prepare_local_ai_side_masks (guessed→resolve)\n"
                        "  bev      : after _reclassify_masks_via_bev\n"
                        "  guide    : final _build_local_mask_guidance result\n"
                        "  err      : crosstrack error [m]  (+right / -left of centre)\n"
                        "  hdg      : heading error [deg]\n"
                        "  steer    : commanded steering [deg]  (+right / -left)\n"
                        "  Lcx/Rcx  : BEV x-centroid of left/right mask [px], centre=320\n"
                        "  src      : YOLO source (g=guessed_single, e=explicit_class, etc)\n"
                        "\n"
                    )
            except Exception as _e:
                print(f"[ Line Following ] WARNING - could not open mask debug log: {_e}")
                self._mask_debug_log_enabled = False

        # Calibration mode state
        self._calib_mode_active = False
        self._calib_log_frame_idx = 0
        self._calib_log_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "..", "..",
                "temp",
                "lane_calib_log.txt",
            )
        )
        self._last_requested_motor_command = None
        self._last_sl_shadow_debug = None  # populated each two-line frame for calib diagnostics
        self._last_state_change_message = None
        self._dynamic_metrics_window = 20
        self._theta_abs_error_history = deque(maxlen=self._dynamic_metrics_window)
        self._offset_abs_error_history = deque(maxlen=self._dynamic_metrics_window)
        self._theta_dmae_deg = 0.0
        self._offset_dma_m = 0.0
        
        # Supercombo remote AI server parameters (same protocol as HybridNets)
        self.supercombo_server_url = "ws://127.0.0.1:8500/ws/steering"
        self.supercombo_jpeg_quality = 70
        self.supercombo_timeout = 2.0
        self._supercombo_client = None
        
        self._init_lstr_detector()
        # Legacy remote clients stay available, but are now initialized lazily
        # only if an old code path explicitly requests them.

        # Debug streaming parameters
        self.stream_debug_view = 0  # 0=off, 1-10=different views
        self.stream_debug_fps = 5  # Max FPS for debug stream (to save bandwidth)
        self.stream_debug_quality = 50  # JPEG quality (1-100)
        self.stream_debug_scale = 0.5  # Scale factor for debug images
        self._debug_frame_counter = 0
        self._debug_images = {}  # Store debug images for streaming
        self._last_stream_time = 0
        self.line_visual_smoothing_alpha = 0.35
        self.line_visual_missing_reset_frames = 2
        self._smoothed_left_line = None
        self._smoothed_right_line = None
        self._left_line_missing_frames = 0
        self._right_line_missing_frames = 0
        self.single_line_transition_blend_frames = int(getattr(_config, 'SINGLE_LINE_BLEND_FRAMES', 2))
        self.duplicate_line_gap_ratio = 0.18

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

        # Perspective transform (bird's eye view)
        self.perspective_M = None
        self.perspective_M_inv = None
        self.perspective_initialized = False
        self.bev_cm_per_px = 0.0  # metric scale: cm per pixel in BEV space (set in _init_perspective_transform)

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
        self.noise_max_steer_jump_deg = float(getattr(_config, 'NOISE_MAX_STEER_JUMP_DEG', 15))
        self.noise_max_reject_frames = int(getattr(_config, 'NOISE_MAX_REJECT_FRAMES', 3))
        self._noise_reject_count = 0         # Current consecutive rejected frames
        self._last_good_error = 0.0          # Last accepted error value
        self._last_good_steering = 0.0       # Last accepted steering angle
        self._last_good_num_lines = 0        # Last accepted line count
        self._prev_direct_error_side = None  # Visible side used by single-line direct_error_m

        # Curve recovery (reverse when curve is impossible)
        self.use_curve_recovery = False      # Disabled: reverse recovery caused incorrect interventions
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
        self._single_line_stop_consecutive = 0
        self._recovery_curve_sign = 0        # +1 for right curve, -1 for left curve

        # BFMC-style single-line tracking
        self.consecutive_single_left = 0
        self.consecutive_single_right = 0
        self.just_seen_two_lines = False

        # Message handlers
        self.speedMotorSender = messageHandlerSender(self.queuesList, SpeedMotor)
        self.steerMotorSender = messageHandlerSender(self.queuesList, SteerMotor)
        self.stateChangeSubscriber = messageHandlerSubscriber(self.queuesList, StateChange, "lastOnly", True)
        self.configSubscriber = messageHandlerSubscriber(self.queuesList, LineFollowingConfig, "lastOnly", True)
        self.localLanePerceptionSubscriber = messageHandlerSubscriber(self.queuesList, LocalLanePerception, "lastOnly", True)
        self.localPerceptionStatusSubscriber = messageHandlerSubscriber(self.queuesList, LocalPerceptionStatus, "lastOnly", True)
        self.actuatorStatusSubscriber = messageHandlerSubscriber(self.queuesList, ActuatorCommandStatus, "lastOnly", True)
        
        # Sensor feedback subscribers (for Stanley controller)
        self.imuDataSubscriber = messageHandlerSubscriber(self.queuesList, ImuData, "lastOnly", True)
        self.currentSpeedSubscriber = messageHandlerSubscriber(self.queuesList, CurrentSpeed, "lastOnly", True)
        self.currentSteerSubscriber = messageHandlerSubscriber(self.queuesList, CurrentSteer, "lastOnly", True)

        # Dashboard command subscribers — used to capture PC commands in the manual-run log
        self.speedMotorSubscriber = messageHandlerSubscriber(self.queuesList, SpeedMotor, "lastOnly", True)
        self.steerMotorSubscriber = messageHandlerSubscriber(self.queuesList, SteerMotor, "lastOnly", True)

        # Parking spot detection subscriber
        self.signDetectedSubscriber = messageHandlerSubscriber(self.queuesList, SignDetected, "lastOnly", True)

        # Calibration mode subscriber
        self.laneCalibModeSubscriber = messageHandlerSubscriber(self.queuesList, LaneCalibMode, "lastOnly", True)

        # Parking state machine variables
        self._parking_state                = ParkingState.IDLE
        self._parking_last_spot_box        = None    # Last known bbox [y1,x1,y2,x2] normalized
        self._parking_last_spot_distance_cm = None   # Metric distance to spot (from SignDetected)
        self._parking_spot_miss_frames     = 0       # Consecutive frames without parking detection
        self._parking_phase_start_time     = 0.0     # Timestamp when current phase started
        self._parking_phase_dist_cm        = 0.0     # Odometry-based distance traveled in current phase
        self._parking_last_sm_time         = 0.0     # Last time state machine was called (for delta dt)
        self._parking_dynamic_forward_cm   = PARKING_D_FORWARD_CM  # Computed at spot-lost transition

        # Debug stream senders
        self.debugStreamSender = messageHandlerSender(self.queuesList, LineFollowingDebug)
        self.statusSender = messageHandlerSender(self.queuesList, LineFollowingStatus)
        self.stateChangeSender = messageHandlerSender(self.queuesList, StateChange)

        # GPS-free tracking state (injected by processCamera after init)
        self._tracking_state = None
        self._maneuver_manager = ManeuverManager(
            state_change_sender=self.stateChangeSender,
            highway_mode_event=self.highway_mode_event,
        )
        self._control_policy_mode = "VISUAL_ASSIST"
        self._control_authority = "visual"
        self._active_maneuver_name = "none"
        self._mission_state_name = "LANE_FOLLOW"
        self._planner_priority_active = False
        self._safety_stop_reason = ""
        self._last_maneuver_notes = ""
        self._last_planner_route_blend = 0.0
        self._last_planner_route_blend_source = "none"
        self._recent_two_line_straight_until = 0.0

        print("\033[1;97m[ Line Following ] :\033[0m \033[1;92mINFO\033[0m - Line following thread initialized")
        print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;92mINFO\033[0m - Debug mode: {self.show_debug}")
        print("\033[1;97m[ Line Following ] :\033[0m \033[1;93mWARNING\033[0m - Windows will open when camera starts streaming")

    def set_tracking_state(self, tracking_state) -> None:
        """Inject the shared TrackingState object from processCamera.

        Once set, ``_compute_lateral_control`` will use the dead-reckoning
        waypoint errors whenever ``tracking_state.waypoint_mode_active`` is True
        (i.e. when the car is near a STOPLINE or INTERSECTION node on the graph).
        """
        self._tracking_state = tracking_state

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
            client_url = self._resolve_hybridnets_inference_url()
            self._hybridnets_client = HybridNetsClient(
                server_url=client_url,
                jpeg_quality=self.hybridnets_jpeg_quality,
                timeout=self.hybridnets_timeout,
            )
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;92mINFO\033[0m - HybridNets client created (server: {client_url})")
        except Exception as e:
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mERROR\033[0m - Failed to create HybridNets client: {e}")
            self._hybridnets_client = None

    def _resolve_hybridnets_inference_url(self):
        """Always use full inference so local BFMC/Stanley consumes remote lane geometry."""
        url = str(self.hybridnets_server_url).strip()
        if "/ws/" in url:
            base = url.split("/ws/", 1)[0]
            return base + "/ws/inference"
        return url.rstrip("/") + "/ws/inference"

    def _normalize_detection_mode(self, mode):
        """Map deprecated remote AI modes to the local AI runtime path."""
        mode_str = str(mode).strip()
        if mode_str in (DetectionMode.HYBRIDNETS.value, DetectionMode.SUPERCOMBO.value):
            return DetectionMode.AI_LOCAL.value
        return mode_str

    def _poll_local_ai_messages(self):
        """Cache the latest local AI and actuator payloads."""
        lane_payload = self.localLanePerceptionSubscriber.receive()
        if lane_payload is not None:
            self._last_local_lane_payload = lane_payload

        status_payload = self.localPerceptionStatusSubscriber.receive()
        if status_payload is not None:
            self._last_local_perception_status = status_payload

        actuator_payload = self.actuatorStatusSubscriber.receive()
        if actuator_payload is not None:
            self._last_actuator_status = actuator_payload

    def _fit_lane_points_line(self, lane_points, img_h, img_w):
        """Compatibility wrapper for generic AI lane-point geometry."""
        return self._fit_remote_lane_line(lane_points, img_h, img_w)

    def _normalize_lane_points(self, lane_points, img_h):
        """Compatibility wrapper for generic AI lane-point geometry."""
        return self._normalize_remote_lane_points(lane_points, img_h)

    def _lane_points_to_lines(self, lane_points, img_h, img_w):
        """Compatibility wrapper for generic AI lane-point geometry."""
        return self._remote_lane_points_to_lines(lane_points, img_h, img_w)

    def _lane_side_points_to_lines(self, lane_side_points, img_h, img_w):
        """Map explicit left/right lane polylines to representative lines."""
        if not isinstance(lane_side_points, dict):
            return None, None

        avg_left = None
        avg_right = None

        left_points = lane_side_points.get('left', [])
        right_points = lane_side_points.get('right', [])

        if isinstance(left_points, (list, tuple)):
            avg_left = self._fit_remote_lane_line(left_points, img_h, img_w)
        if isinstance(right_points, (list, tuple)):
            avg_right = self._fit_remote_lane_line(right_points, img_h, img_w)

        return avg_left, avg_right

    def _coerce_explicit_line(self, line_value, img_h, img_w):
        """Parse an explicitly provided line segment from the local AI payload."""
        if not isinstance(line_value, (list, tuple)):
            return None

        values = line_value
        if len(values) == 1 and isinstance(values[0], (list, tuple, np.ndarray)):
            values = values[0]
        if len(values) < 4:
            return None

        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in values[:4]]
        except (TypeError, ValueError):
            return None

        return np.array([[
            max(0, min(img_w - 1, x1)),
            max(0, min(img_h - 1, y1)),
            max(0, min(img_w - 1, x2)),
            max(0, min(img_h - 1, y2)),
        ]], dtype=np.int32)

    def _lane_side_lines_to_lines(self, lane_side_lines, img_h, img_w):
        """Use explicit left/right line segments provided by the local AI payload."""
        if not isinstance(lane_side_lines, dict):
            return None, None
        return (
            self._coerce_explicit_line(lane_side_lines.get('left', []), img_h, img_w),
            self._coerce_explicit_line(lane_side_lines.get('right', []), img_h, img_w),
        )

    def _normalize_lane_side_sources(self, lane_side_sources):
        """Normalize side-source metadata coming from the local AI payload."""
        normalized = {'left': 'none', 'right': 'none'}
        if not isinstance(lane_side_sources, dict):
            return normalized
        for side in ('left', 'right'):
            value = lane_side_sources.get(side, 'none')
            normalized[side] = str(value or 'none')
        return normalized

    def _prepare_local_ai_side_masks(self, side_masks, lane_side_sources, lane_side_lines, img_h, img_w):
        """Normalize local-AI masks/lines and resolve ambiguous single-lane assignments."""
        normalized_masks = {'left': None, 'right': None}
        normalized_lines = {'left': None, 'right': None}
        resolution = None

        if isinstance(side_masks, dict):
            for side in ('left', 'right'):
                mask = side_masks.get(side)
                if not isinstance(mask, np.ndarray) or mask.size == 0:
                    continue
                mask_2d = mask if mask.ndim == 2 else mask[..., 0]
                if mask_2d.ndim != 2 or mask_2d.shape[:2] != (img_h, img_w):
                    continue
                if not np.any(mask_2d):
                    continue
                normalized_masks[side] = mask_2d

        if isinstance(lane_side_lines, dict):
            normalized_lines['left'] = self._coerce_explicit_line(lane_side_lines.get('left', []), img_h, img_w)
            normalized_lines['right'] = self._coerce_explicit_line(lane_side_lines.get('right', []), img_h, img_w)

        duplicate_collapse = self._collapse_duplicate_local_ai_sides(
            normalized_masks,
            normalized_lines,
            lane_side_sources,
            img_h,
            img_w,
        )
        self._last_local_ai_duplicate_collapse = duplicate_collapse

        left_line = normalized_lines['left']
        right_line = normalized_lines['right']

        if left_line is not None and right_line is not None:
            self._last_two_line_left = left_line.copy()
            self._last_two_line_right = right_line.copy()
            self._last_two_line_ts = time.time()

        detected_sides = [side for side in ('left', 'right') if normalized_masks[side] is not None]
        if len(detected_sides) != 1:
            return normalized_masks, normalized_lines, resolution

        detected_side = detected_sides[0]
        detected_source = str(lane_side_sources.get(detected_side, 'none') or 'none')
        if not detected_source.startswith('guessed_single'):
            return normalized_masks, normalized_lines, resolution

        candidate_line = left_line if detected_side == 'left' else right_line
        if candidate_line is None:
            return normalized_masks, normalized_lines, resolution

        resolved_side, resolution_source = self._resolve_ambiguous_local_lane(
            candidate_line,
            img_h,
            img_w,
            prev_seen_side=getattr(self, 'last_seen_side', None),
        )
        if resolved_side in ('left', 'right') and resolved_side != detected_side:
            normalized_masks[resolved_side] = normalized_masks[detected_side]
            normalized_masks[detected_side] = None
            normalized_lines[resolved_side] = normalized_lines[detected_side]
            normalized_lines[detected_side] = None

        resolution = {
            'detected_side': detected_side,
            'resolved_side': resolved_side or detected_side,
            'source': detected_source,
            'resolution_source': resolution_source,
        }
        return normalized_masks, normalized_lines, resolution

    @staticmethod
    def _prefer_local_lane_line(explicit_line, fitted_line):
        """Prefer explicit local-AI geometry when both a line and polyline exist."""
        return explicit_line if explicit_line is not None else fitted_line

    def _line_x_at_y(self, line, y_target):
        """Project a representative line segment to a specific image row."""
        if line is None:
            return None
        x1, y1, x2, y2 = [float(v) for v in line[0]]
        dy = y2 - y1
        if abs(dy) < 1e-6:
            return (x1 + x2) * 0.5
        t = (float(y_target) - y1) / dy
        return x1 + t * (x2 - x1)

    def _mask_guidance_line(self, guidance, side, img_h, img_w):
        """Build a representative line segment from local mask-guidance points."""
        if not isinstance(guidance, dict) or side not in ("left", "right"):
            return None

        if side == "left":
            bottom_key = "bottom_left_x"
            ref_key = "left_x"
        else:
            bottom_key = "bottom_right_x"
            ref_key = "right_x"

        x_bottom = guidance.get(bottom_key)
        x_ref = guidance.get(ref_key)
        y_bottom = guidance.get("bottom_y")
        y_ref = guidance.get("reference_y")

        if x_bottom is None or x_ref is None or y_bottom is None or y_ref is None:
            return None

        try:
            x_bottom = int(round(float(x_bottom)))
            x_ref = int(round(float(x_ref)))
            y_bottom = int(round(float(y_bottom)))
            y_ref = int(round(float(y_ref)))
        except (TypeError, ValueError):
            return None

        x_bottom = max(0, min(img_w - 1, x_bottom))
        x_ref = max(0, min(img_w - 1, x_ref))
        y_bottom = max(0, min(img_h - 1, y_bottom))
        y_ref = max(0, min(img_h - 1, y_ref))

        if y_bottom == y_ref:
            y_ref = max(0, y_bottom - 1)
            if y_ref == y_bottom:
                return None

        return np.array([[x_bottom, y_bottom, x_ref, y_ref]], dtype=np.int32)

    def _mask_guidance_lines(self, guidance, img_h, img_w):
        """Return (left_line, right_line) built from local mask-guidance geometry."""
        if not isinstance(guidance, dict):
            return None, None

        detected_sides = set(guidance.get("detected_sides", ()))
        left_line = None
        right_line = None
        if "left" in detected_sides:
            left_line = self._mask_guidance_line(guidance, "left", img_h, img_w)
        if "right" in detected_sides:
            right_line = self._mask_guidance_line(guidance, "right", img_h, img_w)
        return left_line, right_line

    def _record_lane_observation(self, observed_sides, duplicate_collapse=None):
        """Store short-term lane visibility history for line-loss context."""
        if not hasattr(self, '_lane_observation_history') or self._lane_observation_history is None:
            self._lane_observation_history = deque(maxlen=12)
        sides = tuple(side for side in ('left', 'right') if side in tuple(observed_sides or ()))
        visible_side = sides[0] if len(sides) == 1 else None
        inferred_lost_side = None
        inferred_curve_direction = 0
        context_sources = []

        last_two_line_ts = float(getattr(self, '_last_two_line_ts', 0.0) or 0.0)
        reference_ttl = max(1.0, float(getattr(self, 'local_ai_max_result_age', 0.35)) * 3.0)
        has_recent_two_line_reference = (
            len(sides) == 1 and
            last_two_line_ts and
            (time.time() - last_two_line_ts) <= reference_ttl and
            getattr(self, '_last_two_line_left', None) is not None and
            getattr(self, '_last_two_line_right', None) is not None
        )
        _in_waypoint_mode = (
            getattr(self, '_tracking_state', None) is not None and
            bool(getattr(self._tracking_state, 'waypoint_mode_active', False))
        )
        if has_recent_two_line_reference and not _in_waypoint_mode:
            inferred_lost_side = 'right' if visible_side == 'left' else 'left'
            inferred_curve_direction = 1 if visible_side == 'left' else -1
            context_sources.append(f"lost_{inferred_lost_side}")
            curve_state = str(getattr(self, '_curve_state', 'STRAIGHT'))
            curve_direction = int(getattr(self, '_curve_direction', 0) or 0)
            if curve_direction != 0 and curve_state in ("ENTERING", "IN_CURVE", "EXITING"):
                inferred_curve_direction = curve_direction
                context_sources.append("curve_state")
            else:
                recent_curve_direction = int(getattr(self, '_recent_curve_direction', 0) or 0)
                recent_curve_frames = int(getattr(self, '_recent_curve_direction_frames', 0) or 0)
                if recent_curve_direction != 0 and recent_curve_frames > 0:
                    inferred_curve_direction = recent_curve_direction
                    context_sources.append("recent_curve")

        previous = self._lane_observation_history[-1] if self._lane_observation_history else None
        if previous is not None:
            previous_sides = set(previous.get('sides', ()))
            current_sides = set(sides)
            delta_lost = tuple(sorted(previous_sides - current_sides))
            if len(delta_lost) == 1 and delta_lost[0] not in context_sources:
                context_sources.append(f"transition_lost_{delta_lost[0]}")

        entry = {
            'timestamp': time.time(),
            'sides': sides,
            'visible_side': visible_side,
            'count': len(sides),
            'inferred_lost_side': inferred_lost_side,
            'inferred_curve_direction': inferred_curve_direction,
            'duplicate_collapse': bool(isinstance(duplicate_collapse, dict)),
            'context_sources': tuple(context_sources),
        }
        if isinstance(duplicate_collapse, dict):
            entry['duplicate_keep_side'] = duplicate_collapse.get('kept_side')
            entry['duplicate_resolution_source'] = duplicate_collapse.get('resolution_source')

        self._lane_observation_history.append(entry)
        return entry

    @staticmethod
    def _lane_observation_debug_fields(entry):
        """Flatten a lane observation entry into debug-friendly fields."""
        if not isinstance(entry, dict):
            return {}
        return {
            'observed_lane_sides': ",".join(entry.get('sides', ())),
            'observed_lane_count': int(entry.get('count', 0) or 0),
            'visible_lane_side': entry.get('visible_side'),
            'lost_lane_side': entry.get('inferred_lost_side'),
            'observed_curve_direction': int(entry.get('inferred_curve_direction', 0) or 0),
            'lane_context_sources': ",".join(entry.get('context_sources', ())),
            'duplicate_line_collapse_active': bool(entry.get('duplicate_collapse', False)),
        }

    def _mask_x_at_y(self, mask, y_target, search_radius=8):
        """Return the average x coordinate of a binary mask at a given row."""
        if not isinstance(mask, np.ndarray) or mask.size == 0:
            return None

        mask_2d = mask if mask.ndim == 2 else mask[..., 0]
        if mask_2d.ndim != 2 or mask_2d.size == 0:
            return None

        height, _ = mask_2d.shape[:2]
        y_target = int(np.clip(y_target, 0, height - 1))
        search_radius = max(0, int(search_radius))

        for delta in range(search_radius + 1):
            rows = [y_target] if delta == 0 else [y_target - delta, y_target + delta]
            for row in rows:
                if row < 0 or row >= height:
                    continue
                xs = np.where(mask_2d[row] > 0)[0]
                if xs.size > 0:
                    return float(xs.mean())
        return None

    def _collapse_duplicate_local_ai_sides(self, side_masks, side_lines, lane_side_sources, img_h, img_w):
        """Collapse duplicate left/right local-AI masks that map to one physical line."""
        left_mask = side_masks.get('left')
        right_mask = side_masks.get('right')
        if not isinstance(left_mask, np.ndarray) or not isinstance(right_mask, np.ndarray):
            return None
        if left_mask.size == 0 or right_mask.size == 0:
            return None
        if left_mask.shape[:2] != right_mask.shape[:2]:
            return None

        left_pixels = int(np.count_nonzero(left_mask))
        right_pixels = int(np.count_nonzero(right_mask))
        if left_pixels == 0 or right_pixels == 0:
            return None

        overlap_pixels = int(np.count_nonzero((left_mask > 0) & (right_mask > 0)))
        overlap_ratio = float(overlap_pixels) / float(min(left_pixels, right_pixels)) if overlap_pixels else 0.0

        sample_rows = [
            max(0, min(img_h - 1, int(round(img_h * ratio))))
            for ratio in (0.95, 0.75, 0.6)
        ]
        gaps = []
        for row in sample_rows:
            left_x = self._mask_x_at_y(left_mask, row, search_radius=6)
            right_x = self._mask_x_at_y(right_mask, row, search_radius=6)
            if left_x is not None and right_x is not None:
                gaps.append(abs(float(right_x) - float(left_x)))

        mean_gap_px = (sum(gaps) / len(gaps)) if gaps else None
        line_gap_px = None
        left_line = side_lines.get('left')
        right_line = side_lines.get('right')
        if left_line is not None and right_line is not None:
            line_gaps = []
            for row in sample_rows:
                left_line_x = self._line_x_at_y(left_line, row)
                right_line_x = self._line_x_at_y(right_line, row)
                if left_line_x is not None and right_line_x is not None:
                    line_gaps.append(abs(float(right_line_x) - float(left_line_x)))
            if line_gaps:
                line_gap_px = sum(line_gaps) / len(line_gaps)

        duplicate_gap_limit = max(6.0, float(img_w) * 0.04)
        duplicate_line_gap_limit = max(4.0, float(img_w) * 0.02)
        is_duplicate = overlap_ratio >= 0.35
        if mean_gap_px is not None:
            is_duplicate = is_duplicate or mean_gap_px <= duplicate_gap_limit
        if line_gap_px is not None:
            is_duplicate = is_duplicate or line_gap_px <= duplicate_line_gap_limit
        if not is_duplicate:
            return None

        merged_mask = cv2.bitwise_or(left_mask, right_mask)
        left_source = str(lane_side_sources.get('left', 'none') or 'none')
        right_source = str(lane_side_sources.get('right', 'none') or 'none')
        resolution_source = 'source_priority'

        keep_side = None
        keep_source = 'guessed_single'
        if left_source != 'guessed_single' and right_source == 'guessed_single':
            keep_side = 'left'
            keep_source = left_source
        elif right_source != 'guessed_single' and left_source == 'guessed_single':
            keep_side = 'right'
            keep_source = right_source

        if keep_side is None:
            candidate_line = side_lines.get('left')
            if candidate_line is None:
                candidate_line = side_lines.get('right')
            if candidate_line is not None:
                resolved_side, resolution_source = self._resolve_ambiguous_local_lane(
                    candidate_line,
                    img_h,
                    img_w,
                    prev_seen_side=getattr(self, 'last_seen_side', None),
                )
                if resolved_side in ('left', 'right'):
                    keep_side = resolved_side

        if keep_side is None:
            resolution_source = 'frame_center'
            y_ref = int(max(0, min(img_h - 1, round(img_h * (1.0 - float(getattr(self, 'lookahead', 0.4)))))))
            merged_x = self._mask_x_at_y(merged_mask, y_ref, search_radius=8)
            if merged_x is None:
                merged_x = float(img_w) / 2.0
            keep_side = 'left' if merged_x < (img_w / 2.0) else 'right'

        drop_side = 'right' if keep_side == 'left' else 'left'
        keep_line = side_lines.get(keep_side)
        drop_line = side_lines.get(drop_side)
        if keep_line is None and drop_line is not None:
            keep_line = drop_line

        side_masks[keep_side] = merged_mask
        side_masks[drop_side] = None
        side_lines[keep_side] = keep_line
        side_lines[drop_side] = None
        lane_side_sources[keep_side] = keep_source
        lane_side_sources[drop_side] = 'none'

        return {
            'kept_side': keep_side,
            'dropped_side': drop_side,
            'resolution_source': resolution_source,
            'overlap_ratio': float(overlap_ratio),
            'mean_gap_px': None if mean_gap_px is None else float(mean_gap_px),
            'line_gap_px': None if line_gap_px is None else float(line_gap_px),
        }

    def _sample_mask_boundary(self, mask, y_rows):
        """Sample a lane mask boundary at the requested rows."""
        return [self._mask_x_at_y(mask, y_value) for y_value in y_rows]

    def _mask_y_bounds(self, mask):
        """Return (top_y, bottom_y) for the active pixels of a binary mask."""
        if not isinstance(mask, np.ndarray) or mask.size == 0:
            return None, None
        mask_2d = mask if mask.ndim == 2 else mask[..., 0]
        if mask_2d.ndim != 2 or mask_2d.size == 0:
            return None, None
        ys = np.where(mask_2d > 0)[0]
        if ys.size == 0:
            return None, None
        return int(ys.min()), int(ys.max())

    def _confirm_single_line_outer_curve(self, line, side, img_h, img_w, curve_strength,
                                         heading_raw=0.0):
        """Confirm that a curve-like single line is likely the outer boundary.

        The outer boundary of a curve tilts in the *same* direction as the turn
        when viewed from behind the car:
          - Outer RIGHT line (left curve)  → tilts LEFT  → heading_raw < 0
          - Outer LEFT  line (right curve) → tilts RIGHT → heading_raw > 0
        If the heading sign is opposite the line is the INNER boundary of a
        curve turning the other way; outer-line bias must not be applied.
        """
        result = {
            'confirmed': False,
            'sources': [],
            'state_confirmed': False,
            'history_confirmed': False,
            'continuity_confirmed': False,
        }
        if line is None or side not in ('left', 'right'):
            return result

        # Heading-delta guard: compare the current heading to the straight-road
        # reference for this line side.  An outer boundary tilts MORE in the
        # curve direction than the reference (delta in the curve direction):
        #   outer RIGHT (left  curve) → delta ≤ 0  (more leftward than ref)
        #   outer LEFT  (right curve) → delta ≥ 0  (more rightward than ref)
        # Inner lines tilt LESS or in the opposite direction relative to ref.
        # Using the delta (not the absolute sign of heading_raw) correctly
        # handles shallow curves where the inner line still appears to tilt
        # in the "wrong" absolute direction due to straight-road perspective.
        ref_angle = float(getattr(
            self,
            '_single_line_heading_ref_left' if side == 'left' else '_single_line_heading_ref_right',
            0.0,
        ))
        heading_delta_from_ref = heading_raw - ref_angle
        is_outer_heading = (
            (side == 'right' and heading_delta_from_ref <= 0) or
            (side == 'left'  and heading_delta_from_ref >= 0)
        )
        allow_state_confirmation = bool(is_outer_heading)
        allow_continuity_confirmation = bool(is_outer_heading)

        min_curve_strength = max(0.0, float(getattr(self, 'single_line_curve_confirm_threshold', 0.35)))
        if curve_strength < min_curve_strength:
            return result

        curve_direction = int(getattr(self, '_curve_direction', 0) or 0)
        curve_state = str(getattr(self, '_curve_state', 'STRAIGHT'))

        state_confirmed = (
            curve_state in ("ENTERING", "IN_CURVE", "EXITING") and
            (
                (curve_direction == 1 and side == 'left') or
                (curve_direction == -1 and side == 'right')
            )
        )
        if allow_state_confirmation and state_confirmed:
            result['state_confirmed'] = True
            result['sources'].append('curve_state')

        y_ref = int(max(0, min(img_h - 1, round(img_h * (1.0 - float(getattr(self, 'lookahead', 0.4)))))))
        line_x = self._line_x_at_y(line, y_ref)
        if line_x is not None:
            reference_ttl = max(1.0, float(getattr(self, 'local_ai_max_result_age', 0.35)) * 3.0)
            last_two_line_ts = float(getattr(self, '_last_two_line_ts', 0.0) or 0.0)
            if last_two_line_ts and (time.time() - last_two_line_ts) <= reference_ttl:
                ref_line = getattr(
                    self,
                    '_last_two_line_left' if side == 'left' else '_last_two_line_right',
                    None,
                )
                ref_x = self._line_x_at_y(ref_line, y_ref)
                if ref_x is not None:
                    cached_lane_width = float(getattr(self, '_last_local_ai_lane_width_px', 0.0) or 0.0)
                    cached_px_per_cm = float(getattr(self, '_last_px_per_cm', 0.0) or 0.0)
                    lane_width_px = cached_lane_width if cached_lane_width > 1.0 else 0.0
                    if lane_width_px <= 1.0 and cached_px_per_cm > 0.5:
                        lane_width_px = cached_px_per_cm * float(self.lane_width_cm)
                    if lane_width_px <= 1.0:
                        lane_width_px = float(img_w) * 0.45

                    history_match_limit_px = max(8.0, lane_width_px * 0.45)
                    if abs(float(line_x) - float(ref_x)) <= history_match_limit_px:
                        result['history_confirmed'] = True
                        result['sources'].append('history')

        lane_history = getattr(self, '_lane_observation_history', None)
        if lane_history:
            latest = lane_history[-1]
            latest_visible_side = latest.get('visible_side')
            expected_lost_side = 'right' if side == 'left' else 'left'
            if (
                latest_visible_side == side and
                latest.get('inferred_lost_side') == expected_lost_side
            ):
                result['history_confirmed'] = True
                result['sources'].append(f'lost_{expected_lost_side}')
            if bool(latest.get('duplicate_collapse', False)) and latest_visible_side == side:
                result['history_confirmed'] = True
                result['sources'].append('duplicate_collapse')

        continuity_frames = int(getattr(
            self,
            'consecutive_single_left' if side == 'left' else 'consecutive_single_right',
            0,
        ) or 0)
        min_continuity_frames = max(1, int(getattr(self, 'single_line_outer_confirm_frames', 2)))
        if allow_continuity_confirmation and continuity_frames >= min_continuity_frames:
            result['continuity_confirmed'] = True
            result['sources'].append('continuity')

        if result['sources']:
            result['sources'] = list(dict.fromkeys(result['sources']))
        result['confirmed'] = bool(result['sources'])
        return result

    def _estimate_virtual_boundary_from_single_side(
        self,
        side,
        visible_x,
        lane_width_px,
        img_w,
        target_factor=0.5,
    ):
        """Estimate the missing lane boundary from one visible side.

        target_factor defines where the lane center should be relative to the
        visible side:
        - 0.50 -> exact lane center
        - <0.50 -> stay closer to the visible side (safer in one-line curves)
        """
        if visible_x is None:
            return None

        lane_width_px = max(1.0, float(lane_width_px))
        target_factor = max(0.20, min(0.50, float(target_factor)))
        if side == 'left':
            virtual_x = float(visible_x) + (2.0 * target_factor * lane_width_px)
        else:
            virtual_x = float(visible_x) - (2.0 * target_factor * lane_width_px)

        return max(0.0, min(float(img_w - 1), virtual_x))

    def _compute_bev_heading(self, side_masks, img_h, img_w, bottom_y, reference_y):
        """Estimate road heading from lane masks in bird's eye view.

        Projects both lane masks into BEV space and samples the midpoint
        between the two lines at two y-rows: bottom_y (near field) and
        reference_y (lookahead).  The heading is atan2(dx, dy) where dx and
        dy are both in BEV pixels, which are proportional to real metric
        distances.  This eliminates the perspective foreshortening that
        inflates the heading estimate computed in image space.

        Returns heading_rad (float) or None when BEV is unavailable or masks
        are too sparse to produce a reliable estimate.
        """
        if not getattr(self, 'perspective_initialized', False):
            return None
        if self.bev_cm_per_px <= 0.0:
            return None
        if not isinstance(side_masks, dict):
            return None

        try:
            warp_size = (img_w, img_h)
            bev_masks = {}
            for side in ('left', 'right'):
                m = side_masks.get(side)
                if not isinstance(m, np.ndarray) or m.size == 0:
                    return None
                m2d = m if m.ndim == 2 else m[..., 0]
                if m2d.shape[:2] != (img_h, img_w):
                    return None
                bev_masks[side] = cv2.warpPerspective(
                    m2d.astype(np.uint8), self.perspective_M, warp_size
                )

            min_pixels = max(4, int(img_w * 4))

            def _sample_bev_midpoint_x(y_row):
                y_row = max(0, min(img_h - 1, int(y_row)))
                xs = {}
                for side in ('left', 'right'):
                    bev = bev_masks[side]
                    # Sample the mean x of active pixels in a band around y_row.
                    band_half = max(2, img_h // 40)
                    y0 = max(0, y_row - band_half)
                    y1 = min(img_h, y_row + band_half + 1)
                    band = bev[y0:y1, :]
                    cols = np.where(band > 0)[1]
                    if len(cols) < min_pixels:
                        return None
                    xs[side] = float(np.mean(cols))
                return (xs['left'] + xs['right']) / 2.0

            mid_bottom = _sample_bev_midpoint_x(bottom_y)
            mid_ref = _sample_bev_midpoint_x(reference_y)

            if mid_bottom is None or mid_ref is None:
                return None

            dy = max(1.0, float(bottom_y - reference_y))
            return math.atan2(mid_ref - mid_bottom, dy)

        except Exception:
            return None

    def _build_local_mask_guidance(self, side_masks, img_h, img_w, side_lines=None):
        """Build direct steering guidance from the raw local-AI lane masks."""
        self._last_local_ai_mask_reject_reason = None
        if not isinstance(side_masks, dict):
            return None

        normalized_masks = {'left': None, 'right': None}
        for side in ('left', 'right'):
            mask = side_masks.get(side)
            if not isinstance(mask, np.ndarray) or mask.size == 0:
                continue
            mask_2d = mask if mask.ndim == 2 else mask[..., 0]
            if mask_2d.ndim != 2 or mask_2d.shape[:2] != (img_h, img_w):
                continue
            if not np.any(mask_2d):
                continue
            normalized_masks[side] = mask_2d

        roi_top_factor = float(getattr(self, 'ai_local_roi_height_start', self.roi_height_start))
        roi_top = max(0, min(img_h - 1, int(roi_top_factor * img_h)))
        roi_bottom = max(roi_top, min(img_h - 1, int(self.roi_height_end * img_h) - 1))

        mask_bounds = {}
        for side in ('left', 'right'):
            mask = normalized_masks[side]
            if mask is None:
                continue
            top_y, bottom_y = self._mask_y_bounds(mask)
            if top_y is None or bottom_y is None:
                continue
            top_y = max(roi_top, min(roi_bottom, int(top_y)))
            bottom_y = max(roi_top, min(roi_bottom, int(bottom_y)))
            if bottom_y < top_y:
                continue
            mask_bounds[side] = (top_y, bottom_y)

        if not mask_bounds:
            return None

        # Dynamic anchor row: use where the mask actually starts (nearest visible row),
        # instead of forcing the ROI bottom. This prevents overreacting to far-curve points
        # when the camera is pitched up and near-field lane pixels are missing.
        measurement_bottom_y = min(bounds[1] for bounds in mask_bounds.values())
        measurement_top_y = max(bounds[0] for bounds in mask_bounds.values())
        measurement_bottom_y = max(roi_top, min(roi_bottom, int(measurement_bottom_y)))
        measurement_top_y = max(roi_top, min(measurement_bottom_y, int(measurement_top_y)))

        reference_y = int(round(img_h * (1.0 - float(getattr(self, 'lookahead', 0.4)))))
        reference_y = max(measurement_top_y, min(measurement_bottom_y, reference_y))
        if measurement_bottom_y - reference_y < 1:
            reference_y = max(measurement_top_y, measurement_bottom_y - max(1, img_h // 8))
        if measurement_bottom_y - reference_y < 1:
            reference_y = max(0, measurement_bottom_y - 1)

        y_rows = [measurement_bottom_y, reference_y]

        samples = {
            'left': {'bottom': None, 'reference': None},
            'right': {'bottom': None, 'reference': None},
        }
        detected_sides = []
        for side in ('left', 'right'):
            mask = normalized_masks[side]
            if mask is None:
                continue
            bottom_x, reference_x = self._sample_mask_boundary(mask, y_rows)
            if reference_x is None:
                reference_x = bottom_x
            if bottom_x is None:
                bottom_x = reference_x
            if bottom_x is None and reference_x is None:
                continue
            samples[side]['bottom'] = float(bottom_x) if bottom_x is not None else None
            samples[side]['reference'] = float(reference_x) if reference_x is not None else None
            detected_sides.append(side)

        if len(detected_sides) == 0:
            return None

        visible_side = detected_sides[0] if len(detected_sides) == 1 else None
        visible_line = None
        if visible_side is not None and isinstance(side_lines, dict):
            candidate_line = side_lines.get(visible_side)
            if isinstance(candidate_line, np.ndarray) and candidate_line.ndim == 2 and candidate_line.shape[1] >= 4:
                visible_line = candidate_line
                for row_key, y_value in (('bottom', measurement_bottom_y), ('reference', reference_y)):
                    if samples[visible_side][row_key] is None:
                        projected_x = self._line_x_at_y(visible_line, y_value)
                        if projected_x is not None:
                            samples[visible_side][row_key] = float(projected_x)

        width_samples = []
        for row_key in ('bottom', 'reference'):
            left_x = samples['left'][row_key]
            right_x = samples['right'][row_key]
            if left_x is not None and right_x is not None and right_x > left_x:
                width_samples.append(float(right_x - left_x))

        if width_samples:
            lane_width_px = sum(width_samples) / len(width_samples)
        else:
            lane_width_px = self._resolve_nominal_lane_width_px(img_w)

        narrow_width_collapse = None
        if len(detected_sides) >= 2 and width_samples:
            cached_lane_width = float(getattr(self, '_last_local_ai_lane_width_px', 0.0) or 0.0)
            min_lane_width_px = max(10.0, float(img_w) * 0.10)
            if cached_lane_width > 1.0:
                min_lane_width_px = max(min_lane_width_px, cached_lane_width * 0.45)
            if float(lane_width_px) < min_lane_width_px:
                overlap_x_values = [
                    samples['left']['reference'],
                    samples['right']['reference'],
                    samples['left']['bottom'],
                    samples['right']['bottom'],
                ]
                overlap_x_values = [float(x) for x in overlap_x_values if x is not None]
                if not overlap_x_values:
                    self._last_local_ai_mask_reject_reason = (
                        f"lane_width_too_small({float(lane_width_px):.1f}<{min_lane_width_px:.1f})"
                    )
                    return None

                overlap_x = float(sum(overlap_x_values) / len(overlap_x_values))
                overlap_top_y = max(0, min(reference_y, measurement_bottom_y - 1))
                overlap_line = np.array([[
                    int(round(max(0.0, min(float(img_w - 1), overlap_x)))),
                    int(measurement_bottom_y),
                    int(round(max(0.0, min(float(img_w - 1), overlap_x)))),
                    int(overlap_top_y),
                ]], dtype=np.int32)
                resolved_side, resolution_source = self._resolve_ambiguous_local_lane(
                    overlap_line,
                    img_h,
                    img_w,
                    prev_seen_side=getattr(self, 'last_seen_side', None),
                )
                if resolved_side not in ('left', 'right'):
                    resolved_side = 'left' if overlap_x < (img_w / 2.0) else 'right'
                    resolution_source = 'frame_center'

                dropped_side = 'right' if resolved_side == 'left' else 'left'
                samples[dropped_side]['bottom'] = None
                samples[dropped_side]['reference'] = None
                detected_sides = [resolved_side]
                visible_side = resolved_side
                visible_line = None
                if isinstance(side_lines, dict):
                    candidate_line = side_lines.get(visible_side)
                    if (
                        isinstance(candidate_line, np.ndarray) and
                        candidate_line.ndim == 2 and
                        candidate_line.shape[1] >= 4
                    ):
                        visible_line = candidate_line
                lane_width_px = max(float(min_lane_width_px), float(self._resolve_nominal_lane_width_px(img_w)))
                narrow_width_collapse = {
                    'kept_side': resolved_side,
                    'dropped_side': dropped_side,
                    'resolution_source': resolution_source,
                    'line_gap_px': float(lane_width_px if not width_samples else sum(width_samples) / len(width_samples)),
                    'line_gap_limit_px': float(min_lane_width_px),
                    'trigger': 'lane_width_too_small',
                }
                self._last_local_ai_mask_reject_reason = None

        curve_context = str(getattr(self, '_curve_state', 'STRAIGHT')) in ("ENTERING", "IN_CURVE", "EXITING")
        prev_seen_side = str(getattr(self, 'last_seen_side', 'none') or 'none').lower()
        used_virtual_boundary = False
        if len(detected_sides) == 1:
            missing_side = 'right' if visible_side == 'left' else 'left'
            if curve_context:
                virtual_target_factor = max(
                    0.25,
                    min(0.50, float(getattr(self, 'single_line_offset_factor', 0.42)))
                )
            else:
                virtual_target_factor = 0.50
            for row_key in ('bottom', 'reference'):
                visible_x = samples[visible_side][row_key]
                if visible_x is None:
                    continue
                samples[missing_side][row_key] = self._estimate_virtual_boundary_from_single_side(
                    visible_side,
                    visible_x,
                    lane_width_px,
                    img_w,
                    target_factor=virtual_target_factor,
                )
            used_virtual_boundary = True

        lane_observation = self._record_lane_observation(
            tuple(detected_sides),
            duplicate_collapse=(
                narrow_width_collapse
                if narrow_width_collapse is not None
                else getattr(self, '_last_local_ai_duplicate_collapse', None)
            ),
        )

        left_bottom_x = samples['left']['bottom']
        right_bottom_x = samples['right']['bottom']
        left_ref_x = samples['left']['reference']
        right_ref_x = samples['right']['reference']
        required_values = [left_bottom_x, right_bottom_x, left_ref_x, right_ref_x]
        if any(value is None for value in required_values):
            return None

        midpoint_bottom_x = (float(left_bottom_x) + float(right_bottom_x)) / 2.0
        midpoint_x = (float(left_ref_x) + float(right_ref_x)) / 2.0
        dy = max(1.0, float(measurement_bottom_y - reference_y))
        _raw_two_line_heading_rad = math.atan2(midpoint_x - midpoint_bottom_x, dy)

        # BEV heading refinement (two-line mode only).
        # The perspective-space estimate above is distorted: the vertical pixel
        # distance between measurement_bottom_y and reference_y is NOT
        # proportional to real forward distance, so atan2(dx_px, dy_px) inflates
        # the heading angle in curves.  _compute_bev_heading() repeats the same
        # midpoint-slope calculation in rectified BEV space where both dx and dy
        # are metrically correct.  When available it replaces the perspective
        # estimate; the same max_two_line_heading_deg cap still applies as a
        # safety guard against bad BEV warps.
        _bev_heading_active = False
        if (
            len(detected_sides) >= 2 and
            getattr(self, 'ai_local_bev_guidance', True) and
            getattr(self, 'perspective_initialized', False) and
            self.bev_cm_per_px > 0.0
        ):
            _bev_h = self._compute_bev_heading(
                normalized_masks, img_h, img_w, measurement_bottom_y, reference_y
            )
            if _bev_h is not None:
                _raw_two_line_heading_rad = _bev_h
                _bev_heading_active = True

        _max_two_line_heading_rad = math.radians(
            float(getattr(self, 'max_two_line_heading_deg', 25.0))
        )
        # When the raw heading exceeds the cap, the estimate is unreliable: the midpoint
        # shift between rows is dominated by the car's lateral offset (perspective effect)
        # rather than true heading. In this case we use 0 so the Stanley crosstrack term
        # drives correction without interference from a saturated heading estimate.
        # The crosstrack error is always physical and reliable; the heading term only adds
        # value when the car's angular alignment relative to the lane is well-measured.
        if abs(_raw_two_line_heading_rad) > _max_two_line_heading_rad:
            heading_rad = 0.0
        else:
            heading_rad = _raw_two_line_heading_rad
        # Apply EMA to smooth frame-to-frame oscillations caused by AI mask FPS < camera FPS.
        _heading_ema_alpha = float(getattr(self, 'two_line_heading_ema_alpha', 0.35))
        _prev_heading_ema = getattr(self, '_two_line_heading_ema', heading_rad)
        self._two_line_heading_ema = _heading_ema_alpha * heading_rad + (1.0 - _heading_ema_alpha) * _prev_heading_ema
        heading_rad = max(
            -_max_two_line_heading_rad,
            min(_max_two_line_heading_rad, self._two_line_heading_ema)
        )
        # Error at reference_y (lookahead): matches the overlay green dot and captures
        # curve geometry visible ahead. midpoint_bottom (near-field) spans nearly the
        # full frame width in perspective → gives ~0 error even in curves.
        error_px = midpoint_x - (img_w / 2.0)

        # Safety margin correction (two-line mode only, where both line positions are physical).
        # Ensures the car body (car_width wide) stays at least lane_safety_margin_cm from each
        # line center. left_ref_x / right_ref_x are the centers of the lane mask regions at
        # reference_y (inner boundary = mask center ≈ line center for thin lines).
        # The correction is additive: if the car is within the margin, the error is amplified
        # to push it back toward center; no effect when car is already safely inside.
        _safety_margin_cm = float(getattr(self, 'lane_safety_margin_cm', 5.0))
        if _safety_margin_cm > 0.0 and lane_width_px > 10.0:
            # Use the lane width AT reference_y (right_ref_x - left_ref_x) as the pixel
            # scale, NOT the averaged lane_width_px which is bottom-heavy due to perspective.
            # At reference_y: right_ref_x - left_ref_x = actual lane width in pixels at that
            # distance, so (right_ref_x - left_ref_x) / lane_width_cm gives the correct px/cm.
            _ref_lw_px = max(1.0, float(right_ref_x) - float(left_ref_x))
            _px_per_cm = _ref_lw_px / max(1e-3, float(self.lane_width_cm))
            _car_half_px = (float(self.car_width) / 2.0) * _px_per_cm
            _safety_px = _safety_margin_cm * _px_per_cm
            _car_center = img_w / 2.0
            # Distance from car edge to line center at reference_y (positive = clear)
            _left_clearance = (_car_center - _car_half_px) - float(left_ref_x)
            _right_clearance = float(right_ref_x) - (_car_center + _car_half_px)
            _correction = 0.0
            if _left_clearance < _safety_px:
                _correction += _safety_px - _left_clearance   # push right (+)
            if _right_clearance < _safety_px:
                _correction -= _safety_px - _right_clearance  # push left  (-)
            if abs(_correction) > 0.5:
                error_px = float(error_px) + _correction

        raw_error_px = float(error_px)
        draw_y = reference_y
        guidance_mode = 'midpoint_ref'
        single_line_projection_debug = None
        single_line_prefer_center = None
        single_line_streak = None

        # ── Camera-based DR yaw correction ────────────────────────────────────
        # With both lines visible the camera heading is reliable.  Compute the
        # car's estimated world-frame yaw = lane_tangent_at_waypoint + heading_rad
        # and push it to the tracking thread as a soft yaw correction hint.
        # Conditions: both lines, heading < 20°, car is moving.
        _dr_yaw_hint_deg = 0.0
        _dr_yaw_hint_conf = 0.0
        _ts_for_yaw = getattr(self, '_tracking_state', None)
        _tracking_cam_min_speed_mps = float(
            getattr(_config, 'TRACKING_CAMERA_CORRECTION_MIN_SPEED_MPS', 0.02) or 0.02
        )
        if (_ts_for_yaw is not None
                and self.is_line_following_active
                and abs(heading_rad) < math.radians(20.0)
                and abs(float(getattr(_ts_for_yaw, 'speed_mps', 0.0) or 0.0)) > _tracking_cam_min_speed_mps):
            _path_psi = float(getattr(_ts_for_yaw, 'path_psi', 0.0) or 0.0)
            _cam_world_yaw = _path_psi + heading_rad
            # Confidence: 1.0 when perfectly aligned, 0.0 at ±20°.
            _dr_yaw_hint_conf = max(0.0, 1.0 - abs(heading_rad) / math.radians(20.0))
            _ts_for_yaw.set_camera_yaw_hint(_cam_world_yaw, _dr_yaw_hint_conf)
            _dr_yaw_hint_deg = math.degrees(_cam_world_yaw)
        # ── End camera DR yaw correction ──────────────────────────────────────

        # Keep single-line heading references up to date from AI mask lines during
        # two-line mode.  In AI local mode avg_left/avg_right (BFMC Hough lines) are
        # null, so the legacy BFMC calibration path updates the refs with 0.0 each
        # frame, actively driving them toward zero.  When a 2→1 transition then
        # occurs, heading_delta = heading_raw − 0 ≈ 46° → camera_shift ≈ ±56 px
        # transient on the very first single-line frame.  Updating here from the
        # actual AI line headings eliminates that transient: heading_delta ≈ 0 right
        # from the start.
        if len(detected_sides) == 2 and isinstance(side_lines, dict):
            _heading_ref_alpha = max(0.01, min(1.0, float(getattr(self, 'single_line_heading_ref_alpha', 0.15))))
            for _side, _ref_attr in (
                ('left', '_single_line_heading_ref_left'),
                ('right', '_single_line_heading_ref_right'),
            ):
                _sl_line = side_lines.get(_side)
                if isinstance(_sl_line, np.ndarray) and _sl_line.ndim == 2 and _sl_line.shape[1] >= 4:
                    _h = self._compute_raw_line_heading(_sl_line)
                    _cur = float(getattr(self, _ref_attr, 0.0))
                    setattr(self, _ref_attr, (1.0 - _heading_ref_alpha) * _cur + _heading_ref_alpha * _h)

        # Shadow single-line computation (diagnostic only, never affects control).
        # When both lines are visible we know the correct error from two-line mode, so
        # compute what each single-line estimator would give independently and log the
        # discrepancy for calibration.
        _sl_shadow = None
        if len(detected_sides) == 2 and isinstance(side_lines, dict):
            _sl_shadow = self._compute_single_line_shadow_for_calib(
                side_lines,
                img_h,
                img_w,
                reference_y,
                raw_error_px,
                float(left_ref_x),
                float(right_ref_x),
            )
        self._last_sl_shadow_debug = _sl_shadow

        if visible_side is not None and visible_line is not None:
            streak_attr = 'consecutive_single_left' if visible_side == 'left' else 'consecutive_single_right'
            previous_streak = int(getattr(self, streak_attr, 0) or 0)
            single_line_streak = previous_streak + 1
            prefer_center_frames = max(1, int(getattr(self, 'single_line_prefer_center_frames', 2) or 0))
            # Only use center-reconstruction briefly on straight transitions.
            single_line_prefer_center = (
                (not curve_context) and
                prev_seen_side == "both" and
                single_line_streak <= prefer_center_frames
            )

            # When the visible line is nearly horizontal (high angle from vertical), the
            # line's x position at the lookahead reference_y is dominated by heading
            # perspective rather than actual lateral offset.  Projecting the lane center
            # from that position inflates the crosstrack error and diverges from what
            # two-line mode would compute for the same geometry.  In that case, keep the
            # midpoint_ref result (virtual boundary + mask x), which is consistent with
            # the two-line midpoint calculation and gives a smoother transition.
            _heading_raw_rad = self._compute_raw_line_heading(visible_line)
            _abs_heading_deg = abs(math.degrees(_heading_raw_rad))
            _heading_cutoff = float(getattr(self, 'single_line_physical_heading_cutoff_deg', 65.0))
            _use_physical_projection = _abs_heading_deg < _heading_cutoff

            # ── Transversal-line detection ───────────────────────────────────────
            # When the car is in a single-line curve and the detected line's
            # heading has drifted far in the "wrong" direction relative to the
            # straight-road reference, the model is tracking the perpendicular
            # end-of-track boundary line (linea transversal) instead of the real
            # lane marking.
            #   • outer RIGHT line (left  curve) → delta ≤ 0 (more left than ref)
            #   • outer LEFT  line (right curve) → delta ≥ 0 (more right than ref)
            # A delta going the OPPOSITE way means the line is not a lane boundary.
            # Detection criterion: |delta| > threshold AND sign is reversed.
            _transversal_ref_rad = float(getattr(
                self,
                '_single_line_heading_ref_left' if visible_side == 'left'
                else '_single_line_heading_ref_right',
                0.0,
            ))
            _transversal_delta_deg = (
                math.degrees(_heading_raw_rad) - math.degrees(_transversal_ref_rad)
            )
            _transversal_threshold_deg = float(
                getattr(self, 'single_line_transversal_delta_threshold_deg', 15.0)
            )
            # Only trigger when the reference has been meaningfully learned
            # (|ref| > 5°) to avoid false positives during initialisation.
            _ref_learned = abs(math.degrees(_transversal_ref_rad)) > 5.0
            # Only trigger transversal recovery when the graphml path confirms a real
            # curve.  On a straight section a perpendicular line is a stop-line marker,
            # not a curve-exit wall — applying the 20° heading bias there would wrongly
            # steer the car into the turn.  path_kappa from DeadReckoning / TrackGraph
            # is low (≈0.11) on straight segments and high (>0.5) inside tight turns.
            _tracking_state = getattr(self, '_tracking_state', None)
            _graph_kappa = abs(float(getattr(_tracking_state, 'path_kappa', 0.0))) \
                if _tracking_state is not None else 0.0
            _transversal_kappa_min = float(
                getattr(self, 'single_line_transversal_kappa_threshold', 0.30)
            )
            _is_transversal = (
                curve_context and
                _ref_learned and
                _graph_kappa >= _transversal_kappa_min and
                (
                    (visible_side == 'right' and _transversal_delta_deg > _transversal_threshold_deg) or
                    (visible_side == 'left'  and _transversal_delta_deg < -_transversal_threshold_deg)
                )
            )
            if _is_transversal:
                # The visible line is the perpendicular end-of-section marker
                # (linea transversal). The car must turn in _recovery_sign direction
                # for the entire duration the transversal is visible, until the real
                # exit lane line comes back into view.
                #
                # Two components:
                #  1. Lateral wall-avoidance: keep the transversal at a safe
                #     clearance on its expected side.  Once the car has rotated
                #     enough that the transversal crosses to the opposite half of
                #     the image the lateral correction drops to zero (the heading
                #     bias carries the remainder of the turn alone).
                #  2. Heading bias: constant turn-direction correction throughout
                #     the whole transversal phase.
                _recovery_sign = 1.0 if visible_side == 'right' else -1.0
                _trans_ref_x = self._line_x_at_y(visible_line, reference_y)
                if _trans_ref_x is None:
                    _trans_ref_x = img_w * (0.75 if visible_side == 'right' else 0.25)
                # Keep the transversal at ≈ 1/4 lane-width clearance from the car
                _clearance_px = float(lane_width_px) * float(
                    getattr(self, 'single_line_transversal_clearance_factor', 0.25)
                )
                if visible_side == 'right':
                    # Only push car right (away from wall); never use the transversal
                    # to push the car left when it has already turned past it.
                    error_px = max(
                        0.0,
                        float(_trans_ref_x) - _clearance_px - (img_w / 2.0),
                    )
                else:
                    error_px = min(
                        0.0,
                        float(_trans_ref_x) + _clearance_px - (img_w / 2.0),
                    )
                # Heading bias — forces the continuous turn regardless of lateral state
                _heading_bias_deg = float(
                    getattr(self, 'single_line_transversal_heading_bias_deg', 20.0)
                )
                heading_rad = _recovery_sign * math.radians(_heading_bias_deg)
                _use_physical_projection = False
                guidance_mode = 'transversal_recovery'
                _transversal_debug = {
                    'single_line_angle_from_vertical_deg': float(_abs_heading_deg),
                    'single_line_heading_raw_deg': float(math.degrees(_heading_raw_rad)),
                    'single_line_ref_angle_deg': float(math.degrees(_transversal_ref_rad)),
                    'single_line_heading_delta_from_ref_deg': float(_transversal_delta_deg),
                    'single_line_is_outer_heading': False,
                    'single_line_heading_reliability': 0.0,
                    'single_line_prefer_center': bool(single_line_prefer_center),
                    'single_line_streak': int(single_line_streak),
                    'single_line_curve_context': bool(curve_context),
                    'single_line_mode': 'transversal_recovery',
                    'single_line_transversal_detected': True,
                    'single_line_transversal_delta_deg': float(_transversal_delta_deg),
                    'single_line_transversal_threshold_deg': float(_transversal_threshold_deg),
                    'single_line_transversal_ref_x': float(_trans_ref_x),
                    'single_line_transversal_clearance_px': float(_clearance_px),
                    'single_line_transversal_heading_bias_deg': float(_heading_bias_deg),
                    'single_line_outer_curve_line': False,
                    'single_line_outer_confirmed': False,
                    'single_line_outer_bias_px': 0.0,
                    'single_line_curve_strength': 0.0,
                    'single_line_curve_heading_bias_deg': 0.0,
                }
                single_line_projection_debug = _transversal_debug
                self._last_single_line_projection_debug = _transversal_debug

            elif _use_physical_projection:
                single_error, single_heading = self._compute_single_line_error(
                    visible_line,
                    visible_side,
                    img_h,
                    img_w,
                    prefer_center=single_line_prefer_center,
                    physical_projection=True,
                    reference_y_override=reference_y,
                )
                single_line_projection_debug = dict(getattr(self, '_last_single_line_projection_debug', {}) or {})
                single_line_projection_debug['single_line_prefer_center'] = bool(single_line_prefer_center)
                single_line_projection_debug['single_line_streak'] = int(single_line_streak)
                single_line_projection_debug['single_line_curve_context'] = bool(curve_context)
                desired_center = float(single_line_projection_debug.get('single_line_target_x', midpoint_x))
                desired_center = max(0.0, min(float(img_w - 1), desired_center))
                midpoint_bottom_x = desired_center
                midpoint_x = desired_center
                heading_rad = float(single_heading)
                error_px = float(single_error)
                raw_error_px = float(single_error)
                draw_y = int(single_line_projection_debug.get('single_line_reference_y', reference_y))
                draw_y = max(0, min(img_h - 1, draw_y))
                guidance_mode = 'single_line_physical'
            else:
                # Line is nearly horizontal: skip physical projection.  Keep the
                # midpoint_ref error already in error_px (mask x + virtual boundary),
                # which matches the two-line midpoint formula.  Record a minimal debug
                # dict so downstream logging still sees single-line fields.
                single_line_projection_debug = {
                    'single_line_angle_from_vertical_deg': float(_abs_heading_deg),
                    'single_line_heading_raw_deg': float(math.degrees(_heading_raw_rad)),
                    'single_line_heading_reliability': 0.0,
                    'single_line_prefer_center': bool(single_line_prefer_center),
                    'single_line_streak': int(single_line_streak),
                    'single_line_curve_context': bool(curve_context),
                    'single_line_mode': 'midpoint_ref_fallback',
                }

        single_side_error_limit_px = None
        if len(detected_sides) == 1:
            limit_factor = 0.85 if guidance_mode == 'single_line_physical' else 0.60
            if curve_context:
                limit_factor = min(limit_factor, 0.55)
            if (
                guidance_mode == 'single_line_physical' and
                isinstance(single_line_projection_debug, dict) and
                bool(single_line_projection_debug.get('single_line_outer_confirmed', False))
            ):
                # Outer single-line curves are the least observable case: be more conservative.
                limit_factor = min(limit_factor, 0.45)
            single_side_error_limit_px = max(8.0, float(lane_width_px) * limit_factor)
            error_px = max(
                -single_side_error_limit_px,
                min(single_side_error_limit_px, float(error_px))
            )

        if len(detected_sides) == 2:
            mask_label = 'BOTH'
        elif detected_sides[0] == 'left':
            mask_label = 'LEFT'
        else:
            mask_label = 'RIGHT'

        # Heading-proportional outward offset — two-line (midpoint_ref) mode ONLY.
        # In midpoint_ref mode the heading is computed as the image slope of the road midline
        # over a short vertical baseline (≈42–79 px), which perspective-amplifies the road
        # curvature angle by 2–7×.  This makes heading dominate Stanley and causes the car to
        # turn too tightly.  The offset boosts the crosstrack term to counteract heading.
        # NOT applied in single_line_physical: the physical projection already produces a
        # meaningful crosstrack error.  Adding the offset there amplified the leftward push
        # (e.g. raw=-162 px → error=-213 px → steering pinned at -25° instead of -17.6°).
        ct_heading_gain = float(getattr(self, 'curve_crosstrack_heading_gain', 0.0))
        heading_abs = abs(float(heading_rad))
        if ct_heading_gain > 0.0 and heading_abs > 0.0 and guidance_mode == 'midpoint_ref':
            max_offset_px = float(lane_width_px) * 0.35
            outward_px = min(heading_abs * ct_heading_gain * float(lane_width_px), max_offset_px)
            curve_ct_offset = math.copysign(outward_px, -float(heading_rad))
            error_px = float(error_px) + curve_ct_offset

        # Metric crosstrack error using BEV scale.  bev_cm_per_px converts BEV
        # pixels to centimetres; lane_width_px is in perspective pixels, so we
        # scale error_px to cm via the same ratio: error_cm approximates the
        # real lateral offset without breaking the existing Stanley k tuning
        # (which operates on perspective pixels via error_px).
        _error_cm = None
        _bev_cm_per_px = float(getattr(self, 'bev_cm_per_px', 0.0) or 0.0)
        _lane_width_cm = float(getattr(self, 'lane_width_cm', 35.0) or 35.0)
        if _bev_cm_per_px > 0.0 and lane_width_px > 0.0 and _lane_width_cm > 0.0:
            _px_to_cm = _lane_width_cm / max(1.0, lane_width_px)
            _error_cm = float(error_px) * _px_to_cm

        guidance = {
            'midpoint_x': float(midpoint_x),
            'midpoint_bottom_x': float(midpoint_bottom_x),
            'error_px': float(error_px),
            'raw_error_px': float(raw_error_px),
            'error_cm': _error_cm,
            'single_side_error_limit_px': (
                float(single_side_error_limit_px) if single_side_error_limit_px is not None else None
            ),
            'heading_rad': float(heading_rad),
            'raw_two_line_heading_rad': float(_raw_two_line_heading_rad) if len(detected_sides) >= 2 else None,
            'bev_heading_active': bool(_bev_heading_active),
            'left_x': float(left_ref_x),
            'right_x': float(right_ref_x),
            'bottom_left_x': float(left_bottom_x),
            'bottom_right_x': float(right_bottom_x),
            'lane_width_px': float(lane_width_px),
            'detected_sides': tuple(detected_sides),
            'reference_y': int(reference_y),
            'bottom_y': int(measurement_bottom_y),
            'draw_y': int(draw_y),
            'used_virtual_boundary': bool(used_virtual_boundary),
            'guidance_mode': 'midpoint' if guidance_mode == 'midpoint_ref' else guidance_mode,
            'single_line_prefer_center': (
                bool(single_line_prefer_center) if single_line_prefer_center is not None else None
            ),
            'single_line_streak': (
                int(single_line_streak) if single_line_streak is not None else None
            ),
            'mask_label': mask_label,
            'lane_observation': lane_observation,
        }
        if single_line_projection_debug is not None:
            guidance['single_line_projection_debug'] = single_line_projection_debug
        if narrow_width_collapse is not None:
            guidance['duplicate_line_collapse'] = narrow_width_collapse
        return guidance

    def _resolve_ambiguous_local_lane(self, line, img_h, img_w, prev_seen_side=None):
        """Resolve a guessed single local-AI lane into left/right using temporal geometry."""
        if line is None:
            return None, 'none'

        y_ref = int(max(0, min(img_h - 1, round(img_h * (1.0 - float(getattr(self, 'lookahead', 0.4)))))))
        line_x = self._line_x_at_y(line, y_ref)
        reference_ttl = max(1.0, float(getattr(self, 'local_ai_max_result_age', 0.35)) * 3.0)
        last_two_line_ts = float(getattr(self, '_last_two_line_ts', 0.0) or 0.0)
        if last_two_line_ts and (time.time() - last_two_line_ts) <= reference_ttl:
            left_ref = getattr(self, '_last_two_line_left', None)
            right_ref = getattr(self, '_last_two_line_right', None)
            left_x = self._line_x_at_y(left_ref, y_ref)
            right_x = self._line_x_at_y(right_ref, y_ref)
            distances = []
            if left_x is not None and line_x is not None:
                distances.append(('left', abs(line_x - left_x)))
            if right_x is not None and line_x is not None:
                distances.append(('right', abs(line_x - right_x)))
            if distances:
                distances.sort(key=lambda item: item[1])
                return distances[0][0], 'history'

        if prev_seen_side in ('left', 'right'):
            return prev_seen_side, 'prev_seen_side'

        reference_error = None
        if (
            bool(getattr(self, 'use_kalman_filter', False)) and
            (bool(getattr(self, '_kalman_initialized', False)) or int(getattr(self, '_kalman_age_frames', 0)) > 0)
        ):
            reference_error = float(getattr(self, '_kalman_prediction', 0.0))
        elif getattr(self, '_last_good_error', None) is not None:
            reference_error = float(self._last_good_error)

        if reference_error is not None:
            best_side = None
            best_delta = None
            for side in ('left', 'right'):
                error, _ = self._compute_single_line_error(
                    line, side, img_h, img_w,
                    prefer_center=True,
                    physical_projection=True,
                )
                delta = abs(float(error) - reference_error)
                if best_delta is None or delta < best_delta:
                    best_side = side
                    best_delta = delta
            if best_side is not None:
                return best_side, 'error_match'

        if line_x is not None:
            return ('left' if line_x < (img_w / 2.0) else 'right'), 'frame_center'

        return None, 'none'

    def _fit_remote_lane_line(self, lane_points, img_h, img_w):
        """Convert remote polyline points into a representative line segment."""
        working_points = self._normalize_remote_lane_points(lane_points, img_h)
        if len(working_points) < 2:
            return None

        roi_top = max(0, min(img_h - 1, int(self.roi_height_start * img_h)))
        roi_bottom = max(roi_top, min(img_h - 1, int(self.roi_height_end * img_h) - 1))

        xs = np.array([p[0] for p in working_points], dtype=np.float32)
        ys = np.array([p[1] for p in working_points], dtype=np.float32)

        if np.ptp(ys) < 1.0:
            order = sorted(working_points, key=lambda p: p[0])
            p1 = order[0]
            p2 = order[-1]
            return np.array([[
                int(round(max(0, min(img_w - 1, p1[0])))),
                int(round(max(roi_top, min(roi_bottom, p1[1])))),
                int(round(max(0, min(img_w - 1, p2[0])))),
                int(round(max(roi_top, min(roi_bottom, p2[1])))),
            ]], dtype=np.int32)

        fit = np.polyfit(ys, xs, 1)
        y_bottom = float(min(np.max(ys), roi_bottom))
        y_top = float(max(np.min(ys), roi_top))
        if y_bottom - y_top < 1.0:
            if y_bottom >= roi_bottom:
                y_top = max(float(roi_top), y_bottom - 1.0)
            else:
                y_bottom = min(float(roi_bottom), y_top + 1.0)
        x_bottom = float(np.polyval(fit, y_bottom))
        x_top = float(np.polyval(fit, y_top))

        x_bottom = max(0.0, min(img_w - 1.0, x_bottom))
        x_top = max(0.0, min(img_w - 1.0, x_top))
        y_bottom = max(0.0, min(img_h - 1.0, y_bottom))
        y_top = max(0.0, min(img_h - 1.0, y_top))

        return np.array([[
            int(round(x_bottom)),
            int(round(y_bottom)),
            int(round(x_top)),
            int(round(y_top)),
        ]], dtype=np.int32)

    def _normalize_remote_lane_points(self, lane_points, img_h):
        """Use the same lower-image ROI as BFMC before fitting AI lane geometry."""
        if not lane_points or len(lane_points) < 2:
            return []

        parsed_points = []
        for point in lane_points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError):
                continue
            parsed_points.append((x, y))

        if len(parsed_points) < 2:
            return []

        roi_top = float(max(0, min(img_h - 1, int(self.roi_height_start * img_h))))
        roi_bottom = float(max(roi_top, min(img_h - 1, int(self.roi_height_end * img_h) - 1)))
        roi_points = [p for p in parsed_points if roi_top <= p[1] <= roi_bottom]

        if len(roi_points) >= 2:
            return roi_points
        return parsed_points

    def _remote_lane_points_to_lines(self, lane_points, img_h, img_w):
        """Map remote lane polylines to left/right representative lines."""
        if not lane_points:
            return None, None

        center_x = img_w / 2.0
        line_candidates = []

        for line_points in lane_points:
            working_points = self._normalize_remote_lane_points(line_points, img_h)
            line = self._fit_remote_lane_line(working_points, img_h, img_w)
            if line is None:
                continue
            mean_x = float(np.mean([p[0] for p in working_points]))
            line_candidates.append((mean_x, line))

        if not line_candidates:
            return None, None

        left_candidates = [item for item in line_candidates if item[0] < center_x]
        right_candidates = [item for item in line_candidates if item[0] >= center_x]

        avg_left = max(left_candidates, key=lambda item: item[0])[1] if left_candidates else None
        avg_right = min(right_candidates, key=lambda item: item[0])[1] if right_candidates else None

        if avg_left is None and avg_right is None and len(line_candidates) == 1:
            mean_x, line = line_candidates[0]
            if mean_x < center_x:
                avg_left = line
            else:
                avg_right = line

        return avg_left, avg_right

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
                    print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;92mINFO\033[0m - HybridNets client started, connecting to {self._hybridnets_client.server_url}")
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
        Detect lanes using the remote AI server, but keep local BFMC control.

        The server only provides lane geometry (lane_points). The local thread
        keeps using the same Stanley/PID, single-lane tracking, swept-path, and
        recovery logic as the OpenCV pipeline.

        Returns:
            tuple: (avg_left, avg_right, img_h, img_w, canny_like, debug_info)
        """
        height, width = frame.shape[:2]

        debug_info = {
            'pipeline_label': 'AI Lanes + BFMC Control',
        }
        empty_mask = np.zeros((height, width), dtype=np.uint8)
        y_start = int(self.roi_height_start * height)
        y_end = int(self.roi_height_end * height)
        roi_vertices = np.array([[(0, y_start), (width, y_start), (width, y_end), (0, y_end)]], dtype=np.int32)
        roi_mask = np.zeros_like(empty_mask)
        cv2.fillPoly(roi_mask, roi_vertices, 255)
        debug_info['roi_vertices'] = roi_vertices

        if self._hybridnets_client is None or not self._hybridnets_client.connected:
            self._start_hybridnets_client()
            if not self._hybridnets_client or not self._hybridnets_client.connected:
                self._smooth_detected_line(None, 'left')
                self._smooth_detected_line(None, 'right')
                debug_info['pipeline_label'] = 'AI Lanes - Connecting'
                debug_info['remote_status'] = 'connecting'
                return None, None, height, width, empty_mask, debug_info

        # Non-blocking send keeps the control loop running at camera rate instead
        # of waiting on network+GPU latency every cycle.
        result = self._hybridnets_client.send_frame(frame, block=False)
        last_result_time = getattr(self._hybridnets_client, '_last_result_time', 0.0)
        result_age = (time.time() - last_result_time) if last_result_time else None

        if result is None:
            self._smooth_detected_line(None, 'left')
            self._smooth_detected_line(None, 'right')
            debug_info['pipeline_label'] = 'AI Lanes - Waiting first result'
            debug_info['remote_status'] = 'warming_up'
            return None, None, height, width, empty_mask, debug_info

        if result_age is None or result_age > self.hybridnets_max_result_age:
            self._smooth_detected_line(None, 'left')
            self._smooth_detected_line(None, 'right')
            debug_info['pipeline_label'] = 'AI Lanes - Stale result'
            debug_info['remote_status'] = 'stale'
            debug_info['remote_result_age_ms'] = round((result_age or 0.0) * 1000, 1)
            return None, None, height, width, empty_mask, debug_info

        lane_points = result.get('lane_points', [])
        server_time = float(result.get('inference_time_ms', 0) or 0)
        roundtrip = float(result.get('roundtrip_ms', 0) or 0)
        frame_id = int(result.get('frame_id', 0) or 0)

        debug_info['remote_server_time_ms'] = server_time
        debug_info['remote_roundtrip_ms'] = roundtrip
        debug_info['remote_frame_id'] = frame_id
        debug_info['remote_lane_count'] = len(lane_points)
        debug_info['remote_result_age_ms'] = round(result_age * 1000, 1)

        lane_mask = empty_mask
        lane_mask_b64 = result.get('lane_mask_b64')
        if lane_mask_b64:
            try:
                lane_bytes = base64.b64decode(lane_mask_b64)
                decoded = cv2.imdecode(np.frombuffer(lane_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                if decoded is not None:
                    lane_mask = decoded
            except Exception:
                pass
        lane_mask = cv2.bitwise_and(lane_mask, roi_mask)

        avg_left, avg_right = self._remote_lane_points_to_lines(lane_points, height, width)
        avg_left = self._smooth_detected_line(avg_left, 'left')
        avg_right = self._smooth_detected_line(avg_right, 'right')
        debug_info['threshold'] = 'AI'
        debug_info['left_lines'] = [avg_left] if avg_left is not None else []
        debug_info['right_lines'] = [avg_right] if avg_right is not None else []

        if not hasattr(self, '_hybridnets_frame_count'):
            self._hybridnets_frame_count = 0
        self._hybridnets_frame_count += 1

        if self._hybridnets_frame_count <= 5 or self._hybridnets_frame_count % 20 == 0:
            print(f"\033[1;97m[ HybridNets ] :\033[0m "
                  f"lanes={len(lane_points)} "
                  f"L={'Y' if avg_left is not None else 'N'} "
                  f"R={'Y' if avg_right is not None else 'N'} | "
                  f"server={server_time:.0f}ms RT={roundtrip:.0f}ms | "
                  f"frame#{self._hybridnets_frame_count}")

        if avg_left is None and avg_right is None:
            debug_info['pipeline_label'] = 'AI Lanes - No lanes'

        return avg_left, avg_right, height, width, lane_mask, debug_info

    def _detect_with_local_ai(self, frame):
        """Consume locally inferred lane geometry and keep BFMC/Stanley as the controller."""
        height, width = frame.shape[:2]

        # Initialize perspective/BEV transform if not already done.
        # This is required for _reclassify_masks_via_bev() to work; without it
        # perspective_initialized stays False and BEV is always skipped.
        if not getattr(self, 'perspective_initialized', False):
            self._init_perspective_transform(width, height)

        debug_info = {
            'pipeline_label': 'AI Local + BFMC Control',
        }
        empty_mask = np.zeros((height, width), dtype=np.uint8)
        y_start = int(self.roi_height_start * height)
        y_end = int(self.roi_height_end * height)
        roi_vertices = np.array([[(0, y_start), (width, y_start), (width, y_end), (0, y_end)]], dtype=np.int32)
        debug_info['roi_vertices'] = roi_vertices

        self._reset_local_ai_context()
        payload = self._last_local_lane_payload
        if not payload:
            self._smooth_detected_line(None, 'left')
            self._smooth_detected_line(None, 'right')
            debug_info['pipeline_label'] = 'AI Local - Waiting first result'
            debug_info['local_ai_status'] = 'warming_up'
            return None, None, height, width, empty_mask, debug_info

        if not bool(payload.get('model_ready', False)):
            self._smooth_detected_line(None, 'left')
            self._smooth_detected_line(None, 'right')
            debug_info['pipeline_label'] = 'AI Local - Model not ready'
            debug_info['local_ai_status'] = 'not_ready'
            return None, None, height, width, empty_mask, debug_info

        now_ts = time.time()
        timestamp = 0.0
        timestamp_source = 'none'
        for candidate_key in ('result_timestamp', 'timestamp', 'source_frame_timestamp'):
            try:
                candidate_ts = float(payload.get(candidate_key, 0.0) or 0.0)
            except (TypeError, ValueError):
                candidate_ts = 0.0
            if candidate_ts > 0.0:
                timestamp = candidate_ts
                timestamp_source = candidate_key
                break
        result_age = max(0.0, (now_ts - timestamp)) if timestamp > 0.0 else None

        try:
            source_frame_timestamp = float(payload.get('source_frame_timestamp', 0.0) or 0.0)
        except (TypeError, ValueError):
            source_frame_timestamp = 0.0
        source_frame_age_ms = (
            round(max(0.0, now_ts - source_frame_timestamp) * 1000.0, 1)
            if source_frame_timestamp > 0.0 else None
        )

        max_result_age = float(self.local_ai_max_result_age)
        stale_grace_limit = max_result_age + max(0.0, float(getattr(self, 'local_ai_stale_grace_s', 0.20)))
        stale_grace_active = (
            result_age is not None and
            max_result_age < result_age <= stale_grace_limit
        )
        if result_age is None or result_age > stale_grace_limit:
            self._smooth_detected_line(None, 'left')
            self._smooth_detected_line(None, 'right')
            debug_info['pipeline_label'] = 'AI Local - Stale result'
            debug_info['local_ai_status'] = 'stale'
            debug_info['remote_result_age_ms'] = round((result_age or 0.0) * 1000, 1)
            debug_info['timestamp_source'] = timestamp_source
            if source_frame_age_ms is not None:
                debug_info['source_frame_age_ms'] = source_frame_age_ms
            return None, None, height, width, empty_mask, debug_info

        infer_ms = float(payload.get('inference_time_ms', 0.0) or 0.0)
        frame_id = int(payload.get('frame_id', 0) or 0)
        local_status_payload = getattr(self, '_last_local_perception_status', None)
        local_status = local_status_payload if isinstance(local_status_payload, dict) else {}
        throughput_fps = local_status.get('fps')
        processing_fps = local_status.get('processing_fps')
        target_fps = local_status.get('target_fps')
        if processing_fps is None and infer_ms > 0.0:
            processing_fps = 1000.0 / infer_ms
        debug_info['remote_server_time_ms'] = infer_ms
        debug_info['remote_roundtrip_ms'] = round((result_age or 0.0) * 1000, 1)
        debug_info['remote_frame_id'] = frame_id
        debug_info['remote_result_age_ms'] = round((result_age or 0.0) * 1000, 1)
        debug_info['timestamp_source'] = timestamp_source
        if source_frame_age_ms is not None:
            debug_info['source_frame_age_ms'] = source_frame_age_ms
        debug_info['local_ai_status'] = 'stale_grace' if stale_grace_active else 'ready'
        self._update_local_ai_context(
            payload,
            debug_info=debug_info,
            result_age=result_age,
            stale_grace=stale_grace_active,
        )
        if stale_grace_active:
            debug_info['pipeline_label'] = 'AI Local + BFMC Control (Stale Grace)'
        if throughput_fps is not None:
            debug_info['local_ai_throughput_fps'] = float(throughput_fps)
        if processing_fps is not None:
            debug_info['local_ai_processing_fps'] = float(processing_fps)
        if target_fps is not None:
            debug_info['local_ai_target_fps'] = float(target_fps)

        side_masks = payload.get('side_masks')
        lane_side_lines = payload.get('lane_side_lines')
        lane_side_sources = self._normalize_lane_side_sources(payload.get('lane_side_sources'))
        lane_mask_payload = payload.get('lane_mask')

        # Capture raw YOLO mask info for mask debug log (before any reclassification)
        if bool(getattr(self, '_mask_debug_log_enabled', False)):
            raw_info = {}
            if isinstance(side_masks, dict):
                for _s in ('left', 'right'):
                    _m = side_masks.get(_s)
                    if isinstance(_m, np.ndarray) and _m.size > 0 and np.any(_m):
                        _cols = np.where(_m > 0)[1]
                        raw_info[_s] = {
                            'px_cx': round(float(np.mean(_cols)), 1) if len(_cols) > 0 else None,
                            'px_area': int(np.sum(_m > 0)),
                            'src': str(lane_side_sources.get(_s, 'none') or 'none'),
                        }
                    else:
                        raw_info[_s] = None
            self._last_yolo_raw_debug = raw_info
            self._last_bev_reclassify_debug = {}  # reset; will be filled by _reclassify_masks_via_bev

        def _collect_render_masks(s_masks, h, w):
            """Build {left, right} dict of 2-D uint8 masks for rendering."""
            result = {'left': None, 'right': None}
            if not isinstance(s_masks, dict):
                return result
            for side in ('left', 'right'):
                m = s_masks.get(side)
                if not isinstance(m, np.ndarray) or m.size == 0:
                    continue
                m2d = m if m.ndim == 2 else m[..., 0]
                if m2d.ndim == 2 and m2d.shape[:2] == (h, w) and np.any(m2d):
                    result[side] = m2d
            return result

        prepared_side_masks, prepared_side_lines, mask_side_resolution = self._prepare_local_ai_side_masks(
            side_masks,
            lane_side_sources,
            lane_side_lines,
            height,
            width,
        )
        duplicate_collapse = getattr(self, '_last_local_ai_duplicate_collapse', None)
        if isinstance(duplicate_collapse, dict):
            debug_info['duplicate_line_collapse'] = duplicate_collapse

        # BEV reclassification: verify left/right assignment using the
        # calibrated homography.  The perspective-space x_center heuristic
        # used by localPerceptionEngine fails when the vehicle is heading at
        # an angle relative to the lane (e.g. entering a curve).  In BEV the
        # x-centroid is metrically correct: the left mask always has a smaller
        # BEV x than the right mask regardless of vehicle heading.
        # Only runs when both masks are present and BEV is calibrated; silently
        # falls back to the original assignment in all other cases.
        prepared_side_masks, prepared_side_lines, _bev_swapped = self._reclassify_masks_via_bev(
            prepared_side_masks, prepared_side_lines
        )
        if _bev_swapped:
            debug_info['bev_reclassified'] = True

        if bool(getattr(self, 'ai_local_use_mask_geometry', True)):
            local_mask_guidance = self._build_local_mask_guidance(
                prepared_side_masks,
                height,
                width,
                side_lines=prepared_side_lines,
            )

            render_side_masks = _collect_render_masks(prepared_side_masks, height, width)
            has_any_mask = any(v is not None for v in render_side_masks.values())

            if local_mask_guidance is not None:
                render_lane_mask = empty_mask
                if isinstance(lane_mask_payload, np.ndarray) and lane_mask_payload.size > 0:
                    lane_mask_2d = lane_mask_payload if lane_mask_payload.ndim == 2 else lane_mask_payload[..., 0]
                    if lane_mask_2d.ndim == 2 and lane_mask_2d.shape[:2] == (height, width):
                        render_lane_mask = lane_mask_2d
                elif render_side_masks['left'] is not None or render_side_masks['right'] is not None:
                    render_lane_mask = np.zeros((height, width), dtype=np.uint8)
                    for side in ('left', 'right'):
                        if render_side_masks[side] is not None:
                            render_lane_mask = cv2.bitwise_or(render_lane_mask, render_side_masks[side])

                self._smoothed_left_line = None
                self._smoothed_right_line = None
                self._left_line_missing_frames = 0
                self._right_line_missing_frames = 0

                debug_info['pipeline_label'] = 'AI Local + Mask Guidance'
                debug_info['threshold'] = 'AI MASK'
                debug_info['remote_lane_count'] = len(local_mask_guidance.get('detected_sides', ()))
                debug_info['control_mode'] = 'mask_guidance'
                debug_info['local_mask_guidance'] = local_mask_guidance
                debug_info['render_side_masks'] = render_side_masks
                debug_info['render_lane_mask'] = render_lane_mask
                debug_info['mask_label'] = local_mask_guidance.get('mask_label', 'NONE')
                debug_info['left_lines'] = []
                debug_info['right_lines'] = []
                debug_info.update(self._lane_observation_debug_fields(
                    local_mask_guidance.get('lane_observation')
                ))
                single_line_projection_debug = local_mask_guidance.get('single_line_projection_debug')
                if isinstance(single_line_projection_debug, dict):
                    debug_info.update(single_line_projection_debug)
                guidance_duplicate_collapse = local_mask_guidance.get('duplicate_line_collapse')
                if isinstance(guidance_duplicate_collapse, dict):
                    debug_info['duplicate_line_collapse'] = guidance_duplicate_collapse
                    debug_info['single_line_side_source'] = (
                        f"duplicate_overlap:{guidance_duplicate_collapse.get('resolution_source', 'unknown')}"
                    )
                    debug_info['single_line_resolved_side'] = guidance_duplicate_collapse.get('kept_side')
                if mask_side_resolution is not None:
                    debug_info['single_line_side_source'] = (
                        f"{mask_side_resolution['source']}:{mask_side_resolution['resolution_source']}"
                    )
                    debug_info['single_line_resolved_side'] = mask_side_resolution['resolved_side']
                    debug_info['single_line_detected_side'] = mask_side_resolution['detected_side']
                elif isinstance(duplicate_collapse, dict):
                    debug_info['single_line_side_source'] = (
                        f"duplicate_overlap:{duplicate_collapse.get('resolution_source', 'unknown')}"
                    )
                    debug_info['single_line_resolved_side'] = duplicate_collapse.get('kept_side')

                return None, None, height, width, render_lane_mask, debug_info

            # If we do have raw masks but can't extract usable geometry, treat it as
            # a bad mask frame and hold the previous steering. When there are no masks
            # at all, fall back to explicit line metadata from the payload.
            if has_any_mask:
                debug_info['pipeline_label'] = 'AI Local - No Mask Guidance'
                debug_info['local_ai_status'] = 'mask_failed'
                reject_reason = getattr(self, '_last_local_ai_mask_reject_reason', None)
                if reject_reason:
                    debug_info['mask_reject_reason'] = reject_reason
                debug_info['left_lines'] = []
                debug_info['right_lines'] = []
                if mask_side_resolution is not None:
                    debug_info['single_line_side_source'] = (
                        f"{mask_side_resolution['source']}:{mask_side_resolution['resolution_source']}"
                    )
                    debug_info['single_line_resolved_side'] = mask_side_resolution['resolved_side']
                    debug_info['single_line_detected_side'] = mask_side_resolution['detected_side']
                elif isinstance(duplicate_collapse, dict):
                    debug_info['single_line_side_source'] = (
                        f"duplicate_overlap:{duplicate_collapse.get('resolution_source', 'unknown')}"
                    )
                    debug_info['single_line_resolved_side'] = duplicate_collapse.get('kept_side')
                debug_info['render_side_masks'] = render_side_masks
                has_line_fallback = any(
                    isinstance(prepared_side_lines.get(side), np.ndarray)
                    and prepared_side_lines.get(side).ndim == 2
                    and prepared_side_lines.get(side).shape[1] >= 4
                    for side in ('left', 'right')
                )
                lane_side_points_payload = payload.get('lane_side_points')
                has_side_points_fallback = (
                    isinstance(lane_side_points_payload, dict)
                    and any(lane_side_points_payload.get(side) for side in ('left', 'right'))
                )
                lane_points_payload = payload.get('lane_points', [])
                has_lane_points_fallback = isinstance(lane_points_payload, (list, tuple)) and len(lane_points_payload) > 0

                # If mask geometry fails but lane geometry exists, continue with legacy fit
                # instead of forcing a "no lanes" frame.
                if not (has_line_fallback or has_side_points_fallback or has_lane_points_fallback):
                    return None, None, height, width, empty_mask, debug_info
                debug_info['pipeline_label'] = 'AI Local - Mask Fallback to Lines'
                debug_info['control_mode'] = 'legacy_line_fit'

        lane_points = payload.get('lane_points', [])
        lane_side_points = payload.get('lane_side_points')
        explicit_left, explicit_right = self._lane_side_lines_to_lines(lane_side_lines, height, width)
        has_explicit_side_lines = explicit_left is not None or explicit_right is not None
        has_explicit_side_points = (
            isinstance(lane_side_points, dict) and
            any(lane_side_points.get(side) for side in ('left', 'right'))
        )
        if has_explicit_side_lines:
            debug_info['remote_lane_count'] = int(explicit_left is not None) + int(explicit_right is not None)
        elif has_explicit_side_points:
            debug_info['remote_lane_count'] = sum(
                1 for side in ('left', 'right') if lane_side_points.get(side)
            )
        else:
            debug_info['remote_lane_count'] = len(lane_points)
        debug_info['control_mode'] = 'legacy_line_fit'

        fitted_left = None
        fitted_right = None
        if has_explicit_side_points:
            fitted_left, fitted_right = self._lane_side_points_to_lines(lane_side_points, height, width)

        side_lines = {
            'left': self._prefer_local_lane_line(explicit_left, fitted_left),
            'right': self._prefer_local_lane_line(explicit_right, fitted_right),
        }
        side_sources = {
            'left': lane_side_sources.get('left', 'none'),
            'right': lane_side_sources.get('right', 'none'),
        }
        for side in ('left', 'right'):
            if side_lines[side] is not None and side_sources[side] == 'none':
                side_sources[side] = 'explicit_class'

        avg_left_raw = None
        avg_right_raw = None
        available_sides = [side for side in ('left', 'right') if side_lines[side] is not None]
        trusted_sources = {'explicit_class', 'split_component'}

        if len(available_sides) == 2:
            avg_left_raw = side_lines['left']
            avg_right_raw = side_lines['right']
        elif len(available_sides) == 1:
            only_side = available_sides[0]
            only_line = side_lines[only_side]
            only_source = side_sources[only_side]
            if only_source in trusted_sources:
                if only_side == 'left':
                    avg_left_raw = only_line
                else:
                    avg_right_raw = only_line
                debug_info['single_line_side_source'] = only_source
            elif only_source == 'guessed_single':
                resolved_side, resolution_source = self._resolve_ambiguous_local_lane(
                    only_line,
                    height,
                    width,
                    prev_seen_side=getattr(self, 'last_seen_side', None),
                )
                if resolved_side == 'left':
                    avg_left_raw = only_line
                elif resolved_side == 'right':
                    avg_right_raw = only_line
                debug_info['single_line_side_source'] = f"guessed_single:{resolution_source}"

        if avg_left_raw is None and avg_right_raw is None:
            avg_left_raw, avg_right_raw = self._lane_points_to_lines(lane_points, height, width)

        avg_left_raw, avg_right_raw, overlap_collapse = self._collapse_overlapping_two_lines(
            avg_left_raw,
            avg_right_raw,
            height,
            width,
            prev_seen_side=getattr(self, 'last_seen_side', None),
        )
        if overlap_collapse is not None:
            debug_info['duplicate_line_collapse'] = overlap_collapse
            debug_info['single_line_side_source'] = (
                f"duplicate_overlap:{overlap_collapse.get('resolution_source', 'unknown')}"
            )
            debug_info['single_line_resolved_side'] = overlap_collapse.get('kept_side')

        if avg_left_raw is not None and avg_right_raw is not None:
            self._last_two_line_left = avg_left_raw.copy()
            self._last_two_line_right = avg_right_raw.copy()
            self._last_two_line_ts = time.time()

        avg_left = self._smooth_detected_line(avg_left_raw, 'left')
        avg_right = self._smooth_detected_line(avg_right_raw, 'right')
        debug_info['threshold'] = 'AI'
        debug_info['left_lines'] = [avg_left] if avg_left is not None else []
        debug_info['right_lines'] = [avg_right] if avg_right is not None else []

        if avg_left is None and avg_right is None:
            debug_info['pipeline_label'] = 'AI Local - No lanes'

        return avg_left, avg_right, height, width, empty_mask, debug_info

    def _get_effective_speeds(self):
        """Return (max_speed, min_speed) considering highway mode."""
        if self.highway_mode_event and self.highway_mode_event.is_set():
            return self.highway_max_speed, self.highway_min_speed
        return self.max_speed, self.min_speed

    def _update_progressive_speed(self, steering_angle, speed_cap=None):
        """Update `_current_speed` from steering, respecting an optional safety cap."""
        eff_max, eff_min = self._get_effective_speeds()
        abs_steer = abs(float(steering_angle))
        steer_ratio = min(abs_steer / max(float(self.max_steering), 1e-3), 1.0)

        is_highway = bool(self.highway_mode_event and self.highway_mode_event.is_set())
        if is_highway:
            # En autopista aplicar un factor configurable: 0.0 = sin reducción de velocidad
            # al corregir, 1.0 = misma reducción que modo normal.
            hw_factor = float(getattr(_config, "LF_HIGHWAY_STEER_SPEED_FACTOR", 0.2))
            target_speed = eff_max - steer_ratio * (eff_max - eff_min) * hw_factor
            ramp_step = float(getattr(_config, "LF_HIGHWAY_SPEED_RAMP_STEP", 3.0))
        else:
            target_speed = eff_max - steer_ratio * (eff_max - eff_min)
            ramp_step = self.speed_ramp_step

        if speed_cap is not None:
            target_speed = min(target_speed, float(speed_cap))

        if target_speed <= self._current_speed:
            # Ramp down gradually so the car doesn't appear to freeze at curve nodes.
            # Exception: if the gap is large (>3 units, e.g. emergency stop) drop instantly.
            gap = self._current_speed - target_speed
            down_step = float(getattr(self, 'speed_ramp_down_step', 1.5))
            if gap > 3.0:
                self._current_speed = target_speed  # instant for large drops (safety)
            else:
                self._current_speed = max(target_speed, self._current_speed - down_step)
        else:
            self._current_speed = min(target_speed, self._current_speed + ramp_step)
        return self._current_speed

    def _get_no_lane_hold_steering(self):
        """Hold the last good steering briefly when both lane lines disappear."""
        hold_frames = max(0, int(getattr(self, 'no_lane_hold_steering_frames', 4) or 0))
        frames_without_line = int(getattr(self, 'frames_without_line', 0) or 0)
        if hold_frames == 0 or frames_without_line <= 0 or frames_without_line > hold_frames:
            return None

        candidate = getattr(self, '_last_good_steering', None)
        if candidate is None:
            candidate = getattr(self, 'last_steering', None)
        if candidate is None:
            return None

        return max(-self.max_steering, min(self.max_steering, float(candidate)))

    def _should_use_route_blind_guidance(self, nav_ctx=None):
        """Return whether blind/no-lane control should follow route tracking."""
        ts = getattr(self, "_tracking_state", None)
        if ts is None or not getattr(ts, "initialized", False):
            return False, "tracking_uninitialized"
        if not bool(getattr(ts, "route_active", False)):
            return False, "route_inactive"

        if not isinstance(nav_ctx, dict):
            nav_ctx = self._get_navigation_context()

        if bool(nav_ctx.get("waypoint_mode_active", False)):
            return True, "waypoint_mode"
        if bool(nav_ctx.get("planner_priority", False)):
            return True, "planner_priority"

        map_match_error_m = abs(float(getattr(ts, "map_match_error_m", 0.0) or 0.0))
        max_map_match_error_m = max(
            0.05,
            float(getattr(_config, "TRACKING_BLIND_ROUTE_MAX_MAP_MATCH_ERROR_M", 0.25) or 0.25),
        )
        if map_match_error_m > max_map_match_error_m:
            return False, "map_match_unreliable"

        path_heading_change_rad = abs(float(getattr(ts, "path_heading_change_rad", 0.0) or 0.0))
        min_heading_change_rad = math.radians(max(
            1.0,
            float(getattr(_config, "TRACKING_BLIND_ROUTE_MIN_HEADING_CHANGE_DEG", 40.0) or 40.0),
        ))
        if path_heading_change_rad < min_heading_change_rad:
            return False, "weak_path_heading"

        frames_without_line = int(getattr(self, "frames_without_line", 0) or 0)
        activation_frames = max(
            1,
            int(getattr(_config, "TRACKING_BLIND_ROUTE_FALLBACK_FRAMES", 2) or 2),
        )
        curve_state = str(getattr(self, "_curve_state", "STRAIGHT"))
        if curve_state in ("ENTERING", "IN_CURVE", "EXITING") and frames_without_line >= activation_frames:
            return True, "curve_state_path_heading"

        hold_frames = max(0, int(getattr(self, "no_lane_hold_steering_frames", 4) or 0))
        if frames_without_line > hold_frames:
            return True, "extended_blind_path_heading"

        return False, "visual_hold"

    def _get_navigation_context(self):
        """Summarize route and maneuver information published by threadTracking."""
        ts = getattr(self, "_tracking_state", None)
        maneuver_type = str(getattr(ts, "maneuver_type", "none") or "none") if ts is not None else "none"
        route_active = bool(getattr(ts, "route_active", False)) if ts is not None else False
        waypoint_mode_active = bool(getattr(ts, "waypoint_mode_active", False)) if ts is not None else False
        current_attr = int(getattr(ts, "current_node_attr", getattr(ts, "node_attr", 0)) or 0) if ts is not None else 0
        upcoming_attr = int(getattr(ts, "upcoming_node_attr", current_attr) or current_attr) if ts is not None else 0
        next_semantic_type = str(getattr(ts, "next_semantic_type", "") or "") if ts is not None else ""
        expected_control_type = str(getattr(ts, "expected_control_type", "") or "") if ts is not None else ""
        next_semantic_distance_m = (
            float(getattr(ts, "next_semantic_distance_m", 0.0))
            if ts is not None and getattr(ts, "next_semantic_distance_m", None) is not None
            else None
        )
        stopline_attr = int(getattr(_config, "TRACKING_STOPLINE_NODE_ATTR", 7) or 7)
        intersection_attr = 2

        planner_priority = route_active and (
            waypoint_mode_active or
            maneuver_type in {
                "stopline",
                "turn_left",
                "turn_right",
                "intersection_straight",
                "roundabout",
                "traffic_light",
            } or
            expected_control_type in {"stop", "traffic_light"} or
            next_semantic_type in {"stopline", "intersection", "roundabout"} or
            current_attr in {intersection_attr, stopline_attr} or
            upcoming_attr in {intersection_attr, stopline_attr}
        )

        return {
            "route_active": route_active,
            "waypoint_mode_active": waypoint_mode_active,
            "planner_priority": planner_priority,
            "maneuver_type": maneuver_type,
            "current_attr": current_attr,
            "upcoming_attr": upcoming_attr,
            "next_semantic_type": next_semantic_type,
            "expected_control_type": expected_control_type,
            "next_semantic_distance_m": next_semantic_distance_m,
        }

    def _should_allow_stopline_waypoint_guidance(self, direct_error_context=None):
        """Allow planner authority in critical route contexts while preserving visual assist."""
        nav_ctx = self._get_navigation_context()
        if not nav_ctx["planner_priority"]:
            return False

        stopline_attr = int(getattr(_config, "TRACKING_STOPLINE_NODE_ATTR", 7) or 7)
        is_stopline = nav_ctx["current_attr"] == stopline_attr or nav_ctx["upcoming_attr"] == stopline_attr
        if not is_stopline:
            return True

        if str(getattr(self, "_curve_state", "STRAIGHT")) != "STRAIGHT":
            return False

        if not isinstance(direct_error_context, dict):
            return False

        return str(direct_error_context.get("source", "") or "") == "single_line"

    def _planner_two_line_takeover_weight(self, nav_ctx):
        """Progressively bias steering toward route tracking on intersection approach."""
        if not isinstance(nav_ctx, dict) or not bool(nav_ctx.get("planner_priority", False)):
            return 0.0

        maneuver_type = str(nav_ctx.get("maneuver_type", "") or "")
        next_semantic_type = str(nav_ctx.get("next_semantic_type", "") or "")
        expected_control_type = str(nav_ctx.get("expected_control_type", "") or "")
        stopline_attr = int(getattr(_config, "TRACKING_STOPLINE_NODE_ATTR", 7) or 7)
        intersection_attr = 2
        current_attr = int(nav_ctx.get("current_attr", 0) or 0)
        upcoming_attr = int(nav_ctx.get("upcoming_attr", current_attr) or current_attr)
        precision_attr_ahead = (
            current_attr in {intersection_attr, stopline_attr} or
            upcoming_attr in {intersection_attr, stopline_attr}
        )
        relevant_maneuver = (
            maneuver_type in {"stopline", "turn_left", "turn_right", "intersection_straight", "roundabout", "traffic_light"} or
            expected_control_type in {"stop", "traffic_light"} or
            next_semantic_type in {"stopline", "intersection", "roundabout"}
        )
        if not (precision_attr_ahead or relevant_maneuver):
            return 0.0

        distance_m = nav_ctx.get("next_semantic_distance_m", None)
        if distance_m is None:
            return 0.75 if precision_attr_ahead else 0.0

        try:
            distance_m = max(0.0, float(distance_m))
        except (TypeError, ValueError):
            return 0.75 if precision_attr_ahead else 0.0

        blend_start_m = max(
            0.25,
            float(getattr(_config, "TRACKING_INTERSECTION_ROUTE_BLEND_START_M", 0.70) or 0.70),
        )
        blend_full_m = max(
            0.05,
            min(
                blend_start_m - 0.05,
                float(getattr(_config, "TRACKING_INTERSECTION_ROUTE_BLEND_FULL_M", 0.18) or 0.18),
            ),
        )
        if distance_m >= blend_start_m:
            return 0.0

        ratio = 1.0
        if distance_m > blend_full_m:
            ratio = (blend_start_m - distance_m) / max(blend_start_m - blend_full_m, 1e-6)

        base_blend = 0.35
        max_blend = 0.85
        return max(0.0, min(1.0, base_blend + (max_blend - base_blend) * float(ratio)))

    def _enforce_single_line_curve_priority(
        self,
        steering_angle,
        heading_rad,
        direct_error_m=None,
        side=None,
        debug_info=None,
    ):
        """Keep steering committed to the visible line while still inside the curve."""
        if steering_angle is None or side not in ("left", "right"):
            return steering_angle

        curve_state = str(getattr(self, "_curve_state", "STRAIGHT"))
        if curve_state not in ("ENTERING", "IN_CURVE"):
            return steering_angle

        # If graph tracking says the path ahead is straight or curving the other way,
        # bypass max-steer enforcement. This prevents
        # two failure modes:
        #   1. Intersection approach: right-lane-only visible → inferred left curve (-1)
        #      but GPS says straight/right → bypass stops wrong -25° steer.
        #   2. Post-curve straight: curve state oscillates back into IN_CURVE after the
        #      real curve ends, but graph lookahead says straight/opposite → bypass stops wrong
        #      +25° steer on the straight.
        # Legitimate curves (graph and visual agree on direction) still get enforced.
        _ts = getattr(self, '_tracking_state', None)
        if _ts is not None and getattr(_ts, 'initialized', False):
            gps_sign, _ = self._graph_curve_direction(_ts)
            _curve_dir = int(getattr(self, '_curve_direction', 0))
            if gps_sign != _curve_dir:
                return steering_angle

        previous_steering = getattr(self, "_last_good_steering", None)
        if previous_steering is None:
            previous_steering = getattr(self, "last_steering", None)

        transversal_recovery_active = False
        if isinstance(debug_info, dict):
            transversal_recovery_active = (
                bool(debug_info.get("single_line_transversal_detected", False)) or
                str(debug_info.get("single_line_mode", "") or "") == "transversal_recovery"
            )
        if transversal_recovery_active:
            curve_direction = int(getattr(self, "_curve_direction", 0) or 0)
            if (
                curve_state == "IN_CURVE" and
                curve_direction != 0 and
                previous_steering is not None and
                float(previous_steering) * curve_direction > 0.0 and
                float(steering_angle) * curve_direction < 0.0
            ):
                hold_ratio = max(
                    0.0,
                    min(1.0, float(getattr(_config, "SINGLE_LINE_CURVE_STEER_HOLD_RATIO", 0.45) or 0.45)),
                )
                min_steer_deg = max(2.0, abs(float(previous_steering)) * hold_ratio)
                guarded_steer = curve_direction * min(float(self.max_steering), float(min_steer_deg))
                if isinstance(debug_info, dict):
                    debug_info["single_line_transversal_curve_hold"] = True
                    debug_info["single_line_transversal_curve_hold_sign"] = int(curve_direction)
                    debug_info["single_line_transversal_curve_hold_input_deg"] = round(float(steering_angle), 3)
                    debug_info["single_line_transversal_curve_hold_prev_deg"] = round(float(previous_steering), 3)
                    debug_info["single_line_transversal_curve_hold_output_deg"] = round(float(guarded_steer), 3)
                return guarded_steer
            return steering_angle

        desired_sign = 0
        curve_direction = int(getattr(self, "_curve_direction", 0) or 0)
        if curve_direction != 0:
            # _curve_direction = 1 = right curve → steer right (+1)
            # _curve_direction = -1 = left curve → steer left (-1)
            desired_sign = curve_direction

        if desired_sign == 0 and direct_error_m is not None and abs(float(direct_error_m)) > 1e-4:
            desired_sign = -1 if float(direct_error_m) > 0.0 else 1

        if desired_sign == 0 and previous_steering is not None and abs(float(previous_steering)) > 0.2:
            desired_sign = 1 if float(previous_steering) > 0.0 else -1

        heading_deg = abs(math.degrees(float(heading_rad or 0.0)))
        if desired_sign == 0 and heading_deg > 0.2:
            desired_sign = 1 if float(heading_rad) > 0.0 else -1

        if desired_sign == 0:
            return steering_angle

        # When IN_CURVE with only one line visible, the car must steer at maximum to
        # complete the curve or recover from having gone outside the outer boundary.
        # This also prevents the steering from tapering off mid-curve (turns too wide).
        if curve_state == "IN_CURVE" and curve_direction != 0:
            steering_sign = 0
            if float(steering_angle) > 1.0:
                steering_sign = 1
            elif float(steering_angle) < -1.0:
                steering_sign = -1
            if steering_sign != 0 and steering_sign != desired_sign:
                if isinstance(debug_info, dict):
                    debug_info["single_line_max_curve_steer_suppressed"] = "input_sign_conflict"
                    debug_info["single_line_max_curve_steer_input_deg"] = round(float(steering_angle), 3)
                    debug_info["single_line_max_curve_steer_desired_sign"] = int(desired_sign)
                return steering_angle
            guarded_steer = float(desired_sign) * float(self.max_steering)
            if isinstance(debug_info, dict):
                debug_info["single_line_max_curve_steer"] = True
                debug_info["single_line_max_curve_steer_deg"] = round(guarded_steer, 3)
                debug_info["single_line_max_curve_steer_input_deg"] = round(float(steering_angle), 3)
            return guarded_steer

        min_steer_deg = max(
            float(getattr(_config, "SINGLE_LINE_CURVE_MIN_STEER_DEG", 2.0) or 2.0),
            heading_deg * float(
                getattr(_config, "SINGLE_LINE_CURVE_HEADING_STEER_GAIN", 1.0) or 1.0
            ),
        )
        # If geometric heading is weak (e.g. inner line is near-horizontal) but the AI hint
        # is confident about a significant curve, use the hint to raise the minimum steering.
        hint_rad = float(getattr(self, "_local_ai_heading_hint_rad", 0.0) or 0.0)
        hint_conf = float(getattr(self, "_local_ai_heading_hint_confidence", 0.0) or 0.0)
        hint_deg = abs(math.degrees(hint_rad))
        if hint_conf > 0.35 and hint_deg > 5.0:
            min_steer_deg = max(min_steer_deg, hint_deg * 0.4)

        if previous_steering is not None and float(previous_steering) * desired_sign > 0.0:
            hold_ratio = max(
                0.0,
                min(1.0, float(getattr(_config, "SINGLE_LINE_CURVE_STEER_HOLD_RATIO", 0.45) or 0.45)),
            )
            min_steer_deg = max(min_steer_deg, abs(float(previous_steering)) * hold_ratio)

        continuity_sign = 0
        continuity_min_deg = 0.0
        continuity_hold_active = False
        if (
            curve_state == "IN_CURVE" and
            previous_steering is not None and
            float(previous_steering) * desired_sign < 0.0
        ):
            single_line_streak = 0
            if isinstance(debug_info, dict):
                try:
                    single_line_streak = int(debug_info.get("single_line_streak", 0) or 0)
                except (TypeError, ValueError):
                    single_line_streak = 0
            continuity_window = max(
                3,
                int(getattr(self, "single_line_prefer_center_frames", 2) or 2) + 2,
            )
            prev_steer_abs = abs(float(previous_steering))
            hint_supports_curve = (
                hint_deg >= 20.0 or
                (hint_conf >= 0.20 and hint_deg >= 10.0) or
                heading_deg >= 4.0
            )
            if (
                0 < single_line_streak <= continuity_window and
                prev_steer_abs >= 4.0 and
                hint_supports_curve
            ):
                continuity_sign = 1 if float(previous_steering) > 0.0 else -1
                continuity_min_deg = max(min_steer_deg, prev_steer_abs * 0.70)
                continuity_hold_active = True

        min_steer_deg = min(float(self.max_steering), float(min_steer_deg))
        guarded_steer = float(steering_angle)
        if guarded_steer * desired_sign < min_steer_deg:
            guarded_steer = desired_sign * min_steer_deg
            if isinstance(debug_info, dict):
                debug_info["single_line_curve_priority"] = True
                debug_info["single_line_curve_priority_sign"] = int(desired_sign)
                debug_info["single_line_curve_priority_min_deg"] = round(float(min_steer_deg), 3)
                debug_info["single_line_curve_priority_input_deg"] = round(float(steering_angle), 3)
                debug_info["single_line_curve_priority_output_deg"] = round(float(guarded_steer), 3)

        if continuity_hold_active:
            continuity_min_deg = min(float(self.max_steering), float(continuity_min_deg))
            if continuity_sign != 0 and guarded_steer * continuity_sign < continuity_min_deg:
                guarded_steer = continuity_sign * continuity_min_deg
                if isinstance(debug_info, dict):
                    debug_info["single_line_curve_continuity_hold"] = True
                    debug_info["single_line_curve_continuity_sign"] = int(continuity_sign)
                    debug_info["single_line_curve_continuity_min_deg"] = round(float(continuity_min_deg), 3)
                    debug_info["single_line_curve_continuity_prev_deg"] = round(float(previous_steering), 3)
                    debug_info["single_line_curve_continuity_output_deg"] = round(float(guarded_steer), 3)

        return guarded_steer

    def _enforce_two_line_curve_priority(self, steering_angle, direct_error_m=None, debug_info=None):
        """Prevent two-line ENTERING from unwinding away from a confirmed curve."""
        if steering_angle is None:
            return steering_angle

        curve_state = str(getattr(self, "_curve_state", "STRAIGHT"))
        curve_origin = str(getattr(self, "_curve_enter_origin", "none"))
        hint_source = str(getattr(self, "_local_ai_heading_hint_source", "none") or "none")
        hint_conf = float(getattr(self, "_local_ai_heading_hint_confidence", 0.0) or 0.0)
        hint_rad = float(getattr(self, "_local_ai_heading_hint_rad", 0.0) or 0.0)
        hint_deg = abs(math.degrees(hint_rad))
        previous_steering = getattr(self, "_last_good_steering", None)
        if previous_steering is None:
            previous_steering = getattr(self, "last_steering", None)
        deadband_deg = max(0.0, float(getattr(self, "mpc_output_deadband_deg", 0.0) or 0.0))

        entering_hold_active = (
            curve_state == "ENTERING" and
            curve_origin == "two_line_hint"
        )
        continuity_hold_active = (
            not entering_hold_active and
            curve_state == "IN_CURVE" and
            hint_source == "two_line" and
            hint_conf >= 0.45 and
            hint_deg >= 16.0 and
            previous_steering is not None and
            abs(float(previous_steering)) >= max(deadband_deg + 0.25, 0.8) and
            abs(float(steering_angle)) <= max(deadband_deg + 0.05, abs(float(previous_steering)) * 0.55)
        )
        if not entering_hold_active and not continuity_hold_active:
            return steering_angle

        desired_sign = 0
        if continuity_hold_active and previous_steering is not None and abs(float(previous_steering)) > 1e-4:
            desired_sign = 1 if float(previous_steering) > 0.0 else -1
        else:
            curve_direction = int(getattr(self, "_curve_direction", 0) or 0)
            if curve_direction != 0:
                # _curve_direction = 1 = right curve → steer right (+1)
                desired_sign = curve_direction
        if desired_sign == 0 and abs(hint_rad) > 1e-4:
            desired_sign = -1 if hint_rad > 0.0 else 1
        if desired_sign == 0:
            return steering_angle

        # In 2-line mode we have a physical direct_error_m. If it says the car is already
        # biased toward the inside of the active curve, don't force extra steering deeper
        # into the curve; let the physical crosstrack controller unwind toward lane centre.
        if direct_error_m is not None:
            inside_error_threshold_m = max(
                0.0,
                float(getattr(_config, "TWO_LINE_CURVE_PRIORITY_INSIDE_ERROR_M", 0.03) or 0.03),
            )
            try:
                direct_error_value = float(direct_error_m)
            except (TypeError, ValueError):
                direct_error_value = 0.0
            if (
                inside_error_threshold_m > 0.0 and
                direct_error_value * desired_sign < -inside_error_threshold_m
            ):
                if isinstance(debug_info, dict):
                    debug_info["two_line_curve_priority_suppressed"] = "inside_line_correction"
                    debug_info["two_line_curve_priority_direct_error_m"] = round(direct_error_value, 4)
                    debug_info["two_line_curve_priority_inside_error_m"] = round(
                        float(inside_error_threshold_m), 4
                    )
                return steering_angle

        min_steer_deg = 0.0
        if entering_hold_active and hint_source == "two_line" and hint_conf >= 0.25 and hint_deg >= 12.0:
            min_steer_deg = max(2.5, hint_deg * 0.55)
        if continuity_hold_active:
            min_steer_deg = max(min_steer_deg, deadband_deg + 0.25)
            if hint_deg >= 12.0:
                min_steer_deg = max(min_steer_deg, hint_deg * 0.12)
        if previous_steering is not None and float(previous_steering) * desired_sign > 0.0:
            hold_ratio = 0.80 if continuity_hold_active else 0.65
            min_steer_deg = max(min_steer_deg, abs(float(previous_steering)) * hold_ratio)

        if min_steer_deg <= 0.0:
            return steering_angle

        min_steer_deg = min(float(self.max_steering), float(min_steer_deg))
        guarded_steer = float(steering_angle)
        if guarded_steer * desired_sign < min_steer_deg:
            guarded_steer = desired_sign * min_steer_deg
            if isinstance(debug_info, dict):
                debug_info["two_line_curve_priority"] = True
                debug_info["two_line_curve_priority_mode"] = (
                    "continuity_hold" if continuity_hold_active else "entering_guard"
                )
                debug_info["two_line_curve_priority_sign"] = int(desired_sign)
                debug_info["two_line_curve_priority_min_deg"] = round(float(min_steer_deg), 3)
                debug_info["two_line_curve_priority_input_deg"] = round(float(steering_angle), 3)
                debug_info["two_line_curve_priority_output_deg"] = round(float(guarded_steer), 3)

        return guarded_steer

    def _graph_curve_direction(self, tracking_state=None, min_heading_change_deg=15.0):
        """Return controller curve direction from graph lookahead heading change."""
        ts = tracking_state if tracking_state is not None else getattr(self, "_tracking_state", None)
        if ts is None or not getattr(ts, "initialized", False):
            return 0, 0.0

        delta_psi = float(getattr(ts, "path_heading_change_rad", 0.0) or 0.0)
        min_rad = math.radians(max(0.0, float(min_heading_change_deg or 0.0)))

        # Graph yaw uses the mathematical convention:
        #   +dpsi = left / counter-clockwise
        #   -dpsi = right / clockwise
        # The controller uses +1 for right curves and -1 for left curves.
        if delta_psi <= -min_rad:
            return 1, delta_psi
        if delta_psi >= min_rad:
            return -1, delta_psi
        return 0, delta_psi

    def _has_recent_two_line_straight_hint(self):
        """Return True briefly after vision reports a stable near-straight two-line heading."""
        hint_source = str(getattr(self, "_local_ai_heading_hint_source", "none") or "none")
        hint_conf = float(getattr(self, "_local_ai_heading_hint_confidence", 0.0) or 0.0)
        hint_rad = float(getattr(self, "_local_ai_heading_hint_rad", 0.0) or 0.0)
        min_conf = max(
            0.0,
            min(1.0, float(getattr(_config, "CURVE_TWO_LINE_STRAIGHT_HINT_CONFIDENCE", 0.65) or 0.65)),
        )
        max_deg = max(
            0.5,
            float(getattr(_config, "CURVE_TWO_LINE_STRAIGHT_HINT_MAX_DEG", 6.0) or 6.0),
        )
        hold_s = max(
            0.0,
            float(getattr(_config, "CURVE_TWO_LINE_STRAIGHT_HINT_HOLD_S", 0.45) or 0.45),
        )
        if (
            hint_source == "two_line" and
            hint_conf >= min_conf and
            abs(math.degrees(hint_rad)) <= max_deg
        ):
            self._recent_two_line_straight_until = time.time() + hold_s
            return True
        return time.time() <= float(getattr(self, "_recent_two_line_straight_until", 0.0) or 0.0)

    def _get_curve_exit_unwind_factor(
        self,
        reinforced_sign=None,
        debug_info=None,
        debug_prefix="curve_exit_unwind",
    ):
        """Return a damping factor when visual exit cues contradict the current curve."""
        curve_state = str(getattr(self, "_curve_state", "STRAIGHT"))
        active_curve_direction = int(getattr(self, "_curve_direction", 0) or 0)
        recent_curve_frames = int(getattr(self, "_recent_curve_direction_frames", 0) or 0)
        if curve_state == "STRAIGHT" and active_curve_direction == 0 and recent_curve_frames > 0:
            active_curve_direction = int(getattr(self, "_recent_curve_direction", 0) or 0)

        if active_curve_direction == 0 or curve_state not in ("EXITING", "STRAIGHT"):
            return 1.0

        reinforced_sign_value = 0
        if reinforced_sign is not None:
            try:
                reinforced_sign_value = 1 if float(reinforced_sign) > 0.0 else -1
            except (TypeError, ValueError):
                reinforced_sign_value = 0
            if reinforced_sign_value * active_curve_direction <= 0:
                return 1.0

        hint_conf = float(getattr(self, "_local_ai_heading_hint_confidence", 0.0) or 0.0)
        hint_rad = float(getattr(self, "_local_ai_heading_hint_rad", 0.0) or 0.0)
        hint_deg = abs(math.degrees(hint_rad))
        if hint_conf < 0.35 or hint_deg < 3.0:
            return 1.0

        hint_sign = 0
        if hint_rad > 1e-4:
            hint_sign = 1
        elif hint_rad < -1e-4:
            hint_sign = -1
        if hint_sign == 0 or hint_sign == active_curve_direction:
            return 1.0

        strong_exit_hint = hint_conf >= 0.60 or hint_deg >= 6.0
        if curve_state == "EXITING":
            unwind_factor = 0.20 if strong_exit_hint else 0.35
        else:
            unwind_factor = 0.35 if strong_exit_hint else 0.55

        if isinstance(debug_info, dict):
            debug_info[debug_prefix] = True
            debug_info[f"{debug_prefix}_factor"] = round(float(unwind_factor), 3)
            debug_info[f"{debug_prefix}_curve_state"] = curve_state
            debug_info[f"{debug_prefix}_curve_direction"] = int(active_curve_direction)
            debug_info[f"{debug_prefix}_hint_deg"] = round(float(math.degrees(hint_rad)), 3)
            debug_info[f"{debug_prefix}_hint_conf"] = round(float(hint_conf), 4)
            debug_info[f"{debug_prefix}_reinforced_sign"] = int(reinforced_sign_value)

        return unwind_factor

    def _resolve_ai_local_blind_control(self, img_w, speed_cap=None, debug_info=None):
        """Return a safe blind-control fallback for AI_LOCAL when no lanes are visible.

        Priority order:
        1. Route tracking when navigation context says the blind segment belongs to the graph.
        2. Brief visual hold using the last stable steering.
        3. Full safety stop.
        """
        nav_ctx = self._get_navigation_context()
        use_route_tracking, route_reason = self._should_use_route_blind_guidance(nav_ctx)
        if isinstance(debug_info, dict):
            ts = getattr(self, "_tracking_state", None)
            debug_info["blind_control_route_candidate"] = bool(use_route_tracking)
            debug_info["blind_control_route_reason"] = str(route_reason)
            debug_info["blind_control_frames_without_line"] = int(
                getattr(self, "frames_without_line", 0) or 0
            )
            debug_info["blind_control_curve_state"] = str(
                getattr(self, "_curve_state", "STRAIGHT")
            )
            debug_info["blind_control_planner_priority"] = bool(
                nav_ctx.get("planner_priority", False)
            )
            if ts is not None:
                debug_info["blind_control_path_heading_change_deg"] = round(
                    math.degrees(float(getattr(ts, "path_heading_change_rad", 0.0) or 0.0)),
                    3,
                )
                debug_info["blind_control_map_match_error_m"] = round(
                    float(getattr(ts, "map_match_error_m", 0.0) or 0.0),
                    5,
                )

        if use_route_tracking:
            ts = self._tracking_state
            if ts is not None and getattr(ts, "initialized", False):
                steering = self._compute_lateral_control(
                    0, 0, 0,
                    speed_cap=speed_cap,
                )
                hold_speed = float(self.min_speed)
                if speed_cap is not None:
                    hold_speed = min(hold_speed, float(speed_cap))
                return steering, hold_speed, "route_tracking"

        blind_hold_steering = self._get_no_lane_hold_steering()
        if blind_hold_steering is not None:
            hold_speed = float(self.min_speed)
            if speed_cap is not None:
                hold_speed = min(hold_speed, float(speed_cap))
            return blind_hold_steering, hold_speed, "visual_fallback"

        # Secondary fallback: decay last steering for a few more frames before stopping
        # (matches master branch behaviour: last_steering * 0.95 for frames_without_line < 5
        # after the hard hold expires).
        frames_without = int(getattr(self, 'frames_without_line', 0) or 0)
        decay_limit = int(getattr(self, 'no_lane_hold_steering_frames', 4) or 4) + 5
        last_steer = getattr(self, '_last_good_steering', None)
        if last_steer is None:
            last_steer = getattr(self, 'last_steering', None)
        if last_steer is not None and frames_without <= decay_limit:
            decayed = float(last_steer) * 0.90
            decayed = max(-self.max_steering, min(self.max_steering, decayed))
            hold_speed = float(self.min_speed)
            if speed_cap is not None:
                hold_speed = min(hold_speed, float(speed_cap))
            return decayed, hold_speed, "visual_fallback"

        return 0.0, 0.0, "safety_stop"

    def _resolve_nominal_lane_width_px(self, img_w):
        """Estimate nominal lane width in pixels from cache or frame fallback."""
        cached_lane_width = float(getattr(self, '_last_local_ai_lane_width_px', 0.0) or 0.0)
        if cached_lane_width > 1.0:
            return cached_lane_width

        cached_px_per_cm = float(getattr(self, '_last_px_per_cm', 0.0) or 0.0)
        lane_width_cm = float(getattr(self, 'lane_width_cm', 35.0) or 35.0)
        if cached_px_per_cm > 0.5 and lane_width_cm > 1e-3:
            return cached_px_per_cm * lane_width_cm

        return max(10.0, float(img_w) * 0.45)

    def _collapse_overlapping_two_lines(self, avg_left, avg_right, img_h, img_w, prev_seen_side=None):
        """Treat near-overlapping two-line detections as a single visible lane line."""
        if avg_left is None or avg_right is None:
            return avg_left, avg_right, None

        y_ref = int(max(0, min(img_h - 1, round(img_h * (1.0 - float(getattr(self, 'lookahead', 0.4)))))))
        left_x = self._line_x_at_y(avg_left, y_ref)
        right_x = self._line_x_at_y(avg_right, y_ref)
        if left_x is None or right_x is None:
            return avg_left, avg_right, None

        line_gap_px = abs(float(right_x) - float(left_x))
        nominal_lane_width_px = self._resolve_nominal_lane_width_px(img_w)
        gap_ratio = max(0.05, min(0.35, float(getattr(self, 'duplicate_line_gap_ratio', 0.18) or 0.18)))
        overlap_gap_limit_px = max(6.0, nominal_lane_width_px * gap_ratio, float(img_w) * 0.02)
        if line_gap_px > overlap_gap_limit_px:
            return avg_left, avg_right, None

        merged_line = np.rint(
            (avg_left.astype(np.float32) + avg_right.astype(np.float32)) * 0.5
        ).astype(np.int32)

        resolved_side = None
        resolution_source = "unknown"
        try:
            resolved_side, resolution_source = self._resolve_ambiguous_local_lane(
                merged_line,
                img_h,
                img_w,
                prev_seen_side=prev_seen_side,
            )
        except Exception:
            resolved_side = None
            resolution_source = "resolution_error"

        if resolved_side not in ('left', 'right'):
            center_x = (float(left_x) + float(right_x)) * 0.5
            resolved_side = 'left' if center_x < (img_w / 2.0) else 'right'
            resolution_source = "frame_center"

        if resolved_side == 'left':
            collapsed_left = merged_line
            collapsed_right = None
        else:
            collapsed_left = None
            collapsed_right = merged_line

        collapse_info = {
            'kept_side': resolved_side,
            'dropped_side': ('right' if resolved_side == 'left' else 'left'),
            'resolution_source': resolution_source,
            'line_gap_px': float(line_gap_px),
            'line_gap_limit_px': float(overlap_gap_limit_px),
            'nominal_lane_width_px': float(nominal_lane_width_px),
        }
        return collapsed_left, collapsed_right, collapse_info

    def _interp_line_x_at_y(self, line, target_y):
        """Return the x coordinate of a line segment at target_y by linear interpolation.

        line must be a numpy array of shape (1, 4) = [[x1, y1, x2, y2]] (BFMC format).
        If target_y is outside the segment or the segment is vertical, returns None.
        """
        try:
            x1, y1, x2, y2 = float(line[0][0]), float(line[0][1]), float(line[0][2]), float(line[0][3])
        except (IndexError, TypeError, ValueError):
            return None
        dy = y2 - y1
        if abs(dy) < 1e-6:
            return None
        t = (float(target_y) - y1) / dy
        if not (0.0 <= t <= 1.0):
            return None   # target_y fuera del segmento → no extrapolar
        return x1 + t * (x2 - x1)

    def _compute_sl_physical_error(self, side, real_ref_x, img_w, px_per_cm, debug_info=None):
        """Physical direct_error_m for single-line mode with safety margin.

        Uses the REAL detected line position at reference_y and the known lane width to
        compute D_left_cm / D_right_cm (one measured, one estimated via lane_width_cm) and
        then applies the same safety margin as two-line direct_error_m.

        Inner vs outer distinction: the formula handles both cases correctly —
        if the visible line is the outer line, D_visible will be small and the safety
        margin pushes the car away from it; if it is the inner line, D_other is small
        and the margin pushes the car back toward center.

        Returns None if the inputs are invalid.
        """
        if px_per_cm < 0.5 or real_ref_x is None or real_ref_x <= 0.0:
            return None

        _lw_cm      = float(self.lane_width_cm)
        _safety_cm  = float(getattr(self, 'lane_safety_margin_cm', 5.0))
        _car_half   = float(self.car_width) / 2.0

        if side == 'left':
            D_left_cm  = (img_w / 2.0 - float(real_ref_x)) / px_per_cm
            D_right_cm = _lw_cm - D_left_cm
        else:
            D_right_cm = (float(real_ref_x) - img_w / 2.0) / px_per_cm
            D_left_cm  = _lw_cm - D_right_cm

        # Rear-axle offtracking: the camera is at the front; when turning, the rear axle
        # follows a tighter radius (offtracking = sqrt(R²+L²) − R, R = wheelbase/tan(δ)).
        # Add this extra distance to the INNER boundary safety so the rear body corner
        # never touches the inner lane line — same principle as a truck turning wide.
        # Controlled by OFFTRACK_SCALE (0=off, 1=full) and OFFTRACK_CURVE_ONLY.
        _steer_val  = float(getattr(self, '_last_good_steering', 0.0) or 0.0)
        _steer_rad  = math.radians(abs(_steer_val))
        _wbase_cm   = float(getattr(self, 'car_wheelbase_cm', 26.0))
        _ot_scale   = float(getattr(_config, 'OFFTRACK_SCALE', 1.0))
        _ot_curve_only = bool(getattr(_config, 'OFFTRACK_CURVE_ONLY', False))
        _in_curve   = getattr(self, '_curve_state', 'STRAIGHT') in ('ENTERING', 'IN_CURVE', 'EXITING')
        if _steer_rad > 0.017 and _ot_scale > 0.0 and (not _ot_curve_only or _in_curve):
            _R_cm        = _wbase_cm / math.tan(_steer_rad)
            _offtrack_cm = (math.sqrt(_R_cm ** 2 + _wbase_cm ** 2) - _R_cm) * _ot_scale
        else:
            _offtrack_cm = 0.0

        direct_error_m = (D_right_cm - _lw_cm / 2.0) / 100.0
        if _safety_cm > 0.0:
            if _steer_val < -1.0:
                # Turning left → inner = left boundary
                _lv = max(0.0, (_car_half + _safety_cm + _offtrack_cm) - D_left_cm)
                _rv = max(0.0, (_car_half + _safety_cm) - D_right_cm)
            elif _steer_val > 1.0:
                # Turning right → inner = right boundary
                _lv = max(0.0, (_car_half + _safety_cm) - D_left_cm)
                _rv = max(0.0, (_car_half + _safety_cm + _offtrack_cm) - D_right_cm)
            else:
                _lv = max(0.0, (_car_half + _safety_cm) - D_left_cm)
                _rv = max(0.0, (_car_half + _safety_cm) - D_right_cm)
            direct_error_m += _lv / 100.0   # push right if too close to left
            direct_error_m -= _rv / 100.0   # push left  if too close to right

        if isinstance(debug_info, dict):
            is_outer = bool(debug_info.get('single_line_outer_curve_line', False))
            debug_info['sl_D_left_cm']       = round(D_left_cm, 2)
            debug_info['sl_D_right_cm']      = round(D_right_cm, 2)
            debug_info['sl_direct_error_m']  = round(direct_error_m, 4)
            debug_info['sl_px_per_cm']       = round(px_per_cm, 4)
            debug_info['sl_is_outer_line']   = is_outer
            debug_info['sl_offtrack_cm']     = round(_offtrack_cm, 2)

        return direct_error_m

    def _get_single_line_physical_direct_error(
        self,
        side,
        mask_guidance_line,
        local_mask_guidance,
        img_w,
        debug_info=None,
    ):
        """Resolve single-line physical direct_error_m at the shared reference_y."""
        local_mask_guidance = local_mask_guidance if isinstance(local_mask_guidance, dict) else {}
        side_key = 'left_x' if side == 'left' else 'right_x'
        _std_ref_y = int(getattr(self, '_last_two_line_reference_y', 0) or 0)
        _sl_ppc = float(getattr(self, '_last_two_line_ref_px_per_cm', 0.0) or 0.0)
        if _std_ref_y > 0 and _sl_ppc > 0.5 and mask_guidance_line is not None:
            _sl_ref_x = self._interp_line_x_at_y(mask_guidance_line, _std_ref_y)
            # A nearly-horizontal line can project outside the image at the shared
            # reference_y; fall back to the native single-line sample to avoid spikes.
            if _sl_ref_x is None or not (1.0 <= _sl_ref_x <= float(img_w) - 1.0):
                if isinstance(debug_info, dict):
                    _sl_ppc = float(debug_info.get('single_line_px_per_cm', 0.0) or 0.0)
                else:
                    _sl_ppc = 0.0
                _sl_ref_x = float(local_mask_guidance.get(side_key, 0.0) or 0.0)
        else:
            if isinstance(debug_info, dict):
                _sl_ppc = float(debug_info.get('single_line_px_per_cm', 0.0) or 0.0)
            else:
                _sl_ppc = 0.0
            _sl_ref_x = float(local_mask_guidance.get(side_key, 0.0) or 0.0)
        return self._compute_sl_physical_error(
            side, _sl_ref_x, img_w, _sl_ppc, debug_info
        )

    def _should_apply_inner_line_escape(self, visible_side, direct_error_m=None, debug_info=None):
        """Allow hard counter-steer only when the inner-line crowding is severe."""
        if not self._is_seeing_inner_line(visible_side):
            return False

        curve_state = str(getattr(self, "_curve_state", "STRAIGHT"))
        if curve_state not in ("IN_CURVE", "EXITING"):
            return False

        # In a confirmed curve, seeing only the inner line means the outer line was lost —
        # the car went outside the curve boundary, not inside. Applying inner_line_escape
        # would steer the car further in the wrong direction. Suppress it so
        # _enforce_single_line_curve_priority can apply max recovery steering instead.
        curve_dir = int(getattr(self, '_curve_direction', 0) or 0)
        if curve_dir != 0:
            if isinstance(debug_info, dict):
                debug_info["inner_line_escape_suppressed"] = "inner_only_means_outer_lost"
            return False

        if isinstance(debug_info, dict):
            debug_info["inner_line_escape_candidate"] = True
            if direct_error_m is not None:
                debug_info["inner_line_escape_direct_error_m"] = round(float(direct_error_m), 4)

        if direct_error_m is None:
            if isinstance(debug_info, dict):
                debug_info["inner_line_escape_suppressed"] = "no_physical_error"
            return False

        escape_sign = 1.0 if visible_side == 'left' else -1.0
        physical_escape_supported = float(direct_error_m) * escape_sign > 0.0
        if not physical_escape_supported:
            if isinstance(debug_info, dict):
                debug_info["inner_line_escape_suppressed"] = "physical_sign_conflict"
            return False

        hard_escape_min_error_m = max(
            0.0,
            float(getattr(_config, "CURVE_INNER_LINE_HARD_ESCAPE_MIN_ERROR_M", 0.09) or 0.09),
        )
        if abs(float(direct_error_m)) < hard_escape_min_error_m:
            if isinstance(debug_info, dict):
                debug_info["inner_line_escape_suppressed"] = "physical_error_below_threshold"
                debug_info["inner_line_escape_min_error_m"] = round(hard_escape_min_error_m, 4)
            return False

        return True

    def _blend_single_line_transition_steering(self, steering_angle, side, prev_seen_side, streak):
        """Blend single-line steering briefly after side/mode transitions to avoid spikes."""
        if steering_angle is None:
            return steering_angle, None
        if side not in ('left', 'right'):
            return steering_angle, None

        blend_frames = max(0, int(getattr(self, 'single_line_transition_blend_frames', 2) or 0))
        streak = max(0, int(streak or 0))
        if blend_frames <= 0 or streak <= 0 or streak > blend_frames:
            return steering_angle, None

        prev_mode = prev_seen_side if prev_seen_side in ('left', 'right', 'both') else 'none'
        if prev_mode == side:
            return steering_angle, None

        previous_steering = getattr(self, 'last_steering', None)
        if previous_steering is None:
            previous_steering = getattr(self, '_last_good_steering', None)
        if previous_steering is None:
            return steering_angle, None

        # If the required steering direction reverses (e.g. previous max-right in a right
        # curve, new max-left in the immediately following left curve), blending would keep
        # the car going the wrong way for several frames.  Skip the blend entirely in that
        # case and apply the new steering immediately.
        curve_state = str(getattr(self, '_curve_state', 'STRAIGHT'))
        if curve_state in ("ENTERING", "IN_CURVE", "EXITING"):
            if (float(previous_steering) > 2.0 and float(steering_angle) < -2.0) or \
               (float(previous_steering) < -2.0 and float(steering_angle) > 2.0):
                return float(steering_angle), None

        alpha = min(1.0, float(streak) / float(blend_frames + 1))
        blended = ((1.0 - alpha) * float(previous_steering)) + (alpha * float(steering_angle))
        max_steer = float(getattr(self, 'max_steering', 25.0) or 25.0)
        blended = max(-max_steer, min(max_steer, blended))
        return blended, {
            'applied': True,
            'alpha': float(alpha),
            'prev_seen_side': prev_mode,
            'resolved_side': side,
            'streak': int(streak),
            'blend_frames': int(blend_frames),
            'input_steering': float(steering_angle),
            'base_steering': float(previous_steering),
        }

    def _stabilize_single_line_error(self, error_px, side, prev_seen_side, streak, lane_width_px=None, img_w=None):
        """Limit abrupt 2->1 line error jumps before lateral control."""
        if error_px is None:
            return None, None
        if side not in ('left', 'right'):
            return float(error_px), None

        stabilized = float(error_px)
        debug = {
            'applied': False,
            'input_error_px': float(error_px),
            'resolved_side': side,
            'prev_seen_side': prev_seen_side if prev_seen_side in ('left', 'right', 'both') else 'none',
            'streak': int(max(0, int(streak or 0))),
        }

        last_error = getattr(self, '_last_good_error', None)
        if last_error is not None:
            last_error = float(last_error)
            blend_frames = max(0, int(getattr(self, 'single_line_transition_blend_frames', 2) or 0))
            transition_active = (
                blend_frames > 0 and
                debug['streak'] > 0 and
                debug['streak'] <= (blend_frames + 1) and
                debug['prev_seen_side'] != side
            )
            if transition_active:
                alpha = min(1.0, float(debug['streak']) / float(blend_frames + 1))
                stabilized = ((1.0 - alpha) * last_error) + (alpha * stabilized)
                debug['transition_blend_alpha'] = float(alpha)
                debug['base_error_px'] = float(last_error)
                debug['applied'] = True

            width_px = 0.0
            if lane_width_px is not None:
                try:
                    width_px = float(lane_width_px)
                except (TypeError, ValueError):
                    width_px = 0.0
            if width_px <= 1.0 and img_w is not None:
                width_px = float(self._resolve_nominal_lane_width_px(img_w))

            jump_limit_px = max(
                20.0,
                float(getattr(self, 'noise_max_error_jump_px', 80)) * 0.55,
                width_px * 0.28 if width_px > 1.0 else 0.0,
            )
            if debug['streak'] > max(1, int(getattr(self, 'single_line_transition_blend_frames', 2) or 0)):
                jump_limit_px *= 1.35

            delta = stabilized - last_error
            if delta > jump_limit_px:
                stabilized = last_error + jump_limit_px
                debug['jump_clamped'] = True
                debug['jump_limit_px'] = float(jump_limit_px)
                debug['applied'] = True
            elif delta < -jump_limit_px:
                stabilized = last_error - jump_limit_px
                debug['jump_clamped'] = True
                debug['jump_limit_px'] = float(jump_limit_px)
                debug['applied'] = True

            abs_limit_px = max(
                45.0,
                float(getattr(self, 'max_error_px', 50.0)) * 1.8,
                width_px * 0.62 if width_px > 1.0 else 0.0,
            )
            limited = max(-abs_limit_px, min(abs_limit_px, stabilized))
            if limited != stabilized:
                stabilized = limited
                debug['absolute_clamped'] = True
                debug['absolute_limit_px'] = float(abs_limit_px)
                debug['applied'] = True

        debug['output_error_px'] = float(stabilized)
        return stabilized, (debug if debug.get('applied', False) else None)

    def _reset_pid_state(self):
        """Reset PID/Stanley state when changing modes to prevent corrupted values."""
        self.pid.reset()
        self.stanley.reset()
        self.lateral_mpc.reset()
        self._init_error_kalman()
        self._heading_error = 0.0
        self._yaw_rate = 0.0
        self._measured_steer_delta = 0.0
        self._last_speed_time = None
        self._stanley_last_psi_ss = 0.0
        self._stanley_last_traj_yaw_rate = 0.0
        self._stanley_last_steer_damping = 0.0
        self._stanley_last_speed_mps = 0.0
        self._stanley_last_speed_source = "none"
        self._stanley_last_error_m = 0.0
        self._stanley_last_px_per_cm = 0.0
        self._stanley_last_curve_guard = "none"
        self._stanley_sched_k = float(getattr(self, '_stanley_base_k', self.stanley_k))
        self._stanley_sched_k_soft = float(getattr(self, '_stanley_base_k_soft', self.stanley_k_soft))
        self._local_ai_heading_hint_rad = 0.0
        self._local_ai_heading_hint_confidence = 0.0
        self._local_ai_heading_hint_source = "none"
        self._local_ai_road_type_class = "unknown"
        self._local_ai_road_type_confidence = 0.0
        self._local_ai_lead_distance_class = "none"
        self._local_ai_lead_distance_confidence = 0.0
        self._local_ai_lead_distance_area = 0.0
        self._theta_abs_error_history.clear()
        self._offset_abs_error_history.clear()
        self._theta_dmae_deg = 0.0
        self._offset_dma_m = 0.0
        self._single_line_heading_ref_left = 0.0
        self._single_line_heading_ref_right = 0.0
        self._last_two_line_left = None
        self._last_two_line_right = None
        self._last_two_line_ts = 0.0
        if hasattr(self, '_lane_observation_history') and self._lane_observation_history is not None:
            self._lane_observation_history.clear()
        self._last_local_ai_duplicate_collapse = None
        self._last_single_line_projection_debug = {}
        self._two_line_curve_hint_count = 0
        self._two_line_curve_hint_direction = 0
        self._curve_enter_origin = "none"
        self.previous_error = 0
        if hasattr(self, 'last_steering'):
            del self.last_steering
        self.last_line_position = None
        self.frames_without_line = 0
        self.last_turn_direction = 0
        self._hybridnets_frame_count = 0
        print("\033[1;97m[ Line Following ] :\033[0m \033[1;96mPID\033[0m - State reset (P/I/D cleared)")

    def _init_error_kalman(self):
        """Initialize 1D Kalman filter for lateral error (state: error + error velocity)."""
        q = max(1e-5, float(self.kalman_process_noise))
        r = max(1e-3, float(self.kalman_measurement_noise))

        self.kalman_prediction_damping = max(0.0, min(0.999, float(self.kalman_prediction_damping)))
        self.kalman_max_prediction_frames = max(0, int(self.kalman_max_prediction_frames))

        self._kalman_error.transitionMatrix = np.array([[1.0, 0.05], [0.0, self.kalman_prediction_damping]], dtype=np.float32)
        self._kalman_error.measurementMatrix = np.array([[1.0, 0.0]], dtype=np.float32)
        self._kalman_error.processNoiseCov = np.array([[q, 0.0], [0.0, q * 3.0]], dtype=np.float32)
        self._kalman_error.measurementNoiseCov = np.array([[r]], dtype=np.float32)
        self._kalman_error.errorCovPost = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self._kalman_error.statePost = np.zeros((2, 1), dtype=np.float32)
        self._kalman_error.statePre = np.zeros((2, 1), dtype=np.float32)

        self._kalman_initialized = False
        self._kalman_last_ts = None
        self._kalman_prediction = 0.0
        self._kalman_age_frames = 0

    def _kalman_predict_error(self):
        """Predict crosstrack error for current frame when measurements are unavailable."""
        if not self.use_kalman_filter or not self._kalman_initialized:
            return None

        now = time.time()
        if self._kalman_last_ts is None:
            dt = 0.05
        else:
            dt = max(0.02, min(now - self._kalman_last_ts, 0.25))
        self._kalman_last_ts = now

        self._kalman_error.transitionMatrix[0, 1] = dt
        self._kalman_error.transitionMatrix[1, 1] = self.kalman_prediction_damping

        prediction = self._kalman_error.predict()
        self._kalman_prediction = float(prediction[0, 0])
        self._kalman_age_frames += 1
        return self._kalman_prediction

    def _kalman_correct_error(self, measured_error):
        """Correct Kalman state with measured crosstrack error and return filtered value."""
        measured = float(measured_error)
        if not self.use_kalman_filter:
            return measured

        now = time.time()
        if not self._kalman_initialized:
            self._kalman_error.statePost = np.array([[measured], [0.0]], dtype=np.float32)
            self._kalman_error.statePre = self._kalman_error.statePost.copy()
            self._kalman_initialized = True
            self._kalman_last_ts = now
            self._kalman_prediction = measured
            self._kalman_age_frames = 0
            return measured

        correction = self._kalman_error.correct(np.array([[measured]], dtype=np.float32))
        filtered = float(correction[0, 0])
        self._kalman_prediction = filtered
        self._kalman_age_frames = 0
        self._kalman_last_ts = now
        return filtered

    @property
    def _needs_debug(self):
        """Check if debug images are needed (local display or dashboard streaming)."""
        return self.show_debug or self.stream_debug_view > 0

    def _is_window_enabled(self, window_key):
        """Check if a specific debug window is enabled via DEBUG_WINDOWS config.
        Returns True if show_debug is on AND the individual window toggle is on."""
        return self.show_debug and self.debug_windows.get(window_key, False)

    def _begin_preview_cycle(self, now=None):
        """Determine whether local preview windows may refresh this loop."""
        if not self.show_debug:
            self._preview_frame_due = False
            self._preview_refresh_pending = False
            return
        current_time = time.time() if now is None else now
        self._preview_frame_due = (current_time - self._last_preview_time) >= self.preview_interval
        self._preview_refresh_pending = False

    def _show_preview_window(self, name, image):
        """Render a preview window only when the refresh budget allows it."""
        if image is None or not self.show_debug or not self._preview_frame_due:
            return
        cv2.imshow(name, image)
        self._preview_refresh_pending = True

    def _flush_preview_windows(self):
        """Flush pending preview updates with a single waitKey."""
        if self._preview_refresh_pending:
            cv2.waitKey(1)
            self._last_preview_time = time.time()
        self._preview_refresh_pending = False
        self._preview_frame_due = False

    def _debug_log(self, key, message, interval=1.0):
        """Rate-limit noisy debug logs to keep the control loop responsive."""
        if not self.show_debug:
            return
        now = time.time()
        last_time = self._debug_log_times.get(key, 0.0)
        if (now - last_time) < interval:
            return
        self._debug_log_times[key] = now
        print(message)

    def _store_debug_image(self, name, image):
        """Store a debug image for potential streaming."""
        if self.stream_debug_view == 0:
            return  # No streaming active, skip copy
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
        mode_colors = {'opencv': (0, 200, 0), 'lstr': (255, 100, 0), 'hybrid': (255, 255, 0), 'ai_local': (255, 180, 0)}
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
        
        self._show_preview_window("Control Panel", panel)

    def _show_final_steering_preview(self, computed_steering, commanded_steering, command_source):
        """Render a compact preview that always shows the final steering angle in use."""
        panel_width = 420
        panel_height = 130
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (22, 22, 30)

        cv2.rectangle(panel, (0, 0), (panel_width, 34), (36, 36, 52), -1)
        cv2.putText(
            panel,
            "FINAL STEERING ANGLE",
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
        )

        final_angle = commanded_steering if commanded_steering is not None else computed_steering
        if final_angle is None:
            value_text = "--"
            steer_color = (160, 160, 160)
        else:
            value_text = f"{float(final_angle):+.1f} deg"
            abs_angle = abs(float(final_angle))
            steer_color = (
                (0, 255, 0)
                if abs_angle < 10
                else (0, 165, 255)
                if abs_angle < 20
                else (0, 0, 255)
            )

        cv2.putText(panel, "Final:", (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(panel, value_text, (85, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.9, steer_color, 2)
        cv2.putText(
            panel,
            f"Computed: {'--' if computed_steering is None else f'{float(computed_steering):+.1f} deg'}",
            (14, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (160, 230, 255),
            1,
        )
        cv2.putText(
            panel,
            f"Commanded: {'--' if commanded_steering is None else f'{float(commanded_steering):+.1f} deg'}",
            (14, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            panel,
            f"Source: {command_source}",
            (230, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
        )

        bar_x1 = 14
        bar_x2 = panel_width - 14
        bar_y = panel_height - 18
        cv2.rectangle(panel, (bar_x1, bar_y - 4), (bar_x2, bar_y + 4), (55, 55, 70), -1)
        center_x = (bar_x1 + bar_x2) // 2
        cv2.line(panel, (center_x, bar_y - 7), (center_x, bar_y + 7), (140, 140, 160), 1)
        if final_angle is not None and self.max_steering > 0:
            clamped = max(-self.max_steering, min(self.max_steering, float(final_angle)))
            offset = int((clamped / float(self.max_steering)) * ((bar_x2 - bar_x1) / 2.0))
            marker_x = max(bar_x1, min(bar_x2, center_x + offset))
            cv2.circle(panel, (marker_x, bar_y), 7, steer_color, -1, cv2.LINE_AA)

        self._show_preview_window("2. Final Steering Angle", panel)

    def _overlay_command_status(self, image, computed_steering, computed_speed,
                                commanded_steering, commanded_speed, command_source):
        """Overlay computed vs commanded control on the debug image."""
        if image is None:
            return None

        overlay = image
        height, width = overlay.shape[:2]
        panel_top = max(0, height - 46)
        cv2.rectangle(overlay, (0, panel_top), (width, height), (15, 15, 15), -1)
        computed_text = (
            f"Computed: {computed_steering:.1f}deg / {computed_speed:.1f}"
            if computed_steering is not None and computed_speed is not None
            else "Computed: --"
        )
        if commanded_steering is not None:
            if commanded_speed is not None:
                commanded_text = f"Commanded: {commanded_steering:.1f}deg / {commanded_speed:.1f}"
            else:
                commanded_text = f"Commanded: {commanded_steering:.1f}deg / speed hold"
        else:
            commanded_text = "Commanded: --"
        source_text = f"Source: {command_source}"
        cv2.putText(overlay, computed_text, (8, panel_top + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(overlay, commanded_text, (8, panel_top + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(overlay, source_text, (max(8, width - 180), panel_top + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        return overlay

    def _send_debug_stream(self, computed_steering, computed_speed,
                           commanded_steering, commanded_speed, command_source):
        """Send the final commanded status and, if enabled, the selected debug stream."""
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
        
        view_name = view_names.get(self.stream_debug_view, 'off')

        try:
            debug_img = None
            if self.stream_debug_view > 0 and view_name in self._debug_images:
                stored_img = self._debug_images.get(view_name)
                if stored_img is not None:
                    debug_img = stored_img.copy()

            if debug_img is not None:
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

            actuator_status = self._last_actuator_status or {}
            status = {
                'steering': round(commanded_steering, 2) if commanded_steering is not None else (round(computed_steering, 2) if computed_steering is not None else None),
                'speed': round(commanded_speed, 2) if commanded_speed is not None else (round(computed_speed, 2) if computed_speed is not None else None),
                'mode': self.detection_mode,
                'control_policy_mode': self._control_policy_mode,
                'control_authority': self._control_authority,
                'active_maneuver': self._active_maneuver_name,
                'planner_priority': bool(self._planner_priority_active),
                'safety_stop_reason': self._safety_stop_reason or None,
                'maneuver_notes': self._last_maneuver_notes or None,
                'view': view_name,
                'active': self.is_line_following_active,
                'lstr_available': self.lstr_detector is not None and self.lstr_detector.is_available,
                'computed_steering': round(computed_steering, 2) if computed_steering is not None else None,
                'computed_speed': round(computed_speed, 2) if computed_speed is not None else None,
                'commanded_steering': round(commanded_steering, 2) if commanded_steering is not None else None,
                'commanded_speed': round(commanded_speed, 2) if commanded_speed is not None else None,
                'command_source': command_source,
                'mission_state': self._mission_state_name,
                'actual_steering': round(self._measured_steer, 2),
                'actual_speed': round(self._measured_speed, 2),
                'actual_speed_mps': round(
                    abs(float(self._measured_speed)) * float(self.stanley_measured_speed_to_mps), 4
                ),
                'stanley_speed_mps': round(float(self._stanley_last_speed_mps), 4),
                'stanley_speed_source': self._stanley_last_speed_source,
                'stanley_error_m': round(float(self._stanley_last_error_m), 4),
                'stanley_px_per_cm': round(float(self._stanley_last_px_per_cm), 4),
                'curve_state': self._curve_state,
                'curve_state_frames': int(self._curve_state_frames),
                'curve_direction': int(self._curve_direction),
                'serial_connected': actuator_status.get('serial_connected'),
                'engine_enabled': actuator_status.get('engine_enabled'),
                'klem_mode': actuator_status.get('klem_mode'),
                'actuator_command_type': actuator_status.get('command_type'),
                'actuator_speed_x10': actuator_status.get('speed_x10'),
                'actuator_steer_x10': actuator_status.get('steer_x10'),
                'actuator_blocked_reason': actuator_status.get('blocked_reason'),
                'frame_age_ms': self._last_frame_age_ms,
                'local_ai_ready': False,
                'local_ai_infer_ms': None,
                'local_ai_fps': None,
            }
            if self._tracking_state is not None:
                status.update({
                    'route_active': bool(getattr(self._tracking_state, 'route_active', False)),
                    'route_id': getattr(self._tracking_state, 'route_id', None),
                    'current_node_id': getattr(self._tracking_state, 'current_node_id', None),
                    'current_node_attr': int(getattr(self._tracking_state, 'current_node_attr', 0) or 0),
                    'upcoming_node_id': getattr(self._tracking_state, 'upcoming_node_id', None),
                    'upcoming_node_attr': int(getattr(self._tracking_state, 'upcoming_node_attr', 0) or 0),
                    'maneuver_type': getattr(self._tracking_state, 'maneuver_type', 'none'),
                    'next_semantic_id': getattr(self._tracking_state, 'next_semantic_id', None),
                    'next_semantic_type': getattr(self._tracking_state, 'next_semantic_type', None),
                    'next_semantic_label': getattr(self._tracking_state, 'next_semantic_label', None),
                    'expected_control_type': getattr(self._tracking_state, 'expected_control_type', None),
                    'route_queue': list(getattr(self._tracking_state, 'route_queue', []) or []),
                    'destination_label': getattr(self._tracking_state, 'destination_label', None),
                    'relocalization_mode': getattr(self._tracking_state, 'relocalization_mode', 'map_match'),
                    'last_relocalization_source': getattr(self._tracking_state, 'last_relocalization_source', 'map_match'),
                    'last_relocalization_error_m': round(float(getattr(self._tracking_state, 'last_relocalization_error_m', 0.0) or 0.0), 5),
                    'destination_node_id': getattr(self._tracking_state, 'destination_node_id', None),
                    'route_progress': round(float(getattr(self._tracking_state, 'route_progress', 0.0) or 0.0), 5),
                    'route_completed': bool(getattr(self._tracking_state, 'route_completed', False)),
                })
            if self._last_local_perception_status:
                status['local_ai_ready'] = bool(self._last_local_perception_status.get('model_ready', False))
                status['local_ai_infer_ms'] = self._last_local_perception_status.get('inference_time_ms')
                status['local_ai_fps'] = self._last_local_perception_status.get('fps')
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

    def _create_hsv_mask(self, hsv, h_min, h_max, s_min, s_max, v_min, v_max):
        """Create an HSV mask, supporting hue ranges that wrap around 0."""
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        if h_min <= h_max:
            return cv2.inRange(hsv, lower, upper)

        lower_wrap = np.array([0, s_min, v_min])
        upper_wrap = np.array([h_max, s_max, v_max])
        lower_tail = np.array([h_min, s_min, v_min])
        upper_tail = np.array([179, s_max, v_max])

        return cv2.bitwise_or(
            cv2.inRange(hsv, lower_wrap, upper_wrap),
            cv2.inRange(hsv, lower_tail, upper_tail)
        )

    def _filter_line_like_components(self, mask):
        """Keep only connected components that resemble elongated lane markings."""
        if not self.use_component_filter:
            return mask

        contours_data = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]
        filtered = np.zeros_like(mask)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.component_min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if max(w, h) < self.component_min_height:
                continue

            rect = cv2.minAreaRect(contour)
            rect_w, rect_h = rect[1]
            long_side = max(rect_w, rect_h)
            short_side = max(min(rect_w, rect_h), 1.0)
            if long_side < self.component_min_height:
                continue

            aspect_ratio = long_side / short_side
            if aspect_ratio < self.component_min_aspect_ratio:
                continue

            vx, vy, _, _ = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = abs(math.degrees(math.atan2(float(vy), float(vx))))
            if angle < self.component_min_angle:
                continue

            cv2.drawContours(filtered, [contour], -1, 255, -1)

        return filtered

    def _adaptive_local_threshold(self, gray_image):
        """Local adaptive thresholding, as suggested by classic layered lane pipelines."""
        block_size = max(3, int(self.local_adaptive_block_size))
        if block_size % 2 == 0:
            block_size += 1

        adaptive_mask = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self.local_adaptive_c
        )
        return self._filter_line_like_components(adaptive_mask)

    def _hough_lines_by_half(self, canny):
        """Run Hough independently on each half of the image and keep strongest segments."""
        height, width = canny.shape[:2]
        split_x = width // 2
        selected_lines = []

        for x_start, x_end in ((0, split_x), (split_x, width)):
            half = canny[:, x_start:x_end]
            half_lines = cv2.HoughLinesP(
                half, 1, np.pi / 180,
                self.hough_threshold,
                minLineLength=self.hough_min_line_length,
                maxLineGap=self.hough_max_line_gap
            )
            if half_lines is None:
                continue

            scored_lines = []
            for line in half_lines:
                x1, y1, x2, y2 = line[0]
                x1 += x_start
                x2 += x_start
                length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
                scored_lines.append(
                    (length_sq, np.array([[x1, y1, x2, y2]], dtype=np.int32))
                )

            scored_lines.sort(key=lambda item: item[0], reverse=True)
            limit = max(1, int(self.hough_max_lines_per_side))
            selected_lines.extend(line for _, line in scored_lines[:limit])

        if len(selected_lines) == 0:
            return None

        return np.array(selected_lines, dtype=np.int32)

    def _preprocess_frame(self, frame):
        """Apply all preprocessing steps to make detection robust to lighting changes.
        
        Returns:
            preprocessed: The preprocessed frame
            debug_info: Dictionary with intermediate results for debugging
        """
        debug_info = {}
        needs_debug = self._needs_debug
        
        # Step 1: Apply CLAHE for lighting normalization
        if self.use_clahe:
            preprocessed = self._apply_clahe(frame)
            if needs_debug:
                debug_info['clahe'] = preprocessed.copy()
        else:
            preprocessed = frame.copy()
        
        # Step 2: Apply brightness/contrast adjustment
        preprocessed = cv2.convertScaleAbs(preprocessed, alpha=self.contrast, beta=self.brightness)
        if needs_debug:
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
        needs_debug = self._needs_debug
        
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)
        if needs_debug:
            debug_info['hsv'] = hsv.copy()
        
        # Standard HSV-based white detection
        white_mask_hsv = cv2.inRange(hsv, self.white_lower, self.white_upper)
        if needs_debug:
            debug_info['white_hsv'] = white_mask_hsv.copy()

        # Yellow detection (usually more stable across lighting)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        if needs_debug:
            debug_info['yellow'] = yellow_mask.copy()
        
        # Start with HSV-based masks
        combined_mask = cv2.bitwise_or(white_mask_hsv, yellow_mask)
        
        # Add adaptive white detection if enabled
        if self.use_adaptive_white:
            adaptive_white_mask, adaptive_threshold = self._adaptive_white_detection(preprocessed)
            if needs_debug:
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
                if needs_debug:
                    debug_info['gradient'] = gradient_mask.copy()
                debug_info['gradient_used'] = True
                
                # Combine gradient mask with color masks
                combined_mask = cv2.bitwise_or(combined_mask, gradient_mask)
            else:
                debug_info['gradient_used'] = False
        
        if needs_debug:
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
                if self._is_window_enabled("ai_analysis"):
                    self._show_preview_window("AI Analysis - LSTR", ai_analysis_frame)
                
                self._store_debug_image('lstr', ai_analysis_frame)
                return None, None, None, None
            
            # Get lane center
            lane_center = self.lstr_detector.get_lane_center(width, y_position_ratio=1.0 - self.lookahead)
            
            if lane_center is None:
                if self._is_window_enabled("ai_analysis"):
                    self._show_preview_window("AI Analysis - LSTR", ai_analysis_frame)
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
            if self._is_window_enabled("ai_analysis"):
                self._show_preview_window("AI Analysis - LSTR", debug_frame)
            
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
        
        if self._is_window_enabled("hybrid_fusion"):
            self._show_preview_window("HYBRID Fusion", debug_frame)
            if final_steering is not None:
                self._debug_log(
                    "hybrid_fusion",
                    f"\033[1;97m[ HYBRID ] :\033[0m \033[1;96m{source}\033[0m - "
                    f"Steer: {final_steering:.1f}° (OpenCV:{opencv_steering}, LSTR:{lstr_steering})",
                )
        
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

        curve_transition_active = self._curve_state in ("ENTERING", "EXITING")
        single_line_curve_commit = bool(
            isinstance(debug_info, dict) and (
                debug_info.get("single_line_max_curve_steer") or
                debug_info.get("single_line_curve_priority") or
                debug_info.get("single_line_transition_blend")
            )
        )
        reversal_min_deg = 2.0
        previous_steer = (
            float(self._last_good_steering)
            if self._last_good_steering is not None else None
        )
        current_steer = float(steering_angle) if steering_angle is not None else None
        steer_reversal = bool(
            previous_steer is not None and
            current_steer is not None and (
                (previous_steer >= reversal_min_deg and current_steer <= -reversal_min_deg) or
                (previous_steer <= -reversal_min_deg and current_steer >= reversal_min_deg)
            )
        )

        # Check 1: Too many Hough lines = reflections/glare
        all_lines = debug_info.get('all_lines')
        if all_lines is not None and len(all_lines) > self.noise_max_hough_lines:
            merged_left = len(debug_info.get('left_lines', []))
            merged_right = len(debug_info.get('right_lines', []))
            merged_total = merged_left + merged_right
            # Let true single-line detections pass through even if raw Hough found
            # many short segments. Otherwise the 1-line controller never runs.
            if not (num_lines == 1 and merged_total <= 6):
                return True, f"hough_overflow({len(all_lines)}>{self.noise_max_hough_lines})"

        # Check 2: Sudden error jump (only if we have a previous reference)
        if error is not None and self._last_good_error is not None:
            error_jump = abs(error - self._last_good_error)
            if error_jump > self.noise_max_error_jump_px and self._noise_reject_count < self.noise_max_reject_frames:
                # During curve entry/exit the target lane center can legitimately jump
                # as the controller switches between straight and curve geometry.
                if curve_transition_active or (num_lines == 1 and single_line_curve_commit) or steer_reversal:
                    return False, ""
                return True, f"error_jump({error_jump:.0f}px>{self.noise_max_error_jump_px})"

        # Check 3: Sudden steering jump
        if steering_angle is not None and self._last_good_steering is not None:
            steer_jump = abs(steering_angle - self._last_good_steering)
            if steer_jump > self.noise_max_steer_jump_deg and self._noise_reject_count < self.noise_max_reject_frames:
                # Exception: don't reject if we're transitioning states (curve entry/exit)
                if curve_transition_active or (num_lines == 1 and single_line_curve_commit):
                    return False, ""
                # Exception: don't reject when steering DIRECTION reverses.
                # A reversal (e.g., last_good=+22° → new=-3°) indicates a legitimate
                # mode transition (2-line heading-dominated → 1-line crosstrack-dominated),
                # not sensor noise. Rejecting it keeps the car going the wrong way for 3
                # frames, allowing dangerous lateral accumulation.
                #
                # Use inclusive thresholds here: curve-priority logic frequently clamps
                # the previous command to exactly +/-2.0°, and strict comparisons would
                # miss the sign change and latch the car into the stale steering command.
                if steer_reversal:
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
                    self.pid.reset()
                    self.stanley.reset()
                    self.lateral_mpc.reset()
                    print(f"\033[1;97m[ Recovery ] :\033[0m \033[1;92mCOMPLETE\033[0m - Resuming normal operation")
                    return steering_angle, speed, False
                # Keep max curve steer while briefly stopped before resuming
                return self._recovery_curve_sign * self.max_steering, 0, True

            return 0, 0, True

        # === DETECT IF RECOVERY IS NEEDED ===
        # Guard 1: Prefer recovery in confirmed curves. In AI_LOCAL, a sustained
        # single-side track is also enough evidence that the car is in a turn or
        # already off the intended line, even if the legacy curve FSM is idle.
        in_curve_context = self._curve_state in ("ENTERING", "IN_CURVE")
        if not in_curve_context and self._is_ai_local_active():
            in_curve_context = getattr(self, 'last_seen_side', None) in ("left", "right")

        if not in_curve_context:
            self._max_steer_consecutive = 0
            self._error_at_max_steer_start = 0.0
            return steering_angle, speed, False

        # Guard 2: Don't trigger recovery when steering AWAY from the curve.
        # This happens when the car went off the INNER line (turned too tight).
        # Steering opposite to curve direction = correcting from inside, NOT stuck.
        if steering_angle is not None and self._curve_direction != 0:
            steering_sign = 1 if steering_angle > 0 else -1
            if steering_sign != self._curve_direction:
                # Steering opposes curve direction → inner line correction, not stuck
                self._max_steer_consecutive = 0
                self._error_at_max_steer_start = 0.0
                return steering_angle, speed, False

        # Guard 3: Don't count frames where the noise filter just rejected data
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

    def _compute_single_line_shadow_for_calib(
        self,
        side_lines,
        img_h,
        img_w,
        reference_y,
        two_line_error_px,
        left_ref_x,
        right_ref_x,
    ):
        """Compute single-line error independently for each line when two lines are visible.

        Never affects control output.  Used only for calibration diagnostics: knowing the
        "correct" two-line error lets us measure each single-line estimator's bias and
        tune parameters so they match when the car transitions to one-line mode.

        For each side we compute two variants:
          - physical: the same projection used in single_line_physical mode
          - midpoint_ref_fallback: virtual boundary from lane_width (stable for high angles)
        """
        _ref_lw_px = max(1.0, float(right_ref_x) - float(left_ref_x))
        _px_per_cm_now = _ref_lw_px / max(1e-3, float(self.lane_width_cm))

        # Temporarily expose current frame's px_per_cm so _compute_single_line_error
        # uses the right scale (the stored attribute is from the previous two-line frame).
        _saved_ppc = getattr(self, '_last_two_line_ref_px_per_cm', None)
        _saved_proj_debug = getattr(self, '_last_single_line_projection_debug', None)

        result = {}
        try:
            self._last_two_line_ref_px_per_cm = _px_per_cm_now

            for side in ('left', 'right'):
                line = side_lines.get(side)
                if not isinstance(line, np.ndarray) or line.ndim != 2 or line.shape[1] < 4:
                    continue

                _heading_raw = self._compute_raw_line_heading(line)
                _abs_heading_deg = abs(math.degrees(_heading_raw))
                _heading_cutoff = float(getattr(self, 'single_line_physical_heading_cutoff_deg', 65.0))

                entry = {
                    'angle_from_vertical_deg': round(float(_abs_heading_deg), 2),
                    'heading_cutoff_deg': float(_heading_cutoff),
                    'would_skip_physical': bool(_abs_heading_deg >= _heading_cutoff),
                    'px_per_cm': round(float(_px_per_cm_now), 4),
                    'ref_lw_px': round(float(_ref_lw_px), 2),
                    'left_ref_x': round(float(left_ref_x), 2),
                    'right_ref_x': round(float(right_ref_x), 2),
                    'two_line_error_px': round(float(two_line_error_px), 2),
                }

                # --- Physical projection ---
                _ref_line_x_at_y = self._line_x_at_y(line, reference_y)
                try:
                    sl_err_phys, _ = self._compute_single_line_error(
                        line, side, img_h, img_w,
                        prefer_center=False,
                        physical_projection=True,
                        reference_y_override=reference_y,
                    )
                    phys_debug = dict(getattr(self, '_last_single_line_projection_debug', {}) or {})
                    entry['physical'] = {
                        'error_px': round(float(sl_err_phys), 2),
                        'delta_vs_two_line_px': round(float(sl_err_phys) - float(two_line_error_px), 2),
                        'ref_line_x_at_y': round(float(_ref_line_x_at_y), 2) if _ref_line_x_at_y is not None else None,
                        'target_x': round(float(phys_debug.get('single_line_target_x', 0.0)), 2),
                        'camera_shift_px': round(float(phys_debug.get('single_line_camera_shift_px', 0.0)), 3),
                        'heading_reliability': round(float(phys_debug.get('single_line_heading_reliability', 0.0)), 3),
                        'curve_strength': round(float(phys_debug.get('single_line_curve_strength', 0.0)), 3),
                        'outer_confirmed': bool(phys_debug.get('single_line_outer_confirmed', False)),
                        'outer_bias_px': round(float(phys_debug.get('single_line_outer_bias_px', 0.0)), 3),
                        'mode': str(phys_debug.get('single_line_mode', '')),
                    }
                except Exception as _exc:
                    entry['physical'] = {'error': str(_exc)}

                # --- Midpoint-ref fallback (virtual boundary using exact current lane width) ---
                ref_line_x_at_y = self._line_x_at_y(line, reference_y)
                if ref_line_x_at_y is not None:
                    if side == 'left':
                        virt_x = min(float(img_w - 1), float(ref_line_x_at_y) + _ref_lw_px)
                    else:
                        virt_x = max(0.0, float(ref_line_x_at_y) - _ref_lw_px)
                    virt_mid = (float(ref_line_x_at_y) + virt_x) / 2.0
                    sl_err_virt = virt_mid - (img_w / 2.0)
                    entry['midpoint_ref_fallback'] = {
                        'error_px': round(float(sl_err_virt), 2),
                        'delta_vs_two_line_px': round(float(sl_err_virt) - float(two_line_error_px), 2),
                        'ref_line_x': round(float(ref_line_x_at_y), 2),
                        'virtual_other_x': round(float(virt_x), 2),
                        'virtual_midpoint_x': round(float(virt_mid), 2),
                    }
                else:
                    entry['midpoint_ref_fallback'] = {'error': 'ref_line_x_at_y is None'}

                result[side] = entry
        finally:
            self._last_two_line_ref_px_per_cm = _saved_ppc
            self._last_single_line_projection_debug = _saved_proj_debug

        return result if result else None

    def _should_apply_single_line_protective_stop(self, local_mask_guidance, steering_angle, error):
        """Stop forward motion if 1-line tracking stays saturated for too long.

        Runtime instances keep this feature behind config because aggressive 1-line
        protective stops are highly track-dependent. Bare test doubles created with
        ``__new__`` default to enabled so the legacy unit tests can still exercise it.
        """
        if not bool(getattr(self, 'use_single_line_protective_stop', True)):
            self._single_line_stop_consecutive = 0
            return False

        if not isinstance(local_mask_guidance, dict):
            self._single_line_stop_consecutive = 0
            return False

        guidance_mode = str(local_mask_guidance.get('guidance_mode', '') or '')
        detected_sides = tuple(local_mask_guidance.get('detected_sides', ()) or ())
        if not guidance_mode.startswith('single_line') or len(detected_sides) != 1:
            self._single_line_stop_consecutive = 0
            return False

        if steering_angle is None or error is None:
            self._single_line_stop_consecutive = 0
            return False

        max_steering = max(1.0, float(getattr(self, 'max_steering', 25.0) or 25.0))
        raw_error_px = float(local_mask_guidance.get('raw_error_px', error) or error)
        error_limit_px = float(getattr(self, 'max_error_px', 40.0) or 40.0) * float(
            getattr(self, 'single_line_stop_error_scale', 1.5) or 1.5
        )
        saturated = abs(float(steering_angle)) >= (max_steering - 0.5)
        error_large = abs(raw_error_px) >= error_limit_px

        if saturated and error_large:
            self._single_line_stop_consecutive += 1
        else:
            self._single_line_stop_consecutive = 0

        return self._single_line_stop_consecutive >= max(
            1, int(getattr(self, 'single_line_stop_max_steer_frames', 6) or 6)
        )

    # ============= STANLEY CONTROLLER HELPERS =============

    def _compute_heading_error(self, avg_left, avg_right):
        """Estimate heading error from lane line geometry.

        When both lines are detected, the average of their angles relative to
        vertical gives the heading error directly:
        - Aligned car: left angle = -α, right angle = +α → average = 0
        - Car heading left: both angles shift positive → average > 0
        - Car heading right: both angles shift negative → average < 0

        Returns:
            float: Heading error in radians. Positive = car heading left of lane.
        """
        angles = []
        for line in [avg_left, avg_right]:
            if line is not None:
                x1, y1, x2, y2 = line[0]
                if y1 < y2:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                dx = x2 - x1
                dy = y1 - y2
                if dy > 1:
                    angles.append(math.atan2(dx, dy))

        if len(angles) == 2:
            heading = (angles[0] + angles[1]) / 2.0
        elif len(angles) == 1:
            heading = angles[0] * 0.5
        else:
            heading = 0.0

        self._heading_error = heading
        return heading

    def _compute_raw_line_heading(self, line):
        """Compute raw heading angle from a single line (radians)."""
        if line is None:
            return 0.0
        x1, y1, x2, y2 = line[0]
        if y1 < y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx = x2 - x1
        dy_up = y1 - y2
        if dy_up > 1:
            return math.atan2(dx, dy_up)
        return 0.0

    def _read_sensor_data(self):
        """Read IMU and speed feedback from Nucleo for Stanley controller.

        Runs every cycle to keep sensor data fresh. All reads are non-blocking.
        If data is unavailable, previous values are retained.
        """
        imu_msg = self.imuDataSubscriber.receive()
        if imu_msg is not None:
            try:
                data = ast.literal_eval(imu_msg)
                # BNO055 compass convention (CW-positive) → math convention (CCW-positive)
                new_yaw = -float(data.get('yaw', 0))
                current_time = time.time()
                if self._last_imu_time is not None:
                    dt = current_time - self._last_imu_time
                    if dt > 0.001:
                        self._yaw_rate = math.radians(new_yaw - self._last_yaw) / dt
                self._last_yaw = new_yaw
                self._last_imu_roll = float(data.get('roll', self._last_imu_roll))
                self._last_imu_pitch = float(data.get('pitch', self._last_imu_pitch))
                self._last_imu_time = current_time
            except (ValueError, SyntaxError):
                pass

        speed_msg = self.currentSpeedSubscriber.receive()
        if speed_msg is not None:
            try:
                self._measured_speed = float(speed_msg)
                self._last_speed_time = time.time()
            except (ValueError, TypeError):
                pass

        steer_msg = self.currentSteerSubscriber.receive()
        self._measured_steer_delta = 0.0
        if steer_msg is not None:
            try:
                new_steer = float(steer_msg)
                self._measured_steer_delta = self._measured_steer - new_steer
                self._measured_steer = new_steer
            except (ValueError, TypeError):
                pass

        # Dashboard commands — always drained so the pipe stays clean; values used by manual log
        cmd_speed_msg = self.speedMotorSubscriber.receive()
        if cmd_speed_msg is not None:
            try:
                self._last_cmd_speed = float(cmd_speed_msg)
            except (ValueError, TypeError):
                pass
        cmd_steer_msg = self.steerMotorSubscriber.receive()
        if cmd_steer_msg is not None:
            try:
                self._last_cmd_steer = float(cmd_steer_msg)
            except (ValueError, TypeError):
                pass

    def _get_stanley_curve_reference(self, avg_left=None, avg_right=None, img_h=None, img_w=None, fallback_curve=None):
        """Return the best available curve estimate for ψ_ss and reference yaw rate."""
        if isinstance(fallback_curve, dict):
            radius_cm = float(fallback_curve.get('curve_radius_cm', 0.0) or 0.0)
            turn_direction = int(fallback_curve.get('turn_direction', 0) or 0)
            confidence = float(fallback_curve.get('confidence', 0.0) or 0.0)
            if radius_cm > 0.0 and turn_direction != 0:
                return {
                    'curve_radius_cm': radius_cm,
                    'turn_direction': turn_direction,
                    'confidence': confidence,
                    'source': fallback_curve.get('source', 'fallback'),
                }

        if avg_left is not None and avg_right is not None and img_h is not None and img_w is not None:
            curve = self._estimate_curvature_from_lines(avg_left, avg_right, img_h, img_w)
            turn_direction = int(curve.get('turn_direction', 0) or 0)
            best_radius, confidence, source = self._get_best_curve_radius(
                vp_radius=curve.get('curve_radius_cm', 0.0),
                vp_confidence=curve.get('confidence', 0.0),
            )
            if turn_direction == 0 and self._curve_direction != 0:
                turn_direction = self._curve_direction
            if best_radius > 0.0 and turn_direction != 0:
                return {
                    'curve_radius_cm': best_radius,
                    'turn_direction': turn_direction,
                    'confidence': confidence,
                    'source': source,
                }

        if self._curve_direction != 0 and self._curve_radius_estimate > 0.0 and self._curve_confidence > 0.0:
            return {
                'curve_radius_cm': float(self._curve_radius_estimate),
                'turn_direction': int(self._curve_direction),
                'confidence': float(self._curve_confidence),
                'source': 'curve_fsm',
            }

        return None

    def _resolve_px_per_cm(self, lane_width_px=None, img_w=None):
        """Estimate pixel density using lane width calibration, cache, or frame fallback."""
        if lane_width_px is not None:
            try:
                lane_width_px = float(lane_width_px)
                if lane_width_px > 1.0 and float(self.lane_width_cm) > 1e-3:
                    return lane_width_px / float(self.lane_width_cm)
            except (TypeError, ValueError):
                pass

        # Two-line reference scale: computed as _ref_lw_px / lane_width_cm where
        # _ref_lw_px = right_x - left_x at reference_y in the AI local overlay.
        # This is the most accurate source because it uses the actual visible lane
        # width at the exact same depth used for D_left/D_right measurements.
        two_line_ppc = float(getattr(self, '_last_two_line_ref_px_per_cm', 0.0) or 0.0)
        if two_line_ppc > 0.5:
            return two_line_ppc

        # AI mask lane width: most reliable source because it uses actual pixel-level
        # segmentation of the lane lines, independent of Hough-line fitting noise.
        # Prioritized above the OpenCV Hough cache to match the AI overlay's coordinate
        # system, which the user confirmed shows the correct centered lane geometry.
        ai_lane_width_px = float(getattr(self, '_last_local_ai_lane_width_px', 0.0) or 0.0)
        if ai_lane_width_px > 1.0 and float(self.lane_width_cm) > 1e-3:
            return ai_lane_width_px / float(self.lane_width_cm)

        # OpenCV Hough-line cache (lower priority than AI mask but still useful when
        # the AI has not produced a fresh lane width in this detection cycle).
        cached_px_per_cm = float(getattr(self, '_last_px_per_cm', 0.0) or 0.0)
        if cached_px_per_cm > 0.5:
            return cached_px_per_cm

        # BEV-derived scale: updated every frame in _estimate_curvature_from_lines
        # when two lane lines are detected. More accurate than the fixed geometric
        # estimate because it uses actual projected lane positions in BEV space.
        bev_cm_per_px = float(getattr(self, 'bev_cm_per_px', 0.0) or 0.0)
        if bev_cm_per_px > 0.0:
            bev_px_per_cm = 1.0 / bev_cm_per_px
            if bev_px_per_cm > 0.5:
                return bev_px_per_cm

        if img_w is not None and float(img_w) > 1.0 and float(self.lane_width_cm) > 1e-3:
            return (float(img_w) * 0.45) / float(self.lane_width_cm)

        return 0.0

    def _convert_error_px_to_m(self, error_px, lane_width_px=None, img_w=None):
        """Convert lateral error from pixels to meters using current lane scale."""
        px_per_cm = max(1e-6, float(self._resolve_px_per_cm(lane_width_px=lane_width_px, img_w=img_w)))
        error_cm = float(error_px) / px_per_cm
        error_m = error_cm / 100.0
        return error_m, px_per_cm

    def _reset_local_ai_context(self):
        self._local_ai_heading_hint_rad = 0.0
        self._local_ai_heading_hint_confidence = 0.0
        self._local_ai_heading_hint_source = "none"
        self._local_ai_road_type_class = "unknown"
        self._local_ai_road_type_confidence = 0.0
        self._local_ai_lead_distance_class = "none"
        self._local_ai_lead_distance_confidence = 0.0
        self._local_ai_lead_distance_area = 0.0

    def _update_local_ai_context(self, payload, debug_info=None, result_age=None, stale_grace=False):
        """Extract optional high-level AI cues from the local lane payload."""
        self._reset_local_ai_context()
        if not isinstance(payload, dict):
            return

        def _safe_float(name, default=0.0):
            try:
                return float(payload.get(name, default) or default)
            except (TypeError, ValueError):
                return float(default)

        age_factor = 1.0
        if result_age is not None:
            age_limit = max(
                1e-6,
                float(getattr(self, 'local_ai_max_result_age', 0.35) or 0.35) +
                max(0.0, float(getattr(self, 'local_ai_stale_grace_s', 0.20) or 0.20))
            )
            age_factor = max(0.0, min(1.0, 1.0 - (float(result_age) / age_limit)))
            if stale_grace:
                age_factor *= 0.6

        heading_hint_rad = _safe_float("heading_hint_rad", 0.0)
        heading_hint_conf = _safe_float("heading_hint_confidence", 0.0)
        heading_hint_conf = max(0.0, min(1.0, heading_hint_conf * age_factor))
        heading_hint_source = str(payload.get("heading_hint_source", "none") or "none")

        road_type_class = str(payload.get("road_type_class", "unknown") or "unknown").strip().lower()
        road_type_conf = _safe_float("road_type_confidence", 0.0)
        road_type_conf = max(0.0, min(1.0, road_type_conf * age_factor))

        lead_distance_class = str(payload.get("lead_distance_class", "none") or "none").strip().lower()
        lead_distance_conf = _safe_float("lead_distance_confidence", 0.0)
        lead_distance_conf = max(0.0, min(1.0, lead_distance_conf * age_factor))
        lead_distance_area = max(0.0, _safe_float("lead_distance_area", 0.0))

        self._local_ai_heading_hint_rad = float(heading_hint_rad)
        self._local_ai_heading_hint_confidence = float(heading_hint_conf)
        self._local_ai_heading_hint_source = heading_hint_source
        self._local_ai_road_type_class = road_type_class
        self._local_ai_road_type_confidence = float(road_type_conf)
        self._local_ai_lead_distance_class = lead_distance_class
        self._local_ai_lead_distance_confidence = float(lead_distance_conf)
        self._local_ai_lead_distance_area = float(lead_distance_area)

        if isinstance(debug_info, dict):
            debug_info["heading_hint_rad"] = float(self._local_ai_heading_hint_rad)
            debug_info["heading_hint_confidence"] = float(self._local_ai_heading_hint_confidence)
            debug_info["heading_hint_source"] = self._local_ai_heading_hint_source
            debug_info["road_type_class"] = self._local_ai_road_type_class
            debug_info["road_type_confidence"] = float(self._local_ai_road_type_confidence)
            debug_info["lead_distance_class"] = self._local_ai_lead_distance_class
            debug_info["lead_distance_confidence"] = float(self._local_ai_lead_distance_confidence)
            debug_info["lead_distance_area"] = float(self._local_ai_lead_distance_area)

    def _fuse_heading_with_local_hint(self, heading_rad, debug_info=None, mode="normal"):
        """Blend geometric heading with AI heading hint to reduce 1-line spikes."""
        try:
            base_heading = float(heading_rad)
        except (TypeError, ValueError):
            base_heading = 0.0

        hint_conf = float(getattr(self, "_local_ai_heading_hint_confidence", 0.0) or 0.0)
        if hint_conf <= 0.02:
            return base_heading

        hint_heading = float(getattr(self, "_local_ai_heading_hint_rad", 0.0) or 0.0)
        mode_name = str(mode or "normal")
        if mode_name == "single_line":
            max_weight = 0.65
            max_delta = math.radians(24.0)
        elif mode_name == "stale":
            max_weight = 0.75
            max_delta = math.radians(20.0)
        else:
            max_weight = 0.35
            max_delta = math.radians(14.0)

        fusion_weight = max(0.0, min(max_weight, hint_conf))

        # In single-line mode, scale the fusion weight by the local heading reliability.
        # When the visible line is nearly horizontal (e.g. very sharp curve), the line
        # angle cannot be reliably converted to a heading — single_line_heading_reliability
        # captures this and will be 0 for near-horizontal lines.  A hint derived from such
        # a line should not dominate the controller even if its confidence is otherwise high.
        if mode_name == "single_line" and isinstance(debug_info, dict):
            local_reliability = float(debug_info.get("single_line_heading_reliability", 1.0) or 0.0)
            fusion_weight *= local_reliability
            # When the line is nearly horizontal (reliability≈0) but we are in a curve and the
            # AI hint is confident, apply a minimum fusion weight so the hint is not fully ignored.
            if local_reliability < 0.1 and fusion_weight < 0.05:
                in_curve = str(getattr(self, '_curve_state', 'STRAIGHT')) in ('ENTERING', 'IN_CURVE')
                if in_curve and hint_conf > 0.3:
                    fusion_weight = hint_conf * 0.25
            if isinstance(debug_info, dict):
                debug_info["heading_fusion_local_reliability"] = round(local_reliability, 3)

        delta = hint_heading - base_heading
        delta = max(-max_delta, min(max_delta, delta))
        fused_heading = base_heading + fusion_weight * delta

        if isinstance(debug_info, dict):
            debug_info["heading_fusion_mode"] = mode_name
            debug_info["heading_fusion_weight"] = float(fusion_weight)
            debug_info["heading_fusion_hint_deg"] = float(math.degrees(hint_heading))
            debug_info["heading_fusion_base_deg"] = float(math.degrees(base_heading))
            debug_info["heading_fusion_out_deg"] = float(math.degrees(fused_heading))
        return fused_heading

    def _apply_stanley_road_type_schedule(self):
        """Schedule Stanley gains with road-type cue (paper-style multi-task fusion)."""
        base_k = float(getattr(self, "_stanley_base_k", self.stanley_k))
        base_k_soft = float(getattr(self, "_stanley_base_k_soft", self.stanley_k_soft))

        road_type = str(getattr(self, "_local_ai_road_type_class", "unknown") or "unknown")
        confidence = float(getattr(self, "_local_ai_road_type_confidence", 0.0) or 0.0)
        confidence = max(0.0, min(1.0, confidence))
        if confidence < 0.15:
            road_type = "unknown"

        target_k = base_k
        target_k_soft = base_k_soft
        if road_type == "highway":
            target_k = base_k * 0.85
            target_k_soft = base_k_soft * 1.20
        elif road_type == "roundabout":
            target_k = base_k * 1.20
            target_k_soft = base_k_soft * 0.90
        elif road_type == "urban":
            target_k = base_k * 1.05
            target_k_soft = base_k_soft * 0.95

        alpha = max(0.05, min(1.0, float(getattr(self, "_stanley_sched_alpha", 0.30))))
        self._stanley_sched_k = (1.0 - alpha) * float(self._stanley_sched_k) + alpha * float(target_k)
        self._stanley_sched_k_soft = (
            (1.0 - alpha) * float(self._stanley_sched_k_soft) + alpha * float(target_k_soft)
        )
        self.stanley.k = float(self._stanley_sched_k)
        self.stanley.k_soft = max(0.1, float(self._stanley_sched_k_soft))

    def _apply_lead_distance_speed_cap(self, speed, speed_cap=None, debug_info=None):
        """Apply conservative longitudinal cap from lead-distance class."""
        lead_class = str(getattr(self, "_local_ai_lead_distance_class", "none") or "none")
        lead_conf = float(getattr(self, "_local_ai_lead_distance_confidence", 0.0) or 0.0)
        if lead_conf < 0.20:
            return speed, speed_cap

        cap = None
        if lead_class == "near":
            cap = float(self.min_speed)
        elif lead_class == "medium":
            cap = float(self.min_speed) + 2.0
        elif lead_class == "far":
            cap = float(self.base_speed)

        if cap is None:
            return speed, speed_cap

        if speed_cap is None:
            new_cap = cap
        else:
            new_cap = min(float(speed_cap), cap)
        new_speed = min(float(speed), float(new_cap))

        if isinstance(debug_info, dict):
            debug_info["lead_speed_cap"] = float(new_cap)
        return new_speed, new_cap

    def _update_dynamic_tracking_metrics(self, error_px=None):
        """Update dynamic moving-average errors inspired by theta dMAE / offset dMA."""
        heading_abs_deg = abs(math.degrees(float(getattr(self, "_heading_error", 0.0) or 0.0)))
        self._theta_abs_error_history.append(float(heading_abs_deg))

        offset_abs_m = None
        if self._should_use_stanley_controller():
            offset_abs_m = abs(float(getattr(self, "_stanley_last_error_m", 0.0) or 0.0))
        elif error_px is not None:
            try:
                px_per_cm = max(1e-6, float(self._resolve_px_per_cm()))
                offset_abs_m = abs(float(error_px)) / px_per_cm / 100.0
            except (TypeError, ValueError):
                offset_abs_m = None
        if offset_abs_m is not None:
            self._offset_abs_error_history.append(float(offset_abs_m))

        if self._theta_abs_error_history:
            self._theta_dmae_deg = float(sum(self._theta_abs_error_history) / len(self._theta_abs_error_history))
        else:
            self._theta_dmae_deg = 0.0
        if self._offset_abs_error_history:
            self._offset_dma_m = float(sum(self._offset_abs_error_history) / len(self._offset_abs_error_history))
        else:
            self._offset_dma_m = 0.0

    def _resolve_stanley_speed_mps(self, speed_value, speed_cap=None):
        """Resolve Stanley speed in m/s, preferring fresh measured speed from Nucleo."""
        command_speed = abs(float(speed_value))
        if speed_cap is not None:
            command_speed = min(command_speed, abs(float(speed_cap)))
        command_speed_mps = command_speed * float(self.stanley_speed_to_mps)

        measured_speed_mps = abs(float(getattr(self, '_measured_speed', 0.0) or 0.0)) * float(
            self.stanley_measured_speed_to_mps
        )
        speed_age = None
        if getattr(self, '_last_speed_time', None) is not None:
            speed_age = time.time() - float(self._last_speed_time)

        use_measured = (
            bool(getattr(self, 'stanley_use_measured_speed', True)) and
            measured_speed_mps > 1e-4 and
            (speed_age is None or speed_age <= float(getattr(self, 'stanley_measured_speed_timeout', 0.4)))
        )

        if use_measured:
            return measured_speed_mps, "measured"
        return command_speed_mps, "command"

    def _get_stanley_dynamic_terms(self, speed_mps, curve_reference=None):
        """Compute ψ_ss, trajectory yaw rate and steering damping terms for Stanley.

        Args:
            speed_mps: Vehicle speed in m/s.
        """
        psi_ss = 0.0
        traj_yaw_rate = 0.0
        # _measured_steer_delta is in steer_x10 units (raw hardware delta, tenths of degrees).
        # Convert to actual degrees first, then to radians for the Stanley formula.
        # Without the /10.0, a 42.3° transition (stored as 423) would be treated as
        # 423 degrees → 7.38 rad → k_d_steer × 7.38 = 42.3° of extra steer, saturating Stanley.
        steer_damping_delta = math.radians(self._measured_steer_delta / 10.0)

        ref = curve_reference if isinstance(curve_reference, dict) else None
        if ref is not None:
            radius_cm = float(ref.get('curve_radius_cm', 0.0) or 0.0)
            turn_direction = int(ref.get('turn_direction', 0) or 0)
            confidence = float(ref.get('confidence', 0.0) or 0.0)
            if (
                radius_cm > 1.0 and
                turn_direction != 0 and
                confidence >= float(self.stanley_curve_min_confidence)
            ):
                speed_mps = max(0.0, float(speed_mps))
                radius_m = max(radius_cm / 100.0, 1e-3)
                traj_yaw_rate_mag = speed_mps / radius_m
                # Positive steering is RIGHT in this project, so the sign is
                # inverted from the turn_direction marker (1=right, -1=left).
                traj_yaw_rate = -turn_direction * traj_yaw_rate_mag
                psi_ss_mag = float(self.stanley_k_ag) * speed_mps * traj_yaw_rate_mag
                psi_ss_limit = math.radians(max(0.0, float(self.stanley_psi_ss_max_deg)))
                psi_ss = -turn_direction * min(psi_ss_mag, psi_ss_limit)

        self._stanley_last_psi_ss = psi_ss
        self._stanley_last_traj_yaw_rate = traj_yaw_rate
        self._stanley_last_steer_damping = steer_damping_delta
        return psi_ss, traj_yaw_rate, steer_damping_delta

    def _is_ai_local_active(self):
        """Return True when the current detection runtime is the local AI path."""
        current_mode = self._normalize_detection_mode(getattr(self, 'detection_mode', DetectionMode.OPENCV.value))
        return current_mode == DetectionMode.AI_LOCAL.value

    def _should_use_stanley_controller(self):
        """AI Local is always Stanley; other modes follow the dashboard toggle."""
        return self._is_ai_local_active() or bool(self.use_stanley)

    def _compute_lateral_control(
        self,
        error,
        heading,
        speed_value,
        curve_reference=None,
        speed_cap=None,
        lane_width_px=None,
        img_w=None,
        direct_error_m=None,
        direct_error_context=None,
    ):
        """Compute steering with the active controller policy for the current mode.

        For Stanley, inputs are converted to physical units:
        - crosstrack error: pixels -> meters (using lane-width scale), OR
          direct_error_m can be provided as a pre-computed physical error (meters)
          derived directly from measured distances to each lane line and the known
          lane_width_cm. When direct_error_m is supplied, _convert_error_px_to_m is
          skipped entirely, removing any dependence on cached px/cm values.
        - speed: controller units / measured Nucleo speed -> m/s
        """
        self._heading_error = float(heading)
        self._last_planner_route_blend = 0.0
        self._last_planner_route_blend_source = "none"

        ts = self._tracking_state
        nav_ctx = self._get_navigation_context()
        _route_active = bool(nav_ctx["route_active"])
        _planner_priority_active = bool(nav_ctx["planner_priority"])
        _track_heading_enabled = bool(getattr(_config, "TRACKING_USE_PATH_HEADING", True))
        _lane_measurement_reliable = bool(
            getattr(ts, "lane_measurement_reliable", False)
        ) if ts is not None else False
        direct_error_source = ""
        direct_error_side = None
        if isinstance(direct_error_context, dict):
            direct_error_source = str(direct_error_context.get("source", "") or "")
            _side_value = direct_error_context.get("side")
            if _side_value is not None:
                direct_error_side = str(_side_value)
        _allow_stopline_waypoint = self._should_allow_stopline_waypoint_guidance(
            direct_error_context=direct_error_context
        )

        # ── Waypoint-mode override (GPS-free tracking at precision zones) ─────
        # When threadTracking detects a STOPLINE / INTERSECTION node ahead it
        # sets waypoint_mode_active=True.  After the stop node, lane lines at
        # intersections are unreliable and must be ignored entirely; GPS
        # waypoints have full authority regardless of line visibility.
        _waypoint_mode_active = bool(nav_ctx["waypoint_mode_active"])
        if (
            ts is not None and
            getattr(ts, "initialized", False) and
            _waypoint_mode_active
        ):
            wp_error_m = float(getattr(ts, "error_m", 0.0))
            wp_heading_rad = float(getattr(ts, "heading_rad", 0.0))
            wp_speed_mps = float(getattr(ts, "speed_mps", 0.0))
            # Use commanded speed as fallback when tracking speed is stale
            if wp_speed_mps < 1e-4:
                wp_speed_mps, _ = self._resolve_stanley_speed_mps(
                    speed_value, speed_cap=speed_cap
                )
            if getattr(self, "use_lateral_mpc", False):
                wp_kappa = float(getattr(ts, 'path_kappa', 0.0))
                wp_kappa = max(min(wp_kappa, 0.3), -0.3)
                wp_psi_ss = (math.atan(wp_kappa * float(self.lateral_mpc.L))
                             if abs(wp_kappa) > 0.01 else 0.0)
                return self.lateral_mpc.compute(
                    wp_error_m, wp_heading_rad, wp_speed_mps,
                    steady_state_heading=wp_psi_ss,
                )
            return self.stanley.compute(
                wp_error_m, wp_heading_rad, wp_speed_mps,
                yaw_rate=self._yaw_rate,
                traj_yaw_rate=0.0,
                steady_state_heading=0.0,
                steer_damping_delta=0.0,
            )
        # ── End waypoint-mode override ─────────────────────────────────────────

        # ── Path-heading override (blind fallback only) ────────────────────────
        # Only replace heading with the IMU-vs-path-tangent estimate when lane
        # detection has completely failed (direct_error_m is None = no lines).
        #
        # Activating this override when lane lines ARE visible causes two bugs:
        #
        #   1. Two-line mode always passes heading=0.0 intentionally, because the
        #      visual heading is dominated by the car's yaw (body rotation) rather
        #      than actual road curvature.  Replacing 0.0 with ts.heading_rad
        #      (IMU body yaw) reintroduces exactly the yaw the caller removed.
        #
        #   2. The crosstrack augmentation then sums ts.heading_rad + crosstrack_term.
        #      When the IMU shows body yaw in the opposite direction to the required
        #      correction (e.g. car heading left while physically right of center),
        #      the two terms cancel → psi_mpc ≈ 0 → degenerate case where the MPC
        #      horizon sees no gradient to correct lateral error.
        #
        # GUARD: only activate once the IMU has sent at least one real message.
        # Without this guard, yaw stays at 0° (or track-start yaw) and
        # heading_rad = 0 - path_psi can be very large → MPC saturates.
        _imu_ready = getattr(ts, "imu_received", False) if ts is not None else False
        # IMPORTANT: some visible-lane modes (e.g. midpoint_ref_fallback in single-line AI_LOCAL)
        # intentionally disable physical direct_error_m. That does NOT mean we are blind.
        # Only fall back to route/tracking guidance when the caller did not provide any
        # visual lane context at all.
        if ts is not None and getattr(ts, "initialized", False) \
                and _track_heading_enabled and _imu_ready \
                and direct_error_m is None \
                and direct_error_source == "" \
                and (_allow_stopline_waypoint or _route_active):
            heading = float(ts.heading_rad)
            self._heading_error = heading
            direct_error_m = float(getattr(ts, "error_m", 0.0))
            direct_error_source = "blind_route"
        # ── End path-heading override ──────────────────────────────────────────

        if self._should_use_stanley_controller():
            self._apply_stanley_road_type_schedule()
            if direct_error_m is not None:
                # Physical error supplied directly: no pixel conversion or cache needed.
                error_m = float(direct_error_m)
                # Rate-limit: cap frame-to-frame jump at 10 cm to suppress single-line
                # extrapolation spikes (e.g. at curve entry when one line disappears).
                _prev_em = getattr(self, '_prev_direct_error_m', None)
                _prev_em_side = getattr(self, '_prev_direct_error_side', None)
                _skip_rate_limit = False
                if direct_error_source == "single_line":
                    # A real 1-line side switch (left -> right or right -> left) can
                    # legitimately invert the physical direct_error_m sign in one frame.
                    # Smoothing across that transition drags the old sign forward and
                    # makes the MPC unwind or countersteer mid-curve.
                    if (
                        direct_error_side is not None and
                        _prev_em_side is not None and
                        direct_error_side != _prev_em_side
                    ):
                        _skip_rate_limit = True
                    elif _prev_em is not None and (error_m * float(_prev_em) < 0.0):
                        _skip_rate_limit = True
                if _prev_em is not None and not _skip_rate_limit:
                    _max_step = 0.10  # 10 cm/frame max change
                    error_m = max(_prev_em - _max_step, min(_prev_em + _max_step, error_m))
                self._prev_direct_error_m = error_m
                self._prev_direct_error_side = (
                    direct_error_side if direct_error_source == "single_line" else None
                )
                # px_per_cm is still needed for the legacy deadband path; compute from
                # current-frame lane width if available, otherwise fall back to cache.
                if lane_width_px is not None and float(lane_width_px) > 1.0:
                    px_per_cm = float(lane_width_px) / max(1e-3, float(self.lane_width_cm))
                else:
                    px_per_cm = max(1e-6, float(self._resolve_px_per_cm(img_w=img_w)))
            else:
                error_m, px_per_cm = self._convert_error_px_to_m(
                    error, lane_width_px=lane_width_px, img_w=img_w
                )
            speed_mps, speed_source = self._resolve_stanley_speed_mps(
                speed_value, speed_cap=speed_cap
            )
            control_heading = float(heading)
            _in_curve = str(getattr(self, "_curve_state", "STRAIGHT")) != "STRAIGHT"
            if (
                ts is not None and
                getattr(ts, 'initialized', False) and
                _imu_ready and
                _planner_priority_active and
                direct_error_source in {"single_line", "blind_route"} and
                not _in_curve
            ):
                path_heading = float(getattr(ts, "heading_rad", control_heading))
                blend = 0.75 if direct_error_source == "single_line" else 1.0
                control_heading = ((1.0 - blend) * float(heading)) + (blend * path_heading)
                self._last_planner_route_blend = float(blend)
                self._last_planner_route_blend_source = str(direct_error_source)
            elif (
                ts is not None and
                getattr(ts, 'initialized', False) and
                _imu_ready and
                _planner_priority_active and
                direct_error_source == "two_line" and
                not _in_curve
            ):
                blend = self._planner_two_line_takeover_weight(nav_ctx)
                if blend > 1e-4:
                    path_heading = float(getattr(ts, "heading_rad", control_heading))
                    route_error_m = float(getattr(ts, "error_m", error_m))
                    route_error_cap_m = max(
                        0.02,
                        float(getattr(_config, "TRACKING_INTERSECTION_ROUTE_ERROR_CAP_M", 0.18) or 0.18),
                    )
                    route_error_m = max(-route_error_cap_m, min(route_error_cap_m, route_error_m))
                    control_heading = ((1.0 - blend) * float(control_heading)) + (blend * path_heading)
                    error_m = ((1.0 - blend) * float(error_m)) + (blend * route_error_m)
                    self._last_planner_route_blend = float(blend)
                    self._last_planner_route_blend_source = "two_line"

            # Preserve legacy px deadband behavior while using physical-state Stanley.
            if float(getattr(self, 'stanley_deadband_crosstrack_m', 0.0) or 0.0) > 0.0:
                crosstrack_deadband = float(self.stanley_deadband_crosstrack_m)
            else:
                crosstrack_deadband = (
                    float(getattr(self, 'stanley_deadband_crosstrack_px', 0.0) or 0.0) /
                    max(px_per_cm, 1e-6) / 100.0
                )
            self.stanley.crosstrack_deadband = max(0.0, float(crosstrack_deadband))
            self.stanley.crosstrack_deadband_px = self.stanley.crosstrack_deadband

            curve_reference_for_control = curve_reference
            self._stanley_last_curve_guard = "none"
            if isinstance(curve_reference, dict):
                turn_direction = int(curve_reference.get('turn_direction', 0) or 0)
                intent_sign = 0
                intent_source = "none"
                error_guard = max(
                    0.001,
                    float(getattr(self, 'stanley_curve_direction_guard_error_m', 0.010) or 0.010),
                )
                heading_guard = math.radians(max(
                    0.1,
                    float(getattr(self, 'stanley_curve_direction_guard_heading_deg', 1.5) or 1.5),
                ))
                if abs(float(error_m)) >= error_guard:
                    intent_sign = 1 if float(error_m) > 0.0 else -1
                    intent_source = "error"
                elif abs(float(heading)) >= heading_guard:
                    intent_sign = 1 if float(heading) > 0.0 else -1
                    intent_source = "heading"

                if turn_direction != 0 and intent_sign != 0 and turn_direction != intent_sign:
                    curve_reference_for_control = None
                    self._stanley_last_curve_guard = (
                        f"direction_conflict(turn={turn_direction}, intent={intent_sign}, src={intent_source})"
                    )

            psi_ss, traj_yaw_rate, steer_damping_delta = self._get_stanley_dynamic_terms(
                speed_mps, curve_reference_for_control
            )
            self._stanley_last_speed_mps = float(speed_mps)
            self._stanley_last_speed_source = str(speed_source)
            self._stanley_last_error_m = float(error_m)
            self._stanley_last_px_per_cm = float(px_per_cm)

            # Route to Lateral MPC or Stanley based on config flag.
            # Both accept the same arguments; MPC ignores yaw_rate / traj_yaw_rate /
            # steer_damping_delta (predictive horizon replaces damping terms).
            if getattr(self, 'use_lateral_mpc', False):
                # Override steady_state_heading with path curvature feedforward when
                # the track graph is available.  Using atan(kappa * L) shifts the
                # initial psi0 seen by the MPC so it commands extra steer to maintain
                # the curve — without this the MPC under-steers in corners.
                effective_psi_ss = psi_ss
                curve_exit_unwind_factor = 1.0
                if direct_error_source == "two_line":
                    curve_hold_direction = int(getattr(self, "_curve_direction", 0) or 0)
                    if curve_hold_direction == 0:
                        curve_hold_direction = int(getattr(self, "_recent_curve_direction", 0) or 0)
                    if curve_hold_direction != 0:
                        curve_exit_unwind_factor = self._get_curve_exit_unwind_factor(
                            reinforced_sign=curve_hold_direction,
                        )
                if (
                    ts is not None and
                    getattr(ts, 'initialized', False) and
                    _imu_ready and
                    (_allow_stopline_waypoint or _route_active)
                ):
                    path_kappa = float(getattr(ts, 'path_kappa', 0.0))
                    # Clamp to a conservative curvature limit.  At ±1.0 rad/m
                    # the feedforward reaches ±14.5° which, added to the
                    # camera heading error (~20° in single-line curve mode),
                    # produces psi0 ≈ 35° and saturates the MPC at max steer
                    # (25°) indefinitely.  At ±0.3 rad/m the feedforward is
                    # capped at ±4.4°, preventing premature turning in the
                    # straight and reducing over-saturation in the curve.
                    path_kappa = max(min(path_kappa, 0.3), -0.3)
                    if abs(path_kappa) > 0.01:  # ignore negligible curvature
                        effective_psi_ss = math.atan(
                            path_kappa * float(self.lateral_mpc.L)
                        )
                        if curve_exit_unwind_factor < 0.999:
                            effective_psi_ss *= float(curve_exit_unwind_factor)
                # When both lane lines provide a direct physical error (psi≈0 is common),
                # augment heading with a Stanley crosstrack term so the MPC sees a
                # non-zero psi even when the car is aligned.  This avoids the degenerate
                # case where de/dt = v·sin(0) = 0 renders the MPC horizon useless.
                heading_mpc = control_heading
                if direct_error_m is not None and direct_error_source != "single_line":
                    k_eff = float(self.stanley_k) * float(self.mpc_crosstrack_k_mult)
                    heading_mpc = control_heading + math.atan2(
                        k_eff * float(error_m),
                        float(self.stanley_k_soft) + float(speed_mps),
                    )
                return self.lateral_mpc.compute(
                    error_m, heading_mpc, speed_mps,
                    steady_state_heading=effective_psi_ss,
                )
            return self.stanley.compute(
                error_m, control_heading, speed_mps,
                yaw_rate=self._yaw_rate,
                traj_yaw_rate=traj_yaw_rate,
                steady_state_heading=psi_ss,
                steer_damping_delta=steer_damping_delta,
            )
        return self.pid.compute(error)

    def _compute_single_line_error(
        self,
        line,
        side,
        img_h,
        img_w,
        prefer_center=False,
        physical_projection=False,
        reference_y_override=None,
    ):
        """Compute crosstrack error from a single visible lane marking.

        The target is reconstructed from the visible marking side plus the
        known track geometry:
        - lane width (inner-edge to inner-edge)
        - line width
        - car width (used to derive the body clearance, then the vehicle center)

        This keeps 1-line mode tied to the actual visible side instead of any
        curve-state heuristic.

        Returns:
            tuple: (crosstrack_error_px, heading_error_rad)
                   crosstrack_error: positive = car left of lane center (steer right)
                   heading_error: from single line angle (dampened)
        """
        x1, y1, x2, y2 = line[0]
        self._last_single_line_projection_debug = {}

        if y1 < y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        segment_top_y = int(max(0, min(img_h - 1, min(y1, y2))))
        segment_bottom_y = int(max(0, min(img_h - 1, max(y1, y2))))

        def _clamp_reference_y(y_target):
            return int(max(segment_top_y, min(segment_bottom_y, int(round(y_target)))))

        # Pixel-to-cm ratio: highest priority is the value cached by the last 2-line
        # frame from _ref_lw_px / lane_width_cm (same reference_y, same coordinate
        # system).  Using a different source (AI overall width, Hough cache) would
        # introduce a scale mismatch that shifts all D measurements by 2× or more.
        two_line_ppc = getattr(self, '_last_two_line_ref_px_per_cm', None)
        if two_line_ppc is not None and two_line_ppc > 0.5:
            px_per_cm = two_line_ppc
        else:
            cached_px_per_cm = getattr(self, '_last_px_per_cm', None)
            if cached_px_per_cm is not None and cached_px_per_cm > 0.5:
                px_per_cm = cached_px_per_cm
            else:
                # Fallback: lane (35cm) typically spans ~45% of image width at bottom
                px_per_cm = (img_w * 0.45) / self.lane_width_cm

        # On a fresh 2→1 transition, aim at the full reconstructed center to
        # avoid a steering jump. Otherwise keep the configured conservative bias.
        center_scale = 1.0 if prefer_center else max(
            0.0,
            min(1.5, float(getattr(self, 'single_line_offset_factor', 0.5)) / 0.5)
        )
        heading_raw = self._compute_raw_line_heading(line)
        ref_angle = float(getattr(
            self,
            '_single_line_heading_ref_left' if side == 'left' else '_single_line_heading_ref_right',
            0.0,
        ))
        abs_heading_deg = abs(math.degrees(heading_raw))
        heading_fade_start_deg = 35.0
        heading_fade_end_deg = 75.0
        if abs_heading_deg <= heading_fade_start_deg:
            heading_reliability = 1.0
        elif abs_heading_deg >= heading_fade_end_deg:
            heading_reliability = 0.0
        else:
            fade_span = max(1.0, heading_fade_end_deg - heading_fade_start_deg)
            heading_reliability = (heading_fade_end_deg - abs_heading_deg) / fade_span
        heading_delta = (heading_raw - ref_angle) * heading_reliability
        # Unscaled delta from the straight-road reference — used for inner/outer
        # direction checks below.  heading_delta is reliability-weighted (→ 0 at
        # high angles) so we keep the raw version for directional discrimination.
        heading_delta_from_ref = heading_raw - ref_angle
        ref_abs_heading_deg = abs(math.degrees(ref_angle))
        curve_bias_start_deg = max(20.0, ref_abs_heading_deg + 8.0)
        curve_bias_full_deg = max(curve_bias_start_deg + 12.0, ref_abs_heading_deg + 28.0)
        curve_strength = 0.0
        curve_heading_bias = 0.0
        if abs_heading_deg > curve_bias_start_deg:
            curve_span = max(1.0, curve_bias_full_deg - curve_bias_start_deg)
            curve_strength = min(1.0, (abs_heading_deg - curve_bias_start_deg) / curve_span)
            # Only apply the heading bias when the line's heading has moved in
            # the direction consistent with being the OUTER boundary of a curve
            # relative to the straight-road reference for this side:
            #   outer RIGHT (left  curve) → delta ≤ 0  (more leftward than ref)
            #   outer LEFT  (right curve) → delta ≥ 0  (more rightward than ref)
            # If the sign is reversed the visible line is the INNER boundary;
            # applying the outer-line heading bias would damp a correct hint.
            is_outer_heading_sign = (
                (side == 'right' and heading_delta_from_ref <= 0) or
                (side == 'left'  and heading_delta_from_ref >= 0)
            )
            if is_outer_heading_sign:
                curve_sign = 1.0 if side == 'left' else -1.0
                curve_heading_bias = curve_sign * math.radians(
                    float(getattr(self, 'single_line_curve_heading_max_deg', 10.0))
                ) * curve_strength
            elif (
                int(getattr(self, '_curve_direction', 0) or 0) == 0 and
                str(getattr(self, '_curve_state', 'STRAIGHT')) == 'STRAIGHT'
            ):
                # When there is no confirmed curve state yet, use a side-based
                # pre-bias so shallow 1-line hints can still seed the turn.
                curve_sign = 1.0 if side == 'left' else -1.0
                curve_heading_bias = curve_sign * math.radians(
                    float(getattr(self, 'single_line_curve_heading_max_deg', 10.0))
                ) * curve_strength

        if physical_projection:
            if reference_y_override is not None:
                reference_y = _clamp_reference_y(reference_y_override)
            else:
                reference_y = _clamp_reference_y(
                    img_h * (1.0 - float(getattr(self, 'lookahead', 0.4)))
                )
            reference_x = self._line_x_at_y(line, reference_y)
            if reference_x is None:
                reference_x = x1

            line_to_center_cm = ((self.line_width_cm * 0.5) + (self.lane_width_cm * 0.5)) * center_scale
            offset_px = line_to_center_cm * px_per_cm
            if side == 'left':
                desired_center = reference_x + offset_px
            else:
                desired_center = reference_x - offset_px

            safe_heading_delta = max(
                -math.radians(45.0),
                min(math.radians(45.0), heading_delta)
            )
            camera_shift_cm = math.tan(safe_heading_delta) * float(getattr(self, 'camera_to_front_axle', 0.0))
            camera_shift_px = camera_shift_cm * px_per_cm
            desired_center -= camera_shift_px
            mode = 'physical'
        else:
            reference_y = _clamp_reference_y(img_h - 1)
            reference_x = self._line_x_at_y(line, reference_y)
            if reference_x is None:
                reference_x = x1

            # Body clearance to the visible line when the vehicle is centered.
            body_clearance_cm = max(0.0, (self.lane_width_cm - self.car_width) * 0.5)
            # Distance from the detected marking center to the desired vehicle center.
            line_to_center_cm = (self.line_width_cm * 0.5) + body_clearance_cm + (self.car_width * 0.5)
            offset_px = line_to_center_cm * center_scale * px_per_cm

            if side == 'left':
                desired_center = reference_x + offset_px
            else:
                desired_center = reference_x - offset_px

            camera_shift_px = 0.0
            mode = 'legacy'

        error = desired_center - (img_w / 2.0)

        curve_direction = int(getattr(self, '_curve_direction', 0) or 0)
        curve_state = str(getattr(self, '_curve_state', 'STRAIGHT'))
        # Heading-delta guard: the outer boundary of a curve tilts MORE in the
        # curve direction relative to the straight-road reference
        # (outer right → delta ≤ 0; outer left → delta ≥ 0).
        # Using the delta (not absolute heading_raw) handles shallow curves where
        # the inner line still has a "wrong-side" absolute tilt from perspective.
        _is_outer_heading_sign = (
            (side == 'right' and heading_delta_from_ref <= 0) or
            (side == 'left'  and heading_delta_from_ref >= 0)
        )
        is_outer_curve_line = (
            _is_outer_heading_sign and
            curve_state in ("ENTERING", "IN_CURVE", "EXITING") and
            (
                (curve_direction == 1 and side == 'left') or
                (curve_direction == -1 and side == 'right')
            )
        )
        outer_line_bias_px = 0.0
        outer_line_proximity = 0.0
        if is_outer_curve_line:
            lane_width_px = max(1.0, self.lane_width_cm * px_per_cm)
            line_gap_px = abs(float(reference_x) - (img_w / 2.0))
            outer_line_proximity = 1.0 - min(1.0, line_gap_px / lane_width_px)
            outer_confirmation = self._confirm_single_line_outer_curve(
                line, side, img_h, img_w, curve_strength, heading_raw
            )
            if outer_confirmation.get('confirmed', False):
                outer_line_weight = max(curve_strength, max(0.0, 1.0 - heading_reliability))
                target_negative_mag = (
                    lane_width_px *
                    max(0.0, float(getattr(self, 'single_line_outer_bias_gain', 0.0))) *
                    outer_line_proximity *
                    outer_line_weight
                )
                target_negative_mag = min(
                    target_negative_mag,
                    lane_width_px * max(0.0, float(getattr(self, 'single_line_outer_bias_limit_factor', 0.18))),
                )
                if target_negative_mag > 0.0:
                    current_negative_mag = max(0.0, -float(error))
                    if current_negative_mag < target_negative_mag:
                        error = -target_negative_mag
                    outer_line_bias_px = float(error) + current_negative_mag
            else:
                is_outer_curve_line = False
        else:
            outer_confirmation = self._confirm_single_line_outer_curve(
                line, side, img_h, img_w, curve_strength, heading_raw
            )
            if outer_confirmation.get('confirmed', False):
                lane_width_px = max(1.0, self.lane_width_cm * px_per_cm)
                line_gap_px = abs(float(reference_x) - (img_w / 2.0))
                outer_line_proximity = 1.0 - min(1.0, line_gap_px / lane_width_px)
                outer_line_weight = max(curve_strength, max(0.0, 1.0 - heading_reliability))
                target_negative_mag = (
                    lane_width_px *
                    max(0.0, float(getattr(self, 'single_line_outer_bias_gain', 0.0))) *
                    outer_line_proximity *
                    outer_line_weight
                )
                target_negative_mag = min(
                    target_negative_mag,
                    lane_width_px * max(0.0, float(getattr(self, 'single_line_outer_bias_limit_factor', 0.18))),
                )
                if target_negative_mag > 0.0:
                    current_negative_mag = max(0.0, -float(error))
                    if current_negative_mag < target_negative_mag:
                        error = -target_negative_mag
                    outer_line_bias_px = float(error) + current_negative_mag
                is_outer_curve_line = True

        # Heading from single line (dampened — less reliable than two-line)
        # Remove static perspective bias from single-line heading.
        heading_gain = 0.2 if prefer_center else 0.3
        if physical_projection and curve_state in ("ENTERING", "IN_CURVE"):
            heading_gain = min(
                0.85,
                heading_gain * float(
                    getattr(_config, "SINGLE_LINE_CURVE_HEADING_GAIN_MULT", 2.5) or 2.5
                ),
            )
        heading = (heading_delta * heading_gain) + curve_heading_bias

        self._last_single_line_projection_debug = {
            'single_line_reference_y': int(reference_y),
            'single_line_px_per_cm': float(px_per_cm),
            'single_line_target_x': float(desired_center),
            'single_line_camera_shift_px': float(camera_shift_px),
            'single_line_heading_raw_deg': float(math.degrees(heading_raw)),
            'single_line_ref_angle_deg': float(math.degrees(ref_angle)),
            'single_line_heading_delta_from_ref_deg': float(math.degrees(heading_delta_from_ref)),
            'single_line_is_outer_heading': bool(_is_outer_heading_sign),
            'single_line_heading_reliability': float(heading_reliability),
            'single_line_angle_from_vertical_deg': float(abs_heading_deg),
            'single_line_curve_like': bool(curve_strength > 0.0),
            'single_line_curve_strength': float(curve_strength),
            'single_line_curve_heading_bias_deg': float(math.degrees(curve_heading_bias)),
            'single_line_heading_gain': float(heading_gain),
            'single_line_outer_curve_line': bool(is_outer_curve_line),
            'single_line_outer_confirmed': bool(outer_confirmation.get('confirmed', False)),
            'single_line_outer_confirm_sources': ",".join(outer_confirmation.get('sources', [])),
            'single_line_outer_proximity': float(outer_line_proximity),
            'single_line_outer_bias_px': float(outer_line_bias_px),
            'single_line_mode': mode,
        }

        return error, heading

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
        needs_debug = self._needs_debug

        # 1. ROI mask — simple rectangular ROI (no hood notch).
        y_start = int(self.roi_height_start * height)
        y_end = int(self.roi_height_end * height)
        roi_vertices = np.array([[(0, y_start), (width, y_start), (width, y_end), (0, y_end)]], dtype=np.int32)
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, roi_vertices, (255, 255, 255))
        masked_image = cv2.bitwise_and(image, mask)
        if needs_debug:
            debug_info['roi'] = masked_image.copy()
            debug_info['roi_vertices'] = roi_vertices.copy()

        # 2. Grayscale
        grey_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        if needs_debug:
            debug_info['grey'] = grey_image.copy()

        # 3. Binary threshold
        _, binary_image = cv2.threshold(grey_image, threshold_val, 255, cv2.THRESH_BINARY)
        if needs_debug:
            debug_info['binary'] = binary_image.copy()

        # 4. Median blur
        noiseless = cv2.medianBlur(binary_image, kernel_val)
        if needs_debug:
            debug_info['noiseless'] = noiseless.copy()

        # 5. Canny
        canny = cv2.Canny(noiseless, self.canny_low, self.canny_high)
        if needs_debug:
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
        avg_left = self._smooth_detected_line(avg_left, 'left')
        avg_right = self._smooth_detected_line(avg_right, 'right')

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

    def _smooth_detected_line(self, line, side):
        """Apply a light EMA to the representative line to reduce frame-to-frame jitter."""
        if side == 'left':
            state_attr = '_smoothed_left_line'
            miss_attr = '_left_line_missing_frames'
        else:
            state_attr = '_smoothed_right_line'
            miss_attr = '_right_line_missing_frames'

        if line is None:
            missing = getattr(self, miss_attr) + 1
            setattr(self, miss_attr, missing)
            previous = getattr(self, state_attr)
            if missing >= self.line_visual_missing_reset_frames:
                setattr(self, state_attr, None)
                return None
            if previous is not None:
                return np.rint(previous).astype(np.int32)
            return None

        setattr(self, miss_attr, 0)
        current = line.astype(np.float32)
        previous = getattr(self, state_attr)
        alpha = max(0.0, min(1.0, self.line_visual_smoothing_alpha))

        if previous is None:
            smoothed = current
        else:
            smoothed = (1.0 - alpha) * previous + alpha * current

        setattr(self, state_attr, smoothed)
        return np.rint(smoothed).astype(np.int32)

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

        # Draw ROI contour used by the image pipeline
        roi_vertices = debug_info.get('roi_vertices')
        roi_top = 0
        if roi_vertices is not None:
            cv2.polylines(debug, roi_vertices, True, (255, 255, 0), 2)
            roi_top = int(np.min(roi_vertices[0][:, 1]))

        render_side_masks = debug_info.get('render_side_masks')
        using_raw_mask = isinstance(render_side_masks, dict) and any(
            isinstance(render_side_masks.get(side), np.ndarray) and render_side_masks.get(side).size > 0
            for side in ('left', 'right')
        )

        is_ai_local_mode = debug_info.get('control_mode') == 'mask_guidance' or \
            'AI Local' in str(debug_info.get('pipeline_label', ''))

        if using_raw_mask:
            mask_overlay = np.zeros_like(debug)
            left_mask = render_side_masks.get('left')
            right_mask = render_side_masks.get('right')
            if isinstance(left_mask, np.ndarray) and left_mask.size > 0:
                mask_overlay[left_mask > 0] = (255, 120, 0)
            if isinstance(right_mask, np.ndarray) and right_mask.size > 0:
                mask_overlay[right_mask > 0] = (0, 200, 255)
            if np.any(mask_overlay):
                debug = cv2.addWeighted(debug, 1.0, mask_overlay, 0.35, 0)
        elif not is_ai_local_mode:
            # Red lines: only in OpenCV/BFMC mode, never in AI_LOCAL mask mode
            if avg_left is not None:
                self._bfmc_draw_extended_line(debug, avg_left, h, (0, 0, 255), 2, top_y=roi_top)
            if avg_right is not None:
                self._bfmc_draw_extended_line(debug, avg_right, h, (0, 0, 255), 2, top_y=roi_top)

        # Draw midpoint and center
        if midpoint_x is not None:
            midpoint_y = int(debug_info.get('midpoint_draw_y', h - 1))
            midpoint_y = max(0, min(h - 1, midpoint_y))
            cv2.circle(debug, (midpoint_x, midpoint_y), 8, (0, 255, 255), -1)  # Cyan = midpoint
            cv2.circle(debug, (w // 2, midpoint_y), 8, (0, 255, 0), -1)       # Green = center
            cv2.line(debug, (midpoint_x, midpoint_y), (w // 2, midpoint_y), (0, 165, 255), 2)  # Orange = error

        # Header panel
        header_height = 70 if using_raw_mask else 55
        cv2.rectangle(debug, (0, 0), (w, header_height), (20, 20, 40), -1)
        pipeline_label = debug_info.get('pipeline_label', "OpenCV - BFMC Pipeline")
        cv2.putText(debug, pipeline_label, (8, 18),
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
        if using_raw_mask:
            local_mask_guidance = debug_info.get('local_mask_guidance', {})
            mask_label = debug_info.get('mask_label', 'NONE')
            lane_width_px = local_mask_guidance.get('lane_width_px')
            width_text = "--"
            if lane_width_px is not None:
                width_text = f"{float(lane_width_px):.0f}px"
            cv2.putText(debug, f"Mode: RAW MASK  Mask: {mask_label}  Width: {width_text}", (8, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1)
        else:
            cv2.putText(debug, f"Thr:{th} L:{l_count} R:{r_count}", (8, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        remote_server_ms = debug_info.get('remote_server_time_ms')
        if remote_server_ms is not None:
            remote_rt_ms = debug_info.get('remote_roundtrip_ms', 0.0)
            perf_x = max(8, w - 250)
            cv2.putText(debug, f"AI:{remote_server_ms:.0f}ms RT:{remote_rt_ms:.0f}ms",
                        (perf_x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 220, 255), 1)
            perf_parts = []
            local_ai_processing_fps = debug_info.get('local_ai_processing_fps')
            if local_ai_processing_fps is not None:
                perf_parts.append(f"Proc:{float(local_ai_processing_fps):.1f}FPS")
            local_ai_throughput_fps = debug_info.get('local_ai_throughput_fps')
            if local_ai_throughput_fps is not None:
                perf_parts.append(f"Thr:{float(local_ai_throughput_fps):.1f}FPS")
            local_ai_target_fps = debug_info.get('local_ai_target_fps')
            if local_ai_target_fps is not None:
                perf_parts.append(f"Cap:{float(local_ai_target_fps):.1f}FPS")
            if perf_parts:
                cv2.putText(debug, " ".join(perf_parts),
                            (perf_x, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 240, 255), 1)
        kalman_error = debug_info.get('kalman_error')
        if kalman_error is not None:
            k_age = int(debug_info.get('kalman_age', 0))
            k_mode = "P" if bool(debug_info.get('kalman_predicted', False)) else "C"
            kalman_y = 66 if using_raw_mask else 52
            cv2.putText(debug, f"K:{k_mode} e={kalman_error:.0f}px a={k_age}", (165, kalman_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 255, 180), 1)

        # === SWEPT PATH VISUALIZATION ===
        swept = self._swept_path_info
        if swept is not None and self.use_swept_path:
            bar_y = max(58, header_height + 3)  # Below the header panel

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
                src_str = swept.get('source', '?')
                cv2.putText(debug, f"CURVE {dir_str} [{mode_tag}|{src_str}]  "
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
                # Straight road indicator
                cv2.rectangle(debug, (0, bar_y), (w, bar_y + 15), (40, 40, 40), -1)
                cv2.putText(debug, "No curve offset",
                            (8, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 200, 150), 1)

        return debug

    def _bfmc_draw_extended_line(self, image, line, height, color, thickness, top_y=0):
        """Draw a line extended between the ROI top and the bottom of the image."""
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        if dx == 0:
            dx = 0.001
        slope = (y2 - y1) / dx
        if slope == 0:
            slope = 0.001
        top_y = max(0, min(height, int(top_y)))
        # Intersection at the ROI top
        x_top = int(x1 + (top_y - y1) / slope)
        # Intersection at y=height (bottom)
        x_bot = int(x1 + (height - y1) / slope)
        cv2.line(image, (x_top, top_y), (x_bot, height), color, thickness)

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
        prev_direction = int(getattr(self, '_curve_direction', 0) or 0)
        self._curve_state_frames += 1
        recent_visual_straight = self._has_recent_two_line_straight_hint()
        if self._recent_curve_direction_frames > 0:
            self._recent_curve_direction_frames -= 1
            if self._recent_curve_direction_frames <= 0:
                self._recent_curve_direction_frames = 0
                self._recent_curve_direction = 0
        two_line_hint_ready, two_line_hint_direction = self._consume_two_line_curve_hint(
            num_lines, avg_left, avg_right
        )

        previous_steering = getattr(self, '_last_good_steering', None)
        if previous_steering is None:
            previous_steering = getattr(self, 'last_steering', None)

        def _resolve_recent_curve_reentry_direction(fallback_direction):
            fallback_direction = int(fallback_direction or 0)
            recent_direction = int(getattr(self, '_recent_curve_direction', 0) or 0)
            recent_frames = int(getattr(self, '_recent_curve_direction_frames', 0) or 0)
            if (
                fallback_direction == 0 or
                recent_direction == 0 or
                recent_frames <= 0 or
                recent_direction == fallback_direction
            ):
                return fallback_direction, False
            if previous_steering is None:
                return fallback_direction, False
            min_reentry_steer = max(
                0.0,
                float(getattr(self, 'curve_reentry_min_steer_deg', 3.0) or 3.0),
            )
            if abs(float(previous_steering)) < min_reentry_steer:
                return fallback_direction, False
            if float(previous_steering) * recent_direction <= 0.0:
                return fallback_direction, False
            return recent_direction, True

        if self._curve_state == "STRAIGHT":
            if num_lines == 1:
                if recent_visual_straight:
                    self._curve_enter_origin = "none"
                else:
                    # Lost a line → entering curve (only reliable trigger)
                    inferred_direction = 1 if avg_left is not None else -1
                    inferred_direction, reused_recent_curve = _resolve_recent_curve_reentry_direction(
                        inferred_direction
                    )
                    self._curve_state = "ENTERING"
                    self._curve_state_frames = 1
                    self._curve_direction = int(inferred_direction)
                    self._curve_enter_origin = "recent_curve_hold" if reused_recent_curve else "single_line"
            elif two_line_hint_ready and two_line_hint_direction != 0:
                # Strong, stable two-line heading hint from AI local: start pre-positioning
                # before the outer line disappears so the car doesn't react one frame late.
                self._curve_state = "ENTERING"
                self._curve_state_frames = 1
                self._curve_direction = int(two_line_hint_direction)
                self._curve_enter_origin = "two_line_hint"
            # NOTE: VP-based detection from 2 lines was removed — too many false positives.
            # The vanishing point shifts off-center when the car is slightly off-center
            # or the camera isn't perfectly aligned, causing constant false ENTERING triggers.
            # The only reliable curve indicator is actually losing a line (2→1 transition).

        elif self._curve_state == "ENTERING":
            if num_lines == 1:
                if self._curve_state_frames >= self.curve_confirm_frames:
                    self._curve_state = "IN_CURVE"
                    self._curve_state_frames = 1
                    self._curve_enter_origin = "none"
                # Once a curve direction is established, keep it while the visible side
                # changes. In real turns the detector can switch from outer to inner line
                # mid-curve; that does not mean the curve reversed.
                if self._curve_direction == 0:
                    self._curve_direction = 1 if avg_left is not None else -1
            elif num_lines == 2:
                two_line_hold = max(1, int(getattr(self, 'curve_two_line_hold_frames', 6) or 6))
                if two_line_hint_ready and two_line_hint_direction != 0:
                    # Keep the early-entering state alive while the two-line hint remains strong.
                    self._curve_direction = int(two_line_hint_direction)
                    self._curve_enter_origin = "two_line_hint"
                elif (
                    self._curve_enter_origin in ("two_line_hint", "recent_curve_hold") and
                    self._curve_state_frames <= two_line_hold
                ):
                    # Short confidence dips are common on the last safe frame before line loss.
                    # Keep ENTERING alive briefly so the steering does not unwind to zero.
                    pass
                elif (
                    self._curve_enter_origin in ("two_line_hint", "recent_curve_hold") and
                    self._curve_state_frames > two_line_hold and
                    self._curve_state_frames <= two_line_hold * 4
                ):
                    # Still seeing 2 lines after hold expired but hint was strong enough to
                    # enter. Promote to IN_CURVE rather than resetting — the car may still be
                    # mid-curve and just hasn't lost the outer line yet.
                    self._curve_state = "IN_CURVE"
                    self._curve_state_frames = 1
                    self._curve_enter_origin = "none"
                else:
                    # Regained both lines → back to STRAIGHT.
                    # If it's a real curve, we'll lose the line again and re-enter ENTERING.
                    self._curve_state = "STRAIGHT"
                    self._curve_state_frames = 0
                    self._curve_direction = 0
                    self._curve_enter_origin = "none"
                    self._steering_radius_history.clear()  # Clear stale steering data
            elif num_lines == 0:
                # Lost all lines in entering - go to IN_CURVE (assume committed)
                self._curve_state = "IN_CURVE"
                self._curve_state_frames = 1
                self._curve_enter_origin = "none"

        elif self._curve_state == "IN_CURVE":
            if num_lines == 2:
                self._curve_state = "EXITING"
                self._curve_state_frames = 1
                self._curve_enter_origin = "none"
                # Reset PID integral so accumulated curve error doesn't push the car wide on exit
                self.pid.integral = 0.0
                self.pid.prev_error = 0
            elif num_lines == 1:
                # Keep the established curve direction while single-line visibility
                # alternates between inner and outer boundaries.
                if self._curve_direction == 0:
                    self._curve_direction = 1 if avg_left is not None else -1

        elif self._curve_state == "EXITING":
            if num_lines == 2:
                if self._curve_state_frames >= self.curve_exit_frames:
                    if prev_direction != 0:
                        self._recent_curve_direction = int(prev_direction)
                        self._recent_curve_direction_frames = max(
                            0,
                            int(getattr(self, 'curve_reentry_hold_frames', 10) or 10),
                        )
                    self._curve_state = "STRAIGHT"
                    self._curve_state_frames = 0
                    self._curve_direction = 0
                    self._curve_enter_origin = "none"
                    self._curve_radius_estimate = 0.0
                    self._curve_confidence = 0.0
                    self._steering_radius_history.clear()
            elif num_lines <= 1:
                # Lost line again - back to IN_CURVE
                self._curve_state = "IN_CURVE"
                self._curve_state_frames = 1
                self._curve_enter_origin = "none"
                if num_lines == 1 and self._curve_direction == 0:
                    self._curve_direction = 1 if avg_left is not None else -1

        # Update radius estimate using steering feedback
        self._update_steering_radius_estimate()

        if self.show_debug and self._curve_state != prev_state:
            dir_s = {-1: "LEFT", 0: "-", 1: "RIGHT"}.get(self._curve_direction, "?")
            print(f"\033[1;97m[ Curve Assist ] :\033[0m \033[1;96mdir={dir_s}\033[0m "
                  f"R={self._curve_radius_estimate:.0f}cm frames={self._curve_state_frames}")

    def _apply_graph_curve_state(self, tracking_state) -> None:
        """Correct visual curve state using graph-based lookahead heading change.

        The dense path is built directly from graph nodes and edges, so
        path_heading_change_rad reflects true track geometry — not camera noise.
        Two corrections are applied when GPS tracking is initialized:

        1. ENTERING with GPS saying straight → false entry (e.g. right lane lost at
           intersection), clear it immediately.
        2. EXITING with GPS saying straight → confirm the curve is done, prevent
           oscillation back to IN_CURVE from lost lines on a straight.
        3. IN_CURVE with wrong GPS direction → fix _curve_direction sign.

        IN_CURVE is intentionally NOT cleared here: the curve arc on this track is
        ~2 m long, and 1.5 m of lookahead from matched_idx exits the arc before the
        car does — premature clearing would kill curve assistance mid-turn.
        The enforcement bypass in _enforce_single_line_curve_priority already prevents
        wrong-direction enforcement while IN_CURVE when the graph lookahead disagrees.
        """
        if tracking_state is None or not getattr(tracking_state, 'initialized', False):
            return
        if self._has_recent_two_line_straight_hint():
            if self._curve_state in ("ENTERING", "IN_CURVE", "EXITING"):
                self._curve_state = "STRAIGHT"
                self._curve_state_frames = 0
                self._curve_direction = 0
                self._curve_enter_origin = "none"
            return
        gps_dir, delta_psi = self._graph_curve_direction(
            tracking_state,
            min_heading_change_deg=15.0,
        )
        strong_curve_ahead = abs(float(delta_psi)) >= math.radians(25.0)

        if gps_dir == 0:
            # GPS sees no significant curve in the next 1.5 m.
            # Clear ENTERING (false visual trigger) or EXITING (confirm done).
            # Do NOT clear IN_CURVE — enforcement bypass handles that case.
            if self._curve_state in ("ENTERING", "EXITING"):
                self._curve_state = "STRAIGHT"
                self._curve_state_frames = 0
                self._curve_direction = 0
                self._curve_enter_origin = "none"
        else:
            if (
                self._curve_state == "EXITING" and
                strong_curve_ahead and
                (self._curve_direction == 0 or self._curve_direction == gps_dir)
            ):
                self._curve_state = "IN_CURVE"
                self._curve_state_frames = 1
                self._curve_direction = gps_dir
                self._curve_enter_origin = "none"
                return
            # GPS sees a real curve — fix direction if visual got the sign wrong
            if self._curve_direction == 0 and self._curve_state in ("ENTERING", "IN_CURVE", "EXITING"):
                self._curve_direction = gps_dir
            elif self._curve_direction != 0 and self._curve_direction != gps_dir:
                self._curve_direction = gps_dir

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

    def _reset_two_line_curve_hint(self):
        self._two_line_curve_hint_count = 0
        self._two_line_curve_hint_direction = 0

    def _consume_two_line_curve_hint(self, num_lines, avg_left, avg_right):
        """Track strong two-line AI heading hints to start ENTERING before line loss."""
        if num_lines != 2 or avg_left is None or avg_right is None:
            self._reset_two_line_curve_hint()
            return False, 0

        hint_source = str(getattr(self, '_local_ai_heading_hint_source', 'none') or 'none')
        if hint_source != 'two_line':
            self._reset_two_line_curve_hint()
            return False, 0

        hint_conf = float(getattr(self, '_local_ai_heading_hint_confidence', 0.0) or 0.0)
        hint_rad = float(getattr(self, '_local_ai_heading_hint_rad', 0.0) or 0.0)
        hint_deg = abs(math.degrees(hint_rad))
        min_conf = max(
            0.0,
            min(1.0, float(getattr(self, 'curve_two_line_heading_confidence', 0.55) or 0.55)),
        )
        min_deg = max(0.0, float(getattr(self, 'curve_two_line_heading_threshold_deg', 24.0) or 24.0))
        if hint_conf < min_conf or hint_deg < min_deg:
            self._reset_two_line_curve_hint()
            return False, 0

        hint_direction = 1 if hint_rad > 0.0 else -1
        if self._two_line_curve_hint_direction == hint_direction:
            self._two_line_curve_hint_count += 1
        else:
            self._two_line_curve_hint_direction = hint_direction
            self._two_line_curve_hint_count = 1

        confirm_frames = max(
            1,
            int(getattr(self, 'curve_two_line_confirm_frames', 2) or 2),
        )
        return self._two_line_curve_hint_count >= confirm_frames, hint_direction

    def _is_seeing_inner_line(self, visible_side):
        """Check if the visible line is the INNER boundary of the current curve.
        
        During a confirmed curve (IN_CURVE/EXITING), we know the direction.
        The inner line is the one on the same side as the turn:
        - RIGHT curve: RIGHT line = inner, LEFT line = outer
        - LEFT curve:  LEFT line = inner, RIGHT line = outer
        
        When the car goes off the INNER line (turned too tight), it needs a
        different correction than going off the outer line (can't make the turn).
        
        Args:
            visible_side: 'left' or 'right' - which line is currently visible
            
        Returns:
            bool: True if the visible line is the inner boundary
        """
        if self._curve_state not in ("IN_CURVE", "EXITING"):
            return False
        if self._curve_direction == 0:
            return False
        # RIGHT curve (dir=1): inner line is RIGHT
        # LEFT curve (dir=-1): inner line is LEFT
        if self._curve_direction == 1 and visible_side == 'right':
            return True
        if self._curve_direction == -1 and visible_side == 'left':
            return True
        return False

    def _is_transversal_recovery_active(self, local_mask_guidance, debug_info=None):
        """Return True when single-line guidance is following a transversal marker."""
        if isinstance(local_mask_guidance, dict):
            if str(local_mask_guidance.get('guidance_mode', '') or '') == 'transversal_recovery':
                return True
            projection_debug = local_mask_guidance.get('single_line_projection_debug')
            if isinstance(projection_debug, dict):
                if bool(projection_debug.get('single_line_transversal_detected', False)):
                    return True
                if str(projection_debug.get('single_line_mode', '') or '') == 'transversal_recovery':
                    return True
        if isinstance(debug_info, dict):
            if bool(debug_info.get('single_line_transversal_detected', False)):
                return True
            if str(debug_info.get('single_line_mode', '') or '') == 'transversal_recovery':
                return True
        return False

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
        # NOTE: Do NOT use steering feedback here — during ENTERING the steering
        # history contains PID correction angles from STRAIGHT, not actual curve radii.
        # Using them causes false R=59cm detections (25° PID correction ≠ real curve).
        if state == "ENTERING":
            if vp_radius and vp_radius > 0 and vp_confidence > 0.3:
                # Blend VP with BFMC default (VP is still somewhat reliable here)
                blended = 0.4 * vp_radius + 0.6 * self.bfmc_default_curve_radius
                blended = max(30.0, min(200.0, blended))
                self._curve_radius_estimate = blended
                self._curve_confidence = 0.7
                return blended, 0.7, "VP+BFMC"
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

        # Recalibrate bev_cm_per_px every frame using actual lane positions projected into BEV.
        # The fixed geometric estimate computed at init (lane_width_cm / dst_width) assumes the
        # full dst rectangle represents one lane, which is incorrect: in practice the lane
        # covers only ~41% of the src trapezoid bottom. By projecting the detected bottom lane
        # points through the perspective matrix we get the true BEV lane width in pixels,
        # making the metric scale accurate and dynamic.
        if (
            getattr(self, 'perspective_initialized', False) and
            getattr(self, 'perspective_M', None) is not None and
            lane_width_px > 10.0
        ):
            try:
                pts = np.array(
                    [[[bottom_left_x, float(img_h)]], [[bottom_right_x, float(img_h)]]],
                    dtype=np.float32,
                )
                bev_pts = cv2.perspectiveTransform(pts, self.perspective_M)
                bev_lane_w = abs(float(bev_pts[1][0][0]) - float(bev_pts[0][0][0]))
                if bev_lane_w > 5.0 and float(self.lane_width_cm) > 0.0:
                    self.bev_cm_per_px = self.lane_width_cm / bev_lane_w
            except Exception:
                pass

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

            # STRAIGHT state: VP data is only used by FSM for curve detection,
            # NOT for steering. Return zero offset to avoid false corrections.
            if self._curve_state == "STRAIGHT":
                return {
                    'offset_px': 0.0, 'curve_radius_cm': curve.get('curve_radius_cm', 0),
                    'turn_direction': curve.get('turn_direction', 0),
                    'min_clearance_cm': float('inf'),
                    'critical_corner': 'straight', 'confidence': curve.get('confidence', 0),
                    'px_per_cm': curve['px_per_cm'], 'clearances': {},
                    'fits': True, 'source': 'none',
                    'curve_state': 'STRAIGHT',
                }

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
            # Apply gain based on curve state:
            #   STRAIGHT: NO offset — VP data is only used by FSM, not steering
            #   ENTERING: pre-position (less aggressive, building up)
            #   IN_CURVE: full curve offset
            #   EXITING:  reduced offset — car needs to straighten, not follow curve
            if self._curve_state == "STRAIGHT":
                gain = 0.0  # Don't apply curve offset in straight — only FSM uses VP data
            elif self._curve_state == "ENTERING":
                gain = self.curve_pre_position_gain
            elif self._curve_state == "EXITING":
                # Progressively reduce gain as we spend more frames in EXITING
                # Frame 1: curve_exit_gain, then fade toward 0 over curve_exit_frames
                exit_progress = min(1.0, self._curve_state_frames / max(1, self.curve_exit_frames))
                gain = self.curve_exit_gain * (1.0 - exit_progress)
            else:
                gain = 1.0  # IN_CURVE: full offset
            offset_px = -offset_result['offset_cm'] * curve['px_per_cm'] * turn_direction * gain

            # Apply additional inner bias (pushes car toward inside of curve)
            if self._curve_state in ("ENTERING", "IN_CURVE", "EXITING") and self.curve_inner_bias_cm > 0:
                bias_gain = gain  # Same gain as offset (fades during EXITING)
                inner_bias_px = self.curve_inner_bias_cm * curve['px_per_cm'] * turn_direction * bias_gain
                offset_px += inner_bias_px

            if self.show_debug:
                print(f"\033[1;97m[ Hybrid 2L ] :\033[0m "
                      f"src={source} R={best_radius:.0f}cm dir={'R' if turn_direction > 0 else 'L'} "
                      f"offset={offset_result['offset_cm']:.1f}cm→{offset_px:.0f}px "
                      f"bias={self.curve_inner_bias_cm:.1f}cm "
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

            # Prefer direction tracked by the curve FSM when available.
            # Side-based fallback is kept for first entering frames.
            if self._curve_direction in (-1, 1):
                turn_direction = self._curve_direction
            else:
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

            # Apply additional inner bias (pushes car toward inside of curve)
            inner_bias_px = 0.0
            if self._curve_state in ("ENTERING", "IN_CURVE", "EXITING") and self.curve_inner_bias_cm > 0:
                inner_bias_px = self.curve_inner_bias_cm * px_per_cm * turn_direction
                offset_px += inner_bias_px

            target_x = lane_center_x + offset_px
            steering_error = target_x - (img_w / 2.0)

            if self.show_debug:
                dir_str = "R" if turn_direction > 0 else "L"
                print(f"\033[1;97m[ Hybrid 1L ] :\033[0m \033[1;93m{self._curve_state} {dir_str}\033[0m "
                      f"src={source} R={best_radius:.0f}cm angle={angle_from_vertical:.1f}° "
                      f"offset={offset_result['offset_cm']:.1f}cm→{offset_px:.0f}px "
                      f"bias={self.curve_inner_bias_cm:.1f}cm({inner_bias_px:.0f}px) "
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
        """Initialize perspective transform matrices for bird's eye view.

        The destination rectangle is anchored to lane_width_cm so that the
        BEV has a known metric scale: bev_cm_per_px cm per BEV pixel.
        The trapezoid src points model the road visible from the camera at
        its mounting position (bottom 40% of image height, 10–90% width).
        """
        # Source points: trapezoid covering the road region in perspective view.
        # Bottom edge spans 10–90% of frame width (the widest visible road).
        # Top edge spans 40–60% of frame width at 60% height (horizon of interest).
        src = np.float32([
            [width * 0.1, height],           # bottom-left
            [width * 0.4, height * 0.6],     # top-left
            [width * 0.6, height * 0.6],     # top-right
            [width * 0.9, height]            # bottom-right
        ])

        # Destination points: rectangle in BEV space.
        # Width of the rectangle (dst_x_right - dst_x_left) represents the
        # lane_width_cm road region visible at the bottom of the src trapezoid.
        # The bottom src span is width*0.8 pixels = lane_width_cm => bev_cm_per_px.
        dst_x_left  = width * 0.2
        dst_x_right = width * 0.8
        dst = np.float32([
            [dst_x_left,  height],   # bottom-left
            [dst_x_left,  0],        # top-left
            [dst_x_right, 0],        # top-right
            [dst_x_right, height]    # bottom-right
        ])

        self.perspective_M     = cv2.getPerspectiveTransform(src, dst)
        self.perspective_M_inv = cv2.getPerspectiveTransform(dst, src)
        self.perspective_initialized = True

        # Metric scale: the dst rectangle spans dst_x_right - dst_x_left pixels
        # and corresponds to the road region whose physical width is lane_width_cm.
        # (The src bottom span width*0.8 maps 1:1 to lane_width_cm at the bottom.)
        bev_lane_width_px = float(dst_x_right - dst_x_left)
        lane_width_cm = float(getattr(self, 'lane_width_cm', 35.0) or 35.0)
        if bev_lane_width_px > 0.0 and lane_width_cm > 0.0:
            self.bev_cm_per_px = lane_width_cm / bev_lane_width_px
        else:
            self.bev_cm_per_px = 0.0

        print(
            f"\033[1;97m[ Line Following ] :\033[0m \033[1;92mINFO\033[0m - "
            f"BEV initialized {width}x{height} | bev_cm_per_px={self.bev_cm_per_px:.4f} "
            f"(lane={lane_width_cm}cm / {bev_lane_width_px:.0f}px)"
        )

    def warp_perspective(self, img):
        """Transform image to bird's eye view."""
        return cv2.warpPerspective(img, self.perspective_M, (img.shape[1], img.shape[0]))

    def unwarp_perspective(self, img):
        """Transform image back from bird's eye view."""
        return cv2.warpPerspective(img, self.perspective_M_inv, (img.shape[1], img.shape[0]))

    def _reclassify_masks_via_bev(self, side_masks, side_lines=None):
        """Correct left/right lane mask classification using bird's eye view centroids.

        In perspective space the x_center heuristic fails when the vehicle is
        heading into a curve: the geometrically-right lane can appear on the
        LEFT half of the camera frame.  In BEV (after homography warp) the x
        position is metrically correct and unambiguous regardless of heading:
        the left lane mask always has a smaller BEV x-centroid than the right.

        Handles BOTH the two-mask case (swap detection) and the single-mask
        case (wrong-side detection):
          - Two masks present: swap if left_bev_cx > right_bev_cx.
          - One mask present:  move to the correct side based on BEV x-centroid
            relative to the BEV image centre.  This fixes single-line mode
            misclassifications where the only visible line is near the image
            centre in perspective but clearly on one side in BEV.

        Returns (corrected_side_masks, corrected_side_lines, was_swapped).
        Falls back without modification when BEV is not calibrated, masks are
        empty, or an exception occurs.
        """
        if not getattr(self, 'perspective_initialized', False):
            return side_masks, side_lines, False
        if not isinstance(side_masks, dict):
            return side_masks, side_lines, False

        left_mask = side_masks.get('left')
        right_mask = side_masks.get('right')
        has_left = isinstance(left_mask, np.ndarray) and left_mask.size > 0 and np.any(left_mask)
        has_right = isinstance(right_mask, np.ndarray) and right_mask.size > 0 and np.any(right_mask)

        if not has_left and not has_right:
            return side_masks, side_lines, False

        try:
            # Pick any mask to get shape
            ref_mask = left_mask if has_left else right_mask
            h, w = ref_mask.shape[:2]
            warp_size = (w, h)
            min_area = max(2, int(w * h * 0.0002))
            bev_center = w / 2.0

            def _bev_cx(mask):
                bev = cv2.warpPerspective(mask.astype(np.uint8), self.perspective_M, warp_size)
                cols = np.where(bev > 0)[1]
                return float(np.mean(cols)) if len(cols) >= min_area else None

            if has_left and has_right:
                # ── Two-mask case: detect swap ────────────────────────────────
                lcx = _bev_cx(left_mask)
                rcx = _bev_cx(right_mask)
                self._last_bev_reclassify_debug = {
                    'case': 'two_masks',
                    'L_bev_cx': round(lcx, 1) if lcx is not None else None,
                    'R_bev_cx': round(rcx, 1) if rcx is not None else None,
                    'bev_center': round(bev_center, 1),
                    'swapped': False,
                }
                if lcx is None or rcx is None:
                    return side_masks, side_lines, False
                if lcx > rcx:
                    self._last_bev_reclassify_debug['swapped'] = True
                    corrected_masks = {'left': right_mask, 'right': left_mask}
                    corrected_lines = side_lines
                    if isinstance(side_lines, dict):
                        corrected_lines = {
                            'left': side_lines.get('right'),
                            'right': side_lines.get('left'),
                        }
                    return corrected_masks, corrected_lines, True
                return side_masks, side_lines, False

            else:
                # ── Single-mask case: verify side via BEV centroid ────────────
                present_side = 'left' if has_left else 'right'
                present_mask = left_mask if has_left else right_mask
                cx = _bev_cx(present_mask)
                correct_side = None if cx is None else ('left' if cx < bev_center else 'right')
                swapped = correct_side is not None and correct_side != present_side
                self._last_bev_reclassify_debug = {
                    'case': 'single_mask',
                    'present_side': present_side,
                    'bev_cx': round(cx, 1) if cx is not None else None,
                    'bev_center': round(bev_center, 1),
                    'correct_side': correct_side,
                    'swapped': swapped,
                }
                if cx is None:
                    return side_masks, side_lines, False
                if swapped:
                    corrected_masks = {'left': None, 'right': None}
                    corrected_masks[correct_side] = present_mask
                    corrected_lines = {'left': None, 'right': None}
                    if isinstance(side_lines, dict):
                        corrected_lines[correct_side] = side_lines.get(present_side)
                    elif side_lines is None:
                        corrected_lines = None
                    return corrected_masks, corrected_lines, True
                return side_masks, side_lines, False

        except Exception:
            pass

        self._last_bev_reclassify_debug = {}
        return side_masks, side_lines, False

    def _estimate_ground_distance_cm(self, box_normalized, img_w, img_h):
        """Estimate metric forward distance to an object from its bounding box.

        Projects the bottom-center of the bounding box through the calibrated
        BEV homography (perspective_M) to obtain a metric ground-plane position.
        Returns (lateral_cm, forward_cm) relative to the camera, or None if
        the BEV is not yet initialized.

        lateral_cm: positive = object to the right of image center.
        forward_cm: distance ahead along the road direction.

        Returns None if BEV is not yet initialized. The former fallback that
        applied horizontal px_per_cm to vertical displacement was removed: that
        approach is geometrically incorrect because perspective projection makes
        vertical and horizontal pixel scales fundamentally different.
        """
        try:
            y1, x1, y2, x2 = [float(v) for v in box_normalized]
            # Bottom-center of the box = approximate ground contact point
            base_x_px = ((x1 + x2) / 2.0) * float(img_w)
            base_y_px = y2 * float(img_h)

            if self.perspective_initialized and self.bev_cm_per_px > 0.0:
                pt = np.array([[[base_x_px, base_y_px]]], dtype=np.float32)
                bev_pt = cv2.perspectiveTransform(pt, self.perspective_M)
                bev_x = float(bev_pt[0][0][0])
                bev_y = float(bev_pt[0][0][1])
                # In BEV: y=img_h is the bottom (close to car), y=0 is the top (far).
                # bev_cm_per_px is updated per frame from actual lane detections and
                # applies equally to both axes in the orthorectified BEV space.
                forward_cm = (float(img_h) - bev_y) * self.bev_cm_per_px
                center_bev_x = float(img_w) / 2.0
                lateral_cm = (bev_x - center_bev_x) * self.bev_cm_per_px
                return max(0.0, round(forward_cm, 1)), round(lateral_cm, 1)

            # No fallback: applying horizontal px_per_cm to vertical pixel displacement
            # is geometrically incorrect (perspective projection makes vertical and
            # horizontal pixel scales fundamentally different). Return None to signal
            # that no reliable metric estimate is available without BEV calibration.

        except (TypeError, ValueError, cv2.error):
            pass
        return None

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
            normalized_config = dict(config)
            if 'detection_mode' in normalized_config:
                normalized_config['detection_mode'] = self._normalize_detection_mode(normalized_config['detection_mode'])

            # String parameters (URLs, mode names) - do NOT convert to number
            string_params = {'detection_mode', 'hybridnets_server_url', 'supercombo_server_url'}
            # Integer parameters - must be int for OpenCV kernels, thresholds, etc.
            int_params = {'blur_kernel', 'morph_kernel', 'canny_low', 'canny_high',
                         'hough_threshold', 'hough_min_line_length', 'hough_max_line_gap',
                         'white_h_min', 'white_h_max', 'white_s_min', 'white_s_max',
                         'white_v_min', 'white_v_max', 'yellow_h_min', 'yellow_h_max',
                         'yellow_s_min', 'yellow_s_max', 'yellow_v_min', 'yellow_v_max',
                         'dilate_kernel', 'use_dilation',
                         'binary_threshold', 'binary_threshold_retry', 'line_angle_filter',
                         'line_merge_distance', 'lstr_model_size', 'stream_debug_view',
                         'stream_debug_fps', 'stream_debug_quality', 'use_clahe',
                         'use_adaptive_white', 'use_gradient_fallback', 'clahe_grid_size',
                         'adaptive_white_tophat_kernel', 'adaptive_white_tophat_min_threshold',
                         'use_local_adaptive_threshold', 'local_adaptive_block_size',
                         'use_auto_threshold', 'auto_threshold_percentile',
                         'auto_threshold_min', 'auto_threshold_max',
                         'integral_reset_interval', 'hybridnets_jpeg_quality',
                         'supercombo_jpeg_quality', 'brightness',
                         'use_component_filter', 'component_min_area', 'component_min_height',
                         'hough_max_lines_per_side',
                         'use_swept_path', 'curve_speed_reduction',
                         'curve_enter_frames', 'curve_confirm_frames', 'curve_exit_frames', 'curve_vp_confirm_frames',
                         'curve_two_line_confirm_frames', 'curve_two_line_hold_frames',
                         'use_noise_filter', 'noise_max_hough_lines',
                         'noise_max_reject_frames',
                         'use_curve_recovery', 'recovery_max_steer_frames',
                         'use_kalman_filter', 'kalman_max_prediction_frames',
                         'ai_local_use_mask_geometry', 'stanley_use_measured_speed'}
            
            params = ['base_speed', 'max_speed', 'min_speed', 'max_error_px', 'kp', 'ki', 'kd', 'smoothing_factor',
                     'dead_zone_ratio', 'max_steering', 'lookahead', 'integral_reset_interval',
                     'wheelbase', 'ff_weight', 'curvature_threshold',
                     'use_stanley', 'stanley_k', 'stanley_k_soft', 'stanley_k_d_yaw',
                     'stanley_k_d_steer', 'stanley_k_ag', 'stanley_speed_to_mps',
                     'stanley_measured_speed_to_mps', 'stanley_measured_speed_timeout',
                     'stanley_use_measured_speed',
                     'stanley_curve_min_confidence', 'stanley_psi_ss_max_deg',
                     'stanley_deadband_crosstrack_m',
                     'stanley_deadband_crosstrack_px', 'stanley_deadband_heading_deg',
                     'stanley_deadband_output_deg',
                     'single_line_offset_factor',
                     'roi_height_start', 'roi_height_end',
                     'roi_width_margin_top', 'roi_width_margin_bottom',
                     'white_h_min', 'white_h_max', 'white_s_min', 'white_s_max',
                     'white_v_min', 'white_v_max', 'yellow_h_min', 'yellow_h_max',
                     'yellow_s_min', 'yellow_s_max', 'yellow_v_min', 'yellow_v_max',
                     'brightness', 'contrast', 'blur_kernel', 'morph_kernel',
                     'dilate_kernel', 'use_dilation',
                     'canny_low', 'canny_high', 'hough_threshold', 'hough_min_line_length',
                     'hough_max_line_gap',
                     # BFMC-style parameters
                     'binary_threshold', 'binary_threshold_retry',
                     'line_angle_filter', 'line_merge_distance',
                     # Adaptive lighting parameters
                     'use_clahe', 'clahe_clip_limit', 'clahe_grid_size',
                     'use_adaptive_white', 'adaptive_white_percentile', 'adaptive_white_min_threshold',
                     'adaptive_white_tophat_kernel', 'adaptive_white_tophat_min_threshold',
                     'use_gradient_fallback', 'gradient_percentile',
                     'use_component_filter', 'component_min_area', 'component_min_height',
                     'component_min_aspect_ratio', 'component_min_angle',
                     'use_local_adaptive_threshold', 'local_adaptive_block_size',
                     'local_adaptive_c', 'hough_max_lines_per_side',
                     'use_auto_threshold', 'auto_threshold_percentile',
                     'auto_threshold_min', 'auto_threshold_max',
                     # Detection mode
                     'detection_mode', 'lstr_model_size',
                     # Debug streaming
                     'stream_debug_view', 'stream_debug_fps', 'stream_debug_quality', 'stream_debug_scale',
                     # HybridNets remote AI server
                     'hybridnets_server_url', 'hybridnets_jpeg_quality', 'hybridnets_timeout', 'local_ai_max_result_age',
                     'ai_local_use_mask_geometry',
                     # Supercombo remote AI server
                     'supercombo_server_url', 'supercombo_jpeg_quality', 'supercombo_timeout',
                     # Swept path / car geometry parameters
                     'use_swept_path', 'car_length', 'car_width', 'car_wheelbase_cm',
                     'car_front_overhang', 'car_rear_overhang',
                     'camera_to_front_axle', 'camera_to_rear_axle',
                     'lane_width_cm', 'line_width_cm',
                     'swept_path_margin_cm', 'curve_offset_gain', 'curve_inner_bias_cm',
                     'curve_confidence_threshold', 'min_clearance_warn_cm',
                     'curve_speed_reduction', 'single_line_radius_k',
                     # Hybrid curve system parameters
                     'bfmc_inner_lane_radius', 'bfmc_outer_lane_radius',
                     'bfmc_default_curve_radius',
                     'curve_enter_frames', 'curve_confirm_frames', 'curve_exit_frames',
                     'curve_vp_threshold', 'curve_vp_confirm_frames',
                     'curve_two_line_heading_threshold_deg',
                     'curve_two_line_heading_confidence',
                     'curve_two_line_confirm_frames',
                     'curve_two_line_hold_frames',
                     'curve_pre_position_gain', 'curve_exit_gain',
                     'curve_inner_line_steer_factor',
                     'single_line_swept_weight_entering', 'single_line_swept_weight_in_curve',
                     'single_line_swept_weight_exiting',
                     # Noise filter parameters
                     'use_noise_filter', 'noise_max_hough_lines',
                     'noise_max_error_jump_px', 'noise_max_steer_jump_deg',
                     'noise_max_reject_frames',
                     # Curve recovery parameters
                     'use_curve_recovery', 'recovery_max_steer_frames',
                     'recovery_reverse_speed', 'recovery_reverse_time_min',
                     'recovery_reverse_time_max', 'recovery_reverse_steer_scale',
                     'recovery_pre_turn_time', 'recovery_realign_time',
                     'recovery_error_shrink_ratio',
                     # Kalman filter parameters
                     'use_kalman_filter', 'kalman_process_noise',
                     'kalman_measurement_noise', 'kalman_max_prediction_frames',
                     'kalman_prediction_damping']
            
            applied = []
            for param in params:
                if param in normalized_config:
                    value = normalized_config[param]
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
            if any(p in normalized_config for p in pid_params):
                self.pid.Kp = self.kp
                self.pid.Ki = self.ki
                self.pid.Kd = self.kd
                self.pid.max_steering = self.max_steering
                self.pid.tolerance = self.dead_zone_ratio
                self.pid.integral_reset_interval = self.integral_reset_interval
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;96mPID\033[0m - Updated: Kp={self.kp} Ki={self.ki} Kd={self.kd} max={self.max_steering}")
            
            # Sync Stanley controller with updated parameters
            stanley_params = [
                'use_stanley', 'stanley_k', 'stanley_k_soft', 'stanley_k_d_yaw',
                'stanley_k_d_steer', 'stanley_k_ag', 'stanley_speed_to_mps',
                'stanley_measured_speed_to_mps', 'stanley_measured_speed_timeout',
                'stanley_use_measured_speed',
                'stanley_curve_min_confidence', 'stanley_psi_ss_max_deg',
                'stanley_deadband_crosstrack_m',
                'stanley_deadband_crosstrack_px', 'stanley_deadband_heading_deg',
                'stanley_deadband_output_deg', 'max_steering'
            ]
            if any(p in normalized_config for p in stanley_params):
                self.stanley.k = self.stanley_k
                self.stanley.k_soft = self.stanley_k_soft
                self._stanley_base_k = float(self.stanley_k)
                self._stanley_base_k_soft = float(self.stanley_k_soft)
                self._stanley_sched_k = float(self.stanley_k)
                self._stanley_sched_k_soft = float(self.stanley_k_soft)
                self.stanley.k_d_yaw = self.stanley_k_d_yaw
                self.stanley.k_d_steer = self.stanley_k_d_steer
                self.stanley.max_steering = self.max_steering
                if float(getattr(self, 'stanley_deadband_crosstrack_m', 0.0) or 0.0) > 0.0:
                    self.stanley.crosstrack_deadband = max(
                        0.0, float(self.stanley_deadband_crosstrack_m)
                    )
                else:
                    # Legacy fallback: if only px deadband is configured, preserve it
                    # for compatibility until a lane scale is available.
                    self.stanley.crosstrack_deadband = max(
                        0.0, float(self.stanley_deadband_crosstrack_px)
                    )
                self.stanley.crosstrack_deadband_px = self.stanley.crosstrack_deadband
                self.stanley.heading_deadband_rad = math.radians(
                    max(0.0, float(self.stanley_deadband_heading_deg))
                )
                self.stanley.output_deadband_deg = max(0.0, float(self.stanley_deadband_output_deg))
                self.lateral_mpc.output_deadband_deg = max(0.0, float(self.mpc_output_deadband_deg))
                mode_str = "STANLEY" if self._should_use_stanley_controller() else "PID"
                if self._is_ai_local_active():
                    mode_str = "STANLEY (forced in AI_LOCAL)"
                # print(
                #     f"\033[1;97m[ Line Following ] :\033[0m \033[1;95mSTANLEY\033[0m - "
                #     f"Mode: {mode_str} k={self.stanley_k} k_soft={self.stanley_k_soft} "
                #     f"k_d_yaw={self.stanley_k_d_yaw} k_d_steer={self.stanley_k_d_steer} "
                #     f"k_ag={self.stanley_k_ag} v_cmd={self.stanley_speed_to_mps} "
                #     f"v_meas={self.stanley_measured_speed_to_mps} use_meas={bool(self.stanley_use_measured_speed)} "
                #     f"db_cte_m={self.stanley_deadband_crosstrack_m} db_cte_px={self.stanley_deadband_crosstrack_px} "
                #     f"db_heading={self.stanley_deadband_heading_deg} db_out={self.stanley_deadband_output_deg}"
                # )

            # Log feed-forward parameter changes
            ff_params = ['wheelbase', 'ff_weight', 'curvature_threshold']
            if any(p in normalized_config for p in ff_params):
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;95mFF\033[0m - Updated: wheelbase={self.wheelbase} ff_weight={self.ff_weight} curv_thresh={self.curvature_threshold}")

            # Sync Kalman filter parameters
            kalman_params = ['use_kalman_filter', 'kalman_process_noise',
                             'kalman_measurement_noise', 'kalman_max_prediction_frames',
                             'kalman_prediction_damping']
            if any(p in normalized_config for p in kalman_params):
                self._init_error_kalman()
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;95mKALMAN\033[0m - "
                      f"enabled={bool(self.use_kalman_filter)} q={self.kalman_process_noise:.4f} "
                      f"r={self.kalman_measurement_noise:.2f} max_pred={self.kalman_max_prediction_frames} "
                      f"damp={self.kalman_prediction_damping:.2f}")
            
            # Log swept path / car geometry parameter changes
            swept_params = ['use_swept_path', 'car_length', 'car_width', 'car_wheelbase_cm',
                           'car_front_overhang', 'car_rear_overhang', 'lane_width_cm',
                           'swept_path_margin_cm', 'curve_offset_gain', 'curve_inner_bias_cm',
                           'curve_confidence_threshold', 'min_clearance_warn_cm', 'curve_speed_reduction']
            if any(p in normalized_config for p in swept_params):
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;95mSWEPT PATH\033[0m - Updated: "
                      f"enabled={self.use_swept_path} car={self.car_length}x{self.car_width}cm "
                      f"wb={self.car_wheelbase_cm}cm ovh_f={self.car_front_overhang} ovh_r={self.car_rear_overhang} "
                      f"lane={self.lane_width_cm}cm margin={self.swept_path_margin_cm}cm "
                      f"gain={self.curve_offset_gain} conf_thr={self.curve_confidence_threshold}")
            
            # Log hybrid curve system parameter changes
            hybrid_params = ['bfmc_inner_lane_radius', 'bfmc_outer_lane_radius',
                            'bfmc_default_curve_radius', 'curve_enter_frames',
                            'curve_confirm_frames', 'curve_exit_frames',
                            'curve_vp_threshold', 'curve_vp_confirm_frames',
                            'curve_two_line_heading_threshold_deg',
                            'curve_two_line_heading_confidence',
                            'curve_two_line_confirm_frames',
                            'curve_two_line_hold_frames',
                            'curve_pre_position_gain', 'curve_exit_gain',
                            'curve_inner_line_steer_factor']
            if any(p in normalized_config for p in hybrid_params):
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;95mHYBRID CURVE\033[0m - Updated: "
                      f"BFMC_R: inner={self.bfmc_inner_lane_radius} outer={self.bfmc_outer_lane_radius} "
                      f"default={self.bfmc_default_curve_radius} "
                      f"FSM: enter={self.curve_enter_frames} confirm={self.curve_confirm_frames} "
                      f"exit={self.curve_exit_frames} vp_thr={self.curve_vp_threshold} "
                      f"vp_confirm={self.curve_vp_confirm_frames} "
                      f"2l_hint_deg={self.curve_two_line_heading_threshold_deg} "
                      f"2l_hint_conf={self.curve_two_line_heading_confidence} "
                      f"2l_confirm={self.curve_two_line_confirm_frames} "
                      f"2l_hold={self.curve_two_line_hold_frames} "
                      f"pre_gain={self.curve_pre_position_gain} exit_gain={self.curve_exit_gain} "
                      f"inner_steer={self.curve_inner_line_steer_factor}")
            
            # Log curve recovery parameter changes
            recovery_params = ['use_curve_recovery', 'recovery_max_steer_frames',
                              'recovery_reverse_speed', 'recovery_reverse_time_min',
                              'recovery_reverse_time_max', 'recovery_reverse_steer_scale',
                              'recovery_realign_time', 'recovery_error_shrink_ratio']
            if any(p in normalized_config for p in recovery_params):
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;95mRECOVERY\033[0m - Updated: "
                      f"enabled={self.use_curve_recovery} max_frames={self.recovery_max_steer_frames} "
                      f"rev_speed={self.recovery_reverse_speed} "
                      f"rev_time={self.recovery_reverse_time_min}-{self.recovery_reverse_time_max}s "
                      f"steer_scale={self.recovery_reverse_steer_scale} realign={self.recovery_realign_time}s "
                      f"shrink_ratio={self.recovery_error_shrink_ratio}")
            
            # Log mode change and reset PID state
            if 'detection_mode' in normalized_config:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;96mMODE\033[0m - Changed to: {normalized_config['detection_mode']}")
                # Reset PID state to prevent corruption between modes
                self._reset_pid_state()
            
            # Reload LSTR model if size changed
            if 'lstr_model_size' in normalized_config:
                self._init_lstr_detector()
            
            # Reconnect HybridNets client if server URL changed
            if 'hybridnets_server_url' in normalized_config:
                self._stop_hybridnets_client()
                self._init_hybridnets_client()
            
            # Reconnect Supercombo client if server URL changed
            if 'supercombo_server_url' in normalized_config:
                self._stop_supercombo_client()
                self._init_supercombo_client()
            
            # Start/stop remote AI clients based on mode
            if 'detection_mode' in normalized_config:
                if normalized_config['detection_mode'] == DetectionMode.HYBRIDNETS.value:
                    self._start_hybridnets_client()
                    self._stop_supercombo_client()
                elif normalized_config['detection_mode'] == DetectionMode.SUPERCOMBO.value:
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

    def _sanitize_log_value(self, value, depth=0):
        """Convert runtime state into JSON-safe, compact log values."""
        if depth > 8:
            return str(type(value).__name__)
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        if isinstance(value, dict):
            sanitized = {}
            for key, item in value.items():
                sanitized[str(key)] = self._sanitize_log_value(item, depth + 1)
            return sanitized
        if isinstance(value, (list, tuple, deque)):
            return [self._sanitize_log_value(item, depth + 1) for item in value]
        return str(value)

    def _set_frame_trace(self, trace):
        """Store the latest per-frame reasoning snapshot for auto-run logging."""
        trace_dict = trace if isinstance(trace, dict) else {"value": trace}
        self._last_frame_trace = self._sanitize_log_value(trace_dict)

    def _reset_auto_run_log(self, state_message):
        """Reset the persistent auto-run text log when entering AUTO mode."""
        log_dir = os.path.dirname(self.auto_run_log_path)
        os.makedirs(log_dir, exist_ok=True)
        self._auto_run_log_enabled = True
        self._auto_run_log_frame_idx = 0
        self._last_frame_trace = {}
        header = {
            "log": "line_following_auto_last_run",
            "reset_timestamp": round(time.time(), 3),
            "state_message": str(state_message),
            "detection_mode": str(getattr(self, "detection_mode", "")),
        }
        with open(self.auto_run_log_path, "w", encoding="utf-8") as log_file:
            log_file.write("=== AUTO RUN RESET ===\n")
            log_file.write(json.dumps(header, indent=2, sort_keys=True))
            log_file.write("\n\n")

    def _append_auto_run_log_entry(self, title, payload):
        """Append one human-readable JSON block to the current auto-run log."""
        if not self._auto_run_log_enabled:
            return
        log_dir = os.path.dirname(self.auto_run_log_path)
        os.makedirs(log_dir, exist_ok=True)
        with open(self.auto_run_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"=== {title} ===\n")
            log_file.write(json.dumps(self._sanitize_log_value(payload), indent=2, sort_keys=True))
            log_file.write("\n\n")

    def _build_tracking_log_snapshot(self, include_route_reference=False):
        """Return a rich tracking/navigation snapshot suitable for manual logs."""
        ts = getattr(self, "_tracking_state", None)
        if ts is None:
            return None

        snapshot = {}
        if hasattr(ts, "snapshot"):
            try:
                raw_snapshot = ts.snapshot()
                if isinstance(raw_snapshot, dict):
                    snapshot = raw_snapshot
            except Exception:
                snapshot = {}

        def _read(name, default=None):
            if name in snapshot:
                return snapshot.get(name, default)
            return getattr(ts, name, default)

        pose_x = float(_read("x", 0.0) or 0.0)
        pose_y = float(_read("y", 0.0) or 0.0)
        pose_yaw = float(_read("yaw", 0.0) or 0.0)
        raw_x = float(_read("raw_x", pose_x) or pose_x)
        raw_y = float(_read("raw_y", pose_y) or pose_y)
        raw_yaw = float(_read("raw_yaw", pose_yaw) or pose_yaw)
        matched_x = float(_read("matched_x", pose_x) or pose_x)
        matched_y = float(_read("matched_y", pose_y) or pose_y)
        matched_yaw = float(_read("matched_yaw", pose_yaw) or pose_yaw)
        heading_rad = float(_read("heading_rad", 0.0) or 0.0)
        path_psi = float(_read("path_psi", 0.0) or 0.0)
        steer_rad = float(_read("steer_rad", 0.0) or 0.0)
        route_queue = list(_read("route_queue", []) or [])
        current_zone_ids = list(_read("current_zone_ids", []) or [])
        current_zone_types = list(_read("current_zone_types", []) or [])

        tracking_log = {
            "initialized": bool(getattr(ts, "initialized", False)),
            "imu_received": bool(getattr(ts, "imu_received", False)),
            "pose": {
                "x_m": round(pose_x, 4),
                "y_m": round(pose_y, 4),
                "yaw_deg": round(math.degrees(pose_yaw), 2),
                "matched_x_m": round(matched_x, 4),
                "matched_y_m": round(matched_y, 4),
                "matched_yaw_deg": round(math.degrees(matched_yaw), 2),
                "raw_x_m": round(raw_x, 4),
                "raw_y_m": round(raw_y, 4),
                "raw_yaw_deg": round(math.degrees(raw_yaw), 2),
            },
            "dead_reckoning": {
                "speed_mps": round(float(_read("speed_mps", 0.0) or 0.0), 4),
                "steer_deg": round(math.degrees(steer_rad), 3),
                "dr_yaw_correction_deg": round(float(getattr(ts, "last_yaw_correction_deg", 0.0) or 0.0), 4),
                "dr_yaw_cam_hint_deg": round(float(getattr(ts, "last_cam_yaw_hint_deg", 0.0) or 0.0), 2),
                "dr_yaw_cam_conf": round(float(getattr(ts, "last_cam_yaw_hint_conf", 0.0) or 0.0), 3),
            },
            "path_tracking": {
                "error_m": round(float(_read("error_m", 0.0) or 0.0), 5),
                "raw_lateral_error_m": round(float(_read("raw_lateral_error_m", 0.0) or 0.0), 5),
                "heading_rad": round(heading_rad, 5),
                "heading_deg": round(math.degrees(heading_rad), 3),
                "path_psi_deg": round(math.degrees(path_psi), 3),
                "path_kappa": round(float(_read("path_kappa", 0.0) or 0.0), 5),
                "map_match_error_m": round(float(_read("map_match_error_m", 0.0) or 0.0), 5),
                "lane_measurement_reliable": bool(_read("lane_measurement_reliable", False)),
                "camera_lateral_correction_m": round(float(_read("camera_lateral_correction_m", 0.0) or 0.0), 5),
                "waypoint_mode": bool(_read("waypoint_mode_active", False)),
                "wp_idx": _read("wp_idx", None),
                "target_idx": _read("target_idx", _read("wp_idx", None)),
                "node_attr": int(_read("node_attr", 0) or 0),
            },
            "navigation": {
                "route_active": bool(_read("route_active", False)),
                "route_id": _read("route_id", None),
                "route_source": str(_read("route_source", "none") or "none"),
                "route_progress": round(float(_read("route_progress", 0.0) or 0.0), 5),
                "route_completed": bool(_read("route_completed", False)),
                "route_replans": int(_read("route_replans", 0) or 0),
                "destination_node_id": _read("destination_node_id", None),
                "destination_label": _read("destination_label", None),
                "destination_point": _read("destination_point", None),
                "route_queue": route_queue,
                "current_node_id": _read("current_node_id", None),
                "current_node_attr": int(_read("current_node_attr", 0) or 0),
                "upcoming_node_id": _read("upcoming_node_id", None),
                "upcoming_node_attr": int(_read("upcoming_node_attr", 0) or 0),
            },
            "semantics": {
                "maneuver_type": str(_read("maneuver_type", "none") or "none"),
                "next_semantic_id": _read("next_semantic_id", None),
                "next_semantic_type": _read("next_semantic_type", None),
                "next_semantic_label": _read("next_semantic_label", None),
                "next_semantic_distance_m": (
                    round(float(_read("next_semantic_distance_m", 0.0)), 5)
                    if _read("next_semantic_distance_m", None) is not None else None
                ),
                "expected_control_type": _read("expected_control_type", None),
                "current_zone_ids": current_zone_ids,
                "current_zone_types": current_zone_types,
            },
            "relocalization": {
                "mode": str(_read("relocalization_mode", "map_match") or "map_match"),
                "last_source": str(_read("last_relocalization_source", "map_match") or "map_match"),
                "last_error_m": round(float(_read("last_relocalization_error_m", 0.0) or 0.0), 5),
            },
        }

        if include_route_reference:
            route_points = list(_read("route_points", []) or [])
            tracking_log["route_reference"] = {
                "route_points_preview": route_points,
                "route_points_count": len(route_points),
                "available_destinations": list(_read("available_destinations", []) or []),
                "map_metadata": dict(_read("map_metadata", {}) or {}),
            }

        return tracking_log

    def _reset_manual_run_log(self, state_message):
        """Reset (overwrite) the manual-run log when entering MANUAL mode.

        Always replaces the previous file so the log only ever contains data
        from the most recent manual session — same behaviour as the auto-run log.
        """
        log_dir = os.path.dirname(self.manual_run_log_path)
        os.makedirs(log_dir, exist_ok=True)
        self._manual_run_log_enabled = True
        self._manual_run_log_frame_idx = 0
        header = {
            "log": "line_following_manual_last_run",
            "reset_timestamp": round(time.time(), 3),
            "state_message": str(state_message),
            "detection_mode": str(getattr(self, "detection_mode", "")),
            "tracking_reference": self._build_tracking_log_snapshot(include_route_reference=True),
            "note": (
                "dashboard_command = speed/steer sent by PC web dashboard (raw units). "
                "nucleo_feedback = actual values reported by Nucleo encoder/servo. "
                "speed units: x10 cm/s (divide by 10 for cm/s). "
                "steer units: x10 deg (divide by 10 for degrees). "
                "tracking_reference.route_points_preview is a decimated preview of the active route; "
                "per-frame current_node/upcoming_node + pose show exact progression over time."
            ),
        }
        with open(self.manual_run_log_path, "w", encoding="utf-8") as f:
            f.write("=== MANUAL RUN RESET ===\n")
            f.write(json.dumps(header, indent=2, sort_keys=True))
            f.write("\n\n")

    def _log_manual_run_frame(self):
        """Append one per-frame sensor snapshot to the manual-run log.

        Captures everything relevant during manual driving:
        - IMU yaw / roll / pitch and computed yaw rate
        - Nucleo encoder speed and servo steer feedback
        - Last speed / steer command received from the PC dashboard
        - Dead reckoning, map-matched pose, nodes, route, semantics, relocalization
        """
        if not self._manual_run_log_enabled:
            return
        self._manual_run_log_frame_idx += 1
        speed_cms = round(abs(float(self._measured_speed)) * 0.1, 2)  # x10 cm/s → cm/s
        steer_deg = round(float(self._measured_steer) * 0.1, 2)       # x10 deg → deg
        cmd_speed_cms = round(abs(float(self._last_cmd_speed)) * 0.1, 2)
        cmd_steer_deg = round(float(self._last_cmd_steer) * 0.1, 2)
        latest_observation = self._lane_observation_history[-1] if self._lane_observation_history else None
        payload = {
            "frame_index": self._manual_run_log_frame_idx,
            "wall_time": round(time.time(), 3),
            "state_change_message": self._last_state_change_message,
            "line_following_active": bool(self.is_line_following_active),
            "detection_mode": str(self.detection_mode),
            "imu": {
                "yaw_deg": round(float(self._last_yaw), 3),
                "roll_deg": round(float(self._last_imu_roll), 3),
                "pitch_deg": round(float(self._last_imu_pitch), 3),
                "yaw_rate_degs": round(math.degrees(float(self._yaw_rate)), 3),
            },
            "nucleo_feedback": {
                "speed_x10": round(float(self._measured_speed), 1),
                "speed_cms": speed_cms,
                "steer_x10": round(float(self._measured_steer), 1),
                "steer_deg": steer_deg,
            },
            "dashboard_command": {
                "speed_x10": round(float(self._last_cmd_speed), 1),
                "speed_cms": cmd_speed_cms,
                "steer_x10": round(float(self._last_cmd_steer), 1),
                "steer_deg": cmd_steer_deg,
            },
            "manual_control_context": {
                "mission_state": getattr(self, "_mission_state_name", None),
                "control_policy_mode": getattr(self, "_control_policy_mode", None),
                "control_authority": getattr(self, "_control_authority", None),
                "active_maneuver": getattr(self, "_active_maneuver_name", None),
                "planner_priority": bool(getattr(self, "_planner_priority_active", False)),
                "safety_stop_reason": getattr(self, "_safety_stop_reason", None) or None,
                "maneuver_notes": getattr(self, "_last_maneuver_notes", None) or None,
            },
            "nucleo_request": getattr(self, "_last_requested_motor_command", None),
            "actuator_status": getattr(self, "_last_actuator_status", None),
            "local_perception_status": getattr(self, "_last_local_perception_status", None),
            "local_lane_payload": self._build_local_lane_payload_log(),
            "lane_observation": latest_observation,
            "frame_trace": getattr(self, "_last_frame_trace", None),
            "controller_state": {
                "heading_error_deg": round(math.degrees(float(getattr(self, "_heading_error", 0.0) or 0.0)), 3),
                "measured_yaw_rate": round(float(self._yaw_rate), 5),
                "measured_speed": round(float(self._measured_speed), 3),
                "measured_speed_mps": round(
                    abs(float(self._measured_speed)) * float(self.stanley_measured_speed_to_mps), 4
                ),
                "measured_steer": round(float(self._measured_steer), 3),
                "measured_steer_delta": round(float(getattr(self, "_measured_steer_delta", 0.0) or 0.0), 3),
                "stanley_speed_mps": round(float(getattr(self, "_stanley_last_speed_mps", 0.0) or 0.0), 4),
                "stanley_speed_source": getattr(self, "_stanley_last_speed_source", None),
                "stanley_error_m": round(float(getattr(self, "_stanley_last_error_m", 0.0) or 0.0), 4),
                "stanley_px_per_cm": round(float(getattr(self, "_stanley_last_px_per_cm", 0.0) or 0.0), 4),
                "theta_dmae_deg": round(float(getattr(self, "_theta_dmae_deg", 0.0) or 0.0), 4),
                "offset_dma_m": round(float(getattr(self, "_offset_dma_m", 0.0) or 0.0), 5),
                "heading_hint_rad": round(float(getattr(self, "_local_ai_heading_hint_rad", 0.0) or 0.0), 5),
                "heading_hint_confidence": round(float(getattr(self, "_local_ai_heading_hint_confidence", 0.0) or 0.0), 4),
                "heading_hint_source": getattr(self, "_local_ai_heading_hint_source", None),
                "road_type_class": getattr(self, "_local_ai_road_type_class", None),
                "road_type_confidence": round(float(getattr(self, "_local_ai_road_type_confidence", 0.0) or 0.0), 4),
                "lead_distance_class": getattr(self, "_local_ai_lead_distance_class", None),
                "lead_distance_confidence": round(float(getattr(self, "_local_ai_lead_distance_confidence", 0.0) or 0.0), 4),
                "curve_state": getattr(self, "_curve_state", None),
                "curve_direction": getattr(self, "_curve_direction", None),
                "frames_without_line": int(self.frames_without_line),
                "last_seen_side": self.last_seen_side,
            },
            "tracking": self._build_tracking_log_snapshot(include_route_reference=False),
        }
        log_dir = os.path.dirname(self.manual_run_log_path)
        os.makedirs(log_dir, exist_ok=True)
        with open(self.manual_run_log_path, "a", encoding="utf-8") as f:
            f.write(f"=== FRAME {self._manual_run_log_frame_idx:04d} ===\n")
            f.write(json.dumps(payload, indent=2, sort_keys=True))
            f.write("\n\n")

    def _write_mask_debug_entry(self, debug_info, steering_angle, speed):
        """Write one compact line to the always-on mask classification debug log.

        Format (one line per frame):
        F000123 HH:MM:SS.mmm | raw:L(cx=NNN,area=NNN,src=X) R(...)
                              | prep:detected=X resolved=Y via=Z  bev_swap=Y/N
                              | bev_case=single cx=NNN center=320 correct=R
                              | guide:MODE detected=LR err=X.XXXm hdg=X.X°
                              | OUT: steer=+X.X° speed=XX.X
        """
        if not self._mask_debug_log_enabled:
            return
        try:
            import datetime
            self._mask_debug_frame_idx += 1
            ts = datetime.datetime.now().strftime('%H:%M:%S.') + \
                 f"{datetime.datetime.now().microsecond // 1000:03d}"
            fi = self._mask_debug_frame_idx

            # ── RAW YOLO ──────────────────────────────────────────────────────
            raw = getattr(self, '_last_yolo_raw_debug', {})
            raw_parts = []
            for side in ('left', 'right'):
                r = raw.get(side)
                if r:
                    src_short = str(r.get('src', '?'))[:3]
                    raw_parts.append(
                        f"{side[0].upper()}(cx={r.get('px_cx','?')},a={r.get('px_area','?')},s={src_short})"
                    )
                else:
                    raw_parts.append(f"{side[0].upper()}=none")
            raw_str = ' '.join(raw_parts)

            # ── PREP (mask_side_resolution from debug_info) ───────────────────
            det_side = debug_info.get('single_line_detected_side')
            res_side = debug_info.get('single_line_resolved_side')
            res_src  = debug_info.get('single_line_side_source')
            dup = debug_info.get('duplicate_line_collapse')
            dup_str = f" dup={dup.get('resolution_source','?')}" if isinstance(dup, dict) else ''
            if det_side is None and res_side is None:
                # Two-line or no-line mode — single-line resolution not applicable
                prep_str = f"2-line{dup_str}" if not dup_str else f"n/a{dup_str}"
            else:
                prep_str = f"det={det_side} res={res_side} via={res_src or '?'}{dup_str}"

            # ── BEV reclassify ────────────────────────────────────────────────
            bev = getattr(self, '_last_bev_reclassify_debug', {})
            bev_swap = debug_info.get('bev_reclassified', False)
            if bev:
                bev_case = bev.get('case', '?')
                bev_center_val = bev.get('bev_center', '?')
                if bev_case == 'two_masks':
                    lcx = bev.get('L_bev_cx', '?')
                    rcx = bev.get('R_bev_cx', '?')
                    bev_str = (
                        f"2mask Lcx={lcx} Rcx={rcx} center={bev_center_val} "
                        f"swap={'YES ←!' if bev_swap else 'no'}"
                    )
                else:
                    px_cx = bev.get('bev_cx', '?')
                    present = bev.get('present_side', '?')
                    correct = bev.get('correct_side', '?')
                    bev_str = (
                        f"1mask present={present} cx={px_cx} center={bev_center_val} "
                        f"correct={correct} swap={'YES ←!' if bev_swap else 'no'}"
                    )
            else:
                bev_str = f"bev=skipped swap={'YES ←!' if bev_swap else 'no'}"

            # ── GUIDANCE result ───────────────────────────────────────────────
            mguid = debug_info.get('local_mask_guidance') or {}
            guide_mode = mguid.get('guidance_mode', debug_info.get('control_mode', '?'))
            guide_sides = ','.join(mguid.get('detected_sides', []) or []) or '?'
            err_m = mguid.get('error_cm')
            if err_m is not None:
                err_str = f"err={err_m/100:.3f}m"
            else:
                err_m2 = mguid.get('error_px')
                err_str = f"err={err_m2:.0f}px" if err_m2 is not None else "err=?"
            # Read heading directly from the computed state (not in debug_info dict)
            _hdg_rad = getattr(self, '_heading_error', None)
            hdg_deg = math.degrees(float(_hdg_rad)) if _hdg_rad is not None else None
            hdg_str = f"hdg={hdg_deg:+.1f}°" if hdg_deg is not None else "hdg=?"
            guide_str = f"{guide_mode} sides={guide_sides} {err_str} {hdg_str}"

            # ── OUTPUT ────────────────────────────────────────────────────────
            steer_s = f"{steering_angle:+.1f}°" if steering_angle is not None else "None"
            speed_s = f"{speed:.1f}" if speed is not None else "?"

            # ── Write line ────────────────────────────────────────────────────
            line = (
                f"F{fi:06d} {ts}"
                f" | raw: {raw_str}"
                f" | prep: {prep_str}"
                f" | bev: {bev_str}"
                f" | guide: {guide_str}"
                f" | OUT steer={steer_s} v={speed_s}\n"
            )
            with open(self._mask_debug_log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as _exc:
            pass  # Never crash the control loop because of logging

    # ------------------------------------------------------------------
    # Calibration mode helpers
    # ------------------------------------------------------------------

    def _check_calib_mode(self):
        """Process incoming LaneCalibMode toggle messages."""
        msg = self.laneCalibModeSubscriber.receive()
        if msg is None:
            return
        active = str(msg).strip().lower() == "true"
        if active and not self._calib_mode_active:
            self._start_calib_mode()
        elif not active and self._calib_mode_active:
            self._stop_calib_mode()

    def _start_calib_mode(self):
        """Reset and open the calibration log file, activate calib mode."""
        self._calib_mode_active = True
        self._calib_log_frame_idx = 0
        log_dir = os.path.dirname(self._calib_log_path)
        os.makedirs(log_dir, exist_ok=True)
        header = {
            "log": "lane_calib_log",
            "start_timestamp": round(time.time(), 3),
            "detection_mode": str(getattr(self, "detection_mode", "")),
            "min_speed": float(self.min_speed),
            "note": (
                "manual_steer = real servo position (from Nucleo feedback). "
                "algorithm_steer = what Stanley would have commanded. "
                "Car moves at min_speed; steering is controlled manually by the user."
            ),
        }
        with open(self._calib_log_path, "w", encoding="utf-8") as f:
            f.write("=== CALIBRATION MODE START ===\n")
            f.write(json.dumps(header, indent=2, sort_keys=True))
            f.write("\n\n")
        print(
            f"\033[1;97m[ Line Following ] :\033[0m \033[1;93mCALIB\033[0m "
            f"- Mode STARTED. Log: {self._calib_log_path}"
        )

    def _stop_calib_mode(self):
        """Deactivate calib mode and write a footer to the log."""
        self._calib_mode_active = False
        footer = {
            "stop_timestamp": round(time.time(), 3),
            "total_frames": self._calib_log_frame_idx,
        }
        log_dir = os.path.dirname(self._calib_log_path)
        os.makedirs(log_dir, exist_ok=True)
        try:
            with open(self._calib_log_path, "a", encoding="utf-8") as f:
                f.write("=== CALIBRATION MODE END ===\n")
                f.write(json.dumps(footer, indent=2, sort_keys=True))
                f.write("\n\n")
        except Exception as exc:
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mCALIB\033[0m - Error writing footer: {exc}")
        print(
            f"\033[1;97m[ Line Following ] :\033[0m \033[1;93mCALIB\033[0m "
            f"- Mode STOPPED after {self._calib_log_frame_idx} frames."
        )

    def _log_calib_frame(self, computed_steering):
        """Append one frame entry to the calibration log.

        Records the manual steering (real servo position from Nucleo) versus
        the steering angle the algorithm would have computed from lane detection.
        Only runs when calibration mode is active; the old log is cleared on start.
        """
        if not self._calib_mode_active:
            return
        self._calib_log_frame_idx += 1
        # _measured_steer is in steer_x10 units (tenths of degrees)
        manual_steer_deg = float(self._measured_steer) / 10.0
        algo_steer_deg = float(computed_steering) if computed_steering is not None else None
        delta = round(algo_steer_deg - manual_steer_deg, 2) if algo_steer_deg is not None else None

        sl_proj = dict(getattr(self, '_last_single_line_projection_debug', {}) or {})
        lane_obs = self._lane_observation_history[-1] if self._lane_observation_history else {}
        heading_source = str(getattr(self, '_local_ai_heading_hint_source', 'unknown') or 'unknown')
        sl_side = (lane_obs.get('visible_side') if isinstance(lane_obs, dict) else None)
        if sl_side is None:
            if 'single_right' in heading_source:
                sl_side = 'right'
            elif 'single_left' in heading_source:
                sl_side = 'left'

        ref_left_deg = round(math.degrees(float(getattr(self, '_single_line_heading_ref_left', 0.0))), 2)
        ref_right_deg = round(math.degrees(float(getattr(self, '_single_line_heading_ref_right', 0.0))), 2)

        # Compact one-liner for quick log scanning
        heading_raw_deg = sl_proj.get('single_line_heading_raw_deg')
        ref_angle_deg = sl_proj.get('single_line_ref_angle_deg')
        delta_from_ref = sl_proj.get('single_line_heading_delta_from_ref_deg')
        is_outer_h = sl_proj.get('single_line_is_outer_heading')
        is_outer_line = sl_proj.get('single_line_outer_curve_line')
        is_outer_confirmed = sl_proj.get('single_line_outer_confirmed')
        curve_strength = sl_proj.get('single_line_curve_strength')
        is_transversal = bool(sl_proj.get('single_line_transversal_detected', False))
        sl_mode = sl_proj.get('single_line_mode', '')
        ctrl_curve_dir = int(getattr(self, '_curve_direction', 0) or 0)
        ctrl_curve_state = str(getattr(self, '_curve_state', 'STRAIGHT'))
        if is_transversal and sl_side is not None:
            summary = (
                f"*** TRANSVERSAL *** src={heading_source} | "
                f"side={sl_side} | "
                f"state={ctrl_curve_state}({ctrl_curve_dir:+d}) | "
                f"raw={heading_raw_deg:+.1f}° | "
                f"ref={ref_angle_deg:+.1f}° | "
                f"Δref={delta_from_ref:+.1f}° | "
                f"RECOVERY → {'RIGHT' if sl_side == 'right' else 'LEFT'} | "
                f"manual={manual_steer_deg:+.1f}° | algo={algo_steer_deg:+.1f}°"
            )
        elif sl_side is not None and heading_raw_deg is not None:
            summary = (
                f"src={heading_source} | "
                f"side={sl_side} | "
                f"state={ctrl_curve_state}({ctrl_curve_dir:+d}) | "
                f"raw={heading_raw_deg:+.1f}° | "
                f"ref={ref_angle_deg:+.1f}° | "
                f"Δref={delta_from_ref:+.1f}° | "
                f"outer_h={is_outer_h} | "
                f"outer_line={is_outer_line} | "
                f"confirmed={is_outer_confirmed} | "
                f"str={curve_strength:.2f} | "
                f"manual={manual_steer_deg:+.1f}° | "
                f"algo={algo_steer_deg:+.1f}°"
            )
        else:
            summary = (
                f"src={heading_source} | "
                f"state={ctrl_curve_state}({ctrl_curve_dir:+d}) | "
                f"manual={manual_steer_deg:+.1f}° | "
                f"algo={algo_steer_deg:+.1f}°"
            )

        payload = {
            "frame_index": self._calib_log_frame_idx,
            "wall_time": round(time.time(), 3),
            "heading_source": heading_source,
            "manual": {
                "steer_deg": round(manual_steer_deg, 2),
                "steer_x10_raw": round(float(self._measured_steer), 1),
            },
            "algorithm_would_compute": {
                "steer_deg": round(algo_steer_deg, 2) if algo_steer_deg is not None else None,
            },
            "delta_algorithm_minus_manual_deg": delta,
            "controller_state": {
                "stanley_error_m": round(float(self._stanley_last_error_m), 4),
                "stanley_px_per_cm": round(float(self._stanley_last_px_per_cm), 4),
                "heading_error_deg": round(math.degrees(self._heading_error), 3),
                "measured_speed": round(float(self._measured_speed), 3),
                "curve_state": self._curve_state,
                "curve_direction": self._curve_direction,
            },
            "single_line_classification": {
                "mode": sl_mode,
                "transversal_detected": is_transversal,
                "transversal_delta_deg": round(sl_proj.get('single_line_transversal_delta_deg', 0.0), 2) if is_transversal else None,
                "transversal_threshold_deg": round(sl_proj.get('single_line_transversal_threshold_deg', 0.0), 2) if is_transversal else None,
                "side": sl_side,
                "heading_raw_deg": round(heading_raw_deg, 2) if heading_raw_deg is not None else None,
                "ref_angle_deg": round(ref_angle_deg, 2) if ref_angle_deg is not None else None,
                "heading_delta_from_ref_deg": round(delta_from_ref, 2) if delta_from_ref is not None else None,
                "is_outer_heading": is_outer_h,
                "curve_strength": round(curve_strength, 3) if curve_strength is not None else None,
                "outer_curve_line": is_outer_line,
                "outer_confirmed": is_outer_confirmed,
                "outer_confirm_sources": sl_proj.get('single_line_outer_confirm_sources'),
                "curve_heading_bias_deg": sl_proj.get('single_line_curve_heading_bias_deg'),
                "outer_bias_px": sl_proj.get('single_line_outer_bias_px'),
                "heading_reliability": sl_proj.get('single_line_heading_reliability'),
                "ref_left_deg_stored": ref_left_deg,
                "ref_right_deg_stored": ref_right_deg,
            },
            "lane_observation": lane_obs,
            "single_line_shadow": self._last_sl_shadow_debug,
        }
        log_dir = os.path.dirname(self._calib_log_path)
        os.makedirs(log_dir, exist_ok=True)
        try:
            with open(self._calib_log_path, "a", encoding="utf-8") as f:
                f.write(f"=== FRAME {self._calib_log_frame_idx:04d} ===\n")
                f.write(f"  {summary}\n")
                f.write(json.dumps(self._sanitize_log_value(payload), indent=2, sort_keys=True))
                f.write("\n\n")
        except Exception as exc:
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mCALIB\033[0m - Error writing frame: {exc}")

    def _build_local_lane_payload_log(self):
        """Return a compact summary of the latest local lane payload."""
        payload = self._last_local_lane_payload
        if not isinstance(payload, dict):
            return None
        lane_side_points = payload.get("lane_side_points") or {}
        lane_side_lines = payload.get("lane_side_lines") or {}
        return {
            "frame_id": payload.get("frame_id"),
            "timestamp": payload.get("timestamp"),
            "result_timestamp": payload.get("result_timestamp"),
            "source_frame_timestamp": payload.get("source_frame_timestamp"),
            "source_frame_sequence": payload.get("source_frame_sequence"),
            "inference_time_ms": payload.get("inference_time_ms"),
            "lane_count": payload.get("lane_count"),
            "model_ready": payload.get("model_ready"),
            "heading_hint_rad": payload.get("heading_hint_rad"),
            "heading_hint_confidence": payload.get("heading_hint_confidence"),
            "heading_hint_source": payload.get("heading_hint_source"),
            "road_type_class": payload.get("road_type_class"),
            "road_type_confidence": payload.get("road_type_confidence"),
            "lead_distance_class": payload.get("lead_distance_class"),
            "lead_distance_confidence": payload.get("lead_distance_confidence"),
            "lead_distance_area": payload.get("lead_distance_area"),
            "lane_side_sources": payload.get("lane_side_sources"),
            "lane_side_point_counts": {
                "left": len(lane_side_points.get("left", []) or []),
                "right": len(lane_side_points.get("right", []) or []),
            },
            "lane_side_lines": lane_side_lines,
        }

    def _log_auto_run_frame(
        self,
        frame_sequence,
        frame_timestamp,
        stale_frame,
        computed_steering,
        computed_speed,
        commanded_steering,
        commanded_speed,
        command_source,
    ):
        """Append a detailed per-frame auto-mode trace to disk."""
        if not self._auto_run_log_enabled:
            return

        self._auto_run_log_frame_idx += 1
        latest_observation = None
        if self._lane_observation_history:
            latest_observation = self._lane_observation_history[-1]

        payload = {
            "frame_index": self._auto_run_log_frame_idx,
            "frame_sequence": frame_sequence,
            "wall_time": round(time.time(), 3),
            "frame_timestamp": round(frame_timestamp, 3) if frame_timestamp else None,
            "frame_age_ms": self._last_frame_age_ms,
            "stale_frame": bool(stale_frame),
            "state_change_message": self._last_state_change_message,
            "line_following_active": bool(self.is_line_following_active),
            "detection_mode": str(self.detection_mode),
            "computed": {
                "steering_deg": computed_steering,
                "speed": computed_speed,
            },
            "commanded": {
                "steering_deg": commanded_steering,
                "speed": commanded_speed,
                "source": command_source,
            },
            "nucleo_request": self._last_requested_motor_command,
            "actuator_status": self._last_actuator_status,
            "local_perception_status": self._last_local_perception_status,
            "local_lane_payload": self._build_local_lane_payload_log(),
            "lane_observation": latest_observation,
            "frame_trace": self._last_frame_trace,
            "controller_state": {
                "heading_error_deg": round(math.degrees(self._heading_error), 3),
                "psi_ss_deg": round(math.degrees(self._stanley_last_psi_ss), 3),
                "traj_yaw_rate": round(float(self._stanley_last_traj_yaw_rate), 5),
                "measured_yaw_rate": round(float(self._yaw_rate), 5),
                "measured_speed": round(float(self._measured_speed), 3),
                "measured_speed_mps": round(
                    abs(float(self._measured_speed)) * float(self.stanley_measured_speed_to_mps), 4
                ),
                "measured_steer": round(float(self._measured_steer), 3),
                "measured_steer_delta": round(float(self._measured_steer_delta), 3),
                "stanley_speed_mps": round(float(self._stanley_last_speed_mps), 4),
                "stanley_speed_source": self._stanley_last_speed_source,
                "stanley_error_m": round(float(self._stanley_last_error_m), 4),
                "stanley_px_per_cm": round(float(self._stanley_last_px_per_cm), 4),
                "theta_dmae_deg": round(float(self._theta_dmae_deg), 4),
                "offset_dma_m": round(float(self._offset_dma_m), 5),
                "heading_hint_rad": round(float(self._local_ai_heading_hint_rad), 5),
                "heading_hint_confidence": round(float(self._local_ai_heading_hint_confidence), 4),
                "road_type_class": self._local_ai_road_type_class,
                "road_type_confidence": round(float(self._local_ai_road_type_confidence), 4),
                "lead_distance_class": self._local_ai_lead_distance_class,
                "lead_distance_confidence": round(float(self._local_ai_lead_distance_confidence), 4),
                "curve_guard": self._stanley_last_curve_guard,
                "planner_route_blend": round(float(getattr(self, "_last_planner_route_blend", 0.0) or 0.0), 4),
                "planner_route_blend_source": str(getattr(self, "_last_planner_route_blend_source", "none") or "none"),
                "current_speed_target": round(float(self._current_speed), 3),
                "frames_without_line": int(self.frames_without_line),
                "last_seen_side": self.last_seen_side,
                "curve_state": self._curve_state,
                "curve_direction": self._curve_direction,
                # Noise filter state
                "noise_reject_count": int(self._noise_reject_count),
                "noise_last_good_steer_deg": round(float(self._last_good_steering), 3),
                # Controller selection
                "use_lateral_mpc": bool(getattr(self, 'use_lateral_mpc', False)),
                # MPC last call debug (empty dict if MPC not called)
                "mpc_debug": dict(getattr(self.lateral_mpc, '_last_debug', {})),
                # Tracking state
                "tracking": (
                    {
                        "initialized": bool(getattr(self._tracking_state, 'initialized', False)),
                        "imu_received": bool(getattr(self._tracking_state, 'imu_received', False)),
                        "x_m": round(float(getattr(self._tracking_state, 'x', 0.0)), 4),
                        "y_m": round(float(getattr(self._tracking_state, 'y', 0.0)), 4),
                        "yaw_deg": round(math.degrees(float(getattr(self._tracking_state, 'yaw', 0.0))), 2),
                        "raw_x_m": round(float(getattr(self._tracking_state, 'raw_x', 0.0)), 4),
                        "raw_y_m": round(float(getattr(self._tracking_state, 'raw_y', 0.0)), 4),
                        "raw_yaw_deg": round(math.degrees(float(getattr(self._tracking_state, 'raw_yaw', 0.0))), 2),
                        "matched_x_m": round(float(getattr(self._tracking_state, 'matched_x', 0.0)), 4),
                        "matched_y_m": round(float(getattr(self._tracking_state, 'matched_y', 0.0)), 4),
                        "matched_yaw_deg": round(math.degrees(float(getattr(self._tracking_state, 'matched_yaw', 0.0))), 2),
                        "heading_rad": round(float(getattr(self._tracking_state, 'heading_rad', 0.0)), 5),
                        "heading_deg": round(math.degrees(float(getattr(self._tracking_state, 'heading_rad', 0.0))), 3),
                        "error_m": round(float(getattr(self._tracking_state, 'error_m', 0.0)), 5),
                        "waypoint_mode": bool(getattr(self._tracking_state, 'waypoint_mode_active', False)),
                        "lane_measurement_reliable": bool(getattr(self._tracking_state, 'lane_measurement_reliable', False)),
                        "camera_lateral_correction_m": round(float(getattr(self._tracking_state, 'camera_lateral_correction_m', 0.0)), 5),
                        "map_match_error_m": round(float(getattr(self._tracking_state, 'map_match_error_m', 0.0)), 5),
                        "path_psi_deg": round(math.degrees(float(getattr(self._tracking_state, 'path_psi', 0.0))), 3),
                        "path_kappa": round(float(getattr(self._tracking_state, 'path_kappa', 0.0)), 5),
                        "path_heading_change_rad": round(float(getattr(self._tracking_state, 'path_heading_change_rad', 0.0)), 4),
                        "target_idx": getattr(
                            self._tracking_state,
                            'target_idx',
                            getattr(self._tracking_state, 'wp_idx', None),
                        ),
                        # Camera-based yaw correction applied to DR this frame
                        "dr_yaw_correction_deg": round(float(getattr(self._tracking_state, 'last_yaw_correction_deg', 0.0)), 4),
                        "dr_yaw_cam_hint_deg": round(float(getattr(self._tracking_state, 'last_cam_yaw_hint_deg', 0.0)), 2),
                        "dr_yaw_cam_conf": round(float(getattr(self._tracking_state, 'last_cam_yaw_hint_conf', 0.0)), 3),
                    }
                    if self._tracking_state is not None else None
                ),
            },
        }
        self._append_auto_run_log_entry(
            f"FRAME {self._auto_run_log_frame_idx:04d}",
            payload,
        )

    def thread_work(self):
        """Main loop for line following."""
        try:
            self.check_state_change()
            self._check_calib_mode()
            self._check_config()
            self._poll_local_ai_messages()
            self._poll_sign_detected()
            self._read_sensor_data()

            # Parking maneuver phases that bypass lane detection entirely
            _maneuver_states = (
                ParkingState.FORWARD_PAST_SPOT,
                ParkingState.WAIT_STEER_1,
                ParkingState.REVERSING_ENTRY,
                ParkingState.WAIT_STEER_2,
                ParkingState.REVERSING_ALIGN,
                ParkingState.WAIT_STEER_3,
                ParkingState.FORWARD_CORRECTION,
                ParkingState.PARKED,
            )
            if self._parking_state in _maneuver_states:
                park_steer, park_speed = self._parking_state_machine()
                if park_steer is not None:
                    self.send_motor_commands(park_steer, park_speed)
                return

            # In SPOT_TRACKED, count consecutive frames without detection.
            # (_poll_sign_detected resets the counter when the spot is seen.)
            if self._parking_state == ParkingState.SPOT_TRACKED:
                self._parking_spot_miss_frames += 1
                if self._parking_spot_miss_frames >= PARKING_SPOT_MISS_THRESHOLD:
                    now_t = time.time()
                    self._parking_state = ParkingState.FORWARD_PAST_SPOT
                    self._parking_phase_start_time = now_t
                    self._parking_phase_dist_cm = 0.0
                    self._parking_last_sm_time = now_t
                    last_dist = self._parking_last_spot_distance_cm
                    # Dynamic advance: if the spot disappeared while still ahead of the
                    # camera (curve-exit case), the car hasn't fully passed it yet.
                    # Add the remaining distance so the car stops at the correct position.
                    _lost_thr = float(getattr(_config, "PARKING_SPOT_LOST_DIST_THRESHOLD_CM", 10.0))
                    if last_dist is not None and last_dist > _lost_thr:
                        extra_cm = last_dist - _lost_thr
                        self._parking_dynamic_forward_cm = PARKING_D_FORWARD_CM + extra_cm
                    else:
                        self._parking_dynamic_forward_cm = PARKING_D_FORWARD_CM
                    dist_str = f"~{last_dist:.0f}cm" if last_dist is not None else "unknown"
                    print(
                        f"\033[1;97m[ Parking ] :\033[0m \033[1;96mFORWARD_PAST_SPOT\033[0m - "
                        f"Spot lost after {self._parking_spot_miss_frames} frames, "
                        f"last distance={dist_str}, "
                        f"advance={self._parking_dynamic_forward_cm:.0f}cm "
                        f"(base={PARKING_D_FORWARD_CM:.0f}cm)"
                    )
                    # Transition immediately to maneuver in this cycle
                    park_steer, park_speed = self._parking_state_machine()
                    if park_steer is not None:
                        self.send_motor_commands(park_steer, park_speed)
                    return

            loop_start = time.time()
            self._begin_preview_cycle(loop_start)

            if self.frame_buffer is None:
                time.sleep(0.02)
                return

            frame, frame_timestamp, frame_sequence = self.frame_buffer.read_latest(copy_frame=True)
            if frame is None:
                time.sleep(0.02)
                return

            if frame_sequence == self._last_processed_sequence:
                return
            self._last_processed_sequence = frame_sequence

            frame_age_ms = None
            if frame_timestamp:
                frame_age_ms = max(0.0, (time.time() - frame_timestamp) * 1000.0)
            self._last_frame_age_ms = round(frame_age_ms, 1) if frame_age_ms is not None else None
            stale_frame = frame_age_ms is not None and frame_age_ms > self.max_frame_age_ms

            if stale_frame:
                steering_angle = self._last_safe_steering
                speed = self.min_speed
                self._set_frame_trace({
                    "status": "stale_frame_hold",
                    "frame_age_ms": self._last_frame_age_ms,
                    "steering_deg": steering_angle,
                    "speed": speed,
                })
                debug_frame = None
                if self._needs_debug:
                    debug_frame = frame.copy()
                    cv2.putText(
                        debug_frame,
                        f"STALE FRAME {frame_age_ms:.0f}ms - HOLD",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                    )
            else:
                steering_angle, speed, debug_frame = self.process_frame(frame)
                if steering_angle is not None:
                    self._last_safe_steering = steering_angle
                    if speed is not None:
                        self._last_safe_speed = speed

            maneuver_decision = self._maneuver_manager.decide(
                time.time(),
                tracking_state=self._tracking_state,
            )
            self._planner_priority_active = bool(maneuver_decision.planner_priority)
            self._active_maneuver_name = str(maneuver_decision.active_maneuver or "none")
            self._mission_state_name = str(maneuver_decision.mission_state or "LANE_FOLLOW")
            self._last_maneuver_notes = str(maneuver_decision.notes or "")
            self._safety_stop_reason = self._last_maneuver_notes if maneuver_decision.safety_stop else ""

            if maneuver_decision.speed_cap is not None and speed is not None:
                speed = min(float(speed), float(maneuver_decision.speed_cap))
            if maneuver_decision.speed_override is not None:
                speed = float(maneuver_decision.speed_override)
            if maneuver_decision.steer_override is not None:
                steering_angle = float(maneuver_decision.steer_override)
            if maneuver_decision.safety_stop:
                if steering_angle is None:
                    steering_angle = 0.0
                speed = 0.0

            # While parking is active (LANE_KEEPING / SPOT_TRACKED), cap speed
            # so the vehicle moves slowly enough to react to the detected spot.
            _parking_active_states = (ParkingState.LANE_KEEPING, ParkingState.SPOT_TRACKED)
            if self._parking_state in _parking_active_states and speed is not None:
                speed = min(speed, PARKING_SEARCH_SPEED)

            nav_ctx = self._get_navigation_context()
            last_trace = self._last_frame_trace or {}
            debug_trace = last_trace.get("debug", {}) if isinstance(last_trace, dict) else {}
            blind_control_mode = str(debug_trace.get("blind_control_mode", "") or "")
            if maneuver_decision.safety_stop:
                self._control_policy_mode = "SAFETY_STOP"
                self._control_authority = "safety"
            elif self._active_maneuver_name != "none":
                self._control_policy_mode = "MANEUVER_CONTROL"
                self._control_authority = "maneuver"
            elif blind_control_mode == "visual_fallback":
                self._control_policy_mode = "VISUAL_FALLBACK"
                self._control_authority = "visual_fallback"
            elif blind_control_mode == "route_tracking" or nav_ctx["planner_priority"]:
                self._control_policy_mode = "ROUTE_TRACKING"
                self._control_authority = "planner"
            elif nav_ctx["route_active"]:
                self._control_policy_mode = "VISUAL_ASSIST"
                self._control_authority = "planner_visual_assist"
            else:
                self._control_policy_mode = "VISUAL_ASSIST"
                self._control_authority = "visual"

            computed_steering = steering_angle
            computed_speed = speed
            commanded_steering = None
            commanded_speed = None
            command_source = "no_command"
            self._last_requested_motor_command = None
            
            if self.is_line_following_active:
                if steering_angle is not None:
                    commanded_steering = steering_angle
                    commanded_speed = speed
                    command_source = "stale_hold" if stale_frame else "normal"
                    self.send_motor_commands(steering_angle, speed)
                    self.frames_without_line = 0
                else:
                    self.frames_without_line += 1
                    if self.show_debug and self.frames_without_line % 10 == 1:
                        print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;93mWAIT\033[0m - No steering, frame {self.frames_without_line}/{self.max_frames_without_line}")
                    if self.frames_without_line > self.max_frames_without_line:
                        commanded_steering = 0
                        commanded_speed = self.min_speed
                        command_source = "normal"
                        self.send_motor_commands(0, self.min_speed)
            else:
                command_source = "blocked_inactive"
                if hasattr(self, '_last_inactive_log') == False:
                    self._last_inactive_log = True

                # CALIBRATION MODE: manual mode only — run algorithm silently, enforce min speed
                if self._calib_mode_active:
                    command_source = "calib_mode"
                    # Enforce minimum speed so the car moves slowly during calibration
                    calib_speed_x10 = int(round(self.min_speed * 10))
                    self.speedMotorSender.send(str(calib_speed_x10))
                    commanded_speed = self.min_speed
                    # Steering is NOT sent — the user's manual web commands control the wheels
                    # Log computed vs manual for calibration analysis
                    self._log_calib_frame(steering_angle)

            if self.sign_action_event and self.sign_action_event.is_set() and commanded_steering is not None:
                if command_source in ("normal", "stale_hold"):
                    command_source = "sign_override"
                commanded_speed = None

            actuator_status = self._last_actuator_status or {}
            if (
                self.is_line_following_active
                and commanded_steering is not None
                and actuator_status.get("engine_enabled") is False
            ):
                command_source = "blocked_klem"

            self._log_auto_run_frame(
                frame_sequence=frame_sequence,
                frame_timestamp=frame_timestamp,
                stale_frame=stale_frame,
                computed_steering=computed_steering,
                computed_speed=computed_speed,
                commanded_steering=commanded_steering,
                commanded_speed=commanded_speed,
                command_source=command_source,
            )
            self._log_manual_run_frame()
            # Always-on mask classification debug log (enabled via LANE_MASK_DEBUG_LOG in config)
            if self._mask_debug_log_enabled and self._last_frame_trace:
                _di = self._last_frame_trace.get('debug') or self._last_frame_trace
                self._write_mask_debug_entry(_di, computed_steering, computed_speed)

            if self.show_debug and (computed_steering is not None or commanded_steering is not None):
                comp_text = f"{computed_steering:.1f}" if computed_steering is not None else "--"
                cmd_text = f"{commanded_steering:.1f}" if commanded_steering is not None else "--"
                self._debug_log(
                    "command_status",
                    f"\033[1;97m[ Line Following ] :\033[0m \033[1;96mCMD\033[0m - "
                    f"computed={comp_text}° commanded={cmd_text}° source={command_source}"
                )

            display_steering = commanded_steering if commanded_steering is not None else computed_steering
            display_speed = commanded_speed if commanded_speed is not None else computed_speed

            if debug_frame is not None:
                self._overlay_command_status(
                    debug_frame,
                    computed_steering,
                    computed_speed,
                    commanded_steering,
                    commanded_speed,
                    command_source,
                )
                self._store_debug_image('final', debug_frame)

            self._send_debug_stream(
                computed_steering,
                computed_speed,
                commanded_steering,
                commanded_speed,
                command_source,
            )
            
            if self.show_debug:
                if debug_frame is not None and self._is_window_enabled("final_result"):
                    status_text = "ACTIVE" if self.is_line_following_active else "INACTIVE (Debug Mode)"
                    cv2.putText(debug_frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                               (0, 255, 0) if self.is_line_following_active else (0, 0, 255), 2)
                    self._show_preview_window("1. Final Result", debug_frame)
                
                # Show control panel with current status
                if self._is_window_enabled("control_panel"):
                    self._show_control_panel(display_steering, display_speed)
                if self._is_window_enabled("steering_angle"):
                    self._show_final_steering_preview(
                        computed_steering,
                        commanded_steering,
                        command_source,
                    )
                self._flush_preview_windows()
        except Exception as e:
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mERROR\033[0m - {e}")

    def process_frame(self, frame):
        """Process frame using selected detection mode.
        
        Supports multiple modes:
        - opencv: Traditional HSV + sliding window (default)
        - lstr: AI-based LSTR transformer model
        - hybrid: FUSION of OpenCV + LSTR (runs both, combines results)
        - ai_local: Local YOLO lane detection + local BFMC/Stanley control
        - hybridnets / supercombo: deprecated aliases mapped to ai_local

Returns:
    tuple: (steering_angle, speed, debug_frame)
"""
        height, width = frame.shape[:2]
        normalized_mode = self._normalize_detection_mode(self.detection_mode)
        if normalized_mode != self.detection_mode:
            self.detection_mode = normalized_mode
        self._set_frame_trace({
            "status": "processing",
            "detection_mode": self.detection_mode,
            "frame_size": {"height": int(height), "width": int(width)},
        })
        
        if self.show_debug:
            self._debug_log(
                "frame_received",
                f"\033[1;97m[ Line Following ] :\033[0m Frame received: {width}x{height} "
                f"Mode: {self.detection_mode}",
            )
        
        # Handle SUPERCOMBO remote AI server mode
        if self.detection_mode == DetectionMode.SUPERCOMBO.value:
            try:
                steering, speed, debug = self._detect_with_supercombo(frame)
                if steering is not None:
                    self.frames_without_line = 0
                self._set_frame_trace({
                    "status": "supercombo",
                    "detection_mode": self.detection_mode,
                    "steering_deg": steering,
                    "speed": speed,
                })
                return steering, speed, debug
            except Exception as e:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mSUPERCOMBO ERROR\033[0m - {e}")
                debug_frame = frame.copy()
                cv2.putText(debug_frame, f"Supercombo Error: {str(e)[:50]}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                self._set_frame_trace({
                    "status": "supercombo_error",
                    "detection_mode": self.detection_mode,
                    "error": str(e),
                })
                return None, self.min_speed, debug_frame
        
        # Handle HYBRID fusion mode (runs both detectors and combines)
        if self.detection_mode == DetectionMode.HYBRID.value:
            try:
                steering, speed, debug = self._detect_hybrid_fusion(frame)
                if steering is not None:
                    self.frames_without_line = 0
                self._set_frame_trace({
                    "status": "hybrid",
                    "detection_mode": self.detection_mode,
                    "steering_deg": steering,
                    "speed": speed,
                })
                return steering, speed, debug
            except Exception as e:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mHYBRID ERROR\033[0m - {e}")
                debug_frame = frame.copy()
                cv2.putText(debug_frame, f"HYBRID Error: {str(e)[:50]}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                self._set_frame_trace({
                    "status": "hybrid_error",
                    "detection_mode": self.detection_mode,
                    "error": str(e),
                })
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
                    self._set_frame_trace({
                        "status": "lstr",
                        "detection_mode": self.detection_mode,
                        "steering_deg": steering,
                        "speed": speed,
                        "lane_center_x": center,
                    })
                    return steering, speed, debug
                else:
                    # LSTR failed, return no detection
                    if self.show_debug:
                        print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;93mLSTR\033[0m - No lanes detected")
                    debug_frame = frame.copy()
                    cv2.putText(debug_frame, "LSTR: No lanes detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self._set_frame_trace({
                        "status": "lstr_no_lanes",
                        "detection_mode": self.detection_mode,
                    })
                    return None, self.min_speed, debug_frame
            except Exception as e:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mLSTR ERROR\033[0m - {e}")
                debug_frame = frame.copy()
                cv2.putText(debug_frame, f"LSTR Error: {str(e)[:50]}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                self._set_frame_trace({
                    "status": "lstr_error",
                    "detection_mode": self.detection_mode,
                    "error": str(e),
                })
                return None, self.min_speed, debug_frame
        
        using_remote_lanes = self.detection_mode in (DetectionMode.AI_LOCAL.value, DetectionMode.HYBRIDNETS.value)

        if using_remote_lanes:
            try:
                if self.detection_mode == DetectionMode.AI_LOCAL.value:
                    avg_left, avg_right, img_h, img_w, canny, debug_info = self._detect_with_local_ai(frame)
                else:
                    avg_left, avg_right, img_h, img_w, canny, debug_info = self._detect_with_hybridnets(frame)
            except Exception as e:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mAI LOCAL ERROR\033[0m - {e}")
                debug_frame = frame.copy()
                cv2.putText(debug_frame, f"AI Local Error: {str(e)[:50]}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                self._set_frame_trace({
                    "status": "ai_local_error",
                    "detection_mode": self.detection_mode,
                    "error": str(e),
                })
                return None, self.min_speed, debug_frame
        else:
            # Match bfmc24-brain StanleyLaneDetector.image_processing(): raw frame goes
            # directly into ROI -> grayscale -> threshold -> median -> canny -> hough.
            avg_left, avg_right, img_h, img_w, canny, debug_info = self._bfmc_image_processing(frame)

            # BFMC retry: if no lines detected, try lower threshold (from MarcosLaneDetector)
            if avg_left is None and avg_right is None:
                avg_left, avg_right, img_h, img_w, canny, debug_info = self._bfmc_image_processing(
                    frame,
                    threshold_override=self.binary_threshold_retry,
                    kernel_override=3
                )

            if 'hsv' in debug_info:
                self._store_debug_image('hsv', cv2.cvtColor(debug_info['hsv'], cv2.COLOR_HSV2BGR))
            if 'white_mask' in debug_info:
                self._store_debug_image('white_mask', cv2.cvtColor(debug_info['white_mask'], cv2.COLOR_GRAY2BGR))
            if 'yellow_mask' in debug_info:
                self._store_debug_image('yellow_mask', cv2.cvtColor(debug_info['yellow_mask'], cv2.COLOR_GRAY2BGR))
            if 'combined_mask' in debug_info:
                self._store_debug_image('combined_mask', cv2.cvtColor(debug_info['combined_mask'], cv2.COLOR_GRAY2BGR))

            # Show debug windows (only enabled ones)
            if self.show_debug:
                if 'roi' in debug_info:
                    self._show_preview_window("ROI", debug_info['roi'])
                if self._is_window_enabled("binary_threshold") and 'binary' in debug_info:
                    bin_display = cv2.cvtColor(debug_info['binary'], cv2.COLOR_GRAY2BGR)
                    cv2.putText(bin_display, f"Threshold: {debug_info.get('threshold', '?')}", (10, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    self._show_preview_window("2. Binary Threshold", bin_display)
                if self._is_window_enabled("canny_edges") and 'canny' in debug_info:
                    self._show_preview_window("3. Canny Edges", debug_info['canny'])
                self._store_debug_image('canny', cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR))

        # Calculate steering based on detected lines
        steering_angle = None
        speed = self.base_speed
        speed_cap = None
        error = None
        midpoint_x = None
        bottom_left_x = None
        bottom_right_x = None
        kalman_prediction_used = False
        self._is_stabilization_frame = False  # Reset each frame; set True by just_seen_two_lines logic
        kalman_pred_error = self._kalman_predict_error()
        stale_grace_active = (debug_info.get('local_ai_status') == 'stale_grace')
        if stale_grace_active:
            debug_info['stale_grace_active'] = True

        avg_left, avg_right, overlap_collapse = self._collapse_overlapping_two_lines(
            avg_left,
            avg_right,
            img_h,
            img_w,
            prev_seen_side=getattr(self, 'last_seen_side', None),
        )
        if overlap_collapse is not None:
            debug_info['duplicate_line_collapse'] = overlap_collapse
            debug_info['single_line_side_source'] = (
                f"duplicate_overlap:{overlap_collapse.get('resolution_source', 'unknown')}"
            )
            debug_info['single_line_resolved_side'] = overlap_collapse.get('kept_side')
            debug_info['left_lines'] = [avg_left] if avg_left is not None else []
            debug_info['right_lines'] = [avg_right] if avg_right is not None else []
            if debug_info.get('control_mode') != 'mask_guidance':
                debug_info['remote_lane_count'] = 1

        # === PRE-CHECK: Reject obviously noisy frames (reflection/glare) ===
        local_mask_guidance = debug_info.get('local_mask_guidance')
        mask_guidance_left_line = None
        mask_guidance_right_line = None
        num_lines = (1 if avg_left is not None else 0) + (1 if avg_right is not None else 0)
        if isinstance(local_mask_guidance, dict):
            num_lines = len(local_mask_guidance.get('detected_sides', ()))
            mask_guidance_left_line, mask_guidance_right_line = self._mask_guidance_lines(
                local_mask_guidance, img_h, img_w
            )
        _pre_noisy, _pre_reason = self._is_frame_noisy(debug_info, None, None, num_lines)
        if _pre_noisy:
            if self._is_ai_local_active() and _pre_reason == "sudden_loss(2→0)":
                debug_info['blind_transition_loss'] = True
                debug_info['blind_transition_loss_reason'] = _pre_reason
            else:
                # Frame is noisy (e.g. too many Hough lines from reflections)
                # Use previous good steering and skip all processing
                self._noise_reject_count += 1
                if self.show_debug:
                    print(f"\033[1;97m[ Noise Filter ] :\033[0m \033[1;91mREJECT\033[0m "
                          f"({self._noise_reject_count}/{self.noise_max_reject_frames}) "
                          f"reason={_pre_reason}")
                steering_angle = self._last_good_steering
                speed = self.min_speed  # Slow down during noise
                # Create debug frame showing rejection (only when debug active)
                if self._needs_debug:
                    debug_frame = self._bfmc_draw_debug(
                        frame, avg_left, avg_right, None, None, None, None, steering_angle, debug_info
                    )
                    cv2.rectangle(debug_frame, (0, debug_frame.shape[0] - 25),
                                 (debug_frame.shape[1], debug_frame.shape[0]), (0, 0, 150), -1)
                    cv2.putText(debug_frame, f"NOISE REJECT: {_pre_reason}",
                               (8, debug_frame.shape[0] - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
                    self._store_debug_image('final', debug_frame)
                else:
                    debug_frame = None
                self._set_frame_trace({
                    "status": "noise_reject",
                    "detection_mode": self.detection_mode,
                    "reason": _pre_reason,
                    "steering_deg": steering_angle,
                    "speed": speed,
                    "noise_reject_count": self._noise_reject_count,
                    "debug": debug_info,
                })
                return steering_angle, speed, debug_frame

        # === UPDATE CURVE STATE MACHINE ===
        # Always run, even when use_swept_path=False: curve_state drives single-line
        # error weights and heading fusion — independent of swept-path geometry.
        if isinstance(local_mask_guidance, dict):
            self._update_curve_state(
                num_lines,
                mask_guidance_left_line,
                mask_guidance_right_line,
                img_h,
                img_w,
            )
        else:
            self._update_curve_state(num_lines, avg_left, avg_right, img_h, img_w)
        self._apply_graph_curve_state(getattr(self, '_tracking_state', None))

        prev_seen_side = self.last_seen_side
        _ts_for_tracking = getattr(self, '_tracking_state', None)
        _tracking_cam_min_speed_mps = float(
            getattr(_config, 'TRACKING_CAMERA_CORRECTION_MIN_SPEED_MPS', 0.02) or 0.02
        )
        _tracking_speed_mps = (
            abs(float(getattr(_ts_for_tracking, 'speed_mps', 0.0) or 0.0))
            if _ts_for_tracking is not None else 0.0
        )
        _tracking_camera_corrections_allowed = bool(
            self.is_line_following_active and
            _ts_for_tracking is not None and
            _tracking_speed_mps > _tracking_cam_min_speed_mps
        )
        if _ts_for_tracking is not None:
            _ts_for_tracking.set_lane_measurement_state(False, 0.0)
        if not isinstance(local_mask_guidance, dict):
            if avg_left is not None and avg_right is not None:
                observed_sides = ('left', 'right')
            elif avg_left is not None:
                observed_sides = ('left',)
            elif avg_right is not None:
                observed_sides = ('right',)
            else:
                observed_sides = ()
            lane_observation = self._record_lane_observation(
                observed_sides,
                duplicate_collapse=debug_info.get('duplicate_line_collapse'),
            )
            debug_info.update(self._lane_observation_debug_fields(lane_observation))

        if isinstance(local_mask_guidance, dict):
            detected_sides = tuple(local_mask_guidance.get('detected_sides', ()))
            lane_width_px = float(local_mask_guidance.get('lane_width_px', 0.0) or 0.0)
            if len(detected_sides) >= 2 and lane_width_px > 1.0:
                self._last_local_ai_lane_width_px = lane_width_px

            error = float(local_mask_guidance.get('error_px', 0.0) or 0.0)
            heading = float(local_mask_guidance.get('heading_rad', 0.0) or 0.0)
            if len(detected_sides) == 1:
                heading = self._fuse_heading_with_local_hint(heading, debug_info, mode='single_line')
            elif stale_grace_active:
                heading = self._fuse_heading_with_local_hint(heading, debug_info, mode='stale')
            self._heading_error = heading
            # Use the LOOKAHEAD midpoint (midpoint_x at reference_y) for the visual dot,
            # not the near-field midpoint (midpoint_bottom_x at measurement_bottom_y).
            # The error is computed at reference_y (lookahead), so the visualization must
            # show the same point — this is what the AI Local Overlay also shows (green dot
            # at height * 0.6). Using midpoint_bottom_x with draw_y=reference_y was mixing
            # near-field X with lookahead Y, making the final_result appear decentered.
            midpoint_x = int(round(local_mask_guidance.get('midpoint_x', img_w / 2.0)))
            midpoint_x = max(0, min(img_w - 1, midpoint_x))
            bottom_left_x = int(round(local_mask_guidance.get('bottom_left_x', 0.0) or 0.0))
            bottom_right_x = int(round(local_mask_guidance.get('bottom_right_x', img_w - 1) or (img_w - 1)))
            bottom_left_x = max(0, min(img_w - 1, bottom_left_x))
            bottom_right_x = max(0, min(img_w - 1, bottom_right_x))
            debug_info['midpoint_draw_y'] = int(
                local_mask_guidance.get('reference_y', local_mask_guidance.get('draw_y', img_h // 2))
            )
            self._swept_path_info = None
            self.just_seen_two_lines = False

            if self.use_kalman_filter:
                raw_error = error
                error = self._kalman_correct_error(error)
                debug_info['kalman_raw_error'] = raw_error
                debug_info['kalman_error'] = error
                debug_info['kalman_age'] = self._kalman_age_frames

            if len(detected_sides) >= 2:
                self.consecutive_single_left = 0
                self.consecutive_single_right = 0
                self.frames_without_line = 0
                self.last_seen_side = "both"

                curve_reference = None
                curve_context = self._curve_state in ("ENTERING", "IN_CURVE", "EXITING")
                if (
                    (self.use_swept_path or curve_context) and
                    mask_guidance_left_line is not None and
                    mask_guidance_right_line is not None
                ):
                    swept = self._predict_swept_path(
                        mask_guidance_left_line,
                        mask_guidance_right_line,
                        img_h,
                        img_w,
                    )
                    self._swept_path_info = swept
                    if swept is not None and abs(float(swept.get('offset_px', 0.0) or 0.0)) > 0.5:
                        error += float(swept['offset_px'])
                        if self.curve_speed_reduction and float(swept.get('min_clearance_cm', 1e9)) < self.min_clearance_warn_cm:
                            speed = self.min_speed
                            speed_cap = speed
                    curve_reference = self._get_stanley_curve_reference(
                        avg_left=mask_guidance_left_line,
                        avg_right=mask_guidance_right_line,
                        img_h=img_h,
                        img_w=img_w,
                        fallback_curve=self._swept_path_info,
                    )

                speed_val = self._current_speed if self._current_speed > 0 else self.base_speed

                # Physical error: computed directly from the distance of the car to each
                # lane line using the KNOWN lane width (lane_width_cm) as the calibration.
                # This avoids using img_w/2 as a universal reference — instead it measures:
                #   D_right_cm = how many cm the car is from the right lane line
                #   D_left_cm  = how many cm the car is from the left lane line
                # Error = D_right_cm - lane_width_cm/2  (+ = car is LEFT, steer RIGHT)
                # The safety margin ensures the car body (car_width cm wide) stays at least
                # lane_safety_margin_cm from each line at all times.
                direct_error_m = None
                _left_x  = float(local_mask_guidance.get('left_x',  0.0) or 0.0)
                _right_x = float(local_mask_guidance.get('right_x', 0.0) or 0.0)
                # Use the lane width AT reference_y as the pixel scale.
                # left_x / right_x are measured at reference_y, so (right_x - left_x) gives
                # the true lane width in pixels at that depth.  This guarantees:
                #   D_left_cm + D_right_cm == lane_width_cm
                # Using the averaged lane_width_px (bottom-heavy) would break this invariant
                # because the lane appears wider near the bottom due to perspective.
                _ref_lw_px = _right_x - _left_x
                if _ref_lw_px > 1.0:
                    _lw_cm = float(self.lane_width_cm)
                    # Cache the authoritative px/cm AND reference_y from 2-line detection.
                    # Single-line mode must use the SAME reference_y so that left_x/right_x
                    # positions and px_per_cm correspond to the same point in the image.
                    self._last_two_line_ref_px_per_cm = _ref_lw_px / max(1e-3, _lw_cm)
                    _two_line_ref_y = local_mask_guidance.get('reference_y')
                    if _two_line_ref_y is not None:
                        self._last_two_line_reference_y = int(_two_line_ref_y)
                    # Distances from camera centre (= car centre) to each line in cm
                    D_right_cm = (_right_x - img_w / 2.0) * _lw_cm / _ref_lw_px
                    D_left_cm  = (img_w / 2.0 - _left_x)  * _lw_cm / _ref_lw_px
                    # Base error: positive → car is left of lane centre → steer right
                    direct_error_m = (D_right_cm - _lw_cm / 2.0) / 100.0
                    # Safety margin + rear-axle offtracking correction.
                    # Controlled by OFFTRACK_SCALE and OFFTRACK_CURVE_ONLY (config.py).
                    _safety_cm   = float(getattr(self, 'lane_safety_margin_cm', 5.0))
                    _car_half_cm = float(self.car_width) / 2.0
                    _steer_v     = float(getattr(self, '_last_good_steering', 0.0) or 0.0)
                    _steer_r     = math.radians(abs(_steer_v))
                    _wbase_cm    = float(getattr(self, 'car_wheelbase_cm', 26.0))
                    _ot_scale2   = float(getattr(_config, 'OFFTRACK_SCALE', 1.0))
                    _ot_c_only2  = bool(getattr(_config, 'OFFTRACK_CURVE_ONLY', False))
                    _in_curve2   = self._curve_state in ('ENTERING', 'IN_CURVE', 'EXITING')
                    if _steer_r > 0.017 and _ot_scale2 > 0.0 and (not _ot_c_only2 or _in_curve2):
                        _R_cm        = _wbase_cm / math.tan(_steer_r)
                        _offtrack_cm = (math.sqrt(_R_cm ** 2 + _wbase_cm ** 2) - _R_cm) * _ot_scale2
                    else:
                        _offtrack_cm = 0.0
                    if _safety_cm > 0.0:
                        if _steer_v < -1.0:
                            _lv = max(0.0, (_car_half_cm + _safety_cm + _offtrack_cm) - D_left_cm)
                            _rv = max(0.0, (_car_half_cm + _safety_cm) - D_right_cm)
                        elif _steer_v > 1.0:
                            _lv = max(0.0, (_car_half_cm + _safety_cm) - D_left_cm)
                            _rv = max(0.0, (_car_half_cm + _safety_cm + _offtrack_cm) - D_right_cm)
                        else:
                            _lv = max(0.0, (_car_half_cm + _safety_cm) - D_left_cm)
                            _rv = max(0.0, (_car_half_cm + _safety_cm) - D_right_cm)
                        direct_error_m += _lv / 100.0   # push right if too close to left
                        direct_error_m -= _rv / 100.0   # push left  if too close to right
                    if direct_error_m is not None and abs(float(direct_error_m)) > 1e-4:
                        exit_unwind_factor = self._get_curve_exit_unwind_factor(
                            reinforced_sign=direct_error_m,
                            debug_info=debug_info,
                            debug_prefix="two_line_exit_unwind",
                        )
                        if exit_unwind_factor < 0.999:
                            raw_direct_error_m = float(direct_error_m)
                            direct_error_m = raw_direct_error_m * float(exit_unwind_factor)
                            debug_info['two_line_exit_unwind_input_error_m'] = round(raw_direct_error_m, 4)
                            debug_info['two_line_exit_unwind_output_error_m'] = round(float(direct_error_m), 4)
                    debug_info['two_line_D_left_cm']  = round(D_left_cm, 2)
                    debug_info['two_line_D_right_cm'] = round(D_right_cm, 2)
                    debug_info['two_line_ref_lw_px']  = round(_ref_lw_px, 1)
                    debug_info['two_line_direct_error_m'] = round(direct_error_m, 4)
                    debug_info['two_line_offtrack_cm']    = round(_offtrack_cm, 2)

                    # In 2-line mode with a physical direct_error_m the heading from the
                    # mask geometry is dominated by the car's YAW angle, not road curvature.
                    # As the Stanley crosstrack correction turns the car body, the heading
                    # grows positively (nose points more left -> heading says "steer right"),
                    # creating a positive-feedback loop that cancels the very correction being
                    # applied. With direct_error_m providing a physically reliable crosstrack
                    # signal, the heading term is not only unnecessary but actively harmful.
                    # Pass heading=0 so crosstrack is the sole driver in 2-line mode.
                    if _tracking_camera_corrections_allowed and direct_error_m is not None:
                        _lat_gain = float(getattr(
                            _config, 'TRACKING_CAMERA_LATERAL_CORRECTION_GAIN', 0.35
                        ) or 0.35)
                        _lat_max = float(getattr(
                            _config, 'TRACKING_CAMERA_LATERAL_CORRECTION_MAX_M', 0.08
                        ) or 0.08)
                        _lat_corr = float(direct_error_m) * _lat_gain
                        if _lat_max > 0.0:
                            _lat_corr = max(-_lat_max, min(_lat_max, _lat_corr))
                        _ts_for_tracking.correct_lateral(_lat_corr)
                        debug_info['tracking_camera_lateral_correction_m'] = round(_lat_corr, 4)
                        debug_info['tracking_lane_measurement_reliable'] = True
                    elif direct_error_m is not None:
                        debug_info['tracking_camera_correction_blocked'] = (
                            'inactive' if not self.is_line_following_active else 'low_speed'
                        )
                    _two_line_heading = 0.0 if direct_error_m is not None else heading
                    steering_angle = self._compute_lateral_control(
                        error, _two_line_heading, speed_val, curve_reference=curve_reference,
                        lane_width_px=lane_width_px, img_w=img_w,
                        direct_error_m=direct_error_m,
                        direct_error_context={"source": "two_line"},
                    )
                    steering_angle = self._enforce_two_line_curve_priority(
                        steering_angle,
                        direct_error_m=direct_error_m,
                        debug_info=debug_info,
                    )

                self.steer_history.append(steering_angle)
                steering_angle = sum(self.steer_history) / len(self.steer_history)
                steering_angle = max(-self.max_steering, min(self.max_steering, steering_angle))
            elif 'left' in detected_sides:
                self.consecutive_single_left += 1
                self.consecutive_single_right = 0
                self.last_seen_side = "left"
                debug_info['single_line_resolved_side'] = 'left'
                debug_info['single_line_side_source'] = 'mask'
                speed = self.min_speed + 2
                speed_cap = speed
                single_line_mode = str(debug_info.get('single_line_mode', '') or '')
                physical_single_line_mode = single_line_mode == 'physical'
                transversal_recovery_active = self._is_transversal_recovery_active(
                    local_mask_guidance,
                    debug_info=debug_info,
                )

                _sl_direct = None
                if not transversal_recovery_active and physical_single_line_mode:
                    _sl_direct = self._get_single_line_physical_direct_error(
                        'left',
                        mask_guidance_left_line,
                        local_mask_guidance,
                        img_w,
                        debug_info=debug_info,
                    )

                if (
                    not transversal_recovery_active and
                    physical_single_line_mode and
                    self._should_apply_inner_line_escape(
                        'left',
                        direct_error_m=_sl_direct,
                        debug_info=debug_info,
                    )
                ):
                    steering_angle = self.max_steering * self.curve_inner_line_steer_factor
                    speed = self.min_speed
                    speed_cap = speed
                    self._swept_path_info = None
                    debug_info['inner_line_escape'] = True
                else:
                    if self._is_seeing_inner_line('left') and transversal_recovery_active:
                        debug_info['inner_line_escape_suppressed'] = 'transversal_recovery'
                    elif self._is_seeing_inner_line('left') and not physical_single_line_mode:
                        debug_info['inner_line_escape_suppressed'] = (
                            f'non_physical_mode:{single_line_mode or "unknown"}'
                        )
                    swept = None
                    if self.use_swept_path and mask_guidance_left_line is not None:
                        swept = self._predict_swept_path_single_line(
                            mask_guidance_left_line, 'left', img_h, img_w
                        )
                        self._swept_path_info = swept
                        if swept is not None:
                            if self._curve_state == "IN_CURVE":
                                w = self.single_line_swept_weight_in_curve
                            elif self._curve_state == "EXITING":
                                w = self.single_line_swept_weight_exiting
                            elif self._curve_state == "ENTERING":
                                w = self.single_line_swept_weight_entering
                            else:
                                w = 0.35
                            w = max(0.0, min(1.0, float(w)))
                            error = (1.0 - w) * error + w * float(swept.get('steering_error', error))
                            heading = heading * (1.0 - 0.6 * w)
                            debug_info['single_line_swept_weight'] = float(w)
                            if (
                                self.curve_speed_reduction and
                                float(swept.get('min_clearance_cm', 1e9)) < self.min_clearance_warn_cm
                            ):
                                speed = self.min_speed
                                speed_cap = speed
                    else:
                        self._swept_path_info = None

                    error, error_stabilization = self._stabilize_single_line_error(
                        error,
                        'left',
                        prev_seen_side,
                        self.consecutive_single_left,
                        lane_width_px=lane_width_px,
                        img_w=img_w,
                    )
                    if error_stabilization is not None:
                        debug_info['single_line_error_stabilization'] = error_stabilization

                    speed_val = self._current_speed if self._current_speed > 0 else self.base_speed
                    curve_reference = self._get_stanley_curve_reference(
                        fallback_curve=self._swept_path_info
                    ) if isinstance(self._swept_path_info, dict) else None

                    # Physical direct_error_m using the real left mask position + safety margin.
                    # CRITICAL: use the position at the 2-line reference_y (lookahead point) and
                    # the px_per_cm calibrated at that same y.  The single-line guidance samples
                    # left_x at a different reference_y (clamped by the mask extent), which is
                    # farther from the lookahead and gives a completely different apparent
                    # position even though the car hasn't moved.
                    if transversal_recovery_active:
                        debug_info['single_line_direct_error_disabled'] = 'transversal_recovery'
                    elif not physical_single_line_mode:
                        debug_info['single_line_direct_error_disabled'] = (
                            f'non_physical_mode:{single_line_mode or "unknown"}'
                        )
                    steering_angle = self._compute_lateral_control(
                        error, heading, speed_val, curve_reference=curve_reference, speed_cap=speed_cap,
                        lane_width_px=lane_width_px, img_w=img_w, direct_error_m=_sl_direct,
                        direct_error_context={"source": "single_line", "side": "left"},
                    )
                    steering_angle = self._enforce_single_line_curve_priority(
                        steering_angle,
                        heading,
                        direct_error_m=_sl_direct,
                        side="left",
                        debug_info=debug_info,
                    )

                    steering_angle = max(-self.max_steering, min(self.max_steering, steering_angle))
                    steering_angle, transition_blend = self._blend_single_line_transition_steering(
                        steering_angle,
                        'left',
                        prev_seen_side,
                        self.consecutive_single_left,
                    )
                    if transition_blend is not None:
                        debug_info['single_line_transition_blend'] = transition_blend
                self.frames_without_line = 0
            elif 'right' in detected_sides:
                self.consecutive_single_right += 1
                self.consecutive_single_left = 0
                self.last_seen_side = "right"
                debug_info['single_line_resolved_side'] = 'right'
                debug_info['single_line_side_source'] = 'mask'
                speed = self.min_speed + 2
                speed_cap = speed
                single_line_mode = str(debug_info.get('single_line_mode', '') or '')
                physical_single_line_mode = single_line_mode == 'physical'
                transversal_recovery_active = self._is_transversal_recovery_active(
                    local_mask_guidance,
                    debug_info=debug_info,
                )

                _sl_direct = None
                if not transversal_recovery_active and physical_single_line_mode:
                    _sl_direct = self._get_single_line_physical_direct_error(
                        'right',
                        mask_guidance_right_line,
                        local_mask_guidance,
                        img_w,
                        debug_info=debug_info,
                    )

                if (
                    not transversal_recovery_active and
                    physical_single_line_mode and
                    self._should_apply_inner_line_escape(
                        'right',
                        direct_error_m=_sl_direct,
                        debug_info=debug_info,
                    )
                ):
                    steering_angle = -self.max_steering * self.curve_inner_line_steer_factor
                    speed = self.min_speed
                    speed_cap = speed
                    self._swept_path_info = None
                    debug_info['inner_line_escape'] = True
                else:
                    if self._is_seeing_inner_line('right') and transversal_recovery_active:
                        debug_info['inner_line_escape_suppressed'] = 'transversal_recovery'
                    elif self._is_seeing_inner_line('right') and not physical_single_line_mode:
                        debug_info['inner_line_escape_suppressed'] = (
                            f'non_physical_mode:{single_line_mode or "unknown"}'
                        )
                    swept = None
                    if self.use_swept_path and mask_guidance_right_line is not None:
                        swept = self._predict_swept_path_single_line(
                            mask_guidance_right_line, 'right', img_h, img_w
                        )
                        self._swept_path_info = swept
                        if swept is not None:
                            if self._curve_state == "IN_CURVE":
                                w = self.single_line_swept_weight_in_curve
                            elif self._curve_state == "EXITING":
                                w = self.single_line_swept_weight_exiting
                            elif self._curve_state == "ENTERING":
                                w = self.single_line_swept_weight_entering
                            else:
                                w = 0.35
                            w = max(0.0, min(1.0, float(w)))
                            error = (1.0 - w) * error + w * float(swept.get('steering_error', error))
                            heading = heading * (1.0 - 0.6 * w)
                            debug_info['single_line_swept_weight'] = float(w)
                            if (
                                self.curve_speed_reduction and
                                float(swept.get('min_clearance_cm', 1e9)) < self.min_clearance_warn_cm
                            ):
                                speed = self.min_speed
                                speed_cap = speed
                    else:
                        self._swept_path_info = None

                    error, error_stabilization = self._stabilize_single_line_error(
                        error,
                        'right',
                        prev_seen_side,
                        self.consecutive_single_right,
                        lane_width_px=lane_width_px,
                        img_w=img_w,
                    )
                    if error_stabilization is not None:
                        debug_info['single_line_error_stabilization'] = error_stabilization

                    speed_val = self._current_speed if self._current_speed > 0 else self.base_speed
                    curve_reference = self._get_stanley_curve_reference(
                        fallback_curve=self._swept_path_info
                    ) if isinstance(self._swept_path_info, dict) else None

                    # Physical direct_error_m using the real right mask position + safety margin.
                    if transversal_recovery_active:
                        debug_info['single_line_direct_error_disabled'] = 'transversal_recovery'
                    elif not physical_single_line_mode:
                        debug_info['single_line_direct_error_disabled'] = (
                            f'non_physical_mode:{single_line_mode or "unknown"}'
                        )
                    steering_angle = self._compute_lateral_control(
                        error, heading, speed_val, curve_reference=curve_reference, speed_cap=speed_cap,
                        lane_width_px=lane_width_px, img_w=img_w, direct_error_m=_sl_direct,
                        direct_error_context={"source": "single_line", "side": "right"},
                    )
                    steering_angle = self._enforce_single_line_curve_priority(
                        steering_angle,
                        heading,
                        direct_error_m=_sl_direct,
                        side="right",
                        debug_info=debug_info,
                    )

                    steering_angle = max(-self.max_steering, min(self.max_steering, steering_angle))
                    steering_angle, transition_blend = self._blend_single_line_transition_steering(
                        steering_angle,
                        'right',
                        prev_seen_side,
                        self.consecutive_single_right,
                    )
                    if transition_blend is not None:
                        debug_info['single_line_transition_blend'] = transition_blend
                self.frames_without_line = 0

        elif avg_left is not None and avg_right is not None:
            # ---- BOTH LINES DETECTED ----
            # BFMC logic: if just_seen_two_lines was False, skip this frame (stabilize)
            if self.just_seen_two_lines:
                error, midpoint_x, bottom_left_x, bottom_right_x = self._bfmc_get_error(
                    avg_left, avg_right, img_h, img_w
                )

                # Calibrate single-line heading references from stable two-line geometry.
                # Skip when a line is None: _compute_raw_line_heading(None) returns 0.0
                # which would actively drive the reference toward zero each frame, causing
                # a large camera_shift transient on the first 2→1 single-line frame.
                alpha = max(0.01, min(1.0, self.single_line_heading_ref_alpha))
                if avg_left is not None:
                    left_raw_heading = self._compute_raw_line_heading(avg_left)
                    self._single_line_heading_ref_left = (
                        (1.0 - alpha) * self._single_line_heading_ref_left + alpha * left_raw_heading
                    )
                if avg_right is not None:
                    right_raw_heading = self._compute_raw_line_heading(avg_right)
                    self._single_line_heading_ref_right = (
                        (1.0 - alpha) * self._single_line_heading_ref_right + alpha * right_raw_heading
                    )
                debug_info['single_heading_ref_l'] = self._single_line_heading_ref_left
                debug_info['single_heading_ref_r'] = self._single_line_heading_ref_right

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
                            speed_cap = speed
                else:
                    self._swept_path_info = None

                if self.use_kalman_filter:
                    raw_error = error
                    error = self._kalman_correct_error(error)
                    debug_info['kalman_raw_error'] = raw_error
                    debug_info['kalman_error'] = error
                    debug_info['kalman_age'] = self._kalman_age_frames

                # Compute steering from crosstrack error
                heading = self._compute_heading_error(avg_left, avg_right)
                if stale_grace_active:
                    heading = self._fuse_heading_with_local_hint(heading, debug_info, mode='stale')
                else:
                    heading = self._fuse_heading_with_local_hint(heading, debug_info, mode='normal')
                speed_val = self._current_speed if self._current_speed > 0 else self.base_speed
                curve_reference = self._get_stanley_curve_reference(
                    avg_left=avg_left,
                    avg_right=avg_right,
                    img_h=img_h,
                    img_w=img_w,
                    fallback_curve=self._swept_path_info,
                )
                steering_angle = self._compute_lateral_control(
                    error, heading, speed_val, curve_reference=curve_reference, speed_cap=speed_cap,
                    lane_width_px=abs(float(bottom_right_x) - float(bottom_left_x)),
                    img_w=img_w,
                    direct_error_context={"source": "two_line"},
                )

                # Moving average of last N steering values (smooths erratic readings from lighting)
                self.steer_history.append(steering_angle)
                steering_angle = sum(self.steer_history) / len(self.steer_history)
                steering_angle = max(-self.max_steering, min(self.max_steering, steering_angle))

                self.just_seen_two_lines = False
            else:
                # First frame with two lines after single/none — use previous steering (stabilize)
                # Mark as stabilization so the noise filter doesn't accept this as a real measurement
                self.just_seen_two_lines = True
                steering_angle = getattr(self, 'last_steering', 0)
                self._is_stabilization_frame = True

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
            self._swept_path_info = None
            use_physical_single_line = (self.detection_mode == DetectionMode.AI_LOCAL.value)
            sl_error, sl_heading = self._compute_single_line_error(
                avg_left, 'left', img_h, img_w,
                prefer_center=(True if use_physical_single_line else (prev_seen_side == "both")),
                physical_projection=use_physical_single_line,
            )
            debug_info.update(self._last_single_line_projection_debug)
            sl_heading = self._fuse_heading_with_local_hint(sl_heading, debug_info, mode='single_line')
            debug_info['single_line_resolved_side'] = 'left'
            debug_info.setdefault('single_line_side_source', 'legacy')
            sl_error, error_stabilization = self._stabilize_single_line_error(
                sl_error,
                'left',
                prev_seen_side,
                self.consecutive_single_left,
                lane_width_px=self._resolve_nominal_lane_width_px(img_w),
                img_w=img_w,
            )
            if error_stabilization is not None:
                debug_info['single_line_error_stabilization'] = error_stabilization
            error = sl_error

            if self.use_kalman_filter:
                raw_error = sl_error
                sl_error = self._kalman_correct_error(sl_error)
                error = sl_error
                debug_info['kalman_raw_error'] = raw_error
                debug_info['kalman_error'] = sl_error
                debug_info['kalman_age'] = self._kalman_age_frames

            speed = self.min_speed + 2
            speed_cap = speed
            speed_val = self._current_speed if self._current_speed > 0 else self.base_speed
            steering_angle = self._compute_lateral_control(
                sl_error, sl_heading, speed_val, curve_reference=None, speed_cap=speed_cap,
                img_w=img_w,
                direct_error_context={"source": "single_line", "side": "left"},
            )
            steering_angle = self._enforce_single_line_curve_priority(
                steering_angle,
                sl_heading,
                side="left",
                debug_info=debug_info,
            )

            steering_angle = max(-self.max_steering, min(self.max_steering, steering_angle))
            steering_angle, transition_blend = self._blend_single_line_transition_steering(
                steering_angle,
                'left',
                prev_seen_side,
                self.consecutive_single_left,
            )
            if transition_blend is not None:
                debug_info['single_line_transition_blend'] = transition_blend
            if self.show_debug:
                ctrl_label = "Stanley 1L" if self._should_use_stanley_controller() else "PID 1L"
                print(f"\033[1;97m[ {ctrl_label} ] :\033[0m \033[1;95mLEFT\033[0m "
                      f"err={sl_error:.0f}px head={math.degrees(sl_heading):.1f}° "
                      f"steer={steering_angle:.1f}°")

            self.frames_without_line = 0

        elif avg_right is not None:
            # ---- ONLY RIGHT LINE ----
            self.consecutive_single_right += 1
            self.consecutive_single_left = 0
            self.last_seen_side = "right"
            self._swept_path_info = None
            use_physical_single_line = (self.detection_mode == DetectionMode.AI_LOCAL.value)
            sl_error, sl_heading = self._compute_single_line_error(
                avg_right, 'right', img_h, img_w,
                prefer_center=(True if use_physical_single_line else (prev_seen_side == "both")),
                physical_projection=use_physical_single_line,
            )
            debug_info.update(self._last_single_line_projection_debug)
            sl_heading = self._fuse_heading_with_local_hint(sl_heading, debug_info, mode='single_line')
            debug_info['single_line_resolved_side'] = 'right'
            debug_info.setdefault('single_line_side_source', 'legacy')
            sl_error, error_stabilization = self._stabilize_single_line_error(
                sl_error,
                'right',
                prev_seen_side,
                self.consecutive_single_right,
                lane_width_px=self._resolve_nominal_lane_width_px(img_w),
                img_w=img_w,
            )
            if error_stabilization is not None:
                debug_info['single_line_error_stabilization'] = error_stabilization
            error = sl_error

            if self.use_kalman_filter:
                raw_error = sl_error
                sl_error = self._kalman_correct_error(sl_error)
                error = sl_error
                debug_info['kalman_raw_error'] = raw_error
                debug_info['kalman_error'] = sl_error
                debug_info['kalman_age'] = self._kalman_age_frames

            speed = self.min_speed + 2
            speed_cap = speed
            speed_val = self._current_speed if self._current_speed > 0 else self.base_speed
            steering_angle = self._compute_lateral_control(
                sl_error, sl_heading, speed_val, curve_reference=None, speed_cap=speed_cap,
                img_w=img_w,
                direct_error_context={"source": "single_line", "side": "right"},
            )
            steering_angle = self._enforce_single_line_curve_priority(
                steering_angle,
                sl_heading,
                side="right",
                debug_info=debug_info,
            )

            steering_angle = max(-self.max_steering, min(self.max_steering, steering_angle))
            steering_angle, transition_blend = self._blend_single_line_transition_steering(
                steering_angle,
                'right',
                prev_seen_side,
                self.consecutive_single_right,
            )
            if transition_blend is not None:
                debug_info['single_line_transition_blend'] = transition_blend
            if self.show_debug:
                ctrl_label = "Stanley 1L" if self._should_use_stanley_controller() else "PID 1L"
                print(f"\033[1;97m[ {ctrl_label} ] :\033[0m \033[1;95mRIGHT\033[0m "
                      f"err={sl_error:.0f}px head={math.degrees(sl_heading):.1f}° "
                      f"steer={steering_angle:.1f}°")

            self.frames_without_line = 0

        else:
            # ---- NO LINES ----
            self._swept_path_info = None  # Reset swept path
            self.frames_without_line += 1
            if self._is_ai_local_active():
                steering_angle, speed, blind_mode = self._resolve_ai_local_blind_control(
                    img_w, speed_cap=speed_cap, debug_info=debug_info
                )
                speed_cap = float(speed)
                debug_info['blind_control_mode'] = blind_mode
                if blind_mode == 'route_tracking':
                    debug_info['control_override'] = 'route_blind_fallback'
                elif blind_mode == 'visual_fallback':
                    debug_info['control_override'] = 'visual_blind_hold'
                else:
                    self._current_speed = 0.0
                    debug_info['control_override'] = 'no_lane_safety_stop'
            else:
                blind_hold_steering = self._get_no_lane_hold_steering()

                if self.show_debug:
                    pass
                    #print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mNO LANES\033[0m - frame {self.frames_without_line}")

                can_use_kalman_pred = (
                    self.use_kalman_filter and
                    self._kalman_initialized and
                    kalman_pred_error is not None and
                    self._kalman_age_frames <= self.kalman_max_prediction_frames
                )
                if can_use_kalman_pred:
                    error = kalman_pred_error
                    kalman_prediction_used = True
                    debug_info['kalman_error'] = error
                    debug_info['kalman_age'] = self._kalman_age_frames
                    debug_info['kalman_predicted'] = True
                    if blind_hold_steering is not None:
                        steering_angle = blind_hold_steering
                        debug_info['blind_steer_hold'] = True
                        debug_info['blind_steer_source'] = (
                            'last_good' if getattr(self, '_last_good_steering', None) is not None else 'last'
                        )
                        if self.show_debug:
                            print(f"\033[1;97m[ Kalman ] :\033[0m \033[1;93mHOLD\033[0m - "
                                  f"e={error:.1f}px age={self._kalman_age_frames}/{self.kalman_max_prediction_frames} "
                                  f"steer={steering_angle:.1f}°")
                    else:
                        speed_val = self._current_speed if self._current_speed > 0 else self.base_speed
                        heading_pred = self._heading_error * 0.5  # stale heading is less reliable while blind
                        heading_pred = self._fuse_heading_with_local_hint(heading_pred, debug_info, mode='stale')
                        curve_reference = self._get_stanley_curve_reference()
                        steering_angle = self._compute_lateral_control(
                            error, heading_pred, speed_val, curve_reference=curve_reference,
                            speed_cap=self.min_speed, img_w=img_w
                        )
                        steering_angle *= 0.9  # conservative while running on prediction only
                        steering_angle = max(-self.max_steering, min(self.max_steering, steering_angle))
                        if self.show_debug:
                            print(f"\033[1;97m[ Kalman ] :\033[0m \033[1;93mPREDICT\033[0m - "
                                  f"e={error:.1f}px age={self._kalman_age_frames}/{self.kalman_max_prediction_frames} "
                                  f"steer={steering_angle:.1f}°")
                elif blind_hold_steering is not None:
                    steering_angle = blind_hold_steering
                    debug_info['blind_steer_hold'] = True
                    debug_info['blind_steer_source'] = (
                        'last_good' if getattr(self, '_last_good_steering', None) is not None else 'last'
                    )
                elif hasattr(self, 'last_steering') and self.frames_without_line < 5:
                    steering_angle = self.last_steering * 0.95
                else:
                    steering_angle = None
                _, eff_min_spd = self._get_effective_speeds()
                speed = eff_min_spd
                speed_cap = speed

        if self._is_ai_local_active():
            speed, speed_cap = self._apply_lead_distance_speed_cap(
                speed,
                speed_cap=speed_cap,
                debug_info=debug_info,
            )

        if stale_grace_active:
            _, eff_min_stale = self._get_effective_speeds()
            conservative_cap = float(eff_min_stale)
            speed = min(float(speed), conservative_cap)
            speed_cap = conservative_cap
            debug_info['stale_grace_speed_cap'] = conservative_cap

        # === POST-CHECK: Reject steering if it looks like a noise spike ===
        if steering_angle is not None and self.use_noise_filter:
            blind_control_mode = debug_info.get('blind_control_mode')
            # Skip noise filter for stabilization frames (just_seen_two_lines transition).
            # These frames use last_steering (not a real measurement) and would falsely
            # reset the reject counter, causing a latch-up where the correct steering
            # value is perpetually rejected.
            if blind_control_mode is not None:
                pass  # Blind AI-local fallback/stop is already the safe override.
            elif self._is_stabilization_frame or kalman_prediction_used:
                pass  # Don't check, don't accept, don't reject — just pass through
            else:
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
                    speed_cap = speed
                else:
                    # Good frame - update reference
                    self._accept_frame(error, steering_angle, num_lines)

        recovery_active = False
        if steering_angle is not None or self._recovery_state != "NONE":
            steering_angle, speed, recovery_active = self._check_curve_recovery(
                steering_angle, speed, frame
            )
        if recovery_active:
            debug_info['control_override'] = 'curve_recovery'
        debug_info['recovery_state'] = self._recovery_state

        protective_stop_active = False
        if not recovery_active and self._should_apply_single_line_protective_stop(
            local_mask_guidance, steering_angle, error
        ):
            protective_stop_active = True
            speed = 0.0
            self._current_speed = 0.0
            debug_info['control_override'] = 'single_line_protective_stop'
        debug_info['single_line_stop_active'] = bool(protective_stop_active)

        # Latency safety guard for AI_LOCAL: prevent oscillations when perception is too old.
        if self._is_ai_local_active():
            latency_samples_ms = []
            for key in ('remote_result_age_ms', 'source_frame_age_ms'):
                value = debug_info.get(key)
                if value is None:
                    continue
                try:
                    latency_samples_ms.append(float(value))
                except (TypeError, ValueError):
                    pass

            if latency_samples_ms:
                latency_ms = max(latency_samples_ms)
                warn_ms = max(0.0, float(getattr(self, 'local_ai_latency_warn_ms', 450.0)))
                stop_ms = max(warn_ms + 50.0, float(getattr(self, 'local_ai_latency_stop_ms', 750.0)))
                stop_frames = max(1, int(getattr(self, 'local_ai_latency_stop_frames', 3) or 0))
                if latency_ms >= warn_ms:
                    self._local_ai_high_latency_streak += 1
                    debug_info['latency_guard_ms'] = float(latency_ms)
                    debug_info['latency_guard_streak'] = int(self._local_ai_high_latency_streak)

                    # Cap speed early on high latency so steering corrections remain physically plausible.
                    conservative_cap = float(self.min_speed)
                    speed = min(float(speed), conservative_cap)
                    speed_cap = conservative_cap if speed_cap is None else min(float(speed_cap), conservative_cap)

                    # Blend steering toward previous stable command as latency grows.
                    previous_steering = getattr(self, '_last_good_steering', None)
                    if previous_steering is None:
                        previous_steering = getattr(self, 'last_steering', None)
                    if steering_angle is not None and previous_steering is not None:
                        if stop_ms > warn_ms:
                            ratio = min(1.0, max(0.0, (latency_ms - warn_ms) / (stop_ms - warn_ms)))
                        else:
                            ratio = 1.0
                        blend_ratio = 0.25 + (0.65 * ratio)
                        steering_angle = (
                            (1.0 - blend_ratio) * float(steering_angle) +
                            blend_ratio * float(previous_steering)
                        )
                        steering_angle = max(-self.max_steering, min(self.max_steering, steering_angle))
                        debug_info['latency_guard_blend'] = float(blend_ratio)
                        debug_info['latency_guard_base_steer'] = float(previous_steering)

                    debug_info['latency_guard_state'] = 'warn'
                    if (
                        latency_ms >= stop_ms and
                        self._local_ai_high_latency_streak >= stop_frames and
                        not recovery_active and
                        not protective_stop_active
                    ):
                        speed = 0.0
                        self._current_speed = 0.0
                        debug_info['latency_guard_state'] = 'stop'
                        debug_info['control_override'] = 'latency_safety_stop'
                else:
                    self._local_ai_high_latency_streak = 0
            else:
                self._local_ai_high_latency_streak = 0
        else:
            self._local_ai_high_latency_streak = 0

        # Update state
        if steering_angle is not None:
            if recovery_active:
                self._last_steering_angle = steering_angle
            elif protective_stop_active:
                self.last_steering = steering_angle
                self._last_steering_angle = steering_angle
                if abs(steering_angle) > 8:
                    self.last_turn_direction = 1 if steering_angle > 0 else -1
            else:
                self.last_steering = steering_angle
                self._last_steering_angle = steering_angle  # For hybrid curve system

                # Update turn direction
                if abs(steering_angle) > 8:
                    self.last_turn_direction = 1 if steering_angle > 0 else -1

                # Progressive speed: linearly map steering magnitude to speed range,
                # but never exceed any safety cap set earlier in the control path.
                speed = self._update_progressive_speed(steering_angle, speed_cap=speed_cap)

                # Stanley already scales the crosstrack term by speed, so applying an
                # extra attenuation here weakens correction twice and deviates from
                # the Hoffmann control law. Keep the attenuation only for PID mode.
                if not self._should_use_stanley_controller():
                    eff_max, _ = self._get_effective_speeds()
                    speed_ratio = speed / eff_max
                    steer_attenuation = 1.0 - self.speed_steer_factor * min(speed_ratio, 1.0)
                    steering_angle *= steer_attenuation
                    steering_angle = max(-self.max_steering, min(self.max_steering, steering_angle))

        self._update_dynamic_tracking_metrics(error_px=error)
        debug_info['theta_dmae_deg'] = float(self._theta_dmae_deg)
        debug_info['offset_dma_m'] = float(self._offset_dma_m)

        if self.show_debug and steering_angle is not None:
            if isinstance(local_mask_guidance, dict):
                src = f"mask:{local_mask_guidance.get('mask_label', 'NONE').lower()}"
            else:
                src = "both" if avg_left is not None and avg_right is not None else \
                      "left" if avg_left is not None else \
                      "right" if avg_right is not None else "none"
            control_mode = debug_info.get('control_mode', 'legacy_line_fit')
            if self._should_use_stanley_controller():
                heading_deg = math.degrees(self._heading_error)
                psi_ss_deg = math.degrees(self._stanley_last_psi_ss)
                steer_damp_deg = math.degrees(self.stanley.k_d_steer * self._stanley_last_steer_damping)
                # print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;95mSTANLEY\033[0m - "
                #       f"Mode: {control_mode} "
                #       f"Steer: {steering_angle:.1f}° Error: {error or 0:.0f}px/{self._stanley_last_error_m:.3f}m "
                #       f"Heading: {heading_deg:.1f}° PsiSS: {psi_ss_deg:.1f}° "
                #       f"YawRate: {self._yaw_rate:.2f}/{self._stanley_last_traj_yaw_rate:.2f} "
                #       f"ServoDamp: {steer_damp_deg:.1f}° "
                #       f"v={self._stanley_last_speed_mps:.3f}m/s({self._stanley_last_speed_source}) "
                #       f"scale={self._stanley_last_px_per_cm:.2f}px/cm Source: {src}")
            else:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;96mPID\033[0m - "
                      f"Mode: {control_mode} "
                      f"Steer: {steering_angle:.1f}° Error: {error or 0:.0f}px Source: {src}")

        # Create debug frame only when needed (saves frame.copy() + draw operations per frame)
        if self._needs_debug:
            debug_frame = self._bfmc_draw_debug(
                frame, avg_left, avg_right, error, midpoint_x,
                bottom_left_x, bottom_right_x, steering_angle, debug_info
            )
            self._store_debug_image('final', debug_frame)
        else:
            debug_frame = None

        self._set_frame_trace({
            "status": "processed",
            "detection_mode": self.detection_mode,
            "pipeline_label": debug_info.get("pipeline_label"),
            "control_mode": debug_info.get("control_mode", "legacy_line_fit"),
            "num_lines": int(num_lines),
            "error_px": error,
            "error_m": self._stanley_last_error_m if self._should_use_stanley_controller() else None,
            "midpoint_x": midpoint_x,
            "bottom_left_x": bottom_left_x,
            "bottom_right_x": bottom_right_x,
            "steering_deg": steering_angle,
            "speed": speed,
            "stanley_speed_mps": self._stanley_last_speed_mps if self._should_use_stanley_controller() else None,
            "stanley_speed_source": self._stanley_last_speed_source if self._should_use_stanley_controller() else None,
            "stanley_px_per_cm": self._stanley_last_px_per_cm if self._should_use_stanley_controller() else None,
            "theta_dmae_deg": self._theta_dmae_deg,
            "offset_dma_m": self._offset_dma_m,
            "heading_hint_rad": self._local_ai_heading_hint_rad,
            "heading_hint_confidence": self._local_ai_heading_hint_confidence,
            "road_type_class": self._local_ai_road_type_class,
            "road_type_confidence": self._local_ai_road_type_confidence,
            "lead_distance_class": self._local_ai_lead_distance_class,
            "lead_distance_confidence": self._local_ai_lead_distance_confidence,
            "curve_guard": self._stanley_last_curve_guard,
            "kalman_prediction_used": bool(kalman_prediction_used),
            "recovery_active": bool(recovery_active),
            "recovery_state": self._recovery_state,
            "single_line_stop_active": bool(protective_stop_active),
            "avg_left_line": avg_left[0].tolist() if avg_left is not None else None,
            "avg_right_line": avg_right[0].tolist() if avg_right is not None else None,
            "lane_observation": self._lane_observation_history[-1] if self._lane_observation_history else None,
            "single_line_projection": self._last_single_line_projection_debug,
            "swept_path": self._swept_path_info,
            "single_line_shadow": self._last_sl_shadow_debug,
            "debug": debug_info,
        })

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
            self._last_requested_motor_command = {
                "timestamp": round(time.time(), 3),
                "steering_deg": float(steering_angle),
                "speed": float(speed),
                "steer_x10": int(steer_value),
                "speed_x10": int(speed_value),
                "sign_action_blocked_speed": False,
                "speed_sent": True,
            }

            # If a sign action (stop, crosswalk, etc.) is active:
            # - SPEED: always blocked to not override the sign action's speed
            # - STEER: kept sending for lane alignment UNLESS steer_override_event is
            #   also set (e.g. hardcoded turn), in which case the sign action controls
            #   both speed and steer.
            if self.sign_action_event and self.sign_action_event.is_set():
                self._sign_action_was_active = True
                self._last_requested_motor_command["sign_action_blocked_speed"] = True
                self._last_requested_motor_command["speed_sent"] = False
                if not (self.steer_override_event and self.steer_override_event.is_set()):
                    self.steerMotorSender.send(str(steer_value))
                    if self.show_debug:
                        print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;96mALIGN\033[0m - Steer: {steer_value} (sign action active, speed blocked)")
                return

            # Just resumed from a sign action — reset speed ramp to effective min_speed
            # so the car starts slow instead of jumping to its previous speed.
            # In highway mode, use highway_min_speed so the car doesn't crawl on the highway.
            if self._sign_action_was_active:
                self._sign_action_was_active = False
                _, resume_min = self._get_effective_speeds()
                self._current_speed = resume_min
                speed_value = int(round(resume_min * 10))
                self._last_requested_motor_command["speed"] = float(resume_min)
                self._last_requested_motor_command["speed_x10"] = int(speed_value)
                print(
                    f"\033[1;97m[ Line Following ] :\033[0m \033[1;92mRESUME\033[0m - "
                    f"Speed reset to min ({resume_min}) after sign action"
                )

            # Normal operation — send both steer and speed
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
            self._last_state_change_message = str(message)
            print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;96mSTATE\033[0m - Received: {message}")
            
            try:
                mode_dict = SystemMode[message].value.get("camera", {}).get("lineFollowing", {})
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;96mSTATE\033[0m - lineFollowing config: {mode_dict}")
                
                if mode_dict.get("enabled", False):
                    self.is_line_following_active = True
                    self._last_inactive_log = False  # Reset inactive log flag
                    self._manual_run_log_enabled = False  # Stop manual log when entering active mode
                    if str(message) == "AUTO":
                        self._reset_auto_run_log(message)
                        self._reset_pid_state()
                        self._last_good_steering = 0.0
                        self._noise_reject_count = 0
                    elif str(message) == "PARKING":
                        self._reset_parking_state()
                        self._parking_state = ParkingState.LANE_KEEPING
                        print("\033[1;97m[ Parking ] :\033[0m \033[1;92mLANE_KEEPING\033[0m - Parking mode activated, searching for spot")
                    print("\033[1;97m[ Line Following ] :\033[0m \033[1;92mACTIVATED\033[0m - Line following is now ACTIVE!")
                else:
                    self.is_line_following_active = False
                    self._auto_run_log_enabled = False
                    if self._parking_state != ParkingState.IDLE:
                        self._reset_parking_state()
                        print("\033[1;97m[ Parking ] :\033[0m \033[1;93mCANCELLED\033[0m - Parking aborted due to mode change")
                    if str(message) == "MANUAL":
                        self._reset_manual_run_log(message)
                    else:
                        self._manual_run_log_enabled = False
                    print("\033[1;97m[ Line Following ] :\033[0m \033[1;93mDEACTIVATED\033[0m - Line following is now INACTIVE")
            except KeyError as e:
                print(f"\033[1;97m[ Line Following ] :\033[0m \033[1;91mERROR\033[0m - Unknown mode: {message} - {e}")

    # ------------------------------------------------------------------
    # Parking maneuver helpers
    # ------------------------------------------------------------------

    def _poll_sign_detected(self):
        """Consume SignDetected messages and update parking-spot tracking.

        Returns True if a parking detection was received this cycle.

        The LANE_KEEPING → SPOT_TRACKED transition is gated by metric distance:
        the spot must be within PARKING_TRIGGER_DISTANCE_CM to activate tracking.
        If distance_cm is not available (no px_per_cm calibration yet), falls back
        to triggering on any detection (original behavior).
        """
        msg = self.signDetectedSubscriber.receive()
        if msg is None:
            return False
        try:
            data = msg if isinstance(msg, dict) else json.loads(msg)
        except Exception:
            return False

        current_mode = str(getattr(self, "_last_state_change_message", "") or "").lower()
        self._maneuver_manager.observe_sign(
            data,
            current_mode=current_mode,
            tracking_state=self._tracking_state,
        )

        if data.get("sign") != "parking":
            return False

        self._parking_last_spot_box = data.get("box")
        self._parking_spot_miss_frames = 0

        raw_dist = data.get("distance_cm")
        if raw_dist is not None:
            try:
                self._parking_last_spot_distance_cm = float(raw_dist)
            except (TypeError, ValueError):
                self._parking_last_spot_distance_cm = None
        else:
            self._parking_last_spot_distance_cm = None

        if self._parking_state == ParkingState.LANE_KEEPING:
            dist = self._parking_last_spot_distance_cm
            # Transition only when spot is within trigger range, or if no metric is available
            within_range = (dist is None) or (dist <= PARKING_TRIGGER_DISTANCE_CM)
            if within_range:
                self._parking_state = ParkingState.SPOT_TRACKED
                dist_str = f" at ~{dist:.0f}cm" if dist is not None else " (dist unknown)"
                print(
                    f"\033[1;97m[ Parking ] :\033[0m \033[1;96mSPOT_TRACKED\033[0m - "
                    f"Spot detected{dist_str}"
                )
        elif self._parking_state == ParkingState.SPOT_TRACKED:
            dist = self._parking_last_spot_distance_cm
            if dist is not None and self.show_debug:
                print(
                    f"\033[1;97m[ Parking ] :\033[0m \033[1;94mTRACKING\033[0m - "
                    f"Spot at ~{dist:.0f}cm"
                )
        return True

    def _parking_advance_distance(self, now):
        """Integrate encoder odometry to get distance traveled in the current phase (cm).

        Uses _measured_speed (Nucleo CurrentSpeed in ×10 cm/s units) converted to
        m/s via stanley_measured_speed_to_mps. If the encoder is not reporting
        (speed == 0), accumulation is skipped and only the timer fallback is used.

        During FORWARD_PAST_SPOT, accumulation is paused while the car is still
        in a curve (curve_state IN_CURVE or ENTERING). This ensures the distance is
        counted only once the car is travelling straight, so the maneuver reference
        point is correct regardless of whether parking is triggered mid-curve.

        Returns the accumulated phase distance in cm.
        """
        dt = now - self._parking_last_sm_time if self._parking_last_sm_time > 0.0 else 0.0
        self._parking_last_sm_time = now

        # In FORWARD_PAST_SPOT: wait for the car to finish any residual curve before
        # counting distance (curve_state must be STRAIGHT or EXITING).
        _wait_straight = bool(getattr(_config, "PARKING_CURVE_WAIT_FOR_STRAIGHT", True))
        _in_forward_past = (self._parking_state == ParkingState.FORWARD_PAST_SPOT)
        _still_curving = self._curve_state in ("IN_CURVE", "ENTERING")
        if _in_forward_past and _wait_straight and _still_curving:
            # Keep timer anchor current so the timer fallback doesn't fire while waiting.
            self._parking_phase_start_time = now
            return self._parking_phase_dist_cm  # no accumulation yet

        speed_mps = abs(float(self._measured_speed or 0.0)) * float(self.stanley_measured_speed_to_mps)
        if speed_mps > 0.001 and dt > 0.0:
            self._parking_phase_dist_cm += speed_mps * 100.0 * dt  # m/s → cm/s × s = cm

        return self._parking_phase_dist_cm

    def _parking_phase_complete(self, now, dist_threshold_cm, timer_threshold_s):
        """Return True when the current phase should transition.

        Primary criterion: encoder distance traveled >= dist_threshold_cm.
        Timer fallback: elapsed time >= timer_threshold_s (used when encoder
        speed is 0 throughout the phase, i.e., no odometry available).
        """
        dist_cm = self._parking_advance_distance(now)
        elapsed = now - self._parking_phase_start_time

        # Threshold of 5.0 = 0.5 cm/s actual speed (measured_speed is in ×10 cm/s units).
        # The old threshold of 0.5 (= 0.05 cm/s) triggered odometry mode on encoder
        # noise at rest, accumulating phantom distance. 5.0 filters noise correctly.
        speed_available = abs(float(self._measured_speed or 0.0)) > 5.0
        if speed_available:
            return dist_cm >= dist_threshold_cm
        # Fallback: use timer when encoder is not reporting
        return elapsed >= timer_threshold_s

    def _parking_state_machine(self):
        """Non-blocking parking state machine (called at 20 Hz from thread_work).

        Returns (steering_angle, speed) to be sent to the motors, or
        (None, None) when the state machine is IDLE.

        Phase transitions use encoder odometry (distance in cm) as the primary
        criterion. Timer fallbacks are used when encoder speed == 0.
        """
        now = time.time()

        if self._parking_state == ParkingState.IDLE:
            return None, None

        # --- LANE_KEEPING / SPOT_TRACKED: delegate to normal line following ---
        if self._parking_state in (ParkingState.LANE_KEEPING, ParkingState.SPOT_TRACKED):
            return None, None  # handled by main pipeline

        # ---------------------------------------------------------------
        # Helper: advance to next state, resetting phase accumulators.
        # ---------------------------------------------------------------
        def _advance(next_state, label):
            dist_done = self._parking_phase_dist_cm
            self._parking_state          = next_state
            self._parking_phase_start_time = now
            self._parking_phase_dist_cm  = 0.0
            self._parking_last_sm_time   = now
            print(f"\033[1;97m[ Parking ] :\033[0m \033[1;96m{label}\033[0m "
                  f"(prev dist={dist_done:.0f}cm)")

        # ---------------------------------------------------------------
        # Phase 1: go forward past the spot (straight, no steer).
        # Uses _parking_dynamic_forward_cm (computed at spot-lost transition)
        # instead of the raw PARKING_D_FORWARD_CM constant so that the advance
        # accounts for any remaining distance when the spot exited the camera
        # FOV (typical when parking is detected while exiting a curve).
        # ---------------------------------------------------------------
        if self._parking_state == ParkingState.FORWARD_PAST_SPOT:
            if self._parking_phase_complete(now, self._parking_dynamic_forward_cm, PARKING_T_FORWARD):
                _advance(ParkingState.WAIT_STEER_1, "WAIT_STEER_1 - stopped, turning wheels to entry steer")
            return 0.0, PARKING_FORWARD_SPEED

        # ---------------------------------------------------------------
        # WAIT_STEER_1: vehicle stopped, wheels turning to ENTRY steer.
        # Speed=0 while servo reaches target angle.
        # ---------------------------------------------------------------
        if self._parking_state == ParkingState.WAIT_STEER_1:
            if now - self._parking_phase_start_time >= PARKING_T_WAIT_STEER:
                _advance(ParkingState.REVERSING_ENTRY, "REVERSING_ENTRY - reverse with max entry steer")
            return PARKING_ENTRY_STEER, 0

        # ---------------------------------------------------------------
        # Phase 2: reverse with max ENTRY steer (swing rear into spot)
        # ---------------------------------------------------------------
        if self._parking_state == ParkingState.REVERSING_ENTRY:
            if self._parking_phase_complete(now, PARKING_D_REVERSING_ENTRY_CM, PARKING_T_REVERSING_ENTRY):
                _advance(ParkingState.WAIT_STEER_2, "WAIT_STEER_2 - stopped, turning wheels to align steer")
            return PARKING_ENTRY_STEER, PARKING_REVERSE_SPEED

        # ---------------------------------------------------------------
        # WAIT_STEER_2: vehicle stopped, wheels turning to ALIGN steer (opposite).
        # ---------------------------------------------------------------
        if self._parking_state == ParkingState.WAIT_STEER_2:
            if now - self._parking_phase_start_time >= PARKING_T_WAIT_STEER:
                _advance(ParkingState.REVERSING_ALIGN, "REVERSING_ALIGN - reverse with max align (opposite) steer")
            return PARKING_ALIGN_STEER, 0

        # ---------------------------------------------------------------
        # Phase 3: reverse with max ALIGN steer (opposite) to straighten in spot
        # ---------------------------------------------------------------
        if self._parking_state == ParkingState.REVERSING_ALIGN:
            if self._parking_phase_complete(now, PARKING_D_REVERSING_ALIGN_CM, PARKING_T_REVERSING_ALIGN):
                _advance(ParkingState.WAIT_STEER_3, "WAIT_STEER_3 - stopped, turning wheels back to entry steer")
            return PARKING_ALIGN_STEER, PARKING_REVERSE_SPEED

        # ---------------------------------------------------------------
        # WAIT_STEER_3: vehicle stopped, wheels turning back to ENTRY steer
        # for forward correction (same direction as the initial entry).
        # ---------------------------------------------------------------
        if self._parking_state == ParkingState.WAIT_STEER_3:
            if now - self._parking_phase_start_time >= PARKING_T_WAIT_STEER:
                _advance(ParkingState.FORWARD_CORRECTION, "FORWARD_CORRECTION - forward with max entry steer")
            return PARKING_ENTRY_STEER, 0

        # ---------------------------------------------------------------
        # Phase 4: go FORWARD with the SAME max steer as the initial entry
        # (PARKING_ENTRY_STEER) to align the car parallel to the curb.
        # ---------------------------------------------------------------
        if self._parking_state == ParkingState.FORWARD_CORRECTION:
            if self._parking_phase_complete(now, PARKING_D_FORWARD_CORR_CM, PARKING_T_FORWARD_CORR):
                dist_done = self._parking_phase_dist_cm
                self._parking_state = ParkingState.PARKED
                print(
                    f"\033[1;97m[ Parking ] :\033[0m \033[1;92mPARKED\033[0m - "
                    f"Corrected {dist_done:.0f}cm forward, maneuver complete"
                )
            return PARKING_ENTRY_STEER, PARKING_FORWARD_SPEED

        # ---------------------------------------------------------------
        # PARKED: hold still
        # ---------------------------------------------------------------
        if self._parking_state == ParkingState.PARKED:
            return 0.0, 0.0

        return None, None

    def _reset_parking_state(self):
        """Reset all parking state variables back to IDLE."""
        self._parking_state                = ParkingState.IDLE
        self._parking_last_spot_box        = None
        self._parking_last_spot_distance_cm = None
        self._parking_spot_miss_frames     = 0
        self._parking_phase_start_time     = 0.0
        self._parking_phase_dist_cm        = 0.0
        self._parking_last_sm_time         = 0.0
        self._parking_dynamic_forward_cm   = PARKING_D_FORWARD_CM

    def stop(self):
        """Stop the thread and cleanup."""
        self._stop_hybridnets_client()
        self._stop_supercombo_client()
        if self.show_debug:
            cv2.destroyAllWindows()
        super(threadLineFollowing, self).stop()
