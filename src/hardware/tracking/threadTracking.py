"""
threadTracking — GPS-free dead-reckoning + waypoint navigation thread.

Runs at ~50 Hz inside processCamera.  Receives CurrentSpeed and ImuData from
the serial handler, integrates position with DeadReckoning, follows waypoints
from TrackGraph, and publishes navigation state via TrackingState (in-process
shared object) and Location messages to the dashboard.

TrackingState is intentionally NOT a multiprocessing.Value — the tracking
thread lives inside the same OS process as threadLineFollowing, so a plain
Python object with a threading.Lock is sufficient and avoids pickling overhead.
"""

import ast
import math
import os
import threading
import time

from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.allMessages import (
    CurrentSpeed,
    CurrentSteer,
    ImuData,
    Location,
    NavigationCommand,
    NavigationStatus,
    SpeedMotor,
    SignDetected,
    StateChange,
    SteerMotor,
)

from src.hardware.tracking.deadReckoning import DeadReckoning
from src.hardware.tracking.trackGraph import ATTR_STOPLINE, TrackGraph
from src.hardware.tracking.pathManager import PathManager

try:
    import config as cfg
    _GRAPHML_PATH = getattr(cfg, "TRACKING_GRAPHML", "Track GraphML File.graphml")
    _SEMANTICS_PATH = getattr(cfg, "TRACKING_SEMANTICS", "track_semantics.json")
    _STEP_M = getattr(cfg, "TRACKING_WAYPOINT_STEP_M", 0.05)
    _ADVANCE_DIST = getattr(cfg, "TRACKING_ADVANCE_DIST_M", 0.15)
    _INTERSECTION_LOOKAHEAD = getattr(cfg, "TRACKING_INTERSECTION_LOOKAHEAD_M", 0.40)
    _PRECISION_LOOKAHEAD_M = getattr(cfg, "TRACKING_PRECISION_LOOKAHEAD_M", 0.10)
    _MAP_MATCH_SEARCH_WP = getattr(cfg, "TRACKING_MAP_MATCH_SEARCH_WP", 18)
    _MAP_MATCH_DISTANCE_W = getattr(cfg, "TRACKING_MAP_MATCH_DISTANCE_W", 1.0)
    _MAP_MATCH_HEADING_W = getattr(cfg, "TRACKING_MAP_MATCH_HEADING_W", 0.35)
    _SHOW_WINDOW = getattr(cfg, "TRACKING_SHOW_WINDOW", True)
    _DEBUG_LOG = getattr(cfg, "TRACKING_DEBUG_LOG", False)
    _LOOP_HZ = 50
    # Speed-adaptive lookahead: lookahead_m = max(_ADVANCE_DIST, v * _LOOKAHEAD_TIME_S).
    # Compensates for servo + control-loop lag (~270ms measured). At 50cm/s this
    # gives 30cm lookahead instead of 15cm, matching actual system latency.
    _LOOKAHEAD_TIME_S  = getattr(cfg, "TRACKING_LOOKAHEAD_TIME_S", 0.6)
    _MAX_LOOKAHEAD_M   = getattr(cfg, "TRACKING_MAX_LOOKAHEAD_M",  0.80)
    # Bicycle model dead-reckoning between IMU updates
    _WHEELBASE_M       = getattr(cfg, "TRACKING_WHEELBASE_M", 0.260)
    # Steering gain for dead reckoning: physical_wheel_angle / commanded_angle.
    # Gain > 1.0 amplifies the measured steering angle seen by the DR model.
    # Keep the default at 1.0 so the measured steering is trusted directly.
    _DR_SPEED_SCALE    = float(getattr(cfg, "TRACKING_DR_SPEED_SCALE", 1.0) or 1.0)
    _STEER_GAIN_DR     = getattr(cfg, "TRACKING_STEER_GAIN_DR", 1.0)
    # Tracking now keeps the steering sign aligned with the actuator feedback so
    # the same convention flows through controller, telemetry, and DR.
    _STEER_SIGN_DR     = float(getattr(cfg, "TRACKING_STEER_SIGN_DR", 1.0) or 1.0)
    # First-order lag filter on the steer angle fed to dead reckoning.
    # Models the servo actuator delay (time for wheels to reach commanded angle).
    # 1.0 = instant (no lag), 0.0 = never responds. Good starting value: 0.5–0.8.
    _STEER_LAG_ALPHA   = float(getattr(cfg, "TRACKING_STEER_LAG_ALPHA", 1.0) or 1.0)
    # Yaw EKF: fuse IMU absolute heading with kinematic yaw rate.
    # K = P / (P + R), R = R_STRAIGHT + R_STEER_K * steer_rad²
    _YAW_EKF_Q          = float(getattr(cfg, "TRACKING_YAW_EKF_Q",           1e-4) or 1e-4)
    _YAW_EKF_R_STRAIGHT = float(getattr(cfg, "TRACKING_YAW_EKF_R_STRAIGHT",  0.005) or 0.005)
    _YAW_EKF_R_STEER_K  = float(getattr(cfg, "TRACKING_YAW_EKF_R_STEER_K",  50.0) or 50.0)
    _YAW_EKF_P_INIT     = float(getattr(cfg, "TRACKING_YAW_EKF_P_INIT",      0.5)  or 0.5)
    _CAMERA_LATERAL_CORRECTION_GAIN = getattr(
        cfg, "TRACKING_CAMERA_LATERAL_CORRECTION_GAIN", 0.35
    )
    _CAMERA_LATERAL_CORRECTION_MAX_M = getattr(
        cfg, "TRACKING_CAMERA_LATERAL_CORRECTION_MAX_M", 0.08
    )
    _CAMERA_LATERAL_CORRECTION_STEP_MAX_M = getattr(
        cfg, "TRACKING_CAMERA_LATERAL_CORRECTION_STEP_MAX_M", 0.015
    )
    _CAMERA_LATERAL_CORRECTION_COOLDOWN_S = getattr(
        cfg, "TRACKING_CAMERA_LATERAL_CORRECTION_COOLDOWN_S", 0.10
    )
    _VISUAL_LANE_RELOCALIZATION_GAIN = getattr(
        cfg, "TRACKING_VISUAL_LANE_RELOCALIZATION_GAIN", 0.10
    )
    _VISUAL_LANE_RELOCALIZATION_ENABLED = bool(
        getattr(cfg, "TRACKING_VISUAL_LANE_RELOCALIZATION_ENABLED", True)
    )
    _VISUAL_LANE_RELOCALIZATION_MAX_M = getattr(
        cfg, "TRACKING_VISUAL_LANE_RELOCALIZATION_MAX_M", 0.03
    )
    _VISUAL_LANE_RELOCALIZATION_MIN_RAW_ERROR_M = getattr(
        cfg, "TRACKING_VISUAL_LANE_RELOCALIZATION_MIN_RAW_ERROR_M", 0.01
    )
    _VISUAL_LANE_RELOCALIZATION_MAX_RAW_ERROR_M = getattr(
        cfg, "TRACKING_VISUAL_LANE_RELOCALIZATION_MAX_RAW_ERROR_M", 0.25
    )
    _VISUAL_LANE_RELOCALIZATION_COOLDOWN_S = getattr(
        cfg, "TRACKING_VISUAL_LANE_RELOCALIZATION_COOLDOWN_S", 0.10
    )
    _VISUAL_LANE_RELOCALIZATION_SPEED_MIN_MPS = getattr(
        cfg, "TRACKING_VISUAL_LANE_RELOCALIZATION_SPEED_MIN_MPS", 0.05
    )
    _SEMANTIC_MATCH_WINDOW_S = getattr(cfg, "TRACKING_SEMANTIC_MATCH_WINDOW_S", 1.0)
    _SEMANTIC_RELOCALIZATION_MAX_DISTANCE_M = getattr(
        cfg, "TRACKING_SEMANTIC_RELOCALIZATION_MAX_DISTANCE_M", 0.45
    )
    _SEMANTIC_RELOCALIZATION_MAX_MAP_ERROR_M = getattr(
        cfg, "TRACKING_SEMANTIC_RELOCALIZATION_MAX_MAP_ERROR_M", 0.30
    )
    _SEMANTIC_RELOCALIZATION_DISTANCE_TOLERANCE_M = getattr(
        cfg, "TRACKING_SEMANTIC_RELOCALIZATION_DISTANCE_TOLERANCE_M", 0.25
    )
    _SEMANTIC_RELOCALIZATION_COOLDOWN_S = getattr(
        cfg, "TRACKING_SEMANTIC_RELOCALIZATION_COOLDOWN_S", 0.75
    )
    _STOPLINE_NODE_ATTR = int(getattr(cfg, "TRACKING_STOPLINE_NODE_ATTR", ATTR_STOPLINE) or ATTR_STOPLINE)
    _VISUAL_STOPLINE_EVENT_MAX_AGE_S = getattr(
        cfg, "TRACKING_VISUAL_STOPLINE_EVENT_MAX_AGE_S", 0.60
    )
    _VISUAL_STOPLINE_RELOCALIZATION_COOLDOWN_S = getattr(
        cfg, "TRACKING_VISUAL_STOPLINE_RELOCALIZATION_COOLDOWN_S", 1.00
    )
    _VISUAL_STOPLINE_ROUTE_BEHIND_M = getattr(
        cfg, "TRACKING_VISUAL_STOPLINE_ROUTE_BEHIND_M", 0.25
    )
    _VISUAL_STOPLINE_ROUTE_AHEAD_M = getattr(
        cfg, "TRACKING_VISUAL_STOPLINE_ROUTE_AHEAD_M", 0.85
    )
    _VISUAL_STOPLINE_MAX_MAP_ERROR_M = getattr(
        cfg, "TRACKING_VISUAL_STOPLINE_MAX_MAP_ERROR_M", 0.75
    )
    _SPEED_FEEDBACK_TIMEOUT_S = getattr(
        cfg, "TRACKING_SPEED_FEEDBACK_TIMEOUT_S", 0.35
    )
    _COMMAND_SPEED_FALLBACK_TIMEOUT_S = getattr(
        cfg, "TRACKING_COMMAND_SPEED_FALLBACK_TIMEOUT_S", 0.50
    )
    _COMMAND_SPEED_FALLBACK_ENABLED = bool(
        getattr(cfg, "TRACKING_COMMAND_SPEED_FALLBACK_ENABLED", True)
    )
    _STEER_FEEDBACK_TIMEOUT_S = getattr(
        cfg, "TRACKING_STEER_FEEDBACK_TIMEOUT_S", 0.35
    )
except Exception:
    _GRAPHML_PATH = "Track GraphML File.graphml"
    _SEMANTICS_PATH = "track_semantics.json"
    _STEP_M = 0.05
    _ADVANCE_DIST = 0.15
    _INTERSECTION_LOOKAHEAD = 0.40
    _PRECISION_LOOKAHEAD_M = 0.10
    _MAP_MATCH_SEARCH_WP = 18
    _MAP_MATCH_DISTANCE_W = 1.0
    _MAP_MATCH_HEADING_W = 0.35
    _SHOW_WINDOW = True
    _DEBUG_LOG = False
    _LOOP_HZ = 50
    _LOOKAHEAD_TIME_S  = 0.6
    _MAX_LOOKAHEAD_M   = 0.80
    _WHEELBASE_M       = 0.260
    _STEER_GAIN_DR     = 1.0
    _STEER_SIGN_DR     = 1.0
    _STEER_LAG_ALPHA   = 1.0
    _YAW_EKF_Q          = 1e-4
    _YAW_EKF_R_STRAIGHT = 0.005
    _YAW_EKF_R_STEER_K  = 50.0
    _YAW_EKF_P_INIT     = 0.5
    _CAMERA_LATERAL_CORRECTION_GAIN = 0.18
    _CAMERA_LATERAL_CORRECTION_MAX_M = 0.02
    _CAMERA_LATERAL_CORRECTION_STEP_MAX_M = 0.015
    _CAMERA_LATERAL_CORRECTION_COOLDOWN_S = 0.10
    _VISUAL_LANE_RELOCALIZATION_GAIN = 0.10
    _VISUAL_LANE_RELOCALIZATION_ENABLED = True
    _VISUAL_LANE_RELOCALIZATION_MAX_M = 0.03
    _VISUAL_LANE_RELOCALIZATION_MIN_RAW_ERROR_M = 0.01
    _VISUAL_LANE_RELOCALIZATION_MAX_RAW_ERROR_M = 0.25
    _VISUAL_LANE_RELOCALIZATION_COOLDOWN_S = 0.10
    _VISUAL_LANE_RELOCALIZATION_SPEED_MIN_MPS = 0.05
    _SEMANTIC_MATCH_WINDOW_S = 1.0
    _SEMANTIC_RELOCALIZATION_MAX_DISTANCE_M = 0.45
    _SEMANTIC_RELOCALIZATION_MAX_MAP_ERROR_M = 0.30
    _SEMANTIC_RELOCALIZATION_DISTANCE_TOLERANCE_M = 0.25
    _SEMANTIC_RELOCALIZATION_COOLDOWN_S = 0.75
    _STOPLINE_NODE_ATTR = ATTR_STOPLINE
    _VISUAL_STOPLINE_EVENT_MAX_AGE_S = 0.60
    _VISUAL_STOPLINE_RELOCALIZATION_COOLDOWN_S = 1.00
    _VISUAL_STOPLINE_ROUTE_BEHIND_M = 0.25
    _VISUAL_STOPLINE_ROUTE_AHEAD_M = 0.85
    _VISUAL_STOPLINE_MAX_MAP_ERROR_M = 0.75
    _SPEED_FEEDBACK_TIMEOUT_S = 0.35
    _COMMAND_SPEED_FALLBACK_TIMEOUT_S = 0.50
    _COMMAND_SPEED_FALLBACK_ENABLED = True
    _STEER_FEEDBACK_TIMEOUT_S = 0.35

# Maximum plausible physical yaw rate of the vehicle (rad/s).
# Used to compute a dynamic re-zero detection threshold that scales with the
# actual time since the last IMU sample, instead of a fixed angle.
# 150°/s is a generous upper bound for this car at any realistic speed.
_MAX_PHYSICAL_YAW_RATE_RADS = math.radians(150.0)

# Measured servo angle (degrees) above which the IMU absolute heading
# correction is SUPPRESSED.  Set to 0.0 (i.e. the condition
# `abs_steer < 0.0` is always False) to disable the correction entirely.
# The BNO055 magnetometer stays corrupted *after* heavy steering ends —
# not just during it.  When the servo returns below any finite threshold
# the magnetometer still reads the biased value accumulated during the
# heavy-steer period, and the rate-limited correction applies 15°/frame
# of wrong heading, destroying the DR estimate.
# The bicycle model alone provides heading integration; the initial
# calibration (first IMU message, _yaw_offset) is still used.
_IMU_STEER_INHIBIT_DEG = 0.0

# Maximum dt (seconds) used for integration steps (yaw bicycle model and DR
# position update).  Caps the error introduced by frame drops: a 440ms gap at
# high speed would otherwise produce ~6-7cm of position error in an unknown
# direction.  When a gap exceeds this limit the integration is truncated; the
# IMU absolute yaw correction on the next frame recovers heading accuracy.
_MAX_INTEGRATION_DT = 0.15

# Attribute name → human label for the log
_ATTR_NAMES = {0: "normal", 1: "crosswalk", 2: "intersection", 3: "oneway",
               4: "hw_left", 5: "hw_right", 6: "roundabout", 7: "stopline",
               8: "dotted", 9: "dotted_xwalk", 11: "intersection_exit"}

# How many dense waypoints correspond to the intersection lookahead distance
_LOOKAHEAD_PTS = max(2, int(_INTERSECTION_LOOKAHEAD / _STEP_M))

# BFMC track physical dimensions in metres (for Location scaling to dashboard)
_TRACK_W_M = 20.67
_TRACK_H_M = 13.76


class TrackingState:
    """Thread-safe shared state between threadTracking and threadLineFollowing.

    Attributes (readable by any thread in the same process):
        x, y, yaw        Current raw pose estimate (dead reckoning + sensor fusion).
        error_m          Lateral crosstrack error vs. current waypoint.
        heading_rad      Heading error vs. current waypoint tangent.
        path_psi         Path tangent angle at the current waypoint (radians).
        speed_mps        Last received forward speed.
        wp_idx           Current waypoint index in the dense spline array.
        waypoint_mode_active  True when the car should use graph waypoints
                               instead of visual lane detection (precision zones).
        node_attr        GraphML attribute of the current waypoint region.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.raw_x = 0.0
        self.raw_y = 0.0
        self.raw_yaw = 0.0
        self.matched_x = 0.0
        self.matched_y = 0.0
        self.matched_yaw = 0.0
        self.steer_rad = 0.0     # Current steering angle (rad) — used for arrow display
        self.error_m = 0.0
        self.heading_rad = 0.0
        self.path_psi = 0.0
        self.path_kappa = 0.0   # Signed path curvature at current wp (1/m)
        self.path_heading_change_rad = 0.0  # Total heading change over next 1.5m of path
        self.map_match_error_m = 0.0
        self.speed_mps = 0.0
        self.speed_source = "none"
        self.speed_feedback_age_s = None
        self.speed_command_age_s = None
        self.wp_idx = 0
        self.target_idx = 0
        self.route_active = False
        self.route_id = None
        self.current_node_id = None
        self.current_node_attr = 0
        self.upcoming_node_id = None
        self.upcoming_node_attr = 0
        self.maneuver_type = "none"
        self.destination_node_id = None
        self.destination_label = None
        self.route_queue = []
        self.route_progress = 0.0
        self.route_points = []
        self.route_completed = False
        self.route_replans = 0
        self.route_source = "none"
        self.destination_point = None
        self.next_semantic_id = None
        self.next_semantic_type = None
        self.next_semantic_label = None
        self.next_semantic_distance_m = None
        self.expected_control_type = None
        self.current_zone_ids = []
        self.current_zone_types = []
        self.map_metadata = {}
        self.available_destinations = []
        self.waypoint_mode_active = False
        self.node_attr = 0
        self.camera_lateral_correction_m = 0.0
        self.lane_measurement_reliable = False
        self.raw_lateral_error_m = 0.0
        self.relocalization_mode = "map_match"
        self.last_relocalization_source = "map_match"
        self.last_relocalization_error_m = 0.0
        self.localization_confidence = 0.0
        self.initialized = False
        self.imu_received = False   # True once a real IMU message has been parsed
        # Reference to the dead reckoning instance — set by threadTracking so
        # threadLineFollowing can push lane-based lateral corrections.
        self._dr = None
        # Camera-based yaw hint: set by threadLineFollowing, consumed by threadTracking.
        self._cam_yaw_hint_rad = None
        self._cam_yaw_hint_conf = 0.0
        self._cam_yaw_hint_fresh = False
        # Public snapshot of the last hint (for logging — never consumed, always readable).
        self.last_cam_yaw_hint_deg = 0.0
        self.last_cam_yaw_hint_conf = 0.0
        # Last yaw correction applied (for logging only).
        self.last_yaw_correction_deg = 0.0
        self._last_camera_lateral_correction_monotonic = 0.0
        # Stopline visual state: written by threadLineFollowing, consumed by
        # threadTracking when a stable visible stopline disappears.
        self.stopline_visible = False
        self.stopline_stable = False
        self.stopline_distance_m = None
        self.stopline_confidence = 0.0
        self.stopline_source = "none"
        self.stopline_expected_node_id = None
        self.stopline_expected_node_attr = 0
        self.stopline_pass_count = 0
        self.stopline_last_pass_distance_m = None
        self._stopline_last_seen_monotonic = 0.0
        self._stopline_last_pass_monotonic = 0.0
        self._stopline_pass_event = None
        self.control_steering_deg = None
        self.control_speed_cmd = None
        self.control_authority = "none"
        self.control_safety_override = False
        self.control_reason = "none"
        # Acados full-MPC reference trajectory (set by threadTracking)
        self.mpc_state_refs = None   # (N+1, 3) ndarray or None
        self.mpc_input_refs = None   # (N, 2) ndarray or None

    def update(self, x, y, yaw, error_m, heading_rad, path_psi, path_kappa,
               speed_mps, wp_idx, waypoint_mode, node_attr, imu_received=False,
               path_heading_change_rad=0.0,
               speed_source="none", speed_feedback_age_s=None, speed_command_age_s=None,
               steer_rad=0.0, target_idx=None, raw_x=None, raw_y=None, raw_yaw=None,
               matched_x=None, matched_y=None, matched_yaw=None, map_match_error_m=0.0,
               route_active=False, route_id=None, current_node_id=None,
               current_node_attr=0, upcoming_node_id=None, upcoming_node_attr=0,
               maneuver_type="none", destination_node_id=None, route_progress=0.0,
               route_points=None, route_completed=False, route_replans=0,
               route_source="none", destination_point=None, destination_label=None,
               route_queue=None, next_semantic_id=None, next_semantic_type=None,
               next_semantic_label=None, next_semantic_distance_m=None,
               expected_control_type=None, current_zone_ids=None,
               current_zone_types=None, map_metadata=None,
               available_destinations=None, relocalization_mode="map_match",
               last_relocalization_source="map_match",
               last_relocalization_error_m=0.0, raw_lateral_error_m=0.0):
        with self._lock:
            self.x = x
            self.y = y
            self.yaw = yaw
            self.raw_x = float(x if raw_x is None else raw_x)
            self.raw_y = float(y if raw_y is None else raw_y)
            self.raw_yaw = float(yaw if raw_yaw is None else raw_yaw)
            self.matched_x = float(x if matched_x is None else matched_x)
            self.matched_y = float(y if matched_y is None else matched_y)
            self.matched_yaw = float(yaw if matched_yaw is None else matched_yaw)
            self.steer_rad = steer_rad
            self.error_m = error_m
            self.heading_rad = heading_rad
            self.path_psi = path_psi
            self.path_kappa = path_kappa
            self.path_heading_change_rad = float(path_heading_change_rad)
            self.map_match_error_m = float(map_match_error_m)
            self.speed_mps = speed_mps
            self.speed_source = str(speed_source or "none")
            self.speed_feedback_age_s = (
                float(speed_feedback_age_s)
                if speed_feedback_age_s is not None else None
            )
            self.speed_command_age_s = (
                float(speed_command_age_s)
                if speed_command_age_s is not None else None
            )
            self.wp_idx = wp_idx
            self.target_idx = int(wp_idx if target_idx is None else target_idx)
            self.route_active = bool(route_active)
            self.route_id = route_id
            self.current_node_id = current_node_id
            self.current_node_attr = int(current_node_attr or 0)
            self.upcoming_node_id = upcoming_node_id
            self.upcoming_node_attr = int(upcoming_node_attr or 0)
            self.maneuver_type = str(maneuver_type or "none")
            self.destination_node_id = destination_node_id
            self.destination_label = destination_label
            self.route_queue = list(route_queue or [])
            self.route_progress = float(route_progress or 0.0)
            self.route_points = list(route_points or [])
            self.route_completed = bool(route_completed)
            self.route_replans = int(route_replans or 0)
            self.route_source = str(route_source or "none")
            self.destination_point = destination_point
            self.next_semantic_id = next_semantic_id
            self.next_semantic_type = next_semantic_type
            self.next_semantic_label = next_semantic_label
            self.next_semantic_distance_m = (
                float(next_semantic_distance_m)
                if next_semantic_distance_m is not None else None
            )
            self.expected_control_type = expected_control_type
            self.current_zone_ids = list(current_zone_ids or [])
            self.current_zone_types = list(current_zone_types or [])
            self.map_metadata = dict(map_metadata or {})
            self.available_destinations = list(available_destinations or [])
            self.waypoint_mode_active = waypoint_mode
            self.node_attr = node_attr
            self.raw_lateral_error_m = float(raw_lateral_error_m or 0.0)
            self.relocalization_mode = str(relocalization_mode or "map_match")
            self.last_relocalization_source = str(last_relocalization_source or "map_match")
            self.last_relocalization_error_m = float(last_relocalization_error_m or 0.0)
            self.localization_confidence = float(getattr(self, "localization_confidence", 0.0) or 0.0)
            self.initialized = True
            if imu_received:
                self.imu_received = True

    def update_from_pose_estimate(self, pose_estimate) -> None:
        with self._lock:
            self.x = float(getattr(pose_estimate.fused_pose, "x", 0.0))
            self.y = float(getattr(pose_estimate.fused_pose, "y", 0.0))
            self.yaw = float(getattr(pose_estimate.fused_pose, "yaw", 0.0))
            self.raw_x = float(getattr(pose_estimate.raw_pose, "x", self.x))
            self.raw_y = float(getattr(pose_estimate.raw_pose, "y", self.y))
            self.raw_yaw = float(getattr(pose_estimate.raw_pose, "yaw", self.yaw))
            self.speed_mps = float(getattr(pose_estimate, "speed_mps", self.speed_mps))
            self.speed_source = str(getattr(pose_estimate, "speed_source", self.speed_source) or "none")
            self.speed_feedback_age_s = getattr(pose_estimate, "speed_feedback_age_s", self.speed_feedback_age_s)
            self.speed_command_age_s = getattr(pose_estimate, "speed_command_age_s", self.speed_command_age_s)
            self.steer_rad = float(getattr(pose_estimate, "steer_rad", self.steer_rad))
            self.raw_lateral_error_m = float(getattr(pose_estimate, "raw_lateral_error_m", 0.0) or 0.0)
            self.lane_measurement_reliable = bool(getattr(pose_estimate, "lane_measurement_reliable", False))
            self.camera_lateral_correction_m = float(
                getattr(pose_estimate, "camera_lateral_correction_m", 0.0) or 0.0
            )
            self.relocalization_mode = str(getattr(pose_estimate, "relocalization_mode", self.relocalization_mode) or "dead_reckoning")
            self.last_relocalization_source = str(
                getattr(pose_estimate, "last_relocalization_source", self.last_relocalization_source) or "none"
            )
            self.last_relocalization_error_m = float(
                getattr(pose_estimate, "last_relocalization_error_m", self.last_relocalization_error_m) or 0.0
            )
            self.localization_confidence = float(getattr(pose_estimate, "localization_confidence", 0.0) or 0.0)
            if bool(getattr(pose_estimate, "imu_received", False)):
                self.imu_received = True
            self.initialized = True

    def update_from_route_context(self, route_context) -> None:
        with self._lock:
            self.matched_x = float(getattr(route_context.matched_pose, "x", self.matched_x))
            self.matched_y = float(getattr(route_context.matched_pose, "y", self.matched_y))
            self.matched_yaw = float(getattr(route_context.matched_pose, "yaw", self.matched_yaw))
            self.map_match_error_m = float(getattr(route_context, "map_match_error_m", 0.0) or 0.0)
            self.wp_idx = int(getattr(route_context, "matched_idx", self.wp_idx) or 0)
            self.target_idx = int(getattr(route_context, "target_idx", self.target_idx) or 0)
            self.error_m = float(getattr(route_context, "error_m", 0.0) or 0.0)
            self.heading_rad = float(getattr(route_context, "heading_rad", 0.0) or 0.0)
            self.path_psi = float(getattr(route_context, "path_psi", 0.0) or 0.0)
            self.path_kappa = float(getattr(route_context, "path_kappa", 0.0) or 0.0)
            self.path_heading_change_rad = float(getattr(route_context, "path_heading_change_rad", 0.0) or 0.0)
            self.route_active = bool(getattr(route_context, "route_active", False))
            self.route_id = getattr(route_context, "route_id", self.route_id)
            self.current_node_id = getattr(route_context, "current_node_id", self.current_node_id)
            self.current_node_attr = int(getattr(route_context, "current_node_attr", 0) or 0)
            self.upcoming_node_id = getattr(route_context, "upcoming_node_id", self.upcoming_node_id)
            self.upcoming_node_attr = int(getattr(route_context, "upcoming_node_attr", 0) or 0)
            self.maneuver_type = str(getattr(route_context, "maneuver_type", self.maneuver_type) or "none")
            self.destination_node_id = getattr(route_context, "destination_node_id", self.destination_node_id)
            self.destination_label = getattr(route_context, "destination_label", self.destination_label)
            self.route_queue = list(getattr(route_context, "route_queue", []) or [])
            self.route_progress = float(getattr(route_context, "route_progress", 0.0) or 0.0)
            self.route_points = list(getattr(route_context, "route_points", []) or [])
            self.route_completed = bool(getattr(route_context, "route_completed", False))
            self.route_replans = int(getattr(route_context, "replans", 0) or 0)
            self.route_source = str(getattr(route_context, "route_source", "none") or "none")
            self.destination_point = getattr(route_context, "destination_point", self.destination_point)
            self.next_semantic_id = getattr(route_context, "next_semantic_id", self.next_semantic_id)
            self.next_semantic_type = getattr(route_context, "next_semantic_type", self.next_semantic_type)
            self.next_semantic_label = getattr(route_context, "next_semantic_label", self.next_semantic_label)
            self.next_semantic_distance_m = getattr(route_context, "next_semantic_distance_m", self.next_semantic_distance_m)
            self.expected_control_type = getattr(route_context, "expected_control_type", self.expected_control_type)
            self.current_zone_ids = list(getattr(route_context, "current_zone_ids", []) or [])
            self.current_zone_types = list(getattr(route_context, "current_zone_types", []) or [])
            self.map_metadata = dict(getattr(route_context, "map_metadata", {}) or {})
            self.available_destinations = list(getattr(route_context, "available_destinations", []) or [])
            self.waypoint_mode_active = bool(getattr(route_context, "waypoint_mode_active", False))
            self.node_attr = int(self.upcoming_node_attr or self.current_node_attr or 0)
            self.initialized = True

    def update_from_control_decision(self, decision) -> None:
        with self._lock:
            self.control_steering_deg = getattr(decision, "steering_deg", None)
            self.control_speed_cmd = getattr(decision, "speed_cmd", None)
            self.control_authority = str(getattr(decision, "authority", "none") or "none")
            self.control_safety_override = bool(getattr(decision, "safety_override", False))
            self.control_reason = str(getattr(decision, "reason", "none") or "none")

    def set_camera_yaw_hint(self, camera_yaw_rad: float, confidence: float) -> None:
        """Store a camera-estimated world-frame yaw for threadTracking to consume.

        Called by threadLineFollowing when both lane lines are visible and the
        car heading relative to the lane is reliable.

        Args:
            camera_yaw_rad: Estimated absolute yaw in map frame (rad).
            confidence:     Quality of the estimate, 0.0-1.0.
        """
        _conf = float(max(0.0, min(1.0, confidence)))
        with self._lock:
            self._cam_yaw_hint_rad = float(camera_yaw_rad)
            self._cam_yaw_hint_conf = _conf
            self._cam_yaw_hint_fresh = True
            # Keep public snapshot for logging (never consumed).
            self.last_cam_yaw_hint_deg = math.degrees(float(camera_yaw_rad))
            self.last_cam_yaw_hint_conf = _conf

    def consume_camera_yaw_hint(self):
        """Return (camera_yaw_rad, confidence) if a fresh hint exists, else (None, 0).

        Marks the hint as consumed so the same value is not applied twice.
        """
        with self._lock:
            if not self._cam_yaw_hint_fresh:
                return None, 0.0
            self._cam_yaw_hint_fresh = False
            return self._cam_yaw_hint_rad, self._cam_yaw_hint_conf

    def correct_lateral(self, lateral_error_m: float) -> None:
        """Push a lane-detection-based lateral correction into dead reckoning.

        Called by threadLineFollowing when two lane lines are visible and the
        measured crosstrack error is reliable.  Nudges the DR position so the
        next heading/error computation is more accurate.

        Args:
            lateral_error_m: Signed lane error from lane detection (m).
                             Positive = car left of lane centre.
        """
        now = time.monotonic()
        with self._lock:
            dr = self._dr
            psi = self.path_psi
            correction_m = float(lateral_error_m)
            self.lane_measurement_reliable = True
            if (now - self._last_camera_lateral_correction_monotonic) < float(
                _CAMERA_LATERAL_CORRECTION_COOLDOWN_S
            ):
                self.camera_lateral_correction_m = 0.0
                return
            max_step_m = float(_CAMERA_LATERAL_CORRECTION_STEP_MAX_M)
            if max_step_m > 0.0:
                correction_m = max(-max_step_m, min(max_step_m, correction_m))
            self.camera_lateral_correction_m = correction_m
            if abs(correction_m) > 1e-9:
                self._last_camera_lateral_correction_monotonic = now
        if dr is not None and abs(correction_m) > 1e-9:
            dr.correct_lateral(correction_m, psi)

    def set_lane_measurement_state(self, reliable: bool, applied_correction_m: float = 0.0) -> None:
        with self._lock:
            self.lane_measurement_reliable = bool(reliable)
            self.camera_lateral_correction_m = float(applied_correction_m if reliable else 0.0)
            if not reliable:
                self.raw_lateral_error_m = 0.0

    def set_stopline_visual_state(
        self,
        visible: bool,
        *,
        stable: bool = False,
        distance_m: float | None = None,
        confidence: float = 0.0,
        source: str = "opencv_bev",
        expected_node_id=None,
        expected_node_attr: int = 0,
        pass_event: dict | None = None,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            self.stopline_visible = bool(visible)
            self.stopline_stable = bool(stable)
            self.stopline_distance_m = (
                float(distance_m) if distance_m is not None else None
            )
            self.stopline_confidence = float(max(0.0, min(1.0, confidence)))
            self.stopline_source = str(source or "none")
            self.stopline_expected_node_id = expected_node_id
            self.stopline_expected_node_attr = int(expected_node_attr or 0)
            if visible:
                self._stopline_last_seen_monotonic = float(now)
            if pass_event is not None:
                payload = dict(pass_event)
                payload.setdefault("observed_at_monotonic", float(now))
                payload["expected_node_id"] = expected_node_id
                payload["expected_node_attr"] = int(expected_node_attr or 0)
                payload["source"] = str(source or "opencv_bev")
                self._stopline_pass_event = payload
                self.stopline_pass_count += 1
                self._stopline_last_pass_monotonic = float(now)
                pass_distance_m = payload.get("distance_m", None)
                self.stopline_last_pass_distance_m = (
                    float(pass_distance_m) if pass_distance_m is not None else None
                )

    def consume_stopline_pass_event(self):
        with self._lock:
            event = self._stopline_pass_event
            self._stopline_pass_event = None
            return dict(event) if isinstance(event, dict) else None

    def snapshot(self):
        with self._lock:
            now = time.monotonic()
            stopline_last_seen_age_s = (
                max(0.0, float(now) - float(self._stopline_last_seen_monotonic))
                if self._stopline_last_seen_monotonic > 0.0 else None
            )
            stopline_last_pass_age_s = (
                max(0.0, float(now) - float(self._stopline_last_pass_monotonic))
                if self._stopline_last_pass_monotonic > 0.0 else None
            )
            return dict(
                state_ts=time.monotonic(),
                x=self.x, y=self.y, yaw=self.yaw,
                raw_x=self.raw_x, raw_y=self.raw_y, raw_yaw=self.raw_yaw,
                matched_x=self.matched_x, matched_y=self.matched_y, matched_yaw=self.matched_yaw,
                steer_rad=self.steer_rad,
                error_m=self.error_m, heading_rad=self.heading_rad,
                path_psi=self.path_psi, path_kappa=self.path_kappa,
                map_match_error_m=self.map_match_error_m,
                speed_mps=self.speed_mps, wp_idx=self.wp_idx,
                speed_source=self.speed_source,
                speed_feedback_age_s=self.speed_feedback_age_s,
                speed_command_age_s=self.speed_command_age_s,
                target_idx=self.target_idx,
                route_active=self.route_active,
                route_id=self.route_id,
                current_node_id=self.current_node_id,
                current_node_attr=self.current_node_attr,
                upcoming_node_id=self.upcoming_node_id,
                upcoming_node_attr=self.upcoming_node_attr,
                maneuver_type=self.maneuver_type,
                destination_node_id=self.destination_node_id,
                destination_label=self.destination_label,
                route_queue=list(self.route_queue),
                route_progress=self.route_progress,
                route_points=list(self.route_points),
                route_completed=self.route_completed,
                route_replans=self.route_replans,
                route_source=self.route_source,
                destination_point=self.destination_point,
                next_semantic_id=self.next_semantic_id,
                next_semantic_type=self.next_semantic_type,
                next_semantic_label=self.next_semantic_label,
                next_semantic_distance_m=self.next_semantic_distance_m,
                expected_control_type=self.expected_control_type,
                current_zone_ids=list(self.current_zone_ids),
                current_zone_types=list(self.current_zone_types),
                map_metadata=dict(self.map_metadata),
                available_destinations=list(self.available_destinations),
                waypoint_mode_active=self.waypoint_mode_active,
                node_attr=self.node_attr,
                camera_lateral_correction_m=self.camera_lateral_correction_m,
                lane_measurement_reliable=self.lane_measurement_reliable,
                raw_lateral_error_m=self.raw_lateral_error_m,
                relocalization_mode=self.relocalization_mode,
                last_relocalization_source=self.last_relocalization_source,
                last_relocalization_error_m=self.last_relocalization_error_m,
                localization_confidence=self.localization_confidence,
                stopline_visible=self.stopline_visible,
                stopline_stable=self.stopline_stable,
                stopline_distance_m=self.stopline_distance_m,
                stopline_confidence=self.stopline_confidence,
                stopline_source=self.stopline_source,
                stopline_expected_node_id=self.stopline_expected_node_id,
                stopline_expected_node_attr=self.stopline_expected_node_attr,
                stopline_pass_count=self.stopline_pass_count,
                stopline_last_pass_distance_m=self.stopline_last_pass_distance_m,
                stopline_last_seen_age_s=stopline_last_seen_age_s,
                stopline_last_pass_age_s=stopline_last_pass_age_s,
                control_steering_deg=self.control_steering_deg,
                control_speed_cmd=self.control_speed_cmd,
                control_authority=self.control_authority,
                control_safety_override=self.control_safety_override,
                control_reason=self.control_reason,
            )


class threadTracking(ThreadWithStop):
    """Dead-reckoning + waypoint tracking thread.

    Args:
        queuesList:     Dict of multiprocessing.Queue objects.
        tracking_state: TrackingState instance (shared with other threads).
        logging:        Python logger.
        debugging:      Enable verbose prints.
        visualizer:     Optional trackVisualizer instance for direct state push.
    """

    def __init__(self, queuesList, tracking_state: TrackingState,
                 logging=None, debugging=False, visualizer=None):
        super().__init__(pause=1.0 / _LOOP_HZ)
        self.queuesList = queuesList
        self.tracking_state = tracking_state
        self.logging = logging
        self.debugging = debugging
        self.visualizer = visualizer

        # Message subscribers (LastOnly → always use most recent value)
        self._speed_sub = messageHandlerSubscriber(
            queuesList, CurrentSpeed, "lastOnly", subscribe=True
        )
        self._speed_cmd_sub = messageHandlerSubscriber(
            queuesList, SpeedMotor, "lastOnly", subscribe=True
        )
        self._imu_sub = messageHandlerSubscriber(
            queuesList, ImuData, "lastOnly", subscribe=True
        )
        self._steer_feedback_sub = messageHandlerSubscriber(
            queuesList, CurrentSteer, "lastOnly", subscribe=True
        )
        self._steer_sub = messageHandlerSubscriber(
            queuesList, SteerMotor, "lastOnly", subscribe=True
        )
        self._nav_cmd_sub = messageHandlerSubscriber(
            queuesList, NavigationCommand, "lastOnly", subscribe=True
        )
        self._state_sub = messageHandlerSubscriber(
            queuesList, StateChange, "lastOnly", subscribe=True
        )
        self._sign_sub = messageHandlerSubscriber(
            queuesList, SignDetected, "lastOnly", subscribe=True
        )
        self._last_steer_rad = 0.0     # latest steering angle in radians (math convention)
        self._steer_filtered_rad = 0.0 # lag-filtered steer angle used by DR
        self._yaw_ekf_p = _YAW_EKF_P_INIT  # EKF heading covariance (rad²)
        # Location sender → dashboard map display
        self._loc_sender = messageHandlerSender(queuesList, Location)
        self._nav_status_sender = messageHandlerSender(queuesList, NavigationStatus)

        # Load the track graph
        graphml_path = _GRAPHML_PATH
        if not os.path.isabs(graphml_path):
            # Resolve relative to workspace root: src/hardware/tracking/ → 3 levels up
            _here = os.path.dirname(os.path.abspath(__file__))
            _root = os.path.join(_here, "..", "..", "..")
            graphml_path = os.path.normpath(os.path.join(_root, graphml_path))

        self._graph = None
        self._dr = None
        self._path_manager = None
        self._start_yaw_rad = 0.0   # path tangent at start — used as yaw fallback
        try:
            semantics_path = _SEMANTICS_PATH
            if not os.path.isabs(semantics_path):
                _here = os.path.dirname(os.path.abspath(__file__))
                _root = os.path.join(_here, "..", "..", "..")
                semantics_path = os.path.normpath(os.path.join(_root, semantics_path))
            preferred_editor_save = os.path.join(os.path.dirname(semantics_path), "Track Editor Save.json")
            if os.path.exists(preferred_editor_save):
                semantics_path = preferred_editor_save
            elif not os.path.exists(semantics_path):
                alt_name = os.path.join(os.path.dirname(semantics_path), "Track Semantics.json")
                semantics_path = alt_name if os.path.exists(alt_name) else None

            self._graph = TrackGraph(graphml_path, step_m=_STEP_M, semantics_path=semantics_path)
            self._path_manager = PathManager(self._graph)
            x0, y0, yaw0 = self._graph.get_start_pose()
            self._dr = DeadReckoning(x0, y0, yaw0)
            self._start_yaw_rad = yaw0  # remember so IMU can be seeded from this
            # Share DR reference so threadLineFollowing can push lateral corrections
            tracking_state._dr = self._dr
            if debugging:
                print(
                    f"[threadTracking] loaded {len(self._graph.waypoints)} waypoints, "
                    f"start=({x0:.3f}, {y0:.3f}, yaw={math.degrees(yaw0):.1f}°)"
                )
        except Exception as exc:
            print(f"[threadTracking] WARNING – could not load track: {exc}")

        self._wp_idx = 0
        self._last_t = time.monotonic()
        self._last_speed = 0.0
        self._last_speed_source = "none"
        self._last_cmd_speed_raw = 0.0
        self._last_cmd_speed_t = None
        self._current_state_message = "DEFAULT"
        # Seed yaw from track start pose so heading error is ~0 before IMU arrives.
        # Without this, yaw=0° vs path_psi≈91° gives a -91° error → MPC saturates.
        self._last_yaw_rad = self._start_yaw_rad
        self._imu_received = False  # True once at least one real IMU message arrives
        # Yaw offset to align IMU reference frame with the GraphML map frame.
        # Computed on the first IMU message: offset = start_yaw_map - first_imu_yaw_raw
        # This compensates for the IMU having an arbitrary reference direction
        # (e.g. wherever the Nucleo was powered on), so that yaw_corrected = 0°
        # when the car faces the track-start tangent direction.
        self._yaw_offset = 0.0
        self._yaw_offset_calibrated = False
        self._last_raw_speed = None   # raw value from message queue (for log)
        self._last_raw_imu = None     # raw imu dict (for log)
        self._last_imu_t = None       # monotonic time of last IMU message
        self._last_speed_t = None     # monotonic time of last speed message
        self._last_sign_observation = None
        self._last_semantic_relocalization_t = 0.0
        self._last_lane_visual_reloc_t = 0.0
        self._last_visual_stopline_relocalization_t = 0.0
        self._frame_idx = 0
        self._log_every = max(1, int(_LOOP_HZ // 5))  # log 5 times/s
        self._last_steer_feedback_rad = 0.0
        self._last_steer_feedback_t = None

        # ── Tracking debug log ──────────────────────────────────────────────
        self._debug_log_enabled = _DEBUG_LOG
        self._debug_log_path = None
        if _DEBUG_LOG:
            _here = os.path.dirname(os.path.abspath(__file__))
            _root = os.path.normpath(os.path.join(_here, "..", "..", ".."))
            _log_dir = os.path.join(_root, "temp")
            os.makedirs(_log_dir, exist_ok=True)
            self._debug_log_path = os.path.join(_log_dir, "tracking_debug.txt")
            try:
                with open(self._debug_log_path, "w") as _f:
                    _f.write(
                        "# tracking_debug.log — dead reckoning + waypoint state\n"
                        "# Each line: F<idx> | speed_raw=<raw> spd=<m/s> | "
                        "imu_yaw=<deg> | raw=(<x>,<y>,<yaw>) matched=(<x>,<y>,<yaw>) | "
                        "wp=<idx>/<total> tgt=<idx> wp_xy=(<x>,<y>) dist=<m> | "
                        "err=<m> hdg=<deg> mm=<m> latcorr=<m> lane=<0/1> | zone=<attr> | dt=<ms>\n"
                    )
            except Exception:
                pass

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_sign_name(sign_name) -> str:
        return str(sign_name or "").strip().lower().replace("-", "_").replace(" ", "_")

    def _consume_sign_observation(self, now: float) -> None:
        payload = self._sign_sub.receive()
        if not isinstance(payload, dict):
            return
        sign_name = self._normalize_sign_name(payload.get("sign"))
        if not sign_name:
            return
        self._last_sign_observation = {
            "sign": sign_name,
            "timestamp": float(payload.get("timestamp", 0.0) or 0.0),
            "observed_at_monotonic": float(now),
            "distance_m": (
                float(payload.get("distance_cm")) / 100.0
                if payload.get("distance_cm") is not None else None
            ),
            "confidence": float(payload.get("confidence", 0.0) or 0.0),
        }

    def _sign_matches_expected_semantic(self, path_update) -> tuple[str, float] | None:
        observation = self._last_sign_observation
        if not isinstance(observation, dict):
            return None
        obs_age = time.monotonic() - float(observation.get("observed_at_monotonic", 0.0) or 0.0)
        if obs_age > float(_SEMANTIC_MATCH_WINDOW_S):
            return None

        sign_name = str(observation.get("sign") or "")
        expected_control = str(path_update.expected_control_type or "")
        next_semantic_type = str(path_update.next_semantic_type or "")
        obs_distance_m = observation.get("distance_m")
        path_distance_m = path_update.next_semantic_distance_m

        def _match_payload() -> tuple[str, float]:
            if obs_distance_m is not None and path_distance_m is not None:
                return (f"sign:{sign_name}", abs(float(obs_distance_m) - float(path_distance_m)))
            if path_distance_m is not None:
                return (f"sign:{sign_name}", float(path_distance_m))
            if obs_distance_m is not None:
                return (f"sign:{sign_name}", float(obs_distance_m))
            return (f"sign:{sign_name}", 0.0)

        if expected_control == "traffic_light" and sign_name in {"red_light", "yellow_light", "green_light"}:
            return _match_payload()
        if expected_control == "stop" and sign_name in {"stop", "no_entry"}:
            return _match_payload()
        if next_semantic_type == "crosswalk" and sign_name == "crosswalk":
            return _match_payload()
        if next_semantic_type == "parking_spot" and sign_name == "parking":
            return _match_payload()
        return None

    @staticmethod
    def _signed_lateral_error_to_path(
        x: float,
        y: float,
        ref_x: float,
        ref_y: float,
        path_psi: float,
    ) -> float:
        dx = float(x) - float(ref_x)
        dy = float(y) - float(ref_y)
        return float(-dx * math.sin(float(path_psi)) + dy * math.cos(float(path_psi)))

    def _apply_lane_visual_relocalization(self, raw_x, raw_y, raw_yaw, path_update, now: float):
        if self._dr is None or path_update is None:
            return raw_x, raw_y, raw_yaw, 0.0, 0.0
        raw_lateral_error_m = self._signed_lateral_error_to_path(
            raw_x,
            raw_y,
            path_update.matched_x,
            path_update.matched_y,
            path_update.matched_yaw,
        )
        self.tracking_state.raw_lateral_error_m = float(raw_lateral_error_m)
        if not bool(getattr(self.tracking_state, "lane_measurement_reliable", False)):
            return raw_x, raw_y, raw_yaw, 0.0, float(raw_lateral_error_m)
        if not bool(_VISUAL_LANE_RELOCALIZATION_ENABLED):
            self.tracking_state.set_lane_measurement_state(True, 0.0)
            return raw_x, raw_y, raw_yaw, 0.0, float(raw_lateral_error_m)
        # Skip in precision zones (intersections, stoplines): the car intentionally
        # deviates from the route centreline here, so snapping it back is wrong.
        if bool(getattr(path_update, "waypoint_mode_active", False)):
            return raw_x, raw_y, raw_yaw, 0.0, float(raw_lateral_error_m)
        # Skip when the car is nearly stopped — prevents fighting parking manoeuvres.
        if abs(self._last_speed) < float(_VISUAL_LANE_RELOCALIZATION_SPEED_MIN_MPS):
            return raw_x, raw_y, raw_yaw, 0.0, float(raw_lateral_error_m)
        abs_raw_error_m = abs(raw_lateral_error_m)
        if abs_raw_error_m < float(_VISUAL_LANE_RELOCALIZATION_MIN_RAW_ERROR_M):
            self.tracking_state.set_lane_measurement_state(True, 0.0)
            self.tracking_state.raw_lateral_error_m = float(raw_lateral_error_m)
            return raw_x, raw_y, raw_yaw, 0.0, float(raw_lateral_error_m)
        if abs_raw_error_m > float(_VISUAL_LANE_RELOCALIZATION_MAX_RAW_ERROR_M):
            self.tracking_state.raw_lateral_error_m = float(raw_lateral_error_m)
            return raw_x, raw_y, raw_yaw, 0.0, float(raw_lateral_error_m)

        # Rate-limit: apply at most every _VISUAL_LANE_RELOCALIZATION_COOLDOWN_S so
        # we don't snap the virtual position 50×/s (which would keep it glued to the
        # map route and mask any genuine physical departure from the lane).
        if (now - self._last_lane_visual_reloc_t) < float(_VISUAL_LANE_RELOCALIZATION_COOLDOWN_S):
            self.tracking_state.raw_lateral_error_m = float(raw_lateral_error_m)
            return raw_x, raw_y, raw_yaw, 0.0, float(raw_lateral_error_m)

        correction_m = float(raw_lateral_error_m) * float(_VISUAL_LANE_RELOCALIZATION_GAIN)
        max_corr_m = float(_VISUAL_LANE_RELOCALIZATION_MAX_M)
        if max_corr_m > 0.0:
            correction_m = max(-max_corr_m, min(max_corr_m, correction_m))
        if abs(correction_m) < 1e-9:
            self.tracking_state.raw_lateral_error_m = float(raw_lateral_error_m)
            return raw_x, raw_y, raw_yaw, 0.0, float(raw_lateral_error_m)

        self._dr.correct_lateral(correction_m, float(path_update.matched_yaw))
        self._last_lane_visual_reloc_t = now
        new_raw_x, new_raw_y, new_raw_yaw = self._dr.get_state()
        self.tracking_state.set_lane_measurement_state(True, correction_m)
        self.tracking_state.raw_lateral_error_m = float(raw_lateral_error_m)
        return new_raw_x, new_raw_y, new_raw_yaw, float(correction_m), float(raw_lateral_error_m)

    def _apply_semantic_relocalization(self, path_update, now: float):
        if self._dr is None or path_update is None:
            return False, None
        semantic_match = self._sign_matches_expected_semantic(path_update)
        if semantic_match is None:
            return False, None

        if (float(now) - float(self._last_semantic_relocalization_t)) < float(_SEMANTIC_RELOCALIZATION_COOLDOWN_S):
            return False, semantic_match

        expected_distance_m = path_update.next_semantic_distance_m
        observation = self._last_sign_observation if isinstance(self._last_sign_observation, dict) else {}
        observed_distance_m = observation.get("distance_m")
        if expected_distance_m is not None and float(expected_distance_m) > float(_SEMANTIC_RELOCALIZATION_MAX_DISTANCE_M):
            return False, semantic_match
        if float(path_update.map_match_error_m or 0.0) > float(_SEMANTIC_RELOCALIZATION_MAX_MAP_ERROR_M):
            return False, semantic_match
        if (
            observed_distance_m is not None
            and expected_distance_m is not None
            and abs(float(observed_distance_m) - float(expected_distance_m))
            > float(_SEMANTIC_RELOCALIZATION_DISTANCE_TOLERANCE_M)
        ):
            return False, semantic_match

        self._dr.reset(
            float(path_update.matched_x),
            float(path_update.matched_y),
            float(path_update.matched_yaw),
        )
        self._last_semantic_relocalization_t = float(now)
        return True, semantic_match

    @staticmethod
    def _signed_route_delta_pts(route, current_idx: int, candidate_idx: int) -> int:
        n = int(len(getattr(route, "wp_node_ids", []) or []))
        if n <= 0:
            return 0
        current_idx = max(0, min(n - 1, int(current_idx)))
        candidate_idx = max(0, min(n - 1, int(candidate_idx)))
        if not bool(getattr(route, "closed_loop", False)):
            return int(candidate_idx - current_idx)
        forward = (candidate_idx - current_idx) % n
        backward = (current_idx - candidate_idx) % n
        return int(forward) if forward <= backward else -int(backward)

    def _route_stopline_anchor(self, path_update, event: dict | None = None) -> dict | None:
        route = self._path_manager.active_route if self._path_manager is not None else None
        if route is None or route.waypoints.size == 0 or len(route.wp_node_ids) == 0:
            return None

        expected_node_id = str((event or {}).get("expected_node_id") or "") or None
        raw_x, raw_y, _ = self._dr.get_state() if self._dr is not None else (0.0, 0.0, 0.0)
        matched_idx = int(getattr(path_update, "matched_idx", getattr(self._path_manager, "matched_idx", 0)) or 0)
        step_m = max(float(getattr(self._graph, "step_m", _STEP_M) or _STEP_M), 1e-6)
        max_behind_m = max(0.0, float(_VISUAL_STOPLINE_ROUTE_BEHIND_M))
        max_ahead_m = max(max_behind_m, float(_VISUAL_STOPLINE_ROUTE_AHEAD_M))

        candidate_groups: dict[str, list[int]] = {}
        for idx, attr in enumerate(route.wp_node_attrs):
            if int(attr or 0) != int(_STOPLINE_NODE_ATTR):
                continue
            node_id = str(route.wp_node_ids[idx] or f"wp:{idx}")
            candidate_groups.setdefault(node_id, []).append(int(idx))

        if not candidate_groups:
            return None

        best = None
        for node_id, indices in candidate_groups.items():
            rep_idx = min(
                indices,
                key=lambda idx: abs(self._signed_route_delta_pts(route, matched_idx, idx)),
            )
            signed_delta_pts = self._signed_route_delta_pts(route, matched_idx, rep_idx)
            signed_delta_m = float(signed_delta_pts) * step_m
            window_penalty = 0
            if signed_delta_m < -max_behind_m or signed_delta_m > max_ahead_m:
                window_penalty = 1
            wx, wy, wyaw = route.waypoints[int(rep_idx)]
            euclid_m = math.hypot(float(wx) - float(raw_x), float(wy) - float(raw_y))
            priority = 0 if expected_node_id is not None and node_id == expected_node_id else 1
            score = (priority, window_penalty, abs(signed_delta_m), euclid_m)
            candidate = {
                "node_id": node_id,
                "idx": int(rep_idx),
                "x": float(wx),
                "y": float(wy),
                "yaw": float(wyaw),
                "signed_delta_m": float(signed_delta_m),
                "euclid_m": float(euclid_m),
                "score": score,
            }
            if best is None or candidate["score"] < best["score"]:
                best = candidate

        return best

    def _apply_visual_stopline_relocalization(self, path_update, now: float):
        if self._dr is None or path_update is None or self._path_manager is None:
            return False, None

        event = self.tracking_state.consume_stopline_pass_event()
        if not isinstance(event, dict):
            return False, None

        event_age = max(
            0.0,
            float(now) - float(event.get("observed_at_monotonic", float(now)) or float(now)),
        )
        if event_age > float(_VISUAL_STOPLINE_EVENT_MAX_AGE_S):
            return False, None
        if (
            float(now) - float(self._last_visual_stopline_relocalization_t)
        ) < float(_VISUAL_STOPLINE_RELOCALIZATION_COOLDOWN_S):
            return False, None
        if not bool(getattr(path_update, "route_active", False)):
            return False, None

        route = self._path_manager.active_route
        if route is None or route.waypoints.size == 0:
            return False, None

        current_attr = int(getattr(path_update, "current_node_attr", 0) or 0)
        upcoming_attr = int(getattr(path_update, "upcoming_node_attr", current_attr) or current_attr)
        next_semantic_type = str(getattr(path_update, "next_semantic_type", "") or "")
        expected_node_id = str(event.get("expected_node_id") or "") or None
        stopline_context = (
            current_attr == int(_STOPLINE_NODE_ATTR)
            or upcoming_attr == int(_STOPLINE_NODE_ATTR)
            or next_semantic_type == "stopline"
            or (
                expected_node_id is not None
                and expected_node_id in {str(node_id) for node_id in getattr(route, "node_ids", [])}
            )
        )
        if not stopline_context:
            return False, None
        if float(path_update.map_match_error_m or 0.0) > float(_VISUAL_STOPLINE_MAX_MAP_ERROR_M):
            return False, None

        anchor = self._route_stopline_anchor(path_update, event)
        if anchor is None:
            return False, None

        old_raw_x, old_raw_y, current_yaw = self._dr.get_state()
        self._dr.reset(
            float(anchor["x"]),
            float(anchor["y"]),
            current_yaw,
        )
        self._path_manager.matched_idx = int(anchor["idx"])
        self._path_manager.target_idx = int(anchor["idx"])
        self._last_visual_stopline_relocalization_t = float(now)

        correction_m = math.hypot(
            float(anchor["x"]) - float(old_raw_x),
            float(anchor["y"]) - float(old_raw_y),
        )
        source = f"stopline_visual:{anchor['node_id']}"
        return True, (source, float(correction_m))

    @staticmethod
    def _parse_speed_mps(raw_value) -> float | None:
        try:
            return float(raw_value) * 0.001
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_steer_rad(raw_value) -> float | None:
        try:
            # Raw steering follows project convention (positive = right).
            # Convert here to mathematical bicycle-model convention.
            return math.radians(float(raw_value) / 10.0) * float(_STEER_SIGN_DR)
        except (TypeError, ValueError):
            return None

    def _resolve_speed_mps(self, now: float) -> float:
        current_state_message = str(
            getattr(self, "_current_state_message", "DEFAULT") or "DEFAULT"
        ).upper()
        speed_raw = self._speed_sub.receive()
        if speed_raw is not None:
            self._last_raw_speed = speed_raw
            self._last_speed_t = now
            parsed_speed = self._parse_speed_mps(speed_raw)
            if parsed_speed is not None:
                self._last_speed = float(parsed_speed)
                self._last_speed_source = "encoder"

        speed_cmd_raw = self._speed_cmd_sub.receive()
        if speed_cmd_raw is not None:
            try:
                self._last_cmd_speed_raw = float(speed_cmd_raw)
                self._last_cmd_speed_t = now
            except (TypeError, ValueError):
                pass

        # In MANUAL mode the Nucleo feedback may stay at 0 (no odometry), so DR
        # would never advance. Only use the commanded speed when encoder
        # feedback is stale or effectively zero; if encoder feedback is fresh
        # and moving, trust it even in MANUAL.
        _encoder_fresh = (
            self._last_speed_t is not None and
            (now - self._last_speed_t) <= float(_SPEED_FEEDBACK_TIMEOUT_S)
        )
        _encoder_moving = abs(float(self._last_speed)) > 1e-4
        _cmd_fresh = (
            self._last_cmd_speed_t is not None and
            (now - self._last_cmd_speed_t) <= float(_COMMAND_SPEED_FALLBACK_TIMEOUT_S)
        )
        if (
            current_state_message == "MANUAL" and
            _cmd_fresh and
            (not _encoder_fresh or not _encoder_moving)
        ):
            cmd_speed = self._parse_speed_mps(self._last_cmd_speed_raw)
            if cmd_speed is not None:
                self._last_speed = float(cmd_speed)
                self._last_speed_source = "manual_command_hold"
                return float(self._last_speed)

        if self._last_speed_t is not None:
            if (now - self._last_speed_t) <= float(_SPEED_FEEDBACK_TIMEOUT_S):
                return float(self._last_speed)

        if _COMMAND_SPEED_FALLBACK_ENABLED and self._last_cmd_speed_t is not None:
            if (now - self._last_cmd_speed_t) <= float(_COMMAND_SPEED_FALLBACK_TIMEOUT_S):
                cmd_speed = self._parse_speed_mps(self._last_cmd_speed_raw)
                if cmd_speed is not None:
                    scale = float(getattr(cfg, "TRACKING_COMMAND_SPEED_FALLBACK_SCALE", 1.0) or 1.0)
                    self._last_speed = float(cmd_speed) * scale
                    self._last_speed_source = "command"
                    return float(self._last_speed)

        self._last_speed = 0.0
        self._last_speed_source = "none"
        return 0.0

    def _consume_state_change(self) -> None:
        message = self._state_sub.receive()
        if message is None:
            return
        state_name = str(message or "").strip().upper() or "DEFAULT"
        if state_name == self._current_state_message:
            return
        self._current_state_message = state_name
        # Avoid carrying a stale command-held speed between modes.
        self._last_speed = 0.0
        self._last_speed_source = "none"
        self._last_cmd_speed_raw = 0.0
        self._last_cmd_speed_t = None

    def _resolve_steer_rad(self, now: float) -> float:
        steer_feedback_raw = self._steer_feedback_sub.receive()
        if steer_feedback_raw is not None:
            parsed_feedback = self._parse_steer_rad(steer_feedback_raw)
            if parsed_feedback is not None:
                self._last_steer_feedback_rad = float(parsed_feedback)
                self._last_steer_feedback_t = now

        steer_raw = self._steer_sub.receive()
        if steer_raw is not None:
            parsed_cmd = self._parse_steer_rad(steer_raw)
            if parsed_cmd is not None:
                self._last_steer_rad = float(parsed_cmd)

        if self._last_steer_feedback_t is not None:
            if (now - self._last_steer_feedback_t) <= float(_STEER_FEEDBACK_TIMEOUT_S):
                return float(self._last_steer_feedback_rad)
        return float(self._last_steer_rad)

    # ------------------------------------------------------------------
    def thread_work(self):
        now = time.monotonic()
        dt = now - self._last_t
        self._last_t = now
        self._consume_state_change()

        # ---- Read latest speed/steering feedback and fall back to commands when needed.
        self._resolve_speed_mps(now)
        self._last_steer_rad = self._resolve_steer_rad(now)

        # ---- Read latest IMU data and apply absolute heading correction.
        # The BNO055 magnetometer is affected by electromagnetic interference
        # from the steering servo (yaw jumps when steering angle changes sharply).
        # We still use it as the primary heading reference (same as the reference
        # repo: `self.yaw = imu.yaw`) but rate-limit each correction to at most
        # _MAX_PHYSICAL_YAW_RATE_RADS × dt so that EMI step spikes are clipped.
        # Between IMU messages, the bicycle model provides per-frame updates.
        imu_raw = self._imu_sub.receive()
        if imu_raw is not None:
            try:
                imu_dict = ast.literal_eval(str(imu_raw))
                self._last_raw_imu = imu_dict
                _prev_imu_t = self._last_imu_t          # save before overwrite
                self._last_imu_t = now
                yaw_deg = float(imu_dict.get("yaw", math.degrees(self._last_yaw_rad)))
                yaw_raw_rad = -math.radians(yaw_deg)

                if not self._yaw_offset_calibrated:
                    # First IMU message: compute frame-alignment offset so that
                    # yaw_imu = yaw_raw_rad + _yaw_offset maps IMU readings to
                    # the GraphML map frame.  _last_yaw_rad is already seeded
                    # from the track-start tangent in __init__; do not overwrite.
                    self._yaw_offset = self._start_yaw_rad - yaw_raw_rad
                    self._yaw_offset_calibrated = True
                    _msg = (
                        f"[threadTracking] IMU ready: "
                        f"raw={yaw_deg:.1f}°  "
                        f"start_map={math.degrees(self._start_yaw_rad):.1f}°  "
                        f"offset={math.degrees(self._yaw_offset):.1f}°"
                    )
                    if self.logging:
                        self.logging.info(_msg)
                    else:
                        print(_msg)
                else:
                    # Subsequent IMU messages: EKF correction of heading.
                    #
                    # Replaces the hard steer-inhibit cutoff with a soft Kalman
                    # gain: K = P / (P + R), where R grows with steering angle²
                    # because the BNO055 magnetometer is biased by servo EMI at
                    # large steer.  Result:
                    #   steer ≈ 0°  → R ≈ R_STRAIGHT (small) → K ≈ 1 → trust IMU
                    #   steer ≈ 25° → R ≈ 10 rad²   (large) → K ≈ 0 → trust kinematics
                    # The transition is smooth, not a hard cutoff.
                    yaw_imu = yaw_raw_rad + self._yaw_offset
                    dt_imu = (now - _prev_imu_t) if _prev_imu_t is not None else 0.05
                    innov = yaw_imu - self._last_yaw_rad
                    while innov > math.pi:
                        innov -= 2.0 * math.pi
                    while innov < -math.pi:
                        innov += 2.0 * math.pi
                    # Rate-limit innovation to reject servo-EMI step spikes
                    max_innov = _MAX_PHYSICAL_YAW_RATE_RADS * max(dt_imu, 0.02)
                    innov = max(-max_innov, min(max_innov, innov))
                    # Kalman gain: R grows with steer² → IMU less reliable when turning
                    R_imu = _YAW_EKF_R_STRAIGHT + _YAW_EKF_R_STEER_K * (self._steer_filtered_rad ** 2)
                    K = self._yaw_ekf_p / (self._yaw_ekf_p + R_imu)
                    self._last_yaw_rad += K * innov
                    self._yaw_ekf_p = (1.0 - K) * self._yaw_ekf_p

                self._imu_received = True
            except Exception:
                pass

        _eff_steer_raw = self._last_steer_rad * _STEER_GAIN_DR
        # First-order lag filter: models the servo actuator delay so the DR does
        # not assume the wheels have already reached the commanded angle.
        self._steer_filtered_rad = (
            self._steer_filtered_rad
            + _STEER_LAG_ALPHA * (_eff_steer_raw - self._steer_filtered_rad)
        )
        _eff_steer_rad = self._steer_filtered_rad
        if self._dr is None or self._graph is None or self._path_manager is None:
            return

        # ---- Dead reckoning update (RK4 + yaw bicycle model)
        # Let the DR own the heading integration so yaw and x/y stay synchronized.
        # Passing a yaw that was already advanced by another integration step was
        # exaggerating both curvature and displacement in the preview.
        dr_dt = min(dt, _MAX_INTEGRATION_DT)
        self._dr.update(self._last_speed * _DR_SPEED_SCALE, self._last_yaw_rad, dr_dt,
                        steer_rad=_eff_steer_rad, wheelbase_m=_WHEELBASE_M)
        raw_x, raw_y, raw_yaw = self._dr.get_state()
        self._last_yaw_rad = float(raw_yaw)
        # EKF process noise: covariance grows over time as kinematic drift accumulates
        self._yaw_ekf_p = min(self._yaw_ekf_p + _YAW_EKF_Q * dr_dt, 1.0)

        # ---- Camera-based yaw correction (soft blend toward camera estimate).
        # Apply it after the DR step so the correction updates heading only, without
        # retroactively inflating the position change of the last interval.
        _cam_yaw, _cam_conf = self.tracking_state.consume_camera_yaw_hint()
        _yaw_correction_rad = 0.0
        if _cam_yaw is not None and _cam_conf > 0.3:
            _alpha = 0.08 * _cam_conf
            _delta = _cam_yaw - self._last_yaw_rad
            while _delta > math.pi:
                _delta -= 2.0 * math.pi
            while _delta < -math.pi:
                _delta += 2.0 * math.pi
            _yaw_correction_rad = _alpha * _delta
            self._last_yaw_rad += _yaw_correction_rad
            self._dr.correct_yaw(_yaw_correction_rad)
            raw_x, raw_y, raw_yaw = self._dr.get_state()
        self.tracking_state.last_yaw_correction_deg = math.degrees(_yaw_correction_rad)

        self._consume_sign_observation(now)

        nav_cmd = self._nav_cmd_sub.receive()
        if isinstance(nav_cmd, dict):
            self._path_manager.handle_command(
                nav_cmd,
                current_pose={"x": raw_x, "y": raw_y},
            )

        path_update = self._path_manager.update(
            raw_x,
            raw_y,
            raw_yaw,
            speed_mps=self._last_speed,
            min_lookahead_m=_ADVANCE_DIST,
            lookahead_time_s=_LOOKAHEAD_TIME_S,
            max_lookahead_m=_MAX_LOOKAHEAD_M,
            precision_lookahead_m=_PRECISION_LOOKAHEAD_M,
            lookahead_pts=_LOOKAHEAD_PTS,
            search_window=_MAP_MATCH_SEARCH_WP,
            distance_weight=_MAP_MATCH_DISTANCE_W,
            heading_weight=_MAP_MATCH_HEADING_W,
        )
        raw_x, raw_y, raw_yaw, lane_relocalization_m, raw_lateral_error_m = self._apply_lane_visual_relocalization(
            raw_x,
            raw_y,
            raw_yaw,
            path_update,
            now,
        )
        if abs(lane_relocalization_m) > 1e-9:
            path_update = self._path_manager.update(
                raw_x,
                raw_y,
                raw_yaw,
                speed_mps=self._last_speed,
                min_lookahead_m=_ADVANCE_DIST,
                lookahead_time_s=_LOOKAHEAD_TIME_S,
                max_lookahead_m=_MAX_LOOKAHEAD_M,
                precision_lookahead_m=_PRECISION_LOOKAHEAD_M,
                lookahead_pts=_LOOKAHEAD_PTS,
                search_window=_MAP_MATCH_SEARCH_WP,
                distance_weight=_MAP_MATCH_DISTANCE_W,
                heading_weight=_MAP_MATCH_HEADING_W,
            )

        semantic_relocalized, semantic_match = self._apply_semantic_relocalization(path_update, now)
        if semantic_relocalized:
            raw_x, raw_y, raw_yaw = self._dr.get_state()
            path_update = self._path_manager.update(
                raw_x,
                raw_y,
                raw_yaw,
                speed_mps=self._last_speed,
                min_lookahead_m=_ADVANCE_DIST,
                lookahead_time_s=_LOOKAHEAD_TIME_S,
                max_lookahead_m=_MAX_LOOKAHEAD_M,
                precision_lookahead_m=_PRECISION_LOOKAHEAD_M,
                lookahead_pts=_LOOKAHEAD_PTS,
                search_window=_MAP_MATCH_SEARCH_WP,
                distance_weight=_MAP_MATCH_DISTANCE_W,
                heading_weight=_MAP_MATCH_HEADING_W,
            )
            raw_lateral_error_m = self._signed_lateral_error_to_path(
                raw_x,
                raw_y,
                path_update.matched_x,
                path_update.matched_y,
                path_update.matched_yaw,
            )
            self.tracking_state.raw_lateral_error_m = float(raw_lateral_error_m)

        visual_stopline_relocalized, visual_stopline_match = self._apply_visual_stopline_relocalization(
            path_update,
            now,
        )
        if visual_stopline_relocalized:
            raw_x, raw_y, raw_yaw = self._dr.get_state()
            path_update = self._path_manager.update(
                raw_x,
                raw_y,
                raw_yaw,
                speed_mps=self._last_speed,
                min_lookahead_m=_ADVANCE_DIST,
                lookahead_time_s=_LOOKAHEAD_TIME_S,
                max_lookahead_m=_MAX_LOOKAHEAD_M,
                precision_lookahead_m=_PRECISION_LOOKAHEAD_M,
                lookahead_pts=_LOOKAHEAD_PTS,
                search_window=_MAP_MATCH_SEARCH_WP,
                distance_weight=_MAP_MATCH_DISTANCE_W,
                heading_weight=_MAP_MATCH_HEADING_W,
            )
            raw_lateral_error_m = self._signed_lateral_error_to_path(
                raw_x,
                raw_y,
                path_update.matched_x,
                path_update.matched_y,
                path_update.matched_yaw,
            )
            self.tracking_state.raw_lateral_error_m = float(raw_lateral_error_m)

        self._wp_idx = int(path_update.matched_idx)
        matched_idx = int(path_update.matched_idx)
        target_idx = int(path_update.target_idx)
        matched_x = float(path_update.matched_x)
        matched_y = float(path_update.matched_y)
        matched_yaw = float(path_update.matched_yaw)
        error_m = float(path_update.error_m)
        heading_rad = float(path_update.heading_rad)
        path_psi = float(path_update.path_psi)
        path_kappa = float(path_update.path_kappa)
        map_match_error_m = float(path_update.map_match_error_m)
        node_attr = int(path_update.upcoming_node_attr or path_update.current_node_attr or 0)

        relocalization_mode = "map_match"
        relocalization_source = "map_match"
        relocalization_error_m = map_match_error_m
        if visual_stopline_relocalized and visual_stopline_match is not None:
            relocalization_mode = "visual_stopline"
            relocalization_source, stopline_error_m = visual_stopline_match
            relocalization_error_m = float(stopline_error_m)
        elif semantic_relocalized and semantic_match is not None:
            relocalization_mode = "semantic"
            relocalization_source, semantic_error_m = semantic_match
            relocalization_error_m = float(semantic_error_m)
        elif abs(_yaw_correction_rad) > math.radians(0.25):
            relocalization_mode = "lane_yaw_reset"
            relocalization_source = "camera_yaw_hint"
            relocalization_error_m = abs(math.degrees(_yaw_correction_rad)) / 180.0
        elif abs(lane_relocalization_m) > 1e-9:
            relocalization_mode = "lane_relocalization"
            relocalization_source = "lane_center"
            relocalization_error_m = abs(float(lane_relocalization_m))

        # ---- Write shared state (consumed by threadLineFollowing & visualizer)
        speed_feedback_age_s = (now - self._last_speed_t) if self._last_speed_t is not None else None
        speed_command_age_s = (now - self._last_cmd_speed_t) if self._last_cmd_speed_t is not None else None
        self.tracking_state.update(
            x=raw_x, y=raw_y, yaw=raw_yaw,
            steer_rad=self._last_steer_rad,
            error_m=error_m, heading_rad=heading_rad,
            path_psi=path_psi, path_kappa=path_kappa,
            path_heading_change_rad=float(path_update.path_heading_change_rad),
            speed_mps=self._last_speed,
            speed_source=self._last_speed_source,
            speed_feedback_age_s=speed_feedback_age_s,
            speed_command_age_s=speed_command_age_s,
            wp_idx=matched_idx,
            waypoint_mode=path_update.waypoint_mode_active,
            node_attr=node_attr,
            imu_received=self._imu_received,
            target_idx=target_idx,
            raw_x=raw_x, raw_y=raw_y, raw_yaw=raw_yaw,
            matched_x=matched_x, matched_y=matched_y, matched_yaw=matched_yaw,
            map_match_error_m=map_match_error_m,
            route_active=path_update.route_active,
            route_id=path_update.route_id,
            current_node_id=path_update.current_node_id,
            current_node_attr=path_update.current_node_attr,
            upcoming_node_id=path_update.upcoming_node_id,
            upcoming_node_attr=path_update.upcoming_node_attr,
            maneuver_type=path_update.maneuver_type,
            destination_node_id=path_update.destination_node_id,
            destination_label=path_update.destination_label,
            route_queue=path_update.route_queue,
            route_progress=path_update.route_progress,
            route_points=path_update.route_points,
            route_completed=path_update.route_completed,
            route_replans=path_update.replans,
            route_source=path_update.route_source,
            destination_point=path_update.destination_point,
            next_semantic_id=path_update.next_semantic_id,
            next_semantic_type=path_update.next_semantic_type,
            next_semantic_label=path_update.next_semantic_label,
            next_semantic_distance_m=path_update.next_semantic_distance_m,
            expected_control_type=path_update.expected_control_type,
            current_zone_ids=path_update.current_zone_ids,
            current_zone_types=path_update.current_zone_types,
            map_metadata=path_update.map_metadata,
            available_destinations=path_update.available_destinations,
            relocalization_mode=relocalization_mode,
            last_relocalization_source=relocalization_source,
            last_relocalization_error_m=relocalization_error_m,
            raw_lateral_error_m=raw_lateral_error_m,
        )

        # ---- Acados full-MPC reference trajectory ----
        try:
            _mpc_N = int(getattr(cfg, "ACADOS_MPC_N", 30))
            _mpc_T = float(getattr(cfg, "ACADOS_MPC_T", 0.05))
            _mpc_v = float(getattr(cfg, "ACADOS_MPC_V_REF", 0.35))
            _sr, _ir = self._path_manager.get_mpc_references(
                matched_idx, _mpc_N, _mpc_T, _mpc_v,
            )
            self.tracking_state.mpc_state_refs = _sr
            self.tracking_state.mpc_input_refs = _ir
        except Exception:
            self.tracking_state.mpc_state_refs = None
            self.tracking_state.mpc_input_refs = None

        try:
            nav_status = self._path_manager.build_navigation_status(path_update)
            nav_status.update(
                {
                    "relocalization_mode": relocalization_mode,
                    "last_relocalization_source": relocalization_source,
                    "last_relocalization_error_m": round(float(relocalization_error_m), 5),
                }
            )
            self._nav_status_sender.send(
                nav_status
            )
        except Exception:
            pass

        # ---- Push state to visualizer if attached
        if self.visualizer is not None:
            try:
                self.visualizer.update_state(self.tracking_state.snapshot())
            except Exception:
                pass

        # ---- Publish Location to dashboard (convert m → normalised dashboard coords)
        # Dashboard MapComponent maps (0..20.67) → (0..100%) width,
        # (0..13.76) → (0..100%) height, so we send raw metres and let the
        # dashboard scale.  Y is inverted because SVG y=0 is at the top.
        try:
            self._loc_sender.send({"x": round(matched_x, 4), "y": round(matched_y, 4)})
        except Exception:
            pass

        # ---- Tracking debug log ───────────────────────────────────────────
        self._frame_idx += 1
        if self._debug_log_enabled and self._debug_log_path and \
                (self._frame_idx % self._log_every == 0):
            self._write_tracking_log(
                raw_x, raw_y, raw_yaw,
                matched_x, matched_y, matched_yaw,
                error_m, heading_rad,
                matched_idx, target_idx,
                int(self._path_manager.active_route.waypoints.shape[0]) if self._path_manager.active_route is not None else 0,
                path_update.waypoint_mode_active, node_attr, dt,
                route_id=path_update.route_id,
                current_node_id=path_update.current_node_id,
                upcoming_node_id=path_update.upcoming_node_id,
                maneuver_type=path_update.maneuver_type,
            )

    def _write_tracking_log(self, raw_x, raw_y, raw_yaw,
                             matched_x, matched_y, matched_yaw,
                             error_m, heading_rad,
                             wp_idx, target_idx, n_wp, in_precision_zone, node_attr, dt,
                             route_id=None, current_node_id=None,
                             upcoming_node_id=None, maneuver_type="none"):
        """Write one line to temp/tracking_debug.txt."""
        try:
            route = self._path_manager.active_route if self._path_manager is not None else None
            now = time.monotonic()
            if route is not None and n_wp > 0:
                wp = route.waypoints[target_idx % n_wp] if route.closed_loop else route.waypoints[min(max(0, target_idx), n_wp - 1)]
                dist_to_wp = math.hypot(matched_x - wp[0], matched_y - wp[1])
            else:
                wp = (float("nan"), float("nan"))
                dist_to_wp = float("nan")

            # Speed field
            raw_spd = self._last_raw_speed
            spd_age = f"{now - self._last_speed_t:.1f}s" if self._last_speed_t else "never"
            cmd_age = f"{now - self._last_cmd_speed_t:.1f}s" if self._last_cmd_speed_t else "never"
            spd_str = (
                f"speed_raw={raw_spd} spd={self._last_speed*100:.1f}cm/s "
                f"src={self._last_speed_source} (enc_age={spd_age}, cmd_age={cmd_age})"
            )

            # IMU field
            imu_age = f"{now - self._last_imu_t:.1f}s" if self._last_imu_t else "never"
            imu = self._last_raw_imu
            imu_ok = "✓" if self._imu_received else "✗NO_IMU"
            if imu:
                imu_yaw_raw = float(imu.get("yaw", math.degrees(self._last_yaw_rad)))
                imu_yaw_corrected = imu_yaw_raw + math.degrees(self._yaw_offset)
                pitch = imu.get("pitch", "?")
                roll  = imu.get("roll",  "?")
                cal_str = (
                    f"offset={math.degrees(self._yaw_offset):.1f}°"
                    if self._yaw_offset_calibrated else "uncal"
                )
                imu_str = (
                    f"imu{imu_ok} raw={imu_yaw_raw:.1f}° corr={imu_yaw_corrected:.1f}° "
                    f"({cal_str}) pitch={pitch} roll={roll} (age={imu_age})"
                    f" steer={math.degrees(self._last_steer_rad):.1f}°"
                )
            else:
                imu_str = (
                    f"imu{imu_ok} yaw={math.degrees(self._last_yaw_rad):.1f}°"
                    f" (seeded_from_track, age={imu_age})"
                    f" steer={math.degrees(self._last_steer_rad):.1f}°"
                )

            zone_str = _ATTR_NAMES.get(node_attr, str(node_attr))
            if in_precision_zone:
                zone_str += "★"

            # Drift warning: speed=0 but DR position actually moved since last log
            # (avoids false positive when car stops with large accumulated error)
            drift_warn = ""
            prev_xy = getattr(self, "_log_prev_xy", None)
            if prev_xy is not None and abs(self._last_speed) < 1e-4:
                moved = math.hypot(matched_x - prev_xy[0], matched_y - prev_xy[1])
                if moved > 0.005:  # more than 5mm movement at zero speed = real drift
                    drift_warn = f" ⚠ DRIFT_WITH_ZERO_SPEED (moved {moved*100:.1f}cm)"
            self._log_prev_xy = (matched_x, matched_y)

            path_kappa = self._path_manager._get_curvature(route, target_idx) if route is not None else 0.0
            ff_deg = math.degrees(math.atan(path_kappa * 0.258)) if abs(path_kappa) > 0.01 else 0.0
            _corr_deg = getattr(self.tracking_state, 'last_yaw_correction_deg', 0.0)
            _cam_hint = getattr(self.tracking_state, '_cam_yaw_hint_rad', None)
            _cam_conf = getattr(self.tracking_state, '_cam_yaw_hint_conf', 0.0)
            _lane_rel = int(bool(getattr(self.tracking_state, 'lane_measurement_reliable', False)))
            _lat_corr = float(getattr(self.tracking_state, 'camera_lateral_correction_m', 0.0))
            _raw_lat = float(getattr(self.tracking_state, 'raw_lateral_error_m', 0.0))
            _mm_err = float(getattr(self.tracking_state, 'map_match_error_m', 0.0))
            _cam_str = (
                f"cam_yaw_hint={math.degrees(_cam_hint):.1f}° conf={_cam_conf:.2f} corr={_corr_deg:+.3f}°"
                if _cam_hint is not None else "cam_yaw_hint=none"
            )
            line = (
                f"F{self._frame_idx:06d} | "
                f"{spd_str} | "
                f"{imu_str} | "
                f"raw=({raw_x:.4f},{raw_y:.4f},{math.degrees(raw_yaw):.1f}°) "
                f"matched=({matched_x:.4f},{matched_y:.4f},{math.degrees(matched_yaw):.1f}°) | "
                f"wp={wp_idx}/{n_wp} tgt={target_idx} wp_xy=({wp[0]:.3f},{wp[1]:.3f}) dist={dist_to_wp:.3f}m | "
                f"err={error_m:+.4f}m hdg={math.degrees(heading_rad):+.2f}° "
                f"mm={_mm_err:.4f}m latcorr={_lat_corr:+.4f}m raw_lat={_raw_lat:+.4f}m lane={_lane_rel} "
                f"kappa={path_kappa:+.3f}/m ff={ff_deg:+.1f}° | "
                f"zone={zone_str} route={route_id or 'none'} curr={current_node_id} "
                f"next={upcoming_node_id} man={maneuver_type} | {_cam_str} | dt={dt*1000:.1f}ms"
                f"{drift_warn}\n"
            )
            with open(self._debug_log_path, "a") as f:
                f.write(line)
        except Exception:
            pass
