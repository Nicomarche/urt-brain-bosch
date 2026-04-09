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
    ImuData,
    Location,
    NavigationCommand,
    NavigationStatus,
    SignDetected,
    SteerMotor,
)

from src.hardware.tracking.deadReckoning import DeadReckoning
from src.hardware.tracking.trackGraph import TrackGraph
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
    _STEER_GAIN_DR     = getattr(cfg, "TRACKING_STEER_GAIN_DR", 1.0)
    _CAMERA_LATERAL_CORRECTION_GAIN = getattr(
        cfg, "TRACKING_CAMERA_LATERAL_CORRECTION_GAIN", 0.35
    )
    _CAMERA_LATERAL_CORRECTION_MAX_M = getattr(
        cfg, "TRACKING_CAMERA_LATERAL_CORRECTION_MAX_M", 0.08
    )
    _VISUAL_LANE_RELOCALIZATION_GAIN = getattr(
        cfg, "TRACKING_VISUAL_LANE_RELOCALIZATION_GAIN", 0.60
    )
    _VISUAL_LANE_RELOCALIZATION_MAX_M = getattr(
        cfg, "TRACKING_VISUAL_LANE_RELOCALIZATION_MAX_M", 0.10
    )
    _VISUAL_LANE_RELOCALIZATION_MIN_RAW_ERROR_M = getattr(
        cfg, "TRACKING_VISUAL_LANE_RELOCALIZATION_MIN_RAW_ERROR_M", 0.01
    )
    _VISUAL_LANE_RELOCALIZATION_MAX_RAW_ERROR_M = getattr(
        cfg, "TRACKING_VISUAL_LANE_RELOCALIZATION_MAX_RAW_ERROR_M", 0.25
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
    _CAMERA_LATERAL_CORRECTION_GAIN = 0.35
    _CAMERA_LATERAL_CORRECTION_MAX_M = 0.08
    _VISUAL_LANE_RELOCALIZATION_GAIN = 0.60
    _VISUAL_LANE_RELOCALIZATION_MAX_M = 0.10
    _VISUAL_LANE_RELOCALIZATION_MIN_RAW_ERROR_M = 0.01
    _VISUAL_LANE_RELOCALIZATION_MAX_RAW_ERROR_M = 0.25
    _SEMANTIC_MATCH_WINDOW_S = 1.0
    _SEMANTIC_RELOCALIZATION_MAX_DISTANCE_M = 0.45
    _SEMANTIC_RELOCALIZATION_MAX_MAP_ERROR_M = 0.30
    _SEMANTIC_RELOCALIZATION_DISTANCE_TOLERANCE_M = 0.25
    _SEMANTIC_RELOCALIZATION_COOLDOWN_S = 0.75

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
               8: "dotted", 9: "dotted_xwalk"}

# How many dense waypoints correspond to the intersection lookahead distance
_LOOKAHEAD_PTS = max(2, int(_INTERSECTION_LOOKAHEAD / _STEP_M))

# BFMC track physical dimensions in metres (for Location scaling to dashboard)
_TRACK_W_M = 20.67
_TRACK_H_M = 13.76


class TrackingState:
    """Thread-safe shared state between threadTracking and threadLineFollowing.

    Attributes (readable by any thread in the same process):
        x, y, yaw        Current matched pose used by the map/visualizer.
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
        self.map_match_error_m = 0.0
        self.speed_mps = 0.0
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

    def update(self, x, y, yaw, error_m, heading_rad, path_psi, path_kappa,
               speed_mps, wp_idx, waypoint_mode, node_attr, imu_received=False,
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
            self.map_match_error_m = float(map_match_error_m)
            self.speed_mps = speed_mps
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
            self.initialized = True
            if imu_received:
                self.imu_received = True

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
        with self._lock:
            dr = self._dr
            psi = self.path_psi
            correction_m = float(lateral_error_m)
            self.camera_lateral_correction_m = correction_m
            self.lane_measurement_reliable = True
        if dr is not None and abs(correction_m) > 1e-9:
            dr.correct_lateral(correction_m, psi)

    def set_lane_measurement_state(self, reliable: bool, applied_correction_m: float = 0.0) -> None:
        with self._lock:
            self.lane_measurement_reliable = bool(reliable)
            self.camera_lateral_correction_m = float(applied_correction_m if reliable else 0.0)
            if not reliable:
                self.raw_lateral_error_m = 0.0

    def snapshot(self):
        with self._lock:
            return dict(
                x=self.x, y=self.y, yaw=self.yaw,
                raw_x=self.raw_x, raw_y=self.raw_y, raw_yaw=self.raw_yaw,
                matched_x=self.matched_x, matched_y=self.matched_y, matched_yaw=self.matched_yaw,
                steer_rad=self.steer_rad,
                error_m=self.error_m, heading_rad=self.heading_rad,
                path_psi=self.path_psi, path_kappa=self.path_kappa,
                map_match_error_m=self.map_match_error_m,
                speed_mps=self.speed_mps, wp_idx=self.wp_idx,
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
        self._imu_sub = messageHandlerSubscriber(
            queuesList, ImuData, "lastOnly", subscribe=True
        )
        self._steer_sub = messageHandlerSubscriber(
            queuesList, SteerMotor, "lastOnly", subscribe=True
        )
        self._nav_cmd_sub = messageHandlerSubscriber(
            queuesList, NavigationCommand, "lastOnly", subscribe=True
        )
        self._sign_sub = messageHandlerSubscriber(
            queuesList, SignDetected, "lastOnly", subscribe=True
        )
        self._last_steer_rad = 0.0   # latest steering angle in radians (math convention)
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
            if not os.path.exists(semantics_path):
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
        self._frame_idx = 0
        self._log_every = max(1, int(_LOOP_HZ // 5))  # log 5 times/s

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

    def _apply_lane_visual_relocalization(self, raw_x, raw_y, raw_yaw, path_update):
        if self._dr is None or path_update is None:
            return raw_x, raw_y, raw_yaw, 0.0, 0.0
        if not bool(getattr(self.tracking_state, "lane_measurement_reliable", False)):
            return raw_x, raw_y, raw_yaw, 0.0, 0.0

        raw_lateral_error_m = self._signed_lateral_error_to_path(
            raw_x,
            raw_y,
            path_update.matched_x,
            path_update.matched_y,
            path_update.matched_yaw,
        )
        abs_raw_error_m = abs(raw_lateral_error_m)
        if abs_raw_error_m < float(_VISUAL_LANE_RELOCALIZATION_MIN_RAW_ERROR_M):
            self.tracking_state.set_lane_measurement_state(True, 0.0)
            self.tracking_state.raw_lateral_error_m = float(raw_lateral_error_m)
            return raw_x, raw_y, raw_yaw, 0.0, float(raw_lateral_error_m)
        if abs_raw_error_m > float(_VISUAL_LANE_RELOCALIZATION_MAX_RAW_ERROR_M):
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

    # ------------------------------------------------------------------
    def thread_work(self):
        now = time.monotonic()
        dt = now - self._last_t
        self._last_t = now

        # ---- Read latest speed (float, raw units = 10 cm/s)
        speed_raw = self._speed_sub.receive()
        if speed_raw is not None:
            self._last_raw_speed = speed_raw
            self._last_speed_t = now
            try:
                self._last_speed = float(speed_raw) * 0.001  # → m/s
            except (TypeError, ValueError):
                pass

        # ---- Read latest steering angle (degrees, same sign as servo command)
        steer_raw = self._steer_sub.receive()
        if steer_raw is not None:
            try:
                # CurrentSteer is in tenths of degrees (angle × 10) — divide before converting.
                # Protocol: send_motor_commands sends int(angle_deg * 10), firmware echoes same units.
                # CurrentSteer > 0 → right turn (CW in world) → yaw decreases in math convention.
                self._last_steer_rad = math.radians(float(steer_raw) / 10.0)
            except (TypeError, ValueError):
                pass

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
                    # Subsequent IMU messages: use as absolute heading reference,
                    # same as the reference repo's `self.yaw = imu.yaw`.
                    # Rate-limited to at most _MAX_PHYSICAL_YAW_RATE_RADS × dt_imu
                    # to reject servo-EMI step spikes (magnetometer interference
                    # when steering angle changes sharply).
                    #
                    # SERVO EMI INHIBIT: at large steering angles the servo's
                    # static magnetic field creates a sustained bias in the
                    # BNO055 magnetometer.  Applying the absolute correction
                    # during those frames pulls _last_yaw_rad toward the biased
                    # reading instead of the true heading, accelerating drift.
                    # When |steer| >= _IMU_STEER_INHIBIT_DEG we skip the IMU
                    # correction entirely; the bicycle model (below) provides
                    # per-frame heading integration instead.
                    _steer_abs_deg = abs(math.degrees(self._last_steer_rad))
                    if _steer_abs_deg < _IMU_STEER_INHIBIT_DEG:
                        yaw_imu = yaw_raw_rad + self._yaw_offset
                        dt_imu = (now - _prev_imu_t) if _prev_imu_t is not None else 0.05
                        delta = yaw_imu - self._last_yaw_rad
                        while delta > math.pi:
                            delta -= 2.0 * math.pi
                        while delta < -math.pi:
                            delta += 2.0 * math.pi
                        max_delta = _MAX_PHYSICAL_YAW_RATE_RADS * max(dt_imu, 0.02)
                        self._last_yaw_rad += max(-max_delta, min(max_delta, delta))

                self._imu_received = True
            except Exception:
                pass

        # ---- Bicycle model yaw integration (per-frame heading update).
        # On frames where an IMU message arrived, this adds the instantaneous
        # turning increment on top of the IMU absolute correction above.
        # On frames without an IMU message, this is the sole heading source.
        # yaw_rate = (v / L) * tan(steer)
        # CurrentSteer > 0 = right (CW) → in math CCW convention yaw decreases.
        # dt is capped at _MAX_INTEGRATION_DT to limit error from frame drops.
        _eff_steer_rad = self._last_steer_rad * _STEER_GAIN_DR
        if abs(self._last_speed) > 0.005 and self._yaw_offset_calibrated:
            yaw_dt = min(dt, _MAX_INTEGRATION_DT)
            yaw_rate = (self._last_speed / _WHEELBASE_M) * math.tan(_eff_steer_rad)
            self._last_yaw_rad -= yaw_rate * yaw_dt

        # ---- Camera-based yaw correction (soft blend toward camera estimate).
        # When the camera sees both lane lines (two-line midpoint_ref mode), it can
        # estimate the car's absolute world-frame yaw as:
        #   cam_yaw = path_psi_at_waypoint + camera_heading_rad
        # This corrects accumulated bicycle-model drift (servo bias, RK4 error)
        # without depending on the BNO055 magnetometer.
        # Alpha ≈ 0.08: corrects ~80 % of a 20° drift over ~20 camera frames (2 s).
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
            if self._dr is not None:
                self._dr.correct_yaw(_yaw_correction_rad)
        self.tracking_state.last_yaw_correction_deg = math.degrees(_yaw_correction_rad)

        if self._dr is None or self._graph is None or self._path_manager is None:
            return

        # ---- Dead reckoning update (RK4)
        # Cap dt to limit position error during frame drops.
        # Pass steer_rad so RK4 can account for heading change within the step —
        # critical at high speed where Euler accumulates O(dt²) error per step.
        dr_dt = min(dt, _MAX_INTEGRATION_DT)
        self._dr.update(self._last_speed, self._last_yaw_rad, dr_dt,
                        steer_rad=_eff_steer_rad, wheelbase_m=_WHEELBASE_M)
        raw_x, raw_y, raw_yaw = self._dr.get_state()
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
        if semantic_relocalized and semantic_match is not None:
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
        self.tracking_state.update(
            x=matched_x, y=matched_y, yaw=matched_yaw,
            steer_rad=self._last_steer_rad,
            error_m=error_m, heading_rad=heading_rad,
            path_psi=path_psi, path_kappa=path_kappa,
            speed_mps=self._last_speed,
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
            if route is None or len(route.waypoints) == 0:
                return
            now = time.monotonic()
            wp = route.waypoints[target_idx % n_wp] if route.closed_loop else route.waypoints[min(max(0, target_idx), n_wp - 1)]
            dist_to_wp = math.hypot(matched_x - wp[0], matched_y - wp[1])

            # Speed field
            raw_spd = self._last_raw_speed
            spd_age = f"{now - self._last_speed_t:.1f}s" if self._last_speed_t else "never"
            spd_str = (
                f"speed_raw={raw_spd} spd={self._last_speed*100:.1f}cm/s (age={spd_age})"
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

            path_kappa = self._path_manager._get_curvature(route, target_idx)
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
