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
from src.utils.messages.allMessages import CurrentSpeed, ImuData, Location

from src.hardware.tracking.deadReckoning import DeadReckoning
from src.hardware.tracking.trackGraph import TrackGraph

try:
    import config as cfg
    _GRAPHML_PATH = getattr(cfg, "TRACKING_GRAPHML", "Track GraphML File.graphml")
    _STEP_M = getattr(cfg, "TRACKING_WAYPOINT_STEP_M", 0.05)
    _ADVANCE_DIST = getattr(cfg, "TRACKING_ADVANCE_DIST_M", 0.15)
    _INTERSECTION_LOOKAHEAD = getattr(cfg, "TRACKING_INTERSECTION_LOOKAHEAD_M", 0.40)
    _SHOW_WINDOW = getattr(cfg, "TRACKING_SHOW_WINDOW", True)
    _LOOP_HZ = 50
except Exception:
    _GRAPHML_PATH = "Track GraphML File.graphml"
    _STEP_M = 0.05
    _ADVANCE_DIST = 0.15
    _INTERSECTION_LOOKAHEAD = 0.40
    _SHOW_WINDOW = True
    _LOOP_HZ = 50

# How many dense waypoints correspond to the intersection lookahead distance
_LOOKAHEAD_PTS = max(2, int(_INTERSECTION_LOOKAHEAD / _STEP_M))

# BFMC track physical dimensions in metres (for Location scaling to dashboard)
_TRACK_W_M = 20.67
_TRACK_H_M = 13.76


class TrackingState:
    """Thread-safe shared state between threadTracking and threadLineFollowing.

    Attributes (readable by any thread in the same process):
        x, y, yaw        Current pose from dead reckoning (metres, radians).
        error_m          Lateral crosstrack error vs. current waypoint.
        heading_rad      Heading error vs. current waypoint tangent.
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
        self.error_m = 0.0
        self.heading_rad = 0.0
        self.speed_mps = 0.0
        self.wp_idx = 0
        self.waypoint_mode_active = False
        self.node_attr = 0
        self.initialized = False

    def update(self, x, y, yaw, error_m, heading_rad, speed_mps,
               wp_idx, waypoint_mode, node_attr):
        with self._lock:
            self.x = x
            self.y = y
            self.yaw = yaw
            self.error_m = error_m
            self.heading_rad = heading_rad
            self.speed_mps = speed_mps
            self.wp_idx = wp_idx
            self.waypoint_mode_active = waypoint_mode
            self.node_attr = node_attr
            self.initialized = True

    def snapshot(self):
        with self._lock:
            return dict(
                x=self.x, y=self.y, yaw=self.yaw,
                error_m=self.error_m, heading_rad=self.heading_rad,
                speed_mps=self.speed_mps, wp_idx=self.wp_idx,
                waypoint_mode_active=self.waypoint_mode_active,
                node_attr=self.node_attr,
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
        # Location sender → dashboard map display
        self._loc_sender = messageHandlerSender(queuesList, Location)

        # Load the track graph
        graphml_path = _GRAPHML_PATH
        if not os.path.isabs(graphml_path):
            # Resolve relative to workspace root (two levels up from this file)
            _here = os.path.dirname(os.path.abspath(__file__))
            _root = os.path.join(_here, "..", "..", "..", "..")
            graphml_path = os.path.normpath(os.path.join(_root, graphml_path))

        self._graph = None
        self._dr = None
        try:
            self._graph = TrackGraph(graphml_path, step_m=_STEP_M)
            x0, y0, yaw0 = self._graph.get_start_pose()
            self._dr = DeadReckoning(x0, y0, yaw0)
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
        self._last_yaw_rad = 0.0

    # ------------------------------------------------------------------
    def thread_work(self):
        now = time.monotonic()
        dt = now - self._last_t
        self._last_t = now

        # ---- Read latest speed (float, raw units = 10 cm/s)
        speed_raw = self._speed_sub.receive()
        if speed_raw is not None:
            try:
                self._last_speed = float(speed_raw) * 0.001  # → m/s
            except (TypeError, ValueError):
                pass

        # ---- Read latest IMU data (str repr of dict, yaw in degrees)
        imu_raw = self._imu_sub.receive()
        if imu_raw is not None:
            try:
                imu_dict = ast.literal_eval(str(imu_raw))
                yaw_deg = float(imu_dict.get("yaw", math.degrees(self._last_yaw_rad)))
                self._last_yaw_rad = math.radians(yaw_deg)
            except Exception:
                pass

        if self._dr is None or self._graph is None:
            return

        # ---- Dead reckoning update
        self._dr.update(self._last_speed, self._last_yaw_rad, dt)
        x, y, yaw = self._dr.get_state()

        # ---- Advance waypoint index when car is close enough
        n_wp = len(self._graph.waypoints)
        if n_wp == 0:
            return

        wp = self._graph.waypoints[self._wp_idx % n_wp]
        dist_to_wp = math.hypot(x - wp[0], y - wp[1])
        if dist_to_wp < _ADVANCE_DIST:
            self._wp_idx = (self._wp_idx + 1) % n_wp

        # ---- Find the lookahead waypoint (used for control)
        target_idx = self._graph.find_waypoint_ahead(
            x, y, self._wp_idx, lookahead_m=_ADVANCE_DIST
        )

        # ---- Compute tracking errors
        error_m, heading_rad = self._graph.compute_tracking_error(x, y, yaw, target_idx)

        # ---- Detect precision zone (intersection / stop-line ahead)
        in_precision_zone = self._graph.is_precision_zone(
            target_idx, lookahead_pts=_LOOKAHEAD_PTS
        )
        node_attr = int(self._graph.wp_node_attrs[target_idx % n_wp])

        # ---- Write shared state (consumed by threadLineFollowing & visualizer)
        self.tracking_state.update(
            x=x, y=y, yaw=yaw,
            error_m=error_m, heading_rad=heading_rad,
            speed_mps=self._last_speed,
            wp_idx=target_idx,
            waypoint_mode=in_precision_zone,
            node_attr=node_attr,
        )

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
            self._loc_sender.send({"x": round(x, 4), "y": round(y, 4)})
        except Exception:
            pass
