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
    _DEBUG_LOG = getattr(cfg, "TRACKING_DEBUG_LOG", False)
    _LOOP_HZ = 50
except Exception:
    _GRAPHML_PATH = "Track GraphML File.graphml"
    _STEP_M = 0.05
    _ADVANCE_DIST = 0.15
    _INTERSECTION_LOOKAHEAD = 0.40
    _SHOW_WINDOW = True
    _DEBUG_LOG = False
    _LOOP_HZ = 50

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
        x, y, yaw        Current pose from dead reckoning (metres, radians).
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
        self.error_m = 0.0
        self.heading_rad = 0.0
        self.path_psi = 0.0
        self.path_kappa = 0.0   # Signed path curvature at current wp (1/m)
        self.speed_mps = 0.0
        self.wp_idx = 0
        self.waypoint_mode_active = False
        self.node_attr = 0
        self.initialized = False
        self.imu_received = False   # True once a real IMU message has been parsed
        # Reference to the dead reckoning instance — set by threadTracking so
        # threadLineFollowing can push lane-based lateral corrections.
        self._dr = None

    def update(self, x, y, yaw, error_m, heading_rad, path_psi, path_kappa,
               speed_mps, wp_idx, waypoint_mode, node_attr, imu_received=False):
        with self._lock:
            self.x = x
            self.y = y
            self.yaw = yaw
            self.error_m = error_m
            self.heading_rad = heading_rad
            self.path_psi = path_psi
            self.path_kappa = path_kappa
            self.speed_mps = speed_mps
            self.wp_idx = wp_idx
            self.waypoint_mode_active = waypoint_mode
            self.node_attr = node_attr
            self.initialized = True
            if imu_received:
                self.imu_received = True

    def correct_lateral(self, lateral_error_m: float) -> None:
        """Push a lane-detection-based lateral correction into dead reckoning.

        Called by threadLineFollowing when two lane lines are visible and the
        measured crosstrack error is reliable.  Nudges the DR position so the
        next heading/error computation is more accurate.

        Args:
            lateral_error_m: Signed lane error from lane detection (m).
                             Positive = car right of lane centre.
        """
        with self._lock:
            dr = self._dr
            psi = self.path_psi
        if dr is not None:
            dr.correct_lateral(lateral_error_m, psi)

    def snapshot(self):
        with self._lock:
            return dict(
                x=self.x, y=self.y, yaw=self.yaw,
                error_m=self.error_m, heading_rad=self.heading_rad,
                path_psi=self.path_psi, path_kappa=self.path_kappa,
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
            # Resolve relative to workspace root: src/hardware/tracking/ → 3 levels up
            _here = os.path.dirname(os.path.abspath(__file__))
            _root = os.path.join(_here, "..", "..", "..")
            graphml_path = os.path.normpath(os.path.join(_root, graphml_path))

        self._graph = None
        self._dr = None
        self._start_yaw_rad = 0.0   # path tangent at start — used as yaw fallback
        try:
            self._graph = TrackGraph(graphml_path, step_m=_STEP_M)
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
                        "imu_yaw=<deg> | x=<m> y=<m> yaw=<deg> | "
                        "wp=<idx>/<total> wp_xy=(<x>,<y>) dist=<m> | "
                        "err=<m> hdg=<deg> | zone=<attr> | dt=<ms>\n"
                    )
            except Exception:
                pass

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

        # ---- Read latest IMU data (str repr of dict, yaw in degrees)
        # Nucleo format: @imu:roll;pitch;yaw;accelx;accely;accelz;;
        # threadRead parses this and sends str({'roll':..., 'yaw':..., ...})
        imu_raw = self._imu_sub.receive()
        if imu_raw is not None:
            try:
                imu_dict = ast.literal_eval(str(imu_raw))
                self._last_raw_imu = imu_dict
                self._last_imu_t = now
                yaw_deg = float(imu_dict.get("yaw", math.degrees(self._last_yaw_rad)))
                yaw_raw_rad = math.radians(yaw_deg)

                if not self._yaw_offset_calibrated:
                    # First IMU message: compute offset so that corrected_yaw == track-
                    # start tangent when the car is physically aligned with the track.
                    # IMU has an arbitrary reference (wherever Nucleo was powered on),
                    # while the GraphML uses its own coordinate system.
                    # offset = map_start_yaw - raw_imu_yaw_at_start
                    self._yaw_offset = self._start_yaw_rad - yaw_raw_rad
                    self._yaw_offset_calibrated = True
                    if self.logging:
                        self.logging.info(
                            f"[threadTracking] IMU calibrated: raw_yaw={yaw_deg:.1f}°, "
                            f"start_yaw={math.degrees(self._start_yaw_rad):.1f}°, "
                            f"offset={math.degrees(self._yaw_offset):.1f}°"
                        )
                    else:
                        print(
                            f"[threadTracking] IMU calibrated: raw={yaw_deg:.1f}°  "
                            f"start_map={math.degrees(self._start_yaw_rad):.1f}°  "
                            f"offset={math.degrees(self._yaw_offset):.1f}°"
                        )

                # Apply offset: corrected yaw is in the same frame as the GraphML map
                corrected_yaw = yaw_raw_rad + self._yaw_offset
                # Continuity unwrapping: compute delta from previous corrected yaw
                # and wrap it to [-pi, pi] to handle ±180° roll-overs.
                delta_yaw = corrected_yaw - self._last_yaw_rad
                while delta_yaw > math.pi:
                    delta_yaw -= 2.0 * math.pi
                while delta_yaw < -math.pi:
                    delta_yaw += 2.0 * math.pi
                # Re-zero detection: the BNO055 occasionally resets its heading
                # reference internally (NDOF magnetometer recalibration). This
                # produces a jump that is physically impossible given the car's
                # max yaw rate (~120 deg/s) and IMU update rate (~10 Hz), which
                # caps the maximum real delta at ~12°. Any jump above 20° is
                # therefore a re-zero event, not an actual rotation.
                # When detected: recompute the offset so that the car's current
                # heading (_last_yaw_rad) stays unchanged and future samples from
                # the new sensor reference frame remain coherent.
                _MAX_YAW_STEP = math.radians(20.0)
                if abs(delta_yaw) > _MAX_YAW_STEP:
                    self._yaw_offset = self._last_yaw_rad - yaw_raw_rad
                    _msg = (
                        f"[threadTracking] BNO055 re-zero detected: "
                        f"raw={yaw_deg:.1f}°  delta={math.degrees(delta_yaw):.1f}°  "
                        f"new_offset={math.degrees(self._yaw_offset):.1f}°"
                    )
                    if self.logging:
                        self.logging.warning(_msg)
                    else:
                        print(_msg)
                    # _last_yaw_rad unchanged — heading is continuous
                else:
                    self._last_yaw_rad = self._last_yaw_rad + delta_yaw
                self._imu_received = True
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

        # ---- Path tangent and curvature at the target waypoint
        path_psi = float(self._graph.waypoints[target_idx % n_wp][2])
        path_kappa = self._graph.get_curvature(target_idx)

        # ---- Detect precision zone (intersection / stop-line ahead)
        in_precision_zone = self._graph.is_precision_zone(
            target_idx, lookahead_pts=_LOOKAHEAD_PTS
        )
        node_attr = int(self._graph.wp_node_attrs[target_idx % n_wp])

        # ---- Write shared state (consumed by threadLineFollowing & visualizer)
        self.tracking_state.update(
            x=x, y=y, yaw=yaw,
            error_m=error_m, heading_rad=heading_rad,
            path_psi=path_psi, path_kappa=path_kappa,
            speed_mps=self._last_speed,
            wp_idx=target_idx,
            waypoint_mode=in_precision_zone,
            node_attr=node_attr,
            imu_received=self._imu_received,
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

        # ---- Tracking debug log ───────────────────────────────────────────
        self._frame_idx += 1
        if self._debug_log_enabled and self._debug_log_path and \
                (self._frame_idx % self._log_every == 0):
            self._write_tracking_log(
                x, y, yaw, error_m, heading_rad,
                target_idx, n_wp, in_precision_zone, node_attr, dt,
            )

    def _write_tracking_log(self, x, y, yaw, error_m, heading_rad,
                             wp_idx, n_wp, in_precision_zone, node_attr, dt):
        """Write one line to temp/tracking_debug.txt."""
        try:
            now = time.monotonic()
            wp = self._graph.waypoints[wp_idx % n_wp]
            dist_to_wp = math.hypot(x - wp[0], y - wp[1])

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
                )
            else:
                imu_str = (
                    f"imu{imu_ok} yaw={math.degrees(self._last_yaw_rad):.1f}°"
                    f" (seeded_from_track, age={imu_age})"
                )

            zone_str = _ATTR_NAMES.get(node_attr, str(node_attr))
            if in_precision_zone:
                zone_str += "★"

            # Drift warning: speed=0 but DR position actually moved since last log
            # (avoids false positive when car stops with large accumulated error)
            drift_warn = ""
            prev_xy = getattr(self, "_log_prev_xy", None)
            if prev_xy is not None and abs(self._last_speed) < 1e-4:
                moved = math.hypot(x - prev_xy[0], y - prev_xy[1])
                if moved > 0.005:  # more than 5mm movement at zero speed = real drift
                    drift_warn = f" ⚠ DRIFT_WITH_ZERO_SPEED (moved {moved*100:.1f}cm)"
            self._log_prev_xy = (x, y)

            path_kappa = self._graph.get_curvature(wp_idx)
            ff_deg = math.degrees(math.atan(path_kappa * 0.258)) if abs(path_kappa) > 0.01 else 0.0
            line = (
                f"F{self._frame_idx:06d} | "
                f"{spd_str} | "
                f"{imu_str} | "
                f"x={x:.4f}m y={y:.4f}m yaw={math.degrees(yaw):.1f}° | "
                f"wp={wp_idx}/{n_wp} wp_xy=({wp[0]:.3f},{wp[1]:.3f}) dist={dist_to_wp:.3f}m | "
                f"err={error_m:+.4f}m hdg={math.degrees(heading_rad):+.2f}° "
                f"kappa={path_kappa:+.3f}/m ff={ff_deg:+.1f}° | "
                f"zone={zone_str} | dt={dt*1000:.1f}ms"
                f"{drift_warn}\n"
            )
            with open(self._debug_log_path, "a") as f:
                f.write(line)
        except Exception:
            pass
