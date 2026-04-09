"""
trackVisualizer — real-time OpenCV map window.

Runs at ~10 Hz in its own thread.  Reads the latest TrackingState snapshot
and draws:
  • Track image background (from Track Editor Save.json + JPG)
  • Track spline path
  • Nodes coloured by attribute:
      NORMAL        → green
      STOPLINE      → red
      INTERSECTION  → blue
      HIGHWAY_LEFT  → yellow
      HIGHWAY_RIGHT → orange
      CROSSWALK     → magenta
  • Current target waypoint (cyan circle)
  • Car pose (scaled rectangle with Ackermann wheels, body heading)

If bg_image_path / track_json_path are not provided, falls back to a plain
black canvas with auto-scaled coordinates.
"""

import json
import math
import threading
import time
import warnings

import numpy as np

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False
    warnings.warn("[trackVisualizer] cv2 not available — visualizer disabled")

from src.hardware.tracking.trackGraph import (
    TrackGraph,
    ATTR_NORMAL, ATTR_CROSSWALK, ATTR_INTERSECTION, ATTR_ONEWAY,
    ATTR_HIGHWAY_LEFT, ATTR_HIGHWAY_RIGHT, ATTR_ROUNDABOUT, ATTR_STOPLINE,
)

# BGR node colours
_NODE_COLORS = {
    ATTR_NORMAL:        (100, 220, 100),   # green
    ATTR_CROSSWALK:     (220, 100, 220),   # magenta
    ATTR_INTERSECTION:  (220, 100, 100),   # blue
    ATTR_ONEWAY:        (200, 200, 200),   # light grey
    ATTR_HIGHWAY_LEFT:  (50,  220, 220),   # yellow
    ATTR_HIGHWAY_RIGHT: (50,  165, 255),   # orange
    ATTR_ROUNDABOUT:    (255, 200,  50),   # teal-ish
    ATTR_STOPLINE:      (50,   50, 220),   # red
}
_DEFAULT_COLOR = (180, 180, 180)

_CANVAS_SIZE  = 700
_MARGIN_PX    = 40
_EDGE_COLOR   = (80, 80, 80)
_WP_COLOR     = (255, 220, 50)
_CAR_COLOR    = (255, 255, 255)
_FPS          = 10
_IMG_DIM      = 0.55   # background image brightness (0–1)


class TrackVisualizer(threading.Thread):
    """OpenCV visualisation thread.

    Args:
        graph:           TrackGraph instance (for static geometry).
        window_name:     cv2 window title.
        canvas_size:     Window size in pixels (square).
        bg_image_path:   Path to the track photo (JPG/PNG).
        track_json_path: Path to Track Editor Save.json (provides
                         metersPerPixel, imgW, imgH).
    """

    def __init__(self, graph: TrackGraph,
                 window_name: str = "Track Navigation",
                 canvas_size: int = _CANVAS_SIZE,
                 bg_image_path: str = None,
                 track_json_path: str = None):
        super().__init__(daemon=True)
        self._graph        = graph
        self._window_name  = window_name
        self._canvas_size  = canvas_size
        self._stop_event   = threading.Event()
        self._state_lock   = threading.Lock()
        self._state        = None
        self._base_canvas  = None

        # Image-mode coordinate params (set in _prerender if image loads OK)
        self._use_image_coords  = False
        self._img_scale         = 1.0
        self._img_offset_x      = 0
        self._img_offset_y      = 0
        self._meters_per_pixel  = None
        self._img_w             = None
        self._img_h             = None

        # Legacy fallback params
        self._scale  = 1.0
        self._ox     = 0.0
        self._y_min  = 0.0

        self._prerender(bg_image_path, track_json_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_state(self, state_snapshot: dict) -> None:
        """Called by threadTracking to push a new state snapshot."""
        with self._state_lock:
            self._state = state_snapshot

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Thread main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        if not _CV2_OK:
            return
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, self._canvas_size, self._canvas_size)
        interval = 1.0 / _FPS
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            self._draw_frame()
            elapsed = time.monotonic() - t0
            wait = max(1, int((interval - elapsed) * 1000))
            key = cv2.waitKey(wait)
            if key == ord("q"):
                break
        cv2.destroyWindow(self._window_name)

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------
    def _world_to_px(self, x: float, y: float):
        """Convert world metres → integer canvas pixel coordinates."""
        if self._use_image_coords:
            # world → image pixel (Y-flip: in the image y=0 is top, world y=0 is bottom)
            img_x = x / self._meters_per_pixel
            img_y = self._img_h - y / self._meters_per_pixel
            px = int(self._img_offset_x + img_x * self._img_scale)
            py = int(self._img_offset_y + img_y * self._img_scale)
            return px, py
        else:
            px = int(self._ox + x * self._scale)
            py = int(self._canvas_size - _MARGIN_PX - (y - self._y_min) * self._scale)
            return px, py

    # ------------------------------------------------------------------
    # Pre-render (static background)
    # ------------------------------------------------------------------
    def _prerender(self, bg_image_path=None, track_json_path=None) -> None:
        """Build the static background canvas with image + edges + nodes."""
        if not _CV2_OK:
            return

        graph       = self._graph
        canvas_size = self._canvas_size

        # ── Attempt image background ──────────────────────────────────────────
        bg = None
        if bg_image_path and track_json_path:
            try:
                with open(track_json_path) as f:
                    jdata = json.load(f)
                mpp  = jdata.get('metersPerPixel')
                imgW = jdata.get('imgW')
                imgH = jdata.get('imgH')
                raw  = cv2.imread(bg_image_path)
                if raw is not None and mpp and imgW and imgH:
                    # Scale image to fit canvas preserving aspect ratio
                    s     = min(canvas_size / imgW, canvas_size / imgH)
                    new_w = int(imgW * s)
                    new_h = int(imgH * s)
                    off_x = (canvas_size - new_w) // 2
                    off_y = (canvas_size - new_h) // 2
                    resized = cv2.resize(raw, (new_w, new_h),
                                        interpolation=cv2.INTER_AREA)
                    # Dim so overlaid nodes are clearly visible
                    resized = (resized * _IMG_DIM).astype(np.uint8)
                    bg = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
                    bg[off_y:off_y + new_h, off_x:off_x + new_w] = resized
                    self._use_image_coords = True
                    self._meters_per_pixel = mpp
                    self._img_w      = imgW
                    self._img_h      = imgH
                    self._img_scale  = s
                    self._img_offset_x = off_x
                    self._img_offset_y = off_y
                    print(f"[TrackVisualizer] Background loaded: {bg_image_path} "
                          f"({imgW}×{imgH}px, {mpp:.6f} m/px)")
            except Exception as e:
                print(f"[TrackVisualizer] Warning — background not loaded: {e}")

        # ── Fallback: plain dark canvas ───────────────────────────────────────
        if bg is None:
            bg = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
            if len(graph.ordered_nodes) > 0:
                xs = [n.x for n in graph.ordered_nodes]
                ys = [n.y for n in graph.ordered_nodes]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                w = max(x_max - x_min, 0.01)
                h = max(y_max - y_min, 0.01)
                drawable     = canvas_size - 2 * _MARGIN_PX
                self._scale  = min(drawable / w, drawable / h)
                self._ox     = _MARGIN_PX - x_min * self._scale
                self._y_min  = y_min

        # ── Draw spline path ──────────────────────────────────────────────────
        wps = graph.waypoints
        if len(wps) > 1:
            step = max(1, len(wps) // 500)
            for i in range(0, len(wps) - 1, step):
                p1 = self._world_to_px(float(wps[i, 0]),     float(wps[i, 1]))
                p2 = self._world_to_px(float(wps[i + 1, 0]), float(wps[i + 1, 1]))
                cv2.line(bg, p1, p2, _EDGE_COLOR, 1)

        # ── Draw nodes ────────────────────────────────────────────────────────
        for node in graph.ordered_nodes:
            px, py = self._world_to_px(node.x, node.y)
            color  = _NODE_COLORS.get(node.attribute, _DEFAULT_COLOR)
            cv2.circle(bg, (px, py), 5, color, -1)
            cv2.circle(bg, (px, py), 6, (0, 0, 0), 1)   # thin outline

        self._base_canvas = bg

    # ------------------------------------------------------------------
    # Dynamic frame + car geometry (TC-04 physical dimensions, metres)
    # ------------------------------------------------------------------
    _CAR_LEN       = 0.365   # total body length
    _CAR_W         = 0.190   # total body width
    _WHEELBASE     = 0.260   # rear axle → front axle
    _REAR_OVERHANG = 0.018   # rear axle → rear bumper
    _FRONT_OVERHANG= 0.072   # front axle → front bumper
    _WHEEL_L       = 0.055   # wheel rectangle length (visual)
    _WHEEL_W       = 0.022   # wheel rectangle width (visual)

    def _world_offset_px(self, base_x: float, base_y: float,
                         fwd: tuple, lat: tuple,
                         long_m: float, lat_m: float):
        """World point = base + long_m * fwd + lat_m * lat, converted to px."""
        wx = base_x + long_m * fwd[0] + lat_m * lat[0]
        wy = base_y + long_m * fwd[1] + lat_m * lat[1]
        return self._world_to_px(wx, wy)

    def _wheel_corners(self, axle_wx: float, axle_wy: float,
                       angle_rad: float) -> list:
        """Four pixel corners of one wheel rectangle centred on its axle point."""
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        fw = (c,  s)
        lw = (-s, c)
        hl, hw = self._WHEEL_L / 2.0, self._WHEEL_W / 2.0
        offsets = [(-hl, -hw), (-hl, +hw), (+hl, +hw), (+hl, -hw)]
        return [self._world_to_px(axle_wx + lo * fw[0] + la * lw[0],
                                  axle_wy + lo * fw[1] + la * lw[1])
                for lo, la in offsets]

    def _draw_frame(self) -> None:
        """Render one frame (static bg + dynamic car/waypoint) and display."""
        if self._base_canvas is None:
            return
        canvas      = self._base_canvas.copy()
        canvas_size = self._canvas_size

        with self._state_lock:
            state = dict(self._state) if self._state is not None else None

        if state is not None:
            x         = state.get("x",         0.0)
            y         = state.get("y",         0.0)
            yaw       = state.get("yaw",       0.0)
            steer_rad = state.get("steer_rad", 0.0)
            wp_idx    = state.get("wp_idx",    0)

            # Active route preview and destination marker
            route_points = state.get("route_points") or []
            if len(route_points) >= 2:
                route_poly = np.array(
                    [
                        self._world_to_px(float(pt.get("x", 0.0)), float(pt.get("y", 0.0)))
                        for pt in route_points
                    ],
                    dtype=np.int32,
                )
                cv2.polylines(canvas, [route_poly.reshape(-1, 1, 2)], False, (120, 255, 255), 2)

            destination_point = state.get("destination_point")
            if isinstance(destination_point, dict):
                dpx, dpy = self._world_to_px(
                    float(destination_point.get("x", 0.0)),
                    float(destination_point.get("y", 0.0)),
                )
                cv2.circle(canvas, (dpx, dpy), 7, (0, 180, 255), 2)
            else:
                # Backwards-compatible target marker on the reference path.
                wps = self._graph.waypoints
                if len(wps) > 0 and wp_idx < len(wps):
                    wx, wy   = wps[wp_idx % len(wps), :2]
                    wpx, wpy = self._world_to_px(float(wx), float(wy))
                    cv2.circle(canvas, (wpx, wpy), 7, _WP_COLOR, 2)

            # ── Car geometry ──────────────────────────────────────────────────
            # (x, y) is the rear-axle centre in world metres (DR reference pt).
            # fwd = car-forward unit vector, lat = car-left unit vector.
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            fwd = (cos_y,  sin_y)
            lat = (-sin_y, cos_y)

            WB = self._WHEELBASE
            FR = self._REAR_OVERHANG
            FF = self._FRONT_OVERHANG
            W2 = self._CAR_W / 2.0

            # Body rectangle (rear-axle is the longitudinal origin)
            body_pts = np.array([
                self._world_offset_px(x, y, fwd, lat, -FR,    -W2),   # rear-right
                self._world_offset_px(x, y, fwd, lat, -FR,    +W2),   # rear-left
                self._world_offset_px(x, y, fwd, lat, WB+FF,  +W2),   # front-left
                self._world_offset_px(x, y, fwd, lat, WB+FF,  -W2),   # front-right
            ], dtype=np.int32)
            cv2.fillPoly(canvas, [body_pts], _CAR_COLOR)
            cv2.polylines(canvas, [body_pts.reshape(-1, 1, 2)],
                          isClosed=True, color=(100, 200, 255), thickness=1)

            # Axle dots
            ra_px = self._world_to_px(x, y)
            fa_wx = x + WB * fwd[0]
            fa_wy = y + WB * fwd[1]
            fa_px = self._world_to_px(fa_wx, fa_wy)
            cv2.circle(canvas, ra_px, 2, (160, 160, 160), -1)
            cv2.circle(canvas, fa_px, 2, (160, 160, 160), -1)

            # ── Wheels ────────────────────────────────────────────────────────
            # Ackermann: front-inner and front-outer wheels have different angles.
            # R = WB / tan(steer);  inner_angle = atan(WB / (R - W2))
            #                       outer_angle = atan(WB / (R + W2))
            _WHEEL_COLOR = (40, 40, 40)

            abs_steer = abs(steer_rad)
            if abs_steer > 1e-3:
                R = WB / math.tan(abs_steer)
                sign = 1.0 if steer_rad > 0 else -1.0
                steer_inner = sign * math.atan2(WB, R - W2)
                steer_outer = sign * math.atan2(WB, R + W2)
            else:
                steer_inner = steer_outer = 0.0

            # Front-left centre world
            fl_wx = fa_wx + W2 * lat[0]
            fl_wy = fa_wy + W2 * lat[1]
            # Front-right centre world
            fr_wx = fa_wx - W2 * lat[0]
            fr_wy = fa_wy - W2 * lat[1]
            # Rear-left centre world
            rl_wx = x + W2 * lat[0]
            rl_wy = y + W2 * lat[1]
            # Rear-right centre world
            rr_wx = x - W2 * lat[0]
            rr_wy = y - W2 * lat[1]

            # Ackermann inner/outer assignment depends on turn direction:
            #   right turn (steer > 0): right wheel is inner (larger angle)
            #   left  turn (steer < 0): left  wheel is inner (larger magnitude)
            if steer_rad >= 0:
                fl_steer, fr_steer = steer_outer, steer_inner
            else:
                fl_steer, fr_steer = steer_inner, steer_outer

            for whl_wx, whl_wy, whl_angle in [
                (fl_wx, fl_wy, yaw + fl_steer),
                (fr_wx, fr_wy, yaw + fr_steer),
                (rl_wx, rl_wy, yaw),
                (rr_wx, rr_wy, yaw),
            ]:
                corners = self._wheel_corners(whl_wx, whl_wy, whl_angle)
                whl_pts = np.array(corners, dtype=np.int32)
                cv2.fillPoly(canvas, [whl_pts], _WHEEL_COLOR)
                cv2.polylines(canvas, [whl_pts.reshape(-1, 1, 2)],
                              isClosed=True, color=(180, 180, 180), thickness=1)

            # ── Telemetry text ────────────────────────────────────────────────
            route_mode = state.get("route_active", False)
            mode = "ROUTE" if route_mode else ("WP MODE" if state.get("waypoint_mode_active") else "VISUAL")
            spd  = state.get("speed_mps", 0.0)
            raw_x = state.get("raw_x", x)
            raw_y = state.get("raw_y", y)
            raw_yaw = state.get("raw_yaw", yaw)
            map_match_error_m = state.get("map_match_error_m", 0.0)
            lat_corr_m = state.get("camera_lateral_correction_m", 0.0)
            raw_lat_err_m = state.get("raw_lateral_error_m", 0.0)
            lane_rel = "cam" if state.get("lane_measurement_reliable") else "graph"
            maneuver_type = state.get("maneuver_type", "none")
            route_progress = float(state.get("route_progress", 0.0) or 0.0)
            current_node_id = state.get("current_node_id")
            upcoming_node_id = state.get("upcoming_node_id")
            next_semantic = state.get("next_semantic_label") or state.get("next_semantic_type") or "none"
            relocalization_mode = state.get("relocalization_mode", "map_match")
            cv2.putText(canvas,
                        f"match x={x:.2f}m  y={y:.2f}m  yaw={math.degrees(yaw):.0f}°"
                        f"  steer={math.degrees(steer_rad):+.0f}°",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(canvas,
                        f"raw x={raw_x:.2f}m  y={raw_y:.2f}m  yaw={math.degrees(raw_yaw):.0f}°",
                        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(canvas,
                        f"e={state.get('error_m', 0):.3f}m  "
                        f"h={math.degrees(state.get('heading_rad', 0)):.1f}°  "
                        f"mm={map_match_error_m:.3f}m  lat={lat_corr_m:+.3f}m  "
                        f"raw_lat={raw_lat_err_m:+.3f}m  [{mode}|{lane_rel}]",
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(canvas,
                        f"v={spd * 100:.1f}cm/s  wp={wp_idx}  tgt={state.get('target_idx', wp_idx)}"
                        f"  man={maneuver_type}",
                        (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(canvas,
                        f"route={state.get('route_id', 'none')}  "
                        f"curr={current_node_id}  next={upcoming_node_id}  "
                        f"prog={route_progress * 100:.0f}%",
                        (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(canvas,
                        f"semantic={next_semantic}  reloc={relocalization_mode}",
                        (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        cv2.imshow(self._window_name, canvas)
