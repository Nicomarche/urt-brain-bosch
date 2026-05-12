"""
trackVisualizer — real-time OpenCV map window.

Runs at ~10 Hz in its own thread.  Reads the latest TrackingState snapshot
and draws:
  • Track image background (SVG via cairosvg, or raster PNG/JPG fallback)
  • Track spline path
  • Nodes coloured by attribute
  • Current target waypoint (cyan circle)
  • Raw car pose (scaled rectangle with Ackermann wheels, body heading)
  • Map-matched pose as a reference marker when it differs from the raw pose
"""

import json
import math
import os
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

from src.routing.lanelet.attributes import (
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

_CANVAS_SIZE       = 700
_MARGIN_PX         = 40
_EDGE_COLOR        = (80, 80, 80)
_WP_COLOR          = (255, 220, 50)
_CAR_COLOR         = (255, 255, 255)
_FPS               = 25
_IMG_DIM           = 0.55   # background image brightness (0–1)
# Additional offset added to the forward prediction to compensate for encoder
# measurement pipeline delay (encoder → MCU → serial → Python), which is not
# captured by state_ts (set after DR integration).
_PIPELINE_DELAY_S  = 0.05


class TrackVisualizer(threading.Thread):
    """OpenCV visualisation thread.

    Args:
        graph:           OSM route handler instance (for static geometry).
        window_name:     cv2 window title.
        canvas_size:     Window size in pixels (square).
        bg_image_path:   Path to the track photo (JPG/PNG).
        track_json_path: Path to Track Editor Save.json (provides
                         metersPerPixel, imgW, imgH).
    """

    def __init__(self, graph,
                 window_name: str = "Track Navigation",
                 canvas_size: int = _CANVAS_SIZE,
                 bg_image_path: str = None,
                 track_json_path: str = None,
                 bg_svg_path: str = None):
        super().__init__(daemon=True)
        self._graph         = graph
        self._window_name   = window_name
        self._canvas_size   = canvas_size
        self._stop_event    = threading.Event()
        self._state_lock    = threading.Lock()
        self._state         = None
        self._base_canvas   = None
        # Latest rendered composite (set by _draw_frame). The PyQt5 GUI
        # ignores this and renders its own scene from SocketIO events; we
        # keep the field for any in-process consumer that prefers the
        # numpy canvas (debug scripts, future MapVisualization sender).
        self._latest_canvas = None

        # Image-mode coordinate params (set in _prerender if image loads OK)
        self._use_image_coords  = False
        self._img_scale         = 1.0
        self._img_offset_x      = 0
        self._img_offset_y      = 0
        self._meters_per_pixel  = None
        self._meters_per_pixel_x = None
        self._meters_per_pixel_y = None
        self._texture_x_min     = 0.0
        self._texture_y_min     = 0.0
        self._y_axis_inverted   = True
        self._img_w             = None
        self._img_h             = None

        # Legacy fallback params
        self._scale  = 1.0
        self._ox     = 0.0
        self._y_min  = 0.0

        self._prerender(bg_image_path, track_json_path, bg_svg_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_state(self, state_snapshot: dict) -> None:
        """Called by threadTracking to push a new state snapshot."""
        with self._state_lock:
            self._state = state_snapshot

    def get_latest_canvas(self):
        """Return the most recently rendered composite (or None).

        The canvas is a BGR ``np.ndarray`` of shape ``(canvas_size,
        canvas_size, 3)``. Callers should treat it as read-only; it may
        be replaced concurrently by the render loop.
        """
        return self._latest_canvas

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Thread main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Background loop.

        Originally this opened an OpenCV window and rendered the live
        track + car overlay at ~25 FPS. The window is gone — the unified
        PyQt5 GUI's ``map_view`` consumes the same data over SocketIO and
        renders a richer, interactive scene.

        We keep the thread alive (callers ``.start()`` it unconditionally)
        but it just produces canvases into ``_latest_canvas`` so any
        in-process consumer that wants the rendered frame can fetch it
        via ``get_latest_canvas()``. If nobody asks, no CPU is spent past
        the slow tick.
        """
        if not _CV2_OK:
            return
        # Slow tick — only useful if get_latest_canvas() is being polled.
        # Without an active consumer, we'd just be heating the Jetson.
        interval = 1.0 / max(1.0, _FPS / 5.0)  # 5 FPS, conservative
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                self._draw_frame()
            except Exception as exc:  # pragma: no cover - best effort
                print(f"[TrackVisualizer] draw error: {exc}")
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, interval - elapsed))

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------
    def _world_to_px(self, x: float, y: float):
        """Convert world metres → integer canvas pixel coordinates."""
        if self._use_image_coords:
            img_x = (float(x) - self._texture_x_min) / self._meters_per_pixel_x
            local_y = (float(y) - self._texture_y_min) / self._meters_per_pixel_y
            img_y = self._img_h - local_y if self._y_axis_inverted else local_y
            px = int(self._img_offset_x + img_x * self._img_scale)
            py = int(self._img_offset_y + img_y * self._img_scale)
            return px, py
        else:
            px = int(self._ox + x * self._scale)
            py = int(self._canvas_size - _MARGIN_PX - (y - self._y_min) * self._scale)
            return px, py

    # ------------------------------------------------------------------
    # Background image loader
    # ------------------------------------------------------------------
    @staticmethod
    def _load_background_image(bg_image_path: str | None,
                               bg_svg_path: str | None,
                               target_width_px: int | None) -> "np.ndarray | None":
        """Load the background image. Order of preference:
        1. SVG via cairosvg (vector source of truth, scales perfectly).
        2. Raster (PNG/JPG) at bg_image_path.
        3. Raster adjacent to the SVG (track.svg → track.png/.jpg).
        Returns a BGR ndarray or None if nothing loaded.
        """
        # Try SVG first if cairosvg is available — keeps the vector workflow.
        if bg_svg_path and os.path.exists(bg_svg_path):
            try:
                import cairosvg  # type: ignore
                png_bytes = cairosvg.svg2png(url=bg_svg_path,
                                             output_width=target_width_px)
                arr = np.frombuffer(png_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            except ImportError:
                pass  # cairosvg not installed → fall through to raster path
            except Exception as e:
                print(f"[TrackVisualizer] cairosvg rasterisation failed: {e}")

        if bg_image_path and os.path.exists(bg_image_path):
            img = cv2.imread(bg_image_path)
            if img is not None:
                return img

        # Last-chance: try a raster sitting next to the SVG.
        if bg_svg_path:
            stem = os.path.splitext(bg_svg_path)[0]
            for ext in (".png", ".jpg", ".jpeg"):
                candidate = stem + ext
                if os.path.exists(candidate):
                    img = cv2.imread(candidate)
                    if img is not None:
                        return img
        return None

    # ------------------------------------------------------------------
    # Pre-render (static background)
    # ------------------------------------------------------------------
    def _prerender(self, bg_image_path=None, track_json_path=None,
                   bg_svg_path=None) -> None:
        """Build the static background canvas with image + edges + nodes."""
        if not _CV2_OK:
            return

        graph       = self._graph
        canvas_size = self._canvas_size

        # ── Attempt image background ──────────────────────────────────────────
        bg = None
        if track_json_path and os.path.exists(track_json_path):
            try:
                with open(track_json_path) as f:
                    jdata = json.load(f)
                mpp  = jdata.get('metersPerPixel')
                mpp_x = jdata.get('metersPerPixelX', mpp)
                mpp_y = jdata.get('metersPerPixelY', mpp_x)
                imgW = jdata.get('imgW')
                imgH = jdata.get('imgH')
                bounds = jdata.get('world_bounds') or {}
                raw  = self._load_background_image(bg_image_path, bg_svg_path,
                                                   target_width_px=imgW)
                if raw is not None and mpp_x and mpp_y and imgW and imgH:
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
                    self._meters_per_pixel = mpp or mpp_x
                    self._meters_per_pixel_x = float(mpp_x)
                    self._meters_per_pixel_y = float(mpp_y)
                    self._texture_x_min = float(bounds.get('x_min', 0.0) or 0.0)
                    self._texture_y_min = float(bounds.get('y_min', 0.0) or 0.0)
                    self._y_axis_inverted = bool(jdata.get('y_axis_inverted', True))
                    self._img_w      = imgW
                    self._img_h      = imgH
                    self._img_scale  = s
                    self._img_offset_x = off_x
                    self._img_offset_y = off_y
                    src = bg_svg_path or bg_image_path or "<unknown>"
                    print(f"[TrackVisualizer] Background loaded: {src} "
                          f"({imgW}×{imgH}px, {float(mpp_x):.6f}×{float(mpp_y):.6f} m/px)")
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
            raw_x = state.get("raw_x", state.get("x", 0.0))
            raw_y = state.get("raw_y", state.get("y", 0.0))
            raw_yaw = state.get("raw_yaw", state.get("yaw", 0.0))
            matched_x = state.get("matched_x", raw_x)
            matched_y = state.get("matched_y", raw_y)
            matched_yaw = state.get("matched_yaw", raw_yaw)
            speed_mps = state.get("speed_mps", 0.0)
            state_ts  = state.get("state_ts")
            # Forward-predict the raw pose by the state snapshot age to
            # compensate for display lag (typically 50–100 ms at 10 FPS).
            # Uses bicycle-model arc motion so both position AND heading update
            # correctly when the car is turning.
            steer_rad = state.get("steer_rad", 0.0)
            # Use matched position directly — already map-snapped, no prediction needed.
            x   = matched_x
            y   = matched_y
            yaw = matched_yaw
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
            draw_yaw = yaw
            cos_y = math.cos(draw_yaw)
            sin_y = math.sin(draw_yaw)
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
            #   left  turn (steer > 0): left  wheel is inner (larger angle)
            #   right turn (steer < 0): right wheel is inner (larger magnitude)
            if steer_rad >= 0:
                fl_steer, fr_steer = steer_inner, steer_outer
            else:
                fl_steer, fr_steer = steer_outer, steer_inner

            for whl_wx, whl_wy, whl_angle in [
                (fl_wx, fl_wy, draw_yaw + fl_steer),
                (fr_wx, fr_wy, draw_yaw + fr_steer),
                (rl_wx, rl_wy, draw_yaw),
                (rr_wx, rr_wy, draw_yaw),
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
            spd_src = state.get("speed_source", "?")
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
            match_gap_m = math.hypot(float(raw_x) - float(matched_x), float(raw_y) - float(matched_y))
            display_yaw_deg = math.degrees(math.atan2(math.sin(draw_yaw), math.cos(draw_yaw)))

            matched_px = self._world_to_px(matched_x, matched_y)
            cv2.circle(canvas, matched_px, 5, (0, 215, 255), 1)
            cv2.circle(canvas, matched_px, 2, (0, 215, 255), -1)
            if match_gap_m > 0.01:
                raw_px = self._world_to_px(raw_x, raw_y)
                cv2.line(canvas, raw_px, matched_px, (0, 215, 255), 1)

            cv2.putText(canvas,
                        f"pos x={x:.2f}m  y={y:.2f}m  yaw_vis={display_yaw_deg:.0f}°"
                        f"  raw={math.degrees(yaw):.0f}°"
                        f"  steer={math.degrees(steer_rad):+.0f}°  [dr x={raw_x:.2f} y={raw_y:.2f}]",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(canvas,
                        f"match x={matched_x:.2f}m  y={matched_y:.2f}m  yaw={math.degrees(matched_yaw):.0f}°"
                        f"  gap={match_gap_m:.3f}m",
                        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(canvas,
                        f"e={state.get('error_m', 0):.3f}m  "
                        f"h={math.degrees(state.get('heading_rad', 0)):.1f}°  "
                        f"mm={map_match_error_m:.3f}m  lat={lat_corr_m:+.3f}m  "
                        f"raw_lat={raw_lat_err_m:+.3f}m  [{mode}|{lane_rel}]",
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(canvas,
                        f"v={spd * 100:.1f}cm/s[{spd_src}]  wp={wp_idx}  tgt={state.get('target_idx', wp_idx)}"
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

        # Stash the rendered canvas for in-process consumers (no imshow).
        # The PyQt5 GUI does its own rendering; this is just a fallback for
        # anyone that wants the raw composite image as a numpy array.
        self._latest_canvas = canvas
