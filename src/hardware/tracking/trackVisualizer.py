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
  • Car pose (white triangle pointing in heading direction)

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
    # Dynamic frame
    # ------------------------------------------------------------------
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

            # The arrow always points where the front wheels face:
            #   display_yaw = car body heading - steering angle
            # Minus because in screen coords (Y-axis flipped), CW rotation = decreasing angle.
            # Steer > 0 = right turn = CW on screen → display_yaw must decrease.
            # This updates even when the car is stopped and only the steering changes.
            display_yaw = yaw - steer_rad

            # Current target waypoint
            wps = self._graph.waypoints
            if len(wps) > 0 and wp_idx < len(wps):
                wx, wy   = wps[wp_idx % len(wps), :2]
                wpx, wpy = self._world_to_px(float(wx), float(wy))
                cv2.circle(canvas, (wpx, wpy), 7, _WP_COLOR, 2)

            # Car triangle — tip points toward display_yaw (wheel direction)
            cx, cy   = self._world_to_px(x, y)
            car_len  = 14
            car_w    = 7
            tip   = (int(cx + car_len * math.cos(display_yaw)),
                     int(cy - car_len * math.sin(display_yaw)))
            left  = (int(cx + car_w * math.cos(display_yaw + math.pi * 0.7)),
                     int(cy - car_w * math.sin(display_yaw + math.pi * 0.7)))
            right = (int(cx + car_w * math.cos(display_yaw - math.pi * 0.7)),
                     int(cy - car_w * math.sin(display_yaw - math.pi * 0.7)))
            pts = np.array([tip, left, right], dtype=np.int32)
            cv2.fillPoly(canvas, [pts], _CAR_COLOR)
            cv2.polylines(canvas, [pts], isClosed=True,
                          color=(100, 200, 255), thickness=1)

            # Telemetry text
            mode = "WP MODE" if state.get("waypoint_mode_active") else "VISUAL"
            spd  = state.get("speed_mps", 0.0)
            cv2.putText(canvas,
                        f"x={x:.2f}m  y={y:.2f}m  yaw={math.degrees(yaw):.0f}°"
                        f"  steer={math.degrees(steer_rad):+.0f}°",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(canvas,
                        f"e={state.get('error_m', 0):.3f}m  "
                        f"h={math.degrees(state.get('heading_rad', 0)):.1f}°  [{mode}]",
                        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(canvas,
                        f"v={spd * 100:.1f}cm/s  wp={wp_idx}",
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        cv2.imshow(self._window_name, canvas)
