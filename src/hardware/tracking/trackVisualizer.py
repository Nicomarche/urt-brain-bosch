"""
trackVisualizer — real-time OpenCV map window.

Runs at ~10 Hz in its own thread.  Reads the latest TrackingState snapshot
and draws:
  • Track edges (white lines)
  • Nodes coloured by attribute:
      NORMAL        → green
      STOPLINE      → red
      INTERSECTION  → blue
      HIGHWAY_LEFT  → yellow
      HIGHWAY_RIGHT → orange
      CROSSWALK     → magenta
  • Current target waypoint (cyan circle)
  • Car pose (white triangle pointing in heading direction)

The window can be toggled on/off via TRACKING_SHOW_WINDOW in config.py.
"""

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
    ATTR_NORMAL:       (100, 220, 100),   # green
    ATTR_CROSSWALK:    (220, 100, 220),   # magenta
    ATTR_INTERSECTION: (220, 100, 100),   # blue
    ATTR_ONEWAY:       (200, 200, 200),   # light grey
    ATTR_HIGHWAY_LEFT: (50, 220, 220),    # yellow
    ATTR_HIGHWAY_RIGHT:(50, 165, 255),    # orange
    ATTR_ROUNDABOUT:   (255, 200, 50),    # teal-ish
    ATTR_STOPLINE:     (50, 50, 220),     # red
}
_DEFAULT_COLOR = (180, 180, 180)

_CANVAS_SIZE = 700          # pixels (square canvas)
_MARGIN_PX = 40             # margin inside canvas
_EDGE_COLOR = (60, 60, 60)  # dark grey
_WP_COLOR = (255, 220, 50)  # cyan waypoint
_CAR_COLOR = (255, 255, 255)
_FPS = 10


class TrackVisualizer(threading.Thread):
    """OpenCV visualisation thread.

    Args:
        graph:          TrackGraph instance (for static geometry).
        window_name:    cv2 window title.
        canvas_size:    Window size in pixels (square).
    """

    def __init__(self, graph: TrackGraph,
                 window_name: str = "Track Navigation",
                 canvas_size: int = _CANVAS_SIZE):
        super().__init__(daemon=True)
        self._graph = graph
        self._window_name = window_name
        self._canvas_size = canvas_size
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._state = None           # latest TrackingState snapshot dict
        self._base_canvas = None     # pre-rendered static background
        self._scale = 1.0
        self._ox = 0.0
        self._oy = 0.0
        self._prerender()

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
    # Rendering helpers
    # ------------------------------------------------------------------
    def _world_to_px(self, x: float, y: float):
        """Convert world metres → integer pixel coordinates."""
        px = int(self._ox + x * self._scale)
        py = int(self._canvas_size - self._margin - y * self._scale)
        return px, py

    @property
    def _margin(self):
        return _MARGIN_PX

    def _prerender(self) -> None:
        """Build a static background canvas with edges and nodes."""
        if not _CV2_OK:
            return
        graph = self._graph
        canvas_size = self._canvas_size
        margin = _MARGIN_PX

        # Compute bounding box
        if len(graph.ordered_nodes) == 0:
            self._base_canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
            return

        xs = [n.x for n in graph.ordered_nodes]
        ys = [n.y for n in graph.ordered_nodes]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w = max(x_max - x_min, 0.01)
        h = max(y_max - y_min, 0.01)
        drawable = canvas_size - 2 * margin
        self._scale = min(drawable / w, drawable / h)
        self._ox = margin - x_min * self._scale
        self._oy_base = margin - y_min * self._scale  # stored for y inversion

        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

        # Draw edges along the spline path
        wps = graph.waypoints
        if len(wps) > 1:
            pts = np.array([
                [int(self._ox + wps[i, 0] * self._scale),
                 int(canvas_size - margin - (wps[i, 1] - y_min) * self._scale)]
                for i in range(0, len(wps), max(1, len(wps) // 500))
            ], dtype=np.int32)
            for i in range(len(pts) - 1):
                cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), _EDGE_COLOR, 1)

        # Draw nodes
        for node in graph.ordered_nodes:
            px = int(self._ox + node.x * self._scale)
            py = int(canvas_size - margin - (node.y - y_min) * self._scale)
            color = _NODE_COLORS.get(node.attribute, _DEFAULT_COLOR)
            cv2.circle(canvas, (px, py), 5, color, -1)

        # Store y_min for dynamic rendering
        self._y_min = y_min
        self._base_canvas = canvas

    def _draw_frame(self) -> None:
        """Render one frame and display it."""
        if self._base_canvas is None:
            return
        canvas = self._base_canvas.copy()
        canvas_size = self._canvas_size
        margin = _MARGIN_PX

        with self._state_lock:
            state = dict(self._state) if self._state is not None else None

        if state is not None:
            x = state.get("x", 0.0)
            y = state.get("y", 0.0)
            yaw = state.get("yaw", 0.0)
            wp_idx = state.get("wp_idx", 0)

            # Current target waypoint (cyan circle)
            wps = self._graph.waypoints
            if len(wps) > 0 and wp_idx < len(wps):
                wx, wy = wps[wp_idx % len(wps), :2]
                wpx = int(self._ox + wx * self._scale)
                wpy = int(canvas_size - margin - (wy - self._y_min) * self._scale)
                cv2.circle(canvas, (wpx, wpy), 7, _WP_COLOR, 2)

            # Car: small triangle pointing in yaw direction
            cx = int(self._ox + x * self._scale)
            cy = int(canvas_size - margin - (y - self._y_min) * self._scale)
            car_len = 14
            car_w = 7
            # Triangle tip
            tip = (
                int(cx + car_len * math.cos(yaw)),
                int(cy - car_len * math.sin(yaw)),
            )
            left = (
                int(cx + car_w * math.cos(yaw + math.pi * 0.7)),
                int(cy - car_w * math.sin(yaw + math.pi * 0.7)),
            )
            right = (
                int(cx + car_w * math.cos(yaw - math.pi * 0.7)),
                int(cy - car_w * math.sin(yaw - math.pi * 0.7)),
            )
            pts = np.array([tip, left, right], dtype=np.int32)
            cv2.fillPoly(canvas, [pts], _CAR_COLOR)
            cv2.polylines(canvas, [pts], isClosed=True, color=(100, 200, 255), thickness=1)

            # Telemetry overlay
            mode_label = "WP MODE" if state.get("waypoint_mode_active") else "VISUAL"
            cv2.putText(canvas,
                        f"x={x:.2f}m  y={y:.2f}m  yaw={math.degrees(yaw):.0f}°",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(canvas,
                        f"e={state.get('error_m', 0):.3f}m  h={math.degrees(state.get('heading_rad', 0)):.1f}°  [{mode_label}]",
                        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            spd = state.get("speed_mps", 0.0)
            cv2.putText(canvas,
                        f"v={spd * 100:.1f}cm/s  wp={wp_idx}",
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(self._window_name, canvas)
