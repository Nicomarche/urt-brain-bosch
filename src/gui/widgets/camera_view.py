"""Live camera widget.

Receives ``QImage`` frames from ``SocketIOClient.camera_frame_signal`` and
renders them as a single ``QLabel``-backed display. The decode is already
done in the client (JPEG bytes → ``QImage``) so this widget just needs to
scale the image to its current viewport and avoid keeping more than one
buffer alive at a time.

Design notes:

* No internal QTimer. Repaint is event-driven: a new signal arrives, we
  drop the old pixmap, scale, ``setPixmap``. Qt repaints when needed.
* ``Qt.KeepAspectRatio`` + ``Qt.SmoothTransformation`` so resizing the
  window doesn't distort the frame.
* Optional FPS counter in the top-left corner — useful for confirming
  the binary stream actually beats the legacy base64 path.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy, QWidget

from ..client.zmq_bus_client import ZmqBusClient as SocketIOClient


class CameraView(QLabel):
    """A self-sizing camera display fed by the SocketIO client."""

    PLACEHOLDER_TEXT = "Sin señal de cámara"
    FPS_HISTORY = 30  # frames kept for the rolling FPS estimate

    # Steering indicator constants
    _STEER_MAX_DEG  = 25.0   # mechanical limit (matches AcadosMPC constraint)
    _NEEDLE_LEN     = 55     # pixels
    _BASE_HALF      = 44     # half-width of the horizontal reference line
    _INDICATOR_Y    = 52     # px from bottom edge to baseline

    def __init__(self, client: SocketIOClient, show_fps: bool = True,
                 parent: Optional[QWidget] = None,
                 video_source=None):
        """
        Args:
            client: telemetry/command client (provides ``motor_command_signal``).
            video_source: anything exposing ``camera_frame_signal``. Defaults
                to ``client`` for the legacy SocketIO path; pass a
                :class:`UdpVideoClient` to consume the new UDP stream.
        """
        super().__init__(parent)
        self._client = client
        self._video_source = video_source if video_source is not None else client
        self._show_fps = show_fps
        self._last_image: Optional[QImage] = None
        self._frame_times: deque[float] = deque(maxlen=self.FPS_HISTORY)

        # MPC command state
        self._mpc_steer_deg: float = 0.0
        self._mpc_source: str = ""
        self._mpc_reason: str = ""

        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #111; color: #888;")
        self.setText(self.PLACEHOLDER_TEXT)

        self._video_source.camera_frame_signal.connect(self._on_frame)
        self._client.motor_command_signal.connect(self._on_motor_command)

    # ------------------------------------------------------------------
    # Frame handling
    # ------------------------------------------------------------------
    def _on_frame(self, image: QImage) -> None:
        if image is None or image.isNull():
            return
        self._last_image = image
        self._frame_times.append(time.monotonic())
        self._repaint_current()

    def _repaint_current(self) -> None:
        if self._last_image is None:
            return
        # Build a pixmap scaled to the current viewport. Recomputing on every
        # frame keeps the frame in sync with resizes; the cost is one O(WxH)
        # bilinear pass which is negligible for 640×480 inputs.
        pix = QPixmap.fromImage(self._last_image).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.setPixmap(pix)

    def resizeEvent(self, event):  # noqa: N802 - Qt convention
        super().resizeEvent(event)
        self._repaint_current()

    # ------------------------------------------------------------------
    # MPC command
    # ------------------------------------------------------------------
    def _on_motor_command(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        try:
            self._mpc_steer_deg = float(payload.get("steering_deg", 0.0))
        except (TypeError, ValueError):
            return
        self._mpc_source = str(payload.get("source", ""))
        self._mpc_reason = str(payload.get("reason", ""))
        self.update()  # trigger repaint for the steering overlay

    # ------------------------------------------------------------------
    # Optional FPS + MPC steering overlay
    # ------------------------------------------------------------------
    def paintEvent(self, event):  # noqa: N802 - Qt convention
        super().paintEvent(event)
        painter = QPainter(self)
        try:
            # FPS counter (top-left)
            if self._show_fps and len(self._frame_times) >= 2:
                oldest = self._frame_times[0]
                newest = self._frame_times[-1]
                dt = newest - oldest
                if dt > 0:
                    fps = (len(self._frame_times) - 1) / dt
                    painter.setPen(Qt.green)
                    painter.drawText(8, 18, f"{fps:5.1f} FPS")

            # MPC steering indicator (bottom-center)
            self._draw_steering_indicator(painter)
        finally:
            painter.end()

    def _draw_steering_indicator(self, painter: QPainter) -> None:
        """Draw a needle-and-baseline steering indicator for the MPC command."""
        # Color: green = MPC activo, naranja = safety_gate fallback
        if self._mpc_source == "motion_controller":
            needle_color = QColor("#00e676")   # green
            label_color  = QColor("#00e676")
        elif self._mpc_source:
            needle_color = QColor("#ffaa00")   # orange = safety_gate
            label_color  = QColor("#ffaa00")
        else:
            return  # no data yet — don't draw

        cx = self.width() // 2
        cy = self.height() - self._INDICATOR_Y

        # Semi-transparent background pill
        bg_w, bg_h = self._BASE_HALF * 2 + 20, self._NEEDLE_LEN + 28
        painter.setBrush(QBrush(QColor(0, 0, 0, 140)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(cx - bg_w // 2, cy - self._NEEDLE_LEN - 8,
                                bg_w, bg_h, 6, 6)

        # Horizontal base line (represents 0° / straight ahead)
        painter.setPen(QPen(QColor(180, 180, 180, 200), 2))
        painter.drawLine(cx - self._BASE_HALF, cy, cx + self._BASE_HALF, cy)
        # Centre tick
        painter.drawLine(cx, cy, cx, cy - 8)

        # Needle: points up (90°) when steer=0; tilts left for positive steer
        # Map ±25° wheel → ±70° visual for easy reading
        visual_deg = max(-70.0, min(70.0,
                         self._mpc_steer_deg * (70.0 / self._STEER_MAX_DEG)))
        angle_rad = math.radians(90.0 + visual_deg)
        nx = int(cx + self._NEEDLE_LEN * math.cos(angle_rad))
        ny = int(cy - self._NEEDLE_LEN * math.sin(angle_rad))
        painter.setPen(QPen(needle_color, 3, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(cx, cy, nx, ny)

        # Arrowhead on needle tip
        head_len = 8
        perp_rad = angle_rad + math.pi * 0.75
        ax1 = int(nx + head_len * math.cos(perp_rad))
        ay1 = int(ny - head_len * math.sin(perp_rad))
        perp_rad2 = angle_rad - math.pi * 0.75
        ax2 = int(nx + head_len * math.cos(perp_rad2))
        ay2 = int(ny - head_len * math.sin(perp_rad2))
        painter.drawLine(nx, ny, ax1, ay1)
        painter.drawLine(nx, ny, ax2, ay2)

        # Text label: angle value + source tag
        src_tag = "MPC" if self._mpc_source == "motion_controller" else "GATE"
        line1 = f"{src_tag}  {self._mpc_steer_deg:+.1f}°"
        painter.setPen(label_color)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(line1)
        painter.drawText(cx - tw // 2, cy + 18, line1)

        # Reason (only shown for safety_gate so the user sees why it's blocking)
        if self._mpc_source != "motion_controller" and self._mpc_reason:
            painter.setPen(QColor(200, 200, 200, 180))
            tw2 = fm.horizontalAdvance(self._mpc_reason)
            painter.drawText(cx - tw2 // 2, cy + 34, self._mpc_reason)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt convention
        return QSize(640, 480)
