"""Line-following debug + status panel.

Consumes two SocketIO events that the line-following thread publishes:

* ``LineFollowingDebug`` — a debug image (lane mask + steering arrow
  overlay) as base64 PNG or raw JPEG bytes. Decoded to ``QImage`` by
  ``SocketIOClient`` so the widget just sets a pixmap.
* ``LineFollowingStatus`` — a dict with current steering, mode, lane
  confidence, etc. Rendered as a stack of labelled rows.

Layout: image fills the upper half, status table sits below.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..client.zmq_bus_client import ZmqBusClient as SocketIOClient


class _DebugImage(QLabel):
    """Auto-scaling image label used for the debug overlay frame."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image: Optional[QImage] = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#111; color:#888;")
        self.setText("(no debug frame yet)")

    def set_image(self, image: QImage) -> None:
        if image is None or image.isNull():
            return
        self._image = image
        self._repaint()

    def _repaint(self) -> None:
        if self._image is None:
            return
        self.setPixmap(QPixmap.fromImage(self._image).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
        ))

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._repaint()


class _StatusGrid(QGroupBox):
    """Live grid of line-following status fields."""

    # Order + labels we expect to see in the payload — anything else is
    # appended dynamically with its raw key as the label.
    KNOWN_FIELDS = (
        ("mode",            "Mode"),
        ("steer_cmd",       "Steer cmd"),
        ("speed_cmd",       "Speed cmd"),
        ("lane_offset",     "Lane offset"),
        ("lane_confidence", "Confidence"),
        ("fps",             "FPS"),
    )

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Status", parent)
        self.setStyleSheet("QGroupBox { color:#aaa; font-weight:600; }")
        self._labels: dict[str, QLabel] = {}

        grid = QGridLayout(self)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)
        for row, (key, label) in enumerate(self.KNOWN_FIELDS):
            name = QLabel(label)
            name.setStyleSheet("color:#888;")
            value = QLabel("--")
            value.setStyleSheet("color:#eee; font-weight:600;")
            grid.addWidget(name, row, 0)
            grid.addWidget(value, row, 1)
            self._labels[key] = value
        self._grid = grid
        self._next_row = len(self.KNOWN_FIELDS)

    def update_status(self, status) -> None:
        if not isinstance(status, dict):
            return
        for key, value in status.items():
            label = self._labels.get(key)
            if label is None:
                # Unknown key — add a row dynamically so we don't lose data.
                from PyQt5.QtWidgets import QLabel as _QL
                name = _QL(key)
                name.setStyleSheet("color:#888;")
                label = _QL("--")
                label.setStyleSheet("color:#eee; font-weight:600;")
                self._grid.addWidget(name, self._next_row, 0)
                self._grid.addWidget(label, self._next_row, 1)
                self._labels[key] = label
                self._next_row += 1
            if isinstance(value, float):
                label.setText(f"{value:.3f}")
            else:
                label.setText(str(value))


class LineFollowingPanel(QWidget):
    """Debug image + live status, side-by-side via splitter."""

    def __init__(self, client: SocketIOClient, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image = _DebugImage()
        self._status = _StatusGrid()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._image)
        splitter.addWidget(self._status)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(splitter)

        client.line_following_debug_signal.connect(self._image.set_image)
        client.line_following_status_signal.connect(self._status.update_status)
