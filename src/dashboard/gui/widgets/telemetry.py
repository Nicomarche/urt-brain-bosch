"""Telemetry widgets — speedometer, steering indicator, battery, consumption,
hardware (CPU/RAM), warning light, side markers.

All of these are tiny consumers of a single ``SocketIOClient`` signal. To
avoid scattering boilerplate across 7 separate files, they live together
here. Each is a self-contained ``QWidget`` you can drop into any layout.

Visualisation choices:

* **Gauges** — ``QGraphicsScene`` with a coloured arc + a needle that
  rotates via ``QGraphicsItem.setRotation()``. Lighter than pyqtgraph for
  these one-shot dials and avoids the dependency on the Driving tab.
* **Battery / instant consumption** — plain ``QLabel`` with a styled
  background. Fast and obvious.
* **Hardware (CPU usage, CPU temp, RAM)** — three small gauges in a row,
  reusing the same ``_ArcGauge`` class.
* **Warning light** — a coloured circle that turns on when a non-empty
  warning string arrives. Could be replaced with the legacy PNG sprites
  later (see ``assets/warning_lights/``); for now we just colour-code.

Cross-cutting:

* All widgets are **passive**: they only listen to backend signals; they
  never emit anything.
* Each widget keeps a sensible visual "no data yet" state so the UI
  doesn't look broken before the first telemetry tick.
"""

from __future__ import annotations

import math
from typing import Optional

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt5.QtWidgets import (
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..client.socketio_client import SocketIOClient


# ----------------------------------------------------------------------
# Helper: a half-circle arc gauge with a needle.
# ----------------------------------------------------------------------
class _ArcGauge(QGraphicsView):
    """Half-circle gauge from ``min_value`` to ``max_value`` (units in label).

    Used by the speedometer, steering indicator and the three hardware
    gauges. Override ``set_value`` to push a new reading; the needle
    rotates linearly between the two extremes.
    """

    def __init__(
        self,
        min_value: float,
        max_value: float,
        unit: str = "",
        size: int = 200,
        arc_color: str = "#5cb85c",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._min = float(min_value)
        self._max = float(max_value)
        self._unit = unit
        self._value: float = 0.0 if self._min <= 0.0 <= self._max else self._min
        self._has_value = False

        self.setRenderHint(QPainter.Antialiasing)
        self.setStyleSheet("background:#111; border:none;")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedSize(size, int(size * 0.6))
        self._size = size

        scene = QGraphicsScene(self)
        self.setScene(scene)
        self._scene = scene
        self._arc_color = QColor(arc_color)
        self._build_scene()

    # ------------------------------------------------------------------
    def _build_scene(self) -> None:
        self._scene.clear()
        s = self._size
        # Background arc (full track).
        track = QPainterPath()
        track.arcMoveTo(QRectF(10, 10, s - 20, s - 20), 180)
        track.arcTo(QRectF(10, 10, s - 20, s - 20), 180, -180)
        track_item = self._scene.addPath(track, QPen(QColor("#333"), 12))
        track_item.setZValue(0)

        # Filled arc (value).
        self._fill = self._scene.addPath(QPainterPath(), QPen(self._arc_color, 12))
        self._fill.setZValue(1)

        # Needle.
        path = QPainterPath()
        path.moveTo(0, 0)
        path.lineTo(0, -(s / 2 - 18))
        self._needle = self._scene.addPath(
            path, QPen(QColor("#eee"), 3, Qt.SolidLine, Qt.RoundCap),
        )
        self._needle.setPos(s / 2, s / 2)
        self._needle.setZValue(2)

        # Centre label.
        self._label = self._scene.addText("--")
        self._label.setDefaultTextColor(QColor("#ddd"))
        font = self._label.font()
        font.setPointSize(int(s / 16))
        font.setBold(True)
        self._label.setFont(font)
        self._reposition_label()

        self.setSceneRect(0, 0, s, s * 0.6)
        self._update_visuals()

    def _reposition_label(self) -> None:
        rect = self._label.boundingRect()
        s = self._size
        self._label.setPos(s / 2 - rect.width() / 2, s / 2 - rect.height() - 4)

    # ------------------------------------------------------------------
    def set_value(self, value: float) -> None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if v < self._min:
            v = self._min
        if v > self._max:
            v = self._max
        self._value = v
        self._has_value = True
        self._update_visuals()

    def _update_visuals(self) -> None:
        s = self._size
        span = self._max - self._min
        if span <= 0:
            return
        frac = (self._value - self._min) / span
        # Needle: -90° at min, +90° at max.
        angle = -90 + frac * 180
        self._needle.setRotation(angle)

        # Filled arc from 180° going clockwise by ``frac * 180`` degrees.
        path = QPainterPath()
        path.arcMoveTo(QRectF(10, 10, s - 20, s - 20), 180)
        path.arcTo(QRectF(10, 10, s - 20, s - 20), 180, -frac * 180)
        self._fill.setPath(path)

        text = (
            f"{self._value:.0f}{(' ' + self._unit) if self._unit else ''}"
            if self._has_value
            else "--"
        )
        self._label.setPlainText(text)
        self._reposition_label()


# ----------------------------------------------------------------------
# Speedometer — current speed in cm/s (or whatever the brain reports).
# ----------------------------------------------------------------------
class Speedometer(_ArcGauge):
    def __init__(self, client: SocketIOClient, parent: Optional[QWidget] = None):
        super().__init__(min_value=-50, max_value=50, unit="cm/s",
                         size=220, arc_color="#5cb85c", parent=parent)
        client.current_speed_signal.connect(self.set_value)


# ----------------------------------------------------------------------
# Steering indicator — current steer angle in degrees.
# ----------------------------------------------------------------------
class SteeringIndicator(_ArcGauge):
    def __init__(self, client: SocketIOClient, parent: Optional[QWidget] = None):
        super().__init__(min_value=-25, max_value=25, unit="°",
                         size=220, arc_color="#2b8fd1", parent=parent)
        client.current_steer_signal.connect(self.set_value)
        client.steering_limits_signal.connect(self._on_limits)

    def _on_limits(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        try:
            upper = float(payload.get("upperLimit", 250)) / 10.0
            lower = float(payload.get("lowerLimit", -250)) / 10.0
        except (TypeError, ValueError):
            return
        self._min = lower
        self._max = upper
        self._build_scene()


# ----------------------------------------------------------------------
# Numeric readouts — battery, instant consumption.
# ----------------------------------------------------------------------
class _NumericReadout(QFrame):
    """Reusable label/value frame with color thresholding."""

    def __init__(self, title: str, unit: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._unit = unit
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame { background:#1a1a1a; border:1px solid #333; border-radius:6px; }"
        )
        self._title = QLabel(title)
        self._title.setStyleSheet("color:#888; font-size:9pt;")
        self._value = QLabel("--")
        self._value.setStyleSheet("color:#eee; font-size:18pt; font-weight:600;")
        self._value.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.addWidget(self._title)
        layout.addWidget(self._value)

    def set_value(self, value: float, color: Optional[str] = None) -> None:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return
        self._value.setText(f"{f:.1f} {self._unit}".strip())
        if color:
            self._value.setStyleSheet(
                f"color:{color}; font-size:18pt; font-weight:600;"
            )


class BatteryIndicator(_NumericReadout):
    def __init__(self, client: SocketIOClient, parent: Optional[QWidget] = None):
        super().__init__("Battery", "%", parent=parent)
        client.battery_signal.connect(self._on_battery)

    def _on_battery(self, percent: float) -> None:
        if percent >= 50:
            color = "#5cb85c"
        elif percent >= 20:
            color = "#f0ad4e"
        else:
            color = "#d9534f"
        self.set_value(percent, color)


class InstantConsumption(_NumericReadout):
    def __init__(self, client: SocketIOClient, parent: Optional[QWidget] = None):
        super().__init__("Power", "W", parent=parent)
        client.instant_consumption_signal.connect(self.set_value)


# ----------------------------------------------------------------------
# Hardware data — CPU usage / CPU temp / RAM, three small gauges.
# ----------------------------------------------------------------------
class HardwareData(QWidget):
    """Three gauges in a row: CPU%, CPU°C, RAM%."""

    def __init__(self, client: SocketIOClient, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._cpu = _ArcGauge(0, 100, "%", size=160, arc_color="#2b8fd1")
        self._temp = _ArcGauge(0, 100, "°C", size=160, arc_color="#f0ad4e")
        self._mem = _ArcGauge(0, 100, "%", size=160, arc_color="#5cb85c")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._labeled(self._cpu, "CPU"))
        layout.addWidget(self._labeled(self._temp, "Temp"))
        layout.addWidget(self._labeled(self._mem, "RAM"))

        client.cpu_signal.connect(self._on_cpu)
        client.memory_signal.connect(self._mem.set_value)

    @staticmethod
    def _labeled(widget: QWidget, label: str) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel(label)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color:#aaa; font-weight:600;")
        layout.addWidget(title)
        layout.addWidget(widget)
        return wrap

    def _on_cpu(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        try:
            self._cpu.set_value(float(payload.get("usage", 0)))
            temp = payload.get("temp")
            if temp is not None:
                self._temp.set_value(float(temp))
        except (TypeError, ValueError):
            pass


# ----------------------------------------------------------------------
# Warning light — coloured indicator driven by ``WarningSignal`` strings.
# ----------------------------------------------------------------------
class WarningLight(QLabel):
    """Coloured circle that lights up when a warning is active.

    The ``WarningSignal`` event from the brain carries a textual code
    (e.g. ``"low_battery"``, ``"emergency_stop"``); for now we just map
    "any non-empty string" → orange "WARN" and the empty string → off.
    Mapping to the legacy PNG icons in ``assets/warning_lights/`` can be
    added later without changing the contract.
    """

    def __init__(self, client: SocketIOClient, parent: Optional[QWidget] = None):
        super().__init__("●", parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(48, 48)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._set_state("")
        client.warning_signal_signal.connect(self._set_state)

    def _set_state(self, code: str) -> None:
        if code:
            self.setToolTip(code)
            self.setStyleSheet(
                "QLabel { color:#fa3; font-size:24pt; "
                "background:#3a2a00; border-radius:24px; border:1px solid #fa3; }"
            )
        else:
            self.setToolTip("")
            self.setStyleSheet(
                "QLabel { color:#444; font-size:24pt; "
                "background:#222; border-radius:24px; border:1px solid #333; }"
            )


# ----------------------------------------------------------------------
# Side markers — left/right blinker indicators.
# ----------------------------------------------------------------------
class SideMarker(QLabel):
    """Single L or R blinker. Listens for an external setter (cluster
    composes both)."""

    def __init__(self, side: str, parent: Optional[QWidget] = None):
        assert side in {"L", "R"}
        super().__init__(side, parent)
        self._side = side
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(36, 36)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.set_active(False)

    def set_active(self, on: bool) -> None:
        if on:
            self.setStyleSheet(
                "QLabel { color:#fff; background:#5cb85c; "
                "border-radius:18px; font-weight:700; font-size:14pt; }"
            )
        else:
            self.setStyleSheet(
                "QLabel { color:#555; background:#222; "
                "border:1px solid #333; border-radius:18px; "
                "font-weight:700; font-size:14pt; }"
            )


# ----------------------------------------------------------------------
# Cluster — the right-hand telemetry strip used by the Telemetry tab.
# ----------------------------------------------------------------------
class TelemetryCluster(QWidget):
    """Vertical strip combining all telemetry widgets — drop into any tab."""

    def __init__(
        self,
        client: SocketIOClient,
        parent: Optional[QWidget] = None,
        *,
        footer_widget: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        gauges_row = QHBoxLayout()
        gauges_row.setSpacing(12)
        gauges_row.addWidget(Speedometer(client))
        gauges_row.addWidget(SteeringIndicator(client))
        gauges_row.addWidget(WarningLight(client))

        readouts_row = QHBoxLayout()
        readouts_row.setSpacing(8)
        readouts_row.addWidget(BatteryIndicator(client))
        readouts_row.addWidget(InstantConsumption(client))

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.addLayout(gauges_row)
        layout.addLayout(readouts_row)
        layout.addWidget(HardwareData(client))
        if footer_widget is not None:
            layout.addWidget(footer_widget, 1)
        else:
            # Lazy import so optional plotting deps are only touched when used.
            from .time_speed_steer_chart import TimeSpeedSteerChart
            layout.addWidget(TimeSpeedSteerChart(client), 1)
