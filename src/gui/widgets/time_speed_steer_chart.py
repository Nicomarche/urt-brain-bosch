"""Rolling 30-second speed + steer chart.

Uses ``pyqtgraph`` because Qt's own ``QChart`` is heavy and slow for
streaming data — pyqtgraph is the de-facto choice for real-time plots
in Qt apps and ships in the dependency list.

Two traces share an X axis (time, in seconds since the chart was
created). Both Y axes are independent because speed (cm/s) and steer
(degrees) have different ranges and meaning.

The buffers are circular ``deque``s of fixed length so memory use is
bounded; we never trim the data on the GUI thread.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QVBoxLayout, QWidget

try:
    import pyqtgraph as pg
    _HAS_PG = True
except ImportError:
    _HAS_PG = False

from ..client.zmq_bus_client import ZmqBusClient as SocketIOClient


class TimeSpeedSteerChart(QWidget):
    """30-second rolling chart of CurrentSpeed and CurrentSteer.

    If ``pyqtgraph`` isn't installed the widget renders a placeholder
    label rather than crashing — useful when running monitor mode on a
    minimal Python install.
    """

    WINDOW_SECONDS = 30.0
    REFRESH_HZ = 20  # 50 ms redraws are smoother than relying on signal rate

    def __init__(self, client: SocketIOClient, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not _HAS_PG:
            from PyQt5.QtWidgets import QLabel
            label = QLabel("pyqtgraph not installed — install requirements.txt to "
                           "enable the speed/steer chart.")
            label.setStyleSheet("color:#888; padding:24px;")
            layout.addWidget(label)
            return

        # Buffers (timestamp + value). 30 s × 50 Hz = 1500 entries max.
        self._t0 = time.monotonic()
        max_points = int(self.WINDOW_SECONDS * 50) + 60
        self._t_buf = deque(maxlen=max_points)
        self._speed_buf = deque(maxlen=max_points)
        self._steer_buf = deque(maxlen=max_points)
        self._latest_speed = 0
        self._latest_steer = 0

        # Configure the plot widget.
        pg.setConfigOptions(antialias=True, background="#111", foreground="#ccc")
        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.setLabel("left", "Speed", units="cm/s")
        self._plot.setLabel("right", "Steer", units="°")
        self._plot.setXRange(-self.WINDOW_SECONDS, 0)
        layout.addWidget(self._plot)

        # Two curves. Steer goes on a secondary ViewBox to get its own y-axis.
        self._speed_curve = self._plot.plot(
            pen=pg.mkPen("#5cb85c", width=2), name="Speed",
        )
        self._steer_view = pg.ViewBox()
        self._plot.scene().addItem(self._steer_view)
        self._plot.getAxis("right").linkToView(self._steer_view)
        self._steer_view.setXLink(self._plot)
        self._steer_view.setYRange(-30, 30)
        self._plot.getViewBox().sigResized.connect(
            lambda: self._steer_view.setGeometry(self._plot.getViewBox().sceneBoundingRect())
        )
        self._steer_curve = pg.PlotDataItem(pen=pg.mkPen("#2b8fd1", width=2))
        self._steer_view.addItem(self._steer_curve)

        # Hook signals.
        client.current_speed_signal.connect(self._on_speed)
        client.current_steer_signal.connect(self._on_steer)

        self._refresh = QTimer(self)
        self._refresh.setInterval(int(1000 / self.REFRESH_HZ))
        self._refresh.timeout.connect(self._tick)
        self._refresh.start()

    # ------------------------------------------------------------------
    def _on_speed(self, value: int) -> None:
        self._latest_speed = int(value)

    def _on_steer(self, value: int) -> None:
        self._latest_steer = int(value)

    def _tick(self) -> None:
        now = time.monotonic() - self._t0
        self._t_buf.append(now)
        self._speed_buf.append(self._latest_speed)
        self._steer_buf.append(self._latest_steer)

        if not self._t_buf:
            return

        # Drop everything older than the rolling window for a stable x-axis.
        cutoff = now - self.WINDOW_SECONDS
        while self._t_buf and self._t_buf[0] < cutoff:
            self._t_buf.popleft()
            self._speed_buf.popleft()
            self._steer_buf.popleft()

        # x relative to "now" so the chart slides left.
        ts = [t - now for t in self._t_buf]
        self._speed_curve.setData(ts, list(self._speed_buf))
        self._steer_curve.setData(ts, list(self._steer_buf))
        self._plot.setXRange(-self.WINDOW_SECONDS, 0, padding=0)
