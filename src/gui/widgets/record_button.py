"""Record toggle.

A single checkable button that mirrors the backend ``Recording`` state.
Click → emits ``Recording`` with the new boolean. Backend confirms by
emitting ``Recording`` back, which we use to re-sync the visual state
(in case the recorder failed to start, the button bounces back to off).
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QPushButton, QWidget

from ..client import events as ev
from ..client.zmq_bus_client import ZmqBusClient as SocketIOClient


class RecordButton(QPushButton):
    """Toggle button for the rosbag/dataset recorder."""

    state_changed = pyqtSignal(bool)

    def __init__(self, client: SocketIOClient, parent: Optional[QWidget] = None):
        super().__init__("● Record", parent)
        self._client = client
        self.setCheckable(True)
        self.setMinimumHeight(38)
        self._apply_style(False)

        self.toggled.connect(self._on_toggled)
        self._client.recording_signal.connect(self._on_backend_state)

    def _apply_style(self, recording: bool) -> None:
        if recording:
            self.setText("■ Stop recording")
            self.setStyleSheet(
                "QPushButton { background:#c33; color:white; "
                "border:1px solid #c33; padding:8px; font-weight:700; }"
            )
        else:
            self.setText("● Record")
            self.setStyleSheet(
                "QPushButton { background:#222; color:#ccc; "
                "border:1px solid #333; padding:8px; font-weight:700; }"
                "QPushButton:hover { color:white; }"
            )

    def _on_toggled(self, checked: bool) -> None:
        self._apply_style(checked)
        self._client.emit_message(ev.CMD_RECORDING, checked)
        self.state_changed.emit(checked)

    def _on_backend_state(self, recording: bool) -> None:
        # Avoid signal-loop: only resync if the backend disagrees with us.
        if recording == self.isChecked():
            return
        self.blockSignals(True)
        try:
            self.setChecked(recording)
            self._apply_style(recording)
        finally:
            self.blockSignals(False)
