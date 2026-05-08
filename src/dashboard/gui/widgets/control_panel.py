"""Control panel — KL switch + driving mode + record + speed/steer readouts.

This is the top-of-screen control strip on the Driving tab. It owns one
``DrivingModel`` (the source of truth for manual speed/steer state) and
one ``KeyboardHandler``; the driving widgets (``StateSwitch``,
``KlSwitch``) hang off them.

The panel exposes:

* ``model`` — the shared ``DrivingModel`` for any other widget that
  wants to read/write driving state without going through the UI.
* ``install_keyboard_on(target)`` — installs the W/A/S/D/space filter on
  the given object (use the MainWindow so it works regardless of focus).
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..client.socketio_client import SocketIOClient
from .keyboard_handler import KeyboardHandler
from .kl_switch import KlSwitch
from .record_button import RecordButton
from .state_switch import DrivingModel, StateSwitch


class _ReadoutBox(QFrame):
    """Small framed numeric readout (Speed / Steer)."""

    def __init__(self, label: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame { background:#1a1a1a; border:1px solid #333; border-radius:6px; }"
        )
        self._label = QLabel(label)
        self._label.setStyleSheet("color:#888; font-size:9pt;")
        self._value = QLabel("0")
        self._value.setStyleSheet("color:#eee; font-size:18pt; font-weight:600;")
        self._value.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.addWidget(self._label)
        layout.addWidget(self._value)

    def set_value(self, value: int) -> None:
        self._value.setText(str(value))


class ControlPanel(QWidget):
    """Driving controls (KL + mode + record) + live speed/steer readouts."""

    def __init__(self, client: SocketIOClient, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._client = client
        self._model = DrivingModel(client, parent=self)
        self._keyboard = KeyboardHandler(self._model, parent=self)

        self._kl = KlSwitch(client, parent=self)
        self._state = StateSwitch(self._model, parent=self)
        self._record = RecordButton(client, parent=self)

        # KL switch → DrivingModel (so the model can gate manual driving on KL=30)
        self._kl.kl_changed.connect(self._model.set_kl)

        # Live readouts. Speed comes from our own model (immediate); steer
        # likewise. Backend telemetry (CurrentSpeed/CurrentSteer) is shown
        # by the speedometer/steering widgets in the cluster panel.
        self._speed_box = _ReadoutBox("Cmd Speed")
        self._steer_box = _ReadoutBox("Cmd Steer")
        self._model.speed_changed.connect(self._speed_box.set_value)
        self._model.steer_changed.connect(self._steer_box.set_value)

        # Lay out: [KL] [State buttons] [Record] [Speed] [Steer]
        row = QHBoxLayout(self)
        row.setSpacing(12)
        row.addWidget(self._kl)
        row.addWidget(self._state, 2)
        row.addWidget(self._record)
        row.addWidget(self._speed_box)
        row.addWidget(self._steer_box)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

    # ------------------------------------------------------------------
    @property
    def model(self) -> DrivingModel:
        return self._model

    def install_keyboard_on(self, target) -> None:
        self._keyboard.install(target)
