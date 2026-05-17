"""KL switch (key-position selector).

Three positions, mirroring a real ignition: ``0`` (off), ``15`` (accessory)
and ``30`` (run). Clicking emits ``Klem`` to the backend with the position
as a string. The button can be force-disabled until the backend signals
``EnableButton`` (mirrors the Angular kl-switch semantics: dashboard
shouldn't allow KL-on until the brain says it's ready).
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..client import events as ev
from ..client.socketio_client import SocketIOClient


class KlSwitch(QWidget):
    """Tri-state kl-switch (0/15/30) backed by EnableButton from the backend."""

    POSITIONS = (
        (ev.KL_OFF, "0",  "Off",  "#666"),
        (ev.KL_ACC, "15", "Acc.", "#f0ad4e"),
        (ev.KL_RUN, "30", "Run",  "#5cb85c"),
    )

    kl_changed = pyqtSignal(str)  # for other widgets (DrivingModel etc.) to consume

    def __init__(self, client: SocketIOClient, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._client = client
        self._enabled_by_backend = False
        self._current = ev.KL_OFF

        title = QLabel("KL Switch")
        title.setStyleSheet("color:#aaa; font-weight:600;")

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}

        row = QHBoxLayout()
        row.setSpacing(4)
        for value, short, _label, color in self.POSITIONS:
            btn = QPushButton(short)
            btn.setCheckable(True)
            btn.setMinimumHeight(40)
            btn.setStyleSheet(self._stylesheet(color))
            btn.clicked.connect(lambda _checked=False, v=value: self._on_clicked(v))
            self._group.addButton(btn)
            self._buttons[value] = btn
            row.addWidget(btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(title)
        layout.addLayout(row)

        self._buttons[ev.KL_OFF].setChecked(True)

        self._client.enable_button_signal.connect(self._on_enable_button)
        # Never emit a KL value automatically on reconnect. A silent Klem=0
        # can leave the UI believing KL=30 while the firmware is actually off.
        # The backend emits EnableButton on connect and once per second.
        self._client.connection_status_changed.connect(self._on_connection_status)

    # ------------------------------------------------------------------
    @staticmethod
    def _stylesheet(color: str) -> str:
        return (
            "QPushButton { background:#222; color:#ccc; border:1px solid #333; "
            "padding:6px; font-weight:700; font-size:13pt; }"
            f"QPushButton:checked {{ background:{color}; color:white; "
            f"border:1px solid {color}; }}"
            "QPushButton:disabled { color:#555; }"
        )

    # ------------------------------------------------------------------
    def _on_clicked(self, value: str) -> None:
        if not self._enabled_by_backend and value != ev.KL_OFF:
            # Bounce back: re-check the current button.
            self._buttons[self._current].setChecked(True)
            return
        self._set_current(value)
        self._client.emit_message(ev.CMD_KLEM, value)

    def _on_enable_button(self, enabled: bool) -> None:
        self._enabled_by_backend = bool(enabled)
        # Visual hint: dim the 15/30 buttons while disabled.
        self._buttons[ev.KL_ACC].setEnabled(self._enabled_by_backend)
        self._buttons[ev.KL_RUN].setEnabled(self._enabled_by_backend)

    def _on_connection_status(self, status: str) -> None:
        if status in {"disconnected", "error"}:
            self._enabled_by_backend = False
            self._set_current(ev.KL_OFF)
            self._on_enable_button(False)

    def _set_current(self, value: str) -> None:
        if value not in self._buttons:
            return
        self._current = value
        self._buttons[value].setChecked(True)
        self.kl_changed.emit(value)

    # ------------------------------------------------------------------
    @property
    def current(self) -> str:
        return self._current
