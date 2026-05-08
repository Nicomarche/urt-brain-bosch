"""Settings/parameters table — sliders + dropdowns persisted to JSON.

Replaces the Angular ``table.component``. Each row is a parameter the
operator can tweak: ``Brightness``, ``Contrast``, ``ToggleInstant``, etc.
The schema mirrors the legacy ``src/utils/table_state.json`` exactly so
the Flask backend keeps reading the same file (we just write to it from
the GUI now).

Schema per entry::

    {
      "type":         "slider" | "dropdown",
      "value":        current,
      "initialValue": original,
      "values":       [min, max] for sliders / [option, ...] for dropdowns,
      "command":      str (the brain channel name to emit),
      "interval":     int (ms; >0 = throttle continuous emits),
      "checked":      bool (dropdown helper),
      "hasChanged":   bool
    }

Persistence: every commit writes the whole table to ``config/params.json``
(GUI source of truth) **and** sends a ``save`` event over SocketIO so the
backend's ``handle_save_table_state`` keeps the legacy ``table_state.json``
in sync. This means the legacy Angular dashboard (still served from
``legacy/dashboard-frontend``) stays functional during the transition.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..client import events as ev
from ..client.socketio_client import SocketIOClient
from ..config import persistence


_logger = logging.getLogger(__name__)


class ParamsTable(QWidget):
    """Editable parameters grid backed by ``config/params.json``."""

    COLUMNS = ("Name", "Type", "Value", "Action")

    def __init__(self, client: SocketIOClient, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._client = client

        # Pull from disk first; fall back to the backend snapshot.
        loaded = persistence.load_config("params")
        self._params: dict = loaded.get("params") if isinstance(
            loaded, dict) and "params" in loaded else loaded
        if not isinstance(self._params, dict):
            self._params = {}

        # ---- Toolbar ----
        toolbar = QHBoxLayout()
        reload_btn = QPushButton("Reload from backend")
        reload_btn.clicked.connect(self._client.emit_load)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        toolbar.addWidget(reload_btn)
        toolbar.addStretch(1)
        toolbar.addWidget(save_btn)

        # ---- Table ----
        self._table = QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self._table, 1)

        # Re-render whenever loadBack arrives.
        self._client.load_back_signal.connect(self._on_load_back)
        self._render()

    # ------------------------------------------------------------------
    def _render(self) -> None:
        self._table.setRowCount(0)
        for name, spec in self._params.items():
            if not isinstance(spec, dict):
                continue
            self._add_row(name, spec)

    def _add_row(self, name: str, spec: dict) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        self._table.setItem(row, 0, QTableWidgetItem(str(name)))
        self._table.setItem(row, 1, QTableWidgetItem(str(spec.get("type", ""))))

        # Editor in column 2.
        editor = self._make_editor(name, spec)
        self._table.setCellWidget(row, 2, editor)

        # Apply button in column 3.
        send = QPushButton("Send")
        send.clicked.connect(lambda _checked=False, n=name: self._send(n))
        self._table.setCellWidget(row, 3, send)

    def _make_editor(self, name: str, spec: dict) -> QWidget:
        """Pick the right editor widget for the spec type."""
        kind = spec.get("type")
        if kind == "slider":
            return self._slider_editor(name, spec)
        if kind == "dropdown":
            return self._dropdown_editor(name, spec)
        # Unknown / Config → show as label, no edits possible.
        label = QLabel(str(spec.get("value", "")))
        label.setStyleSheet("color:#888;")
        return label

    def _slider_editor(self, name: str, spec: dict) -> QWidget:
        wrap = QWidget()
        h = QHBoxLayout(wrap)
        h.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Horizontal)
        values = spec.get("values") or [0, 100]
        try:
            mn, mx = int(values[0]), int(values[-1])
        except (TypeError, ValueError, IndexError):
            mn, mx = 0, 100
        slider.setMinimum(mn)
        slider.setMaximum(mx)
        try:
            slider.setValue(int(spec.get("value", mn)))
        except (TypeError, ValueError):
            slider.setValue(mn)

        readout = QLabel(str(slider.value()))
        readout.setMinimumWidth(40)
        slider.valueChanged.connect(lambda v, lbl=readout, n=name:
                                    self._on_slider_change(n, v, lbl))
        h.addWidget(slider, 1)
        h.addWidget(readout)
        return wrap

    def _dropdown_editor(self, name: str, spec: dict) -> QWidget:
        combo = QComboBox()
        for v in spec.get("values") or []:
            combo.addItem(str(v))
        idx = combo.findText(str(spec.get("value", "")))
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.currentTextChanged.connect(
            lambda text, n=name: self._on_dropdown_change(n, text)
        )
        return combo

    # ------------------------------------------------------------------
    def _on_slider_change(self, name: str, value: int, readout: QLabel) -> None:
        readout.setText(str(value))
        if name in self._params:
            self._params[name]["value"] = value
            self._params[name]["hasChanged"] = True

    def _on_dropdown_change(self, name: str, value: str) -> None:
        if name in self._params:
            self._params[name]["value"] = value
            self._params[name]["hasChanged"] = True

    def _send(self, name: str) -> None:
        spec = self._params.get(name)
        if not isinstance(spec, dict):
            return
        cmd = spec.get("command") or name
        value = spec.get("value")
        # Mirror the Angular formatting: strings for both kinds.
        self._client.emit_message(cmd, str(value))
        spec["hasChanged"] = False

    def _save(self) -> None:
        # Local JSON for version control (push from PC, pull on car).
        persistence.save_config("params", {"params": self._params})
        # Legacy bridge: backend still reads src/utils/table_state.json.
        # Send the same payload on the `save` channel so the backend
        # writes that file too — Angular dashboard stays in sync.
        self._client.emit_save(self._params)

    # ------------------------------------------------------------------
    def _on_load_back(self, payload) -> None:
        if isinstance(payload, dict) and payload:
            self._params = payload
            self._render()
