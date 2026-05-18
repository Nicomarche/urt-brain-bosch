"""Console panel — append-only log viewer.

Mirrors the Angular ``console.component`` behaviour: every ``console_log``
event from the backend gets appended at the bottom, and the view auto-
scrolls so the newest line is always visible (until the user scrolls up,
in which case we stop auto-scrolling so they can read history without
the view jumping).

Implementation:

* ``QPlainTextEdit`` is ~10× faster than ``QTextEdit`` for streaming
  text — avoids the rich-text layout engine which doesn't matter here.
* ``setMaximumBlockCount`` caps memory use: old lines roll off as new
  ones arrive instead of growing unbounded after a long run.
* The "follow tail" behaviour is computed from the scrollbar position
  *before* the new line is inserted — appending then scrolling is what
  causes the jump-on-scroll-up bug in the Angular version.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..client.zmq_bus_client import ZmqBusClient as SocketIOClient


class ConsolePanel(QWidget):
    """Streaming log viewer with autoscroll + clear + pause toggle."""

    DEFAULT_BUFFER_LINES = 5_000

    def __init__(self, client: SocketIOClient, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._client = client
        self._paused = False

        # ---- Toolbar ----
        self._autoscroll = QCheckBox("Autoscroll")
        self._autoscroll.setChecked(True)
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setCheckable(True)
        self._pause_btn.toggled.connect(self._set_paused)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self._autoscroll)
        toolbar.addWidget(self._pause_btn)
        toolbar.addWidget(clear_btn)
        toolbar.addStretch(1)

        # ---- Log view ----
        self._view = QPlainTextEdit(self)
        self._view.setReadOnly(True)
        self._view.setMaximumBlockCount(self.DEFAULT_BUFFER_LINES)
        self._view.setLineWrapMode(QPlainTextEdit.NoWrap)
        # Monospaced — log lines line up.
        self._view.setStyleSheet(
            "QPlainTextEdit { background: #111; color: #ddd; "
            "font-family: 'Menlo','Consolas',monospace; font-size: 11pt; }"
        )

        # ---- Layout ----
        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self._view, 1)

        self._client.console_log_signal.connect(self._on_log)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_log(self, line: str) -> None:
        if self._paused or not line:
            return
        # Decide *before* appending whether the user is at the tail —
        # otherwise the freshly-appended line itself would push us off the
        # bottom and we'd think they had scrolled up.
        scrollbar = self._view.verticalScrollBar()
        was_at_bottom = (
            scrollbar.value() >= scrollbar.maximum() - 4  # small slack
        )

        # Strip a single trailing newline if present so we don't double-space.
        self._view.appendPlainText(line.rstrip("\n"))

        if self._autoscroll.isChecked() and was_at_bottom:
            self._view.moveCursor(QTextCursor.End)
            scrollbar.setValue(scrollbar.maximum())

    def _set_paused(self, paused: bool) -> None:
        self._paused = paused
        self._pause_btn.setText("Resume" if paused else "Pause")

    def _clear(self) -> None:
        self._view.clear()

    # ------------------------------------------------------------------
    # Public helper for other panels (e.g. show local diagnostic msgs)
    # ------------------------------------------------------------------
    def append_local(self, line: str) -> None:
        """Append a locally-generated line — useful for status messages
        from the GUI itself (e.g. "Connecting to 192.168.1.99…").
        """
        self._on_log(line)
