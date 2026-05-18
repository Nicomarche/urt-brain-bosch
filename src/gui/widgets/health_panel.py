"""health_panel — widget que muestra heartbeats por thread.

Plan F4: consume ``THREAD_HEARTBEAT_MSG`` y dibuja una lista con dot
verde/rojo por thread:
  * Verde si heartbeat reciente (< stale_threshold_s) y ok=True.
  * Rojo si stale o ok=False.
  * Latencia mostrada al lado.

Implementación QSS lista para integrarse al main_window — el widget se
auto-actualiza en cada signal del cliente ZmqBus.
"""

from __future__ import annotations

import time
from typing import Any

try:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover
    Qt = None
    QTimer = None
    QHBoxLayout = QLabel = QListWidget = QListWidgetItem = QVBoxLayout = QWidget = None


_STALE_THRESHOLD_S = 3.0


if QWidget is not None:

    class HealthPanel(QWidget):
        """Widget en columna mostrando ``thread_name | dot | latency``."""

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            layout = QVBoxLayout(self)
            title = QLabel("Threads")
            title.setStyleSheet("font-weight:600; font-size:11pt; color:#ddd;")
            layout.addWidget(title)
            self._list = QListWidget()
            self._list.setStyleSheet(
                "QListWidget { background:#1e1e1e; color:#ddd; border:1px solid #333; }"
            )
            layout.addWidget(self._list)
            # Map thread_name → (last_payload, last_recv_monotonic)
            self._state: dict[str, tuple[dict[str, Any], float]] = {}
            self._refresh_timer = QTimer(self)
            self._refresh_timer.timeout.connect(self._refresh)
            self._refresh_timer.start(500)  # 2 Hz redraw

        def on_heartbeat(self, payload: Any) -> None:
            """Slot conectado a ``client.thread_heartbeat_signal``."""
            if not isinstance(payload, dict):
                return
            name = str(payload.get("thread_name", "")).strip()
            if not name:
                return
            self._state[name] = (dict(payload), time.monotonic())

        def _refresh(self) -> None:
            self._list.clear()
            now = time.monotonic()
            for name in sorted(self._state.keys()):
                payload, recv_at = self._state[name]
                stale = (now - recv_at) > _STALE_THRESHOLD_S
                ok = bool(payload.get("ok", False)) and not stale
                latency_ms = float(payload.get("latency_ms", 0.0))
                dot = "🟢" if ok else "🔴"
                text = f"{dot}  {name:24s}  {latency_ms:6.1f} ms"
                item = QListWidgetItem(text)
                item.setForeground(Qt.green if ok else Qt.red)
                self._list.addItem(item)

else:  # pragma: no cover
    class HealthPanel:  # type: ignore[no-redef]
        pass


__all__ = ["HealthPanel"]
