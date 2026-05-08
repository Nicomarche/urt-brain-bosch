"""Top-level dashboard window.

This is intentionally a *skeleton* in this commit — the real panels
(map view, camera, calibration wizard, …) are still being built out and
will plug into the ``QStackedWidget`` keyed by sidebar selection.

What the skeleton already does:

* Owns the shared ``SocketIOClient`` and the ``SessionManager``.
* Shows a minimal login dialog when a password hash is configured.
* Lays out the main window: left sidebar (tab list) + right stack
  (currently a single placeholder).
* Routes connection-status / session events into a status bar so the
  user always knows whether they're online and authenticated.
* Persists window size / splitter position to ``config/gui_layout.json``
  so the next launch comes up the same way.

The window is ready to host real panels: each panel is a ``QWidget``
that receives the ``SocketIOClient`` in its constructor and connects to
whichever signals it cares about. ``add_tab(label, widget)`` registers
one in the sidebar.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..client.session import SessionManager
from ..client.socketio_client import SocketIOClient
from ..config import persistence, settings


_logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Login dialog
# ----------------------------------------------------------------------
class LoginDialog(QDialog):
    """Modal password prompt — only shown when a hash is configured.

    The actual MD5 check happens in ``SessionManager.attempt_login``; this
    dialog just collects the plain text and forwards it.
    """

    def __init__(self, session: SessionManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._session = session
        self.setWindowTitle("BFMC Dashboard — Login")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Enter password:"))
        self._password = QLineEdit()
        self._password.setEchoMode(QLineEdit.Password)
        self._password.returnPressed.connect(self._submit)
        layout.addWidget(self._password)

        self._status = QLabel("")
        self._status.setStyleSheet("color: #c33;")
        layout.addWidget(self._status)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._submit)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Listen for the backend's verdict.
        self._session.authenticated.connect(self.accept)
        self._session.denied.connect(self._on_denied)

    def _submit(self) -> None:
        self._status.setText("Checking…")
        self._session.attempt_login(self._password.text())

    def _on_denied(self, reason: str) -> None:
        self._password.clear()
        self._password.setFocus()
        self._status.setText({
            "password": "Wrong password.",
            "busy":     "Another client is connected. Try again later.",
        }.get(reason, f"Denied ({reason})."))


# ----------------------------------------------------------------------
# Placeholder for tabs not yet implemented
# ----------------------------------------------------------------------
class _PlaceholderPanel(QWidget):
    """Greyed-out panel shown until the real widget is wired up."""

    def __init__(self, label: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        text = QLabel(f"{label}\n\n(not yet implemented)")
        text.setAlignment(Qt.AlignCenter)
        text.setStyleSheet("color: #888; font-size: 14pt;")
        layout.addWidget(text)


# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------
class MainWindow(QMainWindow):
    """Dashboard shell: sidebar + stacked panels + status bar."""

    # Order here = order in the sidebar. Telemetry no aparece como tab
    # propia: vive embebida en el Driving tab (splitter cámara | telemetry)
    # para ver stats mientras se maneja en manual.
    DEFAULT_TABS = (
        ("driving",      "Driving"),
        ("map",          "Map"),
        ("routes",       "Routes"),
        ("calibration",  "Calibration"),
        ("line",         "Line Following"),
        ("console",      "Console"),
    )

    def __init__(self, client: SocketIOClient, session: SessionManager,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._client = client
        self._session = session

        self.setWindowTitle("BFMC Dashboard")
        self._build_ui()
        self._wire_signals()
        self._restore_layout()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Sidebar — selecting an item swaps the panel on the right.
        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(200)
        self._sidebar.setStyleSheet(
            "QListWidget { font-size: 12pt; padding: 8px; }"
            "QListWidget::item { padding: 8px 4px; }"
        )

        self._stack = QStackedWidget()
        self._tab_keys: list[str] = []

        for key, label in self.DEFAULT_TABS:
            self.add_tab(key, label, _PlaceholderPanel(label))

        self._sidebar.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._sidebar.setCurrentRow(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._sidebar)
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self._splitter = splitter

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

        self.setStatusBar(QStatusBar())
        self._connection_label = QLabel("Disconnected")
        self._session_label = QLabel("Not logged in")
        self.statusBar().addPermanentWidget(self._session_label)
        self.statusBar().addPermanentWidget(self._connection_label)
        self._set_connection_status("disconnected")

    def add_tab(self, key: str, label: str, widget: QWidget) -> None:
        """Register a panel under ``key``. If ``key`` already exists it
        is replaced — this lets the bootstrapper swap placeholders for
        real widgets as they come online without reordering the sidebar.
        """
        if key in self._tab_keys:
            idx = self._tab_keys.index(key)
            old = self._stack.widget(idx)
            self._stack.removeWidget(old)
            old.deleteLater()
            self._stack.insertWidget(idx, widget)
            return
        self._tab_keys.append(key)
        item = QListWidgetItem(label)
        self._sidebar.addItem(item)
        self._stack.addWidget(widget)

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------
    def _wire_signals(self) -> None:
        self._client.connection_status_changed.connect(self._set_connection_status)
        self._session.authenticated.connect(self._on_authenticated)
        self._session.lost.connect(self._on_session_lost)

    def _set_connection_status(self, status: str) -> None:
        color = {"connected": "#2a8", "disconnected": "#888", "error": "#c33"}.get(
            status, "#888"
        )
        self._connection_label.setText(status.capitalize())
        self._connection_label.setStyleSheet(f"color: {color}; padding: 0 8px;")

    def _on_authenticated(self) -> None:
        self._session_label.setText("Logged in")
        self._session_label.setStyleSheet("color: #2a8; padding: 0 8px;")

    def _on_session_lost(self, reason: str) -> None:
        self._session_label.setText(f"Session lost ({reason})")
        self._session_label.setStyleSheet("color: #c33; padding: 0 8px;")
        QMessageBox.warning(
            self, "Connection lost",
            f"Lost dashboard session ({reason}). The GUI will keep trying to "
            f"reconnect — log in again once the indicator goes green."
        )

    # ------------------------------------------------------------------
    # Persistence: window/splitter sizes
    # ------------------------------------------------------------------
    def _restore_layout(self) -> None:
        layout = persistence.load_config("gui_layout")
        size = layout.get("window_size", [1280, 800])
        try:
            self.resize(QSize(int(size[0]), int(size[1])))
        except (TypeError, ValueError, IndexError):
            self.resize(1280, 800)
        sizes = layout.get("splitter_ratio")
        if isinstance(sizes, list) and len(sizes) == 2:
            try:
                self._splitter.setSizes([int(sizes[0]), int(sizes[1])])
            except (TypeError, ValueError):
                pass
        active = layout.get("active_tab")
        if isinstance(active, str) and active in self._tab_keys:
            self._sidebar.setCurrentRow(self._tab_keys.index(active))

    def _persist_layout(self) -> None:
        try:
            current_key = self._tab_keys[self._sidebar.currentRow()] \
                if 0 <= self._sidebar.currentRow() < len(self._tab_keys) \
                else self._tab_keys[0]
            persistence.save_config("gui_layout", {
                "window_size":     [self.width(), self.height()],
                "splitter_ratio":  self._splitter.sizes(),
                "active_tab":      current_key,
            })
        except Exception as exc:  # pragma: no cover - best effort on shutdown
            _logger.warning("Failed to persist layout: %s", exc)

    def closeEvent(self, event):  # noqa: N802 - Qt convention
        self._persist_layout()
        try:
            self._session.logout()
        finally:
            self._client.stop()
        super().closeEvent(event)


# ----------------------------------------------------------------------
# Helper used by main.py to bootstrap the whole stack
# ----------------------------------------------------------------------
def _build_driving_tab(client: SocketIOClient, window: "MainWindow") -> QWidget:
    """Driving tab = control + test pad arriba, cámara + telemetry abajo.

    Layout::

        ┌─ ControlPanel (KL + mode + record + speed/steer readouts) ─┐
        ├─ ManualTestPad (debug — botones que bypassean DrivingModel)┤
        ├─────────────────┬───────────────────────────────────────────┤
        │  CameraView     │  TelemetryCluster                         │
        │  (stretch 2)    │  (Speedo/Steer/Warn + Battery/Power +     │
        │                 │   CPU/Temp/RAM + mini map)                │
        │                 │  (stretch 1)                              │
        └─────────────────┴───────────────────────────────────────────┘

    Antes vivían en tabs distintas — había que cambiar de tab para ver
    la velocidad/steer reales mientras manejabas en manual. Acá se ven
    todos juntos. El divisor es un `QSplitter` arrastrable: en pantallas
    chicas el operador puede achicar telemetry o, al revés, agrandarla
    para usar el dashboard como cluster sin ocultar la cámara.

    El control panel instala el keyboard filter sobre la *aplicación*
    (NO sobre la ventana) así W/A/S/D/↑↓←→/space manejan sin importar
    qué widget tiene foco. Si lo instalábamos sobre el MainWindow, la
    sidebar `QListWidget` (que tiene foco al arrancar y `keyboardSearch`
    habilitado) se comía W/A/S/D como búsqueda incremental, y los
    botones consumen Space/Enter — el filter nunca veía la tecla.
    Instalando en `QApplication.instance()` el filter corre PRIMERO,
    antes de que el widget enfocado decida qué hacer.
    """
    from PyQt5.QtCore import Qt as _Qt
    from PyQt5.QtWidgets import QApplication, QSplitter
    from .camera_view import CameraView
    from .control_panel import ControlPanel
    from .manual_test_pad import ManualTestPad
    from .map_view import MapView
    from .telemetry import TelemetryCluster

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(8)

    control = ControlPanel(client)
    control.install_keyboard_on(QApplication.instance())
    layout.addWidget(control)
    layout.addWidget(ManualTestPad(client))

    body = QSplitter(_Qt.Horizontal)
    body.addWidget(CameraView(client, show_fps=True))
    mini_map = MapView(client, compact=True)
    mini_map.setMinimumHeight(240)
    body.addWidget(TelemetryCluster(client, footer_widget=mini_map))
    body.setStretchFactor(0, 2)  # cámara prioritaria
    body.setStretchFactor(1, 1)
    body.setSizes([900, 500])    # default razonable a 1280–1440px wide
    layout.addWidget(body, 1)
    return container


def _install_default_panels(window: "MainWindow", client: SocketIOClient) -> None:
    """Replace placeholder panels with real implementations as they exist.

    Kept here (vs in MainWindow itself) because the MainWindow shouldn't
    know which specific panel modules are available — this function is the
    single place where "what tab gets which widget" is decided. As more
    widgets land, just import + ``window.add_tab(...)`` here.
    """
    # Lazy imports keep startup fast and avoid pulling pyqtgraph etc. unless
    # the corresponding tab is actually exercised.
    from .calibration_wizard import CalibrationWizard
    from .console_panel import ConsolePanel
    from .line_following_panel import LineFollowingPanel
    from .map_view import MapView
    from .params_table import ParamsTable
    from .routes_panel import RoutesPanel

    # Telemetry no se registra como tab independiente: el `_build_driving_tab`
    # ya la embebe junto a la cámara para que el operador vea los gauges
    # mientras maneja en manual.
    window.add_tab("driving", "Driving", _build_driving_tab(client, window))
    window.add_tab("map", "Map", MapView(client))
    window.add_tab("routes", "Routes", RoutesPanel(client))
    window.add_tab("calibration", "Calibration", CalibrationWizard(client))
    window.add_tab("line", "Line Following", LineFollowingPanel(client))
    window.add_tab("params", "Params", ParamsTable(client))
    window.add_tab("console", "Console", ConsolePanel(client))


def show_main_window(client: SocketIOClient, session: SessionManager,
                     skip_login: bool) -> MainWindow:
    """Create and show the MainWindow, gating it on login when needed."""
    window = MainWindow(client, session)
    _install_default_panels(window, client)

    if skip_login or not session.has_password:
        # No password configured — kick off the SessionAccess handshake
        # immediately so the UI shows "logged in" once the backend grants.
        session.attempt_login("")
        window.show()
        return window

    # Password configured — show the dialog modally first.
    dlg = LoginDialog(session, parent=window)
    if dlg.exec_() == QDialog.Accepted:
        window.show()
    else:
        QApplication.instance().quit()
    return window
