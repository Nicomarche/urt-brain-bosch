"""Session lifecycle: login, heartbeat, single-session policy.

The Flask backend implements a "first-client-wins" session: the first user
to send ``SessionAccess`` is granted, and other connected clients are
ignored until the first one calls ``SessionEnd`` (or the heartbeat times
out). This module mirrors the contract on the GUI side.

Pieces:

* ``SessionManager`` — QObject orchestrator. Owns the heartbeat timer and
  the password check. Emits ``authenticated``/``denied``/``lost`` so the
  ``MainWindow`` can swap login dialog ↔ dashboard accordingly.
* Password handling matches Angular's ``app.component.ts``: an MD5 hash is
  configured at build time; if it's empty, login is "no password" (just
  hit enter) and the GUI shows a warning banner.
* Heartbeat: every 20 s the backend emits ``heartbeat``; we reply with a
  ``Heartbeat`` message envelope. If the backend stops sending heartbeats
  it eventually emits ``heartbeat_disconnect`` and we treat that as "lost".
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from . import events as ev
from .socketio_client import SocketIOClient


_logger = logging.getLogger(__name__)


def md5_hex(text: str) -> str:
    """MD5 hash → lowercase hex. Matches Angular's ``CryptoJS.MD5(...).toString()``."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class SessionManager(QObject):
    """Login/heartbeat orchestrator on top of a ``SocketIOClient``.

    The configured password hash is the MD5 of the desired plain-text
    password. Empty hash = no password required (``--no-login`` or unset).
    """

    # ---- Signals to the MainWindow ----
    authenticated = pyqtSignal()              # SessionAccess granted
    denied = pyqtSignal(str)                  # human reason
    lost = pyqtSignal(str)                    # connection lost mid-session

    HEARTBEAT_TIMEOUT_MS = 60_000  # 3 missed heartbeats at 20s each

    def __init__(self, client: SocketIOClient, password_md5: str = "",
                 skip_login: bool = False, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._client = client
        self._password_md5 = (password_md5 or "").strip().lower()
        self._skip_login = skip_login
        self._is_authenticated = False
        # ``attempt_login`` may run before the SocketIO transport is up
        # (the GUI shows its login dialog while ``client.start()`` is still
        # connecting in the background). We capture the *intent* to log in
        # here and replay it whenever the connection becomes "connected".
        # Bonus: this also re-authenticates automatically after a mid-session
        # reconnect, which the backend treats as a fresh socket.
        self._pending_login = False

        # Watchdog: if no heartbeat arrives within HEARTBEAT_TIMEOUT_MS we
        # assume the link is gone. The backend's own watchdog is similar
        # but we don't trust the network in monitor mode.
        self._heartbeat_watchdog = QTimer(self)
        self._heartbeat_watchdog.setSingleShot(True)
        self._heartbeat_watchdog.timeout.connect(self._on_heartbeat_timeout)

        # Wire backend → here.
        self._client.session_access_signal.connect(self._on_session_access)
        self._client.heartbeat_signal.connect(self._on_heartbeat)
        self._client.heartbeat_disconnect_signal.connect(self._on_heartbeat_disconnect)
        self._client.connection_status_changed.connect(self._on_connection_status)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def is_authenticated(self) -> bool:
        return self._is_authenticated

    @property
    def has_password(self) -> bool:
        return bool(self._password_md5)

    def attempt_login(self, plaintext_password: str = "") -> None:
        """Try to log in. If a password is configured we check it locally
        (Angular-compatible behaviour). On match — or in --no-login — we
        record the login intent and send ``SessionAccess`` either now (if
        already connected) or as soon as the SocketIO transport comes up.

        Emits ``denied("password")`` immediately if the local check fails;
        the backend response (granted / single-session-busy) arrives via
        the ``session_access`` event.
        """
        if self._skip_login or not self.has_password:
            _logger.info("Login: no password required")
        else:
            if md5_hex(plaintext_password) != self._password_md5:
                _logger.info("Login: password mismatch")
                self.denied.emit("password")
                return

        self._pending_login = True
        self._send_session_access_if_connected()

    def logout(self) -> None:
        """Tell the backend we're leaving so the next client can take over."""
        if self._is_authenticated:
            self._client.emit_message(ev.CMD_SESSION_END, True)
        self._is_authenticated = False
        self._pending_login = False
        self._heartbeat_watchdog.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _send_session_access_if_connected(self) -> None:
        """Emit SessionAccess now if the socket is up; otherwise wait for
        the ``connected`` status signal to replay it.
        """
        if self._client.is_connected():
            self._client.emit_message(ev.CMD_SESSION_ACCESS)
        else:
            _logger.info("Login queued — will fire once SocketIO connects")

    # ------------------------------------------------------------------
    # Backend → here
    # ------------------------------------------------------------------
    def _on_session_access(self, granted: bool) -> None:
        if granted:
            _logger.info("Session granted")
            self._is_authenticated = True
            self._heartbeat_watchdog.start(self.HEARTBEAT_TIMEOUT_MS)
            self.authenticated.emit()
        else:
            _logger.warning("Session denied (single-session lock?)")
            self._is_authenticated = False
            self.denied.emit("busy")

    def _on_heartbeat(self, _payload) -> None:
        """Backend asked for a heartbeat → bounce one back and reset watchdog."""
        if not self._is_authenticated:
            return
        self._client.emit_message(ev.CMD_HEARTBEAT, "pong")
        self._heartbeat_watchdog.start(self.HEARTBEAT_TIMEOUT_MS)

    def _on_heartbeat_disconnect(self, _payload) -> None:
        _logger.warning("Backend reported heartbeat disconnect")
        self._is_authenticated = False
        self._heartbeat_watchdog.stop()
        self.lost.emit("heartbeat")

    def _on_heartbeat_timeout(self) -> None:
        _logger.warning("Local heartbeat watchdog tripped")
        self._is_authenticated = False
        self.lost.emit("watchdog")

    def _on_connection_status(self, status: str) -> None:
        if status == "connected":
            # Replay any pending login: this covers the very first connect
            # (login is requested before the transport finishes opening) and
            # also reconnects mid-session — the backend resets the per-socket
            # session on disconnect, so we must re-issue SessionAccess.
            if self._pending_login and not self._is_authenticated:
                _logger.info("SocketIO connected — sending queued SessionAccess")
                self._client.emit_message(ev.CMD_SESSION_ACCESS)
        elif status == "disconnected" and self._is_authenticated:
            # The socket dropped — the backend will eventually reset our
            # session. Flag the user immediately rather than letting the
            # watchdog tick down. ``_pending_login`` stays True so reconnect
            # automatically re-authenticates.
            self._is_authenticated = False
            self._heartbeat_watchdog.stop()
            self.lost.emit("disconnected")
