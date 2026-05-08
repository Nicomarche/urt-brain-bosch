"""GUI entry point — ``python3 -m src.dashboard.gui.main``.

Starts a minimal PyQt5 dashboard:

1. Parse args (``--connect host:port``, ``--no-login``, ``--password-md5``).
2. Migrate legacy ``src/utils/table_state.json`` → ``config/routes.json``
   the first time the GUI runs against a fresh repo.
3. Build the shared ``SocketIOClient``, ``SessionManager`` and ``MainWindow``.
4. Hand control to the Qt event loop.

The ``run.sh`` launcher decides whether to start a backend alongside the
GUI (Mac dev mode) or to point it at a remote car (``--monitor IP``). All
the GUI cares about is the ``--connect`` host:port.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from .client.session import SessionManager
from .client.socketio_client import SocketIOClient
from .config import persistence, settings
from .widgets.main_window import show_main_window


def _install_sigint_handler(app: QApplication) -> None:
    """Wire Ctrl+C (SIGINT) so it cleanly quits the GUI.

    Two pieces are needed because Qt's event loop runs in C — Python's
    signal handler can only execute between bytecode instructions, but
    Qt happily blocks in ``app.exec_()`` for seconds without yielding:

    1. A SIGINT handler that asks Qt to quit. ``QApplication.quit()`` is
       thread-safe and just enqueues a quit event on the main loop.
    2. A short-period ``QTimer`` whose ``timeout`` callback does nothing
       — its only job is to keep popping back into Python land so the
       interpreter notices the pending signal and runs the handler.

    Without (2), Ctrl+C is silently swallowed until the user moves the
    mouse or generates some other Qt event.
    """
    def _handle_sigint(_signum, _frame) -> None:  # noqa: ANN001
        sys.stderr.write("\n[bfmc-gui] Ctrl+C — quitting…\n")
        app.quit()

    signal.signal(signal.SIGINT, _handle_sigint)
    # Also handle SIGTERM the same way so ``run.sh``'s cleanup-on-exit
    # plays nicely (the shell sends TERM to the GUI before SIGKILLing it).
    signal.signal(signal.SIGTERM, _handle_sigint)

    tick = QTimer()
    tick.setInterval(150)         # 150 ms — barely measurable, keeps signals live
    tick.timeout.connect(lambda: None)
    tick.start()
    # Keep a reference on the QApplication so the timer survives until exit.
    app._sigint_tick = tick  # type: ignore[attr-defined]


def _parse_host_port(spec: str) -> tuple[str, int]:
    """Split ``host:port`` (port optional, defaults to ``DEFAULT_PORT``).

    Accepts bare IPv4/IPv6 addresses without brackets — when no colon is
    present we apply the default port.
    """
    if ":" in spec and not spec.startswith("["):
        host, _, port_s = spec.rpartition(":")
        try:
            return host or settings.DEFAULT_HOST, int(port_s)
        except ValueError:
            pass
    return spec or settings.DEFAULT_HOST, settings.DEFAULT_PORT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bfmc-gui",
        description="BFMC Dashboard — single-window control surface for the URT car.",
    )
    parser.add_argument(
        "--connect", default=f"{settings.DEFAULT_HOST}:{settings.DEFAULT_PORT}",
        help="Backend SocketIO endpoint as host:port (default localhost:5005).",
    )
    parser.add_argument(
        "--no-login", action="store_true",
        help="Skip the password dialog (still issues SessionAccess to the backend).",
    )
    parser.add_argument(
        "--password-md5", default=settings.DEFAULT_PASSWORD_MD5,
        help="Override the configured MD5 password hash for this run.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level for the GUI (default INFO).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("bfmc-gui")

    # First-launch migration: pull legacy table_state into the new config layout.
    try:
        if persistence.migrate_legacy_table_state():
            logger.info("Migrated legacy table_state.json into config/routes.json")
    except Exception as exc:  # pragma: no cover - non-fatal
        logger.warning("Legacy migration failed (continuing): %s", exc)

    host, port = _parse_host_port(args.connect)
    logger.info("Connecting to %s:%s", host, port)

    app = QApplication(sys.argv)
    app.setApplicationName("BFMC Dashboard")
    _install_sigint_handler(app)

    client = SocketIOClient(host=host, port=port)
    session = SessionManager(
        client,
        password_md5=args.password_md5,
        skip_login=args.no_login,
    )

    window = show_main_window(client, session, skip_login=args.no_login)

    # Kick the SocketIO connection off after the window is up so the user
    # sees the "Disconnected" indicator transition to "Connected".
    client.start()

    exit_code = app.exec_()
    client.stop()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
