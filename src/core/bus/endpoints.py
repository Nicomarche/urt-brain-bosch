"""Endpoint resolution for the pub/sub bus.

All endpoints are resolved through env vars with sensible defaults so:
  - production runs use ``ipc://`` (zero TCP overhead inside the robot);
  - tests can override with ephemeral ``tcp://127.0.0.1:0`` for isolation;
  - the GUI/monitor mode can swap in ``tcp://0.0.0.0:5557`` to expose the
    bus on the LAN without code changes.

Distinguishing ``*_endpoint()`` (where peers CONNECT) from ``*_bind()``
(where the broker BINDS) lets us run a single broker that listens on both
``ipc://`` and ``tcp://`` simultaneously by setting the bind vars
differently from the endpoint vars. Sprint 1 doesn't need that yet, but the
hook is here.
"""

from __future__ import annotations

import os
import pathlib

# IPC sockets land in a per-user directory rather than naked /tmp to avoid
# collisions when multiple developers run the stack on the same machine.
_IPC_DIR = pathlib.Path(os.environ.get("URT_BUS_IPC_DIR", "/tmp/urt-bus"))


def _ipc(name: str) -> str:
    _IPC_DIR.mkdir(parents=True, exist_ok=True)
    return f"ipc://{_IPC_DIR / name}.sock"


def xsub_endpoint() -> str:
    """Where publishers CONNECT their PUB sockets (→ broker XSUB)."""
    return os.environ.get("URT_BUS_XSUB_ENDPOINT", _ipc("xsub"))


def xpub_endpoint() -> str:
    """Where subscribers CONNECT their SUB sockets (→ broker XPUB)."""
    return os.environ.get("URT_BUS_XPUB_ENDPOINT", _ipc("xpub"))


def _split_csv(raw: str) -> list[str]:
    """Split a comma-separated env-var value into trimmed endpoints."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def xsub_binds() -> list[str]:
    """Endpoints the broker BINDS for publishers to CONNECT to.

    Supports multiple binds (CSV) so the same broker can serve robot-internal
    publishers over ``ipc://`` AND a remote GUI over ``tcp://`` simultaneously.
    """
    raw = os.environ.get("URT_BUS_XSUB_BIND")
    if raw is None:
        return [xsub_endpoint()]
    return _split_csv(raw) or [xsub_endpoint()]


def xpub_binds() -> list[str]:
    """Endpoints the broker BINDS for subscribers to CONNECT to."""
    raw = os.environ.get("URT_BUS_XPUB_BIND")
    if raw is None:
        return [xpub_endpoint()]
    return _split_csv(raw) or [xpub_endpoint()]


# Backwards-compatible single-string accessors for callers that only need
# the primary bind (e.g. tests that pass a single tcp:// endpoint).
def xsub_bind() -> str:
    return xsub_binds()[0]


def xpub_bind() -> str:
    return xpub_binds()[0]
