"""Shared pytest fixtures.

The ``broker`` fixture spins up a real :class:`BrokerThread` on
``tcp://127.0.0.1:<ephemeral>`` endpoints and resets the per-process
``Node`` singleton inside :mod:`src.core.bus.shim` so each test that
exercises the bus sees a clean state.

Tests that publish or subscribe via the legacy ``messageHandlerSender`` /
``messageHandlerSubscriber`` shim should request this fixture — without
a running broker, ZMQ publishes either silently drop (best_effort QoS)
or block (reliable QoS).
"""

from __future__ import annotations

import socket
import time

import pytest


def _free_tcp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def bus_endpoints(monkeypatch):
    """Override the bus endpoints to ephemeral TCP loopback ports.

    pytest's tmp_path can blow past ``sockaddr_un``'s 103-char limit on
    macOS, so we use TCP for tests. Production uses ipc:// (default).
    """
    xsub = f"tcp://127.0.0.1:{_free_tcp_port()}"
    xpub = f"tcp://127.0.0.1:{_free_tcp_port()}"
    monkeypatch.setenv("URT_BUS_XSUB_ENDPOINT", xsub)
    monkeypatch.setenv("URT_BUS_XPUB_ENDPOINT", xpub)
    monkeypatch.setenv("URT_BUS_XSUB_BIND", xsub)
    monkeypatch.setenv("URT_BUS_XPUB_BIND", xpub)
    yield xsub, xpub


@pytest.fixture
def broker(bus_endpoints):
    """Start a fresh broker thread, isolate the shim's Node singleton."""
    from src.core.bus import shim
    from src.core.bus.broker import BrokerThread
    from src.core.bus.endpoints import xpub_bind, xsub_bind

    # The shim caches one Node per process; if a previous test created
    # publishers/subscribers pointing at the previous broker's endpoints,
    # they'd silently fail here. Reset before yielding.
    previous_node = shim._NODE
    if previous_node is not None:
        try:
            previous_node.close()
        except Exception:
            pass
        shim._NODE = None

    thread = BrokerThread(xsub_bind=xsub_bind(), xpub_bind=xpub_bind())
    thread.daemon = True
    thread.start()
    # Tiny pause so the binds settle before tests start publishing.
    time.sleep(0.05)

    yield thread

    thread.stop()
    thread.join(timeout=2.0)
    # Final reset: don't leak this Node into the next test's process state.
    final_node = shim._NODE
    if final_node is not None:
        try:
            final_node.close()
        except Exception:
            pass
        shim._NODE = None
