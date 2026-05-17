"""End-to-end smoke test for the pub/sub bus skeleton (Sprint 1).

Verifies, on a single host with ipc:// endpoints in a per-test temp dir:

  1. A Publisher + Subscriber pair can exchange a message through the
     XSUB/XPUB broker.
  2. ``conflate=True`` (the ``STREAM`` preset) keeps only the latest
     message — replaces the legacy ``_AsyncPipeSender(maxsize=1)`` behavior.
  3. The legacy ``messageHandlerSender`` / ``messageHandlerSubscriber``
     shim wraps the bus with byte-for-byte compatibility for an existing
     ``allMessages`` enum.

Each test isolates its own endpoint set via ``monkeypatch`` on env vars
so the bus singletons (``zmq.Context.instance()``, the shim's Node) don't
leak state between tests in the same pytest run.
"""

from __future__ import annotations

import pathlib
import time

import pytest

from src.core.bus import Node, Topic, qos
from src.core.bus.broker import BrokerThread


def _free_tcp_port() -> int:
    """Reserve an ephemeral TCP port on the loopback, then release it.

    Pytest+macOS would push the IPC path past ``sockaddr_un``'s 103-char
    limit, so we use TCP loopback for tests. Production uses ipc:// (set
    by ``run.sh`` / defaults in ``endpoints.py``).
    """
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def bus_endpoints(monkeypatch):
    xsub = f"tcp://127.0.0.1:{_free_tcp_port()}"
    xpub = f"tcp://127.0.0.1:{_free_tcp_port()}"
    monkeypatch.setenv("URT_BUS_XSUB_ENDPOINT", xsub)
    monkeypatch.setenv("URT_BUS_XPUB_ENDPOINT", xpub)
    monkeypatch.setenv("URT_BUS_XSUB_BIND", xsub)
    monkeypatch.setenv("URT_BUS_XPUB_BIND", xpub)
    yield xsub, xpub


@pytest.fixture
def broker(bus_endpoints):
    """Start a broker thread; tear it down after the test."""
    from src.core.bus.endpoints import xpub_bind, xsub_bind

    thread = BrokerThread(xsub_bind=xsub_bind(), xpub_bind=xpub_bind())
    thread.daemon = True
    thread.start()
    # Tiny wait so the binds settle. The pub/sub handshake (XPUB
    # forwarding the SUB to XSUB) is async; the publisher will lose
    # messages sent before the subscriber's filter arrives. We sleep a
    # bit here AND again after subscribing — that's the classic "slow
    # joiner" workaround in ZMQ pub/sub. See zguide §1.
    time.sleep(0.05)
    yield thread
    thread.stop()
    thread.join(timeout=2.0)


def test_pub_sub_roundtrip(broker):
    node = Node("test-roundtrip")
    topic = Topic("/test/roundtrip", dict)
    sub = node.subscriber(topic, qos=qos.EVENT)
    pub = node.publisher(topic, qos=qos.EVENT)
    # Slow-joiner pause: subscriber's filter must reach the broker before
    # the publisher sends.
    time.sleep(0.1)

    pub.publish({"hello": "world", "n": 42})
    msg = sub.recv(timeout_ms=2000)
    node.close()

    assert msg == {"hello": "world", "n": 42}


def test_stream_qos_keeps_only_latest(broker):
    """STREAM preset = best_effort + depth=1 + conflate=True.

    A subscriber that lets the buffer fill up should see only the newest
    message, never the queued old ones. This is the contract that replaces
    the legacy ``_AsyncPipeSender(maxsize=1)``.
    """
    node = Node("test-stream")
    topic = Topic("/test/stream", dict)
    sub = node.subscriber(topic, qos=qos.STREAM)
    pub = node.publisher(topic, qos=qos.STREAM)

    # Slow-joiner handshake: keep sending warm-up packets until the
    # subscriber actually sees one. ZMQ pub/sub silently drops messages
    # sent before a subscription filter has propagated through the proxy,
    # so we can't trust a static sleep. ``warmup_seq`` lets us tell warm-up
    # frames apart from the real burst.
    warmup_deadline = time.monotonic() + 2.0
    warmup_seq = 0
    while True:
        pub.publish({"warmup": True, "seq": warmup_seq})
        msg = sub.recv(timeout_ms=50)
        if msg is not None and msg.get("warmup"):
            break
        warmup_seq += 1
        if time.monotonic() > warmup_deadline:
            node.close()
            raise AssertionError("Subscriber never received the warm-up packet")
    sub.drain()

    # Burst 1000 messages with a tiny pause every 50 — gives the broker
    # time to forward without letting us exit before any messages land at
    # the subscriber.
    for i in range(1000):
        pub.publish({"i": i})
        if i % 50 == 0:
            time.sleep(0.001)
    time.sleep(0.2)

    received = sub.recv(timeout_ms=1000)
    # Drain any additional messages the conflate buffer may have queued
    # after our recv — the second recv should still return only one (or
    # None if conflate kept just the one we already read).
    additional = []
    while True:
        extra = sub.try_recv()
        if extra is None:
            break
        additional.append(extra)
    node.close()

    assert received is not None
    # The contract is "no old backlog": we must NOT see the very early
    # messages from the burst (i < 100 is essentially impossible if conflate
    # is doing its job and the broker is keeping up).
    assert received["i"] >= 500, (
        f"Conflate failed: got i={received['i']} (expected late value)"
    )
    # And at most 1 additional message (since the burst ended before our
    # second recv). In practice this is 0 — we assert ≤1 for robustness
    # against scheduling jitter.
    assert len(additional) <= 1, (
        f"Conflate failed: got {len(additional)} extra messages after the first"
    )


def test_shim_publishes_through_bus(broker):
    """The shim's ``messageHandlerSender`` / ``messageHandlerSubscriber``
    should round-trip a payload using an actual Topic constant.

    Picks ``RECORDING`` (a low-traffic event topic with a simple bool
    payload) to avoid colliding with any other test publisher.
    """
    from src.core.bus.shim import messageHandlerSender, messageHandlerSubscriber
    from src.core.bus.topics import RECORDING

    # The shim ignores queuesList — pass an empty dict to satisfy the
    # legacy signature.
    queues: dict = {}
    sub = messageHandlerSubscriber(queues, RECORDING, "fifo", subscribe=True)
    sender = messageHandlerSender(queues, RECORDING)
    # Materialize the lazy subscriber socket so its filter starts
    # propagating before the first publish.
    sub.subscribe()
    time.sleep(0.1)

    # Slow-joiner retry: ZMQ may drop messages sent before the
    # subscription handshake completes, especially when the sender's
    # ``connect()`` is also brand new. Loop until either we get the
    # message or the deadline elapses.
    out = None
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        sender.send(True)
        time.sleep(0.05)
        out = sub.receive()
        if out is not None:
            break

    assert out is True
