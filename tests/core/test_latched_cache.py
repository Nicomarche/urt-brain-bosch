"""Latched-cache replay tests for the XSUB/XPUB broker.

The broker caches the last frame seen on every topic listed in
:data:`src.core.bus.topics.LATCHED_TOPIC_NAMES` and replays it to any new
subscriber. This file proves the contract end-to-end with real ZMQ sockets
over loopback.

Each test stamps a unique topic name into ``LATCHED_TOPIC_NAMES`` (via
monkeypatch) BEFORE starting the broker so the assertion does not depend
on the production list staying frozen. The broker re-reads
``LATCHED_TOPIC_NAMES`` inside ``run()``, so this monkeypatch lands.
"""

from __future__ import annotations

import socket
import time

import pytest

from src.core.bus import Node, Topic, qos
from src.core.bus import topics as bus_topics
from src.core.bus.broker import BrokerThread


def _free_tcp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def latched_broker(monkeypatch):
    """Broker + endpoint env vars + a latched topic name registered for the test."""
    xsub = f"tcp://127.0.0.1:{_free_tcp_port()}"
    xpub = f"tcp://127.0.0.1:{_free_tcp_port()}"
    monkeypatch.setenv("URT_BUS_XSUB_ENDPOINT", xsub)
    monkeypatch.setenv("URT_BUS_XPUB_ENDPOINT", xpub)
    monkeypatch.setenv("URT_BUS_XSUB_BIND", xsub)
    monkeypatch.setenv("URT_BUS_XPUB_BIND", xpub)

    latched_topic = "/test/latched/state"
    monkeypatch.setattr(
        bus_topics,
        "LATCHED_TOPIC_NAMES",
        frozenset({latched_topic}),
    )

    from src.core.bus.endpoints import xpub_bind, xsub_bind

    thread = BrokerThread(xsub_bind=xsub_bind(), xpub_bind=xpub_bind())
    thread.daemon = True
    thread.start()
    time.sleep(0.05)
    try:
        yield latched_topic
    finally:
        thread.stop()
        thread.join(timeout=2.0)


def _await_publish_landing(pub, sub, payload, *, timeout_s: float = 2.0) -> None:
    """Loop publish+recv until the slow joiner handshake completes.

    ZMQ pub/sub drops messages whose SUB filter hasn't propagated yet. To
    guarantee the broker has actually seen and cached our payload we drive
    a temporary "warm" subscriber until it observes the payload, then we
    discard it. The cache lives in the broker thread and survives the
    warm subscriber going away.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pub.publish(payload)
        observed = sub.recv(timeout_ms=50)
        if observed == payload:
            return
    raise AssertionError(
        f"Warm subscriber never confirmed publish landed (payload={payload!r})"
    )


def test_latched_topic_replays_to_late_subscriber(latched_broker):
    """A subscriber connected AFTER the publish still receives the value."""
    topic = Topic(latched_broker, dict)

    pub_node = Node("test-pub")
    warm_node = Node("test-warm")
    warm_sub = warm_node.subscriber(topic, qos=qos.LATCHED)
    pub = pub_node.publisher(topic, qos=qos.LATCHED)

    _await_publish_landing(pub, warm_sub, {"state": "armed"})
    warm_node.close()
    # Brief pause so the warm subscriber's auto-unsubscribe message is
    # consumed by the broker before the late subscriber arrives.
    time.sleep(0.05)

    late_node = Node("test-late")
    late_sub = late_node.subscriber(topic, qos=qos.LATCHED)

    replayed = late_sub.recv(timeout_ms=2000)
    late_node.close()
    pub_node.close()

    assert replayed == {"state": "armed"}


def test_non_latched_topic_does_not_replay(latched_broker):
    """A topic NOT in LATCHED_TOPIC_NAMES gets no replay."""
    topic = Topic("/test/notlatched/event", dict)

    pub_node = Node("test-pub-nolat")
    warm_node = Node("test-warm-nolat")
    warm_sub = warm_node.subscriber(topic, qos=qos.EVENT)
    pub = pub_node.publisher(topic, qos=qos.EVENT)

    _await_publish_landing(pub, warm_sub, {"once": True})
    warm_node.close()
    time.sleep(0.05)

    late_node = Node("test-late-nolat")
    late_sub = late_node.subscriber(topic, qos=qos.EVENT)

    replayed = late_sub.recv(timeout_ms=400)
    late_node.close()
    pub_node.close()

    assert replayed is None


def test_latched_value_overwritten_by_new_publish(latched_broker):
    """The cache holds only the latest value, not a backlog."""
    topic = Topic(latched_broker, dict)

    pub_node = Node("test-pub-overwrite")
    warm_node = Node("test-warm-overwrite")
    warm_sub = warm_node.subscriber(topic, qos=qos.LATCHED)
    pub = pub_node.publisher(topic, qos=qos.LATCHED)

    _await_publish_landing(pub, warm_sub, {"v": 1})
    warm_sub.drain()
    # Same retry pattern for v=2: depth=1 on the PUB side can drop a send
    # if the warm sub hasn't drained yet; loop until it lands.
    _await_publish_landing(pub, warm_sub, {"v": 2})
    warm_node.close()
    time.sleep(0.05)

    late_node = Node("test-late-overwrite")
    late_sub = late_node.subscriber(topic, qos=qos.LATCHED)
    replayed = late_sub.recv(timeout_ms=2000)
    late_node.close()
    pub_node.close()

    assert replayed == {"v": 2}
