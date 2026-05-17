"""Bus ``Node`` — the API every process uses to publish/subscribe.

One Node per logical participant. A Node owns its publishers and
subscribers, all sharing a process-wide ``zmq.Context`` obtained lazily via
``zmq.Context.instance()`` (pyzmq's built-in singleton, with ``atexit``
shutdown).

Why lazy: ``zmq.Context`` MUST be created in the child process, not
inherited across ``multiprocessing`` ``spawn`` (forced on macOS at
``main.py:78-88``). Holding the context as a module-level singleton would
work too, but ``zmq.Context.instance()`` already does the safe pattern, so
we delegate to it.

Threading: in Sprint 1 we only expose pull-style subscribers
(``try_recv``/``recv``). The optional callback-driven poller thread is a
Sprint 3 deliverable — sketched in the plan, not built yet.
"""

from __future__ import annotations

from typing import Callable, Optional

import zmq

from src.core.bus import qos as _qos
from src.core.bus.endpoints import xpub_endpoint, xsub_endpoint
from src.core.bus.publisher import Publisher
from src.core.bus.qos import QoSProfile
from src.core.bus.subscriber import Subscriber
from src.core.bus.topics import Topic


def _context() -> zmq.Context:
    """Process-wide zmq Context. Safe to call from any thread; pyzmq's
    ``Context.instance()`` already registers an ``atexit`` finalizer.
    """
    return zmq.Context.instance()


class Node:
    """Bus participant. Subclass or instantiate directly.

    Example::

        node = Node("camera")
        speed_pub = node.publisher(topics.CMD_SPEED, qos.COMMAND)
        speed_pub.publish({"value": 12.5})

        imu_sub = node.subscriber(topics.SENSORS_IMU, qos=qos.STREAM)
        msg = imu_sub.try_recv()

        node.close()  # closes every socket the node opened
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._pubs: dict[str, Publisher] = {}
        self._subs: list[Subscriber] = []

    def publisher(
        self,
        topic: Topic,
        qos: QoSProfile = _qos.STREAM,
    ) -> Publisher:
        """Get or create a Publisher for ``topic``.

        Multiple calls with the same topic name return the same Publisher —
        opening many PUB sockets to the same topic from one process is
        wasteful and confuses the broker's subscription accounting.
        """
        existing = self._pubs.get(topic.name)
        if existing is not None:
            return existing

        sock = _context().socket(zmq.PUB)
        sock.setsockopt(zmq.SNDHWM, qos.depth)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(xsub_endpoint())

        pub = Publisher(topic, sock, qos)
        self._pubs[topic.name] = pub
        return pub

    def subscriber(
        self,
        topic: Topic,
        qos: QoSProfile = _qos.STREAM,
        callback: Optional[Callable] = None,
    ) -> Subscriber:
        """Create a Subscriber for ``topic``.

        Each call returns a *new* Subscriber even for the same topic — that
        mirrors the legacy ``messageHandlerSubscriber`` semantics where two
        threads in one process could each hold their own poller.
        """
        sock = _context().socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, qos.depth)
        if qos.conflate:
            sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.LINGER, 0)
        # Filter includes the NUL sentinel so ``/sensors/imu`` doesn't
        # accidentally match ``/sensors/imu_raw``. See publisher.py for
        # the wire-format rationale.
        sock.setsockopt(zmq.SUBSCRIBE, topic.name.encode("utf-8") + b"\x00")
        sock.connect(xpub_endpoint())

        sub = Subscriber(topic, sock, qos, callback)
        self._subs.append(sub)
        return sub

    def close(self) -> None:
        """Close every socket this Node opened. Idempotent."""
        for pub in self._pubs.values():
            pub.close()
        self._pubs.clear()
        for sub in self._subs:
            sub.close()
        self._subs.clear()
