"""Publisher side of the bus.

A ``Publisher`` owns one ZMQ PUB socket connected to the broker's XSUB.
Each ``publish()`` sends one frame with the wire format::

    <topic_name_utf8>\\x00<payload_bytes>

Why single-frame (not multipart): ZMQ's ``ZMQ_CONFLATE`` option silently
discards multipart messages — only single-frame messages are conflated.
Our ``STREAM`` QoS preset sets ``CONFLATE=1`` to give the legacy
"newest-wins" behavior the gateway implemented with
``_AsyncPipeSender(maxsize=1)``, so we MUST keep the wire format single-
frame.

The trailing ``\\x00`` is a sentinel so subscriber filters distinguish
``/sensors/imu`` from ``/sensors/imu_raw`` — without it, ZMQ's
prefix-based filtering would let the shorter subscription accidentally
match longer topic names.
"""

from __future__ import annotations

from typing import Any

from src.core.bus.qos import QoSProfile
from src.core.bus.serialization import encode
from src.core.bus.topics import Topic


class Publisher:
    """Thin wrapper around a ZMQ PUB socket.

    Constructed by :meth:`Node.publisher` — callers should not build this
    directly because the Node owns socket lifecycle (LINGER, close on
    shutdown).
    """

    def __init__(self, topic: Topic, socket, qos: QoSProfile) -> None:
        self._topic = topic
        self._socket = socket
        self._qos = qos
        # Precomputed prefix used on every publish. The trailing NUL is what
        # makes subscriber filters unambiguous (see module docstring).
        self._prefix = topic.name.encode("utf-8") + b"\x00"

    @property
    def topic(self) -> Topic:
        return self._topic

    def publish(self, value: Any) -> None:
        """Serialize and send a message on this topic.

        Backpressure behavior depends on the socket's HWM and the
        ``reliability`` setting in the QoS profile:
          * ``best_effort`` + HWM full → ZMQ silently drops (default).
          * ``reliable``    + HWM full → ZMQ blocks until space is available
            (or raises ``zmq.Again`` if the socket is in non-blocking mode).
        The Node configures the socket according to the QoS, so this method
        does not need to branch.
        """
        payload = encode(value, allow_pickle=self._qos.allow_pickle)
        self._socket.send(self._prefix + payload)

    def close(self) -> None:
        """Idempotent. Sockets are closed with LINGER=0 by the Node."""
        try:
            self._socket.close(linger=0)
        except Exception:
            pass
