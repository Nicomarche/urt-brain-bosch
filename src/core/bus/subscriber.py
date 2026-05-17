"""Subscriber side of the bus.

A ``Subscriber`` owns one ZMQ SUB socket connected to the broker's XPUB,
filtered to a single topic prefix. The API mirrors the legacy
``messageHandlerSubscriber`` so the shim can swap transports invisibly:

  * :meth:`try_recv` — non-blocking, returns ``None`` if no message ready.
  * :meth:`recv` — blocking (optionally with timeout).
  * :meth:`drain` — discard everything currently queued.

For reactive callers a future sprint will add a callback/poller-thread
option. Sprint 1 stays pull-only because every existing consumer polls.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import zmq

from src.core.bus.qos import QoSProfile
from src.core.bus.serialization import decode
from src.core.bus.topics import Topic


class Subscriber:
    """Thin wrapper around a ZMQ SUB socket filtered to one topic."""

    def __init__(
        self,
        topic: Topic,
        socket,
        qos: QoSProfile,
        callback: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self._topic = topic
        self._socket = socket
        self._qos = qos
        self._callback = callback  # reserved for the poller-thread API

    @property
    def topic(self) -> Topic:
        return self._topic

    @property
    def socket(self):
        # Exposed for advanced polling (e.g. multi-socket zmq.Poller in the
        # caller's own loop). Most callers should use try_recv / recv.
        return self._socket

    def try_recv(self) -> Any:
        """Non-blocking. Returns the decoded payload, or ``None`` if nothing
        was buffered.
        """
        try:
            frame = self._socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return None
        return self._decode_frame(frame)

    def recv(self, timeout_ms: Optional[int] = None) -> Any:
        """Block until a message arrives.

        Args:
            timeout_ms: If given, return ``None`` after this many milliseconds
                without a message. ``None`` blocks forever.
        """
        if timeout_ms is None:
            return self._decode_frame(self._socket.recv())

        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        if not dict(poller.poll(timeout_ms)).get(self._socket):
            return None
        return self._decode_frame(self._socket.recv())

    def drain(self) -> None:
        """Discard every buffered message. Use to "skip to now" after a
        long pause (e.g. when a paused thread resumes).
        """
        while self.try_recv() is not None:
            pass

    def has_data(self) -> bool:
        """Cheap peek: True if at least one message is ready to recv.

        Replacement for the legacy ``Pipe.poll()`` call exposed by
        ``messageHandlerSubscriber.is_data_in_pipe``.
        """
        return bool(self._socket.poll(0, zmq.POLLIN))

    def _decode_frame(self, frame: bytes) -> Any:
        # Wire format from ``Publisher.publish``: ``<topic>\\x00<payload>``.
        # We strip the prefix and decode the rest. The SUB filter already
        # ensured the prefix matches, so a malformed message here would
        # only come from a publisher disagreeing with the topic name —
        # surface that loudly.
        prefix_len = len(self._topic.name.encode("utf-8")) + 1  # +1 for NUL
        if len(frame) < prefix_len or frame[prefix_len - 1] != 0:
            raise RuntimeError(
                f"Subscriber({self._topic.name}): malformed wire frame"
            )
        return decode(frame[prefix_len:], allow_pickle=self._qos.allow_pickle)

    def close(self) -> None:
        try:
            self._socket.close(linger=0)
        except Exception:
            pass
