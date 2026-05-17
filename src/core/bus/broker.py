"""XSUB/XPUB broker with latched-cache replay.

Topology: publishers ``SEND`` to XSUB, subscribers ``RECV`` from XPUB; the
broker shuttles data XSUB → XPUB and subscription messages XPUB → XSUB.

We do NOT use ``zmq.proxy()`` because it offers no hook to inspect or
inject frames — and we need both: cache the last value of every latched
topic on the way out, replay it to each new subscriber on the way in. The
manual ``zmq.Poller`` loop costs a few µs per event versus the C proxy,
but the data plane is sub/sub events only when topics actually fire, so
the overhead is invisible at our message rates.

Wire format (kept identical to the legacy zmq.proxy path):
  * Data frame (XSUB → broker → XPUB): ``<topic_utf8>\\x00<payload>``.
  * Subscribe message (XPUB → broker → XSUB): one byte (0x01=subscribe,
    0x00=unsubscribe) followed by the raw filter prefix the subscriber
    set with ``setsockopt(SUBSCRIBE, ...)``. In this bus the filter is
    ``<topic_utf8>\\x00`` so the subscribe frame looks like
    ``\\x01<topic_utf8>\\x00``.

Latched-cache contract:
  * The set of latched topics is owned by
    :data:`src.core.bus.topics.LATCHED_TOPIC_NAMES` (single source of
    truth — the broker does NOT import the shim).
  * On a data frame for a latched topic, we store the full frame so
    replay is a single ``xpub.send(cached)`` with no re-encoding.
  * On a subscribe message for a latched topic with a cached frame, we
    send the cached frame to XPUB BEFORE forwarding the subscribe to
    XSUB. ZMQ delivers it to the new subscriber because the broker's
    XPUB is the same socket every subscriber polls.
"""

from __future__ import annotations

import zmq

from src.core.bus import topics as _topics
from src.templates.threadwithstop import ThreadWithStop


class BrokerThread(ThreadWithStop):
    """XSUB/XPUB forwarder with latched replay, running in its own thread.

    ``run()`` is overridden (not ``thread_work``) because the poller loop
    blocks in ``poller.poll()`` and exits via socket closure. Stopping the
    thread closes XSUB which unblocks the loop.
    """

    def __init__(self, xsub_bind, xpub_bind) -> None:
        """Args:
            xsub_bind: A single endpoint or a list of endpoints. Multiple
                binds let one broker serve both ``ipc://`` (robot-internal)
                and ``tcp://`` (remote GUI) without running two brokers.
            xpub_bind: Same shape as ``xsub_bind`` for the XPUB side.
        """
        super().__init__(pause=0.0)
        self._xsub_binds: list[str] = (
            [xsub_bind] if isinstance(xsub_bind, str) else list(xsub_bind)
        )
        self._xpub_binds: list[str] = (
            [xpub_bind] if isinstance(xpub_bind, str) else list(xpub_bind)
        )
        if not self._xsub_binds or not self._xpub_binds:
            raise ValueError("BrokerThread requires at least one bind endpoint")
        self._xsub_socket = None
        self._xpub_socket = None
        # Set by ``stop()`` so ``run()`` can distinguish "I asked it to
        # close" from a genuine socket-level failure.
        self._stopping = False

    def run(self) -> None:
        ctx = zmq.Context.instance()
        self._xsub_socket = ctx.socket(zmq.XSUB)
        self._xpub_socket = ctx.socket(zmq.XPUB)
        self._xsub_socket.setsockopt(zmq.LINGER, 0)
        self._xpub_socket.setsockopt(zmq.LINGER, 0)
        # XPUB_VERBOSE=1 forwards every subscribe (not just the first per
        # topic). We need it so the replay fires for the Nth subscriber
        # of the same topic, not only the first.
        self._xpub_socket.setsockopt(zmq.XPUB_VERBOSE, 1)
        for endpoint in self._xsub_binds:
            self._xsub_socket.bind(endpoint)
        for endpoint in self._xpub_binds:
            self._xpub_socket.bind(endpoint)

        # Per-topic cache of the last frame seen on a latched topic.
        # Keyed by the encoded topic name (bytes) so the lookup matches
        # the subscribe-message parse without re-decoding.
        cache: dict[bytes, bytes] = {}
        # Read the latched set once per run(): cheap, and lets tests
        # monkeypatch ``topics.LATCHED_TOPIC_NAMES`` before starting a broker.
        latched_bytes: frozenset[bytes] = frozenset(
            name.encode("utf-8") for name in _topics.LATCHED_TOPIC_NAMES
        )

        poller = zmq.Poller()
        poller.register(self._xsub_socket, zmq.POLLIN)
        poller.register(self._xpub_socket, zmq.POLLIN)

        try:
            while True:
                events = dict(poller.poll())

                if self._xsub_socket in events:
                    # Data frame from a publisher. Forward verbatim to XPUB
                    # and, if the topic is latched, snapshot it.
                    frame = self._xsub_socket.recv()
                    self._xpub_socket.send(frame)
                    nul = frame.find(b"\x00")
                    if nul > 0:
                        topic_bytes = frame[:nul]
                        if topic_bytes in latched_bytes:
                            cache[topic_bytes] = frame

                if self._xpub_socket in events:
                    # Subscribe / unsubscribe message from a subscriber.
                    msg = self._xpub_socket.recv()
                    # Replay BEFORE forwarding the subscribe: if we forward
                    # first, the new subscriber's filter is registered and a
                    # later cached.send() would race with the next live
                    # publish. Sending the cached frame first guarantees it
                    # lands before any future data on the same topic.
                    if msg and msg[0] == 1:  # subscribe
                        # The subscribe filter is ``<topic_utf8>\x00``;
                        # strip the trailing NUL to key into the cache.
                        filter_bytes = msg[1:]
                        if filter_bytes.endswith(b"\x00"):
                            topic_bytes = filter_bytes[:-1]
                            cached = cache.get(topic_bytes)
                            if cached is not None:
                                self._xpub_socket.send(cached)
                    self._xsub_socket.send(msg)
        except (zmq.ContextTerminated, zmq.ZMQError):
            # On shutdown the parent closes XSUB which makes the poller
            # raise — ContextTerminated when the whole context goes away,
            # ZMQError(ENOTSOCK / ETERM) when only the socket does. Both
            # are noise if we asked for it; re-raise if we didn't.
            if not self._stopping:
                raise
        finally:
            self._close_sockets()

    def stop(self) -> None:
        # Closing XSUB triggers an exception inside the poller loop →
        # unblocks ``run()`` cleanly. Order matters: set ``_stopping`` so
        # ``run()`` treats the impending exception as shutdown noise, set
        # the blocker so parents don't see a zombie thread, then close the
        # socket.
        self._stopping = True
        super().stop()
        try:
            if self._xsub_socket is not None:
                self._xsub_socket.close(linger=0)
        except Exception:
            pass

    def _close_sockets(self) -> None:
        for sock_attr in ("_xsub_socket", "_xpub_socket"):
            sock = getattr(self, sock_attr, None)
            if sock is None:
                continue
            try:
                sock.close(linger=0)
            except Exception:
                pass
            setattr(self, sock_attr, None)
