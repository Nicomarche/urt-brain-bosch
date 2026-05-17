"""Back-compat façade for the legacy ``messageHandlerSender`` API.

Hundreds of call sites scattered across the codebase still import::

    from src.core.messaging.messageHandlerSender import messageHandlerSender
    from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber

This shim keeps that API alive on top of the new pub/sub bus so the call
sites don't need to know about Nodes, Publishers, or Subscribers. The only
caller-visible difference is the second positional argument: it used to be
a legacy ``allMessages`` enum class, now it is a :class:`Topic` constant
declared in :mod:`src.core.bus.topics`.

``queuesList`` is accepted in the constructor but ignored — the bus does
not need the dict of ``multiprocessing.Queue`` objects, the broker routes
by topic name. We keep the parameter in the signature so call-site
migration is a one-line import change, not a constructor rewrite.

Spawn safety: ZMQ sockets cannot cross a ``multiprocessing.spawn``
boundary. Each shim object defers socket creation until first I/O, so a
sender or subscriber built in the parent process and used in a child works
correctly with the child's own ZMQ context.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any

import zmq

from src.core.bus import qos as _qos
from src.core.bus.node import Node
from src.core.bus.qos import QoSProfile
from src.core.bus.topics import Topic


# ─── QoS preset name → QoSProfile ──────────────────────────────────────────

_QOS_PRESETS: dict[str, QoSProfile] = {
    "STREAM": _qos.STREAM,
    "COMMAND": _qos.COMMAND,
    "LATCHED": _qos.LATCHED,
    "EVENT": _qos.EVENT,
}


def _resolve_qos(preset: str) -> QoSProfile:
    try:
        return _QOS_PRESETS[preset]
    except KeyError as exc:
        raise ValueError(
            f"Unknown QoS preset {preset!r}; expected one of {sorted(_QOS_PRESETS)}"
        ) from exc


# ─── Per-process Node singleton ─────────────────────────────────────────────

_NODE_LOCK = threading.Lock()
_NODE: Node | None = None


def _process_node() -> Node:
    """One shim-owned Node per process. All shim-created sockets share it
    so a single ``Node.close()`` can tear them down on shutdown if needed.
    """
    global _NODE
    if _NODE is not None:
        return _NODE
    with _NODE_LOCK:
        if _NODE is None:
            _NODE = Node(name="shim")
    return _NODE


# ─── Legacy API drop-ins ────────────────────────────────────────────────────


class messageHandlerSender:
    """Drop-in replacement for ``src.core.messaging.messageHandlerSender``.

    Same constructor signature shape, same ``send()`` method.
    ``queuesList`` is accepted and ignored. The ``topic`` argument is now a
    :class:`Topic` constant from :mod:`src.core.bus.topics`.
    """

    def __init__(self, queuesList: Any, topic: Topic) -> None:
        # ``allow_pickle`` keeps byte-for-byte wire compat with the legacy
        # gateway/Pipe pickle path. Legacy payloads include numpy arrays,
        # dataclasses, and bytes that msgpack can't encode natively.
        self._topic = topic
        self._qos = replace(_resolve_qos(topic.qos_preset), allow_pickle=True)
        self._pub = None  # lazy — see _ensure_pub()

    def _ensure_pub(self):
        if self._pub is None:
            self._pub = _process_node().publisher(self._topic, self._qos)
        return self._pub

    def send(self, value: Any) -> None:
        self._ensure_pub().publish(value)

    def __getstate__(self):
        # Drop the (possibly socket-backed) Publisher before pickling so
        # the child rebuilds it lazily with its own ZMQ context.
        state = self.__dict__.copy()
        state["_pub"] = None
        return state


class messageHandlerSubscriber:
    """Drop-in replacement for ``src.core.messaging.messageHandlerSubscriber``.

    Same signature and methods. ``deliveryMode="lastonly"`` is honored by
    flipping the QoS to ``conflate=True, depth=1`` regardless of the topic's
    declared preset.
    """

    def __init__(
        self,
        queuesList: Any,
        topic: Topic,
        deliveryMode: str = "fifo",
        subscribe: bool = False,
    ) -> None:
        qos_p = replace(_resolve_qos(topic.qos_preset), allow_pickle=True)
        if str(deliveryMode).lower() == "lastonly":
            qos_p = replace(qos_p, conflate=True, depth=1)
        self._topic = topic
        self._qos = qos_p
        self._sub = None  # lazy — see _ensure_sub()
        # ``subscribe`` flag is a no-op: subscription happens at socket
        # creation time. The legacy gateway needed the flag because
        # subscription was a separate message; with ZMQ filtering we're
        # subscribed the moment the socket is bound.

    def _ensure_sub(self):
        if self._sub is None:
            self._sub = _process_node().subscriber(self._topic, qos=self._qos)
        return self._sub

    def receive(self) -> Any:
        return self._ensure_sub().try_recv()

    def receive_with_block(self) -> Any:
        return self._ensure_sub().recv()

    def empty(self) -> None:
        self._ensure_sub().drain()

    def subscribe(self) -> None:
        # Materialize the underlying socket eagerly. Some callers pass
        # ``subscribe=True`` precisely to confirm the subscription is live
        # at construction time, even though the ZMQ contract has no notion
        # of a subscribe message we need to drive ourselves.
        self._ensure_sub()

    def unsubscribe(self) -> None:
        sub = self._sub
        if sub is not None:
            sub.close()
            self._sub = None

    def is_data_in_pipe(self) -> bool:
        try:
            return self._ensure_sub().has_data()
        except zmq.ZMQError:
            return False

    def set_delivery_mode_to_fifo(self) -> None:
        # Legacy ``messageHandlerSubscriber`` allowed switching delivery
        # mode mid-flight. In ZMQ that's a socket option (CONFLATE) which
        # can't be changed after the first send. No live caller flips this
        # at runtime — they pick a mode in ``__init__`` and stick with it.
        raise NotImplementedError(
            "Delivery mode is fixed at construction time in the new bus. "
            "Re-create the subscriber with deliveryMode='fifo' if needed."
        )

    def set_delivery_mode_to_last_only(self) -> None:
        raise NotImplementedError(
            "Delivery mode is fixed at construction time in the new bus. "
            "Re-create the subscriber with deliveryMode='lastonly' if needed."
        )

    def __getstate__(self):
        # Same pickling story as the sender — drop the socket-backed
        # Subscriber before crossing a process boundary.
        state = self.__dict__.copy()
        state["_sub"] = None
        return state

    def __del__(self) -> None:
        sub = getattr(self, "_sub", None)
        if sub is not None:
            try:
                sub.close()
            except Exception:
                pass
