"""QoS profiles for the pub/sub bus.

A QoSProfile is the contract between a publisher and its subscribers for a
single topic. We expose four presets (STREAM/COMMAND/LATCHED/EVENT) that map
the four message families we have today onto ZMQ socket options. Code outside
this module SHOULD use the presets, not build QoSProfile by hand — that keeps
the semantics consistent across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QoSProfile:
    """Quality-of-Service knobs that map onto pyzmq socket options.

    Attributes:
        reliability: ``"reliable"`` → publisher blocks (or fails) if HWM is
            full; ``"best_effort"`` → publisher silently drops the message.
            Default ``best_effort`` because lossy is the right default for
            sensor streams (a stalled camera thread is worse than a missed
            frame).
        depth: high-water-mark. Maps to ZMQ_SNDHWM and ZMQ_RCVHWM. Use 1 plus
            ``conflate`` for "last value wins"; 8 for command queues (absorbs
            brief consumer pauses without unbounded growth); 256 for log
            streams (cheap msgs, want history).
        conflate: subscriber-side ZMQ_CONFLATE=1 — every ``recv()`` returns
            the newest unread message; older ones are dropped. Replaces the
            ``_AsyncPipeSender(maxsize=1)`` trick the legacy gateway used.
        latched: republish the last value on this topic to any late
            subscriber. ROS calls this "transient_local"; required for state
            topics (Klem, system_mode) so a fresh GUI sees the current value
            without waiting for the next change. Implemented by the broker,
            not natively in ZMQ.
        allow_pickle: opt-in escape hatch for legacy payloads (the shim uses
            this during migration). Disabled by default so a typo in the
            serializer fails loudly instead of silently shipping arbitrary
            objects with pickle (= remote-code-execution on the wire).
    """

    reliability: Literal["reliable", "best_effort"] = "best_effort"
    depth: int = 1
    conflate: bool = False
    latched: bool = False
    allow_pickle: bool = False


# High-rate sensor / perception streams. Newest-wins, lossy. ImuData,
# LidarScanMsg, LocalLanePerception, DetectedObjectsMsg…
STREAM = QoSProfile(
    reliability="best_effort",
    depth=1,
    conflate=True,
    latched=False,
)

# Control plane: SpeedMotor, SteerMotor, Brake. Reliable with small buffer —
# absorbs a 5–10 ms GC pause without losing commands, but won't queue forever
# if a consumer is hung (after depth=8 the publisher drops; in practice the
# dispatcher runs at 50 Hz so 8 messages ≈ 160 ms of buffer).
COMMAND = QoSProfile(
    reliability="reliable",
    depth=8,
    conflate=False,
    latched=False,
)

# Sticky state (Klem, StateChange). Only the last value matters, but late
# subscribers must receive it on connect.
LATCHED = QoSProfile(
    reliability="reliable",
    depth=1,
    conflate=False,
    latched=True,
)

# Discrete events (logs, calibration, debug traces). Reliable with a generous
# buffer because messages are infrequent and individually informative —
# losing one obscures history.
EVENT = QoSProfile(
    reliability="reliable",
    depth=256,
    conflate=False,
    latched=False,
)
