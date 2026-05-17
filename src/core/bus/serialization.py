"""Wire-format codec for bus payloads.

Default path: ``msgpack`` (compact, fast, cross-language). It handles the
basic Python types (int, float, str, bytes, list, dict, bool, None) which is
all the new topic payloads need once we adopt frozen dataclasses + an
explicit ext-type registry in later sprints.

Escape hatch: when a topic's QoS sets ``allow_pickle=True`` (the shim path),
we fall back to ``pickle``. This preserves byte-for-byte compatibility with
the legacy gateway/Pipe wire format (which used pickle implicitly through
``multiprocessing.Queue``) and lets us migrate consumers one at a time
without rewriting payload classes first.

Sprint 5 deletes the pickle path and adds msgpack ``ext_type`` codecs for
the ~10 real payload classes we have. Until then, ``allow_pickle`` is the
documented, audited escape valve.
"""

from __future__ import annotations

import pickle
from typing import Any

import msgpack


def encode(value: Any, *, allow_pickle: bool = False) -> bytes:
    """Serialize a Python value to bytes for transit over the bus.

    Args:
        value: Anything msgpack can encode (or anything ``pickle`` can, when
            ``allow_pickle`` is True).
        allow_pickle: Use pickle protocol 4 instead of msgpack. Only enable
            this for legacy payloads — pickle on untrusted input is RCE.
    """
    if allow_pickle:
        return pickle.dumps(value, protocol=4)
    return msgpack.packb(value, use_bin_type=True)


def decode(blob: bytes, *, allow_pickle: bool = False) -> Any:
    """Deserialize bytes from the bus back into a Python value.

    Mirrors :func:`encode`'s flags. The publisher and subscriber sides MUST
    agree on ``allow_pickle`` (the QoS profile enforces this — both sides
    derive it from the same ``QoSProfile``).
    """
    if allow_pickle:
        return pickle.loads(blob)
    return msgpack.unpackb(blob, raw=False)
