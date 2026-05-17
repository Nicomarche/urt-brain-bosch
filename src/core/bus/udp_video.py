"""UDP video fragmentation + reassembly.

Wire format per UDP datagram::

    +----------------------------- 20-byte header --------------------------+
    | magic u8 | version u8 | flags u8 | codec u8 | frame_id u32           |
    | total u16 | index u16 | ts_ns u64                                    |
    +-----------------------------------------------------------------------+
    | payload (≤ MTU - 20 - IP/UDP overhead) bytes                          |
    +-----------------------------------------------------------------------+

* ``magic = 0xBF`` (BFMC). Filters stray traffic on shared ports.
* ``version`` lets us evolve the wire format without silent corruption.
* ``flags``: bit0=keyframe, bit1=last_fragment, bits2..7 reserved.
* ``codec``: 0=JPEG (Sprint 2), 1=H.264 NAL (future drop-in).
* ``frame_id`` is monotonic in the encoder. Receiver uses it to detect new
  frames and drop fragments belonging to older ones.
* ``ts_ns`` is the encoder's monotonic clock at frame-encode time. Subtract
  from ``time.monotonic_ns()`` at display to measure end-to-end latency.

Reassembly contract: at most one frame in progress. A fragment with a newer
``frame_id`` discards the partial in-progress frame and starts the new one.
A fragment with an older ``frame_id`` is dropped. Reordering WITHIN a frame
is tolerated via an index-keyed dict. There is no retransmission, no NACK,
no jitter buffer — that's the whole point.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Tuple

MAGIC: int = 0xBF
VERSION: int = 1

# Network byte order ("!"). One-byte fields aren't strictly needed in
# network order but keeping the format string uniform avoids endian
# mistakes on the multi-byte ones.
_HEADER_FMT: str = "!BBBBIHHQ"
HEADER_SIZE: int = struct.calcsize(_HEADER_FMT)  # 20 bytes

# Ethernet MTU 1500 − IPv4 (20) − UDP (8) = 1472 max payload at link layer.
# We default to 1400 to leave headroom for Wi-Fi (which can negotiate
# smaller MTUs at the 802.11 layer) and VPN/tunnel overhead.
DEFAULT_MTU: int = 1400

CODEC_JPEG: int = 0
CODEC_H264_NAL: int = 1

# Flag bits — combined with bitwise OR.
FLAG_KEYFRAME: int = 0x01
FLAG_LAST_FRAGMENT: int = 0x02


@dataclass(frozen=True)
class FragmentHeader:
    """Decoded view of a fragment's 20-byte header."""

    flags: int
    codec: int
    frame_id: int
    total: int
    index: int
    ts_ns: int

    def pack(self) -> bytes:
        return struct.pack(
            _HEADER_FMT,
            MAGIC,
            VERSION,
            self.flags & 0xFF,
            self.codec & 0xFF,
            self.frame_id & 0xFFFFFFFF,
            self.total & 0xFFFF,
            self.index & 0xFFFF,
            self.ts_ns & 0xFFFFFFFFFFFFFFFF,
        )

    @staticmethod
    def unpack(data: bytes) -> "FragmentHeader":
        if len(data) < HEADER_SIZE:
            raise ValueError(f"packet too short: {len(data)} < {HEADER_SIZE}")
        magic, version, flags, codec, frame_id, total, index, ts_ns = struct.unpack_from(
            _HEADER_FMT, data, 0
        )
        if magic != MAGIC:
            raise ValueError(f"bad magic: {magic:#x} != {MAGIC:#x}")
        if version != VERSION:
            raise ValueError(f"unsupported version: {version}")
        return FragmentHeader(
            flags=flags,
            codec=codec,
            frame_id=frame_id,
            total=total,
            index=index,
            ts_ns=ts_ns,
        )


def fragment_frame(
    payload: bytes,
    frame_id: int,
    ts_ns: int,
    *,
    codec: int = CODEC_JPEG,
    keyframe: bool = True,
    mtu: int = DEFAULT_MTU,
) -> list[bytes]:
    """Slice an encoded frame into UDP-sized datagrams with headers.

    Each returned bytes object is one complete datagram (header + slice of
    payload) ready for :py:meth:`socket.socket.sendto`.
    """
    chunk_size = mtu - HEADER_SIZE
    if chunk_size <= 0:
        raise ValueError(f"MTU {mtu} ≤ header size {HEADER_SIZE}")
    total = max(1, (len(payload) + chunk_size - 1) // chunk_size)
    if total > 0xFFFF:
        raise ValueError(
            f"Frame would need {total} fragments — beyond u16 limit. "
            f"Reduce JPEG quality or raise MTU."
        )

    base_flags = (FLAG_KEYFRAME if keyframe else 0)
    datagrams: list[bytes] = []
    for index in range(total):
        start = index * chunk_size
        chunk = payload[start : start + chunk_size]
        flags = base_flags | (FLAG_LAST_FRAGMENT if index == total - 1 else 0)
        header = FragmentHeader(
            flags=flags,
            codec=codec,
            frame_id=frame_id,
            total=total,
            index=index,
            ts_ns=ts_ns,
        )
        datagrams.append(header.pack() + chunk)
    return datagrams


class FrameReassembler:
    """Single-flight reassembler.

    Holds at most one frame in progress. Behavior:
      * fragment of a *newer* frame_id → discard the in-progress frame
        (even if it was partial) and start over with the new frame.
      * fragment of an *older* frame_id → drop on the floor.
      * fragment of the *current* frame_id → store by index; on the last
        missing index → return the completed frame.
      * unexpected/garbled fragment → return None silently (the protocol
        is best-effort; logging every malformed packet would spam).

    Wraparound: ``frame_id`` is u32 (~4.3 G frames; ~3 years at 30 fps).
    For safety we compare with subtraction modulo 2^32, but in practice no
    real session will see wraparound.
    """

    def __init__(self) -> None:
        self._current_frame_id: Optional[int] = None
        self._chunks: dict[int, bytes] = {}
        self._total: int = 0
        self._earliest_ts_ns: int = 0
        self._last_completed_frame_id: Optional[int] = None

    @staticmethod
    def _is_newer(candidate: int, current: Optional[int]) -> bool:
        if current is None:
            return True
        # Signed difference mod 2^32: positive → candidate is newer.
        diff = (candidate - current) & 0xFFFFFFFF
        return 0 < diff < 0x80000000

    def feed(self, packet: bytes) -> Optional[Tuple[int, bytes, int]]:
        """Process one received datagram. Returns ``(frame_id, payload,
        ts_ns)`` once a full frame is reassembled, otherwise ``None``.
        """
        try:
            hdr = FragmentHeader.unpack(packet)
        except ValueError:
            return None
        body = packet[HEADER_SIZE:]

        # Older than the last frame we already completed → drop. Without
        # this check, a stray late fragment could restart reassembly for
        # a frame we already produced, yielding a duplicate downstream.
        if self._last_completed_frame_id is not None and not self._is_newer(
            hdr.frame_id, self._last_completed_frame_id
        ):
            return None

        # Older than in-progress frame → drop.
        if (
            self._current_frame_id is not None
            and hdr.frame_id != self._current_frame_id
            and not self._is_newer(hdr.frame_id, self._current_frame_id)
        ):
            return None

        # Newer than in-progress (or first ever) → restart.
        if hdr.frame_id != self._current_frame_id:
            self._current_frame_id = hdr.frame_id
            self._chunks = {}
            self._total = hdr.total
            self._earliest_ts_ns = hdr.ts_ns

        # Same frame as in-progress → accumulate.
        # Track the earliest ts_ns we saw for this frame (encoder sends the
        # same value in every fragment, but if the encoder is buggy we use
        # the smaller one as a sane default).
        if hdr.ts_ns < self._earliest_ts_ns or self._earliest_ts_ns == 0:
            self._earliest_ts_ns = hdr.ts_ns
        self._chunks[hdr.index] = body

        if len(self._chunks) >= self._total:
            # Assemble in index order.
            try:
                assembled = b"".join(self._chunks[i] for i in range(self._total))
            except KeyError:
                # We somehow have ``total`` chunks but a gap — should not
                # happen because indices are unique, but defensive.
                return None
            completed_id = self._current_frame_id
            ts_ns = self._earliest_ts_ns
            self._last_completed_frame_id = completed_id
            self._current_frame_id = None
            self._chunks = {}
            self._total = 0
            self._earliest_ts_ns = 0
            return completed_id, assembled, ts_ns
        return None
