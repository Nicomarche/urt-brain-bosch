"""Tests for the UDP video fragment layer."""

from __future__ import annotations

import os
import random
import socket
import threading
import time

import pytest

from src.core.bus.udp_video import (
    CODEC_JPEG,
    DEFAULT_MTU,
    FLAG_KEYFRAME,
    FLAG_LAST_FRAGMENT,
    FragmentHeader,
    FrameReassembler,
    HEADER_SIZE,
    fragment_frame,
)


def test_header_roundtrip():
    hdr = FragmentHeader(
        flags=FLAG_KEYFRAME | FLAG_LAST_FRAGMENT,
        codec=CODEC_JPEG,
        frame_id=12345,
        total=42,
        index=7,
        ts_ns=1_700_000_000_123_456_789,
    )
    blob = hdr.pack()
    assert len(blob) == HEADER_SIZE
    decoded = FragmentHeader.unpack(blob)
    assert decoded == hdr


def test_fragment_count_matches_payload():
    payload = os.urandom(10_000)
    mtu = 1400
    chunk = mtu - HEADER_SIZE
    expected = (len(payload) + chunk - 1) // chunk

    fragments = fragment_frame(payload, frame_id=1, ts_ns=0, mtu=mtu)

    assert len(fragments) == expected
    # Every fragment fits inside MTU.
    assert all(len(f) <= mtu for f in fragments)
    # The last fragment has FLAG_LAST_FRAGMENT set; only the last one does.
    flags = [FragmentHeader.unpack(f).flags & FLAG_LAST_FRAGMENT for f in fragments]
    assert flags[:-1] == [0] * (len(fragments) - 1)
    assert flags[-1] == FLAG_LAST_FRAGMENT


def test_reassembler_in_order():
    payload = os.urandom(50_000)
    fragments = fragment_frame(payload, frame_id=42, ts_ns=123, mtu=1400)
    reassembler = FrameReassembler()

    out = None
    for frag in fragments:
        out = reassembler.feed(frag)
    assert out is not None
    frame_id, body, ts_ns = out
    assert frame_id == 42
    assert ts_ns == 123
    assert body == payload


def test_reassembler_out_of_order_within_frame():
    """Fragments arriving shuffled inside a single frame must still reassemble.

    Wired LANs reorder rarely (< 1 %), but Wi-Fi APs and CPU scheduling can
    do it. Test with a 50 KB frame which produces ~36 fragments.
    """
    payload = os.urandom(50_000)
    fragments = fragment_frame(payload, frame_id=99, ts_ns=42, mtu=1400)
    shuffled = fragments[:]
    random.Random(0).shuffle(shuffled)

    reassembler = FrameReassembler()
    out = None
    for frag in shuffled:
        out = reassembler.feed(frag)
    assert out is not None
    _, body, _ = out
    assert body == payload


def test_reassembler_drops_partial_when_newer_frame_arrives():
    """Receiver in-flight with frame 1, then a fragment of frame 2 arrives —
    frame 1 must be abandoned, frame 2 reassembled normally.
    """
    payload_a = os.urandom(50_000)
    payload_b = os.urandom(30_000)
    frags_a = fragment_frame(payload_a, frame_id=1, ts_ns=10, mtu=1400)
    frags_b = fragment_frame(payload_b, frame_id=2, ts_ns=20, mtu=1400)

    reassembler = FrameReassembler()
    # Feed half of frame 1
    for frag in frags_a[: len(frags_a) // 2]:
        assert reassembler.feed(frag) is None
    # Now feed all of frame 2
    out = None
    for frag in frags_b:
        out = reassembler.feed(frag)
    assert out is not None
    frame_id, body, _ = out
    assert frame_id == 2
    assert body == payload_b


def test_reassembler_drops_older_fragments():
    """A late stray fragment from a previous frame is silently discarded."""
    payload = os.urandom(30_000)
    frags_new = fragment_frame(payload, frame_id=5, ts_ns=0, mtu=1400)
    frags_old = fragment_frame(b"old payload", frame_id=4, ts_ns=0, mtu=1400)

    reassembler = FrameReassembler()
    # Complete frame 5.
    out = None
    for frag in frags_new:
        out = reassembler.feed(frag)
    assert out is not None and out[0] == 5

    # Now feed a fragment from frame 4 — must not produce anything.
    for frag in frags_old:
        assert reassembler.feed(frag) is None


def test_udp_loopback_roundtrip():
    """End-to-end: send 100 frames over UDP localhost and reassemble.

    Verifies the whole loop (encode → fragment → sendto → recvfrom →
    reassemble). Some packet loss is acceptable; we assert that the
    *majority* of frames arrive correctly (≥ 90 %).
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", 0))
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    port = listener.getsockname()[1]

    received: list[tuple[int, bytes]] = []
    stop = threading.Event()

    def reader():
        reassembler = FrameReassembler()
        listener.settimeout(0.5)
        while not stop.is_set():
            try:
                packet, _addr = listener.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            out = reassembler.feed(packet)
            if out is not None:
                received.append((out[0], out[1]))

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)

    sent: dict[int, bytes] = {}
    for frame_id in range(100):
        # Frames of variable size to exercise the fragmenter.
        size = 20_000 + (frame_id * 200) % 30_000
        payload = os.urandom(size)
        sent[frame_id] = payload
        for fragment in fragment_frame(
            payload, frame_id=frame_id, ts_ns=time.monotonic_ns(), mtu=DEFAULT_MTU
        ):
            sender.sendto(fragment, ("127.0.0.1", port))
        # Small inter-frame pause — without it the kernel's UDP buffer
        # overflows on macOS loopback and we get heavy loss.
        time.sleep(0.005)

    time.sleep(0.3)
    stop.set()
    th.join(timeout=2.0)
    listener.close()
    sender.close()

    assert len(received) >= 90, f"only {len(received)}/100 frames survived"
    for frame_id, body in received:
        assert sent[frame_id] == body, f"corruption in frame {frame_id}"
