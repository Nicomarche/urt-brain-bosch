"""Tests del módulo `src/core/messaging/header.py`.

Verifica:
  - `SeqCounter` produce seq monotónico creciente.
  - `stamp()` integra source + seq + timestamp.
  - `MessageHeader.to_dict() / from_dict()` es round-trip.
  - `header_from_payload()` es tolerante a payloads legacy sin header.
"""

from __future__ import annotations

import time

import pytest

from src.core.messaging.header import (
    FRAME_BASE_LINK,
    FRAME_MAP,
    MessageHeader,
    SeqCounter,
    header_from_payload,
    stamp,
)


def test_seq_counter_monotonic():
    c = SeqCounter()
    seqs = [c.next() for _ in range(5)]
    assert seqs == [0, 1, 2, 3, 4]


def test_stamp_assigns_increasing_seq_per_producer():
    c = SeqCounter()
    h1 = stamp("pose_estimator", c)
    h2 = stamp("pose_estimator", c)
    h3 = stamp("pose_estimator", c)
    assert h1.seq == 0 and h2.seq == 1 and h3.seq == 2
    assert h1.source == "pose_estimator"
    # Timestamps deben ser monotónicos (no decrecientes; pueden coincidir
    # si el reloj resuelve menos que la latencia de stamp()).
    assert h1.timestamp <= h2.timestamp <= h3.timestamp


def test_stamp_default_timestamp_is_now():
    c = SeqCounter()
    before = time.time()
    h = stamp("x", c)
    after = time.time()
    assert before <= h.timestamp <= after


def test_stamp_accepts_override_timestamp_for_replay():
    c = SeqCounter()
    h = stamp("replay_source", c, timestamp=100.0)
    assert h.timestamp == 100.0


def test_stamp_default_frame_is_map():
    c = SeqCounter()
    h = stamp("x", c)
    assert h.frame_id == FRAME_MAP


def test_stamp_accepts_custom_frame():
    c = SeqCounter()
    h = stamp("x", c, frame_id=FRAME_BASE_LINK)
    assert h.frame_id == FRAME_BASE_LINK


def test_round_trip_via_dict():
    src = MessageHeader(timestamp=12.5, seq=42, source="planner", frame_id="map")
    d = src.to_dict()
    parsed = MessageHeader.from_dict(d)
    assert parsed == src


def test_from_dict_tolerates_missing_fields():
    h = MessageHeader.from_dict({"timestamp": 1.0})
    assert h.timestamp == 1.0
    assert h.seq == 0
    assert h.source == ""
    assert h.frame_id == FRAME_MAP


def test_from_dict_tolerates_non_dict():
    h = MessageHeader.from_dict(None)
    assert h == MessageHeader()


def test_header_from_payload_with_header():
    payload = {"header": {"timestamp": 7.0, "seq": 3, "source": "p", "frame_id": "map"}}
    h = header_from_payload(payload)
    assert h.timestamp == 7.0 and h.seq == 3 and h.source == "p"


def test_header_from_payload_without_header_returns_empty():
    payload = {"x": 1.0, "y": 2.0}
    h = header_from_payload(payload)
    assert h == MessageHeader()


def test_header_from_payload_with_non_dict_returns_empty():
    assert header_from_payload(None) == MessageHeader()
    assert header_from_payload(42) == MessageHeader()
    assert header_from_payload("foo") == MessageHeader()


def test_seq_counter_independent_per_producer():
    """Cada productor mantiene su propio contador — no hay shared state."""
    c_pose = SeqCounter()
    c_planner = SeqCounter()
    stamp("pose", c_pose)
    stamp("pose", c_pose)
    h = stamp("planner", c_planner)
    assert h.seq == 0  # planner empieza en 0 aunque pose ya emitió 2
