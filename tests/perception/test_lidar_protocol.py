from __future__ import annotations

import math

import pytest

from src.perception.lidar.protocol import LD19_PACKET_SIZE, crc8_ld19, parse_ld19_packet


def _packet(start_deg: float, end_deg: float, *, crc_ok: bool = True) -> bytes:
    data = bytearray(LD19_PACKET_SIZE)
    data[0] = 0x54
    data[1] = 0x2C
    data[2:4] = int(300).to_bytes(2, "little")
    data[4:6] = int(round(start_deg * 100.0)).to_bytes(2, "little")
    offset = 6
    for idx in range(12):
        data[offset : offset + 2] = int(100 + idx * 10).to_bytes(2, "little")
        data[offset + 2] = 20 + idx
        offset += 3
    data[42:44] = int(round(end_deg * 100.0)).to_bytes(2, "little")
    data[44:46] = int(1234).to_bytes(2, "little")
    data[46] = crc8_ld19(data[:-1])
    if not crc_ok:
        data[46] ^= 0xFF
    return bytes(data)


def test_ld19_crc_correct_packet_parses() -> None:
    pkt = parse_ld19_packet(_packet(10.0, 21.0))
    assert pkt is not None
    assert pkt.timestamp_ms == 1234
    assert len(pkt.points) == 12


def test_ld19_invalid_crc_returns_none() -> None:
    assert parse_ld19_packet(_packet(10.0, 21.0, crc_ok=False)) is None


def test_ld19_interpolates_angles_normally() -> None:
    pkt = parse_ld19_packet(_packet(10.0, 21.0))
    assert pkt is not None
    assert math.degrees(pkt.points[0].angle_rad) == pytest.approx(10.0)
    assert math.degrees(pkt.points[-1].angle_rad) == pytest.approx(21.0)
    assert math.degrees(pkt.points[1].angle_rad) == pytest.approx(11.0)


def test_ld19_interpolates_wrap_359_to_1() -> None:
    pkt = parse_ld19_packet(_packet(359.0, 1.0))
    assert pkt is not None
    angles = [math.degrees(point.angle_rad) % 360.0 for point in pkt.points]
    assert angles[0] == pytest.approx(359.0)
    assert angles[-1] == pytest.approx(1.0)
    assert angles[6] < 1.0 or angles[6] > 359.0
