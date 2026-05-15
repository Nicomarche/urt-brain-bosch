from __future__ import annotations

import math
from dataclasses import dataclass

from src.core.types.lidar import LidarPoint


LD19_HEADER = 0x54
LD19_VER_LEN = 0x2C
LD19_PACKET_SIZE = 47
LD19_POINTS_PER_PACKET = 12


@dataclass(frozen=True)
class LD19Packet:
    speed_deg_s: float
    start_angle_deg: float
    end_angle_deg: float
    timestamp_ms: int
    points: tuple[LidarPoint, ...]


def crc8_ld19(data: bytes | bytearray | memoryview) -> int:
    """CRC-8 used by LDRobot LD19/STL-19P packets.

    The LD19 reference implementation uses polynomial 0x4D, initial value 0.
    """

    crc = 0
    for raw in bytes(data):
        crc ^= int(raw) & 0xFF
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x4D) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc & 0xFF


def parse_ld19_packet(packet: bytes | bytearray | memoryview) -> LD19Packet | None:
    """Parse one 47-byte LD19/STL-19P packet.

    Returns None when the header/length/CRC are invalid.
    """

    raw = bytes(packet)
    if len(raw) != LD19_PACKET_SIZE:
        return None
    if raw[0] != LD19_HEADER or raw[1] != LD19_VER_LEN:
        return None
    if crc8_ld19(raw[:-1]) != raw[-1]:
        return None

    speed_raw = int.from_bytes(raw[2:4], "little", signed=False)
    start_angle_deg = int.from_bytes(raw[4:6], "little", signed=False) / 100.0
    end_angle_deg = int.from_bytes(raw[42:44], "little", signed=False) / 100.0
    timestamp_ms = int.from_bytes(raw[44:46], "little", signed=False)

    points: list[LidarPoint] = []
    span_deg = end_angle_deg - start_angle_deg
    if span_deg < 0.0:
        span_deg += 360.0
    step_deg = span_deg / float(LD19_POINTS_PER_PACKET - 1)

    offset = 6
    for idx in range(LD19_POINTS_PER_PACKET):
        dist_mm = int.from_bytes(raw[offset : offset + 2], "little", signed=False)
        intensity = float(raw[offset + 2])
        angle_deg = (start_angle_deg + step_deg * idx) % 360.0
        points.append(
            LidarPoint(
                angle_rad=math.radians(angle_deg),
                distance_m=float(dist_mm) * 0.001,
                intensity=intensity,
            )
        )
        offset += 3

    return LD19Packet(
        speed_deg_s=float(speed_raw),
        start_angle_deg=float(start_angle_deg),
        end_angle_deg=float(end_angle_deg),
        timestamp_ms=int(timestamp_ms),
        points=tuple(points),
    )
