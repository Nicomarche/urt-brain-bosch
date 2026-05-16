from __future__ import annotations

import time

from src.perception.lidar.protocol import (
    LD19_HEADER,
    LD19_PACKET_SIZE,
    LD19_VER_LEN,
    LD19Packet,
    parse_ld19_packet,
)


# Sentinel used by callers to detect "no port configured" without catching
# generic RuntimeError. The reader refuses to autodetect: an unset port used
# to silently glob /dev/ttyACM* and steal the Nucleo's bytes from the serial
# handler, leaving the brain blind to telemetry.
class LidarPortNotConfigured(RuntimeError):
    pass


class LidarReader:
    """Serial LD19/STL-19P reader with byte-level resync."""

    def __init__(
        self,
        *,
        port: str | None = None,
        baudrate: int = 230400,
        timeout_s: float = 0.5,
    ) -> None:
        self.port = port or None
        self.baudrate = int(baudrate)
        self.timeout_s = float(timeout_s)
        self.crc_errors = 0
        self.resyncs = 0
        self._ser = None

    def open(self) -> None:
        if self._ser is not None:
            return
        if not self.port:
            raise LidarPortNotConfigured(
                "LIDAR_SERIAL_PORT not set; export URT_LIDAR_SERIAL_PORT=/dev/ttyUSBn "
                "(or set URT_LIDAR_ENABLED=0 if no LiDAR is attached)"
            )
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required for serial LiDAR backend") from exc
        self._ser = serial.Serial(
            self.port,
            self.baudrate,
            timeout=self.timeout_s,
        )

    def close(self) -> None:
        ser = self._ser
        self._ser = None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def read_packet(self) -> LD19Packet | None:
        self.open()
        ser = self._ser
        if ser is None:
            return None

        deadline = time.monotonic() + max(0.01, self.timeout_s)
        while time.monotonic() < deadline:
            first = ser.read(1)
            if not first:
                return None
            if first[0] != LD19_HEADER:
                self.resyncs += 1
                continue
            second = ser.read(1)
            if not second:
                return None
            if second[0] != LD19_VER_LEN:
                self.resyncs += 1
                continue
            rest = ser.read(LD19_PACKET_SIZE - 2)
            if len(rest) != LD19_PACKET_SIZE - 2:
                return None
            packet = bytes([LD19_HEADER, LD19_VER_LEN]) + rest
            parsed = parse_ld19_packet(packet)
            if parsed is None:
                self.crc_errors += 1
                continue
            return parsed
        return None
