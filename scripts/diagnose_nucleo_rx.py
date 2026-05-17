#!/usr/bin/env python3
"""Sprint 7 diagnostic: ¿la Nucleo transmite por el VCP del ST-Link?

Abre el mismo device que usa ``processSerialHandler`` con la misma config
(115200 8N1, timeout=0.1). Imprime el estado del FD (O_NONBLOCK, termios,
DTR/RTS), envía la secuencia de habilitación (#kl:30 + hallspeed + imu) y
escucha 10 s. Si llegan bytes, el firmware funciona y el problema está
en nuestro runtime; si no, el firmware no transmite (o transmite por
otro UART).

Uso (parar antes ``./run.sh``):

    URT_SERIAL_PORT=/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_...-if02 \\
        python3 scripts/diagnose_nucleo_rx.py
"""
from __future__ import annotations

import fcntl
import os
import sys
import termios
import time

import serial


def _flag_dump(fd: int) -> None:
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    print(f"  fd={fd} O_NONBLOCK={bool(fl & os.O_NONBLOCK)} O_RDWR={bool(fl & os.O_RDWR)}")
    attrs = termios.tcgetattr(fd)
    iflag, oflag, cflag, lflag, ispeed, ospeed, cc = attrs
    print(f"  iflag=0x{iflag:08x} oflag=0x{oflag:08x} cflag=0x{cflag:08x} lflag=0x{lflag:08x}")
    print(f"  ispeed={ispeed} ospeed={ospeed}")
    print(f"  ICANON={bool(lflag & termios.ICANON)} ECHO={bool(lflag & termios.ECHO)}")
    print(f"  CRTSCTS={bool(cflag & termios.CRTSCTS)} IXON={bool(iflag & termios.IXON)}")
    print(f"  VMIN={cc[termios.VMIN]} VTIME={cc[termios.VTIME]}")


def main() -> int:
    device = os.environ.get(
        "URT_SERIAL_PORT",
        "/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066DFF495177514867205427-if02",
    )
    print(f"[diag] Opening {device} @ 115200, timeout=0.1, 8N1")
    try:
        ser = serial.Serial(device, 115200, timeout=0.1)
    except serial.SerialException as exc:
        print(f"[diag] open failed: {exc}", file=sys.stderr)
        return 1

    # IMPORTANT: no reset_input_buffer here — quiero ver exactamente lo que
    # esté en el kernel buffer al momento de abrir el FD.
    fd = ser.fileno()
    print("[diag] FD state right after open():")
    _flag_dump(fd)
    print(f"[diag] DTR={ser.dtr} RTS={ser.rts}")
    print(f"[diag] in_waiting={ser.in_waiting} out_waiting={ser.out_waiting}")

    # Drenado explícito, log de cuántos bytes había.
    pre_drain = ser.in_waiting
    if pre_drain:
        chunk = ser.read(pre_drain)
        print(f"[diag] pre-drain captured {pre_drain} bytes: {chunk!r}")
    else:
        print("[diag] pre-drain: 0 bytes")

    # Encender feedback en la Nucleo (mismo orden que threadWrite tras KL=30).
    print("[diag] sending #kl:30;; #hallspeed:1;; #imu:1;;")
    for cmd in (b"#kl:30;;", b"#hallspeed:1;;", b"#imu:1;;"):
        ser.write(cmd)
        time.sleep(0.05)
    print(f"[diag] after writes: in_waiting={ser.in_waiting}")

    # Escucha 10 s, log de cualquier byte.
    t_end = time.monotonic() + 10.0
    total = 0
    sample = b""
    while time.monotonic() < t_end:
        waiting = ser.in_waiting
        if waiting > 0:
            chunk = ser.read(waiting)
            if chunk:
                total += len(chunk)
                if len(sample) < 512:
                    sample += chunk
                print(
                    f"[diag] +{len(chunk)} bytes  total={total}  in_waiting={ser.in_waiting}"
                    f"  preview={chunk[:80]!r}",
                    flush=True,
                )
        else:
            # Cada 2 s log de RX-IDLE para correlacionar con el patrón del runtime.
            time.sleep(0.05)

    print(f"[diag] === 10 s elapsed: total_bytes={total} ===")
    if total == 0:
        print("[diag] verdict: la Nucleo NO está transmitiendo por este VCP.")
        print(
            "[diag] Posibles causas: firmware no usa USART2 para TX (ST-Link"
            " VCP), jumpers de board mal puestos, cable RX/TX roto, o el"
            " firmware no responde a los toggles de hallspeed/imu hasta"
            " que KL=15 (no 30)."
        )
    else:
        print("[diag] verdict: la Nucleo SÍ transmite. El bug está en nuestro runtime.")
        print(f"[diag] sample ascii: {sample.decode('ascii', errors='replace')[:400]}")
    ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
