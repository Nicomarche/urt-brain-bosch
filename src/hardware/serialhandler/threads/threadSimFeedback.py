# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES…  (full BSD-3 — see threadRead.py).

"""threadSimFeedback — synthetic Nucleo feedback for simulator mode.

Why this thread exists
----------------------
On the real car, the Nucleo STM32 sends a continuous serial stream with
encoder speed, steering angle, and BNO055 IMU readings. ``threadRead``
parses it and publishes ``CURRENT_SPEED`` / ``CURRENT_STEER`` / ``IMU_DATA``
onto the IPC queues that the rest of the brain (relocalization,
line-following, dashboard…) subscribes to.

Without that feedback the brain's ``SafetyGate`` (post-Phase-6 single-writer
refactor) cannot validate pose freshness and falls back to ``(0, 0)`` —
the car never moves in sim no matter what AUTO mode tries to do.

In sim, there is no Nucleo. The ``sim_bridge.py`` running alongside Gazebo
synthesises the same signals from the bicycle model it already uses to
drive ``cmd_vel`` and publishes them on a dedicated ZMQ PUB socket
(``ZMQ_FEEDBACK_ENDPOINT``, default ``tcp://localhost:5577``, topic
``b"feedback"``). This thread is the brain-side counterpart: it subscribes
to that channel and translates the JSON payload into the *exact same*
queue messages ``threadRead`` would have produced on hardware — so every
downstream consumer (relocalization, dashboard, line-following, EKF…)
keeps working unchanged.

Lifecycle
---------
Spawned by ``processSerialHandler._init_threads`` only when
``MOTOR_OUTPUT == "zmq"``. Stays paired with ``threadRead`` (which still
runs but no-ops because ``serialCon`` is None in sim mode) — keeping the
two side-by-side avoids special-casing the rest of the code that just
expects "the serial process is alive".
"""

from __future__ import annotations

import ast
import json
import time
from typing import Optional

from src.templates.threadwithstop import ThreadWithStop
from src.core.bus.topics import CURRENT_SPEED, CURRENT_STEER, IMU_DATA
from src.core.messaging.messageHandlerSender import messageHandlerSender


class threadSimFeedback(ThreadWithStop):
    """Subscribe to ``sim_bridge`` feedback PUB and re-emit Nucleo-shaped IPC.

    The pause cadence (``20 ms``) matches ``threadRead`` so the consumer
    side sees the same backpressure profile in sim and on hardware. The
    actual ZMQ poll inside ``thread_work`` uses a 100 ms timeout — that
    bounds latency on a single missed publish without burning CPU when
    the bridge is silent (e.g. while Gazebo is starting up).
    """

    # Default topic kept here too so a misconfigured ``config.py`` (older
    # checkout) still works — the bridge's PUB always uses b"feedback".
    DEFAULT_TOPIC = b"feedback"
    POLL_TIMEOUT_MS = 100

    def __init__(self, queueList, logger, debugger: bool = False):
        super().__init__(pause=0.02)
        self.queuesList = queueList
        self.logger = logger
        self.debugger = debugger

        # Lazy-import zmq so the brain still imports cleanly on machines
        # without pyzmq installed (we'd never *run* this thread there, but
        # the import would fail at module-load otherwise).
        import zmq
        self._zmq = zmq

        # Pull endpoint/topic from config; fall back to hard-coded defaults
        # so a stripped-down test config doesn't break.
        try:
            from config import ZMQ_FEEDBACK_ENDPOINT, ZMQ_FEEDBACK_TOPIC
            self._endpoint = ZMQ_FEEDBACK_ENDPOINT
            self._topic = ZMQ_FEEDBACK_TOPIC
        except ImportError:
            self._endpoint = "tcp://localhost:5577"
            self._topic = self.DEFAULT_TOPIC

        self._ctx = zmq.Context.instance()
        self._sock: Optional["zmq.Socket"] = None
        self._poller: Optional["zmq.Poller"] = None

        self._init_senders()

        # Stats: log every N seconds so a silent feedback channel is
        # noticeable without spamming the console.
        self._msgs = 0
        self._last_stats_t = time.time()
        self._stats_interval_s = 5.0

    # ------------------------------------------------------------------
    def _init_senders(self) -> None:
        """Mirror ``threadRead._init_senders`` for the three feedback streams."""
        self.currentSpeedSender = messageHandlerSender(self.queuesList, CURRENT_SPEED)
        self.currentSteerSender = messageHandlerSender(self.queuesList, CURRENT_STEER)
        self.imuDataSender = messageHandlerSender(self.queuesList, IMU_DATA)

    # ------------------------------------------------------------------
    def _ensure_socket(self) -> bool:
        """Open the SUB socket on first use; return False on failure."""
        if self._sock is not None:
            return True
        try:
            self._sock = self._ctx.socket(self._zmq.SUB)
            self._sock.connect(self._endpoint)
            self._sock.setsockopt(self._zmq.SUBSCRIBE, self._topic)
            # Drop messages older than the buffer instead of growing
            # unbounded — at 50 Hz a 50-message HWM = 1 s of slack.
            try:
                self._sock.setsockopt(self._zmq.RCVHWM, 50)
            except Exception:
                pass
            self._poller = self._zmq.Poller()
            self._poller.register(self._sock, self._zmq.POLLIN)
            print(
                "\033[1;97m[ Sim Feedback ] :\033[0m "
                f"\033[1;92mINFO\033[0m - Subscribed to "
                f"\033[94m{self._endpoint}\033[0m topic="
                f"{self._topic.decode('utf-8', errors='replace')}"
            )
            return True
        except Exception as exc:
            print(
                "\033[1;97m[ Sim Feedback ] :\033[0m "
                f"\033[1;91mERROR\033[0m - Could not open SUB on "
                f"{self._endpoint}: {exc}"
            )
            self._sock = None
            self._poller = None
            return False

    # ------------------------------------------------------------------
    def _emit_feedback(self, payload: dict) -> None:
        """Translate one ``feedback`` JSON message into IPC sender calls.

        Field map (sim_bridge → brain):
            speed_mmps  → CURRENT_SPEED   (float, mm/s — divided by 1000 in
                                          relocalization._parse_speed_mps)
            steer_x10   → CURRENT_STEER   (float, tenths of degree, +=right —
                                          divided by 10 in _parse_steer_rad)
            yaw_deg     → IMU_DATA["yaw"] (degrees in brain_map; downstream
                                          parsing applies config-driven sign)
            roll/pitch/accel{x,y,z}      → forwarded so the dashboard's IMU
                                          panel doesn't show empty fields.
        """
        speed_mmps = payload.get("speed_mmps")
        steer_x10 = payload.get("steer_x10")

        if speed_mmps is not None:
            try:
                self.currentSpeedSender.send(float(speed_mmps))
            except (TypeError, ValueError):
                pass

        if steer_x10 is not None:
            try:
                self.currentSteerSender.send(float(steer_x10))
            except (TypeError, ValueError):
                pass

        # Build the same dict-string shape that threadRead produces from
        # the Nucleo's ``imu:R;P;Y;Ax;Ay;Az`` line. ``relocalization`` reads
        # the message back via ``ast.literal_eval(str(...))`` (see line
        # 1269 of relocalization_thread.py), so the dict's *string* repr
        # matters — we keep numeric values as floats so str() yields a
        # parseable Python literal.
        # `sim_bridge` ya publica yaw absoluto en `brain_map`, así que acá
        # sólo lo reenviamos en forma Nucleo-shaped para el resto del brain.
        yaw_deg = float(payload.get("yaw_deg", 0.0))

        imu_payload = {
            "roll": float(payload.get("roll_deg", 0.0)),
            "pitch": float(payload.get("pitch_deg", 0.0)),
            "yaw": yaw_deg,
            "accelx": float(payload.get("accelx", 0.0)),
            "accely": float(payload.get("accely", 0.0)),
            "accelz": float(payload.get("accelz", 9.81)),
        }
        try:
            self.imuDataSender.send(str(imu_payload))
        except Exception as exc:
            if self.debugger:
                print(f"[ Sim Feedback ] imu send failed: {exc}")

    # ------------------------------------------------------------------
    def _maybe_log_stats(self) -> None:
        now = time.time()
        dt = now - self._last_stats_t
        if dt < self._stats_interval_s:
            return
        rate = self._msgs / dt if dt > 0 else 0.0
        if self._msgs == 0:
            # 0 msg/s con sim_bridge publicando a 50Hz → el ZMQ subscriber se
            # quedó silenciosamente desincronizado (visto en sim local: tras el
            # primer "Subscribed" inicial los mensajes dejan de llegar incluso
            # con el bridge sano publicando). Forzamos cierre+reapertura del
            # socket: la próxima `_ensure_socket()` rearma la suscripción.
            print(
                "\033[1;97m[ Sim Feedback ] :\033[0m "
                "\033[1;93mWARNING\033[0m - 0 msg/s from sim_bridge "
                f"({self._endpoint}) — reabriendo SUB socket."
            )
            self._close_socket()
        elif self.debugger:
            print(f"[ Sim Feedback ] {rate:.1f} msg/s")
        self._msgs = 0
        self._last_stats_t = now

    # ------------------------------------------------------------------
    def thread_work(self) -> None:
        """Poll the SUB socket and forward one or more pending messages."""
        # Owner-thread cleanup: when ``stop()`` flips ``_blocker``, the
        # next tick exits without touching the socket from another thread
        # (closing a zmq socket from a non-owner thread races with poll()
        # and surfaces as ``Socket operation on non-socket``). We close
        # here, inside the same thread that polled, just before returning.
        if self._blocker.is_set():
            self._close_socket()
            return

        if not self._ensure_socket():
            # Back off so we don't hammer the log when sim_bridge is down.
            self._blocker.wait(0.5)
            return

        try:
            socks = dict(self._poller.poll(timeout=self.POLL_TIMEOUT_MS))
        except Exception as exc:
            if self.debugger:
                print(f"[ Sim Feedback ] poll error: {exc}")
            return

        if self._sock not in socks:
            self._maybe_log_stats()
            return

        # Drain the socket — if multiple feedback frames buffered up
        # (e.g. brain GIL hiccup), forwarding them all keeps the encoder
        # signal continuous instead of skipping ahead by 20-30 ms.
        while True:
            try:
                parts = self._sock.recv_multipart(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            except Exception as exc:
                if self.debugger:
                    print(f"[ Sim Feedback ] recv error: {exc}")
                break

            if len(parts) < 2:
                continue
            try:
                payload = json.loads(parts[1].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Tolerate occasional Python-repr style payloads in case
                # someone wires a producer that sends ``str(dict)`` instead
                # of JSON. Better to forward malformed-but-parseable than
                # to silently drop the frame.
                try:
                    payload = ast.literal_eval(parts[1].decode("utf-8", errors="replace"))
                except Exception:
                    continue
            if isinstance(payload, dict):
                self._emit_feedback(payload)
                self._msgs += 1

        self._maybe_log_stats()

    # ------------------------------------------------------------------
    def _close_socket(self) -> None:
        """Close the SUB socket from the owning thread (race-free)."""
        try:
            if self._sock is not None:
                self._sock.close(linger=0)
        except Exception:
            pass
        self._sock = None
        self._poller = None

    # ------------------------------------------------------------------
    # ``stop()`` is intentionally NOT overridden: the base ``ThreadWithStop``
    # implementation just sets ``_blocker``; the next ``thread_work`` tick
    # runs in the *owning* thread and calls ``_close_socket`` itself. This
    # keeps zmq's socket-ownership invariants intact.
