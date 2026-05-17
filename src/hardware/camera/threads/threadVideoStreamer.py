"""UDP video streamer thread.

Replaces the legacy ``serialCamera`` queue/Pipe/SocketIO path. Lives in
``processCamera`` (next to ``threadLocalPerception``) because the overlay
frame already exists there — eliminating an inter-process hop for video.

Pipeline::

    threadLocalPerception
        ↓ writes annotated frame to overlay_buffer (LatestFrameBuffer)
    UdpVideoStreamerThread (this file)
        ↓ reads frame, JPEG-encodes, fragments, sendto()
    UdpVideoClient on the GUI PC
        ↓ recvfrom, reassemble, emit Qt signal

**Activation** (Sprint 6): the GUI publishes a ``VideoSubscribe`` bus
message every 5 s with its host+port. We subscribe to that topic and
latch the target. ``URT_VIDEO_TARGET_HOST/PORT`` env vars still work as
a manual override for headless test rigs. After 10 s without a refresh
from the GUI we clear the target — a crashed GUI stops the encoder.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import Optional, Tuple

from src.core.bus.shim import messageHandlerSubscriber
from src.core.bus.udp_video import DEFAULT_MTU, fragment_frame
from src.core.bus.video_encoder import JpegEncoder, VideoEncoder
from src.core.bus.topics import VIDEO_SUBSCRIBE
from src.core.messaging.buffers import LatestFrameBuffer
from src.templates.threadwithstop import ThreadWithStop


def _resolve_initial_target() -> Optional[Tuple[str, int]]:
    """Read ``URT_VIDEO_TARGET_HOST`` + ``URT_VIDEO_TARGET_PORT`` from env.

    Manual override for headless/test rigs that don't run the GUI.
    Production discovery uses the ``VideoSubscribe`` handshake.
    """
    host = os.environ.get("URT_VIDEO_TARGET_HOST", "").strip()
    port_raw = os.environ.get("URT_VIDEO_TARGET_PORT", "").strip()
    if not host or not port_raw:
        return None
    try:
        return host, int(port_raw)
    except ValueError:
        return None


class UdpVideoStreamerThread(ThreadWithStop):
    """Pulls newest frame from a buffer, encodes, fragments, sends UDP.

    Args:
        frame_buffer: ``LatestFrameBuffer`` that the upstream thread writes
            into. The streamer never blocks on it — when there's no new
            frame it just sleeps until the next FPS tick.
        target: Optional ``(host, port)``. Overrides env vars.
        target_fps: Frames per second cap. Defaults to ``URT_VIDEO_FPS``
            env var (default 15). Set higher to push more video; the
            actual rate is min(target_fps, source_fps).
        mtu: Maximum UDP datagram size including header.
            ``URT_VIDEO_MTU`` env var overrides.
        encoder: Encoder. Defaults to ``JpegEncoder(quality=70)``.
    """

    # If we haven't heard a ``VideoSubscribe`` refresh from the GUI in this
    # many seconds we assume it died and stop encoding. 10 s = 2× the
    # GUI's announce period (5 s) — one missed announce is tolerated.
    SUBSCRIBE_TIMEOUT_S: float = 10.0

    def __init__(
        self,
        frame_buffer: LatestFrameBuffer,
        target: Optional[Tuple[str, int]] = None,
        target_fps: Optional[float] = None,
        mtu: Optional[int] = None,
        encoder: Optional[VideoEncoder] = None,
    ) -> None:
        super().__init__(pause=0.0)  # we manage our own pacing
        self._buffer = frame_buffer
        # Env-var target only used when no GUI handshake will arrive
        # (headless QA rigs). Treated as eternal — never times out.
        self._env_target: Optional[Tuple[str, int]] = target or _resolve_initial_target()
        self._target: Optional[Tuple[str, int]] = self._env_target
        self._target_lock = threading.Lock()
        self._last_subscribe_at: float = 0.0  # set only on GUI announces

        fps_default = float(os.environ.get("URT_VIDEO_FPS", "15"))
        self._target_fps = float(target_fps) if target_fps else fps_default
        self._frame_interval = 1.0 / max(1.0, self._target_fps)

        self._mtu = int(mtu) if mtu else int(os.environ.get("URT_VIDEO_MTU", str(DEFAULT_MTU)))
        self._encoder: VideoEncoder = encoder or JpegEncoder(
            quality=int(os.environ.get("URT_VIDEO_JPEG_QUALITY", "70"))
        )

        # GUI handshake: subscribe to ``VideoSubscribe`` and latch on receive.
        # Sender is in another process — slow-joiner doesn't matter because
        # the GUI publishes every 5 s.
        self._subscribe_sub = messageHandlerSubscriber(
            {}, VIDEO_SUBSCRIBE, "lastonly", subscribe=True
        )
        self._subscribe_sub.subscribe()

        self._sock: Optional[socket.socket] = None
        self._frame_id: int = 0
        self._last_sent_sequence: int = -1
        self._last_send_time: float = 0.0

    def set_target(self, host: str, port: int) -> None:
        """Update the destination atomically.

        Called by the GUI handshake (``VideoSubscribe`` topic) or manually
        from tests / scripts. Refreshes the subscribe-watchdog timestamp so
        the streamer keeps publishing.
        """
        with self._target_lock:
            self._target = (host, int(port))
            self._last_subscribe_at = time.monotonic()

    def clear_target(self) -> None:
        """Stop sending. Subsequent ticks do nothing until ``set_target``.

        Falls back to the env-var target if one was configured (manual
        override for headless rigs survives even when no GUI is connected).
        """
        with self._target_lock:
            self._target = self._env_target
            self._last_subscribe_at = 0.0

    def _get_target(self) -> Optional[Tuple[str, int]]:
        with self._target_lock:
            return self._target

    def _drain_subscribe_messages(self) -> None:
        """Pull every queued ``VideoSubscribe`` from the bus and latch the
        newest target. Called once per ``thread_work`` tick — cheap because
        the subscriber is non-blocking and the GUI publishes only ~0.2 Hz.
        """
        latest = None
        while True:
            msg = self._subscribe_sub.receive()
            if msg is None:
                break
            if isinstance(msg, dict):
                latest = msg
        if latest is None:
            return
        host = str(latest.get("host") or "").strip()
        try:
            port = int(latest.get("port"))
        except (TypeError, ValueError):
            return
        if not host or port <= 0:
            return
        self.set_target(host, port)

    def _check_subscribe_timeout(self) -> None:
        """Drop the target if the GUI has stopped announcing.

        Env-var targets are exempt — they're for unattended test rigs.
        """
        with self._target_lock:
            if self._last_subscribe_at == 0.0:
                return  # no GUI handshake ever arrived; keep env target alive
            if time.monotonic() - self._last_subscribe_at > self.SUBSCRIBE_TIMEOUT_S:
                self._target = self._env_target
                self._last_subscribe_at = 0.0

    def thread_work(self) -> None:
        self._drain_subscribe_messages()
        self._check_subscribe_timeout()

        target = self._get_target()
        if target is None:
            # No subscriber — sleep briefly. Skipping the read avoids a
            # pointless lock-acquire on the buffer every tick.
            self._blocker.wait(0.1)
            return

        # Rate limit. We sleep until the next interval rather than busy-loop.
        now = time.monotonic()
        sleep_for = self._last_send_time + self._frame_interval - now
        if sleep_for > 0:
            self._blocker.wait(sleep_for)
            return

        frame, _ts, sequence = self._buffer.read_latest(copy_frame=False)
        if frame is None or sequence == self._last_sent_sequence:
            # Nothing new — fall through to the next tick. Sleeping a
            # quarter of the interval avoids tight-looping when the camera
            # source pauses.
            self._blocker.wait(self._frame_interval / 4)
            return

        if self._sock is None:
            self._sock = self._make_socket()

        ts_ns = time.monotonic_ns()
        for packet, keyframe in self._encoder.encode(frame, ts_ns):
            datagrams = fragment_frame(
                packet,
                frame_id=self._frame_id,
                ts_ns=ts_ns,
                codec=self._encoder.codec_id,
                keyframe=keyframe,
                mtu=self._mtu,
            )
            for dgram in datagrams:
                try:
                    self._sock.sendto(dgram, target)
                except OSError:
                    # ENOBUFS, host unreachable, network down — drop the
                    # whole frame and keep going. We deliberately don't
                    # break the loop on the first failure because OS
                    # buffer state can change mid-frame.
                    pass
            self._frame_id = (self._frame_id + 1) & 0xFFFFFFFF
        self._last_sent_sequence = sequence
        self._last_send_time = time.monotonic()

    def _make_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 1 MB send buffer absorbs bursts when the kernel scheduler
        # falls behind. Without it, large frames can drop fragments at
        # the kernel boundary even before they hit the wire.
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        except OSError:
            pass
        return sock

    def stop(self) -> None:
        super().stop()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
