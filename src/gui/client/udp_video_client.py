"""UDP video receiver for the GUI.

Sibling of :mod:`socketio_client` but handles the **only** transport for
video frames in the new architecture: raw UDP datagrams with fragmented
JPEG payloads (see :mod:`src.core.bus.udp_video`).

The class is a :class:`PyQt5.QtCore.QObject` so it can emit Qt signals
into the UI thread. The actual socket I/O runs in a daemon
``threading.Thread`` (Qt's event loop is too coarse for tight recvfrom
loops). The thread emits two signals:

  * :attr:`camera_frame_signal` — a decoded ``QImage`` ready to draw.
    Drop-in replacement for ``SocketIOClient.camera_frame_signal``.
  * :attr:`frame_metadata_signal` — ``(frame_id, ts_ns, latency_ms)``
    for diagnostics (FPS chart, latency probe).

Decode happens inside the I/O thread because OpenCV ``imdecode`` releases
the GIL — doing it inline keeps the Qt thread free for repaints.

**Handshake** (Sprint 6): once the receive socket is bound the client
starts a second thread that publishes the GUI's host+port as a
``VideoSubscribe`` bus message every 5 s. The robot's
``UdpVideoStreamerThread`` subscribes to that topic, latches the target,
and starts streaming. A 10 s silence stops the encoder — handles GUI
crashes cleanly.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QImage

from src.core.bus.shim import messageHandlerSender
from src.core.bus.udp_video import FrameReassembler
from src.core.bus.topics import VIDEO_SUBSCRIBE

logger = logging.getLogger(__name__)


def _resolve_local_ip(robot_host: str) -> str:
    """Best-effort local IP discovery.

    Opens a dummy UDP socket "to" the robot's host — the OS picks the
    outbound interface and ``getsockname`` returns its address. No traffic
    is sent. Falls back to ``127.0.0.1`` on failure (loopback handshake
    still works in Mac dev mode).
    """
    if robot_host in ("localhost", "127.0.0.1", "::1"):
        return "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((robot_host, 9))  # discard port — never actually opens
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return "127.0.0.1"

try:
    import cv2
    import numpy as np

    _HAS_CV2 = True
except ImportError:  # pragma: no cover — monitor-mode laptops may not have cv2
    _HAS_CV2 = False


def _decode_jpeg(data: bytes) -> Optional[QImage]:
    """Convert JPEG bytes → QImage. Returns None on failure.

    Identical fallback strategy as :func:`socketio_client._decode_jpeg_bytes`:
    OpenCV when available (5–10× faster on Pi/Jetson), QImage otherwise.
    """
    if not data:
        return None
    if _HAS_CV2:
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        # OpenCV → BGR; Qt → RGB. Convert + copy so QImage owns its buffer.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        return QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    img = QImage()
    if img.loadFromData(data):
        return img
    return None


class UdpVideoClient(QObject):
    """Receives fragmented JPEG video over UDP and exposes a QImage signal.

    Lifecycle:
        client = UdpVideoClient(bind_port=0)
        client.start()                  # opens socket + spawns reader thread
        port = client.bind_port()       # tell robot via handshake
        client.camera_frame_signal.connect(camera_view._on_frame)
        ...
        client.stop()                   # idempotent
    """

    # Same name as SocketIOClient so existing widgets that hard-coded the
    # name keep working when wired to this client instead.
    camera_frame_signal = pyqtSignal(QImage)
    # (frame_id, ts_ns_at_encoder, observed_latency_ms)
    frame_metadata_signal = pyqtSignal(int, int, float)

    def __init__(
        self,
        bind_host: str = "0.0.0.0",
        bind_port: int = 0,
        robot_host: str = "127.0.0.1",
        announce_period_s: float = 5.0,
    ) -> None:
        super().__init__()
        self._bind_host = bind_host
        self._bind_port = bind_port
        self._robot_host = robot_host
        self._announce_period_s = float(announce_period_s)
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._announce_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._subscribe_sender: Optional[messageHandlerSender] = None

    def bind_port(self) -> Optional[int]:
        """Return the actually-bound UDP port (useful when ``bind_port=0``)."""
        if self._sock is None:
            return None
        return self._sock.getsockname()[1]

    def start(self) -> None:
        if self._thread is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 1 MB recv buffer: a 60 KB JPEG frame at 30 fps with 40 fragments
        # each saturates a stock 256 KB kernel buffer if the Qt thread
        # falls behind even briefly.
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        except OSError:
            pass
        sock.bind((self._bind_host, self._bind_port))
        sock.settimeout(0.5)
        self._sock = sock

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="UdpVideoClient", daemon=True
        )
        self._thread.start()

        # Handshake: announce our host+port to the robot every
        # ``announce_period_s`` so the streamer learns where to send. We
        # build the sender once (lazy ZMQ socket) and reuse it.
        self._subscribe_sender = messageHandlerSender({}, VIDEO_SUBSCRIBE)
        self._announce_thread = threading.Thread(
            target=self._announce_loop, name="UdpVideoClient-Announce", daemon=True
        )
        self._announce_thread.start()

        logger.info(
            "UdpVideoClient listening on %s:%d (announcing to robot=%s)",
            self._bind_host,
            self.bind_port(),
            self._robot_host,
        )

    def stop(self) -> None:
        self._stop_event.set()
        sock = self._sock
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
            self._sock = None
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            self._thread = None
        ann = self._announce_thread
        if ann is not None:
            ann.join(timeout=2.0)
            self._announce_thread = None

    # ------------------------------------------------------------------
    # Handshake announce thread
    # ------------------------------------------------------------------
    def _announce_loop(self) -> None:
        """Publish ``VideoSubscribe`` once at startup and every period."""
        port = self.bind_port()
        if port is None:
            return
        host = _resolve_local_ip(self._robot_host)
        fps = int(os.environ.get("URT_VIDEO_FPS", "15"))
        quality = int(os.environ.get("URT_VIDEO_JPEG_QUALITY", "70"))
        payload = {"host": host, "port": port, "fps": fps, "quality": quality}

        sender = self._subscribe_sender
        assert sender is not None

        # First publish immediately so the robot starts streaming on the
        # next frame; then refresh every ``announce_period_s`` until stop.
        while not self._stop_event.is_set():
            try:
                sender.send(payload)
            except Exception as exc:
                logger.debug("VideoSubscribe announce failed: %s", exc)
            if self._stop_event.wait(self._announce_period_s):
                break

    # ------------------------------------------------------------------
    # Reader thread
    # ------------------------------------------------------------------
    def _run(self) -> None:
        reassembler = FrameReassembler()
        sock = self._sock
        assert sock is not None
        while not self._stop_event.is_set():
            try:
                packet, _addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                # Socket closed during stop().
                return

            completed = reassembler.feed(packet)
            if completed is None:
                continue
            frame_id, jpeg_bytes, ts_ns = completed

            image = _decode_jpeg(jpeg_bytes)
            if image is None:
                logger.debug("JPEG decode failed for frame %d", frame_id)
                continue

            # Latency = receiver wallclock − encoder timestamp. ts_ns from
            # the streamer uses ``time.monotonic_ns()`` on the robot; this
            # GUI may run on a different clock domain, so the absolute
            # value is only meaningful when both ends share a host (LAN
            # path with NTP-sync'd peers gets close enough). We still
            # surface it because deltas over time still indicate jitter.
            latency_ms = max(0.0, (time.monotonic_ns() - ts_ns) / 1e6)
            self.camera_frame_signal.emit(image)
            self.frame_metadata_signal.emit(frame_id, ts_ns, latency_ms)
