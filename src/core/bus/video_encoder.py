"""Video encoder abstraction for the UDP streamer.

Sprint 2 ships only the JPEG encoder (``cv2.imencode``). Later sprints can
plug in NVENC H.264 on Jetson by implementing this Protocol — the streamer
does not need to change.
"""

from __future__ import annotations

from typing import Iterable, Protocol, Tuple

from src.core.bus.udp_video import CODEC_JPEG


class VideoEncoder(Protocol):
    """Encoder contract.

    Attributes:
        codec_id: numeric code embedded in the fragment header so the
            receiver knows how to decode the payload.

    Method:
        encode(frame_bgr, ts_ns) yields ``(packet_bytes, is_keyframe)``
        tuples. JPEG yields exactly one (always keyframe). H.264 may yield
        multiple NAL units per frame.
    """

    codec_id: int

    def encode(self, frame_bgr, ts_ns: int) -> Iterable[Tuple[bytes, bool]]:
        ...


class JpegEncoder:
    """``cv2.imencode(".jpg", ...)`` wrapped in the encoder protocol.

    Quality knob exposed because the bandwidth/quality tradeoff is the most
    common tuning lever in the field.
    """

    codec_id: int = CODEC_JPEG

    def __init__(self, quality: int = 70) -> None:
        import cv2  # local import: keeps this module importable in test
                    # environments that don't have OpenCV.

        self._cv2 = cv2
        self._params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]

    def encode(self, frame_bgr, ts_ns: int) -> Iterable[Tuple[bytes, bool]]:
        ok, encoded = self._cv2.imencode(".jpg", frame_bgr, self._params)
        if not ok:
            return
        yield encoded.tobytes(), True
