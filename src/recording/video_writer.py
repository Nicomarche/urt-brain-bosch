"""Wrapper sobre ``cv2.VideoWriter`` para grabar AVI a una sesión manual.

Encapsula el patrón "abrir lazy con la primera shape, redimensionar si
cambia, cerrar limpio" que ya usa
:meth:`src.hardware.camera.threads.threadCamera.threadCamera._ensure_video_writer`.
Lo extraemos acá para que los dos productores de video del modo manual
(frame raw en ``threadCamera`` y frame anotado en ``threadLocalPerception``)
compartan la misma implementación sin duplicar lógica.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover — opencv ausente sólo en CI minimal
    cv2 = None  # type: ignore

_logger = logging.getLogger(__name__)


class SessionVideoWriter:
    """Escritor AVI lazy thread-safe-no.

    No es thread-safe: cada productor de frames usa su propia instancia
    desde su propio thread. Se inicializa cerrado; ``write(frame)`` abre
    el ``cv2.VideoWriter`` con la shape del primer frame. Llamadas
    posteriores con shapes distintas se redimensionan al tamaño inicial
    para mantener el AVI consistente.
    """

    def __init__(self, output_path: str, fps: float, codec: str = "XVID") -> None:
        self._output_path = output_path
        self._fps = float(fps) if fps and fps > 0 else 15.0
        self._codec = codec
        self._writer = None  # type: Optional[object]
        self._size: Optional[Tuple[int, int]] = None
        self._frames_written = 0

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def output_path(self) -> str:
        return self._output_path

    def write(self, frame) -> bool:
        """Escribe un frame al AVI. Devuelve ``True`` si se persistió."""
        if cv2 is None:
            return False
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            return False
        if self._writer is None:
            if not self._open(frame):
                return False
        out = frame
        assert self._size is not None
        target_w, target_h = self._size
        frame_h, frame_w = frame.shape[:2]
        if (int(frame_w), int(frame_h)) != (target_w, target_h):
            out = cv2.resize(frame, (target_w, target_h))  # type: ignore[attr-defined]
        try:
            self._writer.write(out)  # type: ignore[union-attr]
            self._frames_written += 1
            return True
        except Exception:
            _logger.exception("Failed writing frame to %s", self._output_path)
            return False

    def _open(self, first_frame) -> bool:
        height, width = first_frame.shape[:2]
        if width <= 0 or height <= 0:
            return False
        fourcc = cv2.VideoWriter_fourcc(*self._codec)  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(  # type: ignore[attr-defined]
            self._output_path, fourcc, self._fps, (int(width), int(height))
        )
        if not writer.isOpened():
            try:
                writer.release()
            except Exception:
                pass
            _logger.error(
                "Cannot open SessionVideoWriter at %s with size %dx%d",
                self._output_path, width, height,
            )
            return False
        self._writer = writer
        self._size = (int(width), int(height))
        _logger.info(
            "SessionVideoWriter open: %s @ %dx%d %.1f fps",
            self._output_path, width, height, self._fps,
        )
        return True

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.release()  # type: ignore[union-attr]
            except Exception:
                _logger.exception("Failed releasing SessionVideoWriter")
        self._writer = None
        self._size = None


__all__ = ["SessionVideoWriter"]
