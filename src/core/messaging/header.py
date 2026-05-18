"""MessageHeader — metadata uniforme para mensajes inter-thread/inter-process.

Inspirado en `std_msgs/Header` de ROS pero adaptado a Python + ZMQ. Cada
mensaje crítico (PoseEstimate, BehaviorOutput, MotorCommand, OverlayLayer,
ThreadHeartbeat) debe llevar un Header en su payload para garantizar:

  * Trazabilidad temporal con `timestamp` UNIX (segundos float).
  * Detección de gaps via `seq` monotónico por (source, topic).
  * Identificación del productor en `source` para postmortem.
  * Resolución de marco de referencia espacial en `frame_id` (map, base_link,
    camera) — útil para overlays y transforms que cruzan frames.

Diseño:
  * `MessageHeader` es frozen + slots para minimizar overhead per-message.
  * Helpers `stamp(source, seq_counter)` y `now_header(source, ...)` evitan
    duplicar boilerplate; los counters son responsabilidad del productor
    (no hay global state oculto).
  * `serialize() / from_dict()` para round-trip JSON con el bus ZMQ que
    serializa diccionarios.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

FRAME_MAP = "map"
FRAME_BASE_LINK = "base_link"
FRAME_CAMERA = "camera"
FRAME_LIDAR = "lidar"


@dataclass(frozen=True)
class MessageHeader:
    """Metadata adjunta a cada mensaje crítico del bus.

    Attributes:
        timestamp: Tiempo UNIX (segundos float) en que se construyó el
            payload. Usar ``time.time()`` con monotonic backup si hace
            falta correlación entre procesos.
        seq: Secuencia monotónica del productor para ese topic. Comienza
            en 0 y se incrementa por publicación. Permite detectar gaps
            o reordenamientos del bus.
        source: Identificador del thread/proceso productor (ej.
            ``"behavior_planner"``, ``"pose_estimator"``). Útil para
            postmortem con grep.
        frame_id: Marco de referencia espacial del payload. ``"map"`` es
            el frame plano fijo (Lanelet2 proyectado). ``"base_link"`` es
            el chasis del auto. ``"camera"`` la cámara montada. Sólo
            válido cuando el payload contiene geometría.
    """

    timestamp: float = 0.0
    seq: int = 0
    source: str = ""
    frame_id: str = FRAME_MAP

    def to_dict(self) -> dict[str, Any]:
        """Serialización JSON-compatible para el bus ZMQ."""
        return {
            "timestamp": float(self.timestamp),
            "seq": int(self.seq),
            "source": str(self.source),
            "frame_id": str(self.frame_id),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "MessageHeader":
        """Parse defensivo de un dict del bus. Tolera ausencia de campos."""
        if not isinstance(data, dict):
            return cls()
        return cls(
            timestamp=float(data.get("timestamp", 0.0)),
            seq=int(data.get("seq", 0)),
            source=str(data.get("source", "")),
            frame_id=str(data.get("frame_id", FRAME_MAP)),
        )


@dataclass
class SeqCounter:
    """Contador monotónico thread-local para el productor.

    Patrón de uso:

        _seq = SeqCounter()
        ...
        header = stamp("pose_estimator", _seq, frame_id=FRAME_MAP)
        payload = {"header": header.to_dict(), "x": x, "y": y, ...}
        publisher.send(payload)

    No es thread-safe por diseño: cada productor debe tener su propio
    contador. Si dos threads compiten por el mismo contador, mover a
    `threading.Lock` o usar `itertools.count` thread-safe externamente.
    """

    value: int = 0

    def next(self) -> int:
        """Devuelve el seq actual y avanza el contador."""
        v = self.value
        self.value = v + 1
        return v


def stamp(
    source: str,
    seq_counter: SeqCounter,
    *,
    frame_id: str = FRAME_MAP,
    timestamp: float | None = None,
) -> MessageHeader:
    """Construye un `MessageHeader` con el siguiente seq del productor.

    Args:
        source: Nombre del productor. Convención: snake_case del thread.
        seq_counter: Contador local del productor (avanza in-place).
        frame_id: Frame de referencia del payload. Default ``"map"``.
        timestamp: Override opcional del timestamp (para test/replay).
            Si ``None`` usa ``time.time()``.

    Returns:
        Nuevo `MessageHeader` listo para insertar en el payload.
    """
    ts = float(timestamp) if timestamp is not None else float(time.time())
    return MessageHeader(
        timestamp=ts,
        seq=seq_counter.next(),
        source=source,
        frame_id=frame_id,
    )


def header_from_payload(payload: Any) -> MessageHeader:
    """Extrae un Header de un payload del bus, o devuelve uno vacío.

    Consumidores que migran gradualmente: si el productor todavía no
    incluye Header, este helper devuelve un Header default sin romper.
    El consumidor puede leer ``header.timestamp`` con fallback al
    legacy ``payload["timestamp"]`` toplevel.
    """
    if isinstance(payload, dict):
        hdr = payload.get("header")
        if isinstance(hdr, dict):
            return MessageHeader.from_dict(hdr)
    return MessageHeader()
