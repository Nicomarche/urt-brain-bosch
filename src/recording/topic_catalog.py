"""Catálogo de topics a grabar en modo MANUAL.

Se introspecciona :mod:`src.core.bus.topics` para no mantener una lista
hardcoded que se desincronice al agregar topics nuevos. La filosofía ahora
es literal: **todo Topic declarado a toplevel se graba**. Eso incluye toggles
GUI, handshakes, topics de cámara si vuelven a publicarse en el futuro y el
propio lifecycle ``MANUAL_RECORDING_SESSION``.
"""

from __future__ import annotations

from typing import Iterable

from src.core.bus import topics as _topics
from src.core.bus.topics import Topic


def _safe_filename(topic_name: str) -> str:
    """Deriva un nombre de archivo válido del topic name.

    ``"/auto/threadread/imudata"`` → ``"imudata"``. Tomamos sólo el último
    segmento porque el prefijo es redundante en el sistema de archivos.
    """
    parts = [p for p in topic_name.strip("/").split("/") if p]
    if not parts:
        return "unknown"
    return parts[-1]


def iter_loggable_topics() -> Iterable[Topic]:
    """Itera los :class:`Topic` que el recorder debe drenar.

    Filtra contra ``_topics`` por introspección — todo ``Topic`` declarado
    a toplevel se incluye automáticamente al merge siguiente sin tocar este
    archivo. Si dos constantes apuntan al mismo topic name, se graba una sola
    vez para evitar duplicar líneas.
    """
    seen: set[str] = set()
    for attr_name in dir(_topics):
        if attr_name.startswith("_"):
            continue
        obj = getattr(_topics, attr_name)
        if not isinstance(obj, Topic):
            continue
        if obj.name in seen:
            continue
        seen.add(obj.name)
        yield obj


def filename_for_topic(topic: Topic) -> str:
    """Nombre del ``.jsonl`` (sin extensión) en ``per_topic/`` para este topic."""
    return _safe_filename(topic.name)


__all__ = ["iter_loggable_topics", "filename_for_topic"]
