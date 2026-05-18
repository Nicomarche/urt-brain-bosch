"""Resolución de paths de logs — single source of truth para `URT_LOG_RUN_DIR`.

Plan E5: los productores legacy (`serial_history.log`, `tracking_debug.txt`,
`lane_calib_log.txt`) escribían directamente a `temp/` toplevel,
contaminando la raíz del directorio temporal. Centralizamos la resolución
acá:

  * Si la env var ``URT_LOG_RUN_DIR`` existe → escribimos dentro del run
    correspondiente (``temp/logs/run_<ts>/<file>``).
  * Si no existe → fallback a ``temp/<file>`` para compatibilidad con
    invocaciones legacy de ``main.py`` directas.

Uso típico::

    from src.core.log_paths import resolve_log_path
    log_file = resolve_log_path("serial_history.log")
    # → "/path/to/repo/temp/logs/run_20260517_120000/serial_history.log"
    #   o "/path/to/repo/temp/serial_history.log" si no hay env var.
"""

from __future__ import annotations

import os
from pathlib import Path


def _repo_temp_dir() -> Path:
    """Path al directorio ``temp/`` del repo (fallback legacy)."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "temp"


def resolve_log_path(filename: str, *, ensure_parent: bool = True) -> str:
    """Devuelve el path absoluto donde un productor debe escribir ``filename``.

    Args:
        filename: Nombre del archivo (sin ruta). Ej. ``"serial_history.log"``.
        ensure_parent: Si ``True`` crea el directorio padre con mkdir -p.

    Returns:
        Path absoluto como string para pasar a ``open()`` o equivalentes.
    """
    env = os.environ.get("URT_LOG_RUN_DIR", "").strip()
    if env:
        target = Path(env) / filename
    else:
        target = _repo_temp_dir() / filename
    if ensure_parent:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    return str(target)


__all__ = ["resolve_log_path"]
