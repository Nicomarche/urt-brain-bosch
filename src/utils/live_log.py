"""
live_log — append-only JSONL logger for cross-process pipeline debugging.

Cada call escribe una línea JSON al archivo apuntado por `URT_LIVE_LOG_PATH`.
Si esa env var no está seteada (o vale "0"/""), todo es no-op (overhead =
1 comparación). Pensado para que cada thread del brain emita su decisión
clave (pose publicada, scenario elegido, MotorCommand despachado, etc.)
al mismo timeline, leíble en tiempo real por `tail -f` o post-mortem por
`analyze_run.py`.

Diseño:
- Cada proceso abre su propio FD lazy (primer call) con O_APPEND. POSIX
  garantiza que `write(2)` ≤ PIPE_BUF (4 KB en macOS/Linux) es atómico
  cuando el FD se abrió con O_APPEND, así que múltiples procesos pueden
  escribir al mismo archivo sin locks ni interleaving.
- En `spawn` mode (macOS default desde Py 3.8), los FDs no se heredan; cada
  subproceso reabre el suyo en el primer `live_log()` que ejecute.
- Eventlet del dashboard NO monkey-patchea `os.write`, así que el helper
  funciona también desde el proceso del dashboard (aunque preferimos no
  llamarlo desde ahí para no acoplar la GUI al pipeline crítico).
- Si el payload serializado supera ~3 KB, lo truncamos con `_truncated:true`
  para no romper la garantía de atomicidad por línea.

Ejemplo:
    from src.utils.live_log import live_log
    live_log("behavior_planner", event="plan_output",
             scenario="lane_keep", valid=True,
             current_lanelet_id=42, sp0=0.15)

Salida (1 línea, sin newlines internos):
    {"ts":1714759123.456,"mono":12345.678,"pid":47291,
     "thread":"behavior_planner","event":"plan_output",
     "scenario":"lane_keep","valid":true,"current_lanelet_id":42,"sp0":0.15}
"""

import json
import os
import time
from typing import Any, Optional

# Tamaño máximo de línea para garantizar atomicidad (PIPE_BUF típico = 4096 B
# en macOS/Linux). Margen de seguridad: 3 KB.
_MAX_LINE_BYTES = 3072

_fd: Optional[int] = None
_path: Optional[str] = None
_pid_at_open: Optional[int] = None
_disabled: bool = False


def _open_lazy() -> Optional[int]:
    """Abre el FD para este proceso si todavía no está abierto.

    Reabre si cambió el PID (caso fork — multiprocessing.Process(start_method="fork")).
    En spawn mode el módulo se importa fresco en el child, así que `_fd` empieza
    en None y caemos por el path normal.
    """
    global _fd, _path, _pid_at_open, _disabled

    if _disabled:
        return None

    current_pid = os.getpid()
    if _fd is not None and _pid_at_open == current_pid:
        return _fd

    raw_path = os.environ.get("URT_LIVE_LOG_PATH", "")
    if not raw_path or raw_path == "0":
        # No-op: env var ausente o explícitamente desactivada.
        _disabled = True
        return None

    try:
        # O_APPEND garantiza atomicidad por write ≤ PIPE_BUF. O_CREAT por si
        # el orquestador no creó el archivo de antemano. 0o644 estándar.
        _fd = os.open(
            raw_path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o644,
        )
        _path = raw_path
        _pid_at_open = current_pid
    except OSError:
        # Si no podemos abrir (path inválido, permisos), nos desactivamos en
        # silencio para no romper el thread productor.
        _disabled = True
        _fd = None

    return _fd


def live_log(thread: str, **fields: Any) -> None:
    """Emite un evento JSONL al archivo de logs.

    Args:
        thread: tag corto del productor (ej. "behavior_planner", "mpc",
            "dispatcher", "ground_truth"). Se usa para filtrar/agrupar
            en el postmortem.
        **fields: kwargs libres splatted al toplevel del JSON. Convenio:
            siempre incluir `event=<str>` (corto, snake_case) para
            categorizar el tipo de evento dentro del thread. El resto
            son datos crudos (números, strings, listas chicas).

    No retorna nada. Si el archivo está cerrado o la env var no está
    seteada, es no-op silencioso. Nunca lanza — los errores se silencian
    para no romper el productor.
    """
    fd = _open_lazy()
    if fd is None:
        return

    record = {
        "ts": time.time(),
        "mono": time.monotonic(),
        "pid": os.getpid(),
        "thread": thread,
    }
    record.update(fields)

    try:
        line = json.dumps(record, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        # Algún field no serializable que ni siquiera default=str arregla.
        # Ignorar silencioso.
        return

    encoded = line.encode("utf-8") + b"\n"

    if len(encoded) > _MAX_LINE_BYTES:
        # Trunca el payload para mantener atomicidad. Drop fields hasta
        # quedar dentro del límite, marcando el truncate.
        safe_record = {
            "ts": record["ts"],
            "mono": record["mono"],
            "pid": record["pid"],
            "thread": thread,
            "event": fields.get("event", "unknown"),
            "_truncated": True,
            "_original_size": len(encoded),
        }
        encoded = (
            json.dumps(safe_record, default=str, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )

    try:
        os.write(fd, encoded)
    except OSError:
        # Disco lleno, FD inválido por algún motivo, etc. Mejor silenciar
        # que romper el thread productor.
        pass


def is_enabled() -> bool:
    """Devuelve True si el logger está activo (URT_LIVE_LOG_PATH seteada y FD OK)."""
    return _open_lazy() is not None


def get_path() -> Optional[str]:
    """Path del archivo activo (o None si está deshabilitado)."""
    _open_lazy()
    return _path
