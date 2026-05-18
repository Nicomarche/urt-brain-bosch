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
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_temp_dir() -> Path:
    """Path al directorio ``temp/`` del repo (fallback legacy)."""
    return _repo_root() / "temp"


def _as_abs_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _has_path_value(value: str | None) -> bool:
    value = (value or "").strip()
    return bool(value) and value != "0"


def _unique_run_dir(logs_root: Path, now_ts: float) -> Path:
    run_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now_ts))
    run_dir = logs_root / f"run_{run_ts}"
    if not run_dir.exists():
        return run_dir
    pid_suffix = os.getpid()
    candidate = logs_root / f"run_{run_ts}_{pid_suffix}"
    counter = 1
    while candidate.exists():
        counter += 1
        candidate = logs_root / f"run_{run_ts}_{pid_suffix}_{counter}"
    return candidate


def _update_latest_symlink(logs_root: Path, run_dir: Path) -> str | None:
    if run_dir.parent != logs_root:
        return None
    latest = logs_root / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            if latest.is_dir() and not latest.is_symlink():
                return None
            latest.unlink()
        latest.symlink_to(run_dir.name, target_is_directory=True)
        return str(latest)
    except OSError:
        return None


def ensure_live_log_environment(
    *,
    repo_root: Path | None = None,
    now_ts: float | None = None,
    update_latest: bool = True,
) -> dict[str, str | bool | None]:
    """Ensure campaign-style live logging env vars exist before workers spawn.

    ``campaign_runner.py`` starts the brain through ``run.sh`` with
    ``URT_LIVE_LOG_PATH`` pointing at ``temp/logs/run_<ts>/brain.jsonl``.
    Direct invocations of ``main.py`` and the autostart service used to miss
    that bootstrap, leaving :func:`src.utils.live_log.live_log` as a no-op.

    This function mirrors the launcher contract in Python:

    * if ``URT_LOG_RUN_DIR`` is set, use it and default ``brain.jsonl`` there;
    * else if ``URT_LIVE_LOG_PATH`` is set, derive ``URT_LOG_RUN_DIR`` from it;
    * else create ``temp/logs/run_<ts>/brain.jsonl`` and update ``latest``.
    """
    root = Path(repo_root).expanduser() if repo_root is not None else _repo_root()
    if not root.is_absolute():
        root = Path.cwd() / root

    raw_run_dir = os.environ.get("URT_LOG_RUN_DIR")
    raw_live_path = os.environ.get("URT_LIVE_LOG_PATH")
    created_run_dir = False

    if _has_path_value(raw_run_dir):
        run_dir = _as_abs_path(str(raw_run_dir))
        if _has_path_value(raw_live_path):
            live_log_path = _as_abs_path(str(raw_live_path))
        else:
            live_log_path = run_dir / "brain.jsonl"
    elif _has_path_value(raw_live_path):
        live_log_path = _as_abs_path(str(raw_live_path))
        run_dir = live_log_path.parent
    else:
        logs_root = root / "temp" / "logs"
        logs_root.mkdir(parents=True, exist_ok=True)
        run_dir = _unique_run_dir(logs_root, time.time() if now_ts is None else now_ts)
        live_log_path = run_dir / "brain.jsonl"
        created_run_dir = True

    run_dir.mkdir(parents=True, exist_ok=True)
    live_log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        live_log_path.touch(exist_ok=True)
    except OSError:
        pass

    os.environ["URT_LOG_RUN_DIR"] = str(run_dir)
    os.environ["URT_LIVE_LOG_PATH"] = str(live_log_path)

    latest_link = None
    if update_latest:
        latest_link = _update_latest_symlink(root / "temp" / "logs", run_dir)

    return {
        "run_dir": str(run_dir),
        "live_log_path": str(live_log_path),
        "latest_link": latest_link,
        "created_run_dir": created_run_dir,
    }


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


__all__ = ["ensure_live_log_environment", "resolve_log_path"]
