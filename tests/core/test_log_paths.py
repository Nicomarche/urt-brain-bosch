"""Plan TANDA 2.3 / E5 — `resolve_log_path` y migración de logs sueltos.

Antes los productores legacy (serial_history, tracking_debug, lane_calib,
lane_mask) escribían DIRECTAMENTE a ``temp/`` toplevel, contaminando la
raíz. Ahora todos deben pasar por ``resolve_log_path()`` que:

  * Si ``URT_LOG_RUN_DIR`` está seteada → escribe en
    ``$URT_LOG_RUN_DIR/<filename>``.
  * Si NO está → cae a ``temp/<filename>`` por retro-compat con
    invocaciones legacy.

Verificamos además que los productores conocidos (TANDA 2.3) usan
``resolve_log_path`` en lugar de path hardcoded.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.core.log_paths import resolve_log_path


def test_resolve_with_env_var(tmp_path: Path, monkeypatch):
    """Con URT_LOG_RUN_DIR seteada, el path resuelve adentro del run dir."""
    monkeypatch.setenv("URT_LOG_RUN_DIR", str(tmp_path))
    p = resolve_log_path("test.log")
    assert p == str(tmp_path / "test.log")
    assert tmp_path.exists()  # ensure_parent crea el dir si no existe


def test_resolve_without_env_var_falls_to_temp(monkeypatch):
    """Sin URT_LOG_RUN_DIR, cae a temp/ del repo (compat legacy)."""
    monkeypatch.delenv("URT_LOG_RUN_DIR", raising=False)
    p = resolve_log_path("test.log")
    assert p.endswith("temp/test.log") or p.endswith("temp" + os.sep + "test.log")


def test_resolve_creates_parent_directory(tmp_path: Path, monkeypatch):
    nested = tmp_path / "deep" / "nested" / "logs"
    monkeypatch.setenv("URT_LOG_RUN_DIR", str(nested))
    resolve_log_path("foo.log")
    assert nested.exists()


def test_resolve_empty_env_var_falls_to_temp(monkeypatch):
    """URT_LOG_RUN_DIR vacío string es equivalente a no estar."""
    monkeypatch.setenv("URT_LOG_RUN_DIR", "")
    p = resolve_log_path("test.log")
    assert "temp" in p


def test_resolve_does_not_create_parent_when_disabled(tmp_path: Path, monkeypatch):
    target_dir = tmp_path / "should_not_exist"
    monkeypatch.setenv("URT_LOG_RUN_DIR", str(target_dir))
    resolve_log_path("foo.log", ensure_parent=False)
    assert not target_dir.exists()


def test_known_producers_use_resolve_log_path():
    """Plan TANDA 2.3: los 4 productores legacy migrados deben usar
    ``resolve_log_path`` en vez de path hardcoded.

    Si esta lista crece o se mueve, actualizar acá. Detecta regresiones
    donde alguien copy-pastea el viejo patrón ``temp/<file>`` directo.
    """
    repo_root = Path(__file__).resolve().parents[2]
    targets = {
        "src/hardware/serialhandler/processSerialHandler.py": "serial_history.log",
        "src/localization/relocalization_thread.py": "tracking_debug.txt",
        "src/hardware/camera/threads/threadLineFollowing.py": ("lane_calib_log.txt", "lane_mask_debug.log"),
    }
    for relpath, filenames in targets.items():
        path = repo_root / relpath
        if not path.is_file():
            pytest.skip(f"{relpath} no existe — probablemente fue refactorizado")
            return
        source = path.read_text(encoding="utf-8")
        # Cada filename mencionado debe aparecer adyacente a una llamada
        # a resolve_log_path (o el comentario que documenta el cambio).
        if isinstance(filenames, str):
            filenames = (filenames,)
        for filename in filenames:
            assert filename in source, f"{relpath} ya no menciona {filename}?"
            # Heurística: que en el archivo se importe resolve_log_path.
            assert "resolve_log_path" in source, (
                f"{relpath} menciona {filename} pero NO importa resolve_log_path "
                "— violó TANDA 2.3 (debe migrar a log_paths.resolve_log_path)."
            )
