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
import time
from pathlib import Path

import pytest

from src.core.log_paths import ensure_live_log_environment, resolve_log_path


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


def test_ensure_live_log_environment_creates_campaign_style_run(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("URT_LOG_RUN_DIR", raising=False)
    monkeypatch.delenv("URT_LIVE_LOG_PATH", raising=False)

    env = ensure_live_log_environment(repo_root=tmp_path, now_ts=1_800_000_000.0)

    run_dir = Path(env["run_dir"])
    live_log_path = Path(env["live_log_path"])
    expected_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(1_800_000_000.0))
    assert run_dir == tmp_path / "temp" / "logs" / f"run_{expected_ts}"
    assert live_log_path == run_dir / "brain.jsonl"
    assert live_log_path.exists()
    assert os.environ["URT_LOG_RUN_DIR"] == str(run_dir)
    assert os.environ["URT_LIVE_LOG_PATH"] == str(live_log_path)
    latest = tmp_path / "temp" / "logs" / "latest"
    assert latest.is_symlink()
    assert latest.readlink() == Path(run_dir.name)


def test_ensure_live_log_environment_derives_run_dir_from_live_path(tmp_path: Path, monkeypatch):
    custom_live_log = tmp_path / "custom_run" / "brain.jsonl"
    monkeypatch.delenv("URT_LOG_RUN_DIR", raising=False)
    monkeypatch.setenv("URT_LIVE_LOG_PATH", str(custom_live_log))

    env = ensure_live_log_environment(repo_root=tmp_path)

    assert env["run_dir"] == str(custom_live_log.parent)
    assert env["live_log_path"] == str(custom_live_log)
    assert custom_live_log.exists()
    assert os.environ["URT_LOG_RUN_DIR"] == str(custom_live_log.parent)


def test_ensure_live_log_environment_uses_existing_run_dir(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "preset_run"
    monkeypatch.setenv("URT_LOG_RUN_DIR", str(run_dir))
    monkeypatch.delenv("URT_LIVE_LOG_PATH", raising=False)

    env = ensure_live_log_environment(repo_root=tmp_path)

    assert env["run_dir"] == str(run_dir)
    assert env["live_log_path"] == str(run_dir / "brain.jsonl")
    assert (run_dir / "brain.jsonl").exists()
    assert os.environ["URT_LIVE_LOG_PATH"] == str(run_dir / "brain.jsonl")


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
        "src/hardware/camera/threads/threadLineFollowing.py": (
            "lane_calib_log.txt",
            "lane_mask_debug.log",
            "line_following_auto_last_run.txt",
            "line_following_manual_last_run.txt",
        ),
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
