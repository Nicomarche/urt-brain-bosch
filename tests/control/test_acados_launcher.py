from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "ensure_acados.sh"
_VENV_PYTHON = _ROOT / ".venv" / "bin" / "python"
_ACADOS_INSTALL = Path.home() / "acados_install"


def test_ensure_acados_smoke_with_repo_venv() -> None:
    if not _SCRIPT.is_file():
        pytest.skip("ensure_acados.sh not present")
    if not _VENV_PYTHON.is_file():
        pytest.skip("repo .venv python not present")
    if not (_ACADOS_INSTALL / "lib").is_dir():
        pytest.skip("local acados install not present")

    proc = subprocess.run(
        ["bash", str(_SCRIPT), str(_VENV_PYTHON), str(_ACADOS_INSTALL)],
        cwd=str(_ROOT),
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        "Todo OK" in proc.stdout
        or "AcadosMPC listo" in proc.stdout
        or "instalación completa" in proc.stdout
    ), proc.stdout
