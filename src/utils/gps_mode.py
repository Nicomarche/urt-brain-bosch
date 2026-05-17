from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GPS_MODE_PATH = REPO_ROOT / "config" / "gps_mode.json"


def gps_mode_path() -> Path:
    return Path(GPS_MODE_PATH)


def _save_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=True, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def is_gps_disabled() -> bool:
    path = gps_mode_path()
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("disabled", False))


def save_gps_disabled(disabled: bool, *, source: str = "dashboard") -> dict[str, Any]:
    payload = {
        "disabled": bool(disabled),
        "source": str(source),
    }
    _save_json_atomic(gps_mode_path(), payload)
    return dict(payload)
