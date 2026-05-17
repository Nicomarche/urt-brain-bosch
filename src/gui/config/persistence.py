"""Atomic JSON persistence for the GUI config files.

Every write goes through ``save_json_atomic`` so a crash mid-write can't
leave the file truncated. The pattern is the same as Postgres' ``rename``
trick: write to ``<path>.tmp``, ``fsync`` the contents, then ``os.replace``
into place — that final step is atomic on POSIX and Windows.

Also exposes a one-shot ``migrate_legacy_table_state`` so the first launch
of the GUI inherits whatever the Angular dashboard had saved, instead of
starting blank.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from .settings import (
    CONFIG_DIR,
    LEGACY_TABLE_STATE,
    config_path,
    default_for,
    ensure_config_dir,
)


_logger = logging.getLogger(__name__)


def load_json(path: Path | str, default: Any = None) -> Any:
    """Load a JSON file, returning ``default`` if it doesn't exist or is
    unreadable. Never raises for the common cases (missing file, bad JSON).
    """
    p = Path(path)
    if not p.exists():
        return default
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        _logger.warning("Failed to load %s: %s — returning default", p, exc)
        return default


def save_json_atomic(path: Path | str, data: Any) -> None:
    """Write ``data`` as pretty-printed JSON to ``path`` atomically.

    Uses ``tempfile.NamedTemporaryFile`` in the same directory to guarantee
    the rename stays on the same filesystem (cross-fs renames aren't atomic).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # ``delete=False`` because we close it ourselves before the rename.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8",
        dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name

    os.replace(tmp_path, p)


def load_config(key: str) -> dict[str, Any]:
    """Read a known config (by key from settings.py). Missing → defaults."""
    path = config_path(key)
    data = load_json(path, default=None)
    if not isinstance(data, dict):
        return default_for(key)
    return data


def save_config(key: str, data: dict[str, Any]) -> None:
    """Persist a known config (by key) atomically."""
    ensure_config_dir()
    save_json_atomic(config_path(key), data)


# ----------------------------------------------------------------------
# Legacy migration
# ----------------------------------------------------------------------
def migrate_legacy_table_state() -> bool:
    """If ``src/utils/table_state.json`` exists and ``config/routes.json``
    doesn't, copy the legacy state into the new location.

    Returns ``True`` if a migration was performed (so callers can log it).
    Idempotent: subsequent launches see ``routes.json`` already there and
    skip.
    """
    routes_path = config_path("routes")
    if routes_path.exists():
        return False
    if not LEGACY_TABLE_STATE.exists():
        return False

    legacy = load_json(LEGACY_TABLE_STATE, default=None)
    if legacy is None:
        return False

    ensure_config_dir()
    save_json_atomic(routes_path, legacy)
    _logger.info("Migrated legacy table state %s → %s", LEGACY_TABLE_STATE, routes_path)
    return True
