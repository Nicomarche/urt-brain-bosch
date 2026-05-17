"""Where the GUI reads/writes config files and which defaults to use.

Single source of truth for paths so `git mv`-style refactors only touch
this module. Anything else that needs to know "where do routes live" goes
through ``CONFIG_PATHS``.

Layout:

* ``config/`` (repo root) — committed JSON, version-controlled. When you
  push from your laptop and pull on the Jetson, these come along.
* ``src/utils/table_state.json`` — legacy Angular table state. We migrate
  it into ``config/routes.json`` on first launch and never touch it
  again, so the legacy frontend (still served from ``legacy/dashboard-
  frontend``) keeps working as a fallback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    """Path to the repository root.

    Assumes this file is at ``<repo>/src/gui/config/settings.py``. The
    ``parents`` index counts: [0]=config/, [1]=gui/, [2]=src/, [3]=repo.
    """
    return Path(__file__).resolve().parents[3]


REPO_ROOT: Path = _repo_root()
CONFIG_DIR: Path = REPO_ROOT / "config"
LEGACY_TABLE_STATE: Path = REPO_ROOT / "src" / "utils" / "table_state.json"


def _repo_path(path_like: str | os.PathLike[str]) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _default_track_map_dir() -> Path:
    env_override = os.environ.get("URT_TRACK_MAP_DIR")
    if env_override:
        return _repo_path(env_override)
    try:
        import config as backend_config

        backend_track_map_dir = getattr(backend_config, "TRACK_MAP_DIR", None)
        if backend_track_map_dir:
            return _repo_path(str(backend_track_map_dir))
    except Exception:
        pass
    sim_mode = os.environ.get("URT_SIM_MODE", "1" if sys.platform == "darwin" else "0") == "1"
    return REPO_ROOT / "maps" / ("sim" if sim_mode else "jetson")


TRACK_MAP_DIR: Path = _default_track_map_dir()


def default_track_map_dir() -> Path:
    """Return the default track-map directory used by the dashboard map view."""
    return Path(TRACK_MAP_DIR)

# All known config files the GUI persists. New entries must be added here
# (and only here) so panels can refer to them by symbolic name.
CONFIG_PATHS: dict[str, Path] = {
    "routes":      CONFIG_DIR / "routes.json",       # destination node lists
    "params":      CONFIG_DIR / "params.json",       # sliders/dropdowns
    "calibration": CONFIG_DIR / "calibration.json",  # PWM polynomials
    "control":     CONFIG_DIR / "control.json",      # speed/steer/PIDs defaults
    "gui_layout":  CONFIG_DIR / "gui_layout.json",   # splitter ratios, last tab
}


# Defaults applied when the file is missing. Kept tiny on purpose — anything
# more elaborate belongs to the panel that owns the data.
DEFAULTS: dict[str, dict[str, Any]] = {
    "routes":      {"saved_routes": []},
    "params":      {"params": {}},
    "calibration": {},
    "control":     {
        "max_speed": 30,
        "max_steer": 25,
        "default_mode": "stop",
    },
    "gui_layout":  {
        "window_size": [1280, 800],
        "splitter_ratio": [220, 1060],
        "active_tab": "driving",
    },
}


def config_path(key: str) -> Path:
    """Return the absolute path for a config key (raises on unknown key)."""
    try:
        return CONFIG_PATHS[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown config key '{key}'. Add it to CONFIG_PATHS in settings.py."
        ) from exc


def default_for(key: str) -> dict[str, Any]:
    """Return a fresh copy of the default dict for a config key."""
    import copy
    return copy.deepcopy(DEFAULTS.get(key, {}))


def ensure_config_dir() -> None:
    """Create ``config/`` if it doesn't exist yet (no-op otherwise)."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Connection defaults — overridable from the CLI in main.py
# ----------------------------------------------------------------------
DEFAULT_HOST: str = os.environ.get("URT_GUI_HOST", "localhost")
DEFAULT_PORT: int = int(os.environ.get("URT_GUI_PORT", "5005"))

# UDP port the GUI binds for receiving fragmented JPEG video from
# ``UdpVideoStreamerThread``. 0 = ephemeral (kernel picks). Static ports
# are convenient on dev boxes where the robot's URT_VIDEO_TARGET_PORT is
# baked into run.sh.
DEFAULT_UDP_VIDEO_PORT: int = int(os.environ.get("URT_GUI_UDP_VIDEO_PORT", "0"))

# MD5 hash of the password the GUI will accept. Empty = no password.
# We honour the env var so you can ship a binary without baking it in.
DEFAULT_PASSWORD_MD5: str = os.environ.get("URT_GUI_PASSWORD_MD5", "")
