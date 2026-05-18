"""SQLite store del historial de runs del brain.

Plan E4: el `analyze_run.py` ya escribe carpetas ``temp/logs/run_<ts>``
con jsonl. Esta DB es el índice ligero: una fila por run con metadatos
(start_ts, end_ts, sim_mode, total_distance_m, n_errors, notes).

API:
  * ``insert_run_start(run_id, sim_mode)`` al boot de main.py.
  * ``update_run_end(run_id, total_distance_m=..., n_errors=...)`` al
    shutdown handler.
  * ``list_recent_runs(n=10)`` para tools/dev/analyze_run.py --list-runs.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


def _default_db_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "sessions.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    start_ts REAL NOT NULL,
    end_ts REAL,
    sim_mode INTEGER NOT NULL,
    total_distance_m REAL,
    n_errors INTEGER,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_start_ts ON runs(start_ts DESC);
"""


class SessionsDB:
    """Historial de runs con un único writer (main.py)."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else _default_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False, timeout=10.0
        )
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ─── Lifecycle entries ────────────────────────────────────────────
    def insert_run_start(
        self,
        run_id: str,
        *,
        sim_mode: bool,
        notes: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO runs(run_id, start_ts, sim_mode, notes) "
                "VALUES(?, ?, ?, ?)",
                (str(run_id), float(time.time()), int(bool(sim_mode)), notes),
            )
            self._conn.commit()

    def update_run_end(
        self,
        run_id: str,
        *,
        total_distance_m: float | None = None,
        n_errors: int | None = None,
        notes: str | None = None,
    ) -> None:
        fields = ["end_ts=?"]
        values: list[Any] = [float(time.time())]
        if total_distance_m is not None:
            fields.append("total_distance_m=?")
            values.append(float(total_distance_m))
        if n_errors is not None:
            fields.append("n_errors=?")
            values.append(int(n_errors))
        if notes is not None:
            fields.append("notes=?")
            values.append(notes)
        values.append(str(run_id))
        sql = f"UPDATE runs SET {', '.join(fields)} WHERE run_id=?"
        with self._lock:
            self._conn.execute(sql, values)
            self._conn.commit()

    # ─── Queries ──────────────────────────────────────────────────────
    def list_recent_runs(self, n: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT run_id, start_ts, end_ts, sim_mode, total_distance_m, "
                "n_errors, notes FROM runs ORDER BY start_ts DESC LIMIT ?",
                (int(n),),
            )
            return [
                {
                    "run_id": row[0],
                    "start_ts": float(row[1]) if row[1] is not None else None,
                    "end_ts": float(row[2]) if row[2] is not None else None,
                    "sim_mode": bool(row[3]),
                    "total_distance_m": float(row[4]) if row[4] is not None else None,
                    "n_errors": int(row[5]) if row[5] is not None else None,
                    "notes": row[6],
                }
                for row in cur.fetchall()
            ]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT run_id, start_ts, end_ts, sim_mode, total_distance_m, "
                "n_errors, notes FROM runs WHERE run_id=?",
                (str(run_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "run_id": row[0],
                "start_ts": float(row[1]),
                "end_ts": float(row[2]) if row[2] is not None else None,
                "sim_mode": bool(row[3]),
                "total_distance_m": float(row[4]) if row[4] is not None else None,
                "n_errors": int(row[5]) if row[5] is not None else None,
                "notes": row[6],
            }

    # ─── Lifecycle ────────────────────────────────────────────────────
    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self) -> "SessionsDB":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
