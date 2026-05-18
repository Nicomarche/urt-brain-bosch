"""RuntimeParams — singleton de parámetros mutables en caliente.

Plan E1: separamos `config.py` (constantes físicas, endpoints, paths) de
`config/runtime.yaml` (valores que el operador quiere tunear sin reiniciar
el robot).

Características:
  * Polling thread-safe del mtime del archivo cada ~2 s.
  * API estilo `.get("parking.search_speed_mps", default)` con notación
    dotted para navegar el YAML.
  * Listeners pueden suscribirse a cambios via `add_listener(callback)`.
  * El BehaviorPlanner aplica los cambios SOLO en boundaries de scenario
    para no romper al MPC compilado midstride.

Por qué NO usar `fsevents` / watchdog específico de OS:
  * Polling cross-platform (Linux Jetson + macOS dev) sin deps externos.
  * 2 s de latencia es aceptable para un tuning de campo.
  * Evita complejidad de un thread de file events que pueda crashearse.

No usar `RuntimeParams` para algo crítico de seguridad (ese tipo de cosas
viene de `config.py`, fijo en boot). Esta capa es para iterar valores que
no afectan al safety gate ni al control core.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

try:
    import yaml as _yaml  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - yaml es dep estándar del repo
    _yaml = None  # type: ignore[assignment]


def _default_runtime_yaml_path() -> Path:
    """Path canónico al runtime.yaml relativo al repo."""
    # Sube hasta la raíz del repo desde este archivo: src/core/runtime_params.py
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "config" / "runtime.yaml"


class RuntimeParams:
    """Singleton mínimo con polling de mtime.

    Pattern de uso desde un thread del brain::

        from src.core.runtime_params import params
        params.start()                # idempotente
        speed = params.get("parking.search_speed_mps", 0.1)

    Listeners para reconfiguración explícita::

        def on_reload():
            ...
        params.add_listener(on_reload)
    """

    _instance: "RuntimeParams | None" = None
    _instance_lock = threading.Lock()

    def __new__(cls, path: Path | None = None) -> "RuntimeParams":
        with cls._instance_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                cls._instance = inst
            return cls._instance

    def __init__(self, path: Path | None = None) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._path = path or _default_runtime_yaml_path()
        self._data: dict[str, Any] = {}
        self._last_mtime: float = 0.0
        self._lock = threading.RLock()
        self._listeners: list[Callable[[], None]] = []
        self._poll_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._poll_interval_s = 2.0
        # Carga inicial síncrona — disponible al primer get() incluso si el
        # polling todavía no arrancó.
        self._load_once(silent=True)

    # ─── Lifecycle ────────────────────────────────────────────────────
    def start(self) -> None:
        """Arranca el polling en background. Idempotente."""
        with self._lock:
            if self._poll_thread is not None and self._poll_thread.is_alive():
                return
            self._stop_event.clear()
            self._poll_thread = threading.Thread(
                target=self._poll_loop,
                name="RuntimeParams-Poll",
                daemon=True,
            )
            self._poll_thread.start()

    def stop(self) -> None:
        """Para el polling. Llamado en shutdown ordenado."""
        self._stop_event.set()
        t = self._poll_thread
        if t is not None and t.is_alive():
            t.join(timeout=3.0)
        self._poll_thread = None

    # ─── Listener API ─────────────────────────────────────────────────
    def add_listener(self, callback: Callable[[], None]) -> None:
        """Registra un callback que dispara cuando el YAML se recarga.

        El callback corre en el thread de polling — no hacer trabajo
        pesado o publish-blocking en él. Para aplicar cambios al
        scenario activo, el BehaviorPlanner debe checar en cada tick si
        ``params.last_reload_ts > planner.last_apply_ts`` y aplicar en
        boundary de scenario.
        """
        with self._lock:
            self._listeners.append(callback)

    # ─── Lookup ───────────────────────────────────────────────────────
    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Devuelve el valor en ``dotted_key`` (ej ``parking.search_speed_mps``).

        Si la ruta no existe o el YAML está vacío, devuelve ``default``.
        """
        with self._lock:
            cur: Any = self._data
            for part in dotted_key.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return default
            return cur

    def snapshot(self) -> dict[str, Any]:
        """Copia inmutable del estado actual para diagnóstico/persistencia."""
        with self._lock:
            import copy
            return copy.deepcopy(self._data)

    # ─── Internals ────────────────────────────────────────────────────
    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._load_once(silent=False)
            self._stop_event.wait(self._poll_interval_s)

    def _load_once(self, *, silent: bool) -> None:
        try:
            stat = self._path.stat()
        except FileNotFoundError:
            return
        mtime = stat.st_mtime
        with self._lock:
            if mtime == self._last_mtime:
                return
            self._last_mtime = mtime
        try:
            text = self._path.read_text(encoding="utf-8")
            if _yaml is None:
                # Fallback: si yaml no está instalado, intentamos JSON
                # rudimentario. Mejor que silencio absoluto.
                import json
                new_data = json.loads(text)
            else:
                new_data = _yaml.safe_load(text) or {}
        except Exception as exc:
            if not silent:
                print(
                    f"[RuntimeParams] reload failed for {self._path}: {exc}",
                    flush=True,
                )
            return
        if not isinstance(new_data, dict):
            return
        with self._lock:
            self._data = new_data
            listeners = list(self._listeners)
        # Disparar listeners fuera del lock para evitar deadlock.
        for cb in listeners:
            try:
                cb()
            except Exception as exc:  # pragma: no cover - defensivo
                print(f"[RuntimeParams] listener error: {exc}", flush=True)


# Singleton público — importar desde donde haga falta.
params = RuntimeParams()


__all__ = ["RuntimeParams", "params"]
