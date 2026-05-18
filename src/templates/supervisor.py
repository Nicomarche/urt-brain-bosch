"""ThreadSupervisor — backoff exponencial + heartbeat publisher.

Plan G1+G2+F4: el ``ThreadWithStop`` base hoy hace ``try/except`` que
loggea y sigue indefinidamente. Esto enmascara bugs y permite crashes
silenciosos. ``ThreadSupervisor`` agrega:

  * Counter de fallas consecutivas; al superar ``crit_failures`` →
    publica ``WARNING_SIGNAL severity=crit`` y aplica backoff
    exponencial (1s, 2s, 4s, 8s, max 30s).
  * Tras ``max_total_failures`` totales → thread marcado ``dead``,
    deja de reintentar.
  * Heartbeat publish cada ``heartbeat_period_s`` con
    ``{thread_name, last_loop_ts, latency_ms, loop_count, ok}``.
  * Watchdog: si ``time.monotonic() - last_iter_start > max_iter_s``,
    heartbeat publica ``ok=False reason="iter_stuck"``.

API:
  * ``begin_iter()`` — antes de cada iteración.
  * ``end_iter_ok()`` — al finalizar con éxito.
  * ``end_iter_failed(exc)`` — al detectar excepción.
  * ``should_skip_iter()`` — True si está en backoff.

El thread driver usa esto desde ``ThreadWithStop.run()`` — único punto
de toque para todas las subclases.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SupervisorConfig:
    """Tunables del supervisor. Cada subclase de ThreadWithStop puede ajustar."""

    crit_failures: int = 3                # fallas consecutivas para WARNING crit
    max_total_failures: int = 100         # tope antes de marcar dead
    backoff_base_s: float = 1.0           # 1s en la primera falla; 2s, 4s, 8s, ...
    backoff_max_s: float = 30.0           # techo del backoff
    heartbeat_period_s: float = 1.0       # 1Hz típico
    max_iter_s: float = 5.0               # watchdog default — override en lentos


class ThreadSupervisor:
    """Estado interno + helpers del supervisor para un thread."""

    def __init__(
        self,
        *,
        thread_name: str,
        config: SupervisorConfig | None = None,
        heartbeat_publisher=None,  # callable(payload_dict) o None
        warning_publisher=None,    # callable(payload_dict) o None
    ) -> None:
        self.thread_name = thread_name
        self.config = config or SupervisorConfig()
        self._consecutive_failures = 0
        self._total_failures = 0
        self._dead = False
        self._backoff_until_monotonic = 0.0
        self._loop_count = 0
        self._last_iter_start_monotonic: float = 0.0
        self._last_iter_end_monotonic: float = 0.0
        self._last_heartbeat_monotonic = 0.0
        self._last_loop_ok = True
        self._heartbeat_publisher = heartbeat_publisher
        self._warning_publisher = warning_publisher

    # ─── State queries ────────────────────────────────────────────────
    @property
    def is_dead(self) -> bool:
        return self._dead

    def should_skip_iter(self) -> bool:
        """True si todavía estamos en backoff."""
        now = time.monotonic()
        return now < self._backoff_until_monotonic

    # ─── Lifecycle hooks ──────────────────────────────────────────────
    def begin_iter(self) -> None:
        self._last_iter_start_monotonic = time.monotonic()

    def end_iter_ok(self) -> None:
        self._last_iter_end_monotonic = time.monotonic()
        self._loop_count += 1
        if self._consecutive_failures > 0:
            self._consecutive_failures = 0  # reset al primer éxito
        self._last_loop_ok = True
        self._maybe_publish_heartbeat(ok=True)

    def end_iter_failed(self, exc: Exception) -> None:
        self._last_iter_end_monotonic = time.monotonic()
        self._consecutive_failures += 1
        self._total_failures += 1
        self._last_loop_ok = False
        # Backoff exponencial cap
        delay = min(
            self.config.backoff_max_s,
            self.config.backoff_base_s * (2 ** (self._consecutive_failures - 1)),
        )
        self._backoff_until_monotonic = time.monotonic() + delay

        if self._consecutive_failures >= self.config.crit_failures:
            self._emit_warning(severity="crit", reason=str(exc) or type(exc).__name__)
        if self._total_failures >= self.config.max_total_failures:
            self._dead = True
            self._emit_warning(severity="dead", reason="max_total_failures_exceeded")
        self._maybe_publish_heartbeat(ok=False, reason="thread_work_failed")

    def watchdog_check(self) -> None:
        """Plan G2: chequea si la iteración corriente lleva > max_iter_s.

        Llamar periódicamente desde un timer auxiliar o en cada heartbeat.
        Si dispara, publica heartbeat con ok=False reason='iter_stuck'.
        """
        if self._last_iter_start_monotonic == 0:
            return
        now = time.monotonic()
        if now - self._last_iter_start_monotonic > self.config.max_iter_s:
            self._maybe_publish_heartbeat(ok=False, reason="iter_stuck")

    # ─── Heartbeat ────────────────────────────────────────────────────
    def _maybe_publish_heartbeat(self, *, ok: bool, reason: str = "") -> None:
        now = time.monotonic()
        if now - self._last_heartbeat_monotonic < self.config.heartbeat_period_s:
            # Aún no toca, sólo actualizamos el flag interno
            return
        self._last_heartbeat_monotonic = now
        if self._heartbeat_publisher is None:
            return
        latency_ms = (
            (self._last_iter_end_monotonic - self._last_iter_start_monotonic) * 1000.0
            if self._last_iter_start_monotonic > 0 and self._last_iter_end_monotonic > 0
            else 0.0
        )
        payload = {
            "thread_name": self.thread_name,
            "last_loop_ts": float(self._last_iter_end_monotonic),
            "latency_ms": float(latency_ms),
            "loop_count": int(self._loop_count),
            "ok": bool(ok and not self._dead),
            "reason": str(reason),
            "consecutive_failures": int(self._consecutive_failures),
            "dead": bool(self._dead),
        }
        try:
            self._heartbeat_publisher(payload)
        except Exception:
            pass

    def _emit_warning(self, *, severity: str, reason: str) -> None:
        if self._warning_publisher is None:
            return
        try:
            self._warning_publisher({
                "thread": self.thread_name,
                "severity": severity,
                "reason": reason,
                "consecutive_failures": self._consecutive_failures,
            })
        except Exception:
            pass


__all__ = ["ThreadSupervisor", "SupervisorConfig"]
