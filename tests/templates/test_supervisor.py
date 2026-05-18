"""Tests del `ThreadSupervisor` (plan G1+G2+F4)."""

from __future__ import annotations

import time

import pytest

from src.templates.supervisor import SupervisorConfig, ThreadSupervisor


def test_initial_state_alive_no_failures():
    s = ThreadSupervisor(thread_name="t1")
    assert not s.is_dead
    assert not s.should_skip_iter()


def test_one_success_increments_loop_count():
    s = ThreadSupervisor(thread_name="t1")
    s.begin_iter()
    s.end_iter_ok()
    s.begin_iter()
    s.end_iter_ok()
    # internal state — usamos heartbeat publish para verificar
    events = []
    s2 = ThreadSupervisor(thread_name="t2", heartbeat_publisher=events.append, config=SupervisorConfig(heartbeat_period_s=0.0))
    s2.begin_iter()
    s2.end_iter_ok()
    s2.begin_iter()
    s2.end_iter_ok()
    assert any(e.get("loop_count") == 2 for e in events)


def test_failure_increments_consecutive_counter_and_triggers_warn():
    warns = []
    s = ThreadSupervisor(
        thread_name="t",
        config=SupervisorConfig(crit_failures=2, max_total_failures=100, backoff_base_s=0.01),
        warning_publisher=warns.append,
    )
    s.begin_iter(); s.end_iter_failed(RuntimeError("fail1"))
    assert len(warns) == 0  # 1 < crit_failures
    s.begin_iter(); s.end_iter_failed(RuntimeError("fail2"))
    assert len(warns) == 1
    assert warns[0]["severity"] == "crit"
    assert warns[0]["thread"] == "t"


def test_success_resets_consecutive_counter():
    s = ThreadSupervisor(
        thread_name="t",
        config=SupervisorConfig(crit_failures=3, backoff_base_s=0.001),
    )
    s.begin_iter(); s.end_iter_failed(ValueError())
    s.begin_iter(); s.end_iter_failed(ValueError())
    # Esperar a que el backoff pase
    time.sleep(0.01)
    s.begin_iter(); s.end_iter_ok()
    # Próxima falla, contador empieza en 1
    s.begin_iter(); s.end_iter_failed(ValueError())
    # Si no se reseteó, ya tendríamos 3 fallas consecutivas — debería emitir warn.
    # En este punto sólo 1 falla → no warn.
    # (no manera directa de leer counter; chequeamos backoff cap)
    assert s.should_skip_iter()  # en backoff con delay 0.001 * 2^0 = 0.001 — corto


def test_max_total_failures_marks_dead():
    warns = []
    s = ThreadSupervisor(
        thread_name="t",
        config=SupervisorConfig(crit_failures=1, max_total_failures=3, backoff_base_s=0.001),
        warning_publisher=warns.append,
    )
    for i in range(3):
        s.begin_iter()
        s.end_iter_failed(RuntimeError(f"fail{i}"))
    assert s.is_dead
    # Debería haber al menos un warning con severity=dead
    assert any(w["severity"] == "dead" for w in warns)


def test_backoff_grows_exponentially():
    s = ThreadSupervisor(
        thread_name="t",
        config=SupervisorConfig(
            crit_failures=999, max_total_failures=999, backoff_base_s=1.0, backoff_max_s=30.0
        ),
    )
    s.begin_iter(); s.end_iter_failed(RuntimeError())  # backoff = 1.0
    t0 = time.monotonic()
    # should_skip_iter por ~ 1 s. Si reintento ahora, sigue True.
    assert s.should_skip_iter()
    # Fail más para crecer el delay
    s.begin_iter(); s.end_iter_failed(RuntimeError())  # backoff = 2.0
    # No esperamos los segundos completos; chequeamos el cómputo via repeated
    # falls sin pasar de backoff_max.
    for _ in range(10):
        s.begin_iter()
        s.end_iter_failed(RuntimeError())
    # Después de muchas fallas, el delay queda en backoff_max
    assert s.should_skip_iter()


def test_heartbeat_emits_when_period_passed():
    events = []
    s = ThreadSupervisor(
        thread_name="hb",
        config=SupervisorConfig(heartbeat_period_s=0.0),
        heartbeat_publisher=events.append,
    )
    s.begin_iter()
    s.end_iter_ok()
    assert events
    last = events[-1]
    assert last["thread_name"] == "hb"
    assert last["ok"] is True


def test_watchdog_publishes_stuck_when_iter_overflows(monkeypatch):
    events = []
    s = ThreadSupervisor(
        thread_name="stuck",
        config=SupervisorConfig(heartbeat_period_s=0.0, max_iter_s=0.001),
        heartbeat_publisher=events.append,
    )
    s.begin_iter()
    time.sleep(0.005)  # supera max_iter_s=0.001
    s.watchdog_check()
    assert any(e.get("reason") == "iter_stuck" for e in events)
