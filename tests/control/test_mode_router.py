"""``threadModeRouter`` test: GUI publishes ``DrivingMode`` → brain runs
``StateMachine.request_mode`` (which updates the Manager shared state AND
publishes ``StateChange``).

This test runs both ends in the same process and uses real Manager
proxies so the routing behaviour is exercised end-to-end.
"""

from __future__ import annotations

import time

import pytest

from src.core.bus.shim import messageHandlerSender
from src.core.bus.topics import DRIVING_MODE
from src.hardware.serialhandler.threads.threadModeRouter import threadModeRouter
from src.statemachine.stateMachine import StateMachine
from src.statemachine.systemMode import SystemMode


@pytest.fixture
def state_machine(broker):
    """Initialise the StateMachine singleton with fresh Manager-backed state
    so each test starts in STOP. ``broker`` runs first so messageHandler*
    objects can publish.

    ``initialize_starting_mode`` honours ``AUTO_STATE_RUN_IN_SIM`` which is
    True for dev — we override the shared dict directly to land in STOP
    so the test starts from the same known state every run.
    """
    StateMachine.initialize_shared_state({})
    StateMachine._shared_state["mode"] = SystemMode.STOP
    yield StateMachine.get_instance()


def test_mode_router_runs_request_mode(state_machine):
    """Publish ``DrivingMode="auto"`` → router calls
    ``StateMachine.request_mode("dashboard_auto_button")`` → shared state
    transitions STOP → AUTO.

    We assert on the Manager shared dict because that's the contract the
    legacy dashboard exposed to ``motor_command_dispatcher`` (which reads
    ``queueList["__sm_state__"]["mode"]`` directly). The StateChange bus
    publish is exercised indirectly by other tests — slow-joiner makes it
    flaky to assert here in isolation.
    """
    assert StateMachine._shared_state["mode"] is SystemMode.STOP

    router = threadModeRouter({}, debugging=False)
    sender = messageHandlerSender({}, DRIVING_MODE)

    time.sleep(0.1)  # subscriber filter propagation
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        sender.send("auto")
        time.sleep(0.05)
        router.thread_work()  # drain + dispatch
        if StateMachine._shared_state["mode"] is SystemMode.AUTO:
            break

    router._sub.unsubscribe()

    assert StateMachine._shared_state["mode"] is SystemMode.AUTO


def test_mode_router_ignores_unknown_value(state_machine, caplog):
    """An unknown ``DrivingMode`` value should be logged and dropped,
    keeping the StateMachine pristine.
    """
    router = threadModeRouter({}, debugging=False)
    sender = messageHandlerSender({}, DRIVING_MODE)

    time.sleep(0.1)
    for _ in range(20):
        sender.send("bogus_mode")
        time.sleep(0.02)
        router.thread_work()

    router._sub.unsubscribe()
    # StateMachine should still be at the initial mode (STOP).
    assert StateMachine._shared_state["mode"] is SystemMode.STOP
