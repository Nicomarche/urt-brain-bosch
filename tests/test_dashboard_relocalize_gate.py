from types import SimpleNamespace

from src.dashboard.processDashboard import processDashboard
from src.statemachine.systemMode import SystemMode


def test_dashboard_initial_mode_allows_manual_relocalization() -> None:
    initial_mode = processDashboard._initial_mode()

    assert initial_mode is SystemMode.STOP
    assert processDashboard._mode_allows_manual_localisation(initial_mode) is True


def test_dashboard_syncs_current_mode_from_state_machine() -> None:
    dashboard = processDashboard.__new__(processDashboard)
    dashboard.stateMachine = SimpleNamespace(_shared_state={"mode": SystemMode.AUTO})
    dashboard._current_mode = SystemMode.STOP

    dashboard._sync_current_mode_from_state_machine()

    assert dashboard._current_mode is SystemMode.AUTO
