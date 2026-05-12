# tests/control/test_motion_controller.py
#
# Tests del MotionController (capa de contrato) usando un AcadosMPC fake.
# Validamos:
#   1. behavior_output.valid=False → MotorCommand inválido "behavior_invalid".
#   2. stop_required=True → MotorCommand válido con stop (s=0, v=0).
#   3. solver no listo → MotorCommand inválido "mpc_solver_not_ready".
#   4. dimensiones target_path/speed_profile mismatched → "dimension_mismatch_*".
#   5. solver retorna None → "mpc_solver_failure".
#   6. solver retorna NaN → "mpc_nonfinite_output".
#   7. happy path → MotorCommand válido con valores del solver.
#   8. velocidad negativa del solver se clampa a 0 (no reverse en BFMC).
#   9. steering del solver fuera de [-25, +25] se reclampa.
#  10. reset() llama al solver.

from __future__ import annotations

import math

import numpy as np
import pytest

import src.control.motion_controller as motion_controller_mod
from src.control.motion_controller import MotionController, PurePursuitSolver
from src.core.types.behavior import BehaviorOutput, ScenarioName
from src.core.types.control import MotorCommand
from src.core.types.pose import Pose2D, PoseEstimate


# ---------- fake solver --------------------------------------------------


class _FakeSolver:
    """Stub de AcadosMPC para tests deterministas."""

    def __init__(
        self,
        *,
        ready: bool = True,
        N: int = 5,
        result=(0.30, 5.0),  # (v_mps, delta_deg)
        return_none: bool = False,
    ) -> None:
        self.ready = ready
        self._N = N
        self._result = result
        self._return_none = return_none
        self.compute_calls = 0
        self.reset_calls = 0
        self._debug = {"fake": True}

    def compute(self, x_current, state_refs, input_refs, **_):
        self.compute_calls += 1
        if self._return_none:
            return None
        return self._result

    def reset(self) -> None:
        self.reset_calls += 1

    @property
    def N(self) -> int:
        return self._N

    @property
    def debug(self) -> dict:
        return dict(self._debug)


class _CaptureSolver(_FakeSolver):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.last_x_current = None
        self.last_state_refs = None
        self.last_input_refs = None
        self.last_kwargs = None

    def compute(self, x_current, state_refs, input_refs, **kwargs):
        self.last_x_current = np.asarray(x_current, dtype=np.float64).copy()
        self.last_state_refs = np.asarray(state_refs, dtype=np.float64).copy()
        self.last_input_refs = np.asarray(input_refs, dtype=np.float64).copy()
        self.last_kwargs = dict(kwargs)
        return super().compute(x_current, state_refs, input_refs, **kwargs)


def _bo(
    *,
    valid: bool = True,
    stop: bool = False,
    n: int = 5,
    speed_mps: float = 0.30,
    target_path: np.ndarray | None = None,
    notes: dict | None = None,
) -> BehaviorOutput:
    return BehaviorOutput(
        timestamp=0.0,
        dt=0.05,
        target_path=np.zeros((n + 1, 3)) if target_path is None else target_path,
        speed_profile=np.full(n, speed_mps),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=valid,
        stop_required=stop,
        notes=dict(notes or {}),
    )


def _pose() -> PoseEstimate:
    return PoseEstimate(fused_pose=Pose2D(x=1.0, y=2.0, yaw=0.5))


def _disable_rate_limits(mc: MotionController) -> MotionController:
    mc._max_speed_rate_mps2 = 1e6
    mc._max_steer_rate_deg_s = 1e6
    return mc


# ---------- tests --------------------------------------------------------


def test_invalid_behavior_returns_invalid_command() -> None:
    mc = MotionController(solver=_FakeSolver())
    cmd = mc.compute(_bo(valid=False), _pose())
    assert cmd.valid is False
    assert cmd.reason == "behavior_invalid"
    assert cmd.source == "motion_controller"


def test_stop_required_returns_stop_command() -> None:
    mc = MotionController(solver=_FakeSolver())
    cmd = mc.compute(_bo(stop=True), _pose())
    assert cmd.valid is True
    assert cmd.steering_deg == 0.0
    assert cmd.speed_mps == 0.0
    assert cmd.reason == "stop_required"


def test_solver_not_ready_returns_invalid() -> None:
    mc = MotionController(solver=_FakeSolver(ready=False))
    cmd = mc.compute(_bo(), _pose())
    assert cmd.valid is False
    assert cmd.reason == "mpc_solver_not_ready"


def test_dimension_mismatch_target_path() -> None:
    mc = MotionController(solver=_FakeSolver(N=5))
    bad = BehaviorOutput(
        target_path=np.zeros((4, 3)),  # debería ser (6, 3) si speed_profile.shape=5
        speed_profile=np.full(5, 0.3),
        valid=True,
    )
    cmd = mc.compute(bad, _pose())
    assert cmd.valid is False
    assert cmd.reason == "dimension_mismatch_target_path"


def test_horizon_mismatch_with_solver() -> None:
    mc = MotionController(solver=_FakeSolver(N=10))
    cmd = mc.compute(_bo(n=5), _pose())  # planner emite N=5, solver espera N=10
    assert cmd.valid is False
    assert cmd.reason == "horizon_mismatch"


def test_solver_returns_none_marks_failure() -> None:
    mc = MotionController(solver=_FakeSolver(return_none=True))
    cmd = mc.compute(_bo(), _pose())
    assert cmd.valid is False
    assert cmd.reason == "mpc_solver_failure"


def test_solver_nan_marks_nonfinite() -> None:
    mc = MotionController(solver=_FakeSolver(result=(float("nan"), 5.0)))
    cmd = mc.compute(_bo(), _pose())
    assert cmd.valid is False
    assert cmd.reason == "mpc_nonfinite_output"


def test_happy_path_returns_solver_values() -> None:
    mc = _disable_rate_limits(MotionController(solver=_FakeSolver(result=(0.25, -3.5))))
    cmd = mc.compute(_bo(), _pose())
    assert cmd.valid is True
    assert cmd.source == "motion_controller"
    assert cmd.speed_mps == pytest.approx(0.25)
    assert cmd.steering_deg == pytest.approx(-3.5)


def test_compute_out_logs_backend_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict]] = []
    mc = _disable_rate_limits(MotionController(solver=_FakeSolver(result=(0.25, -3.5))))
    monkeypatch.setattr(
        motion_controller_mod,
        "live_log",
        lambda thread, **payload: events.append((thread, payload)),
    )

    cmd = mc.compute(_bo(), _pose())

    assert cmd.valid is True
    compute_out = [payload for thread, payload in events if thread == "mpc" and payload.get("event") == "compute_out"]
    assert compute_out
    assert compute_out[-1]["backend"] == "fakesolver"


def test_negative_speed_clamped_to_zero() -> None:
    """BFMC no usa reverse — el controller fuerza velocidad >= 0."""
    mc = MotionController(solver=_FakeSolver(result=(-0.20, 1.0)))
    cmd = mc.compute(_bo(), _pose())
    assert cmd.speed_mps == 0.0


def test_negative_solver_speed_gets_forward_recovery_on_route_path() -> None:
    """Si la ruta tiene una referencia adelante, no dejamos al auto clavado."""
    target_path = np.array(
        [
            [1.0, 2.0, 0.5],
            [1.08, 2.04, 0.52],
            [1.16, 2.08, 0.52],
            [1.24, 2.12, 0.52],
            [1.32, 2.16, 0.52],
            [1.40, 2.20, 0.52],
        ],
        dtype=float,
    )
    mc = _disable_rate_limits(MotionController(solver=_FakeSolver(result=(-0.05, 1.0))))

    cmd = mc.compute(
        _bo(
            speed_mps=0.20,
            target_path=target_path,
            notes={"path_source": "route_waypoints"},
        ),
        _pose(),
    )

    assert cmd.valid is True
    assert cmd.speed_mps == pytest.approx(0.20)


def test_steering_clamped_to_max() -> None:
    """Si el solver entrega 40°, se reclampa a +25 (max_steering_deg)."""
    mc = _disable_rate_limits(MotionController(
        solver=_FakeSolver(result=(0.30, 40.0)),
        max_steering_deg=25.0,
    ))
    cmd = mc.compute(_bo(), _pose())
    assert cmd.steering_deg == pytest.approx(25.0)


def test_solver_speed_is_capped_to_current_planner_request() -> None:
    """El controller no supera un `speed_profile[0]` competitivo."""
    mc = _disable_rate_limits(MotionController(solver=_FakeSolver(result=(0.50, 1.0))))
    cmd = mc.compute(_bo(speed_mps=0.30), _pose())
    assert cmd.valid is True
    assert cmd.speed_mps == pytest.approx(0.30)


def test_motion_controller_converts_osm_frame_before_calling_solver() -> None:
    solver = _CaptureSolver(result=(0.10, 4.0))
    mc = _disable_rate_limits(MotionController(solver=solver))
    pose = PoseEstimate(fused_pose=Pose2D(x=4.0120, y=5.9004, yaw=0.0))
    behavior_output = BehaviorOutput(
        timestamp=0.0,
        dt=0.05,
        target_path=np.array(
            [
                [4.0120, 5.9004, 0.0],
                [4.5017, 5.9004, 0.0],
                [4.9900, 5.8987, -0.0091],
                [5.4764, 5.8942, -0.0099],
                [5.9424, 5.8896, -0.0099],
                [6.3785, 5.8225, -0.3901],
            ],
            dtype=np.float64,
        ),
        speed_profile=np.full(5, 0.10),
        scenario_name=ScenarioName.INTERSECTION.value,
        valid=True,
    )

    cmd = mc.compute(behavior_output, pose)

    assert cmd.valid is True
    assert solver.last_x_current is not None
    assert solver.last_state_refs is not None
    assert solver.last_x_current.tolist() == pytest.approx([4.0120, -5.9004, 0.0])
    assert solver.last_state_refs[0].tolist() == pytest.approx([4.0120, -5.9004, 0.0])
    assert solver.last_state_refs[-1].tolist() == pytest.approx([6.3785, -5.8225, 0.3901])


def test_acados_reference_is_retimed_by_speed_profile_and_uses_commanded_previous_input() -> None:
    solver = _CaptureSolver(result=(0.20, 2.0), N=5)
    mc = _disable_rate_limits(MotionController(solver=solver))
    mc._backend_name = "acados"
    mc._acados_reference_max_preview_m = 10.0
    mc._prev_speed_mps = 0.12
    mc._prev_steer_deg = 3.0
    pose = PoseEstimate(fused_pose=Pose2D(x=0.0, y=0.0, yaw=0.0))
    behavior_output = BehaviorOutput(
        timestamp=0.0,
        dt=1.0,
        target_path=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        speed_profile=np.full(5, 0.20),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=True,
    )

    cmd = mc.compute(behavior_output, pose)

    assert cmd.valid is True
    assert solver.last_state_refs is not None
    assert solver.last_state_refs[:, 0].tolist() == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert solver.last_state_refs[:, 1].tolist() == pytest.approx([0.0] * 6)
    assert solver.last_kwargs["v_prev"] == pytest.approx(0.12)
    assert solver.last_kwargs["delta_prev"] == pytest.approx(math.radians(3.0))


def test_visual_left_route_in_osm_frame_produces_positive_left_steer() -> None:
    solver = PurePursuitSolver(
        horizon_n=5,
        min_lookahead_m=0.90,
        lookahead_gain_s=0.0,
        max_steering_deg=25.0,
        output_deadband_deg=0.0,
    )
    mc = _disable_rate_limits(MotionController(solver=solver))
    pose = PoseEstimate(fused_pose=Pose2D(x=4.0120, y=5.9004, yaw=0.0))
    behavior_output = BehaviorOutput(
        timestamp=0.0,
        dt=0.05,
        target_path=np.array(
            [
                [4.0120, 5.9004, 0.0],
                [4.2500, 5.9004, 0.0],
                [4.5000, 5.9004, 0.0],
                [4.7600, 5.8600, -0.1600],
                [4.9800, 5.7600, -0.4200],
                [5.1600, 5.5800, -0.9000],
            ],
            dtype=np.float64,
        ),
        speed_profile=np.full(5, 0.10),
        scenario_name=ScenarioName.INTERSECTION.value,
        valid=True,
    )

    cmd = mc.compute(behavior_output, pose)

    assert cmd.valid is True
    assert cmd.steering_deg > 0.0


def test_speed_rate_limit_only_applies_when_accelerating() -> None:
    """El rate limiter no debe bajar un comando de avance debajo del mínimo."""
    mc = MotionController(solver=_FakeSolver(result=(0.30, 0.0)))
    cmd = mc.compute(_bo(speed_mps=0.30), _pose())
    assert cmd.speed_mps == pytest.approx(0.20)


def test_reset_propagates_to_solver() -> None:
    fake = _FakeSolver()
    mc = MotionController(solver=fake)
    mc.reset()
    assert fake.reset_calls == 1


def test_motion_controller_logs_backend_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict]] = []

    class _ReadyAcados:
        def __init__(self, *_, **__):
            self.ready = True
            self.load_error = None
            self._N = 20

        def update_weights(self, **_kwargs):
            return None

        def update_bounds(self, **_kwargs):
            return None

        def reset(self):
            return None

        @property
        def N(self) -> int:
            return self._N

        @property
        def debug(self) -> dict:
            return {"solver": "acados"}

    import config as cfg

    monkeypatch.setattr(cfg, "FORCE_PURE_PURSUIT", False, raising=False)
    monkeypatch.setattr(motion_controller_mod, "AcadosMPC", _ReadyAcados)
    monkeypatch.setattr(
        motion_controller_mod,
        "live_log",
        lambda thread, **payload: events.append((thread, payload)),
    )

    mc = MotionController()
    assert mc.ready is True
    assert mc.horizon == 20
    assert events
    assert events[0][0] == "mpc"
    assert events[0][1]["event"] == "backend_selected"
    assert events[0][1]["backend"] == "acados"
    assert events[0][1]["acados_ready"] is True


def test_motion_controller_requires_acados_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    class _UnavailableAcados:
        def __init__(self, *_, **__):
            self.ready = False
            self.load_error = "mock acados unavailable"
            self._N = 0

        @property
        def N(self) -> int:
            return self._N

        @property
        def debug(self) -> dict:
            return {}

        def reset(self):
            return None

    import config as cfg

    monkeypatch.setattr(cfg, "FORCE_PURE_PURSUIT", False, raising=False)
    monkeypatch.setattr(motion_controller_mod, "AcadosMPC", _UnavailableAcados)

    with pytest.raises(motion_controller_mod.AcadosBackendUnavailableError, match="Acados is required"):
        MotionController()


def test_acados_straight_lateral_offset_does_not_bang_bang() -> None:
    """Regression: small straight-line offsets must not flip steering left/right."""
    import config as cfg

    solver = motion_controller_mod.AcadosMPC(
        output_deadband_deg=float(getattr(cfg, "ACADOS_MPC_OUTPUT_DEADBAND_DEG", 0.75))
    )
    if not solver.ready:
        pytest.skip(f"Acados solver not ready: {solver.load_error}")

    solver.update_weights(
        x_cost=float(getattr(cfg, "ACADOS_MPC_X_COST", 2.0)),
        y_cost=float(getattr(cfg, "ACADOS_MPC_Y_COST", 5.0)),
        yaw_cost=float(getattr(cfg, "ACADOS_MPC_YAW_COST", 1.0)),
        v_cost=float(getattr(cfg, "ACADOS_MPC_V_COST", 1.0)),
        steer_cost=float(getattr(cfg, "ACADOS_MPC_STEER_COST", 2.0)),
        delta_v_cost=float(getattr(cfg, "ACADOS_MPC_DELTA_V_COST", 4.0)),
        delta_steer_cost=float(getattr(cfg, "ACADOS_MPC_DELTA_STEER_COST", 5.0)),
    )
    delta_max_rad = math.radians(float(getattr(cfg, "ACADOS_MPC_DELTA_MAX_DEG", 25.0)))
    solver.update_bounds(
        v_min=-0.05,
        v_max=float(getattr(cfg, "ACADOS_MPC_V_MAX", 0.40)),
        delta_min_rad=-delta_max_rad,
        delta_max_rad=delta_max_rad,
    )

    state = np.array([0.0, -0.05, 0.0], dtype=np.float64)
    dt_s = 0.05
    wheelbase_m = 0.258
    deltas: list[float] = []
    for _ in range(40):
        state_refs = np.column_stack(
            [
                state[0] + np.linspace(0.0, 4.0, solver.N + 1),
                np.zeros(solver.N + 1),
                np.zeros(solver.N + 1),
            ]
        )
        input_refs = np.column_stack(
            [
                np.full(solver.N, 0.20),
                np.zeros(solver.N),
            ]
        )
        result = solver.compute(state, state_refs, input_refs)
        assert result is not None
        _, delta_deg = result
        deltas.append(float(delta_deg))

        delta_rad = math.radians(float(delta_deg))
        state[0] += 0.20 * math.cos(float(state[2])) * dt_s
        state[1] += 0.20 * math.sin(float(state[2])) * dt_s
        state[2] += (0.20 / wheelbase_m) * math.tan(delta_rad) * dt_s

    sign_changes = sum(1 for a, b in zip(deltas, deltas[1:]) if a * b < 0.0)
    assert sign_changes == 0
    assert max(abs(delta) for delta in deltas) < 5.0


# ---------- PurePursuitSolver (fallback usado en sim/dev sin acados) ------


from src.control.motion_controller import PurePursuitSolver


def test_pure_pursuit_solver_is_always_ready() -> None:
    """Sin acados el lazo de control sigue cerrado vía pure-pursuit."""
    s = PurePursuitSolver(horizon_n=20)
    assert s.ready is True
    assert s.N == 20


def test_pure_pursuit_straight_path_yields_zero_steering() -> None:
    """Path recto + auto alineado ⇒ δ ≈ 0 (dentro de la deadband)."""
    s = PurePursuitSolver(horizon_n=20, output_deadband_deg=0.5)
    state_refs = np.column_stack([
        np.linspace(0.0, 2.0, 21),
        np.zeros(21),
        np.zeros(21),
    ])
    input_refs = np.column_stack([np.full(20, 0.5), np.zeros(20)])
    x0 = np.array([0.0, 0.0, 0.0])
    out = s.compute(x0, state_refs, input_refs)
    assert out is not None
    v, delta = out
    assert v == pytest.approx(0.5)
    assert delta == pytest.approx(0.0)


def test_pure_pursuit_left_curve_yields_positive_steering() -> None:
    """Path con goal a la izquierda del auto ⇒ δ > 0 (BFMC: +=izquierda)."""
    s = PurePursuitSolver(horizon_n=20)
    # Goal a 1m al frente y 0.2m a la izquierda. La pendiente es suave
    # (~11°) para que el resultado no sature en la cota mecánica.
    state_refs = np.column_stack([
        np.linspace(0.0, 2.0, 21),
        np.linspace(0.0, 0.4, 21),  # +y monotónicamente
        np.full(21, math.atan(0.2)),
    ])
    input_refs = np.column_stack([np.full(20, 0.5), np.zeros(20)])
    x0 = np.array([0.0, 0.0, 0.0])
    out = s.compute(x0, state_refs, input_refs)
    assert out is not None
    _, delta = out
    assert delta > 1.0, f"expected positive steering, got {delta}"


def test_pure_pursuit_right_curve_yields_negative_steering() -> None:
    """Path con goal a la derecha ⇒ δ < 0 (signo coherente con manual)."""
    s = PurePursuitSolver(horizon_n=20)
    state_refs = np.column_stack([
        np.linspace(0.0, 2.0, 21),
        np.linspace(0.0, -0.4, 21),  # -y = derecha
        np.full(21, -math.atan(0.2)),
    ])
    input_refs = np.column_stack([np.full(20, 0.5), np.zeros(20)])
    x0 = np.array([0.0, 0.0, 0.0])
    out = s.compute(x0, state_refs, input_refs)
    assert out is not None
    _, delta = out
    assert delta < -1.0, f"expected negative steering, got {delta}"


def test_motion_controller_can_force_pure_pursuit(monkeypatch: pytest.MonkeyPatch) -> None:
    import config as cfg

    monkeypatch.setattr(cfg, "FORCE_PURE_PURSUIT", True, raising=False)
    mc = MotionController(fallback_horizon_n=20)
    assert mc.ready is True
    assert isinstance(mc._solver, PurePursuitSolver)
    assert mc.horizon == 20
