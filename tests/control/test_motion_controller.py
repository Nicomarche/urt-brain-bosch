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

from src.control.motion_controller import MotionController
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


def _bo(
    *,
    valid: bool = True,
    stop: bool = False,
    n: int = 5,
    speed_mps: float = 0.30,
) -> BehaviorOutput:
    return BehaviorOutput(
        timestamp=0.0,
        dt=0.05,
        target_path=np.zeros((n + 1, 3)),
        speed_profile=np.full(n, speed_mps),
        scenario_name=ScenarioName.LANE_KEEP.value,
        valid=valid,
        stop_required=stop,
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


def test_negative_speed_clamped_to_zero() -> None:
    """BFMC no usa reverse — el controller fuerza velocidad >= 0."""
    mc = MotionController(solver=_FakeSolver(result=(-0.20, 1.0)))
    cmd = mc.compute(_bo(), _pose())
    assert cmd.speed_mps == 0.0


def test_steering_clamped_to_max() -> None:
    """Si el solver entrega 40°, se reclampa a +25 (max_steering_deg)."""
    mc = _disable_rate_limits(MotionController(
        solver=_FakeSolver(result=(0.30, 40.0)),
        max_steering_deg=25.0,
    ))
    cmd = mc.compute(_bo(), _pose())
    assert cmd.steering_deg == pytest.approx(25.0)


def test_solver_speed_is_capped_to_current_planner_request() -> None:
    """El controller nunca debe superar `speed_profile[0]`."""
    mc = _disable_rate_limits(MotionController(solver=_FakeSolver(result=(0.50, 1.0))))
    cmd = mc.compute(_bo(speed_mps=0.10), _pose())
    assert cmd.valid is True
    assert cmd.speed_mps == pytest.approx(0.10)


def test_speed_rate_limit_only_applies_when_accelerating() -> None:
    """Si el solver acelera fuerte, el controller sube gradual."""
    mc = MotionController(solver=_FakeSolver(result=(0.30, 0.0)))
    cmd = mc.compute(_bo(speed_mps=0.30), _pose())
    assert cmd.speed_mps == pytest.approx(0.0125)


def test_reset_propagates_to_solver() -> None:
    fake = _FakeSolver()
    mc = MotionController(solver=fake)
    mc.reset()
    assert fake.reset_calls == 1


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


def test_motion_controller_falls_back_when_acados_unavailable() -> None:
    """Sin solver inyectado y sin AcadosMPC listo, se usa PurePursuitSolver.

    Este test ata el contrato Phase-6: el sim/dev (sin acados instalado
    ni código C generado) DEBE seguir pudiendo cerrar el lazo de control.
    Si esto se rompe, el dispatcher emite speed=0 indefinidamente y se
    pierde la verificación end-to-end en Gazebo.
    """
    mc = MotionController(fallback_horizon_n=20)
    assert mc.ready is True
    assert isinstance(mc._solver, PurePursuitSolver)
    assert mc.horizon == 20
