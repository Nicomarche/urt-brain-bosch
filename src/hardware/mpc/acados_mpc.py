"""Runtime wrapper for the Acados-generated full MPC solver.

Usage (from threadLineFollowing or any control loop):

    from src.hardware.mpc.acados_mpc import AcadosMPC

    mpc = AcadosMPC()                       # loads generated solver
    v_opt, delta_deg = mpc.compute(
        x_current   = [x, y, yaw],          # current pose
        state_refs  = state_refs,            # (N+1, 3) waypoints [x, y, yaw]
        input_refs  = input_refs,            # (N, 2)   refs [v_ref, delta_ref]
        v_prev      = last_v_cmd,
        delta_prev  = last_delta_rad,
    )

Falls back to ``None`` if acados is not installed or the solver was never
generated.  The caller is responsible for checking the return value and
falling back to the lateral MPC / Stanley.
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import numpy as np

_SOLVER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_generated_code")
_JSON_FILE = os.path.join(_SOLVER_DIR, "acados_ocp_bfmc_bicycle.json")

try:
    from acados_template import AcadosOcpSolver
    _ACADOS_AVAILABLE = True
except ImportError:
    _ACADOS_AVAILABLE = False


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class AcadosMPC:
    """Full-trajectory MPC using an Acados-generated solver.

    The solver is loaded from C code in ``c_generated_code/``.  If the code
    has not been generated yet, ``ready`` will be ``False`` and ``compute``
    returns ``None``.

    Parameters
    ----------
    max_steering_deg : float
        Absolute steering limit (degrees).  Output is clamped to this.
    output_deadband_deg : float
        Steering commands smaller than this are zeroed.
    """

    def __init__(
        self,
        max_steering_deg: float = 25.0,
        output_deadband_deg: float = 0.5,
    ):
        self.max_steering_deg = float(max_steering_deg)
        self.output_deadband_deg = float(output_deadband_deg)
        self._solver: AcadosOcpSolver | None = None
        self._N: int = 0
        self.ready = False
        self._last_debug: dict = {}
        self._prev_v = 0.0
        self._prev_delta = 0.0

        if not _ACADOS_AVAILABLE:
            return
        if not os.path.isfile(_JSON_FILE):
            return

        try:
            self._solver = AcadosOcpSolver(None, json_file=_JSON_FILE)
            self._N = self._solver.acados_ocp.dims.N
            self.ready = True
        except Exception:
            self._solver = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        x_current: np.ndarray,
        state_refs: np.ndarray,
        input_refs: np.ndarray,
        v_prev: float | None = None,
        delta_prev: float | None = None,
    ) -> Tuple[float, float] | None:
        """Solve the OCP and return ``(v_optimal_mps, delta_optimal_deg)``.

        Parameters
        ----------
        x_current : (3,) array  [x, y, psi]
        state_refs : (N+1, 3) array  [x_ref, y_ref, psi_ref]
        input_refs : (N, 2) array  [v_ref, delta_ref]
        v_prev, delta_prev : previous commands (default: stored from last call)

        Returns ``None`` if the solver is not ready.
        """
        if not self.ready or self._solver is None:
            return None

        N = self._N
        x0 = np.asarray(x_current, dtype=np.float64).ravel()[:3]

        if state_refs is None or input_refs is None:
            return None
        sr = np.asarray(state_refs, dtype=np.float64)
        ir = np.asarray(input_refs, dtype=np.float64)
        if sr.shape[0] < N + 1 or ir.shape[0] < N:
            return None

        if v_prev is None:
            v_prev = self._prev_v
        if delta_prev is None:
            delta_prev = self._prev_delta
        p_val = np.array([float(v_prev), float(delta_prev)], dtype=np.float64)

        # ---- Unwrap yaw references to be continuous w.r.t. current yaw ----
        sr = sr.copy()
        ref_yaw = sr[:, 2].copy()
        # Make first reference yaw close to current yaw
        ref_yaw[0] = x0[2] + _wrap_angle(ref_yaw[0] - x0[2])
        for i in range(1, len(ref_yaw)):
            ref_yaw[i] = ref_yaw[i - 1] + _wrap_angle(ref_yaw[i] - ref_yaw[i - 1])
        sr[:, 2] = ref_yaw

        # ---- Set initial state constraint ----
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        # ---- Set stage references (0..N-1) ----
        for j in range(N):
            # yref = [x, y, psi, v, delta, Δv=0, Δδ=0]
            yref = np.zeros(7, dtype=np.float64)
            yref[0:3] = sr[j]
            yref[3] = ir[j, 0]       # v_ref
            yref[4] = ir[j, 1]       # delta_ref (typically 0)
            # yref[5:7] = 0          # delta rate refs always 0
            self._solver.set(j, "yref", yref)
            self._solver.set(j, "p", p_val)

        # ---- Terminal reference (stage N) ----
        yref_e = sr[N, :3].copy()
        self._solver.set(N, "yref", yref_e)

        # ---- Solve ----
        status = self._solver.solve()

        # ---- Extract first control ----
        u_opt = self._solver.get(0, "u")  # [v, delta]
        v_opt = float(u_opt[0])
        delta_rad = float(u_opt[1])

        # Store for next call
        self._prev_v = v_opt
        self._prev_delta = delta_rad

        # Convert to degrees and apply limits
        delta_deg = math.degrees(delta_rad)
        delta_deg = max(-self.max_steering_deg, min(self.max_steering_deg, delta_deg))
        if abs(delta_deg) < self.output_deadband_deg:
            delta_deg = 0.0

        self._last_debug = {
            "solver_status": int(status),
            "v_opt_mps": round(v_opt, 4),
            "delta_opt_deg": round(delta_deg, 3),
            "delta_opt_rad": round(delta_rad, 5),
            "x0": [round(float(c), 4) for c in x0],
            "ref_0": [round(float(c), 4) for c in sr[0]],
            "ref_N": [round(float(c), 4) for c in sr[N]],
            "v_ref_0": round(float(ir[0, 0]), 4),
        }

        return v_opt, delta_deg

    def update_weights(
        self,
        x_cost: float = 2.0,
        y_cost: float = 2.0,
        yaw_cost: float = 0.5,
        v_cost: float = 1.0,
        steer_cost: float = 0.0,
        delta_v_cost: float = 1.5,
        delta_steer_cost: float = 0.75,
    ) -> None:
        """Update cost weights at runtime (no need to regenerate solver)."""
        if not self.ready or self._solver is None:
            return
        W = np.zeros((7, 7))
        W[0, 0] = x_cost
        W[1, 1] = y_cost
        W[2, 2] = yaw_cost
        W[3, 3] = v_cost
        W[4, 4] = steer_cost
        W[5, 5] = delta_v_cost
        W[6, 6] = delta_steer_cost
        W_e = np.diag([x_cost, y_cost, yaw_cost])
        for j in range(self._N):
            self._solver.cost_set(j, "W", W)
        self._solver.cost_set(self._N, "W", W_e)

    def update_bounds(
        self,
        v_min: float = -0.5,
        v_max: float = 0.5,
        delta_min_rad: float = -0.436,
        delta_max_rad: float = 0.436,
    ) -> None:
        """Update control input bounds at runtime."""
        if not self.ready or self._solver is None:
            return
        lbu = np.array([v_min, delta_min_rad])
        ubu = np.array([v_max, delta_max_rad])
        for j in range(self._N):
            self._solver.constraints_set(j, "lbu", lbu)
            self._solver.constraints_set(j, "ubu", ubu)

    def reset(self) -> None:
        """Reset controller internal state."""
        self._prev_v = 0.0
        self._prev_delta = 0.0
        self._last_debug = {}

    @property
    def N(self) -> int:
        return self._N

    @property
    def debug(self) -> dict:
        return dict(self._last_debug)
