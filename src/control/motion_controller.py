# src/control/motion_controller.py
#
# `MotionController` — la ÚNICA fuente de verdad para steering en runtime
# de competencia. Toma un `BehaviorOutput` (target_path + speed_profile)
# producido por el `BehaviorPlanner` y la pose actual del vehículo, y
# devuelve un `MotorCommand` (steering + speed) listo para el firmware.
#
# Capas (SRP estricto):
#
#   ┌─────────────────────────────────────────────────────────────────┐
#   │ BehaviorOutput, PoseEstimate ─► MotionController.compute()      │
#   │                                  │                              │
#   │                                  │  (traduce a arrays MPC)      │
#   │                                  ▼                              │
#   │                         AcadosMPC.compute(state_refs,           │
#   │                                           input_refs, ...)      │
#   │                                  │                              │
#   │                                  │  (resuelve QP no-lineal)     │
#   │                                  ▼                              │
#   │                         (v_opt_mps, delta_deg)                  │
#   │                                  │                              │
#   │                                  ▼                              │
#   │                         MotorCommand                            │
#   └─────────────────────────────────────────────────────────────────┘
#
# Por qué dos clases:
#   - `AcadosMPC` envuelve el solver C generado por acados_template. Su
#     responsabilidad es UNA: dar (v, δ) dado un estado y referencias en
#     forma de array. NO sabe nada de `BehaviorOutput` ni de safety. Esto
#     lo hace fácil de reemplazar por un MPC scipy en Jetson Nano si
#     Acados no compila.
#   - `MotionController` traduce contrato (Behavior↔Motor) y aplica las
#     reglas de borde: behavior inválido → MotorCommand inválido, MPC sin
#     solución → MotorCommand inválido. Implementa `IMotionController`.
#
# Pre/post:
#   - Pre `compute()`: `behavior_output.target_path.shape == (N+1, 3)` y
#     `speed_profile.shape == (N,)` con `N` igual al horizonte del solver
#     ya generado. Si no calzan, devuelve MotorCommand inválido (no crash).
#   - Post: `MotorCommand.steering_deg ∈ [-25, +25]` y `speed_mps >= 0`
#     cuando `valid=True`. El `safety_gate` aguas abajo no necesita
#     re-clampar.

from __future__ import annotations

import math
import os
import time
from typing import Tuple

import numpy as np

from src.core.interfaces.controller import IMotionController
from src.core.types.behavior import BehaviorOutput
from src.core.types.control import MotorCommand
from src.core.types.pose import PoseEstimate

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


# ---------------------------------------------------------------------------
# Low-level Acados solver wrapper. Sin dependencia de tipos del dominio.
# ---------------------------------------------------------------------------
class AcadosMPC:
    """Wrapper del solver Acados generado. Bajo nivel — solo arrays.

    El solver vive en `c_generated_code/acados_ocp_bfmc_bicycle.json`. Si
    el código no se generó (o `acados_template` no está instalado), `ready`
    queda en `False` y `compute()` devuelve `None` — el llamador debe
    chequear antes de usar el resultado.

    Esta clase NO conoce `BehaviorOutput` ni `MotorCommand` — opera sobre
    arrays. Eso permite intercambiarla por un solver scipy con la misma
    interfaz `compute(x, state_refs, input_refs) -> (v, δ)`.

    Parameters
    ----------
    max_steering_deg : float
        Cota absoluta de steering en grados.
    output_deadband_deg : float
        Si |δ| < deadband ⇒ se redondea a 0 (evita jitter del actuador).
    """

    def __init__(
        self,
        max_steering_deg: float = 25.0,
        output_deadband_deg: float = 0.5,
    ) -> None:
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
    def compute(
        self,
        x_current: np.ndarray,
        state_refs: np.ndarray,
        input_refs: np.ndarray,
        v_prev: float | None = None,
        delta_prev: float | None = None,
    ) -> Tuple[float, float] | None:
        """Resuelve el OCP y devuelve `(v_optimal_mps, delta_optimal_deg)`.

        Devuelve `None` si el solver no está listo o las dimensiones no
        calzan con el horizonte interno.
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

        # ---- Yaw refs continuos respecto del yaw actual (sin saltos 2π) --
        sr = sr.copy()
        ref_yaw = sr[:, 2].copy()
        ref_yaw[0] = x0[2] + _wrap_angle(ref_yaw[0] - x0[2])
        for i in range(1, len(ref_yaw)):
            ref_yaw[i] = ref_yaw[i - 1] + _wrap_angle(ref_yaw[i] - ref_yaw[i - 1])
        sr[:, 2] = ref_yaw

        # ---- Estado inicial fijado como restricción dura ----------------
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        # ---- Referencias por stage ---------------------------------------
        for j in range(N):
            yref = np.zeros(7, dtype=np.float64)
            yref[0:3] = sr[j]
            yref[3] = ir[j, 0]      # v_ref
            yref[4] = ir[j, 1]      # delta_ref
            self._solver.set(j, "yref", yref)
            self._solver.set(j, "p", p_val)

        # ---- Stage terminal --------------------------------------------
        yref_e = sr[N, :3].copy()
        self._solver.set(N, "yref", yref_e)

        # ---- Solve ----
        status = self._solver.solve()

        u_opt = self._solver.get(0, "u")
        v_opt = float(u_opt[0])
        delta_rad = float(u_opt[1])

        self._prev_v = v_opt
        self._prev_delta = delta_rad

        # Convertir a grados, clampar y aplicar deadband
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
        """Actualiza los pesos del coste sin regenerar el solver."""
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
        """Actualiza los bounds de input en runtime."""
        if not self.ready or self._solver is None:
            return
        lbu = np.array([v_min, delta_min_rad])
        ubu = np.array([v_max, delta_max_rad])
        for j in range(self._N):
            self._solver.constraints_set(j, "lbu", lbu)
            self._solver.constraints_set(j, "ubu", ubu)

    def reset(self) -> None:
        """Resetea el estado interno (warmstart)."""
        self._prev_v = 0.0
        self._prev_delta = 0.0
        self._last_debug = {}

    @property
    def N(self) -> int:
        return self._N

    @property
    def debug(self) -> dict:
        return dict(self._last_debug)


# ---------------------------------------------------------------------------
# Capa de contrato. ÚNICO punto que produce MotorCommand.
# ---------------------------------------------------------------------------
class MotionController(IMotionController):
    """Implementación de `IMotionController` apoyada en `AcadosMPC`.

    Responsabilidades (estrictas):
      1. Traducir `BehaviorOutput` → arrays que el solver entiende.
      2. Llamar al solver y leer su salida.
      3. Empaquetar el resultado en `MotorCommand` con flags coherentes.
      4. Emitir `MotorCommand(valid=False, ...)` cuando el plan es
         inválido o el solver falla — NUNCA inventar comandos.

    NO responsabilidades (intencionales):
      - Watchdog de staleness — ese rol vive en `safety_gate.py`. Acá se
        confía en que la pose y el behavior_output son recientes; la capa
        de arriba (motor_command_dispatcher) chequea timestamps.
      - Decisión de speed/stop — ya viene cocinada en `speed_profile`. Si
        el escenario quiere parar, emite `stop_required=True` (que acá se
        traduce a MotorCommand inválido para que safety_gate intervenga).
      - Cap de velocidad — `velocity_overlay` ya lo aplicó. Acá solo se
        clampa por seguridad mecánica si el solver entrega algo absurdo.

    Parameters
    ----------
    solver : AcadosMPC | None
        Si None, se construye uno por defecto. Inyectable para tests.
    max_steering_deg : float
        Cota absoluta de steering. Pasada a AcadosMPC y verificada acá.
    output_deadband_deg : float
        Steering < deadband ⇒ 0. Útil para no exigir al actuador a
        baja amplitud.
    """

    def __init__(
        self,
        solver: AcadosMPC | None = None,
        *,
        max_steering_deg: float = 25.0,
        output_deadband_deg: float = 0.5,
    ) -> None:
        self._solver = solver if solver is not None else AcadosMPC(
            max_steering_deg=max_steering_deg,
            output_deadband_deg=output_deadband_deg,
        )
        self.max_steering_deg = float(max_steering_deg)

    # ------------------------------------------------------------------
    # IMotionController
    # ------------------------------------------------------------------
    def compute(
        self, behavior_output: BehaviorOutput, pose: PoseEstimate
    ) -> MotorCommand:
        # 1. Validez del plan: el planner ya nos dijo si pudo computar.
        if not behavior_output.valid:
            return self._invalid("behavior_invalid")

        # 2. Stop request: el scenario o el overlay decidieron frenar.
        #    Devolvemos un MotorCommand explícito de stop — el dispatcher
        #    decide si eso es un fallback (safety) o un freno deseado.
        if behavior_output.stop_required:
            return MotorCommand(
                timestamp=time.time(),
                steering_deg=0.0,
                speed_mps=0.0,
                valid=True,
                source="motion_controller",
                reason="stop_required",
                debug={"scenario": behavior_output.scenario_name},
            )

        # 3. Solver listo? Si no, no hay forma de calcular un δ — el
        #    safety_gate aguas abajo se hará cargo.
        if not self._solver.ready:
            return self._invalid("mpc_solver_not_ready")

        # 4. Construir arrays de referencia. target_path es (N+1, 3) y
        #    speed_profile es (N,). El solver espera input_refs (N, 2)
        #    con [v_ref, delta_ref]; delta_ref siempre es 0 en BFMC
        #    (no tenemos perfil de steering planeado).
        state_refs = np.asarray(behavior_output.target_path, dtype=np.float64)
        speed_profile = np.asarray(behavior_output.speed_profile, dtype=np.float64)

        n = int(speed_profile.shape[0])
        if state_refs.shape != (n + 1, 3):
            return self._invalid("dimension_mismatch_target_path")
        if n != self._solver.N:
            return self._invalid("horizon_mismatch")

        input_refs = np.zeros((n, 2), dtype=np.float64)
        input_refs[:, 0] = speed_profile  # v_ref
        # input_refs[:, 1] = 0  # delta_ref (steering ref siempre 0)

        x_current = np.array(
            [pose.fused_pose.x, pose.fused_pose.y, pose.fused_pose.yaw],
            dtype=np.float64,
        )

        # 5. Llamar al solver. Si devuelve None, el OCP no resolvió.
        result = self._solver.compute(
            x_current=x_current,
            state_refs=state_refs,
            input_refs=input_refs,
        )
        if result is None:
            return self._invalid("mpc_solver_failure")

        v_opt, delta_deg = result

        # 6. Sanity check de bordes mecánicos. AcadosMPC ya clampa, pero
        #    defensivo: cualquier NaN/inf por solver inestable se atrapa
        #    acá y se traduce a inválido (mejor parar que mover absurdo).
        if not (math.isfinite(v_opt) and math.isfinite(delta_deg)):
            return self._invalid("mpc_nonfinite_output")

        # Velocidad negativa no se usa en BFMC (no hay reverse).
        speed_mps = max(0.0, float(v_opt))

        # Reclamp por seguridad — si el solver entrega algo > max_steering.
        steering_deg = max(
            -self.max_steering_deg,
            min(self.max_steering_deg, float(delta_deg)),
        )

        return MotorCommand(
            timestamp=time.time(),
            steering_deg=steering_deg,
            speed_mps=speed_mps,
            valid=True,
            source="motion_controller",
            reason="",
            debug=self._solver.debug,
        )

    def reset(self) -> None:
        self._solver.reset()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _invalid(reason: str) -> MotorCommand:
        """Comando inválido con razón. El dispatcher invocará el safety_gate."""
        return MotorCommand(
            timestamp=time.time(),
            steering_deg=0.0,
            speed_mps=0.0,
            valid=False,
            source="motion_controller",
            reason=reason,
        )

    @property
    def ready(self) -> bool:
        """Expone readiness del solver subyacente (útil para diagnostics)."""
        return self._solver.ready

    @property
    def horizon(self) -> int:
        return self._solver.N
