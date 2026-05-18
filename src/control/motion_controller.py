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
#   - Post: `MotorCommand.steering_deg ∈ [delta_min, delta_max]` y `speed_mps >= 0`
#     cuando `valid=True`. El `safety_gate` aguas abajo no necesita
#     re-clampar.

from __future__ import annotations

import math
import os
import sys
import time
from typing import Tuple

import numpy as np

from src.core.interfaces.controller import IMotionController
from src.core.types.behavior import BehaviorOutput
from src.core.types.control import MotorCommand
from src.core.types.pose import PoseEstimate
from src.utils.live_log import live_log

_SOLVER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_generated_code")
_JSON_FILE = os.path.join(_SOLVER_DIR, "acados_ocp_bfmc_bicycle.json")
try:
    from config import BEHAVIOR_MIN_SPEED_MPS as _FORWARD_RECOVERY_MIN_SPEED_MPS
except Exception:
    _FORWARD_RECOVERY_MIN_SPEED_MPS = 0.20
_FORWARD_RECOVERY_MIN_SAMPLE_FORWARD_M = 0.02
_FORWARD_RECOVERY_MAX_SAMPLE_LATERAL_M = 0.18
_FORWARD_RECOVERY_MAX_HEADING_ERROR_DEG = 45.0


def _solver_lib_path() -> str:
    ext = "dylib" if sys.platform == "darwin" else "so"
    return os.path.join(_SOLVER_DIR, f"libacados_ocp_solver_bfmc_bicycle.{ext}")

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


def _default_steering_bounds_deg(default_max_deg: float = 25.0) -> tuple[float, float]:
    try:
        import config as _cfg
        max_deg = float(getattr(_cfg, "ACADOS_MPC_DELTA_MAX_DEG", default_max_deg))
        min_deg = float(getattr(_cfg, "ACADOS_MPC_DELTA_MIN_DEG", -max_deg))
    except Exception:
        max_deg = float(default_max_deg)
        min_deg = -max_deg
    if min_deg > max_deg:
        min_deg, max_deg = max_deg, min_deg
    return float(min_deg), float(max_deg)


def _osm_pose_to_controller_frame(x: float, y: float, yaw: float) -> np.ndarray:
    """Convert the OSM/map frame to the controller's standard ENU/CCW frame.

    The route planner and dashboard use the Lanelet/OSM geometry as-is:
    `x` grows to the right, `y` grows downward on screen, and headings
    therefore increase clockwise. The bicycle model inside the controller
    assumes the standard robotics frame instead: `x` forward/right,
    `y` upward, yaw positive CCW, and `+delta = left`.

    Mirroring the `y` axis reconciles both worlds:
      controller_x   = osm_x
      controller_y   = -osm_y
      controller_yaw = -osm_yaw
    """

    return np.array(
        [float(x), -float(y), _wrap_angle(-float(yaw))],
        dtype=np.float64,
    )


def _osm_states_to_controller_frame(states: np.ndarray) -> np.ndarray:
    """Vectorized version of `_osm_pose_to_controller_frame()` for paths."""

    arr = np.asarray(states, dtype=np.float64).copy()
    if arr.ndim != 2 or arr.shape[1] < 3:
        return arr
    arr[:, 1] *= -1.0
    arr[:, 2] = np.array([_wrap_angle(-float(yaw)) for yaw in arr[:, 2]], dtype=np.float64)
    return arr


def _path_arc_lengths_xy(states: np.ndarray) -> np.ndarray:
    arr = np.asarray(states, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return np.zeros(0, dtype=np.float64)
    if arr.shape[0] == 1:
        return np.zeros(1, dtype=np.float64)
    seg_lens = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def _retime_state_refs_by_speed(
    state_refs: np.ndarray,
    speed_profile: np.ndarray,
    *,
    dt_s: float,
    min_step_m: float,
    max_preview_m: float | None,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    refs = np.asarray(state_refs, dtype=np.float64)
    speeds = np.asarray(speed_profile, dtype=np.float64).reshape(-1)
    meta: dict[str, float | bool] = {
        "retimed": False,
        "source_preview_m": 0.0,
        "effective_preview_m": 0.0,
        "v_ref_min_mps": 0.0,
        "v_ref_max_mps": 0.0,
        "v_ref_mean_mps": 0.0,
    }
    if refs.ndim != 2 or refs.shape[1] < 3 or refs.shape[0] < 2 or speeds.size == 0:
        return np.array(refs, copy=True), meta

    n = min(int(speeds.size), int(refs.shape[0] - 1))
    speeds = np.abs(speeds[:n])
    meta["v_ref_min_mps"] = float(np.min(speeds))
    meta["v_ref_max_mps"] = float(np.max(speeds))
    meta["v_ref_mean_mps"] = float(np.mean(speeds))

    source_arcs = _path_arc_lengths_xy(refs)
    if source_arcs.size != refs.shape[0] or float(source_arcs[-1]) <= 1e-9:
        return np.array(refs, copy=True), meta
    meta["source_preview_m"] = float(source_arcs[-1])

    min_step = max(0.0, float(min_step_m))
    dt = max(0.0, float(dt_s))
    step_arcs = np.maximum(speeds * dt, min_step)
    target_arcs = np.concatenate([[0.0], np.cumsum(step_arcs)])
    if max_preview_m is not None and math.isfinite(float(max_preview_m)) and float(max_preview_m) > 1e-6:
        target_arcs = np.minimum(target_arcs, float(max_preview_m))
    target_arcs = np.minimum(target_arcs, float(source_arcs[-1]))
    meta["effective_preview_m"] = float(target_arcs[-1])

    if abs(float(target_arcs[-1]) - float(source_arcs[-1])) <= 1e-6:
        return np.array(refs, copy=True), meta

    unique_mask = np.concatenate([[True], np.diff(source_arcs) > 1e-9])
    xp = source_arcs[unique_mask]
    src = refs[unique_mask]
    if xp.size < 2:
        return np.array(refs, copy=True), meta

    out = np.array(refs, copy=True)
    out[: n + 1, 0] = np.interp(target_arcs, xp, src[:, 0])
    out[: n + 1, 1] = np.interp(target_arcs, xp, src[:, 1])
    yaw_unwrapped = np.unwrap(src[:, 2])
    out[: n + 1, 2] = np.array(
        [_wrap_angle(float(yaw)) for yaw in np.interp(target_arcs, xp, yaw_unwrapped)],
        dtype=np.float64,
    )
    if refs.shape[0] > n + 1:
        out[n + 1 :] = out[n]
    meta["retimed"] = True
    return out, meta


def _first_forward_reference_hint(
    target_path: np.ndarray,
    *,
    pose_x: float,
    pose_y: float,
    pose_yaw: float,
) -> dict[str, float] | None:
    path = np.asarray(target_path, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] < 3:
        return None
    cos_yaw = math.cos(float(pose_yaw))
    sin_yaw = math.sin(float(pose_yaw))
    for idx in range(1, path.shape[0]):
        dx = float(path[idx, 0]) - float(pose_x)
        dy = float(path[idx, 1]) - float(pose_y)
        dist_m = math.hypot(dx, dy)
        if dist_m < float(_FORWARD_RECOVERY_MIN_SAMPLE_FORWARD_M):
            continue
        x_fwd_m = (cos_yaw * dx) + (sin_yaw * dy)
        y_left_m = (sin_yaw * dx) - (cos_yaw * dy)
        heading_error_deg = abs(math.degrees(_wrap_angle(float(path[idx, 2]) - float(pose_yaw))))
        return {
            "sample_idx": float(idx),
            "distance_m": float(dist_m),
            "x_fwd_m": float(x_fwd_m),
            "y_left_m": float(y_left_m),
            "heading_error_deg": float(heading_error_deg),
        }
    return None


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
        Cota superior de steering en grados.
    output_deadband_deg : float
        Si |δ| < deadband ⇒ se redondea a 0 (evita jitter del actuador).
    """

    def __init__(
        self,
        max_steering_deg: float = 25.0,
        min_steering_deg: float | None = None,
        output_deadband_deg: float = 0.5,
    ) -> None:
        default_min_deg, default_max_deg = _default_steering_bounds_deg(max_steering_deg)
        self.max_steering_deg = float(default_max_deg if max_steering_deg == 25.0 else max_steering_deg)
        self.min_steering_deg = float(default_min_deg if min_steering_deg is None else min_steering_deg)
        self.output_deadband_deg = float(output_deadband_deg)
        self._solver: AcadosOcpSolver | None = None
        self._N: int = 0
        self.ready = False
        self.load_error: str | None = None
        self._last_debug: dict = {}
        self._prev_v = 0.0
        self._prev_delta = 0.0

        if not _ACADOS_AVAILABLE:
            self.load_error = "acados_template import failed"
            return
        if not os.path.isfile(_JSON_FILE):
            self.load_error = f"solver json missing: {_JSON_FILE}"
            return

        try:
            self._solver = AcadosOcpSolver(None, json_file=_JSON_FILE, generate=False, build=False)
            # acados ≥ 0.5.x: .acados_ocp is None when loading from JSON
            self._N = (self._solver.N if hasattr(self._solver, "N")
                       else self._solver.acados_ocp.dims.N)
            self.ready = True
        except Exception as exc:
            self._solver = None
            self.load_error = f"{type(exc).__name__}: {exc}"

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
        delta_deg = max(self.min_steering_deg, min(self.max_steering_deg, delta_deg))
        if abs(delta_deg) < self.output_deadband_deg:
            delta_deg = 0.0

        self._last_debug = {
            "solver": "acados",
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
        self._active_profile_name = "custom"

    def set_weight_profile(self, profile_name: str) -> bool:
        """Plan B1: aplica un perfil de pesos pre-definido por nombre.

        Lee de ``config.ACADOS_MPC_PROFILES`` (constantes Python) primero,
        y como fallback de ``runtime.yaml`` vía ``RuntimeParams`` para que
        el operador pueda tunear en campo.

        Returns:
            True si el perfil se aplicó; False si no se encontró.
        """
        from src.core.runtime_params import params as _rt_params
        profile = None
        # 1) intenta config.ACADOS_MPC_PROFILES
        try:
            import config as _cfg_mod
            profiles_dict = getattr(_cfg_mod, "ACADOS_MPC_PROFILES", None)
            if isinstance(profiles_dict, dict):
                profile = profiles_dict.get(str(profile_name))
        except Exception:
            pass
        # 2) fallback a runtime.yaml acados_mpc_profiles.<name>
        if profile is None:
            from_yaml = _rt_params.get(f"acados_mpc_profiles.{profile_name}", None)
            if isinstance(from_yaml, dict):
                profile = from_yaml
        if not isinstance(profile, dict):
            return False
        self.update_weights(
            x_cost=float(profile.get("x", 2.0)),
            y_cost=float(profile.get("y", 2.0)),
            yaw_cost=float(profile.get("yaw", 0.5)),
            v_cost=float(profile.get("v", 1.0)),
            steer_cost=float(profile.get("steer", 0.0)),
            delta_v_cost=float(profile.get("dv", 1.5)),
            delta_steer_cost=float(profile.get("dsteer", 0.75)),
        )
        self._active_profile_name = str(profile_name)
        return True

    @property
    def active_profile_name(self) -> str:
        """Nombre del perfil de pesos activo (para telemetría)."""
        return getattr(self, "_active_profile_name", "default")

    def update_bounds(
        self,
        v_min: float = -0.5,
        v_max: float = 0.10,
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
# Fallback solver: pure-pursuit puro-Python (sin Acados, sin dependencias
# nativas). Se activa cuando `acados_template` no está instalado o el código
# C generado todavía no se cocinó (típico en máquinas de dev/sim).
#
# Por qué pure-pursuit y no, p. ej., Stanley o un MPC scipy:
#   - BFMC corre a 0.5–1.5 m/s sobre curvaturas suaves; el sesgo geométrico
#     de pure-pursuit es despreciable a esas velocidades.
#   - 0 dependencias externas (solo `numpy`, ya importado).
#   - ~50 líneas de código auditables.
#
# Geometría (modelo bicicleta, eje trasero como referencia):
#
#       ┌──── L_d ────┐
#       │  (lookahead) │
#       │              ●  goal
#       │           ╱
#       │         ╱  α  (ángulo entre heading del auto y el goal)
#       ●────────  ─ ─ ─ → heading
#      car
#
#   δ = atan( 2 · L_wb · sin(α) / L_d )
#
#   donde L_wb = wheelbase, L_d = norma del vector car→goal.
# ---------------------------------------------------------------------------
_FALLBACK_WHEELBASE_M = 0.260   # debe coincidir con localization/dead_reckoning.py


class PurePursuitSolver:
    """Solver geométrico de fallback con la misma interfaz que `AcadosMPC`.

    No resuelve ningún OCP — calcula δ con pure-pursuit puro y devuelve
    `v_ref` directo del speed_profile (sin optimización temporal). Pensado
    para sim/dev sin acados; suficiente para verificar el lazo de control
    y para corridas a baja velocidad.

    El `N` se expone como propiedad mutable porque, a diferencia de Acados
    (donde el horizonte es fijo en build-time), pure-pursuit no depende
    del horizonte: es una decisión local. Reportamos `N` en el valor que
    el `BehaviorPlanner` esté usando para que `MotionController` no
    rechace los arrays con `horizon_mismatch`.

    Parameters
    ----------
    horizon_n : int
        Horizonte que `BehaviorPlanner` produce (debe coincidir con
        `BEHAVIOR_HORIZON_N` del config). Solo se usa para satisfacer
        el chequeo de dimensión aguas arriba.
    wheelbase_m : float
        Entre-eje del vehículo. Default = 0.260 m (BFMC).
    min_lookahead_m : float
        Lookahead mínimo cuando la velocidad es ≈ 0 — evita pure-pursuit
        degenerado cerca del primer punto del path.
    lookahead_gain_s : float
        Factor de tiempo: `L_d = max(min_lookahead_m, gain * v)`. Más alto
        = trayectoria más suave, peor tracking en curvas cerradas.
    max_steering_deg : float
        Cota superior del actuador.
    output_deadband_deg : float
        Si `|δ| < deadband` ⇒ se devuelve 0 (evita jitter del actuador).
    """

    def __init__(
        self,
        horizon_n: int = 20,
        *,
        wheelbase_m: float = _FALLBACK_WHEELBASE_M,
        min_lookahead_m: float = 0.30,
        lookahead_gain_s: float = 0.6,
        max_steering_deg: float = 25.0,
        min_steering_deg: float | None = None,
        output_deadband_deg: float = 0.5,
    ) -> None:
        self._N = int(horizon_n)
        self._L_wb = float(wheelbase_m)
        self._min_lookahead = float(min_lookahead_m)
        self._lookahead_gain = float(lookahead_gain_s)
        default_min_deg, default_max_deg = _default_steering_bounds_deg(max_steering_deg)
        self.max_steering_deg = float(default_max_deg if max_steering_deg == 25.0 else max_steering_deg)
        self.min_steering_deg = float(default_min_deg if min_steering_deg is None else min_steering_deg)
        self.output_deadband_deg = float(output_deadband_deg)
        self.ready = True
        self._last_debug: dict = {}

    # ------------------------------------------------------------------
    def compute(
        self,
        x_current: np.ndarray,
        state_refs: np.ndarray,
        input_refs: np.ndarray,
        v_prev: float | None = None,
        delta_prev: float | None = None,
    ) -> Tuple[float, float] | None:
        x0 = np.asarray(x_current, dtype=np.float64).ravel()[:3]
        sr = np.asarray(state_refs, dtype=np.float64)
        ir = np.asarray(input_refs, dtype=np.float64)
        if sr.ndim != 2 or sr.shape[1] < 2 or sr.shape[0] < 2:
            return None
        if ir.ndim != 2 or ir.shape[1] < 1 or ir.shape[0] < 1:
            return None

        # 1. v_ref directo del primer stage del speed_profile. Pure-pursuit
        #    no optimiza velocidad — se confía en el planner.
        v_ref = float(ir[0, 0])

        # 2. Distancia objetivo: crece con la velocidad para suavizar
        #    el tracking a medida que el auto va más rápido.
        L_d = max(self._min_lookahead, self._lookahead_gain * abs(v_ref))

        # 3. Punto más cercano al auto sobre el path: el primer índice
        #    cuya distancia acumulada desde x0 supere `L_d`. Si ningún
        #    punto cumple, usamos el último (fin del path).
        cx, cy = float(x0[0]), float(x0[1])
        dx = sr[:, 0] - cx
        dy = sr[:, 1] - cy
        dists = np.hypot(dx, dy)
        # Empieza buscando desde el closest-point para evitar elegir un
        # waypoint detrás del auto cuando el path se curva sobre sí mismo.
        i_closest = int(np.argmin(dists))
        i_goal = i_closest
        for i in range(i_closest, len(dists)):
            if dists[i] >= L_d:
                i_goal = i
                break
        else:
            i_goal = len(dists) - 1

        gx = float(sr[i_goal, 0])
        gy = float(sr[i_goal, 1])
        # 4. α = ángulo entre heading del auto y el vector car→goal.
        #    Importante: lo medimos en el frame del auto para que sin θ
        #    sea positivo cuando el goal está a la izquierda.
        yaw = float(x0[2])
        # Vector car→goal en frame mundo:
        wx = gx - cx
        wy = gy - cy
        # Rotamos al frame del auto (yaw inverso):
        cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
        local_x = cos_y * wx - sin_y * wy
        local_y = sin_y * wx + cos_y * wy
        L_actual = math.hypot(local_x, local_y)
        if L_actual < 1e-3:
            # Goal sobre el auto — sin información de dirección. Mantén δ.
            return v_ref, 0.0
        alpha = math.atan2(local_y, local_x)

        # 5. Ley de pure-pursuit: δ = atan(2·L_wb·sin(α) / L_d).
        delta_rad = math.atan2(2.0 * self._L_wb * math.sin(alpha), L_actual)
        delta_deg = math.degrees(delta_rad)

        # 6. Clamp + deadband — mismas reglas que AcadosMPC.
        delta_deg = max(self.min_steering_deg, min(self.max_steering_deg, delta_deg))
        if abs(delta_deg) < self.output_deadband_deg:
            delta_deg = 0.0

        self._last_debug = {
            "solver": "pure_pursuit",
            "v_ref_mps": round(v_ref, 4),
            "delta_opt_deg": round(delta_deg, 3),
            "lookahead_m": round(L_actual, 3),
            "alpha_deg": round(math.degrees(alpha), 2),
            "i_closest": int(i_closest),
            "i_goal": int(i_goal),
            "x0": [round(float(c), 4) for c in x0],
            "goal": [round(gx, 4), round(gy, 4)],
        }
        return v_ref, delta_deg

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._last_debug = {}

    @property
    def N(self) -> int:
        return self._N

    @N.setter
    def N(self, value: int) -> None:
        # Permite que el caller adapte el horizonte al del planner sin
        # tener que reconstruir el solver. AcadosMPC no expone esto
        # porque su N está horneado en el .json generado.
        self._N = int(value)

    @property
    def debug(self) -> dict:
        return dict(self._last_debug)


class AcadosBackendUnavailableError(RuntimeError):
    """Señala que Acados era obligatorio pero no pudo cargarse."""


def _backend_name_from_solver(solver) -> str:
    class_name = solver.__class__.__name__.lstrip("_").lower()
    if isinstance(solver, PurePursuitSolver):
        return "pure_pursuit"
    if isinstance(solver, AcadosMPC):
        return "acados"
    return class_name


def _backend_context(*, backend: str, acados_ready: bool, acados_error: str | None = None) -> dict[str, object]:
    return {
        "backend": str(backend),
        "python": os.path.realpath(sys.executable),
        "launcher": str(os.environ.get("URT_LAUNCHER", "direct") or "direct"),
        "expected_backend": str(os.environ.get("URT_EXPECTED_MPC_BACKEND", "") or ""),
        "acados_ready": bool(acados_ready),
        "acados_error": str(acados_error) if acados_error else None,
        "json_file": _JSON_FILE,
        "solver_lib": _solver_lib_path(),
        "acados_source_dir": os.environ.get("ACADOS_SOURCE_DIR"),
    }


def _format_acados_required_error(acados: AcadosMPC) -> str:
    return (
        "Acados is required for MotionController but could not be loaded. "
        f"python={os.path.realpath(sys.executable)} "
        f"json={_JSON_FILE} "
        f"solver_lib={_solver_lib_path()} "
        f"acados_source_dir={os.environ.get('ACADOS_SOURCE_DIR') or '<unset>'} "
        f"launcher={os.environ.get('URT_LAUNCHER', 'direct')} "
        f"details={acados.load_error or 'unknown error'}. "
        "Start the stack with ./run.sh or set FORCE_PURE_PURSUIT=True explicitly."
    )


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
        Cota superior de steering. Pasada a AcadosMPC y verificada acá.
    output_deadband_deg : float
        Steering < deadband ⇒ 0. Útil para no exigir al actuador a
        baja amplitud.
    """

    def __init__(
        self,
        solver: "AcadosMPC | PurePursuitSolver | None" = None,
        *,
        max_steering_deg: float = 25.0,
        min_steering_deg: float | None = None,
        output_deadband_deg: float = 0.5,
        fallback_horizon_n: int = 20,
    ) -> None:
        self._backend_name = "unknown"
        self._backend_context: dict[str, object] = {}
        default_min_deg, default_max_deg = _default_steering_bounds_deg(max_steering_deg)
        resolved_max_steering_deg = (
            default_max_deg if max_steering_deg == 25.0 else float(max_steering_deg)
        )
        resolved_min_steering_deg = (
            default_min_deg if min_steering_deg is None else float(min_steering_deg)
        )
        if solver is not None:
            # Solver inyectado — el caller sabe lo que hace (tests).
            self._solver = solver
            self._backend_name = _backend_name_from_solver(solver)
            self._backend_context = _backend_context(
                backend=self._backend_name,
                acados_ready=bool(getattr(solver, "ready", False)),
            )
        else:
            # `FORCE_PURE_PURSUIT` (config) salta Acados y usa PP siempre. Útil
            # cuando el solver Acados pide reversa al overshoot un waypoint y
            # el coche queda atascado contra el `v_min_runtime` (ver comentario
            # más abajo). PP siempre genera comando forward al setpoint del
            # behavior_planner — más robusto cuando el coche se sale del carril
            # y el `at_pose` matchea una lanelet detrás. En real hardware se
            # deja en False para usar Acados (mejor tracking en pista).
            try:
                import config as _cfg_force
                _force_pp = bool(getattr(_cfg_force, "FORCE_PURE_PURSUIT", False))
                _acados_output_deadband_deg = float(
                    getattr(_cfg_force, "ACADOS_MPC_OUTPUT_DEADBAND_DEG", output_deadband_deg)
                )
            except Exception:
                _force_pp = False
                _acados_output_deadband_deg = float(output_deadband_deg)

            # Construcción por defecto: probamos AcadosMPC primero. Si no
            # logra cargar y NO se pidió explícitamente pure-pursuit,
            # abortamos el arranque. Queremos que usar un backend distinto
            # de Acados sea una decisión consciente, no un fallback silencioso.
            acados = None if _force_pp else AcadosMPC(
                max_steering_deg=resolved_max_steering_deg,
                min_steering_deg=resolved_min_steering_deg,
                output_deadband_deg=_acados_output_deadband_deg,
            )
            if acados is not None and acados.ready:
                self._solver = acados
                self._backend_name = "acados"
                # Plan B1.2: aplica perfil activo desde config en lugar de
                # leer las constantes sueltas. Si MPC_WEIGHT_PROFILE no está
                # definido, fallback al "default" (que iguala las constantes
                # legacy). Si set_weight_profile falla (config sin profiles),
                # fallback al patrón legacy de constantes sueltas.
                try:
                    import config as _cfg
                    profile_name = str(getattr(_cfg, "MPC_WEIGHT_PROFILE", "default"))
                    applied = acados.set_weight_profile(profile_name)
                    if not applied:
                        acados.update_weights(
                            x_cost=float(getattr(_cfg, "ACADOS_MPC_X_COST", 2.0)),
                            y_cost=float(getattr(_cfg, "ACADOS_MPC_Y_COST", 2.0)),
                            yaw_cost=float(getattr(_cfg, "ACADOS_MPC_YAW_COST", 0.5)),
                            v_cost=float(getattr(_cfg, "ACADOS_MPC_V_COST", 1.0)),
                            steer_cost=float(getattr(_cfg, "ACADOS_MPC_STEER_COST", 0.0)),
                            delta_v_cost=float(getattr(_cfg, "ACADOS_MPC_DELTA_V_COST", 1.5)),
                            delta_steer_cost=float(getattr(_cfg, "ACADOS_MPC_DELTA_STEER_COST", 0.75)),
                        )
                except Exception:
                    pass  # sin config (tests unitarios) — pesos del solver compilado
                # Limitar v_min en runtime: el .so compilado permite v=-0.5
                # m/s (reversa completa) pero en cartesian (sin Frenet) eso
                # causa que el solver pida reversa cuando el target_path
                # momentáneamente queda detrás del ego (ej. en intersecciones
                # donde at_pose oscila), llevando al auto a un loop de giros.
                # Floor demasiado alto (v_min=0) y el auto queda atascado
                # cuando el solver no puede converger forward — necesita
                # un poquito de reversa para reposicionarse.
                # Compromiso: v_min = -0.05 m/s (~5 cm/s, micro-reversa).
                # Suficiente para liberar atasco, demasiado bajo para hacer
                # loops. urt-ref no tiene este problema porque usa Frenet.
                try:
                    import math as _math
                    import config as _cfg
                    delta_max = _math.radians(float(getattr(_cfg, "ACADOS_MPC_DELTA_MAX_DEG", 25.0)))
                    delta_min = _math.radians(float(getattr(_cfg, "ACADOS_MPC_DELTA_MIN_DEG", -_math.degrees(delta_max))))
                    v_max = float(getattr(_cfg, "ACADOS_MPC_V_MAX", 0.10))
                    v_min_runtime = float(getattr(_cfg, "ACADOS_MPC_V_MIN_RUNTIME", -0.05))
                    acados.update_bounds(
                        v_min=v_min_runtime,
                        v_max=v_max,
                        delta_min_rad=delta_min,
                        delta_max_rad=delta_max,
                    )
                except Exception:
                    pass
                self._backend_context = _backend_context(
                    backend="acados",
                    acados_ready=True,
                    acados_error=acados.load_error,
                )
            else:
                if _force_pp:
                    print(
                        "\033[1;97m[ MotionController ] :\033[0m "
                        "\033[1;94mINFO\033[0m - FORCE_PURE_PURSUIT=True → "
                        "skipping Acados, using PurePursuitSolver."
                    )
                    try:
                        import config as _cfg
                        _min_lookahead = float(
                            getattr(_cfg, "BEHAVIOR_MIN_LOOKAHEAD_M", 0.30)
                        )
                        _lookahead_gain = float(
                            getattr(_cfg, "BEHAVIOR_LOOKAHEAD_GAIN_S", 1.1)
                        )
                    except Exception:
                        _min_lookahead = 0.30
                        _lookahead_gain = 1.1
                    self._solver = PurePursuitSolver(
                        horizon_n=fallback_horizon_n,
                        min_lookahead_m=_min_lookahead,
                        lookahead_gain_s=_lookahead_gain,
                        max_steering_deg=resolved_max_steering_deg,
                        min_steering_deg=resolved_min_steering_deg,
                        output_deadband_deg=output_deadband_deg,
                    )
                    self._backend_name = "pure_pursuit"
                    self._backend_context = _backend_context(
                        backend="pure_pursuit",
                        acados_ready=False,
                        acados_error=(acados.load_error if acados is not None else None),
                    )
                else:
                    print(
                        "\033[1;97m[ MotionController ] :\033[0m "
                        "\033[1;91mERROR\033[0m - AcadosMPC not ready "
                        "(missing Python bindings, runtime libs, or generated solver). "
                        "Acados is required unless FORCE_PURE_PURSUIT=True."
                    )
                    self._backend_name = "acados"
                    self._backend_context = _backend_context(
                        backend="acados",
                        acados_ready=False,
                        acados_error=(acados.load_error if acados is not None else None),
                    )
                    live_log("mpc", event="backend_selected", **self._backend_context)
                    raise AcadosBackendUnavailableError(
                        _format_acados_required_error(acados if acados is not None else AcadosMPC())
                    )
        live_log("mpc", event="backend_selected", **self._backend_context)
        self.max_steering_deg = float(resolved_max_steering_deg)
        self.min_steering_deg = float(resolved_min_steering_deg)

        # Rate limiters de steering y velocidad — independientes del solver.
        # Garantizan transiciones graduales aunque el solver entregue saltos.
        self._prev_steer_deg: float = 0.0
        self._prev_speed_mps: float = 0.0
        try:
            import config as _cfg
            self._max_steer_rate_deg_s: float = float(
                getattr(_cfg, "BEHAVIOR_MAX_STEER_RATE_DEG_S", 60.0)
            )
            self._max_speed_rate_mps2: float = float(
                getattr(_cfg, "BEHAVIOR_MAX_SPEED_RATE_MPS2", 0.15)
            )
            self._steer_dt_s: float = float(getattr(_cfg, "BEHAVIOR_DT_S", 0.05))
            self._min_moving_speed_mps: float = float(
                getattr(_cfg, "BEHAVIOR_MIN_SPEED_MPS", 0.20)
            )
            self._retime_acados_reference_by_speed: bool = bool(
                getattr(_cfg, "ACADOS_MPC_RETIME_REFERENCE_BY_SPEED", True)
            )
            self._acados_reference_min_step_m: float = float(
                getattr(_cfg, "ACADOS_MPC_REFERENCE_MIN_STEP_M", 0.0)
            )
            self._acados_reference_max_preview_m: float = float(
                getattr(_cfg, "ACADOS_MPC_REFERENCE_MAX_PREVIEW_M", 0.85)
            )
        except Exception:
            self._max_steer_rate_deg_s = 60.0
            self._max_speed_rate_mps2 = 0.15
            self._steer_dt_s = 0.05
            self._min_moving_speed_mps = 0.20
            self._retime_acados_reference_by_speed = True
            self._acados_reference_min_step_m = 0.0
            self._acados_reference_max_preview_m = 0.85

    # ------------------------------------------------------------------
    # IMotionController
    # ------------------------------------------------------------------
    def compute(
        self, behavior_output: BehaviorOutput, pose: PoseEstimate
    ) -> MotorCommand:
        # 1. Validez del plan: el planner ya nos dijo si pudo computar.
        import logging as _log
        import time as _t
        if not hasattr(self, "_last_log_t"):
            self._last_log_t: float = 0.0
        _now = _t.monotonic()
        _should_log = (_now - self._last_log_t) >= 2.0
        sp0_for_log = float(behavior_output.speed_profile[0]) if (
            hasattr(behavior_output.speed_profile, "__len__")
            and len(behavior_output.speed_profile) > 0
        ) else 0.0
        notes_reason = (
            str(behavior_output.notes.get("reason", ""))
            if isinstance(getattr(behavior_output, "notes", None), dict)
            else ""
        )
        # Loggear el primer punto del target_path para detectar el clásico
        # "el path está atrás del auto" — síntoma de pose/yaw mal alineado
        # con el mapa, que hace que el solver prefiera reversa (v_min=-0.5).
        tp_first = None
        tp_second = None
        try:
            if hasattr(behavior_output.target_path, "__len__") and len(behavior_output.target_path) > 0:
                tp_first = list(behavior_output.target_path[0])
            if hasattr(behavior_output.target_path, "__len__") and len(behavior_output.target_path) > 1:
                tp_second = list(behavior_output.target_path[1])
        except Exception:
            pass
        tp0_distance_m = None
        tp0_heading_error_deg = None
        if tp_first is not None and pose is not None:
            try:
                dx0 = float(tp_first[0]) - float(pose.fused_pose.x)
                dy0 = float(tp_first[1]) - float(pose.fused_pose.y)
                tp0_distance_m = math.hypot(dx0, dy0)
                tp0_heading_error_deg = math.degrees(
                    _wrap_angle(float(tp_first[2]) - float(pose.fused_pose.yaw))
                )
            except Exception:
                tp0_distance_m = None
                tp0_heading_error_deg = None
        live_log(
            "mpc", event="compute_in",
            valid=bool(behavior_output.valid),
            stop_req=bool(behavior_output.stop_required),
            scenario=behavior_output.scenario_name,
            backend=self._backend_name,
            sp0=sp0_for_log,
            notes_reason=notes_reason,
            pose_x=float(pose.fused_pose.x) if pose else None,
            pose_y=float(pose.fused_pose.y) if pose else None,
            pose_yaw_rad=float(pose.fused_pose.yaw) if pose else None,
            tp0=tp_first,
            tp1=tp_second,
            tp0_distance_m=tp0_distance_m,
            tp0_heading_error_deg=tp0_heading_error_deg,
            n_path=int(len(behavior_output.target_path)) if hasattr(behavior_output.target_path, "__len__") else 0,
        )
        if _should_log:
            self._last_log_t = _now
            _log.getLogger(__name__).info(
                "[ MPC-in ] valid=%s stop_req=%s scen=%s sp[0]=%.3f pose=(%.2f,%.2f,%.1f°)",
                behavior_output.valid,
                behavior_output.stop_required,
                behavior_output.scenario_name,
                sp0_for_log,
                pose.fused_pose.x if pose else 0,
                pose.fused_pose.y if pose else 0,
                math.degrees(pose.fused_pose.yaw) if pose else 0,
            )
        if not behavior_output.valid:
            reason = f"behavior_invalid/{notes_reason}" if notes_reason else "behavior_invalid"
            self._reset_command_memory()
            return self._invalid(reason, backend=self._backend_name)

        # 2. Stop request: el scenario o el overlay decidieron frenar.
        #    Devolvemos un MotorCommand explícito de stop — el dispatcher
        #    decide si eso es un fallback (safety) o un freno deseado.
        if behavior_output.stop_required:
            self._reset_command_memory()
            return MotorCommand(
                timestamp=time.time(),
                steering_deg=0.0,
                speed_mps=0.0,
                valid=True,
                source="motion_controller",
                reason="stop_required",
                debug={
                    "scenario": behavior_output.scenario_name,
                    "backend": self._backend_name,
                },
            )

        direct_cmd = self._direct_motor_command_from_behavior(behavior_output)
        if direct_cmd is not None:
            self._prev_speed_mps = float(direct_cmd.speed_mps)
            self._prev_steer_deg = float(direct_cmd.steering_deg)
            return direct_cmd

        # 3. Solver listo? Si no, no hay forma de calcular un δ — el
        #    safety_gate aguas abajo se hará cargo.
        if not self._solver.ready:
            self._reset_command_memory()
            return self._invalid("mpc_solver_not_ready", backend=self._backend_name)

        behavior_notes = (
            getattr(behavior_output, "notes", None)
            if isinstance(getattr(behavior_output, "notes", None), dict)
            else {}
        )

        # 4. Construir arrays de referencia. target_path es (N+1, 3) y
        #    speed_profile es (N,). El solver espera input_refs (N, 2)
        #    con [v_ref, delta_ref]; delta_ref siempre es 0 en BFMC
        #    (no tenemos perfil de steering planeado).
        state_refs = np.asarray(behavior_output.target_path, dtype=np.float64)
        speed_profile = np.asarray(behavior_output.speed_profile, dtype=np.float64)

        n = int(speed_profile.shape[0])
        if state_refs.shape != (n + 1, 3):
            self._reset_command_memory()
            return self._invalid("dimension_mismatch_target_path", backend=self._backend_name)
        if n != self._solver.N:
            self._reset_command_memory()
            return self._invalid("horizon_mismatch", backend=self._backend_name)

        requested_profile = _desired_weight_profile_name(behavior_notes)
        if requested_profile and requested_profile != self.active_weight_profile():
            self.set_weight_profile(requested_profile)

        # The planner publishes the path in the OSM/map frame (y-down, yaw CW+),
        # while the bicycle model in Acados/PurePursuit expects the standard
        # ENU/CCW convention with `+delta = left`. Mirror Y here so both the
        # solver state and the reference corridor live in the same handedness.
        x_current = _osm_pose_to_controller_frame(
            pose.fused_pose.x,
            pose.fused_pose.y,
            pose.fused_pose.yaw,
        )
        state_refs_for_solver = state_refs
        preview_meta: dict[str, float | bool] = {
            "retimed": False,
            "source_preview_m": float(_path_arc_lengths_xy(state_refs)[-1]) if state_refs.shape[0] > 0 else 0.0,
            "effective_preview_m": float(_path_arc_lengths_xy(state_refs)[-1]) if state_refs.shape[0] > 0 else 0.0,
            "v_ref_min_mps": float(np.min(speed_profile)) if speed_profile.size else 0.0,
            "v_ref_max_mps": float(np.max(speed_profile)) if speed_profile.size else 0.0,
            "v_ref_mean_mps": float(np.mean(speed_profile)) if speed_profile.size else 0.0,
        }
        if self._backend_name == "acados" and bool(self._retime_acados_reference_by_speed):
            state_refs_for_solver, preview_meta = _retime_state_refs_by_speed(
                state_refs,
                speed_profile,
                dt_s=float(getattr(behavior_output, "dt", self._steer_dt_s) or self._steer_dt_s),
                min_step_m=float(self._acados_reference_min_step_m),
                max_preview_m=float(self._acados_reference_max_preview_m),
            )
        live_log(
            "mpc",
            event="reference_preview",
            backend=self._backend_name,
            retimed=bool(preview_meta.get("retimed", False)),
            source_preview_m=float(preview_meta.get("source_preview_m", 0.0) or 0.0),
            effective_preview_m=float(preview_meta.get("effective_preview_m", 0.0) or 0.0),
            v_ref_min_mps=float(preview_meta.get("v_ref_min_mps", 0.0) or 0.0),
            v_ref_max_mps=float(preview_meta.get("v_ref_max_mps", 0.0) or 0.0),
            v_ref_mean_mps=float(preview_meta.get("v_ref_mean_mps", 0.0) or 0.0),
        )
        state_refs_solver = _osm_states_to_controller_frame(state_refs_for_solver)

        input_refs = np.zeros((n, 2), dtype=np.float64)
        input_refs[:, 0] = speed_profile  # v_ref
        # input_refs[:, 1] = 0  # delta_ref (steering ref siempre 0)

        # 5. Llamar al solver. Si devuelve None, el OCP no resolvió.
        result = self._solver.compute(
            x_current=x_current,
            state_refs=state_refs_solver,
            input_refs=input_refs,
            v_prev=float(self._prev_speed_mps),
            delta_prev=math.radians(float(self._prev_steer_deg)),
        )
        if result is None:
            self._reset_command_memory()
            return self._invalid("mpc_solver_failure", backend=self._backend_name)

        v_opt, delta_deg = result

        # 6. Sanity check de bordes mecánicos. AcadosMPC ya clampa, pero
        #    defensivo: cualquier NaN/inf por solver inestable se atrapa
        #    acá y se traduce a inválido (mejor parar que mover absurdo).
        if not (math.isfinite(v_opt) and math.isfinite(delta_deg)):
            self._reset_command_memory()
            return self._invalid("mpc_nonfinite_output", backend=self._backend_name)

        requested_speed_mps = max(0.0, float(speed_profile[0]))

        # Velocidad negativa NO se usa: aunque el solver acados tenga
        # v_min=-0.05 runtime (para evitar que se quede pegado al floor
        # cuando el target apunta atrás), el motion_controller clampa a 0
        # antes de mandar al motor. En cartesian, dejar pasar reversa real
        # generaba loops en intersecciones. El v_min<0 del solver sigue
        # ayudando: el solver puede usar un poquito de "espacio negativo"
        # internamente para converger en su QP, aunque al motor solo
        # mandemos v ≥ 0.
        speed_mps = min(max(0.0, float(v_opt)), requested_speed_mps)
        forward_recovery_reason = ""
        forward_recovery_hint: dict[str, float] | None = None
        notes = behavior_notes
        path_source = str((notes or {}).get("path_source") or "") if isinstance(notes, dict) else ""
        if (
            speed_mps <= 1e-6
            and float(v_opt) < 0.0
            and requested_speed_mps > 1e-6
            and requested_speed_mps >= float(_FORWARD_RECOVERY_MIN_SPEED_MPS)
            and path_source == "route_waypoints"
        ):
            forward_recovery_hint = _first_forward_reference_hint(
                state_refs,
                pose_x=float(pose.fused_pose.x),
                pose_y=float(pose.fused_pose.y),
                pose_yaw=float(pose.fused_pose.yaw),
            )
            if (
                forward_recovery_hint is not None
                and forward_recovery_hint["x_fwd_m"] >= float(_FORWARD_RECOVERY_MIN_SAMPLE_FORWARD_M)
                and abs(forward_recovery_hint["y_left_m"]) <= float(_FORWARD_RECOVERY_MAX_SAMPLE_LATERAL_M)
                and forward_recovery_hint["heading_error_deg"] <= float(_FORWARD_RECOVERY_MAX_HEADING_ERROR_DEG)
            ):
                speed_mps = min(
                    requested_speed_mps,
                    max(
                        float(_FORWARD_RECOVERY_MIN_SPEED_MPS),
                        0.5 * requested_speed_mps,
                    ),
                )
                if speed_mps < float(_FORWARD_RECOVERY_MIN_SPEED_MPS):
                    speed_mps = float(_FORWARD_RECOVERY_MIN_SPEED_MPS)
                forward_recovery_reason = "negative_solver_forward_route_recovery"
            else:
                forward_recovery_reason = "negative_solver_no_forward_hint"

        # El planner es la autoridad de velocidad. Limitamos solo la
        # aceleración hacia arriba; si planner/solver piden menos velocidad,
        # la bajada se aplica en el mismo tick para no quedar por encima
        # del target actual.
        max_speed_step = self._max_speed_rate_mps2 * self._steer_dt_s
        prev_speed_mps = float(self._prev_speed_mps)
        limited_by_accel = False
        if speed_mps > self._prev_speed_mps:
            limited_speed_mps = min(self._prev_speed_mps + max_speed_step, speed_mps)
            limited_by_accel = limited_speed_mps < (speed_mps - 1e-9)
            speed_mps = limited_speed_mps
        min_moving_speed_mps = float(
            _effective_min_moving_speed_mps(behavior_output, self._min_moving_speed_mps)
        )
        if (
            requested_speed_mps > 1e-6
            and speed_mps > 1e-6
            and requested_speed_mps >= min_moving_speed_mps
            and not limited_by_accel
            and (prev_speed_mps >= min_moving_speed_mps or max_speed_step >= requested_speed_mps)
            and speed_mps < min_moving_speed_mps
        ):
            speed_mps = min_moving_speed_mps
        self._prev_speed_mps = speed_mps

        # Reclamp por seguridad — si el solver entrega algo > max_steering.
        steering_deg = max(
            self.min_steering_deg,
            min(self.max_steering_deg, float(delta_deg)),
        )

        # Rate limiter de steering — independiente del solver.
        max_steer_rate_deg_s = _effective_steer_rate_limit_deg_s(
            behavior_notes,
            self._max_steer_rate_deg_s,
        )
        max_steer_step = max_steer_rate_deg_s * self._steer_dt_s
        steering_deg = max(
            self._prev_steer_deg - max_steer_step,
            min(self._prev_steer_deg + max_steer_step, steering_deg),
        )
        self._prev_steer_deg = steering_deg
        solver_debug = (
            dict(self._solver.debug)
            if hasattr(self._solver, "debug")
            else {}
        )

        if _should_log:
            _log.getLogger(__name__).info(
                "[ MPC-out ] v_opt=%.3f→%.3f delta=%.1f° (rate_limited)",
                v_opt,
                speed_mps,
                steering_deg,
            )

        live_log(
            "mpc", event="compute_out",
            valid=True, reason="",
            v_opt_raw=float(v_opt), speed_mps=float(speed_mps),
            delta_deg_raw=float(delta_deg), steering_deg=float(steering_deg),
            scenario=behavior_output.scenario_name,
            requested_speed_mps=float(requested_speed_mps),
            mpc_weight_profile=self.active_weight_profile(),
            steer_rate_limit_deg_s=float(max_steer_rate_deg_s),
            speed_recovery_reason=forward_recovery_reason,
            speed_recovery_hint=forward_recovery_hint,
            backend=self._backend_name,
            **solver_debug,
        )

        return MotorCommand(
            timestamp=time.time(),
            steering_deg=steering_deg,
            speed_mps=speed_mps,
            valid=True,
            source="motion_controller",
            reason="",
            debug=solver_debug,
        )

    def reset(self) -> None:
        self._solver.reset()
        self._reset_command_memory()

    def set_weight_profile(self, profile_name: str) -> bool:
        """Plan B1.2: delega al solver el cambio de perfil de pesos.

        El BehaviorPlanner llama a este método cuando entra a un nuevo
        scenario (ej. ``"parking"``) para que el MPC re-pondere las
        prioridades del costo. Si el solver no es AcadosMPC (ej.
        pure-pursuit), retorna False silenciosamente.
        """
        if hasattr(self._solver, "set_weight_profile"):
            try:
                return bool(self._solver.set_weight_profile(profile_name))
            except Exception:
                return False
        return False

    def active_weight_profile(self) -> str:
        """Nombre del perfil de pesos activo, para telemetría."""
        return str(getattr(self._solver, "active_profile_name", "default"))

    def _reset_command_memory(self) -> None:
        self._prev_steer_deg = 0.0
        self._prev_speed_mps = 0.0

    def _direct_motor_command_from_behavior(
        self,
        behavior_output: BehaviorOutput,
    ) -> MotorCommand | None:
        notes = getattr(behavior_output, "notes", None)
        if not isinstance(notes, dict):
            return None
        raw = notes.get("direct_motor_command")
        if not isinstance(raw, dict):
            return None

        scenario = str(behavior_output.scenario_name or "").lower()
        try:
            steering_deg = float(raw.get("steering_deg", 0.0))
            speed_mps = float(raw.get("speed_mps", 0.0))
        except (TypeError, ValueError):
            return self._invalid("bad_direct_motor_command", backend=self._backend_name)

        allow_reverse = bool(raw.get("allow_reverse", False)) and scenario == "parking"
        if not allow_reverse:
            speed_mps = max(0.0, speed_mps)

        steering_deg = max(
            self.min_steering_deg,
            min(self.max_steering_deg, steering_deg),
        )
        return MotorCommand(
            timestamp=time.time(),
            steering_deg=steering_deg,
            speed_mps=speed_mps,
            valid=True,
            source="motion_controller",
            reason=str(raw.get("reason", "direct_behavior_command") or "direct_behavior_command"),
            debug={
                "scenario": behavior_output.scenario_name,
                "backend": self._backend_name,
                "direct_motor_command": dict(raw),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _invalid(reason: str, *, backend: str = "unknown") -> MotorCommand:
        """Comando inválido con razón. El dispatcher invocará el safety_gate."""
        live_log(
            "mpc", event="compute_out",
            valid=False, reason=reason,
            speed_mps=0.0, steering_deg=0.0,
            backend=backend,
        )
        return MotorCommand(
            timestamp=time.time(),
            steering_deg=0.0,
            speed_mps=0.0,
            valid=False,
            source="motion_controller",
            reason=reason,
            debug={"backend": backend},
        )

    @property
    def ready(self) -> bool:
        """Expone readiness del solver subyacente (útil para diagnostics)."""
        return self._solver.ready

    @property
    def horizon(self) -> int:
        return self._solver.N


def _effective_min_moving_speed_mps(
    behavior_output: BehaviorOutput,
    default_min_mps: float,
) -> float:
    value = getattr(behavior_output, "min_moving_speed_mps", None)
    if value is None and isinstance(getattr(behavior_output, "notes", None), dict):
        value = behavior_output.notes.get("min_moving_speed_mps")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return max(0.0, float(default_min_mps))


def _desired_weight_profile_name(notes: dict | None) -> str:
    if isinstance(notes, dict):
        raw = notes.get("mpc_weight_profile")
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    try:
        import config as _cfg
        return str(getattr(_cfg, "MPC_WEIGHT_PROFILE", "default") or "default")
    except Exception:
        return "default"


def _effective_steer_rate_limit_deg_s(notes: dict | None, default_deg_s: float) -> float:
    if isinstance(notes, dict) and notes.get("steer_rate_limit_deg_s") is not None:
        try:
            value = float(notes.get("steer_rate_limit_deg_s"))
            if math.isfinite(value) and value > 0.0:
                return value
        except (TypeError, ValueError):
            pass
    return float(default_deg_s)
