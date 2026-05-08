#!/usr/bin/env python3
"""Generate the Acados OCP solver for the BFMC bicycle model with slip angle.

Run once (on any machine with acados + casadi installed) to produce C code:

    cd /path/to/urt-brain-bosch
    python -m src.control._acados_solver_gen          # defaults: N=30, T=0.05
    python -m src.control._acados_solver_gen --N 40 --T 0.04   # custom

The solver C code is written to  src/hardware/mpc/c_generated_code/
and compiled into a shared library on first import by AcadosOcpSolver.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def create_bicycle_model(
    wheelbase: float = 0.258,
    l_r: float = 0.103,
) -> "AcadosModel":
    """Kinematic bicycle model with slip-angle (beta variant).

    States  x = [x, y, psi]          (3)
    Controls u = [v, delta]           (2)
    Parameters p = [v_prev, delta_prev]  (2)  — for rate-of-change cost

    Dynamics:
        beta    = atan((l_r / L) * tan(delta))
        x_dot   = v * cos(psi + beta)
        y_dot   = v * sin(psi + beta)
        psi_dot = (v * cos(beta) / L) * tan(delta)
    """
    import casadi as ca
    from acados_template import AcadosModel

    model = AcadosModel()
    model.name = "bfmc_bicycle"

    # --- symbolic variables ---
    x_pos = ca.SX.sym("x_pos")
    y_pos = ca.SX.sym("y_pos")
    psi = ca.SX.sym("psi")
    state = ca.vertcat(x_pos, y_pos, psi)

    v = ca.SX.sym("v")
    delta = ca.SX.sym("delta")
    control = ca.vertcat(v, delta)

    v_prev = ca.SX.sym("v_prev")
    delta_prev = ca.SX.sym("delta_prev")
    params = ca.vertcat(v_prev, delta_prev)

    xdot = ca.SX.sym("xdot", 3)

    # --- dynamics ---
    L = float(wheelbase)
    beta = ca.atan2(float(l_r) * ca.tan(delta), L)
    x_dot = v * ca.cos(psi + beta)
    y_dot = v * ca.sin(psi + beta)
    psi_dot = (v * ca.cos(beta) / L) * ca.tan(delta)
    f_expl = ca.vertcat(x_dot, y_dot, psi_dot)

    model.x = state
    model.u = control
    model.p = params
    model.xdot = xdot
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl

    # --- cost outputs ---
    # Stages 0..N-1:  y = [x, y, psi, v, delta, Δv, Δδ]   (7)
    delta_v = v - v_prev
    delta_steer = delta - delta_prev
    model.cost_y_expr = ca.vertcat(
        x_pos, y_pos, psi, v, delta, delta_v, delta_steer,
    )
    # Terminal:       y_e = [x, y, psi]                     (3)
    model.cost_y_expr_e = ca.vertcat(x_pos, y_pos, psi)

    return model


# ---------------------------------------------------------------------------
# OCP
# ---------------------------------------------------------------------------

def generate_solver(
    N: int = 40,
    T: float = 0.05,
    wheelbase: float = 0.258,
    l_r: float = 0.103,
    v_min: float = -0.5,
    v_max: float = 0.10,
    delta_min: float = -0.436,
    delta_max: float = 0.436,
    x_cost: float = 2.0,
    y_cost: float = 2.0,
    yaw_cost: float = 0.5,
    v_cost: float = 1.0,
    steer_cost: float = 0.0,
    delta_v_cost: float = 1.5,
    delta_steer_cost: float = 0.75,
    output_dir: str | None = None,
) -> None:
    from acados_template import AcadosOcp, AcadosOcpSolver

    if output_dir is None:
        output_dir = os.path.join(_SCRIPT_DIR, "c_generated_code")
    os.makedirs(output_dir, exist_ok=True)

    model = create_bicycle_model(wheelbase=wheelbase, l_r=l_r)

    ocp = AcadosOcp()
    ocp.model = model

    # Dimensions
    nx = 3
    nu = 2
    ny = 7      # [x, y, psi, v, delta, Δv, Δδ]
    ny_e = 3    # [x, y, psi]
    n_params = 2  # [v_prev, delta_prev]

    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = float(N * T)

    # --- Cost (nonlinear least-squares) ---
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    # Weight matrices
    Q = np.diag([x_cost, y_cost, yaw_cost])
    R = np.diag([v_cost, steer_cost])
    S = np.diag([delta_v_cost, delta_steer_cost])
    W = np.zeros((ny, ny))
    W[:3, :3] = Q
    W[3:5, 3:5] = R
    W[5:7, 5:7] = S
    W_e = Q.copy()

    ocp.cost.W = W
    ocp.cost.W_e = W_e
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # --- Control input constraints ---
    ocp.constraints.lbu = np.array([v_min, delta_min])
    ocp.constraints.ubu = np.array([v_max, delta_max])
    ocp.constraints.idxbu = np.array([0, 1])

    # --- Initial state constraint (overwritten at runtime) ---
    ocp.constraints.x0 = np.zeros(nx)

    # --- Parameters ---
    ocp.parameter_values = np.zeros(n_params)

    # --- Solver options ---
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.print_level = 0

    ocp.code_gen_opts.code_export_directory = output_dir

    json_path = os.path.join(output_dir, "acados_ocp_bfmc_bicycle.json")

    # Generate solver (compiles C code + creates shared library)
    _solver = AcadosOcpSolver(ocp, json_file=json_path)
    print(f"Solver generated successfully.")
    print(f"  Model     : {model.name}")
    print(f"  Horizon   : N={N}, T={T:.4f}s  →  {N*T:.2f}s total")
    print(f"  Wheelbase : {wheelbase}m  (l_r={l_r}m)")
    print(f"  v bounds  : [{v_min}, {v_max}] m/s")
    print(f"  δ bounds  : [{delta_min:.4f}, {delta_max:.4f}] rad  "
          f"([{np.degrees(delta_min):.1f}°, {np.degrees(delta_max):.1f}°])")
    print(f"  Costs     : Q=diag({x_cost},{y_cost},{yaw_cost}), "
          f"R=diag({v_cost},{steer_cost}), S=diag({delta_v_cost},{delta_steer_cost})")
    print(f"  Output    : {output_dir}")
    print(f"  JSON      : {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Acados MPC solver for BFMC bicycle model.",
    )
    parser.add_argument("--N", type=int, default=40, help="Prediction horizon steps (default 40)")
    parser.add_argument("--T", type=float, default=0.05, help="Time step in seconds (default 0.05)")
    parser.add_argument("--wheelbase", type=float, default=0.258, help="Wheelbase in metres (default 0.258)")
    parser.add_argument("--l_r", type=float, default=0.103, help="Rear axle to CG in metres (default 0.103)")
    parser.add_argument("--v_max", type=float, default=0.10, help="Max velocity m/s (default 0.10)")
    parser.add_argument("--delta_max_deg", type=float, default=25.0, help="Max steering degrees (default 25)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for C code")
    args = parser.parse_args()

    generate_solver(
        N=args.N,
        T=args.T,
        wheelbase=args.wheelbase,
        l_r=args.l_r,
        v_min=-args.v_max,
        v_max=args.v_max,
        delta_min=-np.radians(args.delta_max_deg),
        delta_max=np.radians(args.delta_max_deg),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
