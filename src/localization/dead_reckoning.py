"""
Dead reckoning position estimator.

Position is integrated using Runge-Kutta 4th order (RK4), which accounts for
the heading change *within* each timestep.  This is critical at high speed or
with large dt (frame drops): Euler integration uses only the start-of-step
heading, accumulating O(dt²) position error per step, while RK4 reduces that
to O(dt⁴).

The ODE is integrated directly in the active OSM/map frame.  The active
localization path supplies fused body yaw and the lagged steering angle:
    beta   = atan((l_r / L) * tan(steer))
    dx/dt  = v * cos(yaw_body - beta)
    dy/dt  = v * sin(yaw_body - beta)

The legacy bicycle-model API is still available for older callers:
    dx/dt   = v * cos(yaw)
    dy/dt   = v * sin(yaw)
    dyaw/dt = -(v / L) * tan(steer)

Frame/convention details:
    - map `x` grows to the right
    - map `y` grows downward (same orientation as the dashboard)
    - map `yaw` is 0 along +x and grows clockwise
    - `steer > 0` still means wheels turned to the LEFT (ISO/controller sign)

With a clockwise-positive yaw, a physical left turn must DECREASE yaw.
That is why the kinematic yaw-rate term carries a leading minus sign here.

Project control / Nucleo feedback uses the opposite raw wire steering
convention (`steer > 0` means right turn), so threadTracking converts it
before calling this module.

In the active pose estimator, BNO055 yaw is treated as a correction signal; in
steering-heavy periods body yaw is propagated by the bicycle model instead of
letting magnetometer jumps rotate the whole odometry estimate.
"""

import math
import threading

# Wheelbase default (m) — overridden by threadTracking via steer_rad argument
_WHEELBASE_M = 0.260
_REAR_AXLE_TO_CG_M = 0.103


def _slip_beta(steer_rad: float, wheelbase_m: float, rear_axle_to_cg_m: float) -> float:
    wheelbase = max(1e-6, float(wheelbase_m))
    return math.atan((float(rear_axle_to_cg_m) / wheelbase) * math.tan(float(steer_rad)))


def curve_speed_compensated(
    speed_mps: float,
    steer_rad: float,
    *,
    wheelbase_m: float = _WHEELBASE_M,
    rear_axle_to_cg_m: float = _REAR_AXLE_TO_CG_M,
) -> float:
    """Convert encoder wheel-path speed to the CG/body speed used by beta DR."""
    beta = _slip_beta(steer_rad, wheelbase_m, rear_axle_to_cg_m)
    cos_beta = math.cos(beta)
    if abs(cos_beta) < 1e-6:
        return float(speed_mps)
    return float(speed_mps) * (math.cos(float(steer_rad) - beta) / cos_beta)


class DeadReckoning:
    """Kinematic dead-reckoning estimator with RK4 position integration.

    Thread-safe: all public methods acquire ``_lock``.
    """

    def __init__(self, x0=0.0, y0=0.0, yaw0=0.0):
        self._lock = threading.Lock()
        self._x = float(x0)
        self._y = float(y0)
        self._yaw = float(yaw0)
        self._last_update = None

    # ------------------------------------------------------------------
    def update(self, speed_mps: float, yaw_rad: float, dt: float,
               steer_rad: float = 0.0, wheelbase_m: float = _WHEELBASE_M,
               rear_axle_to_cg_m: float = _REAR_AXLE_TO_CG_M) -> None:
        """Legacy bicycle-model integration using RK4.

        RK4 evaluates the heading at the start, midpoint, and end of the step
        so position is accurate even when the car is turning at high speed or
        when dt is large due to a frame drop.

        Args:
            speed_mps:   Forward speed in m/s (positive = forward).
            yaw_rad:     Absolute heading from the IMU in radians (start of step).
            dt:          Time step in seconds.
            steer_rad:   Current steering angle in radians.
                         Positive = left turn, negative = right turn.
            wheelbase_m: Distance between axles (m).
        """
        if dt <= 0.0 or dt > 1.0:
            return

        beta = _slip_beta(steer_rad, wheelbase_m, rear_axle_to_cg_m)

        # OSM/map frame convention:
        #   yaw grows clockwise, so a physical LEFT turn must decrease yaw.
        yaw_rate = -(speed_mps * math.cos(beta) / wheelbase_m) * math.tan(steer_rad)

        # RK4 stages — v, steer are constant within the step, so yaw_rate is
        # constant and the yaw trajectory is linear: yaw(t) = yaw_rad + yaw_rate*t.
        # k values are *rates* (m/s); final displacement = dt/6*(k1+2k2+2k3+k4).
        yaw_mid = yaw_rad + (dt * 0.5) * yaw_rate   # heading at t + dt/2
        yaw_end = yaw_rad + dt * yaw_rate             # heading at t + dt

        self._apply_rk4_yaw_trajectory(
            float(speed_mps),
            float(yaw_rad) - beta,
            yaw_mid - beta,
            yaw_end - beta,
            float(dt),
            stored_yaw_rad=yaw_end,
        )

    def update_from_yaw(
        self,
        speed_mps: float,
        yaw_start_rad: float,
        yaw_end_rad: float,
        dt: float,
        *,
        steer_rad: float = 0.0,
        wheelbase_m: float = _WHEELBASE_M,
        rear_axle_to_cg_m: float = _REAR_AXLE_TO_CG_M,
    ) -> None:
        """Integrate one time step using measured yaw at the start and end.

        This is the active pose-estimator path: position follows the IMU yaw
        trajectory. A beta slip angle can be supplied so the CG translation
        follows the front-wheel bicycle geometry used by urt-ref/Acados.
        """
        if dt <= 0.0 or dt > 1.0:
            return

        yaw_start = float(yaw_start_rad)
        yaw_delta = float(yaw_end_rad) - yaw_start
        while yaw_delta > math.pi:
            yaw_delta -= 2.0 * math.pi
        while yaw_delta < -math.pi:
            yaw_delta += 2.0 * math.pi

        yaw_mid = yaw_start + 0.5 * yaw_delta
        yaw_end = yaw_start + yaw_delta
        beta = _slip_beta(steer_rad, wheelbase_m, rear_axle_to_cg_m)
        self._apply_rk4_yaw_trajectory(
            float(speed_mps),
            yaw_start - beta,
            yaw_mid - beta,
            yaw_end - beta,
            float(dt),
            stored_yaw_rad=yaw_end,
        )

    def _apply_rk4_yaw_trajectory(
        self,
        speed_mps: float,
        yaw_start_rad: float,
        yaw_mid_rad: float,
        yaw_end_rad: float,
        dt: float,
        stored_yaw_rad: float | None = None,
    ) -> None:
        # k2 == k3 because yaw_rate is constant (straight-line yaw trajectory)
        k1_x = speed_mps * math.cos(yaw_start_rad)
        k1_y = speed_mps * math.sin(yaw_start_rad)

        k24_x = speed_mps * math.cos(yaw_mid_rad)   # k2 and k3 are identical
        k24_y = speed_mps * math.sin(yaw_mid_rad)

        k4_x = speed_mps * math.cos(yaw_end_rad)
        k4_y = speed_mps * math.sin(yaw_end_rad)

        dx = (dt / 6.0) * (k1_x + 4.0 * k24_x + k4_x)
        dy = (dt / 6.0) * (k1_y + 4.0 * k24_y + k4_y)

        with self._lock:
            self._x += dx
            self._y += dy
            self._yaw = yaw_end_rad if stored_yaw_rad is None else float(stored_yaw_rad)

    def reset(self, x: float, y: float, yaw: float) -> None:
        """Re-anchor position to a known ground truth (e.g. a stop-line node)."""
        with self._lock:
            self._x = float(x)
            self._y = float(y)
            self._yaw = float(yaw)

    def correct_yaw(self, yaw_correction_rad: float) -> None:
        """Nudge the estimated yaw by a camera-derived correction.

        Called by threadTracking when the camera (two-line mode) provides a
        reliable world-frame yaw estimate.  Only the internal yaw store is
        touched; x/y are NOT modified here.

        Args:
            yaw_correction_rad: Signed correction to add to the current yaw (rad).
        """
        with self._lock:
            self._yaw += float(yaw_correction_rad)

    def correct_lateral(self, lateral_error_m: float, path_psi: float) -> None:
        """Nudge the estimated position by the measured lane lateral error.

        Called when lane detection gives a reliable crosstrack measurement
        (e.g. two lines visible) to reduce dead-reckoning drift.

        Args:
            lateral_error_m: Signed lateral error from lane centre (m).
                             Positive = car is left of lane centre (same sign
                             convention as LateralMPC and trackGraph).
            path_psi:        Path tangent angle at the nearest waypoint (rad).
                             Used to convert the lateral offset into (dx, dy).
        """
        perp_psi = path_psi + math.pi / 2.0
        dx = lateral_error_m * math.cos(perp_psi)
        dy = lateral_error_m * math.sin(perp_psi)
        with self._lock:
            self._x -= dx
            self._y -= dy

    # ------------------------------------------------------------------
    @property
    def x(self) -> float:
        with self._lock:
            return self._x

    @property
    def y(self) -> float:
        with self._lock:
            return self._y

    @property
    def yaw(self) -> float:
        with self._lock:
            return self._yaw

    def get_state(self):
        """Return (x, y, yaw) atomically."""
        with self._lock:
            return self._x, self._y, self._yaw
