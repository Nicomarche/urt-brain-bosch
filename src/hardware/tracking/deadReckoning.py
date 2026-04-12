"""
Dead reckoning position estimator using a kinematic bicycle model.

Position is integrated using Runge-Kutta 4th order (RK4), which accounts for
the heading change *within* each timestep.  This is critical at high speed or
with large dt (frame drops): Euler integration uses only the start-of-step
heading, accumulating O(dt²) position error per step, while RK4 reduces that
to O(dt⁴).

The ODE being integrated:
    dx/dt   = v * cos(yaw)
    dy/dt   = v * sin(yaw)
    dyaw/dt = -(v / L) * tan(steer)   [sign: right turn → yaw decreases]

yaw is corrected to IMU absolute heading whenever a fresh IMU sample arrives
(handled in threadTracking), so heading never accumulates drift.  Only
position drifts over time — corrected by re-anchoring to a known node whenever
the car passes close to a stop-line node.
"""

import math
import threading

# Wheelbase default (m) — overridden by threadTracking via steer_rad argument
_WHEELBASE_M = 0.260


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
               steer_rad: float = 0.0, wheelbase_m: float = _WHEELBASE_M) -> None:
        """Integrate one time step using RK4.

        RK4 evaluates the heading at the start, midpoint, and end of the step
        so position is accurate even when the car is turning at high speed or
        when dt is large due to a frame drop.

        Args:
            speed_mps:   Forward speed in m/s (positive = forward).
            yaw_rad:     Absolute heading from the IMU in radians (start of step).
            dt:          Time step in seconds.
            steer_rad:   Current steering angle in radians.
                         Positive = right turn (same sign as servo command).
            wheelbase_m: Distance between axles (m).
        """
        if dt <= 0.0 or dt > 1.0:
            return

        # Yaw rate from bicycle model (right turn → yaw decreases in math convention)
        yaw_rate = -(speed_mps / wheelbase_m) * math.tan(steer_rad)

        # RK4 stages — v, steer are constant within the step, so yaw_rate is
        # constant and the yaw trajectory is linear: yaw(t) = yaw_rad + yaw_rate*t.
        # k values are *rates* (m/s); final displacement = dt/6*(k1+2k2+2k3+k4).
        yaw_mid = yaw_rad + (dt * 0.5) * yaw_rate   # heading at t + dt/2
        yaw_end = yaw_rad + dt * yaw_rate             # heading at t + dt

        # k2 == k3 because yaw_rate is constant (straight-line yaw trajectory)
        k1_x = speed_mps * math.cos(yaw_rad)
        k1_y = speed_mps * math.sin(yaw_rad)

        k24_x = speed_mps * math.cos(yaw_mid)   # k2 and k3 are identical
        k24_y = speed_mps * math.sin(yaw_mid)

        k4_x = speed_mps * math.cos(yaw_end)
        k4_y = speed_mps * math.sin(yaw_end)

        dx = (dt / 6.0) * (k1_x + 4.0 * k24_x + k4_x)
        dy = (dt / 6.0) * (k1_y + 4.0 * k24_y + k4_y)

        with self._lock:
            self._x += dx
            self._y += dy
            self._yaw = yaw_end

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
