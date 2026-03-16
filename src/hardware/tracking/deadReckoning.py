"""
Dead reckoning position estimator using a kinematic bicycle model.

Same formulation as the reference repo (mpc_acados2_clean.py / Utility.cpp):
    x   += v * cos(yaw) * dt
    y   += v * sin(yaw) * dt
    yaw  = IMU yaw (absolute, not integrated)

The IMU provides absolute yaw (degrees, from the NUCLEO), so heading never
drifts.  Only position drifts over time — that is corrected by re-anchoring
to a known node whenever the car passes close to a stop-line node.
"""

import math
import threading


class DeadReckoning:
    """Kinematic dead-reckoning estimator.

    Integrates forward velocity over time using the IMU yaw as heading.
    Thread-safe: all public methods acquire ``_lock``.
    """

    def __init__(self, x0=0.0, y0=0.0, yaw0=0.0):
        self._lock = threading.Lock()
        self._x = float(x0)
        self._y = float(y0)
        self._yaw = float(yaw0)
        self._last_update = None

    # ------------------------------------------------------------------
    def update(self, speed_mps: float, yaw_rad: float, dt: float) -> None:
        """Integrate one time step.

        Args:
            speed_mps: Forward speed in m/s (positive = forward).
            yaw_rad:   Absolute heading from the IMU in radians.
            dt:        Time step in seconds.
        """
        if dt <= 0.0 or dt > 1.0:
            return
        dx = speed_mps * math.cos(yaw_rad) * dt
        dy = speed_mps * math.sin(yaw_rad) * dt
        with self._lock:
            self._x += dx
            self._y += dy
            self._yaw = yaw_rad

    def reset(self, x: float, y: float, yaw: float) -> None:
        """Re-anchor position to a known ground truth (e.g. a stop-line node)."""
        with self._lock:
            self._x = float(x)
            self._y = float(y)
            self._yaw = float(yaw)

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
