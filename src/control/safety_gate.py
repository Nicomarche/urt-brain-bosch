# src/control/safety_gate.py
#
# `SafetyGate` — watchdog del path crítico de control. Decide cuándo el
# `MotorCommand` que produjo el `MotionController` es seguro para mandar
# al firmware, y cuándo hay que sustituirlo por un fallback (parar).
#
# El gate vive entre el `MotionController` y el `motor_command_dispatcher`:
#
#   BehaviorOutput ─► MotionController ─► MotorCommand ─► SafetyGate ─► dispatcher
#                                                          │
#                                                          ├─ pasa? → manda al firmware
#                                                          └─ falla? → fallback() → STOP
#
# Reglas (todas iguales prioridad — cualquier una activa ⇒ fallback):
#
#   1. **Behavior stale**: si el último `BehaviorOutput` recibido tiene más
#      de `behavior_stale_timeout_s` (default 0.5 s), el planner se colgó
#      o el thread se cayó. No hay plan ⇒ stop.
#
#   2. **Pose stale**: si la última `PoseEstimate` tiene más de
#      `pose_stale_timeout_s` (default 0.3 s), el localizador se cayó. Sin
#      pose el MPC no sabe dónde estamos ⇒ stop.
#
#   3. **Command invalid**: si `MotorCommand.valid == False` (el controller
#      no pudo computar un comando — solver fallido, dimensiones mal,
#      behavior inválido), el dispatcher no debe ejecutar nada ⇒ stop.
#
# Por qué un módulo dedicado:
#   - Aislar la decisión "es seguro mandar esto?" del MotionController y
#     del dispatcher. Cada uno hace UNA cosa: el controller produce, el
#     gate verifica, el dispatcher despacha.
#   - Tests independientes: el gate se puede ejercitar inyectando
#     timestamps y MotorCommand sintéticos sin tocar Acados ni el serial.
#   - Cambios de política (umbrales, reglas nuevas) viven en un solo
#     archivo, no dispersos.

from __future__ import annotations

import time
from dataclasses import dataclass

from src.core.types.control import MotorCommand
from src.core.types.lidar import LidarObstacle

try:
    import config as _cfg
except Exception:  # pragma: no cover - config unavailable only in isolated imports
    _cfg = None


@dataclass(frozen=True)
class SafetyGateConfig:
    """Parámetros del watchdog. Inmutable — set en construcción."""

    # Si el último BehaviorOutput tiene más segundos que este, fallback.
    # 500 ms cubre el caso "thread del planner cayó" sin disparar en
    # jitter normal del loop (planner corre a 20 Hz, dt=50 ms).
    behavior_stale_timeout_s: float = 0.5

    # Pose más vieja que esto ⇒ fallback. EKF7 corre nominalmente a >50 Hz
    # (encoder + IMU), así que 300 ms ya indica falla del bus serial.
    pose_stale_timeout_s: float = 0.3

    # El dispatcher corre más rápido que el MotionController; si el último
    # MotorCommand quedó viejo, no debe seguir re-despachándose.
    motor_command_stale_timeout_s: float = float(getattr(_cfg, "MOTOR_COMMAND_STALE_TIMEOUT_S", 0.30))

    lidar_required: bool = bool(getattr(_cfg, "LIDAR_REQUIRED", False))
    lidar_stale_timeout_s: float = float(getattr(_cfg, "LIDAR_STALE_TIMEOUT_S", 0.5))
    lidar_emergency_obstacle_enabled: bool = bool(getattr(_cfg, "LIDAR_EMERGENCY_OBSTACLE_ENABLED", False))
    lidar_emergency_distance_m: float = float(getattr(_cfg, "LIDAR_EMERGENCY_DISTANCE_M", 0.28))
    lidar_emergency_half_width_m: float = float(getattr(_cfg, "LIDAR_EMERGENCY_HALF_WIDTH_M", 0.14))


class SafetyGate:
    """Verifica al MotorCommand antes de despacharlo al firmware.

    Stateless excepto por el último resultado (para diagnóstico). La
    decisión depende solo de los timestamps y flags pasados a `evaluate()`.

    Patrón de uso típico desde el dispatcher:

        gate = SafetyGate()
        ...
        cmd = gate.evaluate(
            motor_command=last_motor_cmd,
            behavior_timestamp=last_behavior_ts,
            pose_timestamp=last_pose_ts,
            now=time.time(),
        )
        # cmd.valid siempre True acá — si las verificaciones fallaron,
        # `cmd` es el `fallback()` (STOP); si pasaron, es el MotorCommand
        # original. El dispatcher manda `cmd` al firmware sin pensar más.
    """

    def __init__(self, config: SafetyGateConfig | None = None) -> None:
        self._config = config or SafetyGateConfig()
        self._last_decision_reason: str = ""

    # ------------------------------------------------------------------
    def evaluate(
        self,
        *,
        motor_command: MotorCommand | None,
        behavior_timestamp: float | None,
        pose_timestamp: float | None,
        motor_command_timestamp: float | None = None,
        lidar_timestamp: float | None = None,
        lidar_obstacles: tuple[LidarObstacle, ...] | list[LidarObstacle] = (),
        now: float | None = None,
    ) -> MotorCommand:
        """Devuelve el comando seguro para despachar.

        Si todas las verificaciones pasan, devuelve el `motor_command`
        original. Si falla alguna, devuelve `fallback()` con la razón
        en `MotorCommand.reason`.

        Parameters
        ----------
        motor_command : MotorCommand | None
            Último comando producido por el `MotionController`. None si
            todavía no se computó ninguno (arranque del sistema).
        behavior_timestamp : float | None
            Timestamp UNIX (segundos) del último `BehaviorOutput`. None si
            nunca se recibió uno (arranque).
        pose_timestamp : float | None
            Timestamp UNIX del último `PoseEstimate`. None si nunca se
            recibió uno (localizador inactivo).
        now : float | None
            Para tests; default `time.time()`.
        """
        t_now = time.time() if now is None else float(now)

        # Regla 3 primero porque es la más barata (sin aritmética de tiempo).
        if motor_command is None:
            return self._fallback(t_now, "no_motor_command")
        if not motor_command.valid:
            reason = motor_command.reason or "motor_command_invalid"
            return self._fallback(t_now, reason)
        cmd_timestamp = (
            float(motor_command_timestamp)
            if motor_command_timestamp is not None
            else float(getattr(motor_command, "timestamp", 0.0) or 0.0)
        )
        if cmd_timestamp <= 0.0:
            return self._fallback(t_now, "no_motor_command_timestamp")
        cmd_age = t_now - cmd_timestamp
        if cmd_age > self._config.motor_command_stale_timeout_s:
            return self._fallback(t_now, f"motor_command_stale_{cmd_age:.2f}s")

        # Regla 1 — behavior stale.
        if behavior_timestamp is None:
            return self._fallback(t_now, "no_behavior_output")
        behavior_age = t_now - float(behavior_timestamp)
        if behavior_age > self._config.behavior_stale_timeout_s:
            return self._fallback(
                t_now, f"behavior_stale_{behavior_age:.2f}s"
            )

        # Regla 2 — pose stale.
        if pose_timestamp is None:
            return self._fallback(t_now, "no_pose_estimate")
        pose_age = t_now - float(pose_timestamp)
        if pose_age > self._config.pose_stale_timeout_s:
            return self._fallback(t_now, f"pose_stale_{pose_age:.2f}s")

        # Regla 4 — LiDAR requerido/stale.
        if self._config.lidar_required:
            if lidar_timestamp is None:
                return self._fallback(t_now, "no_lidar_scan")
            lidar_age = t_now - float(lidar_timestamp)
            if lidar_age > self._config.lidar_stale_timeout_s:
                return self._fallback(t_now, f"lidar_stale_{lidar_age:.2f}s")

        # Regla 5 — emergencia frontal por obstáculo LiDAR.
        if self._config.lidar_emergency_obstacle_enabled:
            emergency = self._lidar_emergency_obstacle(lidar_obstacles)
            if emergency is not None:
                return self._fallback(
                    t_now,
                    f"lidar_emergency_{emergency.distance_min_m:.2f}m",
                )

        # Todo OK — pasa el comando original sin modificar.
        self._last_decision_reason = "passthrough"
        return motor_command

    # ------------------------------------------------------------------
    def fallback(self, now: float | None = None) -> MotorCommand:
        """Comando de stop puro. Útil para arranque y casos manuales.

        Independiente de `evaluate()`; se puede llamar directo si el
        dispatcher detecta una condición que no encaja en las reglas
        estándar (ej. usuario presionó botón de emergencia).
        """
        return self._fallback(time.time() if now is None else float(now), "manual")

    # ------------------------------------------------------------------
    @property
    def last_reason(self) -> str:
        """Última razón de decisión, para diagnóstico/dashboard."""
        return self._last_decision_reason

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _fallback(self, t_now: float, reason: str) -> MotorCommand:
        self._last_decision_reason = reason
        return MotorCommand(
            timestamp=t_now,
            steering_deg=0.0,
            speed_mps=0.0,
            valid=True,           # válido para despachar (es un STOP intencional)
            source="safety_gate",
            reason=reason,
        )

    def _lidar_emergency_obstacle(
        self,
        obstacles: tuple[LidarObstacle, ...] | list[LidarObstacle],
    ) -> LidarObstacle | None:
        if not obstacles:
            return None
        max_x = float(self._config.lidar_emergency_distance_m)
        half_width = float(self._config.lidar_emergency_half_width_m)
        best: LidarObstacle | None = None
        for obs in obstacles:
            if not isinstance(obs, LidarObstacle):
                continue
            x = float(obs.x_m)
            y = float(obs.y_m)
            if x < 0.0 or x > max_x or abs(y) > half_width:
                continue
            if best is None or float(obs.distance_min_m) < float(best.distance_min_m):
                best = obs
        return best
