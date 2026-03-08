import time

import config
from src.utils.messages.allMessages import SpeedMotor, SteerMotor


class SignActions:
    """Shared sign action executor used by local and legacy sign pipelines."""

    BASE_SPEED      = getattr(config, "SIGN_BASE_SPEED",      5)
    LOW_SPEED       = getattr(config, "SIGN_LOW_SPEED",       3)
    SPEED_20        = getattr(config, "SIGN_SPEED_20",        3)
    SPEED_30        = getattr(config, "SIGN_SPEED_30",        5)
    HIGHWAY_SPEED   = getattr(config, "SIGN_HIGHWAY_SPEED",   7)
    STOP_DURATION      = getattr(config, "SIGN_STOP_DURATION",      3.0)
    CROSSWALK_DURATION = getattr(config, "SIGN_CROSSWALK_DURATION", 3.0)
    DEFAULT_CURVE_STRAIGHT_FRAMES = 3
    DEFAULT_CURVE_STRAIGHT_STEER_DEG = 5.0
    DEFAULT_PENDING_TIMEOUT = 8.0

    # Giro izquierda 90° post-stop.
    # Usa ángulo máximo de dirección (-25°) para un giro visible.
    # Radio efectivo ≈ 55.8 cm (wheelbase=26cm, steer=25°).
    # Arco 90° ≈ 87.6 cm → calibrar STOP_TURN_DURATION en el hardware real.
    STOP_LEFT_TURN_ENABLED = getattr(config, "SIGN_STOP_LEFT_TURN_ENABLED", True)
    STOP_TURN_STEER_DEG    = getattr(config, "SIGN_STOP_TURN_STEER_DEG",   -25.0)
    STOP_TURN_SPEED        = getattr(config, "SIGN_STOP_TURN_SPEED",        5)
    STOP_TURN_DURATION     = getattr(config, "SIGN_STOP_TURN_DURATION",     5.0)

    SIGN_ALIASES = {
        # Common canonical variants
        "highway_entry": "highway_entrance",
        "highway_entrance": "highway_entrance",
        "highway_exit": "highway_exit",
        # Short aliases observed in some datasets/logs
        "hw_entry": "highway_entrance",
        "hw_entrance": "highway_entrance",
        "hw_exit": "highway_exit",
    }

    ACTIONABLE_SIGNS = {
        "stop", "no_entry", "crosswalk", "red_light", "yellow_light",
        "green_light", "speed_20", "speed_30",
        "highway_entrance", "highway_exit",
        # "parking" is intentionally excluded: the parking state machine in
        # threadLineFollowing handles it via _poll_sign_detected(). Routing it
        # through SignActions would stop the car indefinitely.
    }

    ACTION_GROUP = {
        "stop": "stop",
        "no_entry": "stop",
        "red_light": "stop",
        "crosswalk": "crosswalk",
        "yellow_light": "traffic_light",
        "green_light": "traffic_light",
        "speed_20": "speed_limit",
        "speed_30": "speed_limit",
        "parking": "parking",
        # Keep entry/exit separate so a recent highway_entry does not block highway_exit.
        "highway_entrance": "highway_entrance",
        "highway_exit": "highway_exit",
    }

    def __init__(self, queuesList, sign_action_event=None, action_cooldown=15.0,
                 highway_mode_event=None, steer_override_event=None):
        self.queuesList = queuesList
        self.sign_action_event = sign_action_event
        self.highway_mode_event = highway_mode_event
        self.steer_override_event = steer_override_event
        self.action_cooldown = action_cooldown
        self.last_sign = None
        self.last_action_time = {}
        self.current_speed = self.BASE_SPEED
        self.is_stopped = False
        self.pending_sign = None
        self.pending_sign_since = 0.0
        self.pending_curve_state = "STRAIGHT"
        self.pending_steering_deg = 0.0
        self._pending_straight_frames = 0
        self.curve_release_straight_frames = self.DEFAULT_CURVE_STRAIGHT_FRAMES
        self.curve_release_steering_deg = self.DEFAULT_CURVE_STRAIGHT_STEER_DEG
        self.pending_sign_timeout = self.DEFAULT_PENDING_TIMEOUT

    @classmethod
    def normalize_sign_name(cls, sign_name):
        normalized = str(sign_name or "").strip().lower().replace("-", "_").replace(" ", "_")
        return cls.SIGN_ALIASES.get(normalized, normalized)

    @classmethod
    def is_actionable_sign(cls, sign_name):
        return cls.normalize_sign_name(sign_name) in cls.ACTIONABLE_SIGNS

    def _execute_canonical_sign(self, sign_name):
        if sign_name in ("stop", "red_light", "no_entry"):
            self._execute_stop()
        elif sign_name == "crosswalk":
            self._execute_crosswalk()
        elif sign_name == "yellow_light":
            self._execute_slow_down()
        elif sign_name == "green_light":
            self._execute_go()
        elif sign_name == "speed_20":
            self._execute_speed_limit(self.SPEED_20)
        elif sign_name == "speed_30":
            self._execute_speed_limit(self.SPEED_30)
        elif sign_name == "parking":
            self._execute_parking()
        elif sign_name == "highway_entrance":
            self._execute_highway_entrance()
        elif sign_name == "highway_exit":
            self._execute_highway_exit()
        else:
            return False
        return True

    @staticmethod
    def _is_curve_state(curve_state):
        return str(curve_state or "").upper() in {"ENTERING", "IN_CURVE", "EXITING"}

    def _requires_curve_exit(self, sign_name):
        return sign_name in {"highway_entrance"}

    def _should_release_pending(self, curve_state, steering_deg):
        curve_state_norm = str(curve_state or "").upper()
        steering_value = abs(float(steering_deg or 0.0))

        if curve_state_norm == "STRAIGHT" and steering_value <= self.curve_release_steering_deg:
            self._pending_straight_frames += 1
        else:
            self._pending_straight_frames = 0

        if self._pending_straight_frames >= max(1, int(self.curve_release_straight_frames)):
            return True

        if (time.time() - self.pending_sign_since) >= float(self.pending_sign_timeout):
            # Timeout fallback: if we're not deep in curve anymore, release.
            return curve_state_norm in {"STRAIGHT", "EXITING"}

        return False

    def tick(self, curve_state=None, steering_deg=0.0):
        """Try executing a previously deferred sign once the car is straight enough."""
        if self.pending_sign is None:
            self._pending_straight_frames = 0
            return False

        self.pending_curve_state = str(curve_state or "")
        self.pending_steering_deg = float(steering_deg or 0.0)

        if not self._should_release_pending(curve_state, steering_deg):
            return False

        sign_name = self.pending_sign
        self.pending_sign = None
        self.pending_sign_since = 0.0
        self._pending_straight_frames = 0

        now = time.time()
        action_group = self.ACTION_GROUP.get(sign_name, sign_name)
        last_time = self.last_action_time.get(action_group, 0.0)
        elapsed = now - last_time
        if elapsed < self.action_cooldown:
            remaining = self.action_cooldown - elapsed
            print(
                f"\033[1;97m[ SignActions ] :\033[0m \033[1;93mCOOLDOWN\033[0m - "
                f"pending {sign_name} ignored ({remaining:.0f}s left)"
            )
            return False

        if self._execute_canonical_sign(sign_name):
            self.last_sign = sign_name
            self.last_action_time[action_group] = now
            print(
                f"\033[1;97m[ SignActions ] :\033[0m \033[1;92mRESUME\033[0m - "
                f"Deferred {sign_name} executed after curve straightening"
            )
            return True
        return False

    def execute(self, sign_name, curve_state=None, steering_deg=0.0):
        """Execute the action mapped to a detected sign."""
        sign_name = self.normalize_sign_name(sign_name)
        if not sign_name:
            return False

        if sign_name not in self.ACTIONABLE_SIGNS:
            print(
                f"\033[1;97m[ SignActions ] :\033[0m \033[1;93mINFO\033[0m - "
                f"{sign_name} detected (no action)"
            )
            return False

        if self.pending_sign == "highway_entrance" and sign_name in {
            "highway_exit", "stop", "red_light", "no_entry", "parking"
        }:
            self.pending_sign = None
            self.pending_sign_since = 0.0
            self._pending_straight_frames = 0
            print(
                f"\033[1;97m[ SignActions ] :\033[0m \033[1;93mCANCEL\033[0m - "
                f"Deferred highway_entrance dropped due to {sign_name}"
            )

        if self._requires_curve_exit(sign_name) and self._is_curve_state(curve_state):
            self.pending_sign = sign_name
            self.pending_sign_since = time.time()
            self.pending_curve_state = str(curve_state or "")
            self.pending_steering_deg = float(steering_deg or 0.0)
            self._pending_straight_frames = 0
            print(
                f"\033[1;97m[ SignActions ] :\033[0m \033[1;96mQUEUED\033[0m - "
                f"{sign_name} saved (curve_state={self.pending_curve_state}, steer={self.pending_steering_deg:.1f}°)"
            )
            return True

        now = time.time()
        action_group = self.ACTION_GROUP.get(sign_name, sign_name)
        last_time = self.last_action_time.get(action_group, 0.0)
        elapsed = now - last_time
        if elapsed < self.action_cooldown:
            remaining = self.action_cooldown - elapsed
            print(
                f"\033[1;97m[ SignActions ] :\033[0m \033[1;93mCOOLDOWN\033[0m - "
                f"{sign_name} ignored ({remaining:.0f}s left)"
            )
            return False

        if not self._execute_canonical_sign(sign_name):
            return False

        self.last_sign = sign_name
        self.last_action_time[action_group] = now
        return True

    def _send_speed(self, speed):
        speed_value = str(int(round(speed * 10)))
        self.queuesList["General"].put({
            "Owner": SpeedMotor.Owner.value,
            "msgID": SpeedMotor.msgID.value,
            "msgType": SpeedMotor.msgType.value,
            "msgValue": speed_value,
        })

    def _send_steer(self, angle_deg):
        steer_value = str(int(round(angle_deg * 10)))
        self.queuesList["General"].put({
            "Owner": SteerMotor.Owner.value,
            "msgID": SteerMotor.msgID.value,
            "msgType": SteerMotor.msgType.value,
            "msgValue": steer_value,
        })

    def _execute_left_turn_90(self):
        """Giro izquierda 90° hardcodeado con radio 1,02 m (R=1,02 m, δ≈-14,3°).

        Asume que steer_override_event ya está activo (lo setea _execute_stop antes
        de llamar a esta función). Las ruedas ya están centradas (steer=0) desde
        el período de stop, así que aquí:
          1. Se aplica el ángulo de giro y se espera que el servo llegue.
          2. Se arranca el movimiento hacia adelante.
          3. Al terminar, steer vuelve a 0 y se devuelve control al line-following.

        Arco 90° = (π/2) × 1,02 m ≈ 1,602 m → calibrar STOP_TURN_DURATION.
        """
        # Aplicar ángulo de giro con el auto aún detenido y esperar que el servo llegue.
        self._send_steer(self.STOP_TURN_STEER_DEG)
        time.sleep(0.5)  # settle time del servo

        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;96mACTION\033[0m - "
            f"LEFT TURN 90° (R=1.02m, δ={self.STOP_TURN_STEER_DEG}°, "
            f"speed={self.STOP_TURN_SPEED}, t={self.STOP_TURN_DURATION}s)"
        )
        try:
            turn_start = time.time()
            while time.time() - turn_start < self.STOP_TURN_DURATION:
                self._send_steer(self.STOP_TURN_STEER_DEG)
                self._send_speed(self.STOP_TURN_SPEED)
                time.sleep(0.02)
            # Detener y centrar ruedas al finalizar el arco
            self._send_speed(0)
            self._send_steer(0)
        finally:
            if self.steer_override_event:
                self.steer_override_event.clear()
            if self.sign_action_event:
                self.sign_action_event.clear()
            print(
                f"\033[1;97m[ SignActions ] :\033[0m \033[1;92mRESUME\033[0m - "
                f"LEFT TURN complete — returning control to line following"
            )

    def _execute_stop(self):
        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;91mACTION\033[0m - "
            f"STOP ({self.STOP_DURATION}s)"
        )
        if self.sign_action_event:
            self.sign_action_event.set()

        if self.STOP_LEFT_TURN_ENABLED:
            # Tomar control del steer inmediatamente para que el line-following
            # no mueva las ruedas durante la parada. Se centra el servo a 0°
            # así las ruedas quedan rectas antes de iniciar el giro.
            if self.steer_override_event:
                self.steer_override_event.set()
            self._send_steer(0)

        self._send_speed(0)
        self.is_stopped = True
        time.sleep(self.STOP_DURATION)
        self.is_stopped = False

        if self.STOP_LEFT_TURN_ENABLED:
            self._execute_left_turn_90()
        else:
            self._send_speed(self.current_speed)
            if self.sign_action_event:
                self.sign_action_event.clear()

    def _execute_crosswalk(self):
        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;93mACTION\033[0m - "
            f"CROSSWALK - slow to {self.LOW_SPEED} ({self.CROSSWALK_DURATION}s)"
        )
        if self.sign_action_event:
            self.sign_action_event.set()
        self._send_speed(self.LOW_SPEED)
        time.sleep(self.CROSSWALK_DURATION)
        self._send_speed(self.current_speed)
        if self.sign_action_event:
            self.sign_action_event.clear()

    def _execute_slow_down(self):
        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;93mACTION\033[0m - "
            f"YELLOW LIGHT - slow to {self.LOW_SPEED}"
        )
        self._send_speed(self.LOW_SPEED)

    def _execute_go(self):
        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;92mACTION\033[0m - "
            f"GREEN LIGHT - resume speed {self.current_speed}"
        )
        self.is_stopped = False
        self._send_speed(self.current_speed)

    def _execute_speed_limit(self, speed):
        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;96mACTION\033[0m - "
            f"SPEED LIMIT {speed}"
        )
        self.current_speed = speed
        if not self.is_stopped:
            self._send_speed(speed)

    def _execute_parking(self):
        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;96mACTION\033[0m - "
            f"PARKING - stop"
        )
        if self.sign_action_event:
            self.sign_action_event.set()
        self._send_speed(0)
        self.is_stopped = True

    def _execute_highway_entrance(self):
        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;92mACTION\033[0m - "
            f"HIGHWAY ENTRANCE - speed up to {self.HIGHWAY_SPEED}"
        )
        self.current_speed = self.HIGHWAY_SPEED
        if not self.is_stopped:
            self._send_speed(self.HIGHWAY_SPEED)
        if self.highway_mode_event:
            self.highway_mode_event.set()

    def _execute_highway_exit(self):
        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;93mACTION\033[0m - "
            f"HIGHWAY EXIT - slow down to {self.BASE_SPEED}"
        )
        self.current_speed = self.BASE_SPEED
        if not self.is_stopped:
            self._send_speed(self.BASE_SPEED)
        if self.highway_mode_event:
            self.highway_mode_event.clear()
