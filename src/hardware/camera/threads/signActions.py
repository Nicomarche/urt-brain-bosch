import threading
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

    # Giro a la derecha tras el PRIMER STOP de la corrida.
    # Todos los STOP frenan SIGN_STOP_DURATION segundos; solo el primero añade
    # un giro a la derecha de STOP_TURN_DURATION a la velocidad de crucero.
    STOP_TURN_STEER_DEG = getattr(config, "SIGN_STOP_TURN_STEER_DEG", 25.0)
    STOP_TURN_DURATION  = getattr(config, "SIGN_STOP_TURN_DURATION",  8.0)

    # Rotonda: trigger por trailing edge.
    # Se registra la señal mientras se ve (sin actuar). Cuando deja de verse
    # durante ROUNDABOUT_LOST_THRESHOLD segundos, se asume que el auto ya la
    # pasó y se inicia un giro a la derecha fijo (ignorando la cámara) por
    # ROUNDABOUT_DURATION segundos hasta caer en el siguiente carril.
    ROUNDABOUT_STEER_DEG    = getattr(config, "SIGN_ROUNDABOUT_STEER_DEG",    25.0)
    ROUNDABOUT_DURATION     = getattr(config, "SIGN_ROUNDABOUT_DURATION",     8.0)
    ROUNDABOUT_LOST_THRESHOLD = getattr(config, "SIGN_ROUNDABOUT_LOST_THRESHOLD", 0.4)

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
        "roundabout",
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
        "roundabout": "roundabout",
    }

    # Tiempo de gracia después de un giro fijo durante el cual se bypasea el
    # requisito de esperar curva recta para ejecutar señales pendientes.
    # Esto permite que hw_entry y otras señales se ejecuten inmediatamente
    # después del giro en lugar de quedar bloqueadas por curve_state=IN_CURVE.
    DEFAULT_FIXED_TURN_GRACE_S = 0.5

    def __init__(self, queuesList, sign_action_event=None, action_cooldown=15.0,
                 highway_mode_event=None, steer_override_event=None,
                 crosswalk_done_callback=None):
        self.queuesList = queuesList
        self.sign_action_event = sign_action_event
        self.highway_mode_event = highway_mode_event
        self.steer_override_event = steer_override_event
        self.action_cooldown = action_cooldown
        # Called (with no args) after a crosswalk action completes.
        self.crosswalk_done_callback = crosswalk_done_callback
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
        self._fixed_turn_end_time = 0.0
        self.fixed_turn_grace_s = self.DEFAULT_FIXED_TURN_GRACE_S
        # Hilo daemon que ejecuta acciones bloqueantes (stop+giro, crosswalk).
        # Permite que thread_work() siga corriendo para detección e inferencia.
        self._blocking_action_thread = None
        # Rotonda: timestamp de la última detección. tick() dispara el giro
        # cuando esto supera ROUNDABOUT_LOST_THRESHOLD segundos sin ver la señal.
        self._roundabout_last_seen_at = 0.0
        # Contador de STOPs ejecutados en esta corrida. El primero dispara el
        # giro a la derecha; los siguientes solo frenan y reanudan.
        self._stop_count = 0

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

    # Señales cuya acción bloquea el hilo (tienen time.sleep internos).
    # Se ejecutan en un hilo daemon para no congelar thread_work().
    _ASYNC_BLOCKING_SIGNS = frozenset({"stop", "red_light", "no_entry", "crosswalk"})

    def _is_blocking_action_running(self):
        """True si hay una acción bloqueante corriendo en el hilo daemon."""
        return (self._blocking_action_thread is not None and
                self._blocking_action_thread.is_alive())

    def _in_fixed_turn_grace(self):
        """True si estamos dentro del período de gracia post-giro fijo."""
        return (self._fixed_turn_end_time > 0.0 and
                time.time() - self._fixed_turn_end_time < float(self.fixed_turn_grace_s))

    def _requires_curve_exit(self, sign_name):
        if sign_name not in {"highway_entrance"}:
            return False
        # Durante el período de gracia post-giro fijo no esperamos curva recta.
        if self._in_fixed_turn_grace():
            return False
        return True

    def _should_release_pending(self, curve_state, steering_deg):
        # Durante el período de gracia post-giro fijo, liberar inmediatamente.
        if self._in_fixed_turn_grace():
            print(
                f"\033[1;97m[ SignActions ] :\033[0m \033[1;96mGRACE\033[0m - "
                f"Liberando señal pendiente '{self.pending_sign}' (post-giro fijo)"
            )
            return True

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
        # Rotonda: si ya no la vemos hace ROUNDABOUT_LOST_THRESHOLD seg., asumimos
        # que el auto la pasó y disparamos el giro hardcodeado a la derecha.
        if self._roundabout_last_seen_at > 0.0:
            elapsed = time.time() - self._roundabout_last_seen_at
            if elapsed >= self.ROUNDABOUT_LOST_THRESHOLD:
                self._roundabout_last_seen_at = 0.0
                if not self._is_blocking_action_running():
                    now = time.time()
                    self.last_sign = "roundabout"
                    self.last_action_time["roundabout"] = now

                    t = threading.Thread(
                        target=self._execute_roundabout,
                        daemon=True,
                        name="sign-roundabout",
                    )
                    self._blocking_action_thread = t
                    t.start()

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

        # No iniciar nueva acción mientras una bloqueante sigue corriendo.
        if self._is_blocking_action_running():
            return False

        # Rotonda: trigger por trailing edge. Solo registramos cuándo se vio;
        # tick() ejecuta el giro cuando la señal deja de verse.
        if sign_name == "roundabout":
            last_time = self.last_action_time.get("roundabout", 0.0)
            if (time.time() - last_time) < self.action_cooldown:
                return False
            now = time.time()
            if self._roundabout_last_seen_at == 0.0:
                print(
                    f"\033[1;97m[ SignActions ] :\033[0m \033[1;96mARMED\033[0m - "
                    f"ROUNDABOUT visible — esperando perder la señal para girar"
                )
            self._roundabout_last_seen_at = now
            return True

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

        # Acciones bloqueantes (tienen sleeps) se corren en un hilo daemon para
        # que thread_work() siga iterando: inferencia, preview y detección continúan.
        if sign_name in self._ASYNC_BLOCKING_SIGNS:
            self.last_sign = sign_name
            self.last_action_time[action_group] = now

            def _run(sn=sign_name):
                self._execute_canonical_sign(sn)

            t = threading.Thread(target=_run, daemon=True, name=f"sign-{sign_name}")
            self._blocking_action_thread = t
            t.start()
            return True

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

    def _execute_first_stop_right_turn(self):
        """Giro a la derecha hardcodeado tras el PRIMER STOP de la corrida.

        Se llama con el auto ya completamente detenido (después de STOP_DURATION).
        Mantiene la velocidad de crucero actual durante todo el giro.
        Secuencia:
          1. Toma control del steer (bloquea line-following).
          2. Centra ruedas a 0° (breve settle).
          3. Aplica ángulo de giro a la derecha y deja que el servo llegue.
          4. Avanza durante STOP_TURN_DURATION a la velocidad de crucero.
          5. Centra ruedas (manteniendo la velocidad de crucero) y devuelve
             control al line-following.
        """
        cruise_speed = self.current_speed

        if self.steer_override_event:
            self.steer_override_event.set()

        # Centrar ruedas primero (el auto está parado, esto no afecta la trayectoria)
        self._send_steer(0)
        time.sleep(0.3)

        # Aplicar ángulo de giro y esperar que el servo llegue al ángulo
        self._send_steer(self.STOP_TURN_STEER_DEG)
        time.sleep(0.5)

        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;96mACTION\033[0m - "
            f"FIRST STOP RIGHT TURN (δ={self.STOP_TURN_STEER_DEG}°, "
            f"speed={cruise_speed}, t={self.STOP_TURN_DURATION}s)"
        )
        try:
            turn_start = time.time()
            while time.time() - turn_start < self.STOP_TURN_DURATION:
                self._send_steer(self.STOP_TURN_STEER_DEG)
                self._send_speed(cruise_speed)
                time.sleep(0.02)
            # Centrar ruedas y mantener la velocidad de crucero al salir.
            self._send_steer(0)
            self._send_speed(cruise_speed)
        finally:
            if self.steer_override_event:
                self.steer_override_event.clear()
            if self.sign_action_event:
                self.sign_action_event.clear()
            self._fixed_turn_end_time = time.time()
            print(
                f"\033[1;97m[ SignActions ] :\033[0m \033[1;92mRESUME\033[0m - "
                f"FIRST STOP RIGHT TURN complete — returning control to line following "
                f"(grace {self.fixed_turn_grace_s:.1f}s)"
            )

    def _execute_roundabout(self):
        """Giro derecha fijo para entrar en el primer carril de la rotonda.

        Se llama desde tick() cuando la señal dejó de verse (trailing edge),
        asumiendo que el auto ya pasó el cartel y está en la entrada.
        Secuencia:
          1. Toma control de speed y steer (bloquea line-following).
          2. Aplica ángulo a la derecha y mantiene durante ROUNDABOUT_DURATION,
             ignorando la cámara.
          3. Centra ruedas y devuelve control a la autonomía.
        """
        # Mantiene la velocidad de crucero actual durante el giro (no cambia speed).
        cruise_speed = self.current_speed
        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;95mACTION\033[0m - "
            f"ROUNDABOUT - hard right (δ={self.ROUNDABOUT_STEER_DEG}°, "
            f"speed={cruise_speed}, t={self.ROUNDABOUT_DURATION}s)"
        )
        if self.sign_action_event:
            self.sign_action_event.set()
        if self.steer_override_event:
            self.steer_override_event.set()

        try:
            turn_start = time.time()
            while time.time() - turn_start < self.ROUNDABOUT_DURATION:
                self._send_steer(self.ROUNDABOUT_STEER_DEG)
                self._send_speed(cruise_speed)
                time.sleep(0.02)
            # Centrar ruedas y mantener la velocidad de crucero al salir.
            self._send_steer(0)
            self._send_speed(cruise_speed)
        finally:
            if self.steer_override_event:
                self.steer_override_event.clear()
            if self.sign_action_event:
                self.sign_action_event.clear()
            self._fixed_turn_end_time = time.time()
            print(
                f"\033[1;97m[ SignActions ] :\033[0m \033[1;92mRESUME\033[0m - "
                f"ROUNDABOUT complete — returning control to line following"
            )

    def _execute_stop(self):
        self._stop_count += 1
        is_first_stop = (self._stop_count == 1)

        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;91mACTION\033[0m - "
            f"STOP #{self._stop_count} ({self.STOP_DURATION}s)"
            + (" → RIGHT TURN" if is_first_stop else "")
        )
        if self.sign_action_event:
            self.sign_action_event.set()

        # Frenar primero. Durante el frenado y la parada, el line-following sigue
        # controlando el steer para mantener la alineación de carril (comportamiento correcto).
        self._send_speed(0)
        self.is_stopped = True
        time.sleep(self.STOP_DURATION)
        self.is_stopped = False

        if is_first_stop:
            # Auto completamente parado: tomar control del steer y girar a la derecha.
            self._execute_first_stop_right_turn()
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
        if self.crosswalk_done_callback is not None:
            try:
                self.crosswalk_done_callback()
            except Exception as e:
                print(
                    f"\033[1;97m[ SignActions ] :\033[0m \033[1;91mERROR\033[0m - "
                    f"crosswalk_done_callback failed: {e}"
                )

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
