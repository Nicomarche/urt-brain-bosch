import time

from src.utils.messages.allMessages import SpeedMotor


class SignActions:
    """Shared sign action executor used by local and legacy sign pipelines."""

    BASE_SPEED = 5
    LOW_SPEED = 3
    SPEED_20 = 3
    SPEED_30 = 5
    HIGHWAY_SPEED = 7
    STOP_DURATION = 3.0
    CROSSWALK_DURATION = 3.0

    ACTIONABLE_SIGNS = {
        "stop", "no_entry", "crosswalk", "red_light", "yellow_light",
        "green_light", "speed_20", "speed_30", "parking",
        "highway_entrance", "highway_exit",
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
        "highway_entrance": "highway",
        "highway_exit": "highway",
    }

    def __init__(self, queuesList, sign_action_event=None, action_cooldown=15.0,
                 highway_mode_event=None):
        self.queuesList = queuesList
        self.sign_action_event = sign_action_event
        self.highway_mode_event = highway_mode_event
        self.action_cooldown = action_cooldown
        self.last_sign = None
        self.last_action_time = {}
        self.current_speed = self.BASE_SPEED
        self.is_stopped = False

    def execute(self, sign_name):
        """Execute the action mapped to a detected sign."""
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
            print(
                f"\033[1;97m[ SignActions ] :\033[0m \033[1;93mINFO\033[0m - "
                f"{sign_name} detected (no action)"
            )
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

    def _execute_stop(self):
        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;91mACTION\033[0m - "
            f"STOP ({self.STOP_DURATION}s)"
        )
        if self.sign_action_event:
            self.sign_action_event.set()
        self._send_speed(0)
        self.is_stopped = True
        time.sleep(self.STOP_DURATION)
        self.is_stopped = False
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
