"""Constants for every SocketIO event the GUI exchanges with ``processDashboard.py``.

Single source of truth: every other module imports from here instead of
sprinkling string literals around. The names mirror the contract exposed by
``src/dashboard/processDashboard.py`` (which is what the legacy Angular
dashboard also uses) so changing a name here means changing the matching emit
in the backend, and vice versa.

Categories:

* ``CMD_*``      — generic ``message`` envelope. Sent as
                   ``{"Name": Cmd, "Value": payload}``.
* ``EVT_*``      — events the backend pushes to clients.
* ``ROUTE_*``    — table-state IO (legacy ``save``/``load`` events).
* ``SYS_*``      — session and heartbeat events that don't follow the
                   Name/Value envelope.
"""

# ---- Outgoing (GUI → backend), wrapped as {"Name": <CMD>, "Value": ...} ----
CMD_KLEM = "Klem"                            # "0" / "15" / "30"
CMD_DRIVING_MODE = "DrivingMode"             # "stop" / "manual" / "legacy" / "auto" / "parking"
CMD_SPEED = "SpeedMotor"                     # int -50..+50
CMD_STEER = "SteerMotor"                     # int (tenths of deg) -600..+600 in sim, ±250 hw
CMD_BRAKE = "Brake"                          # bool
CMD_RECORDING = "Recording"                  # bool
CMD_CALIBRATION = "Calibration"              # dict (action + payload)
CMD_NAV_COMMAND = "NavigationCommand"        # {"x": float, "y": float} or {"mode": "clear"}
CMD_LOCALISATION = "Localisation"            # {"world_x": float, "world_y": float}
CMD_GET_SERIAL_STATE = "GetCurrentSerialConnectionState"  # ()
CMD_REQUEST_STEER_LIMITS = "RequestSteerLimits"           # bool

# ---- Session / heartbeat (out) — sent as the Name field of `message` ----
CMD_SESSION_ACCESS = "SessionAccess"
CMD_HEARTBEAT = "Heartbeat"
CMD_SESSION_END = "SessionEnd"

# ---- Incoming (backend → GUI), each is its own SocketIO event ----
EVT_ENABLE_BUTTON = "EnableButton"
EVT_RESPONSE = "response"
EVT_SESSION_ACCESS = "session_access"
EVT_CURRENT_SERIAL_STATE = "current_serial_connection_state"
EVT_HEARTBEAT = "heartbeat"
EVT_HEARTBEAT_DISCONNECT = "heartbeat_disconnect"
EVT_MEMORY = "memory_channel"
EVT_CPU = "cpu_channel"
EVT_BATTERY = "BatteryLvl"
EVT_INSTANT_CONSUMPTION = "InstantConsumption"
EVT_CURRENT_SPEED = "CurrentSpeed"
EVT_CURRENT_STEER = "CurrentSteer"
EVT_STEERING_LIMITS = "SteeringLimits"
EVT_STATE_CHANGE = "StateChange"
EVT_WARNING_SIGNAL = "WarningSignal"
EVT_RECORDING = "Recording"
EVT_LOCATION = "Location"
EVT_CARS = "Cars"
EVT_SEMAPHORES = "Semaphores"
EVT_NAVIGATION_STATUS = "NavigationStatus"
EVT_BEHAVIOR_OUTPUT = "BehaviorOutputMsg"
EVT_LIDAR_SCAN = "LidarScanMsg"
EVT_LIDAR_STATUS = "LidarStatusMsg"
EVT_DETECTED_OBJECTS = "DetectedObjectsMsg"
EVT_TRACKED_OBJECTS = "TrackedObjectsMsg"
EVT_CAMERA_BASE64 = "serialCamera"           # legacy: base64 PNG (JSON envelope)
EVT_CAMERA_BIN = "serialCamera_bin"          # NEW: raw JPEG bytes (preferred)
EVT_LINE_FOLLOWING_DEBUG = "LineFollowingDebug"
EVT_LINE_FOLLOWING_STATUS = "LineFollowingStatus"
EVT_CALIBRATION = "Calibration"
EVT_CALIB_RUN_DONE = "CalibRunDone"
EVT_CALIB_PWM_DATA = "CalibPWMData"
EVT_CONSOLE_LOG = "console_log"
EVT_LOAD_BACK = "loadBack"
EVT_MOTOR_COMMAND_MSG = "MotorCommandMsg"
EVT_GPS_FIX = "GpsFix"                  # {"world_x": float, "world_y": float, "timestamp": float}
EVT_DEAD_RECKONING = "DeadReckoning"    # {"x": float, "y": float, "yaw_deg": float, "timestamp": float}

# ---- Routes / params persistence (table-state contract) -------------------
ROUTE_SAVE = "save"
ROUTE_LOAD = "load"

# ---- Driving modes (canonical strings the backend expects) ---------------
MODE_STOP = "stop"
MODE_MANUAL = "manual"
MODE_LEGACY = "legacy"
MODE_AUTO = "auto"
MODE_PARKING = "parking"

# ---- KL switch positions --------------------------------------------------
KL_OFF = "0"
KL_ACC = "15"
KL_RUN = "30"
