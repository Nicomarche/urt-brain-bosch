"""Topic registry.

Authoritative catalog of every channel on the pub/sub bus. Replaces the
legacy ``src/core/messaging/allMessages.py`` enum module — instead of
introspecting enum members to derive topic names and QoS at runtime, every
topic is an explicit ``Topic[T]`` constant declared here.

A ``Topic`` carries:

  * ``name``: the ZMQ subscription filter prefix used on the wire. Format
    is ``/auto/<owner>/<classname>`` (all lowercased) to preserve byte-for-
    byte compatibility with the names the shim used to derive from
    ``allMessages``; that way pubs and subs that haven't migrated yet stay
    on the same wire-name as those that have.
  * ``payload_type``: the documented payload class. Not enforced at runtime
    yet; used as a type hint for callers.
  * ``qos_preset``: one of ``"STREAM" | "COMMAND" | "LATCHED" | "EVENT"``.
    The shim translates this to the matching ``QoSProfile`` from
    :mod:`src.core.bus.qos` when it builds publishers/subscribers.

The naming convention for constants is SCREAMING_SNAKE_CASE of the legacy
Enum class name (``StateChange`` → ``STATE_CHANGE``, ``mainCamera`` →
``MAIN_CAMERA``). The order below mirrors the section grouping of the old
``allMessages.py`` so a reader who knew the file can find each message in
the same place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Type, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Topic(Generic[T]):
    """Named, typed channel on the pub/sub bus.

    Args:
        name: Hierarchical string used as the ZMQ subscription filter
            prefix. Avoid trailing slashes.
        payload_type: Documented payload class. Type-hint only — no runtime
            ``isinstance`` enforcement.
        qos_preset: ``"STREAM"`` (high-rate sensor / perception, lossy
            newest-wins), ``"COMMAND"`` (reliable command plane, small
            buffer), ``"LATCHED"`` (sticky state, broker replays last value
            to late subscribers) or ``"EVENT"`` (discrete reliable events).
    """

    name: str
    payload_type: Type[T]
    qos_preset: str = "EVENT"


# Demo topic — exercised by the smoke test in tests/core/test_bus_smoke.py.
DEMO_HEARTBEAT: Topic[dict] = Topic("/demo/heartbeat", dict)


# ─── processCamera ──────────────────────────────────────────────────────────
MAIN_CAMERA: Topic[str] = Topic("/auto/threadcamera/maincamera", str, "STREAM")
SERIAL_CAMERA: Topic[str] = Topic("/auto/threadcamera/serialcamera", str, "STREAM")
RECORDING: Topic[bool] = Topic("/auto/threadcamera/recording", bool, "EVENT")
SIGNAL: Topic[str] = Topic("/auto/threadcamera/signal", str, "EVENT")
LANE_KEEPING: Topic[int] = Topic("/auto/threadcamera/lanekeeping", int, "EVENT")

# ─── processCarsAndSemaphores ───────────────────────────────────────────────
CARS: Topic[dict] = Topic("/auto/threadcarsandsemaphores/cars", dict, "EVENT")
SEMAPHORES: Topic[dict] = Topic("/auto/threadcarsandsemaphores/semaphores", dict, "EVENT")

# ─── From Dashboard ─────────────────────────────────────────────────────────
SPEED_MOTOR: Topic[str] = Topic("/auto/dashboard/speedmotor", str, "COMMAND")
STEER_MOTOR: Topic[str] = Topic("/auto/dashboard/steermotor", str, "COMMAND")
CONTROL: Topic[dict] = Topic("/auto/dashboard/control", dict, "COMMAND")
BRAKE: Topic[str] = Topic("/auto/dashboard/brake", str, "COMMAND")
RECORD: Topic[str] = Topic("/auto/dashboard/record", str, "EVENT")
CONFIG: Topic[dict] = Topic("/auto/dashboard/config", dict, "EVENT")
# Klem is the relay-on state of the car (engine-ON gate). LATCHED so a GUI
# that connects after the operator armed/disarmed sees the current value.
KLEM: Topic[str] = Topic("/auto/dashboard/klem", str, "LATCHED")
DRIVING_MODE: Topic[str] = Topic("/auto/dashboard/drivingmode", str, "EVENT")
TOGGLE_INSTANT: Topic[str] = Topic("/auto/dashboard/toggleinstant", str, "EVENT")
TOGGLE_BATTERY_LVL: Topic[str] = Topic("/auto/dashboard/togglebatterylvl", str, "EVENT")
TOGGLE_IMU_DATA: Topic[str] = Topic("/auto/dashboard/toggleimudata", str, "EVENT")
TOGGLE_RESOURCE_MONITOR: Topic[str] = Topic("/auto/dashboard/toggleresourcemonitor", str, "EVENT")
STATE: Topic[str] = Topic("/auto/dashboard/state", str, "EVENT")
BRIGHTNESS: Topic[str] = Topic("/auto/dashboard/brightness", str, "EVENT")
CONTRAST: Topic[str] = Topic("/auto/dashboard/contrast", str, "EVENT")
DROPDOWN_CHANNEL_EXAMPLE: Topic[str] = Topic(
    "/auto/dashboard/dropdownchannelexample", str, "EVENT"
)
SLIDER_CHANNEL_EXAMPLE: Topic[str] = Topic(
    "/auto/dashboard/sliderchannelexample", str, "EVENT"
)
CONTROL_CALIB: Topic[dict] = Topic("/auto/dashboard/controlcalib", dict, "EVENT")
IS_ALIVE: Topic[bool] = Topic("/auto/dashboard/isalive", bool, "EVENT")
REQUEST_STEER_LIMITS: Topic[bool] = Topic("/auto/dashboard/requeststeerlimits", bool, "EVENT")
LINE_FOLLOWING_CONFIG: Topic[dict] = Topic("/auto/dashboard/linefollowingconfig", dict, "EVENT")
LANE_CALIB_MODE: Topic[str] = Topic("/auto/dashboard/lanecalibmode", str, "EVENT")
NAVIGATION_COMMAND: Topic[dict] = Topic("/auto/dashboard/navigationcommand", dict, "EVENT")
LOCALISATION: Topic[dict] = Topic("/auto/dashboard/localisation", dict, "EVENT")
# Backwards-compatible alias for the same localisation channel — same topic
# name as LOCALISATION (msgID=24 in the legacy enum).
LOCALISATION_FIX: Topic[dict] = Topic("/auto/dashboard/localisationfix", dict, "EVENT")
TOGGLE_HALL_SPEED: Topic[str] = Topic("/auto/dashboard/togglehallspeed", str, "EVENT")
ODO_RESET: Topic[str] = Topic("/auto/dashboard/odoreset", str, "EVENT")

# ─── Calibration (bidirectional) ────────────────────────────────────────────
# GUI sends commands ({"Action": ...}); threadCalibration emits replies
# ({"action": ..., "data": ...}) on the same topic. Replaced the legacy
# SocketIO ``Calibration`` event when the Flask/eventlet bridge was removed.
CALIBRATION: Topic[dict] = Topic("/auto/threadcalibration/calibration", dict, "EVENT")

# ─── GUI handshake ──────────────────────────────────────────────────────────
# Payload: ``{"host": str, "port": int, "fps": int, "quality": int}``.
# Publisher: UdpVideoClient (every 5s while the GUI is alive). Subscriber:
# UdpVideoStreamerThread wires the target on receive; clears it after 10s
# of silence so a dead GUI stops the encoder.
VIDEO_SUBSCRIBE: Topic[dict] = Topic("/auto/gui/videosubscribe", dict, "EVENT")

# ─── From Line Following ────────────────────────────────────────────────────
LINE_FOLLOWING_DEBUG: Topic[str] = Topic(
    "/auto/threadlinefollowing/linefollowingdebug", str, "STREAM"
)
LINE_FOLLOWING_STATUS: Topic[dict] = Topic(
    "/auto/threadlinefollowing/linefollowingstatus", dict, "STREAM"
)

# ─── From Local Perception ──────────────────────────────────────────────────
LOCAL_LANE_PERCEPTION: Topic[dict] = Topic(
    "/auto/threadlocalperception/locallaneperception", dict, "STREAM"
)
LOCAL_PERCEPTION_STATUS: Topic[dict] = Topic(
    "/auto/threadlocalperception/localperceptionstatus", dict, "STREAM"
)

# ─── From Serial Writer ─────────────────────────────────────────────────────
ACTUATOR_COMMAND_STATUS: Topic[dict] = Topic(
    "/auto/threadwrite/actuatorcommandstatus", dict, "STREAM"
)

# ─── From Sign Detection (now published by threadLocalPerception) ───────────
SIGN_DETECTED: Topic[dict] = Topic("/auto/threadlocalperception/signdetected", dict, "EVENT")
SIGN_DETECTION_DEBUG: Topic[str] = Topic(
    "/auto/threadlocalperception/signdetectiondebug", str, "STREAM"
)
SIGN_DETECTION_STATUS: Topic[dict] = Topic(
    "/auto/threadlocalperception/signdetectionstatus", dict, "STREAM"
)
DETECTED_OBJECTS_MSG: Topic[dict] = Topic(
    "/auto/threadlocalperception/detectedobjectsmsg", dict, "STREAM"
)

# ─── From LiDAR ─────────────────────────────────────────────────────────────
LIDAR_SCAN_MSG: Topic[dict] = Topic("/auto/threadlidar/lidarscanmsg", dict, "STREAM")
LIDAR_STATUS_MSG: Topic[dict] = Topic("/auto/threadlidar/lidarstatusmsg", dict, "STREAM")

# ─── From Nucleo (threadRead) ───────────────────────────────────────────────
BATTERY_LVL: Topic[int] = Topic("/auto/threadread/batterylvl", int, "EVENT")
IMU_DATA: Topic[str] = Topic("/auto/threadread/imudata", str, "STREAM")
INSTANT_CONSUMPTION: Topic[float] = Topic(
    "/auto/threadread/instantconsumption", float, "EVENT"
)
RESOURCE_MONITOR: Topic[dict] = Topic("/auto/threadread/resourcemonitor", dict, "EVENT")
CURRENT_SPEED: Topic[float] = Topic("/auto/threadread/currentspeed", float, "STREAM")
CURRENT_STEER: Topic[float] = Topic("/auto/threadread/currentsteer", float, "STREAM")
IMU_ACK: Topic[str] = Topic("/auto/threadread/imuack", str, "EVENT")
SHUT_DOWN_SIGNAL: Topic[str] = Topic("/auto/threadread/shutdownsignal", str, "EVENT")
CALIB_PWM_DATA: Topic[dict] = Topic("/auto/threadread/calibpwmdata", dict, "EVENT")
CALIB_RUN_DONE: Topic[bool] = Topic("/auto/threadread/calibrundone", bool, "EVENT")
ALIVE_SIGNAL: Topic[bool] = Topic("/auto/threadread/alivesignal", bool, "EVENT")
STEERING_LIMITS: Topic[dict] = Topic("/auto/threadread/steeringlimits", dict, "EVENT")
CURRENT_DISTANCE: Topic[int] = Topic("/auto/threadread/currentdistance", int, "STREAM")

# ─── From Locsys ────────────────────────────────────────────────────────────
LOCATION: Topic[dict] = Topic("/auto/threadtrafficcommunication/location", dict, "EVENT")

# ─── From processSerialHandler ──────────────────────────────────────────────
ENABLE_BUTTON: Topic[bool] = Topic("/auto/threadwrite/enablebutton", bool, "EVENT")
WARNING_SIGNAL: Topic[str] = Topic("/auto/brain/warningsignal", str, "EVENT")
SERIAL_CONNECTION_STATE: Topic[bool] = Topic(
    "/auto/processserialhandler/serialconnectionstate", bool, "EVENT"
)

# ─── From StateMachine ──────────────────────────────────────────────────────
# The current driving mode. LATCHED so a fresh GUI sees the state without
# waiting for the next transition.
STATE_CHANGE: Topic[str] = Topic("/auto/statemachine/statechange", str, "LATCHED")

# ─── From threadTracking ────────────────────────────────────────────────────
TRACKING_DEBUG: Topic[dict] = Topic("/auto/threadtracking/trackingdebug", dict, "EVENT")
NAVIGATION_STATUS: Topic[dict] = Topic("/auto/threadtracking/navigationstatus", dict, "EVENT")

# ─── From Behavior Planner ──────────────────────────────────────────────────
BEHAVIOR_OUTPUT_MSG: Topic[dict] = Topic(
    "/auto/threadbehaviorplanner/behavioroutputmsg", dict, "EVENT"
)
BEHAVIOR_PLANNER_STATUS: Topic[dict] = Topic(
    "/auto/threadbehaviorplanner/behaviorplannerstatus", dict, "EVENT"
)

# ─── From Object Tracker (MOT) ──────────────────────────────────────────────
TRACKED_OBJECTS_MSG: Topic[dict] = Topic(
    "/auto/threadobjecttracker/trackedobjectsmsg", dict, "STREAM"
)

# ─── Sim control (sim mode only) ────────────────────────────────────────────
SIM_RELOCALIZE: Topic[dict] = Topic("/auto/threadposeestimator/simrelocalize", dict, "EVENT")

# ─── From MotionController + Dispatcher ─────────────────────────────────────
# MotorCommandMsg is the dispatcher's post-safety-gate snapshot, NOT the
# critical path (that's SPEED_MOTOR + STEER_MOTOR). Pure telemetry for the
# operator UI.
MOTOR_COMMAND_MSG: Topic[dict] = Topic(
    "/auto/threadmotorcommanddispatcher/motorcommandmsg", dict, "EVENT"
)


# Topics whose last value the broker must replay to late subscribers (ROS:
# "transient_local"). Derived from the LATCHED constants above so we don't
# duplicate the membership list.
LATCHED_TOPIC_NAMES: frozenset[str] = frozenset({KLEM.name, STATE_CHANGE.name})


# Back-compat lookup: legacy ``allMessages`` enum class names (PascalCase /
# camelCase) → Topic constant. Used by code that still routes by SocketIO-
# style name strings (mainly :mod:`src.gui.client.zmq_bus_client`). New
# code should import the Topic constants directly.
LEGACY_NAME_TO_TOPIC: dict[str, Topic] = {
    "mainCamera": MAIN_CAMERA,
    "serialCamera": SERIAL_CAMERA,
    "Recording": RECORDING,
    "Signal": SIGNAL,
    "LaneKeeping": LANE_KEEPING,
    "Cars": CARS,
    "Semaphores": SEMAPHORES,
    "SpeedMotor": SPEED_MOTOR,
    "SteerMotor": STEER_MOTOR,
    "Control": CONTROL,
    "Brake": BRAKE,
    "Record": RECORD,
    "Config": CONFIG,
    "Klem": KLEM,
    "DrivingMode": DRIVING_MODE,
    "ToggleInstant": TOGGLE_INSTANT,
    "ToggleBatteryLvl": TOGGLE_BATTERY_LVL,
    "ToggleImuData": TOGGLE_IMU_DATA,
    "ToggleResourceMonitor": TOGGLE_RESOURCE_MONITOR,
    "State": STATE,
    "Brightness": BRIGHTNESS,
    "Contrast": CONTRAST,
    "DropdownChannelExample": DROPDOWN_CHANNEL_EXAMPLE,
    "SliderChannelExample": SLIDER_CHANNEL_EXAMPLE,
    "ControlCalib": CONTROL_CALIB,
    "Calibration": CALIBRATION,
    "VideoSubscribe": VIDEO_SUBSCRIBE,
    "IsAlive": IS_ALIVE,
    "RequestSteerLimits": REQUEST_STEER_LIMITS,
    "LineFollowingConfig": LINE_FOLLOWING_CONFIG,
    "LaneCalibMode": LANE_CALIB_MODE,
    "NavigationCommand": NAVIGATION_COMMAND,
    "Localisation": LOCALISATION,
    "LocalisationFix": LOCALISATION_FIX,
    "ToggleHallSpeed": TOGGLE_HALL_SPEED,
    "OdoReset": ODO_RESET,
    "LineFollowingDebug": LINE_FOLLOWING_DEBUG,
    "LineFollowingStatus": LINE_FOLLOWING_STATUS,
    "LocalLanePerception": LOCAL_LANE_PERCEPTION,
    "LocalPerceptionStatus": LOCAL_PERCEPTION_STATUS,
    "ActuatorCommandStatus": ACTUATOR_COMMAND_STATUS,
    "SignDetected": SIGN_DETECTED,
    "SignDetectionDebug": SIGN_DETECTION_DEBUG,
    "SignDetectionStatus": SIGN_DETECTION_STATUS,
    "DetectedObjectsMsg": DETECTED_OBJECTS_MSG,
    "LidarScanMsg": LIDAR_SCAN_MSG,
    "LidarStatusMsg": LIDAR_STATUS_MSG,
    "BatteryLvl": BATTERY_LVL,
    "ImuData": IMU_DATA,
    "InstantConsumption": INSTANT_CONSUMPTION,
    "ResourceMonitor": RESOURCE_MONITOR,
    "CurrentSpeed": CURRENT_SPEED,
    "CurrentSteer": CURRENT_STEER,
    "ImuAck": IMU_ACK,
    "ShutDownSignal": SHUT_DOWN_SIGNAL,
    "CalibPWMData": CALIB_PWM_DATA,
    "CalibRunDone": CALIB_RUN_DONE,
    "AliveSignal": ALIVE_SIGNAL,
    "SteeringLimits": STEERING_LIMITS,
    "CurrentDistance": CURRENT_DISTANCE,
    "Location": LOCATION,
    "EnableButton": ENABLE_BUTTON,
    "WarningSignal": WARNING_SIGNAL,
    "SerialConnectionState": SERIAL_CONNECTION_STATE,
    "StateChange": STATE_CHANGE,
    "TrackingDebug": TRACKING_DEBUG,
    "NavigationStatus": NAVIGATION_STATUS,
    "BehaviorOutputMsg": BEHAVIOR_OUTPUT_MSG,
    "BehaviorPlannerStatus": BEHAVIOR_PLANNER_STATUS,
    "TrackedObjectsMsg": TRACKED_OBJECTS_MSG,
    "SimRelocalize": SIM_RELOCALIZE,
    "MotorCommandMsg": MOTOR_COMMAND_MSG,
}
