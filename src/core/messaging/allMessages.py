# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

from enum import Enum

####################################### processCamera #######################################
class mainCamera(Enum):
    Queue = "General"
    Owner = "threadCamera"
    msgID = 1
    msgType = "str"

class serialCamera(Enum):
    Queue = "General"
    Owner = "threadCamera"
    msgID = 2
    msgType = "str"

class Recording(Enum):
    Queue = "General"
    Owner = "threadCamera"
    msgID = 3
    msgType = "bool"

class Signal(Enum):
    Queue = "General"
    Owner = "threadCamera"
    msgID = 4
    msgType = "str"

class LaneKeeping(Enum):
    Queue = "General"
    Owner = "threadCamera" # here you will send an offset of the car position between the lanes of the road + - from 0 point to dashboard
    msgID = 5
    msgType = "int"

################################# processCarsAndSemaphores ##################################
class Cars(Enum):
    Queue = "General"
    Owner = "threadCarsAndSemaphores"
    msgID = 1
    msgType = "dict"

class Semaphores(Enum):
    Queue = "General"
    Owner = "threadCarsAndSemaphores"
    msgID = 2
    msgType = "dict"

################################# From Dashboard ##################################
class SpeedMotor(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 1
    msgType = "str"

class SteerMotor(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 2
    msgType = "str"

class Control(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 3
    msgType = "dict"

class Brake(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 4
    msgType = "str"

class Record(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 5
    msgType = "str"

class Config(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 6
    msgType = "dict"

class Klem(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 7
    msgType = "str"

class DrivingMode(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 8
    msgType = "str"

class ToggleInstant(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 9
    msgType = "str"

class ToggleBatteryLvl(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 10
    msgType = "str"

class ToggleImuData(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 11
    msgType = "str"

class ToggleResourceMonitor(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 12
    msgType = "str"

class State(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 13
    msgType = "str"

class Brightness(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 14
    msgType = "str"

class Contrast(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 15
    msgType = "str"

class DropdownChannelExample(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 16
    msgType = "str"

class SliderChannelExample(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 17
    msgType = "str"

class ControlCalib(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 18
    msgType = "dict"

class IsAlive(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 19
    msgType = "bool"

class RequestSteerLimits(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 20
    msgType = "bool"

class LineFollowingConfig(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 21
    msgType = "dict"

class LaneCalibMode(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 22
    msgType = "str"  # "true" / "false"

class NavigationCommand(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 23
    msgType = "dict"  # {"x": float, "y": float} or {"mode": "clear"|"reset"|"go_to"...}

class Localisation(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 24
    # ROS-compatible localisation payload:
    # {"timestamp": float, "posA": float, "posB": float, "rotA": float, "rotB": float}
    # Optional manual metadata may be included under "meta".
    msgType = "dict"

class LocalisationFix(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 24
    # Backwards-compatible alias for the same localisation channel.
    msgType = "dict"

################################# From Line Following ##################################
class LineFollowingDebug(Enum):
    Queue = "General"
    Owner = "threadLineFollowing"
    msgID = 1
    msgType = "str"  # base64 encoded image

class LineFollowingStatus(Enum):
    Queue = "General"
    Owner = "threadLineFollowing"
    msgID = 2
    msgType = "dict"  # steering, speed, mode, fps, etc.

################################# From Local Perception ##################################
class LocalLanePerception(Enum):
    Queue = "General"
    Owner = "threadLocalPerception"
    msgID = 1
    msgType = "dict"  # lane_points, lane_side_*, timing, frame metadata

class LocalPerceptionStatus(Enum):
    Queue = "General"
    Owner = "threadLocalPerception"
    msgID = 2
    msgType = "dict"  # model_ready, fps, last_sign, etc.

################################# From Serial Writer ##################################
class ActuatorCommandStatus(Enum):
    Queue = "General"
    Owner = "threadWrite"
    msgID = 2
    msgType = "dict"  # klem, serial state, last movement command, block reason

################################# From Sign Detection ##################################
# NOTA: hasta el cleanup de Fase 0 estos mensajes los publicaba el threadSignDetection
# remoto (eliminado). Hoy el publisher es threadLocalPerception (YOLO local). El campo
# Owner se actualiza para que el routing del gateway no quede con metadata estancada.
class SignDetected(Enum):
    Queue = "General"
    Owner = "threadLocalPerception"
    msgID = 1
    msgType = "dict"  # {"sign": str, "confidence": float, "timestamp": float}

class SignDetectionDebug(Enum):
    Queue = "General"
    Owner = "threadLocalPerception"
    msgID = 2
    msgType = "str"  # base64 encoded debug image with bounding boxes

class SignDetectionStatus(Enum):
    Queue = "General"
    Owner = "threadLocalPerception"
    msgID = 3
    msgType = "dict"  # {"enabled": bool, "fps": float, "last_sign": str}


################################# From Nucleo ##################################
class BatteryLvl(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 1
    msgType = "int"

class ImuData(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 2
    msgType = "str"

class InstantConsumption(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 3
    msgType = "float"

class ResourceMonitor(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 4
    msgType = "dict"

class CurrentSpeed(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 5
    msgType = "float"

class CurrentSteer(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 6
    msgType = "float"

class ImuAck(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 7
    msgType = "str"
    
class ShutDownSignal(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 8
    msgType = "str"

class CalibPWMData(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 9
    msgType = "dict"

class CalibRunDone(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 10
    msgType = "bool"

class AliveSignal(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 11
    msgType = "bool"

class SteeringLimits(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 12
    msgType = "dict"


################################# From Locsys ##################################
class Location(Enum):
    Queue = "General"
    Owner = "threadTrafficCommunication"
    msgID = 1
    msgType = "dict"

######################    From processSerialHandler  ###########################
class EnableButton(Enum):
    Queue = "General"
    Owner = "threadWrite"
    msgID = 1
    msgType = "bool"

class WarningSignal(Enum):
    Queue = "General"
    Owner = "brain"
    msgID = 2
    msgType = "str"

class SerialConnectionState(Enum):
    Queue = "General"
    Owner = "processSerialHandler"
    msgID = 3
    msgType = "bool"

################################# From StateMachine ##################################
class StateChange(Enum):
    Queue = "Critical"
    Owner = "stateMachine"
    msgID = 1
    msgType = "str"

### It will have this format: {"WarningName":"name1", "WarningID": 1}

################################# From threadTracking ##################################
class TrackingDebug(Enum):
    Queue = "General"
    Owner = "threadTracking"
    msgID = 1
    # value format: {"x": float, "y": float, "yaw": float,
    #                "error_m": float, "heading_rad": float,
    #                "speed_mps": float, "wp_idx": int,
    #                "waypoint_mode_active": bool, "node_attr": int}
    msgType = "dict"

class NavigationStatus(Enum):
    Queue = "General"
    Owner = "threadTracking"
    msgID = 2
    msgType = "dict"  # route state, current/upcoming node, maneuver, route preview, lanelet diagnostics

################################# From Behavior Planner ##################################
# BehaviorOutput es la única fuente de verdad de velocidad y referencia de path:
# producido por BehaviorPlanner.plan() a partir de pose+route+lane+regulators+tracked_objects.
# El MotionController (Acados MPC) lo consume para generar steering+throttle.
class BehaviorOutputMsg(Enum):
    Queue = "General"
    Owner = "threadBehaviorPlanner"
    msgID = 1
    # value format: {"timestamp": float, "dt": float, "scenario_name": str,
    #                "valid": bool, "stop_required": bool,
    #                "target_path": list[list[float]],   # (N+1, 3) -> [x, y, psi]
    #                "speed_profile": list[float],        # (N,)
    #                "notes": dict}
    msgType = "dict"

class BehaviorPlannerStatus(Enum):
    Queue = "General"
    Owner = "threadBehaviorPlanner"
    msgID = 2
    # value format: {"active_scenario": str, "horizon_n": int, "fps": float,
    #                "last_plan_dt_ms": float}
    msgType = "dict"

################################# From Object Tracker (MOT) ##################################
class TrackedObjectsMsg(Enum):
    Queue = "General"
    Owner = "threadObjectTracker"
    msgID = 1
    # value format: {"timestamp": float,
    #                "objects": [{"track_id": int, "kind": str,
    #                             "x": float, "y": float, "vx": float, "vy": float,
    #                             "confidence": float, "age_frames": int}, ...]}
    msgType = "dict"

################################# Sim control (sim mode only) ##################################
class SimRelocalize(Enum):
    Queue = "General"
    Owner = "threadPoseEstimator"
    msgID = 1
    msgType = "dict"  # {"world_x": float, "world_y": float, "yaw_rad": float, "z"?: float} (legacy: gz_*)

################################# From MotionController + Dispatcher ##################################
# MotorCommandMsg: snapshot del comando que el dispatcher acaba de despachar
# (post-safety-gate). Lo usa el dashboard para mostrar al operador qué
# manda al firmware en cada instante. NO es el path crítico — el path
# crítico es SpeedMotor + SteerMotor (ya existen arriba). Esto es solo
# telemetría.
class MotorCommandMsg(Enum):
    Queue = "General"
    Owner = "threadMotorCommandDispatcher"
    msgID = 1
    # value format: {"timestamp": float, "steering_deg": float,
    #                "speed_mps": float, "valid": bool, "source": str,
    #                "reason": str}
    msgType = "dict"
