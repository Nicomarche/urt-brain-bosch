# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC orginazers
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
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../../..")

import queue
import psutil
import json
import inspect
import eventlet
import os
import time
import base64

import config
from flask import Flask, request
from flask_socketio import SocketIO
from flask_cors import CORS
from enum import Enum

from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.templates.workerprocess import WorkerProcess
from src.core.messaging.allMessages import Localisation, Semaphores, StateChange
from src.statemachine.stateMachine import StateMachine
from src.statemachine.transitionTable import TransitionTable
from src.statemachine.systemMode import SystemMode
from src.utils.live_log import live_log
from src.dashboard.components.calibration import Calibration
from src.dashboard.components.ip_manger import IpManager

import src.core.messaging.allMessages as allMessages


class processDashboard(WorkerProcess):
    """This process handles the dashboard interactions, updating the UI based on the system's state.
    
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool): Enable debugging mode.
    """
    # ====================================== INIT ==========================================
    def __init__(self, queueList, logging, ready_event=None, debugging = False, stream_camera=True):

        self.running = True
        self.queueList = queueList
        self.logger = logging
        self.debugging = debugging
        self.stream_camera = stream_camera
        self.emit_legacy_camera = bool(
            getattr(config, "DASHBOARD_EMIT_LEGACY_CAMERA_BASE64", False)
        )
        
        # ip replacement
        IpManager.replace_ip_in_file()

        self.stateChangeSender = messageHandlerSender(queueList, StateChange)
        self._current_mode = SystemMode.DEFAULT

        self.messages = {}
        self.sendMessages = {}
        self.messagesAndVals = {}
        self._dashboard_trace_channels = {
            "ActuatorCommandStatus",
            "AliveSignal",
            "BatteryLvl",
            "CalibPWMData",
            "CalibRunDone",
            "CurrentDistance",
            "CurrentSpeed",
            "CurrentSteer",
            "EnableButton",
            "ImuAck",
            "ImuData",
            "InstantConsumption",
            "Klem",
            "ResourceMonitor",
            "SerialConnectionState",
            "SpeedMotor",
            "SteerMotor",
            "SteeringLimits",
            "ToggleHallSpeed",
            "OdoReset",
            "Brake",
        }

        self.memoryUsage = 0
        self.cpuCoreUsage = 0
        self.cpuTemperature = 0

        self.heartbeat_last_sent = time.time()
        self.heartbeat_retries = 0
        self.heartbeat_max_retries = 3
        self.heartbeat_time_between_heartbeats = 20
        self.heartbeat_time_between_retries = 5
        self.heartbeat_received = False

        self.sessionActive = False
        self.activeUser = None
        self.serialConnected = False
        self.table_state_file = self._get_table_state_path()

        # Flask, SocketIO, Calibration, and websocket handlers are NOT created here
        # because threading.Lock inside Flask/SocketIO is not picklable — spawn mode
        # serialises this object before sending it to the child process.
        # _init_flask_server() is called at the start of run() instead.
        self.app = None
        self.socketio = None
        self.calibration = None
        self.stateMachine = None

        self._initialize_messages()

        super(processDashboard, self).__init__(self.queueList, ready_event)
    

    def _init_flask_server(self):
        """Create Flask/SocketIO in the child process (post-spawn/fork).

        Flask and SocketIO contain threading.Lock objects that are not picklable,
        so they cannot be created in __init__ when using spawn mode.
        """
        IpManager.replace_ip_in_file()
        try:
            self.stateMachine = StateMachine.get_instance()
        except RuntimeError:
            # Spawn mode: StateMachine class state is not inherited by the child.
            # handle_driving_mode() already handles self.stateMachine being None.
            self.stateMachine = None
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='eventlet')
        CORS(self.app, supports_credentials=True)
        self.calibration = Calibration(self.queueList, self.socketio)
        self._setup_websocket_handlers()

    def _get_table_state_path(self):
        """Get the path for table state file."""
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(base_path, 'src', 'utils', 'table_state.json')
    

    def _initialize_messages(self):
        """Initialize message handling systems."""
        self.get_name_and_vals()
        ignored_channels = {
            "mainCamera",
            "Semaphores",
            # Internal high-volume/debug channels not consumed by the Angular dashboard.
            # Subscribing the dashboard to them can block the gateway and starve control
            # messages such as Klem/Speed/Steer when the pipe backs up.
            "LocalLanePerception",
            "LocalPerceptionStatus",
            "SignDetectionDebug",
            "SignDetectionStatus",
        }
        for channel in ignored_channels:
            self.messagesAndVals.pop(channel, None)
        if not self.stream_camera:
            self.messagesAndVals.pop("serialCamera", None)
        self.subscribe()
    

    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers."""
        self.socketio.on_event('message', self.handle_message)
        self.socketio.on_event('save', self.handle_save_table_state)
        self.socketio.on_event('load', self.handle_load_table_state)

    @staticmethod
    def _is_ego_location_payload(payload):
        if not isinstance(payload, dict):
            return False
        meta = payload.get("meta")
        if isinstance(meta, dict):
            source = str(meta.get("source") or "").strip().lower()
            if source.startswith("ego_pose"):
                return True
        # Backward compatibility: before the dashboard-specific tagging landed,
        # internal ego poses had x/y/yaw but no locsys/traffic `id`.
        return payload.get("id") is None and "x" in payload and "y" in payload

    @staticmethod
    def _location_payload_to_car(payload):
        if not isinstance(payload, dict):
            return None
        try:
            x = float(payload.get("x", payload.get("world_x", payload.get("posA"))))
            y = float(payload.get("y", payload.get("world_y", payload.get("posB"))))
        except (TypeError, ValueError):
            return None
        car = {"x": x, "y": y}
        if payload.get("id") is not None:
            car["id"] = payload.get("id")
        if payload.get("yaw") is not None:
            car["yaw"] = payload.get("yaw")
        return car

    @staticmethod
    def _is_manual_dashboard_localisation(payload):
        if not isinstance(payload, dict):
            return False
        meta = payload.get("meta")
        if isinstance(meta, dict):
            if bool(meta.get("manual")):
                return True
            source = str(meta.get("source") or "").strip().lower()
            if source.startswith("manual"):
                return True
        source = str(payload.get("source") or "").strip().lower()
        return source.startswith("manual")

    @staticmethod
    def _summarize_navigation_command(payload):
        if not isinstance(payload, dict):
            return {"type": "invalid"}
        summary = {
            "mode": str(payload.get("mode", "") or "").lower() or "direct_xy",
        }
        if payload.get("lanelet_id") is not None:
            summary["lanelet_id"] = str(payload.get("lanelet_id"))
        if payload.get("x") is not None and payload.get("y") is not None:
            try:
                summary["x"] = round(float(payload["x"]), 3)
                summary["y"] = round(float(payload["y"]), 3)
            except (TypeError, ValueError):
                pass
        destinations = list(payload.get("destinations", []) or [])
        if destinations:
            summary["destination_count"] = len(destinations)
            preview = []
            for item in destinations[:4]:
                if not isinstance(item, dict):
                    continue
                row = {}
                if item.get("lanelet_id") is not None:
                    row["lanelet_id"] = str(item.get("lanelet_id"))
                if item.get("x") is not None and item.get("y") is not None:
                    try:
                        row["x"] = round(float(item["x"]), 3)
                        row["y"] = round(float(item["y"]), 3)
                    except (TypeError, ValueError):
                        pass
                if row:
                    preview.append(row)
            summary["destinations_preview"] = preview
        return summary
    
    
    def _start_background_tasks(self):
        """Start background monitoring tasks inside the child process.

        Must be called from run() (after fork) using socketio.start_background_task()
        so the green threads belong to the correct eventlet hub.
        """
        psutil.cpu_percent(interval=1, percpu=False)  # warm up

        self.socketio.start_background_task(self.update_hardware_data)
        self.socketio.start_background_task(self.send_continuous_messages)
        self.socketio.start_background_task(self.send_hardware_data_to_frontend)
        self.socketio.start_background_task(self.send_heartbeat)
        self.socketio.start_background_task(self.stream_console_logs)

    def stream_console_logs(self):
        """Monitor the Log queue and emit messages to frontend."""
        log_queue = self.queueList.get("Log")
        if not log_queue:
            return

        while self.running:
            try:
                while not log_queue.empty():
                    msg = log_queue.get_nowait()
                    self.socketio.emit('console_log', {'data': msg})
                    eventlet.sleep(0)
                
                eventlet.sleep(0.5)  # 500ms — logs no requieren baja latencia
            except queue.Empty:
                eventlet.sleep(0.5)
            except Exception as e:
                if self.debugging:
                    self.logger.error(f"Error streaming logs: {e}")
                eventlet.sleep(1)


    # ===================================== STOP ==========================================
    def stop(self):
        """Stop the dashboard process."""
        super(processDashboard, self).stop()
        self.running = False


    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing method."""
        self._init_flask_server()

        # Register connect handler: enables the KL-switch button on the frontend as
        # soon as any client connects.
        @self.socketio.on('connect')
        def on_client_connect():
            self.socketio.emit('EnableButton', {'value': True})
            if self.debugging:
                self.logger.info("Client connected — EnableButton sent")

        @self.socketio.on('disconnect')
        def on_client_disconnect():
            socketId = request.sid
            if self.sessionActive and self.activeUser == socketId:
                print(
                    f"\033[1;97m[ Dashboard ] :\033[0m \033[1;93mWARNING\033[0m"
                    f" - Active dashboard peer disconnected \033[94m{socketId}\033[0m;"
                    " releasing session"
                )
                self._clear_active_session()

        # Start background tasks here (in the child process, after fork) so that
        # the eventlet green threads belong to the correct hub.
        self._start_background_tasks()

        if self.ready_event:
            self.ready_event.set()

        self.socketio.run(self.app, host='0.0.0.0', port=5005)


    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway."""
        for name, enum in self.messagesAndVals.items():
            if enum["owner"] != "Dashboard":
                subscriber = messageHandlerSubscriber(self.queueList, enum["enum"], "lastOnly", True)
                self.messages[name] = {"obj": subscriber}
                if name in self._dashboard_trace_channels:
                    enum_cls = enum["enum"]
                    print(
                        f"\033[1;97m[ Dashboard ] :\033[0m \033[1;96mSUB\033[0m"
                        f" channel={name} owner={enum_cls.Owner.value}"
                        f" msgID={enum_cls.msgID.value} queue={enum_cls.Queue.value}"
                        f" mode=lastOnly"
                    )
            else:
                sender = messageHandlerSender(self.queueList, enum["enum"])
                self.sendMessages[str(name)] = {"obj": sender}
                if name in self._dashboard_trace_channels:
                    enum_cls = enum["enum"]
                    print(
                        f"\033[1;97m[ Dashboard ] :\033[0m \033[1;96mSEND-REG\033[0m"
                        f" channel={name} owner={enum_cls.Owner.value}"
                        f" msgID={enum_cls.msgID.value} queue={enum_cls.Queue.value}"
                    )

        subscriber = messageHandlerSubscriber(self.queueList, Semaphores, "fifo", True)
        self.messages["Semaphores"] = {"obj": subscriber}

        # Receive GPS fixes published by threadLocSys on the Localisation channel.
        # We add a dedicated subscriber (separate from the sender already created
        # above) so the dashboard can forward them to the frontend as "GpsFix".
        # Filtering by meta.source happens in send_continuous_messages().
        self._gps_fix_subscriber = messageHandlerSubscriber(
            self.queueList, Localisation, "lastOnly", True
        )
        print(
            f"[GPS-DBG] processDashboard.subscribe: _gps_fix_subscriber registered "
            f"(Owner={Localisation.Owner.value} msgID={Localisation.msgID.value})",
            flush=True,
        )


    def get_name_and_vals(self):
        """Extract all message names and values for processing."""
        classes = inspect.getmembers(allMessages, inspect.isclass)
        for name, cls in classes:
            if name != "Enum" and issubclass(cls, Enum):
                self.messagesAndVals[name] = {"enum": cls, "owner": cls.Owner.value} # type: ignore


    def send_message_to_brain(self, dataName, dataDict):
        """Send messages to the backend."""
        if dataName in self.sendMessages:
            value = dataDict.get("Value")
            self.sendMessages[dataName]["obj"].send(value)
            if dataName in self._dashboard_trace_channels:
                enum_cls = self.messagesAndVals[dataName]["enum"]
                queue_name = enum_cls.Queue.value
                print(
                    f"\033[1;97m[ Dashboard ] :\033[0m \033[1;96mQUEUE\033[0m"
                    f" frontend -> brain channel={dataName}"
                    f" queue={queue_name} owner={enum_cls.Owner.value}"
                    f" msgID={enum_cls.msgID.value}"
                    f" qsize={self._queue_depth(queue_name)}"
                    f" value={self._preview(value)}"
                )
        elif dataName in self._dashboard_trace_channels:
            print(
                f"\033[1;97m[ Dashboard ] :\033[0m \033[1;93mNO-SENDER\033[0m"
                f" channel={dataName} payload={self._preview(dataDict)}"
            )


    def handle_message(self, data):
        """Handle incoming WebSocket messages."""
        if self.debugging:
            self.logger.info("Received message: " + str(data))

        socketId = None
        try:
            dataDict = json.loads(data)
            dataName = dataDict["Name"]
            socketId = request.sid

            if dataName == "SessionAccess":
                self.handle_single_user_session(socketId)
            elif self.sessionActive and self.activeUser != socketId:
                print(f"\033[1;97m[ Dashboard ] :\033[0m \033[1;93mWARNING\033[0m - Message received from unauthorized user \033[94m{socketId}\033[0m")
                return

            if dataName == "Heartbeat":
                self.handle_heartbeat()
            elif dataName == "SessionEnd":
                self.handle_session_end(socketId)
            elif dataName == "DrivingMode":
                self.handle_driving_mode(dataDict)
            elif dataName == "Calibration":
                self.handle_calibration(dataDict, socketId)
            elif dataName == "GetCurrentSerialConnectionState":
                self.handle_get_current_serial_connection_state(socketId)
            else:
                if dataName == "Klem":
                    print(
                        f"\033[1;97m[ Dashboard ] :\033[0m \033[1;92mINFO\033[0m"
                        f" - Received Klem \033[94m{dataDict.get('Value')}\033[0m from \033[94m{socketId}\033[0m"
                    )
                # Log explícito del carril manual (Speed/Steer/Brake) para
                # que el operador vea el message llegando al brain. Si el
                # GUI emitió `[ DrivingModel ] EMIT CMD_SPEED` pero ESTA
                # línea no aparece, el message se perdió en SocketIO o el
                # `sessionActive`/`activeUser` lo rechazó arriba.
                if dataName in ("SpeedMotor", "SteerMotor", "Brake"):
                    is_registered = dataName in self.sendMessages
                    print(
                        f"\033[1;97m[ Dashboard ] :\033[0m \033[1;96mRECV\033[0m"
                        f" {dataName}={dataDict.get('Value')!r}"
                        f" registered={is_registered}"
                    )
                if (
                    dataName == "Localisation"
                    and self._is_manual_dashboard_localisation(dataDict.get("Value"))
                    and self._current_mode != SystemMode.STOP
                ):
                    print(
                        f"\033[1;97m[ Dashboard ] :\033[0m \033[1;93mWARNING\033[0m"
                        " - Manual localisation ignored because mode is not STOP"
                    )
                    if socketId:
                        self.socketio.emit(
                            'response',
                            {'error': 'Manual localisation is only allowed in STOP mode'},
                            room=socketId,
                        )
                    return
                if dataName == "NavigationCommand":
                    live_log(
                        "dashboard", event="navigation_command_received",
                        command=self._summarize_navigation_command(dataDict.get("Value")),
                        dashboard_mode=getattr(self._current_mode, "name", str(self._current_mode)),
                        socket_id=socketId,
                        session_active=bool(self.sessionActive),
                    )
                self.send_message_to_brain(dataName, dataDict)
                # When the kl-switch component loads it sends Klem="0" before it
                # subscribes to EnableButton — reply immediately so the toggle
                # unlocks as soon as the subscription is set up.
                if dataName == "Klem":
                    self.socketio.emit('EnableButton', {'value': True}, room=socketId)

            self.socketio.emit('response', {'data': 'Message received: ' + str(data)}, room=socketId) # type: ignore
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON message: {e}")
            if socketId:
                self.socketio.emit('response', {'error': 'Invalid JSON format'}, room=socketId) # type: ignore
        except Exception as e:
            import traceback
            print(
                f"\033[1;97m[ Dashboard ] :\033[0m \033[1;91mERROR\033[0m"
                f" - Unhandled exception in handle_message: {e}\n{traceback.format_exc()}"
            )
            if socketId:
                self.socketio.emit('response', {'error': str(e)}, room=socketId) # type: ignore


    def handle_heartbeat(self):
        """Handle heartbeat message."""
        self.heartbeat_retries = 0
        self.heartbeat_last_sent = time.time()
        self.heartbeat_received = True


    def _clear_active_session(self):
        """Release the single-user lock and reset heartbeat state."""
        self.sessionActive = False
        self.activeUser = None
        self.heartbeat_retries = 0
        self.heartbeat_received = False


    def _is_socket_connected(self, socketId):
        """Best-effort check for stale Socket.IO session ids."""
        if not socketId or self.socketio is None:
            return False
        try:
            manager = self.socketio.server.manager
            return bool(manager.is_connected(socketId, namespace="/"))
        except Exception:
            # Flask-SocketIO manager internals vary by version. If the check is
            # unavailable, keep the existing session policy and let heartbeat
            # timeout clear the lock.
            return True


    def handle_driving_mode(self, dataDict):
        """Handle driving mode change.

        Uses a direct StateChange sender as primary path to avoid the
        multiprocessing.Manager proxy that breaks under eventlet monkey-patching.
        Falls back to the Manager-based StateMachine only if the direct path fails.
        """
        requested_value = dataDict.get('Value', '').lower()
        action = f"dashboard_{requested_value}_button"

        # Validate transition locally (no Manager proxy needed)
        transition = TransitionTable.get_next_mode(self._current_mode, action)
        if not transition['transition_valid']:
            print(
                f"\033[1;97m[ Dashboard ] :\033[0m \033[1;93mWARNING\033[0m"
                f" - Invalid transition from \033[1;94m{self._current_mode.name}\033[0m"
                f" with action \033[1;94m{action}\033[0m"
            )
            return

        next_mode = transition['next_mode']

        # Self-transition: nothing to do
        if next_mode == self._current_mode:
            return

        self._current_mode = next_mode
        mode_name = next_mode.name

        # Send via direct queue (works even when Manager proxy is broken)
        try:
            self.stateChangeSender.send(mode_name)
            live_log(
                "dashboard", event="state_change_request",
                from_mode=getattr(self._current_mode, "name", str(self._current_mode)),
                to_mode=mode_name,
                source="dashboard_handle_driving_mode",
            )
            print(
                f"\033[1;97m[ Dashboard ] :\033[0m \033[1;92mINFO\033[0m"
                f" - Mode \033[1;94m{mode_name}\033[0m sent to brain"
            )
            # Also update the Manager-based shared state so other processes stay in sync
            try:
                self.stateMachine._shared_state['mode'] = next_mode
            except Exception:
                pass  # Manager proxy optional; local state is the source of truth
        except Exception as e:
            print(
                f"\033[1;97m[ Dashboard ] :\033[0m \033[1;91mERROR\033[0m"
                f" - Failed to send StateChange: {e}"
            )


    def handle_calibration(self, dataDict, socketId):
        """Handle calibration signals from frontend."""
        self.calibration.handle_calibration_signal(dataDict, socketId)


    def handle_get_current_serial_connection_state(self, socketId):
        """Handle getting the current serial connection state."""
        self.socketio.emit('current_serial_connection_state', {'data': self.serialConnected}, room=socketId)


    def handle_single_user_session(self, socketId):
        """Handle session access for a single user."""
        if (
            self.sessionActive
            and self.activeUser != socketId
            and not self._is_socket_connected(self.activeUser)
        ):
            print(
                f"\033[1;97m[ Dashboard ] :\033[0m \033[1;93mWARNING\033[0m"
                f" - Releasing stale dashboard session \033[94m{self.activeUser}\033[0m"
            )
            self._clear_active_session()

        if not self.sessionActive:
            self.sessionActive = True
            self.activeUser = socketId
            self.heartbeat_retries = 0
            self.heartbeat_received = True
            print(f"\033[1;97m[ Dashboard ] :\033[0m \033[1;92mINFO\033[0m - Session access granted to \033[94m{socketId}\033[0m")
            self.socketio.emit('session_access', {'data': True}, room=socketId)
            self.send_message_to_brain("RequestSteerLimits", {"Value": True})
        elif self.activeUser == socketId:
            self.socketio.emit('session_access', {'data': True}, room=socketId)
            self.send_message_to_brain("RequestSteerLimits", {"Value": True})
        else:
            print(f"\033[1;97m[ Dashboard ] :\033[0m \033[1;92mINFO\033[0m - Session access denied to \033[94m{socketId}\033[0m")
            self.socketio.emit('session_access', {'data': False}, room=socketId)


    def handle_session_end(self, socketId):
        """Handle session end for the single user."""
        if self.sessionActive and self.activeUser == socketId:
            self._clear_active_session()


    def handle_save_table_state(self, data):
        """Handle saving the table state to a JSON file."""
        if self.debugging:
            self.logger.info("Received save message: " + data)

        try:
            dataDict = json.loads(data)
            os.makedirs(os.path.dirname(self.table_state_file), exist_ok=True)
            
            with open(self.table_state_file, 'w') as json_file:
                json.dump(dataDict, json_file, indent=4)
                
            self.socketio.emit('response', {'data': 'Table state saved successfully'})
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON for save: {e}")
            self.socketio.emit('response', {'error': 'Invalid JSON format'})
        except OSError as e:
            self.logger.error(f"Failed to save table state: {e}")
            self.socketio.emit('response', {'error': 'Failed to save table state'})


    def handle_load_table_state(self, data):
        """Handle loading the table state from a JSON file."""
        try:
            with open(self.table_state_file, 'r') as json_file:
                dataDict = json.load(json_file)
            self.socketio.emit('loadBack', {'data': dataDict})
        except FileNotFoundError:
            self.socketio.emit('response', {'error': 'File not found. Please save the table state first.'})
        except json.JSONDecodeError:
            self.socketio.emit('response', {'error': 'Failed to parse JSON data from the file.'})
        except OSError as e:
            self.logger.error(f"Failed to load table state: {e}")
            self.socketio.emit('response', {'error': 'Failed to load table state'})

    def _make_json_safe(self, value):
        """Convert nested values to Socket.IO/JSON-safe primitives."""
        if isinstance(value, dict):
            return {str(key): self._make_json_safe(item) for key, item in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._make_json_safe(item) for item in value]

        if hasattr(value, "tolist"):
            try:
                return self._make_json_safe(value.tolist())
            except Exception:
                pass

        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return str(value)

    def _preview(self, value, limit=180):
        text = repr(value)
        if len(text) > limit:
            return text[:limit] + "..."
        return text

    def _queue_depth(self, queue_name):
        try:
            return self.queueList[queue_name].qsize()
        except Exception:
            return "n/a"


    def update_hardware_data(self):
        """Monitor and update hardware metrics periodically."""
        self.cpuCoreUsage = psutil.cpu_percent(interval=None, percpu=False)
        self.memoryUsage = psutil.virtual_memory().percent
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                first_sensor = next(iter(temps.values()))
                if first_sensor and first_sensor[0].current is not None:
                    self.cpuTemperature = round(first_sensor[0].current)
        except Exception:
            self.cpuTemperature = 0

        eventlet.spawn_after(3, self.update_hardware_data)  # 3s — datos de hardware no cambian rapido


    def send_heartbeat(self):
        """Send a heartbeat message to the frontend."""
        if not self.running:
            return

        if not self.heartbeat_received and self.sessionActive:
            active_user = self.activeUser
            self.heartbeat_retries += 1
            if self.heartbeat_retries < self.heartbeat_max_retries:
                self.socketio.emit('heartbeat', {'data': 'Heartbeat'}, room=active_user)
            else:
                print(f"\033[1;97m[ Dashboard ] :\033[0m \033[1;93mWARNING\033[0m - Connection lost with peer \033[94m{active_user}\033[0m")
                self.socketio.emit('heartbeat_disconnect', {'data': 'Heartbeat timeout'}, room=active_user)
                self._clear_active_session()

            eventlet.spawn_after(self.heartbeat_time_between_retries, self.send_heartbeat)
        else:
            self.heartbeat_received = False
            eventlet.spawn_after(self.heartbeat_time_between_heartbeats, self.send_heartbeat)


    def send_continuous_messages(self):
        """Process and send subscriber messages to the frontend."""
        if not self.running:
            return

        for msg, subscriber in self.messages.items():
            try:
                resp = subscriber["obj"].receive()
                if resp is None:
                    continue

                if msg in self._dashboard_trace_channels:
                    print(
                        f"\033[1;97m[ Dashboard ] :\033[0m \033[1;96mIPC\033[0m"
                        f" channel={msg} received type={type(resp).__name__}"
                        f" value={self._preview(resp)}"
                    )

                if msg == "SerialConnectionState":
                    self.serialConnected = resp
                elif msg == "StateChange":
                    # Keep local mode tracker in sync with whatever the brain reports
                    try:
                        self._current_mode = SystemMode[resp]
                    except (KeyError, Exception):
                        pass

                # ── Camera fast-path ─────────────────────────────────────────
                # The perception thread sends JPEG → base64 through the local
                # IPC queue. The PyQt GUI consumes raw JPEG bytes via
                # `serialCamera_bin`, so only emit the binary frame to the
                # active dashboard session. Sending the legacy base64 event in
                # parallel roughly doubles monitor CPU/bandwidth and can back
                # up the SocketIO/Qt queues during remote operation.
                if msg == "serialCamera" and isinstance(resp, str):
                    active_room = self.activeUser if self.sessionActive else None
                    if not active_room:
                        continue
                    if not self._is_socket_connected(active_room):
                        self._clear_active_session()
                        continue

                    emitted_binary = False
                    try:
                        jpg_bytes = base64.b64decode(resp)
                        self.socketio.emit(
                            "serialCamera_bin", jpg_bytes, room=active_room
                        )
                        emitted_binary = True
                    except Exception as exc:
                        # If decoding fails, fall through and at least emit
                        # the legacy text frame so something still shows.
                        if self.debugging:
                            self.logger.warning(
                                f"serialCamera_bin decode failed: {exc}")
                    if self.emit_legacy_camera or not emitted_binary:
                        self.socketio.emit(msg, {"value": resp}, room=active_room)
                    continue

                if msg == "Location" and isinstance(resp, dict):
                    safe_resp = self._make_json_safe(resp)
                    if self._is_ego_location_payload(safe_resp):
                        self.socketio.emit(msg, {"value": safe_resp})
                    else:
                        car_payload = self._location_payload_to_car(safe_resp)
                        if car_payload is not None:
                            self.socketio.emit("Cars", {"value": [car_payload]})
                    continue

                safe_resp = self._make_json_safe(resp)
                self.socketio.emit(msg, {"value": safe_resp})
                if msg in self._dashboard_trace_channels:
                    print(
                        f"\033[1;97m[ Dashboard ] :\033[0m \033[1;96mEMIT\033[0m"
                        f" socketio channel={msg} value={self._preview(safe_resp)}"
                    )
                if self.debugging:
                    self.logger.info(f"{msg}: {safe_resp}")
            except Exception as e:
                print(
                    f"\033[1;97m[ Dashboard ] :\033[0m \033[1;91mERROR\033[0m"
                    f" - Failed to forward \033[94m{msg}\033[0m ({e})"
                )

        # Forward GPS fixes from threadLocSys to the frontend as "GpsFix".
        # We read the dedicated subscriber (separate from the Localisation sender)
        # and only emit when the message comes from the GPS hardware.
        try:
            gps_payload = self._gps_fix_subscriber.receive()
            if gps_payload is not None:
                meta_src = (
                    gps_payload.get("meta", {}).get("source")
                    if isinstance(gps_payload, dict)
                    else None
                )
                print(
                    f"[GPS-DBG] processDashboard.send_continuous_messages: "
                    f"_gps_fix_subscriber.receive() got payload "
                    f"type={type(gps_payload).__name__} meta.source={meta_src!r}",
                    flush=True,
                )
                if (
                    isinstance(gps_payload, dict)
                    and meta_src == "gps_localisation"
                ):
                    self.socketio.emit("GpsFix", {"value": gps_payload})
                    print(
                        f"[GPS-DBG] processDashboard.send_continuous_messages: "
                        f"EMIT GpsFix world_x={gps_payload.get('world_x')} "
                        f"world_y={gps_payload.get('world_y')}",
                        flush=True,
                    )
                else:
                    print(
                        f"[GPS-DBG] processDashboard.send_continuous_messages: "
                        f"payload IGNORED (no es gps_localisation)",
                        flush=True,
                    )
        except Exception as exc:
            print(
                f"[GPS-DBG] processDashboard.send_continuous_messages: "
                f"exception {type(exc).__name__}: {exc}",
                flush=True,
            )

        eventlet.spawn_after(0.15, self.send_continuous_messages)  # 150ms (~7Hz) — balance entre CPU y fluidez del stream


    def send_hardware_data_to_frontend(self):
        """Send hardware monitoring data to the frontend."""
        if not self.running:
            return

        self.socketio.emit('memory_channel', {'data': self.memoryUsage})
        self.socketio.emit('cpu_channel', {
            'data': {
                'usage': self.cpuCoreUsage,
                'temp': self.cpuTemperature
            }
        })
        # Re-broadcast EnableButton every second so every client gets it even if
        # the initial connect-event emit arrived before the Angular component
        # subscribed its observable.
        self.socketio.emit('EnableButton', {'value': True})

        eventlet.spawn_after(1.0, self.send_hardware_data_to_frontend)
