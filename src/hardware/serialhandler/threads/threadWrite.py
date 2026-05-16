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

import json
import threading
import time
from datetime import datetime, timedelta

from src.hardware.serialhandler.threads.messageconverter import MessageConverter
from src.templates.threadwithstop import ThreadWithStop
from src.core.messaging.allMessages import (
    ActuatorCommandStatus,
    Klem,
    Control,
    SteerMotor,
    SpeedMotor,
    Brake,
    ToggleBatteryLvl,
    ToggleHallSpeed,
    ToggleImuData,
    ToggleInstant,
    ToggleResourceMonitor,
    SerialConnectionState,
    ControlCalib,
    IsAlive,
    RequestSteerLimits,
    OdoReset,
    SimRelocalize,
)
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.utils.live_log import live_log


class threadWrite(ThreadWithStop):
    """This thread write the data that Raspberry PI send to NUCLEO.\n

    Args:
        queues (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        process (processSerialHandler): ProcessSerialHandler object.
        logFile (FileHandler): The path to the history file where you can find the logs from the connection.
        example (bool, optional): Flag for exmaple activation. Defaults to False.
    """

    # ===================================== INIT =========================================
    def __init__(self, process, logFile, queues, logger, debugger = False, example=False):
        super(threadWrite, self).__init__(pause=0.01)  # 100Hz — suficiente para comandos de motor a ~5 FPS
        self.process = process
        self.queuesList = queues
        self.logFile = logFile
        self.exampleFlag = example
        self.logger = logger
        self.debugger = debugger

        self.running = False
        self.engineEnabled = False
        self.currentKlemMode = 0
        self.messageConverter = MessageConverter()
        self.steerMotorSender = messageHandlerSender(self.queuesList, SteerMotor)
        self.speedMotorSender = messageHandlerSender(self.queuesList, SpeedMotor)
        self.configPath = "src/utils/table_state.json"
        self.motionCommandTime = 0
        self.last_speed_cmd = None
        self.last_steer_cmd = None
        self.last_speed_ts = 0.0
        self.last_steer_ts = 0.0
        self.last_motion_command = None
        self.last_motion_command_ts = 0.0
        self.last_motion_speed_x10 = None
        self.last_motion_steer_x10 = None
        self.last_motion_raw_command = None
        self.last_blocked_reason = None
        self._last_status_snapshot = None
        self._last_status_send_time = 0.0
        self._last_continuous_motion_send_monotonic = 0.0
        self._last_forced_feedback_enable = 0.0
        self._auto_kl_run_in_sim = False
        # En ZMQ sim el bridge espera comandos refrescados con cierta
        # frecuencia; si dejamos de reenviar porque speed/steer quedaron
        # constantes, el auto termina frenándose aunque el planner siga en
        # AUTO. Reemitimos el último par válido a ~20 Hz mientras el motor
        # esté habilitado.
        self._continuous_motion_keepalive_s = 0.05

        # error rate limiting
        self.last_error_time = None
        self.error_cooldown = timedelta(seconds=3)

        self._init_senders()
        self._init_subscribers()
        self._init_motor_output()
        self.load_config("init")

        if example:
            self.i = 0.0
            self.j = -1.0
            self.s = 0.0
            self.example()

    def _init_subscribers(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""
        self.klSubscriber = messageHandlerSubscriber(self.queuesList, Klem, "lastOnly", True)
        self.controlSubscriber = messageHandlerSubscriber(self.queuesList, Control, "lastOnly", True)
        self.steerMotorSubscriber = messageHandlerSubscriber(self.queuesList, SteerMotor, "lastOnly", True)
        self.speedMotorSubscriber = messageHandlerSubscriber(self.queuesList, SpeedMotor, "lastOnly", True)
        self.brakeSubscriber = messageHandlerSubscriber(self.queuesList, Brake, "lastOnly", True)
        self.instantSubscriber = messageHandlerSubscriber(self.queuesList, ToggleInstant, "lastOnly", True)
        self.batterySubscriber = messageHandlerSubscriber(self.queuesList, ToggleBatteryLvl, "lastOnly", True)
        self.resourceMonitorSubscriber = messageHandlerSubscriber(self.queuesList, ToggleResourceMonitor, "lastOnly", True)
        self.imuSubscriber = messageHandlerSubscriber(self.queuesList, ToggleImuData, "lastOnly", True)
        self.hallSpeedSubscriber = messageHandlerSubscriber(self.queuesList, ToggleHallSpeed, "lastOnly", True)
        self.odoResetSubscriber = messageHandlerSubscriber(self.queuesList, OdoReset, "lastOnly", True)
        self.controlCalibSubscriber = messageHandlerSubscriber(self.queuesList, ControlCalib, "lastOnly", True)
        self.isAliveSubscriber = messageHandlerSubscriber(self.queuesList, IsAlive, "lastOnly", True)
        self.requestSteerLimitsSubscriber = messageHandlerSubscriber(self.queuesList, RequestSteerLimits, "lastOnly", True)
        self.simRelocalizeSubscriber = messageHandlerSubscriber(self.queuesList, SimRelocalize, "lastOnly", True)
        
    def _init_senders(self):
        self.serialConnectionStateSender = messageHandlerSender(self.queuesList, SerialConnectionState)
        self.actuatorStatusSender = messageHandlerSender(self.queuesList, ActuatorCommandStatus)

    def _init_motor_output(self):
        """Pick motor backend based on config.MOTOR_OUTPUT ('serial' | 'zmq').
        ZMQ mode binds a PUB socket and short-circuits send_to_serial(); the
        Nucleo/serial path is never touched and the sim_bridge subscribes."""
        try:
            from config import MOTOR_OUTPUT
        except ImportError:
            MOTOR_OUTPUT = "serial"
        self.motor_output = MOTOR_OUTPUT
        self._zmq_sock = None
        if self.motor_output != "zmq":
            return

        try:
            import zmq
            from config import ZMQ_MOTOR_ENDPOINT, ZMQ_MOTOR_TOPIC
        except ImportError as e:
            print(f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;91mERROR\033[0m"
                  f" - MOTOR_OUTPUT='zmq' but pyzmq missing ({e}). Falling back to serial.")
            self.motor_output = "serial"
            return

        try:
            self._zmq_motor_topic = ZMQ_MOTOR_TOPIC
            self._zmq_ctx = zmq.Context.instance()
            self._zmq_sock = self._zmq_ctx.socket(zmq.PUB)
            # The bridge binds the cmd SUB socket (it's the long-running endpoint).
            # The brain — which is the transient publisher — connects to it. With both
            # binding, neither receives anything because PUB↔SUB needs one bind + one
            # connect to form the pair.
            self._zmq_sock.connect(ZMQ_MOTOR_ENDPOINT)
            time.sleep(0.2)  # give the connection a chance to attach before first send
            print(f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;92mINFO\033[0m"
                  f" - Motor output → ZMQ {ZMQ_MOTOR_ENDPOINT} (topic={ZMQ_MOTOR_TOPIC!r})")
        except Exception as e:
            print(f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;91mERROR\033[0m"
                  f" - Failed to connect ZMQ motor socket: {e}")
            self.motor_output = "serial"
            self._zmq_sock = None
            return

        # ── Auto-engage KL en sim ──────────────────────────────────────────
        # En el auto físico, KL es la llave de contacto: KL=0 (off) /
        # KL=15 (electrónica on) / KL=30 (motor encendido). El brain bloquea
        # SpeedMotor/SteerMotor hasta que el operador la lleva a 30 desde
        # el dashboard — es la red de seguridad que evita arranques
        # accidentales con gente cerca del auto.
        #
        # En sim no hay actuador físico que proteger — la "llave" es solo
        # un slider en una UI. Forzarla cada vez que arrancás `run_sim.sh`
        # es fricción sin beneficio. Activable/desactivable por config para
        # quien quiera testear la máquina de estados de KL en sim.
        try:
            from config import AUTO_KL_RUN_IN_SIM
        except ImportError:
            AUTO_KL_RUN_IN_SIM = True  # default: convenience-on en sim
        self._auto_kl_run_in_sim = bool(AUTO_KL_RUN_IN_SIM and self.motor_output == "zmq")
        if self._auto_kl_run_in_sim:
            self.running = True
            self.engineEnabled = True
            self.currentKlemMode = 30
            self.last_blocked_reason = None
            print(
                "\033[1;97m[ Serial Handler ] :\033[0m "
                "\033[1;92mINFO\033[0m - SIM auto-engaged "
                "\033[94mKL=30\033[0m (engineEnabled=True). Override desde "
                "el dashboard si necesitás testear el state machine de KL."
            )

    # ==================================== SENDING =======================================

    def send_to_serial(self, msg):
        action = msg.get("action")
        if self.motor_output == "zmq" and self._zmq_sock is not None:
            try:
                payload = json.dumps(msg).encode("utf-8")
                self._zmq_sock.send_multipart([self._zmq_motor_topic, payload])
                self.last_blocked_reason = None
                live_log(
                    "zmq_motor", event="motor_cmd_sent",
                    action=str(action),
                    speed=msg.get("speed"),
                    steer=msg.get("steer") if "steer" in msg else msg.get("steerAngle"),
                    cmd_time=msg.get("time"),
                )
                return True, json.dumps(msg)
            except Exception as e:
                if self._should_send_error():
                    print(f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;91mERROR\033[0m"
                          f" - ZMQ send failed for {action}: {e}")
                self.last_blocked_reason = "zmq_send_failed"
                return False, None
        command_msg = self.messageConverter.get_command(**msg)
        if command_msg != "error":
            try:
                with self.process.serialLock:
                    serialCon = self.process.serialCon
                    if serialCon and self.process.serialConnected and serialCon.is_open:
                        payload = command_msg.encode("ascii")
                        written = serialCon.write(payload)
                        serialCon.flush()
                        if written != len(payload):
                            self.last_blocked_reason = "serial_short_write"
                            print(
                                f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;91mERROR\033[0m"
                                f" - Short serial write for \033[94m{action.upper() if action else 'UNKNOWN'}\033[0m:"
                                f" wrote {written}/{len(payload)} bytes ({command_msg.strip()})"
                            )
                            self._publish_actuator_status(force=True)
                            return False, command_msg
                        self.logFile.write(command_msg)
                        self._log_serial_tx(action, command_msg, written, serialCon)
                        if action == "kl":
                            print(
                                f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;92mINFO\033[0m"
                                f" - Sent KL command \033[94m{command_msg.strip()}\033[0m"
                            )
                        return True, command_msg

                if action == "kl" or self._should_send_error():
                    print(
                        f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;93mWARNING\033[0m"
                        f" - Cannot send \033[94m{action.upper() if action else 'UNKNOWN'}\033[0m"
                        f" ({command_msg.strip()}) because serial is disconnected"
                    )
                self.last_blocked_reason = "serial_disconnected"
                self._publish_actuator_status(force=True)

            except Exception as e:
                if action == "kl" or self._should_send_error():
                    self.serialConnectionStateSender.send(False)
                    print(
                        f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;91mERROR\033[0m"
                        f" - Failed to write \033[94m{action.upper() if action else 'UNKNOWN'}\033[0m to serial ({e})"
                    )
                self.last_blocked_reason = "serial_disconnected"
                self._publish_actuator_status(force=True)
                return False, command_msg
        else:
            self.last_blocked_reason = "invalid_command"
            self._publish_actuator_status(force=True)
            return False, None
        return False, command_msg

    def _publish_actuator_status(self, force=False):
        now = time.time()
        snapshot = {
            "serial_connected": bool(self.process.serialConnected),
            "klem_mode": int(self.currentKlemMode),
            "engine_enabled": bool(self.engineEnabled),
            "command_type": self.last_motion_command,
            "speed_x10": self.last_motion_speed_x10,
            "steer_x10": self.last_motion_steer_x10,
            "last_serial_command": self.last_motion_raw_command,
            "raw_command": self.last_motion_raw_command,
            "blocked_reason": self.last_blocked_reason,
            "last_serial_command_ts": self.last_motion_command_ts or None,
        }
        should_send = (
            force
            or snapshot != self._last_status_snapshot
            or (now - self._last_status_send_time) >= 0.5
        )
        if not should_send:
            return

        payload = dict(snapshot)
        payload["timestamp"] = now
        self.actuatorStatusSender.send(payload)
        print(
            f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;96mQUEUE\033[0m"
            f" threadWrite -> ActuatorCommandStatus queue=General"
            f" qsize={self._queue_depth('General')}"
            f" payload={self._preview(payload)}"
        )
        self._last_status_snapshot = snapshot
        self._last_status_send_time = now

    def _record_motion_command(self, command_type, raw_command, speed_x10=None, steer_x10=None,
                               blocked_reason=None, force=False):
        self.last_motion_command = command_type
        self.last_motion_command_ts = time.time()
        self.last_motion_speed_x10 = speed_x10
        self.last_motion_steer_x10 = steer_x10
        self.last_motion_raw_command = raw_command
        self.last_blocked_reason = blocked_reason
        self._publish_actuator_status(force=force)

    def _log_serial_tx(self, action, command_msg, written, serial_con):
        """Print every successful wire write so RX silence can be diagnosed."""
        try:
            waiting = serial_con.in_waiting
        except Exception as exc:
            waiting = f"err:{exc}"
        print(
            f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;96mTX\033[0m"
            f" action={action!r} raw={self._preview(command_msg.strip())}"
            f" bytes={written} device={getattr(self.process, 'serialDevice', None)!r}"
            f" connected={getattr(self.process, 'serialConnected', None)}"
            f" open={getattr(serial_con, 'is_open', None)} rx_waiting_after_write={waiting}"
        )

    def _force_feedback_streams(self, source):
        """Make Nucleo feedback explicit after KL changes.

        The current firmware fork should boot with hallspeed active, but sending
        these toggles removes ambiguity when the dashboard shows RX-IDLE.
        """
        now = time.monotonic()
        if now - self._last_forced_feedback_enable < 0.5:
            return
        self._last_forced_feedback_enable = now
        commands = (
            {"action": "hallspeed", "activate": 1},
            {"action": "imu", "activate": 1},
        )
        for command in commands:
            sent, raw_command = self.send_to_serial(command)
            print(
                f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;96mFEEDBACK-ENABLE\033[0m"
                f" source={source} action={command['action']!r} sent={sent}"
                f" raw={self._preview(raw_command.strip() if raw_command else raw_command)}"
            )
            time.sleep(0.05)

    def _handle_vcd_command(self, speed_x10, steer_x10):
        command = {
            "action": "vcd",
            "speed": int(speed_x10),
            "steer": int(steer_x10),
            "time": int(self.motionCommandTime),
        }
        sent, raw_command = self.send_to_serial(command)
        blocked_reason = None if sent else "serial_disconnected"
        self._record_motion_command(
            "vcd",
            raw_command,
            speed_x10=int(speed_x10),
            steer_x10=int(steer_x10),
            blocked_reason=blocked_reason,
            force=True,
        )
        if sent and self.debugger:
            self.logger.info(f"VCD sent: speed={int(speed_x10)} steer={int(steer_x10)}")
        return sent

    def _handle_continuous_motion_command(self, speed_x10, steer_x10):
        speed_command = {
            "action": "speed",
            "speed": int(speed_x10),
        }
        steer_command = {
            "action": "steer",
            "steerAngle": int(steer_x10),
        }

        speed_sent, speed_raw_command = self.send_to_serial(speed_command)
        steer_sent, steer_raw_command = self.send_to_serial(steer_command)

        raw_parts = []
        if speed_raw_command:
            raw_parts.append(speed_raw_command.strip())
        if steer_raw_command:
            raw_parts.append(steer_raw_command.strip())
        raw_command = " | ".join(raw_parts) if raw_parts else None

        blocked_reason = None if (speed_sent and steer_sent) else "serial_disconnected"
        self._record_motion_command(
            "speed_steer",
            raw_command,
            speed_x10=int(speed_x10),
            steer_x10=int(steer_x10),
            blocked_reason=blocked_reason,
            force=True,
        )

        if speed_sent and steer_sent and self.debugger:
            self.logger.info(
                f"Continuous command sent: speed={int(speed_x10)} steer={int(steer_x10)}"
            )
        if speed_sent and steer_sent:
            self._last_continuous_motion_send_monotonic = time.monotonic()

        # Visibilidad sin --debugger: imprimimos solo cuando el par
        # (speed, steer) cambia respecto al último despachado. Así
        # el operador VE el carril manual flowing (CMD_SPEED del dashboard
        # → IPC → este punto → firmware/ZMQ) sin floodear stdout durante
        # el ramp de steer (50 ms ticks). Esto NO confunde con el log del
        # `[ Dispatcher ]`, que muestra el output de MotionController y
        # NUNCA escribe motors en MANUAL.
        pair = (int(speed_x10), int(steer_x10))
        if pair != getattr(self, "_last_motion_log_pair", None):
            tag = "\033[1;92mSENT\033[0m" if (speed_sent and steer_sent) else "\033[1;91mBLOCK\033[0m"
            raw_note = f" raw={raw_command}" if raw_command else ""
            print(
                f"\033[1;97m[ Serial Handler ] :\033[0m {tag} "
                f"speed={pair[0]} steer={pair[1]} "
                f"(KL={self.currentKlemMode}, engineEnabled={self.engineEnabled})"
                f"{raw_note}"
            )
            self._last_motion_log_pair = pair

        return speed_sent and steer_sent

    def _handle_brake_command(self, brake_value):
        command = {"action": "brake", "steerAngle": int(brake_value)}
        sent, raw_command = self.send_to_serial(command)
        blocked_reason = None if sent else "serial_disconnected"
        self._record_motion_command(
            "brake",
            raw_command,
            speed_x10=0,
            steer_x10=int(brake_value),
            blocked_reason=blocked_reason,
            force=True,
        )
        return sent

    def load_config(self, configType):
        with open(self.configPath, "r") as file:
            data = json.load(file)

        if configType == "init":
            capacity = data["init"]["batteryCapacity"]["capacity"]
            command = {"action": "batteryCapacity", "capacity": capacity}
            self.send_to_serial(command)
        else:
            toggle_keys = [
                "ToggleInstant",
                "ToggleBatteryLvl",
                "ToggleImuData",
                "ToggleResourceMonitor",
                "ToggleHallSpeed",
            ]
            for key in toggle_keys:
                if key not in data:
                    continue
                toggle = data[key]
                value_str = toggle["value"]
                value = 0 if str(value_str) == "False" else 1
                command_name = toggle["command"]
                command = {"action": command_name, "activate": value}
                print(
                    f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;96mCONFIG-TX\033[0m"
                    f" config={key} action={command_name!r} value={value}"
                )
                self.send_to_serial(command)
                time.sleep(0.05)

    def convert_fc(self,instantRecv):
        if instantRecv =="True":
            return 1
        else :
            return 0
        
    # ===================================== RUN ==========================================
    def _block_motion_command(self, command_type, speed=None, steer=None, reason=None):
        reason = reason or ("klem_off" if int(self.currentKlemMode) == 0 else "klem_not_30")
        self.last_blocked_reason = reason
        print(
            f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;91mBLOCK\033[0m "
            f"{command_type} speed={speed} steer={steer} "
            f"(KL={self.currentKlemMode}, engineEnabled={self.engineEnabled}, reason={reason})"
        )
        self._record_motion_command(
            command_type,
            None,
            speed_x10=speed,
            steer_x10=steer,
            blocked_reason=reason,
            force=True,
        )

    def thread_work(self):
        """In this function we check if we got the enable engine signal. After we got it we will start getting messages from raspberry PI. It will transform them into NUCLEO commands and send them."""
        try:
            klRecv = self.klSubscriber.receive()
            if klRecv is not None:
                kl_value = str(klRecv)
                live_log(
                    "zmq_motor" if self.motor_output == "zmq" else "serial_writer",
                    event="klem_received",
                    value=kl_value,
                    engine_enabled=bool(self.engineEnabled),
                    current_klem_mode=int(self.currentKlemMode),
                    auto_kl_run_in_sim=bool(self._auto_kl_run_in_sim),
                )
                print(
                    f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;92mINFO\033[0m"
                    f" - threadWrite received Klem \033[94m{kl_value}\033[0m"
                )
                if self.debugger:
                    self.logger.info(kl_value)
                if self._auto_kl_run_in_sim and kl_value in {"0", "15"}:
                    print(
                        "\033[1;97m[ Serial Handler ] :\033[0m \033[1;93mWARNING\033[0m"
                        f" - Ignoring KL=\033[94m{kl_value}\033[0m because "
                        "AUTO_KL_RUN_IN_SIM keeps the simulated motor enabled"
                    )
                    live_log(
                        "zmq_motor",
                        event="klem_ignored",
                        value=kl_value,
                        reason="auto_kl_run_in_sim",
                        engine_enabled=bool(self.engineEnabled),
                        current_klem_mode=int(self.currentKlemMode),
                    )
                    self._publish_actuator_status(force=True)
                elif kl_value == "30":
                    self.running = True
                    self.engineEnabled = True
                    self.currentKlemMode = 30
                    self.last_blocked_reason = None
                    command = {"action": "kl", "mode": 30}
                    self.send_to_serial(command)
                    self.load_config("sensors")
                    self._force_feedback_streams("kl30")
                    if self.last_speed_cmd is not None and self.last_steer_cmd is not None:
                        self._handle_continuous_motion_command(self.last_speed_cmd, self.last_steer_cmd)
                    else:
                        self._publish_actuator_status(force=True)
                elif kl_value == "15":
                    self.running = True
                    self.engineEnabled = False
                    self.currentKlemMode = 15
                    self.last_blocked_reason = "klem_not_30"
                    command = {"action": "kl", "mode": 15}
                    self.send_to_serial(command)
                    self.load_config("sensors")
                    self._force_feedback_streams("kl15")
                    self._publish_actuator_status(force=True)
                elif kl_value == "0":
                    self.running = False
                    self.engineEnabled = False
                    self.currentKlemMode = 0
                    self.last_blocked_reason = "klem_off"
                    command = {"action": "kl", "mode": 0}
                    self.send_to_serial(command)
                    self._publish_actuator_status(force=True)

            isAliveRecv = self.isAliveSubscriber.receive()
            if isAliveRecv is not None:
                if self.debugger:
                    self.logger.info(isAliveRecv)
                command = {"action": "alive", "activate": 0}
                self.send_to_serial(command)

            requestSteerLimitsRecv = self.requestSteerLimitsSubscriber.receive()
            if requestSteerLimitsRecv is not None:
                if self.debugger:
                    self.logger.info(requestSteerLimitsRecv)
                command = {"action": "steerLimits", "request": 0}
                self.send_to_serial(command)

            hallSpeedRecv = self.hallSpeedSubscriber.receive()
            if hallSpeedRecv is not None:
                if self.debugger:
                    self.logger.info(hallSpeedRecv)
                command = {"action": "hallspeed", "activate": self._payload_to_int(hallSpeedRecv)}
                self.send_to_serial(command)

            odoResetRecv = self.odoResetSubscriber.receive()
            if odoResetRecv is not None:
                if self.debugger:
                    self.logger.info(odoResetRecv)
                command = {"action": "odoreset", "request": self._payload_to_int(odoResetRecv, default=1)}
                self.send_to_serial(command)

            simRelocalizeRecv = self.simRelocalizeSubscriber.receive()
            if simRelocalizeRecv is not None and self.motor_output == "zmq":
                self.send_to_serial({"action": "set_pose", **simRelocalizeRecv})

            if not self.running:
                brakeRecv = self.brakeSubscriber.receive()
                speedRecv = self.speedMotorSubscriber.receive()
                steerRecv = self.steerMotorSubscriber.receive()
                controlRecv = self.controlSubscriber.receive()

                if speedRecv is not None:
                    self._block_motion_command(
                        "speed_steer",
                        speed=int(float(speedRecv)),
                        steer=self.last_steer_cmd,
                    )
                if steerRecv is not None:
                    self._block_motion_command(
                        "speed_steer",
                        speed=self.last_speed_cmd,
                        steer=int(float(steerRecv)),
                    )
                if brakeRecv is not None:
                    self._block_motion_command(
                        "brake",
                        speed=0,
                        steer=int(float(brakeRecv)),
                    )
                if controlRecv is not None:
                    self._block_motion_command(
                        "vcd",
                        speed=int(controlRecv.get("Speed", 0)),
                        steer=int(controlRecv.get("Steer", 0)),
                    )

            if self.running:
                brakeRecv = self.brakeSubscriber.receive()
                speedRecv = self.speedMotorSubscriber.receive()
                steerRecv = self.steerMotorSubscriber.receive()

                if speedRecv is not None:
                    self.last_speed_cmd = int(float(speedRecv))
                    self.last_speed_ts = time.time()
                    if self.debugger:
                        self.logger.info(f"Speed cached: {speedRecv} -> {self.last_speed_cmd}")
                    # Log siempre (sin --debugger): si esta línea aparece
                    # pero `[ Serial Handler ] SENT speed=…` no, el dispatch
                    # quedó gated (engineEnabled=False, brake activo, etc.).
                    print(
                        f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;96mRECV\033[0m"
                        f" SpeedMotor={speedRecv!r} cached={self.last_speed_cmd}"
                    )

                if steerRecv is not None:
                    self.last_steer_cmd = int(float(steerRecv))
                    self.last_steer_ts = time.time()
                    if self.debugger:
                        self.logger.info(f"Steer cached: {steerRecv} -> {self.last_steer_cmd}")
                    # Rate-limited: la rampa de steer (50ms) emitiría
                    # cada tick. Solo logueamos si el cached value cambió.
                    last_logged = getattr(self, "_last_logged_steer", None)
                    if self.last_steer_cmd != last_logged:
                        print(
                            f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;96mRECV\033[0m"
                            f" SteerMotor={steerRecv!r} cached={self.last_steer_cmd}"
                        )
                        self._last_logged_steer = self.last_steer_cmd

                if brakeRecv is not None:
                    if self.debugger:
                        self.logger.info(brakeRecv)
                    self.last_speed_cmd = None
                    self.last_speed_ts = 0.0
                    if self.engineEnabled:
                        self._handle_brake_command(int(float(brakeRecv)))
                    else:
                        self._block_motion_command(
                            "brake",
                            speed=0,
                            steer=int(float(brakeRecv)),
                        )

                speed_or_steer_updated = speedRecv is not None or steerRecv is not None
                if speed_or_steer_updated and brakeRecv is None:
                    if self.engineEnabled:
                        # En manual el dashboard manda eventos por eje: ↑/↓
                        # solo emite SpeedMotor, ←/→ solo emite SteerMotor.
                        # Antes exigíamos que AMBOS estuvieran cacheados para
                        # despachar — eso dejaba "↑ sin tocar steer" sin
                        # comando, así que el auto no arrancaba. Defaulteamos
                        # el eje no tocado a 0 (= centro / sin movimiento)
                        # apenas tengamos el otro.
                        speed_to_send = self.last_speed_cmd if self.last_speed_cmd is not None else 0
                        steer_to_send = self.last_steer_cmd if self.last_steer_cmd is not None else 0
                        self._handle_continuous_motion_command(speed_to_send, steer_to_send)
                    else:
                        self._block_motion_command(
                            "speed_steer",
                            speed=self.last_speed_cmd,
                            steer=self.last_steer_cmd,
                        )

                controlRecv = self.controlSubscriber.receive()
                if controlRecv is not None:
                    if self.debugger:
                        self.logger.info(controlRecv)
                    speed_value = int(controlRecv["Speed"])
                    steer_value = int(controlRecv["Steer"])
                    time_value = int(controlRecv["Time"])
                    self.last_speed_cmd = speed_value
                    self.last_steer_cmd = steer_value
                    if self.engineEnabled:
                        command = {
                            "action": "vcd",
                            "speed": speed_value,
                            "steer": steer_value,
                            "time": time_value,
                        }
                        sent, raw_command = self.send_to_serial(command)
                        blocked_reason = None if sent else "serial_disconnected"
                        self._record_motion_command(
                            "vcd",
                            raw_command,
                            speed_x10=speed_value,
                            steer_x10=steer_value,
                            blocked_reason=blocked_reason,
                            force=True,
                        )
                    else:
                        self._block_motion_command(
                            "vcd",
                            speed=speed_value,
                            steer=steer_value,
                        )

                controlCalibRecv = self.controlCalibSubscriber.receive()
                if controlCalibRecv is not None:
                    if self.debugger:
                        self.logger.info(controlCalibRecv)
                    speed_value = int(controlCalibRecv["Speed"])
                    steer_value = int(controlCalibRecv["Steer"])
                    time_value = int(controlCalibRecv["Time"])
                    if self.engineEnabled:
                        command = {
                            "action": "vcdCalib",
                            "speed": speed_value,
                            "steer": steer_value,
                            "time": time_value,
                        }
                        sent, raw_command = self.send_to_serial(command)
                        blocked_reason = None if sent else "serial_disconnected"
                        self._record_motion_command(
                            "vcdCalib",
                            raw_command,
                            speed_x10=speed_value,
                            steer_x10=steer_value,
                            blocked_reason=blocked_reason,
                            force=True,
                        )
                    else:
                        self._block_motion_command(
                            "vcdCalib",
                            speed=speed_value,
                            steer=steer_value,
                        )

                should_refresh_continuous_motion = (
                    self.motor_output == "zmq"
                    and self.engineEnabled
                    and brakeRecv is None
                    and controlRecv is None
                    and controlCalibRecv is None
                    and speedRecv is None
                    and steerRecv is None
                    and (self.last_speed_cmd is not None or self.last_steer_cmd is not None)
                    and (time.monotonic() - self._last_continuous_motion_send_monotonic)
                    >= float(self._continuous_motion_keepalive_s)
                )
                if should_refresh_continuous_motion:
                    speed_to_send = self.last_speed_cmd if self.last_speed_cmd is not None else 0
                    steer_to_send = self.last_steer_cmd if self.last_steer_cmd is not None else 0
                    self._handle_continuous_motion_command(speed_to_send, steer_to_send)

                instantRecv = self.instantSubscriber.receive()
                if instantRecv is not None:
                    if self.debugger:
                        self.logger.info(instantRecv) 
                    command = {"action": "instant", "activate": self._payload_to_int(instantRecv)}
                    self.send_to_serial(command)

                batteryRecv = self.batterySubscriber.receive()
                if batteryRecv is not None: 
                    if self.debugger:
                        self.logger.info(batteryRecv)
                    command = {"action": "battery", "activate": self._payload_to_int(batteryRecv)}
                    self.send_to_serial(command)

                resourceMonitorRecv = self.resourceMonitorSubscriber.receive()
                if resourceMonitorRecv is not None: 
                    if self.debugger:
                        self.logger.info(resourceMonitorRecv)
                    command = {"action": "resourceMonitor", "activate": self._payload_to_int(resourceMonitorRecv)}
                    self.send_to_serial(command)

                imuRecv = self.imuSubscriber.receive()
                if imuRecv is not None: 
                    if self.debugger:
                        self.logger.info(imuRecv)
                    command = {"action": "imu", "activate": self._payload_to_int(imuRecv)}
                    self.send_to_serial(command)

            self._publish_actuator_status()

        except Exception as e:
            print(f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;91mERROR\033[0m - {e}")
            self.serialConnectionStateSender.send(False)

    # ==================================== START =========================================
    def start(self):
        super(threadWrite, self).start()

    # ==================================== STOP ==========================================
    def stop(self):
        """This function will close the thread and will stop the car."""
        self.exampleFlag = False
        command = {"action": "kl", "mode": 0}
        self.send_to_serial(command)
        if self._zmq_sock is not None:
            try:
                self._zmq_sock.close(linger=100)
            except Exception:
                pass
            self._zmq_sock = None
        super(threadWrite, self).stop()

    # ================================== EXAMPLE =========================================
    def example(self):
        """This function simulte the movement of the car."""

        if self.exampleFlag:
            self.speedMotorSender.send({"Type": "Speed", "value": self.s})
            self.steerMotorSender.send({"Type": "Steer", "value": self.i})
            self.i += self.j
            if self.i >= 21.0:
                self.i = 21.0
                self.s = self.i / 7
                self.j *= -1
            if self.i <= -21.0:
                self.i = -21.0
                self.s = self.i / 7
                self.j *= -1.0
            threading.Timer(0.01, self.example).start()

    def _should_send_error(self):
        """Check if we should send an error message (rate limiting)."""
        now = datetime.now()
        if self.last_error_time is None or (now - self.last_error_time) >= self.error_cooldown:
            self.last_error_time = now
            return True
        return False

    def _queue_depth(self, queue_name):
        try:
            return self.queuesList[queue_name].qsize()
        except Exception:
            return "n/a"

    def _preview(self, value, limit=180):
        text = repr(value)
        if len(text) > limit:
            return text[:limit] + "..."
        return text

    def _payload_to_int(self, value, default=0):
        if isinstance(value, bool):
            return 1 if value else 0
        text = str(value).strip().lower()
        if text in {"true", "on", "yes"}:
            return 1
        if text in {"false", "off", "no"}:
            return 0
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)
