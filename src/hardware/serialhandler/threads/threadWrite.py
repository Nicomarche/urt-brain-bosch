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
    ToggleImuData,
    ToggleInstant,
    ToggleResourceMonitor,
    SerialConnectionState,
    ControlCalib,
    IsAlive,
    RequestSteerLimits
)
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.core.messaging.messageHandlerSender import messageHandlerSender


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

        # error rate limiting
        self.last_error_time = None
        self.error_cooldown = timedelta(seconds=3)

        self._init_senders()
        self._init_subscribers()
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
        self.controlCalibSubscriber = messageHandlerSubscriber(self.queuesList, ControlCalib, "lastOnly", True)
        self.isAliveSubscriber = messageHandlerSubscriber(self.queuesList, IsAlive, "lastOnly", True)
        self.requestSteerLimitsSubscriber = messageHandlerSubscriber(self.queuesList, RequestSteerLimits, "lastOnly", True)
        
    def _init_senders(self):
        self.serialConnectionStateSender = messageHandlerSender(self.queuesList, SerialConnectionState)
        self.actuatorStatusSender = messageHandlerSender(self.queuesList, ActuatorCommandStatus)

    # ==================================== SENDING =======================================

    def send_to_serial(self, msg):
        action = msg.get("action")
        command_msg = self.messageConverter.get_command(**msg)
        if command_msg != "error":
            try:
                with self.process.serialLock:
                    serialCon = self.process.serialCon
                    if serialCon and self.process.serialConnected and serialCon.is_open:
                        serialCon.write(command_msg.encode("ascii"))
                        self.logFile.write(command_msg)
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
            ]
            for key in toggle_keys:
                toggle = data[key]
                value_str = toggle["value"]
                value = 0 if str(value_str) == "False" else 1
                command_name = toggle["command"]
                command = {"action": command_name, "activate": value}
                self.send_to_serial(command)
                time.sleep(0.05)

    def convert_fc(self,instantRecv):
        if instantRecv =="True":
            return 1
        else :
            return 0
        
    # ===================================== RUN ==========================================
    def thread_work(self):
        """In this function we check if we got the enable engine signal. After we got it we will start getting messages from raspberry PI. It will transform them into NUCLEO commands and send them."""
        try:
            klRecv = self.klSubscriber.receive()
            if klRecv is not None:
                print(
                    f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;92mINFO\033[0m"
                    f" - threadWrite received Klem \033[94m{klRecv}\033[0m"
                )
                if self.debugger:
                    self.logger.info(klRecv)
                if klRecv == "30":
                    self.running = True
                    self.engineEnabled = True
                    self.currentKlemMode = 30
                    self.last_blocked_reason = None
                    command = {"action": "kl", "mode": 30}
                    self.send_to_serial(command)
                    self.load_config("sensors")
                    if self.last_speed_cmd is not None and self.last_steer_cmd is not None:
                        self._handle_continuous_motion_command(self.last_speed_cmd, self.last_steer_cmd)
                    else:
                        self._publish_actuator_status(force=True)
                elif klRecv == "15":
                    self.running = True
                    self.engineEnabled = False
                    self.currentKlemMode = 15
                    self.last_blocked_reason = "klem_not_30"
                    command = {"action": "kl", "mode": 15}
                    self.send_to_serial(command)
                    self.load_config("sensors")
                    self._publish_actuator_status(force=True)
                elif klRecv == "0":
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

            if self.running:
                brakeRecv = self.brakeSubscriber.receive()
                speedRecv = self.speedMotorSubscriber.receive()
                steerRecv = self.steerMotorSubscriber.receive()

                if speedRecv is not None:
                    self.last_speed_cmd = int(float(speedRecv))
                    self.last_speed_ts = time.time()
                    if self.debugger:
                        self.logger.info(f"Speed cached: {speedRecv} -> {self.last_speed_cmd}")

                if steerRecv is not None:
                    self.last_steer_cmd = int(float(steerRecv))
                    self.last_steer_ts = time.time()
                    if self.debugger:
                        self.logger.info(f"Steer cached: {steerRecv} -> {self.last_steer_cmd}")

                if brakeRecv is not None:
                    if self.debugger:
                        self.logger.info(brakeRecv)
                    self.last_speed_cmd = None
                    self.last_speed_ts = 0.0
                    if self.engineEnabled:
                        self._handle_brake_command(int(float(brakeRecv)))
                    else:
                        self.last_blocked_reason = "klem_not_30"
                        self._record_motion_command(
                            "brake",
                            None,
                            speed_x10=None,
                            steer_x10=int(float(brakeRecv)),
                            blocked_reason=self.last_blocked_reason,
                            force=True,
                        )

                speed_or_steer_updated = speedRecv is not None or steerRecv is not None
                if speed_or_steer_updated and brakeRecv is None:
                    if self.engineEnabled:
                        if self.last_speed_cmd is not None and self.last_steer_cmd is not None:
                            self._handle_continuous_motion_command(self.last_speed_cmd, self.last_steer_cmd)
                    else:
                        self.last_blocked_reason = "klem_not_30"
                        self._record_motion_command(
                            "speed_steer",
                            None,
                            speed_x10=self.last_speed_cmd,
                            steer_x10=self.last_steer_cmd,
                            blocked_reason=self.last_blocked_reason,
                            force=True,
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
                        self.last_blocked_reason = "klem_not_30"
                        self._record_motion_command(
                            "vcd",
                            None,
                            speed_x10=speed_value,
                            steer_x10=steer_value,
                            blocked_reason=self.last_blocked_reason,
                            force=True,
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
                        self.last_blocked_reason = "klem_not_30"
                        self._record_motion_command(
                            "vcdCalib",
                            None,
                            speed_x10=speed_value,
                            steer_x10=steer_value,
                            blocked_reason=self.last_blocked_reason,
                            force=True,
                        )

                instantRecv = self.instantSubscriber.receive()
                if instantRecv is not None: 
                    if self.debugger:
                        self.logger.info(instantRecv) 
                    command = {"action": "instant", "activate": int(instantRecv)}
                    self.send_to_serial(command)

                batteryRecv = self.batterySubscriber.receive()
                if batteryRecv is not None: 
                    if self.debugger:
                        self.logger.info(batteryRecv)
                    command = {"action": "battery", "activate": int(batteryRecv)}
                    self.send_to_serial(command)

                resourceMonitorRecv = self.resourceMonitorSubscriber.receive()
                if resourceMonitorRecv is not None: 
                    if self.debugger:
                        self.logger.info(resourceMonitorRecv)
                    command = {"action": "resourceMonitor", "activate": int(resourceMonitorRecv)}
                    self.send_to_serial(command)

                imuRecv = self.imuSubscriber.receive()
                if imuRecv is not None: 
                    if self.debugger:
                        self.logger.info(imuRecv)
                    command = {"action": "imu", "activate": int(imuRecv)}
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
