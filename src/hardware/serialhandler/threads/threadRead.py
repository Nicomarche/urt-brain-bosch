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
import logging
import time
import threading
import re
import os
import serial
from datetime import datetime, timedelta

from src.templates.threadwithstop import ThreadWithStop
from src.core.messaging.allMessages import (
    BatteryLvl,
    ImuData,
    ImuAck,
    InstantConsumption,
    EnableButton,
    ResourceMonitor,
    CurrentSpeed,
    CurrentSteer,
    CurrentDistance,
    ShutDownSignal,
    SerialConnectionState,
    CalibPWMData,
    CalibRunDone,
    SteeringLimits,
    AliveSignal
)
from src.core.messaging.messageHandlerSender import messageHandlerSender


class threadRead(ThreadWithStop):
    """This thread read the data that NUCLEO send to Raspberry PI.\n

    Args:
        process (processSerialHandler): ProcessSerialHandler object.
        logFile (FileHandler): The path to the history file where you can find the logs from the connection.
        queueList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
    """

    # ===================================== INIT =========================================
    def __init__(self, process, logFile, queueList, logger, debugger = False):
        super(threadRead, self).__init__(pause=0.02)  # 50Hz — suficiente para leer serial NUCLEO
        self.process = process
        self.logFile = logFile
        self.buffer = ""
        self.queuesList = queueList
        self.logger = logger
        self.debugger = debugger
        self.event = threading.Event()
        self._init_senders()

        self.expectedValues = {"kl": "0, 15 or 30", "instant": "1 or 0", "battery": "1 or 0",
                               "resourceMonitor": "1 or 0", "imu": "1 or 0", "steer" : "between -25 and 25",
                               "speed": "between -500 and 500", "break": "between -250 and 250"}

        self.warningPattern = r'^(-?[0-9]+)H(-?[0-5]?[0-9])M(-?[0-5]?[0-9])S$'
        self.warningColonPattern = r'^(-?[0-9]+):(-?[0-5]?[0-9]):(-?[0-5]?[0-9])$'
        self.resourceMonitorPattern = r'Heap \((\d+\.\d+)\);Stack \((\d+\.\d+)\)'

        # error rate limiting
        self.last_error_time = None
        self.error_cooldown = timedelta(seconds=3)
        self._last_feedback_log = {}
        self._last_feedback_log_time = {}
        self._serial_debug_enabled = True
        self._last_read_status_log = 0.0
        self._read_bytes_total = 0
        self._read_frames_total = 0
        self._read_empty_ticks = 0

        self.queue_sending()

    def _init_senders(self):
        self.enableButtonSender = messageHandlerSender(self.queuesList, EnableButton)
        self.batteryLvlSender = messageHandlerSender(self.queuesList, BatteryLvl)
        self.instantConsumptionSender = messageHandlerSender(self.queuesList, InstantConsumption)
        self.imuDataSender = messageHandlerSender(self.queuesList, ImuData)
        self.imuAckSender = messageHandlerSender(self.queuesList, ImuAck)
        self.resourceMonitorSender = messageHandlerSender(self.queuesList, ResourceMonitor)
        self.currentSpeedSender = messageHandlerSender(self.queuesList, CurrentSpeed)
        self.currentSteerSender = messageHandlerSender(self.queuesList, CurrentSteer)
        self.currentDistanceSender = messageHandlerSender(self.queuesList, CurrentDistance)
        self.warningSender = messageHandlerSender(self.queuesList, ShutDownSignal)
        self.serialConnectionStateSender = messageHandlerSender(self.queuesList, SerialConnectionState)
        self.calibPWMDataSender = messageHandlerSender(self.queuesList, CalibPWMData)
        self.calibRunDoneSender = messageHandlerSender(self.queuesList, CalibRunDone)
        self.steeringLimitsSender = messageHandlerSender(self.queuesList, SteeringLimits)
        self.aliveSignalSender = messageHandlerSender(self.queuesList, AliveSignal)

    # ====================================== RUN ==========================================
    def thread_work(self):
        try:
            with self.process.serialLock:
                serial_con = self.process.serialCon
                if serial_con is None or not self.process.serialConnected or not serial_con.is_open:
                    self._log_read_status(serial_con, reason="not_connected")
                    return

                waiting = serial_con.in_waiting
                if waiting > 0:
                    try:
                        data = serial_con.read(waiting).decode("ascii", errors="ignore")
                        before_len = len(self.buffer)
                        self.buffer += data
                        self._read_bytes_total += len(data)
                        self._read_empty_ticks = 0
                        self._log_serial_read(data, waiting, before_len, len(self.buffer))

                    except Exception as e:
                        if self._should_send_error():
                            self.serialConnectionStateSender.send(False)
                            print(f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;91mERROR\033[0m - Reading from serial ({e})")
                        return
                else:
                    self._read_empty_ticks += 1
                    self._log_read_status(serial_con, reason="idle", waiting=waiting)

            while ";;" in self.buffer:
                msg, self.buffer = self.buffer.split(";;", 1)

                if msg.strip():
                    try:
                        self._read_frames_total += 1
                        self._log_serial_frame(msg.strip(), len(self.buffer))
                        self.send_queue(msg.strip())
                    except Exception as e:
                        print(f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;91mERROR\033[0m - Processing message \033[94m{msg.strip()}\033[0m ({e})")

        except Exception as e:
            if self._should_send_error():
                self.serialConnectionStateSender.send(False)
                print(f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;91mERROR\033[0m - Thread run method ({e})")

    # ==================================== SENDING =======================================
    def queue_sending(self):
        """Callback function for enable button flag."""
        self._send_and_log(EnableButton, self.enableButtonSender, True, "heartbeat", "queue_sending")
        threading.Timer(1, self.queue_sending).start()

    def send_queue(self, buff):
        """This function select which type of message we receive from NUCLEO and send the data further."""

        if ':' not in buff:
            self._log_ignored_frame(buff, "missing ':' separator")
            return

        raw_action, value = buff.split(":", 1)  # Limit to first colon only
        marker = raw_action[0] if raw_action[:1] in ("@", "#") else ""
        action = re.sub(r'[^a-zA-Z0-9]', '', raw_action)
        if not action:
            self._log_ignored_frame(buff, "empty action after cleanup")
            return

        self._log_parsed_action(buff, marker, action, value)
        if marker != "@":
            reason = (
                "expected Nucleo feedback marker '@'; '#' looks like command echo"
                if marker == "#"
                else "expected Nucleo feedback marker '@'"
            )
            self._log_ignored_frame(buff, reason, warning=True)
            return
        if self.debugger:
            self.logger.info(buff)

        if action == "imu":
            splittedValue = value.split(";")
            if(len(buff)>20):
                data = {
                    "roll": splittedValue[0],
                    "pitch": splittedValue[1],
                    "yaw": splittedValue[2],
                    "accelx": splittedValue[3],
                    "accely": splittedValue[4],
                    "accelz": splittedValue[5],
                }
                self._send_and_log(ImuData, self.imuDataSender, str(data), action, buff)
            else:
                self._send_and_log(ImuAck, self.imuAckSender, splittedValue[0], action, buff)

        elif action == "brake":
            self._send_and_log(CurrentSpeed, self.currentSpeedSender, 0.0, action, buff)
            self._send_and_log(CurrentSteer, self.currentSteerSender, 0.0, action, buff)

        elif action == "speed":
            speed = value.split(",")[0]
            if self.is_float(speed):
                speed_value = float(speed)
                self._send_and_log(CurrentSpeed, self.currentSpeedSender, speed_value, action, buff)
                self._log_feedback_value("speed", speed_value)
            else:
                self._log_ignored_frame(buff, f"speed is not numeric: {speed!r}", warning=True)

        elif action == "steer":
            steer = value.split(",")[0]
            if self.is_float(steer):
                steer_value = float(steer)
                self._send_and_log(CurrentSteer, self.currentSteerSender, steer_value, action, buff)
                self._log_feedback_value("steer", steer_value)
            else:
                self._log_ignored_frame(buff, f"steer is not numeric: {steer!r}", warning=True)

        elif action == "odo":
            distance = value.split(";")[0]
            if self.is_float(distance):
                distance_mm = int(float(distance))
                self._send_and_log(
                    CurrentDistance,
                    self.currentDistanceSender,
                    distance_mm,
                    action,
                    buff,
                )
                self._log_feedback_value("odo_mm", distance_mm)
            else:
                self._log_ignored_frame(buff, f"odo is not numeric: {distance!r}", warning=True)

        elif action == "vcdCalib":
            splittedValue = value.split(";")
            speedPWM = splittedValue[0]
            steerPWM = splittedValue[1]

            if speedPWM == "0" and steerPWM == "0":
                self._send_and_log(CalibRunDone, self.calibRunDoneSender, True, action, buff)
            else:
                self._send_and_log(
                    CalibPWMData,
                    self.calibPWMDataSender,
                    {"speedPWM": speedPWM, "steerPWM": steerPWM},
                    action,
                    buff,
                )

        elif action == "vcd":
            self._log_ack(action, value, buff)

        elif action == "alive":
            self._send_and_log(AliveSignal, self.aliveSignalSender, True, action, buff)

        elif action == "steerLimits":
            splittedValue = value.split(";")
            lowerLimit = splittedValue[0]
            upperLimit = splittedValue[1]
            self._send_and_log(
                SteeringLimits,
                self.steeringLimitsSender,
                {"lowerLimit": lowerLimit, "upperLimit": upperLimit},
                action,
                buff,
            )

        elif action == "instant":
            if self.check_valid_value(action, value):
                self._send_and_log(
                    InstantConsumption,
                    self.instantConsumptionSender,
                    float(value),
                    action,
                    buff,
                )

        elif action == "battery":
            if self.check_valid_value(action, value):
                percentage = (int(value)-7000)/14
                percentage = max(0, min(100, round(percentage)))

                self._send_and_log(BatteryLvl, self.batteryLvlSender, percentage, action, buff)

        elif action == "resourceMonitor":
            if self.check_valid_value(action, value):
                data = re.match(self.resourceMonitorPattern, value)
                if data:
                    message = {"heap": data.group(1), "stack": data.group(2)}
                    self._send_and_log(ResourceMonitor, self.resourceMonitorSender, message, action, buff)
                else:
                    self._log_ignored_frame(buff, "resourceMonitor regex did not match", warning=True)

        elif action == "warning":
            data = re.match(self.warningPattern, value) or re.match(self.warningColonPattern, value)
            if data:
                warning_value = f"{data.group(1)}:{data.group(2)}:{data.group(3)}"
                print(f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;93mWARNING\033[0m - Shutdown in \033[94m{data.group(1)}h {data.group(2)}m {data.group(3)}s\033[0m")
                self._send_and_log(ShutDownSignal, self.warningSender, warning_value, action, buff)
            else:
                self._log_ignored_frame(buff, "warning regex did not match", warning=True)

        elif action == "shutdown":
            if value == "ack":
                self._log_ack(action, value, buff)
            else:
                print(f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;93mWARNING\033[0m - \033[94mShutting down now!\033[0m")
                self.event.wait(3)
                os.system("sudo shutdown -h now")

        elif action in ("kl", "hallspeed", "odoreset", "batteryCapacity"):
            self._log_ack(action, value, buff)

        else:
            self._log_ignored_frame(buff, f"unknown action={action!r}", warning=True)
            
    def check_valid_value(self, action, message):
        if message == "syntax error":
            print(f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;93mWARNING\033[0m - Invalid \033[94m{action.upper()}\033[0m value (expected {self.expectedValues[action]})")
            return False
    
        if message == "kl 15/30 is required!!":
            print(f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;93mWARNING\033[0m - KL 15/30 required for \033[94m{action.upper()}\033[0m")
            return False
        
        if message == "ack":
            return False
        return True
    
    def is_float(self, string):
        try:
            float(string)
        except ValueError:
            return False

        return True

    def _log_feedback_value(self, name, value):
        """Log feedback changes lightly so manual-command issues are traceable."""
        now = time.time()
        last_value = self._last_feedback_log.get(name)
        last_time = self._last_feedback_log_time.get(name, 0.0)
        changed = last_value is None or abs(float(value) - float(last_value)) >= 1.0
        if not changed and (now - last_time) < 1.0:
            return

        self._last_feedback_log[name] = float(value)
        self._last_feedback_log_time[name] = now
        print(
            f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;96mFEEDBACK\033[0m"
            f" {name}={float(value):.1f}"
        )

    def _preview(self, value, limit=180):
        text = repr(value)
        if len(text) > limit:
            return text[:limit] + "..."
        return text

    def _queue_depth(self, queue_name):
        try:
            return self.queuesList[queue_name].qsize()
        except Exception:
            return "n/a"

    def _log_serial_read(self, data, waiting, before_len, after_len):
        if not self._serial_debug_enabled:
            return
        print(
            f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;96mRAW\033[0m"
            f" threadRead read bytes={len(data)} requested={waiting}"
            f" buffer={before_len}->{after_len} chunk={self._preview(data)}"
        )

    def _log_read_status(self, serial_con, reason, waiting=None):
        if not self._serial_debug_enabled:
            return
        now = time.time()
        if now - self._last_read_status_log < 2.0:
            return
        self._last_read_status_log = now
        try:
            is_open = bool(serial_con and serial_con.is_open)
        except Exception:
            is_open = False
        if waiting is None and serial_con is not None:
            try:
                waiting = serial_con.in_waiting
            except Exception as exc:
                waiting = f"err:{exc}"
        print(
            f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;96mRX-IDLE\033[0m"
            f" reason={reason} device={getattr(self.process, 'serialDevice', None)!r}"
            f" connected={getattr(self.process, 'serialConnected', None)}"
            f" open={is_open} in_waiting={waiting}"
            f" buffer_len={len(self.buffer)} empty_ticks={self._read_empty_ticks}"
            f" bytes_total={self._read_bytes_total} frames_total={self._read_frames_total}"
        )

    def _log_serial_frame(self, frame, remaining_buffer_len):
        if not self._serial_debug_enabled:
            return
        print(
            f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;96mFRAME\033[0m"
            f" threadRead frame={self._preview(frame)}"
            f" remaining_buffer={remaining_buffer_len}"
        )

    def _log_parsed_action(self, raw, marker, action, value):
        if not self._serial_debug_enabled:
            return
        marker_text = marker or "<none>"
        print(
            f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;96mPARSE\033[0m"
            f" marker={marker_text!r} action={action!r}"
            f" value={self._preview(value)} raw={self._preview(raw)}"
        )

    def _log_ignored_frame(self, raw, reason, warning=False):
        if not self._serial_debug_enabled:
            return
        tag = "\033[1;93mWARN\033[0m" if warning else "\033[1;93mDROP\033[0m"
        print(
            f"\033[1;97m[ Serial Handler ] :\033[0m {tag}"
            f" threadRead ignored raw={self._preview(raw)} reason={reason}"
        )

    def _log_ack(self, action, value, raw):
        if not self._serial_debug_enabled:
            return
        print(
            f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;96mACK\033[0m"
            f" action={action!r} value={self._preview(value)} raw={self._preview(raw)}"
        )

    def _send_and_log(self, enum_cls, sender, value, action, raw):
        sender.send(value)
        if not self._serial_debug_enabled:
            return
        queue_name = enum_cls.Queue.value
        print(
            f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;96mQUEUE\033[0m"
            f" threadRead -> {enum_cls.__name__}"
            f" queue={queue_name} owner={enum_cls.Owner.value}"
            f" msgID={enum_cls.msgID.value} type={enum_cls.msgType.value}"
            f" qsize={self._queue_depth(queue_name)}"
            f" value={self._preview(value)}"
            f" from_action={action!r} raw={self._preview(raw)}"
        )

    def _should_send_error(self):
        """Check if we should send an error message (rate limiting)."""
        now = datetime.now()
        if self.last_error_time is None or (now - self.last_error_time) >= self.error_cooldown:
            self.last_error_time = now
            return True
        return False
