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
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWITrueSE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

import queue
import threading
import time

from src.templates.threadwithstop import ThreadWithStop
from src.core.messaging.allMessages import (
    ActuatorCommandStatus,
    AliveSignal,
    BatteryLvl,
    CalibPWMData,
    CalibRunDone,
    CurrentDistance,
    CurrentSpeed,
    CurrentSteer,
    DetectedObjectsMsg,
    EnableButton,
    Brake,
    ImuData,
    InstantConsumption,
    Klem,
    OdoReset,
    LidarScanMsg,
    LidarStatusMsg,
    LineFollowingDebug,
    LineFollowingStatus,
    LocalLanePerception,
    LocalPerceptionStatus,
    SpeedMotor,
    SignDetectionDebug,
    SignDetectionStatus,
    SteerMotor,
    ToggleHallSpeed,
    ResourceMonitor,
    SerialConnectionState,
    SteeringLimits,
    ShutDownSignal,
    TrackedObjectsMsg,
    mainCamera,
    serialCamera,
)


class _AsyncPipeSender:
    """Owns one subscriber pipe so a slow receiver cannot block Gateway."""

    def __init__(self, receiver, pipe, logger, latest_only=False):
        self.receiver = receiver
        self.pipe = pipe
        self.logger = logger
        self.latest_only = bool(latest_only)
        self._closed = False
        self._queue = queue.Queue(maxsize=1 if self.latest_only else 32)
        self._sentinel = object()
        self._thread = threading.Thread(
            target=self._run,
            name=f"GatewayPipeSender-{receiver}",
            daemon=True,
        )
        self._thread.start()

    @property
    def closed(self):
        return self._closed

    def submit(self, payload):
        if self._closed:
            return False

        if self.latest_only:
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self._queue.put_nowait(payload)
            return True
        except queue.Full:
            # FIFO command queues should almost never fill. If they do, drop
            # for this receiver only instead of freezing the global gateway.
            return False

    def close(self):
        self._closed = True
        try:
            self.pipe.close()
        except Exception:
            pass
        try:
            self._queue.put_nowait(self._sentinel)
        except queue.Full:
            pass

    def _run(self):
        while not self._closed:
            payload = self._queue.get()
            if payload is self._sentinel:
                break
            try:
                self.pipe.send(payload)
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._closed = True
                try:
                    self.logger.warning(
                        f"[ Gateway ] Pipe send failed for subscriber "
                        f"'{self.receiver}' ({exc})"
                    )
                except Exception:
                    pass
                break


class threadGateway(ThreadWithStop):
    """Thread which will handle processGateway functionalities.\n
    Args:
        queuesList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
    """

    # ===================================== INIT =========================================

    def __init__(self, queueList, logger, debugging):
        super(threadGateway, self).__init__(pause=0.005)  # 200Hz — evita backlog en streams de camara/debug
        self.logger = logger
        self.debugging = debugging
        self.sendingList = {}
        self.queuesList = queueList
        self.messageApproved = []
        self.max_priority_batch = 32
        self.max_general_batch = 64
        self.max_config_batch = 32
        self._klem_key = (Klem.Owner.value, Klem.msgID.value)
        self._manual_control_latest_keys = {
            (SpeedMotor.Owner.value, SpeedMotor.msgID.value),
            (SteerMotor.Owner.value, SteerMotor.msgID.value),
            (Brake.Owner.value, Brake.msgID.value),
        }
        self._latest_only_keys = {
            (mainCamera.Owner.value, mainCamera.msgID.value),
            (serialCamera.Owner.value, serialCamera.msgID.value),
            (LineFollowingDebug.Owner.value, LineFollowingDebug.msgID.value),
            (LineFollowingStatus.Owner.value, LineFollowingStatus.msgID.value),
            (LocalLanePerception.Owner.value, LocalLanePerception.msgID.value),
            (LocalPerceptionStatus.Owner.value, LocalPerceptionStatus.msgID.value),
            (SignDetectionDebug.Owner.value, SignDetectionDebug.msgID.value),
            (SignDetectionStatus.Owner.value, SignDetectionStatus.msgID.value),
            (LidarScanMsg.Owner.value, LidarScanMsg.msgID.value),
            (LidarStatusMsg.Owner.value, LidarStatusMsg.msgID.value),
            (DetectedObjectsMsg.Owner.value, DetectedObjectsMsg.msgID.value),
            (TrackedObjectsMsg.Owner.value, TrackedObjectsMsg.msgID.value),
            (ActuatorCommandStatus.Owner.value, ActuatorCommandStatus.msgID.value),
            (ImuData.Owner.value, ImuData.msgID.value),
            (CurrentSpeed.Owner.value, CurrentSpeed.msgID.value),
            (CurrentSteer.Owner.value, CurrentSteer.msgID.value),
            (CurrentDistance.Owner.value, CurrentDistance.msgID.value),
            *self._manual_control_latest_keys,
        }
        self._serial_trace_message_classes = (
            ActuatorCommandStatus,
            AliveSignal,
            BatteryLvl,
            CalibPWMData,
            CalibRunDone,
            CurrentDistance,
            CurrentSpeed,
            CurrentSteer,
            EnableButton,
            ImuData,
            InstantConsumption,
            ResourceMonitor,
            SerialConnectionState,
            SteeringLimits,
            ShutDownSignal,
            SpeedMotor,
            SteerMotor,
            Brake,
            Klem,
            ToggleHallSpeed,
            OdoReset,
        )
        self._serial_trace_keys = {
            (cls.Owner.value, cls.msgID.value)
            for cls in self._serial_trace_message_classes
        }
        self._serial_trace_labels = {
            (cls.Owner.value, cls.msgID.value): cls.__name__
            for cls in self._serial_trace_message_classes
        }

    # =================================== SUBSCRIBE ======================================

    def subscribe(self, message):
        """This functin will add the pipe into the approved messages list and it will be added into the dictionary of sending
        Args:
            message(dictionary): Dictionary received from the multiprocessing queues ( the config one).
        """
        
        # Declaration of variables:
        Owner = message["Owner"]
        Id = message["msgID"]
        To = message["To"]["receiver"]
        Pipe = message["To"]["pipe"]
        message_key = (Owner, Id)
        if not Owner in self.sendingList.keys():
            self.sendingList[Owner] = {}
        if not Id in self.sendingList[Owner].keys():
            self.sendingList[Owner][Id] = {}
        existing = self.sendingList[Owner][Id].get(To)
        if existing is None or getattr(existing, "closed", False):
            if existing is not None:
                try:
                    existing.close()
                except Exception:
                    pass
            self.sendingList[Owner][Id][To] = _AsyncPipeSender(
                To,
                Pipe,
                self.logger,
                latest_only=message_key in self._latest_only_keys,
            )
        if (Owner, Id) not in self.messageApproved:
            self.messageApproved.append((Owner, Id))
        if message_key in self._serial_trace_keys:
            print(
                f"\033[1;97m[ Gateway ] :\033[0m \033[1;96mSUB\033[0m"
                f" {self._message_label(message_key)} subscribed by \033[94m{To}\033[0m"
                f" latest_only={message_key in self._latest_only_keys}"
                f" total_subscribers={len(self.sendingList[Owner][Id])}"
            )
        if (Owner, Id) == self._klem_key:
            print(
                f"\033[1;97m[ Gateway ] :\033[0m \033[1;92mINFO\033[0m"
                f" - Klem subscribed by \033[94m{To}\033[0m"
            )
        # Debugging( you can comment this):
        if self.debugging:
            self.print_list()

    # ================================== UNSUBSCRIBE =====================================

    def unsubscribe(self, message):
        """This functin will remove the pipe into the approved messages list and it will be added into the dictionary of sending
        Args:
            message(dictionary): Dictionary received from the multiprocessing queues ( the config one).
        """

        Owner = message["Owner"]
        Id = message["msgID"]
        To = message["To"]["receiver"]

        # We delete the value from Dictionary
        sender = self.sendingList.get(Owner, {}).get(Id, {}).pop(To, None)
        if sender is not None:
            try:
                sender.close()
            except Exception:
                pass
        if not self.sendingList.get(Owner, {}).get(Id) and (Owner, Id) in self.messageApproved:
            self.messageApproved.remove((Owner, Id))
        if (Owner, Id) in self._serial_trace_keys:
            remaining = len(self.sendingList.get(Owner, {}).get(Id, {}))
            print(
                f"\033[1;97m[ Gateway ] :\033[0m \033[1;96mUNSUB\033[0m"
                f" {self._message_label((Owner, Id))} unsubscribed by \033[94m{To}\033[0m"
                f" remaining_subscribers={remaining}"
            )
        if (Owner, Id) == self._klem_key:
            print(
                f"\033[1;97m[ Gateway ] :\033[0m \033[1;92mINFO\033[0m"
                f" - Klem unsubscribed by \033[94m{To}\033[0m"
            )
        if self.debugging:
            self.print_list()

    # =================================== SENDING ========================================

    def send(self, message):
        """This functin will send the message on all the pipes that are in the sending list of the message ID.
        Args:
            message(dictionary): Dictionary received from the multiprocessing queues ( the config one).
        """

        Owner = message["Owner"]
        Id = message["msgID"]
        Type = message["msgType"]
        Value = message["msgValue"]
        message_key = (Owner, Id)
        trace_message = message_key in self._serial_trace_keys
        if message_key == self._klem_key:
            receivers = list(self.sendingList.get(Owner, {}).get(Id, {}).keys())
            print(
                f"\033[1;97m[ Gateway ] :\033[0m \033[1;92mINFO\033[0m"
                f" - Dispatching Klem \033[94m{Value}\033[0m to \033[94m{len(receivers)}\033[0m subscriber(s)"
            )
        elif trace_message:
            receivers = list(self.sendingList.get(Owner, {}).get(Id, {}).keys())
            print(
                f"\033[1;97m[ Gateway ] :\033[0m \033[1;96mROUTE\033[0m"
                f" {self._message_label(message_key)} owner={Owner} msgID={Id}"
                f" type={Type} subscribers={len(receivers)}"
                f" approved={message_key in self.messageApproved}"
                f" value={self._preview(Value)}"
            )

        if message_key in self.messageApproved:
            # Snapshot de claves para poder mutar sendingList durante la iteración
            # (removemos pipes rotos in-place).
            broken = []
            for element in list(self.sendingList[Owner][Id].keys()):
                sender = self.sendingList[Owner][Id][element]
                if getattr(sender, "closed", False):
                    broken.append(element)
                    continue
                try:
                    # We send a dictionary that contain the type of the message and message
                    submitted = sender.submit(
                        {"Type": Type, "value": Value, "id": Id, "Owner": Owner}
                    )
                    if not submitted:
                        if message_key == self._klem_key or trace_message or self.debugging:
                            msg = (
                                f"[ Gateway ] Subscriber queue full; dropping "
                                f"{self._message_label(message_key)} for '{element}' "
                                f"(Owner={Owner}, msgID={Id})"
                            )
                            self.logger.warning(msg)
                            print(
                                f"\033[1;97m[ Gateway ] :\033[0m \033[1;91mDROP\033[0m"
                                f" {msg}"
                            )
                        continue
                except Exception as exc:
                    # El recv-end del subscriber se cerró (spawn mode, proceso
                    # terminado, o doble-registro desde el proceso padre).
                    # Lo removemos para no volver a fallar en mensajes futuros,
                    # y continuamos con el resto de los subscribers.
                    broken.append(element)
                    self.logger.warning(
                        f"[ Gateway ] BrokenPipe → removing subscriber "
                        f"'{element}' (Owner={Owner}, msgID={Id}, error={exc})"
                    )
                    continue
                if message_key == self._klem_key:
                    print(
                        f"\033[1;97m[ Gateway ] :\033[0m \033[1;92mINFO\033[0m"
                        f" - Queued Klem \033[94m{Value}\033[0m to \033[94m{element}\033[0m"
                    )
                elif trace_message:
                    print(
                        f"\033[1;97m[ Gateway ] :\033[0m \033[1;96mPIPE\033[0m"
                        f" {self._message_label(message_key)} -> \033[94m{element}\033[0m"
                        f" latest_only={getattr(sender, 'latest_only', False)}"
                        f" submitted={submitted}"
                    )
                if self.debugging:
                    self.logger.warning(message)
            for element in broken:
                try:
                    sender = self.sendingList[Owner][Id].pop(element)
                    sender.close()
                except KeyError:
                    pass
        elif message_key == self._klem_key:
            print(
                f"\033[1;97m[ Gateway ] :\033[0m \033[1;93mWARNING\033[0m"
                f" - Klem \033[94m{Value}\033[0m has no approved subscribers"
            )
        elif trace_message:
            print(
                f"\033[1;97m[ Gateway ] :\033[0m \033[1;93mNO-SUB\033[0m"
                f" {self._message_label(message_key)} owner={Owner} msgID={Id}"
                f" value={self._preview(Value)}"
            )

    # ====================================================================================

    # Function for debugging:
    def print_list(self):
        """Made for debugging"""

        self.logger.warning(self.sendingList)

    def _message_label(self, message_key):
        return self._serial_trace_labels.get(message_key, f"{message_key[0]}:{message_key[1]}")

    def _preview(self, value, limit=180):
        text = repr(value)
        if len(text) > limit:
            return text[:limit] + "..."
        return text

    def _drain_queue(self, queue_name, max_items, collapse_latest=False):
        """Drain several pending messages in one cycle to avoid queue backlog."""
        target_queue = self.queuesList[queue_name]
        messages = []
        latest_positions = {}

        for _ in range(max_items):
            try:
                message = target_queue.get_nowait()
            except queue.Empty:
                break

            if collapse_latest:
                key = (message.get("Owner"), message.get("msgID"))
                if key in self._latest_only_keys:
                    previous_idx = latest_positions.get(key)
                    if previous_idx is None:
                        latest_positions[key] = len(messages)
                        messages.append(message)
                    else:
                        messages[previous_idx] = message
                    continue

            messages.append(message)

        return messages

    # ==================================== RUN ===========================================

    def _process_config_messages(self):
        for message2 in self._drain_queue("Config", self.max_config_batch):
            if str.lower(message2["Subscribe/Unsubscribe"]) == "subscribe":
                self.subscribe(message2)
            else:
                self.unsubscribe(message2)

    def thread_work(self):
        """This function will take the messages in priority order form the queues.\n
        the prioirty is: Critical > Warning > General
        """

        self._process_config_messages()

        messages = []
        source_queue = None
        if not self.queuesList["Critical"].empty():
            messages = self._drain_queue("Critical", self.max_priority_batch)
            source_queue = "Critical"
        elif not self.queuesList["Warning"].empty():
            messages = self._drain_queue("Warning", self.max_priority_batch)
            source_queue = "Warning"
        elif not self.queuesList["General"].empty():
            messages = self._drain_queue("General", self.max_general_batch, collapse_latest=True)
            source_queue = "General"

        for message in messages:
            message_key = (message.get("Owner"), message.get("msgID"))
            if message_key in self._serial_trace_keys:
                print(
                    f"\033[1;97m[ Gateway ] :\033[0m \033[1;96mQUEUE\033[0m"
                    f" {source_queue} -> {self._message_label(message_key)}"
                    f" owner={message.get('Owner')} msgID={message.get('msgID')}"
                    f" type={message.get('msgType')}"
                    f" value={self._preview(message.get('msgValue'))}"
                )
            if message_key == self._klem_key:
                print(
                    f"\033[1;97m[ Gateway ] :\033[0m \033[1;92mINFO\033[0m"
                    f" - Received Klem \033[94m{message.get('msgValue')}\033[0m from \033[94m{source_queue}\033[0m queue"
                )
            self.send(message)

        self._process_config_messages()

        # print(time.perf_counter_ns())

    def stop(self):
        for owner in list(self.sendingList.values()):
            for msg_receivers in list(owner.values()):
                for sender in list(msg_receivers.values()):
                    try:
                        sender.close()
                    except Exception:
                        pass
        super(threadGateway, self).stop()


# =====================================================================================
