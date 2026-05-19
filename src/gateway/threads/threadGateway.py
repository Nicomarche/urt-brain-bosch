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
import time

from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (
    ActuatorCommandStatus,
    CurrentSpeed,
    CurrentSteer,
    ImuData,
    Klem,
    LineFollowingDebug,
    LineFollowingStatus,
    LocalLanePerception,
    LocalPerceptionStatus,
    SignDetectionDebug,
    SignDetectionStatus,
    StartupMoveStatus,
    mainCamera,
    serialCamera,
)

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
        self._latest_only_keys = {
            (mainCamera.Owner.value, mainCamera.msgID.value),
            (serialCamera.Owner.value, serialCamera.msgID.value),
            (LineFollowingDebug.Owner.value, LineFollowingDebug.msgID.value),
            (LineFollowingStatus.Owner.value, LineFollowingStatus.msgID.value),
            (StartupMoveStatus.Owner.value, StartupMoveStatus.msgID.value),
            (LocalLanePerception.Owner.value, LocalLanePerception.msgID.value),
            (LocalPerceptionStatus.Owner.value, LocalPerceptionStatus.msgID.value),
            (SignDetectionDebug.Owner.value, SignDetectionDebug.msgID.value),
            (SignDetectionStatus.Owner.value, SignDetectionStatus.msgID.value),
            (ActuatorCommandStatus.Owner.value, ActuatorCommandStatus.msgID.value),
            (ImuData.Owner.value, ImuData.msgID.value),
            (CurrentSpeed.Owner.value, CurrentSpeed.msgID.value),
            (CurrentSteer.Owner.value, CurrentSteer.msgID.value),
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
        if not Owner in self.sendingList.keys():
            self.sendingList[Owner] = {}
        if not Id in self.sendingList[Owner].keys():
            self.sendingList[Owner][Id] = {}
        if not To in self.sendingList[Owner][Id].keys():
            self.sendingList[Owner][Id][To] = Pipe
        self.messageApproved.append((Owner, Id))
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
        del self.sendingList[Owner][Id][To]
        self.messageApproved.remove((Owner, Id))
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
        if message_key == self._klem_key:
            receivers = list(self.sendingList.get(Owner, {}).get(Id, {}).keys())
            print(
                f"\033[1;97m[ Gateway ] :\033[0m \033[1;92mINFO\033[0m"
                f" - Dispatching Klem \033[94m{Value}\033[0m to \033[94m{len(receivers)}\033[0m subscriber(s)"
            )

        if message_key in self.messageApproved:
            for element in self.sendingList[Owner][Id]:
                # We send a dictionary that contain the type of the message and message
                self.sendingList[Owner][Id][element].send(
                    {"Type": Type, "value": Value, "id": Id, "Owner": Owner}
                )
                if message_key == self._klem_key:
                    print(
                        f"\033[1;97m[ Gateway ] :\033[0m \033[1;92mINFO\033[0m"
                        f" - Delivered Klem \033[94m{Value}\033[0m to \033[94m{element}\033[0m"
                    )
                if self.debugging:
                    self.logger.warning(message)
        elif message_key == self._klem_key:
            print(
                f"\033[1;97m[ Gateway ] :\033[0m \033[1;93mWARNING\033[0m"
                f" - Klem \033[94m{Value}\033[0m has no approved subscribers"
            )

    # ====================================================================================

    # Function for debugging:
    def print_list(self):
        """Made for debugging"""

        self.logger.warning(self.sendingList)

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

    def thread_work(self):
        """This function will take the messages in priority order form the queues.\n
        the prioirty is: Critical > Warning > General
        """

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
            if (message.get("Owner"), message.get("msgID")) == self._klem_key:
                print(
                    f"\033[1;97m[ Gateway ] :\033[0m \033[1;92mINFO\033[0m"
                    f" - Received Klem \033[94m{message.get('msgValue')}\033[0m from \033[94m{source_queue}\033[0m queue"
                )
            self.send(message)

        for message2 in self._drain_queue("Config", self.max_config_batch):
            if str.lower(message2["Subscribe/Unsubscribe"]) == "subscribe":
                self.subscribe(message2)
            else:
                self.unsubscribe(message2)

        # print(time.perf_counter_ns())


# =====================================================================================
