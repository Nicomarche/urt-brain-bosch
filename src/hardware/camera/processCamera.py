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

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../../..")

import threading

from cv2 import meanShift
from src.templates.workerprocess import WorkerProcess
from src.hardware.camera.threads.threadCamera import threadCamera
from src.hardware.camera.threads.threadLineFollowing import threadLineFollowing
from src.statemachine.stateMachine import StateMachine
from src.statemachine.systemMode import SystemMode
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.allMessages import StateChange

# Sign detection (optional — requires tflite-runtime and model file)
try:
    from src.hardware.camera.threads.threadSignDetection import threadSignDetection
    SIGN_DETECTION_AVAILABLE = True
except ImportError as e:
    SIGN_DETECTION_AVAILABLE = False
    print(f"\033[1;97m[ processCamera ] :\033[0m \033[1;93mWARNING\033[0m - Sign detection not available: {e}")


class processCamera(WorkerProcess):
    """This process handle camera.\n
    Args:
            queueList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
            logging (logging object): Made for debugging.
            debugging (bool, optional): A flag for debugging. Defaults to False.
            camera_type (str): "picamera" for CSI camera (default), "usb" for USB camera.
            usb_device (int|str): USB device index or path (e.g. 0, 2, "/dev/video0").
            usb_resolution (tuple): (width, height) for USB camera. Default (640, 480).
    """

    # ====================================== INIT ==========================================
    def __init__(self, queueList, logging, ready_event=None, debugging=False,
                 camera_type="picamera", usb_device=0, usb_resolution=(640, 480),
                 show_preview=False, debug_windows=None,
                 enable_sign_detection=True, sign_detection_actions=False,
                 sign_min_confidence=0.50, sign_server_url="ws://172.20.10.4:8500/ws/signs",
                 sign_min_box_area=0.01,
                 sign_action_cooldown=15.0):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.camera_type = camera_type
        self.usb_device = usb_device
        self.usb_resolution = usb_resolution
        self.show_preview = show_preview
        self.debug_windows = debug_windows or {}
        self.enable_sign_detection = enable_sign_detection
        self.sign_detection_actions = sign_detection_actions
        self.sign_min_confidence = sign_min_confidence
        self.sign_server_url = sign_server_url
        self.sign_min_box_area = sign_min_box_area
        self.sign_action_cooldown = sign_action_cooldown
        self.stateChangeSubscriber = messageHandlerSubscriber(self.queuesList, StateChange, "lastOnly", True)

        super(processCamera, self).__init__(self.queuesList, ready_event)

    # ================================ STATE CHANGE HANDLER ========================================
    def state_change_handler(self):
        message = self.stateChangeSubscriber.receive()
        if message is not None:
            modeDict = SystemMode[message].value["camera"]["process"]

            if modeDict["enabled"] == True:
                self.resume_threads()
            elif modeDict["enabled"] == False:
                self.pause_threads()

    # ===================================== INIT TH ======================================
    def _init_threads(self):
        """Create the Camera Publisher thread, Line Following thread, and Sign Detection thread."""
        # Shared event: when set, a sign action (stop, crosswalk, etc.) is active
        # and line following must NOT send motor commands.
        sign_action_event = threading.Event()

        # Shared event: when set, car is on highway — line following uses higher speeds.
        highway_mode_event = threading.Event()

        # Camera preview window: only if master switch AND individual toggle are on
        show_cam_preview = self.show_preview and self.debug_windows.get("camera_preview", False)
        camTh = threadCamera(
         self.queuesList, self.logging, self.debugging,
         show_preview=show_cam_preview,
         camera_type=self.camera_type, usb_device=self.usb_device,
         usb_resolution=self.usb_resolution
        )
        self.threads.append(camTh)
        
        # Add line following thread
        # show_debug=True only when master switch SHOW_CAMERA_PREVIEW is on.
        # Individual window toggles are controlled by debug_windows dict.
        lineFollowingTh = threadLineFollowing(
            self.queuesList, self.logging, self.debugging, show_debug=self.show_preview,
            debug_windows=self.debug_windows,
            sign_action_event=sign_action_event,
            highway_mode_event=highway_mode_event
        )
        self.threads.append(lineFollowingTh)

        # Add sign detection thread (optional — requires tflite-runtime + model)
        if self.enable_sign_detection and SIGN_DETECTION_AVAILABLE:
            signDetTh = threadSignDetection(
                self.queuesList, self.logging, self.debugging,
                server_url=self.sign_server_url,
                enable_actions=self.sign_detection_actions,
                min_confidence=self.sign_min_confidence,
                min_box_area=self.sign_min_box_area,
                action_cooldown=self.sign_action_cooldown,
                show_debug=self.show_preview,
                sign_action_event=sign_action_event,
                highway_mode_event=highway_mode_event,
            )
            self.threads.append(signDetTh)
        elif self.enable_sign_detection and not SIGN_DETECTION_AVAILABLE:
            print(
                f"\033[1;97m[ processCamera ] :\033[0m \033[1;93mWARNING\033[0m - "
                f"Sign detection enabled in config but tflite-runtime not installed"
            )


# =================================== EXAMPLE =========================================
#             ++    THIS WILL RUN ONLY IF YOU RUN THE CODE FROM HERE  ++
#                  in terminal:    python3 processCamera.py
if __name__ == "__main__":
    from multiprocessing import Queue, Event
    import time
    import logging
    import cv2
    import base64
    import numpy as np

    allProcesses = list()

    debugg = True

    queueList = {
        "Critical": Queue(),
        "Warning": Queue(),
        "General": Queue(),
        "Config": Queue(),
    }

    logger = logging.getLogger()

    process = processCamera(queueList, logger, debugg)

    process.daemon = True
    process.start()

    time.sleep(4)
    if debugg:
        logger.warning("getting")
    img = {"msgValue": 1}
    while not isinstance(img["msgValue"], str):
        img = queueList["General"].get()
    
    msg_value = img["msgValue"]
    if isinstance(msg_value, str):
        image_data = base64.b64decode(msg_value)
    else:
        raise ValueError("Expected string for base64 decoding")
    img = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if debugg:
        logger.warning("got")
    cv2.imwrite("test.jpg", image)
    process.stop()
