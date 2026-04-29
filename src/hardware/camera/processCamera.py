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
import time

from src.perception.lane.lane_observer_thread import threadLaneObserver
from src.templates.workerprocess import WorkerProcess
from src.hardware.camera.threads.threadCamera import threadCamera
from src.hardware.camera.threads.threadLocalPerception import threadLocalPerception
from src.hardware.camera.threads.threadVisualController import threadVisualController
from src.hardware.control.threads.threadControlCoordinator import threadControlCoordinator
from src.core.messaging.buffers import LatestFrameBuffer, LatestValueBuffer
from src.statemachine.stateMachine import StateMachine
from src.statemachine.systemMode import SystemMode
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.core.messaging.allMessages import StateChange

# GPS-free tracking (optional — requires scipy for spline interpolation)
try:
    from src.hardware.tracking.threadNavigationPlanner import threadNavigationPlanner
    from src.hardware.tracking.threadPoseEstimator import threadPoseEstimator
    from src.hardware.tracking.threadTracking import TrackingState
    from src.hardware.tracking.trackVisualizer import TrackVisualizer
    import config as _cfg
    _TRACKING_ENABLED = True
    _TRACKING_SHOW_WINDOW = getattr(_cfg, "TRACKING_SHOW_WINDOW", True)
except Exception as _tracking_import_err:
    _TRACKING_ENABLED = False
    _TRACKING_SHOW_WINDOW = False
    print(f"\033[1;97m[ processCamera ] :\033[0m \033[1;93mWARNING\033[0m - Tracking not available: {_tracking_import_err}")


class processCamera(WorkerProcess):
    """This process handle camera.\n
    Args:
            queueList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
            logging (logging object): Made for debugging.
            debugging (bool, optional): A flag for debugging. Defaults to False.
            camera_type (str): "jetson" for Jetson CSI, "picamera" for CSI camera (RPi), "usb" for USB camera.
            usb_device (int|str): USB device index or path (e.g. 0, 2, "/dev/video0").
            usb_resolution (tuple): (width, height) for USB camera. Default (640, 480).
    """

    # ====================================== INIT ==========================================
    def __init__(self, queueList, logging, ready_event=None, debugging=False,
                 camera_type="picamera", usb_device=0, usb_resolution=(640, 480),
                 jetson_sensor_id=0, jetson_capture_resolution=(1920, 1080),
                 jetson_output_resolution=(960, 720), jetson_framerate=30,
                 jetson_flip_method=0,
                 show_preview=False, debug_windows=None,
                 picamera_hdr_enabled=True, picamera_hdr_always_on=False,
                 picamera_hdr_glare_threshold=0.04,
                 publish_serial_stream=True,
                 enable_sign_detection=True, sign_detection_actions=False,
                 sign_min_confidence=0.50,
                 sign_min_box_area=0.01,
                 sign_action_cooldown=15.0):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.camera_type = camera_type
        self.usb_device = usb_device
        self.usb_resolution = usb_resolution
        self.jetson_sensor_id = jetson_sensor_id
        self.jetson_capture_resolution = jetson_capture_resolution
        self.jetson_output_resolution = jetson_output_resolution
        self.jetson_framerate = jetson_framerate
        self.jetson_flip_method = jetson_flip_method
        self.show_preview = show_preview
        self.debug_windows = debug_windows or {}
        self.picamera_hdr_enabled = picamera_hdr_enabled
        self.picamera_hdr_always_on = picamera_hdr_always_on
        self.picamera_hdr_glare_threshold = picamera_hdr_glare_threshold
        self.publish_serial_stream = bool(publish_serial_stream)
        self.enable_sign_detection = enable_sign_detection
        self.sign_detection_actions = sign_detection_actions
        self.sign_min_confidence = sign_min_confidence
        self.sign_min_box_area = sign_min_box_area
        self.sign_action_cooldown = sign_action_cooldown
        self.frame_buffer = LatestFrameBuffer()
        self.local_lane_buffer = LatestValueBuffer()
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

        # Shared event: when set, a sign action controls BOTH speed AND steer
        # (e.g. hardcoded 90° left turn after stop sign). Line following must not
        # send steer commands while this event is active.
        steer_override_event = threading.Event()

        # Camera preview window: only if master switch AND individual toggle are on
        show_cam_preview = self.show_preview and self.debug_windows.get("camera_preview", False)
        camTh = threadCamera(
         self.queuesList, self.logging, self.debugging,
         show_preview=show_cam_preview,
         frame_buffer=self.frame_buffer,
         publish_serial_stream=self.publish_serial_stream,
         camera_type=self.camera_type, usb_device=self.usb_device,
         usb_resolution=self.usb_resolution,
         jetson_sensor_id=self.jetson_sensor_id,
         jetson_capture_resolution=self.jetson_capture_resolution,
         jetson_output_resolution=self.jetson_output_resolution,
         jetson_framerate=self.jetson_framerate,
         jetson_flip_method=self.jetson_flip_method,
         picamera_hdr_enabled=self.picamera_hdr_enabled,
         picamera_hdr_always_on=self.picamera_hdr_always_on,
         picamera_hdr_glare_threshold=self.picamera_hdr_glare_threshold,
        )
        self.threads.append(camTh)
        
        # Local AI perception (TensorRT engine) now runs inside the camera process.
        localPerceptionTh = threadLocalPerception(
            self.queuesList, self.logging, self.debugging,
            frame_buffer=self.frame_buffer,
            local_lane_buffer=self.local_lane_buffer,
            show_debug=self.show_preview,
            debug_windows=self.debug_windows,
            enable_sign_detection=self.enable_sign_detection,
            enable_actions=self.sign_detection_actions,
            sign_min_confidence=self.sign_min_confidence,
            sign_min_box_area=self.sign_min_box_area,
            action_cooldown=self.sign_action_cooldown,
            sign_action_event=sign_action_event,
            highway_mode_event=highway_mode_event,
            steer_override_event=steer_override_event,
        )
        self.threads.append(localPerceptionTh)

        lane_observation_buffer = LatestValueBuffer()
        stopline_observation_buffer = LatestValueBuffer()
        pose_estimate_buffer = LatestValueBuffer()
        route_context_buffer = LatestValueBuffer()
        visual_candidate_buffer = LatestValueBuffer()
        visual_state_buffer = LatestValueBuffer()
        control_decision_buffer = LatestValueBuffer()

        # GPS-free tracking: dead reckoning + waypoint follower + map visualizer
        tracking_state = None
        if _TRACKING_ENABLED:
            tracking_state = TrackingState()

            visualizer = None
            if _TRACKING_SHOW_WINDOW:
                try:
                    from src.hardware.tracking.trackGraph import TrackGraph
                    import config as _cfg_vis
                    _graphml = getattr(_cfg_vis, "TRACKING_GRAPHML", "Track GraphML File.graphml")
                    _step = getattr(_cfg_vis, "TRACKING_WAYPOINT_STEP_M", 0.05)
                    import os
                    if not os.path.isabs(_graphml):
                        _root = os.path.normpath(
                            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", "..")
                        )
                        _graphml = os.path.join(_root, _graphml)
                    graph = TrackGraph(_graphml, step_m=_step)
                    _json_path = os.path.join(_root, "Track Editor Save.json")
                    _img_path  = os.path.join(_root, "CamScanner 16-3-26 18.52_1.JPG")
                    visualizer = TrackVisualizer(
                        graph,
                        bg_image_path=_img_path,
                        track_json_path=_json_path,
                    )
                    # Start manually — do NOT add to self.threads because
                    # workerprocess.run() would try to set .daemon on an already-running
                    # thread, which Python forbids.  TrackVisualizer is already daemon=True
                    # so it will terminate automatically when the process exits.
                    visualizer.start()
                except Exception as _vis_err:
                    print(f"[ processCamera ] WARNING - visualizer failed: {_vis_err}")
                    visualizer = None

            poseEstimatorTh = threadPoseEstimator(
                self.queuesList,
                tracking_state,
                lane_observation_buffer=lane_observation_buffer,
                stopline_observation_buffer=stopline_observation_buffer,
                pose_estimate_buffer=pose_estimate_buffer,
                route_context_buffer=route_context_buffer,
                logging=self.logging,
                debugging=self.debugging,
            )
            plannerTh = threadNavigationPlanner(
                self.queuesList,
                tracking_state,
                pose_estimate_buffer=pose_estimate_buffer,
                route_context_buffer=route_context_buffer,
                logging=self.logging,
                debugging=self.debugging,
                visualizer=visualizer,
            )
            self.threads.append(poseEstimatorTh)
            self.threads.append(plannerTh)

        laneObserverTh = threadLaneObserver(
            self.queuesList,
            visual_state_buffer=visual_state_buffer,
            lane_observation_buffer=lane_observation_buffer,
            stopline_observation_buffer=stopline_observation_buffer,
        )
        self.threads.append(laneObserverTh)

        visualControllerTh = threadVisualController(
            self.queuesList, self.logging, self.debugging, frame_buffer=self.frame_buffer,
            local_lane_buffer=self.local_lane_buffer,
            show_debug=self.show_preview,
            debug_windows=self.debug_windows,
            sign_action_event=sign_action_event,
            highway_mode_event=highway_mode_event,
            steer_override_event=steer_override_event,
            visual_candidate_buffer=visual_candidate_buffer,
            visual_state_buffer=visual_state_buffer,
        )
        if tracking_state is not None:
            visualControllerTh.set_tracking_state(tracking_state)
        self.threads.append(visualControllerTh)

        controlCoordinatorTh = threadControlCoordinator(
            self.queuesList,
            tracking_state=tracking_state,
            visual_candidate_buffer=visual_candidate_buffer,
            control_decision_buffer=control_decision_buffer,
            sign_action_event=sign_action_event,
            highway_mode_event=highway_mode_event,
            steer_override_event=steer_override_event,
        )
        self.threads.append(controlCoordinatorTh)


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
