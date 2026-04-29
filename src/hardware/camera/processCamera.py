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

# Phase 4–6 control pipeline:
#   BehaviorPlanner (Phase 4) → BehaviorOutput → MotionController (Phase 6)
#   → MotorCommand → safety_gate → SpeedMotor/SteerMotor (firmware).
# Estos cinco componentes son la ÚNICA fuente de verdad para el motor en
# runtime de competencia. Sin BehaviorPlanner el dispatcher no recibe
# BehaviorOutput fresco, el safety_gate dispara su fallback() y el auto
# se queda con speed=0. Por eso el thread del planner se instancia acá.
from src.behavior.planner import BehaviorPlanner
from src.behavior.planner_thread import threadBehaviorPlanner
from src.behavior.scenarios.crosswalk import Crosswalk
from src.behavior.scenarios.highway import Highway
from src.behavior.scenarios.intersection import Intersection
from src.behavior.scenarios.lane_keep import LaneKeep
from src.behavior.scenarios.parking import Parking
from src.behavior.scenarios.roundabout import Roundabout
from src.control.controller_thread import threadMotionController
from src.control.motion_controller import MotionController
from src.control.motor_command_dispatcher import threadMotorCommandDispatcher
from src.control.safety_gate import SafetyGate

from src.core.messaging.buffers import LatestFrameBuffer, LatestValueBuffer
from src.statemachine.stateMachine import StateMachine
from src.statemachine.systemMode import SystemMode
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.core.messaging.allMessages import StateChange

# GPS-free tracking (optional — requires scipy for spline interpolation)
try:
    from src.routing.navigation_planner_thread import threadNavigationPlanner
    from src.localization.pose_estimator_thread import threadPoseEstimator
    from src.localization.relocalization_thread import TrackingState
    from src.routing.visualizer import TrackVisualizer
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

        # Phase 4–6: buffers del path de control nuevo.
        # `behavior_output_buffer` lo escribe `threadBehaviorPlanner` (single
        # source of truth de velocidad y target_path) y lo leen
        # `threadMotionController` (para el MPC) y `threadMotorCommandDispatcher`
        # (para el watchdog del safety_gate).
        # `motor_command_buffer` es el contrato entre `threadMotionController`
        # y `threadMotorCommandDispatcher`: el primero produce un MotorCommand
        # tipado, el segundo lo escribe a SpeedMotor/SteerMotor.
        behavior_output_buffer = LatestValueBuffer()
        motor_command_buffer = LatestValueBuffer()

        # GPS-free tracking: dead reckoning + waypoint follower + map visualizer.
        # El TrackGraph se carga acá una sola vez y se reusa para:
        #   1. dar contexto al pose estimator + navigation planner (route_context),
        #   2. construir un `LaneletMap` consumido por el `BehaviorPlanner`
        #      (escenarios `Intersection`, `Crosswalk`, `Highway`, etc.),
        #   3. opcionalmente alimentar el `TrackVisualizer` (ventana OpenCV).
        # Si el GraphML falta, todo el bloque se desactiva — `BehaviorPlanner`
        # corre con `lanelet_map=None` y solo `LaneKeep` (priority=0) puede
        # disparar.
        tracking_state = None
        track_graph = None
        lanelet_map = None
        if _TRACKING_ENABLED:
            tracking_state = TrackingState()

            try:
                from src.routing.lanelet.from_graphml import TrackGraph
                from src.routing.lanelet.lanelet_map import from_track_graph
                import config as _cfg_track
                _graphml = getattr(_cfg_track, "TRACKING_GRAPHML", "Track GraphML File.graphml")
                _step = getattr(_cfg_track, "TRACKING_WAYPOINT_STEP_M", 0.05)
                import os
                if not os.path.isabs(_graphml):
                    _root = os.path.normpath(
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", "..")
                    )
                    _graphml = os.path.join(_root, _graphml)
                track_graph = TrackGraph(_graphml, step_m=_step)
                # `from_track_graph` densifica centerlines a 0.20 m por defecto:
                # más resolución que el waypoint step (0.05 m) hace `at_pose`
                # más preciso sin disparar memoria.
                lanelet_map = from_track_graph(track_graph)
            except Exception as _graph_err:
                print(f"[ processCamera ] WARNING - track graph load failed: {_graph_err}")
                track_graph = None
                lanelet_map = None

            visualizer = None
            if _TRACKING_SHOW_WINDOW and track_graph is not None:
                try:
                    import os as _os_vis
                    _root = _os_vis.path.normpath(
                        _os_vis.path.join(_os_vis.path.dirname(_os_vis.path.abspath(__file__)),
                                          "..", "..", "..")
                    )
                    _json_path = _os_vis.path.join(_root, "Track Editor Save.json")
                    _img_path  = _os_vis.path.join(_root, "CamScanner 16-3-26 18.52_1.JPG")
                    visualizer = TrackVisualizer(
                        track_graph,
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

        # ── Phase 4 BehaviorPlanner ─────────────────────────────────────
        # Compone los seis escenarios y arranca el thread del planner. Este
        # es el único productor legítimo de `behavior_output_buffer` —
        # antes de wireado, el dispatcher veía el buffer vacío, el
        # `safety_gate` disparaba `fallback()` y el firmware recibía
        # speed=0 indefinidamente.
        #
        # Inyectamos dependencias por constructor (DIP):
        #   - `BehaviorPlanner` recibe la lista de scenarios. Cada scenario
        #     respeta la interfaz `IScenario` (priority + is_active + plan).
        #   - El `lanelet_map` puede ser `None` cuando el GraphML no está
        #     disponible — el planner_thread degrada a un `RouteContext`
        #     vacío y solo `LaneKeep` se activa.
        #   - Los buffers se comparten con pose_estimator, navigation_planner,
        #     lane_observer y motion_controller, todos creados arriba.
        try:
            import config as _cfg_behavior
            _behavior_dt_s = float(getattr(_cfg_behavior, "BEHAVIOR_DT_S", 0.05))
            _behavior_horizon_n = int(getattr(_cfg_behavior, "BEHAVIOR_HORIZON_N", 20))
            _behavior_nominal = float(getattr(_cfg_behavior, "BEHAVIOR_NOMINAL_SPEED_MPS", 0.50))
            _behavior_max = float(getattr(_cfg_behavior, "BEHAVIOR_MAX_SPEED_MPS", 1.00))
            _behavior_pause = float(getattr(_cfg_behavior, "BEHAVIOR_THREAD_PAUSE_S", 0.05))
        except Exception:
            _behavior_dt_s = 0.05
            _behavior_horizon_n = 20
            _behavior_nominal = 0.50
            _behavior_max = 1.00
            _behavior_pause = 0.05

        behavior_planner = BehaviorPlanner(
            scenarios=[
                Parking(),
                Crosswalk(),
                Intersection(),
                Roundabout(),
                Highway(),
                LaneKeep(),
            ]
        )
        behaviorPlannerTh = threadBehaviorPlanner(
            self.queuesList,
            planner=behavior_planner,
            lanelet_map=lanelet_map,
            pose_estimate_buffer=pose_estimate_buffer,
            route_context_buffer=route_context_buffer,
            lane_observation_buffer=lane_observation_buffer,
            stopline_observation_buffer=stopline_observation_buffer,
            tracked_objects_buffer=None,  # Phase 5 MOTTracker aún sin thread wireado
            behavior_output_buffer=behavior_output_buffer,
            dt_s=_behavior_dt_s,
            horizon_n=_behavior_horizon_n,
            nominal_speed_mps=_behavior_nominal,
            max_speed_mps=_behavior_max,
            pause_s=_behavior_pause,
            logging=self.logging,
            debugging=self.debugging,
        )
        self.threads.append(behaviorPlannerTh)

        # ── Phase 6 control pipeline ──────────────────────────────────────
        # Single source of truth: BehaviorPlanner produce el plan
        # (target_path + speed_profile), MotionController lo ejecuta en
        # un MPC acoplado, el dispatcher lo manda al firmware con un
        # safety_gate que detecta staleness.
        motion_controller = MotionController()
        motionControllerTh = threadMotionController(
            self.queuesList,
            controller=motion_controller,
            behavior_output_buffer=behavior_output_buffer,
            pose_estimate_buffer=pose_estimate_buffer,
            motor_command_buffer=motor_command_buffer,
            logging=self.logging,
            debugging=self.debugging,
        )
        self.threads.append(motionControllerTh)

        dispatcherTh = threadMotorCommandDispatcher(
            self.queuesList,
            motor_command_buffer=motor_command_buffer,
            safety_gate=SafetyGate(),
            behavior_output_buffer=behavior_output_buffer,
            pose_estimate_buffer=pose_estimate_buffer,
            logging=self.logging,
            debugging=self.debugging,
        )
        self.threads.append(dispatcherTh)


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
