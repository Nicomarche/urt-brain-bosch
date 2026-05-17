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

from src.core.messaging.buffers import LatestFrameBuffer, LatestValueBuffer
from src.statemachine.stateMachine import StateMachine
from src.statemachine.systemMode import SystemMode
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.core.bus.topics import STATE_CHANGE


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
                 stream_to_dashboard=True,
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
        self.stream_to_dashboard = bool(stream_to_dashboard)
        self.enable_sign_detection = enable_sign_detection
        self.sign_detection_actions = sign_detection_actions
        self.sign_min_confidence = sign_min_confidence
        self.sign_min_box_area = sign_min_box_area
        self.sign_action_cooldown = sign_action_cooldown
        self.frame_buffer = LatestFrameBuffer()
        self.local_lane_buffer = LatestValueBuffer()
        self.stateChangeSubscriber = messageHandlerSubscriber(self.queuesList, STATE_CHANGE, "lastOnly", True)

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
        # Lazy imports — must only execute in the child process (post-fork).
        #
        # Root cause of SIGSEGV in __kmp_suspend_initialize_thread:
        #   If these are module-level, main.py imports processCamera in the
        #   PARENT, which transitively loads trajectory_builder → numpy →
        #   OpenBLAS → Homebrew libomp (UUID 63e2ee98).  After fork the child
        #   inherits stale OpenMP thread structures (only the calling thread
        #   survives fork); then torch loads its own bundled libomp (UUID
        #   e56febf1).  Two conflicting runtimes + stale thread state →
        #   NULL kmp_info_t* → crash at address 0x580.
        #
        #   Importing here instead means BOTH libomp instances are initialised
        #   fresh inside the child — no stale state, KMP_DUPLICATE_LIB_OK=TRUE
        #   keeps them from aborting on the duplicate.
        from src.hardware.camera.threads.threadCamera import threadCamera
        from src.hardware.camera.threads.threadLocalPerception import threadLocalPerception
        from src.hardware.camera.threads.threadVisualController import threadVisualController
        from src.behavior.planner import BehaviorPlanner
        from src.behavior.planner_thread import threadBehaviorPlanner
        from src.behavior.scenarios.crosswalk import Crosswalk
        from src.behavior.scenarios.highway import Highway
        from src.behavior.scenarios.intersection import Intersection
        from src.behavior.scenarios.lane_keep import LaneKeep
        from src.behavior.scenarios.overtake import Overtake
        from src.behavior.scenarios.parking import Parking
        from src.behavior.scenarios.roundabout import Roundabout
        from src.behavior.scenarios.stop_sign import StopSign
        from src.control.controller_thread import threadMotionController
        from src.control.motion_controller import MotionController
        from src.control.motor_command_dispatcher import threadMotorCommandDispatcher
        from src.control.safety_gate import SafetyGate
        from src.perception.lidar.thread_lidar import threadLidar
        from src.perception.tracking.object_tracker_thread import threadObjectTracker
        from src.perception.tracking.sort_tracker import MOTTracker

        _TRACKING_ENABLED = False
        _TRACKING_SHOW_WINDOW = False
        try:
            from src.routing.navigation_planner_thread import threadNavigationPlanner
            from src.localization.pose_estimator_thread import threadPoseEstimator
            from src.localization.relocalization_thread import TrackingState
            from src.routing.visualizer import TrackVisualizer
            import config as _cfg
            _TRACKING_ENABLED = True
            _TRACKING_SHOW_WINDOW = getattr(_cfg, "TRACKING_SHOW_WINDOW", True)
        except Exception as _tracking_import_err:
            print(f"\033[1;97m[ processCamera ] :\033[0m \033[1;93mWARNING\033[0m - Tracking not available: {_tracking_import_err}")

        # Shared event: when set, car is on highway — line following uses higher
        # speeds. Lo setea/clarea `ManeuverManager` (dentro de threadLineFollowing)
        # cuando observa highway_entrance/exit signs. Es lo único que sobrevive
        # de la familia de Events que coordinaba el viejo `signActions.py`:
        # `sign_action_event` y `steer_override_event` desaparecieron al
        # quedar inertes tras Phase 6 (motor writes salen exclusivamente del
        # `motor_command_dispatcher`).
        highway_mode_event = threading.Event()

        # Camera preview window: only if master switch AND individual toggle are on
        show_cam_preview = self.show_preview and self.debug_windows.get("camera_preview", False)
        camTh = threadCamera(
         self.queuesList, self.logging, self.debugging,
         show_preview=show_cam_preview,
         frame_buffer=self.frame_buffer,
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
        
        lane_observation_buffer = LatestValueBuffer()
        stopline_observation_buffer = LatestValueBuffer()
        pose_estimate_buffer = LatestValueBuffer()
        route_context_buffer = LatestValueBuffer()
        visual_candidate_buffer = LatestValueBuffer()
        visual_state_buffer = LatestValueBuffer()
        control_decision_buffer = LatestValueBuffer()
        detection_buffer = LatestValueBuffer()
        tracked_objects_buffer = LatestValueBuffer()
        lidar_scan_buffer = LatestValueBuffer()
        lidar_obstacles_buffer = LatestValueBuffer()
        sign_hints_buffer = LatestValueBuffer()

        lidarTh = threadLidar(
            self.queuesList,
            self.logging,
            self.debugging,
            lidar_scan_buffer=lidar_scan_buffer,
            lidar_obstacles_buffer=lidar_obstacles_buffer,
            pose_estimate_buffer=pose_estimate_buffer,
        )
        self.threads.append(lidarTh)

        # Local AI perception (TensorRT engine) now runs inside the camera process.
        # El frame anotado se escribe a ``overlay_buffer`` (intra-proceso, sin
        # encoding ni queue). El ``UdpVideoStreamerThread`` lo consume y lo
        # manda fragmentado por UDP al GUI — replaza el viejo path
        # serialCamera (JPEG → base64 → multiprocessing.Queue → SocketIO).
        from src.hardware.camera.threads.threadVideoStreamer import UdpVideoStreamerThread

        overlay_buffer = LatestFrameBuffer()

        localPerceptionTh = threadLocalPerception(
            self.queuesList, self.logging, self.debugging,
            frame_buffer=self.frame_buffer,
            local_lane_buffer=self.local_lane_buffer,
            detection_buffer=detection_buffer,
            lidar_scan_buffer=lidar_scan_buffer,
            pose_estimate_buffer=pose_estimate_buffer,
            sign_hints_buffer=sign_hints_buffer,
            overlay_buffer=overlay_buffer,
            enable_sign_detection=self.enable_sign_detection,
            enable_actions=self.sign_detection_actions,
            sign_min_confidence=self.sign_min_confidence,
            sign_min_box_area=self.sign_min_box_area,
            action_cooldown=self.sign_action_cooldown,
            stream_to_dashboard=self.stream_to_dashboard,
        )
        self.threads.append(localPerceptionTh)

        videoStreamerTh = UdpVideoStreamerThread(frame_buffer=overlay_buffer)
        self.threads.append(videoStreamerTh)
        self._video_streamer = videoStreamerTh  # exposed for handshake REP wiring (Sprint 4)

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
        # El runtime usa OSM/Lanelet como única fuente de verdad: el route
        # handler denso, el `LaneletMap` del behavior planner y el visualizador
        # comparten exactamente la misma geometría.
        tracking_state = None
        route_graph = None
        lanelet_map = None
        if _TRACKING_ENABLED:
            tracking_state = TrackingState()

            try:
                import os
                import config as _cfg_track
                from src.routing.lanelet.osm_router import OsmRouteGraph
                _lanelet2_osm = getattr(_cfg_track, "TRACKING_LANELET2_OSM", "")
                _step = getattr(_cfg_track, "TRACKING_WAYPOINT_STEP_M", 0.05)
                _start_lanelet_id = getattr(_cfg_track, "TRACKING_START_LANELET_ID", None)
                if not os.path.isabs(_lanelet2_osm):
                    _root = os.path.normpath(
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", "..")
                    )
                    _lanelet2_osm = os.path.join(_root, _lanelet2_osm)

                if _lanelet2_osm and os.path.exists(_lanelet2_osm):
                    route_graph = OsmRouteGraph(
                        _lanelet2_osm,
                        step_m=_step,
                        start_lanelet_id=_start_lanelet_id,
                    )
                    lanelet_map = route_graph.lanelet_map
            except Exception as _graph_err:
                print(f"[ processCamera ] WARNING - OSM route graph load failed: {_graph_err}")
                route_graph = None
                lanelet_map = None

            visualizer = None
            if _TRACKING_SHOW_WINDOW and route_graph is not None:
                try:
                    import os as _os_vis
                    _root = _os_vis.path.normpath(
                        _os_vis.path.join(_os_vis.path.dirname(_os_vis.path.abspath(__file__)),
                                          "..", "..", "..")
                    )
                    def _resolve(rel):
                        return rel if _os_vis.path.isabs(rel) else _os_vis.path.join(_root, rel)
                    _json_path   = _resolve(getattr(_cfg_track, "TRACKING_META_JSON", "track_meta.json"))
                    _svg_path    = _resolve(getattr(_cfg_track, "TRACKING_BG_SVG", ""))
                    _raster_path = _resolve(getattr(_cfg_track, "TRACKING_BG_RASTER", ""))
                    visualizer = TrackVisualizer(
                        route_graph,
                        bg_image_path=_raster_path,
                        bg_svg_path=_svg_path,
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
                # Phase 3: pasamos el `lanelet_map` para que el navigation
                # planner pueda popular `RouteContext.current_lanelet_id` vía
                # `at_pose(x, y)`. Sin esto, LaneKeep no puede construir un
                # target_path y el dispatcher emite speed=0 indefinidamente
                # (ese era el bug — feedback OK, comandos OK, pero el
                # BehaviorPlanner sin lanelet_id caía en `_fallback_plan`).
                lanelet_map=lanelet_map,
                sign_hints_buffer=sign_hints_buffer,
                logging=self.logging,
                debugging=self.debugging,
                visualizer=visualizer,
            )
            self.threads.append(poseEstimatorTh)
            self.threads.append(plannerTh)

        objectTrackerTh = threadObjectTracker(
            self.queuesList,
            tracker=MOTTracker(),
            detection_buffer=detection_buffer,
            pose_estimate_buffer=pose_estimate_buffer,
            tracked_objects_buffer=tracked_objects_buffer,
            logging=self.logging,
            debugging=self.debugging,
        )
        self.threads.append(objectTrackerTh)

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
            highway_mode_event=highway_mode_event,
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
        #   - El `lanelet_map` puede ser `None` cuando el OSM no está
        #     disponible — el planner_thread degrada a un `RouteContext`
        #     vacío y solo `LaneKeep` se activa.
        #   - Los buffers se comparten con pose_estimator, navigation_planner,
        #     lane_observer y motion_controller, todos creados arriba.
        try:
            import config as _cfg_behavior
            _behavior_dt_s = float(getattr(_cfg_behavior, "BEHAVIOR_DT_S", 0.05))
            _behavior_horizon_n = int(getattr(_cfg_behavior, "BEHAVIOR_HORIZON_N", 20))
            _behavior_nominal = float(getattr(_cfg_behavior, "BEHAVIOR_NOMINAL_SPEED_MPS", 0.10))
            _behavior_max = float(getattr(_cfg_behavior, "BEHAVIOR_MAX_SPEED_MPS", 0.10))
            _behavior_pause = float(getattr(_cfg_behavior, "BEHAVIOR_THREAD_PAUSE_S", 0.05))
        except Exception:
            _behavior_dt_s = 0.05
            _behavior_horizon_n = 20
            _behavior_nominal = 0.10
            _behavior_max = 0.10
            _behavior_pause = 0.05

        behavior_planner = BehaviorPlanner(
            scenarios=[
                StopSign(),
                Parking(),
                Crosswalk(),
                Intersection(),
                Roundabout(),
                Highway(),
                Overtake(),
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
            tracked_objects_buffer=tracked_objects_buffer,
            lidar_scan_buffer=lidar_scan_buffer,
            lidar_obstacles_buffer=lidar_obstacles_buffer,
            sign_hints_buffer=sign_hints_buffer,
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
        # `fallback_horizon_n` solo entra en juego si AcadosMPC no logra
        # cargar (sim/dev sin código C generado). El PurePursuitSolver lo
        # usa para reportar `solver.N` y satisfacer el chequeo dimensional
        # de `MotionController.compute()` contra el `BehaviorOutput`.
        motion_controller = MotionController(
            fallback_horizon_n=_behavior_horizon_n,
        )
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
            route_context_buffer=route_context_buffer,
            lidar_scan_buffer=lidar_scan_buffer,
            lidar_obstacles_buffer=lidar_obstacles_buffer,
            logging=self.logging,
            debugging=self.debugging,
        )
        self.threads.append(dispatcherTh)


# Standalone demo removed when the message backend was migrated to ZMQ —
# the old queue-based capture loop no longer reflects the production data
# path (frames go via UDP, not the bus). Run ``python3 main.py`` to
# exercise this process inside the real graph.
