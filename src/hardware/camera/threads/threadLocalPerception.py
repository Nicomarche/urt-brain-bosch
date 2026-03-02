import base64
import time

import cv2
import numpy as np

import config
from src.hardware.camera.threads.localPerceptionEngine import LocalPerceptionEngine
from src.hardware.camera.threads.signActions import SignActions
from src.statemachine.systemMode import SystemMode
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (
    LineFollowingConfig,
    LocalLanePerception,
    LocalPerceptionStatus,
    SignDetected,
    SignDetectionStatus,
    StateChange,
    serialCamera,
)
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber


class threadLocalPerception(ThreadWithStop):
    """Runs the local `best.pt` model and publishes lane/sign perception."""

    def __init__(self, queuesList, logger, debugger, show_debug=False, debug_windows=None,
                 enable_sign_detection=True, enable_actions=False,
                 sign_min_confidence=0.50, sign_min_box_area=0.01,
                 action_cooldown=15.0, sign_action_event=None,
                 highway_mode_event=None):
        super(threadLocalPerception, self).__init__(pause=0.05)
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        self.show_debug = show_debug
        self.debug_windows = debug_windows or {}

        self.enable_sign_detection = enable_sign_detection
        self.enable_actions = enable_actions
        self.sign_min_confidence = float(sign_min_confidence)
        self.sign_min_box_area = float(sign_min_box_area)
        self.is_sign_actions_active = False

        self.local_ai_interval = float(getattr(config, "LOCAL_AI_INTERVAL", 0.10))
        self.local_ai_model_path = str(getattr(config, "LOCAL_AI_MODEL_PATH", "models/lane_segmentation/best.pt"))
        self.local_ai_min_confidence = float(getattr(config, "LOCAL_AI_MIN_CONFIDENCE", 0.35))
        self.local_ai_imgsz = int(getattr(config, "LOCAL_AI_IMGSZ", 320))
        self.local_ai_device = str(getattr(config, "LOCAL_AI_DEVICE", "auto"))

        self.last_infer_time = 0.0
        self.last_status_time = 0.0
        self.fps_timer = time.time()
        self.frame_counter = 0
        self.current_fps = 0.0
        self.detection_count = 0
        self.last_sign_name = ""
        self._last_result = None

        self.serialCameraSubscriber = messageHandlerSubscriber(
            self.queuesList, serialCamera, "lastOnly", True
        )
        self.stateChangeSubscriber = messageHandlerSubscriber(
            self.queuesList, StateChange, "lastOnly", True
        )
        self.configSubscriber = messageHandlerSubscriber(
            self.queuesList, LineFollowingConfig, "lastOnly", True
        )

        self.localLaneSender = messageHandlerSender(self.queuesList, LocalLanePerception)
        self.localStatusSender = messageHandlerSender(self.queuesList, LocalPerceptionStatus)
        self.signDetectedSender = messageHandlerSender(self.queuesList, SignDetected)
        self.signStatusSender = messageHandlerSender(self.queuesList, SignDetectionStatus)

        self.sign_actions = SignActions(
            self.queuesList,
            sign_action_event=sign_action_event,
            action_cooldown=action_cooldown,
            highway_mode_event=highway_mode_event,
        )

        self.engine = self._build_engine()
        print(
            f"\033[1;97m[ Local AI ] :\033[0m \033[1;92mINFO\033[0m - "
            f"Thread ready (signs={'ON' if self.enable_sign_detection else 'OFF'}, "
            f"actions={'ON' if self.enable_actions else 'OFF'})"
        )

    def _build_engine(self):
        return LocalPerceptionEngine(
            model_path=self.local_ai_model_path,
            min_confidence=self.local_ai_min_confidence,
            imgsz=self.local_ai_imgsz,
            device=self.local_ai_device,
        )

    def _is_window_enabled(self, window_key):
        return self.show_debug and self.debug_windows.get(window_key, False)

    def state_change_handler(self):
        message = self.stateChangeSubscriber.receive()
        if message is None:
            return
        try:
            mode_dict = SystemMode[message].value
            camera_config = mode_dict.get("camera", {})
            sign_config = camera_config.get("signDetection", {})
            self.is_sign_actions_active = bool(sign_config.get("enabled", False))
        except (KeyError, TypeError):
            self.is_sign_actions_active = False

    def _check_config(self):
        cfg = self.configSubscriber.receive()
        if cfg is None:
            return

        reload_engine = False
        if "local_ai_interval" in cfg:
            try:
                self.local_ai_interval = max(0.02, float(cfg["local_ai_interval"]))
            except (TypeError, ValueError):
                pass

        if "local_ai_min_confidence" in cfg:
            try:
                self.local_ai_min_confidence = max(0.01, min(1.0, float(cfg["local_ai_min_confidence"])))
                reload_engine = True
            except (TypeError, ValueError):
                pass

        if "local_ai_imgsz" in cfg:
            try:
                self.local_ai_imgsz = max(64, int(float(cfg["local_ai_imgsz"])))
                reload_engine = True
            except (TypeError, ValueError):
                pass

        if "local_ai_device" in cfg:
            self.local_ai_device = str(cfg["local_ai_device"])
            reload_engine = True

        if reload_engine:
            self.engine = self._build_engine()

    def _publish_sign(self, detections, now):
        if not self.enable_sign_detection or not detections:
            return

        best = detections[0]
        confidence = float(best.get("confidence", 0.0))
        if confidence < self.sign_min_confidence:
            return

        sign_name = str(best.get("class", ""))
        box = best.get("box", [0, 0, 0, 0])
        if len(box) != 4:
            box = [0, 0, 0, 0]
        box_area = max(0.0, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])))

        self.detection_count += 1
        self.last_sign_name = sign_name
        self.signDetectedSender.send({
            "sign": sign_name,
            "confidence": round(confidence, 3),
            "box_area": round(box_area, 5),
            "timestamp": now,
        })

        if (
            self.enable_actions
            and self.is_sign_actions_active
            and box_area >= self.sign_min_box_area
            and sign_name in SignActions.ACTIONABLE_SIGNS
        ):
            self.sign_actions.execute(sign_name)

    def _publish_status(self, result, now):
        if now - self.last_status_time < 1.0:
            return
        self.last_status_time = now

        infer_ms = float(result.get("inference_time_ms", 0.0)) if result else 0.0
        frame_id = int(result.get("frame_id", 0)) if result else 0
        model_ready = bool(result.get("model_ready", False)) if result else False
        detections = result.get("detections", []) if result else []

        self.localStatusSender.send({
            "enabled": True,
            "model_ready": model_ready,
            "fps": round(self.current_fps, 1),
            "inference_time_ms": round(infer_ms, 1),
            "last_frame_id": frame_id,
            "last_sign": self.last_sign_name,
            "detections_count": len(detections),
        })

        self.signStatusSender.send({
            "enabled": self.is_sign_actions_active,
            "fps": round(self.current_fps, 1),
            "last_sign": self.last_sign_name,
            "total_detections": self.detection_count,
            "server_connected": False,
            "local_model_ready": model_ready,
        })

    def _show_debug_windows(self, result):
        if not self.show_debug or not result:
            return

        lane_debug = result.get("lane_debug", {})
        if self._is_window_enabled("ai_local_overlay"):
            overlay = lane_debug.get("overlay")
            if overlay is not None:
                cv2.imshow("AI Local - Overlay", overlay)
        if self._is_window_enabled("ai_local_masks"):
            masks = lane_debug.get("masks")
            if masks is not None:
                cv2.imshow("AI Local - Masks", masks)
        if self._is_window_enabled("ai_local_signs"):
            signs = lane_debug.get("signs")
            if signs is not None:
                cv2.imshow("AI Local - Signs", signs)
        cv2.waitKey(1)

    def thread_work(self):
        self.state_change_handler()
        self._check_config()
        camera_message = self.serialCameraSubscriber.receive()
        if camera_message is None:
            time.sleep(0.02)
            return

        now = time.time()
        if now - self.last_infer_time < self.local_ai_interval:
            return
        self.last_infer_time = now

        try:
            img_data = base64.b64decode(camera_message)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return

            result = self.engine.infer(frame)
            self._last_result = result
            self.frame_counter += 1

            elapsed = now - self.fps_timer
            if elapsed >= 1.0:
                self.current_fps = self.frame_counter / elapsed
                self.frame_counter = 0
                self.fps_timer = now

            lane_points = result.get("lane_points", [])
            lane_side_points = result.get("lane_side_points", {"left": [], "right": []})
            self.localLaneSender.send({
                "lane_points": lane_points,
                "lane_side_points": lane_side_points,
                "inference_time_ms": float(result.get("inference_time_ms", 0.0)),
                "frame_id": int(result.get("frame_id", 0)),
                "timestamp": now,
                "lane_count": len(lane_points),
                "model_ready": bool(result.get("model_ready", False)),
            })

            self._publish_sign(result.get("detections", []), now)
            self._publish_status(result, now)
            self._show_debug_windows(result)
        except Exception as e:
            print(f"\033[1;97m[ Local AI ] :\033[0m \033[1;91mERROR\033[0m - {e}")

    def stop(self):
        for name in ("AI Local - Overlay", "AI Local - Masks", "AI Local - Signs"):
            try:
                cv2.destroyWindow(name)
            except Exception:
                pass
        super(threadLocalPerception, self).stop()
