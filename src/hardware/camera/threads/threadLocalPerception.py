import time

import cv2

import config
from src.hardware.camera.threads.localPerceptionEngine import LocalPerceptionEngine
from src.hardware.camera.threads.signActions import SignActions
from src.statemachine.systemMode import SystemMode
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (
    LineFollowingConfig,
    LineFollowingStatus,
    LocalLanePerception,
    LocalPerceptionStatus,
    SignDetected,
    SignDetectionStatus,
    StateChange,
)
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber


class threadLocalPerception(ThreadWithStop):
    """Runs the local lane/sign model and publishes perception outputs."""

    def __init__(self, queuesList, logger, debugger, frame_buffer=None,
                 show_debug=False, debug_windows=None,
                 enable_sign_detection=True, enable_actions=False,
                 sign_min_confidence=0.50, sign_min_box_area=0.01,
                 action_cooldown=15.0, sign_action_event=None,
                 highway_mode_event=None):
        # Keep scheduler granularity tight so local_ai_interval can reach high FPS targets.
        super(threadLocalPerception, self).__init__(pause=0.001)
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        self.frame_buffer = frame_buffer
        self.show_debug = show_debug
        self.debug_windows = debug_windows or {}

        self.enable_sign_detection = enable_sign_detection
        self.enable_actions = enable_actions
        self.sign_min_confidence = float(sign_min_confidence)
        self.sign_min_box_area = float(sign_min_box_area)
        self.is_sign_actions_active = False

        self.local_ai_interval = float(getattr(config, "LOCAL_AI_INTERVAL", 0.10))
        self.local_ai_model_path = str(
            getattr(config, "LOCAL_AI_MODEL_PATH", "models/lane_segmentation/Best416px.engine")
        )
        self.local_ai_min_confidence = float(getattr(config, "LOCAL_AI_MIN_CONFIDENCE", 0.35))
        self.local_ai_imgsz = int(getattr(config, "LOCAL_AI_IMGSZ", 416))
        self.local_ai_device = str(getattr(config, "LOCAL_AI_DEVICE", "auto"))

        self.last_infer_time = 0.0
        self.last_status_time = 0.0
        self.fps_timer = time.time()
        self.frame_counter = 0
        self.current_fps = 0.0
        self.detection_count = 0
        self.last_sign_name = ""
        self._last_result = None
        self._last_frame_sequence = 0
        self._preview_interval = 1.0 / 5.0
        self._last_preview_time = 0.0
        self._lf_curve_state = "STRAIGHT"
        self._lf_curve_state_frames = 0
        self._lf_steering_deg = 0.0

        self.stateChangeSubscriber = messageHandlerSubscriber(
            self.queuesList, StateChange, "lastOnly", True
        )
        self.configSubscriber = messageHandlerSubscriber(
            self.queuesList, LineFollowingConfig, "lastOnly", True
        )
        self.lineFollowingStatusSubscriber = messageHandlerSubscriber(
            self.queuesList, LineFollowingStatus, "lastOnly", True
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

    def _should_build_debug(self, now):
        if not self.show_debug:
            return False
        if not any(
            self._is_window_enabled(window_key)
            for window_key in ("ai_local_overlay", "ai_local_masks", "ai_local_signs")
        ):
            return False
        return (now - self._last_preview_time) >= self._preview_interval

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

    def _poll_line_following_context(self):
        status = self.lineFollowingStatusSubscriber.receive()
        if not isinstance(status, dict):
            return

        curve_state = status.get("curve_state")
        if curve_state is not None:
            self._lf_curve_state = str(curve_state)

        curve_state_frames = status.get("curve_state_frames")
        if curve_state_frames is not None:
            try:
                self._lf_curve_state_frames = int(curve_state_frames)
            except (TypeError, ValueError):
                pass

        steering_deg = status.get("commanded_steering")
        if steering_deg is None:
            steering_deg = status.get("steering")
        if steering_deg is not None:
            try:
                self._lf_steering_deg = float(steering_deg)
            except (TypeError, ValueError):
                pass

    def _publish_sign(self, detections, now):
        if not self.enable_sign_detection or not detections:
            return

        best = detections[0]
        confidence = float(best.get("confidence", 0.0))
        if confidence < self.sign_min_confidence:
            return

        raw_sign_name = str(best.get("class", ""))
        sign_name = SignActions.normalize_sign_name(raw_sign_name) or raw_sign_name
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

        is_close = box_area >= self.sign_min_box_area
        is_actionable = SignActions.is_actionable_sign(sign_name)
        sign_display = raw_sign_name if raw_sign_name == sign_name else f"{raw_sign_name}->{sign_name}"
        print(
            f"\033[1;97m[ Local AI ] :\033[0m \033[1;96mDETECTED\033[0m - "
            f"{sign_display} ({confidence:.1%}) box={box_area:.3%}"
            f"{'' if is_close else f' (TOO FAR <{self.sign_min_box_area:.1%})'}"
            f"{'' if self.enable_actions else ' [actions=OFF]'}"
            f"{'' if self.is_sign_actions_active else ' [inactive_mode]'}"
            f"{'' if is_actionable else ' [not_actionable]'}"
        )

        if (
            self.enable_actions
            and self.is_sign_actions_active
            and is_close
            and is_actionable
        ):
            self.sign_actions.execute(
                sign_name,
                curve_state=self._lf_curve_state,
                steering_deg=self._lf_steering_deg,
            )

    def _publish_status(self, result, now):
        if now - self.last_status_time < 1.0:
            return
        self.last_status_time = now

        infer_ms = float(result.get("inference_time_ms", 0.0)) if result else 0.0
        processing_fps = (1000.0 / infer_ms) if infer_ms > 0.0 else 0.0
        target_fps = (1.0 / self.local_ai_interval) if self.local_ai_interval > 0.0 else 0.0
        frame_id = int(result.get("frame_id", 0)) if result else 0
        model_ready = bool(result.get("model_ready", False)) if result else False
        detections = result.get("detections", []) if result else []

        self.localStatusSender.send({
            "enabled": True,
            "model_ready": model_ready,
            "fps": round(self.current_fps, 1),
            "processing_fps": round(processing_fps, 1),
            "target_fps": round(target_fps, 1),
            "inference_time_ms": round(infer_ms, 1),
            "last_frame_id": frame_id,
            "last_sign": self.last_sign_name,
            "detections_count": len(detections),
        })

        self.signStatusSender.send({
            "enabled": self.is_sign_actions_active,
            "fps": round(self.current_fps, 1),
            "processing_fps": round(processing_fps, 1),
            "target_fps": round(target_fps, 1),
            "last_sign": self.last_sign_name,
            "total_detections": self.detection_count,
            "server_connected": False,
            "local_model_ready": model_ready,
        })

    def _annotate_debug_frame(self, frame, result):
        if frame is None or result is None or not hasattr(frame, "shape"):
            return frame

        height, width = frame.shape[:2]
        header_height = 54
        infer_ms = float(result.get("inference_time_ms", 0.0) or 0.0)
        processing_fps = (1000.0 / infer_ms) if infer_ms > 0.0 else 0.0
        throughput_fps = max(0.0, float(self.current_fps or 0.0))
        target_fps = (1.0 / self.local_ai_interval) if self.local_ai_interval > 0.0 else 0.0
        lane_count = len(result.get("lane_points", []) or [])
        sign_count = len(result.get("detections", []) or [])
        frame_id = int(result.get("frame_id", 0) or 0)

        cv2.rectangle(frame, (0, 0), (width, header_height), (0, 0, 0), -1)
        cv2.putText(
            frame,
            (
                f"AI LOCAL | infer:{infer_ms:.0f}ms | proc:{processing_fps:.1f} FPS "
                f"| thr:{throughput_fps:.1f} FPS"
            ),
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"cap:{target_fps:.1f} FPS | lanes:{lane_count} | signs:{sign_count} | frame:{frame_id}",
            (8, min(height - 6, 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _show_debug_windows(self, result, now):
        if not self.show_debug or not result:
            return

        lane_debug = result.get("lane_debug", {})
        rendered = False
        if self._is_window_enabled("ai_local_overlay"):
            overlay = lane_debug.get("overlay")
            if overlay is not None:
                overlay = self._annotate_debug_frame(overlay, result)
                cv2.imshow("AI Local - Overlay", overlay)
                rendered = True
        if self._is_window_enabled("ai_local_masks"):
            masks = lane_debug.get("masks")
            if masks is not None:
                masks = self._annotate_debug_frame(masks, result)
                cv2.imshow("AI Local - Masks", masks)
                rendered = True
        if self._is_window_enabled("ai_local_signs"):
            signs = lane_debug.get("signs")
            if signs is not None:
                signs = self._annotate_debug_frame(signs, result)
                cv2.imshow("AI Local - Signs", signs)
                rendered = True
        if rendered:
            cv2.waitKey(1)
            self._last_preview_time = now

    def thread_work(self):
        self.state_change_handler()
        self._check_config()
        self._poll_line_following_context()
        if self.enable_actions and self.is_sign_actions_active:
            self.sign_actions.tick(
                curve_state=self._lf_curve_state,
                steering_deg=self._lf_steering_deg,
            )

        if self.frame_buffer is None:
            time.sleep(0.02)
            return

        frame, frame_timestamp, frame_sequence = self.frame_buffer.read_latest(copy_frame=True)
        if frame is None:
            time.sleep(0.02)
            return
        if frame_sequence == self._last_frame_sequence:
            return

        now = time.time()
        if now - self.last_infer_time < self.local_ai_interval:
            return
        self.last_infer_time = now
        self._last_frame_sequence = frame_sequence

        try:
            build_debug = self._should_build_debug(now)
            result = self.engine.infer(frame, build_debug=build_debug)
            result_timestamp = time.time()
            self._last_result = result
            self.frame_counter += 1

            elapsed = now - self.fps_timer
            if elapsed >= 1.0:
                self.current_fps = self.frame_counter / elapsed
                self.frame_counter = 0
                self.fps_timer = now

            lane_points = result.get("lane_points", [])
            lane_side_points = result.get("lane_side_points", {"left": [], "right": []})
            lane_side_lines = result.get("lane_side_lines", {"left": [], "right": []})
            lane_side_sources = result.get("lane_side_sources", {"left": "none", "right": "none"})
            side_masks = result.get("side_masks", {"left": None, "right": None})
            lane_mask = result.get("lane_mask")
            self.localLaneSender.send({
                "lane_points": lane_points,
                "lane_side_points": lane_side_points,
                "lane_side_lines": lane_side_lines,
                "lane_side_sources": lane_side_sources,
                "side_masks": side_masks,
                "lane_mask": lane_mask,
                "inference_time_ms": float(result.get("inference_time_ms", 0.0)),
                "frame_id": int(result.get("frame_id", 0)),
                # timestamp now represents "result published" time; keep source frame time separately.
                "result_timestamp": result_timestamp,
                "timestamp": result_timestamp,
                "source_frame_timestamp": frame_timestamp or 0.0,
                "source_frame_sequence": int(frame_sequence or 0),
                "lane_count": len(lane_points),
                "model_ready": bool(result.get("model_ready", False)),
                "heading_hint_rad": float(result.get("heading_hint_rad", 0.0) or 0.0),
                "heading_hint_confidence": float(result.get("heading_hint_confidence", 0.0) or 0.0),
                "heading_hint_source": str(result.get("heading_hint_source", "none") or "none"),
                "road_type_class": str(result.get("road_type_class", "unknown") or "unknown"),
                "road_type_confidence": float(result.get("road_type_confidence", 0.0) or 0.0),
                "road_type_source": result.get("road_type_source"),
                "lead_distance_class": str(result.get("lead_distance_class", "none") or "none"),
                "lead_distance_confidence": float(result.get("lead_distance_confidence", 0.0) or 0.0),
                "lead_distance_area": float(result.get("lead_distance_area", 0.0) or 0.0),
                "lead_distance_source": result.get("lead_distance_source"),
            })

            self._publish_sign(result.get("detections", []), now)
            self._publish_status(result, now)
            if build_debug:
                self._show_debug_windows(result, now)
        except Exception as e:
            print(f"\033[1;97m[ Local AI ] :\033[0m \033[1;91mERROR\033[0m - {e}")

    def stop(self):
        for name in ("AI Local - Overlay", "AI Local - Masks", "AI Local - Signs"):
            try:
                cv2.destroyWindow(name)
            except Exception:
                pass
        super(threadLocalPerception, self).stop()
