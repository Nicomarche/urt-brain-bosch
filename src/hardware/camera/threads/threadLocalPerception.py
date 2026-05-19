import math
import time

import cv2

import config
from src.hardware.camera.threads.localPerceptionEngine import LocalPerceptionEngine
from src.hardware.camera.threads.signActions import SignActions
from src.hardware.camera.threads.trafficLightClassifier import TrafficLightClassifier
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

    PARKING_SIGN_CLASSES = frozenset({"parking", "parking_sign"})
    PARKING_AREA_CLASSES = frozenset({"parking_area", "parking_spot"})
    TRAFFIC_LIGHT_CLASSES = frozenset({
        "traffic_light",
        "traffic_light_unknown",
        "red",
        "yellow",
        "green",
        "red_light",
        "yellow_light",
        "green_light",
    })

    def __init__(self, queuesList, logger, debugger, frame_buffer=None,
                 show_debug=False, debug_windows=None,
                 enable_sign_detection=True, enable_actions=False,
                 sign_min_confidence=0.50, sign_min_box_area=0.01,
                 action_cooldown=15.0, sign_action_event=None,
                 highway_mode_event=None, steer_override_event=None):
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
        self.sign_min_box_area_per_sign = dict(
            getattr(config, "SIGN_MIN_BOX_AREA_PER_SIGN", {})
        )
        self.is_sign_actions_active = False
        self.traffic_light_opencv_enabled = bool(
            getattr(config, "TRAFFIC_LIGHT_OPENCV_ENABLED", True)
        )
        self.traffic_light_min_box_area = float(
            getattr(config, "TRAFFIC_LIGHT_MIN_BOX_AREA", self.sign_min_box_area)
        )
        self.traffic_light_classifier = (
            TrafficLightClassifier() if self.traffic_light_opencv_enabled else None
        )

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
        self._preview_interval = 1.0 / 25.0
        self._last_preview_time = 0.0
        self._lf_curve_state = "STRAIGHT"
        self._lf_curve_state_frames = 0
        self._lf_steering_deg = 0.0
        self._current_mode = ""
        self._px_per_cm = 0.0       # cached from LineFollowingStatus.stanley_px_per_cm
        self._last_frame_shape = None  # (height, width) of last processed frame

        # Walk-area pedestrian stop logic
        self._walk_area_min_box_area      = float(getattr(config, "WALK_AREA_MIN_BOX_AREA",  0.04))
        self._walk_area_slow_speed        = float(getattr(config, "WALK_AREA_SLOW_SPEED_CM_S", 10.0))
        self._walk_area_clear_grace       = float(getattr(config, "WALK_AREA_CLEAR_GRACE", 0.5))
        self._walk_area_active            = False   # True while handling a walk_area (slow/stop)
        self._walk_area_mode              = None    # "slow" or "stop"
        self._walk_area_last_seen         = 0.0
        self._parking_sign_cooldown       = float(getattr(config, "PARKING_SIGN_COOLDOWN", 8.0))
        self._parking_sign_last_triggered = 0.0

        # AUTO-mode pedestrian/obstacle stop (independent of walk_area)
        self._pedestrian_stop_active = False

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
        self.stateChangeSender = messageHandlerSender(self.queuesList, StateChange)

        self.sign_actions = SignActions(
            self.queuesList,
            sign_action_event=sign_action_event,
            action_cooldown=action_cooldown,
            highway_mode_event=highway_mode_event,
            steer_override_event=steer_override_event,
            crosswalk_done_callback=self._on_crosswalk_done,
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
            self._current_mode = mode_dict.get("mode", "").lower()
            camera_config = mode_dict.get("camera", {})
            sign_config = camera_config.get("signDetection", {})
            self.is_sign_actions_active = bool(sign_config.get("enabled", False))
        except (KeyError, TypeError):
            self._current_mode = ""
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

        # Keep SignActions' cruise speed in sync with what line-following is
        # actually commanding, so hardcoded maneuvers (e.g. roundabout) can
        # hold the current speed instead of falling back to BASE_SPEED.
        cruise_speed = status.get("commanded_speed")
        if cruise_speed is None:
            cruise_speed = status.get("speed")
        if cruise_speed is not None:
            try:
                cruise_val = float(cruise_speed)
                if cruise_val > 0.0:
                    self.sign_actions.current_speed = cruise_val
            except (TypeError, ValueError):
                pass

        px_per_cm = status.get("stanley_px_per_cm")
        if px_per_cm is not None:
            try:
                val = float(px_per_cm)
                if val > 0.1:
                    self._px_per_cm = val
            except (TypeError, ValueError):
                pass

    def _estimate_sign_distance_cm(self, box, img_height):
        """Estimate forward distance to a ground-level object from its bounding box.

        Uses a pinhole perspective camera model:

            d = H × (cos β − P·sin β) / (P·cos β + sin β)

        where:
            P   = (base_y_px − center_y) / f_y   [normalised image coordinate]
            H   = CAMERA_HEIGHT_CM  (camera height above ground, 17 cm)
            β   = CAMERA_PITCH_DEG  (downward tilt from horizontal, ~16.4°)
            f_y = CAMERA_FY_480 × (img_height / 480)  (vertical focal length)

        Camera parameters are derived from IMX219 (RPi Cam v2) specs:
            Capture 1280×720 via Jetson CSI (2×2 binned crop of 2560×1440),
            then non-uniformly scaled to 640×480. The resulting f_y at 480px
            is 905 px; pitch is back-calculated from the observable px_per_cm
            and camera height (verified: β ≈ 16.4° places reference_y at 288 px
            for d_ref ≈ 48 cm, matching the lane-following calibration).

        A final PARKING_DISTANCE_SCALE_FACTOR (default 1.0) can fine-tune
        the result after physical measurement.

        Returns None if calibration data is not yet available.
        """
        if img_height is None or img_height < 1:
            return None
        try:
            H       = float(getattr(config, "CAMERA_HEIGHT_CM",  17.0))
            beta    = math.radians(float(getattr(config, "CAMERA_PITCH_DEG", 16.4)))
            fy_480  = float(getattr(config, "CAMERA_FY_480",     905.0))
            scale   = float(getattr(config, "PARKING_DISTANCE_SCALE_FACTOR", 1.0))

            f_y     = fy_480 * (float(img_height) / 480.0)
            center_y = float(img_height) / 2.0

            base_y_px = float(box[2]) * float(img_height)   # bottom of bbox in px
            P = (base_y_px - center_y) / f_y                # normalised coord

            cos_b = math.cos(beta)
            sin_b = math.sin(beta)
            denom = P * cos_b + sin_b
            if abs(denom) < 1e-6:
                return None

            distance_cm = H * (cos_b - P * sin_b) / denom
            if distance_cm < 0.0:
                return None

            return max(0.0, round(distance_cm * scale, 1))
        except (TypeError, ValueError, IndexError, ZeroDivisionError):
            return None

    def _on_crosswalk_done(self):
        """Called by SignActions after a crosswalk action completes (unused for now)."""
        pass

    def _enter_parking_mode(self):
        """Send a StateChange PARKING when in AUTO mode to start the parking sequence."""
        if self._current_mode != "auto":
            return
        now = time.time()
        if now - self._parking_sign_last_triggered < self._parking_sign_cooldown:
            return
        self._parking_sign_last_triggered = now
        print(
            f"\033[1;97m[ Local AI ] :\033[0m \033[1;92mPARKING_SIGN→PARKING\033[0m - "
            f"Parking sign detectado en modo AUTO, cambiando a modo PARKING"
        )
        try:
            self.stateChangeSender.send("PARKING")
        except Exception as e:
            print(
                f"\033[1;97m[ Local AI ] :\033[0m \033[1;91mERROR\033[0m - "
                f"No se pudo enviar StateChange PARKING: {e}"
            )

    def _handle_walk_area(self, detections, now):
        """Slow down in empty walk_area; stop only when pedestrians/obstacles are present.

        Rules:
        - Only applies in AUTO mode; MANUAL/PARKING must not be overridden.
        - Only trigger when the walk_area bbox area >= WALK_AREA_MIN_BOX_AREA
          (filters detections that are too far away).
        - If obstacle/pedestrian is visible in the walk_area → stop the car.
        - If the walk_area is clear → hold speed at WALK_AREA_SLOW_SPEED_CM_S.
        - Once the walk_area is no longer visible/close → return control to line following.
        """
        if self._current_mode != "auto":
            if self._walk_area_active:
                self._walk_area_active = False
                self._walk_area_mode = None
                if self.sign_actions.sign_action_event:
                    self.sign_actions.sign_action_event.clear()
            return

        _OBSTACLE_CLASSES = frozenset({
            "obstacle", "pedestrian", "person", "yaya", "human",
        })
        _WALK_AREA_CLASSES = frozenset({
            "walk_area", "walk area", "zebra_crossing", "zebra crossing",
        })

        # Find best (largest) walk_area detection and its box area
        best_walk_area_box_area = 0.0
        for d in detections:
            if str(d.get("class", "")).strip().lower() not in _WALK_AREA_CLASSES:
                continue
            box = d.get("box", [])
            if len(box) == 4:
                try:
                    area = max(0.0, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])))
                    best_walk_area_box_area = max(best_walk_area_box_area, area)
                except (TypeError, ValueError):
                    pass

        walk_area_close = best_walk_area_box_area >= self._walk_area_min_box_area
        obstacle_seen = any(
            str(d.get("class", "")).strip().lower() in _OBSTACLE_CLASSES
            for d in detections
        )

        if not walk_area_close:
            if self._walk_area_active and now - self._walk_area_last_seen >= self._walk_area_clear_grace:
                self._walk_area_active = False
                self._walk_area_mode = None
                if self.sign_actions.sign_action_event:
                    self.sign_actions.sign_action_event.clear()
                print(
                    f"\033[1;97m[ Local AI ] :\033[0m \033[1;92mWALK_AREA\033[0m - "
                    f"Walk area pasada, velocidad normal"
                )
            return

        self._walk_area_last_seen = now

        if obstacle_seen:
            if not self._walk_area_active or self._walk_area_mode != "stop":
                self._walk_area_active = True
                self._walk_area_mode = "stop"
                if self.sign_actions.sign_action_event:
                    self.sign_actions.sign_action_event.set()
                print(
                    f"\033[1;97m[ Local AI ] :\033[0m \033[1;91mWALK_AREA\033[0m - "
                    f"Peatón/obstáculo en walk area (box={best_walk_area_box_area:.1%}), auto detenido"
                )
            self.sign_actions._send_speed(0)
            return

        if not self._walk_area_active or self._walk_area_mode != "slow":
            self._walk_area_active = True
            self._walk_area_mode = "slow"
            if self.sign_actions.sign_action_event:
                self.sign_actions.sign_action_event.set()
            print(
                f"\033[1;97m[ Local AI ] :\033[0m \033[1;93mWALK_AREA\033[0m - "
                f"Walk area libre (box={best_walk_area_box_area:.1%}), velocidad {self._walk_area_slow_speed:.0f} cm/s"
            )

        self.sign_actions._send_speed(self._walk_area_slow_speed)

    def _handle_pedestrian_obstacle(self, detections, now):
        """In AUTO mode, freeze the car while a pedestrian/obstacle is visible.

        Resumes immediately when the frame no longer contains any
        pedestrian/obstacle. Independent of walk_area handling — that one has
        its own timers and PARKING transition.
        """
        _OBSTACLE_CLASSES = frozenset({
            "obstacle", "pedestrian", "person", "yaya", "human",
        })

        in_auto = (self._current_mode == "auto")

        # If we are not in AUTO anymore, drop any active stop we own.
        if not in_auto:
            if self._pedestrian_stop_active:
                self._pedestrian_stop_active = False
                # Only release the shared event if walk_area isn't also using it.
                if not self._walk_area_active and self.sign_actions.sign_action_event:
                    self.sign_actions.sign_action_event.clear()
            return

        # Walk-area handler already controls the stop in this frame.
        if self._walk_area_active:
            return

        # A blocking sign action (stop/red_light/crosswalk) is in progress —
        # don't interfere with its speed control.
        if self.sign_actions._is_blocking_action_running():
            return

        obstacle_seen = any(
            str(d.get("class", "")).strip().lower() in _OBSTACLE_CLASSES
            for d in detections
        )

        if obstacle_seen:
            if not self._pedestrian_stop_active:
                self._pedestrian_stop_active = True
                if self.sign_actions.sign_action_event:
                    self.sign_actions.sign_action_event.set()
                print(
                    f"\033[1;97m[ Local AI ] :\033[0m \033[1;91mAUTO_STOP\033[0m - "
                    f"Pedestrian/obstacle detectado, frenando hasta que despeje"
                )
            self.sign_actions._send_speed(0)
        elif self._pedestrian_stop_active:
            self._pedestrian_stop_active = False
            if self.sign_actions.sign_action_event:
                self.sign_actions.sign_action_event.clear()
            print(
                f"\033[1;97m[ Local AI ] :\033[0m \033[1;92mAUTO_STOP\033[0m - "
                f"Vía despejada, reanudando marcha"
            )

    def _normalized_detection_class(self, detection):
        raw_name = str(detection.get("class", "") or "")
        normalized = SignActions.normalize_sign_name(raw_name) or raw_name
        sign_map = getattr(config, "LOCAL_AI_SIGN_CLASS_MAP", {})
        mapped = sign_map.get(normalized)
        if mapped is not None:
            return mapped
        for alias, canonical in sign_map.items():
            if SignActions.normalize_sign_name(alias) == normalized:
                return canonical
        return normalized

    def _box_area(self, detection):
        box = detection.get("box", [])
        if len(box) != 4:
            return 0.0
        try:
            return max(0.0, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])))
        except (TypeError, ValueError):
            return 0.0

    def _best_detection_for_classes(self, detections, target_classes):
        best = None
        best_score = -1.0
        for detection in detections:
            sign_name = self._normalized_detection_class(detection)
            if sign_name not in target_classes:
                continue
            confidence = float(detection.get("confidence", 0.0) or 0.0)
            box_area = self._box_area(detection)
            score = confidence * max(box_area, 1e-6)
            if score > best_score:
                best = detection
                best_score = score
        return best

    def _handle_parking_sign(self, detections):
        """Enter parking mode only from the parking sign, not from the parking area."""
        parking_sign = self._best_detection_for_classes(detections, self.PARKING_SIGN_CLASSES)
        if parking_sign is None:
            return
        confidence = float(parking_sign.get("confidence", 0.0) or 0.0)
        if confidence < self.sign_min_confidence:
            return
        box_area = self._box_area(parking_sign)
        effective_min_box = self.sign_min_box_area_per_sign.get(
            "parking_sign", self.sign_min_box_area
        )
        if box_area < effective_min_box:
            return
        self._enter_parking_mode()

    def _best_traffic_light_detection(self, detections):
        detection = self._best_detection_for_classes(detections, self.TRAFFIC_LIGHT_CLASSES)
        if detection is None:
            return None
        confidence = float(detection.get("confidence", 0.0) or 0.0)
        if confidence < self.sign_min_confidence:
            return None
        min_box_area = float(
            getattr(self, "traffic_light_min_box_area", self.sign_min_box_area)
        )
        if self._box_area(detection) < min_box_area:
            return None
        return detection

    def _classify_traffic_light_detection(self, sign_name, box, frame):
        sign_name = TrafficLightClassifier.normalize_sign_name(sign_name)
        if getattr(self, "traffic_light_opencv_enabled", True):
            if getattr(self, "traffic_light_classifier", None) is None:
                self.traffic_light_classifier = TrafficLightClassifier()

            if frame is not None:
                traffic_light_info = self.traffic_light_classifier.classify(frame, box)
                if traffic_light_info.get("sign") != TrafficLightClassifier.UNKNOWN_SIGN:
                    return self._coerce_traffic_light_no_yellow(traffic_light_info)

        if sign_name in {"red", "green", "red_light", "green_light"}:
            return TrafficLightClassifier.payload_for_known_sign(sign_name)
        if sign_name in {"yellow", "yellow_light"}:
            return self._coerce_traffic_light_no_yellow(
                TrafficLightClassifier.payload_for_known_sign(sign_name)
            )

        if not getattr(self, "traffic_light_opencv_enabled", True):
            return {
                "sign": TrafficLightClassifier.UNKNOWN_SIGN,
                "color": TrafficLightClassifier.UNKNOWN_COLOR,
                "state": TrafficLightClassifier.UNKNOWN_SIGN,
                "reason": "opencv_disabled",
                "scores": {},
            }

        if frame is None:
            return {
                "sign": TrafficLightClassifier.UNKNOWN_SIGN,
                "color": TrafficLightClassifier.UNKNOWN_COLOR,
                "state": TrafficLightClassifier.UNKNOWN_SIGN,
                "reason": "missing_frame",
                "scores": {},
            }

        return self._coerce_traffic_light_no_yellow(
            self.traffic_light_classifier.classify(frame, box)
        )

    def _coerce_traffic_light_no_yellow(self, traffic_light_info):
        """This track has only red/green lights; yellow is treated as a red hold."""
        if not isinstance(traffic_light_info, dict):
            return traffic_light_info
        sign = TrafficLightClassifier.normalize_sign_name(
            traffic_light_info.get("sign") or traffic_light_info.get("state")
        )
        color = str(traffic_light_info.get("color") or "").strip().lower()
        if sign not in {"yellow", "yellow_light"} and color != "yellow":
            return traffic_light_info

        coerced = dict(traffic_light_info)
        coerced["source_sign"] = coerced.get("sign", sign)
        coerced["source_color"] = coerced.get("color", color)
        coerced["sign"] = "red_light"
        coerced["state"] = "red_light"
        coerced["color"] = "red"
        reason = str(coerced.get("reason") or "")
        coerced["reason"] = f"{reason}+yellow_as_red" if reason else "yellow_as_red"
        return coerced

    def _publish_sign(self, detections, now, img_shape=None, frame=None):
        if not self.enable_sign_detection or not detections:
            return

        best = detections[0]
        if self._current_mode == "parking":
            parking_area = self._best_detection_for_classes(
                detections, self.PARKING_AREA_CLASSES
            )
            if parking_area is not None:
                best = parking_area
        else:
            traffic_light = self._best_traffic_light_detection(detections)
            if traffic_light is not None:
                best = traffic_light
        confidence = float(best.get("confidence", 0.0))
        if confidence < self.sign_min_confidence:
            return

        raw_sign_name = str(best.get("class", ""))
        sign_name = self._normalized_detection_class(best)
        box = best.get("box", [0, 0, 0, 0])
        if len(box) != 4:
            box = [0, 0, 0, 0]
        box_area = max(0.0, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])))

        img_height = img_shape[0] if img_shape is not None else None
        distance_cm = self._estimate_sign_distance_cm(box, img_height)
        traffic_light_info = None
        if TrafficLightClassifier.is_traffic_light_sign(sign_name):
            traffic_light_info = self._classify_traffic_light_detection(sign_name, box, frame)
            sign_name = traffic_light_info.get("sign") or TrafficLightClassifier.UNKNOWN_SIGN

        self.detection_count += 1
        self.last_sign_name = sign_name
        payload = {
            "sign": sign_name,
            "confidence": round(confidence, 3),
            "box": [round(float(v), 5) for v in box],
            "box_area": round(box_area, 5),
            "distance_cm": distance_cm,
            "timestamp": now,
        }
        if traffic_light_info is not None:
            payload.update({
                "traffic_light_color": traffic_light_info.get("color", "unknown"),
                "traffic_light_state": traffic_light_info.get("state", sign_name),
                "traffic_light_reason": traffic_light_info.get("reason", ""),
                "traffic_light_scores": traffic_light_info.get("scores", {}),
                "traffic_light_source_sign": raw_sign_name,
            })
            crop = traffic_light_info.get("crop")
            if isinstance(crop, dict):
                payload["traffic_light_crop"] = crop
        self.signDetectedSender.send(payload)

        if traffic_light_info is not None:
            effective_min_box = float(
                getattr(self, "traffic_light_min_box_area", self.sign_min_box_area)
            )
        else:
            effective_min_box = self.sign_min_box_area_per_sign.get(
                sign_name, self.sign_min_box_area
            )
        is_close = box_area >= effective_min_box
        is_actionable = SignActions.is_actionable_sign(sign_name)
        sign_display = raw_sign_name if raw_sign_name == sign_name else f"{raw_sign_name}->{sign_name}"
        dist_str = f" ~{distance_cm:.0f}cm" if distance_cm is not None else ""
        traffic_str = ""
        if traffic_light_info is not None:
            traffic_str = (
                f" color={traffic_light_info.get('color', 'unknown')}"
                f" reason={traffic_light_info.get('reason', '')}"
            )
        print(
            f"\033[1;97m[ Local AI ] :\033[0m \033[1;96mDETECTED\033[0m - "
            f"{sign_display} ({confidence:.1%}) box={box_area:.3%}{dist_str}{traffic_str}"
            f"{'' if is_close else f' (TOO FAR <{effective_min_box:.1%})'}"
            f"{'' if self.enable_actions else ' [actions=OFF]'}"
            f"{'' if self.is_sign_actions_active else ' [inactive_mode]'}"
            f"{'' if is_actionable else ' [not_actionable]'}"
        )

        # In PARKING mode the parking area/spot triggers the state machine in
        # threadLineFollowing instead of signActions, so we skip the action here
        # to avoid signActions stopping the vehicle indefinitely.
        _parking_mode_active = self._current_mode == "parking"
        _skip_parking_action = _parking_mode_active and sign_name in self.PARKING_AREA_CLASSES
        _skip_traffic_light_action = traffic_light_info is not None

        if (
            self.enable_actions
            and self.is_sign_actions_active
            and is_close
            and is_actionable
            and not _skip_parking_action
            and not _skip_traffic_light_action
            and not self._walk_area_active
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
            self._last_frame_shape = frame.shape[:2] if frame is not None else None
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
            lead_distance_px_height = float(result.get("lead_distance_px_height", 0.0) or 0.0)
            lead_distance_cm = None
            if lead_distance_px_height > 0.0 and self._px_per_cm > 0.1 and self._last_frame_shape:
                lead_distance_cm = round(
                    (self._last_frame_shape[0] - lead_distance_px_height) / self._px_per_cm, 1
                )
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
                "lead_distance_cm": lead_distance_cm,
            })

            detections = result.get("detections", [])
            self._handle_walk_area(detections, now)
            self._handle_pedestrian_obstacle(detections, now)
            self._handle_parking_sign(detections)
            self._publish_sign(detections, now, img_shape=self._last_frame_shape, frame=frame)
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
