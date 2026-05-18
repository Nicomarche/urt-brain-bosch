import math
import time

import cv2

import config
from src.hardware.camera.threads.localPerceptionEngine import LocalPerceptionEngine
from src.perception.signs.sign_classifier import (
    is_actionable_sign,
    normalize_sign_name,
)
from src.perception.object_classes import confidence_threshold_for, needs_lidar_confirmation
from src.statemachine.systemMode import SystemMode
from src.templates.threadwithstop import ThreadWithStop
from src.core.bus.topics import DETECTED_OBJECTS_MSG, LINE_FOLLOWING_CONFIG, LINE_FOLLOWING_STATUS, LOCAL_LANE_PERCEPTION, LOCAL_PERCEPTION_STATUS, MANUAL_RECORDING_SESSION, SIGN_DETECTED, SIGN_DETECTION_STATUS, STATE_CHANGE
from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.core.types.perception import DetectedObject
from src.core.types.pose import PoseEstimate
from src.perception.lidar.processing import distance_in_sector
from src.recording.video_writer import SessionVideoWriter
from src.utils.live_log import live_log


class threadLocalPerception(ThreadWithStop):
    """Runs the local lane/sign model and publishes perception outputs."""

    def __init__(self, queuesList, logger, debugger, frame_buffer=None,
                 local_lane_buffer=None,
                 detection_buffer=None,
                 lidar_scan_buffer=None,
                 pose_estimate_buffer=None,
                 sign_hints_buffer=None,
                 overlay_buffer=None,
                 enable_sign_detection=True, enable_actions=False,
                 sign_min_confidence=0.50, sign_min_box_area=0.01,
                 action_cooldown=15.0,
                 stream_to_dashboard=True):
        # Phase 6: `sign_action_event` y `steer_override_event` se borraron
        # del constructor cuando `signActions.py` desapareció. Ya no había
        # ningún consumidor con efecto real (las ramas en threadLineFollowing
        # eran no-op tras Phase 6, y `_handle_walk_area` los seteaba sin que
        # nada en el path del dispatcher los leyera). `highway_mode_event`
        # sigue vivo pero ya no se setea desde acá: lo gestiona ManeuverManager
        # en threadLineFollowing.
        del action_cooldown  # parámetro legacy aceptado para compat de llamadores
        # Keep scheduler granularity tight so local_ai_interval can reach high FPS targets.
        super(threadLocalPerception, self).__init__(pause=0.001)
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        self.frame_buffer = frame_buffer
        self.local_lane_buffer = local_lane_buffer
        self.detection_buffer = detection_buffer
        self.lidar_scan_buffer = lidar_scan_buffer
        self.pose_estimate_buffer = pose_estimate_buffer
        self.sign_hints_buffer = sign_hints_buffer
        # Shared with ``UdpVideoStreamerThread``: every annotated frame
        # produced by the inference engine lands here; the streamer reads
        # the latest at its own pace. ``None`` disables dashboard streaming
        # at the producer level.
        self.overlay_buffer = overlay_buffer

        self.enable_sign_detection = enable_sign_detection
        self.enable_actions = enable_actions
        self.sign_min_confidence = float(sign_min_confidence)
        self.sign_min_box_area = float(sign_min_box_area)
        self.sign_min_box_area_per_sign = dict(
            getattr(config, "SIGN_MIN_BOX_AREA_PER_SIGN", {})
        )
        self.is_sign_actions_active = False

        self.local_ai_interval = float(getattr(config, "LOCAL_AI_INTERVAL", 0.10))
        self.local_ai_model_path = str(
            getattr(config, "LOCAL_AI_MODEL_PATH", "models/lane_segmentation/Best416px.engine")
        )
        self.local_ai_min_confidence = float(getattr(config, "LOCAL_AI_MIN_CONFIDENCE", 0.35))
        self.local_ai_imgsz = int(getattr(config, "LOCAL_AI_IMGSZ", 416))
        self.local_ai_device = str(getattr(config, "LOCAL_AI_DEVICE", "auto"))

        # Dashboard preview: este thread escribe el frame anotado (carriles +
        # señales) al ``overlay_buffer``. El ``UdpVideoStreamerThread`` lo
        # lee a su propia tasa y lo envía por UDP. Rate-limit y JPEG quality
        # ahora viven en el streamer — acá publicamos cada vez que el motor
        # produce overlay nuevo.
        self.stream_to_dashboard = bool(stream_to_dashboard) and overlay_buffer is not None

        self.last_infer_time = 0.0
        self.last_status_time = 0.0
        self.fps_timer = time.time()
        self.frame_counter = 0
        self.current_fps = 0.0
        self.detection_count = 0
        self.last_sign_name = ""
        self._last_result = None
        self._last_frame_sequence = 0
        self._lf_curve_state = "STRAIGHT"
        self._lf_curve_state_frames = 0
        self._lf_steering_deg = 0.0
        self._current_mode = ""
        self._px_per_cm = 0.0       # cached from LINE_FOLLOWING_STATUS.stanley_px_per_cm
        self._last_frame_shape = None  # (height, width) of last processed frame

        # Walk-area pedestrian stop logic
        self._walk_area_stop_duration     = float(getattr(config, "WALK_AREA_STOP_DURATION", 3.0))
        self._walk_area_min_box_area      = float(getattr(config, "WALK_AREA_MIN_BOX_AREA",  0.04))
        self._walk_area_active            = False   # True while stopped for a walk_area
        self._walk_area_no_obstacle_since = None    # timestamp when obstacle last cleared
        self._walk_area_cooldown          = float(getattr(config, "WALK_AREA_COOLDOWN", 10.0))
        self._walk_area_last_cleared      = 0.0     # timestamp of last walk_area resume

        self.stateChangeSubscriber = messageHandlerSubscriber(
            self.queuesList, STATE_CHANGE, "lastOnly", True
        )
        self.configSubscriber = messageHandlerSubscriber(
            self.queuesList, LINE_FOLLOWING_CONFIG, "lastOnly", True
        )
        self.lineFollowingStatusSubscriber = messageHandlerSubscriber(
            self.queuesList, LINE_FOLLOWING_STATUS, "lastOnly", True
        )
        # LATCHED: si la sesión MANUAL ya estaba abierta cuando este thread
        # arranca, recibimos el último valor en el primer poll.
        self.manualSessionSubscriber = messageHandlerSubscriber(
            self.queuesList, MANUAL_RECORDING_SESSION, "lastOnly", True
        )
        self._manual_session_writer: SessionVideoWriter | None = None
        self._manual_session_dir: str | None = None

        self.localLaneSender = messageHandlerSender(self.queuesList, LOCAL_LANE_PERCEPTION)
        self.localStatusSender = messageHandlerSender(self.queuesList, LOCAL_PERCEPTION_STATUS)
        self.signDetectedSender = messageHandlerSender(self.queuesList, SIGN_DETECTED)
        self.signStatusSender = messageHandlerSender(self.queuesList, SIGN_DETECTION_STATUS)
        self.stateChangeSender = messageHandlerSender(self.queuesList, STATE_CHANGE)
        self.detectedObjectsSender = messageHandlerSender(self.queuesList, DETECTED_OBJECTS_MSG)

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

    def _bbox_center_angle_rad(self, box) -> float | None:
        """Map a normalized bbox center to LiDAR angle.

        Engine boxes are [y1, x1, y2, x2] normalized. Image x grows right;
        LiDAR angle grows left, so the sign is inverted.
        """

        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        try:
            x_center = 0.5 * (float(box[1]) + float(box[3]))
            hfov = math.radians(float(getattr(config, "CAMERA_HORIZONTAL_FOV_DEG", 62.2)))
            return (0.5 - x_center) * hfov
        except (TypeError, ValueError):
            return None

    def _estimate_lidar_distance_cm(
        self,
        box,
        *,
        broad_sector: bool = False,
        sign_sector: bool = False,
    ) -> float | None:
        scan = self.lidar_scan_buffer.read_latest() if self.lidar_scan_buffer is not None else None
        angle = self._bbox_center_angle_rad(box)
        if scan is None or angle is None:
            return None
        try:
            y1, x1, y2, x2 = [float(v) for v in box]
            width_norm = max(0.02, min(0.40, x2 - x1))
        except (TypeError, ValueError):
            width_norm = 0.08
        hfov = math.radians(float(getattr(config, "CAMERA_HORIZONTAL_FOV_DEG", 62.2)))
        base_half_width = 0.5 * width_norm * hfov
        if broad_sector:
            min_half_deg = float(getattr(config, "LIDAR_AI_OBJECT_SECTOR_MIN_HALF_WIDTH_DEG", 24.0))
            extra_deg = float(getattr(config, "LIDAR_AI_OBJECT_SECTOR_EXTRA_DEG", 4.0))
            max_half_deg = float(getattr(config, "LIDAR_AI_OBJECT_SECTOR_MAX_HALF_WIDTH_DEG", 30.0))
        elif sign_sector:
            min_half_deg = float(getattr(config, "LIDAR_AI_SIGN_SECTOR_MIN_HALF_WIDTH_DEG", 6.0))
            extra_deg = float(getattr(config, "LIDAR_AI_SIGN_SECTOR_EXTRA_DEG", 2.0))
            max_half_deg = float(getattr(config, "LIDAR_AI_SIGN_SECTOR_MAX_HALF_WIDTH_DEG", 16.0))
        else:
            min_half_deg = float(getattr(config, "LIDAR_AI_SECTOR_MIN_HALF_WIDTH_DEG", 2.0))
            extra_deg = float(getattr(config, "LIDAR_AI_SECTOR_EXTRA_DEG", 0.0))
            max_half_deg = float(getattr(config, "LIDAR_AI_SECTOR_MAX_HALF_WIDTH_DEG", 12.0))
        half_width = max(math.radians(min_half_deg), base_half_width + math.radians(extra_deg))
        half_width = min(math.radians(max_half_deg), half_width)
        dist_m = distance_in_sector(scan, center_angle_rad=angle, half_width_rad=half_width)
        if dist_m is None:
            return None
        return round(float(dist_m) * 100.0, 1)

    def _project_detection_to_world(
        self,
        box,
        distance_cm: float | None,
    ) -> tuple[float, float] | None:
        """Proyecta una detección (bbox + distancia) al frame mundo.

        Plan TANDA 0.2: la proyección no depende intrínsecamente de LiDAR.
        Sólo necesita ``distance_cm`` válida y la pose ego. Cuando LiDAR
        está ausente, ``_estimate_sign_distance_cm`` (pinhole de cámara)
        provee una estimación razonable para objetos a nivel del piso
        (peatones, autos, signs). Antes este método se llamaba
        ``_project_lidar_detection_to_world`` y se invocaba sólo si la
        fuente de distancia era LiDAR — el efecto secundario era que
        peatones detectados sin LiDAR llegaban al PedestrianYield con
        ``position_world_xy=None`` y eran ignorados silenciosamente.
        """
        if distance_cm is None:
            return None
        pose = self.pose_estimate_buffer.read_latest() if self.pose_estimate_buffer is not None else None
        if not isinstance(pose, PoseEstimate):
            return None
        angle = self._bbox_center_angle_rad(box)
        if angle is None:
            return None
        distance_m = float(distance_cm) * 0.01
        body_x = distance_m * math.cos(angle)
        body_y = distance_m * math.sin(angle)
        yaw = float(pose.fused_pose.yaw)
        world_x = float(pose.fused_pose.x) + body_x * math.cos(yaw) - body_y * math.sin(yaw)
        world_y = float(pose.fused_pose.y) + body_x * math.sin(yaw) + body_y * math.cos(yaw)
        return (float(world_x), float(world_y))

    def _handle_walk_area(self, detections, now):
        """Track walk-area detections sin forzar cambios de modo.

        Phase 6 (post-port Autoware): este método YA NO frena el auto. El
        freno por crosswalk/walk-area lo decide el `BehaviorPlanner` —
        scenarios `Crosswalk` (priority 70) e `Intersection` (priority 60)
        producen `BehaviorOutput` con `stop_required=True` o `speed_profile`
        bajo cuando la lanelet activa tiene los attributes correspondientes.
        Lo único que sigue siendo responsabilidad de este método es:

          1. Observar walk_area + presencia/ausencia de obstáculo.
          2. Mantener una pequeña máquina de estados con cooldown para
             saber cuándo "se cruzó" una zona walk_area.
          3. Marcar la zona como cruzada para evitar retrigger inmediato.

        Reglas:
        - Sólo se considera la zona si bbox area >= WALK_AREA_MIN_BOX_AREA
          (filtra detecciones lejanas con poca confianza espacial).
        - Activación con cooldown: WALK_AREA_COOLDOWN segundos desde el
          último resume. Esto evita re-disparos si la cámara sigue viendo
          la misma zona unos frames después de haberla cruzado.
        - Mientras está activa, esperamos WALK_AREA_STOP_DURATION segundos
          sin obstáculo antes de considerar la zona "cruzada".
        """
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

        if not self._walk_area_active:
            # Only activate when the walk_area is close (large enough bbox)
            if not walk_area_close:
                return
            # Respect cooldown after last resume
            if now - self._walk_area_last_cleared < self._walk_area_cooldown:
                return
            # Activate walk-area observation window. La detención efectiva
            # del auto la dispara el scenario Crosswalk vía BehaviorOutput.
            self._walk_area_active = True
            self._walk_area_no_obstacle_since = None if obstacle_seen else now
            print(
                f"\033[1;97m[ Local AI ] :\033[0m \033[1;91mWALK_AREA\033[0m - "
                f"Walk area observada (box={best_walk_area_box_area:.1%}) "
                f"{'(obstáculo presente)' if obstacle_seen else '(sin obstáculo, esperando 3s)'}"
            )
            return

        # Walk area window activa — actualizar reloj según presencia de obstáculo
        if obstacle_seen:
            if self._walk_area_no_obstacle_since is not None:
                print(
                    f"\033[1;97m[ Local AI ] :\033[0m \033[1;93mWALK_AREA\033[0m - "
                    f"Obstáculo detectado, reiniciando espera"
                )
            self._walk_area_no_obstacle_since = None
        else:
            if self._walk_area_no_obstacle_since is None:
                self._walk_area_no_obstacle_since = now
                print(
                    f"\033[1;97m[ Local AI ] :\033[0m \033[1;93mWALK_AREA\033[0m - "
                    f"Sin obstáculo, esperando {self._walk_area_stop_duration:.0f}s para considerar zona cruzada"
                )
            elif now - self._walk_area_no_obstacle_since >= self._walk_area_stop_duration:
                # Zona despejada por suficiente tiempo: se considera cruzada y
                # se cierra la ventana local. El cambio de modo walk_area→parking
                # era un comportamiento legacy y ya no debe ocurrir.
                self._walk_area_active = False
                self._walk_area_no_obstacle_since = None
                self._walk_area_last_cleared = now
                print(
                    f"\033[1;97m[ Local AI ] :\033[0m \033[1;92mWALK_AREA\033[0m - "
                    f"Zona cruzada, sin cambio de modo"
                )

    def _publish_sign(self, detections, now, img_shape=None):
        if not self.enable_sign_detection or not detections:
            return

        best = detections[0]
        confidence = float(best.get("confidence", 0.0))
        # Filtro inicial barato — descarta detecciones bien por debajo
        # del threshold global (sign_min_confidence típicamente 0.50).
        if confidence < self.sign_min_confidence:
            return

        raw_sign_name = str(best.get("class", ""))
        sign_name = normalize_sign_name(raw_sign_name) or raw_sign_name

        # Plan C1.1: threshold por clase. STOP y RED_LIGHT requieren más
        # certeza (0.65–0.70); peatón usa 0.65 también para compensar la
        # ausencia de confirmación LiDAR obligatoria (plan TANDA 0.2).
        per_class_threshold = confidence_threshold_for(sign_name)
        if confidence < per_class_threshold:
            return
        box = best.get("box", [0, 0, 0, 0])
        if len(box) != 4:
            box = [0, 0, 0, 0]
        box_area = max(0.0, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])))

        img_height = img_shape[0] if img_shape is not None else None
        lidar_distance_cm = self._estimate_lidar_distance_cm(box, sign_sector=True)
        camera_distance_cm = self._estimate_sign_distance_cm(box, img_height)
        if lidar_distance_cm is not None:
            distance_cm = lidar_distance_cm
            distance_source = "lidar"
        elif camera_distance_cm is not None:
            distance_cm = camera_distance_cm
            distance_source = "camera"
        else:
            distance_cm = None
            distance_source = "none"

        # Plan C4: para clases que exigen confirmación LiDAR (STOP, CAR,
        # BLOCK), descartamos el hint si no hubo match LiDAR. El dashboard
        # sigue viendo la detección via SIGN_DETECTED para diagnóstico —
        # solo no llega al sign_hints_buffer que consume el BehaviorPlanner.
        lidar_required = bool(needs_lidar_confirmation(sign_name))
        lidar_confirmed = distance_source == "lidar"
        actionable_for_planner = (not lidar_required) or lidar_confirmed

        self.detection_count += 1
        self.last_sign_name = sign_name
        self.signDetectedSender.send({
            "sign": sign_name,
            "confidence": round(confidence, 3),
            "box": [round(float(v), 5) for v in box],
            "box_area": round(box_area, 5),
            "distance_cm": distance_cm,
            "distance_source": distance_source,
            "lidar_required": lidar_required,
            "actionable": actionable_for_planner,
            "timestamp": now,
        })
        if self.sign_hints_buffer is not None and actionable_for_planner:
            hint = {
                "kind": sign_name,
                "raw_class": raw_sign_name,
                "distance_m": (float(distance_cm) * 0.01) if distance_cm is not None else None,
                "distance_source": distance_source,
                "timestamp": now,
                "confidence": round(confidence, 3),
                "box_area": round(box_area, 5),
            }
            self.sign_hints_buffer.write((hint,), timestamp=now)
            live_log(
                "local_perception",
                event="sign_hint_published",
                kind=hint["kind"],
                raw_class=hint["raw_class"],
                confidence=float(hint["confidence"]),
                box_area=float(hint["box_area"]),
                distance_m=hint["distance_m"],
                distance_source=hint["distance_source"],
            )
        else:
            live_log(
                "local_perception",
                event="sign_detected_no_hint_buffer",
                kind=sign_name,
                raw_class=raw_sign_name,
                confidence=float(confidence),
                box_area=float(box_area),
                distance_m=(float(distance_cm) * 0.01) if distance_cm is not None else None,
                distance_source=distance_source,
            )

        effective_min_box = self.sign_min_box_area_per_sign.get(
            sign_name, self.sign_min_box_area
        )
        is_close = box_area >= effective_min_box
        is_actionable = is_actionable_sign(sign_name)
        sign_display = raw_sign_name if raw_sign_name == sign_name else f"{raw_sign_name}->{sign_name}"
        dist_str = f" ~{distance_cm:.0f}cm/{distance_source}" if distance_cm is not None else ""
        print(
            f"\033[1;97m[ Local AI ] :\033[0m \033[1;96mDETECTED\033[0m - "
            f"{sign_display} ({confidence:.1%}) box={box_area:.3%}{dist_str}"
            f"{'' if is_close else f' (TOO FAR <{effective_min_box:.1%})'}"
            f"{'' if self.enable_actions else ' [actions=OFF]'}"
            f"{'' if self.is_sign_actions_active else ' [inactive_mode]'}"
            f"{'' if is_actionable else ' [not_actionable]'}"
        )

        # In PARKING mode the parking-spot sign triggers the state machine in
        # threadLineFollowing instead of signActions, so we skip the action here
        # to avoid signActions stopping the vehicle indefinitely.
        _parking_mode_active = self._current_mode == "parking"
        _skip_parking_action = _parking_mode_active and sign_name == "parking"

        # Sign actions are now handled centrally by ManeuverManager inside
        # threadLineFollowing. Local perception publishes observations only.

    def _publish_detected_objects(self, detections, now, img_shape=None):
        objects: list[DetectedObject] = []
        dashboard_objects: list[dict] = []
        for det in detections or []:
            class_name = str(det.get("class", "") or "").strip().lower()
            if not self._is_tracked_object_class(class_name):
                continue
            try:
                confidence = float(det.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            box = det.get("box", [0.0, 0.0, 0.0, 0.0])
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                box = [0.0, 0.0, 0.0, 0.0]

            lidar_distance_cm = self._estimate_lidar_distance_cm(box, broad_sector=True)
            camera_distance_cm = self._estimate_sign_distance_cm(
                box,
                img_shape[0] if img_shape is not None else None,
            )
            if lidar_distance_cm is not None:
                distance_cm = lidar_distance_cm
                distance_source = "lidar"
            elif camera_distance_cm is not None:
                distance_cm = camera_distance_cm
                distance_source = "camera"
            else:
                distance_cm = None
                distance_source = "none"
            angle = self._bbox_center_angle_rad(box)
            # Plan TANDA 0.2: proyectar al mundo siempre que tengamos
            # distancia confiable, sea LiDAR o cámara pinhole. El
            # PedestrianYield necesita `position_world_xy` para evaluar el
            # cono; sin esto, el peatón se ignora silenciosamente cuando
            # el LiDAR no llega o no cubre el sector.
            world_xy = (
                self._project_detection_to_world(box, distance_cm)
                if distance_cm is not None
                else None
            )
            bbox_xyxy = _box_yxyx_to_xyxy(box)
            obj = DetectedObject(
                timestamp=now,
                class_id=int(det.get("class_id", -1) or -1) if str(det.get("class_id", "-1")).lstrip("-").isdigit() else -1,
                class_name=class_name,
                confidence=confidence,
                bbox_xyxy=tuple(bbox_xyxy),
                position_world_xy=world_xy,
            )
            objects.append(obj)
            dashboard_objects.append(
                {
                    "kind": class_name,
                    "confidence": round(confidence, 4),
                    "bbox_xyxy": [round(float(v), 5) for v in bbox_xyxy],
                    "angle_rad": float(angle) if angle is not None else None,
                    "distance_m": (float(distance_cm) * 0.01) if distance_cm is not None else None,
                    "distance_source": distance_source,
                    "world_xy": list(world_xy) if world_xy is not None else None,
                }
            )

        if self.detection_buffer is not None:
            self.detection_buffer.write(objects, timestamp=now)
        try:
            self.detectedObjectsSender.send({
                "timestamp": now,
                "objects": dashboard_objects,
            })
        except Exception:
            if self.logger is not None:
                self.logger.exception("failed to publish DETECTED_OBJECTS_MSG")

    @staticmethod
    def _is_tracked_object_class(class_name: str) -> bool:
        return str(class_name).lower() in {
            "obstacle",
            "road_block",
            "vehicle",
            "car",
            "bus",
            "truck",
            "pedestrian",
            "person",
            "cyclist",
            "bicycle",
            "motorcycle",
            "traffic_light",
        }

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

    def _publish_dashboard_frame(self, result, frame, now):
        """Escribe el frame anotado al ``overlay_buffer``.

        El ``UdpVideoStreamerThread`` (mismo proceso, hilo paralelo) lee de
        este buffer al ritmo configurado por ``URT_VIDEO_FPS`` y lo envía
        fragmentado por UDP al GUI. No hay encoding ni rate-limit acá: la
        inferencia escribe el frame freshest disponible, el streamer decide
        qué hacer con él (skippea si no cambió, dropea si llega muy rápido).

        ``result["lane_debug"]["overlay"]`` es la imagen BGR con carriles y
        cajas de señales dibujadas. Si no existe (modelo no listo todavía),
        se publica el frame raw para evitar pantalla negra en el GUI.
        """
        overlay = None
        if isinstance(result, dict):
            lane_debug = result.get("lane_debug")
            if isinstance(lane_debug, dict):
                overlay = lane_debug.get("overlay")

        img = overlay if overlay is not None else frame
        if img is None:
            return

        # Sesión MANUAL: el mismo frame anotado que se envía al GUI se
        # archiva a ``annotated_video.avi`` dentro de la subcarpeta de la
        # sesión. Independiente de ``stream_to_dashboard`` para que
        # corridas headless también queden grabadas.
        self._handle_manual_session_message()
        if self._manual_session_writer is not None:
            self._manual_session_writer.write(img)

        if not self.stream_to_dashboard or self.overlay_buffer is None:
            return
        self.overlay_buffer.write(img)

    def _handle_manual_session_message(self):
        """Abre/cierra el writer del anotado segun MANUAL_RECORDING_SESSION."""
        try:
            payload = self.manualSessionSubscriber.receive()
        except Exception:
            return
        if payload is None or not isinstance(payload, dict):
            return
        active = bool(payload.get("active", False))
        session_dir = payload.get("session_dir") or ""
        if active and session_dir and session_dir != self._manual_session_dir:
            self._close_manual_session_writer()
            try:
                import os
                os.makedirs(session_dir, exist_ok=True)
                target = os.path.join(session_dir, "annotated_video.avi")
                # FPS objetivo del overlay = 1 / local_ai_interval (la
                # cadencia real a la que escribimos overlay frames).
                fps = (
                    1.0 / self.local_ai_interval if self.local_ai_interval > 0 else 10.0
                )
                self._manual_session_writer = SessionVideoWriter(target, fps=fps)
                self._manual_session_dir = session_dir
                print(
                    f"\033[1;97m[ Local AI ] :\033[0m \033[1;92mMANUAL REC\033[0m - "
                    f"annotated → {target}"
                )
            except Exception as e:
                print(
                    f"\033[1;97m[ Local AI ] :\033[0m \033[1;91mERROR\033[0m - "
                    f"cannot open annotated writer: {e}"
                )
                self._manual_session_writer = None
                self._manual_session_dir = None
        elif not active:
            self._close_manual_session_writer()

    def _close_manual_session_writer(self):
        if self._manual_session_writer is not None:
            try:
                self._manual_session_writer.close()
            except Exception:
                pass
            self._manual_session_writer = None
            self._manual_session_dir = None

    def _build_local_lane_payload(self, result, frame_timestamp, frame_sequence):
        result_timestamp = time.time()
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

        return {
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
        }

    def thread_work(self):
        # Poll siempre: si STOP corta el flujo de frames/inferencia, igual hay
        # que cerrar ``annotated_video.avi`` y soltar el fd de inmediato.
        self._handle_manual_session_message()

        self.state_change_handler()
        self._check_config()
        self._poll_line_following_context()
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
        if self.stream_to_dashboard and (
            self._last_result is None
            or not bool(self._last_result.get("model_ready", False))
        ):
            # First paint matters in monitor mode: publish a raw frame before
            # the first lazy model load/warmup can block for seconds.
            self._publish_dashboard_frame(None, frame, now)

        if now - self.last_infer_time < self.local_ai_interval:
            return
        self.last_infer_time = now
        self._last_frame_sequence = frame_sequence

        try:
            # build_debug=True siempre: el `CameraView` del dashboard consume el
            # frame anotado que sale de `_draw_debug_views`. El costo extra
            # (~1-2 ms de cv2.polylines / addWeighted / rectangle) es despreciable
            # frente a los ~30-80 ms de la inferencia YOLO.
            result = self.engine.infer(frame, build_debug=True)
            self._last_result = result
            self._last_frame_shape = frame.shape[:2] if frame is not None else None
            self.frame_counter += 1

            elapsed = now - self.fps_timer
            if elapsed >= 1.0:
                self.current_fps = self.frame_counter / elapsed
                self.frame_counter = 0
                self.fps_timer = now

            lane_payload = self._build_local_lane_payload(result, frame_timestamp, frame_sequence)
            if self.local_lane_buffer is not None:
                self.local_lane_buffer.write(lane_payload)
                queue_payload = dict(lane_payload)
                # Keep the gateway payload light: the full-resolution masks are large
                # numpy arrays and were creating several seconds of transport lag.
                queue_payload["side_masks"] = {"left": None, "right": None}
                queue_payload["lane_mask"] = None
                self.localLaneSender.send(queue_payload)
            else:
                self.localLaneSender.send(lane_payload)

            detections = result.get("detections", [])
            self._handle_walk_area(detections, now)
            self._publish_sign(detections, now, img_shape=self._last_frame_shape)
            self._publish_detected_objects(detections, now, img_shape=self._last_frame_shape)
            self._publish_status(result, now)
            self._publish_dashboard_frame(result, frame, now)
        except Exception as e:
            print(f"\033[1;97m[ Local AI ] :\033[0m \033[1;91mERROR\033[0m - {e}")

    def stop(self):
        self._close_manual_session_writer()
        super(threadLocalPerception, self).stop()


def _box_yxyx_to_xyxy(box) -> tuple[float, float, float, float]:
    try:
        y1, x1, y2, x2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0, 0.0)
    return (x1, y1, x2, y2)
