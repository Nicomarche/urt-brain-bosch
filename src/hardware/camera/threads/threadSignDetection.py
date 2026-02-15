# Thread for traffic sign detection via remote AI Server (WebSocket).
#
# Sends camera frames to the AI Server which runs MobilenetV2 SSD TFLite,
# receives detected signs, publishes results, and optionally executes
# vehicle actions (stop, slow down, etc.)
#
# The heavy inference runs on the server PC, keeping the RPi lightweight.

import cv2
import numpy as np
import base64
import time
import os
import json
import queue
import threading
import asyncio

from src.utils.messages.allMessages import (
    serialCamera,
    SpeedMotor,
    SteerMotor,
    StateChange,
    SignDetected,
    SignDetectionStatus,
)
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.templates.threadwithstop import ThreadWithStop
from src.statemachine.systemMode import SystemMode

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print(
        f"\033[1;97m[ SignDetection ] :\033[0m \033[1;91mERROR\033[0m - "
        f"websockets not installed. Run: pip install websockets"
    )


# ============================================================================
#  Lightweight WebSocket client for sign detection
# ============================================================================

class SignDetectionClient:
    """WebSocket client that sends frames to the AI Server for sign detection.

    Runs an asyncio event loop in a background thread. Frames are sent via
    a thread-safe queue, and results are received asynchronously.

    Protocol (matches /ws/signs endpoint on the server):
      Client → Server: raw JPEG bytes
      Server → Client: JSON {"d": [...], "t": float, "f": int}
    """

    def __init__(self, server_url, jpeg_quality=70, timeout=2.0,
                 reconnect_interval=3.0):
        self.server_url = server_url
        self.jpeg_quality = jpeg_quality
        self.timeout = timeout
        self.reconnect_interval = reconnect_interval

        self._frame_queue = queue.Queue(maxsize=1)
        self._result_queue = queue.Queue(maxsize=1)

        self._running = False
        self._connected = False
        self._thread = None
        self._last_result = None
        self.frames_sent = 0
        self.frames_received = 0

    @property
    def connected(self):
        return self._connected

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="SignDetClient"
        )
        self._thread.start()
        print(
            f"\033[1;97m[ SignDetClient ] :\033[0m \033[1;92mINFO\033[0m - "
            f"Connecting to {self.server_url}"
        )

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._connected = False
        print(
            f"\033[1;97m[ SignDetClient ] :\033[0m \033[1;93mINFO\033[0m - "
            f"Stopped ({self.frames_sent} sent, {self.frames_received} received)"
        )

    def send_jpeg(self, jpeg_bytes, block=False):
        """Send raw JPEG bytes to server (avoids decode/encode cycle on RPi).

        Args:
            jpeg_bytes: Raw JPEG bytes (from base64.b64decode of camera data).
            block: If True, wait for response (up to timeout).

        Returns:
            dict with detections or None.
        """
        if not self._connected:
            return None

        # Replace old frame in queue (keep only latest)
        try:
            self._frame_queue.get_nowait()
        except queue.Empty:
            pass
        self._frame_queue.put(jpeg_bytes)
        self.frames_sent += 1

        if not block:
            try:
                return self._result_queue.get_nowait()
            except queue.Empty:
                return None

        try:
            result = self._result_queue.get(timeout=self.timeout)
            return result
        except queue.Empty:
            return None

    def send_frame(self, frame, block=True):
        """Send OpenCV frame to server (encodes to JPEG first).

        Prefer send_jpeg() when you already have JPEG bytes to avoid
        unnecessary decode/encode cycles.
        """
        if not self._connected:
            return None

        _, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        return self.send_jpeg(encoded.tobytes(), block=block)

    def _run_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._connection_loop())

    async def _connection_loop(self):
        while self._running:
            try:
                await self._connect_and_process()
            except Exception as e:
                if self._running:
                    print(
                        f"\033[1;97m[ SignDetClient ] :\033[0m \033[1;91mERROR\033[0m - "
                        f"Connection lost: {e}"
                    )
                    self._connected = False
                    await asyncio.sleep(self.reconnect_interval)

    async def _connect_and_process(self):
        async with websockets.connect(
            self.server_url,
            max_size=4 * 1024 * 1024,
            ping_interval=30,
            ping_timeout=20,
        ) as ws:
            self._connected = True
            print(
                f"\033[1;97m[ SignDetClient ] :\033[0m \033[1;92mINFO\033[0m - "
                f"Connected to {self.server_url}"
            )
            consecutive_timeouts = 0

            while self._running:
                try:
                    jpeg_bytes = self._frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                await ws.send(jpeg_bytes)

                try:
                    response_data = await asyncio.wait_for(
                        ws.recv(), timeout=self.timeout
                    )
                except asyncio.TimeoutError:
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= 3:
                        print(
                            f"\033[1;97m[ SignDetClient ] :\033[0m \033[1;91mERROR\033[0m - "
                            f"3 consecutive timeouts, reconnecting..."
                        )
                        break
                    continue

                consecutive_timeouts = 0
                self.frames_received += 1

                try:
                    result = json.loads(response_data)
                except json.JSONDecodeError:
                    continue

                self._last_result = result

                # Push to result queue (replace old)
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    pass
                self._result_queue.put(result)


# ============================================================================
#  Sign execution actions
# ============================================================================

class SignActions:
    """Executes vehicle actions in response to detected traffic signs.

    Handles signs from YOLOv8 model (mapped from Turkish class names):
      stop, no_entry, crosswalk, parking, red_light, yellow_light,
      green_light, speed_20, speed_30, turn_right, turn_left, etc.
    """

    BASE_SPEED = 5
    LOW_SPEED = 3
    SPEED_20 = 3
    SPEED_30 = 5
    STOP_DURATION = 3.0
    CROSSWALK_DURATION = 3.0
    RED_LIGHT_CHECK_INTERVAL = 0.5  # Re-check every 0.5s

    # Signs that should trigger an action
    ACTIONABLE_SIGNS = {
        "stop", "no_entry", "crosswalk", "red_light", "yellow_light",
        "green_light", "speed_20", "speed_30", "parking",
    }

    # Cooldown groups: signs that share the same cooldown timer.
    # e.g. no_entry, stop, red_light all trigger STOP — if one fires,
    # the others should be on cooldown too (same physical action).
    ACTION_GROUP = {
        "stop": "stop", "no_entry": "stop", "red_light": "stop",
        "crosswalk": "crosswalk",
        "yellow_light": "traffic_light", "green_light": "traffic_light",
        "speed_20": "speed_limit", "speed_30": "speed_limit",
        "parking": "parking",
    }

    def __init__(self, queuesList, sign_action_event=None, action_cooldown=15.0):
        self.queuesList = queuesList
        self.sign_action_event = sign_action_event  # Shared with line following
        self.action_cooldown = action_cooldown
        self.last_sign = None
        self.last_action_time = {}  # {action_group: timestamp} — cooldown per group
        self.current_speed = self.BASE_SPEED
        self.is_stopped = False

    def execute(self, sign_name):
        """Execute action for a detected sign. Returns True if action was taken."""
        now = time.time()

        # Cooldown by action group (stop/no_entry/red_light share "stop" group)
        action_group = self.ACTION_GROUP.get(sign_name, sign_name)
        last_time = self.last_action_time.get(action_group, 0)
        elapsed = now - last_time
        if elapsed < self.action_cooldown:
            remaining = self.action_cooldown - elapsed
            print(
                f"\033[1;97m[ SignActions ] :\033[0m \033[1;93mCOOLDOWN\033[0m - "
                f"{sign_name} ignored ({remaining:.0f}s left)"
            )
            return False

        if sign_name == "stop":
            self._execute_stop()
        elif sign_name == "red_light":
            self._execute_stop()
        elif sign_name == "no_entry":
            self._execute_stop()
        elif sign_name == "crosswalk":
            self._execute_crosswalk()
        elif sign_name == "yellow_light":
            self._execute_slow_down()
        elif sign_name == "green_light":
            self._execute_go()
        elif sign_name == "speed_20":
            self._execute_speed_limit(self.SPEED_20)
        elif sign_name == "speed_30":
            self._execute_speed_limit(self.SPEED_30)
        elif sign_name == "parking":
            self._execute_parking()
        else:
            # Non-actionable signs (direction, objects, etc.) — just log
            print(
                f"\033[1;97m[ SignActions ] :\033[0m \033[1;93mINFO\033[0m - "
                f"{sign_name} detected (no action)"
            )
            return False

        # Set cooldown AFTER successful action (not before)
        self.last_sign = sign_name
        self.last_action_time[action_group] = now
        return True

    def _send_speed(self, speed):
        # SpeedMotor.msgType is "str" — must send string to avoid type mismatch warning.
        # Value is speed * 10 to match Nucleo protocol (same as line following).
        speed_value = str(int(round(speed * 10)))
        self.queuesList["General"].put({
            "Owner": SpeedMotor.Owner.value,
            "msgID": SpeedMotor.msgID.value,
            "msgType": SpeedMotor.msgType.value,
            "msgValue": speed_value,
        })

    def _execute_stop(self):
        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;91mACTION\033[0m - "
            f"STOP ({self.STOP_DURATION}s)"
        )
        if self.sign_action_event:
            self.sign_action_event.set()  # Block line following from sending speed
        self._send_speed(0)
        self.is_stopped = True
        time.sleep(self.STOP_DURATION)
        self.is_stopped = False
        self._send_speed(self.current_speed)
        if self.sign_action_event:
            self.sign_action_event.clear()  # Resume line following

    def _execute_crosswalk(self):
        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;93mACTION\033[0m - "
            f"CROSSWALK - slow to {self.LOW_SPEED} ({self.CROSSWALK_DURATION}s)"
        )
        if self.sign_action_event:
            self.sign_action_event.set()
        self._send_speed(self.LOW_SPEED)
        time.sleep(self.CROSSWALK_DURATION)
        self._send_speed(self.current_speed)
        if self.sign_action_event:
            self.sign_action_event.clear()

    def _execute_slow_down(self):
        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;93mACTION\033[0m - "
            f"YELLOW LIGHT - slow to {self.LOW_SPEED}"
        )
        self._send_speed(self.LOW_SPEED)

    def _execute_go(self):
        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;92mACTION\033[0m - "
            f"GREEN LIGHT - resume speed {self.current_speed}"
        )
        self.is_stopped = False
        self._send_speed(self.current_speed)

    def _execute_speed_limit(self, speed):
        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;96mACTION\033[0m - "
            f"SPEED LIMIT {speed}"
        )
        self.current_speed = speed
        if not self.is_stopped:
            self._send_speed(speed)

    def _execute_parking(self):
        print(
            f"\033[1;97m[ SignActions ] :\033[0m \033[1;96mACTION\033[0m - "
            f"PARKING - stop"
        )
        if self.sign_action_event:
            self.sign_action_event.set()
        self._send_speed(0)
        self.is_stopped = True


# ============================================================================
#  Main sign detection thread
# ============================================================================

class threadSignDetection(ThreadWithStop):
    """Thread that detects traffic signs via remote AI Server (WebSocket).

    Sends camera frames to the AI Server, receives detections,
    publishes results, and optionally executes vehicle actions.

    Args:
        queuesList: Dictionary of multiprocessing queues.
        logger: Logging object.
        debugger: Debug flag.
        server_url: WebSocket URL of the AI Server sign detection endpoint.
        enable_actions: If True, execute sign actions (stop/slow/speed).
        min_confidence: Minimum confidence to accept (client-side filter).
        detection_interval: Min seconds between sending frames (~FPS).
        show_debug: Write debug images to /tmp/sign_detection_debug.jpg.
    """

    def __init__(self, queuesList, logger, debugger,
                 server_url="ws://192.168.80.15:8500/ws/signs",
                 enable_actions=False,
                 min_confidence=0.50,
                 min_box_area=0.01,
                 action_cooldown=15.0,
                 detection_interval=0.2,
                 show_debug=False,
                 sign_action_event=None):
        super(threadSignDetection, self).__init__(pause=0.05)  # 20Hz, suficiente para ~3 FPS de detección
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        self.server_url = server_url
        self.enable_actions = enable_actions
        self.min_confidence = min_confidence
        self.min_box_area = min_box_area
        self.detection_interval = detection_interval
        self.show_debug = show_debug

        # State
        self.sign_actions = SignActions(self.queuesList, sign_action_event, action_cooldown)
        self.is_active = False
        self.last_detection_time = 0
        self.frame_count = 0
        self.detection_count = 0
        self.fps_timer = time.time()
        self.current_fps = 0.0
        self.last_sign_name = None
        self.last_sign_time = 0
        self.last_status_time = 0

        # Debug image output
        self.debug_image_path = "/tmp/sign_detection_debug.jpg"
        self.debug_image_tmp = "/tmp/sign_detection_debug_tmp.jpg"

        # Subscribers and senders
        self.serialCameraSubscriber = messageHandlerSubscriber(
            self.queuesList, serialCamera, "lastOnly", True
        )
        self.stateChangeSubscriber = messageHandlerSubscriber(
            self.queuesList, StateChange, "lastOnly", True
        )
        self.signDetectedSender = messageHandlerSender(self.queuesList, SignDetected)
        self.statusSender = messageHandlerSender(self.queuesList, SignDetectionStatus)

        # WebSocket client
        self.client = None
        if WEBSOCKETS_AVAILABLE:
            self.client = SignDetectionClient(self.server_url)
            self.client.start()
            print(
                f"\033[1;97m[ SignDetection ] :\033[0m \033[1;92mINFO\033[0m - "
                f"Remote sign detection via {self.server_url} "
                f"(actions={'ON' if self.enable_actions else 'OFF'}, "
                f"min_box={self.min_box_area:.1%}, cooldown={self.sign_actions.action_cooldown:.0f}s)"
            )
        else:
            print(
                f"\033[1;97m[ SignDetection ] :\033[0m \033[1;91mERROR\033[0m - "
                f"Cannot start: websockets not installed"
            )

    # ================================ STATE CHANGE ============================
    def state_change_handler(self):
        message = self.stateChangeSubscriber.receive()
        if message is not None:
            try:
                mode_dict = SystemMode[message].value
                camera_config = mode_dict.get("camera", {})
                sign_config = camera_config.get("signDetection", {})
                was_active = self.is_active
                self.is_active = sign_config.get("enabled", False)

                if self.is_active != was_active:
                    status = "ENABLED" if self.is_active else "DISABLED"
                    color = "92" if self.is_active else "93"
                    print(
                        f"\033[1;97m[ SignDetection ] :\033[0m \033[1;{color}mINFO\033[0m - "
                        f"Sign actions {status} (mode={message})"
                    )
            except (KeyError, TypeError):
                pass

    # ================================ MAIN LOOP ==============================
    def thread_work(self):
        # Always drain the camera pipe to prevent gateway blocking
        camera_message = self.serialCameraSubscriber.receive()

        if self.client is None:
            time.sleep(0.1)
            return

        # Rate limiting
        now = time.time()
        if now - self.last_detection_time < self.detection_interval:
            return

        if camera_message is None:
            time.sleep(0.05)
            return

        try:
            # Decode base64 → raw JPEG bytes (cheap, no OpenCV needed)
            jpeg_bytes = base64.b64decode(camera_message)

            self.last_detection_time = now
            self.frame_count += 1

            # Send raw JPEG bytes directly to server (non-blocking).
            # This skips cv2.imdecode + cv2.imencode on the RPi,
            # saving ~10-15ms of CPU per frame.
            result = self.client.send_jpeg(jpeg_bytes, block=False)

            # Update FPS
            elapsed = now - self.fps_timer
            if elapsed >= 2.0:
                self.current_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_timer = now

            # Parse server response
            detections = []
            server_time_ms = 0
            if result and "d" in result:
                server_time_ms = result.get("t", 0)
                for det in result["d"]:
                    conf = det.get("c", 0)
                    if conf >= self.min_confidence:
                        detections.append({
                            "sign": det["s"],
                            "confidence": conf,
                            "box": det.get("b", [0, 0, 0, 0]),
                        })

            # Best detection
            sign_name = None
            confidence = 0.0
            box_area = 0.0
            if detections:
                sign_name = detections[0]["sign"]
                confidence = detections[0]["confidence"]
                box = detections[0]["box"]
                # box = [ymin, xmin, ymax, xmax] normalized (0-1)
                box_area = (box[2] - box[0]) * (box[3] - box[1])

            if sign_name is not None:
                self.detection_count += 1
                self.last_sign_name = sign_name
                self.last_sign_time = now
                is_close = box_area >= self.min_box_area
                print(
                    f"\033[1;97m[ SignDetection ] :\033[0m \033[1;96mDETECTED\033[0m - "
                    f"{sign_name} ({confidence:.1%}) [server: {server_time_ms:.0f}ms] "
                    f"box={box_area:.3%}{'' if is_close else ' (TOO FAR)'}"
                )

                self.signDetectedSender.send({
                    "sign": sign_name,
                    "confidence": round(confidence, 3),
                    "box_area": round(box_area, 5),
                    "timestamp": now,
                })

                # Execute actions ONLY in AUTO mode AND when sign is close enough
                if self.is_active and self.enable_actions and is_close:
                    self.sign_actions.execute(sign_name)
                elif self.enable_actions and is_close and sign_name in SignActions.ACTIONABLE_SIGNS:
                    # Actionable sign detected but conditions not met — log why
                    if not self.is_active:
                        print(
                            f"\033[1;97m[ SignActions ] :\033[0m \033[1;91mSKIPPED\033[0m - "
                            f"{sign_name} — is_active=False (need AUTO mode)"
                        )

            # Debug image (only decode frame if debug is on — expensive!)
            if self.show_debug and detections:
                nparr = np.frombuffer(jpeg_bytes, np.uint8)
                debug_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if debug_frame is not None:
                    self._write_debug_image(debug_frame, detections, now, server_time_ms)

            # Status update (every 2s)
            if now - self.last_status_time >= 2.0:
                self.last_status_time = now
                self.statusSender.send({
                    "enabled": self.is_active,
                    "fps": round(self.current_fps, 1),
                    "last_sign": sign_name if sign_name else "",
                    "total_detections": self.detection_count,
                    "server_connected": self.client.connected if self.client else False,
                })

        except Exception as e:
            print(
                f"\033[1;97m[ SignDetection ] :\033[0m \033[1;91mERROR\033[0m - {e}"
            )

    # ================================ DEBUG ===================================

    SIGN_COLORS = {
        "stop": (0, 0, 255), "no_entry": (0, 0, 200),
        "parking": (255, 160, 0), "crosswalk": (0, 200, 255),
        "highway_entrance": (0, 255, 0), "highway_exit": (0, 180, 0),
        "roundabout": (255, 0, 255), "priority": (0, 255, 255),
        "one_way": (255, 255, 0),
    }

    def _write_debug_image(self, frame, detections, now, server_time_ms=0):
        """Write debug image to /tmp/sign_detection_debug.jpg."""
        try:
            debug_frame = frame.copy()
            h, w = debug_frame.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Draw bounding boxes
            for det in detections:
                sign = det["sign"]
                conf = det["confidence"]
                box = det["box"]
                color = self.SIGN_COLORS.get(sign, (0, 255, 0))

                ymin, xmin, ymax, xmax = (
                    int(box[0] * h), int(box[1] * w),
                    int(box[2] * h), int(box[3] * w),
                )
                cv2.rectangle(debug_frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f"{sign} {conf:.0%}"
                (tw, th_t), _ = cv2.getTextSize(label, font, 0.55, 2)
                label_y = max(ymin, th_t + 8)
                cv2.rectangle(debug_frame, (xmin, label_y - th_t - 6), (xmin + tw + 6, label_y + 2), color, -1)
                cv2.putText(debug_frame, label, (xmin + 3, label_y - 3), font, 0.55, (0, 0, 0), 2)

            # Status bar
            cv2.rectangle(debug_frame, (0, 0), (w, 70), (0, 0, 0), -1)
            cv2.putText(debug_frame, "SIGN DETECTION (REMOTE)", (10, 20), font, 0.5, (0, 255, 255), 1)

            conn_text = "CONNECTED" if (self.client and self.client.connected) else "DISCONNECTED"
            conn_color = (0, 255, 0) if (self.client and self.client.connected) else (0, 0, 255)
            cv2.putText(debug_frame, conn_text, (w - 150, 20), font, 0.45, conn_color, 1)

            cv2.putText(debug_frame, f"FPS: {self.current_fps:.1f} | Server: {server_time_ms:.0f}ms", (10, 42),
                         font, 0.45, (200, 200, 200), 1)

            mode_text = "AUTO (actions ON)" if self.is_active else "DETECT ONLY"
            mode_color = (0, 255, 0) if self.is_active else (0, 165, 255)
            cv2.putText(debug_frame, mode_text, (w - 180, 42), font, 0.4, mode_color, 1)

            if detections:
                best = detections[0]
                text = f">>> {best['sign'].upper()} ({best['confidence']:.0%}) <<<"
                cv2.putText(debug_frame, text, (10, 63), font, 0.55, (0, 255, 0), 2)
            else:
                if self.last_sign_name and (now - self.last_sign_time) < 3.0:
                    cv2.putText(debug_frame, f"Last: {self.last_sign_name} ({now - self.last_sign_time:.1f}s ago)",
                                 (10, 63), font, 0.45, (100, 100, 100), 1)
                else:
                    cv2.putText(debug_frame, "No signs detected", (10, 63), font, 0.45, (0, 0, 200), 1)

            cv2.imwrite(self.debug_image_tmp, debug_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            os.replace(self.debug_image_tmp, self.debug_image_path)
        except Exception:
            pass

    def stop(self):
        print(
            f"\033[1;97m[ SignDetection ] :\033[0m \033[1;93mINFO\033[0m - "
            f"Stopping (detections: {self.detection_count})"
        )
        if self.client:
            self.client.stop()
        for path in (self.debug_image_path, self.debug_image_tmp):
            try:
                os.remove(path)
            except OSError:
                pass
        super(threadSignDetection, self).stop()
