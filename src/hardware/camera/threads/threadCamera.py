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

import cv2
import threading
import time
import numpy as np

try:
    import picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

from src.core.bus.topics import (
    BRIGHTNESS,
    CONTRAST,
    MAIN_CAMERA,
    MANUAL_RECORDING_SESSION,
    RECORD,
    RECORDING,
    STATE_CHANGE,
)
from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.recording.video_writer import SessionVideoWriter
from src.statemachine.systemMode import SystemMode
from src.templates.threadwithstop import ThreadWithStop

class threadCamera(ThreadWithStop):
    """Thread which will handle camera functionalities.\n
    Args:
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
        camera_type (str): "jetson" for Jetson CSI, "picamera" for CSI camera (default), "usb" for USB camera via OpenCV.
        usb_device (int|str): USB camera device index or path (e.g. 0, 2, "/dev/video0"). Only used when camera_type="usb".
        usb_resolution (tuple): (width, height) for USB camera. Default (640, 480).
        jetson_sensor_id (int): CSI sensor index on Jetson (0=CAM0, 1=CAM1).
        jetson_capture_resolution (tuple): Sensor capture size before conversion.
        jetson_output_resolution (tuple): Final BGR size passed to OpenCV appsink.
        jetson_framerate (int): Camera framerate requested to nvarguscamerasrc.
        jetson_flip_method (int): nvvidconv flip-method (0 = no flip).
        picamera_hdr_enabled (bool): Enable Mertens HDR fusion on PiCamera lores stream.
        picamera_hdr_always_on (bool): Apply HDR on every frame. If False, only when glare is detected.
        picamera_hdr_glare_threshold (float): Saturated pixel ratio threshold to trigger HDR.
    """

    # ================================ INIT ===============================================
    def __init__(self, queuesList, logger, debugger, show_preview=False,
                 frame_buffer=None,
                 camera_type="picamera", usb_device=0, usb_resolution=(640, 480),
                 jetson_sensor_id=0, jetson_capture_resolution=(1920, 1080),
                 jetson_output_resolution=(960, 720), jetson_framerate=30,
                 jetson_flip_method=0,
                 picamera_hdr_enabled=True, picamera_hdr_always_on=False,
                 picamera_hdr_glare_threshold=0.04):
        super(threadCamera, self).__init__(pause=0.01)  # Reduce idle latency between captured frames
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        self.frame_rate = 5
        self.recording = False
        self.show_preview = show_preview
        self.frame_buffer = frame_buffer
        self.camera_type = camera_type
        self.usb_device = usb_device
        self.usb_resolution = usb_resolution
        self.jetson_sensor_id = int(jetson_sensor_id)
        self.jetson_capture_resolution = jetson_capture_resolution
        self.jetson_output_resolution = jetson_output_resolution
        self.jetson_framerate = max(1, int(jetson_framerate))
        self.jetson_flip_method = int(jetson_flip_method)
        self.picamera_hdr_enabled = bool(picamera_hdr_enabled and camera_type == "picamera")
        self.picamera_hdr_always_on = bool(picamera_hdr_always_on)
        self.picamera_hdr_glare_threshold = max(0.0, min(1.0, float(picamera_hdr_glare_threshold)))

        self.video_writer = None
        self._recording_path = None
        self._recording_size = None

        # Grabador de ``raw_video.avi`` por sesión MANUAL. Lo activa/desactiva
        # un suscriptor a ``MANUAL_RECORDING_SESSION`` (ver ``subscribe()``).
        # Independiente del flag ``recording`` del dashboard: ambos pueden
        # coexistir sin pisarse (escriben a archivos distintos).
        self._manual_session_writer: SessionVideoWriter | None = None
        self._manual_session_dir: str | None = None

        # PiCamera adaptive exposure controls (robustness against direct sunlight).
        self._picamera_control_lock = threading.RLock()
        self._picamera_brightness = 0.0  # libcamera range: [-1.0, 1.0], 0.0 = neutral
        self._picamera_contrast = 1.0    # neutral contrast
        self._picamera_ev_supported = True
        self._picamera_control_warning_printed = False
        self._ae_current_ev = 0.0
        self._ae_target_ev = 0.0
        self._ae_update_interval_s = 0.25
        self._ae_last_update_ts = 0.0
        self._ae_step = 0.15
        self._ae_min_ev = -2.5
        self._ae_max_ev = 1.0

        # Mertens HDR fusion on low-resolution stream (for line following/sign detection).
        self._picamera_hdr_warning_printed = False
        self._picamera_hdr_dark_alpha = 0.65
        self._picamera_hdr_dark_beta = -20
        self._picamera_hdr_bright_alpha = 1.35
        self._picamera_hdr_bright_beta = 16
        self._picamera_mertens = None
        if self.picamera_hdr_enabled:
            try:
                self._picamera_mertens = cv2.createMergeMertens(1.0, 1.0, 0.0)  # type: ignore
            except Exception as e:
                self._picamera_mertens = None
                self.picamera_hdr_enabled = False
                print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;93mWARNING\033[0m - Failed to initialize Mertens HDR, disabling HDR: {e}")

        self.recordingSender = messageHandlerSender(self.queuesList, RECORDING)
        self.mainCameraSender = messageHandlerSender(self.queuesList, MAIN_CAMERA)

        self.subscribe()
        self._init_camera()
        self.queue_sending()
        self.configs()

    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""

        self.recordSubscriber = messageHandlerSubscriber(self.queuesList, RECORD, "lastOnly", True)
        self.brightnessSubscriber = messageHandlerSubscriber(self.queuesList, BRIGHTNESS, "lastOnly", True)
        self.contrastSubscriber = messageHandlerSubscriber(self.queuesList, CONTRAST, "lastOnly", True)
        self.stateChangeSubscriber = messageHandlerSubscriber(self.queuesList, STATE_CHANGE, "lastOnly", True)
        # MANUAL_RECORDING_SESSION es LATCHED: si la sesión ya estaba activa
        # cuando arrancamos (caso típico cuando el camera process tarda en
        # bootear y el recorder ya publicó el "active=True"), el primer
        # ``receive()`` nos entrega el último valor.
        self.manualSessionSubscriber = messageHandlerSubscriber(
            self.queuesList, MANUAL_RECORDING_SESSION, "lastOnly", True
        )

    def queue_sending(self):
        """Callback function for recording flag."""
        if self._blocker.is_set():
            return
        self.recordingSender.send(self.recording)
        threading.Timer(1, self.queue_sending).start()

    # ================================ RUN ================================================
    def thread_work(self):
        """This function will run while the running flag is True. 
        It captures the image from camera and make the required modifies 
        and then it send the data to process gateway."""
        # Poll siempre: si llega STOP y la cámara queda sin frames, igual hay
        # que cerrar ``raw_video.avi`` inmediatamente.
        self._handle_manual_session_message()

        # if camera is not available, skip processing
        if self.camera is None:
            time.sleep(0.1)
            return
            
        try:
            recordRecv = self.recordSubscriber.receive()
            if recordRecv is not None:
                if self._record_command_enabled(recordRecv):
                    if not self.recording:
                        self._prepare_recording()
                else:
                    self.recording = False
                    self._release_video_writer()

        except Exception as e:
            print(f"\033[1;97m[ Camera ] :\033[0m \033[1;91mERROR\033[0m - {e}")

        try:
            need_main_frame = (
                self.recording or self.show_preview or self._manual_session_writer is not None
            )
            if self.camera_type in ("usb", "jetson", "zmq"):
                mainRequest, serialRequest = self._capture_usb(need_main_frame)
            else:
                mainRequest, serialRequest = self._capture_picamera(need_main_frame)

            if serialRequest is None:
                return

            if self.camera_type == "picamera":
                self._update_picamera_auto_exposure(serialRequest)
                serialRequest = self._apply_picamera_hdr(serialRequest)

            if self.frame_buffer is not None:
                self.frame_buffer.write(serialRequest)

            if self.recording == True and mainRequest is not None:
                self._write_recording_frame(mainRequest)

            if self._manual_session_writer is not None and mainRequest is not None:
                self._manual_session_writer.write(mainRequest)

            # Phase 6: este thread ya no publica `serialCamera`. El productor
            # único de ese canal pasó a ser `threadLocalPerception`, que envía
            # el frame YA ANOTADO (carriles + cajas de señales). Acá nos
            # quedamos sólo con captura → frame_buffer (entrada de inferencia)
            # y MAIN_CAMERA (grabación a disco).
        except Exception as e:
            print(f"\033[1;97m[ Camera ] :\033[0m \033[1;91mERROR\033[0m - {e}")

    def _capture_picamera(self, need_main_frame=True):
        """Capture frames from PiCamera (CSI). Returns (main_frame, serial_frame)."""
        mainRequest = self.camera.capture_array("main") if need_main_frame else None
        serialRequest = self.camera.capture_array("lores")
        serialRequest = cv2.cvtColor(serialRequest, cv2.COLOR_YUV2BGR_I420)  # type: ignore
        return mainRequest, serialRequest

    def _prepare_recording(self):
        """Arm recording; the writer is opened lazily with the first frame size."""
        self._release_video_writer()
        self._recording_path = "output_video" + str(time.time()) + ".avi"
        self._recording_size = None
        self.recording = True

    def _handle_manual_session_message(self):
        """Reacciona a la apertura/cierre de una sesión MANUAL del recorder.

        El payload viene como dict con ``active`` y ``session_dir``. Sólo
        actuamos en cambios de estado: abrir un writer cuando antes no
        había y cerrar cuando se desactiva. ``LATCHED`` garantiza que el
        valor está disponible aún si arrancamos después del recorder.
        """
        try:
            payload = self.manualSessionSubscriber.receive()
        except Exception as e:
            print(
                f"\033[1;97m[ Camera ] :\033[0m \033[1;93mWARNING\033[0m - "
                f"manual session subscriber error: {e}"
            )
            return
        if payload is None:
            return
        if not isinstance(payload, dict):
            return
        active = bool(payload.get("active", False))
        session_dir = payload.get("session_dir") or ""
        if active and session_dir and session_dir != self._manual_session_dir:
            self._close_manual_session_writer()
            try:
                import os
                os.makedirs(session_dir, exist_ok=True)
                target = os.path.join(session_dir, "camera.avi")
                self._manual_session_writer = SessionVideoWriter(
                    target, fps=self.frame_rate
                )
                self._manual_session_dir = session_dir
                print(
                    f"\033[1;97m[ Camera ] :\033[0m \033[1;92mMANUAL REC\033[0m - "
                    f"raw → {target}"
                )
            except Exception as e:
                print(
                    f"\033[1;97m[ Camera ] :\033[0m \033[1;91mERROR\033[0m - "
                    f"cannot open manual raw writer: {e}"
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

    def _record_command_enabled(self, value):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"start", "true", "1", "on", "yes"}:
            return True
        if text in {"stop", "false", "0", "off", "no", ""}:
            return False
        return bool(value)

    def _release_video_writer(self):
        writer = self.video_writer
        if writer is not None:
            try:
                writer.release()
            except Exception as e:
                print(f"\033[1;97m[ Camera ] :\033[0m \033[1;93mWARNING\033[0m - Failed to release recording writer: {e}")
        self.video_writer = None
        self._recording_path = None
        self._recording_size = None

    def _ensure_video_writer(self, frame):
        if self.video_writer is not None:
            return self.video_writer
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            return None
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            return None
        path = self._recording_path or ("output_video" + str(time.time()) + ".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore
        writer = cv2.VideoWriter(path, fourcc, self.frame_rate, (int(width), int(height)))
        if not writer.isOpened():
            try:
                writer.release()
            except Exception:
                pass
            print(
                f"\033[1;97m[ Camera ] :\033[0m \033[1;91mERROR\033[0m - "
                f"Could not open recording writer at {path} with size {width}x{height}"
            )
            return None
        self.video_writer = writer
        self._recording_path = path
        self._recording_size = (int(width), int(height))
        print(
            f"\033[1;97m[ Camera ] :\033[0m \033[1;92mRECORDING\033[0m - "
            f"{path} @ {width}x{height} {self.frame_rate}fps"
        )
        return writer

    def _write_recording_frame(self, frame):
        writer = self._ensure_video_writer(frame)
        if writer is None:
            return
        out_frame = frame
        if self._recording_size is not None:
            width, height = self._recording_size
            frame_height, frame_width = frame.shape[:2]
            if (int(frame_width), int(frame_height)) != (width, height):
                out_frame = cv2.resize(frame, (width, height))
        writer.write(out_frame)

    def _should_apply_picamera_hdr(self, frame):
        """Decide whether HDR fusion is needed based on glare saturation."""
        if not self.picamera_hdr_enabled or self._picamera_mertens is None:
            return False
        if self.picamera_hdr_always_on:
            return True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        saturated_ratio = float(np.mean(gray >= 245))
        p99 = float(np.percentile(gray, 99))
        return saturated_ratio >= self.picamera_hdr_glare_threshold or p99 >= 250.0

    def _apply_picamera_hdr(self, frame):
        """Apply Mertens exposure fusion using synthetic brackets from the same frame."""
        if not self._should_apply_picamera_hdr(frame):
            return frame
        try:
            frame_f = frame.astype(np.float32)
            dark = np.clip(
                frame_f * self._picamera_hdr_dark_alpha + self._picamera_hdr_dark_beta,
                0,
                255,
            ).astype(np.uint8)
            bright = np.clip(
                frame_f * self._picamera_hdr_bright_alpha + self._picamera_hdr_bright_beta,
                0,
                255,
            ).astype(np.uint8)
            fusion = self._picamera_mertens.process([dark, frame, bright])  # type: ignore
            return np.clip(fusion * 255.0, 0, 255).astype(np.uint8)
        except Exception as e:
            if not self._picamera_hdr_warning_printed:
                print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;93mWARNING\033[0m - HDR fusion failed, using base frame: {e}")
                self._picamera_hdr_warning_printed = True
            return frame

    def _normalize_picamera_brightness(self, value):
        """Normalize dashboard brightness to libcamera range [-1, 1]."""
        v = float(value)
        # Legacy dashboard slider sends 0..1 (after dividing 0..100 by 100).
        if 0.0 <= v <= 1.0:
            v = (v * 2.0) - 1.0
        return max(-1.0, min(1.0, v))

    def _normalize_picamera_contrast(self, value):
        """Normalize contrast to a safe practical range [0.5, 2.0]."""
        v = float(value)
        if v <= 0.0:
            return 1.0
        # Legacy dashboard slider can send up to 32.0 (0..3200 / 100).
        if v > 4.0:
            v = v / 16.0
        return max(0.5, min(2.0, v))

    def _apply_picamera_controls(self):
        """Apply PiCamera controls with AE/AWB enabled for dynamic lighting."""
        if self.camera_type != "picamera" or self.camera is None:
            return

        with self._picamera_control_lock:
            controls = {
                "AeEnable": True,
                "AwbEnable": True,
                "BRIGHTNESS": float(self._picamera_brightness),
                "CONTRAST": float(self._picamera_contrast),
            }
            if self._picamera_ev_supported:
                controls["ExposureValue"] = float(self._ae_current_ev)

            try:
                self.camera.set_controls(controls)
                return
            except Exception as e:
                # Fallback if ExposureValue is not supported by this sensor/driver.
                if "ExposureValue" in controls and self._picamera_ev_supported:
                    self._picamera_ev_supported = False
                    controls.pop("ExposureValue", None)
                    try:
                        self.camera.set_controls(controls)
                        print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;93mWARNING\033[0m - ExposureValue control not supported on this PiCamera. Keeping AE/AWB enabled.")
                        return
                    except Exception as fallback_err:
                        if not self._picamera_control_warning_printed:
                            print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - Failed to apply PiCamera controls: {fallback_err}")
                            self._picamera_control_warning_printed = True
                        return

                if not self._picamera_control_warning_printed:
                    print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - Failed to apply PiCamera controls: {e}")
                    self._picamera_control_warning_printed = True

    def _update_picamera_auto_exposure(self, frame):
        """Dynamically adapt EV for strong glare while keeping AE active."""
        if self.camera_type != "picamera" or self.camera is None:
            return

        now = time.time()
        if now - self._ae_last_update_ts < self._ae_update_interval_s:
            return
        self._ae_last_update_ts = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p90 = float(np.percentile(gray, 90))
        p99 = float(np.percentile(gray, 99))
        saturated_ratio = float(np.mean(gray >= 245))
        dark_ratio = float(np.mean(gray <= 20))

        with self._picamera_control_lock:
            target = self._ae_target_ev

            # Overexposure handling (direct sun / glare).
            if saturated_ratio > 0.10 or p99 > 252:
                target -= 0.35
            elif saturated_ratio > 0.05 or p90 > 220:
                target -= 0.15
            # Underexposure recovery.
            elif dark_ratio > 0.45 and p90 < 110:
                target += 0.15
            else:
                # Relax towards neutral when scene is balanced.
                target *= 0.9

            target = max(self._ae_min_ev, min(self._ae_max_ev, target))
            self._ae_target_ev = target

            delta = self._ae_target_ev - self._ae_current_ev
            if abs(delta) < 0.03:
                return

            step = max(-self._ae_step, min(self._ae_step, delta))
            self._ae_current_ev = max(self._ae_min_ev, min(self._ae_max_ev, self._ae_current_ev + step))

        self._apply_picamera_controls()

    def _capture_usb(self, need_main_frame=True):
        """Wait for a new frame from the USB reader thread, then return it.
        Blocks until the reader thread signals a fresh frame is available,
        so thread_work() runs at the camera's actual FPS (not 1000fps)."""
        # Wait up to 1s for a new frame from the reader thread
        if not self._usb_new_frame.wait(timeout=1.0):
            return None, None  # Timeout — no new frame
        self._usb_new_frame.clear()  # Reset for next frame

        with self._usb_frame_lock:
            frame = self._usb_latest_frame
        if frame is None:
            return None, None
        # main = full resolution frame
        mainRequest = frame.copy() if need_main_frame else None
        # serial (lores) = resized to 640x384 to match PiCamera lores output
        serialRequest = cv2.resize(frame, (640, 384))  # type: ignore
        return mainRequest, serialRequest

    # ================================ STATE CHANGE HANDLER ========================================
    def state_change_handler(self):
        message = self.stateChangeSubscriber.receive()
        if message is not None:
            modeDict = SystemMode[message].value["camera"]["thread"]

            if "resolution" in modeDict:
                print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;92mINFO\033[0m - Resolution changed to {modeDict['resolution']}")

    # ================================ INIT CAMERA ========================================
    def _init_camera(self):
        """Initialize the camera. Supports jetson (GStreamer/nvarguscamerasrc), picamera2 (CSI), USB (OpenCV VideoCapture), and zmq (sim_bridge)."""

        if self.camera_type == "usb":
            self._init_usb_camera()
        elif self.camera_type == "jetson":
            self._init_jetson_camera()
        elif self.camera_type == "zmq":
            self._init_zmq_camera()
        else:
            self._init_picamera()

    def _init_picamera(self):
        """Initialize Raspberry Pi CSI camera via picamera2."""
        try:
            if not PICAMERA2_AVAILABLE:
                print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - picamera2 not installed. Install with: sudo apt install python3-picamera2")
                self.camera = None
                return

            if len(picamera2.Picamera2.global_camera_info()) == 0:
                print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - No CSI camera detected. Camera functionality will be disabled.")
                self.camera = None
                return
            
            self.camera = picamera2.Picamera2()
            config = self.camera.create_preview_configuration(
                buffer_count=1,
                queue=False,
                main={"format": "RGB888", "size": (2048, 1080)},
                lores={"size": (640, 384)},
                encode="lores",
            )
            self.camera.configure(config) # type: ignore
            self.camera.start()
            self._apply_picamera_controls()
            if self.picamera_hdr_enabled:
                mode = "always-on" if self.picamera_hdr_always_on else f"glare-threshold={self.picamera_hdr_glare_threshold:.2f}"
                print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;92mINFO\033[0m - PiCamera HDR (Mertens) enabled ({mode})")
            print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;92mINFO\033[0m - PiCamera initialized successfully")
        except Exception as e:
            print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - Failed to initialize PiCamera: {e}")
            self.camera = None

    def _build_jetson_gstreamer_pipeline(self, output_resolution=None):
        """Build a Jetson CSI GStreamer pipeline suitable for OpenCV appsink."""
        cap_w, cap_h = self.jetson_capture_resolution
        if output_resolution is None:
            out_w, out_h = self.jetson_output_resolution
        else:
            out_w, out_h = output_resolution

        return (
            f"nvarguscamerasrc sensor-id={self.jetson_sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){cap_w}, height=(int){cap_h}, "
            f"framerate=(fraction){self.jetson_framerate}/1 ! "
            f"nvvidconv flip-method={self.jetson_flip_method} ! "
            f"video/x-raw, width=(int){out_w}, height=(int){out_h}, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

    def _warmup_and_get_frame(self, timeout_s=8.0):
        """Read frames until one valid frame arrives or timeout is reached."""
        if self.camera is None:
            return None

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                return frame
            time.sleep(0.03)
        return None

    def _init_jetson_camera(self):
        """Initialize CSI camera on Jetson via GStreamer nvarguscamerasrc pipeline."""
        import subprocess

        gst_check = subprocess.run(["gst-inspect-1.0", "nvarguscamerasrc"], capture_output=True, timeout=5)
        if gst_check.returncode != 0:
            print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - "
                  f"nvarguscamerasrc not available. Install nvidia-l4t-gstreamer-plugins.")
            self.camera = None
            return

        # First try the configured output size (default 960x720), then fallback to sensor size.
        pipeline_candidates = [
            self._build_jetson_gstreamer_pipeline(),
            self._build_jetson_gstreamer_pipeline(output_resolution=self.jetson_capture_resolution),
        ]

        for gst_pipeline in pipeline_candidates:
            try:
                self.camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                if not self.camera.isOpened():
                    if self.camera is not None:
                        self.camera.release()
                    self.camera = None
                    continue

                print(
                    f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;93mINFO\033[0m - "
                    f"Jetson CSI warming up (sensor-id={self.jetson_sensor_id})..."
                )
                test_frame = self._warmup_and_get_frame(timeout_s=8.0)
                if test_frame is None:
                    print(
                        f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;93mWARNING\033[0m - "
                        "Jetson camera opened but no frames yet, trying fallback pipeline..."
                    )
                    self.camera.release()
                    self.camera = None
                    continue

                self._usb_latest_frame = test_frame
                self._usb_frame_lock = threading.Lock()
                self._usb_new_frame = threading.Event()
                self._usb_new_frame.set()
                self._usb_reader_running = True
                self._usb_reader_thread = threading.Thread(target=self._jetson_reader_loop, daemon=True)
                self._usb_reader_thread.start()

                h, w = test_frame.shape[:2]
                print(
                    f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;92mINFO\033[0m - "
                    f"Jetson CSI initialized: {w}x{h} at {self.jetson_framerate} FPS "
                    f"(sensor-id={self.jetson_sensor_id})"
                )
                return
            except Exception as e:
                if self.camera is not None:
                    self.camera.release()
                self.camera = None
                print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;93mWARNING\033[0m - "
                      f"Jetson pipeline attempt failed: {e}")

        print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - "
              "Failed to initialize Jetson CSI camera. "
              "Verify sensor-id and test with: "
              "gst-launch-1.0 nvarguscamerasrc sensor-id=<id> ! nveglglessink")
        self.camera = None

    def _init_usb_camera(self):
        """Initialize USB camera via OpenCV VideoCapture with V4L2 backend.
        Uses a dedicated reader thread to prevent V4L2 buffer stalls."""
        try:
            # Use V4L2 backend explicitly on Linux for better USB camera support
            self.camera = cv2.VideoCapture(self.usb_device, cv2.CAP_V4L2)
            if not self.camera.isOpened():
                print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - USB camera device {self.usb_device} not found.")
                self.camera = None
                return

            # Set MJPG format (most USB cameras support it, much faster than raw)
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            # Minimize internal buffer to avoid stale frames and V4L2 stalls
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            w, h = self.usb_resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

            actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Warmup: discard first frames (USB cameras need a few frames to stabilize)
            print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;93mINFO\033[0m - USB camera warming up...")
            for i in range(10):
                ret, _ = self.camera.read()
                if not ret:
                    time.sleep(0.1)

            # Verify we can actually read a frame
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - USB camera opened but cannot read frames. Try a different device index.")
                self.camera.release()
                self.camera = None
                return

            # Start a background thread that continuously grabs frames.
            # This keeps the V4L2 buffer drained so it never stalls.
            # thread_work() only processes when _usb_new_frame is True (new frame available).
            self._usb_latest_frame = test_frame
            self._usb_frame_lock = threading.Lock()
            self._usb_new_frame = threading.Event()
            self._usb_new_frame.set()  # First frame is ready
            self._usb_reader_running = True
            self._usb_reader_thread = threading.Thread(target=self._usb_reader_loop, daemon=True)
            self._usb_reader_thread.start()

            print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;92mINFO\033[0m - USB camera initialized: device={self.usb_device}, resolution={actual_w}x{actual_h}")
        except Exception as e:
            print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - Failed to initialize USB camera: {e}")
            self.camera = None

    def _init_zmq_camera(self):
        """Subscribe to JPEG frames from sim_bridge over ZMQ. Frames flow through the
        same _usb_latest_frame pipeline used by USB/Jetson, so _capture_usb() works as-is."""
        try:
            import zmq
            from config import ZMQ_CAMERA_ENDPOINT, ZMQ_CAMERA_TOPIC
        except ImportError as e:
            print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - "
                  f"ZMQ camera unavailable ({e}). Install: pip install pyzmq")
            self.camera = None
            return

        try:
            self._zmq_ctx = zmq.Context.instance()
            self._zmq_sock = self._zmq_ctx.socket(zmq.SUB)
            self._zmq_sock.setsockopt(zmq.RCVHWM, 2)
            self._zmq_sock.setsockopt(zmq.CONFLATE, 0)
            self._zmq_sock.connect(ZMQ_CAMERA_ENDPOINT)
            self._zmq_sock.setsockopt(zmq.SUBSCRIBE, ZMQ_CAMERA_TOPIC)

            self.camera = "zmq"
            self._usb_latest_frame = None
            self._usb_frame_lock = threading.Lock()
            self._usb_new_frame = threading.Event()
            self._usb_reader_running = True
            self._usb_reader_thread = threading.Thread(target=self._zmq_reader_loop, daemon=True)
            self._usb_reader_thread.start()

            print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;92mINFO\033[0m - "
                  f"ZMQ camera subscribed at {ZMQ_CAMERA_ENDPOINT} (topic={ZMQ_CAMERA_TOPIC!r})")
        except Exception as e:
            print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - "
                  f"Failed to initialize ZMQ camera: {e}")
            self.camera = None

    def _zmq_reader_loop(self):
        """Background reader: pull JPEG frames from the sim_bridge PUB and decode."""
        import zmq
        poller = zmq.Poller()
        poller.register(self._zmq_sock, zmq.POLLIN)
        while self._usb_reader_running:
            try:
                socks = dict(poller.poll(timeout=200))
                if self._zmq_sock not in socks:
                    continue
                parts = self._zmq_sock.recv_multipart(flags=zmq.NOBLOCK)
                if len(parts) < 2:
                    continue
                payload = parts[1]
                buf = np.frombuffer(payload, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # type: ignore
                if frame is None:
                    continue
                with self._usb_frame_lock:
                    self._usb_latest_frame = frame
                self._usb_new_frame.set()
            except zmq.Again:
                continue
            except Exception as e:
                print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;93mWARNING\033[0m - "
                      f"ZMQ frame decode error: {e}")
                time.sleep(0.05)

    def _jetson_reader_loop(self):
        """Background reader loop for Jetson GStreamer/appsink."""
        while self._usb_reader_running and self.camera is not None:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                with self._usb_frame_lock:
                    self._usb_latest_frame = frame
                self._usb_new_frame.set()
            else:
                time.sleep(0.005)

    def _usb_reader_loop(self):
        """Background thread that continuously grabs frames from the USB camera.
        
        Uses grab()/retrieve() separation to minimize latency:
        - grab() is called in a tight loop (instant, just pulls from V4L2 buffer)
        - retrieve() (slow JPEG decode) only runs when the consumer is ready
        
        This means the decoded frame is always the most recently captured one,
        not a stale frame sitting in a buffer."""
        while self._usb_reader_running and self.camera is not None:
            grabbed = self.camera.grab()  # Fast: pull latest from V4L2, don't decode
            if grabbed:
                # Only decode when consumer has processed the previous frame
                if not self._usb_new_frame.is_set():
                    ret, frame = self.camera.retrieve()  # Slow: decode JPEG → numpy
                    if ret and frame is not None:
                        with self._usb_frame_lock:
                            self._usb_latest_frame = frame
                        self._usb_new_frame.set()
            else:
                time.sleep(0.001)

    # =============================== STOP ================================================
    def stop(self):
        if self.recording and self.video_writer:
            self.video_writer.release() # type: ignore
        self._close_manual_session_writer()
        if self.camera is not None:
            if self.camera_type in ("usb", "jetson"):
                self._usb_reader_running = False
                if hasattr(self, '_usb_reader_thread'):
                    self._usb_reader_thread.join(timeout=2)
                self.camera.release()
            elif self.camera_type == "zmq":
                self._usb_reader_running = False
                if hasattr(self, '_usb_reader_thread'):
                    self._usb_reader_thread.join(timeout=2)
                if hasattr(self, '_zmq_sock'):
                    self._zmq_sock.close(linger=0)
            else:
                self.camera.stop()
        # Local imshow preview was removed (see _capture_and_publish above);
        # nothing to destroy here. ``show_preview`` is kept on the instance
        # in case any external caller still sets it, but it is now inert.
        super(threadCamera, self).stop()

    # =============================== CONFIG ==============================================
    def configs(self):
        """Callback function for receiving configs on the pipe.
        Note: BRIGHTNESS/CONTRAST controls currently apply to PiCamera (picamera2 API).
        USB cameras ignore these for now (could be extended with cv2.VideoCapture props).
        """
        if self._blocker.is_set():
            return
        if self.brightnessSubscriber.is_data_in_pipe():
            message = self.brightnessSubscriber.receive()
            if self.debugger:
                self.logger.info(str(message))
            if self.camera_type == "picamera" and self.camera is not None:
                with self._picamera_control_lock:
                    self._picamera_brightness = self._normalize_picamera_brightness(message)
                self._apply_picamera_controls()
        if self.contrastSubscriber.is_data_in_pipe():
            message = self.contrastSubscriber.receive()
            if self.debugger:
                self.logger.info(str(message))
            if self.camera_type == "picamera" and self.camera is not None:
                with self._picamera_control_lock:
                    self._picamera_contrast = self._normalize_picamera_contrast(message)
                self._apply_picamera_controls()
        threading.Timer(1, self.configs).start()
