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
import base64
import time

try:
    import picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

from src.utils.messages.allMessages import (
    mainCamera,
    serialCamera,
    Recording,
    Record,
    Brightness,
    Contrast,
)
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import StateChange
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.statemachine.systemMode import SystemMode

class threadCamera(ThreadWithStop):
    """Thread which will handle camera functionalities.\n
    Args:
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
        camera_type (str): "picamera" for CSI camera (default), "usb" for USB camera via OpenCV.
        usb_device (int|str): USB camera device index or path (e.g. 0, 2, "/dev/video0"). Only used when camera_type="usb".
        usb_resolution (tuple): (width, height) for USB camera. Default (640, 480).
    """

    # ================================ INIT ===============================================
    def __init__(self, queuesList, logger, debugger, show_preview=False,
                 camera_type="picamera", usb_device=0, usb_resolution=(640, 480)):
        super(threadCamera, self).__init__(pause=0.1)  # 10 FPS — suficiente para line following a ~5 FPS y dashboard
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        self.frame_rate = 5
        self.recording = False
        self.show_preview = show_preview
        self.camera_type = camera_type
        self.usb_device = usb_device
        self.usb_resolution = usb_resolution

        self.video_writer = ""

        self.recordingSender = messageHandlerSender(self.queuesList, Recording)
        self.mainCameraSender = messageHandlerSender(self.queuesList, mainCamera)
        self.serialCameraSender = messageHandlerSender(self.queuesList, serialCamera)

        self.subscribe()
        self._init_camera()
        self.queue_sending()
        self.configs()

    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""

        self.recordSubscriber = messageHandlerSubscriber(self.queuesList, Record, "lastOnly", True)
        self.brightnessSubscriber = messageHandlerSubscriber(self.queuesList, Brightness, "lastOnly", True)
        self.contrastSubscriber = messageHandlerSubscriber(self.queuesList, Contrast, "lastOnly", True)
        self.stateChangeSubscriber = messageHandlerSubscriber(self.queuesList, StateChange, "lastOnly", True)

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
        # if camera is not available, skip processing
        if self.camera is None:
            time.sleep(0.1)
            return
            
        try:
            recordRecv = self.recordSubscriber.receive()
            if recordRecv is not None: 
                self.recording = bool(recordRecv)
                if recordRecv == False:
                    self.video_writer.release() # type: ignore
                else:
                    fourcc = cv2.VideoWriter_fourcc( # type: ignore
                        *"XVID"
                    )  # You can choose different codecs, e.g., 'MJPG', 'XVID', 'H264', etc.
                    self.video_writer = cv2.VideoWriter(
                        "output_video" + str(time.time()) + ".avi",
                        fourcc,
                        self.frame_rate,
                        (2048, 1080),
                    )

        except Exception as e:
            print(f"\033[1;97m[ Camera ] :\033[0m \033[1;91mERROR\033[0m - {e}")

        try:
            if self.camera_type == "usb":
                mainRequest, serialRequest = self._capture_usb()
            else:
                mainRequest, serialRequest = self._capture_picamera()

            if mainRequest is None or serialRequest is None:
                return

            if self.recording == True:
                self.video_writer.write(mainRequest) # type: ignore

            # Show preview window if enabled
            if self.show_preview:
                preview_frame = cv2.resize(mainRequest, (1024, 540))  # type: ignore
                cv2.imshow("Camera Preview", preview_frame)  # type: ignore
                cv2.waitKey(1)  # type: ignore

            # Only encode serialCamera (640x384) — used by line following, sign detection, and dashboard.
            # mainCamera (2048x1080) is not consumed by any subscriber, so we skip encoding it entirely.
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
            _, serialEncodedImg = cv2.imencode(".jpg", serialRequest, encode_params) # type: ignore
            serialEncodedImageData = base64.b64encode(serialEncodedImg).decode("utf-8") # type: ignore

            if self._blocker.is_set():
                return

            self.serialCameraSender.send(serialEncodedImageData)
        except Exception as e:
            print(f"\033[1;97m[ Camera ] :\033[0m \033[1;91mERROR\033[0m - {e}")

    def _capture_picamera(self):
        """Capture frames from PiCamera (CSI). Returns (main_frame, serial_frame)."""
        mainRequest = self.camera.capture_array("main")
        serialRequest = self.camera.capture_array("lores")
        serialRequest = cv2.cvtColor(serialRequest, cv2.COLOR_YUV2BGR_I420)  # type: ignore
        return mainRequest, serialRequest

    def _capture_usb(self):
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
        mainRequest = frame.copy()
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
        """Initialize the camera. Supports picamera2 (CSI) and USB (OpenCV VideoCapture)."""

        if self.camera_type == "usb":
            self._init_usb_camera()
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
            print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;92mINFO\033[0m - PiCamera initialized successfully")
        except Exception as e:
            print(f"\033[1;97m[ Camera Thread ] :\033[0m \033[1;91mERROR\033[0m - Failed to initialize PiCamera: {e}")
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
        if self.camera is not None:
            if self.camera_type == "usb":
                # Stop the reader thread first
                self._usb_reader_running = False
                if hasattr(self, '_usb_reader_thread'):
                    self._usb_reader_thread.join(timeout=2)
                self.camera.release()
            else:
                self.camera.stop()
        if self.show_preview:
            cv2.destroyAllWindows()  # type: ignore
        super(threadCamera, self).stop()

    # =============================== CONFIG ==============================================
    def configs(self):
        """Callback function for receiving configs on the pipe.
        Note: Brightness/Contrast controls only work with PiCamera (picamera2 API).
        USB cameras ignore these for now (could be extended with cv2.VideoCapture props).
        """
        if self._blocker.is_set():
            return
        if self.brightnessSubscriber.is_data_in_pipe():
            message = self.brightnessSubscriber.receive()
            if self.debugger:
                self.logger.info(str(message))
            if self.camera_type != "usb" and self.camera is not None:
                self.camera.set_controls(
                    {
                        "AeEnable": False,
                        "AwbEnable": False,
                        "Brightness": max(0.0, min(1.0, float(message))), # type: ignore
                    }
                )
        if self.contrastSubscriber.is_data_in_pipe():
            message = self.contrastSubscriber.receive()
            if self.debugger:
                self.logger.info(str(message))
            if self.camera_type != "usb" and self.camera is not None:
                self.camera.set_controls(
                    {
                        "AeEnable": False,
                        "AwbEnable": False,
                        "Contrast": max(0.0, min(32.0, float(message))), # type: ignore
                    }
                )
        threading.Timer(1, self.configs).start()