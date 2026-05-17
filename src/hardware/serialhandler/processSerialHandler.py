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

import re
import serial
import serial.tools.list_ports
import threading
import time
from threading import Lock

from src.templates.workerprocess import WorkerProcess
from src.hardware.serialhandler.threads.filehandler import FileHandler
from src.hardware.serialhandler.threads.threadRead import threadRead
from src.hardware.serialhandler.threads.threadSimFeedback import threadSimFeedback
from src.hardware.serialhandler.threads.threadWrite import threadWrite
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.statemachine.systemMode import SystemMode
from src.core.bus.topics import STATE_CHANGE, SERIAL_CONNECTION_STATE

class processSerialHandler(WorkerProcess):
    """This process handle connection between NUCLEO and Raspberry PI.\n
    Args:
        queueList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool, optional): A flag for debugging. Defaults to False.
        example (bool, optional): A flag for running the example. Defaults to False.
    """

    # ===================================== INIT =========================================
    def __init__(self, queueList, logging, ready_event=None, dashboard_ready=None, debugging=False, example=False):
        # devFile = "/dev/ttyACM0"
        logFile = "temp/serial_history.log"

        self.logger = logging
        self.queuesList = queueList
        self.debugging = debugging
        self.example = example
        self.dashboard_ready = dashboard_ready

        # comm init
        self.serialCon = None
        self.serialConnected = False
        self.serialDevice = None
        self.serialLock = Lock()
        self.reconnecting = False
        self._last_port_scan_log = 0.0

        try:
            from config import MOTOR_OUTPUT
            self.motor_output = MOTOR_OUTPUT
        except ImportError:
            self.motor_output = "serial"

        self._init_subscribers()
        self._init_senders()

        # log file init
        self.historyFile = FileHandler(logFile)

        super(processSerialHandler, self).__init__(self.queuesList, ready_event)

    def __getstate__(self):
        state = self.__dict__.copy()
        # threading.Lock and serial.Serial are not picklable (needed for spawn).
        # serialCon is always None at startup and gets opened in run().
        state.pop('serialLock', None)
        state.pop('serialCon', None)
        state.pop('historyFile', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        from threading import Lock
        self.serialLock = Lock()
        self.serialCon = None
        logFile = "temp/serial_history.log"
        from src.hardware.serialhandler.threads.filehandler import FileHandler
        self.historyFile = FileHandler(logFile)

    def _init_subscribers(self):
        self.stateChangeSubscriber = messageHandlerSubscriber(self.queuesList, STATE_CHANGE, "lastOnly", True)
        self.serialConnectionStateSubscriber = messageHandlerSubscriber(self.queuesList, SERIAL_CONNECTION_STATE, "lastOnly", True)

    def _init_senders(self):
        self.serialConnectedSender = messageHandlerSender(self.queuesList, SERIAL_CONNECTION_STATE)

    def _safe_close_serial(self):
        """Safely close the serial connection with proper error handling."""
        if self.serialCon and hasattr(self.serialCon, 'is_open') and self.serialCon.is_open:
            try:
                self.serialCon.close()
            except (OSError, serial.SerialException) as e:
                print(f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;93mWARNING\033[0m - Error closing serial connection: {e}")
            except Exception as e:
                print(f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;91mERROR\033[0m - Unexpected error closing serial: {e}")

    def _try_serial_connection(self):
        """Try to connect to the serial device."""
        if self.motor_output == "zmq":
            # Sim mode: no Nucleo. Pretend connected so dashboard/threads behave normally.
            self.serialCon = None
            self.serialConnected = True
            self.serialDevice = "zmq://sim_bridge"
            return
        with self.serialLock:
            try:
                # clean up existing connection safely
                self._safe_close_serial()

                try:
                    from config import SERIAL_PORT, LIDAR_SERIAL_PORT
                except ImportError:
                    SERIAL_PORT = None
                    LIDAR_SERIAL_PORT = None

                configured_port = str(SERIAL_PORT).strip() if SERIAL_PORT else None
                lidar_port = str(LIDAR_SERIAL_PORT).strip() if LIDAR_SERIAL_PORT else None
                ports = list(serial.tools.list_ports.comports())

                if configured_port:
                    self.serialDevice = configured_port
                    selected_from = "URT_SERIAL_PORT"
                else:
                    # Match /dev/ttyACM* (Nucleo via USB) or /dev/ttyUSB*
                    # (USB-serial adapter). Prefer ACM because STM32 Nucleo
                    # boards usually enumerate there; USB devices can be LiDAR
                    # adapters and may accept writes while motors never move.
                    candidates = [
                        port
                        for port in ports
                        if re.match(r"/dev/tty(ACM|USB)\d+$", port.device)
                        and port.device != lidar_port
                    ]
                    candidates.sort(
                        key=lambda port: (
                            0 if re.match(r"/dev/ttyACM\d+$", port.device) else 1,
                            port.device,
                        )
                    )
                    self.serialDevice = candidates[0].device if candidates else None
                    selected_from = "autodetect"

                if self.serialDevice is None:
                    self.serialConnected = False
                    now = time.monotonic()
                    if self.debugging or (now - self._last_port_scan_log) >= 5.0:
                        available = ", ".join(
                            f"{port.device} ({port.description})" for port in ports
                        ) or "none"
                        print(
                            f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;93mWARNING\033[0m"
                            f" - No Nucleo serial port found. Available: {available}"
                        )
                        self._last_port_scan_log = now
                    return

                # Mantener `timeout=0.1` como en master. Un fork puso `timeout=0`
                # en algún momento y eso rompe la RX en el CDC ACM de la Jetson:
                # con O_NONBLOCK la cola de RX nunca entrega bytes (in_waiting=0
                # eterno) aunque la nucleo esté transmitiendo. Con 0.1 s pyserial
                # no setea O_NONBLOCK y el driver funciona como se espera.
                self.serialCon = serial.Serial(self.serialDevice, 115200, timeout=0.1)
                self.serialCon.reset_input_buffer()
                self.serialCon.reset_output_buffer()
                self.serialConnected = True
                available = ", ".join(
                    f"{port.device} ({port.description})" for port in ports
                ) or self.serialDevice
                print(
                    f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;92mINFO\033[0m"
                    f" - Connected to \033[94m{self.serialDevice}\033[0m"
                    f" via {selected_from}. Ports: {available}"
                )
                if selected_from == "autodetect" and re.match(r"/dev/ttyUSB\d+$", self.serialDevice):
                    print(
                        f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;93mWARNING\033[0m"
                        " - Selected a ttyUSB port. If KL/speed logs appear but the Nucleo"
                        " does not move, run with URT_SERIAL_PORT=/dev/ttyACM0 or a"
                        " stable /dev/serial/by-id/... path for the F401RE."
                    )

            except PermissionError as e:
                self._safe_close_serial()
                self.serialCon = None
                self.serialConnected = False
                print(
                    f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;91mERROR\033[0m"
                    f" - Permission denied on \033[94m{self.serialDevice}\033[0m: {e}"
                    f"\n  → Run: \033[1msudo usermod -a -G dialout $USER\033[0m then logout/login"
                    f"\n  → Or run the program with: \033[1msudo ./run.sh\033[0m"
                )
            except (serial.SerialException, FileNotFoundError, OSError) as e:
                self._safe_close_serial()
                self.serialCon = None
                self.serialConnected = False
                now = time.monotonic()
                if self.debugging or (now - self._last_port_scan_log) >= 5.0:
                    print(
                        f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;93mWARNING\033[0m"
                        f" - Serial connect failed on \033[94m{self.serialDevice}\033[0m: {e}"
                    )
                    self._last_port_scan_log = now

    def _try_reconnect(self):
        """Try to reconnect to serial device (called by timer)."""
        if self.reconnecting:
            return # another reconnection attempt is already in progress

        self.reconnecting = True

        self._try_serial_connection()

        if self.serialConnected:
            # reset thread error states after successful reconnection
            self.serialConnectedSender.send(True)
            self._reset_thread_error_states()
            self.reconnecting = False
        else:
            # schedule next attempt
            self.reconnecting = False
            threading.Timer(1, self._try_reconnect).start()

    def _reset_thread_error_states(self):
        """Reset error states in threads after successful reconnection."""
        if self.threads:
            for thread in self.threads:
                if hasattr(thread, 'last_error_time'):
                    thread.last_error_time = None

    def _wait_for_dashboard_and_notify(self):
        """Wait for dashboard to be ready, then notify connected state once without blocking init."""
        self.dashboard_ready.wait()
        if self.dashboard_ready.is_set():
            self.serialConnectedSender.send(self.serialConnected)


    def _handle_serial_disconnection(self):
        """Handle serial disconnection by pausing threads and starting reconnection."""
        if self.motor_output == "zmq":
            return

        with self.serialLock:
            # check if already handling disconnection
            if self.reconnecting or not self.serialConnected:
                return

            print(f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;93mWARNING\033[0m - Serial device disconnected")

            # mark as disconnected
            self.serialConnected = False

            # clean up serial connection safely
            self._safe_close_serial()

            self.serialCon = None

            threading.Timer(1, self._try_reconnect).start()

    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads."""
        self._try_serial_connection()

        if not self.serialConnected and self.motor_output != "zmq":
            print(f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;93mWARNING\033[0m - No serial connection found")
            threading.Timer(1, self._try_reconnect).start()

        if self.dashboard_ready is not None:
            if self.dashboard_ready.is_set():
                self.serialConnectedSender.send(self.serialConnected)
            else:
                threading.Thread(target=self._wait_for_dashboard_and_notify, daemon=True).start()

        super(processSerialHandler, self).run()
        self.historyFile.close()

    # ===================================== PROCESS WORK ==========================================
    def process_work(self):
        serialConnectionStateMessage = self.serialConnectionStateSubscriber.receive()
        if serialConnectionStateMessage is False:
            self._handle_serial_disconnection()

    # ================================ STATE CHANGE HANDLER ========================================
    def state_change_handler(self):
        message = self.stateChangeSubscriber.receive()
        if message is not None:
            modeDict = SystemMode[message].value["serial_handler"]["process"]

            if modeDict["enabled"] == True:
                # only resume if serial is connected
                if self.serialConnected:
                    self.resume_threads()

            elif modeDict["enabled"] == False:
                self.pause_threads()

    # ===================================== STOP ==========================================
    def stop(self):
        """Close the history file and stop the process."""
        # close serial connection
        with self.serialLock:
            if self.serialCon:
                try:
                    self.serialCon.close()
                except Exception as e:
                    print(f"\033[1;97m[ Serial Handler ] :\033[0m \033[1;93mWARNING\033[0m - Error closing serial port: {e}")

        super(processSerialHandler, self).stop()

    # ===================================== INIT TH =================================
    def _init_threads(self):
        """Initializes the read and the write thread.

        On hardware (``motor_output == "serial"``) only the read+write pair
        is needed — the Nucleo provides the encoder/IMU stream that fills
        ``CurrentSpeed`` / ``CurrentSteer`` / ``ImuData``.

        In simulator mode (``motor_output == "zmq"``) there is no Nucleo,
        so we additionally spawn ``threadSimFeedback`` which subscribes to
        the bridge's synthetic feedback PUB and re-emits the same IPC
        messages. ``threadRead`` is kept around (it no-ops because
        ``serialCon`` is None) so the rest of the pipeline that just
        expects "serial process running" stays happy — and so the
        ``EnableButton`` heartbeat from ``threadRead.queue_sending``
        keeps firing.
        """
        readTh = threadRead(self, self.historyFile, self.queuesList, self.logger, self.debugging)
        writeTh = threadWrite(self, self.historyFile, self.queuesList, self.logger, self.debugging, self.example)
        self.threads.extend([readTh, writeTh])

        # threadCalibration was previously instantiated inside processDashboard.
        # It now lives in this process: the calibration FSM commands the motors
        # via the same bus topics the dispatcher uses, and the wizard's reply
        # stream goes back over the bus instead of SocketIO.
        from src.hardware.serialhandler.threads.threadCalibration import threadCalibration
        from src.hardware.serialhandler.threads.threadModeRouter import threadModeRouter
        from src.hardware.serialhandler.threads.threadResourceMonitor import threadResourceMonitor
        calibTh = threadCalibration(self.queuesList, self.logger, self.debugging)
        modeRouterTh = threadModeRouter(self.queuesList, self.logger, self.debugging)
        monitorTh = threadResourceMonitor(self.queuesList, self.logger, self.debugging)
        self.threads.extend([calibTh, modeRouterTh, monitorTh])

        if self.motor_output == "zmq":
            simFeedbackTh = threadSimFeedback(self.queuesList, self.logger, self.debugging)
            self.threads.append(simFeedbackTh)

        try:
            from config import GPS_ENABLED
        except ImportError:
            GPS_ENABLED = False
        try:
            from src.utils.gps_mode import is_gps_disabled
            gps_disabled = is_gps_disabled()
        except Exception:
            gps_disabled = False

        print(
            f"[GPS-DBG] processSerialHandler._init_threads: "
            f"motor_output={self.motor_output!r} GPS_ENABLED={GPS_ENABLED} "
            f"gps_disabled={gps_disabled}",
            flush=True,
        )
        if GPS_ENABLED and not gps_disabled:
            from src.hardware.gps.threadLocSys import threadLocSys
            gpsTh = threadLocSys(self.queuesList, self.logger, self.debugging)
            self.threads.append(gpsTh)
            print(
                f"[GPS-DBG] processSerialHandler._init_threads: threadLocSys "
                f"INSTANTIATED y agregado a self.threads (total threads="
                f"{len(self.threads)})",
                flush=True,
            )
        else:
            print(
                f"[GPS-DBG] processSerialHandler._init_threads: GPS disabled, "
                f"threadLocSys NO ARRANCA",
                flush=True,
            )


# Standalone demo removed when the message backend was migrated to ZMQ —
# it depended on a queue-based gateway that no longer exists. Run
# ``python3 main.py`` to exercise this process inside the real graph.
