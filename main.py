# main.py
#
# Punto de entrada del stack URT Brain. Lanza 4 procesos independientes que
# se comunican vía bus pub/sub ZeroMQ (``src/core/bus``) — ya no hay colas
# multiprocessing.Queue ni Pipes:
#
#   1. processCamera     — perception + control (cámara, YOLO, lane fitting,
#                          MOT, EKF7, route, BehaviorPlanner, MotionController,
#                          dispatcher). El frame anotado sale por UDP, todo
#                          el resto por ZMQ.
#   2. processSerial     — UART al Nucleo STM32 (lee IMU/encoder, escribe
#                          motor).
#   3. processDashboard  — Flask + WebSocket para operación remota (durante
#                          la transición; Sprint 5 lo borra y el GUI habla
#                          directo al bus).
#   4. processBus        — broker XSUB/XPUB de ZeroMQ. Reemplaza al viejo
#                          processGateway. Corre el ``zmq.proxy`` en hilo C
#                          (libera el GIL).
#
# El flujo de control va:
#
#   cámara → percepción → localización → ruta → BehaviorPlanner →
#   MotionController (MPC) → dispatcher (safety_gate) → serial → Nucleo
#
# Una sola fuente de verdad por decisión:
#   - Steering sale SOLO de `MotionController.compute(BehaviorOutput, Pose)
#     → MotorCommand`.
#   - Speed sale SOLO de `BehaviorPlanner.plan(...).speed_profile[0]`,
#     ejecutada por el MPC.
#   - El `motor_command_dispatcher` es el ÚNICO writer de `SpeedMotor` /
#     `SteerMotor` en runtime de competencia.
#   - El `safety_gate` emite STOP automático si BehaviorOutput stale > 500
#     ms o PoseEstimate stale > 300 ms.
#
# Estructura de paquetes (post-Fase 6):
#   src/core/         — tipos compartidos (frozen dataclasses) + Protocols
#                       (DIP). Sin lógica.
#   src/perception/   — captura + lane + signs + tracking (MOT).
#   src/localization/ — EKF7 + relocalization.
#   src/routing/      — Lanelet HD map + Dijkstra route planner.
#   src/behavior/     — BehaviorPlanner + scenarios + velocity overlay.
#   src/control/      — MotionController + safety_gate + dispatcher.
#
# Para ejecutar:
#       chmod +x setup.sh
#       ./setup.sh
#       cd legacy/dashboard-frontend     # (legacy Angular UI, opcional)
#       npm start
#       (CTRL+C cuando arranque la UI)
#       cd ../..
#       python3 main.py
#
# Recomendado: usar la nueva GUI PyQt5
#       ./run.sh                          # Mac dev (sim) o Jetson (servidor)
#       ./run.sh --monitor <IP>           # Solo GUI conectada al auto remoto
#
# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved. (BSD-3 — texto original abajo)
# ===================================================================
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED "AS IS" — see project root for full BSD-3 text.

# ===================================== GENERAL IMPORTS ==================================

import sys
import time
import os
import psutil
import multiprocessing as mp

# macOS Apple Silicon: default 'spawn' breaks several closures and pickling,
# and many libs (cv2, picamera2 stubs, etc.) re-import on each child. Use fork
# to match Jetson/Linux behavior. Suppress Objective-C fork-safety warnings.
if sys.platform == "darwin":
    # spawn: each child gets a fresh Python interpreter — no stale libomp/OpenMP
    # thread state inherited from the parent.  This is the only reliable fix for
    # EXC_BAD_ACCESS in __kmp_suspend_initialize_thread on Apple Silicon when
    # torch + numpy/OpenBLAS each bundle their own libomp.dylib.
    # Jetson/Linux keeps the default (fork) because Linux fork + exec is safe and
    # the Tegra runtime doesn't ship duplicate libomp builds.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    # Limit OpenMP/BLAS thread pools in every spawned child — the env vars are
    # inherited by child processes through the OS environment.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

# Pin to CPU cores 0–3 (Linux-only; psutil.cpu_affinity not available on macOS).
if hasattr(psutil.Process(os.getpid()), "cpu_affinity"):
    available_cores = list(range(psutil.cpu_count()))
    psutil.Process(os.getpid()).cpu_affinity(available_cores)

sys.path.append(".")
from multiprocessing import Event
from src.utils.bigPrintMessages import BigPrint
import logging
import logging.handlers

logging.basicConfig(level=logging.INFO)

# ===================================== BUS ENDPOINTS ====================================
# Dual-bind the broker on both ipc:// (intra-robot, zero-copy) and
# tcp://0.0.0.0:5556/5557 so the PyQt GUI on a remote PC can connect to the
# same bus over LAN. Setting the env vars HERE (before any child spawns)
# means every process — broker, camera, serial — inherits the same config.
os.environ.setdefault(
    "URT_BUS_XSUB_BIND",
    "ipc:///tmp/urt-bus/xsub.sock,tcp://0.0.0.0:5556",
)
os.environ.setdefault(
    "URT_BUS_XPUB_BIND",
    "ipc:///tmp/urt-bus/xpub.sock,tcp://0.0.0.0:5557",
)
# Local peers (children) keep using ipc:// for low latency.
os.environ.setdefault("URT_BUS_XSUB_ENDPOINT", "ipc:///tmp/urt-bus/xsub.sock")
os.environ.setdefault("URT_BUS_XPUB_ENDPOINT", "ipc:///tmp/urt-bus/xpub.sock")

# ===================================== PROCESS IMPORTS ==================================

from src.core.bus.processBus import processBus
from src.hardware.camera.processCamera import processCamera
from src.hardware.serialhandler.processSerialHandler import processSerialHandler
from src.data.Semaphores.processSemaphores import processSemaphores
from src.data.TrafficCommunication.processTrafficCommunication import processTrafficCommunication
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber
from src.core.bus.topics import STATE_CHANGE
from src.statemachine.stateMachine import StateMachine
from src.statemachine.systemMode import SystemMode
from config import (
    CAMERA_TYPE, USB_DEVICE, USB_RESOLUTION, SHOW_CAMERA_PREVIEW,
    STREAM_CAMERA_TO_DASHBOARD, DEBUG_WINDOWS,
    ENABLE_SIGN_DETECTION, SIGN_DETECTION_ACTIONS, SIGN_MIN_CONFIDENCE,
    SIGN_MIN_BOX_AREA, SIGN_ACTION_COOLDOWN,
    PICAMERA_HDR_ENABLED, PICAMERA_HDR_ALWAYS_ON, PICAMERA_HDR_GLARE_THRESHOLD,
    JETSON_SENSOR_ID, JETSON_CAPTURE_RESOLUTION, JETSON_OUTPUT_RESOLUTION,
    JETSON_FRAMERATE, JETSON_FLIP_METHOD,
    AUTO_STATE_RUN_IN_SIM,
)
# ``STREAM_CAMERA_TO_DASHBOARD`` kept as a config flag so users can disable
# the video streamer (e.g. headless competition runs). The dashboard process
# itself was removed; the flag now gates ``UdpVideoStreamerThread``.

# ------ New component imports starts here ------#


# ------ New component imports ends here ------#

# ===================================== SHUTDOWN PROCESS ====================================

def shutdown_process(process, timeout=1):
    """Helper function to gracefully shutdown a process."""
    process.join(timeout)
    if process.is_alive():
        print(f"The process {process} cannot normally stop, it's blocked somewhere! Terminate it!")
        process.terminate()  # force terminate if it won't stop
        process.join(timeout)  # give it a moment to terminate
        if process.is_alive():
            print(f"The process {process} is still alive after terminate, killing it!")
            process.kill()  # last resort
    print(f"The process {process} stopped")

# ===================================== PROCESS MANAGEMENT ==================================

def manage_process_life(process_class, process_instance, process_args, enabled, allProcesses):
    """Start or stop a process based on the enabled flag."""
    if enabled:
        if process_instance is None:
            process_instance = process_class(*process_args)
            allProcesses.append(process_instance)
            process_instance.start()
    else:
        if process_instance is not None and process_instance.is_alive():
            shutdown_process(process_instance)
            allProcesses.remove(process_instance)
            process_instance = None
    return process_instance 

# ======================================== SETTING UP ====================================

def main():
    print(BigPrint.PLEASE_WAIT.value)
    allProcesses = list()
    allEvents = list()

    # ``queueList`` is now a thin dict kept for signature compatibility with
    # process constructors (every WorkerProcess subclass takes it positionally).
    # The new bus does not consume it — the legacy ``messageHandlerSender`` /
    # ``messageHandlerSubscriber`` (now thin shims over ZMQ) ignore it. The
    # only entries that matter are the StateMachine Manager proxies attached
    # below, which subprocesses use to read shared state without bus latency.
    queueList: dict = {}
    logger = logging.getLogger()

    # ===================================== INITIALIZE BUS ==================================
    # The broker must come up BEFORE any other process spawns: ZMQ pub/sub
    # is a "slow joiner" — messages sent before the subscriber's filter
    # reaches the broker are silently dropped. Starting the broker first
    # gives every child process a stable endpoint to connect to.
    processBusInstance = processBus(queueList, logger)
    processBusInstance.start()
    # Tiny pause so the broker's ipc:// sockets are bound before children
    # try to connect. 200 ms is overkill for ipc:// (typically < 5 ms) but
    # cheap relative to the rest of startup.
    time.sleep(0.2)

    # ===================================== INITIALIZE ==================================

    stateChangeSubscriber = messageHandlerSubscriber(queueList, STATE_CHANGE, "lastOnly", True)
    StateMachine.initialize_shared_state(queueList)

    # Exponer los proxies del Manager en queueList para que los subprocesos
    # (dispatcher, etc.) puedan leer el estado directamente sin pasar por el
    # bus. ``DictProxy`` y ``LockProxy`` son serializables y se pasan via
    # pickle al subprocess durante el spawn.
    queueList["__sm_state__"] = StateMachine._shared_state
    queueList["__sm_lock__"] = StateMachine._process_lock

    # ===================================== INITIALIZE PROCESSES ==================================

    # processDashboard is gone — the GUI talks directly to the bus over
    # tcp://...:5556/5557 (commands/telemetry) and over UDP (video). Auth,
    # heartbeat and the SocketIO bridge no longer have a place in the
    # control plane.

    # Initializing camera
    camera_ready = Event()
    processCameraInstance = processCamera(queueList, logger, camera_ready, debugging = False,
                                          camera_type=CAMERA_TYPE, usb_device=USB_DEVICE,
                                          usb_resolution=USB_RESOLUTION,
                                          jetson_sensor_id=JETSON_SENSOR_ID,
                                          jetson_capture_resolution=JETSON_CAPTURE_RESOLUTION,
                                          jetson_output_resolution=JETSON_OUTPUT_RESOLUTION,
                                          jetson_framerate=JETSON_FRAMERATE,
                                          jetson_flip_method=JETSON_FLIP_METHOD,
                                          show_preview=SHOW_CAMERA_PREVIEW,
                                          debug_windows=DEBUG_WINDOWS,
                                          picamera_hdr_enabled=PICAMERA_HDR_ENABLED,
                                          picamera_hdr_always_on=PICAMERA_HDR_ALWAYS_ON,
                                          picamera_hdr_glare_threshold=PICAMERA_HDR_GLARE_THRESHOLD,
                                          stream_to_dashboard=STREAM_CAMERA_TO_DASHBOARD,
                                          enable_sign_detection=ENABLE_SIGN_DETECTION,
                                          sign_detection_actions=SIGN_DETECTION_ACTIONS,
                                          sign_min_confidence=SIGN_MIN_CONFIDENCE,
                                          sign_min_box_area=SIGN_MIN_BOX_AREA,
                                          sign_action_cooldown=SIGN_ACTION_COOLDOWN)

    # # Initializing semaphores
    # semaphore_ready = Event()
    # processSemaphore = processSemaphores(queueList, logging, semaphore_ready, debugging = False)

    # # Initializing GPS
    # traffic_com_ready = Event()
    # processTrafficCom = processTrafficCommunication(queueList, logging, 3, traffic_com_ready, debugging = False)

    # Initializing serial connection NUCLEO - > PI
    serial_handler_ready = Event()
    processSerialHandlerInstance = processSerialHandler(
        queueList, logger, serial_handler_ready, None, debugging=False,
    )

    allProcesses.extend([processCameraInstance, processSerialHandlerInstance])
    allEvents.extend([camera_ready, serial_handler_ready])

    # ------ New component initialize starts here ------#

    # ------ New component initialize ends here ------#

    # ===================================== START PROCESSES ==================================

    for process in allProcesses:
        process.daemon = True
        process.start()

    # ===================================== STAYING ALIVE ====================================

    blocker = Event()
    try:
        # wait for all events to be set
        # Timeout de 60 s por evento: si algún proceso tarda más de un minuto
        # en iniciar (ej. carga de modelo ONNX lenta), continuamos igual
        # para que el dispatcher entre en AUTO. El proceso tardío eventualmente
        # se pone al día y empieza a funcionar.
        for event in allEvents:
            event.wait(timeout=60.0)

        # apply starting mode
        StateMachine.initialize_starting_mode()

        # En sim, arrancamos directo en AUTO para no tener que clickear el
        # botón del dashboard en cada restart. Espera 2 s para que todos los
        # subscribers (dispatcher, threadWrite, etc.) ya hayan arrancado y
        # registrado sus listeners antes del primer StateChange.
        if AUTO_STATE_RUN_IN_SIM:
            time.sleep(2.0)
            sm = StateMachine.get_instance()
            sm.request_mode("dashboard_auto_button")

        time.sleep(10)
        print(BigPrint.C4_BOMB.value)
        print(BigPrint.PRESS_CTRL_C.value)

        while True:
            message = stateChangeSubscriber.receive()
            if message is not None:
                # modeDictSemaphore = SystemMode[message].value["semaphore"]["process"]
                # modeDictTrafficCom = SystemMode[message].value["traffic_com"]["process"]

                #processSemaphore = manage_process_life(processSemaphores, processSemaphore, [queueList, logging, semaphore_ready, False], modeDictSemaphore["enabled"], allProcesses)
                # processTrafficCom = manage_process_life(processTrafficCommunication, processTrafficCom, [queueList, logging, 3, traffic_com_ready, False], modeDictTrafficCom["enabled"], allProcesses)
                pass

            blocker.wait(0.1)

    except KeyboardInterrupt:
        print("\nCatching a KeyboardInterruption exception! Shutdown all processes.\n")

        for proc in reversed(allProcesses):
            proc.stop()
        processBusInstance.stop()

        # wait for all processes to finish before exiting
        for proc in reversed(allProcesses):
            shutdown_process(proc)
        shutdown_process(processBusInstance)


if __name__ == "__main__":
    main()
