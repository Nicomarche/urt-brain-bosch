# Modelo de procesos y threads — decisión arquitectónica

## TL;DR

El brain corre en **3 procesos** (bus, camera, serial). Dentro de
processCamera viven ~12 threads compartiendo heap. NO partimos esos
threads en procesos separados pese a que el repo de referencia ROS los
trata como nodos independientes. Esta decisión es deliberada, y este
documento explica los trade-offs.

## Por qué NO un proceso por nodo

El plan original "estilo REF" sería N procesos = 1 por nodo ROS
(perception, control, planner, etc.). El REF logra aislamiento real:
un crash en uno no tira los demás. El costo es **latencia inter-proceso**:

| Comunicación | Latencia típica |
|---|---|
| Acceso a memoria compartida (mismo proceso) | < 100 ns |
| ZMQ inproc/ipc (mismo host) | 10-100 μs |
| TCP loopback | 100-500 μs |
| TCP red | 1-10 ms |

En el camino crítico **camera → lane → behavior → MPC → motor** el
budget es **< 50 ms total** (objetivo de 20 Hz). Con 4 saltos de proceso
a 200 μs cada uno son 800 μs sólo en transporte. Sumando serialización
JSON de mensajes y wake-ups del scheduler, ese overhead crece a varios
ms. Eso es aceptable en un robot urbano a 50 km/h pero **no** en BFMC
donde el horizonte de control es ~1 segundo a < 0.5 m/s.

## Aislamiento sin partir procesos

Plan G1+G2+G3 implementa supervisión por thread sin cambiar la
topología de procesos:

* **G1 — Supervisor por thread**: `ThreadWithStop` integra
  `ThreadSupervisor` que cuenta fallas consecutivas, aplica backoff
  exponencial (1s, 2s, 4s, ..., max 30s) y declara el thread `dead`
  tras 100 fallas totales.
* **G2 — Watchdog**: el supervisor publica heartbeat con `ok=False`
  reason="iter_stuck" cuando `time.monotonic() - last_iter_start >
  max_iter_s` (5 s default, override per-subclase).
* **F4 — Heartbeats**: `THREAD_HEARTBEAT_MSG` cada 1 Hz con
  `{thread_name, last_loop_ts, latency_ms, loop_count, ok}`. El widget
  `health_panel` lo dibuja en el GUI.
* **G3 — Shutdown coordinado**: `main.py` suscribe `WARNING_SIGNAL`;
  si llega `severity=crit` o `dead` de un thread crítico (planner,
  pose_estimator, dispatcher, threadWrite), inicia shutdown ordenado.

## Mapa thread → responsabilidad → criticidad

### processBus (1 proceso)
- `XPubXSubBroker` — broker ZMQ del bus pub/sub. **Crítico**.

### processCamera (1 proceso, ~12 threads)

| Thread | Responsabilidad | Criticidad |
|---|---|---|
| threadCamera | Captura RGB + JPEG encode | Crítica |
| threadLineFollowing | Visión clásica + sign detection | Crítica |
| threadLocalPerception | YOLO inferencia | No-crítica (degrade) |
| threadLaneObserver | LSTR neural lane | No-crítica (degrade) |
| threadStoplineObserver | Detección stopline | No-crítica |
| threadObjectTracker | MOTTracker SORT | No-crítica |
| threadPoseEstimator | DR + IMU fusion | **Crítica** |
| threadBehaviorPlanner | Scenarios + path/velocity | **Crítica** |
| threadMotorCommandDispatcher | MPC + safety gate + send motors | **Crítica** |
| threadLidar | Driver RPLidar + procesamiento | No-crítica |
| threadResourceMonitor | CPU/RAM/temp | No-crítica |
| threadVideoStreamer | UDP encode al GUI | No-crítica |

### processSerialHandler (1 proceso, ~5 threads)

| Thread | Responsabilidad | Criticidad |
|---|---|---|
| threadRead | Drena Nucleo serial (speed/IMU) | **Crítica** |
| threadWrite | Envía cmds al Nucleo | **Crítica** |
| threadCalibration | FSM de calibración | No-crítica |
| threadModeRouter | DrivingMode → StateMachine | Crítica |
| threadSimFeedback | Loop de feedback en sim | Crítica (sólo sim) |

## Comparación con REF (ROS Noetic)

| Aspecto | MI repo | REF (ROS) |
|---|---|---|
| Topología | 3 procesos × N threads | N procesos × 1 thread cada uno |
| Comunicación intra-grupo | Heap directo | TCP/UDP via roscore |
| Aislamiento de crashes | Por-thread supervisor (G1) | Por-proceso (sistema) |
| Recovery automático | Backoff exponencial G1 | roslaunch respawn |
| Latencia crítica | < 100 ns | 100-500 μs |
| Heartbeats globales | THREAD_HEARTBEAT_MSG (F4) | std_msgs/Empty roundtrip |
| Lifecycle | main.py orquesta start/stop | roslaunch + roscore |

## Cuándo SÍ partir procesos

Reservamos el "partir procesos" para tres casos:

1. **Aislamiento de drivers de hardware inestables** — ej. RPLidar
   conectado por USB que ocasionalmente se reset-ea. Vive en su
   propio thread dentro de processCamera; si se vuelve un problema,
   moverlo a processLidar separado.
2. **Cargas CPU-bound que no liberan GIL** — el YOLO inferencia ya
   libera el GIL en `cv2.dnn` y `onnxruntime`, así que el thread es OK.
3. **Servicios bloqueantes (web, gRPC)** — no aplica al brain.

## Referencias

- Plan G1+G2+F4+G3: `src/templates/supervisor.py`,
  `src/templates/threadwithstop.py`, `main.py` shutdown handler.
- Tests: `tests/templates/test_supervisor.py`.
- Topics nuevos: `THREAD_HEARTBEAT_MSG`, `WARNING_SIGNAL` (extendido).
