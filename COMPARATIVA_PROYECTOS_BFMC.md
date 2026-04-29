# Comparativa Detallada de Proyectos BFMC

Documento generado el 15 de abril de 2026.

Se comparan tres proyectos orientados a la competencia **Bosch Future Mobility Challenge (BFMC)**:

| Abreviatura | Proyecto | Ubicación |
|-------------|----------|-----------|
| **URT** | Mi proyecto (URT Brain) | `/Users/luciogarcia/urt-brain-bosch` |
| **REF** | Proyecto de referencia URT | `/Users/luciogarcia/urt-ref/urt-brain-bosch` |
| **SHUBH** | Bfmc-Autonomous-Vehicle | `github.com/Shubh131102/Bfmc-Autonomous-Vehicle` |

---

## 1. Resumen Ejecutivo

| Aspecto | URT (Mi Proyecto) | REF | SHUBH |
|---------|-------------------|-----|-------|
| **Madurez** | Produccion, muy completo | Produccion, muy completo | Template/esqueleto documentacional |
| **Lenguaje principal** | Python | C++ / Python | Python |
| **Middleware** | Ninguno (multiprocessing propio) | ROS Noetic (ROS 1) | ROS2 Humble |
| **Hardware compute** | Raspberry Pi 5 | Jetson Orin Nano | Raspberry Pi |
| **Camara** | RPi CSI / USB | Intel RealSense D455 (RGB-D + IMU) | Camara USB generica |
| **Microcontrolador** | Nucleo STM32 | STM32 L476 (mbed OS) | No especificado |
| **GPU remota** | Si (AI Server via WebSocket) | No (GPU local en Jetson) | No |
| **Lineas de codigo clave** | ~15,000+ (solo threadLineFollowing: 12,680) | ~10,000+ (C++ distribuido en modulos) | ~500 (configs + launch, sin implementacion real) |

---

## 2. Arquitectura del Sistema

### URT (Mi Proyecto)
```
RPi 5 (main.py — multiprocessing)
├── processCamera
│   ├── threadCamera (captura)
│   ├── threadLineFollowing (12,680 lineas)
│   ├── threadLocalPerception
│   ├── threadVisualController
│   ├── threadLaneObserver
│   ├── threadSignDetection
│   ├── threadTracking (dead reckoning)
│   └── threadPoseEstimator
├── processSerialHandler (UART 9600 baud)
├── processDashboard (Angular SPA + Flask)
└── processGateway (enrutamiento de mensajes)
         │
    WebSocket ──► PC/GPU (AI Server: HybridNets, Supercombo, YOLOv8)
         │
    UART ──► Nucleo STM32 (motor + servo)
```

- **IPC**: Colas de mensajes con 5 niveles de prioridad + buffers compartidos thread-safe.
- **Patron**: Pub/Sub propio con `messageHandlerSender`/`messageHandlerSubscriber`.
- **Estado**: Maquina de estados (DEFAULT → AUTO → MANUAL → STOP).

### REF
```
Jetson Orin Nano (ROS Noetic)
├── Perception (lane.cpp, CameraNode.cpp)
├── Control (Controller.cpp — FSM de 8+ estados)
├── Planning (PathPlanner.cpp — Dijkstra + splines)
├── Localization (robot_localization EKF — 15 estados)
├── Communication (TCP client/server binario)
├── GUI (PyQt5 + SQLite)
├── Persistence (SQLite3)
└── Simulation (Gazebo)
         │
    ROS Topics ──► Comunicacion inter-nodos
         │
    Serial 115200 baud ──► STM32 L476 (mbed OS)
```

- **IPC**: ROS Topics y Services estandar.
- **Estado**: FSM en C++ con 8+ estados (INIT, AUTONOMOUS, PARKING, etc.).
- **Simulacion**: Gazebo integrado con modelo del vehiculo.

### SHUBH
```
Raspberry Pi (ROS2 Humble)
├── bfmc_perception (CNN signs + lane + fusion)
├── bfmc_control (FSM 5 estados + safety monitor)
├── bfmc_planning (A* + DWA)
├── bfmc_msgs (mensajes custom)
└── Docker container
         │
    ROS2 Topics ──► Comunicacion inter-nodos
```

- **IPC**: ROS2 DDS.
- **Estado**: FSM con 5 estados (CRUISE, APPROACHING_STOP, STOPPED, YIELD, EMERGENCY_BRAKE).
- **Nota**: Solo estructura documentada; las implementaciones reales no estan en el repositorio.

---

## 3. Deteccion de Carriles (Lane Detection)

| Caracteristica | URT | REF | SHUBH |
|---------------|-----|-----|-------|
| **Metodo clasico** | CLAHE + HSV + Sobel/Canny + Hough + Sliding Window | Histograma columnar en escala de grises | ROI-based (solo documentado) |
| **Deep Learning local** | LSTR (Transformer, ONNX, 5 resoluciones) + YOLOv8 segmentacion | No | No |
| **Deep Learning remoto** | HybridNets + Supercombo (GPU Server via WebSocket) | No aplica (GPU local) | No |
| **Modos de fusion** | Hybrid (OpenCV 40% + LSTR 60%), bonus ×1.2 si coinciden | Unico (histograma) | Unico (CNN documentada) |
| **Deteccion de linea de parada** | Si (BEV + mapa semantico) | Si (segmentacion en imagen) | No especificado |
| **Lineas de codigo** | ~12,680 (threadLineFollowing.py) | ~500 (lane.cpp) | 0 (no implementado) |
| **Filtrado de ruido** | Rechaza >40 lineas, saltos >80px error, >15° steer | Suavizado temporal entre frames | No especificado |

**Ventajas URT**: Seis modos de deteccion seleccionables desde el dashboard, fusion multi-metodo con ponderacion configurable, adaptacion a iluminacion via CLAHE.

**Ventajas REF**: Simplicidad y velocidad (histograma columnar es muy rapido), integrado directamente como nodo ROS compilado en C++.

---

## 4. Deteccion de Senales de Trafico

| Caracteristica | URT | REF | SHUBH |
|---------------|-----|-----|-------|
| **Modelo local** | MobilenetV2 SSD (TFLite) | YOLOFastestV2 (NCNN, CPU) | CNN PyTorch (documentado) |
| **Modelo GPU** | YOLOv8 (remoto via WebSocket) | YOLOv8/YOLOv11 (TensorRT local) | No |
| **Tracking temporal** | Cooldown 15s por tipo de senal | SORT (Hungarian + Kalman) | No especificado |
| **Senales soportadas** | Stop, No Entry, Semaforo (R/Y/G), Crosswalk, Vel 20/30, Highway, Parking | Stop, senales de trafico, semaforos (RGB), vehiculos, peatones | 12 clases (documentadas) |
| **Deteccion de semaforos** | Si (color integrado en deteccion) | Si (clasificacion de color RGB separada) | Si (documentado) |
| **Deteccion de peatones** | No | Si (YOLO + LiDAR depth) | Si (LiDAR, documentado) |
| **Deteccion de vehiculos** | No | Si (YOLO + tracking SORT) | No especificado |
| **Area minima de accion** | 1% del frame (ignora senales lejanas) | Umbral de confianza por clase | 0.7 confianza (documentado) |
| **Frecuencia** | Variable (depende del modo) | ~30 Hz (CPU) / ~60 Hz (TensorRT) | 20 Hz (documentado) |

**Ventajas URT**: Arquitectura dual local/remota que permite funcionar sin GPU dedicada. Cooldown anti-duplicados robusto.

**Ventajas REF**: TensorRT nativo en Jetson (60 Hz), tracking SORT con filtro de Kalman por objeto, deteccion de vehiculos y peatones con profundidad RealSense.

---

## 5. Control de Direccion y Velocidad

| Caracteristica | URT | REF | SHUBH |
|---------------|-----|-----|-------|
| **Control lateral** | Stanley + MPC lateral opcional | MPC completo (Acados) | No implementado |
| **Stanley** | Kp=25, Ki=1, Kd=4, dead zone 50px, anti-windup | No usa Stanley | N/A |
| **MPC** | Horizonte N=10, dt=0.033s, wheelbase 0.258m | Acados, 25-500 muestras de horizonte, T=0.1s | N/A |
| **Modelo dinamico MPC** | Bicicleta cinematico (lateral solo) | Bicicleta cinematico 7 estados (x,y,yaw,v,steer,ax,ay) | N/A |
| **Velocidad adaptativa** | Basada en angulo de giro (<10°=max, >15°=min, rampa +0.5/frame) | Ratios por escenario (highway, curva) | max_speed=0.6 m/s (documentado) |
| **Compensacion de offtracking** | Si (geometria Ackermann, solo en curvas) | No explicito | No |
| **Maquina de estados curva** | STRAIGHT→ENTERING→IN_CURVE→EXITING (4 estados) | Implicita en FSM principal | No |
| **Recuperacion de curva** | Auto-reversa si max steering >8 frames | No explicito | No |

**Ventajas URT**: Control mas fino con Stanley+MPC dual, anticipacion de curvas conocidas (radios BFMC 66.5/103.5 cm), compensacion de offtracking.

**Ventajas REF**: MPC mucho mas sofisticado con Acados (solver optimizado, hasta 500 muestras de horizonte), modelo de 7 estados mas completo.

---

## 6. Navegacion y Localizacion

| Caracteristica | URT | REF | SHUBH |
|---------------|-----|-----|-------|
| **Metodo principal** | Dead reckoning (bicicleta + RK4) | EKF multi-sensor (robot_localization) | EKF (documentado) |
| **Sensores de pose** | Encoder (velocidad) + IMU (heading) del Nucleo | RealSense IMU + encoder + GPS opcional | Camara + LiDAR + IMU (documentado) |
| **GPS** | No (GPS-free por diseno) | Opcional (integrado en EKF) | Si (documentado) |
| **Mapa** | GraphML de waypoints + track_semantics.json | GraphML de waypoints (XML) | No especificado |
| **Planificacion de ruta** | Interpolacion de waypoints (paso 0.05m) | Dijkstra/A* + splines B-cubicas | A* + DWA (documentado) |
| **Relocalizacion** | Visual (ganancia 0.18) + semantica (tolerancia 0.45m) | Lane/sign-based yaw reset | No especificado |
| **Fusion yaw** | EKF de yaw (IMU absoluto + cinematico) | EKF de 15 estados | EKF (documentado) |
| **Precision integracion** | RK4 (4to orden) | EKF con prediccion IMU | No implementado |
| **Deteccion de linea de parada** | BEV + nodos de precision del mapa (lookahead 0.1m) | Segmentacion en imagen | No especificado |

**Ventajas URT**: Funciona sin GPS (critico si no hay cobertura), relocalizacion visual y semantica, integracion RK4 de alta precision.

**Ventajas REF**: EKF de 15 estados es matematicamente mas robusto, fusion multi-sensor verdadera (no solo yaw), GPS como referencia absoluta opcional, simulacion con GPS ruidoso en Gazebo.

---

## 7. Estacionamiento Autonomo

| Caracteristica | URT | REF | SHUBH |
|---------------|-----|-----|-------|
| **Implementacion** | 10 estados (maquina de estados detallada) | Si (MPC separado para reversa) | No |
| **Metodo** | Estacionamiento paralelo inverso basado en distancia | MPC con formulacion especifica de parking | N/A |
| **Deteccion de espacio** | Mascara YOLOv8 (segmentacion de parking) | Deteccion + logica de terminacion | N/A |
| **Calibracion** | Distancias fijas (55cm adelante, 75cm reversa entrada, 45cm alineacion) | Parametros YAML configurables | N/A |
| **Fallback** | Temporizadores si el encoder no esta disponible | No explicito | N/A |

**Ventajas URT**: Maquina de estados muy detallada con 10 fases, fallback por timeout.

**Ventajas REF**: MPC para parking permite trayectorias mas suaves y optimas.

---

## 8. Dashboard / Interfaz de Usuario

| Caracteristica | URT | REF | SHUBH |
|---------------|-----|-----|-------|
| **Tecnologia** | Angular SPA + Flask + WebSocket | PyQt5 nativa + TCP server | RViz + RQT (estandar ROS2) |
| **Control manual** | Si (velocidad, direccion, modo) | Si (start/stop, replan, override) | No explicito |
| **Stream de camara** | MJPEG opcional en dashboard | RGB + depth en GUI | Solo RViz |
| **Telemetria en tiempo real** | Si (velocidad, posicion, angulo) | Si (velocidad, posicion, steering, waypoints) | Solo topics ROS2 |
| **Mapa interactivo** | Si (con nodos del track) | Si (waypoints, objetos, trayectoria) | No |
| **Base de datos** | No | SQLite3 (historial, calibracion, topologia) | No |
| **Acceso remoto** | Si (web, cualquier navegador) | Si (TCP + HTTP/WebSocket) | Docker + X11 forwarding |

**Ventajas URT**: Dashboard web accesible desde cualquier dispositivo con navegador, no requiere instalacion.

**Ventajas REF**: GUI nativa mas robusta con visualizacion de profundidad, base de datos para historial y calibracion.

---

## 9. Comunicacion Serial (MCU)

| Caracteristica | URT | REF | SHUBH |
|---------------|-----|-----|-------|
| **Baudrate** | 9600 | 115200 | No especificado |
| **Protocolo** | ASCII simple (`str(speed*10)`, `str(angle)`) | ASCII + binario (IMU, encoder) | ROS2 topics |
| **Firmware MCU** | No incluido en el repo (Nucleo STM32) | Completo (mbed OS, tasks RTOS) | No incluido |
| **Sensores MCU** | Encoder, IMU | BNO055 IMU (I2C), AS5048A encoder (PWM), servo feedback | No especificado |
| **Filtrado MCU** | No (se filtra en RPi) | Butterworth en firmware, PID en servo | N/A |

**Ventajas URT**: Protocolo simple y facil de depurar.

**Ventajas REF**: 12x mas rapido (115200 vs 9600), firmware completo incluido con filtrado Butterworth en MCU, scheduler pseudo-RTOS.

---

## 10. Infraestructura de Pruebas y CI/CD

| Caracteristica | URT | REF | SHUBH |
|---------------|-----|-----|-------|
| **Tests unitarios** | 13 archivos pytest (tracking, Stanley, pipeline, etc.) | Logs historicos de pruebas manuales | Scripts de analisis (analyze_logs.py) |
| **Simulacion** | No incluida | Gazebo (vehiculo, pista, obstaculos, GPS ruidoso) | Gazebo (documentado en launch files) |
| **CI/CD** | No | Jenkinsfile | Docker build |
| **Calibracion** | Templates en `/calibration/` | YAML + SQLite3 | Script calibrate_camera.py |
| **Logging** | 5 niveles de prioridad, archivos en `/temp/` | ROS logging + test_log/ | ROS2 logging |

**Ventajas URT**: Suite de tests automatizados mas completa (13 tests unitarios cubriendo modulos criticos).

**Ventajas REF**: Simulacion Gazebo completa, CI/CD con Jenkins, logs historicos de pruebas en pista real.

---

## 11. Despliegue y Servicios del Sistema

| Caracteristica | URT | REF | SHUBH |
|---------------|-----|-----|-------|
| **Auto-inicio** | systemd services (brain, dashboard, WiFi fallback) | ROS launch files | Docker Compose |
| **Instalacion** | `setup.sh` + `pip install` | `catkin build` + dependencias manuales | `colcon build` + Docker |
| **Contenedorizacion** | No | No | Si (Docker) |
| **WiFi fallback** | Si (servicio systemd) | No explicito | Host networking (Docker) |

---

## 12. Tabla Comparativa de Modelos de IA

| Modelo | URT | REF | SHUBH |
|--------|-----|-----|-------|
| **LSTR (Transformer)** | Si (ONNX, 5 resoluciones, local) | No | No |
| **YOLOv8** | Si (remoto GPU + local segmentacion) | Si (TensorRT local) | No |
| **YOLOv11** | No | Si (TensorRT local) | No |
| **YOLOFastestV2** | No | Si (NCNN CPU) | No |
| **MobilenetV2 SSD** | Si (TFLite local) | No | No |
| **HybridNets** | Si (remoto GPU, PyTorch) | No | No |
| **Supercombo** | Si (remoto GPU, ONNX, recurrente GRU) | No | No |
| **CNN custom** | No | No | Si (PyTorch, documentado) |
| **Runtime local** | ONNX Runtime, TFLite | TensorRT, NCNN | PyTorch (documentado) |
| **Runtime remoto** | PyTorch + ONNX (WebSocket) | N/A | N/A |

---

## 13. Diferencias Clave Resumidas

### URT vs REF

| Dimension | URT Mejor | REF Mejor |
|-----------|-----------|-----------|
| **Flexibilidad de deteccion** | 6 modos de lane detection intercambiables | — |
| **Independencia de GPU** | Funciona sin GPU local (offload a servidor) | — |
| **Dashboard web** | Accesible desde cualquier navegador | — |
| **Tests automatizados** | 13 tests pytest | — |
| **Documentacion en espanol** | Informe tecnico completo | — |
| **GPS-free** | Funciona sin GPS por diseno | — |
| **MPC** | — | Acados completo, 7 estados, hasta 500 muestras |
| **Sensor depth** | — | RealSense D455 (profundidad + IMU integrado) |
| **Deteccion de objetos** | — | Vehiculos + peatones con profundidad |
| **Simulacion** | — | Gazebo completo con modelo del carro |
| **Firmware MCU** | — | Incluido, con filtrado y pseudo-RTOS |
| **CI/CD** | — | Jenkins pipeline |
| **Velocidad serial** | — | 115200 baud (12x mas rapido) |
| **Localizacion** | — | EKF de 15 estados, fusion multi-sensor real |
| **Inferencia GPU local** | — | TensorRT nativo en Jetson (60 Hz) |

### URT vs SHUBH

| Dimension | Nota |
|-----------|------|
| **Completitud** | URT es un sistema completo y funcional; SHUBH es un esqueleto/template con documentacion pero sin implementacion real |
| **Codigo** | URT tiene 15,000+ lineas de codigo funcional; SHUBH tiene ~500 lineas (configs y launch files) |
| **Middleware** | URT usa multiprocessing propio (sin dependencia de ROS); SHUBH requiere ROS2 Humble |
| **Contenedorizacion** | SHUBH tiene Docker; URT no |
| **Documentacion de concepto** | SHUBH documenta bien la arquitectura deseada (metricas: 85% compliance, 92% deteccion, 88% sim-to-real) |

### REF vs SHUBH

| Dimension | Nota |
|-----------|------|
| **Completitud** | REF es un sistema completo; SHUBH es un template |
| **Middleware** | REF usa ROS1 Noetic; SHUBH usa ROS2 Humble (mas moderno) |
| **MPC** | REF tiene Acados MPC completo; SHUBH no tiene implementacion |
| **Hardware** | REF en Jetson Orin Nano (mucho mas potente que RPi) |

---

## 14. Oportunidades de Mejora para URT (Mi Proyecto)

Basado en las ventajas observadas en REF y SHUBH:

1. **Velocidad serial**: Considerar subir de 9600 a 115200 baud para reducir latencia de comunicacion con el MCU.
2. **Deteccion de vehiculos/peatones**: REF detecta vehiculos y peatones con profundidad; URT no los detecta.
3. **Sensor de profundidad**: Una RealSense agregaria deteccion de obstaculos 3D y IMU integrado de alta calidad.
4. **Simulacion**: Integrar un entorno de simulacion (Gazebo o similar) para probar sin el carro fisico.
5. **MPC mas completo**: El MPC de REF con Acados es significativamente mas sofisticado (7 estados, horizontes largos).
6. **Firmware MCU en el repo**: Incluir el firmware del Nucleo para tener todo el sistema versionado.
7. **Contenedorizacion**: Docker facilitaria el setup en nuevas maquinas (como hace SHUBH).
8. **CI/CD**: Agregar pipeline de integracion continua (como el Jenkinsfile de REF).
9. **Localizacion multi-sensor**: Considerar un EKF completo como el de REF (15 estados) en lugar de solo fusion de yaw.
10. **Safety monitor independiente**: SHUBH documenta un monitor de seguridad con override prioritario — URT podria beneficiarse de uno similar.

---

## 15. Fortalezas Unicas de URT (Mi Proyecto)

Cosas que ninguno de los otros dos proyectos tiene:

1. **Seis modos de lane detection** intercambiables en tiempo real desde el dashboard.
2. **Arquitectura distribuida RPi + GPU remota** via WebSocket — funciona con hardware barato.
3. **Modelos de vanguardia remotos**: Supercombo (recurrente con GRU), HybridNets (multi-tarea).
4. **Parking autonomo de 10 estados** con fallback por timeout.
5. **Relocalizacion semantica** usando landmarks del mapa (tolerancia 0.45m).
6. **Anticipacion de curvas** basada en radios conocidos del track BFMC.
7. **Suite de tests automatizados** mas completa de los tres proyectos.
8. **Dashboard web** accesible desde cualquier dispositivo.
9. **Maquina de estados de curva** dedicada (4 estados con recuperacion automatica).
10. **INFORME_TECNICO.md** completo en espanol (39 KB de documentacion tecnica detallada).
