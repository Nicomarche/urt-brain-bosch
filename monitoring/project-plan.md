# Project Plan - BFMC Brain (URT)

## Información General

- **Proyecto:** Bosch Future Mobility Challenge - Sistema de Control Vehicular
- **Equipo:** URT (Universidad Rafael Urdaneta)
- **Plataforma:** Jetson Nano/Orin + STM32 Nucleo (anteriormente Raspberry Pi)
- **Repositorio:** urt-brain-bosch

---

## 1. Descripción del Proyecto

Sistema de control autónomo para vehículo a escala en la competencia BFMC. El sistema incluye:

- Control de movimiento (velocidad y dirección)
- Procesamiento de imagen con cámara (CSI IMX219 via GStreamer/Jetson)
- Seguimiento de línea automático con múltiples modos de detección
- Servidor de inferencia IA remoto con GPU (aiserver)
- Percepción local unificada con modelo YOLO (carriles + señales)
- Dashboard web para monitoreo y control remoto
- Comunicación con servidores de tráfico y semáforos
- Múltiples modos de operación (Manual, Autónomo, Legacy)
- Maniobra de estacionamiento paralelo autónomo
- Conducción en autopista con velocidad elevada
- Detección y respuesta a señales de tráfico

---

## 2. Arquitectura del Sistema

### 2.1 Componentes Hardware
- Jetson Nano / Jetson Orin (procesamiento principal)
- STM32 Nucleo (control de motores)
- Cámara CSI IMX219 (visión, 1280×720 → escala a 640×480)
- IMU (orientación)
- Motores DC (tracción y dirección)
- PC/servidor externo con GPU (para aiserver, opcional)

### 2.2 Módulos de Software

| Módulo | Ubicación | Función |
|--------|-----------|---------|
| Main | `main.py` | Orquestador de procesos |
| Config | `config.py` | Parámetros globales del sistema |
| Camera | `src/hardware/camera/` | Captura y procesamiento de imagen |
| Line Following | `src/hardware/camera/threads/threadLineFollowing.py` | Seguimiento de carril (OpenCV/Stanley/LSTR/AI) |
| Local Perception | `src/hardware/camera/threads/localPerceptionEngine.py` | Motor YOLO unificado (carriles + señales) |
| Sign Detector | `src/hardware/camera/threads/signDetector.py` | Detector TFLite MobilenetV2 SSD |
| Sign Actions | `src/hardware/camera/threads/signActions.py` | Ejecutor de acciones por señales |
| LSTR Detector | `src/hardware/camera/threads/lstrDetector.py` | Detector de carriles con Transformers (ONNX) |
| Serial Handler | `src/hardware/serialhandler/` | Comunicación con Nucleo |
| Gateway | `src/gateway/` | Enrutamiento de mensajes entre procesos |
| State Machine | `src/statemachine/` | Control de estados del sistema |
| Dashboard | `src/dashboard/` | Interfaz web Angular |
| Traffic Comm | `src/data/TrafficCommunication/` | Comunicación con servidor de tráfico |
| Semaphores | `src/data/Semaphores/` | Recepción de estado de semáforos |
| AI Server | `aiserver/` | Servidor FastAPI de inferencia remota con GPU |

### 2.3 Modos de Detección de Carril

| Modo | Descripción | Estado |
|------|-------------|--------|
| `OPENCV` | Pipeline BFMC clásico: Threshold + Canny + Hough | Funcional |
| `LSTR` | Modelo transformer ONNX para formas de carril | Funcional |
| `HYBRID` | Fusión OpenCV + LSTR | Funcional |
| `AI_LOCAL` | YOLO local en Jetson (TensorRT) + control Stanley | Principal (producción) |
| `HYBRIDNETS` | Alias legacy → cliente WebSocket al aiserver | Legacy |
| `SUPERCOMBO` | Alias legacy → cliente WebSocket al aiserver | Legacy |

### 2.4 AI Server (Servidor Remoto de Inferencia)

Servidor FastAPI con WebSocket ubicado en `aiserver/`, pensado para correr en PC con GPU:

| Endpoint | Tipo | Función |
|----------|------|---------|
| `GET /` | HTTP | Estado básico |
| `GET /status` | HTTP | Info detallada del modelo y GPU |
| `WS /ws/inference` | WebSocket | Inferencia completa (carriles + máscaras) |
| `WS /ws/steering` | WebSocket | Solo ángulo de dirección (menor latencia) |
| `WS /ws/signs` | WebSocket | Detección de señales de tráfico |
| `GET /viz/lanes` | MJPEG | Stream visual de carriles detectados |
| `GET /viz/signs` | MJPEG | Stream visual con bounding boxes de señales |
| `GET /viz` | HTML | Página de visualización en vivo |

Motores soportados (configurar con `ENGINE_TYPE` en `aiserver/config.py`):
- `hybridnets` — Segmentación + detección (PyTorch + GPU)
- `supercombo` — Modelo de OpenPilot (ONNX Runtime)
- `yolo_lane_seg` — Segmentación izquierda/derecha (Ultralytics)

---

## 3. Funcionalidades Implementadas

### 3.1 Completadas ✅

- [x] Estructura base del proyecto (plantillas de threads y procesos)
- [x] Comunicación serial con Nucleo (lectura/escritura)
- [x] Captura y streaming de cámara (USB, PiCamera, Jetson CSI)
- [x] Dashboard web con Angular
- [x] Control manual de velocidad y dirección
- [x] Sistema de estados (STOP, MANUAL, AUTO, LEGACY)
- [x] Seguimiento de línea básico con OpenCV (Threshold + Canny + Hough)
- [x] Transformación de perspectiva (Bird's Eye View)
- [x] Algoritmo Sliding Window para detección de carril
- [x] **Controlador Stanley** para dirección (reemplaza PID)
- [x] Detección de modo 1 línea y 2 líneas con blend de transición
- [x] Filtro de ruido (noise filter) y suavizado de historial de steering
- [x] Detección de línea transversal (recovery en curvas)
- [x] Offtracking correction (corrección de eje trasero en curvas)
- [x] Ajuste de parámetros desde el dashboard
- [x] Autostart services para Jetson/Raspberry Pi
- [x] Fallback automático WiFi
- [x] **Servidor de inferencia remota** (aiserver) con FastAPI + WebSocket
- [x] Motor HybridNets (segmentación de carril + detección, GPU)
- [x] Motor YOLO Lane Seg (segmentación izquierda/derecha, Ultralytics)
- [x] Motor Supercombo (OpenPilot ONNX)
- [x] **Detector LSTR** (transformers ONNX, múltiples tamaños de modelo)
- [x] **Motor de percepción local** (LocalPerceptionEngine) — YOLO TensorRT en Jetson
- [x] **Detección de señales** con MobilenetV2 SSD TFLite (SignDetector)
- [x] **Acciones por señales**: stop, no_entry, crosswalk, speed_20, speed_30, red_light, yellow_light, green_light, highway_entrance, highway_exit
- [x] Giro a la izquierda post-STOP (90° calibrado)
- [x] **Conducción en autopista** con velocidades elevadas (highway mode)
- [x] **Detección de zona de paso peatonal** (walk area) con parada automática
- [x] **Maniobra de estacionamiento paralelo** (estado completo en `threadLineFollowing.py`)
  - Secuencia: LANE_KEEPING → SPOT_TRACKED → FORWARD_PAST_SPOT → WAIT_STEER_1 → REVERSING_ENTRY → WAIT_STEER_2 → REVERSING_ALIGN → WAIT_STEER_3 → FORWARD_CORRECTION → PARKED
  - Compensación de curva (pausa odometría mientras gira)
  - Ajuste dinámico de avance cuando se pierde el spot
- [x] Geometría de cámara para estimación de distancia (parámetros intrínsecos IMX219)
- [x] HDR por fusión de exposiciones (Mertens) para PiCamera con detección de glare
- [x] CPU affinity (pinado de procesos a cores disponibles)
- [x] Logging centralizado a cola + MultiWriter
- [x] Tests unitarios (StanleyController, AI local lane/side mapping)
- [x] Scripts de herramientas (`tools/extract_stanley_pdf.swift`)

### 3.2 En Progreso 🔄

- [ ] Calibración fina del giro post-STOP (SIGN_STOP_TURN_DURATION)
- [ ] Calibración de la maniobra de estacionamiento en hardware real
- [ ] Corrección de fórmula de estimación de distancia al parking spot (actualmente usa px/cm horizontal para distancia vertical — ver TODO en config.py)
- [ ] Optimización de parámetros HSV para diferentes condiciones de iluminación
- [ ] Ajuste de STANLEY_K y STANLEY_K_SOFT en curvas cerradas del circuito

### 3.3 Pendiente 📋

- [ ] Integración con servidor de localización (GPS/mapa)
- [ ] Planificación de ruta
- [ ] Manejo de intersecciones
- [ ] Detección de obstáculos y peatones en movimiento (más allá del walk area estático)
- [ ] Integración plena con servidores de tráfico y semáforos (actualmente comentados en `main.py`)
- [ ] Roundabout (rotonda)
- [ ] Preparación final para la competencia (pruebas de pista completa)

---

## 4. Fases del Proyecto

### Fase 1: Infraestructura Base ✅ COMPLETADA
- Setup de Jetson Nano/Orin
- Comunicación serial funcional
- Dashboard básico operativo
- Modos de operación implementados

### Fase 2: Visión y Seguimiento de Línea ✅ COMPLETADA
- Captura y procesamiento con cámara CSI
- Detección de líneas blancas y amarillas
- Transformación de perspectiva (BEV)
- Control automático de dirección (OpenCV → Stanley)
- Ajuste de velocidad en curvas
- Modos LSTR, HYBRID, AI_LOCAL implementados

### Fase 3: Detección de Objetos ✅ COMPLETADA (mayor parte)
- [x] Detección de señales de tráfico (TFLite + YOLO local)
- [x] Reconocimiento de semáforos (vía señales red/yellow/green_light)
- [x] Detección de zona peatonal (walk area)
- [ ] Detección de obstáculos dinámicos

### Fase 4: Navegación Avanzada 🔄 EN PROGRESO
- [x] Conducción en autopista con velocidad elevada
- [x] Maniobra de estacionamiento autónomo
- [ ] Integración con servidor de localización
- [ ] Planificación de ruta
- [ ] Manejo de intersecciones
- [ ] Roundabout

### Fase 5: Integración y Testing 📋 FUTURO
- Pruebas en pista completa
- Optimización de rendimiento
- Manejo de casos borde
- Preparación para la competencia

---

## 5. Parámetros Actuales del Sistema

### 5.1 Controlador Stanley

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| STANLEY_K | 2.5 | Ganancia del error lateral (crosstrack) |
| STANLEY_K_SOFT | 3.0 | Suavizado a baja velocidad |
| STANLEY_K_D_STEER | 0.10 | Amortiguamiento del servo de dirección |
| STEER_HISTORY_LEN | 2 | Frames para promedio de steering |
| NOISE_MAX_STEER_JUMP_DEG | 15° | Salto máximo de steering permitido entre frames |
| NOISE_MAX_REJECT_FRAMES | 3 | Frames consecutivos rechazables |

### 5.2 Velocidades de Seguimiento de Línea

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| LF_BASE_SPEED | 15 | Velocidad inicial al arrancar |
| LF_MAX_SPEED | 15 | Velocidad máxima en modo normal |
| LF_MIN_SPEED | 10 | Velocidad mínima en modo normal (curva) |
| LF_HIGHWAY_MAX_SPEED | 30 | Velocidad máxima en autopista |
| LF_HIGHWAY_MIN_SPEED | 28 | Velocidad mínima en autopista |
| LF_SPEED_RAMP_STEP | 1.0 | Incremento máximo por frame (aceleración) |
| LF_HIGHWAY_SPEED_RAMP_STEP | 3.0 | Aceleración rápida en autopista |

### 5.3 Geometría del Vehículo (TC-04)

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| CAR_WHEELBASE_CM | 26.0 cm | Distancia entre ejes |
| LANE_WIDTH_CM | 35.0 cm | Ancho del carril (borde a borde interior) |
| LINE_WIDTH_CM | 2.0 cm | Ancho de marcas viales |
| LANE_SAFETY_MARGIN_CM | 5.0 cm | Margen mínimo con las líneas de carril |
| OFFTRACK_SCALE | 0.5 | Factor de corrección de offtracking (0–1) |

### 5.4 Cámara (IMX219 en Jetson)

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| CAMERA_TYPE | "jetson" | Tipo de cámara activo |
| JETSON_CAPTURE_RESOLUTION | 1280×720 | Resolución de captura |
| JETSON_OUTPUT_RESOLUTION | 640×480 | Resolución entregada a OpenCV |
| JETSON_FRAMERATE | 60 fps | Framerate de captura |
| CAMERA_HEIGHT_CM | 17.0 cm | Altura del lente al suelo |
| CAMERA_FY_480 | 905.0 px | Focal length vertical en imagen de 480px |
| CAMERA_PITCH_DEG | 16.4° | Inclinación de la cámara hacia abajo |

### 5.5 Maniobra de Estacionamiento

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| PARKING_SEARCH_SPEED | 13 | Velocidad buscando el spot |
| PARKING_FORWARD_SPEED | 13 | Velocidad en fases de avance |
| PARKING_REVERSE_SPEED | -13 | Velocidad en fases de reversa |
| PARKING_ENTRY_STEER | 25.0° | Giro máx derecha (entrada/corrección) |
| PARKING_ALIGN_STEER | -25.0° | Giro máx izquierda (alineación) |
| PARKING_D_FORWARD_CM | 55.0 cm | Avance más allá del spot |
| PARKING_D_REVERSING_ENTRY_CM | 75.0 cm | Reversa de entrada (trasero al spot) |
| PARKING_D_REVERSING_ALIGN_CM | 45.0 cm | Reversa de alineación |
| PARKING_D_FORWARD_CORR_CM | 20.0 cm | Corrección final hacia adelante |
| PARKING_TRIGGER_DISTANCE_CM | 100.0 cm | Distancia para activar SPOT_TRACKED |

### 5.6 Percepción Local AI (YOLO TensorRT en Jetson)

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| LOCAL_AI_MODEL_PATH | `Best_weights_reentrenado_416px.engine` | Modelo TensorRT activo |
| LOCAL_AI_MIN_CONFIDENCE | 0.35 | Confianza mínima de detección |
| LOCAL_AI_IMGSZ | 416 px | Tamaño de entrada al modelo |
| LOCAL_AI_DEVICE | "auto" | Dispositivo de inferencia |
| LOCAL_AI_INTERVAL | 0.04 s | Intervalo mínimo entre inferencias (~25 FPS) |
| LOCAL_AI_NMS_IOU | 0.45 | Umbral NMS para supresión de duplicados |

### 5.7 Detección de Señales

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| SIGN_MIN_CONFIDENCE | 0.50 | Confianza mínima para ejecutar acciones |
| SIGN_MIN_BOX_AREA | 0.010 | Área mínima del bounding box (1% de la imagen) |
| SIGN_ACTION_COOLDOWN | 15.0 s | Cooldown entre acciones de la misma señal |
| SIGN_STOP_DURATION | 3.0 s | Tiempo detenido en señal STOP / no_entry |
| SIGN_CROSSWALK_DURATION | 3.0 s | Tiempo detenido en cruce peatonal |
| SIGN_STOP_TURN_DURATION | 21 s | Duración del giro 90° post-STOP |
| SIGN_HIGHWAY_SPEED | 10 | Velocidad en autopista (post-highway_entrance) |

### 5.8 Detección de Color HSV (modo OpenCV)

**Líneas Blancas:**
- H: 81–180 · S: 0–98 · V: 200–255

**Líneas Amarillas:**
- H: 173–86 · S: 100–255 · V: 100–255

---

## 6. Problemas Conocidos y Soluciones

### 6.1 Seguimiento de Línea

| Problema | Causa | Solución |
|----------|-------|----------|
| Oscilación en recta | STANLEY_K demasiado alto | Reducir a 1.5–2.5 |
| Corte brusco en curvas | Offtracking sobredimensionado | Reducir OFFTRACK_SCALE a 0.3 |
| Líneas no detectadas | HSV descalibrado | Recalibrar para la iluminación |
| Punto rosa incorrecto | Coordenadas sin transformar inversa | Aplicar transformación inversa |
| Detección de línea transversal | Curva cerrada con una sola línea | Lógica de recovery ya implementada |

### 6.2 Estacionamiento

| Problema | Causa | Solución |
|----------|-------|----------|
| Spot detectado siempre a <30cm | Fórmula px/cm horizontal usada para distancia vertical | TODO: reemplazar con geometría de cámara real (CAMERA_HEIGHT_CM, CAMERA_FY_480) |
| Avance mal calibrado saliendo de curva | Odometría cuenta arco en lugar de distancia recta | PARKING_CURVE_WAIT_FOR_STRAIGHT = True |
| Auto no termina de girar en reversa | Tiempo fallback insuficiente | Ajustar PARKING_T_REVERSING_ENTRY |

### 6.3 Sistema General

| Problema | Causa | Solución |
|----------|-------|----------|
| Dashboard no conecta | WebSocket caído | Reiniciar servicio brain-monitor |
| Cámara no responde | Proceso bloqueado | Verificar permisos de cámara |
| Serial timeout | Nucleo desconectado | Verificar conexión USB |
| aiserver lento | CPU sin GPU | Verificar que ENGINE corre en CUDA |

---

## 7. Estructura de Archivos Clave

```
urt-brain-bosch/
├── main.py                              # Punto de entrada principal
├── config.py                            # Configuración global (>500 líneas)
├── requirements.txt                     # Dependencias Python (Raspberry Pi / Jetson)
├── run.sh                               # Script de arranque rápido
├── setup.sh                             # Setup inicial del proyecto
├── src/
│   ├── hardware/
│   │   ├── camera/
│   │   │   ├── processCamera.py         # Proceso de cámara
│   │   │   └── threads/
│   │   │       ├── threadCamera.py      # Captura de frames
│   │   │       ├── threadLineFollowing.py  # Seguimiento de línea + estacionamiento
│   │   │       ├── threadLocalPerception.py # Hilo de percepción YOLO local
│   │   │       ├── threadSignDetection.py   # Hilo de detección de señales TFLite
│   │   │       ├── localPerceptionEngine.py # Motor YOLO unificado (carriles + señales)
│   │   │       ├── lstrDetector.py      # Detector LSTR (Transformers + ONNX)
│   │   │       ├── signDetector.py      # Detector TFLite MobilenetV2 SSD
│   │   │       └── signActions.py       # Ejecutor de acciones por señales
│   │   └── serialhandler/
│   │       ├── processSerialHandler.py
│   │       └── threads/
│   │           ├── threadRead.py        # Lectura serial (Nucleo→Pi)
│   │           └── threadWrite.py       # Escritura serial (Pi→Nucleo)
│   ├── statemachine/
│   │   ├── stateMachine.py              # Máquina de estados
│   │   ├── systemMode.py               # Definiciones de modos
│   │   └── transitionTable.py          # Tabla de transiciones
│   ├── gateway/
│   │   └── processGateway.py           # Enrutador de mensajes
│   ├── dashboard/
│   │   ├── processDashboard.py
│   │   └── components/
│   │       ├── calibration.py
│   │       └── ip_manger.py
│   ├── data/
│   │   ├── Semaphores/                  # Recepción de semáforos UDP
│   │   └── TrafficCommunication/        # Cliente TCP/UDP servidor de tráfico
│   ├── templates/                       # Plantillas base (ThreadWithStop, WorkerProcess)
│   └── utils/
│       ├── messages/                    # Mensajes del sistema (allMessages.py)
│       └── outputWriters.py
├── aiserver/                            # Servidor de inferencia remota (GPU)
│   ├── server.py                        # FastAPI + WebSocket endpoints
│   ├── client.py                        # Cliente WebSocket para la Jetson
│   ├── config.py                        # Configuración del servidor
│   ├── inference.py                     # Motor HybridNets
│   ├── sign_detection_engine.py         # Motor YOLOv8 de señales
│   ├── yolo_lane_seg_engine.py          # Motor YOLO Lane Seg
│   ├── supercombo_engine.py             # Motor Supercombo (OpenPilot)
│   ├── models/                          # Modelos del aiserver
│   │   ├── lane_segmentation/
│   │   └── sign_detection/
│   └── HybridNets/                      # Submodule HybridNets
├── models/                              # Modelos para Jetson (local)
│   ├── lane_segmentation/
│   │   ├── best.pt / best.onnx          # YOLO PyTorch / ONNX
│   │   ├── Best416px.engine             # TensorRT 416px
│   │   ├── Best_weights_reentrenado_416px.engine  # TensorRT reentrenado (ACTIVO)
│   │   └── build_trt.py                 # Script de generación TensorRT
│   ├── lstr/                            # Modelos ONNX para LSTR
│   └── sign_detection/
│       ├── detect.tflite                # MobilenetV2 SSD TFLite
│       ├── labelmap.txt
│       ├── bfmc_detect_320_best.pt      # YOLOv8 BFMC 320px
│       └── bfmc_detect_640_best.pt      # YOLOv8 BFMC 640px
├── services/                            # Servicios systemd
│   ├── brain-autostart
│   └── angular-autostart
├── tests/
│   ├── test_stanley_controller.py       # Tests unitarios del controlador Stanley
│   ├── test_stanley_physical_units.py
│   └── test_ai_local_lane_side_mapping.py
├── calibration/                         # Herramientas de calibración de cámara
├── scripts/                             # Scripts auxiliares
├── tools/
│   └── extract_stanley_pdf.swift        # Extracción de datos del paper Stanley
├── temp/                                # Archivos de logs en tiempo real
│   ├── lane_calib_log.txt
│   ├── line_following_auto_last_run.txt
│   └── serial_history.log
└── monitoring/
    ├── project-plan.md                  # Este archivo
    └── 01_full_architecture.pdf
```

---

## 8. Comandos Útiles

### Arrancar el sistema
```bash
cd /home/pi/Documents/urt-brain-bosch
python3 main.py
# o con el script de arranque:
./run.sh
```

### Arrancar el aiserver (en PC con GPU)
```bash
cd aiserver
source venv/bin/activate
python server.py
# o: uvicorn server:app --host 0.0.0.0 --port 8500
```

### Generar engine TensorRT en Jetson
```bash
cd models/lane_segmentation
python build_trt.py
```

### Ver logs
```bash
journalctl -u brain-monitor -f
journalctl -u angular-dashboard -f
```

### Reiniciar servicios
```bash
sudo systemctl restart brain-monitor
sudo systemctl restart angular-dashboard
```

### Ejecutar tests
```bash
python -m pytest tests/
```

---

## 9. Contacto y Recursos

- **Documentación BFMC:** https://bosch-future-mobility-challenge-documentation.readthedocs-hosted.com/
- **Repositorio:** /home/pi/Documents/urt-brain-bosch
- **Reports de avance:** `Project status/`
  - Project Status 1 (PDF + video)
  - Project Status 2 (PDF + video)
  - Qualifications (PDF + video comprimido)
- **Informe técnico:** `INFORME_TECNICO.md`
- **README seguimiento de línea:** `LINE_FOLLOWING_README.md`

---

## 10. Historial de Cambios

| Fecha | Cambio | Autor |
|-------|--------|--------|
| 2026-03-09 | Actualización completa del plan: aiserver, percepción local YOLO, estacionamiento, autopista, Stanley, señales | - |
| 2026-03-09 | Ajuste SIGN_STOP_TURN_DURATION a 21s; mejora de timing en SignActions | - |
| 2026-03-09 | Detección de zona peatonal (walk area) con parada automática | - |
| 2026-03-06 | Detección de línea transversal y recovery en threadLineFollowing | - |
| 2026-03-05 | Lógica de ajuste dinámico de avance al perder el parking spot | - |
| 2026-03-04 | Soporte de threading en SignActions (acciones bloqueantes en daemon thread) | - |
| 2026-03-02 | Conducción en autopista (HIGHWAY_MAX_SPEED=30) y ajuste de rampa | - |
| 2026-02-27 | Motor de percepción local (LocalPerceptionEngine) con TensorRT | - |
| 2026-02-25 | Maniobra de estacionamiento paralelo autónomo (secuencia completa) | - |
| 2026-02-13 | Integración LSTR Detector (ONNX Transformers) y modo HYBRID | - |
| 2026-02-03 | Controlador Stanley (reemplaza PID) con offtracking correction | - |
| 2026-02-02 | Calculado ángulo de dirección; limit steering a 25°; plan creado | - |

---

*Última actualización: 9 de marzo de 2026*
