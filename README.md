# URT Brain — BFMC 2026

Sistema de conducción autónoma para auto a escala 1:10 de la **Bosch Future Mobility Challenge (BFMC)**. Corre sobre Raspberry Pi 5, con offload de inferencia pesada a un servidor GPU remoto vía WebSocket.

## Arquitectura

```
Raspberry Pi 5                          PC / Laptop (GPU)
┌──────────────────────────┐            ┌─────────────────────────┐
│  main.py                 │            │  aiserver/server.py     │
│  ├─ processCamera        │  WebSocket │  ├─ /ws/steering        │
│  │  ├─ threadCamera      │◄──────────►│  │  (HybridNets o       │
│  │  ├─ threadLineFollow  │  JPEG→JSON │  │   Supercombo)        │
│  │  └─ threadSignDetect  │◄──────────►│  ├─ /ws/signs           │
│  ├─ processSerialHandler │            │  │  (YOLOv8)            │
│  │  ├─ threadRead        │            │  └─ /viz (MJPEG debug)  │
│  │  └─ threadWrite       │            └─────────────────────────┘
│  ├─ processDashboard     │
│  │  └─ Angular frontend  │◄── Browser (control manual + telemetría)
│  └─ processGateway       │
│          │                │
│          │ UART           │
│  ┌───────▼──────────┐    │
│  │  Nucleo STM32    │    │
│  │  (motor + servo) │    │
│  └──────────────────┘    │
└──────────────────────────┘
```

**Comunicación interna**: Colas `multiprocessing.Queue` clasificadas por prioridad (`Critical`, `Warning`, `General`, `Config`, `Log`). Patrón pub/sub con `messageHandlerSender` / `messageHandlerSubscriber`.

**Máquina de estados**: `DEFAULT` → `AUTO` → `MANUAL` → `STOP`. En modo **AUTO** se activan line following y sign detection.

## Setup — Raspberry Pi

```bash
chmod +x setup.sh
./setup.sh
```

Esto instala las dependencias del sistema (Node.js 20, Angular CLI, OpenCV, PiCamera2, pyserial, etc.) y las de Python:

```bash
sudo pip3 install -r requirements.txt
```

### Dependencias opcionales (RPi)

```bash
# LSTR AI lane detection (modelo ONNX local)
pip install onnxruntime

# Detección de señales local (TFLite)
pip install ai-edge-litert

# Cliente WebSocket para AI Server remoto
pip install websockets
```

## Setup — AI Server (PC con GPU)

El AI Server corre en una máquina separada con GPU (NVIDIA, Apple Silicon, o CPU potente).

```bash
cd aiserver

# Instalar dependencias
pip install -r requirements.txt

# Instalar PyTorch según tu GPU:
# CUDA 12.1:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Apple MPS:  pip install torch torchvision
# CPU only:   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Descargar modelo Supercombo (openpilot, ~47MB)
python setup_supercombo.py

# Descargar modelo de señales (YOLOv8, ~22MB)
# Colocar trafic.pt en aiserver/models/sign_detection/
```

Configurar en `aiserver/config.py`:
```python
ENGINE_TYPE = "supercombo"   # "hybridnets" | "supercombo"
DEVICE = "cuda"              # "cuda" | "mps" | "cpu"
SIGN_DETECTION_ENABLED = True
```

Iniciar:
```bash
python server.py
# Servidor escucha en 0.0.0.0:8500
```

## Uso

```bash
# Terminal 1: Iniciar el brain
python main.py

# Terminal 2: Dashboard Angular
cd src/dashboard/frontend
npm start
```

El dashboard web permite:
- Cambiar entre modos (Manual / Auto / Stop)
- Control manual de velocidad y steering
- Visualización del stream de cámara
- Configuración en vivo de parámetros de line following

## Configuración

Toda la configuración del brain está en `config.py`:

| Parámetro | Descripción |
|---|---|
| `CAMERA_TYPE` | `"jetson"` (CSI Jetson), `"picamera"` (CSI RPi) o `"usb"` |
| `JETSON_SENSOR_ID` | Sensor CSI en Jetson (`0` CAM0, `1` CAM1) |
| `JETSON_CAPTURE_RESOLUTION` | Resolución nativa del sensor Jetson (ej. `(1920, 1080)`) |
| `JETSON_OUTPUT_RESOLUTION` | Resolución final enviada a OpenCV/dashboard (ej. `(960, 720)`) |
| `JETSON_FRAMERATE` | FPS objetivo para `nvarguscamerasrc` |
| `JETSON_FLIP_METHOD` | `flip-method` de `nvvidconv` (0 = sin flip) |
| `SHOW_CAMERA_PREVIEW` | Master switch para ventanas de debug OpenCV |
| `DEBUG_WINDOWS` | Dict para habilitar ventanas individuales |
| `ENABLE_SIGN_DETECTION` | Activar detección de señales via AI Server |
| `SIGN_SERVER_URL` | URL WebSocket del AI Server (`ws://ip:8500/ws/signs`) |
| `SIGN_DETECTION_ACTIONS` | Ejecutar acciones al detectar señales (stop, frenar, etc.) |
| `SIGN_MIN_CONFIDENCE` | Umbral de confianza mínimo (0.0–1.0) |
| `SIGN_MIN_BOX_AREA` | Área mínima del bounding box para ejecutar acciones |
| `SIGN_ACTION_COOLDOWN` | Segundos de cooldown entre acciones de la misma señal |

## Seguimiento de Líneas

El módulo de line following soporta **5 modos de detección** intercambiables desde el dashboard:

### OpenCV (procesamiento clásico)

Pipeline: CLAHE → HSV filtering → umbral binario → Canny edges → Hough Lines → Sliding Window → Polynomial fit.

- **Iluminación adaptativa**: CLAHE ecualiza el histograma localmente, detección adaptativa de blanco ajusta el umbral V dinámicamente según el percentil 92 del frame actual
- **Fallback por gradiente**: Si la detección por color falla (< 1% de píxeles), recurre a Sobel/Canny como respaldo
- **Filtro de ruido**: Rechaza frames con más de 40 líneas Hough (reflejos), saltos de error > 80px, o saltos de steering > 15° entre frames

### LSTR (Transformer local)

Modelo LSTR (WACV 2021) ejecutado con ONNX Runtime directamente en la RPi. Predice parámetros de forma de carril (no segmentación pixel a pixel). Más robusto a cambios de iluminación que OpenCV.

### Hybrid (OpenCV + LSTR)

Fusión con pesos configurables (40% OpenCV, 60% LSTR). Bonus de confianza ×1.2 cuando ambos métodos coinciden en dirección.

### HybridNets (GPU remoto)

Red multi-tarea (EfficientNet + BiFPN): segmentación de ruta + detección de carriles + detección de objetos. Corre en el AI Server con GPU. Comunicación por WebSocket con JPEG crudo.

### Supercombo (openpilot, GPU remoto)

Modelo recurrente de comma.ai. Procesa 2 frames YUV con estado GRU persistente entre frames. Predice 4 carriles × 33 puntos 3D y 5 trayectorias planeadas.

## Control PID

```
steering = Kp·error + Ki·∫error·dt + Kd·d(error)/dt
```

- **Kp=25.0**: Respuesta proporcional inmediata al error
- **Ki=1.0**: Corrige offsets persistentes acumulando error en el tiempo
- **Kd=4.0**: Amortigua oscilaciones basándose en la tasa de cambio
- **Zona muerta**: Errores < 50px ignorados para estabilidad en recta
- **Anti-windup**: Reset del integral cada 10 iteraciones
- **Feed-forward**: Componente predictivo basado en curvatura estimada y modelo de Ackermann (`δ = arctan(L/R)`, L=26.5cm wheelbase)

### Velocidad adaptativa

| Steering | Velocidad |
|---|---|
| < 10° | max_speed (10) |
| 10°–15° | interpolación lineal |
| > 15° | min_speed (5) |
| Highway mode | 10–25 |

Rampa de aceleración: máximo +0.5 unidades por frame.

## Máquina de Estados de Curvas

```
STRAIGHT ──(1 línea por ≥1 frame)──► ENTERING
    ▲                                    │
    │                              (≥2 frames 1 línea)
    │                                    ▼
EXITING ◄──(2 líneas por ≥3 frames)── IN_CURVE
```

Usa radios conocidos de la pista BFMC (66.5cm carril interior, 103.5cm exterior) para pre-posicionar el auto antes de la curva.

**Recuperación de curva**: Si el auto queda saturado en máximo steering por >8 frames, ejecuta una maniobra de reversa automática: frena → gira ruedas → retrocede → reposiciona → resume.

## Detección de Señales de Tráfico

Arquitectura dual:

| Componente | Modelo | Dónde corre | Protocolo |
|---|---|---|---|
| `signDetector.py` | MobilenetV2 SSD (TFLite) | RPi local | Directo |
| `sign_detection_engine.py` | YOLOv8 (`trafic.pt`) | AI Server | WebSocket |

### Señales soportadas y acciones

| Señal | Acción |
|---|---|
| Stop / No Entry / Red Light | Frena 3 segundos, luego resume |
| Crosswalk | Reduce velocidad por 3 segundos |
| Yellow Light | Reduce velocidad |
| Green Light | Resume velocidad normal |
| Speed 20 / Speed 30 | Cambia velocidad base |
| Highway Entrance | Sube velocidad, activa highway mode |
| Highway Exit | Baja velocidad, desactiva highway mode |
| Parking | Detiene el auto |

### Filtros de seguridad

- **Cooldown por grupo**: 15s entre acciones del mismo tipo (evita frenar 3 veces por el mismo stop)
- **Área mínima de box**: Si la señal ocupa < 1% del frame (está lejos), solo se detecta pero no se ejecuta acción
- **Solo en modo AUTO**: Las acciones vehiculares solo se ejecutan en modo autónomo
- **Coordinación con line following**: Un `threading.Event` compartido bloquea los comandos de motor del line following mientras se ejecuta una acción de señal

## Protocolo Serial (RPi ↔ Nucleo STM32)

| Mensaje | Formato | Ejemplo |
|---|---|---|
| `SpeedMotor` | `str(speed * 10)` | Velocidad 5.0 → `"50"` |
| `SteerMotor` | `str(angle)` | 15 grados → `"15"` |

## Estructura del Proyecto

```
.
├── main.py                          # Entry point — orquesta todos los procesos
├── config.py                        # Configuración global del brain
├── setup.sh                         # Script de instalación (RPi)
├── requirements.txt                 # Dependencias Python (RPi)
├── newComponent.py                  # Generador de nuevos módulos
│
├── src/
│   ├── hardware/
│   │   ├── camera/
│   │   │   ├── processCamera.py     # Proceso de cámara
│   │   │   └── threads/
│   │   │       ├── threadCamera.py          # Captura de frames
│   │   │       ├── threadLineFollowing.py   # Detección de carriles + PID
│   │   │       ├── threadSignDetection.py   # Detección de señales (WebSocket)
│   │   │       ├── signDetector.py          # MobilenetV2 SSD TFLite (local)
│   │   │       └── lstrDetector.py          # LSTR Transformer (ONNX local)
│   │   └── serialhandler/           # Comunicación UART con Nucleo
│   ├── statemachine/
│   │   ├── stateMachine.py          # Lógica de transiciones
│   │   ├── systemMode.py           # Definición de modos (AUTO, MANUAL, etc.)
│   │   └── transitionTable.py
│   ├── dashboard/                   # Angular frontend + WebSocket backend
│   ├── gateway/                     # Router de mensajes internos
│   ├── data/
│   │   ├── Semaphores/              # Procesamiento de semáforos (UDP)
│   │   └── TrafficCommunication/    # Comunicación con servidor de tráfico
│   ├── templates/
│   │   ├── workerprocess.py         # Base class para procesos
│   │   └── threadwithstop.py        # Base class para threads
│   └── utils/
│       └── messages/                # Sistema de mensajería pub/sub
│
├── aiserver/                        # AI Server (corre en PC con GPU)
│   ├── server.py                    # FastAPI + WebSocket endpoints
│   ├── config.py                    # Configuración del servidor
│   ├── inference.py                 # HybridNets engine (PyTorch)
│   ├── supercombo_engine.py         # Supercombo engine (ONNX, openpilot)
│   ├── sign_detection_engine.py     # YOLOv8 engine para señales
│   ├── client.py                    # WebSocket client (usado por RPi)
│   ├── setup_supercombo.py          # Descarga modelo Supercombo (~47MB)
│   ├── requirements.txt             # Dependencias del servidor
│   └── HybridNets/                  # Repo HybridNets (modelo + utilidades)
│
├── services/
│   ├── brain-autostart/             # Servicio systemd para iniciar brain al boot
│   ├── angular-autostart/           # Servicio systemd para iniciar dashboard
│   └── rpi-wifi-fallback/           # Fallback WiFi automático
│
├── models/                          # Modelos de ML (no versionados)
│   ├── lstr/                        # Modelos LSTR ONNX
│   └── sign_detection/              # Modelo TFLite de señales
│
└── calibration/                     # Templates para calibración de motores
```

## Servicios Systemd

| Servicio | Descripción |
|---|---|
| `brain-autostart` | Inicia `main.py` al bootear la RPi |
| `angular-autostart` | Inicia el dashboard Angular |
| `rpi-wifi-fallback` | Si la WiFi principal no está disponible, conecta a una red de respaldo |

Instalar:
```bash
cd services/brain-autostart && sudo ./install.sh
cd services/angular-autostart && sudo ./install.sh
cd services/rpi-wifi-fallback && sudo ./install.sh
```

## Dimensiones del Auto

| Medida | Valor |
|---|---|
| Largo total | 36.5 cm |
| Ancho total | 19.0 cm |
| Distancia entre ejes | 27.5 cm |
| Ancho de carril BFMC | 35.0 cm |
| Radio curva interior | 66.5 cm (al centro del carril) |
| Radio curva exterior | 103.5 cm (al centro del carril) |

## Licencia

BSD 3-Clause. Basado en el [BFMC Starter Project](https://github.com/ECC-BFMC) de Bosch Engineering Center Cluj.
