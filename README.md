# URT Brain — BFMC 2026

Autonomous driving system for a 1:10 scale car in the **Bosch Future Mobility Challenge (BFMC)**. Runs on Raspberry Pi 5, with heavy inference offloaded to a remote GPU server via WebSocket.

## Architecture

```text
Raspberry Pi 5                          PC / Laptop (GPU)
┌──────────────────────────┐            ┌─────────────────────────┐
│  main.py                 │            │  aiserver/server.py     │
│  ├─ processCamera        │  WebSocket │  ├─ /ws/steering        │
│  │  ├─ threadCamera      │◄──────────►│  │  (HybridNets or      │
│  │  ├─ threadLineFollow  │  JPEG→JSON │  │   Supercombo)        │
│  │  └─ threadSignDetect  │◄──────────►│  ├─ /ws/signs           │
│  ├─ processSerialHandler │            │  │  (YOLOv8)            │
│  │  ├─ threadRead        │            │  └─ /viz (MJPEG debug)  │
│  │  └─ threadWrite       │            └─────────────────────────┘
│  ├─ processDashboard     │
│  │  └─ Angular frontend  │◄── Browser (manual control + telemetry)
│  └─ processGateway       │
│          │                │
│          │ UART           │
│  ┌───────▼──────────┐    │
│  │  Nucleo STM32    │    │
│  │  (motor + servo) │    │
│  └──────────────────┘    │
└──────────────────────────┘
```

**Internal communication**: `multiprocessing.Queue` queues classified by priority (`Critical`, `Warning`, `General`, `Config`, `Log`). Pub/sub pattern with `messageHandlerSender` / `messageHandlerSubscriber`.

**State machine**: `DEFAULT` → `AUTO` → `MANUAL` → `STOP`. In **AUTO** mode, line following and sign detection are enabled.

## Setup — Raspberry Pi

```bash
chmod +x setup.sh
./setup.sh
```

This installs the system dependencies (Node.js 20, Angular CLI, OpenCV, PiCamera2, pyserial, etc.) and the Python dependencies:

```bash
sudo pip3 install -r requirements.txt
```

### Optional dependencies (RPi)

```bash
# LSTR AI lane detection (local ONNX model)
pip install onnxruntime

# Local sign detection (TFLite)
pip install ai-edge-litert

# WebSocket client for remote AI Server
pip install websockets
```

## Setup — AI Server (PC with GPU)

The AI Server runs on a separate machine with a GPU (NVIDIA, Apple Silicon, or a powerful CPU).

```bash
cd aiserver

# Install dependencies
pip install -r requirements.txt

# Install PyTorch according to your GPU:
# CUDA 12.1:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Apple MPS:  pip install torch torchvision
# CPU only:   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Download Supercombo model (openpilot, ~47MB)
python setup_supercombo.py

# Download sign detection model (YOLOv8, ~22MB)
# Place trafic.pt in aiserver/models/sign_detection/
```

Configure in `aiserver/config.py`:

```python
ENGINE_TYPE = "supercombo"   # "hybridnets" | "supercombo"
DEVICE = "cuda"              # "cuda" | "mps" | "cpu"
SIGN_DETECTION_ENABLED = True
```

Start:

```bash
python server.py
# Server listens on 0.0.0.0:8500
```

## Usage

The recommended entry point is `run.sh`, which auto-selects mode by platform:

```bash
# Mac (development): launches brain in sim mode + PyQt5 GUI on localhost
./run.sh

# Mac (monitor): GUI only, connects to a remote car
./run.sh --monitor 192.168.1.99

# Jetson (production): brain only, headless
./run.sh
```

The PyQt5 GUI (`src/dashboard/gui/`) provides:

* Map view with the car's live position, GraphML nodes and active route
* Map editor (GraphML + SVG + semantics) — single source of truth, JSON-persisted
* Camera stream with lane / sign / mask overlays
* Manual control (joystick + keyboard wasd / spacebar)
* Switching between modes (Manual / Legacy / Auto / Stop / Parking)
* PWM calibration wizard
* Telemetry (battery, CPU, memory, temperature, speed/steer chart)
* Live console logs
* Persistent JSON configuration (`config/*.json`) versioned with the repo

### Legacy Angular dashboard

The original web dashboard has been moved to `legacy/dashboard-frontend/`. It still
works against the same SocketIO backend on `:5005`:

```bash
cd legacy/dashboard-frontend
npm start
```

See [`legacy/README.md`](legacy/README.md) for migration status.

## Configuration

All brain configuration is in `config.py`:

| Parameter                   | Description                                                   |
| --------------------------- | ------------------------------------------------------------- |
| `CAMERA_TYPE`               | `"jetson"` (Jetson CSI), `"picamera"` (RPi CSI), or `"usb"`   |
| `JETSON_SENSOR_ID`          | CSI sensor on Jetson (`0` CAM0, `1` CAM1)                     |
| `JETSON_CAPTURE_RESOLUTION` | Native resolution of the Jetson sensor (e.g. `(1920, 1080)`)  |
| `JETSON_OUTPUT_RESOLUTION`  | Final resolution sent to OpenCV/dashboard (e.g. `(960, 720)`) |
| `JETSON_FRAMERATE`          | Target FPS for `nvarguscamerasrc`                             |
| `JETSON_FLIP_METHOD`        | `flip-method` for `nvvidconv` (0 = no flip)                   |
| `SHOW_CAMERA_PREVIEW`       | Master switch for OpenCV debug windows                        |
| `DEBUG_WINDOWS`             | Dict to enable individual windows                             |
| `ENABLE_SIGN_DETECTION`     | Enable sign detection via AI Server                           |
| `SIGN_SERVER_URL`           | AI Server WebSocket URL (`ws://ip:8500/ws/signs`)             |
| `SIGN_DETECTION_ACTIONS`    | Execute actions when signs are detected (stop, brake, etc.)   |
| `SIGN_MIN_CONFIDENCE`       | Minimum confidence threshold (0.0–1.0)                        |
| `SIGN_MIN_BOX_AREA`         | Minimum bounding box area to execute actions                  |
| `SIGN_ACTION_COOLDOWN`      | Cooldown in seconds between actions for the same sign         |

## Line Following

The line-following module supports **5 interchangeable detection modes**, switchable from the dashboard:

### OpenCV (classical processing)

Pipeline: CLAHE → HSV filtering → binary threshold → Canny edges → Hough Lines → Sliding Window → Polynomial fit.

* **Adaptive lighting**: CLAHE locally equalizes the histogram; adaptive white detection dynamically adjusts the V threshold based on the 92nd percentile of the current frame
* **Gradient fallback**: If color-based detection fails (< 1% of pixels), it falls back to Sobel/Canny as backup
* **Noise filtering**: Rejects frames with more than 40 Hough lines (reflections), error jumps > 80px, or steering jumps > 15° between frames

### LSTR (local Transformer)

LSTR model (WACV 2021) executed with ONNX Runtime directly on the RPi. Predicts lane shape parameters instead of pixel-by-pixel segmentation. More robust to lighting changes than OpenCV.

### Hybrid (OpenCV + LSTR)

Fusion with configurable weights (40% OpenCV, 60% LSTR). Confidence bonus ×1.2 when both methods agree on direction.

### HybridNets (remote GPU)

Multi-task network (EfficientNet + BiFPN): drivable area segmentation + lane detection + object detection. Runs on the AI Server with GPU. Communication over WebSocket with raw JPEG frames.

### Supercombo (openpilot, remote GPU)

Recurrent comma.ai model. Processes 2 YUV frames with persistent GRU state across frames. Predicts 4 lanes × 33 3D points and 5 planned trajectories.

## PID Control

```text
steering = Kp·error + Ki·∫error·dt + Kd·d(error)/dt
```

* **Kp=25.0**: Immediate proportional response to error
* **Ki=1.0**: Corrects persistent offsets by accumulating error over time
* **Kd=4.0**: Damps oscillations based on the rate of change
* **Dead zone**: Errors < 50px are ignored for straight-line stability
* **Anti-windup**: Integral is reset every 10 iterations
* **Feed-forward**: Predictive component based on estimated curvature and the Ackermann model (`δ = arctan(L/R)`, L=26.5cm wheelbase)

### Adaptive speed

| Steering     | Speed                |
| ------------ | -------------------- |
| < 10°        | max_speed (10)       |
| 10°–15°      | linear interpolation |
| > 15°        | min_speed (5)        |
| Highway mode | 10–25                |

Acceleration ramp: maximum +0.5 units per frame.

## Curve State Machine

```text
STRAIGHT ──(1 line for ≥1 frame)──► ENTERING
    ▲                                  │
    │                            (≥2 frames with 1 line)
    │                                  ▼
EXITING ◄──(2 lines for ≥3 frames)── IN_CURVE
```

Uses known BFMC track radii (66.5cm inner lane, 103.5cm outer lane) to pre-position the car before entering a curve.

**Curve recovery**: If the car remains saturated at maximum steering for >8 frames, it executes an automatic reverse maneuver: brake → turn wheels → reverse → reposition → resume.

## Traffic Sign Detection

Dual architecture:

| Component                  | Model                    | Where it runs | Protocol  |
| -------------------------- | ------------------------ | ------------- | --------- |
| `signDetector.py`          | MobilenetV2 SSD (TFLite) | Local RPi     | Direct    |
| `sign_detection_engine.py` | YOLOv8 (`trafic.pt`)     | AI Server     | WebSocket |

### Supported signs and actions

| Sign                        | Action                               |
| --------------------------- | ------------------------------------ |
| Stop / No Entry / Red Light | Brake for 3 seconds, then resume     |
| Crosswalk                   | Reduce speed for 3 seconds           |
| Yellow Light                | Reduce speed                         |
| Green Light                 | Resume normal speed                  |
| Speed 20 / Speed 30         | Change base speed                    |
| Highway Entrance            | Increase speed, enable highway mode  |
| Highway Exit                | Decrease speed, disable highway mode |
| Parking                     | Stop the car                         |

### Safety filters

* **Per-group cooldown**: 15s between actions of the same type (prevents braking 3 times for the same stop sign)
* **Minimum box area**: If the sign occupies < 1% of the frame (too far away), it is only detected, but no action is executed
* **AUTO mode only**: Vehicle actions are only executed in autonomous mode
* **Coordination with line following**: A shared `threading.Event` blocks motor commands from line following while a sign action is being executed

## Serial Protocol (RPi ↔ Nucleo STM32)

| Message      | Format            | Example             |
| ------------ | ----------------- | ------------------- |
| `SpeedMotor` | `str(speed * 10)` | Speed 5.0 → `"50"`  |
| `SteerMotor` | `str(angle)`      | 15 degrees → `"15"` |

## Project Structure

```text
.
├── main.py                          # Entry point — orchestrates all processes
├── config.py                        # Global brain configuration
├── setup.sh                         # Installation script (RPi)
├── requirements.txt                 # Python dependencies (RPi)
├── newComponent.py                  # New module generator
│
├── src/
│   ├── hardware/
│   │   ├── camera/
│   │   │   ├── processCamera.py     # Camera process
│   │   │   └── threads/
│   │   │       ├── threadCamera.py          # Frame capture
│   │   │       ├── threadLineFollowing.py   # Lane detection + PID
│   │   │       ├── threadSignDetection.py   # Sign detection (WebSocket)
│   │   │       ├── signDetector.py          # MobilenetV2 SSD TFLite (local)
│   │   │       └── lstrDetector.py          # LSTR Transformer (local ONNX)
│   │   └── serialhandler/           # UART communication with Nucleo
│   ├── statemachine/
│   │   ├── stateMachine.py          # Transition logic
│   │   ├── systemMode.py            # Mode definitions (AUTO, MANUAL, etc.)
│   │   └── transitionTable.py
│   ├── dashboard/                   # Angular frontend + WebSocket backend
│   ├── gateway/                     # Internal message router
│   ├── data/
│   │   ├── Semaphores/              # Traffic light processing (UDP)
│   │   └── TrafficCommunication/    # Communication with traffic server
│   ├── templates/
│   │   ├── workerprocess.py         # Base class for processes
│   │   └── threadwithstop.py        # Base class for threads
│   └── utils/
│       └── messages/                # Pub/sub messaging system
│
├── aiserver/                        # AI Server (runs on PC with GPU)
│   ├── server.py                    # FastAPI + WebSocket endpoints
│   ├── config.py                    # Server configuration
│   ├── inference.py                 # HybridNets engine (PyTorch)
│   ├── supercombo_engine.py         # Supercombo engine (ONNX, openpilot)
│   ├── sign_detection_engine.py     # YOLOv8 engine for signs
│   ├── client.py                    # WebSocket client (used by RPi)
│   ├── setup_supercombo.py          # Downloads Supercombo model (~47MB)
│   ├── requirements.txt             # Server dependencies
│   └── HybridNets/                  # HybridNets repo (model + utilities)
│
├── services/
│   ├── brain-autostart/             # systemd service to start brain on boot
│   ├── angular-autostart/           # systemd service to start dashboard
│   └── rpi-wifi-fallback/           # Automatic WiFi fallback
│
├── models/                          # ML models (not versioned)
│   ├── lstr/                        # LSTR ONNX models
│   └── sign_detection/              # TFLite sign model
│
└── calibration/                     # Templates for motor calibration
```

## Systemd Services

| Service             | Description                                                      |
| ------------------- | ---------------------------------------------------------------- |
| `brain-autostart`   | Starts `main.py` when the RPi boots                              |
| `angular-autostart` | Starts the Angular dashboard                                     |
| `rpi-wifi-fallback` | If the primary WiFi is unavailable, connects to a backup network |

Install:

```bash
cd services/brain-autostart && sudo ./install.sh
cd services/angular-autostart && sudo ./install.sh
cd services/rpi-wifi-fallback && sudo ./install.sh
```

## Vehicle Dimensions

| Measurement        | Value                     |
| ------------------ | ------------------------- |
| Total length       | 36.5 cm                   |
| Total width        | 19.0 cm                   |
| Wheelbase          | 27.5 cm                   |
| BFMC lane width    | 35.0 cm                   |
| Inner curve radius | 66.5 cm (to lane center)  |
| Outer curve radius | 103.5 cm (to lane center) |

## License

BSD 3-Clause. Based on the [BFMC Starter Project](https://github.com/ECC-BFMC) by Bosch Engineering Center Cluj.

If you want, I can also turn this into a cleaner README-style English version with more natural technical wording while preserving the exact structure.
