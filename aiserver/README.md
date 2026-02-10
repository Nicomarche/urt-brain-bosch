# HybridNets AI Server

Servidor de inferencia remota para detección de carriles con **HybridNets**.

La Raspberry Pi captura frames de la cámara, los envía por WebSocket al servidor potente,
y el servidor responde con los resultados de segmentación de carril + ángulo de dirección.

## Arquitectura

```
┌──────────────┐     WebSocket (JPEG)      ┌──────────────────┐
│  Raspberry Pi │ ────────────────────────> │  Servidor (GPU)  │
│               │                           │                  │
│  - Cámara     │ <──────────────────────── │  - HybridNets    │
│  - Motores    │     JSON (steering)       │  - PyTorch       │
│  - PID Control│                           │  - FastAPI       │
└──────────────┘                            └──────────────────┘
```

## Flujo de datos

1. **RPi** captura frame de la cámara (512x270 o configurable)
2. **RPi** comprime a JPEG (calidad 70) y envía por WebSocket
3. **Server** decodifica, redimensiona a 640x384, ejecuta HybridNets
4. **Server** extrae líneas de carril, calcula ángulo de dirección
5. **Server** responde con JSON: `{steering, confidence, lane_points, ...}`
6. **RPi** aplica el ángulo a los motores con PID control

## Setup del Servidor

### 1. Requisitos
- Python 3.8+
- GPU con CUDA (recomendado) o CPU potente
- 4GB+ RAM

### 2. Instalación rápida

```bash
cd aiserver
python setup_server.py
```

Esto automáticamente:
- Instala dependencias de Python
- Clona el repositorio de HybridNets
- Descarga los pesos pre-entrenados (~200MB)

### 3. Instalación manual

```bash
# Instalar PyTorch (elegir según tu GPU)
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Instalar dependencias del servidor
pip install -r requirements.txt

# Clonar HybridNets
git clone https://github.com/datvuthanh/HybridNets.git

# Descargar pesos
mkdir -p weights
curl -L -o weights/hybridnets.pth https://github.com/datvuthanh/HybridNets/releases/download/v1.0/hybridnets.pth
```

### 4. Ejecutar el servidor

```bash
python server.py
```

El servidor escucha en `0.0.0.0:8500` por defecto.

## Setup del Cliente (Raspberry Pi)

### 1. Instalar dependencias mínimas

```bash
pip install -r requirements_client.txt
```

### 2. Test rápido

```bash
# Con una imagen
python client.py --server ws://IP_DEL_SERVER:8500/ws/steering --test-image foto.jpg

# Con cámara USB
python client.py --server ws://IP_DEL_SERVER:8500/ws/steering --camera 0
```

### 3. Integración con el sistema existente

El cliente se integra automáticamente con `threadLineFollowing.py`.
En el dashboard, selecciona el modo de detección `hybridnets` y configura
la IP del servidor.

## Endpoints

| Endpoint | Tipo | Descripción |
|----------|------|-------------|
| `GET /` | HTTP | Estado básico del servidor |
| `GET /status` | HTTP | Info detallada (GPU, modelo, etc.) |
| `WS /ws/inference` | WebSocket | Inferencia completa (steering + masks + detections) |
| `WS /ws/steering` | WebSocket | Solo steering (mínima latencia) |

## Protocolo WebSocket

### `/ws/steering` (recomendado para producción)

**Enviar:** bytes JPEG del frame

**Recibir:** JSON compacto
```json
{
  "s": 5.2,      // steering_angle (grados, -25 a 25)
  "c": 0.95,     // confidence (0 a 1)
  "e": 0.12,     // error_normalized (-1 a 1)
  "t": 23.5,     // inference_time_ms
  "f": 142       // frame_id
}
```

### `/ws/inference` (para debug/visualización)

**Enviar:** bytes JPEG del frame

**Recibir:** JSON completo con máscaras en base64, puntos de carril, detecciones, etc.

## Configuración

Editar `config.py` para ajustar:

- `SERVER_PORT`: Puerto del servidor (default: 8500)
- `DEVICE`: "cuda", "cpu", etc.
- `INPUT_WIDTH/HEIGHT`: Resolución del modelo (640x384)
- `USE_HALF`: FP16 para mayor velocidad en GPU
- `LOOKAHEAD_RATIO`: Qué tan adelante mirar para calcular dirección
- `STEERING_SMOOTHING`: Suavizado del ángulo

## Performance esperado

| Configuración | FPS inferencia | Latencia total |
|---------------|---------------|----------------|
| RTX 3060 + FP16 | ~60 FPS | ~25ms |
| RTX 2060 + FP16 | ~45 FPS | ~30ms |
| GTX 1660 + FP32 | ~25 FPS | ~50ms |
| CPU (i7) | ~5 FPS | ~200ms |

La latencia total incluye: compresión JPEG + envío + inferencia + respuesta.
En red local (WiFi 5GHz), el overhead de red es ~5-10ms.
