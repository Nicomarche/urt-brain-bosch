"""
Configuración del AI Server para HybridNets.
Modifica estos valores según tu entorno.
"""

# ======================== SERVER ========================
SERVER_HOST = "0.0.0.0"       # Escucha en todas las interfaces
SERVER_PORT = 8500            # Puerto del servidor
MAX_CLIENTS = 4               # Máximo de clientes simultáneos

# ======================== MODEL ========================
# Peso del modelo HybridNets (se descarga automáticamente si no existe)
WEIGHTS_PATH = "weights/hybridnets.pth"

# Resolución de entrada para el modelo (ancho x alto)
# Debe ser múltiplo de 32. Menor = más rápido, mayor = más preciso
INPUT_WIDTH = 640
INPUT_HEIGHT = 384

# Backbone EfficientNet coefficient (0-7). Paper usa 3.
BACKBONE_COEFFICIENT = 3

# Umbral de confianza para detección de objetos
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.3

# ======================== INFERENCE ========================
# Dispositivo: "cuda", "cuda:0", "cuda:1", "cpu", "mps" (Apple Silicon)
DEVICE = "cuda"

# Usar half precision (FP16) para mayor velocidad en GPU
USE_HALF = True

# ======================== PROTOCOL ========================
# Formato de compresión para la respuesta de segmentación
# "msgpack" (binario, rápido), "json" (texto, debug friendly)
RESPONSE_FORMAT = "msgpack"

# Calidad JPEG al recibir frames (el cliente comprime antes de enviar)
# Si el servidor necesita re-comprimir para debug: 1-100
DEBUG_JPEG_QUALITY = 80

# ======================== LANE FOLLOWING ========================
# Parámetros para calcular el ángulo de dirección desde las detecciones
LOOKAHEAD_RATIO = 0.4          # Porcentaje de la imagen hacia adelante para calcular centro
LANE_WIDTH_ESTIMATE = 0.5      # Ancho estimado del carril como fracción del frame
STEERING_SMOOTHING = 0.6       # Suavizado del ángulo (0=sin suavizar, 1=máximo)

# Clases de segmentación de HybridNets
SEG_CLASSES = ['road', 'lane']  # 0=road (área manejable), 1=lane (líneas)

# Clases de detección de objetos
DET_CLASSES = ['car']

# ======================== VISUALIZATION ========================
# Mostrar ventanas de debug con OpenCV en el servidor
# True = abre ventanas mostrando entrada, segmentacion, detecciones, steering
# Requiere display (X11/Wayland). Desactivar en servidores headless.
SHOW_VISUALIZATION = True
