"""
Configuración del AI Server.
Soporta multiples motores de inferencia: HybridNets, Supercombo, YOLO lane seg.
Modifica estos valores según tu entorno.
"""

# ======================== ENGINE SELECTION ========================
# Motor de inferencia: "hybridnets" | "supercombo" | "yolo_lane_seg"
ENGINE_TYPE = "yolo_lane_seg"

# ======================== SERVER ========================
SERVER_HOST = "0.0.0.0"       # Escucha en todas las interfaces
SERVER_PORT = 8500            # Puerto del servidor
MAX_CLIENTS = 4               # Máximo de clientes simultáneos

# ======================== HYBRIDNETS MODEL ========================
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
# En Jetson Nano usa "cuda" para aprovechar la GPU.
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

# ======================== YOLO LANE SEG ========================
# Modelo YOLOv8 de segmentacion para carril izquierdo/derecho.
# El modelo actual viene del ZIP BFMC YOLOv8n Segment 6402.
LANE_SEG_MODEL_PATH = "models/lane_segmentation/best.pt"
LANE_SEG_MIN_CONFIDENCE = 0.35
LANE_SEG_IMGSZ = 320
LANE_SEG_DEVICE = "auto"  # "auto" | "cuda" | "cpu" | "mps"

# Si solo se detecta un lado del carril, se estima el centro usando este ancho.
# Se interpreta como fraccion del ancho del frame.
LANE_SEG_FALLBACK_CENTER_OFFSET = 0.5

# Alias de clases del modelo para identificar izquierda/derecha.
# Si el modelo usa nombres distintos, ajustalos aqui.
LANE_SEG_LEFT_CLASS_ALIASES = [
    "left",
    "left_lane",
    "lane_left",
    "left line",
    "left-line",
    "izquierda",
    "carril_izquierdo",
]
LANE_SEG_RIGHT_CLASS_ALIASES = [
    "right",
    "right_lane",
    "lane_right",
    "right line",
    "right-line",
    "derecha",
    "carril_derecho",
]
LANE_SEG_GENERIC_CLASS_ALIASES = [
    "lane",
    "lanes",
    "lane_line",
    "line",
    "carril",
]

# Area minima de un componente conectado para considerarlo como carril.
LANE_SEG_MIN_COMPONENT_AREA_RATIO = 0.0008

# ======================== LANE FOLLOWING ========================
# Parámetros para calcular el ángulo de dirección desde las detecciones
LOOKAHEAD_RATIO = 0.4          # Porcentaje de la imagen hacia adelante para calcular centro
LANE_WIDTH_ESTIMATE = 0.5      # Ancho estimado del carril como fracción del frame
STEERING_SMOOTHING = 0.6       # Suavizado del ángulo (0=sin suavizar, 1=máximo)

# Clases de segmentación de HybridNets
SEG_CLASSES = ['road', 'lane']  # 0=road (área manejable), 1=lane (líneas)

# Clases de detección de objetos
DET_CLASSES = ['car']

# ======================== SUPERCOMBO MODEL ========================
# Modelo Supercombo de openpilot (comma.ai)
SUPERCOMBO_MODEL_PATH = "models/supercombo.onnx"
SUPERCOMBO_INPUT_WIDTH = 512    # Fijo por el modelo
SUPERCOMBO_INPUT_HEIGHT = 256   # Fijo por el modelo
SUPERCOMBO_LOOKAHEAD_IDX = 15   # Índice en X_IDXS (0-32) para calcular centro
SUPERCOMBO_STEERING_GAIN = 25.0 # Ganancia para convertir error normalizado a grados
SUPERCOMBO_SMOOTHING = 0.7      # Suavizado del ángulo (0=sin suavizar, 1=máximo)
SUPERCOMBO_USE_PATH = False     # True=usar trayectoria planeada, False=usar centro de lanes

# ======================== SIGN DETECTION ========================
# Deteccion de señales/elementos de trafico (YOLOv8)
# Requiere: pip install ultralytics
SIGN_DETECTION_ENABLED = True
SIGN_MODEL_DIR = "models/sign_detection"   # Directorio base de modelos
SIGN_MODEL_FILE = "trafic.pt"              # Fallback (modo 1 modelo)
SIGN_MIN_CONFIDENCE = 0.40                 # Umbral de confianza (0.0 - 1.0)
SIGN_IMGSZ = 320                           # Tamano de entrada (320=rapido, 640=preciso)
SIGN_DEVICE = "auto"                      # "auto" | "mps" (Apple Silicon) | "cuda" | "cpu"

# Elegir EXACTAMENTE un modelo activo para ejecutar.
# Opciones: "trafic_legacy" | "bfmc_320" | "bfmc_640"
SIGN_ACTIVE_MODEL = "bfmc_320"

# Perfiles disponibles de modelos.
SIGN_MODEL_OPTIONS = {
    "trafic_legacy": {
        "name": "trafic_legacy",
        "file": "trafic.pt",
        "kind": "sign",
        "imgsz": 320,
        "min_confidence": 0.40,
    },
    "bfmc_320": {
        "name": "bfmc_320",
        "file": "bfmc_detect_320_best.pt",
        "kind": "sign",
        "imgsz": 320,
        "min_confidence": 0.40,
    },
    "bfmc_640": {
        "name": "bfmc_640",
        "file": "bfmc_detect_640_best.pt",
        "kind": "sign",
        "imgsz": 640,
        "min_confidence": 0.40,
    },
}

# Orden de prioridad al devolver detecciones
SIGN_DETECTION_SORT_ORDER = {
    "sign": 0,
    "element": 1,
}

# Mapeo de nombres de clase del modelo -> nombres canonicos del auto.
# Las clases no listadas aqui se pasan tal cual (lowercase, espacios->_).
SIGN_CLASS_MAP = {
    # --- Modelo BFMC (ingles) ---
    "car": "vehicle",
    "closed-road-stand": "road_block",
    "crosswalk-sign": "crosswalk",
    "highway-entry-sign": "highway_entrance",
    "highway-exit-sign": "highway_exit",
    "no-entry-road-sign": "no_entry",
    "one-way-road-sign": "one_way",
    "parking-sign": "parking",
    "parking-spot": "parking_spot",
    "pedestrian": "pedestrian",
    "priority-sign": "priority",
    "round-about-sign": "roundabout",
    "stop-line": "stop_line",
    "stop-sign": "stop",
    "traffic-light": "traffic_light",
    # --- Modelo turco legacy (trafic.pt) ---
    "dur": "stop",
    "girisyok": "no_entry",
    "park": "parking",
    "yayagecidi": "crosswalk",
    "tasitrafiginekapali": "no_entry",
    # Semaforos
    "kirmizi": "red_light",
    "sari": "yellow_light",
    "yesil": "green_light",
    # Velocidad
    "20": "speed_20",
    "30": "speed_30",
    # Direccion
    "sag": "turn_right",
    "sol": "turn_left",
    "sagadonulmez": "no_right_turn",
    "soladonulmez": "no_left_turn",
    "ilerisag": "straight_or_right",
    "ilerisol": "straight_or_left",
    # Estacionamiento
    "parkyasak": "no_parking",
    "parkyasak2": "no_parking",
    # Autopista
    "otoyolgiris": "highway_entrance",
    "otoyolcikis": "highway_exit",
    "otobangiris": "highway_entrance",
    "otobancikis": "highway_exit",
    "otoyol_giris": "highway_entrance",
    "otoyol_cikis": "highway_exit",
    # Otros
    "durak": "bus_stop",
    "arac": "vehicle",
    "yaya": "pedestrian",
    "otobus": "bus",
    "bisikletli": "cyclist",
}

# ======================== VISUALIZATION ========================
# Mostrar ventanas de debug con OpenCV en el servidor
# True = abre ventanas mostrando entrada, segmentacion, detecciones, steering
# Requiere display (X11/Wayland). Desactivar en servidores headless.
SHOW_VISUALIZATION = True    # Abre ventana con bounding boxes en el servidor
