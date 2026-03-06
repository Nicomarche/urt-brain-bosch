"""
Configuracion general del proyecto URT Brain.
Modifica estos valores para cambiar el comportamiento del auto.
"""

# ===================== PHYSICAL DIMENSIONS =====================
# Lane and road geometry (BFMC spec)
LANE_WIDTH_CM = 35.0           # carril: distancia entre bordes interiores de líneas
LINE_WIDTH_CM = 2.0            # ancho de las marcas viales pintadas

# Parking spot dimensions (BFMC spec)
PARKING_SPOT_LENGTH_CM = 76.5  # longitud del espacio de estacionamiento
PARKING_SPOT_WIDTH_CM  = 35.0  # ancho del espacio de estacionamiento

# Known object heights for metric distance estimation
SIGN_HEIGHT_CM          = 15.0  # altura típica de señal BFMC
TRAFFIC_LIGHT_HEIGHT_CM = 20.0  # altura típica de semáforo BFMC

# ===================== PARKING MANEUVER =====================
# Secuencia: LANE_KEEPING → SPOT_TRACKED → FORWARD_PAST_SPOT →
#            WAIT_STEER_1 → REVERSING_ENTRY → WAIT_STEER_2 →
#            REVERSING_ALIGN → WAIT_STEER_3 → FORWARD_CORRECTION → PARKED
# El spot se asume a la DERECHA del carril.

# --- Velocidades (mismas unidades que min_speed / max_speed) ---
PARKING_SEARCH_SPEED  = 13   # Velocidad buscando el spot
PARKING_FORWARD_SPEED = 13   # Velocidad en fases de avance
PARKING_REVERSE_SPEED = -13  # Velocidad en fases de reversa (negativo = atrás)

# --- Ángulos de dirección (grados; + = derecha, − = izquierda) ---
PARKING_ENTRY_STEER =  20.0  # Giro máx DERECHA: REVERSING_ENTRY y FORWARD_CORRECTION
PARKING_ALIGN_STEER = -20.0  # Giro máx IZQUIERDA: REVERSING_ALIGN

# --- Detección / seguimiento del spot ---
PARKING_SPOT_MISS_THRESHOLD  = 8      # Frames consecutivos sin detección → spot perdido
PARKING_TRIGGER_DISTANCE_CM  = 100.0  # Activar SPOT_TRACKED cuando el spot está a ≤ esta distancia (cm)

# --- Distancias de cada fase (odometría del encoder) ---
PARKING_D_FORWARD_CM         = 40.0   # Avanzar más allá del spot antes de reversar
PARKING_D_REVERSING_ENTRY_CM = 60.0   # Reversa con entry steer (meter el trasero al spot)
PARKING_D_REVERSING_ALIGN_CM = 22.0   # Reversa con align steer (alinear dentro del spot)
PARKING_D_FORWARD_CORR_CM    = 20.0   # Corrección hacia adelante con entry steer

# --- Tiempos fallback (cuando el encoder no reporta velocidad) ---
PARKING_T_FORWARD         = 1.5  # seg — FORWARD_PAST_SPOT
PARKING_T_REVERSING_ENTRY = 3.0  # seg — REVERSING_ENTRY
PARKING_T_REVERSING_ALIGN = 1.5  # seg — REVERSING_ALIGN
PARKING_T_FORWARD_CORR    = 1.5  # seg — FORWARD_CORRECTION

# --- Tiempo de espera para que el servo alcance el ángulo (seg) ---
PARKING_T_WAIT_STEER = 1.0  # WAIT_STEER_1 / WAIT_STEER_2 / WAIT_STEER_3

# ======================== CAMERA ========================
# Tipo de camara: "jetson" (CSI via GStreamer) | "picamera" (CSI via picamera2, RPi only) | "usb" (USB webcam)
CAMERA_TYPE = "jetson"

# Configuracion USB (solo aplica si CAMERA_TYPE = "usb")
# Device: numero de indice (0, 2, 4...) o path ("/dev/video0")
# Tip: correr `ls /dev/video*` para ver camaras disponibles
USB_DEVICE = 1  # /dev/video9 (USB Camera-B4.09.24.1)
USB_RESOLUTION = (640, 480)  # (ancho, alto)

# Configuracion Jetson CSI (solo aplica si CAMERA_TYPE = "jetson")
# Equivale al pipeline probado con `gst-launch-1.0 nvarguscamerasrc ...`.
JETSON_SENSOR_ID = 0         # CAM0=0, CAM1=1
JETSON_CAPTURE_RESOLUTION = (1280, 720)   # IMX219: 1280x720 solo soporta 60fps
JETSON_OUTPUT_RESOLUTION = (640, 480)     # Resolucion final entregada a OpenCV
JETSON_FRAMERATE = 60
JETSON_FLIP_METHOD = 2       # 2 = rotacion 180° (imagen invertida). 0 = sin flip

# HDR por fusion de exposiciones (Mertens) para PiCamera.
# Se aplica sobre el stream lores (640x384) usado por line following/sign detection.
# Default: activado para mejorar robustez en contraluz/sol directo.
PICAMERA_HDR_ENABLED = True

# Si True, aplica HDR en todos los frames. Si False, solo cuando detecta glare/saturacion.
PICAMERA_HDR_ALWAYS_ON = False

# Umbral de pixeles muy brillantes (0.0-1.0) para activar HDR cuando ALWAYS_ON=False.
# Ejemplo 0.04 = 4% de pixeles saturados.
PICAMERA_HDR_GLARE_THRESHOLD = 0.04

# Transmitir video de la camara al dashboard web (consume CPU por JPEG encode + base64).
# False = no envia video al browser (ahorra CPU), True = stream en vivo en la web
STREAM_CAMERA_TO_DASHBOARD = False

# ===================== DEBUG WINDOWS =====================
# Ventanas de OpenCV para debug visual (requieren monitor/display conectado).
# SHOW_CAMERA_PREVIEW actua como master switch: si es False, ninguna ventana se abre.
# Si es True, puedes elegir cuales abrir individualmente con DEBUG_WINDOWS.
SHOW_CAMERA_PREVIEW = True

# Ventanas individuales de debug (solo aplican si SHOW_CAMERA_PREVIEW = True)
DEBUG_WINDOWS = {
    "camera_preview":   False,  # Preview directo de la camara (raw frame)
    "final_result":     True,  # Resultado final con lineas detectadas y steering
    "binary_threshold": True,  # Vista del threshold binario
    "canny_edges":      True,  # Vista de bordes Canny
    "control_panel":    False,  # Panel de control con PID, velocidad, steering
    "steering_angle":   True,  # Angulo final de giro (calculado/comandado) en tiempo real
    "ai_analysis":      False,  # Analisis de LSTR / AI
    "hybrid_fusion":    False,  # Fusion hibrida OpenCV + LSTR
    "ai_local_overlay": True,  # Visualizacion local del modelo de IA (carriles + senales)
    "ai_local_masks":   True,  # Mascaras izquierda/derecha/combinada del modelo local
    "ai_local_signs":   True,  # Detecciones de senales/objetos no-carril del modelo local
}

# ===================== SIGN DETECTION =====================
# Deteccion de senales de trafico via AI Server remoto (WebSocket).
# El modelo MobilenetV2 SSD TFLite corre en el servidor, no en la RPi.
# Requiere: pip install websockets
ENABLE_SIGN_DETECTION = True

# URL WebSocket del AI Server (endpoint de senales de trafico)
# Legacy: ya no se usa en runtime cuando corre la percepcion local.
SIGN_SERVER_URL = "ws://localhost:8500/ws/signs"

# Ejecutar acciones al detectar senales (stop, reducir velocidad, etc.)
# False = solo detecta y publica (modo seguro para testing)
# True  = controla velocidad/direccion del auto
SIGN_DETECTION_ACTIONS = True

# Confianza minima para aceptar una deteccion (0.0 - 1.0)
SIGN_MIN_CONFIDENCE = 0.50

# Cooldown entre acciones de la misma senal (en segundos).
# Evita que frene multiples veces por la misma senal de stop al pasar cerca.
# Ejemplo: 15.0 = despues de frenar por un stop, ignora stops por 15 segundos.
SIGN_ACTION_COOLDOWN = 15.0

# Area minima del bounding box para EJECUTAR acciones (stop, frenar, etc.)
# Valor normalizado (0.0 - 1.0) = fraccion del area total de la imagen.
# Si la senal es muy chica (lejos), solo se detecta/publica pero NO se frena.
# Ejemplo: 0.01 = 1% del area de imagen (~senal a 2-3m de distancia)
#          0.03 = 3% del area de imagen (~senal a 1-1.5m de distancia)
# Tip: mirar los logs "box=X.X%" para calibrar este valor con tu camara.
SIGN_MIN_BOX_AREA = 0.03

# ===================== LOCAL AI PERCEPTION =====================
# Modelo local unificado (carriles + senales) ejecutado dentro de processCamera.
# En Jetson Nano debe usar el engine TensorRT generado en esta misma placa.
LOCAL_AI_MODEL_PATH = "models/lane_segmentation/Best416px.engine"
LOCAL_AI_MIN_CONFIDENCE = 0.35
LOCAL_AI_IMGSZ = 416
LOCAL_AI_DEVICE = "auto"  # "auto" | "cuda" | "cpu" | "mps"
# 0.04s ~= 25 FPS objetivo (permite sostener >=24 FPS si hardware acompana).
LOCAL_AI_INTERVAL = 0.04
LOCAL_AI_MAX_RESULT_AGE = 0.35
# Filtro post-proceso para evitar explosiones de cajas en engines TensorRT.
LOCAL_AI_NMS_IOU = 0.45
LOCAL_AI_MAX_DETECTIONS = 80
# Señales: umbral y limites de salida/debug.
LOCAL_AI_SIGN_MIN_CONFIDENCE = 0.55
LOCAL_AI_SIGN_MAX_DETECTIONS = 20
LOCAL_AI_SIGN_DEBUG_MAX_DETECTIONS = 8
# Si el engine devuelve etiquetas fallback ("class40"), las descarta salvo que
# exista un nombre en LOCAL_AI_CLASS_ID_NAME_MAP para ese ID.
LOCAL_AI_DROP_UNKNOWN_CLASSES = False
# Mapa opcional id->nombre para engines sin metadata. Dejar vacio por defecto.
LOCAL_AI_CLASS_ID_NAME_MAP = {}

# Alias de clases de carril del modelo local.
LOCAL_AI_LEFT_CLASS_ALIASES = [
    "left",
    "left_lane",
    "lane_left",
    "left line",
    "left-line",
    "izquierda",
    "carril_izquierdo",
]
LOCAL_AI_RIGHT_CLASS_ALIASES = [
    "right",
    "right_lane",
    "lane_right",
    "right line",
    "right-line",
    "derecha",
    "carril_derecho",
]
LOCAL_AI_GENERIC_LANE_ALIASES = [
    "lane",
    "lanes",
    "lane_line",
    "line",
    "carril",
]

# Mapeo de clases del modelo local hacia nombres canonicos usados por SignActions.
LOCAL_AI_SIGN_CLASS_MAP = {
    "car": "vehicle",
    "closed-road-stand": "road_block",
    "crosswalk-sign": "crosswalk",
    "highway-entry-sign": "highway_entrance",
    "highway-exit-sign": "highway_exit",
    "no-entry-road-sign": "no_entry",
    "one-way-road-sign": "one_way",
    "parking-sign": "parking",
    "parking-spot": "parking",
    "pedestrian": "pedestrian",
    "priority-sign": "priority",
    "round-about-sign": "roundabout",
    "stop-line": "stop_line",
    "stop-sign": "stop",
    "traffic-light": "traffic_light",
    "dur": "stop",
    "girisyok": "no_entry",
    "park": "parking",
    "yayagecidi": "crosswalk",
    "tasitrafiginekapali": "no_entry",
    "kirmizi": "red_light",
    "sari": "yellow_light",
    "yesil": "green_light",
    "20": "speed_20",
    "30": "speed_30",
    "sag": "turn_right",
    "sol": "turn_left",
    "sagadonulmez": "no_right_turn",
    "soladonulmez": "no_left_turn",
    "ilerisag": "straight_or_right",
    "ilerisol": "straight_or_left",
    "parkyasak": "no_parking",
    "parkyasak2": "no_parking",
    "otoyolgiris": "highway_entrance",
    "otoyolcikis": "highway_exit",
    "otobangiris": "highway_entrance",
    "otobancikis": "highway_exit",
    "otoyol_giris": "highway_entrance",
    "otoyol_cikis": "highway_exit",
    # Alias cortos usados por algunos datasets/modelos
    "hw_entry": "highway_entrance",
    "hw_entrance": "highway_entrance",
    "highway_entry": "highway_entrance",
    "hw_exit": "highway_exit",
    "durak": "bus_stop",
    "arac": "vehicle",
    "yaya": "pedestrian",
    "otobus": "bus",
    "bisikletli": "cyclist",
}
