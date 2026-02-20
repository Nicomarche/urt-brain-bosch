"""
Configuracion general del proyecto URT Brain.
Modifica estos valores para cambiar el comportamiento del auto.
"""

# ======================== CAMERA ========================
# Tipo de camara: "picamera" (CSI ribbon cable) | "usb" (USB webcam)
CAMERA_TYPE = "picamera"

# Configuracion USB (solo aplica si CAMERA_TYPE = "usb")
# Device: numero de indice (0, 2, 4...) o path ("/dev/video0")
# Tip: correr `ls /dev/video*` para ver camaras disponibles
USB_DEVICE = 8  # /dev/video9 (USB Camera-B4.09.24.1)
USB_RESOLUTION = (640, 480)  # (ancho, alto)

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
    "ai_analysis":      False,  # Analisis de LSTR / AI
    "hybrid_fusion":    False,  # Fusion hibrida OpenCV + LSTR
}

# ===================== SIGN DETECTION =====================
# Deteccion de senales de trafico via AI Server remoto (WebSocket).
# El modelo MobilenetV2 SSD TFLite corre en el servidor, no en la RPi.
# Requiere: pip install websockets
ENABLE_SIGN_DETECTION = True

# URL WebSocket del AI Server (endpoint de senales de trafico)
SIGN_SERVER_URL = "ws://172.20.10.4:8500/ws/signs"

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
# Tip: mirar los logs "box=X.X%" para calibrar este valor con tu camara.
SIGN_MIN_BOX_AREA = 0.01
