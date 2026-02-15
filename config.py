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

# Mostrar ventana de preview de la camara (requiere monitor/display)
SHOW_CAMERA_PREVIEW = False

# ===================== SIGN DETECTION =====================
# Deteccion de senales de trafico via AI Server remoto (WebSocket).
# El modelo MobilenetV2 SSD TFLite corre en el servidor, no en la RPi.
# Requiere: pip install websockets
ENABLE_SIGN_DETECTION = True

# URL WebSocket del AI Server (endpoint de senales de trafico)
SIGN_SERVER_URL = "ws://192.168.80.15:8500/ws/signs"

# Ejecutar acciones al detectar senales (stop, reducir velocidad, etc.)
# False = solo detecta y publica (modo seguro para testing)
# True  = controla velocidad/direccion del auto
SIGN_DETECTION_ACTIONS = False

# Confianza minima para aceptar una deteccion (0.0 - 1.0)
SIGN_MIN_CONFIDENCE = 0.50
