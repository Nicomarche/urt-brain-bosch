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
USB_DEVICE = 9  # /dev/video9 (USB Camera-B4.09.24.1)
USB_RESOLUTION = (640, 480)  # (ancho, alto)

# Mostrar ventana de preview de la camara (requiere monitor/display)
SHOW_CAMERA_PREVIEW = False
