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

# ── Geometría de cámara para estimación de distancia al parking spot ─────────
# Cámara: IMX219 (Raspberry Pi Camera v2) en Jetson Nano/Orin.
# Pipeline: captura 1280×720 (modo 2×2 bin de crop 2560×1440) → escala a 640×480.
#
# CAMERA_HEIGHT_CM: altura del centro óptico (lente) al suelo en cm.
CAMERA_HEIGHT_CM = 17.0
#
# CAMERA_FY_480: focal length vertical en píxeles para imagen de 480px de alto.
# Derivación:
#   IMX219 f=3.04mm, pixel=1.12µm, crop 2560×1440, 2×2 binning → 1280×720
#   f_y @ 720px = (720/2)/tan(14.84°) = 1357px (pixels cuadrados)
#   Jetson escala 720→480 de forma no uniforme: f_y @ 480px = 1357 × 480/720 = 905px
CAMERA_FY_480 = 905.0
#
# CAMERA_PITCH_DEG: ángulo de inclinación de la cámara hacia abajo (desde horizontal).
# Derivado de parámetros IMX219 + observable px_per_cm ≈ 14.1 en reference_y=288px
# con H=17cm: β = 16.4°. Medir con inclinómetro para calibración precisa.
# Rango típico para robots BFMC: 10°–25°.
CAMERA_PITCH_DEG = 16.4

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
PARKING_ENTRY_STEER =  25.0  # Giro máx DERECHA: REVERSING_ENTRY y FORWARD_CORRECTION
PARKING_ALIGN_STEER = -25.0  # Giro máx IZQUIERDA: REVERSING_ALIGN

# --- Detección / seguimiento del spot ---
PARKING_SPOT_MISS_THRESHOLD  = 8      # Frames consecutivos sin detección → spot perdido
PARKING_TRIGGER_DISTANCE_CM  = 100.0  # Activar SPOT_TRACKED cuando el spot está a ≤ esta distancia (cm)

# --- Distancias de cada fase (odometría del encoder) ---
PARKING_D_FORWARD_CM         = 55.0   # Avanzar más allá del spot antes de reversar
PARKING_D_REVERSING_ENTRY_CM = 75.0   # Reversa con entry steer (meter el trasero al spot)
PARKING_D_REVERSING_ALIGN_CM = 45.0   # Reversa con align steer (alinear dentro del spot)
PARKING_D_FORWARD_CORR_CM    = 20.0   # Corrección hacia adelante con entry steer

# --- Tiempos fallback (cuando el encoder no reporta velocidad) ---
PARKING_T_FORWARD         = 1.5  # seg — FORWARD_PAST_SPOT
PARKING_T_REVERSING_ENTRY = 3.0  # seg — REVERSING_ENTRY
PARKING_T_REVERSING_ALIGN = 1.5  # seg — REVERSING_ALIGN
PARKING_T_FORWARD_CORR    = 1.5  # seg — FORWARD_CORRECTION

# --- Tiempo de espera para que el servo alcance el ángulo (seg) ---
PARKING_T_WAIT_STEER = 1.0  # WAIT_STEER_1 / WAIT_STEER_2 / WAIT_STEER_3

# --- Calibración de distancia cámara → spot ---
# ADVERTENCIA: La fórmula de estimación de distancia usa px_per_cm HORIZONTAL
# (calibrado del ancho de carril) para medir distancia VERTICAL hacia adelante.
# Esto es incorrecto en una imagen de perspectiva: px_per_cm horizontal ≈ 14,
# pero el scale vertical hacia adelante es ~1-2 px/cm. La fórmula siempre da
# < 30 cm, por lo que el trigger PARKING_TRIGGER_DISTANCE_CM=100 nunca se
# alcanza y el spot siempre se detecta en el primer frame. El factor de escala
# existe para calibración pero no resuelve el problema fundamental.
# TODO: reemplazar con estimación basada en parámetros intrínsecos de la cámara.
PARKING_DISTANCE_SCALE_FACTOR = 1.0

# --- Compensación de curva: pausar conteo de avance mientras el auto gira ---
# Cuando el parking se activa saliendo de una curva, el auto todavía está
# girando al iniciar FORWARD_PAST_SPOT. Si se contaran los 55 cm en arco,
# el auto quedaría mal posicionado. Con True: la odometría se pausa mientras
# curve_state sea IN_CURVE o ENTERING, y solo empieza a contar cuando el
# auto está en STRAIGHT o EXITING. El auto avanza hasta enderezarse, y luego
# cuenta los 55 cm en línea recta.
PARKING_CURVE_WAIT_FOR_STRAIGHT = True

# --- Compensación de avance cuando el spot se pierde estando adelante ---
# Con la fórmula de perspectiva corregida (CAMERA_*), last_dist es realista:
#   - Desaparición normal (spot al costado del auto): last_dist ≈ 30–45cm
#   - Desaparición en curva (spot todavía adelante): last_dist ≈ 50–80cm
# Si last_dist > threshold → se agrega el exceso al avance FORWARD_PAST_SPOT.
# Ajustar al ~90% del valor de desaparición normal observado en pruebas.
PARKING_SPOT_LOST_DIST_THRESHOLD_CM = 45.0

# --- Geometría del auto y límites de maniobra ---
# Longitud estimada del auto. Usada para calcular el espacio útil dentro del spot.
# Medir con cinta métrica el auto real y ajustar.
PARKING_CAR_LENGTH_CM        = 20.0

# Máximo de ciclos completos de (reversa-alineación-corrección) permitidos.
# El primer ciclo hace ENTRY+ALIGN+CORRECTION; los siguientes solo ALIGN+CORRECTION.
PARKING_MAX_MANEUVER_CYCLES  = 4

# Margen de seguridad respecto al fondo del spot: la reversa se detiene cuando
# la distancia restante al fondo es <= este valor.
PARKING_REAR_MARGIN_CM       = 5.0

# Margen de seguridad respecto a la entrada del spot: el avance de corrección
# se detiene cuando la distancia restante hacia adelante es <= este valor.
PARKING_FRONT_MARGIN_CM      = 5.0

# Reversa mínima útil: si el espacio restante hacia atrás es menor que esto,
# no se inicia otro ciclo de maniobra (el auto ya está bien posicionado).
PARKING_MIN_USEFUL_REVERSE_CM = 8.0

# Evita reenviar PARKING en cada frame mientras el parking_sign sigue visible.
PARKING_SIGN_COOLDOWN        = 8.0

# ═══════════════════════════════════════════════════════════════════════════
#                     PARÁMETROS DE CALIBRACIÓN DE GIRO
# ═══════════════════════════════════════════════════════════════════════════

# ── 1. MARGEN DE SEGURIDAD BASE ─────────────────────────────────────────────
# Distancia mínima (cm) que el borde del auto debe mantener con cada línea del
# carril. El ancho del auto es 19 cm → espacio libre en un carril de 35 cm =
# (35-19)/2 = 8 cm. Con LANE_SAFETY_MARGIN_CM = 5 empieza a corregir cuando
# queda < 5 cm de holgura entre el auto y la línea.
# Rango típico: 2.0 – 7.0 cm
#   → Más alto: dobla más abierto (más lejos de la línea interior)
#   → Más bajo: permite ir más cerca de la línea
LANE_SAFETY_MARGIN_CM = 5.0

# ── 2. GEOMETRÍA DEL AUTO ────────────────────────────────────────────────────
# Batalla (distancia entre ejes) en cm. Se usa para calcular el "offtracking"
# del eje trasero: cuando el auto gira, la parte trasera sigue un radio MÁS
# CORTO que la cámara delantera (igual que un camión).
# Fórmula: offtracking = sqrt(R² + L²) − R   con R = wheelbase / tan(|δ|)
# A 25° de giro con wheelbase=26 cm → offtracking ≈ 5.8 cm.
# Valor físico del TC-04: 26.0 cm
CAR_WHEELBASE_CM = 26.0

# ── 3. CORRECCIÓN DE OFFTRACKING (desvío del eje trasero en curvas) ──────────
# Cuando el auto gira, el eje trasero corta más hacia adentro que la cámara.
# Esta corrección amplía el margen de seguridad del lado interior en la cantidad
# calculada de offtracking, para que la ESQUINA TRASERA no toque la línea.
#
# OFFTRACK_SCALE:
#   Factor de escala 0.0 – 1.0 que multiplica el offtracking calculado.
#   → 0.0: corrección desactivada (comportamiento pre-fix, más cerrado en curvas)
#   → 0.5: corrección al 50% (compromiso)
#   → 1.0: corrección física completa (más abierto en curvas)
#   PROBLEMA CONOCIDO: el offtracking usa el steering del frame anterior;
#   si ese valor fue ±25° (corrección extrema), puede generar oscilación en
#   la recta. Si eso ocurre, reducir a 0.3–0.5 o usar OFFTRACK_CURVE_ONLY=True.
OFFTRACK_SCALE = 0.5

# OFFTRACK_CURVE_ONLY:
#   True  → el offtracking SÓLO se aplica durante ENTERING / IN_CURVE / EXITING
#            (evita la oscilación en recta causada por el steering previo extremo)
#   False → se aplica siempre (incluso en STRAIGHT)
OFFTRACK_CURVE_ONLY = True

# ── 4. CONTROLADOR STANLEY ───────────────────────────────────────────────────
# Fórmula: δ = heading_error + arctan(STANLEY_K · e / (STANLEY_K_SOFT + v))
# En modo 2-líneas, heading_error = 0 (sólo el término crosstrack importa).
#
# STANLEY_K [1/s] — ganancia del término de error lateral (crosstrack):
#   Con K_SOFT=3.0, el denominador efectivo es ~3.20 a v=0.20 m/s.
#   Ejemplos de salida crosstrack pura (sin heading) a v=0.20 m/s:
#     k=0.8  → e=10cm:  1.4°   e=25cm:  3.6°   e=40cm:  5.7°  (muy lento en recta)
#     k=2.5  → e=10cm:  4.5°   e=25cm: 11.2°   e=40cm: 17.4°  (optimo: recta rápida)
#     k=4.0  → e=10cm:  7.1°   e=25cm: 17.2°   e=40cm: 23.0°  (agresivo, puede oscilar)
#   Rango recomendado (con K_SOFT=3.0): 1.5 – 4.0
#   → Más alto: corrección en recta más rápida. Curvas con error >20cm pueden saturar.
#   → Más bajo: corrección en recta muy lenta (error acumula antes de corregir).
#   NOTA: K_SOFT=3.0 hace que K necesite ser 2-4x mayor que el valor histórico de 0.8.
STANLEY_K      = 2.5

# STANLEY_K_SOFT [m/s] — suavizado a baja velocidad (evita saturar el término crosstrack):
#   Con v_min = 0.13 m/s, el denominador efectivo es (K_SOFT + v).
#   Rango calibrado para este vehículo: 1.0 – 5.0 m/s
#   → 3.0: óptimo — crosstrack contribuye 2-7° en curvas, heading domina (gradual)
#   → 1.0: más agresivo — empieza a saturar en curvas cerradas
#   → 5.0: muy suave — corrección lateral muy lenta en rectas
#   NOTA: el valor previo de 0.20 causaba saturación inmediata (25°) en toda curva
#         porque atan(0.8×e / 0.33) >> 25° para cualquier e > 0.04m
STANLEY_K_SOFT = 3.0

# STANLEY_K_D_STEER [adimensional] — amortiguamiento de servo de dirección:
#   Término de "lead" del paper de Hoffmann: k_d_steer × (δ_meas(i) - δ_meas(i+1))
#   Resiste cambios bruscos del servo. Se aplica en radianes al steering command.
#   Rango: 0.0 – 0.30
#   → 0.0: sin amortiguamiento (más reactivo, puede oscilar en transiciones de curva)
#   → 0.10: valor nominal (atenúa ~4° en transiciones de 42°)
#   → 0.30: muy amortiguado (respuesta lenta pero muy suave)
#   NOTA: el bug de steer_x10 fue corregido. Con el fix, el efecto es 10x menor
#   que antes. Si el auto oscila, bajar a 0.05; si va muy rígido, subir a 0.20.
STANLEY_K_D_STEER = 0.10

# ── 5. AMORTIGUAMIENTO DEL STEERING (historial de frames) ────────────────────
# El steering final se promedia sobre los últimos N frames para suavizar
# cambios bruscos entre frames. Con N=1 no hay suavizado (reacción inmediata).
# Rango: 1 – 5
#   → 1: sin suavizado (más reactivo, puede oscilar)
#   → 3: buen compromiso
#   → 5: muy suave pero con mayor retardo
STEER_HISTORY_LEN = 2

# ── 6. FILTRO DE RUIDO (noise filter) ────────────────────────────────────────
# Descarta frames donde el cambio de steering es demasiado grande de golpe
# (probablemente ruido de detección), manteniendo el steering previo por hasta
# NOISE_MAX_REJECT_FRAMES frames consecutivos.
#
# NOISE_MAX_STEER_JUMP_DEG: cambio máximo permitido de steering entre frames [°]
#   Rango: 10 – 30°
#   → Más bajo: filtra más agresivamente (puede ignorar correcciones legítimas)
#   → Más alto: permite cambios bruscos (menos filtrado)
NOISE_MAX_STEER_JUMP_DEG   = 15

# NOISE_MAX_REJECT_FRAMES: frames consecutivos que se pueden rechazar antes de
# aceptar el nuevo valor incondicionalmente.
#   Rango: 1 – 5
#   → Más alto: filtra por más tiempo (puede causar retardo en curvas)
NOISE_MAX_REJECT_FRAMES    = 3

# ── 7. FACTOR DE GIRO EN MODO UNA SOLA LÍNEA ─────────────────────────────────
# Cuando el auto sólo ve la línea INTERIOR (está saliendo del carril por la curva
# más cerrada), aplica un steering de escape igual a:
#   CURVE_INNER_LINE_STEER_FACTOR × MAX_STEERING
# Rango: 0.2 – 0.8
#   → 0.4: gira al 40% del máximo (suave, evita escaparse por el otro lado)
#   → 0.7: gira más fuerte (corrección más rápida pero riesgo de sobrepasar)
CURVE_INNER_LINE_STEER_FACTOR = 0.6

# ── 8. SUAVIZADO EN TRANSICIONES DE MODO ─────────────────────────────────────
# Al pasar de modo 2-líneas a 1-línea (o cambio de lado), el steering se mezcla
# gradualmente durante N frames para evitar un salto brusco.
# Rango: 0 – 4
#   → 0: sin suavizado (inmediato)
#   → 2: mezcla en 2 frames (valor por defecto)
SINGLE_LINE_BLEND_FRAMES = 2

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
# IMPORTANTE: Los signos de hw_entry / hw_exit son verticales y se ven de costado;
# su box area máxima en el track es ~1.5–2% (nunca llegan a 3%). Las señales
# frontales (STOP, cruce) sí pueden superar el 3%, pero 1% es suficiente filtro
# dado que ya hay un umbral de confianza (SIGN_MIN_CONFIDENCE=0.50).
# Calibración por observación de logs "box=X.X%":
#   box ≈ 0.3–2%  → highway_entry / highway_exit (signos de pared, vistos lateralmente)
#   box ≈ 1–10%   → señales frontales a 0.5–1.5m
#   box ≈ 15–25%  → parking spot (suelo, muy cerca)
SIGN_MIN_BOX_AREA = 0.010

# Umbral de box area por señal, sobreescribe SIGN_MIN_BOX_AREA para señales
# específicas. Útil para señales vistas lateralmente a alta velocidad (hw_exit),
# donde el bounding box nunca supera el 1% aunque el auto esté cerca.
# Calibración por observación de logs: hw_exit llega a ~0.6% máximo.
SIGN_MIN_BOX_AREA_PER_SIGN = {
    "highway_exit":     0.003,   # 0.3% – señal lateral, vista a alta velocidad
    "highway_entrance": 0.003,   # 0.3% – señal lateral, igual que exit
}

# ===================== TRAFFIC LIGHT OPENCV =====================
# El modelo local detecta el bbox del semaforo ("traffic_light"). El color se
# decide dentro de ese recorte con OpenCV, sin otro modelo de AI.
TRAFFIC_LIGHT_OPENCV_ENABLED = True

# En AUTO, si hay un semaforo visible y vigente, solo se permite velocidad
# positiva cuando el estado clasificado es green_light.
TRAFFIC_LIGHT_HOLD_ENABLED = True
TRAFFIC_LIGHT_HOLD_TIMEOUT_S = 1.5
TRAFFIC_LIGHT_MIN_BOX_AREA = SIGN_MIN_BOX_AREA
TRAFFIC_LIGHT_GREEN_CONFIRMATIONS = 2

# AUTO startup replay waits here instead of consuming the hardcoded move at red.
STARTUP_MOVE_WAIT_FOR_GREEN_LIGHT = True

# Defaults portados del LightClassifier de referencia.
TRAFFIC_LIGHT_RED_HSV_1 = ((0, 150, 150), (10, 255, 255))
TRAFFIC_LIGHT_RED_HSV_2 = ((170, 150, 150), (180, 255, 255))
TRAFFIC_LIGHT_YELLOW_HSV = ((22, 150, 150), (28, 255, 255))
TRAFFIC_LIGHT_GREEN_HSV = ((60, 150, 150), (85, 255, 255))
TRAFFIC_LIGHT_ADAPTIVE_SAT_MIN = 100.0
TRAFFIC_LIGHT_ADAPTIVE_SAT_MAX = 200.0
TRAFFIC_LIGHT_ADAPTIVE_VAL_MIN = 50.0
TRAFFIC_LIGHT_ADAPTIVE_VAL_MAX = 150.0
TRAFFIC_LIGHT_MIN_CIRCULARITY = 0.357
TRAFFIC_LIGHT_MIN_LIGHT_HEIGHT_RATIO = 0.08
TRAFFIC_LIGHT_MAX_LIGHT_HEIGHT_RATIO = 0.40
TRAFFIC_LIGHT_MAX_CENTER_X_OFFSET_RATIO = 0.20
TRAFFIC_LIGHT_BRIGHTNESS_THRESHOLD_SCALE = 0.8595
TRAFFIC_LIGHT_BRIGHTNESS_THRESHOLD_MIN = 0.20
TRAFFIC_LIGHT_BACKUP_COLOR_RATIO_MIN = 0.20

# ===================== LINE FOLLOWING - VELOCIDADES =====================
# Escala interna 0–25; el motor recibe speed*10 (ej: LF_MAX_SPEED=13 → 130 PWM).
# Valores actuales: base normal muy baja para que la autopista sea notoria.
LF_BASE_SPEED         = 15   # Velocidad inicial al arrancar
LF_MAX_SPEED          = 15   # Velocidad máxima en modo normal (recta)  → motor 130
LF_MIN_SPEED          = 10    # Velocidad mínima en modo normal (curva)   → motor 80
LF_HIGHWAY_MAX_SPEED         = 30   # Velocidad máxima en autopista             → motor 300
LF_HIGHWAY_MIN_SPEED         = 28   # Velocidad mínima en autopista (curva HW)  → motor 280
LF_SPEED_RAMP_STEP           = 1.0  # Incremento máximo por frame (aceleración gradual)
LF_HIGHWAY_SPEED_RAMP_STEP   = 3.0  # Aceleración más rápida en autopista (recuperación post-curva)
# Factor de reducción de velocidad por steering en autopista (0.0 = sin reducción, 1.0 = igual que modo normal).
# Con 0.0 el auto mantiene max speed incluso durante correcciones en autopista.
LF_HIGHWAY_STEER_SPEED_FACTOR = 0.2

# ===================== AUTO STARTUP MANUAL MOVE =====================
# Trayectoria manual fija que se reproduce una vez al entrar en AUTO,
# antes de liberar el seguimiento autonomo de carril.
STARTUP_MOVE_PATH = "temp/startup_manual_trajectory.json"
STARTUP_MOVE_MAX_DURATION_S = 20.0
STARTUP_MOVE_AUTO_REPLAY = True

# ===================== SIGN ACTIONS - VELOCIDADES =====================
# Velocidades enviadas al motor por las acciones de señales.
# Escala interna 0–10; se transmiten como speed*10 sobre el canal SpeedMotor
# (ej: SIGN_HIGHWAY_SPEED=7 → envía "70").
# Estas velocidades son interpretadas por el controlador de motor directamente
# y son INDEPENDIENTES de la rampa de velocidad del line-following (0–250).
SIGN_BASE_SPEED      = 5    # Velocidad base: marcha normal fuera de autopista
SIGN_LOW_SPEED       = 3    # Velocidad reducida: semáforo amarillo, zona escolar
SIGN_SPEED_20        = 3    # Límite 20 km/h
SIGN_SPEED_30        = 5    # Límite 30 km/h
SIGN_HIGHWAY_SPEED   = 10   # Velocidad autopista (se activa con highway_entrance)
SIGN_STOP_DURATION      = 3.0   # Segundos detenido en señal STOP / no_entry
SIGN_CROSSWALK_DURATION = 3.0   # Segundos detenido en cruce peatonal

# ── Giro a la izquierda tras señal STOP ───────────────────────────────────────
# Después de la parada completa, el auto realiza un giro izquierda de 90° con
# un radio de 1,02 m de forma hardcodeada, ignorando el seguimiento de línea.
#
# Estrategia de control:
#   Se usa el ÁNGULO MÁXIMO de dirección (-25°) para que las ruedas giren a tope
#   y el giro sea claramente visible. El arco recorrido se controla con la
#   DURACIÓN: más segundos → más grados girados.
#
#   Radio efectivo con steer máximo (TC-04, wheelbase=26 cm, steer=25°):
#     R = L / tan(25°) = 26 / 0.466 ≈ 55.8 cm
#   Arco 90° a ese radio: arc = (π/2) × 55.8 ≈ 87.6 cm
#
# SIGN_STOP_TURN_DURATION: tiempo para recorrer el arco de 90°. Calibrar en el
# hardware real: aumentar si el auto gira menos de 90°, reducir si gira más.
# Secuencia completa: STOP → ruedas a 0° → ruedas a -25° (0.5s settle) → AVANZAR.
SIGN_STOP_LEFT_TURN_ENABLED = True    # Activar/desactivar el giro post-stop
SIGN_STOP_TURN_STEER_DEG   = -25.0  # Ángulo máximo izquierda (negativo = izquierda)
SIGN_STOP_TURN_SPEED       = 10      # Velocidad durante el giro (misma escala que SIGN_BASE_SPEED)
SIGN_STOP_TURN_DURATION    = 21   # Segundos para completar el arco de 90° — CALIBRAR

# ===================== LOCAL AI PERCEPTION =====================
# Modelo local unificado (carriles + senales) ejecutado dentro de processCamera.
# En Jetson Nano debe usar el engine TensorRT generado en esta misma placa.
# Para generar el engine desde el ONNX reentrenado, ejecutar:
#   cd models/lane_segmentation && python build_trt.py
LOCAL_AI_MODEL_PATH = "models/lane_segmentation/Best_weights_reentrenado_416px.engine"
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
    "parking-sign": "parking_sign",
    "parking_sign": "parking_sign",
    "parking sign": "parking_sign",
    "parking-spot": "parking_area",
    "parking_spot": "parking_area",
    "parking-area": "parking_area",
    "parking_area": "parking_area",
    "pedestrian": "pedestrian",
    "priority-sign": "priority",
    "round-about-sign": "roundabout",
    "stop-line": "stop_line",
    "stop-sign": "stop",
    "traffic-light": "traffic_light",
    "dur": "stop",
    "girisyok": "no_entry",
    "park": "parking_sign",
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
    # Walk area / pedestrian crossing area
    "walk_area": "walk_area",
    "walk area": "walk_area",
    "zebra_crossing": "walk_area",
    "zebra crossing": "walk_area",
    "yayabolgesi": "walk_area",
}

# Walk area behavior
WALK_AREA_MIN_BOX_AREA  = 0.04  # min bbox area (normalized) to react — filters far detections
WALK_AREA_SLOW_SPEED_CM_S = 10.0  # speed cap while crossing an empty walk_area
WALK_AREA_CLEAR_GRACE   = 0.5   # seconds without walk_area before returning to normal speed
