"""
Configuracion general del proyecto URT Brain.
Modifica estos valores para cambiar el comportamiento del auto.
"""

import math as _math
import os as _os, sys as _sys
# Sim mode: defaults ON for macOS (no Jetson HW), OFF for Linux. Override with
# URT_SIM_MODE=1/0. Used downstream to flip CAMERA_TYPE, MOTOR_OUTPUT, and
# disable cv2 GUI windows that crash with the headless pip-installed OpenCV.
_SIM_MODE = _os.environ.get("URT_SIM_MODE", "1" if _sys.platform == "darwin" else "0") == "1"

# ===================== PHYSICAL DIMENSIONS =====================
# Lane and road geometry (BFMC spec)
LANE_WIDTH_CM = 35.0           # carril: distancia entre bordes interiores de líneas
LINE_WIDTH_CM = 2.0            # ancho de las marcas viales pintadas

# ── Visual lane → MPC reference (paradigma urt-ref) ─────────────────────────
# La percepción ajusta polinomios a las líneas detectadas y emite waypoints
# del centro de carril; esos waypoints son la referencia primaria del MPC en
# `LaneKeep`. Mantienen la curvatura del polinomio incluso cuando se ve una
# sola línea (la otra se sintetiza desplazando lateralmente LANE_WIDTH_M).
LANE_VISUAL_WAYPOINT_DENSITY_M = 0.032          # paso entre waypoints generados (≈ urt-ref)
LANE_VISUAL_WAYPOINT_COUNT = 40                 # cantidad mínima de waypoints emitidos
LANE_VISUAL_MIN_POLY_POINTS = 8                 # mínimo de muestras por línea para fittear
LANE_VISUAL_POLY_DEGREE_HIGH = 3                # grado del polinomio con ≥12 muestras
LANE_VISUAL_POLY_DEGREE_LOW = 2                 # grado con <12 muestras (evita overshoot)
LANE_VISUAL_MIN_QUALITY_FOR_PRIMARY_PATH = 0.55 # quality mínima para usar waypoints como path

# Lane keeping: autoridad primaria de cámara sólo en tramos normales. En zonas
# de maniobra (turn_direction/intersection/roundabout/stopline/etc.) manda mapa.
LANE_VISUAL_PRIMARY_ENABLED = True
LANE_VISUAL_PRIMARY_ALLOWED_SCENARIOS = {"lane_keep"}
LANE_VISUAL_PRIMARY_TWO_LINE_MIN_QUALITY = 0.75
LANE_VISUAL_PRIMARY_SINGLE_LINE_MIN_QUALITY = 0.85
LANE_VISUAL_PRIMARY_MAP_AUTHORITY_DISTANCE_M = 0.60
LANE_VISUAL_PRIMARY_MIN_FORWARD_SPAN_M = 0.25
LANE_VISUAL_PRIMARY_MAX_LATERAL_TO_FORWARD_RATIO = 1.2
LANE_VISUAL_PRIMARY_MAX_HEADING_ERROR_DEG = 35.0
LANE_VISUAL_PRIMARY_ENTER_TICKS = 3
LANE_VISUAL_PRIMARY_EXIT_TICKS = 2

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
# Horizontal field of view used to align AI bounding boxes with 2D LiDAR
# sectors. RPi Camera v2 is ~62.2° horizontal in the 16:9 capture mode.
CAMERA_HORIZONTAL_FOV_DEG = 62.2
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

# ── LATERAL MPC ───────────────────────────────────────────────────────────────
# Reemplaza el controlador Stanley por un MPC de horizonte reciente usando el
# mismo modelo cinemático de bicicleta que el repo de referencia (ACADOS MPC),
# pero implementado en Python puro (scipy) sin necesidad de GPS ni ACADOS.
#
# USE_LATERAL_MPC: True = activa MPC, False = usa Stanley clásico.
USE_LATERAL_MPC = True

# MPC_WHEELBASE [m]: distancia entre ejes del robot.
# BFMC robot: 0.260 m (igual que los modelos *_beta del repo de referencia).
MPC_WHEELBASE = 0.260

# MPC_N: horizonte de predicción (número de pasos).
# Más alto = anticipa mejor las curvas pero más lento de resolver.
# Rango recomendado: 8–15. Con N=10 y dt=0.033s → 330ms de anticipación.
MPC_N = 10

# MPC_DT [s]: paso de tiempo de la predicción.
# Debe coincidir con la tasa de control (loop de ~30Hz → 0.033s).
MPC_DT = 0.033

# Pesos del costo cuadrático:
#   Q_e    — penaliza error lateral (m²). Mayor = corrección más agresiva.
#   Q_psi  — penaliza error de heading (rad²). Mayor = mantiene mejor la alineación.
#   R      — penaliza el esfuerzo de steering (rad²). Mayor = steering más suave.
#   R_rate — penaliza el cambio de steering entre pasos (rad²). Mayor = más suave.
#   Q_e_N / Q_psi_N — pesos del costo terminal (N-ésimo paso).
MPC_Q_E    = 10.0
MPC_Q_PSI  = 5.0
MPC_R      = 0.5
# R_RATE=2.0 era demasiado alto: bloqueaba el optimizador cerca del prev_delta
# y causaba salidas con signo incorrecto cuando la inercia venía de una
# corrección anterior en sentido contrario.  0.1 permite reversiones rápidas.
MPC_R_RATE = 0.1
MPC_Q_E_N  = 20.0
MPC_Q_PSI_N = 10.0

# MPC_HEADING_DEADBAND_DEG: zona muerta de heading del MPC [°].
# Stanley usa 1.0° — en el MPC se fija en 0° porque el término crosstrack
# aumentado puede producir psi_raw muy pequeño (~0.05°) por cancelación con el
# heading real del IMU; zeroing ese valor deja el degenerate case psi=0.
MPC_HEADING_DEADBAND_DEG = 0.0

# MPC_CROSSTRACK_K_MULT: multiplicador del término crosstrack en el MPC.
# k_eff = STANLEY_K × k_mult.  Con k_mult=1.0 el término es idéntico al
# Stanley clásico (atan2(k×e, k_soft+v)).  Valores > 1 hacen la corrección
# más agresiva. 1.4 era necesario para superar la cancelación causada por el
# override de heading del IMU en modo 2-líneas; ese override fue corregido
# (ahora solo activa cuando no hay detección de líneas), así que k_mult=1.0
# es suficiente y corresponde exactamente al gain probado de Stanley.
MPC_CROSSTRACK_K_MULT = 1.0

# MPC_OUTPUT_DEADBAND_DEG: zona muerta de salida del MPC [°].
# Stanley usa 1.2° — el MPC necesita una zona muerta más pequeña porque
# con el término crosstrack augmentado la salida óptima es pequeña para
# errores moderados (~5-8 cm). 0.5° permite correcciones finas sin chattering.
MPC_OUTPUT_DEADBAND_DEG = 0.5

# ── ACADOS FULL MPC (trajectory-tracking) ────────────────────────────────────
# MPC completo que optimiza VELOCIDAD + DIRECCIÓN simultáneamente usando el
# solver Acados (código C generado).  Requiere generar el solver una vez:
#   python -m src.control._acados_solver_gen
#
# USE_ACADOS_MPC pertenece al controlador visual legacy de threadLineFollowing.
# El runtime principal ya usa MotionController -> AcadosMPC por defecto y solo
# cambia a PurePursuit si se setea URT_FORCE_PURE_PURSUIT=1 explícitamente.
# Mantener este flag en False evita activar dos ramas de steering en el legacy.
USE_ACADOS_MPC = False

# Horizonte de predicción del solver Acados.  N × T = tiempo interno total.
# urt-ref usa N=40, T=0.10 en los modelos *_beta; el control runtime puede
# seguir corriendo a 20 Hz, pero la dinámica que resuelve el OCP mira 4.0 s.
ACADOS_MPC_N = 40
ACADOS_MPC_T = 0.10

# Velocidad de referencia por defecto [m/s].  El MPC optimiza alrededor de
# este valor.  En competencia ajustar según la zona (highway vs curva).
ACADOS_MPC_V_REF = 0.20

# Modelo del vehículo.
ACADOS_MPC_WHEELBASE = 0.260      # distancia entre ejes [m] (urt-ref)
ACADOS_MPC_L_R = 0.105            # eje trasero a CG [m] (urt-ref)
ACADOS_MPC_L_F = 0.155            # eje delantero a CG [m]

# Límites de control.
ACADOS_MPC_V_MAX = 0.40           # velocidad máxima [m/s] en AUTO = 40 cm/s
ACADOS_MPC_V_MIN = -0.50          # velocidad mínima [m/s] (reversa)
ACADOS_MPC_DELTA_MIN_DEG = -21.5  # steering mínimo [°], asimétrico como urt-ref
ACADOS_MPC_DELTA_MAX_DEG = 25.0   # steering máximo [°]

# Pesos del costo (NONLINEAR_LS).
#   Q: penaliza desviación de la trayectoria de referencia (x, y, yaw).
#   R: penaliza esfuerzo de control (v, delta).
#   S: penaliza tasa de cambio de controles (Δv, Δδ).
ACADOS_MPC_X_COST = 2.0
ACADOS_MPC_Y_COST = 2.0
ACADOS_MPC_YAW_COST = 0.5
ACADOS_MPC_V_COST = 1.0
ACADOS_MPC_STEER_COST = 0.0
ACADOS_MPC_DELTA_V_COST = 1.5
ACADOS_MPC_DELTA_STEER_COST = 0.75

# Plan B1.2: perfiles de pesos por escenario. El BehaviorPlanner llama a
# MotionController.set_weight_profile() al entrar al scenario. ACADOS soporta
# update_weights() runtime (no requiere regenerar el solver).
#
# Valores del perfil "parking" tomados del REF (mpc_config_park.yaml):
# y_cost=7 → fuerza alineación lateral (clave para parallel parking).
ACADOS_MPC_PROFILES = {
    "default": {
        "x": 2.0, "y": 2.0, "yaw": 0.5, "v": 1.0,
        "steer": 0.0, "dv": 1.5, "dsteer": 0.75,
    },
    "parking": {
        "x": 1.0, "y": 7.0, "yaw": 1.0, "v": 0.5,
        "steer": 0.0, "dv": 0.0, "dsteer": 0.0,
    },
    "highway": {
        "x": 2.0, "y": 1.5, "yaw": 0.3, "v": 1.5,
        "steer": 0.0, "dv": 2.0, "dsteer": 1.0,
    },
    "lane_keep_visual": {
        "x": 1.5, "y": 5.0, "yaw": 1.2, "v": 1.0,
        "steer": 0.0, "dv": 0.5, "dsteer": 0.25,
    },
    "map_turn_authority": {
        "x": 2.5, "y": 3.5, "yaw": 1.6, "v": 1.0,
        "steer": 0.0, "dv": 0.5, "dsteer": 0.35,
    },
    # Override perfilable A/B (plan B1.1): ref tiene dv=0.25, dsteer=0.5 —
    # más permisivo. Probarlo en sim antes de hacer default.
    "ref_permissive": {
        "x": 2.0, "y": 2.0, "yaw": 0.5, "v": 1.0,
        "steer": 0.0, "dv": 0.25, "dsteer": 0.5,
    },
}

# Override perfilable: si está seteado a nombre de profile válido, el
# MotionController arranca con ese perfil. Por defecto "default".
MPC_WEIGHT_PROFILE = "default"

# Zona muerta de salida del MPC completo [°].
ACADOS_MPC_OUTPUT_DEADBAND_DEG = 0.25

# USE_ACADOS_SPEED: True = usar velocidad del MPC para el motor.
# False = solo usar steering del MPC, mantener control de velocidad heurístico.
USE_ACADOS_SPEED = False

# Plan A3.3: motor de fusión de localización seleccionado en boot.
#   * "yaw_ekf_1d"     : default — yaw 1D + DR plano (lo de hoy).
#   * "ekf7"           : activa el EKF7 modular de src.localization.ekf.state_filter.
#   * "dead_reckoning" : sólo DR sin filtro (debug).
# Override por env: URT_LOCALIZATION_FILTER.
LOCALIZATION_FILTER = "yaw_ekf_1d"

# ═══════════════════════════════════════════════════════════════════════════════
# GPS-FREE TRACKING (dead reckoning + OSM lanelet centerlines)
# ═══════════════════════════════════════════════════════════════════════════════

# Track assets switch by platform: maps/sim/ for the Mac+simulator scenario,
# maps/jetson/ for the real BFMC car. Override the sub-directory with
# URT_TRACK_MAP_DIR=<absolute_or_relative_path> when testing alternative maps.
_DEFAULT_TRACK_MAP_DIR = "maps/sim" if _SIM_MODE else "maps/jetson"
TRACK_MAP_DIR = _os.environ.get("URT_TRACK_MAP_DIR", _DEFAULT_TRACK_MAP_DIR)

def _default_lanelet2_osm_path(track_map_dir: str) -> str:
    candidates = ("lanelet2_map.osm",)
    for name in candidates:
        candidate = _os.path.join(track_map_dir, name)
        if _os.path.exists(candidate):
            return candidate
    return _os.path.join(track_map_dir, candidates[0])

TRACKING_LANELET2_OSM = _os.environ.get(
    "URT_TRACKING_LANELET2_OSM",
    _default_lanelet2_osm_path(TRACK_MAP_DIR),
)
TRACKING_START_LANELET_ID = _os.environ.get("URT_TRACKING_START_LANELET_ID")
TRACKING_META_JSON = _os.path.join(TRACK_MAP_DIR, "track_meta.json")
# Background for the OpenCV visualizer. Prefers the SVG (vector source of
# truth); falls back to PNG/JPG with the same basename if cairosvg is not
# available at runtime. See src/routing/visualizer.py:_load_background.
TRACKING_BG_SVG    = _os.path.join(TRACK_MAP_DIR, "track.svg")
_TRACKING_BG_PNG = _os.path.join(TRACK_MAP_DIR, "track.png")
_TRACKING_BG_JPG = _os.path.join(TRACK_MAP_DIR, "track.jpg")
TRACKING_BG_RASTER = _os.path.join(
    TRACK_MAP_DIR,
    "track.png" if _SIM_MODE or _os.path.exists(_TRACKING_BG_PNG) else "track.jpg",
)

# Spline interpolation step (metres).  Smaller → denser waypoints, more CPU.
TRACKING_WAYPOINT_STEP_M = 0.05

# Car advances to next waypoint when it is within this distance (metres).
TRACKING_ADVANCE_DIST_M = 0.15

# Lookahead distance (metres) used to detect STOPLINE/INTERSECTION nodes ahead.
# When a precision node is within this distance the tracker switches to
# waypoint-mode control in threadLineFollowing.
TRACKING_INTERSECTION_LOOKAHEAD_M = 0.40

# Inside STOPLINE/INTERSECTION precision zones use a much shorter control
# lookahead so the target waypoint stays on the active node segment instead of
# jumping into the next curve before the car actually reaches the node.
TRACKING_PRECISION_LOOKAHEAD_M = 0.10

# Speed scale applied to the encoder reading before dead-reckoning integration.
# 1.0 = use encoder speed as-is.  If the virtual car advances faster than the
# real one, reduce this value (e.g. 0.85).  If it lags behind, increase it.
# Typical range: 0.7 – 1.0 (wheel slip, encoder calibration, actuator lag).
TRACKING_DR_SPEED_SCALE = 1.0

# Steering gain used by dead reckoning.  1.0 = trust the measured steering
# angle directly; values > 1.0 amplify the effective wheel angle.
TRACKING_STEER_GAIN_DR = 1.0
# Steering sign used by tracking dead reckoning.
# El pipeline nuevo mantiene la misma convención que publican control y Nucleo
# para no tener una inversión escondida entre pose, planner y dashboard.
#
# NOTA (sim): probé `=-1` en sim para invertir el sentido de rotación del
# DR kinemático y alinearlo con el frame OSM (sim_bridge integra `ω=+v·tan(δ)/WB`
# que crece en right turn, opuesto a CW-desde-este). El experimento confirmó
# alineación perfecta de yaw vs ground truth (mm_match_error p90=0.083m,
# pose_drift mean=0.107m), PERO rompió el motor publishing path: dispatcher
# emite motion_controller cmds pero zmq_motor stops at ~5s. Hay un guard
# downstream (probable safety_gate o CurrentSteer feedback loop) que recibe
# steer con signo invertido y rechaza. Pendiente: tracear cuál guard es y
# decidir si fixear el guard o aplicar la negación más arriba (motion
# controller output) en lugar del DR input.
TRACKING_STEER_SIGN_DR = 1.0

# Signo del yaw IMU antes de sumarlo al offset de calibración.
#
#   Real hardware observado: un giro a la derecha debe hacer crecer yaw en el
#   frame OSM/dashboard. Mantenemos +1.0 como default para que el yaw IMU no
#   quede espejado; si una IMU/firmware concreto publica el signo opuesto, este
#   valor sigue siendo el override de compatibilidad.
#
#   Sim (sim_bridge feedback_yaw_sign=1.0): el simulador integra
#   ω = +v·tan(δ)/WB, por lo que un giro a la derecha hace crecer
#   yaw_deg → la negación del hardware daría el signo incorrecto.
#   Con _IMU_YAW_SIGN = +1 la señal del sim coincide con la convención
#   del dead reckoning sin invertir.
TRACKING_IMU_YAW_SIGN = 1.0

# Confianza del yaw absoluto del BNO055 frente al servo de dirección.
# El pose estimator propaga yaw con modelo bicicleta (velocidad + wheelbase +
# steering filtrado) y usa el IMU sólo como corrección cuando la dirección está
# casi quieta. Esto evita que el campo/ruido del servo rote la odometría.
TRACKING_IMU_STEER_TRUST_FULL_DEG = 3.0       # ≤ este steer: IMU con confianza plena
TRACKING_IMU_STEER_TRUST_ZERO_DEG = 8.0       # ≥ este steer: IMU inhibido
TRACKING_IMU_STEER_RATE_TRUST_FULL_DEGS = 20.0  # servo lento: IMU permitido
TRACKING_IMU_STEER_RATE_TRUST_ZERO_DEGS = 90.0  # servo rápido: IMU inhibido
TRACKING_IMU_STEER_INHIBIT_HOLD_S = 0.35      # holdoff tras steer fuerte/rápido

# DEPRECATED: el yaw de sim ya llega en `brain_map` desde `sim_bridge`, así que
# no hace falta una corrección extra en el brain. Se conserva en config sólo
# para compatibilidad con checkouts viejos.
SIM_IMU_YAW_OFFSET_DEG = 0.0

# Forzar PurePursuit en lugar de AcadosMPC. Default: Acados también en sim.
# PurePursuit queda como escape manual para diagnóstico sin cambiar código:
#   URT_FORCE_PURE_PURSUIT=1  fuerza PurePursuit
#   URT_FORCE_PURE_PURSUIT=0  fuerza Acados
_FORCE_PP_ENV = _os.environ.get("URT_FORCE_PURE_PURSUIT")
FORCE_PURE_PURSUIT = (
    False
    if _FORCE_PP_ENV is None
    else _FORCE_PP_ENV.strip().lower() in {"1", "true", "yes", "on"}
)

# Filtro de lag del actuador de dirección para el dead reckoning.
# Modela el delay entre el comando de steering y la posición real de las ruedas.
# 1.0 = instantáneo (sin filtro), 0.0 = ruedas nunca responden.
# Empezar con 0.7 y ajustar: si el DR gira más rápido que el auto real → bajar.
TRACKING_STEER_LAG_ALPHA = 1.0

# Yaw estimation EKF — fuses IMU absolute heading with the kinematic (bicycle) yaw rate.
# Kalman gain:  K = P / (P + R)
# Measurement noise:  R = R_STRAIGHT + R_STEER_K × steer_rad²
#   steer ≈  0° → R ≈ R_STRAIGHT (small)  → K ≈ 1  → trust IMU fully
#   steer ≈ 25° → R ≈ 0.005 + 50×0.19 ≈ 9.7 rad²  → K ≈ 0  → ignore IMU, use kinematics
# Replaces the hard _IMU_STEER_INHIBIT_DEG cutoff with a smooth, principled transition.
#
# Tuning guide:
#   R_STRAIGHT:  increase if IMU is noisy when straight (vibration, mag interference)
#   R_STEER_K:   increase if heading drifts during turns (servo EMI strong on your unit)
#   Q:           increase if kinematic model drifts fast (encoder slip, steer inaccuracy)
TRACKING_YAW_EKF_Q          = 1e-4   # process noise [rad²/s] — kinematic drift rate
TRACKING_YAW_EKF_R_STRAIGHT = 0.005  # IMU noise [rad²] when straight (≈ 4° std dev)
TRACKING_YAW_EKF_R_STEER_K  = 50.0   # R grows by this per rad² of steering angle
TRACKING_YAW_EKF_P_INIT     = 0.5    # initial covariance [rad²] — high = trust first IMU

# ---------------------------------------------------------------------------
# EKF7 — Phase 2: full 7-state planar localizer.
# Replaces the legacy yaw-only EKF above when Phase 2 wires the new pose
# estimator. Sigmas are *standard deviations*, not variances.
#
# El filtro vive en `src/localization/ekf/state_filter.py` y se inicializa
# con (lat0, lon0) del origen ENU local. La pista de Cluj BFMC entra en
# ~50×50 m, así que el plano TM es plano-plano.
# ---------------------------------------------------------------------------

# Origen del marco ENU local. Cluj-Napoca, BFMC 2024 venue.
# Si corrés en otra pista, sustituí estos valores con un GPS fix
# tomado al borde de la pista al inicio de la sesión.
LOCAL_FRAME_ORIGIN_LATLON = (46.7682, 23.5870)

# Ruido GPS (m). El GPS BFMC tiene 1–5 m noise + 100–300 ms latencia,
# así que conservamos un sigma alto para que no domine la fusión.
EKF_GPS_R_M = 1.5

# Ruido IMU (yaw_rate en rad/s, accel longitudinal en m/s²).
EKF_IMU_R_OMEGA = 0.015        # ≈ 0.86°/s std
EKF_IMU_R_ACCEL = 0.30         # m/s² — incluye vibración de chasis

# Ruido del encoder de rueda (m/s).
EKF_ENCODER_R_VX = 0.10

# Ruido del lane-normal update (offset lateral en m).
EKF_R_LANE_NORMAL_M = 0.05

# Ruido del landmark match (sigma metros, asume isotropía simple).
EKF_LANDMARK_R_M = 0.30

# Tiempo máximo aceptable de un GPS fix antes de descartarlo (s).
# A 100–300 ms de latencia BFMC, 0.5 s es generoso pero no patológico.
EKF_GPS_MAX_AGE_S = 0.5

# ============================================================================
# BFMC LOCSYS GPS
# ============================================================================
# Protocolo de competencia: el coche conecta al TrafficCommunicationServer
# para obtener la IP del locsys device, luego conecta al locsys device
# (TCP:4691) y recibe {"x": float, "y": float}\n a 1 Hz.
# `TRAFFIC_COMM_HOST="auto"` escucha el broadcast UDP del servidor (puerto
# 9000 por defecto). Para fijar la IP de competencia, cambiarlo acá a algo
# como "192.168.50.2".
#
# En simulación: sim_bridge expone el servidor locsys en localhost:4691
# y usa el ground truth de Gazebo transformado al frame del mapa OSM.
#
# El cliente (threadLocSys) envía el fix como Localisation IPC con
# world_x/world_y (coordenadas OSM directas, sin conversión de imagen).

LOCSYS_PORT          = 4691
# Fallback directo al locsys device. En competencia normal NO debería usarse:
# el coche debe pedir esa IP al TrafficCommunicationServer. Queda como escape
# manual sólo si `LOCSYS_DIRECT_FALLBACK_ENABLED=True`.
# No es la IP del TrafficCommunicationServer; esa se controla con TRAFFIC_COMM_HOST.
# Este valor queda ignorado mientras LOCSYS_DIRECT_FALLBACK_ENABLED=False.
LOCSYS_HOST_COMP     = "192.168.50.2"
TRAFFIC_COMM_HOST    = _os.environ.get(
    "URT_TRAFFIC_COMM_HOST",
    "127.0.0.1" if _SIM_MODE else "auto",
)
TRAFFIC_COMM_PORT    = 5000
TRAFFIC_COMM_AUTODISCOVERY_ENABLED = str(TRAFFIC_COMM_HOST).strip().lower() == "auto"
TRAFFIC_COMM_DISCOVERY_PORT = 9000
TRAFFIC_COMM_DISCOVERY_TIMEOUT_S = 5.0
TRAFFIC_COMM_PUBLIC_KEY_PATH = "auto"
LOCSYS_DEVICE_ID     = 3
# Modo de GPS via TrafficCommunicationServer:
#   "auto"      -> intenta request locsysDevice; si el server no lo reconoce,
#                  usa suscripcion locIDsub directa al TrafficCommunicationServer.
#   "request"   -> solo locsysDevice -> IP:puerto del locsys device.
#   "subscribe" -> solo locIDsub -> stream {"type":"location","x","y","z"}.
TRAFFIC_COMM_LOCSYS_MODE = "auto"
TRAFFIC_COMM_LOCSYS_SUB_FREQ = 0.25
# El stream locIDsub del TrafficCommunicationServer llega en metros.
# El frame del server usa origen abajo-izquierda, x+ hacia la derecha e y+
# hacia arriba. El GPS thread lo transforma al frame world del Lanelet/OSM.
TRAFFIC_COMM_LOCSYS_SUB_COORD_SCALE = 1.0
TRAFFIC_COMM_LOCSYS_SUB_COORD_FRAME = "track_bottom_left"
# "auto" usa el lower-left de TRACKING_LANELET2_OSM/track_meta como origen
# world correspondiente al (0,0) del TrafficCommunicationServer.
TRAFFIC_COMM_LOCSYS_SUB_ORIGIN_WORLD_X = "auto"
TRAFFIC_COMM_LOCSYS_SUB_ORIGIN_WORLD_Y = "auto"
# Si el server usa una extension distinta a la del OSM, completar estos
# valores para normalizar: world_width / server_width y world_height / server_height.
# "auto" usa el ancho/alto de track_meta.world_bounds, que coincide con los
# ejes amarillos de la pista.
TRAFFIC_COMM_LOCSYS_SUB_MAP_WIDTH_M = "auto"
TRAFFIC_COMM_LOCSYS_SUB_MAP_HEIGHT_M = "auto"

SIM_LOCSYS_HOST      = "localhost"
SIM_LOCSYS_PORT      = 4691
LOCSYS_USE_TRAFFIC_COMM_SERVER = False if _SIM_MODE else True
LOCSYS_DIRECT_FALLBACK_ENABLED = _os.environ.get("URT_LOCSYS_DIRECT_FALLBACK", "0") == "1"
TRAFFIC_COMM_SEND_EGO_DATA = False if _SIM_MODE else True
TRAFFIC_COMM_SEND_PERIOD_S = 0.25

_GPS_ENABLED_DEFAULT = "0" if _SIM_MODE else "1"
GPS_ENABLED          = _os.environ.get("URT_GPS_ENABLED", _GPS_ENABLED_DEFAULT) == "1"  # habilitar threadLocSys
GPS_RECONNECT_S      = 2.0             # segundos entre reintentos de conexión

# DEPRECATED: el bridge nativo ahora hace toda la conversión `brain_map ↔ gz`
# vía `sim_bridge_frames.json`. El brain ya no debería usar estos offsets para
# teleport ni para logging; quedan sólo como referencia de migración.
GZ_OFFSET_X   = 2.984
GZ_OFFSET_Y   = -0.596
GZ_LATERAL_OFFSET_M = 1.09
GZ_SPAWN_Z    = 0.002   # altura de spawn del coche sobre el plano

# ============================================================================
# BEHAVIOR PLANNER (Phase 4 — Autoware-inspired single source of truth)
# ============================================================================
# El BehaviorPlanner es la ÚNICA fuente de verdad de velocidad: ningún otro
# módulo decide cuánto va a ir el auto. El MotionController (MPC) consume
# `BehaviorOutput.target_path` + `speed_profile` y produce el MotorCommand.
#
# Tasa nominal: 20 Hz (dt = 0.05 s). El MPC corre al mismo dt, así que el
# tamaño del horizonte se mide en steps de 50 ms. El solver Acados generado
# puede usar T=0.10 internamente; esta tasa es la del planner/control loop.
BEHAVIOR_DT_S = 0.05
BEHAVIOR_HORIZON_N = 40  # must match N_horizon in c_generated_code/acados_ocp_bfmc_bicycle.json
                         # N=40 queda alineado con urt-ref; T del solver vive en ACADOS_MPC_T.

# El solver Acados tiene N fijo (=40). El repo urt-ref generaba referencias
# con paso espacial v_ref*T ~= 0.32*0.10 = 3.2 cm, o sea ~1.28 m de preview.
# Si retimeamos sólo con v*dt a 25 cm/s, el preview cae a 0.5 m y el auto
# empieza a doblar tarde.
ACADOS_MPC_RETIME_REFERENCE_BY_SPEED = True
ACADOS_MPC_REFERENCE_MIN_STEP_M = 0.032
ACADOS_MPC_REFERENCE_MAX_PREVIEW_M = 1.28

# Límites competitivos de velocidad. 0.0 sigue siendo válido para STOP; cuando
# el auto está en movimiento no debe mandar menos de 20 cm/s.
BEHAVIOR_MIN_SPEED_MPS = 0.20       # velocidad mínima de movimiento = 20 cm/s
BEHAVIOR_CITY_MIN_SPEED_MPS = 0.20  # ciudad: mínimo de movimiento = 20 cm/s
BEHAVIOR_HIGHWAY_SPEED_MPS = 0.25 if _SIM_MODE else 0.80
BEHAVIOR_HIGHWAY_MIN_SPEED_MPS = 0.20 if _SIM_MODE else 0.40  # autopista: mínimo de movimiento
TRAFFIC_SIGN_LOW_SPEED_MPS = 0.10   # parking/crosswalk: excepción visible = 10 cm/s
TRAFFIC_SIGN_STOP_HOLD_S = 3.0      # STOP: tiempo detenido antes de continuar
SIGN_HINT_MAX_AGE_S = 1.5           # vida útil de hints de señales en BehaviorPlanner
TRAFFIC_SIGN_STOP_COMMAND_DISTANCE_M = 0.40  # STOP: con lidar <= 40cm, frenar y sostener
TRAFFIC_SIGN_STOP_CAMERA_FALLBACK_DISTANCE_M = 0.40  # STOP: fallback si no hubo asociación lidar
TRAFFIC_SIGN_STOP_CAMERA_FALLBACK_MIN_CONFIDENCE = 0.65
TRAFFIC_SIGN_STOP_CAMERA_FALLBACK_MIN_BOX_AREA = 0.010
TRAFFIC_SIGN_STOP_RETRIGGER_SUPPRESS_S = 5.0
TRAFFIC_SIGN_NO_ENTRY_MIN_CONFIDENCE = 0.70  # no-entry debe ser estable/fuerte para bloquear ruta
TRAFFIC_SIGN_NO_ENTRY_MIN_BOX_AREA = 0.003   # 0.3% del frame, evita falsos positivos chicos
TRAFFIC_SIGN_NO_ENTRY_CONFIRM_TICKS = 2      # detecciones consecutivas antes de bloquear lanelet
LIDAR_AI_SIGN_SECTOR_MIN_HALF_WIDTH_DEG = 6.0
LIDAR_AI_SIGN_SECTOR_EXTRA_DEG = 2.0
LIDAR_AI_SIGN_SECTOR_MAX_HALF_WIDTH_DEG = 16.0

# Velocidad nominal de lane_keep "limpio" (sin signs, sin regulators).
BEHAVIOR_NOMINAL_SPEED_MPS = 0.25   # target base; Acados re-temporiza preview con v_ref.
                                    # El planner sigue siendo la fuente única
                                    # de verdad para el perfil de velocidad.

# Cap por curvatura. A escala BFMC, el valor viejo (0.45 m/s²) casi nunca
# bajaba una velocidad nominal de 0.25 m/s; este límite fuerza curvas cerradas
# a caer al piso competitivo antes de entrar.
BEHAVIOR_CURVE_A_LAT_MAX_MPS2 = 0.08
BEHAVIOR_CURVE_SPEED_FLOOR_MPS = 0.10
BEHAVIOR_INTERSECTION_SPEED_MPS = 0.12
BEHAVIOR_INTERSECTION_MIN_SPEED_MPS = 0.08

# Hard cap absoluto. Aplicado por velocity_overlay al final, ningún
# scenario puede emitir velocidades por encima.
BEHAVIOR_MAX_SPEED_MPS = 0.40       # cap absoluto de AUTO = 40 cm/s

# Geometría y containment lateral del planner. Los valores están escalados
# para BFMC (carril 35 cm, vehículo ~19 cm de ancho) y se usan para exigir
# que la referencia publicada al MPC permanezca dentro del corredor del mapa.
BEHAVIOR_VEHICLE_WIDTH_M = 0.19
BEHAVIOR_CONTAINMENT_CLEARANCE_M = 0.01
BEHAVIOR_CONTAINMENT_WARN_ERROR_M = 0.05
BEHAVIOR_CONTAINMENT_CRAWL_ERROR_M = 0.07
BEHAVIOR_CONTAINMENT_CRAWL_SPEED_MPS = 0.25

# Stuck-recovery del LaneContainmentRule. Si el robot lleva varios ticks
# en crawl (4 cm/s) sin que el error lateral decrezca, sube temporalmente
# el cap a RECOVERY_SPEED_MPS para que pueda maniobrar y volver al carril.
# A 4 cm/s con steering al máximo el yaw rate alcanza pero la traslación
# es tan lenta que el centerline del próximo lanelet rota más rápido que
# el ego avanza → loop estable de crawl. STUCK_TICKS=40 ≈ 2 s a 20 Hz.
BEHAVIOR_CONTAINMENT_STUCK_TICKS = 40
BEHAVIOR_CONTAINMENT_RECOVERY_SPEED_MPS = 0.25

# Aceleración máxima del ramp de velocidad en el BehaviorPlanner [m/s²].
# 0.25 m/s² → llega a 0.15 m/s en ~0.6 s (12 ticks a 20 Hz).
BEHAVIOR_ACCEL_MPS2 = 0.25

# Rate limiter de velocidad en el output del MotionController [m/s²].
# Garantiza arranque gradual independientemente del solver.
# 0.25 m/s² → 0→0.15 m/s en ~0.6 s (12 ticks).
BEHAVIOR_MAX_SPEED_RATE_MPS2 = 0.25

# Entrada a AUTO: relocalización GPS, ruta y lanzamiento suave.
AUTO_GPS_SAMPLE_COUNT = 3
AUTO_GPS_COLLECTION_TIMEOUT_S = 2.0
AUTO_GPS_MAX_FIX_AGE_S = 1.0
AUTO_GPS_MAX_SPREAD_M = 0.35
AUTO_GPS_MAX_LANELET_DISTANCE_M = 0.50
AUTO_GPS_MAX_YAW_DIFF_RAD = _math.pi / 2.0
# Calibracion manual/startup en el punto de inicio: promedia varias lecturas
# GPS antes de anclar el GPS crudo al start guardado.
START_GPS_CALIBRATION_SAMPLE_COUNT = 5
START_GPS_CALIBRATION_TIMEOUT_S = 4.0
START_GPS_CALIBRATION_MAX_FIX_AGE_S = 1.5
START_GPS_CALIBRATION_MAX_SPREAD_M = 0.35
START_GPS_CALIBRATION_STATUS_HOLD_S = 4.0
AUTO_ENTRY_ROUTE_RESET_ENABLED = True
AUTO_LAUNCH_READY_TICKS = 3
AUTO_LAUNCH_HOLD_TIMEOUT_S = 3.0
MOTOR_COMMAND_STALE_TIMEOUT_S = 0.30

# Velocidad máxima de cambio del ángulo de steering [°/s].
# A 20 Hz (dt=0.05 s) → 120°/s = 6°/tick → 0→25° en ~4 ticks (0.2 s).
# Bajar si el auto sigue oscilando; subir si es demasiado lento en curvas.
BEHAVIOR_MAX_STEER_RATE_DEG_S = 120.0

# Lookahead del PurePursuitSolver (fallback sin acados).
# Con el piso reglamentario de 20 cm/s, 30 cm de lookahead abre demasiado la
# entrada de curvas chicas. Apuntamos a ~22 cm a 20 cm/s y dejamos que crezca
# con la velocidad hasta el techo de competencia.
BEHAVIOR_MIN_LOOKAHEAD_M = 0.22
BEHAVIOR_LOOKAHEAD_GAIN_S = 1.1

# Tasa de actualización del thread (s). Con pause=0.05 corremos a ~20 Hz,
# emparejando dt del MPC.
BEHAVIOR_THREAD_PAUSE_S = 0.05

# ===================== 2D LIDAR =====================
# Native 2D LiDAR input. Hardware reads LD19/STL-19P over serial; sim/mac reads
# LaserScan snapshots from Gazebo through sim_bridge.py over ZMQ.
# En hardware queda opt-in por defecto. El reader YA NO autodetecta el puerto:
# antes globbeaba /dev/ttyACM* y se quedaba con los bytes que la Nucleo le
# mandaba al serial handler, dejando al brain ciego a la telemetría. Sin
# URT_LIDAR_SERIAL_PORT el thread publica `no_port_configured` y no abre nada.
# Activar con:
#   URT_LIDAR_ENABLED=1 URT_LIDAR_SERIAL_PORT=/dev/ttyUSB0 ./run.sh
LIDAR_ENABLED = _os.environ.get("URT_LIDAR_ENABLED", "1" if _SIM_MODE else "0") == "1"
LIDAR_BACKEND = _os.environ.get(
    "URT_LIDAR_BACKEND",
    "zmq" if _SIM_MODE else "serial",
).strip().lower()
LIDAR_SERIAL_PORT = _os.environ.get("URT_LIDAR_SERIAL_PORT") or None
LIDAR_SERIAL_BAUD = 230400
LIDAR_SERIAL_TIMEOUT_S = 0.5
LIDAR_BINS = 720
LIDAR_RANGE_MIN_M = 0.05
LIDAR_RANGE_MAX_M = 4.0
LIDAR_ROLLING_TTL_S = 0.5
LIDAR_YAW_OFFSET_DEG = 0.0
LIDAR_CLOCKWISE = False
LIDAR_OBSTACLE_CLUSTER_RANGE_M = 1.2
LIDAR_CLUSTER_MIN_INTENSITY = 5.0
LIDAR_CLUSTER_MAX_GAP_DEG = 5.0
LIDAR_CLUSTER_MIN_POINTS = 3
LIDAR_CLUSTER_KEEP_SINGLE_BELOW_M = 1.20
LIDAR_REQUIRED = False
LIDAR_STALE_TIMEOUT_S = 0.5
LIDAR_EMERGENCY_OBSTACLE_ENABLED = False
LIDAR_EMERGENCY_DISTANCE_M = 0.28
LIDAR_EMERGENCY_HALF_WIDTH_M = 0.14
LIDAR_OBSTACLE_STOP_ENABLED = False
LIDAR_OBSTACLE_STOP_DISTANCE_M = 0.45
LIDAR_OBSTACLE_SLOW_DISTANCE_M = 0.90
LIDAR_OBSTACLE_CLOSE_SLOW_DISTANCE_M = 0.35
LIDAR_OBSTACLE_CLOSE_SLOW_SPEED_MPS = BEHAVIOR_MIN_SPEED_MPS
LIDAR_OBSTACLE_CORRIDOR_HALF_WIDTH_M = 0.14
LIDAR_OBSTACLE_MIN_FORWARD_X_M = 0.12
LIDAR_AI_SECTOR_MIN_HALF_WIDTH_DEG = 2.0
LIDAR_AI_SECTOR_EXTRA_DEG = 0.0
LIDAR_AI_SECTOR_MAX_HALF_WIDTH_DEG = 12.0
LIDAR_AI_OBJECT_SECTOR_MIN_HALF_WIDTH_DEG = 24.0
LIDAR_AI_OBJECT_SECTOR_EXTRA_DEG = 4.0
LIDAR_AI_OBJECT_SECTOR_MAX_HALF_WIDTH_DEG = 30.0
LIDAR_AI_MATCH_DISTANCE_M = 0.30
LIDAR_AI_MATCH_ANGLE_DEG = 28.0
LIDAR_AI_MATCH_XY_M = 0.28
LIDAR_AI_MATCH_MIN_FORWARD_X_M = -0.02
LIDAR_AI_MATCH_MAX_ABS_ANGLE_DEG = 120.0
LIDAR_AI_MATCH_MAX_SCORE = 1.35
ZMQ_LIDAR_ENDPOINT = "tcp://localhost:5578"
ZMQ_LIDAR_TOPIC = b"lidar"

# ===================== OVERTAKE =====================
OVERTAKE_ENABLED = True
OVERTAKE_FORWARD_MIN_M = LIDAR_OBSTACLE_MIN_FORWARD_X_M
OVERTAKE_FORWARD_MAX_M = 1.25
OVERTAKE_LATERAL_OFFSET_M = 0.16
OVERTAKE_MIN_OBSTACLE_CLEARANCE_M = 0.04
# Plan B1.3: half-lane BFMC del repo de referencia (0.31 m). Usamos este
# valor como techo del desplazamiento lateral durante overtake — alinear
# con el ref permite usar todo el carril opuesto cuando hace falta.
OVERTAKE_LANE_OFFSET_M = 0.31
OVERTAKE_MAX_LATERAL_OFFSET_M = OVERTAKE_LANE_OFFSET_M
OVERTAKE_SIDE_CLEAR_FORWARD_M = 1.25
OVERTAKE_SIDE_CLEAR_HALF_WIDTH_M = 0.12
OVERTAKE_SPEED_MPS = 0.30
OVERTAKE_RAW_LIDAR_FALLBACK_ENABLED = True
OVERTAKE_RAW_LIDAR_REQUIRES_HIGHWAY = True
OVERTAKE_RAW_LIDAR_FORWARD_MAX_M = 0.70
OVERTAKE_RAW_LIDAR_CONFIRM_TICKS = 3
OVERTAKE_RAW_LIDAR_MIN_POINTS = 12
OVERTAKE_RAW_LIDAR_MIN_WIDTH_DEG = 8.0
OVERTAKE_RAW_LIDAR_MIN_CONFIDENCE = 0.55

# ----------------------------------------------------------------------------
# Auto-lap mode (sim/demo): el coche da vueltas siguiendo el loop de lanelets
# OSM cuando NADIE le mandó un destino. Para operación normal conviene dejarlo
# apagado así AUTO sin ruta = auto detenido hasta recibir una misión.
#
# Implementación: navigation_planner_thread llama `reset_route(current_pose)`
# del PathManager una vez que hay pose válida. PathManager.reset_route ya rota
# `reference_node_ids` para empezar por la lanelet más cercana y cierra el loop.
# ----------------------------------------------------------------------------
AUTO_LAP_MODE = False
# Cada cuánto reintentamos `reset_route` mientras todavía no hay ruta activa
# (solo aplica antes de que el primer reset_route exitoso). En segundos.
AUTO_LAP_RETRY_PERIOD_S = 1.0

# Camera-based lateral correction applied to dead reckoning when both lane lines
# are visible and the physical lane error is reliable.
TRACKING_CAMERA_LATERAL_CORRECTION_GAIN = 0.18
TRACKING_CAMERA_LATERAL_CORRECTION_MAX_M = 0.02
TRACKING_CAMERA_CORRECTION_MIN_SPEED_MPS = 0.02
# Extra limiter over the per-frame camera lateral correction so two-line mode
# can nudge the DR back toward the real lane center without teleporting it.
TRACKING_CAMERA_LATERAL_CORRECTION_STEP_MAX_M = 0.015
TRACKING_CAMERA_LATERAL_CORRECTION_COOLDOWN_S = 0.10

# Plan B2: perfiles de corrección visual por escenario. La autoridad de la
# visión sobre la pose depende del scenario activo:
#   * lane_keep      → más fuerte (estamos en carril limpio).
#   * default        → estándar.
#   * intersection,
#     roundabout    → débil (mapa OSM manda en geometría compleja).
#   * parking        → desactivada (la maniobra es servo-directo, no
#                      queremos que el lane perception meta artefactos).
# El BehaviorPlanner setea el perfil activo en TrackingState al entrar al
# scenario; pose_estimator/relocalization lee de ahí cuál usar.
VISUAL_CORRECTION_PROFILES = {
    "default": {
        "gain": 0.18,
        "max_m": 0.02,
        "step_max_m": 0.015,
    },
    "lane_keep": {
        # Recta autopista: la cámara manda. Cuando el mapa OSM tiene error
        # de localización (~5-8 cm), este gain alto permite que la corrección
        # visual jale al pose lateralmente con autoridad real (~1.5 cm/frame,
        # techo 4 cm). Antes 0.22/0.025/0.018 → respuesta lenta dominada por
        # el target_path rígido del MPC.
        "gain": 0.45,
        "max_m": 0.06,
        "step_max_m": 0.035,
    },
    "intersection": {
        # En sim sin GPS/GT, la odometría puede quedarse "en ruta" mientras
        # el chasis físico se abre hacia el carril paralelo en bifurcaciones
        # (post-mortem run_to282_no_gps_iter2, salida 1247→221). Cuando hay
        # two_line, usamos la cámara con autoridad moderada también en
        # intersección. Single_line sigue ignorado desde pose_estimator.
        "gain": 0.10,
        "max_m": 0.02,
        "step_max_m": 0.008,
    },
    "roundabout": {
        "gain": 0.10,
        "max_m": 0.02,
        "step_max_m": 0.008,
    },
    "parking": {
        "gain": 0.0,
        "max_m": 0.0,
        "step_max_m": 0.0,
    },
}

# Corrección visual adicional hacia la ruta OSM/Lanelet. En la arquitectura nueva
# esta corrección forma parte del pose estimator y queda activa por defecto.
TRACKING_VISUAL_LANE_RELOCALIZATION_ENABLED = True
TRACKING_VISUAL_LANE_RELOCALIZATION_GAIN = 0.15
TRACKING_VISUAL_LANE_RELOCALIZATION_MAX_M = 0.03
TRACKING_VISUAL_LANE_RELOCALIZATION_MIN_RAW_ERROR_M = 0.01
TRACKING_VISUAL_LANE_RELOCALIZATION_MAX_RAW_ERROR_M = 0.25

# Route tracking visual assist: keep the OSM route as the global reference, but
# allow the local MPC target to nudge toward the visually observed lane center
# when the lanelet map and the simulator texture disagree by a few centimetres.
BEHAVIOR_ROUTE_VISUAL_REENTRY_ENABLED = True
BEHAVIOR_ROUTE_VISUAL_REENTRY_PROFILES = {
    "default": {
        "enabled": True,
        "gain_scale": 1.0,
        "max_shift_m": 0.10,
        "fade_distance_m": 0.80,
        "min_error_m": 0.03,
    },
    "lane_keep": {
        "enabled": True,
        "gain_scale": 1.0,
        "max_shift_m": 0.10,
        "fade_distance_m": 0.80,
        "min_error_m": 0.03,
    },
    "intersection": {
        "enabled": True,
        "gain_scale": 0.50,
        "max_shift_m": 0.05,
        "fade_distance_m": 0.50,
        "min_error_m": 0.035,
    },
    "roundabout": {
        "enabled": True,
        "gain_scale": 0.50,
        "max_shift_m": 0.05,
        "fade_distance_m": 0.50,
        "min_error_m": 0.035,
    },
    "parking": {
        "enabled": False,
        "gain_scale": 0.0,
        "max_shift_m": 0.0,
        "fade_distance_m": 0.0,
        "min_error_m": 0.03,
    },
}

# Recovery del matcher/ruta a escala BFMC. En un carril de 35 cm no podemos
# esperar 60-75 cm de error para intentar volver a la ruta.
TRACKING_ROUTE_RECOVERY_ERROR_M = 0.10
TRACKING_ROUTE_GLOBAL_RECOVERY_ERROR_M = 0.14
TRACKING_ROUTE_LANELET_OVERRIDE_ERROR_M = 0.12

# Reanclaje semántico: cuando una señal esperada coincide cerca del próximo
# evento de ruta, el DR puede resetearse a la pose matcheada del mapa.
TRACKING_SEMANTIC_RELOCALIZATION_MAX_DISTANCE_M = 0.45
TRACKING_SEMANTIC_RELOCALIZATION_MAX_MAP_ERROR_M = 0.30
TRACKING_SEMANTIC_RELOCALIZATION_DISTANCE_TOLERANCE_M = 0.25
TRACKING_SEMANTIC_RELOCALIZATION_COOLDOWN_S = 0.75

# Stopline visual con OpenCV:
# 1) se arma solo cuando la ruta espera un stopline cerca,
# 2) detecta una banda horizontal blanca en vista cenital (BEV),
# 3) cuando desaparece despues de haberse visto establemente, threadTracking
#    relocaliza el auto al nodo stopline mas cercano de la ruta activa.
TRACKING_VISUAL_STOPLINE_ENABLED = True
TRACKING_VISUAL_STOPLINE_ARM_DISTANCE_M = 0.85
TRACKING_VISUAL_STOPLINE_MIN_DISTANCE_M = 0.04
TRACKING_VISUAL_STOPLINE_MAX_DISTANCE_M = 0.70
TRACKING_VISUAL_STOPLINE_MIN_WIDTH_RATIO = 0.55
TRACKING_VISUAL_STOPLINE_MIN_BAND_ROWS = 4
TRACKING_VISUAL_STOPLINE_MIN_CONFIDENCE = 0.35
TRACKING_VISUAL_STOPLINE_STABLE_FRAMES = 2
TRACKING_VISUAL_STOPLINE_LOST_FRAMES = 2
TRACKING_VISUAL_STOPLINE_X_MARGIN_LANES = 1.10
TRACKING_VISUAL_STOPLINE_HORIZONTAL_CLOSE_RATIO = 0.12
TRACKING_VISUAL_STOPLINE_ADAPTIVE_BLOCK_SIZE = 31
TRACKING_VISUAL_STOPLINE_ADAPTIVE_C = 7.0
TRACKING_VISUAL_STOPLINE_EVENT_MAX_AGE_S = 0.60
TRACKING_VISUAL_STOPLINE_RELOCALIZATION_COOLDOWN_S = 1.00
TRACKING_VISUAL_STOPLINE_ROUTE_BEHIND_M = 0.25
TRACKING_VISUAL_STOPLINE_ROUTE_AHEAD_M = 0.85
TRACKING_VISUAL_STOPLINE_MAX_MAP_ERROR_M = 0.75
# Ángulo máximo que puede tener la banda detectada respecto de la horizontal en BEV.
# Una stopline real es casi horizontal; si la línea ajustada supera este ángulo
# se descarta como falso positivo (líneas de lane oblicuas, reflejos, etc.).
TRACKING_VISUAL_STOPLINE_MAX_ANGLE_DEG = 20.0
# Máximo de frames consecutivos que la stopline puede ser visible antes de disparar
# pass_event aunque no haya desaparecido. Cubre el caso donde el auto llega a la línea
# y la cámara la sigue viendo porque el auto está encima de ella (missing_streak nunca sube).
# Debe ser > TRACKING_VISUAL_STOPLINE_STABLE_FRAMES. Con 15fps, 6 frames ≈ 0.4s al llegar.
TRACKING_VISUAL_STOPLINE_MAX_VISIBLE_STREAK = 6

TRACKING_SPEED_FEEDBACK_TIMEOUT_S = 0.35
TRACKING_COMMAND_SPEED_FALLBACK_TIMEOUT_S = 0.50
TRACKING_COMMAND_SPEED_FALLBACK_SCALE = 1.0
# True = si falta encoder reciente, el pose estimator puede propagar la pose con
# el último comando de velocidad fresco en lugar de congelar el DR.
TRACKING_COMMAND_SPEED_FALLBACK_ENABLED = (
    _os.environ.get("URT_TRACKING_COMMAND_SPEED_FALLBACK_ENABLED", "1") == "1"
)
TRACKING_STEER_FEEDBACK_TIMEOUT_S = 0.35

# Filtro de encoder portado de urt-ref: si el encoder fresco contradice fuerte
# los últimos comandos de velocidad, se reemplaza por el comando. También clava
# velocidades muy chicas a cero para no integrar ruido cuando el auto está quieto.
TRACKING_ENCODER_FILTER_ENABLED = True
TRACKING_ENCODER_FILTER_WINDOW = 5
TRACKING_ENCODER_FILTER_OUTLIER_DIFF_MPS = 0.15
TRACKING_ENCODER_FILTER_ZERO_EPS_MPS = 0.03
TRACKING_ENCODER_FILTER_STOP_CMD_DIFF_MPS = 0.07

# Graph node attr that represents a physical stopline. Graph guidance is only
# allowed to take authority in this exact node type.
TRACKING_STOPLINE_NODE_ATTR = 7

# Local map-matching weights/search around the current dense waypoint index.
TRACKING_MAP_MATCH_SEARCH_WP = 18
TRACKING_MAP_MATCH_DISTANCE_W = 1.0
TRACKING_MAP_MATCH_HEADING_W = 0.35
TRACKING_SEMANTIC_MATCH_WINDOW_S = 1.0

# Set True to open the OpenCV "Track Navigation" debug window.
# Override with URT_SHOW_PREVIEW=0 if cv2 windows crash (macOS forked child issue).
TRACKING_SHOW_WINDOW = _os.environ.get("URT_SHOW_PREVIEW", "1") == "1"

# TRACKING_DEBUG_LOG: True = escribe temp/tracking_debug.log con posición DR,
# velocidad, yaw, waypoint actual, errores de tracking, etc.
# Útil para diagnosticar por qué el mapa muestra movimiento cuando el auto está quieto.
TRACKING_DEBUG_LOG = True

# TRACKING_USE_PATH_HEADING: True = usa el heading del IMU vs tangente del path
# (como hace el repo de referencia) en lugar del ángulo estimado por detección
# de líneas. Más estable en modo una-sola-línea: evita el spike a ±25° cuando
# se pierde una línea.  False = comportamiento anterior (heading de visión).
TRACKING_USE_PATH_HEADING = True

# Single-line curve priority: if only one lane line is visible inside a curve,
# the steering cannot unwind below this heading-based floor until the car exits
# the curve. This keeps the visible line above graph/MPC hints.
SINGLE_LINE_CURVE_HEADING_GAIN_MULT = 2.5
SINGLE_LINE_CURVE_MIN_STEER_DEG = 2.0
SINGLE_LINE_CURVE_HEADING_STEER_GAIN = 1.0
SINGLE_LINE_CURVE_STEER_HOLD_RATIO = 0.65

# ── Lane mask classification debug log ──────────────────────────────────────
# Set True to write temp/lane_mask_debug.log continuously while the car runs.
# Each line shows: raw YOLO output → after prep → BEV centroid → guidance result → steering.
# This is separate from the AUTO RUN log and is always-on (not tied to klem state).
LANE_MASK_DEBUG_LOG = True

# Wheelbase already defined as MPC_WHEELBASE; kept here as alias for clarity.
TRACKING_WHEELBASE_M = MPC_WHEELBASE  # 0.260 m
TRACKING_REAR_AXLE_TO_CG_M = ACADOS_MPC_L_R  # beta slip-angle model, same as urt-ref
TRACKING_ENCODER_CURVE_SPEED_COMPENSATION = True

# Reset suave/seguro de yaw con carriles rectos visibles, equivalente funcional
# al lane_yaw_reset de urt-ref. No se usa en curva ni con una sola línea.
TRACKING_LANE_YAW_RESET_ENABLED = True
TRACKING_LANE_YAW_RESET_CONSECUTIVE = 2
TRACKING_LANE_YAW_RESET_QUALITY_MIN = 0.80
TRACKING_LANE_YAW_RESET_STRAIGHT_THRESH_DEG = 1.1
TRACKING_LANE_YAW_RESET_MAX_ERROR_DEG = 3.75
TRACKING_LANE_YAW_RESET_COOLDOWN_S = 30.0

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
# CURVE_INNER_LINE_HARD_ESCAPE_MIN_ERROR_M:
#   error físico mínimo (en metros) para permitir el "escape duro" al lado opuesto
#   mientras la curva sigue confirmada. Por debajo de este valor se deja actuar al
#   controlador normal para evitar desarmar la curva por ver sólo la línea interior.
CURVE_INNER_LINE_HARD_ESCAPE_MIN_ERROR_M = 0.09

# ── 8. SUAVIZADO EN TRANSICIONES DE MODO ─────────────────────────────────────
# Al pasar de modo 2-líneas a 1-línea (o cambio de lado), el steering se mezcla
# gradualmente durante N frames para evitar un salto brusco.
# Rango: 0 – 4
#   → 0: sin suavizado (inmediato)
#   → 2: mezcla en 2 frames (valor por defecto)
SINGLE_LINE_BLEND_FRAMES = 2

# ======================== CAMERA ========================
# Tipo de camara:
#   "jetson"   — CSI Jetson via GStreamer/nvarguscamerasrc (produccion en Jetson Nano/Orin)
#   "picamera" — CSI Raspberry Pi via picamera2 (RPi only)
#   "usb"      — USB webcam via OpenCV VideoCapture
#   "zmq"      — frames JPEG entregados por sim_bridge (modo simulador, ver seccion SIMULATOR)
CAMERA_TYPE = "zmq" if _SIM_MODE else "jetson"

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

# Transmitir video de la cámara al dashboard (PyQt5 GUI o Angular legacy).
#
# Costos: JPEG encode y emisión SocketIO binaria.
# Tras la migración a `serialCamera_bin` (bytes JPEG en vez de PNG base64) el
# costo es ~5-10x menor que en la era Angular, así que el default queda en
# True: el GUI PyQt5 lo necesita o no muestra nada en el panel "Driving".
#
# Para correr el auto en competencia totalmente headless (sin operador
# conectado), poner `URT_STREAM_CAMERA=0` antes de lanzar `./run.sh` —
# el thread de cámara dejará de encodear JPEG y `processDashboard` no
# se suscribirá al canal `serialCamera`.
STREAM_CAMERA_TO_DASHBOARD = _os.environ.get("URT_STREAM_CAMERA", "1") == "1"

# Stream del preview remoto. En Jetson la cámara puede capturar a 60 FPS, pero
# el monitor no necesita esa cadencia: bajar FPS/calidad evita colas grandes de
# SocketIO/Qt cuando se usa `./run.sh --monitor <jetson>:5005`.
def _env_float(name, default):
    try:
        return float(_os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name, default):
    try:
        return int(_os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


DASHBOARD_CAMERA_FPS = max(0.0, _env_float("URT_DASHBOARD_CAMERA_FPS", 6.0))
DASHBOARD_CAMERA_JPEG_QUALITY = max(
    35, min(90, _env_int("URT_DASHBOARD_CAMERA_JPEG_QUALITY", 55))
)
# Por default ya no reenviamos el frame base64 legacy junto al binario: el GUI
# PyQt usa `serialCamera_bin` y emitir ambos duplica CPU/ancho de banda.
DASHBOARD_EMIT_LEGACY_CAMERA_BASE64 = (
    _os.environ.get("URT_DASHBOARD_EMIT_LEGACY_CAMERA", "0") == "1"
)

# ===================== DEBUG WINDOWS =====================
# Ventanas de OpenCV para debug visual (requieren monitor/display conectado).
# SHOW_CAMERA_PREVIEW actua como master switch: si es False, ninguna ventana se abre.
# Si es True, puedes elegir cuales abrir individualmente con DEBUG_WINDOWS.
SHOW_CAMERA_PREVIEW = _os.environ.get("URT_SHOW_PREVIEW", "1") == "1"  # off if URT_SHOW_PREVIEW=0

# Ventanas individuales de debug (solo aplican si SHOW_CAMERA_PREVIEW = True)
DEBUG_WINDOWS = {
    "camera_preview":   False,  # Preview directo de la camara (raw frame)
    "final_result":     True,  # Resultado final con lineas detectadas y steering
    "binary_threshold": True,  # Vista del threshold binario
    "canny_edges":      True,  # Vista de bordes Canny
    "control_panel":    False,  # Panel de control con PID, velocidad, steering
    "steering_angle":   True,  # Angulo final de giro (calculado/comandado) en tiempo real
    "ai_local_overlay": True,  # Visualizacion local del modelo de IA (carriles + senales)
    "ai_local_masks":   True,  # Mascaras izquierda/derecha/combinada del modelo local
    "ai_local_signs":   True,  # Detecciones de senales/objetos no-carril del modelo local
}

# ===================== SIGN DETECTION =====================
# Deteccion de senales de trafico embebida en el motor de percepcion local
# (YOLO local TensorRT). Sin servidor remoto.
ENABLE_SIGN_DETECTION = True

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
# best.pt/ONNX funcionan en CPU/MPS/CUDA (dev en Mac). En Linux hardware
# preferimos TensorRT si el engine existe, para no caer a ONNX Runtime CPU en
# Jetson ni disparar auto-installs de onnxruntime-gpu durante el arranque.
# Para regenerar producción TensorRT en Jetson:
#   cd models/lane_segmentation && python build_trt.py  # genera el engine en la misma placa
_LOCAL_AI_TRT_MODEL_PATH = "models/lane_segmentation/Best_weights_reentrenado_416px.engine"
_LOCAL_AI_ONNX_MODEL_PATH = "models/lane_segmentation/Best weights_reentrenado.onnx"
_LOCAL_AI_DEFAULT_MODEL_PATH = _LOCAL_AI_ONNX_MODEL_PATH
if not _SIM_MODE and _sys.platform.startswith("linux"):
    _candidate = _os.path.join(_os.path.dirname(__file__), _LOCAL_AI_TRT_MODEL_PATH)
    if _os.path.isfile(_candidate):
        _LOCAL_AI_DEFAULT_MODEL_PATH = _LOCAL_AI_TRT_MODEL_PATH
LOCAL_AI_MODEL_PATH = _os.environ.get(
    "URT_LOCAL_AI_MODEL_PATH",
    _LOCAL_AI_DEFAULT_MODEL_PATH
)
LOCAL_AI_MIN_CONFIDENCE = 0.35
LOCAL_AI_IMGSZ = 416
LOCAL_AI_DEVICE = "cpu" if _SIM_MODE else "auto"  # ONNX+MPS en Mac no retorna mask proto
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
    # Walk area / pedestrian crossing area
    "walk_area": "walk_area",
    "walk area": "walk_area",
    "zebra_crossing": "walk_area",
    "zebra crossing": "walk_area",
    "yayabolgesi": "walk_area",
}

# Walk area stop behavior
WALK_AREA_STOP_DURATION = 3.0   # seconds to wait after pedestrians clear
WALK_AREA_COOLDOWN      = 10.0  # seconds before a new walk_area stop can trigger
WALK_AREA_MIN_BOX_AREA  = 0.04  # min bbox area (normalized) to trigger stop — filters far detections

# ===================== SIMULATOR (ZMQ BRIDGE) =====================
# Activar el modo simulador setea CAMERA_TYPE="zmq" arriba (lee frames del bridge)
# y MOTOR_OUTPUT="zmq" abajo (publica comandos al bridge en vez de a serial).
# El bridge corre en /Users/luciogarcia/urt-simulator/sim_bridge.py.
ZMQ_CAMERA_ENDPOINT = "tcp://localhost:5575"
ZMQ_CAMERA_TOPIC    = b"frame"

# Salida de comandos motor:
#   "serial" — UART al Nucleo STM32 (produccion, default Jetson)
#   "zmq"    — publish JSON al sim_bridge en ZMQ_MOTOR_ENDPOINT (modo simulador)
MOTOR_OUTPUT        = "zmq" if _SIM_MODE else "serial"

# Puerto de la Nucleo/F401RE. Si queda vacío, processSerialHandler autodetecta
# priorizando /dev/ttyACM* y luego /dev/ttyUSB*. En Jetson conviene fijarlo con
# URT_SERIAL_PORT=/dev/serial/by-id/... para no confundirlo con otros USB serial.
SERIAL_PORT = _os.environ.get("URT_SERIAL_PORT") or None

ZMQ_MOTOR_ENDPOINT  = "tcp://localhost:5576"
ZMQ_MOTOR_TOPIC     = b"cmd"

# Feedback IMU + encoder sintético del sim_bridge (modo "zmq" únicamente).
# El sim_bridge integra el modelo bicicleta del comando que ya está enviando
# a Gazebo y publica yaw/speed/steer en este endpoint a 50 Hz, imitando el
# stream que el `threadRead` lee del Nucleo via UART en hardware real.
# `threadSimFeedback` (sólo se levanta cuando MOTOR_OUTPUT == "zmq") consume
# este canal y mete los mensajes en las mismas IPC queues (CurrentSpeed,
# CurrentSteer, ImuData) que usa el resto del cerebro — así el SafetyGate
# tiene la pose fresca que necesita para salir del fallback (0,0).
ZMQ_FEEDBACK_ENDPOINT = "tcp://localhost:5577"
ZMQ_FEEDBACK_TOPIC    = b"feedback"

# En el auto físico KL es la llave de contacto: el operador la lleva a 30
# (motor encendido) DESDE el dashboard antes de soltar comandos al motor —
# es la red de seguridad que evita arranques accidentales con gente cerca.
# En sim no hay actuador físico que proteger; forzar el slider cada vez
# es solo fricción. Cuando esto está en True (default en sim), threadWrite
# arranca con `engineEnabled=True / KL=30` y obedece SpeedMotor/SteerMotor
# desde el primer tick. Poner en False si necesitás testear la máquina de
# estados de KL en sim.
AUTO_KL_RUN_IN_SIM = bool(_SIM_MODE)

# AUTO_STATE_RUN_IN_SIM: si True, el brain pide automáticamente el modo AUTO al
# state machine ~2 s después del arranque. Equivalente a que el usuario clickee
# el botón AUTO en el dashboard desde el primer momento. Poner en False si
# querés arrancar en DEFAULT y usar el dashboard para disparar manualmente.
AUTO_STATE_RUN_IN_SIM = False  # arrancar siempre en STOP; el operador activa AUTO desde el dashboard
