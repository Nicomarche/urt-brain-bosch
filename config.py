"""
Configuracion general del proyecto URT Brain.
Modifica estos valores para cambiar el comportamiento del auto.
"""

import os as _os, sys as _sys
# Sim mode: defaults ON for macOS (no Jetson HW), OFF for Linux. Override with
# URT_SIM_MODE=1/0. Used downstream to flip CAMERA_TYPE, MOTOR_OUTPUT, and
# disable cv2 GUI windows that crash with the headless pip-installed OpenCV.
_SIM_MODE = _os.environ.get("URT_SIM_MODE", "1" if _sys.platform == "darwin" else "0") == "1"


def _argv_value(*names: str):
    """Read simple ROS-style/config-style CLI overrides from sys.argv."""
    normalized = {str(name).strip().lstrip("-") for name in names if str(name).strip()}
    argv = list(getattr(_sys, "argv", []) or [])
    for idx, arg in enumerate(argv[1:]):
        item = str(arg)
        if ":=" in item:
            key, value = item.split(":=", 1)
            if key.strip().lstrip("-") in normalized:
                return value
        if item.startswith("--") and "=" in item:
            key, value = item[2:].split("=", 1)
            if key.strip() in normalized:
                return value
        if item.startswith("--") and item[2:].strip() in normalized and idx + 2 < len(argv):
            return argv[idx + 2]
    return None


def _setting_raw(*names: str, env: tuple[str, ...] = ()):
    arg_value = _argv_value(*names)
    if arg_value is not None:
        return arg_value
    for key in env:
        if key in _os.environ:
            return _os.environ[key]
    return None


def _setting_bool(default: bool, *names: str, env: tuple[str, ...] = ()) -> bool:
    raw = _setting_raw(*names, env=env)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _setting_float(default, *names: str, env: tuple[str, ...] = ()):
    raw = _setting_raw(*names, env=env)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


_REAL_ARG = _argv_value("real")
if _REAL_ARG is not None:
    _SIM_MODE = str(_REAL_ARG).strip().lower() not in {"1", "true", "yes", "on"}
SIM_SUBMODEL = _setting_bool(True, "subModel", env=("URT_SIM_SUBMODEL",))

# ===================== PHYSICAL DIMENSIONS =====================
# Lane and road geometry (BFMC spec)
# 37 cm alineado con urt-ref (`LANE_WIDTH_PIXEL = 0.37 / METER_PER_PIXEL_X`
# en LaneDetector.hpp:182). Afecta síntesis de single-line (la línea ausente
# se desplaza por LANE_WIDTH/2) y validación de ancho ±25%.
LANE_WIDTH_CM = 37.0           # carril: distancia entre bordes interiores de líneas
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
LANE_VISUAL_TWO_LINE_PRIMARY_MIN_POINTS = 4      # two-line confiable no debe caer a mapa por baja densidad temporal
LANE_VISUAL_PRIMARY_ENABLED = _setting_bool(
    True,
    "lane_visual_primary",
    env=("LANE_VISUAL_PRIMARY_ENABLED", "URT_LANE_VISUAL_PRIMARY"),
)
# Opción C (Paso 6): cuando el path viene de visual, pasarlo al MPC en
# frame body del coche (pose tratada como (0,0,0)) en vez de OSM.
#
# DESHABILITADO por default (False): el `path_optimizer.py` aplica
# transformaciones `_world_to_body_xy` / `_body_to_world_xy` sobre el path
# asumiendo que está en world, y eso rompe el path si llega ya en body
# (queda mixto: tp[0] en body, tp[1..N] en world después del re-sample).
# Con encoder físico (Paso 9a en sim, firmware Nucleo en real) el EKF
# tiene drift mínimo y frame world es seguro como en urt-ref. Si en el
# futuro se quiere desacoplar del EKF, hay que adaptar TODO el
# path_optimizer para detectar el flag y no aplicar transformaciones.
LANE_VISUAL_PATH_FRAME_BODY = False
LANE_VISUAL_SINGLE_LINE_PRIMARY_MIN_QUALITY = 0.6
LANE_VISUAL_SINGLE_LINE_PRIMARY_MIN_STREAK_FRAMES = 6
LANE_VISUAL_SINGLE_LINE_ASSIST_PRIMARY_ENABLED = True
LANE_VISUAL_SINGLE_LINE_ASSIST_PRIMARY_MIN_QUALITY = 0.82
LANE_VISUAL_SINGLE_LINE_TRANSITION_MAX_ERROR_DELTA_M = 0.05
LANE_VISUAL_TWO_LINE_ERROR_MEMORY_MAX_AGE_S = 0.75
LANE_VISUAL_SINGLE_LINE_SIDE_MEMORY_MAX_AGE_S = 0.85
LANE_VISUAL_SINGLE_LINE_MAX_PREFIX_ABS_Y_M = (LANE_WIDTH_CM / 200.0) + 0.08
LANE_VISUAL_SINGLE_LINE_MAX_NEAR_YAW_DEG = 45.0  # ref usa 25° pero el detector activo necesita más tolerancia en curva

# ── Override del rule `route_precision_context:turn_heading_change_*` ───────
# El planner por default rechaza `visual_lane_waypoints` y fuerza
# `route_waypoints` cuando el OSM dice que viene una curva > 25° (regla
# geométrica para detectar intersecciones en el mapa, en
# `src/behavior/scenarios/base.py:_route_precision_context`).
#
# Problema: en algunas curvas suaves del track real (BFMC sim) el OSM tiene
# artifacts de hasta 90° por geometría triangulada del lanelet, mientras la
# cámara ve una curva continua de ~40° con dos líneas perfectas (q=1.0).
# La regla rechaza el visual y el MPC sigue el OSM saturando ±25° en sentido
# OPUESTO a la curva real (campaña manual run_manual_20260511_192422 lo
# confirmó en lanelets 60/70/85).
#
# Override: cuando la visual lane es two_line con quality >= umbral, ignoramos
# el rule de heading_change y dejamos al MPC seguir lo que ve. El check
# semántico (intersection/stopline/crosswalk/parking/roundabout) queda intacto.
# Para deshabilitar el override (volver al comportamiento anterior):
# `LANE_VISUAL_OVERRIDES_HEADING_CHANGE_RULE = False`.
LANE_VISUAL_OVERRIDES_HEADING_CHANGE_RULE = True
LANE_VISUAL_OVERRIDE_HEADING_CHANGE_MIN_QUALITY = 0.95
# Umbral mínimo de cambio de heading visual (rad) para activar la comprobación
# de conflicto visual↔ruta. Si el path visual cambia menos de este ángulo en
# sentido opuesto a la ruta, se ignora (es ruido de curvatura, no un fork).
# 0.174 rad ≈ 10°.
LANE_VISUAL_ROUTE_CONFLICT_MIN_RAD = 0.174
# Si el heading del robot difiere del waypoint de ruta en más de este umbral (20°),
# se rechaza la visual: indica que el robot está entrando a una bifurcación donde
# la cámara ve ambas ramas y calcula un centro incorrecto (ej. ll90 fork → 23°).
# El chequeo persiste hasta que el robot gire para alinearse con la ruta.
LANE_VISUAL_ROUTE_ALIGNMENT_MAX_RAD = 0.349  # 20°

# ── Deadband del error lateral del lane observer ────────────────────────────
# Si la cámara reporta |line_center_offset_m| < deadband, lo zeroizamos antes
# de que llegue al MPC. Sin esto, pure-pursuit con lookahead corto (~0.24m
# a 0.2 m/s) amplifica offsets pequeños a alpha grande (5cm → ~12° alpha)
# y satura el steer a 16°. El operador humano tolera offsets pequeños y solo
# corrige cuando se aleja >5cm — el deadband replica esa tolerancia.
#
# Calibrado iterativamente contra el manejo manual del operador:
#   - run_manual_20260511_193516 lanelet 138: visual offset ~+1.9cm, manual 0°,
#     MPC pedía +15° → deadband 5cm solucionó el caso (Δp50 16°→7.6°)
#   - run_manual_20260511_194608 lanelet 995: visual offset sostenido +11cm,
#     manual 0°, MPC pedía +16° (cap) → 5cm no alcanzó.
#
# El operador tolera offsets de hasta ~10-11cm sin corregir (su cross-track
# p90 en manual fue 17cm). Subimos el deadband a 10cm para matchear esa
# tolerancia. Beyond 10cm el MPC vuelve a comportarse normal — corrige
# offsets reales que el operador también corregiría.
#
# Trade-off: con 10cm el MPC va a ignorar oscilaciones de hasta una banda
# completa entre líneas (carril ~37cm, half=18cm). Eso es lo que vos hacés.
#
# Para desactivar: `LANE_CONTROL_ERROR_DEADBAND_M = 0.0`.
LANE_CONTROL_ERROR_DEADBAND_M = 0.12
LANE_VISUAL_SINGLE_LINE_REQUIRE_MAP_MATCH_WITH_ROUTE = False  # estilo ref: visual gana sin pedir map_match
LANE_VISUAL_PRIMARY_PRECISION_ZONE_M = 0.45

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
# BFMC robot: 0.258 m (igual que mpc_acados2_clean.py del repo de referencia).
MPC_WHEELBASE = 0.258

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
# USE_ACADOS_MPC: True = activa MPC completo (Acados), False = usa lateral MPC.
# Si Acados no está instalado o el solver no fue generado, cae al MPC lateral.
USE_ACADOS_MPC = False

# Horizonte de predicción.  N × T = tiempo total de anticipación.
# N=30, T=0.05 → 1.5s de horizonte.
ACADOS_MPC_N = 30
ACADOS_MPC_T = 0.05

# Velocidad de referencia por defecto [m/s].  El MPC optimiza alrededor de
# este valor.  En competencia ajustar según la zona (highway vs curva).
# 0.32 m/s alineado con urt-ref (`v_ref = 0.32` en PathManager.h:31).
# Si el coche tiene problemas de tracking, bajar gradualmente (0.20-0.32).
ACADOS_MPC_V_REF = 0.32

# Modelo del vehículo.
ACADOS_MPC_WHEELBASE = 0.258      # distancia entre ejes [m]
ACADOS_MPC_L_R = 0.103            # eje trasero a CG [m]
ACADOS_MPC_L_F = 0.155            # eje delantero a CG [m]

# Límites de control.
ACADOS_MPC_V_MAX = 0.22           # velocidad máxima [m/s]. Shadow analysis: iguala la vel del conductor
                                  # (~0.20 m/s) para que ACADOS planifique a la velocidad real y no
                                  # sobrecompense el steer asumiendo que va al doble de rápido.
ACADOS_MPC_V_MIN = -0.50          # velocidad mínima [m/s] (reversa)
ACADOS_MPC_DELTA_MAX_DEG = 25.0   # steering máximo [°]

# Pesos del costo (NONLINEAR_LS).
#   Q: penaliza desviación de la trayectoria de referencia (x, y, yaw).
#   R: penaliza esfuerzo de control (v, delta).
#   S: penaliza tasa de cambio de controles (Δv, Δδ).
ACADOS_MPC_X_COST = 2.0
ACADOS_MPC_Y_COST = 2.0
ACADOS_MPC_YAW_COST = 0.5
ACADOS_MPC_V_COST = 1.0
ACADOS_MPC_STEER_COST = 0.5   # >0 evita bang-bang ±25°; era 0.3
ACADOS_MPC_DELTA_V_COST = 4.0   # sube rampa de arranque; era 1.5 (llegaba a v_ref demasiado rápido)
ACADOS_MPC_DELTA_STEER_COST = 1.5  # compromiso: era 0.75 (oscilación ±25°), subió a 3.0 (muy sticky en ll50/ll996), ahora 1.5

# Zona muerta de salida del MPC completo [°].
ACADOS_MPC_OUTPUT_DEADBAND_DEG = 0.5

# USE_ACADOS_SPEED: True = usar velocidad del MPC para el motor.
# False = solo usar steering del MPC, mantener control de velocidad heurístico.
USE_ACADOS_SPEED = False

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
TRACKING_BG_RASTER = _os.path.join(
    TRACK_MAP_DIR, "track.png" if _SIM_MODE else "track.jpg"
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

# Default localization profile used by Auto.
#
# Auto keeps the classic behavior: use encoder/CurrentSpeed when available and
# accept GPS after validation. The UI Odometry mode overrides these at runtime:
# no GPS and no encoder, advancing from SpeedMotor command + IMU yaw.
#
# Overrides:
#   URT_USE_ENCODER=0 / use_encoder:=false
#   URT_USE_GPS=0     / use_gps:=false
# urt-ref `origin/main` usa `use_encoder default="true"` en
# controller.launch:16. El encoder físico calibrado alimenta el EKF.
TRACKING_USE_ENCODER = _setting_bool(
    True,
    "use_encoder",
    env=("URT_USE_ENCODER", "TRACKING_USE_ENCODER"),
)
# GPS habilitado: el sim_bridge entrega GPS locsys tcp://4691
# (sim_bridge.py:597). Sin GPS, el dead reckoning del brain diverge 14m del
# ground truth real del sim (iter19 pose_drift_max=13.97m). Con GPS la
# pose se mantiene a 0.3m del GT (iter20 mean=0.288m). El `map_match_error`
# aparente bueno sin GPS era engañoso — comparaba contra el mapa interno
# del brain, no contra el mundo real.
TRACKING_USE_GPS = _setting_bool(
    True,
    "use_gps",
    env=("URT_USE_GPS", "TRACKING_USE_GPS", "GPS_ENABLED"),
)

# Initial pose override for no-GPS starts. Values are in the same map/world
# frame used by the OSM route graph; yaw is radians. These mirror the ROS launch
# style names for convenience:
#   x0:=3.83 y0:=0.376 yaw0:=3.14159
TRACKING_START_X = _setting_float(None, "x0", env=("URT_X0", "URT_START_X"))
TRACKING_START_Y = _setting_float(None, "y0", env=("URT_Y0", "URT_START_Y"))
TRACKING_START_YAW_RAD = _setting_float(
    None,
    "yaw0",
    env=("URT_YAW0", "URT_START_YAW_RAD"),
)
TRACKING_START_YAW_DEG = _setting_float(
    None,
    "yaw0_deg",
    env=("URT_YAW0_DEG", "URT_START_YAW_DEG"),
)

# Speed scale applied to the encoder reading before dead-reckoning integration.
# 1.0 = use encoder speed as-is.  If the virtual car advances faster than the
# real one, reduce this value (e.g. 0.85).  If it lags behind, increase it.
# Typical range: 0.7 – 1.0 (wheel slip, encoder calibration, actuator lag).
TRACKING_DR_SPEED_SCALE = 1.0

# Speed scale applied only when odometry uses the commanded SpeedMotor value.
# Calibration routine:
#   1) command 0.20 m/s for 5 s, measure real distance
#   2) command 0.25 m/s for 5 s, measure real distance
#   3) command 0.30 m/s for 5 s, measure real distance
#   factor = mean(real_distance / (commanded_speed * 5.0))
# Then set URT_COMMAND_SPEED_CALIBRATION_SCALE=<factor>.
TRACKING_COMMAND_SPEED_CALIBRATION_SCALE = _setting_float(
    1.0,
    "command_speed_scale",
    "velocity_command_scale",
    env=(
        "URT_COMMAND_SPEED_CALIBRATION_SCALE",
        "URT_VELOCITY_COMMAND_SCALE",
        "TRACKING_COMMAND_SPEED_CALIBRATION_SCALE",
    ),
)

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
#   Real hardware (BNO055): el firmware Nucleo envía yaw CCW-positivo
#   (convención right-hand rule), así que un giro a la derecha produce
#   yaw_deg < 0 → necesitamos _IMU_YAW_SIGN = -1 para que yaw_raw_rad
#   sea positivo (= crece hacia el sur en el frame OSM, que es el +π/2 de la
#   trayectoria CW-desde-el-este que usa el dead reckoning).
#
#   Sim (sim_bridge feedback_yaw_sign=1.0): el simulador integra
#   ω = +v·tan(δ)/WB, por lo que un giro a la derecha hace crecer
#   yaw_deg → la negación del hardware daría el signo incorrecto.
#   Con _IMU_YAW_SIGN = +1 la señal del sim coincide con la convención
#   del dead reckoning sin invertir.
TRACKING_IMU_YAW_SIGN = 1.0 if _SIM_MODE else -1.0

# DEPRECATED: el yaw de sim ya llega en `brain_map` desde `sim_bridge`, así que
# no hace falta una corrección extra en el brain. Se conserva en config sólo
# para compatibilidad con checkouts viejos.
SIM_IMU_YAW_OFFSET_DEG = 0.0

# Forzar PurePursuit en lugar de AcadosMPC. En simulador las curvas cerradas
# del mapa actual se siguen mucho mejor con PurePursuit (evita que Acados
# sature el steering contra una referencia cartesiana difícil). En real
# hardware mantenemos Acados como default, pero se puede overridear sin tocar
# código:
#   URT_FORCE_PURE_PURSUIT=1  fuerza PurePursuit
#   URT_FORCE_PURE_PURSUIT=0  fuerza Acados
_FORCE_PP_ENV = _os.environ.get("URT_FORCE_PURE_PURSUIT")
FORCE_PURE_PURSUIT = (
    _SIM_MODE
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
# (TCP:5000) para obtener la IP del locsys device, luego conecta al locsys
# device (TCP:4691) y recibe {"x": float, "y": float}\n a 1 Hz.
#
# En simulación: sim_bridge expone el servidor locsys en localhost:4691
# y usa el ground truth de Gazebo transformado al frame del mapa OSM.
#
# El cliente (threadLocSys) envía el fix como Localisation IPC con
# world_x/world_y (coordenadas OSM directas, sin conversión de imagen).

LOCSYS_PORT          = 4691
LOCSYS_HOST_COMP     = "192.168.50.11"  # IP locsys device en competencia BFMC
TRAFFIC_COMM_HOST    = "192.168.1.1"    # TrafficCommunicationServer en competencia
TRAFFIC_COMM_PORT    = 5000
LOCSYS_DEVICE_ID     = 1               # ID del coche en la red BFMC (1–4)

SIM_LOCSYS_HOST      = "localhost"
SIM_LOCSYS_PORT      = 4691

GPS_ENABLED          = TRACKING_USE_GPS # habilitar threadLocSys (sim + competencia)
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
# tamaño del horizonte se mide en steps de 50 ms.
BEHAVIOR_DT_S = 0.05
BEHAVIOR_HORIZON_N = 40  # must match N_horizon in c_generated_code/acados_ocp_bfmc_bicycle.json
                         # 40 × 0.05 = 2.0s preview (era 30=1.5s) — alineado con urt-ref

# Límites competitivos de velocidad. 0.0 sigue siendo válido para STOP; cuando
# el auto está en movimiento no debe mandar menos de 20 cm/s.
BEHAVIOR_MIN_SPEED_MPS = 0.20       # velocidad mínima de movimiento = 20 cm/s

# Velocidad nominal de lane_keep "limpio" (sin signs, sin regulators).
BEHAVIOR_NOMINAL_SPEED_MPS = 0.20   # target base = mínimo competitivo.
                                    # El planner sigue siendo la fuente única
                                    # de verdad para el perfil de velocidad.

# Hard cap absoluto. Aplicado por velocity_overlay al final, ningún
# scenario puede emitir velocidades por encima.
BEHAVIOR_MAX_SPEED_MPS = 0.40       # cap absoluto de AUTO = 40 cm/s

# Geometría y containment lateral del planner. Los valores están escalados
# para BFMC (carril 35 cm, vehículo ~19 cm de ancho) y se usan para exigir
# que la referencia publicada al MPC permanezca dentro del corredor del mapa.
#
# === BYPASS DEL PATH_OPTIMIZER ESTILO REF ===
# urt-ref pasa el path Dijkstra/spline directo al MPC sin smoothing/blending/
# containment. Activar para replicar ese flujo (recomendado para alinear con
# ref main). Cuando True, `PathOptimizer.optimize` retorna el raw_path
# resampleado a horizon_n+1 sin más procesamiento si `path_source="route_waypoints"`.
BEHAVIOR_PATH_OPTIMIZER_BYPASS_ROUTE = _setting_bool(
    False,
    "bypass_path_optimizer",
    env=("URT_BYPASS_PATH_OPTIMIZER", "BEHAVIOR_PATH_OPTIMIZER_BYPASS_ROUTE"),
)

BEHAVIOR_VEHICLE_WIDTH_M = 0.19
BEHAVIOR_CONTAINMENT_CLEARANCE_M = 0.01
BEHAVIOR_CONTAINMENT_WARN_ERROR_M = 0.05
BEHAVIOR_CONTAINMENT_CRAWL_ERROR_M = 0.07
BEHAVIOR_CONTAINMENT_CRAWL_SPEED_MPS = 0.20

# Stuck-recovery del LaneContainmentRule. Si el robot lleva varios ticks
# en crawl (4 cm/s) sin que el error lateral decrezca, sube temporalmente
# el cap a RECOVERY_SPEED_MPS para que pueda maniobrar y volver al carril.
# A 4 cm/s con steering al máximo el yaw rate alcanza pero la traslación
# es tan lenta que el centerline del próximo lanelet rota más rápido que
# el ego avanza → loop estable de crawl. STUCK_TICKS=40 ≈ 2 s a 20 Hz.
BEHAVIOR_CONTAINMENT_STUCK_TICKS = 40
BEHAVIOR_CONTAINMENT_RECOVERY_SPEED_MPS = 0.20

# Curvature-anticipating apex offset en la referencia del controlador.
# Desplaza cada waypoint del path de referencia hacia el interior de la
# curva próxima para que el controlador "quiera" estar en la posición apex
# en lugar de la centerline pura. APEX_GAIN=0 → sin offset (baseline).
BEHAVIOR_APEX_GAIN = 0.0
BEHAVIOR_APEX_LOOKAHEAD_M = 1.5
# Curvatura de referencia [m]: al invertirla, da el radio al que el offset
# llega al 100% del gain. 0.4 → radio ~40 cm (curva pronunciada BFMC).
BEHAVIOR_APEX_SENSITIVITY_M = 0.4

# Aceleración máxima del ramp de velocidad en el BehaviorPlanner [m/s²].
# 0.25 m/s² → llega a 0.15 m/s en ~0.6 s (12 ticks a 20 Hz).
BEHAVIOR_ACCEL_MPS2 = 0.25

# Rate limiter de velocidad en el output del MotionController [m/s²].
# Garantiza arranque gradual independientemente del solver.
# 0.25 m/s² → 0→0.15 m/s en ~0.6 s (12 ticks).
BEHAVIOR_MAX_SPEED_RATE_MPS2 = 0.25

# Velocidad máxima de cambio del ángulo de steering [°/s].
# A 20 Hz (dt=0.05 s) → 60°/s = 3°/tick → 0→25° en ~12 ticks (0.6 s).
# Con 200°/s y dirección incorrecta, el actuador se bloquea en el extremo
# equivocado en 2-3 ticks antes de que el pure-pursuit se auto-corrija.
# Con 60°/s hay ~12 ticks de margen → el robot avanza y corrige la dirección.
BEHAVIOR_MAX_STEER_RATE_DEG_S = 100.0

# Cuando el path primario viene de la cámara, el control lateral debe entrar
# como corrección fina, no como giro máximo hacia el centro visual instantáneo.
BEHAVIOR_VISUAL_PRIMARY_MAX_STEER_DEG = 10.0
BEHAVIOR_VISUAL_PRIMARY_RECOVERY_MAX_STEER_DEG = 16.0
BEHAVIOR_VISUAL_PRIMARY_RECOVERY_ERROR_START_M = 0.10
BEHAVIOR_VISUAL_PRIMARY_RECOVERY_ERROR_FULL_M = 0.15
BEHAVIOR_VISUAL_PRIMARY_CURVE_MAX_STEER_DEG = 16.0
BEHAVIOR_VISUAL_PRIMARY_CURVE_STEER_FULL_DEG = 16.0
BEHAVIOR_VISUAL_PRIMARY_CURVE_MAX_STEER_RATE_DEG_S = 100.0
BEHAVIOR_VISUAL_PRIMARY_MAX_STEER_RATE_DEG_S = 35.0
BEHAVIOR_VISUAL_PRIMARY_RECOVERY_MAX_STEER_RATE_DEG_S = 60.0
BEHAVIOR_LINE_LOSS_STRAIGHT_HOLD_ROUTE_ESCAPE_MIN_TICKS = 24
BEHAVIOR_LINE_LOSS_ROUTE_TURN_MAX_MAP_ERROR_M = 0.25
BEHAVIOR_LINE_LOSS_PREVIOUS_VISUAL_HOLD_ENABLED = True
BEHAVIOR_VISUAL_TWO_LINE_ABSURD_CORRIDOR_VIOLATION_M = 20.0
BEHAVIOR_VISUAL_ABSURD_CORRIDOR_VIOLATION_M = 0.50
BEHAVIOR_VISUAL_PREVIOUS_PATH_MAX_SAMPLE1_GAP_M = 0.08

# Lookahead del PurePursuitSolver (fallback sin acados).
# El kink geométrico al inicio de cada lanelet tiene un "crossing point" a
# distancia variable: lanelet 120 ≈ 0.25 m, lanelets 70/90/125 > 0.40 m.
# Con 0.70 m el goal siempre queda más allá del cruce → dirección correcta.
# Sólo vinculante a v < 0.636 m/s (luego toma L_d = 1.1·v dinámica).
BEHAVIOR_MIN_LOOKAHEAD_M = 0.70
BEHAVIOR_LOOKAHEAD_GAIN_S = 1.1

# Tasa de actualización del thread (s). Con pause=0.05 corremos a ~20 Hz,
# emparejando dt del MPC.
BEHAVIOR_THREAD_PAUSE_S = 0.05

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
# 0.5s alineado con `lane_localization_cooldown` del ref
# (urt-ref/.../tunable_params.yaml:9). Antes era 0.75s.
TRACKING_SEMANTIC_RELOCALIZATION_COOLDOWN_S = 0.5

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
TRACKING_COMMAND_SPEED_FALLBACK_SCALE = TRACKING_COMMAND_SPEED_CALIBRATION_SCALE
# True = si falta encoder reciente, el pose estimator puede propagar la pose con
# el último comando de velocidad fresco en lugar de congelar el DR.
TRACKING_COMMAND_SPEED_FALLBACK_ENABLED = True
TRACKING_STEER_FEEDBACK_TIMEOUT_S = 0.35

# GPS authority. "hard" reset rompe el dead reckoning (coche se queda
# anclado). "init_recovery_soft" aplica corrección incremental con gain.
LOCALIZATION_GPS_AUTHORITY = (
    _setting_raw(
        "localization_gps_authority",
        "gps_authority",
        env=("LOCALIZATION_GPS_AUTHORITY", "URT_GPS_AUTHORITY"),
    )
    or "init_recovery_soft"
)
# GPS validation: sim_bridge entrega GPS a 10Hz con noise ±15cm (radio).
# Thresholds calibrados al noise real:
#   - OUTLIER_DISTANCE_M = 2×noise para descartar samples atípicos
#   - MAX_EXPECTED_ERROR_M = 3×noise para distinguir "GPS off" vs noise normal
# El SAMPLE_WINDOW=5 median ya promedia 5 fixes para filtrar ruido a ±5cm.
TRACKING_GPS_VALIDATION_ENABLED = True
TRACKING_GPS_SAMPLE_WINDOW = 5
TRACKING_GPS_MIN_SAMPLES = 3
TRACKING_GPS_ZERO_EPS_M = 1e-3
TRACKING_GPS_OUTLIER_DISTANCE_M = 0.30  # 2×noise (15cm)
TRACKING_GPS_MAX_JUMP_M = 1.00  # ~7×noise, suficiente para movimiento real
TRACKING_GPS_MAX_EXPECTED_ERROR_M = 0.45  # 3×noise (umbral discrim. real drift)
TRACKING_GPS_SOFT_GAIN = 0.50  # aplica 50% del error medio-filtrado
TRACKING_GPS_SOFT_MAX_STEP_M = 0.050  # 5cm/paso para corregir drift de yaw
TRACKING_GPS_RECOVERY_MAX_STEP_M = 0.100
TRACKING_GPS_RECOVERY_ERROR_M = 0.30
TRACKING_GPS_SOFT_LATERAL_GAIN = 0.40
TRACKING_GPS_VISUAL_LATERAL_BLOCK_M = 0.05
TRACKING_GPS_VISUAL_PROTECT_QUALITY = 0.55

# Mini-YabLoc 2D: compara la curva visual del carril contra la ventana de ruta
# activa. No aplica offsets fijos por escenario; solo produce una medición con
# confianza para telemetría/gating.
VISUAL_MAP_MATCH_ENABLED = True
VISUAL_MAP_MATCH_MIN_CONFIDENCE = 0.45
VISUAL_MAP_MATCH_MAX_LATERAL_ERROR_M = float(LANE_WIDTH_CM) / 200.0
VISUAL_MAP_MATCH_MAX_YAW_ERROR_DEG = 6.0
VISUAL_MAP_MATCH_MAX_SAMPLE_YAW_ERROR_DEG = 8.0
VISUAL_MAP_MATCH_PRIMARY_MAX_YAW_ERROR_DEG = 15.0
VISUAL_MAP_MATCH_PRIMARY_MAX_SAMPLE_YAW_ERROR_DEG = 20.0
VISUAL_MAP_MATCH_CORRECTION_GAIN = 0.20
VISUAL_MAP_MATCH_CORRECTION_STEP_MAX_M = 0.008
VISUAL_MAP_MATCH_CORRECTION_COOLDOWN_S = 0.10
VISUAL_MAP_MATCH_YAW_CORRECTION_GAIN = 0.10
VISUAL_MAP_MATCH_YAW_CORRECTION_STEP_MAX_DEG = 0.50

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
TRACKING_WHEELBASE_M = MPC_WHEELBASE  # 0.258 m

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
# Costos: JPEG encode (~70% calidad, ~50KB/frame) y emisión SocketIO binaria.
# Tras la migración a `serialCamera_bin` (bytes JPEG en vez de PNG base64) el
# costo es ~5-10x menor que en la era Angular, así que el default queda en
# True: el GUI PyQt5 lo necesita o no muestra nada en el panel "Driving".
#
# Para correr el auto en competencia totalmente headless (sin operador
# conectado), poner `URT_STREAM_CAMERA=0` antes de lanzar `./run.sh` —
# el thread de cámara dejará de encodear JPEG y `processDashboard` no
# se suscribirá al canal `serialCamera`.
STREAM_CAMERA_TO_DASHBOARD = _os.environ.get("URT_STREAM_CAMERA", "1") == "1"

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
# best.pt funciona en CPU/MPS/CUDA (dev en Mac, Jetson con GPU).
# Para producción TensorRT en Jetson Nano:
#   export URT_LOCAL_AI_MODEL_PATH=models/lane_segmentation/Best_weights_reentrenado_416px.engine
#   cd models/lane_segmentation && python build_trt.py  # genera el engine en la misma placa
LOCAL_AI_MODEL_PATH = _os.environ.get(
    "URT_LOCAL_AI_MODEL_PATH",
    "models/lane_segmentation/Best weights_reentrenado.onnx"
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
# Guarda post-proceso: si la segmentacion de carril se pega a una linea
# transversal/horizontal, borrar solo ese tramo antes de calcular geometria.
LOCAL_AI_DISCARD_HORIZONTAL_LANE_SEGMENTS = True
LOCAL_AI_HORIZONTAL_SEGMENT_MAX_ANGLE_DEG = 22.0
LOCAL_AI_HORIZONTAL_SEGMENT_FAR_MAX_ANGLE_DEG = 40.0
LOCAL_AI_HORIZONTAL_SEGMENT_FAR_Y_MAX_RATIO = 0.60
LOCAL_AI_HORIZONTAL_SEGMENT_MIN_LENGTH_RATIO = 0.12
LOCAL_AI_HORIZONTAL_SEGMENT_ERASE_THICKNESS_PX = 11
LOCAL_AI_REJECT_CROSSING_LANE_PAIRS = True
LOCAL_AI_CROSSING_LANE_MIN_CROSS_PX = 5.0
LOCAL_AI_CROSSING_LANE_Y_MAX_RATIO = 0.70
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
