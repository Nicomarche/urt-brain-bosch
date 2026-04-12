#!/usr/bin/env python3
"""
calibrate_dr.py — Calibración de Dead Reckoning (BFMC TC-04)

Parsea el log de tracking y calcula el valor correcto de TRACKING_STEER_GAIN_DR
usando dos métodos complementarios.

══════════════════════════════════════════════════════════════════════
MÉTODO 2 — Tasa de giro IMU  ← RECOMENDADO (necesita solo 3-5 s de arco)
══════════════════════════════════════════════════════════════════════

PROCEDIMIENTO
─────────────
1. Activar el log:  TRACKING_DEBUG_LOG = True  en config.py
2. Poner el auto en MANUAL en espacio abierto (≥ 1.5 m de radio libre)
3. Aplicar steer constante (ej: 25° derecha) ANTES de arrancar
4. Acelerar hasta 25-30 cm/s  (el encoder solo registra a partir de esa vel.)
5. Mantener velocidad y steer CONSTANTES durante 3-5 segundos
6. Detener el auto
7. Ejecutar:   python tools/calibrate_dr.py [temp/tracking_debug.txt]

No es necesaria una vuelta completa. Con 30-60° de arco a velocidad
constante donde el encoder esté activo ya alcanza.

══════════════════════════════════════════════════════════════════════
MÉTODO 1 — Ajuste de círculo (necesita ≥360° de arco completo)
══════════════════════════════════════════════════════════════════════

PROCEDIMIENTO
─────────────
1-3. Igual que arriba
4.  Hacer al menos 1 vuelta completa (360°) a 25-30 cm/s
5-7. Igual que arriba

NOTA SOBRE EL MÉTODO
────────────────────
El DR integra la posición usando el yaw del IMU (cuando está disponible),
así que las posiciones (x, y) del log reflejan la trayectoria real del auto.
Ajustando un círculo a esas posiciones obtenemos el radio real de giro.

  gain = atan(L / r_real) / steer_comando_rad
  donde L = wheelbase, r_real = radio ajustado al log.
"""

import re
import sys
import math
import argparse
import copy

# ── Regex para parsear el log de threadTracking ─────────────────────────────
# Formato de línea:
#   F000010 | speed_raw=... spd=15.0cm/s src=encoder ... steer=25.0° | raw=(x,y,yaw°) ... dt=20ms
_LOG_RE = re.compile(
    r'F(?P<frame>\d+)\s*\|'
    r'.*?spd=(?P<spd>[+-]?[\d.]+)cm/s\s+src=(?P<src>\S+)'
    r'.*?steer=(?P<steer>[+-]?[\d.]+)°'
    r'.*?raw=\((?P<x>[+-]?[\d.]+),(?P<y>[+-]?[\d.]+),(?P<yaw>[+-]?[\d.]+)°\)'
    r'.*?dt=(?P<dt>[\d.]+)ms'
)

_DEFAULT_WHEELBASE = 0.260   # metros — igual que TRACKING_WHEELBASE_M


# ── Parseo del log ────────────────────────────────────────────────────────────
def parse_log(path: str) -> list:
    """Lee TODAS las entradas del log (sin filtro de velocidad)."""
    entries = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = _LOG_RE.search(line)
            if not m:
                continue
            entries.append({
                'frame':   int(m.group('frame')),
                'spd_cms': float(m.group('spd')),
                'src':     m.group('src'),
                'steer':   float(m.group('steer')),   # grados, con signo
                'x':       float(m.group('x')),       # metros
                'y':       float(m.group('y')),       # metros
                'yaw':     float(m.group('yaw')),     # grados
                'dt_s':    float(m.group('dt')) / 1000.0,
            })
    return entries


def filter_moving(entries: list, min_speed_cms: float = 3.0) -> list:
    return [e for e in entries if abs(e['spd_cms']) >= min_speed_cms]


# ── Ajuste de círculo (mínimos cuadrados algebraicos, sin numpy) ─────────────
def _solve3(A: list, b: list) -> list:
    """Eliminación gaussiana para sistema 3×3."""
    A = copy.deepcopy(A)
    b = list(b)
    n = 3
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(A[r][col]))
        A[col], A[pivot] = A[pivot], A[col]
        b[col], b[pivot] = b[pivot], b[col]
        if abs(A[col][col]) < 1e-14:
            raise ZeroDivisionError("Matriz singular — trayectoria casi recta")
        for row in range(col + 1, n):
            f = A[row][col] / A[col][col]
            for k in range(col, n):
                A[row][k] -= f * A[col][k]
            b[row] -= f * b[col]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]
    return x


def fit_circle(xs: list, ys: list) -> tuple:
    """
    Ajuste algebraico de círculo por mínimos cuadrados.
    Retorna (cx, cy, radio, rms_error_m).
    """
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    u = [x - mx for x in xs]
    v = [y - my for y in ys]

    Suu  = sum(a**2       for a    in u)
    Svv  = sum(b**2       for b    in v)
    Suv  = sum(a*b        for a,b  in zip(u,v))
    Su   = sum(u)
    Sv   = sum(v)
    Suuu = sum(a**3       for a    in u)
    Svvv = sum(b**3       for b    in v)
    Suvv = sum(a*b**2     for a,b  in zip(u,v))
    Suuv = sum(a**2*b     for a,b  in zip(u,v))

    A = [[Suu, Suv, Su],
         [Suv, Svv, Sv],
         [Su,  Sv,  n ]]
    b_vec = [0.5*(Suuu + Suvv),
             0.5*(Svvv + Suuv),
             0.5*(Suu  + Svv )]

    sol = _solve3(A, b_vec)
    a_c, c_c, d_c = sol

    cx = a_c + mx
    cy = c_c + my
    r  = math.sqrt(a_c**2 + c_c**2 + d_c +
                   mx**2 + my**2 - 2*mx*a_c - 2*my*c_c)

    dists = [math.sqrt((xi - cx)**2 + (yi - cy)**2) for xi, yi in zip(xs, ys)]
    rms = math.sqrt(sum((d - r)**2 for d in dists) / n)
    return cx, cy, r, rms


# ── Rotación total de heading en el log ──────────────────────────────────────
def total_heading_deg(entries: list) -> float:
    """Suma de cambios absolutos de yaw entre entradas consecutivas."""
    total = 0.0
    for i in range(1, len(entries)):
        delta = entries[i]['yaw'] - entries[i-1]['yaw']
        while delta >  180: delta -= 360
        while delta < -180: delta += 360
        total += abs(delta)
    return total


# ── Método 2: Tasa de giro IMU ───────────────────────────────────────────────
def yaw_rate_calibration(entries: list, wheelbase: float,
                         min_steer_deg: float = 5.0,
                         min_speed_cms: float = 5.0,
                         min_omega_deg_s: float = 1.0) -> list:
    """
    Método 2: radio de giro instantáneo a partir de velocidad (encoder) y
    tasa de cambio de yaw (IMU integrado en DR).

    Para cada par consecutivo de entradas donde ambas tienen src=encoder:
      - Calcula el intervalo de tiempo real entre las dos entradas usando
        la diferencia de frame × dt_por_frame.
      - Calcula omega = |Δyaw| / dt_intervalo
      - Calcula R = v / omega
      - Calcula gain = atan(WB / R) / steer_cmd_rad

    Retorna lista de estimaciones individuales.
    """
    estimates = []
    for i in range(len(entries) - 1):
        e0 = entries[i]
        e1 = entries[i + 1]

        # Ambas entradas deben tener velocidad del encoder
        if e0['src'] != 'encoder' or e1['src'] != 'encoder':
            continue

        v0_cms = abs(e0['spd_cms'])
        v1_cms = abs(e1['spd_cms'])
        if v0_cms < min_speed_cms or v1_cms < min_speed_cms:
            continue

        steer_deg = abs(e0['steer'])
        if steer_deg < min_steer_deg:
            continue

        # Intervalo de tiempo entre las dos entradas del log.
        # Cada entrada del log es cada `log_every` ciclos (típicamente 10).
        # La diferencia de frame × dt_por_frame da el intervalo real.
        frame_diff = e1['frame'] - e0['frame']
        if frame_diff <= 0:
            continue
        dt_interval_s = frame_diff * e0['dt_s']   # ≈ 10 × 20ms = 200ms

        # Cambio de yaw en ese intervalo (el yaw en el log ya incluye IMU)
        delta_yaw_deg = e1['yaw'] - e0['yaw']
        while delta_yaw_deg >  180: delta_yaw_deg -= 360
        while delta_yaw_deg < -180: delta_yaw_deg += 360

        omega_deg_s = abs(delta_yaw_deg) / dt_interval_s
        if omega_deg_s < min_omega_deg_s:
            continue   # rotación insignificante, probablemente ruido

        # Radio de giro real
        v_mps = (v0_cms + v1_cms) / 2.0 / 100.0
        omega_rad_s = math.radians(omega_deg_s)
        R = v_mps / omega_rad_s

        if R <= 0.05 or R > 15.0:   # sanity check
            continue

        steer_rad = math.radians(steer_deg)
        effective_angle_rad = math.atan(wheelbase / R)
        gain = effective_angle_rad / steer_rad

        estimates.append({
            'gain':         gain,
            'R':            R,
            'steer_deg':    steer_deg,
            'v_cms':        v_mps * 100.0,
            'omega_deg_s':  omega_deg_s,
            'delta_yaw':    abs(delta_yaw_deg),
        })

    return estimates


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('logfile', nargs='?', default='temp/tracking_debug.txt',
                    help='Ruta al log (default: temp/tracking_debug.txt)')
    ap.add_argument('--wheelbase', '-L', type=float, default=_DEFAULT_WHEELBASE,
                    help=f'Distancia entre ejes [m] (default: {_DEFAULT_WHEELBASE})')
    ap.add_argument('--steer', '-s', type=float, default=None,
                    help='Forzar ángulo de steer [°]. Si no: usa el promedio del log.')
    ap.add_argument('--min-speed', type=float, default=5.0,
                    help='Velocidad mínima [cm/s] para incluir muestras (default: 5.0)')
    ap.add_argument('--skip', type=int, default=0,
                    help='Descartar los primeros N puntos (para ignorar aceleración inicial)')
    args = ap.parse_args()

    SEP  = "═" * 64
    SEP2 = "─" * 64

    # ── Leer log ──────────────────────────────────────────────────────────────
    print(f"\n[calibrate_dr] Leyendo {args.logfile} ...")
    try:
        all_entries = parse_log(args.logfile)
    except FileNotFoundError:
        print(f"\nERROR: No se encontró '{args.logfile}'")
        print("  Verificar: TRACKING_DEBUG_LOG = True en config.py")
        sys.exit(1)

    if not all_entries:
        print("\nERROR: No hay entradas en el log.")
        print("  Verificar: el auto se movió y TRACKING_DEBUG_LOG = True")
        sys.exit(1)

    all_entries = all_entries[args.skip:]

    # Estadísticas generales
    moving = filter_moving(all_entries, args.min_speed)
    encoder_entries = [e for e in all_entries if e['src'] == 'encoder']
    n_total = len(all_entries)
    n_moving = len(moving)
    n_encoder = len(encoder_entries)

    steers_all = [abs(e['steer']) for e in all_entries if abs(e['steer']) > 1.0]
    steer_mean = sum(steers_all) / len(steers_all) if steers_all else 0.0
    yaw_rot    = total_heading_deg(moving) if moving else 0.0

    print(f"  Entradas totales en log:   {n_total}")
    print(f"  Entradas en movimiento:    {n_moving}  (spd ≥ {args.min_speed:.0f} cm/s)")
    print(f"  Entradas con encoder:      {n_encoder}")
    print(f"  Steer medio (cuando >1°):  {steer_mean:.1f}°")
    print(f"  Rotación total (yaw):      {yaw_rot:.1f}°")

    # ── Método 2: Tasa de giro IMU ────────────────────────────────────────────
    print()
    print(SEP)
    print("  MÉTODO 2 — Tasa de giro IMU  (recomendado)")
    print(SEP)

    yr_estimates = yaw_rate_calibration(
        all_entries, args.wheelbase,
        min_steer_deg=5.0,
        min_speed_cms=args.min_speed,
    )

    if not yr_estimates:
        print()
        print("  ✗  Sin datos suficientes para Método 2.")
        print()
        print("  Causa probable: el encoder no registró datos durante el giro.")
        print("  El encoder BFMC solo genera pulsos a ≥ ~25 cm/s.")
        print()
        print("  SOLUCIÓN: repetir el ensayo a 25-30 cm/s con steer constante")
        print("  durante al menos 3-5 segundos. No es necesaria una vuelta completa.")
        print()
        yr_gain_mean = None
    else:
        gains  = [e['gain']  for e in yr_estimates]
        Rs     = [e['R']     for e in yr_estimates]
        omegas = [e['omega_deg_s'] for e in yr_estimates]
        dyaws  = [e['delta_yaw']   for e in yr_estimates]

        yr_gain_mean = sum(gains) / len(gains)
        yr_gain_std  = math.sqrt(sum((g - yr_gain_mean)**2 for g in gains) / len(gains)) if len(gains) > 1 else 0.0
        R_mean   = sum(Rs) / len(Rs)
        arc_total = sum(dyaws)

        steer_ref = args.steer if args.steer is not None else (
            sum(e['steer_deg'] for e in yr_estimates) / len(yr_estimates)
        )
        r_model_gain1 = args.wheelbase / math.tan(math.radians(abs(steer_ref))) if steer_ref > 1 else float('inf')

        print()
        print(f"  Pares de muestras usados:   {len(yr_estimates)}")
        print(f"  Arco cubierto (Δyaw total): {arc_total:.1f}°")
        print(f"  Radio real promedio:        {R_mean:.4f} m")
        print(f"  Radio modelo (gain=1.0):    {r_model_gain1:.4f} m")
        print(f"  ω promedio:                 {sum(omegas)/len(omegas):.1f}°/s")
        print()
        _print_gain_result(yr_gain_mean, yr_gain_std, steer_ref, args.wheelbase,
                           R_mean, r_model_gain1, len(yr_estimates), method=2)

        # Detalle de cada muestra
        if len(yr_estimates) <= 20:
            print()
            print("  Detalle por par de muestras:")
            print(f"  {'v(cm/s)':>8}  {'steer°':>7}  {'Δyaw°':>6}  {'R(m)':>6}  {'gain':>6}")
            print(f"  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}")
            for e in yr_estimates:
                print(f"  {e['v_cms']:8.1f}  {e['steer_deg']:7.1f}  {e['delta_yaw']:6.2f}  {e['R']:6.3f}  {e['gain']:6.3f}")

    # ── Método 1: Ajuste de círculo ───────────────────────────────────────────
    print()
    print(SEP)
    print("  MÉTODO 1 — Ajuste de círculo")
    print(SEP)

    if not moving:
        print()
        print("  ✗  Sin muestras en movimiento para ajuste de círculo.")
        print(f"     (necesita entradas con spd ≥ {args.min_speed:.0f} cm/s)")
        print()
        circle_gain = None
    elif yaw_rot < 90:
        print()
        print(f"  ✗  Solo {yaw_rot:.0f}° de rotación — insuficiente para ajuste de círculo.")
        print("     Se necesitan al menos 270-360° para un resultado confiable.")
        print()
        circle_gain = None
    else:
        steer_deg_vals = [abs(e['steer']) for e in moving]
        steer_mean_m   = sum(steer_deg_vals) / len(steer_deg_vals)
        steer_std_m    = math.sqrt(sum((s - steer_mean_m)**2 for s in steer_deg_vals) / len(steer_deg_vals))
        speed_vals     = [abs(e['spd_cms']) for e in moving]
        speed_mean_m   = sum(speed_vals) / len(speed_vals)

        print()
        print(f"  Muestras en movimiento:  {n_moving}")
        print(f"  Steer medio:             {steer_mean_m:.2f}° ± {steer_std_m:.2f}°")
        print(f"  Velocidad media:         {speed_mean_m:.1f} cm/s")
        print(f"  Rotación total (yaw):    {yaw_rot:.1f}°")

        if steer_std_m > 5.0:
            print(f"\n  ⚠  Steer muy variable (±{steer_std_m:.1f}°).")
        if yaw_rot < 270:
            print(f"\n  ⚠  Solo {yaw_rot:.0f}° de rotación. Se recomiendan ≥360°.")
        if n_moving < 50:
            print(f"\n  ⚠  Pocas muestras ({n_moving}).")

        steer_deg_use = args.steer if args.steer is not None else steer_mean_m
        steer_deg_use = abs(steer_deg_use)
        if steer_deg_use < 2.0:
            print(f"\n  ✗  Ángulo de steer {steer_deg_use:.1f}° demasiado pequeño.")
            circle_gain = None
        else:
            steer_rad_use = math.radians(steer_deg_use)
            r_model_gain1 = args.wheelbase / math.tan(steer_rad_use)

            xs = [e['x'] for e in moving]
            ys = [e['y'] for e in moving]

            try:
                cx, cy, r_dr, rms_m = fit_circle(xs, ys)
            except (ZeroDivisionError, ValueError) as exc:
                print(f"\n  ✗  Error al ajustar círculo: {exc}")
                circle_gain = None
            else:
                effective_angle_rad = math.atan(args.wheelbase / r_dr)
                circle_gain = effective_angle_rad / steer_rad_use
                circle_gain_std = 0.0

                print()
                print(f"  Radio modelo (gain=1.0):    {r_model_gain1:.4f} m")
                print(f"  Radio real (ajuste DR):     {r_dr:.4f} m")
                print(f"  RMS del ajuste:             {rms_m*100:.1f} cm")
                print(f"  Centro del círculo:         ({cx:.3f}, {cy:.3f}) m")
                print()
                _print_gain_result(circle_gain, circle_gain_std, steer_deg_use,
                                   args.wheelbase, r_dr, r_model_gain1,
                                   n_moving, method=1)

                if rms_m > 0.10:
                    print()
                    print(f"  ⚠  RMS alto ({rms_m*100:.1f} cm). Resultado poco confiable.")
                    print("     Repetir con velocidad más constante o más vueltas.")

                _ascii_plot(xs, ys, cx, cy, r_dr)

    # ── Resumen final ─────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  RESUMEN")
    print(SEP)
    if yr_gain_mean is not None:
        print(f"  Método 2 (yaw-rate): TRACKING_STEER_GAIN_DR = {yr_gain_mean:.3f}  ← usar este")
    if circle_gain is not None:
        print(f"  Método 1 (círculo):  TRACKING_STEER_GAIN_DR = {circle_gain:.3f}")
    if yr_gain_mean is None and circle_gain is None:
        print()
        print("  No fue posible calcular el gain.")
        print()
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │  QUÉ HACER:                                         │")
        print("  │  1. Activar steer ~25° ANTES de arrancar            │")
        print("  │  2. Acelerar a 25-30 cm/s                           │")
        print("  │  3. Mantener constante 3-5 segundos                 │")
        print("  │  4. Frenar y volver a correr este script            │")
        print("  └─────────────────────────────────────────────────────┘")
    print(SEP)
    print()


def _print_gain_result(gain: float, gain_std: float, steer_deg: float,
                       wheelbase: float, R_real: float, R_model: float,
                       n_samples: int, method: int) -> None:
    """Imprime el resultado de ganancia con interpretación."""
    effective_angle_deg = math.degrees(math.atan(wheelbase / R_real))

    if abs(gain - 1.0) < 0.04:
        print(f"  ✓  TRACKING_STEER_GAIN_DR = 1.0 ya es correcto.")
        print(f"     (radio real {R_real:.3f} m vs modelo {R_model:.3f} m, diff={100*(R_real-R_model)/R_model:+.1f}%)")
    else:
        std_str = f" ± {gain_std:.3f}" if gain_std > 0.001 else ""
        print(f"  TRACKING_STEER_GAIN_DR sugerido:  {gain:.3f}{std_str}")
        print()
        if gain < 1.0:
            print(f"  Interpretación: el DR gira MÁS CERRADO que el auto real.")
            print(f"  Las ruedas giran menos de lo que el servo comanda.")
            print(f"    Ángulo efectivo real: {effective_angle_deg:.1f}°  vs  comando: {steer_deg:.1f}°")
            print(f"  → Reducir TRACKING_STEER_GAIN_DR a {gain:.3f}")
        else:
            print(f"  Interpretación: el DR gira MÁS ABIERTO que el auto real.")
            print(f"  Las ruedas giran más de lo que el servo comanda.")
            print(f"    Ángulo efectivo real: {effective_angle_deg:.1f}°  vs  comando: {steer_deg:.1f}°")
            print(f"  → Aumentar TRACKING_STEER_GAIN_DR a {gain:.3f}")

    print()
    print(f"  Línea para copiar en config.py:")
    print(f"    TRACKING_STEER_GAIN_DR = {gain:.3f}")


def _ascii_plot(xs, ys, cx, cy, r, width=60, height=25):
    """Mini-visualización ASCII de la trayectoria ajustada."""
    if not xs:
        return
    x_min = min(xs + [cx - r])
    x_max = max(xs + [cx + r])
    y_min = min(ys + [cy - r])
    y_max = max(ys + [cy + r])

    pad = max(x_max - x_min, y_max - y_min) * 0.08
    x_min -= pad; x_max += pad
    y_min -= pad; y_max += pad

    def to_col(x): return int((x - x_min) / (x_max - x_min) * (width - 1))
    def to_row(y): return height - 1 - int((y - y_min) / (y_max - y_min) * (height - 1))

    grid = [[' '] * width for _ in range(height)]

    for deg in range(0, 360, 2):
        rad = math.radians(deg)
        px, py = cx + r * math.cos(rad), cy + r * math.sin(rad)
        c, rw = to_col(px), to_row(py)
        if 0 <= c < width and 0 <= rw < height:
            grid[rw][c] = '·'

    for pt_x, pt_y in zip(xs, ys):
        c, rw = to_col(pt_x), to_row(pt_y)
        if 0 <= c < width and 0 <= rw < height:
            grid[rw][c] = '█'

    cc, cr = to_col(cx), to_row(cy)
    if 0 <= cc < width and 0 <= cr < height:
        grid[cr][cc] = '+'

    c0, r0 = to_col(xs[0]), to_row(ys[0])
    if 0 <= c0 < width and 0 <= r0 < height:
        grid[r0][c0] = 'S'

    cn, rn = to_col(xs[-1]), to_row(ys[-1])
    if 0 <= cn < width and 0 <= rn < height:
        grid[rn][cn] = 'E'

    print()
    print("  Trayectoria DR (█) vs círculo ajustado (·)")
    print("  S=inicio  E=fin  +=centro del círculo")
    print()
    border = '  +' + '-' * width + '+'
    print(border)
    for row in grid:
        print('  |' + ''.join(row) + '|')
    print(border)
    print()


if __name__ == '__main__':
    main()
