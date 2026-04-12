#!/usr/bin/env python3
"""
calibrate_dr.py — Calibración de Dead Reckoning (BFMC TC-04)

Parsea el log de tracking generado durante un ensayo de círculos y calcula
el valor correcto de TRACKING_STEER_GAIN_DR.

PROCEDIMIENTO
─────────────
1. Activar el log:  TRACKING_DEBUG_LOG = True  en config.py
2. Poner el auto en MANUAL en un espacio abierto (radio libre ≥ 1.5 m)
3. Aplicar steer constante (ej: 25° derecha) y velocidad constante (~15 cm/s)
4. Dejar que el auto complete al menos 1 vuelta completa (360°)
5. Detener el auto
6. Ejecutar:
       python tools/calibrate_dr.py [temp/tracking_debug.txt]

SALIDA
──────
  • Radio del círculo trazado según el DR (con corrección IMU)
  • TRACKING_STEER_GAIN_DR sugerido
  • Interpretación: si el modelo gira más cerrado/abierto que el auto real

NOTA SOBRE EL MÉTODO
────────────────────
El DR integra posición usando el yaw del IMU (cuando está disponible), así que
las posiciones (x, y) del log reflejan la trayectoria real del auto, no la
predicha por el modelo de steer. Ajustando un círculo a esas posiciones
obtenemos el radio real de giro. Comparamos ese radio con el predicho por el
modelo cinemático a gain=1.0 para derivar el gain correcto.

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
#   F000010 | spd=15.0cm/s src=encoder ... steer=25.0° | raw=(x,y,yaw°) ...
_LOG_RE = re.compile(
    r'spd=(?P<spd>[+-]?[\d.]+)cm/s'
    r'.*?steer=(?P<steer>[+-]?[\d.]+)°'
    r'.*?raw=\((?P<x>[+-]?[\d.]+),(?P<y>[+-]?[\d.]+),(?P<yaw>[+-]?[\d.]+)°\)'
    r'.*?dt=(?P<dt>[\d.]+)ms'
)

_DEFAULT_WHEELBASE = 0.260   # metros — igual que TRACKING_WHEELBASE_M


# ── Parseo del log ────────────────────────────────────────────────────────────
def parse_log(path: str, min_speed_cms: float = 3.0) -> list:
    entries = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = _LOG_RE.search(line)
            if not m:
                continue
            spd_cms = float(m.group('spd'))
            if abs(spd_cms) < min_speed_cms:
                continue
            entries.append({
                'spd_cms': spd_cms,
                'steer':   float(m.group('steer')),   # grados, con signo
                'x':       float(m.group('x')),       # metros
                'y':       float(m.group('y')),       # metros
                'yaw':     float(m.group('yaw')),     # grados
                'dt_s':    float(m.group('dt')) / 1000.0,
            })
    return entries


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
    # Centrado para estabilidad numérica
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
    """Suma de cambios absolutos de yaw (grados)."""
    total = 0.0
    for i in range(1, len(entries)):
        delta = entries[i]['yaw'] - entries[i-1]['yaw']
        while delta >  180: delta -= 360
        while delta < -180: delta += 360
        total += abs(delta)
    return total


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
    ap.add_argument('--min-speed', type=float, default=3.0,
                    help='Velocidad mínima [cm/s] para incluir muestras (default: 3.0)')
    ap.add_argument('--skip', type=int, default=0,
                    help='Descartar los primeros N puntos (para ignorar la aceleración inicial)')
    args = ap.parse_args()

    # ── Leer log ──────────────────────────────────────────────────────────────
    print(f"\n[calibrate_dr] Leyendo {args.logfile} ...")
    try:
        entries = parse_log(args.logfile, min_speed_cms=args.min_speed)
    except FileNotFoundError:
        print(f"\nERROR: No se encontró '{args.logfile}'")
        print("  Verificar: TRACKING_DEBUG_LOG = True en config.py")
        sys.exit(1)

    if not entries:
        print("\nERROR: No hay muestras en movimiento en el log.")
        print("  Verificar: el auto se movió y TRACKING_DEBUG_LOG = True")
        sys.exit(1)

    entries = entries[args.skip:]

    n = len(entries)
    steers_abs = [abs(e['steer']) for e in entries]
    speeds_cms  = [abs(e['spd_cms']) for e in entries]
    steer_mean  = sum(steers_abs) / n
    steer_std   = math.sqrt(sum((s - steer_mean)**2 for s in steers_abs) / n)
    speed_mean  = sum(speeds_cms) / n
    yaw_rot     = total_heading_deg(entries)

    print(f"  Muestras en movimiento:  {n}")
    print(f"  Steer medio:             {steer_mean:.2f}° ± {steer_std:.2f}°")
    print(f"  Velocidad media:         {speed_mean:.1f} cm/s")
    print(f"  Rotación total (yaw):    {yaw_rot:.1f}°")

    # ── Advertencias ──────────────────────────────────────────────────────────
    warnings_found = False
    if steer_std > 5.0:
        print(f"\n  ⚠  Steer muy variable (±{steer_std:.1f}°). Para calibración precisa")
        print(f"     mantener steer constante durante el ensayo.")
        warnings_found = True
    if yaw_rot < 270:
        print(f"\n  ⚠  Solo {yaw_rot:.0f}° de rotación total. Se recomienda al menos 360°.")
        print(f"     El ajuste puede ser poco preciso.")
        warnings_found = True
    if n < 50:
        print(f"\n  ⚠  Pocas muestras ({n}). El ajuste puede ser poco preciso.")
        warnings_found = True

    # ── Ángulo de steer a usar ────────────────────────────────────────────────
    steer_deg = args.steer if args.steer is not None else steer_mean
    steer_deg = abs(steer_deg)
    if steer_deg < 2.0:
        print(f"\nERROR: Ángulo de steer {steer_deg:.1f}° demasiado pequeño para calibrar.")
        print("  Repetir el ensayo con steer ≥ 10°.")
        sys.exit(1)

    steer_rad = math.radians(steer_deg)
    r_model_gain1 = args.wheelbase / math.tan(steer_rad)

    # ── Ajuste del círculo ────────────────────────────────────────────────────
    xs = [e['x'] for e in entries]
    ys = [e['y'] for e in entries]

    try:
        cx, cy, r_dr, rms_m = fit_circle(xs, ys)
    except (ZeroDivisionError, ValueError) as exc:
        print(f"\nERROR al ajustar círculo: {exc}")
        print("  Verificar que el auto haya girado suficiente (≥ 360°).")
        sys.exit(1)

    # ── Ganancia sugerida ─────────────────────────────────────────────────────
    # Queremos que el modelo prediga el mismo radio que el que traza físicamente:
    #   WB / tan(steer_cmd * gain) = r_dr
    #   gain = atan(WB / r_dr) / steer_cmd_rad
    effective_angle_rad = math.atan(args.wheelbase / r_dr)
    gain_suggested = effective_angle_rad / steer_rad

    # ── Resultado ─────────────────────────────────────────────────────────────
    SEP = "─" * 64
    print()
    print(SEP)
    print("  RESULTADO DE CALIBRACIÓN — TRACKING_STEER_GAIN_DR")
    print(SEP)
    print(f"  Steer de referencia:        {steer_deg:.2f}°")
    print(f"  Wheelbase:                  {args.wheelbase:.3f} m")
    print()
    print(f"  Radio modelo (gain=1.0):    {r_model_gain1:.4f} m")
    print(f"  Radio real (ajuste DR):     {r_dr:.4f} m")
    print(f"  RMS del ajuste:             {rms_m*100:.1f} cm")
    print(f"  Centro del círculo:         ({cx:.3f}, {cy:.3f}) m")
    print()

    err_pct = (r_dr - r_model_gain1) / r_model_gain1 * 100
    if abs(gain_suggested - 1.0) < 0.04:
        print(f"  ✓  TRACKING_STEER_GAIN_DR = 1.0 ya es correcto.")
        print(f"     (diferencia de radio: {err_pct:+.1f}%)")
    else:
        print(f"  TRACKING_STEER_GAIN_DR sugerido:  {gain_suggested:.3f}")
        print()
        if gain_suggested < 1.0:
            print(f"  Interpretación: el DR gira MÁS CERRADO que el auto real.")
            print(f"  El ángulo efectivo de las ruedas es menor que el comando del servo.")
            print(f"    ({effective_angle_rad*180/math.pi:.1f}° real vs {steer_deg:.1f}° comando)")
            print(f"  → Reducir TRACKING_STEER_GAIN_DR a {gain_suggested:.3f}")
        else:
            print(f"  Interpretación: el DR gira MÁS ABIERTO que el auto real.")
            print(f"  El ángulo efectivo de las ruedas es mayor que el comando del servo.")
            print(f"    ({effective_angle_rad*180/math.pi:.1f}° real vs {steer_deg:.1f}° comando)")
            print(f"  → Aumentar TRACKING_STEER_GAIN_DR a {gain_suggested:.3f}")

    if rms_m > 0.05:
        print()
        print(f"  ⚠  RMS alto ({rms_m*100:.1f} cm). El resultado puede ser poco confiable.")
        print(f"     Repetir con velocidad más constante o más vueltas.")

    print()
    print("  Línea para copiar en config.py:")
    print(f"    TRACKING_STEER_GAIN_DR = {gain_suggested:.3f}")
    print(SEP)

    # ── Plot ASCII de la trayectoria (opcional, siempre mostrar) ─────────────
    _ascii_plot(xs, ys, cx, cy, r_dr)


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

    # Círculo ajustado
    for deg in range(0, 360, 2):
        rad = math.radians(deg)
        px, py = cx + r * math.cos(rad), cy + r * math.sin(rad)
        c, rw = to_col(px), to_row(py)
        if 0 <= c < width and 0 <= rw < height:
            grid[rw][c] = '·'

    # Trayectoria real
    for pt_x, pt_y in zip(xs, ys):
        c, rw = to_col(pt_x), to_row(pt_y)
        if 0 <= c < width and 0 <= rw < height:
            grid[rw][c] = '█'

    # Centro
    cc, cr = to_col(cx), to_row(cy)
    if 0 <= cc < width and 0 <= cr < height:
        grid[cr][cc] = '+'

    # Primer punto
    c0, r0 = to_col(xs[0]), to_row(ys[0])
    if 0 <= c0 < width and 0 <= r0 < height:
        grid[r0][c0] = 'S'   # Start

    # Último punto
    cn, rn = to_col(xs[-1]), to_row(ys[-1])
    if 0 <= cn < width and 0 <= rn < height:
        grid[rn][cn] = 'E'   # End

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
