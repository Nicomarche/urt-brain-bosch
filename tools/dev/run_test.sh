#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_test.sh — orquesta un test end-to-end del brain Bosch contra el sim.
#
# Flujo:
#   1) Limpia zombies, crea dirs de logs timestamped (run_<TS>) en ambos repos
#   2) Levanta gz sim --headless + sim_bridge.py
#   3) Levanta gz_pose_logger.py (ground-truth tap)
#   4) Espera readiness del bridge (puerto ZMQ frame :5575)
#   5) Levanta brain con URT_LIVE_LOG_PATH apuntando al JSONL del run
#   6) Espera readiness del brain (puerto SocketIO :5005 + primer pose_published)
#   7) Lanza headless_controller.py (manda DrivingMode=auto, espera N s, manda stop)
#   8) Cleanup tree (TERM con timeout, luego KILL)
#   9) Corre analyze_run.py → report.txt en el dir del run
#  10) Symlink temp/logs/latest -> run_<TS>
#
# Uso:
#   bash tools/dev/run_test.sh [DURATION_SECONDS]
#   bash tools/dev/run_test.sh 60      # default
#   bash tools/dev/run_test.sh 120
#
# Variables de entorno respetadas:
#   URT_LIVE_LOG_PATH — si está seteada, override del path automático
#   SKIP_ANALYZE=1    — saltea el postmortem (útil si querés revisar JSONL crudo)
# -----------------------------------------------------------------------------

set -e
set -u

DURATION="${1:-60}"
BRAIN_DIR="/Users/luciogarcia/urt-brain-bosch"
SIM_DIR="/Users/luciogarcia/urt-simulator"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
BRAIN_RUN_DIR="$BRAIN_DIR/temp/logs/run_$RUN_TS"
SIM_RUN_DIR="$SIM_DIR/temp/logs/run_$RUN_TS"
BRAIN_JSONL="$BRAIN_RUN_DIR/brain.jsonl"
SIM_JSONL="$SIM_RUN_DIR/sim_bridge.jsonl"

mkdir -p "$BRAIN_RUN_DIR" "$SIM_RUN_DIR"
: > "$BRAIN_JSONL"
: > "$SIM_JSONL"

# Resolución de pythons:
#   BRAIN_PY  = venv del brain (tiene socketio+requests para el headless_controller)
#   GZ_PY     = python con gz bindings (Homebrew python3.13 con gz)
#   GZ_PYPATH = site-packages de Homebrew donde vive gz
BRAIN_PY="$BRAIN_DIR/.venv/bin/python"
if [ ! -x "$BRAIN_PY" ]; then
    BRAIN_PY="python3"
fi

GZ_PY="python3"
GZ_PYPATH=""
if command -v brew >/dev/null 2>&1; then
    BREW_PREFIX="$(brew --prefix)"
    for v in 3.13 3.12; do
        if [ -d "$BREW_PREFIX/lib/python$v/site-packages/gz" ] && command -v "python$v" >/dev/null 2>&1; then
            GZ_PY="python$v"
            GZ_PYPATH="$BREW_PREFIX/lib/python$v/site-packages"
            break
        fi
    done
fi

echo "[run_test] BRAIN_PY=$BRAIN_PY"
echo "[run_test] GZ_PY=$GZ_PY (PYTHONPATH=$GZ_PYPATH)"

echo "[run_test] === run_$RUN_TS, duration=${DURATION}s ==="
echo "[run_test] brain logs:    $BRAIN_RUN_DIR/"
echo "[run_test] sim logs:      $SIM_RUN_DIR/"

# ---- 1) Cleanup zombies de runs previos ------------------------------------
echo "[run_test] killing zombies..."
pkill -9 -f "gz-sim-main"        2>/dev/null || true
pkill -9 -f "gz-transport-topic" 2>/dev/null || true
pkill -9 -f "ruby .*ign-gazebo"  2>/dev/null || true
pkill -9 -f "$SIM_DIR/sim_bridge.py"        2>/dev/null || true
pkill -9 -f "$SIM_DIR/tools/dev/gz_pose_logger.py" 2>/dev/null || true
pkill -9 -f "$BRAIN_DIR/main.py" 2>/dev/null || true
pkill -9 -f "$BRAIN_DIR/.venv/bin/python.*main.py" 2>/dev/null || true
pkill -9 -f "headless_controller.py" 2>/dev/null || true
sleep 2

# ---- Cleanup function ------------------------------------------------------
SIM_PID=""
GZ_LOGGER_PID=""
BRAIN_PID=""
CTRL_PID=""
SNAP_PID=""

cleanup() {
    echo "[run_test] cleanup..."
    [ -n "${SNAP_PID:-}" ] && kill "$SNAP_PID" 2>/dev/null || true
    [ -n "${CTRL_PID:-}" ] && kill -TERM "$CTRL_PID" 2>/dev/null || true
    [ -n "${BRAIN_PID:-}" ] && kill -TERM "$BRAIN_PID" 2>/dev/null || true
    [ -n "${GZ_LOGGER_PID:-}" ] && kill -TERM "$GZ_LOGGER_PID" 2>/dev/null || true
    [ -n "${SIM_PID:-}" ] && kill -TERM "$SIM_PID" 2>/dev/null || true
    sleep 2
    # Force kill anything still around
    pkill -9 -f "gz-sim-main"        2>/dev/null || true
    pkill -9 -f "$SIM_DIR/sim_bridge.py"  2>/dev/null || true
    pkill -9 -f "$SIM_DIR/tools/dev/gz_pose_logger.py" 2>/dev/null || true
    pkill -9 -f "$BRAIN_DIR/main.py" 2>/dev/null || true
    pkill -9 -f "$BRAIN_DIR/.venv/bin/python.*main.py" 2>/dev/null || true
    pkill -9 -f "headless_controller.py" 2>/dev/null || true
    # Free port :5005 si quedó tomado
    local stragglers
    stragglers=$(lsof -nP -iTCP:5005 -sTCP:LISTEN -t 2>/dev/null || true)
    if [ -n "$stragglers" ]; then
        kill -9 $stragglers 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM HUP

# ---- 2) Levanta gz sim + bridge --------------------------------------------
echo "[run_test] starting gz sim --headless + sim_bridge..."
(
    cd "$SIM_DIR"
    ./run_sim.sh --headless > "$SIM_RUN_DIR/sim.stdout" 2>&1
) &
SIM_PID=$!

# ---- 3) Ground-truth pose logger -------------------------------------------
# Esperamos un poco para que gz server arranque y registre el modelo antes
# de que el logger intente suscribirse.
sleep 5

echo "[run_test] starting gz_pose_logger ($GZ_PY)..."
(
    cd "$SIM_DIR"
    [ -n "$GZ_PYPATH" ] && export PYTHONPATH="$GZ_PYPATH${PYTHONPATH:+:$PYTHONPATH}"
    "$GZ_PY" "$SIM_DIR/tools/dev/gz_pose_logger.py" \
        --out "$SIM_JSONL" \
        > "$SIM_RUN_DIR/gz_logger.stdout" 2>&1
) &
GZ_LOGGER_PID=$!

# ---- 4) Esperar readiness del bridge (ZMQ :5575) ---------------------------
echo "[run_test] waiting for sim_bridge ZMQ :5575..."
ready=0
for i in $(seq 1 60); do
    if nc -z localhost 5575 2>/dev/null; then
        ready=1
        echo "[run_test] sim_bridge ZMQ ready (after ${i}s)"
        break
    fi
    sleep 1
done
if [ "$ready" != "1" ]; then
    echo "[run_test] FATAL: sim_bridge ZMQ :5575 nunca subió. sim.stdout:"
    tail -30 "$SIM_RUN_DIR/sim.stdout" || true
    exit 1
fi

# ---- 5) Levanta brain ------------------------------------------------------
echo "[run_test] starting brain (--no-gui)..."
(
    cd "$BRAIN_DIR"
    export URT_LIVE_LOG_PATH="$BRAIN_JSONL"
    export URT_SIM_MODE="${URT_SIM_MODE:-1}"
    # Durante tests automatizados, el detector de parking signs cortaba
    # los runs ~16s después de entrar a AUTO. Para tests de lane following
    # no queremos esa interrupción — el maneuverManager ignora el parking
    # sign request pero respeta el resto (stop, semáforo, etc.).
    export URT_DISABLE_AUTO_PARKING="${URT_DISABLE_AUTO_PARKING:-1}"
    ./run.sh --no-gui > "$BRAIN_RUN_DIR/brain.stdout" 2>&1
) &
BRAIN_PID=$!

# ---- 6) Esperar readiness del brain ----------------------------------------
echo "[run_test] waiting for brain SocketIO :5005..."
ready=0
for i in $(seq 1 60); do
    if nc -z localhost 5005 2>/dev/null; then
        ready=1
        echo "[run_test] brain SocketIO ready (after ${i}s)"
        break
    fi
    sleep 1
done
if [ "$ready" != "1" ]; then
    echo "[run_test] FATAL: brain SocketIO :5005 nunca subió. brain.stdout (últimas 40):"
    tail -40 "$BRAIN_RUN_DIR/brain.stdout" || true
    exit 1
fi

echo "[run_test] waiting for first pose_published in JSONL..."
ready=0
for i in $(seq 1 90); do
    if [ -f "$BRAIN_JSONL" ] && grep -q '"event":"pose_published"' "$BRAIN_JSONL" 2>/dev/null; then
        ready=1
        echo "[run_test] pose_published seen (after ${i}s)"
        break
    fi
    sleep 1
done
if [ "$ready" != "1" ]; then
    echo "[run_test] WARN: no pose_published seen after 90s. Continuing anyway."
    echo "[run_test] (esto puede pasar si live_log aún no está conectado en el thread de localización)"
fi

# ---- 7) Headless controller — manda DrivingMode=auto -----------------------
echo "[run_test] launching headless_controller (duration=${DURATION}s, py=$BRAIN_PY)..."
(
    cd "$BRAIN_DIR"
    "$BRAIN_PY" tools/dev/headless_controller.py --duration "$DURATION" \
        > "$BRAIN_RUN_DIR/controller.stdout" 2>&1
) &
CTRL_PID=$!

# ---- 8) Snapshot periódico al stdout para visibilidad en background --------
(
    while kill -0 "$CTRL_PID" 2>/dev/null; do
        sleep 10
        echo "=== T+$(date +%H:%M:%S) live snapshot (last 20 events) ==="
        if [ -f "$BRAIN_JSONL" ]; then
            tail -n 20 "$BRAIN_JSONL" 2>/dev/null | python3 -c "
import sys, json
for l in sys.stdin:
    try:
        d = json.loads(l)
        kvs = ' '.join(f'{k}={v}' for k,v in list(d.items())[4:9] if k not in ('ts','mono','pid','thread','event'))
        print(f'  {d[\"thread\"]:18s} {d[\"event\"]:24s} {kvs[:120]}')
    except Exception:
        pass
" || true
        fi
    done
) &
SNAP_PID=$!

# Esperar a que termine el controller
wait "$CTRL_PID" || true
echo "[run_test] headless_controller finished"

# ---- 9) Cleanup gradual ----------------------------------------------------
kill "$SNAP_PID" 2>/dev/null || true

echo "[run_test] stopping brain..."
kill -TERM "$BRAIN_PID" 2>/dev/null || true
sleep 3
kill -KILL "$BRAIN_PID" 2>/dev/null || true

echo "[run_test] stopping gz_pose_logger..."
kill -TERM "$GZ_LOGGER_PID" 2>/dev/null || true

echo "[run_test] stopping gz sim + bridge..."
kill -TERM "$SIM_PID" 2>/dev/null || true
sleep 2
pkill -9 -f "gz-sim-main" 2>/dev/null || true
pkill -9 -f "$SIM_DIR/sim_bridge.py" 2>/dev/null || true

# ---- 10) Análisis postmortem ----------------------------------------------
if [ "${SKIP_ANALYZE:-0}" != "1" ]; then
    echo "[run_test] running postmortem..."
    "$BRAIN_PY" "$BRAIN_DIR/tools/dev/analyze_run.py" "$BRAIN_RUN_DIR" \
        --sim-jsonl "$SIM_JSONL" \
        --report "$BRAIN_RUN_DIR/report.txt" || true

    # Comparativa brain-vs-gt time-series — detecta cuándo el yaw del brain
    # se desincroniza del ground truth del simulador (síntoma típico:
    # `Sim Feedback ] WARNING - 0 msg/s from sim_bridge` deja al brain con
    # pose congelada mientras el coche sigue rotando en gz). Salida: CSV
    # con (ts, brain_xy/yaw, gt_xy/yaw) + estadísticas de drift.
    echo "[run_test] running brain-vs-gt comparison..."
    "$BRAIN_PY" "$BRAIN_DIR/tools/dev/compare_brain_vs_gt.py" "$BRAIN_RUN_DIR" \
        --every 50 \
        > "$BRAIN_RUN_DIR/compare_brain_vs_gt.txt" 2>&1 || true
fi

# ---- 11) Symlink "latest" -------------------------------------------------
ln -sfn "run_$RUN_TS" "$BRAIN_DIR/temp/logs/latest"
ln -sfn "run_$RUN_TS" "$SIM_DIR/temp/logs/latest"

# Stats finales
brain_lines=$(wc -l < "$BRAIN_JSONL" 2>/dev/null || echo 0)
sim_lines=$(wc -l < "$SIM_JSONL" 2>/dev/null || echo 0)
echo "[run_test] DONE — brain.jsonl: $brain_lines lines, sim_bridge.jsonl: $sim_lines lines"
echo "[run_test] symlinks updated:"
echo "  $BRAIN_DIR/temp/logs/latest -> run_$RUN_TS"
echo "  $SIM_DIR/temp/logs/latest -> run_$RUN_TS"
