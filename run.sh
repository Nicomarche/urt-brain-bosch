#!/bin/bash
# -----------------------------------------------------------------------------
# Launcher for urt-brain-bosch — three modes:
#
#   ./run.sh                     Mac dev: backend (sim) + GUI on localhost.
#   ./run.sh --monitor <IP[:P]>  Mac monitor: only the GUI, pointed at a remote
#                                car. No backend on this machine.
#   ./run.sh                     Jetson: backend only (no GUI), with full
#                                CUDA / Tegra environment.
#   ./run.sh --no-gui            Force backend-only (debug).
#
# Any extra args after a flag are forwarded to ``main.py`` (or to the GUI in
# monitor mode), so e.g. ``./run.sh --no-gui --quick`` still works.
# -----------------------------------------------------------------------------

set -e
# Job control: each ``cmd &`` becomes its own process group leader, so
# ``kill -- -PGID`` reaps the whole eventlet/multiprocess tree at once
# instead of orphaning the children.
set -m

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Live logging ----------------------------------------------------------
# Default manual/dev behavior:
#   - si `URT_LIVE_LOG_PATH` ya viene seteada, la respetamos;
#   - si no, creamos `temp/logs/run_<ts>/brain.jsonl`;
#   - en ambos casos garantizamos que exista el directorio padre;
#   - para el path default, actualizamos `temp/logs/latest` al run actual.
setup_live_log() {
    local logs_root run_ts run_dir latest_link
    logs_root="$SCRIPT_DIR/temp/logs"
    mkdir -p "$logs_root"

    if [[ -n "${URT_LIVE_LOG_PATH:-}" && "${URT_LIVE_LOG_PATH}" != "0" ]]; then
        mkdir -p "$(dirname "$URT_LIVE_LOG_PATH")"
        echo "[run.sh] Live log preset -> $URT_LIVE_LOG_PATH"
        return 0
    fi

    run_ts="$(date +"%Y%m%d_%H%M%S")"
    run_dir="$logs_root/run_$run_ts"
    if [[ -e "$run_dir" ]]; then
        run_dir="${run_dir}_$$"
    fi
    mkdir -p "$run_dir"

    export URT_LIVE_LOG_PATH="$run_dir/brain.jsonl"

    latest_link="$logs_root/latest"
    ln -sfn "$(basename "$run_dir")" "$latest_link"

    echo "[run.sh] Live log -> $URT_LIVE_LOG_PATH"
    echo "[run.sh] latest -> $latest_link"
}

# ---- Argument parsing -------------------------------------------------------
MODE="auto"          # auto | monitor | no-gui
MONITOR_TARGET=""    # host or host:port for --monitor
EXTRA_ARGS=()
GUI_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --monitor)
            MODE="monitor"
            MONITOR_TARGET="${2:?--monitor requires an IP[:PORT] argument}"
            shift 2
            ;;
        --no-gui)
            MODE="no-gui"
            shift
            ;;
        --gui-arg)
            # Pass-through for GUI-only flags (e.g. --gui-arg=--no-login)
            GUI_ARGS+=("${2:?--gui-arg needs a value}")
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Default port if user gave just an IP.
if [[ "$MODE" == "monitor" && "$MONITOR_TARGET" != *:* ]]; then
    MONITOR_TARGET="${MONITOR_TARGET}:5005"
fi

# ---- Platform-specific environment (before monitor so --monitor uses .venv) -
PYTHON_BIN=python3
PLATFORM="$(uname -s)"

if [[ "$PLATFORM" == "Darwin" ]]; then
    # Apple Silicon / macOS — sim mode only.
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    export URT_SIM_MODE="${URT_SIM_MODE:-1}"
    # acados — built from source; no Homebrew formula exists.
    export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-$HOME/acados_install}"
    export DYLD_LIBRARY_PATH="${ACADOS_SOURCE_DIR}/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    # torch + casadi both bundle libomp; allow duplicate to avoid abort on macOS.
    export KMP_DUPLICATE_LIB_OK=TRUE
    # LLVM libomp (bundled with torch) crashes in __kmp_suspend_initialize_thread
    # in forked children. Three env vars prevent the crash (see main.py for details).
    export OMP_NUM_THREADS=1
    export KMP_BLOCKTIME=0
    export KMP_AFFINITY=disabled
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1

    # Pick a Python that has gz bindings (brew installs them under one specific
    # python version, e.g. /opt/homebrew/lib/python3.13/site-packages/gz).
    # Stick to that interpreter — mixing PYTHONPATHs across versions breaks
    # numpy/cv2 native extensions (cpython-3XX ABI mismatch).
    if command -v brew >/dev/null 2>&1; then
        BREW_PREFIX="$(brew --prefix)"
        BASE_PYTHON=""
        for v in 3.13 3.12 3.11; do
            sp="$BREW_PREFIX/lib/python$v/site-packages"
            if [ -d "$sp/gz" ] && command -v "python$v" >/dev/null 2>&1; then
                BASE_PYTHON="python$v"
                break
            fi
        done
        if [ -n "$BASE_PYTHON" ]; then
            # Use a venv so pip can install packages (Homebrew Python is PEP-668
            # externally-managed; direct pip install is blocked without --break-system-packages).
            # --system-site-packages inherits gz and other Homebrew packages into the venv.
            if [ ! -d ".venv" ]; then
                echo "[run.sh] Creating .venv (system-site-packages)…"
                "$BASE_PYTHON" -m venv --system-site-packages .venv
            fi
            echo "[run.sh] Syncing requirements into .venv…"
            .venv/bin/pip install --quiet -r requirements.txt
            PYTHON_BIN=".venv/bin/python"
            # gz packages are inherited from the system site; no PYTHONPATH needed.
        fi
    fi
else
    # Jetson / Linux — full CUDA + Tegra runtime paths.
    export URT_SIM_MODE="${URT_SIM_MODE:-0}"
    export CUDA_HOME=/usr/local/cuda
    export PATH="/usr/local/cuda/bin:/home/urt/.local/bin:$PATH"
    export LD_LIBRARY_PATH="/home/urt/.local/lib/cusparselt:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH}"
    export PYTHONPATH="/home/urt/.local/lib/python3.10/site-packages:${PYTHONPATH}"
    # acados en Jetson vive en el mismo lugar que en Mac.
    export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-$HOME/acados_install}"
    export LD_LIBRARY_PATH="${ACADOS_SOURCE_DIR}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export URT_LAUNCHER="${URT_LAUNCHER:-run.sh}"
if [[ -z "${URT_EXPECTED_MPC_BACKEND:-}" ]]; then
    _URT_FORCE_PP_LC="$(printf '%s' "${URT_FORCE_PURE_PURSUIT:-}" | tr '[:upper:]' '[:lower:]')"
    if [[ "$_URT_FORCE_PP_LC" =~ ^(0|false|no|off)$ ]]; then
        export URT_EXPECTED_MPC_BACKEND="acados"
    elif [[ "$_URT_FORCE_PP_LC" =~ ^(1|true|yes|on)$ || "${URT_SIM_MODE:-0}" == "1" ]]; then
        export URT_EXPECTED_MPC_BACKEND="pure_pursuit"
    else
        export URT_EXPECTED_MPC_BACKEND="acados"
    fi
    unset _URT_FORCE_PP_LC
fi
export URT_PYTHON_BIN="$PYTHON_BIN"

# macOS + --monitor: if no brew/gz interpreter was chosen, we still need a venv
# (Homebrew Python is PEP-668; system ``python3`` has no PyQt5).
if [[ "$PLATFORM" == "Darwin" && "$MODE" == "monitor" && "$PYTHON_BIN" == "python3" ]]; then
    if [ ! -d ".venv" ]; then
        echo "[run.sh] Creating .venv for GUI (no Homebrew gz Python matched)…"
        python3 -m venv .venv
    fi
    echo "[run.sh] Syncing requirements into .venv…"
    .venv/bin/pip install --quiet -r requirements.txt
    PYTHON_BIN=".venv/bin/python"
fi

# ---- Pure GUI mode (remote monitor) -----------------------------------------
if [[ "$MODE" == "monitor" ]]; then
    echo "[run.sh] Monitor mode → GUI only, connecting to $MONITOR_TARGET"
    exec "$PYTHON_BIN" -m src.dashboard.gui.main \
        --connect "$MONITOR_TARGET" "${GUI_ARGS[@]}"
fi

# ---- Asegurar acados + solver (Mac y Jetson) --------------------------------
# Si todo está OK tarda ~1 s. Si falta algo, compila/descarga/genera sólo
# lo que falta. Pasar SKIP_ACADOS_CHECK=1 para saltear (CI sin red).
if [[ "${SKIP_ACADOS_CHECK:-0}" != "1" ]]; then
    bash "$SCRIPT_DIR/scripts/ensure_acados.sh" "$PYTHON_BIN" "$ACADOS_SOURCE_DIR"
fi
export URT_ACADOS_CHECK_PASSED=1

# ---- Backend-only (Jetson, or --no-gui on Mac) -----------------------------
if [[ "$MODE" == "no-gui" ]] || [[ "$PLATFORM" != "Darwin" ]]; then
    setup_live_log
    echo "[run.sh] Backend-only mode ($PLATFORM)"
    exec "$PYTHON_BIN" main.py "${EXTRA_ARGS[@]}"
fi

# ---- Mac dev: backend + GUI ------------------------------------------------
setup_live_log
echo "[run.sh] Mac dev mode → backend (sim) + GUI on localhost:5005"
echo "[run.sh] Press Ctrl+C in this terminal to stop both backend and GUI."

"$PYTHON_BIN" main.py "${EXTRA_ARGS[@]}" &
BACKEND_PID=$!

# Recursive kill: walk the parent → children tree from ``$1`` and send
# ``$2`` (default TERM) to every PID we find, leaves first. ``main.py``
# spawns several multiprocessing workers (camera, gateway, line-following…),
# so a flat ``kill $BACKEND_PID`` leaves them orphaned and the next launch
# can't bind :5005.
kill_tree() {
    local parent=$1 sig=${2:-TERM} child
    if ! kill -0 "$parent" 2>/dev/null; then return; fi
    for child in $(pgrep -P "$parent" 2>/dev/null || true); do
        kill_tree "$child" "$sig"
    done
    kill -"$sig" "$parent" 2>/dev/null || true
}

# Idempotent cleanup: SIGTERM the backend tree, give workers up to 2s to
# shut down, then SIGKILL anything still around. Runs on every exit path
# (Ctrl+C, GUI quit, script error) thanks to the trap below.
cleanup() {
    [[ -z "${BACKEND_PID:-}" ]] && return 0
    if kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "[run.sh] Stopping backend tree (pid=$BACKEND_PID)…"
        kill_tree "$BACKEND_PID" TERM
        # Wait up to ~2s for graceful shutdown.
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$BACKEND_PID" 2>/dev/null || break
            sleep 0.2
        done
        # Force-kill anything still alive (eventlet sometimes ignores TERM
        # when stuck in a C extension call — common with cv2 / numpy).
        kill_tree "$BACKEND_PID" KILL
    fi
    # Best-effort: free :5005 in case a stray child is still holding it.
    local stragglers
    stragglers=$(lsof -nP -iTCP:5005 -sTCP:LISTEN -t 2>/dev/null || true)
    if [[ -n "$stragglers" ]]; then
        echo "[run.sh] Killing :5005 holdouts: $stragglers"
        kill -KILL $stragglers 2>/dev/null || true
    fi
}

# Trap on every exit reason so cleanup is unavoidable: INT (Ctrl+C),
# TERM (e.g. ``kill <run.sh PID>``), HUP (terminal closed), EXIT (any
# normal exit including ``set -e`` failures).
trap cleanup EXIT INT TERM HUP

# Wait up to 10 seconds for the SocketIO port to come up before launching
# the GUI — the GUI handles reconnects fine, but starting after the backend
# is ready avoids a "disconnected" flash on first paint.
echo "[run.sh] Waiting for backend on :5005…"
for _ in $(seq 1 40); do
    if (exec 3<>/dev/tcp/127.0.0.1/5005) 2>/dev/null; then
        exec 3>&- 3<&-
        break
    fi
    sleep 0.25
done

# Run the GUI in the foreground; allow it to exit non-zero (Ctrl+C → 130,
# Cmd+Q → 0) without tripping ``set -e``. Whatever exit code we get we
# pass through, *after* the cleanup trap has had a chance to run.
set +e
"$PYTHON_BIN" -m src.dashboard.gui.main \
    --connect "localhost:5005" "${GUI_ARGS[@]}"
GUI_RC=$?
set -e
exit $GUI_RC
