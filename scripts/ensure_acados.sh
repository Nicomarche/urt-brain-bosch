#!/usr/bin/env bash
# scripts/ensure_acados.sh — garantiza que acados + solver estén listos.
#
# Llamado automáticamente por run.sh antes de arrancar el backend.
# Es idempotente: si todo está OK lo detecta en ~1 s y sale.
#
# Uso:
#   scripts/ensure_acados.sh <python_bin> <acados_install_dir>
#
# Ejemplos:
#   scripts/ensure_acados.sh .venv/bin/python ~/acados_install   # Mac
#   scripts/ensure_acados.sh python3          ~/acados_install   # Jetson

set -euo pipefail

PYTHON_BIN="${1:-python3}"
ACADOS_INSTALL_DIR="${2:-$HOME/acados_install}"
ACADOS_SRC_DIR="$HOME/acados"
TERA_VERSION="v0.2.0"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOLVER_JSON="$PROJECT_ROOT/src/control/c_generated_code/acados_ocp_bfmc_bicycle.json"

PLATFORM="$(uname -s)"
ARCH="$(uname -m)"

# ── Colores ───────────────────────────────────────────────────────────────────
_G="\033[1;92m"; _Y="\033[1;93m"; _R="\033[1;91m"; _W="\033[1;97m"; _0="\033[0m"
log()  { echo -e "${_W}[acados]${_0} ${_G}$*${_0}"; }
warn() { echo -e "${_W}[acados]${_0} ${_Y}$*${_0}"; }
err()  { echo -e "${_W}[acados]${_0} ${_R}$*${_0}" >&2; }

log "Using python: $PYTHON_BIN"
log "ACADOS install dir: $ACADOS_INSTALL_DIR"

# ── Extensión de shared lib según plataforma ──────────────────────────────────
if [[ "$PLATFORM" == "Darwin" ]]; then
    LIB_EXT="dylib"
    ACADOS_LIB_NAME="libacados.dylib"
    if [[ "$ARCH" == "arm64" ]]; then
        TERA_ASSET="t_renderer-${TERA_VERSION}-osx-arm64"
    else
        TERA_ASSET="t_renderer-${TERA_VERSION}-osx-amd64"
    fi
else
    LIB_EXT="so"
    ACADOS_LIB_NAME="libacados.so"
    if [[ "$ARCH" == "aarch64" ]]; then
        TERA_ASSET="t_renderer-${TERA_VERSION}-linux-arm64"
    else
        TERA_ASSET="t_renderer-${TERA_VERSION}-linux-amd64"
    fi
fi

ACADOS_LIB="$ACADOS_INSTALL_DIR/lib/$ACADOS_LIB_NAME"
TERA_BIN="$ACADOS_INSTALL_DIR/bin/t_renderer"

# ── Helpers de verificación ───────────────────────────────────────────────────
_has_libs()     { [[ -f "$ACADOS_LIB" ]]; }
_has_tera()     { [[ -f "$TERA_BIN" && -x "$TERA_BIN" ]]; }
_has_template() {
    ACADOS_SOURCE_DIR="$ACADOS_INSTALL_DIR" \
        "$PYTHON_BIN" -c "from acados_template import AcadosOcpSolver" 2>/dev/null
}
_solver_json_matches_host() {
    [[ -f "$SOLVER_JSON" ]] || return 1
    "$PYTHON_BIN" - "$SOLVER_JSON" "$PROJECT_ROOT/src/control/c_generated_code" "$ACADOS_INSTALL_DIR" ".$LIB_EXT" <<'PY'
import json
import os
import sys

json_path, solver_dir, acados_dir, expected_ext = sys.argv[1:5]

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

opts = data.get("code_gen_opts", {})

def same_path(actual: object, expected: str) -> bool:
    if not isinstance(actual, str) or not actual:
        return False
    return os.path.abspath(os.path.expanduser(actual)) == os.path.abspath(os.path.expanduser(expected))

checks = [
    same_path(opts.get("code_export_directory"), solver_dir),
    same_path(opts.get("json_file"), json_path),
    same_path(opts.get("acados_lib_path"), os.path.join(acados_dir, "lib")),
    same_path(opts.get("acados_include_path"), os.path.join(acados_dir, "include")),
    opts.get("shared_lib_ext") == expected_ext,
]

raise SystemExit(0 if all(checks) else 1)
PY
}
_has_solver() {
    [[ -f "$SOLVER_JSON" ]] || return 1
    if [[ "$PLATFORM" == "Darwin" ]]; then
        local lib_path="$PROJECT_ROOT/src/control/c_generated_code/libacados_ocp_solver_bfmc_bicycle.dylib"
    else
        local lib_path="$PROJECT_ROOT/src/control/c_generated_code/libacados_ocp_solver_bfmc_bicycle.so"
    fi
    [[ -f "$lib_path" ]] || return 1
    _solver_json_matches_host
}
_solver_loads() {
    cd "$PROJECT_ROOT"
    ACADOS_SOURCE_DIR="$ACADOS_INSTALL_DIR" \
    DYLD_LIBRARY_PATH="$ACADOS_INSTALL_DIR/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
    LD_LIBRARY_PATH="$ACADOS_INSTALL_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    KMP_DUPLICATE_LIB_OK=TRUE \
        "$PYTHON_BIN" -c "
from src.control.motion_controller import AcadosMPC
m = AcadosMPC()
assert m.ready, 'solver not ready'
" 2>/dev/null
}

# ── Check rápido: ¿todo está OK? ──────────────────────────────────────────────
if _has_libs && _has_tera && _has_template && _has_solver && _solver_loads; then
    log "Todo OK — acados listo en $ACADOS_INSTALL_DIR"
    exit 0
fi

warn "acados incompleto — iniciando instalación automática…"
warn "  libs:     $(_has_libs     && echo 'OK' || echo 'FALTA')"
warn "  tera:     $(_has_tera     && echo 'OK' || echo 'FALTA')"
warn "  template: $(_has_template && echo 'OK' || echo 'FALTA')"
warn "  solver:   $(_has_solver   && echo 'OK' || echo 'FALTA')"

# ── 1. Dependencias del sistema ───────────────────────────────────────────────
log "Paso 1/4 — verificando dependencias del sistema…"
if [[ "$PLATFORM" == "Darwin" ]]; then
    for dep in cmake gcc git curl; do
        command -v "$dep" >/dev/null 2>&1 || {
            err "Falta '$dep'. Instalá con: brew install $dep"
            exit 1
        }
    done
else
    # Jetson/Linux
    for dep in cmake gcc g++ git curl; do
        command -v "$dep" >/dev/null 2>&1 || {
            warn "Falta '$dep'. Intentando instalar con apt…"
            sudo apt-get install -y cmake gcc g++ git curl liblapack-dev libblas-dev
            break
        }
    done
fi

# ── 2. Compilar acados si las libs faltan ─────────────────────────────────────
if ! _has_libs; then
    log "Paso 2/4 — compilando acados desde fuente (≈ 5-10 min)…"

    if [[ ! -d "$ACADOS_SRC_DIR" ]]; then
        log "  Clonando repositorio…"
        git clone https://github.com/acados/acados.git "$ACADOS_SRC_DIR"
    else
        log "  Fuente ya presente en $ACADOS_SRC_DIR — actualizando submodules…"
    fi

    cd "$ACADOS_SRC_DIR"
    git submodule update --init --recursive

    mkdir -p build && cd build
    cmake .. \
        -DACADOS_WITH_QPOASES=ON \
        -DACADOS_INSTALL_DIR="$ACADOS_INSTALL_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON

    JOBS=$(command -v nproc >/dev/null 2>&1 && nproc || sysctl -n hw.ncpu)
    make -j"$JOBS"
    make install

    # link_libs.json no se instala automáticamente; copiarlo a mano
    [[ -f "$ACADOS_SRC_DIR/lib/link_libs.json" ]] && \
        cp "$ACADOS_SRC_DIR/lib/link_libs.json" "$ACADOS_INSTALL_DIR/lib/"

    cd "$PROJECT_ROOT"
    log "  acados compilado e instalado en $ACADOS_INSTALL_DIR"
else
    log "Paso 2/4 — libs OK, saltando compilación"
fi

# ── 3. Tera renderer ──────────────────────────────────────────────────────────
if ! _has_tera; then
    log "Paso 3/4 — descargando tera renderer (${TERA_ASSET})…"
    mkdir -p "$ACADOS_INSTALL_DIR/bin"
    TERA_URL="https://github.com/acados/tera_renderer/releases/download/${TERA_VERSION}/${TERA_ASSET}"
    curl -fsSL "$TERA_URL" -o "$TERA_BIN"
    chmod +x "$TERA_BIN"
    log "  tera renderer descargado"
else
    log "Paso 3/4 — tera renderer OK"
fi

# ── 4. acados_template Python ─────────────────────────────────────────────────
if ! _has_template; then
    log "Paso 4a/4 — instalando acados_template Python…"

    # Dependencias previas
    if [[ "$PLATFORM" == "Darwin" ]]; then
        "$PYTHON_BIN" -m pip install vcs-versioning casadi --break-system-packages -q
        "$PYTHON_BIN" -m pip install \
            "$ACADOS_SRC_DIR/interfaces/acados_template" \
            --no-deps --break-system-packages -q
    else
        "$PYTHON_BIN" -m pip install vcs-versioning casadi -q
        "$PYTHON_BIN" -m pip install \
            "$ACADOS_SRC_DIR/interfaces/acados_template" \
            --no-deps -q
    fi
    log "  acados_template instalado"
else
    log "Paso 4a/4 — acados_template OK"
fi

# ── 5. Generar solver si falta ────────────────────────────────────────────────
if ! _has_solver; then
    log "Paso 4b/4 — generando solver OCP (bicicleta BFMC)…"
    cd "$PROJECT_ROOT"
    ACADOS_SOURCE_DIR="$ACADOS_INSTALL_DIR" \
    DYLD_LIBRARY_PATH="$ACADOS_INSTALL_DIR/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
    LD_LIBRARY_PATH="$ACADOS_INSTALL_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    KMP_DUPLICATE_LIB_OK=TRUE \
        "$PYTHON_BIN" -m src.control._acados_solver_gen
    log "  solver generado"
else
    log "Paso 4b/4 — solver OK"
fi

# ── Verificación final ────────────────────────────────────────────────────────
log "Verificando que AcadosMPC carga correctamente…"
if _solver_loads; then
    log "✓ AcadosMPC listo — instalación completa en $ACADOS_INSTALL_DIR"
else
    err "AcadosMPC no cargó después de la instalación."
    err "python=$PYTHON_BIN acados_install=$ACADOS_INSTALL_DIR solver_json=$SOLVER_JSON"
    err "Probá arrancar con ./run.sh para usar el entorno soportado."
    exit 1
fi
