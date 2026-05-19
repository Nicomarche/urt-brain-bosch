#!/bin/bash
# Launcher for urt-brain-bosch on Jetson Orin Nano (run as root)
#
# Usage (VNC or physical monitor):
#   xhost +SI:localuser:root        # one-time per X session, from the user's terminal
#   sudo -E ./run.sh                # -E preserves DISPLAY/XAUTHORITY from the VNC env
#
# If you forget -E, sudo strips DISPLAY and the script falls back to detecting the
# VNC user's session below.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:/home/urt/.local/bin:$PATH"
export LD_LIBRARY_PATH="/home/urt/.local/lib/cusparselt:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH}"
export PYTHONPATH="/home/urt/.local/lib/python3.10/site-packages:${PYTHONPATH}"

# Inherit the X session so cv2.imshow() / GTK windows can render under sudo.
DESKTOP_USER="${SUDO_USER:-urt}"
USER_UID="$(id -u "${DESKTOP_USER}" 2>/dev/null || echo 1000)"

# Best path: DISPLAY/XAUTHORITY were preserved (sudo -E). Otherwise sniff them
# from a process running as the desktop user (Xtigervnc, Xvnc, Xorg, xfce-session…)
# so VNC sessions on :1/:2/:101 keep working without manual edits.
if [ -z "${DISPLAY:-}" ]; then
    DETECTED_DISPLAY=$(ps -u "${DESKTOP_USER}" -o args= 2>/dev/null \
        | grep -oE 'DISPLAY=:[0-9]+' | head -1 | cut -d= -f2)
    if [ -z "${DETECTED_DISPLAY}" ]; then
        DETECTED_DISPLAY=$(ps -u "${DESKTOP_USER}" -o args= 2>/dev/null \
            | grep -oE '(Xtigervnc|Xvnc|Xorg) :[0-9]+' | head -1 | awk '{print $2}')
    fi
    export DISPLAY="${DETECTED_DISPLAY:-:0}"
fi

if [ -z "${XAUTHORITY:-}" ]; then
    for candidate in \
        "/home/${DESKTOP_USER}/.Xauthority" \
        "/home/${DESKTOP_USER}/.vnc/xauthority" \
        "/run/user/${USER_UID}/gdm/Xauthority"; do
        if [ -r "${candidate}" ]; then
            export XAUTHORITY="${candidate}"
            break
        fi
    done
fi

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/${USER_UID}}"

echo "[ run.sh ] DISPLAY=${DISPLAY} XAUTHORITY=${XAUTHORITY} (user=${DESKTOP_USER})"

cd "$SCRIPT_DIR"
exec python3 main.py "$@"
