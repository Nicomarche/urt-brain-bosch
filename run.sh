#!/bin/bash
# Launcher for urt-brain-bosch on Jetson Orin Nano (run as root)
# Usage: sudo ./run.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:/home/urt/.local/bin:$PATH"
export LD_LIBRARY_PATH="/home/urt/.local/lib/cusparselt:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH}"
export PYTHONPATH="/home/urt/.local/lib/python3.10/site-packages:${PYTHONPATH}"

cd "$SCRIPT_DIR"
exec python3 main.py "$@"
