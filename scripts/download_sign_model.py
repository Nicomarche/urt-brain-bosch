#!/usr/bin/env python3
"""
Download the MobilenetV2 SSD TFLite model for traffic sign detection.

Source: ricardolopezb/bfmc24-brain (GitHub)
The model detects 9 traffic sign classes:
  stop, parking, priority, crosswalk, highway_entrance,
  highway_exit, roundabout, one_way, no_entry

Usage:
    python scripts/download_sign_model.py

The model will be saved to: models/sign_detection/detect.tflite
"""

import os
import sys
import subprocess
import tempfile
import shutil

# GitHub repo info
REPO_URL = "https://github.com/ricardolopezb/bfmc24-brain.git"
MODEL_SRC_PATH = "src/austral/signals/mobilenet_model/detect.tflite"

# Local destination
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "sign_detection")
MODEL_DEST = os.path.join(MODEL_DIR, "detect.tflite")


def download_with_git_sparse():
    """Download the model using git sparse checkout (efficient, only downloads the model file)."""
    print(f"Descargando modelo desde {REPO_URL}...")
    print(f"  Archivo: {MODEL_SRC_PATH}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Initialize sparse checkout
            subprocess.run(
                ["git", "clone", "--no-checkout", "--depth", "1", REPO_URL, tmpdir],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "sparse-checkout", "init", "--cone"],
                cwd=tmpdir,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "sparse-checkout", "set", "src/austral/signals/mobilenet_model"],
                cwd=tmpdir,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "checkout"],
                cwd=tmpdir,
                check=True,
                capture_output=True,
                text=True,
            )

            src_file = os.path.join(tmpdir, MODEL_SRC_PATH)
            if not os.path.isfile(src_file):
                print(f"ERROR: Archivo no encontrado en el repo: {MODEL_SRC_PATH}")
                return False

            os.makedirs(MODEL_DIR, exist_ok=True)
            shutil.copy2(src_file, MODEL_DEST)

            size_mb = os.path.getsize(MODEL_DEST) / (1024 * 1024)
            print(f"Modelo descargado exitosamente!")
            print(f"  Destino: {MODEL_DEST}")
            print(f"  Tamaño:  {size_mb:.1f} MB")
            return True

        except subprocess.CalledProcessError as e:
            print(f"Error durante la descarga con git: {e}")
            if e.stderr:
                print(f"  stderr: {e.stderr.strip()}")
            return False


def download_with_curl():
    """Fallback: try downloading the raw file via GitHub's raw URL."""
    raw_url = f"https://raw.githubusercontent.com/ricardolopezb/bfmc24-brain/master/{MODEL_SRC_PATH}"
    print(f"Intentando descarga directa desde: {raw_url}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        subprocess.run(
            ["curl", "-L", "-o", MODEL_DEST, raw_url],
            check=True,
        )
        if os.path.isfile(MODEL_DEST) and os.path.getsize(MODEL_DEST) > 1000:
            size_mb = os.path.getsize(MODEL_DEST) / (1024 * 1024)
            print(f"Modelo descargado exitosamente!")
            print(f"  Destino: {MODEL_DEST}")
            print(f"  Tamaño:  {size_mb:.1f} MB")
            return True
        else:
            print("ERROR: El archivo descargado parece invalido (muy pequeño)")
            if os.path.isfile(MODEL_DEST):
                os.remove(MODEL_DEST)
            return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: curl no disponible o fallo la descarga")
        return False


def main():
    if os.path.isfile(MODEL_DEST):
        size_mb = os.path.getsize(MODEL_DEST) / (1024 * 1024)
        print(f"El modelo ya existe en: {MODEL_DEST} ({size_mb:.1f} MB)")
        response = input("¿Descargar de nuevo? (s/N): ").strip().lower()
        if response not in ("s", "si", "y", "yes"):
            print("Cancelado.")
            return

    # Try git sparse checkout first, then curl fallback
    if not download_with_git_sparse():
        print()
        print("Intentando metodo alternativo (curl)...")
        if not download_with_curl():
            print()
            print("=" * 60)
            print("No se pudo descargar el modelo automaticamente.")
            print("Descarga manual:")
            print(f"  1. Clona el repo: git clone {REPO_URL}")
            print(f"  2. Copia el archivo:")
            print(f"     cp bfmc24-brain/{MODEL_SRC_PATH} {MODEL_DEST}")
            print("=" * 60)
            sys.exit(1)

    print()
    print("Siguiente paso: instalar tflite-runtime")
    print("  pip install tflite-runtime")


if __name__ == "__main__":
    main()
