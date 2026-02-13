"""
Script de setup para el motor Supercombo.

Descarga el modelo supercombo.onnx desde el repositorio de la comunidad
(MTammvee/openpilot-supercombo-model) y lo coloca en aiserver/models/.

Uso:
  cd aiserver
  python setup_supercombo.py

Requisitos:
  pip install onnxruntime>=1.16.0
"""

import os
import sys
import urllib.request
import hashlib


# ============================================================
# Configuración
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_FILENAME = "supercombo.onnx"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# URL de descarga (Git LFS redirect del repo de MTammvee)
MODEL_URL = "https://github.com/MTammvee/openpilot-supercombo-model/raw/main/supercombo.onnx"

# Tamaño esperado ~47MB
MIN_FILE_SIZE = 40 * 1024 * 1024  # 40MB mínimo


def download_model():
    """Descargar supercombo.onnx si no existe."""

    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        if os.path.getsize(MODEL_PATH) > MIN_FILE_SIZE:
            print(f"[Setup] Modelo ya existe: {MODEL_PATH} ({size_mb:.1f}MB)")
            return True
        else:
            print(f"[Setup] Modelo existe pero es muy pequeño ({size_mb:.1f}MB), re-descargando...")

    # Crear directorio
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"[Setup] Descargando supercombo.onnx...")
    print(f"[Setup] URL: {MODEL_URL}")
    print(f"[Setup] Destino: {MODEL_PATH}")

    try:
        # Descargar con barra de progreso simple
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r[Setup] Progreso: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=report_progress)
        print()  # Nueva línea después de la barra de progreso

        # Verificar tamaño
        size = os.path.getsize(MODEL_PATH)
        size_mb = size / (1024 * 1024)

        if size < MIN_FILE_SIZE:
            print(f"[Setup] ERROR: Archivo descargado muy pequeño ({size_mb:.1f}MB)")
            print(f"[Setup] Posible error de descarga. Intenta manualmente:")
            print(f"  wget {MODEL_URL} -O {MODEL_PATH}")
            os.remove(MODEL_PATH)
            return False

        print(f"[Setup] Descarga exitosa: {size_mb:.1f}MB")
        return True

    except Exception as e:
        print(f"\n[Setup] ERROR al descargar: {e}")
        print(f"[Setup] Intenta descargar manualmente:")
        print(f"  wget {MODEL_URL} -O {MODEL_PATH}")
        print(f"  # o")
        print(f"  curl -L {MODEL_URL} -o {MODEL_PATH}")
        return False


def verify_model():
    """Verificar que el modelo se puede cargar con ONNX Runtime."""
    print("\n[Setup] Verificando modelo...")

    try:
        import onnxruntime as ort
        print(f"[Setup] ONNX Runtime version: {ort.__version__}")
        print(f"[Setup] Providers disponibles: {ort.get_available_providers()}")
    except ImportError:
        print("[Setup] WARN: onnxruntime no instalado.")
        print("  pip install onnxruntime>=1.16.0")
        return False

    try:
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        print(f"\n[Setup] Modelo cargado exitosamente!")
        print(f"[Setup] Inputs:")
        for inp in inputs:
            print(f"  {inp.name}: {inp.shape} ({inp.type})")
        print(f"[Setup] Outputs:")
        for out in outputs:
            print(f"  {out.name}: {out.shape} ({out.type})")

        return True
    except Exception as e:
        print(f"[Setup] ERROR al verificar modelo: {e}")
        return False


def install_dependencies():
    """Instalar dependencias necesarias."""
    print("\n[Setup] Instalando dependencias...")

    deps = [
        "onnxruntime>=1.16.0",
        "fastapi",
        "uvicorn[standard]",
        "websockets",
        "opencv-python",
        "numpy",
    ]

    for dep in deps:
        print(f"  pip install {dep}")
        os.system(f"{sys.executable} -m pip install -q {dep}")


def main():
    print("=" * 60)
    print("  Setup Supercombo (openpilot) para AI Server")
    print("=" * 60)

    # 1. Descargar modelo
    if not download_model():
        print("\n[Setup] FALLO: No se pudo descargar el modelo")
        sys.exit(1)

    # 2. Verificar modelo
    if not verify_model():
        print("\n[Setup] WARN: No se pudo verificar el modelo (puede funcionar igual)")

    # 3. Resumen
    print("\n" + "=" * 60)
    print("  Setup completado!")
    print("=" * 60)
    print(f"  Modelo: {MODEL_PATH}")
    print(f"  Config: Edita config.py y setea ENGINE_TYPE = 'supercombo'")
    print(f"  Iniciar: python server.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
