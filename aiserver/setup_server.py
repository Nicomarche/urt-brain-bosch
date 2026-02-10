"""
Script de setup para el AI Server de HybridNets.

Descarga el repositorio de HybridNets y los pesos pre-entrenados.

Uso:
  python setup_server.py
"""

import os
import subprocess
import sys
import urllib.request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HYBRIDNETS_DIR = os.path.join(BASE_DIR, "HybridNets")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "hybridnets.pth")

HYBRIDNETS_REPO = "https://github.com/datvuthanh/HybridNets.git"
WEIGHTS_URL = "https://github.com/datvuthanh/HybridNets/releases/download/v1.0/hybridnets.pth"


def run_cmd(cmd, cwd=None):
    """Ejecutar un comando y mostrar salida."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"  WARN: Comando retornó código {result.returncode}")
    return result.returncode


def clone_hybridnets():
    """Clonar el repositorio de HybridNets."""
    if os.path.isdir(HYBRIDNETS_DIR):
        print(f"[Setup] HybridNets ya existe en {HYBRIDNETS_DIR}")
        print("[Setup] Actualizando...")
        run_cmd("git pull", cwd=HYBRIDNETS_DIR)
    else:
        print(f"[Setup] Clonando HybridNets...")
        run_cmd(f"git clone {HYBRIDNETS_REPO} {HYBRIDNETS_DIR}")
    
    # Instalar dependencias del repo
    req_file = os.path.join(HYBRIDNETS_DIR, "requirements.txt")
    if os.path.exists(req_file):
        print("[Setup] Instalando dependencias de HybridNets...")
        run_cmd(f"{sys.executable} -m pip install -r {req_file}")


def download_weights():
    """Descargar pesos pre-entrenados."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    if os.path.exists(WEIGHTS_FILE):
        size_mb = os.path.getsize(WEIGHTS_FILE) / 1024 / 1024
        print(f"[Setup] Pesos ya existen: {WEIGHTS_FILE} ({size_mb:.1f} MB)")
        return
    
    print(f"[Setup] Descargando pesos pre-entrenados...")
    print(f"  URL: {WEIGHTS_URL}")
    print(f"  Destino: {WEIGHTS_FILE}")
    print("  (Esto puede tardar unos minutos...)")
    
    try:
        def progress_hook(count, block_size, total_size):
            percent = min(100, count * block_size * 100 / total_size)
            mb_done = count * block_size / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            sys.stdout.write(f"\r  Descargando: {percent:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_FILE, progress_hook)
        print()
        
        size_mb = os.path.getsize(WEIGHTS_FILE) / 1024 / 1024
        print(f"[Setup] Pesos descargados: {size_mb:.1f} MB")
    except Exception as e:
        print(f"\n[Setup] ERROR al descargar pesos: {e}")
        print(f"[Setup] Descarga manualmente desde: {WEIGHTS_URL}")
        print(f"[Setup] Y colócalo en: {WEIGHTS_FILE}")


def install_dependencies():
    """Instalar dependencias de Python."""
    print("[Setup] Instalando dependencias del servidor...")
    req_file = os.path.join(BASE_DIR, "requirements.txt")
    if os.path.exists(req_file):
        run_cmd(f"{sys.executable} -m pip install -r {req_file}")
    else:
        print("[Setup] WARN: requirements.txt no encontrado")


def verify_setup():
    """Verificar que todo está correctamente instalado."""
    print("\n" + "=" * 50)
    print("  Verificación de instalación")
    print("=" * 50)
    
    checks = []
    
    # PyTorch
    try:
        import torch
        gpu = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu else "N/A"
        print(f"  PyTorch: {torch.__version__} ({'CUDA' if gpu else 'CPU'}) {gpu_name}")
        checks.append(True)
    except ImportError:
        print("  PyTorch: NO INSTALADO")
        checks.append(False)
    
    # OpenCV
    try:
        import cv2
        print(f"  OpenCV: {cv2.__version__}")
        checks.append(True)
    except ImportError:
        print("  OpenCV: NO INSTALADO")
        checks.append(False)
    
    # FastAPI
    try:
        import fastapi
        print(f"  FastAPI: {fastapi.__version__}")
        checks.append(True)
    except ImportError:
        print("  FastAPI: NO INSTALADO")
        checks.append(False)
    
    # uvicorn
    try:
        import uvicorn
        print(f"  uvicorn: {uvicorn.__version__}")
        checks.append(True)
    except ImportError:
        print("  uvicorn: NO INSTALADO")
        checks.append(False)
    
    # websockets
    try:
        import websockets
        print(f"  websockets: {websockets.__version__}")
        checks.append(True)
    except ImportError:
        print("  websockets: NO INSTALADO")
        checks.append(False)
    
    # HybridNets repo
    if os.path.isdir(HYBRIDNETS_DIR):
        print(f"  HybridNets repo: OK")
        checks.append(True)
    else:
        print(f"  HybridNets repo: NO ENCONTRADO")
        checks.append(False)
    
    # Pesos
    if os.path.exists(WEIGHTS_FILE):
        size_mb = os.path.getsize(WEIGHTS_FILE) / 1024 / 1024
        print(f"  Pesos modelo: OK ({size_mb:.1f} MB)")
        checks.append(True)
    else:
        print(f"  Pesos modelo: NO ENCONTRADOS")
        checks.append(False)
    
    print()
    if all(checks):
        print("  [OK] Todo listo! Ejecuta: python server.py")
    else:
        print("  [!!] Algunas dependencias faltan. Revisa los mensajes arriba.")
    print()


def main():
    print("=" * 50)
    print("  HybridNets AI Server - Setup")
    print("=" * 50)
    print()
    
    # 1. Instalar dependencias de Python
    install_dependencies()
    print()
    
    # 2. Clonar HybridNets
    clone_hybridnets()
    print()
    
    # 3. Descargar pesos
    download_weights()
    print()
    
    # 4. Verificar
    verify_setup()


if __name__ == "__main__":
    main()
