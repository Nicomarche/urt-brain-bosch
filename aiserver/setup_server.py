"""
Script de setup para el AI Server de HybridNets.

Descarga el repositorio de HybridNets y los pesos pre-entrenados.
Crea un entorno virtual (venv) para las dependencias de Python.

Uso:
  python3 setup_server.py
"""

import os
import subprocess
import sys
import urllib.request
import venv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, "venv")
HYBRIDNETS_DIR = os.path.join(BASE_DIR, "HybridNets")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "hybridnets.pth")

HYBRIDNETS_REPO = "https://github.com/datvuthanh/HybridNets.git"
WEIGHTS_URL = "https://github.com/datvuthanh/HybridNets/releases/download/v1.0/hybridnets.pth"


def get_venv_python():
    """Obtener la ruta al ejecutable de Python dentro del venv."""
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python3")


def get_venv_pip():
    """Obtener la ruta al ejecutable de pip dentro del venv."""
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", "pip.exe")
    return os.path.join(VENV_DIR, "bin", "pip3")


def run_cmd(cmd, cwd=None):
    """Ejecutar un comando y mostrar salida."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"  WARN: Comando retornó código {result.returncode}")
    return result.returncode


def create_venv():
    """Crear entorno virtual si no existe."""
    if os.path.isdir(VENV_DIR) and os.path.exists(get_venv_python()):
        print(f"[Setup] Entorno virtual ya existe en {VENV_DIR}")
        return
    
    print(f"[Setup] Creando entorno virtual en {VENV_DIR}...")
    venv.create(VENV_DIR, with_pip=True)
    
    # Actualizar pip dentro del venv
    venv_pip = get_venv_pip()
    venv_python = get_venv_python()
    run_cmd(f"{venv_python} -m pip install --upgrade pip")
    
    print(f"[Setup] Entorno virtual creado correctamente")


def clone_hybridnets():
    """Clonar el repositorio de HybridNets."""
    if os.path.isdir(HYBRIDNETS_DIR):
        print(f"[Setup] HybridNets ya existe en {HYBRIDNETS_DIR}")
        print("[Setup] Actualizando...")
        run_cmd("git pull", cwd=HYBRIDNETS_DIR)
    else:
        print(f"[Setup] Clonando HybridNets...")
        run_cmd(f"git clone {HYBRIDNETS_REPO} {HYBRIDNETS_DIR}")
    
    # Instalar dependencias del repo en el venv
    req_file = os.path.join(HYBRIDNETS_DIR, "requirements.txt")
    if os.path.exists(req_file):
        print("[Setup] Instalando dependencias de HybridNets en venv...")
        venv_pip = get_venv_pip()
        run_cmd(f"{venv_pip} install -r {req_file}")


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
    """Instalar dependencias de Python en el venv."""
    print("[Setup] Instalando dependencias del servidor en venv...")
    req_file = os.path.join(BASE_DIR, "requirements.txt")
    if os.path.exists(req_file):
        venv_pip = get_venv_pip()
        run_cmd(f"{venv_pip} install -r {req_file}")
    else:
        print("[Setup] WARN: requirements.txt no encontrado")


def verify_setup():
    """Verificar que todo está correctamente instalado usando el venv."""
    print("\n" + "=" * 50)
    print("  Verificación de instalación")
    print("=" * 50)
    
    venv_python = get_venv_python()
    checks = []
    
    def check_package(name, import_name=None):
        """Verificar si un paquete está instalado en el venv."""
        import_name = import_name or name
        cmd = f'{venv_python} -c "import {import_name}; print({import_name}.__version__)"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, version
        return False, None
    
    # PyTorch
    cmd = f'{venv_python} -c "import torch; gpu=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if gpu else \'N/A\'; print(torch.__version__, \'CUDA\' if gpu else \'CPU\', name)"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  PyTorch: {result.stdout.strip()}")
        checks.append(True)
    else:
        print("  PyTorch: NO INSTALADO")
        checks.append(False)
    
    # OpenCV
    ok, ver = check_package("OpenCV", "cv2")
    print(f"  OpenCV: {ver}" if ok else "  OpenCV: NO INSTALADO")
    checks.append(ok)
    
    # FastAPI
    ok, ver = check_package("FastAPI", "fastapi")
    print(f"  FastAPI: {ver}" if ok else "  FastAPI: NO INSTALADO")
    checks.append(ok)
    
    # uvicorn
    ok, ver = check_package("uvicorn", "uvicorn")
    print(f"  uvicorn: {ver}" if ok else "  uvicorn: NO INSTALADO")
    checks.append(ok)
    
    # websockets
    ok, ver = check_package("websockets", "websockets")
    print(f"  websockets: {ver}" if ok else "  websockets: NO INSTALADO")
    checks.append(ok)
    
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
    
    # Info del venv
    print(f"\n  Entorno virtual: {VENV_DIR}")
    print(f"  Python del venv: {venv_python}")
    
    print()
    if all(checks):
        print("  [OK] Todo listo!")
        print(f"  Para ejecutar el servidor:")
        if sys.platform == "win32":
            print(f"    & \"{os.path.join(VENV_DIR, 'Scripts', 'Activate.ps1')}\"")
        else:
            print(f"    source {VENV_DIR}/bin/activate")
        print(f"    python server.py")
    else:
        print("  [!!] Algunas dependencias faltan. Revisa los mensajes arriba.")
    print()


def main():
    print("=" * 50)
    print("  HybridNets AI Server - Setup")
    print("=" * 50)
    print()
    
    # 1. Crear entorno virtual
    create_venv()
    print()
    
    # 2. Instalar dependencias de Python en el venv
    install_dependencies()
    print()
    
    # 3. Clonar HybridNets
    clone_hybridnets()
    print()
    
    # 4. Descargar pesos
    download_weights()
    print()
    
    # 5. Verificar
    verify_setup()


if __name__ == "__main__":
    main()
