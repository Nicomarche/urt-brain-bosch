# Instalacion de Acados para el MPC completo

## macOS Apple Silicon (desarrollo/simulador)

```bash
# 1. Clonar y compilar acados
cd ~
git clone https://github.com/acados/acados.git
cd acados
git submodule update --init --recursive
mkdir -p build && cd build
cmake .. -DACADOS_WITH_QPOASES=ON -DACADOS_INSTALL_DIR=~/acados_install -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
make install
cp ~/acados/lib/link_libs.json ~/acados_install/lib/

# 2. Descargar tera renderer (ARM64)
mkdir -p ~/acados_install/bin
curl -L "https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-osx-arm64" \
     -o ~/acados_install/bin/t_renderer
chmod +x ~/acados_install/bin/t_renderer

# 3. Instalar la interfaz Python
pip3.13 install vcs-versioning --break-system-packages
pip3.13 install ~/acados/interfaces/acados_template --no-deps --break-system-packages
pip3.13 install casadi --break-system-packages

# 4. Generar el solver (desde la raiz del proyecto)
ACADOS_SOURCE_DIR=~/acados_install \
DYLD_LIBRARY_PATH=~/acados_install/lib \
KMP_DUPLICATE_LIB_OK=TRUE \
python3.13 -m src.control._acados_solver_gen
```

`run.sh` ya exporta `ACADOS_SOURCE_DIR`, `DYLD_LIBRARY_PATH` y `KMP_DUPLICATE_LIB_OK`
automaticamente en macOS — no hace falta hacer nada más para correr.

---

## Jetson (produccion)

### Dependencias previas

```bash
sudo apt-get install cmake gcc g++ liblapack-dev libblas-dev
pip install casadi numpy
```

### Instalar Acados

```bash
cd ~
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init
mkdir -p build && cd build
cmake .. -DACADOS_WITH_QPOASES=ON -DACADOS_INSTALL_DIR=~/acados
make -j$(nproc)
make install
pip install vcs-versioning
pip install -e ~/acados/interfaces/acados_template
echo 'export ACADOS_SOURCE_DIR=~/acados' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/acados/lib' >> ~/.bashrc
source ~/.bashrc
```

### Generar el solver

```bash
cd /path/to/urt-brain-bosch
python -m src.control._acados_solver_gen
```

Opciones:
```bash
python -m src.control._acados_solver_gen --N 30 --T 0.05        # 1.5s horizonte (default)
python -m src.control._acados_solver_gen --N 40 --T 0.04        # 1.6s horizonte
python -m src.control._acados_solver_gen --N 20 --T 0.05        # 1.0s (mas rapido en RPi)
python -m src.control._acados_solver_gen --delta_max_deg 23      # limitar steering
```

Esto genera codigo C en `src/control/c_generated_code/` y lo compila en una shared library.

---

## Verificar que funciona

```bash
python3 -c "
from src.control.motion_controller import AcadosMPC
mpc = AcadosMPC()
print('Acados MPC ready:', mpc.ready)
if mpc.ready:
    print('Horizon N:', mpc._N)
"
```

---

## Configuracion en config.py

```python
USE_ACADOS_MPC = True    # activa el MPC completo (Acados)
USE_ACADOS_SPEED = False  # True para que el MPC tambien controle velocidad
```

Si `USE_ACADOS_MPC = True` pero Acados no esta instalado o el solver no fue generado,
el sistema cae automaticamente al PurePursuitSolver.

---

## Tuning de pesos

Los pesos se pueden cambiar en `config.py` sin regenerar el solver:

```python
ACADOS_MPC_X_COST = 2.0           # seguimiento de posicion X
ACADOS_MPC_Y_COST = 2.0           # seguimiento de posicion Y
ACADOS_MPC_YAW_COST = 0.5         # seguimiento de heading
ACADOS_MPC_V_COST = 1.0           # penaliza desviacion de v_ref
ACADOS_MPC_STEER_COST = 0.0       # penaliza steering (0 = libre)
ACADOS_MPC_DELTA_V_COST = 1.5     # penaliza cambios de velocidad
ACADOS_MPC_DELTA_STEER_COST = 0.75  # penaliza cambios de steering
```

Solo necesitas regenerar el solver si cambias: `N`, `T`, `wheelbase`, `l_r`, o los limites de control.
