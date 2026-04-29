# Instalacion de Acados para el MPC completo

## 1. Dependencias previas

```bash
sudo apt-get install cmake gcc g++ liblapack-dev libblas-dev
pip install casadi numpy
```

## 2. Instalar Acados

### Opcion A: desde pip (mas simple, si hay wheel para tu plataforma)

```bash
pip install acados_template
```

Si no hay wheel para ARM (Raspberry Pi), usa la opcion B.

### Opcion B: compilar desde fuente (ARM / RPi 5)

```bash
cd ~
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init

mkdir -p build && cd build
cmake .. \
    -DACADOS_WITH_QPOASES=ON \
    -DACADOS_INSTALL_DIR=~/acados
make -j$(nproc)
make install

# Agregar al entorno
echo 'export ACADOS_SOURCE_DIR=~/acados' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/acados/lib' >> ~/.bashrc
source ~/.bashrc

# Instalar la interfaz Python
pip install -e ~/acados/interfaces/acados_template
```

## 3. Generar el solver

Desde la raiz del proyecto:

```bash
cd /path/to/urt-brain-bosch
python -m src.hardware.mpc.generate_solver
```

Opciones:
```bash
python -m src.hardware.mpc.generate_solver --N 30 --T 0.05        # 1.5s horizonte (default)
python -m src.hardware.mpc.generate_solver --N 40 --T 0.04        # 1.6s horizonte
python -m src.hardware.mpc.generate_solver --N 20 --T 0.05        # 1.0s (mas rapido en RPi)
python -m src.hardware.mpc.generate_solver --delta_max_deg 23      # limitar steering
```

Esto genera codigo C en `src/hardware/mpc/c_generated_code/` y lo compila en una shared library.

## 4. Activar el MPC completo

En `config.py`:

```python
USE_ACADOS_MPC = True    # activa el MPC completo (Acados)
USE_ACADOS_SPEED = False  # True para que el MPC tambien controle velocidad
```

Si `USE_ACADOS_MPC = True` pero Acados no esta instalado o el solver no fue generado, el sistema cae automaticamente al MPC lateral (scipy) o Stanley.

## 5. Verificar que funciona

```bash
python3 -c "
from src.hardware.mpc.acados_mpc import AcadosMPC
mpc = AcadosMPC()
print('Acados MPC ready:', mpc.ready)
if mpc.ready:
    print('Horizon N:', mpc.N)
"
```

## 6. Tuning de pesos

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
