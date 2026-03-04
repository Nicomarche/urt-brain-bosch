# build_trt.py (corre en la Jetson)
# Convierte Best416px.onnx -> Best416px.engine usando trtexec (TensorRT)
import subprocess
import shutil
from pathlib import Path

ONNX_MODEL = Path("Best416px.onnx")
ENGINE_OUT  = ONNX_MODEL.with_suffix(".engine")
WORKSPACE_MB = 4096   # 4 GiB

TRTEXEC = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"

cmd = [
    TRTEXEC,
    f"--onnx={ONNX_MODEL}",
    f"--saveEngine={ENGINE_OUT}",
    "--fp16",
    f"--memPoolSize=workspace:{WORKSPACE_MB}",
]

print("Ejecutando:", " ".join(cmd))
result = subprocess.run(cmd, check=True)
print("Engine guardado en:", ENGINE_OUT)

