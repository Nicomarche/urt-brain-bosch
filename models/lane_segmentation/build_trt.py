"""
build_trt.py (corre en la Jetson)

Genera un engine TensorRT *compatible con Ultralytics YOLO*.

Nota importante:
- Convertir ONNX->engine con `trtexec` puede perder metadata/NMS y luego el runtime
  de Ultralytics interpreta mal las salidas (clases "classXX", demasiadas cajas).
- Este script usa `YOLO.export(format="engine")` para conservar salida compatible.
"""

import shutil
from pathlib import Path


PT_MODEL = Path("best.pt")
ENGINE_OUT = Path("Best416px.engine")
IMGSZ = 416
USE_FP16 = True
DEVICE = 0  # Jetson GPU


def main():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics no instalado. Instala con: pip install ultralytics"
        ) from exc

    if not PT_MODEL.is_file():
        raise SystemExit(f"No existe el modelo base: {PT_MODEL}")

    print(f"Cargando: {PT_MODEL}")
    model = YOLO(str(PT_MODEL))

    print("Exportando TensorRT engine compatible con Ultralytics...")
    # nms=True es clave para evitar una salida cruda sin supresion.
    exported_path = model.export(
        format="engine",
        imgsz=IMGSZ,
        half=USE_FP16,
        device=DEVICE,
        nms=True,
    )
    exported_path = Path(str(exported_path))

    if not exported_path.is_file():
        raise SystemExit(f"Export fallo, archivo no encontrado: {exported_path}")

    if exported_path.resolve() != ENGINE_OUT.resolve():
        if ENGINE_OUT.exists():
            ENGINE_OUT.unlink()
        shutil.move(str(exported_path), str(ENGINE_OUT))

    print(f"Engine guardado en: {ENGINE_OUT.resolve()}")


if __name__ == "__main__":
    main()
