"""
build_trt.py (corre en la Jetson)

Genera un engine TensorRT *compatible con Ultralytics YOLO*.

Uso:
  python build_trt.py                          # convierte el modelo por defecto (SRC_MODEL)
  python build_trt.py "Best weights_reentrenado.onnx"  # convierte un modelo especifico

Nota:
- Usa `ultralytics.utils.export.engine.onnx2engine` internamente, la misma funcion que
  usa `YOLO.export(format='engine')` desde .pt, por lo que el engine generado es 100%
  compatible con el runtime de Ultralytics (metadatos, nombres de clases, tarea, etc.).
- Soporta tanto .pt (PyTorch) como .onnx como modelo fuente.
  Para .pt se exporta primero a ONNX y luego al engine.
"""

import ast
import json
import shutil
import sys
from pathlib import Path


# ── Modelo fuente por defecto ─────────────────────────────────────────────────
SRC_MODEL = Path("Best weights_reentrenado.onnx")

USE_FP16 = True
DEVICE = 0  # Jetson GPU (se necesita para .pt; ignorado en el path ONNX directo)
WORKSPACE_GB = 4  # Memoria de workspace para TensorRT (GB)


def _read_onnx_metadata(onnx_path: Path) -> dict:
    """Extrae metadatos del ONNX y devuelve un dict compatible con Ultralytics."""
    import onnx  # type: ignore

    model = onnx.load(str(onnx_path))

    raw = {p.key: p.value for p in model.metadata_props}

    # Dimensiones de entrada [N, C, H, W]
    dims = model.graph.input[0].type.tensor_type.shape.dim
    imgsz = [dims[2].dim_value, dims[3].dim_value]

    # Parsear 'names' desde string tipo "{0: 'car', 1: 'crosswalk', ...}"
    names_raw = raw.get("names", "{}")
    try:
        names = ast.literal_eval(names_raw)
    except Exception:
        names = json.loads(names_raw)

    # Parsear 'args' si existe
    args_raw = raw.get("args", "{}")
    try:
        args = ast.literal_eval(args_raw)
    except Exception:
        try:
            args = json.loads(args_raw)
        except Exception:
            args = {}

    metadata = {
        "description": raw.get("description", f"Converted from {onnx_path.name}"),
        "author":      raw.get("author", "Ultralytics"),
        "date":        raw.get("date", ""),
        "version":     raw.get("version", ""),
        "license":     raw.get("license", ""),
        "docs":        raw.get("docs", ""),
        "stride":      int(raw.get("stride", 32)),
        "task":        raw.get("task", "segment"),
        "batch":       int(raw.get("batch", 1)),
        "imgsz":       imgsz,
        "names":       names,
        "args":        args,
        "channels":    int(raw.get("channels", 3)),
        "end2end":     raw.get("end2end", "False").lower() == "true",
    }

    return metadata


def _engine_name_from_src(src: Path, imgsz: int) -> Path:
    """Genera un nombre de engine limpio (sin espacios) basado en el archivo fuente."""
    stem = src.stem.replace(" ", "_")
    return Path(f"{stem}_{imgsz}px.engine")


def main():
    src_model = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC_MODEL

    if not src_model.is_file():
        raise SystemExit(f"No existe el modelo fuente: {src_model}")

    ext = src_model.suffix.lower()
    if ext not in (".pt", ".onnx"):
        raise SystemExit(f"Formato no soportado: '{ext}'. Usa .pt o .onnx")

    # ── Ruta ONNX directa ────────────────────────────────────────────────────
    if ext == ".onnx":
        try:
            from ultralytics.utils.export.engine import onnx2engine
        except ImportError as exc:
            raise SystemExit(
                "ultralytics>=8.3 no instalado o version sin onnx2engine. "
                "Instala con: pip install 'ultralytics>=8.3'"
            ) from exc

        print(f"Leyendo metadatos desde ONNX: {src_model}")
        metadata = _read_onnx_metadata(src_model)
        imgsz = metadata["imgsz"][0]
        task = metadata["task"]
        names = metadata["names"]
        engine_out = _engine_name_from_src(src_model, imgsz)

        print(f"\nFuente  : {src_model}")
        print(f"Tarea   : {task}")
        print(f"Tamaño  : {imgsz}px")
        print(f"Clases  : {len(names)} → {list(names.values())}")
        print(f"Salida  : {engine_out}")
        print(f"FP16    : {USE_FP16}")
        print(f"\nConstruyendo engine TensorRT (esto puede tardar varios minutos)...")

        onnx2engine(
            onnx_file=str(src_model),
            engine_file=str(engine_out),
            workspace=WORKSPACE_GB,
            half=USE_FP16,
            shape=(1, 3, imgsz, imgsz),
            metadata=metadata,
        )

    # ── Ruta desde .pt (comportamiento original) ─────────────────────────────
    else:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit(
                "ultralytics no instalado. Instala con: pip install ultralytics"
            ) from exc

        imgsz = 416
        engine_out = _engine_name_from_src(src_model, imgsz)

        print(f"Fuente  : {src_model}")
        print(f"Tamaño  : {imgsz}px")
        print(f"Salida  : {engine_out}")
        print(f"FP16    : {USE_FP16}")
        print(f"\nCargando modelo PyTorch y exportando a TensorRT...")

        model = YOLO(str(src_model))
        exported_path = model.export(
            format="engine",
            imgsz=imgsz,
            half=USE_FP16,
            device=DEVICE,
            nms=True,
        )
        exported_path = Path(str(exported_path))

        if not exported_path.is_file():
            raise SystemExit(f"Export falló, archivo no encontrado: {exported_path}")

        if exported_path.resolve() != engine_out.resolve():
            if engine_out.exists():
                engine_out.unlink()
            shutil.move(str(exported_path), str(engine_out))

    if not engine_out.is_file():
        raise SystemExit(f"Algo salió mal: el engine no existe en {engine_out}")

    size_mb = engine_out.stat().st_size / 1e6
    print(f"\nEngine guardado: {engine_out.resolve()} ({size_mb:.1f} MB)")
    print(f"\nActualiza config.py con:")
    print(f'  LOCAL_AI_MODEL_PATH = "models/lane_segmentation/{engine_out.name}"')
    print(f'  LOCAL_AI_IMGSZ = {imgsz}')


if __name__ == "__main__":
    main()
