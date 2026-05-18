#!/usr/bin/env python3
"""Reportar y reconstruir el ``annotated_video.avi`` de una sesión manual.

Uso::

    python tools/render_manual_annotated.py temp/logs/run_XXXX/manual_YYYY/

Por defecto:
    - Reporta el estado de la sesión (manifest, eventos, video raw/anotado).
    - Si ya existe ``annotated_video.avi`` (grabado por threadLocalPerception
      en runtime), no hace nada — ese es el escenario normal.
    - Si falta y existe ``raw_video.avi`` + ``per_topic/overlay_layer_msg.jsonl``,
      reconstruye un anotado best-effort dibujando las capas vectoriales sobre
      cada frame, ordenadas por timestamp más cercano.

``--force`` reconstruye aunque ya exista. ``--report`` sólo imprime el estado.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SessionReport:
    session_dir: Path
    manifest: Optional[dict]
    raw_video: Optional[Path]
    annotated_video: Optional[Path]
    events_lines: int
    per_topic_counts: dict


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def build_report(session_dir: Path) -> SessionReport:
    manifest_path = session_dir / "manifest.json"
    manifest: Optional[dict] = None
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except json.JSONDecodeError:
            manifest = None
    raw = session_dir / "raw_video.avi"
    annotated = session_dir / "annotated_video.avi"
    events = session_dir / "events.jsonl"
    per_topic_dir = session_dir / "per_topic"
    per_topic_counts: dict = {}
    if per_topic_dir.is_dir():
        for child in sorted(per_topic_dir.iterdir()):
            if child.suffix == ".jsonl":
                per_topic_counts[child.stem] = _count_lines(child)
    return SessionReport(
        session_dir=session_dir,
        manifest=manifest,
        raw_video=raw if raw.exists() else None,
        annotated_video=annotated if annotated.exists() else None,
        events_lines=_count_lines(events),
        per_topic_counts=per_topic_counts,
    )


def print_report(rep: SessionReport) -> None:
    print(f"Session: {rep.session_dir}")
    if rep.manifest:
        print(f"  start: {rep.manifest.get('started_at_iso')}")
        print(f"  end:   {rep.manifest.get('end_ts_iso')}")
        print(f"  reason:{rep.manifest.get('reason')}")
        print(f"  host:  {rep.manifest.get('host')}")
        print(f"  git:   {rep.manifest.get('git_sha')}")
    else:
        print("  manifest: MISSING")
    print(f"  raw_video:       {'OK' if rep.raw_video else 'MISSING'}")
    print(f"  annotated_video: {'OK' if rep.annotated_video else 'MISSING'}")
    print(f"  events.jsonl:    {rep.events_lines} lines")
    if rep.per_topic_counts:
        nonzero = {k: v for k, v in rep.per_topic_counts.items() if v > 0}
        print(f"  per_topic non-empty ({len(nonzero)}/{len(rep.per_topic_counts)} topics):")
        for name, count in sorted(nonzero.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {name}: {count}")
        if len(nonzero) > 10:
            print(f"    ... ({len(nonzero) - 10} more)")


def _load_overlay_events(path: Path) -> list[dict]:
    events: list[dict] = []
    if not path.exists():
        return events
    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _closest_overlay(overlays: list[dict], target_ts: float) -> Optional[dict]:
    if not overlays:
        return None
    best = overlays[0]
    best_dt = abs(best.get("ts", 0.0) - target_ts)
    for event in overlays[1:]:
        dt = abs(event.get("ts", 0.0) - target_ts)
        if dt < best_dt:
            best = event
            best_dt = dt
    return best


def reconstruct(session_dir: Path, output: Path, force: bool) -> int:
    """Genera ``annotated_video.avi`` re-pintando overlays sobre el raw.

    Best-effort: dibuja sólo lo que conoce (lane points, detected_objects
    bboxes). No replica fielmente la lógica del GUI; sirve para revisar
    sesiones donde threadLocalPerception no estaba corriendo.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        print("ERROR: opencv-python no instalado; no puedo reconstruir.")
        return 2

    raw_path = session_dir / "raw_video.avi"
    if not raw_path.exists():
        print(f"ERROR: falta {raw_path}; no hay nada que renderizar.")
        return 2
    if output.exists() and not force:
        print(f"{output} ya existe; usá --force para sobrescribir.")
        return 0

    overlay_path = session_dir / "per_topic" / "overlaylayermsg.jsonl"
    overlays = _load_overlay_events(overlay_path)
    objects_path = session_dir / "per_topic" / "detectedobjectsmsg.jsonl"
    detections = _load_overlay_events(objects_path)

    cap = cv2.VideoCapture(str(raw_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"ERROR: cv2.VideoWriter no abrió {output}")
        cap.release()
        return 2

    # Asumimos un timestamp inicial del manifest si está disponible. Si no,
    # alineamos por índice de frame y FPS — peor aproximación pero
    # determinística.
    manifest = build_report(session_dir).manifest or {}
    base_ts = float(manifest.get("started_at") or 0.0)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_ts = base_ts + (frame_idx / fps) if base_ts > 0 else frame_idx / fps

        ov_event = _closest_overlay(overlays, frame_ts)
        if ov_event is not None:
            payload = ov_event.get("payload") or {}
            layer = payload.get("layer") if isinstance(payload, dict) else None
            polylines = (payload.get("polylines") if isinstance(payload, dict) else None) or []
            for poly in polylines:
                pts = poly if isinstance(poly, list) else []
                if len(pts) >= 2:
                    try:
                        import numpy as np
                        arr = np.array([(int(p[0]), int(p[1])) for p in pts], dtype="int32")
                        cv2.polylines(frame, [arr], False, (0, 255, 0), 2)
                    except Exception:
                        pass
            _ = layer  # solo informativo, no se usa por ahora

        det_event = _closest_overlay(detections, frame_ts)
        if det_event is not None:
            payload = det_event.get("payload") or {}
            objects = payload.get("objects") if isinstance(payload, dict) else None
            if isinstance(objects, list):
                for obj in objects:
                    box = obj.get("box") if isinstance(obj, dict) else None
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        try:
                            x1, y1, x2, y2 = [int(v) for v in box[:4]]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        except Exception:
                            pass

        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()
    print(f"OK: wrote {frame_idx} frames to {output}")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session_dir", help="Path a temp/logs/run_*/manual_*/")
    p.add_argument("--report", action="store_true",
                   help="Sólo reportar estado, no reconstruir nada.")
    p.add_argument("--force", action="store_true",
                   help="Reconstruir annotated_video.avi aunque ya exista.")
    p.add_argument("--output", default=None,
                   help="Path de salida (default: <session_dir>/annotated_video.avi)")
    args = p.parse_args(argv)

    session_dir = Path(args.session_dir).resolve()
    if not session_dir.is_dir():
        print(f"ERROR: {session_dir} no es un directorio")
        return 2

    rep = build_report(session_dir)
    print_report(rep)
    if args.report:
        return 0
    output = Path(args.output) if args.output else (session_dir / "annotated_video.avi")
    if rep.annotated_video is not None and not args.force:
        print(f"\nannotated_video.avi ya existe en disco — usá --force para regenerar.")
        return 0
    return reconstruct(session_dir, output, force=args.force)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
