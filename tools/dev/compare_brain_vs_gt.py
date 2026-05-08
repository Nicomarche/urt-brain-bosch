#!/usr/bin/env python3
"""Compara la pose fusionada del brain contra el ground truth del simulador.

El brain publica pose de `base_link` en `brain_map`. El logger del simulador
guarda dos etapas:
  - pose raw Gazebo (`gz_x/y/yaw_rad`) del `model_frame`
  - pose adaptada a `brain_map` (`brain_map_x/y/yaw_rad`)

Este tool usa el mismo `FrameAdapter` compartido con `sim_bridge.py` para
reconstruir la pose raw de Gazebo desde la pose del brain y compararla 1:1
contra el ground truth del logger.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


BRAIN_ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = BRAIN_ROOT.parent / "urt-simulator"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from sim_bridge_frames import ModelPose2D, load_frame_adapter, wrap_angle_pi


def _load_jsonl(path: str, event_filter: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("event") == event_filter:
                out.append(data)
    return out


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="/Users/luciogarcia/urt-brain-bosch/temp/logs/latest",
        help="Directorio del run del brain (con brain.jsonl).",
    )
    parser.add_argument(
        "--frame-config",
        default=str(SIM_ROOT / "sim_bridge_frames.json"),
        help="path al JSON de frames del simulator",
    )
    parser.add_argument(
        "--max-skew-s",
        type=float,
        default=0.05,
        help="Máxima diferencia de timestamp permitida al alinear (default 50ms).",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Submuestreo: 1=todas, 10=cada 10 brain samples (default 1).",
    )
    args = parser.parse_args()

    adapter = load_frame_adapter(args.frame_config)

    brain_run = os.path.realpath(args.run_dir)
    brain_jsonl = os.path.join(brain_run, "brain.jsonl")
    if not os.path.exists(brain_jsonl):
        print(f"FATAL: no existe {brain_jsonl}", file=sys.stderr)
        return 1

    run_id = os.path.basename(brain_run)
    sim_jsonl = os.path.join(
        "/Users/luciogarcia/urt-simulator/temp/logs",
        run_id,
        "sim_bridge.jsonl",
    )
    if not os.path.exists(sim_jsonl):
        alt = os.path.join(brain_run, "sim_bridge.jsonl")
        if os.path.exists(alt):
            sim_jsonl = alt
        else:
            print(f"WARN: no existe {sim_jsonl} (tampoco {alt})", file=sys.stderr)

    brain_events = _load_jsonl(brain_jsonl, "pose_published")
    gt_events = _load_jsonl(sim_jsonl, "gz_pose") if os.path.exists(sim_jsonl) else []
    if not brain_events:
        print(f"FATAL: 0 pose_published events en {brain_jsonl}", file=sys.stderr)
        return 1
    if not gt_events:
        print("FATAL: 0 gz_pose events. ¿gz_pose_logger.py corrió durante el run?", file=sys.stderr)
        return 1

    gt_sorted = sorted(gt_events, key=lambda event: event["ts"])
    gt_ts = [event["ts"] for event in gt_sorted]

    print(f"# run_dir: {brain_run}")
    print(f"# brain pose_published: {len(brain_events)}")
    print(f"# gt gz_pose:           {len(gt_events)}")
    print(f"# every={args.every} max_skew_s={args.max_skew_s}")
    print(
        "# columns: ts | "
        "brain_map_x brain_map_y brain_map_yaw | "
        "gt_map_x gt_map_y gt_map_yaw | "
        "brain_gz_x brain_gz_y brain_gz_yaw | "
        "gt_gz_x gt_gz_y gt_gz_yaw | "
        "dxy_map dyaw_map_deg dxy_gz dyaw_gz_deg"
    )

    dxy_map_list: list[float] = []
    dyaw_map_list: list[float] = []
    dxy_gz_list: list[float] = []
    dyaw_gz_list: list[float] = []
    n_aligned = 0
    n_skipped = 0

    for idx, brain in enumerate(brain_events):
        if idx % args.every != 0:
            continue
        bts = float(brain["ts"])
        i = bisect.bisect_left(gt_ts, bts)
        candidates = []
        if i > 0:
            candidates.append(gt_sorted[i - 1])
        if i < len(gt_sorted):
            candidates.append(gt_sorted[i])
        if not candidates:
            n_skipped += 1
            continue
        gt = min(candidates, key=lambda event: abs(event["ts"] - bts))
        if abs(float(gt["ts"]) - bts) > args.max_skew_s:
            n_skipped += 1
            continue

        brain_map_x = _float_or_none(brain.get("fused_x"))
        brain_map_y = _float_or_none(brain.get("fused_y"))
        brain_map_yaw = _float_or_none(brain.get("fused_yaw_rad"))
        gt_gz_x = _float_or_none(gt.get("gz_x"))
        gt_gz_y = _float_or_none(gt.get("gz_y"))
        gt_gz_yaw = _float_or_none(gt.get("gz_yaw_rad"))
        gt_map_x = _float_or_none(gt.get("brain_map_x", gt.get("graphml_x")))
        gt_map_y = _float_or_none(gt.get("brain_map_y", gt.get("graphml_y")))
        gt_map_yaw = _float_or_none(gt.get("brain_map_yaw_rad", gt.get("graphml_yaw_rad")))
        if gt_map_x is None or gt_map_y is None or gt_map_yaw is None:
            if None not in (gt_gz_x, gt_gz_y, gt_gz_yaw):
                gt_model_pose = ModelPose2D.from_yaw(
                    x=float(gt_gz_x),
                    y=float(gt_gz_y),
                    z=float(gt.get("gz_z", 0.0) or 0.0),
                    yaw_rad=float(gt_gz_yaw),
                )
                gt_map_pose = adapter.adapt_model_pose(gt_model_pose)
                gt_map_x = float(gt_map_pose.x)
                gt_map_y = float(gt_map_pose.y)
                gt_map_yaw = float(gt_map_pose.yaw_rad)
        if any(
            value is None
            for value in (
                brain_map_x,
                brain_map_y,
                brain_map_yaw,
                gt_map_x,
                gt_map_y,
                gt_map_yaw,
                gt_gz_x,
                gt_gz_y,
                gt_gz_yaw,
            )
        ):
            n_skipped += 1
            continue

        brain_model_pose = adapter.brain_map_pose_to_model_pose(
            x=float(brain_map_x),
            y=float(brain_map_y),
            yaw_rad=float(brain_map_yaw),
        )
        brain_gz_x = float(brain_model_pose.x)
        brain_gz_y = float(brain_model_pose.y)
        brain_gz_yaw = wrap_angle_pi(float(brain_model_pose.yaw_rad))

        gt_map_yaw = wrap_angle_pi(float(gt_map_yaw))
        gt_gz_yaw = wrap_angle_pi(float(gt_gz_yaw))
        brain_map_yaw = wrap_angle_pi(float(brain_map_yaw))

        dxy_map = math.hypot(float(brain_map_x) - float(gt_map_x), float(brain_map_y) - float(gt_map_y))
        dyaw_map = wrap_angle_pi(brain_map_yaw - gt_map_yaw)
        dxy_gz = math.hypot(brain_gz_x - float(gt_gz_x), brain_gz_y - float(gt_gz_y))
        dyaw_gz = wrap_angle_pi(brain_gz_yaw - gt_gz_yaw)

        dxy_map_list.append(dxy_map)
        dyaw_map_list.append(abs(math.degrees(dyaw_map)))
        dxy_gz_list.append(dxy_gz)
        dyaw_gz_list.append(abs(math.degrees(dyaw_gz)))
        n_aligned += 1

        print(
            f"{bts:.3f} | "
            f"{float(brain_map_x):>+6.2f} {float(brain_map_y):>+6.2f} {math.degrees(brain_map_yaw):>+7.1f} | "
            f"{float(gt_map_x):>+6.2f} {float(gt_map_y):>+6.2f} {math.degrees(gt_map_yaw):>+7.1f} | "
            f"{brain_gz_x:>+6.2f} {brain_gz_y:>+6.2f} {math.degrees(brain_gz_yaw):>+7.1f} | "
            f"{float(gt_gz_x):>+6.2f} {float(gt_gz_y):>+6.2f} {math.degrees(gt_gz_yaw):>+7.1f} | "
            f"{dxy_map:>6.3f} {math.degrees(dyaw_map):>+7.2f} {dxy_gz:>6.3f} {math.degrees(dyaw_gz):>+7.2f}"
        )

    print()
    print(f"# aligned={n_aligned} skipped={n_skipped}")
    if dxy_map_list:
        print(
            f"# brain_map dxy_m: mean={sum(dxy_map_list)/len(dxy_map_list):.3f} "
            f"p50={_percentile(dxy_map_list, 50):.3f} "
            f"p90={_percentile(dxy_map_list, 90):.3f} "
            f"max={max(dxy_map_list):.3f}"
        )
    if dyaw_map_list:
        print(
            f"# brain_map |dyaw|_deg: mean={sum(dyaw_map_list)/len(dyaw_map_list):.2f} "
            f"p50={_percentile(dyaw_map_list, 50):.2f} "
            f"p90={_percentile(dyaw_map_list, 90):.2f} "
            f"max={max(dyaw_map_list):.2f}"
        )
    if dxy_gz_list:
        print(
            f"# gz_frame dxy_m: mean={sum(dxy_gz_list)/len(dxy_gz_list):.3f} "
            f"p50={_percentile(dxy_gz_list, 50):.3f} "
            f"p90={_percentile(dxy_gz_list, 90):.3f} "
            f"max={max(dxy_gz_list):.3f}"
        )
    if dyaw_gz_list:
        print(
            f"# gz_frame |dyaw|_deg: mean={sum(dyaw_gz_list)/len(dyaw_gz_list):.2f} "
            f"p50={_percentile(dyaw_gz_list, 50):.2f} "
            f"p90={_percentile(dyaw_gz_list, 90):.2f} "
            f"max={max(dyaw_gz_list):.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
