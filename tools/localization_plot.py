#!/usr/bin/env python3
"""tools/localization_plot.py — visualización offline de pose_estimator.

Plan A1.1: standalone, sin acoplamiento al runtime. Lee:

  * ``temp/pose_estimator_debug.txt`` (legacy, formato F<idx>...) si existe.
  * ``temp/logs/run_*/brain.jsonl`` (post-E5) si se pasa ``--jsonl``.

y produce un PNG con trayectorias (raw vs fused vs matched), yaw vs time,
y correlación cross-source.

Inspirado en el ``localization_plot.py`` del REF (matplotlib offline),
sin meter ROS ni rosbag. Sólo Python + matplotlib.

Uso:
    python tools/localization_plot.py                  # autodetecta archivos
    python tools/localization_plot.py --jsonl temp/logs/run_<ts>/brain.jsonl
    python tools/localization_plot.py --debug-txt temp/pose_estimator_debug.txt
    python tools/localization_plot.py --out fig.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")  # backend off-screen — funciona en SSH/cron.
    import matplotlib.pyplot as plt
except ImportError:
    sys.stderr.write("ERROR: matplotlib no instalado. pip install matplotlib\n")
    sys.exit(1)


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


_FX_REGEX = re.compile(r"raw=\(([-\d.]+),([-\d.]+),([-\d.]+)\).*matched=\(([-\d.]+),([-\d.]+),([-\d.]+)\)")


def _parse_legacy_debug(path: Path) -> dict[str, list[float]]:
    """Parse ``temp/pose_estimator_debug.txt`` formato F<idx> | raw=(...) matched=(...) ..."""
    out: dict[str, list[float]] = {
        "idx": [],
        "raw_x": [],
        "raw_y": [],
        "raw_yaw": [],
        "matched_x": [],
        "matched_y": [],
        "matched_yaw": [],
    }
    idx = 0
    with path.open() as f:
        for line in f:
            m = _FX_REGEX.search(line)
            if not m:
                continue
            out["idx"].append(idx)
            out["raw_x"].append(float(m.group(1)))
            out["raw_y"].append(float(m.group(2)))
            out["raw_yaw"].append(float(m.group(3)))
            out["matched_x"].append(float(m.group(4)))
            out["matched_y"].append(float(m.group(5)))
            out["matched_yaw"].append(float(m.group(6)))
            idx += 1
    return out


def _parse_jsonl(path: Path) -> dict[str, list[float]]:
    """Parse ``brain.jsonl`` entries con topic LOCALISATION/LOCATION."""
    out: dict[str, list[float]] = {
        "ts": [],
        "raw_x": [],
        "raw_y": [],
        "raw_yaw": [],
        "matched_x": [],
        "matched_y": [],
        "matched_yaw": [],
        "fused_x": [],
        "fused_y": [],
        "fused_yaw": [],
    }
    with path.open() as f:
        for raw_line in f:
            try:
                rec = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            topic = rec.get("topic", "")
            payload = rec.get("payload") if isinstance(rec, dict) else None
            if not isinstance(payload, dict):
                continue
            # Topics: /auto/dashboard/localisation, /auto/threadtrafficcommunication/location
            if "loc" not in topic.lower():
                continue
            ts = payload.get("header", {}).get("timestamp") if isinstance(payload.get("header"), dict) else None
            if ts is None:
                ts = payload.get("timestamp", 0.0)
            try:
                out["ts"].append(float(ts))
            except (TypeError, ValueError):
                continue
            out["raw_x"].append(float(payload.get("raw_x", 0.0)))
            out["raw_y"].append(float(payload.get("raw_y", 0.0)))
            out["raw_yaw"].append(float(payload.get("raw_yaw", 0.0)))
            out["matched_x"].append(float(payload.get("matched_x", 0.0)))
            out["matched_y"].append(float(payload.get("matched_y", 0.0)))
            out["matched_yaw"].append(float(payload.get("matched_yaw", 0.0)))
            out["fused_x"].append(float(payload.get("x", 0.0)))
            out["fused_y"].append(float(payload.get("y", 0.0)))
            out["fused_yaw"].append(float(payload.get("yaw", 0.0)))
    return out


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def _plot(data: dict[str, list[float]], *, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_xy, ax_yaw, ax_error_xy, ax_error_yaw = (
        axes[0, 0],
        axes[0, 1],
        axes[1, 0],
        axes[1, 1],
    )
    ax_xy.set_title("Trajectory (XY)")
    ax_xy.plot(data["raw_x"], data["raw_y"], "-", color="C0", label="raw", linewidth=0.8)
    if data.get("fused_x"):
        ax_xy.plot(data["fused_x"], data["fused_y"], "-", color="C1", label="fused", linewidth=1.2)
    ax_xy.plot(
        data["matched_x"], data["matched_y"], "-", color="C2", label="matched", linewidth=0.8
    )
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.legend()
    ax_xy.grid(True, alpha=0.3)

    ax_yaw.set_title("Yaw vs time/idx")
    x_axis = data.get("ts") or data.get("idx") or list(range(len(data["raw_yaw"])))
    ax_yaw.plot(x_axis, data["raw_yaw"], "-", color="C0", label="raw_yaw")
    if data.get("fused_yaw"):
        ax_yaw.plot(x_axis, data["fused_yaw"], "-", color="C1", label="fused_yaw")
    ax_yaw.plot(x_axis, data["matched_yaw"], "-", color="C2", label="matched_yaw")
    ax_yaw.legend()
    ax_yaw.grid(True, alpha=0.3)
    ax_yaw.set_xlabel("t [s] o idx")
    ax_yaw.set_ylabel("yaw [deg]")

    # Error series
    err_xy = [
        math.hypot(r - m, ry - my)
        for r, m, ry, my in zip(
            data["raw_x"], data["matched_x"], data["raw_y"], data["matched_y"]
        )
    ]
    err_yaw = [
        abs(r - m) for r, m in zip(data["raw_yaw"], data["matched_yaw"])
    ]
    ax_error_xy.set_title(f"|raw - matched| XY  (RMS={_rms(err_xy):.3f} m)")
    ax_error_xy.plot(x_axis, err_xy, "-", color="C3")
    ax_error_xy.grid(True, alpha=0.3)
    ax_error_xy.set_xlabel("t [s] o idx")
    ax_error_xy.set_ylabel("error xy [m]")

    ax_error_yaw.set_title(f"|raw - matched| yaw  (RMS={_rms(err_yaw):.2f} deg)")
    ax_error_yaw.plot(x_axis, err_yaw, "-", color="C3")
    ax_error_yaw.grid(True, alpha=0.3)
    ax_error_yaw.set_xlabel("t [s] o idx")
    ax_error_yaw.set_ylabel("error yaw [deg]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[localization_plot] wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=None, help="brain.jsonl path")
    parser.add_argument("--debug-txt", type=Path, default=None, help="legacy pose_estimator_debug.txt")
    parser.add_argument("--out", type=Path, default=Path("temp/localization_plot.png"))
    args = parser.parse_args(argv)

    root = _resolve_repo_root()
    data: dict[str, list[float]] = {}
    if args.jsonl:
        data = _parse_jsonl(args.jsonl)
        if not data.get("ts"):
            sys.stderr.write(f"WARN: no Location records found in {args.jsonl}\n")
    elif args.debug_txt:
        data = _parse_legacy_debug(args.debug_txt)
    else:
        # Autodetect
        legacy = root / "temp" / "pose_estimator_debug.txt"
        if legacy.exists():
            data = _parse_legacy_debug(legacy)
            print(f"[localization_plot] using legacy {legacy}")
        else:
            # buscar el run más reciente
            candidates = sorted((root / "temp" / "logs").glob("run_*/brain.jsonl"))
            if not candidates:
                sys.stderr.write("ERROR: no input file found and no temp/logs/run_*/brain.jsonl\n")
                return 1
            data = _parse_jsonl(candidates[-1])
            print(f"[localization_plot] using jsonl {candidates[-1]}")

    if not data.get("raw_x"):
        sys.stderr.write("ERROR: no records parsed\n")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _plot(data, out_path=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
