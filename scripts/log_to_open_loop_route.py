#!/usr/bin/env python3
"""Convert a manual-drive logger run into an open-loop mission JSON.

Why this exists
---------------
When AUTO is unreliable, the quickest fallback is often:

1. drive a section manually while the notebook logger is running,
2. extract the raw commanded speed/steer timeline,
3. replay that exact timeline later from the GUI's scripted-route pad.

This script converts ``timeseries.csv`` from ``manual_bus_logger.py`` into
``missions/open_loop_route.json``-compatible JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Sample:
    t_s: float
    speed_cms: float
    steer_deg: float


@dataclass
class Segment:
    start_s: float
    end_s: float
    speed_cms: float
    steer_deg: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def _resolve_csv(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_dir():
        candidate = path / "timeseries.csv"
        if candidate.exists():
            return candidate
    return path


def _round_to_step(value: float, step: float) -> float:
    if step <= 0.0:
        return value
    return round(value / step) * step


def _load_samples(
    csv_path: Path,
    *,
    source_speed: str,
    source_steer: str,
    from_s: float | None,
    to_s: float | None,
    speed_step: float,
    steer_step: float,
) -> list[Sample]:
    speed_col = "cmd_speed_cms" if source_speed == "cmd" else "feedback_speed_cms"
    steer_col = "cmd_steer_deg_wire" if source_steer == "cmd" else "feedback_steer_deg_wire"

    out: list[Sample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            t_s = _safe_float(row.get("logger_elapsed_s"))
            if t_s is None:
                continue
            if from_s is not None and t_s < from_s:
                continue
            if to_s is not None and t_s > to_s:
                continue

            speed = _safe_float(row.get(speed_col))
            steer = _safe_float(row.get(steer_col))
            if speed is None:
                speed = 0.0
            if steer is None:
                steer = 0.0

            out.append(
                Sample(
                    t_s=t_s,
                    speed_cms=_round_to_step(speed, speed_step),
                    steer_deg=_round_to_step(steer, steer_step),
                )
            )
    return out


def _collapse_segments(samples: list[Sample], *, min_segment_s: float) -> list[Segment]:
    if not samples:
        return []

    raw_segments: list[Segment] = []
    start = samples[0]
    last = samples[0]

    for sample in samples[1:]:
        if sample.speed_cms == last.speed_cms and sample.steer_deg == last.steer_deg:
            last = sample
            continue

        raw_segments.append(
            Segment(
                start_s=start.t_s,
                end_s=sample.t_s,
                speed_cms=start.speed_cms,
                steer_deg=start.steer_deg,
            )
        )
        start = sample
        last = sample

    raw_segments.append(
        Segment(
            start_s=start.t_s,
            end_s=last.t_s,
            speed_cms=start.speed_cms,
            steer_deg=start.steer_deg,
        )
    )

    merged: list[Segment] = []
    for seg in raw_segments:
        if seg.duration_s >= min_segment_s:
            merged.append(seg)
            continue
        if merged:
            merged[-1].end_s = max(merged[-1].end_s, seg.end_s)
            continue
        # If the very first segment is too short, keep it only if there
        # is nothing else to merge into later.
        merged.append(seg)
    return merged


def _trim_leading_and_trailing_idle(
    segments: list[Segment],
    *,
    drop_idle: bool,
    movement_speed_threshold_cms: float,
    movement_steer_threshold_deg: float,
) -> list[Segment]:
    if not drop_idle or not segments:
        return segments

    def is_idle(seg: Segment) -> bool:
        return (
            abs(seg.speed_cms) < movement_speed_threshold_cms
            and abs(seg.steer_deg) < movement_steer_threshold_deg
        )

    start = 0
    end = len(segments)
    while start < end and is_idle(segments[start]):
        start += 1
    while end > start and is_idle(segments[end - 1]):
        end -= 1
    return segments[start:end]


def _segment_label(idx: int, seg: Segment) -> str:
    speed = seg.speed_cms
    steer = seg.steer_deg
    if abs(speed) < 1e-9 and abs(steer) < 1e-9:
        return f"Neutral {idx}"
    if abs(speed) < 1e-9:
        side = "Right" if steer > 0 else "Left"
        return f"{side} steer hold {idx}"
    if abs(steer) < 1e-9:
        direction = "Reverse" if speed < 0 else "Straight"
        return f"{direction} {idx}"
    arc_side = "Right" if steer > 0 else "Left"
    direction = "Reverse " if speed < 0 else ""
    return f"{direction}{arc_side} arc {idx}"


def _build_mission_payload(
    segments: list[Segment],
    *,
    name: str,
    description: str,
    final_brake: bool,
    add_neutral_prefix_s: float,
    add_neutral_suffix_s: float,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    if add_neutral_prefix_s > 0.0:
        steps.append(
            {
                "label": "Arm neutral",
                "duration_s": round(add_neutral_prefix_s, 3),
                "speed_cms": 0.0,
                "steer_deg": 0.0,
            }
        )

    for idx, seg in enumerate(segments, start=1):
        steps.append(
            {
                "label": _segment_label(idx, seg),
                "duration_s": round(seg.duration_s, 3),
                "speed_cms": round(seg.speed_cms, 3),
                "steer_deg": round(seg.steer_deg, 3),
            }
        )

    if add_neutral_suffix_s > 0.0:
        steps.append(
            {
                "label": "Stop neutral",
                "duration_s": round(add_neutral_suffix_s, 3),
                "speed_cms": 0.0,
                "steer_deg": 0.0,
            }
        )

    return {
        "name": name,
        "description": description,
        "final_brake": final_brake,
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert manual_bus_logger timeseries.csv into an open-loop "
            "mission JSON for the Scripted Route button."
        )
    )
    parser.add_argument(
        "input",
        help="Path to a run directory or directly to timeseries.csv",
    )
    parser.add_argument(
        "--output",
        default="missions/open_loop_route.json",
        help="Output mission JSON path (default: missions/open_loop_route.json).",
    )
    parser.add_argument(
        "--name",
        default="Open-Loop Route From Log",
        help="Mission name.",
    )
    parser.add_argument(
        "--description",
        default="Generated from a manual-drive logger run.",
        help="Mission description.",
    )
    parser.add_argument(
        "--source-speed",
        choices=("cmd", "feedback"),
        default="cmd",
        help="Use commanded or feedback speed as mission source (default: cmd).",
    )
    parser.add_argument(
        "--source-steer",
        choices=("cmd", "feedback"),
        default="cmd",
        help="Use commanded or feedback steer as mission source (default: cmd).",
    )
    parser.add_argument(
        "--from-s",
        type=float,
        default=None,
        help="Trim samples before this logger elapsed time.",
    )
    parser.add_argument(
        "--to-s",
        type=float,
        default=None,
        help="Trim samples after this logger elapsed time.",
    )
    parser.add_argument(
        "--min-segment-s",
        type=float,
        default=0.35,
        help="Merge away segments shorter than this duration (default: 0.35s).",
    )
    parser.add_argument(
        "--speed-step",
        type=float,
        default=0.5,
        help="Quantize speed to this step in cm/s before segmenting (default: 0.5).",
    )
    parser.add_argument(
        "--steer-step",
        type=float,
        default=2.5,
        help="Quantize steer to this step in degrees before segmenting (default: 2.5).",
    )
    parser.add_argument(
        "--keep-leading-idle",
        action="store_true",
        help="Keep neutral segments at the beginning and end instead of trimming them.",
    )
    parser.add_argument(
        "--idle-speed-threshold-cms",
        type=float,
        default=0.25,
        help="Below this speed a segment counts as idle when trimming (default: 0.25).",
    )
    parser.add_argument(
        "--idle-steer-threshold-deg",
        type=float,
        default=0.25,
        help="Below this steer a segment counts as idle when trimming (default: 0.25).",
    )
    parser.add_argument(
        "--neutral-prefix-s",
        type=float,
        default=0.3,
        help="Optional neutral step inserted before replay starts (default: 0.3).",
    )
    parser.add_argument(
        "--neutral-suffix-s",
        type=float,
        default=0.5,
        help="Optional neutral step inserted after replay ends (default: 0.5).",
    )
    parser.add_argument(
        "--no-final-brake",
        action="store_true",
        help="Do not request a final brake at mission completion.",
    )
    args = parser.parse_args()

    csv_path = _resolve_csv(args.input)
    if not csv_path.exists():
        raise SystemExit(f"timeseries CSV not found: {csv_path}")

    samples = _load_samples(
        csv_path,
        source_speed=args.source_speed,
        source_steer=args.source_steer,
        from_s=args.from_s,
        to_s=args.to_s,
        speed_step=args.speed_step,
        steer_step=args.steer_step,
    )
    if not samples:
        raise SystemExit("no samples found in the selected range")

    segments = _collapse_segments(samples, min_segment_s=args.min_segment_s)
    segments = _trim_leading_and_trailing_idle(
        segments,
        drop_idle=not args.keep_leading_idle,
        movement_speed_threshold_cms=args.idle_speed_threshold_cms,
        movement_steer_threshold_deg=args.idle_steer_threshold_deg,
    )
    if not segments:
        raise SystemExit("no non-idle segments remained after trimming")

    payload = _build_mission_payload(
        segments,
        name=args.name,
        description=args.description,
        final_brake=not args.no_final_brake,
        add_neutral_prefix_s=args.neutral_prefix_s,
        add_neutral_suffix_s=args.neutral_suffix_s,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    total_s = sum(float(step["duration_s"]) for step in payload["steps"])
    print(f"[log_to_open_loop_route] csv={csv_path}")
    print(f"[log_to_open_loop_route] segments={len(segments)} steps={len(payload['steps'])}")
    print(f"[log_to_open_loop_route] total_duration_s={total_s:.3f}")
    print(f"[log_to_open_loop_route] output={output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
