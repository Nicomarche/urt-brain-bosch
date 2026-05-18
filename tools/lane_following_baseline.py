#!/usr/bin/env python3
"""Analiza un run guardado en `temp/logs/<run>/brain.jsonl` y contrasta las
hipótesis H1-H5 del plan de "el auto no sigue el carril".

Uso:
    python tools/lane_following_baseline.py temp/logs/latest/brain.jsonl
    python tools/lane_following_baseline.py temp/logs/latest/brain.jsonl --json

El script NO modifica nada — sólo lee el live_log y escribe a stdout (o a un
markdown si se pasa `--out`). Pensado para correrse después de cada run de
calibración para comparar contra baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _iter_events(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def analyze(path: Path) -> dict[str, Any]:
    mpc_in_tp_dist: list[float] = []
    mpc_in_tp_heading: list[float] = []
    mpc_out_delta_raw: list[float] = []
    mpc_out_delta_rate_limited: list[float] = []
    mpc_out_speed: list[float] = []
    mpc_out_invalid_reasons: Counter[str] = Counter()
    mpc_path_first_distance: list[float] = []

    lane_quality: list[float] = []
    lane_direct_error: list[float] = []
    lane_valid_count = 0
    lane_total_count = 0
    lane_reject_reasons: Counter[str] = Counter()
    lane_measurement_modes: Counter[str] = Counter()

    path_source_counter: Counter[str] = Counter()
    visual_reentry_events = 0
    containment_recovery_events = 0

    delta_min_deg = -21.5
    delta_max_deg = 25.0
    saturate_left = 0
    saturate_right = 0
    rate_limited_ticks = 0

    for event in _iter_events(path):
        logger = event.get("thread") or event.get("logger") or event.get("topic") or ""
        kind = event.get("event") or ""

        if logger == "mpc" and kind == "compute_in":
            d = _safe_float(event.get("tp0_distance_m"))
            h = _safe_float(event.get("tp0_heading_error_deg"))
            if d is not None:
                mpc_in_tp_dist.append(d)
            if h is not None:
                mpc_in_tp_heading.append(h)
            if isinstance(event.get("tp0"), list) and len(event["tp0"]) >= 2:
                mpc_path_first_distance.append(_safe_float(event.get("tp0_distance_m")) or 0.0)

        elif logger == "mpc" and kind == "compute_out":
            if event.get("valid") is False:
                mpc_out_invalid_reasons[str(event.get("reason") or "unknown")] += 1
                continue
            raw = _safe_float(event.get("delta_deg_raw"))
            cmd = _safe_float(event.get("steering_deg"))
            v = _safe_float(event.get("speed_mps"))
            if raw is not None:
                mpc_out_delta_raw.append(raw)
                if raw <= delta_min_deg + 0.05:
                    saturate_left += 1
                elif raw >= delta_max_deg - 0.05:
                    saturate_right += 1
            if cmd is not None:
                mpc_out_delta_rate_limited.append(cmd)
            if v is not None:
                mpc_out_speed.append(v)
            if raw is not None and cmd is not None and abs(raw - cmd) > 0.05:
                rate_limited_ticks += 1

        elif logger == "lane_observer" and kind == "lane_obs":
            lane_total_count += 1
            if event.get("direct_error_valid"):
                lane_valid_count += 1
            q = _safe_float(event.get("quality"))
            if q is not None:
                lane_quality.append(q)
            err = _safe_float(event.get("direct_error_m"))
            if err is not None:
                lane_direct_error.append(err)
            mode = str(event.get("measurement_mode") or "none")
            lane_measurement_modes[mode] += 1
            debug = event.get("debug") or {}
            if isinstance(debug, dict) and not debug.get("line_geometry_valid", True):
                lane_reject_reasons[str(debug.get("line_geometry_reject_reason") or "unknown")] += 1

        elif kind in {"target_path_built", "route_target_path_built", "visual_target_path_built"}:
            if kind == "visual_target_path_built":
                path_source_counter["visual_lane_waypoints"] += 1
            elif kind == "route_target_path_built":
                path_source_counter["route_waypoints"] += 1
            else:
                path_source_counter["lanelet_centerline"] += 1

        elif kind == "visual_lane_reentry_bias":
            visual_reentry_events += 1

        elif kind == "containment_recovery":
            containment_recovery_events += 1

    summary: dict[str, Any] = {
        "ticks": {
            "mpc_in": len(mpc_in_tp_dist),
            "mpc_out_valid": len(mpc_out_delta_raw),
            "lane_obs": lane_total_count,
        },
        "H1_DR_drift": {
            "tp0_distance_m_p50": _percentile(mpc_in_tp_dist, 0.5),
            "tp0_distance_m_p95": _percentile(mpc_in_tp_dist, 0.95),
            "tp0_distance_m_first": mpc_in_tp_dist[0] if mpc_in_tp_dist else None,
            "tp0_distance_m_last": mpc_in_tp_dist[-1] if mpc_in_tp_dist else None,
            "monotonic_growth": (
                bool(mpc_in_tp_dist)
                and mpc_in_tp_dist[-1] > 2.0 * (mpc_in_tp_dist[0] or 0.01)
            ),
        },
        "H2_start_pose_bad": {
            "tp0_distance_m_first_10_mean": (
                statistics.mean(mpc_in_tp_dist[:10]) if len(mpc_in_tp_dist) >= 10 else None
            ),
            "tp0_heading_error_deg_first_10_mean_abs": (
                statistics.mean(abs(x) for x in mpc_in_tp_heading[:10])
                if len(mpc_in_tp_heading) >= 10
                else None
            ),
        },
        "H3_lane_obs_rejected": {
            "valid_fraction": (lane_valid_count / lane_total_count) if lane_total_count else None,
            "reject_reasons": dict(lane_reject_reasons.most_common(5)),
            "measurement_modes": dict(lane_measurement_modes),
            "quality_p50": _percentile(lane_quality, 0.5),
        },
        "H4_steering_saturation": {
            "saturated_left_pct": (
                100.0 * saturate_left / len(mpc_out_delta_raw) if mpc_out_delta_raw else None
            ),
            "saturated_right_pct": (
                100.0 * saturate_right / len(mpc_out_delta_raw) if mpc_out_delta_raw else None
            ),
            "delta_raw_p05": _percentile(mpc_out_delta_raw, 0.05),
            "delta_raw_p95": _percentile(mpc_out_delta_raw, 0.95),
        },
        "H5_rate_limit": {
            "rate_limited_pct": (
                100.0 * rate_limited_ticks / len(mpc_out_delta_raw) if mpc_out_delta_raw else None
            ),
        },
        "path_source": dict(path_source_counter),
        "visual_reentry_events": visual_reentry_events,
        "containment_recovery_events": containment_recovery_events,
        "mpc_invalid_reasons": dict(mpc_out_invalid_reasons.most_common(10)),
    }
    return summary


def _verdict(summary: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    h1 = summary["H1_DR_drift"]
    if h1["tp0_distance_m_p95"] and h1["tp0_distance_m_p95"] > 0.25:
        notes.append(
            f"H1 ENCENDIDA: tp0_distance p95={h1['tp0_distance_m_p95']:.2f}m (target <0.10m). "
            "Probable drift de DR — calibrar encoder/wheelbase/servo-lag."
        )
    h2 = summary["H2_start_pose_bad"]
    first_d = h2["tp0_distance_m_first_10_mean"]
    if first_d is not None and first_d > 0.15:
        notes.append(
            f"H2 ENCENDIDA: tp0_distance medio en los primeros 10 ticks={first_d:.2f}m. "
            "Start pose desalineado del spawn físico."
        )
    h3 = summary["H3_lane_obs_rejected"]
    vf = h3["valid_fraction"]
    if vf is not None and vf < 0.6:
        notes.append(
            f"H3 ENCENDIDA: sólo {vf*100:.1f}% de lane_obs son válidas. "
            f"Razones de rechazo: {h3['reject_reasons']}. "
            "Subir `lane.width_consistency_tolerance` en runtime.yaml o revisar IPM."
        )
    h4 = summary["H4_steering_saturation"]
    sl = h4["saturated_left_pct"] or 0
    sr = h4["saturated_right_pct"] or 0
    if max(sl, sr) > 5:
        notes.append(
            f"H4 ENCENDIDA: steering saturado L={sl:.1f}% R={sr:.1f}%. "
            "Bounds asimétricos (-21.5/+25) o curvas demasiado cerradas vs wheelbase."
        )
    h5 = summary["H5_rate_limit"]
    rl = h5["rate_limited_pct"] or 0
    if rl > 30:
        notes.append(
            f"H5 ENCENDIDA: rate-limiter mordió en {rl:.1f}% de los ticks. "
            "BEHAVIOR_MAX_STEER_RATE_DEG_S=60°/s puede ser bajo para curvas cerradas."
        )
    if summary["containment_recovery_events"] > 0:
        notes.append(
            f"⚠ {summary['containment_recovery_events']} eventos `containment_recovery`. "
            "LaneContainmentRule disparó — error lateral >7cm."
        )
    if not notes:
        notes.append("Ninguna hipótesis encendida sobre umbrales por defecto. "
                     "Revisar manualmente si el síntoma persiste.")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Ruta a brain.jsonl")
    parser.add_argument("--json", action="store_true", help="Salida JSON cruda")
    parser.add_argument("--out", type=Path, default=None, help="Escribir markdown a este archivo")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"ERROR: no existe {args.path}", file=sys.stderr)
        return 2

    summary = analyze(args.path)
    verdict = _verdict(summary)

    if args.json:
        print(json.dumps({"summary": summary, "verdict": verdict}, indent=2, ensure_ascii=False))
        return 0

    lines: list[str] = []
    lines.append(f"# Diagnóstico lane-following: {args.path}")
    lines.append("")
    lines.append("## Hipótesis encendidas")
    for v in verdict:
        lines.append(f"- {v}")
    lines.append("")
    lines.append("## Métricas")
    lines.append("```json")
    lines.append(json.dumps(summary, indent=2, ensure_ascii=False))
    lines.append("```")
    text = "\n".join(lines)

    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Escrito a {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
