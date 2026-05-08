#!/usr/bin/env python3
"""
analyze_run — postmortem de un test del brain Bosch.

Lee:
  - <run_dir>/brain.jsonl           (eventos del brain)
  - <run_dir>/sim_bridge.jsonl  o   --sim-jsonl path  (ground truth del sim)

Calcula métricas y emite un report en stdout (y opcionalmente a un archivo).

Uso:
    python3 analyze_run.py <run_dir>
    python3 analyze_run.py <run_dir> --sim-jsonl /path/to/sim_bridge.jsonl
    python3 analyze_run.py <run_dir> --report report.txt

Métricas reportadas:
  - Pose drift (fused vs ground truth)
  - Lane offset (distribución, stretches malos)
  - Scenario timeline
  - Ruta cumplida (nodos visitados)
  - SafetyGate fallback frequency
  - MPC failure rate
  - Throughput motor commands
  - State changes timeline
"""

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Lee un archivo JSONL en una lista de dicts. Ignora líneas malformadas."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _percentile(values: List[float], p: float) -> float:
    """Percentil simple (p ∈ [0, 100])."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _filter(events: List[Dict[str, Any]], **predicates) -> List[Dict[str, Any]]:
    """Filtra eventos por kwargs (campo == valor)."""
    out = []
    for ev in events:
        ok = True
        for k, v in predicates.items():
            if ev.get(k) != v:
                ok = False
                break
        if ok:
            out.append(ev)
    return out


def _nearest_by_ts(events: List[Dict[str, Any]], target_ts: float,
                   max_skew_s: float = 0.05) -> Optional[Dict[str, Any]]:
    """Devuelve el evento más cercano en `ts` a target_ts (o None si skew > max).
    Asume `events` ordenado por `ts` (binary search-like sin numpy)."""
    if not events:
        return None
    lo, hi = 0, len(events) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if events[mid]["ts"] < target_ts:
            lo = mid + 1
        else:
            hi = mid
    candidates = []
    for idx in (lo - 1, lo, lo + 1):
        if 0 <= idx < len(events):
            candidates.append(events[idx])
    best = min(candidates, key=lambda e: abs(e["ts"] - target_ts), default=None)
    if best is None or abs(best["ts"] - target_ts) > max_skew_s:
        return None
    return best


# ----------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------

def metric_pose_drift(brain_events: List[Dict[str, Any]],
                      gt_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Drift entre fused_pose del brain y ground truth del simulador."""
    pose_pubs = _filter(brain_events, thread="pose_estimator", event="pose_published")
    gt_poses = _filter(gt_events, thread="ground_truth", event="gz_pose")
    gt_poses_sorted = sorted(gt_poses, key=lambda e: e["ts"])
    first_gps_fix_ts = next(
        (
            float(ev["ts"])
            for ev in pose_pubs
            if str(ev.get("reloc_mode") or "") == "gps_fix"
        ),
        None,
    )

    drifts = []
    samples = []
    warmup_skipped = 0
    for pose in pose_pubs:
        if first_gps_fix_ts is not None and float(pose["ts"]) < first_gps_fix_ts:
            warmup_skipped += 1
            continue
        gt = _nearest_by_ts(gt_poses_sorted, pose["ts"], max_skew_s=0.10)
        if gt is None:
            continue
        fx = pose.get("fused_x")
        fy = pose.get("fused_y")
        gx = gt.get("brain_map_x", gt.get("graphml_x"))
        gy = gt.get("brain_map_y", gt.get("graphml_y"))
        if any(v is None for v in (fx, fy, gx, gy)):
            continue
        d = math.hypot(fx - gx, fy - gy)
        drifts.append(d)
        samples.append({
            "ts": pose["ts"], "fused_x": fx, "fused_y": fy,
            "gt_x": gx, "gt_y": gy, "drift_m": d,
        })

    if not drifts:
        return {"available": False, "reason": "no matching pose+gt pairs"}

    return {
        "available": True,
        "samples": len(drifts),
        "mean_m": statistics.mean(drifts),
        "p50_m": _percentile(drifts, 50),
        "p90_m": _percentile(drifts, 90),
        "p99_m": _percentile(drifts, 99),
        "max_m": max(drifts),
        "min_m": min(drifts),
        "warmup_skipped_samples": warmup_skipped,
        "first_gps_fix_ts": first_gps_fix_ts,
        "first5": samples[:5],
        "worst5": sorted(samples, key=lambda s: -s["drift_m"])[:5],
    }


def metric_lane_offset(brain_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Distribución de |lateral_offset| cuando quality > 0.5."""
    lane_obs = _filter(brain_events, thread="lane_observer", event="lane_obs")
    # Fallback: pose_estimator también tiene camera_lateral_correction_m
    pose_pubs = _filter(brain_events, thread="pose_estimator", event="pose_published")

    offsets = []
    for ev in lane_obs:
        q = ev.get("quality", 0)
        off = ev.get("offset_m")
        if q is not None and q > 0.5 and off is not None:
            offsets.append(abs(off))

    bad_stretches = []  # stretches > 3 s con |offset| > 0.20 m
    if lane_obs:
        sorted_obs = sorted(lane_obs, key=lambda e: e["ts"])
        stretch_start = None
        for ev in sorted_obs:
            q = ev.get("quality", 0)
            off = ev.get("offset_m", 0)
            is_bad = q is not None and q > 0.5 and off is not None and abs(off) > 0.20
            if is_bad and stretch_start is None:
                stretch_start = ev["ts"]
            elif not is_bad and stretch_start is not None:
                dur = ev["ts"] - stretch_start
                if dur > 3.0:
                    bad_stretches.append({"start_ts": stretch_start, "dur_s": dur})
                stretch_start = None
        if stretch_start is not None:
            last_ts = sorted_obs[-1]["ts"]
            dur = last_ts - stretch_start
            if dur > 3.0:
                bad_stretches.append({"start_ts": stretch_start, "dur_s": dur})

    if not offsets:
        # Fallback usando lateral correction del pose_estimator
        offsets = [
            abs(ev.get("camera_lateral_correction_m", 0))
            for ev in pose_pubs
            if ev.get("camera_lateral_correction_m") not in (None, 0)
        ]
        return {
            "available": bool(offsets),
            "source": "pose_estimator.camera_lateral_correction_m (fallback)",
            "samples": len(offsets),
            "p50_m": _percentile(offsets, 50) if offsets else 0,
            "p90_m": _percentile(offsets, 90) if offsets else 0,
            "max_m": max(offsets) if offsets else 0,
            "bad_stretches": bad_stretches,
        }

    return {
        "available": True,
        "source": "lane_observer.offset_m (quality>0.5)",
        "samples": len(offsets),
        "mean_m": statistics.mean(offsets),
        "p50_m": _percentile(offsets, 50),
        "p90_m": _percentile(offsets, 90),
        "max_m": max(offsets),
        "bad_stretches": bad_stretches,
    }


def metric_scenario_timeline(brain_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tiempo en cada scenario."""
    plans = _filter(brain_events, thread="behavior_planner", event="plan_output")
    plans_sorted = sorted(plans, key=lambda e: e["ts"])
    if len(plans_sorted) < 2:
        return {"available": False, "reason": "fewer than 2 plan_output events"}

    durations: Counter[str] = Counter()
    transitions: List[Dict[str, Any]] = []
    prev = None
    for ev in plans_sorted:
        scen = ev.get("scenario", "?")
        if prev is None:
            prev = ev
            continue
        dt = ev["ts"] - prev["ts"]
        prev_scen = prev.get("scenario", "?")
        durations[prev_scen] += dt
        if prev_scen != scen:
            transitions.append({
                "ts": ev["ts"], "from": prev_scen, "to": scen,
            })
        prev = ev
    total = sum(durations.values()) or 1.0
    return {
        "available": True,
        "total_seconds": total,
        "by_scenario": {k: round(v, 2) for k, v in durations.items()},
        "by_scenario_pct": {k: round(v / total * 100, 1) for k, v in durations.items()},
        "transitions": len(transitions),
        "first_transitions": transitions[:10],
    }


def metric_route_progress(brain_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Nodos visitados vs ruta inicial."""
    nav_updates = _filter(brain_events, thread="nav_planner")
    node_changes = [e for e in nav_updates if e.get("event") == "node_change"]
    route_updates = [e for e in nav_updates if e.get("event") == "route_update"]

    if not route_updates:
        return {"available": False, "reason": "no nav_planner events"}

    visited_nodes = []
    seen = set()
    for ev in nav_updates:
        node_id = ev.get("current_node_id")
        if node_id is not None and node_id not in seen:
            seen.add(node_id)
            visited_nodes.append(node_id)

    map_match_errors = [e.get("map_match_error_m", 0) for e in route_updates
                        if e.get("map_match_error_m") is not None]

    return {
        "available": True,
        "visited_nodes_count": len(seen),
        "visited_nodes": visited_nodes[:20],
        "node_change_events": len(node_changes),
        "map_match_error_p50_m": _percentile(map_match_errors, 50) if map_match_errors else 0,
        "map_match_error_p90_m": _percentile(map_match_errors, 90) if map_match_errors else 0,
        "map_match_error_max_m": max(map_match_errors) if map_match_errors else 0,
    }


def metric_dispatch(brain_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Dispatcher decisions: SafetyGate fallbacks, despachos exitosos."""
    decisions = _filter(brain_events, thread="dispatcher", event="dispatch_decision")
    if not decisions:
        return {"available": False, "reason": "no dispatch_decision events"}

    fallback_reasons = ("stale", "no_motor_command", "no_pose_estimate",
                        "motor_command_invalid", "behavior_stale", "pose_stale")
    fallbacks = [d for d in decisions if any(
        r in str(d.get("cmd_reason", "")) for r in fallback_reasons
    )]
    by_reason = Counter(d.get("cmd_reason") for d in decisions)
    states = Counter(d.get("state") for d in decisions)

    return {
        "available": True,
        "total_decisions": len(decisions),
        "fallback_count": len(fallbacks),
        "fallback_pct": round(len(fallbacks) / len(decisions) * 100, 2),
        "by_reason": dict(by_reason.most_common(10)),
        "by_state": dict(states),
    }


def metric_motor_throughput(brain_events: List[Dict[str, Any]],
                            duration_s: float) -> Dict[str, Any]:
    """Cuántos motor_cmd_sent salieron al sim."""
    sends = _filter(brain_events, thread="zmq_motor", event="motor_cmd_sent")
    if not sends:
        return {"available": False, "reason": "no motor_cmd_sent events"}
    rate = len(sends) / duration_s if duration_s > 0 else 0
    return {
        "available": True,
        "count": len(sends),
        "rate_hz": round(rate, 2),
    }


def metric_state_timeline(brain_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cronología de cambios de modo (STOP/MANUAL/AUTO)."""
    state_events = _filter(brain_events, thread="state_machine", event="state_change")
    return {
        "available": bool(state_events),
        "transitions": [
            {"ts": e["ts"], "from": e.get("from"), "to": e.get("to")}
            for e in state_events
        ],
    }


def metric_mpc_failures(brain_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """% de compute_out con valid=False, y stats de v_opt clamping."""
    mpc_outs = _filter(brain_events, thread="mpc", event="compute_out")
    if not mpc_outs:
        return {"available": False, "reason": "no mpc compute_out events"}
    failures = [e for e in mpc_outs if not e.get("valid", True)]
    by_reason = Counter(e.get("reason") for e in failures)
    # v_opt analysis — el solver puede saturar en lower bound (típico -0.5)
    # y entonces el clamp `max(0, v_opt)` lo deja en 0. Si esto pasa siempre,
    # el auto no se mueve aunque scenario diga lane_keep+valid.
    v_opt_raws = [e.get("v_opt_raw") for e in mpc_outs if e.get("v_opt_raw") is not None]
    v_opt_neg = [v for v in v_opt_raws if v < 0]
    v_opt_clamped_to_zero = [
        e for e in mpc_outs
        if e.get("v_opt_raw") is not None and e["v_opt_raw"] < 0 and e.get("speed_mps", 0) == 0.0
    ]
    return {
        "available": True,
        "total": len(mpc_outs),
        "failures": len(failures),
        "failure_pct": round(len(failures) / len(mpc_outs) * 100, 2),
        "by_reason": dict(by_reason.most_common(5)),
        "v_opt_raw_p50": _percentile(v_opt_raws, 50) if v_opt_raws else 0,
        "v_opt_raw_p99": _percentile(v_opt_raws, 99) if v_opt_raws else 0,
        "v_opt_raw_min": min(v_opt_raws) if v_opt_raws else 0,
        "v_opt_raw_max": max(v_opt_raws) if v_opt_raws else 0,
        "v_opt_negative_count": len(v_opt_neg),
        "v_opt_negative_pct": round(len(v_opt_neg) / len(v_opt_raws) * 100, 1) if v_opt_raws else 0,
        "v_opt_clamped_to_zero_count": len(v_opt_clamped_to_zero),
    }


# ----------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------

def _section(title: str) -> str:
    return f"\n{'='*60}\n  {title}\n{'='*60}\n"


def render_report(
    run_dir: str,
    brain_events: List[Dict[str, Any]],
    gt_events: List[Dict[str, Any]],
) -> str:
    lines = []

    # Duración aproximada
    all_ts = [e["ts"] for e in brain_events if "ts" in e]
    duration = (max(all_ts) - min(all_ts)) if len(all_ts) >= 2 else 0
    lines.append(_section("RUN OVERVIEW"))
    lines.append(f"  run_dir:        {run_dir}")
    lines.append(f"  brain events:   {len(brain_events)}")
    lines.append(f"  gt events:      {len(gt_events)}")
    lines.append(f"  duration:       {duration:.1f} s")
    if all_ts:
        lines.append(f"  start ts:       {min(all_ts):.3f}")
        lines.append(f"  end ts:         {max(all_ts):.3f}")

    # Por-thread breakdown
    by_thread = Counter(e.get("thread") for e in brain_events)
    lines.append(_section("EVENTS BY THREAD"))
    for thread, count in by_thread.most_common():
        lines.append(f"  {thread:30s} {count:>8d}")

    # Por-event breakdown (top 20)
    by_event = Counter(f"{e.get('thread')}.{e.get('event')}" for e in brain_events)
    lines.append(_section("TOP EVENTS"))
    for ev, count in by_event.most_common(20):
        lines.append(f"  {ev:50s} {count:>8d}")

    # State timeline
    state_m = metric_state_timeline(brain_events)
    lines.append(_section("STATE TIMELINE"))
    if state_m["available"]:
        for t in state_m["transitions"]:
            lines.append(f"  ts={t['ts']:.3f}  {t['from']} -> {t['to']}")
    else:
        lines.append("  (no state_change events — agregar live_log a stateMachine.py)")

    # Pose drift vs ground truth
    drift = metric_pose_drift(brain_events, gt_events)
    lines.append(_section("POSE DRIFT (fused vs ground truth)"))
    if drift["available"]:
        lines.append(f"  samples:        {drift['samples']}")
        if drift.get("warmup_skipped_samples"):
            lines.append(
                f"  warmup skipped: {drift['warmup_skipped_samples']} samples"
                f"  (until first gps_fix @ ts={drift['first_gps_fix_ts']:.3f})"
            )
        lines.append(f"  mean:           {drift['mean_m']:.3f} m")
        lines.append(f"  p50:            {drift['p50_m']:.3f} m")
        lines.append(f"  p90:            {drift['p90_m']:.3f} m   (target < 0.30)")
        lines.append(f"  p99:            {drift['p99_m']:.3f} m")
        lines.append(f"  max:            {drift['max_m']:.3f} m   (target < 0.60)")
        lines.append(f"  min:            {drift['min_m']:.3f} m")
        lines.append("\n  WORST 5:")
        for s in drift["worst5"]:
            lines.append(f"    ts={s['ts']:.3f}  fused=({s['fused_x']:.2f},{s['fused_y']:.2f})  "
                         f"gt=({s['gt_x']:.2f},{s['gt_y']:.2f})  drift={s['drift_m']:.3f} m")
    else:
        lines.append(f"  (not available: {drift.get('reason', '?')})")

    # Lane offset
    lane = metric_lane_offset(brain_events)
    lines.append(_section("LANE OFFSET"))
    if lane["available"]:
        lines.append(f"  source:         {lane['source']}")
        lines.append(f"  samples:        {lane['samples']}")
        lines.append(f"  p50:            {lane['p50_m']:.3f} m")
        lines.append(f"  p90:            {lane['p90_m']:.3f} m   (target < 0.20)")
        lines.append(f"  max:            {lane['max_m']:.3f} m")
        if lane["bad_stretches"]:
            lines.append(f"  BAD STRETCHES (> 3 s con |offset| > 0.20):")
            for s in lane["bad_stretches"]:
                lines.append(f"    ts={s['start_ts']:.3f}  dur={s['dur_s']:.2f} s")
        else:
            lines.append("  (no bad stretches detected — OK)")
    else:
        lines.append(f"  (not available: {lane.get('reason', '?')})")

    # Scenario timeline
    scen = metric_scenario_timeline(brain_events)
    lines.append(_section("SCENARIO TIMELINE"))
    if scen["available"]:
        lines.append(f"  total:          {scen['total_seconds']:.1f} s")
        lines.append(f"  transitions:    {scen['transitions']}")
        for s, pct in sorted(scen["by_scenario_pct"].items(), key=lambda kv: -kv[1]):
            lines.append(f"    {s:30s} {pct:>5.1f}%   {scen['by_scenario'][s]:>5.1f} s")
        if scen["first_transitions"]:
            lines.append("\n  FIRST 10 TRANSITIONS:")
            for t in scen["first_transitions"]:
                lines.append(f"    ts={t['ts']:.3f}  {t['from']} -> {t['to']}")
    else:
        lines.append(f"  (not available: {scen.get('reason', '?')})")

    # Route progress
    route = metric_route_progress(brain_events)
    lines.append(_section("ROUTE PROGRESS"))
    if route["available"]:
        lines.append(f"  visited nodes:  {route['visited_nodes_count']}")
        lines.append(f"  first 20:       {route['visited_nodes']}")
        lines.append(f"  node_change events: {route['node_change_events']}")
        lines.append(f"  map_match_error p50:  {route['map_match_error_p50_m']:.3f} m")
        lines.append(f"  map_match_error p90:  {route['map_match_error_p90_m']:.3f} m")
        lines.append(f"  map_match_error max:  {route['map_match_error_max_m']:.3f} m")
    else:
        lines.append(f"  (not available: {route.get('reason', '?')})")

    # Dispatch
    disp = metric_dispatch(brain_events)
    lines.append(_section("DISPATCHER (SafetyGate)"))
    if disp["available"]:
        lines.append(f"  total decisions: {disp['total_decisions']}")
        lines.append(f"  fallbacks:       {disp['fallback_count']}  ({disp['fallback_pct']}%)   (target < 2%)")
        lines.append(f"  by reason:")
        for r, c in disp["by_reason"].items():
            lines.append(f"    {str(r):40s} {c:>8d}")
        lines.append(f"  by state:        {disp['by_state']}")
    else:
        lines.append(f"  (not available: {disp.get('reason', '?')})")

    # MPC failures + v_opt analysis
    mpc = metric_mpc_failures(brain_events)
    lines.append(_section("MPC FAILURES + V_OPT ANALYSIS"))
    if mpc["available"]:
        lines.append(f"  total computes:  {mpc['total']}")
        lines.append(f"  failures:        {mpc['failures']}  ({mpc['failure_pct']}%)   (target < 1%)")
        if mpc["by_reason"]:
            lines.append(f"  failure reasons:")
            for r, c in mpc["by_reason"].items():
                lines.append(f"    {str(r):40s} {c:>8d}")
        lines.append(f"\n  v_opt_raw distribution (antes de clamp >= 0):")
        lines.append(f"    p50:           {mpc['v_opt_raw_p50']:.3f} m/s")
        lines.append(f"    p99:           {mpc['v_opt_raw_p99']:.3f} m/s")
        lines.append(f"    min:           {mpc['v_opt_raw_min']:.3f} m/s   (típico lower bound del solver)")
        lines.append(f"    max:           {mpc['v_opt_raw_max']:.3f} m/s")
        lines.append(f"    negative:      {mpc['v_opt_negative_count']}  ({mpc['v_opt_negative_pct']}%)")
        lines.append(f"    clamped to 0:  {mpc['v_opt_clamped_to_zero_count']}   (auto no se mueve)")
        if mpc['v_opt_negative_pct'] > 50:
            lines.append(f"\n  ⚠ WARNING: {mpc['v_opt_negative_pct']}% del tiempo el MPC pide ir hacia atrás.")
            lines.append(f"  ⚠ Probable causa: target_path desalineado con pose actual (yaw mal,")
            lines.append(f"  ⚠ pose offset, o costos del solver fuera de balance).")
    else:
        lines.append(f"  (not available: {mpc.get('reason', '?')})")

    # Motor throughput
    moto = metric_motor_throughput(brain_events, duration)
    lines.append(_section("MOTOR COMMAND THROUGHPUT"))
    if moto["available"]:
        lines.append(f"  count:           {moto['count']}")
        lines.append(f"  rate:            {moto['rate_hz']} Hz   (target ~50 Hz)")
    else:
        lines.append(f"  (not available: {moto.get('reason', '?')})")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Postmortem analyzer")
    parser.add_argument("run_dir", help="Run directory (contiene brain.jsonl)")
    parser.add_argument("--sim-jsonl", default=None,
                        help="Path al sim_bridge.jsonl (default: <run_dir>/sim_bridge.jsonl, "
                             "luego buscar en urt-simulator/temp/logs/run_<TS>/)")
    parser.add_argument("--report", default=None,
                        help="Si está seteado, guarda el report a este path")
    args = parser.parse_args()

    run_dir = args.run_dir.rstrip("/")
    brain_path = os.path.join(run_dir, "brain.jsonl")

    sim_path = args.sim_jsonl
    if sim_path is None:
        # Probar primero en el mismo run_dir, luego en urt-simulator
        candidates = [
            os.path.join(run_dir, "sim_bridge.jsonl"),
            os.path.join("/Users/luciogarcia/urt-simulator/temp/logs",
                         os.path.basename(run_dir), "sim_bridge.jsonl"),
        ]
        for c in candidates:
            if os.path.exists(c):
                sim_path = c
                break

    if not os.path.exists(brain_path):
        print(f"ERROR: brain.jsonl not found at {brain_path}", file=sys.stderr)
        return 2

    brain_events = _load_jsonl(brain_path)
    gt_events = _load_jsonl(sim_path) if sim_path else []

    report = render_report(run_dir, brain_events, gt_events)
    print(report)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[analyze_run] saved report to {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
