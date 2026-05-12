#!/usr/bin/env python3
"""manual_shadow_analyze — compara MPC shadow vs manejo manual.

Lee `brain.jsonl` (todos los eventos del brain) + `sim_bridge.jsonl`
(ground truth del simulador) de una run generada por
`manual_campaign_runner.py`, y produce:

    <run_dir>/manual_vs_mpc.json   reporte estructurado
    <run_dir>/manual_vs_mpc.txt    render texto a stdout (escribe el caller)

Idea: el operador manejó en MANUAL desde la GUI, mientras el MPC corrió
en shadow (`URT_SHADOW_PLANNER=1`). Para cada evento `mpc compute_out`
con `valid=True`, snapshot del último `zmq_motor motor_cmd_sent` (lo que
el operador realmente comandó). Diff = "lo que el controlador quería vs
lo que el humano hizo".

Conversión de unidades (DrivingModel + dispatcher + motion_controller):
- `motor_cmd_sent.steer`  = wire = wheel_deg × 10 con `+wire = derecha`
  (convención legacy Nucleo/sim_bridge). ÷10 → deg en wire frame.
- `motor_cmd_sent.speed`  = wire = m/s × 1000. ÷1000 → m/s.
- `mpc.steering_deg`      = ISO 8855 frame: `+deg = ruedas a la izquierda`
  (`motion_controller.py:240`). El dispatcher en motor_command_dispatcher.py:254
  hace `wire = -mpc_steering_deg × 10` antes de mandar al firmware/sim.

  Para que la comparación sea correcta, en este analizador NEGAMOS la salida
  del MPC para llevarla al wire frame. Si no, MPC=+25° (izq) y manual=-25°
  (izq) aparecerían como Δ=+50° aunque AMBOS apuntan a la misma dirección.

Uso:
    python tools/dev/manual_shadow_analyze.py <run_dir>
    python tools/dev/manual_shadow_analyze.py <run_dir> --sim-jsonl path
    python tools/dev/manual_shadow_analyze.py <run_dir> --route-json path
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path
from typing import Any


_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from campaign_runner import (
    _finite_float,
    _load_jsonl,
    _percentile,
    _point_to_polyline_distance,
    _summary,
)


# Conversión wire ↔ unidades físicas (ver docstring).
_STEER_WIRE_TO_DEG = 1.0 / 10.0
_SPEED_WIRE_TO_MPS = 1.0 / 1000.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", type=Path, help="Directorio de la run (contiene brain.jsonl).")
    p.add_argument("--sim-jsonl", type=Path, default=None,
                   help="Path explícito al ground-truth JSONL. Default: <run_dir>/sim_bridge.jsonl.")
    p.add_argument("--route-json", type=Path, default=None,
                   help="Path explícito al route_expected.json. Default: <run_dir>/route_expected.json.")
    p.add_argument("--match-tolerance-s", type=float, default=0.5,
                   help="Tolerancia máxima entre mpc compute_out y manual snapshot (s).")
    p.add_argument("--worst-n", type=int, default=30,
                   help="Cuántos peores instantes guardar en worst_samples.")
    return p.parse_args()


def _build_manual_timeline(motor_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convierte cada motor_cmd_sent a un snapshot (latest_speed_wire, latest_steer_wire).

    threadWrite emite eventos separados para action="speed" y action="steer"
    (vía `_handle_continuous_motion_command`) y combinados para action="vcd".
    Acá ensamblamos un timeline ordenado por ts donde cada entry tiene los
    últimos valores conocidos de speed y steer.
    """
    timeline: list[dict[str, Any]] = []
    last_speed: int | None = None
    last_steer: int | None = None
    for ev in motor_events:
        ts = _finite_float(ev.get("ts"))
        if ts is None:
            continue
        action = str(ev.get("action") or "")
        speed_raw = ev.get("speed")
        steer_raw = ev.get("steer")
        if action == "speed":
            if speed_raw is not None:
                last_speed = int(speed_raw)
        elif action == "steer":
            if steer_raw is not None:
                last_steer = int(steer_raw)
        elif action == "vcd":
            if speed_raw is not None:
                last_speed = int(speed_raw)
            if steer_raw is not None:
                last_steer = int(steer_raw)
        elif action == "brake":
            last_speed = 0
            if steer_raw is not None:
                last_steer = int(steer_raw)
        else:
            # Otros action (kl, batteryCapacity, etc.) no afectan al timeline
            # de motion.
            continue
        timeline.append({
            "ts": ts,
            "speed_wire": last_speed,
            "steer_wire": last_steer,
            "action": action,
        })
    return timeline


def _snapshot_at(timeline: list[dict[str, Any]], ts: float, tolerance_s: float) -> dict[str, Any] | None:
    """Devuelve el último snapshot ≤ ts dentro de la tolerancia. None si no hay."""
    if not timeline:
        return None
    timestamps = [item["ts"] for item in timeline]
    idx = bisect.bisect_right(timestamps, ts) - 1
    if idx < 0:
        return None
    snap = timeline[idx]
    if (ts - snap["ts"]) > tolerance_s:
        # Hay datos pero son demasiado viejos — el operador soltó controles.
        # Lo devolvemos igual pero marcamos el age para que el caller decida.
        pass
    return {
        "ts": float(snap["ts"]),
        "age_s": float(ts - snap["ts"]),
        "speed_wire": snap["speed_wire"],
        "steer_wire": snap["steer_wire"],
        "action": snap["action"],
    }


def _lanelet_at(route_updates: list[dict[str, Any]], route_ts: list[float], ts: float) -> str | None:
    if not route_ts:
        return None
    idx = bisect.bisect_right(route_ts, ts) - 1
    if idx < 0:
        return None
    lanelet = route_updates[idx].get("current_lanelet_id")
    return str(lanelet) if lanelet is not None else None


def _expected_polyline(route_json: Path) -> list[tuple[float, float]]:
    if not route_json.exists():
        return []
    data = json.loads(route_json.read_text(encoding="utf-8"))
    return [
        (float(pt[0]), float(pt[1]))
        for pt in data.get("waypoints", [])
        if isinstance(pt, list) and len(pt) >= 2
    ]


def _cross_track_at(gt_events: list[dict[str, Any]], expected_xy: list[tuple[float, float]]) -> dict[str, Any]:
    """Cross-track del GT contra ruta esperada (igual idea que campaign_runner)."""
    if not expected_xy:
        return {"available": False, "reason": "no_expected_route"}
    distances: list[float] = []
    actual_xy: list[tuple[float, float]] = []
    for ev in gt_events:
        if ev.get("thread") != "ground_truth" or ev.get("event") != "gz_pose":
            continue
        x = ev.get("brain_map_x", ev.get("graphml_x"))
        y = ev.get("brain_map_y", ev.get("graphml_y"))
        if x is None or y is None:
            continue
        point = (float(x), float(y))
        actual_xy.append(point)
        distances.append(_point_to_polyline_distance(point, expected_xy))
    return {
        "available": bool(distances),
        "samples": len(distances),
        "error_m": _summary(distances),
    }


def analyze(run_dir: Path, *, sim_jsonl: Path, route_json: Path,
            match_tolerance_s: float, worst_n: int) -> dict[str, Any]:
    brain_jsonl = run_dir / "brain.jsonl"
    brain_events = _load_jsonl(brain_jsonl)
    gt_events = _load_jsonl(sim_jsonl)
    expected_xy = _expected_polyline(route_json)

    mpc_outs = [
        e for e in brain_events
        if e.get("thread") == "mpc" and e.get("event") == "compute_out"
    ]
    motor_sends = [
        e for e in brain_events
        if e.get("thread") == "zmq_motor" and e.get("event") == "motor_cmd_sent"
    ]
    route_updates = [
        e for e in brain_events
        if e.get("thread") == "nav_planner" and e.get("event") == "route_update"
    ]
    route_ts = [float(e.get("ts", 0.0) or 0.0) for e in route_updates]
    dispatch_decisions = [
        e for e in brain_events
        if e.get("thread") == "dispatcher" and e.get("event") == "dispatch_decision"
    ]

    manual_timeline = _build_manual_timeline(motor_sends)

    samples: list[dict[str, Any]] = []
    invalid_mpc = 0
    for mpc in mpc_outs:
        ts = _finite_float(mpc.get("ts"))
        if ts is None:
            continue
        if not bool(mpc.get("valid", False)):
            invalid_mpc += 1
            continue
        snap = _snapshot_at(manual_timeline, ts, match_tolerance_s)
        if snap is None or snap.get("speed_wire") is None or snap.get("steer_wire") is None:
            # El operador no había emitido aún ningún comando completo.
            continue
        manual_steer_deg = float(snap["steer_wire"]) * _STEER_WIRE_TO_DEG
        manual_speed_mps = float(snap["speed_wire"]) * _SPEED_WIRE_TO_MPS
        # MPC publica '+deg = ruedas izquierda' (ISO 8855); el wire usa
        # '+deg = ruedas derecha'. Negamos para llevar al wire frame antes
        # de comparar con el manejo manual (ver docstring del módulo).
        mpc_steer_deg = -(_finite_float(mpc.get("steering_deg")) or 0.0)
        mpc_speed_mps = _finite_float(mpc.get("speed_mps")) or 0.0
        lanelet_id = _lanelet_at(route_updates, route_ts, ts)
        samples.append({
            "ts": ts,
            "lanelet_id": lanelet_id,
            "mpc_steering_deg": mpc_steer_deg,
            "manual_steering_deg": manual_steer_deg,
            "delta_steering_deg": mpc_steer_deg - manual_steer_deg,
            "mpc_speed_mps": mpc_speed_mps,
            "manual_speed_mps": manual_speed_mps,
            "delta_speed_mps": mpc_speed_mps - manual_speed_mps,
            "manual_snapshot_age_s": snap["age_s"],
            "manual_last_action": snap["action"],
            "scenario": mpc.get("scenario"),
            "path_source": mpc.get("path_source"),
            "visual_steer_cap_applied": mpc.get("visual_steer_cap_applied"),
            "speed_recovery_reason": mpc.get("speed_recovery_reason"),
        })

    deltas_steer = [s["delta_steering_deg"] for s in samples]
    deltas_speed = [s["delta_speed_mps"] for s in samples]
    abs_deltas_steer = [abs(v) for v in deltas_steer]
    abs_deltas_speed = [abs(v) for v in deltas_speed]

    per_lanelet: dict[str, dict[str, list[float]]] = {}
    for s in samples:
        lanelet = s.get("lanelet_id") or "unknown"
        bucket = per_lanelet.setdefault(lanelet, {
            "delta_steering_deg": [],
            "delta_speed_mps": [],
            "abs_delta_steering_deg": [],
            "abs_delta_speed_mps": [],
        })
        bucket["delta_steering_deg"].append(s["delta_steering_deg"])
        bucket["delta_speed_mps"].append(s["delta_speed_mps"])
        bucket["abs_delta_steering_deg"].append(abs(s["delta_steering_deg"]))
        bucket["abs_delta_speed_mps"].append(abs(s["delta_speed_mps"]))
    per_lanelet_summary = {
        lanelet: {
            "samples": len(bucket["delta_steering_deg"]),
            "delta_steering_deg_signed": _summary(bucket["delta_steering_deg"]),
            "delta_steering_deg_abs": _summary(bucket["abs_delta_steering_deg"]),
            "delta_speed_mps_signed": _summary(bucket["delta_speed_mps"]),
            "delta_speed_mps_abs": _summary(bucket["abs_delta_speed_mps"]),
        }
        for lanelet, bucket in sorted(per_lanelet.items())
    }

    worst_by_steer = sorted(samples, key=lambda s: abs(s["delta_steering_deg"]), reverse=True)[:worst_n]
    worst_by_speed = sorted(samples, key=lambda s: abs(s["delta_speed_mps"]), reverse=True)[:worst_n]

    dispatch_states = {}
    for ev in dispatch_decisions:
        state = str(ev.get("state") or "UNKNOWN")
        dispatch_states[state] = dispatch_states.get(state, 0) + 1
    shadow_blocked = sum(
        1 for ev in dispatch_decisions if not bool(ev.get("dispatched", True))
    )

    cross_track = _cross_track_at(gt_events, expected_xy)

    report = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "samples": len(samples),
        "mpc": {
            "total_events": len(mpc_outs),
            "valid_events": len(mpc_outs) - invalid_mpc,
            "invalid_events": invalid_mpc,
        },
        "manual": {
            "motor_cmd_sent_events": len(motor_sends),
            "first_command_ts": (manual_timeline[0]["ts"] if manual_timeline else None),
            "last_command_ts":  (manual_timeline[-1]["ts"] if manual_timeline else None),
        },
        "dispatcher": {
            "decisions": len(dispatch_decisions),
            "by_state": dispatch_states,
            "shadow_blocked": shadow_blocked,
            "shadow_blocked_pct": (
                round(shadow_blocked / len(dispatch_decisions) * 100, 2)
                if dispatch_decisions else None
            ),
        },
        "match_tolerance_s": match_tolerance_s,
        "delta_steering_deg": {
            "signed": _summary(deltas_steer),
            "abs":    _summary(abs_deltas_steer),
        },
        "delta_speed_mps": {
            "signed": _summary(deltas_speed),
            "abs":    _summary(abs_deltas_speed),
        },
        "per_lanelet": per_lanelet_summary,
        "cross_track_error_m": cross_track,
        "worst_samples_by_steer_delta": worst_by_steer,
        "worst_samples_by_speed_delta": worst_by_speed,
    }
    return report


def _fmt_summary(name: str, summary: dict[str, Any], unit: str) -> str:
    s = summary or {}
    samples = s.get("samples", 0) or 0
    if samples <= 0:
        return f"  {name:30s} (sin samples)"
    p10 = s.get("p10")
    p50 = s.get("p50")
    p90 = s.get("p90")
    mx = s.get("max")
    mn = s.get("min")
    def f(v):
        return f"{v:+.3f}" if isinstance(v, (int, float)) else "-"
    return (
        f"  {name:30s} n={samples:5d}  "
        f"p10={f(p10)} p50={f(p50)} p90={f(p90)} "
        f"min={f(mn)} max={f(mx)} {unit}"
    )


def render_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"manual_vs_mpc — run={report['run_id']}")
    lines.append("=" * 72)
    lines.append(f"matched pairs:        {report['samples']}")
    lines.append(f"mpc compute_out:      total={report['mpc']['total_events']} valid={report['mpc']['valid_events']} invalid={report['mpc']['invalid_events']}")
    lines.append(f"manual motor_cmd_sent: {report['manual']['motor_cmd_sent_events']}")
    disp = report.get("dispatcher", {})
    lines.append(
        f"dispatcher decisions: {disp.get('decisions', 0)} "
        f"shadow_blocked={disp.get('shadow_blocked', 0)} "
        f"({disp.get('shadow_blocked_pct')}%)"
    )
    lines.append(f"by_state: {disp.get('by_state', {})}")
    lines.append("")
    lines.append("Δ steering (deg)   — MPC quería vs lo que vos comandaste")
    lines.append(_fmt_summary("signed", report["delta_steering_deg"]["signed"], "deg"))
    lines.append(_fmt_summary("abs",    report["delta_steering_deg"]["abs"],    "deg"))
    lines.append("")
    lines.append("Δ speed (m/s)      — MPC quería vs lo que vos comandaste")
    lines.append(_fmt_summary("signed", report["delta_speed_mps"]["signed"], "m/s"))
    lines.append(_fmt_summary("abs",    report["delta_speed_mps"]["abs"],    "m/s"))
    lines.append("")
    ct = report.get("cross_track_error_m") or {}
    if ct.get("available"):
        lines.append("cross-track del manejo manual vs ruta esperada (ground truth)")
        lines.append(_fmt_summary("error_m", ct["error_m"], "m"))
    else:
        lines.append(f"cross-track no disponible: {ct.get('reason', 'unknown')}")
    lines.append("")
    lines.append("Per-lanelet (Δ steering abs):")
    for lanelet, summary in (report.get("per_lanelet") or {}).items():
        ds = summary["delta_steering_deg_abs"]
        sp = summary["delta_speed_mps_abs"]
        lines.append(
            f"  lanelet={lanelet:<8s} n={summary['samples']:5d}  "
            f"|Δsteer|: p50={ds.get('p50'):+.2f} p90={ds.get('p90'):+.2f} max={ds.get('max'):+.2f} deg  "
            f"|Δspeed|: p50={sp.get('p50'):+.2f} p90={sp.get('p90'):+.2f} max={sp.get('max'):+.2f} m/s"
        )
    lines.append("")
    lines.append("Top divergencias de steering (MPC quería X vs vos hiciste Y):")
    for s in (report.get("worst_samples_by_steer_delta") or [])[:10]:
        lines.append(
            f"  ts={s['ts']:.3f} lanelet={s.get('lanelet_id') or '-':>6s}  "
            f"mpc={s['mpc_steering_deg']:+6.2f}°  manual={s['manual_steering_deg']:+6.2f}°  "
            f"Δ={s['delta_steering_deg']:+6.2f}°  scenario={s.get('scenario')}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    sim_jsonl = (args.sim_jsonl or (run_dir / "sim_bridge.jsonl")).resolve()
    route_json = (args.route_json or (run_dir / "route_expected.json")).resolve()
    if not run_dir.exists():
        print(f"manual_shadow_analyze: run_dir no existe: {run_dir}", file=sys.stderr)
        return 2
    report = analyze(
        run_dir,
        sim_jsonl=sim_jsonl,
        route_json=route_json,
        match_tolerance_s=args.match_tolerance_s,
        worst_n=args.worst_n,
    )
    (run_dir / "manual_vs_mpc.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    sys.stdout.write(render_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
