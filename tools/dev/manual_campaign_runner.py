#!/usr/bin/env python3
"""Manual campaign runner — driver-in-the-loop con MPC en shadow mode.

A diferencia de `tools/dev/campaign_runner.py`, NO pone el brain en AUTO:
levanta la GUI PyQt5 para que un operador maneje el auto en MANUAL desde
el dashboard, mientras el MPC corre "fantasma" (`URT_SHADOW_PLANNER=1`)
y va loggeando en `brain.jsonl` lo que habría comandado en cada momento.

Postmortem: `tools/dev/manual_shadow_analyze.py` compara `mpc compute_out`
(lo que el controlador quería) contra `zmq_motor motor_cmd_sent` (lo que
el operador realmente envió) para descubrir dónde y por qué diverge.

La orquestación (procesos, sim, ground-truth logger, dashboard client) se
reusa via import desde el campaign_runner viejo — sin modificarlo.

Uso:
    python tools/dev/manual_campaign_runner.py [--destination-lanelet 138]
    python tools/dev/manual_campaign_runner.py --duration 300 --no-record
    python tools/dev/manual_campaign_runner.py --no-gui  # backend-only debug

Stop: Ctrl+C en la terminal del runner, o cerrar la ventana de la GUI.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


# El campaign_runner inserta BRAIN_DIR en sys.path; replicamos el bootstrap.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Reusamos helpers del campaign_runner viejo. NO lo modificamos — solo
# importamos sus piezas públicas-en-la-práctica (prefijo `_` pero estables).
from campaign_runner import (
    BRAIN_DIR,
    DEFAULT_OSM_PATH,
    DEFAULT_SIM_DIR,
    DEFAULT_TRACK_PNG,
    DashboardClient,
    ManagedProcess,
    RouteCommandResult,
    _apply_mode_env,
    _build_expected_route,
    _cleanup_known_processes,
    _collect_camera_video,
    _collect_visual_lane_samples,
    _extract_video_frames,
    _find_brain_python,
    _find_gz_python,
    _finite_float,
    _load_jsonl,
    _maybe_reexec_with_brain_venv,
    _promote_gui_overlay_camera_video,
    _run,
    _update_latest_symlink,
    _wait_for_jsonl_event,
    _wait_for_navigation_acceptance,
    _wait_port,
    _write_campaign_digest,
    _write_debug_overlay_video,
    _write_overlay,
    _write_visual_lane_overlay,
)


def _ground_truth_xy(gt_events: list[dict[str, Any]]) -> list[list[float]]:
    """Extrae la trayectoria (x, y) real del auto desde gz_pose_logger.

    Mismo criterio que `_compute_metrics` del runner viejo: prefiere las
    coordenadas en frame de brain map, cae a graphml_x/y si faltan.
    """
    actual: list[list[float]] = []
    for ev in gt_events:
        if ev.get("thread") != "ground_truth" or ev.get("event") != "gz_pose":
            continue
        x = ev.get("brain_map_x", ev.get("graphml_x"))
        y = ev.get("brain_map_y", ev.get("graphml_y"))
        if x is None or y is None:
            continue
        actual.append([float(x), float(y)])
    return actual


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Segundos máximos de la sesión. Default: ilimitado (Ctrl+C o cerrar GUI).",
    )
    parser.add_argument(
        "--destination-lanelet", default="138",
        help="Lanelet destino para armar la ruta esperada (input al BehaviorPlanner shadow).",
    )
    parser.add_argument("--destination-x", type=float, default=None)
    parser.add_argument("--destination-y", type=float, default=None)
    parser.add_argument("--run-id", default=None, help="Default: run_manual_<YYYYmmdd_HHMMSS>.")
    parser.add_argument("--brain-dir", type=Path, default=BRAIN_DIR)
    parser.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    parser.add_argument("--osm-path", type=Path, default=DEFAULT_OSM_PATH)
    parser.add_argument("--track-png", type=Path, default=DEFAULT_TRACK_PNG)
    parser.add_argument("--host", default="http://localhost:5005")
    parser.add_argument(
        "--command-speed-scale", type=float, default=None,
        help="URT_COMMAND_SPEED_CALIBRATION_SCALE para el brain (idem campaign_runner).",
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Debug: no lanzar la GUI. Hace inútil el shadow MPC, pero útil para validar boot.",
    )
    parser.add_argument(
        "--no-record", action="store_true",
        help="No iniciar grabación de cámara desde el dashboard.",
    )
    parser.add_argument("--skip-cleanup-before", action="store_true")
    parser.add_argument("--keep-running", action="store_true",
                        help="Dejar gz_sim/brain/GUI vivos al terminar (debug).")
    parser.add_argument("--skip-analyze", action="store_true",
                        help="No correr manual_shadow_analyze.py al final.")
    return parser.parse_args()


def _launch_gui(brain_dir: Path, brain_py: str, run_dir: Path) -> ManagedProcess:
    """Lanza la GUI PyQt5 apuntando al dashboard local (mismo que `./run.sh` Mac dev)."""
    return _run(
        [brain_py, "-m", "src.dashboard.gui.main", "--connect", "localhost:5005", "--no-login"],
        cwd=brain_dir,
        stdout_path=run_dir / "gui.stdout",
        name="gui",
    )


def _wait_for_stop(
    *,
    stop_event: threading.Event,
    gui_proc: ManagedProcess | None,
    duration_s: float | None,
    started_at: float,
) -> str:
    """Bloquea hasta que se pida stop. Devuelve la razón del stop."""
    print("[manual] sesión en curso. Ctrl+C o cerrar la GUI para terminar.")
    while not stop_event.is_set():
        if gui_proc is not None and gui_proc.popen.poll() is not None:
            return "gui_exited"
        if duration_s is not None and (time.time() - started_at) >= duration_s:
            return "duration_reached"
        time.sleep(0.5)
    return "signal_received"


def main() -> int:
    args = _parse_args()
    brain_dir = args.brain_dir.resolve()
    sim_dir = args.sim_dir.resolve()
    if (args.destination_x is None) != (args.destination_y is None):
        print(
            "[manual] ERROR: --destination-x y --destination-y van juntos",
            file=sys.stderr,
        )
        return 2

    run_id = args.run_id or f"run_manual_{time.strftime('%Y%m%d_%H%M%S')}"
    brain_run_dir = brain_dir / "temp" / "logs" / run_id
    sim_run_dir = sim_dir / "temp" / "logs" / run_id
    brain_jsonl = brain_run_dir / "brain.jsonl"
    sim_jsonl = sim_run_dir / "sim_bridge.jsonl"
    copied_sim_jsonl = brain_run_dir / "sim_bridge.jsonl"

    brain_run_dir.mkdir(parents=True, exist_ok=True)
    sim_run_dir.mkdir(parents=True, exist_ok=True)
    brain_jsonl.write_text("", encoding="utf-8")
    sim_jsonl.write_text("", encoding="utf-8")

    destination_lanelet = None if args.destination_x is not None else args.destination_lanelet
    expected_route, nav_command = _build_expected_route(
        osm_path=args.osm_path.resolve(),
        destination_lanelet=destination_lanelet,
        destination_x=args.destination_x,
        destination_y=args.destination_y,
    )
    (brain_run_dir / "route_expected.json").write_text(
        json.dumps(expected_route, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (brain_run_dir / "navigation_command.json").write_text(
        json.dumps(nav_command, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    brain_py = _find_brain_python(brain_dir)
    gz_py, gz_pypath = _find_gz_python()
    processes: list[ManagedProcess] = []
    client: DashboardClient | None = None
    started_at = time.time()
    record_start_ts: float | None = None
    record_stop_ts: float | None = None
    gui_overlay_video: dict[str, Any] | None = None
    stop_event = threading.Event()
    stop_reason = "unknown"

    def _on_signal(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print(f"[manual] run_id={run_id}")
    print(f"[manual] brain logs: {brain_run_dir}")
    print(f"[manual] sim logs:   {sim_run_dir}")
    print(f"[manual] brain_py={brain_py}")
    print(f"[manual] gz_py={gz_py} PYTHONPATH+={gz_pypath or '<none>'}")

    try:
        if not args.skip_cleanup_before:
            _cleanup_known_processes(brain_dir, sim_dir)

        processes.append(
            _run(
                ["./run_sim.sh", "--headless"],
                cwd=sim_dir,
                stdout_path=sim_run_dir / "sim.stdout",
                name="gz_sim",
            )
        )

        time.sleep(5.0)
        gz_env = os.environ.copy()
        if gz_pypath:
            gz_env["PYTHONPATH"] = f"{gz_pypath}{os.pathsep}{gz_env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
        processes.append(
            _run(
                [
                    gz_py,
                    str(sim_dir / "tools" / "dev" / "gz_pose_logger.py"),
                    "--out",
                    str(sim_jsonl),
                ],
                cwd=sim_dir,
                stdout_path=sim_run_dir / "gz_logger.stdout",
                env=gz_env,
                name="gz_pose_logger",
            )
        )

        print("[manual] esperando sim_bridge :5575")
        if not _wait_port("localhost", 5575, 60.0):
            raise RuntimeError("sim_bridge ZMQ port :5575 no levantó")

        brain_env = os.environ.copy()
        brain_env["URT_LIVE_LOG_PATH"] = str(brain_jsonl)
        brain_env["URT_SIM_MODE"] = "1"
        brain_env["URT_SHADOW_PLANNER"] = "1"
        brain_env.setdefault("URT_DISABLE_AUTO_PARKING", "1")
        # Shadow analysis requires the real MPC solver. URT_SIM_MODE=1 defaults
        # FORCE_PURE_PURSUIT to True on Mac; override so run.sh uses Acados.
        # If Acados libs are missing the brain will crash immediately with
        # AcadosBackendUnavailableError — no silent fallback.
        brain_env.setdefault("URT_FORCE_PURE_PURSUIT", "0")
        # Mode env del runner viejo: solo aplica si pedimos modo "odometry".
        # Acá usamos MANUAL en runtime; pero respetamos `--command-speed-scale`
        # por si el operador quiere comparar con la misma calibración.
        _apply_mode_env(
            brain_env,
            mode="manual",
            expected_route=expected_route,
            command_speed_scale=args.command_speed_scale,
        )
        processes.append(
            _run(
                ["./run.sh", "--no-gui"],
                cwd=brain_dir,
                stdout_path=brain_run_dir / "brain.stdout",
                env=brain_env,
                name="brain",
            )
        )

        print("[manual] esperando dashboard :5005")
        if not _wait_port("localhost", 5005, 60.0):
            raise RuntimeError("brain dashboard :5005 no levantó")

        print("[manual] esperando primer pose_published")
        _wait_for_jsonl_event(
            brain_jsonl,
            thread="pose_estimator",
            event="pose_published",
            timeout_s=90.0,
        )

        client = DashboardClient(args.host)
        client.connect()
        # IMPORTANTE: NO pedimos SessionAccess. El dashboard solo permite
        # una sesión activa por vez; si la tomamos nosotros, la GUI queda
        # como "unauthorized user" y el dashboard descarta silenciosamente
        # todos los SpeedMotor/SteerMotor que ella mande
        # (`processDashboard.handle_message` line 366-368: el check
        # `elif self.sessionActive and self.activeUser != socketId: return`
        # bloquea cualquier mensaje del que no posee la sesión).
        #
        # Sin sesión nuestra, `sessionActive=False`, ese check no aplica, y
        # NavigationCommand/Record pasan al brain via `send_message_to_brain`.
        # Después la GUI levanta, pide sesión y la obtiene — desde ahí ella
        # es la única autorizada a mandar comandos de manejo.
        time.sleep(1.0)

        if not args.no_record:
            client.start_gui_video_recording(brain_run_dir / "camera_gui_overlay.avi", fps=7.0)
            client.emit_message("Record", True, wait_response=True)
            record_start_ts = time.time()
            time.sleep(0.5)

        # Armar la ruta esperada (sin cambiar a AUTO). El BehaviorPlanner
        # shadow ya está corriendo y la usará como path target del MPC.
        nav_ok = False
        for attempt in range(1, 4):
            print(f"[manual] enviando NavigationCommand attempt {attempt}: {nav_command}")
            if not client.emit_message("NavigationCommand", nav_command, wait_response=True):
                print("[manual] WARNING: dashboard no ackeó NavigationCommand")
            if _wait_for_navigation_acceptance(brain_jsonl, 5.0):
                nav_ok = True
                break
            time.sleep(0.5)
        if not nav_ok:
            print(
                "[manual] WARNING: navigation command no fue aceptado. La sesión sigue, "
                "pero el MPC shadow no va a tener path target y emitirá compute_out inválidos."
            )

        if not args.no_gui:
            gui_proc = _launch_gui(brain_dir, brain_py, brain_run_dir)
            processes.append(gui_proc)
        else:
            gui_proc = None
            print("[manual] --no-gui activo: sesión sin GUI (solo backend + sim).")

        stop_reason = _wait_for_stop(
            stop_event=stop_event,
            gui_proc=gui_proc,
            duration_s=args.duration,
            started_at=started_at,
        )
        print(f"[manual] stop_reason={stop_reason}")

        if not args.no_record:
            try:
                client.emit_message("Record", False, wait_response=True)
                record_stop_ts = time.time()
            except Exception:
                pass
            gui_overlay_video = client.stop_gui_video_recording()
        time.sleep(0.5)

    except Exception as exc:
        print(f"[manual] ERROR: {exc}", file=sys.stderr)
        stop_reason = "exception"
    finally:
        if client is not None:
            if gui_overlay_video is None:
                gui_overlay_video = client.stop_gui_video_recording()
            client.close()
        if not args.keep_running:
            for proc in reversed(processes):
                print(f"[manual] deteniendo {proc.name}")
                proc.terminate()
            _cleanup_known_processes(brain_dir, sim_dir)

    if sim_jsonl.exists():
        shutil.copy2(sim_jsonl, copied_sim_jsonl)

    # ----------------------------------------------------------------
    # Artefactos derivados — mismo set que `campaign_runner.py` produce
    # tras cerrar la run: video crudo de cámara, GUI overlay promovido,
    # extracción de frames, PNG con expected vs actual, debug overlay
    # video con panel de telemetría, y el campaign_digest (markdown +
    # frames anotados). Reusamos las funciones del runner viejo vía
    # import; lo único nuevo es construir `actual_xy` desde el ground
    # truth, porque el runner viejo lo derivaba dentro de
    # `_compute_metrics` (que está atado a la lógica de pass/fail
    # autónomo y no aplica acá).
    # ----------------------------------------------------------------
    raw_video_path: str | None = (
        None if args.no_record else _collect_camera_video(brain_dir, brain_run_dir, started_at)
    )
    video_path, raw_video_path = _promote_gui_overlay_camera_video(
        run_dir=brain_run_dir,
        raw_video_path=raw_video_path,
        gui_overlay_video=gui_overlay_video,
    )
    video_frames: dict[str, Any] | None = None
    if video_path:
        video_frames = _extract_video_frames(video_path, brain_run_dir)

    brain_events = _load_jsonl(brain_jsonl)
    gt_events = _load_jsonl(copied_sim_jsonl)
    actual_xy = _ground_truth_xy(gt_events)

    visual_samples = _collect_visual_lane_samples(
        brain_events,
        record_start_ts=record_start_ts,
        fps=float(video_frames.get("fps", 0.0) or 0.0) if video_frames else None,
    )

    overlay_written = _write_overlay(
        out_path=brain_run_dir / "overlay.png",
        track_png=args.track_png.resolve(),
        expected_route=expected_route,
        actual_xy=actual_xy,
    )
    visual_overlay_written = _write_visual_lane_overlay(
        out_path=brain_run_dir / "overlay_visual_lane.png",
        track_png=args.track_png.resolve(),
        expected_route=expected_route,
        actual_xy=actual_xy,
        visual_samples=visual_samples,
    )

    # Debug overlay video: si tenemos el GUI overlay (que ya trae la
    # máscara IA superpuesta), usamos ese como source y solo agregamos
    # el panel derecho. Si no, fallback al raw y dibujamos la lane
    # overlay nosotros.
    debug_overlay_video: dict[str, Any] | None = None
    debug_source_video: str | None = None
    debug_draw_lane_overlay = True
    if gui_overlay_video is not None and gui_overlay_video.get("written") and gui_overlay_video.get("path"):
        debug_source_video = str(gui_overlay_video["path"])
        debug_draw_lane_overlay = False
    elif video_path:
        debug_source_video = video_path
    if debug_source_video:
        debug_overlay_video = _write_debug_overlay_video(
            video_path=debug_source_video,
            out_path=brain_run_dir / "camera_debug_overlay.avi",
            brain_events=brain_events,
            record_start_ts=(
                _finite_float(gui_overlay_video.get("started_at"))
                if not debug_draw_lane_overlay and gui_overlay_video is not None
                else record_start_ts
            ),
            draw_lane_overlay=debug_draw_lane_overlay,
            frame_timestamps=(
                gui_overlay_video.get("_frame_timestamps")
                if not debug_draw_lane_overlay and gui_overlay_video is not None
                else None
            ),
        )

    campaign_digest = _write_campaign_digest(
        run_dir=brain_run_dir,
        brain_events=brain_events,
        video_frames=video_frames,
        frame_timestamps=(
            gui_overlay_video.get("_frame_timestamps")
            if gui_overlay_video is not None
            else None
        ),
        debug_video_path=(
            str(brain_run_dir / "camera_debug_overlay.avi")
            if debug_overlay_video is not None and debug_overlay_video.get("written")
            else None
        ),
        track_png=args.track_png.resolve(),
        expected_route=expected_route,
        actual_xy=actual_xy,
    )

    session_summary = {
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": time.time(),
        "duration_s": time.time() - started_at,
        "stop_reason": stop_reason,
        "record_start_ts": record_start_ts,
        "record_stop_ts": record_stop_ts,
        "gui_launched": not args.no_gui,
        "no_record": bool(args.no_record),
        "destination": (
            {"lanelet_id": args.destination_lanelet}
            if args.destination_x is None
            else {"x": args.destination_x, "y": args.destination_y}
        ),
        "expected_route_waypoint_count": int(expected_route.get("waypoint_count", 0)),
        "gui_overlay_video": (
            {k: v for k, v in gui_overlay_video.items() if k != "_frame_timestamps"}
            if gui_overlay_video is not None
            else None
        ),
        "ground_truth_samples": len(actual_xy),
        "visual_lane_samples": len(visual_samples),
        "debug_overlay_video": (
            {k: v for k, v in (debug_overlay_video or {}).items() if k != "_frame_timestamps"}
            if debug_overlay_video
            else None
        ),
        "campaign_digest": (
            {k: v for k, v in (campaign_digest or {}).items() if k != "rows"}
            if campaign_digest
            else None
        ),
        "video_frames": (
            {k: v for k, v in video_frames.items() if k != "frames"}
            if video_frames
            else None
        ),
        "artifacts": {
            "brain_jsonl": str(brain_jsonl),
            "sim_bridge_jsonl": str(copied_sim_jsonl),
            "route_expected_json": str(brain_run_dir / "route_expected.json"),
            "navigation_command_json": str(brain_run_dir / "navigation_command.json"),
        },
    }
    if video_path:
        session_summary["artifacts"]["camera_video"] = video_path
        if raw_video_path and raw_video_path != video_path:
            session_summary["artifacts"]["camera_raw_video"] = raw_video_path
        if video_frames:
            session_summary["artifacts"]["camera_frames_dir"] = str(brain_run_dir / "frames")
            session_summary["artifacts"]["camera_frames_index_json"] = str(brain_run_dir / "frames" / "index.json")
    if gui_overlay_video is not None and gui_overlay_video.get("written"):
        session_summary["artifacts"]["camera_gui_overlay_video"] = str(gui_overlay_video.get("path"))
        if gui_overlay_video.get("frame_index_path"):
            session_summary["artifacts"]["camera_gui_overlay_frame_index_json"] = str(
                gui_overlay_video.get("frame_index_path")
            )
    if overlay_written:
        session_summary["artifacts"]["overlay_png"] = str(brain_run_dir / "overlay.png")
    if visual_overlay_written:
        session_summary["artifacts"]["overlay_visual_lane_png"] = str(brain_run_dir / "overlay_visual_lane.png")
    if debug_overlay_video is not None and debug_overlay_video.get("written"):
        session_summary["artifacts"]["camera_debug_overlay_video"] = str(brain_run_dir / "camera_debug_overlay.avi")
    if campaign_digest and campaign_digest.get("written"):
        if campaign_digest.get("digest_md"):
            session_summary["artifacts"]["campaign_digest_md"] = str(campaign_digest.get("digest_md"))
        if campaign_digest.get("digest_json"):
            session_summary["artifacts"]["campaign_digest_json"] = str(campaign_digest.get("digest_json"))

    (brain_run_dir / "session.json").write_text(
        json.dumps(session_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if not args.skip_analyze:
        analyzer = brain_dir / "tools" / "dev" / "manual_shadow_analyze.py"
        if analyzer.exists():
            report_txt = brain_run_dir / "manual_vs_mpc.txt"
            with report_txt.open("w", encoding="utf-8") as out:
                subprocess.run(
                    [
                        brain_py, str(analyzer), str(brain_run_dir),
                        "--sim-jsonl", str(copied_sim_jsonl),
                        "--route-json", str(brain_run_dir / "route_expected.json"),
                    ],
                    cwd=str(brain_dir),
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            session_summary["artifacts"]["manual_vs_mpc_txt"] = str(report_txt)
            mvm_json = brain_run_dir / "manual_vs_mpc.json"
            if mvm_json.exists():
                session_summary["artifacts"]["manual_vs_mpc_json"] = str(mvm_json)
            (brain_run_dir / "session.json").write_text(
                json.dumps(session_summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        else:
            print(f"[manual] WARNING: analyzer no encontrado en {analyzer}")

    _update_latest_symlink(brain_dir / "temp" / "logs", brain_run_dir)
    _update_latest_symlink(sim_dir / "temp" / "logs", sim_run_dir)

    print(f"[manual] session.json -> {brain_run_dir / 'session.json'}")
    return 0


if __name__ == "__main__":
    _maybe_reexec_with_brain_venv()
    sys.exit(main())
