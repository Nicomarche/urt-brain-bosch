"""Helper to load a BFMC waypoint-planner ``plan.json`` and emit it to the brain.

Used by the Map toolbar's ``Load BFMC plan`` button. The brain executes the
plan via the ``lanelet_sequence`` navigation mode added in
``src/routing/route_planner.py`` (see plan: "Integrar topología corregida +
ejecución de planes BFMC").
"""

from __future__ import annotations

import json
import os

from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget

from ..client import events as ev


_BRAIN_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)


def _default_plans_dir() -> str:
    brain_plans = os.path.join(_BRAIN_ROOT, "plans")
    if os.path.isdir(brain_plans):
        return brain_plans
    fallback = os.path.expanduser(
        "~/Desktop/Projects/BFMC/bfmc-waypoint-planner/data/outputs/plans/"
    )
    if os.path.isdir(fallback):
        return fallback
    return os.path.expanduser("~")


def load_and_emit_bfmc_plan(parent: QWidget, client) -> None:
    """Open a file picker, parse a plan JSON and emit a ``lanelet_sequence`` command.

    The dialog defaults to ``urt-brain-bosch/plans/`` (the landing zone populated
    by ``bfmc-waypoint-planner/tools/sync_to_brain.py --plan NAME``).
    """
    fname, _ = QFileDialog.getOpenFileName(
        parent,
        "Load BFMC plan",
        _default_plans_dir(),
        "BFMC plan (*.json)",
    )
    if not fname:
        return
    try:
        with open(fname, "r", encoding="utf-8") as fh:
            plan = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        QMessageBox.warning(parent, "Load failed", f"Could not parse plan: {exc}")
        return

    lanelet_ids = plan.get("lanelet_sequence") or []
    if not lanelet_ids:
        QMessageBox.warning(parent, "Invalid plan", "Plan has no 'lanelet_sequence'.")
        return

    start_pose = plan.get("start_pose") or {}
    if start_pose:
        try:
            sp_x = float(start_pose.get("x", 0.0))
            sp_y = float(start_pose.get("y", 0.0))
            sp_yaw = float(start_pose.get("yaw_rad", 0.0))
        except (TypeError, ValueError):
            sp_x = sp_y = sp_yaw = None  # type: ignore[assignment]
        if sp_x is not None:
            resp = QMessageBox.information(
                parent,
                "Place the car",
                (
                    "Place the car at:\n"
                    f"  x = {sp_x:.3f} m\n"
                    f"  y = {sp_y:.3f} m\n"
                    f"  yaw = {sp_yaw:.3f} rad\n\n"
                    "Then press OK to send the plan to the brain."
                ),
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Ok,
            )
            if resp != QMessageBox.Ok:
                return

    plan_name = (
        os.path.basename(os.path.dirname(fname))
        or os.path.splitext(os.path.basename(fname))[0]
    )
    payload = {
        "mode": "lanelet_sequence",
        "lanelet_ids": [str(x) for x in lanelet_ids],
        "plan_name": plan_name,
    }
    client.emit_message(ev.CMD_NAV_COMMAND, payload)
    QMessageBox.information(
        parent,
        "Plan loaded",
        (
            f"Sent {len(lanelet_ids)} lanelets to the brain "
            f"(plan '{plan_name}').\n"
            "Watch the route overlay on the map for activation. "
            "If the brain rejects the plan, check the logs."
        ),
    )
