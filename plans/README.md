# BFMC plans landing zone

This folder is the default location where the brain looks for BFMC plan JSONs
when the operator clicks **Load BFMC plan** in the dashboard.

## How to populate

From `bfmc-waypoint-planner/`:

```bash
python3 tools/sync_to_brain.py --brain-path ../urt-brain-bosch --plan NAME
```

This copies `data/outputs/plans/NAME/plan.json` to `urt-brain-bosch/plans/NAME.json`.

## Schema

Each plan JSON is the planner output (`plan.json`) and must contain at least:

- `lanelet_sequence`: ordered list of lanelet IDs to follow.
- `waypoint_sequence`: list of waypoint IDs (used by analysis tools).
- `start_pose`: `{x, y, yaw_rad}` — the operator must place the car here before
  starting the route.

The brain validates connectivity (par a par) against its topology and refuses
plans whose sequence has gaps.

## Git

The `*.json` files dropped here are working artifacts and are ignored by git
(see `.gitignore`). This README is tracked.
