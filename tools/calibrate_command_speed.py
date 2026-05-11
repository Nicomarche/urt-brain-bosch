#!/usr/bin/env python3
"""Compute TRACKING_COMMAND_SPEED_CALIBRATION_SCALE from straight runs.

Example:
    python tools/calibrate_command_speed.py --duration 5 \
        0.20 0.93 \
        0.25 1.18 \
        0.30 1.44

Each pair is:
    commanded_speed_mps measured_distance_m

The output factor is:
    measured_distance_m / (commanded_speed_mps * duration_s)
"""

from __future__ import annotations

import argparse
import statistics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate command-speed odometry scale for no-encoder mode."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Run duration in seconds for each measurement. Default: 5.0",
    )
    parser.add_argument(
        "values",
        nargs="+",
        type=float,
        help="Pairs: commanded_speed_mps measured_distance_m",
    )
    args = parser.parse_args()
    if len(args.values) % 2:
        parser.error("values must be pairs: commanded_speed_mps measured_distance_m")
    if args.duration <= 0.0:
        parser.error("--duration must be > 0")
    return args


def main() -> int:
    args = _parse_args()
    factors: list[float] = []
    rows: list[tuple[float, float, float, float]] = []

    it = iter(args.values)
    for commanded_mps, measured_m in zip(it, it):
        if commanded_mps == 0.0:
            raise SystemExit("commanded_speed_mps cannot be 0")
        expected_m = float(commanded_mps) * float(args.duration)
        factor = float(measured_m) / expected_m
        factors.append(factor)
        rows.append((float(commanded_mps), float(measured_m), expected_m, factor))

    mean_factor = statistics.fmean(factors)
    stdev = statistics.pstdev(factors) if len(factors) > 1 else 0.0

    print("commanded_mps measured_m expected_m factor")
    for commanded_mps, measured_m, expected_m, factor in rows:
        print(f"{commanded_mps:>12.3f} {measured_m:>10.3f} {expected_m:>10.3f} {factor:>7.4f}")
    print()
    print(f"TRACKING_COMMAND_SPEED_CALIBRATION_SCALE = {mean_factor:.4f}")
    print(f"# spread = {stdev:.4f}")
    print(f"export URT_COMMAND_SPEED_CALIBRATION_SCALE={mean_factor:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
