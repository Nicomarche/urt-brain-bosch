"""Tests de `src/localization/lane_correction.py`."""

from __future__ import annotations

import pytest

from src.localization.lane_correction import compute_lateral_correction
from src.localization.visual_correction_profile import VisualCorrectionProfile


DEFAULT_PROFILE = VisualCorrectionProfile(gain=0.2, max_m=0.05, step_max_m=0.03)


def test_disabled_profile_no_correction():
    profile = VisualCorrectionProfile(gain=0.0, max_m=0.0, step_max_m=0.0)
    r = compute_lateral_correction(
        lateral_error_m=0.10,
        speed_mps=0.5,
        now_s=10.0,
        last_correction_time_s=0.0,
        profile=profile,
    )
    assert not r.applied
    assert r.correction_m == 0.0
    assert r.reason == "profile_disabled"


def test_low_speed_inhibits():
    r = compute_lateral_correction(
        lateral_error_m=0.20,
        speed_mps=0.0,
        now_s=10.0,
        last_correction_time_s=0.0,
        profile=DEFAULT_PROFILE,
    )
    assert not r.applied
    assert r.reason == "speed_below_min"


def test_normal_case_applies_with_gain():
    r = compute_lateral_correction(
        lateral_error_m=0.10,
        speed_mps=0.5,
        now_s=10.0,
        last_correction_time_s=0.0,
        profile=DEFAULT_PROFILE,
    )
    assert r.applied
    # gain=0.2, error=0.10 → raw=0.02. step_max=0.03, max=0.05. No clamp.
    assert r.correction_m == pytest.approx(0.02)
    assert r.raw_correction_m == pytest.approx(0.02)


def test_clamp_by_max():
    profile = VisualCorrectionProfile(gain=1.0, max_m=0.05, step_max_m=0.04)
    r = compute_lateral_correction(
        lateral_error_m=0.10,  # raw = 0.10
        speed_mps=0.5,
        now_s=10.0,
        last_correction_time_s=0.0,
        profile=profile,
    )
    # raw=0.10 → clamped by max_m=0.05 → 0.05 → step_max=0.04 lo recorta
    assert r.applied
    assert r.correction_m == pytest.approx(0.04)


def test_clamp_negative():
    profile = VisualCorrectionProfile(gain=1.0, max_m=0.05, step_max_m=0.10)
    r = compute_lateral_correction(
        lateral_error_m=-0.20,
        speed_mps=0.5,
        now_s=10.0,
        last_correction_time_s=0.0,
        profile=profile,
    )
    assert r.applied
    assert r.correction_m == pytest.approx(-0.05)


def test_sign_preserved():
    profile = VisualCorrectionProfile(gain=0.5, max_m=1.0, step_max_m=1.0)
    pos = compute_lateral_correction(
        lateral_error_m=0.04, speed_mps=0.5, now_s=1.0, last_correction_time_s=0.0, profile=profile
    )
    neg = compute_lateral_correction(
        lateral_error_m=-0.04, speed_mps=0.5, now_s=1.0, last_correction_time_s=0.0, profile=profile
    )
    assert pos.correction_m == -neg.correction_m
