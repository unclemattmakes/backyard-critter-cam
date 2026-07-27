"""
Tests for WhiteBalanceWatchdog (backyard_cam) -- the guard that pulls the glass-door camera
back out of its broken MANUAL white balance.

Background (measured on the live rig, 2026-07-25): with AUTO_WB off this camera renders red
~40% below green at EVERY colour temperature from 2800 K to 6500 K, which is the cyan/green
cast on the washed-out crops; with AUTO_WB on it sits at R/G ~0.98. So the watchdog watches the
frame's red/green ratio and re-asserts auto when it drops into the red-starved band.

These drive the pure logic against synthetic frames and a fake capture -- no camera, no cv2
capture, no sleeping. Time is passed in explicitly, so the rate limiting is testable.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import backyard_cam
import config


class FakeCap:
    """Stands in for cv2.VideoCapture: records every property set on it."""

    def __init__(self, fail: bool = False):
        self.sets: list[tuple[int, float]] = []
        self.fail = fail

    def set(self, prop, val):
        if self.fail:
            raise RuntimeError("driver rejected the write")
        self.sets.append((prop, val))
        return True


def frame(r: int, g: int, b: int, size: int = 64):
    """A flat BGR frame with the given channel levels."""
    f = np.zeros((size, size, 3), dtype=np.uint8)
    f[:, :, 0], f[:, :, 1], f[:, :, 2] = b, g, r
    return f


NEUTRAL = frame(120, 120, 120)          # R/G = 1.00 -- healthy
STARVED = frame(72, 120, 120)           # R/G = 0.60 -- the measured broken manual-WB state
DARK_STARVED = frame(12, 20, 20)        # same ratio, too dark to judge (a warm night)


def cfg(**over):
    base = dataclasses.replace(config.CONFIG, wb_recover_interval_s=0.0, wb_recover_strikes=2)
    return dataclasses.replace(base, **over) if over else base


def wd(**over):
    return backyard_cam.WhiteBalanceWatchdog(cfg(**over), tag="[test]")


def auto_wb_writes(cap):
    import cv2
    return [v for p, v in cap.sets if p == cv2.CAP_PROP_AUTO_WB]


# ---- the trip decision ---------------------------------------------------------------

def test_neutral_frame_is_left_alone():
    g, cap = wd(), FakeCap()
    for t in range(6):
        assert g.check(cap, NEUTRAL, float(t)) is False
    assert cap.sets == []
    assert g.last_ratio == pytest.approx(1.0, abs=0.01)


def test_red_starved_frame_restores_auto_wb_after_the_strike_count():
    g, cap = wd(), FakeCap()
    assert g.check(cap, STARVED, 0.0) is False      # strike 1 -- one odd frame is not enough
    assert cap.sets == []
    assert g.check(cap, STARVED, 1.0) is True       # strike 2 -- act
    assert auto_wb_writes(cap) == [1.0]
    assert g.recoveries == 1


def test_a_neutral_frame_between_strikes_resets_the_count():
    g, cap = wd(), FakeCap()
    g.check(cap, STARVED, 0.0)
    g.check(cap, NEUTRAL, 1.0)                      # recovered on its own
    assert g.check(cap, STARVED, 2.0) is False      # back to strike 1, not 2
    assert cap.sets == []


def test_a_dark_frame_is_never_judged():
    """A night scene under an amber yard light is legitimately far from neutral."""
    g, cap = wd(), FakeCap()
    for t in range(6):
        assert g.check(cap, DARK_STARVED, float(t)) is False
    assert cap.sets == []


# ---- honouring a deliberate manual choice --------------------------------------------

def test_dashboard_manual_wb_stands_the_watchdog_down():
    g, cap = wd(), FakeCap()
    g.note_settings({"AUTO_WB": 0, "WB_TEMPERATURE": 4600})
    assert g.user_manual is True
    for t in range(6):
        assert g.check(cap, STARVED, float(t)) is False
    assert cap.sets == []


def test_switching_auto_back_on_re_arms_the_watchdog():
    g, cap = wd(), FakeCap()
    g.note_settings({"AUTO_WB": 0})
    g.note_settings({"AUTO_WB": 1})
    assert g.user_manual is False
    g.check(cap, STARVED, 0.0)
    assert g.check(cap, STARVED, 1.0) is True


def test_unrelated_settings_do_not_change_the_manual_flag():
    g = wd()
    g.note_settings({"AUTO_WB": 0})
    g.note_settings({"CONTRAST": 50, "GAMMA": 110})
    assert g.user_manual is True


# ---- rate limiting, disabling, and failure ------------------------------------------

def test_checks_are_rate_limited_to_the_configured_interval():
    g, cap = wd(wb_recover_interval_s=20.0), FakeCap()
    assert g.check(cap, STARVED, 100.0) is False    # strike 1, next check at t=120
    assert g.check(cap, STARVED, 110.0) is False    # inside the window -- not sampled at all
    assert g.check(cap, STARVED, 119.9) is False
    assert g.check(cap, STARVED, 120.0) is True     # strike 2
    assert auto_wb_writes(cap) == [1.0]


def test_disabled_watchdog_never_touches_the_camera():
    g, cap = wd(wb_auto_recover=False), FakeCap()
    for t in range(6):
        assert g.check(cap, STARVED, float(t)) is False
    assert cap.sets == []


def test_a_missing_frame_is_ignored():
    g, cap = wd(), FakeCap()
    assert g.check(cap, None, 0.0) is False
    assert cap.sets == []


def test_a_driver_that_rejects_the_write_does_not_raise():
    """The rig must survive a camera that refuses AUTO_WB -- capture matters more."""
    g, cap = wd(), FakeCap(fail=True)
    g.check(cap, STARVED, 0.0)
    assert g.check(cap, STARVED, 1.0) is False
    assert g.recoveries == 0


def test_ratio_is_sampled_from_a_non_uniform_frame():
    """The strided sample must still read a real gradient frame, not just flat colour."""
    f = np.zeros((80, 80, 3), dtype=np.uint8)
    f[:, :, 1] = np.tile(np.linspace(60, 200, 80, dtype=np.uint8), (80, 1))   # green ramp
    f[:, :, 2] = f[:, :, 1] // 2                                             # red = half of it
    ratio, luma = backyard_cam.WhiteBalanceWatchdog._sample(f)
    assert ratio == pytest.approx(0.5, abs=0.02)
    assert luma > 0
