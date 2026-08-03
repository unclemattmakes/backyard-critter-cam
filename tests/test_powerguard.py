"""Tests for powerguard.py -- the battery watch + USB-wedge detect->heal->replug ladder.

These pin the state machine against the two REAL wedges it was built from (2026-07-30 striped
static / 2026-07-31 torn slabs) and the false-positive cases that must NOT trip it (a close
animal filling the frame, a stale WB streak after a night-time heal). Pure python; the module
deliberately imports no cv2/torch, so these run anywhere.
"""
from __future__ import annotations

from types import SimpleNamespace

import powerguard


def _cfg(**over):
    d = dict(
        power_warn=True, power_poll_s=30.0, power_warn_repeat_s=600.0,
        wedge_guard=True, wedge_wb_failures=2,
        wedge_motion_frac=0.55, wedge_motion_sustain_s=60.0, wedge_clear_s=30.0,
        wedge_self_heal=True, wedge_heal_task="BackyardCritterCam-UsbReset",
        wedge_heal_verify_s=90.0, wedge_heal_max_per_hour=2,
    )
    d.update(over)
    return SimpleNamespace(**d)


class _Healer:
    """Scripted stand-in: try_heal answers from a list, and remembers being asked."""

    def __init__(self, answers=(), available=True):
        self._answers = list(answers)
        self._available = available
        self.calls = []

    def available(self):
        return self._available

    def try_heal(self, now, reason):
        self.calls.append((now, reason))
        return self._answers.pop(0) if self._answers else False


# ---- interpret_power ----------------------------------------------------------------
def test_interpret_power_maps_ac_battery_and_unknowns():
    assert powerguard.interpret_power(1, 80) == (True, 80)
    assert powerguard.interpret_power(0, 43) == (False, 43)
    assert powerguard.interpret_power(255, 255) == (None, None)   # both unknown
    assert powerguard.interpret_power(0, 255) == (False, None)    # pct unknown, state known


# ---- PowerMonitor -------------------------------------------------------------------
def test_powermonitor_warns_on_battery_and_clears_on_ac():
    msgs = []
    state = {"v": (True, 100)}
    mon = powerguard.PowerMonitor(_cfg(), notify=msgs.append, reader=lambda: state["v"])
    mon.poll(0.0)
    assert mon.warning is None and not msgs
    state["v"] = (False, 43)
    mon.poll(10.0)
    assert "ON BATTERY (43%)" in mon.warning
    assert len(msgs) == 1 and "WARNING" in msgs[0]
    mon.poll(20.0)                       # still on battery, inside the repeat window: no re-print
    assert len(msgs) == 1
    mon.poll(10.0 + 601.0)               # past power_warn_repeat_s: nag again
    assert len(msgs) == 2
    state["v"] = (True, 44)
    mon.poll(700.0)
    assert mon.warning is None
    assert "back on AC" in msgs[-1]


def test_powermonitor_unknown_state_stays_quiet():
    msgs = []
    mon = powerguard.PowerMonitor(_cfg(), notify=msgs.append, reader=lambda: (None, None))
    mon.poll(0.0)
    assert mon.warning is None and not msgs


# ---- WedgeDetector: the two real signatures -----------------------------------------
def test_pegged_motion_with_no_detections_wedges_and_asks_for_replug():
    msgs = []
    w = powerguard.WedgeDetector(_cfg(), "[cam]", healer=None, notify=msgs.append)
    for t in range(0, 70):
        w.note_motion(float(t), 0.98)    # the 07-31 torn-slab frame: ~98% one blob
        w.update(float(t))
    assert w.state == w.REPLUG
    assert "unplug/replug" in w.message
    assert any("CAMERA WEDGE detected" in m for m in msgs)
    assert any("setup_selfheal" in m for m in msgs)   # the no-task hint, once


def test_wb_failed_streak_wedges():
    w = powerguard.WedgeDetector(_cfg(), "[cam]", healer=None, notify=lambda m: None)
    w.note_motion(0.0, 0.01)             # motion calm: this is the 07-30 variant
    w.note_wb(2)                         # two AUTO_WB re-asserts that didn't take
    w.update(1.0)
    assert w.state == w.REPLUG


def test_a_detection_inside_the_pegged_window_vetoes_the_motion_rule():
    w = powerguard.WedgeDetector(_cfg(), "[cam]", healer=None, notify=lambda m: None)
    for t in range(0, 120):
        w.note_motion(float(t), 0.98)    # a raccoon nose-to-glass fills the frame...
        if t == 30:
            w.note_detections(30.0)      # ...but MegaDetector SEES it
        w.update(float(t))
    assert w.state == w.OK


def test_motion_dipping_below_the_threshold_resets_the_pegged_clock():
    w = powerguard.WedgeDetector(_cfg(), "[cam]", healer=None, notify=lambda m: None)
    for t in range(0, 50):
        w.note_motion(float(t), 0.98)
        w.update(float(t))
    w.note_motion(50.0, 0.10)            # gate closes: the clock must restart
    w.update(50.0)
    for t in range(51, 100):
        w.note_motion(float(t), 0.98)
        w.update(float(t))
    assert w.state == w.OK               # neither stretch reached 60 s on its own


def test_note_reconnect_clears_motion_evidence():
    w = powerguard.WedgeDetector(_cfg(), "[cam]", healer=None, notify=lambda m: None)
    for t in range(0, 50):
        w.note_motion(float(t), 0.98)
    w.note_reconnect(50.0)
    for t in range(51, 100):
        w.note_motion(float(t), 0.98)
        w.update(float(t))
    assert w.state == w.OK


# ---- WedgeDetector: the heal ladder -------------------------------------------------
def test_heal_then_clean_returns_to_ok():
    msgs = []
    h = _Healer(answers=[True])
    w = powerguard.WedgeDetector(_cfg(), "[cam]", healer=h, notify=msgs.append)
    for t in range(0, 61):
        w.note_motion(float(t), 0.98)
        w.update(float(t))
    assert w.state == w.HEALING and len(h.calls) == 1
    assert "resetting its USB device" in w.message
    # post-reset: frames come back clean -> signature-free for wedge_clear_s -> ok
    for t in range(61, 95):
        w.note_motion(float(t), 0.02)
        w.update(float(t))
    assert w.state == w.OK and w.message is None
    assert any("healthy again" in m for m in msgs)


def test_failed_heal_escalates_to_second_attempt_then_replug():
    h = _Healer(answers=[True, True, False])   # two resets granted, then the budget says no
    w = powerguard.WedgeDetector(_cfg(), "[cam]", healer=h, notify=lambda m: None)
    t = 0.0
    while w.state == w.OK:                     # first trigger -> heal #1
        w.note_motion(t, 0.98)
        w.update(t)
        t += 1.0
    assert w.state == w.HEALING
    # the wedge persists: still pegged straight through the grace window and beyond
    for _ in range(0, 200):                    # 90 s grace + a fresh 60 s pegged stretch
        w.note_motion(t, 0.98)
        w.update(t)
        t += 1.0
    assert len(h.calls) == 3                   # heal #2, then the refused third ask
    assert w.state == w.REPLUG
    assert "didn't hold" in w.message


def test_stale_wb_streak_after_a_heal_is_not_fresh_evidence():
    # Night-time heal: the watchdog can't judge dark frames, so its failed_streak stays frozen
    # at the pre-heal value. That stale number must not re-trigger the wedge after the grace.
    h = _Healer(answers=[True])
    w = powerguard.WedgeDetector(_cfg(), "[cam]", healer=h, notify=lambda m: None)
    w.note_motion(0.0, 0.01)
    w.note_wb(2)
    w.update(1.0)
    assert w.state == w.HEALING
    t = 2.0
    for _ in range(0, 200):                    # streak stays 2 (frozen), motion calm
        w.note_motion(t, 0.01)
        w.note_wb(2)
        w.update(t)
        t += 1.0
    assert w.state == w.OK                     # cleared via the baseline, no false re-trigger
    assert len(h.calls) == 1


def test_disabled_guard_does_nothing():
    w = powerguard.WedgeDetector(_cfg(wedge_guard=False), "[cam]", notify=lambda m: None)
    for t in range(0, 200):
        w.note_motion(float(t), 0.98)
        w.update(float(t))
    assert w.state == w.OK and w.message is None


# ---- SelfHealer ---------------------------------------------------------------------
def _runner_factory(calls, query_rc=0, run_rc=0, stderr=""):
    def runner(cmd, **kw):
        calls.append(cmd)
        rc = query_rc if "/query" in cmd else run_rc
        return SimpleNamespace(returncode=rc, stdout="", stderr=stderr)
    return runner


def test_selfhealer_runs_the_registered_task(monkeypatch):
    monkeypatch.setattr(powerguard.sys, "platform", "win32")
    calls, msgs = [], []
    h = powerguard.SelfHealer(_cfg(), "[usb-heal]", notify=msgs.append,
                              runner=_runner_factory(calls))
    assert h.available()
    assert h.try_heal(0.0, "test")
    h.last_thread.join(timeout=5)
    assert ["schtasks", "/run", "/tn", "BackyardCritterCam-UsbReset"] in calls
    assert any("reset task started" in m for m in msgs)


def test_selfhealer_unregistered_task_means_unavailable(monkeypatch):
    monkeypatch.setattr(powerguard.sys, "platform", "win32")
    h = powerguard.SelfHealer(_cfg(), runner=_runner_factory([], query_rc=1),
                              notify=lambda m: None)
    assert not h.available()
    assert not h.try_heal(0.0, "test")


def test_selfhealer_access_denied_notifies_the_setup_hint(monkeypatch):
    monkeypatch.setattr(powerguard.sys, "platform", "win32")
    msgs = []
    h = powerguard.SelfHealer(_cfg(), notify=msgs.append,
                              runner=_runner_factory([], run_rc=1, stderr="ERROR: Access is denied."))
    assert h.try_heal(0.0, "test")
    h.last_thread.join(timeout=5)
    assert any("setup_selfheal.bat" in m for m in msgs)


def test_selfhealer_budget_is_two_per_hour_and_refills(monkeypatch):
    monkeypatch.setattr(powerguard.sys, "platform", "win32")
    h = powerguard.SelfHealer(_cfg(), runner=_runner_factory([]), notify=lambda m: None)
    assert h.try_heal(0.0, "a")
    assert h.try_heal(10.0, "b")
    assert not h.try_heal(20.0, "c")          # budget spent
    assert h.try_heal(3700.0, "d")            # the first attempt aged out of the hour window


def test_selfhealer_off_windows_is_unavailable(monkeypatch):
    monkeypatch.setattr(powerguard.sys, "platform", "linux")
    h = powerguard.SelfHealer(_cfg(), runner=_runner_factory([]), notify=lambda m: None)
    assert not h.available()
