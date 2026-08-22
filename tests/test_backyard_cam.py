"""
Tests for backyard_cam: the naming-helper sweep, the ignore zones, the motion gate's retained
mask, the detector census, and the reference-image veto's SHADOW-MODE wiring into the capture
loop.

A rig that dies WITHOUT running its finally block (taskkill /F, a crash, the OOM kills)
leaves its classify.py --watch helper running forever; the next launch must kill exactly
those helpers -- and nothing else. Everything here drives the PURE decision function
(_stale_naming_pids) and the parsers with synthetic process tables: no PowerShell, no
taskkill, no real processes. The one sweep-level test stubs the process query and
subprocess.run to check the taskkill wiring and the never-kill-myself guard.

The synthetic rows mirror the venv-redirector reality: every spawned python shows as TWO
rows (0-CPU shim + real interpreter) with IDENTICAL command lines, and a stale helper must
have both rows reaped.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import backyard_cam
import config
import db
import refimg
from detector import Detection

# Fake install paths: they only ever get spliced into the synthetic command-line strings below,
# so nothing here is launched and the real interpreter's location is irrelevant.
PY = r"C:\rig\backyard\.venv\Scripts\python.exe"
ROOT = r"C:\rig\backyard"


def _rig(pid: int, source: int = 0) -> tuple[int, str]:
    """A live rig's process row (both the shim and real rows look exactly like this)."""
    return (pid, f"{PY} {ROOT}\\backyard_cam.py --source {source}")


def _helper(shim_pid: int, real_pid: int, rig_pid: int) -> list[tuple[int, str]]:
    """A naming helper's TWO rows (venv shim + real interpreter, identical command lines),
    tagged with the pid of the rig that spawned it."""
    cmd = (f"{PY} {ROOT}\\classify.py --watch --device cpu --interval 20.0 "
           f"--tag backyard-naming-{rig_pid}")
    return [(shim_pid, cmd), (real_pid, cmd)]


# ---- _stale_naming_pids: the kill/keep decision -------------------------------------

def test_helper_of_dead_rig_reaped_both_rows():
    # Rig 999 is gone (no python row at all: exited, or pid reused by non-python).
    rows = [_rig(500)] + _helper(11, 12, 999)
    assert sorted(backyard_cam._stale_naming_pids(rows, my_pid=500)) == [11, 12]


def test_helper_of_live_rig_survives():
    rows = [_rig(4242)] + _helper(11, 12, 4242)
    assert backyard_cam._stale_naming_pids(rows, my_pid=500) == []


def test_two_rigs_side_by_side_only_dead_ones_helper_reaped():
    # Rig 100 (source 0) is alive; rig 200 (source 1) died hard. Each left a helper.
    rows = [_rig(100, source=0)] + _helper(11, 12, 100) + _helper(21, 22, 200)
    assert sorted(backyard_cam._stale_naming_pids(rows, my_pid=300)) == [21, 22]


def test_helper_wearing_my_own_pid_is_stale():
    # We sweep BEFORE spawning our helper, so a helper already tagged with OUR pid can only
    # be a leftover from a dead rig whose pid we inherited -- even though the pid (us) is a
    # live python running backyard_cam.py.
    rows = [_rig(4242)] + _helper(11, 12, 4242)
    assert sorted(backyard_cam._stale_naming_pids(rows, my_pid=4242)) == [11, 12]


def test_rig_pid_reused_by_other_python_is_stale():
    # Pid 999 lives again, but as a clipmotion batch -- readable command line, wrong script.
    rows = [(999, f"{PY} {ROOT}\\clipmotion.py --days 7")] + _helper(11, 12, 999)
    assert sorted(backyard_cam._stale_naming_pids(rows, my_pid=500)) == [11, 12]


def test_unreadable_rig_cmdline_keeps_helper():
    # Pid 999 is a live python whose command line we could NOT read (CIM gave $null).
    # Uncertain -> leave the helper alone; a stray helper beats killing a live rig's.
    rows = [(999, "")] + _helper(11, 12, 999)
    assert backyard_cam._stale_naming_pids(rows, my_pid=500) == []


def test_tag_lookalike_without_classify_is_not_a_helper():
    # Some python whose argv merely CONTAINS the tag text (a grep, a REPL experiment) must
    # never be treated as a helper, even though its "rig" 999 is long dead.
    rows = [(77, f"{PY} -c \"print('backyard-naming-999')\"")]
    assert backyard_cam._stale_naming_pids(rows, my_pid=500) == []


# ---- parsers / plumbing -------------------------------------------------------------

def test_parse_process_rows_first_tab_splits_and_junk_dropped():
    text = ("ProcessId CommandLine\r\n"           # header-ish junk: no leading pid
            "\r\n"                                 # blank
            "123\tC:\\py\\python.exe a.py\r\n"
            "456\tC:\\py\\python.exe b.py --arg \tweird\ttabs\r\n"   # tabs INSIDE cmdline
            "789\t\r\n")                           # live python, unreadable ($null) cmdline
    assert backyard_cam._parse_process_rows(text) == [
        (123, "C:\\py\\python.exe a.py"),
        (456, "C:\\py\\python.exe b.py --arg \tweird\ttabs"),
        (789, ""),
    ]


def test_naming_pids_filters_rows_by_tag(monkeypatch):
    rows = _helper(11, 12, 4242) + _helper(21, 22, 999) + [_rig(4242)]
    monkeypatch.setattr(backyard_cam, "_python_process_rows", lambda timeout=15: rows)
    assert backyard_cam._naming_pids("backyard-naming-4242") == ["11", "12"]


def test_sweep_taskkills_stale_pids_and_never_itself(monkeypatch):
    # One genuinely stale helper (rig 999 gone) plus a pathological row claiming OUR pid runs
    # a helper: the sweep must taskkill the former's two rows and refuse to kill os.getpid().
    me = os.getpid()
    rows = _helper(11, 12, 999) + [(me, _helper(0, 0, 999)[0][1])]
    calls = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(backyard_cam, "_python_process_rows", lambda timeout=15: rows)
    monkeypatch.setattr(backyard_cam.subprocess, "run",
                        lambda args, **kw: calls.append(args))
    backyard_cam._sweep_stale_naming()
    assert calls == [["taskkill", "/F", "/PID", "11", "/PID", "12"]]


def test_sweep_quiet_when_nothing_is_stale(monkeypatch):
    rows = [_rig(4242)] + _helper(11, 12, 4242)
    calls = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(backyard_cam, "_python_process_rows", lambda timeout=15: rows)
    monkeypatch.setattr(backyard_cam.subprocess, "run",
                        lambda args, **kw: calls.append(args))
    backyard_cam._sweep_stale_naming()
    assert calls == []


def test_sweep_swallows_taskkill_failure(monkeypatch):
    def boom(args, **kw):
        raise OSError("taskkill missing")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(backyard_cam, "_python_process_rows",
                        lambda timeout=15: _helper(11, 12, 999))
    monkeypatch.setattr(backyard_cam.subprocess, "run", boom)
    backyard_cam._sweep_stale_naming()   # must not raise: best-effort, silent on failure


# ---- Ignore zones: the static-false-fire filter -------------------------------------
# Real numbers from the 2026-07-20 dusk: the retaining-wall opening fired "animal" at
# (1141,608)-(1220,687) on every detector run, while the actual raccoon walked past with
# far bigger, drifting boxes. The zone below is that box padded ~12 px (see config_local).

ZONE = (1127, 595, 1234, 701)


class _Det:
    def __init__(self, bbox):
        self.bbox = bbox


def test_box_iou_identical_and_disjoint():
    assert backyard_cam.box_iou(ZONE, ZONE) == 1.0
    assert backyard_cam.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert backyard_cam.box_iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0   # degenerate box


def test_wall_opening_false_fire_is_dropped():
    hole = _Det((1141.0, 608.0, 1220.0, 687.0))        # the measured false-fire box
    kept, dropped = backyard_cam.drop_ignored([hole], [ZONE], 0.45)
    assert kept == [] and dropped == [hole]


def test_animal_walking_through_the_zone_is_kept():
    # A raccoon-sized box OVERLAPPING the zone: IoU with the zone stays tiny -> kept.
    raccoon = _Det((950.0, 500.0, 1350.0, 800.0))
    kept, dropped = backyard_cam.drop_ignored([raccoon], [ZONE], 0.45)
    assert kept == [raccoon] and dropped == []


def test_mixed_detections_split_correctly():
    hole = _Det((1140.0, 609.0, 1222.0, 689.0))
    bird = _Det((300.0, 300.0, 420.0, 400.0))          # elsewhere in frame entirely
    kept, dropped = backyard_cam.drop_ignored([hole, bird], [ZONE], 0.45)
    assert kept == [bird] and dropped == [hole]


def test_no_zones_is_a_no_op():
    d = _Det((1141.0, 608.0, 1220.0, 687.0))
    kept, dropped = backyard_cam.drop_ignored([d], [], 0.45)
    assert kept == [d] and dropped == []


# ---- Synthetic scene ----------------------------------------------------------------
# One deterministic "yard": coloured (so refimg calls it 'day', not 'ir'), textured (so the
# sobel/SSIM metrics and the edge fingerprint have something to hold on to), and carrying a
# static pale slab standing in for the furniture that fires the detector every night.

FURNITURE = (200.0, 200.0, 360.0, 360.0)      # full-frame px; the tipped-watering-can stand-in
ELSEWHERE = (800.0, 400.0, 940.0, 540.0)      # a box at a spot that has never fired before


def _yard(w: int = 1280, h: int = 720, seed: int = 11):
    """A static, deterministic colour frame. Base B/G/R 120/165/205 -> chroma 85 (not IR) and a
    grey median ~172 (>= 90, so 'day'); the noise is added equally to all three channels so it
    textures the scene without touching that classification."""
    rng = np.random.default_rng(seed)
    frame = np.zeros((h, w, 3), np.int16)
    frame[:, :, 0], frame[:, :, 1], frame[:, :, 2] = 120, 165, 205
    frame += rng.integers(0, 40, size=(h, w, 1), dtype=np.int16)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    x1, y1, x2, y2 = (int(v) for v in FURNITURE)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (215, 225, 235), -1)
    cv2.rectangle(frame, (620, 120), (700, 620), (60, 70, 90), -1)      # a post, for edges
    return frame


# ---- MotionGate: the retained mask ---------------------------------------------------
# The gate now KEEPS its cleaned foreground mask (refimg's coverage channel). The area it
# returns must be exactly what it always was: the largest blob, in FULL-FRAME px^2.

def _gate_cfg():
    cfg = config.Config()
    cfg.motion_gate_width = 640          # 1280-wide frames -> a 2x downscale, area_scale 4.0
    return cfg


def _moving_blob_frames(n: int = 8):
    base = _yard()
    out = []
    for i in range(n):
        f = base.copy()
        cv2.rectangle(f, (400 + i * 25, 300), (460 + i * 25, 360), (250, 250, 250), -1)
        out.append(f)
    return out


def test_motion_gate_exposes_the_mask_and_its_area_still_matches_it():
    gate = backyard_cam.MotionGate(_gate_cfg())
    assert gate.mask is None                       # nothing observed yet
    seen_a_blob = False
    for frame in _moving_blob_frames():
        area = gate.update(frame)
        assert gate.mask is not None
        assert gate.mask.shape == (360, 640)       # the gate's own downscaled resolution
        assert set(np.unique(gate.mask)) <= {0, 255}
        contours, _ = cv2.findContours(gate.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 1280 -> 640 is a 0.5 scale, so blob areas scale back by 1/0.5^2 = 4.
        expected = max((cv2.contourArea(c) for c in contours), default=0.0) * 4.0
        assert area == pytest.approx(expected)
        seen_a_blob = seen_a_blob or area > 0
    assert seen_a_blob                              # the fixture really does move something


def test_motion_gate_areas_are_unchanged_and_reproducible():
    frames = _moving_blob_frames()
    a = [backyard_cam.MotionGate(_gate_cfg()).update(f) for f in frames]   # fresh gate each frame
    g1, g2 = backyard_cam.MotionGate(_gate_cfg()), backyard_cam.MotionGate(_gate_cfg())
    assert [g1.update(f) for f in frames] == [g2.update(f) for f in frames]
    assert a[0] == 0.0 or a[0] > 0.0               # (just pinning that the first frame is defined)


def test_motion_gate_on_a_still_scene_reports_no_motion_and_an_empty_mask():
    gate = backyard_cam.MotionGate(_gate_cfg())
    frame = _yard()
    areas = [gate.update(frame.copy()) for _ in range(20)]
    assert areas[-1] == 0.0
    assert int(gate.mask.max()) == 0


# ---- DetectorCensus: measurement, on whether or not the veto is ----------------------

def test_census_counts_runs_and_empties():
    c = backyard_cam.DetectorCensus()
    c.note(0.0, 2)
    c.note(1.0, 0)
    c.note(2.0, 0)
    assert (c.runs, c.empty) == (3, 2)


def test_census_longest_empty_run_is_the_longest_unbroken_stretch():
    c = backyard_cam.DetectorCensus()
    for t in (0.0, 5.0, 9.0):
        c.note(t, 0)
    assert c.longest_empty_s == pytest.approx(9.0)
    c.note(10.0, 1)                     # a detection breaks the run
    c.note(20.0, 0)
    c.note(23.0, 0)
    assert c.longest_empty_s == pytest.approx(9.0)     # the newer 3 s stretch doesn't win


def test_census_rolls_once_per_period_and_resets():
    c = backyard_cam.DetectorCensus(period_s=100.0)
    assert c.roll(0.0) is None          # the first call only starts the clock
    c.note(1.0, 0)
    c.note(2.0, 0)
    assert c.roll(50.0) is None
    line = c.roll(120.0)
    assert line is not None
    assert "2 run(s), 2 empty (100%)" in line and "longest empty run 1s" in line
    assert (c.runs, c.empty, c.longest_empty_s) == (0, 0, 0.0)
    assert c.roll(150.0) is None        # the new period has only just started


# ---- VetoCensus: the abstentions, which are the whole shadow-mode question ------------
# db.record_suppression writes a row only for SUPPRESS, so an inert veto and a perfectly precise
# one look identical in the database. That is the state the 2026-08-09 review found, and digging
# out the reason meant replaying the whole shadow week against the banked reference PNGs.

def _decision(reason, cover=None, decision=refimg.ABSTAIN):
    return refimg.Decision(decision=decision, reason=reason, cover=cover)


def test_veto_census_counts_every_decision_not_only_the_suppressions():
    c = backyard_cam.VetoCensus()
    c.note(_decision("reference_has_no_pixels_here", cover=0.0))
    c.note(_decision("reference_has_no_pixels_here", cover=0.12))
    c.note(_decision("pixels_differ_from_empty", cover=0.95, decision=refimg.KEEP))
    c.note(None)                                        # no prepared frame: counted, not dropped
    assert c.reasons == {"reference_has_no_pixels_here": 2,
                         "pixels_differ_from_empty": 1, "not_evaluated": 1}


def test_veto_census_reports_how_close_the_abstentions_came():
    """"How many did it flag" is the easy half. "How close did the rest come" is the half that
    says whether the bar is wrong or the accumulation under it is."""
    c = backyard_cam.VetoCensus(period_s=100.0)
    assert c.roll(0.0) is None
    for cover in (0.0, 0.0, 0.0, 0.95):
        c.note(_decision("reference_has_no_pixels_here", cover=cover))
    line = c.roll(200.0)
    assert "4 box(es) judged" in line
    assert "cover p50" in line and "1 at or above the bar" in line
    assert c.reasons == {} and c.covers == []


def test_veto_census_prints_an_hour_in_which_nothing_was_judged():
    c = backyard_cam.VetoCensus(period_s=100.0)
    c.roll(0.0)
    line = c.roll(200.0)
    assert line is not None and "0 box(es) judged" in line
    assert "no box reached the coverage gate" in line


# ---- HUD: one extra token, and nothing else -----------------------------------------

def test_hud_without_a_ref_token_is_pixel_identical():
    a, b = np.zeros((200, 640, 3), np.uint8), np.zeros((200, 640, 3), np.uint8)
    kw = dict(fps=21.6, motion_area=1234, motion=True, saved=7, source="glass door",
              model="MDV6-yolov10-c")
    backyard_cam.draw_hud(a, **kw)
    backyard_cam.draw_hud(b, ref_status=None, **kw)
    assert np.array_equal(a, b)


def test_hud_with_a_ref_token_draws_more_pixels():
    a, b = np.zeros((200, 640, 3), np.uint8), np.zeros((200, 640, 3), np.uint8)
    kw = dict(fps=21.6, motion_area=1234, motion=True, saved=7, source="glass door",
              model="MDV6-yolov10-c")
    backyard_cam.draw_hud(a, **kw)
    backyard_cam.draw_hud(b, ref_status="ref night 7m", **kw)
    assert b.sum() > a.sum()


def test_short_age_reads_at_a_glance():
    assert backyard_cam._short_age(42) == "42s"
    assert backyard_cam._short_age(7 * 60 + 30) == "7m"
    assert backyard_cam._short_age(3.14 * 3600) == "3.1h"
    assert backyard_cam._short_age(-5) == "0s"


# ---- open_capture: the all-auto baseline ---------------------------------------------
# UVC webcams remember manual focus/WB/exposure IN HARDWARE across sessions, so a manual
# nudge in one session (a dashboard slider, a tune.py experiment) used to leave every later
# session silently starting in manual mode -- on this cam, the broken red-starved manual-WB
# state the watchdog exists for, and unfocusable glass at night. open_capture now asserts
# auto focus / WB / exposure on every open; deliberate locks (cfg.exposure, camera_controls)
# are applied afterwards, so they still win.

class _RecordingCap:
    """A VideoCapture stand-in that opens, records every set() in order, and 'reads' frames."""

    def __init__(self, *args):
        self.sets = []

    def isOpened(self):
        return True

    def set(self, prop, val):
        self.sets.append((prop, float(val)))
        return True

    def get(self, prop):
        return 0.0

    def read(self):
        self.sets.append(("READ", None))   # marks stream start in the set/read timeline
        return True, None            # warmup reads; open_capture ignores the frames

    def release(self):
        pass

    def writes(self, prop):
        return [v for p, v in self.sets if p == prop]

    def writes_after_first_read(self, prop):
        i = self.sets.index(("READ", None))
        return [v for p, v in self.sets[i:] if p == prop]


def _open_recorded(monkeypatch, **over):
    cap = _RecordingCap()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: cap)
    cfg = config.Config()
    for key, val in over.items():
        setattr(cfg, key, val)
    assert backyard_cam.open_capture(cfg.camera_specs()[0], cfg) is cap
    return cap


def test_open_asserts_the_all_auto_baseline(monkeypatch):
    cap = _open_recorded(monkeypatch)
    assert cap.writes(cv2.CAP_PROP_AUTOFOCUS) == [1.0, 1.0]     # pre-stream AND mid-stream
    assert cap.writes(cv2.CAP_PROP_AUTO_WB) == [1.0, 1.0]
    assert cap.writes(cv2.CAP_PROP_AUTO_EXPOSURE) == [0.75, 0.75]


def test_the_auto_baseline_is_reasserted_after_the_stream_starts(monkeypatch):
    # This driver DROPS auto-mode sets issued before the capture graph runs (2026-08-07: the
    # panel's mid-stream toggle visibly re-focused the image minutes after open "asserted"
    # auto). The assert that counts is the one after the first read.
    cap = _open_recorded(monkeypatch)
    assert cap.writes_after_first_read(cv2.CAP_PROP_AUTOFOCUS) == [1.0]
    assert cap.writes_after_first_read(cv2.CAP_PROP_AUTO_WB) == [1.0]
    assert cap.writes_after_first_read(cv2.CAP_PROP_AUTO_EXPOSURE) == [0.75]


def test_a_deliberate_config_lock_beats_the_baseline(monkeypatch):
    cap = _open_recorded(monkeypatch, exposure=-6.0,
                         camera_controls={"AUTOFOCUS": 0, "FOCUS": 200})
    af = cap.writes(cv2.CAP_PROP_AUTOFOCUS)
    assert af[0] == 1.0 and af[-1] == 0.0                     # baseline first; the lock wins
    assert cap.writes_after_first_read(cv2.CAP_PROP_AUTOFOCUS) == [0.0]   # mid-stream too
    assert cap.writes(cv2.CAP_PROP_AUTO_EXPOSURE) == [0.25, 0.25]   # manual exposure, never auto
    assert cap.writes(cv2.CAP_PROP_EXPOSURE)[-1] == -6.0
    assert cap.writes(cv2.CAP_PROP_AUTO_WB) == [1.0, 1.0]     # WB keeps the auto baseline


# The dashboard's auto checkboxes show CommandedAutoState, not cap.get(): this driver answers
# -1.0 / 0.0 for the auto props whatever the real mode is (measured 2026-08-07 -- the panel
# showed manual while the frame's R/G ratio proved auto-WB was live). The rig is the only
# writer of these modes, so the last write is the state.

def test_commanded_auto_state_tracks_every_writer():
    cfg = config.Config()
    a = backyard_cam.CommandedAutoState(cfg.camera_specs()[0], cfg)
    assert a.state == {"autofocus": 1.0, "auto_wb": 1.0, "auto_exposure": 0.75}
    a.note({"AUTO_WB": 0, "WB_TEMPERATURE": 4600})   # dashboard WB slider -> manual
    assert a.state["auto_wb"] == 0.0
    a.note({"AUTO_WB": 1})                           # WB watchdog recovery
    assert a.state["auto_wb"] == 1.0
    a.note({"exposure": -7.0})                       # a profile locks exposure ...
    assert a.state["auto_exposure"] == 0.25
    a.note({"exposure": None})                       # ... and the other flips it back to auto
    assert a.state["auto_exposure"] == 0.75
    a.note({"AUTOFOCUS": 0, "FOCUS": 300})           # dashboard focus slider -> manual
    assert a.state["autofocus"] == 0.0
    a.note({"BACKLIGHT": 0})                         # unrelated settings change nothing
    assert a.state == {"autofocus": 0.0, "auto_wb": 1.0, "auto_exposure": 0.75}


def test_commanded_auto_state_seeds_from_the_config_locks():
    cfg = config.Config()
    cfg.exposure = -6.0
    cfg.camera_controls = {"AUTOFOCUS": 0, "FOCUS": 200}
    a = backyard_cam.CommandedAutoState(cfg.camera_specs()[0], cfg)
    assert a.state == {"autofocus": 0.0, "auto_wb": 1.0, "auto_exposure": 0.25}


# ---- The capture loop, driven end to end on fake frames ------------------------------
# A fake capture + a scripted detector run the REAL _run_camera: motion gate, detector gating,
# crop + DB write, the census and (when enabled) the shadow veto. Nothing here opens a camera,
# a model or the real database.

class _FakeCap:
    """A VideoCapture stand-in that plays a frame list and then ends the run."""

    def __init__(self, frames, stop_event):
        self._frames = list(frames)
        self._stop = stop_event
        self.released = False

    def read(self):
        if not self._frames:
            self._stop.set()            # the loop's own exit condition; no reconnect is attempted
            return False, None
        return True, self._frames.pop(0).copy()

    def isOpened(self):
        return True

    def release(self):
        self.released = True

    def set(self, *a, **k):
        return True

    def get(self, *a, **k):
        return 0.0


class _ScriptedDetector:
    """Returns a scripted verdict per CALL -- the loop only calls it on motion frames."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def detect(self, frame):
        i, self.calls = self.calls, self.calls + 1
        return list(self.script[i]) if i < len(self.script) else []


def _animal(box, conf: float = 0.9) -> Detection:
    return Detection(class_id=0, class_name="animal", confidence=conf,
                     bbox=tuple(float(v) for v in box))


def _loop_cfg(root, **over):
    """A hermetic config: the SHIPPED defaults (never Matt's config_local), every path inside
    the test's tmp dir, and the knobs that would otherwise reach hardware turned off."""
    cfg = config.Config()
    root.mkdir(parents=True, exist_ok=True)
    cfg.db_path = root / "test.db"
    cfg.crops_dir = root / "crops"
    cfg.frames_dir = root / "frames"
    cfg.clips_dir = root / "clips"
    cfg.refimg_store_dir = root / "refimg_store"
    cfg.source = "test_cam"
    cfg.show_preview = False
    cfg.serve = False
    cfg.record_clips = False
    cfg.save_full_frame = False
    cfg.classify_live = False
    cfg.use_time_of_day_profiles = False
    cfg.wb_auto_recover = False
    cfg.wedge_guard = False
    cfg.ignore_zones = {}
    cfg.motion_gate_width = 640
    cfg.motion_min_area = -1.0            # a still synthetic scene still wakes the detector
    cfg.detector_min_interval_s = 0.0     # ... on every frame, instead of once a second
    for key, val in over.items():
        setattr(cfg, key, val)
    return cfg


# 10 empty verdicts (the reference certifies), then the same furniture box twice, then a box
# somewhere the detector has never fired before.
_SCRIPT = ([[] for _ in range(10)]
           + [[_animal(FURNITURE)], [_animal(FURNITURE)], [_animal(ELSEWHERE)]])


def _drive(cfg, monkeypatch, script=_SCRIPT):
    frames = [_yard()] * (backyard_cam.MOTION_WARMUP_FRAMES + len(script))
    stop = threading.Event()
    cap = _FakeCap(frames, stop)
    monkeypatch.setattr(backyard_cam, "open_capture", lambda spec, c: cap)
    detector = _ScriptedDetector(script)
    results: dict = {}
    backyard_cam._run_camera(cfg.camera_specs()[0], cfg, detector, threading.Lock(),
                             {}, {}, {}, threading.Lock(), stop, results)
    assert cap.released
    return results, detector


def _rows(cfg):
    conn = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM detections ORDER BY id")]
    finally:
        conn.close()


def _saved_shape(rows):
    """Everything about a saved row that shadow mode must not change (timestamps and the
    timestamped crop filename are wall-clock and therefore differ between runs)."""
    return [(r["source"], r["detection_class"], r["confidence"], r["bbox_x1"], r["bbox_y1"],
             r["bbox_x2"], r["bbox_y2"], r["frame_w"], r["frame_h"], r["species"],
             r["individual_id"], r["crop_quality"]) for r in rows]


def test_disabled_veto_leaves_the_save_path_exactly_as_it_was(tmp_path, monkeypatch):
    cfg = _loop_cfg(tmp_path / "off")            # refimg_enabled defaults to False
    assert cfg.refimg_enabled is False
    results, detector = _drive(cfg, monkeypatch)

    rows = _rows(cfg)
    assert len(rows) == 3 and results["test_cam"]["saved"] == 3
    for r in rows:
        assert r["suppressed_at"] is None and r["suppressed_by"] is None
        assert r["suppress_ref_id"] is None and r["suppress_detail"] is None
        assert db.crop_abspath(r["crop_path"]).exists()
    assert results["test_cam"]["suppressed"] == 0
    # Nothing of the veto ran: no reference images, no view epochs, not even its store directory.
    conn = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM reference_images").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM view_epochs").fetchone()[0] == 0
    finally:
        conn.close()
    assert not cfg.refimg_store_dir.exists()


def test_shadow_mode_saves_exactly_the_same_rows_as_the_disabled_rig(tmp_path, monkeypatch):
    off = _loop_cfg(tmp_path / "off")
    on = _loop_cfg(tmp_path / "on", refimg_enabled=True, refimg_certify_hold_s=0.0,
                   refimg_recurrence_min_events=2, refimg_recurrence_event_gap_s=0.0,
                   refimg_recurrence_min_days=1)
    _drive(off, monkeypatch)
    _drive(on, monkeypatch)
    # Identical detections, box for box: the veto only ever ADDS metadata to a saved row.
    assert _saved_shape(_rows(off)) == _saved_shape(_rows(on))


def test_shadow_veto_flags_a_recurring_furniture_box_and_keeps_everything(tmp_path, monkeypatch):
    cfg = _loop_cfg(tmp_path / "on", refimg_enabled=True, refimg_certify_hold_s=0.0,
                    refimg_recurrence_min_events=2, refimg_recurrence_event_gap_s=0.0,
                    refimg_recurrence_min_days=1)
    results, _ = _drive(cfg, monkeypatch)
    rows = _rows(cfg)
    assert len(rows) == 3
    first, second, elsewhere = rows

    # The FIRST firing at the furniture spot matches the empty reference on every pixel metric
    # but has no history yet -- the recurrence gate is what keeps it.
    assert first["suppressed_at"] is None
    assert json.loads(first["suppress_detail"] or "null") is None
    # The SECOND firing at the same spot is the suppression.
    assert second["suppressed_at"] is not None
    assert second["suppressed_by"] == "refimg_veto"
    detail = json.loads(second["suppress_detail"])
    assert detail["decision"] == "SUPPRESS"
    assert detail["reason"] == "matches_empty_reference_at_a_recurring_spot"
    assert detail["provenance"] == refimg.PROVENANCE_MOTION_MASKED
    assert detail["recurrence"]["events"] >= 2
    assert set(detail["thresholds"]) >= {"lum", "dssim", "sobel"}
    # A first-time box somewhere else is NOT suppressed, even though its pixels match too.
    assert elsewhere["suppressed_at"] is None

    # SHADOW MODE: the flagged row and its crop are still there, unchanged.
    for r in rows:
        assert db.crop_abspath(r["crop_path"]).exists()
        assert r["detection_class"] == "animal" and r["confidence"] == pytest.approx(0.9)
    assert results["test_cam"]["saved"] == 3
    assert results["test_cam"]["suppressed"] == 1

    # The suppression points at a reference image that is on disk and replayable.
    conn = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
    try:
        ref = db.reference_image(conn, second["suppress_ref_id"])
    finally:
        conn.close()
    assert ref is not None and ref["source"] == "test_cam" and ref["illumination"] == "day"
    assert ref["provenance"] == refimg.PROVENANCE_MOTION_MASKED
    assert Path(ref["image_path"]).exists()
    # The recurrence ledger survives the run, so a restart does not forget the spot.
    assert (cfg.refimg_store_dir / "test_cam" / "recurrence.json").exists()


def test_shadow_is_none_when_the_veto_is_disabled(tmp_path, conn):
    cfg = _loop_cfg(tmp_path / "off")
    assert backyard_cam.RefimgShadow.create(cfg, "test_cam", conn) is None


def test_hud_token_goes_from_nothing_to_a_certified_reference(tmp_path, conn):
    cfg = _loop_cfg(tmp_path / "hud", refimg_enabled=True, refimg_certify_hold_s=0.0)
    shadow = backyard_cam.RefimgShadow.create(cfg, "test_cam", conn)
    assert shadow is not None
    assert shadow.hud() == "ref --"                    # nothing certified yet: the veto abstains
    frame, mask = _yard(), np.zeros((360, 640), np.uint8)
    start = time.time()
    for i in range(6):                                  # six empty detector verdicts, motion quiet
        shadow.observe(conn, frame, [], mask, start + i * 0.1)
    assert shadow.hud() == "ref day 0s"                 # a coloured, bright scene reads as 'day'
    assert shadow.ref is not None and shadow.ref.id is not None


def test_census_line_is_logged_even_with_the_veto_off(tmp_path, monkeypatch, capsys):
    cfg = _loop_cfg(tmp_path / "census")
    # A zero-length reporting period so the hourly line renders inside a 28-frame test.
    real = backyard_cam.DetectorCensus
    monkeypatch.setattr(backyard_cam, "DetectorCensus", lambda: real(period_s=0.0))
    _drive(cfg, monkeypatch)
    out = capsys.readouterr().out
    assert "detector census:" in out
    assert "run(s)" in out and "longest empty run" in out


def test_veto_census_line_reaches_the_log_from_the_real_loop(tmp_path, monkeypatch, capsys):
    """The unit tests above prove VetoCensus counts; this proves the capture loop actually CALLS
    it. That distinction is the whole reason this class exists -- refimg's shadow week wrote only
    its suppressions, so "it flagged nothing" and "it is perfectly precise" were the same log, and
    telling them apart cost a full replay of the week off recorded clips."""
    cfg = _loop_cfg(tmp_path / "vetocensus", refimg_enabled=True, refimg_certify_hold_s=0.0)
    real = backyard_cam.VetoCensus
    monkeypatch.setattr(backyard_cam, "VetoCensus", lambda: real(period_s=0.0))
    _drive(cfg, monkeypatch)
    out = capsys.readouterr().out
    assert "veto census:" in out
    assert "box(es) judged" in out


# ---- the guard that keeps a camera password out of the log ---------------------------
#
# The masking itself is cameras.safe_src and is tested in tests/test_cameras.py -- it moved out of
# this module when web.py and tools/camprobe.py needed it too, since importing it from here would
# close the cycle backyard_cam -> web -> camprobe -> backyard_cam. What stays here is the property
# that only this file can have: that every human-facing print of a camera src actually goes
# through it.


def test_no_print_site_still_formats_a_raw_src():
    """The guard that outlives this change: masking helps only if EVERY human-facing print goes
    through it, so a new `{spec.src!r}` added later should fail here rather than quietly write a
    password into the log for a week before anyone notices."""
    source = Path(backyard_cam.__file__).read_text(encoding="utf-8")
    assert "src!r" not in source
