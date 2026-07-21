"""
Tests for the startup sweep that reaps orphaned naming helpers (backyard_cam).

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

import os
import sys

import backyard_cam

PY = r"C:\Users\you\projects\backyard\.venv\Scripts\python.exe"
ROOT = r"C:\Users\you\projects\backyard"


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
