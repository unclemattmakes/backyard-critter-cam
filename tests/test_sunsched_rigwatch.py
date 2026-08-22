"""The two scheduling guards.

sunsched exists because a hardcoded clock time drifts out of the glare window as the season turns;
rigwatch exists because nothing brought the rig back after the 2026-08-21 bugcheck. Both are
allowed to do nothing, and neither may ever do something surprising -- restarting a rig a human
just stopped, or parking the batch on top of the raccoon peak."""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone

import pytest

import config
import rigwatch
import sunsched


@pytest.fixture(autouse=True)
def _fixed_location(monkeypatch):
    """Pin the observer to a fixed, PUBLIC location.

    Two reasons. Without it the suite reads config_local.py, which is gitignored and therefore
    ABSENT on CI -- latitude/longitude default to None there and sunsched raises SystemExit.
    And the operator's real coordinates are precisely what config_local.py exists to keep OUT
    of this repo, so these are the placeholder ones from config_local.example.py. What is under
    test is a SHAPE -- an offset from sunset, clamped, tracking the season -- and that shape
    holds anywhere; every assertion below passes at either location."""
    monkeypatch.setattr(config.CONFIG, "latitude", 40.7128)
    monkeypatch.setattr(config.CONFIG, "longitude", -74.0060)


# The test location's own zone. Passed explicitly to every target_time() call: CI runs UTC, where an
# August sunset there lands after midnight and the "is it 17:xx?" assertions would be nonsense.
DST = timezone(timedelta(hours=-4))
STD = timezone(timedelta(hours=-5))


def _target(d: date):
    return sunsched.target_time(d, tz=(DST if 3 <= d.month <= 10 else STD))


# --------------------------------------------------------------------------- sunsched
def test_target_is_the_configured_offset_before_sunset():
    d = date(2026, 8, 21)
    t, s = _target(d)
    assert (s["sunset"] - t) == timedelta(hours=sunsched.SUNSET_OFFSET_H)


def test_target_lands_in_the_measured_glare_window_in_late_august():
    # The window the module aims at: measured on the real rig, glare peaked 17:00-18:00 and
    # crop_quality bottomed at 18:00. sunset-2.5h lands in that hour at the test location too.
    t, _ = _target(date(2026, 8, 21))
    assert 17 <= t.hour < 18, f"expected the late-afternoon glare window, got {t:%H:%M}"


def test_the_time_tracks_the_season_instead_of_standing_still():
    """The whole reason this module exists: 17:39 in August is not the right hour in October."""
    aug, _ = _target(date(2026, 8, 21))
    oct_, _ = _target(date(2026, 10, 21))
    assert oct_.hour * 60 + oct_.minute < aug.hour * 60 + aug.minute - 45


def test_it_never_schedules_outside_the_clamps_across_a_whole_year():
    lo = datetime.strptime(sunsched.EARLIEST, "%H:%M").time()
    hi = datetime.strptime(sunsched.LATEST, "%H:%M").time()
    d = date(2026, 1, 1)
    while d.year == 2026:
        t, _ = _target(d)
        assert lo <= t.time() <= hi, f"{d} -> {t:%H:%M} escaped the clamps"
        d += timedelta(days=7)


def test_never_lands_on_the_raccoon_peak_across_a_whole_year():
    """Raccoons run 21:00-05:00. A batch that starts in there is the failure this guards."""
    d = date(2026, 1, 1)
    while d.year == 2026:
        t, _ = _target(d)
        assert not (t.hour >= 21 or t.hour < 5), f"{d} -> {t:%H:%M} is inside the raccoon peak"
        d += timedelta(days=7)


def test_arm_reports_but_changes_nothing_on_a_dry_run(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(sunsched.subprocess, "run", lambda *a, **k: called.append(a))
    assert sunsched.arm(date(2026, 8, 21), dry_run=True) == 0
    assert called == []
    assert "batch at" in capsys.readouterr().out


def test_a_failing_schtasks_is_a_warning_not_a_failure(monkeypatch, capsys):
    """Losing the re-arm costs ~1.5 min/day of drift. Failing the whole batch costs a night."""
    class R:
        returncode, stdout, stderr = 1, "", "ERROR: task not found"
    monkeypatch.setattr(sunsched.subprocess, "run", lambda *a, **k: R())
    assert sunsched.arm(date(2026, 8, 21)) == 0
    assert "WARNING" in capsys.readouterr().err


# --------------------------------------------------------------------------- rigwatch
@pytest.fixture(autouse=True)
def _isolated_rigwatch(tmp_path, monkeypatch):
    monkeypatch.setattr(rigwatch, "PAUSE_MARKER", tmp_path / ".rig_pause")
    monkeypatch.setattr(rigwatch, "STATE_FILE", tmp_path / ".rigwatch_state.json")
    monkeypatch.setattr(rigwatch, "LOG_FILE", tmp_path / "logs" / "rigwatch.log")
    yield


def test_no_marker_means_not_paused():
    assert rigwatch.paused() is False


def test_a_marker_written_since_boot_is_a_deliberate_stop(monkeypatch):
    monkeypatch.setattr(rigwatch, "boot_time", lambda: time.time() - 3600)
    rigwatch.PAUSE_MARKER.write_text("stopped from the video window")
    assert rigwatch.paused() is True


def test_a_marker_older_than_the_boot_is_stale_so_a_reboot_starts_the_rig(monkeypatch):
    """After a reboot the rig comes back even if it was stopped by hand beforehand -- the entire
    point is that it does not depend on Matt being there."""
    rigwatch.PAUSE_MARKER.write_text("stopped from the video window")
    monkeypatch.setattr(rigwatch, "boot_time", lambda: time.time() + 60)
    assert rigwatch.paused() is False


def test_a_running_rig_is_left_alone(monkeypatch):
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [123])
    monkeypatch.setattr(rigwatch, "start_rig", lambda: pytest.fail("must not restart a live rig"))
    monkeypatch.setattr("sys.argv", ["rigwatch.py"])
    assert rigwatch.main() == 0


def test_a_down_rig_is_started(monkeypatch):
    started = []
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [])
    monkeypatch.setattr(rigwatch, "start_rig", lambda: started.append(True) or 0)
    monkeypatch.setattr("sys.argv", ["rigwatch.py"])
    assert rigwatch.main() == 0
    assert started == [True]


def test_a_deliberately_stopped_rig_is_not_started(monkeypatch):
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [])
    monkeypatch.setattr(rigwatch, "paused", lambda: True)
    monkeypatch.setattr(rigwatch, "start_rig", lambda: pytest.fail("must respect a human stop"))
    monkeypatch.setattr("sys.argv", ["rigwatch.py"])
    assert rigwatch.main() == 0


def test_restart_storms_are_capped(monkeypatch):
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [])
    monkeypatch.setattr("sys.argv", ["rigwatch.py"])
    for _ in range(rigwatch.MAX_STARTS_PER_HOUR):
        rigwatch._record_start()
    monkeypatch.setattr(rigwatch, "start_rig",
                        lambda: pytest.fail("should have backed off, not started again"))
    assert rigwatch.main() == 1                       # nonzero: a human needs to look


def test_starts_older_than_an_hour_do_not_count_against_the_cap():
    rigwatch.STATE_FILE.write_text('{"starts": [1, 2, 3]}')   # epoch 1970
    assert rigwatch._recent_starts() == 0


# --------------------------------------------------------------------------- stale pause markers
# 2026-08-21: only start_critter_cam.bat ever cleared the marker, so a rig started by the LAN
# launcher or straight from python left an old one standing. Being newer than the last boot, it
# read as "paused by human" and the watchdog stopped guarding a rig that was up -- silently, with
# nothing to notice, until the next reboot aged it out.

def test_a_running_rig_clears_a_stale_pause_marker(monkeypatch):
    """The marker means "a human stopped this"; a RUNNING rig means that stop is spent, whoever
    undid it and however they started it."""
    monkeypatch.setattr(rigwatch, "boot_time", lambda: time.time() - 3600)
    rigwatch.PAUSE_MARKER.write_text("stopped from the video window")
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [123])
    monkeypatch.setattr(rigwatch, "start_rig", lambda: pytest.fail("must not restart a live rig"))
    monkeypatch.setattr("sys.argv", ["rigwatch.py"])
    assert rigwatch.main() == 0
    assert not rigwatch.PAUSE_MARKER.exists()
    assert rigwatch.paused() is False          # and so the rig is guarded from here on


def test_clearing_the_marker_is_a_no_op_when_there_is_none():
    assert rigwatch.clear_pause_marker() is False


def test_status_reports_a_stale_marker_without_clearing_it(monkeypatch):
    """--status promises to change nothing -- including this."""
    monkeypatch.setattr(rigwatch, "boot_time", lambda: time.time() - 3600)
    rigwatch.PAUSE_MARKER.write_text("stopped from the video window")
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [123])
    monkeypatch.setattr("sys.argv", ["rigwatch.py", "--status"])
    assert rigwatch.main() == 0
    assert rigwatch.PAUSE_MARKER.exists()


def test_a_stopped_rig_keeps_its_marker_and_stays_stopped(monkeypatch):
    """The regression that matters: self-healing must not eat a REAL deliberate stop. Rig down +
    marker present is exactly the case the marker exists for."""
    monkeypatch.setattr(rigwatch, "boot_time", lambda: time.time() - 3600)
    rigwatch.PAUSE_MARKER.write_text("stopped from the video window")
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [])
    monkeypatch.setattr(rigwatch, "start_rig", lambda: pytest.fail("must respect a human stop"))
    monkeypatch.setattr("sys.argv", ["rigwatch.py"])
    assert rigwatch.main() == 0
    assert rigwatch.PAUSE_MARKER.exists()


def test_it_refuses_to_act_when_it_cannot_tell_whether_the_rig_is_running(monkeypatch):
    """psutil is a hard dependency of rig_pids. If it is missing, "no pids" is indistinguishable
    from "rig is down" -- and acting on that would restart a healthy rig every five minutes,
    forever. Not knowing must mean doing nothing, loudly."""
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: None)
    monkeypatch.setattr(rigwatch, "start_rig",
                        lambda: pytest.fail("must not start a rig it cannot see"))
    monkeypatch.setattr("sys.argv", ["rigwatch.py"])
    assert rigwatch.main() == 1
