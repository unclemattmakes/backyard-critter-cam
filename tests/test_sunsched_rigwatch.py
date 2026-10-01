"""The two scheduling guards.

sunsched exists because a hardcoded clock time drifts out of the glare window as the season turns;
rigwatch exists because nothing brought the rig back after the 2026-08-21 bugcheck. Both are
allowed to do nothing, and neither may ever do something surprising -- restarting a rig a human
just stopped, or parking the batch on top of the raccoon peak."""
from __future__ import annotations

import json
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
    monkeypatch.setattr(rigwatch, "HOLD_MARKER", tmp_path / ".rig_hold")
    monkeypatch.setattr(rigwatch, "STATE_FILE", tmp_path / ".rigwatch_state.json")
    monkeypatch.setattr(rigwatch, "LOG_FILE", tmp_path / "logs" / "rigwatch.log")
    monkeypatch.setattr(rigwatch, "REPORTS_DIR", tmp_path / "reports")
    # A config_local.py heartbeat_url must never be pinged from a test, and nothing may reach the net.
    monkeypatch.setattr(config.CONFIG, "heartbeat_url", None)
    monkeypatch.setattr(rigwatch.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("network call from a test"))
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


def test_a_hold_survives_a_reboot_and_keeps_the_rig_down(monkeypatch):
    """--hold exists because the HOST was hanging under the rig: a pause aged out at every
    crash-reboot and the watchdog started the rig straight back into the hang."""
    monkeypatch.setattr("sys.argv", ["rigwatch.py", "--hold"])
    assert rigwatch.main() == 0
    monkeypatch.setattr(rigwatch, "boot_time", lambda: time.time() + 60)   # "rebooted since"
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [])
    monkeypatch.setattr(rigwatch, "start_rig", lambda: pytest.fail("must respect --hold"))
    monkeypatch.setattr("sys.argv", ["rigwatch.py"])
    assert rigwatch.main() == 0


def test_a_hold_is_not_cleared_by_a_running_rig_and_release_undoes_it(monkeypatch):
    rigwatch.HOLD_MARKER.write_text("held")
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [123])
    monkeypatch.setattr("sys.argv", ["rigwatch.py"])
    rigwatch.main()
    assert rigwatch.held() is True
    monkeypatch.setattr("sys.argv", ["rigwatch.py", "--release"])
    assert rigwatch.main() == 0
    assert rigwatch.held() is False


def _log_text() -> str:
    return rigwatch.LOG_FILE.read_text(encoding="utf-8") if rigwatch.LOG_FILE.exists() else ""


def test_a_hold_that_keeps_a_down_rig_down_says_so_once_an_hour(monkeypatch):
    rigwatch.HOLD_MARKER.write_text("held")
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [])
    monkeypatch.setattr(rigwatch, "start_rig", lambda: pytest.fail("must respect --hold"))
    monkeypatch.setattr("sys.argv", ["rigwatch.py"])
    assert rigwatch.main() == 0
    assert rigwatch.main() == 0                         # five minutes later: no second line
    assert _log_text().count("HELD") == 1
    st = rigwatch._state()
    st["hold_logged_at"] = time.time() - rigwatch.REPEAT_ALARM_S - 1
    rigwatch._write_state(st)
    assert rigwatch.main() == 0
    assert _log_text().count("HELD") == 2


def test_a_hold_on_a_running_rig_logs_nothing(monkeypatch):
    rigwatch.HOLD_MARKER.write_text("held")
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [123])
    monkeypatch.setattr("sys.argv", ["rigwatch.py"])
    assert rigwatch.main() == 0
    assert "HELD" not in _log_text()


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


# --------------------------------------------------------------------------- nightly batch
# The batch hung for two days and nothing noticed. Every successful night writes an eval artifact,
# so an old newest artifact is the signal.

def _eval_artifact(hours_ago: float) -> None:
    rigwatch.REPORTS_DIR.mkdir(exist_ok=True)
    when = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    (rigwatch.REPORTS_DIR / f"eval_{when:%Y%m%dT%H%M%SZ}.json").write_text("{}")


def test_a_recent_eval_artifact_means_the_batch_is_fine():
    _eval_artifact(48)
    _eval_artifact(20)
    h = rigwatch.batch_health()
    assert h["present"] and not h["stale"]
    assert 19 * 3600 < h["age_s"] < 21 * 3600            # the NEWEST one counts
    assert rigwatch.check_batch(h) is False


def test_no_artifacts_at_all_is_not_an_alarm():
    """A machine that has never run the batch has nothing to be late for."""
    h = rigwatch.batch_health()
    assert h["present"] is False and h["stale"] is False


def test_a_stale_batch_is_logged_on_every_path_once_an_hour(monkeypatch):
    _eval_artifact(50)
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [123])
    monkeypatch.setattr("sys.argv", ["rigwatch.py"])
    assert rigwatch.main() == 0                          # alarms do not change the exit code
    assert rigwatch.main() == 0
    assert _log_text().count("NIGHTLY BATCH IS STALE") == 1
    rigwatch.HOLD_MARKER.write_text("held")              # a held, down rig still checks the batch
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [])
    st = rigwatch._state()
    st["batch_logged_at"] = 0
    rigwatch._write_state(st)
    assert rigwatch.main() == 0
    assert _log_text().count("NIGHTLY BATCH IS STALE") == 2


def test_the_stale_threshold_is_configurable(monkeypatch):
    _eval_artifact(30)
    assert rigwatch.batch_health()["stale"] is False     # default 36 h
    monkeypatch.setattr(config.CONFIG, "batch_stale_hours", 24.0)
    assert rigwatch.batch_health()["stale"] is True


def test_an_unstamped_artifact_falls_back_to_its_mtime():
    import os
    rigwatch.REPORTS_DIR.mkdir()
    p = rigwatch.REPORTS_DIR / "eval_handmade.json"
    p.write_text("{}")
    old = time.time() - 72 * 3600
    os.utime(p, (old, old))
    h = rigwatch.batch_health()
    assert h["newest"] == "eval_handmade.json" and h["stale"] is True


def test_status_shows_the_batch_and_changes_nothing(monkeypatch, capsys):
    _eval_artifact(50)
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [123])
    monkeypatch.setattr("sys.argv", ["rigwatch.py", "--status"])
    assert rigwatch.main() == 0
    out = capsys.readouterr().out
    assert "nightly batch" in out and "STALE" in out
    assert "NIGHTLY BATCH" not in _log_text()


# --------------------------------------------------------------------------- off-host heartbeat
# Every alarm above lives on the same box; when the box is down, only an outside service that
# notices the pings stopping can tell anyone.
URL = "https://hc-ping.example/0b1c-secret-token"


class _Resp:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        return b"OK"


@pytest.fixture
def pings(monkeypatch):
    sent = []

    def fake_urlopen(req, timeout=None):
        assert timeout is not None and timeout <= 10
        assert req.get_method() == "GET"
        sent.append(req.full_url)
        return _Resp()
    monkeypatch.setattr(config.CONFIG, "heartbeat_url", URL)
    monkeypatch.setattr(rigwatch.urllib.request, "urlopen", fake_urlopen)
    # Pinned so a real .naming_status.json on the machine cannot decide healthy vs /fail.
    monkeypatch.setattr(rigwatch, "naming_health", lambda: {"present": False, "stale": False})
    return sent


def _run(monkeypatch, pids, *argv):
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: pids)
    monkeypatch.setattr("sys.argv", ["rigwatch.py", *argv])
    return rigwatch.main()


def test_a_healthy_run_pings_the_heartbeat(monkeypatch, pings):
    assert _run(monkeypatch, [123]) == 0
    assert pings == [URL]


def test_a_held_or_paused_down_rig_still_counts_as_healthy(monkeypatch, pings):
    rigwatch.HOLD_MARKER.write_text("held")
    assert _run(monkeypatch, []) == 0
    rigwatch.HOLD_MARKER.unlink()
    monkeypatch.setattr(rigwatch, "paused", lambda: True)
    assert _run(monkeypatch, []) == 0
    assert pings == [URL, URL]


def test_stale_naming_pings_fail(monkeypatch, pings):
    monkeypatch.setattr(rigwatch, "naming_health", lambda: {
        "present": True, "state": "ready", "age_s": 3600.0, "stale": True, "backlog": None})
    assert _run(monkeypatch, [123]) == 0
    assert pings == [URL + "/fail"]


def test_a_stale_batch_pings_fail(monkeypatch, pings):
    _eval_artifact(50)
    assert _run(monkeypatch, [123]) == 0
    assert pings == [URL + "/fail"]


def test_a_restart_storm_or_blind_watchdog_pings_fail(monkeypatch, pings):
    assert _run(monkeypatch, None) == 1
    for _ in range(rigwatch.MAX_STARTS_PER_HOUR):
        rigwatch._record_start()
    assert _run(monkeypatch, []) == 1
    assert pings == [URL + "/fail"] * 2


def test_fail_keeps_a_query_string_intact(monkeypatch, pings):
    monkeypatch.setattr(config.CONFIG, "heartbeat_url", URL + "/?rid=abc")
    rigwatch.ping_heartbeat(False)
    assert pings == [URL + "/fail?rid=abc"]


def test_the_restart_happens_before_the_ping_and_survives_a_dead_network(monkeypatch):
    order = []

    def boom(req, timeout=None):
        order.append("ping")
        raise OSError("connect to https://hc-ping.example/0b1c-secret-token timed out")
    monkeypatch.setattr(config.CONFIG, "heartbeat_url", URL)
    monkeypatch.setattr(rigwatch.urllib.request, "urlopen", boom)
    monkeypatch.setattr(rigwatch, "start_rig", lambda: order.append("start") or 0)
    assert _run(monkeypatch, []) == 0
    assert order == ["start", "ping"]
    text = _log_text()
    assert "heartbeat to hc-ping.example failed" in text
    assert "secret-token" not in text                    # the URL is the secret: host only


def test_no_url_means_no_network_at_all(monkeypatch):
    # The autouse fixture fails the test on any urlopen; heartbeat_url is None there.
    assert _run(monkeypatch, [123]) == 0


def test_status_never_pings_and_shows_only_the_host(monkeypatch, pings, capsys):
    assert _run(monkeypatch, [123], "--status") == 0
    out = capsys.readouterr().out
    assert pings == []
    assert "hc-ping.example" in out and "secret-token" not in out


# --------------------------------------------------------------------------- the health record
# The log above is for post-mortems; nothing reads it live. The state file's "alarms" is what the
# morning email and the dashboard read (health.py), so it must hold exactly what is true now.

def _alarms() -> dict:
    return rigwatch._state().get("alarms")


def test_a_standing_alarm_keeps_its_first_seen_and_clears_when_the_condition_does(monkeypatch):
    _eval_artifact(50)
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [123])
    monkeypatch.setattr(rigwatch, "naming_health", lambda: {"present": False, "stale": False})
    monkeypatch.setattr("sys.argv", ["rigwatch.py"])
    assert rigwatch.main() == 0
    first = _alarms()["batch_stale"]
    assert first["severity"] == "warn" and "50 h" in first["message"]
    assert first["first_seen"] == first["last_seen"]
    assert rigwatch.main() == 0                          # five minutes later: still standing
    again = _alarms()["batch_stale"]
    assert again["first_seen"] == first["first_seen"] and again["last_seen"] >= first["last_seen"]
    _eval_artifact(1)                                    # the batch finished a night
    assert rigwatch.main() == 0
    assert _alarms() == {}
    assert "cleared: batch_stale" in _log_text()
    assert time.time() - rigwatch._state()["checked_at"] < 60


def test_every_alarm_kind_lands_in_the_record(monkeypatch):
    monkeypatch.setattr("sys.argv", ["rigwatch.py"])
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: None)
    rigwatch.main()
    assert set(_alarms()) == {"psutil_missing"}

    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [])
    rigwatch.HOLD_MARKER.write_text("held")
    rigwatch.main()
    assert set(_alarms()) == {"rig_held"}                # psutil_missing cleared itself
    rigwatch.main()                                      # the LOG line is hourly; the record is not
    assert set(_alarms()) == {"rig_held"}
    rigwatch.HOLD_MARKER.unlink()

    for _ in range(rigwatch.MAX_STARTS_PER_HOUR):
        rigwatch._record_start()
    monkeypatch.setattr(rigwatch, "start_rig", lambda: pytest.fail("should back off"))
    rigwatch.main()
    assert set(_alarms()) == {"restart_storm"}
    assert _alarms()["restart_storm"]["severity"] == "alarm"


def test_naming_alarms_land_in_the_record(monkeypatch):
    monkeypatch.setattr("sys.argv", ["rigwatch.py"])
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [123])
    monkeypatch.setattr(rigwatch, "naming_health", lambda: {
        "present": True, "state": "ready", "age_s": 3600.0, "stale": True, "backlog": None})
    rigwatch.main()
    assert "60 min ago" in _alarms()["naming_stale"]["message"]
    monkeypatch.setattr(rigwatch, "naming_health", lambda: {
        "present": True, "state": "error", "age_s": 5.0, "stale": False, "backlog": 3,
        "detail": "CUDA out of memory"})
    rigwatch.main()
    assert set(_alarms()) == {"naming_error"}
    assert "CUDA out of memory" in _alarms()["naming_error"]["message"]


def test_a_corrupt_state_file_is_survived_and_replaced(monkeypatch):
    rigwatch.STATE_FILE.write_text("{torn mid-wri")
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [123])
    monkeypatch.setattr(rigwatch, "naming_health", lambda: {"present": False, "stale": False})
    monkeypatch.setattr("sys.argv", ["rigwatch.py"])
    assert rigwatch.main() == 0
    st = json.loads(rigwatch.STATE_FILE.read_text())
    assert st["alarms"] == {} and "checked_at" in st
    assert [p.name for p in rigwatch.STATE_FILE.parent.glob("*.tmp")] == []


def test_status_lists_the_active_alarms_and_writes_nothing(monkeypatch, capsys):
    rigwatch._write_state({"starts": [], "checked_at": 1.0, "alarms": {
        "rig_held": {"severity": "warn", "message": "The rig is down and held.",
                     "first_seen": 1.0, "last_seen": 1.0}}})
    before = rigwatch.STATE_FILE.read_text()
    monkeypatch.setattr(rigwatch, "rig_pids", lambda: [])
    monkeypatch.setattr("sys.argv", ["rigwatch.py", "--status"])
    assert rigwatch.main() == 0
    assert "[warn] rig_held: The rig is down and held." in capsys.readouterr().out
    assert rigwatch.STATE_FILE.read_text() == before
