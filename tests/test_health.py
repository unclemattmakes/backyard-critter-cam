"""
health.rig_health: rigwatch's alarms + backup's verdict, combined into the one list the morning
email and the dashboard show. Pure file I/O in tmp_path; the live records are never read.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta

import pytest

import health

NOW = 1_790_900_000.0          # 2026-10-01, a fixed "now" so ages are exact


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


@pytest.fixture
def rec(tmp_path):
    """Paths for the two records, plus writers for each."""
    rw, bk = tmp_path / ".rigwatch_state.json", tmp_path / ".backup_state.json"

    def run(**kw):
        return health.rig_health(NOW, rigwatch_state=rw, backup_state=bk, **kw)
    run.rw, run.bk = rw, bk
    return run


def _rigwatch(path, checked_at, alarms=None):
    path.write_text(json.dumps({"starts": [], "checked_at": checked_at, "alarms": alarms or {}}))


def _alarm(sev, msg, first):
    return {"severity": sev, "message": msg, "first_seen": first, "last_seen": NOW - 60}


def test_no_records_means_nothing_to_say(rec):
    assert rec() == []


def test_a_quiet_fresh_watchdog_and_a_good_backup_say_nothing(rec):
    _rigwatch(rec.rw, NOW - 120)
    rec.bk.write_text(json.dumps({"last_run": _iso(NOW - 3600), "last_ok": _iso(NOW - 3600),
                                  "failures": 0, "lost_clip_days": {}}))
    assert rec() == []


def test_rigwatch_alarms_come_through_most_urgent_first(rec):
    _rigwatch(rec.rw, NOW - 120, {
        "batch_stale": _alarm("warn", "The nightly batch has not finished.", NOW - 7200),
        "naming_stale": _alarm("alarm", "Species naming has stopped.", NOW - 600),
    })
    items = rec()
    assert [i["kind"] for i in items] == ["naming_stale", "batch_stale"]
    assert items[0] == {"severity": "alarm", "kind": "naming_stale",
                        "message": "Species naming has stopped.", "since": _iso(NOW - 600)}


def test_a_stale_rigwatch_record_is_itself_the_alarm_and_its_old_alarms_are_dropped(rec):
    """Its alarms are as old as its last run; what is true NOW is that nothing is watching."""
    _rigwatch(rec.rw, NOW - 3 * 3600,
              {"batch_stale": _alarm("warn", "old news", NOW - 9 * 3600)})
    items = rec()
    assert [i["kind"] for i in items] == ["rigwatch_silent"]
    assert items[0]["severity"] == "alarm" and "3 h" in items[0]["message"]
    assert items[0]["since"] == _iso(NOW - 3 * 3600)


def test_a_record_from_before_checked_at_uses_its_mtime(rec):
    import os
    rec.rw.write_text('{"starts": [], "naming_backlog": 0}')
    os.utime(rec.rw, (NOW - 7200, NOW - 7200))
    assert [i["kind"] for i in rec()] == ["rigwatch_silent"]


@pytest.mark.parametrize("junk", ["{not json", "[]", "null", ""])
def test_a_corrupt_record_is_no_record(rec, junk):
    rec.rw.write_text(junk)
    rec.bk.write_text(junk)
    assert rec() == []


def test_a_failed_backup_is_an_alarm_since_it_started_failing(rec):
    rec.bk.write_text(json.dumps({"last_run": _iso(NOW - 3600), "last_ok": _iso(NOW - 25 * 3600),
                                  "failures": 2, "failing_since": _iso(NOW - 3600)}))
    items = rec()
    assert [i["kind"] for i in items] == ["backup_failed"]
    assert "2 failures" in items[0]["message"] and items[0]["since"] == _iso(NOW - 3600)


def test_no_good_backup_in_36_hours_is_an_alarm(rec):
    rec.bk.write_text(json.dumps({"last_run": _iso(NOW - 40 * 3600),
                                  "last_ok": _iso(NOW - 40 * 3600), "failures": 0}))
    items = rec()
    assert [i["kind"] for i in items] == ["backup_stale"]
    assert "40 h" in items[0]["message"]


def test_lost_clip_days_are_news_for_one_day_only(rec):
    rec.bk.write_text(json.dumps({
        "last_run": _iso(NOW - 3600), "last_ok": _iso(NOW - 3600), "failures": 0,
        "lost_clip_days": {"glass_door_cam/2026-08-24": _iso(NOW - 9 * 86400),
                           "glass_door_cam/2026-09-29": _iso(NOW - 3600)}}))
    items = rec()
    assert [i["kind"] for i in items] == ["clips_lost"]
    msg = items[0]["message"]
    assert "1 day of clips was pruned" in msg and "glass_door_cam/2026-09-29" in msg
    assert "2026-08-24" not in msg and "2 such days in all" in msg
    # The next morning it is history, not news.
    later = health.rig_health(NOW + 86400, rigwatch_state=rec.rw, backup_state=rec.bk)
    assert later == []


def test_both_sources_combine_and_urls_never_leave(rec):
    _rigwatch(rec.rw, NOW - 60, {
        "naming_error": _alarm("alarm", "Species naming is erroring: GET "
                                        "http://admin:hunter2@192.168.1.20/cgi failed.", NOW - 60)})
    rec.bk.write_text(json.dumps({"last_run": _iso(NOW - 3600), "last_ok": None, "failures": 1,
                                  "failing_since": _iso(NOW - 3600)}))
    items = rec()
    assert {i["kind"] for i in items} == {"naming_error", "backup_failed"}
    text = json.dumps(items)
    assert "hunter2" not in text and "<url>" in text


def test_write_json_is_atomic_and_leaves_no_temp(tmp_path):
    p = tmp_path / ".backup_state.json"
    assert health.write_json(p, {"a": 1}) is True
    assert health.read_json(p) == {"a": 1}
    assert [q.name for q in tmp_path.iterdir()] == [".backup_state.json"]


def test_now_accepts_a_datetime(rec):
    _rigwatch(rec.rw, NOW - 120)
    assert health.rig_health(datetime.fromtimestamp(NOW).astimezone(),
                             rigwatch_state=rec.rw, backup_state=rec.bk) == []
    assert health.rig_health(datetime.fromtimestamp(NOW) + timedelta(hours=1),
                             rigwatch_state=rec.rw, backup_state=rec.bk)[0]["kind"] == \
        "rigwatch_silent"


def test_defaults_read_the_configured_paths(monkeypatch, tmp_path):
    import config
    monkeypatch.setattr(config, "RIGWATCH_STATE_FILE", tmp_path / "rw.json")
    monkeypatch.setattr(config, "BACKUP_STATE_FILE", tmp_path / "bk.json")
    _rigwatch(tmp_path / "rw.json", time.time() - 7200)
    assert [i["kind"] for i in health.rig_health()] == ["rigwatch_silent"]
