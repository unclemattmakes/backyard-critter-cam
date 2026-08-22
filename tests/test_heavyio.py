"""heavyio's whole job is to be conservative in one direction only: it may let two jobs overlap,
but it must never wedge a backup or a night of tracks behind a lock nobody is holding. These tests
are mostly about the ways a lock can be dead, because the crash this module exists to mitigate is
exactly the thing that leaves half-written files behind."""
from __future__ import annotations

import json
import time

import pytest

import heavyio


@pytest.fixture(autouse=True)
def _isolated_lockdir(tmp_path, monkeypatch):
    monkeypatch.setattr(heavyio, "LOCK_DIR", tmp_path / ".locks")
    yield


def _write_lock(name: str, **fields) -> None:
    heavyio.LOCK_DIR.mkdir(parents=True, exist_ok=True)
    base = {"name": name, "pid": 999999, "host": "test",
            "started": "2026-08-21T00:00:00-07:00", "started_epoch": time.time(), "note": ""}
    base.update(fields)
    (heavyio.LOCK_DIR / f"{name}.lock").write_text(json.dumps(base), encoding="utf-8")


def test_acquire_then_release_round_trip():
    heavyio.acquire("batch", wait_s=1, note="hello")
    held = heavyio.read_holders()
    assert [h["name"] for h in held] == ["batch"]
    assert held[0]["note"] == "hello"
    heavyio.release("batch")
    assert heavyio.read_holders() == []


def test_release_of_a_lock_never_taken_is_not_an_error():
    assert heavyio.release("batch") == 0


def test_a_holder_whose_process_is_gone_is_reaped(monkeypatch):
    monkeypatch.setattr(heavyio, "_alive", lambda pid: False)
    _write_lock("backup")
    assert heavyio.read_holders() == []
    assert not (heavyio.LOCK_DIR / "backup.lock").exists()   # reaping actually deletes


def test_a_live_pid_holder_is_respected(monkeypatch):
    monkeypatch.setattr(heavyio, "_alive", lambda pid: True)
    _write_lock("backup")
    assert [h["name"] for h in heavyio.read_holders()] == ["backup"]


def test_absurdly_old_pid_holder_is_reaped_even_if_the_pid_lives(monkeypatch):
    # Pids get recycled, so "the pid exists" is not proof the JOB exists.
    monkeypatch.setattr(heavyio, "_alive", lambda pid: True)
    _write_lock("backup", started_epoch=time.time() - heavyio.MAX_LOCK_AGE_S - 1)
    assert heavyio.read_holders() == []


def test_ttl_lock_survives_a_dead_pid_until_it_expires(monkeypatch):
    # The .bat case: the python that took the lock exited immediately and by design.
    monkeypatch.setattr(heavyio, "_alive", lambda pid: False)
    _write_lock("batch", expires_epoch=time.time() + 600)
    assert [h["name"] for h in heavyio.read_holders()] == ["batch"]


def test_ttl_lock_is_reaped_once_expired(monkeypatch):
    monkeypatch.setattr(heavyio, "_alive", lambda pid: True)
    _write_lock("batch", expires_epoch=time.time() - 1)
    assert heavyio.read_holders() == []


def test_acquire_with_ttl_records_an_expiry():
    heavyio.acquire("batch", wait_s=1, ttl_min=6)
    info = json.loads((heavyio.LOCK_DIR / "batch.lock").read_text(encoding="utf-8"))
    assert info["expires_epoch"] == pytest.approx(time.time() + 360, abs=10)


def test_a_torn_lock_file_gets_no_authority():
    # A NUL-filled or truncated file is what an unclean shutdown leaves; it must not wedge a job.
    heavyio.LOCK_DIR.mkdir(parents=True, exist_ok=True)
    (heavyio.LOCK_DIR / "backup.lock").write_bytes(b"\x00" * 90)
    assert heavyio.read_holders() == []


def test_waiting_times_out_and_runs_anyway_rather_than_skipping(monkeypatch, capsys):
    """The fail-open contract: a backup that silently did not happen is worse than an overlap."""
    monkeypatch.setattr(heavyio, "_alive", lambda pid: True)
    _write_lock("batch")
    t0 = time.monotonic()
    assert heavyio.acquire("backup", wait_s=0.1) == 0
    assert time.monotonic() - t0 < 30                       # gave up promptly, did not hang
    out = capsys.readouterr().out
    assert "WARNING" in out and "WITHOUT a clear box" in out
    assert {h["name"] for h in heavyio.read_holders()} == {"batch", "backup"}


def test_reacquiring_your_own_name_does_not_wait_on_yourself(monkeypatch):
    monkeypatch.setattr(heavyio, "_alive", lambda pid: True)
    _write_lock("backup")
    t0 = time.monotonic()
    heavyio.acquire("backup", wait_s=30)                    # same name: not an "other" holder
    assert time.monotonic() - t0 < 5


def test_held_context_manager_releases_even_on_an_exception():
    with pytest.raises(RuntimeError):
        with heavyio.held("backup", wait_s=1):
            assert heavyio.read_holders()
            raise RuntimeError("boom")
    assert heavyio.read_holders() == []


def test_drive_quiet_returns_immediately_when_drive_is_not_running(monkeypatch):
    monkeypatch.setattr(heavyio, "_drive_io_bytes", lambda: None)
    assert heavyio.wait_drive_quiet(timeout_s=999) == 0


def test_drive_quiet_returns_once_the_byte_counter_stops_climbing(monkeypatch):
    # Busy for a few samples, then flat -- the wait must end, and only after the calm streak.
    seq = iter([0, 50 * 2**20, 100 * 2**20] + [150 * 2**20] * 40)
    monkeypatch.setattr(heavyio, "_drive_io_bytes", lambda: next(seq))
    monkeypatch.setattr(heavyio.time, "sleep", lambda s: None)
    monkeypatch.setattr(heavyio, "DRIVE_SAMPLE_S", 0.0001)
    assert heavyio.wait_drive_quiet(timeout_s=999) == 0
