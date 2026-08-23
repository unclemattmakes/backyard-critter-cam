"""heavyio's whole job is to be conservative in one direction only: it may let two jobs overlap,
but it must never wedge a backup or a night of tracks behind a lock nobody is holding. These tests
are mostly about the ways a lock can be dead, because the crash this module exists to mitigate is
exactly the thing that leaves half-written files behind."""
from __future__ import annotations

import itertools
import json
import sys
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


def test_drive_quiet_reports_absent_when_drive_is_not_running(monkeypatch):
    monkeypatch.setattr(heavyio, "_drive_io_bytes", lambda: None)
    assert heavyio.wait_drive_quiet(timeout_s=999) == heavyio.DRIVE_ABSENT
    assert heavyio.DRIVE_ABSENT in heavyio.DRIVE_OK      # genuinely nothing outstanding


def test_drive_quiet_reports_drained_once_the_byte_counter_stops_climbing(monkeypatch):
    # Busy for a few samples, then flat -- the wait must end, and only after the calm streak.
    seq = iter([0, 50 * 2**20, 100 * 2**20] + [150 * 2**20] * 40)
    monkeypatch.setattr(heavyio, "_drive_io_bytes", lambda: next(seq))
    monkeypatch.setattr(heavyio.time, "sleep", lambda s: None)
    monkeypatch.setattr(heavyio, "DRIVE_SAMPLE_S", 0.0001)
    assert heavyio.wait_drive_quiet(timeout_s=999) == heavyio.DRIVE_DRAINED
    assert heavyio.DRIVE_DRAINED in heavyio.DRIVE_OK


def test_giving_up_waiting_is_not_reported_as_a_finished_upload(monkeypatch):
    """The whole reason these outcomes are not all 0. wait_drive_quiet returned 0 when Drive
    drained AND when the wait timed out, so migrate.py pack() -- which called it in a finally
    block and dropped the value on the floor -- announced a complete bundle for an upload that
    was still running. For a machine migration that is the one lie that costs you the data."""
    climbing = itertools.count(0, 50 * 2**20)               # never goes idle
    monkeypatch.setattr(heavyio, "_drive_io_bytes", lambda: next(climbing))
    monkeypatch.setattr(heavyio.time, "sleep", lambda s: None)
    monkeypatch.setattr(heavyio, "DRIVE_SAMPLE_S", 0.0001)
    assert heavyio.wait_drive_quiet(timeout_s=0.05) == heavyio.DRIVE_TIMEOUT
    assert heavyio.DRIVE_TIMEOUT not in heavyio.DRIVE_OK


def test_a_check_that_cannot_run_is_unknown_and_never_absent(monkeypatch, capsys):
    """None used to mean both "Drive is not running" and "psutil would not import", and every
    caller reads the first as "nothing to wait for". So on 2026-08-23 a pack run under the SYSTEM
    interpreter instead of .venv printed the reassuring line, skipped the wait, and exited 0
    while Drive was uploading."""
    def no_psutil():
        raise heavyio.DriveCheckUnavailable(
            r"psutil is not importable by C:\Python314\python.exe: No module named 'psutil'")
    monkeypatch.setattr(heavyio, "_drive_io_bytes", no_psutil)

    assert heavyio.wait_drive_quiet(timeout_s=999) == heavyio.DRIVE_UNKNOWN
    assert heavyio.DRIVE_UNKNOWN not in heavyio.DRIVE_OK
    out = capsys.readouterr().out
    assert "BROKEN CHECK" in out
    assert "python.exe" in out                  # names the interpreter: that IS the diagnosis
    assert "is not running" not in out          # must never read as the harmless case


def test_drive_io_bytes_raises_rather_than_returning_none_when_psutil_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)        # makes `import psutil` raise
    with pytest.raises(heavyio.DriveCheckUnavailable):
        heavyio._drive_io_bytes()


def test_drive_quiet_exit_code_can_gate_a_shell_caller(monkeypatch):
    """`heavyio.py --drive-quiet && <next step>` has to mean what it looks like it means."""
    monkeypatch.setattr(sys, "argv", ["heavyio.py", "--drive-quiet"])
    monkeypatch.setattr(heavyio, "wait_drive_quiet", lambda *a, **k: heavyio.DRIVE_TIMEOUT)
    assert heavyio.main() == 1
    monkeypatch.setattr(heavyio, "wait_drive_quiet", lambda *a, **k: heavyio.DRIVE_UNKNOWN)
    assert heavyio.main() == 1
    monkeypatch.setattr(heavyio, "wait_drive_quiet", lambda *a, **k: heavyio.DRIVE_DRAINED)
    assert heavyio.main() == 0


def test_status_says_it_cannot_tell_rather_than_not_running(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["heavyio.py", "--status"])
    monkeypatch.setattr(heavyio, "_drive_io_bytes",
                        lambda: (_ for _ in ()).throw(heavyio.DriveCheckUnavailable("no psutil")))
    assert heavyio.main() == 0
    out = capsys.readouterr().out
    assert "CANNOT TELL" in out
    assert "not running" not in out
