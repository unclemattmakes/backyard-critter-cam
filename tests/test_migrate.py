"""
Tests for migrate.py -- moving the whole rig between machines -- plus the backup.py additions
it stands on (meta coverage for the reference batches and the DB's sidecar ledgers; the settle
window; include_today).

Same discipline as test_backup.py: a fake OLD project and a fake NEW project are built under
tmp, and migrate.ROOT/CONFIG and backup.ROOT/CONFIG are pointed at whichever machine the step
under test is "on" -- pack runs against the old root, restore against the new one, exactly the
two-machine reality. heavyio is stubbed out (there is no Google Drive in CI, and a test that
sleeps 30 s waiting for one would be worse than none). The databases are real: db.connect()
builds the full schema, so the pruned_at column, the ledgers convention and connect_readonly
behave exactly as on a rig.

The properties pinned here, in order of how expensive they'd be to learn in production:

  * the round trip is COMPLETE -- db, crops, live clips, reference photos, ledgers, config all
    arrive, and the DB's rows point at files that exist;
  * restore resurrects only what the old machine actually held: a pruned clip stays
    archive-only instead of fighting the new machine's disk budget;
  * a corrupt DB snapshot is never installed -- the next-oldest good one is, or nothing is;
  * restore never overwrites anything, which is also what makes it re-runnable after a crash;
  * restore refuses to run over an existing rig (it moves rigs, it does not merge them).
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import backup
import db
import migrate

# Real dates relative to the wall clock, because pack() asks date.today() itself: two finished
# days for ordinary content, and today for the include-today path.
D2 = (date.today() - timedelta(days=2)).isoformat()
D1 = (date.today() - timedelta(days=1)).isoformat()
D0 = date.today().isoformat()

LIVE_CLIP = f"clips/glass_door_cam/{D1}/live.mp4"
PRUNED_CLIP = f"clips/glass_door_cam/{D2}/pruned.mp4"
CROP_A = f"crops/{D1}/a.jpg"
CROP_B = f"crops/{D2}/b.jpg"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """`use(root)` points both modules' ROOT and CONFIG at `root` and returns the fake CONFIG --
    call it once for the old machine, again for the new one. Also stubs heavyio (no locks, no
    Drive wait) and zeroes the pack settle window, because every file a test writes is seconds
    old by definition; the settle behaviour itself is pinned separately with backdated mtimes."""
    def use(root: Path) -> SimpleNamespace:
        root.mkdir(parents=True, exist_ok=True)
        cfg = SimpleNamespace(
            db_path=root / "backyard.db",
            clips_dir=root / "clips",
            crops_dir=root / "crops",
            frames_dir=root / "frames",
            clip_crops_dir=root / "clip_crops",
            reference_dir=root / "reference",
            reference_crops_dir=root / "reference_crops",
            backup_dest=None,
        )
        for mod in (migrate, backup):
            monkeypatch.setattr(mod, "ROOT", root)
            monkeypatch.setattr(mod, "CONFIG", cfg)
        return cfg

    monkeypatch.setattr(migrate.heavyio, "acquire", lambda *a, **k: 0)
    monkeypatch.setattr(migrate.heavyio, "release", lambda *a, **k: 0)
    monkeypatch.setattr(migrate.heavyio, "wait_drive_quiet", lambda *a, **k: 0)
    monkeypatch.setattr(migrate, "HF_HUB", tmp_path / "hf-hub")
    monkeypatch.setattr(migrate, "PACK_SETTLE_S", 0.0)
    return use


def _file(root: Path, rel: str, content: bytes = b"x" * 32) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def _add_clip(root: Path, rel: str, started: str, pruned_at: str | None = None) -> int:
    conn = db.connect(root / "backyard.db")
    try:
        cid = db.insert_clip(conn, source="glass_door_cam", clip_path=rel, started_at=started,
                             ended_at=None, fps=15.0, width=640, height=360, frame_count=10,
                             detection_count=1, max_confidence=0.9)
        if pruned_at:
            conn.execute("UPDATE clips SET pruned_at = ? WHERE id = ?", (pruned_at, cid))
            conn.commit()
        return cid
    finally:
        conn.close()


def _add_detection(root: Path, crop_rel: str, ts: str) -> int:
    conn = db.connect(root / "backyard.db")
    try:
        return db.insert_detection(conn, timestamp=ts, source="glass_door_cam",
                                   detection_class="animal", confidence=0.9,
                                   bbox=(0, 0, 10, 10), frame_w=640, frame_h=360,
                                   crop_path=crop_rel)
    finally:
        conn.close()


def build_old_rig(root: Path) -> None:
    """A small but honest rig: two crops with rows, one live clip, one clip the pruner already
    ate (row kept, pruned_at set, file gone -- the soft-prune reality), the hand-shot reference
    batch, an import ledger, and the machine's private config."""
    _file(root, CROP_A, b"crop-a")
    _file(root, CROP_B, b"crop-b")
    _add_detection(root, CROP_A, f"{D1}T10:00:00")
    _add_detection(root, CROP_B, f"{D2}T21:00:00")
    _file(root, LIVE_CLIP, b"live-video")
    _file(root, PRUNED_CLIP, b"pruned-video")
    _add_clip(root, LIVE_CLIP, f"{D1}T10:00:00")
    _add_clip(root, PRUNED_CLIP, f"{D2}T21:00:00", pruned_at=f"{D1}T03:00:00")
    _file(root, "reference/Alpha/portrait.jpg", b"alpha")
    _file(root, "reference_crops/Alpha/portrait_crop.jpg", b"alpha-crop")
    _file(root, "backyard.db.imported-trail_cam_sd.txt", b"IMAG0001.mp4\n")
    _file(root, "config_local.py", b"# the old machine's private config\n")


def _prune_file(root: Path) -> None:
    """What the disk budget does between backups: the file goes, the row stays."""
    (root / PRUNED_CLIP).unlink()


# --- the round trip ----------------------------------------------------------------------

def test_round_trip_moves_the_whole_rig(env, tmp_path, capsys):
    """Pack on one machine, restore on another: the database arrives intact and everything it
    references -- crops, live clips, references, ledger, config -- is on the new disk."""
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    # A weekly backup ran while the pruned clip's file still existed -- that zip is the only
    # copy of it now, exactly the archive-outlives-the-prune contract.
    backup.archive_media(old / "clips", dest / "clips", date.today(), dry_run=False,
                         include_today=True)
    _prune_file(old)

    assert migrate.pack(dest, include_weights=False) == 0

    use(new)
    assert migrate.restore(dest) == 0

    assert (new / "backyard.db").is_file()
    conn = sqlite3.connect(f"file:{(new / 'backyard.db').as_posix()}?mode=ro", uri=True)
    try:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 2
    finally:
        conn.close()
    assert (new / CROP_A).read_bytes() == b"crop-a"
    assert (new / CROP_B).read_bytes() == b"crop-b"
    assert (new / LIVE_CLIP).read_bytes() == b"live-video"
    assert (new / "reference/Alpha/portrait.jpg").read_bytes() == b"alpha"
    assert (new / "reference_crops/Alpha/portrait_crop.jpg").read_bytes() == b"alpha-crop"
    assert (new / "backyard.db.imported-trail_cam_sd.txt").is_file()
    assert (new / "config_local.py").read_bytes() == b"# the old machine's private config\n"
    out = capsys.readouterr().out
    assert "every sampled file present" in out
    assert "every file present" in out


def test_pruned_clips_stay_archive_only(env, tmp_path, capsys):
    """The archive deliberately holds more than the machine does. Restore must rebuild the
    machine, not the archive: the pruned clip's file stays in the zip, its row stays pruned."""
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    backup.archive_media(old / "clips", dest / "clips", date.today(), dry_run=False,
                         include_today=True)
    _prune_file(old)
    migrate.pack(dest, include_weights=False)

    use(new)
    migrate.restore(dest)

    assert (new / LIVE_CLIP).is_file()
    assert not (new / PRUNED_CLIP).exists()
    out = capsys.readouterr().out
    assert "1 pruned -- staying archive-only" in out
    # ...and the bytes really are still in the archive for web.py's archive_cache to serve.
    day_zip = dest / "clips" / f"clips-glass_door_cam-{D2}.zip"
    with zipfile.ZipFile(day_zip) as zf:
        assert zf.read(PRUNED_CLIP) == b"pruned-video"


def test_restore_from_a_weekly_backup_says_what_is_missing(env, tmp_path, capsys):
    """Disaster recovery is the same restore pointed at the weekly folder -- which legitimately
    lacks whatever was recorded after its last run. That gap must be reported, not discovered
    as broken images in the dashboard."""
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "backup-folder"
    use = env
    use(old)
    build_old_rig(old)
    migrate.pack(dest, include_weights=False)
    # Recorded after the pack, so it is in the DB snapshot below but in no media zip.
    _file(old, f"clips/glass_door_cam/{D0}/after.mp4", b"too-new")
    _add_clip(old, f"clips/glass_door_cam/{D0}/after.mp4", f"{D0}T09:00:00")
    _file(old, f"crops/{D0}/after.jpg", b"too-new")
    _add_detection(old, f"crops/{D0}/after.jpg", f"{D0}T09:00:00")
    backup.snapshot_db(old / "backyard.db", dest / "snapshots", date.today(), False)

    use(new)
    assert migrate.restore(dest) == 0

    assert (new / "backyard.db").is_file()
    out = capsys.readouterr().out
    assert "MISSING" in out


# --- refusing to make things worse -------------------------------------------------------

def test_restore_refuses_an_existing_rig(env, tmp_path):
    """This tool moves a rig onto a machine; it must never quietly merge two or replace one."""
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    migrate.pack(dest, include_weights=False)

    use(new)
    _file(new, "backyard.db", b"an existing rig's database")
    with pytest.raises(SystemExit, match="already a rig here"):
        migrate.restore(dest)
    assert (new / "backyard.db").read_bytes() == b"an existing rig's database"


def test_restore_refuses_a_stale_wal_sidecar(env, tmp_path):
    """A leftover -wal without its DB would be replayed into the restored database the moment
    SQLite opens it -- silent corruption, refused up front."""
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    migrate.pack(dest, include_weights=False)

    use(new)
    _file(new, "backyard.db-wal", b"stale")
    with pytest.raises(SystemExit, match="-wal"):
        migrate.restore(dest)
    assert not (new / "backyard.db").exists()


def test_restore_refuses_a_source_inside_the_project(env, tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    use = env
    use(old)
    build_old_rig(old)
    use(new)
    inside = new / "bundle"
    inside.mkdir()
    with pytest.raises(SystemExit, match="inside the project"):
        migrate.restore(inside)


def test_restore_refuses_a_folder_that_is_not_a_backup(env, tmp_path):
    use = env
    use(tmp_path / "new")
    not_a_backup = tmp_path / "random"
    not_a_backup.mkdir()
    with pytest.raises(SystemExit, match="not a backup/pack folder"):
        migrate.restore(not_a_backup)


# --- the database gate -------------------------------------------------------------------

def test_a_corrupt_newest_snapshot_falls_back_to_the_good_one(env, tmp_path, caplog):
    """The 08-19 crash shape: the newest snapshot is garbage. Restore must say so and install
    the next-oldest snapshot that passes quick_check -- stale beats corrupt."""
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    migrate.pack(dest, include_weights=False)
    bad = dest / "snapshots" / "backyard-db-9999-01-01.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("backyard.db", b"this is not a database")

    use(new)
    with caplog.at_level(logging.WARNING, logger="migrate"):
        assert migrate.restore(dest) == 0

    assert "FAILS quick_check" in " ".join(r.getMessage() for r in caplog.records)
    conn = sqlite3.connect(f"file:{(new / 'backyard.db').as_posix()}?mode=ro", uri=True)
    try:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    finally:
        conn.close()


def test_all_snapshots_corrupt_installs_nothing(env, tmp_path):
    """A restore that cannot trust any database must not half-restore around the hole."""
    new, dest = tmp_path / "new", tmp_path / "bundle"
    use = env
    use(new)
    (dest / "snapshots").mkdir(parents=True)
    with zipfile.ZipFile(dest / "snapshots" / "backyard-db-2026-01-01.zip", "w") as zf:
        zf.writestr("backyard.db", b"garbage")
    (dest / "clips").mkdir()
    with zipfile.ZipFile(dest / "clips" / "clips-2026-01-01.zip", "w") as zf:
        zf.writestr("clips/2026-01-01/x.mp4", b"video")

    with pytest.raises(SystemExit, match="integrity check"):
        migrate.restore(dest)
    assert not (new / "backyard.db").exists()
    assert not (new / "clips").exists()


# --- never overwrite, therefore re-runnable ----------------------------------------------

def test_restore_keeps_what_the_new_machine_already_wrote(env, tmp_path, capsys):
    """config_local.py hand-written on the new machine survives; the old machine's copy stays
    in the meta zip. Same rule for any file: existing wins, loudly for the config."""
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    migrate.pack(dest, include_weights=False)

    use(new)
    _file(new, "config_local.py", b"# written fresh on THIS machine\n")
    _file(new, CROP_A, b"already-here")
    assert migrate.restore(dest) == 0

    assert (new / "config_local.py").read_bytes() == b"# written fresh on THIS machine\n"
    assert (new / CROP_A).read_bytes() == b"already-here"
    assert "keeping this machine's own config_local.py" in capsys.readouterr().out


def test_interrupted_restore_is_finished_by_rerunning(env, tmp_path):
    """The database lands last, so 'crashed mid-restore' == 'files partly there, no DB' -- and
    that state passes every preflight and simply continues, skipping what already landed."""
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    migrate.pack(dest, include_weights=False)

    use(new)
    # The crash state: some media made it, plus a torn half-file at a tmp name; no database.
    _file(new, CROP_A, b"crop-a")
    torn = _file(new, CROP_B + migrate.TMP_SUFFIX, b"half a")
    assert migrate.restore(dest) == 0

    assert not torn.exists(), "a torn tmp from the crashed run must be swept, not kept"
    assert (new / CROP_B).read_bytes() == b"crop-b"
    assert (new / "backyard.db").is_file()


def test_dry_run_restores_nothing(env, tmp_path, capsys):
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    migrate.pack(dest, include_weights=False)

    use(new)
    assert migrate.restore(dest, dry_run=True) == 0

    assert "Would restore" in capsys.readouterr().out
    assert not (new / "backyard.db").exists()
    assert not (new / "crops").exists()
    assert not (new / "clips").exists()


def test_a_member_that_escapes_the_project_is_refused(env, tmp_path, caplog):
    """Zip members name where they land; one aimed outside the project is an attack or a bug,
    and either way it is skipped -- the rest of the archive still restores."""
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    migrate.pack(dest, include_weights=False)
    with zipfile.ZipFile(dest / "crops" / "crops-2026-01-01.zip", "w") as zf:
        zf.writestr("../escaped.txt", b"outside the project")

    use(new)
    with caplog.at_level(logging.WARNING, logger="migrate"):
        assert migrate.restore(dest) == 0

    assert not (tmp_path / "escaped.txt").exists()
    assert not (new / "escaped.txt").exists()
    assert "escapes the project" in " ".join(r.getMessage() for r in caplog.records)
    assert (new / CROP_A).is_file()


# --- weights routing ---------------------------------------------------------------------

def test_weights_mirror_restores_to_project_and_hub(env, tmp_path):
    """The mirror holds two families: weights/* belongs in the project, models--* belongs in
    the Hugging Face hub cache -- the same split its own zip comment documents."""
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    migrate.pack(dest, include_weights=False)
    with zipfile.ZipFile(dest / "snapshots" / "weights-archive.zip", "w") as zf:
        zf.writestr("weights/MDV6-yolov10x.pt", b"detector")
        zf.writestr("models--BVRA--MegaDescriptor-L-384/snapshots/abc/model.safetensors",
                    b"embedder")

    use(new)
    assert migrate.restore(dest) == 0

    assert (new / "weights/MDV6-yolov10x.pt").read_bytes() == b"detector"
    hub = tmp_path / "hf-hub"
    assert (hub / "models--BVRA--MegaDescriptor-L-384/snapshots/abc/model.safetensors"
            ).read_bytes() == b"embedder"


def test_no_weights_skips_the_mirror(env, tmp_path):
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    migrate.pack(dest, include_weights=False)
    with zipfile.ZipFile(dest / "snapshots" / "weights-archive.zip", "w") as zf:
        zf.writestr("weights/MDV6-yolov10x.pt", b"detector")

    use(new)
    assert migrate.restore(dest, include_weights=False) == 0
    assert not (new / "weights").exists()


# --- pack --------------------------------------------------------------------------------

def test_pack_includes_today_and_reruns_top_up(env, tmp_path):
    """The weekly run leaves today's folder for tomorrow; a move cannot. And because the
    archives are append-only, the second pack (rig stopped) only adds the delta."""
    old, dest = tmp_path / "old", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    _file(old, f"clips/glass_door_cam/{D0}/today.mp4", b"today")
    assert migrate.pack(dest, include_weights=False) == 0

    today_zip = dest / "clips" / f"clips-glass_door_cam-{D0}.zip"
    with zipfile.ZipFile(today_zip) as zf:
        assert f"clips/glass_door_cam/{D0}/today.mp4" in zf.namelist()

    _file(old, f"clips/glass_door_cam/{D0}/later.mp4", b"later")
    assert migrate.pack(dest, include_weights=False) == 0
    with zipfile.ZipFile(today_zip) as zf:
        names = zf.namelist()
        assert f"clips/glass_door_cam/{D0}/later.mp4" in names
        assert f"clips/glass_door_cam/{D0}/today.mp4" in names


def test_pack_refuses_a_machine_with_no_rig(env, tmp_path):
    use = env
    use(tmp_path / "empty")
    with pytest.raises(SystemExit, match="no rig here"):
        migrate.pack(tmp_path / "bundle", include_weights=False)


def test_pack_refuses_to_pack_into_itself(env, tmp_path):
    old = tmp_path / "old"
    use = env
    use(old)
    build_old_rig(old)
    with pytest.raises(SystemExit, match="into itself"):
        migrate.pack(old / "bundle", include_weights=False)


def test_pack_leaves_a_file_still_being_written(env, tmp_path, monkeypatch):
    """The recorder writes mp4s in place at their final name and the archives diff by name, so
    a half-written file sealed now would block the finished one forever. Anything younger than
    the settle window waits for the next pass; the next pass merges it in."""
    old, dest = tmp_path / "old", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    recording = _file(old, f"clips/glass_door_cam/{D0}/recording.mp4", b"half")
    settled = _file(old, f"clips/glass_door_cam/{D0}/done.mp4", b"whole")
    hour_ago = time.time() - 3600
    os.utime(settled, (hour_ago, hour_ago))
    for rel in (LIVE_CLIP, PRUNED_CLIP, CROP_A, CROP_B):
        os.utime(old / rel, (hour_ago, hour_ago))
    monkeypatch.setattr(migrate, "PACK_SETTLE_S", 60.0)

    assert migrate.pack(dest, include_weights=False) == 0
    today_zip = dest / "clips" / f"clips-glass_door_cam-{D0}.zip"
    with zipfile.ZipFile(today_zip) as zf:
        assert f"clips/glass_door_cam/{D0}/done.mp4" in zf.namelist()
        assert f"clips/glass_door_cam/{D0}/recording.mp4" not in zf.namelist()

    # The rig stops, the file settles, the final pack picks it up -- the two-pass flow.
    os.utime(recording, (hour_ago, hour_ago))
    assert migrate.pack(dest, include_weights=False) == 0
    with zipfile.ZipFile(today_zip) as zf:
        assert f"clips/glass_door_cam/{D0}/recording.mp4" in zf.namelist()


# --- the backup.py meta additions --------------------------------------------------------

def test_meta_snapshot_carries_references_and_ledgers(env, tmp_path):
    """The two holes migration surfaced in the weekly backup itself: the hand-shot reference
    batches (irreplaceable) and the DB sidecar ledgers (a restore without the import ledger
    would duplicate the next SD-card import). Both ride in the meta zip now."""
    old, dest = tmp_path / "old", tmp_path / "dest"
    use = env
    use(old)
    build_old_rig(old)
    _file(old, "backyard.db.static-dropped-glass_door_cam.txt", b"crops/x.jpg\n")

    backup.snapshot_meta(dest / "snapshots", date.today(), False)

    meta = dest / "snapshots" / f"meta-{D0}.zip"
    with zipfile.ZipFile(meta) as zf:
        names = set(zf.namelist())
    assert "reference/Alpha/portrait.jpg" in names
    assert "reference_crops/Alpha/portrait_crop.jpg" in names
    assert "backyard.db.imported-trail_cam_sd.txt" in names
    assert "backyard.db.static-dropped-glass_door_cam.txt" in names
    assert "config_local.py" in names


def test_meta_item_outside_the_project_warns_and_archives_the_rest(env, tmp_path, caplog):
    """config_local.py may repoint reference_dir outside the project (refcam.py suggests it for
    shared photo folders). Such an item has no honest project-relative arcname: it is skipped
    with a warning, and everything else still lands."""
    old, dest = tmp_path / "old", tmp_path / "dest"
    use = env
    cfg = use(old)
    build_old_rig(old)
    elsewhere = tmp_path / "photos-on-another-drive"
    (elsewhere / "Alpha").mkdir(parents=True)
    (elsewhere / "Alpha" / "p.jpg").write_bytes(b"x")
    cfg.reference_dir = elsewhere

    with caplog.at_level(logging.WARNING, logger="backup"):
        backup.snapshot_meta(dest / "snapshots", date.today(), False)

    assert "outside the project root" in " ".join(r.getMessage() for r in caplog.records)
    with zipfile.ZipFile(dest / "snapshots" / f"meta-{D0}.zip") as zf:
        names = set(zf.namelist())
    assert "config_local.py" in names
    assert "reference_crops/Alpha/portrait_crop.jpg" in names   # the inside item still lands
    assert not any("photos-on-another-drive" in n for n in names)


# --- restoring where a rig already lives ---------------------------------------------------
# "Replace" never deletes: the previous rig moves WHOLE into replaced-rig-<stamp>/ inside the
# project, and the optional safety pack is a portable copy on top of that. What these pin, in
# order of expense: declining changes nothing; nothing proceeds non-interactively without the
# explicit flags; the move carries the previous rig's data byte-for-byte; the safety pack is a
# pack OF THE PREVIOUS rig (not the bundle's); and the machine's own config never moves.

def _existing_rig(new: Path) -> None:
    """A small rig already living on the 'new' machine: one detection, its crop, a reference --
    distinct bytes from build_old_rig's so the two rigs can never be confused in asserts."""
    _file(new, f"crops/{D0}/mine.jpg", b"the-new-machines-own-crop")
    _add_detection(new, f"crops/{D0}/mine.jpg", f"{D0}T07:00:00")
    _file(new, "reference/Beta/portrait.jpg", b"beta")
    _file(new, "config_local.py", b"# this machine's own config\n")


def test_replace_via_flags_moves_the_previous_rig_aside(env, tmp_path, capsys):
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    migrate.pack(dest, include_weights=False)

    use(new)
    _existing_rig(new)
    assert migrate.restore(dest, replace_existing=True, backup_first=False) == 0

    replaced = list(new.glob("replaced-rig-*"))
    assert len(replaced) == 1
    # The previous rig arrived whole -- db and files, byte for byte.
    assert (replaced[0] / "crops" / D0 / "mine.jpg").read_bytes() == b"the-new-machines-own-crop"
    assert (replaced[0] / "reference/Beta/portrait.jpg").read_bytes() == b"beta"
    prev = sqlite3.connect(f"file:{(replaced[0] / 'backyard.db').as_posix()}?mode=ro", uri=True)
    try:
        assert prev.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 1
    finally:
        prev.close()
    # The restored rig is the bundle's, on a clean slate.
    now = sqlite3.connect(f"file:{(new / 'backyard.db').as_posix()}?mode=ro", uri=True)
    try:
        assert now.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 2
    finally:
        now.close()
    assert not (new / "crops" / D0 / "mine.jpg").exists()
    # The machine's own config is machine state, not rig state: it stays, and it wins.
    assert (new / "config_local.py").read_bytes() == b"# this machine's own config\n"
    assert "kept whole at" in capsys.readouterr().out


def test_replace_with_backup_first_packs_the_previous_rig(env, tmp_path):
    """The safety pack must hold the PREVIOUS rig -- the one about to be replaced -- in the
    standard format, so `migrate.py restore` can bring it back like any other bundle."""
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    migrate.pack(dest, include_weights=False)

    use(new)
    _existing_rig(new)
    prevpack = tmp_path / "previous-rig-pack"
    assert migrate.restore(dest, replace_existing=True, backup_first=True,
                           backup_dest=prevpack) == 0

    snaps = sorted((prevpack / "snapshots").glob("backyard-db-*.zip"))
    assert snaps, "the safety pack wrote no database snapshot"
    with zipfile.ZipFile(snaps[-1]) as zf:
        (prevpack / "check.db").write_bytes(zf.read(zf.namelist()[0]))
    packed = sqlite3.connect(f"file:{(prevpack / 'check.db').as_posix()}?mode=ro", uri=True)
    try:
        assert packed.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 1
    finally:
        packed.close()
    assert (new / "backyard.db").is_file()
    assert list(new.glob("replaced-rig-*")), "the move-aside still happens with a backup"


def test_replace_prompt_defaults_to_no_and_declining_changes_nothing(env, tmp_path, monkeypatch):
    """A bare Enter on the replace question is a NO -- the destructive answer is never the
    default -- and a declined replace leaves every byte where it was."""
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    migrate.pack(dest, include_weights=False)

    use(new)
    _existing_rig(new)
    before = (new / "backyard.db").read_bytes()
    monkeypatch.setattr(migrate, "_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    with pytest.raises(SystemExit, match="Left everything"):
        migrate.restore(dest)

    assert (new / "backyard.db").read_bytes() == before
    assert not list(new.glob("replaced-rig-*"))
    assert (new / "crops" / D0 / "mine.jpg").exists()


def test_replace_prompt_yes_then_no_backup_restores(env, tmp_path, monkeypatch):
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    migrate.pack(dest, include_weights=False)

    use(new)
    _existing_rig(new)
    monkeypatch.setattr(migrate, "_interactive", lambda: True)
    answers = iter(["y", "n"])                      # replace? yes; pack a backup first? no
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert migrate.restore(dest) == 0

    assert list(new.glob("replaced-rig-*"))
    now = sqlite3.connect(f"file:{(new / 'backyard.db').as_posix()}?mode=ro", uri=True)
    try:
        assert now.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 2
    finally:
        now.close()


def test_replace_without_a_terminal_requires_explicit_answers(env, tmp_path, monkeypatch):
    """No TTY and no flags: the old hard refusal. --replace alone is still not enough -- what
    happens to the previous rig's data must be said (--backup-to or --no-backup)."""
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    migrate.pack(dest, include_weights=False)

    use(new)
    _existing_rig(new)
    monkeypatch.setattr(migrate, "_interactive", lambda: False)
    with pytest.raises(SystemExit, match="already a rig here"):
        migrate.restore(dest)
    with pytest.raises(SystemExit, match="--no-backup"):
        migrate.restore(dest, replace_existing=True)
    assert not list(new.glob("replaced-rig-*"))
    assert (new / "crops" / D0 / "mine.jpg").exists()


def test_replace_aborts_whole_if_the_safety_pack_fails(env, tmp_path, monkeypatch):
    """A failed safety pack must stop the replace cold: the point of packing first is that the
    previous rig is safe BEFORE anything moves."""
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    migrate.pack(dest, include_weights=False)

    use(new)
    _existing_rig(new)
    monkeypatch.setattr(migrate, "pack", lambda *a, **k: 1)
    with pytest.raises(SystemExit, match="safety pack FAILED"):
        migrate.restore(dest, replace_existing=True, backup_first=True,
                        backup_dest=tmp_path / "wontmatter")
    assert not list(new.glob("replaced-rig-*"))
    assert (new / "crops" / D0 / "mine.jpg").exists()


def test_dry_run_on_an_existing_rig_changes_nothing_and_says_why(env, tmp_path, capsys):
    old, new, dest = tmp_path / "old", tmp_path / "new", tmp_path / "bundle"
    use = env
    use(old)
    build_old_rig(old)
    migrate.pack(dest, include_weights=False)

    use(new)
    _existing_rig(new)
    before = (new / "backyard.db").read_bytes()
    assert migrate.restore(dest, dry_run=True) == 0

    assert (new / "backyard.db").read_bytes() == before
    assert not list(new.glob("replaced-rig-*"))
    assert "already a rig here" in capsys.readouterr().out
