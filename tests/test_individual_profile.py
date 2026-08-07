"""
Tests for the individual profile (stats.individual_profile + the crops_page individual filter +
load_clips include_pruned) and the archived-clip plumbing in web.py (zip naming, restore out of
a backup zip, and the API response cache). Pure DB + tempfile logic; no camera / GPU / model.

The contract under test, in one line: "look up an animal and see every visit, photo and clip --
and if the disk budget pruned a clip, the profile still offers the backup archive's copy."
"""
from __future__ import annotations

import time
import zipfile
from dataclasses import replace
from datetime import datetime, timedelta

import config
import db
import stats
import visits
import web


def _cfg(db_path):
    return replace(config.CONFIG, db_path=db_path)


_T0 = datetime(2026, 8, 1, 21, 0, 0)


def _iso(minutes: float) -> str:
    return (_T0 + timedelta(minutes=minutes)).isoformat()


def _det(conn, *, minutes=0.0, iid=None, species="raccoon", source="glass_door_cam",
         crop="crops/2026-08-01/x.jpg", quality=0.5):
    return db.insert_detection(
        conn, timestamp=_iso(minutes), source=source, detection_class="animal",
        confidence=0.9, bbox=(0, 0, 50, 50), frame_w=100, frame_h=100,
        crop_path=crop, species=species, individual_id=iid, crop_quality=quality)


def _clip(conn, *, start_min, end_min, path, source="glass_door_cam", pruned=False):
    cid = db.insert_clip(conn, source=source, clip_path=path, started_at=_iso(start_min),
                         ended_at=_iso(end_min), fps=10.0, width=640, height=360,
                         frame_count=100, detection_count=3, max_confidence=0.8)
    if pruned:
        conn.execute("UPDATE clips SET pruned_at = ? WHERE id = ?", (_iso(end_min + 1), cid))
        conn.commit()
    return cid


def _two_visit_corpus(conn):
    """Two visits 40 min apart (gap > default 10): Stan solo, then Stan + Pedro together.
    One live clip overlaps visit 1; one PRUNED clip overlaps visit 2."""
    for m in (0, 1, 2):
        _det(conn, minutes=m, iid="Stan")
    for m in (60, 61):
        _det(conn, minutes=m, iid="Stan", crop="crops/2026-08-01/s2.jpg", quality=0.9)
    for m in (60.5, 61.5):
        _det(conn, minutes=m, iid="Pedro", crop="crops/2026-08-01/p.jpg", quality=0.4)
    _clip(conn, start_min=0, end_min=2, path="clips/glass_door_cam/2026-08-01/a.mp4")
    _clip(conn, start_min=60, end_min=62, path="clips/glass_door_cam/2026-08-01/b.mp4",
          pruned=True)
    visits.refresh(conn, 10.0)


# ---- the profile itself ------------------------------------------------------------

def test_profile_unknown_name_and_blank(conn, db_path):
    _det(conn, iid="Stan")
    assert stats.individual_profile(_cfg(db_path), "Nobody")["found"] is False
    assert stats.individual_profile(_cfg(db_path), "  ")["found"] is False


def test_profile_collects_visits_photos_and_companions(conn, db_path):
    _two_visit_corpus(conn)
    p = stats.individual_profile(_cfg(db_path), "Stan")
    assert p["found"] and p["n_crops"] == 5 and p["n_visits"] == 2
    assert p["unfiled"] == 0
    assert [c["name"] for c in p["companions"]] == ["Pedro"]
    # newest first; the shared visit lists both names
    assert p["visits"][0]["individuals"] == ["Stan", "Pedro"]
    assert p["visits"][1]["individuals"] == ["Stan"]
    # the visit dicts are shaped like visits_page's (the dashboard shares one card renderer)
    for k in ("start", "end", "source", "count", "minutes", "title", "classes",
              "individuals", "rep_crop", "clips"):
        assert k in p["visits"][0], k


def test_profile_rep_crop_prefers_the_named_animals_own_crop(conn, db_path):
    """In the shared visit Pedro's best crop must front PEDRO's profile even though Stan's
    higher-quality crop would win the visit-wide score."""
    _two_visit_corpus(conn)
    p = stats.individual_profile(_cfg(db_path), "Pedro")
    assert p["visits"][0]["rep_crop"] == "crops/2026-08-01/p.jpg"


def test_profile_flags_archived_clips_with_ids(conn, db_path):
    _two_visit_corpus(conn)
    p = stats.individual_profile(_cfg(db_path), "Stan")
    clips = [c for v in p["visits"] for c in v["clips"]]
    assert len(clips) == 2
    live = [c for c in clips if not c.get("archived")]
    arch = [c for c in clips if c.get("archived")]
    assert len(live) == 1 and len(arch) == 1
    assert "id" in arch[0] and isinstance(arch[0]["id"], int)
    assert "id" not in live[0]          # everyday payloads stay lean


def test_profile_counts_unfiled_crops(conn, db_path):
    _two_visit_corpus(conn)
    _det(conn, minutes=200, iid="Stan")          # after the ledger refresh -> visit_id NULL
    p = stats.individual_profile(_cfg(db_path), "Stan")
    assert p["unfiled"] == 1
    assert p["n_crops"] == 6                     # unfiled photos still count as photos


def test_load_clips_default_still_excludes_pruned(conn, db_path):
    _two_visit_corpus(conn)
    assert len(stats.load_clips(conn)) == 1
    assert len(stats.load_clips(conn, include_pruned=True)) == 2


def test_crops_page_individual_filter(conn, db_path):
    _two_visit_corpus(conn)
    out = stats.crops_page(_cfg(db_path), individual="Pedro")
    assert out["total"] == 2
    assert all("p.jpg" in c["crop_path"] for c in out["crops"])


# ---- archived-clip plumbing (web.py) ----------------------------------------------

def test_archive_zip_naming_both_layouts():
    assert web._archive_zip_for("clips/2026-06-15/x.mp4") == (
        "clips-2026-06-15.zip", "clips/2026-06-15/x.mp4")
    assert web._archive_zip_for("clips\\glass_door_cam\\2026-08-01\\b.mp4") == (
        "clips-glass_door_cam-2026-08-01.zip", "clips/glass_door_cam/2026-08-01/b.mp4")
    # reels are a cache (never archived), and junk must never resolve to a zip
    assert web._archive_zip_for("clips/reels/r.mp4") is None
    assert web._archive_zip_for("../../etc/passwd") is None
    assert web._archive_zip_for("") is None


def test_restore_archived_clip_round_trip(tmp_path):
    """A member stored the way backup.py stores it comes back out byte-identical, lands in the
    cache mirror, and a second call serves the cached copy without the zip."""
    member = "clips/glass_door_cam/2026-08-01/b.mp4"
    payload = b"\x00\x00\x00\x18ftypmp42-fake-bytes"
    dest = tmp_path / "backupdest"
    (dest / "clips").mkdir(parents=True)
    with zipfile.ZipFile(dest / "clips" / "clips-glass_door_cam-2026-08-01.zip", "w",
                         compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(member, payload)
    cache = tmp_path / "archive_cache"
    out = web._restore_archived_clip(dest, member, cache)
    assert out is not None and out.read_bytes() == payload
    assert out == (cache / member).resolve()
    (dest / "clips" / "clips-glass_door_cam-2026-08-01.zip").unlink()
    again = web._restore_archived_clip(dest, member, cache)   # zip gone -> cache still serves
    assert again == out


def test_restore_archived_clip_missing_is_none(tmp_path):
    assert web._restore_archived_clip(tmp_path, "clips/2026-01-01/x.mp4",
                                      tmp_path / "cache") is None
    assert web._restore_archived_clip(None, "clips/2026-01-01/x.mp4",
                                      tmp_path / "cache") is None


def test_prune_archive_cache_keeps_newest(tmp_path):
    for i in range(6):
        p = tmp_path / f"c{i}.mp4"
        p.write_bytes(b"x")
        t = time.time() - (10 - i)
        import os
        os.utime(p, (t, t))
    web._prune_archive_cache(tmp_path, keep=2)
    left = sorted(p.name for p in tmp_path.glob("*.mp4"))
    assert left == ["c4.mp4", "c5.mp4"]


# ---- the API response cache -------------------------------------------------------

def test_api_cache_serves_hits_and_clears(conn, db_path, tmp_path):
    cfg = _cfg(db_path)
    web.clear_api_cache()
    calls = []
    build = lambda: calls.append(1) or {"n": len(calls)}
    a = web._cached(cfg, "t1", build)
    b = web._cached(cfg, "t1", build)
    assert a is b and len(calls) == 1            # unchanged DB -> pure cache hit
    web.clear_api_cache()
    web._cached(cfg, "t1", build)
    assert len(calls) == 2                       # an explicit clear forces a rebuild


def test_api_cache_holds_down_rebuilds_while_db_churns(conn, db_path):
    """While capture writes every few seconds the signature always differs, but rebuilds must
    still be capped by the hold-down -- that was the whole point."""
    cfg = _cfg(db_path)
    web.clear_api_cache()
    calls = []
    build = lambda: calls.append(1) or {"n": len(calls)}
    web._cached(cfg, "t2", build, hold_s=60)
    _det(conn)                                    # DB signature changes
    out = web._cached(cfg, "t2", build, hold_s=60)
    assert len(calls) == 1 and out["n"] == 1      # inside the hold-down -> stale served
    out = web._cached(cfg, "t2", build, hold_s=0.0)
    assert len(calls) == 2 and out["n"] == 2      # hold expired + sig changed -> rebuilt
