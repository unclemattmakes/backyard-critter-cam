"""
Smoke + sanity tests for stats.py -- the digest / overview engine, the most intricate read-side
logic in the project and (until now) untested. The review flagged it as the module most likely to
crash on a fresh / empty database, so these focus on the empty-DB paths plus a small populated
check. Pure DB logic; no GPU / camera / model.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import config
import db
import stats
import visits


def _cfg(db_path):
    return replace(config.CONFIG, db_path=db_path)


def _add_detection(conn, *, species="raccoon", cls="animal", conf=0.9):
    db.insert_detection(conn, timestamp=db.now_local_iso(), source="glass_door_cam",
                        detection_class=cls, confidence=conf, bbox=(0, 0, 10, 10),
                        frame_w=100, frame_h=100, crop_path="crops/x.jpg",
                        species=species, crop_quality=1.0)


# ---- empty database: must not crash (the review's main concern) --------------------
def test_compute_stats_empty_db(conn, db_path):
    s = stats.compute_stats(_cfg(db_path))
    assert s is None or s.get("total_crops", 0) == 0


def test_species_overview_empty_db(conn, db_path):
    o = stats.species_overview(_cfg(db_path))
    assert o is None or isinstance(o, dict)


def test_period_digest_empty_db(conn, db_path):
    d = stats.period_digest(_cfg(db_path))
    assert isinstance(d, dict)            # returns a dict (empty:true), never raises


# ---- small populated database ------------------------------------------------------
def test_compute_stats_counts_crops(conn, db_path):
    for _ in range(3):
        _add_detection(conn)
    s = stats.compute_stats(_cfg(db_path))
    assert s is not None and s["total_crops"] == 3


def test_period_digest_with_data_does_not_crash(conn, db_path):
    for _ in range(3):
        _add_detection(conn)
    visits.build_visits(conn, config.CONFIG.visit_gap_minutes, verbose=False)
    d = stats.period_digest(_cfg(db_path))
    assert isinstance(d, dict)


# ---- species_overview drops non-critter labels (the catalogue / "Rarely Seen" fix) -----
def test_species_overview_filters_non_critter(conn, db_path):
    _add_detection(conn, species="raccoon")
    _add_detection(conn, species="chair")          # a false-trigger human correction
    _add_detection(conn, species="not an animal")  # the clip-filter gate's label
    o = stats.species_overview(_cfg(db_path))
    names = {s["species"] for s in o["species"]}
    assert "raccoon" in names
    assert "chair" not in names and "not an animal" not in names


# ---- cast_rollcall: the named-cast last-seen / overdue roll -------------------------
def _add_named(conn, iid, *, species="raccoon", days_ago=0, conf=0.9):
    ts = (datetime.now().astimezone() - timedelta(days=days_ago)).isoformat()
    db.insert_detection(conn, timestamp=ts, source="glass_door_cam", detection_class="animal",
                        confidence=conf, bbox=(0, 0, 10, 10), frame_w=100, frame_h=100,
                        crop_path="crops/x.jpg", species=species, individual_id=iid, crop_quality=1.0)


def test_cast_rollcall_empty_db(conn, db_path):
    rc = stats.cast_rollcall(_cfg(db_path))
    assert isinstance(rc, dict) and rc.get("cast") == []


def test_cast_rollcall_excludes_placeholder_clusters(conn, db_path):
    _add_named(conn, "Notch")
    _add_named(conn, "raccoon_c01")               # reid auto-cluster -> not the named cast
    ids = [c["id"] for c in stats.cast_rollcall(_cfg(db_path))["cast"]]
    assert "Notch" in ids and "raccoon_c01" not in ids


def test_cast_rollcall_flags_overdue_regular(conn, db_path):
    for d in (12, 11, 10):                         # a regular (3 distinct days), last seen 10d ago
        _add_named(conn, "Gus", days_ago=d)
    _add_named(conn, "Solo", days_ago=12)          # a one-off, also long gone
    by = {c["id"]: c for c in stats.cast_rollcall(_cfg(db_path))["cast"]}
    assert by["Gus"]["overdue"] is True and by["Gus"]["regular"] is True
    assert by["Solo"]["overdue"] is False          # not a regular -> never "overdue"


def test_cast_rollcall_recent_regular_not_overdue(conn, db_path):
    for d in (3, 2, 1, 0):
        _add_named(conn, "Stan", days_ago=d)
    c = {x["id"]: x for x in stats.cast_rollcall(_cfg(db_path))["cast"]}["Stan"]
    assert c["regular"] is True and c["overdue"] is False and c["days_since"] == 0


# ---- review_queue: the prioritized "most likely mislabeled" pass --------------------
def test_review_queue_empty_db(conn, db_path):
    rq = stats.review_queue(_cfg(db_path))
    assert rq["crops"] == [] and rq["total"] == 0


def test_review_queue_flags_suspect_species_only(conn, db_path):
    _add_detection(conn, species="brown rat", conf=0.9)    # suspect label -> flagged
    _add_detection(conn, species="raccoon", conf=0.95)     # normal + confident -> not flagged
    flagged = {c["species"] for c in stats.review_queue(_cfg(db_path))["crops"]}
    assert "brown rat" in flagged and "raccoon" not in flagged


def test_review_queue_excludes_verified(conn, db_path):
    _add_detection(conn, species="brown rat", conf=0.9)
    conn.execute("UPDATE detections SET species_verified = 1")
    conn.commit()
    assert stats.review_queue(_cfg(db_path))["total"] == 0


def test_crops_page_clamps_negative_limit(conn, db_path):
    """A negative ?limit must not pass through to SQLite, where LIMIT -1 means UNBOUNDED and would
    dump the entire detections table. crops_page clamps limit to >=1 and offset to >=0 itself, so
    it's safe regardless of the caller (the dashboard's /api/crops takes the value from the URL)."""
    for _ in range(3):
        _add_detection(conn)
    page = stats.crops_page(_cfg(db_path), limit=-1, offset=-5)
    assert page["limit"] == 1 and page["offset"] == 0
    assert len(page["crops"]) <= 1


# ---- current_live_visit: the span the Live tab's "who's here now?" control names ----
def _at(conn, dt, species="raccoon"):
    db.insert_detection(conn, timestamp=dt.isoformat(), source="glass_door_cam",
                        detection_class="animal", confidence=0.9, bbox=(0, 0, 10, 10),
                        frame_w=100, frame_h=100, crop_path="crops/x.jpg",
                        species=species, crop_quality=1.0)


def test_current_live_visit_empty_db(conn, db_path):
    v = stats.current_live_visit(_cfg(db_path))
    assert v["count"] == 0


def test_current_live_visit_active_run(conn, db_path):
    now = datetime.now().astimezone()
    for s in (40, 25, 10, 1):                         # four detections in the last minute
        _at(conn, now - timedelta(seconds=s))
    conn.commit()
    v = stats.current_live_visit(_cfg(db_path))
    assert v["count"] == 4 and v["active"] is True
    assert v["species"] == {"raccoon": 4} and v["latest_age_s"] < 30


def test_current_live_visit_stops_at_the_gap(conn, db_path):
    """A gap >= visit_gap_minutes ends the span: only the most recent run is the 'current' visit."""
    now = datetime.now().astimezone()
    gap = config.CONFIG.visit_gap_minutes
    # Insert in capture order (oldest first), like the live rig -- so id order == time order, which
    # is what current_live_visit walks back over. The older run is separated by more than the gap.
    for m in (gap + 31, gap + 30):                    # an older run, must NOT be folded in
        _at(conn, now - timedelta(minutes=m))
    for s in (20, 5):                                 # the current run (last ~20s)
        _at(conn, now - timedelta(seconds=s))
    conn.commit()
    v = stats.current_live_visit(_cfg(db_path))
    assert v["count"] == 2 and v["active"] is True
