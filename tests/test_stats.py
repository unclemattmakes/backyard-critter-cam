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
