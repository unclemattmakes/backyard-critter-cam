"""
Tests for visits.py -- VISIT-EVENT COLLAPSING.

build_visits(conn, gap_minutes) groups consecutive detections on the same source that are
< gap minutes apart into one visit row, stamps detections.visit_id, picks a dominant species and
the highest-confidence crop as the representative, and records the span (started..ended -> dwell).

We drive it with synthetic detections inserted via db.insert_detection at controlled timestamps,
so the gap boundary, dominant-label selection, and representative pick are all deterministic.
conn.row_factory is sqlite3.Row (the `conn` fixture sets it) -- build_visits reads columns by name.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import db
import visits

# A fixed local time WITH offset, matching the project's timestamp convention.
BASE = datetime(2026, 6, 7, 19, 0, 0, tzinfo=datetime.now().astimezone().tzinfo)


def _ts(minutes: float = 0.0) -> str:
    """ISO 8601 local timestamp `minutes` after BASE (what db stores in detections.timestamp)."""
    return (BASE + timedelta(minutes=minutes)).isoformat()


def _det(conn, *, minutes, confidence=0.5, species=None, individual_id=None,
         source=db.SOURCE_GLASS_DOOR_CAM, detection_class="animal"):
    return db.insert_detection(
        conn, timestamp=_ts(minutes), source=source, detection_class=detection_class,
        confidence=confidence, bbox=(0, 0, 10, 10), frame_w=100, frame_h=100,
        crop_path=f"crops/{minutes}.jpg", species=species, individual_id=individual_id,
    )


def _visit_rows(conn):
    return conn.execute("SELECT * FROM visits ORDER BY started_at").fetchall()


# --- gap grouping -----------------------------------------------------------------------

def test_grouping_splits_across_gap_boundary(conn):
    """Two detections within the gap collapse into one visit; a third beyond the gap starts a new
    one. With gap=5min: t=0 and t=4 are one visit; t=12 (8 min after t=4) is a second."""
    _det(conn, minutes=0)
    _det(conn, minutes=4)     # 4 min after the first -> same visit (< 5)
    _det(conn, minutes=12)    # 8 min after the previous -> new visit (> 5)

    summary = visits.build_visits(conn, gap_minutes=5, verbose=False)
    assert summary["visits"] == 2
    assert summary["detections"] == 3

    rows = _visit_rows(conn)
    assert [r["detection_count"] for r in rows] == [2, 1]


def test_gap_boundary_is_strict_greater_than(conn):
    """A gap EXACTLY equal to gap_minutes stays in the same visit (split is on `> gap`, not >=)."""
    _det(conn, minutes=0)
    _det(conn, minutes=5)   # exactly 5 min later -> NOT split (5 is not > 5)
    visits.build_visits(conn, gap_minutes=5, verbose=False)
    rows = _visit_rows(conn)
    assert len(rows) == 1
    assert rows[0]["detection_count"] == 2


def test_larger_gap_merges_more(conn):
    """Re-running with a wider gap collapses what was two visits into one (full rebuild each run)."""
    _det(conn, minutes=0)
    _det(conn, minutes=4)
    _det(conn, minutes=12)

    visits.build_visits(conn, gap_minutes=5, verbose=False)
    assert len(_visit_rows(conn)) == 2

    visits.build_visits(conn, gap_minutes=30, verbose=False)  # everything within 30 min
    rows = _visit_rows(conn)
    assert len(rows) == 1
    assert rows[0]["detection_count"] == 3


# --- visit_id stamping ------------------------------------------------------------------

def test_visit_id_stamped_on_member_detections(conn):
    a = _det(conn, minutes=0)
    b = _det(conn, minutes=2)
    c = _det(conn, minutes=20)   # separate visit
    visits.build_visits(conn, gap_minutes=5, verbose=False)

    va = conn.execute("SELECT visit_id FROM detections WHERE id=?", (a,)).fetchone()["visit_id"]
    vb = conn.execute("SELECT visit_id FROM detections WHERE id=?", (b,)).fetchone()["visit_id"]
    vc = conn.execute("SELECT visit_id FROM detections WHERE id=?", (c,)).fetchone()["visit_id"]
    assert va is not None and va == vb        # same visit
    assert vc is not None and vc != va        # different visit


def test_rebuild_clears_old_visits_first(conn):
    """build_visits does a from-scratch rebuild: a second run doesn't accumulate stale rows."""
    _det(conn, minutes=0)
    _det(conn, minutes=2)
    visits.build_visits(conn, gap_minutes=5, verbose=False)
    visits.build_visits(conn, gap_minutes=5, verbose=False)
    assert len(_visit_rows(conn)) == 1   # not 2


# --- dominant species -------------------------------------------------------------------

def test_dominant_species_selected(conn):
    """A visit's species is the most common label among its crops (2x raccoon beats 1x opossum)."""
    _det(conn, minutes=0, species="raccoon")
    _det(conn, minutes=1, species="raccoon")
    _det(conn, minutes=2, species="Virginia opossum")
    visits.build_visits(conn, gap_minutes=5, verbose=False)
    rows = _visit_rows(conn)
    assert len(rows) == 1
    assert rows[0]["species"] == "raccoon"


def test_dominant_individual_selected(conn):
    _det(conn, minutes=0, individual_id="Notch")
    _det(conn, minutes=1, individual_id="Notch")
    _det(conn, minutes=2, individual_id="Gimpy")
    visits.build_visits(conn, gap_minutes=5, verbose=False)
    rows = _visit_rows(conn)
    assert rows[0]["individual_id"] == "Notch"


def test_species_none_when_unclassified(conn):
    """No crop classified -> visit species stays NULL (not a crash, not 'animal')."""
    _det(conn, minutes=0)
    _det(conn, minutes=1)
    visits.build_visits(conn, gap_minutes=5, verbose=False)
    assert _visit_rows(conn)[0]["species"] is None


# --- representative = highest-confidence crop -------------------------------------------

def test_representative_is_highest_confidence_crop(conn):
    """representative_detection_id and max_confidence track the most readable (highest-conf) crop,
    regardless of position within the visit."""
    _det(conn, minutes=0, confidence=0.40)
    best = _det(conn, minutes=1, confidence=0.97)   # the readable one, in the middle
    _det(conn, minutes=2, confidence=0.55)
    visits.build_visits(conn, gap_minutes=5, verbose=False)
    row = _visit_rows(conn)[0]
    assert row["representative_detection_id"] == best
    assert row["max_confidence"] == 0.97


# --- dwell span -------------------------------------------------------------------------

def test_dwell_span_matches_first_and_last(conn):
    """started_at/ended_at are the first/last member timestamps -> dwell = their difference."""
    _det(conn, minutes=0)
    _det(conn, minutes=3)
    _det(conn, minutes=6)
    visits.build_visits(conn, gap_minutes=10, verbose=False)
    row = _visit_rows(conn)[0]
    started = datetime.fromisoformat(row["started_at"])
    ended = datetime.fromisoformat(row["ended_at"])
    assert started == BASE
    assert (ended - started) == timedelta(minutes=6)


# --- multi-source separation ------------------------------------------------------------

def test_sources_are_grouped_separately(conn):
    """Detections on different sources never share a visit even if they're close in time."""
    _det(conn, minutes=0, source=db.SOURCE_GLASS_DOOR_CAM)
    _det(conn, minutes=1, source=db.SOURCE_TRAIL_CAM_SD)
    visits.build_visits(conn, gap_minutes=30, verbose=False)
    rows = _visit_rows(conn)
    assert len(rows) == 2
    assert {r["source"] for r in rows} == {db.SOURCE_GLASS_DOOR_CAM, db.SOURCE_TRAIL_CAM_SD}


def test_empty_db_builds_no_visits(conn):
    summary = visits.build_visits(conn, gap_minutes=5, verbose=False)
    assert summary == {"visits": 0, "detections": 0}
    assert _visit_rows(conn) == []
