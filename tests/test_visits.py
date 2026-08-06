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


def _det(conn, *, minutes, confidence=0.5, species=None, species_confidence=None,
         individual_id=None, source=db.SOURCE_GLASS_DOOR_CAM, detection_class="animal"):
    det_id = db.insert_detection(
        conn, timestamp=_ts(minutes), source=source, detection_class=detection_class,
        confidence=confidence, bbox=(0, 0, 10, 10), frame_w=100, frame_h=100,
        crop_path=f"crops/{minutes}.jpg", species=species, individual_id=individual_id,
    )
    if species_confidence is not None:
        conn.execute("UPDATE detections SET species_confidence = ? WHERE id = ?",
                     (float(species_confidence), det_id))
        conn.commit()
    return det_id


def _bulk_dets(conn, *, n, start_minutes, step_minutes, species, species_confidence,
               confidence=0.5, source=db.SOURCE_GLASS_DOOR_CAM):
    """Many crops at once (the 800-vs-100 vote case) -- one executemany instead of 900 commits."""
    conn.executemany(
        """INSERT INTO detections (timestamp, source, detection_class, confidence,
                                   bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_w, frame_h,
                                   crop_path, species, species_confidence)
           VALUES (?, ?, 'animal', ?, 0, 0, 10, 10, 100, 100, ?, ?, ?)""",
        [(_ts(start_minutes + i * step_minutes), source, confidence,
          f"crops/bulk-{species}-{i}.jpg", species, species_confidence) for i in range(n)])
    conn.commit()


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
    assert summary["visits"] == 0
    assert summary["detections"] == 0
    assert summary["sighting_conflicts"] == 0
    assert _visit_rows(conn) == []


# --- refresh (the shared best-effort rebuild every pipeline step calls) -------------------

def test_refresh_folds_new_species_into_existing_visits(conn):
    """THE trail-cam sequence (2026-07-22): visits built while species were still NULL must pick
    up their dominant label on the next refresh -- classify.py refreshes after labeling for
    exactly this."""
    a = _det(conn, minutes=0)
    b = _det(conn, minutes=1)
    assert visits.refresh(conn, gap_minutes=5) is True
    assert _visit_rows(conn)[0]["species"] is None       # built before classification

    for det_id in (a, b):
        db.set_species(conn, det_id, "raccoon", 0.9)
    conn.commit()
    assert visits.refresh(conn, gap_minutes=5) is True
    assert _visit_rows(conn)[0]["species"] == "raccoon"  # the re-refresh folds the labels in


def test_refresh_restores_row_factory(conn):
    """Callers keep using their connection afterwards (classify.py unpacks rows positionally), so
    refresh must not leave sqlite3.Row set on a connection that didn't have it."""
    conn.row_factory = None
    _det(conn, minutes=0)
    assert visits.refresh(conn, gap_minutes=5) is True
    assert conn.row_factory is None


def test_refresh_never_raises(db_path):
    """The best-effort contract: a failed refresh reports False, never an exception -- it must not
    take down the import/labeling/shutdown that already succeeded."""
    c = db.connect(db_path)
    c.close()
    assert visits.refresh(c, gap_minutes=5) is False


# --- confidence-weighted species vote (phase C5) -----------------------------------------
#
# The measured failure: a static artifact firing 800 low-confidence frames outvotes a real
# animal's 100 high-confidence ones, and the result is committed silently. Species GATES re-ID, so
# a raccoon relabelled to opossum can never be matched to Stan again -- hence the vote ships OFF
# and there is a read-only report.

def _artifact_vs_animal(conn):
    """800 crops of a low-confidence static artifact + 100 high-confidence raccoon crops, all
    inside one visit (0.005 min = 0.3 s apart, far under any sane gap)."""
    _bulk_dets(conn, n=800, start_minutes=0.0, step_minutes=0.005,
               species="brown rat", species_confidence=0.31)
    _bulk_dets(conn, n=100, start_minutes=0.0025, step_minutes=0.005,
               species="raccoon", species_confidence=0.95)


def test_unweighted_vote_lets_volume_beat_certainty(conn):
    """The shipped default is unchanged: 800 crops at 0.31 still outvote 100 at 0.95. This test
    exists so the regression is visible, not so the behaviour is endorsed."""
    _artifact_vs_animal(conn)
    visits.build_visits(conn, gap_minutes=5, verbose=False, weighted_species=False)
    rows = _visit_rows(conn)
    assert len(rows) == 1
    assert rows[0]["species"] == "brown rat"


def test_weighted_vote_flips_the_artifact_case_to_the_animal(conn):
    """With the weighted vote on, only crops clearing the confidence bar vote, so the real animal
    wins its own visit."""
    _artifact_vs_animal(conn)
    visits.build_visits(conn, gap_minutes=5, verbose=False, weighted_species=True,
                        species_min_confidence=0.8)
    rows = _visit_rows(conn)
    assert len(rows) == 1
    assert rows[0]["species"] == "raccoon"
    assert rows[0]["detection_count"] == 900          # the artifact crops are still IN the visit


def test_weighted_vote_falls_back_when_no_crop_clears_the_bar(conn):
    """A visit the classifier was unsure about is still a visit: if nothing clears the bar the
    whole visit votes ungated rather than losing its species."""
    _det(conn, minutes=0, species="raccoon", species_confidence=0.4)
    _det(conn, minutes=1, species="raccoon", species_confidence=0.35)
    _det(conn, minutes=2, species="Virginia opossum", species_confidence=0.3)
    visits.build_visits(conn, gap_minutes=5, verbose=False, weighted_species=True,
                        species_min_confidence=0.8)
    assert _visit_rows(conn)[0]["species"] == "raccoon"


def test_species_margin_recorded_and_bounded(conn):
    """The margin surfaces near-ties instead of silently committing them."""
    _det(conn, minutes=0, species="raccoon", species_confidence=0.9)
    _det(conn, minutes=1, species="raccoon", species_confidence=0.9)
    visits.build_visits(conn, gap_minutes=5, verbose=False)
    assert _visit_rows(conn)[0]["species_margin"] == 1.0        # unanimous

    _det(conn, minutes=2, species="Virginia opossum", species_confidence=0.9)
    _det(conn, minutes=3, species="Virginia opossum", species_confidence=0.9)
    visits.build_visits(conn, gap_minutes=5, verbose=False)
    assert _visit_rows(conn)[0]["species_margin"] == 0.0        # a 2-2 coin flip, now visible


def test_species_margin_null_when_nothing_is_classified(conn):
    _det(conn, minutes=0)
    visits.build_visits(conn, gap_minutes=5, verbose=False)
    row = _visit_rows(conn)[0]
    assert row["species"] is None and row["species_margin"] is None


def test_unscored_species_label_weighs_full(conn):
    """A crop with a species but no species_confidence is a legacy/hand-set label. Treating it as
    low-confidence would be the dangerous direction, so it weighs 1.0."""
    _det(conn, minutes=0, species="raccoon")                       # species_confidence NULL
    _det(conn, minutes=1, species="Virginia opossum", species_confidence=0.85)
    _det(conn, minutes=2, species="raccoon")
    visits.build_visits(conn, gap_minutes=5, verbose=False, weighted_species=True,
                        species_min_confidence=0.8)
    assert _visit_rows(conn)[0]["species"] == "raccoon"


# --- the dry run: report BEFORE anything is written --------------------------------------

def _snapshot(conn):
    return (conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0],
            [tuple(r) for r in conn.execute(
                "SELECT id, species, species_confidence, individual_id, visit_id "
                "FROM detections ORDER BY id")])


def test_species_vote_report_writes_nothing(conn):
    """THE guarantee: a mass species relabel is not reversible in practice, so the report must be
    inspectable without committing to it. Not one row may change."""
    _artifact_vs_animal(conn)
    visits.build_visits(conn, gap_minutes=5, verbose=False)      # a normal, unweighted rebuild
    before = _snapshot(conn)

    report = visits.species_vote_report(conn, gap_minutes=5, species_min_confidence=0.8)

    assert _snapshot(conn) == before
    assert conn.execute("SELECT species FROM visits").fetchone()["species"] == "brown rat"
    assert len(report["flips"]) == 1                              # ... and it still SAW the flip


def test_species_vote_report_describes_the_flip(conn):
    _artifact_vs_animal(conn)
    report = visits.species_vote_report(conn, gap_minutes=5, species_min_confidence=0.8)
    assert report["visits"] == 1 and report["labelled"] == 1
    flip = report["flips"][0]
    assert (flip["from"], flip["to"]) == ("brown rat", "raccoon")
    assert flip["detections"] == 900
    assert (flip["n_above"], flip["n_below"]) == (100, 800)
    assert report["into"] == {"raccoon": 1}
    assert report["out_of"] == {"brown rat": 1}
    # The gate alone already moves it -- 100 crops clear 0.8 and 800 don't -- so both readings of
    # "a >=0.8-confidence-only vote" agree here. They do NOT agree everywhere (108 vs 158 on the
    # live DB), which is why both are reported.
    assert report["flips_gate_only"] == 1
    assert report["near_ties"] == 0


def test_species_vote_report_is_empty_when_nothing_moves(conn):
    _det(conn, minutes=0, species="raccoon", species_confidence=0.95)
    _det(conn, minutes=1, species="raccoon", species_confidence=0.92)
    report = visits.species_vote_report(conn, gap_minutes=5, species_min_confidence=0.8)
    assert report["flips"] == []


def test_species_vote_report_runs_on_a_read_only_connection(conn, db_path):
    """The CLI hands it a read-only handle, so the 'writes nothing' promise is enforced by SQLite
    rather than by review."""
    _artifact_vs_animal(conn)
    conn.commit()
    ro = db.connect_readonly(db_path)
    try:
        report = visits.species_vote_report(ro, gap_minutes=5, species_min_confidence=0.8)
    finally:
        ro.close()
    assert len(report["flips"]) == 1


# --- conflicting live sightings raise the multi-animal flag on the visit ------------------

def test_conflicting_sightings_flag_the_overlapping_visit(conn):
    """Three single-name logs over one span (the real 25/26/27 shape) -- the visit they overlap is
    flagged, so a contested span is never mistaken for clean single-animal ground truth."""
    _det(conn, minutes=0, species="raccoon", species_confidence=0.9)
    _det(conn, minutes=1, species="raccoon", species_confidence=0.9)
    for name in ("Clippy", "Clippy Friend", "Stan"):
        db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=[name],
                                span_start=_ts(0), span_end=_ts(1))

    summary = visits.build_visits(conn, gap_minutes=5, verbose=False)
    assert summary["sighting_conflicts"] == 1
    assert _visit_rows(conn)[0]["sighting_conflict"] == 1


def test_agreeing_sightings_leave_the_visit_clean(conn):
    """A re-log of the SAME name is a duplicate, not a disagreement -- 0, not NULL, so a reader can
    tell 'checked and clean' from 'never computed'."""
    _det(conn, minutes=0, species="raccoon", species_confidence=0.9)
    _det(conn, minutes=1, species="raccoon", species_confidence=0.9)
    for _ in range(2):
        db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Stan"],
                                span_start=_ts(0), span_end=_ts(1))
    visits.build_visits(conn, gap_minutes=5, verbose=False)
    assert _visit_rows(conn)[0]["sighting_conflict"] == 0


def test_conflict_flag_does_not_leak_to_other_visits_or_sources(conn):
    """A conflict is bounded by its span AND its camera: the trail cam's visit at the same instant
    is untouched, and so is a later glass-door visit."""
    _det(conn, minutes=0, species="raccoon", species_confidence=0.9)
    _det(conn, minutes=1, species="raccoon", species_confidence=0.9)
    _det(conn, minutes=0.5, species="raccoon", species_confidence=0.9,
         source=db.SOURCE_TRAIL_CAM_SD)
    _det(conn, minutes=90, species="raccoon", species_confidence=0.9)   # a separate evening visit
    for name in ("Notch", "Elliot"):
        db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=[name],
                                span_start=_ts(0), span_end=_ts(1))

    visits.build_visits(conn, gap_minutes=5, verbose=False)
    flagged = {(r["source"], r["started_at"], r["sighting_conflict"]) for r in _visit_rows(conn)}
    assert (db.SOURCE_GLASS_DOOR_CAM, _ts(0), 1) in flagged
    assert (db.SOURCE_TRAIL_CAM_SD, _ts(0.5), 0) in flagged
    assert (db.SOURCE_GLASS_DOOR_CAM, _ts(90), 0) in flagged
