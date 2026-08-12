"""
Tests for db.py -- the stdlib-sqlite3 storage layer.

Covers: connect() builds the schema and runs _migrate(); the four real tables exist; migration
is idempotent (connect twice on the same file is a no-op); and insert/read round-trips for every
writer V1+phase-4 uses (insert_detection, insert_visit, insert_clip, set_species,
set_individual_bulk, clear_visits / assign_visit).

Plus the 2026-08-05 LABEL-INTEGRITY work (docs/identity-eval-2026-08-05.md phase 0 + C3), which
exists to protect the 21,110-detection human label set:
  * detections.labelled_at -- WHEN a label was applied, as opposed to when the animal was seen.
  * live_sightings supersede + cross-row conflict detection -- 48 rows stamp 3,758 crops, and ids
    25/26/27 logged three DIFFERENT names over the same two crops in 33 seconds.
  * individual_status (the ROSTER) -- the human's record that an animal has stopped visiting, and
    the day it was last here. Identity is free text on detections/visits, so this is a small side
    table; it governs nothing except what the auto tier is allowed to WRITE.

Plus the 2026-08-07 REFERENCE-IMAGE VETO storage (docs/refimg-design-2026-08-07.md section 7),
which is additive and, for now, entirely inert:
  * detections.suppressed_at/_by/suppress_ref_id/suppress_detail -- a furniture box is written
    exactly as before and FLAGGED. Nothing is dropped: an erased animal writes no row at all and is
    a silent permanent loss, while a wrongly flagged one is a row a human can clear.
  * reference_images -- "this view with nothing in it", kept (retired, never deleted) so any
    suppression can be replayed against the exact image months later.
  * view_epochs -- "the camera moved" as a recorded event, because a stale guard fails SILENTLY.

All DBs are throwaway tempfiles (the `conn` / `db_path` fixtures); the real backyard.db is untouched.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

import db


# --- schema -----------------------------------------------------------------------------

def _table_names(conn) -> set[str]:
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(conn, table) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_connect_creates_all_tables(conn):
    """connect() executescript(SCHEMA) -- the four real tables must all be present."""
    tables = _table_names(conn)
    for expected in ("detections", "detection_embeddings", "visits", "clips"):
        assert expected in tables, f"{expected} table missing after connect()"


def test_detections_has_migrated_columns(conn):
    """_migrate() adds the post-original columns; a freshly connected DB has them too (the SCHEMA
    already declares them, and re-adding is guarded), so the live writers can target them."""
    cols = _columns(conn, "detections")
    for c in ("species", "species_confidence", "species_verified", "species_source",
              "individual_id", "visit_id"):
        assert c in cols, f"detections.{c} missing"


def test_connect_is_idempotent(db_path):
    """Calling connect() twice on the same file must not raise (CREATE TABLE IF NOT EXISTS +
    _migrate's guarded ALTERs) and must not duplicate tables."""
    c1 = db.connect(db_path)
    tables_first = _table_names(c1)
    c1.close()

    c2 = db.connect(db_path)  # second open on the now-existing file
    tables_second = _table_names(c2)
    # A column that _migrate would ALTER-ADD must appear exactly once (no double-add crash above,
    # and no duplicate column).
    n_visit_id = sum(1 for r in c2.execute("PRAGMA table_info(detections)") if r[1] == "visit_id")
    c2.close()

    assert tables_first == tables_second
    assert n_visit_id == 1


def test_migrate_directly_idempotent(conn):
    """_migrate() itself is safe to re-run (it's called on every connect)."""
    db._migrate(conn)
    db._migrate(conn)  # must not raise "duplicate column name"
    cols = _columns(conn, "detections")
    assert "species_source" in cols


# --- insert_detection round-trip --------------------------------------------------------

def _insert_one(conn, **over):
    kw = dict(
        timestamp="2026-06-07T19:25:59.000000-07:00",
        source=db.SOURCE_GLASS_DOOR_CAM,
        detection_class="animal",
        confidence=0.87,
        bbox=(10.0, 20.0, 110.0, 220.0),
        frame_w=1280,
        frame_h=720,
        crop_path="crops/2026-06-07/x.jpg",
    )
    kw.update(over)
    return db.insert_detection(conn, **kw)


def test_insert_detection_round_trip(conn):
    det_id = _insert_one(conn)
    assert isinstance(det_id, int) and det_id > 0

    row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
    assert row["source"] == db.SOURCE_GLASS_DOOR_CAM
    assert row["detection_class"] == "animal"
    assert row["confidence"] == 0.87
    # bbox stored as four queryable columns, in absolute pixels.
    assert (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]) == (10.0, 20.0, 110.0, 220.0)
    assert row["frame_w"] == 1280 and row["frame_h"] == 720
    assert row["crop_path"] == "crops/2026-06-07/x.jpg"
    # V1 leaves these NULL.
    assert row["frame_path"] is None
    assert row["species"] is None
    assert row["individual_id"] is None
    assert row["visit_id"] is None


def test_insert_detection_autoincrement(conn):
    a = _insert_one(conn)
    b = _insert_one(conn)
    assert b > a


# --- set_species / set_individual_bulk --------------------------------------------------

def test_set_species_round_trip(conn):
    det_id = _insert_one(conn)
    db.set_species(conn, det_id, "raccoon", 0.91)
    conn.commit()
    row = conn.execute(
        "SELECT species, species_confidence, species_source FROM detections WHERE id = ?",
        (det_id,)).fetchone()
    assert row["species"] == "raccoon"
    assert row["species_confidence"] == 0.91
    assert row["species_source"] == "bioclip"   # auto-classified marker


def test_set_individual_bulk_round_trip(conn):
    ids = [_insert_one(conn) for _ in range(3)]
    n = db.set_individual_bulk(conn, ids, "Notch")
    assert n == 3
    labelled = {r[0] for r in conn.execute(
        "SELECT individual_id FROM detections WHERE id IN (?,?,?)", ids)}
    assert labelled == {"Notch"}

    # None clears the label back out.
    db.set_individual_bulk(conn, ids, None)
    cleared = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE individual_id IS NULL").fetchone()[0]
    assert cleared == 3


# --- insert_visit + assign_visit + clear_visits -----------------------------------------

def test_insert_visit_and_assign_round_trip(conn):
    d1 = _insert_one(conn, confidence=0.5)
    d2 = _insert_one(conn, confidence=0.95)
    vid = db.insert_visit(
        conn, source=db.SOURCE_GLASS_DOOR_CAM, species="raccoon", individual_id=None,
        started_at="2026-06-07T19:25:00-07:00", ended_at="2026-06-07T19:31:00-07:00",
        detection_count=2, max_confidence=0.95, representative_detection_id=d2,
    )
    db.assign_visit(conn, [d1, d2], vid)
    conn.commit()

    v = conn.execute("SELECT * FROM visits WHERE id = ?", (vid,)).fetchone()
    assert v["species"] == "raccoon"
    assert v["detection_count"] == 2
    assert v["max_confidence"] == 0.95
    assert v["representative_detection_id"] == d2

    # both detections now carry the visit_id stamp.
    stamped = {r[0] for r in conn.execute(
        "SELECT visit_id FROM detections WHERE id IN (?, ?)", (d1, d2))}
    assert stamped == {vid}


def test_clear_visits_resets_table_and_stamps(conn):
    d1 = _insert_one(conn)
    vid = db.insert_visit(
        conn, source=db.SOURCE_GLASS_DOOR_CAM, species=None, individual_id=None,
        started_at="2026-06-07T19:25:00-07:00", ended_at="2026-06-07T19:25:00-07:00",
        detection_count=1, max_confidence=0.87, representative_detection_id=d1,
    )
    db.assign_visit(conn, [d1], vid)
    conn.commit()

    db.clear_visits(conn)

    assert conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0] == 0
    # every detection un-stamped (a from-scratch rebuild can run cleanly).
    remaining = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE visit_id IS NOT NULL").fetchone()[0]
    assert remaining == 0


# --- insert_clip round-trip -------------------------------------------------------------

def test_insert_clip_round_trip(conn):
    clip_id = db.insert_clip(
        conn, source=db.SOURCE_GLASS_DOOR_CAM, clip_path="clips/2026-06-07/c.mp4",
        started_at="2026-06-07T19:25:00-07:00", ended_at="2026-06-07T19:25:08-07:00",
        fps=15.0, width=160, height=120, frame_count=120, detection_count=4,
        max_confidence=0.93,
    )
    assert isinstance(clip_id, int) and clip_id > 0
    row = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    assert row["clip_path"] == "clips/2026-06-07/c.mp4"
    assert row["fps"] == 15.0
    assert (row["width"], row["height"]) == (160, 120)
    assert row["frame_count"] == 120
    assert row["detection_count"] == 4
    assert row["max_confidence"] == 0.93


def test_insert_clip_allows_nullable_fields(conn):
    """A clip cut off mid-write has a NULL ended_at; fps/size are nullable too."""
    clip_id = db.insert_clip(
        conn, source=db.SOURCE_GLASS_DOOR_CAM, clip_path="clips/x.mp4",
        started_at="2026-06-07T19:25:00-07:00", ended_at=None, fps=None,
        width=None, height=None, frame_count=0, detection_count=0, max_confidence=None,
    )
    row = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    assert row["ended_at"] is None
    assert row["fps"] is None
    assert row["width"] is None and row["height"] is None


# --- labelled_at: WHEN a label was applied (phase 0) -------------------------------------

def _visit_over(conn, det_ids, *, species="raccoon", source=db.SOURCE_GLASS_DOOR_CAM,
                started="2026-06-07T19:25:00-07:00", ended="2026-06-07T19:31:00-07:00"):
    """A visits row covering `det_ids`, with the detections stamped -- what label_visit needs."""
    vid = db.insert_visit(conn, source=source, species=species, individual_id=None,
                          started_at=started, ended_at=ended, detection_count=len(det_ids),
                          max_confidence=0.9, representative_detection_id=det_ids[0])
    db.assign_visit(conn, det_ids, vid)
    conn.commit()
    return vid


def test_labelled_at_column_present_and_null_by_default(conn):
    """Additive and honest: a freshly captured crop has no label, so it has no label TIME."""
    assert "labelled_at" in _columns(conn, "detections")
    det_id = _insert_one(conn)
    row = conn.execute("SELECT labelled_at FROM detections WHERE id = ?", (det_id,)).fetchone()
    assert row["labelled_at"] is None


def test_migrate_is_safe_when_labelled_at_already_exists(conn):
    """The migration guard is a column check, so re-running it on a DB that already has the column
    is a no-op -- not a 'duplicate column name' crash (connect() runs _migrate EVERY open)."""
    db._migrate(conn)
    db._migrate(conn)
    n = sum(1 for r in conn.execute("PRAGMA table_info(detections)") if r[1] == "labelled_at")
    assert n == 1


def test_migrate_adds_labelled_at_to_a_db_that_predates_it(db_path):
    """The real migration path: an OLD detections table (no labelled_at) gains the column, and its
    existing rows are backfilled to NULL -- nothing is invented for the 21k labels applied before
    the column existed."""
    raw = sqlite3.connect(db_path)
    raw.execute("""CREATE TABLE detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, source TEXT NOT NULL,
        detection_class TEXT NOT NULL, confidence REAL NOT NULL,
        bbox_x1 REAL NOT NULL, bbox_y1 REAL NOT NULL, bbox_x2 REAL NOT NULL, bbox_y2 REAL NOT NULL,
        frame_w INTEGER NOT NULL, frame_h INTEGER NOT NULL, crop_path TEXT NOT NULL,
        frame_path TEXT, species TEXT, individual_id TEXT)""")
    raw.execute("INSERT INTO detections (timestamp, source, detection_class, confidence, "
                "bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_w, frame_h, crop_path, individual_id) "
                "VALUES ('2026-06-01T20:00:00-07:00', 'glass_door_cam', 'animal', 0.9, "
                "0, 0, 10, 10, 100, 100, 'crops/old.jpg', 'Stan')")
    raw.commit()
    raw.close()

    c = db.connect(db_path)           # runs _migrate over the legacy table
    c.row_factory = sqlite3.Row
    try:
        row = c.execute("SELECT individual_id, labelled_at FROM detections").fetchone()
        assert row["individual_id"] == "Stan"      # the label itself is untouched
        assert row["labelled_at"] is None          # ... but its TIME is honestly unknown
    finally:
        c.close()


def test_label_visit_stamps_labelled_at(conn):
    a = _insert_one(conn, species="raccoon")
    b = _insert_one(conn, species="raccoon")
    untouched = _insert_one(conn, species="raccoon")   # not in the visit
    vid = _visit_over(conn, [a, b])

    assert db.label_visit(conn, vid, "Stan") == 2
    stamps = {r[0]: r[1] for r in conn.execute(
        "SELECT id, labelled_at FROM detections")}
    assert stamps[a] is not None and stamps[b] is not None
    assert stamps[a] == stamps[b]                      # one write, one timestamp
    assert stamps[untouched] is None                   # pre-existing rows stay NULL


def test_label_visit_reject_stamps_labelled_at(conn):
    """A reject ('leave this unnamed') is a labelling ACT: id clears, source stays human, and the
    time is recorded -- that time is exactly what a confirmation-bias measurement needs."""
    a = _insert_one(conn, species="raccoon")
    vid = _visit_over(conn, [a])
    db.label_visit(conn, vid, "Stan")
    db.label_visit(conn, vid, None, reject=True)
    row = conn.execute("SELECT individual_id, individual_source, labelled_at FROM detections "
                       "WHERE id = ?", (a,)).fetchone()
    assert row["individual_id"] is None
    assert row["individual_source"] == "human"         # the tombstone
    assert row["labelled_at"] is not None


def test_apply_visit_label_stamps_labelled_at(conn):
    """record_live_sighting's solo stamp routes through apply_visit_label, so live confirmations
    are timestamped too."""
    a = _insert_one(conn, species="raccoon")
    vid = _visit_over(conn, [a])
    db.apply_visit_label(conn, visit_id=vid, name="Notch")
    assert conn.execute("SELECT labelled_at FROM detections WHERE id = ?",
                        (a,)).fetchone()["labelled_at"] is not None


def test_coverage_dark_seconds_pairs_transitions_and_admits_ignorance(conn):
    """The effort ledger: up/down transitions pair into dark spans; a window the ledger can't
    speak for (no event at or before its start) reads None -- unknown and fully-covered must
    never be conflated."""
    from datetime import datetime, timedelta
    tz = datetime.now().astimezone().tzinfo
    base = datetime(2026, 8, 7, 21, 0, 0, tzinfo=tz)
    src = db.SOURCE_GLASS_DOOR_CAM

    def ev(event, minutes, reason=None):
        conn.execute("INSERT INTO coverage_events (source, event, at, reason) VALUES (?, ?, ?, ?)",
                     (src, event, (base + timedelta(minutes=minutes)).isoformat(), reason))
    # Watching from 21:00; down 22:00-22:30 (a wedge); watching again until the window ends.
    ev("up", 0, "opened"); ev("down", 60, "read-failed"); ev("up", 90, "reconnected")
    conn.commit()

    win = (base, base + timedelta(hours=8))
    dark = db.coverage_dark_seconds(conn, src, *win)
    assert dark == 30 * 60                                   # exactly the wedge half-hour

    # A trailing 'down' with no later 'up' counts dark through the end of the window.
    ev("down", 300, "stopped"); conn.commit()
    dark2 = db.coverage_dark_seconds(conn, src, *win)
    assert dark2 == 30 * 60 + (8 * 60 - 300) * 60            # wedge + the tail

    # A window that opens before any event is honestly unknowable.
    early = (base - timedelta(hours=2), base - timedelta(hours=1))
    assert db.coverage_dark_seconds(conn, src, *early) is None
    # Another source's ledger says nothing about this one.
    assert db.coverage_dark_seconds(conn, "other_cam", *win) is None


def test_species_writes_do_not_stamp_labelled_at(conn):
    """labelled_at means 'when individual_id was written'. A species correction is a different
    label and must not fake an identity-labelling event."""
    a = _insert_one(conn)
    db.set_species(conn, a, "raccoon", 0.9)
    db.correct_species(conn, a, "Virginia opossum")
    assert conn.execute("SELECT labelled_at FROM detections WHERE id = ?",
                        (a,)).fetchone()["labelled_at"] is None


# --- live_sightings: supersede + cross-row conflict (phase C3) ---------------------------

SPAN_A = "2026-07-05T21:05:04.386941-07:00"     # the real ids 25/26/27 span, to the microsecond
SPAN_B = "2026-07-05T21:06:07.787823-07:00"


def _span_dets(conn, timestamps, *, species="raccoon"):
    return [_insert_one(conn, timestamp=t, species=species) for t in timestamps]


def test_relogging_a_span_supersedes_rather_than_duplicating(conn):
    """Two logs over one span: the older row is MARKED superseded (kept -- it is ground truth and
    the correction sequence is signal), the newer is the live one."""
    _span_dets(conn, [SPAN_A, SPAN_B])
    first = db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Clippy"],
                                    span_start=SPAN_A, span_end=SPAN_B)
    second = db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Stan"],
                                     span_start=SPAN_A, span_end=SPAN_B)

    assert second["superseded"] == [first["sighting_id"]]
    assert second["conflict"] is True
    rows = {r[0]: r for r in conn.execute(
        "SELECT id, names, superseded_at, superseded_by FROM live_sightings")}
    assert len(rows) == 2                                  # NOTHING was deleted
    assert json.loads(rows[first["sighting_id"]][1]) == ["Clippy"]     # history preserved verbatim
    assert rows[first["sighting_id"]][2] is not None
    assert rows[first["sighting_id"]][3] == second["sighting_id"]
    assert rows[second["sighting_id"]][2] is None           # the newest row is live


def test_relogging_the_same_name_supersedes_without_conflict(conn):
    """Typing 'Stan' twice over one span is a duplicate, not a disagreement."""
    _span_dets(conn, [SPAN_A, SPAN_B])
    db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Stan"],
                            span_start=SPAN_A, span_end=SPAN_B)
    again = db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["stan"],
                                    span_start=SPAN_A, span_end=SPAN_B)
    assert len(again["superseded"]) == 1        # de-duped
    assert again["conflict"] is False           # case-insensitive: same claim
    assert db.sighting_conflict_groups(conn) == []


def test_three_single_name_rows_over_one_span_are_detected(conn):
    """THE bug, in its real shape (live_sightings 25/26/27, 2026-07-05): 'Clippy', then
    'Clippy Friend', then 'Stan' over the SAME two crops within 33 seconds, each solo stamp
    overwriting the last, so the stored label is simply whatever was typed last.

    individuals.multi_name_sighting_spans cannot see this -- it reads rows carrying 2+ NAMES, and
    this conflict arrived as three separate SINGLE-name rows. Detecting it needs the comparison
    ACROSS rows."""
    d1, d2 = _span_dets(conn, [SPAN_A, SPAN_B])
    ids = [db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=[n],
                                   span_start=SPAN_A, span_end=SPAN_B,
                                   observed_at=obs)["sighting_id"]
           for n, obs in (("Clippy", "2026-07-05T21:14:42.844398-07:00"),
                          ("Clippy Friend", "2026-07-05T21:15:06.521727-07:00"),
                          ("Stan", "2026-07-05T21:15:15.580962-07:00"))]

    # The stored label really is whatever was logged last -- that part is unchanged on purpose
    # (the human's latest word is their best word); what changes is that it is no longer SILENT.
    assert {r[0] for r in conn.execute(
        "SELECT individual_id FROM detections WHERE id IN (?, ?)", (d1, d2))} == {"Stan"}

    groups = db.sighting_conflict_groups(conn)
    assert len(groups) == 1
    g = groups[0]
    assert g["sighting_ids"] == ids
    assert sorted(n.casefold() for n in g["names"]) == ["clippy", "clippy friend", "stan"]
    assert g["resolved"] is True          # every row but the newest carries a supersede mark
    assert g["source"] == db.SOURCE_GLASS_DOOR_CAM


def test_conflict_detection_finds_unsuperseded_history(conn):
    """The five known groups in the live DB were logged BEFORE any supersede column existed, so
    detection must be derived from the spans, never read off a stored flag. Rows written straight
    into the table (no supersede marks at all) are still found -- and reported unresolved."""
    conn.executemany(
        "INSERT INTO live_sightings (source, observed_at, span_start, span_end, names, stamped) "
        "VALUES (?, ?, ?, ?, ?, 0)",
        [(db.SOURCE_GLASS_DOOR_CAM, SPAN_A, SPAN_A, SPAN_B, json.dumps(["Clippy"])),
         (db.SOURCE_GLASS_DOOR_CAM, SPAN_B, SPAN_A, SPAN_B, json.dumps(["Stan"]))])
    conn.commit()
    groups = db.sighting_conflict_groups(conn)
    assert len(groups) == 1
    assert groups[0]["resolved"] is False


def test_solo_log_overlapping_a_pair_log_is_a_conflict(conn):
    """The dangerous kind (live ids 12/13 and 46/47): a PAIR is logged over a span, then a SOLO
    name is logged over an overlapping span and STAMPS every crop with one name. One label on two
    animals mislabels both."""
    _span_dets(conn, [SPAN_A, SPAN_B])
    db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Stan", "Pedro"],
                            span_start=SPAN_A, span_end=SPAN_B)
    solo = db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Pedro"],
                                   span_start=SPAN_A, span_end=SPAN_B)
    assert solo["conflict"] is True
    assert solo["stamped"] == 2                      # the stamp still happens (asymmetry unchanged)
    assert len(db.sighting_conflict_groups(conn)) == 1


def test_sightings_on_different_sources_never_conflict(conn):
    """Two cameras seeing two animals at the same instant is not a disagreement -- and the trail
    cam is a separate domain entirely."""
    db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Stan"],
                            span_start=SPAN_A, span_end=SPAN_B)
    other = db.record_live_sighting(conn, source=db.SOURCE_TRAIL_CAM_SD, names=["Notch"],
                                    span_start=SPAN_A, span_end=SPAN_B)
    assert other["superseded"] == []
    assert other["conflict"] is False
    assert db.sighting_conflict_groups(conn) == []


def test_non_overlapping_sightings_are_independent(conn):
    """Two different animals half an hour apart are two sightings, not a correction."""
    db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Stan"],
                            span_start="2026-07-05T21:05:00-07:00",
                            span_end="2026-07-05T21:06:00-07:00")
    later = db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Pedro"],
                                    span_start="2026-07-05T21:35:00-07:00",
                                    span_end="2026-07-05T21:36:00-07:00")
    assert later["superseded"] == []
    assert db.sighting_conflict_groups(conn) == []


def test_conflicting_sighting_spans_shape_matches_multi_name_spans(conn):
    """conflicting_sighting_spans returns (source, start, end) triples -- deliberately the same
    shape individuals.multi_name_sighting_spans returns, so the two lists concatenate."""
    _span_dets(conn, [SPAN_A, SPAN_B])
    db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Clippy"],
                            span_start=SPAN_A, span_end=SPAN_B)
    db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Stan"],
                            span_start=SPAN_A, span_end=SPAN_B)
    spans = db.conflicting_sighting_spans(conn)
    assert spans == [(db.SOURCE_GLASS_DOOR_CAM, SPAN_A, SPAN_B)]


def test_sighting_helpers_are_quiet_on_a_db_without_the_table(db_path):
    """A read-only dashboard pointed at a DB no writer has migrated yet must get [] , not a crash."""
    raw = sqlite3.connect(db_path)
    raw.execute("CREATE TABLE detections (id INTEGER PRIMARY KEY)")
    raw.commit()
    raw.close()
    ro = db.connect_readonly(db_path)
    try:
        assert db.sighting_conflict_groups(ro) == []
        assert db.conflicting_sighting_spans(ro) == []
    finally:
        ro.close()


# --- the roster: individual_status ---------------------------------------------------------
# One raccoon here was last photographed 2026-06-30 and never came back, but its 46 templates
# stayed live -- and the auto tier lined up to write that name onto two 2026-07-03 visits. No
# evaluation can see that error (LOO scores against labels, and a departed animal's labels stop),
# so the fact has to come from the human. This table is where it lands.

def test_individual_status_round_trip_records_a_date_not_a_boolean(conn):
    row = db.set_individual_status(conn, "Notch", status="departed",
                                   effective_date="2026-06-30", note="last seen at the dish")
    assert row["name"] == "Notch" and row["status"] == "departed"
    assert row["effective_date"] == "2026-06-30" and row["updated_at"]
    stored = db.individual_statuses(conn)["Notch"]
    assert stored["effective_date"] == "2026-06-30"
    assert stored["note"] == "last seen at the dish"
    assert db.departed_individuals(conn) == {"notch": ("2026-06-30", None)}


def test_individual_status_is_an_upsert_and_reversible(conn):
    db.set_individual_status(conn, "Notch", status="departed", effective_date="2026-06-30")
    db.set_individual_status(conn, "Notch", status="departed", effective_date="2026-07-02")
    assert len(db.individual_statuses(conn)) == 1          # one row per name, not a log
    assert db.departed_individuals(conn) == {"notch": ("2026-07-02", None)}
    # Undo: the row is KEPT (updated_at says when the call was reversed), but the name is off the
    # departed list, so the auto tier may write it again.
    db.set_individual_status(conn, "Notch", status="resident")
    assert db.departed_individuals(conn) == {}
    assert db.individual_statuses(conn)["Notch"]["status"] == "resident"


def test_individual_status_matches_names_case_insensitively(conn):
    """individual_id is free text typed by hand; "notch" and "Notch" are one animal."""
    db.set_individual_status(conn, "Notch", status="departed", effective_date="2026-06-30")
    db.set_individual_status(conn, "notch", status="departed", effective_date="2026-06-29")
    assert len(db.individual_statuses(conn)) == 1
    assert db.departed_individuals(conn) == {"notch": ("2026-06-29", None)}


def test_individual_status_rejects_a_bad_date_or_status(conn):
    """A typo must be a loud error, not a guard that silently never fires."""
    for bad in ("30-06-2026", "yesterday", "2026-13-01"):
        with pytest.raises(ValueError):
            db.set_individual_status(conn, "Notch", effective_date=bad)
    with pytest.raises(ValueError):
        db.set_individual_status(conn, "Notch", status="dead-ish")
    with pytest.raises(ValueError):
        db.set_individual_status(conn, "   ", status="departed")
    assert db.departed_individuals(conn) == {}


def test_individual_status_accepts_a_full_timestamp_as_the_day(conn):
    """The dashboard has ISO timestamps to hand (last_seen); take the calendar day off one."""
    row = db.set_individual_status(conn, "Notch", status="departed",
                                   effective_date="2026-06-30T23:41:07.123-07:00")
    assert row["effective_date"] == "2026-06-30"


def test_departed_with_no_date_fails_closed(conn):
    """No date = no way to tell which visits predate the departure, so the name is simply never
    machine-written (individuals.VisitMatcher.is_departed treats None as always-departed)."""
    db.set_individual_status(conn, "Notch", status="departed")
    assert db.departed_individuals(conn) == {"notch": (None, None)}


def test_individual_status_survives_a_second_connect(db_path):
    """Additive + idempotent: the table is created by SCHEMA's CREATE TABLE IF NOT EXISTS, so a
    second connect() on an existing file neither raises nor loses the row."""
    c1 = db.connect(db_path)
    db.set_individual_status(c1, "Notch", status="departed", effective_date="2026-06-30")
    tables_first = _table_names(c1)
    c1.close()
    c2 = db.connect(db_path)
    try:
        assert _table_names(c2) == tables_first
        assert "individual_status" in tables_first
        assert db.departed_individuals(c2) == {"notch": ("2026-06-30", None)}
        assert len(db.individual_statuses(c2)) == 1
    finally:
        c2.close()


def test_individual_statuses_is_quiet_on_a_db_without_the_table(db_path):
    """A read-only clone of a DB no writer has migrated must read {} , not crash the dashboard."""
    raw = sqlite3.connect(db_path)
    raw.execute("CREATE TABLE detections (id INTEGER PRIMARY KEY)")
    raw.commit()
    raw.close()
    ro = db.connect_readonly(db_path)
    try:
        assert db.individual_statuses(ro) == {}
        assert db.departed_individuals(ro) == {}
    finally:
        ro.close()


# --- the reference-image veto's storage (2026-08-07) -----------------------------------------
# docs/refimg-design-2026-08-07.md section 7. Three additive pieces: four suppression columns on
# detections, the reference_images the suppressions can be REPLAYED against, and view_epochs so
# "the camera moved" is a recorded event instead of an inference. Everything here is inert -- the
# veto ships in shadow mode, so the flag is written and NOTHING reads it for a week.

GD = db.SOURCE_GLASS_DOOR_CAM


def test_suppression_columns_exist_and_default_to_live(conn):
    """NULL suppressed_at == a live row, which is what lets every existing query keep working
    unchanged: a freshly captured crop is not suppressed by anybody."""
    cols = _columns(conn, "detections")
    for c in ("suppressed_at", "suppressed_by", "suppress_ref_id", "suppress_detail"):
        assert c in cols, f"detections.{c} missing"
    det_id = _insert_one(conn)
    row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
    assert row["suppressed_at"] is None
    assert row["suppressed_by"] is None
    assert row["suppress_ref_id"] is None
    assert row["suppress_detail"] is None


def test_refimg_tables_exist(conn):
    """reference_images + view_epochs are created by SCHEMA's CREATE TABLE IF NOT EXISTS."""
    tables = _table_names(conn)
    assert "reference_images" in tables
    assert "view_epochs" in tables
    assert _columns(conn, "reference_images") == {
        "id", "source", "illumination", "view_epoch", "captured_at", "provenance",
        "image_path", "cover_path", "edge_fp", "n_frames", "span_s", "retired_at"}
    assert _columns(conn, "view_epochs") == {
        "id", "source", "epoch", "started_at", "detected_by", "corr"}
    # The lookup index the veto reads references through, on the exact key order it queries.
    idx = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='reference_images'")}
    assert "idx_refimg_lookup" in idx


def test_refimg_migration_is_idempotent(conn):
    """connect() runs _migrate on EVERY open, so a second (and third) pass must be a no-op, not a
    'duplicate column name' crash."""
    db._migrate(conn)
    db._migrate(conn)
    for c in ("suppressed_at", "suppressed_by", "suppress_ref_id", "suppress_detail"):
        n = sum(1 for r in conn.execute("PRAGMA table_info(detections)") if r[1] == c)
        assert n == 1, f"detections.{c} added {n} times"


def test_refimg_migration_survives_a_second_connect(db_path):
    """The whole-file path: connect, write a reference and an epoch, close, connect again. Nothing
    is dropped, nothing is duplicated, and the recorded state is still there."""
    c1 = db.connect(db_path)
    ref = db.save_reference_image(c1, source=GD, illumination="night", view_epoch=0,
                                  provenance="certified+motion_masked",
                                  image_path="refs/gd/night_0.png")
    db.bump_view_epoch(c1, GD, corr=0.41)
    tables_first = _table_names(c1)
    c1.close()

    c2 = db.connect(db_path)
    try:
        assert _table_names(c2) == tables_first
        assert db.current_view_epoch(c2, GD) == 1
        assert db.reference_image(c2, ref)["image_path"] == "refs/gd/night_0.png"
    finally:
        c2.close()


def test_migrate_adds_suppression_to_a_db_that_predates_it(db_path):
    """The real migration path: a LEGACY detections table (no suppression columns at all) gains all
    four, its existing rows read as LIVE, and the two new tables appear alongside -- additive, safe
    on the running rig's DB, and safe to run twice."""
    raw = sqlite3.connect(db_path)
    raw.execute("""CREATE TABLE detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, source TEXT NOT NULL,
        detection_class TEXT NOT NULL, confidence REAL NOT NULL,
        bbox_x1 REAL NOT NULL, bbox_y1 REAL NOT NULL, bbox_x2 REAL NOT NULL, bbox_y2 REAL NOT NULL,
        frame_w INTEGER NOT NULL, frame_h INTEGER NOT NULL, crop_path TEXT NOT NULL,
        frame_path TEXT, species TEXT, individual_id TEXT)""")
    raw.execute("INSERT INTO detections (timestamp, source, detection_class, confidence, "
                "bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_w, frame_h, crop_path, individual_id) "
                "VALUES ('2026-06-01T20:00:00-07:00', 'glass_door_cam', 'animal', 0.9, "
                "0, 0, 10, 10, 100, 100, 'crops/old.jpg', 'Stan')")
    raw.commit()
    raw.close()

    c = db.connect(db_path)           # runs SCHEMA + _migrate over the legacy table
    c.row_factory = sqlite3.Row
    try:
        db._migrate(c)                # ... and again, to prove the guard, not just the ALTER
        cols = _columns(c, "detections")
        for col in ("suppressed_at", "suppressed_by", "suppress_ref_id", "suppress_detail"):
            assert col in cols
        row = c.execute("SELECT individual_id, suppressed_at, suppressed_by FROM detections"
                        ).fetchone()
        assert row["individual_id"] == "Stan"     # the pre-existing row is untouched ...
        assert row["suppressed_at"] is None       # ... and reads as LIVE
        assert row["suppressed_by"] is None
        assert "reference_images" in _table_names(c)
        assert "view_epochs" in _table_names(c)
    finally:
        c.close()


# --- record_suppression / clear_suppression -------------------------------------------------

DETAIL = {"decision": "SUPPRESS", "reason": "matches_empty_reference_at_a_recurring_spot",
          "provenance": "certified+motion_masked", "view_corr": 0.93, "age_s": 312.4,
          "scores": {"lum": 4.70, "dssim": 0.0245, "sobel": 0.0922},
          "thresholds": {"lum": 11.406, "dssim": 0.2346, "sobel": 0.4854},
          "recurrence": {"events": 9, "days": 3, "n": 214}}


def test_record_suppression_writes_all_four_columns(conn):
    """The flag carries its own evidence: who decided, against which reference, and the whole gate
    trace -- so a suppression that turns out to be a raccoon is diagnosable months later without
    re-running the veto."""
    det_id = _insert_one(conn)
    ref = db.save_reference_image(conn, source=GD, illumination="night", view_epoch=0,
                                  provenance="certified+motion_masked",
                                  image_path="refs/gd/night_0.png")
    assert db.record_suppression(conn, det_id, db.SUPPRESSED_BY_REFIMG_VETO, ref, DETAIL) is True

    row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
    assert row["suppressed_at"] is not None
    assert row["suppressed_by"] == "refimg_veto"
    assert row["suppress_ref_id"] == ref
    assert json.loads(row["suppress_detail"])["scores"]["dssim"] == 0.0245
    # SHADOW MODE: the detection itself is completely unaltered -- box, crop, confidence, class.
    assert row["crop_path"] == "crops/2026-06-07/x.jpg"
    assert row["confidence"] == 0.87
    assert row["detection_class"] == "animal"
    assert (row["bbox_x1"], row["bbox_y2"]) == (10.0, 220.0)
    # ... and it is still returned by a query that knows nothing about suppression.
    assert conn.execute("SELECT COUNT(*) FROM detections WHERE detection_class = 'animal'"
                        ).fetchone()[0] == 1


def test_record_suppression_accepts_a_prebuilt_json_string(conn):
    """A caller that already encoded its trace must not get it double-encoded into a JSON string
    containing JSON."""
    det_id = _insert_one(conn)
    db.record_suppression(conn, det_id, "staticfilter", None, json.dumps(DETAIL))
    stored = conn.execute("SELECT suppress_detail FROM detections WHERE id = ?",
                          (det_id,)).fetchone()[0]
    assert json.loads(stored) == DETAIL


def test_record_suppression_refuses_to_overwrite(conn):
    """The first mechanism to flag a row owns the explanation. Silently replacing it would destroy
    the evidence for the decision that actually happened."""
    det_id = _insert_one(conn)
    assert db.record_suppression(conn, det_id, "refimg_veto", 7, DETAIL) is True
    first = conn.execute("SELECT suppressed_at FROM detections WHERE id = ?",
                         (det_id,)).fetchone()[0]

    assert db.record_suppression(conn, det_id, "staticfilter", 99, {"decision": "SUPPRESS"}) is False

    row = conn.execute("SELECT suppressed_at, suppressed_by, suppress_ref_id, suppress_detail "
                       "FROM detections WHERE id = ?", (det_id,)).fetchone()
    assert row["suppressed_at"] == first            # nothing moved
    assert row["suppressed_by"] == "refimg_veto"
    assert row["suppress_ref_id"] == 7
    assert json.loads(row["suppress_detail"])["reason"].startswith("matches_empty")


def test_record_suppression_on_a_missing_row_is_false_not_a_crash(conn):
    assert db.record_suppression(conn, 999999, "refimg_veto", None, None) is False


def test_record_suppression_requires_a_mechanism(conn):
    """An unattributed flag is an unauditable flag -- 'something suppressed this' is not a finding."""
    det_id = _insert_one(conn)
    for bad in ("", "   ", None):
        with pytest.raises(ValueError):
            db.record_suppression(conn, det_id, bad)
    assert conn.execute("SELECT suppressed_at FROM detections WHERE id = ?",
                        (det_id,)).fetchone()[0] is None


def test_clear_suppression_restores_a_live_row(conn):
    """The human's 'that was an animal' verdict off the audit sheet -- and the reason the veto is
    allowed to be wrong. All four columns go back to NULL, and the row can be flagged again."""
    det_id = _insert_one(conn)
    db.record_suppression(conn, det_id, "refimg_veto", 1, DETAIL)

    assert db.clear_suppression(conn, det_id) is True
    row = conn.execute("SELECT suppressed_at, suppressed_by, suppress_ref_id, suppress_detail "
                       "FROM detections WHERE id = ?", (det_id,)).fetchone()
    assert tuple(row) == (None, None, None, None)

    assert db.clear_suppression(conn, det_id) is False      # already live: nothing to do
    # cleared means clearable again -- re-flagging is now allowed (no stale refusal).
    assert db.record_suppression(conn, det_id, "refimg_veto", 2, DETAIL) is True


# --- reference_images lifecycle ---------------------------------------------------------------

def test_reference_lifecycle_save_latest_retire(conn):
    """save -> latest -> a reposition retires the old epoch -> latest is None until the new epoch
    has its own reference. What the camera looked like empty before it was moved is not what it
    looks like empty now, and handing out the old one is how a stale guard fails silently."""
    first = db.save_reference_image(
        conn, source=GD, illumination="night", view_epoch=0,
        captured_at="2026-08-06T00:07:00-07:00", provenance="certified+motion_masked",
        image_path="refs/gd/n0.png", cover_path="refs/gd/n0_cover.png",
        edge_fp=b"\x00\x01\x02\x03", n_frames=1, span_s=0.0)
    got = db.latest_reference(conn, GD, "night", 0)
    assert got["id"] == first
    assert got["provenance"] == "certified+motion_masked"
    assert got["cover_path"] == "refs/gd/n0_cover.png"
    assert got["edge_fp"] == b"\x00\x01\x02\x03"
    assert got["captured_at"] == "2026-08-06T00:07:00-07:00"
    assert got["retired_at"] is None

    # A newer reference in the same key wins.
    second = db.save_reference_image(conn, source=GD, illumination="night", view_epoch=0,
                                     provenance="certified", image_path="refs/gd/n1.png")
    assert db.latest_reference(conn, GD, "night", 0)["id"] == second

    # The camera moves. Everything from the old epoch is retired in the same call.
    epoch = db.bump_view_epoch(conn, GD, corr=0.38)
    assert epoch == 1
    assert db.latest_reference(conn, GD, "night", 0) is None      # retired: never handed out again
    assert db.latest_reference(conn, GD, "night", 1) is None      # the new view has no reference yet

    # ... but the retired rows are KEPT, so a suppression written last week still resolves to the
    # exact image that justified it.
    replay = db.reference_image(conn, first)
    assert replay["image_path"] == "refs/gd/n0.png"
    assert replay["retired_at"] is not None
    assert conn.execute("SELECT COUNT(*) FROM reference_images").fetchone()[0] == 2

    third = db.save_reference_image(conn, source=GD, illumination="night", view_epoch=epoch,
                                    provenance="certified", image_path="refs/gd/n2.png")
    assert db.latest_reference(conn, GD, "night", epoch)["id"] == third


def test_references_are_keyed_by_illumination_and_source(conn):
    """(camera, view epoch, illumination) is the key, SWITCHED never blended: the day<->night flip
    is the largest pixel event on this camera, and the trail cam is a different world entirely."""
    night = db.save_reference_image(conn, source=GD, illumination="night", view_epoch=0,
                                    provenance="certified", image_path="n.png")
    day = db.save_reference_image(conn, source=GD, illumination="day", view_epoch=0,
                                  provenance="certified", image_path="d.png")
    tc = db.save_reference_image(conn, source=db.SOURCE_TRAIL_CAM_SD, illumination="ir",
                                 view_epoch=0, provenance="rank_p50", image_path="ir.png")
    assert db.latest_reference(conn, GD, "night", 0)["id"] == night
    assert db.latest_reference(conn, GD, "day", 0)["id"] == day
    assert db.latest_reference(conn, db.SOURCE_TRAIL_CAM_SD, "ir", 0)["id"] == tc
    assert db.latest_reference(conn, GD, "ir", 0) is None
    assert db.latest_reference(conn, db.SOURCE_TRAIL_CAM_SD, "night", 0) is None


def test_retiring_one_source_leaves_the_other_alone(conn):
    """Two cameras move independently; one reposition must not blind the other."""
    db.save_reference_image(conn, source=GD, illumination="night", view_epoch=0,
                            provenance="certified", image_path="n.png")
    db.save_reference_image(conn, source=db.SOURCE_TRAIL_CAM_SD, illumination="ir", view_epoch=0,
                            provenance="rank_p50", image_path="ir.png")
    assert db.retire_reference_images(conn, GD, 1) == 1
    assert db.retire_reference_images(conn, GD, 1) == 0        # idempotent
    assert db.latest_reference(conn, GD, "night", 0) is None
    assert db.latest_reference(conn, db.SOURCE_TRAIL_CAM_SD, "ir", 0) is not None


def test_save_reference_rejects_an_unknown_illumination(conn):
    """illumination is a lookup KEY: a typo would make latest_reference return nothing forever --
    a guard that silently never fires, which is the exact failure mode this project bans."""
    for bad in ("dusk", "infrared", "", None):
        with pytest.raises(ValueError):
            db.save_reference_image(conn, source=GD, illumination=bad, view_epoch=0,
                                    provenance="certified", image_path="x.png")
    # ... but case/whitespace is normalized, not rejected.
    rid = db.save_reference_image(conn, source=GD, illumination=" Night ", view_epoch=0,
                                  provenance="certified", image_path="x.png")
    assert db.reference_image(conn, rid)["illumination"] == "night"
    assert db.latest_reference(conn, GD, "night", 0)["id"] == rid


def test_save_reference_requires_source_and_provenance(conn):
    with pytest.raises(ValueError):
        db.save_reference_image(conn, source="", illumination="day", view_epoch=0,
                                provenance="certified", image_path="x.png")
    with pytest.raises(ValueError):
        db.save_reference_image(conn, source=GD, illumination="day", view_epoch=0,
                                provenance="  ", image_path="x.png")


# --- view_epochs -------------------------------------------------------------------------------

def test_view_epoch_starts_at_zero_and_bumps_monotonically(conn):
    """0 is the implicit 'has not been seen to move' state, so a rig that never detects a
    reposition still has a valid, stable reference key."""
    assert db.current_view_epoch(conn, GD) == 0
    assert conn.execute("SELECT COUNT(*) FROM view_epochs").fetchone()[0] == 0

    assert db.bump_view_epoch(conn, GD, corr=0.41) == 1
    assert db.current_view_epoch(conn, GD) == 1
    assert db.bump_view_epoch(conn, GD, "manual") == 2
    assert db.current_view_epoch(conn, GD) == 2

    rows = conn.execute("SELECT epoch, detected_by, corr, started_at FROM view_epochs "
                        "WHERE source = ? ORDER BY epoch", (GD,)).fetchall()
    assert [r["epoch"] for r in rows] == [1, 2]
    assert [r["detected_by"] for r in rows] == ["edge_fp_corr", "manual"]
    assert rows[0]["corr"] == 0.41
    assert rows[1]["corr"] is None            # a manual bump has no correlation to report
    assert all(r["started_at"] for r in rows)


def test_view_epochs_are_per_source(conn):
    """The trail cam moves ~3x as often as the glass door; their epochs are separate counters."""
    db.bump_view_epoch(conn, GD)
    db.bump_view_epoch(conn, GD)
    assert db.bump_view_epoch(conn, db.SOURCE_TRAIL_CAM_SD) == 1
    assert db.current_view_epoch(conn, GD) == 2
    assert db.current_view_epoch(conn, db.SOURCE_TRAIL_CAM_SD) == 1


def test_view_epoch_unique_constraint_holds(conn):
    """UNIQUE(source, epoch): an epoch number means one view, and re-using it would silently merge
    two different framings under one reference key."""
    db.bump_view_epoch(conn, GD)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO view_epochs (source, epoch, started_at, detected_by) "
                     "VALUES (?, 1, '2026-08-07T12:00:00-07:00', 'edge_fp_corr')", (GD,))
    conn.rollback()
    # The same epoch number on the OTHER camera is fine -- the constraint is per source.
    conn.execute("INSERT INTO view_epochs (source, epoch, started_at, detected_by) "
                 "VALUES (?, 1, '2026-08-07T12:00:00-07:00', 'edge_fp_corr')",
                 (db.SOURCE_TRAIL_CAM_SD,))
    conn.commit()


def test_bump_view_epoch_requires_a_detector(conn):
    with pytest.raises(ValueError):
        db.bump_view_epoch(conn, GD, "")
    assert db.current_view_epoch(conn, GD) == 0


def test_refimg_helpers_are_quiet_on_a_db_without_the_tables(db_path):
    """A read-only clone of a DB no writer has migrated must read the neutral answer -- no
    reference (so the veto abstains) and epoch 0 -- rather than crashing a reporting surface."""
    raw = sqlite3.connect(db_path)
    raw.execute("CREATE TABLE detections (id INTEGER PRIMARY KEY)")
    raw.commit()
    raw.close()
    ro = db.connect_readonly(db_path)
    try:
        assert db.latest_reference(ro, GD, "night", 0) is None
        assert db.reference_image(ro, 1) is None
        assert db.current_view_epoch(ro, GD) == 0
    finally:
        ro.close()


# =====================================================================================
# clips.video_scanned_at -- "we looked at the video" as a fact distinct from "we found something"
#
# clips.detection_count/max_confidence describe the STILL that triggered a trail-cam recording,
# not the video (measured 2026-08-10: 1,089 of 1,101 trail-cam clips carry a max_confidence
# byte-identical to some still's, and that camera starts rolling ~2-3 s after the photo). These
# tests pin the two things that fall out of recording the video's own evidence.
# =====================================================================================
def _a_clip(conn, source="trail_cam_sd", path="clips/x/a.mp4"):
    cur = conn.execute(
        "INSERT INTO clips (source, clip_path, started_at, ended_at, fps, width, height, "
        "frame_count, detection_count, max_confidence) "
        "VALUES (?, ?, '2026-08-09T01:48:15-07:00', '2026-08-09T01:48:19-07:00', 30.0, "
        "3840, 2160, 121, 1, 0.3738)", (source, path))
    conn.commit()
    return int(cur.lastrowid)


def test_video_scan_is_recorded_separately_from_the_trigger_stills_numbers(conn):
    cid = _a_clip(conn)
    db.set_clip_video_scan(conn, cid, n_boxes=0, max_conf=None)
    r = conn.execute("SELECT detection_count, max_confidence, video_scanned_at, "
                     "video_detections, video_max_conf FROM clips WHERE id = ?", (cid,)).fetchone()
    # The trigger still's numbers are untouched -- "what tripped the camera" stays a real fact...
    assert r[0] == 1 and abs(r[1] - 0.3738) < 1e-9
    # ...while the video's own verdict is recorded beside it: looked, found nothing.
    assert r[2] is not None
    assert r[3] == 0 and r[4] is None


def test_an_empty_clip_is_not_rescanned_forever(conn):
    """The bug this fixes: a video with no animal produces no clip_tracks rows, so under a
    track-existence test it was indistinguishable from one never processed and got re-detected on
    every nightly run (154 trail-cam clips were in that loop)."""
    cid = _a_clip(conn)
    model = "megadetector-v6"
    assert cid in [r["id"] for r in db.clips_needing_tracks(conn, model)]
    db.set_clip_video_scan(conn, cid, n_boxes=0, max_conf=None)   # looked; nothing there
    assert cid not in [r["id"] for r in db.clips_needing_tracks(conn, model)]


def test_a_clip_with_a_real_animal_records_what_the_video_showed(conn):
    cid = _a_clip(conn)
    db.set_clip_video_scan(conn, cid, n_boxes=37, max_conf=0.91)
    r = conn.execute("SELECT video_detections, video_max_conf FROM clips WHERE id = ?",
                     (cid,)).fetchone()
    assert r[0] == 37 and abs(r[1] - 0.91) < 1e-9


def test_departed_individuals_carries_a_return_date_when_the_animal_came_back(conn):
    """The pair is an absence INTERVAL. Notch left and returned 43 days later; a single date
    cannot express that, and with only a departure the guard would refuse his real visits ever
    after (see individuals.VisitMatcher.is_departed)."""
    db.set_individual_status(conn, "Notch", status="departed",
                             effective_date="2026-06-30", returned_on="2026-08-06")
    assert db.departed_individuals(conn) == {"notch": ("2026-06-30", "2026-08-06")}
    assert db.individual_statuses(conn)["Notch"]["returned_on"] == "2026-08-06"
    # Coming back for good clears the flag entirely, as before.
    db.set_individual_status(conn, "Notch", status="resident")
    assert db.departed_individuals(conn) == {}
