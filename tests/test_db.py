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
    assert db.departed_individuals(conn) == {"notch": "2026-06-30"}


def test_individual_status_is_an_upsert_and_reversible(conn):
    db.set_individual_status(conn, "Notch", status="departed", effective_date="2026-06-30")
    db.set_individual_status(conn, "Notch", status="departed", effective_date="2026-07-02")
    assert len(db.individual_statuses(conn)) == 1          # one row per name, not a log
    assert db.departed_individuals(conn) == {"notch": "2026-07-02"}
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
    assert db.departed_individuals(conn) == {"notch": "2026-06-29"}


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
    assert db.departed_individuals(conn) == {"notch": None}


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
        assert db.departed_individuals(c2) == {"notch": "2026-06-30"}
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
