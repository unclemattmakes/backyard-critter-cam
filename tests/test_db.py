"""
Tests for db.py -- the stdlib-sqlite3 storage layer.

Covers: connect() builds the schema and runs _migrate(); the four real tables exist; migration
is idempotent (connect twice on the same file is a no-op); and insert/read round-trips for every
writer V1+phase-4 uses (insert_detection, insert_visit, insert_clip, set_species,
set_individual_bulk, clear_visits / assign_visit).

All DBs are throwaway tempfiles (the `conn` / `db_path` fixtures); the real backyard.db is untouched.
"""
from __future__ import annotations

import sqlite3

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
