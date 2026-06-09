"""
SQLite layer for the backyard-critter rig.

Design goals:
  * V1 populates `detections` for the live glass-door cam.
  * A second source (the wider-yard trail cam, batch SD-card import, IR at night) plugs in
    later by writing rows with a different `source` value -- no schema change needed.
  * Phase 2 (species classification) fills `species`; phase 3 (re-identification) fills
    `individual_id` and writes vectors into `detection_embeddings`. Both columns/tables
    already exist so future phases are pure INSERT/UPDATE, never a migration.

Only stdlib sqlite3 is used.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

# Known `source` values. V1 only ever writes the first; the rest document the plan so the
# meaning of the column is obvious to whoever reads the DB later.
SOURCE_GLASS_DOOR_CAM = "glass_door_cam"  # V1: live webcam at the glass door. Primary rig, all species, day+night.
SOURCE_TRAIL_CAM_SD = "trail_cam_sd"      # FUTURE: wider-yard weatherproof trail cam; batch SD-card import, IR at night.

SCHEMA = """
-- One row per detected object above the confidence threshold.
CREATE TABLE IF NOT EXISTS detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Local time WITH UTC offset, ISO 8601 (e.g. 2026-06-07T19:25:59.123456-07:00).
    -- See config.py "Timezone convention" and db.now_local_iso().
    timestamp       TEXT    NOT NULL,

    -- Which capture rig produced this row. V1: 'glass_door_cam'. Future: 'trail_cam_sd'.
    source          TEXT    NOT NULL,

    -- Coarse MegaDetector label: 'animal' | 'person' | 'vehicle'. This is NOT the species
    -- (that is the nullable `species` column, filled in phase 2).
    detection_class TEXT    NOT NULL,

    -- Detector score, 0..1. Doubles as a crop-USABILITY score: high-confidence crops are the
    -- readable ones worth feeding to phase-2 species ID and phase-3 re-ID (filter on this).
    confidence      REAL    NOT NULL,

    -- Bounding box in ABSOLUTE pixel coordinates of the captured frame. Stored as four
    -- columns (not one blob) so boxes are directly queryable -- e.g. filter by size. The
    -- frame dimensions are stored alongside so a box stays interpretable, and can be
    -- re-normalized, even if the capture resolution changes between runs.
    bbox_x1         REAL    NOT NULL,
    bbox_y1         REAL    NOT NULL,
    bbox_x2         REAL    NOT NULL,
    bbox_y2         REAL    NOT NULL,
    frame_w         INTEGER NOT NULL,
    frame_h         INTEGER NOT NULL,

    crop_path       TEXT    NOT NULL,   -- path to the saved crop (relative to project root). Always written.
    frame_path      TEXT,               -- path to the full frame; NULL unless save_full_frame is on.

    species             TEXT,           -- Phase 2 (species classification) fills this, e.g. 'raccoon'.
    species_confidence  REAL,           -- Phase 2 classifier score 0..1 for `species` (NULL until classified).
    species_verified    INTEGER,        -- Human review via dashboard: NULL = unreviewed, 1 = confirmed, 0 = wrong.
    species_source      TEXT,           -- 'bioclip' (auto) or 'human' (corrected in the dashboard).
    individual_id       TEXT            -- NULLABLE. Phase 3 (re-identification) fills this. NULL until phase 3.
);

CREATE INDEX IF NOT EXISTS idx_detections_timestamp  ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_detections_source     ON detections(source);
CREATE INDEX IF NOT EXISTS idx_detections_class      ON detections(detection_class);
CREATE INDEX IF NOT EXISTS idx_detections_species    ON detections(species);
CREATE INDEX IF NOT EXISTS idx_detections_individual ON detections(individual_id);

-- FUTURE (phase 3, re-identification): vector embeddings used to cluster crops into
-- individual animals. Kept in a side table keyed to a detection (one detection may get
-- several vectors, e.g. from different embedding models) so the hot `detections` table
-- stays lean. V1 NEVER writes here -- the table exists only so phase 3 is a pure insert.
CREATE TABLE IF NOT EXISTS detection_embeddings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
    model        TEXT    NOT NULL,   -- which embedding model produced the vector.
    dim          INTEGER NOT NULL,   -- vector length.
    embedding    BLOB    NOT NULL,   -- raw float32 bytes, e.g. np.asarray(vec, np.float32).tobytes().
    created_at   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_detection ON detection_embeddings(detection_id);

-- FUTURE (phase 4 prep): VISIT-EVENT COLLAPSING -- not implemented in V1, captured here so
-- the plan lives at the schema. One animal lingering fires many detections (the first dusk
-- session logged ~45 crops of a single raccoon). Raw crop counts over-count "visits" ~10x,
-- so any frequency / behaviour statistic must count VISITS, not crops. A visit = consecutive
-- detections of the SAME individual on the SAME source separated by < N minutes -- which
-- needs individual_id (phase 3) to be precise, so it lands in phase 4. Planned shape:
--   CREATE TABLE visits (
--     id            INTEGER PRIMARY KEY AUTOINCREMENT,
--     source        TEXT,  species TEXT,  individual_id TEXT,  -- individual_id NULL until re-ID
--     started_at    TEXT,  ended_at TEXT,
--     detection_count INTEGER,  max_confidence REAL,
--     representative_detection_id INTEGER REFERENCES detections(id)   -- the most readable crop
--   );
-- plus a detections.visit_id FK assigned by a collapsing pass. (Counting crows also needs
-- co-occurrence -- who arrived together -- see PLAN.md phase 4.)
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open (creating if needed) the DB and ensure the schema exists."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    # WAL: lets a future reader (analysis notebook, phase-2 tagger) read while the live
    # rig keeps writing, without lock contention.
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")  # wait out the live rig's writes rather than erroring
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def connect_readonly(db_path: Path | str) -> Optional[sqlite3.Connection]:
    """Open the DB read-only (WAL-safe) for reporting -- usable while the live rig writes.
    Returns None if the database doesn't exist yet."""
    p = Path(db_path)
    if not p.exists():
        return None
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def now_local_iso() -> str:
    """Current local time with UTC offset, ISO 8601. The rig's one timestamp source."""
    return datetime.now().astimezone().isoformat()


def insert_detection(
    conn: sqlite3.Connection,
    *,
    timestamp: str,
    source: str,
    detection_class: str,
    confidence: float,
    bbox: Sequence[float],          # (x1, y1, x2, y2) absolute pixels.
    frame_w: int,
    frame_h: int,
    crop_path: str,
    frame_path: Optional[str] = None,
    species: Optional[str] = None,       # Left NULL in V1.
    individual_id: Optional[str] = None,  # Left NULL in V1.
) -> int:
    """Insert one detection row; returns its new id."""
    x1, y1, x2, y2 = bbox
    cur = conn.execute(
        """
        INSERT INTO detections (
            timestamp, source, detection_class, confidence,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_w, frame_h,
            crop_path, frame_path, species, individual_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp, source, detection_class, float(confidence),
            float(x1), float(y1), float(x2), float(y2), int(frame_w), int(frame_h),
            crop_path, frame_path, species, individual_id,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the original schema (keeps existing DBs current)."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(detections)")}
    if "species_confidence" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN species_confidence REAL")
    if "species_verified" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN species_verified INTEGER")
    if "species_source" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN species_source TEXT")


def set_species(conn: sqlite3.Connection, detection_id: int, species: str,
                confidence: float) -> None:
    """Phase 2: write an auto-classified species + score onto a detection."""
    conn.execute(
        "UPDATE detections SET species = ?, species_confidence = ?, species_source = 'bioclip' "
        "WHERE id = ?",
        (species, float(confidence), int(detection_id)),
    )


def set_species_verified(conn: sqlite3.Connection, detection_id: int, verified) -> None:
    """Dashboard review: mark a detection's species confirmed (1), wrong (0), or unreviewed (None)."""
    conn.execute("UPDATE detections SET species_verified = ? WHERE id = ?",
                 (verified, int(detection_id)))
    conn.commit()


def correct_species(conn: sqlite3.Connection, detection_id: int, species: str) -> None:
    """Dashboard correction: set a human-chosen species (confirmed, full confidence, source=human)."""
    conn.execute(
        "UPDATE detections SET species = ?, species_confidence = 1.0, species_verified = 1, "
        "species_source = 'human' WHERE id = ?",
        (species, int(detection_id)),
    )
    conn.commit()
