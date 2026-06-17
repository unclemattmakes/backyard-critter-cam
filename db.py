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
    individual_id       TEXT,           -- NULLABLE. Phase 3 (re-identification) fills this. NULL until phase 3.
    visit_id            INTEGER,        -- Phase 4: which visit (visits.id) this crop belongs to; stamped by visits.py.

    -- "How good a shot is this?" Image-derived (sharpness x night-eyeshine boost; see quality.py),
    -- so the dashboard can lead a visit with its CUTEST/sharpest frame, not just the most confident
    -- one. NULL until scored (live at capture time, or backfilled by `python quality.py`).
    crop_quality        REAL
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

-- Phase 4: VISIT-EVENT COLLAPSING. One animal lingering fires many detections (the first dusk
-- session logged ~45 crops of a single raccoon), so raw crop counts over-count "visits" ~10x and
-- every frequency / behaviour statistic must count VISITS, not crops. A visit = consecutive
-- detections on the SAME source separated by < N minutes (visits.py builds this table and stamps
-- detections.visit_id). It can't yet split two animals present at once -- that needs reliable
-- individual_id -- so `individual_id` here is the visit's DOMINANT label when available, else NULL.
-- (Counting crows also needs co-occurrence -- who arrived together -- which behaviour.py reads
-- off these rows; see PLAN.md phase 4.)
CREATE TABLE IF NOT EXISTS visits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT    NOT NULL,   -- which rig (matches detections.source).
    species         TEXT,               -- dominant species across the visit's crops (NULL if none classified).
    individual_id   TEXT,               -- dominant individual_id (NULL until phase-3 labels exist).
    started_at      TEXT    NOT NULL,   -- first detection's timestamp (local ISO 8601 w/ offset).
    ended_at        TEXT    NOT NULL,   -- last detection's timestamp; dwell = ended-started.
    detection_count INTEGER NOT NULL,   -- crops in the visit (a rough "how long / how active").
    max_confidence  REAL,               -- best detector score in the visit.
    representative_detection_id INTEGER REFERENCES detections(id)   -- the most readable crop.
);
CREATE INDEX IF NOT EXISTS idx_visits_started ON visits(started_at);
CREATE INDEX IF NOT EXISTS idx_visits_source  ON visits(source);
CREATE INDEX IF NOT EXISTS idx_visits_species ON visits(species);
CREATE INDEX IF NOT EXISTS idx_visits_individual ON visits(individual_id);

-- Phase 4 CAPTURE: one short VIDEO clip recorded around a visit (clips.py / --record-clips).
-- Stills say WHO/WHEN; a clip says HOW (gait, approach, dwell, co-occurrence) -- the substance
-- of behaviour, and a confound-robust second shot at individual ID via motion. A clip spans
-- [started_at, ended_at] on one `source`, so the detections written during that window join to
-- it by time -- no FK needed yet. This is a real, populated table (unlike the visits sketch
-- above), but it stays lean: behaviour ANALYSIS reads the clips later, off the live path.
CREATE TABLE IF NOT EXISTS clips (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT    NOT NULL,   -- which rig recorded it (matches detections.source).
    clip_path       TEXT    NOT NULL,   -- path to the saved .mp4 (relative to project root).
    started_at      TEXT    NOT NULL,   -- local ISO 8601 w/ offset, like detections.timestamp.
    ended_at        TEXT,               -- set when the clip is finalized (NULL if cut off mid-write).
    fps             REAL,               -- playback rate written into the file (measured live rate).
    width           INTEGER,
    height          INTEGER,
    frame_count     INTEGER,            -- frames written (pre-roll + live + post-roll).
    detection_count INTEGER,            -- detector hits during the clip (rough "how busy").
    max_confidence  REAL                -- best detector score in the clip (a usability proxy).
);
CREATE INDEX IF NOT EXISTS idx_clips_started ON clips(started_at);
CREATE INDEX IF NOT EXISTS idx_clips_source  ON clips(source);

-- Phase 4 ANALYSIS: motion tracks per clip (clipmotion.py). The clip's frames are sampled,
-- the detector finds the animals in each, and per-frame boxes are associated into one row PER
-- ANIMAL (track_idx 0..N -- a pair visit yields two tracklets). Each box-centre trajectory
-- becomes a MOTION FINGERPRINT: how fast, how straight, how hesitant, approaching or
-- retreating -- plus a GAIT estimate (stride cadence from the body-bob periodicity while
-- walking). Motion is the behaviour signal stills can't carry -- and a confound-robust second
-- shot at individual ID (a limp reads the same from any angle). `track` keeps the raw
-- normalized trajectory as JSON so richer gait features can be re-derived later WITHOUT
-- re-running the detector over the video.
CREATE TABLE IF NOT EXISTS clip_tracks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id         INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    model           TEXT    NOT NULL,   -- detector used to build the track.
    track_idx       INTEGER NOT NULL DEFAULT 0,  -- which animal within the clip (0-based).
    created_at      TEXT    NOT NULL,
    n_samples       INTEGER,            -- frames sampled from the clip.
    n_hits          INTEGER,            -- samples in THIS tracklet (its length in boxes).
    track           TEXT,               -- JSON [[t_s, cx, cy, w, h, conf], ...], coords normalized 0..1.
    duration_s      REAL,               -- span of the track (first hit .. last hit).
    path_len        REAL,               -- total distance travelled (normalized units).
    net_disp        REAL,               -- straight-line start->end distance.
    straightness    REAL,               -- net_disp / path_len: 1 = beeline, ~0 = milling about.
    avg_speed       REAL,               -- mean speed while moving (normalized units / s).
    peak_speed      REAL,
    moving_frac     REAL,               -- fraction of samples in motion (vs stationary, e.g. eating).
    area_trend      REAL,               -- end/start box-area ratio: >1 approached the camera, <1 left.
    stride_hz       REAL,               -- gait cadence (body-bob frequency while walking); NULL = none found.
    stride_strength REAL,               -- autocorrelation peak 0..1 backing stride_hz; NULL with it.
    walk_s          REAL                -- seconds of continuous walking the gait estimate is based on.
);
CREATE INDEX IF NOT EXISTS idx_clip_tracks_clip ON clip_tracks(clip_id);

-- Phase 4 (part 3): one APPEARANCE prototype per clip tracklet (clipembed.py). A tracklet is a
-- single animal tracked through a clip, so embedding its sharpest frames gives a clean per-ANIMAL
-- appearance vector EVEN inside a multi-animal visit -- which the still pipeline can only treat as
-- a blended whole. Same 1536-d MegaDescriptor space as detection_embeddings, so clip and still
-- vectors are directly comparable (a tracklet can be matched to a confirmed individual, two
-- tracklets in one clip can be tested as a different-individual pair, and a pair visit's tracklets
-- can be clustered to un-blend the two animals). Keyed by (track, model); cascades when a track is
-- recomputed.
CREATE TABLE IF NOT EXISTS clip_track_embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id    INTEGER NOT NULL REFERENCES clip_tracks(id) ON DELETE CASCADE,
    model       TEXT    NOT NULL,   -- embedder tag, e.g. 'megadescriptor-l-384'.
    dim         INTEGER NOT NULL,
    embedding   BLOB    NOT NULL,   -- float32, L2-normalized (mean of the sampled frames).
    n_frames    INTEGER,            -- how many frames were pooled into this prototype.
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clip_track_emb ON clip_track_embeddings(track_id, model);
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
    crop_quality: Optional[float] = None,  # Image-derived shot quality (quality.py); NULL if unscored.
) -> int:
    """Insert one detection row; returns its new id."""
    x1, y1, x2, y2 = bbox
    cur = conn.execute(
        """
        INSERT INTO detections (
            timestamp, source, detection_class, confidence,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_w, frame_h,
            crop_path, frame_path, species, individual_id, crop_quality
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp, source, detection_class, float(confidence),
            float(x1), float(y1), float(x2), float(y2), int(frame_w), int(frame_h),
            crop_path, frame_path, species, individual_id,
            None if crop_quality is None else float(crop_quality),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def insert_clip(conn: sqlite3.Connection, *, source: str, clip_path: str, started_at: str,
                ended_at: Optional[str], fps: Optional[float], width: Optional[int],
                height: Optional[int], frame_count: int, detection_count: int,
                max_confidence: Optional[float]) -> int:
    """Phase 4: record one behaviour clip's metadata; returns its new id."""
    cur = conn.execute(
        """
        INSERT INTO clips (source, clip_path, started_at, ended_at, fps, width, height,
                           frame_count, detection_count, max_confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (source, clip_path, started_at, ended_at,
         None if fps is None else float(fps),
         None if width is None else int(width),
         None if height is None else int(height),
         int(frame_count), int(detection_count),
         None if max_confidence is None else float(max_confidence)),
    )
    conn.commit()
    return int(cur.lastrowid)


def insert_clip_tracks(conn: sqlite3.Connection, *, clip_id: int, model: str, n_samples: int,
                       tracklets: Sequence[dict]) -> int:
    """Store ALL of one clip's motion tracklets (one row per animal, track_idx 0..N) + their
    derived features (clipmotion.py). Each tracklet dict: {"track_json", "n_hits", "features"}.
    Replaces every existing row for the same (clip, model) so --redo is idempotent. A clip with
    NO animal found still gets one empty marker row (n_hits=0) so the resumable batch doesn't
    re-process it forever. Returns rows inserted."""
    conn.execute("DELETE FROM clip_tracks WHERE clip_id = ? AND model = ?", (int(clip_id), model))
    if not tracklets:
        tracklets = [{"track_json": "[]", "n_hits": 0, "features": {}}]
    for idx, t in enumerate(tracklets):
        f = t.get("features") or {}
        conn.execute(
            """INSERT INTO clip_tracks (clip_id, model, track_idx, created_at, n_samples, n_hits,
                                        track, duration_s, path_len, net_disp, straightness,
                                        avg_speed, peak_speed, moving_frac, area_trend,
                                        stride_hz, stride_strength, walk_s)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (int(clip_id), model, idx, now_local_iso(), int(n_samples), int(t["n_hits"]),
             t["track_json"], f.get("duration_s"), f.get("path_len"), f.get("net_disp"),
             f.get("straightness"), f.get("avg_speed"), f.get("peak_speed"),
             f.get("moving_frac"), f.get("area_trend"),
             f.get("stride_hz"), f.get("stride_strength"), f.get("walk_s")),
        )
    conn.commit()
    return len(tracklets)


def clips_needing_tracks(conn: sqlite3.Connection, model: str):
    """Clips that don't yet have a motion track for `model` (resumable batch processing)."""
    return conn.execute(
        """SELECT c.id, c.clip_path, c.fps FROM clips c
           WHERE NOT EXISTS (SELECT 1 FROM clip_tracks t WHERE t.clip_id = c.id AND t.model = ?)
           ORDER BY c.id""", (model,)
    ).fetchall()


def insert_clip_track_embedding(conn: sqlite3.Connection, *, track_id: int, model: str, dim: int,
                                embedding: bytes, n_frames: int) -> None:
    """Store one tracklet's appearance prototype (clipembed.py). Raw float32 bytes, already
    L2-normalized so a dot product is cosine similarity. Idempotent per (track, model)."""
    conn.execute("DELETE FROM clip_track_embeddings WHERE track_id = ? AND model = ?",
                 (int(track_id), model))
    conn.execute(
        "INSERT INTO clip_track_embeddings (track_id, model, dim, embedding, n_frames, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (int(track_id), model, int(dim), embedding, int(n_frames), now_local_iso()))
    conn.commit()


def clip_tracks_needing_embedding(conn: sqlite3.Connection, model: str, min_hits: int):
    """Sustained tracklets (>= min_hits boxes) that have a stored track but no appearance vector
    for `model` yet -- the resumable work-list for clipembed.py. Joins the clip path + fps so the
    embedder can re-open the video and seek to the tracklet's frames."""
    return conn.execute(
        """SELECT t.id AS track_id, t.clip_id, t.track_idx, t.track, t.n_hits,
                  c.clip_path, c.fps
           FROM clip_tracks t JOIN clips c ON c.id = t.clip_id
           WHERE t.n_hits >= ? AND t.track IS NOT NULL
             AND NOT EXISTS (SELECT 1 FROM clip_track_embeddings e
                             WHERE e.track_id = t.id AND e.model = ?)
           ORDER BY t.clip_id, t.track_idx""",
        (int(min_hits), model)).fetchall()


def load_clip_track_embeddings(conn: sqlite3.Connection, model: str, *, species=None):
    """Read back tracklet appearance prototypes joined to clip + (time-overlapping) context, for
    matching/analysis. Returns rows with: track_id, clip_id, track_idx, n_hits, embedding,
    clip_started_at, clip_source. (Visit/individual attribution is done by the caller via time
    overlap, as visits renumber on rebuild.)"""
    return conn.execute(
        """SELECT e.track_id, t.clip_id, t.track_idx, t.n_hits, e.embedding,
                  c.started_at AS clip_started_at, c.ended_at AS clip_ended_at, c.source AS clip_source
           FROM clip_track_embeddings e
           JOIN clip_tracks t ON t.id = e.track_id
           JOIN clips c ON c.id = t.clip_id
           WHERE e.model = ? ORDER BY c.started_at, t.track_idx""", (model,)).fetchall()


def clear_visits(conn: sqlite3.Connection) -> None:
    """Drop all visit rows and un-stamp every detection -- visits.py does a full rebuild (the
    detection set is small and a from-scratch pass is simpler/safer than incremental upkeep)."""
    conn.execute("DELETE FROM visits")
    conn.execute("UPDATE detections SET visit_id = NULL WHERE visit_id IS NOT NULL")
    conn.commit()


def insert_visit(conn: sqlite3.Connection, *, source: str, species: Optional[str],
                 individual_id: Optional[str], started_at: str, ended_at: str,
                 detection_count: int, max_confidence: Optional[float],
                 representative_detection_id: Optional[int]) -> int:
    """Phase 4: write one collapsed visit event; returns its new id."""
    cur = conn.execute(
        """
        INSERT INTO visits (source, species, individual_id, started_at, ended_at,
                            detection_count, max_confidence, representative_detection_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (source, species, individual_id, started_at, ended_at, int(detection_count),
         None if max_confidence is None else float(max_confidence),
         None if representative_detection_id is None else int(representative_detection_id)),
    )
    return int(cur.lastrowid)


def assign_visit(conn: sqlite3.Connection, detection_ids: Sequence[int], visit_id: int) -> None:
    """Stamp a visit's id onto its member detections (detections.visit_id)."""
    conn.executemany("UPDATE detections SET visit_id = ? WHERE id = ?",
                     [(int(visit_id), int(i)) for i in detection_ids])


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the original schema (keeps existing DBs current)."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(detections)")}
    if "species_confidence" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN species_confidence REAL")
    if "species_verified" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN species_verified INTEGER")
    if "species_source" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN species_source TEXT")
    if "visit_id" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN visit_id INTEGER")
    if "crop_quality" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN crop_quality REAL")
    if "individual_source" not in cols:
        # Who assigned individual_id: 'cluster' (reid.py look-alike proposal) vs 'human'
        # (confirmed in the dashboard / --name). The suggestion engine builds its appearance
        # templates ONLY from human-confirmed labels; placeholder clusters never feed back.
        conn.execute("ALTER TABLE detections ADD COLUMN individual_source TEXT")
        # Every label written before this column existed was a reid.py placeholder cluster.
        conn.execute("UPDATE detections SET individual_source = 'cluster' "
                     "WHERE individual_id IS NOT NULL")
    # clip_tracks grew multi-animal + gait columns (2026-06-11, phase 4 part 2).
    ct_cols = {r[1] for r in conn.execute("PRAGMA table_info(clip_tracks)")}
    if "track_idx" not in ct_cols:
        conn.execute("ALTER TABLE clip_tracks ADD COLUMN track_idx INTEGER NOT NULL DEFAULT 0")
    for col in ("stride_hz", "stride_strength", "walk_s"):
        if col not in ct_cols:
            conn.execute(f"ALTER TABLE clip_tracks ADD COLUMN {col} REAL")


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


def set_crop_quality_bulk(conn: sqlite3.Connection, pairs: Sequence[tuple]) -> int:
    """Store image-derived crop_quality for many detections at once -- `pairs` = [(det_id, q), ...].
    Used by quality.py to backfill crops captured before scoring existed. Returns rows updated."""
    conn.executemany("UPDATE detections SET crop_quality = ? WHERE id = ?",
                     [(float(q), int(i)) for i, q in pairs])
    conn.commit()
    return len(pairs)


# ---------------------------------------------------------------------------
# Phase 3 -- re-identification: appearance embeddings + individual labels.
#
# embed.py writes one L2-normalized appearance vector per readable crop into
# detection_embeddings (keyed by `model`, so a second embedder can be added later without a
# migration). reid.py reads them back as a matrix to cluster crops into individuals; once a
# cluster is named, individual_id is written onto the member detections. Counting stays in
# crops here -- VISIT collapsing (PLAN.md phase 4) needs individual_id first, which is exactly
# what this produces.
# ---------------------------------------------------------------------------

def insert_embedding(conn: sqlite3.Connection, detection_id: int, model: str, dim: int,
                     embedding: bytes, created_at: Optional[str] = None) -> None:
    """Store one appearance vector for a detection. `embedding` is raw float32 bytes (already
    L2-normalized, so a dot product between two vectors is their cosine similarity). Replaces
    any existing vector for the same (detection, model) so a --redo run is idempotent."""
    conn.execute(
        "DELETE FROM detection_embeddings WHERE detection_id = ? AND model = ?",
        (int(detection_id), model),
    )
    conn.execute(
        "INSERT INTO detection_embeddings (detection_id, model, dim, embedding, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (int(detection_id), model, int(dim), embedding, created_at or now_local_iso()),
    )


def embedded_ids(conn: sqlite3.Connection, model: str) -> set[int]:
    """Detection ids that already have a vector for `model` -- lets embed.py skip them (resumable)."""
    return {r[0] for r in conn.execute(
        "SELECT detection_id FROM detection_embeddings WHERE model = ?", (model,))}


def fetch_for_embedding(conn: sqlite3.Connection, model: str, *, species: Optional[str],
                        min_confidence: float, redo: bool, limit: int = 0):
    """Animal crops that should get an appearance vector: above the usability gate, optionally
    restricted to one species. Unless --redo, rows already embedded for `model` are skipped."""
    where = "detection_class = 'animal' AND confidence >= ?"
    params: list = [min_confidence]
    if species:
        where += " AND species = ?"
        params.append(species)
    rows = conn.execute(
        f"SELECT id, crop_path FROM detections WHERE {where} ORDER BY id", params
    ).fetchall()
    if not redo:
        done = embedded_ids(conn, model)
        rows = [r for r in rows if r[0] not in done]
    return rows[:limit] if limit else rows


def load_embeddings(conn: sqlite3.Connection, model: str, *, species: Optional[str] = None,
                    min_confidence: float = 0.0):
    """Read back every stored vector for `model` (optionally one species / confidence gate),
    joined to the detection metadata reid.py needs. Returns a list of sqlite3.Row with
    columns: id, crop_path, confidence, species, individual_id, timestamp, embedding (bytes)."""
    where = "e.model = ? AND d.detection_class = 'animal' AND d.confidence >= ?"
    params: list = [model, min_confidence]
    if species:
        where += " AND d.species = ?"
        params.append(species)
    return conn.execute(
        f"""SELECT d.id, d.crop_path, d.confidence, d.species, d.individual_id, d.timestamp,
                   e.embedding
            FROM detection_embeddings e JOIN detections d ON d.id = e.detection_id
            WHERE {where} ORDER BY d.id""",
        params,
    ).fetchall()


def set_individual(conn: sqlite3.Connection, detection_id: int,
                   individual_id: Optional[str], source: Optional[str] = "human") -> None:
    """Phase 3: assign (or clear, with None) the individual a crop belongs to."""
    conn.execute("UPDATE detections SET individual_id = ?, individual_source = ? WHERE id = ?",
                 (individual_id, None if individual_id is None else source, int(detection_id)))
    conn.commit()


def set_individual_bulk(conn: sqlite3.Connection, detection_ids: Sequence[int],
                        individual_id: Optional[str], source: Optional[str] = "human") -> int:
    """Assign one individual to many crops at once (naming a whole cluster). Returns the count.
    `source` records who decided: 'human' (a confirmation -- feeds the suggestion templates) or
    'cluster' (a reid.py look-alike proposal -- never feeds back into suggestions)."""
    src = None if individual_id is None else source
    conn.executemany("UPDATE detections SET individual_id = ?, individual_source = ? WHERE id = ?",
                     [(individual_id, src, int(i)) for i in detection_ids])
    conn.commit()
    return len(detection_ids)


def rename_individual(conn: sqlite3.Connection, old: str, new: Optional[str]) -> int:
    """Rename every crop labelled `old` to `new` (the dashboard's naming action). Naming two
    groups the same `new` MERGES them -- that's a feature: several look-alike clusters often turn
    out to be one animal. `new=None` clears the label back to unassigned. Returns rows changed.
    A rename is a human act: the renamed crops become source='human' (a cleared label loses its
    source too)."""
    cur = conn.execute(
        "UPDATE detections SET individual_id = ?, individual_source = ? WHERE individual_id = ?",
        (new, None if new is None else "human", old))
    conn.commit()
    return cur.rowcount


def label_visit(conn: sqlite3.Connection, visit_id: int, individual_id: Optional[str],
                source: str = "human") -> int:
    """Confirm WHO a visit was: stamp `individual_id` onto the visit's detections that match the
    visit's dominant species (a stray mid-visit crow crop keeps its own identity), and mirror it
    onto the visits row. The visit is the labelling unit -- one solo animal per visit, so one
    confirmation labels every crop AND makes the visit a template for future suggestions.
    `individual_id=None` clears. Returns crops stamped."""
    v = conn.execute("SELECT species FROM visits WHERE id = ?", (int(visit_id),)).fetchone()
    if v is None:
        return 0
    sp = v[0]
    where = "visit_id = ?" + ("" if sp is None else " AND species = ?")
    params = [int(visit_id)] + ([] if sp is None else [sp])
    cur = conn.execute(
        f"UPDATE detections SET individual_id = ?, individual_source = ? WHERE {where}",
        [individual_id, None if individual_id is None else source] + params)
    conn.execute("UPDATE visits SET individual_id = ? WHERE id = ?",
                 (individual_id, int(visit_id)))
    conn.commit()
    return cur.rowcount


_UNSET = object()   # sentinel: "argument not provided" (distinct from None = "clear the label")


def apply_visit_label(conn: sqlite3.Connection, *, visit_id: Optional[int] = None,
                      source: Optional[str] = None, start: Optional[str] = None,
                      end: Optional[str] = None, name=_UNSET, species: Optional[str] = None,
                      verify: bool = False) -> dict:
    """Confirm/correct a whole visit's SPECIES and/or assign its INDIVIDUAL, in one shot. The
    visit is identified EITHER by `visit_id` (the Individuals queue, which has it) OR by a
    (`source`, `start`, `end`) time span (the Explorer computes visits on the fly and has no
    visits.id) -- both resolve to the same set of detections, so the two review surfaces share
    this one backend.

      species='raccoon'  -> correct every crop in the set to that species (confidence 1,
                            verified, source='human').
      verify=True        -> (no species) just confirm the existing species on every crop.
      name='Stan'        -> set individual_id on the crops matching the visit's DOMINANT species
                            (a stray crow crop in a raccoon visit keeps its own identity);
                            name=None clears; omit `name` to leave identity untouched.

    Species is applied first, so naming after a correction scopes to the corrected species. Any
    `visits` rows the detections belong to are synced (species/individual_id) so the queue and the
    Explorer agree. Returns a small summary dict. Operates via WHERE clauses (not id lists) so a
    1000-crop visit never hits SQLite's bound-variable limit."""
    if visit_id is not None:
        where, params = "visit_id = ?", [int(visit_id)]
    elif source and start and end:
        where, params = "source = ? AND timestamp >= ? AND timestamp <= ?", [source, start, end]
    else:
        return {"detections": 0, "error": "need visit_id or source+start+end"}

    n = conn.execute(f"SELECT COUNT(*) FROM detections WHERE {where}", params).fetchone()[0]
    if not n:
        return {"detections": 0}

    if species:
        conn.execute(
            f"UPDATE detections SET species = ?, species_confidence = 1.0, species_verified = 1, "
            f"species_source = 'human' WHERE {where}", [species] + params)
    elif verify:
        conn.execute(f"UPDATE detections SET species_verified = 1 WHERE {where}", params)

    dominant = None
    if name is not _UNSET:
        row = conn.execute(
            f"SELECT species FROM detections WHERE {where} AND species IS NOT NULL "
            f"GROUP BY species ORDER BY COUNT(*) DESC LIMIT 1", params).fetchone()
        dominant = row[0] if row else None
        src = None if name is None else "human"
        if dominant is None:
            conn.execute(f"UPDATE detections SET individual_id = ?, individual_source = ? "
                         f"WHERE {where}", [name, src] + params)
        else:
            conn.execute(f"UPDATE detections SET individual_id = ?, individual_source = ? "
                         f"WHERE {where} AND species = ?", [name, src] + params + [dominant])

    # Sync the visits-table rows these detections belong to (subquery, no big IN list).
    vsub = f"id IN (SELECT DISTINCT visit_id FROM detections WHERE {where} AND visit_id IS NOT NULL)"
    if species:
        conn.execute(f"UPDATE visits SET species = ? WHERE {vsub}", [species] + params)
    if name is not _UNSET:
        conn.execute(f"UPDATE visits SET individual_id = ? WHERE {vsub}", [name] + params)
    conn.commit()
    return {"detections": int(n), "dominant_species": dominant,
            "species_set": species or None, "named": None if name is _UNSET else name}


def confirmed_visit_labels(conn: sqlite3.Connection, species: Optional[str] = None) -> dict:
    """{visit_id: individual_id} for every visit with human-confirmed crops (the suggestion
    templates). Read from DETECTIONS, not visits -- visits.py rebuilds from scratch, but the
    stamped crops persist, so labels survive a rebuild. Dominant name per visit."""
    where = "individual_source = 'human' AND individual_id IS NOT NULL AND visit_id IS NOT NULL"
    params: list = []
    if species:
        where += " AND species = ?"
        params.append(species)
    rows = conn.execute(
        f"""SELECT visit_id, individual_id, COUNT(*) n FROM detections WHERE {where}
            GROUP BY visit_id, individual_id ORDER BY visit_id, n DESC""", params).fetchall()
    out: dict = {}
    for r in rows:                       # first row per visit = dominant (ordered by n DESC)
        out.setdefault(r[0], r[1])
    return out
