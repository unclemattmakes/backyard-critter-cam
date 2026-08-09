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

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

# Project root (the folder holding this file; identical to config.ROOT, but resolved here without a
# config import so db stays import-light). Every stored crop/clip path is relative to this root.
_ROOT = Path(__file__).resolve().parent

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
    crop_quality        REAL,

    -- WHEN individual_id was last written onto this crop (local ISO 8601 w/ offset). Every OTHER
    -- timestamp in this schema is an OBSERVATION time -- when the animal was in front of the
    -- camera -- so before this column there was no way to ask when a LABEL was applied. Two things
    -- need it: label VELOCITY (grouping individual_source='human' by week otherwise groups by when
    -- the animal visited, and would move even in a week nobody labelled anything), and any bound on
    -- the confirmation bias that caps every accuracy number in docs/identity-eval-2026-08-05.md
    -- (the confirmed corpus was built by a human agreeing with matches the model proposed; sizing
    -- that needs to know which labels landed after which suggestions).
    -- Written by every writer of individual_id: label_visit, apply_visit_label, set_individual,
    -- set_individual_bulk, rename_individual -- including a CLEAR/reject, whose time is exactly what
    -- a bias measurement wants. Additive and backfilled to NULL: every label written before
    -- 2026-08-05 has no stamp and NOTHING may assume this is set.
    labelled_at         TEXT,

    -- SOFT SUPPRESSION (2026-08-07, docs/refimg-design-2026-08-07.md section 7). A box the
    -- reference-image veto believes is FURNITURE -- the tipped watering can that MegaDetector calls
    -- a raccoon 60 times -- is written exactly as before and then FLAGGED here. Nothing is dropped,
    -- because an erased animal writes no row at all and is a silent permanent loss, while a wrongly
    -- flagged one is a row somebody can look at and un-flag. Same soft-delete convention as
    -- clips.pruned_at.
    --
    -- NULL suppressed_at == a LIVE row, so every existing query keeps working unchanged until it
    -- opts in. The feature ships in SHADOW MODE: the veto writes these columns for a week and NO
    -- consumer honours them -- not the dashboard, not individuals.still_tracklets, not co-presence,
    -- not stats -- so a week of flagged crops can be audited on one contact sheet before anything
    -- acts on the flag. When consumers do opt in, the read-side contract is: exclude from
    -- still_tracklets' cannot-link constraint, co-presence badges and species statistics; keep
    -- everywhere else, and ALWAYS keep in eval.py, because the only way to learn the veto's
    -- precision is to keep scoring the boxes it removed.
    suppressed_at       TEXT,    -- when it was flagged (local ISO 8601 w/ offset). NULL = live row.
    suppressed_by       TEXT,    -- which mechanism decided: 'refimg_veto' | 'staticfilter' | ...
    suppress_ref_id     INTEGER, -- reference_images.id the box was compared against (NULL if none).
    -- The WHOLE decision as JSON, so a bad suppression is diagnosable months later without
    -- re-running anything: {"decision":"SUPPRESS","reason":...,"provenance":...,"view_corr":...,
    -- "age_s":...,"scores":{...},"thresholds":{...},"recurrence":{"events":9,"days":3,"n":214}}.
    suppress_detail     TEXT
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
    representative_detection_id INTEGER REFERENCES detections(id),  -- the most readable crop.

    -- How decisive the species vote was: (winner - runner-up) / total vote weight, 0..1. 1.0 = every
    -- crop agreed, ~0 = a coin flip that got silently committed. Species labels GATE the re-ID
    -- gallery (a raccoon relabelled to opossum can never be matched to Stan), so a near-tie is a
    -- thing to surface, not to bury. NULL when no crop in the visit carries a species.
    species_margin  REAL,

    -- 1 when a human live-logged CONFLICTING names over a span overlapping this visit (see
    -- sighting_conflict_groups): "Clippy", then "Clippy Friend", then "Stan" over the same two
    -- crops in 33 seconds. Either two animals were present or the human corrected themselves, and
    -- nothing in the data can tell which -- so the visit is flagged rather than trusted as a
    -- single-animal template. 1 = conflict, 0 = checked and clean, NULL = never computed (a visit
    -- written by something other than visits.build_visits, or built before this column existed).
    sighting_conflict INTEGER
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
    max_confidence  REAL,               -- best detector score in the clip (a usability proxy).
    pruned_at       TEXT                -- set when the rolling-window pruner deleted the FILE.
                                        -- The row (and its clip_tracks / clip_track_embeddings /
                                        -- individual links) outlives the video: derived re-ID and
                                        -- behaviour data must not reset every disk-budget cycle.
                                        -- backup.py archives the file itself before pruning
                                        -- reaches it, when scheduled. NULL = playable on disk.
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
    rep_crop    TEXT,               -- path to a representative frame-crop of this tracklet (UI thumbnail).
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clip_track_emb ON clip_track_embeddings(track_id, model);

-- Phase 3 (re-ID), LIVE ground truth: the human names who is visiting AS IT HAPPENS, from the
-- dashboard's Live tab ("Notch and Elliot are here right now"). That real-time recognition is the
-- gold input the whole suggest-confirm loop is built on -- captured at the moment, not reconstructed
-- from crops later. One row = one "who's here" log over the current visit span on `source`.
--   * SOLO (one name): record_live_sighting ALSO stamps that individual_id onto the span's crops
--     (a live equivalent of confirming a solo visit -- it feeds the appearance templates).
--   * PAIR (2+ names): the names are kept here as co-presence ground truth ONLY; we do NOT stamp a
--     single name across the span, because one label on two animals mislabels both (the documented
--     pair gotcha) and contaminates the template. The un-blend step is what splits a pair later;
--     this row tells it WHO the two clusters are.
CREATE TABLE IF NOT EXISTS live_sightings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT    NOT NULL,            -- which rig was being watched (matches detections.source).
    observed_at TEXT    NOT NULL,            -- when the human logged it (local ISO 8601 w/ offset).
    span_start  TEXT,                        -- first detection of the named visit span (NULL if none active).
    span_end    TEXT,                        -- last detection of the span.
    names       TEXT    NOT NULL,            -- JSON array of individuals present, e.g. ["Notch","Elliot"].
    stamped     INTEGER NOT NULL DEFAULT 0,  -- crops whose individual_id this set (solo case); 0 for co-presence.
    note        TEXT,                        -- optional free-text remark.

    -- SUPERSEDE, not duplicate. Re-logging a span that is already logged used to leave BOTH rows
    -- standing while each solo stamp overwrote the last, so the stored label was simply whatever
    -- was typed last and nothing recorded that a correction had happened. Now the older row is
    -- marked here (kept, never deleted -- this is ground truth, and the correction SEQUENCE is
    -- itself signal) and the newest row is the live one. NULL on every row logged before
    -- 2026-08-05 and on every row nothing has re-logged: absence of a mark is NOT absence of a
    -- conflict, which is why sighting_conflict_groups DERIVES conflicts from the spans rather than
    -- reading these columns.
    superseded_at TEXT,                      -- when it was superseded (local ISO 8601 w/ offset).
    superseded_by INTEGER REFERENCES live_sightings(id)   -- the sighting that replaced it.
);
CREATE INDEX IF NOT EXISTS idx_live_sightings_observed ON live_sightings(observed_at);

-- THE ROSTER (2026-08-05). Animals move on. One of this yard's raccoons was last photographed on
-- 2026-06-30 and simply never came back -- but its 46 confirmed templates stayed in the matcher,
-- and at the operating point the identity evaluation recommends, the auto tier proposed writing
-- that name onto two visits three days AFTER the animal stopped existing here. Almost certainly a
-- different raccoon, matched against a dead template set.
--
-- No evaluation can catch that class of error: leave-one-visit-out scores against the LABELS, and
-- a departed animal's labels simply stop, so the probe set contains no example of "you named an
-- animal that no longer lives here". It is structurally invisible to measurement, which is why it
-- needs a fact from the human rather than a threshold. (A recency gate on template age WAS tried
-- and measured: it made the wrong-name rate WORSE, 0.119 -> 0.137, while cutting coverage
-- 18.0% -> 16.1%. Staleness is not departure.)
--
-- There is no `individuals` table -- identity lives as free text in detections.individual_id and
-- visits.individual_id -- so this is a small side table keyed by that same name, holding only what
-- a human can actually know. It is INERT for everything except what the machine WRITES: a departed
-- individual is still ranked, still suggested, still a template, still on every surface. Matt may
-- well decide an unreviewed June visit is that animal, and he should be able to.
CREATE TABLE IF NOT EXISTS individual_status (
    name           TEXT PRIMARY KEY COLLATE NOCASE,  -- matches detections.individual_id (free text).
    status         TEXT NOT NULL,        -- 'resident' (the default for anyone absent here) | 'departed'.
    -- The LAST DAY the individual was resident, 'YYYY-MM-DD'. A date, not a boolean, because
    -- "last seen 2026-06-30" is the fact the human actually holds, and it keeps the guard narrow:
    -- a visit that STARTED ON OR BEFORE this date may still be auto-named (those visits happened
    -- while the animal was here), and only later ones are refused. NULL means departed with no date
    -- known -- read as "fail closed", no auto-name at any time (see auto_assign's `departed` skip).
    effective_date TEXT,
    note           TEXT,                 -- optional free text ("last seen 06-30, kits stayed").
    updated_at     TEXT NOT NULL         -- when this row was last written (local ISO 8601 w/ offset).
);

-- COVERAGE (2026-08-08): when each camera was actually WATCHING -- the effort ledger every
-- absence claim silently needs. The rig has documented multi-hour blind spells (Modern-Standby
-- deaths, USB wedges, the trail cam's midnight-to-noon video gap), and without this table
-- "first raccoon in 3 days" and "no robin this night" treat a dark camera as an empty yard.
-- Append-only events, written at the RARE transitions (open / read-failure / reconnect / stop),
-- never per frame; a reader pairs up->down spans. History before the table existed is honestly
-- unknowable and left unwritten.
CREATE TABLE IF NOT EXISTS coverage_events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    source  TEXT NOT NULL,                -- which camera (matches detections.source).
    event   TEXT NOT NULL,                -- 'up' (frames flowing) | 'down' (lost/stopped).
    at      TEXT NOT NULL,                -- local ISO 8601 w/ offset.
    reason  TEXT                          -- 'opened' | 'read-failed' | 'reconnected' | 'stopped' ...
);

-- LIFE EVENTS (2026-08-08): the cast's STORY as data -- dated, append-only free text per
-- individual ("kits first emerged", "limping on the left front paw", "presumed dispersed").
-- Distinct from individual_status on purpose: that table is one row of CURRENT residency per
-- name (a state), this is a ledger (a history). Three litters were the biggest biological story
-- this yard ever produced and they existed only as group-stamp strings plus the owner's memory;
-- when a kit vanishes, or next spring's litter debuts a week earlier than this one, the
-- comparison lives here or nowhere. Purely additive: nothing machine-side reads it -- it is
-- provenance and narrative for the humans (profile timeline), exported with the label ledger.
CREATE TABLE IF NOT EXISTS life_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL COLLATE NOCASE,  -- matches detections.individual_id (free text).
    event_date TEXT,                          -- 'YYYY-MM-DD' the event happened (NULL = undated note).
    note       TEXT NOT NULL,                 -- the event, in the human's words.
    labeled_by TEXT,                          -- who recorded it (attribution; NULL = the operator).
    created_at TEXT NOT NULL                  -- when the row was written (local ISO 8601 w/ offset).
);

-- REFERENCE IMAGES (2026-08-07, docs/refimg-design-2026-08-07.md sections 6-7): "what this camera
-- looks like with nothing in it", so a suppression can be REPLAYED against the exact image months
-- later. A reference is a frame the detector ACTUALLY RAN ON and certified empty (zero boxes, motion
-- quiet, held for a hold period) -- not a rolling background model, which is forbidden here: MOG2
-- absorbs a stationary object in ~23 s at 21.6 fps, against a measured 823 s of verified animal
-- residency, so a decaying model would learn the sleeping raccoon and then erase it.
--
-- Keyed on (source, illumination, view_epoch), SWITCHED not blended, because the day<->night flip is
-- the single largest pixel event on this camera (lum 94.3, DSSIM 0.87) and a camera reposition
-- invalidates everything. Illumination is DERIVED FROM THE FRAME (chroma < 6 => ir; else median < 90
-- => night; else day), never from a clock.
--
-- `cover_path` is load-bearing and not decoration: a certified reference can still CONTAIN an
-- undetected animal (measured -- a raccoon walked the wall inside a frame the detector certified
-- empty), so policy E marks every pixel that motion-blobbed in the last 3600 s as NOT COVERED and
-- the veto ABSTAINS there. An unknown pixel is not evidence of emptiness. NULL means fully covered.
--
-- Rows are RETIRED (retired_at set on an epoch change), never deleted: a suppression written last
-- week must still resolve to the image that justified it.
CREATE TABLE IF NOT EXISTS reference_images (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT    NOT NULL,   -- matches detections.source.
    illumination  TEXT    NOT NULL,   -- 'day' | 'night' | 'ir', DERIVED FROM THE FRAME.
    view_epoch    INTEGER NOT NULL,   -- view_epochs.epoch this reference belongs to.
    captured_at   TEXT    NOT NULL,   -- local ISO 8601 w/ offset, like detections.timestamp.
    provenance    TEXT    NOT NULL,   -- 'certified' | 'certified+motion_masked' | 'rank_p50'.
    image_path    TEXT    NOT NULL,   -- PNG, 320x180 grey (the resolution the thresholds are for).
    cover_path    TEXT,               -- PNG mask of KNOWN pixels; NULL == fully covered.
    edge_fp       BLOB,               -- edge fingerprint (float32 bytes) for the view gate.
    n_frames      INTEGER,            -- 1 for a snapshot, pool size for a rank reference.
    span_s        REAL,               -- 0 for a snapshot.
    retired_at    TEXT                -- set on epoch change; the row is NEVER deleted.
);
CREATE INDEX IF NOT EXISTS idx_refimg_lookup
    ON reference_images(source, illumination, view_epoch, captured_at);

-- CAMERA VIEW EPOCHS (2026-08-07): "the camera moved" as a first-class RECORDED event rather than
-- something inferred after the fact -- the failure mode config.py warns about, where a stale
-- hand-measured zone fails SILENTLY. The glass door is repositioned about once every 4 days and the
-- trail cam about once every 1.3 days, so a reference outlives its own validity constantly.
--
-- Detected from DAY frames only, by edge-fingerprint correlation: IR frames from 12 days spanning
-- confirmed repositions all collapse into ONE view cluster, so a night reference can never learn it
-- has been invalidated by looking at itself. A bump flushes every illumination's reference.
CREATE TABLE IF NOT EXISTS view_epochs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT    NOT NULL,   -- matches detections.source.
    epoch        INTEGER NOT NULL,   -- monotonic per source; 0 is the implicit epoch before any bump.
    started_at   TEXT    NOT NULL,   -- when the new view was first believed (local ISO 8601 w/ offset).
    detected_by  TEXT    NOT NULL,   -- 'edge_fp_corr' (or 'manual' when a human says the cam moved).
    corr         REAL,               -- the correlation that triggered it (NULL for a manual bump).
    UNIQUE(source, epoch)
);

-- Ignore zones: persistent static false-fire spots the detector should disregard (a dark wall
-- opening that scores "animal" on every dusk run). Formerly only a hand-edited dict in
-- config_local.py; the table is the runtime source of truth so the dashboard can add/remove
-- zones without a restart. config.ignore_zones still SEEDS this table -- once per exact
-- rectangle (see seed_ignore_zones) -- so an existing config keeps working the first time this
-- schema appears. Rows are soft-deleted (deleted_at): the tombstone is what stops the config
-- seed from resurrecting a rectangle someone deleted in the UI.
CREATE TABLE IF NOT EXISTS ignore_zones (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT    NOT NULL,   -- matches detections.source (a live camera).
    x1           INTEGER NOT NULL,   -- FULL-RES frame pixels; x1<x2, y1<y2 (add_ignore_zone
    y1           INTEGER NOT NULL,   -- normalizes before insert).
    x2           INTEGER NOT NULL,
    y2           INTEGER NOT NULL,
    note         TEXT,               -- optional human label ("wall gap").
    created_by   TEXT    NOT NULL DEFAULT 'web',  -- 'web' (dashboard) | 'config' (seeded).
    created_at   TEXT    NOT NULL,   -- local ISO 8601 w/ offset; compared against
                                     -- view_epochs.started_at to flag zones drawn before the
                                     -- camera last moved.
    deleted_at   TEXT                -- tombstone; live zones have NULL.
);
CREATE INDEX IF NOT EXISTS idx_ignore_zones_source ON ignore_zones(source, deleted_at);
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
    # Build the file: URI safely -- as_uri() percent-encodes characters like '#' and '?' that, raw,
    # would be parsed as a URI fragment/query and silently open the wrong (empty) database.
    conn = sqlite3.connect(p.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def now_local_iso() -> str:
    """Current local time with UTC offset, ISO 8601. The rig's one timestamp source."""
    return datetime.now().astimezone().isoformat()


def parse_local(ts):
    """Parse a stored ISO-8601 timestamp (the rig writes local-time-WITH-offset; see
    now_local_iso). Returns a tz-aware datetime, or None if unparseable. A legacy NAIVE string is
    made tz-aware in local time, so mixing it with the offset-bearing rows never raises
    'can't subtract offset-naive and offset-aware'. This is the one ISO parser for the analysis
    tools (visits / behavior / twoaxis), replacing three copy-pasted _parse helpers."""
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo is not None else dt.astimezone()


def rel_to_root(path) -> str:
    """Stored form of a crop/clip path: relative to the project root, with the OS's native
    separators (backslashes on Windows). Falls back to the absolute string if `path` is outside the
    root. crop_abspath() is the inverse. Consolidates the per-module _rel helpers."""
    try:
        return str(Path(path).relative_to(_ROOT))
    except ValueError:
        return str(path)


def crop_abspath(rel) -> Path:
    """Absolute filesystem path for a stored crop/clip path. The DB stores paths relative to the
    project root with the writer's native separators (backslashes on Windows); normalize to forward
    slashes so the join also resolves on macOS/Linux. The inverse of rel_to_root()."""
    return _ROOT / str(rel).replace("\\", "/")


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
    """Clips that don't yet have a motion track for `model` (resumable batch processing).
    Pruned clips are excluded -- their video is gone, so there is nothing to extract from."""
    return conn.execute(
        """SELECT c.id, c.clip_path, c.fps FROM clips c
           WHERE c.pruned_at IS NULL
             AND NOT EXISTS (SELECT 1 FROM clip_tracks t WHERE t.clip_id = c.id AND t.model = ?)
           ORDER BY c.id""", (model,)
    ).fetchall()


def insert_clip_track_embedding(conn: sqlite3.Connection, *, track_id: int, model: str, dim: int,
                                embedding: bytes, n_frames: int, rep_crop: Optional[str] = None) -> None:
    """Store one tracklet's appearance prototype (clipembed.py). Raw float32 bytes, already
    L2-normalized so a dot product is cosine similarity. `rep_crop` is a saved frame-crop path
    for the UI. Idempotent per (track, model)."""
    conn.execute("DELETE FROM clip_track_embeddings WHERE track_id = ? AND model = ?",
                 (int(track_id), model))
    conn.execute(
        "INSERT INTO clip_track_embeddings (track_id, model, dim, embedding, n_frames, rep_crop, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (int(track_id), model, int(dim), embedding, int(n_frames), rep_crop, now_local_iso()))
    conn.commit()


def set_clip_track_individual(conn: sqlite3.Connection, track_ids: Sequence[int],
                              individual_id: Optional[str], source: str = "human") -> int:
    """Assign (or clear) the individual a set of clip TRACKLETS belong to -- the un-blend action
    (label cluster A 'Notch', cluster B 'Elliot'). Clip-space label, separate from the still-space
    detections.individual_id. Returns rows changed."""
    src = None if individual_id is None else source
    conn.executemany("UPDATE clip_tracks SET individual_id = ?, individual_source = ? WHERE id = ?",
                     [(individual_id, src, int(t)) for t in track_ids])
    conn.commit()
    return len(track_ids)


def clip_tracks_needing_embedding(conn: sqlite3.Connection, model: str, min_hits: int):
    """Sustained tracklets (>= min_hits boxes) that have a stored track but no appearance vector
    for `model` yet -- the resumable work-list for clipembed.py. Joins the clip path + fps so the
    embedder can re-open the video and seek to the tracklet's frames. Pruned clips are excluded
    (no video to seek into) -- which is also why the nightly batch embeds BEFORE pruning ages a
    clip out: vectors extracted in time survive the prune (load_clip_track_embeddings keeps
    returning them)."""
    return conn.execute(
        """SELECT t.id AS track_id, t.clip_id, t.track_idx, t.track, t.n_hits,
                  c.clip_path, c.fps
           FROM clip_tracks t JOIN clips c ON c.id = t.clip_id
           WHERE c.pruned_at IS NULL AND t.n_hits >= ? AND t.track IS NOT NULL
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
        """SELECT e.track_id, t.clip_id, t.track_idx, t.n_hits, e.embedding, e.rep_crop,
                  t.individual_id, c.started_at AS clip_started_at, c.ended_at AS clip_ended_at,
                  c.source AS clip_source
           FROM clip_track_embeddings e
           JOIN clip_tracks t ON t.id = e.track_id
           JOIN clips c ON c.id = t.clip_id
           WHERE e.model = ? AND e.n_frames > 0
           ORDER BY c.started_at, t.track_idx""", (model,)).fetchall()


def clear_visits(conn: sqlite3.Connection) -> None:
    """Drop all visit rows and un-stamp every detection -- visits.py does a full rebuild (the
    detection set is small and a from-scratch pass is simpler/safer than incremental upkeep)."""
    conn.execute("DELETE FROM visits")
    conn.execute("UPDATE detections SET visit_id = NULL WHERE visit_id IS NOT NULL")
    conn.commit()


def insert_visit(conn: sqlite3.Connection, *, source: str, species: Optional[str],
                 individual_id: Optional[str], started_at: str, ended_at: str,
                 detection_count: int, max_confidence: Optional[float],
                 representative_detection_id: Optional[int],
                 species_margin: Optional[float] = None,
                 sighting_conflict: Optional[bool] = None) -> int:
    """Phase 4: write one collapsed visit event; returns its new id.

    `species_margin` (how decisive the species vote was) and `sighting_conflict` (a human logged
    conflicting names over this span) are optional so every existing caller keeps working and a
    visit written without them is honestly NULL rather than falsely confident."""
    cur = conn.execute(
        """
        INSERT INTO visits (source, species, individual_id, started_at, ended_at,
                            detection_count, max_confidence, representative_detection_id,
                            species_margin, sighting_conflict)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (source, species, individual_id, started_at, ended_at, int(detection_count),
         None if max_confidence is None else float(max_confidence),
         None if representative_detection_id is None else int(representative_detection_id),
         None if species_margin is None else float(species_margin),
         None if sighting_conflict is None else int(bool(sighting_conflict))),
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
    if "model_species" not in cols:
        # The model's ORIGINAL auto-prediction, preserved so a later human CORRECTION -- which
        # overwrites species and forces species_confidence=1.0 -- doesn't destroy what the
        # classifier actually said. Without this, a corrected crop can never grade the model, which
        # was eval.py's single biggest data gap (only the confirmed/rejected slice was gradable).
        # set_species snapshots it at classify time; correct_species / apply_visit_label preserve it.
        conn.execute("ALTER TABLE detections ADD COLUMN model_species TEXT")
        conn.execute("ALTER TABLE detections ADD COLUMN model_species_confidence REAL")
        conn.execute("ALTER TABLE detections ADD COLUMN model_species_source TEXT")
        # Backfill from every row whose species is STILL the model's (never human-overwritten):
        # snapshot its prediction now so a future correction keeps it. Rows a human already corrected
        # had their prediction destroyed in place and are unrecoverable -- left NULL, honestly.
        conn.execute(
            "UPDATE detections SET model_species = species, "
            "model_species_confidence = species_confidence, "
            "model_species_source = species_source "
            "WHERE species IS NOT NULL AND (species_source IS NULL OR species_source != 'human')")
    # Soft prune (2026-07-17): the rolling-window pruner marks pruned_at instead of deleting the
    # clips row, so clip_tracks / clip_track_embeddings / individual links survive the disk budget
    # (June 2026's 550 tracklet vectors + all Notch/Elliot track links were cascade-lost this way).
    clip_cols = {r[1] for r in conn.execute("PRAGMA table_info(clips)")}
    if clip_cols and "pruned_at" not in clip_cols:
        conn.execute("ALTER TABLE clips ADD COLUMN pruned_at TEXT")
    # clip_tracks grew multi-animal + gait columns (2026-06-11, phase 4 part 2).
    ct_cols = {r[1] for r in conn.execute("PRAGMA table_info(clip_tracks)")}
    if "track_idx" not in ct_cols:
        conn.execute("ALTER TABLE clip_tracks ADD COLUMN track_idx INTEGER NOT NULL DEFAULT 0")
    for col in ("stride_hz", "stride_strength", "walk_s"):
        if col not in ct_cols:
            conn.execute(f"ALTER TABLE clip_tracks ADD COLUMN {col} REAL")
    # Un-blending (2026-06-16): a tracklet can be assigned to an individual (clip-space label,
    # distinct from detections.individual_id which is the still-space label).
    if "individual_id" not in ct_cols:
        conn.execute("ALTER TABLE clip_tracks ADD COLUMN individual_id TEXT")
    if "individual_source" not in ct_cols:
        conn.execute("ALTER TABLE clip_tracks ADD COLUMN individual_source TEXT")
    if conn.execute("SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='clip_track_embeddings'").fetchone():
        if "rep_crop" not in {r[1] for r in conn.execute("PRAGMA table_info(clip_track_embeddings)")}:
            conn.execute("ALTER TABLE clip_track_embeddings ADD COLUMN rep_crop TEXT")
    # Label integrity (2026-08-05, docs/identity-eval-2026-08-05.md phase 0 + C3/C5). All FOUR are
    # nullable adds backfilled to NULL -- no existing row is rewritten and no reader may assume a
    # value is present (see each column's comment in SCHEMA).
    if "labelled_at" not in cols:
        # WHEN a label was applied, as opposed to when the animal was seen. Deliberately NOT
        # backfilled: there is no honest value for the 21,110 detections labelled before this
        # existed, and inventing one (e.g. the observation time) would silently produce a
        # label-velocity curve that is really the visit curve -- the exact confusion the column
        # exists to end.
        conn.execute("ALTER TABLE detections ADD COLUMN labelled_at TEXT")
    v_cols = {r[1] for r in conn.execute("PRAGMA table_info(visits)")}
    if v_cols and "species_margin" not in v_cols:
        conn.execute("ALTER TABLE visits ADD COLUMN species_margin REAL")
    if v_cols and "sighting_conflict" not in v_cols:
        conn.execute("ALTER TABLE visits ADD COLUMN sighting_conflict INTEGER")
    ls_cols = {r[1] for r in conn.execute("PRAGMA table_info(live_sightings)")}
    if ls_cols and "superseded_at" not in ls_cols:
        conn.execute("ALTER TABLE live_sightings ADD COLUMN superseded_at TEXT")
    if ls_cols and "superseded_by" not in ls_cols:
        conn.execute("ALTER TABLE live_sightings ADD COLUMN superseded_by INTEGER")
    # Label attribution (2026-08-08). WHO applied a human verdict, not just that one was applied.
    # Nullable and never backfilled: every existing label was the operator's, but writing a name
    # onto 21k historical rows would fake a provenance trail that was never recorded -- NULL reads
    # as "the operator, before attribution existed". This single column is what lets the household
    # become a second label supply someday (a viewer's verdicts arrive attributed and reviewable,
    # never anonymously mixed into ground truth); without it that whole design space is closed.
    if "labeled_by" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN labeled_by TEXT")
    if ls_cols and "labeled_by" not in ls_cols:
        conn.execute("ALTER TABLE live_sightings ADD COLUMN labeled_by TEXT")
    # Soft suppression (2026-08-07, docs/refimg-design-2026-08-07.md section 7). Four nullable adds,
    # backfilled to NULL, so every row already in the DB reads as LIVE and every existing query keeps
    # working untouched -- exactly the clips.pruned_at pattern. The columns exist ahead of any
    # consumer honouring them, on purpose: the veto ships in shadow mode and writes them for a week
    # before anything reads them. (reference_images / view_epochs need no ALTER path -- they are new
    # tables, created by SCHEMA's CREATE TABLE IF NOT EXISTS on the same connect.)
    for col, decl in (("suppressed_at", "TEXT"), ("suppressed_by", "TEXT"),
                      ("suppress_ref_id", "INTEGER"), ("suppress_detail", "TEXT")):
        if col not in cols:
            conn.execute(f"ALTER TABLE detections ADD COLUMN {col} {decl}")


def set_species(conn: sqlite3.Connection, detection_id: int, species: str,
                confidence: float, source: str = "bioclip") -> None:
    """Phase 2: write an auto-classified species + score onto a detection. `source` records WHICH
    automatic stage decided it -- 'bioclip' (the species namer) or 'clip-filter' (the general-CLIP
    non-animal gate in clipfilter.py). Human corrections go through correct_species (source
    'human', verified) and are never overwritten by either stage.

    Also snapshots the model's call into model_species/_confidence/_source -- the classifier's word
    on this crop, kept intact so a later human correction can still be graded against what the model
    predicted (eval.py). classify.py skips already-verified crops, so this never clobbers a human
    label."""
    conn.execute(
        "UPDATE detections SET species = ?, species_confidence = ?, species_source = ?, "
        "model_species = ?, model_species_confidence = ?, model_species_source = ? "
        "WHERE id = ?",
        (species, float(confidence), source, species, float(confidence), source, int(detection_id)),
    )


def set_species_verified(conn: sqlite3.Connection, detection_id: int, verified) -> None:
    """Dashboard review: mark a detection's species confirmed (1), wrong (0), or unreviewed (None)."""
    conn.execute("UPDATE detections SET species_verified = ? WHERE id = ?",
                 (verified, int(detection_id)))
    conn.commit()


def correct_species(conn: sqlite3.Connection, detection_id: int, species: str) -> None:
    """Dashboard correction: set a human-chosen species (confirmed, full confidence, source=human).

    Preserves the model's prediction into model_species first (COALESCE, so an already-snapshotted
    value from set_species is kept, not clobbered by an intermediate correction), so the model can
    still be graded on this crop even though its live species is now the human's answer."""
    conn.execute(
        "UPDATE detections SET "
        "model_species = COALESCE(model_species, species), "
        "model_species_confidence = COALESCE(model_species_confidence, species_confidence), "
        "model_species_source = COALESCE(model_species_source, species_source), "
        "species = ?, species_confidence = 1.0, species_verified = 1, "
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
                        min_confidence: float, redo: bool, limit: int = 0,
                        visit_ids=None):
    """Animal crops that should get an appearance vector: above the usability gate, optionally
    restricted to one species. Unless --redo, rows already embedded for `model` are skipped.
    `visit_ids` (a set) restricts to those visits' detections -- the --co-present widening pass
    embeds LOW-confidence crops only where a second animal is plausible, so the global gate
    keeps meaning what it always meant everywhere else."""
    where = "detection_class = 'animal' AND confidence >= ?"
    params: list = [min_confidence]
    if species:
        where += " AND species = ?"
        params.append(species)
    rows = conn.execute(
        f"SELECT id, crop_path, visit_id FROM detections WHERE {where} ORDER BY id", params
    ).fetchall()
    if visit_ids is not None:
        wanted = set(visit_ids)
        rows = [r for r in rows if r[2] in wanted]
    rows = [(r[0], r[1]) for r in rows]
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
    """Phase 3: assign (or clear, with None) the individual a crop belongs to. Stamps labelled_at
    (WHEN, as opposed to every other timestamp here, which is a WHEN-SEEN)."""
    conn.execute("UPDATE detections SET individual_id = ?, individual_source = ?, labelled_at = ? "
                 "WHERE id = ?",
                 (individual_id, None if individual_id is None else source, now_local_iso(),
                  int(detection_id)))
    conn.commit()


def set_individual_bulk(conn: sqlite3.Connection, detection_ids: Sequence[int],
                        individual_id: Optional[str], source: Optional[str] = "human") -> int:
    """Assign one individual to many crops at once (naming a whole cluster). Returns the count.
    `source` records who decided: 'human' (a confirmation -- feeds the suggestion templates) or
    'cluster' (a reid.py look-alike proposal -- never feeds back into suggestions). Stamps
    labelled_at with the time of the write."""
    src = None if individual_id is None else source
    at = now_local_iso()
    conn.executemany("UPDATE detections SET individual_id = ?, individual_source = ?, "
                     "labelled_at = ? WHERE id = ?",
                     [(individual_id, src, at, int(i)) for i in detection_ids])
    conn.commit()
    return len(detection_ids)


def rename_individual(conn: sqlite3.Connection, old: str, new: Optional[str]) -> int:
    """Rename every crop labelled `old` to `new` (the dashboard's naming action). Naming two
    groups the same `new` MERGES them -- that's a feature: several look-alike clusters often turn
    out to be one animal. `new=None` clears the label back to unassigned. Returns rows changed.
    A rename is a human act: the renamed crops become source='human' (a cleared label loses its
    source too) and labelled_at moves to now -- the name on these crops was decided today."""
    cur = conn.execute(
        "UPDATE detections SET individual_id = ?, individual_source = ?, labelled_at = ? "
        "WHERE individual_id = ?",
        (new, None if new is None else "human", now_local_iso(), old))
    conn.commit()
    return cur.rowcount


def is_group_label(name) -> bool:
    """True when an individual label names a GROUP rather than one animal -- the family-stamp
    convention ("Stan + Kits"): one archive label deliberately written across a span that held
    several bodies. The marker is a literal " + " in the name, the same separator the labels were
    typed with.

    A group label is a real identity everywhere a HUMAN reads (the cast strip, profiles, the visit
    log) and is NEVER single-animal evidence anywhere a MATCHER learns: templates, ranking, the
    auto-assign floor and the eval corpus all refuse it, because the prototype of a mom-with-kits
    span is a blend of several animals that matches none of them (measured 2026-08-08: 4 of the 9
    then-confirmed group visits carried 0-1 co-present frames, i.e. the stills alone would have
    certified them solo, and 3 cleared the prototype gates -- one nightly embed away from a
    "Pedro + Kits" pseudo-individual competing in rank_templates). Mirrors refcam's
    identity_scope='group' for reference media, as a naming convention rather than a column,
    because these labels already exist in the archive and a convention needs no migration."""
    return " + " in str(name or "")


def label_visit(conn: sqlite3.Connection, visit_id: int, individual_id: Optional[str],
                source: str = "human", *, reject: bool = False,
                labeled_by: Optional[str] = None) -> int:
    """Confirm WHO a visit was: stamp `individual_id` onto the visit's detections that match the
    visit's dominant species (a stray mid-visit crow crop keeps its own identity), and mirror it
    onto the visits row. The visit is the labelling unit -- one solo animal per visit, so one
    confirmation labels every crop AND makes the visit a template for future suggestions.
    `individual_id=None` clears. Returns crops stamped.

    `source` records who decided: 'human' feeds the suggestion templates, 'auto' (the nightly
    auto-assign) never does. `reject=True` (with individual_id=None) is the human's "leave this
    unnamed" verdict: the id clears but individual_source keeps `source` ('human'), which is the
    tombstone the auto-assign pass respects -- without it, clearing a wrong auto name would just
    invite the next nightly run to stamp it again.

    Stamps detections.labelled_at with the time of THIS write -- including a clear/reject, whose
    time is exactly what a confirmation-bias measurement wants (see the column's SCHEMA comment)."""
    v = conn.execute("SELECT species FROM visits WHERE id = ?", (int(visit_id),)).fetchone()
    if v is None:
        return 0
    sp = v[0]
    where = "visit_id = ?" + ("" if sp is None else " AND species = ?")
    params = [int(visit_id)] + ([] if sp is None else [sp])
    cur = conn.execute(
        f"UPDATE detections SET individual_id = ?, individual_source = ?, labelled_at = ?, "
        f"labeled_by = ? WHERE {where}",
        [individual_id, None if (individual_id is None and not reject) else source,
         now_local_iso(), labeled_by] + params)
    conn.execute("UPDATE visits SET individual_id = ? WHERE id = ?",
                 (individual_id, int(visit_id)))
    conn.commit()
    return cur.rowcount


_UNSET = object()   # sentinel: "argument not provided" (distinct from None = "clear the label")


def apply_visit_label(conn: sqlite3.Connection, *, visit_id: Optional[int] = None,
                      source: Optional[str] = None, start: Optional[str] = None,
                      end: Optional[str] = None, name=_UNSET, species: Optional[str] = None,
                      verify: bool = False, labeled_by: Optional[str] = None) -> dict:
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

    Species is applied first, so naming after a correction scopes to the corrected species. Naming
    also stamps detections.labelled_at (WHEN the label was applied -- record_live_sighting's solo
    stamp routes through here, so live confirmations are timestamped too). Any
    `visits` rows the detections belong to are synced (species/individual_id) so the queue and the
    Explorer agree. Returns a small summary dict. Operates via WHERE clauses (not id lists) so a
    1000-crop visit never hits SQLite's bound-variable limit."""
    if visit_id is not None:
        where, params = "visit_id = ?", [int(visit_id)]
    elif source and start and end:
        # Resolve the span to explicit detection ids by INSTANT, not lexical string order: ISO
        # timestamps with different UTC offsets (a DST seam inside one visit) don't sort by instant,
        # so a lexical BETWEEN would grab the wrong rows at the fall-back/spring-forward hour. A
        # visit is small, so an id list is fine. Pad the coarse lexical pre-filter by a day each
        # side (offsets are <= 14 h) to guarantee a superset, then filter exactly via parse_local.
        lo, hi = parse_local(start), parse_local(end)
        if lo is None or hi is None:
            return {"detections": 0, "error": "bad start/end timestamp"}
        lo_pad, hi_pad = (lo - timedelta(days=1)).isoformat(), (hi + timedelta(days=1)).isoformat()
        cand = conn.execute(
            "SELECT id, timestamp FROM detections "
            "WHERE source = ? AND timestamp >= ? AND timestamp <= ?",
            (source, lo_pad, hi_pad)).fetchall()
        ids = [row[0] for row in cand
               if (p := parse_local(row[1])) is not None and lo <= p <= hi]
        if not ids:
            return {"detections": 0}
        where, params = "id IN (%s)" % ",".join("?" * len(ids)), list(ids)
    else:
        return {"detections": 0, "error": "need visit_id or source+start+end"}

    n = conn.execute(f"SELECT COUNT(*) FROM detections WHERE {where}", params).fetchone()[0]
    if not n:
        return {"detections": 0}

    if species:
        # Preserve each crop's model prediction before overwriting (COALESCE keeps an existing
        # snapshot), the whole-visit twin of correct_species -- so a bulk visit correction stays
        # gradable against what the classifier said. The COALESCE reads the row's own columns, so
        # no extra bound params: species for SET, `params` for the WHERE, unchanged.
        conn.execute(
            f"UPDATE detections SET "
            f"model_species = COALESCE(model_species, species), "
            f"model_species_confidence = COALESCE(model_species_confidence, species_confidence), "
            f"model_species_source = COALESCE(model_species_source, species_source), "
            f"species = ?, species_confidence = 1.0, species_verified = 1, "
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
        at = now_local_iso()      # WHEN the label was applied (detections.labelled_at)
        if dominant is None:
            conn.execute(f"UPDATE detections SET individual_id = ?, individual_source = ?, "
                         f"labelled_at = ?, labeled_by = ? WHERE {where}",
                         [name, src, at, labeled_by] + params)
        else:
            conn.execute(f"UPDATE detections SET individual_id = ?, individual_source = ?, "
                         f"labelled_at = ?, labeled_by = ? WHERE {where} AND species = ?",
                         [name, src, at, labeled_by] + params + [dominant])

    # Sync the visits-table rows these detections belong to (subquery, no big IN list).
    vsub = f"id IN (SELECT DISTINCT visit_id FROM detections WHERE {where} AND visit_id IS NOT NULL)"
    if species:
        conn.execute(f"UPDATE visits SET species = ? WHERE {vsub}", [species] + params)
    if name is not _UNSET:
        conn.execute(f"UPDATE visits SET individual_id = ? WHERE {vsub}", [name] + params)
    conn.commit()
    return {"detections": int(n), "dominant_species": dominant,
            "species_set": species or None, "named": None if name is _UNSET else name}


def visit_labels_by_source(conn: sqlite3.Connection, source: str,
                           species: Optional[str] = None) -> dict:
    """{visit_id: individual_id} for every visit whose crops carry a label of `individual_source`
    = `source` ('human' = confirmations, 'auto' = the nightly auto-assign pass). Read from
    DETECTIONS, not visits -- visits.py rebuilds from scratch, but the stamped crops persist, so
    labels survive a rebuild. Dominant name per visit."""
    where = "individual_source = ? AND individual_id IS NOT NULL AND visit_id IS NOT NULL"
    params: list = [source]
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


def confirmed_visit_labels(conn: sqlite3.Connection, species: Optional[str] = None) -> dict:
    """{visit_id: individual_id} for every visit with human-confirmed crops (the suggestion
    templates; auto-assigned names deliberately don't qualify)."""
    return visit_labels_by_source(conn, "human", species)


def rejected_visit_ids(conn: sqlite3.Connection, species: Optional[str] = None) -> set:
    """Visit ids a human explicitly left UNNAMED (label_visit(..., None, reject=True): id NULL but
    individual_source 'human'). The auto-assign pass skips these -- a human already looked."""
    where = "individual_source = 'human' AND individual_id IS NULL AND visit_id IS NOT NULL"
    params: list = []
    if species:
        where += " AND species = ?"
        params.append(species)
    return {r[0] for r in conn.execute(
        f"SELECT DISTINCT visit_id FROM detections WHERE {where}", params)}


# ---------------------------------------------------------------------------
# THE ROSTER: who still lives here. See the individual_status table comment for why this is a
# human-entered fact and not something the matcher could ever infer for itself.
# ---------------------------------------------------------------------------

INDIVIDUAL_STATUSES = ("resident", "departed")


def as_date(value) -> Optional[str]:
    """Normalize a user/API-supplied day to 'YYYY-MM-DD', accepting either a bare date or any of
    the project's ISO timestamps (whose first 10 characters ARE the local calendar day -- the same
    slice every day-keyed surface in this codebase uses). None/'' -> None. Raises ValueError on
    anything else, so a typo becomes a 400 rather than a silently inert guard."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    head = s[:10]
    try:
        datetime.strptime(head, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"date must be YYYY-MM-DD, got {value!r}")
    return head


def set_individual_status(conn: sqlite3.Connection, name: str, *, status: str = "departed",
                          effective_date=None, note: Optional[str] = None,
                          updated_at: Optional[str] = None) -> dict:
    """Record what the human knows about an individual's residency: 'departed' (with the last day
    it was here) or 'resident' (the default state, written back to UNDO a departure).

    The row is kept either way rather than deleted on 'resident' -- updated_at then says when the
    call was reversed, and an empty table stays honest about meaning "nobody has said anything",
    not "everyone was checked and is here". Returns the stored row."""
    nm = str(name or "").strip()
    if not nm:
        raise ValueError("name is required")
    st = str(status or "").strip().lower()
    if st not in INDIVIDUAL_STATUSES:
        raise ValueError(f"status must be one of {', '.join(INDIVIDUAL_STATUSES)}")
    day = as_date(effective_date)
    note_clean = (str(note).strip() or None) if note else None
    at = updated_at or now_local_iso()
    conn.execute(
        """INSERT INTO individual_status (name, status, effective_date, note, updated_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET status = excluded.status,
                                           effective_date = excluded.effective_date,
                                           note = excluded.note,
                                           updated_at = excluded.updated_at""",
        (nm, st, day, note_clean, at))
    conn.commit()
    return {"name": nm, "status": st, "effective_date": day, "note": note_clean, "updated_at": at}


def individual_statuses(conn: sqlite3.Connection) -> dict:
    """{name: {name, status, effective_date, note, updated_at}} for every individual a human has
    said something about. Empty dict when the table doesn't exist -- a read-only clone of a DB no
    writer has migrated must not explode on a reporting surface (same contract as
    _live_sighting_rows)."""
    try:
        rows = conn.execute("SELECT name, status, effective_date, note, updated_at "
                            "FROM individual_status").fetchall()
    except sqlite3.OperationalError:
        return {}
    out = {}
    for r in rows:
        nm, st, day, note, at = tuple(r)[:5]
        out[nm] = {"name": nm, "status": st, "effective_date": day, "note": note,
                   "updated_at": at}
    return out


def record_coverage(db_path, source: str, event: str, reason: Optional[str] = None) -> None:
    """Append one coverage transition ('up'/'down') for `source`. Called from the CAPTURE THREAD
    at its rare transitions, so it must never hurt the rig: its own short-lived connection, and
    every failure -- locked DB, missing table, anything -- is swallowed. Losing one coverage row
    is a rounding error; stalling a capture thread on the WAL lock is a documented incident
    class (2026-06-30)."""
    try:
        conn = connect(db_path)
        try:
            conn.execute("INSERT INTO coverage_events (source, event, at, reason) "
                         "VALUES (?, ?, ?, ?)",
                         (source, "up" if event == "up" else "down", now_local_iso(), reason))
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def coverage_dark_seconds(conn: sqlite3.Connection, source: str, start, end) -> Optional[float]:
    """Seconds within [start, end] (tz-aware datetimes) this camera was NOT watching, from the
    coverage ledger. None when the ledger has no events at or before the window -- 'unknown' and
    'fully covered' must never be conflated (the whole point of the table is honest absence).
    A trailing 'down' with no later 'up' counts as dark through the window's end; a crash that
    never wrote 'down' therefore reads as covered, which is the accepted error direction (the
    ledger flags known gaps; it cannot conjure unknown ones)."""
    try:
        rows = conn.execute(
            "SELECT event, at FROM coverage_events WHERE source = ? ORDER BY at", (source,)
        ).fetchall()
    except sqlite3.OperationalError:
        return None
    events = [(r[0], parse_local(r[1])) for r in rows]
    events = [(e, t) for e, t in events if t is not None]
    if not events or events[0][1] > start:
        return None                       # nothing at or before the window opens -> unknown
    state, dark, cursor = None, 0.0, start
    for e, t in events:
        if t <= start:
            state = e
            continue
        if t > end:
            break
        if state == "down":
            dark += max(0.0, (min(t, end) - cursor).total_seconds())
        cursor = max(cursor, t)
        state = e
    if state == "down":
        dark += max(0.0, (end - cursor).total_seconds())
    return dark


def add_life_event(conn: sqlite3.Connection, name: str, note: str, *,
                   event_date=None, labeled_by: Optional[str] = None) -> dict:
    """Append one dated event to an individual's story (the life_events ledger). Free text is
    the point -- 'Stan's kits first emerged', 'favouring the left front paw'. Append-only by
    design; a wrong note is corrected by another note, the way a field notebook is."""
    nm = str(name or "").strip()
    note_clean = str(note or "").strip()
    if not nm or not note_clean:
        raise ValueError("name and note are required")
    day = as_date(event_date)
    at = now_local_iso()
    cur = conn.execute(
        "INSERT INTO life_events (name, event_date, note, labeled_by, created_at) "
        "VALUES (?, ?, ?, ?, ?)", (nm, day, note_clean[:500], labeled_by, at))
    conn.commit()
    return {"id": int(cur.lastrowid), "name": nm, "event_date": day, "note": note_clean[:500],
            "labeled_by": labeled_by, "created_at": at}


def life_events(conn: sqlite3.Connection, name: Optional[str] = None) -> list:
    """The event ledger, newest-dated first (undated notes sort by created_at). One name or all.
    Empty on a DB no writer has migrated yet -- the read-only-clone contract."""
    try:
        if name:
            rows = conn.execute(
                "SELECT id, name, event_date, note, labeled_by, created_at FROM life_events "
                "WHERE name = ? ORDER BY COALESCE(event_date, substr(created_at,1,10)) DESC, id DESC",
                (str(name).strip(),)).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, name, event_date, note, labeled_by, created_at FROM life_events "
                "ORDER BY COALESCE(event_date, substr(created_at,1,10)) DESC, id DESC").fetchall()
    except sqlite3.OperationalError:
        return []
    return [{"id": r[0], "name": r[1], "event_date": r[2], "note": r[3],
             "labeled_by": r[4], "created_at": r[5]} for r in rows]


def departed_individuals(conn: sqlite3.Connection) -> dict:
    """{casefolded name: effective_date or None} for the individuals a human has marked departed.
    Case-folded because the names are free text typed by hand and "notch" is "Notch"; the value is
    the last day the animal was resident (None = no date given, i.e. always departed)."""
    return {str(s["name"]).strip().casefold(): s["effective_date"]
            for s in individual_statuses(conn).values() if s["status"] == "departed"}


_MAX_SIGHTING_NAMES = 12   # a "who's here now" log of more than a dozen named animals isn't real.


def _sighting_names(names_json) -> list:
    """Decode a live_sightings.names JSON array, tolerating a corrupt/NULL value (-> [])."""
    try:
        return json.loads(names_json) if names_json else []
    except (ValueError, TypeError):
        return []


def _name_key(names) -> frozenset:
    """Case-insensitive identity of a name SET -- the unit two sightings are compared on. Order and
    case are noise ("stan" typed twice is one claim); membership is the claim."""
    return frozenset(str(n).strip().casefold() for n in (names or []) if str(n).strip())


def _live_sighting_rows(conn: sqlite3.Connection, source: Optional[str] = None) -> list:
    """Every live sighting as a dict, oldest first, with its span resolved by the module's
    convention (a row with no span falls back to its observed_at instant). Empty list on a DB
    no writer has migrated yet -- reporting surfaces must not explode on a read-only clone."""
    where, params = ("WHERE source = ?", [source]) if source else ("", [])
    base = "SELECT id, source, observed_at, span_start, span_end, names, stamped"
    rows, has_supersede = None, True
    for cols, has_supersede in ((base + ", superseded_at, superseded_by", True), (base, False)):
        try:
            rows = conn.execute(f"{cols} FROM live_sightings {where} ORDER BY id", params).fetchall()
            break
        except sqlite3.OperationalError:
            # First failure: a READ-ONLY connection to a DB no writer has migrated yet -- the table
            # exists but the supersede columns don't, and a read-only conn cannot add them. Retry
            # without them (every row then reads as never-superseded, which is the truth for a DB
            # that has never run the supersede path). Second failure: no live_sightings at all.
            rows = None
    if rows is None:
        return []
    out = []
    for r in rows:
        t = tuple(r)
        sid, src, observed, s0, s1, names_json, stamped = t[:7]
        sup_at, sup_by = (t[7], t[8]) if has_supersede else (None, None)
        names = _sighting_names(names_json)
        out.append({"id": sid, "source": src, "observed_at": observed,
                    "span_start": s0, "span_end": s1, "names": names,
                    "key": _name_key(names), "stamped": stamped,
                    "start": s0 or observed, "end": s1 or observed,
                    "superseded_at": sup_at, "superseded_by": sup_by})
    return out


def record_live_sighting(conn: sqlite3.Connection, *, source: str, names,
                         span_start: Optional[str] = None, span_end: Optional[str] = None,
                         note: Optional[str] = None, observed_at: Optional[str] = None,
                         stamp: bool = True, labeled_by: Optional[str] = None) -> dict:
    """Log a human's real-time identification of who is visiting NOW (the dashboard Live tab).
    `names` is the list of individuals present over [span_start, span_end] on `source`.

    SOLO (exactly one name): ALSO stamp that individual_id onto the span's crops via
    apply_visit_label -- the live equivalent of confirming a solo visit, so the appearance
    templates grow from it. PAIR (two or more): record the co-presence note ONLY, with NO stamp
    -- one name on two animals mislabels both (the pair gotcha); the names ride here as ground
    truth the un-blend step consumes. De-dupes names case-insensitively, keeping first-seen order.

    RE-LOGGING A SPAN SUPERSEDES IT (2026-08-05). Live sightings are typed in the moment and get
    corrected in the moment: ids 25/26/27 logged "Clippy", then "Clippy Friend", then "Stan" over
    the SAME two crops within 33 seconds, each solo stamp overwriting the last, so the stored label
    was simply whatever was typed last and the DB recorded nothing about the correction. Now every
    still-live sighting whose span overlaps this one on this source is MARKED superseded (kept, not
    deleted -- it is ground truth, and the correction sequence is itself signal), and if any of them
    named someone DIFFERENT the return says so: conflicting names over one span mean either two
    animals or a human changing their mind, and the data cannot tell which, so the overlapping
    visits are flagged multi-animal (visits.sighting_conflict, stamped by visits.build_visits)
    rather than trusted as a single-animal template.

    The stamping asymmetry is deliberately UNCHANGED: the newest solo name still stamps the span
    (the human's latest word is their best word), and a pair still stamps nothing.

    GROUP STRING (2026-08-08): one name that is itself a group label ("Stan + Kits" --
    is_group_label) both STAMPS and counts as MULTI. The stamp is the family convention: the whole
    span belongs to that household on the archive, and per-kit names don't exist to type. But the
    span held several bodies, so the sighting must feed every multi-animal signal
    (individuals.multi_name_sighting_spans, embed --co-present, the is_multi badge) exactly as a
    two-name log does -- before this, a family night's sighting read as SOLO here, and its blended
    prototype was one nightly embed away from becoming a "Pedro + Kits" template.

    `stamp=False` records the sighting as testimony WITHOUT the solo stamp -- the viewer tier's
    path (web.py's operator/viewer split): a family member's "that's Stan!" lands attributed
    (`labeled_by`) and reviewable, and never writes ground truth or feeds a template until the
    operator promotes it. `labeled_by` rides on the sighting row either way.

    Returns {sighting_id, stamped, multi, group, names, superseded: [ids], conflict: bool}."""
    ordered, seen = [], set()
    for n in (names or []):
        s = str(n).strip()
        k = s.casefold()
        if s and k not in seen:
            seen.add(k)
            ordered.append(s)
        if len(ordered) >= _MAX_SIGHTING_NAMES:
            break
    if not ordered:
        return {"error": "no names", "sighting_id": None, "stamped": 0, "multi": False,
                "names": [], "superseded": [], "conflict": False}

    observed = observed_at or now_local_iso()
    s0, s1 = span_start or observed, span_end or observed

    # Who is this re-logging? Every not-yet-superseded sighting on this source whose span overlaps.
    prior = [p for p in _live_sighting_rows(conn, source)
             if p["superseded_at"] is None and _spans_overlap(p["start"], p["end"], s0, s1)]
    key = _name_key(ordered)
    conflict = any(p["key"] != key for p in prior)

    multi = len(ordered) > 1
    group = (not multi) and is_group_label(ordered[0])
    stamped = 0
    # Solo => stamp the span (feeds re-ID). Pair => never stamp a single name across two animals.
    # A GROUP string is the solo path's stamp with the pair path's meaning: it stamps (the family
    # convention) AND reports multi below (several bodies -- the sighting must feed is_multi).
    # A viewer's log (stamp=False) records testimony only.
    if stamp and not multi and span_start and span_end:
        res = apply_visit_label(conn, source=source, start=span_start, end=span_end,
                                name=ordered[0], labeled_by=labeled_by)
        stamped = int(res.get("detections") or 0)

    note_clean = (str(note).strip() or None) if note else None
    cur = conn.execute(
        "INSERT INTO live_sightings (source, observed_at, span_start, span_end, names, stamped, "
        "note, labeled_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (source, observed, span_start, span_end, json.dumps(ordered), int(stamped), note_clean,
         labeled_by))
    sid = int(cur.lastrowid)
    if prior:
        conn.executemany(
            "UPDATE live_sightings SET superseded_at = ?, superseded_by = ? WHERE id = ?",
            [(observed, sid, int(p["id"])) for p in prior])
    conn.commit()
    return {"sighting_id": sid, "stamped": stamped, "multi": multi or group, "group": group,
            "names": ordered,
            "superseded": [int(p["id"]) for p in prior], "conflict": conflict}


def sighting_conflict_groups(conn: sqlite3.Connection, source: Optional[str] = None) -> list:
    """Groups of live sightings that overlap in time on one source but do NOT agree on who was
    there -- the DB's noisiest ground truth, made visible.

    DERIVED from the spans on every read, never read off a stored flag, because the known groups in
    the live DB were all logged before any flag existed. Rows are joined into a group transitively
    (a overlaps b, b overlaps c => one group) and a group is reported only when its rows carry two
    or more DISTINCT name sets.

    Measured 2026-08-05 against the live DB: SEVEN groups, not the five
    docs/identity-eval-2026-08-05.md lists. It finds all five -- (6,7), (8,9,10), (19,20),
    (25,26,27) with (24) correctly joined to it, (40,41,42) -- plus two the eval's single-name
    reading skipped, and those two are the dangerous kind: (12,13) logged the pair Stan+Pedro and
    then stamped 77 crops of that same span solo "Pedro", and (46,47) stamped 8 crops solo "Stan"
    over the span the human then logged as Stan + three kits (the 2026-07-31 kit mis-stamp). A solo
    row overlapping a PAIR row is a conflict too: one name went onto crops the human said held two
    animals.

    This is the gap `individuals.multi_name_sighting_spans` cannot see: it reads rows carrying 2+
    names, and a conflict arrives as several SINGLE-name rows. Detecting it needs the comparison
    ACROSS rows, which is what this does.

    Each group: {source, span_start, span_end, sighting_ids, names, name_sets, resolved} --
    `resolved` is True when every row but the newest carries a supersede mark, i.e. the correction
    sequence has been recorded rather than left as several equally-live claims (the "5 -> 0
    unresolved" metric in docs/identity-eval-2026-08-05.md)."""
    rows = _live_sighting_rows(conn, source)
    by_source: dict = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    groups = []
    for src, items in by_source.items():
        # Transitive overlap grouping. Sightings are few (tens), so an O(n^2) sweep is the boring
        # choice; it also stays correct with out-of-order / open-ended spans, which a sort would not.
        unassigned = list(items)
        while unassigned:
            seed = unassigned.pop(0)
            comp = [seed]
            changed = True
            while changed:
                changed = False
                for cand in list(unassigned):
                    if any(_spans_overlap(cand["start"], cand["end"], m["start"], m["end"])
                           for m in comp):
                        comp.append(cand)
                        unassigned.remove(cand)
                        changed = True
            keys = {m["key"] for m in comp}
            if len(keys) < 2:
                continue                      # everyone agreed -- a re-log, not a conflict
            comp.sort(key=lambda m: m["id"])
            # Span of the whole group by INSTANT (parse_local), so ISO strings with different UTC
            # offsets -- a DST seam inside one evening -- can't reorder them lexically. Unparseable
            # timestamps are dropped rather than guessed at.
            starts = [(p, m["start"]) for m in comp if (p := parse_local(m["start"])) is not None]
            ends = [(p, m["end"]) for m in comp if (p := parse_local(m["end"])) is not None]
            lo = min(starts)[1] if starts else None
            hi = max(ends)[1] if ends else None
            names: list = []
            seen_n: set = set()
            for m in comp:
                for n in m["names"]:
                    k = str(n).casefold()
                    if k not in seen_n:
                        seen_n.add(k)
                        names.append(n)
            groups.append({
                "source": src, "span_start": lo, "span_end": hi,
                "sighting_ids": [int(m["id"]) for m in comp],
                "names": names,
                "name_sets": [sorted(m["key"]) for m in comp],
                "resolved": all(m["superseded_at"] is not None for m in comp[:-1]),
            })
    groups.sort(key=lambda g: (g["source"], g["sighting_ids"][0]))
    return groups


def conflicting_sighting_spans(conn: sqlite3.Connection) -> list:
    """(source, span_start, span_end) per conflicting sighting GROUP -- deliberately the same shape
    `individuals.multi_name_sighting_spans` returns, so a caller can concatenate the two lists and
    treat "the human logged two different names over this span" exactly like "the human logged two
    names in one row". visits.build_visits uses it to stamp visits.sighting_conflict."""
    return [(g["source"], g["span_start"], g["span_end"])
            for g in sighting_conflict_groups(conn)
            if g["span_start"] and g["span_end"]]


def recent_live_sightings(conn: sqlite3.Connection, *, source: Optional[str] = None,
                          limit: int = 8) -> list:
    """The most-recent live sightings (newest first) for the Live tab's running log. Works on a
    read-only or read-write connection -- columns are read positionally so no row_factory is
    required."""
    where, params = "", []
    if source:
        where = "WHERE source = ?"
        params.append(source)
    try:
        rows = conn.execute(
            f"SELECT id, source, observed_at, span_start, span_end, names, stamped, note "
            f"FROM live_sightings {where} ORDER BY id DESC LIMIT ?",
            params + [int(limit)]).fetchall()
    except sqlite3.OperationalError:
        # The table is created on the first read-write connect(); a read-only dashboard pointed at
        # a DB no writer has migrated yet simply has no sightings to show.
        return []
    out = []
    for r in rows:
        sid, src, observed, sstart, send, names_json, stamped, note = tuple(r)
        try:
            names = json.loads(names_json) if names_json else []
        except (ValueError, TypeError):
            names = []
        out.append({"id": sid, "source": src, "observed_at": observed,
                    "span_start": sstart, "span_end": send, "names": names,
                    "stamped": stamped, "note": note})
    return out


def _spans_overlap(a0, a1, b0, b1) -> bool:
    """True if the two timestamp spans overlap at all, compared by INSTANT (parse_local handles
    the stored offsets / a legacy naive string)."""
    pa0, pa1, pb0, pb1 = parse_local(a0), parse_local(a1), parse_local(b0), parse_local(b1)
    if None in (pa0, pa1, pb0, pb1):
        return False
    return pa0 <= pb1 and pb0 <= pa1


def co_present_sighting_names(conn: sqlite3.Connection, source: str,
                              started_at: str, ended_at: str) -> dict:
    """Who a human logged as present during a visit's window: the union of every live sighting on
    `source` whose span overlaps [started_at, ended_at] (a sighting with no span falls back to its
    observed_at instant). This is the ground truth that seeds un-blend -- the human said WHO the two
    animals are, even before any appearance template exists. De-dupes case-insensitively, newest
    first; `observed_at` is the most recent overlapping sighting (for display). Returns {names,
    observed_at, n}. Matched by absolute time, so it survives the visit-id renumbering of a rebuild."""
    try:
        rows = conn.execute(
            "SELECT observed_at, span_start, span_end, names FROM live_sightings WHERE source = ? "
            "ORDER BY id DESC", (source,)).fetchall()
    except sqlite3.OperationalError:        # table not created yet (read-only, un-migrated DB)
        return {"names": [], "observed_at": None, "n": 0}
    names, seen, observed = [], set(), None
    for r in rows:
        observed_at_r, span_start, span_end, names_json = tuple(r)
        s0 = span_start or observed_at_r
        s1 = span_end or observed_at_r
        if not _spans_overlap(s0, s1, started_at, ended_at):
            continue
        if observed is None:
            observed = observed_at_r
        try:
            nm = json.loads(names_json) if names_json else []
        except (ValueError, TypeError):
            nm = []
        for n in nm:
            k = str(n).casefold()
            if k not in seen:
                seen.add(k)
                names.append(n)
    return {"names": names, "observed_at": observed, "n": len(names)}


# ---------------------------------------------------------------------------
# THE REFERENCE-IMAGE VETO (2026-08-07, docs/refimg-design-2026-08-07.md).
#
# A tipped watering can at the glass door fires MegaDetector as `raccoon` sixty times; a static
# bracket fired as `Anna's hummingbird` four hundred times. Comparing a box's pixels against a
# reference frame of the same view WITH NOTHING IN IT suppresses 588 of 652 such furniture
# evaluations while touching 0 of 4,649 animal evaluations -- but only as a CONJUNCTION of gates,
# every one of which was measured to be load-bearing. The bare pixel test erases raccoons (131 of
# 372 in one visit), and location recurrence alone flags the FOOD BOWL, where real raccoons have
# stood for 27 days.
#
# This module contributes only the STORAGE, and it is deliberately inert:
#   * a suppression is a FLAG on a row that was written normally (record_suppression), never a
#     dropped, hidden or altered detection. An erased animal writes no row at all and is a silent
#     permanent loss; a wrongly flagged one is a row a human can look at and clear. That asymmetry
#     is the whole reason these are columns and not a DELETE.
#   * SHADOW MODE ships first: nothing in this codebase reads suppressed_at yet. A week of flagged
#     crops goes onto one contact sheet, gets audited, and only then does any consumer opt in.
# ---------------------------------------------------------------------------

# Known `suppressed_by` values -- WHICH mechanism made the call, so a bad week is attributable to
# one of them rather than to "something suppressed this".
SUPPRESSED_BY_REFIMG_VETO = "refimg_veto"     # the reference-image veto (this design).
SUPPRESSED_BY_STATICFILTER = "staticfilter"   # the self-calibrating static-furniture sweep.

# Illumination is DERIVED FROM THE FRAME (chroma < 6 => ir; else median < 90 => night; else day),
# never from a clock, and a reference is switched between these -- never blended across them.
ILLUMINATIONS = ("day", "night", "ir")


def _detail_json(detail) -> Optional[str]:
    """Normalize a gate trace to the TEXT actually stored. A mapping/sequence is dumped; a string is
    trusted as already-JSON and stored verbatim (so a caller that built it with its own encoder is
    not double-encoded); None stays None."""
    if detail is None:
        return None
    if isinstance(detail, str):
        return detail
    return json.dumps(detail, default=str)


def record_suppression(conn: sqlite3.Connection, detection_id: int, by: str,
                       ref_id: Optional[int] = None, detail_json=None, *,
                       suppressed_at: Optional[str] = None) -> bool:
    """Flag one detection as suppressed. The row itself is NOT touched -- box, crop, species,
    identity and timestamps all stay exactly as written -- so nothing is lost and every existing
    query keeps returning it until that query opts in.

    `by` says which mechanism decided (SUPPRESSED_BY_*), `ref_id` the reference_images row it was
    compared against, `detail_json` the WHOLE decision (dict or JSON string): scores, thresholds,
    provenance, reference age, recurrence counts. That trace is the point -- a suppression that
    turns out to be a raccoon has to be diagnosable months later without re-running the veto.

    REFUSES TO OVERWRITE an existing suppression: the first mechanism to flag a row owns the
    explanation, and silently replacing it would destroy the evidence for the decision that
    actually happened. Returns True if this call flagged the row, False if it was already flagged
    (or the id doesn't exist) -- never raises for either. Use clear_suppression to un-flag first."""
    who = str(by or "").strip()
    if not who:
        raise ValueError("suppressed_by is required (which mechanism decided)")
    row = conn.execute("SELECT suppressed_at FROM detections WHERE id = ?",
                       (int(detection_id),)).fetchone()
    if row is None or row[0] is not None:
        return False
    conn.execute(
        "UPDATE detections SET suppressed_at = ?, suppressed_by = ?, suppress_ref_id = ?, "
        "suppress_detail = ? WHERE id = ? AND suppressed_at IS NULL",
        (suppressed_at or now_local_iso(), who,
         None if ref_id is None else int(ref_id), _detail_json(detail_json), int(detection_id)))
    conn.commit()
    return True


def clear_suppression(conn: sqlite3.Connection, detection_id: int) -> bool:
    """Un-flag a detection: all four suppression columns go back to NULL and the row is live again.
    This is the human's "that was an animal" verdict from the audit sheet, and it is the reason the
    veto is allowed to be wrong. Returns True if the row was suppressed and is now clear."""
    cur = conn.execute(
        "UPDATE detections SET suppressed_at = NULL, suppressed_by = NULL, "
        "suppress_ref_id = NULL, suppress_detail = NULL "
        "WHERE id = ? AND suppressed_at IS NOT NULL", (int(detection_id),))
    conn.commit()
    return cur.rowcount > 0


def save_reference_image(conn: sqlite3.Connection, *, source: str, illumination: str,
                         view_epoch: int, provenance: str, image_path: str,
                         captured_at: Optional[str] = None, cover_path: Optional[str] = None,
                         edge_fp: Optional[bytes] = None, n_frames: Optional[int] = None,
                         span_s: Optional[float] = None) -> int:
    """Store one reference image ("this view with nothing in it"); returns its new id.

    `cover_path` is the mask of pixels the reference actually KNOWS -- NULL means fully covered, and
    anything else means the veto must abstain outside the mask. An unknown pixel is not evidence of
    emptiness: a detector-certified frame was measured to contain an undetected raccoon walking the
    wall, and the motion mask is what keeps that miss costing an abstention instead of an erasure.

    `illumination` is validated against ILLUMINATIONS because it is a lookup KEY -- a typo would
    make latest_reference silently return nothing forever, the stale-guard-fails-silently failure
    this project has already been bitten by."""
    illum = str(illumination or "").strip().lower()
    if illum not in ILLUMINATIONS:
        raise ValueError(f"illumination must be one of {', '.join(ILLUMINATIONS)}, got {illumination!r}")
    if not str(source or "").strip():
        raise ValueError("source is required")
    if not str(provenance or "").strip():
        raise ValueError("provenance is required (how this reference was built)")
    cur = conn.execute(
        """INSERT INTO reference_images (source, illumination, view_epoch, captured_at, provenance,
                                         image_path, cover_path, edge_fp, n_frames, span_s)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (source, illum, int(view_epoch), captured_at or now_local_iso(), provenance,
         image_path, cover_path, edge_fp,
         None if n_frames is None else int(n_frames),
         None if span_s is None else float(span_s)))
    conn.commit()
    return int(cur.lastrowid)


def _reference_row(row) -> dict:
    """One reference_images row as a plain dict (read positionally, so no row_factory required)."""
    (rid, source, illumination, view_epoch, captured_at, provenance, image_path,
     cover_path, edge_fp, n_frames, span_s, retired_at) = tuple(row)[:12]
    return {"id": rid, "source": source, "illumination": illumination,
            "view_epoch": view_epoch, "captured_at": captured_at, "provenance": provenance,
            "image_path": image_path, "cover_path": cover_path, "edge_fp": edge_fp,
            "n_frames": n_frames, "span_s": span_s, "retired_at": retired_at}


_REFIMG_COLS = ("id, source, illumination, view_epoch, captured_at, provenance, image_path, "
                "cover_path, edge_fp, n_frames, span_s, retired_at")


def latest_reference(conn: sqlite3.Connection, source: str, illumination: str,
                     epoch: int) -> Optional[dict]:
    """The newest LIVE (not retired) reference for exactly this (source, illumination, view_epoch),
    as a dict -- or None, which is a perfectly normal answer and means the veto abstains.

    Ordered by id, not by captured_at: the writer inserts in capture order, and ISO timestamps
    carrying different UTC offsets do not sort by instant (the DST-seam trap this module handles
    with parse_local everywhere it compares times). The age gate is the caller's job -- captured_at
    is returned so it can apply the 2 h limit itself.

    Returns None on a DB whose writer has never migrated (the read-only-clone contract shared with
    individual_statuses / _live_sighting_rows)."""
    illum = str(illumination or "").strip().lower()
    try:
        row = conn.execute(
            f"SELECT {_REFIMG_COLS} FROM reference_images "
            f"WHERE source = ? AND illumination = ? AND view_epoch = ? AND retired_at IS NULL "
            f"ORDER BY id DESC LIMIT 1", (source, illum, int(epoch))).fetchone()
    except sqlite3.OperationalError:
        return None
    return None if row is None else _reference_row(row)


def reference_image(conn: sqlite3.Connection, ref_id: int) -> Optional[dict]:
    """One reference by id, RETIRED OR NOT -- the replay path for an audit: a suppression written
    last week carries suppress_ref_id, and the image that justified it must still be resolvable
    even though its epoch is long gone. That is why references are retired and never deleted."""
    try:
        row = conn.execute(f"SELECT {_REFIMG_COLS} FROM reference_images WHERE id = ?",
                           (int(ref_id),)).fetchone()
    except sqlite3.OperationalError:
        return None
    return None if row is None else _reference_row(row)


def retire_reference_images(conn: sqlite3.Connection, source: str, before_epoch: int, *,
                            retired_at: Optional[str] = None) -> int:
    """Retire every live reference on `source` from an epoch OLDER than `before_epoch` -- what the
    camera looked like empty before it was moved is not what it looks like empty now. The rows are
    kept (retired_at set, never deleted) so past suppressions stay replayable. Returns rows retired.
    Idempotent: a second call finds nothing left to mark."""
    cur = conn.execute(
        "UPDATE reference_images SET retired_at = ? "
        "WHERE source = ? AND view_epoch < ? AND retired_at IS NULL",
        (retired_at or now_local_iso(), source, int(before_epoch)))
    conn.commit()
    return cur.rowcount


def current_view_epoch(conn: sqlite3.Connection, source: str) -> int:
    """Which view epoch `source` is in now. DEFAULTS TO 0 when nothing has been recorded -- epoch 0
    is the implicit "the camera has not been seen to move" state, so a rig that has never detected a
    reposition still has a valid, stable reference key. 0 is also the answer on a DB whose writer has
    never migrated (read-only-clone contract)."""
    try:
        row = conn.execute("SELECT MAX(epoch) FROM view_epochs WHERE source = ?",
                           (source,)).fetchone()
    except sqlite3.OperationalError:
        return 0
    return 0 if row is None or row[0] is None else int(row[0])


def bump_view_epoch(conn: sqlite3.Connection, source: str, detected_by: str = "edge_fp_corr",
                    corr: Optional[float] = None, *, started_at: Optional[str] = None) -> int:
    """Record that `source` has moved: allocate the next epoch, write the event, and RETIRE every
    reference from the older epochs in the same call -- so a stale reference can never be handed out
    after a reposition. Returns the new epoch (monotonic, starting at 1 over the implicit 0).

    Making the move a recorded event rather than an after-the-fact inference is the point: this
    project's standing rule is that a stale hand-measured zone fails SILENTLY, and the glass door is
    repositioned about once every 4 days (the trail cam about once every 1.3). UNIQUE(source, epoch)
    means a concurrent second writer raises rather than quietly re-using an epoch; the rig is a
    single writer, and a raise here is the honest outcome."""
    who = str(detected_by or "").strip()
    if not who:
        raise ValueError("detected_by is required (what decided the camera moved)")
    epoch = current_view_epoch(conn, source) + 1
    conn.execute(
        "INSERT INTO view_epochs (source, epoch, started_at, detected_by, corr) "
        "VALUES (?, ?, ?, ?, ?)",
        (source, epoch, started_at or now_local_iso(), who,
         None if corr is None else float(corr)))
    retire_reference_images(conn, source, epoch, retired_at=started_at)
    conn.commit()
    return epoch


def view_epoch_started(conn: sqlite3.Connection, source: str) -> Optional[str]:
    """When the CURRENT view epoch began -- i.e. the last time `source` was seen to move. None in
    the implicit epoch 0 (never seen to move, or never migrated): there is then no moment to
    compare a zone's age against, so callers must treat None as 'not stale', not 'unknown'."""
    try:
        row = conn.execute("SELECT started_at FROM view_epochs WHERE source = ? "
                           "ORDER BY epoch DESC LIMIT 1", (source,)).fetchone()
    except sqlite3.OperationalError:
        return None
    return None if row is None else row[0]


# ---- Ignore zones (static false-fire spots, dashboard-editable) ---------------------
def _zone_row(row) -> dict:
    return {"id": int(row[0]), "source": row[1],
            "x1": int(row[2]), "y1": int(row[3]), "x2": int(row[4]), "y2": int(row[5]),
            "note": row[6], "created_by": row[7], "created_at": row[8]}


_ZONE_COLS = "id, source, x1, y1, x2, y2, note, created_by, created_at"


def list_ignore_zones(conn: sqlite3.Connection, source: Optional[str] = None) -> list:
    """Live (non-tombstoned) zones, oldest first -- for `source`, or all sources when None."""
    if source is None:
        rows = conn.execute(f"SELECT {_ZONE_COLS} FROM ignore_zones "
                            "WHERE deleted_at IS NULL ORDER BY id")
    else:
        rows = conn.execute(f"SELECT {_ZONE_COLS} FROM ignore_zones "
                            "WHERE source = ? AND deleted_at IS NULL ORDER BY id", (source,))
    return [_zone_row(r) for r in rows]


def add_ignore_zone(conn: sqlite3.Connection, source: str, x1, y1, x2, y2, *,
                    note: Optional[str] = None, created_by: str = "web") -> dict:
    """Insert one zone and return its stored row. Coordinates are normalized (ints, corners
    swapped so x1<x2 / y1<y2, clamped at 0) rather than trusted -- they arrive from a browser
    drag. A degenerate rectangle (under 4 px a side after normalizing) raises ValueError: it is
    always a slip of the pointer, and a 2-px zone would silently never match any detection's IoU."""
    src = str(source or "").strip()
    if not src:
        raise ValueError("source is required")
    ax1, ay1, ax2, ay2 = (max(0, int(round(float(v)))) for v in (x1, y1, x2, y2))
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    if (ax2 - ax1) < 4 or (ay2 - ay1) < 4:
        raise ValueError("zone too small (under 4 px a side)")
    note_clean = (str(note).strip()[:120] or None) if note else None
    at = now_local_iso()
    cur = conn.execute(
        "INSERT INTO ignore_zones (source, x1, y1, x2, y2, note, created_by, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (src, ax1, ay1, ax2, ay2, note_clean, created_by, at))
    conn.commit()
    return {"id": int(cur.lastrowid), "source": src, "x1": ax1, "y1": ay1, "x2": ax2, "y2": ay2,
            "note": note_clean, "created_by": created_by, "created_at": at}


def remove_ignore_zone(conn: sqlite3.Connection, zone_id) -> Optional[dict]:
    """Soft-delete one zone (stamp deleted_at). Returns the row it removed, or None if the id is
    unknown / already deleted. The row STAYS in the table on purpose: the tombstone is what keeps
    seed_ignore_zones from resurrecting a config rectangle the human deleted in the UI."""
    row = conn.execute(f"SELECT {_ZONE_COLS} FROM ignore_zones "
                       "WHERE id = ? AND deleted_at IS NULL", (int(zone_id),)).fetchone()
    if row is None:
        return None
    conn.execute("UPDATE ignore_zones SET deleted_at = ? WHERE id = ?",
                 (now_local_iso(), int(zone_id)))
    conn.commit()
    return _zone_row(row)


def seed_ignore_zones(conn: sqlite3.Connection, zones_by_source) -> int:
    """Copy config.ignore_zones ({source: [(x1,y1,x2,y2), ...]}) into the table -- ONCE per exact
    rectangle. Identity is (source, x1, y1, x2, y2) against ALL rows including tombstones, so a
    zone deleted in the dashboard stays deleted across restarts, while re-measuring a moved camera
    in config_local.py (new coordinates) still lands as a new row. Returns how many were added."""
    n = 0
    for source, rects in (zones_by_source or {}).items():
        for rect in (rects or ()):
            try:
                x1, y1, x2, y2 = (int(v) for v in rect)
            except (TypeError, ValueError):
                continue                     # a malformed config rect: skip, don't crash the rig
            hit = conn.execute("SELECT 1 FROM ignore_zones WHERE source = ? AND x1 = ? "
                               "AND y1 = ? AND x2 = ? AND y2 = ? LIMIT 1",
                               (source, x1, y1, x2, y2)).fetchone()
            if hit is not None:
                continue
            conn.execute(
                "INSERT INTO ignore_zones (source, x1, y1, x2, y2, note, created_by, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'config', ?)",
                (source, x1, y1, x2, y2, None, now_local_iso()))
            n += 1
    if n:
        conn.commit()
    return n
