r"""
CERTIFIED IDENTITY REFERENCE BATCHES -- ground truth a human vouched for, kept OUT of the pipeline.

WHY THIS EXISTS. Every individual label in this database was produced by the same loop: the
embedding proposed a name, a human agreed with it. That makes the corpus circular, and
docs/identity-eval-2026-08-05.md says so plainly -- "every accuracy figure in this document is
optimistic by an unknown amount, because the confirmed corpus was built by a human AGREEING with
matches the embedding proposed", with no way to size the bias. A batch of photos where the human
says "this batch is <name>" BEFORE any model has an opinion is the one kind of data that breaks
that circle. It is also the only identity data in the project that does not decay: the measured
leave-one-visit-out top-1 falls 0.818 -> 0.482 -> 0.222 at a 0/7/21-day probe-to-template gap, so
a corpus label is worth less every week, while a certified photograph of a known animal is worth
the same in a year.

So: Matt drops photos into one folder per individual and runs one command.

    reference/
      <name>/         <- the folder name IS the individual. Nothing else infers identity.
        IMG_0007.JPG
        IMG_0008.JPG
        IMG_0009.MOV  <- videos too (see VIDEO REFERENCE FRAMES below)
        note.txt      <- optional; its text becomes the batch note
      <other-name>/
        ...

    python refcam.py reference                      # DRY RUN (the default) -- reports, writes nothing
    python refcam.py reference --apply              # ... actually record them
    python refcam.py reference --apply --crop       # ... and cut a detector crop from each photo
    python refcam.py reference --camera iphone      # source='reference_iphone' (default 'handheld')
    python refcam.py flat_folder --individual Name  # a folder that isn't named after the animal
    python refcam.py reference --apply --no-video-frames   # photos only; leave the videos alone
    python refcam.py --list                         # what is already certified, per individual
    python refcam.py --list --check                 # ... plus the "never in detections" guard

################################################################################################
# THESE ARE NOT GALLERY TEMPLATES. THEY ARE GROUND TRUTH FOR EVALUATION AND TRAIT DEFINITION.   #
################################################################################################

A reference photo NEVER becomes something the matcher can name a visit with, and that is a
measurement, not a preference:

  * Cross-domain appearance matching is not a weak signal here, it is NO signal. The maximum
    similarity across the entire 397x93 glass-door/trail-cam matrix is 0.363, below the 0.31
    novelty cut's useful band and far below any threshold that names anything. That is between two
    fixed outdoor cameras. A phone held at arm's length in daylight is a THIRD domain, further from
    the night webcam than the trail cam is.
  * individuals.py already refuses to rank, refit or auto-name across sources for exactly this
    reason (its SOURCE GUARD). References are written with their own `source` string
    ('reference_<camera>', see REFERENCE_SOURCE_PREFIX) which no capture rig ever writes, so that
    guard covers them for free -- but only because they live in their OWN TABLE and never enter
    `detections`. If a reference row ever reached `detections`, one confirmed match would let
    refit() offer the rest of the yard under that name, which is the documented mass-mislabel
    (docs/identity-eval-2026-08-05.md section 4). `check_not_in_detections()` asserts it can't
    happen, tests/test_refcam.py asserts it again, and an --apply run prints the verdict.

What they ARE for:
  * A held-out set with no suggest-then-confirm circularity: score the matcher against labels no
    model ever proposed.
  * TRAIT definition (traits.py). A permanent physical mark -- an ear notch, a torn ear, a scar --
    is era-invariant in a way the appearance embedding is not, and a high-resolution daylight
    reference is where such a mark can actually be SEEN and its appearance in the low-resolution
    night crops calibrated. A 2026-08-05 study concluded notches were undetectable at the median
    crop size; it was looking at 6-8 px of a night crop, not at a photograph.
  * Settling label questions about the historical corpus that the corpus itself cannot settle.

IDEMPOTENCY (Matt WILL run it twice). Keyed on the file's SHA-256 CONTENT hash, not its name:
phones rename on export, and the same photo copied twice is the same photo. A re-run over the same
folder is a no-op. The hash is UNIQUE in the table, so the same bytes filed under two different
names is not silently accepted -- it is reported LOUDLY as a certification conflict and skipped,
because only Matt can say which folder was right.

################################################################################################
# VIDEO REFERENCE FRAMES -- AND WHY A FRAME OFF A VIDEO IS WEAKER EVIDENCE THAN A PHOTOGRAPH.  #
################################################################################################

A batch folder may hold VIDEOS (.mp4/.mov/.m4v/.avi) as well as photos, and a folder holding
NOTHING but videos is a real batch -- it used to be skipped in silence, which is how the one
folder in this project that documents Stan's three kits imported as nothing at all. That folder
is the reason this exists: kits do not hold still, so a video is often the ONLY way to get a
usable reference still of one, and as of 2026-08-06 there are ZERO labelled kit detections
anywhere in the corpus. The least-documented animals in the project are documented on video or
not at all.

A video is decoded, sampled, and the SAME detector the rigs use (detector.Detector) picks which
frames are worth keeping -- best confidence and biggest animal box, capped per video, and never
two frames closer together than `--video-min-gap` seconds, because consecutive frames of one
video are near-duplicates and a reference set full of them is one observation wearing twelve
hats. Frames are written under <reference-crops>/_frames/<name>/ (a regenerable cache, like the
crops: the video itself is never copied), and each one gets a row of its own, with the source
video path, the source FRAME INDEX and the offset in seconds into the video.

THE PART THAT MATTERS. A frame is not a photograph, and this module refuses to pretend otherwise:

  * `media_kind` is 'video_frame' rather than 'photo' on every such row (legacy rows predate the
    column and read as NULL = 'photo'), so ANY query can separate the two, and an evaluation that
    wants only images a human personally framed can have exactly that. A photograph is a
    deliberate act -- Matt pointed a camera at one animal and filed the result. A frame is chosen
    by a detector out of footage nobody curated frame by frame. Same folder, same human, weaker
    evidence.
  * `identity_scope` is the load-bearing one. The folder "Stans Kids" means "Stan and her kits
    are in here"; it does NOT mean "every box is Stan", and nothing in a multi-animal video says
    WHICH animal any given box is. So when the scan sees two or more animals in ANY sampled frame,
    EVERY frame kept from that video is written with identity_scope='group': the individual_id
    still records the human's exact claim, but the claim is over a SET, and no consumer may treat
    such a row as single-animal ground truth. Only a video in which the detector never once saw
    two animals is scoped 'individual' -- there the human's "this video is <name>" distributes
    over the frames exactly as it does over a folder of photos.
    The judgement is per VIDEO, not per frame, and deliberately so: a solo frame lifted out of a
    video that elsewhere shows four raccoons is still one of four raccoons. It is also taken over
    every SCANNED frame, not just the kept ones -- a video demonstrably holds a second animal even
    if the frame proving it scored too low to keep.
    The failure direction is chosen too. The detector can double-box one animal (over-count, which
    only marks MORE rows 'group' -- harmless) and can miss a second animal in one frame
    (under-count, which would wrongly certify a group as an individual). Under-counting is the
    dangerous one, so the test is "two boxes ANYWHERE in the scan", the most sensitive form
    available: missing it means the detector never once separated the animals across ~100 sampled
    frames. `n_animals` is stored per row so the raw count is auditable rather than implied.
  * Getting this wrong would inject precisely the blended labels the rest of this project spends
    its time fighting (see the still-tracklet un-blend work, and reid_co_presence_min): a
    prototype averaged over two animals names neither.

CAPTURE TIME FOR A VIDEO. EXIF means nothing here, so the container is asked instead (ffprobe:
Apple's com.apple.quicktime.creationdate, which carries a real UTC offset, else creation_time in
UTC). Failing that, a filename that is itself an ISO timestamp is trusted -- that is exactly what
this rig names its own clips ('2026-08-06T01-55-50-431.mp4' is clips.py's `_start` wall clock).
Failing both, NULL. NEVER the mtime, for the reason captured_at() gives. A kept frame's own
captured_at is the video's start plus its offset into the video, which is arithmetic on a
recorded time, not a guess -- and it is NULL whenever the start time is.
Only the two time tags are read out of the container. Phone videos also carry a GPS location tag;
this module does not ask for it, does not store it, and should not start.

IDEMPOTENCY FOR FRAMES. The extracted JPEG's own bytes are NOT the key -- re-encoding is not
byte-stable, so the same frame would certify twice. The key is SHA-256 of "<video sha>:frame:<n>"
(frame_content_key), where <video sha> is the content hash of the source video, so it survives a
re-run, a re-encode and a rename. The video's own hash is stored alongside in `source_sha256`,
which makes the second run cheap AND settled: a video that has already been scanned is skipped
without decoding a frame, and the same video dropped under a second name is the same LOUD
conflict a duplicated photo is. Changing --video-max-frames and re-running therefore does NOT
top up an already-scanned video; delete its rows if you want it scanned again.

VIDEOS NEED THE DETECTOR, and that is not the optional extra --crop is. A photo is certified by
the human's word alone (the crop is a convenience); a video frame does not EXIST until a detector
has chosen it, so an --apply run with videos in it loads the model whether or not --crop was
given, says so, and writes the boxes it already computed as crops. --no-video-frames opts out.
Decoding prefers ffmpeg (the rawvideo-pipe idiom clips.py records with, run in reverse, with
hardware decode where the box has it -- measured 57s -> 17s on a 4K 60fps phone video); with no
ffmpeg on PATH it falls back to cv2.VideoCapture, which is slower and just as correct.

HEIC. iPhones shoot .HEIC by default. Pillow cannot read it without the `pillow-heif` package, and
this project does not add a dependency for a file format (the box dies of memory-commit
exhaustion; PytorchWildlife was rejected on dependency weight alone). If pillow-heif happens to be
installed it is used; otherwise HEIC files are counted, skipped, and the run ends by telling you
how to export as JPEG. Nothing is guessed and nothing is half-imported.

Robust by design, like the trail-cam importer: an unreadable or corrupt file is warned about and
skipped, and one bad file never aborts a batch.

PRIVACY. Reference photos are your own camera roll and they are NOT copied into the project -- the
DB stores a path, so wherever you keep them is where they stay. But the default drop folder sits
inside the repo, and THIS IS A PUBLIC REPO, so `.gitignore` ignores reference/ and reference_crops/
alongside crops/, frames/ and clips/ (added 2026-08-06: captured data, plus a regenerable cache).
A reference batch is the most personally identifying data this project touches -- these are
hand-held shots, so unlike the rig's tight animal crops they capture whatever the photographer was
standing in front of. If you repoint `reference_dir` via config_local.py, put it outside the
working tree or ignore it yourself: nothing here commits anything, but nothing here can stop
`git add .` either.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

import config
import db
# The project's ONE EXIF reader lives in the trail-cam importer; reusing it keeps a single parser
# for "when did this camera fire" rather than letting two drift apart. Deliberately NOT
# image_timestamp(), whose mtime fallback would invent a capture time for a copied file -- a
# reference photo's mtime is when it was AirDropped, and a wrong date is worse than no date.
import import_trailcam

# ---- Source naming -------------------------------------------------------------------------
# Reference rows carry their own `source`, which no capture rig ever writes. Two things lean on
# that: individuals.py's source guard (plain source equality, so a reference can never be ranked
# against a camera template), and check_not_in_detections(), which greps `detections` for this
# prefix and shouts if it ever finds one.
REFERENCE_SOURCE_PREFIX = "reference_"
# What the photos came off when nothing says otherwise. Kept generic on purpose -- this is a
# public repo and the next person's reference camera is not Matt's phone. `--camera iphone` gives
# 'reference_iphone'; a second camera gets its own string and stays distinguishable forever.
DEFAULT_CAMERA = "handheld"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# Read only when pillow-heif is already installed; never a dependency (see the module docstring).
HEIC_EXTS = {".heic", ".heif"}
# Video containers a reference batch may hold. Broader than import_trailcam.VIDEO_EXTS ({.mp4})
# on purpose: that one reads ONE camera model's card, this one reads whatever a human's phone
# handed them -- an iPhone writes .MOV, an Android .mp4, a screen grab .m4v, an old camcorder
# dump .avi. Matched case-insensitively (phones shout their extensions: IMG_5175.MOV).
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi"}

# ---- What KIND of evidence a row is ----------------------------------------------------------
# Stored in identity_references.media_kind. NULL means 'photo': every row written before video
# support existed was a photograph, and backfilling a value onto them would claim this module had
# looked at them when it hadn't. Read it as COALESCE(media_kind, 'photo').
MEDIA_PHOTO = "photo"
MEDIA_VIDEO_FRAME = "video_frame"

# ---- How far the human's identity claim reaches ----------------------------------------------
# Stored in identity_references.identity_scope; NULL reads as 'individual' for the same reason.
# 'individual': this row is ground truth for ONE named animal.
# 'group':      individual_id names a set that was demonstrably present ("Stans Kids"), and which
#               animal is in this particular box is UNKNOWN. Never single-animal ground truth.
# See the module docstring -- this is the decision the whole video path is built around.
SCOPE_INDIVIDUAL = "individual"
SCOPE_GROUP = "group"

# What a planned SOURCE file is (Plan.kind), as opposed to what a written row is (media_kind).
KIND_PHOTO = "photo"
KIND_VIDEO = "video"

# ---- Video frame-selection defaults ----------------------------------------------------------
# Consecutive frames of a video are near-duplicates: keeping them all would flood the reference
# set with correlated images and let one observation vote a dozen times in any evaluation built
# on these rows. So a video contributes a SPREAD, bounded three ways.
DEFAULT_VIDEO_MAX_FRAMES = 12      # hard cap per video
DEFAULT_VIDEO_MIN_GAP_S = 1.0      # no two kept frames closer together than this, in video time
DEFAULT_VIDEO_SCAN_FPS = 2.0       # sample the video at this rate to decide what's worth keeping
DEFAULT_VIDEO_MAX_SCANS = 300      # ... but never run the detector more than this many times
# The scan pass only has to RANK frames, so it runs on a downscaled copy (MegaDetector letterboxes
# to 640 anyway). The frames actually kept are re-extracted and re-detected at full resolution, so
# every stored box was measured on the image that was stored.
DEFAULT_VIDEO_SCAN_WIDTH = 1280

# ffmpeg is used the same way clips.py uses it -- a rawvideo pipe over subprocess, here read
# instead of written. Absent, cv2.VideoCapture decodes on its own (slower, equally correct), so
# neither binary is a dependency.
_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")

# Optional per-folder note: its text becomes the batch note when --note isn't given, so Matt can
# write "shot 2026-08-06, back fence, daylight" next to the photos instead of retyping it.
NOTE_FILENAME = "note.txt"
NOTE_MAX_CHARS = 500

# A reference photo is a photograph, not an SD-card frame, but the decompression-bomb ceiling is
# the same problem (cv2/PIL allocate from the declared header size), so reuse the importer's cap
# rather than inventing a second number.
MAX_DECODE_PIXELS = import_trailcam.MAX_DECODE_PIXELS

# Same 'src-<stem>' marker the trail-cam importer bakes into crop filenames, so a reference crop
# is traceable back to the photograph it came from by looking at its name alone.
SRC_TAG = import_trailcam.SRC_TAG


# ---- Schema (additive, idempotent -- the project's CREATE TABLE IF NOT EXISTS idiom) ---------
# Defined HERE rather than in db.py's SCHEMA because these tables are this module's business and
# nothing else in the pipeline reads them yet. ensure_schema() is called only on a WRITE path, so
# a dry run never touches the database at all.
SCHEMA = """
-- One certified reference photograph. NOT a detection: no rig produced it, no detector chose it,
-- and it is never a re-ID template (see the module docstring). It is a human saying "this is
-- <name>" about a specific image file, with enough provenance to be believed a year later.
CREATE TABLE IF NOT EXISTS identity_references (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,

    -- The individual, taken VERBATIM from the folder name (free text, exactly like
    -- detections.individual_id -- there is no individuals table). No name is ever inferred from
    -- the image, and no shipped constant in this repo holds an animal's name.
    individual_id   TEXT    NOT NULL,

    -- Which camera the batch came off, as 'reference_<camera>'. Distinct from every
    -- detections.source value on purpose: it is what makes the cross-domain guard automatic.
    source          TEXT    NOT NULL,

    captured_at     TEXT,               -- EXIF DateTimeOriginal, local ISO 8601 w/ offset. NULL when
                                        -- the file carries no EXIF -- never guessed from the mtime.
    imported_at     TEXT    NOT NULL,   -- when THIS row was written (local ISO 8601 w/ offset).

    file_path       TEXT    NOT NULL,   -- the photo, relative to the project root when it lives
                                        -- inside it, else absolute. The file is NOT copied: a
                                        -- reference batch is Matt's own photo library, and the
                                        -- pipeline has no business owning a second copy of it.

    -- Dedupe key, and the whole of the idempotency story. Content, not filename: phones rename on
    -- export. UNIQUE, so the same bytes cannot be certified twice -- including under two different
    -- names, which is reported as a conflict rather than resolved by guessing.
    content_sha256  TEXT    NOT NULL UNIQUE,

    width           INTEGER,            -- pixels, for the record: a reference is high-resolution and
    height          INTEGER,            -- that is precisely what makes it useful for trait work.

    batch           TEXT,               -- the folder this arrived in (provenance of one certification).
    note            TEXT,               -- free text from --note or the folder's note.txt.

    -- ---- Video provenance (all NULL on a photograph) ----------------------------------------
    -- Added 2026-08-06 with video support, and added ADDITIVELY: _migrate_references() puts these
    -- same columns on an existing table, so nothing is rewritten and no reader may assume a value
    -- is present. Rows written before this existed carry NULL in all of them.

    -- 'photo' | 'video_frame'. NULL = 'photo' (see MEDIA_PHOTO). The one column every query needs
    -- to tell a picture a human framed from a frame a detector picked.
    media_kind      TEXT,

    -- 'individual' | 'group'. NULL = 'individual'. THE honest-provenance column: 'group' means
    -- individual_id names a SET that was present, not the animal in this box. See SCOPE_GROUP.
    identity_scope  TEXT,

    source_video    TEXT,               -- the video this frame came out of (stored like file_path).
    source_sha256   TEXT,               -- content hash of THAT VIDEO. Idempotency at video level:
                                        -- a scanned video is skipped without decoding anything,
                                        -- and the same video under two names is a conflict.
    video_frame_index INTEGER,          -- frame number in the source video (0-based, exact).
    video_time_s    REAL,               -- offset into the video, seconds.
    n_animals       INTEGER             -- animal boxes the detector found in THIS frame. Stored so
                                        -- the group/individual call above is auditable, not implied.
);
CREATE INDEX IF NOT EXISTS idx_identity_refs_individual ON identity_references(individual_id);
CREATE INDEX IF NOT EXISTS idx_identity_refs_source     ON identity_references(source);

-- OPTIONAL, and clearly separable: the animal box cut out of a reference photo by the SAME
-- detector the rigs use (detector.py / MegaDetector v6), so a reference is comparable to a crop
-- instead of being a whole photograph with a lawn in it. Separate table because cropping is a
-- second, re-runnable step: deleting every row here loses nothing a re-run can't rebuild, while
-- the reference row above is the irreplaceable part (a human's word).
CREATE TABLE IF NOT EXISTS identity_reference_crops (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_id    INTEGER NOT NULL REFERENCES identity_references(id) ON DELETE CASCADE,
    crop_path       TEXT    NOT NULL,   -- relative to the project root, like every other stored path.
    detection_class TEXT,               -- MegaDetector's coarse label ('animal').
    confidence      REAL,
    bbox_x1         REAL,
    bbox_y1         REAL,
    bbox_x2         REAL,
    bbox_y2         REAL,
    crop_quality    REAL,               -- quality.score_crop, same scale as detections.crop_quality.
    model           TEXT,               -- which detector version cut it (crops are re-runnable).
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_identity_ref_crops_ref ON identity_reference_crops(reference_id);
"""


# Columns added to identity_references AFTER the first release of this module. Applied by
# _migrate_references() with the project's PRAGMA table_info / ALTER TABLE ADD COLUMN idiom
# (db._migrate), so an existing database grows them without a rewrite and a fresh one gets them
# from the CREATE TABLE above. Every one is nullable: no existing row is touched, and NULL means
# exactly what each column's comment in SCHEMA says it means.
REFERENCE_COLUMNS_ADDED = {
    "media_kind": "TEXT",
    "identity_scope": "TEXT",
    "source_video": "TEXT",
    "source_sha256": "TEXT",
    "video_frame_index": "INTEGER",
    "video_time_s": "REAL",
    "n_animals": "INTEGER",
}


def _migrate_references(conn: sqlite3.Connection) -> None:
    """Bring an EXISTING identity_references table up to the current column set. Additive and
    idempotent -- it only ever adds a missing nullable column, never drops, rewrites or backfills
    one (a backfilled media_kind would claim this module had inspected rows it never saw). No-op on
    a database that has no reference table yet: CREATE TABLE above already gives it every column."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(identity_references)")}
    if not cols:
        return
    for name, decl in REFERENCE_COLUMNS_ADDED.items():
        if name not in cols:
            conn.execute(f"ALTER TABLE identity_references ADD COLUMN {name} {decl}")
    # Created HERE and not in SCHEMA: on an old table the column doesn't exist until the ALTERs
    # above have run, and a CREATE INDEX over a missing column fails the whole executescript.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_identity_refs_video "
                 "ON identity_references(source_sha256)")


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the reference tables if they don't exist, and add any columns a older database is
    missing. Additive and idempotent, per the project's schema rule -- safe to call on every write,
    and called on NO read path so a dry run leaves the database byte-identical."""
    conn.executescript(SCHEMA)
    _migrate_references(conn)
    conn.commit()


def reference_columns(conn: sqlite3.Connection) -> set[str]:
    """The columns identity_references actually has right now. Read paths consult this instead of
    assuming, because a dry run may be reporting against a database written by an older build (and
    must not create or alter anything to find out)."""
    return {r[1] for r in conn.execute("PRAGMA table_info(identity_references)")}


def has_schema(conn: sqlite3.Connection) -> bool:
    """True when identity_references exists. Read paths consult this instead of creating the
    table, so `refcam.py <folder>` (a dry run) can report against a database it never writes to."""
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='identity_references'"
    ).fetchone() is not None


# ---- Naming --------------------------------------------------------------------------------

def _slug(text: str) -> str:
    """Filesystem-and-source-safe form of a free-text name: non-alphanumerics become '_'. Used for
    the `source` string and the per-individual crop folder, never for the stored individual_id
    (which keeps the human's exact spelling)."""
    return "".join(c if c.isalnum() else "_" for c in str(text).strip()).strip("_").lower()


def reference_source(camera: str = DEFAULT_CAMERA) -> str:
    """The `source` string for a reference batch off `camera` -- 'reference_iphone', and so on.
    One string per physical camera, because two phones are two domains and a year from now the
    only way to tell them apart is this column."""
    return REFERENCE_SOURCE_PREFIX + (_slug(camera) or _slug(DEFAULT_CAMERA))


def is_reference_source(source: str | None) -> bool:
    """True for any source string this module writes. The one place the prefix convention is
    interpreted, so nothing else has to hard-code it."""
    return bool(source) and str(source).startswith(REFERENCE_SOURCE_PREFIX)


# ---- Reading the drop folder ----------------------------------------------------------------

def _heic_ready() -> bool:
    """True when HEIC can be decoded because `pillow-heif` is ALREADY installed. Never installs
    anything and never raises -- a False here becomes a printed instruction to export as JPEG."""
    try:
        import pillow_heif  # type: ignore
        pillow_heif.register_heif_opener()
        return True
    except Exception:
        return False


def folder_note(folder: Path) -> str | None:
    """The text of the folder's note.txt, trimmed, or None. Best-effort: an unreadable note never
    stops an import (the photos are the point)."""
    p = folder / NOTE_FILENAME
    try:
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace").strip()
            return text[:NOTE_MAX_CHARS] or None
    except OSError:
        pass
    return None


MEDIA_EXTS = IMAGE_EXTS | HEIC_EXTS | VIDEO_EXTS


def is_video(path) -> bool:
    """True for a file this module would read as VIDEO rather than as a still."""
    return Path(path).suffix.lower() in VIDEO_EXTS


def list_photos(folder: Path, recursive: bool = True) -> list[Path]:
    """Every candidate photo in `folder`, sorted. Includes HEIC so it can be REPORTED rather than
    silently ignored -- a batch that looks empty because the phone shot HEIC is the exact
    confusion this module exists to avoid."""
    walker = folder.rglob("*") if recursive else folder.glob("*")
    exts = IMAGE_EXTS | HEIC_EXTS
    return sorted(p for p in walker if p.is_file() and p.suffix.lower() in exts)


def list_media(folder: Path, recursive: bool = True) -> list[Path]:
    """Every candidate reference file in `folder` -- photos AND videos -- sorted.

    This, not list_photos, is what defines a batch. A folder holding nothing but videos is a real
    batch and must never come back empty: that silence is how the only footage of Stan's kits
    imported as nothing at all (2026-08-06)."""
    walker = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in walker if p.is_file() and p.suffix.lower() in MEDIA_EXTS)


def find_batches(root: Path, *, individual: str | None = None,
                 recursive: bool = True) -> tuple[list[tuple[str, list[Path]]], list[Path]]:
    """Split a drop folder into ``([(individual_name, [file, ...]), ...], loose_files)``.

    THE NAME COMES FROM THE DIRECTORY, and from nothing else: each immediate subfolder of `root`
    is one individual, named exactly as the folder is (spelling preserved -- the DB stores free
    text). Hidden and '_'-prefixed folders are skipped so a '_rejects' or '.DS_Store' pile can sit
    alongside without being certified as an animal.

    `individual` overrides that for a FLAT folder ("all of these are <name>"), which is what a
    fresh AirDrop dump looks like before anyone has sorted it.

    `loose_files` are photos/videos sitting directly in `root` with no per-individual folder. They
    are NEVER imported under a guessed name -- the caller warns about them and moves on.
    """
    if individual:
        return [(individual.strip(), list_media(root, recursive))], []

    batches: list[tuple[str, list[Path]]] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if sub.name.startswith((".", "_")):
            continue
        media = list_media(sub, recursive)
        if media:
            batches.append((sub.name.strip(), media))
    loose = [p for p in root.glob("*") if p.is_file() and p.suffix.lower() in MEDIA_EXTS]
    return batches, loose


# ---- Per-file facts -------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    """Streaming SHA-256 of a file's bytes -- the dedupe key. Chunked so a 40 MP ProRAW export
    isn't read whole into RAM (same shape as detector._sha256 for the pinned weights)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exif_orientation_swaps_axes(path: Path) -> bool:
    """True when the EXIF orientation tag rotates the image by 90/270 degrees.

    Phones record the sensor's native landscape frame and describe the rotation in EXIF rather than
    re-encoding the pixels, so a portrait photo decodes SIDEWAYS unless the tag is honoured. Values
    5-8 are the quarter-turn cases."""
    try:
        from PIL import Image, ExifTags
        with Image.open(path) as im:
            tag = (im.getexif() or {}).get(next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), 274))
        return int(tag) in (5, 6, 7, 8)
    except Exception:
        return False


def load_upright(path: Path):
    """Decode a photo with its EXIF orientation APPLIED, as a BGR array for cv2/the detector.

    Every consumer of a reference photo must go through here. An iPhone frame decodes 90 degrees
    off without this, which would hand the detector a sideways raccoon (well outside its training
    distribution) and make every stored crop useless as a visual reference."""
    from PIL import Image, ImageOps
    import numpy as np
    Image.MAX_IMAGE_PIXELS = MAX_DECODE_PIXELS
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im)[:, :, ::-1].copy()


def image_size(path: Path) -> tuple[int, int] | None:
    """(width, height) AS DISPLAYED -- i.e. with EXIF orientation applied -- or None if the file
    isn't a readable image. Uses Pillow's lazy header read, so no pixels are decoded and a corrupt
    file is caught without allocating anything; the orientation tag lives in the header too."""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = MAX_DECODE_PIXELS
        with Image.open(path) as im:
            w, h = im.size
        if exif_orientation_swaps_axes(path):
            w, h = h, w
        return int(w), int(h)
    except ImportError:
        pass
    except Exception:
        return None
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return None
        return int(img.shape[1]), int(img.shape[0])
    except Exception:
        return None


def captured_at(path: Path) -> str | None:
    """EXIF DateTimeOriginal as a local ISO 8601 string with offset, or None when the file carries
    no EXIF. NEVER falls back to the mtime: a reference photo's mtime is when it was copied off
    the phone, and a confidently wrong capture date is worse than an honest NULL. A naive EXIF
    stamp is read in this machine's local zone, matching db.now_local_iso()'s convention."""
    dt = import_trailcam._exif_datetime(path)
    return None if dt is None else dt.astimezone().isoformat()


# ---- Per-VIDEO facts -------------------------------------------------------------------------

def video_meta(path: Path) -> dict | None:
    """``{duration_s, fps, frame_count, width, height}`` for one video, or None if it can't be
    read at all. Reuses the trail-cam importer's probe (ffprobe first, OpenCV second) so the
    project keeps ONE answer to "what is in this container", then normalises the gaps that probe
    is allowed to leave: a missing fps is derived from frames/duration and vice versa. Every value
    it returns is positive, so callers can divide by them without checking. Never raises."""
    meta = import_trailcam._probe_video(path)
    if not meta:
        return None
    dur = float(meta.get("duration_s") or 0.0)
    fps = float(meta.get("fps") or 0.0)
    n = int(meta.get("frame_count") or 0)
    w, h = int(meta.get("width") or 0), int(meta.get("height") or 0)
    if fps <= 0 and dur > 0 and n > 0:
        fps = n / dur
    if n <= 0 and fps > 0 and dur > 0:
        n = int(round(dur * fps))
    if not (n > 0 and fps > 0 and w > 0 and h > 0):
        return None
    return {"duration_s": dur if dur > 0 else n / fps, "fps": fps,
            "frame_count": n, "width": w, "height": h}


# A filename that IS a timestamp: '2026-08-06T01-55-50-431.mp4' (what clips.py names this rig's own
# clips), and the obvious near-spellings. Anchored on the ISO date so 'IMG_5175' can't match.
_FILENAME_TS_RE = re.compile(
    r"(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})[T_ ](?P<h>\d{2})[-:.](?P<mi>\d{2})[-:.](?P<s>\d{2})"
    r"(?:[-_.](?P<ms>\d{1,6}))?")

# Container tags that hold a real capture time, best first. Apple's carries the local UTC offset;
# `creation_time` is UTC. NOTHING ELSE is requested: a phone video also carries a GPS location tag
# (com.apple.quicktime.location.ISO6709) and this module has no business reading it.
_VIDEO_TIME_TAGS = ("com.apple.quicktime.creationdate", "creation_time")


def _parse_iso_loose(raw: str):
    """A tz-aware datetime from an ISO-8601-ish container tag, or None. Tolerates the two spellings
    ffprobe hands back that datetime.fromisoformat rejects on Python 3.10/3.11: a trailing 'Z', and
    a colon-less UTC offset ('-0700'). A naive stamp is read in this machine's local zone, matching
    captured_at()'s convention."""
    text = str(raw or "").strip()
    if not text:
        return None
    text = re.sub(r"[Zz]$", "+00:00", text)
    text = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", text)
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.astimezone()


def _ffprobe_time_tags(path: Path) -> list[str]:
    """The values of _VIDEO_TIME_TAGS present in the container, best-first. Empty list when ffprobe
    is missing or says nothing -- never raises, so a machine with no ffmpeg just falls through to
    the filename."""
    if _FFPROBE is None:
        return []
    try:
        r = subprocess.run(
            [_FFPROBE, "-v", "error", "-show_entries",
             "format_tags=" + ",".join(_VIDEO_TIME_TAGS), "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30)
        tags = (json.loads(r.stdout or "{}").get("format") or {}).get("tags") or {}
    except Exception:
        return []
    return [str(tags[t]) for t in _VIDEO_TIME_TAGS if tags.get(t)]


def video_captured_at(path: Path) -> str | None:
    """When a VIDEO was shot, as a local ISO 8601 string with offset, or None.

    EXIF is meaningless for a video container, so this asks, in order: the container's own creation
    tags (via ffprobe), then a filename that is itself an ISO timestamp -- which is precisely how
    this rig names its own clips, so a clip pulled out of clips/ and dropped into a reference
    folder keeps its real capture time even though ffmpeg wrote no creation tag into it.

    Then NULL. NEVER the mtime, for exactly the reason captured_at() gives: a file's mtime is when
    it was copied off the phone, and a confidently wrong capture date is worse than an honest gap.
    """
    for raw in _ffprobe_time_tags(path):
        dt = _parse_iso_loose(raw)
        if dt is not None:
            return dt.astimezone().isoformat()
    m = _FILENAME_TS_RE.search(Path(path).stem)
    if m:
        g = m.groupdict()
        try:
            dt = datetime(int(g["y"]), int(g["mo"]), int(g["d"]), int(g["h"]), int(g["mi"]),
                          int(g["s"]), int((g["ms"] or "0").ljust(6, "0")))
        except ValueError:
            return None
        return dt.astimezone().isoformat()
    return None


def frame_content_key(video_sha256: str, frame_index: int) -> str:
    """The dedupe key for ONE frame extracted from a video.

    NOT the extracted JPEG's own bytes. Re-encoding is not byte-stable -- a different OpenCV build,
    a different JPEG quality, even a different decode path gives different bytes for the same
    frame -- so hashing the output would certify the same frame again on every re-run, which is the
    one thing this module's idempotency story cannot allow. Hashing the SOURCE video's content plus
    the frame index is stable across all of that, and still content-keyed rather than name-keyed:
    the same video renamed, re-copied or re-dropped produces the same keys."""
    return hashlib.sha256(
        f"{video_sha256}:frame:{int(frame_index)}".encode("utf-8")).hexdigest()


def reference_frames_dir(cfg: config.Config) -> Path:
    """Where extracted video frames are written: '_frames' beside the per-individual crop folders.

    A regenerable cache, exactly like the crops -- the frame can always be cut out of the video
    again (frame_content_key makes that deterministic), so nothing irreplaceable lives here. The
    leading underscore keeps it out of find_batches(), so re-pointing a drop folder at the crops
    tree can't certify a cache as an animal. Honours cfg.reference_frames_dir if config ever grows
    one; until then it derives from the knob that already exists."""
    override = getattr(cfg, "reference_frames_dir", None)
    return Path(override) if override else Path(cfg.reference_crops_dir) / "_frames"


# ---- Planning (what an import WOULD do) ------------------------------------------------------

# Statuses a planned reference can carry. 'new' is the only one that writes anything.
NEW = "new"
DUPLICATE = "duplicate"          # already certified (same bytes, same name) -- the idempotent case
CONFLICT = "conflict"            # same bytes, DIFFERENT name -- only a human can resolve it
HEIC_UNSUPPORTED = "heic"        # readable in principle, not by this install
UNREADABLE = "unreadable"        # corrupt, oversized, or vanished


@dataclass
class Plan:
    """One file's verdict before anything is written. A dry run prints these; an --apply run
    processes exactly the `new` ones.

    `kind` says whether this is a photograph (one file -> one reference row) or a VIDEO (one file
    -> up to --video-max-frames rows, chosen by the detector, and none at all if it holds no
    animal). A video plan therefore cannot promise a row count in advance: the dry run reports the
    ceiling, and only an --apply run -- which is the only run that loads a detector -- can say
    what a video actually yielded."""
    individual: str
    path: Path
    status: str
    sha256: str | None = None
    captured_at: str | None = None
    width: int | None = None
    height: int | None = None
    detail: str = ""
    kind: str = KIND_PHOTO
    video_meta: dict | None = None


@dataclass
class VideoOptions:
    """Frame-selection knobs for the video path, bundled so they travel as one thing (the CLI has
    a flag per field). Defaults are the module constants; see each for why."""
    max_frames: int = DEFAULT_VIDEO_MAX_FRAMES
    min_gap_s: float = DEFAULT_VIDEO_MIN_GAP_S
    scan_fps: float = DEFAULT_VIDEO_SCAN_FPS
    max_scans: int = DEFAULT_VIDEO_MAX_SCANS
    scan_width: int = DEFAULT_VIDEO_SCAN_WIDTH
    frames_dir: Path | None = None      # None = reference_frames_dir(cfg)
    enabled: bool = True                # --no-video-frames turns the whole path off


def existing_hashes(conn: sqlite3.Connection | None) -> dict[str, tuple[int, str, str]]:
    """{content_sha256: (id, individual_id, file_path)} for everything already certified. Empty
    when the table doesn't exist yet (a fresh DB, or a dry run against one), which is what lets
    the read path avoid creating it."""
    if conn is None or not has_schema(conn):
        return {}
    return {r[0]: (int(r[1]), r[2], r[3]) for r in conn.execute(
        "SELECT content_sha256, id, individual_id, file_path FROM identity_references")}


def existing_video_hashes(conn: sqlite3.Connection | None) -> dict[str, tuple[str, int]]:
    """{source_sha256: (individual_id, n_frames_certified)} for every video already scanned.

    This is video-level idempotency, and it is what makes a second run CHEAP as well as correct: a
    video whose hash is here is skipped without decoding a single frame -- which matters when the
    file is a 578 MB 4K phone video. Empty on any database that predates the column, since nothing
    there can have been scanned."""
    if conn is None or not has_schema(conn):
        return {}
    if "source_sha256" not in reference_columns(conn):
        return {}
    return {r[0]: (r[1], int(r[2])) for r in conn.execute(
        "SELECT source_sha256, MIN(individual_id), COUNT(*) FROM identity_references "
        "WHERE source_sha256 IS NOT NULL GROUP BY source_sha256")}


def _plan_video(path: Path, individual: str, video_seen: dict) -> Plan:
    """One video's verdict. Same four outcomes a photo can have, decided on the VIDEO's own content
    hash: already scanned under this name (DUPLICATE), scanned under another name (CONFLICT --
    refused, exactly like a duplicated photograph, because only a human can say which folder was
    right), unprobeable (UNREADABLE), or NEW. Nothing is decoded here; a plan is a promise to
    scan, and the scan needs a detector this path never loads."""
    try:
        sha = file_sha256(path)
    except OSError as e:
        return Plan(individual, path, UNREADABLE, kind=KIND_VIDEO, detail=str(e))
    prior = video_seen.get(sha)
    if prior is not None:
        prior_name, n_frames = prior
        if str(prior_name).casefold() == individual.casefold():
            return Plan(individual, path, DUPLICATE, sha256=sha, kind=KIND_VIDEO,
                        detail=f"already scanned as {prior_name} ({n_frames} frame(s) certified)")
        return Plan(individual, path, CONFLICT, sha256=sha, kind=KIND_VIDEO,
                    detail=f"identical video already certified as '{prior_name}'")
    meta = video_meta(path)
    if meta is None:
        return Plan(individual, path, UNREADABLE, sha256=sha, kind=KIND_VIDEO,
                    detail="unreadable/corrupt video (neither ffprobe nor OpenCV could read it)")
    return Plan(individual, path, NEW, sha256=sha, kind=KIND_VIDEO,
                captured_at=video_captured_at(path),
                width=meta["width"], height=meta["height"], video_meta=meta)


def plan_batches(conn: sqlite3.Connection | None,
                 batches: list[tuple[str, list[Path]]]) -> list[Plan]:
    """Decide what each file would do, WITHOUT writing anything -- and, for videos, without
    decoding anything either. Dedupe is checked against the DB and against this run's own files
    (the same bytes twice in one drop folder is a duplicate, not a UNIQUE-constraint crash);
    photos key on their own content hash, videos on theirs. Nothing here raises: a file that can't
    be read becomes an UNREADABLE plan and the batch continues."""
    seen = dict(existing_hashes(conn))
    video_seen = dict(existing_video_hashes(conn))
    heic_ok: bool | None = None
    plans: list[Plan] = []
    for individual, photos in batches:
        for path in photos:
            ext = path.suffix.lower()
            if ext in VIDEO_EXTS:
                plan = _plan_video(path, individual, video_seen)
                plans.append(plan)
                if plan.status == NEW and plan.sha256:
                    # Within-run dedupe: the same video dropped twice in one batch is scanned once.
                    video_seen[plan.sha256] = (individual, 0)
                continue
            if ext in HEIC_EXTS:
                if heic_ok is None:
                    heic_ok = _heic_ready()
                if not heic_ok:
                    plans.append(Plan(individual, path, HEIC_UNSUPPORTED,
                                      detail="pillow-heif not installed"))
                    continue
            try:
                sha = file_sha256(path)
            except OSError as e:
                plans.append(Plan(individual, path, UNREADABLE, detail=str(e)))
                continue
            prior = seen.get(sha)
            if prior is not None:
                _, prior_name, prior_path = prior
                # Case-insensitive, matching individual_status's COLLATE NOCASE: a folder renamed
                # from 'notch' to 'Notch' is a typo, not two animals, and raising a CONFLICT on it
                # would spend the loudest signal this module has on noise.
                if prior_name.casefold() == individual.casefold():
                    plans.append(Plan(individual, path, DUPLICATE, sha256=sha,
                                      detail=f"already certified as {prior_name}"))
                else:
                    # The same photograph filed under two names. Guessing either way would write a
                    # false certification, and a certification is the one label in this project
                    # nothing else can check. Report and refuse.
                    plans.append(Plan(individual, path, CONFLICT, sha256=sha,
                                      detail=f"identical bytes already certified as "
                                             f"'{prior_name}' ({prior_path})"))
                continue
            size = image_size(path)
            if size is None:
                plans.append(Plan(individual, path, UNREADABLE, sha256=sha,
                                  detail="unreadable/corrupt image, or over the decode cap"))
                continue
            w, h = size
            plans.append(Plan(individual, path, NEW, sha256=sha, captured_at=captured_at(path),
                              width=w, height=h))
            seen[sha] = (-1, individual, str(path))   # within-run dedupe
    return plans


# ---- Writing --------------------------------------------------------------------------------

def insert_reference(conn: sqlite3.Connection, *, individual_id: str, source: str,
                     file_path: Path, content_sha256: str, captured_at: str | None = None,
                     width: int | None = None, height: int | None = None,
                     batch: str | None = None, note: str | None = None,
                     media_kind: str = MEDIA_PHOTO, identity_scope: str = SCOPE_INDIVIDUAL,
                     source_video: Path | None = None, source_sha256: str | None = None,
                     video_frame_index: int | None = None, video_time_s: float | None = None,
                     n_animals: int | None = None) -> int:
    """Insert one certified reference; returns its id. The single writer of identity_references --
    and note what it does NOT do: it never touches `detections`, `visits`, or any table the
    matcher reads. That is the whole containment story, in one function.

    `media_kind` / `identity_scope` default to the strong case ('photo' certified for an
    'individual') because that is what a photograph is; the video path passes both explicitly and
    is the only caller that ever passes 'group'. Every video column is NULL for a photo."""
    cur = conn.execute(
        """
        INSERT INTO identity_references (
            individual_id, source, captured_at, imported_at, file_path,
            content_sha256, width, height, batch, note,
            media_kind, identity_scope, source_video, source_sha256,
            video_frame_index, video_time_s, n_animals
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (individual_id, source, captured_at, db.now_local_iso(), db.rel_to_root(file_path),
         content_sha256,
         None if width is None else int(width), None if height is None else int(height),
         batch, note,
         media_kind, identity_scope,
         None if source_video is None else db.rel_to_root(source_video), source_sha256,
         None if video_frame_index is None else int(video_frame_index),
         None if video_time_s is None else float(video_time_s),
         None if n_animals is None else int(n_animals)),
    )
    conn.commit()
    return int(cur.lastrowid)


def insert_reference_crop(conn: sqlite3.Connection, *, reference_id: int, crop_path: Path,
                          detection_class: str, confidence: float, bbox, crop_quality=None,
                          model: str | None = None) -> int:
    """Record one detector-cut crop of a reference photo; returns its id."""
    x1, y1, x2, y2 = bbox
    cur = conn.execute(
        """
        INSERT INTO identity_reference_crops (
            reference_id, crop_path, detection_class, confidence,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, crop_quality, model, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (int(reference_id), db.rel_to_root(crop_path), detection_class, float(confidence),
         float(x1), float(y1), float(x2), float(y2),
         None if crop_quality is None else float(crop_quality), model, db.now_local_iso()),
    )
    conn.commit()
    return int(cur.lastrowid)


def crop_reference(conn: sqlite3.Connection, cfg: config.Config, reference_id: int, path: Path,
                   individual: str, detector, *, stamp: str | None = None,
                   frame=None, dets=None) -> int:
    """OPTIONAL SECOND STEP: run the animal detector over one reference photo and save the box(es)
    as crops, so a reference is comparable to a pipeline crop instead of being a whole photograph.

    `frame` / `dets` let a caller that has ALREADY decoded the image and already run the detector
    on it hand both in rather than paying for them twice -- that is the video path, where the
    boxes are the very reason the frame was selected. Passing neither is the photo path and behaves
    exactly as it always has.

    Reuses the project's detector (detector.Detector, passed in already built) and the live rig's
    save_crop, so a reference crop is cut with the SAME padding, the same JPEG quality and the
    same quality score as every other crop in the project -- the only way "comparable" means
    anything. Crops land under cfg.reference_crops_dir/<name>/, never in crops/, because crops/ is
    the pipeline's tree and backup.py zips it by capture date.

    Returns the number of crops written. Never raises: a photo the detector can't handle is warned
    about and skipped, exactly like a bad frame in a card import.
    """
    # Imported lazily: backyard_cam pulls in the whole rig (web, powerguard, stats), and `import
    # refcam` should stay cheap for anything that only wants to READ references.
    import cv2
    from backyard_cam import save_crop

    # load_upright, not cv2.imread: imread ignores the EXIF orientation tag, so a phone photo
    # arrives 90 degrees off and the detector sees a sideways animal.
    if frame is None:
        try:
            frame = load_upright(path)
        except Exception:
            frame = cv2.imread(str(path))
    if frame is None:
        print(f"  [warn] could not decode for cropping (reference kept): {path.name}")
        return 0
    if dets is None:
        try:
            dets = detector.detect(frame)
        except Exception as e:
            print(f"  [detector] error on {path.name} (reference kept, no crop): {e}")
            return 0
    keep = [d for d in dets if d.class_name in cfg.save_classes]
    if not keep:
        print(f"  [crop] no animal found in {path.name} -- reference kept, no crop.")
        return 0

    crop_cfg = replace(cfg, crops_dir=cfg.reference_crops_dir)
    day = _slug(individual) or "unnamed"      # crops group by INDIVIDUAL here, not by date
    # The stamp leads with the reference's content hash (see import_plans) so two photos that
    # share a filename -- IMG_0007.JPG happens twice the moment two phones are involved -- can
    # never overwrite each other's crop, and any crop is traceable back to its row by name alone.
    stamp = stamp or f"{SRC_TAG}{path.stem}"
    n = 0
    for i, det in enumerate(keep):
        result = save_crop(frame, det, crop_cfg, day, stamp, i)
        if result is None:
            continue
        crop_path, crop_q = result
        insert_reference_crop(conn, reference_id=reference_id, crop_path=crop_path,
                              detection_class=det.class_name, confidence=det.confidence,
                              bbox=det.bbox, crop_quality=crop_q,
                              model=getattr(cfg, "model_version", None))
        n += 1
        print(f"    crop -> {db.rel_to_root(crop_path)}  ({det.class_name} {det.confidence:.2f})")
    return n


# ---- Video: decoding ---------------------------------------------------------------------------

def _ffmpeg_select(indices=None, stride: int | None = None) -> str:
    """An ffmpeg `select` filter that passes exactly the wanted SOURCE frames.

    Two spellings, because the two passes want different things: a STRIDE ('every 30th frame') is
    one short expression whatever the video's length, while an explicit index list is what pins the
    extraction pass to precisely the frames the scan chose. The commas inside mod()/eq() are
    escaped because ffmpeg's filtergraph parser -- not the shell -- treats a bare comma as the end
    of a filter."""
    if stride and stride > 1:
        return f"select=not(mod(n\\,{int(stride)}))"
    return "select=" + "+".join(f"eq(n\\,{int(i)})" for i in indices or [])


def _ffmpeg_frames(path: Path, select: str, native_size: tuple[int, int],
                   scale_to: tuple[int, int] | None = None):
    """Yield BGR frames that `select` passes, decoded straight into memory over a rawvideo pipe.

    The same subprocess idiom clips.py records with (-f rawvideo -pix_fmt bgr24, one Popen, no
    temp files), run in the other direction. '-hwaccel auto' uses the box's video decoder when it
    has one and silently decodes in software when it doesn't -- measured 57s -> 17.5s over a 52s
    4K/60 phone video, which is the difference between a usable import and one nobody runs twice.
    '-fps_mode passthrough' keeps the output frame count EXACTLY equal to the number of frames the
    filter passed, which is what lets the caller pair frame i of the stream with source index
    indices[i]. Yields nothing (rather than raising) if ffmpeg is missing or fails, so the caller
    can fall back."""
    import numpy as np
    if _FFMPEG is None:
        return
    w, h = scale_to or native_size
    vf = select if scale_to is None else f"{select},scale={int(w)}:{int(h)}"
    proc = subprocess.Popen(
        [_FFMPEG, "-y", "-loglevel", "error", "-hwaccel", "auto", "-i", str(path),
         "-an", "-vf", vf, "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    nbytes = int(w) * int(h) * 3
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if not buf or len(buf) < nbytes:
                break               # clean end of stream, or a torn tail we refuse to reshape
            # .copy(): frombuffer hands back a read-only view of the pipe's bytes, and cv2 needs a
            # writable, owned array.
            yield np.frombuffer(buf, np.uint8).reshape(int(h), int(w), 3).copy()
    finally:
        for close in (lambda: proc.stdout.close(), proc.kill):
            try:
                close()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def _cv2_frames(path: Path, indices):
    """Yield the frames at `indices` using OpenCV alone -- the no-ffmpeg fallback.

    Decodes SEQUENTIALLY (grab() past what we don't want, read() what we do) rather than seeking:
    cv2's frame seek is codec- and build-dependent and can land on the wrong frame, and a fallback
    has to be correct before it is fast. Yields nothing if the file won't open."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return
    try:
        pos = 0
        for target in indices:
            while pos < target:
                if not cap.grab():
                    return
                pos += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                return
            pos += 1
            yield frame
    finally:
        cap.release()


def iter_video_frames(path: Path, indices, *, native_size: tuple[int, int],
                      scale_to: tuple[int, int] | None = None, stride: int | None = None):
    """Yield ``(source_frame_index, BGR frame)`` for the given SOURCE frame indices.

    ffmpeg first, OpenCV if ffmpeg is absent or produced nothing at all. Note that the fallback
    ignores `scale_to` and yields full-size frames: every caller reads the frame's own .shape, so
    a fallback that decodes bigger images is slower and still right."""
    indices = [int(i) for i in indices]
    if not indices:
        return
    if _FFMPEG is not None:
        n = 0
        for frame in _ffmpeg_frames(path, _ffmpeg_select(indices, stride), native_size, scale_to):
            if n >= len(indices):
                break
            yield indices[n], frame
            n += 1
        if n:
            return
        print(f"  [video] ffmpeg returned no frames for {path.name} -- decoding with OpenCV.")
    for i, frame in zip(indices, _cv2_frames(path, indices)):
        yield i, frame


# ---- Video: choosing which frames are worth keeping ---------------------------------------------

@dataclass
class ScannedFrame:
    """One sampled frame of a video, scored by what the detector found in it."""
    index: int
    time_s: float
    score: float
    confidence: float
    n_animals: int


def frame_score(dets, frame_w: int, frame_h: int) -> float:
    """How useful one frame is as a REFERENCE: for its best animal box, confidence plus how much of
    the frame's linear extent the box spans (sqrt of the area fraction).

    Two terms, both 0..1, weighted equally and deliberately not tuned. Confidence alone would keep
    twelve crisp frames of a distant animal 40 px across -- useless for the trait work these
    references exist for; size alone would keep whatever the detector was least sure about as long
    as it was close. Linear extent rather than area because a box twice as wide is twice as
    informative, not four times."""
    area = float(max(1, int(frame_w) * int(frame_h)))
    best = 0.0
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        frac = max(0.0, min(1.0, ((x2 - x1) * (y2 - y1)) / area))
        best = max(best, float(d.confidence) + math.sqrt(frac))
    return best


def scan_plan(meta: dict, opts: VideoOptions) -> tuple[list[int], int]:
    """``(source frame indices to sample, stride)`` for one video.

    Sampling at opts.scan_fps, then widened if that would run the detector more than
    opts.max_scans times -- so a 10-second clip and a 10-minute one both cost a bounded number of
    detector passes, and the long one is simply sampled more sparsely."""
    n, fps = int(meta["frame_count"]), float(meta["fps"])
    stride = max(1, int(round(fps / max(0.01, opts.scan_fps))))
    if opts.max_scans and (n / stride) > opts.max_scans:
        stride = max(stride, math.ceil(n / opts.max_scans))
    return list(range(0, n, stride)), stride


def scan_video(path: Path, meta: dict, detector, cfg: config.Config,
               opts: VideoOptions) -> tuple[list[ScannedFrame], int]:
    """Sample a video and score every sampled frame by what the detector found. Returns
    ``(frames that held at least one animal, frames actually scanned)``.

    Runs on a downscaled copy (opts.scan_width) because this pass only has to RANK -- MegaDetector
    letterboxes to 640 regardless, and the frames actually kept are re-detected at full resolution
    later, so no stored box is ever one measured on a shrunken image. A detector error on one frame
    is warned about and skipped; one bad frame never aborts a video, exactly as one bad file never
    aborts a batch."""
    indices, stride = scan_plan(meta, opts)
    fps = float(meta["fps"])
    native = (int(meta["width"]), int(meta["height"]))
    scale_to = None
    if opts.scan_width and native[0] > opts.scan_width:
        sw = max(2, int(opts.scan_width) & ~1)          # even dims: some encoders/filters demand it
        sh = max(2, int(round(native[1] * sw / native[0])) & ~1)
        scale_to = (sw, sh)

    hits: list[ScannedFrame] = []
    scanned = 0
    for idx, frame in iter_video_frames(path, indices, native_size=native,
                                        scale_to=scale_to, stride=stride):
        scanned += 1
        try:
            dets = detector.detect(frame)
        except Exception as e:  # noqa: BLE001 -- one frame must never kill the video
            print(f"    [detector] error on frame {idx} of {path.name} (skipped): {e}")
            continue
        animals = [d for d in dets if d.class_name in cfg.save_classes]
        if not animals:
            continue
        h, w = frame.shape[:2]
        hits.append(ScannedFrame(
            index=idx, time_s=idx / fps, score=frame_score(animals, w, h),
            confidence=max(float(d.confidence) for d in animals), n_animals=len(animals)))
    return hits, scanned


def select_frames(hits: list[ScannedFrame], *, max_frames: int,
                  min_gap_s: float) -> list[ScannedFrame]:
    """Pick the frames to keep: best score first, never two within `min_gap_s` of each other in
    VIDEO time, at most `max_frames` of them. Returned in time order.

    The gap is the point. Twelve consecutive frames of a raccoon are twelve pictures of one instant
    -- they would inflate any count built on these rows and teach a reader nothing the first one
    didn't. Greedy-with-a-gap gives a spread across the whole clip while still leading with the
    frames where the animal is closest and most confidently seen."""
    kept: list[ScannedFrame] = []
    for f in sorted(hits, key=lambda x: (-x.score, x.index)):
        if len(kept) >= max(1, int(max_frames)):
            break
        if any(abs(f.time_s - k.time_s) < float(min_gap_s) for k in kept):
            continue
        kept.append(f)
    return sorted(kept, key=lambda x: x.index)


def video_identity_scope(hits: list[ScannedFrame]) -> str:
    """SCOPE_GROUP if the detector ever saw two animals at once anywhere in this video, else
    SCOPE_INDIVIDUAL.

    Read the module docstring before changing this. It is judged over the WHOLE video and over
    EVERY scanned frame, not per kept frame: a solo frame lifted out of a video that elsewhere
    shows four raccoons is still one of four raccoons, and the folder name ("Stans Kids") is a
    claim about the set, not about any box. Certifying such a box as single-animal ground truth
    would inject exactly the blended labels the rest of this project exists to undo."""
    return SCOPE_GROUP if any(h.n_animals >= 2 for h in hits) else SCOPE_INDIVIDUAL


# ---- Video: writing the references --------------------------------------------------------------

def import_video_reference(conn: sqlite3.Connection, cfg: config.Config, plan: Plan, *,
                           source: str, detector, opts: VideoOptions,
                           note: str | None = None) -> dict:
    """Scan ONE video, keep the best spread of frames, and certify each as its own reference row.

    Returns a tally: ``{frames, crops, scanned, hits, scope}``. Never raises -- an undecodable
    video reports and returns zeros, like every other bad file here.

    Each kept frame is written to <frames-dir>/<name>/ as a JPEG (a regenerable cache; the video
    itself is never copied), re-detected AT FULL RESOLUTION so the boxes stored alongside are the
    boxes in the stored image, and rowed with its source video, frame index, offset in seconds and
    animal count. Crops are cut unconditionally here rather than under --crop: the detector is
    already loaded, the boxes are already computed, and they are the whole reason this frame beat
    the others.

    identity_scope comes from video_identity_scope() and applies to EVERY frame of the video."""
    out = {"frames": 0, "crops": 0, "scanned": 0, "hits": 0, "scope": SCOPE_INDIVIDUAL}
    import cv2

    meta = plan.video_meta or video_meta(plan.path)
    if meta is None:
        print(f"  [video] {plan.path.name}: unreadable -- nothing certified.")
        return out
    indices, _stride = scan_plan(meta, opts)
    print(f"  [video] {plan.path.name}: {meta['duration_s']:.1f}s, {meta['frame_count']} frames, "
          f"{meta['width']}x{meta['height']} -- scanning {len(indices)} sampled frame(s) ...")

    hits, scanned = scan_video(plan.path, meta, detector, cfg, opts)
    out["scanned"], out["hits"] = scanned, len(hits)
    if not hits:
        print(f"  [video] {plan.path.name}: no animal in any of {scanned} scanned frame(s) -- "
              "nothing certified. (Re-run to try again; nothing was recorded for this video.)")
        return out

    kept = select_frames(hits, max_frames=opts.max_frames, min_gap_s=opts.min_gap_s)
    scope = video_identity_scope(hits)
    out["scope"] = scope
    max_seen = max(h.n_animals for h in hits)
    print(f"  [video] {plan.path.name}: {len(hits)}/{scanned} scanned frame(s) held an animal; "
          f"keeping {len(kept)} (cap {opts.max_frames}, min gap {opts.min_gap_s:g}s). "
          f"Most animals seen at once: {max_seen} -> identity_scope='{scope}'.")
    if scope == SCOPE_GROUP:
        print(f"      ^ '{plan.individual}' is recorded as a GROUP claim for these frames: the "
              "video holds more than one animal, so no single box can be certified as one named "
              "individual. They are NOT single-animal ground truth.")

    frames_dir = Path(opts.frames_dir or reference_frames_dir(cfg))
    out_dir = frames_dir / (_slug(plan.individual) or "unnamed")
    out_dir.mkdir(parents=True, exist_ok=True)
    by_index = {k.index: k for k in kept}
    base_dt = db.parse_local(plan.captured_at) if plan.captured_at else None

    for idx, frame in iter_video_frames(plan.path, [k.index for k in kept],
                                        native_size=(int(meta["width"]), int(meta["height"]))):
        hit = by_index.get(idx)
        if hit is None:
            continue
        key = frame_content_key(plan.sha256 or "", idx)
        stem = f"{(plan.sha256 or '')[:8]}_{SRC_TAG}{plan.path.stem}_f{idx:06d}"
        frame_path = out_dir / f"{stem}.jpg"
        try:
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        except Exception as e:  # noqa: BLE001 -- a write failure loses one frame, not the video
            print(f"    [warn] could not write frame {idx} of {plan.path.name}: {e}")
            continue
        # Re-detect on the FULL-RESOLUTION frame that was just stored, so every recorded box was
        # measured on the image the row points at (the scan ran on a downscaled copy).
        try:
            dets = detector.detect(frame)
        except Exception as e:  # noqa: BLE001
            print(f"    [detector] error on stored frame {idx} of {plan.path.name}: {e}")
            dets = []
        animals = [d for d in dets if d.class_name in cfg.save_classes]
        if not animals:
            # The scan saw an animal here at 1280px and the full-res pass doesn't. Rare, and the
            # honest response is to drop the frame rather than certify a picture of nothing.
            frame_path.unlink(missing_ok=True)
            print(f"    [skip] frame {idx}: no animal at full resolution after all.")
            continue
        h, w = frame.shape[:2]
        when = None if base_dt is None else (base_dt + timedelta(seconds=hit.time_s)).isoformat()
        try:
            ref_id = insert_reference(
                conn, individual_id=plan.individual, source=source, file_path=frame_path,
                content_sha256=key, captured_at=when, width=w, height=h,
                batch=plan.path.parent.name, note=note,
                media_kind=MEDIA_VIDEO_FRAME, identity_scope=scope,
                source_video=plan.path, source_sha256=plan.sha256,
                video_frame_index=idx, video_time_s=hit.time_s, n_animals=len(animals))
        except sqlite3.IntegrityError as e:
            print(f"    [skip] frame {idx}: already certified ({e}).")
            frame_path.unlink(missing_ok=True)
            continue
        out["frames"] += 1
        out["crops"] += crop_reference(conn, cfg, ref_id, frame_path, plan.individual, detector,
                                       stamp=f"{key[:8]}_{SRC_TAG}{plan.path.stem}_f{idx:06d}",
                                       frame=frame, dets=dets)
        print(f"    frame {idx:>6} @ {hit.time_s:6.2f}s  {w}x{h}  "
              f"{len(animals)} animal(s), best {hit.confidence:.2f}  "
              f"({when or 'no container date'})")
    return out


def import_plans(conn: sqlite3.Connection | None, cfg: config.Config, plans: list[Plan], *,
                 source: str, note: str | None = None, batch_notes: dict | None = None,
                 apply: bool = False, detector=None, video: VideoOptions | None = None) -> dict:
    """Write the `new` plans (or, by DEFAULT, write nothing and just count them).

    `apply=False` is the default everywhere in this module: an import that mislabels a certified
    batch corrupts the only non-circular ground truth the project has, so the destructive form is
    the one you have to ask for. Returns a tally dict.

    tally[NEW] counts SOURCE FILES certified -- one per photo, and one per video that yielded at
    least one frame -- so it means the same thing in a dry run (where a video is one plan) as in an
    apply run. The frames a video contributed are counted separately in tally['video_frames'],
    because "12 references" and "12 photographs Matt took" are very different claims.
    """
    tally = {NEW: 0, DUPLICATE: 0, CONFLICT: 0, HEIC_UNSUPPORTED: 0, UNREADABLE: 0, "crops": 0,
             "videos": 0, "video_frames": 0, "video_scanned": 0, "video_empty": 0,
             "video_skipped": 0, "video_group": 0}
    video = video or VideoOptions()
    new_plans = [p for p in plans if p.status == NEW]
    for p in plans:
        if p.status != NEW:
            tally[p.status] += 1
    if not apply:
        tally[NEW] = len(new_plans)
        tally["videos"] = sum(1 for p in new_plans if p.kind == KIND_VIDEO)
        return tally
    if conn is None:
        raise ValueError("import_plans(apply=True) needs an open, writable connection")

    ensure_schema(conn)
    for p in new_plans:
        batch_note = note or (batch_notes or {}).get(p.individual)
        if p.kind == KIND_VIDEO:
            if not video.enabled:
                tally["video_skipped"] += 1
                print(f"  [skip] {p.path.name}: videos disabled (--no-video-frames).")
                continue
            if detector is None:
                # Not the same as a photo without --crop. A photo is certified by the human's word
                # and the crop is a convenience; a video FRAME does not exist until a detector has
                # chosen it, so there is nothing to certify here. Say so rather than quietly
                # dropping the file -- silent skipping of these videos is the bug being fixed.
                tally["video_skipped"] += 1
                print(f"  [skip] {p.path.name}: a video reference needs the detector to choose "
                      "its frames, and none is loaded.")
                continue
            res = import_video_reference(conn, cfg, p, source=source, detector=detector,
                                         opts=video, note=batch_note)
            tally["videos"] += 1
            tally["video_frames"] += res["frames"]
            tally["video_scanned"] += res["scanned"]
            tally["crops"] += res["crops"]
            if res["frames"]:
                tally[NEW] += 1
                if res["scope"] == SCOPE_GROUP:
                    tally["video_group"] += res["frames"]
            else:
                tally["video_empty"] += 1
            continue
        try:
            ref_id = insert_reference(
                conn, individual_id=p.individual, source=source, file_path=p.path,
                content_sha256=p.sha256, captured_at=p.captured_at,
                width=p.width, height=p.height, batch=p.path.parent.name, note=batch_note)
        except sqlite3.IntegrityError as e:
            # The UNIQUE hash raced us (two processes, or a plan built before another run wrote).
            # Skipping is the idempotent answer -- the row that exists is as good as the one we'd
            # have written.
            print(f"  [skip] {p.path.name}: already certified ({e}).")
            tally[DUPLICATE] += 1
            continue
        tally[NEW] += 1
        when = p.captured_at or "no EXIF date"
        print(f"  [{p.individual}] {p.path.name}  {p.width}x{p.height}  ({when})")
        if detector is not None:
            tally["crops"] += crop_reference(
                conn, cfg, ref_id, p.path, p.individual, detector,
                stamp=f"{p.sha256[:8]}_{SRC_TAG}{p.path.stem}")
    return tally


# ---- Guard + reporting -----------------------------------------------------------------------

def check_not_in_detections(conn: sqlite3.Connection) -> int:
    """How many rows in `detections` carry a reference source. MUST be 0, always, by construction
    -- this module writes only identity_references / identity_reference_crops. Cheap enough to run
    after every --apply, and asserted in tests/test_refcam.py, because the failure it guards
    against is silent: a reference in the detections table would become a re-ID template on a
    domain where the maximum measured cross-source similarity is 0.363, and one accepted match
    would let refit() propose the same name across the yard."""
    return int(conn.execute(
        "SELECT COUNT(*) FROM detections WHERE source LIKE ?",
        (REFERENCE_SOURCE_PREFIX + "%",)).fetchone()[0])


def summary_rows(conn: sqlite3.Connection) -> list[dict]:
    """Per-individual roll-up of what is certified: how many references, from which cameras, over
    what date range, how many carry a crop -- and how many are VIDEO FRAMES rather than
    photographs, and how many of those are group-scoped rather than individual ground truth. Empty
    list when nothing has ever been imported.

    The two video counts read through COALESCE, and are reported as 0 against a database written
    before the columns existed: every row there is a photograph, which is exactly what NULL means
    (see MEDIA_PHOTO)."""
    if not has_schema(conn):
        return []
    cols = reference_columns(conn)
    if "media_kind" in cols:
        video_cols = (f"SUM(CASE WHEN COALESCE(r.media_kind, '{MEDIA_PHOTO}') = "
                      f"'{MEDIA_VIDEO_FRAME}' THEN 1 ELSE 0 END) AS n_frames, "
                      f"SUM(CASE WHEN COALESCE(r.identity_scope, '{SCOPE_INDIVIDUAL}') = "
                      f"'{SCOPE_GROUP}' THEN 1 ELSE 0 END) AS n_group")
    else:
        video_cols = "0 AS n_frames, 0 AS n_group"
    rows = conn.execute(
        f"""
        SELECT r.individual_id AS name,
               COUNT(*)                          AS n,
               COUNT(DISTINCT r.source)          AS n_sources,
               GROUP_CONCAT(DISTINCT r.source)   AS sources,
               MIN(r.captured_at)                AS first_capture,
               MAX(r.captured_at)                AS last_capture,
               (SELECT COUNT(*) FROM identity_reference_crops c
                 JOIN identity_references r2 ON r2.id = c.reference_id
                WHERE r2.individual_id = r.individual_id) AS n_crops,
               {video_cols}
          FROM identity_references r
         GROUP BY r.individual_id
         ORDER BY n DESC, name
        """).fetchall()
    return [{"name": r[0], "n": r[1], "n_sources": r[2], "sources": r[3],
             "first_capture": r[4], "last_capture": r[5], "n_crops": r[6],
             "n_frames": int(r[7] or 0), "n_group": int(r[8] or 0)} for r in rows]


def describe_plans(plans: list[Plan], *, prefix: str = "  ") -> list[str]:
    """Human-readable lines for a dry run -- shared with the apply path so both say the same thing
    in the same shape (the staticfilter.describe() convention). Videos are called out separately
    and never counted as photographs, because they aren't."""
    lines: list[str] = []
    by_individual: dict[str, list[Plan]] = {}
    for p in plans:
        by_individual.setdefault(p.individual, []).append(p)
    for name, items in by_individual.items():
        counts: dict[str, int] = {}
        for p in items:
            counts[p.status] = counts.get(p.status, 0) + 1
        bits = ", ".join(f"{v} {k}" for k, v in counts.items())
        n_video = sum(1 for p in items if p.kind == KIND_VIDEO)
        what = f"{len(items) - n_video} photo(s)"
        if n_video:
            what += f" + {n_video} video(s)"
        lines.append(f"{prefix}{name}: {what} -- {bits}")
        for p in items:
            if p.status != NEW:
                lines.append(f"{prefix}  [{p.status}] {p.path.name}: {p.detail}")
            elif p.kind == KIND_VIDEO:
                meta = p.video_meta or {}
                lines.append(
                    f"{prefix}  [video] {p.path.name}: {meta.get('duration_s', 0):.1f}s, "
                    f"{meta.get('width')}x{meta.get('height')} -- frames chosen by the detector "
                    f"on --apply (capture time: {p.captured_at or 'unknown'})")
    return lines


# ---- CLI --------------------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    c = config.CONFIG
    p = argparse.ArgumentParser(
        description="Import CERTIFIED identity reference batches -- photos (and videos) a human "
                    "vouched for, one folder per individual. They are ground truth for evaluation "
                    "and trait work, NEVER re-ID gallery templates, and they never enter the "
                    "detections table. Frames pulled out of a video are marked as such, and a "
                    "video holding several animals is recorded as a GROUP claim, never as "
                    "single-animal ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("folder", nargs="?", default=str(c.reference_dir),
                   help="Drop folder holding one subfolder per individual (the folder name IS "
                        "the name).")
    p.add_argument("--individual", default=None,
                   help="Treat the folder as a FLAT batch of one named animal, instead of "
                        "inferring a name per subfolder.")
    p.add_argument("--camera", default=DEFAULT_CAMERA,
                   help="Which camera shot the batch; stored as source='reference_<camera>'.")
    p.add_argument("--note", default=None,
                   help="Free-text note stored on every reference in this run (a folder's "
                        "note.txt is used when this is absent).")
    p.add_argument("--db", default=str(c.db_path), help="SQLite database to write to.")
    p.add_argument("--crops-dir", default=str(c.reference_crops_dir),
                   help="Where --crop writes reference crops (foldered per individual). "
                        "Deliberately not the pipeline's crops/ tree.")
    p.add_argument("--no-recursive", dest="recursive", action="store_false", default=True,
                   help="Don't walk subfolders inside each individual's folder.")
    p.add_argument("--crop", action="store_true",
                   help="Also run the animal detector over each photo and save the crop "
                        "(optional second step; needs the model weights and is the only part "
                        "that loads a GPU).")
    p.add_argument("--device", default=c.device, choices=["cuda", "cpu", "auto"],
                   help="Inference device for --crop and for scanning videos.")
    p.add_argument("--no-video-frames", dest="videos", action="store_false", default=True,
                   help="Ignore videos entirely and certify only the still photos.")
    p.add_argument("--video-max-frames", type=int, default=DEFAULT_VIDEO_MAX_FRAMES,
                   help="Most reference frames to keep from any ONE video.")
    p.add_argument("--video-min-gap", type=float, default=DEFAULT_VIDEO_MIN_GAP_S,
                   help="Least time (seconds, in video time) between two kept frames -- "
                        "consecutive frames are near-duplicates and would flood the set.")
    p.add_argument("--video-scan-fps", type=float, default=DEFAULT_VIDEO_SCAN_FPS,
                   help="How often per second of video to sample a frame and look for animals.")
    p.add_argument("--video-max-scans", type=int, default=DEFAULT_VIDEO_MAX_SCANS,
                   help="Cap on detector passes per video (a long video is sampled more sparsely "
                        "rather than more expensively).")
    p.add_argument("--frames-dir", default=None,
                   help="Where extracted video frames are written "
                        "(default: <crops-dir>/_frames). A regenerable cache -- the frame can "
                        "always be cut out of the video again.")
    p.add_argument("--list", action="store_true",
                   help="Show what is already certified, per individual, and exit.")
    p.add_argument("--check", action="store_true",
                   help="With --list: also verify no reference row ever reached `detections`.")
    p.add_argument("--apply", action="store_true",
                   help="Actually write the rows. WITHOUT this the run is a DRY RUN that opens "
                        "the database read-only and changes nothing.")
    p.add_argument("--dry-run", action="store_true",
                   help="The default. Accepted explicitly, and it WINS over --apply if both are "
                        "given (fail-safe: a certified label is not something to write by "
                        "accident).")
    return p.parse_args(argv)


def _print_list(conn) -> None:
    rows = summary_rows(conn)
    if not rows:
        print("No certified references yet. Drop photos or videos into "
              "<folder>/<name>/ and run this with --apply.")
        return
    print(f"{'individual':<20} {'refs':>5} {'photo':>6} {'frame':>6} {'group':>6} {'crops':>6}"
          "  sources / first..last capture")
    for r in rows:
        span = " .. ".join(x[:10] for x in (r["first_capture"], r["last_capture"]) if x) or "no EXIF dates"
        print(f"{r['name']:<20} {r['n']:>5} {r['n'] - r['n_frames']:>6} {r['n_frames']:>6} "
              f"{r['n_group']:>6} {r['n_crops']:>6}  {r['sources']}  {span}")
    if any(r["n_group"] for r in rows):
        print("\n  'group' counts rows whose identity_scope is 'group': they came off a video "
              "holding\n  MORE THAN ONE animal, so the name is a claim about the set, not about "
              "the animal in\n  the box. Never treat one as single-animal ground truth "
              "(WHERE COALESCE(identity_scope,\n  'individual') = 'individual' excludes them).")


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = replace(config.CONFIG, db_path=Path(args.db), device=args.device,
                  reference_crops_dir=Path(args.crops_dir))
    apply = bool(args.apply) and not args.dry_run
    source = reference_source(args.camera)

    if args.list:
        conn = db.connect_readonly(cfg.db_path)
        if conn is None:
            print(f"No database at {cfg.db_path} yet.")
            return 0
        try:
            _print_list(conn)
            if not args.check:
                return 0
            # A non-zero exit here is deliberate: --check is the form you'd put in a script, and a
            # containment breach should fail that script rather than scroll past.
            n = check_not_in_detections(conn)
            print(f"\nGuard: {n} reference row(s) in `detections` -- "
                  + ("OK, references are contained." if n == 0 else
                     "*** NOT OK: references must never be pipeline detections ***"))
            return 0 if n == 0 else 1
        finally:
            conn.close()

    root = Path(args.folder)
    if not root.is_dir():
        print(f"[ERROR] Not a folder: {root}")
        print("  Create it and put one subfolder per individual inside "
              "(e.g. <folder>/<name>/IMG_0001.JPG).")
        return 1

    video_opts = VideoOptions(
        max_frames=args.video_max_frames, min_gap_s=args.video_min_gap,
        scan_fps=args.video_scan_fps, max_scans=args.video_max_scans,
        frames_dir=Path(args.frames_dir) if args.frames_dir else None,
        enabled=bool(args.videos))

    batches, loose = find_batches(root, individual=args.individual, recursive=args.recursive)
    if loose:
        print(f"[warn] {len(loose)} file(s) sit directly in {root} with no per-individual "
              f"folder -- SKIPPED. Identity comes from the folder name; move them into "
              f"{root}/<name>/ (or re-run with --individual <name>).")
    if not batches:
        print(f"No per-individual folders with photos or videos under {root}.")
        return 0

    print(f"{len(batches)} batch(es) under {root}, as source='{source}' "
          f"({'APPLY' if apply else 'dry run'}):")

    # A dry run opens the database READ-ONLY, so "writes nothing" is enforced by the connection,
    # not merely by this module's branching.
    conn = db.connect(cfg.db_path) if apply else db.connect_readonly(cfg.db_path)
    if conn is None:
        print(f"  (no database at {cfg.db_path} yet -- every photo below would be new)")
    try:
        plans = plan_batches(conn, batches)
        for line in describe_plans(plans):
            print(line)

        # A video reference does not exist until a detector has chosen its frames, so videos pull
        # the model in whether or not --crop was asked for -- unlike a photo, where the crop is a
        # convenience on top of a certification the human already made.
        n_videos = sum(1 for p in plans if p.kind == KIND_VIDEO and p.status == NEW)
        want_detector = args.crop or (video_opts.enabled and n_videos > 0)
        detector = None
        if apply and want_detector:
            from detector import CudaUnavailableError, Detector
            why = " for --crop" if args.crop else ""
            if n_videos and video_opts.enabled:
                why = (f" to choose reference frames from {n_videos} video(s)"
                       + (" and for --crop" if args.crop else ""))
            print(f"\nLoading MegaDetector v6 ({cfg.model_version}) on {cfg.device}{why} ...")
            try:
                detector = Detector(cfg.model_version, cfg.device, cfg.min_confidence,
                                    classes=cfg.detect_classes)
            except CudaUnavailableError as e:
                print(f"\n[CUDA ERROR]\n{e}")
                return 2
        elif want_detector:
            print("  (dry run: no detector is loaded, so no crop is cut and no video is scanned. "
                  "The videos above are listed, not read.)")
        if n_videos and not video_opts.enabled:
            print(f"  ({n_videos} video(s) found but --no-video-frames was given -- skipping them.)")

        notes = {name: folder_note(photos[0].parent) for name, photos in batches if photos}
        tally = import_plans(conn, cfg, plans, source=source, note=args.note,
                             batch_notes=notes, apply=apply, detector=detector,
                             video=video_opts)

        print(f"\n{'Imported' if apply else 'Would import'} {tally[NEW]} reference(s)"
              + (f" ({tally['videos']} of them video(s))" if not apply and tally["videos"] else "")
              + f"; {tally[DUPLICATE]} already certified, {tally[CONFLICT]} conflicting, "
              f"{tally[UNREADABLE]} unreadable.")
        if apply and (tally["videos"] or tally["video_skipped"]):
            print(f"  Videos: scanned {tally['videos']}, {tally['video_scanned']} frame(s) "
                  f"examined, {tally['video_frames']} frame(s) certified"
                  + (f", {tally['video_empty']} held no animal" if tally["video_empty"] else "")
                  + (f", {tally['video_skipped']} skipped" if tally["video_skipped"] else "") + ".")
            if tally["video_group"]:
                print(f"  *** {tally['video_group']} of those frame(s) are identity_scope='group': "
                      "their video held MORE THAN ONE animal, so the folder name is a claim about "
                      "the\n      GROUP, not about the animal in any one box. They are reference "
                      "imagery and trait material -- they are NOT single-animal ground truth. ***")
        if tally["crops"]:
            print(f"  {tally['crops']} crop(s) written under {cfg.reference_crops_dir}.")
        if tally[HEIC_UNSUPPORTED]:
            print(f"  {tally[HEIC_UNSUPPORTED]} HEIC file(s) SKIPPED: this install cannot decode "
                  "HEIC, and adding a package for it is not something this project does.\n"
                  "  On the iPhone: Settings > Camera > Formats > Most Compatible (shoots JPEG), "
                  "or share the photos with 'Automatic' conversion, or export them as JPEG.")
        if tally[CONFLICT]:
            print("  *** CONFLICT: the same photo or video is certified under two different "
                  "names. Nothing was written for those files -- only you can say which folder "
                  "is right. ***")
        if apply:
            n_bad = check_not_in_detections(conn)
            print(f"  Containment check: {n_bad} reference row(s) in `detections` "
                  f"({'OK' if n_bad == 0 else '*** NOT OK ***'}). References are ground truth for "
                  "evaluation and traits -- they are never re-ID templates.")
        else:
            print("  Dry run -- nothing was written. Re-run with --apply to record these.")
    finally:
        if conn is not None:
            conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
