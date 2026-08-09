"""
Standards-shaped data export -- the observation record, decoupled from this codebase.

The database is an irreplaceable scientific record (a documented three-litter raccoon season,
tens of thousands of human verdicts) that only this exact Python codebase can currently read.
That coupling is wrong for a solo hobby project: data longevity must outlive code longevity.
This writes the record as PLAIN CSV plus a DATA.md dictionary distilled from db.py's schema
comments -- legible to future-you on any stack, to a spreadsheet, or to a scientist, with no
venv and no torch. The shape follows the spirit of Camtrap DP (the camera-trap community's
Frictionless-Data standard: deployments / media / observations), without claiming full
conformance -- honest CSV beats aspirational YAML.

    python export.py                    # writes export-<today>.zip next to the DB
    python export.py --out DIR          # ... into DIR (backup.py passes its snapshots/ dir)

Read-only over the DB (connect_readonly), safe beside the live rig. backup.py calls this
weekly so the export lands on the Drive-synced destination beside the DB snapshot.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from datetime import date
from pathlib import Path

import config
import db

ROOT = config.ROOT

# table -> (columns, one-line meaning). Columns are SELECTed verbatim; a column missing on an
# older DB is skipped with a note rather than failing the whole export.
TABLES = {
    "observations": (
        "detections",
        ["id", "timestamp", "source", "detection_class", "confidence",
         "species", "species_confidence", "species_verified", "species_source",
         "model_species", "model_species_confidence",
         "individual_id", "individual_source", "labelled_at", "labeled_by",
         "visit_id", "crop_path", "crop_quality",
         "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "frame_w", "frame_h",
         "suppressed_at", "suppressed_by"],
        "One detector hit: a cropped animal at an instant. timestamp is OBSERVATION time, "
        "local ISO 8601 with UTC offset. species_verified: 1 human-confirmed, 0 human-rejected, "
        "NULL unreviewed. individual_source: 'human' is ground truth, 'auto' is the nightly "
        "assigner, 'cluster' a look-alike proposal. labeled_by: which human (NULL = the "
        "operator, before attribution existed). suppressed_at non-NULL = the furniture veto "
        "flagged it (shadow mode; row retained)."),
    "visits": (
        "visits",
        ["id", "source", "species", "individual_id", "started_at", "ended_at",
         "detection_count", "max_confidence", "species_margin", "sighting_conflict"],
        "One animal's stay: consecutive detections on one camera collapsed on a gap rule. The "
        "honest counting unit -- one lingering critter fires hundreds of observations."),
    "media_clips": (
        "clips",
        ["id", "source", "clip_path", "started_at", "ended_at", "fps", "width", "height",
         "frame_count", "detection_count", "max_confidence", "pruned_at"],
        "Short video around a visit. pruned_at non-NULL = the rolling disk window deleted the "
        "file; the row (and any derived tracks) outlive it, and backups may still hold the "
        "video."),
    "live_sightings": (
        "live_sightings",
        ["id", "source", "observed_at", "span_start", "span_end", "names", "stamped", "note",
         "labeled_by", "superseded_at", "superseded_by"],
        "A human's real-time 'who is here NOW' testimony -- the strongest label class. names is "
        "a JSON list; 2+ names (or one 'X + Kits' group string) = several animals, no solo "
        "stamp. superseded_* = corrected in the moment; the sequence is itself signal."),
    "individual_status": (
        "individual_status",
        ["name", "status", "effective_date", "note", "updated_at"],
        "Residency: 'departed' with the last resident day, or 'resident' written to undo it. "
        "One row per name; an absent name means nobody has said anything."),
    "life_events": (
        "life_events",
        ["id", "name", "event_date", "note", "labeled_by", "created_at"],
        "The cast's story as dated notes -- litters, injuries, debuts. Append-only."),
    "deployments_coverage": (
        "coverage_events",
        ["id", "source", "event", "at", "reason"],
        "When each camera was actually WATCHING: 'up'/'down' transitions. Pair them for effort; "
        "windows before the ledger existed are unknown, not covered."),
}

DATA_MD_HEAD = """# Backyard Critter Cam -- data dictionary

Written by export.py on {today}. One CSV per table, UTF-8, header row first. Timestamps are
LOCAL time with UTC offset (ISO 8601), e.g. `2026-08-07T21:08:50.026145-07:00` -- sortable,
unambiguous, and readable off a wall clock. NULL is an empty field.

The one relationship map: observations.visit_id -> visits.id; clips overlap visits by time on
the same `source` (no foreign key -- match by window); every `source` string names a camera
(`glass_door_cam` = the live rig, `trail_cam_sd` = SD-card imports).

Label trust, in one paragraph: `species_verified = 1` and `individual_source = 'human'` are a
human's word and the only rows the machine ever learns from. `model_species` preserves what the
classifier originally said even after a human correction overwrote `species`. Names containing
`" + "` ("Stan + Kits") are GROUP identities -- one archive label over several animals; never
read one as a single animal's ground truth.

## Tables
"""


def export_bundle(out_dir: Path, *, db_path: Path | None = None, dry_run: bool = False) -> Path:
    """Write export-<today>.zip (CSV per table + DATA.md) into out_dir. Returns the zip path."""
    today = date.today().isoformat()
    out_zip = out_dir / f"export-{today}.zip"
    conn = db.connect_readonly(db_path or config.CONFIG.db_path)
    if conn is None:
        raise SystemExit("no database to export yet")
    if dry_run:
        print(f"would write {out_zip}")
        conn.close()
        return out_zip
    out_dir.mkdir(parents=True, exist_ok=True)
    datamd = [DATA_MD_HEAD.format(today=today)]
    tmp = out_zip.with_name(out_zip.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname, (table, cols, meaning) in TABLES.items():
                have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
                if not have:
                    datamd.append(f"### {fname}.csv\n\n(not present in this database)\n")
                    continue
                use = [c for c in cols if c in have]
                missing = [c for c in cols if c not in have]
                buf = io.StringIO()
                w = csv.writer(buf, lineterminator="\n")
                w.writerow(use)
                for row in conn.execute(f"SELECT {', '.join(use)} FROM {table}"):
                    w.writerow(["" if v is None else v for v in row])
                zf.writestr(f"{fname}.csv", buf.getvalue())
                note = f" (older DB: columns absent -- {', '.join(missing)})" if missing else ""
                datamd.append(f"### {fname}.csv\n\n{meaning}{note}\n\nColumns: "
                              f"`{'`, `'.join(use)}`\n")
            zf.writestr("DATA.md", "\n".join(datamd))
        tmp.replace(out_zip)
    finally:
        conn.close()
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return out_zip


def main() -> int:
    ap = argparse.ArgumentParser(description="Export the observation record as CSV + DATA.md.")
    ap.add_argument("--out", type=Path, default=ROOT,
                    help="directory to write export-<date>.zip into (default: project root)")
    ap.add_argument("--db", type=Path, default=None, help="database path (default: config)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    p = export_bundle(args.out, db_path=args.db, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"wrote {p}  ({p.stat().st_size / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
