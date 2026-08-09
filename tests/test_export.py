"""
Tests for export.py -- the plain-CSV escape hatch.

The point of this module is that the observation record outlives the code that made it, so the
tests care about exactly that: every table lands as readable CSV with a header, the dictionary
travels with the data, NULL is empty rather than the string "None", and an older database that
predates a column exports what it HAS instead of raising. Pure stdlib + a throwaway DB.
"""
from __future__ import annotations

import csv
import io
import zipfile

import db
import export


def _det(conn, **kw):
    fields = dict(timestamp=db.now_local_iso(), source=db.SOURCE_GLASS_DOOR_CAM,
                  detection_class="animal", confidence=0.9, bbox=(0, 0, 10, 10),
                  frame_w=100, frame_h=100, crop_path="crops/x.jpg", species="raccoon",
                  crop_quality=1.0)
    fields.update(kw)
    return db.insert_detection(conn, **fields)


def test_export_bundle_writes_csvs_and_a_dictionary(conn, db_path, tmp_path):
    _det(conn)
    conn.commit()
    out = export.export_bundle(tmp_path, db_path=db_path)
    assert out.exists() and out.suffix == ".zip"
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "DATA.md" in names
        # One CSV per exported table, named for the concept rather than the SQLite table.
        for expected in ("observations.csv", "visits.csv", "media_clips.csv",
                         "live_sightings.csv", "life_events.csv", "deployments_coverage.csv"):
            assert expected in names, f"missing {expected}"
        rows = list(csv.reader(io.StringIO(zf.read("observations.csv").decode("utf-8"))))
        assert rows[0][0] == "id" and "species" in rows[0]      # header first
        assert len(rows) == 2                                    # header + the one detection
        doc = zf.read("DATA.md").decode("utf-8")
        assert "observations.csv" in doc and "ISO 8601" in doc
        # The dictionary must explain the two things a stranger cannot infer from the columns.
        assert "group" in doc.lower() and "human" in doc.lower()


def test_export_writes_empty_string_for_null_not_the_word_none(conn, db_path, tmp_path):
    """A NULL that exports as "None" silently becomes a species called None in a spreadsheet."""
    _det(conn, species=None)
    conn.commit()
    with zipfile.ZipFile(export.export_bundle(tmp_path, db_path=db_path)) as zf:
        rows = list(csv.reader(io.StringIO(zf.read("observations.csv").decode("utf-8"))))
    species = rows[1][rows[0].index("species")]
    assert species == ""
    assert "None" not in rows[1]


def test_export_survives_a_database_missing_newer_columns(conn, db_path, tmp_path):
    """An older DB (or a table this build doesn't have) exports what exists and says so in the
    dictionary, rather than raising -- the same read-only-clone contract the rest of db.py keeps."""
    _det(conn)
    conn.execute("DROP TABLE IF EXISTS life_events")
    conn.commit()
    with zipfile.ZipFile(export.export_bundle(tmp_path, db_path=db_path)) as zf:
        doc = zf.read("DATA.md").decode("utf-8")
        assert "life_events.csv" not in zf.namelist()
        assert "not present in this database" in doc


def test_export_dry_run_writes_nothing(conn, db_path, tmp_path):
    out = export.export_bundle(tmp_path, db_path=db_path, dry_run=True)
    assert not out.exists()
