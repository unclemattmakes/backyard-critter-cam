"""
Tests for makingof_export's privacy rails -- the one code path whose failure publishes
something that should never be public. Two mechanisms, both pinned here:

- PRIVACY_DENY: label-level denylist (person labels, non-critter labels, the operator's
  config.privacy_deny_names). Filters by what a crop is CALLED.
- PRIVACY_DENY_IDS: id-level veto for detections whose label is simply wrong (the detector
  boxed something it shouldn't have and the classifier confidently misnamed it). Filters by
  what a crop IS, regardless of name.

Importing makingof_export is cheap (numpy + the db/stats helpers; no torch, no DB open).
"""
from __future__ import annotations

import sqlite3

import makingof_export
from stats import _NON_CRITTER


# ---- PRIVACY_DENY: the label denylist ------------------------------------------------
def test_privacy_deny_always_contains_person_labels():
    # Whatever the operator configures, a crop labelled as a person must never export.
    assert {"person", "people", "human"} <= makingof_export.PRIVACY_DENY


def test_privacy_deny_contains_the_non_critter_labels():
    # Human-correction labels (bricks, food, blur, ...) ride along from stats._NON_CRITTER.
    assert set(_NON_CRITTER) <= makingof_export.PRIVACY_DENY


# ---- PRIVACY_DENY_IDS: the id-level veto ---------------------------------------------
def test_deny_ids_sql_binds_rather_than_interpolates():
    frag, params = makingof_export._deny_ids_sql()
    # The ids travel as bound parameters; the SQL text carries only placeholders.
    assert frag.count("?") == len(makingof_export.PRIVACY_DENY_IDS)
    assert params == sorted(makingof_export.PRIVACY_DENY_IDS)
    for did in makingof_export.PRIVACY_DENY_IDS:
        assert str(did) not in frag


def test_deny_ids_sql_excludes_exactly_the_vetoed_rows():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE detections (id INTEGER PRIMARY KEY)")
    keep = [1, 2]
    rows = [(i,) for i in keep] + [(i,) for i in makingof_export.PRIVACY_DENY_IDS]
    c.executemany("INSERT INTO detections (id) VALUES (?)", rows)
    frag, params = makingof_export._deny_ids_sql()
    got = {r[0] for r in c.execute("SELECT id FROM detections WHERE 1=1" + frag, params)}
    assert got == set(keep)


def test_deny_ids_sql_respects_column_alias():
    frag, _ = makingof_export._deny_ids_sql("d.id")
    assert "d.id NOT IN" in frag
