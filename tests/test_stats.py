"""
Smoke + sanity tests for stats.py -- the digest / overview engine, the most intricate read-side
logic in the project and (until now) untested. The review flagged it as the module most likely to
crash on a fresh / empty database, so these focus on the empty-DB paths plus a small populated
check. Pure DB logic; no GPU / camera / model.
"""
from __future__ import annotations

from dataclasses import replace

import config
import db
import stats
import visits


def _cfg(db_path):
    return replace(config.CONFIG, db_path=db_path)


def _add_detection(conn, *, species="raccoon", cls="animal", conf=0.9):
    db.insert_detection(conn, timestamp=db.now_local_iso(), source="glass_door_cam",
                        detection_class=cls, confidence=conf, bbox=(0, 0, 10, 10),
                        frame_w=100, frame_h=100, crop_path="crops/x.jpg",
                        species=species, crop_quality=1.0)


# ---- empty database: must not crash (the review's main concern) --------------------
def test_compute_stats_empty_db(conn, db_path):
    s = stats.compute_stats(_cfg(db_path))
    assert s is None or s.get("total_crops", 0) == 0


def test_species_overview_empty_db(conn, db_path):
    o = stats.species_overview(_cfg(db_path))
    assert o is None or isinstance(o, dict)


def test_period_digest_empty_db(conn, db_path):
    d = stats.period_digest(_cfg(db_path))
    assert isinstance(d, dict)            # returns a dict (empty:true), never raises


# ---- small populated database ------------------------------------------------------
def test_compute_stats_counts_crops(conn, db_path):
    for _ in range(3):
        _add_detection(conn)
    s = stats.compute_stats(_cfg(db_path))
    assert s is not None and s["total_crops"] == 3


def test_period_digest_with_data_does_not_crash(conn, db_path):
    for _ in range(3):
        _add_detection(conn)
    visits.build_visits(conn, config.CONFIG.visit_gap_minutes, verbose=False)
    d = stats.period_digest(_cfg(db_path))
    assert isinstance(d, dict)
