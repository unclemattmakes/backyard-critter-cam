"""Tests for the night-by-night naming mode: VisitMatcher.night_groups, the queue payload that
carries it, and eval.night_group_purity, which replays it on visits already named.

Timestamps are fixed (21:00 on a fixed date) rather than relative to now, because a night key
is the date of (timestamp - 12h): a relative clock would let two "same night" visits straddle
noon and land on different nights depending on when the suite runs.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

import config
import db
import eval as evalmod
import individuals
import web

BASE = datetime(2026, 6, 10, 21, 0, 0, tzinfo=datetime.now().astimezone().tzinfo)
DAY = 24 * 60
NIGHT0, NIGHT1, NIGHT2 = "2026-06-10", "2026-06-11", "2026-06-12"


def _u(*xs):
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


# STAN_OLD is the same animal a night earlier: close to tonight's look, but not identical.
STAN, STAN_ISH, STAN_OLD, NOTCH = (_u(1, 0, 0, 0), _u(1, 0.1, 0, 0), _u(1, 0, 0.3, 0),
                                   _u(0, 1, 0, 0))


def _cfg(db_path):
    # Shipped defaults, so a local config_local.py can't change what these tests mean.
    return replace(config.Config(), db_path=db_path, reid_co_presence_min=1)


def _visit(conn, vec, start_min, *, source=db.SOURCE_GLASS_DOOR_CAM, pair=False):
    """A three-frame visit whose crops all embed as `vec`. `pair` puts a second, separated box in
    every frame, which is what the 2+-animals badge keys on."""
    ids = []
    for k in range(3):
        ts = (BASE + timedelta(minutes=start_min + k)).isoformat()
        boxes = [(0, 0, 10, 10)] + ([(60, 60, 70, 70)] if pair else [])
        for j, box in enumerate(boxes):
            d = db.insert_detection(
                conn, timestamp=ts, source=source, detection_class="animal", confidence=0.9,
                bbox=box, frame_w=100, frame_h=100,
                crop_path=f"crops/{source}-{start_min}-{k}-{j}.jpg", species="raccoon",
                crop_quality=10.0)
            db.insert_embedding(conn, d, individuals.EMBED_MODEL, len(vec),
                                np.asarray(vec, dtype=np.float32).tobytes())
            ids.append(d)
    vid = db.insert_visit(
        conn, source=source, species="raccoon", individual_id=None,
        started_at=(BASE + timedelta(minutes=start_min)).isoformat(),
        ended_at=(BASE + timedelta(minutes=start_min + 3)).isoformat(),
        detection_count=len(ids), max_confidence=0.9, representative_detection_id=ids[0])
    db.assign_visit(conn, ids, vid)
    conn.commit()
    return vid


@pytest.fixture
def nights(conn):
    """Night 0: a Stan and a Notch template. Night 1: two Stan look-alikes and a Notch nobody has
    named, a Stan already confirmed that night, a pair visit and a trail-cam visit. Night 2: one
    unnamed Notch, so it is the newest night with work left."""
    ids = {}
    ids["stan_t"] = _visit(conn, STAN_OLD, 0)
    ids["notch_t"] = _visit(conn, NOTCH, 30)
    ids["a1"] = _visit(conn, STAN_ISH, DAY)
    ids["a2"] = _visit(conn, STAN, DAY + 120)
    ids["b1"] = _visit(conn, NOTCH, DAY + 60)
    ids["stan_n1"] = _visit(conn, STAN, DAY + 200)       # 00:20, still night 1
    ids["pair"] = _visit(conn, STAN, DAY + 240, pair=True)
    ids["trail"] = _visit(conn, STAN, DAY + 90, source=db.SOURCE_TRAIL_CAM_SD)
    ids["b2"] = _visit(conn, NOTCH, 2 * DAY)
    for v, name in (("stan_t", "Stan"), ("notch_t", "Notch"), ("stan_n1", "Stan")):
        db.label_visit(conn, ids[v], name)
    return ids


def _matcher(conn, db_path):
    return individuals.VisitMatcher(conn, "raccoon", _cfg(db_path))


def test_defaults_to_the_newest_night_with_work_and_lists_the_others(nights, conn, db_path):
    out = _matcher(conn, db_path).night_groups()
    assert out["night"] == NIGHT2
    assert [g["visits"] for g in out["groups"]] == [[nights["b2"]]]
    # Newest first; night 0 has nothing left, so it isn't offered.
    assert out["nights"] == [{"night": NIGHT2, "n_visits": 1}, {"night": NIGHT1, "n_visits": 3}]


def test_groups_one_night_by_look_and_says_what_it_left_out(nights, conn, db_path):
    out = _matcher(conn, db_path).night_groups(night=NIGHT1)
    assert out["night"] == NIGHT1 and out["n_visits"] == 3
    big, small = out["groups"]                               # biggest first
    assert big["visits"] == sorted([nights["a1"], nights["a2"]])
    assert small["visits"] == [nights["b1"]]
    assert big["source"] == db.SOURCE_GLASS_DOOR_CAM
    # The Stan group's nearest template is the Stan confirmed earlier THIS night.
    assert big["suggestion"]["name"] == "Stan" and big["suggestion"]["same_night"] is True
    assert big["suggestion"]["via_visit"] == nights["stan_n1"]
    assert small["suggestion"]["name"] == "Notch" and small["suggestion"]["same_night"] is False
    # Confirmed visits are not offered again; the pair goes to un-blend and the trail cam has no
    # templates, and both are counted rather than silently dropped.
    offered = {v for g in out["groups"] for v in g["visits"]}
    assert nights["stan_n1"] not in offered
    assert out["skipped"] == {"multi": 1, "untemplated_source": 1}


def test_a_rejected_visit_is_not_offered(nights, conn, db_path):
    db.label_visit(conn, nights["b1"], None, reject=True)
    out = _matcher(conn, db_path).night_groups(night=NIGHT1)
    assert [g["visits"] for g in out["groups"]] == [sorted([nights["a1"], nights["a2"]])]


def test_an_auto_named_visit_is_offered_with_its_auto_name(nights, conn, db_path):
    db.label_visit(conn, nights["a1"], "Stan", source="auto")
    out = _matcher(conn, db_path).night_groups(night=NIGHT1)
    big = out["groups"][0]
    assert dict(zip(big["visits"], big["auto_as"])) == {nights["a1"]: "Stan", nights["a2"]: None}


def test_a_night_with_nothing_left_comes_back_empty_not_as_another_night(nights, conn, db_path):
    out = _matcher(conn, db_path).night_groups(night=NIGHT0)
    assert out["night"] == NIGHT0 and out["groups"] == []


def test_queue_payload_carries_the_groups_with_a_thumbnail_per_visit(nights, db_path):
    out = web._reid_queue(_cfg(db_path), species="raccoon", night=NIGHT1)
    t = out["tonight"]
    assert t["night"] == NIGHT1
    members = t["groups"][0]["members"]
    assert [m["visit_id"] for m in members] == t["groups"][0]["visits"]
    assert all(m["rep_crop"] and m["started"] for m in members)
    assert "lapse" in t["groups"][0]["suggestion"]


def test_no_templates_means_no_night_groups(conn, db_path):
    _visit(conn, STAN, 0)
    _visit(conn, STAN, 60)
    assert web._reid_queue(_cfg(db_path), species="raccoon")["tonight"] is None


def test_purity_replay_counts_what_bulk_naming_would_get_wrong(nights, conn, db_path):
    # Night 1 now holds three confirmed solo visits: two Stans and a Notch.
    db.label_visit(conn, nights["a1"], "Stan")
    db.label_visit(conn, nights["b1"], "Notch")
    m = _matcher(conn, db_path)
    solo = {v: n for v, n in m.confirmed.items() if not m.is_multi(v)}
    rows = {r["distance"]: r for r in evalmod.night_group_purity(m, solo, distances=(0.45, 2.0))}
    assert rows[0.45]["configured"] is True
    assert rows[0.45]["nights"] == 2              # night 0 (Stan + Notch) and night 1
    assert rows[0.45]["minority_visits"] == 0
    assert rows[0.45]["groups"] == 4              # Stan / Notch on each night
    # A cut that merges everything puts Notch inside the Stan group on both nights.
    assert rows[2.0]["groups"] == 2 and rows[2.0]["minority_visits"] == 2
