"""
Smoke + sanity tests for stats.py -- the digest / overview engine, the most intricate read-side
logic in the project and (until now) untested. The review flagged it as the module most likely to
crash on a fresh / empty database, so these focus on the empty-DB paths plus a small populated
check. Pure DB logic; no GPU / camera / model.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

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


# ---- species_overview drops non-critter labels (the catalogue / "Rarely Seen" fix) -----
def test_species_overview_filters_non_critter(conn, db_path):
    _add_detection(conn, species="raccoon")
    _add_detection(conn, species="chair")          # a false-trigger human correction
    _add_detection(conn, species="not an animal")  # the clip-filter gate's label
    o = stats.species_overview(_cfg(db_path))
    names = {s["species"] for s in o["species"]}
    assert "raccoon" in names
    assert "chair" not in names and "not an animal" not in names


# ---- cast_rollcall: the named-cast last-seen / overdue roll -------------------------
def _add_named(conn, iid, *, species="raccoon", days_ago=0, conf=0.9):
    ts = (datetime.now().astimezone() - timedelta(days=days_ago)).isoformat()
    db.insert_detection(conn, timestamp=ts, source="glass_door_cam", detection_class="animal",
                        confidence=conf, bbox=(0, 0, 10, 10), frame_w=100, frame_h=100,
                        crop_path="crops/x.jpg", species=species, individual_id=iid, crop_quality=1.0)


def test_cast_rollcall_empty_db(conn, db_path):
    rc = stats.cast_rollcall(_cfg(db_path))
    assert isinstance(rc, dict) and rc.get("cast") == []


def test_cast_rollcall_excludes_placeholder_clusters(conn, db_path):
    _add_named(conn, "Notch")
    _add_named(conn, "raccoon_c01")               # reid auto-cluster -> not the named cast
    ids = [c["id"] for c in stats.cast_rollcall(_cfg(db_path))["cast"]]
    assert "Notch" in ids and "raccoon_c01" not in ids


def test_cast_rollcall_flags_overdue_regular(conn, db_path):
    for d in (12, 11, 10):                         # a regular (3 distinct days), last seen 10d ago
        _add_named(conn, "Gus", days_ago=d)
    _add_named(conn, "Solo", days_ago=12)          # a one-off, also long gone
    by = {c["id"]: c for c in stats.cast_rollcall(_cfg(db_path))["cast"]}
    assert by["Gus"]["overdue"] is True and by["Gus"]["regular"] is True
    assert by["Solo"]["overdue"] is False          # not a regular -> never "overdue"


def test_cast_rollcall_recent_regular_not_overdue(conn, db_path):
    for d in (3, 2, 1, 0):
        _add_named(conn, "Stan", days_ago=d)
    c = {x["id"]: x for x in stats.cast_rollcall(_cfg(db_path))["cast"]}["Stan"]
    assert c["regular"] is True and c["overdue"] is False and c["days_since"] == 0


# ---- review_queue: the prioritized "most likely mislabeled" pass --------------------
def test_review_queue_empty_db(conn, db_path):
    rq = stats.review_queue(_cfg(db_path))
    assert rq["crops"] == [] and rq["total"] == 0


def test_review_queue_flags_suspect_species_only(conn, db_path):
    _add_detection(conn, species="brown rat", conf=0.9)    # suspect label -> flagged
    _add_detection(conn, species="raccoon", conf=0.95)     # normal + confident -> not flagged
    flagged = {c["species"] for c in stats.review_queue(_cfg(db_path))["crops"]}
    assert "brown rat" in flagged and "raccoon" not in flagged


def test_review_queue_excludes_verified(conn, db_path):
    _add_detection(conn, species="brown rat", conf=0.9)
    conn.execute("UPDATE detections SET species_verified = 1")
    conn.commit()
    assert stats.review_queue(_cfg(db_path))["total"] == 0


def test_crops_page_clamps_negative_limit(conn, db_path):
    """A negative ?limit must not pass through to SQLite, where LIMIT -1 means UNBOUNDED and would
    dump the entire detections table. crops_page clamps limit to >=1 and offset to >=0 itself, so
    it's safe regardless of the caller (the dashboard's /api/crops takes the value from the URL)."""
    for _ in range(3):
        _add_detection(conn)
    page = stats.crops_page(_cfg(db_path), limit=-1, offset=-5)
    assert page["limit"] == 1 and page["offset"] == 0
    assert len(page["crops"]) <= 1


# ---- current_live_visit: the span the Live tab's "who's here now?" control names ----
def _at(conn, dt, species="raccoon"):
    db.insert_detection(conn, timestamp=dt.isoformat(), source="glass_door_cam",
                        detection_class="animal", confidence=0.9, bbox=(0, 0, 10, 10),
                        frame_w=100, frame_h=100, crop_path="crops/x.jpg",
                        species=species, crop_quality=1.0)


def test_current_live_visit_empty_db(conn, db_path):
    v = stats.current_live_visit(_cfg(db_path))
    assert v["count"] == 0


def test_current_live_visit_active_run(conn, db_path):
    now = datetime.now().astimezone()
    for s in (40, 25, 10, 1):                         # four detections in the last minute
        _at(conn, now - timedelta(seconds=s))
    conn.commit()
    v = stats.current_live_visit(_cfg(db_path))
    assert v["count"] == 4 and v["active"] is True
    assert v["species"] == {"raccoon": 4} and v["latest_age_s"] < 30


def test_current_live_visit_stops_at_the_gap(conn, db_path):
    """A gap >= visit_gap_minutes ends the span: only the most recent run is the 'current' visit."""
    now = datetime.now().astimezone()
    gap = config.CONFIG.visit_gap_minutes
    # Insert in capture order (oldest first), like the live rig -- so id order == time order, which
    # is what current_live_visit walks back over. The older run is separated by more than the gap.
    for m in (gap + 31, gap + 30):                    # an older run, must NOT be folded in
        _at(conn, now - timedelta(minutes=m))
    for s in (20, 5):                                 # the current run (last ~20s)
        _at(conn, now - timedelta(seconds=s))
    conn.commit()
    v = stats.current_live_visit(_cfg(db_path))
    assert v["count"] == 2 and v["active"] is True


# ---- compute_stats.by_day: per-day tallies across distinct days (guards the single-pass refactor) --
def test_compute_stats_by_day_across_days(conn, db_path):
    """by_day is built in one pass keyed by timestamp[:10]. Insert crops dated on three distinct days
    and assert the per-day crop counts + ascending day ordering (the refactor kept both correct)."""
    days = {"2026-06-10": 2, "2026-06-11": 3, "2026-06-12": 1}
    for day, n in days.items():
        for i in range(n):
            db.insert_detection(conn, timestamp=f"{day}T21:0{i}:00-07:00", source="glass_door_cam",
                                detection_class="animal", confidence=0.9, bbox=(0, 0, 10, 10),
                                frame_w=100, frame_h=100, crop_path="crops/x.jpg",
                                species="raccoon", crop_quality=1.0)
    conn.commit()
    s = stats.compute_stats(_cfg(db_path))
    by_day = s["by_day"]
    assert [d["day"] for d in by_day] == ["2026-06-10", "2026-06-11", "2026-06-12"]   # ascending
    assert {d["day"]: d["crops"] for d in by_day} == days
    assert s["total_crops"] == sum(days.values())


# ---- MOTION surfacing: visit_motion / individual_motion (phase-4 clip_tracks read-out) ------------
# clip_tracks carry NO individual_id via insert_clip_tracks (it's stamped later by
# set_clip_track_individual); the _track helper writes it directly so individual_motion has rows.
# NOTE: insert_clip_tracks REPLACES every (clip, model) row, so each _track call gets its OWN clip --
# that keeps tracks from clobbering each other and matches the real one-clip-one-track fixtures.
_CLIP_N = [0]


def _motion_clip(conn, *, source=db.SOURCE_GLASS_DOOR_CAM, started_at, ended_at):
    _CLIP_N[0] += 1
    return db.insert_clip(conn, source=source, clip_path=f"clips/x{_CLIP_N[0]}.mp4",
                          started_at=started_at, ended_at=ended_at, fps=10.0, width=1280, height=720,
                          frame_count=300, detection_count=12, max_confidence=0.9)


def _track(conn, clip_id, *, avg_speed=None, peak_speed=None, straightness=None,
           moving_frac=None, area_trend=None, individual_id=None, n_hits=9):
    """Insert one tracklet on a clip, optionally stamping individual_id directly (as the un-blend
    action would). Returns the new clip_tracks.id. One track per clip_id (insert replaces rows)."""
    db.insert_clip_tracks(conn, clip_id=clip_id, model="m", n_samples=300, tracklets=[
        {"track_json": "[[0,0.5,0.5,0.1,0.1,0.9]]", "n_hits": n_hits,
         "features": {"avg_speed": avg_speed, "peak_speed": peak_speed, "straightness": straightness,
                      "moving_frac": moving_frac, "area_trend": area_trend}}])
    tid = conn.execute("SELECT id FROM clip_tracks WHERE clip_id=? ORDER BY id DESC LIMIT 1",
                       (clip_id,)).fetchone()["id"]
    if individual_id is not None:
        conn.execute("UPDATE clip_tracks SET individual_id=? WHERE id=?", (individual_id, tid))
        conn.commit()
    return tid


def _linked_track(conn, individual_id, **feats):
    """A tracklet stamped to an individual, on its own throwaway clip (time window irrelevant --
    individual_motion queries by individual_id only)."""
    cid = _motion_clip(conn, started_at="2026-06-10T21:00:00-07:00",
                       ended_at="2026-06-10T21:00:30-07:00")
    return _track(conn, cid, individual_id=individual_id, **feats)


def _visit(conn, *, source=db.SOURCE_GLASS_DOOR_CAM, started_at, ended_at):
    vid = db.insert_visit(conn, source=source, species="raccoon", individual_id=None,
                          started_at=started_at, ended_at=ended_at, detection_count=3,
                          max_confidence=0.9, representative_detection_id=None)
    conn.commit()
    return vid


# -- visit_motion -----------------------------------------------------------------------------------
def test_visit_motion_empty_db(conn, db_path):
    assert stats.visit_motion(_cfg(db_path), 999) == {
        "tracks": 0, "avg_speed": None, "peak_speed": None, "straightness": None,
        "moving_frac": None, "area_trend": None, "approach": "steady"}


def test_visit_motion_overlapping_track_is_approach(conn, db_path):
    vid = _visit(conn, started_at="2026-06-10T21:00:00-07:00", ended_at="2026-06-10T21:05:00-07:00")
    cid = _motion_clip(conn, started_at="2026-06-10T21:01:00-07:00",
                       ended_at="2026-06-10T21:01:30-07:00")           # clip window inside the visit
    _track(conn, cid, avg_speed=0.12, peak_speed=0.30, straightness=0.9,
           moving_frac=0.8, area_trend=1.4)                            # area_trend > 1.15 -> approach
    m = stats.visit_motion(_cfg(db_path), vid)
    assert m["tracks"] == 1
    assert m["approach"] == "approach"
    assert m["avg_speed"] == pytest.approx(0.12, abs=1e-3)             # reflects the single track
    assert m["peak_speed"] == pytest.approx(0.30, abs=1e-3)
    assert m["area_trend"] == pytest.approx(1.4, abs=1e-3)
    for k in ("straightness", "moving_frac"):
        assert m[k] is not None


def test_visit_motion_ignores_non_overlapping_clip(conn, db_path):
    """A clip on the SAME source but a time window OUTSIDE the visit must not be counted -- proves
    the time-overlap filter (a source-only match would wrongly fold it in)."""
    vid = _visit(conn, started_at="2026-06-10T21:00:00-07:00", ended_at="2026-06-10T21:05:00-07:00")
    cid = _motion_clip(conn, started_at="2026-06-10T22:00:00-07:00",     # an hour later, no overlap
                       ended_at="2026-06-10T22:00:30-07:00")
    _track(conn, cid, avg_speed=0.5, area_trend=1.4)
    assert stats.visit_motion(_cfg(db_path), vid)["tracks"] == 0


def test_visit_motion_ignores_other_source(conn, db_path):
    """An overlapping clip on a DIFFERENT source is not this visit's -- filtered by source."""
    vid = _visit(conn, source=db.SOURCE_GLASS_DOOR_CAM,
                 started_at="2026-06-10T21:00:00-07:00", ended_at="2026-06-10T21:05:00-07:00")
    cid = _motion_clip(conn, source="trail_cam", started_at="2026-06-10T21:01:00-07:00",
                       ended_at="2026-06-10T21:01:30-07:00")             # overlaps in time, wrong src
    _track(conn, cid, avg_speed=0.5, area_trend=1.4)
    assert stats.visit_motion(_cfg(db_path), vid)["tracks"] == 0


# -- individual_motion ------------------------------------------------------------------------------
def test_individual_motion_no_linked_tracks(conn, db_path):
    assert stats.individual_motion(_cfg(db_path), "Notch") == {
        "tracks": 0, "avg_speed_mean": None, "avg_speed_median": None, "straightness": None,
        "moving_frac": None, "approach": 0, "retreat": 0, "steady": 0}


def test_individual_motion_counts_by_area_trend(conn, db_path):
    """Several tracklets stamped to one individual, mixed area_trend -> per-track approach/retreat/
    steady counts, plus a sane mean/median avg_speed."""
    _linked_track(conn, "Notch", avg_speed=0.10, straightness=0.9, moving_frac=0.8,
                  area_trend=1.40)                                        # approach
    _linked_track(conn, "Notch", avg_speed=0.20, area_trend=0.50)        # retreat
    _linked_track(conn, "Notch", avg_speed=0.30, area_trend=1.00)        # steady
    _linked_track(conn, "Notch", avg_speed=0.40, area_trend=1.05)        # steady
    m = stats.individual_motion(_cfg(db_path), "Notch")
    assert m["tracks"] == 4
    assert (m["approach"], m["retreat"], m["steady"]) == (1, 1, 2)
    assert m["approach"] + m["retreat"] + m["steady"] == m["tracks"]     # all have area data here
    assert m["avg_speed_mean"] == pytest.approx((0.10 + 0.20 + 0.30 + 0.40) / 4, abs=1e-3)
    assert m["avg_speed_median"] == pytest.approx(0.30, abs=1e-3)        # mid-element of sorted speeds
    assert m["straightness"] is not None and m["moving_frac"] is not None


def test_individual_motion_null_area_trend_counts_in_tracks_only(conn, db_path):
    """A track with NULL area_trend is counted in `tracks` but in NONE of approach/retreat/steady."""
    _linked_track(conn, "Stan", avg_speed=0.10, area_trend=1.40)        # approach
    _linked_track(conn, "Stan", avg_speed=0.20, area_trend=None)        # no direction
    m = stats.individual_motion(_cfg(db_path), "Stan")
    assert m["tracks"] == 2
    assert (m["approach"], m["retreat"], m["steady"]) == (1, 0, 0)
    assert m["approach"] + m["retreat"] + m["steady"] == 1              # the NULL one counts to none
