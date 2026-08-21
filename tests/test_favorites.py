"""
FAVOURITES -- "keep this one": the dashboard's ♡ on a crop or a visit, and the album it fills.

Two things are worth pinning here, and they are the two that would rot silently.

1. A STAR IS NOT A LABEL. Every other human verdict in this project is evidence about the animal
   and feeds the models. This one must never leak into that: starring a crop may not touch its
   species, its verified flag, or its individual_id. If it ever does, "the cutest photo of Stan"
   quietly becomes training evidence that it IS Stan.

2. A STARRED VISIT MUST SURVIVE RE-CLUSTERING. Visits are not durable objects here -- stats
   .visits_page re-clusters them out of raw detections on every request and visits.py renumbers
   the ledger from scratch -- so favourites key a visit on (source, started_at) and the gallery
   re-derives the span. The case that breaks a naive design is starring a visit that is still in
   progress: its end moves, and the star must stay on it.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import urllib.error
import urllib.request
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

import config
import db
import stats
import web


def _cfg(db_path, **kw):
    """A config on the throwaway DB, from the SHIPPED defaults (see test_web._rq_cfg)."""
    return replace(config.Config(), db_path=db_path, **kw)


def _det(conn, when, *, source=db.SOURCE_GLASS_DOOR_CAM, species="raccoon", conf=0.9,
         individual=None):
    """One detection at `when` (a datetime), returning its id."""
    return db.insert_detection(
        conn, timestamp=when.isoformat(), source=source, detection_class="animal",
        confidence=conf, bbox=(10, 10, 90, 90), frame_w=100, frame_h=100,
        crop_path=f"crops/{when.strftime('%H%M%S%f')}.jpg", species=species,
        individual_id=individual)


def _at(minute, *, day="2026-08-19"):
    return datetime.fromisoformat(f"{day}T21:00:00-07:00") + timedelta(minutes=minute)


# =====================================================================================
# The store: one star per thing, idempotent, and validated in exactly one place.
# =====================================================================================
def test_starring_a_crop_round_trips(conn):
    det = _det(conn, _at(0))
    row = db.add_favorite(conn, "detection", detection_id=det, labeled_by="matt")
    assert row["kind"] == "detection" and row["detection_id"] == det
    assert row["labeled_by"] == "matt" and row["created_at"]
    assert db.favorite_keys(conn)["detections"] == {det}


def test_starring_twice_keeps_one_row_and_the_original_keeper(conn):
    """A double-tap on a phone over a flaky LAN must not leave two rows -- and whoever kept it
    FIRST is the fact worth preserving, so a second tap does not re-attribute it."""
    det = _det(conn, _at(0))
    first = db.add_favorite(conn, "detection", detection_id=det, labeled_by="matt")
    again = db.add_favorite(conn, "detection", detection_id=det, labeled_by="someone else")
    assert again["id"] == first["id"]
    assert again["labeled_by"] == "matt"
    assert again["created_at"] == first["created_at"]
    assert len(db.favorites(conn)) == 1


def test_unstarring_is_idempotent(conn):
    det = _det(conn, _at(0))
    db.add_favorite(conn, "detection", detection_id=det)
    assert db.remove_favorite(conn, "detection", detection_id=det) is True
    assert db.remove_favorite(conn, "detection", detection_id=det) is False
    assert db.favorite_keys(conn)["detections"] == set()


def test_a_visit_is_keyed_on_its_start_not_its_end(conn):
    """THE case a visit-id (or a start+end) key gets wrong: you star a visit while the animal is
    still on camera, then it stays another four minutes. Same star, updated span."""
    key = dict(source=db.SOURCE_GLASS_DOOR_CAM, started_at=_at(0).isoformat())
    first = db.add_favorite(conn, "visit", ended_at=_at(1).isoformat(), **key)
    later = db.add_favorite(conn, "visit", ended_at=_at(5).isoformat(), **key)
    assert later["id"] == first["id"]
    assert len(db.favorites(conn)) == 1
    assert db.favorite_keys(conn)["visits"] == {(db.SOURCE_GLASS_DOOR_CAM, _at(0).isoformat())}


def test_the_same_moment_on_two_cameras_is_two_favourites(conn):
    """Both rigs can be recording at once; the key is (source, start), not the clock alone."""
    start = _at(0).isoformat()
    db.add_favorite(conn, "visit", source=db.SOURCE_GLASS_DOOR_CAM, started_at=start)
    db.add_favorite(conn, "visit", source=db.SOURCE_TRAIL_CAM_SD, started_at=start)
    assert len(db.favorite_keys(conn)["visits"]) == 2


def test_notes_are_set_and_cleared_but_never_create_a_star(conn):
    det = _det(conn, _at(0))
    assert db.set_favorite_note(conn, "detection", "nice", detection_id=det) is None
    assert db.favorites(conn) == []                      # a note is not a way to star something
    db.add_favorite(conn, "detection", detection_id=det)
    assert db.set_favorite_note(conn, "detection", "  kits, first night out  ",
                                detection_id=det)["note"] == "kits, first night out"
    assert db.set_favorite_note(conn, "detection", "", detection_id=det)["note"] is None


def test_a_long_note_is_capped_not_refused(conn):
    det = _det(conn, _at(0))
    row = db.add_favorite(conn, "detection", detection_id=det, note="x" * 5000)
    assert len(row["note"]) == db._FAV_NOTE_MAX


def test_a_half_keyed_or_unknown_favourite_is_refused(conn):
    """Validation lives in db._fav_key alone, so no caller can write a 'visit' carrying a
    detection_id (or the reverse) however it phrases the request."""
    for bad in (dict(kind="clip", detection_id=1),
                dict(kind="detection"),
                dict(kind="detection", detection_id="not a number"),
                dict(kind="visit", source="glass_door_cam"),
                dict(kind="visit", started_at="2026-08-19T21:00:00-07:00"),
                dict(kind="")):
        kind = bad.pop("kind")
        with pytest.raises(ValueError):
            db.add_favorite(conn, kind, **bad)


def test_starring_a_crop_changes_nothing_about_the_animal(conn):
    """The whole point of the separation: a star is taste, not evidence."""
    det = _det(conn, _at(0), species="raccoon")
    before = dict(conn.execute("SELECT * FROM detections WHERE id = ?", (det,)).fetchone())
    db.add_favorite(conn, "detection", detection_id=det, note="the good one", labeled_by="matt")
    after = dict(conn.execute("SELECT * FROM detections WHERE id = ?", (det,)).fetchone())
    assert before == after


def test_purging_a_crop_takes_its_star_with_it(conn):
    """The furniture sweeps really do DELETE detections. A star pointing at a row that no longer
    exists is not a fact about anything, so the FK cascades rather than blocking the purge."""
    det = _det(conn, _at(0))
    db.add_favorite(conn, "detection", detection_id=det)
    conn.execute("DELETE FROM detections WHERE id = ?", (det,))
    conn.commit()
    assert db.favorite_keys(conn)["detections"] == set()


def test_an_unmigrated_database_reads_as_no_favourites(conn, db_path):
    """The read-only-clone contract every reader here follows: an old DB (or a reporting clone
    nobody has migrated) renders with no stars rather than 500ing."""
    conn.execute("DROP TABLE favorites")
    conn.commit()
    assert db.favorites(conn) == []
    assert db.favorite_keys(conn) == {"detections": set(), "visits": set()}


# =====================================================================================
# The surfaces: a heart renders filled wherever the thing is shown.
# =====================================================================================
def test_the_photo_grid_marks_which_crops_are_kept(conn, db_path):
    kept, plain = _det(conn, _at(0)), _det(conn, _at(1))
    db.add_favorite(conn, "detection", detection_id=kept)
    by_id = {c["id"]: c for c in stats.crops_page(_cfg(db_path))["crops"]}
    assert by_id[kept]["favorite"] is True
    assert by_id[plain]["favorite"] is False


def test_the_species_sheet_and_review_queue_mark_them_too(conn, db_path):
    """Same crop, four surfaces -- a heart that is filled on one and hollow on another is a lie
    about what you already kept."""
    kept = _det(conn, _at(0), species="brown rat", conf=0.2)   # 'brown rat' => review-suspect
    db.add_favorite(conn, "detection", detection_id=kept)
    cfg = _cfg(db_path)
    assert next(c for c in stats.species_crops(cfg, "brown rat") if c["id"] == kept)["favorite"]
    assert next(c for c in stats.review_queue(cfg)["crops"] if c["id"] == kept)["favorite"]


def test_the_visit_log_marks_kept_visits(conn, db_path):
    for m in (0, 1, 2):
        _det(conn, _at(m))
    for m in (60, 61):                                   # a second visit, an hour later
        _det(conn, _at(m))
    cfg = _cfg(db_path)
    visits = stats.visits_page(cfg)["visits"]
    assert len(visits) == 2 and all(v["favorite"] is False for v in visits)

    target = visits[-1]                                  # the older one
    db.add_favorite(conn, "visit", source=target["source"], started_at=target["start"],
                    ended_at=target["end"])
    marked = {v["start"]: v["favorite"] for v in stats.visits_page(cfg)["visits"]}
    assert marked[target["start"]] is True
    assert sum(marked.values()) == 1


# =====================================================================================
# The album.
# =====================================================================================
def test_the_album_lists_kept_crops_and_visits_newest_star_first(conn, db_path):
    det = _det(conn, _at(0))
    for m in (30, 31, 32):
        _det(conn, _at(m))
    db.add_favorite(conn, "detection", detection_id=det, note="the good one", labeled_by="matt")
    db.add_favorite(conn, "visit", source=db.SOURCE_GLASS_DOOR_CAM,
                    started_at=_at(30).isoformat(), ended_at=_at(32).isoformat())

    out = stats.favorites_page(_cfg(db_path))
    assert out["total"] == 2 and out["crops"] == 1 and out["visits"] == 1
    assert [f["kind"] for f in out["favorites"]] == ["visit", "detection"]   # newest star first

    visit = out["favorites"][0]["visit"]
    assert visit["count"] == 3 and visit["title"] == "raccoon" and visit["favorite"] is True
    assert visit["start"] == _at(30).isoformat()

    crop = out["favorites"][1]
    assert crop["note"] == "the good one" and crop["labeled_by"] == "matt"
    assert crop["crop"]["id"] == det and crop["gone"] is False


def test_a_kept_visit_shows_its_finished_span_not_the_one_it_was_starred_at(conn, db_path):
    """Starred at minute 1 of a visit that ran to minute 6: the album re-derives the visit from
    the detections, so the card grows up. (`ended_at` is kept only for the tombstone case.)"""
    cfg = _cfg(db_path)
    _det(conn, _at(0))
    _det(conn, _at(1))
    db.add_favorite(conn, "visit", source=db.SOURCE_GLASS_DOOR_CAM,
                    started_at=_at(0).isoformat(), ended_at=_at(1).isoformat())
    for m in (3, 4, 6):                                  # gaps < visit_gap_minutes: same visit
        _det(conn, _at(m))

    visit = stats.favorites_page(cfg)["favorites"][0]["visit"]
    assert visit["count"] == 5
    assert visit["end"] == _at(6).isoformat()
    assert visit["minutes"] == 6.0


def test_a_kept_visit_that_merged_forward_still_resolves(conn, db_path):
    """A later import can put a detection in FRONT of a starred visit, moving the cluster's start
    behind the star. The album asks for the visit CONTAINING that moment, so it widens instead of
    losing the favourite."""
    cfg = _cfg(db_path)
    for m in (10, 11, 12):
        _det(conn, _at(m))
    db.add_favorite(conn, "visit", source=db.SOURCE_GLASS_DOOR_CAM,
                    started_at=_at(10).isoformat(), ended_at=_at(12).isoformat())
    _det(conn, _at(8))                                   # imported later; within the gap

    visit = stats.favorites_page(cfg)["favorites"][0]["visit"]
    assert visit["start"] == _at(8).isoformat() and visit["count"] == 4


def test_a_kept_visit_does_not_swallow_the_next_one(conn, db_path):
    """The re-derivation must stop at the first real gap, or the album shows a card that no other
    surface would ever draw."""
    cfg = _cfg(db_path)
    for m in (0, 1):
        _det(conn, _at(m))
    for m in (40, 41, 42):
        _det(conn, _at(m))
    db.add_favorite(conn, "visit", source=db.SOURCE_GLASS_DOOR_CAM,
                    started_at=_at(0).isoformat(), ended_at=_at(1).isoformat())
    visit = stats.favorites_page(cfg)["favorites"][0]["visit"]
    assert visit["count"] == 2 and visit["end"] == _at(1).isoformat()


def test_a_visit_too_long_for_the_scan_bound_says_so(conn, db_path, monkeypatch):
    """A raccoon that settles in for an hour can out-run the album's per-visit row cap. Better a
    card that admits it is showing the first N than one that quietly reports a shorter visit than
    the Visit Log draws for the same span."""
    monkeypatch.setattr(stats, "_FAV_VISIT_SCAN_ROWS", 4)
    for m in (0, 1, 2, 3, 4, 5):
        _det(conn, _at(m))
    db.add_favorite(conn, "visit", source=db.SOURCE_GLASS_DOOR_CAM,
                    started_at=_at(0).isoformat(), ended_at=_at(1).isoformat())
    visit = stats.favorites_page(_cfg(db_path))["favorites"][0]["visit"]
    assert visit["truncated"] is True and visit["count"] == 4

    monkeypatch.setattr(stats, "_FAV_VISIT_SCAN_ROWS", 6000)     # room to spare: no caveat
    visit = stats.favorites_page(_cfg(db_path))["favorites"][0]["visit"]
    assert "truncated" not in visit and visit["count"] == 6


def test_a_favourite_whose_observations_are_gone_is_listed_not_dropped(conn, db_path):
    """A purge can empty the span behind a star. Saying so beats a favourite that evaporates --
    and the note someone wrote is still theirs."""
    cfg = _cfg(db_path)
    for m in (0, 1):
        _det(conn, _at(m))
    db.add_favorite(conn, "visit", source=db.SOURCE_GLASS_DOOR_CAM,
                    started_at=_at(0).isoformat(), ended_at=_at(1).isoformat(),
                    note="the night of the kits")
    conn.execute("DELETE FROM detections")
    conn.commit()

    out = stats.favorites_page(cfg)
    assert out["total"] == 1
    gone = out["favorites"][0]
    assert gone["gone"] is True and gone["visit"] is None
    assert gone["note"] == "the night of the kits"
    assert gone["started_at"] == _at(0).isoformat()      # enough to say WHICH visit it was


def test_an_empty_album_is_an_empty_album(conn, db_path):
    assert stats.favorites_page(_cfg(db_path)) == {"favorites": [], "total": 0,
                                                   "crops": 0, "visits": 0}


def test_the_album_reads_a_database_that_does_not_exist_yet(tmp_path):
    cfg = _cfg(tmp_path / "nope.db")
    assert stats.favorites_page(cfg)["favorites"] == []


# =====================================================================================
# The endpoint, over a real socket.
# =====================================================================================
def _post(port, path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Origin": f"http://127.0.0.1:{port}"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


@pytest.fixture
def served(conn, db_path):
    """The real server on a loopback port, pointed at the throwaway DB."""
    cfg = _cfg(db_path, web_host="127.0.0.1", web_port=0)
    buffers = {cfg.source: web.FrameBuffer()}
    server = web.make_server(cfg, buffers, {cfg.source: web.CameraControlBridge()})
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=5)


def test_favourite_endpoint_round_trip(conn, served):
    det = _det(conn, _at(0))
    for m in (30, 31):
        _det(conn, _at(m))

    status, body = _post(served, "/api/favorite",
                         {"kind": "detection", "detection_id": det, "logged_by": "matt"})
    assert status == 200 and body["favorite"] is True and body["row"]["labeled_by"] == "matt"

    status, body = _post(served, "/api/favorite",
                         {"kind": "visit", "source": db.SOURCE_GLASS_DOOR_CAM,
                          "start": _at(30).isoformat(), "end": _at(31).isoformat(),
                          "note": "worth keeping"})
    assert status == 200 and body["favorite"] is True

    status, album = _get(served, "/api/favorites")
    assert status == 200 and album["total"] == 2
    assert {f["kind"] for f in album["favorites"]} == {"detection", "visit"}
    assert next(f for f in album["favorites"] if f["kind"] == "visit")["note"] == "worth keeping"

    status, body = _post(served, "/api/favorite",
                         {"kind": "detection", "detection_id": det, "on": False})
    assert status == 200 and body["favorite"] is False and body["removed"] is True
    assert _get(served, "/api/favorites")[1]["total"] == 1


def test_a_note_can_be_cleared_through_the_endpoint(conn, served):
    det = _det(conn, _at(0))
    _post(served, "/api/favorite", {"kind": "detection", "detection_id": det, "note": "keep"})
    _post(served, "/api/favorite", {"kind": "detection", "detection_id": det, "note": ""})
    assert _get(served, "/api/favorites")[1]["favorites"][0]["note"] is None


def test_a_favourite_the_server_cannot_key_is_a_400_not_a_500(conn, served):
    for bad in ({"kind": "clip", "detection_id": 1},
                {"kind": "visit", "source": db.SOURCE_GLASS_DOOR_CAM},
                {"detection_id": 1},
                {}):
        status, body = _post(served, "/api/favorite", bad)
        assert status == 400 and "error" in body, bad


def test_a_viewer_cannot_keep_anything(conn, db_path):
    """The operator/viewer gate covers new endpoints by construction (do_POST refuses everything
    but the sighting log). Pinned here because the album is exactly the surface a household guest
    will find first -- and the dashboard hides the ♡ from them to match."""
    cfg = _cfg(db_path, web_host="127.0.0.1", web_port=0, operator_token="sesame")
    buffers = {cfg.source: web.FrameBuffer()}
    server = web.make_server(cfg, buffers, {cfg.source: web.CameraControlBridge()})
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        # Loopback is implicitly the operator (you are at the rig), so this asks the pure rule
        # the socket cannot: a tokened rig, a request from elsewhere on the LAN, no token sent.
        assert web._operator_decision("sesame", "192.168.1.44", None) is False
        assert web._operator_decision("sesame", "192.168.1.44", "sesame") is True
        assert _get(port, "/api/favorites")[1]["total"] == 0      # reading is never gated
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=5)
