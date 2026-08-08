"""
Unit tests for web.py's pure helpers -- HTTP Range parsing (used to stream/seek video clips),
the camera-control whitelist, the media-path containment check (path-traversal guard), and the
cross-site guard that decides which POSTs the dashboard will act on. Most of these
never start a server or open a socket; they exercise the parsing/validation logic the dashboard
relies on. (web.py imports only stdlib + db/stats/behavior/config, so importing it is cheap.)

Plus the re-ID REVIEW QUEUE (_reid_queue and its mode filters), which does touch a DB -- always a
throwaway one from the `conn` fixture, never the live backyard.db. The queue is where every
appearance template comes from, so its filters, its pagination and its honesty about what it
cannot know are load-bearing, not cosmetic. Plus the ROSTER on that same cast surface -- the human
recording that an individual has stopped visiting, the one fact the matcher cannot infer. Two
tests at the end bind a real loopback socket to smoke the GET and POST endpoints end to end.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import config
import db
import individuals
import web


# ---- _parse_range: 'bytes=START-END' against a known file size ---------------------
def test_parse_range_explicit():
    assert web._parse_range("bytes=0-99", 1000) == (0, 99, False)


def test_parse_range_open_ended():
    assert web._parse_range("bytes=100-", 1000) == (100, 999, True)


def test_parse_range_suffix_last_n_bytes():
    assert web._parse_range("bytes=-50", 1000) == (950, 999, False)


def test_parse_range_clamps_end_to_size():
    assert web._parse_range("bytes=0-99999", 1000) == (0, 999, False)


def test_parse_range_rejects_garbage_and_out_of_bounds():
    assert web._parse_range("bytes=abc", 1000) is None
    assert web._parse_range("bytes=2000-", 1000) is None     # start past EOF
    assert web._parse_range("bytes=-0", 1000) is None        # zero-length suffix


# ---- _clean_settings: only whitelisted controls, coerced to float (None passes) ----
def test_clean_settings_filters_and_coerces():
    out = web._clean_settings({"EXPOSURE": "5", "gain": None, "bogus": 1, "FOCUS": 12.5})
    assert out == {"EXPOSURE": 5.0, "gain": None, "FOCUS": 12.5}


def test_clean_settings_drops_unknown_keys():
    assert web._clean_settings({"rm -rf": 1, "DROP TABLE": 2}) == {}


# ---- _is_within: media path containment (path-traversal guard) ---------------------
def test_is_within_true_for_child(tmp_path):
    child = tmp_path / "crops" / "2026" / "x.jpg"
    assert web._is_within(child, tmp_path) is True


def test_is_within_false_for_outside(tmp_path):
    assert web._is_within(Path("/etc/passwd"), tmp_path / "crops") is False


# ---- _is_allowed_host: the DNS-rebinding guard --------------------------------------
# The peer-IP check can't stop a site that resolves its OWN name to the rig's LAN IP -- the
# request comes from the victim's local browser. The Host header is what gives it away.
def test_allowed_host_localhost_and_loopback():
    assert web._is_allowed_host("localhost:8000") is True
    assert web._is_allowed_host("127.0.0.1:8000") is True
    assert web._is_allowed_host("[::1]:8000") is True


def test_allowed_host_private_and_link_local_literals():
    assert web._is_allowed_host("192.168.1.23:8000") is True
    assert web._is_allowed_host("10.0.0.5") is True
    assert web._is_allowed_host("169.254.9.9") is True


def test_allowed_host_rejects_public_names_and_ips():
    # The rebinding shape: the attacker's own hostname, pointed at the rig's LAN IP.
    assert web._is_allowed_host("evil.example:8000") is False
    assert web._is_allowed_host("8.8.8.8") is False


def test_allowed_host_accepts_configured_web_host_only_when_configured():
    assert web._is_allowed_host("mycam.lan:8000", web_host="mycam.lan") is True
    assert web._is_allowed_host("mycam.lan:8000") is False


def test_allowed_host_missing_header_allowed():
    # HTTP/1.0 tooling may omit Host; a browser's rebinding fetch cannot.
    assert web._is_allowed_host("") is True


def test_allowed_host_ipv4_mapped_ipv6():
    assert web._is_allowed_host("[::ffff:192.168.1.23]:8000") is True
    assert web._is_allowed_host("[::ffff:8.8.8.8]:8000") is False


# ---- the bind default: loopback unless the operator deliberately opens up -----------
def test_web_host_defaults_to_loopback():
    # A silent regression here (0.0.0.0) would put the no-auth dashboard on the LAN by default.
    assert config.Config().web_host == "127.0.0.1"


# ---- _csrf_refusal: only THIS dashboard's own pages may POST ------------------------
# The peer-IP and Host guards both pass for a request the operator's own browser makes from some
# other site, so a POST additionally needs a same-origin Origin and a JSON Content-Type.
_JSON = {"Content-Type": "application/json"}


def _headers(origin=None, host="127.0.0.1:8000", ctype="application/json"):
    h = {"Host": host}
    if origin is not None:
        h["Origin"] = origin
    if ctype is not None:
        h["Content-Type"] = ctype
    return h


def test_csrf_allows_own_origin_post():
    assert web._csrf_refusal("POST", _headers("http://127.0.0.1:8000"), "127.0.0.1", 8000) is None
    assert web._csrf_refusal("POST", _headers("http://localhost:8000", host="localhost:8000"),
                             "127.0.0.1", 8000) is None


def test_csrf_allows_lan_origin_matching_own_host():
    # LAN mode (bound to 0.0.0.0, reached by the machine's IP): the browser sends that IP as both
    # Origin and Host, which is the same-origin case and must keep working.
    assert web._csrf_refusal("POST", _headers("http://192.168.1.50:8000", host="192.168.1.50:8000"),
                             "0.0.0.0", 8000) is None


def test_csrf_refuses_cross_site_origin():
    assert web._csrf_refusal("POST", _headers("http://evil.example"), "127.0.0.1", 8000) is not None
    # A rebinding name can't satisfy both headers: the Host guard rejects it, so the Origin/Host
    # match is not available to it.
    assert web._csrf_refusal("POST", _headers("http://evil.example", host="evil.example"),
                             "127.0.0.1", 8000) is not None
    assert web._csrf_refusal("POST", _headers("null"), "127.0.0.1", 8000) is not None


def test_csrf_refuses_missing_origin():
    # fetch/XHR always send Origin, so an absent one is hand-rolled tooling -- refuse it rather
    # than blanket-trusting the omission.
    assert web._csrf_refusal("POST", {"Host": "127.0.0.1:8000", **_JSON},
                             "127.0.0.1", 8000) is not None


def test_csrf_refuses_non_json_content_type():
    # text/plain is the "simple request" that needs no CORS preflight -- the actual attack shape.
    assert web._csrf_refusal("POST", _headers("http://127.0.0.1:8000", ctype="text/plain"),
                             "127.0.0.1", 8000) is not None
    assert web._csrf_refusal("POST", _headers("http://127.0.0.1:8000", ctype=None),
                             "127.0.0.1", 8000) is not None
    # A charset parameter is fine.
    assert web._csrf_refusal(
        "POST", _headers("http://127.0.0.1:8000", ctype="application/json; charset=utf-8"),
        "127.0.0.1", 8000) is None


def test_csrf_leaves_gets_alone():
    # Reads are unaffected: no Origin, no Content-Type, cross-site referrer -- all still served.
    assert web._csrf_refusal("GET", {"Host": "127.0.0.1:8000"}, "127.0.0.1", 8000) is None
    assert web._csrf_refusal("GET", _headers("http://evil.example", ctype="text/plain"),
                             "127.0.0.1", 8000) is None


def test_is_json_content_type_variants():
    assert web._is_json_content_type("application/json") is True
    assert web._is_json_content_type("APPLICATION/JSON ;charset=UTF-8") is True
    assert web._is_json_content_type("text/plain") is False
    assert web._is_json_content_type("") is False


# ---- _live_now: the Live tab's "who's here now?" payload (visit + cast + recent log) ----
def test_live_now_lists_human_cast_and_recent_sightings(conn, db_path):
    cfg = replace(config.CONFIG, db_path=db_path)
    # A human-confirmed individual is offered as a quick-pick chip; a cluster placeholder is not.
    d = db.insert_detection(conn, timestamp=db.now_local_iso(), source=db.SOURCE_GLASS_DOOR_CAM,
                            detection_class="animal", confidence=0.9, bbox=(0, 0, 10, 10),
                            frame_w=100, frame_h=100, crop_path="crops/x.jpg", species="raccoon")
    db.set_individual_bulk(conn, [d], "Stan", source="human")
    other = db.insert_detection(conn, timestamp=db.now_local_iso(), source=db.SOURCE_GLASS_DOOR_CAM,
                                detection_class="animal", confidence=0.9, bbox=(0, 0, 10, 10),
                                frame_w=100, frame_h=100, crop_path="crops/y.jpg", species="raccoon")
    db.set_individual_bulk(conn, [other], "raccoon_c01", source="cluster")
    db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Notch", "Elliot"])
    out = web._live_now(cfg)
    assert out["cast"] == ["Stan"]                       # only the human-named individual
    assert out["recent"] and out["recent"][0]["names"] == ["Notch", "Elliot"]
    assert "visit" in out and out["source"] == cfg.source


# =====================================================================================
# The re-ID review queue: modes, pagination, freshness, the funnel, and the two places
# the panel has to tell the truth (a camera it cannot match from, and a human's "not them").
#
# Embeddings are hand-built 3-D unit vectors, as in test_individuals.py, so who-matches-whom is
# exact by construction and no model is loaded. Timestamps are placed relative to NOW, because
# template staleness is measured against the clock.
# =====================================================================================

NOW = datetime.now().astimezone()


def _at(days_ago: float, minutes: float = 0.0) -> str:
    return (NOW - timedelta(days=days_ago) + timedelta(minutes=minutes)).isoformat()


def _unit(*xs) -> np.ndarray:
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def _rq_cfg(db_path, **kw):
    """A config pointed at the throwaway DB, built from the SHIPPED defaults -- config.Config(),
    not config.CONFIG. This repo is public: a stranger's checkout has no config_local.py and its
    auto tier is disabled, so that is the case the queue must be correct in. Tuning the operating
    point locally must not be able to break someone else's tests."""
    return replace(config.Config(), db_path=db_path, **kw)


def _visit_with(conn, *, vec, days_ago, source=db.SOURCE_GLASS_DOOR_CAM, n=3, species="raccoon",
                minutes=0.0):
    """One visit of `n` crops, each embedded with the same vector (so the visit prototype IS it).
    Crops are one minute apart, so no two share a timestamp and the co-presence badge stays off."""
    ids = []
    for i in range(n):
        ids.append(db.insert_detection(
            conn, timestamp=_at(days_ago, minutes + i), source=source, detection_class="animal",
            confidence=0.9, bbox=(0, 0, 10, 10), frame_w=100, frame_h=100,
            crop_path=f"crops/{source}-{days_ago}-{minutes}-{i}.jpg", species=species,
            crop_quality=10.0))
    vid = db.insert_visit(
        conn, source=source, species=species, individual_id=None,
        started_at=_at(days_ago, minutes), ended_at=_at(days_ago, minutes + n),
        detection_count=n, max_confidence=0.9, representative_detection_id=ids[0])
    db.assign_visit(conn, ids, vid)
    for d in ids:
        db.insert_embedding(conn, d, individuals.EMBED_MODEL, len(vec),
                            np.asarray(vec, dtype=np.float32).tobytes())
    conn.commit()
    return vid


@pytest.fixture
def corpus(conn):
    """A miniature version of the real situation: two confirmed individuals whose templates are
    different ages, an unreviewed visit that clearly matches the STALE one, an unreviewed visit
    caught between the two, one visit the nightly pass named by itself, and one visit from a
    second camera that no template comes from.

        Stan   -- template 30 days old  (stale)
        Notch  -- template  1 day  old  (fresh)
    """
    ids = {}
    ids["stan_t"] = _visit_with(conn, vec=_unit(1, 0, 0), days_ago=30)
    ids["notch_t"] = _visit_with(conn, vec=_unit(0, 1, 0), days_ago=1)
    db.label_visit(conn, ids["stan_t"], "Stan")
    db.label_visit(conn, ids["notch_t"], "Notch")
    # Clearly Stan (whose template is stale), nobody has reviewed it.
    ids["stale_probe"] = _visit_with(conn, vec=_unit(1, 0.05, 0), days_ago=0.5)
    # Caught between Stan and Notch: a strong match, but the two names are a hair apart.
    ids["ambiguous"] = _visit_with(conn, vec=_unit(1, 1.02, 0), days_ago=0.4)
    # Named by the nightly auto-assign pass; no human has kept or rejected it.
    ids["auto"] = _visit_with(conn, vec=_unit(1, 0, 0), days_ago=0.3)
    db.label_visit(conn, ids["auto"], "Stan", source="auto")
    # A second camera. No confirmed template comes from it.
    ids["trail"] = _visit_with(conn, vec=_unit(1, 0, 0), days_ago=0.2,
                               source=db.SOURCE_TRAIL_CAM_SD)
    return ids


def _queue(cfg, **kw):
    return web._reid_queue(cfg, species="raccoon", **kw)


def _ids(out):
    return [v["visit_id"] for v in out["queue"]]


# ---- modes -------------------------------------------------------------------------
def test_recent_is_the_default_and_is_unchanged(corpus, db_path):
    """The default must keep showing what it always showed: every visit, newest first. The other
    modes are opt-in tabs; a silent change to the landing view is a change to Matt's habits."""
    cfg = _rq_cfg(db_path)
    out = _queue(cfg)
    assert out["mode"] == "recent"
    # Newest first, and nothing filtered out.
    starts = [v["started_at"] for v in out["queue"]]
    assert starts == sorted(starts, reverse=True)
    assert len(out["queue"]) == len(corpus) == out["n_matched"]
    # An unknown mode falls back to 'recent' rather than erroring or showing nothing.
    assert _queue(cfg, mode="nonsense")["mode"] == "recent"


def test_unreviewed_auto_lists_only_machine_names_no_human_has_judged(corpus, conn, db_path):
    cfg = _rq_cfg(db_path)
    assert _ids(_queue(cfg, mode="unreviewed_auto")) == [corpus["auto"]]
    # KEEP it -> it becomes a human confirmation, and leaves the mode.
    db.label_visit(conn, corpus["auto"], "Stan", source="human")
    assert _ids(_queue(cfg, mode="unreviewed_auto")) == []


def test_unreviewed_auto_drops_a_visit_the_human_rejected(corpus, conn, db_path):
    cfg = _rq_cfg(db_path)
    db.label_visit(conn, corpus["auto"], None, reject=True)     # the "✗ not them" tombstone
    assert _ids(_queue(cfg, mode="unreviewed_auto")) == []


def test_ambiguous_is_what_the_auto_tier_correctly_refuses(corpus, db_path):
    """Clears the similarity bar, but the lead over the runner-up individual is inside the margin
    -- the machine cannot call it and says so. Those are the highest-information clicks available.
    With the tier disabled (shipped default) the bars fall back to the novelty cut + the queue's
    own margin, so the mode still works before anyone turns auto-assign on."""
    cfg = _rq_cfg(db_path)
    assert _ids(_queue(cfg, mode="ambiguous")) == [corpus["ambiguous"]]


def test_ambiguous_tracks_the_auto_tier_bars_once_it_is_enabled(corpus, db_path):
    """With the tier live, the mode is exactly its `ambiguous` skip bucket. The near-tie visit
    scores ~0.71, so a 0.76 similarity bar puts it out of the tier's reach -- and out of here."""
    tight = _rq_cfg(db_path, reid_auto_threshold=0.76, reid_auto_margin=0.12)
    assert _ids(_queue(tight, mode="ambiguous")) == []
    loose = _rq_cfg(db_path, reid_auto_threshold=0.5, reid_auto_margin=0.12)
    assert _ids(_queue(loose, mode="ambiguous")) == [corpus["ambiguous"]]


def test_stale_finds_visits_whose_best_candidate_has_an_old_template(corpus, db_path):
    cfg = _rq_cfg(db_path, reid_queue_stale_days=14)
    got = _ids(_queue(cfg, mode="stale"))
    assert corpus["stale_probe"] in got          # its top candidate is Stan: template 30 days old
    assert corpus["ambiguous"] not in got        # its top candidate is Notch: template 1 day old
    assert corpus["notch_t"] not in got          # already confirmed -- nothing to review
    # The cut is a knob, not a constant: raise it past 30 days and nothing is stale any more.
    assert _ids(_queue(_rq_cfg(db_path, reid_queue_stale_days=90), mode="stale")) == []


def test_modes_never_offer_a_visit_the_human_already_settled(corpus, conn, db_path):
    cfg = _rq_cfg(db_path)
    db.label_visit(conn, corpus["stale_probe"], None, reject=True)
    assert corpus["stale_probe"] not in _ids(_queue(cfg, mode="stale"))
    db.label_visit(conn, corpus["ambiguous"], "Stan")
    assert corpus["ambiguous"] not in _ids(_queue(cfg, mode="ambiguous"))


# ---- pagination ---------------------------------------------------------------------
def test_pagination_walks_the_whole_pool_without_repeats(corpus, db_path):
    """Paginate rather than raise the limit: the matcher is rebuilt per call, so the answer to
    'the window only reaches 6 of 208' is a filter plus pages, not a bigger number."""
    cfg = _rq_cfg(db_path)
    total = len(corpus)
    seen = []
    for off in range(0, total, 2):
        page = _queue(cfg, limit=2, offset=off)
        assert page["n_matched"] == total       # the count is of the MODE, not of the page
        assert page["offset"] == off and page["limit"] == 2
        assert len(page["queue"]) <= 2
        seen += _ids(page)
    assert len(seen) == total and len(set(seen)) == total
    assert seen == _ids(_queue(cfg, limit=100))          # same order, just sliced
    # Past the end is empty, not an error.
    assert _queue(cfg, limit=2, offset=999)["queue"] == []


def test_query_params_survive_garbage():
    q = {"limit": ["abc"], "offset": ["-5"], "mode": ["stale"]}
    assert web._qs_int(q, "limit", 30, 1, 100) == 30      # unparseable -> the default
    assert web._qs_int(q, "offset", 0, 0, 10 ** 6) == 0   # clamped, not negative
    assert web._qs_int(q, "missing", 7, 1, 100) == 7
    assert web._qs_int({"limit": ["9999"]}, "limit", 30, 1, 100) == 100


# ---- template freshness --------------------------------------------------------------
def test_cast_carries_days_since_newest_template(corpus, db_path):
    """The decay curve makes this the priority list: top-1 falls 0.818 -> 0.482 -> 0.222 as the
    newest usable template ages 0 -> 7 -> 21 days."""
    cast = {c["name"]: c for c in _queue(_rq_cfg(db_path))["cast"]}
    assert round(cast["Stan"]["days_since_template"]) == 30
    assert round(cast["Notch"]["days_since_template"]) == 1
    assert cast["Stan"]["n_templates"] == 1


def test_a_confirmation_on_a_multi_animal_visit_is_not_a_fresh_template(conn, db_path):
    """n_visits can look healthy while the individual is unrecognisable: a confirmation on a
    2+-animal visit blends two animals and is excluded from templates()."""
    old = _visit_with(conn, vec=_unit(1, 0, 0), days_ago=40)
    db.label_visit(conn, old, "Stan")
    # A recent visit with two separated boxes in the same frames -> multi, so not a template.
    ids = []
    for i in range(4):
        for box in ((0, 0, 10, 10), (50, 50, 60, 60)):
            ids.append(db.insert_detection(
                conn, timestamp=_at(0.1, i), source=db.SOURCE_GLASS_DOOR_CAM,
                detection_class="animal", confidence=0.9, bbox=box, frame_w=100, frame_h=100,
                crop_path=f"crops/pair-{i}-{box[0]}.jpg", species="raccoon", crop_quality=10.0))
    vid = db.insert_visit(conn, source=db.SOURCE_GLASS_DOOR_CAM, species="raccoon",
                          individual_id=None, started_at=_at(0.1), ended_at=_at(0.1, 5),
                          detection_count=len(ids), max_confidence=0.9,
                          representative_detection_id=ids[0])
    db.assign_visit(conn, ids, vid)
    for d in ids:
        db.insert_embedding(conn, d, individuals.EMBED_MODEL, 3,
                            np.asarray(_unit(1, 0, 0), dtype=np.float32).tobytes())
    conn.commit()
    db.label_visit(conn, vid, "Stan")
    cast = {c["name"]: c for c in _queue(_rq_cfg(db_path))["cast"]}
    assert cast["Stan"]["n_visits"] == 2            # two confirmations...
    assert cast["Stan"]["n_templates"] == 1         # ...but only one is usable
    assert round(cast["Stan"]["days_since_template"]) == 40   # and it is the OLD one


# ---- the funnel ----------------------------------------------------------------------
def test_funnel_counts_are_computed_live(corpus, db_path):
    f = _queue(_rq_cfg(db_path))["funnel"]
    assert f["visits"] == len(corpus)
    assert f["with_prototype"] == len(corpus)
    assert f["confirmed"] == 2 and f["templates"] == 2
    assert f["auto_named"] == 1
    # addressable = has a prototype, unconfirmed, not multi, not tombstoned.
    assert f["addressable"] == len(corpus) - 2
    by_source = {s["source"]: s for s in f["by_source"]}
    assert by_source[db.SOURCE_GLASS_DOOR_CAM]["templated"] is True
    assert by_source[db.SOURCE_TRAIL_CAM_SD]["templated"] is False
    assert by_source[db.SOURCE_TRAIL_CAM_SD]["confirmed"] == 0


def test_funnel_counts_the_tombstones(corpus, conn, db_path):
    db.label_visit(conn, corpus["stale_probe"], None, reject=True)
    f = _queue(_rq_cfg(db_path))["funnel"]
    assert f["rejected"] == 1
    assert f["addressable"] == len(corpus) - 3      # the tombstoned one drops out


# ---- telling the truth about a camera it cannot match from ----------------------------
def test_a_visit_from_an_untemplated_camera_is_flagged_cross_source(corpus, db_path):
    """Measured: trail-cam prototypes score a median 0.249 (max 0.363) against every glass-door
    template, and trail-cam-to-trail-cam similarity is flat. A top-1 there is a fact about the
    CAMERA, not the animal, so the card says so instead of 'possibly someone new'."""
    by_id = {v["visit_id"]: v for v in _queue(_rq_cfg(db_path), limit=100)["queue"]}
    assert by_id[corpus["trail"]]["cross_source"] is True
    assert by_id[corpus["trail"]]["source"] == db.SOURCE_TRAIL_CAM_SD
    assert by_id[corpus["stale_probe"]]["cross_source"] is False


def test_cross_source_is_derived_from_the_data_not_from_a_camera_name(conn, db_path):
    """Nothing here is keyed to 'trail_cam_sd'. Confirm one visit on that camera and it stops
    being cross-source; the glass door, with no template of its own, becomes it."""
    trail = _visit_with(conn, vec=_unit(1, 0, 0), days_ago=2, source=db.SOURCE_TRAIL_CAM_SD)
    glass = _visit_with(conn, vec=_unit(1, 0, 0), days_ago=1)
    db.label_visit(conn, trail, "Bandit")
    by_id = {v["visit_id"]: v for v in _queue(_rq_cfg(db_path), limit=100)["queue"]}
    assert by_id[trail]["cross_source"] is False
    assert by_id[glass]["cross_source"] is True


# ---- the reject tombstone (the path that had never once been exercised) ---------------
def test_reject_round_trip_is_visible_in_the_queue_and_undoable(corpus, conn, db_path):
    cfg = _rq_cfg(db_path)
    vid = corpus["stale_probe"]
    assert next(v for v in _queue(cfg, limit=100)["queue"] if v["visit_id"] == vid)["rejected"] \
        is False
    # ✗ not them  (exactly what web.py's /api/reid/confirm does with reject=true)
    db.label_visit(conn, vid, None, reject=True)
    assert vid in db.rejected_visit_ids(conn, "raccoon")
    card = next(v for v in _queue(cfg, limit=100)["queue"] if v["visit_id"] == vid)
    assert card["rejected"] is True and card["confirmed_as"] is None
    # ↺ undo: clear WITHOUT reject wipes the tombstone and puts the visit back in play.
    db.label_visit(conn, vid, None, reject=False)
    assert db.rejected_visit_ids(conn, "raccoon") == set()
    card = next(v for v in _queue(cfg, limit=100)["queue"] if v["visit_id"] == vid)
    assert card["rejected"] is False


def test_keep_promotes_an_auto_name_to_a_real_template(corpus, conn, db_path):
    """The ✓ keep path: an auto name feeds nothing until a human agrees with it."""
    before = _queue(_rq_cfg(db_path))["funnel"]
    db.label_visit(conn, corpus["auto"], "Stan", source="human")
    after = _queue(_rq_cfg(db_path))["funnel"]
    assert after["templates"] == before["templates"] + 1
    assert after["auto_named"] == before["auto_named"] - 1


# ---- temporal context: shown, never scored -------------------------------------------
def test_temporal_context_chip_names_the_adjacent_confirmed_visit(conn, db_path):
    _visit_with(conn, vec=_unit(0, 1, 0), days_ago=20)          # a far-away template for Notch
    neighbour = _visit_with(conn, vec=_unit(1, 0, 0), days_ago=1, minutes=0)
    db.label_visit(conn, neighbour, "Stan")
    probe = _visit_with(conn, vec=_unit(0, 1, 0), days_ago=1, minutes=6)   # 6 min later
    far = _visit_with(conn, vec=_unit(0, 1, 0), days_ago=1, minutes=600)   # 10 h later
    by_id = {v["visit_id"]: v for v in _queue(_rq_cfg(db_path), limit=100)["queue"]}
    ctx = by_id[probe]["context"]
    assert ctx and ctx["name"] == "Stan" and ctx["direction"] == "after"
    assert 300 <= ctx["gap_s"] <= 420                            # ~6 minutes
    assert by_id[far]["context"] is None                         # outside the hour
    assert by_id[neighbour]["context"] is None                   # already named: no call to make


def test_temporal_context_never_changes_the_appearance_ranking(conn, db_path):
    """The two-axis rule, and a measured one: as a ranking input under session blocking, adjacency
    fired three times and was wrong three times. The probe here LOOKS like Notch and SITS beside a
    visit named Stan -- the chip must say Stan and the ranking must still say Notch."""
    _visit_with(conn, vec=_unit(1, 0, 0), days_ago=25)           # Stan's template
    notch_t = _visit_with(conn, vec=_unit(0, 1, 0), days_ago=25, minutes=600)
    db.label_visit(conn, notch_t, "Notch")
    stan_recent = _visit_with(conn, vec=_unit(1, 0, 0), days_ago=1, minutes=0)
    db.label_visit(conn, stan_recent, "Stan")
    probe = _visit_with(conn, vec=_unit(0, 1, 0), days_ago=1, minutes=5)
    card = next(v for v in _queue(_rq_cfg(db_path), limit=100)["queue"]
                if v["visit_id"] == probe)
    assert card["context"]["name"] == "Stan"                     # the chip
    assert card["candidates"][0]["name"] == "Notch"              # the ranking, untouched

    # And the ranking is IDENTICAL to the same corpus with the neighbour moved out of the window:
    # the appearance answer cannot depend on who happened to visit six minutes earlier.
    conn.execute("UPDATE visits SET started_at = ?, ended_at = ? WHERE id = ?",
                 (_at(1, -600), _at(1, -597), stan_recent))
    conn.execute("UPDATE detections SET timestamp = ? WHERE visit_id = ?",
                 (_at(1, -600), stan_recent))
    conn.commit()
    moved = next(v for v in _queue(_rq_cfg(db_path), limit=100)["queue"]
                 if v["visit_id"] == probe)
    assert moved["context"] is None
    assert [(c["name"], c["similarity"]) for c in moved["candidates"]] \
        == [(c["name"], c["similarity"]) for c in card["candidates"]]


# ---- the endpoints, over a real socket -----------------------------------------------
def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def test_reid_queue_endpoint_returns_200_and_valid_json(corpus, db_path):
    cfg = _rq_cfg(db_path, web_host="127.0.0.1", web_port=0)
    buffers = {cfg.source: web.FrameBuffer()}
    server = web.make_server(cfg, buffers, {cfg.source: web.CameraControlBridge()})
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        for path in ("/api/reid/queue",
                     "/api/reid/queue?mode=ambiguous",
                     "/api/reid/queue?mode=unreviewed_auto&limit=5&offset=0",
                     "/api/reid/queue?mode=stale",
                     "/api/reid/queue?mode=bogus&limit=abc&offset=-3"):
            status, body = _get(port, path)
            assert status == 200, path
            assert set(body) >= {"queue", "cast", "mode", "offset", "limit", "n_matched", "funnel"}
            assert body["mode"] in web.QUEUE_MODES
            assert isinstance(body["queue"], list)
        assert _get(port, "/api/reid/queue?mode=bogus&limit=abc&offset=-3")[1]["mode"] == "recent"
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=5)


# ---- the roster: marking an individual as no longer resident -------------------------
# Templates outlive the animal. Notch's last labelled crop is 2026-06-30 and Matt confirms it
# stopped coming, but at the recommended operating point the auto tier still lined up to write
# that name onto two 2026-07-03 visits. Nothing the machine can measure sees that (an evaluation
# scores against labels, and a departed animal's labels just stop), so the cast surface has to let
# the human say it -- and then show it, so the panel stops nagging for a template nobody can make.
def test_cast_reports_residency_and_defaults_to_resident(corpus, conn, db_path):
    cast = {c["name"]: c for c in _queue(_rq_cfg(db_path))["cast"]}
    assert cast["Notch"]["status"] == "resident" and cast["Notch"]["departed_on"] is None

    db.set_individual_status(conn, "Notch", status="departed", effective_date="2026-06-30",
                             note="stopped coming")
    cast = {c["name"]: c for c in _queue(_rq_cfg(db_path))["cast"]}
    assert cast["Notch"]["status"] == "departed"
    assert cast["Notch"]["departed_on"] == "2026-06-30"
    assert cast["Notch"]["status_note"] == "stopped coming"
    assert cast["Stan"]["status"] == "resident"           # one name, not the whole cast
    # Nothing else about the individual changes: still confirmed, still a template, still ranked.
    assert cast["Notch"]["n_templates"] == 1
    assert any(v["candidates"] and v["candidates"][0]["name"] in ("Stan", "Notch")
               for v in _queue(_rq_cfg(db_path), limit=100)["queue"])


def test_roster_status_matches_the_cast_name_case_insensitively(corpus, conn, db_path):
    db.set_individual_status(conn, "notch", status="departed", effective_date="2026-06-30")
    cast = {c["name"]: c for c in _queue(_rq_cfg(db_path))["cast"]}
    assert cast["Notch"]["status"] == "departed"


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


def test_individual_status_endpoint_round_trip(corpus, db_path):
    cfg = _rq_cfg(db_path, web_host="127.0.0.1", web_port=0)
    buffers = {cfg.source: web.FrameBuffer()}
    server = web.make_server(cfg, buffers, {cfg.source: web.CameraControlBridge()})
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        status, body = _post(port, "/api/individual/status",
                             {"name": "Notch", "status": "departed",
                              "effective_date": "2026-06-30"})
        assert status == 200 and body["ok"] and body["effective_date"] == "2026-06-30"
        cast = {c["name"]: c for c in _get(port, "/api/reid/queue")[1]["cast"]}
        assert cast["Notch"]["status"] == "departed"

        # A typo is a 400, not a guard that silently never fires.
        bad, err = _post(port, "/api/individual/status",
                         {"name": "Notch", "effective_date": "30-06-2026"})
        assert bad == 400 and "error" in err
        empty, err2 = _post(port, "/api/individual/status", {"name": "  "})
        assert empty == 400 and "error" in err2
        # ...and the earlier, good value is untouched by the rejected write.
        cast = {c["name"]: c for c in _get(port, "/api/reid/queue")[1]["cast"]}
        assert cast["Notch"]["departed_on"] == "2026-06-30"

        # Undo puts the name back on the roster.
        assert _post(port, "/api/individual/status",
                     {"name": "Notch", "status": "resident"})[0] == 200
        cast = {c["name"]: c for c in _get(port, "/api/reid/queue")[1]["cast"]}
        assert cast["Notch"]["status"] == "resident"
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=5)


# ---- ignore zones: the DB semantics the dashboard editor rests on --------------------
def test_zone_add_normalizes_corners_and_lists(conn):
    # A browser drag can end anywhere: floats, swapped corners, a start past the end.
    row = db.add_ignore_zone(conn, "cam", 200.6, 300.2, 100.0, 250.0, note="  wall gap  ")
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == (100, 250, 201, 300)
    assert row["note"] == "wall gap"
    zones = db.list_ignore_zones(conn, "cam")
    assert [z["id"] for z in zones] == [row["id"]]
    assert db.list_ignore_zones(conn, "other_cam") == []


def test_zone_add_rejects_a_slip_of_the_pointer(conn):
    with pytest.raises(ValueError):
        db.add_ignore_zone(conn, "cam", 100, 100, 103, 200)   # 3 px wide: a click, not a zone
    with pytest.raises(ValueError):
        db.add_ignore_zone(conn, "", 0, 0, 50, 50)            # no source


def test_zone_delete_tombstone_outlives_the_config_seed(conn):
    """The restart story: a zone deleted in the dashboard must STAY deleted even though
    config_local.py still lists it -- the soft-delete tombstone is what blocks the re-seed."""
    seeded = db.seed_ignore_zones(conn, {"cam": [(10, 10, 60, 60)]})
    assert seeded == 1
    zid = db.list_ignore_zones(conn, "cam")[0]["id"]
    assert db.remove_ignore_zone(conn, zid)["id"] == zid
    assert db.list_ignore_zones(conn, "cam") == []
    # The same config runs again at next startup -- and must not resurrect the rectangle...
    assert db.seed_ignore_zones(conn, {"cam": [(10, 10, 60, 60)]}) == 0
    assert db.list_ignore_zones(conn, "cam") == []
    # ...while a RE-MEASURED rectangle (the camera moved) is genuinely new and lands.
    assert db.seed_ignore_zones(conn, {"cam": [(20, 20, 70, 70)]}) == 1
    # Removing an unknown / already-dead id reports None rather than inventing work.
    assert db.remove_ignore_zone(conn, zid) is None
    assert db.remove_ignore_zone(conn, 99999) is None


def test_zone_seed_skips_malformed_rects_rather_than_crashing_the_rig(conn):
    n = db.seed_ignore_zones(conn, {"cam": [(1, 2, 3), "junk", None, (5, 5, 55, 55)]})
    assert n == 1
    assert [(z["x1"], z["y1"], z["x2"], z["y2"]) for z in db.list_ignore_zones(conn, "cam")] \
        == [(5, 5, 55, 55)]


# ---- IgnoreZoneStore: what the capture threads actually read -------------------------
def test_zone_store_seeds_from_config_and_serves_the_hot_path(db_path):
    cfg = _rq_cfg(db_path)
    cfg = replace(cfg, ignore_zones={"glass_door_cam": [(1127, 595, 1234, 701)]})
    store = web.IgnoreZoneStore.load(cfg)
    assert store.rects("glass_door_cam") == ((1127, 595, 1234, 701),)
    assert store.rects("some_other_cam") == ()
    assert store.counts() == {"glass_door_cam": 1}

    row = store.add("glass_door_cam", 10, 10, 90, 90, note="grill")
    assert store.rects("glass_door_cam") == ((1127, 595, 1234, 701), (10, 10, 90, 90))
    assert store.remove(row["id"]) is True
    assert store.remove(row["id"]) is False               # already gone
    assert store.rects("glass_door_cam") == ((1127, 595, 1234, 701),)

    # Durability: a fresh store (= next rig start) reloads the same state from the table.
    again = web.IgnoreZoneStore.load(cfg)
    assert again.rects("glass_door_cam") == ((1127, 595, 1234, 701),)


# ---- the zone endpoints, over a real socket ------------------------------------------
def test_zone_endpoints_round_trip_and_reach_the_shared_store(db_path):
    cfg = _rq_cfg(db_path, web_host="127.0.0.1", web_port=0)
    store = web.IgnoreZoneStore.load(cfg)
    buffers = {cfg.source: web.FrameBuffer()}
    bridge = web.CameraControlBridge()
    bridge.publish({"frame_w": 1920, "frame_h": 1080})    # what the capture thread publishes
    server = web.make_server(cfg, buffers, {cfg.source: bridge}, store)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        status, body = _get(port, "/api/zones")
        assert status == 200
        assert body["zones"] == [] and body["source"] == cfg.source
        assert body["frame"] == {"w": 1920, "h": 1080}

        # Add: a drag that ran off the right edge clamps to the frame instead of 400ing.
        status, body = _post(port, "/api/zones",
                             {"x1": 1800, "y1": 500, "x2": 2400, "y2": 700, "note": "wall gap"})
        assert status == 200 and body["ok"]
        z = body["zone"]
        assert (z["x1"], z["y1"], z["x2"], z["y2"]) == (1800, 500, 1920, 700)
        # The endpoint's whole point: the capture threads' shared store sees it immediately.
        assert store.rects(cfg.source) == ((1800, 500, 1920, 700),)

        status, body = _get(port, "/api/zones")
        assert [w["id"] for w in body["zones"]] == [z["id"]]
        assert body["zones"][0]["note"] == "wall gap"
        assert body["zones"][0]["stale"] is False          # camera never seen to move

        # A zone drawn BEFORE the camera last moved gets flagged.
        conn2 = db.connect(db_path)
        try:
            db.bump_view_epoch(conn2, cfg.source, detected_by="manual")
        finally:
            conn2.close()
        assert _get(port, "/api/zones")[1]["zones"][0]["stale"] is True

        # Garbage coordinates are a 400, not a 500 (and not a row).
        assert _post(port, "/api/zones", {"x1": "a", "y1": 0, "x2": 50, "y2": 50})[0] == 400
        assert _post(port, "/api/zones", {"x1": 0, "y1": 0, "x2": 2, "y2": 2})[0] == 400
        assert len(_get(port, "/api/zones")[1]["zones"]) == 1

        # Delete round trip; a second delete of the same id is a 404, not a silent ok.
        assert _post(port, "/api/zones/delete", {"id": z["id"]})[0] == 200
        assert store.rects(cfg.source) == ()
        assert _get(port, "/api/zones")[1]["zones"] == []
        assert _post(port, "/api/zones/delete", {"id": z["id"]})[0] == 404
        assert _post(port, "/api/zones/delete", {"id": "junk"})[0] == 400
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=5)
