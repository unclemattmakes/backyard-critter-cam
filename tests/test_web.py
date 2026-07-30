"""
Unit tests for web.py's pure helpers -- HTTP Range parsing (used to stream/seek video clips),
the camera-control whitelist, the media-path containment check (path-traversal guard), and the
cross-site guard that decides which POSTs the dashboard will act on. These
never start a server or open a socket; they exercise the parsing/validation logic the dashboard
relies on. (web.py imports only stdlib + db/stats/behavior/config, so importing it is cheap.)
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import config
import db
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
