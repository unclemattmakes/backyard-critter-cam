"""
Unit tests for web.py's pure helpers -- HTTP Range parsing (used to stream/seek video clips),
the camera-control whitelist, and the media-path containment check (path-traversal guard). These
never start a server or open a socket; they exercise the parsing/validation logic the dashboard
relies on. (web.py imports only stdlib + db/stats/behavior/config, so importing it is cheap.)
"""
from __future__ import annotations

from pathlib import Path

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
