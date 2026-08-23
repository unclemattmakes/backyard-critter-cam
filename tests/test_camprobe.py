"""
Tests for tools/camprobe.py -- the two pure decisions inside a tool whose whole job is
otherwise talking to a camera that isn't there during a test run.

Both of these caused real trouble on 2026-08-21 while adding the first network camera:

  * parse_rtsp_url. A password containing '@' splits wrong on the FIRST '@', producing a
    truncated password and a garbage hostname -- and that failure is indistinguishable from a
    wrong password, which is the single most expensive thing it could be confused with. It cost
    two rounds of isolating "is it the shell, the encoding, or the credential" before the
    answer turned out to be the credential after all.

  * suggested_motion_area. motion_min_area is FULL-FRAME PIXELS (the gate downscales but
    converts blob areas back), so it is resolution-dependent, and a value copied from a 1080p
    camera to a 640x360 one is 9x too strict -- a camera that simply never triggers, with
    nothing in any log to say why.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import camprobe  # noqa: E402


class _Cfg:
    """The three fields suggested_motion_area reads, without depending on Config's defaults."""

    def __init__(self, motion_min_area, frame_width, frame_height):
        self.motion_min_area = motion_min_area
        self.frame_width = frame_width
        self.frame_height = frame_height


REF = _Cfg(1800, 1920, 1080)          # this rig's glass-door camera, the reference


# ---- parse_rtsp_url ------------------------------------------------------------------

def test_full_url_splits_into_its_five_parts():
    assert camprobe.parse_rtsp_url(
        "rtsp://rig:swordfish@192.168.1.105:554/h264Preview_01_sub"
    ) == ("rig", "swordfish", "192.168.1.105", 554, "h264Preview_01_sub")


def test_password_containing_an_at_sign_splits_on_the_LAST_one():
    """The one that matters: first-'@' splitting yields ('rig', 'p', 'ss@192.168.1.105', ...) --
    a wrong password AND a wrong host, reported as a login failure."""
    user, pw, host, port, path = camprobe.parse_rtsp_url(
        "rtsp://rig:p@ssw0rd@192.168.1.105:554/stream")
    assert (user, pw, host) == ("rig", "p@ssw0rd", "192.168.1.105")
    assert (port, path) == (554, "stream")


def test_url_without_credentials_parses_with_empty_ones():
    assert camprobe.parse_rtsp_url("rtsp://192.168.1.105:554/stream") == (
        "", "", "192.168.1.105", 554, "stream")


def test_missing_port_defaults_to_554():
    assert camprobe.parse_rtsp_url("rtsp://rig:pw@cam.local/stream1")[3] == 554


def test_path_may_carry_a_query_string():
    """Dahua and Amcrest put the channel in the query, so the path is not always a bare word."""
    assert camprobe.parse_rtsp_url(
        "rtsp://rig:pw@10.0.0.5:554/cam/realmonitor?channel=1&subtype=1"
    )[4] == "cam/realmonitor?channel=1&subtype=1"


@pytest.mark.parametrize("bad", [
    "http://192.168.1.105/stream",      # not RTSP at all
    "rtsp://",                          # no host
    "rtsp://rig:pw@:554/stream",        # empty host
    "rtsp://cam.local:nope/stream",     # unparseable port
    "just a string",
])
def test_things_that_are_not_a_usable_rtsp_url_return_none(bad):
    assert camprobe.parse_rtsp_url(bad) is None


# ---- suggested_motion_area -----------------------------------------------------------

def test_same_resolution_as_the_reference_returns_the_reference_value():
    assert camprobe.suggested_motion_area(REF, 1920, 1080) == 1800


def test_a_quarter_of_the_pixels_wants_a_quarter_of_the_area():
    """960x540 is a quarter of 1920x1080's pixel count, so the same FRACTION of frame is a
    quarter of the pixels."""
    assert camprobe.suggested_motion_area(REF, 960, 540) == 450


def test_the_real_sub_stream_case():
    """640x360 against the glass door's 1800 at 1080p -- the number that shipped on 2026-08-21."""
    assert camprobe.suggested_motion_area(REF, 640, 360) == 200


def test_a_4k_camera_wants_a_proportionally_larger_area():
    assert camprobe.suggested_motion_area(REF, 3840, 2160) == 7200


def test_the_trigger_fraction_is_what_is_actually_held_constant():
    ref_fraction = REF.motion_min_area / (REF.frame_width * REF.frame_height)
    for w, h in [(640, 360), (1280, 720), (2560, 1440), (896, 512)]:
        got = camprobe.suggested_motion_area(REF, w, h)
        assert got / (w * h) == pytest.approx(ref_fraction, rel=0.01)


def test_a_tiny_frame_never_scales_down_to_a_zero_trigger():
    """0 would mean 'any motion at all', i.e. the gate stops being a gate."""
    assert camprobe.suggested_motion_area(REF, 32, 24) >= 1


def test_missing_reference_fields_fall_back_instead_of_dividing_by_zero():
    assert camprobe.suggested_motion_area(_Cfg(None, None, None), 640, 360) > 0
