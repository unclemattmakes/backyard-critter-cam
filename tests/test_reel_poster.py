"""
Tests for the reel POSTER frame (reel.py).

Background: the Dispatch hero and the clip player used to back themselves with a per-moment
`thumb`, which is an animal CROP. Measured 2026-07-26 on the night reel, the crops in play were
389x292 and 96x103 — stretched to a ~1500 px hero that's a 4x and 16x upscale, which is the
"stack of blurry ovals" the reel appeared to be. The fix lifts a real 1280x720 frame out of the
stitched reel and puts its path in the manifest.

These cover the selection rule and the manifest wiring. `_make_poster` shells out to ffmpeg, so
that call is stubbed — what's tested is that the biggest candidate JPEG wins, that failures stay
soft (callers must fall back to the old backdrop), and that the manifest only advertises a
poster that actually exists.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import reel


@pytest.fixture
def reel_mp4(tmp_path):
    """A stand-in for a stitched reel on disk."""
    d = tmp_path / "clips" / "reels"
    d.mkdir(parents=True)
    p = d / "reel_2026-07-26_night_abc123.mp4"
    p.write_bytes(b"not really an mp4")
    return p


def fake_ffmpeg(sizes):
    """Stub _run_ffmpeg: write candidate JPEGs of the given byte sizes into the output pattern's
    directory, the way `-vf fps=... c_%03d.jpg` would."""
    def _run(args, timeout):
        out = Path(args[-1])
        for i, n in enumerate(sizes, start=1):
            (out.parent / f"c_{i:03d}.jpg").write_bytes(b"\xff" * n)
    return _run


# ---- choosing the frame -------------------------------------------------------------

def test_the_largest_candidate_wins(reel_mp4, tmp_path, monkeypatch):
    """Biggest JPEG at fixed quality = most detail = the sharpest, best-lit frame."""
    monkeypatch.setattr(reel, "_run_ffmpeg", fake_ffmpeg([100, 900, 400]))
    poster = reel._make_poster(reel_mp4, tmp_path / "work")
    assert poster == reel_mp4.with_suffix(".jpg")
    assert poster.read_bytes() == b"\xff" * 900


def test_poster_sits_beside_the_reel_with_a_jpg_suffix(reel_mp4, tmp_path, monkeypatch):
    """The pruner finds it via with_suffix('.jpg'), so the name must line up exactly."""
    monkeypatch.setattr(reel, "_run_ffmpeg", fake_ffmpeg([10]))
    poster = reel._make_poster(reel_mp4, tmp_path / "work")
    assert poster.parent == reel_mp4.parent
    assert poster.stem == reel_mp4.stem and poster.suffix == ".jpg"


# ---- failing softly -----------------------------------------------------------------

def test_ffmpeg_failure_returns_none(reel_mp4, tmp_path, monkeypatch):
    def boom(args, timeout):
        raise subprocess.CalledProcessError(1, "ffmpeg")
    monkeypatch.setattr(reel, "_run_ffmpeg", boom)
    assert reel._make_poster(reel_mp4, tmp_path / "work") is None


def test_no_candidates_returns_none(reel_mp4, tmp_path, monkeypatch):
    """ffmpeg 'succeeded' but wrote nothing -- must not claim a poster."""
    monkeypatch.setattr(reel, "_run_ffmpeg", lambda args, timeout: None)
    assert reel._make_poster(reel_mp4, tmp_path / "work") is None


# ---- manifest wiring ----------------------------------------------------------------

class FakeCfg:
    def __init__(self, root):
        self.clips_dir = root / "clips"


def a_plan():
    return {"edition": "night", "anchor": "2026-07-26", "title": "Last Night",
            "start": "2026-07-25T21:00:00-07:00", "end": "2026-07-26T05:00:00-07:00",
            "n_source_clips": 222,
            "segments": [{"seconds": 6.0, "actual_s": 6.04, "start": "2026-07-25T21:29:03-07:00",
                          "species": "raccoon", "individuals": [], "thumb": "crops/x.jpg",
                          "n_dets": 2}]}


def test_manifest_carries_a_media_relative_poster_path(reel_mp4, tmp_path):
    cfg = FakeCfg(tmp_path)
    poster = reel_mp4.with_suffix(".jpg")
    poster.write_bytes(b"jpeg")
    man = reel._manifest_from(cfg, a_plan(), reel_mp4, poster)
    assert man["poster_path"] == "clips/reels/reel_2026-07-26_night_abc123.jpg"
    # /media/ resolves against clips_dir.parent, same as clip_path -- keep them consistent.
    assert man["clip_path"] == "clips/reels/reel_2026-07-26_night_abc123.mp4"


def test_manifest_omits_poster_when_there_is_none(reel_mp4, tmp_path):
    """Old cached reels have no poster; the dashboard's ||-fallback needs the key ABSENT."""
    man = reel._manifest_from(FakeCfg(tmp_path), a_plan(), reel_mp4, None)
    assert "poster_path" not in man


def test_manifest_omits_poster_that_vanished(reel_mp4, tmp_path):
    """Never advertise a path that would 404 in the hero's background-image."""
    man = reel._manifest_from(FakeCfg(tmp_path), a_plan(), reel_mp4,
                              reel_mp4.with_suffix(".jpg"))    # never written
    assert "poster_path" not in man


def test_manifest_is_json_serialisable(reel_mp4, tmp_path):
    poster = reel_mp4.with_suffix(".jpg")
    poster.write_bytes(b"jpeg")
    man = reel._manifest_from(FakeCfg(tmp_path), a_plan(), reel_mp4, poster)
    assert json.loads(json.dumps(man))["poster_path"].endswith(".jpg")
