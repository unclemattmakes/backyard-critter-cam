"""
Tests for the morning email (newsletter.py).

Scope, matching the suite's spine: pure logic only. The digest payloads are canned dicts shaped
exactly like stats.period_digest's output (the renderer is a THIN layer over that payload, so
the contract under test is "given this digest, say this"); Resend is never contacted (urlopen is
monkeypatched); ffmpeg is stubbed. The only real I/O is PIL writing tiny JPEGs into tmp_path,
because the image budget rules (downscale, square-crop, the byte cap) are worth testing against
real encoders rather than mocks.
"""
from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import newsletter


# ---- canned payloads -----------------------------------------------------------------

def mkdigest(**over):
    """A period_digest-shaped night with one raccoon visit; override any field per test."""
    d = {
        "edition": "night", "title": "Last Night", "empty": False,
        "start": "2026-08-11T20:45:00-07:00", "end": "2026-08-12T05:55:00-07:00",
        "anchor": "2026-08-12", "latest": True,
        "moon": {"phase": 24.1, "name": "Waning Crescent", "glyph": "☾", "illum_pct": 18},
        "visits": 3, "crops": 41, "n_surprising": 0,
        "coverage": None, "novel": [], "quiet": [], "reel": [],
        "visit_log": [{
            "start": "2026-08-11T22:10:00-07:00", "end": "2026-08-11T22:24:00-07:00",
            "minutes": 14.0, "count": 21, "source": "glass_door_cam",
            "species": ["raccoon"], "individuals": ["stan + kits"],
            "rep_crop": "crops/2026-08-11/a.jpg", "clips": [{"clip_path": "clips/x.mp4"}],
            "motion": {"tag": "fed here"},
        }],
        "species": [{
            "species": "raccoon", "visits": 3, "crops": 41,
            "rep_crop": "crops/2026-08-11/a.jpg", "rep_conf": 0.97,
            "first": "2026-08-11T22:10:00-07:00", "last": "2026-08-12T03:02:00-07:00",
            "novelty": {"first_ever": False, "days_since": 1},
            "streak": 5, "typical": "9pm–3am", "hours": [0] * 24, "active_hours": [22, 3],
            "clip": None,
        }],
        "plate": {"crop_path": "crops/2026-08-11/a.jpg", "species": "raccoon",
                  "conf": 0.97, "time": "2026-08-11T22:12:00-07:00",
                  "clip": {"clip_path": "clips/x.mp4", "start": "2026-08-11T22:09:30-07:00",
                           "seconds": 200.0, "dets": 21, "conf": 0.97}},
        "first_visitor": {"species": "raccoon", "time": "2026-08-11T22:10:00-07:00"},
        "last_visitor": {"species": "raccoon", "time": "2026-08-12T03:02:00-07:00"},
        "busiest_hour": {"hour": 22, "visits": 2},
        "crowd": {"n": 0, "at": None, "source": None, "by_species": {}},
    }
    d.update(over)
    return d


def mkbundle(d=None, rc=None):
    return {"d": d or mkdigest(), "rc": rc or {"cast": []}, "issue_no": 34,
            "base_url": "http://rig:8000", "generated": "2026-08-12T06:30:00-07:00"}


def text_of(html):
    """The visible text of a rendered issue. Assertions about WORDING should not break when the
    markup around a name changes -- which is exactly what happened when every name in the cast
    became a deep link."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html))


def jpeg_bytes(w=200, h=150, color=(120, 90, 60)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "JPEG")
    return buf.getvalue()


def mkcfg(tmp_path, **over):
    root = tmp_path
    (root / "clips").mkdir(exist_ok=True)
    cfg = SimpleNamespace(db_path=root / "backyard.db", clips_dir=root / "clips",
                          web_port=8000, email_to="me@example.com",
                          email_from="Dispatch <d@example.com>",
                          email_resend_api_key="re_test", email_dashboard_url=None,
                          email_send_quiet=True)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


# ---- the words -----------------------------------------------------------------------

def test_subject_leads_with_novelty():
    d = mkdigest(novel=["virginia opossum"],
                 species=mkdigest()["species"] + [{
                     "species": "virginia opossum", "visits": 1, "crops": 2,
                     "rep_crop": None, "rep_conf": 0.8,
                     "first": "2026-08-12T01:00:00-07:00", "last": "2026-08-12T01:04:00-07:00",
                     "novelty": {"first_ever": False, "days_since": 12},
                     "streak": 1, "typical": None, "hours": [0] * 24, "active_hours": [1],
                     "clip": None}])
    s = newsletter.compose_subject(mkbundle(d))
    assert "First Virginia Opossum In 12 Days".lower() in s.lower()
    assert "3 visits" in s


def test_subject_crowd_beats_names():
    d = mkdigest(crowd={"n": 4, "at": "2026-08-12T02:07:11-07:00", "source": "glass_door_cam",
                        "by_species": {"raccoon": 4}})
    assert "at least 4" in newsletter.compose_subject(mkbundle(d))


def test_subject_falls_back_to_named_individuals():
    s = newsletter.compose_subject(mkbundle())
    assert "Stan + Kits" in s and "came by" in s


def test_subject_quiet_night_names_the_dark_camera():
    d = mkdigest(empty=True, visits=0, species=[], visit_log=[], plate=None,
                 coverage={"source": "glass_door_cam", "dark_minutes": 190, "frac_dark": 0.35})
    s = newsletter.compose_subject(mkbundle(d))
    assert "quiet night" in s and "camera" in s


def test_lede_counts_names_and_caveats():
    d = mkdigest(coverage={"source": "glass_door_cam", "dark_minutes": 45, "frac_dark": 0.08})
    lede = " ".join(newsletter.compose_lede(mkbundle(d)))
    assert "3 visits" in lede and "1 species" in lede
    assert "Stan + Kits" in lede
    assert "camera" in lede and "45 min" in lede          # the dark stretch is never hidden
    # The floor wording must survive any rewrite of the crowd sentence.
    d2 = mkdigest(crowd={"n": 3, "at": "2026-08-12T02:07:00-07:00", "source": "g",
                         "by_species": {"raccoon": 3}})
    assert "floor" in " ".join(newsletter.compose_lede(mkbundle(d2)))


def test_clock_normalizes_both_time_frames():
    """Period boundaries arrive in the yard's SOLAR zone (-08:00, no DST) while detection rows
    carry the wall clock (-07:00 in a PDT summer). Printed side by side unconverted, the first
    live issue read "between 8:01 PM and 4:26 AM" over a visit log ending 5:12 AM. One frame,
    always: the target zone (machine-local on the rig; explicit here so the test is portable)."""
    wall = timezone(timedelta(hours=-7))
    assert newsletter._clock("2026-08-12T04:26:00-08:00", tz=wall) == "5:26 AM"
    assert newsletter._clock("2026-08-12T05:12:00-07:00", tz=wall) == "5:12 AM"
    assert newsletter._clock(None) == ""


def test_species_line_caps_the_melee_spray():
    """A kit-melee family visit carries 15+ forced labels; three plus a count reads true."""
    many = ["raccoon", "virginia opossum", "brown rat", "bushtit", "varied thrush", "house sparrow"]
    assert newsletter._species_line(many) == "Raccoon + Virginia Opossum + Brown Rat (+3 more)"
    assert newsletter._species_line(["raccoon"]) == "Raccoon"
    assert newsletter._species_line([]) == "Unidentified"


# ---- the layout ----------------------------------------------------------------------

def test_render_email_escapes_hostile_labels():
    """Species and individual names are user/model text; a name must never become markup."""
    d = mkdigest()
    d["visit_log"][0]["individuals"] = ['<script>alert(1)</script>']
    d["species"][0]["species"] = 'raccoon"><img src=x>'
    html = newsletter.render_email(mkbundle(d), {}, lambda cid: None)
    assert "<script>" not in html and "<img src=x>" not in html


def test_render_email_cid_vs_data_modes():
    images = {"hero": {"bytes": jpeg_bytes(), "mime": "image/jpeg", "hero": True}}
    b = mkbundle()
    mail = newsletter.render_email(b, images, newsletter._img_src_cid(images))
    assert 'src="cid:hero"' in mail and "data:image" not in mail
    web = newsletter.render_email(b, images, newsletter._img_src_data(images))
    assert "data:image/jpeg;base64," in web and "cid:" not in web
    # An image that failed to load renders NO tag at all -- never a broken cid reference.
    none = newsletter.render_email(b, {}, newsletter._img_src_cid({}))
    assert "cid:" not in none


def test_render_email_carries_the_masthead_and_hedges():
    d = mkdigest(crowd={"n": 2, "at": "2026-08-12T02:07:00-07:00", "source": "g",
                        "by_species": {"raccoon": 2}},
                 n_surprising=1,
                 species=mkdigest()["species"] + [{
                     "species": "anna's hummingbird", "visits": 1, "crops": 1,
                     "rep_crop": None, "rep_conf": 0.5,
                     "first": "2026-08-12T02:00:00-07:00", "last": "2026-08-12T02:01:00-07:00",
                     "novelty": {"first_ever": False, "days_since": 2}, "streak": 1,
                     "typical": "6am–10am", "hours": [0] * 24, "active_hours": [2],
                     "clip": None, "surprising": True, "surprise_note": "off-hours"}])
    html = newsletter.render_email(mkbundle(d), {}, lambda cid: None)
    assert "The Morning Dispatch" in html
    assert "No. 34" in html
    assert "a floor" in html                       # crowd tally never reads as a census
    assert "verify" in html                        # surprising species listed as a question
    assert "floors, not censuses" in html          # the footer disclaimer
    assert "#dispatch/2026-08-12/night" in html    # deep link to this very issue


def test_render_text_is_a_faithful_summary():
    txt = newsletter.render_text(mkbundle())
    assert "THE MORNING DISPATCH" in txt
    assert "Raccoon" in txt and "Stan + Kits" in txt
    assert "http://rig:8000/#dispatch/2026-08-12/night" in txt


def test_cast_rollcall_dedupes_groups_and_judges_by_period():
    """Two live-issue lessons: a group stamp ("Pedro + Kits") must not appear BESIDE its base
    raccoon (cast_rollcall already folds its recency into the solo via via_group), and "came by"
    is about the period -- an animal seen at 11:57 PM is days_since=1 by the morning send, and
    calendar arithmetic must not call a present animal absent."""
    d = mkdigest(visit_log=[], species=[], plate=None)     # isolate the cast section
    rc = {"cast": [
        {"id": "pedro", "species": "raccoon", "days_since": 1, "overdue": False,
         "last_seen": "2026-08-11T23:57:00-07:00", "via_group": "pedro + kits"},
        {"id": "pedro + kits", "species": "raccoon", "days_since": 1, "overdue": False,
         "last_seen": "2026-08-11T23:57:00-07:00"},
        {"id": "notch", "species": "raccoon", "days_since": 6, "overdue": False,
         "last_seen": "2026-08-06T02:00:00-07:00"},
        # Spelling drift: a group whose base solo does NOT exist ("cutie" vs a "cutiepie" cast).
        # Absent, it must vanish (a group's absence is never its own fact); present, it may
        # stand in for its unnamed base.
        {"id": "cutie + kits", "species": "raccoon", "days_since": 4, "overdue": False,
         "last_seen": "2026-08-08T01:00:00-07:00"},
        {"id": "ziggy + kits", "species": "raccoon", "days_since": 1, "overdue": False,
         "last_seen": "2026-08-11T23:00:00-07:00"},
    ]}
    text = text_of(newsletter.render_email(mkbundle(d, rc), {}, lambda cid: None))
    assert "Came by: Pedro (with the kits)" in text
    assert "Pedro + Kits" not in text                       # the group never gets its own line
    assert "Notch (6d)" in text
    assert "Cutie + Kits" not in text                       # absent group: not a fact
    assert "Ziggy + Kits" in text                           # present group with no base: shown


def test_quiet_issue_still_renders_moon_and_rollcall():
    d = mkdigest(empty=True, visits=0, species=[], visit_log=[], plate=None, novel=[], quiet=[])
    rc = {"cast": [{"id": "stan", "species": "raccoon", "days_since": 3, "overdue": False,
                    "last_seen": "2026-08-09T02:00:00-07:00"}]}
    text = text_of(newsletter.render_email(mkbundle(d, rc), {}, lambda cid: None))
    assert "quiet night" in text and "Waning Crescent" in text and "Stan (3d)" in text


# ---- images --------------------------------------------------------------------------

def test_gather_images_caps_and_square_thumbs(tmp_path, monkeypatch):
    from PIL import Image
    cfg = mkcfg(tmp_path)
    cropdir = tmp_path / "crops" / "2026-08-11"
    cropdir.mkdir(parents=True)
    (cropdir / "a.jpg").write_bytes(jpeg_bytes(300, 180))
    d = mkdigest(plate=None)   # no plate: this test is about the thumbnails
    d["visit_log"] = d["visit_log"] * (newsletter.MAX_VISIT_THUMBS + 5)
    images = newsletter.gather_images(cfg, d)
    vthumbs = [k for k in images if k.startswith("v")]
    assert len(vthumbs) == newsletter.MAX_VISIT_THUMBS
    w, h = Image.open(io.BytesIO(images["v0"]["bytes"])).size
    assert w == h <= newsletter.THUMB_PX                  # square, bounded, never upscaled
    # Byte cap: with an absurdly small budget the gatherer degrades to fewer images, not a crash.
    monkeypatch.setattr(newsletter, "MAX_IMAGE_BYTES", 1)
    assert newsletter.gather_images(cfg, d) == {}


def test_missing_files_cost_a_photo_never_the_issue(tmp_path):
    cfg = mkcfg(tmp_path)
    d = mkdigest()            # plate crop + rep crops all point at files that don't exist
    assert newsletter.gather_images(cfg, d) == {}


def test_hero_seek_is_clamped_inside_the_clip(tmp_path, monkeypatch):
    """A plate stamped at the last buffered detection can sit PAST the clip's final frame; the
    -ss ffmpeg gets must stay inside [0, seconds-0.5] or the hero silently comes back empty."""
    cfg = mkcfg(tmp_path)
    (tmp_path / "clips" / "x.mp4").write_bytes(b"not really an mp4")
    calls = {}

    def fake_ffmpeg(args, timeout):
        # Combined seek: coarse -ss before -i plus accurate -ss after; the MOMENT is their sum.
        calls["ss"] = sum(float(args[i + 1]) for i, a in enumerate(args) if a == "-ss")
        Path(args[-1]).write_bytes(jpeg_bytes(1280, 720))

    monkeypatch.setattr(newsletter, "_run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(newsletter.shutil, "which", lambda n: "ffmpeg")
    plate = {"crop_path": None, "species": "raccoon",
             "time": "2026-08-11T22:20:00-07:00",     # 630 s after a 200 s clip started
             "clip": {"clip_path": "clips/x.mp4", "start": "2026-08-11T22:09:30-07:00",
                      "seconds": 200.0}}
    img = newsletter._extract_clip_frame(cfg, plate)
    assert img is not None and calls["ss"] == pytest.approx(199.5)
    # Media time is not wall time: a clip can hold 120s of playback across 60 wall-seconds
    # (nominal-fps writer vs capture rate). With the wall span known, a moment 30 wall-seconds
    # in must seek to media second 60 -- unscaled, the hero shows the yard before the animal.
    plate2 = {"crop_path": None, "species": "raccoon",
              "time": "2026-08-11T22:10:00-07:00",
              "clip": {"clip_path": "clips/x.mp4", "start": "2026-08-11T22:09:30-07:00",
                       "seconds": 120.0, "wall_seconds": 60.0}}
    assert newsletter._extract_clip_frame(cfg, plate2) is not None
    assert calls["ss"] == pytest.approx(60.0)
    # And a machine without ffmpeg falls back soft.
    monkeypatch.setattr(newsletter.shutil, "which", lambda n: None)
    assert newsletter._extract_clip_frame(cfg, plate) is None


def test_hero_prefers_the_big_crop_itself(tmp_path):
    """A hero-sized crop IS the hero -- the one image guaranteed to show the moment. (Clip
    frames were measured seconds off the moment: nominal-fps media time is not wall time.)"""
    from PIL import Image
    cfg = mkcfg(tmp_path)
    (tmp_path / "crops").mkdir()
    (tmp_path / "crops" / "big.jpg").write_bytes(jpeg_bytes(800, 600))
    data = newsletter._hero_bytes(cfg, {"crop_path": "crops/big.jpg"})
    assert Image.open(io.BytesIO(data)).size == (800, 600)
    # A small crop must NOT become the hero (that's the blurry-ovals upscale); nothing else
    # available -> None, and the caller shows it small instead.
    (tmp_path / "crops" / "small.jpg").write_bytes(jpeg_bytes(200, 150))
    assert newsletter._hero_bytes(cfg, {"crop_path": "crops/small.jpg"}) is None


def test_hero_is_framed_around_the_animal(tmp_path, monkeypatch):
    """_hero_bytes must crop the extracted frame to the bbox neighbourhood -- the whole-yard
    hero ('sharp bricks, small blurry animal') is the thing this iteration buried."""
    from PIL import Image
    cfg = mkcfg(tmp_path)
    (tmp_path / "clips" / "x.mp4").write_bytes(b"not really an mp4")
    monkeypatch.setattr(newsletter, "_run_ffmpeg",
                        lambda args, timeout: Path(args[-1]).write_bytes(jpeg_bytes(1280, 720)))
    monkeypatch.setattr(newsletter.shutil, "which", lambda n: "ffmpeg")
    plate = {"crop_path": None, "species": "raccoon", "time": "2026-08-11T22:10:10-07:00",
             "clip": {"clip_path": "clips/x.mp4", "start": "2026-08-11T22:10:00-07:00",
                      "seconds": 60.0},
             "bbox": (800, 400, 1100, 700), "frame_size": (1920, 1080), "source": "glass_door_cam"}
    data = newsletter._hero_bytes(cfg, plate)
    w, h = Image.open(io.BytesIO(data)).size
    assert (w, h) != (1280, 720)                     # not the whole frame
    assert 400 <= w <= 900 and 300 <= h <= 720       # a neighbourhood, with minimums honoured


# ---- cuteness ------------------------------------------------------------------------

def noisy_image(w, h, sharp_centre: bool):
    """A synthetic crop: high-frequency texture in the centre box or in the border ring."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(7)
    flat = np.full((h, w), 120, dtype=np.uint8)
    noise = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    y0, y1, x0, x1 = h // 4, 3 * h // 4, w // 4, 3 * w // 4
    out = noise.copy() if not sharp_centre else flat.copy()
    if sharp_centre:
        out[y0:y1, x0:x1] = noise[y0:y1, x0:x1]
    else:
        out[y0:y1, x0:x1] = flat[y0:y1, x0:x1]
    return Image.fromarray(out, "L")


def test_focus_stats_finds_where_the_sharpness_lives():
    c, b = newsletter._focus_stats(noisy_image(200, 160, sharp_centre=True))
    assert c > b * 3                                  # sharp animal, soft background
    c2, b2 = newsletter._focus_stats(noisy_image(200, 160, sharp_centre=False))
    assert b2 > c2 * 3                                # sharp bricks, blurred animal


def test_cuteness_prefers_the_close_animal_over_sharp_bricks():
    """The failure that started this: a small motion-blurred raccoon on crisply focused pavers
    outscored everything on whole-crop sharpness. Size + focus-on-subject must invert that."""
    big_soft = {"bbox_x1": 400, "bbox_y1": 300, "bbox_x2": 1000, "bbox_y2": 800,
                "frame_w": 1920, "frame_h": 1080, "verified": None,
                "individual_id": None, "species": "raccoon", "detection_class": "animal",
                "source": "glass_door_cam"}
    tiny_bricks = {"bbox_x1": 900, "bbox_y1": 500, "bbox_x2": 1000, "bbox_y2": 580,
                   "frame_w": 1920, "frame_h": 1080, "verified": None,
                   "individual_id": None, "species": "raccoon", "detection_class": "animal",
                   "source": "trail_cam_sd"}
    animal_focus = {"centre_var": 900.0, "border_var": 150.0, "eyeshine": 0.0}
    brick_focus = {"centre_var": 150.0, "border_var": 2400.0, "eyeshine": 0.0}
    assert (newsletter._cuteness(big_soft, animal_focus)
            > 3 * newsletter._cuteness(tiny_bricks, brick_focus))
    # Eyeshine (facing the camera) must be worth something on otherwise-equal shots.
    eyes = dict(animal_focus, eyeshine=0.5)
    assert newsletter._cuteness(big_soft, eyes) > newsletter._cuteness(big_soft, animal_focus)


def test_hero_region_math():
    # Scaled: bbox in 1920x1080 frame coords, image is the 1280x720 clip frame.
    r = newsletter._hero_region((960, 540, 1260, 840), (1280, 720), (1920, 1080))
    x0, y0, x1, y1 = r
    assert 0 <= x0 < 640 <= 840 <= x1 <= 1280        # contains the scaled bbox (640..840)
    assert x1 - x0 >= 480 and y1 - y0 >= 360         # padded neighbourhood, not a tight cut
    # Minimum size: a tiny animal gets scene context, slid inside the image at the corner.
    r = newsletter._hero_region((10, 10, 60, 50), (1280, 720), (1280, 720))
    assert r[0] == 0 and r[1] == 0 and r[2] - r[0] >= 560 and r[3] - r[1] >= 420
    # Trail cam: the region stays above the OSD banner...
    r = newsletter._hero_region((600, 300, 900, 500), (1280, 720), (1280, 720),
                                source="trail_cam_sd")
    assert r[3] <= 720 * (1 - newsletter.TRAILCAM_BANNER_FRAC) + 1
    # ...unless the animal itself stands in it (never cut the animal to dodge the banner).
    r = newsletter._hero_region((600, 500, 900, 715), (1280, 720), (1280, 720),
                                source="trail_cam_sd")
    assert r[3] >= 714


def test_banner_touches():
    row = {"source": "trail_cam_sd", "frame_h": 1080, "bbox_y2": 1040}
    assert newsletter._banner_touches(row)
    assert not newsletter._banner_touches({**row, "bbox_y2": 900})
    assert not newsletter._banner_touches({**row, "source": "glass_door_cam"})


def test_pick_plate_falls_back_without_a_db(tmp_path):
    cfg = mkcfg(tmp_path)                             # db_path points at nothing
    assert newsletter.pick_plate(cfg, mkdigest()) is None
    assert newsletter.pick_plate(cfg, mkdigest(empty=True)) is None


# ---- sending -------------------------------------------------------------------------

def test_resend_payload_shape(tmp_path):
    cfg = mkcfg(tmp_path)
    images = {"hero": {"bytes": b"\xff\xd8fake", "mime": "image/jpeg", "hero": True}}
    p = newsletter.resend_payload(cfg, "subj", "<b>h</b>", "t", images)
    assert p["from"] == cfg.email_from and p["to"] == ["me@example.com"]
    att = p["attachments"][0]
    assert att["content_id"] == "hero" and att["filename"] == "hero.jpg"
    assert base64.b64decode(att["content"]) == b"\xff\xd8fake"
    # --to override reroutes without touching cfg.
    assert newsletter.resend_payload(cfg, "s", "h", "t", {}, to="x@y.z")["to"] == ["x@y.z"]


def test_send_issue_success_and_refusal(tmp_path, monkeypatch):
    cfg = mkcfg(tmp_path)
    seen = {}

    class OkResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"id": "email_123"}).encode()

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["auth"] = req.headers.get("Authorization")
        # Header keys are capitalized by urllib's Request; read case-insensitively.
        seen["ua"] = next((v for k, v in req.headers.items() if k.lower() == "user-agent"), None)
        seen["body"] = json.loads(req.data.decode())
        return OkResp()

    monkeypatch.setattr(newsletter.urllib.request, "urlopen", fake_urlopen)
    out = newsletter.send_issue(cfg, "s", "<p>h</p>", "t", {})
    assert out == "email_123"
    assert seen["url"] == newsletter.RESEND_URL and seen["auth"] == "Bearer re_test"
    assert seen["body"]["subject"] == "s"
    # A User-Agent is MANDATORY: Resend sits behind Cloudflare, which bans urllib's default
    # "Python-urllib/3.x" signature outright -- 403 "error code: 1010", before the API sees it.
    assert seen["ua"] and "python-urllib" not in seen["ua"].lower()

    def refuse(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 422, "Unprocessable", {},
                                     io.BytesIO(b'{"message":"domain not verified"}'))

    monkeypatch.setattr(newsletter.urllib.request, "urlopen", refuse)
    with pytest.raises(RuntimeError, match="domain not verified"):
        newsletter.send_issue(cfg, "s", "h", "t", {})


def test_cloudflare_block_is_named_as_such(tmp_path, monkeypatch):
    """A Cloudflare 1010 says nothing about keys or domains, and reads exactly like an auth
    failure -- so the error must name it, or the next reader debugs the wrong thing."""
    cfg = mkcfg(tmp_path)

    def blocked(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {},
                                     io.BytesIO(b"error code: 1010"))

    monkeypatch.setattr(newsletter.urllib.request, "urlopen", blocked)
    with pytest.raises(RuntimeError, match="CLOUDFLARE"):
        newsletter.send_issue(cfg, "s", "h", "t", {})


def test_each_recipient_gets_their_own_message_and_never_sees_the_others(tmp_path, monkeypatch):
    """Two readers, two POSTs, each To: carrying exactly one address -- and neither reader's
    address appearing ANYWHERE in the other's body. A shared To: header is a disclosure you
    cannot un-send, so this asserts the whole payload, not just the To: field."""
    cfg = mkcfg(tmp_path, email_to="a@x.com, b@y.com")
    bodies = []

    class OkResp:
        def __init__(self, eid):
            self.eid = eid

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"id": self.eid}).encode()

    def fake_urlopen(req, timeout):
        body = json.loads(req.data.decode())
        bodies.append(body)
        # Tie the id to the address, so the returned ids prove WHICH send produced them.
        return OkResp("id_" + body["to"][0])

    monkeypatch.setattr(newsletter.urllib.request, "urlopen", fake_urlopen)
    out = newsletter.send_issue(cfg, "s", "<p>h</p>", "t", {})

    assert len(bodies) == 2, "one message per recipient, not one message to a shared header"
    assert [b["to"] for b in bodies] == [["a@x.com"], ["b@y.com"]]
    for body, mine in zip(bodies, ["a@x.com", "b@y.com"]):
        blob = json.dumps(body)
        for other in {"a@x.com", "b@y.com"} - {mine}:
            assert other not in blob, "%s must not appear anywhere in %s's message" % (other, mine)
    # Both ids come back, so the log line still says what was actually sent.
    assert out == "id_a@x.com, id_b@y.com"


def test_a_partial_send_names_who_already_received_it(tmp_path, monkeypatch):
    """One send per reader means a run can half-succeed. Re-running would give the delivered
    readers a SECOND copy, so the error has to name them rather than leave it to be guessed."""
    cfg = mkcfg(tmp_path, email_to="a@x.com, b@y.com")
    calls = []

    class OkResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"id": "email_ok"}).encode()

    def first_ok_then_refuse(req, timeout):
        calls.append(json.loads(req.data.decode())["to"])
        if len(calls) == 1:
            return OkResp()
        raise urllib.error.HTTPError(req.full_url, 422, "Unprocessable", {},
                                     io.BytesIO(b'{"message":"mailbox unavailable"}'))

    monkeypatch.setattr(newsletter.urllib.request, "urlopen", first_ok_then_refuse)
    with pytest.raises(RuntimeError) as excinfo:
        newsletter.send_issue(cfg, "s", "h", "t", {})
    msg = str(excinfo.value)
    assert "a@x.com" in msg and "second copy" in msg
    assert "b@y.com" in msg and "mailbox unavailable" in msg
    # The failure must not stop the loop early in a way that hides who was tried.
    assert calls == [["a@x.com"], ["b@y.com"]]


def test_a_network_drop_mid_loop_still_names_who_received_it(tmp_path, monkeypatch):
    """The 07:00 task runs unattended over home Wi-Fi, so a DNS or socket failure -- NOT an HTTP
    refusal -- is the likeliest way a send half-succeeds. _post_issue converts only HTTPError, so
    this used to escape the loop: the log said just "send FAILED", and the obvious response to
    that (re-run it) sends a second copy to whoever already had one."""
    cfg = mkcfg(tmp_path, email_to="a@x.com, b@y.com")
    tried = []

    class OkResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"id": "email_ok"}).encode()

    def drop_on_second(req, timeout):
        tried.append(json.loads(req.data.decode())["to"][0])
        if len(tried) == 1:
            return OkResp()
        raise urllib.error.URLError("[Errno 11001] getaddrinfo failed")

    monkeypatch.setattr(newsletter.urllib.request, "urlopen", drop_on_second)
    with pytest.raises(RuntimeError) as excinfo:
        newsletter.send_issue(cfg, "s", "h", "t", {})
    msg = str(excinfo.value)
    assert "a@x.com" in msg and "second copy" in msg, "must name who already has it"
    assert "b@y.com" in msg and "unreachable" in msg
    assert tried == ["a@x.com", "b@y.com"]


def test_a_failure_on_the_first_reader_still_tries_the_rest(tmp_path, monkeypatch):
    """A transport error on reader one used to abort the whole loop, so reader two was never
    attempted -- strictly worse than the single-POST send it replaced, which at least failed for
    everyone equally."""
    cfg = mkcfg(tmp_path, email_to="a@x.com, b@y.com")
    tried = []

    class OkResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"id": "email_ok"}).encode()

    def fail_first(req, timeout):
        tried.append(json.loads(req.data.decode())["to"][0])
        if len(tried) == 1:
            raise urllib.error.URLError("connection refused")
        return OkResp()

    monkeypatch.setattr(newsletter.urllib.request, "urlopen", fail_first)
    with pytest.raises(RuntimeError) as excinfo:
        newsletter.send_issue(cfg, "s", "h", "t", {})
    assert tried == ["a@x.com", "b@y.com"], "reader two must still be attempted"
    assert "b@y.com" in str(excinfo.value), "and the reader who DID receive it must be named"


def test_a_timeout_says_maybe_delivered_rather_than_failed(tmp_path, monkeypatch):
    """Resend can accept the message and THEN the read times out. Reporting that address as
    simply failed invites a re-run that duplicates it, so the error has to say the delivery is
    unknown rather than negative."""
    cfg = mkcfg(tmp_path, email_to="a@x.com")

    def time_out(req, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(newsletter.urllib.request, "urlopen", time_out)
    with pytest.raises(RuntimeError, match="MAY have been delivered"):
        newsletter.send_issue(cfg, "s", "h", "t", {})


def test_send_with_no_recipients_is_an_error_not_a_silent_noop(tmp_path):
    """email_configured() gates this upstream, but a send that quietly does nothing would log
    'sent' to nobody -- which reads exactly like a quiet night."""
    with pytest.raises(RuntimeError, match="No recipients"):
        newsletter.send_issue(mkcfg(tmp_path, email_to=""), "s", "h", "t", {})


def test_email_configured(tmp_path):
    assert newsletter.email_configured(mkcfg(tmp_path))
    assert not newsletter.email_configured(mkcfg(tmp_path, email_resend_api_key=None))


def test_say_survives_pythonw(monkeypatch, tmp_path, capsys):
    """The scheduled task runs under pythonw.exe, where sys.stdout is None and a bare print()
    RAISES AttributeError -- the failure mode that would kill the 07:00 run before it rendered
    anything. Console-less, _say must both survive and leave the line in logs/newsletter.log."""
    log = tmp_path / "logs" / "newsletter.log"
    monkeypatch.setattr(newsletter, "_LOG_PATH", log)
    monkeypatch.setattr(newsletter.sys, "stdout", None)
    newsletter._say("issue written")                      # must not raise
    assert "issue written" in log.read_text(encoding="utf-8")


# ---- the dawn wait -------------------------------------------------------------------

def test_wait_for_dawn(monkeypatch, tmp_path):
    cfg = mkcfg(tmp_path)
    tz = timezone(timedelta(hours=-8))
    dawn = datetime(2026, 12, 21, 8, 1, tzinfo=tz)          # midwinter: dawn after 8am
    monkeypatch.setattr(newsletter.stats, "_sun", lambda c, d: (dawn, dawn + timedelta(hours=8)))
    naps = []
    # Fired at 07:00 the script must wait until 08:11 (dawn + 10 min), in bounded steps.
    waited = newsletter.wait_for_dawn(cfg, now=datetime(2026, 12, 21, 7, 0, tzinfo=tz),
                                      _sleep=naps.append)
    assert waited == pytest.approx(71 * 60) and sum(naps) == waited
    # Fired after dawn (every summer morning) it must not sleep at all.
    assert newsletter.wait_for_dawn(cfg, now=datetime(2026, 12, 21, 9, 0, tzinfo=tz),
                                    _sleep=naps.append) == 0.0


def test_archive_written_with_data_uris(tmp_path):
    cfg = mkcfg(tmp_path)
    images = {"hero": {"bytes": jpeg_bytes(), "mime": "image/jpeg", "hero": True}}
    p = newsletter.write_archive(cfg, mkbundle(), images)
    assert p == tmp_path / "reports" / "mail" / "2026-08-12-night.html"
    body = p.read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in body and "cid:" not in body


# ---- recipients ----------------------------------------------------------------------

def test_recipients_accepts_every_way_a_person_types_them(tmp_path):
    """A household grows. One address, several comma-separated (what people actually type into
    a config), or a list -- all mean the same To: line."""
    r = newsletter.recipients
    assert r(mkcfg(tmp_path, email_to="a@x.com")) == ["a@x.com"]
    assert r(mkcfg(tmp_path, email_to="a@x.com, b@y.com")) == ["a@x.com", "b@y.com"]
    assert r(mkcfg(tmp_path, email_to="a@x.com; b@y.com")) == ["a@x.com", "b@y.com"]
    assert r(mkcfg(tmp_path, email_to=["a@x.com", "b@y.com"])) == ["a@x.com", "b@y.com"]
    assert r(mkcfg(tmp_path, email_to=("a@x.com",))) == ["a@x.com"]
    # Blanks from a trailing comma, and a duplicate that would otherwise be a second billed
    # recipient of the same paper.
    assert r(mkcfg(tmp_path, email_to="a@x.com, , b@y.com, A@X.com")) == ["a@x.com", "b@y.com"]
    assert r(mkcfg(tmp_path, email_to=None)) == []
    # An explicit --to overrides config entirely, and takes the same forms.
    assert r(mkcfg(tmp_path, email_to="a@x.com"), "c@z.com, d@z.com") == ["c@z.com", "d@z.com"]


def test_payload_and_configured_follow_the_list(tmp_path):
    cfg = mkcfg(tmp_path, email_to="a@x.com, b@y.com")
    # resend_payload is a per-recipient body: send_issue calls it once per address. Its
    # no-override form still mirrors the whole list, but no real send uses that shape --
    # see test_each_recipient_gets_their_own_message_and_never_sees_the_others.
    assert newsletter.resend_payload(cfg, "s", "h", "t", {}, to="a@x.com")["to"] == ["a@x.com"]
    assert newsletter.resend_payload(cfg, "s", "h", "t", {})["to"] == ["a@x.com", "b@y.com"]
    assert newsletter.email_configured(cfg)
    assert not newsletter.email_configured(mkcfg(tmp_path, email_to=""))
    assert not newsletter.email_configured(mkcfg(tmp_path, email_to=[]))


# ---- where the links point -----------------------------------------------------------

def test_dashboard_base_prefers_lan_ip_over_hostname(tmp_path, monkeypatch):
    """A bare Windows hostname does not resolve from a phone (mDNS, not NetBIOS), and the phone
    is where this is read -- so the LAN IP wins whenever one can be found."""
    cfg = mkcfg(tmp_path, email_dashboard_url=None)
    monkeypatch.setattr(newsletter, "_lan_ip", lambda: "192.168.1.101")
    assert newsletter.dashboard_base(cfg) == "http://192.168.1.101:8000"
    # No LAN to find -> the hostname is still better than nothing.
    monkeypatch.setattr(newsletter, "_lan_ip", lambda: None)
    assert newsletter.dashboard_base(cfg).startswith("http://")
    # An explicit setting always wins, trailing slash trimmed.
    assert newsletter.dashboard_base(
        mkcfg(tmp_path, email_dashboard_url="https://yard.example/")) == "https://yard.example"


def test_lan_ip_refuses_a_public_address(monkeypatch):
    """A public address here would mean the rig sits directly on the internet; a link to it does
    not belong in an email."""
    class FakeSock:
        def __init__(self, ip):
            self.ip = ip

        def settimeout(self, t):
            pass

        def connect(self, addr):
            pass

        def getsockname(self):
            return (self.ip, 9)

        def close(self):
            pass

    monkeypatch.setattr(newsletter.socket, "socket", lambda *a: FakeSock("8.8.8.8"))
    assert newsletter._lan_ip() is None
    monkeypatch.setattr(newsletter.socket, "socket", lambda *a: FakeSock("192.168.1.101"))
    assert newsletter._lan_ip() == "192.168.1.101"


def test_dashboard_answering(monkeypatch):
    import contextlib
    monkeypatch.setattr(newsletter.socket, "create_connection",
                        lambda addr, timeout: contextlib.nullcontext())
    assert newsletter.dashboard_answering("http://192.168.1.101:8000") is True

    def refuse(addr, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(newsletter.socket, "create_connection", refuse)
    assert newsletter.dashboard_answering("http://192.168.1.101:8000") is False
    assert newsletter.dashboard_answering("not a url") is None


def test_issue_deep_links_into_the_dashboard():
    """Every thing the paper talks about should open the page about that thing -- not the front
    door. The dashboard routes on the URL hash, so these are the routes it actually parses."""
    rc = {"cast": [{"id": "miss b.", "species": "raccoon", "days_since": 2, "overdue": False,
                    "last_seen": "2026-08-09T02:00:00-07:00"}]}
    b = mkbundle(mkdigest(), rc)
    html = newsletter.render_email(b, {}, lambda cid: None)
    assert "http://rig:8000/#dispatch/2026-08-12/night" in html   # the hero + masthead link
    assert "http://rig:8000/#day/2026-08-11" in html              # the visit row's own day
    assert "http://rig:8000/#species/raccoon" in html             # the roll row's sheet
    assert "http://rig:8000/#profile/miss%20b." in html           # the cast member's profile
    assert "http://rig:8000/#live" in html                        # footer: the other rooms
    # Percent-encoding matters: the cast contains spaces, apostrophes and the " + " of a family
    # stamp, and a raw '+' in a URL fragment would decode as a space.
    assert "%20%2B%20" in newsletter.render_email(
        mkbundle(mkdigest(), {"cast": [{"id": "stan + kits", "species": "raccoon",
                                        "days_since": 0, "overdue": False,
                                        "last_seen": "2026-08-11T23:00:00-07:00"}]}),
        {}, lambda cid: None)


def test_unidentified_species_gets_no_dead_link():
    """'animal' has no catalogue sheet -- the dashboard leaves it unclickable, so must the paper."""
    d = mkdigest()
    d["species"][0]["species"] = "animal"
    html = newsletter.render_email(mkbundle(d), {}, lambda cid: None)
    assert "#species/animal" not in html


def test_footer_says_when_the_rig_was_not_answering():
    """A dead tap should accuse the right thing: the dashboard being down, not the address."""
    up = newsletter.render_email({**mkbundle(), "lan_ok": True}, {}, lambda cid: None)
    assert "work on the home Wi-Fi" in up and "wasn’t answering" not in up
    down = newsletter.render_email({**mkbundle(), "lan_ok": False}, {}, lambda cid: None)
    assert "wasn’t answering" in down


def test_text_part_carries_the_links_too():
    txt = newsletter.render_text(mkbundle())
    assert "http://rig:8000/#dispatch/2026-08-12/night" in txt
    assert "http://rig:8000/#live" in txt
    assert "home Wi-Fi" in txt


def test_preview_never_waits_for_dawn(tmp_path, monkeypatch):
    """--no-send renders a preview. Waiting hours to look at a file is absurd, and because the
    wait is silent on a buffered pipe it reads as a hang (measured: a midnight `--no-send` sat
    for 15 minutes having used 0.015s of CPU)."""
    called = []
    monkeypatch.setattr(newsletter, "wait_for_dawn", lambda *a, **k: called.append(1))
    monkeypatch.setattr(newsletter, "collect_issue",
                        lambda *a, **k: {"d": {"empty": True, "reason": "no data"}})
    newsletter.main(["--no-send"])
    assert not called
