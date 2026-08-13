"""
Tests for the self-calibrating static false-fire filter (staticfilter).

The filter deletes rows, so what it must NEVER do matters more than what it does. The guards
here are, in order of how much a failure would cost:

  * a real animal is never furniture. A moving animal's box changes shape frame to frame, so it
    should not even cluster; and a lingering one that DOES hold still is caught by the span rule
    rather than by luck. Both are asserted.
  * a burst is not a span. A trail cam fires six photos in two seconds; those boxes are nearly
    identical and would clear min_count on their own. Only the span rule separates them from a
    grill, so it is tested directly.
  * deletion is survivable. visits.representative_detection_id references detections WITHOUT
    ON DELETE CASCADE (unlike detection_embeddings, which does), so a visit pointing at a doomed
    row used to abort the whole sweep with a FOREIGN KEY error -- a real failure, hit on the
    first live run 2026-08-05.
  * a batch sweep stays inside its batch: rows imported by earlier cycles must be untouched,
    because "the same spot" only means something within one camera placement.

No detector, no images: rows are written straight to a temp DB, which is all the filter reads.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import db
import staticfilter


def _conn(tmp_path):
    return db.connect(tmp_path / "t.db")


def _add(conn, *, when: datetime, box, species="brown rat", source="trail_cam_sd",
         conf=0.4) -> int:
    """One detection row at `when` with `box`; returns its id."""
    db.insert_detection(
        conn, timestamp=when.astimezone().isoformat(), source=source,
        detection_class="animal", confidence=conf, bbox=box, frame_w=2560, frame_h=1440,
        crop_path=f"crops/x/{when.strftime('%H%M%S%f')}.jpg", frame_path=None,
    )
    return conn.execute("SELECT MAX(id) FROM detections").fetchone()[0]


T0 = datetime(2026, 8, 4, 22, 0, 0)
GRILL = (1885.0, 645.0, 2118.0, 1129.0)


def _fill_static(conn, n=40, minutes=300, box=GRILL):
    """A grill: the same box, spread evenly over `minutes`."""
    for i in range(n):
        _add(conn, when=T0 + timedelta(minutes=minutes * i / (n - 1)), box=box)


def test_static_object_is_found(tmp_path):
    conn = _conn(tmp_path)
    _fill_static(conn)
    rows = staticfilter.load_rows(conn, "trail_cam_sd")
    static, _ = staticfilter.find_static(rows)
    assert len(static) == 1
    assert static[0].count == 40
    assert static[0].span_minutes == 300


def test_moving_animal_never_clusters(tmp_path):
    """A raccoon walking across frame: each box overlaps the last a little, none by 0.75."""
    conn = _conn(tmp_path)
    for i in range(60):
        x = 300.0 + i * 40
        _add(conn, when=T0 + timedelta(seconds=i * 20), species="raccoon",
             box=(x, 600.0 + (i % 5) * 12, x + 230, 830.0 + (i % 5) * 12))
    rows = staticfilter.load_rows(conn, "trail_cam_sd")
    static, clusters = staticfilter.find_static(rows)
    assert static == []
    assert len(clusters) > 1, "a moving animal must not collapse into one spot"


def test_lingering_animal_survives_on_count(tmp_path):
    """An animal that DOES hold one box -- but only 12 times -- stays: min_count protects it.
    This is the deliberate false-negative side of the trade (see the module docstring)."""
    conn = _conn(tmp_path)
    for i in range(12):
        _add(conn, when=T0 + timedelta(minutes=i * 20), species="raccoon", box=GRILL)
    rows = staticfilter.load_rows(conn, "trail_cam_sd")
    static, _ = staticfilter.find_static(rows)
    assert static == []


def test_burst_is_not_static(tmp_path):
    """40 near-identical boxes inside three seconds -- clears min_count, fails on span."""
    conn = _conn(tmp_path)
    for i in range(40):
        _add(conn, when=T0 + timedelta(milliseconds=i * 75), species="raccoon", box=GRILL)
    rows = staticfilter.load_rows(conn, "trail_cam_sd")
    static, _ = staticfilter.find_static(rows)
    assert static == [], "a photo burst is not five hours of sitting still"


def test_apply_deletes_rows_and_cascades_embeddings(tmp_path):
    conn = _conn(tmp_path)
    _fill_static(conn)
    ids = [r[0] for r in conn.execute("SELECT id FROM detections")]
    db.insert_embedding(conn, detection_id=ids[0], model="m", dim=2,
                        embedding=b"\x00\x01", created_at=None)
    rows = staticfilter.load_rows(conn, "trail_cam_sd")
    static, _ = staticfilter.find_static(rows)
    n = staticfilter.apply(conn, tmp_path / "t.db", "trail_cam_sd", static)
    assert n == 40
    assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM detection_embeddings").fetchone()[0] == 0
    assert staticfilter.manifest_path(tmp_path / "t.db", "trail_cam_sd").exists()


def test_apply_releases_visit_representative(tmp_path):
    """The FOREIGN KEY regression: a visit whose representative crop is about to be deleted."""
    conn = _conn(tmp_path)
    _fill_static(conn)
    doomed = conn.execute("SELECT MIN(id) FROM detections").fetchone()[0]
    db.insert_visit(conn, source="trail_cam_sd", species="brown rat", individual_id=None,
                    started_at=T0.astimezone().isoformat(), ended_at=T0.astimezone().isoformat(),
                    detection_count=1, max_confidence=0.4, representative_detection_id=doomed)
    rows = staticfilter.load_rows(conn, "trail_cam_sd")
    static, _ = staticfilter.find_static(rows)
    assert staticfilter.apply(conn, tmp_path / "t.db", "trail_cam_sd", static) == 40
    assert conn.execute(
        "SELECT representative_detection_id FROM visits").fetchone()[0] is None


def test_sweep_batch_ignores_earlier_rows(tmp_path):
    """Last cycle's rows are below the watermark and must survive, even though they look static:
    a previous card is a previous camera placement."""
    conn = _conn(tmp_path)
    _fill_static(conn)                                  # "previous cycle"
    watermark = conn.execute("SELECT MAX(id) FROM detections").fetchone()[0]
    _fill_static(conn, box=(100.0, 100.0, 300.0, 500.0))  # "this cycle"
    cfg = type("C", (), {"db_path": tmp_path / "t.db"})()
    n = staticfilter.sweep_batch(conn, cfg, "trail_cam_sd", min_id=watermark)
    assert n == 40
    survivors = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    assert survivors == 40, "rows from before the watermark must be untouched"


def test_other_sources_are_never_swept(tmp_path):
    """The live rig has its own ignore_zones; a trail-cam sweep must not reach across sources."""
    conn = _conn(tmp_path)
    _fill_static(conn)
    for i in range(40):
        _add(conn, when=T0 + timedelta(minutes=i * 8), box=GRILL, source="glass_door_cam")
    rows = staticfilter.load_rows(conn, "trail_cam_sd")
    static, _ = staticfilter.find_static(rows)
    staticfilter.apply(conn, tmp_path / "t.db", "trail_cam_sd", static)
    assert conn.execute(
        "SELECT COUNT(*) FROM detections WHERE source='glass_door_cam'").fetchone()[0] == 40


# --- Rigidity: the second test, for furniture that STOPS before it clears the span ------------
# These need real crops on disk (the span tests above are pure DB), so each writes JPEGs into
# tmp_path and points crops_root at it. The span is kept far under min_span_minutes throughout,
# so anything caught here was caught by rigidity and nothing else.

def _pattern(shift=0, bias=0):
    """A deterministic 64x64 texture. `shift` moves it (an animal changing pose); `bias`
    brightens every pixel equally (the IR gain or the sun, which must cost nothing)."""
    from PIL import Image
    im = Image.new("L", (64, 64))
    im.putdata([max(0, min(255, ((x + shift) * 7 ^ (y * 13)) % 256 + bias))
                for y in range(64) for x in range(64)])
    return im


def _add_with_crop(conn, tmp_path, *, when, box, img, name):
    (tmp_path / "crops").mkdir(exist_ok=True)
    rel = f"crops/{name}.jpg"
    img.save(tmp_path / rel, quality=95)
    db.insert_detection(
        conn, timestamp=when.astimezone().isoformat(), source="trail_cam_sd",
        detection_class="animal", confidence=0.4, bbox=box, frame_w=2560, frame_h=1440,
        crop_path=rel, frame_path=None,
    )


BIN = (2071.0, 490.0, 2299.0, 970.0)


def _fill_short(conn, tmp_path, imgs, box=BIN, minutes=25):
    """20 detections holding one box for `minutes` -- under the 60-minute span rule on purpose."""
    for i, img in enumerate(imgs):
        _add_with_crop(conn, tmp_path, when=T0 + timedelta(minutes=minutes * i / (len(imgs) - 1)),
                       box=box, img=img, name=f"c{i:03d}")


def test_rigid_object_is_found_even_though_span_would_miss_it(tmp_path):
    """2026-08-11: the battery died and a bin that had fired 179 times in 29 minutes stopped
    before it could clear the span. It was labelled 'brown rat' and kept. It must not be."""
    conn = _conn(tmp_path)
    _fill_short(conn, tmp_path, [_pattern() for _ in range(20)])
    rows = staticfilter.load_rows(conn, "trail_cam_sd")
    static, _ = staticfilter.find_static(
        rows, rigid_maxpair=staticfilter.DEFAULT_RIGID_MAXPAIR, crops_root=tmp_path)
    assert len(static) == 1
    assert static[0].reason == "rigid", "span cannot reach a 25-minute cluster"
    assert static[0].rigidity < staticfilter.DEFAULT_RIGID_MAXPAIR


def test_rigidity_survives_an_exposure_shift(tmp_path):
    """The same object under a changing IR gain is still the same object. This is the whole
    reason the crops are z-normalised before they are compared."""
    conn = _conn(tmp_path)
    _fill_short(conn, tmp_path, [_pattern(bias=b) for b in range(0, 60, 3)])
    rows = staticfilter.load_rows(conn, "trail_cam_sd")
    static, _ = staticfilter.find_static(
        rows, rigid_maxpair=staticfilter.DEFAULT_RIGID_MAXPAIR, crops_root=tmp_path)
    assert len(static) == 1 and static[0].reason == "rigid"


def test_lingering_animal_is_never_rigid(tmp_path):
    """The guard that matters. A raccoon that feeds in one spot for 23 minutes holds its box --
    it is only its CHANGING pixels that save it, so this asserts they do."""
    conn = _conn(tmp_path)
    _fill_short(conn, tmp_path, [_pattern(shift=i * 3) for i in range(20)])
    rows = staticfilter.load_rows(conn, "trail_cam_sd")
    static, clusters = staticfilter.find_static(
        rows, rigid_maxpair=staticfilter.DEFAULT_RIGID_MAXPAIR, crops_root=tmp_path)
    assert static == [], "a moving animal must never be called furniture"
    assert clusters[0].rigidity > staticfilter.DEFAULT_RIGID_MAXPAIR


def test_unreadable_crops_are_never_deleted(tmp_path):
    """What cannot be measured must not be deleted -- the same rule the module already applies
    to a row whose timestamp will not parse. Here the crops simply are not there."""
    conn = _conn(tmp_path)
    for i in range(20):
        _add(conn, when=T0 + timedelta(minutes=25 * i / 19), box=BIN)   # crop_path points nowhere
    rows = staticfilter.load_rows(conn, "trail_cam_sd")
    static, clusters = staticfilter.find_static(
        rows, rigid_maxpair=staticfilter.DEFAULT_RIGID_MAXPAIR, crops_root=tmp_path)
    assert static == []
    assert clusters[0].rigidity is None, "unjudged, which must not read as 0.0"


def test_rigidity_is_opt_in(tmp_path):
    """find_static without a threshold is the pre-2026-08-12 filter exactly, and reads no disk."""
    conn = _conn(tmp_path)
    _fill_short(conn, tmp_path, [_pattern() for _ in range(20)])
    rows = staticfilter.load_rows(conn, "trail_cam_sd")
    static, clusters = staticfilter.find_static(rows)
    assert static == []
    assert clusters[0].rigidity is None
