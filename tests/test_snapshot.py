"""Full-resolution snapshot crops (snapshot.py): matching, cutting, the worker's swap, and the
snapshot URL template's validation and resolution. No network: the fetcher is a stub that returns
a synthetic still."""
import ssl
import threading

import cv2
import numpy as np
import pytest

import cameras
import config
import db
import snapshot
from detector import Detection

STREAM_W, STREAM_H = 640, 360
STILL_W, STILL_H = 3840, 2160          # 6x the stream, same 16:9


def _animal(box, conf=0.9, cls="animal"):
    return Detection(class_id=0, class_name=cls, confidence=conf,
                     bbox=tuple(float(v) for v in box))


def _still(w=STILL_W, h=STILL_H):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.rectangle(img, (1200, 600), (1800, 1200), (200, 180, 160), -1)
    return img


def _cfg(tmp_path, **over):
    cfg = config.Config()
    cfg.db_path = tmp_path / "test.db"
    cfg.crops_dir = tmp_path / "crops"
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _spec():
    return config.CameraSpec("yard_ir", "rtsp://rig:pw@192.168.1.50:554/h264Preview_01_sub",
                             snapshot_url="http://192.168.1.50/snap?password=pw")


def _detection_row(cfg, box=(200, 100, 300, 200)):
    """A saved stream detection: the row plus its crop file, exactly as the capture loop leaves
    them. Returns (id, Detection, stream crop path as stored)."""
    det = _animal(box)
    day = cfg.crops_dir / "2026-10-18"
    day.mkdir(parents=True, exist_ok=True)
    crop = day / "2026-10-18T21-00-00-000_0_animal_0.90.jpg"
    cv2.imwrite(str(crop), np.full((120, 120, 3), 90, np.uint8))
    rel = db.rel_to_root(crop)
    conn = db.connect(cfg.db_path)
    try:
        det_id = db.insert_detection(conn, timestamp="2026-10-18T21:00:00-07:00", source="yard_ir",
                                     detection_class="animal", confidence=0.9, bbox=box,
                                     frame_w=STREAM_W, frame_h=STREAM_H, crop_path=rel,
                                     crop_quality=12.5)
    finally:
        conn.close()
    return det_id, det, rel


def _row(cfg, det_id):
    conn = db.connect(cfg.db_path)
    try:
        r = conn.execute("SELECT crop_path, stream_crop_path, crop_quality FROM detections "
                         "WHERE id = ?", (det_id,)).fetchone()
        return {"crop_path": r[0], "stream_crop_path": r[1], "crop_quality": r[2]}
    finally:
        conn.close()


def _cropper(cfg, snap_dets, still=None, fetch=None):
    return snapshot.SnapshotCropper(
        _spec(), cfg, lambda img: list(snap_dets), threading.Lock(),
        fetch=fetch or (lambda url, timeout: _still() if still is None else still))


# ---- matching ----------------------------------------------------------------------

def test_a_snapshot_box_is_matched_in_stream_coordinates():
    stream = [_animal((200, 100, 300, 200))]
    snap = [_animal((1206, 606, 1806, 1206))]          # the same animal, a few px later, at 6x
    got = snapshot.match(stream, snap, 1 / 6, 1 / 6, 0.1)
    assert got == {0: snap[0]}


def test_two_animals_do_not_share_one_snapshot_box():
    stream = [_animal((200, 100, 300, 200)), _animal((210, 105, 310, 205))]
    snap = [_animal((1200, 600, 1800, 1200))]
    got = snapshot.match(stream, snap, 1 / 6, 1 / 6, 0.1)
    assert len(got) == 1


def test_a_different_class_is_never_a_match():
    stream = [_animal((200, 100, 300, 200))]
    snap = [_animal((1200, 600, 1800, 1200), cls="person")]
    assert snapshot.match(stream, snap, 1 / 6, 1 / 6, 0.1) == {}


def test_an_animal_that_left_is_not_matched_to_one_elsewhere():
    stream = [_animal((200, 100, 300, 200))]
    snap = [_animal((3000, 1500, 3400, 1900))]
    assert snapshot.match(stream, snap, 1 / 6, 1 / 6, 0.1) == {}


# ---- cutting -----------------------------------------------------------------------

def test_a_big_crop_is_shrunk_to_the_cap_and_a_small_one_is_never_enlarged():
    img = _still()
    big = snapshot.cut(img, (0, 0, 3000, 1500), 0.0, 1280)
    assert max(big.shape[:2]) == 1280
    small = snapshot.cut(img, (100, 100, 300, 200), 0.0, 1280)
    assert small.shape[:2] == (100, 200)


def test_a_degenerate_box_cuts_nothing():
    assert snapshot.cut(_still(), (500, 500, 500, 900), 0.15, 1280) is None


# ---- the worker's swap -------------------------------------------------------------

def test_a_matched_detection_gets_the_hires_crop_and_keeps_the_stream_one(tmp_path):
    cfg = _cfg(tmp_path)
    det_id, det, rel = _detection_row(cfg)
    c = _cropper(cfg, [_animal((1206, 606, 1806, 1206))])
    assert c.process(snapshot.Job([(det_id, det, rel)], STREAM_W, STREAM_H)) == 1
    row = _row(cfg, det_id)
    assert row["stream_crop_path"] == rel
    assert row["crop_path"].endswith("_hires.jpg")
    img = cv2.imread(str(db.crop_abspath(row["crop_path"])))
    assert img is not None and max(img.shape[:2]) > 120     # more pixels than the stream crop
    # Sharpness depends on pixel scale, so the score stays the stream crop's.
    assert row["crop_quality"] == 12.5


def test_a_second_swap_never_overwrites_the_stream_crop(tmp_path):
    cfg = _cfg(tmp_path)
    det_id, det, rel = _detection_row(cfg)
    c = _cropper(cfg, [_animal((1206, 606, 1806, 1206))])
    c.process(snapshot.Job([(det_id, det, rel)], STREAM_W, STREAM_H))
    c.process(snapshot.Job([(det_id, det, rel)], STREAM_W, STREAM_H))
    assert _row(cfg, det_id)["stream_crop_path"] == rel


@pytest.mark.parametrize("case", ["fetch raises", "not an image", "animal gone",
                                  "different shape"])
def test_every_failure_leaves_the_row_as_the_stream_wrote_it(tmp_path, case):
    cfg = _cfg(tmp_path)
    det_id, det, rel = _detection_row(cfg)

    def boom(url, timeout):
        raise TimeoutError("timed out")

    c = {"fetch raises": lambda: _cropper(cfg, [], fetch=boom),
         "not an image": lambda: _cropper(cfg, [], fetch=lambda u, t: None),
         "animal gone": lambda: _cropper(cfg, []),
         "different shape": lambda: _cropper(cfg, [_animal((1206, 606, 1806, 1206))],
                                             still=_still(2880, 2160))}[case]()
    assert c.process(snapshot.Job([(det_id, det, rel)], STREAM_W, STREAM_H)) == 0
    assert _row(cfg, det_id) == {"crop_path": rel, "stream_crop_path": None, "crop_quality": 12.5}


def test_a_busy_worker_drops_the_next_job_instead_of_queueing(tmp_path):
    c = _cropper(_cfg(tmp_path), [])
    job = snapshot.Job([(1, _animal((0, 0, 10, 10)), "x.jpg")], STREAM_W, STREAM_H)
    assert c.submit(job) is True                # not started, so this one sits in the slot
    assert c.submit(job) is False
    assert c.n_skipped == 1


def test_the_running_worker_swaps_in_the_background(tmp_path):
    cfg = _cfg(tmp_path)
    det_id, det, rel = _detection_row(cfg)
    done = threading.Event()

    def fetch(url, timeout):
        return _still()

    c = _cropper(cfg, [_animal((1206, 606, 1806, 1206))], fetch=fetch).start()
    real = c.process
    c.process = lambda job: (real(job), done.set())[0]
    try:
        c.submit(snapshot.Job([(det_id, det, rel)], STREAM_W, STREAM_H))
        assert done.wait(10)
    finally:
        c.close()
    assert _row(cfg, det_id)["crop_path"].endswith("_hires.jpg")


# ---- fetching ------------------------------------------------------------------------

class _Reply:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.mark.parametrize("url, unverified", [("https://192.168.1.50/snap", True),
                                             ("HTTPS://192.168.1.50/snap", True),
                                             ("http://192.168.1.50/snap", False)])
def test_https_fetches_are_encrypted_without_checking_the_self_signed_cert(
        monkeypatch, url, unverified):
    seen = {}
    jpeg = cv2.imencode(".jpg", _still(64, 36))[1].tobytes()

    def urlopen(u, timeout, context=None):
        seen["context"] = context
        return _Reply(jpeg)

    monkeypatch.setattr(snapshot.urllib.request, "urlopen", urlopen)
    assert snapshot.fetch_jpeg(url, 5).shape[:2] == (36, 64)
    ctx = seen["context"]
    if unverified:
        assert ctx.verify_mode == ssl.CERT_NONE and ctx.check_hostname is False
    else:
        assert ctx is None


def test_a_json_error_reply_is_not_an_image(monkeypatch):
    monkeypatch.setattr(snapshot.urllib.request, "urlopen",
                        lambda u, timeout, context=None: _Reply(b'[{"code": 1, "error": {}}]'))
    assert snapshot.fetch_jpeg("https://192.168.1.50/snap", 5) is None


# ---- the URL template ----------------------------------------------------------------

REOLINK_SNAP = "https://{host}/cgi-bin/api.cgi?cmd=Snap&channel=0&user={user}&password={password}"


def test_the_template_is_filled_from_the_row_and_percent_encoded():
    row = {"kind": "network", "url_host": "192.168.1.50", "username": "rig",
           "snapshot_url": REOLINK_SNAP}
    url = cameras.resolve_snapshot_url(row, "p&ss#1")
    assert url == ("https://192.168.1.50/cgi-bin/api.cgi?cmd=Snap&channel=0&user=rig"
                   "&password=p%26ss%231")
    assert "p%26ss" not in cameras.safe_src(url)


def test_no_template_means_snapshots_are_off():
    assert cameras.resolve_snapshot_url({"kind": "network", "url_host": "h"}, "pw") is None


@pytest.mark.parametrize("bad, why", [
    ("ftp://{host}/snap", "http"),
    ("http://rig:pw@{host}/snap", "'@'"),
    ("http://{host}/snap?password=hunter2", "literal"),
    ("http://{host}/snap?token=abc", "literal"),
    ("http://{host}/snap?x={secret}", "placeholders"),
])
def test_a_snapshot_url_that_could_carry_a_secret_is_refused(bad, why):
    with pytest.raises(ValueError, match=why):
        db.clean_snapshot_url(bad)


def test_the_reolink_template_is_accepted_and_blank_means_off():
    assert db.clean_snapshot_url(REOLINK_SNAP) == REOLINK_SNAP
    assert db.clean_snapshot_url("  ") is None


def test_a_camera_round_trips_its_snapshot_url_and_the_spec_resolves_it(conn):
    db.add_camera(conn, source="yard_ir", kind="network", url_host="192.168.1.50",
                  url_port=554, url_path="h264Preview_01_sub", username="rig",
                  password="hunter2", snapshot_url=REOLINK_SNAP)
    row = db.list_cameras(conn)[0]
    assert row["snapshot_url"] == REOLINK_SNAP
    spec = cameras.row_to_spec(row, db.camera_password(conn, "yard_ir"))
    assert spec.snapshot_url.endswith("user=rig&password=hunter2")


def test_a_local_camera_never_has_a_snapshot_url(conn):
    db.add_camera(conn, source="door", kind="local", device_index=0, snapshot_url=REOLINK_SNAP)
    assert db.list_cameras(conn)[0]["snapshot_url"] is None


# ---- Reolink token login ---------------------------------------------------------------

TOKEN_SNAP = "https://{host}/cgi-bin/api.cgi?cmd=Snap&channel=0&token={token}"


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class _Camera:
    """A Reolink's login and snapshot endpoints: hands out numbered tokens, and returns a still
    only for the token it currently honours."""

    def __init__(self, lease=3600, refuse=False):
        self.lease, self.refuse = lease, refuse
        self.logins, self.fetched, self.honoured = [], [], None

    def post(self, url, payload, timeout):
        self.logins.append((url, payload))
        if self.refuse:
            return [{"cmd": "Login", "code": 1, "error": {"detail": "login failed",
                                                          "rspCode": -7}}]
        self.honoured = f"tok{len(self.logins)}"
        return [{"cmd": "Login", "code": 0,
                 "value": {"Token": {"leaseTime": self.lease, "name": self.honoured}}}]

    def fetch(self, url, timeout):
        self.fetched.append(url)
        return _still() if url.endswith(f"token={self.honoured}") else None


def _token(cam, clock=None):
    return snapshot.ReolinkToken("https://192.168.1.50/cgi-bin/api.cgi?cmd=Snap&channel=0"
                                 "&token={token}", ("rig", "hunter2"), post=cam.post,
                                 clock=clock or _Clock())


def test_the_password_goes_in_one_login_body_and_never_into_a_snapshot_url():
    cam = _Camera()
    tok = _token(cam)
    assert tok.fetch(cam.fetch, 5) is not None
    assert tok.fetch(cam.fetch, 5) is not None
    assert len(cam.logins) == 1                                   # the token is reused
    url, body = cam.logins[0]
    assert url == "https://192.168.1.50/cgi-bin/api.cgi?cmd=Login"
    assert body[0]["param"]["User"] == {"Version": "0", "userName": "rig", "password": "hunter2"}
    assert all("hunter2" not in u and u.endswith("token=tok1") for u in cam.fetched)


def test_the_token_is_renewed_before_its_lease_runs_out():
    cam, clock = _Camera(lease=3600), _Clock()
    tok = _token(cam, clock)
    tok.fetch(cam.fetch, 5)
    clock.t += 3600 - snapshot.ReolinkToken.RENEW_EARLY_S - 1
    tok.fetch(cam.fetch, 5)
    assert len(cam.logins) == 1
    clock.t += 2
    tok.fetch(cam.fetch, 5)
    assert len(cam.logins) == 2


def test_a_dropped_token_is_replaced_once_and_the_still_still_arrives():
    cam, clock = _Camera(), _Clock()
    tok = _token(cam, clock)
    tok.fetch(cam.fetch, 5)
    clock.t += 600
    cam.honoured = "rebooted"                       # the camera forgot tok1
    assert tok.fetch(cam.fetch, 5) is not None
    assert len(cam.logins) == 2


def test_a_camera_failing_right_after_login_does_not_get_a_login_per_detection():
    """Every login opens a camera session; a camera that keeps answering with something other
    than a still must not have its session table filled one detection at a time."""
    cam, clock = _Camera(), _Clock()
    tok = _token(cam, clock)
    broken = lambda url, timeout: None
    for _ in range(5):
        assert tok.fetch(broken, 5) is None
        clock.t += 3
    assert len(cam.logins) == 1


def test_a_refused_login_backs_off_and_never_echoes_the_password():
    cam, clock = _Camera(refuse=True), _Clock()
    tok = _token(cam, clock)
    with pytest.raises(RuntimeError) as e:
        tok.fetch(cam.fetch, 5)
    assert "hunter2" not in str(e.value) and "login failed" in str(e.value)
    for _ in range(10):                              # a detection every 3 s for half a minute
        clock.t += 3
        with pytest.raises(RuntimeError, match="backing off"):
            tok.fetch(cam.fetch, 5)
    assert len(cam.logins) == 1                      # no lockout-by-retry
    clock.t += snapshot.ReolinkToken.LOGIN_BACKOFF_S
    cam.refuse = False
    assert tok.fetch(cam.fetch, 5) is not None


def test_a_token_camera_spec_carries_the_login_apart_from_the_url(conn):
    db.add_camera(conn, source="yard_ir", kind="network", url_host="192.168.1.50",
                  url_port=554, url_path="h264Preview_01_sub", username="rig",
                  password="hunter2", snapshot_url=TOKEN_SNAP)
    spec = cameras.row_to_spec(db.list_cameras(conn)[0], db.camera_password(conn, "yard_ir"))
    assert spec.snapshot_url == "https://192.168.1.50/cgi-bin/api.cgi?cmd=Snap&channel=0&token={token}"
    assert spec.snapshot_login == ("rig", "hunter2")
    assert "snapshot_login" not in repr(spec)


def test_the_cropper_uses_the_token_login_when_the_spec_asks_for_it(tmp_path):
    cfg = _cfg(tmp_path)
    det_id, det, rel = _detection_row(cfg)
    cam = _Camera()
    spec = config.CameraSpec("yard_ir", "rtsp://h/x",
                             snapshot_url="https://192.168.1.50/cgi-bin/api.cgi?cmd=Snap"
                                          "&channel=0&token={token}",
                             snapshot_login=("rig", "hunter2"))
    c = snapshot.SnapshotCropper(spec, cfg, lambda img: [_animal((1206, 606, 1806, 1206))],
                                 threading.Lock(), fetch=cam.fetch, post=cam.post)
    assert c.process(snapshot.Job([(det_id, det, rel)], STREAM_W, STREAM_H)) == 1
    assert len(cam.logins) == 1


def test_a_token_placeholder_is_accepted_and_a_literal_token_is_not():
    assert db.clean_snapshot_url(TOKEN_SNAP) == TOKEN_SNAP
    with pytest.raises(ValueError, match="literal"):
        db.clean_snapshot_url("https://{host}/cgi-bin/api.cgi?cmd=Snap&token=abc123")
