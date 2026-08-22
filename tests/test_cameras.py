"""
Tests for cameras.py and the `cameras` DB table -- the camera list moving out of config_local.py
and into something the dashboard can edit (2026-08-22).

Three things here are load-bearing and everything else is detail:

  * THE PASSWORD MUST NOT COME BACK OUT. cameras.password is the only secret this database holds.
    It leaves by exactly one function, db.camera_password(), which the rig calls at startup to
    build a capture URL. Every listing, every row dict, every API payload downstream of them must
    carry has_password and never the value. Several tests here exist only to assert that absence.

  * SOURCE IS WRITE-ONCE AND TOMBSTONES ARE FOREVER. source is the partition key of detections,
    visits, coverage_events, ignore_zones, view_epochs and the clips/<source>/ directory. Deleting
    a camera must not free the name for a different camera, and re-adding it must reattach to the
    same row -- otherwise a year of recorded rows quietly belongs to nobody.

  * CONFIG ONLY SEEDS. Once a source exists in the table, editing config_local.py must stop having
    any effect, or a restart would silently overwrite what someone set in the dashboard. The
    startup notes are what makes that discoverable instead of baffling.

The masking tests (safe_src) moved here from test_backyard_cam.py when the function moved: web.py
and tools/camprobe.py both need it, and importing it from backyard_cam would close the import
cycle backyard_cam -> web -> camprobe -> backyard_cam and kill the rig at startup.
"""
from __future__ import annotations

import config
import db
import pytest

import cameras


# ---- safe_src: a camera password must never reach the log ---------------------------

REOLINK = "rtsp://rig:hunter2@192.168.0.105:554/h264Preview_01_sub"


def test_int_index_is_unchanged():
    """The single-camera case has nothing to hide and must read exactly as it always did."""
    assert cameras.safe_src(1) == "1"
    assert cameras.safe_src(0) == "0"


def test_rtsp_password_is_masked_and_everything_else_survives():
    out = cameras.safe_src(REOLINK)
    assert "hunter2" not in out
    assert out == "'rtsp://rig:***@192.168.0.105:554/h264Preview_01_sub'"


def test_username_survives_because_a_rejected_login_is_the_thing_you_debug():
    assert "rig" in cameras.safe_src(REOLINK)


def test_password_containing_an_at_sign_is_still_fully_masked():
    out = cameras.safe_src("rtsp://rig:p@ss@192.168.0.105:554/stream")
    assert "p@ss" not in out
    assert out == "'rtsp://rig:***@192.168.0.105:554/stream'"


def test_url_without_credentials_is_untouched():
    url = "http://192.168.1.51:81/stream"
    assert cameras.safe_src(url) == repr(url)


def test_username_only_url_keeps_its_shape():
    assert cameras.safe_src("rtsp://rig@host/stream") == "'rtsp://rig@host/stream'"


def test_local_file_path_source_is_untouched():
    path = r"C:/downloads/raccoon_visit.mp4"
    assert cameras.safe_src(path) == repr(path)


def test_secret_query_parameter_is_masked_by_name():
    out = cameras.safe_src("http://cam.local/stream?user=rig&pwd=hunter2&res=hd")
    assert "hunter2" not in out
    assert "pwd=***" in out and "res=hd" in out


# ---- parse_stream_url / build_src ----------------------------------------------------

def test_url_splits_into_columns():
    assert cameras.parse_stream_url(REOLINK) == {
        "url_scheme": "rtsp", "url_host": "192.168.0.105", "url_port": 554,
        "url_path": "h264Preview_01_sub", "username": "rig", "password": "hunter2"}


def test_password_with_an_at_sign_splits_on_the_last_one():
    got = cameras.parse_stream_url("rtsp://rig:p@ssw0rd@10.0.0.5:554/stream")
    assert got["password"] == "p@ssw0rd" and got["url_host"] == "10.0.0.5"


def test_missing_port_is_none_not_a_guess():
    """The DB stores NULL and build_src omits it, letting the scheme's default apply -- rather
    than baking 554 into a row for an http camera."""
    assert cameras.parse_stream_url("rtsp://cam.local/stream1")["url_port"] is None


@pytest.mark.parametrize("bad", ["http://h/s", "ftp://h/s", "rtsp://", "not a url", 5, None])
def test_unusable_urls_return_none(bad):
    assert cameras.parse_stream_url(bad, schemes=("rtsp",)) is None


def test_build_src_local_is_the_bare_index():
    assert cameras.build_src({"kind": "local", "device_index": 1}) == 1


def test_build_src_round_trips_a_url():
    row = dict(cameras.parse_stream_url(REOLINK), kind="network")
    assert cameras.build_src(row, "hunter2") == REOLINK


def test_build_src_omits_credentials_when_there_is_no_user():
    row = {"kind": "network", "url_scheme": "rtsp", "url_host": "cam", "url_port": None,
           "url_path": "s", "username": None}
    assert cameras.build_src(row, "ignored") == "rtsp://cam/s"


def test_a_password_needing_encoding_is_encoded_and_a_plain_one_is_not():
    """Encoding everything would rewrite an ordinary '!' as %21 for no benefit; encoding nothing
    would break any password containing '@' or ':'. So it encodes exactly what must be."""
    row = {"kind": "network", "url_scheme": "rtsp", "url_host": "cam", "url_port": None,
           "url_path": "s", "username": "rig"}
    assert cameras.build_src(row, "plain!") == "rtsp://rig:plain!@cam/s"
    assert cameras.build_src(row, "p@ss") == "rtsp://rig:p%40ss@cam/s"


# ---- spec <-> row --------------------------------------------------------------------

def test_local_spec_round_trips():
    spec = config.CameraSpec("glass_door_cam", 1, name="Glass door", frame_width=1920,
                             frame_height=1080, motion_min_area=1800)
    back = cameras.row_to_spec(cameras.spec_to_row(spec))
    assert (back.source, back.src, back.name) == ("glass_door_cam", 1, "Glass door")
    assert (back.frame_width, back.frame_height, back.motion_min_area) == (1920, 1080, 1800)


def test_network_spec_round_trips_including_the_password():
    spec = config.CameraSpec("yard_ir", REOLINK, name="Yard", motion_min_area=200)
    row = cameras.spec_to_row(spec)
    assert row["kind"] == "network" and row["password"] == "hunter2"
    assert cameras.row_to_spec(row, row["password"]).src == REOLINK


def test_overrides_left_unset_stay_none_so_they_inherit_config():
    """None is not a missing value here -- it is the instruction 'use the Config default', and a
    row that turned it into 0 or 1920 would silently pin a camera to the wrong settings."""
    row = cameras.spec_to_row(config.CameraSpec("c", 0))
    back = cameras.row_to_spec(row)
    assert back.frame_width is None and back.motion_min_area is None
    assert back.record_clips is None


def test_a_file_path_src_is_not_seeded():
    """A local video file is a legal src (the canned-video demo) but has no row shape; it must be
    skipped, not mangled into a URL with an empty host."""
    assert cameras.spec_to_row(config.CameraSpec("demo", "C:/clips/raccoon.mp4")) == {}


# ---- the table: CRUD, tombstones, and the password ------------------------------------

def _net(conn, source="yard_ir", **kw):
    kw.setdefault("url_host", "192.168.0.105")
    kw.setdefault("url_port", 554)
    kw.setdefault("url_path", "h264Preview_01_sub")
    kw.setdefault("username", "rig")
    return db.add_camera(conn, source=source, kind="network", **kw)


def test_add_then_list(conn):
    _net(conn, name="Yard")
    rows = db.list_cameras(conn)
    assert [r["source"] for r in rows] == ["yard_ir"]
    assert rows[0]["name"] == "Yard" and rows[0]["created_by"] == "web"


def test_the_password_is_never_in_a_listing(conn):
    _net(conn, password="hunter2")
    row = db.list_cameras(conn)[0]
    assert "password" not in row
    assert row["has_password"] is True
    assert "hunter2" not in repr(row)


def test_has_password_is_false_when_none_is_set(conn):
    _net(conn)
    assert db.list_cameras(conn)[0]["has_password"] is False


def test_camera_password_is_the_one_read_path(conn):
    _net(conn, password="hunter2")
    assert db.camera_password(conn, "yard_ir") == "hunter2"
    assert db.camera_password(conn, "nope") is None


def test_a_live_duplicate_source_is_refused(conn):
    _net(conn)
    with pytest.raises(ValueError, match="already exists"):
        _net(conn)


def test_removing_tombstones_rather_than_deleting(conn):
    cam = _net(conn)
    assert db.remove_camera(conn, cam["id"])["source"] == "yard_ir"
    assert db.list_cameras(conn) == []
    still_there = conn.execute("SELECT deleted_at FROM cameras WHERE id = ?",
                               (cam["id"],)).fetchone()
    assert still_there[0] is not None


def test_removing_twice_returns_none_rather_than_pretending(conn):
    cam = _net(conn)
    db.remove_camera(conn, cam["id"])
    assert db.remove_camera(conn, cam["id"]) is None


def test_re_adding_a_removed_source_undeletes_the_same_row(conn):
    """Not a second row: the detections, visits and clips/<source>/ folder recorded under that
    name still exist and still belong to it."""
    cam = _net(conn, password="hunter2")
    db.remove_camera(conn, cam["id"])
    again = _net(conn, name="Yard again")
    assert again["id"] == cam["id"]
    assert again["name"] == "Yard again"
    assert db.camera_password(conn, "yard_ir") == "hunter2"   # not re-typed, not lost


def test_update_without_a_password_leaves_the_stored_one_alone(conn):
    """The edit form can never show the operator the stored password, so submitting the form
    must not be able to wipe it."""
    cam = _net(conn, password="hunter2")
    db.update_camera(conn, cam["id"], kind="network", url_host="192.168.0.199",
                     url_path="h264Preview_01_main", username="rig")
    assert db.camera_password(conn, "yard_ir") == "hunter2"
    assert db.list_cameras(conn)[0]["url_host"] == "192.168.0.199"


def test_an_empty_password_string_also_leaves_it_alone(conn):
    cam = _net(conn, password="hunter2")
    db.update_camera(conn, cam["id"], kind="network", url_host="h", username="rig", password="")
    assert db.camera_password(conn, "yard_ir") == "hunter2"


def test_clearing_a_password_takes_an_explicit_flag(conn):
    cam = _net(conn, password="hunter2")
    db.update_camera(conn, cam["id"], kind="network", url_host="h", username="rig",
                     clear_password=True)
    assert db.camera_password(conn, "yard_ir") is None


def test_update_of_an_unknown_id_is_none_not_a_crash(conn):
    assert db.update_camera(conn, 999, kind="local", device_index=0) is None


# ---- validation ----------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "  ", "front yard", "yard/ir", "../etc", "_leading",
                                 "x" * 41, "yard\nir"])
def test_bad_sources_are_refused(conn, bad):
    """source becomes a directory name under clips/, so anything that is not a plain word is
    refused at the door rather than silently rewritten by clips._safe_source later."""
    with pytest.raises(ValueError):
        db.add_camera(conn, source=bad, kind="local", device_index=0)


def test_a_local_camera_needs_an_index(conn):
    with pytest.raises(ValueError, match="device index"):
        db.add_camera(conn, source="usb", kind="local")


def test_a_network_camera_needs_a_host(conn):
    with pytest.raises(ValueError, match="host"):
        db.add_camera(conn, source="net", kind="network")


def test_a_host_that_is_really_a_url_is_refused(conn):
    """Pasting the whole rtsp:// URL into the host box is the obvious mistake; catching it here
    beats assembling rtsp://rtsp://user:pass@host/path and reporting 'could not open'."""
    with pytest.raises(ValueError, match="just the address"):
        db.add_camera(conn, source="net", kind="network",
                      url_host="rtsp://rig:pw@192.168.0.105:554/x")


def test_control_characters_are_refused(conn):
    with pytest.raises(ValueError, match="control characters"):
        db.add_camera(conn, source="net", kind="network", url_host="cam\r\nEvil: header")


def test_an_unknown_kind_is_refused(conn):
    with pytest.raises(ValueError, match="kind"):
        db.add_camera(conn, source="x", kind="carrier-pigeon")


def test_a_bad_port_is_refused(conn):
    with pytest.raises(ValueError, match="port"):
        _net(conn, url_port=99999)


# ---- seeding from config --------------------------------------------------------------

def _cfg(specs):
    cfg = config.Config()
    cfg.cameras = list(specs)
    return cfg


def test_seed_creates_rows_marked_as_config(conn):
    cfg = _cfg([config.CameraSpec("glass_door_cam", 1, name="Glass door")])
    specs, notes = cameras.load_specs(cfg, conn)
    assert [s.source for s in specs] == ["glass_door_cam"]
    assert db.list_cameras(conn)[0]["created_by"] == "config"
    assert any("seeded" in n for n in notes)


def test_seeding_is_once_per_source_ever(conn):
    cfg = _cfg([config.CameraSpec("glass_door_cam", 1)])
    cameras.load_specs(cfg, conn)
    cameras.load_specs(cfg, conn)
    assert len(db.list_cameras(conn)) == 1


def test_a_camera_deleted_in_the_ui_stays_deleted_across_restarts(conn):
    """The whole point of the tombstone: config_local.py still lists it, and it must not come
    back on the next start."""
    cfg = _cfg([config.CameraSpec("glass_door_cam", 1), config.CameraSpec("yard_ir", REOLINK)])
    cameras.load_specs(cfg, conn)
    yard = [c for c in db.list_cameras(conn) if c["source"] == "yard_ir"][0]
    db.remove_camera(conn, yard["id"])
    specs, _ = cameras.load_specs(cfg, conn)
    assert [s.source for s in specs] == ["glass_door_cam"]


def test_the_database_wins_over_config_and_says_so(conn):
    """Editing config_local.py after the first seed has no effect. That is the bargain, and an
    unexplained one would be baffling -- so it is called out in the startup notes."""
    cfg = _cfg([config.CameraSpec("yard_ir", REOLINK)])
    cameras.load_specs(cfg, conn)
    cam = db.list_cameras(conn)[0]
    db.update_camera(conn, cam["id"], kind="network", url_host="192.168.0.199",
                     url_port=554, url_path="h264Preview_01_main", username="rig")
    specs, notes = cameras.load_specs(cfg, conn)
    assert "192.168.0.199" in str(specs[0].src)
    assert any("differs from config_local.py" in n for n in notes)


def test_that_note_never_contains_the_password(conn):
    cfg = _cfg([config.CameraSpec("yard_ir", REOLINK)])
    cameras.load_specs(cfg, conn)
    cam = db.list_cameras(conn)[0]
    db.update_camera(conn, cam["id"], kind="network", url_host="192.168.0.199",
                     username="rig", password="hunter2")
    _, notes = cameras.load_specs(cfg, conn)
    assert notes and not any("hunter2" in n for n in notes)


def test_a_disabled_camera_is_not_run(conn):
    cfg = _cfg([config.CameraSpec("glass_door_cam", 1), config.CameraSpec("yard_ir", REOLINK)])
    cameras.load_specs(cfg, conn)
    yard = [c for c in db.list_cameras(conn) if c["source"] == "yard_ir"][0]
    db.update_camera(conn, yard["id"], kind="network", url_host="192.168.0.105",
                     username="rig", enabled=False)
    specs, _ = cameras.load_specs(cfg, conn)
    assert [s.source for s in specs] == ["glass_door_cam"]


def test_the_fallback_does_not_resurrect_a_deleted_camera(conn):
    """The safety net must not undo a deliberate deletion. With one camera deleted and the other
    merely disabled, the fallback runs the disabled one back -- and leaves the deleted one gone."""
    cfg = _cfg([config.CameraSpec("glass_door_cam", 1), config.CameraSpec("yard_ir", REOLINK)])
    cameras.load_specs(cfg, conn)
    by_src = {c["source"]: c for c in db.list_cameras(conn)}
    db.remove_camera(conn, by_src["yard_ir"]["id"])
    db.update_camera(conn, by_src["glass_door_cam"]["id"], kind="local", device_index=1,
                     enabled=False)
    specs, notes = cameras.load_specs(cfg, conn)
    assert [s.source for s in specs] == ["glass_door_cam"]
    assert any("WARNING" in n and "Falling back" in n for n in notes)


def test_with_everything_deleted_it_runs_config_and_says_so(conn):
    """A rig with no cameras exits as soon as its last capture thread ends. Rather than refuse to
    start OR quietly undo the deletions, it does the visible thing and announces it."""
    cfg = _cfg([config.CameraSpec("glass_door_cam", 1)])
    cameras.load_specs(cfg, conn)
    for cam in db.list_cameras(conn):
        db.remove_camera(conn, cam["id"])
    specs, notes = cameras.load_specs(cfg, conn)
    assert [s.source for s in specs] == ["glass_door_cam"]
    assert any("WARNING" in n and "INCLUDING cameras removed" in n for n in notes)


def test_the_single_camera_rig_seeds_its_synthesized_spec(conn):
    """A config that never set cfg.cameras still has exactly one camera -- camera_specs()
    synthesizes it from the flat fields -- and it must land in the table like any other."""
    cfg = config.Config()
    specs, _ = cameras.load_specs(cfg, conn)
    assert len(specs) == 1
    assert db.list_cameras(conn)[0]["source"] == cfg.source


# ---- fixes from the adversarial review, 2026-08-22 ------------------------------------

def test_an_at_sign_in_the_stream_path_does_not_become_a_credential():
    """rtsp://cam/live@2x is a valid URL whose host is 'cam'. Searching the whole remainder for
    '@' instead of just the authority read the host as '2x' and 'cam/live' as a username -- a row
    silently pointing somewhere else entirely."""
    got = cameras.parse_stream_url("rtsp://cam.local/live@2x")
    assert got["url_host"] == "cam.local" and got["url_path"] == "live@2x"
    assert got["username"] is None and got["password"] is None


def test_credentials_still_parse_when_the_path_also_has_an_at_sign():
    got = cameras.parse_stream_url("rtsp://rig:pw@cam.local:554/live@2x")
    assert (got["username"], got["password"]) == ("rig", "pw")
    assert got["url_host"] == "cam.local" and got["url_path"] == "live@2x"


def test_a_hyphen_is_refused_because_the_clips_folder_would_collide(conn):
    """clips._safe_source maps every non-alphanumeric character to '_', so 'yard-ir' and
    'yard_ir' would be two sources sharing one clips/ directory -- and one prune budget."""
    import clips
    assert clips._safe_source("yard-ir") == clips._safe_source("yard_ir")
    with pytest.raises(ValueError, match="letters, digits and underscore"):
        db.add_camera(conn, source="yard-ir", kind="local", device_index=0)


def test_a_password_without_a_username_is_refused(conn):
    """_userinfo drops it when assembling the URL, so it would be stored, reported as
    'password set', and never sent -- a login failure with no visible cause."""
    with pytest.raises(ValueError, match="username"):
        _net(conn, username=None, password="hunter2")


def test_clearing_the_username_off_a_camera_that_has_a_password_is_refused(conn):
    cam = _net(conn, password="hunter2")
    with pytest.raises(ValueError, match="username"):
        db.update_camera(conn, cam["id"], kind="network", url_host="192.168.0.105",
                         username=None)


def test_a_login_in_the_stream_path_is_refused(conn):
    for bad in ("stream?pwd=hunter2", "rig:hunter2@10.0.0.9/stream", "s?token=abc"):
        with pytest.raises(ValueError):
            _net(conn, source="probe", url_path=bad)


def test_undelete_restores_every_column_not_just_the_ones_we_remembered(conn):
    """A stale value surviving from the tombstoned row would be invisible until the camera
    behaved oddly, so this checks the whole row rather than a sample of it."""
    cam = _net(conn, name="Old", url_port=554, url_path="old_path", frame_width=640,
               frame_height=360, motion_min_area=200, record_clips=False)
    db.remove_camera(conn, cam["id"])
    again = db.add_camera(conn, source="yard_ir", kind="network", name="New",
                          url_host="10.0.0.9", url_port=8554, url_path="new_path",
                          username="rig2", frame_width=1920, frame_height=1080,
                          motion_min_area=1800, record_clips=True)
    assert again["id"] == cam["id"]
    for field, want in [("name", "New"), ("url_host", "10.0.0.9"), ("url_port", 8554),
                        ("url_path", "new_path"), ("username", "rig2"),
                        ("frame_width", 1920), ("frame_height", 1080),
                        ("motion_min_area", 1800), ("record_clips", True),
                        ("enabled", True)]:
        assert again[field] == want, f"{field} kept a stale value from the tombstone"


def test_seeding_carries_the_password_from_config(conn):
    """The seed is the ONLY way an existing rig's cameras arrive, so a seed that stored NULL
    would silently break every one of them at the first restart."""
    cfg = _cfg([config.CameraSpec("yard_ir", REOLINK)])
    cameras.load_specs(cfg, conn)
    assert db.camera_password(conn, "yard_ir") == "hunter2"
    assert db.list_cameras(conn)[0]["has_password"] is True


def test_a_file_path_camera_keeps_running_even_though_it_cannot_be_stored(conn):
    """The canned-video demo has no row shape. Dropping it from the running list would have been
    a silent regression whose only symptom is a missing pane."""
    cfg = _cfg([config.CameraSpec("glass_door_cam", 1),
                config.CameraSpec("demo", "C:/clips/raccoon.mp4")])
    specs, notes = cameras.load_specs(cfg, conn)
    assert sorted(s.source for s in specs) == ["demo", "glass_door_cam"]
    assert [c["source"] for c in db.list_cameras(conn)] == ["glass_door_cam"]
    assert any("cannot be edited in the dashboard" in n for n in notes)


def test_two_config_cameras_sharing_a_source_still_raise(conn):
    """seed_cameras dedupes by source, so without an explicit check the second camera would be
    silently dropped and the rig would run one where two were configured."""
    cfg = _cfg([config.CameraSpec("yard_ir", REOLINK),
                config.CameraSpec("yard_ir", "rtsp://rig:pw@10.0.0.9:554/x")])
    with pytest.raises(RuntimeError, match="Duplicate camera source"):
        cameras.load_specs(cfg, conn)
