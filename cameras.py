"""The camera list: the bridge between the `cameras` DB table and the CameraSpec the rig runs.

Cameras used to live only in config_local.py, which meant adding one was a Python edit. Since
2026-08-22 the DB table is the runtime source of truth and config only SEEDS it, exactly as
ignore_zones did before -- see the schema comment in db.py, which is the authority on the table.

WHY THIS IS ITS OWN MODULE, and not part of web.py or backyard_cam.py: `backyard_cam` imports
`web`, so anything `web` imports must not reach back into `backyard_cam`. safe_src() lived in
backyard_cam and is needed by web.py, tools/camprobe.py and here -- importing it from there would
close the cycle backyard_cam -> web -> camprobe -> backyard_cam, and because safe_src is defined
hundreds of lines AFTER `import web`, the partially-initialised module would not even have the
attribute yet. The rig would die at startup with an ImportError. This module imports only config
and db, so nothing can form a cycle through it.

CAMERA CHANGES DO NOT APPLY LIVE. The rig reads this list once, at startup, and gives each camera
its own capture thread; ignore_zones can be edited live because a zone is just data a running
thread re-reads, while a camera is a thread plus a frame buffer, a clip recorder and a motion
gate. Everything here is therefore written for "saved now, running after the next restart", and
the dashboard says so rather than pretending otherwise.
"""
from __future__ import annotations

import re
from urllib.parse import quote

import config
import db

# Secret-looking query parameters (an ESP32-CAM or a DVR may carry the login there instead of in
# the netloc): ?pwd=... / &token=... . The name is kept, the value is not.
_URL_SECRET_QS = re.compile(r"(?i)(?<=[?&])(pass(?:word)?|pwd|token|auth|secret|key)=[^&]*")

# Characters that must be percent-encoded inside a URL's userinfo, or the URL reparses wrongly.
_USERINFO_NEEDS_ENCODING = set("@:/?#[] ")


def safe_src(src) -> str:
    """A camera `src` rendered for a HUMAN to read, with any credentials masked.

    A networked camera's src carries its password in the URL (rtsp://user:pass@host/...), and
    every line that prints one goes to the console AND to logs/backyard_cam.log -- which
    backup.py sweeps into the meta zip and off the machine, and which is what gets pasted into
    an issue when something breaks. So each human-facing print of a src goes through here; the
    real URL is handed to VideoCapture untouched.

    The USERNAME survives -- it is exactly what you need to see when a camera rejects a login --
    and the password does not. A plain int webcam index has nothing to hide and comes back
    unchanged, so the single-camera case reads as it always did.

    Returns the repr (quoted for a URL, bare for an index), because it replaces `!r` at the
    call sites."""
    if not isinstance(src, str):
        return repr(src)
    out = src
    scheme, sep, rest = out.partition("://")
    if sep and "@" in rest:
        # rpartition, not partition: a password may itself contain '@', and it is the LAST one
        # that delimits the host.
        creds, _, host = rest.rpartition("@")
        user, colon, _pw = creds.partition(":")
        out = f"{scheme}://{user}{':***' if colon else ''}@{host}"
    return repr(_URL_SECRET_QS.sub(r"\1=***", out))


def parse_stream_url(url: str, schemes=("rtsp", "http", "https")):
    """Split scheme://[user[:pass]@]host[:port]/path into a dict, or None if it isn't one.

    rpartition on the '@', because a password may legally contain one and it is the LAST '@'
    that delimits the host. Splitting on the first would hand back a truncated password and a
    nonsense hostname -- and that failure looks exactly like a wrong password, which is the most
    expensive thing it could possibly be mistaken for."""
    if not isinstance(url, str):
        return None
    scheme, sep, rest = url.partition("://")
    scheme = scheme.lower()
    if not sep or scheme not in schemes:
        return None
    # Authority FIRST, then credentials inside it. Looking for '@' across the whole remainder
    # instead would let an '@' in the PATH masquerade as the credential delimiter: for
    # rtsp://cam/live@2x the host would come back as '2x' and 'cam/live' would be read as a
    # username -- a URL that opens fine in VLC, stored as a row pointing somewhere else entirely.
    # RFC 3986 ends the authority at the first '/', which is also why _userinfo percent-encodes a
    # '/' in a password rather than leaving it to be guessed at here.
    authority, _, path = rest.partition("/")
    user = pw = ""
    if "@" in authority:
        creds, _, authority = authority.rpartition("@")
        user, _, pw = creds.partition(":")
    host, _, port = authority.partition(":")
    if not host:
        return None
    try:
        port_n = int(port) if port else None
    except ValueError:
        return None
    return {"url_scheme": scheme, "url_host": host, "url_port": port_n,
            "url_path": path or None, "username": user or None, "password": pw or None}


def _userinfo(username, password) -> str:
    """The 'user:pass@' part of a stream URL, percent-encoding only when it would otherwise
    reparse wrongly. Encoding unconditionally would rewrite every ordinary password (a '!'
    becomes %21) for no benefit and a small risk that some camera's parser disagrees; encoding
    never would break any password containing '@' or ':'. So: encode the ones that need it."""
    if not username:
        return ""
    def enc(v):
        return quote(v, safe="") if any(c in _USERINFO_NEEDS_ENCODING for c in v) else v
    out = enc(str(username))
    if password:
        out += ":" + enc(str(password))
    return out + "@"


def build_src(row: dict, password=None):
    """The value handed to cv2.VideoCapture for this camera row: an int index for a local
    camera, or a fully assembled URL (credentials included) for a networked one."""
    if row.get("kind") == "local":
        return int(row.get("device_index") or 0)
    scheme = row.get("url_scheme") or "rtsp"
    host = row.get("url_host") or ""
    port = row.get("url_port")
    path = (row.get("url_path") or "").lstrip("/")
    hostport = f"{host}:{port}" if port else host
    return f"{scheme}://{_userinfo(row.get('username'), password)}{hostport}/{path}"


def spec_to_row(spec) -> dict:
    """A CameraSpec decomposed into `cameras` columns, for seeding the table from config.

    A URL src is split so the stored row carries no credential in its URL columns -- username and
    password get their own, and everything that renders a camera can be handed the credential-free
    pieces."""
    row = {"source": spec.source, "name": spec.name,
           "frame_width": spec.frame_width, "frame_height": spec.frame_height,
           "motion_min_area": spec.motion_min_area, "record_clips": spec.record_clips,
           "kind": "local", "device_index": None, "url_scheme": None, "url_host": None,
           "url_port": None, "url_path": None, "username": None, "password": None}
    if isinstance(spec.src, str):
        parts = parse_stream_url(spec.src)
        if parts is not None:
            row.update(parts)
            row["kind"] = "network"
        else:
            # A local FILE PATH src (the README's canned-video demo) is a legal spec but has no
            # row shape here; keep it out of the table rather than mangling it into a URL.
            return {}
    else:
        row["device_index"] = int(spec.src)
    return row


def row_to_spec(row: dict, password=None) -> config.CameraSpec:
    """One DB row as the CameraSpec the capture loop takes. Every override left NULL stays None,
    which is what makes it inherit the matching Config value at runtime -- the same contract a
    hand-written spec in config_local.py has."""
    return config.CameraSpec(
        source=row["source"], src=build_src(row, password), name=row.get("name"),
        frame_width=row.get("frame_width"), frame_height=row.get("frame_height"),
        motion_min_area=row.get("motion_min_area"), record_clips=row.get("record_clips"))


def load_specs(cfg, conn) -> tuple[list, list]:
    """The cameras the rig should run, plus human-readable notes for the startup banner.

    Seeds the table from cfg.camera_specs() (once per source, ever), then returns what the table
    says. Notes call out the two things that surprise people:

      * a config-listed camera whose stored row now DIFFERS from config -- because after the
        first seed the table owns it and editing config_local.py has no effect;
      * the safety net below.

    THE SAFETY NET: if the table ends up with no live, enabled camera -- every one deleted or
    disabled -- this falls back to the config list rather than returning nothing. A rig with zero
    cameras exits as soon as its last capture thread ends, so without this a couple of clicks in
    the dashboard could leave a box that will not start and no obvious way back. web.py refuses
    to remove or disable the last one; this is the second line of defence, and it is loud."""
    notes: list[str] = []
    cfg_specs = list(cfg.camera_specs())

    # Duplicate sources have to be caught HERE, on the config list, because seed_cameras dedupes
    # by source: two config cameras sharing a name would quietly seed the first and drop the
    # second, and the rig would run one camera where the operator wrote two. The startup guard in
    # backyard_cam.run() sees only the deduped DB list and can no longer catch it.
    seen: set = set()
    for spec in cfg_specs:
        if spec.source in seen:
            raise RuntimeError(f"Duplicate camera source '{spec.source}' in cfg.cameras -- "
                               f"give each camera a unique source name.")
        seen.add(spec.source)

    # A spec the table cannot represent (a local FILE PATH src -- the README's canned-video demo)
    # is not a database camera and never will be, so it is carried through unchanged rather than
    # dropped. Without this, moving the list into the DB would silently stop running any camera
    # the row schema has no shape for, and the only symptom would be a missing pane.
    unstorable = [s for s in cfg_specs if not spec_to_row(s)]
    seed_rows = [r for r in (spec_to_row(s) for s in cfg_specs) if r]
    added = db.seed_cameras(conn, seed_rows)
    if added:
        notes.append(f"camera list: seeded {added} camera(s) from config into the database; "
                     f"the dashboard owns them from now on (editing them in config_local.py, "
                     f"or passing --camera-index/--width/--height, no longer changes anything)")

    rows = db.list_cameras(conn, include_disabled=False)
    if not rows:
        # THE SAFETY NET, and it must not undo a deliberate deletion. Falling back to the raw
        # config list would resurrect every camera someone removed in the dashboard -- the exact
        # thing the tombstone exists to prevent -- so only cameras that were never deleted are
        # eligible. web.py refuses to remove or disable the last one; this is the second line.
        deleted = {r[0] for r in conn.execute(
            "SELECT source FROM cameras WHERE deleted_at IS NOT NULL")}
        revived = [s for s in cfg_specs if s.source not in deleted]
        if revived:
            notes.append("camera list: WARNING -- no enabled cameras in the database. Falling "
                         f"back to {len(revived)} camera(s) from config so the rig keeps "
                         "watching; re-enable one in the dashboard to stop this.")
            return revived, notes
        # Nothing survives even after honouring the tombstones. Running deleted cameras is wrong,
        # and running none is worse -- the rig would exit at startup with no camera thread alive,
        # from a box whose whole job is to be watching. So it runs them and says exactly that,
        # rather than quietly undoing a deletion or quietly refusing to start.
        notes.append("camera list: WARNING -- every camera has been deleted or disabled. Running "
                     "the config list anyway, INCLUDING cameras removed in the dashboard, "
                     "because a rig with no camera stops at startup. Add one in the dashboard.")
        return cfg_specs, notes

    specs = [row_to_spec(r, db.camera_password(conn, r["source"])) for r in rows]

    by_source = {s.source: s for s in cfg_specs}
    for spec in specs:
        other = by_source.get(spec.source)
        if other is not None and str(other.src) != str(spec.src):
            notes.append(f"camera list: {spec.source} differs from config_local.py "
                         f"(running {safe_src(spec.src)}); the database wins -- edit it in the "
                         f"dashboard, not in config")
    for spec in unstorable:
        notes.append(f"camera list: {spec.source} runs from config ({safe_src(spec.src)}) -- a "
                     f"file source is not a database camera and cannot be edited in the dashboard")
    return specs + unstorable, notes
