"""
The Creature Report, delivered: a morning email of last night's yard activity.

The dashboard's Creature Report tab already writes the story of each completed sun-period --
stats.period_digest() knows the visits, the named individuals, the plate of the night, who was
oddly absent and whether the camera was even watching. This module is a THIN RENDERER over that
same payload: it asks for the most recently completed night, lays it out as a small newspaper
(same masthead the dashboard uses -- see MASTHEAD), and mails it. No new analysis
happens here; if the email and the dashboard ever disagree, the email is wrong.

Delivery is Resend's REST API (https://resend.com), one stdlib urllib POST -- no SDK, no new
dependency, mirroring the no-deps spine of the rest of the project. Photos are embedded as
inline CID attachments rather than linked or data-URI'd, because both alternatives actually
fail: the dashboard's media URLs are LAN-only (Gmail's image proxy can never reach them) and
Gmail strips data: URIs from HTML mail. A browser-viewable archive copy (same layout, data:
URIs, which browsers do render) is written under reports/mail/ on every run, so the issue is
never lost even when sending is unconfigured or down.

Hero image: the digest's plate is the SHARPEST frame, and whole-crop sharpness once led an
issue with crisply focused brick pavers wearing a motion-blurred raccoon. The email re-judges
the period for CUTENESS instead (pick_plate): how much animal is in frame (peaked -- a too-close
animal is a body part), whether the focus lives ON the animal rather than the background
(centre-vs-border ring), night eyeshine as a facing-the-camera proxy, and a nudge for animals
we know by name. The winner's own crop usually IS the hero (the score prefers close animals,
so the crop is large; at 588 css px it renders downscaled -- the "wall of blurry ovals" rule
is about upscaling, and it still holds: a small crop is never stretched, and frame sources are
only composed around the bbox as fallbacks).

Run it by hand, or schedule it (README "A morning email"):

    python newsletter.py                 # render + archive + send if configured (scheduled mode)
    python newsletter.py --no-send       # render + archive only
    python newsletter.py --send          # strict: fail loudly if email isn't configured
    python newsletter.py --date 2026-08-10 --edition night   # re-issue a past night
    python newsletter.py --to me@example.com                 # one-off recipient override

Scheduled at 07:00, the night may not be COMPLETE yet in midwinter (dawn at this latitude can
fall after 8am); the script notices and sleeps until just past dawn rather than mailing "The
Night Before" by mistake. Configure in config_local.py (never here -- the values are private):
email_to / email_from / email_resend_api_key, plus email_dashboard_url if the rig's hostname
isn't reachable from your phone. Nothing is sent until all three are set; the scheduled run
stays a quiet no-op (exit 0) so the task can be registered before the account exists.

Privacy note (this repo is public, so say it plainly): sending an email moves yard photos off
this machine, through Resend, to your inbox. That is the point, but it is also the widest the
project's media ever travels automatically -- point email_to only at people you'd show the
dashboard to.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time as _time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import db
import mdns
import stats

# What the paper is called, in the subject line and over the masthead. ONE name for every
# edition: the period is stated inside the issue (and in the subject's own words), so putting
# Morning/Evening in the title too was saying it twice and made the paper look like two
# publications. dashboard.js carries the same string over its Creature Report tab, so a reader
# who taps a link lands on a page named the same as the thing they were reading -- the ROUTE
# behind that tab stays `#dispatch`, because every issue ever sent links to it.
MASTHEAD = "Creature Report"

# ---------------------------------------------------------------------------
# Layout knobs. The email must stay light: Gmail clips messages whose HTML part exceeds ~102 KB
# (the photos ride as attachments and don't count), and a phone on cell data shouldn't download
# megabytes before coffee. Budgets, not aspirations -- the renderer enforces them.
# ---------------------------------------------------------------------------
HERO_MAX_W = 880          # px; a 1280x720 frame resized to this reads sharp in a 620px column
HERO_JPEG_Q = 80
THUMB_PX = 112            # square visit/species thumbnails (displayed at 56 css px = 2x sharp)
THUMB_JPEG_Q = 78
MAX_VISIT_THUMBS = 10     # visits beyond this still get a text row, just no photo
MAX_ROLL_ROWS = 8         # species rows with photos; a longer roll is summarized in one line
MAX_IMAGE_BYTES = 8_000_000   # absolute safety cap across all embedded images (Resend caps at 40MB)
RESEND_URL = "https://api.resend.com/emails"
# Resend's API sits behind Cloudflare, which BANS urllib's default "Python-urllib/3.x" signature:
# the request never reaches Resend and comes back as an HTML-less `403 error code: 1010` -- a
# Cloudflare code, not an API error, which is why it says nothing about keys or domains. Any
# honest agent string clears it (verified 2026-08-12: default -> 403/1010, this -> a real API
# response). Identify the tool rather than impersonate a browser.
USER_AGENT = "backyard-critter-cam/1.0 (+https://github.com/unclemattmakes/backyard-critter-cam)"

# Playful pseudo-taxonomic labels, mirrored from dashboard.js (flavour only, same caveat).
_LATIN = {
    "raccoon": "Procyon lotor", "american crow": "Corvus brachyrhynchos",
    "eastern gray squirrel": "Sciurus carolinensis", "dark-eyed junco": "Junco hyemalis",
    "domestic cat": "Felis catus", "virginia opossum": "Didelphis virginiana",
    "spotted towhee": "Pipilo maculatus", "brown rat": "Rattus norvegicus",
    "steller’s jay": "Cyanocitta stelleri", "black-capped chickadee": "Poecile atricapillus",
    "house finch": "Haemorhous mexicanus", "american robin": "Turdus migratorius",
    "european starling": "Sturnus vulgaris", "northern flicker": "Colaptes auratus",
    "varied thrush": "Ixoreus naevius", "band-tailed pigeon": "Patagioenas fasciata",
    "eastern cottontail": "Sylvilagus floridanus", "house sparrow": "Passer domesticus",
    "townsend’s chipmunk": "Neotamias townsendii",
}

# The paper. One palette, defined once -- the email renders on cream even in dark-mode clients
# (bgcolor attributes survive where CSS is stripped), because a newspaper is a printed thing.
_C = {
    "paper": "#f2ecdc", "page": "#fbf7ec", "ink": "#2a2118", "soft": "#5c4a34",
    "faint": "#8a7a64", "rule": "#e0d5c0", "gilt": "#8a5c14", "flag_bg": "#f6efe0",
    "new_bg": "#e9f0e2", "new_ink": "#3c5a2e", "warn_bg": "#f7e8df", "warn_ink": "#8a4a2e",
}


# ---------------------------------------------------------------------------
# Small formatting helpers (Python mirrors of the dashboard's esc/cap1/fmtClock).
# ---------------------------------------------------------------------------

def _esc(s) -> str:
    return (str("" if s is None else s)
            .replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _cap1(s: str) -> str:
    """Word-initial capitals, bird-guide style: "band-tailed pigeon" -> "Band-tailed Pigeon"."""
    out, up = [], True
    for ch in str(s or ""):
        out.append(ch.upper() if up and ch.isalpha() else ch)
        up = ch in " ("
    return "".join(out)


def _name_of(sp) -> str:
    return "Unidentified" if (sp or "animal") == "animal" else _cap1(sp)


def _latin_of(sp) -> str:
    return _LATIN.get((sp or "").lower().replace("'", "’"), "")


def _clock(iso, tz=None) -> str:
    """Wall-clock 'h:mm AM'. The digest carries TWO time frames: detection rows are stamped in
    the machine's wall clock, but the period boundaries come from stats._sun in the yard's SOLAR
    zone (round(lon/15) -- no DST), which in a PDT summer prints an hour early. The instants are
    identical; the printed hours are not, and one email mixing both frames read "between 8:01 PM
    and 4:26 AM" above a visit log ending at 5:12 AM. So: normalize every aware time to ONE
    frame before printing -- the machine's local zone (`tz` overrides, for tests), exactly what
    the dashboard's in-browser Date rendering has been doing all along."""
    try:
        d = datetime.fromisoformat(str(iso))
    except (TypeError, ValueError):
        return ""
    if d.tzinfo is not None:
        d = d.astimezone(tz)
    return f"{d.hour % 12 or 12}:{d.minute:02d} {'AM' if d.hour < 12 else 'PM'}"


def _fmt_hour(h: int) -> str:
    return f"{h % 12 or 12}{'am' if h < 12 else 'pm'}"


def _dur_text(minutes) -> str:
    m = float(minutes or 0)
    return "brief" if m < 1.5 else f"{round(m)} min"


def _species_line(splist, cap=3) -> str:
    """A visit's species mix, leading species first, capped. A kit-melee family visit can carry
    15+ labels (the classifier forced onto blurred multi-animal crops -- the "27 species" family
    night); the digest already sorts a visit's species by count, so the head of the list is what
    was actually there and the tail is mostly noise. Named in full on the dashboard; in an email
    row, three plus a count reads true without shouting."""
    names = [_name_of(x) for x in (splist or [])]
    if not names:
        return "Unidentified"
    if len(names) <= cap:
        return " + ".join(names)
    return " + ".join(names[:cap]) + f" (+{len(names) - cap} more)"


# ---------------------------------------------------------------------------
# WHO IT GOES TO, and WHERE ITS LINKS POINT.
# ---------------------------------------------------------------------------

def recipients(cfg, override=None) -> list[str]:
    """The issue's To: list. Accepts one address, a comma/semicolon-separated string, or a
    list/tuple -- a household grows, and `email_to = "a@x.com, b@y.com"` is what a person
    naturally types. Order is preserved, blanks dropped, case-insensitive duplicates collapsed
    (Resend treats a repeat as a second recipient).

    Nobody on this list learns anyone else's address: send_issue posts ONE message per recipient,
    each addressed only to its reader. It still reads like one letter to the family rather than a
    mail-merge, and adding an address no longer discloses the existing readers to the new one --
    which is what made the old shared To: header a decision rather than a detail."""
    raw = override if override is not None else getattr(cfg, "email_to", None)
    if raw is None:
        return []
    parts = re.split(r"[,;]", raw) if isinstance(raw, str) else list(raw)
    out, seen = [], set()
    for part in parts:
        addr = str(part).strip()
        if addr and addr.casefold() not in seen:
            seen.add(addr.casefold())
            out.append(addr)
    return out


def _lan_ip() -> str | None:
    """This machine's address ON THE LAN, or None.

    The implementation moved to mdns.lan_ip when the rig started announcing itself by name: the
    address the email LINKS to and the address the rig ANNOUNCES have to be the same one, and two
    copies of a UDP-connect trick is exactly the kind of pair that drifts. Kept as a name here
    because it is what this module's callers and tests reach for."""
    return mdns.lan_ip()


def dashboard_base(cfg) -> str:
    """Where a phone on the sofa should point to reach this rig.

    An explicit email_dashboard_url always wins. Otherwise the LAN IP -- because the bare Windows
    hostname this used to emit ("http://<this-pc>:8000") frequently does NOT resolve from a
    phone: iOS and Android ask mDNS for `name.local`, they do not speak NetBIOS, so the one link
    in the paper died exactly where it was meant to be tapped. The address is re-derived for every
    issue, so a DHCP reassignment heals itself with tomorrow's edition rather than needing a
    config edit. Falls back to the hostname when there is no LAN to find.

    Built through mdns.url so a link on the default port 80 comes out as "http://192.168.1.50"
    rather than "...:80" -- the same number a browser would have assumed, and one more thing for
    a reader to mistrust in a link they are being asked to tap."""
    base = getattr(cfg, "email_dashboard_url", None)
    if base:
        return str(base).rstrip("/")
    return mdns.url(cfg, _lan_ip() or socket.gethostname().lower())


def dashboard_answering(base, timeout=1.5) -> bool | None:
    """Is something accepting connections at `base`? None when the URL can't be parsed.

    Advisory only -- the links print either way, because the rig may well be started between the
    07:00 send and the reader's coffee, and a link that works later is not a lie. What this buys
    is the footnote: a dead tap should tell you the DASHBOARD was down, not leave you wondering
    whether the email got the address wrong."""
    try:
        parts = urllib.parse.urlsplit(base)
        host, port = parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)
        if not host:
            return None
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Gathering the issue: the digest + the cast, plus the couple of scalars only the email needs.
# ---------------------------------------------------------------------------

def collect_issue(cfg, edition="night", date=None, now=None) -> dict:
    """Everything one issue renders from. Wraps period_digest + cast_rollcall untouched and adds
    the issue number (days since the record began -- a newspaper wants a masthead number) and the
    dashboard link base. Returns {"d", "rc", "issue_no", "base_url", "generated"}."""
    d = stats.period_digest(cfg, edition=edition, date=date, now=now)
    rc = stats.cast_rollcall(cfg, now=now)
    issue_no = None
    conn = db.connect_readonly(cfg.db_path)
    if conn is not None:
        try:
            first = conn.execute("SELECT MIN(timestamp) t FROM detections").fetchone()
            f = db.parse_local(first["t"]) if first and first["t"] else None
            if f is not None and d.get("anchor"):
                anchor = datetime.fromisoformat(d["anchor"]).date()
                issue_no = max(1, (anchor - f.date()).days + 1)
        except Exception:
            issue_no = None
        finally:
            conn.close()
    base = dashboard_base(cfg)
    # The email re-judges the digest's plate for CUTENESS (see pick_plate); the digest's own
    # sharpest-frame plate stays as the fallback whenever the re-judging can't run.
    try:
        plate = pick_plate(cfg, d) or d.get("plate")
    except Exception:
        plate = d.get("plate")
    return {"d": d, "rc": rc, "plate": plate, "issue_no": issue_no,
            "base_url": base, "lan_ok": dashboard_answering(base),
            "generated": (now or datetime.now().astimezone()).isoformat()}


def _present_names(d) -> list[tuple[str, int]]:
    """Named individuals in this period's visit log, most visits first. Group stamps ("Stan +
    Kits") count as themselves -- never split a family span into solo claims (raccoon-cast rule)."""
    counts: dict[str, int] = {}
    for v in d.get("visit_log") or []:
        for name in v.get("individuals") or []:
            counts[name] = counts.get(name, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


# ---------------------------------------------------------------------------
# The words. Composed from digest fields only, hedges preserved: the crowd number is a floor,
# a dark camera is not an empty yard, and "surprising" species are questions, not fauna.
# ---------------------------------------------------------------------------

def compose_subject(bundle) -> str:
    d = bundle["d"]
    ed = d.get("edition") or "night"
    masthead = MASTHEAD
    if d.get("empty"):
        cov = d.get("coverage")
        quiet = "a quiet night" if ed == "night" else "a quiet day"
        if cov and (cov.get("dark_minutes") or 0) >= 30:
            quiet += " (camera was dark part of it)"
        return f"{masthead} · {quiet}"
    n = d.get("visits") or 0
    head = None
    novel = d.get("novel") or []
    if novel:
        sp = novel[0]
        s = next((x for x in d.get("species", []) if x["species"] == sp), None)
        nov = (s or {}).get("novelty") or {}
        head = (f"a first-ever {_name_of(sp)}" if nov.get("first_ever")
                else f"first {_name_of(sp)} in {nov.get('days_since')} days")
    crowd = d.get("crowd") or {}
    if head is None and (crowd.get("n") or 0) >= 3:
        head = f"at least {crowd['n']} in the yard at once"
    if head is None:
        names = _present_names(d)
        if names:
            head = " & ".join(_cap1(nm) for nm, _k in names[:2]) + " came by"
    if head is None:
        plate = bundle.get("plate") or d.get("plate")
        if plate:
            head = f"{_name_of(plate.get('species'))} at {_clock(plate.get('time'))}"
    subject = f"{masthead} · {n} visit{'' if n == 1 else 's'}"
    return f"{subject} · {head}" if head else subject


def compose_lede(bundle) -> list[str]:
    """Two to four short sentences a person actually wants at 7am. Returned as a list so the
    text part can join them its own way."""
    d = bundle["d"]
    out = []
    if d.get("empty"):
        out.append("No visitors recorded. The yard kept its own counsel.")
    else:
        n_sp = len(d.get("species") or []) - (d.get("n_surprising") or 0)
        out.append(f"Between {_clock(d.get('start'))} and {_clock(d.get('end'))}, the yard "
                   f"logged {d.get('visits')} visit{'' if d.get('visits') == 1 else 's'} from "
                   f"{n_sp} species.")
        names = _present_names(d)
        if names:
            bits = [f"{_cap1(nm)}" + (f" ({k}×)" if k > 1 else "") for nm, k in names[:3]]
            more = f" and {len(names) - 3} more" if len(names) > 3 else ""
            out.append("Familiar faces: " + ", ".join(bits) + more + ".")
        crowd = d.get("crowd") or {}
        if (crowd.get("n") or 0) >= 2:
            out.append(f"The fullest moment came at {_clock(crowd.get('at'))} — at least "
                       f"{crowd['n']} animals in frame at once (a floor, not a count).")
        elif d.get("busiest_hour"):
            bh = d["busiest_hour"]
            out.append(f"The busiest hour was {_fmt_hour(bh['hour'])}, "
                       f"with {bh['visits']} arrival{'' if bh['visits'] == 1 else 's'}.")
    cov = d.get("coverage")
    if cov and (cov.get("dark_minutes") or 0) >= 30:
        mins = cov["dark_minutes"]
        span = f"{round(mins / 60, 1)} h" if mins >= 90 else f"{mins} min"
        out.append(f"Caveat: the camera was dark for {span} of this period — "
                   "absence claims are about the camera, not the yard.")
    return out


# ---------------------------------------------------------------------------
# THE PLATE, RE-JUDGED FOR CUTENESS (2026-08-12). The digest's plate is _shot_score's "sharpest
# frame", and its sharpness is quality.py's whole-crop Laplacian variance -- which is exactly how
# the first issue led with a crisp photo of brick pavers wearing a motion-blurred raccoon: high-
# frequency background texture outscores the animal, and a head-down butt-shot outscores a face.
# The email's hero wants the CUTEST shot instead, judged by three things the row + crop actually
# contain: how much animal there is (bbox fraction -- closeness is detail), whether the sharpness
# lives ON the animal (centre-of-crop focus vs the border ring -- a tight bbox means centre ==
# body, ring == background), and night eyeshine (bright specks in a dark crop = the animal is
# LOOKING AT YOU; quality.py's own trick, reused as a face proxy). Pixel checks run only on a
# short shortlist -- this is a once-a-day email, not a hot path. The dashboard's plate is
# deliberately untouched: this is an editorial choice for the newspaper, not new truth.
# ---------------------------------------------------------------------------
PLATE_SHORTLIST = 40      # rows that get their crop file opened and focus-judged
TRAILCAM_BANNER_FRAC = 0.055   # bottom strip of a trail-cam frame that is OSD timestamp, not yard


def _focus_stats(img) -> tuple[float, float]:
    """(centre_var, border_var): gradient-energy variance of the crop's central 25-75% box vs the
    border ring. PIL+numpy only (no cv2 here): a plain finite-difference gradient magnitude is
    Laplacian-variance-like and entirely sufficient for a RATIO."""
    import numpy as np
    g = np.asarray(img.convert("L"), dtype=np.float32)
    if g.shape[0] < 8 or g.shape[1] < 8:
        return 0.0, 0.0
    e = (np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1, :])
    h, w = e.shape
    cy0, cy1, cx0, cx1 = h // 4, 3 * h // 4, w // 4, 3 * w // 4
    centre = e[cy0:cy1, cx0:cx1]
    ring = e.copy()
    ring[cy0:cy1, cx0:cx1] = np.nan
    ring = ring[~np.isnan(ring)]
    return (float(centre.var()) if centre.size else 0.0,
            float(ring.var()) if ring.size else 0.0)


def _judge_crop(path: Path) -> dict | None:
    """Pixel evidence for the cuteness court: subject sharpness, where the focus lives, and
    eyeshine. None when the file can't be read (the candidate is simply excused)."""
    try:
        from PIL import Image
        import numpy as np
        with Image.open(path) as im:
            centre_var, border_var = _focus_stats(im)
            g = np.asarray(im.convert("L"), dtype=np.float32)
            eyeshine = 0.0
            if g.mean() < 90:
                # Bright specks in a dark crop are eyes -- but only SPECKS. quality.py's
                # unconditional version was measured against this corpus's trail-cam closeups
                # and awarded its full bonus to an IR-floodlit raccoon BUTT (the fur blows out
                # >240 across a third of the crop). Eyes are tiny; a large bright area is lit
                # fur, and gets nothing.
                bright_frac = float((g > 240).mean())
                if 0.0 < bright_frac <= 0.03:
                    eyeshine = min(bright_frac * 60.0, 0.5)
        return {"centre_var": centre_var, "border_var": border_var, "eyeshine": eyeshine}
    except Exception:
        return None


SIZE_PEAK = 0.35    # bbox fraction where "close" stops meaning "good": past this the animal is
#                     half out of frame -- last night's top-size candidate was a tail at 93%


def _size_term(size: float) -> float:
    """Peaked closeness: rises like sqrt(size) to SIZE_PEAK, then declines. A raccoon PORTRAIT
    fills a tenth to a third of a frame; a bbox near the whole frame is a walk-past body part
    (measured on this corpus: the 0.93-fraction candidate was fur and tail, no face in frame)."""
    if size <= 0:
        return 0.0
    if size <= SIZE_PEAK:
        return size ** 0.5
    return (SIZE_PEAK ** 0.5) * (SIZE_PEAK / size) ** 0.75


def _cuteness(row, judged) -> float:
    """One number for "lead the paper with this". Peaked closeness (a near animal is the shot; a
    TOO-near animal is a body part), then subject sharpness weighted by WHERE the focus is --
    centre/border ratio below 1 means the camera focused on the yard, not the visitor -- then
    the eyeshine face bonus. Verified species and a solo name nudge; an unidentified 'animal'
    blob (usually a melee smear) sags."""
    import math
    fw, fh = (row.get("frame_w") or 0), (row.get("frame_h") or 0)
    size = 0.0
    if fw and fh:
        size = max(0.0, (row["bbox_x2"] - row["bbox_x1"]) * (row["bbox_y2"] - row["bbox_y1"])
                   / float(fw * fh))
    cx = (row["bbox_x1"] + row["bbox_x2"]) / 2.0 / (fw or 1)
    cy = (row["bbox_y1"] + row["bbox_y2"]) / 2.0 / (fh or 1)
    center = 1.0 - min(1.0, abs(cx - 0.5) + abs(cy - 0.5))
    score = _size_term(size) * (0.75 + 0.25 * center)
    subject = math.sqrt(max(judged["centre_var"], 0.0))
    ratio = judged["centre_var"] / (judged["border_var"] + 1e-6)
    score *= subject * min(max(ratio, 0.2), 2.5) ** 0.5
    score *= 1.0 + judged["eyeshine"]
    if row.get("verified") == 1:
        score *= 1.15
    iid = row.get("individual_id")
    if iid and not db.is_group_label(iid):
        score *= 1.10          # somebody we KNOW leads the paper over an anonymous visitor
    elif iid:
        score *= 1.08          # a family stamp is the same news, slightly weaker attribution --
        #                        this is an editorial nudge, never matcher truth (is_group_label)
    if (row.get("species") or row.get("detection_class")) in (None, "animal"):
        score *= 0.85
    if _banner_touches(row):
        score *= 0.85                              # OSD timestamp strip in frame -- ugly hero
    return score


def _banner_touches(row) -> bool:
    """True when a trail-cam bbox reaches into the OSD timestamp band at the frame's bottom."""
    fh = row.get("frame_h") or 0
    return bool(row.get("source") == "trail_cam_sd" and fh
                and row["bbox_y2"] >= fh * (1.0 - TRAILCAM_BANNER_FRAC))


def pick_plate(cfg, d) -> dict | None:
    """Re-judge the period's rows for the email's hero. Returns a plate-shaped dict (species /
    time / crop_path / clip) carrying its bbox, or None (caller falls back to the digest's
    plate). Shortlists by animal-size cheaply, then opens only the shortlist's crop files."""
    if not d.get("start") or d.get("empty"):
        return None
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return None
    try:
        rows = conn.execute(
            "SELECT id, timestamp, source, detection_class, species, confidence, "
            "species_confidence, species_verified, crop_path, crop_quality, individual_id, "
            "frame_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_w, frame_h "
            "FROM detections WHERE timestamp >= ? AND timestamp < ? AND crop_path IS NOT NULL",
            (d["start"][:19], d["end"])).fetchall()
        clips = stats.load_clips(conn)
    finally:
        conn.close()
    start = datetime.fromisoformat(d["start"])
    end = datetime.fromisoformat(d["end"])
    cands = []
    for r in rows:
        dt = db.parse_local(r["timestamp"])
        if dt is None:
            continue
        try:
            if not (start <= dt < end):
                continue
        except TypeError:
            continue                                  # naive/aware mix -- skip rather than guess
        label = (r["species"] or r["detection_class"] or "").lower()
        if label in stats._NON_CRITTER:
            continue
        if r["bbox_x1"] is None or not r["frame_w"] or not r["frame_h"]:
            continue
        row = dict(r)
        row["dt"] = dt
        row["label"] = r["species"] or r["detection_class"]
        row["verified"] = r["species_verified"]
        size = ((r["bbox_x2"] - r["bbox_x1"]) * (r["bbox_y2"] - r["bbox_y1"])
                / float(r["frame_w"] * r["frame_h"]))
        row["_presize"] = size
        cands.append(row)
    if not cands:
        return None
    cands.sort(key=lambda r: _size_term(r["_presize"]), reverse=True)
    root = cfg.clips_dir.parent
    best, best_score = None, 0.0
    for row in cands[:PLATE_SHORTLIST]:
        judged = _judge_crop(root / row["crop_path"].replace("\\", "/"))
        if judged is None:
            continue
        s = _cuteness(row, judged)
        if s > best_score:
            best, best_score = row, s
    if best is None:
        return None
    clip = stats.clip_at(clips, best["source"], best["dt"])
    clip_out = stats._clip_out(clip)
    if clip_out and clip:
        # The wall span rides along so the hero seek can map wall time onto media time (see
        # _extract_clip_frame); _clip_out drops the parsed datetimes it needs.
        clip_out["wall_seconds"] = max(0.0, (clip["edt"] - clip["sdt"]).total_seconds())
    return {
        "crop_path": best["crop_path"].replace("\\", "/"),
        "species": best["label"], "time": best["dt"].isoformat(),
        "conf": round(best["species_confidence"] or best["confidence"] or 0.0, 3),
        "clip": clip_out,
        "frame_path": (best["frame_path"].replace("\\", "/") if best["frame_path"] else None),
        "bbox": (best["bbox_x1"], best["bbox_y1"], best["bbox_x2"], best["bbox_y2"]),
        "frame_size": (best["frame_w"], best["frame_h"]),
        "source": best["source"],
        "individual": (best["individual_id"]
                       if best["individual_id"] and not db.is_group_label(best["individual_id"])
                       else None),
        # A group stamp can't name THIS animal (the crop is one body out of a family span), but
        # "one of the family" is honest and worth the caption.
        "family": (best["individual_id"]
                   if best["individual_id"] and db.is_group_label(best["individual_id"])
                   else None),
    }


# ---------------------------------------------------------------------------
# Images. Every function here fails SOFT (returns None) -- a missing crop, a pruned clip or a
# machine without ffmpeg must cost the email a photo, never the morning its issue.
# ---------------------------------------------------------------------------

def _load_jpeg(path: Path, max_w: int, quality: int, square: bool = False) -> bytes | None:
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            if square:
                side = min(im.size)
                left, top = (im.width - side) // 2, (im.height - side) // 2
                im = im.crop((left, top, left + side, top + side))
            if im.width > max_w:
                im = im.resize((max_w, max(1, round(im.height * max_w / im.width))))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality, optimize=True)
            return buf.getvalue()
    except Exception:
        return None


def _run_ffmpeg(args, timeout):
    subprocess.run(args, check=True, timeout=timeout,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _hero_region(bbox, img_size, frame_size, source=None,
                 pad=0.7, min_w=560, min_h=420) -> tuple[int, int, int, int]:
    """Where to crop the full frame so the hero is ABOUT the animal. The first issue showed the
    entire 1920x1080 yard -- brick pavers with a small visitor -- because a frame is not a
    composition. This pads the bbox into a portrait (double the animal's size in context), grows
    to a minimum so tiny animals get scene rather than a gross upscale, clamps inside the image,
    and on trail-cam frames stays above the OSD timestamp band unless the animal itself stands in
    it. bbox is in FRAME coordinates; the image may be a scaled clip frame, so everything is
    mapped through img/frame scale first."""
    iw, ih = img_size
    fw, fh = frame_size
    sx, sy = (iw / fw if fw else 1.0), (ih / fh if fh else 1.0)
    x1, y1, x2, y2 = bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    rx1, ry1, rx2, ry2 = x1 - pad * bw, y1 - pad * bh, x2 + pad * bw, y2 + pad * bh
    # Grow each axis to its minimum, centred on the animal.
    if rx2 - rx1 < min_w:
        cx = (rx1 + rx2) / 2
        rx1, rx2 = cx - min_w / 2, cx + min_w / 2
    if ry2 - ry1 < min_h:
        cy = (ry1 + ry2) / 2
        ry1, ry2 = cy - min_h / 2, cy + min_h / 2
    # Trail-cam OSD band: keep the frame's bottom strip out unless the bbox reaches into it.
    bottom_limit = float(ih)
    if source == "trail_cam_sd":
        bottom_limit = max(ih * (1.0 - TRAILCAM_BANNER_FRAC), min(y2, ih))
    # Slide (not shrink) back inside the image, then hard-clamp.
    if rx1 < 0:
        rx1, rx2 = 0, rx2 - rx1
    if rx2 > iw:
        rx1, rx2 = rx1 - (rx2 - iw), iw
    if ry1 < 0:
        ry1, ry2 = 0, ry2 - ry1
    if ry2 > bottom_limit:
        ry1, ry2 = ry1 - (ry2 - bottom_limit), bottom_limit
    return (int(max(0, rx1)), int(max(0, ry1)),
            int(min(iw, max(rx1 + 1, rx2))), int(min(bottom_limit, max(ry1 + 1, ry2))))


def _extract_clip_frame(cfg, plate):
    """PIL Image of the frame rolling at the plate's instant, or None. The seek offset is
    clamped inside the clip (a plate stamped at the final buffered detection can land a hair
    past the last frame, and ffmpeg would return nothing)."""
    clip = (plate or {}).get("clip") or {}
    rel = clip.get("clip_path")
    if not rel or shutil.which("ffmpeg") is None:
        return None
    src = (cfg.clips_dir.parent / rel)
    if not src.exists():
        return None
    t_plate = db.parse_local(plate.get("time"))
    t_start = db.parse_local(clip.get("start"))
    off = 0.0
    if t_plate is not None and t_start is not None:
        off = max(0.0, (t_plate - t_start).total_seconds())
    secs = clip.get("seconds")
    # MEDIA time is not WALL time: the recorder writes at a nominal fps while frames arrive at
    # capture rate, so a clip can hold 134.7s of playback across 59.6 wall seconds (measured on
    # this rig, 2026-08-12 -- the first bbox-framed hero seeked 2.8 wall-seconds into that clip
    # and showed the empty yard 3.5 media-seconds before the animal arrived). When the caller
    # knows the wall span, map proportionally; without it, the unscaled offset is the best guess.
    wall = clip.get("wall_seconds")
    if secs and wall and float(wall) > 0.5:
        off *= float(secs) / float(wall)
    if secs:
        off = min(off, max(0.0, float(secs) - 0.5))
    # Combined seek: -ss BEFORE -i is a fast KEYFRAME seek, and these pipe-recorded clips carry
    # long GOPs -- alone it lands whole seconds off the moment (measured: the hero showed the
    # yard before the animal arrived). Coarse-seek to 15s shy of the target, then decode-accurate
    # seek the remainder, so the frame is exact and the decode stays bounded.
    coarse = max(0.0, off - 15.0)
    fine = off - coarse
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "hero.jpg"
        try:
            _run_ffmpeg(["ffmpeg", "-hide_banner", "-loglevel", "error",
                         "-ss", f"{coarse:.2f}", "-i", str(src), "-ss", f"{fine:.2f}",
                         "-frames:v", "1", "-q:v", "3", "-y", str(out)], timeout=120)
            data = out.read_bytes() if out.exists() else b""
        except Exception:
            return None
    if not data:
        return None
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(data))
        im.load()
        return im
    except Exception:
        return None


HERO_CROP_MIN_W = 480   # a crop at least this wide IS the hero (588 css px shows it downscaled)


def _hero_bytes(cfg, plate) -> bytes | None:
    """The hero JPEG, sources tried most-truthful-first.

    1. THE CROP ITSELF, when it is hero-sized. It is the one image guaranteed to show the
       moment -- it was cut live from the capture frame -- and because the cuteness score
       prefers close animals, the winner's crop usually IS large. (The old never-lead-with-a-
       crop rule was about 96-389px crops upscaled 4-16x; a >=480px crop at 588 css px is a
       downscale, which is the rule's actual point.)
    2. The stored full FRAME (frame_path), composed around the bbox -- also the exact moment.
    3. A frame pulled from the covering CLIP, composed the same way -- last, because these
       clips' media time is nominal-fps time, not wall time: even proportionally rescaled and
       accurately sought, the extracted frame was measured seconds off the moment (bbox drawn
       on it framed empty pavers; the animal had not arrived). Scene-true, moment-approximate.
    None -> the caller falls back to the small-capped crop."""
    try:
        from PIL import Image
    except Exception:
        return None
    root = cfg.clips_dir.parent
    if plate.get("crop_path"):
        try:
            img = Image.open(root / plate["crop_path"])
            img.load()
            if img.width >= HERO_CROP_MIN_W:
                return _encode_hero(img)
        except Exception:
            pass
    for source in ("frame", "clip"):
        img = None
        if source == "frame" and plate.get("frame_path"):
            try:
                img = Image.open(root / plate["frame_path"])
                img.load()
            except Exception:
                img = None
        elif source == "clip":
            img = _extract_clip_frame(cfg, plate)
        if img is None:
            continue
        try:
            img = img.convert("RGB")
            if plate.get("bbox") and plate.get("frame_size"):
                img = img.crop(_hero_region(plate["bbox"], img.size, plate["frame_size"],
                                            source=plate.get("source")))
            return _encode_hero(img)
        except Exception:
            continue
    return None


def _encode_hero(img) -> bytes:
    img = img.convert("RGB")
    if img.width > HERO_MAX_W:
        img = img.resize((HERO_MAX_W, max(1, round(img.height * HERO_MAX_W / img.width))))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=HERO_JPEG_Q, optimize=True)
    return buf.getvalue()


def gather_images(cfg, d, plate=None) -> dict:
    """cid -> {"bytes", "mime", "hero": bool} for everything the layout may reference. Bounded:
    thumbnail counts are capped and the total payload must stay under MAX_IMAGE_BYTES -- past
    the cap further photos are simply dropped (rows render text-only, which is honest enough).
    `plate` is the (possibly re-judged) hero subject; defaults to the digest's own plate."""
    images: dict[str, dict] = {}
    total = 0

    def add(cid, data, hero=False):
        nonlocal total
        if data is None or total + len(data) > MAX_IMAGE_BYTES:
            return False
        images[cid] = {"bytes": data, "mime": "image/jpeg", "hero": hero}
        total += len(data)
        return True

    root = cfg.clips_dir.parent
    plate = plate or d.get("plate")
    if plate:
        hero = _hero_bytes(cfg, plate)
        if hero is not None:
            add("hero", hero, hero=True)
        elif plate.get("crop_path"):
            # No frame source covered the instant: the crop, at NATIVE size (capped, never upscaled).
            add("plate-crop", _load_jpeg(root / plate["crop_path"], 320, HERO_JPEG_Q))
    for i, v in enumerate((d.get("visit_log") or [])[:MAX_VISIT_THUMBS]):
        if v.get("rep_crop"):
            add(f"v{i}", _load_jpeg(root / v["rep_crop"], THUMB_PX, THUMB_JPEG_Q, square=True))
    for i, s in enumerate((d.get("species") or [])[:MAX_ROLL_ROWS]):
        if s.get("rep_crop"):
            add(f"s{i}", _load_jpeg(root / s["rep_crop"], THUMB_PX, THUMB_JPEG_Q, square=True))
    return images


# ---------------------------------------------------------------------------
# The layout. One renderer, two "src modes": the email references images by cid: (inline
# attachments -- the only scheme Gmail reliably shows), the archive copy embeds data: URIs
# (which browsers render and Gmail strips). Table-free except where email clients require it;
# every style inline; nothing external -- an email is a sealed envelope, not a web page.
# ---------------------------------------------------------------------------

def _img_src_cid(images):
    return lambda cid: f"cid:{cid}" if cid in images else None


def _img_src_data(images):
    def src(cid):
        im = images.get(cid)
        if not im:
            return None
        return f"data:{im['mime']};base64,{base64.b64encode(im['bytes']).decode()}"
    return src


# Every link is ABSOLUTE and points at the rig on the LAN (see dashboard_base). The dashboard
# routes on the URL hash -- #day/<date>, #species/<name>, #profile/<name>, #dispatch/<date>/<ed>,
# and the bare view names -- so the paper can hand a reader the exact page for the thing they are
# reading about, instead of one link to the front door. Names are percent-encoded because the
# cast contains spaces, apostrophes and the " + " of a family stamp, and the dashboard decodes
# with decodeURIComponent.
def _url(base, frag="") -> str:
    return f"{base}/#{frag}" if frag else (base or "")


def _q(name) -> str:
    return urllib.parse.quote(str(name or ""), safe="")


def _a(href, inner, style="") -> str:
    """An anchor, or the bare content when there is nowhere to point. Link colour is stated
    inline: mail clients drop stylesheets, and an unstyled link renders as default blue
    underline, which would pepper a newspaper with hyperlink-blue."""
    if not href:
        return inner
    return (f'<a href="{_esc(href)}" style="color:inherit;text-decoration:none;{style}">'
            f'{inner}</a>')


def _flag(text, bg=_C["flag_bg"], ink=_C["soft"]) -> str:
    return (f'<span style="display:inline-block;background:{bg};color:{ink};'
            f'border:1px solid {_C["rule"]};border-radius:10px;padding:2px 10px;'
            f'margin:0 6px 6px 0;font-size:12px;">{text}</span>')


def render_email(bundle, images, img_src) -> str:
    d, rc = bundle["d"], bundle["rc"]
    ed = d.get("edition") or "night"
    masthead = MASTHEAD
    period_word = "Night" if ed == "night" else "Day"
    anchor = d.get("anchor") or ""
    try:
        datestr = datetime.fromisoformat(anchor).strftime("%A, %B %d, %Y").replace(" 0", " ")
    except ValueError:
        datestr = anchor
    issue = f"No. {bundle['issue_no']} · " if bundle.get("issue_no") else ""
    moon = d.get("moon")
    moonstr = f" · {moon['glyph']} {_esc(moon['name'])}, {moon['illum_pct']}% lit" if moon else ""
    base = bundle.get("base_url") or ""
    dash = _url(base, f"dispatch/{anchor}/{ed}") if anchor else base

    lede = compose_lede(bundle)
    parts = []

    # -- masthead ---------------------------------------------------------------
    parts.append(f"""
    <div style="text-align:center;padding:26px 0 14px;border-bottom:3px double {_C['ink']};">
      <div style="font-size:11px;letter-spacing:3px;color:{_C['faint']};text-transform:uppercase;">Backyard Critter Cam</div>
      <div style="font-size:34px;font-weight:700;letter-spacing:1px;margin:6px 0 4px;">{_esc(masthead)}</div>
      <div style="font-size:12px;color:{_C['soft']};">{issue}{_esc(d.get('title') or '')} · {_esc(datestr)}{moonstr}</div>
    </div>""")

    # -- lede + flags -------------------------------------------------------------
    parts.append(f'<p style="font-size:15px;line-height:1.55;margin:18px 2px 10px;">{" ".join(_esc(s) for s in lede)}</p>')
    flags = []
    for sp in d.get("novel") or []:
        s = next((x for x in d.get("species", []) if x["species"] == sp), None)
        nov = (s or {}).get("novelty") or {}
        lead = "First ever recorded" if nov.get("first_ever") else (
            f"First in {nov.get('days_since')} days" if nov.get("days_since") else "Notable")
        flags.append(_flag(f"❋ {lead}: {_esc(_name_of(sp))}", _C["new_bg"], _C["new_ink"]))
    for q in d.get("quiet") or []:
        flags.append(_flag(f"— No {_esc(_name_of(q['species']))} this {ed} "
                           f"(usually {round(q['frac'] * 100)}% of {ed}s)"))
    cov = d.get("coverage")
    if cov and (cov.get("dark_minutes") or 0) >= 30:
        mins = cov["dark_minutes"]
        span = f"{round(mins / 60, 1)} h" if mins >= 90 else f"{mins} min"
        flags.append(_flag(f"⚠ camera dark {span} of this {ed}", _C["warn_bg"], _C["warn_ink"]))
    if flags:
        parts.append(f'<div style="margin:0 0 6px;">{"".join(flags)}</div>')

    if d.get("empty"):
        parts.append(f'<p style="color:{_C["faint"]};font-style:italic;margin:14px 2px;">'
                     f'A quiet {ed} — the feeder waits, the record holds.</p>')
    else:
        # -- hero: plate of the period (the CUTEST shot -- see pick_plate) -------------
        plate = bundle.get("plate") or d.get("plate") or {}
        hero_src = img_src("hero")
        plate_src = img_src("plate-crop")
        who = _name_of(plate.get("species"))
        if plate.get("individual"):
            who = f'{_cap1(plate["individual"])} the {who}'
        elif plate.get("family"):
            who = f'{who} (of {_cap1(plate["family"])})'
        cap = (f'<span style="color:{_C["gilt"]};font-weight:700;">Plate of the {period_word}</span>'
               f' — {_esc(who)}'
               + (f' <i style="color:{_C["faint"]};">({_esc(_latin_of(plate.get("species")))})</i>'
                  if _latin_of(plate.get("species")) else "")
               + f', {_esc(_clock(plate.get("time")))} · the {ed}’s cutest shot')
        if hero_src:
            parts.append(f"""
      <div style="margin:14px 0 4px;">
        {_a(dash, f'<img src="{hero_src}" width="588" alt="{_esc(_name_of(plate.get("species")))} at {_esc(_clock(plate.get("time")))}" style="width:100%;max-width:588px;border-radius:6px;border:1px solid {_C["rule"]};display:block;">')}
        <div style="font-size:12px;color:{_C['soft']};margin-top:6px;">{cap}</div>
      </div>""")
        elif plate_src:
            parts.append(f"""
      <div style="margin:14px 0 4px;text-align:center;background:{_C['page']};border:1px solid {_C['rule']};border-radius:6px;padding:16px;">
        <img src="{plate_src}" alt="{_esc(_name_of(plate.get('species')))}"
             style="max-width:320px;width:auto;border-radius:4px;display:inline-block;">
        <div style="font-size:12px;color:{_C['soft']};margin-top:8px;">{cap}</div>
      </div>""")

        # -- the night in numbers ----------------------------------------------------
        n_sp = len(d.get("species") or []) - (d.get("n_surprising") or 0)
        tallies = [(str(d.get("visits") or 0), "visits"), (str(n_sp), "species")]
        if d.get("busiest_hour"):
            tallies.append((_fmt_hour(d["busiest_hour"]["hour"]), "busiest hour"))
        crowd = d.get("crowd") or {}
        if (crowd.get("n") or 0) >= 2:
            tallies.append((f"≥{crowd['n']}", "at once (a floor)"))
        cells = "".join(
            f'<td align="center" style="padding:10px 6px;background:{_C["page"]};'
            f'border:1px solid {_C["rule"]};border-radius:6px;">'
            f'<div style="font-size:22px;font-weight:700;">{_esc(v)}</div>'
            f'<div style="font-size:11px;color:{_C["faint"]};text-transform:uppercase;'
            f'letter-spacing:1px;">{_esc(k)}</div></td>' for v, k in tallies)
        parts.append(f'<table role="presentation" width="100%" cellspacing="6" cellpadding="0" '
                     f'style="margin:12px 0;border-collapse:separate;"><tr>{cells}</tr></table>')

        # -- the visits ---------------------------------------------------------------
        vlog = d.get("visit_log") or []
        if vlog:
            first, last = _clock(vlog[0]["start"]), _clock(vlog[-1]["start"])
            parts.append(_section("The Visits", f"{len(vlog)} · first {first} · last {last}"))
            rows = []
            for i, v in enumerate(vlog):
                sp = _species_line(v.get("species"))
                inds = "".join(
                    f'<span style="background:{_C["flag_bg"]};border:1px solid {_C["rule"]};'
                    f'border-radius:8px;padding:0 7px;margin-left:6px;font-size:11px;'
                    f'color:{_C["gilt"]};font-weight:700;">{_esc(_cap1(nm))}</span>'
                    for nm in (v.get("individuals") or []))
                tag = (v.get("motion") or {}).get("tag")
                tagstr = (f'<div style="font-size:11px;color:{_C["faint"]};font-style:italic;">'
                          f'{_esc(tag)}</div>') if tag else ""
                nclips = len(v.get("clips") or [])
                right = (f'{v.get("count")} obs' + (f' · ▶ {nclips} clip'
                                                    f'{"" if nclips == 1 else "s"}' if nclips else ""))
                # The thumb <td> is emitted even with no image (rows past MAX_VISIT_THUMBS, or a
                # crop that failed to load): every row must have the same cell count or email
                # clients drift the columns.
                src = img_src(f"v{i}")
                vurl = _url(base, f"day/{str(v.get('start') or '')[:10]}") if base else ""
                thumb = (f'<td width="56" style="padding:6px 10px 6px 0;vertical-align:middle;">'
                         + (_a(vurl, f'<img src="{src}" width="56" height="56" alt="" '
                                     f'style="border-radius:6px;display:block;">') if src else "")
                         + '</td>')
                rows.append(f"""
        <tr>
          <td width="74" style="padding:6px 8px 6px 0;vertical-align:top;white-space:nowrap;">
            {_a(vurl, f"<b style='font-size:13px'>{_esc(_clock(v.get('start')))}</b>")}
            <div style="font-size:11px;color:{_C['faint']};">{_esc(_dur_text(v.get('minutes')))}</div>
          </td>{thumb}
          <td style="padding:6px 0;vertical-align:middle;">
            {_a(vurl, f"<span style='font-size:14px;font-weight:600'>{_esc(sp)}</span>")}{inds}{tagstr}
          </td>
          <td align="right" style="padding:6px 0;vertical-align:middle;font-size:11px;color:{_C['faint']};white-space:nowrap;">{right}</td>
        </tr>
        <tr><td colspan="4" style="border-bottom:1px solid {_C['rule']};font-size:0;line-height:0;">&nbsp;</td></tr>""")
            parts.append(f'<table role="presentation" width="100%" cellspacing="0" cellpadding="0">'
                         f'{"".join(rows)}</table>')

        # -- the roll -----------------------------------------------------------------
        roll = d.get("species") or []
        if roll:
            n_surp = d.get("n_surprising") or 0
            sub = (f"{len(roll) - n_surp} species" + (f" + {n_surp} to verify" if n_surp else ""))
            parts.append(_section("The Roll", sub))
            rows = []
            for i, s in enumerate(roll[:MAX_ROLL_ROWS]):
                badges = []
                nov = s.get("novelty") or {}
                if s.get("surprising"):
                    badges.append(_flag("⚠ off-hours — verify", _C["warn_bg"], _C["warn_ink"]))
                elif nov.get("first_ever"):
                    badges.append(_flag("❋ new", _C["new_bg"], _C["new_ink"]))
                elif (nov.get("days_since") or 0) >= 3:
                    badges.append(_flag(f"first in {nov['days_since']}d", _C["new_bg"], _C["new_ink"]))
                if (s.get("streak") or 0) >= 3:
                    badges.append(_flag(f"{s['streak']} {ed}s running"))
                latin = _latin_of(s.get("species"))
                latinstr = (f' <i style="font-size:11px;color:{_C["faint"]};font-weight:400;">'
                            f'{_esc(latin)}</i>') if latin else ""
                typ = f' · usually {_esc(s["typical"])}' if s.get("typical") else ""
                src = img_src(f"s{i}")
                # 'animal' has no catalogue sheet (the dashboard leaves it unclickable too).
                surl = (_url(base, f"species/{_q(s.get('species'))}")
                        if base and s.get("species") not in (None, "animal") else "")
                thumb = (f'<td width="56" style="padding:6px 10px 6px 0;">'
                         + _a(surl, f'<img src="{src}" width="56" height="56" alt="" '
                                    f'style="border-radius:6px;display:block;">') + '</td>'
                         if src else '<td width="0"></td>')
                rows.append(f"""
        <tr>{thumb}
          <td style="padding:6px 0;vertical-align:middle;">
            {_a(surl, f"<span style='font-size:14px;font-weight:600'>{_esc(_name_of(s.get('species')))}</span>")}{latinstr}
            <div style="font-size:11px;color:{_C['soft']};">{_esc(_clock(s.get('first')))}–{_esc(_clock(s.get('last')))}{typ}</div>
            {"".join(badges)}
          </td>
          <td align="right" style="vertical-align:middle;font-size:13px;white-space:nowrap;">
            <b>{s.get('visits')}</b> <span style="font-size:11px;color:{_C['faint']};">visit{"" if s.get('visits') == 1 else "s"}</span>
          </td>
        </tr>
        <tr><td colspan="3" style="border-bottom:1px solid {_C['rule']};font-size:0;line-height:0;">&nbsp;</td></tr>""")
            if len(roll) > MAX_ROLL_ROWS:
                # The overflow line keeps the digest's distinction: species the record supports
                # vs "surprising" ones (off-hours for their own history -- almost always a
                # mislabeled melee crop). Blending them would assert fauna the digest only asks about.
                tail = roll[MAX_ROLL_ROWS:]
                real = ", ".join(_name_of(s["species"]) for s in tail if not s.get("surprising"))
                surp = ", ".join(_name_of(s["species"]) for s in tail if s.get("surprising"))
                line = []
                if real:
                    line.append(f"Also on the record: {_esc(real)}.")
                if surp:
                    line.append(f'<span style="color:{_C["warn_ink"]};">Listed to verify '
                                f'(off-hours for their species): {_esc(surp)}.</span>')
                rows.append(f'<tr><td colspan="3" style="padding:8px 0;font-size:12px;'
                            f'color:{_C["faint"]};">{" ".join(line)}</td></tr>')
            parts.append(f'<table role="presentation" width="100%" cellspacing="0" cellpadding="0">'
                         f'{"".join(rows)}</table>')

    # -- cast roll call (one honest line each; photos stay on the dashboard) -----------
    cast = (rc or {}).get("cast") or []
    if cast:
        # Two subtleties, both learned from the first live issue. (1) A group stamp ("Pedro +
        # Kits") is evidence of its BASE animal, and cast_rollcall already folds its recency
        # into the solo entry (via_group) -- so a group row whose base is in the cast would name
        # the same raccoon twice. (2) "Came by" is judged against the PERIOD, not days_since: a
        # night spans two calendar dates, so an animal seen at 11 PM is days_since=1 by the
        # morning send -- calendar arithmetic calling a present animal absent.
        solos = {c["id"].casefold() for c in cast if not db.is_group_label(c["id"])}
        try:
            period_start = datetime.fromisoformat(d["start"]) if d.get("start") else None
        except ValueError:
            period_start = None

        def _in_period(c):
            last = db.parse_local(c.get("last_seen"))
            try:
                return bool(period_start and last and last >= period_start)
            except TypeError:      # naive/aware mix from an odd DB -- fall back to days_since
                return False

        seen, gone = [], []
        for c in cast:
            ds = c.get("days_since")
            if ds is None:
                continue
            if db.is_group_label(c["id"]):
                # A group label is evidence OF ITS BASE animal, never a cast member of its own:
                # beside the base it double-names the raccoon, and in the absent list it invents
                # a fact ("Cutie + Kits, 4d" under a CutiePie who came by is spelling drift, not
                # absence). Shown only as presence, only when no solo base row exists to carry it.
                base_known = c["id"].split(" + ", 1)[0].strip().casefold() in solos
                if not base_known and _in_period(c):
                    seen.append(_a(_url(base, f"profile/{_q(c['id'])}") if base else "",
                                   f'<b>{_esc(_cap1(c["id"]))}</b>'))
                continue
            label = _cap1(c["id"])
            link = _a(_url(base, f"profile/{_q(c['id'])}") if base else "",
                      f'<b>{_esc(label)}</b>')
            if _in_period(c) or ds <= 0:
                seen.append(link + (" (with the kits)" if c.get("via_group") else ""))
            elif c.get("overdue"):
                gone.append(f"{link} ({ds}d — overdue)")
            elif ds <= 10:
                gone.append(f"{link} ({ds}d)")
        lines = []
        if seen:
            lines.append(f"<b>Came by:</b> {', '.join(seen)}.")
        if gone:
            lines.append(f"<b>Not {('tonight' if ed == 'night' else 'today')}:</b> {', '.join(gone)}.")
        if lines:
            parts.append(_section("Cast Roll Call", f"{len(solos) or len(cast)} named"))
            parts.append(f'<p style="font-size:13px;line-height:1.6;margin:8px 2px;">'
                         f'{"<br>".join(lines)}</p>')

    # -- footer -------------------------------------------------------------------
    # The links reach the rig over the HOME NETWORK and nowhere else -- that is deliberate (the
    # dashboard has no login), but a reader tapping from a bus deserves to know why nothing
    # happened, and so does one tapping at the kitchen table while the rig is off.
    rooms = " &nbsp;·&nbsp; ".join(
        _a(_url(base, frag), f'<span style="color:{_C["gilt"]};">{label}</span>')
        for frag, label in (("live", "Live camera"), ("visits", "Visit log"),
                            ("indiv", "The cast"), ("calendar", "Calendar")))
    reach = ("Links open the rig's dashboard — they work on the home Wi-Fi."
             if bundle.get("lan_ok") is not False else
             "Links open the rig's dashboard on the home Wi-Fi — it wasn’t answering when this "
             "issue was sent, so start the rig if a link goes nowhere.")
    parts.append(f"""
    <div style="margin-top:22px;padding-top:12px;border-top:3px double {_C['ink']};text-align:center;">
      <a href="{_esc(dash)}" style="color:{_C['gilt']};font-weight:700;font-size:14px;">Open the full Creature Report → highlight reel &amp; clips</a>
      <p style="font-size:12px;margin:8px 0 0;">{rooms}</p>
      <p style="font-size:11px;color:{_C['faint']};margin:10px 0 0;line-height:1.6;">
        {_esc(reach)}<br>
        Counts are what the camera demonstrably saw — floors, not censuses.<br>
        Backyard Critter Cam · generated {_esc(bundle['generated'][:16].replace('T', ' '))}
      </p>
    </div>""")

    body = "".join(parts)
    return f"""<!doctype html><html><body style="margin:0;padding:0;background:{_C['paper']};">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" bgcolor="{_C['paper']}"><tr><td align="center" style="padding:18px 8px;">
    <table role="presentation" width="620" cellspacing="0" cellpadding="0" style="max-width:620px;width:100%;"><tr><td
      style="font-family:Georgia,'Times New Roman',serif;color:{_C['ink']};font-size:14px;">
      {body}
    </td></tr></table>
  </td></tr></table></body></html>"""


def _section(title, note) -> str:
    return (f'<h2 style="font-size:13px;letter-spacing:2px;text-transform:uppercase;'
            f'border-bottom:1px solid {_C["ink"]};padding-bottom:4px;margin:22px 0 8px;">'
            f'{_esc(title)} <span style="color:{_C["faint"]};font-weight:400;'
            f'letter-spacing:0;text-transform:none;font-size:12px;">{_esc(note)}</span></h2>')


def render_text(bundle) -> str:
    """The plain-text alternative part. Gmail scores HTML-only mail as spammier, and text keeps
    the issue readable anywhere (same reasoning as the bench-neighbors digest)."""
    d = bundle["d"]
    ed = d.get("edition") or "night"
    masthead = "THE MORNING DISPATCH" if ed == "night" else "THE EVENING DISPATCH"
    out = [masthead, d.get("title") or "", ""]
    out += compose_lede(bundle) + [""]
    for v in d.get("visit_log") or []:
        sp = _species_line(v.get("species"))
        inds = (" [" + ", ".join(_cap1(n) for n in v["individuals"]) + "]") if v.get("individuals") else ""
        out.append(f"  {_clock(v.get('start')):>8}  {sp}{inds} · {_dur_text(v.get('minutes'))}"
                   f" · {v.get('count')} obs")
    if d.get("visit_log"):
        out.append("")
    for s in d.get("species") or []:
        out.append(f"  {_name_of(s.get('species'))}: {s.get('visits')} visit"
                   f"{'' if s.get('visits') == 1 else 's'}, {_clock(s.get('first'))}–{_clock(s.get('last'))}"
                   + (" (off-hours; verify)" if s.get("surprising") else ""))
    base = bundle.get("base_url") or ""
    frag = f"dispatch/{d.get('anchor')}/{ed}" if d.get("anchor") else "dispatch"
    out += ["",
            f"Full dispatch: {_url(base, frag)}",
            f"Live camera:   {_url(base, 'live')}",
            f"The cast:      {_url(base, 'indiv')}",
            "(these reach the rig on the home Wi-Fi)",
            "",
            "Counts are floors, not censuses. Generated "
            + bundle["generated"][:16].replace("T", " ") + "."]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Sending (Resend) + the archive copy.
# ---------------------------------------------------------------------------

def resend_payload(cfg, subject, html, text, images, to=None) -> dict:
    """The exact JSON body POSTed to Resend -- a pure function so tests can hold it up to the
    light. Attachment content_id is what turns an attachment into an inline image (the HTML
    references cid:<content_id>). send_issue calls this once per recipient with to=<one address>,
    so a real send never carries more than one address in the To: header."""
    return {
        "from": cfg.email_from,
        "to": recipients(cfg, to),
        "subject": subject,
        "html": html,
        "text": text,
        "attachments": [
            {"filename": f"{cid}.jpg", "content": base64.b64encode(im["bytes"]).decode(),
             "content_type": im["mime"], "content_id": cid}
            for cid, im in images.items()
        ],
    }


def _post_issue(cfg, subject, html, text, images, to) -> str:
    """One POST, for ONE recipient; returns the email id. Raises RuntimeError with the response
    body on any non-2xx, because a silent morning-email failure would just look like a boring
    yard. Split out of send_issue so the per-reader loop has a single place to catch."""
    payload = resend_payload(cfg, subject, html, text, images, to)
    req = urllib.request.Request(
        RESEND_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": USER_AGENT,
                 "Authorization": f"Bearer {cfg.email_resend_api_key}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode() or "{}").get("id", "?")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        hint = ""
        if "1010" in body or "error code:" in body:
            # A Cloudflare block, not an API verdict -- say so, or the next reader spends the
            # morning checking a key that was never the problem.
            hint = (" -- that is a CLOUDFLARE block (not a Resend API error): the request never "
                    "reached the API. Usually the User-Agent header went missing.")
        elif e.code == 403 and "domain" in body.lower():
            hint = (" -- verify the sending domain in Resend (Domains -> add the DNS records) "
                    "and make sure email_from uses an address at that domain.")
        elif e.code == 401:
            hint = " -- check email_resend_api_key, and that it has SEND permission."
        raise RuntimeError(f"Resend refused the email ({e.code}): {body}{hint}") from e
    except urllib.error.URLError as e:
        # AFTER the HTTPError clause on purpose: HTTPError subclasses URLError. Reaching here
        # means the transport failed rather than the API answering -- DNS, refused connection,
        # TLS, a dropped link. The request never got a verdict.
        raise RuntimeError(f"Resend was unreachable: {e.reason}") from e
    except (OSError, ValueError) as e:
        # A socket timeout (OSError) or a 2xx whose body would not parse (JSONDecodeError is a
        # ValueError). Both are genuinely AMBIGUOUS: Resend may have accepted the message before
        # the read timed out, so this address is UNKNOWN rather than undelivered -- said plainly
        # here because the person acting on it is deciding whether to re-run.
        raise RuntimeError(f"Resend gave no usable verdict ({type(e).__name__}: {e}) -- this "
                           "address MAY have been delivered anyway") from e


def send_issue(cfg, subject, html, text, images, to=None) -> str:
    """Send the issue, ONE MESSAGE PER RECIPIENT, so no reader ever sees another's address.

    Returns the email id, or the ids joined by ", " when there is more than one recipient.

    The cost of that privacy is one POST -- and one re-upload of the inline images -- per reader.
    For a household list that is nothing; for a real mailing list it would be the wrong shape, and
    the right answer there is a list provider, not a loop.

    A PARTIAL failure names who did receive it. Re-running after one would send those readers a
    second copy, so that has to be a decision the reader of the error makes knowingly rather than
    a guess -- which is also why this raises instead of returning a tally."""
    addrs = recipients(cfg, to)
    if not addrs:
        raise RuntimeError("No recipients -- set email_to in config_local.py.")
    sent: list[tuple[str, str]] = []
    failed: list[tuple[str, Exception]] = []
    for addr in addrs:
        try:
            sent.append((addr, _post_issue(cfg, subject, html, text, images, addr)))
        except Exception as e:
            # Deliberately broad. _post_issue is meant to raise only RuntimeError, but if that
            # contract ever slips, the cost is not one bad message -- it is the loop aborting, so
            # the remaining readers are never attempted AND the record of who already received a
            # copy is destroyed. Catching wide keeps both promises no matter what comes out.
            failed.append((addr, e))
    if failed:
        delivered = ", ".join(a for a, _ in sent) or "nobody"
        detail = "; ".join(f"{a}: {e}" for a, e in failed)
        raise RuntimeError(
            f"Delivered to {len(sent)} of {len(addrs)} recipient(s) ({delivered}); "
            f"re-running would send those a second copy. Failed for {detail}")
    return ", ".join(i for _, i in sent)


def write_archive(cfg, bundle, images, out_path: Path | None = None) -> Path:
    """The browser-viewable copy (data: URIs). reports/ is gitignored, so issues accumulate
    locally without ever entering the public repo."""
    d = bundle["d"]
    name = f"{d.get('anchor') or 'undated'}-{d.get('edition') or 'auto'}.html"
    path = out_path or (cfg.db_path.parent / "reports" / "mail" / name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_email(bundle, images, _img_src_data(images)), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The morning wait: at this latitude a 07:00 task can fire BEFORE midwinter dawn, and
# period_digest would then (correctly) hand back the previous night. Sleeping until just past
# dawn keeps "the night's activity" meaning LAST night, all year, without moving the task.
# ---------------------------------------------------------------------------

def wait_for_dawn(cfg, now=None, max_wait_s=4 * 3600, _sleep=_time.sleep) -> float:
    """Seconds waited. No-op whenever dawn is already past (the common case all summer)."""
    now = now or datetime.now().astimezone()
    try:
        dawn, _dusk = stats._sun(cfg, now.date())
    except Exception:
        return 0.0
    target = dawn + timedelta(minutes=10)
    remaining = (target - now).total_seconds()
    if remaining <= 0:
        return 0.0
    remaining = min(remaining, max_wait_s)
    _say(f"[newsletter] night not complete yet -- waiting {round(remaining / 60)} min "
         f"for dawn ({dawn.strftime('%H:%M')}) before writing the issue.")
    waited = 0.0
    while waited < remaining:
        step = min(60.0, remaining - waited)
        _sleep(step)
        waited += step
    return waited


def email_configured(cfg) -> bool:
    return bool(recipients(cfg) and getattr(cfg, "email_from", None)
                and getattr(cfg, "email_resend_api_key", None))


_LOG_PATH = Path(__file__).resolve().parent / "logs" / "newsletter.log"


def _say(msg: str, always_log: bool = False) -> None:
    """Console when there is one, logs/newsletter.log when there isn't (or when the line is
    worth keeping either way). The scheduled task runs under pythonw.exe, where sys.stdout is
    None and a bare print() RAISES AttributeError -- backup.py only survives that because
    logging swallows emit errors. Here the fallback is explicit, so the 07:00 run both survives
    and leaves a trail."""
    if sys.stdout is not None:
        # flush: this process can then sleep for hours (wait_for_dawn), and a buffered
        # explanation of WHY is worth nothing -- it arrives with the exit.
        print(msg, flush=True)
    if always_log or sys.stdout is None:
        try:
            _LOG_PATH.parent.mkdir(exist_ok=True)
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")
        except Exception:
            pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Email the Creature Report: last night's yard, as a morning newsletter.")
    p.add_argument("--edition", choices=("night", "day", "auto"), default="night",
                   help="Which completed period to write up (default: night).")
    p.add_argument("--date", default=None, help="Anchor date YYYY-MM-DD for a back-issue.")
    p.add_argument("--to", default=None,
                   help="One-off recipient override, comma-separated (still needs from/key).")
    p.add_argument("--out", default=None, help="Write the archive copy to this path.")
    p.add_argument("--send", action="store_true",
                   help="Strict: exit non-zero if email is unconfigured or refused.")
    p.add_argument("--no-send", action="store_true", help="Render + archive only.")
    p.add_argument("--no-wait", action="store_true",
                   help="Skip the wait-for-dawn (back-issues never wait).")
    args = p.parse_args(argv)

    from config import CONFIG as cfg

    # The dawn wait exists so the 07:00 task never mails the night before by mistake. It is
    # wrong for every other way of running this: --no-send renders a preview (waiting four
    # hours to look at a file is absurd), and a pinned --date is already a completed period.
    if (args.edition == "night" and not args.date and not args.no_wait and not args.no_send):
        wait_for_dawn(cfg)

    bundle = collect_issue(cfg, edition=args.edition, date=args.date)
    d = bundle["d"]
    if d.get("empty") and not d.get("start"):
        # No period at all (empty DB / bad --date) -- nothing to render, not even a quiet issue.
        _say(f"[newsletter] nothing to write: {d.get('reason', 'no data')}")
        return 0

    images = gather_images(cfg, d, bundle.get("plate"))
    subject = compose_subject(bundle)
    html = render_email(bundle, images, _img_src_cid(images))
    text = render_text(bundle)
    path = write_archive(cfg, bundle, images, Path(args.out) if args.out else None)
    _say(f"[newsletter] issue written: {path}  ({len(images)} photo"
         f"{'' if len(images) == 1 else 's'}, subject: {subject!r})")

    if args.no_send:
        return 0
    if d.get("empty") and not getattr(cfg, "email_send_quiet", True):
        _say("[newsletter] quiet period and email_send_quiet=False -- not sending.")
        return 0
    if not email_configured(cfg):
        _say("[newsletter] email not configured -- set email_to / email_from / "
             "email_resend_api_key in config_local.py (see config_local.example.py); "
             "issue archived locally.", always_log=True)
        return 1 if args.send else 0
    try:
        mail_id = send_issue(cfg, subject, html, text, images, to=args.to)
    except Exception as e:
        _say(f"[newsletter] send FAILED: {e}", always_log=True)
        return 1
    _say(f"[newsletter] sent {d.get('anchor')} {d.get('edition')} -> "
         f"{', '.join(recipients(cfg, args.to))} (id {mail_id})", always_log=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
