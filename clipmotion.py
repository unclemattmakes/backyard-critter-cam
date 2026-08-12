"""
Phase 4 -- MOTION fingerprints from the behaviour clips.

A still says who was there; the clip says HOW THEY MOVED -- and that's both behaviour (bold
beeline to the dish vs hesitant stop-start creep) and a confound-robust shot at individual ID
(a limp or a characteristic gait reads the same from any angle, where still-appearance re-ID
failed on pose + glass). This tool turns each recorded clip (clips.py / --record-clips) into
motion tracks + a small feature fingerprint PER ANIMAL:

    sample frames -> MegaDetector boxes per sample -> NMS (kill the detector's double-boxes)
        -> greedy centre-distance association into TRACKLETS (a pair visit = two tracks)
        -> per tracklet: duration, path length, straightness (beeline=1), avg/peak speed,
           moving fraction (vs stationary, e.g. head-down eating), area trend
           (approach/retreat), and a GAIT estimate: stride cadence (Hz) from the body-bob
           periodicity while walking, with the autocorrelation strength backing it.

The RAW track (normalized positions + times) is stored as JSON alongside the features, so richer
gait work later (limp asymmetry, per-leg phase) can re-derive from the track without re-running
the detector over the video. Default sampling is EVERY frame -- the clips are ~10 fps, and gait
at a raccoon's ~1.5-2.5 strides/s needs all the temporal resolution the corpus has (Nyquist is
already tight); pass --sample-hz to subsample when you only want the coarse behaviour features.

Batch + resumable, mirroring embed.py: only clips without tracks are processed unless --redo.

  python clipmotion.py                # process all new clips (GPU), every frame
  python clipmotion.py --device cpu   # no GPU contention with the live rig
  python clipmotion.py --sample-hz 5  # cheaper coarse pass (no gait fidelity)
  python clipmotion.py --clip 2       # one clip, by clips.id
  python clipmotion.py --redo         # recompute existing tracks too
  python clipmotion.py --show         # print the stored fingerprints (no processing)
  python clipmotion.py --report       # motion features grouped by individual/visit + pair clips
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

import config
import db
from individuals import iou

# clip_tracks.model is the rig's configured detector (e.g. 'MDV6-yolov10-c') so a clip's track
# is built by the same eyes that triggered it.

# A sample counts as "moving" above this speed (normalized frame units / second). 0.04 ~ 50 px/s
# at 1280 wide -- below a slow amble, above breathing/jitter. Eyeballed against real tracks.
MOVING_SPEED = 0.04

# Tracker knobs (normalized frame units). A box may join a tracklet if its centre is within
# JUMP_BASE + JUMP_RATE * (seconds since the tracklet's last box) -- generous enough to ride out
# a ~1 s detection dropout, tight enough not to teleport between two animals across the patio.
# A tracklet starved longer than TRACK_GAP_S closes; shorter than MIN_HITS boxes is flicker.
JUMP_BASE = 0.05
JUMP_RATE = 0.25
TRACK_GAP_S = 1.5
MIN_HITS = 5

# A SUSTAINED tracklet -- enough boxes (~3 s at the clips' ~10 fps) to be a real animal-presence,
# not a detection-dropout fragment of one already-tracked animal. The detector loses a
# through-glass raccoon for >TRACK_GAP_S often enough that one 60 s visit yields one long track
# PLUS several short fragments; counting fragments as animals inflates "co-presence" badly (133
# raw vs 45 sustained on the corpus). The --report co-presence count uses this gate. Empirical
# (2026-06-11): real second animals sustain >=30 boxes; fragments cluster <=10.
SUSTAINED_HITS = 30

# "Confident enough to SAY there is an animal in this video", for clips.video_detections.
#
# NOT the detector floor: MDV6 hallucinates on dark IR background. The 2026-08-09 01:48 clip
# yielded a 91-box, 3.7 s track sitting on empty shrub at 0.25-0.50 while the frames plainly show
# nothing, so a raw box count would have badged that video "96 detections, 50%".
#
# THIS BAR ONLY LICENSES THE POSITIVE CLAIM. Measured 2026-08-10 against the 1,244 glass-door
# clips that overlap a HUMAN-CONFIRMED animal visit (ground truth: an animal is present), the
# max-confidence distribution is min 0.26 / p10 0.50 / median 0.80 -- so ANY bar that rejects the
# 0.50 phantom also rejects a fifth of the real animals:
#     bar 0.40 -> 4.0% of confirmed-animal clips called empty
#     bar 0.50 -> 9.8%
#     bar 0.60 -> 19.8%   <- and the phantom is 0.502, so it survives 0.40 and 0.50 anyway
# There is no separating threshold: confidence alone cannot tell "animal in frame" from "phantom
# on dark background". So the absence claim is NOT SHIPPED -- the dashboard says "shows an animal"
# above this bar and hedges below it, never "no animal in this video". Finding a signal that CAN
# support absence (track straightness? the refimg reference?) is open work, not a tuning job.
VIDEO_ANIMAL_MIN_CONF = 0.60

# Gait band: plausible raccoon stride cadence. Below 0.8 Hz it's milling, above 3.2 Hz it's
# detector jitter (and unresolvable anyway at the clips' ~10 fps).
GAIT_BAND = (0.8, 3.2)


def track_features(track: list) -> dict:
    """Derive the motion fingerprint from a [[t, cx, cy, w, h, conf], ...] track (normalized).
    Pure function so tests (and later gait work) can re-derive from stored JSON."""
    if len(track) < 2:
        return {"duration_s": 0.0, "path_len": 0.0, "net_disp": 0.0, "straightness": None,
                "avg_speed": None, "peak_speed": None, "moving_frac": 0.0, "area_trend": None}
    t = [p[0] for p in track]
    xy = [(p[1], p[2]) for p in track]
    duration = t[-1] - t[0]

    path = 0.0
    speeds = []
    for i in range(1, len(track)):
        dt = t[i] - t[i - 1]
        d = ((xy[i][0] - xy[i - 1][0]) ** 2 + (xy[i][1] - xy[i - 1][1]) ** 2) ** 0.5
        path += d
        if dt > 0:
            speeds.append(d / dt)
    net = ((xy[-1][0] - xy[0][0]) ** 2 + (xy[-1][1] - xy[0][1]) ** 2) ** 0.5
    moving = [s for s in speeds if s > MOVING_SPEED]

    # Box-area trend: mean of the last few hits over the first few. >1 = grew = approached the
    # camera; <1 = shrank = retreated. (Padding/edge clamping adds noise; treat as a coarse cue.)
    areas = [p[3] * p[4] for p in track]
    k = max(1, min(3, len(areas) // 2))
    a0 = sum(areas[:k]) / k
    a1 = sum(areas[-k:]) / k
    return {
        "duration_s": round(duration, 2),
        "path_len": round(path, 4),
        "net_disp": round(net, 4),
        "straightness": round(net / path, 3) if path > 1e-6 else None,
        "avg_speed": round(sum(moving) / len(moving), 4) if moving else 0.0,
        "peak_speed": round(max(speeds), 4) if speeds else None,
        "moving_frac": round(len(moving) / len(speeds), 3) if speeds else 0.0,
        "area_trend": round(a1 / a0, 3) if a0 > 1e-9 else None,
    }


def gait_features(track: list, *, band=GAIT_BAND, min_walk_s: float = 2.5,
                  min_strength: float = 0.3) -> dict:
    """Stride cadence from the body-bob: while a quadruped walks, its body (so its box centre-y)
    oscillates once per stride. Take the longest continuous WALKING run, detrend centre-y, and
    look for a periodic peak in its autocorrelation inside the plausible stride band. Returns
    {stride_hz, stride_strength, walk_s} -- honest Nones when there's no usable walking run or
    no clear periodicity (through-glass detector jitter can easily drown a ~few-pixel bob; the
    strength value says how much to trust the estimate). Pure function over the stored JSON."""
    out = {"stride_hz": None, "stride_strength": None, "walk_s": 0.0}
    if len(track) < 12:
        return out
    t = np.array([p[0] for p in track], dtype=np.float64)
    cx = np.array([p[1] for p in track], dtype=np.float64)
    cy = np.array([p[2] for p in track], dtype=np.float64)
    dt = np.diff(t)
    if not (dt > 0).any():
        return out
    med_dt = float(np.median(dt[dt > 0]))
    if med_dt <= 0:
        return out

    # Steps that are both MOVING and gap-free; the longest such run is the walking bout.
    speed = np.hypot(np.diff(cx), np.diff(cy)) / np.maximum(dt, 1e-9)
    ok = (speed > MOVING_SPEED) & (dt <= 3 * med_dt)
    runs, start = [], None
    for i, v in enumerate(ok):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))         # step run [start, i) covers samples start..i
            start = None
    if start is not None:
        runs.append((start, len(ok)))
    if not runs:
        return out
    s, e = max(runs, key=lambda r: t[r[1]] - t[r[0]])
    walk_s = float(t[e] - t[s])
    out["walk_s"] = round(walk_s, 1)
    if walk_s < min_walk_s:
        return out

    # Uniform resample of centre-y over the run, then remove the slow trend (the walk itself)
    # with a ~1.2 s moving average, keeping only the per-stride bob. np.convolve zero-pads, so
    # the trend estimate is garbage within half a window of each end -- TRIM those samples, or
    # the edge spikes (huge next to a few-pixel bob) own the autocorrelation.
    n = max(int(round(walk_s / med_dt)) + 1, 8)
    tg = np.linspace(t[s], t[e], n)
    y = np.interp(tg, t[s:e + 1], cy[s:e + 1])
    w = max(3, int(round(1.2 / med_dt)) | 1)                  # odd window
    half = w // 2
    if n - 2 * half < 12:
        return out                                            # too short once edges go
    trend = np.convolve(y, np.ones(w) / w, mode="same")
    x = (y - trend)[half:n - half]
    x -= x.mean()
    denom = float((x * x).sum())
    if denom < 1e-12:
        return out

    k_lo = max(1, int(round(1.0 / band[1] / med_dt)))         # shortest plausible stride lag
    k_hi = min(len(x) - 2, int(round(1.0 / band[0] / med_dt)))  # longest
    if k_lo >= k_hi:
        return out
    r = np.array([(x[:-k] * x[k:]).sum() / denom for k in range(k_lo, k_hi + 1)])
    peak = float(r.max())
    if peak < min_strength:
        return out
    # Smallest lag within 95% of the peak -- so a 2-stride harmonic doesn't halve the cadence.
    k_star = k_lo + int(np.argmax(r >= 0.95 * peak))
    out["stride_hz"] = round(1.0 / (k_star * med_dt), 2)
    out["stride_strength"] = round(peak, 2)
    return out


# ---------------------------------------------------------------------------
# Multi-animal tracking: NMS + greedy centre-distance association.
# ---------------------------------------------------------------------------

def _corners(b):
    cx, cy, w, h = b[0], b[1], b[2], b[3]
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def nms_boxes(boxes: list, iou_max: float = 0.6) -> list:
    """Suppress the detector's double-boxes: keep boxes highest-confidence-first, dropping any
    that overlaps a kept box at IoU >= iou_max. Two huddled-but-distinct raccoons sit below
    ~0.45 IoU (measured on the pair corpus), so real pairs survive."""
    kept: list = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        if all(iou(_corners(b), _corners(k)) < iou_max for k in kept):
            kept.append(b)
    return kept


def build_tracks(samples: list, *, jump_base: float = JUMP_BASE, jump_rate: float = JUMP_RATE,
                 max_gap_s: float = TRACK_GAP_S, min_hits: int = MIN_HITS) -> list:
    """Associate per-frame boxes into per-animal tracklets. `samples` = [(t_s, [boxes]), ...]
    time-ordered, each box (cx, cy, w, h, conf) normalized 0..1 (NMS'd). Greedy nearest-centre
    matching: closest (tracklet, box) pairs link first, within a jump budget that grows with the
    time since the tracklet's last box (rides out detection dropouts). Unmatched boxes start new
    tracklets; a tracklet starved past max_gap_s closes. Returns [[ [t,cx,cy,w,h,conf], ...], ...]
    longest-first, dropping flicker shorter than min_hits. Pure function -> unit-testable."""
    active: list = []
    closed: list = []
    for t, boxes in samples:
        still = []
        for tr in active:
            (closed if t - tr[-1][0] > max_gap_s else still).append(tr)
        active = still

        pairs = []
        for ai, tr in enumerate(active):
            lt, lx, ly = tr[-1][0], tr[-1][1], tr[-1][2]
            budget = jump_base + jump_rate * (t - lt)
            for bi, b in enumerate(boxes):
                d = ((b[0] - lx) ** 2 + (b[1] - ly) ** 2) ** 0.5
                if d <= budget:
                    pairs.append((d, ai, bi))
        pairs.sort(key=lambda p: p[0])
        used_a, used_b = set(), set()
        for d, ai, bi in pairs:
            if ai in used_a or bi in used_b:
                continue
            used_a.add(ai)
            used_b.add(bi)
            active[ai].append([t, *boxes[bi]])
        for bi, b in enumerate(boxes):
            if bi not in used_b:
                active.append([[t, *b]])
    closed += active
    tracks = [tr for tr in closed if len(tr) >= min_hits]
    tracks.sort(key=len, reverse=True)
    return tracks


def extract_tracks(clip_path, detector, sample_hz: float | None):
    """Sample a clip's frames (every frame when sample_hz is None -- the default, for gait
    fidelity at the corpus' ~10 fps), detect in each, NMS, and associate into per-animal
    tracklets. Returns (tracks, n_samples, raw); (None, 0, raw) if the video can't be opened.

    `raw` answers a different question from the tracklets -- "is an animal actually VISIBLE in
    these frames?" -- and is what gets stamped onto clips.video_detections / video_max_conf (see
    the clips migration in db.py). Two deliberate choices:

      * counted before track association, because a brief or partly-occluded animal can fail
        MIN_HITS and produce no tracklet while plainly being in the video; tracklets measure
        sustained motion, not presence;
      * `n_boxes` counts only boxes at or above VIDEO_ANIMAL_MIN_CONF, while `max_conf` keeps the
        best score seen at ANY confidence. Raw counts are actively misleading here -- MDV6 tracks
        phantoms across dark IR background for seconds at a time -- so the count carries the bar
        and the max carries the honesty about what the bar rejected."""
    import cv2
    raw = {"n_boxes": 0, "max_conf": None}
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None, 0, raw
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not (fps and fps > 0 and np.isfinite(fps)):   # some containers report 0.0 or NaN here
            fps = 15.0
        step = 1 if not sample_hz else max(1, round(fps / sample_hz))
        samples = []
        n_samples = 0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                n_samples += 1
                try:
                    dets = detector.detect(frame)
                except Exception:  # one bad frame never kills the clip
                    dets = []
                h, w = frame.shape[:2]
                boxes = []
                for d in dets:
                    x1, y1, x2, y2 = d.bbox
                    boxes.append((round(((x1 + x2) / 2) / w, 4), round(((y1 + y2) / 2) / h, 4),
                                  round((x2 - x1) / w, 4), round((y2 - y1) / h, 4),
                                  round(d.confidence, 3)))
                kept = nms_boxes(boxes)
                for b in kept:
                    if b[4] >= VIDEO_ANIMAL_MIN_CONF:
                        raw["n_boxes"] += 1
                    if raw["max_conf"] is None or b[4] > raw["max_conf"]:
                        raw["max_conf"] = b[4]
                samples.append((round(idx / fps, 3), kept))
            idx += 1
    finally:
        cap.release()
    return build_tracks(samples), n_samples, raw


def fingerprint_line(clip_id, idx, feats, n_hits, n_samples) -> str:
    s = feats
    def f(v, fmt="{:.2f}"):
        return fmt.format(v) if v is not None else "--"
    stride = (f"{s['stride_hz']:.1f}Hz({s['stride_strength']:.2f})"
              if s.get("stride_hz") and s.get("stride_strength") is not None else "--")
    return (f"  clip #{clip_id:<4}.{idx} {n_hits}/{n_samples} hits  "
            f"dur {f(s['duration_s'], '{:.1f}')}s  path {f(s['path_len'])}  "
            f"straight {f(s['straightness'])}  v_avg {f(s['avg_speed'])}  "
            f"moving {f(s['moving_frac'])}  area x{f(s['area_trend'])}  "
            f"stride {stride}  walk {f(s.get('walk_s'), '{:.0f}')}s")


def show_tracks(conn) -> int:
    rows = conn.execute(
        """SELECT t.clip_id, t.track_idx, t.n_samples, t.n_hits, t.duration_s, t.path_len,
                  t.net_disp, t.straightness, t.avg_speed, t.peak_speed, t.moving_frac,
                  t.area_trend, t.stride_hz, t.stride_strength, t.walk_s, c.started_at
           FROM clip_tracks t JOIN clips c ON c.id = t.clip_id
           WHERE t.n_hits > 0 ORDER BY t.clip_id, t.track_idx"""
    ).fetchall()
    if not rows:
        print("No motion tracks yet -- run clipmotion.py after recording clips (--record-clips).")
        return 0
    print(f"{len(rows)} motion fingerprint(s):")
    for r in rows:
        feats = {k: r[k] for k in ("duration_s", "path_len", "net_disp", "straightness",
                                   "avg_speed", "peak_speed", "moving_frac", "area_trend",
                                   "stride_hz", "stride_strength", "walk_s")}
        print(fingerprint_line(r["clip_id"], r["track_idx"], feats, r["n_hits"], r["n_samples"])
              + f"  [{(r['started_at'] or '')[5:16]}]")
    return 0


# ---------------------------------------------------------------------------
# The report: motion features grouped by who (visit / individual), + pair clips.
# ---------------------------------------------------------------------------

def _overlap_s(a0, a1, b0, b1) -> float:
    """Seconds of overlap between ISO spans [a0,a1] and [b0,b1]; 0 when disjoint/unparseable."""
    try:
        s = max(datetime.fromisoformat(a0), datetime.fromisoformat(b0))
        e = min(datetime.fromisoformat(a1), datetime.fromisoformat(b1))
        return max(0.0, (e - s).total_seconds())
    except (ValueError, TypeError):
        return 0.0


def tracks_with_visits(conn, species: str = "raccoon"):
    """Every stored tracklet joined (by time overlap on the same source) to its visit. Returns
    [{...track features, clip_id, started_at, visit_id, individual_id}, ...]. Visits renumber on
    rebuild, so this join is computed fresh per call rather than stored."""
    visits = conn.execute(
        "SELECT id, source, individual_id, started_at, ended_at FROM visits WHERE species = ?",
        (species,)).fetchall()
    rows = conn.execute(
        """SELECT t.*, c.started_at AS c_start, c.ended_at AS c_end, c.source AS c_source
           FROM clip_tracks t JOIN clips c ON c.id = t.clip_id
           WHERE t.n_hits >= ? ORDER BY c.started_at, t.clip_id, t.track_idx""",
        (MIN_HITS,)).fetchall()
    out = []
    for r in rows:
        best, best_ov = None, 0.0
        for v in visits:
            if v["source"] != r["c_source"]:
                continue
            ov = _overlap_s(r["c_start"], r["c_end"] or r["c_start"],
                            v["started_at"], v["ended_at"])
            if ov > best_ov:
                best, best_ov = v, ov
        if best is None:
            continue                      # clip didn't overlap any visit of this species
        out.append({**{k: r[k] for k in r.keys()},
                    "visit_id": best["id"], "individual_id": best["individual_id"]})
    return out


def _med(vals):
    vals = sorted(v for v in vals if v is not None)
    return vals[len(vals) // 2] if vals else None


def report(conn, species: str = "raccoon") -> int:
    rows = tracks_with_visits(conn, species)
    if not rows:
        print(f"No tracklets overlap any {species} visit yet -- run clipmotion.py first "
              f"(and visits.py after new capture).")
        return 0

    groups = defaultdict(list)
    for r in rows:
        groups[r["individual_id"] or "(unconfirmed)"].append(r)
    print(f"Motion fingerprints by individual ({len(rows)} tracklet(s) over "
          f"{len({r['visit_id'] for r in rows})} {species} visit(s)):\n")
    print(f"  {'who':18} {'tracks':>6} {'v_avg':>7} {'straight':>8} {'moving':>7} "
          f"{'stride Hz':>9} {'gait found':>10}")
    for who, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        gait = [r for r in rs if r["stride_hz"] is not None]
        print(f"  {who:18} {len(rs):>6} {_med([r['avg_speed'] for r in rs]) or 0:>7.3f} "
              f"{_med([r['straightness'] for r in rs]) or 0:>8.2f} "
              f"{_med([r['moving_frac'] for r in rs]) or 0:>7.2f} "
              f"{(_med([r['stride_hz'] for r in gait]) or 0):>9.2f} "
              f"{len(gait):>6}/{len(rs)}")

    # Pair clips: >= 2 SUSTAINED tracklets in ONE clip = two animals under identical conditions
    # -- the controlled comparison (same light, same glass, same moment). Sustained-only, because
    # detection dropouts fragment one animal's track into a long track + short stubs; counting
    # those as animals inflates co-presence ~3x (see SUSTAINED_HITS).
    by_clip = defaultdict(list)
    for r in rows:
        by_clip[r["clip_id"]].append(r)
    raw_multi = sum(1 for rs in by_clip.values() if len(rs) >= 2)
    pairs = {cid: [r for r in rs if r["n_hits"] >= SUSTAINED_HITS]
             for cid, rs in by_clip.items()}
    pairs = {cid: rs for cid, rs in pairs.items() if len(rs) >= 2}
    print(f"\nCo-presence: {len(pairs)} clip(s) with 2+ SUSTAINED tracks "
          f"(>={SUSTAINED_HITS} boxes each) -- same-conditions animal pairs. "
          f"({raw_multi} clips had 2+ raw tracks; the rest is dropout fragmentation.)")
    for cid, rs in sorted(pairs.items())[:20]:
        when = (rs[0]["c_start"] or "")[5:16]
        print(f"  clip #{cid} [{when}] visit #{rs[0]['visit_id']}:")
        for r in sorted(rs, key=lambda r: -r["n_hits"]):
            stride = (f"{r['stride_hz']:.1f}Hz({r['stride_strength']:.2f})"
                      if r["stride_hz"] else "--")
            print(f"    track {r['track_idx']}: {r['n_hits']:>4} boxes  dur "
                  f"{r['duration_s'] or 0:>5.1f}s  v_avg {r['avg_speed'] or 0:.3f}  "
                  f"straight {r['straightness'] if r['straightness'] is not None else 0:.2f}  "
                  f"stride {stride}")
    if len(pairs) > 20:
        print(f"  ... and {len(pairs) - 20} more co-presence clip(s).")
    return 0


def _is_placeholder(iid: str) -> bool:
    """True for a reid auto-cluster id like 'raccoon_c01' (not a human-confirmed name)."""
    return bool(iid) and "_c" in iid and iid.rsplit("_c", 1)[-1].isdigit()


def _species_with_named_visits(conn) -> list:
    """Species that have at least one visit assigned to a confirmed (non-placeholder) individual --
    the species worth linking tracks for (raccoon today; cats/others as the cast grows)."""
    rows = conn.execute(
        "SELECT DISTINCT species, individual_id FROM visits WHERE individual_id IS NOT NULL").fetchall()
    return sorted({r["species"] for r in rows if not _is_placeholder(r["individual_id"])})


def link_tracks_to_individuals(conn, species: str = "raccoon", *,
                               sustained: int = SUSTAINED_HITS, dry_run: bool = False) -> dict:
    """Backfill clip_tracks.individual_id for the UNAMBIGUOUS solo clips: a clip whose SINGLE
    sustained tracklet overlaps (in time) a visit assigned to a confirmed, NAMED individual gets
    that name written to the tracklet (individual_source 'overlap'). This turns the dark motion
    fingerprints into per-individual behaviour data without guessing -- pair clips (2+ sustained
    tracks) and placeholder/unnamed visits are deliberately LEFT for the manual un-blend queue,
    and a human un-blend label is never overwritten. Only HUMAN-confirmed visit names qualify:
    a nightly auto-assigned name (individual_source 'auto' on the visit's crops) is a prediction,
    not ground truth, so behaviour links wait until the human promotes it. Idempotent +
    resumable. Returns a summary."""
    rows = tracks_with_visits(conn, species)         # each track + its best-overlap visit's individual
    human = db.confirmed_visit_labels(conn, species)  # {visit_id: name}, human-confirmed only
    by_clip = defaultdict(list)
    for r in rows:
        by_clip[r["clip_id"]].append(r)
    assigned, skip = 0, Counter()
    for rs in by_clip.values():
        sustained_tracks = [r for r in rs if (r["n_hits"] or 0) >= sustained]
        if len(sustained_tracks) != 1:
            skip["not_solo"] += 1                     # 0 sustained, or a pair -> leave for un-blend
            continue
        tr = sustained_tracks[0]
        iid = tr["individual_id"]                     # the overlapping visit's individual
        if not iid:
            skip["visit_unnamed"] += 1
            continue
        if _is_placeholder(iid):
            skip["visit_placeholder"] += 1            # raccoon_c01 -- not a confirmed name
            continue
        if human.get(tr["visit_id"]) != iid:
            skip["visit_not_human_confirmed"] += 1    # auto/stale name -- not behaviour ground truth
            continue
        if tr["individual_source"] == "human":
            skip["human_locked"] += 1                 # never clobber a manual un-blend
            continue
        if not dry_run:
            db.set_clip_track_individual(conn, [tr["id"]], iid, source="overlap")
        assigned += 1
    return {"species": species, "assigned": assigned, "clips_seen": len(by_clip),
            "skipped": dict(skip)}


def link_all(conn, *, dry_run: bool = False, species: str | None = None) -> int:
    """Run the solo-overlap backfill for every species that has named individuals (or just one)."""
    species_list = [species] if species else (_species_with_named_visits(conn) or ["raccoon"])
    total = 0
    for sp in species_list:
        res = link_tracks_to_individuals(conn, sp, dry_run=dry_run)
        total += res["assigned"]
        print(f"  {sp}: {'would link' if dry_run else 'linked'} {res['assigned']} solo "
              f"tracklet(s) over {res['clips_seen']} clip(s); skipped {res['skipped']}")
    verb = "Would link" if dry_run else "Linked"
    print(f"{verb} {total} solo tracklet(s) to named individuals (individual_source 'overlap'); "
          f"pair/unnamed clips left for the un-blend queue.")
    return 0


def backfill_scan(conn, dry_run: bool = False) -> int:
    """Give clips that already have tracks their video_detections / video_max_conf, derived from
    the stored track JSON rather than a fresh detector run.

    Why this exists: `clips_needing_tracks` deliberately skips clips processed before the scan
    stamp existed, so without a backfill ~90% of the corpus would read "video not checked"
    forever and the badge would be useless on everything but the newest clips. Re-detecting them
    all is not an option -- 304 clips took 10,600 s, so the full 3,474 would be well over a day.

    WHAT IS AND ISN'T EXACT. The track JSON holds every box that ASSOCIATED into a tracklet, with
    its confidence, from the same detector over the same frames -- so for those boxes the numbers
    are exact. What it cannot see is a detection too brief to form a track (MIN_HITS). That makes
    the derived count CONSERVATIVE in the one direction that matters least here: a clip whose
    animal was tracked is reported correctly, and a clip with no track at all is not touched by
    this path at all (it has no rows, so it goes through the real scan queue instead). The
    residual error is a clip with tracks whose ONLY strong box never associated, which the track
    builder makes unlikely by construction."""
    rows = conn.execute(
        """SELECT c.id, t.track FROM clips c JOIN clip_tracks t ON t.clip_id = c.id
           WHERE c.video_scanned_at IS NULL AND c.pruned_at IS NULL""").fetchall()
    per_clip: dict = {}
    for r in rows:
        try:
            pts = json.loads(r["track"]) or []
        except (TypeError, ValueError):
            continue
        n, best = per_clip.get(r["id"], (0, None))
        for p in pts:
            if len(p) < 6:
                continue
            conf = p[5]
            if conf >= VIDEO_ANIMAL_MIN_CONF:
                n += 1
            if best is None or conf > best:
                best = conf
        per_clip[r["id"]] = (n, best)

    strong = sum(1 for n, _ in per_clip.values() if n)
    print(f"backfill: {len(per_clip)} clip(s) with tracks but no scan stamp; "
          f"{strong} show an animal at >= {VIDEO_ANIMAL_MIN_CONF:.2f}, "
          f"{len(per_clip) - strong} do not.")
    if dry_run:
        print("dry run -- nothing written.")
        return 0
    for cid, (n, best) in per_clip.items():
        db.set_clip_video_scan(conn, cid, n_boxes=n, max_conf=best)
    print(f"wrote {len(per_clip)} clip(s).")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 4: motion fingerprints from behaviour clips.")
    p.add_argument("--device", default=config.CONFIG.device, choices=["cuda", "cpu", "auto"],
                   help="Detector device (cpu avoids GPU contention with the live rig).")
    p.add_argument("--sample-hz", type=float, default=None,
                   help="Detector samples per second of clip (default: EVERY frame, for gait).")
    p.add_argument("--clip", type=int, default=None, help="Process one clip by its clips.id.")
    p.add_argument("--redo", action="store_true", help="Recompute clips that already have tracks.")
    p.add_argument("--show", action="store_true", help="Print stored fingerprints and exit.")
    p.add_argument("--report", action="store_true",
                   help="Motion features grouped by individual/visit + pair-clip comparison.")
    p.add_argument("--backfill-scan", action="store_true",
                   help="Fill clips.video_detections/video_max_conf for clips that ALREADY have "
                        "tracks, deriving them from the stored track JSON instead of re-running "
                        "the detector (no GPU, seconds not hours).")
    p.add_argument("--link", action="store_true",
                   help="Backfill clip_tracks.individual_id from solo-clip/visit overlap (no GPU). "
                        "Run after extraction (and after naming visits) to feed the two-axis readout.")
    p.add_argument("--dry-run", action="store_true",
                   help="With --link: report what would be assigned, write nothing.")
    p.add_argument("--species", default="raccoon", help="Species for --report (default raccoon).")
    p.add_argument("--visit-species", default=None, metavar="SPECIES",
                   help="Only process clips overlapping a visit of this species (e.g. raccoon) "
                        "-- prioritizes the corpus that matters; the rest stays queued for the "
                        "next plain run (the batch is resumable).")
    args = p.parse_args()

    cfg = config.CONFIG
    conn = db.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    try:
        if args.show:
            return show_tracks(conn)
        if args.report:
            return report(conn, args.species)
        if args.backfill_scan:
            return backfill_scan(conn, dry_run=args.dry_run)
        if args.link:
            return link_all(conn, dry_run=args.dry_run)

        model = cfg.model_version
        if args.clip is not None:
            rows = conn.execute("SELECT id, clip_path, fps FROM clips WHERE id = ?",
                                (args.clip,)).fetchall()
        elif args.redo:
            # Soft-pruned clips have no video left to re-extract from; their existing tracks stay.
            rows = conn.execute("SELECT id, clip_path, fps FROM clips "
                                "WHERE pruned_at IS NULL ORDER BY id").fetchall()
        else:
            rows = db.clips_needing_tracks(conn, model)
        if args.visit_species:
            visits = conn.execute(
                "SELECT source, started_at, ended_at FROM visits WHERE species = ?",
                (args.visit_species,)).fetchall()
            spans = conn.execute("SELECT id, source, started_at, ended_at FROM clips").fetchall()
            span_of = {s["id"]: s for s in spans}
            def _overlaps(clip_id):
                c = span_of.get(clip_id)
                return c is not None and any(
                    v["source"] == c["source"] and _overlap_s(
                        c["started_at"], c["ended_at"] or c["started_at"],
                        v["started_at"], v["ended_at"]) > 0 for v in visits)
            before = len(rows)
            rows = [r for r in rows if _overlaps(r["id"])]
            print(f"--visit-species {args.visit_species}: {len(rows)} of {before} queued clips "
                  f"overlap a {args.visit_species} visit.")
        if not rows:
            print("Nothing to process -- every clip already has motion tracks "
                  "(record more with --record-clips, or --redo).")
            return 0

        from detector import Detector
        rate = f"{args.sample_hz:g} samples/s" if args.sample_hz else "every frame"
        print(f"Building motion tracks for {len(rows)} clip(s) "
              f"({model} on {args.device}, {rate})...")
        detector = Detector(model, args.device, cfg.min_confidence, classes=cfg.detect_classes)
        t0 = time.time()
        done = 0
        for r in rows:
            clip_path = db.crop_abspath(r["clip_path"])
            if not clip_path.exists():
                print(f"  clip #{r['id']}: file missing ({r['clip_path']}) -- skipped.")
                continue
            tracks, n_samples, raw = extract_tracks(clip_path, detector, args.sample_hz)
            if tracks is None:
                print(f"  clip #{r['id']}: could not open video -- skipped.")
                continue
            tracklets = []
            for tr in tracks:
                feats = track_features(tr) | gait_features(tr)
                tracklets.append({"track_json": json.dumps(tr), "n_hits": len(tr),
                                  "features": feats})
            try:
                db.insert_clip_tracks(conn, clip_id=r["id"], model=model, n_samples=n_samples,
                                      tracklets=tracklets)
            except sqlite3.IntegrityError:
                # The rolling disk budget (clips.prune_clips) deleted this clip mid-batch.
                print(f"  clip #{r['id']}: pruned while processing -- skipped.")
                continue
            # Stamp what the detector saw in THIS VIDEO. Two things fall out of recording it:
            # the dashboard can stop implying a clip shows what its trigger STILL showed, and a
            # genuinely empty clip finally has a marker -- without one it has no clip_tracks
            # rows, is indistinguishable from "never processed", and gets re-detected on every
            # nightly run forever (measured 2026-08-10: 154 trail-cam clips in that state).
            db.set_clip_video_scan(conn, r["id"], n_boxes=raw["n_boxes"],
                                   max_conf=raw["max_conf"])
            for idx, t in enumerate(tracklets):
                print(fingerprint_line(r["id"], idx, t["features"], t["n_hits"], n_samples))
            if not tracklets:
                seen = (f"{raw['n_boxes']} loose box(es), best {raw['max_conf']:.2f}"
                        if raw["n_boxes"] else "NOTHING in the video")
                print(f"  clip #{r['id']:<4}   no animal tracked ({n_samples} samples) -- {seen}")
            done += 1
        print(f"\nDone. {done} clip(s) in {time.time() - t0:.0f}s. "
              f"View any time: python clipmotion.py --show   |   --report")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
