"""
The condensed HIGHLIGHT REEL -- "last night in about a minute".

The Dispatch used to offer the period's clips back-to-back (24 full clips, ~9 minutes on a busy
night); nobody watches that daily. This module instead PLANS a short cut -- the best few seconds
of the best few clips -- and STITCHES it into one cached, shareable H.264 mp4 with ffmpeg.

Planning (pure data, fast, no ffmpeg):
  * candidates = the period's clips that caught a real visitor (same non-critter filter as the
    digest), each scored by its best still (stats._shot_score) x how much the animal actually
    MOVED (clip_tracks.moving_frac) x a rarity/named-individual lift;
  * near-identical bursts collapse (four clips of the same raccoon between 1:25 and 1:28 become
    one moment) -- the fix for the reel being 10 interchangeable raccoon clips;
  * every species (and every named individual) present in the period gets at least its best
    moment before anything else fills the budget, so the one opossum never loses its slot to the
    fifteenth raccoon;
  * each chosen clip contributes one ~6 s segment, placed where its motion tracks say the action
    is (fallback: centred on its busiest detections), and the final cut runs in wall-clock order.

Building (ffmpeg, background thread, cached):
  * segments are cut with an accurate input seek and re-encoded to uniform H.264 720p, then
    concatenated stream-copy into clips/reels/reel_<anchor>_<edition>_<hash>.mp4 (+ .json
    manifest with per-segment chapter offsets for the player's filmstrip);
  * the hash covers the exact cut list, so a finished period builds once and is instant after;
    /media/ serves the result like any other clip (H.264 -> no transcode).

`reel_status()` is the single web entry point: plan (fast) -> ready | building | empty |
unavailable | failed, kicking off at most one background build per reel.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
import time as _time
from datetime import timedelta
from pathlib import Path

import db
import stats

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")

# Every segment is re-encoded to THIS fixed rate. The source clips play at whatever rate the
# capture loop measured (9.9 fps one night, 10.4 the next), which gives each cut segment its own
# timebase -- and stream-copy concat across mixed timebases mangles durations (a 72 s reel came
# out as 44 s of fast-forward). One fixed CFR rate makes the copy-concat exact; duplicating a
# ~10 fps source up to 24 fps is visually identical and the dupes cost x264 almost nothing.
_REEL_FPS = 24

SEG_S = 6.0            # seconds each chosen moment contributes
MIN_SEG_S = 2.0        # clips shorter than this aren't worth a slot
TARGET_S = 75.0        # aim for about this much total reel
MAX_SEGMENTS = 14      # hard cap on moments
BURST_GAP_MIN = 8.0    # same-species clips closer than this collapse into one burst
BURST_EXTRA_MIN = 25.0 # a burst longer than this earns a second moment (a long feeding session)
KEEP_REELS = 40        # built reels kept on disk (older ones deleted after a successful build)
_POSTER_EVERY_S = 3    # sample the finished reel this often when choosing its poster frame
_PLAN_VERSION = 2      # bump to invalidate every cached reel (encoder/planner changes)

# key -> (state, monotonic_ts); state is 'building' or 'failed: <reason>'. Failed builds retry
# after _RETRY_S rather than wedging until a restart.
_BUILDS: dict = {}
_BUILD_LOCK = threading.Lock()
_RETRY_S = 600


def _reels_dir(cfg) -> Path:
    return cfg.clips_dir / "reels"


def _label_of(r) -> str:
    return (r["species"] or r["detection_class"] or "").lower()


def plan_reel(cfg, edition="auto", date=None, now=None) -> dict | None:
    """The cut list for one completed period: {edition, anchor, title, start, end, segments:[...]}
    with each segment {clip_path(abs), offset_s, seconds, start(iso wall), species, individuals,
    thumb, score}. None when there's no such period; segments=[] when it has no usable clips."""
    chosen = stats.resolve_period(cfg, edition, now, date)
    if chosen is None:
        return None
    start, end, label, anchor = chosen

    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return None
    try:
        pad = timedelta(minutes=10)
        clips = conn.execute(
            "SELECT id, source, clip_path, started_at, ended_at, fps, frame_count "
            "FROM clips WHERE started_at >= ? AND started_at <= ? AND pruned_at IS NULL "
            "ORDER BY started_at",
            ((start - pad).isoformat(), end.isoformat())).fetchall()
        dets = conn.execute(
            "SELECT timestamp, source, detection_class, species, confidence, species_confidence, "
            "crop_path, crop_quality, individual_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, "
            "frame_w, frame_h FROM detections WHERE timestamp >= ? AND timestamp <= ? "
            "ORDER BY timestamp",
            ((start - pad).isoformat(), end.isoformat())).fetchall()
        try:
            tracks = conn.execute(
                "SELECT clip_id, track, moving_frac, n_hits FROM clip_tracks "
                "WHERE clip_id IN (SELECT id FROM clips WHERE started_at >= ? AND started_at <= ?)",
                ((start - pad).isoformat(), end.isoformat())).fetchall()
        except Exception:
            tracks = []                    # an older DB without clip_tracks
    finally:
        conn.close()

    rows = []
    for r in dets:
        if _label_of(r) in stats._NON_CRITTER:
            continue
        dt = db.parse_local(r["timestamp"])
        if dt is None or not (start <= dt < end):
            continue
        rows.append({
            "dt": dt, "source": r["source"], "label": r["species"] or r["detection_class"],
            "confidence": r["confidence"] or 0.0, "species_confidence": r["species_confidence"],
            "crop_path": r["crop_path"], "crop_quality": r["crop_quality"],
            "individual_id": r["individual_id"],
            "bbox_x1": r["bbox_x1"], "bbox_y1": r["bbox_y1"], "bbox_x2": r["bbox_x2"],
            "bbox_y2": r["bbox_y2"], "frame_w": r["frame_w"], "frame_h": r["frame_h"],
        })

    tracks_by_clip: dict = {}
    for t in tracks:
        tracks_by_clip.setdefault(t["clip_id"], []).append(t)

    period_sp = {r["label"] for r in rows}
    sp_counts = {sp: sum(1 for r in rows if r["label"] == sp) for sp in period_sp}
    dominant = max(sp_counts, key=sp_counts.get) if sp_counts else None
    pre_roll = float(getattr(cfg, "clip_pre_roll_s", 0.0) or 0.0)

    # --- one candidate per clip that caught a real visitor ---
    cands = []
    for c in clips:
        sdt = db.parse_local(c["started_at"])
        if sdt is None or sdt > end:
            continue
        edt = db.parse_local(c["ended_at"]) if c["ended_at"] else None
        if edt is not None and edt < start:
            continue
        dur = (c["frame_count"] / c["fps"]) if (c["fps"] and c["frame_count"]) else \
              ((edt - sdt).total_seconds() if edt else 0.0)
        if dur < MIN_SEG_S:
            continue
        inside = [r for r in rows if r["source"] == c["source"]
                  and sdt - timedelta(seconds=pre_roll + 1) <= r["dt"] <= (edt or sdt) + timedelta(seconds=2)]
        if not inside:
            continue
        best = max(inside, key=stats._shot_score)
        species = max(set(r["label"] for r in inside),
                      key=lambda sp: sum(1 for r in inside if r["label"] == sp))
        named = sorted({r["individual_id"] for r in inside
                       if r["individual_id"] and not
                       ("_c" in r["individual_id"] and r["individual_id"].rsplit("_c", 1)[-1].isdigit())})
        ctracks = tracks_by_clip.get(c["id"], [])
        moving = max((t["moving_frac"] or 0.0) for t in ctracks) if ctracks else None
        seg = min(SEG_S, dur)
        off = _action_offset(ctracks, inside, sdt, pre_roll, dur, seg)
        score = stats._shot_score(best)
        score *= 0.7 + 0.8 * (moving if moving is not None else 0.35)   # favour actual motion
        if named:
            score *= 1.2                                        # a named regular on camera
        if dominant and species != dominant:
            score *= 1.3                                        # the night's rarer guests
        # Thumb: the best still that falls INSIDE the chosen window, else the clip's best.
        w0 = sdt + timedelta(seconds=off - pre_roll)
        w1 = w0 + timedelta(seconds=seg + 2)
        inwin = [r for r in inside if w0 <= r["dt"] <= w1]
        thumb = max(inwin, key=stats._shot_score) if inwin else best
        cands.append({
            "clip_id": c["id"], "clip_path": c["clip_path"], "sdt": sdt,
            "offset_s": round(off, 2), "seconds": round(seg, 2),
            "species": species, "individuals": named, "n_dets": len(inside),
            "thumb": stats._web(thumb["crop_path"]), "score": score,
        })

    segments = _select(cands) if cands else []
    for s in segments:
        s["start"] = (s["sdt"] + timedelta(seconds=max(0.0, s["offset_s"] - pre_roll))).isoformat()
    return {
        "edition": label, "anchor": anchor.isoformat(),
        "title": stats._title_for(label, anchor, (now or stats._local_now()).date()),
        "start": start.isoformat(), "end": end.isoformat(),
        "segments": segments, "n_source_clips": len(cands),
    }


def _action_offset(ctracks, inside, sdt, pre_roll, dur, seg) -> float:
    """Where the action is, as seconds from the START OF THE FILE. Preferred: slide a `seg`
    window over the clip's motion-track trajectories (t is file-relative) maximizing distance
    covered, weighted toward bigger boxes (a closer animal). Fallback: centre on the median
    detection (wall time -> file time via started_at, shifted by the pre-roll the file opens
    with). Always clamped inside the file."""
    hi = max(0.0, dur - seg)
    samples = []
    for t in ctracks:
        try:
            pts = json.loads(t["track"] or "[]")
        except (TypeError, ValueError):
            continue
        for i in range(1, len(pts)):
            t0, x0, y0 = pts[i - 1][0], pts[i - 1][1], pts[i - 1][2]
            t1, x1, y1, w1, h1 = pts[i][0], pts[i][1], pts[i][2], pts[i][3], pts[i][4]
            if t1 <= t0:
                continue
            step = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            weight = 0.5 + min(1.0, (w1 * h1) * 6.0)       # bigger box = closer = better viewing
            samples.append((t0, step * weight))
    if samples:
        samples.sort()
        best_t, best_v = 0.0, -1.0
        i0 = 0
        acc = 0.0
        for i1 in range(len(samples)):
            acc += samples[i1][1]
            while samples[i1][0] - samples[i0][0] > seg:
                acc -= samples[i0][1]
                i0 += 1
            if acc > best_v:
                best_v, best_t = acc, samples[i0][0]
        return min(max(0.0, best_t), hi)
    if inside:
        mid = sorted(r["dt"] for r in inside)[len(inside) // 2]
        off = (mid - sdt).total_seconds() + pre_roll - seg / 2.0
        return min(max(0.0, off), hi)
    return min(max(0.0, dur / 2.0 - seg / 2.0), hi)


def _select(cands: list) -> list:
    """Pick the reel's moments: collapse same-species bursts, guarantee every species and every
    named individual its best moment, then fill by score -- long bursts may earn a second moment.
    Returns the chosen candidates in wall-clock order."""
    by_time = sorted(cands, key=lambda c: c["sdt"])
    bursts = []
    for c in by_time:
        b = bursts[-1] if bursts else None
        if (b and b["species"] == c["species"]
                and (c["sdt"] - b["members"][-1]["sdt"]).total_seconds() <= BURST_GAP_MIN * 60):
            b["members"].append(c)
        else:
            bursts.append({"species": c["species"], "members": [c]})
    for b in bursts:
        b["members"].sort(key=lambda c: c["score"], reverse=True)
        b["span_min"] = abs((b["members"][0]["sdt"] - b["members"][-1]["sdt"]).total_seconds()) / 60.0

    chosen: list = []

    def _budget_left():
        return (len(chosen) < MAX_SEGMENTS
                and sum(c["seconds"] for c in chosen) < TARGET_S)

    def _take(c):
        if c is not None and c not in chosen and _budget_left():
            chosen.append(c)

    # 1. every species' best moment (rarest first, so they can't be crowded out) ...
    species = sorted({c["species"] for c in cands},
                     key=lambda sp: sum(c["n_dets"] for c in cands if c["species"] == sp))
    for sp in species:
        _take(max((c for c in cands if c["species"] == sp), key=lambda c: c["score"], default=None))
    # 2. ... every named individual's best moment ...
    named = {n for c in cands for n in c["individuals"]}
    for n in sorted(named):
        _take(max((c for c in cands if n in c["individuals"]), key=lambda c: c["score"], default=None))
    # 3. ... then burst leaders by score, then SECOND moments from the busier/longer sessions
    # (a 45-minute feeding block reads wrong as one 6 s beat), while the budget lasts.
    leaders = sorted((b["members"][0] for b in bursts), key=lambda c: c["score"], reverse=True)
    for c in leaders:
        _take(c)
    def _next_pick(b, taken_from_b):
        """The burst's best remaining member, preferring one a few minutes away from what's
        already chosen (two beats 11 seconds apart are the same beat)."""
        remaining = [m for m in b["members"] if m not in taken_from_b]
        apart = [m for m in remaining
                 if all(abs((m["sdt"] - t["sdt"]).total_seconds()) >= 180 for t in taken_from_b)]
        pool = apart or remaining
        return max(pool, key=lambda c: c["score"], default=None)

    extras = [b for b in bursts if len(b["members"]) >= 2]
    extras.sort(key=lambda b: (b["span_min"] >= BURST_EXTRA_MIN, len(b["members"]),
                               b["members"][0]["score"]), reverse=True)
    for b in extras:
        _take(_next_pick(b, [m for m in b["members"] if m in chosen]))
    # A really long session may earn a third beat once every burst has had its turn.
    for b in extras:
        if len(b["members"]) >= 3 and b["span_min"] >= BURST_EXTRA_MIN:
            _take(_next_pick(b, [m for m in b["members"] if m in chosen]))

    chosen.sort(key=lambda c: c["sdt"])
    return chosen


def _plan_key(cfg, plan) -> str:
    basis = json.dumps(
        [_PLAN_VERSION] + [[c["clip_path"], c["offset_s"], c["seconds"]] for c in plan["segments"]],
        sort_keys=True)
    return hashlib.sha1(basis.encode()).hexdigest()[:10]


def _paths_for(cfg, plan):
    key = _plan_key(cfg, plan)
    stem = f"reel_{plan['anchor']}_{plan['edition']}_{key}"
    out = _reels_dir(cfg) / f"{stem}.mp4"
    return key, out, out.with_suffix(".json")


def _manifest_from(cfg, plan, out: Path, poster: Path | None = None) -> dict:
    """Chapter offsets accumulate the segments' MEASURED durations (an encoder rounds each cut
    to whole frames), so a seek to chapter 12 lands on chapter 12, not half a second early."""
    at, segs = 0.0, []
    for c in plan["segments"]:
        secs = c.get("actual_s") or c["seconds"]
        segs.append({"at": round(at, 2), "seconds": round(secs, 2), "start": c["start"],
                     "species": c["species"], "individuals": c["individuals"],
                     "thumb": c["thumb"], "n_dets": c["n_dets"]})
        at += secs
    rel = out.relative_to(cfg.clips_dir.parent).as_posix()   # "clips/reels/..." for /media/
    man = {"version": _PLAN_VERSION, "edition": plan["edition"], "anchor": plan["anchor"],
           "title": plan["title"], "start": plan["start"], "end": plan["end"],
           "clip_path": rel, "seconds": round(at, 1), "segments": segs,
           "n_source_clips": plan["n_source_clips"]}
    # Absent on reels built before posters existed -- the dashboard falls back to the old
    # plate/thumb backdrop, so an old cached reel still renders.
    if poster is not None and poster.exists():
        man["poster_path"] = poster.relative_to(cfg.clips_dir.parent).as_posix()
    return man


def _run_ffmpeg(args, timeout):
    subprocess.run([_FFMPEG, "-y", "-loglevel", "error"] + args, check=True, timeout=timeout,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _probe_duration(p: Path) -> float | None:
    if _FFPROBE is None:
        return None
    try:
        r = subprocess.run([_FFPROBE, "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", str(p)],
                           capture_output=True, text=True, timeout=20)
        return float((r.stdout or "").strip())
    except (ValueError, subprocess.SubprocessError, OSError):
        return None


def _make_poster(out: Path, workdir: Path) -> Path | None:
    """Pick a real full-size frame OUT OF THE STITCHED REEL to use as its poster image.

    Why: the Dispatch hero and the player's poster used to fall back to a per-moment `thumb`,
    which is an animal CROP -- and crops are tight cutouts, often tiny. Measured 2026-07-26 on
    the night reel: the chosen plate crop was 389x292 (~4x upscale to a ~1500 px hero) and the
    chapter-thumb fallback was 96x103 (~16x). Blown up that far a crop is a wall of soft ovals,
    which reads as "the camera is broken" when the video behind it is fine 1280x720.

    Selection is dependency-free (no cv2 in the web path): dump a candidate every few seconds in
    one decode pass, then keep the LARGEST JPEG. At a fixed quality, the biggest file is the
    frame carrying the most detail -- a sharp, well-lit frame beats a dark or motion-smeared one.
    Returns the poster path, or None if anything fails (callers fall back to the old behaviour)."""
    cand = workdir / "poster"
    cand.mkdir(parents=True, exist_ok=True)
    try:
        _run_ffmpeg(["-i", str(out), "-vf", f"fps=1/{_POSTER_EVERY_S}", "-q:v", "2",
                     str(cand / "c_%03d.jpg")], timeout=120)
    except (subprocess.SubprocessError, OSError):
        return None
    shots = sorted(cand.glob("c_*.jpg"), key=lambda p: p.stat().st_size, reverse=True)
    if not shots:
        return None
    poster = out.with_suffix(".jpg")
    try:
        shutil.copyfile(shots[0], poster)
    except OSError:
        return None
    return poster


def build_reel(cfg, plan) -> dict:
    """Cut + stitch the planned reel (blocking; run me in a thread). Returns the manifest.
    Segments are re-encoded to identical H.264/720p/_REEL_FPS params so the final concat can be
    stream-copy; the result's duration is verified and anything off falls back to a re-encoding
    concat. Atomic publish via .tmp."""
    key, out, man_path = _paths_for(cfg, plan)
    if out.exists() and man_path.exists():
        return json.loads(man_path.read_text(encoding="utf-8"))
    workdir = _reels_dir(cfg) / f".build_{key}"
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        seg_files = []
        for i, c in enumerate(plan["segments"]):
            src = (cfg.clips_dir.parent / c["clip_path"]).resolve()
            if not src.exists():
                continue                       # pruned since planning -> just skip the moment
            seg = workdir / f"seg{i:02d}.mp4"
            _run_ffmpeg(
                ["-ss", f"{c['offset_s']:.2f}", "-t", f"{c['seconds']:.2f}", "-i", str(src),
                 "-vf", f"fps={_REEL_FPS},scale=1280:720:force_original_aspect_ratio=decrease,"
                        "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                 "-pix_fmt", "yuv420p", "-an", str(seg)], timeout=90)
            if seg.exists() and seg.stat().st_size > 0:
                c["actual_s"] = _probe_duration(seg) or c["seconds"]
                seg_files.append((i, seg))
        if not seg_files:
            raise RuntimeError("no segments could be cut (source clips missing?)")
        # Drop plan segments whose cut failed, so manifest chapters match the actual file.
        kept = {i for i, _p in seg_files}
        plan = {**plan, "segments": [c for i, c in enumerate(plan["segments"]) if i in kept]}
        want_s = sum(c.get("actual_s") or c["seconds"] for c in plan["segments"])

        lst = workdir / "list.txt"
        lst.write_text("".join(f"file '{p.as_posix()}'\n" for _i, p in seg_files), encoding="utf-8")
        tmp = out.with_suffix(".tmp.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)

        def _concat_reencode():
            _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst),
                         "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                         "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(tmp)],
                        timeout=300)

        try:
            _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst),
                         "-c", "copy", "-movflags", "+faststart", str(tmp)], timeout=120)
            got_s = _probe_duration(tmp)
            # Copy-concat silently mangles durations when segment timebases differ; if the
            # stitched length is off by more than a beat, re-encode the concat instead.
            if got_s is not None and abs(got_s - want_s) > max(1.5, 0.05 * want_s):
                raise RuntimeError(f"copy-concat came out {got_s:.1f}s, wanted {want_s:.1f}s")
        except Exception:
            _concat_reencode()
        tmp.replace(out)
        poster = _make_poster(out, workdir)      # a real frame, not an upscaled crop
        manifest = _manifest_from(cfg, plan, out, poster)
        man_path.write_text(json.dumps(manifest), encoding="utf-8")
        _prune_reels(cfg)
        return manifest
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _prune_reels(cfg) -> None:
    reels = sorted(_reels_dir(cfg).glob("reel_*.mp4"), key=lambda p: p.stat().st_mtime)
    for p in reels[:-KEEP_REELS]:
        p.unlink(missing_ok=True)
        p.with_suffix(".json").unlink(missing_ok=True)
        p.with_suffix(".jpg").unlink(missing_ok=True)     # its poster frame


def reel_status(cfg, edition="auto", date=None, now=None) -> dict:
    """The web endpoint's whole contract: plan the period's reel and report
    {status: ready|building|empty|unavailable|failed, ...}; 'ready' carries the manifest
    (clip_path is /media/-servable). Kicks off at most one background build per reel."""
    plan = plan_reel(cfg, edition=edition, date=date, now=now)
    if plan is None:
        return {"status": "empty", "reason": "no such period"}
    if not plan["segments"]:
        return {"status": "empty", "edition": plan["edition"], "anchor": plan["anchor"],
                "reason": "no usable clips this period"}
    key, out, man_path = _paths_for(cfg, plan)
    if out.exists() and man_path.exists():
        try:
            return {"status": "ready", **json.loads(man_path.read_text(encoding="utf-8"))}
        except (OSError, ValueError):
            pass                                # corrupt manifest -> rebuild below
    if _FFMPEG is None:
        return {"status": "unavailable", "reason": "ffmpeg not found on PATH"}

    planned = {"edition": plan["edition"], "anchor": plan["anchor"],
               "segments_planned": len(plan["segments"]),
               "seconds_planned": round(sum(c["seconds"] for c in plan["segments"]), 1)}
    with _BUILD_LOCK:
        state = _BUILDS.get(key)
        if state:
            kind, ts = state
            if kind == "building":
                return {"status": "building", **planned}
            if kind.startswith("failed") and _time.monotonic() - ts < _RETRY_S:
                return {"status": "failed", "reason": kind, **planned}
        _BUILDS[key] = ("building", _time.monotonic())

    def _worker():
        try:
            build_reel(cfg, plan)
            with _BUILD_LOCK:
                _BUILDS.pop(key, None)          # ready is now answered by the manifest on disk
        except Exception as e:  # noqa: BLE001 -- a failed build must never take the server down
            print(f"[reel] build failed for {plan['anchor']} {plan['edition']}: {e}")
            with _BUILD_LOCK:
                _BUILDS[key] = (f"failed: {e}", _time.monotonic())

    threading.Thread(target=_worker, name=f"reel-{key}", daemon=True).start()
    return {"status": "building", **planned}
