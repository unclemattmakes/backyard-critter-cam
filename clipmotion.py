"""
Phase 4 -- MOTION fingerprints from the behaviour clips.

A still says who was there; the clip says HOW THEY MOVED -- and that's both behaviour (bold
beeline to the dish vs hesitant stop-start creep) and a confound-robust shot at individual ID
(a limp or a characteristic gait reads the same from any angle, where still-appearance re-ID
failed on pose + glass). This tool turns each recorded clip (clips.py / --record-clips) into a
motion track + a small feature fingerprint:

    sample frames -> MegaDetector box per sample -> box-centre trajectory ->
        duration, path length, straightness (beeline=1), avg/peak speed,
        moving fraction (vs stationary, e.g. head-down eating), area trend (approach/retreat)

The RAW track (normalized positions + times) is stored as JSON alongside the features, so richer
gait work later (stride rhythm, limp detection) can re-derive from the track without re-running
the detector over the video. One track per (clip, model); single-target assumption -- the
highest-confidence box per sample is "the" animal (fine for a feeding-spot rig; documented limit
for multi-animal visits).

Batch + resumable, mirroring embed.py: only clips without a track are processed unless --redo.

  python clipmotion.py                # process all new clips (GPU)
  python clipmotion.py --device cpu   # no GPU contention with the live rig
  python clipmotion.py --clip 2       # one clip, by clips.id
  python clipmotion.py --redo         # recompute existing tracks too
  python clipmotion.py --show         # print the stored fingerprints (no processing)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time

import config
import db

# clip_tracks.model is the rig's configured detector (e.g. 'MDV6-yolov10-c') so a clip's track
# is built by the same eyes that triggered it.

# Target sampling rate over the clip. ~5 boxes/second is plenty for path/speed features while
# keeping a 30 s clip to ~150 detector passes.
SAMPLE_HZ = 5.0

# A sample counts as "moving" above this speed (normalized frame units / second). 0.04 ~ 50 px/s
# at 1280 wide -- below a slow amble, above breathing/jitter. Eyeballed against real tracks.
MOVING_SPEED = 0.04


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


def extract_track(clip_path, detector, sample_hz: float):
    """Sample a clip's frames and detect in each. Returns (track, n_samples) where track is
    [[t_s, cx, cy, w, h, conf], ...] -- the highest-confidence box per sampled frame, coordinates
    normalized 0..1 by frame size, t from the clip's stored fps. Missing/undetected samples are
    simply absent (gap-aware features handle it)."""
    import cv2
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    step = max(1, round(fps / sample_hz))
    track = []
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
            if dets:
                best = max(dets, key=lambda d: d.confidence)
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = best.bbox
                track.append([round(idx / fps, 3),
                              round(((x1 + x2) / 2) / w, 4), round(((y1 + y2) / 2) / h, 4),
                              round((x2 - x1) / w, 4), round((y2 - y1) / h, 4),
                              round(best.confidence, 3)])
        idx += 1
    cap.release()
    return track, n_samples


def fingerprint_line(clip_id, feats, n_hits, n_samples) -> str:
    s = feats
    def f(v, fmt="{:.2f}"):
        return fmt.format(v) if v is not None else "--"
    return (f"  clip #{clip_id:<3} {n_hits}/{n_samples} hits  "
            f"dur {f(s['duration_s'], '{:.1f}')}s  path {f(s['path_len'])}  "
            f"straight {f(s['straightness'])}  v_avg {f(s['avg_speed'])}  "
            f"v_peak {f(s['peak_speed'])}  moving {f(s['moving_frac'])}  "
            f"area x{f(s['area_trend'])}")


def show_tracks(conn) -> int:
    rows = conn.execute(
        """SELECT t.clip_id, t.n_samples, t.n_hits, t.duration_s, t.path_len, t.net_disp,
                  t.straightness, t.avg_speed, t.peak_speed, t.moving_frac, t.area_trend,
                  c.started_at, c.clip_path
           FROM clip_tracks t JOIN clips c ON c.id = t.clip_id ORDER BY t.clip_id"""
    ).fetchall()
    if not rows:
        print("No motion tracks yet -- run clipmotion.py after recording clips (--record-clips).")
        return 0
    print(f"{len(rows)} motion fingerprint(s):")
    for r in rows:
        feats = {k: r[k] for k in ("duration_s", "path_len", "net_disp", "straightness",
                                   "avg_speed", "peak_speed", "moving_frac", "area_trend")}
        print(fingerprint_line(r["clip_id"], feats, r["n_hits"], r["n_samples"])
              + f"  [{(r['started_at'] or '')[5:16]}]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 4: motion fingerprints from behaviour clips.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"],
                   help="Detector device (cpu avoids GPU contention with the live rig).")
    p.add_argument("--sample-hz", type=float, default=SAMPLE_HZ,
                   help=f"Detector samples per second of clip (default {SAMPLE_HZ:g}).")
    p.add_argument("--clip", type=int, default=None, help="Process one clip by its clips.id.")
    p.add_argument("--redo", action="store_true", help="Recompute clips that already have a track.")
    p.add_argument("--show", action="store_true", help="Print stored fingerprints and exit.")
    args = p.parse_args()

    cfg = config.CONFIG
    conn = db.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    try:
        if args.show:
            return show_tracks(conn)

        model = cfg.model_version
        if args.clip is not None:
            rows = conn.execute("SELECT id, clip_path, fps FROM clips WHERE id = ?",
                                (args.clip,)).fetchall()
        elif args.redo:
            rows = conn.execute("SELECT id, clip_path, fps FROM clips ORDER BY id").fetchall()
        else:
            rows = db.clips_needing_tracks(conn, model)
        if not rows:
            print("Nothing to process -- every clip already has a motion track "
                  "(record more with --record-clips, or --redo).")
            return 0

        from detector import Detector
        print(f"Building motion tracks for {len(rows)} clip(s) "
              f"({model} on {args.device}, ~{args.sample_hz:g} samples/s)...")
        detector = Detector(model, args.device, cfg.min_confidence, classes=cfg.detect_classes)
        t0 = time.time()
        done = 0
        for r in rows:
            clip_path = config.ROOT / str(r["clip_path"]).replace("\\", "/")
            if not clip_path.exists():
                print(f"  clip #{r['id']}: file missing ({r['clip_path']}) -- skipped.")
                continue
            track, n_samples = extract_track(clip_path, detector, args.sample_hz)
            if track is None:
                print(f"  clip #{r['id']}: could not open video -- skipped.")
                continue
            feats = track_features(track)
            db.insert_clip_track(conn, clip_id=r["id"], model=model, n_samples=n_samples,
                                 n_hits=len(track), track_json=json.dumps(track), features=feats)
            print(fingerprint_line(r["id"], feats, len(track), n_samples))
            done += 1
        print(f"\nDone. {done} track(s) in {time.time() - t0:.0f}s. "
              f"View any time with: python clipmotion.py --show")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
