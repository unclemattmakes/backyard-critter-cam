"""
Phase 4 (part 3) -- APPEARANCE embeddings from clip TRACKLETS.

embed.py embeds the live still crops; this embeds the per-animal motion tracklets clipmotion.py
found in the behaviour clips. Why it matters: a clip is full-frame-rate and tracks each animal
separately, so a tracklet is a clean per-ANIMAL appearance sample EVEN inside a multi-animal
(pair) visit -- which the still pipeline can only treat as one blended whole. That unlocks:

  * un-blending pairs -- the two tracklets in a pair clip are the two animals, so each can get its
    own appearance template (e.g. a clean Elliot, who almost only ever appears WITH Notch);
  * a far larger different-individual test -- every 2-tracklet clip is two animals under identical
    light / glass / instant (the controlled negative the stills gave only n=3 of);
  * more pose/viewpoint coverage than the sparse live crops happened to bank.

One L2-normalized prototype per tracklet = mean of its sharpest sampled frames' MegaDescriptor
vectors, stored in clip_track_embeddings under the SAME model tag as detection_embeddings, so clip
and still vectors live in one comparable space. Batch + resumable, mirroring embed.py/clipmotion.py:
only sustained tracklets (>= --min-hits boxes) without a vector are processed unless --redo.

  python clipembed.py                 # embed sustained tracklets without a vector (GPU, resumable)
  python clipembed.py --device cpu    # no GPU contention with the live rig (slow)
  python clipembed.py --redo          # recompute existing tracklet vectors
  python clipembed.py --frames 12     # frames pooled per tracklet (default 10)
  python clipembed.py --clip 339      # just one clip's tracklets
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time

import numpy as np

import config
import db
from clipmotion import SUSTAINED_HITS
from embed import EMBED_DIM, MODEL_NAME, build_embedder

# Box padding: grow the detector box ~8% each side before cropping, so the embedder sees a little
# context (ear/tail tips the tight box clips) without swamping the animal in background.
BOX_PAD = 0.08


def select_frames(track: list, n: int) -> list:
    """Pick up to `n` points from a tracklet for embedding: spread them evenly across the track's
    timeline (pose/viewpoint coverage), taking the highest-confidence point in each time bin (the
    sharpest, most-readable crop there). Pure -> unit-testable. `track` = [[t,cx,cy,w,h,conf],...]."""
    if not track:
        return []
    if len(track) <= n:
        return list(track)
    pts = sorted(track, key=lambda p: p[0])           # by time
    out = []
    for b in range(n):
        lo = b * len(pts) // n
        hi = max(lo + 1, (b + 1) * len(pts) // n)
        out.append(max(pts[lo:hi], key=lambda p: p[5]))   # sharpest (highest conf) in the bin
    return out


def _crop(frame, box, pad: float = BOX_PAD):
    """Crop a (cx,cy,w,h)-normalized box from a BGR frame, padded and clamped, as an RGB PIL image.
    Returns None if the box is degenerate."""
    from PIL import Image
    h, w = frame.shape[:2]
    cx, cy, bw, bh = box
    bw, bh = bw * (1 + 2 * pad), bh * (1 + 2 * pad)
    x1 = max(0, int((cx - bw / 2) * w))
    y1 = max(0, int((cy - bh / 2) * h))
    x2 = min(w, int((cx + bw / 2) * w))
    y2 = min(h, int((cy + bh / 2) * h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return Image.fromarray(frame[y1:y2, x1:x2][:, :, ::-1])   # BGR -> RGB


def extract_clip_crops(clip_path, tracks, n_frames):
    """For one clip, sample each tracklet's frames and crop the animal out. `tracks` =
    [(track_id, [points]), ...]. Returns {track_id: [PIL crops]} by reading the video ONCE
    (targets gathered across all tracklets, dispatched as the matching frame is decoded)."""
    import cv2
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    # frame_idx -> list of (track_id, box); a frame may serve several tracklets (a pair moment).
    targets: dict = {}
    for tid, pts in tracks:
        for p in select_frames(pts, n_frames):
            fidx = int(round(p[0] * fps))
            targets.setdefault(fidx, []).append((tid, (p[1], p[2], p[3], p[4])))
    crops: dict = {tid: [] for tid, _ in tracks}
    if not targets:
        cap.release()
        return crops
    last = max(targets)
    idx = 0
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, box in targets.get(idx, ()):
            im = _crop(frame, box)
            if im is not None:
                crops[tid].append(im)
        idx += 1
    cap.release()
    return crops


def _save_rep_crop(track_id, crops):
    """Save the largest (closest-to-camera = most readable) of a tracklet's sampled crops as the
    UI thumbnail under config.clip_crops_dir; return its project-relative path (or None)."""
    if not crops:
        return None
    best = max(crops, key=lambda im: im.size[0] * im.size[1])
    out = config.CONFIG.clip_crops_dir / f"track_{int(track_id)}.jpg"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        best.convert("RGB").save(out, quality=85)
        return str(out.relative_to(config.ROOT)).replace("\\", "/")
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 4: per-tracklet appearance embeddings from clips.")
    p.add_argument("--device", default=config.CONFIG.device, choices=["cuda", "cpu", "auto"],
                   help="cuda | cpu | auto (default from config).")
    p.add_argument("--frames", type=int, default=10, help="Frames pooled per tracklet (default 10).")
    p.add_argument("--min-hits", type=int, default=SUSTAINED_HITS,
                   help=f"Only tracklets with >= this many boxes (default {SUSTAINED_HITS}).")
    p.add_argument("--clip", type=int, default=None, help="Only this clip's tracklets (clips.id).")
    p.add_argument("--redo", action="store_true", help="Recompute tracklets that already have a vector.")
    args = p.parse_args()

    conn = db.connect(config.CONFIG.db_path)
    conn.row_factory = sqlite3.Row
    try:
        if args.redo:
            rows = conn.execute(
                """SELECT t.id AS track_id, t.clip_id, t.track_idx, t.track, t.n_hits,
                          c.clip_path, c.fps
                   FROM clip_tracks t JOIN clips c ON c.id = t.clip_id
                   WHERE t.n_hits >= ? AND t.track IS NOT NULL ORDER BY t.clip_id, t.track_idx""",
                (args.min_hits,)).fetchall()
        else:
            rows = db.clip_tracks_needing_embedding(conn, MODEL_NAME, args.min_hits)
        if args.clip is not None:
            rows = [r for r in rows if r["clip_id"] == args.clip]
        if not rows:
            print("Nothing to embed -- every sustained tracklet already has a vector "
                  "(run clipmotion.py first, or --redo).")
            return 0

        # Group the work by clip so each video is opened once.
        by_clip: dict = {}
        for r in rows:
            by_clip.setdefault((r["clip_id"], r["clip_path"]), []).append(r)
        n_tracks = len(rows)
        print(f"Embedding {n_tracks} tracklet(s) across {len(by_clip)} clip(s) with MegaDescriptor "
              f"on {args.device} ({args.frames} frames each)...")

        import torch
        import torch.nn.functional as F
        t0 = time.time()
        model, transform, device = build_embedder(args.device)
        print(f"  model ready on {device} ({time.time() - t0:.0f}s).")

        done = 0
        missing = 0
        for (clip_id, clip_path), trows in by_clip.items():
            full = config.ROOT / str(clip_path).replace("\\", "/")
            if not full.exists():
                missing += len(trows)
                continue
            tracks = [(r["track_id"], json.loads(r["track"])) for r in trows]
            crops = extract_clip_crops(full, tracks, args.frames)
            if crops is None:
                print(f"  clip #{clip_id}: could not open video -- skipped.")
                continue
            for tid, ims in crops.items():
                if not ims:
                    # No usable crop for this sustained tracklet (all sampled frames degenerate or
                    # undecodable). Write a zero-frame MARKER so clip_tracks_needing_embedding won't
                    # re-select it (and re-decode the clip) on every run; load_clip_track_embeddings
                    # filters n_frames > 0, so the marker never reaches matching/analysis.
                    db.insert_clip_track_embedding(conn, track_id=tid, model=MODEL_NAME,
                                                   dim=0, embedding=b"", n_frames=0, rep_crop=None)
                    continue
                batch = torch.stack([transform(im) for im in ims]).to(device)
                with torch.no_grad():
                    feats = F.normalize(model(batch), dim=1)
                proto = F.normalize(feats.mean(dim=0, keepdim=True), dim=1)[0]
                vec = proto.cpu().numpy().astype(np.float32)
                rep = _save_rep_crop(tid, ims)
                db.insert_clip_track_embedding(conn, track_id=tid, model=MODEL_NAME,
                                               dim=EMBED_DIM, embedding=vec.tobytes(),
                                               n_frames=len(ims), rep_crop=rep)
                done += 1
            print(f"  clip #{clip_id}: embedded {sum(1 for v in crops.values() if v)}/{len(trows)} "
                  f"tracklet(s)  [{done}/{n_tracks}]")
        msg = f"\nDone. {done} tracklet vector(s) in {time.time() - t0:.0f}s (model '{MODEL_NAME}')."
        if missing:
            msg += f" {missing} skipped (clip file pruned)."
        print(msg)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
