"""
Phase 3 (part 1) -- compute & store APPEARANCE embeddings for re-identification.

Each readable animal crop is run through MegaDescriptor (BVRA/MegaDescriptor-L-384), the
wildlife re-ID foundation model, producing a 1536-d appearance vector. Vectors are
L2-normalized and written to the detection_embeddings table (model='megadescriptor-l-384'),
so a cosine similarity between two crops is just a dot product. These vectors are the reusable
substrate the rest of phase 3 stands on: reid.py clusters them into candidate individuals and
answers "closest appearance match: Notch 78%, Gimpy 41%".

Why MegaDescriptor (verified current 2026-06-09): it is the purpose-built foundation model for
INDIVIDUAL re-ID across species (Swin-L, ArcFace metric learning over 29 wildlife datasets) and
beats general features like CLIP / DINOv2 on animal re-ID. It loads through timm with no extra
dependency. The schema keys vectors by `model`, so a second embedder (e.g. MiewID, which
generalizes better to unseen taxa) can be added later as ANOTHER row per crop -- no migration,
and reid.py can compare or fuse them.

This is appearance ONLY. Per PLAN.md the whole system keeps appearance and behaviour on
separate axes; behaviour profiles are phase 4. Embedding is a one-shot batch job (not folded
into the live rig), so unlike the species namer it can build the model on the main thread and
use the GPU freely.

  python embed.py                      # embed high-conf raccoon crops on the GPU (resumable)
  python embed.py --species all        # every species' animal crops, not just raccoons
  python embed.py --species "American crow"
  python embed.py --min-confidence 0.6 # widen the crop-usability gate (default 0.8)
  python embed.py --redo               # recompute even crops already embedded
  python embed.py --device cpu         # no GPU (slow; the model is heavy)
  python embed.py --limit 50           # just the first 50 (quick check)
  python embed.py --species all --co-present --min-confidence 0.25
                                       # LOW-conf crops, but ONLY in plausibly multi-animal
                                       # visits -- the still-tracklet splitter's food (the
                                       # second animal is nearly always a low-conf box)
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time

import config
import db
import detector

# The re-ID foundation model. MODEL_NAME is the base tag stored in detection_embeddings.model;
# HF_MODEL is how timm pulls the weights (cached under ~/.cache after the first ~900MB fetch).
MODEL_NAME = "megadescriptor-l-384"
HF_MODEL = "hf-hub:BVRA/MegaDescriptor-L-384"
EMBED_DIM = 1536

def model_tag() -> str:
    """The detection_embeddings.model value for a run. One embedder, so always MODEL_NAME -- kept
    as a function because reid.py / twoaxis.py call it (it once also produced a '-seg' variant for
    an SAM background-removal experiment that was tested, didn't help, and has been removed)."""
    return MODEL_NAME


# Default crop-usability gate. Detector confidence doubles as a readability score, and re-ID
# wants the legible crops -- a blurry distant raccoon embeds to noise. 0.8 keeps ~1,600 of the
# raccoon crops (the trustworthy ones); lower it with --min-confidence to widen the net.
DEFAULT_MIN_CONFIDENCE = 0.8


def build_embedder(device: str):
    """Construct MegaDescriptor + its preprocessing transform on the resolved device (GPU when it
    genuinely runs, else CPU; see detector.build_with_fallback). Returns (model, transform, used)."""
    import timm

    def _make(dev: str):
        model = timm.create_model(HF_MODEL, pretrained=True, num_classes=0).eval().to(dev)
        cfg = timm.data.resolve_data_config({}, model=model)
        return model, timm.data.create_transform(**cfg)

    (model, transform), device = detector.build_with_fallback(_make, device, what="re-ID embedder")
    return model, transform, device


def _load_image(path):
    """Open a crop as RGB; return None (and warn) if it's missing or unreadable so one bad
    file never aborts a long batch run."""
    from PIL import Image
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:  # noqa: BLE001
        print(f"  skip (unreadable): {path} ({e})")
        return None


def embed_rows(conn, model, transform, device, rows, *, batch_size, total=None,
               model_name=MODEL_NAME):
    """Embed (id, crop_path) rows in batches and store each L2-normalized vector under
    `model_name`. Returns (n_done, device) -- device may flip to cpu if the GPU runs out of
    memory mid-run."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    total = total if total is not None else len(rows)
    done = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        tensors, ids = [], []
        for rid, cp in chunk:
            img = _load_image(str(db.crop_abspath(cp)))
            if img is None:
                continue
            tensors.append(transform(img))
            ids.append(rid)
        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        try:
            with torch.no_grad():
                feats = model(batch)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device != "cpu":
                print("  GPU out of memory -- switching to CPU for the remainder.")
                torch.cuda.empty_cache()
                device = "cpu"
                model = model.to("cpu")
                batch = batch.to("cpu")
                with torch.no_grad():
                    feats = model(batch)
            else:
                raise

        feats = F.normalize(feats, dim=1).cpu().numpy().astype(np.float32)
        for rid, vec in zip(ids, feats):
            db.insert_embedding(conn, rid, model_name, EMBED_DIM, vec.tobytes())
        conn.commit()
        done += len(ids)
        print(f"  {done}/{total} embedded ...")
    return done, device


def main() -> int:
    p = argparse.ArgumentParser(
        description="Phase 3: MegaDescriptor appearance embeddings for re-identification.")
    p.add_argument("--species", default="raccoon",
                   help="Species to embed (default 'raccoon'; 'all' = every species' animal crops).")
    p.add_argument("--device", default=config.CONFIG.device, choices=["cuda", "cpu", "auto"],
                   help="cuda | cpu | auto (default from config). The model is heavy; a GPU is far "
                        "faster, but auto falls back to CPU when there's no usable GPU.")
    p.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                   help=f"Only embed crops with detector confidence >= this "
                        f"(usability gate; default {DEFAULT_MIN_CONFIDENCE}).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="Max crops to process (0 = all).")
    p.add_argument("--redo", action="store_true",
                   help="Re-embed crops that already have a vector (e.g. after a model change).")
    p.add_argument("--co-present", action="store_true",
                   help="Only visits that plausibly hold 2+ animals (same-frame separated boxes, "
                        "or a multi-name live sighting). Pair with a LOW --min-confidence: the "
                        "second animal is nearly always the low-confidence box (measured "
                        "2026-07-31: 2,423 of 2,500 unembedded co-present pair sides sat under "
                        "the 0.5 gate), and the still-tracklet splitter is blind without these "
                        "vectors. The nightly batch runs this after the main pass.")
    args = p.parse_args()

    species = None if args.species.lower() == "all" else args.species
    tag = model_tag()
    conn = db.connect(config.CONFIG.db_path)

    visit_ids = None
    if args.co_present:
        # Lazy import: individuals pulls numpy/scipy, which a plain embed run doesn't need.
        import individuals
        conn.row_factory = sqlite3.Row
        visit_ids = individuals.co_present_visit_ids(conn)
        print(f"Co-present pass: {len(visit_ids)} candidate multi-animal visit(s).")

    rows = db.fetch_for_embedding(conn, tag, species=species,
                                  min_confidence=args.min_confidence, redo=args.redo,
                                  limit=args.limit, visit_ids=visit_ids)
    if not rows:
        print(f"Nothing to embed -- all caught up (model '{tag}', "
              f"{'all species' if species is None else species}, "
              f"confidence >= {args.min_confidence}).")
        conn.close()
        return 0

    label = "all species" if species is None else species
    print(f"Embedding {len(rows)} {label} crop(s) with MegaDescriptor on {args.device} "
          f"(confidence >= {args.min_confidence})...")
    t0 = time.time()
    model, transform, device = build_embedder(args.device)
    print(f"  model ready on {device} ({time.time() - t0:.0f}s).")

    done, device = embed_rows(conn, model, transform, device, rows,
                              batch_size=args.batch_size, model_name=tag)
    conn.close()
    print(f"\nDone. Embedded {done} crop(s) in {time.time() - t0:.0f}s "
          f"(model '{tag}', {EMBED_DIM}-d). Next: python reid.py --species {args.species}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
