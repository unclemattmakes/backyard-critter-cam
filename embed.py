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
"""
from __future__ import annotations

import argparse
import sys
import time

import config
import db

# The re-ID foundation model. MODEL_NAME is the base tag stored in detection_embeddings.model;
# HF_MODEL is how timm pulls the weights (cached under ~/.cache after the first ~900MB fetch).
MODEL_NAME = "megadescriptor-l-384"
HF_MODEL = "hf-hub:BVRA/MegaDescriptor-L-384"
EMBED_DIM = 1536

# Optional background removal (--segment): MobileSAM masks the animal off the crop before
# embedding, so the embedding can't key on the (identical, across every crop) patio/brick
# background. Stored under a DIFFERENT model tag so the two embedding sets live side by side
# and reid.py can compare them. The animal is masked onto the ImageNet mean colour, which the
# MegaDescriptor transform normalizes to ~0 -- i.e. a maximally neutral background.
SAM_WEIGHTS = "mobile_sam.pt"           # small/fast; auto-downloads from Ultralytics on first use
SEG_SUFFIX = "-seg"
IMAGENET_MEAN_RGB = (124, 116, 104)     # 255 * (0.485, 0.456, 0.406)


def model_tag(segment: bool) -> str:
    """The detection_embeddings.model value for a run: base, or base+'-seg' when masking."""
    return MODEL_NAME + (SEG_SUFFIX if segment else "")


# Default crop-usability gate. Detector confidence doubles as a readability score, and re-ID
# wants the legible crops -- a blurry distant raccoon embeds to noise. 0.8 keeps ~1,600 of the
# raccoon crops (the trustworthy ones); lower it with --min-confidence to widen the net.
DEFAULT_MIN_CONFIDENCE = 0.8


def build_embedder(device: str):
    """Construct MegaDescriptor + its preprocessing transform, falling back to CPU if the GPU
    init fails (mirrors classify.build_classifier). Returns (model, transform, device_used)."""
    import timm
    import torch

    def _make(dev: str):
        model = timm.create_model(HF_MODEL, pretrained=True, num_classes=0)
        model = model.eval().to(dev)
        cfg = timm.data.resolve_data_config({}, model=model)
        transform = timm.data.create_transform(**cfg)
        return model, transform

    try:
        model, transform = _make(device)
        return model, transform, device
    except Exception as e:  # noqa: BLE001 -- any GPU/init failure should degrade, not crash
        if device != "cpu":
            print(f"  {device} init failed ({e}); falling back to CPU.")
            torch.cuda.empty_cache()
            model, transform = _make("cpu")
            return model, transform, "cpu"
        raise


def build_segmenter(device: str):
    """Construct MobileSAM for background removal (--segment). Returns the model, or None on
    CPU/no-GPU or any load failure -- masking is an optional enhancement, never a hard
    requirement, so a failure degrades to embedding the full crop rather than crashing."""
    try:
        from ultralytics import SAM
        sam = SAM(SAM_WEIGHTS)
        sam.to(device)
        return sam
    except Exception as e:  # noqa: BLE001
        print(f"  segmentation unavailable ({e}); embedding full crops instead.")
        return None


def _segment_image(sam, img, device):
    """Mask the central animal out of a crop and composite it on the ImageNet mean colour.
    Prompts SAM with a point at the crop centre (the detector already centred the animal in
    the box). Falls back to the untouched image if SAM returns no usable mask."""
    import numpy as np
    from PIL import Image
    w, h = img.size
    try:
        res = sam(img, points=[[w // 2, h // 2]], labels=[1], device=device, verbose=False)
        mask = res[0].masks.data[0].cpu().numpy().astype(bool)
        if mask.mean() < 0.02:           # essentially empty -> trust the full crop instead
            return img
        arr = np.array(img)
        bg = np.empty_like(arr)
        bg[:] = IMAGENET_MEAN_RGB
        return Image.fromarray(np.where(mask[..., None], arr, bg))
    except Exception:  # noqa: BLE001 -- a bad mask shouldn't sink the batch
        return img


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
               segmenter=None, model_name=MODEL_NAME):
    """Embed (id, crop_path) rows in batches and store each L2-normalized vector under
    `model_name`. With a `segmenter`, each crop's background is masked out first. Returns
    (n_done, device) -- device may flip to cpu if the GPU runs out of memory mid-run."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    total = total if total is not None else len(rows)
    done = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        tensors, ids = [], []
        for rid, cp in chunk:
            img = _load_image(str(config.ROOT / cp.replace("\\", "/")))
            if img is None:
                continue
            if segmenter is not None:
                img = _segment_image(segmenter, img, device)
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
    p.add_argument("--device", default="cuda",
                   help="cuda (default) or cpu. The model is heavy; GPU is strongly preferred.")
    p.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                   help=f"Only embed crops with detector confidence >= this "
                        f"(usability gate; default {DEFAULT_MIN_CONFIDENCE}).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="Max crops to process (0 = all).")
    p.add_argument("--redo", action="store_true",
                   help="Re-embed crops that already have a vector (e.g. after a model change).")
    p.add_argument("--segment", action="store_true",
                   help="Mask the animal off the background with MobileSAM before embedding "
                        "(removes the patio/brick confound). Stored under a separate '-seg' tag.")
    args = p.parse_args()

    species = None if args.species.lower() == "all" else args.species
    tag = model_tag(args.segment)
    conn = db.connect(config.CONFIG.db_path)

    rows = db.fetch_for_embedding(conn, tag, species=species,
                                  min_confidence=args.min_confidence, redo=args.redo,
                                  limit=args.limit)
    if not rows:
        print(f"Nothing to embed -- all caught up (model '{tag}', "
              f"{'all species' if species is None else species}, "
              f"confidence >= {args.min_confidence}).")
        conn.close()
        return 0

    label = "all species" if species is None else species
    print(f"Embedding {len(rows)} {label} crop(s) with MegaDescriptor on {args.device} "
          f"(confidence >= {args.min_confidence}{', SAM-masked' if args.segment else ''})...")
    t0 = time.time()
    model, transform, device = build_embedder(args.device)
    segmenter = build_segmenter(device) if args.segment else None
    if args.segment and segmenter is None:
        tag = model_tag(False)  # masking failed to load -> fall back to plain embeddings + tag
    print(f"  model{'+SAM' if segmenter else ''} ready on {device} ({time.time() - t0:.0f}s).")

    done, device = embed_rows(conn, model, transform, device, rows,
                              batch_size=args.batch_size, segmenter=segmenter, model_name=tag)
    conn.close()
    print(f"\nDone. Embedded {done} crop(s) in {time.time() - t0:.0f}s "
          f"(model '{tag}', {EMBED_DIM}-d). Next: python reid.py --species {args.species}"
          f"{' --segment' if args.segment else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
