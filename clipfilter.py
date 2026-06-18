"""
Non-animal prefilter: a GENERAL-CLIP gate that runs BEFORE BioCLIP species naming.

Why this exists
---------------
MegaDetector's coarse 'animal' class occasionally false-fires on things that aren't animals --
at this glass-door rig, a plate of food, a pet bowl, or bare ground. BioCLIP (the species namer)
can't reject those: it's an ORGANISM-ONLY model, so every crop is forced onto the nearest real
species. In this DB those false-fires piled up as "brown rat" (none were real rats). We tried
giving BioCLIP non-animal "decoy" labels and it was inert -- its text encoder won't embed
non-organism prompts strongly enough to ever win (see the NB in classify.py).

A GENERAL CLIP can. open_clip (already installed -- BioCLIP rides on it) with everyday-image
weights embeds "a plate of food" vs "an animal" just fine, so it can answer the one question
BioCLIP can't: "is this even an animal?" We build two text PROTOTYPES (the mean text embedding
of an ANIMAL prompt set and of a NON-ANIMAL prompt set -- averaging makes the decision
independent of how many prompts are in each list), then for each crop compare its image
embedding to the two prototypes. Crops that land on the non-animal prototype with enough
confidence are labelled NONANIMAL_LABEL and skip BioCLIP.

Zero-shot, like classify.py: editing the two prompt lists below is free -- no training.

CLI (read-only validation; writes NOTHING to the DB):
  python clipfilter.py --sample              # score real crops (food vs known animals) +
                                             # print a threshold sweep, so you can pick the cut
  python clipfilter.py --sample --device cpu # ... on CPU
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import config
import db

# The label written onto crops the gate judges non-animal. It is listed in stats._NON_CRITTER
# (and the dashboard's mirror) so the digest/calendar/glances hide it, exactly like the
# "cat food"/"blur" hand-corrections. Keep these three spellings in sync if you change it.
NONANIMAL_LABEL = "not an animal"

# --- Prompt sets. Both are ensembled into a single prototype each, so list length doesn't tilt
# the decision -- add phrasings freely. Tuned for a PNW glass-door cam (night IR, food plates set
# out for raccoons, a wooden deck + brick patio). Common, concrete phrasings beat exotic ones.
ANIMAL_PROMPTS = [
    "a photo of an animal", "a wild animal", "an animal in a backyard at night",
    "a furry mammal", "a small mammal", "a bird", "a raccoon", "a squirrel",
    "an opossum", "a rat", "a cat", "a dog", "a crow", "an animal on a deck at night",
]
NONANIMAL_PROMPTS = [
    "a plate of food", "a bowl of pet food", "a dish of leftover food",
    "a pile of food scraps", "food on the ground with no animal",
    "an empty wooden deck", "an empty patio at night", "bare ground", "a brick surface",
    "an empty scene with no animal", "leaves and dirt", "a blurry photo of nothing",
]


def decision(p_nonanimal: float, threshold: float) -> bool:
    """Map a non-animal probability to a verdict. True == REJECT (it's not an animal). The cut is
    deliberately > 0.5 (set by config.nonanimal_threshold) so the gate only fires when it's quite
    sure, and never shaves off a genuine but odd-looking critter."""
    return p_nonanimal >= threshold


class AnimalFilter:
    """A general-CLIP 'is this an animal?' gate. Loads once, then judges crops in batches.

    Construction loads the open_clip checkpoint and precomputes the two text prototypes. Heavy
    imports (torch / open_clip / PIL) happen here, not at module import, so `from clipfilter
    import NONANIMAL_LABEL` stays cheap for callers that never build a filter.
    """

    def __init__(self, model_name: str, pretrained: str, device: str, threshold: float):
        import open_clip
        import torch

        self.threshold = float(threshold)
        self.model_name = model_name
        self.pretrained = pretrained

        # Mirror build_classifier()'s policy: try the requested device, fall back to CPU if GPU
        # init fails, so a wrong-arch torch build never takes the whole naming path down.
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device)
            self.device = device
        except Exception as e:
            if device != "cpu":
                print(f"  non-animal filter: {device} init failed ({e}); falling back to CPU.")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained, device="cpu")
                self.device = "cpu"
            else:
                raise

        model.eval()
        self.model = model
        self.preprocess = preprocess
        self._torch = torch
        tokenizer = open_clip.get_tokenizer(model_name)

        with torch.no_grad():
            def _proto(prompts):
                feats = model.encode_text(tokenizer(prompts).to(self.device))
                feats = feats / feats.norm(dim=-1, keepdim=True)
                m = feats.mean(dim=0)
                return m / m.norm()
            # (2, D): row 0 = ANIMAL prototype, row 1 = NON-ANIMAL prototype.
            self.protos = torch.stack([_proto(ANIMAL_PROMPTS), _proto(NONANIMAL_PROMPTS)])
            # The model's calibrated temperature, so the softmax below gives sane probabilities.
            self.scale = float(model.logit_scale.exp().item())

    @staticmethod
    def score_features(img_feats, protos, scale):
        """Pure scoring step (factored out so it's unit-testable without a model download): given
        L2-normalizable image features (N, D), the (2, D) prototypes and the logit scale, return
        an (N, 2) probability tensor [p_animal, p_nonanimal]."""
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        logits = scale * img_feats @ protos.T
        return logits.softmax(dim=-1)

    def judge(self, paths) -> list[tuple[bool, float]]:
        """Judge crop image paths. Returns one (is_animal, p_nonanimal) per path, in order.
        Unreadable images are treated as animals (is_animal=True) -- fail-open, so a corrupt crop
        is left for BioCLIP/quality to handle rather than being silently dropped as 'not animal'."""
        from PIL import Image
        torch = self._torch

        tensors, idx = [], []
        results: list[tuple[bool, float]] = [(True, 0.0)] * len(paths)
        for i, p in enumerate(paths):
            try:
                tensors.append(self.preprocess(Image.open(p).convert("RGB")))
                idx.append(i)
            except Exception:
                pass  # leave the fail-open default for this one
        if not tensors:
            return results

        with torch.no_grad():
            batch = torch.stack(tensors).to(self.device)
            img_feats = self.model.encode_image(batch)
            probs = self.score_features(img_feats, self.protos, self.scale)
            p_non = probs[:, 1].tolist()

        for slot, pn in zip(idx, p_non):
            results[slot] = (not decision(pn, self.threshold), float(pn))
        return results


# --------------------------------------------------------------------------------------------
# Validation CLI: score REAL crops so we can pick the threshold before touching the DB.
# --------------------------------------------------------------------------------------------

def _cohort(conn, where: str, params, limit: int):
    rows = conn.execute(
        f"SELECT id, crop_path FROM detections WHERE {where} ORDER BY id", params).fetchall()
    import os
    out = []
    for rid, cp in rows:
        pth = str(config.ROOT / cp.replace("\\", "/"))
        if os.path.exists(pth):
            out.append(pth)
        if limit and len(out) >= limit:
            break
    return out


def run_sample(conn, args) -> int:
    """Score a 'should be non-animal' cohort (the food false-fires that became 'brown rat') and a
    'should be animal' cohort (confidently-detected known species), then print how each prompt
    threshold would split them. Writes nothing."""
    af = AnimalFilter(args.model, args.pretrained, args.device, args.threshold)
    print(f"Loaded {args.model}/{args.pretrained} on {af.device}. Scoring real crops (no writes)...\n")

    food = _cohort(conn, "species = 'brown rat' AND COALESCE(species_verified,0)!=1", [], args.limit)
    # Known-good animals: confidently detected, not human-touched, a spread of species.
    animals = _cohort(
        conn,
        "detection_class='animal' AND confidence >= 0.6 AND COALESCE(species_verified,0)!=1 "
        "AND species IN ('raccoon','American crow','dark-eyed junco','domestic dog',"
        "'Virginia opossum','domestic cat','eastern gray squirrel','spotted towhee')",
        [], args.limit)

    food_p = [pn for _, pn in af.judge(food)]
    animal_p = [pn for _, pn in af.judge(animals)]
    print(f"cohorts: {len(food_p)} 'brown rat' (expect NON-animal), "
          f"{len(animal_p)} known animals (expect animal)\n")

    print(f"{'threshold':>9} | {'food caught':>12} | {'animals wrongly flagged':>24}")
    print("-" * 52)
    for t in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        caught = sum(p >= t for p in food_p)
        wrong = sum(p >= t for p in animal_p)
        fc = f"{caught}/{len(food_p)} ({100*caught/max(1,len(food_p)):.0f}%)"
        wf = f"{wrong}/{len(animal_p)} ({100*wrong/max(1,len(animal_p)):.0f}%)"
        star = "  <- config" if abs(t - args.threshold) < 1e-6 else ""
        print(f"{t:>9.2f} | {fc:>12} | {wf:>24}{star}")
    print("\nPick the highest 'food caught' you can get while 'animals wrongly flagged' stays ~0.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="General-CLIP non-animal prefilter (validation CLI).")
    p.add_argument("--sample", action="store_true",
                   help="Score real food vs known-animal crops and print a threshold sweep (no writes).")
    p.add_argument("--device", default="cuda", help="cuda or cpu.")
    p.add_argument("--model", default=config.CONFIG.nonanimal_model)
    p.add_argument("--pretrained", default=config.CONFIG.nonanimal_pretrained)
    p.add_argument("--threshold", type=float, default=config.CONFIG.nonanimal_threshold)
    p.add_argument("--limit", type=int, default=300, help="Max crops per cohort (default 300).")
    args = p.parse_args()

    conn = db.connect(config.CONFIG.db_path)
    try:
        if args.sample:
            return run_sample(conn, args)
        p.print_help()
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
