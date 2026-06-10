"""
Phase 3 (part 2) -- re-identification: cluster appearance embeddings into candidate
individuals, and answer "which known crops does this one look like?".

Reads the L2-normalized MegaDescriptor vectors that embed.py stored in detection_embeddings
and does two things, both designed to AUGMENT the critter-knower rather than to be an oracle
(PLAN.md's core idea -- appearance is one axis; never collapse to a single "who is this?"):

  * cluster (default): agglomerative-cluster the crops on cosine distance into provisional
    individuals, then write a CONTACT-SHEET MONTAGE per cluster to reid/<species>/ so you can
    eyeball whether the model is actually separating your raccoons. Clustering only PROPOSES
    groups -- you name the real ones (that's the cheap human step the plan asks for). Nothing
    is written to the database unless you pass --write-clusters.

  * neighbors: given one crop, montage the N most visually similar crops in the corpus. This
    is the kernel of the live "closest appearance match: Notch 78%, Gimpy 41%" read-out, and
    the quickest way to see with your own eyes that the embedding pulls the same animal
    together.

Boring and robust: scipy hierarchical clustering on a cosine-distance matrix (no sklearn,
no training), PIL for the montages. The whole thing is a pure read over the vectors, so it's
re-runnable at any threshold without recomputing embeddings.

  python reid.py                              # cluster high-conf raccoons -> reid/raccoon/*.jpg
  python reid.py --threshold 0.40             # tighter clusters (raise to split, lower to merge)
  python reid.py --min-cluster-size 5         # only montage clusters with >=5 crops
  python reid.py --species "Virginia opossum" # opossums next (per the plan)
  python reid.py --neighbors 1234             # the 25 crops most similar to detection 1234
  python reid.py --write-clusters             # ALSO stamp provisional individual_id='cluster_NN'

After eyeballing, name a cluster for real:
  python reid.py --name cluster_03 Notch      # set individual_id='Notch' on every crop in cluster 3
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

import numpy as np

import config
import db
from embed import DEFAULT_MIN_CONFIDENCE, model_tag


class EmbeddingStore:
    """All stored vectors for one species/model, loaded once as a matrix for fast similarity."""

    def __init__(self, conn, species, min_confidence, model):
        self.model = model
        rows = db.load_embeddings(conn, model, species=species,
                                  min_confidence=min_confidence)
        self.ids = [r["id"] for r in rows]
        self.crop_paths = [r["crop_path"] for r in rows]
        self.confidences = [r["confidence"] for r in rows]
        self.timestamps = [r["timestamp"] for r in rows]
        self.individual_ids = [r["individual_id"] for r in rows]
        if rows:
            # Vectors are stored already L2-normalized, so X @ X.T is the cosine-similarity matrix.
            self.X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        else:
            self.X = np.zeros((0, 0), dtype=np.float32)
        self._index = {rid: i for i, rid in enumerate(self.ids)}

    def __len__(self):
        return len(self.ids)

    def neighbors(self, detection_id, k, min_gap_minutes=0.0):
        """(neighbor_id, crop_path, cosine_sim) for the k crops most similar to detection_id,
        nearest first, excluding the query itself. With min_gap_minutes > 0, also exclude crops
        captured within that many minutes of the query -- the same-visit burst of near-duplicate
        frames otherwise swamps the list (a lingering raccoon fires ~10 frames at 0.97+), hiding
        the genuinely informative CROSS-SESSION look-alikes underneath."""
        i = self._index.get(detection_id)
        if i is None:
            raise KeyError(f"detection {detection_id} has no '{self.model}' embedding "
                           f"(is it an above-gate animal crop? run embed.py first)")
        sims = self.X @ self.X[i]
        order = np.argsort(-sims)
        q_t = self._dt(i)
        out = []
        for j in order:
            if j == i:
                continue
            if min_gap_minutes and q_t is not None:
                t = self._dt(j)
                if t is not None and abs((t - q_t).total_seconds()) < min_gap_minutes * 60:
                    continue
            out.append((self.ids[j], self.crop_paths[j], float(sims[j])))
            if len(out) >= k:
                break
        return out

    def _dt(self, i):
        """Parse a member's timestamp to a datetime (None if unparseable)."""
        from datetime import datetime
        try:
            return datetime.fromisoformat(self.timestamps[i])
        except (ValueError, TypeError):
            return None

    def cluster(self, threshold, method):
        """Agglomerative clustering on cosine distance. Returns labels[] (1-based cluster id per
        crop, as scipy.fcluster gives) aligned to self.ids."""
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist
        if len(self) < 2:
            return np.ones(len(self), dtype=int)
        # X is L2-normalized -> 'cosine' pdist is 1 - cos_sim, in [0, 2].
        condensed = pdist(self.X, metric="cosine")
        Z = linkage(condensed, method=method)
        return fcluster(Z, t=threshold, criterion="distance")


def _grid_montage(items, out_path, *, cols=5, thumb=180, pad=6, label_h=16):
    """Write a labelled contact sheet. `items` = list of (crop_path, caption). Best-effort:
    unreadable crops are skipped, never fatal."""
    from PIL import Image, ImageDraw

    cells = []
    for cp, caption in items:
        try:
            im = Image.open(config.ROOT / cp.replace("\\", "/")).convert("RGB")
        except Exception:  # noqa: BLE001 -- a missing crop shouldn't sink the sheet
            continue
        im.thumbnail((thumb, thumb))
        cells.append((im, caption))
    if not cells:
        return False

    rows = (len(cells) + cols - 1) // cols
    cell_w, cell_h = thumb + pad, thumb + pad + label_h
    canvas = Image.new("RGB", (cols * cell_w + pad, rows * cell_h + pad), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    for idx, (im, caption) in enumerate(cells):
        r, c = divmod(idx, cols)
        x, y = pad + c * cell_w, pad + r * cell_h
        canvas.paste(im, (x + (thumb - im.width) // 2, y + (thumb - im.height) // 2))
        draw.text((x + 2, y + thumb + 2), caption, fill=(210, 210, 210))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=88)
    return True


def _cap(store, i):
    """Short caption for a crop: detection id, detector confidence, time of day."""
    ts = store.timestamps[i] or ""
    hm = ts[11:16] if len(ts) >= 16 else ""
    return f"#{store.ids[i]} {store.confidences[i]:.2f} {hm}"


def do_cluster(conn, store, args):
    if len(store) == 0:
        print(f"No '{store.model}' embeddings for species='{args.species}'. "
              f"Run: python embed.py --species \"{args.species}\""
              f"{' --segment' if args.segment else ''}")
        return 0

    labels = store.cluster(args.threshold, args.method)
    uniq, counts = np.unique(labels, return_counts=True)
    order = sorted(zip(uniq, counts), key=lambda kc: -kc[1])  # biggest clusters first

    species_slug = args.species.lower().replace(" ", "_").replace("'", "")
    out_dir = config.ROOT / "reid" / species_slug
    big = [(c, n) for c, n in order if n >= args.min_cluster_size]
    print(f"{len(store)} crops -> {len(uniq)} clusters at cosine-distance threshold "
          f"{args.threshold} ({args.method} linkage). "
          f"{len(big)} cluster(s) with >= {args.min_cluster_size} crops:")

    montaged = 0
    for rank, (cid, n) in enumerate(big, 1):
        members = [i for i in range(len(store)) if labels[i] == cid]
        # Cohesion = mean pairwise cosine sim within the cluster (1.0 = identical-looking).
        sub = store.X[members]
        sims = sub @ sub.T
        cohesion = float((sims.sum() - len(members)) / max(len(members) * (len(members) - 1), 1))
        # Show the most readable crops first (highest detector confidence).
        members.sort(key=lambda i: -store.confidences[i])
        shown = members[:args.max_per_montage]
        out_path = out_dir / f"cluster_{rank:02d}_n{n}.jpg"
        if _grid_montage([(store.crop_paths[i], _cap(store, i)) for i in shown], out_path):
            montaged += 1
        print(f"  cluster_{rank:02d}: {n:4d} crops  cohesion {cohesion:.2f}  -> {out_path.name}")

    if montaged:
        print(f"\nWrote {montaged} montage(s) to {out_dir}. Open them and see whether each "
              f"sheet is ONE raccoon.\nName a good one: python reid.py --name cluster_03 Notch")

    if args.write_clusters:
        # Placeholder individual_id per cluster, e.g. 'raccoon_c01'. Species-scoped so several
        # species can be clustered into the same column without colliding. These are PROPOSALS:
        # "these crops look alike", not confirmed individuals -- rename the real ones with --name.
        n_written = 0
        for rank, (cid, _) in enumerate(big, 1):
            members = [store.ids[i] for i in range(len(store)) if labels[i] == cid]
            n_written += db.set_individual_bulk(conn, members, f"{species_slug}_c{rank:02d}")
        print(f"\nStamped placeholder individual_id='{species_slug}_cNN' on {n_written} crop(s) "
              f"across {len(big)} clusters (--write-clusters). These group look-alikes; "
              f"rename real individuals with --name.")
    return 0


def do_neighbors(conn, store, args):
    nid = args.neighbors
    try:
        nbrs = store.neighbors(nid, args.k, min_gap_minutes=args.min_gap_minutes)
    except KeyError as e:
        print(e)
        return 1
    species_slug = (args.species or "all").lower().replace(" ", "_").replace("'", "")
    out_path = config.ROOT / "reid" / species_slug / f"neighbors_{nid}.jpg"
    items = [(store.crop_paths[store._index[nid]], f"QUERY #{nid}")]
    items += [(cp, f"#{i} sim {s:.2f}") for i, cp, s in nbrs]
    _grid_montage(items, out_path)
    print(f"Top {len(nbrs)} look-alikes for detection #{nid}:")
    for i, _cp, s in nbrs[:10]:
        print(f"  #{i:<6} cosine {s:.3f}")
    print(f"Montage (query first): {out_path}")
    return 0


def do_name(conn, store, args):
    """Rename a montage cluster ('cluster_03') to a real individual ('Notch'). Re-derives the
    same clustering used for the montages so the cluster numbers line up."""
    cluster_tag, name = args.name
    if not cluster_tag.startswith("cluster_"):
        print("First arg to --name is the montage tag, e.g. cluster_03.")
        return 1
    rank = int(cluster_tag.split("_")[1])

    labels = store.cluster(args.threshold, args.method)
    uniq, counts = np.unique(labels, return_counts=True)
    order = sorted(zip(uniq, counts), key=lambda kc: -kc[1])
    big = [(c, n) for c, n in order if n >= args.min_cluster_size]
    if rank < 1 or rank > len(big):
        print(f"No {cluster_tag} at the current threshold/min-size "
              f"({len(big)} clusters shown). Use the same --threshold you montaged with.")
        return 1
    cid = big[rank - 1][0]
    members = [store.ids[i] for i in range(len(store)) if labels[i] == cid]
    n = db.set_individual_bulk(conn, members, name)
    print(f"Set individual_id='{name}' on {n} crop(s) (was {cluster_tag}). "
          f"Re-run clustering or check the dashboard to confirm.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Phase 3: cluster re-ID embeddings into individuals; find look-alikes.")
    p.add_argument("--species", default="raccoon",
                   help="Species to work on (default 'raccoon'). Use 'all' to ignore species.")
    p.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                   help="Only consider crops with detector confidence >= this.")
    p.add_argument("--segment", action="store_true",
                   help="Use the SAM background-masked embeddings (embed.py --segment) instead "
                        "of the full-crop ones.")
    p.add_argument("--threshold", type=float, default=0.45,
                   help="Cosine-distance cut for clustering (0..2). Lower = fewer, looser "
                        "clusters; higher = more, tighter. Default 0.45.")
    p.add_argument("--method", default="average",
                   help="scipy linkage method: average (default), complete, ward, single.")
    p.add_argument("--min-cluster-size", type=int, default=3,
                   help="Ignore clusters smaller than this when writing montages (default 3).")
    p.add_argument("--max-per-montage", type=int, default=25,
                   help="Max crops drawn per cluster contact sheet (default 25).")
    p.add_argument("--neighbors", type=int, default=None, metavar="DETECTION_ID",
                   help="Instead of clustering, montage the crops most similar to this detection.")
    p.add_argument("--k", type=int, default=25, help="How many neighbors for --neighbors (default 25).")
    p.add_argument("--min-gap-minutes", type=float, default=30.0,
                   help="For --neighbors, exclude crops within this many minutes of the query so "
                        "the same-visit burst doesn't crowd out cross-session look-alikes (default 30).")
    p.add_argument("--write-clusters", action="store_true",
                   help="Also stamp provisional individual_id='cluster_NN' on each clustered crop.")
    p.add_argument("--name", nargs=2, metavar=("CLUSTER_TAG", "NAME"), default=None,
                   help="Rename a montage cluster to a real individual, e.g. --name cluster_03 Notch.")
    args = p.parse_args()

    species = None if args.species.lower() == "all" else args.species
    conn = db.connect(config.CONFIG.db_path)
    conn.row_factory = sqlite3.Row  # load_embeddings returns rows accessed by column name
    store = EmbeddingStore(conn, species, args.min_confidence, model_tag(args.segment))

    try:
        if args.name is not None:
            return do_name(conn, store, args)
        if args.neighbors is not None:
            return do_neighbors(conn, store, args)
        return do_cluster(conn, store, args)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
