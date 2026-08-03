"""Shared numeric helpers for the re-ID modules (embed / reid / individuals / clipembed).

Deliberately separate from db.py so the stdlib-only web-dashboard import path stays numpy-free --
these are imported only where numpy / scipy are already loaded. Pure functions; unit-tested in
tests/test_reidutil.py. Each consolidates a primitive that was previously copy-pasted across the
re-ID code (per the code-review's shared-helper sweep): the float32 vector decode, the in-cluster
cohesion score, and agglomerative clustering on cosine distance.
"""
from __future__ import annotations

import numpy as np


def decode_vector(blob) -> np.ndarray:
    """Decode a stored embedding BLOB back to its float32 vector. Embeddings are written
    L2-normalized, so a dot product between two decoded vectors is their cosine similarity."""
    return np.frombuffer(blob, dtype=np.float32)


def mean_pairwise_cosine(X) -> float:
    """Mean off-diagonal cosine similarity among the rows of `X` -- an in-cluster 'cohesion' score
    (1.0 = identical-looking). Rows must be L2-normalized, so the diagonal of X @ X.T is 1. Fewer
    than two rows returns 0.0; callers that prefer 1.0 for a lone member special-case that."""
    n = len(X)
    if n < 2:
        return 0.0
    sims = X @ X.T
    return float((sims.sum() - n) / (n * (n - 1)))


def cluster_cosine(X=None, *, dist=None, threshold, method="average"):
    """Agglomerative clustering -> 1-based fcluster labels (criterion='distance'), aligned to the
    input order. Provide EITHER `X` (rows of L2-normalized vectors, clustered on cosine distance)
    OR `dist` (a precomputed square distance matrix). Fewer than two samples returns all-ones (one
    trivial cluster), matching scipy's degenerate case without raising."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist, squareform
    if dist is not None:
        n = len(dist)
        if n < 2:
            return np.ones(n, dtype=int)
        condensed = squareform(np.asarray(dist, dtype=float), checks=False)
    else:
        n = 0 if X is None else len(X)
        if n < 2:
            return np.ones(n, dtype=int)
        condensed = pdist(X, metric="cosine")
    return fcluster(linkage(condensed, method=method), t=threshold, criterion="distance")


def cluster_cosine_constrained(dist, threshold, cannot=()):
    """Agglomerative clustering (average linkage) with CANNOT-LINK constraints -> 1-based labels
    aligned to the input order, matching cluster_cosine's contract. `dist` is a square distance
    matrix; `cannot` is an iterable of (i, j) index pairs that must never share a cluster.

    Built for still-tracklet un-blend, where the constraints are free and absolute: two tracklets
    observed in the SAME frame are two different animals, whatever their appearance similarity
    says (littermate kits can look near-identical -- the constraint is what keeps them apart when
    the embedding can't). Plain greedy merge instead of scipy: repeatedly join the closest pair of
    clusters whose average distance is below `threshold` and which no cannot-link pair spans.
    O(n^3) worst case, fine at un-blend scale (tens of tracklets per visit)."""
    n = len(dist)
    if n < 2:
        return np.ones(n, dtype=int)
    D = np.asarray(dist, dtype=float)
    forbid = {(min(i, j), max(i, j)) for i, j in cannot if i != j}
    clusters = [{i} for i in range(n)]
    while len(clusters) > 1:
        best = None                      # (avg_dist, a, b) of the closest mergeable pair
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if any((min(i, j), max(i, j)) in forbid
                       for i in clusters[a] for j in clusters[b]):
                    continue
                d = float(np.mean([D[i, j] for i in clusters[a] for j in clusters[b]]))
                if d < threshold and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is None:
            break
        _, a, b = best
        clusters[a] |= clusters[b]
        del clusters[b]
    labels = np.zeros(n, dtype=int)
    for l, members in enumerate(clusters, start=1):
        for i in members:
            labels[i] = l
    return labels
