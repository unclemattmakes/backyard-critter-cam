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


def session_nuisance_basis(vectors, sessions, k: int, *, min_per_session: int = 2) -> np.ndarray:
    """The top-`k` directions along which per-SESSION mean vectors spread -> a (d, k) orthonormal
    basis. A session is one camera on one night: one light, one white balance, one camera angle.
    Different animals seen in the same session share all of that, so the spread of session means
    is mostly scene, and projecting it out (project_out) removes what a probe would otherwise
    match on when it matches the yard rather than the animal. Label-free, so it can learn from
    every visit, named or not.

    Sessions with fewer than `min_per_session` vectors are skipped: a lone visit's "session mean"
    is just that animal, which would teach the basis identity. Two visits can still be the same
    animal, so on a night one raccoon owns, some identity leaks into the basis anyway; that is
    what the eval arm measures rather than assumes. k is capped at (usable sessions - 1)."""
    X = np.asarray(vectors, dtype=np.float64)
    d = X.shape[1] if X.ndim == 2 else 0
    groups: dict = {}
    for i, s in enumerate(sessions):
        if s is not None:
            groups.setdefault(s, []).append(i)
    keep = [ii for ii in groups.values() if len(ii) >= min_per_session]
    k = min(int(k), len(keep) - 1)
    if k <= 0:
        return np.zeros((d, 0))
    means = np.stack([X[ii].mean(axis=0) for ii in keep])
    n = np.array([len(ii) for ii in keep], dtype=np.float64)
    grand = (means * n[:, None]).sum(axis=0) / n.sum()
    _, _, vt = np.linalg.svd((means - grand) * np.sqrt(n)[:, None], full_matrices=False)
    return vt[:k].T


def project_out(X, U) -> np.ndarray:
    """Remove span(U) from each row of `X` and re-L2-normalize, so a dot product is still a
    cosine. An empty basis (k = 0) returns the rows unchanged."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if U.shape[1]:
        X = X - (X @ U) @ U.T
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (X / norms).astype(np.float32)
