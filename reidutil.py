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
