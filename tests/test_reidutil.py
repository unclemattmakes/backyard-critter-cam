"""Tests for reidutil.py -- the shared re-ID numeric helpers (vector decode, cohesion, clustering).

These pin the exact behaviour the re-ID modules (reid/individuals) rely on, so the consolidation
that replaced the copy-pasted primitives is verifiably equivalent. Pure numpy/scipy; no GPU/model.
"""
from __future__ import annotations

import numpy as np

import reidutil


def _unit(*v):
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_decode_vector_roundtrips_float32():
    v = _unit(1.0, 2.0, 3.0)
    out = reidutil.decode_vector(v.tobytes())
    assert out.dtype == np.float32
    assert np.allclose(out, v)


def test_mean_pairwise_cosine_identical_rows_is_one():
    X = np.stack([_unit(1, 0, 0)] * 4)
    assert reidutil.mean_pairwise_cosine(X) == 1.0


def test_mean_pairwise_cosine_orthogonal_rows_is_zero():
    X = np.stack([_unit(1, 0, 0), _unit(0, 1, 0)])
    assert abs(reidutil.mean_pairwise_cosine(X)) < 1e-6


def test_mean_pairwise_cosine_singleton_and_empty_are_zero():
    assert reidutil.mean_pairwise_cosine(np.stack([_unit(1, 0, 0)])) == 0.0
    assert reidutil.mean_pairwise_cosine(np.zeros((0, 3), dtype=np.float32)) == 0.0


def test_mean_pairwise_cosine_matches_the_old_inline_formula():
    """Equivalence guard: the previous code computed (sims.sum() - n) / max(n*(n-1), 1)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 8)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    sims = X @ X.T
    n = len(X)
    old = float((sims.sum() - n) / max(n * (n - 1), 1))
    assert reidutil.mean_pairwise_cosine(X) == old


def test_cluster_cosine_separates_two_tight_groups_from_vectors():
    # Two clusters of near-identical unit vectors, far apart on the sphere.
    a = _unit(1, 0, 0)
    b = _unit(0, 1, 0)
    X = np.stack([a, a, a, b, b])
    labels = reidutil.cluster_cosine(X, threshold=0.3)
    assert len(labels) == 5
    assert len(set(labels)) == 2
    assert labels[0] == labels[1] == labels[2]      # the three 'a's group together
    assert labels[3] == labels[4]                   # the two 'b's group together
    assert labels[0] != labels[3]


def test_cluster_cosine_from_precomputed_distance_matches_vector_path():
    a, b = _unit(1, 0, 0), _unit(0, 1, 0)
    X = np.stack([a, a, b, b])
    S = X @ X.T
    D = np.clip(1.0 - S, 0.0, None)
    np.fill_diagonal(D, 0.0)
    from_vec = reidutil.cluster_cosine(X, threshold=0.3)
    from_dist = reidutil.cluster_cosine(dist=D, threshold=0.3)
    # Same partition (cluster ids may be permuted, so compare the grouping, not the raw labels).
    def groups(lbls):
        return {frozenset(i for i in range(len(lbls)) if lbls[i] == c) for c in set(lbls)}
    assert groups(from_vec) == groups(from_dist)


def test_cluster_cosine_singleton_returns_ones():
    assert list(reidutil.cluster_cosine(np.stack([_unit(1, 0, 0)]), threshold=0.3)) == [1]
    assert list(reidutil.cluster_cosine(dist=np.zeros((1, 1)), threshold=0.3)) == [1]


def test_cluster_cosine_constrained_matches_unconstrained_without_cannot():
    a, b = _unit(1, 0, 0), _unit(0, 1, 0)
    X = np.stack([a, a, b, b])
    S = X @ X.T
    D = np.clip(1.0 - S, 0.0, None)
    np.fill_diagonal(D, 0.0)
    labels = reidutil.cluster_cosine_constrained(D, 0.3)
    def groups(lbls):
        return {frozenset(i for i in range(len(lbls)) if lbls[i] == c) for c in set(lbls)}
    assert groups(labels) == {frozenset({0, 1}), frozenset({2, 3})}


def test_cluster_cosine_constrained_cannot_link_holds_lookalikes_apart():
    # Three identical vectors (distance 0 everywhere) -- unconstrained clustering would make one
    # blob. The cannot-link between 0 and 1 (same-frame tracklets = two bodies) must hold them
    # apart, while 2 still joins one of them.
    D = np.zeros((3, 3))
    labels = reidutil.cluster_cosine_constrained(D, 0.3, cannot={(0, 1)})
    assert labels[0] != labels[1]
    assert len(set(labels)) == 2


def test_cluster_cosine_constrained_singleton_and_empty():
    assert list(reidutil.cluster_cosine_constrained(np.zeros((1, 1)), 0.3)) == [1]
    assert list(reidutil.cluster_cosine_constrained(np.zeros((0, 0)), 0.3)) == []


def test_session_nuisance_basis_finds_the_scene_not_the_animal():
    # Two animals (dims 0, 1) seen on three nights, each night with its own scene offset (dims
    # 2-4) and BOTH animals present: the session means differ only by scene, so the basis must
    # lie entirely in the scene dims.
    rows, sessions = [], []
    for night, scene_dim in ((1, 2), (2, 3), (3, 4)):
        for animal in (0, 1):
            v = np.zeros(5)
            v[animal], v[scene_dim] = 1.0, 2.0
            rows.append(v)
            sessions.append(("cam", night))
    U = reidutil.session_nuisance_basis(rows, sessions, 2)
    assert U.shape == (5, 2)
    assert np.allclose(U.T @ U, np.eye(2))
    assert np.allclose(U[:2], 0.0)


def test_session_nuisance_basis_skips_lone_visits_and_caps_k():
    rows = [_unit(1, 0, 0), _unit(0, 1, 0), _unit(1, 1, 0), _unit(0, 0, 1)]
    # One two-visit session plus two lone visits: a single usable session has no spread.
    assert reidutil.session_nuisance_basis(rows, ["a", "a", "b", "c"], 4).shape == (3, 0)
    # Two usable sessions give at most one direction, whatever k asks for; None never counts.
    U = reidutil.session_nuisance_basis(rows, ["a", "a", "b", "b"], 4)
    assert U.shape == (3, 1)
    assert reidutil.session_nuisance_basis(rows, [None] * 4, 2).shape == (3, 0)


def test_project_out_removes_the_span_and_renormalizes():
    U = np.array([[0.0], [0.0], [1.0]])
    Y = reidutil.project_out(np.stack([_unit(1, 0, 1), _unit(0, 2, 3)]), U)
    assert np.allclose(Y[:, 2], 0.0)
    assert np.allclose(np.linalg.norm(Y, axis=1), 1.0)
    X = np.stack([_unit(1, 2, 3)])
    assert np.allclose(reidutil.project_out(X, np.zeros((3, 0))), X)
