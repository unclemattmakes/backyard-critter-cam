"""
Tests for clipfilter -- the general-CLIP non-animal gate that runs before BioCLIP.

Model-free by design (matches the suite: no GPU, no checkpoint download). We exercise the two
pieces that actually make the keep/reject call:
  * AnimalFilter.score_features -- the pure image-vs-prototype softmax, and
  * clipfilter.decision         -- the threshold cut.
Hand-built unit vectors stand in for real CLIP embeddings, so a crop pointing at the ANIMAL
prototype is kept and one pointing at the NON-ANIMAL prototype is rejected -- without ever
loading open_clip.
"""
from __future__ import annotations

import pytest

# The only hard third-party import in the suite that isn't cv2 or numpy. Skip rather than fail
# collection so a lean checkout (or a CPU-only CI job that skipped the 2 GB torch wheel) still
# runs the other 280-odd tests.
torch = pytest.importorskip("torch")

import clipfilter
from clipfilter import AnimalFilter, decision


def _protos():
    # Orthonormal stand-ins for the two text prototypes: row 0 = ANIMAL, row 1 = NON-ANIMAL.
    return torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


def test_score_features_routes_to_the_nearer_prototype():
    protos = _protos()
    feats = torch.tensor([
        [2.0, 0.0, 0.0, 0.0],   # clearly animal (un-normalized on purpose -> exercises the L2 norm)
        [0.0, 3.0, 0.0, 0.0],   # clearly non-animal
        [1.0, 1.0, 0.0, 0.0],   # right on the fence
    ])
    p_non = AnimalFilter.score_features(feats, protos, scale=100.0)[:, 1]
    assert float(p_non[0]) < 0.01                  # animal -> almost no non-animal mass
    assert float(p_non[1]) > 0.99                  # non-animal -> almost all of it
    assert abs(float(p_non[2]) - 0.5) < 0.01       # tie -> ~half


def test_probabilities_sum_to_one():
    probs = AnimalFilter.score_features(
        torch.tensor([[0.3, 0.7, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]), _protos(), scale=100.0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_decision_is_an_inclusive_ge_cut():
    assert decision(0.65, 0.65) is True            # boundary is inclusive (>=)
    assert decision(0.649, 0.65) is False
    assert decision(0.99, 0.65) is True
    assert decision(0.10, 0.65) is False


def test_label_is_hidden_by_the_denylist():
    """The gate's label must be filtered downstream, so it has to live in stats._NON_CRITTER.
    Guards against renaming NONANIMAL_LABEL in one place but not the other."""
    import stats
    assert clipfilter.NONANIMAL_LABEL.lower() in stats._NON_CRITTER
