"""
Tests for detector.resolve_device -- the rig's device-selection policy (cpu / cuda / auto, plus
input normalization). resolve_device gates on verify_cuda(), which actually probes the GPU with a
real matmul; every test here MONKEYPATCHES detector.verify_cuda so nothing ever touches a GPU. We
only assert the policy: which branch is taken, what pair comes back, and whether the CUDA error
propagates (cuda) or is swallowed into a CPU fallback (auto).
"""
from __future__ import annotations

import pytest

import detector


@pytest.fixture
def fake_cuda(monkeypatch):
    """Replace the real GPU probe with a controllable stub. Returns a small handle whose .calls
    counts invocations; set .raise_ = True to make it raise CudaUnavailableError (a wrong-arch /
    missing GPU), else it returns .name. Tests never hit a real GPU."""
    class _Probe:
        def __init__(self):
            self.calls = 0
            self.raise_ = False
            self.name = "Fake RTX 5050"

        def __call__(self):
            self.calls += 1
            if self.raise_:
                raise detector.CudaUnavailableError("no usable GPU\nsecond line of the fix")
            return self.name

    probe = _Probe()
    monkeypatch.setattr(detector, "verify_cuda", probe)
    return probe


# ---- 'cpu': never probes the GPU --------------------------------------------------------------
def test_resolve_device_cpu_skips_probe(fake_cuda):
    assert detector.resolve_device("cpu") == ("cpu", "CPU")
    assert fake_cuda.calls == 0                       # 'cpu' must not call verify_cuda at all


# ---- 'cuda': require the GPU; propagate its error -------------------------------------------
def test_resolve_device_cuda_returns_name(fake_cuda):
    assert detector.resolve_device("cuda") == ("cuda", "Fake RTX 5050")
    assert fake_cuda.calls == 1


def test_resolve_device_cuda_propagates_error(fake_cuda):
    fake_cuda.raise_ = True
    with pytest.raises(detector.CudaUnavailableError):
        detector.resolve_device("cuda")              # 'cuda' fails loud, never falls back


# ---- 'auto': GPU when it computes, else quiet CPU fallback ----------------------------------
def test_resolve_device_auto_uses_gpu_when_available(fake_cuda):
    assert detector.resolve_device("auto") == ("cuda", "Fake RTX 5050")
    assert fake_cuda.calls == 1


def test_resolve_device_auto_falls_back_to_cpu(fake_cuda):
    fake_cuda.raise_ = True
    assert detector.resolve_device("auto") == ("cpu", "CPU")   # swallows CudaUnavailableError
    assert fake_cuda.calls == 1


# ---- unknown / malformed device strings ----------------------------------------------------
def test_resolve_device_unknown_raises_value_error(fake_cuda):
    with pytest.raises(ValueError):
        detector.resolve_device("bogus")
    assert fake_cuda.calls == 0                       # rejected before any probe


@pytest.mark.parametrize("device", ["", None, "  CUDA  ", "Cuda", "\tcuda\n"])
def test_resolve_device_defaults_to_cuda(fake_cuda, device):
    """Blank / None / whitespace / mixed-case all normalize to the 'cuda' path: (device or 'cuda')
    then .strip().lower(). Verified by the monkeypatched probe returning the GPU name + being hit."""
    assert detector.resolve_device(device) == ("cuda", "Fake RTX 5050")
    assert fake_cuda.calls == 1
