"""
Tests for quality.score_crop -- the per-crop shot-quality score that lets the dashboard lead a
visit with its sharpest / cutest frame. Pure cv2/numpy, no DB or camera.

The score is sharpness (Laplacian variance) with a night-eyeshine boost, so we verify: a crisp crop
beats its blurred self, an empty crop is 0, and a dark crop with bright "eye" specks beats the same
dark crop without them.
"""
from __future__ import annotations

import cv2
import numpy as np

import quality


def test_sharper_crop_scores_higher_than_blurred():
    rng = np.random.default_rng(0)
    sharp = (rng.random((120, 120, 3)) * 255).astype(np.uint8)   # high-frequency detail
    blurry = cv2.GaussianBlur(sharp, (11, 11), 0)                # same image, defocused
    assert quality.score_crop(sharp) > quality.score_crop(blurry)


def test_empty_or_none_crop_is_zero():
    assert quality.score_crop(None) == 0.0
    assert quality.score_crop(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0


def test_night_eyeshine_boosts_a_dark_crop():
    """A dark crop with bright specular specks (eyes catching the light) outscores the same dark
    crop without them -- the night 'facing the camera' bonus."""
    base = np.full((80, 80, 3), 20, dtype=np.uint8)
    base[::4, ::4] = 60                         # faint texture so sharpness isn't identically zero
    eyes = base.copy()
    eyes[30:34, 28:32] = 255                    # two bright "eye" blobs
    eyes[30:34, 48:52] = 255
    assert quality.score_crop(eyes) > quality.score_crop(base)


def test_score_is_non_negative_float():
    val = quality.score_crop(np.full((50, 50, 3), 128, dtype=np.uint8))
    assert isinstance(val, float) and val >= 0.0
