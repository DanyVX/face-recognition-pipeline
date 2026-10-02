from __future__ import annotations

import numpy as np
import pytest
from bench.degrade import (
    brightness_shift,
    downscale,
    gaussian_blur,
    hflip,
    jpeg_roundtrip,
    occlude,
    rotate,
    standard_degradations,
)

from frp.quality.features import sharpness


def textured(size: int = 96, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)


def test_blur_reduces_sharpness_monotonically() -> None:
    img = textured()
    values = [sharpness(img)] + [sharpness(gaussian_blur(img, s)) for s in (1, 2, 4)]
    assert values == sorted(values, reverse=True)


def test_jpeg_quality_controls_the_damage() -> None:
    img = textured()
    errors = [float(np.abs(jpeg_roundtrip(img, q).astype(int) - img).mean()) for q in (90, 50, 10)]
    assert errors == sorted(errors) and errors[0] > 0
    assert jpeg_roundtrip(img, 90).shape == img.shape and jpeg_roundtrip(img, 90).dtype == np.uint8


def test_brightness_shift_clips_and_preserves_dtype() -> None:
    img = np.array([[[10, 128, 250]]], dtype=np.uint8)
    assert brightness_shift(img, 40).tolist() == [[[50, 168, 255]]]
    assert brightness_shift(img, -40).tolist() == [[[0, 88, 210]]]


def test_downscale_shrinks_the_image() -> None:
    assert downscale(textured(100), 0.5).shape == (50, 50, 3)
    assert downscale(textured(100), 0.25).shape == (25, 25, 3)
    assert downscale(textured(3), 0.1).shape == (1, 1, 3)  # never collapses to nothing
    with pytest.raises(ValueError):
        downscale(textured(), 1.5)


def test_rotation_keeps_size_and_zero_rotation_is_identity() -> None:
    img = textured()
    np.testing.assert_array_equal(rotate(img, 0.0), img)
    r = rotate(img, 30.0)
    assert r.shape == img.shape and (r[0, 0] == 0).all()  # a black corner appears


def test_flip_is_an_involution() -> None:
    img = textured()
    np.testing.assert_array_equal(hflip(hflip(img)), img)
    assert not np.array_equal(hflip(img), img)


@pytest.mark.parametrize("fraction", [0.1, 0.2, 0.4])
def test_occlusion_covers_the_requested_area_and_is_seeded(fraction: float) -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = occlude(img, fraction, np.random.default_rng(1))
    covered = float((out[..., 0] == 127).mean())
    assert covered == pytest.approx(fraction, abs=0.03)
    np.testing.assert_array_equal(out, occlude(img, fraction, np.random.default_rng(1)))
    assert not np.array_equal(out, occlude(img, fraction, np.random.default_rng(2)))
    assert (img == 0).all()  # input untouched
    with pytest.raises(ValueError):
        occlude(img, 1.0, np.random.default_rng(0))


def test_the_standard_grid_is_complete_and_every_entry_runs() -> None:
    grid = standard_degradations()
    assert {d.family for d in grid} == {
        "blur",
        "jpeg",
        "brightness",
        "downscale",
        "rotation",
        "flip",
        "occlusion",
    }
    assert len(grid) == len({d.name for d in grid})  # names are unique
    img = textured(64)
    for degradation in grid:
        out = degradation.apply(img, np.random.default_rng(0))
        assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3, degradation.name
