"""Synthetic image degradations for the robustness slices (pure functions on RGB uint8 arrays)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


def gaussian_blur(image: Image, sigma: float) -> Image:
    return np.asarray(cv2.GaussianBlur(image, (0, 0), sigma), dtype=np.uint8)


def jpeg_roundtrip(image: Image, quality: int) -> Image:
    """Compress to JPEG at ``quality`` (1-100) and decode again."""
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not ok:  # pragma: no cover - cv2 only fails on invalid parameters
        raise ValueError("JPEG encoding failed")
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if decoded is None:  # pragma: no cover - we just encoded this buffer
        raise ValueError("JPEG decoding failed")
    return np.asarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def brightness_shift(image: Image, delta: int) -> Image:
    """Add ``delta`` to every channel (clipped to 0-255)."""
    return np.asarray(np.clip(image.astype(np.int16) + int(delta), 0, 255), dtype=np.uint8)


def downscale(image: Image, factor: float) -> Image:
    """Shrink by ``factor`` (0 < factor <= 1); the result is a smaller image, not re-enlarged."""
    if not 0.0 < factor <= 1.0:
        raise ValueError("factor must be in (0, 1]")
    h, w = image.shape[:2]
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    return np.asarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA), dtype=np.uint8)


def rotate(image: Image, degrees: float) -> Image:
    """Rotate about the center keeping the canvas size; uncovered corners become black."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, 1.0)
    return np.asarray(
        cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)),
        dtype=np.uint8,
    )


def hflip(image: Image) -> Image:
    return np.ascontiguousarray(image[:, ::-1])


def occlude(image: Image, area_fraction: float, rng: np.random.Generator) -> Image:
    """Cover a random rectangle with mid-gray. ``area_fraction`` is its share of the image area.

    The rectangle's aspect ratio is random in [0.5, 2]; its position is uniform among the places
    where it fits entirely inside the image.
    """
    if not 0.0 < area_fraction < 1.0:
        raise ValueError("area_fraction must be in (0, 1)")
    h, w = image.shape[:2]
    aspect = float(rng.uniform(0.5, 2.0))
    rect_h = int(min(h, max(1, round(np.sqrt(area_fraction * h * w / aspect)))))
    rect_w = int(min(w, max(1, round(area_fraction * h * w / rect_h))))
    top = int(rng.integers(0, h - rect_h + 1))
    left = int(rng.integers(0, w - rect_w + 1))
    out = image.copy()
    out[top : top + rect_h, left : left + rect_w] = 127
    return out


@dataclass(frozen=True)
class Degradation:
    """A named degradation at one severity."""

    family: str
    severity: str  # printable, e.g. "sigma=3"
    level: float  # numeric severity for plotting (ordered within a family)
    apply: Callable[[Image, np.random.Generator], Image]

    @property
    def name(self) -> str:
        return f"{self.family}:{self.severity}"


Transform = Callable[[Image, np.random.Generator], Image]


def _blur(sigma: float) -> Transform:
    return lambda image, _rng: gaussian_blur(image, sigma)


def _jpeg(quality: int) -> Transform:
    return lambda image, _rng: jpeg_roundtrip(image, quality)


def _brightness(delta: int) -> Transform:
    return lambda image, _rng: brightness_shift(image, delta)


def _downscale(factor: float) -> Transform:
    return lambda image, _rng: downscale(image, factor)


def _rotation(degrees: float) -> Transform:
    return lambda image, _rng: rotate(image, degrees)


def _occlusion(fraction: float) -> Transform:
    return lambda image, rng: occlude(image, fraction, rng)


def standard_degradations() -> list[Degradation]:
    """The robustness grid: blur, JPEG, brightness, downscale, rotation, flip and occlusion."""
    grid: list[Degradation] = []
    for sigma in (1.0, 2.0, 3.0, 5.0):
        grid.append(Degradation("blur", f"sigma={sigma:g}", sigma, _blur(sigma)))
    for q in (90, 70, 50, 30, 10):
        grid.append(Degradation("jpeg", f"q={q}", float(q), _jpeg(q)))
    for delta in (-80, -40, 40, 80):
        grid.append(Degradation("brightness", f"{delta:+d}", float(delta), _brightness(delta)))
    for factor in (0.75, 0.5, 0.35, 0.25):
        grid.append(Degradation("downscale", f"x{factor:g}", factor, _downscale(factor)))
    for degrees in (-30, -15, 15, 30):
        grid.append(Degradation("rotation", f"{degrees:+d}deg", float(degrees), _rotation(degrees)))
    grid.append(Degradation("flip", "horizontal", 1.0, lambda image, _rng: hflip(image)))
    for fraction in (0.1, 0.2, 0.3, 0.4):
        grid.append(Degradation("occlusion", f"{fraction:.0%}", fraction, _occlusion(fraction)))
    return grid
