"""5-point similarity alignment to the canonical 112x112 ArcFace template.

The transform is the closed-form least-squares similarity (Umeyama, rotation + uniform scale +
translation, no reflection), so it is deterministic and has no RANSAC randomness. Inputs that
cannot produce a trustworthy crop raise :class:`frp.errors.InvalidLandmarksError` with a stable
``reason`` rather than yielding a garbage face for the embedder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from frp.errors import InvalidLandmarksError

ALIGNED_SIZE: Final = 112

# Canonical template for 112x112 ArcFace crops: left eye, right eye, nose, left mouth, right mouth.
ARCFACE_TEMPLATE: Final[NDArray[np.float32]] = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
_TEMPLATE_IOD: Final = float(np.linalg.norm(ARCFACE_TEMPLATE[1] - ARCFACE_TEMPLATE[0]))

# Reasons carried by InvalidLandmarksError.reason
BAD_SHAPE: Final = "BAD_SHAPE"
NOT_FINITE: Final = "NOT_FINITE"
DEGENERATE: Final = "DEGENERATE"
OUT_OF_IMAGE: Final = "OUT_OF_IMAGE"
BAD_FIT: Final = "BAD_FIT"

# Fit error above this fraction of the template inter-ocular distance means the five points do
# not form a face-like layout (for example mirrored left/right order). PROVISIONAL: picked from
# a handful of measurements (real detections of one portrait and its flips/rotations: 0.07-0.31;
# mirrored left/right order: 0.66), so it sits between the two. M5 recalibrates it on a
# validation split; until then treat it as a coarse sanity check, not a quality score.
DEFAULT_MAX_FIT_ERROR: Final = 0.4
# Smallest singular value / largest of the centered points below this means (near-)collinear.
_DEGENERACY_RATIO: Final = 1e-2


@dataclass(frozen=True, eq=False)
class AlignedFace:
    """A 112x112 RGB crop plus how it was produced."""

    image: NDArray[np.uint8]  # (112, 112, 3) RGB
    matrix: NDArray[np.float64]  # (2, 3) forward transform: source image -> aligned crop
    landmarks: NDArray[np.float32]  # (5, 2) source landmarks mapped into the crop
    fit_error: float  # RMS residual to the template, as a fraction of its inter-ocular distance


def _validate(landmarks: NDArray[np.floating]) -> NDArray[np.float64]:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (5, 2):
        raise InvalidLandmarksError(
            f"Expected landmarks of shape (5, 2), got {points.shape}.", BAD_SHAPE
        )
    if not np.isfinite(points).all():
        raise InvalidLandmarksError("Landmarks contain NaN or infinite values.", NOT_FINITE)
    return points


def estimate_similarity(
    landmarks: NDArray[np.floating],
    template: NDArray[np.floating] = ARCFACE_TEMPLATE,
) -> NDArray[np.float64]:
    """Least-squares similarity transform mapping ``landmarks`` onto ``template``.

    Returns a (2, 3) matrix ``M`` such that ``template ~= landmarks @ M[:, :2].T + M[:, 2]``.
    """
    src = _validate(landmarks)
    dst = np.asarray(template, dtype=np.float64)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean

    singular = np.linalg.svd(src_c, compute_uv=False)
    if singular[0] <= 1e-9 or singular[-1] / singular[0] < _DEGENERACY_RATIO:
        raise InvalidLandmarksError(
            "Landmarks are degenerate (coincident or collinear points).", DEGENERATE
        )

    n = src.shape[0]
    cov = dst_c.T @ src_c / n
    u, s, vt = np.linalg.svd(cov)
    sign = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:  # forbid reflections
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    scale = float((s * sign).sum() / ((src_c**2).sum() / n))
    translation = dst_mean - scale * rotation @ src_mean
    return np.hstack([scale * rotation, translation[:, None]])


def apply_transform(
    matrix: NDArray[np.floating], points: NDArray[np.floating]
) -> NDArray[np.float64]:
    pts = np.asarray(points, dtype=np.float64)
    return pts @ np.asarray(matrix, dtype=np.float64)[:, :2].T + np.asarray(matrix)[:, 2]


def align_face(
    image: NDArray[np.uint8],
    landmarks: NDArray[np.floating],
    *,
    size: int = ALIGNED_SIZE,
    allow_outside: bool = False,
    max_fit_error: float = DEFAULT_MAX_FIT_ERROR,
) -> AlignedFace:
    """Warp ``image`` (RGB, uint8) so that ``landmarks`` land on the ArcFace template.

    Args:
        allow_outside: accept landmarks lying outside the image (the missing area is zero-padded).
            Off by default because such a face is cut off by the frame.
        max_fit_error: reject layouts whose similarity-fit residual, relative to the template's
            inter-ocular distance, exceeds this (catches mirrored/shuffled landmark order).
    """
    if size != ALIGNED_SIZE:
        raise ValueError(f"Only the canonical {ALIGNED_SIZE}x{ALIGNED_SIZE} size is supported.")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a uint8 RGB array of shape (H, W, 3)")
    points = _validate(landmarks)
    height, width = image.shape[:2]
    outside = (
        (points[:, 0] < 0)
        | (points[:, 0] > width - 1)
        | (points[:, 1] < 0)
        | (points[:, 1] > height - 1)
    )
    if outside.any() and not allow_outside:
        raise InvalidLandmarksError(
            "One or more landmarks lie outside the image (face cut off by the frame).",
            OUT_OF_IMAGE,
        )

    matrix = estimate_similarity(points)
    mapped = apply_transform(matrix, points)
    residual = float(np.sqrt(((mapped - ARCFACE_TEMPLATE) ** 2).sum(axis=1).mean()))
    fit_error = residual / _TEMPLATE_IOD
    if fit_error > max_fit_error:
        raise InvalidLandmarksError(
            f"Landmarks do not form a face-like layout (fit error {fit_error:.2f} > "
            f"{max_fit_error:.2f}); check left/right order and detector output.",
            BAD_FIT,
        )
    crop = cv2.warpAffine(
        image,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return AlignedFace(
        image=np.ascontiguousarray(crop),
        matrix=matrix,
        landmarks=mapped.astype(np.float32),
        fit_error=fit_error,
    )
