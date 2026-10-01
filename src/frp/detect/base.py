"""Detector protocol, the ``Face`` result type and multi-face selection policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from frp.errors import MultipleFacesError, NoFaceError
from frp.imageio import LoadedImage

# Landmark order everywhere in this project: left eye, right eye, nose, left mouth, right mouth
# ("left" is the subject's left, which appears on the image's left for a frontal face).
LANDMARK_NAMES = ("left_eye", "right_eye", "nose", "left_mouth", "right_mouth")


@dataclass(frozen=True, eq=False)
class Face:
    """One detected face. Coordinates are pixels in the coordinate frame of the input image."""

    box: NDArray[np.float32]  # (4,) x1, y1, x2, y2, clipped to the image
    landmarks: NDArray[np.float32]  # (5, 2) x, y; NOT clipped (may lie outside the image)
    score: float
    visible_fraction: float = 1.0  # clipped box area / raw box area
    landmarks_inside: bool = True
    # The model's box reaches the image border. For a face cut off by the frame the model tends
    # to predict a box that ends exactly at the edge, so this catches what visible_fraction misses.
    touches_border: bool = False

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])

    @property
    def size_px(self) -> float:
        """Shorter side of the (clipped) box."""
        return min(self.width, self.height)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def truncated(self) -> bool:
        """True when the face may be partially out of frame (conservative)."""
        return self.touches_border or (not self.landmarks_inside) or self.visible_fraction < 0.9

    def scaled(self, sx: float, sy: float) -> Face:
        factor = np.array([sx, sy], dtype=np.float32)
        return Face(
            box=(self.box.reshape(2, 2) * factor).reshape(4).astype(np.float32),
            landmarks=(self.landmarks * factor).astype(np.float32),
            score=self.score,
            visible_fraction=self.visible_fraction,
            landmarks_inside=self.landmarks_inside,
            touches_border=self.touches_border,
        )


class FaceDetector(Protocol):
    """Anything that finds faces in an RGB ``uint8`` image of shape (H, W, 3)."""

    def detect(self, image: NDArray[np.uint8]) -> list[Face]:
        """Return faces sorted by descending score, in the coordinate frame of ``image``."""
        ...


class FacePolicy(StrEnum):
    """What to do when an image contains more than one face."""

    LARGEST = "largest"  # keep only the largest face (by box area)
    ALL = "all"  # keep every face
    REJECT = "reject"  # raise MultipleFacesError


def select_faces(faces: list[Face], policy: FacePolicy = FacePolicy.LARGEST) -> list[Face]:
    """Apply the multi-face policy. Raises ``NoFaceError`` if ``faces`` is empty."""
    if not faces:
        raise NoFaceError("No face detected.")
    if policy is FacePolicy.ALL:
        return sorted(faces, key=lambda f: f.area, reverse=True)
    if policy is FacePolicy.REJECT and len(faces) > 1:
        raise MultipleFacesError(f"{len(faces)} faces detected; exactly one is required.")
    return [max(faces, key=lambda f: f.area)]


def detect_in_original_space(detector: FaceDetector, loaded: LoadedImage) -> list[Face]:
    """Run ``detector`` on a loaded image and map results back to original-image coordinates."""
    sx, sy = loaded.scale
    return [face.scaled(1.0 / sx, 1.0 / sy) for face in detector.detect(loaded.array)]
