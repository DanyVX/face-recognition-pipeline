"""Deterministic synthetic scene for alignment tests: known landmarks, no randomness."""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

from frp.align import ARCFACE_TEMPLATE

MARKER_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
SCENE_SIZE = (240, 200)  # width, height

# The similarity that maps the template into the scene (so aligning must invert it).
SCENE_SCALE = 1.6
SCENE_ANGLE_DEG = 12.0
SCENE_TRANSLATION = (70.0, 30.0)


def similarity_matrix(scale: float, angle_deg: float, tx: float, ty: float) -> NDArray[np.float64]:
    a = math.radians(angle_deg)
    c, s = scale * math.cos(a), scale * math.sin(a)
    return np.array([[c, -s, tx], [s, c, ty]], dtype=np.float64)


def scene_landmarks() -> NDArray[np.float32]:
    """The five landmarks of the synthetic scene, in scene pixel coordinates."""
    m = similarity_matrix(SCENE_SCALE, SCENE_ANGLE_DEG, *SCENE_TRANSLATION)
    return (ARCFACE_TEMPLATE.astype(np.float64) @ m[:, :2].T + m[:, 2]).astype(np.float32)


def synthetic_scene() -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """An RGB image with a colored disc at each landmark and an asymmetric line pattern."""
    width, height = SCENE_SIZE
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    image = np.stack(
        [(xs * 255 // (width - 1)), (ys * 255 // (height - 1)), np.full_like(xs, 90)], axis=-1
    ).astype(np.uint8)
    landmarks = scene_landmarks()
    # Asymmetric structure (eye line plus an extra stroke) so a flip or rotation cannot match.
    cv2.line(
        image,
        tuple(np.round(landmarks[0]).astype(int)),
        tuple(np.round(landmarks[3]).astype(int)),
        (255, 255, 255),
        2,
    )
    cv2.line(
        image,
        tuple(np.round(landmarks[1]).astype(int)),
        tuple(np.round(landmarks[2]).astype(int)),
        (0, 0, 0),
        2,
    )
    for point, color in zip(landmarks, MARKER_COLORS, strict=True):
        cv2.circle(image, (round(point[0]), round(point[1])), 4, color, -1)
    return image, landmarks
