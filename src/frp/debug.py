"""Visual debugging helpers (used by ``frp debug-align``)."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from frp.align import ALIGNED_SIZE, ARCFACE_TEMPLATE, AlignedFace
from frp.detect import LANDMARK_NAMES, Face

_LANDMARK_COLORS = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 160, 0), (200, 0, 200)]  # RGB
_PANEL_HEIGHT = 336
_FOOTER_HEIGHT = 66
OVERLAY_HEIGHT = _PANEL_HEIGHT + _FOOTER_HEIGHT
_CROP_SCALE = 3


def _text(
    img: NDArray[np.uint8], text: str, origin: tuple[int, int], color: tuple[int, int, int]
) -> None:
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def detection_panel(image: NDArray[np.uint8], faces: list[Face]) -> NDArray[np.uint8]:
    """The input image with each face's box and labelled landmarks, scaled to a fixed height."""
    canvas = image.copy()
    thickness = max(1, round(max(image.shape[:2]) / 400))
    for face in faces:
        x1, y1, x2, y2 = (round(v) for v in face.box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), thickness)
        for (x, y), color in zip(face.landmarks, _LANDMARK_COLORS, strict=True):
            cv2.circle(canvas, (round(x), round(y)), thickness + 1, color, -1)
        _text(canvas, f"{face.score:.2f}", (x1, max(12, y1 - 5)), (0, 220, 0))
    scale = _PANEL_HEIGHT / canvas.shape[0]
    width = max(1, round(canvas.shape[1] * scale))
    return np.ascontiguousarray(
        cv2.resize(canvas, (width, _PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
    )


def aligned_panel(aligned: AlignedFace | None, message: str | None = None) -> NDArray[np.uint8]:
    """The 112x112 crop enlarged, with template points (green rings) and mapped landmarks (dots)."""
    side = ALIGNED_SIZE * _CROP_SCALE
    if aligned is None:
        panel = np.full((_PANEL_HEIGHT, side, 3), 40, dtype=np.uint8)
        _text(panel, "alignment failed", (10, 20), (255, 255, 255))
        if message:
            for i, line in enumerate(_wrap(message, 38)):
                _text(panel, line, (10, 44 + 18 * i), (255, 200, 120))
        return panel
    panel = np.full((_PANEL_HEIGHT, side, 3), 40, dtype=np.uint8)
    crop = cv2.resize(aligned.image, (side, side), interpolation=cv2.INTER_NEAREST)
    panel[:side] = crop
    for point in ARCFACE_TEMPLATE * _CROP_SCALE:
        cv2.circle(panel, (round(point[0]), round(point[1])), 7, (0, 255, 0), 1)
    for point, color in zip(aligned.landmarks * _CROP_SCALE, _LANDMARK_COLORS, strict=True):
        cv2.circle(panel, (round(point[0]), round(point[1])), 3, color, -1)
    return panel


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines


def alignment_overlay(
    image: NDArray[np.uint8],
    faces: list[Face],
    aligned: AlignedFace | None,
    message: str | None = None,
) -> NDArray[np.uint8]:
    """Side-by-side debug picture: detections on the input, and the aligned crop."""
    left = detection_panel(image, faces)
    right = aligned_panel(aligned, message)
    gap = np.full((_PANEL_HEIGHT, 8, 3), 255, dtype=np.uint8)
    top = np.concatenate([left, gap, right], axis=1)
    footer = np.full((_FOOTER_HEIGHT, top.shape[1], 3), 238, dtype=np.uint8)
    lines = []
    if aligned is not None:
        lines.append(f"fit error {aligned.fit_error:.3f} (RMS landmark residual / template IOD)")
        lines.append("green ring = template point, dot = detected landmark")
    lines.append("landmark order: " + ", ".join(LANDMARK_NAMES))
    for i, line in enumerate(lines):
        cv2.putText(
            footer,
            line,
            (8, 18 + 20 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return np.ascontiguousarray(np.concatenate([top, footer], axis=0))
