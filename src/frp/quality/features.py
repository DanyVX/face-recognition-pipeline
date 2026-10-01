"""Image-quality measurements for one face, shared by the gates and the calibration script."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from frp.align import AlignedFace
from frp.quality.base import GateContext
from frp.quality.pose import pose_from_aligned


@dataclass(frozen=True)
class QualityFeatures:
    face_px: float  # shorter side of the face box, in pixels of the image it was found in
    det_score: float  # detector confidence (also a rough occlusion proxy)
    laplacian_var: float  # sharpness: variance of the Laplacian of the aligned grayscale crop
    brightness: float  # mean luminance of the aligned crop, 0-255
    contrast: float  # standard deviation of the luminance of the aligned crop
    yaw_ratio: float
    pitch_ratio: float
    fit_error: float
    truncated: bool

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def luminance(crop: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Rec. 601 luma of an RGB crop (float32, 0-255)."""
    rgb = crop.astype(np.float32)
    return np.asarray(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])


def sharpness(crop: NDArray[np.uint8]) -> float:
    """Variance of the Laplacian of the aligned crop (higher is sharper).

    Measured on the fixed-size 112x112 crop so values are comparable across source resolutions.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_features(aligned: AlignedFace) -> tuple[float, float, float]:
    """(laplacian_var, brightness, contrast) of an aligned crop."""
    luma = luminance(aligned.image)
    return sharpness(aligned.image), float(luma.mean()), float(luma.std())


def extract_features(ctx: GateContext) -> QualityFeatures:
    laplacian_var, brightness, contrast = crop_features(ctx.aligned)
    pose = pose_from_aligned(ctx.aligned)
    return QualityFeatures(
        face_px=float(ctx.face.size_px),
        det_score=float(ctx.face.score),
        laplacian_var=laplacian_var,
        brightness=brightness,
        contrast=contrast,
        yaw_ratio=pose.yaw_ratio,
        pitch_ratio=pose.pitch_ratio,
        fit_error=float(ctx.aligned.fit_error),
        truncated=bool(ctx.face.truncated),
    )
