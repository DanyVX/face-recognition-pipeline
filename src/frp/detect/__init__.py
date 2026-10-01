"""Face detection: protocol, result type, policies and the SCRFD backend."""

from frp.detect.base import (
    LANDMARK_NAMES,
    Face,
    FaceDetector,
    FacePolicy,
    detect_in_original_space,
    select_faces,
)
from frp.detect.scrfd import ScrfdDetector

__all__ = [
    "LANDMARK_NAMES",
    "Face",
    "FaceDetector",
    "FacePolicy",
    "ScrfdDetector",
    "detect_in_original_space",
    "select_faces",
]
