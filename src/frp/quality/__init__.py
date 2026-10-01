"""Quality gates: each returns a reason code, not just pass/fail."""

from frp.quality.base import Gate, GateContext, GateResult
from frp.quality.features import QualityFeatures, extract_features
from frp.quality.gates import (
    BLURRY,
    FACE_CUT_OFF,
    FACE_TOO_SMALL,
    LOW_CONFIDENCE,
    LOW_CONTRAST,
    TOO_BRIGHT,
    TOO_DARK,
    ConfidenceGate,
    ExposureGate,
    FramingGate,
    QualityReport,
    QualityThresholds,
    SharpnessGate,
    SizeGate,
    assess,
    build_gates,
)
from frp.quality.pose import POSE_PITCH, POSE_ROLL, POSE_YAW, Pose, PoseGate, pose_from_aligned

__all__ = [
    "BLURRY",
    "FACE_CUT_OFF",
    "FACE_TOO_SMALL",
    "LOW_CONFIDENCE",
    "LOW_CONTRAST",
    "POSE_PITCH",
    "POSE_ROLL",
    "POSE_YAW",
    "TOO_BRIGHT",
    "TOO_DARK",
    "ConfidenceGate",
    "ExposureGate",
    "FramingGate",
    "Gate",
    "GateContext",
    "GateResult",
    "Pose",
    "PoseGate",
    "QualityFeatures",
    "QualityReport",
    "QualityThresholds",
    "SharpnessGate",
    "SizeGate",
    "assess",
    "build_gates",
    "extract_features",
    "pose_from_aligned",
]
