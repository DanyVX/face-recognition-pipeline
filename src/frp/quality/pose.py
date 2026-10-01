"""Head-pose proxies computed from the five landmarks, and the pose gate.

These are geometric proxies, not calibrated angles in degrees: they are measured in the aligned
(roll- and scale-normalized) frame, where a frontal face has both ratios near zero. Thresholds
are provisional until M5 tunes them on a validation split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from frp.align import ARCFACE_TEMPLATE, AlignedFace
from frp.quality.base import GateContext, GateResult

POSE_YAW: Final = "POSE_YAW"
POSE_PITCH: Final = "POSE_PITCH"
POSE_ROLL: Final = "POSE_ROLL"

_T = ARCFACE_TEMPLATE
_FRONTAL_PITCH_RATIO: Final = float(
    (_T[2, 1] - _T[:2, 1].mean()) / (_T[3:, 1].mean() - _T[:2, 1].mean())
)


@dataclass(frozen=True)
class Pose:
    yaw_ratio: float  # nose offset from the eye midpoint / inter-ocular distance (0 = frontal)
    pitch_ratio: float  # nose position between eye and mouth lines, relative to the frontal value
    roll_deg: float  # rotation that was applied to upright the face (0 = already upright)


def pose_from_aligned(aligned: AlignedFace) -> Pose:
    """Pose proxies from a face's landmarks in the aligned frame."""
    lm: NDArray[np.float32] = aligned.landmarks
    eye_mid = lm[:2].mean(axis=0)
    iod = float(np.linalg.norm(lm[1] - lm[0]))
    mouth_mid = lm[3:].mean(axis=0)
    yaw = float((lm[2, 0] - eye_mid[0]) / iod)
    span = float(mouth_mid[1] - eye_mid[1])
    pitch = float((lm[2, 1] - eye_mid[1]) / span - _FRONTAL_PITCH_RATIO)
    m = aligned.matrix
    roll = math.degrees(math.atan2(m[1, 0], m[0, 0]))
    return Pose(yaw_ratio=yaw, pitch_ratio=pitch, roll_deg=roll)


@dataclass(frozen=True)
class PoseGate:
    """Rejects faces turned or tilted too far for reliable recognition.

    ``max_abs_roll_deg`` is ``None`` by default: roll is removed by the similarity alignment, so
    an upside-down face still produces an upright crop (limit it here if you do not want that).
    """

    max_abs_yaw_ratio: float = 0.35
    max_abs_pitch_ratio: float = 0.25
    max_abs_roll_deg: float | None = None
    name: str = "pose"

    def check(self, ctx: GateContext) -> GateResult:
        pose = pose_from_aligned(ctx.aligned)
        if abs(pose.yaw_ratio) > self.max_abs_yaw_ratio:
            return GateResult(self.name, False, POSE_YAW, pose.yaw_ratio, self.max_abs_yaw_ratio)
        if abs(pose.pitch_ratio) > self.max_abs_pitch_ratio:
            return GateResult(
                self.name, False, POSE_PITCH, pose.pitch_ratio, self.max_abs_pitch_ratio
            )
        if self.max_abs_roll_deg is not None and abs(pose.roll_deg) > self.max_abs_roll_deg:
            return GateResult(self.name, False, POSE_ROLL, pose.roll_deg, self.max_abs_roll_deg)
        return GateResult(self.name, True, None, pose.yaw_ratio, self.max_abs_yaw_ratio)
