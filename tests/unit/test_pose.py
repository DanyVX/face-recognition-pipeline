from __future__ import annotations

import numpy as np
import pytest

from frp.align import ARCFACE_TEMPLATE, AlignedFace, align_face, apply_transform
from frp.detect import Face
from frp.quality import POSE_PITCH, POSE_ROLL, POSE_YAW, GateContext, PoseGate, pose_from_aligned
from tests.synthetic import similarity_matrix

BLANK = np.full((700, 700, 3), 128, dtype=np.uint8)
IOD = float(np.linalg.norm(ARCFACE_TEMPLATE[1] - ARCFACE_TEMPLATE[0]))


def aligned_for(template_like: np.ndarray, *, angle: float = 0.0) -> AlignedFace:
    """Place ``template_like`` landmarks in the image (rotated by ``angle``) and align them."""
    m = similarity_matrix(1.5, angle, 300.0, 300.0)
    src = apply_transform(m, template_like).astype(np.float32)
    return align_face(BLANK, src, max_fit_error=1.0)


def context(aligned: AlignedFace) -> GateContext:
    face = Face(
        box=np.array([0, 0, 10, 10], dtype=np.float32), landmarks=aligned.landmarks, score=1.0
    )
    return GateContext(face=face, aligned=aligned, image_shape=BLANK.shape[:2])


def with_nose(dx: float = 0.0, dy_ratio: float | None = None) -> np.ndarray:
    pts = ARCFACE_TEMPLATE.copy()
    pts[2, 0] += dx * IOD
    if dy_ratio is not None:
        eye_y = pts[:2, 1].mean()
        mouth_y = pts[3:, 1].mean()
        pts[2, 1] = eye_y + dy_ratio * (mouth_y - eye_y)
    return pts


def test_frontal_face_has_near_zero_pose_proxies() -> None:
    pose = pose_from_aligned(aligned_for(ARCFACE_TEMPLATE))
    assert abs(pose.yaw_ratio) < 0.02
    assert abs(pose.pitch_ratio) < 0.02
    assert abs(pose.roll_deg) < 0.5


@pytest.mark.parametrize("angle", [-60.0, -30.0, 30.0, 90.0])
def test_roll_is_reported_and_does_not_leak_into_yaw_or_pitch(angle: float) -> None:
    pose = pose_from_aligned(aligned_for(ARCFACE_TEMPLATE, angle=angle))
    assert abs(abs(pose.roll_deg) - abs(angle)) < 1.0
    assert abs(pose.yaw_ratio) < 0.02
    assert abs(pose.pitch_ratio) < 0.02


def test_yaw_proxy_grows_with_nose_offset_and_gate_rejects_extreme_pose() -> None:
    gate = PoseGate()
    mild = gate.check(context(aligned_for(with_nose(dx=0.1))))
    assert mild.passed and mild.reason_code is None
    extreme = gate.check(context(aligned_for(with_nose(dx=0.6))))
    assert not extreme.passed
    assert extreme.reason_code == POSE_YAW
    assert extreme.value is not None and extreme.threshold is not None
    assert abs(extreme.value) > extreme.threshold


def test_yaw_sign_follows_the_direction_of_the_offset() -> None:
    right = pose_from_aligned(aligned_for(with_nose(dx=0.3))).yaw_ratio
    left = pose_from_aligned(aligned_for(with_nose(dx=-0.3))).yaw_ratio
    assert right > 0.2 and left < -0.2


def test_pitch_gate_rejects_nose_far_from_its_frontal_position() -> None:
    gate = PoseGate()
    down = gate.check(context(aligned_for(with_nose(dy_ratio=0.95))))
    assert not down.passed and down.reason_code == POSE_PITCH
    up = gate.check(context(aligned_for(with_nose(dy_ratio=0.15))))
    assert not up.passed and up.reason_code == POSE_PITCH


def test_roll_is_not_gated_by_default_but_can_be() -> None:
    upside_down = context(aligned_for(ARCFACE_TEMPLATE, angle=180.0))
    assert PoseGate().check(upside_down).passed  # alignment already uprights the crop
    strict = PoseGate(max_abs_roll_deg=45.0).check(upside_down)
    assert not strict.passed and strict.reason_code == POSE_ROLL


def test_thresholds_are_configurable_and_boundary_is_inclusive_of_the_limit() -> None:
    aligned = aligned_for(with_nose(dx=0.3))
    yaw = abs(pose_from_aligned(aligned).yaw_ratio)
    assert PoseGate(max_abs_yaw_ratio=yaw + 1e-6).check(context(aligned)).passed
    assert not PoseGate(max_abs_yaw_ratio=yaw - 1e-3).check(context(aligned)).passed
