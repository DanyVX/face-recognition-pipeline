from __future__ import annotations

import numpy as np
import pytest

from frp.detect import Face, FacePolicy
from frp.errors import (
    InvalidLandmarksError,
    LowQualityError,
    MultipleFacesError,
    NoFaceError,
)
from frp.imageio import LoadedImage
from frp.pipeline import FacePipeline
from frp.quality import GateContext, GateResult, PoseGate
from tests.synthetic import synthetic_scene


class FixedDetector:
    def __init__(self, faces: list[Face]) -> None:
        self.faces = faces

    def detect(self, image: np.ndarray) -> list[Face]:
        return self.faces


def scene_loaded() -> tuple[LoadedImage, Face]:
    scene, landmarks = synthetic_scene()
    box = np.array([40, 20, 200, 180], dtype=np.float32)
    loaded = LoadedImage(array=scene, original_size=(scene.shape[1], scene.shape[0]), format="PNG")
    return loaded, Face(box=box, landmarks=landmarks, score=0.9)


class AlwaysReject:
    name = "always"

    def check(self, ctx: GateContext) -> GateResult:
        return GateResult(self.name, False, "ALWAYS", 1.0, 0.0)


def test_prepare_returns_an_aligned_crop_and_counts_faces() -> None:
    loaded, face = scene_loaded()
    prepared = FacePipeline(FixedDetector([face])).prepare(loaded)
    assert prepared.aligned.image.shape == (112, 112, 3)
    assert prepared.n_faces == 1
    assert prepared.image_shape == loaded.array.shape[:2]


def test_no_face_raises() -> None:
    loaded, _ = scene_loaded()
    with pytest.raises(NoFaceError):
        FacePipeline(FixedDetector([])).prepare(loaded)


def test_policy_is_applied() -> None:
    loaded, face = scene_loaded()
    other = Face(
        box=np.array([0, 0, 20, 20], dtype=np.float32), landmarks=face.landmarks, score=0.99
    )
    detector = FixedDetector([other, face])
    assert FacePipeline(detector, policy=FacePolicy.LARGEST).prepare(loaded).n_faces == 2
    with pytest.raises(MultipleFacesError):
        FacePipeline(detector, policy=FacePolicy.REJECT).prepare(loaded)


def test_bad_landmarks_surface_as_invalid_landmarks() -> None:
    loaded, face = scene_loaded()
    bad = Face(box=face.box, landmarks=face.landmarks[[1, 0, 2, 4, 3]], score=0.9)
    with pytest.raises(InvalidLandmarksError):
        FacePipeline(FixedDetector([bad])).prepare(loaded)


def test_failed_gate_raises_low_quality_with_reason_codes() -> None:
    loaded, face = scene_loaded()
    with pytest.raises(LowQualityError) as info:
        FacePipeline(FixedDetector([face]), gates=[AlwaysReject()]).prepare(loaded)
    assert info.value.reasons == ("ALWAYS",)
    assert "ALWAYS" in str(info.value)


def test_passing_gates_are_reported() -> None:
    loaded, face = scene_loaded()
    prepared = FacePipeline(FixedDetector([face]), gates=[PoseGate()]).prepare(loaded)
    assert [r.gate for r in prepared.gate_results] == ["pose"]
    assert prepared.gate_results[0].passed
