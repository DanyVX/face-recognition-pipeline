"""M2 alignment: similarity transform, guards for bad landmarks, and the stored reference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from frp.align import (
    ALIGNED_SIZE,
    ARCFACE_TEMPLATE,
    align_face,
    apply_transform,
    estimate_similarity,
)
from frp.align.aligner import BAD_FIT, BAD_SHAPE, DEGENERATE, NOT_FINITE, OUT_OF_IMAGE
from frp.errors import InvalidLandmarksError
from tests.conftest import FIXTURES
from tests.synthetic import MARKER_COLORS, similarity_matrix, synthetic_scene

BLANK = np.full((600, 600, 3), 128, dtype=np.uint8)


def landmarks_from_template(scale: float, angle: float, tx: float, ty: float) -> np.ndarray:
    m = similarity_matrix(scale, angle, tx, ty)
    return apply_transform(m, ARCFACE_TEMPLATE).astype(np.float32)


# --- the transform itself ------------------------------------------------------------------------


def test_template_maps_to_itself() -> None:
    m = estimate_similarity(ARCFACE_TEMPLATE)
    np.testing.assert_allclose(m, [[1, 0, 0], [0, 1, 0]], atol=1e-6)


def test_recovers_a_known_transform() -> None:
    src = landmarks_from_template(1.6, 12.0, 70.0, 30.0)
    m = estimate_similarity(src)
    np.testing.assert_allclose(apply_transform(m, src), ARCFACE_TEMPLATE, atol=1e-3)
    # The recovered transform is the inverse of the one that generated the landmarks.
    forward = similarity_matrix(1.6, 12.0, 70.0, 30.0)
    composed = apply_transform(m, apply_transform(forward, ARCFACE_TEMPLATE))
    np.testing.assert_allclose(composed, ARCFACE_TEMPLATE, atol=1e-3)


@settings(max_examples=60, deadline=None)
@given(
    scale=st.floats(0.3, 6.0),
    angle=st.floats(-180.0, 180.0),
    tx=st.floats(-500, 500),
    ty=st.floats(-500, 500),
)
def test_any_similarity_of_the_template_is_undone_exactly(
    scale: float, angle: float, tx: float, ty: float
) -> None:
    src = landmarks_from_template(scale, angle, tx, ty)
    m = estimate_similarity(src)
    np.testing.assert_allclose(apply_transform(m, src), ARCFACE_TEMPLATE, atol=5e-3)
    assert np.linalg.det(m[:, :2]) > 0  # never a reflection


@pytest.mark.parametrize("angle", [90.0, 180.0, 270.0, -90.0])
def test_head_tilt_is_removed_by_the_transform(angle: float) -> None:
    # Edge-case matrix: head tilt of 90/180 degrees. The crop comes out upright.
    src = landmarks_from_template(1.0, angle, 300.0, 300.0)
    aligned = align_face(BLANK, src)
    np.testing.assert_allclose(aligned.landmarks, ARCFACE_TEMPLATE, atol=1e-2)
    assert aligned.fit_error < 1e-3


def test_float_input_dtypes_and_nested_lists_are_accepted() -> None:
    src = landmarks_from_template(1.0, 5.0, 200.0, 200.0)
    a = align_face(BLANK, src.astype(np.float64))
    b = align_face(BLANK, src.tolist())
    np.testing.assert_array_equal(a.image, b.image)


# --- guards --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "points",
    [
        np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]),  # collinear
        np.array([[0, 100], [10, 100], [20, 100], [30, 100], [40, 100]]),  # horizontal line
        np.full((5, 2), 100.0),  # all points coincide
    ],
)
def test_degenerate_landmarks_are_rejected(points: np.ndarray) -> None:
    with pytest.raises(InvalidLandmarksError) as info:
        align_face(BLANK, points)
    assert info.value.reason == DEGENERATE


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_landmarks_are_rejected(bad: float) -> None:
    src = landmarks_from_template(1.0, 0.0, 200.0, 200.0)
    src[3, 1] = bad
    with pytest.raises(InvalidLandmarksError) as info:
        align_face(BLANK, src)
    assert info.value.reason == NOT_FINITE
    with pytest.raises(InvalidLandmarksError):
        estimate_similarity(src)


@pytest.mark.parametrize("shape", [(4, 2), (5, 3), (5,), (6, 2)])
def test_wrong_landmark_shape_is_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(InvalidLandmarksError) as info:
        align_face(BLANK, np.zeros(shape))
    assert info.value.reason == BAD_SHAPE


def test_landmarks_outside_the_image_are_rejected_by_default() -> None:
    src = landmarks_from_template(1.0, 0.0, 200.0, 200.0)
    src[0] = [-5.0, 250.0]
    with pytest.raises(InvalidLandmarksError) as info:
        align_face(BLANK, src)
    assert info.value.reason == OUT_OF_IMAGE
    src[1] = [700.0, 250.0]  # beyond the right edge as well
    with pytest.raises(InvalidLandmarksError):
        align_face(BLANK, src)


def test_landmarks_outside_can_be_allowed_and_are_zero_padded() -> None:
    src = landmarks_from_template(1.0, 0.0, 0.0, 0.0)  # eyes sit at the very corner
    src[:, 0] -= 30  # shifts everything partly off the left edge
    aligned = align_face(BLANK, src, allow_outside=True)
    assert aligned.image.shape == (ALIGNED_SIZE, ALIGNED_SIZE, 3)
    assert (aligned.image == 0).any()  # the missing area is padded with zeros


def test_mirrored_left_right_order_is_rejected_not_silently_embedded() -> None:
    src = landmarks_from_template(1.0, 0.0, 200.0, 200.0)
    swapped = src[[1, 0, 2, 4, 3]]  # left/right labels exchanged
    with pytest.raises(InvalidLandmarksError) as info:
        align_face(BLANK, swapped)
    assert info.value.reason == BAD_FIT


def test_threshold_separates_real_jitter_from_mirrored_order() -> None:
    """Measurement behind DEFAULT_MAX_FIT_ERROR: noisy-but-valid vs mirrored layouts."""
    rng = np.random.default_rng(0)
    base = landmarks_from_template(2.0, 10.0, 250.0, 250.0)
    iod = float(np.linalg.norm(base[1] - base[0]))

    def fit(points: np.ndarray) -> float:
        m = estimate_similarity(points)
        diff = apply_transform(m, points) - ARCFACE_TEMPLATE
        return float(np.sqrt((diff**2).sum(axis=1).mean()) / 35.2)

    noisy = [fit(base + rng.normal(0, 0.05 * iod, base.shape)) for _ in range(200)]
    assert np.percentile(noisy, 95) < 0.3  # 5% of IOD jitter stays well below the threshold
    assert fit(base[[1, 0, 2, 4, 3]]) > 0.5  # mirrored order is far above it


def test_invalid_image_inputs_are_rejected() -> None:
    src = landmarks_from_template(1.0, 0.0, 200.0, 200.0)
    with pytest.raises(ValueError, match="uint8 RGB"):
        align_face(np.zeros((600, 600), dtype=np.uint8), src)
    with pytest.raises(ValueError, match="uint8 RGB"):
        align_face(np.zeros((600, 600, 3), dtype=np.float32), src)
    with pytest.raises(ValueError, match="112"):
        align_face(BLANK, src, size=128)


# --- determinism and the stored reference --------------------------------------------------------


def test_alignment_is_deterministic_and_does_not_mutate_inputs() -> None:
    scene, landmarks = synthetic_scene()
    scene_before, landmarks_before = scene.copy(), landmarks.copy()
    first = align_face(scene, landmarks)
    second = align_face(scene, landmarks)
    np.testing.assert_array_equal(first.image, second.image)  # bitwise
    np.testing.assert_array_equal(scene, scene_before)
    np.testing.assert_array_equal(landmarks, landmarks_before)


def test_synthetic_scene_markers_land_on_the_template_points() -> None:
    """Independent of the stored image: each colored disc must end up on its template point."""
    scene, landmarks = synthetic_scene()
    aligned = align_face(scene, landmarks)
    for point, color in zip(ARCFACE_TEMPLATE, MARKER_COLORS, strict=True):
        x, y = round(point[0]), round(point[1])
        pixel = aligned.image[y, x].astype(int)
        assert np.abs(pixel - np.array(color)).max() <= 40, (point, color, pixel)


def test_matches_the_stored_reference_within_tolerance() -> None:
    scene, landmarks = synthetic_scene()
    produced = align_face(scene, landmarks).image.astype(int)
    reference = np.asarray(Image.open(FIXTURES / "align_reference.png").convert("RGB")).astype(int)
    assert produced.shape == reference.shape == (112, 112, 3)
    diff = np.abs(produced - reference)
    assert diff.mean() < 0.5
    assert diff.max() <= 12  # tolerance for interpolation differences across OpenCV builds


def test_reference_file_exists_and_is_small() -> None:
    path: Path = FIXTURES / "align_reference.png"
    assert path.is_file()
    assert path.stat().st_size < 50_000
