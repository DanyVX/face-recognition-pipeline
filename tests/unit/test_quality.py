"""Quality gates: measurements, thresholds, reason codes, and rebuilding aligned faces."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from frp.align import ARCFACE_TEMPLATE, align_face, aligned_from_crop
from frp.detect import Face
from frp.quality import (
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
    GateContext,
    QualityThresholds,
    SharpnessGate,
    SizeGate,
    assess,
    build_gates,
    extract_features,
)
from frp.quality.features import luminance, sharpness
from tests.synthetic import synthetic_scene

RNG = np.random.default_rng(0)


def textured(seed: int = 0) -> np.ndarray:
    """A sharp, face-sized RGB texture (random blocks give strong edges)."""
    blocks = np.random.default_rng(seed).integers(0, 255, (14, 14, 3), dtype=np.uint8)
    return cv2.resize(blocks, (112, 112), interpolation=cv2.INTER_NEAREST)


def blurred(image: np.ndarray, sigma: float) -> np.ndarray:
    return np.asarray(cv2.GaussianBlur(image, (0, 0), sigma), dtype=np.uint8)


def solid(value: int) -> np.ndarray:
    return np.full((112, 112, 3), value, dtype=np.uint8)


def ctx(
    crop: np.ndarray,
    *,
    face_px: float = 80.0,
    score: float = 0.95,
    truncated: bool = False,
) -> GateContext:
    box = np.array([10, 10, 10 + face_px, 10 + face_px], dtype=np.float32)
    face = Face(
        box=box,
        landmarks=ARCFACE_TEMPLATE.copy(),
        score=score,
        touches_border=truncated,
        visible_fraction=1.0,
    )
    return GateContext(
        face=face, aligned=aligned_from_crop(crop, ARCFACE_TEMPLATE), image_shape=(300, 300)
    )


# --- measurements --------------------------------------------------------------------------------


def test_sharpness_orders_sharp_above_blurred_above_flat() -> None:
    sharp = textured()
    soft = blurred(sharp, 2.0)
    softer = blurred(sharp, 5.0)
    values = [sharpness(sharp), sharpness(soft), sharpness(softer), sharpness(solid(120))]
    assert values == sorted(values, reverse=True)
    assert values[-1] == 0.0 and values[0] > 100


def test_luminance_weights_and_exact_values() -> None:
    gray = solid(100)
    assert float(luminance(gray).mean()) == pytest.approx(100.0, abs=0.01)
    pure_green = np.zeros((112, 112, 3), dtype=np.uint8)
    pure_green[..., 1] = 255
    assert float(luminance(pure_green).mean()) == pytest.approx(0.587 * 255, abs=0.01)


def test_extract_features_reports_every_quantity() -> None:
    f = extract_features(ctx(textured(), face_px=64.0, score=0.9))
    assert f.face_px == 64.0 and f.det_score == pytest.approx(0.9)
    assert f.laplacian_var > 100 and 50 < f.brightness < 200 and f.contrast > 30
    assert abs(f.yaw_ratio) < 0.05 and abs(f.pitch_ratio) < 0.05 and f.fit_error < 1e-6
    assert f.truncated is False
    assert set(f.as_dict()) == {
        "face_px",
        "det_score",
        "laplacian_var",
        "brightness",
        "contrast",
        "yaw_ratio",
        "pitch_ratio",
        "fit_error",
        "truncated",
    }


# --- individual gates ----------------------------------------------------------------------------


def test_size_gate_boundary_is_inclusive() -> None:
    gate = SizeGate(min_face_px=64.0)
    assert gate.check(ctx(textured(), face_px=64.0)).passed
    result = gate.check(ctx(textured(), face_px=63.9))
    assert not result.passed and result.reason_code == FACE_TOO_SMALL
    assert result.value == pytest.approx(63.9, abs=1e-3) and result.threshold == 64.0


def test_confidence_gate() -> None:
    gate = ConfidenceGate(0.6)
    assert gate.check(ctx(textured(), score=0.6)).passed
    bad = gate.check(ctx(textured(), score=0.59))
    assert not bad.passed and bad.reason_code == LOW_CONFIDENCE


def test_sharpness_gate_flags_blur() -> None:
    sharp, soft = textured(), blurred(textured(), 4.0)
    threshold = (sharpness(sharp) + sharpness(soft)) / 2
    gate = SharpnessGate(threshold)
    assert gate.check(ctx(sharp)).passed
    result = gate.check(ctx(soft))
    assert not result.passed and result.reason_code == BLURRY


def test_exposure_gate_reasons() -> None:
    gate = ExposureGate(min_brightness=40, max_brightness=220, min_contrast=15)
    assert gate.check(ctx(textured())).passed
    dark = gate.check(ctx((textured() * 0.1).astype(np.uint8)))
    assert dark.reason_code == TOO_DARK
    bright = gate.check(ctx(np.clip(textured().astype(int) + 200, 0, 255).astype(np.uint8)))
    assert bright.reason_code == TOO_BRIGHT
    flat = gate.check(ctx(solid(120)))
    assert not flat.passed and flat.reason_code == LOW_CONTRAST


def test_exposure_gate_with_nothing_set_always_passes() -> None:
    assert ExposureGate().check(ctx(solid(0))).passed


def test_framing_gate_rejects_cut_off_faces() -> None:
    assert FramingGate().check(ctx(textured())).passed
    result = FramingGate().check(ctx(textured(), truncated=True))
    assert not result.passed and result.reason_code == FACE_CUT_OFF


# --- thresholds and assessment ------------------------------------------------------------


def test_default_thresholds_build_the_expected_gates() -> None:
    names = [g.name for g in build_gates(QualityThresholds())]
    assert names == [
        "size",
        "confidence",
        "framing",
        "exposure",
        "pose",
    ]  # sharpness needs calibration


def test_none_disables_a_check_and_all_none_means_no_gates() -> None:
    empty = QualityThresholds(
        min_face_px=None,
        min_det_score=None,
        min_laplacian_var=None,
        min_brightness=None,
        max_brightness=None,
        min_contrast=None,
        max_abs_yaw_ratio=None,
        max_abs_pitch_ratio=None,
        reject_cut_off_faces=False,
    )
    assert build_gates(empty) == []
    report = assess(ctx(solid(0), face_px=1.0, score=0.01), empty)
    assert report.passed and report.results == ()


def test_assess_reports_every_failing_reason_at_once() -> None:
    thresholds = QualityThresholds(min_face_px=100.0, min_det_score=0.99, min_laplacian_var=1e6)
    report = assess(ctx(solid(120), face_px=50.0, score=0.8, truncated=True), thresholds)
    assert not report.passed
    assert set(report.reasons) >= {
        FACE_TOO_SMALL,
        LOW_CONFIDENCE,
        FACE_CUT_OFF,
        BLURRY,
        LOW_CONTRAST,
    }


def test_a_good_face_passes_default_gates() -> None:
    report = assess(ctx(textured()))
    assert report.passed and report.reasons == ()


def test_pose_gate_with_only_one_axis_enabled() -> None:
    only_yaw = QualityThresholds(max_abs_yaw_ratio=0.1, max_abs_pitch_ratio=None)
    (pose,) = [g for g in build_gates(only_yaw) if g.name == "pose"]
    assert pose.check(ctx(textured())).passed


def test_thresholds_round_trip_through_json(tmp_path: Path) -> None:
    original = QualityThresholds(min_laplacian_var=42.5, min_face_px=None)
    wrapped = tmp_path / "calibrated.json"
    wrapped.write_text(
        json.dumps({"thresholds": original.to_dict(), "note": "x"}), encoding="utf-8"
    )
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(original.to_dict()), encoding="utf-8")
    assert QualityThresholds.load(wrapped) == original == QualityThresholds.load(bare)


def test_unknown_threshold_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown quality threshold"):
        QualityThresholds.from_dict({"min_blur": 3})


# --- rebuilding aligned faces from cached crops --------------------------------------------------


def test_aligned_from_crop_matches_align_face() -> None:
    scene, landmarks = synthetic_scene()
    direct = align_face(scene, landmarks)
    rebuilt = aligned_from_crop(direct.image, landmarks)
    np.testing.assert_array_equal(rebuilt.image, direct.image)
    np.testing.assert_allclose(rebuilt.matrix, direct.matrix)
    np.testing.assert_allclose(rebuilt.landmarks, direct.landmarks, atol=1e-5)
    assert rebuilt.fit_error == pytest.approx(direct.fit_error)


def test_aligned_from_crop_validates_its_input() -> None:
    with pytest.raises(ValueError, match="crop must be"):
        aligned_from_crop(np.zeros((100, 100, 3), np.uint8), ARCFACE_TEMPLATE)
    with pytest.raises(ValueError, match="crop must be"):
        aligned_from_crop(np.zeros((112, 112, 3), np.float32), ARCFACE_TEMPLATE)
