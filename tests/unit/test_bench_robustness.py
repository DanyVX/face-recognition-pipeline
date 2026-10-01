from __future__ import annotations

import numpy as np
import pytest
from bench.degrade import Degradation, brightness_shift
from bench.robustness import (
    FEATURE_NAMES,
    STATUS_NO_FACE,
    STATUS_OK,
    choose_samples,
    run_setting,
    summarize_setting,
)

from frp.align import align_face
from frp.detect import Face
from frp.embed import ArcFaceEmbedder
from frp.errors import NoFaceError
from frp.imageio import LoadedImage
from frp.pipeline import PreparedFace
from tests.synthetic import synthetic_scene
from tests.unit.fake_arcface import FAKE_INFO, FakeArcFaceSession

N_PAIRS = 600


def pair_arrays() -> tuple[np.ndarray, ...]:
    same = np.tile([True, False], N_PAIRS // 2)
    fold = np.repeat(np.arange(10), N_PAIRS // 10)
    usable = np.ones(N_PAIRS, dtype=bool)
    usable[::7] = False  # some pairs are unusable and must never be sampled
    rows = np.arange(N_PAIRS, dtype=np.int64)
    return same, fold, usable, rows, rows + 1000


def test_samples_respect_splits_classes_and_usability() -> None:
    same, fold, usable, ra, rb = pair_arrays()
    samples = choose_samples(same, fold, usable, ra, rb, per_class=20, seed=0)
    assert len(samples) == 80
    for split, folds in (("val", {0, 1}), ("test", set(range(2, 10)))):
        subset = [s for s in samples if s.split == split]
        assert len(subset) == 40 and sum(s.same for s in subset) == 20
        assert all(int(fold[s.pair_index]) in folds for s in subset)
        assert all(usable[s.pair_index] for s in subset)
    assert len({s.pair_index for s in samples}) == len(samples)  # no pair twice
    assert {s.row_b for s in samples} <= set(rb.tolist())


def test_sampling_is_seeded() -> None:
    same, fold, usable, ra, rb = pair_arrays()
    a = choose_samples(same, fold, usable, ra, rb, 20, seed=1)
    b = choose_samples(same, fold, usable, ra, rb, 20, seed=1)
    c = choose_samples(same, fold, usable, ra, rb, 20, seed=2)
    assert a == b and a != c


def test_sampling_takes_what_exists_when_a_class_is_short() -> None:
    same, fold, usable, ra, rb = pair_arrays()
    samples = choose_samples(same, fold, usable, ra, rb, per_class=10_000, seed=0)
    assert 0 < len(samples) < 4 * 10_000


def test_summary_counts_failures_as_errors_and_reports_an_interval() -> None:
    scores = np.array([0.9, 0.8, 0.1, 0.2, np.nan, np.nan])
    same = np.array([True, True, False, False, True, False])
    status = np.array([0, 0, 0, 0, STATUS_NO_FACE, STATUS_NO_FACE], dtype=np.uint8)
    s = summarize_setting(scores, same, status, threshold=0.5)
    assert s["accuracy"] == pytest.approx(4 / 6)  # both failed pairs are wrong, whatever the label
    assert s["n_failed"] == 2 and s["failure_rate"] == pytest.approx(2 / 6) and s["no_face"] == 2
    lo, hi = s["accuracy_ci95"]
    assert lo < s["accuracy"] < hi
    assert s["mean_score_genuine"] == pytest.approx(0.85)


def test_summary_threshold_boundary_is_inclusive() -> None:
    s = summarize_setting(np.array([0.5, 0.5]), np.array([True, False]), np.zeros(2, np.uint8), 0.5)
    assert s["accuracy"] == pytest.approx(0.5)  # both accepted: genuine right, impostor wrong


class FakePipeline:
    """Fails when the image is too dark; otherwise returns the synthetic scene's aligned face."""

    def prepare(self, loaded: LoadedImage) -> PreparedFace:
        if loaded.array.mean() < 20:
            raise NoFaceError("dark")
        scene, landmarks = synthetic_scene()
        face = Face(
            box=np.array([40, 20, 200, 180], dtype=np.float32), landmarks=landmarks, score=0.9
        )
        return PreparedFace(
            face=face, aligned=align_face(scene, landmarks), n_faces=1, image_shape=(200, 240)
        )


def test_run_setting_scores_degraded_probes_and_records_failures_and_features() -> None:
    probes = [np.full((64, 64, 3), 100, dtype=np.uint8) for _ in range(4)]
    probes[2] = np.full((64, 64, 3), 5, dtype=np.uint8)  # already dark: the fake detector fails
    embedder = ArcFaceEmbedder(FakeArcFaceSession(), FAKE_INFO)
    clean_a = embedder.embed(np.stack([align_face(*synthetic_scene()).image] * 4)).vectors
    dim = Degradation("brightness", "-10", -10.0, lambda im, _r: brightness_shift(im, -10))
    out = run_setting(dim, probes, clean_a, FakePipeline(), embedder, seed=0)  # type: ignore[arg-type]
    assert out["status"].tolist() == [STATUS_OK, STATUS_OK, STATUS_NO_FACE, STATUS_OK]
    assert np.isnan(out["scores"][2]) and np.isfinite(out["scores"][[0, 1, 3]]).all()
    assert out["scores"][0] == pytest.approx(1.0, abs=1e-5)  # same crop on both sides
    assert out["features"].shape == (4, len(FEATURE_NAMES)) and np.isnan(out["features"][2]).all()
    assert out["features"][0, FEATURE_NAMES.index("det_score")] == pytest.approx(0.9)


def test_run_setting_is_deterministic_for_random_degradations() -> None:
    from bench.degrade import occlude

    probes = [np.full((64, 64, 3), 100, dtype=np.uint8)] * 3
    embedder = ArcFaceEmbedder(FakeArcFaceSession(), FAKE_INFO)
    clean_a = embedder.embed(np.stack([align_face(*synthetic_scene()).image] * 3)).vectors
    occ = Degradation("occlusion", "30%", 0.3, lambda im, r: occlude(im, 0.3, r))
    a = run_setting(occ, probes, clean_a, FakePipeline(), embedder, seed=5)  # type: ignore[arg-type]
    b = run_setting(occ, probes, clean_a, FakePipeline(), embedder, seed=5)  # type: ignore[arg-type]
    np.testing.assert_array_equal(a["scores"], b["scores"])
