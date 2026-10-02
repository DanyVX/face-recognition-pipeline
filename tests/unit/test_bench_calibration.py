"""Quality calibration logic on synthetic samples with a known relationship to accuracy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from bench.calibrate_quality import (
    FEATURE_ORDER,
    RULES,
    Rule,
    SampleSet,
    calibrate,
    candidate_thresholds,
    curves,
    evaluate,
    gain_at,
    gated_rejection,
    pair_value,
    rejected,
    select_threshold,
    thresholds_from,
)

from frp.quality import QualityThresholds

N = 4000


def feature_dict(rng: np.random.Generator, sharp: np.ndarray) -> dict[str, np.ndarray]:
    base = {
        "face_px": rng.uniform(60, 120, N),
        "det_score": rng.uniform(0.8, 1.0, N),
        "laplacian_var": sharp,
        "brightness": rng.uniform(80, 180, N),
        "contrast": rng.uniform(30, 70, N),
        "yaw_ratio": rng.normal(0, 0.05, N),
        "pitch_ratio": rng.normal(0, 0.05, N),
        "fit_error": rng.uniform(0.05, 0.2, N),
        "truncated": np.zeros(N),
    }
    assert set(base) == set(FEATURE_ORDER)
    return base


def make_samples(seed: int = 0, informative: bool = True) -> SampleSet:
    """Accuracy collapses for blurry pairs (low laplacian_var) when ``informative``."""
    rng = np.random.default_rng(seed)
    sharp_a = rng.uniform(50, 500, N)
    sharp_b = rng.uniform(50, 500, N)
    same = rng.random(N) < 0.5
    worst = np.minimum(sharp_a, sharp_b)
    flip_prob = np.where(worst < 120, 0.45, 0.02) if informative else np.full(N, 0.1)
    wrong = rng.random(N) < flip_prob
    ideal = np.where(same, 0.7, 0.1)
    score = np.where(wrong, np.where(same, 0.1, 0.7), ideal) + rng.normal(0, 0.02, N)
    return SampleSet(
        score=score,
        same=same,
        source=np.where(worst < 120, "blur", "clean"),
        a=feature_dict(rng, sharp_a),
        b=feature_dict(rng, sharp_b),
    )


# --- building blocks -----------------------------------------------------------------------------


def test_pair_value_combinations() -> None:
    a, b = np.array([1.0, -3.0]), np.array([2.0, 1.0])
    assert pair_value(Rule("x", "f", "min", "low"), a, b).tolist() == [1.0, -3.0]
    assert pair_value(Rule("x", "f", "max", "high"), a, b).tolist() == [2.0, 1.0]
    assert pair_value(Rule("x", "f", "maxabs", "high"), a, b).tolist() == [2.0, 3.0]


def test_rejection_direction_is_strict_inequality() -> None:
    v = np.array([1.0, 2.0, 3.0])
    assert rejected(v, 2.0, "low").tolist() == [True, False, False]  # value == t is kept
    assert rejected(v, 2.0, "high").tolist() == [False, False, True]


def test_gain_at_matches_hand_computation() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    correct = np.array([False, False, True, True])
    reject, kept, gain = gain_at(values, correct, 2.5, "low")  # rejects the two wrong pairs
    assert (reject, kept, gain) == (0.5, 1.0, 0.5)
    assert gain_at(values, correct, 10.0, "low")[0] == 1.0  # everything rejected
    assert np.isnan(gain_at(values, correct, 10.0, "low")[2])


def test_candidate_thresholds_stay_within_the_budget() -> None:
    v = np.random.default_rng(0).normal(size=5000)
    for direction in ("low", "high"):
        for t in candidate_thresholds(v, direction, 0.1):
            assert float(rejected(v, float(t), direction).mean()) <= 0.1 + 1e-3


# --- selection -----------------------------------------------------------------------------------


def test_an_informative_feature_is_selected_and_found_near_the_true_boundary() -> None:
    s = make_samples(informative=True)
    rule = next(r for r in RULES if r.threshold_name == "min_laplacian_var")
    out = select_threshold(s.values(rule), s.correct(0.4), "low", budget=0.3, seed=1)
    assert out["significant"] and out["selected"] is not None
    assert 90 < out["selected"] < 130  # accuracy collapses below 120
    assert out["gain_ci95"][0] > 0 and out["best_candidate"]["reject_rate"] <= 0.3 + 1e-9
    assert out["out_of_fold_gain"] > 0 and len(out["fold_thresholds"]) == 5


def test_an_uninformative_feature_is_not_selected() -> None:
    s = make_samples(informative=False)
    for rule in RULES:
        if rule.feature == "laplacian_var":
            continue
        out = select_threshold(s.values(rule), s.correct(0.4), rule.direction, budget=0.1, seed=2)
        # Pure noise: the corrected CI must not certify a gain (fixed seed, so deterministic).
        assert out["selected"] is None, rule.threshold_name


def test_calibrate_enables_only_the_informative_gate() -> None:
    s = make_samples(seed=3, informative=True)
    selection = calibrate(s, tau=0.4, budget=0.3)
    thresholds = thresholds_from(selection)
    assert thresholds.min_laplacian_var is not None and 90 < thresholds.min_laplacian_var < 130
    assert thresholds.min_face_px is None and thresholds.max_abs_yaw_ratio is None
    assert thresholds.reject_cut_off_faces is True  # policy, never calibrated away


def test_evaluation_on_held_out_data_shows_the_gain_and_breaks_down_by_source() -> None:
    train, test = make_samples(seed=3), make_samples(seed=103)
    thresholds = thresholds_from(calibrate(train, 0.4, 0.3))
    ev = evaluate(test, 0.4, thresholds)
    assert ev["accuracy_kept"] > ev["accuracy_all"] and ev["gain"] > 0 and ev["gain_ci95"][0] > 0
    assert 0.15 < ev["reject_rate"] < 0.35  # about the 29% of pairs that are blurry
    assert set(ev["by_source"]) == {"blur", "clean"}
    assert ev["by_source"]["blur"]["reject_rate"] > ev["by_source"]["clean"]["reject_rate"]


def test_no_enabled_gates_means_no_rejection_and_no_gain_entry() -> None:
    s = make_samples()
    off = QualityThresholds(
        min_face_px=None,
        min_det_score=None,
        min_laplacian_var=None,
        min_brightness=None,
        max_brightness=None,
        min_contrast=None,
        max_abs_yaw_ratio=None,
        max_abs_pitch_ratio=None,
    )
    assert not gated_rejection(s, off).any()
    ev = evaluate(s, 0.4, off)
    assert ev["reject_rate"] == 0.0 and "gain" not in ev


def test_curves_cover_every_gate_with_monotone_reject_rates() -> None:
    c = curves(make_samples(), 0.4)
    assert set(c) == {r.threshold_name for r in RULES}
    rates = [row["reject_rate"] for row in c["min_laplacian_var"]]
    assert rates == sorted(rates)
    assert (
        c["min_laplacian_var"][-1]["accuracy_kept"]
        > c["min_laplacian_var"][0]["accuracy_kept"] - 0.2
    )


def test_sample_sets_concat_and_drop_invalid_rows() -> None:
    s = make_samples()
    broken = SampleSet(
        s.score.copy(),
        s.same,
        s.source,
        {k: v.copy() for k, v in s.a.items()},
        {k: v.copy() for k, v in s.b.items()},
    )
    broken.score[0] = np.nan
    broken.b["contrast"][1] = np.nan
    both = SampleSet.concat([s, broken])
    assert both.score.size == 2 * N and both.only_valid().score.size == 2 * N - 2


# --- the file the library loads ------------------------------------------------------------------


def test_thresholds_load_from_a_benchmark_result_file(tmp_path: Path) -> None:
    thresholds = QualityThresholds(min_laplacian_var=77.0, min_face_px=None)
    path = tmp_path / "quality_thresholds.json"
    path.write_text(
        json.dumps({"name": "x", "result": {"thresholds": thresholds.to_dict()}}), encoding="utf-8"
    )
    assert QualityThresholds.load(path) == thresholds


@pytest.mark.parametrize("name", [r.threshold_name for r in RULES])
def test_every_rule_names_a_real_threshold_and_feature(name: str) -> None:
    rule = next(r for r in RULES if r.threshold_name == name)
    assert hasattr(QualityThresholds(), rule.threshold_name) and rule.feature in FEATURE_ORDER


@pytest.mark.slow
def test_familywise_false_positive_rate_of_the_whole_procedure_is_controlled() -> None:
    """Over many simulated datasets the informative gate is always found and spurious gates are
    rare. Measured while developing: 60/60 found, spurious gate in 5/60 runs (nominal 5%)."""
    found = spurious_runs = 0
    runs = 40
    for seed in range(runs):
        selection = calibrate(make_samples(seed=1000 + seed), 0.4, 0.3)
        enabled = {k for k, v in selection.items() if v["selected"] is not None}
        found += "min_laplacian_var" in enabled
        spurious_runs += bool(enabled - {"min_laplacian_var"})
    assert found == runs
    assert spurious_runs / runs <= 0.20  # generous upper bound on a ~5% nominal rate
