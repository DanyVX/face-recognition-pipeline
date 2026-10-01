"""Metrics are checked against brute-force reference implementations (including ties)."""

from __future__ import annotations

import numpy as np
import pytest
from bench.metrics import (
    accuracy_at,
    auc,
    best_threshold,
    bootstrap_ci,
    det_curve,
    eer,
    far_resolution,
    fnir_at_fpir,
    mean_std,
    rank1_rate,
    roc_curve,
    threshold_at_far,
    wilson_interval,
)
from hypothesis import given, settings
from hypothesis import strategies as st

# Scores drawn from a coarse grid so that ties are common.
grid_scores = st.lists(st.integers(0, 12).map(lambda i: i / 12), min_size=2, max_size=40)


def labelled(draw_scores: list[float], seed: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.array(draw_scores, dtype=np.float64)
    labels = np.random.default_rng(seed).random(len(scores)) < 0.5
    if labels.all() or not labels.any():
        labels[0], labels[-1] = True, False
    return scores, labels


def brute_far_tar(scores: np.ndarray, labels: np.ndarray, t: float) -> tuple[float, float]:
    far = float((scores[~labels] >= t).mean())
    tar = float((scores[labels] >= t).mean())
    return far, tar


# --- ROC / AUC / EER -----------------------------------------------------------------------------


def test_roc_hand_computed_example() -> None:
    scores = np.array([0.9, 0.8, 0.4, 0.7, 0.3, 0.2])
    labels = np.array([True, True, True, False, False, False])
    curve = roc_curve(scores, labels)
    at = {
        round(t, 2): (f, r) for t, f, r in zip(curve.thresholds, curve.far, curve.tar, strict=True)
    }
    assert at[0.8] == (0.0, pytest.approx(2 / 3))
    assert at[0.7] == (pytest.approx(1 / 3), pytest.approx(2 / 3))
    assert at[0.2] == (1.0, 1.0)
    assert curve.far[0] == 0.0 and curve.tar[0] == 0.0  # first threshold accepts nothing


@settings(max_examples=60, deadline=None)
@given(grid_scores, st.integers(0, 1000))
def test_roc_matches_brute_force(raw: list[float], seed: int) -> None:
    scores, labels = labelled(raw, seed)
    curve = roc_curve(scores, labels)
    for t, far, tar in zip(curve.thresholds, curve.far, curve.tar, strict=True):
        bf_far, bf_tar = brute_far_tar(scores, labels, t)
        assert far == pytest.approx(bf_far) and tar == pytest.approx(bf_tar)
    assert np.all(np.diff(curve.far) >= -1e-12) and np.all(np.diff(curve.tar) >= -1e-12)


@settings(max_examples=60, deadline=None)
@given(grid_scores, st.integers(0, 1000))
def test_auc_equals_the_pairwise_ranking_probability(raw: list[float], seed: int) -> None:
    scores, labels = labelled(raw, seed)
    g, i = scores[labels], scores[~labels]
    wins = (g[:, None] > i[None, :]).mean() + 0.5 * (g[:, None] == i[None, :]).mean()
    assert auc(roc_curve(scores, labels)) == pytest.approx(wins, abs=1e-9)


def test_eer_for_perfect_and_useless_scorers() -> None:
    labels = np.array([True] * 50 + [False] * 50)
    perfect = np.concatenate([np.linspace(0.6, 1.0, 50), np.linspace(0.0, 0.4, 50)])
    assert eer(perfect, labels)[0] == pytest.approx(0.0, abs=1e-9)
    rng = np.random.default_rng(0)
    useless = rng.random(4000)
    useless_labels = np.arange(4000) % 2 == 0
    assert eer(useless, useless_labels)[0] == pytest.approx(0.5, abs=0.05)


@settings(max_examples=40, deadline=None)
@given(grid_scores, st.integers(0, 1000))
def test_eer_is_between_the_far_and_frr_of_neighbouring_thresholds(
    raw: list[float], seed: int
) -> None:
    scores, labels = labelled(raw, seed)
    rate, _ = eer(scores, labels)
    curve = roc_curve(scores, labels)
    frr = 1.0 - curve.tar
    brute = min(max(f, r) for f, r in zip(curve.far, frr, strict=True))
    # The interpolated crossing lies at or below the best discrete max(FAR, FRR).
    assert rate <= brute + 1e-9
    assert 0.0 <= rate <= 1.0


def test_known_symmetric_eer() -> None:
    # Genuine uniform on [0.2, 1.0], impostor uniform on [0.0, 0.8]. FAR(t) = (0.8 - t) / 0.8 and
    # FRR(t) = (t - 0.2) / 0.8 cross at t = 0.5, where both equal 0.3 / 0.8 = 0.375.
    genuine = np.linspace(0.2, 1.0, 100)
    impostor = np.linspace(0.0, 0.8, 100)
    scores = np.concatenate([genuine, impostor])
    labels = np.array([True] * 100 + [False] * 100)
    rate, threshold = eer(scores, labels)
    assert rate == pytest.approx(0.375, abs=0.02)
    assert threshold == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize(
    "scores, labels",
    [
        (np.array([0.1, np.nan]), np.array([True, False])),
        (np.array([0.1, 0.2, 0.3]), np.array([True, False])),
        (np.array([0.1, 0.2]), np.array([True, True])),
        (np.array([[0.1], [0.2]]), np.array([True, False])),
    ],
)
def test_invalid_inputs_are_rejected(scores: np.ndarray, labels: np.ndarray) -> None:
    with pytest.raises(ValueError):
        roc_curve(scores, labels)


# --- thresholds ----------------------------------------------------------------------------------


@settings(max_examples=80, deadline=None)
@given(grid_scores, st.sampled_from([0.5, 0.25, 0.1, 0.01, 0.001]))
def test_threshold_at_far_keeps_far_within_the_target(raw: list[float], far: float) -> None:
    imp = np.array(raw)
    t = threshold_at_far(imp, far)
    achieved = float((imp >= t).mean())
    assert achieved <= far + 1e-12
    # Maximal: lowering to the next lower distinct score would exceed the target (unless none).
    lower = imp[imp < t]
    if lower.size:
        assert float((imp >= lower.max()).mean()) > far - 1e-12 or (
            achieved == 0.0 and far < 1 / imp.size
        )


def test_threshold_at_far_extremes_and_resolution_limit() -> None:
    imp = np.arange(1000) / 1000.0
    assert float((imp >= threshold_at_far(imp, 0.01)).mean()) <= 0.01
    assert (imp >= threshold_at_far(imp, 1.0)).all()
    t_tiny = threshold_at_far(imp, 1e-6)  # finer than 1/n: nothing passes
    assert (imp >= t_tiny).sum() == 0
    assert far_resolution(3000) == pytest.approx(1 / 3000)
    with pytest.raises(ValueError):
        threshold_at_far(imp, 0.0)
    with pytest.raises(ValueError):
        threshold_at_far(np.array([]), 0.1)


def test_accuracy_boundary_is_inclusive() -> None:
    scores = np.array([0.5, 0.5, 0.4])
    labels = np.array([True, False, False])
    assert accuracy_at(scores, labels, 0.5) == pytest.approx(2 / 3)  # both 0.5 accepted
    assert accuracy_at(scores, labels, np.nextafter(0.5, 1.0)) == pytest.approx(2 / 3)
    assert accuracy_at(scores, labels, 0.4) == pytest.approx(1 / 3)


@settings(max_examples=80, deadline=None)
@given(grid_scores, st.integers(0, 1000))
def test_best_threshold_is_optimal_by_brute_force(raw: list[float], seed: int) -> None:
    scores, labels = labelled(raw, seed)
    t, acc = best_threshold(scores, labels)
    assert accuracy_at(scores, labels, t) == pytest.approx(acc)
    candidates = np.unique(scores)
    brute = max(
        accuracy_at(scores, labels, c) for c in np.concatenate([candidates, [candidates.max() + 1]])
    )
    assert acc == pytest.approx(brute)
    assert t not in scores  # never sits exactly on a score


def test_best_threshold_separable_and_degenerate_cases() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([False, False, True, True])
    t, acc = best_threshold(scores, labels)
    assert acc == 1.0 and 0.2 < t < 0.8
    t_all, acc_all = best_threshold(np.array([0.3, 0.4]), np.array([True, True]))
    assert acc_all == 1.0 and t_all < 0.3
    with pytest.raises(ValueError):
        best_threshold(np.array([]), np.array([], dtype=bool))


# --- open-set ------------------------------------------------------------------------------------


def brute_det(ms: np.ndarray, mc: np.ndarray, ns: np.ndarray, t: float) -> tuple[float, float]:
    fpir = float((ns >= t).mean())
    fnir = 1.0 - float(((ms >= t) & mc).mean())
    return fpir, fnir


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.integers(0, 10).map(lambda i: i / 10), min_size=2, max_size=25),
    st.lists(st.integers(0, 10).map(lambda i: i / 10), min_size=2, max_size=25),
    st.integers(0, 1000),
)
def test_det_curve_matches_brute_force(
    mated: list[float], nonmated: list[float], seed: int
) -> None:
    ms, ns = np.array(mated), np.array(nonmated)
    mc = np.random.default_rng(seed).random(ms.size) < 0.7
    curve = det_curve(ms, mc, ns)
    for t, fpir, fnir in zip(curve.thresholds, curve.fpir, curve.fnir, strict=True):
        bf_fpir, bf_fnir = brute_det(ms, mc, ns, t)
        assert fpir == pytest.approx(bf_fpir) and fnir == pytest.approx(bf_fnir)


def test_fnir_at_fpir_picks_the_loosest_allowed_threshold() -> None:
    mated = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, True, False])  # the 0.6 probe is a wrong rank-1
    nonmated = np.array([0.75, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02])
    curve = det_curve(mated, correct, nonmated)
    fnir, fpir, threshold = fnir_at_fpir(curve, 0.1)  # one non-mated probe (0.75) may pass
    # The loosest threshold keeping FPIR <= 0.1 is 0.6: the next observed score down, 0.5, would
    # let a second non-mated probe through. At 0.6 the three correct probes (0.9/0.8/0.7) are
    # found; the 0.6 probe is a wrong rank-1 and stays a miss.
    assert fpir == pytest.approx(0.1) and threshold == pytest.approx(0.6)
    assert fnir == pytest.approx(0.25)
    fnir0, fpir0, _ = fnir_at_fpir(curve, 0.0)
    assert fpir0 == 0.0 and fnir0 == pytest.approx(1 - 2 / 4)


def test_rank1_rate_and_open_set_validation() -> None:
    assert rank1_rate(np.array([True, True, False, True])) == 0.75
    with pytest.raises(ValueError):
        rank1_rate(np.array([], dtype=bool))
    with pytest.raises(ValueError):
        det_curve(np.array([0.5]), np.array([True, False]), np.array([0.1]))
    with pytest.raises(ValueError):
        det_curve(np.array([]), np.array([], dtype=bool), np.array([0.1]))


def test_closed_set_rank1_can_look_perfect_while_open_set_fails() -> None:
    """Why rank-1 alone is misleading: every enrolled probe is found, yet strangers match too."""
    mated = np.array([0.55, 0.60, 0.58, 0.62])
    correct = np.ones(4, dtype=bool)
    strangers = np.array([0.56, 0.57, 0.59, 0.61, 0.54, 0.55, 0.60, 0.58, 0.57, 0.56])
    assert rank1_rate(correct) == 1.0
    curve = det_curve(mated, correct, strangers)
    fnir, fpir, _ = fnir_at_fpir(curve, 0.1)
    assert fnir > 0.5  # at 10% false positives, most enrolled people are no longer recognised
    assert fpir <= 0.1


# --- uncertainty ---------------------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_estimate_and_is_reproducible() -> None:
    data = np.random.default_rng(1).normal(0.9, 0.05, 300)
    lo, hi = bootstrap_ci(lambda idx: float(data[idx].mean()), data.size, n_boot=500, seed=3)
    assert lo < data.mean() < hi and hi - lo < 0.03
    assert (lo, hi) == bootstrap_ci(
        lambda idx: float(data[idx].mean()), data.size, n_boot=500, seed=3
    )
    with pytest.raises(ValueError):
        bootstrap_ci(lambda idx: 0.0, 1)


def test_mean_std_sem() -> None:
    mean, std, sem = mean_std([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5 and std == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert sem == pytest.approx(std / 2)
    assert mean_std([5.0]) == (5.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        mean_std([])


def test_wilson_interval_known_values_and_edges() -> None:
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = wilson_interval(100, 100)  # all successes: upper bound is 1, lower is well below
    assert hi == pytest.approx(1.0) and 0.95 < lo < 0.99
    lo0, hi0 = wilson_interval(0, 20)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.2
    narrow = wilson_interval(900, 1000)
    wide = wilson_interval(9, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])
    for bad in ((1, 0), (-1, 5), (6, 5)):
        with pytest.raises(ValueError):
            wilson_interval(*bad)
