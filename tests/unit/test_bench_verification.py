"""Verification protocol and extended-impostor machinery on synthetic, LFW-shaped data."""

from __future__ import annotations

import numpy as np
import pytest
from bench.impostors import (
    BINS,
    ScoreHistogram,
    cross_identity_histograms,
    pool,
)
from bench.lfw_verification import (
    FAILED_IMPOSTOR_SCORE,
    fold_identities,
    leakage_report,
    verification_report,
)

from frp.lfw import LfwPair

N_FOLDS = 10
PER_CLASS = 60


def synthetic_scores(
    seed: int = 0, genuine_mean: float = 0.65, impostor_mean: float = 0.05, spread: float = 0.12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores, same, fold = [], [], []
    for k in range(N_FOLDS):
        scores += list(rng.normal(genuine_mean, spread, PER_CLASS))
        same += [True] * PER_CLASS
        scores += list(rng.normal(impostor_mean, spread, PER_CLASS))
        same += [False] * PER_CLASS
        fold += [k] * (2 * PER_CLASS)
    return np.array(scores), np.array(same), np.array(fold)


# --- histograms ----------------------------------------------------------------------------------


def test_histogram_far_and_threshold_agree_with_exact_quantiles() -> None:
    rng = np.random.default_rng(1)
    scores = rng.normal(0.0, 0.2, 100_000).clip(-1, 1)
    hist = ScoreHistogram()
    hist.add(scores)
    assert hist.total == scores.size
    width = 2.0 / BINS
    for far in (0.1, 0.01, 1e-3, 1e-4):
        t = hist.threshold_at_far(far)
        assert float((scores >= t).mean()) <= far + 1e-12  # never exceeds the target
        assert (
            float((scores >= t - 3 * width).mean()) > far * 0.5 or far < 2e-4
        )  # and is nearly tight
        assert hist.far_at(t) == pytest.approx(float((scores >= t).mean()), abs=1e-9)


def test_histogram_edge_cases() -> None:
    hist = ScoreHistogram()
    with pytest.raises(ValueError, match="empty"):
        hist.threshold_at_far(0.1)
    with pytest.raises(ValueError, match="empty"):
        hist.far_at(0.0)
    hist.add(np.array([-1.0, 1.0, 0.0]))  # the extremes land in the first and last bins
    assert hist.total == 3 and hist.counts[0] == 1 and hist.counts[-1] == 1
    with pytest.raises(ValueError):
        hist.threshold_at_far(0.0)
    assert hist.far_at(1.0) == 0.0
    other = ScoreHistogram()
    other.add(np.array([0.5]))
    hist += other
    assert hist.total == 4


def blob_embeddings(
    n_people: int, per_person: int, dim: int = 24, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_people, dim))
    rows, ids = [], []
    for p in range(n_people):
        v = centers[p] + 0.4 * rng.normal(size=(per_person, dim))
        rows.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        ids += [p] * per_person
    return np.concatenate(rows).astype(np.float32), np.array(ids, dtype=np.int64)


def test_cross_identity_histograms_count_every_different_person_pair_once() -> None:
    emb, ids = blob_embeddings(12, 3)
    groups = np.repeat(np.arange(3), 12)  # 4 people (12 images) per group
    hists = cross_identity_histograms(emb, ids, groups, block=7)
    n = len(emb)
    different = sum(1 for i in range(n) for j in range(i + 1, n) if ids[i] != ids[j])
    assert sum(h.total for h in hists.values()) == different
    # Restricting to groups {0, 2} counts exactly those groups' cross-person pairs.
    sub = pool(hists, {0, 2})
    members = [i for i in range(n) if groups[i] in (0, 2)]
    expected = sum(
        1
        for a in range(len(members))
        for b in range(a + 1, len(members))
        if ids[members[a]] != ids[members[b]]
    )
    assert sub.total == expected


def test_histogram_scores_match_brute_force_distribution() -> None:
    emb, ids = blob_embeddings(20, 2)
    groups = np.zeros(len(emb), dtype=np.int64)
    hists = cross_identity_histograms(emb, ids, groups)
    brute = np.array(
        [
            emb[i] @ emb[j]
            for i in range(len(emb))
            for j in range(i + 1, len(emb))
            if ids[i] != ids[j]
        ]
    )
    h = hists[(0, 0)]
    for t in (0.0, 0.2, 0.4):
        assert h.far_at(t) == pytest.approx(float((brute >= t).mean()), abs=2e-4)


def test_rows_without_embeddings_or_group_are_ignored() -> None:
    emb, ids = blob_embeddings(6, 2)
    emb[0] = np.nan
    groups = np.zeros(len(emb), dtype=np.int64)
    groups[1] = -1
    total = sum(h.total for h in cross_identity_histograms(emb, ids, groups).values())
    usable = [i for i in range(len(emb)) if i not in (0, 1)]
    expected = sum(1 for a in usable for b in usable if a < b and ids[a] != ids[b])
    assert total == expected


# --- leakage -------------------------------------------------------------------------------------


def pair(fold: int, a: str, b: str, same: bool, ia: int = 1, ib: int = 2) -> LfwPair:
    return LfwPair(fold, same, a, ia, b, ib)


def test_leakage_report_passes_for_identity_disjoint_folds() -> None:
    pairs = [pair(0, "A", "A", True), pair(0, "B", "C", False), pair(1, "D", "D", True)]
    report = leakage_report(pairs, 2)
    assert report["passed"] and report["identities_in_multiple_folds"] == 0
    assert report["identities_per_fold"] == [3, 1]
    assert fold_identities(pairs, 2) == [{"A", "B", "C"}, {"D"}]


def test_leakage_report_flags_shared_identities_and_images() -> None:
    pairs = [pair(0, "A", "A", True), pair(1, "A", "B", False, ia=1, ib=1)]
    report = leakage_report(pairs, 2)
    assert not report["passed"]
    assert report["identities_in_multiple_folds"] == 1 and report["example_identities"] == ["A"]
    assert report["images_in_multiple_folds"] == 1  # A_0001 is used in both folds


# --- the protocol --------------------------------------------------------------------------------


def test_report_structure_and_reasonable_numbers() -> None:
    scores, same, fold = synthetic_scores()
    report = verification_report(scores, same, fold)
    assert len(report["folds"]) == N_FOLDS and report["n_failed_pairs"] == 0
    assert 0.95 < report["accuracy_mean"] < 1.0
    assert report["accuracy_std"] >= 0 and report["accuracy_sem"] == pytest.approx(
        report["accuracy_std"] / np.sqrt(N_FOLDS)
    )
    assert 0.0 < report["pooled"]["eer"] < 0.1 and report["pooled"]["auc"] > 0.97
    assert report["pooled"]["eer_failure_aware"] == pytest.approx(report["pooled"]["eer"])
    assert all(0.2 < f["threshold"] < 0.5 for f in report["folds"])  # between the two classes


def test_a_folds_threshold_never_depends_on_its_own_scores() -> None:
    scores, same, fold = synthetic_scores()
    base = verification_report(scores, same, fold)
    tampered = scores.copy()
    in_fold = fold == 3
    tampered[in_fold & same] = -0.9  # adversarial: genuine pairs look like impostors...
    tampered[in_fold & ~same] = 0.95  # ...and impostor pairs look genuine
    after = verification_report(tampered, same, fold)
    assert after["folds"][3]["threshold"] == base["folds"][3]["threshold"]  # held-out: unchanged
    others = [k for k in range(N_FOLDS) if k != 3]
    # Every other fold trains on fold 3, so at least one of their thresholds must react to it.
    assert any(after["folds"][k]["threshold"] != base["folds"][k]["threshold"] for k in others)


def test_failed_pairs_count_as_errors_not_as_dropped() -> None:
    scores, same, fold = synthetic_scores()
    clean = verification_report(scores, same, fold)
    failed = scores.copy()
    failed[[0, 1, 2, 130]] = np.nan  # three failures in fold 0 and one in fold 1
    report = verification_report(failed, same, fold)
    assert report["n_failed_pairs"] == 4 and report[
        "detection_failure_rate_pairs"
    ] == pytest.approx(4 / 1200)
    assert report["folds"][0]["n_failed_pairs"] == 3
    assert report["folds"][0]["accuracy"] == pytest.approx(
        report["folds"][0]["accuracy_valid_only"] * (120 - 3) / 120
    )
    assert report["accuracy_mean"] < report["accuracy_valid_only_mean"]
    assert report["accuracy_mean"] <= clean["accuracy_mean"] + 1e-9


def test_failed_genuine_lowers_tar_but_failed_impostor_does_not_raise_far() -> None:
    scores, same, fold = synthetic_scores()
    base = verification_report(scores, same, fold)
    genuine_idx = np.flatnonzero(same & (fold == 0))[:5]
    impostor_idx = np.flatnonzero(~same & (fold == 0))[:5]
    gen_failed = scores.copy()
    gen_failed[genuine_idx] = np.nan
    imp_failed = scores.copy()
    imp_failed[impostor_idx] = np.nan
    key = "0.001"
    b = base["tar_at_far_standard_protocol"]["targets"][key]["per_fold"][0]
    g = verification_report(gen_failed, same, fold)["tar_at_far_standard_protocol"]["targets"][key][
        "per_fold"
    ][0]
    i = verification_report(imp_failed, same, fold)["tar_at_far_standard_protocol"]["targets"][key][
        "per_fold"
    ][0]
    assert g["tar"] <= b["tar"] - 4 / PER_CLASS + 1e-9 or g["tar"] < b["tar"]  # misses count
    assert i["tar"] == pytest.approx(b["tar"], abs=0.05)


def test_failure_aware_eer_is_never_better() -> None:
    scores, same, fold = synthetic_scores()
    scores = scores.copy()
    scores[np.flatnonzero(~same)[:30]] = np.nan  # failed impostor pairs
    report = verification_report(scores, same, fold)
    assert report["pooled"]["eer_failure_aware"] >= report["pooled"]["eer"]
    assert FAILED_IMPOSTOR_SCORE > 1.0


def test_tar_is_monotone_in_the_far_target_and_flags_the_resolution_floor() -> None:
    scores, same, fold = synthetic_scores()
    report = verification_report(scores, same, fold)
    std = report["tar_at_far_standard_protocol"]
    assert std["targets"]["0.001"]["tar_mean"] >= std["targets"]["0.0001"]["tar_mean"]
    assert std["far_resolution_floor"] == pytest.approx(1 / (N_FOLDS * PER_CLASS))  # 1/n_impostor
    assert "tar_at_far_extended_impostors" not in report  # only when histograms are supplied


def test_extended_impostors_give_finer_far_control() -> None:
    scores, same, fold = synthetic_scores()
    emb, ids = blob_embeddings(40, 4, seed=3)
    groups = np.repeat(np.arange(N_FOLDS + 1), len(emb) // (N_FOLDS + 1) + 1)[: len(emb)]
    hists = cross_identity_histograms(emb, ids, groups)
    report = verification_report(scores, same, fold, histograms=hists)
    ext = report["tar_at_far_extended_impostors"]["targets"]["0.001"]
    assert len(ext["per_fold"]) == N_FOLDS
    assert ext["per_fold"][0]["n_impostor_train"] > 1000
    assert "NOT part of the standard protocol" in report["tar_at_far_extended_impostors"]["note"]
