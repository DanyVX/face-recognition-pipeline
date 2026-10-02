"""Evaluation metrics for verification (1:1) and open-set identification (1:N).

Conventions: a *score* is a similarity (higher = more alike); a pair/probe is *accepted* when
``score >= threshold`` (inclusive, the same rule the library uses). Pure numpy, no plotting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RocCurve:
    thresholds: FloatArray  # descending; the first one accepts nothing (+inf)
    far: FloatArray  # false accept rate at each threshold
    tar: FloatArray  # true accept rate at each threshold


def _split(
    scores: NDArray[np.floating], labels: NDArray[np.bool_]
) -> tuple[FloatArray, FloatArray]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    if s.shape != y.shape or s.ndim != 1:
        raise ValueError("scores and labels must be 1-D arrays of the same length")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite (count failed pairs separately, do not use NaN)")
    return s[y], s[~y]


def roc_curve(scores: NDArray[np.floating], labels: NDArray[np.bool_]) -> RocCurve:
    """ROC over every distinct score threshold (accept when score >= threshold)."""
    genuine, impostor = _split(scores, labels)
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("need at least one genuine and one impostor score")
    thresholds = np.unique(np.concatenate([genuine, impostor]))[::-1]
    thresholds = np.concatenate([[np.inf], thresholds])
    g_sorted, i_sorted = np.sort(genuine), np.sort(impostor)
    tar = 1.0 - np.searchsorted(g_sorted, thresholds, side="left") / genuine.size
    far = 1.0 - np.searchsorted(i_sorted, thresholds, side="left") / impostor.size
    return RocCurve(thresholds, far.astype(np.float64), tar.astype(np.float64))


def auc(curve: RocCurve) -> float:
    """Area under the ROC curve (trapezoid rule over FAR)."""
    order = np.argsort(curve.far, kind="stable")
    return float(np.trapezoid(curve.tar[order], curve.far[order]))


def eer(scores: NDArray[np.floating], labels: NDArray[np.bool_]) -> tuple[float, float]:
    """Equal error rate and the threshold where FAR and FRR cross (linear interpolation)."""
    curve = roc_curve(scores, labels)
    frr = 1.0 - curve.tar
    # Thresholds descend: FAR rises from 0 and FRR falls to 0, so FAR - FRR rises from -1 to +1.
    diff = curve.far - frr
    idx = int(np.argmax(diff >= 0))  # first threshold at or past the crossing (always exists)
    d1 = diff[idx]
    if idx == 0 or d1 == 0:
        return float(curve.far[idx]), float(curve.thresholds[max(idx, 1)])
    d0 = diff[idx - 1]  # negative: FAR still below FRR just before the crossing
    w = -d0 / (d1 - d0)
    rate = float(curve.far[idx - 1] + w * (curve.far[idx] - curve.far[idx - 1]))
    t0, t1 = curve.thresholds[idx - 1], curve.thresholds[idx]
    threshold = float(t1 if not np.isfinite(t0) else t0 + w * (t1 - t0))
    return rate, threshold


def threshold_at_far(impostor_scores: NDArray[np.floating], far: float) -> float:
    """Smallest threshold whose FAR on ``impostor_scores`` is at most ``far``.

    With ``n`` impostor scores the finest resolvable FAR is ``1/n``; asking for less returns a
    threshold above every impostor (FAR 0), which says nothing about the real FAR. Callers must
    report the resolution limit (see :func:`far_resolution`).
    """
    if not 0.0 < far <= 1.0:
        raise ValueError("far must be in (0, 1]")
    imp = np.sort(np.asarray(impostor_scores, dtype=np.float64))
    n = imp.size
    if n == 0:
        raise ValueError("no impostor scores")
    allowed = int(np.floor(far * n))  # impostors allowed at or above the threshold
    if allowed >= n:
        return float(imp[0])
    # Threshold just above the (allowed+1)-th largest score so at most `allowed` pass.
    cutoff = imp[n - allowed - 1]
    return float(np.nextafter(cutoff, np.inf))


def far_resolution(n_impostor: int) -> float:
    """Smallest non-zero FAR measurable with ``n_impostor`` impostor scores."""
    return 1.0 / max(n_impostor, 1)


def accuracy_at(scores: NDArray[np.floating], labels: NDArray[np.bool_], threshold: float) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    if s.size == 0:
        raise ValueError("no scores")
    return float(((s >= threshold) == y).mean())


def best_threshold(scores: NDArray[np.floating], labels: NDArray[np.bool_]) -> tuple[float, float]:
    """Threshold maximizing accuracy, and that accuracy. Ties go to the midpoint of the plateau.

    Candidate thresholds are midpoints between consecutive distinct scores, so the chosen value
    never coincides with a score (no ambiguity about ``>=``).
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    if s.size == 0:
        raise ValueError("no scores")
    order = np.argsort(s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    # Accept everything >= candidate: correct = impostors below + genuine at/above.
    impostors_below = np.concatenate([[0], np.cumsum(~y_sorted)])
    genuine_above = y_sorted.sum() - np.concatenate([[0], np.cumsum(y_sorted)])
    correct = impostors_below + genuine_above  # candidate k splits before s_sorted[k]
    # A split between two equal scores is not reachable by any threshold: exclude it.
    reachable = np.ones(s.size + 1, dtype=bool)
    reachable[1 : s.size] = s_sorted[:-1] < s_sorted[1:]
    correct = np.where(reachable, correct, -1)
    best = correct.max()
    ks = np.flatnonzero(correct == best)
    k = int(ks[len(ks) // 2])
    if k == 0:
        threshold = float(np.nextafter(s_sorted[0], -np.inf))
    elif k == s.size:
        threshold = float(np.nextafter(s_sorted[-1], np.inf))
    else:
        threshold = float((s_sorted[k - 1] + s_sorted[k]) / 2.0)
    return threshold, float(best / s.size)


# --- open-set identification ---------------------------------------------------------------------


@dataclass(frozen=True)
class DetCurve:
    thresholds: FloatArray
    fpir: FloatArray  # false positive identification rate
    fnir: FloatArray  # false negative identification rate


def rank1_rate(mated_correct: NDArray[np.bool_]) -> float:
    """Closed-set style rank-1 identification rate over mated probes (no threshold)."""
    c = np.asarray(mated_correct, dtype=bool)
    if c.size == 0:
        raise ValueError("no mated probes")
    return float(c.mean())


def det_curve(
    mated_top_score: NDArray[np.floating],
    mated_correct: NDArray[np.bool_],
    nonmated_top_score: NDArray[np.floating],
) -> DetCurve:
    """Open-set DET data from each probe's best candidate.

    ``mated_top_score[i]`` is the top candidate's score for a probe whose identity IS enrolled and
    ``mated_correct[i]`` says whether that candidate is the right identity;
    ``nonmated_top_score[j]`` is the top score for a probe whose identity is NOT enrolled.
    At threshold ``t``: FPIR = P(non-mated top score >= t); FNIR = 1 - P(correct and score >= t)
    (a mated probe is a miss if the right identity is not rank 1 or is below the threshold).
    """
    ms = np.asarray(mated_top_score, dtype=np.float64)
    mc = np.asarray(mated_correct, dtype=bool)
    ns = np.asarray(nonmated_top_score, dtype=np.float64)
    if ms.size == 0 or ns.size == 0 or ms.shape != mc.shape:
        raise ValueError("need matching mated arrays and at least one non-mated probe")
    thresholds = np.unique(np.concatenate([ms, ns]))[::-1]
    thresholds = np.concatenate([[np.inf], thresholds])
    ns_sorted = np.sort(ns)
    hits_sorted = np.sort(ms[mc])
    fpir = 1.0 - np.searchsorted(ns_sorted, thresholds, side="left") / ns.size
    correct_above = hits_sorted.size - np.searchsorted(hits_sorted, thresholds, side="left")
    fnir = 1.0 - correct_above / ms.size
    return DetCurve(thresholds, fpir.astype(np.float64), fnir.astype(np.float64))


def fnir_at_fpir(curve: DetCurve, target_fpir: float) -> tuple[float, float, float]:
    """(FNIR, achieved FPIR, threshold) at the loosest threshold with FPIR <= ``target_fpir``."""
    ok = np.flatnonzero(curve.fpir <= target_fpir)
    idx = int(ok[-1])  # thresholds descend, so the last ok index is the lowest allowed threshold
    return float(curve.fnir[idx]), float(curve.fpir[idx]), float(curve.thresholds[idx])


# --- uncertainty ---------------------------------------------------------------------------------


def bootstrap_ci(
    statistic: Callable[[NDArray[np.intp]], float],
    n_items: int,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``statistic(indices)`` over resamples of ``n_items`` items."""
    if n_items < 2:
        raise ValueError("need at least two items to bootstrap")
    rng = np.random.default_rng(seed)
    values = np.array(
        [statistic(rng.integers(0, n_items, n_items)) for _ in range(n_boot)], dtype=np.float64
    )
    lo = (1.0 - confidence) / 2.0
    return float(np.quantile(values, lo)), float(np.quantile(values, 1.0 - lo))


def mean_std(values: NDArray[np.floating] | list[float]) -> tuple[float, float, float]:
    """(mean, sample standard deviation, standard error of the mean)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("no values")
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std, std / float(np.sqrt(arr.size))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (good for small ``n`` and p near 1)."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("need n > 0 and 0 <= successes <= n")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * float(np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)
