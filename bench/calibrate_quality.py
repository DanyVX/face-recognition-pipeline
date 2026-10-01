"""Calibrate quality-gate thresholds on a validation split; evaluate once on held-out data.

Usage: ``uv run python -m bench.calibrate_quality [--budget 0.10] [--limit N]``
Needs ``bench.prepare_lfw``, ``bench.embed_lfw`` and ``bench.robustness`` to have been run.

Method
* **Fixed decision threshold** ``tau``: best-accuracy threshold on *clean* validation pairs
  (LFW folds 0-1). It is never re-tuned, so a gate cannot "improve" accuracy by changing ``tau``.
* **Validation set** = clean pairs of folds 0-1 plus the degraded validation probes from
  ``bench.robustness`` (blur, JPEG, brightness, downscale, rotation, occlusion...). LFW alone is
  near-saturated (only a handful of errors), which cannot calibrate anything; degradations are
  what the gates exist to catch. This choice is stated in docs/RESULTS.md.
* **Per gate**: a pair is rejected when its *worse* image fails the gate. Candidate thresholds are
  quantiles of the pair-level feature giving reject rates up to ``--budget``. The chosen threshold
  maximizes ``accuracy(kept) - accuracy(all)`` and is only *kept* if a paired bootstrap 95% CI of
  that gain excludes zero; otherwise the gate stays disabled (``None``) rather than shipping an
  unjustified value.
* **Held-out evaluation** (clean folds 2-9 plus degraded test probes), done once with the chosen
  thresholds; the trade-off curves (reject rate vs accuracy of kept pairs) are reported for both
  splits.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from bench.common import write_result
from bench.lfw_data import (
    LfwCache,
    drop_unprepared,
    load_embeddings,
    load_pairs,
    pair_scores,
    resolve_pairs,
)
from bench.metrics import best_threshold
from frp.align import aligned_from_crop
from frp.config import load_settings
from frp.factory import build_embedder
from frp.quality import QualityThresholds
from frp.quality.features import crop_features
from frp.quality.pose import pose_from_aligned

log = logging.getLogger("calibrate_quality")

Direction = Literal["low", "high"]
Combine = Literal["min", "max", "maxabs"]
FEATURE_ORDER = (
    "face_px",
    "det_score",
    "laplacian_var",
    "brightness",
    "contrast",
    "yaw_ratio",
    "pitch_ratio",
    "fit_error",
    "truncated",
)


@dataclass(frozen=True)
class Rule:
    """How one threshold of ``QualityThresholds`` maps onto per-image features."""

    threshold_name: str
    feature: str
    combine: Combine  # how the two images of a pair are combined into one value
    direction: Direction  # "low": reject when value < t;  "high": reject when value > t


RULES: tuple[Rule, ...] = (
    Rule("min_face_px", "face_px", "min", "low"),
    Rule("min_det_score", "det_score", "min", "low"),
    Rule("min_laplacian_var", "laplacian_var", "min", "low"),
    Rule("min_brightness", "brightness", "min", "low"),
    Rule("max_brightness", "brightness", "max", "high"),
    Rule("min_contrast", "contrast", "min", "low"),
    Rule("max_abs_yaw_ratio", "yaw_ratio", "maxabs", "high"),
    Rule("max_abs_pitch_ratio", "pitch_ratio", "maxabs", "high"),
)


def pair_value(rule: Rule, a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    if rule.combine == "min":
        return np.minimum(a, b)
    if rule.combine == "max":
        return np.maximum(a, b)
    return np.maximum(np.abs(a), np.abs(b))


def rejected(
    values: NDArray[np.float64], threshold: float, direction: Direction
) -> NDArray[np.bool_]:
    return np.asarray(values < threshold if direction == "low" else values > threshold)


def gain_at(
    values: NDArray[np.float64],
    correct: NDArray[np.bool_],
    threshold: float,
    direction: Direction,
    idx: NDArray[np.intp] | None = None,
) -> tuple[float, float, float]:
    """(reject rate, accuracy of kept pairs, gain over accuracy of all) on ``idx`` rows."""
    v = values if idx is None else values[idx]
    c = correct if idx is None else correct[idx]
    keep = ~rejected(v, threshold, direction)
    if not keep.any():
        return 1.0, float("nan"), float("nan")
    acc_all = float(c.mean())
    acc_kept = float(c[keep].mean())
    return float(1.0 - keep.mean()), acc_kept, acc_kept - acc_all


def candidate_thresholds(
    values: NDArray[np.float64], direction: Direction, budget: float, n: int = 24
) -> NDArray[np.float64]:
    """Thresholds whose reject rates are spread over (0, budget]."""
    levels = np.linspace(budget / n, budget, n)
    q = levels if direction == "low" else 1.0 - levels
    return np.unique(np.quantile(values, q))


def sweep_rows(
    values: NDArray[np.float64], correct: NDArray[np.bool_], direction: Direction, budget: float
) -> list[dict[str, float]]:
    """Candidate thresholds with their reject rate, kept-pair accuracy and gain."""
    rows: list[dict[str, float]] = []
    for t in candidate_thresholds(values, direction, budget):
        reject, acc_kept, gain = gain_at(values, correct, float(t), direction)
        rows.append(
            {"threshold": float(t), "reject_rate": reject, "accuracy_kept": acc_kept, "gain": gain}
        )
    return rows


def best_row(rows: list[dict[str, float]], budget: float) -> dict[str, float] | None:
    """Highest-gain candidate within the reject budget (ties: fewer rejections)."""
    usable = [r for r in rows if np.isfinite(r["gain"]) and r["reject_rate"] <= budget + 1e-9]
    return max(usable, key=lambda r: (r["gain"], -r["reject_rate"])) if usable else None


def select_threshold(
    values: NDArray[np.float64],
    correct: NDArray[np.bool_],
    direction: Direction,
    *,
    budget: float,
    alpha: float = 0.05,
    n_boot: int = 1000,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Pick the best-gain threshold within the reject budget; keep it only if it generalizes.

    Choosing the best of many thresholds on a sample and then bootstrapping *that* threshold's
    gain on the same sample is optimistically biased (the winner's curse). Significance is
    therefore judged by **cross-fitting**: for each of ``n_splits`` folds the threshold is chosen on
    the other folds and applied to the held-out fold; the pooled out-of-fold decisions give the
    gain whose bootstrap ``1 - alpha`` CI must exclude zero. The reported ``selected`` threshold is
    then the best one on the full validation set. ``alpha`` should already be corrected for the
    number of gates tested (see :func:`calibrate`, which applies Bonferroni).
    """
    rows = sweep_rows(values, correct, direction, budget)
    result: dict[str, Any] = {
        "sweep": rows,
        "accuracy_all": float(correct.mean()),
        "selected": None,
        "significant": False,
    }
    best = best_row(rows, budget)
    if best is None:
        return result
    rng = np.random.default_rng(seed)
    assignment = rng.permutation(values.size) % n_splits
    out_of_fold_reject = np.zeros(values.size, dtype=bool)
    fold_thresholds: list[float] = []
    for fold in range(n_splits):
        train, held_out = assignment != fold, assignment == fold
        candidate = best_row(sweep_rows(values[train], correct[train], direction, budget), budget)
        if candidate is not None:
            fold_thresholds.append(candidate["threshold"])
            out_of_fold_reject[held_out] = rejected(
                values[held_out], candidate["threshold"], direction
            )

    def oof_gain(idx: NDArray[np.intp]) -> float:
        keep = ~out_of_fold_reject[idx]
        if not keep.any():
            return float("nan")
        return float(correct[idx][keep].mean() - correct[idx].mean())

    gains = np.array([oof_gain(rng.integers(0, values.size, values.size)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(gains, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    significant = bool(out_of_fold_reject.any() and lo > 0.0)
    result.update(
        {
            "best_candidate": best,
            "out_of_fold_gain": oof_gain(np.arange(values.size)),
            "out_of_fold_reject_rate": float(out_of_fold_reject.mean()),
            "fold_thresholds": fold_thresholds,
            "ci_confidence": 1.0 - alpha,
            "gain_ci95": [float(lo), float(hi)],  # name kept for readers; see ci_confidence
            "significant": significant,
            "selected": best["threshold"] if significant else None,
        }
    )
    return result


def image_features(cache: LfwCache) -> dict[str, NDArray[np.float64]]:
    """Per-image quality features for every cached crop (NaN where no face was found)."""
    n = len(cache)
    out = {name: np.full(n, np.nan) for name in FEATURE_ORDER}
    height = width = 250  # LFW images are 250 x 250
    for i in np.flatnonzero(cache.ok):
        aligned = aligned_from_crop(np.asarray(cache.aligned[i]), cache.landmarks[i])
        lap, bright, contrast = crop_features(aligned)
        pose = pose_from_aligned(aligned)
        box, lm = cache.box[i], cache.landmarks[i]
        touches = bool(box[0] <= 1 or box[1] <= 1 or box[2] >= width - 1 or box[3] >= height - 1)
        inside = bool(
            (lm[:, 0] >= 0).all()
            and (lm[:, 0] <= width).all()
            and (lm[:, 1] >= 0).all()
            and (lm[:, 1] <= height).all()
        )
        out["face_px"][i] = float(min(box[2] - box[0], box[3] - box[1]))
        out["det_score"][i] = float(cache.det_score[i])
        out["laplacian_var"][i] = lap
        out["brightness"][i] = bright
        out["contrast"][i] = contrast
        out["yaw_ratio"][i] = pose.yaw_ratio
        out["pitch_ratio"][i] = pose.pitch_ratio
        out["fit_error"][i] = float(aligned.fit_error)
        out["truncated"][i] = float(touches or not inside)
    return out


@dataclass
class SampleSet:
    """Pairs with scores and the pair-level feature values for every rule."""

    score: NDArray[np.float64]
    same: NDArray[np.bool_]
    source: NDArray[np.str_]  # "clean" or the degradation name
    a: dict[str, NDArray[np.float64]]  # per-image features of the first (clean) image
    b: dict[str, NDArray[np.float64]]  # ... of the second image (clean or degraded probe)

    def correct(self, tau: float) -> NDArray[np.bool_]:
        return np.asarray((self.score >= tau) == self.same)

    def values(self, rule: Rule) -> NDArray[np.float64]:
        return pair_value(rule, self.a[rule.feature], self.b[rule.feature])

    @classmethod
    def concat(cls, sets: list[SampleSet]) -> SampleSet:
        return cls(
            score=np.concatenate([s.score for s in sets]),
            same=np.concatenate([s.same for s in sets]),
            source=np.concatenate([s.source for s in sets]),
            a={k: np.concatenate([s.a[k] for s in sets]) for k in sets[0].a},
            b={k: np.concatenate([s.b[k] for s in sets]) for k in sets[0].b},
        )

    def only_valid(self) -> SampleSet:
        keep = np.isfinite(self.score)
        for name in FEATURE_ORDER:
            keep &= np.isfinite(self.a[name]) & np.isfinite(self.b[name])
        return SampleSet(
            self.score[keep],
            self.same[keep],
            self.source[keep],
            {k: v[keep] for k, v in self.a.items()},
            {k: v[keep] for k, v in self.b.items()},
        )


def clean_sample_set(
    folds: tuple[int, ...],
    scores: NDArray[np.float64],
    same: NDArray[np.bool_],
    fold: NDArray[np.int64],
    row_a: NDArray[np.int64],
    row_b: NDArray[np.int64],
    feats: dict[str, NDArray[np.float64]],
) -> SampleSet:
    sel = np.isin(fold, folds) & (row_a >= 0) & (row_b >= 0)
    return SampleSet(
        score=scores[sel],
        same=same[sel],
        source=np.full(int(sel.sum()), "clean"),
        a={k: v[row_a[sel]] for k, v in feats.items()},
        b={k: v[row_b[sel]] for k, v in feats.items()},
    )


def degraded_sample_set(npz: Any, split: str, feats: dict[str, NDArray[np.float64]]) -> SampleSet:
    """All degraded probes of ``split`` from the robustness cache, as pairs."""
    names = [str(n) for n in npz["feature_names"]]
    same = npz[f"{split}_same"]
    row_a = npz[f"{split}_row_a"]
    prefix, suffix = f"{split}_", "_scores"
    sets: list[SampleSet] = []
    for key in npz.files:
        if not (key.startswith(prefix) and key.endswith(suffix)) or key.endswith("clean_scores"):
            continue
        setting = key[len(prefix) : -len(suffix)]
        probe_feats = npz[f"{prefix}{setting}_features"]
        sets.append(
            SampleSet(
                score=npz[key],
                same=same,
                source=np.full(same.size, setting),
                a={k: v[row_a] for k, v in feats.items()},
                b={n: probe_feats[:, names.index(n)] for n in names},
            )
        )
    return SampleSet.concat(sets)


def calibrate(
    validation: SampleSet, tau: float, budget: float, seed: int = 0, alpha: float = 0.05
) -> dict[str, dict[str, Any]]:
    """Select (or reject) every gate. ``alpha`` is Bonferroni-corrected over the gates tested, so
    that testing eight gates does not make a spurious "significant" gate likely."""
    correct = validation.correct(tau)
    corrected = alpha / len(RULES)
    return {
        rule.threshold_name: select_threshold(
            validation.values(rule),
            correct,
            rule.direction,
            budget=budget,
            alpha=corrected,
            seed=seed,
        )
        for rule in RULES
    }


def thresholds_from(selection: dict[str, dict[str, Any]]) -> QualityThresholds:
    chosen: dict[str, Any] = {
        r.threshold_name: selection[r.threshold_name]["selected"] for r in RULES
    }
    return QualityThresholds(
        min_face_px=chosen["min_face_px"],
        min_det_score=chosen["min_det_score"],
        min_laplacian_var=chosen["min_laplacian_var"],
        min_brightness=chosen["min_brightness"],
        max_brightness=chosen["max_brightness"],
        min_contrast=chosen["min_contrast"],
        max_abs_yaw_ratio=chosen["max_abs_yaw_ratio"],
        max_abs_pitch_ratio=chosen["max_abs_pitch_ratio"],
        reject_cut_off_faces=True,  # a policy choice, not a calibrated value
    )


def gated_rejection(samples: SampleSet, thresholds: QualityThresholds) -> NDArray[np.bool_]:
    """Pairs rejected by any enabled calibrated gate (the truncation policy is excluded)."""
    reject = np.zeros(samples.score.size, dtype=bool)
    for rule in RULES:
        t = getattr(thresholds, rule.threshold_name)
        if t is not None:
            reject |= rejected(samples.values(rule), float(t), rule.direction)
    return reject


def evaluate(
    samples: SampleSet,
    tau: float,
    thresholds: QualityThresholds,
    *,
    n_boot: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    correct = samples.correct(tau)
    reject = gated_rejection(samples, thresholds)
    out: dict[str, Any] = {
        "n": int(correct.size),
        "accuracy_all": float(correct.mean()),
        "reject_rate": float(reject.mean()),
        "accuracy_kept": float(correct[~reject].mean()) if (~reject).any() else None,
    }
    if (~reject).any() and reject.any():
        rng = np.random.default_rng(seed)
        gains = []
        for _ in range(n_boot):
            idx = rng.integers(0, correct.size, correct.size)
            kept = ~reject[idx]
            gains.append(
                float(correct[idx][kept].mean() - correct[idx].mean()) if kept.any() else np.nan
            )
        out["gain"] = out["accuracy_kept"] - out["accuracy_all"]
        out["gain_ci95"] = [float(x) for x in np.nanpercentile(gains, [2.5, 97.5])]
    by_source: dict[str, dict[str, float]] = {}
    for src in sorted({str(s) for s in samples.source}):
        m = samples.source == src
        by_source[src] = {
            "n": int(m.sum()),
            "accuracy_all": float(correct[m].mean()),
            "reject_rate": float(reject[m].mean()),
        }
    out["by_source"] = by_source
    return out


def curves(
    samples: SampleSet, tau: float, budget: float = 0.3
) -> dict[str, list[dict[str, float]]]:
    """Reject-rate vs accuracy-of-kept-pairs trade-off curve for every gate (for the plots)."""
    correct = samples.correct(tau)
    out: dict[str, list[dict[str, float]]] = {}
    for rule in RULES:
        values = samples.values(rule)
        rows = []
        for t in candidate_thresholds(values, rule.direction, budget, n=30):
            reject, acc_kept, _ = gain_at(values, correct, float(t), rule.direction)
            rows.append({"threshold": float(t), "reject_rate": reject, "accuracy_kept": acc_kept})
        out[rule.threshold_name] = rows
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--robustness", default="robustness")
    parser.add_argument("--name", default="quality_calibration")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    settings = load_settings()
    cache = LfwCache.load(args.limit)
    info = build_embedder().info
    embeddings = load_embeddings(cache, info)
    resolved = resolve_pairs(load_pairs(), cache)
    if args.limit:
        resolved = drop_unprepared(resolved)
    scores = pair_scores(resolved, embeddings, cache.ok)
    feats = image_features(cache)
    npz = np.load(settings.data_dir / "cache" / f"{args.robustness}_samples.npz")

    clean_val = clean_sample_set(
        (0, 1), scores, resolved.same, resolved.fold, resolved.row_a, resolved.row_b, feats
    ).only_valid()
    tau, tau_acc = best_threshold(clean_val.score, clean_val.same)
    log.info(
        "tau %.4f from %d clean validation pairs (accuracy %.4f)",
        tau,
        clean_val.score.size,
        tau_acc,
    )
    validation = SampleSet.concat(
        [clean_val, degraded_sample_set(npz, "val", feats).only_valid()]
    ).only_valid()
    selection = calibrate(validation, tau, args.budget)
    thresholds = thresholds_from(selection)
    clean_test = clean_sample_set(
        tuple(range(2, 10)),
        scores,
        resolved.same,
        resolved.fold,
        resolved.row_a,
        resolved.row_b,
        feats,
    ).only_valid()
    test = SampleSet.concat(
        [clean_test, degraded_sample_set(npz, "test", feats).only_valid()]
    ).only_valid()
    result = {
        "tau": tau,
        "tau_source": "best-accuracy threshold on clean validation pairs (folds 0-1)",
        "budget": args.budget,
        "validation": {
            "n": int(validation.score.size),
            "evaluation": evaluate(validation, tau, thresholds),
        },
        "selection": selection,
        "thresholds": thresholds.to_dict(),
        "held_out": evaluate(test, tau, thresholds),
        "curves_validation": curves(validation, tau),
        "curves_test": curves(test, tau),
        "clean_only_held_out": evaluate(clean_test, tau, thresholds),
    }
    write_result(args.name, result, {"budget": args.budget, "limit": args.limit})
    if not args.limit:
        write_result(
            "quality_thresholds",
            {"thresholds": thresholds.to_dict()},
            {"source": args.name, "tau": tau},
        )
    enabled = [k for k, v in thresholds.to_dict().items() if v not in (None, True, False)]
    log.info("enabled gates: %s", enabled or "none (no gate showed a significant gain)")
    log.info(
        "held-out: %s",
        {k: result["held_out"][k] for k in ("accuracy_all", "reject_rate", "accuracy_kept")},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
