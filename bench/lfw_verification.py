"""LFW 1:1 verification: standard 10-fold protocol, ROC/EER, TAR@FAR, leakage check.

Usage: ``uv run python -m bench.lfw_verification [--limit N] [--name lfw_verification]``

Conventions (all recorded in the result file):
* A pair is *accepted* when its cosine score is ``>= threshold``.
* The threshold of each fold is chosen on the other nine folds only and applied to the held-out
  fold; the held-out fold is never used to pick it.
* Detection/alignment failures are not dropped. For **accuracy** a failed pair counts as an error
  whatever its label. For **TAR/FAR/EER** the ISO/IEC 19795-1 convention is used: a failed genuine
  pair is a false rejection (it lowers TAR); failed impostor pairs are not false accepts, but the
  failure counts are reported next to every number. ``eer_failure_aware`` additionally treats
  failed impostor pairs as false accepts.
* TAR@FAR is reported twice: from the standard 3000 impostor pairs (whose resolution floor is
  about 3.3e-4, so FAR=1e-4 is not measurable there) and from an *extended* impostor set made of
  every cross-identity image pair, which is not part of the standard protocol and is labelled so.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bench.common import write_result
from bench.impostors import ScoreHistogram, cross_identity_histograms, pool
from bench.lfw_data import (
    LfwCache,
    drop_unprepared,
    load_embeddings,
    load_pairs,
    pair_scores,
    resolve_pairs,
)
from bench.metrics import (
    auc,
    best_threshold,
    eer,
    far_resolution,
    mean_std,
    roc_curve,
    threshold_at_far,
)
from frp.embed import ModelInfo
from frp.factory import build_embedder
from frp.lfw import LfwPair

log = logging.getLogger("lfw_verification")

FAR_TARGETS = (1e-3, 1e-4)
FAILED_GENUINE_SCORE = -2.0  # below any cosine: never accepted
FAILED_IMPOSTOR_SCORE = 2.0  # above any cosine: always accepted


def fold_identities(pairs: list[LfwPair], n_folds: int) -> list[set[str]]:
    sets: list[set[str]] = [set() for _ in range(n_folds)]
    for p in pairs:
        sets[p.fold].update((p.name_a, p.name_b))
    return sets


def leakage_report(pairs: list[LfwPair], n_folds: int) -> dict[str, Any]:
    """Identities (and images) that appear in pairs of more than one fold."""
    people = fold_identities(pairs, n_folds)
    overlapping = sorted(
        {
            name
            for i in range(n_folds)
            for j in range(i + 1, n_folds)
            for name in people[i] & people[j]
        }
    )
    images: dict[tuple[str, int], set[int]] = {}
    for p in pairs:
        images.setdefault((p.name_a, p.index_a), set()).add(p.fold)
        images.setdefault((p.name_b, p.index_b), set()).add(p.fold)
    shared_images = sum(1 for folds in images.values() if len(folds) > 1)
    return {
        "identities_in_multiple_folds": len(overlapping),
        "example_identities": overlapping[:5],
        "images_in_multiple_folds": shared_images,
        "identities_per_fold": [len(s) for s in people],
        "passed": not overlapping and shared_images == 0,
    }


def roc_points(
    scores: NDArray[np.float64], same: NDArray[np.bool_], max_points: int = 400
) -> dict[str, list[float]]:
    curve = roc_curve(scores, same)
    step = max(1, len(curve.far) // max_points)
    return {"far": curve.far[::step].tolist(), "tar": curve.tar[::step].tolist()}


def verification_report(
    scores: NDArray[np.float64],
    same: NDArray[np.bool_],
    fold: NDArray[np.int64],
    *,
    n_folds: int = 10,
    fars: tuple[float, ...] = FAR_TARGETS,
    histograms: dict[tuple[int, int], ScoreHistogram] | None = None,
) -> dict[str, Any]:
    """All verification numbers from per-pair scores (NaN = failed pair)."""
    valid = np.isfinite(scores)
    folds: list[dict[str, Any]] = []
    standard: dict[str, list[dict[str, float]]] = {f"{far:g}": [] for far in fars}
    extended: dict[str, list[dict[str, float]]] = {f"{far:g}": [] for far in fars}
    for k in range(n_folds):
        test = fold == k
        train_valid = (fold != k) & valid
        tau, train_acc = best_threshold(scores[train_valid], same[train_valid])
        test_valid = test & valid
        correct = int(((scores[test_valid] >= tau) == same[test_valid]).sum())
        folds.append(
            {
                "fold": k,
                "threshold": tau,
                "train_accuracy": train_acc,
                "accuracy": correct / int(test.sum()),  # failed pairs count as errors
                "accuracy_valid_only": correct / max(int(test_valid.sum()), 1),
                "n_pairs": int(test.sum()),
                "n_failed_pairs": int((test & ~valid).sum()),
            }
        )
        genuine_test = test & same
        genuine_valid = genuine_test & valid
        impostor_test_valid = test & ~same & valid
        train_impostors = scores[train_valid & ~same]
        for far in fars:
            key = f"{far:g}"
            tau_far = threshold_at_far(train_impostors, far)
            standard[key].append(
                {
                    "fold": k,
                    "threshold": tau_far,
                    "tar": float(
                        (scores[genuine_valid] >= tau_far).sum() / max(genuine_test.sum(), 1)
                    ),
                    "far": float((scores[impostor_test_valid] >= tau_far).mean())
                    if impostor_test_valid.any()
                    else float("nan"),
                    "n_impostor_train": int(train_impostors.size),
                }
            )
            if histograms is not None:
                train_hist = pool(histograms, set(range(n_folds)) - {k} | {n_folds})
                held_hist = histograms.get((k, k))
                tau_ext = train_hist.threshold_at_far(far)
                extended[key].append(
                    {
                        "fold": k,
                        "threshold": tau_ext,
                        "tar": float(
                            (scores[genuine_valid] >= tau_ext).sum() / max(genuine_test.sum(), 1)
                        ),
                        "far": held_hist.far_at(tau_ext)
                        if held_hist and held_hist.total
                        else float("nan"),
                        "n_impostor_train": train_hist.total,
                        "n_impostor_heldout": held_hist.total if held_hist else 0,
                    }
                )
    acc_mean, acc_std, acc_sem = mean_std([f["accuracy"] for f in folds])
    valid_mean, _, _ = mean_std([f["accuracy_valid_only"] for f in folds])

    curve = roc_curve(scores[valid], same[valid])
    eer_rate, eer_threshold = eer(scores[valid], same[valid])
    aware = np.where(valid, scores, np.where(same, FAILED_GENUINE_SCORE, FAILED_IMPOSTOR_SCORE))
    eer_aware, _ = eer(aware, same)

    def summarize(rows: list[dict[str, float]]) -> dict[str, Any]:
        tars = [r["tar"] for r in rows]
        fars_measured = [r["far"] for r in rows if np.isfinite(r["far"])]
        mean, std, _ = mean_std(tars)
        return {
            "tar_mean": mean,
            "tar_std": std,
            "far_measured_mean": float(np.mean(fars_measured)) if fars_measured else float("nan"),
            "per_fold": rows,
        }

    n_impostor_std = int((~same).sum())
    report: dict[str, Any] = {
        "n_pairs": int(scores.size),
        "n_failed_pairs": int((~valid).sum()),
        "detection_failure_rate_pairs": float((~valid).mean()),
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "accuracy_sem": acc_sem,
        "accuracy_valid_only_mean": valid_mean,
        "folds": folds,
        "pooled": {
            "auc": auc(curve),
            "eer": eer_rate,
            "eer_threshold": eer_threshold,
            "eer_failure_aware": eer_aware,
            "roc": roc_points(scores[valid], same[valid]),
        },
        "tar_at_far_standard_protocol": {
            "note": "thresholds from the other folds' impostor pairs, applied to the held-out fold",
            "far_resolution_floor": far_resolution(n_impostor_std),
            "targets": {key: summarize(rows) for key, rows in standard.items()},
        },
        "mean_fold_threshold": float(np.mean([f["threshold"] for f in folds])),
    }
    if histograms is not None:
        report["tar_at_far_extended_impostors"] = {
            "note": (
                "NOT part of the standard protocol: impostor distribution from every "
                "cross-identity image pair; thresholds from training identities, FAR measured "
                "on held-out identities"
            ),
            "targets": {key: summarize(rows) for key, rows in extended.items()},
        }
    return report


def identity_groups(
    cache: LfwCache, pairs: list[LfwPair], n_folds: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """(identity id, fold group) per image; group ``n_folds`` holds people in no pair."""
    people = fold_identities(pairs, n_folds)
    group_of: dict[str, int] = {}
    for k in range(n_folds - 1, -1, -1):  # an identity in several folds goes to the lowest one
        for name in people[k]:
            group_of[name] = k
    names = cache.identities
    ids = {name: i for i, name in enumerate(sorted(set(names)))}
    identity_ids = np.array([ids[n] for n in names], dtype=np.int64)
    groups = np.array([group_of.get(n, n_folds) for n in names], dtype=np.int64)
    return identity_ids, groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--name", default="lfw_verification")
    parser.add_argument("--no-extended", action="store_true", help="skip the extended impostor set")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cache = LfwCache.load(args.limit)
    embedder_info: ModelInfo = build_embedder().info
    embeddings = load_embeddings(cache, embedder_info)
    pairs = load_pairs()
    resolved = resolve_pairs(pairs, cache)
    n_folds = 10
    if args.limit:
        resolved = drop_unprepared(resolved)
        log.warning("tiny run: %d of %d pairs evaluated", len(resolved), len(pairs))
        pairs = resolved.pairs
    scores = pair_scores(resolved, embeddings, cache.ok)
    histograms = None
    if not args.no_extended:
        log.info("building extended impostor histograms")
        identity_ids, groups = identity_groups(cache, pairs, n_folds)
        histograms = cross_identity_histograms(embeddings, identity_ids, groups)
    report = verification_report(
        scores, resolved.same, resolved.fold, n_folds=n_folds, histograms=histograms
    )
    report["leakage_check"] = leakage_report(pairs, n_folds)
    report["model"] = embedder_info.tag
    report["detection"] = {
        "images_total": len(cache),
        "images_failed": int((~cache.ok).sum()),
        "image_status_counts": cache.manifest["status_counts"],
        "multi_face_policy": cache.manifest["policy"],
    }
    path = write_result(
        args.name, report, {"limit": args.limit, "n_folds": n_folds, "fars": list(FAR_TARGETS)}
    )
    log.info(
        "accuracy %.4f +/- %.4f (std over folds); EER %.4f; wrote %s",
        report["accuracy_mean"],
        report["accuracy_std"],
        report["pooled"]["eer"],
        path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
