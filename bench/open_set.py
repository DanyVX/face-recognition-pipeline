"""LFW open-set identification (1:N) through the real ``Gallery`` service.

Usage: ``uv run python -m bench.open_set [--limit N] [--seeds 5] [--name open_set]``

Protocol (per random seed):
* People with at least ``n_enroll + 1`` images are *eligible*. Half of them (random) are enrolled
  with ``n_enroll`` images each (the first usable ones); the other half are NOT enrolled.
* **Mated probes**: the remaining images of enrolled people. **Non-mated probes**: every image of
  the non-enrolled eligible people plus every image of people with too few images to be eligible.
  Non-mated probes are therefore strangers the system has never seen, which is what an open set
  means; closed-set rank-1 alone ignores them and is misleading.
* Metrics: rank-1 identification rate (mated probes only), FNIR at fixed FPIR (1e-1, 1e-2, and
  1e-3 when enough non-mated probes exist to resolve it) and the DET curve.
* Failures are not dropped: a mated probe whose face is not found is a miss (it counts in FNIR and
  in the failure-inclusive rank-1). Failed non-mated probes cannot produce a false positive and
  are excluded from the FPIR denominator, with their number reported (ISO/IEC 19795-1 style).
* People whose enrollment images contain no usable face cannot be enrolled; their number is
  reported as enrollment failures.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bench.common import write_result
from bench.lfw_data import LfwCache, load_embeddings
from bench.metrics import det_curve, fnir_at_fpir, mean_std, rank1_rate
from frp.embed import EmbeddingBatch, ModelInfo
from frp.factory import build_embedder
from frp.gallery import (
    Aggregation,
    DuplicatePolicy,
    Gallery,
    GalleryConfig,
)
from frp.index import Backend
from frp.store import IdentityStore

log = logging.getLogger("open_set")

FPIR_TARGETS = (1e-1, 1e-2, 1e-3)
MISS_SCORE = -2.0  # a failed mated probe never reaches any threshold


def run_once(
    embeddings: NDArray[np.float32],
    usable: NDArray[np.bool_],
    people: list[str],
    info: ModelInfo,
    *,
    seed: int,
    n_enroll: int = 1,
    aggregation: Aggregation = Aggregation.MAX,
    backend: Backend = Backend.FLAT,
) -> dict[str, Any]:
    """One random enrolled/non-enrolled split, evaluated end to end through ``Gallery``."""
    rows_of: dict[str, list[int]] = defaultdict(list)
    for row, person in enumerate(people):
        rows_of[person].append(row)
    eligible = sorted(p for p, rows in rows_of.items() if len(rows) >= n_enroll + 1)
    ineligible = sorted(set(rows_of) - set(eligible))
    order = np.random.default_rng(seed).permutation(len(eligible))
    enrolled_names = {eligible[i] for i in order[: len(eligible) // 2]}

    store = IdentityStore(":memory:")
    gallery = Gallery(
        store,
        info,
        GalleryConfig(
            backend=backend, aggregation=aggregation, duplicate_policy=DuplicatePolicy.IGNORE
        ),
    )
    id_of: dict[str, str] = {}
    mated_rows: list[tuple[int, str]] = []  # (row, true identity id)
    enrollment_failures = 0
    unenrolled_rows: list[int] = [r for p in ineligible for r in rows_of[p]]
    for person in eligible:
        rows = rows_of[person]
        if person not in enrolled_names:
            unenrolled_rows += rows
            continue
        good = [r for r in rows if usable[r]]
        if len(good) < n_enroll:
            enrollment_failures += 1
            continue
        enroll_rows = good[:n_enroll]
        identity_id = f"p{len(id_of)}"
        id_of[person] = identity_id
        gallery.enroll(
            EmbeddingBatch(embeddings[enroll_rows], info), name=person, identity_id=identity_id
        )
        mated_rows += [(r, identity_id) for r in rows if r not in set(enroll_rows)]

    mated_scores: list[float] = []
    mated_correct: list[bool] = []
    for row, truth in mated_rows:
        if not usable[row]:
            mated_scores.append(MISS_SCORE)
            mated_correct.append(False)
            continue
        top = gallery.identify(EmbeddingBatch(embeddings[row : row + 1], info), k=1).candidates[0]
        mated_scores.append(top.score)
        mated_correct.append(top.identity_id == truth)
    nonmated_scores: list[float] = []
    nonmated_failed = 0
    for row in unenrolled_rows:
        if not usable[row]:
            nonmated_failed += 1
            continue
        top = gallery.identify(EmbeddingBatch(embeddings[row : row + 1], info), k=1).candidates[0]
        nonmated_scores.append(top.score)

    ms, mc = np.array(mated_scores), np.array(mated_correct)
    ns = np.array(nonmated_scores)
    curve = det_curve(ms, mc, ns)
    usable_mated = ms > MISS_SCORE
    result: dict[str, Any] = {
        "seed": seed,
        "n_enroll": n_enroll,
        "aggregation": aggregation.value,
        "backend": backend.value,
        "n_enrolled_identities": len(id_of),
        "enrolled_digest": hashlib.sha256("|".join(sorted(id_of)).encode()).hexdigest()[:12],
        "n_mated_probes": len(mated_rows),
        "n_mated_failed": int((~usable_mated).sum()),
        "n_nonmated_probes": int(ns.size),
        "n_nonmated_failed": nonmated_failed,
        "n_enrollment_failures": enrollment_failures,
        "rank1_closed_set": rank1_rate(mc[usable_mated]),  # usable mated probes only
        "rank1_failures_as_misses": rank1_rate(mc),
        "fnir_at_fpir": {},
    }
    for target in FPIR_TARGETS:
        resolvable = ns.size * target >= 1.0  # at least one non-mated probe must be allowed
        fnir, fpir, threshold = fnir_at_fpir(curve, target)
        result["fnir_at_fpir"][f"{target:g}"] = {
            "fnir": fnir,
            "fpir_achieved": fpir,
            "threshold": threshold,
            "resolvable": bool(resolvable),
        }
    step = max(1, len(curve.fpir) // 300)
    result["det"] = {"fpir": curve.fpir[::step].tolist(), "fnir": curve.fnir[::step].tolist()}
    return result


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_runs": len(runs)}
    for key in ("rank1_closed_set", "rank1_failures_as_misses"):
        mean, std, _ = mean_std([r[key] for r in runs])
        out[key] = {"mean": mean, "std": std}
    out["fnir_at_fpir"] = {}
    for target in FPIR_TARGETS:
        key = f"{target:g}"
        values = [r["fnir_at_fpir"][key]["fnir"] for r in runs]
        mean, std, _ = mean_std(values)
        out["fnir_at_fpir"][key] = {
            "mean": mean,
            "std": std,
            "resolvable_in_all_runs": all(r["fnir_at_fpir"][key]["resolvable"] for r in runs),
            "threshold_mean": float(np.mean([r["fnir_at_fpir"][key]["threshold"] for r in runs])),
        }
    for key in (
        "n_enrolled_identities",
        "n_mated_probes",
        "n_nonmated_probes",
        "n_mated_failed",
        "n_nonmated_failed",
        "n_enrollment_failures",
    ):
        out[key] = float(np.mean([r[key] for r in runs]))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--name", default="open_set")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cache = LfwCache.load(args.limit)
    info = build_embedder().info
    embeddings = load_embeddings(cache, info)
    usable = cache.ok & np.isfinite(embeddings).all(axis=1)
    people = cache.identities

    experiments: dict[str, list[dict[str, Any]]] = {}
    configs: list[tuple[str, dict[str, Any]]] = [
        ("enroll1_max", {"n_enroll": 1, "aggregation": Aggregation.MAX}),
        ("enroll3_max", {"n_enroll": 3, "aggregation": Aggregation.MAX}),
        ("enroll3_mean", {"n_enroll": 3, "aggregation": Aggregation.MEAN}),
        (
            "enroll1_max_hnsw",
            {"n_enroll": 1, "aggregation": Aggregation.MAX, "backend": Backend.HNSW},
        ),
    ]
    for label, kwargs in configs:
        log.info("experiment %s", label)
        experiments[label] = [
            run_once(embeddings, usable, people, info, seed=seed, **kwargs)
            for seed in range(args.seeds)
        ]
    result: dict[str, Any] = {
        "model": info.tag,
        "experiments": {
            label: {"summary": summarize(runs), "runs": runs} for label, runs in experiments.items()
        },
        "notes": (
            "Mated probes are other images of enrolled people; non-mated probes are people who "
            "are not enrolled. FNIR@FPIR needs enough non-mated probes: see resolvable flags."
        ),
    }
    path = write_result(args.name, result, {"limit": args.limit, "seeds": args.seeds})
    first = result["experiments"]["enroll1_max"]["summary"]
    log.info(
        "rank-1 %.4f, FNIR@FPIR=0.01 %.4f; wrote %s",
        first["rank1_closed_set"]["mean"],
        first["fnir_at_fpir"]["0.01"]["mean"],
        path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
