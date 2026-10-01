"""Robustness slices: verif
    drop_unprepared,ication accuracy as synthetic degradation severity grows.

Usage: ``uv run python -m bench.robustness [--pairs-per-class 100] [--name robustness]``

Design (small and honest):
* Two disjoint samples of LFW pairs, both with usable clean images: a **validation** sample drawn
  from folds 0-1 and a **test** sample drawn from folds 2-9, ``--pairs-per-class`` genuine and
  impostor pairs each. Samples are seeded and recorded.
* The enrollment image of a pair stays clean; the **probe** image is degraded (blur, JPEG quality
  10-90, brightness shift, downscale, rotation +/-30 degrees, horizontal flip, partial occlusion).
  The detector, aligner and embedder run on the degraded image end to end.
* One decision threshold is used everywhere: chosen on validation-fold *clean* pairs only
  (``best_threshold`` over folds 0-1), never on degraded or test data.
* Accuracy counts a failed detection as an error. Wilson 95% intervals are given; with a few
  hundred pairs per sample, differences of a few points are not statistically significant.
* Per-sample quality features of every degraded probe are saved (gitignored cache) so the quality
  gates can be calibrated against degradations (see ``bench.calibrate_quality``).
"""

from __future__ import annotations

import argparse
import logging
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bench.common import write_result
from bench.degrade import Degradation, standard_degradations
from bench.lfw_data import (
    LfwCache,
    drop_unprepared,
    load_embeddings,
    load_pairs,
    pair_scores,
    resolve_pairs,
)
from bench.metrics import best_threshold, wilson_interval
from frp.config import load_settings
from frp.detect import FacePolicy
from frp.embed import ArcFaceEmbedder
from frp.errors import FrpError
from frp.factory import build_detector, build_embedder
from frp.imageio import LoadedImage, load_image
from frp.pipeline import FacePipeline
from frp.quality import GateContext, extract_features
from frp.quality.features import QualityFeatures

log = logging.getLogger("robustness")

FEATURE_NAMES = (
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
STATUS_OK, STATUS_NO_FACE, STATUS_OTHER = 0, 1, 2


@dataclass(frozen=True)
class Sample:
    """One pair for the robustness slice."""

    pair_index: int
    same: bool
    row_a: int  # clean enrollment image (cache row)
    row_b: int  # probe image that will be degraded (cache row)
    split: str  # "val" or "test"


def choose_samples(
    same: NDArray[np.bool_],
    fold: NDArray[np.int64],
    usable_pair: NDArray[np.bool_],
    row_a: NDArray[np.int64],
    row_b: NDArray[np.int64],
    per_class: int,
    seed: int,
) -> list[Sample]:
    """Seeded sample of ``per_class`` genuine and impostor pairs for each split."""
    rng = np.random.default_rng(seed)
    samples: list[Sample] = []
    for split, folds in (("val", (0, 1)), ("test", tuple(range(2, 10)))):
        in_split = np.isin(fold, folds) & usable_pair
        for label in (True, False):
            candidates = np.flatnonzero(in_split & (same == label))
            take = np.sort(
                rng.choice(candidates, size=min(per_class, candidates.size), replace=False)
            )
            samples += [
                Sample(int(i), bool(label), int(row_a[i]), int(row_b[i]), split) for i in take
            ]
    return samples


def run_setting(
    degradation: Degradation,
    probes: list[NDArray[np.uint8]],
    clean_embeddings_a: NDArray[np.float32],
    pipeline: FacePipeline,
    embedder: ArcFaceEmbedder,
    seed: int,
) -> dict[str, NDArray[Any]]:
    """Degrade every probe, run the full pipeline and score against the clean enrollment."""
    n = len(probes)
    scores = np.full(n, np.nan, dtype=np.float64)
    status = np.zeros(n, dtype=np.uint8)
    features = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    salt = zlib.crc32(degradation.name.encode())
    for i, probe in enumerate(probes):
        rng = np.random.default_rng((seed, salt, i))
        degraded = degradation.apply(probe, rng)
        loaded = LoadedImage(
            array=degraded, original_size=(degraded.shape[1], degraded.shape[0]), format="SYNTH"
        )
        try:
            prepared = pipeline.prepare(loaded)
        except FrpError as exc:
            status[i] = STATUS_NO_FACE if exc.code == "NO_FACE" else STATUS_OTHER
            continue
        vector = embedder.embed(prepared.aligned.image[None]).vectors[0]
        scores[i] = float(clean_embeddings_a[i] @ vector)
        feats: QualityFeatures = extract_features(
            GateContext(prepared.face, prepared.aligned, prepared.image_shape)
        )
        features[i] = [float(v) for v in feats.as_dict().values()]
    return {"scores": scores, "status": status, "features": features}


def summarize_setting(
    scores: NDArray[np.float64],
    same: NDArray[np.bool_],
    status: NDArray[np.uint8],
    threshold: float,
) -> dict[str, Any]:
    valid = np.isfinite(scores)
    correct = int((((scores >= threshold) == same) & valid).sum())  # failures are errors
    lo, hi = wilson_interval(correct, scores.size)
    return {
        "n": int(scores.size),
        "accuracy": correct / scores.size,
        "accuracy_ci95": [lo, hi],
        "n_failed": int((~valid).sum()),
        "failure_rate": float((~valid).mean()),
        "no_face": int((status == STATUS_NO_FACE).sum()),
        "mean_score_genuine": float(np.nanmean(scores[same])) if (valid & same).any() else None,
        "mean_score_impostor": float(np.nanmean(scores[~same])) if (valid & ~same).any() else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--name", default="robustness")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    settings = load_settings()
    cache = LfwCache.load(args.limit)
    embedder = build_embedder()
    embeddings = load_embeddings(cache, embedder.info)
    pairs = load_pairs()
    resolved = resolve_pairs(pairs, cache)
    if args.limit:
        resolved = drop_unprepared(resolved)
    scores = pair_scores(resolved, embeddings, cache.ok)

    val_mask = np.isin(resolved.fold, (0, 1)) & np.isfinite(scores)
    threshold, val_clean_acc = best_threshold(scores[val_mask], resolved.same[val_mask])
    log.info(
        "threshold %.4f from %d clean validation pairs (accuracy %.4f)",
        threshold,
        int(val_mask.sum()),
        val_clean_acc,
    )

    usable_pair = np.isfinite(scores)
    samples = choose_samples(
        resolved.same,
        resolved.fold,
        usable_pair,
        resolved.row_a,
        resolved.row_b,
        args.pairs_per_class,
        args.seed,
    )
    lfw_root = settings.data_dir / "lfw" / "lfw"
    pipeline = FacePipeline(build_detector(settings), policy=FacePolicy.CENTER)
    degradations = standard_degradations()

    result: dict[str, Any] = {
        "threshold": threshold,
        "threshold_source": "best-accuracy threshold on clean validation pairs (folds 0-1)",
        "clean_validation_accuracy": val_clean_acc,
        "samples": {s: sum(1 for x in samples if x.split == s) for s in ("val", "test")},
        "settings": {},
    }
    arrays: dict[str, NDArray[Any]] = {}
    for split in ("val", "test"):
        subset = [s for s in samples if s.split == split]
        same = np.array([s.same for s in subset])
        probes = [load_image((lfw_root / cache.images[s.row_b]).read_bytes()).array for s in subset]
        emb_a = embeddings[[s.row_a for s in subset]]
        clean_scores = np.einsum("ij,ij->i", emb_a, embeddings[[s.row_b for s in subset]]).astype(
            np.float64
        )
        clean_status = np.zeros(len(subset), dtype=np.uint8)
        result["settings"].setdefault("clean", {})[split] = summarize_setting(
            clean_scores, same, clean_status, threshold
        )
        arrays[f"{split}_same"] = same
        arrays[f"{split}_row_a"] = np.array([s.row_a for s in subset], dtype=np.int64)
        arrays[f"{split}_row_b"] = np.array([s.row_b for s in subset], dtype=np.int64)
        arrays[f"{split}_pair_index"] = np.array([s.pair_index for s in subset], dtype=np.int64)
        arrays[f"{split}_clean_scores"] = clean_scores
        for degradation in degradations:
            log.info("%s %s", split, degradation.name)
            out = run_setting(degradation, probes, emb_a, pipeline, embedder, args.seed)
            entry = result["settings"].setdefault(
                degradation.name, {"family": degradation.family, "level": degradation.level}
            )
            entry[split] = summarize_setting(out["scores"], same, out["status"], threshold)
            key = f"{split}_{degradation.name}"
            arrays[f"{key}_scores"] = out["scores"]
            arrays[f"{key}_features"] = out["features"]
    cache_path = settings.data_dir / "cache" / f"{args.name}_samples.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        **arrays,
        "feature_names": np.array(FEATURE_NAMES),
        "threshold": np.array(threshold),
    }
    np.savez_compressed(cache_path, **payload)
    path = write_result(
        args.name,
        result,
        {"pairs_per_class": args.pairs_per_class, "seed": args.seed, "limit": args.limit},
    )
    log.info("wrote %s and %s", path, Path(cache_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
