"""fp16 vs fp32 embeddings: agreement, decision flips and CPU speed.

Usage: ``uv run python -m bench.precision [--limit N] [--name precision]``

The fp16 model is produced from the fp32 ONNX file with ``onnxconverter-common`` (weights and
internal activations in float16, float32 inputs and outputs kept) and cached under
``DATA_DIR/cache``. Everything is measured on this machine's CPU through ONNX Runtime; GPU fp16 is
not measured here. Comparison:
* per-image cosine similarity between the fp32 and fp16 embedding of the same crop;
* the LFW 10-fold verification result for each precision (same protocol and threshold rule);
* how many verification pairs flip accept/reject at the fp32 mean-fold threshold;
* warm per-batch inference time for both models.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from numpy.typing import NDArray
from onnxconverter_common import float16

from bench.common import write_result
from bench.embed_lfw import compute_embeddings
from bench.lfw_data import (
    LfwCache,
    drop_unprepared,
    embedding_path,
    load_embeddings,
    load_pairs,
    pair_scores,
    resolve_pairs,
)
from bench.lfw_verification import verification_report
from frp.config import load_settings
from frp.embed import ArcFaceEmbedder
from frp.factory import build_embedder
from frp.models import MODELS

log = logging.getLogger("precision")


def convert_to_fp16(source: Path, destination: Path) -> Path:
    """Write a float16 copy of ``source`` (float32 inputs/outputs preserved)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(source))
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(converted, str(destination))
    return destination


def agreement(
    a: NDArray[np.float32], b: NDArray[np.float32], usable: NDArray[np.bool_]
) -> dict[str, float]:
    """Per-image cosine and element-wise differences between two embedding sets."""
    x, y = a[usable], b[usable]
    cosine = np.einsum("ij,ij->i", x, y)
    diff = np.abs(x - y)
    return {
        "n": float(cosine.size),
        "cosine_min": float(cosine.min()),
        "cosine_p1": float(np.percentile(cosine, 1)),
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(np.median(cosine)),
        "max_abs_element_diff": float(diff.max()),
        "mean_abs_element_diff": float(diff.mean()),
    }


def decision_flips(
    scores_a: NDArray[np.float64], scores_b: NDArray[np.float64], threshold: float
) -> dict[str, float]:
    valid = np.isfinite(scores_a) & np.isfinite(scores_b)
    flips = (scores_a[valid] >= threshold) != (scores_b[valid] >= threshold)
    return {
        "pairs_compared": float(valid.sum()),
        "pairs_flipped": float(flips.sum()),
        "max_abs_score_diff": float(np.abs(scores_a[valid] - scores_b[valid]).max()),
    }


def time_embedder(
    embedder: ArcFaceEmbedder, crops: NDArray[np.uint8], batch: int, repeats: int = 5
) -> dict[str, float]:
    """Warm timing: one warm-up call excluded, then ``repeats`` calls on the same batch."""
    chunk = crops[:batch]
    embedder.embed(chunk)
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        embedder.embed(chunk)
        samples.append(time.perf_counter() - t)
    arr = np.asarray(samples)
    return {
        "batch": float(batch),
        "median_ms_per_batch": float(np.median(arr) * 1000),
        "median_ms_per_image": float(np.median(arr) * 1000 / batch),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--name", default="precision")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    settings = load_settings()
    cache = LfwCache.load(args.limit)
    fp32 = build_embedder()
    fp32_path = settings.model_dir / MODELS["arcface_r50"].filename
    fp16_path = convert_to_fp16(fp32_path, settings.data_dir / "cache" / "w600k_r50_fp16.onnx")
    fp16 = ArcFaceEmbedder.from_file(
        fp16_path,
        providers=settings.provider_list,
        intra_op_threads=settings.onnx_threads,
        model_id="insightface-buffalo_l/w600k_r50-fp16",
        max_batch=settings.embed_max_batch,
    )
    emb32 = load_embeddings(cache, fp32.info)
    if embedding_path(cache, fp16.info).is_file():
        emb16 = load_embeddings(cache, fp16.info)
    else:
        log.info("embedding %d crops with fp16", int(cache.ok.sum()))
        emb16 = compute_embeddings(cache, fp16)
    usable = cache.ok & np.isfinite(emb32).all(axis=1) & np.isfinite(emb16).all(axis=1)

    resolved = resolve_pairs(load_pairs(), cache)
    if args.limit:
        resolved = drop_unprepared(resolved)
    s32 = pair_scores(resolved, emb32, cache.ok)
    s16 = pair_scores(resolved, emb16, cache.ok)
    rep32 = verification_report(s32, resolved.same, resolved.fold)
    rep16 = verification_report(s16, resolved.same, resolved.fold)
    crops = np.asarray(cache.aligned[np.flatnonzero(cache.ok)[:64]])
    result: dict[str, Any] = {
        "fp32_model": fp32.info.tag,
        "fp16_model": fp16.info.tag,
        "fp16_conversion": (
            "onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True)"
        ),
        "model_bytes": {"fp32": fp32_path.stat().st_size, "fp16": fp16_path.stat().st_size},
        "embedding_agreement": agreement(emb32, emb16, usable),
        "verification": {
            "fp32": {k: rep32[k] for k in ("accuracy_mean", "accuracy_std")}
            | {"eer": rep32["pooled"]["eer"]},
            "fp16": {k: rep16[k] for k in ("accuracy_mean", "accuracy_std")}
            | {"eer": rep16["pooled"]["eer"]},
            "decision_flips_at_fp32_mean_threshold": decision_flips(
                s32, s16, rep32["mean_fold_threshold"]
            ),
        },
        "cpu_timing": {
            "fp32": {str(b): time_embedder(fp32, crops, b) for b in (1, 16)},
            "fp16": {str(b): time_embedder(fp16, crops, b) for b in (1, 16)},
        },
        "providers": fp32.providers,
        "scope": "CPU only; GPU fp16 not measured",
    }
    write_result(args.name, result, {"limit": args.limit})
    log.info(
        "cosine fp32 vs fp16: min %.6f mean %.6f; accuracy fp32 %.4f fp16 %.4f",
        result["embedding_agreement"]["cosine_min"],
        result["embedding_agreement"]["cosine_mean"],
        rep32["accuracy_mean"],
        rep16["accuracy_mean"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
