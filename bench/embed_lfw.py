"""Stage 2 of the LFW benchmarks: embed every cached crop (resumable).

Usage: ``uv run python -m bench.embed_lfw [--limit N] [--model-path PATH --model-id ID]``

Writes ``embeddings_<model-hash>_<preprocess>.npy`` (N x 512 float32, NaN rows for images without
a usable face) next to the crops. Embeddings are biometric data: the file is gitignored with the
rest of ``data/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from bench.common import write_json_atomic
from bench.lfw_data import LfwCache, embedding_path
from frp.config import load_settings
from frp.embed import ArcFaceEmbedder
from frp.factory import build_embedder
from frp.models import MODELS

log = logging.getLogger("embed_lfw")


def compute_embeddings(
    cache: LfwCache,
    embedder: ArcFaceEmbedder,
    *,
    batch_size: int = 32,
    save_every: int = 1000,
) -> NDArray[np.float32]:
    """Embed all usable crops; resumes from a partial file if one exists."""
    out_path = embedding_path(cache, embedder.info)
    partial = out_path.with_suffix(".partial.npy")
    progress = out_path.with_suffix(".progress.json")
    n, dim = len(cache), embedder.info.dim
    rows = np.flatnonzero(cache.ok)
    vectors = np.full((n, dim), np.nan, dtype=np.float32)
    done = 0
    if partial.is_file() and progress.is_file():
        saved = np.load(partial)
        if saved.shape == vectors.shape:
            vectors = saved
            done = int(json.loads(progress.read_text(encoding="utf-8"))["done"])
            log.info("resuming at %d/%d", done, len(rows))
    t0 = time.perf_counter()
    since_save = 0
    for start in range(done, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        vectors[chunk] = embedder.embed(np.asarray(cache.aligned[chunk])).vectors
        since_save += len(chunk)
        finished = start + len(chunk)
        if since_save >= save_every or finished == len(rows):
            np.save(partial, vectors)
            write_json_atomic(progress, {"done": finished})
            since_save = 0
            rate = (finished - done) / max(time.perf_counter() - t0, 1e-9)
            log.info("%d/%d  %.1f img/s", finished, len(rows), rate)
    np.save(out_path, vectors)
    partial.unlink(missing_ok=True)
    progress.unlink(missing_ok=True)
    return vectors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-path", type=Path, default=None, help="alternative ONNX file")
    parser.add_argument("--model-id", default=None, help="model id to tag alternative embeddings")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cache = LfwCache.load(args.limit)
    if args.model_path is not None:
        settings = load_settings()
        embedder = ArcFaceEmbedder.from_file(
            args.model_path,
            providers=settings.provider_list,
            intra_op_threads=settings.onnx_threads,
            model_id=args.model_id or f"custom/{args.model_path.stem}",
            max_batch=args.batch_size,
        )
    else:
        embedder = build_embedder()
        if embedder.info.model_hash != MODELS["arcface_r50"].sha256:
            raise SystemExit(
                "model hash does not match the manifest; run scripts/download_models.py"
            )
    log.info("embedding %d crops with %s", int(cache.ok.sum()), embedder.info.tag)
    compute_embeddings(cache, embedder, batch_size=args.batch_size)
    log.info("wrote %s", embedding_path(cache, embedder.info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
