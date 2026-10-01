"""Index scaling benchmark: Flat vs HNSW vs IVF-Flat at growing gallery sizes.

Usage: ``uv run python -m bench.index_scaling [--sizes 1000,10000,100000] [--tiny]``
Writes ``results/index_scaling.json`` (with hardware info and the git commit).

Gallery vectors are synthetic random unit vectors (allowed by the spec for scaling tests). Two query
sets are used because random data is a worst case for approximate search:

* ``planted``: each query is a noisy copy of a random gallery vector, with cosine to the original
  about ``--query-cos``. A true nearest neighbour exists (like a probe of an enrolled person).
* ``independent``: random unit vectors with no planted neighbour. All similarities are close, so
  "recall" here measures agreement on essentially arbitrary ties; it is reported for honesty, not
  as a realistic accuracy figure. Recall on *real* embeddings is measured separately in M6.

Timing should be taken on an otherwise idle machine; the result file records the hardware.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import time
from typing import Any

import faiss
import numpy as np
from numpy.typing import NDArray

from bench.common import write_result
from frp.embed import EmbeddingBatch, ModelInfo
from frp.index import Backend, GalleryIndex, IndexParams

log = logging.getLogger("index_scaling")


def unit_vectors(n: int, dim: int, rng: np.random.Generator) -> NDArray[np.float32]:
    v = rng.standard_normal((n, dim), dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def planted_queries(
    gallery: NDArray[np.float32], n: int, cos: float, rng: np.random.Generator
) -> NDArray[np.float32]:
    """Noisy copies of random gallery vectors with cosine to the source of about ``cos``."""
    dim = gallery.shape[1]
    sigma = math.sqrt(max(1.0 / (cos * cos) - 1.0, 0.0) / dim)
    base = gallery[rng.integers(0, gallery.shape[0], n)]
    q = base + sigma * rng.standard_normal((n, dim), dtype=np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return np.asarray(q, dtype=np.float32)


def percentiles(samples: list[float]) -> dict[str, float]:
    arr = np.asarray(samples) * 1000.0  # milliseconds
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def recall_at(approx: NDArray[np.int64], exact: NDArray[np.int64], k: int) -> float:
    hits = [
        len(set(a[:k].tolist()) & set(e[:k].tolist())) / k
        for a, e in zip(approx, exact, strict=True)
    ]
    return float(np.mean(hits))


def measure_backend(
    backend: Backend,
    params: IndexParams,
    info: ModelInfo,
    gallery: NDArray[np.float32],
    query_sets: dict[str, NDArray[np.float32]],
    exact_ids: dict[str, NDArray[np.int64]],
    *,
    latency_queries: int,
    sweep: list[dict[str, int]],
) -> dict[str, Any]:
    n = gallery.shape[0]
    index = GalleryIndex(info, backend, params)
    t0 = time.perf_counter()
    index.add(np.arange(n, dtype=np.int64), EmbeddingBatch(gallery, info))
    add_seconds = time.perf_counter() - t0
    warm = EmbeddingBatch(query_sets["planted"][:5], info)
    t0 = time.perf_counter()
    index.search(warm, 1)  # builds the FAISS structure
    build_seconds = time.perf_counter() - t0

    result: dict[str, Any] = {
        "backend": backend.value,
        "params": params.to_dict(),
        "add_seconds": add_seconds,
        "build_seconds": build_seconds,
        "master_bytes": int(gallery.nbytes),
        "faiss_serialized_bytes": int(faiss.serialize_index(index._index).size),
        "sweep": [],
    }
    for point in sweep:
        index.params = dataclasses.replace(index.params, **point)
        row: dict[str, Any] = {"search_params": point}
        for name, queries in query_sets.items():
            batch = EmbeddingBatch(queries, info)
            _, ids = index.search(batch, 10)
            row[f"recall@1_{name}"] = recall_at(ids, exact_ids[name], 1)
            row[f"recall@10_{name}"] = recall_at(ids, exact_ids[name], 10)
        singles = [
            EmbeddingBatch(query_sets["planted"][i : i + 1], info) for i in range(latency_queries)
        ]
        for single in singles[:20]:  # warm-up, excluded from the statistics
            index.search(single, 10)
        samples = []
        for single in singles:
            t = time.perf_counter()
            index.search(single, 10)
            samples.append(time.perf_counter() - t)
        row["latency_single_query_k10"] = percentiles(samples)
        big = EmbeddingBatch(query_sets["planted"], info)
        t = time.perf_counter()
        index.search(big, 10)
        row["throughput_qps_batch"] = float(len(big) / (time.perf_counter() - t))
        result["sweep"].append(row)

    # Cost of a deletion: the FAISS structure is rebuilt on the next search.
    victim = int(index.ids[n // 2])
    t = time.perf_counter()
    index.remove([victim])
    remove_seconds = time.perf_counter() - t
    t = time.perf_counter()
    index.search(warm, 1)
    result["delete_one_seconds"] = remove_seconds
    result["first_search_after_delete_seconds"] = time.perf_counter() - t
    return result


def sweep_for(backend: Backend) -> list[dict[str, int]]:
    if backend is Backend.HNSW:
        return [{"hnsw_ef_search": v} for v in (16, 64, 256)]
    if backend is Backend.IVF:
        return [{"ivf_nprobe": v} for v in (1, 8, 32)]
    return [{}]


def run_benchmark(
    sizes: list[int],
    *,
    dim: int = 512,
    n_queries: int = 200,
    query_cos: float = 0.6,
    latency_queries: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    info = ModelInfo("synthetic/random-unit", "0" * 64, "none", dim)
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"dim": dim, "sizes": {}}
    for n in sizes:
        log.info("size %d", n)
        gallery = unit_vectors(n, dim, rng)
        query_sets = {
            "planted": planted_queries(gallery, n_queries, query_cos, rng),
            "independent": unit_vectors(n_queries, dim, rng),
        }
        reference = GalleryIndex(info, Backend.FLAT)
        reference.add(np.arange(n, dtype=np.int64), EmbeddingBatch(gallery, info))
        exact_ids = {
            name: reference.search_exact(EmbeddingBatch(q, info), 10)[1]
            for name, q in query_sets.items()
        }
        per_backend = []
        for backend in (Backend.FLAT, Backend.HNSW, Backend.IVF):
            log.info("  %s", backend.value)
            per_backend.append(
                measure_backend(
                    backend,
                    IndexParams(seed=seed),
                    info,
                    gallery,
                    query_sets,
                    exact_ids,
                    latency_queries=min(latency_queries, n_queries),
                    sweep=sweep_for(backend),
                )
            )
        out["sizes"][str(n)] = per_backend
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1000,10000,100000")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--query-cos", type=float, default=0.6)
    parser.add_argument("--tiny", action="store_true", help="CI smoke run (small, fast)")
    parser.add_argument("--name", default="index_scaling")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sizes = [200, 600] if args.tiny else [int(s) for s in args.sizes.split(",")]
    dim = 32 if args.tiny else args.dim
    queries = 40 if args.tiny else args.queries
    params = {
        "sizes": sizes,
        "dim": dim,
        "queries": queries,
        "query_cos": args.query_cos,
        "faiss_version": faiss.__version__,
        "faiss_omp_threads": faiss.omp_get_max_threads(),
        "tiny": args.tiny,
    }
    result = run_benchmark(sizes, dim=dim, n_queries=queries, query_cos=args.query_cos)
    path = write_result(args.name, result, params)
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
