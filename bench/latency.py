"""Per-stage latency: decode, detect, align, embed, search (p50/p95/p99), cold vs warm.

Usage: ``uv run python -m bench.latency [--images 200] [--providers CPUExecutionProvider]``
Writes ``results/<name>.json`` (default ``latency``) with hardware, providers and thread settings.

* **Cold**: model-load time and the very first inference after loading (this process has already
  imported the libraries, so interpreter/import time is not included).
* **Warm**: ``--warmup`` calls are executed and discarded, then every sample is timed.
* Stages are timed separately per image so that each has its own distribution:
  ``decode`` (bytes -> RGB array), ``detect``, ``align``, ``embed`` for batch sizes 1/4/16/32 (time
  per image *and* per batch), and ``search`` against synthetic Flat galleries.
* Two input scenarios: small LFW originals (250x250) and a 1920x1080 composite (a realistic phone
  or webcam upload) built from the portrait fixture.
* GPU numbers are TBD unless this script is run with a CUDA provider on a machine that has one
  (see docs/COLAB.md); the result file records which providers were actually used.
"""

from __future__ import annotations

import argparse
import io
import logging
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from bench.common import ROOT, write_result
from bench.index_scaling import unit_vectors
from bench.lfw_data import LfwCache
from frp.align import align_face
from frp.config import Settings, load_settings
from frp.detect import FacePolicy, ScrfdDetector, select_faces
from frp.embed import ArcFaceEmbedder, EmbeddingBatch, ModelInfo
from frp.errors import FrpError
from frp.factory import build_detector, build_embedder
from frp.imageio import load_image
from frp.index import Backend, GalleryIndex

log = logging.getLogger("latency")


def percentile_stats(samples_seconds: list[float]) -> dict[str, float]:
    ms = np.asarray(samples_seconds) * 1000.0
    return {
        "n": float(ms.size),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "mean_ms": float(ms.mean()),
        "max_ms": float(ms.max()),
    }


def time_each(fn: Callable[[Any], Any], items: list[Any], warmup: int) -> list[float]:
    """Time ``fn(item)`` for every item; the first ``warmup`` calls (on the first items) are
    executed but not recorded."""
    for item in items[:warmup]:
        fn(item)
    samples: list[float] = []
    for item in items:
        t = time.perf_counter()
        fn(item)
        samples.append(time.perf_counter() - t)
    return samples


def stage_timings(
    images: list[bytes],
    detector: ScrfdDetector,
    embedder: ArcFaceEmbedder,
    warmup: int,
) -> dict[str, Any]:
    """decode / detect / align / embed(1) distributions for one scenario."""
    decoded = [load_image(b) for b in images]
    detections = []
    for loaded in decoded:
        faces = detector.detect(loaded.array)
        detections.append(select_faces(faces, FacePolicy.LARGEST)[0] if faces else None)
    usable = [(d, f) for d, f in zip(decoded, detections, strict=True) if f is not None]
    crops = [align_face(d.array, f.landmarks).image for d, f in usable if _alignable(d, f)]
    return {
        "n_images": len(images),
        "n_with_face": len(usable),
        "decode": percentile_stats(time_each(load_image, images, warmup)),
        "detect": percentile_stats(time_each(lambda d: detector.detect(d.array), decoded, warmup)),
        "align": percentile_stats(
            time_each(lambda pair: align_face(pair[0].array, pair[1].landmarks), usable, warmup)
            if usable
            else [0.0]
        ),
        "embed_single": percentile_stats(
            time_each(lambda c: embedder.embed(c[None]), crops, warmup) if crops else [0.0]
        ),
    }


def _alignable(loaded: Any, face: Any) -> bool:
    try:
        align_face(loaded.array, face.landmarks)
    except FrpError:
        return False
    return True


def embed_batches(
    embedder: ArcFaceEmbedder,
    crops: NDArray[np.uint8],
    batch_sizes: list[int],
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for size in batch_sizes:
        batch = np.ascontiguousarray(np.resize(crops, (size, *crops.shape[1:])))
        for _ in range(warmup):
            embedder.embed(batch)
        samples = []
        for _ in range(repeats):
            t = time.perf_counter()
            embedder.embed(batch)
            samples.append(time.perf_counter() - t)
        stats = percentile_stats(samples)
        stats["per_image_p50_ms"] = stats["p50_ms"] / size
        out[str(size)] = stats
    return out


def search_timings(
    info: ModelInfo, sizes: list[int], n_queries: int, warmup: int, seed: int = 0
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for n in sizes:
        index = GalleryIndex(info, Backend.FLAT)
        index.add(
            np.arange(n, dtype=np.int64), EmbeddingBatch(unit_vectors(n, info.dim, rng), info)
        )
        queries = [EmbeddingBatch(unit_vectors(1, info.dim, rng), info) for _ in range(n_queries)]
        index.search(queries[0], 1)  # build the structure (not part of the timing)
        search = partial(index.search, k=5)  # binds this iteration's index
        out[str(n)] = percentile_stats(time_each(search, queries, warmup))
    return out


def make_1080p(portrait: Path) -> bytes:
    """A 1920x1080 JPEG with the portrait scaled up in the middle (a realistic upload size)."""
    canvas = Image.new("RGB", (1920, 1080), (70, 90, 110))
    face = Image.open(portrait).convert("RGB").resize((1000, 1000), Image.Resampling.LANCZOS)
    canvas.paste(face, (460, 40))
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def cold_start(settings: Settings) -> dict[str, Any]:
    """Model-load time and first-inference latency of fresh sessions."""
    out: dict[str, Any] = {}
    t = time.perf_counter()
    detector = build_detector(settings)
    out["detector_load_s"] = time.perf_counter() - t
    t = time.perf_counter()
    embedder = build_embedder(settings)
    out["embedder_load_s"] = time.perf_counter() - t
    image = np.random.default_rng(0).integers(0, 255, (250, 250, 3), dtype=np.uint8)
    t = time.perf_counter()
    detector.detect(image)
    out["detector_first_inference_ms"] = (time.perf_counter() - t) * 1000
    crop = np.zeros((1, 112, 112, 3), dtype=np.uint8)
    t = time.perf_counter()
    embedder.embed(crop)
    out["embedder_first_inference_ms"] = (time.perf_counter() - t) * 1000
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--providers", default=None, help="comma-separated ONNX providers")
    parser.add_argument("--name", default="latency")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    settings = load_settings()
    if args.providers:
        settings = settings.model_copy(update={"onnx_providers": args.providers})
    log.info("cold start")
    cold = cold_start(settings)
    detector, embedder = build_detector(settings), build_embedder(settings)

    cache = LfwCache.load(args.limit)
    lfw_root = settings.data_dir / "lfw" / "lfw"
    ok_rows = np.flatnonzero(cache.ok)
    rng = np.random.default_rng(0)
    chosen = np.sort(rng.choice(ok_rows, size=min(args.images, ok_rows.size), replace=False))
    lfw_bytes = [(lfw_root / cache.images[r]).read_bytes() for r in chosen]
    big = make_1080p(ROOT / "tests" / "fixtures" / "astronaut.jpg")

    log.info("stage timings (LFW 250x250)")
    result: dict[str, Any] = {
        "providers": embedder.providers,
        "detector_providers": detector._session.get_providers()
        if hasattr(detector._session, "get_providers")
        else [],
        "intra_op_threads": settings.onnx_threads,
        "cold_start": cold,
        "warmup_calls_excluded": args.warmup,
        "scenarios": {
            "lfw_250x250": stage_timings(lfw_bytes, detector, embedder, args.warmup),
        },
    }
    log.info("stage timings (1080p)")
    result["scenarios"]["upload_1920x1080"] = stage_timings(
        [big] * 60, detector, embedder, args.warmup
    )
    crops = np.asarray(cache.aligned[chosen[:32]])
    log.info("embed batches")
    result["embed_batches"] = embed_batches(embedder, crops, [1, 4, 16, 32], repeats=15, warmup=3)
    log.info("search")
    result["search_flat"] = search_timings(embedder.info, [1000, 10000, 100000], 100, 10)
    result["gpu"] = (
        "measured on CUDA (see providers)"
        if any("CUDA" in p for p in embedder.providers)
        else "TBD (not yet measured): no CUDA provider on this machine"
    )
    path = write_result(
        args.name,
        result,
        {"images": args.images, "warmup": args.warmup, "providers": args.providers},
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
