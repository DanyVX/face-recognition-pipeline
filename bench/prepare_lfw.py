"""Stage 1 of the LFW benchmarks: detect and align every image once, cache the 112x112 crops.

Usage: ``uv run python -m bench.prepare_lfw [--limit N]``  (resumable; Ctrl+C is safe)

Output directory ``DATA_DIR/cache/lfw_aligned[_N]/`` (gitignored; the crops are face images):
  aligned.npy   (N, 112, 112, 3) uint8 memmap, row i = crop of images[i] (zeros if no crop)
  meta.npz      per-image detection/alignment fields (status codes below)
  manifest.json image list, parameters, status counts, environment, git commit

Status codes: 0 OK, 1 NO_FACE, 2 ALIGNMENT_FAILED, 3 LOAD_ERROR.
Failures are recorded, never dropped: downstream benchmarks count them as errors.
Multi-face policy: LFW photos are centered on the labeled person, so the face closest to the
image center is used (documented in DATA.md and docs/RESULTS.md).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Final

import numpy as np

from bench.common import ROOT, write_json_atomic
from frp.config import load_settings
from frp.detect import FacePolicy
from frp.envinfo import collect_env
from frp.errors import FrpError, InvalidImageError, InvalidLandmarksError, NoFaceError
from frp.factory import build_detector
from frp.imageio import load_image
from frp.lfw import identities
from frp.pipeline import FacePipeline

STATUS_OK: Final = 0
STATUS_NO_FACE: Final = 1
STATUS_ALIGNMENT_FAILED: Final = 2
STATUS_LOAD_ERROR: Final = 3
STATUS_NAMES: Final = {0: "OK", 1: "NO_FACE", 2: "ALIGNMENT_FAILED", 3: "LOAD_ERROR"}

log = logging.getLogger("prepare_lfw")


def cache_dir(limit: int | None) -> Path:
    base = load_settings().data_dir / "cache"
    return base / (f"lfw_aligned_{limit}" if limit else "lfw_aligned")


def list_images(lfw_root: Path, limit: int | None) -> list[Path]:
    paths = [p for images in identities(lfw_root).values() for p in images]
    if limit:
        step = max(1, len(paths) // limit)  # spread over identities, deterministic
        paths = paths[::step][:limit]
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="process about N images (tiny run)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    settings = load_settings()
    lfw_root = settings.data_dir / "lfw" / "lfw"
    paths = list_images(lfw_root, args.limit)
    out = cache_dir(args.limit)
    out.mkdir(parents=True, exist_ok=True)
    n = len(paths)
    log.info("%d images -> %s", n, out)

    aligned_path = out / "aligned.npy"
    progress_path = out / "progress.json"
    done = 0
    if progress_path.is_file() and aligned_path.is_file():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if state.get("n") == n:
            done = int(state["done"])
    mode = "r+" if done else "w+"
    aligned = np.lib.format.open_memmap(
        aligned_path, mode=mode, dtype=np.uint8, shape=(n, 112, 112, 3)
    )
    meta_path = out / "meta_partial.npz"
    fields: dict[str, Any] = {
        "status": np.zeros(n, np.uint8),
        "n_faces": np.zeros(n, np.uint8),
        "det_score": np.zeros(n, np.float32),
        "fit_error": np.full(n, np.nan, np.float32),
        "box": np.zeros((n, 4), np.float32),
        "landmarks": np.zeros((n, 5, 2), np.float32),
    }
    if done and meta_path.is_file():
        saved = np.load(meta_path)
        for key in fields:
            fields[key] = saved[key]

    pipeline = FacePipeline(build_detector(), policy=FacePolicy.CENTER)
    started, t_last = time.perf_counter(), time.perf_counter()
    for i in range(done, n):
        try:
            loaded = load_image(paths[i].read_bytes())
            prepared = pipeline.prepare(loaded)
        except NoFaceError:
            fields["status"][i] = STATUS_NO_FACE
        except InvalidLandmarksError:
            fields["status"][i] = STATUS_ALIGNMENT_FAILED
        except (InvalidImageError, OSError):
            fields["status"][i] = STATUS_LOAD_ERROR
        except FrpError:  # any other expected failure is still recorded, not dropped
            fields["status"][i] = STATUS_ALIGNMENT_FAILED
        else:
            aligned[i] = prepared.aligned.image
            fields["status"][i] = STATUS_OK
            fields["n_faces"][i] = min(prepared.n_faces, 255)
            fields["det_score"][i] = prepared.face.score
            fields["fit_error"][i] = prepared.aligned.fit_error
            fields["box"][i] = prepared.face.box
            fields["landmarks"][i] = prepared.face.landmarks
        if (i + 1) % 250 == 0 or i + 1 == n:
            aligned.flush()
            np.savez(meta_path, **fields)
            write_json_atomic(progress_path, {"n": n, "done": i + 1})
            now = time.perf_counter()
            rate = 250 / max(now - t_last, 1e-9)
            t_last = now
            remaining = (n - i - 1) / max(rate, 1e-9)
            log.info("%d/%d  %.1f img/s  ~%.0f min left", i + 1, n, rate, remaining / 60)
    elapsed = time.perf_counter() - started

    aligned.flush()
    np.savez(out / "meta.npz", **fields)
    counts = {STATUS_NAMES[c]: int((fields["status"] == c).sum()) for c in STATUS_NAMES}
    write_json_atomic(
        out / "manifest.json",
        {
            "images": [p.relative_to(lfw_root).as_posix() for p in paths],
            "status_counts": counts,
            "policy": FacePolicy.CENTER.value,
            "detector": {
                "min_confidence": settings.det_min_confidence,
                "min_face_px": settings.min_face_px,
                "max_faces": settings.max_faces,
            },
            "limit": args.limit,
            "seconds_this_run": round(elapsed, 1),
            "env": collect_env(ROOT),
        },
    )
    log.info("done: %s", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
