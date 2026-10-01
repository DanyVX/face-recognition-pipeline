"""Access to the cached LFW crops, per-image detection metadata, embeddings and pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from bench.prepare_lfw import STATUS_OK, cache_dir
from frp.config import load_settings
from frp.embed.base import ModelInfo
from frp.lfw import LfwPair, parse_pairs


def identity_of(relative_path: str) -> str:
    """``Name/Name_0001.jpg`` -> ``Name``."""
    return relative_path.split("/", 1)[0]


def image_key(name: str, index: int) -> str:
    return f"{name}/{name}_{index:04d}.jpg"


@dataclass(frozen=True, eq=False)
class LfwCache:
    """The output of ``bench.prepare_lfw``: crops plus metadata, row-aligned with ``images``."""

    root: Path
    images: list[str]  # relative paths, e.g. "Abel_Pacheco/Abel_Pacheco_0001.jpg"
    status: NDArray[np.uint8]
    det_score: NDArray[np.float32]
    fit_error: NDArray[np.float32]
    box: NDArray[np.float32]
    landmarks: NDArray[np.float32]
    n_faces: NDArray[np.uint8]
    aligned: NDArray[np.uint8]  # (N, 112, 112, 3) memmap
    manifest: dict[str, object]

    @classmethod
    def load(cls, limit: int | None = None, *, base: Path | None = None) -> LfwCache:
        root = (base / ("lfw_aligned_" + str(limit))) if (base and limit) else cache_dir(limit)
        if base and not limit:
            root = base / "lfw_aligned"
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        meta = np.load(root / "meta.npz")
        return cls(
            root=root,
            images=list(manifest["images"]),
            status=meta["status"],
            det_score=meta["det_score"],
            fit_error=meta["fit_error"],
            box=meta["box"],
            landmarks=meta["landmarks"],
            n_faces=meta["n_faces"],
            aligned=np.load(root / "aligned.npy", mmap_mode="r"),
            manifest=manifest,
        )

    def __len__(self) -> int:
        return len(self.images)

    @property
    def ok(self) -> NDArray[np.bool_]:
        return np.asarray(self.status == STATUS_OK)

    @property
    def index(self) -> dict[str, int]:
        return {p: i for i, p in enumerate(self.images)}

    @property
    def identities(self) -> list[str]:
        return [identity_of(p) for p in self.images]


@dataclass(frozen=True, eq=False)
class ResolvedPairs:
    """Pairs mapped to cache rows. ``row_*`` is -1 when the image is not in the cache at all."""

    pairs: list[LfwPair]
    row_a: NDArray[np.int64]
    row_b: NDArray[np.int64]
    same: NDArray[np.bool_]
    fold: NDArray[np.int64]

    def __len__(self) -> int:
        return len(self.pairs)

    def select(self, mask: NDArray[np.bool_]) -> ResolvedPairs:
        keep = np.flatnonzero(mask)
        return ResolvedPairs(
            pairs=[self.pairs[i] for i in keep],
            row_a=self.row_a[keep],
            row_b=self.row_b[keep],
            same=self.same[keep],
            fold=self.fold[keep],
        )


def load_pairs(path: Path | None = None) -> list[LfwPair]:
    pairs_path = path or (load_settings().data_dir / "lfw" / "pairs.txt")
    return parse_pairs(pairs_path.read_text(encoding="utf-8"))


def drop_unprepared(resolved: ResolvedPairs) -> ResolvedPairs:
    """Exclude (not count as failed) pairs whose images are outside a tiny-run subset."""
    return resolved.select((resolved.row_a >= 0) & (resolved.row_b >= 0))


def resolve_pairs(pairs: list[LfwPair], cache: LfwCache) -> ResolvedPairs:
    index = cache.index
    row_a = np.array([index.get(image_key(p.name_a, p.index_a), -1) for p in pairs], dtype=np.int64)
    row_b = np.array([index.get(image_key(p.name_b, p.index_b), -1) for p in pairs], dtype=np.int64)
    return ResolvedPairs(
        pairs=pairs,
        row_a=row_a,
        row_b=row_b,
        same=np.array([p.same for p in pairs], dtype=bool),
        fold=np.array([p.fold for p in pairs], dtype=np.int64),
    )


def embedding_path(cache: LfwCache, info: ModelInfo) -> Path:
    return cache.root / f"embeddings_{info.model_hash[:12]}_{info.preprocess_version}.npy"


def load_embeddings(cache: LfwCache, info: ModelInfo) -> NDArray[np.float32]:
    """(N, D) float32; rows of images without a usable face are NaN."""
    return np.asarray(np.load(embedding_path(cache, info)), dtype=np.float32)


def pair_scores(
    resolved: ResolvedPairs, embeddings: NDArray[np.float32], ok: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Cosine score per pair; NaN where either image has no usable face (a failed pair)."""
    scores = np.full(len(resolved), np.nan, dtype=np.float64)
    usable = (resolved.row_a >= 0) & (resolved.row_b >= 0)
    usable &= ok[np.where(usable, resolved.row_a, 0)] & ok[np.where(usable, resolved.row_b, 0)]
    a = embeddings[resolved.row_a[usable]]
    b = embeddings[resolved.row_b[usable]]
    scores[usable] = np.einsum("ij,ij->i", a, b)
    return scores
