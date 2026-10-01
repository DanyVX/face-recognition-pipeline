"""Vector index over (int64 id -> unit vector) with Flat, HNSW and IVF-Flat FAISS backends.

The master copy of the vectors lives in numpy arrays owned by this class; the FAISS structure is
a search accelerator built from it. That makes deletion simple and certain (the vector is dropped
from the master and the FAISS structure is rebuilt lazily on the next search), which matters
because FAISS HNSW cannot delete in place. The trade-off is that deletions on large HNSW/IVF
indexes pay a rebuild; measured numbers are in docs/RESULTS.md.

Scores are inner products of unit vectors, i.e. cosine similarity. All public methods are
serialized by one re-entrant lock, so concurrent enroll/search/delete is safe; searches inside
FAISS still use OpenMP threads.
"""

from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import faiss
import numpy as np
from numpy.typing import NDArray

from frp.embed.base import EmbeddingBatch, ModelInfo, check_unit_norm
from frp.errors import DuplicateIdError, VectorIndexError
from frp.index.format import (
    IndexSnapshot,
    PayloadCipher,
    load_snapshot,
    save_snapshot,
)

_MIN_CAPACITY: Final = 64
_IVF_RETRAIN_FACTOR: Final = 4  # retrain IVF centroids when the index grew this much


class Backend(StrEnum):
    FLAT = "flat"  # exact inner-product search (baseline)
    HNSW = "hnsw"  # approximate graph search
    IVF = "ivf"  # approximate inverted-file search


@dataclass(frozen=True)
class IndexParams:
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64
    ivf_nlist: int | None = None  # None: about sqrt(N), at least 39 training points per list
    ivf_nprobe: int = 8
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> IndexParams:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


class GalleryIndex:
    """Thread-safe vector index bound to one embedding model (``info``)."""

    def __init__(
        self,
        info: ModelInfo,
        backend: Backend | str = Backend.FLAT,
        params: IndexParams | None = None,
    ) -> None:
        self.info = info
        self.backend = Backend(backend)
        self.params = params or IndexParams()
        self._lock = threading.RLock()
        self._n = 0
        self._ids_buf = np.zeros(_MIN_CAPACITY, dtype=np.int64)
        self._vec_buf = np.zeros((_MIN_CAPACITY, info.dim), dtype=np.float32)
        self._row: dict[int, int] = {}
        self._index: Any = None  # faiss.Index
        self._quantizer: Any = None  # keeps the IVF coarse quantizer alive
        self._dirty = True
        self._trained_n = 0

    # -- basic views -------------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n

    def __contains__(self, vector_id: object) -> bool:
        return vector_id in self._row

    @property
    def ids(self) -> NDArray[np.int64]:
        with self._lock:
            return self._ids_buf[: self._n].copy()

    def vectors_for(self, ids: list[int] | NDArray[np.int64]) -> NDArray[np.float32]:
        """Stored vectors for ``ids`` (raises ``KeyError`` for an unknown id)."""
        with self._lock:
            rows = [self._row[int(i)] for i in ids]
            return self._vec_buf[rows].copy()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend": self.backend.value,
                "count": self._n,
                "dim": self.info.dim,
                "faiss_stale": self._dirty,
                "ivf_trained_on": self._trained_n,
            }

    # -- mutation ----------------------------------------------------------------------------

    def _grow(self, needed: int) -> None:
        if needed <= self._ids_buf.shape[0]:
            return
        capacity = max(needed, self._ids_buf.shape[0] * 2)
        ids = np.zeros(capacity, dtype=np.int64)
        vecs = np.zeros((capacity, self.info.dim), dtype=np.float32)
        ids[: self._n] = self._ids_buf[: self._n]
        vecs[: self._n] = self._vec_buf[: self._n]
        self._ids_buf, self._vec_buf = ids, vecs

    def add(self, ids: NDArray[np.int64] | list[int], batch: EmbeddingBatch) -> None:
        """Add vectors under ``ids``. Refuses mixed model versions and duplicate ids."""
        self.info.ensure_compatible(batch.info)
        new_ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if new_ids.shape[0] != len(batch):
            raise VectorIndexError(f"{new_ids.shape[0]} ids for {len(batch)} vectors.")
        if len(batch) == 0:
            return
        check_unit_norm(batch.vectors)
        if np.unique(new_ids).shape[0] != new_ids.shape[0]:
            raise DuplicateIdError("The same id appears twice in one add().")
        with self._lock:
            clash = [int(i) for i in new_ids if int(i) in self._row]
            if clash:
                raise DuplicateIdError(f"Vector id(s) already present: {clash[:5]}")
            start = self._n
            self._grow(start + len(batch))
            self._ids_buf[start : start + len(batch)] = new_ids
            self._vec_buf[start : start + len(batch)] = batch.vectors
            self._n += len(batch)
            for offset, vector_id in enumerate(new_ids.tolist()):
                self._row[vector_id] = start + offset
            if self._dirty or self._index is None:
                return
            if self.backend is Backend.IVF and self._n > _IVF_RETRAIN_FACTOR * max(
                self._trained_n, 1
            ):
                self._dirty = True  # centroids are stale: retrain on the next search
            else:
                self._index.add(np.ascontiguousarray(batch.vectors))

    def remove(self, ids: NDArray[np.int64] | list[int]) -> int:
        """Remove ``ids`` (unknown ids are ignored); returns how many were removed."""
        targets = {int(i) for i in np.asarray(ids, dtype=np.int64).reshape(-1)}
        with self._lock:
            present = [i for i in targets if i in self._row]
            if not present:
                return 0
            keep = ~np.isin(self._ids_buf[: self._n], np.asarray(present, dtype=np.int64))
            kept = int(keep.sum())
            self._ids_buf[:kept] = self._ids_buf[: self._n][keep]
            self._vec_buf[:kept] = self._vec_buf[: self._n][keep]
            # Overwrite the freed tail so removed vectors do not linger in memory.
            self._ids_buf[kept : self._n] = 0
            self._vec_buf[kept : self._n] = 0.0
            self._n = kept
            self._row = {int(v): r for r, v in enumerate(self._ids_buf[:kept].tolist())}
            self._dirty = True
            self._index = None
            self._quantizer = None
            return len(present)

    # -- FAISS structure ----------------------------------------------------------------------

    def _ivf_nlist(self, n: int) -> int:
        if self.params.ivf_nlist is not None:
            return max(1, min(self.params.ivf_nlist, n))
        return max(1, min(int(math.sqrt(n)), n // 39))

    def _build(self) -> None:
        n, dim = self._n, self.info.dim
        vectors = np.ascontiguousarray(self._vec_buf[:n])
        self._quantizer = None
        index: Any
        if self.backend is Backend.FLAT:
            index = faiss.IndexFlatIP(dim)
        elif self.backend is Backend.HNSW:
            index = faiss.IndexHNSWFlat(dim, self.params.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.params.hnsw_ef_construction
        else:
            self._quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(
                self._quantizer, dim, self._ivf_nlist(n), faiss.METRIC_INNER_PRODUCT
            )
            index.cp.seed = self.params.seed
            index.train(vectors)
        if n:
            index.add(vectors)
        self._index = index
        self._trained_n = n
        self._dirty = False

    def _ensure_index(self) -> Any:
        if self._index is None or self._dirty:
            self._build()
        if self.backend is Backend.HNSW:
            self._index.hnsw.efSearch = self.params.hnsw_ef_search
        elif self.backend is Backend.IVF:
            self._index.nprobe = min(self.params.ivf_nprobe, self._index.nlist)
        return self._index

    # -- search ------------------------------------------------------------------------------

    def search(
        self, queries: EmbeddingBatch, k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Top-``k`` neighbours per query as ``(scores, ids)`` of shape (Q, k).

        Missing results (``k`` larger than the index, or an empty index) are padded with score
        ``-inf`` and id ``-1``.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        self.info.ensure_compatible(queries.info)
        check_unit_norm(queries.vectors)
        q = len(queries)
        scores = np.full((q, k), -np.inf, dtype=np.float32)
        out_ids = np.full((q, k), -1, dtype=np.int64)
        if q == 0:
            return scores, out_ids
        with self._lock:
            if self._n == 0:
                return scores, out_ids
            kk = min(k, self._n)
            index = self._ensure_index()
            d, pos = index.search(np.ascontiguousarray(queries.vectors), kk)
            found = pos >= 0
            scores[:, :kk] = np.where(found, d, -np.inf)
            out_ids[:, :kk] = np.where(found, self._ids_buf[: self._n][np.where(found, pos, 0)], -1)
        return scores, out_ids

    def search_exact(
        self, queries: EmbeddingBatch, k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Brute-force numpy search over the master vectors (reference for recall checks)."""
        if k < 1:
            raise ValueError("k must be >= 1")
        self.info.ensure_compatible(queries.info)
        check_unit_norm(queries.vectors)
        q = len(queries)
        scores = np.full((q, k), -np.inf, dtype=np.float32)
        out_ids = np.full((q, k), -1, dtype=np.int64)
        with self._lock:
            if q == 0 or self._n == 0:
                return scores, out_ids
            kk = min(k, self._n)
            sims = queries.vectors @ self._vec_buf[: self._n].T
            order = np.argsort(-sims, axis=1, kind="stable")[:, :kk]
            scores[:, :kk] = np.take_along_axis(sims, order, axis=1)
            out_ids[:, :kk] = self._ids_buf[: self._n][order]
        return scores, out_ids

    # -- persistence -------------------------------------------------------------------------

    def snapshot(self) -> IndexSnapshot:
        with self._lock:
            return IndexSnapshot(
                info=self.info,
                backend=self.backend.value,
                params=self.params.to_dict(),
                ids=self._ids_buf[: self._n].copy(),
                vectors=self._vec_buf[: self._n].copy(),
            )

    def save(self, path: Path, cipher: PayloadCipher | None = None) -> None:
        """Atomically write a versioned, checksummed snapshot of the master vectors."""
        save_snapshot(path, self.snapshot(), cipher)

    @classmethod
    def from_snapshot(
        cls, snapshot: IndexSnapshot, *, expected_info: ModelInfo | None = None
    ) -> GalleryIndex:
        if expected_info is not None:
            expected_info.ensure_compatible(snapshot.info)
        index = cls(snapshot.info, snapshot.backend, IndexParams.from_dict(snapshot.params))
        if snapshot.ids.shape[0]:
            index.add(snapshot.ids, EmbeddingBatch(snapshot.vectors, snapshot.info))
        return index

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_info: ModelInfo | None = None,
        cipher: PayloadCipher | None = None,
    ) -> GalleryIndex:
        """Load a snapshot, verifying checksum, structure, norms and (optionally) model info."""
        return cls.from_snapshot(load_snapshot(path, cipher), expected_info=expected_info)
