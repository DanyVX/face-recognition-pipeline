"""Gallery service: identities (SQLite store) + fast search (GalleryIndex), under one lock.

The store is the source of truth; the in-memory index is rebuilt from it at startup. Every
operation takes a single re-entrant lock, so concurrent enroll/search/delete is safe and a search
never sees a half-applied change (for example an identity deleted while a search is running is
either fully present or fully absent).

Scores are cosine similarities. ``verify`` accepts when ``score >= threshold`` (inclusive).
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from frp.embed.base import EmbeddingBatch, ModelInfo, check_unit_norm, cosine_similarity
from frp.errors import (
    DuplicateIdentityError,
    EmbeddingError,
    IdentityExistsError,
    IdentityNotFoundError,
    InvalidIdentityError,
    ModelMismatchError,
)
from frp.index import Backend, GalleryIndex, IndexParams
from frp.store import IdentityRecord, IdentityStore, new_identity_id, normalize_name

logger = logging.getLogger(__name__)

_LOAD_CHUNK: Final = 20_000


class Aggregation(StrEnum):
    """How an identity's several embeddings are combined into one score."""

    MAX = "max"  # best match among the identity's embeddings
    MEAN = "mean"  # similarity to the identity's normalized mean embedding


class DuplicatePolicy(StrEnum):
    IGNORE = "ignore"  # enroll silently
    WARN = "warn"  # enroll and report the matches
    REJECT = "reject"  # raise DuplicateIdentityError


class OnExisting(StrEnum):
    ERROR = "error"  # an existing identity id is a conflict
    APPEND = "append"  # add the new embeddings to the existing identity


@dataclass(frozen=True)
class GalleryConfig:
    backend: Backend = Backend.FLAT
    index_params: IndexParams = field(default_factory=IndexParams)
    aggregation: Aggregation = Aggregation.MAX
    # PROVISIONAL: replaced by a value derived from LFW validation results in M6.
    duplicate_threshold: float = 0.6
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.WARN
    max_embeddings_per_identity: int = 50
    overscan: int = 8  # candidate embeddings fetched per requested identity under MAX


@dataclass(frozen=True)
class DuplicateMatch:
    identity_id: str
    name: str
    score: float


@dataclass(frozen=True)
class EnrollResult:
    identity_id: str
    embedding_ids: tuple[int, ...]
    created: bool
    duplicates: tuple[DuplicateMatch, ...] = ()


@dataclass(frozen=True)
class Candidate:
    identity_id: str
    name: str
    score: float


@dataclass(frozen=True)
class IdentifyResult:
    candidates: tuple[Candidate, ...]  # best first
    match: Candidate | None  # the top candidate if it reaches the threshold, else None
    threshold: float | None


@dataclass(frozen=True)
class VerifyResult:
    identity_id: str
    score: float
    accepted: bool
    threshold: float


def verify_pair(a: EmbeddingBatch, b: EmbeddingBatch, threshold: float) -> tuple[float, bool]:
    """1:1 comparison of two single embeddings. Returns ``(score, accepted)``; accepts when
    ``score >= threshold``. Refuses mixed model versions and non-finite scores."""
    _check_threshold(threshold)
    if len(a) != 1 or len(b) != 1:
        raise ValueError("verify_pair compares exactly one embedding with one embedding")
    score = float(cosine_similarity(a, b)[0, 0])
    if not math.isfinite(score):
        raise EmbeddingError("Similarity is not finite.")
    return score, score >= threshold


def _check_threshold(threshold: float) -> None:
    if not isinstance(threshold, int | float) or math.isnan(threshold):
        raise ValueError("threshold must be a number (not NaN)")


def _centroid(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    mean = vectors.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-9:
        raise EmbeddingError("Identity embeddings cancel out; cannot form a mean embedding.")
    return np.asarray(mean / norm, dtype=np.float32)


class Gallery:
    def __init__(
        self, store: IdentityStore, info: ModelInfo, config: GalleryConfig | None = None
    ) -> None:
        self.store = store
        self.info = info
        self.config = config or GalleryConfig()
        self._lock = threading.RLock()
        self._index = GalleryIndex(info, self.config.backend, self.config.index_params)
        self._centroids = GalleryIndex(info, Backend.FLAT)  # keyed by identity rid
        self._owner: dict[int, int] = {}  # embedding_id -> rid
        self._by_rid: dict[int, list[int]] = {}  # rid -> embedding ids
        self._meta: dict[int, tuple[str, str]] = {}  # rid -> (identity_id, name)
        self._load()

    # -- startup -----------------------------------------------------------------------------

    def _load(self) -> None:
        wanted = (
            self.info.model_id,
            self.info.model_hash,
            self.info.preprocess_version,
            self.info.dim,
        )
        foreign = {tag: n for tag, n in self.store.model_tags().items() if tag != wanted}
        if foreign:
            raise ModelMismatchError(
                f"The store holds embeddings from other model versions {sorted(foreign)} "
                f"but this gallery uses {self.info.tag}. Re-enroll with the current model."
            )
        for record in self.store.list_identities(limit=2**31 - 1):
            self._meta[record.rid] = (record.identity_id, record.name)
            self._by_rid[record.rid] = []
        ids: list[int] = []
        vectors: list[NDArray[np.float32]] = []

        def flush() -> None:
            if ids:
                self._index.add(ids, EmbeddingBatch(np.stack(vectors), self.info))
                ids.clear()
                vectors.clear()

        for row in self.store.iter_embeddings():
            ids.append(row.embedding_id)
            vectors.append(row.vector)
            self._owner[row.embedding_id] = row.rid
            self._by_rid[row.rid].append(row.embedding_id)
            if len(ids) >= _LOAD_CHUNK:
                flush()
        flush()
        for rid in self._meta:
            self._refresh_centroid(rid)
        logger.info(
            "gallery loaded: %d identities, %d embeddings", len(self._meta), len(self._owner)
        )

    def _refresh_centroid(self, rid: int) -> None:
        self._centroids.remove([rid])
        eids = self._by_rid.get(rid, [])
        if eids:
            centroid = _centroid(self._index.vectors_for(eids))
            self._centroids.add([rid], EmbeddingBatch(centroid[None], self.info))

    # -- helpers -----------------------------------------------------------------------------

    def _check_batch(self, batch: EmbeddingBatch) -> None:
        self.info.ensure_compatible(batch.info)
        check_unit_norm(batch.vectors)

    def _identity_scores(
        self, probe: NDArray[np.float32], rid: int, aggregation: Aggregation
    ) -> float:
        vectors = self._index.vectors_for(self._by_rid[rid])
        if aggregation is Aggregation.MAX:
            return float((vectors @ probe).max())
        return float(_centroid(vectors) @ probe)

    def _check_capacity(self, current: int, adding: int) -> None:
        limit = self.config.max_embeddings_per_identity
        if current + adding > limit:
            raise InvalidIdentityError(f"An identity holds at most {limit} embeddings.")

    def _find_duplicates(
        self, batch: EmbeddingBatch, exclude_rid: int | None
    ) -> tuple[DuplicateMatch, ...]:
        if len(self._index) == 0:
            return ()
        k = min(len(self._index), 10)
        scores, ids = self._index.search(batch, k)
        best: dict[int, float] = {}
        for row_scores, row_ids in zip(scores, ids, strict=True):
            for score, embedding_id in zip(row_scores, row_ids, strict=True):
                if embedding_id < 0 or score < self.config.duplicate_threshold:
                    continue
                rid = self._owner[int(embedding_id)]
                if rid != exclude_rid:
                    best[rid] = max(best.get(rid, -1.0), float(score))
        matches = [
            DuplicateMatch(self._meta[rid][0], self._meta[rid][1], score)
            for rid, score in best.items()
        ]
        return tuple(sorted(matches, key=lambda m: (-m.score, m.identity_id)))

    # -- enroll / update / delete -----------------------------------------------------------------

    def enroll(
        self,
        embeddings: EmbeddingBatch,
        *,
        name: str,
        identity_id: str | None = None,
        on_existing: OnExisting = OnExisting.ERROR,
    ) -> EnrollResult:
        """Create an identity (or extend one) from one or more embeddings of that person.

        ``name`` is only used when the identity is created. The duplicate policy compares the
        new embeddings with every *other* identity.
        """
        self._check_batch(embeddings)
        if len(embeddings) == 0:
            raise ValueError("enroll needs at least one embedding")
        name = normalize_name(name)
        with self._lock:
            identity_id = identity_id or new_identity_id()
            existing: IdentityRecord | None = None
            if self.store.has_identity(identity_id):
                if on_existing is OnExisting.ERROR:
                    raise IdentityExistsError(f"Identity {identity_id!r} already exists.")
                existing = self.store.get_identity(identity_id)
                self._check_capacity(existing.n_embeddings, len(embeddings))
            else:
                self._check_capacity(0, len(embeddings))
            duplicates: tuple[DuplicateMatch, ...] = ()
            if self.config.duplicate_policy is not DuplicatePolicy.IGNORE:
                duplicates = self._find_duplicates(embeddings, existing.rid if existing else None)
                if duplicates and self.config.duplicate_policy is DuplicatePolicy.REJECT:
                    raise DuplicateIdentityError(
                        f"This face is already enrolled as {duplicates[0].identity_id!r} "
                        f"(similarity {duplicates[0].score:.3f}).",
                        tuple((d.identity_id, d.score) for d in duplicates),
                    )
            created = existing is None
            record = existing or self.store.create_identity(identity_id, name)
            try:
                new_ids = self.store.add_embeddings(identity_id, embeddings)
                self._index.add(new_ids, embeddings)
            except BaseException:
                self._rollback_enroll(record.rid, identity_id, created)
                raise
            self._meta[record.rid] = (identity_id, record.name)
            for eid in new_ids:
                self._owner[eid] = record.rid
            self._by_rid.setdefault(record.rid, []).extend(new_ids)
            self._refresh_centroid(record.rid)
            return EnrollResult(identity_id, tuple(new_ids), created, duplicates)

    def _rollback_enroll(self, rid: int, identity_id: str, created: bool) -> None:
        """Undo a half-applied enroll so the store and the in-memory index stay consistent."""
        known = set(self._by_rid.get(rid, []))
        stored = [r.embedding_id for r in self.store.embeddings_of(identity_id)]
        fresh = [e for e in stored if e not in known]
        self._index.remove(fresh)
        if created:
            self.store.delete_identity(identity_id)
        elif fresh:
            self.store.delete_embeddings(fresh)

    def update(
        self,
        identity_id: str,
        *,
        name: str | None = None,
        embeddings: EmbeddingBatch | None = None,
        replace: bool = False,
    ) -> IdentityRecord:
        """Rename an identity and/or add embeddings; ``replace=True`` swaps out all old ones."""
        if replace and embeddings is None:
            raise ValueError("replace=True needs new embeddings")
        if embeddings is not None:
            self._check_batch(embeddings)
            if len(embeddings) == 0:
                raise ValueError("embeddings must not be empty")
        with self._lock:
            record = self.store.get_identity(identity_id)
            if name is not None:
                record = self.store.rename_identity(identity_id, name)
                self._meta[record.rid] = (identity_id, record.name)
            if embeddings is not None:
                self._check_capacity(0 if replace else record.n_embeddings, len(embeddings))
                if self.config.duplicate_policy is DuplicatePolicy.REJECT:
                    duplicates = self._find_duplicates(embeddings, record.rid)
                    if duplicates:
                        raise DuplicateIdentityError(
                            f"These embeddings match {duplicates[0].identity_id!r}.",
                            tuple((d.identity_id, d.score) for d in duplicates),
                        )
                old_ids = list(self._by_rid.get(record.rid, [])) if replace else []
                new_ids = self.store.add_embeddings(identity_id, embeddings)
                self._index.add(new_ids, embeddings)
                for eid in new_ids:
                    self._owner[eid] = record.rid
                self._by_rid.setdefault(record.rid, []).extend(new_ids)
                if old_ids:
                    self.store.delete_embeddings(old_ids)
                    self._index.remove(old_ids)
                    for eid in old_ids:
                        self._owner.pop(eid, None)
                    self._by_rid[record.rid] = [
                        e for e in self._by_rid[record.rid] if e not in set(old_ids)
                    ]
                self._refresh_centroid(record.rid)
                record = self.store.get_identity(identity_id)
            return record

    def delete(self, identity_id: str) -> int:
        """Delete an identity and all its embeddings; returns how many embeddings were removed."""
        with self._lock:
            record = self.store.get_identity(identity_id)
            removed = self.store.delete_identity(identity_id)
            self._index.remove(removed)
            self._centroids.remove([record.rid])
            for eid in removed:
                self._owner.pop(eid, None)
            self._by_rid.pop(record.rid, None)
            self._meta.pop(record.rid, None)
            return len(removed)

    # -- queries ----------------------------------------------------------------------------

    def get(self, identity_id: str) -> IdentityRecord:
        return self.store.get_identity(identity_id)

    def list_identities(self, limit: int = 100, offset: int = 0) -> list[IdentityRecord]:
        return self.store.list_identities(limit, offset)

    def __len__(self) -> int:
        return len(self._meta)

    def has_embedding(self, embedding_id: int) -> bool:
        """Whether the in-memory index still holds this embedding (used to verify erasure)."""
        with self._lock:
            return embedding_id in self._index

    def save_snapshot(self, path: Path) -> None:
        """Write an index snapshot (encrypted when the store has a key) and register it.

        Registered snapshots are rewritten without an identity's vectors when it is erased.
        """
        with self._lock:
            self._index.save(path, self.store.cipher)
            self.store.register_snapshot(str(path.resolve()))

    def stats(self) -> dict[str, object]:
        with self._lock:
            return {
                "identities": len(self._meta),
                "embeddings": len(self._owner),
                "backend": self.config.backend.value,
                "aggregation": self.config.aggregation.value,
                "model": self.info.tag,
            }

    def verify(
        self,
        probe: EmbeddingBatch,
        identity_id: str,
        *,
        threshold: float,
        aggregation: Aggregation | None = None,
    ) -> VerifyResult:
        """1:1 check of ``probe`` against an enrolled identity (accepts when score >= threshold)."""
        self._check_batch(probe)
        _check_threshold(threshold)
        if len(probe) != 1:
            raise ValueError("verify takes exactly one probe embedding")
        agg = aggregation or self.config.aggregation
        with self._lock:
            record = self.store.get_identity(identity_id)
            if not self._by_rid.get(record.rid):
                raise IdentityNotFoundError(f"Identity {identity_id!r} has no embeddings.")
            score = self._identity_scores(probe.vectors[0], record.rid, agg)
        if not math.isfinite(score):
            raise EmbeddingError("Similarity is not finite.")
        return VerifyResult(identity_id, score, score >= threshold, threshold)

    def identify(
        self,
        probe: EmbeddingBatch,
        *,
        k: int = 5,
        threshold: float | None = None,
        aggregation: Aggregation | None = None,
    ) -> IdentifyResult:
        """1:N search (open set): the best ``k`` identities, plus a match only if the top
        candidate reaches ``threshold``."""
        self._check_batch(probe)
        if k < 1:
            raise ValueError("k must be >= 1")
        if len(probe) != 1:
            raise ValueError("identify takes exactly one probe embedding")
        if threshold is not None:
            _check_threshold(threshold)
        agg = aggregation or self.config.aggregation
        with self._lock:
            candidates = (
                self._identify_max(probe, k)
                if agg is Aggregation.MAX
                else self._identify_mean(probe, k)
            )
        match = (
            candidates[0]
            if candidates and threshold is not None and candidates[0].score >= threshold
            else None
        )
        return IdentifyResult(tuple(candidates), match, threshold)

    def _identify_max(self, probe: EmbeddingBatch, k: int) -> list[Candidate]:
        total = len(self._index)
        if total == 0:
            return []
        fetch = min(total, max(k, k * self.config.overscan))
        while True:
            scores, ids = self._index.search(probe, fetch)
            best: dict[int, float] = {}
            for score, embedding_id in zip(scores[0], ids[0], strict=True):
                if embedding_id >= 0:
                    rid = self._owner[int(embedding_id)]
                    best[rid] = max(best.get(rid, -math.inf), float(score))
            if len(best) >= k or fetch >= total:
                break
            fetch = min(total, fetch * 2)  # a few identities dominate the hits: look deeper
        return self._rank(best, k)

    def _identify_mean(self, probe: EmbeddingBatch, k: int) -> list[Candidate]:
        scores, ids = self._centroids.search(probe, min(k, max(len(self._centroids), 1)))
        best = {int(rid): float(s) for s, rid in zip(scores[0], ids[0], strict=True) if rid >= 0}
        return self._rank(best, k)

    def _rank(self, best: dict[int, float], k: int) -> list[Candidate]:
        ranked = sorted(best.items(), key=lambda item: (-item[1], self._meta[item[0]][0]))[:k]
        return [Candidate(self._meta[rid][0], self._meta[rid][1], score) for rid, score in ranked]
