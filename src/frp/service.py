"""Recognition service: the operations behind both the HTTP API and the CLI.

Everything here is synchronous and thread-safe (the detector and embedder are stateless, the
gallery serializes its own operations), so callers may use it from a thread pool. No operation
logs or returns raw images or embeddings.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, TypeVar

import numpy as np
from numpy.typing import NDArray

from frp.config import Settings
from frp.detect import FaceDetector, FacePolicy
from frp.embed import EmbeddingBatch, FaceEmbedder
from frp.errors import FrpError, InvalidImageError
from frp.gallery import (
    EnrollResult,
    Gallery,
    IdentifyResult,
    OnExisting,
    VerifyResult,
    verify_pair,
)
from frp.imageio import LoadedImage, load_image
from frp.pipeline import FacePipeline, PreparedFace
from frp.quality import QualityThresholds, build_gates

logger = logging.getLogger(__name__)
T = TypeVar("T")

STAGES: Final = ("decode", "detect_align", "embed", "match")


@dataclass(frozen=True)
class FaceSummary:
    """What was found in one image, in original-image pixel coordinates."""

    box: tuple[float, float, float, float]
    score: float
    n_faces: int


@dataclass(frozen=True)
class EnrollOutcome:
    result: EnrollResult
    faces: tuple[FaceSummary, ...]
    embeddings: EmbeddingBatch  # returned to the API only when RETURN_EMBEDDINGS is enabled


@dataclass(frozen=True)
class PairOutcome:
    score: float
    accepted: bool
    threshold: float
    faces: tuple[FaceSummary, FaceSummary]
    embeddings: EmbeddingBatch


@dataclass(frozen=True)
class VerifyOutcome:
    result: VerifyResult
    face: FaceSummary
    embedding: EmbeddingBatch


@dataclass(frozen=True)
class IdentifyOutcome:
    result: IdentifyResult
    face: FaceSummary
    embedding: EmbeddingBatch


StageTimer = Callable[[str, float], None]


def _summary(prepared: PreparedFace, loaded: LoadedImage) -> FaceSummary:
    corners = loaded.to_original(prepared.face.box.reshape(2, 2)).reshape(4)
    x1, y1, x2, y2 = (float(v) for v in corners)
    return FaceSummary(
        box=(x1, y1, x2, y2), score=float(prepared.face.score), n_faces=prepared.n_faces
    )


class RecognitionService:
    """detect -> align -> gate -> embed -> match, behind simple methods that take image bytes."""

    def __init__(
        self,
        pipeline: FacePipeline,
        embedder: FaceEmbedder,
        gallery: Gallery,
        settings: Settings,
        *,
        stage_timer: StageTimer | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.embedder = embedder
        self.gallery = gallery
        self.settings = settings
        self._timer = stage_timer
        self._enroll_policy = FacePolicy(settings.enroll_face_policy)
        self._query_policy = FacePolicy(settings.query_face_policy)

    def set_stage_timer(self, timer: StageTimer | None) -> None:
        """Attach a callback ``(stage, seconds)`` (used for metrics)."""
        self._timer = timer

    # -- helpers -----------------------------------------------------------------------------

    def _timed(self, stage: str, fn: Callable[[], T]) -> T:
        start = time.perf_counter()
        try:
            return fn()
        finally:
            if self._timer is not None:
                self._timer(stage, time.perf_counter() - start)

    def _decode(self, data: bytes) -> LoadedImage:
        return self._timed(
            "decode",
            lambda: load_image(
                data,
                max_side=self.settings.max_image_side,
                max_pixels=self.settings.max_image_pixels,
            ),
        )

    def _prepare(self, data: bytes, policy: FacePolicy) -> tuple[PreparedFace, LoadedImage]:
        loaded = self._decode(data)
        prepared = self._timed("detect_align", lambda: self.pipeline.prepare(loaded, policy))
        return prepared, loaded

    def _embed(self, prepared: Sequence[PreparedFace]) -> EmbeddingBatch:
        crops: NDArray[np.uint8] = np.stack([p.aligned.image for p in prepared])
        return self._timed("embed", lambda: self.embedder.embed(crops))

    @staticmethod
    def _with_index(exc: FrpError, index: int) -> FrpError:
        exc.details = {**(exc.details or {}), "image_index": index}
        return exc

    # -- operations --------------------------------------------------------------------------

    def enroll(
        self,
        images: Sequence[bytes],
        *,
        name: str,
        identity_id: str | None = None,
        on_existing: OnExisting = OnExisting.ERROR,
    ) -> EnrollOutcome:
        """Enroll one person from one or more photos. All photos must pass: it is all or nothing."""
        if not images:
            raise InvalidImageError("At least one image is required.")
        prepared: list[PreparedFace] = []
        summaries: list[FaceSummary] = []
        for index, data in enumerate(images):
            try:
                face, loaded = self._prepare(data, self._enroll_policy)
            except FrpError as exc:
                raise self._with_index(exc, index) from exc
            prepared.append(face)
            summaries.append(_summary(face, loaded))
        embeddings = self._embed(prepared)
        result = self._timed(
            "match",
            lambda: self.gallery.enroll(
                embeddings, name=name, identity_id=identity_id, on_existing=on_existing
            ),
        )
        return EnrollOutcome(result, tuple(summaries), embeddings)

    def verify_images(
        self, image_a: bytes, image_b: bytes, threshold: float | None = None
    ) -> PairOutcome:
        """1:1 comparison of two photos."""
        limit = self.settings.verify_threshold if threshold is None else threshold
        outcomes: list[tuple[PreparedFace, LoadedImage]] = []
        for index, data in enumerate((image_a, image_b)):
            try:
                outcomes.append(self._prepare(data, self._query_policy))
            except FrpError as exc:
                raise self._with_index(exc, index) from exc
        embeddings = self._embed([o[0] for o in outcomes])
        a = EmbeddingBatch(embeddings.vectors[:1], embeddings.info)
        b = EmbeddingBatch(embeddings.vectors[1:], embeddings.info)
        score, accepted = self._timed("match", lambda: verify_pair(a, b, limit))
        faces = (_summary(*outcomes[0]), _summary(*outcomes[1]))
        return PairOutcome(score, accepted, limit, faces, embeddings)

    def verify_identity(
        self, image: bytes, identity_id: str, threshold: float | None = None
    ) -> VerifyOutcome:
        """1:1 check of a photo against an enrolled identity."""
        limit = self.settings.verify_threshold if threshold is None else threshold
        prepared, loaded = self._prepare(image, self._query_policy)
        embedding = self._embed([prepared])
        result = self._timed(
            "match", lambda: self.gallery.verify(embedding, identity_id, threshold=limit)
        )
        return VerifyOutcome(result, _summary(prepared, loaded), embedding)

    def identify(
        self, image: bytes, *, k: int = 5, threshold: float | None = None
    ) -> IdentifyOutcome:
        """1:N open-set search; ``match`` is set only when the best score reaches the threshold."""
        limit = self.settings.identify_threshold if threshold is None else threshold
        prepared, loaded = self._prepare(image, self._query_policy)
        embedding = self._embed([prepared])
        result = self._timed(
            "match", lambda: self.gallery.identify(embedding, k=k, threshold=limit)
        )
        return IdentifyOutcome(result, _summary(prepared, loaded), embedding)

    def delete(self, identity_id: str) -> int:
        """Erase an identity and all its embeddings; returns how many embeddings were removed."""
        return self._timed("match", lambda: self.gallery.delete(identity_id))

    def close(self) -> None:
        self.gallery.store.close()


def load_thresholds(settings: Settings) -> QualityThresholds:
    path = settings.quality_thresholds_file
    if path is None:
        return QualityThresholds()
    return QualityThresholds.load(path)


def make_pipeline(detector: FaceDetector, settings: Settings) -> FacePipeline:
    """Detector plus the configured quality gates."""
    return FacePipeline(
        detector,
        policy=FacePolicy(settings.query_face_policy),
        gates=build_gates(load_thresholds(settings)),
    )
