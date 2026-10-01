"""Embedder protocol and the version-tagged embedding container.

Every embedding travels with a :class:`ModelInfo` (model id, model file hash, preprocessing
version). Comparing or indexing embeddings whose tags differ raises ``ModelMismatchError``, so
vectors from different model versions cannot be mixed by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

from frp.errors import EmbeddingError, ModelMismatchError

# Unit-norm tolerance for embeddings in float32.
UNIT_NORM_TOLERANCE: Final = 1e-3


@dataclass(frozen=True)
class ModelInfo:
    """Identity of the model that produced an embedding."""

    model_id: str  # human-readable, e.g. "insightface-buffalo_l/w600k_r50"
    model_hash: str  # SHA-256 of the model file
    preprocess_version: str  # bump when pre-processing changes in any way
    dim: int

    @property
    def tag(self) -> str:
        return f"{self.model_id}@{self.model_hash[:12]}/{self.preprocess_version}"

    def ensure_compatible(self, other: ModelInfo) -> None:
        if self != other:
            raise ModelMismatchError(
                f"Embeddings come from different models: {self.tag} vs {other.tag}. "
                "Re-enroll with the current model or use a matching index."
            )


@dataclass(frozen=True, eq=False)
class EmbeddingBatch:
    """``vectors`` is (N, D) float32 with unit L2 norm; ``info`` says who made them."""

    vectors: NDArray[np.float32]
    info: ModelInfo

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2 or self.vectors.shape[1] != self.info.dim:
            raise EmbeddingError(
                f"Expected vectors of shape (N, {self.info.dim}), got {self.vectors.shape}."
            )

    def __len__(self) -> int:
        return int(self.vectors.shape[0])


class FaceEmbedder(Protocol):
    """Maps aligned 112x112 RGB faces to unit-length embeddings."""

    info: ModelInfo

    def embed(self, faces: NDArray[np.uint8]) -> EmbeddingBatch:
        """``faces``: (N, 112, 112, 3) RGB uint8 (N may be 0)."""
        ...


def l2_normalize(vectors: NDArray[np.floating], *, eps: float = 1e-12) -> NDArray[np.float32]:
    """Row-wise L2 normalization. Raises ``EmbeddingError`` on NaN/inf or zero-length rows."""
    arr = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise EmbeddingError("Embedding contains NaN or infinite values.")
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    if (norms <= eps).any():
        raise EmbeddingError("Embedding has zero length and cannot be normalized.")
    return np.asarray(arr / norms, dtype=np.float32)


def check_unit_norm(vectors: NDArray[np.floating], *, tol: float = UNIT_NORM_TOLERANCE) -> None:
    """Raise if any row is not unit length (catches 'normalization forgotten')."""
    arr = np.asarray(vectors)
    if arr.size == 0:
        return
    if not np.isfinite(arr).all():
        raise EmbeddingError("Vectors contain NaN or infinite values.")
    norms = np.linalg.norm(arr, axis=-1)
    if np.abs(norms - 1.0).max() > tol:
        raise EmbeddingError(
            f"Vectors are not L2-normalized (norms range {norms.min():.4f}-{norms.max():.4f})."
        )


def cosine_similarity(a: EmbeddingBatch, b: EmbeddingBatch) -> NDArray[np.float32]:
    """(len(a), len(b)) cosine similarities. Refuses mixed model versions."""
    a.info.ensure_compatible(b.info)
    check_unit_norm(a.vectors)
    check_unit_norm(b.vectors)
    scores = np.asarray(a.vectors @ b.vectors.T, dtype=np.float32)
    if np.isnan(scores).any():
        raise EmbeddingError("Similarity contains NaN.")
    return scores
