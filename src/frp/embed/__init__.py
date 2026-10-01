"""Embedding: protocol, version-tagged results and the ArcFace backend."""

from frp.embed.arcface import MODEL_ID, PREPROCESS_VERSION, ArcFaceEmbedder
from frp.embed.base import (
    EmbeddingBatch,
    FaceEmbedder,
    ModelInfo,
    check_unit_norm,
    cosine_similarity,
    l2_normalize,
)

__all__ = [
    "MODEL_ID",
    "PREPROCESS_VERSION",
    "ArcFaceEmbedder",
    "EmbeddingBatch",
    "FaceEmbedder",
    "ModelInfo",
    "check_unit_norm",
    "cosine_similarity",
    "l2_normalize",
]
