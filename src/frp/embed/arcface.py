"""ArcFace (InsightFace ``w600k_r50``) embedder on ONNX Runtime.

Pre-processing follows the reference: aligned 112x112 RGB, ``(x - 127.5) / 127.5``, NCHW float32.
Output is the 512-d embedding, L2-normalized here (the raw network output is not unit length).

License note: the pretrained weights are for non-commercial research use only; see MODEL_CARD.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
from numpy.typing import NDArray

from frp.align import ALIGNED_SIZE
from frp.embed.base import EmbeddingBatch, ModelInfo, l2_normalize
from frp.errors import EmbeddingError
from frp.onnx import create_session, session_providers, verify_model_file

PREPROCESS_VERSION: Final = "arcface-rgb-127.5-v1"
MODEL_ID: Final = "insightface-buffalo_l/w600k_r50"
EMBEDDING_DIM: Final = 512
_MEAN: Final = 127.5
_STD: Final = 127.5


class EmbedSessionLike(Protocol):
    """The subset of ``onnxruntime.InferenceSession`` used here (lets tests inject a fake)."""

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[Any]: ...

    def get_inputs(self) -> list[Any]: ...


class ArcFaceEmbedder:
    """``FaceEmbedder`` backend for the ArcFace ResNet-50 model.

    Thread-safe: it keeps no mutable state, and ONNX Runtime sessions allow concurrent ``run``.

    Args:
        session: ONNX session (or compatible object).
        info: model identity stored with every embedding.
        max_batch: faces per inference call; larger inputs are processed in chunks so memory
            stays bounded.
    """

    def __init__(self, session: EmbedSessionLike, info: ModelInfo, *, max_batch: int = 16) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        self._session = session
        self._input_name: str = session.get_inputs()[0].name
        self.info = info
        self.max_batch = max_batch

    @classmethod
    def from_file(
        cls,
        model_path: Path,
        *,
        providers: list[str] | None = None,
        intra_op_threads: int | None = None,
        expected_sha256: str | None = None,
        max_batch: int = 16,
        model_id: str = MODEL_ID,
        preprocess_version: str = PREPROCESS_VERSION,
    ) -> ArcFaceEmbedder:
        model_hash = verify_model_file(model_path, expected_sha256)
        session = create_session(model_path, providers=providers, intra_op_threads=intra_op_threads)
        info = ModelInfo(
            model_id=model_id,
            model_hash=model_hash,
            preprocess_version=preprocess_version,
            dim=EMBEDDING_DIM,
        )
        return cls(session, info, max_batch=max_batch)

    @property
    def providers(self) -> list[str]:
        """Execution providers actually in use."""
        return session_providers(self._session)

    @staticmethod
    def _blob(faces: NDArray[np.uint8]) -> NDArray[np.float32]:
        blob = (faces.astype(np.float32) - _MEAN) / _STD  # RGB, as the model expects
        return np.ascontiguousarray(blob.transpose(0, 3, 1, 2))

    def _run_chunk(self, faces: NDArray[np.uint8]) -> NDArray[np.float32]:
        outputs = self._session.run(None, {self._input_name: self._blob(faces)})
        raw = np.asarray(outputs[0], dtype=np.float32)
        if raw.shape != (faces.shape[0], self.info.dim):
            raise EmbeddingError(
                f"Embedder returned shape {raw.shape}, "
                f"expected ({faces.shape[0]}, {self.info.dim})."
            )
        return l2_normalize(raw)

    def embed(self, faces: NDArray[np.uint8]) -> EmbeddingBatch:
        """Embed aligned faces of shape (N, 112, 112, 3), RGB uint8. ``N`` may be zero."""
        arr = np.asarray(faces)
        if (
            arr.dtype != np.uint8
            or arr.ndim != 4
            or arr.shape[1:] != (ALIGNED_SIZE, ALIGNED_SIZE, 3)
        ):
            raise ValueError(
                f"faces must be uint8 RGB of shape (N, {ALIGNED_SIZE}, {ALIGNED_SIZE}, 3), "
                f"got {arr.dtype} {arr.shape}"
            )
        if arr.shape[0] == 0:
            return EmbeddingBatch(np.zeros((0, self.info.dim), dtype=np.float32), self.info)
        chunks = [
            self._run_chunk(arr[start : start + self.max_batch])
            for start in range(0, arr.shape[0], self.max_batch)
        ]
        return EmbeddingBatch(np.concatenate(chunks, axis=0), self.info)

    def embed_one(self, face: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Convenience for a single (112, 112, 3) face; returns a (512,) unit vector."""
        return np.asarray(self.embed(np.asarray(face)[None]).vectors[0], dtype=np.float32)
