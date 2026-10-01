"""A fake ONNX session standing in for the ArcFace model (deterministic, no weights needed)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frp.embed import ModelInfo

DIM = 512
FAKE_INFO = ModelInfo(
    model_id="fake/arcface", model_hash="f" * 64, preprocess_version="fake-v1", dim=DIM
)


@dataclass
class FakeInput:
    name: str = "input.1"


class FakeArcFaceSession:
    """Embedding = fixed random projection of an 8x8 average-pooled copy of the input."""

    def __init__(self, *, delay: float = 0.0, output: NDArray[np.float32] | None = None) -> None:
        self._w = np.random.default_rng(1234).normal(size=(3 * 8 * 8, DIM)).astype(np.float32)
        self.batch_sizes: list[int] = []
        self.last_blob: NDArray[np.float32] | None = None
        self._delay = delay
        self._forced = output
        self._lock = threading.Lock()

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput()]

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[Any]:
        blob: NDArray[np.float32] = input_feed["input.1"]
        with self._lock:
            self.batch_sizes.append(blob.shape[0])
            self.last_blob = blob
        if self._delay:
            time.sleep(self._delay)
        if self._forced is not None:
            return [self._forced[: blob.shape[0]]]
        n = blob.shape[0]
        pooled = blob.reshape(n, 3, 8, 14, 8, 14).mean(axis=(3, 5)).reshape(n, -1)
        return [pooled @ self._w + 0.1]
