"""A fake ONNX session that emits SCRFD-shaped outputs for faces placed by the test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

STRIDES = (8, 16, 32)


@dataclass
class FakeInput:
    name: str = "input.1"


@dataclass
class PlacedFace:
    """A face in *canvas* coordinates (what the real model sees after letterboxing)."""

    box: tuple[float, float, float, float]
    landmarks: NDArray[np.float32]
    score: float
    stride: int = 8
    anchor: int = 0  # which of the two anchors in the cell fires


def default_landmarks(box: tuple[float, float, float, float]) -> NDArray[np.float32]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return np.array(
        [
            [x1 + 0.3 * w, y1 + 0.38 * h],
            [x1 + 0.7 * w, y1 + 0.38 * h],
            [x1 + 0.5 * w, y1 + 0.58 * h],
            [x1 + 0.35 * w, y1 + 0.78 * h],
            [x1 + 0.65 * w, y1 + 0.78 * h],
        ],
        dtype=np.float32,
    )


class FakeScrfdSession:
    def __init__(self, faces: list[PlacedFace] | None = None, *, input_size: int = 640) -> None:
        self.faces = faces or []
        self.input_size = input_size
        self.last_blob: NDArray[np.float32] | None = None
        self.calls = 0

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput()]

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[Any]:
        self.calls += 1
        self.last_blob = input_feed["input.1"]
        scores, boxes, kps = {}, {}, {}
        for s in STRIDES:
            n = (self.input_size // s) ** 2 * 2
            scores[s] = np.zeros((n, 1), dtype=np.float32)
            boxes[s] = np.zeros((n, 4), dtype=np.float32)
            kps[s] = np.zeros((n, 10), dtype=np.float32)
        for face in self.faces:
            s = face.stride
            cells = self.input_size // s
            x1, y1, x2, y2 = face.box
            col = int(np.clip(round((x1 + x2) / 2 / s), 0, cells - 1))
            row = int(np.clip(round((y1 + y2) / 2 / s), 0, cells - 1))
            ax, ay = col * s, row * s
            idx = (row * cells + col) * 2 + face.anchor
            scores[s][idx, 0] = face.score
            boxes[s][idx] = [(ax - x1) / s, (ay - y1) / s, (x2 - ax) / s, (y2 - ay) / s]
            kps[s][idx] = ((face.landmarks - np.array([ax, ay])) / s).reshape(10)
        return (
            [scores[s] for s in STRIDES] + [boxes[s] for s in STRIDES] + [kps[s] for s in STRIDES]
        )
