"""Quality-gate types. Every gate returns a reason code, not just pass/fail."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from frp.align import AlignedFace
from frp.detect import Face


@dataclass(frozen=True)
class GateResult:
    """Outcome of one gate."""

    gate: str  # which gate produced this, e.g. "pose"
    passed: bool
    reason_code: str | None  # None when passed; otherwise a stable token such as "POSE_YAW"
    value: float | None = None  # the measured quantity
    threshold: float | None = None  # the limit it was compared against


@dataclass(frozen=True, eq=False)
class GateContext:
    """Everything a gate may look at for one face."""

    face: Face  # detection, in the coordinates of ``image_shape``
    aligned: AlignedFace
    image_shape: tuple[int, int]  # (height, width) of the image the face was detected in


class Gate(Protocol):
    @property
    def name(self) -> str: ...

    def check(self, ctx: GateContext) -> GateResult: ...
