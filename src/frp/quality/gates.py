"""Quality gates and thresholds. Each gate returns a stable reason code, not just pass/fail.

A threshold of ``None`` disables that check. Defaults are PROVISIONAL and lenient; calibrated values
(tuned on a validation split only) are loaded from a JSON file produced by
``bench/calibrate_quality.py``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Final

from frp.quality.base import Gate, GateContext, GateResult
from frp.quality.features import crop_features
from frp.quality.pose import PoseGate

FACE_TOO_SMALL: Final = "FACE_TOO_SMALL"
LOW_CONFIDENCE: Final = "LOW_CONFIDENCE"
BLURRY: Final = "BLURRY"
TOO_DARK: Final = "TOO_DARK"
TOO_BRIGHT: Final = "TOO_BRIGHT"
LOW_CONTRAST: Final = "LOW_CONTRAST"
FACE_CUT_OFF: Final = "FACE_CUT_OFF"


@dataclass(frozen=True)
class QualityThresholds:
    min_face_px: float | None = 20.0
    min_det_score: float | None = 0.5
    min_laplacian_var: float | None = None  # no sensible default before calibration
    min_brightness: float | None = 25.0
    max_brightness: float | None = 235.0
    min_contrast: float | None = 10.0
    max_abs_yaw_ratio: float | None = 0.35
    max_abs_pitch_ratio: float | None = 0.25
    reject_cut_off_faces: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> QualityThresholds:
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown quality threshold(s): {sorted(unknown)}")
        return cls(**raw)

    @classmethod
    def load(cls, path: Path) -> QualityThresholds:
        """Load thresholds from JSON: bare, under ``thresholds``, or a benchmark result file
        (``{"result": {"thresholds": {...}}}``)."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw = raw.get("result", raw)
        return cls.from_dict(raw.get("thresholds", raw))


@dataclass(frozen=True)
class SizeGate:
    min_face_px: float
    name: str = "size"

    def check(self, ctx: GateContext) -> GateResult:
        value = float(ctx.face.size_px)
        ok = value >= self.min_face_px
        return GateResult(self.name, ok, None if ok else FACE_TOO_SMALL, value, self.min_face_px)


@dataclass(frozen=True)
class ConfidenceGate:
    """Detector confidence as a cheap occlusion/ambiguity proxy."""

    min_det_score: float
    name: str = "confidence"

    def check(self, ctx: GateContext) -> GateResult:
        value = float(ctx.face.score)
        ok = value >= self.min_det_score
        return GateResult(self.name, ok, None if ok else LOW_CONFIDENCE, value, self.min_det_score)


@dataclass(frozen=True)
class SharpnessGate:
    min_laplacian_var: float
    name: str = "sharpness"

    def check(self, ctx: GateContext) -> GateResult:
        value, _, _ = crop_features(ctx.aligned)
        ok = value >= self.min_laplacian_var
        return GateResult(self.name, ok, None if ok else BLURRY, value, self.min_laplacian_var)


@dataclass(frozen=True)
class ExposureGate:
    """Brightness window and minimum contrast of the aligned crop."""

    min_brightness: float | None = None
    max_brightness: float | None = None
    min_contrast: float | None = None
    name: str = "exposure"

    def check(self, ctx: GateContext) -> GateResult:
        _, brightness, contrast = crop_features(ctx.aligned)
        if self.min_brightness is not None and brightness < self.min_brightness:
            return GateResult(self.name, False, TOO_DARK, brightness, self.min_brightness)
        if self.max_brightness is not None and brightness > self.max_brightness:
            return GateResult(self.name, False, TOO_BRIGHT, brightness, self.max_brightness)
        if self.min_contrast is not None and contrast < self.min_contrast:
            return GateResult(self.name, False, LOW_CONTRAST, contrast, self.min_contrast)
        return GateResult(self.name, True, None, brightness, None)


@dataclass(frozen=True)
class FramingGate:
    """Rejects faces cut off by the image frame (box touching the border, landmarks outside)."""

    name: str = "framing"

    def check(self, ctx: GateContext) -> GateResult:
        cut = bool(ctx.face.truncated)
        return GateResult(
            self.name,
            not cut,
            FACE_CUT_OFF if cut else None,
            float(ctx.face.visible_fraction),
            None,
        )


def build_gates(thresholds: QualityThresholds) -> list[Gate]:
    """Instantiate the gates enabled by ``thresholds`` (disabled ones are skipped)."""
    t = thresholds
    gates: list[Gate] = []
    if t.min_face_px is not None:
        gates.append(SizeGate(t.min_face_px))
    if t.min_det_score is not None:
        gates.append(ConfidenceGate(t.min_det_score))
    if t.reject_cut_off_faces:
        gates.append(FramingGate())
    if t.min_laplacian_var is not None:
        gates.append(SharpnessGate(t.min_laplacian_var))
    if any(v is not None for v in (t.min_brightness, t.max_brightness, t.min_contrast)):
        gates.append(ExposureGate(t.min_brightness, t.max_brightness, t.min_contrast))
    if t.max_abs_yaw_ratio is not None or t.max_abs_pitch_ratio is not None:
        gates.append(
            PoseGate(
                max_abs_yaw_ratio=t.max_abs_yaw_ratio if t.max_abs_yaw_ratio is not None else 1e9,
                max_abs_pitch_ratio=(
                    t.max_abs_pitch_ratio if t.max_abs_pitch_ratio is not None else 1e9
                ),
            )
        )
    return gates


@dataclass(frozen=True)
class QualityReport:
    passed: bool
    results: tuple[GateResult, ...]

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(r.reason_code for r in self.results if not r.passed and r.reason_code)


def assess(ctx: GateContext, thresholds: QualityThresholds | None = None) -> QualityReport:
    """Run every enabled gate (all of them, so every failing reason is reported)."""
    results = tuple(g.check(ctx) for g in build_gates(thresholds or QualityThresholds()))
    return QualityReport(passed=all(r.passed for r in results), results=results)
