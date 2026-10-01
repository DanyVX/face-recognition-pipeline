"""Build pipeline components from ``Settings``."""

from __future__ import annotations

from frp.config import Settings, load_settings
from frp.detect import ScrfdDetector
from frp.models import MODELS


def build_detector(settings: Settings | None = None) -> ScrfdDetector:
    """SCRFD detector configured from settings, with its model verified by checksum."""
    cfg = settings or load_settings()
    spec = MODELS["scrfd_10g"]
    return ScrfdDetector.from_file(
        cfg.model_dir / spec.filename,
        providers=cfg.provider_list,
        intra_op_threads=cfg.onnx_threads,
        expected_sha256=spec.sha256,
        min_confidence=cfg.det_min_confidence,
        min_face_px=cfg.min_face_px,
        max_faces=cfg.max_faces,
    )
