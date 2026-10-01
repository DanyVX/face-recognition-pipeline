"""Build pipeline components from ``Settings``."""

from __future__ import annotations

from frp.config import Settings, load_settings
from frp.detect import ScrfdDetector
from frp.embed import ArcFaceEmbedder
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


def build_embedder(settings: Settings | None = None) -> ArcFaceEmbedder:
    """ArcFace embedder configured from settings, with its model verified by checksum."""
    cfg = settings or load_settings()
    spec = MODELS["arcface_r50"]
    return ArcFaceEmbedder.from_file(
        cfg.model_dir / spec.filename,
        providers=cfg.provider_list,
        intra_op_threads=cfg.onnx_threads,
        expected_sha256=spec.sha256,
        max_batch=cfg.embed_max_batch,
    )
