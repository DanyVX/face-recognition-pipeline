"""Build pipeline components from ``Settings``."""

from __future__ import annotations

from frp.config import Settings, load_settings
from frp.detect import ScrfdDetector
from frp.embed import ArcFaceEmbedder
from frp.gallery import (
    Aggregation,
    DuplicatePolicy,
    Gallery,
    GalleryConfig,
)
from frp.index import Backend
from frp.models import MODELS
from frp.service import RecognitionService, StageTimer, make_pipeline
from frp.store import IdentityStore


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


def gallery_config(cfg: Settings) -> GalleryConfig:
    return GalleryConfig(
        backend=Backend(cfg.index_backend),
        aggregation=Aggregation(cfg.aggregation),
        duplicate_threshold=cfg.duplicate_threshold,
        duplicate_policy=DuplicatePolicy(cfg.duplicate_policy),
        max_embeddings_per_identity=cfg.max_embeddings_per_identity,
    )


def build_service(
    settings: Settings | None = None, *, stage_timer: StageTimer | None = None
) -> RecognitionService:
    """The full recognition service: models, quality gates, identity store and gallery."""
    cfg = settings or load_settings()
    detector = build_detector(cfg)
    embedder = build_embedder(cfg)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    store = IdentityStore(cfg.db_path)
    gallery = Gallery(store, embedder.info, gallery_config(cfg))
    return RecognitionService(
        make_pipeline(detector, cfg), embedder, gallery, cfg, stage_timer=stage_timer
    )
