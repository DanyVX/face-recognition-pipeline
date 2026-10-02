"""Build pipeline components from ``Settings``."""

from __future__ import annotations

import logging

from frp.config import ConfigError, Settings, load_settings
from frp.crypto import AesGcmCipher, StaticKeyProvider, env_provider, file_provider
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
from frp.rawstore import RawImageStore
from frp.service import RecognitionService, StageTimer, make_pipeline
from frp.store import IdentityStore

logger = logging.getLogger(__name__)


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


def build_cipher(cfg: Settings) -> AesGcmCipher | None:
    """The AES-256-GCM cipher for embeddings and snapshots; ``None`` only when explicitly allowed.

    Without keys the service refuses to start (``ALLOW_UNENCRYPTED=true`` opts out for dev).
    """
    provider: StaticKeyProvider | None = None
    if cfg.encryption_keys:
        provider = env_provider(cfg.encryption_keys, cfg.encryption_active_key)
    elif cfg.encryption_key_file is not None:
        provider = file_provider(
            cfg.encryption_key_file, generate_if_missing=cfg.allow_key_generation
        )
    if provider is not None:
        return AesGcmCipher(provider)
    if cfg.allow_unencrypted:
        logger.warning("ALLOW_UNENCRYPTED=true: embeddings are stored WITHOUT encryption")
        return None
    raise ConfigError(
        "No encryption key configured. Set ENCRYPTION_KEYS (generate one with "
        "`frp keys generate-encryption-key`) or ENCRYPTION_KEY_FILE; for development only, "
        "set ALLOW_UNENCRYPTED=true."
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
    cipher = build_cipher(cfg)  # fail fast on missing keys, before loading any model
    if cfg.store_raw_images and cipher is None:
        raise ConfigError(
            "STORE_RAW_IMAGES=true requires encryption keys: raw images are only stored sealed."
        )
    detector = build_detector(cfg)
    embedder = build_embedder(cfg)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    store = IdentityStore(cfg.db_path, cipher=cipher)
    gallery = Gallery(store, embedder.info, gallery_config(cfg))
    raw_store = (
        RawImageStore(cfg.data_dir / "raw", store, cipher)
        if cfg.store_raw_images and cipher is not None
        else None
    )
    return RecognitionService(
        make_pipeline(detector, cfg),
        embedder,
        gallery,
        cfg,
        stage_timer=stage_timer,
        raw_store=raw_store,
    )
