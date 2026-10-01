"""A fully wired RecognitionService with fakes: image-dependent embeddings, no model weights."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from frp.align import ARCFACE_TEMPLATE
from frp.config import Settings
from frp.detect import Face
from frp.embed import ArcFaceEmbedder
from frp.gallery import Gallery, GalleryConfig
from frp.pipeline import FacePipeline
from frp.quality import build_gates
from frp.service import RecognitionService, load_thresholds
from frp.store import IdentityStore
from tests.unit.fake_arcface import FAKE_INFO, FakeArcFaceSession

WIDTH, HEIGHT = 240, 200


FACE_BOX = (65.0, 45.0, 165.0, 145.0)
_OFFSET = np.array(
    [60.0, 40.0], dtype=np.float32
)  # identity transform: crop = image[40:152, 60:172]


class SceneDetector:
    """Finds ``n_faces`` faces whose aligned crop lies fully inside the image (no black border).

    The landmarks are the canonical template shifted by (60, 40), so alignment is a pure crop and
    the embedding reflects the person's texture rather than padding.
    """

    def __init__(self, n_faces: int = 1) -> None:
        self.n_faces = n_faces

    def detect(self, image: np.ndarray) -> list[Face]:
        faces = []
        for i in range(self.n_faces):
            shift = np.array([6.0 * i, 0.0], dtype=np.float32)
            box = np.array(FACE_BOX, dtype=np.float32) - np.array(
                [i * 3, 0, i * 3, 0], dtype=np.float32
            )
            faces.append(
                Face(box=box, landmarks=ARCFACE_TEMPLATE + _OFFSET + shift, score=0.95 - 0.01 * i)
            )
        return faces


class NoFaceDetector:
    def detect(self, image: np.ndarray) -> list[Face]:
        return []


def person_png(seed: int, noise: float = 0.0, noise_seed: int = 0) -> bytes:
    """A texture identifying person ``seed``; ``noise`` gives another photo of the same person."""
    base = np.random.default_rng(seed).integers(40, 215, (HEIGHT, WIDTH, 3)).astype(np.float32)
    if noise:
        base += np.random.default_rng(10_000 + noise_seed).normal(0, noise, base.shape)
    buf = io.BytesIO()
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "data_dir": tmp_path / "data",
        "model_dir": tmp_path / "models",
        "verify_threshold": 0.9,
        "identify_threshold": 0.9,
        "duplicate_threshold": 0.97,
        "min_face_px": 20,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_service(
    tmp_path: Path,
    detector: Any = None,
    stage_timer: Any = None,
    session: FakeArcFaceSession | None = None,
    **overrides: Any,
) -> RecognitionService:
    settings = make_settings(tmp_path, **overrides)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    store = IdentityStore(settings.db_path)
    embedder = ArcFaceEmbedder(session or FakeArcFaceSession(), FAKE_INFO)
    gallery = Gallery(
        store, embedder.info, GalleryConfig(duplicate_threshold=settings.duplicate_threshold)
    )
    pipeline = FacePipeline(
        detector or SceneDetector(), gates=build_gates(load_thresholds(settings))
    )
    return RecognitionService(pipeline, embedder, gallery, settings, stage_timer=stage_timer)
