"""ArcFace embedding on the real model: determinism, batching, version tags, thread safety."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image, ImageFilter

from frp.align import align_face
from frp.detect import ScrfdDetector
from frp.embed import PREPROCESS_VERSION, ArcFaceEmbedder, cosine_similarity
from frp.errors import ModelFileError
from frp.models import MODELS
from tests.conftest import FIXTURES

pytestmark = pytest.mark.needs_models


@pytest.fixture(scope="module")
def embedder(models_dir: Path) -> ArcFaceEmbedder:
    spec = MODELS["arcface_r50"]
    return ArcFaceEmbedder.from_file(
        models_dir / spec.filename, expected_sha256=spec.sha256, max_batch=4
    )


@pytest.fixture(scope="module")
def detector(models_dir: Path) -> ScrfdDetector:
    spec = MODELS["scrfd_10g"]
    return ScrfdDetector.from_file(models_dir / spec.filename, expected_sha256=spec.sha256)


@pytest.fixture(scope="module")
def portrait() -> Image.Image:
    return Image.open(FIXTURES / "astronaut.jpg").convert("RGB")


def aligned_crop(detector: ScrfdDetector, img: Image.Image) -> NDArray[np.uint8]:
    arr = np.asarray(img)
    return align_face(arr, detector.detect(arr)[0].landmarks).image


@pytest.fixture(scope="module")
def crops(detector: ScrfdDetector, portrait: Image.Image) -> NDArray[np.uint8]:
    variants = [
        portrait,
        portrait.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        portrait.filter(ImageFilter.GaussianBlur(3)),
        portrait.resize((1024, 1024)),
        portrait.rotate(30, resample=Image.Resampling.BICUBIC),
        portrait,  # duplicate of the first, on purpose
        portrait.filter(ImageFilter.GaussianBlur(1)),
    ]
    return np.stack([aligned_crop(detector, v) for v in variants])


def test_model_info_is_tagged_from_the_manifest(embedder: ArcFaceEmbedder) -> None:
    assert embedder.info.model_hash == MODELS["arcface_r50"].sha256
    assert embedder.info.preprocess_version == PREPROCESS_VERSION
    assert embedder.info.dim == 512


def test_embeddings_are_unit_length_float32(
    embedder: ArcFaceEmbedder, crops: NDArray[np.uint8]
) -> None:
    batch = embedder.embed(crops)
    assert batch.vectors.shape == (len(crops), 512)
    assert batch.vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(batch.vectors, axis=1), 1.0, atol=1e-5)


def test_same_image_twice_has_cosine_one_within_1e5(
    embedder: ArcFaceEmbedder, crops: NDArray[np.uint8]
) -> None:
    pair = embedder.embed(np.stack([crops[0], crops[5]]))  # identical aligned crops
    assert abs(float(cosine_similarity(pair, pair)[0, 1]) - 1.0) < 1e-5


def test_repeated_runs_are_bitwise_identical_on_cpu(
    embedder: ArcFaceEmbedder, crops: NDArray[np.uint8]
) -> None:
    np.testing.assert_array_equal(embedder.embed(crops).vectors, embedder.embed(crops).vectors)


@pytest.mark.parametrize("n", [0, 1, 3, 7])
def test_batch_sizes_and_chunking_match_single_inference(
    embedder: ArcFaceEmbedder, crops: NDArray[np.uint8], n: int
) -> None:
    batch = embedder.embed(crops[:n])  # max_batch=4, so n=7 is processed in 2 chunks
    assert batch.vectors.shape == (n, 512)
    singles = [embedder.embed(crops[i : i + 1]).vectors[0] for i in range(n)]
    for got, want in zip(batch.vectors, singles, strict=True):
        # Observed bitwise equal on this CPU/ORT build; a tolerance is the portable guarantee.
        np.testing.assert_allclose(got, want, atol=1e-5)


def test_embeddings_of_variants_of_one_photo_are_similar(
    embedder: ArcFaceEmbedder, crops: NDArray[np.uint8]
) -> None:
    batch = embedder.embed(crops)
    sims = cosine_similarity(batch, batch)
    # flip, upscale, rotation and mild blur of the same photo stay close to the original...
    for i in (1, 3, 4, 6):
        assert sims[0, i] > 0.8, i
    # ...and nothing here is *more* similar to the original than the original itself.
    assert sims[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_concurrent_inference_on_one_session_is_consistent(
    embedder: ArcFaceEmbedder, crops: NDArray[np.uint8]
) -> None:
    expected = embedder.embed(crops).vectors
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: embedder.embed(crops).vectors, range(8)))
    for got in results:
        np.testing.assert_array_equal(got, expected)


def test_unavailable_cuda_falls_back_to_cpu_with_a_log_line(
    models_dir: Path, crops: NDArray[np.uint8], caplog: pytest.LogCaptureFixture
) -> None:
    spec = MODELS["arcface_r50"]
    with caplog.at_level(logging.WARNING, logger="frp.onnx"):
        cpu_only = ArcFaceEmbedder.from_file(
            models_dir / spec.filename,
            providers=["CUDAExecutionProvider"],
            expected_sha256=spec.sha256,
        )
    assert cpu_only.embed(crops[:1]).vectors.shape == (1, 512)  # it works, it does not crash
    if "CUDAExecutionProvider" not in cpu_only.providers:
        assert cpu_only.providers == ["CPUExecutionProvider"]
        assert "unavailable" in caplog.text


def test_corrupt_model_file_is_refused_with_an_actionable_error(
    tmp_path: Path, models_dir: Path
) -> None:
    broken = tmp_path / "w600k_r50.onnx"
    broken.write_bytes((models_dir / MODELS["arcface_r50"].filename).read_bytes()[:-1000])
    with pytest.raises(ModelFileError, match="integrity check"):
        ArcFaceEmbedder.from_file(broken, expected_sha256=MODELS["arcface_r50"].sha256)


def test_missing_model_file_is_refused_with_an_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ModelFileError, match=r"download_models\.py"):
        ArcFaceEmbedder.from_file(tmp_path / "nope.onnx")
