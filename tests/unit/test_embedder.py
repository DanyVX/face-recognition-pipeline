"""M3 embedding behavior, tested with a fake ONNX session (no model weights needed)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from frp.embed import (
    ArcFaceEmbedder,
    EmbeddingBatch,
    ModelInfo,
    check_unit_norm,
    cosine_similarity,
    l2_normalize,
)
from frp.errors import EmbeddingError, ModelMismatchError
from tests.unit.fake_arcface import DIM, FAKE_INFO, FakeArcFaceSession


def faces(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (n, 112, 112, 3), dtype=np.uint8)


def make(session: FakeArcFaceSession | None = None, **kw: int) -> ArcFaceEmbedder:
    return ArcFaceEmbedder(session or FakeArcFaceSession(), FAKE_INFO, **kw)


# --- shapes, batching, chunking ------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 7])
def test_batch_sizes_zero_one_and_many(n: int) -> None:
    batch = make().embed(faces(n))
    assert batch.vectors.shape == (n, DIM)
    assert batch.vectors.dtype == np.float32
    assert len(batch) == n
    if n:
        np.testing.assert_allclose(np.linalg.norm(batch.vectors, axis=1), 1.0, atol=1e-5)


def test_empty_batch_does_not_call_the_model() -> None:
    session = FakeArcFaceSession()
    make(session).embed(faces(0))
    assert session.batch_sizes == []


def test_large_inputs_are_chunked_and_order_is_preserved() -> None:
    session = FakeArcFaceSession()
    embedder = make(session, max_batch=16)
    data = faces(70)
    batch = embedder.embed(data)
    assert session.batch_sizes == [16, 16, 16, 16, 6]
    one_by_one = np.concatenate([embedder.embed(data[i : i + 1]).vectors for i in range(70)])
    np.testing.assert_allclose(batch.vectors, one_by_one, atol=1e-6)


def test_max_batch_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_batch"):
        make(max_batch=0)


def test_embed_one_returns_a_single_unit_vector() -> None:
    vec = make().embed_one(faces(1)[0])
    assert vec.shape == (DIM,)
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)


# --- determinism and similarity ------------------------------------------------------------------


def test_same_input_gives_bitwise_identical_embeddings() -> None:
    embedder = make()
    data = faces(3)
    np.testing.assert_array_equal(embedder.embed(data).vectors, embedder.embed(data).vectors)


def test_same_image_twice_has_cosine_one() -> None:
    embedder = make()
    face = faces(1)[0]
    pair = embedder.embed(np.stack([face, face]))
    sims = cosine_similarity(pair, pair)
    assert abs(float(sims[0, 1]) - 1.0) < 1e-5


def test_different_images_are_less_similar_than_identical_ones() -> None:
    embedder = make()
    a, b = faces(1, seed=1), faces(1, seed=2)
    assert float(cosine_similarity(embedder.embed(a), embedder.embed(b))[0, 0]) < 0.99


# --- preprocessing -------------------------------------------------------------------------------


def test_preprocessing_is_rgb_normalized_nchw() -> None:
    session = FakeArcFaceSession()
    red = np.zeros((1, 112, 112, 3), dtype=np.uint8)
    red[..., 0] = 255
    make(session).embed(red)
    blob = session.last_blob
    assert blob is not None
    assert blob.shape == (1, 3, 112, 112)
    assert blob.dtype == np.float32
    assert blob[0, 0, 5, 5] == pytest.approx(1.0)  # channel 0 is red: (255 - 127.5) / 127.5
    assert blob[0, 1, 5, 5] == pytest.approx(-1.0)
    assert blob[0, 2, 5, 5] == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 112, 112, 3), dtype=np.float32),  # wrong dtype
        np.zeros((112, 112, 3), dtype=np.uint8),  # missing batch axis
        np.zeros((2, 100, 100, 3), dtype=np.uint8),  # wrong size
        np.zeros((2, 112, 112, 4), dtype=np.uint8),  # RGBA
        np.zeros((2, 3, 112, 112), dtype=np.uint8),  # channels first
    ],
)
def test_invalid_input_arrays_are_rejected(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="uint8 RGB"):
        make().embed(bad)


# --- numerical failures fail loudly --------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_output_is_an_error(bad: float) -> None:
    out = np.ones((4, DIM), dtype=np.float32)
    out[1, 7] = bad
    with pytest.raises(EmbeddingError, match="NaN or infinite"):
        make(FakeArcFaceSession(output=out)).embed(faces(4))


def test_zero_output_is_an_error() -> None:
    with pytest.raises(EmbeddingError, match="zero length"):
        make(FakeArcFaceSession(output=np.zeros((2, DIM), dtype=np.float32))).embed(faces(2))


def test_wrong_output_dimension_is_an_error() -> None:
    with pytest.raises(EmbeddingError, match="shape"):
        make(FakeArcFaceSession(output=np.ones((2, 256), dtype=np.float32))).embed(faces(2))


def test_l2_normalize_and_unit_norm_check() -> None:
    v = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    n = l2_normalize(v)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)
    check_unit_norm(n)
    with pytest.raises(EmbeddingError, match="not L2-normalized"):
        check_unit_norm(v)  # normalization forgotten
    check_unit_norm(np.zeros((0, 4), dtype=np.float32))  # empty is fine


# --- version tagging -----------------------------------------------------------------------------


def test_every_batch_carries_its_model_info() -> None:
    batch = make().embed(faces(2))
    assert batch.info == FAKE_INFO
    assert "fake/arcface" in batch.info.tag and "fake-v1" in batch.info.tag


def test_mixing_model_versions_is_impossible() -> None:
    a = make().embed(faces(2))
    for field in ("model_id", "model_hash", "preprocess_version"):
        other_info = ModelInfo(**{**FAKE_INFO.__dict__, field: "different"})
        b = EmbeddingBatch(a.vectors, other_info)
        with pytest.raises(ModelMismatchError):
            cosine_similarity(a, b)


def test_cosine_similarity_refuses_unnormalized_vectors() -> None:
    a = make().embed(faces(2))
    broken = EmbeddingBatch(a.vectors * 2.0, FAKE_INFO)
    with pytest.raises(EmbeddingError, match="not L2-normalized"):
        cosine_similarity(a, broken)


def test_batch_rejects_wrong_dimension_for_its_info() -> None:
    with pytest.raises(EmbeddingError, match="shape"):
        EmbeddingBatch(np.zeros((2, 10), dtype=np.float32), FAKE_INFO)


# --- thread safety -------------------------------------------------------------------------------


def test_concurrent_calls_return_the_same_results_as_sequential_ones() -> None:
    embedder = make(FakeArcFaceSession(delay=0.002), max_batch=4)
    inputs = [faces(9, seed=s) for s in range(8)]
    expected = [embedder.embed(x).vectors for x in inputs]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda x: embedder.embed(x).vectors, inputs * 3))
    for i, got in enumerate(results):
        np.testing.assert_array_equal(got, expected[i % 8])
