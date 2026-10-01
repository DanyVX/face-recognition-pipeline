"""GalleryIndex behavior across the Flat, HNSW and IVF-Flat backends."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from frp.embed import EmbeddingBatch, ModelInfo
from frp.errors import (
    DuplicateIdError,
    EmbeddingError,
    IndexCorruptError,
    ModelMismatchError,
    VectorIndexError,
)
from frp.index import Backend, GalleryIndex, IndexParams

DIM = 32
INFO = ModelInfo("test/model", "b" * 64, "pre-v1", DIM)
OTHER = ModelInfo("test/model", "c" * 64, "pre-v1", DIM)  # same name, different weights
BACKENDS = [Backend.FLAT, Backend.HNSW, Backend.IVF]


def unit(n: int, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def batch(vectors: np.ndarray, info: ModelInfo = INFO) -> EmbeddingBatch:
    return EmbeddingBatch(np.ascontiguousarray(vectors, dtype=np.float32), info)


def filled(backend: Backend, n: int = 400, seed: int = 0) -> tuple[GalleryIndex, np.ndarray]:
    index = GalleryIndex(INFO, backend)
    vectors = unit(n, seed)
    index.add(np.arange(n, dtype=np.int64), batch(vectors))
    return index, vectors


# --- search basics -------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_stored_vectors_find_themselves(backend: Backend) -> None:
    index, vectors = filled(backend)
    scores, ids = index.search(batch(vectors), 5)
    hit = (ids[:, 0] == np.arange(len(vectors))).mean()
    assert hit >= (1.0 if backend is Backend.FLAT else 0.95)
    assert scores[:, 0].min() > 0.99 or backend is not Backend.FLAT
    assert (np.diff(scores, axis=1) <= 1e-6).all()  # sorted best-first
    for row in ids:
        assert len(set(row.tolist())) == 5  # no id repeated within a result


@pytest.mark.parametrize("backend", BACKENDS)
def test_k_larger_than_the_gallery_is_padded_not_an_error(backend: Backend) -> None:
    index, vectors = filled(backend, n=10)
    scores, ids = index.search(batch(vectors[:2]), 25)
    assert scores.shape == ids.shape == (2, 25)
    assert (ids[:, :10] >= 0).all()
    assert (ids[:, 10:] == -1).all() and np.isneginf(scores[:, 10:]).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_gallery_search_returns_no_results(backend: Backend) -> None:
    index = GalleryIndex(INFO, backend)
    scores, ids = index.search(batch(unit(3)), 4)
    assert (ids == -1).all() and np.isneginf(scores).all()
    assert len(index) == 0


def test_zero_queries_and_invalid_k() -> None:
    index, _ = filled(Backend.FLAT, 5)
    scores, ids = index.search(batch(np.zeros((0, DIM), np.float32)), 3)
    assert scores.shape == ids.shape == (0, 3)
    with pytest.raises(ValueError, match="k must"):
        index.search(batch(unit(1)), 0)


def test_exact_search_matches_flat_search() -> None:
    index, _ = filled(Backend.FLAT, 300)
    queries = batch(unit(20, seed=5))
    s1, i1 = index.search(queries, 7)
    s2, i2 = index.search_exact(queries, 7)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_allclose(s1, s2, atol=1e-5)


@pytest.mark.parametrize("backend", [Backend.HNSW, Backend.IVF])
def test_ann_recall_on_clustered_data_is_reasonable(backend: Backend) -> None:
    rng = np.random.default_rng(1)
    centers = unit(40, seed=2)
    vectors = centers[rng.integers(0, 40, 1500)] + 0.25 * rng.normal(size=(1500, DIM)).astype(
        np.float32
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = GalleryIndex(INFO, backend)
    index.add(np.arange(1500, dtype=np.int64), batch(vectors))
    queries = vectors[:200] + 0.05 * rng.normal(size=(200, DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    _, approx = index.search(batch(queries), 1)
    _, exact = index.search_exact(batch(queries), 1)
    assert (approx[:, 0] == exact[:, 0]).mean() >= 0.85


@pytest.mark.parametrize("vector_id", [0, -5, 2**62, 123456789012])
def test_unusual_int64_ids_work(vector_id: int) -> None:
    index = GalleryIndex(INFO)
    v = unit(1)
    index.add([vector_id], batch(v))
    _, ids = index.search(batch(v), 1)
    assert ids[0, 0] == vector_id


# --- add / remove --------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_vectors_added_after_a_search_are_found(backend: Backend) -> None:
    index, _ = filled(backend, 200)
    index.search(batch(unit(1)), 1)  # builds the FAISS structure
    extra = unit(1, seed=99)
    index.add([10_000], batch(extra))
    _, ids = index.search(batch(extra), 1)
    assert ids[0, 0] == 10_000


def test_ivf_retrains_when_the_gallery_grows_a_lot() -> None:
    index = GalleryIndex(INFO, Backend.IVF)
    index.add(np.arange(100, dtype=np.int64), batch(unit(100)))
    index.search(batch(unit(1)), 1)
    assert index.stats()["ivf_trained_on"] == 100
    index.add(np.arange(100, 900, dtype=np.int64), batch(unit(800, seed=3)))
    assert index.stats()["faiss_stale"] is True  # centroids were trained on 100 vectors
    index.search(batch(unit(1)), 1)
    assert index.stats()["ivf_trained_on"] == 900


@pytest.mark.parametrize("backend", BACKENDS)
def test_removed_vectors_are_never_returned(backend: Backend) -> None:
    index, vectors = filled(backend, 300)
    index.search(batch(vectors[:1]), 1)  # make sure a FAISS structure exists before the delete
    gone = list(range(0, 300, 3))
    assert index.remove(gone) == len(gone)
    assert len(index) == 200
    _, ids = index.search(batch(vectors), 20)
    assert not set(ids.ravel().tolist()) & set(gone)
    assert not any(g in index for g in gone)
    with pytest.raises(KeyError):
        index.vectors_for([gone[0]])


def test_remove_unknown_ids_is_a_no_op_and_everything_can_be_removed() -> None:
    index, _ = filled(Backend.FLAT, 10)
    assert index.remove([999, 1000]) == 0
    assert index.remove(list(range(10))) == 10
    assert len(index) == 0
    _, ids = index.search(batch(unit(1)), 3)
    assert (ids == -1).all()


def test_a_removed_id_can_be_reused() -> None:
    index, _ = filled(Backend.FLAT, 5)
    index.remove([2])
    new = unit(1, seed=77)
    index.add([2], batch(new))
    _, ids = index.search(batch(new), 1)
    assert ids[0, 0] == 2


def test_removal_zeroes_freed_memory() -> None:
    index, _ = filled(Backend.FLAT, 10)
    index.remove([9])
    assert not np.any(index._vec_buf[9])  # the removed slot no longer holds the vector


# --- validation ----------------------------------------------------------------------------------


def test_duplicate_ids_are_rejected_without_a_partial_add() -> None:
    index, _ = filled(Backend.FLAT, 5)
    with pytest.raises(DuplicateIdError):
        index.add([100, 3], batch(unit(2, seed=1)))  # 3 already exists
    assert len(index) == 5 and 100 not in index  # nothing from the failed call was added
    with pytest.raises(DuplicateIdError, match="twice"):
        index.add([7, 7], batch(unit(2, seed=1)))


def test_id_count_must_match_vector_count() -> None:
    index = GalleryIndex(INFO)
    with pytest.raises(VectorIndexError, match="ids for"):
        index.add([1, 2, 3], batch(unit(2)))


def test_unnormalized_and_nan_vectors_are_refused() -> None:
    index = GalleryIndex(INFO)
    with pytest.raises(EmbeddingError, match="not L2-normalized"):
        index.add([1], batch(unit(1) * 2.0))
    bad = unit(1)
    bad[0, 0] = np.nan
    with pytest.raises(EmbeddingError):
        index.add([1], batch(bad))
    with pytest.raises(EmbeddingError, match="not L2-normalized"):
        index.search(batch(unit(1) * 0.5), 1)


def test_dimension_mismatch_cannot_be_constructed() -> None:
    with pytest.raises(EmbeddingError, match="shape"):
        batch(np.zeros((2, DIM + 1), np.float32))


def test_mixing_model_versions_is_blocked_on_add_and_query() -> None:
    index, vectors = filled(Backend.FLAT, 5)
    with pytest.raises(ModelMismatchError):
        index.add([50], batch(unit(1), OTHER))
    with pytest.raises(ModelMismatchError):
        index.search(batch(vectors[:1], OTHER), 1)
    with pytest.raises(ModelMismatchError):
        index.search_exact(batch(vectors[:1], OTHER), 1)
    assert len(index) == 5


# --- persistence ---------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_load_round_trip(tmp_path: Path, backend: Backend) -> None:
    index = GalleryIndex(INFO, backend, IndexParams(hnsw_m=16, ivf_nprobe=4, seed=7))
    vectors = unit(250, seed=4)
    index.add(np.arange(1000, 1250, dtype=np.int64), batch(vectors))
    path = tmp_path / "gallery.frpidx"
    index.save(path)
    loaded = GalleryIndex.load(path, expected_info=INFO)
    assert loaded.backend is backend and loaded.params == index.params
    np.testing.assert_array_equal(loaded.ids, index.ids)
    np.testing.assert_array_equal(loaded.vectors_for(index.ids), index.vectors_for(index.ids))
    _, i1 = index.search(batch(vectors[:30]), 3)
    _, i2 = loaded.search(batch(vectors[:30]), 3)
    if backend is Backend.FLAT:
        np.testing.assert_array_equal(i1, i2)
    else:
        assert (i1[:, 0] == i2[:, 0]).mean() >= 0.95


def test_deleted_vectors_stay_deleted_after_save_and_reload(tmp_path: Path) -> None:
    index, vectors = filled(Backend.FLAT, 50)
    path = tmp_path / "gallery.frpidx"
    index.save(path)
    assert vectors[7].tobytes() in path.read_bytes()  # sanity: the vector is in the file
    index.remove([7, 8])
    index.save(path)
    data = path.read_bytes()
    assert vectors[7].tobytes() not in data and vectors[8].tobytes() not in data
    assert vectors[9].tobytes() in data  # neighbours are untouched
    reloaded = GalleryIndex.load(path)
    assert 7 not in reloaded and 8 not in reloaded and len(reloaded) == 48
    _, ids = reloaded.search(batch(vectors[7:8]), 5)
    assert 7 not in ids


def test_load_checks_the_expected_model(tmp_path: Path) -> None:
    index, _ = filled(Backend.FLAT, 5)
    path = tmp_path / "g.frpidx"
    index.save(path)
    with pytest.raises(ModelMismatchError):
        GalleryIndex.load(path, expected_info=OTHER)
    # Without an expectation the file loads, but queries from another model are still refused.
    loaded = GalleryIndex.load(path)
    with pytest.raises(ModelMismatchError):
        loaded.search(batch(unit(1), OTHER), 1)


def test_corrupted_file_is_refused_at_load(tmp_path: Path) -> None:
    index, _ = filled(Backend.FLAT, 20)
    path = tmp_path / "g.frpidx"
    index.save(path)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(IndexCorruptError):
        GalleryIndex.load(path)
    path.write_bytes(bytes(data[: len(data) // 2]))  # partially written file
    with pytest.raises(IndexCorruptError):
        GalleryIndex.load(path)


def test_load_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        GalleryIndex.load(tmp_path / "nope.frpidx")


# --- concurrency ---------------------------------------------------------------------------------


def test_concurrent_enroll_and_search_never_fail_or_return_unknown_ids() -> None:
    index = GalleryIndex(INFO, Backend.FLAT)
    vectors = unit(300, seed=11)
    index.add([0], batch(vectors[:1]))
    errors: list[BaseException] = []
    stop = threading.Event()
    added: list[int] = [0]

    def writer() -> None:
        try:
            for i in range(1, 300):
                index.add([i], batch(vectors[i : i + 1]))
                added.append(i)
        except BaseException as exc:
            errors.append(exc)
        finally:
            stop.set()

    def reader(seed: int) -> None:
        rng = np.random.default_rng(seed)
        try:
            while not stop.is_set():
                q = vectors[rng.integers(0, 300) :][:1]
                _, ids = index.search(batch(q), 5)
                valid = ids[ids >= 0]
                assert set(valid.tolist()) <= set(range(300))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer)] + [
        threading.Thread(target=reader, args=(s,)) for s in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    assert len(index) == 300


def test_vector_removed_while_searches_run_is_never_half_present() -> None:
    index, vectors = filled(Backend.FLAT, 200)
    errors: list[BaseException] = []
    stop = threading.Event()

    def searcher() -> None:
        try:
            while not stop.is_set():
                _, ids = index.search(batch(vectors[:10]), 3)
                assert ids.shape == (10, 3)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=searcher) for _ in range(3)]
    for t in threads:
        t.start()
    for i in range(0, 200, 2):
        index.remove([i])
    stop.set()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    _, ids = index.search(batch(vectors), 10)
    assert not any(i % 2 == 0 for i in ids.ravel().tolist() if i >= 0)


# --- randomized comparison against a brute-force model ------------------------------------------


class IndexModel(RuleBasedStateMachine):
    """Random add/remove/search sequences must agree with a plain dict + brute force."""

    def __init__(self) -> None:
        super().__init__()
        self.index = GalleryIndex(INFO, Backend.FLAT)
        self.model: dict[int, np.ndarray] = {}
        self.counter = 0

    @rule(n=st.integers(1, 4), seed=st.integers(0, 10_000))
    def add(self, n: int, seed: int) -> None:
        vectors = unit(n, seed)
        ids = list(range(self.counter, self.counter + n))
        self.counter += n
        self.index.add(ids, batch(vectors))
        self.model.update(dict(zip(ids, vectors, strict=True)))

    @precondition(lambda self: bool(self.model))
    @rule(data=st.data())
    def remove(self, data: st.DataObject) -> None:
        victim = data.draw(st.sampled_from(sorted(self.model)))
        assert self.index.remove([victim]) == 1
        del self.model[victim]

    @precondition(lambda self: bool(self.model))
    @rule(seed=st.integers(0, 10_000), k=st.integers(1, 6))
    def search(self, seed: int, k: int) -> None:
        q = unit(1, seed)
        scores, ids = self.index.search(batch(q), k)
        ranked = sorted(self.model, key=lambda i: -float(self.model[i] @ q[0]))[:k]
        got = [i for i in ids[0].tolist() if i >= 0]
        assert len(got) == min(k, len(self.model))
        # Compare scores (ties could permute ids): the best-k score multiset must agree.
        want_scores = sorted((float(self.model[i] @ q[0]) for i in ranked), reverse=True)
        np.testing.assert_allclose(scores[0, : len(got)], want_scores[: len(got)], atol=1e-5)

    @invariant()
    def sizes_agree(self) -> None:
        assert len(self.index) == len(self.model)
        assert set(self.index.ids.tolist()) == set(self.model)


TestIndexModel = IndexModel.TestCase
TestIndexModel.settings = settings(max_examples=30, stateful_step_count=25, deadline=None)
