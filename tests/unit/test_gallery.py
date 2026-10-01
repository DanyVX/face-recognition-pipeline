"""Gallery service: enroll/verify/identify/delete/update with duplicate policy and persistence."""

from __future__ import annotations

import math
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from frp.embed import EmbeddingBatch, ModelInfo
from frp.errors import (
    DuplicateIdentityError,
    EmbeddingError,
    IdentityExistsError,
    IdentityNotFoundError,
    InvalidIdentityError,
    ModelMismatchError,
)
from frp.gallery import (
    Aggregation,
    DuplicatePolicy,
    Gallery,
    GalleryConfig,
    OnExisting,
    verify_pair,
)
from frp.index import Backend
from frp.store import IdentityStore

DIM = 32
INFO = ModelInfo("test/model", "a1" * 32, "pre-v1", DIM)
OTHER = ModelInfo("test/model", "b2" * 32, "pre-v1", DIM)


def unit(n: int, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def batch(vectors: np.ndarray, info: ModelInfo = INFO) -> EmbeddingBatch:
    return EmbeddingBatch(np.ascontiguousarray(vectors, dtype=np.float32), info)


def person(seed: int, n: int = 3, noise: float = 0.15) -> np.ndarray:
    """``n`` noisy photos of one synthetic person."""
    rng = np.random.default_rng(seed)
    center = unit(1, seed + 1000)[0]
    v = center + noise * rng.normal(size=(n, DIM)).astype(np.float32)
    return np.asarray(v / np.linalg.norm(v, axis=1, keepdims=True), dtype=np.float32)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[IdentityStore]:
    with IdentityStore(tmp_path / "frp.db") as s:
        yield s


def make(store: IdentityStore, **cfg: object) -> Gallery:
    return Gallery(store, INFO, GalleryConfig(**cfg))  # type: ignore[arg-type]


# --- enroll --------------------------------------------------------------------------------------


def test_enroll_creates_an_identity_with_all_its_embeddings(store: IdentityStore) -> None:
    g = make(store)
    result = g.enroll(batch(person(1, 4)), name="Alice")
    assert result.created and len(result.embedding_ids) == 4
    assert len(result.identity_id) == 32  # generated uuid hex
    assert g.get(result.identity_id).n_embeddings == 4
    assert len(g) == 1 and g.stats()["embeddings"] == 4


def test_enroll_with_a_chosen_id_and_collision_handling(store: IdentityStore) -> None:
    g = make(store)
    g.enroll(batch(person(1)), name="Alice", identity_id="alice")
    with pytest.raises(IdentityExistsError):
        g.enroll(batch(person(2)), name="Other Alice", identity_id="alice")
    assert g.get("alice").n_embeddings == 3  # the failed enroll changed nothing
    appended = g.enroll(
        batch(person(1, 2, noise=0.1)),
        name="ignored",
        identity_id="alice",
        on_existing=OnExisting.APPEND,
    )
    assert not appended.created
    assert g.get("alice").n_embeddings == 5 and g.get("alice").name == "Alice"


def test_enroll_validates_its_inputs(store: IdentityStore) -> None:
    g = make(store)
    with pytest.raises(ValueError, match="at least one"):
        g.enroll(batch(np.zeros((0, DIM), np.float32)), name="x")
    with pytest.raises(InvalidIdentityError):
        g.enroll(batch(person(1)), name="   ")
    with pytest.raises(InvalidIdentityError):
        g.enroll(batch(person(1)), name="ok", identity_id="bad id!")
    with pytest.raises(EmbeddingError, match="not L2-normalized"):
        g.enroll(batch(person(1) * 2), name="x")
    with pytest.raises(ModelMismatchError):
        g.enroll(batch(person(1), OTHER), name="x")
    assert len(g) == 0 and store.count_embeddings() == 0  # nothing leaked into the store


def test_per_identity_embedding_limit(store: IdentityStore) -> None:
    g = make(store, max_embeddings_per_identity=4)
    with pytest.raises(InvalidIdentityError, match="at most 4"):
        g.enroll(batch(person(1, 5)), name="x")
    g.enroll(batch(person(1, 3)), name="x", identity_id="x")
    with pytest.raises(InvalidIdentityError):
        g.enroll(batch(person(1, 2)), name="x", identity_id="x", on_existing=OnExisting.APPEND)


@pytest.mark.parametrize("name", ["李雷", "José", "\U0001f600 Zoë", "x" * 256])
def test_unicode_and_long_names_work_end_to_end(store: IdentityStore, name: str) -> None:
    g = make(store)
    res = g.enroll(batch(person(5)), name=name)
    assert g.get(res.identity_id).name == name
    assert g.identify(batch(person(5, 1)), k=1).candidates[0].name == name


# --- duplicates ----------------------------------------------------------------------------------


def test_same_person_under_two_identities_is_reported_by_default(store: IdentityStore) -> None:
    g = make(store, duplicate_threshold=0.8)
    g.enroll(batch(person(7)), name="First", identity_id="first")
    second = g.enroll(batch(person(7, 2, noise=0.05)), name="Second", identity_id="second")
    assert [d.identity_id for d in second.duplicates] == ["first"]
    assert second.duplicates[0].score >= 0.8
    assert g.get("second").n_embeddings == 2  # WARN still enrolls


def test_identical_image_enrolled_under_two_identities_is_a_perfect_duplicate(
    store: IdentityStore,
) -> None:
    g = make(store, duplicate_threshold=0.9)
    one = batch(person(8, 1))
    g.enroll(one, name="A", identity_id="a")
    result = g.enroll(one, name="B", identity_id="b")
    assert result.duplicates and result.duplicates[0].score == pytest.approx(1.0, abs=1e-5)


def test_reject_policy_blocks_the_duplicate_and_leaves_no_trace(store: IdentityStore) -> None:
    g = make(store, duplicate_threshold=0.8, duplicate_policy=DuplicatePolicy.REJECT)
    g.enroll(batch(person(9)), name="First", identity_id="first")
    with pytest.raises(DuplicateIdentityError) as info:
        g.enroll(batch(person(9, 2, noise=0.05)), name="Second", identity_id="second")
    assert info.value.matches[0][0] == "first"
    assert not store.has_identity("second") and len(g) == 1


def test_ignore_policy_skips_the_check(store: IdentityStore) -> None:
    g = make(store, duplicate_threshold=0.5, duplicate_policy=DuplicatePolicy.IGNORE)
    g.enroll(batch(person(9)), name="First", identity_id="first")
    assert g.enroll(batch(person(9, 2)), name="Second").duplicates == ()


def test_adding_more_photos_of_the_same_identity_is_not_a_duplicate(store: IdentityStore) -> None:
    g = make(store, duplicate_threshold=0.5, duplicate_policy=DuplicatePolicy.REJECT)
    g.enroll(batch(person(3, 2)), name="A", identity_id="a")
    again = g.enroll(batch(person(3, 2)), name="A", identity_id="a", on_existing=OnExisting.APPEND)
    assert again.duplicates == ()


def test_distinct_people_are_not_flagged(store: IdentityStore) -> None:
    g = make(store, duplicate_threshold=0.6)
    g.enroll(batch(person(1)), name="A")
    assert g.enroll(batch(person(2)), name="B").duplicates == ()


# --- verify (1:1) --------------------------------------------------------------------------------


def test_verify_accepts_the_right_person_and_rejects_another(store: IdentityStore) -> None:
    g = make(store)
    g.enroll(batch(person(1)), name="A", identity_id="a")
    g.enroll(batch(person(2)), name="B", identity_id="b")
    probe = batch(person(1, 1, noise=0.15))
    assert g.verify(probe, "a", threshold=0.5).accepted
    assert not g.verify(probe, "b", threshold=0.5).accepted


def test_verify_boundary_is_inclusive(store: IdentityStore) -> None:
    g = make(store)
    g.enroll(batch(person(1)), name="A", identity_id="a")
    probe = batch(person(1, 1))
    score = g.verify(probe, "a", threshold=0.0).score
    assert g.verify(probe, "a", threshold=score).accepted  # score == threshold -> accept
    assert not g.verify(probe, "a", threshold=math.nextafter(score, 2.0)).accepted


def test_verify_errors(store: IdentityStore) -> None:
    g = make(store)
    g.enroll(batch(person(1)), name="A", identity_id="a")
    with pytest.raises(IdentityNotFoundError):
        g.verify(batch(person(1, 1)), "ghost", threshold=0.5)
    with pytest.raises(ValueError, match="NaN"):
        g.verify(batch(person(1, 1)), "a", threshold=float("nan"))
    with pytest.raises(ValueError, match="exactly one"):
        g.verify(batch(person(1, 2)), "a", threshold=0.5)
    with pytest.raises(ModelMismatchError):
        g.verify(batch(person(1, 1), OTHER), "a", threshold=0.5)


def test_verify_pair_is_symmetric_and_inclusive() -> None:
    a, b = batch(unit(1, 1)), batch(unit(1, 2))
    s_ab, _ = verify_pair(a, b, 0.0)
    s_ba, _ = verify_pair(b, a, 0.0)
    assert s_ab == pytest.approx(s_ba, abs=1e-7)
    assert verify_pair(a, b, s_ab)[1] is True
    assert verify_pair(a, a, 1.0 - 1e-6)[1] is True
    with pytest.raises(ModelMismatchError):
        verify_pair(a, batch(unit(1, 2), OTHER), 0.5)
    with pytest.raises(ValueError):
        verify_pair(batch(unit(2)), b, 0.5)
    with pytest.raises(ValueError, match="NaN"):
        verify_pair(a, b, float("nan"))


def test_aggregation_policies_can_disagree(store: IdentityStore) -> None:
    g = make(store)
    # One photo is far from the others: max-sim finds it, the mean embedding does not.
    center = unit(1, 1)[0]
    odd = unit(1, 2)[0]
    photos = np.stack([center, center, odd])
    photos /= np.linalg.norm(photos, axis=1, keepdims=True)
    g.enroll(batch(photos), name="Mixed", identity_id="m")
    probe = batch(odd[None])
    by_max = g.verify(probe, "m", threshold=0.0, aggregation=Aggregation.MAX).score
    by_mean = g.verify(probe, "m", threshold=0.0, aggregation=Aggregation.MEAN).score
    assert by_max == pytest.approx(1.0, abs=1e-5)
    assert by_mean < by_max - 0.2


# --- identify (1:N, open set) --------------------------------------------------------------------


def populated(store: IdentityStore, people: int = 12, **cfg: object) -> Gallery:
    g = make(store, **cfg)
    for i in range(people):
        g.enroll(batch(person(i, 3)), name=f"Person {i}", identity_id=f"p{i}")
    return g


@pytest.mark.parametrize("backend", [Backend.FLAT, Backend.HNSW, Backend.IVF])
def test_identify_finds_the_enrolled_person(store: IdentityStore, backend: Backend) -> None:
    g = populated(store, backend=backend)
    result = g.identify(batch(person(4, 1, noise=0.15)), k=3, threshold=0.5)
    assert result.candidates[0].identity_id == "p4"
    assert result.match is not None and result.match.identity_id == "p4"
    assert len({c.identity_id for c in result.candidates}) == len(result.candidates)  # per identity
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_open_set_probe_of_an_unenrolled_person_is_not_matched(store: IdentityStore) -> None:
    g = populated(store)
    stranger = batch(person(500, 1))
    result = g.identify(stranger, k=3, threshold=0.6)
    assert result.candidates  # still ranked, for inspection...
    assert result.match is None  # ...but no identity is claimed


def test_identify_without_threshold_never_claims_a_match(store: IdentityStore) -> None:
    g = populated(store)
    assert g.identify(batch(person(4, 1)), k=2).match is None


def test_identify_k_larger_than_the_gallery_and_empty_gallery(store: IdentityStore) -> None:
    g = make(store)
    assert g.identify(batch(person(1, 1)), k=5).candidates == ()
    g.enroll(batch(person(1)), name="A")
    g.enroll(batch(person(2)), name="B")
    assert len(g.identify(batch(person(1, 1)), k=50).candidates) == 2


def test_identify_looks_deeper_when_one_identity_floods_the_hits(store: IdentityStore) -> None:
    g = make(store, overscan=1)  # deliberately tiny so identity "big" fills the first window
    g.enroll(batch(person(1, 30, noise=0.02)), name="Big", identity_id="big")
    g.enroll(batch(person(2, 1)), name="Small", identity_id="small")
    result = g.identify(batch(person(1, 1, noise=0.02)), k=2)
    assert {c.identity_id for c in result.candidates} == {"big", "small"}


def test_identify_with_mean_aggregation(store: IdentityStore) -> None:
    g = populated(store, aggregation=Aggregation.MEAN)
    result = g.identify(batch(person(6, 1)), k=2, threshold=0.5)
    assert result.candidates[0].identity_id == "p6"


def test_identify_input_validation(store: IdentityStore) -> None:
    g = populated(store, 2)
    with pytest.raises(ValueError, match="k must"):
        g.identify(batch(person(1, 1)), k=0)
    with pytest.raises(ValueError, match="exactly one"):
        g.identify(batch(person(1, 2)))
    with pytest.raises(ModelMismatchError):
        g.identify(batch(person(1, 1), OTHER))


# --- delete / update / list -----------------------------------------------------------------------


@pytest.mark.parametrize("backend", [Backend.FLAT, Backend.HNSW, Backend.IVF])
def test_deleted_identity_is_gone_from_search_and_store(
    store: IdentityStore, backend: Backend
) -> None:
    g = populated(store, backend=backend)
    probe = batch(person(3, 1, noise=0.05))
    assert g.identify(probe, k=1).candidates[0].identity_id == "p3"
    assert g.delete("p3") == 3
    assert all(c.identity_id != "p3" for c in g.identify(probe, k=12).candidates)
    assert not store.has_identity("p3") and len(g) == 11
    with pytest.raises(IdentityNotFoundError):
        g.verify(probe, "p3", threshold=0.5)
    with pytest.raises(IdentityNotFoundError):
        g.delete("p3")
    mean = g.identify(probe, k=12, aggregation=Aggregation.MEAN)
    assert all(c.identity_id != "p3" for c in mean.candidates)  # centroid removed too


def test_delete_then_reload_still_gone(store: IdentityStore) -> None:
    g = populated(store, 6)
    g.delete("p2")
    reloaded = Gallery(store, INFO)
    assert len(reloaded) == 5 and "p2" not in {r.identity_id for r in reloaded.list_identities()}
    assert all(
        c.identity_id != "p2" for c in reloaded.identify(batch(person(2, 1)), k=6).candidates
    )


def test_update_name_add_and_replace_embeddings(store: IdentityStore) -> None:
    g = populated(store, 3)
    assert g.update("p1", name="Renamed").name == "Renamed"
    assert g.identify(batch(person(1, 1)), k=1).candidates[0].name == "Renamed"
    assert g.update("p1", embeddings=batch(person(1, 2))).n_embeddings == 5
    replaced = g.update("p1", embeddings=batch(person(99, 2)), replace=True)
    assert replaced.n_embeddings == 2
    # The old photos no longer match; the new ones do.
    assert g.verify(batch(person(99, 1)), "p1", threshold=0.5).accepted
    assert not g.verify(batch(person(1, 1)), "p1", threshold=0.5).accepted
    with pytest.raises(ValueError, match="replace"):
        g.update("p1", replace=True)
    with pytest.raises(IdentityNotFoundError):
        g.update("ghost", name="x")


def test_list_identities_paging(store: IdentityStore) -> None:
    g = populated(store, 5)
    assert [r.identity_id for r in g.list_identities(limit=2)] == ["p0", "p1"]
    assert [r.identity_id for r in g.list_identities(limit=2, offset=4)] == ["p4"]


# --- persistence and model guard ----------------------------------------------------------------


def test_gallery_rebuilds_identically_from_the_store(store: IdentityStore) -> None:
    g = populated(store, 8)
    probe = batch(person(5, 1))
    before = g.identify(probe, k=5)
    after = Gallery(store, INFO).identify(probe, k=5)
    assert [(c.identity_id, round(c.score, 5)) for c in before.candidates] == [
        (c.identity_id, round(c.score, 5)) for c in after.candidates
    ]


def test_opening_a_store_with_another_model_version_is_refused(store: IdentityStore) -> None:
    populated(store, 2)
    with pytest.raises(ModelMismatchError, match="other model versions"):
        Gallery(store, OTHER)


# --- concurrency ---------------------------------------------------------------------------------


def test_concurrent_enroll_search_and_delete_stay_consistent(store: IdentityStore) -> None:
    g = populated(store, 6)
    errors: list[BaseException] = []
    stop = threading.Event()

    def searcher(seed: int) -> None:
        try:
            while not stop.is_set():
                result = g.identify(batch(person(seed % 6, 1)), k=4)
                ids = [c.identity_id for c in result.candidates]
                assert len(ids) == len(set(ids))
        except BaseException as exc:
            errors.append(exc)

    def churn() -> None:
        try:
            for i in range(100, 140):
                g.enroll(batch(person(i, 2)), name=f"N{i}", identity_id=f"n{i}")
                if i % 3 == 0:
                    g.delete(f"n{i}")
        except BaseException as exc:
            errors.append(exc)
        finally:
            stop.set()

    threads = [threading.Thread(target=searcher, args=(s,)) for s in range(3)]
    threads.append(threading.Thread(target=churn))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors, errors
    survivors = [
        i for i in range(100, 140) if i % 3 != 0
    ]  # identities with i % 3 == 0 were deleted
    assert len(g) == 6 + len(survivors)
    assert store.count_identities() == len(g)


def test_identity_deleted_while_being_verified_gives_a_clean_result_or_error(
    store: IdentityStore,
) -> None:
    g = populated(store, 4)
    outcomes: list[str] = []

    def verify_loop() -> None:
        for _ in range(200):
            try:
                g.verify(batch(person(1, 1)), "p1", threshold=0.5)
                outcomes.append("ok")
            except IdentityNotFoundError:
                outcomes.append("gone")

    t = threading.Thread(target=verify_loop)
    t.start()
    g.delete("p1")
    t.join(timeout=60)
    assert set(outcomes) <= {"ok", "gone"} and len(outcomes) == 200  # never a crash or a ghost
