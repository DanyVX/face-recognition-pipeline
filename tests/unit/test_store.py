"""Identity store and validation: CRUD, Unicode, persistence, erasure, encryption hook."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from frp.embed import EmbeddingBatch, ModelInfo
from frp.errors import (
    IdentityExistsError,
    IdentityNotFoundError,
    InvalidIdentityError,
)
from frp.store import (
    MAX_NAME_CHARS,
    IdentityStore,
    StoreSchemaError,
    new_identity_id,
    normalize_name,
    validate_identity_id,
)
from tests.unit.toy_cipher import ToyCipher

DIM = 16
INFO = ModelInfo("test/model", "d" * 64, "pre-v1", DIM)
OTHER = ModelInfo("test/model", "e" * 64, "pre-v1", DIM)


def unit(n: int, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def batch(n: int = 1, seed: int = 0, info: ModelInfo = INFO) -> EmbeddingBatch:
    return EmbeddingBatch(unit(n, seed), info)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[IdentityStore]:
    with IdentityStore(tmp_path / "frp.db") as s:
        yield s


# --- validation ----------------------------------------------------------------------------------


@pytest.mark.parametrize("good", ["a", "alice", "A1", "user.name", "id:42", "x@y-z_1", "0" * 128])
def test_valid_identity_ids(good: str) -> None:
    assert validate_identity_id(good) == good


@pytest.mark.parametrize(
    "bad",
    [
        "",
        " ",
        "-lead",
        ".lead",
        "has space",
        "slash/x",
        "\u00dcn\u00ef",
        "a" * 129,
        "x\n",
        "<script>",
    ],
)
def test_invalid_identity_ids(bad: str) -> None:
    with pytest.raises(InvalidIdentityError):
        validate_identity_id(bad)


def test_generated_ids_are_valid_and_unique() -> None:
    ids = {new_identity_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(validate_identity_id(i) for i in ids)


@pytest.mark.parametrize(
    "name",
    [
        "Jos\u00e9 Garc\u00eda",
        "\u674e\u96f7",
        "\u041e\u043b\u044c\u0433\u0430",
        "\u0645\u062d\u0645\u062f",
        "\u05e9\u05dc\u05d5\u05dd",
        "Zo\u00eb \U0001f600",
        "O'Brien-Smith Jr.",
        "\u0ba8\u0bbf\u0bb2\u0bbe",
        "a",
    ],
)
def test_unicode_names_are_accepted(name: str) -> None:
    assert normalize_name(name) == name


def test_names_are_nfc_normalized_and_trimmed() -> None:
    assert normalize_name("  Jose\u0301  ") == "Jos\u00e9"  # combining accent composed


def test_name_length_boundary() -> None:
    assert len(normalize_name("\u00e9" * MAX_NAME_CHARS)) == MAX_NAME_CHARS
    with pytest.raises(InvalidIdentityError, match="too long"):
        normalize_name("\u00e9" * (MAX_NAME_CHARS + 1))


@pytest.mark.parametrize(
    "bad", ["", "   ", "\t\n", "a\x00b", "line\nbreak", "bidi\u202eevil", "\ud800"]
)
def test_bad_names_are_rejected(bad: str) -> None:
    with pytest.raises(InvalidIdentityError):
        normalize_name(bad)


def test_non_text_names_are_rejected() -> None:
    with pytest.raises(InvalidIdentityError):
        normalize_name(123)  # type: ignore[arg-type]


@given(st.text(min_size=1, max_size=40))
def test_normalization_is_idempotent_or_rejects(text: str) -> None:
    try:
        once = normalize_name(text)
    except InvalidIdentityError:
        return
    assert normalize_name(once) == once


# --- identities ----------------------------------------------------------------------------------


def test_create_get_list_count(store: IdentityStore) -> None:
    a = store.create_identity("alice", "Alice")
    store.create_identity("bob", "Bob")
    assert a.identity_id == "alice" and a.n_embeddings == 0
    assert store.get_identity("alice").name == "Alice"
    assert [r.identity_id for r in store.list_identities()] == ["alice", "bob"]
    assert [r.identity_id for r in store.list_identities(limit=1, offset=1)] == ["bob"]
    assert store.list_identities(limit=10, offset=5) == []
    assert store.count_identities() == 2
    assert store.has_identity("alice") and not store.has_identity("carol")


def test_identity_id_collision_is_an_error(store: IdentityStore) -> None:
    store.create_identity("alice", "Alice")
    with pytest.raises(IdentityExistsError):
        store.create_identity("alice", "Another Alice")
    assert store.get_identity("alice").name == "Alice"  # unchanged


def test_unknown_identity_errors(store: IdentityStore) -> None:
    with pytest.raises(IdentityNotFoundError):
        store.get_identity("ghost")
    with pytest.raises(IdentityNotFoundError):
        store.rename_identity("ghost", "x")
    with pytest.raises(IdentityNotFoundError):
        store.delete_identity("ghost")
    with pytest.raises(IdentityNotFoundError):
        store.add_embeddings("ghost", batch())
    with pytest.raises(IdentityNotFoundError):
        store.embeddings_of("ghost")


def test_unicode_and_long_names_survive_a_round_trip(store: IdentityStore) -> None:
    store.create_identity("u1", "\u674e\u96f7 \U0001f600 Jos\u00e9")
    store.create_identity("u2", "\u00e9" * MAX_NAME_CHARS)
    assert store.get_identity("u1").name == "\u674e\u96f7 \U0001f600 Jos\u00e9"
    assert len(store.get_identity("u2").name) == MAX_NAME_CHARS


def test_rename(store: IdentityStore) -> None:
    store.create_identity("alice", "Alice")
    assert store.rename_identity("alice", "Alicia").name == "Alicia"


def test_list_rejects_negative_paging(store: IdentityStore) -> None:
    with pytest.raises(ValueError):
        store.list_identities(limit=-1)


# --- embeddings ----------------------------------------------------------------------------------


def test_embeddings_round_trip_exactly_with_their_model_tag(store: IdentityStore) -> None:
    store.create_identity("alice", "Alice")
    vectors = batch(3, seed=1)
    ids = store.add_embeddings("alice", vectors)
    assert len(ids) == 3 and ids == sorted(ids)
    rows = store.embeddings_of("alice")
    np.testing.assert_array_equal(np.stack([r.vector for r in rows]), vectors.vectors)
    assert all(r.info == INFO and r.identity_id == "alice" for r in rows)
    assert [r.embedding_id for r in rows] == ids
    assert store.get_identity("alice").n_embeddings == 3
    assert store.count_embeddings() == 3


def test_model_tags_are_counted_per_version(store: IdentityStore) -> None:
    store.create_identity("a", "A")
    store.add_embeddings("a", batch(2))
    store.add_embeddings("a", batch(1, info=OTHER))
    tags = store.model_tags()
    assert tags[("test/model", "d" * 64, "pre-v1", DIM)] == 2
    assert tags[("test/model", "e" * 64, "pre-v1", DIM)] == 1


def test_empty_batch_adds_nothing(store: IdentityStore) -> None:
    store.create_identity("a", "A")
    assert store.add_embeddings("a", EmbeddingBatch(np.zeros((0, DIM), np.float32), INFO)) == []


def test_iter_embeddings_covers_all_identities_in_id_order(store: IdentityStore) -> None:
    store.create_identity("a", "A")
    store.create_identity("b", "B")
    store.add_embeddings("a", batch(2, 1))
    store.add_embeddings("b", batch(2, 2))
    store.add_embeddings("a", batch(1, 3))
    rows = list(store.iter_embeddings())
    assert [r.identity_id for r in rows] == ["a", "a", "b", "b", "a"]
    assert [r.embedding_id for r in rows] == sorted(r.embedding_id for r in rows)


def test_delete_embeddings_and_identity_cascade(store: IdentityStore) -> None:
    store.create_identity("a", "A")
    ids = store.add_embeddings("a", batch(3))
    assert store.delete_embeddings(ids[:1]) == 1
    assert store.get_identity("a").n_embeddings == 2
    removed = store.delete_identity("a")
    assert sorted(removed) == sorted(ids[1:])
    assert store.count_embeddings() == 0 and store.count_identities() == 0


def test_failed_batch_insert_is_rolled_back_completely(tmp_path: Path) -> None:
    class FlakyCipher(ToyCipher):
        calls = 0

        def encrypt(self, plaintext: bytes, aad: bytes) -> bytes:
            FlakyCipher.calls += 1
            if FlakyCipher.calls == 3:
                raise RuntimeError("encryption backend down")
            return super().encrypt(plaintext, aad)

    with IdentityStore(tmp_path / "f.db", cipher=FlakyCipher()) as s:
        s.create_identity("a", "A")
        with pytest.raises(RuntimeError, match="backend down"):
            s.add_embeddings("a", batch(5))
        assert s.count_embeddings() == 0  # none of the first two rows survived


# --- persistence, schema, concurrency ------------------------------------------------------------


def test_data_survives_reopening_the_file(tmp_path: Path) -> None:
    path = tmp_path / "frp.db"
    with IdentityStore(path) as s:
        s.create_identity("alice", "Alice")
        s.add_embeddings("alice", batch(2, seed=9))
    with IdentityStore(path) as s2:
        assert s2.get_identity("alice").n_embeddings == 2
        np.testing.assert_array_equal(
            np.stack([r.vector for r in s2.embeddings_of("alice")]), batch(2, seed=9).vectors
        )


def test_newer_schema_versions_are_refused_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "frp.db"
    IdentityStore(path).close()
    db = sqlite3.connect(path)
    db.execute("PRAGMA user_version = 99")
    db.commit()
    db.close()
    with pytest.raises(StoreSchemaError, match="not supported"):
        IdentityStore(path)


def test_concurrent_writers_do_not_lose_rows(tmp_path: Path) -> None:
    with IdentityStore(tmp_path / "c.db") as s:
        for i in range(8):
            s.create_identity(f"id{i}", f"Name {i}")
        errors: list[BaseException] = []

        def work(i: int) -> None:
            try:
                for j in range(10):
                    s.add_embeddings(f"id{i}", batch(1, seed=i * 100 + j))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not errors, errors
        assert s.count_embeddings() == 80
        assert all(s.get_identity(f"id{i}").n_embeddings == 10 for i in range(8))


# --- erasure -------------------------------------------------------------------------------------


def test_deleted_name_and_vectors_do_not_remain_in_the_database_file(tmp_path: Path) -> None:
    path = tmp_path / "frp.db"
    marker = "ERASE_ME_UNIQUE_NAME_\u4e2d\u6587"
    vectors = batch(2, seed=21)
    with IdentityStore(path) as s:
        s.create_identity("victim", marker)
        s.add_embeddings("victim", vectors)
        s.create_identity("keeper", "Keeper Person")
        s.add_embeddings("keeper", batch(1, seed=22))
    raw = path.read_bytes()
    assert marker.encode() in raw  # sanity: the test can see the data before deletion
    assert vectors.vectors[0].tobytes() in raw
    with IdentityStore(path) as s:
        s.delete_identity("victim")
        s.vacuum()
    raw = path.read_bytes()
    assert marker.encode() not in raw
    assert vectors.vectors[0].tobytes() not in raw and vectors.vectors[1].tobytes() not in raw
    assert b"Keeper Person" in raw  # other identities are untouched


# --- encryption hook -----------------------------------------------------------------------------


def test_blobs_are_encrypted_at_rest_and_bound_to_their_identity(tmp_path: Path) -> None:
    path = tmp_path / "enc.db"
    vectors = batch(1, seed=5)
    with IdentityStore(path, cipher=ToyCipher()) as s:
        s.create_identity("a", "A")
        s.create_identity("b", "B")
        s.add_embeddings("a", vectors)
        np.testing.assert_array_equal(s.embeddings_of("a")[0].vector, vectors.vectors[0])
    assert vectors.vectors[0].tobytes() not in path.read_bytes()

    db = sqlite3.connect(path)  # an attacker copies identity a's blob onto identity b's row
    (blob,) = db.execute("SELECT vector FROM embeddings").fetchone()
    db.execute(
        "INSERT INTO embeddings(rid, model_id, model_hash, preprocess_version, dim, vector, "
        "created_at) VALUES (2, 'test/model', ?, 'pre-v1', ?, ?, 'x')",
        ("d" * 64, DIM, blob),
    )
    db.commit()
    db.close()
    with (
        IdentityStore(path, cipher=ToyCipher()) as s,
        pytest.raises(ValueError, match="authentication"),
    ):
        s.embeddings_of("b")
