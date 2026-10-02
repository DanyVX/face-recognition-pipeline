"""Encryption at rest, raw-image store, audit log, rotation and full erasure."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from frp.api.audit import AuditEvent, SqliteAuditLog, redacted_id
from frp.api.idempotency import IdempotencyStore
from frp.config import ConfigError
from frp.crypto import (
    AesGcmCipher,
    DecryptionError,
    StaticKeyProvider,
    format_key,
    generate_key,
)
from frp.embed import ArcFaceEmbedder, EmbeddingBatch
from frp.erasure import erase_identity
from frp.errors import IdentityNotFoundError
from frp.factory import build_cipher
from frp.gallery import Gallery, GalleryConfig
from frp.index.format import load_snapshot
from frp.pipeline import FacePipeline
from frp.rawstore import RawImageStore
from frp.rotation import key_usage, rotate_encryption
from frp.service import RecognitionService
from frp.store import IdentityStore
from tests.unit.fake_arcface import FAKE_INFO, FakeArcFaceSession
from tests.unit.service_fixtures import SceneDetector, make_settings, person_png

DIM = 512


def cipher_for(*ids: str, active: str | None = None) -> AesGcmCipher:
    keys = {i: generate_key() for i in ids}
    return AesGcmCipher(StaticKeyProvider(keys, active or ids[0]))


def vectors(n: int, seed: int = 0) -> EmbeddingBatch:
    v = np.random.default_rng(seed).normal(size=(n, DIM)).astype(np.float32)
    return EmbeddingBatch(v / np.linalg.norm(v, axis=1, keepdims=True), FAKE_INFO)


def service(
    tmp_path: Path, cipher: AesGcmCipher | None, *, raw: bool = False
) -> RecognitionService:
    settings = make_settings(tmp_path, store_raw_images=raw)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    store = IdentityStore(settings.db_path, cipher=cipher)
    gallery = Gallery(store, FAKE_INFO, GalleryConfig(duplicate_threshold=0.99))
    raw_store = RawImageStore(settings.data_dir / "raw", store, cipher) if raw and cipher else None
    return RecognitionService(
        FacePipeline(SceneDetector()),
        ArcFaceEmbedder(FakeArcFaceSession(), FAKE_INFO),
        gallery,
        settings,
        raw_store=raw_store,
    )


# --- encrypted store --------------------------------------------------------------------


def test_embeddings_are_encrypted_in_the_database_file_and_still_searchable(tmp_path: Path) -> None:
    cipher = cipher_for("k1")
    path = tmp_path / "frp.db"
    batch = vectors(2, seed=3)
    with IdentityStore(path, cipher=cipher) as store:
        store.create_identity("alice", "Alice")
        store.add_embeddings("alice", batch)
        np.testing.assert_array_equal(store.embeddings_of("alice")[0].vector, batch.vectors[0])
    raw = path.read_bytes()
    assert batch.vectors[0].tobytes() not in raw and batch.vectors[1].tobytes() not in raw
    with IdentityStore(path, cipher=cipher) as store:
        assert (
            Gallery(store, FAKE_INFO)
            .identify(EmbeddingBatch(batch.vectors[:1], FAKE_INFO), k=1)
            .candidates[0]
            .identity_id
            == "alice"
        )


def test_encrypted_rows_cannot_be_read_without_or_with_the_wrong_key(tmp_path: Path) -> None:
    path = tmp_path / "frp.db"
    with IdentityStore(path, cipher=cipher_for("k1")) as store:
        store.create_identity("a", "A")
        store.add_embeddings("a", vectors(1))
    with IdentityStore(path) as store, pytest.raises(DecryptionError, match="no encryption key"):
        store.embeddings_of("a")
    with IdentityStore(path, cipher=cipher_for("k1")) as store, pytest.raises(DecryptionError):
        store.embeddings_of("a")  # same key id, different key material


def test_plaintext_rows_are_refused_when_encryption_is_enabled_and_migrated_on_request(
    tmp_path: Path,
) -> None:
    path = tmp_path / "frp.db"
    batch = vectors(2, seed=5)
    with IdentityStore(path) as plain:
        plain.create_identity("a", "A")
        plain.add_embeddings("a", batch)
    cipher = cipher_for("k1")
    with IdentityStore(path, cipher=cipher) as store:
        with pytest.raises(DecryptionError, match="migrate"):
            store.embeddings_of("a")
        assert key_usage(store) == {"plaintext": 2}
        assert rotate_encryption(store).embeddings_resealed == 2
        assert key_usage(store) == {"k1": 2}
        np.testing.assert_array_equal(store.embeddings_of("a")[1].vector, batch.vectors[1])
    assert batch.vectors[0].tobytes() not in path.read_bytes()


def test_blob_swapped_between_identities_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "frp.db"
    cipher = cipher_for("k1")
    with IdentityStore(path, cipher=cipher) as store:
        store.create_identity("a", "A")
        store.create_identity("b", "B")
        store.add_embeddings("a", vectors(1, 1))
        store.add_embeddings("b", vectors(1, 2))
    db = sqlite3.connect(path)
    (blob_a,) = db.execute("SELECT vector FROM embeddings WHERE embedding_id = 1").fetchone()
    db.execute("UPDATE embeddings SET vector = ? WHERE embedding_id = 2", (blob_a,))
    db.commit()
    db.close()
    with IdentityStore(path, cipher=cipher) as store, pytest.raises(DecryptionError):
        store.embeddings_of("b")


# --- key rotation -----------------------------------------------------------------------


def test_rotation_reseals_everything_under_the_new_key_and_old_key_can_then_go(
    tmp_path: Path,
) -> None:
    k1, k2 = generate_key(), generate_key()
    path = tmp_path / "frp.db"
    batch = vectors(3, seed=9)
    old = AesGcmCipher(StaticKeyProvider({"k1": k1}, "k1"))
    with IdentityStore(path, cipher=old) as store:
        store.create_identity("a", "A")
        store.add_embeddings("a", batch)
    new = AesGcmCipher(StaticKeyProvider({"k1": k1, "k2": k2}, "k2"))  # k2 active, k1 kept
    with IdentityStore(path, cipher=new) as store:
        assert key_usage(store) == {"k1": 3}
        report = rotate_encryption(store)
        assert report.embeddings_resealed == 3 and key_usage(store) == {"k2": 3}
    only_new = AesGcmCipher(StaticKeyProvider({"k2": k2}, "k2"))  # k1 removed
    with IdentityStore(path, cipher=only_new) as store:
        np.testing.assert_array_equal(store.embeddings_of("a")[0].vector, batch.vectors[0])
    with IdentityStore(path, cipher=old) as store, pytest.raises(DecryptionError):
        store.embeddings_of("a")  # the retired key no longer opens anything


def test_rotation_also_reseals_registered_snapshots(tmp_path: Path) -> None:
    k1, k2 = generate_key(), generate_key()
    snap = tmp_path / "idx.frpidx"
    old = AesGcmCipher(StaticKeyProvider({"k1": k1}, "k1"))
    svc = service(tmp_path, old)
    svc.enroll([person_png(1)], name="A", identity_id="a")
    svc.gallery.save_snapshot(snap)
    svc.close()
    new = AesGcmCipher(StaticKeyProvider({"k1": k1, "k2": k2}, "k2"))
    with IdentityStore(tmp_path / "data" / "frp.db", cipher=new) as store:
        assert rotate_encryption(store).snapshots_resealed == 1
    only_new = AesGcmCipher(StaticKeyProvider({"k2": k2}, "k2"))
    assert load_snapshot(snap, only_new).ids.shape == (1,)


# --- configuration ----------------------------------------------------------------------


def test_the_service_refuses_to_run_without_a_key_unless_explicitly_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("ENCRYPTION_KEYS", "ENCRYPTION_KEY_FILE", "ALLOW_UNENCRYPTED"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigError, match="No encryption key"):
        build_cipher(make_settings(tmp_path))
    assert build_cipher(make_settings(tmp_path, allow_unencrypted=True)) is None
    key = format_key("env1", generate_key())
    from_env = build_cipher(make_settings(tmp_path, encryption_keys=key))
    assert from_env is not None and from_env.provider.active_key_id() == "env1"
    demo = build_cipher(
        make_settings(tmp_path, encryption_key_file=tmp_path / "k.json", allow_key_generation=True)
    )
    assert demo is not None and (tmp_path / "k.json").is_file()
    with pytest.raises(Exception, match="not found"):
        build_cipher(make_settings(tmp_path, encryption_key_file=tmp_path / "missing.json"))


# --- raw image store --------------------------------------------------------------------


def test_raw_images_are_stored_sealed_and_only_when_enabled(tmp_path: Path) -> None:
    cipher = cipher_for("k1")
    png = person_png(1)
    svc = service(tmp_path / "on", cipher, raw=True)
    svc.enroll([png], name="A", identity_id="a")
    assert svc.raw_store is not None
    (name,) = svc.raw_store.files_present("a")
    stored = (tmp_path / "on" / "data" / "raw" / name).read_bytes()
    assert png not in stored and b"PNG" not in stored[:64]
    assert svc.raw_store.read("a", name) == png
    off = service(tmp_path / "off", cipher)
    off.enroll([png], name="A", identity_id="a")
    assert not (tmp_path / "off" / "data" / "raw").exists()  # default: nothing is retained


# --- audit log --------------------------------------------------------------------------


def test_audit_log_records_events_without_biometric_content_and_can_be_redacted(
    tmp_path: Path,
) -> None:
    log = SqliteAuditLog(tmp_path / "audit.db")
    log(AuditEvent("req-1", "key1", "enroll", "alice", "ok"))
    log(AuditEvent("req-2", "key1", "identify", "alice", "ok", "match"))
    log(AuditEvent("req-3", "key2", "identify", None, "NO_FACE"))
    rows = log.query(identity_id="alice")
    assert [r["action"] for r in rows] == ["identify", "enroll"] and log.count() == 3
    assert set(rows[0]) == {
        "id",
        "ts",
        "request_id",
        "key_id",
        "action",
        "identity_id",
        "outcome",
        "detail",
    }
    assert log.redact_identity("alice") == 2
    assert log.query(identity_id="alice") == []
    assert {r["identity_id"] for r in log.query()} == {redacted_id("alice"), None}
    assert log.count() == 3  # redaction keeps the rows (counts, timing, actors)


def test_audit_failures_never_break_the_request(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log = SqliteAuditLog(tmp_path / "audit.db")
    log.close()
    log(AuditEvent("r", "k", "enroll", "a", "ok"))  # closed database: logged, not raised
    assert "failed to write an audit event" in caplog.text


# --- erasure ----------------------------------------------------------------------------


def erasure_setup(
    tmp_path: Path,
) -> tuple[RecognitionService, IdempotencyStore, SqliteAuditLog, AesGcmCipher, Path]:
    cipher = cipher_for("k1")
    svc = service(tmp_path, cipher, raw=True)
    idem = IdempotencyStore(tmp_path / "data" / "idempotency.db", 3600)
    audit = SqliteAuditLog(tmp_path / "data" / "audit.db")
    return svc, idem, audit, cipher, tmp_path / "snap.frpidx"


def test_erasure_removes_every_trace_and_verifies(tmp_path: Path) -> None:
    svc, idem, audit, cipher, snap = erasure_setup(tmp_path)
    marker = "ERASE_THIS_PERSON_中文"
    svc.enroll([person_png(1), person_png(1, 4, 1)], name=marker, identity_id="victim")
    svc.enroll([person_png(2)], name="Keeper Person", identity_id="keeper")
    svc.gallery.save_snapshot(snap)
    idem.run("c", "k1", "h", lambda: (201, {"identity_id": "victim"}, "victim"))
    audit(AuditEvent("r1", "key", "enroll", "victim", "ok"))
    victim_ids = svc.gallery.store.embedding_ids_of("victim")
    assert marker.encode() in Path(svc.gallery.store.path).read_bytes()  # sanity: present before

    report = erase_identity(svc, "victim", idempotency=idem, audit_log=audit, actor="tester")

    assert report.verified and report.embeddings_removed == 2 and report.raw_images_removed == 2
    assert report.snapshots_rewritten == 1 and report.idempotency_records_purged == 1
    assert report.audit_rows_redacted == 1 and all(report.notes.values())
    assert not svc.gallery.store.has_identity("victim")
    assert not any(svc.gallery.has_embedding(i) for i in victim_ids)
    assert marker.encode() not in Path(svc.gallery.store.path).read_bytes()
    assert not set(victim_ids) & set(load_snapshot(snap, cipher).ids.tolist())
    assert svc.raw_store is not None and svc.raw_store.files_present("victim") == []
    assert (
        list((tmp_path / "data" / "raw").glob("*.frpimg"))
        and svc.gallery.get("keeper").n_embeddings == 1
    )
    assert all(r["identity_id"] != "victim" for r in audit.query())
    assert (
        audit.query(limit=1)[0]["action"] == "erase"
        and audit.query(limit=1)[0]["key_id"] == "tester"
    )
    assert svc.identify(person_png(1, 4, 7)).result.match is None  # no longer recognised


def test_store_level_deletion_needs_no_key_but_the_service_cannot_start_without_it(
    tmp_path: Path,
) -> None:
    svc, *_ = erasure_setup(tmp_path)
    svc.enroll([person_png(1)], name="Gone", identity_id="gone")
    svc.close()
    with pytest.raises(DecryptionError):  # the gallery decrypts everything at startup
        service(tmp_path, cipher_for("k1"))  # same key id, different key material
    db_path = tmp_path / "data" / "frp.db"
    with IdentityStore(db_path) as store:  # no cipher at all: row ids only, no decryption
        assert store.embedding_ids_of("gone")
        store.delete_identity("gone")
        assert not store.has_identity("gone")


def test_erasing_an_unknown_identity_is_a_clean_error(tmp_path: Path) -> None:
    svc, idem, audit, _, _ = erasure_setup(tmp_path)
    with pytest.raises(IdentityNotFoundError):
        erase_identity(svc, "nobody", idempotency=idem, audit_log=audit)


def test_unreadable_registered_snapshot_is_reported_not_skipped(tmp_path: Path) -> None:
    svc, idem, audit, _, snap = erasure_setup(tmp_path)
    svc.enroll([person_png(1)], name="A", identity_id="a")
    svc.gallery.save_snapshot(snap)
    snap.write_bytes(b"corrupted snapshot")
    from frp.erasure import ErasureVerificationError

    with pytest.raises(ErasureVerificationError, match="cannot be read"):
        erase_identity(svc, "a", idempotency=idem, audit_log=audit)


def test_missing_snapshot_files_are_forgotten(tmp_path: Path) -> None:
    svc, idem, audit, _, snap = erasure_setup(tmp_path)
    svc.enroll([person_png(1)], name="A", identity_id="a")
    svc.gallery.save_snapshot(snap)
    snap.unlink()
    report = erase_identity(svc, "a", idempotency=idem, audit_log=audit)
    assert report.snapshots_missing == 1 and svc.gallery.store.list_snapshots() == []


def test_store_schema_v1_databases_migrate_to_v2(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with IdentityStore(path) as store:
        store.create_identity("a", "A")
    db = sqlite3.connect(path)  # downgrade it to look like a v1 file
    db.execute("DROP TABLE snapshots")
    db.execute("DROP TABLE raw_images")
    db.execute("PRAGMA user_version = 1")
    db.commit()
    db.close()
    with IdentityStore(path) as store:
        assert store.get_identity("a").name == "A"
        store.register_snapshot("x.frpidx")
        assert store.list_snapshots() == ["x.frpidx"]
    assert json.dumps(sqlite3.connect(path).execute("PRAGMA user_version").fetchone()) == "[2]"
