"""CLI commands for key generation, erasure, snapshots, audit and encryption migration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from frp import cli
from frp.crypto import AesGcmCipher, env_provider, format_key, generate_key
from frp.embed import EmbeddingBatch
from frp.store import IdentityStore
from tests.unit.fake_arcface import FAKE_INFO
from tests.unit.service_fixtures import make_service, person_png

runner = CliRunner()


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(cli, "build_service", lambda: make_service(tmp_path))
    return tmp_path


def test_generate_encryption_key_prints_a_parsable_key_and_validates_the_id() -> None:
    out = runner.invoke(cli.app, ["keys", "generate-encryption-key", "--id", "prod.1"])
    assert out.exit_code == 0
    provider = env_provider(out.stdout.strip(), None)
    assert provider.active_key_id() == "prod.1"
    bad = runner.invoke(cli.app, ["keys", "generate-encryption-key", "--id", "has space"])
    assert bad.exit_code == 2


def test_snapshot_erase_and_audit_through_the_cli(env: Path) -> None:
    service = make_service(env)
    service.enroll([person_png(1)], name="Alice", identity_id="alice")
    service.enroll([person_png(2)], name="Bob", identity_id="bob")
    service.close()

    snap = env / "idx.frpidx"
    assert runner.invoke(cli.app, ["snapshot", str(snap)]).exit_code == 0 and snap.is_file()

    result = runner.invoke(cli.app, ["erase", "--identity", "alice"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["verified"] is True and report["embeddings_removed"] == 1
    assert report["snapshots_rewritten"] == 1

    events = json.loads(runner.invoke(cli.app, ["audit"]).stdout)
    assert events[0]["action"] == "erase" and events[0]["key_id"] == "cli"
    assert "alice" not in json.dumps(events)  # redacted, not just hidden
    assert json.loads(runner.invoke(cli.app, ["audit", "--identity", "alice"]).stdout) == []

    ghost = runner.invoke(cli.app, ["erase", "--identity", "alice"])
    assert ghost.exit_code == 1 and "IDENTITY_NOT_FOUND" in ghost.output


def test_encryption_migrate_encrypts_a_plaintext_store(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = env / "data" / "frp.db"
    db.parent.mkdir(parents=True)
    vectors = np.random.default_rng(1).normal(size=(1, 512)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    with IdentityStore(db) as plain:
        plain.create_identity("a", "A")
        plain.add_embeddings("a", EmbeddingBatch(vectors, FAKE_INFO))

    no_key = runner.invoke(cli.app, ["encryption", "migrate"])
    assert no_key.exit_code == 2 and "No encryption key" in no_key.output

    key = format_key("k1", generate_key())
    monkeypatch.setenv("ENCRYPTION_KEYS", key)
    out = runner.invoke(cli.app, ["encryption", "migrate"])
    assert out.exit_code == 0, out.output
    report = json.loads(out.stdout)
    assert report["embeddings_resealed"] == 1 and report["key_usage_after"] == {"k1": 1}
    assert vectors.tobytes() not in db.read_bytes()
    with IdentityStore(db, cipher=AesGcmCipher(env_provider(key, None))) as sealed:
        np.testing.assert_array_equal(sealed.embeddings_of("a")[0].vector, vectors[0])


def test_encryption_migrate_refuses_when_encryption_is_only_opted_out(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALLOW_UNENCRYPTED", "true")
    out = runner.invoke(cli.app, ["encryption", "migrate"])
    assert out.exit_code == 2 and "not configured" in out.output
