"""AES-GCM sealing, keyrings, providers and rotation primitives."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from frp.crypto import (
    NONCE_BYTES,
    AesGcmCipher,
    DecryptionError,
    KeyConfigError,
    StaticKeyProvider,
    blob_key_id,
    env_provider,
    file_provider,
    format_key,
    generate_key,
    parse_keyring,
)


def cipher(**keys: bytes) -> AesGcmCipher:
    ring = keys or {"k1": generate_key()}
    return AesGcmCipher(StaticKeyProvider(ring, next(iter(ring))))


# --- sealing -------------------------------------------------------------------------------------


def test_round_trip_and_the_plaintext_is_not_visible() -> None:
    c = cipher()
    secret = b"embedding-bytes-" * 32
    blob = c.encrypt(secret, b"alice")
    assert secret not in blob and secret[:16] not in blob
    assert c.decrypt(blob, b"alice") == secret


@given(st.binary(max_size=2000), st.binary(max_size=64))
def test_round_trip_property(data: bytes, aad: bytes) -> None:
    c = AesGcmCipher(StaticKeyProvider({"k": b"\x01" * 32}, "k"))
    assert c.decrypt(c.encrypt(data, aad), aad) == data


def test_every_encryption_uses_a_fresh_nonce() -> None:
    c = cipher()
    blobs = [c.encrypt(b"same plaintext", b"ctx") for _ in range(200)]
    nonces = {b[2 + b[1] : 2 + b[1] + NONCE_BYTES] for b in blobs}
    assert len(nonces) == 200 and len(set(blobs)) == 200  # identical input never repeats


def test_associated_data_binds_a_blob_to_its_context() -> None:
    c = cipher()
    blob = c.encrypt(b"vector", b"identity:alice")
    with pytest.raises(DecryptionError, match="wrong key, wrong context"):
        c.decrypt(blob, b"identity:bob")  # moved to another identity


def test_any_modification_is_detected() -> None:
    c = cipher()
    blob = bytearray(c.encrypt(b"vector-bytes", b"ctx"))
    for position in range(len(blob)):
        tampered = bytearray(blob)
        tampered[position] ^= 0x01
        with pytest.raises(DecryptionError):
            c.decrypt(bytes(tampered), b"ctx")
    with pytest.raises(DecryptionError):
        c.decrypt(bytes(blob[:-1]), b"ctx")  # truncated


def test_wrong_key_and_garbage_blobs_are_refused() -> None:
    sealed = cipher(k1=b"\x01" * 32).encrypt(b"x", b"a")
    with pytest.raises(DecryptionError, match="Unknown encryption key id"):
        cipher(other=b"\x02" * 32).decrypt(sealed, b"a")
    same_id_other_key = AesGcmCipher(StaticKeyProvider({"k1": b"\x09" * 32}, "k1"))
    with pytest.raises(DecryptionError, match="wrong key"):
        same_id_other_key.decrypt(sealed, b"a")
    for junk in (b"", b"\x01", b"plain text blob", b"\x02\x02k1" + b"0" * 40):
        with pytest.raises(DecryptionError):
            cipher().decrypt(junk, b"a")


def test_scheme_name_is_stable() -> None:
    assert cipher().scheme == "aes-256-gcm"


# --- rotation: old data stays readable while new data uses the active key -----------------------


def test_old_keys_keep_decrypting_after_the_active_key_changes() -> None:
    old, new = b"\x01" * 32, b"\x02" * 32
    before = AesGcmCipher(StaticKeyProvider({"k1": old}, "k1"))
    sealed_old = before.encrypt(b"payload", b"ctx")
    after = AesGcmCipher(StaticKeyProvider({"k1": old, "k2": new}, "k2"))
    assert blob_key_id(sealed_old) == "k1"
    assert after.decrypt(sealed_old, b"ctx") == b"payload"  # still readable
    sealed_new = after.encrypt(b"payload", b"ctx")
    assert blob_key_id(sealed_new) == "k2"  # new writes use the active key
    assert before.provider.key_ids() == ["k1"]
    with pytest.raises(DecryptionError):
        before.decrypt(sealed_new, b"ctx")  # a keyring without k2 cannot read it


# --- keyrings and providers ----------------------------------------------------------------------


def test_parse_keyring_and_env_provider() -> None:
    k1, k2 = generate_key(), generate_key()
    spec = f"{format_key('a', k1)}, {format_key('b', k2)}"
    assert parse_keyring(spec) == {"a": k1, "b": k2}
    provider = env_provider(spec, None)
    assert provider.active_key_id() == "a" and provider.key_ids() == ["a", "b"]
    assert env_provider(spec, "b").active_key_id() == "b"
    assert "a" in repr(provider) and base64.b64encode(k1).decode() not in repr(provider)


@pytest.mark.parametrize(
    "spec",
    [
        "no-colon",
        "k:not-base64!!",
        f"k:{base64.b64encode(b'short').decode()}",
        f"bad id:{base64.b64encode(b'0' * 32).decode()}",
        f"k:{base64.b64encode(b'0' * 32).decode()},k:{base64.b64encode(b'1' * 32).decode()}",
    ],
)
def test_invalid_keyrings_fail_with_a_clear_error(spec: str) -> None:
    with pytest.raises(KeyConfigError):
        parse_keyring(spec)


def test_empty_keyring_and_unknown_active_key_are_errors() -> None:
    with pytest.raises(KeyConfigError):
        env_provider("", None)
    with pytest.raises(KeyConfigError, match="not in the keyring"):
        env_provider(format_key("a", generate_key()), "zzz")
    with pytest.raises(KeyConfigError):
        StaticKeyProvider({}, "a")


def test_key_file_provider_generates_loads_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    with pytest.raises(KeyConfigError, match="not found"):
        file_provider(path)
    created = file_provider(path, generate_if_missing=True)
    assert created.active_key_id() == "k1" and path.is_file()
    again = file_provider(path)
    assert again.get_key("k1") == created.get_key("k1")  # same key on reload
    if os.name == "posix":
        assert path.stat().st_mode & 0o077 == 0
        path.chmod(0o644)
        with pytest.raises(KeyConfigError, match="chmod 600"):
            file_provider(path)
        path.chmod(0o600)
    path.write_text(json.dumps({"active": "k1"}), encoding="utf-8")
    with pytest.raises(KeyConfigError, match="not valid"):
        file_provider(path)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(KeyConfigError):
        file_provider(path)


def test_generated_keys_are_unique_and_the_right_size() -> None:
    keys = {generate_key() for _ in range(100)}
    assert len(keys) == 100 and all(len(k) == 32 for k in keys)
