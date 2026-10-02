"""Encryption at rest: AES-256-GCM with a key-provider abstraction and key rotation.

Embeddings are biometric data, so every stored embedding (database blobs and index snapshots) is
sealed with AES-256-GCM. Each blob records the id of the key that sealed it, so a keyring can hold
several keys: new data uses the *active* key while old data stays readable until it is rotated.

Blob layout::

    version (1 byte, 0x01) | key id length (1 byte) | key id (utf-8) | nonce (12 bytes) | ct + tag

A fresh random 96-bit nonce is drawn for every blob. With random nonces a single key should not
seal more than about 2**32 blobs; rotate keys long before that (see docs/SECURITY_OPERATIONS.md).
The caller passes *associated data* (for example the identity id) which is authenticated but not
stored, binding a blob to its row so it cannot be moved to another identity.

The key provider is a small protocol so a KMS-style backend can replace the env/file providers
without touching the rest of the code. Keys are never logged.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import secrets
import stat
from pathlib import Path
from typing import Final, Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from frp.errors import FrpError

VERSION: Final = 1
KEY_BYTES: Final = 32
NONCE_BYTES: Final = 12
_KEY_ID: Final = re.compile(r"^[A-Za-z0-9._-]{1,32}$")


class KeyConfigError(FrpError):
    """Encryption keys are missing or malformed."""

    code = "ENCRYPTION_KEY_ERROR"
    http_status = 500


class DecryptionError(FrpError):
    """A blob could not be decrypted: wrong key, unknown key id, or tampering."""

    code = "DECRYPTION_FAILED"
    http_status = 500


class KeyProvider(Protocol):
    """Source of encryption keys (env, file, or a KMS adapter)."""

    def active_key_id(self) -> str: ...

    def get_key(self, key_id: str) -> bytes: ...

    def key_ids(self) -> list[str]: ...


def generate_key() -> bytes:
    return secrets.token_bytes(KEY_BYTES)


def _validate(key_id: str, key: bytes) -> None:
    if not _KEY_ID.fullmatch(key_id):
        raise KeyConfigError("Key ids must be 1-32 characters from A-Z a-z 0-9 . _ -")
    if len(key) != KEY_BYTES:
        raise KeyConfigError(f"Key {key_id!r} must be exactly {KEY_BYTES} bytes (AES-256).")


def _decode_key(key_id: str, encoded: str) -> bytes:
    try:
        key = base64.b64decode(encoded.strip(), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise KeyConfigError(f"Key {key_id!r} is not valid base64.") from exc
    _validate(key_id, key)
    return key


class StaticKeyProvider:
    """An in-memory keyring with one active key."""

    def __init__(self, keys: dict[str, bytes], active: str) -> None:
        if not keys:
            raise KeyConfigError("The keyring is empty.")
        for key_id, key in keys.items():
            _validate(key_id, key)
        if active not in keys:
            raise KeyConfigError(f"Active key {active!r} is not in the keyring.")
        self._keys = dict(keys)
        self._active = active

    def active_key_id(self) -> str:
        return self._active

    def get_key(self, key_id: str) -> bytes:
        try:
            return self._keys[key_id]
        except KeyError:
            raise DecryptionError(f"Unknown encryption key id {key_id!r}.") from None

    def key_ids(self) -> list[str]:
        return sorted(self._keys)

    def __repr__(self) -> str:  # never print key material
        return f"StaticKeyProvider(ids={self.key_ids()}, active={self._active!r})"


def parse_keyring(spec: str) -> dict[str, bytes]:
    """Parse ``"id1:BASE64,id2:BASE64"`` (the ENCRYPTION_KEYS format)."""
    keys: dict[str, bytes] = {}
    for item in (part for part in spec.split(",") if part.strip()):
        key_id, sep, encoded = item.strip().partition(":")
        if not sep:
            raise KeyConfigError("ENCRYPTION_KEYS entries look like 'key_id:BASE64KEY'.")
        if key_id in keys:
            raise KeyConfigError(f"Duplicate key id {key_id!r}.")
        keys[key_id] = _decode_key(key_id, encoded)
    return keys


def env_provider(keys_spec: str, active: str | None) -> StaticKeyProvider:
    """Keyring from ``ENCRYPTION_KEYS``; ``active`` defaults to the first key listed."""
    keys = parse_keyring(keys_spec)
    if not keys:
        raise KeyConfigError("ENCRYPTION_KEYS is empty.")
    return StaticKeyProvider(keys, active or next(iter(keys)))


def format_key(key_id: str, key: bytes) -> str:
    return f"{key_id}:{base64.b64encode(key).decode('ascii')}"


def file_provider(path: Path, *, generate_if_missing: bool = False) -> StaticKeyProvider:
    """Keyring from a JSON file ``{"active": "k1", "keys": {"k1": "BASE64"}}``.

    ``generate_if_missing`` creates a fresh single-key file (mode 0600): intended for demos only,
    because a key stored next to the data does not protect against theft of the data volume.
    """
    if not path.exists():
        if not generate_if_missing:
            raise KeyConfigError(f"Encryption key file not found: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "active": "k1",
            "keys": {"k1": base64.b64encode(generate_key()).decode("ascii")},
        }
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
    elif os.name == "posix" and path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise KeyConfigError(
            f"Key file {path} must not be readable by group or others (chmod 600)."
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        keys = {k: _decode_key(k, v) for k, v in document["keys"].items()}
        active = str(document["active"])
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise KeyConfigError(f"Key file {path} is not valid.") from exc
    return StaticKeyProvider(keys, active)


def blob_key_id(blob: bytes) -> str:
    """The id of the key that sealed ``blob`` (without decrypting it)."""
    if len(blob) < 2 or blob[0] != VERSION or len(blob) < 2 + blob[1] + NONCE_BYTES:
        raise DecryptionError("Not an encrypted blob.")
    return blob[2 : 2 + blob[1]].decode("utf-8", errors="replace")


class AesGcmCipher:
    """``PayloadCipher`` implementation: AES-256-GCM sealed with the provider's active key."""

    scheme = "aes-256-gcm"

    def __init__(self, provider: KeyProvider) -> None:
        self.provider = provider

    def encrypt(self, plaintext: bytes, aad: bytes) -> bytes:
        key_id = self.provider.active_key_id()
        key_id_bytes = key_id.encode("utf-8")
        nonce = os.urandom(NONCE_BYTES)
        sealed = AESGCM(self.provider.get_key(key_id)).encrypt(nonce, plaintext, aad)
        return bytes([VERSION, len(key_id_bytes)]) + key_id_bytes + nonce + sealed

    def decrypt(self, blob: bytes, aad: bytes) -> bytes:
        key_id = blob_key_id(blob)
        start = 2 + blob[1]
        nonce, sealed = blob[start : start + NONCE_BYTES], blob[start + NONCE_BYTES :]
        try:
            return AESGCM(self.provider.get_key(key_id)).decrypt(nonce, sealed, aad)
        except InvalidTag:
            raise DecryptionError(
                "Decryption failed: wrong key, wrong context, or the data was modified."
            ) from None
