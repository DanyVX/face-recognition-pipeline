"""On-disk format for a gallery index snapshot.

Layout (all integers little-endian)::

    magic       8 bytes   b"FRPIDX\\x00\\x01"   (the last byte is the container version)
    header_len  4 bytes   length of the JSON header
    header      JSON      format_version, backend, params, model info, count, encryption, ...
    payload     bytes     ids (int64 * N) then vectors (float32 * N * D); AEAD-encrypted when
                          a cipher is supplied
    trailer     32 bytes  SHA-256 over everything above

Writes are atomic (temp file in the same directory, fsync, rename), so a crash can never leave a
half-written index in place. A load verifies the trailer before parsing anything else, then
validates counts, dimensions, finiteness, uniqueness and unit norm. Unknown versions are refused
with an explicit error (no silent migration).
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
from numpy.typing import NDArray

from frp.embed.base import ModelInfo, check_unit_norm
from frp.errors import EmbeddingError, IndexCorruptError, IndexVersionError

MAGIC: Final = b"FRPIDX\x00\x01"
FORMAT_VERSION: Final = 1
_TRAILER: Final = 32
_MAX_HEADER: Final = 1 << 20


class PayloadCipher(Protocol):
    """Authenticated encryption for the payload (implemented with AES-GCM in M8)."""

    scheme: str

    def encrypt(self, plaintext: bytes, aad: bytes) -> bytes: ...

    def decrypt(self, blob: bytes, aad: bytes) -> bytes: ...


@dataclass(frozen=True, eq=False)
class IndexSnapshot:
    """Everything stored in an index file."""

    info: ModelInfo
    backend: str
    params: dict[str, Any]
    ids: NDArray[np.int64]
    vectors: NDArray[np.float32]


def _header_bytes(snapshot: IndexSnapshot, encryption: dict[str, Any] | None) -> bytes:
    header = {
        "format_version": FORMAT_VERSION,
        "backend": snapshot.backend,
        "params": snapshot.params,
        "model": {
            "model_id": snapshot.info.model_id,
            "model_hash": snapshot.info.model_hash,
            "preprocess_version": snapshot.info.preprocess_version,
            "dim": snapshot.info.dim,
        },
        "count": int(snapshot.ids.shape[0]),
        "encryption": encryption,
    }
    return json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _aad(header: bytes) -> bytes:
    """Bind the ciphertext to its header so the two cannot be mixed and matched."""
    return hashlib.sha256(header).digest()


def serialize(snapshot: IndexSnapshot, cipher: PayloadCipher | None = None) -> bytes:
    ids = np.ascontiguousarray(snapshot.ids, dtype="<i8")
    vectors = np.ascontiguousarray(snapshot.vectors, dtype="<f4")
    if vectors.shape != (ids.shape[0], snapshot.info.dim):
        raise IndexCorruptError("Refusing to write inconsistent ids/vectors/dimension.")
    payload = ids.tobytes() + vectors.tobytes()
    encryption = None if cipher is None else {"scheme": cipher.scheme}
    header = _header_bytes(snapshot, encryption)
    if cipher is not None:
        payload = cipher.encrypt(payload, _aad(header))
    body = MAGIC + struct.pack("<I", len(header)) + header + payload
    return body + hashlib.sha256(body).digest()


def deserialize(data: bytes, cipher: PayloadCipher | None = None) -> IndexSnapshot:
    if len(data) < len(MAGIC) + 4 + _TRAILER:
        raise IndexCorruptError("Index file is too short (truncated?).")
    body, trailer = data[:-_TRAILER], data[-_TRAILER:]
    if data[: len(MAGIC) - 1] != MAGIC[:-1]:
        raise IndexCorruptError("Not an index file (bad magic bytes).")
    if hashlib.sha256(body).digest() != trailer:
        raise IndexCorruptError("Index file checksum mismatch (corrupted or partially written).")
    if body[: len(MAGIC)] != MAGIC:
        raise IndexVersionError(
            f"Unsupported index container version {body[len(MAGIC) - 1]}; this build reads "
            f"version {MAGIC[-1]}. Rebuild the index from the identity store."
        )
    (header_len,) = struct.unpack("<I", body[len(MAGIC) : len(MAGIC) + 4])
    start = len(MAGIC) + 4
    if header_len > _MAX_HEADER or start + header_len > len(body):
        raise IndexCorruptError("Index header length is invalid.")
    header_raw = body[start : start + header_len]
    payload = body[start + header_len :]
    try:
        header = json.loads(header_raw)
        version = header["format_version"]
        model = header["model"]
        count = int(header["count"])
    except (ValueError, KeyError, TypeError) as exc:
        raise IndexCorruptError("Index header is not valid.") from exc
    if version != FORMAT_VERSION:
        raise IndexVersionError(
            f"Unsupported index format version {version}; this build reads version "
            f"{FORMAT_VERSION}. Rebuild the index from the identity store."
        )
    encrypted = header.get("encryption")
    if encrypted is not None:
        if cipher is None:
            raise IndexCorruptError("Index is encrypted but no key was provided.")
        if encrypted.get("scheme") != cipher.scheme:
            raise IndexCorruptError("Index was encrypted with a different scheme.")
        try:
            payload = cipher.decrypt(payload, _aad(header_raw))
        except Exception as exc:  # AEAD failure: wrong key or tampering
            raise IndexCorruptError("Index payload failed authenticated decryption.") from exc
    elif cipher is not None:
        raise IndexCorruptError("Index is not encrypted but a cipher was supplied.")
    try:
        info = ModelInfo(
            model_id=str(model["model_id"]),
            model_hash=str(model["model_hash"]),
            preprocess_version=str(model["preprocess_version"]),
            dim=int(model["dim"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IndexCorruptError("Index model info is not valid.") from exc
    expected = count * 8 + count * info.dim * 4
    if count < 0 or info.dim < 1 or len(payload) != expected:
        raise IndexCorruptError(
            f"Index payload size {len(payload)} does not match {count} vectors of "
            f"dimension {info.dim} (expected {expected})."
        )
    ids = np.frombuffer(payload[: count * 8], dtype="<i8").astype(np.int64)
    vectors = (
        np.frombuffer(payload[count * 8 :], dtype="<f4").reshape(count, info.dim).astype(np.float32)
    )
    if np.unique(ids).shape[0] != count:
        raise IndexCorruptError("Index contains duplicate vector ids.")
    try:
        check_unit_norm(vectors)
    except EmbeddingError as exc:
        raise IndexCorruptError(f"Stored vectors are invalid: {exc}") from exc
    return IndexSnapshot(
        info=info,
        backend=str(header.get("backend", "flat")),
        params=dict(header.get("params", {})),
        ids=ids,
        vectors=vectors,
    )


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file in the same directory and an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        Path(tmp_name).replace(path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def save_snapshot(path: Path, snapshot: IndexSnapshot, cipher: PayloadCipher | None = None) -> None:
    write_atomic(path, serialize(snapshot, cipher))


def load_snapshot(path: Path, cipher: PayloadCipher | None = None) -> IndexSnapshot:
    return deserialize(path.read_bytes(), cipher)
