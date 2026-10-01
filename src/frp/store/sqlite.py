"""SQLite identity store: identity metadata plus model-tagged embedding blobs.

Embeddings are biometric data. Rows carry the model tag (id, file hash, preprocessing version)
so vectors from different model versions can never be mixed. A ``PayloadCipher`` can be supplied
to encrypt blobs at rest (AES-GCM in M8); the blob is bound to its identity and model through the
AEAD associated data.

Erasure notes: ``secure_delete`` is on and the journal mode is the default rollback journal (not
WAL), so deleted content does not linger in a write-ahead log; ``vacuum`` rewrites the file.
"""

from __future__ import annotations

import sqlite3
import threading
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from frp.embed.base import EmbeddingBatch, ModelInfo
from frp.errors import (
    FrpError,
    IdentityExistsError,
    IdentityNotFoundError,
    ModelMismatchError,
)
from frp.index.format import PayloadCipher
from frp.store.validation import normalize_name, validate_identity_id

SCHEMA_VERSION: Final = 1

_SCHEMA = """
CREATE TABLE identities (
    rid         INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_id TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE embeddings (
    embedding_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    rid                INTEGER NOT NULL REFERENCES identities(rid) ON DELETE CASCADE,
    model_id           TEXT NOT NULL,
    model_hash         TEXT NOT NULL,
    preprocess_version TEXT NOT NULL,
    dim                INTEGER NOT NULL,
    vector             BLOB NOT NULL,
    created_at         TEXT NOT NULL
);
CREATE INDEX embeddings_by_identity ON embeddings(rid);
"""


class StoreSchemaError(FrpError):
    code = "STORE_SCHEMA_UNSUPPORTED"


@dataclass(frozen=True)
class IdentityRecord:
    rid: int  # internal integer key (also used for centroid ids in the gallery)
    identity_id: str
    name: str
    created_at: str
    updated_at: str
    n_embeddings: int


@dataclass(frozen=True, eq=False)
class EmbeddingRow:
    embedding_id: int
    rid: int
    identity_id: str
    info: ModelInfo
    vector: NDArray[np.float32]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _aad(identity_id: str, info: ModelInfo) -> bytes:
    return f"emb|{identity_id}|{info.model_hash}|{info.preprocess_version}".encode()


class IdentityStore:
    """Thread-safe SQLite store (one connection, serialized by a lock)."""

    def __init__(self, path: Path | str, *, cipher: PayloadCipher | None = None) -> None:
        self._path = str(path)
        self._cipher = cipher
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA secure_delete = ON")
        self._db.execute("PRAGMA synchronous = FULL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            (version,) = self._db.execute("PRAGMA user_version").fetchone()
            if version == 0:
                self._db.executescript(
                    "BEGIN;" + _SCHEMA + f"PRAGMA user_version={SCHEMA_VERSION};COMMIT;"
                )
            elif version != SCHEMA_VERSION:
                raise StoreSchemaError(
                    f"Store schema version {version} is not supported (this build uses "
                    f"{SCHEMA_VERSION}). Export your data and re-create the store."
                )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> IdentityStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- identities --------------------------------------------------------------------------

    def _record(self, row: sqlite3.Row) -> IdentityRecord:
        return IdentityRecord(
            rid=row["rid"],
            identity_id=row["identity_id"],
            name=row["name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            n_embeddings=row["n_embeddings"],
        )

    _SELECT = (
        "SELECT i.rid, i.identity_id, i.name, i.created_at, i.updated_at, "
        "(SELECT COUNT(*) FROM embeddings e WHERE e.rid = i.rid) AS n_embeddings "
        "FROM identities i"
    )

    def create_identity(self, identity_id: str, name: str) -> IdentityRecord:
        identity_id = validate_identity_id(identity_id)
        name = normalize_name(name)
        now = _now()
        with self._lock:
            try:
                self._db.execute(
                    "INSERT INTO identities(identity_id, name, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (identity_id, name, now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise IdentityExistsError(f"Identity {identity_id!r} already exists.") from exc
            return self.get_identity(identity_id)

    def get_identity(self, identity_id: str) -> IdentityRecord:
        with self._lock:
            row = self._db.execute(
                self._SELECT + " WHERE i.identity_id = ?", (identity_id,)
            ).fetchone()
        if row is None:
            raise IdentityNotFoundError(f"Identity {identity_id!r} not found.")
        return self._record(row)

    def has_identity(self, identity_id: str) -> bool:
        with self._lock:
            return (
                self._db.execute(
                    "SELECT 1 FROM identities WHERE identity_id = ?", (identity_id,)
                ).fetchone()
                is not None
            )

    def list_identities(self, limit: int = 100, offset: int = 0) -> list[IdentityRecord]:
        if limit < 0 or offset < 0:
            raise ValueError("limit and offset must be non-negative")
        with self._lock:
            rows = self._db.execute(
                self._SELECT + " ORDER BY i.rid LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return [self._record(r) for r in rows]

    def count_identities(self) -> int:
        with self._lock:
            (n,) = self._db.execute("SELECT COUNT(*) FROM identities").fetchone()
        return int(n)

    def rename_identity(self, identity_id: str, name: str) -> IdentityRecord:
        name = normalize_name(name)
        with self._lock:
            cur = self._db.execute(
                "UPDATE identities SET name = ?, updated_at = ? WHERE identity_id = ?",
                (name, _now(), identity_id),
            )
            if cur.rowcount == 0:
                raise IdentityNotFoundError(f"Identity {identity_id!r} not found.")
            return self.get_identity(identity_id)

    def delete_identity(self, identity_id: str) -> list[int]:
        """Delete an identity and all its embeddings; returns the removed embedding ids."""
        with self._lock:
            record = self.get_identity(identity_id)
            ids = [
                int(r["embedding_id"])
                for r in self._db.execute(
                    "SELECT embedding_id FROM embeddings WHERE rid = ?", (record.rid,)
                )
            ]
            self._db.execute("BEGIN")
            try:
                # Overwrite blobs first so the plaintext/ciphertext bytes are not left in pages.
                self._db.execute(
                    "UPDATE embeddings SET vector = zeroblob(length(vector)) WHERE rid = ?",
                    (record.rid,),
                )
                self._db.execute("DELETE FROM embeddings WHERE rid = ?", (record.rid,))
                self._db.execute("DELETE FROM identities WHERE rid = ?", (record.rid,))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return ids

    # -- embeddings --------------------------------------------------------------------------

    def add_embeddings(self, identity_id: str, batch: EmbeddingBatch) -> list[int]:
        """Store each vector of ``batch`` for ``identity_id``; returns the new embedding ids."""
        info = batch.info
        with self._lock:
            record = self.get_identity(identity_id)
            now = _now()
            new_ids: list[int] = []
            self._db.execute("BEGIN")
            try:
                for vector in batch.vectors:
                    raw = np.asarray(vector, dtype="<f4").tobytes()
                    if self._cipher is not None:
                        raw = self._cipher.encrypt(raw, _aad(identity_id, info))
                    cur = self._db.execute(
                        "INSERT INTO embeddings(rid, model_id, model_hash, preprocess_version, "
                        "dim, vector, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            record.rid,
                            info.model_id,
                            info.model_hash,
                            info.preprocess_version,
                            info.dim,
                            raw,
                            now,
                        ),
                    )
                    if cur.lastrowid is None:  # pragma: no cover - sqlite always sets it
                        raise RuntimeError("SQLite did not return a row id")
                    new_ids.append(int(cur.lastrowid))
                self._db.execute(
                    "UPDATE identities SET updated_at = ? WHERE rid = ?", (now, record.rid)
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return new_ids

    def delete_embeddings(self, embedding_ids: list[int]) -> int:
        with self._lock:
            removed = 0
            self._db.execute("BEGIN")
            try:
                for eid in embedding_ids:
                    self._db.execute(
                        "UPDATE embeddings SET vector = zeroblob(length(vector)) "
                        "WHERE embedding_id = ?",
                        (eid,),
                    )
                    removed += self._db.execute(
                        "DELETE FROM embeddings WHERE embedding_id = ?", (eid,)
                    ).rowcount
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return removed

    def _row_to_embedding(self, row: sqlite3.Row) -> EmbeddingRow:
        info = ModelInfo(
            model_id=row["model_id"],
            model_hash=row["model_hash"],
            preprocess_version=row["preprocess_version"],
            dim=int(row["dim"]),
        )
        raw: bytes = row["vector"]
        if self._cipher is not None:
            raw = self._cipher.decrypt(raw, _aad(row["identity_id"], info))
        vector = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        if vector.shape[0] != info.dim:
            raise ModelMismatchError(
                f"Stored embedding {row['embedding_id']} has {vector.shape[0]} values but its "
                f"model tag says dimension {info.dim}."
            )
        return EmbeddingRow(
            embedding_id=int(row["embedding_id"]),
            rid=int(row["rid"]),
            identity_id=str(row["identity_id"]),
            info=info,
            vector=vector,
        )

    _EMB_SELECT = (
        "SELECT e.embedding_id, e.rid, i.identity_id, e.model_id, e.model_hash, "
        "e.preprocess_version, e.dim, e.vector FROM embeddings e "
        "JOIN identities i ON i.rid = e.rid"
    )

    def embeddings_of(self, identity_id: str) -> list[EmbeddingRow]:
        with self._lock:
            self.get_identity(identity_id)
            rows = self._db.execute(
                self._EMB_SELECT + " WHERE i.identity_id = ? ORDER BY e.embedding_id",
                (identity_id,),
            ).fetchall()
            return [self._row_to_embedding(r) for r in rows]

    def iter_embeddings(self) -> Iterator[EmbeddingRow]:
        """All embeddings in id order (used to rebuild the in-memory index at startup)."""
        with self._lock:
            rows = self._db.execute(self._EMB_SELECT + " ORDER BY e.embedding_id").fetchall()
            decoded = [self._row_to_embedding(r) for r in rows]
        yield from decoded

    def count_embeddings(self) -> int:
        with self._lock:
            (n,) = self._db.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return int(n)

    def model_tags(self) -> Counter[tuple[str, str, str, int]]:
        """How many stored embeddings exist per (model_id, model_hash, preprocess, dim)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT model_id, model_hash, preprocess_version, dim, COUNT(*) AS n "
                "FROM embeddings GROUP BY model_id, model_hash, preprocess_version, dim"
            ).fetchall()
        return Counter(
            {
                (r["model_id"], r["model_hash"], r["preprocess_version"], int(r["dim"])): int(
                    r["n"]
                )
                for r in rows
            }
        )

    # -- maintenance -------------------------------------------------------------------------

    def vacuum(self) -> None:
        """Rewrite the database file, discarding free pages (part of erasure)."""
        with self._lock:
            self._db.execute("VACUUM")
