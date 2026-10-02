"""API keys: created from the CLI, stored only as hashes, checked in constant time.

Key format ``frp_<id>_<secret>``: ``id`` (8 hex characters) is public and identifies the key in
logs, metrics and rate limiting; ``secret`` (256 random bits) is shown once at creation. Only a
SHA-256 digest of the secret is stored. A fast hash is appropriate here because the secret is
high-entropy random data (unlike a human password, it cannot be guessed or brute-forced offline).
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

_KEY_PATTERN: Final = re.compile(r"^frp_([0-9a-f]{8})_([A-Za-z0-9_-]{43})$")
_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS api_keys (
    key_id      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    secret_hash TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    revoked_at  TEXT
);
"""


@dataclass(frozen=True)
class ApiKeyRecord:
    key_id: str
    name: str
    created_at: str
    revoked: bool


def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("ascii")).hexdigest()


class ApiKeyStore:
    def __init__(self, path: Path | str) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def create(self, name: str) -> tuple[str, str]:
        """Create a key; returns ``(key_id, full_key)``. The full key is never stored."""
        name = name.strip()
        if not name or len(name) > 100:
            raise ValueError("key name must be 1-100 characters")
        key_id = secrets.token_hex(4)
        secret = secrets.token_urlsafe(32)
        with self._lock:
            self._db.execute(
                "INSERT INTO api_keys(key_id, name, secret_hash, created_at) VALUES (?, ?, ?, ?)",
                (key_id, name, _digest(secret), datetime.now(UTC).isoformat(timespec="seconds")),
            )
        return key_id, f"frp_{key_id}_{secret}"

    def verify(self, presented: str | None) -> ApiKeyRecord | None:
        """The matching active key, or ``None``. Compares digests in constant time."""
        match = _KEY_PATTERN.fullmatch(presented or "")
        row = None
        if match:
            with self._lock:
                row = self._db.execute(
                    "SELECT * FROM api_keys WHERE key_id = ?", (match.group(1),)
                ).fetchone()
        # Compare against a dummy digest when the id is unknown so timing does not reveal it.
        stored = row["secret_hash"] if row is not None else "0" * 64
        presented_digest = _digest(match.group(2)) if match else "1" * 64
        equal = hmac.compare_digest(stored, presented_digest)
        if row is None or not equal or row["revoked_at"] is not None:
            return None
        return ApiKeyRecord(row["key_id"], row["name"], row["created_at"], False)

    def list(self) -> list[ApiKeyRecord]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM api_keys ORDER BY created_at, key_id").fetchall()
        return [
            ApiKeyRecord(r["key_id"], r["name"], r["created_at"], r["revoked_at"] is not None)
            for r in rows
        ]

    def revoke(self, key_id: str) -> bool:
        with self._lock:
            cur = self._db.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE key_id = ? AND revoked_at IS NULL",
                (datetime.now(UTC).isoformat(timespec="seconds"), key_id),
            )
        return cur.rowcount > 0
