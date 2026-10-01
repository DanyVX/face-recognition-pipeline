"""Idempotency keys for enrollment: a retried request replays the first response.

Behavior for a client-chosen ``Idempotency-Key`` (scoped per API key):
* first request: runs and, if it succeeded (2xx), its response is stored;
* same key and same request content: the stored response is replayed without running again;
* same key but different content: ``IDEMPOTENCY_KEY_REUSED`` (422);
* identical requests arriving *concurrently*: the second waits for the first, then replays.

Stored responses contain no embeddings or images. The identity id is recorded so that erasure can
purge the replay record of a deleted identity.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from frp.api.errors import IdempotencyKeyReusedError, InvalidRequestError

_KEY_PATTERN: Final = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS idempotency (
    scope        TEXT NOT NULL,
    key          TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    identity_id  TEXT,
    status       INTEGER NOT NULL,
    body         TEXT NOT NULL,
    created      REAL NOT NULL,
    PRIMARY KEY (scope, key)
);
"""


class _KeyedLocks:
    """One lock per key, created on demand and discarded when nobody holds or awaits it."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, tuple[threading.Lock, int]] = {}

    def acquire(self, key: str) -> None:
        with self._guard:
            lock, users = self._locks.get(key, (threading.Lock(), 0))
            self._locks[key] = (lock, users + 1)
        lock.acquire()

    def release(self, key: str) -> None:
        with self._guard:
            lock, users = self._locks[key]
            if users <= 1:
                del self._locks[key]
            else:
                self._locks[key] = (lock, users - 1)
        lock.release()


def validate_key(key: str) -> str:
    if not _KEY_PATTERN.fullmatch(key):
        raise InvalidRequestError(
            "Idempotency-Key must be 1-128 characters from A-Z a-z 0-9 . _ : -"
        )
    return key


class IdempotencyStore:
    def __init__(
        self, path: Path | str, ttl_seconds: int, *, clock: Callable[[], float] = time.time
    ) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._db_lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA secure_delete = ON")
        self._db.execute(_SCHEMA)
        self._locks = _KeyedLocks()

    def close(self) -> None:
        self._db.close()

    def run(
        self,
        scope: str,
        key: str,
        request_hash: str,
        producer: Callable[[], tuple[int, dict[str, Any], str | None]],
    ) -> tuple[int, dict[str, Any], bool]:
        """Run ``producer`` once per (scope, key). Returns ``(status, body, replayed)``.

        ``producer`` returns ``(status, body, identity_id)``; only 2xx results are stored.
        """
        validate_key(key)
        lock_key = f"{scope}\x00{key}"
        self._locks.acquire(lock_key)
        try:
            with self._db_lock:
                self._db.execute(
                    "DELETE FROM idempotency WHERE created < ?", (self._clock() - self._ttl,)
                )
                row = self._db.execute(
                    "SELECT request_hash, status, body FROM idempotency "
                    "WHERE scope = ? AND key = ?",
                    (scope, key),
                ).fetchone()
            if row is not None:
                if row[0] != request_hash:
                    raise IdempotencyKeyReusedError(
                        "This Idempotency-Key was already used with a different request."
                    )
                return int(row[1]), json.loads(row[2]), True
            status, body, identity_id = producer()
            if 200 <= status < 300:
                with self._db_lock:
                    self._db.execute(
                        "INSERT INTO idempotency(scope, key, request_hash, identity_id, "
                        "status, body, created) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            scope,
                            key,
                            request_hash,
                            identity_id,
                            status,
                            json.dumps(body),
                            self._clock(),
                        ),
                    )
            return status, body, False
        finally:
            self._locks.release(lock_key)

    def purge_identity(self, identity_id: str) -> int:
        """Delete replay records that mention ``identity_id`` (used by erasure)."""
        with self._db_lock:
            cur = self._db.execute("DELETE FROM idempotency WHERE identity_id = ?", (identity_id,))
            return cur.rowcount
