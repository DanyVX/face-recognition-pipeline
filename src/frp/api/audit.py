"""Audit log: who did what, when, with what outcome. Never any biometric content.

Rows hold the request id, the *public id* of the API key (never the secret), the action, the
identity id and a short outcome. The log is append-only through this API. Erasing an identity
*redacts* its id in earlier rows (replaced by a salted-free hash prefix) so the log keeps its
counts and timing for security review but no longer links back to the erased person.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    request_id  TEXT NOT NULL,
    key_id      TEXT,
    action      TEXT NOT NULL,
    identity_id TEXT,
    outcome     TEXT NOT NULL,
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS audit_by_identity ON audit_log(identity_id);
"""


@dataclass(frozen=True)
class AuditEvent:
    request_id: str
    key_id: str | None  # public id of the API key, never the secret ("cli" for local commands)
    action: str  # "enroll", "verify", "identify", "delete", "erase", ...
    identity_id: str | None
    outcome: str  # "ok" or an error code
    detail: str | None = None  # short, non-biometric context (for example "match"/"no_match")


AuditSink = Callable[[AuditEvent], None]


def discard(event: AuditEvent) -> None:
    """Sink used when auditing is disabled."""


def redacted_id(identity_id: str) -> str:
    return "erased:" + hashlib.sha256(identity_id.encode()).hexdigest()[:12]


class SqliteAuditLog:
    """Append-only audit log in SQLite. Usable directly as an ``AuditSink``."""

    def __init__(self, path: Path | str) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA secure_delete = ON")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def __call__(self, event: AuditEvent) -> None:
        """Record an event. A failure to audit is logged, not raised into the request."""
        try:
            with self._lock:
                self._db.execute(
                    "INSERT INTO audit_log(ts, request_id, key_id, action, identity_id, outcome, "
                    "detail) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        datetime.now(UTC).isoformat(timespec="milliseconds"),
                        event.request_id,
                        event.key_id,
                        event.action,
                        event.identity_id,
                        event.outcome,
                        event.detail,
                    ),
                )
        except sqlite3.Error:
            logger.exception("failed to write an audit event")

    def query(self, *, identity_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM audit_log"
        params: tuple[Any, ...] = ()
        if identity_id is not None:
            sql, params = sql + " WHERE identity_id = ?", (identity_id,)
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY id DESC LIMIT ?", (*params, limit)).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            (n,) = self._db.execute("SELECT COUNT(*) FROM audit_log").fetchone()
        return int(n)

    def redact_identity(self, identity_id: str) -> int:
        """Unlink ``identity_id`` from earlier rows (used by erasure); returns rows changed."""
        with self._lock:
            cur = self._db.execute(
                "UPDATE audit_log SET identity_id = ? WHERE identity_id = ?",
                (redacted_id(identity_id), identity_id),
            )
            return cur.rowcount
