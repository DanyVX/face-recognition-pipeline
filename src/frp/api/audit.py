"""Audit events: who did what, with no biometric content. The sink is wired in M8."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class AuditEvent:
    request_id: str
    key_id: str | None  # public id of the API key, never the secret
    action: str  # "enroll", "verify", "identify", "delete", ...
    identity_id: str | None
    outcome: str  # "ok" or an error code
    detail: str | None = None  # short, non-biometric context (for example a score band)


AuditSink = Callable[[AuditEvent], None]


def discard(event: AuditEvent) -> None:
    """Default sink: audit is disabled until a sink is configured."""
