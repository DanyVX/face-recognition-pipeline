"""Right to erasure: remove a person's data everywhere this system keeps it, then verify.

What is removed:
* the identity row and all embedding rows (blobs are overwritten with zeros first, and the
  database file is vacuumed so free pages do not keep the bytes);
* the vectors in the in-memory index;
* encrypted raw images (overwritten, then deleted);
* the vectors in every *registered* index snapshot (each is rewritten without them);
* replay records in the idempotency store;
* the identity id in earlier audit rows (redacted; the erase event itself is recorded).

What this cannot reach: copies the system does not know about (database or volume backups, disk
snapshots, files copied elsewhere, a running replica's memory). Erase from those separately, or
keep their retention short; see docs/SECURITY_OPERATIONS.md. Erasing while a *service process* is
running must go through that process (the HTTP DELETE endpoint does exactly this), because its
in-memory index would otherwise still match the person until restart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from frp.api.audit import AuditEvent, SqliteAuditLog, redacted_id
from frp.errors import FrpError
from frp.index.format import IndexSnapshot, load_snapshot, save_snapshot

if TYPE_CHECKING:
    from frp.api.idempotency import IdempotencyStore
    from frp.service import RecognitionService


class ErasureVerificationError(FrpError):
    """Erasure ran but a verification check failed; the report names what remains."""

    code = "ERASURE_NOT_VERIFIED"
    http_status = 500


@dataclass(frozen=True)
class ErasureReport:
    identity_id: str
    embeddings_removed: int
    raw_images_removed: int
    snapshots_rewritten: int
    snapshots_missing: int
    idempotency_records_purged: int
    audit_rows_redacted: int
    verification: dict[str, bool]
    notes: dict[str, bool]  # informational checks that can false-positive (substring matches)

    @property
    def verified(self) -> bool:
        return all(self.verification.values())


def _rewrite_snapshots(service: RecognitionService, removed_ids: list[int]) -> tuple[int, int]:
    """Drop ``removed_ids`` from every registered snapshot. Returns (rewritten, missing)."""
    store = service.gallery.store
    rewritten = missing = 0
    for raw_path in store.list_snapshots():
        path = Path(raw_path)
        if not path.is_file():
            store.forget_snapshot(raw_path)
            missing += 1
            continue
        try:
            snapshot = load_snapshot(path, store.cipher)
        except FrpError as exc:
            raise ErasureVerificationError(
                f"Registered snapshot {path} cannot be read ({exc.code}); delete it manually."
            ) from exc
        keep = ~np.isin(snapshot.ids, np.asarray(removed_ids, dtype=np.int64))
        if keep.all():
            continue
        save_snapshot(
            path,
            IndexSnapshot(
                info=snapshot.info,
                backend=snapshot.backend,
                params=snapshot.params,
                ids=snapshot.ids[keep],
                vectors=snapshot.vectors[keep],
            ),
            store.cipher,
        )
        rewritten += 1
    return rewritten, missing


def _snapshots_clean(service: RecognitionService, removed_ids: list[int]) -> bool:
    store = service.gallery.store
    for raw_path in store.list_snapshots():
        path = Path(raw_path)
        if path.is_file() and np.isin(load_snapshot(path, store.cipher).ids, removed_ids).any():
            return False
    return True


def erase_identity(
    service: RecognitionService,
    identity_id: str,
    *,
    idempotency: IdempotencyStore | None = None,
    audit_log: SqliteAuditLog | None = None,
    actor: str = "cli",
    request_id: str = "local",
) -> ErasureReport:
    """Remove ``identity_id`` everywhere and verify it (``IdentityNotFoundError`` if unknown)."""
    gallery = service.gallery
    store = gallery.store
    record = store.get_identity(identity_id)
    name = record.name
    embedding_ids = store.embedding_ids_of(identity_id)  # no decryption: works without the key

    gallery.delete(identity_id)  # rows (zeroed first) and in-memory vectors
    raw_removed = service.raw_store.delete_identity(identity_id) if service.raw_store else 0
    rewritten, missing = _rewrite_snapshots(service, embedding_ids)
    purged = idempotency.purge_identity(identity_id) if idempotency else 0
    redacted = audit_log.redact_identity(identity_id) if audit_log else 0
    store.vacuum()

    database = Path(store.path)
    raw_bytes = database.read_bytes() if store.path != ":memory:" and database.is_file() else b""
    verification = {
        "identity_absent": not store.has_identity(identity_id),
        "index_vectors_absent": not any(gallery.has_embedding(i) for i in embedding_ids),
        "snapshots_clean": _snapshots_clean(service, embedding_ids),
        "raw_images_absent": not (
            service.raw_store and service.raw_store.files_present(identity_id)
        ),
    }
    notes = {
        # Substring checks can false-positive when another identity's data contains the text.
        "database_free_of_name": name.encode("utf-8") not in raw_bytes,
        "database_free_of_identity_id": identity_id.encode("utf-8") not in raw_bytes,
    }
    report = ErasureReport(
        identity_id=identity_id,
        embeddings_removed=len(embedding_ids),
        raw_images_removed=raw_removed,
        snapshots_rewritten=rewritten,
        snapshots_missing=missing,
        idempotency_records_purged=purged,
        audit_rows_redacted=redacted,
        verification=verification,
        notes=notes,
    )
    if audit_log is not None:
        audit_log(
            AuditEvent(
                request_id,
                actor,
                "erase",
                redacted_id(identity_id),
                "ok" if report.verified else "UNVERIFIED",
                f"embeddings={len(embedding_ids)}",
            )
        )
    if not report.verified:
        failed = [k for k, ok in verification.items() if not ok]
        raise ErasureVerificationError(f"Erasure could not be verified: {', '.join(failed)}.")
    return report
