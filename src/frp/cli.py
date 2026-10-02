"""Command-line interface (``frp``). Subcommands are added milestone by milestone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image

from frp import __version__
from frp.align import AlignedFace, align_face
from frp.config import ConfigError, load_settings
from frp.debug import alignment_overlay
from frp.detect import FacePolicy, select_faces
from frp.envinfo import collect_env
from frp.errors import FrpError, InvalidLandmarksError
from frp.factory import build_detector, build_service
from frp.gallery import OnExisting
from frp.imageio import load_image
from frp.logging_setup import configure_logging

app = typer.Typer(help="Face recognition pipeline.", no_args_is_help=True, add_completion=False)
keys_app = typer.Typer(help="Manage API keys for the HTTP service.", no_args_is_help=True)
app.add_typer(keys_app, name="keys")


@app.callback()
def main_callback() -> None:
    """Configure logging from LOG_LEVEL / LOG_FORMAT before any subcommand runs."""
    try:
        settings = load_settings()
    except ConfigError:
        return  # `check-config` reports the problem; other commands fail with their own message
    configure_logging(settings.log_level, settings.log_format)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def env() -> None:
    """Print hardware, library and git info as JSON (same data stored in result files)."""
    typer.echo(json.dumps(collect_env(), indent=2))


@app.command()
def check_config() -> None:
    """Validate configuration from the environment and exit non-zero if it is invalid."""
    from frp.factory import build_cipher

    try:
        build_cipher(load_settings())  # also fails when no encryption key is configured
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo("configuration OK")


@app.command("debug-align")
def debug_align(
    image: Annotated[Path, typer.Argument(help="Input image (JPEG, PNG, GIF, WebP, BMP, TIFF).")],
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Where to write the overlay PNG.")
    ] = None,
) -> None:
    """Detect the largest face, align it, and write a debug overlay picture.

    The overlay shows the detection (box, five landmarks) next to the aligned 112x112 crop with
    the canonical template points. The output contains the input's face: treat it as sensitive.
    """
    target = out or Path(f"{image.stem}_align.png")
    try:
        loaded = load_image(image.read_bytes())
        faces = build_detector().detect(loaded.array)
        chosen = select_faces(faces, FacePolicy.LARGEST)[0]
    except FrpError as exc:
        typer.echo(f"{exc.code}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        typer.echo(f"Cannot read {image}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    aligned: AlignedFace | None = None
    message: str | None = None
    try:
        aligned = align_face(loaded.array, chosen.landmarks)
    except InvalidLandmarksError as exc:
        message = f"{exc.reason}: {exc}"
    overlay = alignment_overlay(loaded.array, [chosen], aligned, message)
    Image.fromarray(overlay).save(target)
    typer.echo(f"wrote {target}" + ("" if aligned else f" (alignment failed: {message})"))
    if aligned is None:
        raise typer.Exit(code=1)


def _fail(exc: FrpError) -> typer.Exit:
    typer.echo(f"{exc.code}: {exc}", err=True)
    if exc.details:
        typer.echo(json.dumps(exc.details), err=True)
    return typer.Exit(code=1)


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        typer.echo(f"Cannot read {path}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _json(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, default=str))


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Bind address (default API_HOST).")] = None,
    port: Annotated[int | None, typer.Option(help="Port (default API_PORT).")] = None,
) -> None:
    """Run the HTTP API. Models load in the background; /readyz turns true when they are ready."""
    import uvicorn

    from frp.api.app import create_app

    settings = load_settings()
    uvicorn.run(
        create_app(settings),
        host=host or settings.api_host,
        port=port or settings.api_port,
        timeout_graceful_shutdown=int(settings.shutdown_grace_seconds),
        log_config=None,  # our own logging configuration stays in charge
    )


@keys_app.command("create")
def keys_create(name: Annotated[str, typer.Option(help="A label for this key.")]) -> None:
    """Create an API key. The key is printed ONCE and cannot be recovered."""
    from frp.api.auth import ApiKeyStore

    settings = load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    key_id, key = ApiKeyStore(settings.data_dir / "auth.db").create(name)
    typer.echo(f"key id: {key_id}", err=True)
    typer.echo(key)


@keys_app.command("list")
def keys_list() -> None:
    """List API keys (ids, names and state; never secrets)."""
    from frp.api.auth import ApiKeyStore

    settings = load_settings()
    store = ApiKeyStore(settings.data_dir / "auth.db")
    _json(
        [
            {"key_id": k.key_id, "name": k.name, "created_at": k.created_at, "revoked": k.revoked}
            for k in store.list()
        ]
    )


@keys_app.command("revoke")
def keys_revoke(key_id: str) -> None:
    """Revoke an API key by id."""
    from frp.api.auth import ApiKeyStore

    settings = load_settings()
    if not ApiKeyStore(settings.data_dir / "auth.db").revoke(key_id):
        typer.echo(f"No active key with id {key_id}.", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"revoked {key_id}")


@app.command()
def enroll(
    images: Annotated[list[Path], typer.Argument(help="One or more photos of the same person.")],
    name: Annotated[str, typer.Option(help="Display name.")],
    identity_id: Annotated[str | None, typer.Option(help="Chosen id (default: generated).")] = None,
    append: Annotated[bool, typer.Option(help="Add to an existing identity.")] = False,
) -> None:
    """Enroll a person from photos into the local gallery."""
    service = build_service()
    try:
        outcome = service.enroll(
            [_read(p) for p in images],
            name=name,
            identity_id=identity_id,
            on_existing=OnExisting.APPEND if append else OnExisting.ERROR,
        )
    except FrpError as exc:
        raise _fail(exc) from exc
    finally:
        service.close()
    _json(
        {
            "identity_id": outcome.result.identity_id,
            "created": outcome.result.created,
            "embeddings_added": len(outcome.result.embedding_ids),
            "duplicates": [d.__dict__ for d in outcome.result.duplicates],
        }
    )


@app.command()
def verify(
    image_a: Path,
    image_b: Annotated[Path | None, typer.Argument(help="Second photo.")] = None,
    identity: Annotated[
        str | None, typer.Option(help="Compare with this enrolled identity.")
    ] = None,
    threshold: Annotated[float | None, typer.Option(help="Accept when score >= threshold.")] = None,
) -> None:
    """1:1 verification: two photos, or one photo against an enrolled identity."""
    if (image_b is None) == (identity is None):
        typer.echo("Give exactly one of IMAGE_B or --identity.", err=True)
        raise typer.Exit(code=2)
    service = build_service()
    try:
        if image_b is not None:
            pair = service.verify_images(_read(image_a), _read(image_b), threshold)
            _json({"accepted": pair.accepted, "score": pair.score, "threshold": pair.threshold})
        else:
            if identity is None:  # unreachable: validated above
                raise typer.Exit(code=2)
            result = service.verify_identity(_read(image_a), identity, threshold).result
            _json(
                {"accepted": result.accepted, "score": result.score, "threshold": result.threshold}
            )
    except FrpError as exc:
        raise _fail(exc) from exc
    finally:
        service.close()


@app.command()
def identify(
    image: Path,
    k: Annotated[int, typer.Option(help="Candidates to return.")] = 5,
    threshold: Annotated[float | None, typer.Option(help="Match threshold.")] = None,
) -> None:
    """1:N open-set identification against the local gallery."""
    service = build_service()
    try:
        out = service.identify(_read(image), k=k, threshold=threshold).result
    except FrpError as exc:
        raise _fail(exc) from exc
    finally:
        service.close()
    _json(
        {
            "match": None if out.match is None else out.match.__dict__,
            "candidates": [c.__dict__ for c in out.candidates],
            "threshold": out.threshold,
        }
    )


@app.command("list")
def list_identities(limit: int = 100, offset: int = 0) -> None:
    """List enrolled identities (ids, names, counts; never embeddings)."""
    service = build_service()
    try:
        rows = service.gallery.list_identities(limit, offset)
    finally:
        service.close()
    _json(
        [
            {
                "identity_id": r.identity_id,
                "name": r.name,
                "embeddings": r.n_embeddings,
                "created_at": r.created_at,
            }
            for r in rows
        ]
    )


@app.command()
def delete(identity_id: str) -> None:
    """Delete an identity and its embeddings from the gallery (see `erase` for complete erasure)."""
    service = build_service()
    try:
        removed = service.delete(identity_id)
    except FrpError as exc:
        raise _fail(exc) from exc
    finally:
        service.close()
    _json({"identity_id": identity_id, "embeddings_removed": removed})


@keys_app.command("generate-encryption-key")
def keys_generate_encryption_key(
    key_id: Annotated[str, typer.Option("--id", help="Key id (letters, digits, . _ -).")] = "k1",
) -> None:
    """Print a new AES-256 key as ``id:BASE64`` for ENCRYPTION_KEYS. It is not stored anywhere."""
    from frp.crypto import KeyConfigError, format_key, generate_key, parse_keyring

    key = generate_key()
    line = format_key(key_id, key)
    try:
        parse_keyring(line)  # validates the id
    except KeyConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(line)


@app.command("erase")
def erase_command(
    identity: Annotated[str, typer.Option(help="Identity id to erase completely.")],
) -> None:
    """Erase a person from the local data: rows, vectors, raw images, snapshots, replay, audit.

    Run it while the HTTP service is stopped, or use `DELETE /v1/identities/{id}` on the running
    service (a running process keeps its own in-memory index). Backups are not reached.
    """
    from frp.api.audit import SqliteAuditLog
    from frp.api.idempotency import IdempotencyStore
    from frp.erasure import erase_identity

    settings = load_settings()
    service = build_service()
    audit_log = SqliteAuditLog(settings.data_dir / "audit.db") if settings.audit_enabled else None
    idempotency = IdempotencyStore(
        settings.data_dir / "idempotency.db", settings.idempotency_ttl_seconds
    )
    try:
        report = erase_identity(
            service, identity, idempotency=idempotency, audit_log=audit_log, actor="cli"
        )
    except FrpError as exc:
        raise _fail(exc) from exc
    finally:
        service.close()
        idempotency.close()
        if audit_log is not None:
            audit_log.close()
    _json({**report.__dict__, "verified": report.verified})


encryption_app = typer.Typer(help="Encryption maintenance.", no_args_is_help=True)
app.add_typer(encryption_app, name="encryption")


@encryption_app.command("migrate")
def encryption_migrate() -> None:
    """Re-seal all embeddings and registered snapshots with the ACTIVE key.

    Use after rotating keys, or to encrypt a store created without encryption.
    """
    from frp.factory import build_cipher
    from frp.rotation import key_usage, rotate_encryption
    from frp.store import IdentityStore

    settings = load_settings()
    try:
        cipher = build_cipher(settings)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    if cipher is None:
        typer.echo("Encryption is not configured (ALLOW_UNENCRYPTED is set).", err=True)
        raise typer.Exit(code=2)
    with IdentityStore(settings.db_path, cipher=cipher) as store:
        report = rotate_encryption(store)
        _json({**report.__dict__, "key_usage_after": key_usage(store)})


@app.command("snapshot")
def snapshot_command(path: Path) -> None:
    """Save an encrypted index snapshot and register it, so erasure can reach it."""
    service = build_service()
    try:
        service.gallery.save_snapshot(path)
    finally:
        service.close()
    typer.echo(f"wrote and registered {path}")


@app.command("audit")
def audit_command(
    identity: Annotated[str | None, typer.Option(help="Only events of this identity.")] = None,
    limit: int = 50,
) -> None:
    """Show recent audit events (who did what; never biometric content)."""
    from frp.api.audit import SqliteAuditLog

    settings = load_settings()
    log = SqliteAuditLog(settings.data_dir / "audit.db")
    try:
        _json(log.query(identity_id=identity, limit=limit))
    finally:
        log.close()


if __name__ == "__main__":
    app()
