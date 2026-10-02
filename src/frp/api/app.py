"""FastAPI application factory.

Endpoints: ``POST /v1/enroll``, ``POST /v1/verify``, ``POST /v1/identify``,
``DELETE /v1/identities/{id}``, ``GET /healthz``, ``GET /readyz``, ``GET /metrics``.

Inference is blocking, so handlers parse the upload asynchronously and run the service in a worker
thread, throttled by a semaphore (``MAX_CONCURRENT_INFERENCE``) so a small machine is not
oversubscribed. Models load in the background after startup: the process answers ``/healthz`` and
``/readyz`` immediately, and inference endpoints return ``NOT_READY`` until loading finishes.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import math
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar

import anyio.to_thread
import numpy as np
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect
from starlette.responses import Response

from frp import __version__
from frp.api.audit import AuditEvent, AuditSink, discard
from frp.api.auth import ApiKeyRecord, ApiKeyStore
from frp.api.errors import (
    InvalidRequestError,
    NotReadyError,
    OverloadedError,
    RateLimitedError,
    UnauthorizedError,
    UnsupportedMediaTypeError,
)
from frp.api.idempotency import IdempotencyStore
from frp.api.metrics import Metrics
from frp.api.middleware import InFlight, ProtectionMiddleware, error_payload
from frp.api.ratelimit import TokenBucketLimiter
from frp.api.schemas import (
    DeleteResponse,
    EnrollResponse,
    ErrorResponse,
    HealthResponse,
    IdentifyResponse,
    ReadyResponse,
    VerifyResponse,
)
from frp.config import ConfigError, Settings
from frp.errors import FrpError
from frp.gallery import OnExisting
from frp.service import FaceSummary, RecognitionService

logger = logging.getLogger("frp.api")
T = TypeVar("T")

DESCRIPTION = """
Face detection, verification (1:1) and identification (1:N).

**Research use only.** The default pretrained models are licensed for non-commercial research
(see MODEL_CARD.md in the repository). Faces and embeddings are biometric data: uploads are not
stored unless the operator enables it, and embeddings are never returned unless explicitly enabled.

Authenticate with an API key in the `X-API-Key` header (or `Authorization: Bearer ...`).
Errors always use `{"error": {"code", "message", "request_id", "details"?}}`.
"""

COMMON_ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "Missing or invalid API key."},
    413: {"model": ErrorResponse, "description": "Upload too large."},
    415: {"model": ErrorResponse, "description": "Not multipart/form-data."},
    422: {"model": ErrorResponse, "description": "Invalid request or unusable image."},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded (see Retry-After)."},
    503: {"model": ErrorResponse, "description": "Not ready or overloaded."},
}


@dataclass
class Runtime:
    settings: Settings
    metrics: Metrics
    keys: ApiKeyStore
    limiter: TokenBucketLimiter
    idempotency: IdempotencyStore
    in_flight: InFlight
    slots: threading.BoundedSemaphore
    audit: AuditSink = discard
    service: RecognitionService | None = None
    ready: bool = False
    draining: bool = False
    load_error: str | None = None
    background: list[asyncio.Task[None]] = field(default_factory=list)


def runtime_of(request: Request) -> Runtime:
    rt: Runtime = request.app.state.runtime
    return rt


def request_id_of(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


# --- helpers shared by the endpoints ------------------------------------------------------------


def authenticate(request: Request) -> ApiKeyRecord:
    """Validate the API key, then apply the per-key rate limit."""
    rt = runtime_of(request)
    presented = request.headers.get("x-api-key")
    if presented is None:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            presented = auth[7:].strip()
    record = rt.keys.verify(presented)
    if record is None:
        rt.metrics.auth_failures.inc()
        raise UnauthorizedError("Missing or invalid API key.")
    decision = rt.limiter.check(record.key_id)
    if not decision.allowed:
        rt.metrics.rate_limited.inc()
        raise RateLimitedError("Rate limit exceeded.", decision.retry_after)
    request.state.api_key = record
    return record


def require_ready(rt: Runtime) -> RecognitionService:
    if rt.draining:
        raise NotReadyError("The service is shutting down.")
    if not rt.ready or rt.service is None:
        raise NotReadyError(rt.load_error or "Models are still loading.")
    return rt.service


def run_limited(rt: Runtime, fn: Callable[[], T]) -> T:
    """Run blocking work under the inference semaphore (called from a worker thread)."""
    wait = max(rt.settings.request_timeout_seconds / 2.0, 1.0)
    if not rt.slots.acquire(timeout=wait):
        raise OverloadedError("The server is busy; retry shortly.")
    try:
        return fn()
    finally:
        rt.slots.release()


async def read_form(
    request: Request, *, files: set[str], texts: set[str], max_files: int
) -> tuple[dict[str, list[bytes]], dict[str, str]]:
    """Parse a multipart body with strict field checking and bounded part counts."""
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise UnsupportedMediaTypeError("Send the request as multipart/form-data.")
    try:
        form = await request.form(max_files=max_files, max_fields=len(texts) + 4)
    except StarletteHTTPException as exc:  # Starlette turns multipart errors into a 400
        if exc.status_code != 400:
            raise
        raise InvalidRequestError(f"Malformed multipart body: {exc.detail}") from exc
    try:
        unexpected = sorted(set(form.keys()) - files - texts)
        if unexpected:
            err = InvalidRequestError(f"Unexpected form field(s): {', '.join(unexpected)}.")
            err.details = {
                "unexpected_fields": unexpected,
                "files": sorted(files),
                "fields": sorted(texts),
            }
            raise err
        file_data: dict[str, list[bytes]] = {}
        for name in files:
            parts = form.getlist(name)
            if any(not isinstance(p, UploadFile) for p in parts):
                raise InvalidRequestError(f"Field '{name}' must be a file upload.")
            file_data[name] = [await p.read() for p in parts if isinstance(p, UploadFile)]
        text_data: dict[str, str] = {}
        for name in texts:
            values = form.getlist(name)
            if len(values) > 1:
                raise InvalidRequestError(f"Field '{name}' was sent more than once.")
            if values:
                if isinstance(values[0], UploadFile):
                    raise InvalidRequestError(f"Field '{name}' must be plain text, not a file.")
                text_data[name] = str(values[0])
        return file_data, text_data
    finally:
        await form.close()  # release the spooled temporary files backing the uploads


def one_file(files: dict[str, list[bytes]], name: str) -> bytes:
    parts = files.get(name, [])
    if len(parts) != 1:
        raise InvalidRequestError(f"Exactly one file is required in field '{name}'.")
    return parts[0]


def parse_float(raw: str | None, name: str) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise InvalidRequestError(f"Field '{name}' must be a number.") from exc
    if not math.isfinite(value):
        raise InvalidRequestError(f"Field '{name}' must be finite.")
    return value


def face_out(face: FaceSummary) -> dict[str, Any]:
    return {
        "box": [round(v, 2) for v in face.box],
        "score": face.score,
        "n_faces": face.n_faces,
    }


def embedding_b64(vectors: np.ndarray[Any, Any]) -> list[str]:
    return [base64.b64encode(np.asarray(v, dtype="<f4").tobytes()).decode("ascii") for v in vectors]


def allow_embedding(rt: Runtime, request: Request) -> bool:
    wanted = request.query_params.get("include_embedding", "false").lower() in ("1", "true", "yes")
    if wanted and not rt.settings.return_embeddings:
        raise InvalidRequestError("Returning embeddings is disabled on this server.")
    return wanted


def audit(
    rt: Runtime,
    request: Request,
    action: str,
    identity_id: str | None,
    outcome: str,
    detail: str | None = None,
) -> None:
    key = getattr(request.state, "api_key", None)
    rt.audit(
        AuditEvent(
            request_id_of(request),
            key.key_id if key else None,
            action,
            identity_id,
            outcome,
            detail,
        )
    )


def error_response(request: Request, exc: FrpError) -> JSONResponse:
    rt = runtime_of(request)
    rt.metrics.errors.labels(code=exc.code).inc()
    headers: dict[str, str] = {}
    if isinstance(exc, RateLimitedError):
        headers["Retry-After"] = str(max(1, math.ceil(exc.retry_after)))
    if isinstance(exc, UnauthorizedError):
        headers["WWW-Authenticate"] = 'ApiKey realm="frp"'
    return JSONResponse(
        error_payload(exc.code, str(exc), request_id_of(request), exc.details),
        status_code=exc.http_status,
        headers=headers,
    )


# --- the app ------------------------------------------------------------------------------------


def create_app(
    settings: Settings,
    *,
    service: RecognitionService | None = None,
    service_factory: Callable[[], RecognitionService] | None = None,
    audit_sink: AuditSink | None = None,
) -> FastAPI:
    """Build the application. ``service`` (ready immediately) or ``service_factory`` (loaded in
    the background) supplies the recognition service; the default factory loads the real models."""
    if settings.store_raw_images:
        raise ConfigError(
            "STORE_RAW_IMAGES=true is not available until encrypted storage is configured; "
            "raw face images are never written unencrypted."
        )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    metrics = Metrics()
    rt = Runtime(
        settings=settings,
        metrics=metrics,
        keys=ApiKeyStore(settings.data_dir / "auth.db"),
        limiter=TokenBucketLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst),
        idempotency=IdempotencyStore(
            settings.data_dir / "idempotency.db", settings.idempotency_ttl_seconds
        ),
        in_flight=InFlight(),
        slots=threading.BoundedSemaphore(settings.max_concurrent_inference),
        audit=audit_sink or discard,
    )

    def default_factory() -> RecognitionService:
        from frp.factory import build_service

        return build_service(settings, stage_timer=metrics.observe_stage)

    loader = service_factory or default_factory

    def mark_ready(svc: RecognitionService) -> None:
        svc.set_stage_timer(metrics.observe_stage)
        rt.service = svc
        rt.ready = True
        metrics.ready.set(1)

    async def load_in_background() -> None:
        try:
            svc = await anyio.to_thread.run_sync(loader)
        except Exception as exc:
            rt.load_error = f"Model loading failed: {exc}"
            logger.error("model loading failed: %s", exc)
            return
        mark_ready(svc)
        logger.info("service ready")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is not None:
            mark_ready(service)
        else:
            rt.background.append(asyncio.create_task(load_in_background()))
        yield
        # Graceful shutdown: stop reporting ready, let in-flight requests finish, then close.
        rt.draining = True
        metrics.ready.set(0)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + settings.shutdown_grace_seconds
        while rt.in_flight.count > 0 and loop.time() < deadline:
            await asyncio.sleep(0.05)
        for task in rt.background:
            task.cancel()
        if rt.service is not None:
            rt.service.close()
        rt.keys.close()
        rt.idempotency.close()

    app = FastAPI(
        title="face-recognition-pipeline",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.runtime = rt
    app.add_middleware(
        ProtectionMiddleware,
        max_body_bytes=settings.max_upload_bytes,
        upload_timeout=settings.upload_timeout_seconds,
        request_timeout=settings.request_timeout_seconds,
        in_flight=rt.in_flight,
        metrics=metrics,
    )

    # -- error handling --------------------------------------------------------------------------

    @app.exception_handler(FrpError)
    async def frp_error_handler(request: Request, exc: FrpError) -> JSONResponse:
        return error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = [
            {"field": ".".join(str(p) for p in err["loc"]), "problem": err["msg"]}
            for err in exc.errors()
        ]
        err = InvalidRequestError("The request is not valid.")
        err.details = {"problems": problems}
        return error_response(request, err)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        rt.metrics.errors.labels(code=f"HTTP_{exc.status_code}").inc()
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(
            exc.status_code, f"HTTP_{exc.status_code}"
        )
        return JSONResponse(
            error_payload(code, str(exc.detail), request_id_of(request)),
            status_code=exc.status_code,
        )

    @app.exception_handler(ClientDisconnect)
    async def disconnect_handler(request: Request, exc: ClientDisconnect) -> Response:
        logger.info("client disconnected mid-request", extra={"request_id": request_id_of(request)})
        return Response(status_code=499)

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", extra={"request_id": request_id_of(request)})
        rt.metrics.errors.labels(code="INTERNAL_ERROR").inc()
        return JSONResponse(
            error_payload("INTERNAL_ERROR", "Internal error.", request_id_of(request)),
            status_code=500,
            headers={"x-request-id": request_id_of(request)},  # outside the request-id wrapper
        )

    # -- probes ----------------------------------------------------------------------------------

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"], summary="Liveness")
    async def healthz() -> HealthResponse:
        return HealthResponse()

    @app.get(
        "/readyz",
        response_model=ReadyResponse,
        tags=["ops"],
        summary="Readiness (false until models are loaded; false while draining)",
        responses={503: {"model": ReadyResponse}},
    )
    async def readyz() -> JSONResponse:
        reasons = []
        if rt.draining:
            reasons.append("shutting down")
        if not rt.ready:
            reasons.append(rt.load_error or "models loading")
        body = ReadyResponse(ready=not reasons, reasons=reasons)
        return JSONResponse(body.model_dump(), status_code=200 if body.ready else 503)

    def metrics_auth(request: Request) -> None:
        if rt.settings.metrics_require_auth:
            authenticate(request)

    @app.get(
        "/metrics", tags=["ops"], summary="Prometheus metrics", response_class=PlainTextResponse
    )
    def prometheus(_: None = Depends(metrics_auth)) -> Response:
        if rt.service is not None:
            stats = rt.service.gallery.stats()
            metrics.identities.set(float(str(stats["identities"])))
            metrics.embeddings.set(float(str(stats["embeddings"])))
        return Response(metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8")

    # -- recognition endpoints -------------------------------------------------------------------

    @app.post(
        "/v1/enroll",
        status_code=201,
        response_model=EnrollResponse,
        tags=["recognition"],
        summary="Enroll a person from one or more photos",
        description=(
            "Multipart fields: `images` (one or more files, required), `name` (required), "
            "`identity_id` (optional), `on_existing` (`error` or `append`). Send an "
            "`Idempotency-Key` header to make retries safe. All photos must contain exactly one "
            "usable face or nothing is enrolled."
        ),
        responses=COMMON_ERRORS,
    )
    async def enroll(request: Request, key: ApiKeyRecord = Depends(authenticate)) -> JSONResponse:
        svc = require_ready(rt)
        files, texts = await read_form(
            request,
            files={"images"},
            texts={"name", "identity_id", "on_existing"},
            max_files=settings.max_images_per_enroll,
        )
        images = files["images"]
        if not images:
            raise InvalidRequestError("Field 'images' (one or more image files) is required.")
        if "name" not in texts:
            raise InvalidRequestError("Field 'name' is required.")
        try:
            on_existing = OnExisting(texts.get("on_existing", "error"))
        except ValueError as exc:
            raise InvalidRequestError("Field 'on_existing' must be 'error' or 'append'.") from exc
        identity_id = texts.get("identity_id") or None

        def produce() -> tuple[int, dict[str, Any], str | None]:
            outcome = run_limited(
                rt,
                lambda: svc.enroll(
                    images, name=texts["name"], identity_id=identity_id, on_existing=on_existing
                ),
            )
            body = {
                "identity_id": outcome.result.identity_id,
                "created": outcome.result.created,
                "embeddings_added": len(outcome.result.embedding_ids),
                "faces": [face_out(f) for f in outcome.faces],
                "duplicates": [
                    {"identity_id": d.identity_id, "name": d.name, "score": d.score}
                    for d in outcome.result.duplicates
                ],
            }
            return 201, body, outcome.result.identity_id

        idempotency_key = request.headers.get("idempotency-key")
        try:
            if idempotency_key is None:
                status, body, _ = await anyio.to_thread.run_sync(produce)
                replayed = False
            else:
                digest = hashlib.sha256()
                for part in (texts["name"], identity_id or "", on_existing.value):
                    digest.update(hashlib.sha256(part.encode()).digest())
                for data in images:
                    digest.update(hashlib.sha256(data).digest())
                status, body, replayed = await anyio.to_thread.run_sync(
                    rt.idempotency.run, key.key_id, idempotency_key, digest.hexdigest(), produce
                )
        except FrpError as exc:
            audit(rt, request, "enroll", identity_id, exc.code)
            raise
        audit(rt, request, "enroll", str(body["identity_id"]), "ok", "replay" if replayed else None)
        headers = {"Idempotent-Replay": "true"} if replayed else {}
        return JSONResponse(body, status_code=status, headers=headers)

    @app.post(
        "/v1/verify",
        response_model=VerifyResponse,
        tags=["recognition"],
        summary="1:1 verification: two images, or an image and an enrolled identity",
        description=(
            "Multipart fields: `image_a` (required) plus either `image_b` (a second photo) or "
            "`identity_id` (compare with an enrolled person); optional `threshold`. "
            "Accepts when score >= threshold."
        ),
        responses=COMMON_ERRORS,
    )
    async def verify(request: Request, _key: ApiKeyRecord = Depends(authenticate)) -> JSONResponse:
        svc = require_ready(rt)
        files, texts = await read_form(
            request, files={"image_a", "image_b"}, texts={"identity_id", "threshold"}, max_files=2
        )
        image_a = one_file(files, "image_a")
        has_b, identity_id = bool(files["image_b"]), texts.get("identity_id")
        if has_b == (identity_id is not None):
            raise InvalidRequestError("Send exactly one of 'image_b' or 'identity_id'.")
        threshold = parse_float(texts.get("threshold"), "threshold")
        with_embedding = allow_embedding(rt, request)
        try:
            if has_b:
                image_b = one_file(files, "image_b")
                pair = await anyio.to_thread.run_sync(
                    lambda: run_limited(rt, lambda: svc.verify_images(image_a, image_b, threshold))
                )
                body: dict[str, Any] = {
                    "accepted": pair.accepted,
                    "score": pair.score,
                    "threshold": pair.threshold,
                    "identity_id": None,
                    "faces": [face_out(f) for f in pair.faces],
                }
                if with_embedding:
                    body["embedding_b64"] = embedding_b64(pair.embeddings.vectors)
            else:
                if identity_id is None:  # unreachable: validated above
                    raise InvalidRequestError("Send exactly one of 'image_b' or 'identity_id'.")
                outcome = await anyio.to_thread.run_sync(
                    lambda: run_limited(
                        rt, lambda: svc.verify_identity(image_a, identity_id, threshold)
                    )
                )
                body = {
                    "accepted": outcome.result.accepted,
                    "score": outcome.result.score,
                    "threshold": outcome.result.threshold,
                    "identity_id": identity_id,
                    "faces": [face_out(outcome.face)],
                }
                if with_embedding:
                    body["embedding_b64"] = embedding_b64(outcome.embedding.vectors)
        except FrpError as exc:
            audit(rt, request, "verify", identity_id, exc.code)
            raise
        audit(
            rt, request, "verify", identity_id, "ok", "accepted" if body["accepted"] else "rejected"
        )
        return JSONResponse(body)

    @app.post(
        "/v1/identify",
        response_model=IdentifyResponse,
        tags=["recognition"],
        summary="1:N open-set identification",
        description=(
            "Multipart fields: `image` (required), `k` (1-50, default 5), `threshold`. `match` is "
            "null unless the best candidate reaches the threshold: an unenrolled person must not "
            "be matched just because someone is closest."
        ),
        responses=COMMON_ERRORS,
    )
    async def identify(
        request: Request, _key: ApiKeyRecord = Depends(authenticate)
    ) -> JSONResponse:
        svc = require_ready(rt)
        files, texts = await read_form(
            request, files={"image"}, texts={"k", "threshold"}, max_files=1
        )
        image = one_file(files, "image")
        try:
            k = int(texts.get("k", "5"))
        except ValueError as exc:
            raise InvalidRequestError("Field 'k' must be an integer.") from exc
        if not 1 <= k <= 50:
            raise InvalidRequestError("Field 'k' must be between 1 and 50.")
        threshold = parse_float(texts.get("threshold"), "threshold")
        with_embedding = allow_embedding(rt, request)
        try:
            outcome = await anyio.to_thread.run_sync(
                lambda: run_limited(rt, lambda: svc.identify(image, k=k, threshold=threshold))
            )
        except FrpError as exc:
            audit(rt, request, "identify", None, exc.code)
            raise
        match = outcome.result.match
        body: dict[str, Any] = {
            "match": None
            if match is None
            else {
                "identity_id": match.identity_id,
                "name": match.name,
                "score": match.score,
            },
            "candidates": [
                {"identity_id": c.identity_id, "name": c.name, "score": c.score}
                for c in outcome.result.candidates
            ],
            "threshold": outcome.result.threshold,
            "face": face_out(outcome.face),
        }
        if with_embedding:
            body["embedding_b64"] = embedding_b64(outcome.embedding.vectors)[0]
        audit(
            rt,
            request,
            "identify",
            match.identity_id if match else None,
            "ok",
            "match" if match else "no_match",
        )
        return JSONResponse(body)

    @app.delete(
        "/v1/identities/{identity_id}",
        response_model=DeleteResponse,
        tags=["recognition"],
        summary="Erase an identity and all its embeddings",
        responses={
            **COMMON_ERRORS,
            404: {"model": ErrorResponse, "description": "No such identity."},
        },
    )
    async def delete_identity(
        identity_id: str, request: Request, _key: ApiKeyRecord = Depends(authenticate)
    ) -> JSONResponse:
        svc = require_ready(rt)
        try:
            removed = await anyio.to_thread.run_sync(svc.delete, identity_id)
        except FrpError as exc:
            audit(rt, request, "delete", identity_id, exc.code)
            raise
        await anyio.to_thread.run_sync(rt.idempotency.purge_identity, identity_id)
        audit(rt, request, "delete", identity_id, "ok")
        return JSONResponse({"identity_id": identity_id, "embeddings_removed": removed})

    return app
