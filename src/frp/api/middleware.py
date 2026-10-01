"""Pure-ASGI protection: request ids, body-size and upload-time limits, request timeout.

Implemented below the framework so the limits apply *while the body is being received*, before
any multipart parsing buffers it: a client cannot make the server hold an unbounded upload or tie
up a worker with a slow one.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass
from typing import Any, Final

from starlette.responses import JSONResponse

from frp.api.errors import PayloadTooLargeError, RequestTimeoutError, UploadTimeoutError
from frp.api.metrics import Metrics

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_REQUEST_ID: Final = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
REQUEST_ID_HEADER: Final = b"x-request-id"


@dataclass
class InFlight:
    """Counts requests being processed; read by readiness and graceful shutdown."""

    count: int = 0


def error_payload(code: str, message: str, request_id: str, details: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message, "request_id": request_id}
    if details:
        body["details"] = details
    return {"error": body}


async def _reject(
    scope: Scope,
    receive: Receive,
    send: Send,
    request_id: str,
    exc: Exception,
    status: int,
    code: str,
) -> None:
    response = JSONResponse(
        error_payload(code, str(exc), request_id),
        status_code=status,
        headers={"x-request-id": request_id, "connection": "close"},
    )
    await response(scope, receive, send)


class ProtectionMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int,
        upload_timeout: float,
        request_timeout: float,
        in_flight: InFlight,
        metrics: Metrics | None = None,
    ) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.upload_timeout = upload_timeout
        self.request_timeout = request_timeout
        self.in_flight = in_flight
        self.metrics = metrics

    @staticmethod
    def _request_id(scope: Scope) -> str:
        for name, value in scope.get("headers", []):
            if name == REQUEST_ID_HEADER:
                candidate: str = value.decode("latin-1")
                if _REQUEST_ID.fullmatch(candidate):
                    return candidate
        return uuid.uuid4().hex

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = self._request_id(scope)
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status_holder = {"status": 0}
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_holder["status"] = int(message["status"])
                headers = list(message.get("headers", []))
                if not any(name == REQUEST_ID_HEADER for name, _ in headers):
                    headers.append((REQUEST_ID_HEADER, request_id.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        content_length = next(
            (v for k, v in scope.get("headers", []) if k == b"content-length"), None
        )
        if (
            content_length is not None
            and content_length.isdigit()
            and int(content_length) > self.max_body_bytes
        ):
            too_large = PayloadTooLargeError(
                f"Request body exceeds the {self.max_body_bytes:,} byte limit."
            )
            status_holder["status"] = too_large.http_status
            await _reject(
                scope, receive, send, request_id, too_large, too_large.http_status, too_large.code
            )
            self._record(scope, status_holder["status"], started)
            return

        total = 0
        first_read: float | None = None

        async def limited_receive() -> Message:
            nonlocal total, first_read
            now = time.monotonic()
            if first_read is None:
                first_read = now
            remaining = self.upload_timeout - (now - first_read)
            if remaining <= 0:
                raise UploadTimeoutError("The upload took too long.")
            try:
                message = await asyncio.wait_for(receive(), timeout=remaining)
            except TimeoutError as exc:
                raise UploadTimeoutError("The upload took too long.") from exc
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_body_bytes:
                    raise PayloadTooLargeError(
                        f"Request body exceeds the {self.max_body_bytes:,} byte limit."
                    )
            return message

        self.in_flight.count += 1
        if self.metrics:
            self.metrics.in_flight.inc()
        try:
            await asyncio.wait_for(
                self.app(scope, limited_receive, send_wrapper), timeout=self.request_timeout
            )
        except TimeoutError:
            if not response_started:
                slow = RequestTimeoutError("The request took too long to process.")
                status_holder["status"] = slow.http_status
                await _reject(
                    scope, receive, send_wrapper, request_id, slow, slow.http_status, slow.code
                )
        finally:
            self.in_flight.count -= 1
            if self.metrics:
                self.metrics.in_flight.dec()
            self._record(scope, status_holder["status"], started)

    def _record(self, scope: Scope, status: int, started: float) -> None:
        if self.metrics is None:
            return
        route = getattr(scope.get("route"), "path", None) or "unmatched"
        method = str(scope.get("method", "GET"))
        self.metrics.requests.labels(route=route, method=method, status=str(status)).inc()
        self.metrics.request_seconds.labels(route=route, method=method).observe(
            time.perf_counter() - started
        )
