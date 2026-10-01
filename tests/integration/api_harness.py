"""Helpers to run the real ASGI stack in tests: app, lifespan, API key, httpx client, raw ASGI."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI

from frp.api.app import Runtime, create_app
from frp.service import RecognitionService
from tests.unit.fake_arcface import FakeArcFaceSession
from tests.unit.service_fixtures import make_service, make_settings, person_png


@dataclass
class Harness:
    app: FastAPI
    client: httpx.AsyncClient
    key: str
    runtime: Runtime
    service: RecognitionService | None

    def auth(self, key: str | None = None) -> dict[str, str]:
        return {"X-API-Key": key or self.key}

    async def enroll(
        self,
        seeds: list[int] | None = None,
        *,
        name: str = "Alice",
        identity_id: str | None = None,
        headers: dict[str, str] | None = None,
        extra: dict[str, str] | None = None,
        images: list[bytes] | None = None,
    ) -> httpx.Response:
        parts = images if images is not None else [person_png(s) for s in (seeds or [1])]
        data = {
            "name": name,
            **({"identity_id": identity_id} if identity_id else {}),
            **(extra or {}),
        }
        return await self.client.post(
            "/v1/enroll",
            files=[("images", (f"photo{i}.png", img, "image/png")) for i, img in enumerate(parts)],
            data=data,
            headers={**self.auth(), **(headers or {})},
        )

    async def identify(self, seed: int, noise_seed: int = 1, **fields: str) -> httpx.Response:
        return await self.client.post(
            "/v1/identify",
            files={"image": ("q.png", person_png(seed, 4, noise_seed), "image/png")},
            data=fields,
            headers=self.auth(),
        )


@asynccontextmanager
async def api(
    tmp_path: Path,
    *,
    detector: Any = None,
    delay: float = 0.0,
    service_factory: Callable[[], RecognitionService] | None = None,
    start_lifespan: bool = True,
    **overrides: Any,
) -> AsyncIterator[Harness]:
    defaults: dict[str, Any] = {"rate_limit_per_minute": 6000, "rate_limit_burst": 1000}
    defaults.update(overrides)
    settings = make_settings(tmp_path, **defaults)
    service = (
        None
        if service_factory is not None
        else make_service(
            tmp_path, detector=detector, session=FakeArcFaceSession(delay=delay), **defaults
        )
    )
    app = create_app(settings, service=service, service_factory=service_factory)
    rt: Runtime = app.state.runtime
    _, key = rt.keys.create("tests")
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        harness = Harness(app, client, key, rt, service)
        if start_lifespan:
            async with app.router.lifespan_context(app):
                yield harness
        else:
            yield harness


async def raw_request(
    app: Any,
    path: str,
    *,
    headers: dict[str, str],
    body: bytes,
    content_type: str,
    receive_script: list[Any] | None = None,
    method: str = "POST",
) -> tuple[int | None, dict[str, Any] | None, list[dict[str, Any]]]:
    """Drive the ASGI app directly. ``receive_script`` items are bytes (a body chunk),
    ``("sleep", seconds)`` or ``"disconnect"``; by default the whole body arrives at once."""
    script = receive_script if receive_script is not None else [body]
    queue = list(script)
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        if not queue:
            await asyncio.sleep(3600)
        item = queue.pop(0)
        if item == "disconnect":
            return {"type": "http.disconnect"}
        if isinstance(item, tuple):
            await asyncio.sleep(item[1])
            return await receive()
        return {"type": "http.request", "body": item, "more_body": bool(queue)}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    header_list = [(b"host", b"test"), (b"content-type", content_type.encode())]
    header_list += [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": header_list,
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }
    await app(scope, receive, send)
    start = next((m for m in sent if m["type"] == "http.response.start"), None)
    payload = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    parsed = json.loads(payload) if payload else None
    return (start["status"] if start else None), parsed, sent


def multipart(fields: dict[str, str], files: list[tuple[str, str, bytes]]) -> tuple[bytes, str]:
    """(body, content-type) for a multipart request, built the same way httpx does."""
    request = httpx.Request(
        "POST",
        "http://test/x",
        data=fields,
        files=[(name, (filename, data, "image/png")) for name, filename, data in files],
    )
    return request.read(), request.headers["content-type"]


Check = Callable[[httpx.Response], Awaitable[None]]
