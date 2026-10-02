"""Load smoke tests: concurrent traffic must never produce 5xx errors or leak in-flight state."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

from frp.api.app import create_app
from tests.integration.api_harness import api
from tests.unit.service_fixtures import make_service, make_settings, person_png

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_many_concurrent_identify_requests_all_succeed(tmp_path: Path) -> None:
    async with api(tmp_path, delay=0.005, max_concurrent_inference=2) as h:
        for seed in range(5):
            assert (await h.enroll([seed], identity_id=f"p{seed}")).status_code == 201

        async def one(i: int) -> tuple[int, float]:
            started = time.perf_counter()
            r = await h.identify(i % 5, noise_seed=i)
            return r.status_code, time.perf_counter() - started

        results = await asyncio.gather(*[one(i) for i in range(80)])
        statuses = [s for s, _ in results]
        assert statuses.count(200) == 80, statuses
        latencies = sorted(t for _, t in results)
        assert latencies[int(0.95 * len(latencies))] < 10.0  # nothing hung (generous bound)
        assert h.runtime.in_flight.count == 0  # all requests accounted for
        metrics = (await h.client.get("/metrics", headers=h.auth())).text
        assert (
            'frp_http_requests_total{method="POST",route="/v1/identify",status="200"} 80.0'
            in metrics
        )


async def test_mixed_concurrent_traffic_never_returns_a_server_error(tmp_path: Path) -> None:
    async with api(tmp_path, delay=0.002) as h:
        for seed in range(4):
            await h.enroll([seed], identity_id=f"base{seed}")

        async def churn(i: int) -> int:
            if i % 4 == 0:
                return (await h.enroll([100 + i], identity_id=f"n{i}")).status_code
            if i % 4 == 1:
                return (
                    await h.client.delete(f"/v1/identities/n{i - 1}", headers=h.auth())
                ).status_code
            return (await h.identify(i % 4, noise_seed=i)).status_code

        codes = await asyncio.gather(*[churn(i) for i in range(60)])
        assert all(c < 500 for c in codes), sorted(set(codes))
        assert set(codes) <= {200, 201, 404, 409, 422}
        assert h.runtime.in_flight.count == 0
        assert (await h.client.get("/readyz")).status_code == 200


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.slow
def test_real_uvicorn_server_handles_parallel_clients_and_stops_gracefully(tmp_path: Path) -> None:
    import uvicorn

    settings = make_settings(tmp_path, rate_limit_per_minute=6000, rate_limit_burst=1000)
    service = make_service(tmp_path)
    app = create_app(settings, service=service)
    _, key = app.state.runtime.keys.create("load")
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{base}/readyz", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    results: list[int] = []
    lock = threading.Lock()

    def client_worker(seed: int) -> None:
        with httpx.Client(base_url=base, headers={"X-API-Key": key}, timeout=30) as c:
            enrolled = c.post(
                "/v1/enroll",
                files=[("images", ("a.png", person_png(seed), "image/png"))],
                data={"name": f"P{seed}", "identity_id": f"real{seed}"},
            )
            found = c.post(
                "/v1/identify", files={"image": ("q.png", person_png(seed, 4, 1), "image/png")}
            )
            with lock:
                results.extend([enrolled.status_code, found.status_code])

    workers = [threading.Thread(target=client_worker, args=(s,)) for s in range(8)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=60)
    assert sorted(results) == [200] * 8 + [201] * 8
    assert np.all(np.array(results) < 400)
    server.should_exit = True  # graceful stop: lifespan shutdown drains and closes the stores
    thread.join(timeout=30)
    assert not thread.is_alive()
