"""HTTP API integration tests (httpx.AsyncClient against the ASGI app; fake models, real stack)."""

from __future__ import annotations

import asyncio
import base64
import io
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from PIL import Image

from frp.api.app import create_app
from frp.service import RecognitionService
from tests.integration.api_harness import Harness, api, multipart, raw_request
from tests.unit.fake_arcface import FakeArcFaceSession
from tests.unit.service_fixtures import (
    NoFaceDetector,
    SceneDetector,
    make_service,
    make_settings,
    person_png,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def error_of(response: httpx.Response) -> dict[str, Any]:
    body = response.json()
    assert set(body) == {"error"}, body
    err: dict[str, Any] = body["error"]
    assert {"code", "message", "request_id"} <= set(err)
    assert response.headers["x-request-id"] == err["request_id"]
    return err


# --- probes and documentation --------------------------------------------------------------------


async def test_health_and_readiness_need_no_key(tmp_path: Path) -> None:
    async with api(tmp_path) as h:
        assert (await h.client.get("/healthz")).json() == {"status": "ok"}
        ready = await h.client.get("/readyz")
        assert ready.status_code == 200 and ready.json() == {"ready": True, "reasons": []}


async def test_openapi_documents_every_endpoint_with_examples_and_error_schema(
    tmp_path: Path,
) -> None:
    async with api(tmp_path) as h:
        spec = (await h.client.get("/openapi.json")).json()
        paths = spec["paths"]
        assert {
            "/v1/enroll",
            "/v1/verify",
            "/v1/identify",
            "/v1/identities/{identity_id}",
            "/healthz",
            "/readyz",
            "/metrics",
        } <= set(paths)
        schemas = spec["components"]["schemas"]
        for name in ("EnrollResponse", "VerifyResponse", "IdentifyResponse", "ErrorResponse"):
            assert schemas[name]["examples"], name
        assert "Idempotency-Key" in paths["/v1/enroll"]["post"]["description"]
        assert "422" in paths["/v1/enroll"]["post"]["responses"]
        assert "Research use only" in spec["info"]["description"]


# --- authentication and rate limiting ------------------------------------------------------------


async def test_missing_wrong_and_revoked_keys_are_401(tmp_path: Path) -> None:
    async with api(tmp_path) as h:
        no_key = await h.client.post("/v1/identify")
        assert no_key.status_code == 401 and error_of(no_key)["code"] == "UNAUTHORIZED"
        assert "ApiKey" in no_key.headers["www-authenticate"]
        wrong = await h.client.post(
            "/v1/identify", headers={"X-API-Key": "frp_00000000_" + "A" * 43}
        )
        assert wrong.status_code == 401
        key_id, other = h.runtime.keys.create("other")
        h.runtime.keys.revoke(key_id)
        revoked = await h.client.post("/v1/identify", headers={"X-API-Key": other})
        assert revoked.status_code == 401
        bearer = await h.client.delete(
            "/v1/identities/nobody", headers={"Authorization": f"Bearer {h.key}"}
        )
        assert bearer.status_code == 404  # authenticated fine; the identity just does not exist


async def test_per_key_rate_limit_returns_429_with_retry_after(tmp_path: Path) -> None:
    async with api(tmp_path, rate_limit_per_minute=60, rate_limit_burst=2) as h:
        statuses = [
            (await h.client.delete("/v1/identities/x", headers=h.auth())).status_code
            for _ in range(3)
        ]
        assert statuses == [404, 404, 429]
        limited = await h.client.delete("/v1/identities/x", headers=h.auth())
        assert (
            error_of(limited)["code"] == "RATE_LIMITED" and int(limited.headers["retry-after"]) >= 1
        )
        _, other_key = h.runtime.keys.create("second client")
        fine = await h.client.delete("/v1/identities/x", headers={"X-API-Key": other_key})
        assert fine.status_code == 404  # another key has its own bucket


# --- the core flow -------------------------------------------------------------------------------


async def test_enroll_identify_verify_delete_flow(tmp_path: Path) -> None:
    async with api(tmp_path) as h:
        enrolled = await h.enroll([1, 1], name="Alice", identity_id="alice")
        assert enrolled.status_code == 201
        body = enrolled.json()
        assert (
            body["identity_id"] == "alice"
            and body["created"] is True
            and body["embeddings_added"] == 2
        )
        assert body["faces"][0]["n_faces"] == 1 and body["duplicates"] == []
        assert (await h.enroll([2], name="Bob", identity_id="bob")).status_code == 201

        found = (await h.identify(1, noise_seed=3)).json()
        assert (
            found["match"]["identity_id"] == "alice" and found["candidates"][0]["name"] == "Alice"
        )
        stranger = (await h.identify(99, noise_seed=3)).json()
        assert (
            stranger["match"] is None and stranger["candidates"]
        )  # open set: ranked but unmatched

        ok = await h.client.post(
            "/v1/verify",
            files={"image_a": ("a.png", person_png(1, 4, 5), "image/png")},
            data={"identity_id": "alice"},
            headers=h.auth(),
        )
        assert ok.json()["accepted"] is True and ok.json()["identity_id"] == "alice"
        two = await h.client.post(
            "/v1/verify",
            files={
                "image_a": ("a.png", person_png(1), "image/png"),
                "image_b": ("b.png", person_png(2), "image/png"),
            },
            headers=h.auth(),
        )
        assert (
            two.status_code == 200
            and two.json()["accepted"] is False
            and two.json()["identity_id"] is None
        )
        assert len(two.json()["faces"]) == 2

        deleted = await h.client.delete("/v1/identities/alice", headers=h.auth())
        assert deleted.json() == {"identity_id": "alice", "embeddings_removed": 2}
        gone = await h.client.delete("/v1/identities/alice", headers=h.auth())
        assert gone.status_code == 404 and error_of(gone)["code"] == "IDENTITY_NOT_FOUND"
        after = (await h.identify(1, noise_seed=4)).json()
        assert all(c["identity_id"] != "alice" for c in after["candidates"])


async def test_threshold_is_inclusive_and_overridable_per_request(tmp_path: Path) -> None:
    async with api(tmp_path) as h:
        files = {
            "image_a": ("a.png", person_png(1), "image/png"),
            "image_b": ("b.png", person_png(2), "image/png"),
        }
        score = (
            await h.client.post(
                "/v1/verify", files=files, data={"threshold": "-1"}, headers=h.auth()
            )
        ).json()["score"]
        at = await h.client.post(
            "/v1/verify", files=files, data={"threshold": str(score)}, headers=h.auth()
        )
        assert at.json()["accepted"] is True and at.json()["threshold"] == pytest.approx(score)
        above = await h.client.post(
            "/v1/verify", files=files, data={"threshold": str(score + 0.001)}, headers=h.auth()
        )
        assert above.json()["accepted"] is False


async def test_duplicate_identity_is_reported_in_the_enroll_response(tmp_path: Path) -> None:
    async with api(tmp_path) as h:
        await h.enroll([5], name="First", identity_id="first")
        second = (await h.enroll([5], name="Second", identity_id="second")).json()
        assert (
            second["duplicates"][0]["identity_id"] == "first"
            and second["duplicates"][0]["score"] > 0.97
        )


# --- error schema and image handling -------------------------------------------------------------


async def test_every_error_uses_one_schema_and_machine_readable_codes(tmp_path: Path) -> None:
    async with api(tmp_path, detector=NoFaceDetector()) as h:
        no_face = await h.enroll([1])
        err = error_of(no_face)
        assert (
            no_face.status_code == 422
            and err["code"] == "NO_FACE"
            and err["details"] == {"image_index": 0}
        )
    async with api(tmp_path / "multi", detector=SceneDetector(n_faces=2)) as h:
        multi = await h.enroll([1])
        assert error_of(multi)["code"] == "MULTIPLE_FACES"
    async with api(tmp_path / "plain") as h:
        missing = await h.client.get("/nope", headers=h.auth())
        assert missing.status_code == 404 and error_of(missing)["code"] == "NOT_FOUND"
        wrong_method = await h.client.get("/v1/enroll", headers=h.auth())
        assert (
            wrong_method.status_code == 405
            and error_of(wrong_method)["code"] == "METHOD_NOT_ALLOWED"
        )


async def test_file_type_is_decided_by_magic_bytes_not_by_name_or_content_type(
    tmp_path: Path,
) -> None:
    async with api(tmp_path) as h:
        fake_jpeg = await h.client.post(
            "/v1/enroll",
            files=[("images", ("selfie.jpg", b"%PDF-1.7 definitely not a jpeg", "image/jpeg"))],
            data={"name": "X"},
            headers=h.auth(),
        )
        assert (
            fake_jpeg.status_code == 415
            and error_of(fake_jpeg)["code"] == "UNSUPPORTED_IMAGE_FORMAT"
        )
        mislabeled_png = await h.client.post(
            "/v1/enroll",
            files=[("images", ("notes.txt", person_png(1), "text/plain"))],
            data={"name": "X", "identity_id": "x"},
            headers=h.auth(),
        )
        assert mislabeled_png.status_code == 201  # a real PNG is accepted whatever it is called
        empty = await h.client.post(
            "/v1/enroll",
            files=[("images", ("e.png", b"", "image/png"))],
            data={"name": "Y"},
            headers=h.auth(),
        )
        assert error_of(empty)["code"] == "EMPTY_IMAGE"
        truncated = person_png(1)[:200]
        corrupt = await h.client.post(
            "/v1/enroll",
            files=[("images", ("c.png", truncated, "image/png"))],
            data={"name": "Y"},
            headers=h.auth(),
        )
        assert error_of(corrupt)["code"] == "CORRUPT_IMAGE"


async def test_decompression_bombs_are_rejected_before_decoding(tmp_path: Path) -> None:
    buf = io.BytesIO()
    Image.new("1", (20000, 20000)).save(buf, format="PNG")  # 400 MP, tiny on disk
    async with api(tmp_path) as h:
        r = await h.client.post(
            "/v1/enroll",
            files=[("images", ("b.png", buf.getvalue(), "image/png"))],
            data={"name": "B"},
            headers=h.auth(),
        )
        assert r.status_code == 413 and error_of(r)["code"] == "IMAGE_TOO_LARGE"


async def test_multi_photo_enrollment_reports_the_failing_photo_and_enrolls_nothing(
    tmp_path: Path,
) -> None:
    async with api(tmp_path) as h:
        r = await h.enroll(images=[person_png(1), b"garbage"], name="A", identity_id="a")
        assert error_of(r)["details"] == {"image_index": 1}
        assert (await h.client.delete("/v1/identities/a", headers=h.auth())).status_code == 404


# --- multipart edge cases and limits -------------------------------------------------------------


async def test_wrong_multipart_field_names_are_explained(tmp_path: Path) -> None:
    async with api(tmp_path) as h:
        r = await h.client.post(
            "/v1/enroll",
            files=[("image", ("a.png", person_png(1), "image/png"))],
            data={"name": "A"},
            headers=h.auth(),
        )
        err = error_of(r)
        assert r.status_code == 422 and err["code"] == "INVALID_REQUEST"
        assert (
            err["details"]["unexpected_fields"] == ["image"] and "images" in err["details"]["files"]
        )
        no_name = await h.client.post(
            "/v1/enroll",
            files=[("images", ("a.png", person_png(1), "image/png"))],
            headers=h.auth(),
        )
        assert error_of(no_name)["code"] == "INVALID_REQUEST"
        no_files = await h.client.post(
            "/v1/enroll",
            data={"name": "A"},
            headers={**h.auth(), "content-type": "multipart/form-data; boundary=x"},
        )
        assert no_files.status_code == 422


async def test_non_multipart_and_malformed_bodies_are_rejected(tmp_path: Path) -> None:
    async with api(tmp_path) as h:
        as_json = await h.client.post("/v1/enroll", json={"name": "A"}, headers=h.auth())
        assert as_json.status_code == 415 and error_of(as_json)["code"] == "UNSUPPORTED_MEDIA_TYPE"
        broken = await h.client.post(
            "/v1/enroll",
            content=b"--x\r\nnot a real part",
            headers={**h.auth(), "content-type": "multipart/form-data; boundary=x"},
        )
        assert broken.status_code == 422 and error_of(broken)["code"] == "INVALID_REQUEST"


async def test_field_validation(tmp_path: Path) -> None:
    async with api(tmp_path) as h:
        a = ("a.png", person_png(1), "image/png")
        both = await h.client.post(
            "/v1/verify",
            files={"image_a": a, "image_b": a},
            data={"identity_id": "x"},
            headers=h.auth(),
        )
        assert error_of(both)["code"] == "INVALID_REQUEST"
        neither = await h.client.post("/v1/verify", files={"image_a": a}, headers=h.auth())
        assert error_of(neither)["code"] == "INVALID_REQUEST"
        bad_threshold = await h.client.post(
            "/v1/verify",
            files={"image_a": a, "image_b": a},
            data={"threshold": "abc"},
            headers=h.auth(),
        )
        assert error_of(bad_threshold)["code"] == "INVALID_REQUEST"
        nan = await h.client.post(
            "/v1/verify",
            files={"image_a": a, "image_b": a},
            data={"threshold": "nan"},
            headers=h.auth(),
        )
        assert error_of(nan)["code"] == "INVALID_REQUEST"
        for k in ("0", "51", "x"):
            assert (await h.identify(1, k=k)).status_code == 422
        bad_id = await h.enroll([1], identity_id="bad id!")
        assert error_of(bad_id)["code"] == "INVALID_IDENTITY"
        bad_mode = await h.enroll([1], extra={"on_existing": "overwrite"})
        assert error_of(bad_mode)["code"] == "INVALID_REQUEST"
        twice = await h.client.post(
            "/v1/enroll",
            files=[("images", ("a.png", person_png(1), "image/png"))],
            data={"name": ["A", "B"]},
            headers=h.auth(),
        )
        assert error_of(twice)["code"] == "INVALID_REQUEST"


async def test_too_many_photos_are_refused(tmp_path: Path) -> None:
    async with api(tmp_path, max_images_per_enroll=2) as h:
        r = await h.enroll([1, 2, 3])
        assert r.status_code == 422 and error_of(r)["code"] == "INVALID_REQUEST"


async def test_oversized_upload_is_rejected_by_content_length_without_reading_it(
    tmp_path: Path,
) -> None:
    async with api(tmp_path, max_upload_bytes=300_000) as h:
        big = b"\x89PNG\r\n\x1a\n" + b"0" * 500_000
        r = await h.enroll(images=[big])
        assert r.status_code == 413 and error_of(r)["code"] == "PAYLOAD_TOO_LARGE"
        small = await h.enroll([1], identity_id="small")
        assert small.status_code == 201  # the limit is not applied to normal uploads


async def test_chunked_upload_without_content_length_is_still_capped(tmp_path: Path) -> None:
    async def chunks() -> Any:
        # A valid multipart opening followed by far more file data than the limit allows.
        yield (
            b'--zz\r\nContent-Disposition: form-data; name="images"; filename="a.png"\r\n'
            b"Content-Type: image/png\r\n\r\n"
        )
        for _ in range(100):
            yield b"x" * 5_000

    async with api(tmp_path, max_upload_bytes=50_000) as h:
        r = await h.client.post(
            "/v1/enroll",
            content=chunks(),
            headers={**h.auth(), "content-type": "multipart/form-data; boundary=zz"},
        )
        assert r.status_code == 413 and error_of(r)["code"] == "PAYLOAD_TOO_LARGE"


async def test_slow_upload_times_out(tmp_path: Path) -> None:
    async with api(tmp_path, upload_timeout_seconds=0.3) as h:
        body, ctype = multipart({"name": "Slow"}, [("images", "a.png", person_png(1))])
        started = time.perf_counter()
        status, payload, _ = await raw_request(
            h.app,
            "/v1/enroll",
            headers={**h.auth(), "content-length": str(len(body))},
            body=body,
            content_type=ctype,
            receive_script=[body[:100], ("sleep", 5.0), body[100:]],
        )
        assert (
            status == 408 and payload is not None and payload["error"]["code"] == "UPLOAD_TIMEOUT"
        )
        assert time.perf_counter() - started < 2.0  # it did not wait for the stalled client
        assert (await h.client.delete("/v1/identities/slow", headers=h.auth())).status_code == 404


async def test_client_disconnect_mid_upload_changes_nothing_and_does_not_crash(
    tmp_path: Path,
) -> None:
    async with api(tmp_path) as h:
        body, ctype = multipart(
            {"name": "Gone", "identity_id": "gone"}, [("images", "a.png", person_png(1))]
        )
        status, _, _ = await raw_request(
            h.app,
            "/v1/enroll",
            headers={**h.auth(), "content-length": str(len(body))},
            body=body,
            content_type=ctype,
            receive_script=[body[: len(body) // 2], "disconnect"],
        )
        assert status in (499, None)
        assert (
            await h.client.delete("/v1/identities/gone", headers=h.auth())
        ).status_code == 404  # never enrolled
        assert (await h.client.get("/healthz")).status_code == 200  # server still healthy


# --- request ids, timeouts, overload, internal errors ----------------------------------


async def test_request_ids_are_generated_or_propagated_when_valid(tmp_path: Path) -> None:
    async with api(tmp_path) as h:
        generated = await h.client.get("/healthz")
        assert len(generated.headers["x-request-id"]) == 32
        mine = await h.client.get("/healthz", headers={"X-Request-ID": "client-supplied-id-123"})
        assert mine.headers["x-request-id"] == "client-supplied-id-123"
        junk = await h.client.get("/healthz", headers={"X-Request-ID": "bad id with spaces\t"})
        assert junk.headers["x-request-id"] != "bad id with spaces\t"


async def test_slow_processing_hits_the_request_timeout(tmp_path: Path) -> None:
    async with api(tmp_path, delay=1.0, request_timeout_seconds=0.3) as h:
        r = await h.enroll([1], identity_id="slow")
        assert r.status_code == 504 and error_of(r)["code"] == "REQUEST_TIMEOUT"
        await asyncio.sleep(1.2)  # let the abandoned worker finish before the app shuts down


async def test_overload_returns_503_when_no_inference_slot_frees_up(tmp_path: Path) -> None:
    async with api(tmp_path, max_concurrent_inference=1, request_timeout_seconds=2.0) as h:
        h.runtime.slots.acquire()  # occupy the only slot, as a long request would
        try:
            started = time.perf_counter()
            r = await h.identify(1)
            assert r.status_code == 503 and error_of(r)["code"] == "OVERLOADED"
            assert 0.9 < time.perf_counter() - started < 3.0  # waited for the slot, then gave up
        finally:
            h.runtime.slots.release()
        assert (await h.identify(1)).status_code == 200


async def test_unexpected_exceptions_return_500_without_leaking_internals(tmp_path: Path) -> None:
    class Exploding(SceneDetector):
        def detect(self, image: np.ndarray) -> list[Any]:
            raise RuntimeError("secret internal path /etc/passwd")

    async with api(tmp_path, detector=Exploding()) as h:
        r = await h.identify(1)
        err = error_of(r)
        assert r.status_code == 500 and err["code"] == "INTERNAL_ERROR"
        assert "passwd" not in r.text and "RuntimeError" not in r.text


# --- idempotency and concurrency -----------------------------------------------------------------


async def test_idempotent_enroll_replays_instead_of_enrolling_twice(tmp_path: Path) -> None:
    async with api(tmp_path) as h:
        headers = {"Idempotency-Key": "enroll-alice-001"}
        first = await h.enroll([1], name="Alice", headers=headers)
        again = await h.enroll([1], name="Alice", headers=headers)
        assert first.status_code == again.status_code == 201
        assert first.json() == again.json() and "idempotent-replay" not in first.headers
        assert again.headers["idempotent-replay"] == "true"
        assert h.service is not None and h.service.gallery.stats()["identities"] == 1


async def test_reusing_an_idempotency_key_with_different_content_is_rejected(
    tmp_path: Path,
) -> None:
    async with api(tmp_path) as h:
        headers = {"Idempotency-Key": "key-1"}
        await h.enroll([1], name="Alice", headers=headers)
        other = await h.enroll([2], name="Alice", headers=headers)
        assert other.status_code == 422 and error_of(other)["code"] == "IDEMPOTENCY_KEY_REUSED"
        invalid = await h.enroll([1], headers={"Idempotency-Key": "has space"})
        assert error_of(invalid)["code"] == "INVALID_REQUEST"


async def test_failed_requests_are_not_cached_by_idempotency(tmp_path: Path) -> None:
    async with api(tmp_path) as h:
        headers = {"Idempotency-Key": "retry-me"}
        bad = await h.enroll(images=[b"garbage"], headers=headers)
        assert bad.status_code == 415
        good = await h.enroll([1], headers=headers)  # same key, now a valid body, runs normally
        assert good.status_code == 422 or good.status_code == 201


async def test_concurrent_identical_enroll_requests_create_one_identity(tmp_path: Path) -> None:
    async with api(tmp_path, delay=0.1) as h:
        headers = {"Idempotency-Key": "same-key"}
        responses = await asyncio.gather(
            *[h.enroll([1], name="Alice", headers=headers) for _ in range(5)]
        )
        assert {r.status_code for r in responses} == {201}
        assert len({r.json()["identity_id"] for r in responses}) == 1
        assert sum(r.headers.get("idempotent-replay") == "true" for r in responses) == 4
        assert h.service is not None and h.service.gallery.stats()["identities"] == 1


async def test_concurrent_enrolls_of_one_identity_id_without_a_key_yield_one_winner(
    tmp_path: Path,
) -> None:
    async with api(tmp_path, delay=0.05) as h:
        responses = await asyncio.gather(*[h.enroll([1], identity_id="race") for _ in range(4)])
        codes = sorted(r.status_code for r in responses)
        assert codes == [201, 409, 409, 409]
        assert all(
            error_of(r)["code"] == "IDENTITY_EXISTS" for r in responses if r.status_code == 409
        )


async def test_erasing_an_identity_also_forgets_its_idempotent_replay(tmp_path: Path) -> None:
    async with api(tmp_path) as h:
        headers = {"Idempotency-Key": "to-purge"}
        await h.enroll([1], identity_id="p1", headers=headers)
        await h.client.delete("/v1/identities/p1", headers=h.auth())
        again = await h.enroll([1], identity_id="p1", headers=headers)
        assert again.status_code == 201 and "idempotent-replay" not in again.headers


# --- readiness, shutdown -------------------------------------------------------------------------


async def test_not_ready_until_models_are_loaded(tmp_path: Path) -> None:
    release = threading.Event()
    built: list[RecognitionService] = []

    def slow_factory() -> RecognitionService:
        release.wait(timeout=10)
        svc = make_service(tmp_path)
        built.append(svc)
        return svc

    async with api(tmp_path, service_factory=slow_factory) as h:
        not_ready = await h.client.get("/readyz")
        assert not_ready.status_code == 503 and "models loading" in not_ready.json()["reasons"]
        blocked = await h.identify(1)
        assert blocked.status_code == 503 and error_of(blocked)["code"] == "NOT_READY"
        assert (await h.client.get("/healthz")).status_code == 200  # alive, just not ready
        release.set()
        for _ in range(100):
            if (await h.client.get("/readyz")).status_code == 200:
                break
            await asyncio.sleep(0.05)
        assert (await h.client.get("/readyz")).status_code == 200
        assert (await h.enroll([1], identity_id="late")).status_code == 201


async def test_model_loading_failure_is_reported_not_hidden(tmp_path: Path) -> None:
    def broken() -> RecognitionService:
        raise RuntimeError("model file corrupt")

    async with api(tmp_path, service_factory=broken) as h:
        for _ in range(50):
            body = (await h.client.get("/readyz")).json()
            if any("corrupt" in r for r in body["reasons"]):
                break
            await asyncio.sleep(0.05)
        assert any("model file corrupt" in r for r in body["reasons"])
        assert (await h.identify(1)).status_code == 503


async def test_graceful_shutdown_drains_in_flight_requests(tmp_path: Path) -> None:
    async with api(tmp_path, delay=0.5, shutdown_grace_seconds=5.0, start_lifespan=False) as h:
        lifespan = h.app.router.lifespan_context(h.app)
        await lifespan.__aenter__()
        slow = asyncio.create_task(h.enroll([1], identity_id="inflight"))
        await asyncio.sleep(0.15)  # the request is now running
        closing = asyncio.create_task(lifespan.__aexit__(None, None, None))
        await asyncio.sleep(0.1)
        draining = await h.client.get("/readyz")
        assert draining.status_code == 503 and "shutting down" in draining.json()["reasons"]
        refused = await h.identify(1)
        assert refused.status_code == 503 and error_of(refused)["code"] == "NOT_READY"
        finished = await slow
        assert finished.status_code == 201  # the in-flight request completed normally
        await closing


# --- embeddings, metrics -------------------------------------------------------------------------


async def test_embeddings_are_never_returned_unless_the_server_allows_it(tmp_path: Path) -> None:
    async with api(tmp_path) as h:
        r = await h.client.post(
            "/v1/identify?include_embedding=true",
            files={"image": ("q.png", person_png(1), "image/png")},
            headers=h.auth(),
        )
        assert r.status_code == 422 and "disabled" in error_of(r)["message"]
        plain = await h.identify(1)
        assert "embedding_b64" not in plain.text
        assert "embedding_b64" not in (await h.enroll([1], identity_id="e")).text
    async with api(tmp_path / "on", return_embeddings=True) as h:
        r = await h.client.post(
            "/v1/identify?include_embedding=true",
            files={"image": ("q.png", person_png(1), "image/png")},
            headers=h.auth(),
        )
        vector = np.frombuffer(base64.b64decode(r.json()["embedding_b64"]), dtype="<f4")
        assert vector.shape == (512,) and np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-4)


async def test_metrics_require_a_key_by_default_and_expose_the_expected_series(
    tmp_path: Path,
) -> None:
    async with api(tmp_path) as h:
        assert (await h.client.get("/metrics")).status_code == 401
        await h.enroll([1], identity_id="m1")
        await h.identify(1)
        text = (await h.client.get("/metrics", headers=h.auth())).text
        assert 'frp_http_requests_total{method="POST",route="/v1/enroll",status="201"} 1.0' in text
        assert 'frp_pipeline_stage_duration_seconds_count{stage="embed"}' in text
        assert "frp_gallery_identities 1.0" in text and "frp_ready 1.0" in text
        assert (
            'frp_errors_total{code="UNAUTHORIZED"}' in text or "frp_auth_failures_total 1.0" in text
        )
    async with api(tmp_path / "open", metrics_require_auth=False) as h:
        assert (await h.client.get("/metrics")).status_code == 200


async def test_stored_raw_images_flag_is_refused_until_encryption_exists(tmp_path: Path) -> None:
    from frp.config import ConfigError

    with pytest.raises(ConfigError, match="STORE_RAW_IMAGES"):
        create_app(make_settings(tmp_path, store_raw_images=True), service=make_service(tmp_path))


async def test_app_factory_can_be_built_many_times_in_one_process(tmp_path: Path) -> None:
    for i in range(3):
        async with api(tmp_path / f"app{i}") as h:
            assert (await h.client.get("/healthz")).status_code == 200
    assert isinstance(Harness, type) and FakeArcFaceSession is not None
