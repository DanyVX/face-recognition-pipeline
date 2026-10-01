# HTTP API

Interactive documentation (with examples for every response and the error schema) is served at
`/docs` and `/openapi.json` when the service runs. This page is a short guide.

```bash
uv run frp keys create --name demo        # prints the key ONCE; store it safely
uv run frp serve                          # models load in the background; wait for /readyz
export KEY=frp_xxxxxxxx_...               # the key you created
```

| Method and path | Purpose |
|---|---|
| `POST /v1/enroll` | Enroll a person from one or more photos (all-or-nothing). Optional `Idempotency-Key` header. |
| `POST /v1/verify` | 1:1: `image_a` plus either `image_b` or `identity_id`. Accepts when `score >= threshold`. |
| `POST /v1/identify` | 1:N open set: ranked candidates, and `match` only if the best score reaches the threshold. |
| `DELETE /v1/identities/{id}` | Delete an identity and all its embeddings. |
| `GET /healthz` | Liveness (no key). |
| `GET /readyz` | Readiness: 503 until models are loaded and while shutting down (no key). |
| `GET /metrics` | Prometheus metrics (key required unless `METRICS_REQUIRE_AUTH=false`). |

All inference endpoints take `multipart/form-data`.

```bash
curl -H "X-API-Key: $KEY" -H "Idempotency-Key: enroll-alice-1" \
  -F images=@alice1.jpg -F images=@alice2.jpg -F name=Alice \
  http://127.0.0.1:8000/v1/enroll

curl -H "X-API-Key: $KEY" -F image=@probe.jpg -F k=3 http://127.0.0.1:8000/v1/identify

curl -H "X-API-Key: $KEY" -F image_a=@a.jpg -F image_b=@b.jpg http://127.0.0.1:8000/v1/verify
```

## Errors

Every error has the same shape and a stable `code`:

```json
{"error": {"code": "NO_FACE", "message": "No face detected.", "request_id": "3f2c...", "details": {"image_index": 0}}}
```

Common codes: `UNAUTHORIZED` (401), `RATE_LIMITED` (429, see `Retry-After`), `PAYLOAD_TOO_LARGE`
(413), `UNSUPPORTED_MEDIA_TYPE` (415), `UNSUPPORTED_IMAGE_FORMAT` (415), `IMAGE_TOO_LARGE` (413),
`CORRUPT_IMAGE` / `EMPTY_IMAGE` / `INVALID_REQUEST` / `INVALID_IDENTITY` (422), `NO_FACE`,
`MULTIPLE_FACES`, `LOW_QUALITY` (422), `IDENTITY_NOT_FOUND` (404), `IDENTITY_EXISTS` /
`DUPLICATE_IDENTITY` / `MODEL_MISMATCH` (409), `UPLOAD_TIMEOUT` (408), `REQUEST_TIMEOUT` (504),
`NOT_READY` / `OVERLOADED` (503). The `X-Request-ID` header (accepted from the client when valid,
otherwise generated) matches `request_id` in the body and in server logs.

## Operational notes

- Uploads are limited by `MAX_UPLOAD_BYTES` (checked from `Content-Length` and again while
  streaming), `MAX_IMAGES_PER_ENROLL`, `MAX_IMAGE_PIXELS` (decompression-bomb guard) and a total
  upload time (`UPLOAD_TIMEOUT_SECONDS`). The image type is decided by magic bytes, never by the
  file name or the part's content type.
- `STORE_RAW_IMAGES` is `false` and uploaded images are never written to disk by the service. The
  server refuses to start with `STORE_RAW_IMAGES=true` until encrypted storage exists.
- Embeddings are returned only if the operator sets `RETURN_EMBEDDINGS=true` **and** the client
  adds `?include_embedding=true` (verify and identify only; never enroll).
- Timeouts return an error to the client but cannot interrupt a worker thread already running
  inference; that work finishes in the background and its result is discarded.
