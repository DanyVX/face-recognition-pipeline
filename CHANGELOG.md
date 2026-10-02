# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-02

### Added
- Repository skeleton, tooling (uv, ruff, mypy, pytest, pre-commit), CI, and settings/environment
  logging (M0).
- Image loader with format sniffing, bomb protection, EXIF/alpha/palette/CMYK/16-bit handling and
  bounded downscaling with coordinate mapping (M1).
- `FaceDetector` protocol and SCRFD backend (boxes, 5 landmarks, confidence), multi-face policy,
  checksum-verified model downloader (M1).
- 5-point similarity alignment to the 112x112 ArcFace template with strict input guards, pose
  proxies and a pose quality gate, and `frp debug-align` overlay tool (M2).
- Detection settings (`DET_MIN_CONFIDENCE`, `MIN_FACE_PX`, `MAX_FACES`, `ONNX_PROVIDERS`,
  `ONNX_THREADS`) (M2).

- `FaceEmbedder` protocol and ArcFace backend: batched, chunked, L2-normalized, deterministic,
  version-tagged embeddings; mixing model versions is refused (M3).
- `FacePipeline` (detect, select, align, gates), `center` multi-face policy, LFW downloader and
  pair parser, shared size-capped downloader, benchmark result writer and LFW preparation
  stage (M3).

- Versioned, checksummed, atomically written vector index with Flat, HNSW and IVF-Flat backends;
  SQLite identity store with model-tagged embeddings; gallery service with enroll, verify,
  identify, delete, update, list, duplicate-identity policy and locking; index scaling
  benchmark (M4).

- Quality features and gates (size, confidence, framing, sharpness, exposure, pose) with reason
  codes, loadable thresholds and `aligned_from_crop` (M5).

- Benchmark suite: metrics (ROC, EER, TAR@FAR, open-set DET, bootstrap/Wilson intervals), LFW
  10-fold verification with extended impostors and leakage check, open-set identification,
  robustness slices, per-stage latency, fp16 vs fp32, quality calibration, report generator,
  `make bench` and `make bench-smoke` (M6). Results are generated separately and are not
  committed with the code.

- Recognition service layer, FastAPI service (enroll, verify, identify, delete, health, readiness,
  Prometheus metrics) with API keys, per-key rate limits, upload limits and timeouts, idempotent
  enroll, consistent error schema, graceful shutdown; CLI commands (`serve`, `keys`, `enroll`,
  `verify`, `identify`, `list`, `delete`); structured JSON logging (M7).

- AES-256-GCM encryption at rest for embeddings and index snapshots with a keyring and key
  rotation (`frp encryption migrate`), optional sealed raw-image store, append-only audit log,
  orchestrated and verified identity erasure (`frp erase`, `DELETE /v1/identities/{id}`),
  `frp keys generate-encryption-key`, `frp snapshot`, `frp audit`, threat model, security
  operations guide, Dockerfile, compose file and CI docker job (M8). The service now refuses to
  start without an encryption key unless `ALLOW_UNENCRYPTED=true`.

### Changed
- `DELETE /v1/identities/{id}` now also returns `erasure_verified`.
- `frp check-config` also validates the encryption configuration.

### Fixed
- `bench.latency` crashed when a detection could not be aligned (implausible landmarks); it now
  times only alignable detections and reports `n_alignable`.
