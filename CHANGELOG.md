# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
