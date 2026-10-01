# Design

Architecture and the decision log. Decisions are added as they are made (CLAUDE.md section 2).

## Pipeline (planned)

detect (SCRFD) -> align (5-point similarity, 112x112) -> quality gates -> embed (ArcFace, ONNX)
-> index (FAISS) -> verify (1:1) / identify (1:N), exposed via CLI and a FastAPI service.
Backends sit behind `FaceDetector` / `FaceEmbedder` protocols so they can be swapped.

## Decision log

### D-001 (2026-10-01): uv for environments and lockfile
- Options: uv, pip + requirements.txt, poetry.
- Choice: **uv** (the default in CLAUDE.md). One tool for Python install, env, and a committed
  `uv.lock`, which gives reproducible installs on Windows and Linux.
- Trade-off: contributors need uv installed. No pip fallback is provided yet.

### D-002 (2026-10-01): Python 3.12 locally, 3.11 and 3.12 in CI
- Options: 3.13 (installed on the dev machine), 3.12, 3.11.
- Choice: develop on **3.12** (`.python-version`); CI tests 3.11 and 3.12.
- Why: ML wheels (onnxruntime, faiss-cpu) often lag the newest Python. I have **not verified**
  wheel availability for 3.13 yet; 3.12 avoids the question.

### D-003 (2026-10-01): task logic in `scripts/dev.py`, Makefile is a thin wrapper
- Options: Makefile only, justfile, Python task runner.
- Choice: Python runner plus a Makefile that delegates to it.
- Why: the dev machine is Windows without `make`; CLAUDE.md requires cross-platform tasks and the
  spec requires `make setup && make test`. One implementation, two entry points.

### D-004 (2026-10-01): environment variables have no prefix
- Options: `FRP_`-prefixed names, names exactly as in the spec.
- Choice: no prefix (for example `STORE_RAW_IMAGES`), matching the spec.
- Trade-off: higher chance of colliding with unrelated variables in a shared environment.

### D-005 (2026-10-01): secret scanning split between pre-commit and CI
- Choice: gitleaks via pre-commit locally (staged changes) and `gitleaks-action` in CI (full
  history, `fetch-depth: 0`).
- Why: the pre-commit hook only scans staged files, so CI is what covers history. The action is
  free for personal accounts but needs a license secret on organization repos.

### D-006 (2026-10-01): Windows included in the CI test matrix
- CLAUDE.md marks Windows optional; the project is developed on Windows and the matrix is cheap at
  this stage. Revisit if heavy dependencies make Windows jobs slow.

### D-007 (2026-10-01): own SCRFD pre/post-processing on onnxruntime, not the `insightface` package
- Options: `insightface` Python package, own decode over `onnxruntime`.
- Choice: **own implementation** (`detect/scrfd.py`), following the public reference algorithm.
- Why: the package pulls in heavy build dependencies, would make the backend swap less visible,
  and the decode (about 100 lines) is easy to unit-test with a fake session. The risk is a subtle
  deviation from the reference; it is mitigated by testing against the real model on a real face
  (box and landmarks sane, stable under flips/rotations/scaling) and by the M6 benchmarks.

### D-008 (2026-10-01): "original image space" and the image loader
- Definition: the EXIF-upright pixel grid at the original resolution. `LoadedImage` keeps the
  (possibly downscaled) array plus the factors to map points back.
- Decode with Pillow (format sniffing by magic bytes, bomb guard, EXIF, mode conversion);
  OpenCV is used only for resizing and warping. Pillow handles palette/CMYK/16-bit/animated
  inputs that `cv2.imdecode` handles inconsistently.
- Bounded downscale: long side <= 4096 (configurable). 108 MP photos load; more than 128 MP is
  rejected as a potential decompression bomb, before pixel decoding.
- Alpha is composited on white; 16-bit grayscale is scaled by 1/257 (65535 -> 255).

### D-009 (2026-10-01): `touches_border` in addition to `visible_fraction`
- Observed on the real model: when a crop slices through a face, SCRFD predicts a box ending at
  the image edge, so `visible_fraction` stays near 1. A face whose box touches the border is
  therefore flagged truncated. Trade-off: a face filling the whole frame is also flagged
  (conservative; the quality gate in M5 decides what to do with it).

### D-010 (2026-10-01): model downloads pinned by SHA-256, recorded on first download
- Upstream does not publish checksums that I verified, so pins protect against later corruption
  or substitution, not against a bad first download. Models above 500 MB are refused by code.
- Default tests that need weights carry `@pytest.mark.needs_models` and skip when the weights
  are absent, so the default suite needs no network.

### D-011 (2026-10-01): closed-form (Umeyama) similarity instead of `cv2.estimateAffinePartial2D`
- Options: OpenCV's RANSAC/LMEDS-based estimator, `skimage.transform.SimilarityTransform`, own
  Umeyama least squares.
- Choice: **own Umeyama** (`align/aligner.py`, about 30 lines, numpy only).
- Why: deterministic by construction (no random sampling, which matters for the "same input ->
  same embedding" criterion), explicit control over reflection (never allowed) and degeneracy
  checks, and no extra dependency. With exactly five correspondences a robust estimator buys
  nothing.

### D-012 (2026-10-01): alignment fails loudly; the fit error is a provisional sanity check
- Every unusable input raises `InvalidLandmarksError` with a stable `reason`
  (`BAD_SHAPE`, `NOT_FINITE`, `DEGENERATE`, `OUT_OF_IMAGE`, `BAD_FIT`), never a best-effort crop.
- `BAD_FIT` uses the RMS residual of the similarity fit relative to the template's inter-ocular
  distance. Measured on one portrait: real detections 0.07-0.31 (the 180-degree-rotated copy has
  the worst landmark localization), mirrored left/right order 0.66. Threshold set to 0.4 between
  the two. **Provisional**: it is based on a few examples and is recalibrated in M5 on a
  validation split.
- Landmarks outside the image are rejected by default (face cut off by the frame);
  `allow_outside=True` zero-pads instead.

### D-013 (2026-10-01): pose proxies computed in the aligned frame; gate scaffold
- Yaw/pitch are geometric ratios measured after roll and scale are removed by the alignment, so
  head tilt does not leak into them (tested). They are **not calibrated degrees**.
- `quality/base.py` defines `GateResult` (gate, passed, reason code, value, threshold) and a
  `Gate` protocol; the pose gate is the first implementation, the rest arrive in M5.
- Roll is not gated by default: the alignment uprights a rotated face. That is a feature for
  tilted phone photos, and means an upside-down face still produces an upright crop.

### D-014 (2026-10-02): embeddings are version-tagged by construction
- `FaceEmbedder.embed` returns an `EmbeddingBatch(vectors, info)`, never a bare array. `ModelInfo`
  holds `model_id`, the SHA-256 of the model file, `preprocess_version` and the dimension.
  `cosine_similarity` refuses to compare batches with different infos (`ModelMismatchError`) and
  refuses non-unit vectors (the "normalization forgotten" case). The index (M4) enforces the same
  check at load and at query time.
- `preprocess_version` must be bumped whenever pre-processing changes in any way, including the
  alignment template or interpolation, because it changes what the model sees.

### D-015 (2026-10-02): batching, chunking and thread safety
- Inputs are processed in chunks of `max_batch` (default 16) to bound memory; an empty batch
  returns `(0, 512)` without calling the model.
- The embedder holds no mutable state and ONNX Runtime sessions allow concurrent `run`, so no lock
  is used. Tested with 4 and 8 concurrent callers (real model and fake session respectively).
- Measured on this machine (CPU, ONNX Runtime 1.30.0): a face embedded inside a batch of 1, 2 or
  5 is bitwise identical to the same face embedded alone, and repeated runs are bitwise
  identical. Tests assert a 1e-5 tolerance for batch-vs-single (the portable guarantee) and
  bitwise equality only for repeated identical calls.

### D-016 (2026-10-02): `FacePipeline` and the `center` multi-face policy
- `FacePipeline` composes detect -> select -> align -> quality gates and raises the specific
  `FrpError` subclass for each failure. It is what the benchmarks and, in M7, the API use.
- LFW photos are centered on the labeled person by construction, so the LFW benchmarks use the
  `center` policy (face closest to the image center, larger wins ties) instead of `largest`,
  which can pick a bystander. The policy is recorded in every cache manifest.

### D-017 (2026-10-02): LFW preparation is a cached two-stage process
- Stage 1 (`bench/prepare_lfw.py`) detects and aligns every image once and stores the 112x112
  crops (about 0.5 GB, gitignored, resumable). Embeddings, quality features, fp16/fp32
  comparisons and robustness slices all start from these crops, so the slow detector never runs
  twice. Failures are stored with a status code and counted as errors downstream.
- A shared downloader (`frp/download.py`) enforces the 500 MB cap while streaming (not only from
  the `Content-Length` header) and verifies SHA-256 and md5 where available.

### D-018 (2026-10-02): the index keeps a master copy; FAISS is a rebuildable accelerator
- Options: FAISS-only (`IndexIDMap2` with `remove_ids`), master numpy arrays plus a FAISS
  structure rebuilt on demand.
- Choice: **master copy + lazy rebuild** (`index/gallery_index.py`).
- Why: FAISS HNSW cannot delete vectors, so any HNSW design needs either tombstones or a
  rebuild; one mechanism for all three backends keeps deletion certain: the vector is removed
  from the master (and its freed memory zeroed), the stale FAISS structure is dropped, and the
  next search rebuilds it. Deletion is therefore O(rebuild) on large HNSW/IVF indexes (measured
  in `docs/RESULTS.md`), which is the price of simplicity. Batched deletes pay one rebuild.
- Adds go straight into FAISS for Flat/HNSW; IVF retrains its centroids when the gallery has
  grown 4x since training (and uses `nlist` of about sqrt(N) with at least 39 training vectors
  per list, FAISS's own guidance).

### D-019 (2026-10-02): index file format v1
- `magic | header length | JSON header | payload | SHA-256 trailer`, written atomically (temp
  file in the same directory, fsync, rename). The loader verifies the trailer first, then the
  structure (counts, dimension, finiteness, unique ids, unit norm). Unknown versions are
  refused with an explicit message instead of being migrated silently.
- The payload is vectors only; FAISS structures are not serialized, so files stay valid across
  FAISS versions and a corrupted graph cannot be loaded. Loading rebuilds the structure lazily.
- A `PayloadCipher` hook (AEAD with the header hash as associated data) is part of v1 so that
  encryption at rest (M8) does not need a format change.

### D-020 (2026-10-02): SQLite is the source of truth, rebuilt into memory at startup
- Tables `identities` and `embeddings`; every embedding row carries its model tag. The gallery
  refuses to open a store containing embeddings from another model version.
- `secure_delete=ON`, rollback journal (not WAL, so deleted bytes do not linger in a WAL file),
  `synchronous=FULL`; deleting an identity first overwrites its blobs with zeros, and `vacuum`
  rewrites the file. A test scans the database file bytes to prove a deleted name and vectors
  are gone.
- One connection guarded by a lock: simple and correct for a single process. Multi-process
  deployments would need a different design (see LIMITATIONS).

### D-021 (2026-10-02): gallery semantics
- Per-identity score: `max` over the identity's embeddings (default) or similarity to the
  normalized mean embedding. The default is a placeholder; M6 compares them empirically.
- `verify` accepts when `score >= threshold` (inclusive); NaN thresholds and NaN scores raise.
- `identify` is open-set: it returns the ranked candidates and a `match` only when the top score
  reaches the threshold. Under `max`, if a few identities fill the fetched window the search
  doubles the window until `k` distinct identities are found or the gallery is exhausted.
- Duplicate detection compares new embeddings with every *other* identity (policy
  ignore/warn/reject). The threshold (0.6) is a placeholder until M6 derives it from validation
  data. Adding more photos of the same identity is never a duplicate.
- Ids: caller-chosen ids are restricted to `[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}`, otherwise a
  random UUID4 hex is generated. Names are NFC-normalized, up to 256 characters, any script,
  rejecting control characters, lone surrogates and bidirectional override characters.
- One re-entrant lock serializes all gallery operations: simple, and an identity deleted during
  a search is either fully visible or fully gone.
