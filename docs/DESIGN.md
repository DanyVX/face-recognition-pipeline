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
