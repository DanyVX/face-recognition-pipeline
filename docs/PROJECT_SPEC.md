PROJECT 01: face-recognition-pipeline

## Mission
Build an end-to-end, benchmarked face recognition pipeline: detect -> align -> embed -> index -> verify (1:1) / identify (1:N), exposed through a small API and CLI. The differentiator is rigorous measurement (FAR/FRR, ROC, EER, open-set identification, per-stage latency) and privacy-aware engineering, not just "it recognizes faces".

## Non-goals

* Not training a face model from scratch (use pretrained, evaluate honestly).
* Not a commercial product; not a surveillance tool. No scraping faces from the internet.
* Liveness/anti-spoofing lives in repo 02 (leave a clean interface hook for it).

## Stack (default; justify any deviation in DESIGN.md)

* Python 3.11+, `onnxruntime` (CPU + optional CUDA provider), InsightFace model pack (SCRFD detector + ArcFace ResNet embedding) loaded through an abstraction layer (`FaceDetector`, `FaceEmbedder` protocols) so backends are swappable.
* `numpy`, `opencv-python-headless`, `Pillow`, `faiss-cpu` (baseline `IndexFlatIP`; compare HNSW and IVF), `FastAPI` + `uvicorn`, `pydantic v2`, `typer` (CLI), SQLite (identity metadata), `cryptography` (encryption at rest).
* Licensing warning (must handle): InsightFace's pretrained models are typically released for non-commercial research use. Verify the current license text, record it in `MODEL_CARD.md` and `DATA.md`, put a visible notice in the README, and do NOT claim the repo is commercially deployable. Implement a second backend (e.g., an open-license alternative you verify) behind the same interface if feasible, or document why not.

## Repo structure

```
face-recognition-pipeline/
  src/frp/
    detect/        # detector protocol + SCRFD backend
    align/         # 5-point similarity transform -> 112x112
    embed/         # embedder protocol + ArcFace backend, model version tagging
    index/         # FAISS wrapper, persistence, versioned index format
    quality/       # blur, pose, size, brightness gates
    store/         # SQLite identity store, encrypted embedding blobs
    api/           # FastAPI app, schemas, auth, rate limiting
    cli.py
    config.py
  bench/           # lfw_verification.py, open_set_identification.py, latency.py, report.py
  tests/           # unit/, integration/, fixtures/ (synthetic + license-clear images only)
  scripts/         # download_models.py, download_lfw.py (with checksums)
  docs/  results/  Dockerfile  docker-compose.yml  Makefile  pyproject.toml

```

## Milestones and acceptance criteria

### M0: Setup
Env discovery, repo skeleton, CI (lint/type/test/secret-scan), pre-commit, licensing notes drafted. Accept: CI green on an empty-but-structured repo; `make setup && make test` works on a fresh clone.

### M1: Detection
Wrapper returns boxes, 5 landmarks, confidence. Parameters: `min_confidence`, `min_face_px`, `max_faces`.

* Edge cases: no face; multiple faces (choose largest / return all / reject per policy flag); face partially out of frame; tiny face; extreme aspect ratios; grayscale, RGBA, palette, CMYK, 16-bit images; EXIF orientation (rotated phone photos); corrupt/truncated files; zero-byte file; huge images (e.g., 12k px -> bounded downscale with coordinate mapping back); non-image bytes with an image extension; animated GIF/WebP (use first frame or reject, documented); BGR vs RGB channel confusion (test with a colored fixture). Accept: tests for every case above; detector returns coordinates in original image space.

### M2: Alignment
Similarity transform from 5 landmarks to the canonical 112x112 template.

* Edge cases: degenerate/collinear landmarks; landmarks outside the image; extreme pose (yaw beyond threshold -> quality gate rejects, does not silently embed garbage); mirrored landmarks order; NaN landmarks.
* Test: transform a known synthetic image with known landmarks and assert the aligned output matches a stored reference within tolerance. Accept: deterministic output; visual debug tool `frp debug-align <img>` writes an overlay.

### M3: Embedding
Batched ONNX inference, float32, L2-normalized, deterministic.

* Store `model_id`, `model_hash`, `preprocess_version` with every embedding. Mixing embeddings from different model versions must be impossible (guard at index load and at query time).
* Edge cases: batch of size 0/1/N; batch larger than memory (chunking); NaN/inf in output (detect, fail loudly); ONNX provider fallback (CUDA unavailable -> CPU with a log line, not a crash); thread-safety of the session; fp16 vs fp32 difference measured and documented. Accept: same input -> same embedding (bitwise on CPU, tolerance on GPU); cosine similarity of same-image-twice == 1.0 within 1e-5.

### M4: Index and search
Enroll, verify, identify, delete, update, list, persist, reload.

* Backends: Flat (exact) baseline; HNSW; IVF-Flat. Compare recall vs exact search and latency at gallery sizes 1k / 10k / 100k (synthetic random unit vectors OK for scaling tests, real embeddings for accuracy tests).
* Multiple embeddings per identity (enrollment with several photos; aggregation policy: max-sim vs mean-embedding, compared empirically).
* Edge cases: enroll the same person twice (duplicate-identity detection above a similarity threshold -> warn/reject by policy); enroll with no detectable face; empty gallery search; k > gallery size; delete then search (ensure vector is really gone, including after persist/reload); concurrent enroll + search (locking); corrupted/partially written index file (atomic write via temp file + rename, checksum on load); dimension mismatch; identity ID collisions; Unicode names; very long names; index format versioning with migration or explicit refusal. Accept: persistence round-trip test; ANN recall@1 vs exact reported; delete verified by re-loading from disk.

### M5: Quality gates
Blur (variance of Laplacian, calibrated), face size, pose (from landmarks), brightness/contrast, occlusion proxy (detector confidence). Each gate returns a reason code, not just pass/fail.

* Edge cases: thresholds tuned on a validation split, not test; report how much accuracy improves vs how many samples are rejected (trade-off curve). Accept: documented reject-rate vs accuracy curve.

### M6: Benchmarks (the core selling point; all numbers generated by scripts)

1. Verification (1:1): LFW pairs protocol (standard 10-fold, 6000 pairs) plus at least one harder set (e.g., CFP-FP and/or AgeDB-30 if obtainable with verified licenses). Report accuracy +/- std across folds, ROC curve, EER, TAR @ FAR = 1e-3 / 1e-4. Threshold chosen on training folds, applied to the held-out fold.
2. Open-set identification (1:N): split identities into gallery and impostors (probe identities NOT in the gallery). Report rank-1 identification rate, FNIR at fixed FPIR (1e-1, 1e-2), and the DET curve. Explain why closed-set rank-1 alone is misleading.
3. Latency: per stage (decode, detect, align, embed, search) p50/p95/p99, CPU vs GPU, batch sizes, cold vs warm start, includes warm-up exclusion. Log hardware.
4. Robustness slices (small, honest): synthetic degradations on a validation subset: gaussian blur, JPEG quality 10-90, brightness shifts, downscale, rotation +/- 30 deg, horizontal flip, partial occlusion (random rectangle). Plot accuracy vs severity.

* Caveats to document: LFW is near-saturated and demographically skewed; subgroup performance is not measured unless a properly licensed labeled set is used; never imply fairness.
* Edge cases: pairs where detection fails (count them as errors, do not silently drop; report the detection-failure rate); duplicate identities across folds (leakage check); label noise in the dataset (note known LFW label issues if applicable). Accept: `make bench` reproduces all tables/plots into `results/` and `docs/RESULTS.md`; results include env + commit hash.

### M7: API
Endpoints: `POST /v1/enroll`, `POST /v1/verify` (two images or image + identity_id), `POST /v1/identify`, `DELETE /v1/identities/{id}`, `GET /healthz`, `GET /readyz`, `GET /metrics` (Prometheus).

* Upload limits (size, dimensions, content-type sniffing by magic bytes, not by extension); decompression-bomb protection (`Image.MAX_IMAGE_PIXELS`); request timeout; API-key auth; per-key rate limiting; idempotency key on enroll; consistent error schema with machine-readable codes (`NO_FACE`, `MULTIPLE_FACES`, `LOW_QUALITY`, `MODEL_MISMATCH`...).
* No raw images persisted by default (`STORE_RAW_IMAGES=false`); OpenAPI docs complete with examples.
* Edge cases: concurrent identical enroll requests; client disconnect mid-upload; multipart with wrong field names; extremely slow uploads; readiness false until models are loaded; graceful shutdown drains in-flight requests. Accept: integration tests via `httpx.AsyncClient`; load smoke test.

### M8: Privacy and security hardening

* Embeddings are biometric data: encrypt at rest (AES-GCM, key from env/KMS-style abstraction, key rotation procedure documented), never log embeddings, never return embeddings via API unless a flag is explicitly enabled.
* Right-to-erasure: `frp erase --identity <id>` removes DB rows, vectors, and backups metadata; verified by test.
* Audit log (who enrolled/queried what, no biometric content).
* Threat model in `docs/THREAT_MODEL.md` (STRIDE-lite): template theft, replay, enrollment poisoning, threshold probing (rate limits), model substitution.
* `pip-audit` clean; Docker image runs as non-root, read-only rootfs where possible.

### M9: Docs, demo, release
Architecture diagram, demo GIF using consenting/self or license-clear faces only, results table in README, LIMITATIONS (fairness, saturation of LFW, non-commercial model license, no liveness), v0.1.0 tag.

## Full edge-case matrix (each must be tested or listed as not covered)
Twins/siblings/look-alikes (expect false accepts; document); aging; glasses, masks, beards; heavy makeup; low light; backlight; IR/grayscale images; profile faces; children; infrared cameras; sunglasses; head tilt 90/180 deg; multiple faces where the wrong one is larger; photo-of-a-photo (-> repo 02); identical image enrolled under two identities; threshold at exact boundary (>= vs >); NaN similarity; cosine vs L2 confusion; embedding normalization forgotten; gallery vectors not normalized after load; float precision differences across platforms; identity deleted while a search is running; model file missing/corrupt (fail with actionable error and hash check).

## Definition of Done
Universal DoD from CLAUDE.md, plus: benchmark tables generated by scripts; open-set results reported; encryption + erasure tested; license warning visible in README; demo works from a fresh clone via Docker Compose in one command.
