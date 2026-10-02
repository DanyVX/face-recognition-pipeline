# Limitations

Known limitations, negative results, and spec edge cases that are **not covered**, each with a
reason. Updated every milestone. Currently only limitations known from the design are listed.

## Known from the design
- **Model license:** the planned pretrained InsightFace weights are non-commercial research use
  only (see `MODEL_CARD.md`). The project must not be presented as commercially deployable.
- **No liveness / anti-spoofing:** out of scope; planned for a separate repo. Photo-of-a-photo
  attacks are not defended against.
- **Fairness:** subgroup performance will not be measured unless a properly licensed, labeled set is
  used. Nothing here should be read as a fairness claim.
- **LFW:** near-saturated and demographically skewed; high accuracy there says little about
  real-world performance.
- **Hardware:** development and measurement machine is CPU-only (2 cores, ~8 GB RAM). GPU results
  are TBD (not yet measured) unless run on Colab/Kaggle.

## Edge cases not covered
The full spec-to-test map is `docs/EDGE_CASES.md`. These spec edge cases have **no test**, each
for the reason given (the per-milestone sections below list the ones that are tested):
- **Twins, siblings, look-alikes:** no labeled data available or licensed, so the expected false
  accepts are not measured. (A synthetic look-alike test exists only for the open-set metric.)
- **Aging:** needs AgeDB-30 or similar; access terms unverified, not obtained.
- **Heavy makeup, children, backlight:** no labeled data; no fixture or degradation models them.
  LFW has few children (`DATA.md`).
- **Real glasses, masks, beards, sunglasses, infrared-camera and real profile images:** only
  synthetic proxies are tested (random-rectangle occlusion, grayscale decoding, pose-gate logic on
  synthetic landmarks); no real occluder or camera data.
- **Head tilt 90/180 degrees, embedding accuracy:** detection was observed on one portrait; the
  robustness grid covers rotation of +/-30 degrees only.
- **Photo-of-a-photo and a liveness interface hook:** out of scope (repo 02). The spec also asks
  for a clean hook; none exists in `src/frp` beyond the quality-gate chain.
- **Float precision across platforms, CPUs, ONNX Runtime versions and GPU:** bitwise determinism was
  observed on one machine only. CI is configured for Linux and Windows but nothing compares
  embeddings across platforms.
- **fp16 vs fp32 on LFW crops, and GPU latency:** `bench/precision.py` and the GPU rows of
  `bench/latency.py` have not produced results (no `results/precision.json`, no GPU).
- **LFW label noise** (known label errors) is not checked: `DATA.md` records the errata page as not
  yet consulted. **CFP-FP / AgeDB-30** were not obtained. No subgroup or fairness evaluation.
- **Erasure outside the application's artefacts** (unregistered backups, data-directory copies,
  SSD remnants) and **embedding leakage in API log lines** (only the service layer is tested).
- **Docker build, `pip-audit` result and fresh-clone quickstart timing** are unverified; no
  second (open-license) model backend exists yet.

## M1 (detection and image loading): observations and conventions
Observations below come from one public-domain portrait (see `DATA.md`), so they are anecdotes
about this model on this image, **not measured rates**. Measured detection-failure rates on LFW
come in M6.
- A 32x32 and a 64x64 version of the portrait: no face found (the face is too small).
- A 160 px face inside a 4000x160 strip: not found (the letterbox shrinks it to about 25 px).
- Rotated copies (90/180/270 degrees) and a mirrored copy were all detected on this image. Whether
  the *embedding* of a rotated face is useful is a separate question (M6 robustness slices).
- A frame that slices through the face still yields a detection, flagged `truncated`.

Conventions that change behavior (documented, tested):
- RGBA/palette-with-transparency images are composited on a white background.
- 16-bit grayscale is mapped linearly (65535 -> 255); an image using only a small part of the
  16-bit range will look dark. 16-bit RGB is read as 8-bit by Pillow before we see it.
- Animated GIF/WebP: first frame only (or reject via `animated="reject"`).
- `min_face_px` is measured in the pixels of the array handed to the detector (after any
  downscale by the loader), not in original-image pixels.

All M1 spec edge cases are covered by tests; none are listed as not covered.

## M2 (alignment and pose): observations and conventions
- On one portrait, the 180-degree-rotated copy gave visibly worse landmark localization (mouth
  corners about 12 px apart instead of about 40) and a fit error of 0.31, against 0.07-0.14 for
  the other orientations. Rotated faces are therefore usable but less reliable; measured effect
  on verification accuracy comes with the M6 robustness slices.
- `DEFAULT_MAX_FIT_ERROR = 0.4` and the pose-gate thresholds are provisional (see D-012/D-013).
- Alignment resamples with bilinear interpolation and zero-pads outside the source image.
- The stored reference image (`tests/fixtures/align_reference.png`) is produced by this
  implementation and guards against accidental change; its correctness is separately checked by
  the marker-position test, which does not use the stored image.

All M2 spec edge cases (degenerate/collinear landmarks, landmarks outside the image, extreme
pose, mirrored landmark order, NaN landmarks) are covered by tests; none are listed as not covered.

## M3 (embedding): observations and conventions
- Bitwise determinism was observed on one machine (Intel i5-7300U, ONNX Runtime 1.30.0 CPU,
  default thread count). Results across CPUs, ONNX Runtime versions or thread counts, and on
  GPU, are not measured; floating-point summation order can differ. Tests use tolerances for
  anything but repeated calls on the same machine.
- Concurrency was tested with at most 8 callers on a 2-core machine.
- GPU execution (CUDA provider) has not been run; only the CPU fallback path is exercised.
- fp16 vs fp32: a probe on random input gave cosine 0.999997 (4 samples, not a result). The real
  measurement on LFW crops is `bench/precision.py` (M6); until then there is no fp16 claim.

## M4 (index and search): limits and conventions
- **Deletion cost:** removing a vector invalidates the FAISS structure; the next search rebuilds
  it. Cheap for Flat, expensive for HNSW/IVF at 100k vectors (see `docs/RESULTS.md` once
  measured). A production system would use tombstones or an index that deletes natively.
- **Memory:** the master vectors and the FAISS structure are both held in RAM (roughly 2x the
  raw vectors for Flat; more for HNSW). Galleries larger than RAM are out of scope.
- **Single process:** the lock and SQLite connection serve one process. Running several API
  workers against one store is not supported.
- **Serialized searches:** all gallery operations take one lock, so concurrent searches do not run
  in parallel at the Python level (FAISS still uses its OpenMP threads inside a search).
- **ANN determinism:** HNSW construction with several threads, and IVF k-means, can give slightly
  different structures run to run; approximate results are not guaranteed identical across
  rebuilds. The exact (Flat) backend is deterministic.
- **Placeholders:** the duplicate-detection threshold and the default `max` aggregation are not
  yet derived from data (M6).
- Scaling beyond 100k vectors was not tested.

### Spec edge cases for M4 and where they are tested
| Edge case | Test |
|---|---|
| Enroll the same person twice | `test_same_person_under_two_identities_is_reported_by_default`, `..._reject_policy_...` |
| Enroll with no detectable face | `tests/unit/test_pipeline.py::test_no_face_raises` (pipeline stage) |
| Empty gallery search; k > gallery size | `test_empty_gallery_search_*`, `test_k_larger_than_the_gallery_*` |
| Delete then search, also after persist/reload | `test_removed_vectors_are_never_returned`, `test_deleted_vectors_stay_deleted_after_save_and_reload`, `test_delete_then_reload_still_gone` |
| Concurrent enroll + search; delete during search | `test_concurrent_*`, `test_identity_deleted_while_being_verified_*` |
| Corrupted / partially written index; atomic write; checksum | `tests/unit/test_index_format.py` (truncation, bit flips, failed rename, failed write) |
| Dimension mismatch; mixed model versions | `test_dimension_mismatch_cannot_be_constructed`, `test_mixing_model_versions_is_blocked_*`, `test_opening_a_store_with_another_model_version_is_refused` |
| Identity id collisions | `test_identity_id_collision_is_an_error`, `test_enroll_with_a_chosen_id_and_collision_handling` |
| Unicode and very long names | `test_unicode_names_are_accepted`, `test_name_length_boundary`, `test_unicode_and_long_names_work_end_to_end` |
| Index format versioning | `test_unknown_format_version_is_refused_explicitly` (+ container and schema version tests) |
| Identical image under two identities; threshold boundary; NaN similarity; unnormalized vectors | `test_identical_image_enrolled_under_two_identities_*`, `test_verify_boundary_is_inclusive`, `test_verify_errors`, `test_unnormalized_and_nan_vectors_are_refused` |

## M5 (quality gates): limits
- The built-in thresholds are lenient defaults, not calibrated values; until
  `results/quality_thresholds.json` is produced and loaded, the gates are a coarse safety net.
- Pose gating uses geometric proxies from five landmarks (nose offset, nose position between the
  eye and mouth lines). They are not calibrated angles, and landmark error limits their accuracy
  (see the 180-degree observation in the M2 section).
- The occlusion check is only detector confidence plus the framing check; there is no occlusion
  segmentation, so masks, sunglasses and hands over the face are handled only to the extent they
  lower the detector score or the embedding similarity.
- Calibrated thresholds describe LFW-like images plus the synthetic degradations in
  `bench/degrade.py`; they may not transfer to other cameras or populations.
- Rejecting low-quality faces is a trade-off, not a free improvement: every gate has a reject
  rate, reported next to its accuracy gain.

## M7 (API): limits
- **Single node.** The rate limiter, idempotency locks and gallery lock are per process. Several
  API workers or replicas would each have their own buckets and could race on idempotency keys.
- **Unauthenticated traffic is not rate limited** (only authenticated keys are). Put a reverse
  proxy in front for IP-level limits and TLS; the service speaks plain HTTP.
- **Timeouts do not cancel work.** After a request timeout the worker thread keeps running to
  completion and its result is discarded.
- **Responses include names** (identify returns candidate names); treat the API as handling
  personal data and restrict who holds keys.
- **No pagination or listing endpoint** over HTTP; listing is available through the CLI.
- **STORE_RAW_IMAGES** is refused by the API unless the service was built with the sealed raw
  image store (M8); by default nothing uploaded is written to disk.
- No CORS configuration is provided; browser clients need a proxy or an explicit change.

### Spec edge cases for M7 and where they are tested (`tests/integration/test_api.py`)
| Edge case | Test |
|---|---|
| Upload limits (size, dimensions), magic-byte sniffing, decompression bombs | `test_oversized_upload_*`, `test_chunked_upload_*`, `test_file_type_is_decided_by_magic_bytes_*`, `test_decompression_bombs_*` |
| Request timeout; overload | `test_slow_processing_hits_the_request_timeout`, `test_overload_returns_503_*` |
| API-key auth; per-key rate limit | `test_missing_wrong_and_revoked_keys_are_401`, `test_per_key_rate_limit_*` |
| Idempotency key on enroll; concurrent identical requests | `test_idempotent_enroll_*`, `test_concurrent_identical_enroll_requests_*`, `test_reusing_an_idempotency_key_*` |
| Consistent machine-readable errors | `test_every_error_uses_one_schema_*`, `error_of` used throughout |
| No raw images persisted; embeddings not returned by default | `test_stored_raw_images_flag_is_refused_*`, `test_embeddings_are_never_returned_*` |
| Client disconnect mid-upload; extremely slow uploads | `test_client_disconnect_mid_upload_*`, `test_slow_upload_times_out` |
| Multipart with wrong field names | `test_wrong_multipart_field_names_are_explained`, `test_non_multipart_and_malformed_bodies_*` |
| Readiness false until models loaded; load failure | `test_not_ready_until_models_are_loaded`, `test_model_loading_failure_is_reported_*` |
| Graceful shutdown drains in-flight requests | `test_graceful_shutdown_drains_in_flight_requests` |
| Load smoke | `tests/integration/test_api_load.py` (in-process concurrency; real uvicorn socket test is `slow`) |

## M8 (privacy and security): limits
See `docs/THREAT_MODEL.md` for the threat-by-threat view; the items below are the honest gaps.
- **Not encrypted:** identity ids, names, timestamps, embedding counts and the audit log. Only
  embedding vectors, index snapshots and raw images are sealed.
- **Key handling is minimal:** keys come from an environment variable or a JSON file. There is no
  KMS/HSM integration (the `KeyProvider` protocol is the extension point), no automatic rotation
  schedule, and the demo option `ALLOW_KEY_GENERATION` stores the key beside the data.
- **Lost key = lost gallery**, and the service (including `frp erase`) cannot start without it.
- **Erasure scope:** it cannot reach backups, copies of the data directory, snapshots that were
  not registered (`frp snapshot` registers), or physical remnants (SSD wear levelling, OS caches,
  swap). The in-process verification checks the application's own artefacts, not the disk.
- **Erasure while serving:** `frp erase` is a separate process; a running service keeps its own
  in-memory index. Use the `DELETE` endpoint on a running service, or stop it first.
- **Audit log redaction is reversible by guessing:** erased ids become `erased:<12 hex of
  unsalted SHA-256>`, so an easily guessed id (e.g. a chosen id like "alice") can be re-linked by
  hashing candidates. Generated ids are random. The log is not tamper-evident and a failed audit
  write is logged, not fatal.
- **No liveness / presentation-attack detection, no demographic fairness evaluation.** A photo of
  an enrolled person will likely verify.
- **Scores are returned to authenticated callers**, which enables hill-climbing probes; rate
  limits slow it down but do not stop it.
- **Docker artefacts are unverified locally:** the Dockerfile, compose file and entrypoint were
  written on a machine without Docker. They are first built by the CI `docker` job; until that
  has run on GitHub, treat them as untested. Required system library (`libgomp1`) and the pinned
  `uv` image tag are likewise unverified.
- **Encryption performance** was not benchmarked separately; startup decrypts every embedding.

## M6 (benchmarks): observations, caveats and what was not measured
All numbers are in `docs/RESULTS.md` (generated from `results/*.json`). Caveats that belong next to them:
- **Provenance is dirty.** Every result file records `dirty: true` (the working tree had uncommitted
  changes when it ran) and the latency/index files record the commit just before the M8 docs.
  They were produced by the scripts in `bench/`, but a clean-tree re-run was not done.
- **fp16 vs fp32 was not measured** (`TBD`). `bench/precision.py` needs a second full embedding pass
  at roughly 4-5 images/s on this CPU (about 45 minutes for all LFW crops) and was skipped for time.
  A `--limit` subset run was attempted; its preparation alone took about 40 minutes, so it was
  abandoned. No fp16 number is claimed anywhere.
- **Index scaling stops at 10,000 vectors.** The HNSW build at 100,000 vectors did not finish in
  several minutes on this 8 GB machine, so the run used `--sizes 1000,10000`. Flat search latency at
  100,000 comes from `bench.latency`. HNSW build at 10,000 vectors already takes about 15 s with the
  current parameters; the delete-then-rebuild path costs the same.
- **Timing noise.** CPU latency was measured on a busy laptop (other processes and background doc
  work were running; free RAM was often under 1 GB). Treat p50/p95 as indicative, not as a clean
  benchmark. No GPU was available, so GPU latency is `TBD`.
- **Occlusion collapses accuracy:** held-out accuracy is 96.0% at a 10% occlusion patch, 66.5% at 20%,
  31.0% at 30% and 9.5% at 40%. Failed detections count as errors, and the failure rate at 20/30/40%
  is 29% / 63% / 85%; the split between detection failures and wrong matches was not analysed.
  Masks and large occluders are not handled.
- **Robustness samples are small** (200 validation and 200 test pairs per setting): differences
  of a few points are not significant, and most settings are unchanged from clean by construction
  of the samples. Degradations are synthetic proxies, not real cameras.
- **The detection-score gate has a cost:** the calibrated quality gates reject 11.87% of held-out
  pairs (accuracy of kept pairs 99.58% vs 99.14% for all pairs), including clean ones.
- **Extended-impostor TAR** at FAR 1e-4 uses every cross-identity pair and is not part of the standard
  LFW protocol; it is labelled as such everywhere.
- **LFW is near-saturated and its demographic make-up is not representative** (it is a set of
  public figures; its composition was not measured here). 99.5% on LFW says nothing about fairness
  or real-world KYC conditions; no demographic or cross-pose/age benchmark was run.
- **Default thresholds are still provisional.** `VERIFY_THRESHOLD`/`IDENTIFY_THRESHOLD` default to
  0.4 and `DUPLICATE_THRESHOLD` to 0.6 for this model. The LFW operating point measured from clean
  validation pairs was about 0.22 (see RESULTS). The higher defaults lean toward fewer false accepts
  at the cost of more false rejects. Choose thresholds on your own data.
