# Threat model

STRIDE-lite for the face-recognition service. Scope: one service process, its SQLite files under
`DATA_DIR`, its ONNX model files, and the HTTP API. It is a design-level analysis written by the
author, **not** a penetration test or an audit; nothing here has been reviewed by a third party.

**Assets** (most sensitive first): enrolled embeddings and any stored raw images (biometric data -
unlike a password, a face cannot be changed after a leak); the identity -> name mapping; API keys;
encryption keys; model files; the audit log.

**Trust boundaries:** client <-> HTTP API (untrusted input, authenticated by API key); API process
<-> disk (disk is assumed readable by anyone who steals a backup or a disk image); process <->
key source (environment / key file; assumed not on the same disk image as the data).

**Out of scope:** presentation attacks (printed photo, replay video, deepfakes - there is **no
liveness detection**), TLS termination (use a reverse proxy), multi-tenant isolation, a compromised
host or root user, side channels, supply-chain compromise of dependencies beyond `pip-audit`.

## Threats and mitigations

| # | Threat | STRIDE | Mitigation in this repo | Residual risk |
|---|--------|--------|-------------------------|---------------|
| T1 | **Template theft**: someone copies `frp.db`, a snapshot, or a backup | Information disclosure | Embeddings and snapshots sealed with AES-256-GCM; key kept outside the data directory; API never returns embeddings unless `RETURN_EMBEDDINGS=true` *and* the caller asks; AAD ties each blob to its identity/model. Tested by a byte scan of the DB file. | Names, identity ids, timestamps and metadata are **not** encrypted. If the key is stolen together with the data, everything is readable. Embeddings are still invertible-ish in principle (published model-inversion work reconstructs recognisable faces from embeddings), so treat even encrypted-then-leaked data as a serious incident. |
| T2 | **Replay** of a captured request (re-enroll, re-verify) | Spoofing / Tampering | Idempotency keys make enroll replays return the stored response instead of enrolling twice; keys are per API key; request ids in logs. | Verify/identify are reads, so a replayed *request* is harmless, but a replayed *image* is accepted - that is a presentation attack (see below), not something this layer can detect. No request signing/nonce scheme beyond TLS at the proxy. |
| T3 | **Enrollment poisoning**: attacker enrolls their face under a victim's identity, or enrolls junk to cause false matches | Tampering | Enroll needs an API key; quality gates reject unusable images; duplicate-identity policy (ignore/warn/reject) flags a face that already matches another identity; per-identity embedding cap; audit log of every enroll. | An authorised key can still poison its own gallery; there is no human-in-the-loop review or two-person approval. Duplicate detection only helps if the other identity is enrolled and the threshold is right (the shipped threshold is provisional - see RESULTS/LIMITATIONS). |
| T4 | **Threshold probing**: an attacker with a key submits many probes to learn the score of a target or hill-climb to a match | Information disclosure / Spoofing | Per-key rate limiting; the audit log shows probe volume per key. (`/v1/identify` returns the top-k candidates *with* scores even when none passes the threshold - that is useful for ranking and is also what a prober wants.) | Scores are returned in responses (needed for verify), which is exactly what hill-climbing needs. Rate limiting is per key, in-process (not distributed), and does not cap total attempts over weeks. Consider coarsening scores for untrusted callers - not implemented. |
| T5 | **Model substitution**: swap `w600k_r50.onnx` for a backdoored model | Tampering | Models are verified by SHA-256 on every load; the download is size-capped and checksum-pinned (trust-on-first-use for the digest, recorded in `DESIGN.md` D-010); embeddings carry `model_id`/`model_hash` and mixing models raises `ModelMismatchError`. | The pinned digest was recorded the first time the file was downloaded, so it proves "same file as when I first fetched it", not "file the vendor intended". A repo maintainer who edits the pin defeats it. ONNX is a graph format, not code, but runtime bugs in onnxruntime/OpenCV parsers remain a risk (see T6). |
| T6 | **Malicious image** (decompression bomb, crafted file, huge upload) | Denial of service / Elevation | Magic-byte sniffing (not the extension), pixel-count bomb guard, bounded downscale, streaming body cap, upload deadline, request timeout, concurrency semaphore, non-root read-only container. | Image decoders (Pillow/OpenCV) are native code; a zero-day there is not mitigated beyond the container sandbox and keeping dependencies patched. |
| T7 | **Credential theft** (API keys leaked) | Spoofing | Keys stored as SHA-256 digests, shown once, revocable; constant-time comparison; `gitleaks` in pre-commit and CI; `.env` ignored. | A leaked key works until revoked; there is no expiry or scoping (all keys are equally powerful). |
| T8 | **Abuse of retained data** (data kept longer than needed, unclear deletion) | Repudiation / Privacy | Default retention: no raw images; one command / endpoint erases a person across rows, vectors, raw images, registered snapshots, replay records and redacts the audit log, then verifies; the audit log records the erasure. | Backups, copies of the data directory, unregistered snapshots and SSD wear-levelling are outside the tool's reach (see `SECURITY_OPERATIONS.md`). Erasure of a *derived model* is N/A (no training on user data). |
| T9 | **Tampering with the audit log** | Repudiation | Append-only by construction in the code path (no update/delete API except erasure redaction). | SQLite is not tamper-evident: anyone with file access can edit it. No hash chain or remote log shipping. |
| T10 | **Log leakage** | Information disclosure | Structured logs carry request id, key id, status and timing - no images, embeddings or names by design; `tests/unit/test_service.py` checks that embedding content is not logged at DEBUG level for the service layer. | The check covers the service layer only, not every API log line; a future change could regress this. |
| T11 | **Resource exhaustion** | Denial of service | Body caps, deadlines, rate limits, bounded inference concurrency with an `OVERLOADED` response, graceful drain. | Unauthenticated floods are not rate-limited in-process (use a reverse proxy). |
| T12 | **Bias leading to unequal error rates** (a fairness harm rather than a classic security threat) | n/a | Robustness slices measured in `bench/robustness.py`. | Slices are synthetic degradations on LFW, which is demographically skewed; **no demographic fairness evaluation was done**. Do not use this for decisions about people. |

## Assumptions the mitigations depend on

1. Keys are provided through a mechanism an attacker with only the data disk cannot read.
2. TLS is terminated in front of the service; the API itself speaks plain HTTP.
3. Operators do not enable `ALLOW_UNENCRYPTED`, `RETURN_EMBEDDINGS` or `STORE_RAW_IMAGES` in a
   shared environment without understanding the consequences.
4. The host is not compromised.

## Not mitigated, by design of this project

- Liveness / anti-spoofing: a printed photo of an enrolled person will likely match.
- Fairness guarantees, legal compliance (GDPR/BIPA/etc.): this repo provides technical controls for
  erasure and minimisation, not legal advice or a compliance claim.
- Commercial use: the pretrained InsightFace models are for non-commercial research use only.
