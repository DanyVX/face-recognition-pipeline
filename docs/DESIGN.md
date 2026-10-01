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
