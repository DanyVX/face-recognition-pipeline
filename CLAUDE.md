# CLAUDE.md: Shared engineering rules

## 1. Context
- Owner: Dany (GitHub: DanyVX), third-year CS student building a portfolio aimed at an
  AI/ML engineering internship at a company that builds production biometric, KYC,
  computer-vision, and AI-platform systems.
- The repo must read as **production-minded engineering**, not a tutorial. Reviewers will
  look for: measured results, honest limitations, tests, reproducibility, security/privacy
  awareness, and clean history.
- Tone with me: casual, direct, no corporate fluff. If my request drifts from the spec or
  from good engineering, say so plainly and propose the better path.

## 2. Working agreement
- **Plan before code.** For each milestone: restate the goal, list files you'll touch,
  then implement.
- **One milestone per branch/PR.** Branch name `m<N>-<slug>`. Merge to `main` only when
  lint, types, tests, and CI pass.
- **Ask early, ask in batches.** If something is ambiguous, ask all questions at once.
- **Never invent facts.** If you are unsure about an API, a license, a dataset's terms, or an
  eBay/vendor policy, say so and tell me where to verify. Do not guess.
- **Explain trade-offs** when choosing a library or algorithm. Record the decision in
  `docs/DESIGN.md` under "Decision log" (date, options considered, choice, why).
- **Small commits**, Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`,
  `perf:`, `refactor:`, `ci:`). No commit larger than one logical change.
- Do not install system-level packages, change global configs, or download files larger
  than 500 MB without asking me first.

## 3. Environment discovery (do this first, every repo)
Detect and report: OS and shell (Windows/Linux/macOS/WSL), Python version, `uv`/`pip`
availability, GPU + CUDA version (or none), RAM, free disk, internet access, Docker
availability, ffmpeg availability, git identity configured.
- **If there is no GPU**: design for CPU-first, provide a tiny-config path that finishes
  in minutes, and document how to run the heavy path on Colab/Kaggle/Lambda.
- **If on Windows**: use `pathlib` everywhere, no hardcoded `/` separators, avoid
  bash-only scripts (provide a cross-platform Python or `just`/`make` alternative), watch
  for long-path and line-ending problems (`.gitattributes` with `* text=auto`).
- Log the environment (hardware, library versions, git commit) into every benchmark
  result file automatically.

## 4. Tooling defaults (Python projects)
- Python 3.11+, `uv` for env + lockfile (fallback: `pip` + `requirements.txt`).
- `pyproject.toml` as the single config source. `src/` layout.
- `ruff` (lint + format), `mypy` or `pyright` (strict on core modules), `pytest`,
  `pytest-cov`, `pre-commit`.
- `Makefile` (or `justfile`) targets: `setup`, `lint`, `format`, `typecheck`, `test`,
  `test-fast`, `bench`, `run`, `docker-build`, `clean`.
- `.env.example` with every variable documented; real `.env` gitignored.
- Config via `pydantic-settings`; validate at startup and fail fast with a clear message.
- Logging: structured (JSON option), levels, request/correlation IDs, **no PII, no
  secrets, no raw biometrics, no embeddings in logs.**

## 5. Repo hygiene
- Required files: `README.md`, `LICENSE`, `CHANGELOG.md`, `CONTRIBUTING.md` (short),
  `SECURITY.md` (short), `.gitignore`, `.gitattributes`, `.env.example`,
  `docs/DESIGN.md`, `docs/RESULTS.md`, `docs/LIMITATIONS.md`, `DATA.md`,
  `MODEL_CARD.md` (if any model is used or trained).
- License: MIT by default **unless a dependency, dataset, or model license forces
  otherwise**. Check licenses of every dependency, dataset, and pretrained model, and
  record them in `DATA.md` / `MODEL_CARD.md`. Flag anything non-commercial, AGPL, or
  research-only prominently in the README.
- Secret scanning: `gitleaks` (or `detect-secrets`) in pre-commit and CI. If a secret is
  ever committed, tell me immediately; rotating the key is required, not just deleting
  the file.
- **Never commit**: datasets, model weights, personal images/videos, real ID documents,
  API keys, `.env`, notebooks with outputs containing personal data, large binaries.
  Provide download scripts with checksum verification instead.
- Git LFS is not used. Keep repo size small (< 50 MB).
- Tag releases with semver (`v0.1.0`), update `CHANGELOG.md` (Keep a Changelog format).

## 5b. Honest-numbers policy (most important rule)
- **Never fabricate, estimate, or copy benchmark numbers.** Every number in the README
  or `docs/RESULTS.md` must be produced by a script in `bench/` or `eval/`, saved to
  `results/*.json` with hardware info and commit hash, and rendered into tables/plots by
  another script.
- If a result is not yet measured, the README shows `TBD (not yet measured)`, never a
  plausible-looking number.
- Numbers from papers or model cards may be cited **only** as clearly labeled "reported
  by <source>", never presented as our own result.
- Report uncertainty where sample sizes are small (confidence intervals, seeds, or an
  explicit "small sample, not statistically significant" note).
- Report negative results and failures in `docs/LIMITATIONS.md`. This builds credibility.
- Train/validation/test discipline: pick thresholds and hyperparameters on validation
  only; touch the test set once for the final number.

## 6. Data, privacy, and ethics
- Treat faces, embeddings, ID documents, voices, and video of people as **sensitive
  biometric/personal data**.
- Use only: public-domain/permissively licensed data, synthetic data, or data from
  consenting people (written consent noted in a private place, never in the repo).
- Data minimization: do not store raw inputs unless a flag explicitly enables it; default
  retention = none.
- Encrypt sensitive data at rest where persisted; provide a delete/erase path.
- Defensive purpose only: never build tooling whose main use is attacking or spoofing
  real systems or people.
- Document fairness/bias limits (demographic skew of datasets, lighting/camera skew).

## 7. Testing standards
- Unit tests for all core logic; integration tests for the pipeline/API; regression
  tests for every bug fixed.
- Deterministic tests: fixed seeds, no network, no GPU required for the default test
  suite. Heavy/slow tests marked `@pytest.mark.slow` or `@pytest.mark.gpu` and skipped
  by default.
- LLM/API calls in tests are **mocked or replayed from recorded fixtures** (cassettes).
  Real-API tests are opt-in via an env var.
- Property-based tests (`hypothesis`) for parsers, validators, and numeric code where it
  makes sense.
- Coverage target: >= 70% overall, >= 85% on core logic. Coverage is a floor, not the
  goal; test behavior and edge cases.
- Every edge case listed in the project spec must map to at least one test, or be
  explicitly listed as "not covered" in `docs/LIMITATIONS.md` with a reason.

## 8. CI (GitHub Actions)
- On push and PR: lint, format check, typecheck, tests with coverage, secret scan,
  dependency audit (`pip-audit`), Docker build (if a Dockerfile exists).
- Matrix: Python 3.11 and 3.12; Linux required, Windows optional if cheap.
- Cache dependencies. Fail on warnings that matter. Show badges in README.
- Heavy benchmarks do NOT run in CI; a tiny smoke benchmark does.

## 9. README standard
Sections in order: one-line pitch, badges, demo GIF/screenshot, **Key results table**
(generated), why this exists, architecture diagram (Mermaid), quickstart (copy-paste,
works on a fresh clone in < 5 min), usage examples, configuration, how to reproduce the
benchmarks, design decisions (short, link to DESIGN.md), limitations, roadmap, license,
acknowledgements/citations.

## 10. Definition of Done (universal)
- [ ] Fresh clone -> quickstart works on a clean machine (verified)
- [ ] All spec milestones met; every edge case covered or documented
- [ ] CI green, coverage targets met, no secrets, no large files in history
- [ ] Results reproducible with one documented command; results files committed
- [ ] README + DESIGN + RESULTS + LIMITATIONS + DATA + MODEL_CARD complete
- [ ] Demo GIF/video, tagged release v0.1.0+, CHANGELOG updated
- [ ] I can explain every design decision in an interview (DESIGN.md decision log)

## 11. Anti-patterns (do not do these)
- Copy-pasting tutorial code without understanding or attribution
- Giant single-file scripts; notebooks as the only deliverable
- Unpinned dependencies; "works on my machine"
- Claims like "production-ready" or "state-of-the-art" without evidence
- Hard-coded paths, thresholds, or API keys
- Catch-all `except:` that hides errors
- Silent fallbacks that change behavior without a log line
