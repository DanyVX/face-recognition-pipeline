# Contributing

This is a solo portfolio project, but it follows normal engineering rules (details in
[`CLAUDE.md`](CLAUDE.md)).

- One milestone per branch, named `m<N>-<slug>`; merge to `main` only with CI green.
- [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `test:`, `docs:`,
  `chore:`, `perf:`, `refactor:`, `ci:`. One logical change per commit.
- Run `make lint typecheck test` (or `uv run python scripts/dev.py <task>` on Windows) before pushing.
- Every number in the docs must come from a script in `bench/` or `eval/`; unmeasured results say
  `TBD (not yet measured)`.
- Never commit datasets, model weights, face images of real people, or secrets.
