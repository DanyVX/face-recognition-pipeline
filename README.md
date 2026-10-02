# face-recognition-pipeline

End-to-end, benchmarked face recognition pipeline: detect, align, embed, index, verify (1:1) and identify (1:N), exposed through a small API and CLI. The focus is rigorous measurement and privacy-aware engineering.

[![CI](https://github.com/DanyVX/face-recognition-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/DanyVX/face-recognition-pipeline/actions/workflows/ci.yml)

> **Status: early development (M0 done, nothing recognizes faces yet).**

> **Model license notice.** The code in this repo is MIT-licensed. The pretrained face models this
> project plans to use (InsightFace model packs) are, per the InsightFace README, available for
> **non-commercial research purposes only**. Do **not** treat this project as commercially
> deployable. Details and the unverified items are in [`MODEL_CARD.md`](MODEL_CARD.md).

> **Not a surveillance tool.** Faces and embeddings are sensitive biometric data. This project is for
> research and education only.

## Key results

<!-- RESULTS:START -->
TBD (not yet measured)
<!-- RESULTS:END -->

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and git.

```bash
git clone https://github.com/DanyVX/face-recognition-pipeline.git
cd face-recognition-pipeline
uv run python scripts/dev.py setup     # or: make setup
uv run python scripts/dev.py test      # or: make test
uv run frp env                         # prints hardware/library/git info as JSON
```

Other tasks: `lint`, `format`, `typecheck`, `test-fast`, `clean` (run
`uv run python scripts/dev.py --help` for the full list). Copy `.env.example` to `.env` to configure.

## Documents

- [`CLAUDE.md`](CLAUDE.md): engineering rules for this repo
- [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md): project specification and milestones
- [`docs/DESIGN.md`](docs/DESIGN.md): architecture and decision log
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md), [`DATA.md`](DATA.md), [`MODEL_CARD.md`](MODEL_CARD.md)

## License

Code: MIT (see [`LICENSE`](LICENSE)). Models and datasets carry their own licenses; see the notice above.
