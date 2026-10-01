# Data (DRAFT, M0)

**No data is committed to this repo, and none is downloaded yet.** Datasets are fetched by script
(with checksum verification) into the gitignored `data/` directory. Faces are sensitive biometric
data; see `CLAUDE.md` section 6.

## Labeled Faces in the Wild (LFW)

| Item | Status |
|---|---|
| Purpose | 1:1 verification (standard 10-fold, 6000 pairs) and open-set identification (M6) |
| Official source | http://vis-www.cs.umass.edu/lfw/ (not reachable from the dev machine on 2026-10-01, so **not read**) |
| Terms of use / license | **UNVERIFIED.** A Kaggle mirror listing claims CC0, but a third-party mirror is not authoritative. Read the official page and record the exact terms before any download or redistribution claim. |
| Download size, file checksums | TBD: measured when `scripts/download_lfw.py` is written; must be under the 500 MB approval limit |
| Known caveats to document | Demographically skewed, near-saturated for modern models, some label noise (details to be sourced from the dataset page/papers, not from memory) |

## Synthetic and test fixtures
Unit-test fixtures are synthetic or license-clear images only. Each fixture's provenance and license
is listed here when added.

## Harder verification sets (CFP-FP, AgeDB-30)
Not obtained. Access terms are UNVERIFIED. Likely documented as a gap in `docs/LIMITATIONS.md`.
