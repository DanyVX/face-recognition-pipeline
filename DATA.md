# Data (DRAFT, M0)

**No data is committed to this repo, and none is downloaded yet.** Datasets are fetched by script
(with checksum verification) into the gitignored `data/` directory. Faces are sensitive biometric
data; see `CLAUDE.md` section 6.

## Labeled Faces in the Wild (LFW)

| Item | Status |
|---|---|
| Purpose | 1:1 verification (standard 10-fold, 6000 pairs) and open-set identification (M6) |
| Official source | http://vis-www.cs.umass.edu/lfw/ (unreachable from the dev machine; the page was read through an archive.org snapshot dated 2024 on 2026-10-01) |
| Archive | `lfw.tgz`, 173 MB, md5 `a17d05bd522c52d84eca14327a23d494`, as stated on that page |
| Mirror used by scikit-learn | figshare file 5976018, SHA-256 `055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0` (from scikit-learn's source). The `pairs.txt` mirror (file 5976006) was downloaded on 2026-10-01 and its SHA-256 `ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592` matches scikit-learn's pinned value |
| License | **No explicit license text was found** on the page snapshot. A Kaggle mirror listing claims CC0, which is not authoritative. The images were collected from the web, so their individual copyrights are not ours to relicense. Treat as research-only and do not redistribute |
| Disclaimer on the page | LFW "should not be used to conclude that an algorithm is suitable for any commercial purpose"; it notes under-representation of many groups (few children, no babies, few people over 80, relatively few women) |
| Known label errors | Not yet checked against the dataset's errata page |

## Synthetic and test fixtures
Unit-test fixtures are synthetic or license-clear images only. Each fixture's provenance and license
is listed here when added.

## Harder verification sets (CFP-FP, AgeDB-30)
Not obtained. Access terms are UNVERIFIED. Likely documented as a gap in `docs/LIMITATIONS.md`.

### Fixtures committed to the repo
| File | What | Provenance and license |
|---|---|---|
| `tests/fixtures/astronaut.jpg` | 512x512 portrait of astronaut Eileen Collins | Bundled as `skimage.data.astronaut()`. scikit-image's documentation states it was "downloaded from the NASA Great Images database" with "No known copyright restrictions, released into the public domain" (read 2026-10-01). It shows a real person: use it only for tests and demos |
