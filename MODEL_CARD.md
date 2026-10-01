# Model card (DRAFT, M0)

No model is used or shipped yet. This file records licensing facts gathered so far; it is completed
in M1/M3 when models are wired in. **Anything marked UNVERIFIED must not be relied on.**

## InsightFace model packs (SCRFD detector + ArcFace embedder)

| Item | Status |
|---|---|
| InsightFace **code** license | MIT, per the repository README (retrieved 2026-10-01) |
| Pretrained **model** license | **Non-commercial research use only**, per the repository README (retrieved 2026-10-01) |
| Commercial use | README says to contact the vendor for a commercial license (`buffalo_l` pack: `recognition-oss-pack@insightface.ai`, as listed in the README) |
| Exact license text shipped with the model files | **UNVERIFIED**: still to be read from the downloaded pack and the primary README |
| Per-model checksums | TBD: recorded when `scripts/download_models.py` is written |

Source: https://github.com/deepinsight/insightface (README, section on licenses). The README text was
retrieved through a summarizing web-fetch tool, so **re-read the primary text before release** and
copy the exact wording here.

Quoted from the README as retrieved: "The training data containing the annotation (and the models
trained with these data) are available for non-commercial research purposes only."

### Consequences for this repo
- The repo's own code is MIT, but the full pipeline **must not be described as commercially
  deployable** while it depends on these weights.
- A visible notice is in the README. Models are downloaded by script, never committed.

## Second, open-license backend

**Not yet investigated (UNVERIFIED).** The spec asks for an open-license alternative behind the same
interface if feasible. In M3 I will search for candidates, read each license from its primary
source, and either implement one or document in `docs/LIMITATIONS.md` why not.

## Intended use, metrics, limitations
TBD: filled from measured results (see `docs/RESULTS.md`) and `docs/LIMITATIONS.md`.
