# Model card (DRAFT, M0)

No model is used or shipped yet. This file records licensing facts gathered so far; it is completed
in M1/M3 when models are wired in. **Anything marked UNVERIFIED must not be relied on.**

## InsightFace model packs (SCRFD detector + ArcFace embedder)

| Item | Status |
|---|---|
| InsightFace **code** license | MIT, per the repository README (retrieved 2026-10-01) |
| Pretrained **model** license | **Non-commercial research use only**, per the repository README (retrieved 2026-10-01) |
| Commercial use | README says to contact the vendor for a commercial license (`buffalo_l` pack: `recognition-oss-pack@insightface.ai`, as listed in the README) |
| License text shipped with the model files | **None.** `buffalo_l.zip` contains only five `.onnx` files (checked 2026-10-01), so the README statement above is the only license information available. The primary README should still be re-read before release |
| Pack | `buffalo_l.zip` from the InsightFace GitHub release v0.7, 288,621,354 bytes, SHA-256 `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f` (recorded on 2026-10-01, trust on first use) |
| `det_10g.onnx` (SCRFD detector) | 16,923,827 bytes, SHA-256 `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |
| `w600k_r50.onnx` (ArcFace ResNet-50 embedder, 512-d) | 174,383,860 bytes, SHA-256 `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |

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
