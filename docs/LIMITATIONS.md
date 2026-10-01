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
None recorded yet (no milestone with edge cases has been implemented).
