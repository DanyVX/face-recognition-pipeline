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

## M1 (detection and image loading): observations and conventions
Observations below come from one public-domain portrait (see `DATA.md`), so they are anecdotes
about this model on this image, **not measured rates**. Measured detection-failure rates on LFW
come in M6.
- A 32x32 and a 64x64 version of the portrait: no face found (the face is too small).
- A 160 px face inside a 4000x160 strip: not found (the letterbox shrinks it to about 25 px).
- Rotated copies (90/180/270 degrees) and a mirrored copy were all detected on this image. Whether
  the *embedding* of a rotated face is useful is a separate question (M6 robustness slices).
- A frame that slices through the face still yields a detection, flagged `truncated`.

Conventions that change behavior (documented, tested):
- RGBA/palette-with-transparency images are composited on a white background.
- 16-bit grayscale is mapped linearly (65535 -> 255); an image using only a small part of the
  16-bit range will look dark. 16-bit RGB is read as 8-bit by Pillow before we see it.
- Animated GIF/WebP: first frame only (or reject via `animated="reject"`).
- `min_face_px` is measured in the pixels of the array handed to the detector (after any
  downscale by the loader), not in original-image pixels.

All M1 spec edge cases are covered by tests; none are listed as not covered.

## M2 (alignment and pose): observations and conventions
- On one portrait, the 180-degree-rotated copy gave visibly worse landmark localization (mouth
  corners about 12 px apart instead of about 40) and a fit error of 0.31, against 0.07-0.14 for
  the other orientations. Rotated faces are therefore usable but less reliable; measured effect
  on verification accuracy comes with the M6 robustness slices.
- `DEFAULT_MAX_FIT_ERROR = 0.4` and the pose-gate thresholds are provisional (see D-012/D-013).
- Alignment resamples with bilinear interpolation and zero-pads outside the source image.
- The stored reference image (`tests/fixtures/align_reference.png`) is produced by this
  implementation and guards against accidental change; its correctness is separately checked by
  the marker-position test, which does not use the stored image.

All M2 spec edge cases (degenerate/collinear landmarks, landmarks outside the image, extreme
pose, mirrored landmark order, NaN landmarks) are covered by tests; none are listed as not covered.

## M3 (embedding): observations and conventions
- Bitwise determinism was observed on one machine (Intel i5-7300U, ONNX Runtime 1.30.0 CPU,
  default thread count). Results across CPUs, ONNX Runtime versions or thread counts, and on
  GPU, are not measured; floating-point summation order can differ. Tests use tolerances for
  anything but repeated calls on the same machine.
- Concurrency was tested with at most 8 callers on a 2-core machine.
- GPU execution (CUDA provider) has not been run; only the CPU fallback path is exercised.
- fp16 vs fp32: a probe on random input gave cosine 0.999997 (4 samples, not a result). The real
  measurement on LFW crops is `bench/precision.py` (M6); until then there is no fp16 claim.
