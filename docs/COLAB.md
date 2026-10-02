# Running the heavy benchmarks on a GPU (Colab / Kaggle / any CUDA machine)

> **Status: not verified.** The development machine has no GPU, so these steps have not been run
> here. They describe what should work; if a step fails, fix this document. GPU latency rows in
> `docs/RESULTS.md` stay `TBD (not yet measured)` until a result file produced this way is
> committed.

## What a GPU changes
Only inference speed. The accuracy benchmarks (`lfw_verification`, `open_set`, `robustness`,
`calibrate_quality`) are CPU-feasible (hours, not days, on a 2-core laptop) and their numbers do
not depend on the provider beyond tiny floating-point differences. The latency benchmark is the
one that needs a GPU for GPU rows.

## Steps (sketch)
```bash
git clone https://github.com/DanyVX/face-recognition-pipeline.git && cd face-recognition-pipeline
pip install uv
uv sync
uv pip install onnxruntime-gpu          # replaces the CPU build in the project environment
uv run python scripts/download_models.py
uv run python scripts/download_lfw.py
uv run python -m bench.prepare_lfw      # detect + align (use the GPU provider via ONNX_PROVIDERS)
uv run python -m bench.embed_lfw
ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider \
  uv run python -m bench.latency --name latency_cuda
```
Then download `results/latency_cuda.json` (it records the providers actually used, the GPU
machine's hardware and the git commit) and commit it. Check `providers` inside the file: if it
lists only `CPUExecutionProvider`, CUDA was not available and the run is a CPU run.

## Notes
- LFW and the model weights are downloaded by script with checksum verification; never upload
  them to a notebook's shared storage or commit them.
- The InsightFace weights are for non-commercial research use only (see `MODEL_CARD.md`).
