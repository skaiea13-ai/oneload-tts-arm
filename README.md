# OneLoad TTS for Arm

OneLoad turns a directory of short narration scripts into a reproducible batch
of WAV files without reloading the speech model for every scene. It runs the
Apache-2.0 Qwen3-TTS 1.7B CustomVoice 6-bit model locally through MLX on Apple
Silicon and uses the calm `Aiden` voice by default.

The optimization is intentionally narrow and measurable. A cold baseline starts
one Python process per scene, so the same verified 2.69 GB model snapshot is
loaded repeatedly. OneLoad starts one process, loads the model once, and renders
the complete validated scene manifest. Both paths use identical text, settings,
and seeds, and the benchmark rejects the quality comparison unless every WAV is
bit-identical by SHA-256.

## Why this is useful

Demo videos, tutorials, and accessibility tracks are usually assembled from
scene-sized takes because smaller clips are easier to revise. That workflow is
also where local TTS wastes time: a convenient one-shot command pays the cold
model-loading cost over and over. OneLoad preserves the editable scene workflow
while removing redundant loads.

Everything runs on the device after the model download. There is no API key,
per-generation fee, background service, uploaded script, or voice cloning.

## Requirements

- Apple Silicon Mac (`arm64`)
- Python 3.12
- `uv`
- about 3 GB of free space for the pinned model weights

## Install

```bash
uv sync --frozen --python 3.12
./scripts/download-model.sh
```

The downloader resolves the exact model revision recorded in
`model-lock.json`. Before it writes, OneLoad creates and checks an owner-protected
target and rejects links, special files, and unexpected entries. Before MLX reads
anything, OneLoad checks the complete downloaded snapshot by size and SHA-256 and
freezes those exact bytes in a private copy-on-write loader view. To keep the
snapshot elsewhere, set `ONELOAD_MODEL_DIR` to a local directory before running
the script.

For model, output, and benchmark paths, use directories owned by the current
user. OneLoad fails closed on unprotected shared writable parents and accepts a
root- or user-owned sticky temporary directory when a shared location is needed.

## Validate and render

```bash
export ONELOAD_MODEL_DIR=/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit

uv run oneload-tts validate \
  --model-path "$ONELOAD_MODEL_DIR"

uv run oneload-tts render \
  --manifest examples/demo-manifest.json \
  --model-path "$ONELOAD_MODEL_DIR" \
  --output-dir output/demo
```

The renderer writes one receipt for the batch and one receipt per scene. Public
receipts contain relative output names, model provenance, hashes, timing, audio
length, and peak model memory. They never expose the local model path.

## Reproduce the cold-versus-persistent benchmark

```bash
export ONELOAD_MODEL_DIR=/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit
./scripts/run-benchmark.sh
```

The command caps the benchmark at 16 render subprocesses and 30 minutes. It gives
every child an immutable private copy of the parent-validated
manifest, renders each scene in an isolated cold process, renders the same
manifest with one persistent model load, compares every child manifest digest
and all WAV hashes, and writes the sanitized result to
`benchmarks/apple-m4.json`. The report binds the manifest, model-lock file,
verified model byte count, and exact runtime source hashes used for the run. The
measurement method is documented in
[docs/benchmark-method.md](docs/benchmark-method.md).

### Apple M4 result

The committed three-scene run produced 17.84 seconds of audio. Across three
alternating-order trials, the cold path had a median end-to-end time of 19.766
seconds and loaded the model three times per trial. The persistent path had a
median time of 12.234 seconds and loaded it once. That is a 38.1% wall-clock
reduction and a 66.7% reduction in model loads, with no increase in measured
peak model memory. All three optimized WAV files are bit-identical to their
cold-baseline counterparts in every trial.

These numbers describe this manifest on this machine, not a universal model
speed claim. The JSON report includes source- and model-bound provenance to
rerun the same test on another Apple Silicon Mac.

## Verification

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Security and privacy boundaries are documented in
[docs/security.md](docs/security.md). Generated audio, model weights, virtual
environments, caches, and temporary benchmark files are excluded from version
control.

## License and AI disclosure

OneLoad is MIT licensed. The pinned Qwen3-TTS model is Apache-2.0 licensed.
OpenAI Codex assisted with implementation, test design, documentation, and
review. The project idea, constraints, benchmark acceptance criteria, and final
submission decisions were directed and reviewed by the entrant.
