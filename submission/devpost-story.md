# OneLoad TTS for Arm

## Project overview

I make demo videos one short scene at a time. If a sentence needs another take, I
can replace that clip instead of rendering the whole narration again. The annoying
part is that a simple local TTS command often starts a new process and reloads the
same multi-gigabyte model for every scene. The scripts stay private, but I wait for
the same setup work over and over.

OneLoad keeps the pinned 6-bit Qwen3-TTS model alive for the whole scene manifest.
It runs locally through MLX on an Apple M4, uses the calm Aiden voice by default,
and writes a separate WAV for every editable scene. There is no hosted service,
API key, usage fee, or script upload.

This is a Mobile AI track project: it optimizes an on-device speech workflow for
an Arm-powered laptop. I created OneLoad during the challenge submission period.

## Functionality and output

A creator supplies one bounded JSON manifest with the narration text, output
filenames, generation settings, and a deterministic seed for each scene. OneLoad
validates the manifest and the complete pinned model snapshot. It then gives MLX
a private frozen view of those exact bytes, loads the model once, renders the scenes
in order, and writes every WAV atomically. The receipt records relative filenames,
audio properties, timings, hashes, and public model provenance without exposing
the local model path.

The repository also includes a cold-versus-persistent benchmark. The baseline
starts one fresh process per scene. The optimized path uses the same text,
settings, seeds, model revision, and machine, but keeps one process alive. The
benchmark bounds and validates each child receipt, then reopens and hashes the
actual WAV files itself. It refuses to publish a speed comparison if any output
hash differs. Its report also binds the validated manifest, model lock, frozen
dependency lock, verified model bytes, and runtime source hashes.

On the committed three-scene Apple M4 run, using the median of three trials with
alternating execution order:

- Cold baseline: 19.823 seconds, three render processes per trial
- OneLoad: 12.209 seconds, one render process per trial
- Wall-clock reduction: 38.4%
- Render-process reduction: 66.7%
- Audio equivalence: all three WAV files are bit-identical by SHA-256

This number covers the whole workflow. It does not mean the model generates
speech 38.4% faster. The difference comes from removing repeated process startup,
model validation, and model loading from the multi-scene workflow.

## Why it should win

I built OneLoad around a recurring annoyance in my own creative workflow. Local
speech models keep narration private and avoid usage fees on an Arm laptop, but a
command that reloads the model for every scene spends more time on setup. OneLoad
still gives me one WAV per scene without repeating that loading work.

The benchmark result is not a loose timing claim. The repository locks the model
revision, exact byte sizes, and SHA-256 digests for the complete downloaded
snapshot. The benchmark runs offline. Its JSON report records the source and
model hashes used for the run, so another Apple Silicon user can repeat the test
or substitute a different scene manifest.

## Setup instructions

Requirements: an Apple Silicon Mac, Python 3.12, `uv`, and about 3 GB of free disk
space for the model.

```bash
uv sync --frozen --python 3.12
./scripts/download-model.sh

export ONELOAD_MODEL_DIR=/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit

uv run oneload-tts validate --model-path "$ONELOAD_MODEL_DIR"

uv run oneload-tts render \
  --manifest examples/demo-manifest.json \
  --model-path "$ONELOAD_MODEL_DIR" \
  --output-dir output/demo

./scripts/run-benchmark.sh
```

The model downloader uses the exact revision in `model-lock.json` and refuses
unprotected targets or filesystem links. After the download, rendering and
benchmarking run locally with Hugging Face telemetry disabled and offline mode
enabled. The benchmark also caps its subprocess count and total runtime.

## Built with

Python 3.12, MLX, MLX Audio, Qwen3-TTS CustomVoice 1.7B 6-bit, SoundFile, HTTPX,
uv, pytest, and Ruff on Apple M4 Arm64.

## Challenges and lessons

Timing was the easy part. The harder part was proving that the faster workflow did
not quietly change the audio. Both paths reset the random state for each scene and
compare the final PCM WAV hashes. I kept child-reported diagnostics out of the
public result and used only the end-to-end time measured by the parent process. I
also alternated the run order to reduce timing bias.

## What's next

Next I want to test longer manifests and additional Arm client machines. An
optional warm worker for editors would be useful too, along with per-scene retry
and resume that preserve deterministic receipts.

## AI use disclosure

OpenAI Codex assisted with implementation, tests, documentation, and review. The
project direction, constraints, benchmark acceptance criteria, and final
submission decisions were directed and reviewed by the entrant. The narration
voice is generated locally with the disclosed Qwen3-TTS model.
