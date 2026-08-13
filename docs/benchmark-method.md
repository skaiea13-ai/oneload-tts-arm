# Benchmark method

The benchmark compares two end-to-end paths on the same Arm64 machine, with the
same pinned 6-bit Qwen3-TTS model, manifest, speaker, generation settings, text,
and per-scene random seeds.

The cold baseline starts a fresh Python process for each scene. Every process
validates and loads the model, renders one WAV, writes it atomically, and exits.
The optimized path starts one Python process, validates and loads the model once,
then renders all scenes from the manifest in sequence. The default benchmark
runs three trials and alternates which path runs first. Reported timing values
are medians; peak-memory values are the maximum observed across the trials.
The runner refuses workloads requiring more than 16 render subprocesses and
enforces one 30-minute deadline across all trials.

Wall-clock measurements include interpreter startup, model validation, model
loading, generation, and WAV output. Model-load and generation time are also
reported separately. Output hashes are compared by scene. A speed result is
accepted only when every baseline and optimized WAV has the same SHA-256 hash.
The runner writes a fresh result to `output/benchmark/apple-m4.json` and refuses
to replace an existing report. Use an empty output tree for each reproduction.

The committed result contains the Arm architecture, chip family, model identity
and revision, verified model byte count, manifest and model-lock hashes, exact
runtime source hashes, timings, and output hashes. It deliberately omits run
timestamps, operating-system and Python versions, usernames, device names,
serial numbers, file-system paths, and model weights.
