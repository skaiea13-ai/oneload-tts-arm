# Benchmark method

The benchmark compares two end-to-end paths on the same Arm64 machine, with the
same pinned 6-bit Qwen3-TTS model, manifest, speaker, generation settings, text,
and per-scene random seeds.

The cold baseline starts a fresh Python process for each scene. Every process
validates and loads the model, renders one WAV, writes it atomically, and exits.
The optimized path starts one Python process, validates and loads the model once,
then renders all scenes from the manifest in sequence. The default benchmark
runs three trials and alternates which path runs first. Reported timing values
are medians of wall time measured by the parent process.
The runner refuses workloads requiring more than 16 render subprocesses and
enforces one 30-minute deadline across all trials.

Wall-clock measurements include interpreter startup, model validation, model
loading, generation, and WAV output. Child output is bounded and parsed as one
strict receipt, but child-reported timing and memory fields are not published as
benchmark evidence. The parent independently measures elapsed wall time,
inventories the expected WAV files, and hashes their actual bytes. A speed result
is accepted only when every baseline and optimized WAV has the same SHA-256 hash
across all trials.
The runner writes a fresh result to `output/benchmark/apple-m4.json` and refuses
to replace an existing report. Use an empty output tree for each reproduction.

The committed result contains the Arm architecture, chip family, model identity
and revision, verified model byte count, manifest, model-lock, and dependency-lock
hashes, exact runtime source hashes, timings, and ordinal output hashes. It deliberately omits run
timestamps, operating-system and Python versions, usernames, device names,
serial numbers, file-system paths, model weights, and child-reported numeric
metrics.
