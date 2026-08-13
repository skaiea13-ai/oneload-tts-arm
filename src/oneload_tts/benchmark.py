from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

from oneload_tts._filesystem import (
    commit_open_file,
    open_or_create_bound_directory,
    private_temporary_file,
    read_regular_file_bounded,
)
from oneload_tts.manifest import MAX_MANIFEST_BYTES, Manifest

MAX_BENCHMARK_TOKENS = 65_536
RENDER_TIMEOUT_SECONDS = 15 * 60
BENCHMARK_TIMEOUT_SECONDS = 30 * 60
MAX_BENCHMARK_RENDER_INVOCATIONS = 16
HARDWARE_PROBE_TIMEOUT_SECONDS = 5
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGED_MODEL_LOCK_PATH = PACKAGE_ROOT / "model-lock.json"
MODEL_LOCK_PATH = (
    PACKAGED_MODEL_LOCK_PATH
    if PACKAGED_MODEL_LOCK_PATH.is_file()
    else PROJECT_ROOT / "model-lock.json"
)
IMPLEMENTATION_FILES = (
    "__init__.py",
    "__main__.py",
    "_filesystem.py",
    "benchmark.py",
    "cli.py",
    "engine.py",
    "manifest.py",
)
SECURITY_CONTROL_FILES = ("download_guard.py",)


def _command_output(command: list[str]) -> str:
    result = subprocess.run(  # noqa: S603
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=HARDWARE_PROBE_TIMEOUT_SECONDS,
    )
    return result.stdout.strip()


def _hardware() -> dict[str, str]:
    chip = "unknown"
    if platform.system() == "Darwin":
        try:
            chip = _command_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"])
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    return {
        "architecture": platform.machine(),
        "chip": chip,
    }


def _run_render(
    command: list[str], environment: dict[str, str], *, timeout_seconds: float
) -> tuple[float, dict]:
    if timeout_seconds <= 0:
        raise RuntimeError("benchmark deadline exceeded")
    started = time.monotonic()
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
            timeout=min(RENDER_TIMEOUT_SECONDS, timeout_seconds),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("render subprocess timed out") from None
    except subprocess.CalledProcessError:
        raise RuntimeError("render subprocess failed") from None
    except OSError:
        raise RuntimeError("could not start render subprocess") from None
    elapsed = time.monotonic() - started
    lines = completed.stdout.splitlines()
    start = next((index for index, line in enumerate(lines) if line.lstrip().startswith("{")), -1)
    if start < 0:
        raise RuntimeError("render subprocess did not return a JSON receipt")
    return elapsed, json.loads("\n".join(lines[start:]))


def _require_matching_outputs(
    baseline_hashes: dict[str, str], optimized_hashes: dict[str, str]
) -> None:
    if baseline_hashes != optimized_hashes:
        differing = sorted(
            segment_id
            for segment_id in set(baseline_hashes) | set(optimized_hashes)
            if baseline_hashes.get(segment_id) != optimized_hashes.get(segment_id)
        )
        raise RuntimeError("baseline and persistent outputs differ for: " + ", ".join(differing))


def _require_manifest_hash(receipt: dict, expected_sha256: str) -> None:
    if receipt.get("manifest_sha256") != expected_sha256:
        raise RuntimeError("render subprocess used a different manifest")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_sha256() -> dict[str, str]:
    return {
        f"src/oneload_tts/{name}": _sha256_path(PACKAGE_ROOT / name)
        for name in IMPLEMENTATION_FILES
    }


def _security_control_sha256() -> dict[str, str]:
    return {
        f"src/oneload_tts/{name}": _sha256_path(PACKAGE_ROOT / name)
        for name in SECURITY_CONTROL_FILES
    }


def _write_manifest_snapshot(directory: Path, directory_fd: int, manifest: Manifest) -> Path:
    payload = manifest.source_bytes
    if payload is None:
        payload = read_regular_file_bounded(
            manifest.path,
            maximum_bytes=MAX_MANIFEST_BYTES,
            label="manifest",
        )
    if hashlib.sha256(payload).hexdigest() != manifest.sha256:
        raise RuntimeError("manifest changed before benchmark")
    snapshot_name = "manifest.json"
    snapshot = directory / snapshot_name
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(snapshot_name, flags, 0o600, dir_fd=directory_fd)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        os.fsync(directory_fd)
    except OSError:
        raise RuntimeError("could not freeze benchmark manifest") from None
    finally:
        if "descriptor" in locals():
            os.close(descriptor)
    return snapshot


def _run_cold_path(
    *,
    base: list[str],
    manifest: Manifest,
    output_dir: Path,
    environment: dict[str, str],
    expected_manifest_sha256: str,
    deadline: float,
) -> tuple[dict[str, float], dict[str, str]]:
    runs: list[dict] = []
    wall_seconds = 0.0
    for segment in manifest.segments:
        elapsed, receipt = _run_render(
            [
                *base,
                "--output-dir",
                str(output_dir),
                "--only",
                segment.segment_id,
            ],
            environment,
            timeout_seconds=deadline - time.monotonic(),
        )
        _require_manifest_hash(receipt, expected_manifest_sha256)
        wall_seconds += elapsed
        runs.append(receipt)
    hashes = {receipt["id"]: receipt["sha256"] for run in runs for receipt in run["receipts"]}
    return (
        {
            "wall_seconds": wall_seconds,
            "model_load_seconds": sum(float(run["model_load_seconds"]) for run in runs),
            "generation_seconds": sum(float(run["generation_seconds"]) for run in runs),
            "peak_model_memory_gb": max(float(run["peak_model_memory_gb"]) for run in runs),
        },
        hashes,
    )


def _run_persistent_path(
    *,
    base: list[str],
    output_dir: Path,
    environment: dict[str, str],
    expected_manifest_sha256: str,
    deadline: float,
) -> tuple[dict[str, float], dict[str, str], dict]:
    wall_seconds, receipt = _run_render(
        [*base, "--output-dir", str(output_dir)],
        environment,
        timeout_seconds=deadline - time.monotonic(),
    )
    _require_manifest_hash(receipt, expected_manifest_sha256)
    hashes = {item["id"]: item["sha256"] for item in receipt["receipts"]}
    return (
        {
            "wall_seconds": wall_seconds,
            "model_load_seconds": float(receipt["model_load_seconds"]),
            "generation_seconds": float(receipt["generation_seconds"]),
            "peak_model_memory_gb": float(receipt["peak_model_memory_gb"]),
        },
        hashes,
        receipt,
    )


def run_benchmark(*, manifest: Manifest, model_path: Path, trials: int = 3) -> dict[str, object]:
    if not 1 <= trials <= 9:
        raise ValueError("benchmark trials must be between 1 and 9")
    requested_tokens = 2 * trials * len(manifest.segments) * manifest.settings.max_tokens
    if requested_tokens > MAX_BENCHMARK_TOKENS:
        raise ValueError("benchmark generation budget exceeds 65,536 tokens")
    render_invocations = trials * (len(manifest.segments) + 1)
    if render_invocations > MAX_BENCHMARK_RENDER_INVOCATIONS:
        raise ValueError("benchmark exceeds 16 render subprocesses")
    environment = os.environ.copy()
    for variable in ("BASH_ENV", "ENV", "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"):
        environment.pop(variable, None)
    environment.update(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    implementation_sha256 = _implementation_sha256()
    security_control_sha256 = _security_control_sha256()
    model_lock_sha256 = _sha256_path(MODEL_LOCK_PATH)
    deadline = time.monotonic() + BENCHMARK_TIMEOUT_SECONDS
    temporary_parent = Path(tempfile.gettempdir()).resolve(strict=True)
    with tempfile.TemporaryDirectory(
        prefix="oneload-benchmark-", dir=temporary_parent
    ) as temporary:
        temporary_root = Path(temporary)
        _, temporary_fd = open_or_create_bound_directory(
            temporary_root,
            failure_message="benchmark temporary directory is not private",
        )
        try:
            manifest_snapshot = _write_manifest_snapshot(temporary_root, temporary_fd, manifest)
            base = [
                sys.executable,
                "-I",
                "-B",
                "-m",
                "oneload_tts",
                "render",
                "--manifest",
                str(manifest_snapshot),
                "--model-path",
                str(model_path.expanduser().resolve()),
            ]
            trial_results: list[dict[str, object]] = []
            reference_hashes: dict[str, str] | None = None
            optimized_receipt: dict | None = None
            for trial_index in range(trials):
                trial_root = temporary_root / f"trial-{trial_index + 1}"
                if trial_index % 2 == 0:
                    order = ["cold", "persistent"]
                    baseline, baseline_hashes = _run_cold_path(
                        base=base,
                        manifest=manifest,
                        output_dir=trial_root / "cold",
                        environment=environment,
                        expected_manifest_sha256=manifest.sha256,
                        deadline=deadline,
                    )
                    optimized, optimized_hashes, optimized_receipt = _run_persistent_path(
                        base=base,
                        output_dir=trial_root / "persistent",
                        environment=environment,
                        expected_manifest_sha256=manifest.sha256,
                        deadline=deadline,
                    )
                else:
                    order = ["persistent", "cold"]
                    optimized, optimized_hashes, optimized_receipt = _run_persistent_path(
                        base=base,
                        output_dir=trial_root / "persistent",
                        environment=environment,
                        expected_manifest_sha256=manifest.sha256,
                        deadline=deadline,
                    )
                    baseline, baseline_hashes = _run_cold_path(
                        base=base,
                        manifest=manifest,
                        output_dir=trial_root / "cold",
                        environment=environment,
                        expected_manifest_sha256=manifest.sha256,
                        deadline=deadline,
                    )
                _require_matching_outputs(baseline_hashes, optimized_hashes)
                if reference_hashes is None:
                    reference_hashes = optimized_hashes
                else:
                    _require_matching_outputs(reference_hashes, optimized_hashes)
                trial_results.append(
                    {
                        "trial": trial_index + 1,
                        "order": order,
                        "baseline": {
                            key: round(float(value), 3) for key, value in baseline.items()
                        },
                        "optimized": {
                            key: round(float(value), 3) for key, value in optimized.items()
                        },
                    }
                )
        finally:
            os.close(temporary_fd)
    if (
        _implementation_sha256() != implementation_sha256
        or _security_control_sha256() != security_control_sha256
        or _sha256_path(MODEL_LOCK_PATH) != model_lock_sha256
    ):
        raise RuntimeError("benchmark implementation changed during measurement")
    if optimized_receipt is None or reference_hashes is None:
        raise RuntimeError("benchmark produced no result")
    baseline_wall = median(float(result["baseline"]["wall_seconds"]) for result in trial_results)
    optimized_wall = median(float(result["optimized"]["wall_seconds"]) for result in trial_results)
    baseline_load_seconds = median(
        float(result["baseline"]["model_load_seconds"]) for result in trial_results
    )
    baseline_generation_seconds = median(
        float(result["baseline"]["generation_seconds"]) for result in trial_results
    )
    optimized_load_seconds = median(
        float(result["optimized"]["model_load_seconds"]) for result in trial_results
    )
    optimized_generation_seconds = median(
        float(result["optimized"]["generation_seconds"]) for result in trial_results
    )
    baseline_peak_memory_gb = max(
        float(result["baseline"]["peak_model_memory_gb"]) for result in trial_results
    )
    optimized_peak_memory_gb = max(
        float(result["optimized"]["peak_model_memory_gb"]) for result in trial_results
    )
    speedup = baseline_wall / optimized_wall
    memory_change_percent = ((optimized_peak_memory_gb / baseline_peak_memory_gb) - 1.0) * 100.0
    return {
        "schema_version": 3,
        "metric_scope": "median end-to-end wall clock including process startup and model loading",
        "hardware": _hardware(),
        "manifest_sha256": manifest.sha256,
        "segments": len(manifest.segments),
        "trials": trials,
        "model_id": optimized_receipt["model_id"],
        "model_revision": optimized_receipt["model_revision"],
        "model_license": optimized_receipt["model_license"],
        "model_verified_bytes": optimized_receipt["model_verified_bytes"],
        "model_lock_sha256": model_lock_sha256,
        "implementation_sha256": implementation_sha256,
        "security_control_sha256": security_control_sha256,
        "baseline": {
            "mode": "one cold process per scene",
            "model_loads": len(manifest.segments),
            "wall_seconds": round(baseline_wall, 3),
            "model_load_seconds": round(baseline_load_seconds, 3),
            "generation_seconds": round(baseline_generation_seconds, 3),
            "peak_model_memory_gb": baseline_peak_memory_gb,
        },
        "optimized": {
            "mode": "one persistent process for the scene manifest",
            "model_loads": 1,
            "wall_seconds": round(optimized_wall, 3),
            "model_load_seconds": round(optimized_load_seconds, 3),
            "generation_seconds": round(optimized_generation_seconds, 3),
            "audio_seconds": optimized_receipt["audio_seconds"],
            "peak_model_memory_gb": optimized_peak_memory_gb,
        },
        "improvement": {
            "wall_clock_speedup": round(speedup, 3),
            "wall_clock_reduction_percent": round((1.0 - 1.0 / speedup) * 100.0, 1),
            "model_load_reduction_percent": round((1.0 - 1.0 / len(manifest.segments)) * 100.0, 1),
            "peak_model_memory_change_percent": round(memory_change_percent, 1),
            "bit_identical_outputs": True,
        },
        "trial_results": trial_results,
        "output_sha256": reference_hashes,
    }


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    requested = path.expanduser()
    destination_name = requested.name
    if destination_name in {"", ".", ".."}:
        raise ValueError("result path must name a file")

    _, directory_fd = open_or_create_bound_directory(
        requested.parent,
        failure_message="result directory changed before writing",
    )
    try:
        with private_temporary_file(
            directory_fd,
            prefix="oneload-result",
            suffix=".tmp",
            failure_message="could not commit benchmark result",
        ) as (temporary_name, temporary_fd):
            with os.fdopen(os.dup(temporary_fd), "w", encoding="utf-8") as temporary:
                temporary.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            commit_open_file(
                temporary_fd,
                temporary_name,
                destination_name,
                directory_fd,
                failure_message="could not commit benchmark result",
            )
    finally:
        os.close(directory_fd)
