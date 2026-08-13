from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

from oneload_tts._filesystem import (
    commit_open_file,
    open_bound_directory,
    open_or_create_bound_directory,
    private_unlinked_file,
    read_regular_file_bounded,
)
from oneload_tts.engine import load_model_lock
from oneload_tts.manifest import MAX_MANIFEST_BYTES, Manifest

MAX_BENCHMARK_TOKENS = 65_536
RENDER_TIMEOUT_SECONDS = 15 * 60
BENCHMARK_TIMEOUT_SECONDS = 30 * 60
MAX_BENCHMARK_RENDER_INVOCATIONS = 16
HARDWARE_PROBE_TIMEOUT_SECONDS = 5
MAX_RENDER_STREAM_BYTES = 256 * 1024
MAX_WAV_FILE_BYTES = 256 * 1024 * 1024
MAX_RECEIPT_SAMPLE_RATE = 384_000
MAX_RECEIPT_MEMORY_GB = 1_024.0
MAX_RECEIPT_AUDIO_SECONDS = 30 * 60
MAX_RECEIPT_SEGMENT_SECONDS = 10 * 60
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGED_MODEL_LOCK_PATH = PACKAGE_ROOT / "model-lock.json"
MODEL_LOCK_PATH = (
    PACKAGED_MODEL_LOCK_PATH
    if PACKAGED_MODEL_LOCK_PATH.is_file()
    else PROJECT_ROOT / "model-lock.json"
)
UV_LOCK_PATH = PROJECT_ROOT / "uv.lock"
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


def _reject_non_finite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _run_render(
    command: list[str], environment: dict[str, str], *, timeout_seconds: float
) -> tuple[float, dict]:
    if timeout_seconds <= 0:
        raise RuntimeError("benchmark deadline exceeded")
    started = time.monotonic()
    timeout = min(RENDER_TIMEOUT_SECONDS, timeout_seconds)
    try:
        process = subprocess.Popen(  # noqa: S603
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            start_new_session=True,
        )
    except OSError:
        raise RuntimeError("could not start render subprocess") from None
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("could not capture render subprocess")

    stdout_fd = process.stdout.fileno()
    streams = selectors.DefaultSelector()
    buffers: dict[int, bytearray] = {}
    for stream in (process.stdout, process.stderr):
        os.set_blocking(stream.fileno(), False)
        streams.register(stream, selectors.EVENT_READ)
        buffers[stream.fileno()] = bytearray()
    deadline = started + timeout
    failure: str | None = None
    try:
        while streams.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = "render subprocess timed out"
                break
            for key, _ in streams.select(min(0.05, remaining)):
                try:
                    block = os.read(key.fd, 64 * 1024)
                except BlockingIOError:
                    continue
                if not block:
                    streams.unregister(key.fileobj)
                    continue
                buffer = buffers[key.fd]
                buffer.extend(block)
                if len(buffer) > MAX_RENDER_STREAM_BYTES:
                    failure = "render subprocess output exceeded the receipt limit"
                    break
            if failure is not None:
                break
            if process.poll() is not None and streams.get_map():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        return_code: int | None = None
        if failure is None:
            try:
                return_code = process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                failure = "render subprocess timed out"
        if failure is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (PermissionError, ProcessLookupError):
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    finally:
        streams.close()
        process.stdout.close()
        process.stderr.close()

    if failure is not None:
        raise RuntimeError(failure) from None
    if return_code != 0:
        raise RuntimeError("render subprocess failed") from None
    elapsed = time.monotonic() - started
    try:
        stdout = bytes(buffers[stdout_fd]).decode("utf-8")
    except UnicodeDecodeError:
        raise RuntimeError("render subprocess did not return a JSON receipt") from None
    try:
        receipt = json.loads(stdout, parse_constant=_reject_non_finite_json)
    except (json.JSONDecodeError, ValueError):
        raise RuntimeError("render subprocess did not return a JSON receipt") from None
    if not isinstance(receipt, dict):
        raise RuntimeError("render subprocess did not return a JSON receipt")
    return elapsed, receipt


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


def _bounded_number(
    payload: dict,
    key: str,
    *,
    maximum: float,
    positive: bool = False,
) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError("render subprocess returned an invalid receipt")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or parsed > maximum or (positive and parsed <= 0):
        raise RuntimeError("render subprocess returned an invalid receipt")
    return parsed


def _validated_receipt(
    receipt: dict,
    *,
    manifest: Manifest,
    selected_ids: tuple[str, ...],
    model_lock: dict,
) -> dict[str, str]:
    _require_manifest_hash(receipt, manifest.sha256)
    expected_segments = tuple(
        segment for segment in manifest.segments if segment.segment_id in selected_ids
    )
    expected_receipt_keys = {
        "status",
        "mode",
        "manifest_sha256",
        "model_id",
        "model_revision",
        "model_license",
        "model_verified_bytes",
        "speaker",
        "segments_rendered",
        "model_loads",
        "model_load_seconds",
        "generation_seconds",
        "wall_seconds",
        "audio_seconds",
        "real_time_factor",
        "peak_model_memory_gb",
        "receipts",
    }
    if (
        set(receipt) != expected_receipt_keys
        or receipt.get("status") != "ok"
        or receipt.get("mode")
        != ("persistent-batch" if len(expected_segments) > 1 else "single-cold-process")
        or receipt.get("speaker") != manifest.settings.speaker
        or receipt.get("segments_rendered") != len(expected_segments)
        or receipt.get("model_loads") != 1
        or receipt.get("model_id") != model_lock["model_id"]
        or receipt.get("model_revision") != model_lock["revision"]
        or receipt.get("model_license") != model_lock["license"]
        or receipt.get("model_verified_bytes")
        != sum(item["size"] for item in model_lock["required_files"].values())
    ):
        raise RuntimeError("render subprocess returned an invalid receipt")
    items = receipt.get("receipts")
    if not isinstance(items, list) or len(items) != len(expected_segments):
        raise RuntimeError("render subprocess returned an invalid receipt")
    hashes: dict[str, str] = {}
    durations: list[float] = []
    generation_times: list[float] = []
    peak_memories: list[float] = []
    expected_item_keys = {
        "id",
        "output",
        "sha256",
        "seed",
        "characters",
        "sample_rate",
        "samples",
        "duration_seconds",
        "generation_seconds",
        "peak_amplitude",
        "rms_amplitude",
        "peak_model_memory_gb",
    }
    for item, segment in zip(items, expected_segments, strict=True):
        if not isinstance(item, dict):
            raise RuntimeError("render subprocess returned an invalid receipt")
        sha256 = item.get("sha256")
        sample_rate = item.get("sample_rate")
        samples = item.get("samples")
        if (
            set(item) != expected_item_keys
            or item.get("id") != segment.segment_id
            or item.get("output") != segment.output.as_posix()
            or item.get("seed") != segment.seed
            or item.get("characters") != len(segment.text)
            or isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or not 1 <= sample_rate <= MAX_RECEIPT_SAMPLE_RATE
            or isinstance(samples, bool)
            or not isinstance(samples, int)
            or not 1 <= samples <= sample_rate * MAX_RECEIPT_SEGMENT_SECONDS
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise RuntimeError("render subprocess returned an invalid receipt")
        duration = _bounded_number(
            item,
            "duration_seconds",
            maximum=MAX_RECEIPT_SEGMENT_SECONDS,
            positive=True,
        )
        if abs(duration - samples / sample_rate) > 0.001:
            raise RuntimeError("render subprocess returned an invalid receipt")
        generation_times.append(
            _bounded_number(
                item,
                "generation_seconds",
                maximum=RENDER_TIMEOUT_SECONDS,
            )
        )
        for key in ("peak_amplitude", "rms_amplitude"):
            _bounded_number(item, key, maximum=16.0, positive=True)
        peak_memories.append(
            _bounded_number(
                item,
                "peak_model_memory_gb",
                maximum=MAX_RECEIPT_MEMORY_GB,
                positive=True,
            )
        )
        durations.append(duration)
        hashes[segment.segment_id] = sha256
    model_load_seconds = _bounded_number(
        receipt,
        "model_load_seconds",
        maximum=RENDER_TIMEOUT_SECONDS,
    )
    generation_seconds = _bounded_number(
        receipt,
        "generation_seconds",
        maximum=RENDER_TIMEOUT_SECONDS,
    )
    wall_seconds = _bounded_number(
        receipt,
        "wall_seconds",
        maximum=RENDER_TIMEOUT_SECONDS,
        positive=True,
    )
    audio_seconds = _bounded_number(
        receipt,
        "audio_seconds",
        maximum=MAX_RECEIPT_AUDIO_SECONDS,
        positive=True,
    )
    real_time_factor = _bounded_number(
        receipt,
        "real_time_factor",
        maximum=RENDER_TIMEOUT_SECONDS,
        positive=True,
    )
    peak_memory = _bounded_number(
        receipt,
        "peak_model_memory_gb",
        maximum=MAX_RECEIPT_MEMORY_GB,
        positive=True,
    )
    aggregate_tolerance = 0.001 * max(1, len(items))
    if (
        abs(audio_seconds - sum(durations)) > aggregate_tolerance
        or abs(generation_seconds - sum(generation_times)) > aggregate_tolerance
        or abs(peak_memory - max(peak_memories)) > 0.001
        or wall_seconds + 0.001 < model_load_seconds + generation_seconds
        or abs(real_time_factor - wall_seconds / audio_seconds) > 0.001
    ):
        raise RuntimeError("render subprocess returned an invalid receipt")
    return hashes


def _open_existing_directory(root_fd: int, parts: tuple[str, ...]) -> int:
    current_fd = os.dup(root_fd)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        for part in parts:
            expected = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            next_fd = os.open(part, flags, dir_fd=current_fd)
            opened = os.fstat(next_fd)
            current = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            identity = (expected.st_dev, expected.st_ino)
            if (
                part in {"", ".", ".."}
                or "/" in part
                or not stat.S_ISDIR(expected.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or not stat.S_ISDIR(current.st_mode)
                or (opened.st_dev, opened.st_ino) != identity
                or (current.st_dev, current.st_ino) != identity
            ):
                os.close(next_fd)
                raise RuntimeError("benchmark output directory changed")
            os.close(current_fd)
            current_fd = next_fd
    except (OSError, RuntimeError):
        os.close(current_fd)
        raise RuntimeError("benchmark output directory changed") from None
    return current_fd


def _inventory_output_files(root_fd: int, expected_files: set[str]) -> set[str]:
    expected_directories = {
        tuple(Path(relative).parts[:depth])
        for relative in expected_files
        for depth in range(1, len(Path(relative).parts))
    }
    found: set[str] = set()
    pending: list[tuple[int, tuple[str, ...]]] = [(os.dup(root_fd), ())]
    try:
        while pending:
            directory_fd, parent_parts = pending.pop()
            try:
                with os.scandir(directory_fd) as entries:
                    for entry in entries:
                        parts = (*parent_parts, entry.name)
                        relative = "/".join(parts)
                        if entry.is_symlink():
                            raise RuntimeError("benchmark output set is invalid")
                        if entry.is_dir(follow_symlinks=False):
                            if parts not in expected_directories:
                                raise RuntimeError("benchmark output set is invalid")
                            pending.append((_open_existing_directory(root_fd, parts), parts))
                        elif entry.is_file(follow_symlinks=False):
                            if relative not in expected_files:
                                raise RuntimeError("benchmark output set is invalid")
                            found.add(relative)
                        else:
                            raise RuntimeError("benchmark output set is invalid")
            finally:
                os.close(directory_fd)
    finally:
        for directory_fd, _ in pending:
            os.close(directory_fd)
    return found


def _hash_output_files(output_dir: Path, manifest: Manifest) -> dict[str, str]:
    expected_files = {segment.output.as_posix() for segment in manifest.segments}
    try:
        _, root_fd = open_bound_directory(
            output_dir,
            failure_message="benchmark output directory is missing",
        )
    except RuntimeError:
        raise RuntimeError("benchmark output directory is missing") from None
    hashes: dict[str, str] = {}
    try:
        if _inventory_output_files(root_fd, expected_files) != expected_files:
            raise RuntimeError("benchmark output set is incomplete")
        for segment in manifest.segments:
            parts = segment.output.parts
            parent_fd = _open_existing_directory(root_fd, tuple(parts[:-1]))
            file_fd: int | None = None
            try:
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                )
                expected = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
                file_fd = os.open(parts[-1], flags, dir_fd=parent_fd)
                opened = os.fstat(file_fd)
                current = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
                identity = (expected.st_dev, expected.st_ino)
                if (
                    not stat.S_ISREG(expected.st_mode)
                    or not stat.S_ISREG(opened.st_mode)
                    or not stat.S_ISREG(current.st_mode)
                    or opened.st_nlink != 1
                    or opened.st_size <= 0
                    or opened.st_size > MAX_WAV_FILE_BYTES
                    or (opened.st_dev, opened.st_ino) != identity
                    or (current.st_dev, current.st_ino) != identity
                ):
                    raise RuntimeError("benchmark output file is invalid")
                digest = hashlib.sha256()
                offset = 0
                while offset < opened.st_size:
                    block = os.pread(file_fd, min(1024 * 1024, opened.st_size - offset), offset)
                    if not block:
                        raise RuntimeError("benchmark output file changed")
                    digest.update(block)
                    offset += len(block)
                after = os.fstat(file_fd)
                if (
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                    after.st_dev,
                    after.st_ino,
                ) != (
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_ctime_ns,
                    opened.st_dev,
                    opened.st_ino,
                ):
                    raise RuntimeError("benchmark output file changed")
                hashes[segment.segment_id] = digest.hexdigest()
            except OSError:
                raise RuntimeError("benchmark output file is invalid") from None
            finally:
                if file_fd is not None:
                    os.close(file_fd)
                os.close(parent_fd)
    finally:
        os.close(root_fd)
    return hashes


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
    model_lock: dict,
    deadline: float,
) -> tuple[dict[str, float], dict[str, str]]:
    claimed_hashes: dict[str, str] = {}
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
        hashes = _validated_receipt(
            receipt,
            manifest=manifest,
            selected_ids=(segment.segment_id,),
            model_lock=model_lock,
        )
        wall_seconds += elapsed
        claimed_hashes.update(hashes)
    hashes = _hash_output_files(output_dir, manifest)
    _require_matching_outputs(claimed_hashes, hashes)
    return (
        {"wall_seconds": wall_seconds},
        hashes,
    )


def _run_persistent_path(
    *,
    base: list[str],
    manifest: Manifest,
    output_dir: Path,
    environment: dict[str, str],
    expected_manifest_sha256: str,
    model_lock: dict,
    deadline: float,
) -> tuple[dict[str, float], dict[str, str]]:
    wall_seconds, receipt = _run_render(
        [*base, "--output-dir", str(output_dir)],
        environment,
        timeout_seconds=deadline - time.monotonic(),
    )
    _require_manifest_hash(receipt, expected_manifest_sha256)
    claimed_hashes = _validated_receipt(
        receipt,
        manifest=manifest,
        selected_ids=tuple(segment.segment_id for segment in manifest.segments),
        model_lock=model_lock,
    )
    hashes = _hash_output_files(output_dir, manifest)
    _require_matching_outputs(claimed_hashes, hashes)
    return {"wall_seconds": wall_seconds}, hashes


def run_benchmark(*, manifest: Manifest, model_path: Path, trials: int = 3) -> dict[str, object]:
    if not 1 <= trials <= 9:
        raise ValueError("benchmark trials must be between 1 and 9")
    requested_tokens = 2 * trials * len(manifest.segments) * manifest.settings.max_tokens
    if requested_tokens > MAX_BENCHMARK_TOKENS:
        raise ValueError("benchmark generation budget exceeds 65,536 tokens")
    render_invocations = trials * (len(manifest.segments) + 1)
    if render_invocations > MAX_BENCHMARK_RENDER_INVOCATIONS:
        raise ValueError("benchmark exceeds 16 render subprocesses")
    model_lock = load_model_lock(MODEL_LOCK_PATH)
    implementation_sha256 = _implementation_sha256()
    security_control_sha256 = _security_control_sha256()
    model_lock_sha256 = _sha256_path(MODEL_LOCK_PATH)
    runtime_lock_sha256 = _sha256_path(UV_LOCK_PATH)
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
            for private_name in ("home", "tmp"):
                os.mkdir(private_name, mode=0o700, dir_fd=temporary_fd)
            environment = {
                "HOME": str(temporary_root / "home"),
                "TMPDIR": str(temporary_root / "tmp"),
                "PATH": "/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
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
                        model_lock=model_lock,
                        deadline=deadline,
                    )
                    optimized, optimized_hashes = _run_persistent_path(
                        base=base,
                        manifest=manifest,
                        output_dir=trial_root / "persistent",
                        environment=environment,
                        expected_manifest_sha256=manifest.sha256,
                        model_lock=model_lock,
                        deadline=deadline,
                    )
                else:
                    order = ["persistent", "cold"]
                    optimized, optimized_hashes = _run_persistent_path(
                        base=base,
                        manifest=manifest,
                        output_dir=trial_root / "persistent",
                        environment=environment,
                        expected_manifest_sha256=manifest.sha256,
                        model_lock=model_lock,
                        deadline=deadline,
                    )
                    baseline, baseline_hashes = _run_cold_path(
                        base=base,
                        manifest=manifest,
                        output_dir=trial_root / "cold",
                        environment=environment,
                        expected_manifest_sha256=manifest.sha256,
                        model_lock=model_lock,
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
        or _sha256_path(UV_LOCK_PATH) != runtime_lock_sha256
    ):
        raise RuntimeError("benchmark implementation changed during measurement")
    if reference_hashes is None:
        raise RuntimeError("benchmark produced no result")
    baseline_wall = median(float(result["baseline"]["wall_seconds"]) for result in trial_results)
    optimized_wall = median(float(result["optimized"]["wall_seconds"]) for result in trial_results)
    speedup = baseline_wall / optimized_wall
    public_hashes = {
        f"scene-{index:02d}": reference_hashes[segment.segment_id]
        for index, segment in enumerate(manifest.segments, start=1)
    }
    return {
        "schema_version": 4,
        "metric_scope": "median end-to-end wall clock including process startup and model loading",
        "hardware": _hardware(),
        "manifest_sha256": manifest.sha256,
        "segments": len(manifest.segments),
        "trials": trials,
        "model_id": model_lock["model_id"],
        "model_revision": model_lock["revision"],
        "model_license": model_lock["license"],
        "model_verified_bytes": sum(item["size"] for item in model_lock["required_files"].values()),
        "model_lock_sha256": model_lock_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
        "implementation_sha256": implementation_sha256,
        "security_control_sha256": security_control_sha256,
        "baseline": {
            "mode": "one cold process per scene",
            "render_processes": len(manifest.segments),
            "wall_seconds": round(baseline_wall, 3),
        },
        "optimized": {
            "mode": "one persistent process for the scene manifest",
            "render_processes": 1,
            "wall_seconds": round(optimized_wall, 3),
        },
        "improvement": {
            "wall_clock_speedup": round(speedup, 3),
            "wall_clock_reduction_percent": round((1.0 - 1.0 / speedup) * 100.0, 1),
            "render_process_reduction_percent": round(
                (1.0 - 1.0 / len(manifest.segments)) * 100.0, 1
            ),
            "bit_identical_outputs": True,
        },
        "trial_results": trial_results,
        "output_sha256": public_hashes,
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
        with private_unlinked_file(
            directory_fd,
            prefix="oneload-result",
            suffix=".tmp",
            failure_message="could not commit benchmark result",
        ) as temporary_fd:
            with os.fdopen(os.dup(temporary_fd), "w", encoding="utf-8") as temporary:
                temporary.write(
                    json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            commit_open_file(
                temporary_fd,
                destination_name,
                directory_fd,
                failure_message="could not commit benchmark result",
            )
    finally:
        os.close(directory_fd)
