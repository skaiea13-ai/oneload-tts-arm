from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oneload_tts._filesystem import (
    commit_open_file,
    copy_regular_file_from_descriptor,
    open_bound_directory,
    open_or_create_bound_directory,
    open_relative_directory,
    private_staging_directory,
    private_temporary_file,
)
from oneload_tts.manifest import Manifest, Segment, canonical_output_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGED_MODEL_LOCK_PATH = Path(__file__).with_name("model-lock.json")
MODEL_LOCK_PATH = (
    PACKAGED_MODEL_LOCK_PATH
    if PACKAGED_MODEL_LOCK_PATH.is_file()
    else PROJECT_ROOT / "model-lock.json"
)
MAX_AUDIO_SECONDS_PER_SEGMENT = 10 * 60
MAX_AUDIO_SECONDS_PER_MANIFEST = 30 * 60


@dataclass(frozen=True)
class _FileFingerprint:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


def _fingerprint_state(state: os.stat_result) -> _FileFingerprint:
    return _FileFingerprint(
        device=state.st_dev,
        inode=state.st_ino,
        size=state.st_size,
        modified_ns=state.st_mtime_ns,
        changed_ns=state.st_ctime_ns,
    )


def _fingerprint(path: Path) -> _FileFingerprint:
    return _fingerprint_state(path.stat())


def _fingerprint_descriptor(file_descriptor: int) -> _FileFingerprint:
    return _fingerprint_state(os.fstat(file_descriptor))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model_lock(path: Path = MODEL_LOCK_PATH) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {"model_id", "revision", "license", "required_files"}
    missing = required.difference(raw)
    if missing:
        raise RuntimeError(f"model lock is missing: {', '.join(sorted(missing))}")
    if not isinstance(raw["required_files"], dict) or not raw["required_files"]:
        raise RuntimeError("model lock required_files must be a non-empty object")
    for relative, specification in raw["required_files"].items():
        if not isinstance(relative, str) or not relative:
            raise RuntimeError("model lock file names must be non-empty strings")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError("model lock file names must stay inside the model directory")
        if not isinstance(specification, dict):
            raise RuntimeError(f"model lock entry must be an object: {relative}")
        expected_size = specification.get("size")
        expected_sha256 = specification.get("sha256")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size <= 0
        ):
            raise RuntimeError(f"model lock size must be a positive integer: {relative}")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise RuntimeError(f"model lock sha256 must be lowercase hexadecimal: {relative}")
    return raw


def _model_files(model_root: Path, expected_files: set[str]) -> set[str]:
    """Enumerate only locked model paths, rejecting the first unexpected entry."""

    expected_directories = {
        tuple(Path(relative).parts[:depth])
        for relative in expected_files
        for depth in range(1, len(Path(relative).parts))
    }
    files: set[str] = set()
    pending: list[tuple[Path, tuple[str, ...]]] = [(model_root, ())]
    while pending:
        directory, parent_parts = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    parts = (*parent_parts, entry.name)
                    if parts == (".cache",):
                        continue
                    relative = "/".join(parts)
                    if entry.is_symlink():
                        raise RuntimeError("model directory contains a symbolic link")
                    if entry.is_dir(follow_symlinks=False):
                        if parts not in expected_directories:
                            raise RuntimeError("model directory does not match the locked snapshot")
                        pending.append((Path(entry.path), parts))
                    elif entry.is_file(follow_symlinks=False):
                        if relative not in expected_files:
                            raise RuntimeError("model directory does not match the locked snapshot")
                        files.add(relative)
                    else:
                        raise RuntimeError("model directory does not match the locked snapshot")
        except OSError:
            raise RuntimeError("model directory changed while enumerating") from None
    return files


def _same_stat_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _open_existing_model_directory(
    model_root_fd: int,
    parts: tuple[str, ...],
    *,
    failure_message: str,
) -> int:
    """Open an existing model subdirectory without following mutable path components."""

    directory_fd = os.dup(model_root_fd)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        for part in parts:
            if part in {"", ".", ".."} or "/" in part:
                raise RuntimeError(failure_message)
            expected = os.stat(part, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(expected.st_mode) or not stat.S_ISDIR(expected.st_mode):
                raise RuntimeError(failure_message)
            next_fd: int | None = None
            try:
                next_fd = os.open(part, flags, dir_fd=directory_fd)
                opened = os.fstat(next_fd)
                current = os.stat(part, dir_fd=directory_fd, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or not stat.S_ISDIR(current.st_mode)
                    or not _same_stat_identity(expected, opened)
                    or not _same_stat_identity(opened, current)
                ):
                    raise RuntimeError(failure_message)
            except (OSError, RuntimeError):
                if next_fd is not None:
                    os.close(next_fd)
                raise
            os.close(directory_fd)
            directory_fd = next_fd
    except (OSError, RuntimeError):
        os.close(directory_fd)
        raise RuntimeError(failure_message) from None
    return directory_fd


def _model_files_from_descriptor(model_root_fd: int, expected_files: set[str]) -> set[str]:
    """Enumerate the exact locked snapshot through descriptor-bound directories."""

    expected_directories = {
        tuple(Path(relative).parts[:depth])
        for relative in expected_files
        for depth in range(1, len(Path(relative).parts))
    }
    files: set[str] = set()
    pending: list[tuple[int, tuple[str, ...]]] = [(os.dup(model_root_fd), ())]
    try:
        while pending:
            directory_fd, parent_parts = pending.pop()
            try:
                with os.scandir(directory_fd) as entries:
                    for entry in entries:
                        parts = (*parent_parts, entry.name)
                        if parts == (".cache",):
                            continue
                        relative = "/".join(parts)
                        state = os.stat(entry.name, dir_fd=directory_fd, follow_symlinks=False)
                        if stat.S_ISLNK(state.st_mode):
                            raise RuntimeError("model directory contains a symbolic link")
                        if stat.S_ISDIR(state.st_mode):
                            if parts not in expected_directories:
                                raise RuntimeError(
                                    "model directory does not match the locked snapshot"
                                )
                            child_fd = _open_existing_model_directory(
                                directory_fd,
                                (entry.name,),
                                failure_message="model directory changed while enumerating",
                            )
                            pending.append((child_fd, parts))
                        elif stat.S_ISREG(state.st_mode):
                            if relative not in expected_files:
                                raise RuntimeError(
                                    "model directory does not match the locked snapshot"
                                )
                            files.add(relative)
                        else:
                            raise RuntimeError("model directory does not match the locked snapshot")
            finally:
                os.close(directory_fd)
    except OSError:
        raise RuntimeError("model directory changed while enumerating") from None
    finally:
        for directory_fd, _ in pending:
            os.close(directory_fd)
    return files


def _open_locked_model_file(
    model_root_fd: int,
    relative: str,
    *,
    expected_size: int,
    size_failure_message: str,
    failure_message: str,
) -> tuple[int, int, str, _FileFingerprint]:
    """Open one locked model leaf without following or blocking on a replacement."""

    parts = tuple(Path(relative).parts)
    if not parts or any(part in {"", ".", ".."} or "/" in part for part in parts):
        raise RuntimeError(failure_message)
    parent_fd = _open_existing_model_directory(
        model_root_fd,
        parts[:-1],
        failure_message=failure_message,
    )
    source_fd: int | None = None
    leaf = parts[-1]
    try:
        expected = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(expected.st_mode) or not stat.S_ISREG(expected.st_mode):
            raise RuntimeError(failure_message)
        if expected.st_size != expected_size:
            raise RuntimeError(size_failure_message)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        source_fd = os.open(leaf, flags, dir_fd=parent_fd)
        opened = os.fstat(source_fd)
        current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or not _same_stat_identity(expected, opened)
            or not _same_stat_identity(opened, current)
            or opened.st_size != expected_size
        ):
            raise RuntimeError(failure_message)
        opened_fingerprint = _fingerprint_state(opened)
        if opened_fingerprint != _fingerprint_state(expected):
            raise RuntimeError(failure_message)
        return source_fd, parent_fd, leaf, opened_fingerprint
    except (OSError, RuntimeError):
        if source_fd is not None:
            os.close(source_fd)
        os.close(parent_fd)
        raise RuntimeError(
            size_failure_message
            if "expected" in locals()
            and stat.S_ISREG(expected.st_mode)
            and expected.st_size != expected_size
            else failure_message
        ) from None


def _require_stable_open_model_file(
    source_fd: int,
    parent_fd: int,
    leaf: str,
    before: _FileFingerprint,
    *,
    failure_message: str,
) -> _FileFingerprint:
    try:
        after_state = os.fstat(source_fd)
        current_state = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        raise RuntimeError(failure_message) from None
    after = _fingerprint_state(after_state)
    if (
        not stat.S_ISREG(after_state.st_mode)
        or not stat.S_ISREG(current_state.st_mode)
        or before != after
        or after != _fingerprint_state(current_state)
    ):
        raise RuntimeError(failure_message)
    return after


def _sha256_descriptor_exact(
    file_descriptor: int,
    expected_size: int,
    *,
    failure_message: str,
) -> str:
    """Hash exactly a locked byte count and reject truncation or growth."""

    digest = hashlib.sha256()
    offset = 0
    remaining = expected_size
    try:
        while remaining:
            block = os.pread(file_descriptor, min(1024 * 1024, remaining), offset)
            if not block:
                raise RuntimeError(failure_message)
            digest.update(block)
            offset += len(block)
            remaining -= len(block)
        if os.pread(file_descriptor, 1, expected_size):
            raise RuntimeError(failure_message)
    except OSError:
        raise RuntimeError(failure_message) from None
    return digest.hexdigest()


def _validate_model_state(
    model_path: Path, lock: dict[str, Any]
) -> tuple[Path, dict[str, object], dict[str, _FileFingerprint]]:
    try:
        resolved, model_root_fd = open_bound_directory(
            model_path,
            failure_message="pinned model directory does not exist",
        )
    except RuntimeError:
        raise RuntimeError("pinned model directory does not exist")
    expected_files = set(lock["required_files"])
    verified: dict[str, int] = {}
    fingerprints: dict[str, _FileFingerprint] = {}
    try:
        if _model_files_from_descriptor(model_root_fd, expected_files) != expected_files:
            raise RuntimeError("model directory does not match the locked snapshot")
        for relative, specification in lock["required_files"].items():
            expected_size = specification["size"]
            source_fd, parent_fd, leaf, before = _open_locked_model_file(
                model_root_fd,
                relative,
                expected_size=expected_size,
                size_failure_message="model file size mismatch with the locked snapshot",
                failure_message="model file changed during validation",
            )
            try:
                actual_sha256 = _sha256_descriptor_exact(
                    source_fd,
                    expected_size,
                    failure_message="model file changed during validation",
                )
                after = _require_stable_open_model_file(
                    source_fd,
                    parent_fd,
                    leaf,
                    before,
                    failure_message="model file changed during validation",
                )
            finally:
                os.close(source_fd)
                os.close(parent_fd)
            expected_sha256 = specification["sha256"]
            if actual_sha256 != expected_sha256:
                raise RuntimeError("model file sha256 mismatch with the locked snapshot")
            verified[relative] = after.size
            fingerprints[relative] = after
    finally:
        os.close(model_root_fd)
    receipt = {
        "model_id": lock["model_id"],
        "model_revision": lock["revision"],
        "model_license": lock["license"],
        "verified_files": verified,
        "verified_bytes": sum(verified.values()),
    }
    return resolved, receipt, fingerprints


def validate_model(model_path: Path, lock: dict[str, Any]) -> dict[str, object]:
    _, receipt, _ = _validate_model_state(model_path, lock)
    return receipt


def _require_unchanged_model(model_root: Path, fingerprints: dict[str, _FileFingerprint]) -> None:
    expected_files = set(fingerprints)
    if _model_files(model_root, expected_files) != expected_files:
        raise RuntimeError("model directory changed while loading")
    for relative, expected in fingerprints.items():
        candidate = model_root / relative
        if not candidate.is_file() or _fingerprint(candidate) != expected:
            raise RuntimeError("model file changed while loading")


def _same_file(left: _FileFingerprint, right: _FileFingerprint) -> bool:
    return (
        left.device == right.device
        and left.inode == right.inode
        and left.size == right.size
        and left.modified_ns == right.modified_ns
    )


@contextmanager
def _verified_model_view(
    model_path: Path,
    lock: dict[str, Any],
) -> Iterator[tuple[int, dict[str, object], dict[str, _FileFingerprint]]]:
    try:
        resolved, model_root_fd = open_bound_directory(
            model_path,
            failure_message="pinned model directory does not exist",
        )
    except RuntimeError:
        raise RuntimeError("pinned model directory does not exist")
    expected_files = set(lock["required_files"])
    parent_fd: int | None = None
    try:
        if _model_files_from_descriptor(model_root_fd, expected_files) != expected_files:
            raise RuntimeError("model directory does not match the locked snapshot")
        _, parent_fd = open_bound_directory(
            resolved.parent,
            failure_message="could not bind the verified model snapshot",
        )
        with private_staging_directory(
            parent_fd,
            prefix="oneload-model-view",
            failure_message="could not bind the verified model snapshot",
        ) as staging_fd:
            view_fd = open_relative_directory(
                staging_fd,
                ("model",),
                failure_message="could not bind the verified model snapshot",
            )
            directory_fds: dict[tuple[str, ...], int] = {(): view_fd}
            view_entries: list[tuple[tuple[str, ...], str, _FileFingerprint]] = []
            operation_failed = False
            try:
                bound_fingerprints: dict[str, _FileFingerprint] = {}
                verified: dict[str, int] = {}
                for relative, specification in lock["required_files"].items():
                    relative_path = Path(relative)
                    parent_parts = tuple(relative_path.parts[:-1])
                    for depth in range(1, len(parent_parts) + 1):
                        parts = parent_parts[:depth]
                        if parts not in directory_fds:
                            directory_fds[parts] = open_relative_directory(
                                directory_fds[parts[:-1]],
                                (parts[-1],),
                                failure_message="could not bind the verified model snapshot",
                            )
                    target_parent_fd = directory_fds[parent_parts]
                    expected_size = specification["size"]
                    source_fd, source_parent_fd, source_name, source_opened = (
                        _open_locked_model_file(
                            model_root_fd,
                            relative,
                            expected_size=expected_size,
                            size_failure_message=(
                                "model file size mismatch with the locked snapshot"
                            ),
                            failure_message="model directory changed while binding",
                        )
                    )
                    try:
                        target_fd = copy_regular_file_from_descriptor(
                            source_fd,
                            target_parent_fd,
                            relative_path.name,
                            expected_size=expected_size,
                            failure_message="could not freeze the verified model snapshot",
                        )
                        try:
                            frozen = _fingerprint_descriptor(target_fd)
                            view_entries.append((parent_parts, relative_path.name, frozen))
                            if frozen.size != expected_size:
                                raise RuntimeError(
                                    "model file size mismatch with the locked snapshot"
                                )
                            if (
                                _sha256_descriptor_exact(
                                    target_fd,
                                    expected_size,
                                    failure_message=(
                                        "model file size mismatch with the locked snapshot"
                                    ),
                                )
                                != specification["sha256"]
                            ):
                                raise RuntimeError(
                                    "model file sha256 mismatch with the locked snapshot"
                                )
                            after = _fingerprint_descriptor(target_fd)
                        finally:
                            os.close(target_fd)
                        if frozen != after:
                            raise RuntimeError("model file changed while binding")
                        _require_stable_open_model_file(
                            source_fd,
                            source_parent_fd,
                            source_name,
                            source_opened,
                            failure_message="model file changed while binding",
                        )
                        bound_fingerprints[relative] = after
                        verified[relative] = after.size
                        view_entries[-1] = (parent_parts, relative_path.name, after)
                    finally:
                        os.close(source_fd)
                        os.close(source_parent_fd)
                receipt = {
                    "model_id": lock["model_id"],
                    "model_revision": lock["revision"],
                    "model_license": lock["license"],
                    "verified_files": verified,
                    "verified_bytes": sum(verified.values()),
                }
                yield view_fd, receipt, bound_fingerprints
            except OSError:
                operation_failed = True
                raise RuntimeError("could not bind the verified model snapshot") from None
            except BaseException:
                operation_failed = True
                raise
            finally:
                cleanup_failed = False
                for parent_parts, name, expected in reversed(view_entries):
                    try:
                        current = os.stat(
                            name,
                            dir_fd=directory_fds[parent_parts],
                            follow_symlinks=False,
                        )
                        if not stat.S_ISREG(current.st_mode) or (
                            current.st_dev,
                            current.st_ino,
                        ) != (expected.device, expected.inode):
                            cleanup_failed = True
                        else:
                            os.unlink(name, dir_fd=directory_fds[parent_parts])
                    except OSError:
                        cleanup_failed = True
                for parts in sorted(
                    (parts for parts in directory_fds if parts),
                    key=len,
                    reverse=True,
                ):
                    child_fd = directory_fds[parts]
                    try:
                        opened = os.fstat(child_fd)
                        current = os.stat(
                            parts[-1],
                            dir_fd=directory_fds[parts[:-1]],
                            follow_symlinks=False,
                        )
                        if not stat.S_ISDIR(current.st_mode) or (opened.st_dev, opened.st_ino) != (
                            current.st_dev,
                            current.st_ino,
                        ):
                            cleanup_failed = True
                        else:
                            os.rmdir(parts[-1], dir_fd=directory_fds[parts[:-1]])
                    except OSError:
                        cleanup_failed = True
                    finally:
                        os.close(child_fd)
                try:
                    opened = os.fstat(view_fd)
                    current = os.stat("model", dir_fd=staging_fd, follow_symlinks=False)
                    if not stat.S_ISDIR(current.st_mode) or (opened.st_dev, opened.st_ino) != (
                        current.st_dev,
                        current.st_ino,
                    ):
                        cleanup_failed = True
                    else:
                        os.rmdir("model", dir_fd=staging_fd)
                except OSError:
                    cleanup_failed = True
                finally:
                    os.close(view_fd)
                if cleanup_failed and not operation_failed:
                    raise RuntimeError("could not remove the private model view") from None
    finally:
        if parent_fd is not None:
            os.close(parent_fd)
        os.close(model_root_fd)


@contextmanager
def _bound_working_directory(directory_fd: int) -> Iterator[Path]:
    """Use an already-open directory object for all relative model loads."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        original_fd = os.open(".", flags)
        os.fchdir(directory_fd)
    except OSError:
        if "original_fd" in locals():
            os.close(original_fd)
        raise RuntimeError("could not enter the verified model snapshot") from None
    try:
        yield Path(".")
    finally:
        try:
            os.fchdir(original_fd)
        except OSError:
            raise RuntimeError("could not restore the working directory") from None
        finally:
            os.close(original_fd)


def _seed_everything(seed: int) -> None:
    import mlx.core as mx
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def _sha256_descriptor(file_descriptor: int) -> str:
    digest = hashlib.sha256()
    original_offset = os.lseek(file_descriptor, 0, os.SEEK_CUR)
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    for block in iter(lambda: os.read(file_descriptor, 1024 * 1024), b""):
        digest.update(block)
    os.lseek(file_descriptor, original_offset, os.SEEK_SET)
    return digest.hexdigest()


def _write_wav_at(
    directory_fd: int,
    destination_name: str,
    audio: object,
    sample_rate: int,
) -> str:
    import soundfile as sf

    if destination_name in {"", ".", ".."} or "/" in destination_name:
        raise RuntimeError("invalid WAV destination")
    try:
        with private_temporary_file(
            directory_fd,
            prefix="oneload-wav",
            suffix=".tmp",
            failure_message="could not write WAV output",
        ) as (temporary_name, temporary_fd):
            sf.write(
                temporary_fd,
                audio,
                sample_rate,
                subtype="PCM_16",
                format="WAV",
                closefd=False,
            )
            os.fsync(temporary_fd)
            before_hash = os.fstat(temporary_fd)
            digest = _sha256_descriptor(temporary_fd)
            after_hash = os.fstat(temporary_fd)
            if before_hash != after_hash:
                raise RuntimeError("WAV output changed while hashing")
            commit_open_file(
                temporary_fd,
                temporary_name,
                destination_name,
                directory_fd,
                failure_message="could not write WAV output",
            )
        return digest
    except OSError:
        raise RuntimeError("could not write WAV output") from None


def _write_wav_atomic(path: Path, audio: object, sample_rate: int) -> str:
    requested = path.expanduser()
    _, directory_fd = open_or_create_bound_directory(
        requested.parent,
        failure_message="output directory changed before writing",
    )
    try:
        return _write_wav_at(directory_fd, requested.name, audio, sample_rate)
    finally:
        os.close(directory_fd)


def _render_segment(
    *,
    model: object,
    manifest: Manifest,
    segment: Segment,
    output_root_fd: int,
) -> dict[str, object]:
    import mlx.core as mx
    import numpy as np

    _seed_everything(segment.seed)
    mx.reset_peak_memory()
    started = time.monotonic()
    parts: list[object] = []
    sample_rate: int | None = None
    total_samples = 0
    peak_model_memory_gb = 0.0
    with redirect_stdout(io.StringIO()):
        for result in model.generate_custom_voice(
            text=segment.text,
            speaker=manifest.settings.speaker,
            language=manifest.settings.language,
            instruct=manifest.settings.instruction,
            temperature=manifest.settings.temperature,
            max_tokens=manifest.settings.max_tokens,
            top_k=manifest.settings.top_k,
            top_p=manifest.settings.top_p,
            repetition_penalty=manifest.settings.repetition_penalty,
            verbose=False,
        ):
            current_sample_rate = int(result.sample_rate)
            if sample_rate is None:
                sample_rate = current_sample_rate
            elif current_sample_rate != sample_rate:
                raise RuntimeError(f"inconsistent sample rates for {segment.segment_id}")
            part = np.asarray(result.audio, dtype=np.float32).reshape(-1)
            total_samples += int(part.size)
            if total_samples > sample_rate * MAX_AUDIO_SECONDS_PER_SEGMENT:
                raise RuntimeError(f"audio duration limit exceeded for {segment.segment_id}")
            parts.append(part)
            peak_model_memory_gb = max(peak_model_memory_gb, float(result.peak_memory_usage))
    generation_seconds = time.monotonic() - started
    if not parts or sample_rate is None:
        raise RuntimeError(f"model returned no audio for {segment.segment_id}")
    audio = np.concatenate(parts)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError(f"invalid waveform for {segment.segment_id}")
    peak = float(np.max(np.abs(audio)))
    rms = float(math.sqrt(float(np.mean(np.square(audio, dtype=np.float64)))))
    if peak <= 0.0 or rms <= 0.0:
        raise RuntimeError(f"silent waveform for {segment.segment_id}")
    parent_fd = open_relative_directory(
        output_root_fd,
        tuple(segment.output.parts[:-1]),
        failure_message="segment output directory changed before writing",
    )
    try:
        output_sha256 = _write_wav_at(parent_fd, segment.output.name, audio, sample_rate)
    finally:
        os.close(parent_fd)
    receipt = {
        "id": segment.segment_id,
        "output": segment.output.as_posix(),
        "sha256": output_sha256,
        "seed": segment.seed,
        "characters": len(segment.text),
        "sample_rate": sample_rate,
        "samples": int(audio.size),
        "duration_seconds": round(audio.size / sample_rate, 3),
        "generation_seconds": round(generation_seconds, 3),
        "peak_amplitude": round(peak, 6),
        "rms_amplitude": round(rms, 6),
        "peak_model_memory_gb": round(peak_model_memory_gb, 3),
    }
    mx.clear_cache()
    return receipt


def render_manifest(
    *, manifest: Manifest, model_path: Path, output_dir: Path, only: str | None = None
) -> dict[str, object]:
    from mlx_audio.tts.utils import load_model

    lock = load_model_lock()
    selected = tuple(
        segment for segment in manifest.segments if only is None or segment.segment_id == only
    )
    if not selected:
        raise RuntimeError(f"segment not found: {only}")
    destination_keys = {canonical_output_key(segment.output) for segment in selected}
    if len(destination_keys) != len(selected):
        raise RuntimeError("selected segments resolve to duplicate output files")
    requested_output_root = output_dir.expanduser()
    _, output_root_fd = open_or_create_bound_directory(
        requested_output_root,
        failure_message="output directory changed before rendering",
    )
    try:
        with _verified_model_view(model_path, lock) as (
            bound_model_fd,
            model_validation,
            model_fingerprints,
        ):
            with _bound_working_directory(bound_model_fd) as bound_model:
                wall_started = time.monotonic()
                load_started = time.monotonic()
                try:
                    with redirect_stdout(io.StringIO()):
                        model = load_model(bound_model)
                except Exception:
                    raise RuntimeError("pinned model could not be loaded") from None
                _require_unchanged_model(bound_model, model_fingerprints)
                model_load_seconds = time.monotonic() - load_started
                receipts: list[dict[str, object]] = []
                for segment in selected:
                    receipt = _render_segment(
                        model=model,
                        manifest=manifest,
                        segment=segment,
                        output_root_fd=output_root_fd,
                    )
                    receipts.append(receipt)
                    if sum(float(item["duration_seconds"]) for item in receipts) > (
                        MAX_AUDIO_SECONDS_PER_MANIFEST
                    ):
                        raise RuntimeError("manifest audio duration exceeds 1,800 seconds")
                _require_unchanged_model(bound_model, model_fingerprints)
    finally:
        os.close(output_root_fd)
    wall_seconds = time.monotonic() - wall_started
    audio_seconds = sum(float(receipt["duration_seconds"]) for receipt in receipts)
    return {
        "status": "ok",
        "mode": "persistent-batch" if len(selected) > 1 else "single-cold-process",
        "manifest_sha256": manifest.sha256,
        "model_id": model_validation["model_id"],
        "model_revision": model_validation["model_revision"],
        "model_license": model_validation["model_license"],
        "model_verified_bytes": model_validation["verified_bytes"],
        "speaker": manifest.settings.speaker,
        "segments_rendered": len(receipts),
        "model_loads": 1,
        "model_load_seconds": round(model_load_seconds, 3),
        "generation_seconds": round(
            sum(float(receipt["generation_seconds"]) for receipt in receipts), 3
        ),
        "wall_seconds": round(wall_seconds, 3),
        "audio_seconds": round(audio_seconds, 3),
        "real_time_factor": round(wall_seconds / audio_seconds, 4),
        "peak_model_memory_gb": max(float(receipt["peak_model_memory_gb"]) for receipt in receipts),
        "receipts": receipts,
    }
