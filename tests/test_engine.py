from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import oneload_tts._filesystem as filesystem
import oneload_tts.benchmark as benchmark
import oneload_tts.engine as engine
from oneload_tts.benchmark import (
    _require_matching_outputs,
    _run_render,
    run_benchmark,
    write_json_atomic,
)
from oneload_tts.engine import (
    _bound_working_directory,
    _fingerprint,
    _require_unchanged_model,
    _verified_model_view,
    _write_wav_atomic,
    load_model_lock,
    sha256_file,
    validate_model,
)
from oneload_tts.manifest import load_manifest


def _single_file_lock(model: Path) -> dict[str, object]:
    weight = model / "weight.bin"
    return {
        "model_id": "example/model",
        "revision": "abc123",
        "license": "Apache-2.0",
        "required_files": {
            "weight.bin": {
                "size": weight.stat().st_size,
                "sha256": sha256_file(weight),
            }
        },
    }


def test_validate_model_checks_exact_file_sizes_and_hashes(tmp_path: Path) -> None:
    model = tmp_path / "model"
    tokenizer = model / "speech_tokenizer"
    tokenizer.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"model")
    (tokenizer / "model.safetensors").write_bytes(b"token")
    lock = {
        "model_id": "example/model",
        "revision": "abc123",
        "license": "Apache-2.0",
        "required_files": {
            "model.safetensors": {
                "size": 5,
                "sha256": sha256_file(model / "model.safetensors"),
            },
            "speech_tokenizer/model.safetensors": {
                "size": 5,
                "sha256": sha256_file(tokenizer / "model.safetensors"),
            },
        },
    }

    result = validate_model(model, lock)

    assert result["verified_bytes"] == 10
    assert result["model_revision"] == "abc123"


def test_validate_model_skips_cache_without_descending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    cache = model / ".cache" / "nested"
    cache.mkdir(parents=True)
    (model / "weight.bin").write_bytes(b"locked")
    (cache / "ignored.bin").write_bytes(b"ignored")
    lock = _single_file_lock(model)
    original_scandir = engine.os.scandir

    def guarded_scandir(path):
        if not isinstance(path, int):
            assert ".cache" not in Path(path).parts
        return original_scandir(path)

    monkeypatch.setattr(engine.os, "scandir", guarded_scandir)

    assert validate_model(model, lock)["verified_bytes"] == 6


def test_validate_model_rejects_unexpected_directory_without_descending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    unexpected = model / "unexpected"
    unexpected.mkdir(parents=True)
    (model / "weight.bin").write_bytes(b"locked")
    (unexpected / "many.bin").write_bytes(b"extra")
    lock = _single_file_lock(model)
    original_scandir = engine.os.scandir

    def guarded_scandir(path):
        if not isinstance(path, int):
            assert Path(path) != unexpected
        return original_scandir(path)

    monkeypatch.setattr(engine.os, "scandir", guarded_scandir)

    with pytest.raises(RuntimeError, match="locked snapshot"):
        validate_model(model, lock)


def test_validate_model_rejects_size_mismatch(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "weight.bin").write_bytes(b"short")
    lock = {
        "model_id": "example/model",
        "revision": "abc123",
        "license": "Apache-2.0",
        "required_files": {
            "weight.bin": {
                "size": 99,
                "sha256": sha256_file(model / "weight.bin"),
            }
        },
    }

    with pytest.raises(RuntimeError, match="size mismatch"):
        validate_model(model, lock)


def test_validate_model_rejects_hash_mismatch(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "weight.bin").write_bytes(b"first")
    expected_hash = sha256_file(model / "weight.bin")
    (model / "weight.bin").write_bytes(b"other")
    lock = {
        "model_id": "example/model",
        "revision": "abc123",
        "license": "Apache-2.0",
        "required_files": {
            "weight.bin": {
                "size": 5,
                "sha256": expected_hash,
            }
        },
    }

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        validate_model(model, lock)


def test_validate_model_rejects_unlocked_snapshot_file(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    locked = model / "weight.bin"
    locked.write_bytes(b"model")
    (model / "config.json").write_text("{}", encoding="utf-8")
    lock = {
        "model_id": "example/model",
        "revision": "abc123",
        "license": "Apache-2.0",
        "required_files": {
            "weight.bin": {"size": 5, "sha256": sha256_file(locked)},
        },
    }

    with pytest.raises(RuntimeError, match="locked snapshot"):
        validate_model(model, lock)


def test_validate_model_rejects_symlinked_snapshot_file(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    external = tmp_path / "external.bin"
    external.write_bytes(b"model")
    (model / "weight.bin").symlink_to(external)
    lock = {
        "model_id": "example/model",
        "revision": "abc123",
        "license": "Apache-2.0",
        "required_files": {
            "weight.bin": {"size": 5, "sha256": sha256_file(external)},
        },
    }

    with pytest.raises(RuntimeError, match="symbolic link"):
        validate_model(model, lock)


def test_validate_model_rejects_leaf_replacement_during_descriptor_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    locked = model / "weight.bin"
    locked.write_bytes(b"model")
    lock = _single_file_lock(model)
    original_open = engine.os.open
    swapped = False

    def swapping_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and path == "weight.bin" and kwargs.get("dir_fd") is not None:
            locked.unlink()
            os.mkfifo(locked)
            swapped = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(engine.os, "open", swapping_open)

    with pytest.raises(RuntimeError, match="model"):
        validate_model(model, lock)

    assert swapped


def test_descriptor_copy_rejects_source_growth_beyond_locked_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"model")
    source_fd = os.open(source, os.O_RDONLY)
    target = tmp_path / "target"
    target.mkdir()
    parent_fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
    original_pread = filesystem.os.pread
    grew = False

    def growing_pread(file_descriptor: int, size: int, offset: int) -> bytes:
        nonlocal grew
        if file_descriptor == source_fd and not grew:
            with source.open("ab") as handle:
                handle.write(b"-unbounded")
            grew = True
        return original_pread(file_descriptor, size, offset)

    monkeypatch.setattr(filesystem, "_clone_file_from_descriptor", lambda *args: False)
    monkeypatch.setattr(filesystem.os, "pread", growing_pread)
    try:
        with pytest.raises(RuntimeError, match="could not freeze test model"):
            filesystem.copy_regular_file_from_descriptor(
                source_fd,
                parent_fd,
                "weight.bin",
                expected_size=5,
                failure_message="could not freeze test model",
            )
    finally:
        os.close(source_fd)
        os.close(parent_fd)

    assert grew
    assert not (target / "weight.bin").exists()


def test_descriptor_copy_fallback_preserves_locked_regular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"model")
    source_fd = os.open(source, os.O_RDONLY)
    target = tmp_path / "target"
    target.mkdir()
    parent_fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
    monkeypatch.setattr(filesystem, "_clone_file_from_descriptor", lambda *args: False)
    target_fd: int | None = None
    try:
        target_fd = filesystem.copy_regular_file_from_descriptor(
            source_fd,
            parent_fd,
            "weight.bin",
            expected_size=5,
            failure_message="could not freeze test model",
        )
        assert os.pread(target_fd, 6, 0) == b"model"
        assert os.fstat(target_fd).st_size == 5
    finally:
        if target_fd is not None:
            os.close(target_fd)
        os.close(source_fd)
        os.close(parent_fd)


def test_model_fingerprint_detects_change_after_validation(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    locked = model / "weight.bin"
    locked.write_bytes(b"first")
    fingerprints = {"weight.bin": _fingerprint(locked)}
    locked.write_bytes(b"other")

    with pytest.raises(RuntimeError, match="changed while loading"):
        _require_unchanged_model(model, fingerprints)


def test_verified_model_view_survives_source_directory_swap(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "weight.bin").write_bytes(b"model")
    lock = _single_file_lock(model)
    saved_model = tmp_path / "saved-model"

    with _verified_model_view(model, lock) as (view_fd, receipt, fingerprints):
        with _bound_working_directory(view_fd) as view:
            model.rename(saved_model)
            model.mkdir()
            (model / "weight.bin").write_bytes(b"evil!")
            try:
                assert (view / "weight.bin").read_bytes() == b"model"
                assert receipt["verified_bytes"] == 5
                _require_unchanged_model(view, fingerprints)
            finally:
                shutil.rmtree(model)
                saved_model.rename(model)


def test_verified_model_view_freezes_in_place_source_change(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    weight = model / "weight.bin"
    weight.write_bytes(b"model")
    lock = _single_file_lock(model)

    with _verified_model_view(model, lock) as (view_fd, _, fingerprints):
        with _bound_working_directory(view_fd) as view:
            weight.write_bytes(b"other")

            assert (view / "weight.bin").read_bytes() == b"model"
            assert (view / "weight.bin").stat().st_ino != weight.stat().st_ino
            _require_unchanged_model(view, fingerprints)


def test_verified_model_view_stays_bound_when_private_ancestor_is_retargeted(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "weight.bin").write_bytes(b"model")
    lock = _single_file_lock(model)

    with _verified_model_view(model, lock) as (view_fd, _, fingerprints):
        staging = next(
            candidate
            for candidate in tmp_path.iterdir()
            if candidate.name.startswith(".oneload-model-view.")
            and candidate.name.endswith(".stage")
        )
        saved_staging = tmp_path / "saved-model-view"
        staging.rename(saved_staging)
        (staging / "model").mkdir(parents=True)
        (staging / "model" / "weight.bin").write_bytes(b"evil!")
        try:
            with _bound_working_directory(view_fd) as view:
                assert (view / "weight.bin").read_bytes() == b"model"
                _require_unchanged_model(view, fingerprints)
        finally:
            shutil.rmtree(staging)
            saved_staging.rename(staging)


def test_private_staging_directory_rejects_create_open_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, parent_fd = filesystem.open_bound_directory(
        tmp_path,
        failure_message="could not bind test directory",
    )
    original_open = filesystem.os.open
    saved = tmp_path / "saved-created-stage"
    substituted: Path | None = None

    def swapping_open(path, flags, *args, **kwargs):
        nonlocal substituted
        if (
            substituted is None
            and isinstance(path, str)
            and path.startswith(".oneload-model-view.")
            and path.endswith(".stage")
        ):
            substituted = tmp_path / path
            substituted.rename(saved)
            substituted.mkdir(mode=0o700)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(filesystem.os, "open", swapping_open)
    try:
        with pytest.raises(RuntimeError, match="could not bind model view"):
            with filesystem.private_staging_directory(
                parent_fd,
                prefix="oneload-model-view",
                failure_message="could not bind model view",
            ):
                pass
    finally:
        os.close(parent_fd)
        if substituted is not None and substituted.exists():
            substituted.rmdir()
        if saved.exists():
            saved.rmdir()


def test_private_staging_directory_rejects_unprotected_parent(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    parent_fd = os.open(shared, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(RuntimeError, match="protected parent required"):
            with filesystem.private_staging_directory(
                parent_fd,
                prefix="oneload-test",
                failure_message="protected parent required",
            ):
                pass
    finally:
        os.close(parent_fd)


def test_relative_bound_directory_anchors_the_live_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = tmp_path / "live"
    substitute = tmp_path / "substitute"
    saved_live = tmp_path / "saved-live"
    live.mkdir()
    substitute.mkdir()
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    original_cwd_fd = os.open(".", flags)
    original_open = filesystem.os.open
    swapped = False
    bound_fd: int | None = None

    def swapping_open(path, open_flags, *args, **kwargs):
        nonlocal swapped
        if (
            not swapped
            and kwargs.get("dir_fd") is None
            and os.fspath(path) in {".", os.fspath(live)}
        ):
            live.rename(saved_live)
            substitute.rename(live)
            swapped = True
        return original_open(path, open_flags, *args, **kwargs)

    try:
        os.chdir(live)
        monkeypatch.setattr(filesystem.os, "open", swapping_open)
        _, bound_fd = filesystem.open_or_create_bound_directory(
            Path("results"),
            failure_message="could not bind relative output",
        )
        assert swapped
        assert os.path.samestat(os.fstat(bound_fd), os.stat(saved_live / "results"))
        assert not (live / "results").exists()
    finally:
        if bound_fd is not None:
            os.close(bound_fd)
        os.fchdir(original_cwd_fd)
        os.close(original_cwd_fd)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS ACL behavior")
def test_private_staging_directory_rejects_extended_allow_acl(tmp_path: Path) -> None:
    protected = tmp_path / "acl-parent"
    protected.mkdir()
    subprocess.run(  # noqa: S603
        [
            "/bin/chmod",
            "+a",
            (
                "everyone allow list,add_file,search,delete,add_subdirectory,delete_child,"
                "readattr,writeattr,readextattr,writeextattr,readsecurity,file_inherit,"
                "directory_inherit"
            ),
            str(protected),
        ],
        check=True,
    )
    parent_fd = os.open(protected, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(RuntimeError, match="protected parent required"):
            with filesystem.private_staging_directory(
                parent_fd,
                prefix="oneload-test",
                failure_message="protected parent required",
            ):
                pass
    finally:
        os.close(parent_fd)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS ACL behavior")
def test_private_staging_directory_allows_deny_only_acl(tmp_path: Path) -> None:
    protected = tmp_path / "acl-parent"
    protected.mkdir()
    subprocess.run(  # noqa: S603
        ["/bin/chmod", "+a", "everyone deny writeextattr", str(protected)],
        check=True,
    )
    parent_fd = os.open(protected, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with filesystem.private_staging_directory(
            parent_fd,
            prefix="oneload-test",
            failure_message="protected parent required",
        ):
            pass
    finally:
        os.close(parent_fd)


def test_model_lock_has_no_machine_path() -> None:
    lock = load_model_lock()

    serialized = json.dumps(lock)
    assert "/Users/" not in serialized
    assert "/Volumes/" not in serialized
    assert "local_path" not in lock


def test_write_json_atomic_and_hash(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "result.json"

    write_json_atomic(destination, {"status": "ok", "value": 3})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "ok",
        "value": 3,
    }
    assert len(sha256_file(destination)) == 64


def test_write_json_atomic_does_not_follow_predictable_temp_symlink(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("original", encoding="utf-8")
    planted = tmp_path / ".result.json.tmp"
    planted.symlink_to(victim)

    destination = tmp_path / "result.json"
    write_json_atomic(destination, {"status": "ok"})

    assert victim.read_text(encoding="utf-8") == "original"
    assert planted.is_symlink()
    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "ok"}


def test_write_json_atomic_replaces_final_symlink_without_following_it(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("original", encoding="utf-8")
    destination = tmp_path / "result.json"
    destination.symlink_to(victim)

    write_json_atomic(destination, {"status": "ok"})

    assert victim.read_text(encoding="utf-8") == "original"
    assert not destination.is_symlink()
    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "ok"}


def test_write_json_atomic_rejects_temporary_entry_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result.json"
    original_replace = benchmark.os.replace
    retargeted = False

    def retargeting_replace(source, target, *args, **kwargs) -> None:
        nonlocal retargeted
        temporary = tmp_path / source
        saved_staging = tmp_path / "saved-result-temp"
        temporary.rename(saved_staging)
        temporary.write_text('{"status":"attacker"}', encoding="utf-8")
        retargeted = True
        original_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(benchmark.os, "replace", retargeting_replace)

    with pytest.raises(RuntimeError, match="could not commit benchmark result"):
        write_json_atomic(destination, {"status": "intended"})

    assert retargeted
    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "attacker"}


def test_write_json_atomic_rejects_parent_swap_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "results"
    parent.mkdir()
    saved_parent = tmp_path / "saved-results"
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = parent / "result.json"
    original_open = filesystem.os.open
    swapped = False

    def swapping_open(path: os.PathLike[str] | str, flags: int, *args, **kwargs) -> int:
        nonlocal swapped
        if not swapped and path == parent.name and kwargs.get("dir_fd") is not None:
            swapped = True
            parent.rename(saved_parent)
            parent.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(filesystem.os, "open", swapping_open)
    try:
        with pytest.raises(RuntimeError, match="result directory changed before writing"):
            write_json_atomic(destination, {"status": "ok"})
    finally:
        if parent.is_symlink():
            parent.unlink()
        if saved_parent.exists():
            saved_parent.rename(parent)

    assert swapped
    assert not (outside / "result.json").exists()


def test_write_json_atomic_rejects_intermediate_component_retarget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    saved_shared = tmp_path / "saved-shared"
    outside = tmp_path / "outside"
    (outside / "results").mkdir(parents=True)
    destination = shared / "results" / "result.json"
    original_open = filesystem.os.open
    swapped = False

    def swapping_open(path: os.PathLike[str] | str, flags: int, *args, **kwargs) -> int:
        nonlocal swapped
        if not swapped and path == shared.name and kwargs.get("dir_fd") is not None:
            swapped = True
            shared.rename(saved_shared)
            shared.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(filesystem.os, "open", swapping_open)
    try:
        with pytest.raises(RuntimeError, match="result directory changed before writing"):
            write_json_atomic(destination, {"status": "ok"})
    finally:
        if shared.is_symlink():
            shared.unlink()
        if saved_shared.exists():
            saved_shared.rename(shared)

    assert swapped
    assert not (outside / "results" / "result.json").exists()


def test_write_wav_atomic_uses_and_hashes_the_open_descriptor(tmp_path: Path) -> None:
    destination = tmp_path / "audio.wav"
    audio = np.asarray([0.0, 0.25, -0.25, 0.0], dtype=np.float32)

    digest = _write_wav_atomic(destination, audio, 24_000)
    decoded, sample_rate = sf.read(destination)

    assert digest == sha256_file(destination)
    assert sample_rate == 24_000
    assert decoded.shape == audio.shape


def test_write_wav_atomic_rejects_temporary_entry_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "audio.wav"
    original_replace = engine.os.replace
    retargeted = False

    def retargeting_replace(source, target, *args, **kwargs) -> None:
        nonlocal retargeted
        temporary = tmp_path / source
        saved_staging = tmp_path / "saved-wav-temp"
        temporary.rename(saved_staging)
        temporary.write_bytes(b"attacker")
        retargeted = True
        original_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(engine.os, "replace", retargeting_replace)
    audio = np.asarray([0.0, 0.25, -0.25, 0.0], dtype=np.float32)

    with pytest.raises(RuntimeError, match="could not write WAV output"):
        _write_wav_atomic(destination, audio, 24_000)

    assert retargeted
    assert destination.read_bytes() == b"attacker"


def test_benchmark_rejects_nonidentical_audio_hashes() -> None:
    with pytest.raises(RuntimeError, match="baseline and persistent outputs differ"):
        _require_matching_outputs(
            {"intro": "baseline-hash"},
            {"intro": "persistent-hash"},
        )


def test_benchmark_accepts_identical_audio_hashes() -> None:
    _require_matching_outputs(
        {"intro": "same-hash"},
        {"intro": "same-hash"},
    )


def test_benchmark_rejects_excessive_render_process_count_before_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "defaults": {"speaker": "Aiden", "language": "English", "max_tokens": 1},
                "segments": [
                    {
                        "id": f"scene-{index}",
                        "text": "Safe text.",
                        "output": f"scene-{index}.wav",
                        "seed": index,
                    }
                    for index in range(32)
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    monkeypatch.setattr(
        benchmark,
        "_run_render",
        lambda *args, **kwargs: pytest.fail("render should not start"),
    )

    with pytest.raises(ValueError, match="16 render subprocesses"):
        run_benchmark(manifest=manifest, model_path=tmp_path / "model", trials=3)


def test_benchmark_children_use_parent_manifest_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    original = {
        "schema_version": 1,
        "defaults": {"speaker": "Aiden", "language": "English", "max_tokens": 16},
        "segments": [{"id": "scene", "text": "Original text.", "output": "scene.wav", "seed": 1}],
    }
    replacement = {
        **original,
        "segments": [
            {"id": "scene", "text": "Replacement text.", "output": "scene.wav", "seed": 1}
        ],
    }
    manifest_path.write_text(json.dumps(original), encoding="utf-8")
    manifest = load_manifest(manifest_path)
    manifest_path.write_text(json.dumps(replacement), encoding="utf-8")
    replacement_hash = load_manifest(manifest_path).sha256
    child_hashes: list[str] = []

    def fake_run_render(
        command: list[str], environment: dict[str, str], *, timeout_seconds: float
    ) -> tuple[float, dict]:
        del environment
        assert 0 < timeout_seconds <= benchmark.BENCHMARK_TIMEOUT_SECONDS
        child_path = Path(command[command.index("--manifest") + 1])
        child_manifest = load_manifest(child_path)
        child_hashes.append(child_manifest.sha256)
        return 0.1, {
            "manifest_sha256": child_manifest.sha256,
            "model_id": "example/model",
            "model_revision": "abc123",
            "model_license": "Apache-2.0",
            "model_verified_bytes": 123,
            "model_load_seconds": 0.1,
            "generation_seconds": 0.1,
            "peak_model_memory_gb": 1.0,
            "audio_seconds": 1.0,
            "receipts": [{"id": "scene", "sha256": child_manifest.sha256}],
        }

    monkeypatch.setattr(benchmark, "_run_render", fake_run_render)
    monkeypatch.setattr(benchmark, "_hardware", lambda: {})

    result = run_benchmark(manifest=manifest, model_path=tmp_path / "model", trials=1)

    assert child_hashes == [manifest.sha256, manifest.sha256]
    assert manifest.sha256 != replacement_hash
    assert result["manifest_sha256"] == manifest.sha256
    assert result["schema_version"] == 3
    assert result["model_verified_bytes"] == 123
    assert result["model_lock_sha256"] == sha256_file(benchmark.MODEL_LOCK_PATH)
    assert set(result["implementation_sha256"]) == {
        f"src/oneload_tts/{name}" for name in benchmark.IMPLEMENTATION_FILES
    }
    assert set(result["security_control_sha256"]) == {
        f"src/oneload_tts/{name}" for name in benchmark.SECURITY_CONTROL_FILES
    }
    assert "created_at" not in result


def test_hardware_probe_uses_absolute_bounded_command_and_coarse_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs):
        observed.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="Apple M4\n", stderr="")

    monkeypatch.setattr(benchmark.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(benchmark.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)

    assert benchmark._hardware() == {"architecture": "arm64", "chip": "Apple M4"}
    assert observed == [
        (
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            {"check": True, "capture_output": True, "text": True, "timeout": 5},
        )
    ]


def test_committed_benchmark_is_sanitized_and_binds_current_sources() -> None:
    result = json.loads((benchmark.PROJECT_ROOT / "benchmarks/apple-m4.json").read_text())
    serialized = json.dumps(result)

    assert set(result["hardware"]) == {"architecture", "chip"}
    assert "created_at" not in result
    assert "operating_system" not in serialized
    assert "os_version" not in serialized
    assert "/Users/" not in serialized
    assert "/Volumes/" not in serialized
    assert result["model_lock_sha256"] == sha256_file(benchmark.MODEL_LOCK_PATH)
    assert (
        result["manifest_sha256"]
        == load_manifest(benchmark.PROJECT_ROOT / "examples/demo-manifest.json").sha256
    )
    assert result["implementation_sha256"] == {
        f"src/oneload_tts/{name}": sha256_file(benchmark.PACKAGE_ROOT / name)
        for name in benchmark.IMPLEMENTATION_FILES
    }
    assert result["security_control_sha256"] == {
        f"src/oneload_tts/{name}": sha256_file(benchmark.PACKAGE_ROOT / name)
        for name in benchmark.SECURITY_CONTROL_FILES
    }


def test_benchmark_children_use_isolated_imports_and_sanitized_python_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "defaults": {"speaker": "Aiden", "language": "English"},
                "segments": [
                    {"id": "scene", "text": "Safe text.", "output": "scene.wav", "seed": 1}
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    observed: list[tuple[list[str], dict[str, str]]] = []

    def fake_run_render(
        command: list[str], environment: dict[str, str], *, timeout_seconds: float
    ) -> tuple[float, dict]:
        observed.append((command, environment))
        assert 0 < timeout_seconds <= benchmark.BENCHMARK_TIMEOUT_SECONDS
        return 0.1, {
            "manifest_sha256": manifest.sha256,
            "model_id": "example/model",
            "model_revision": "abc123",
            "model_license": "Apache-2.0",
            "model_verified_bytes": 123,
            "model_load_seconds": 0.1,
            "generation_seconds": 0.1,
            "peak_model_memory_gb": 1.0,
            "audio_seconds": 1.0,
            "receipts": [{"id": "scene", "sha256": "same"}],
        }

    monkeypatch.setenv("PYTHONHOME", "/attacker/home")
    monkeypatch.setenv("PYTHONPATH", "/attacker/path")
    monkeypatch.setenv("PYTHONSTARTUP", "/attacker/startup.py")
    monkeypatch.setenv("BASH_ENV", "/attacker/bash-startup")
    monkeypatch.setenv("ENV", "/attacker/shell-startup")
    monkeypatch.setattr(benchmark, "_run_render", fake_run_render)
    monkeypatch.setattr(benchmark, "_hardware", lambda: {})

    run_benchmark(manifest=manifest, model_path=tmp_path / "model", trials=1)

    assert len(observed) == 2
    for command, environment in observed:
        assert command[1:5] == ["-I", "-B", "-m", "oneload_tts"]
        assert "PYTHONHOME" not in environment
        assert "PYTHONPATH" not in environment
        assert "PYTHONSTARTUP" not in environment
        assert "BASH_ENV" not in environment
        assert "ENV" not in environment


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS ACL behavior")
def test_benchmark_rejects_acl_bearing_temporary_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "defaults": {"speaker": "Aiden", "language": "English"},
                "segments": [{"id": "scene", "text": "Safe.", "output": "scene.wav", "seed": 1}],
            }
        ),
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    temporary_root = tmp_path / "benchmark-temporary"
    temporary_root.mkdir(mode=0o700)
    subprocess.run(  # noqa: S603
        [
            "/bin/chmod",
            "+a",
            (
                "everyone allow list,add_file,search,delete,add_subdirectory,delete_child,"
                "readattr,writeattr,readextattr,writeextattr,readsecurity,file_inherit,"
                "directory_inherit"
            ),
            str(temporary_root),
        ],
        check=True,
    )
    monkeypatch.setattr(
        benchmark.tempfile,
        "TemporaryDirectory",
        lambda *args, **kwargs: nullcontext(str(temporary_root)),
    )
    monkeypatch.setattr(
        benchmark,
        "_run_render",
        lambda *args, **kwargs: pytest.fail("render should not start"),
    )

    with pytest.raises(RuntimeError, match="temporary directory is not private"):
        run_benchmark(manifest=manifest, model_path=tmp_path / "model", trials=1)


def test_benchmark_rejects_child_manifest_digest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "defaults": {"speaker": "Aiden", "language": "English"},
                "segments": [
                    {"id": "scene", "text": "Safe text.", "output": "scene.wav", "seed": 1}
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)

    monkeypatch.setattr(
        benchmark,
        "_run_render",
        lambda *args, **kwargs: (
            0.1,
            {
                "manifest_sha256": "0" * 64,
                "receipts": [{"id": "scene", "sha256": "same"}],
            },
        ),
    )

    with pytest.raises(RuntimeError, match="different manifest"):
        run_benchmark(manifest=manifest, model_path=tmp_path / "model", trials=1)


def test_benchmark_subprocess_failure_does_not_expose_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_command = [
        "/private/operator/python",
        "--output-dir",
        "/private/oneload-benchmark-secret/trial-1",
    ]

    def fail(*args, **kwargs) -> None:
        raise subprocess.CalledProcessError(1, private_command, stderr="private child detail")

    monkeypatch.setattr(benchmark.subprocess, "run", fail)

    with pytest.raises(RuntimeError) as captured:
        _run_render(private_command, {}, timeout_seconds=1)

    assert str(captured.value) == "render subprocess failed"
    assert "/private/" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True


def test_benchmark_render_uses_remaining_global_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_timeout = None

    def complete(command, **kwargs):
        nonlocal observed_timeout
        observed_timeout = kwargs["timeout"]
        return subprocess.CompletedProcess(command, 0, stdout='{"status":"ok"}\n', stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", complete)

    _run_render([sys.executable], {}, timeout_seconds=2.5)

    assert observed_timeout == 2.5


def test_benchmark_render_rejects_expired_global_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("expired benchmark must not start a child"),
    )

    with pytest.raises(RuntimeError, match="deadline exceeded"):
        _run_render([sys.executable], {}, timeout_seconds=0)
