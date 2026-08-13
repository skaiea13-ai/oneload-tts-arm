from __future__ import annotations

import json
from pathlib import Path

import pytest

from oneload_tts.download_guard import validate_download_target


def _lock(path: Path) -> Path:
    path.write_text(
        json.dumps({"required_files": {"weight.bin": {"size": 1, "sha256": "0" * 64}}}),
        encoding="utf-8",
    )
    return path


def test_download_guard_accepts_private_expected_partial_target(tmp_path: Path) -> None:
    target = tmp_path / "model"
    target.mkdir(mode=0o700)
    (target / "weight.bin").write_bytes(b"partial")

    validate_download_target(target, _lock(tmp_path / "lock.json"))


def test_download_guard_rejects_symlinked_target(tmp_path: Path) -> None:
    real_target = tmp_path / "real-model"
    real_target.mkdir(mode=0o700)
    linked_target = tmp_path / "linked-model"
    linked_target.symlink_to(real_target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="not protected"):
        validate_download_target(linked_target, _lock(tmp_path / "lock.json"))


def test_download_guard_rejects_unexpected_entry(tmp_path: Path) -> None:
    target = tmp_path / "model"
    target.mkdir(mode=0o700)
    (target / "unexpected.bin").write_bytes(b"attacker")

    with pytest.raises(RuntimeError, match="unexpected entry"):
        validate_download_target(target, _lock(tmp_path / "lock.json"))
