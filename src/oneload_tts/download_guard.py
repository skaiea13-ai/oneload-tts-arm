from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

from oneload_tts._filesystem import (
    _require_protected_directory,
    open_relative_directory,
    read_regular_file_bounded,
)

MAX_MODEL_LOCK_BYTES = 256 * 1024


def _lock_files(lock_path: Path) -> set[str]:
    payload = read_regular_file_bounded(
        lock_path,
        maximum_bytes=MAX_MODEL_LOCK_BYTES,
        label="model lock",
    )
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        raise RuntimeError("model lock is not valid JSON") from None
    required = raw.get("required_files") if isinstance(raw, dict) else None
    if not isinstance(required, dict) or not required:
        raise RuntimeError("model lock does not list required files")
    files = set(required)
    for relative in files:
        parts = Path(relative).parts
        if not parts or any(part in {"", ".", ".."} or "/" in part for part in parts):
            raise RuntimeError("model lock contains an unsafe file path")
    return files


def _is_allowed_entry(parts: tuple[str, ...], *, required_files: set[str]) -> bool:
    if parts and parts[0] == ".cache":
        return True
    relative = "/".join(parts)
    if relative in required_files:
        return True
    return any(candidate.startswith(relative + "/") for candidate in required_files)


def _require_owner_not_world_writable(state: os.stat_result) -> None:
    if state.st_uid != os.geteuid() or state.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeError("model download target contains an unprotected entry")


def validate_download_target(target: Path, lock_path: Path) -> None:
    required_files = _lock_files(lock_path)
    requested = target.expanduser()
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    if requested.is_absolute():
        if requested.anchor != "/":
            raise RuntimeError("model download target is not protected")
        anchor: str | Path = Path("/")
        parts = tuple(requested.parts[1:])
    else:
        anchor = "."
        parts = tuple(requested.parts)
    if not parts or any(part in {"", ".", ".."} or "/" in part for part in parts):
        raise RuntimeError("model download target is not protected")
    anchor_fd = os.open(anchor, flags)
    try:
        root_fd = open_relative_directory(
            anchor_fd,
            parts,
            failure_message="model download target is not protected",
        )
    finally:
        os.close(anchor_fd)
    _require_protected_directory(root_fd, failure_message="model download target is not protected")
    try:
        pending: list[tuple[int, tuple[str, ...]]] = [(os.dup(root_fd), ())]
        try:
            while pending:
                directory_fd, parent_parts = pending.pop()
                try:
                    entries = list(os.scandir(directory_fd))
                    for entry in entries:
                        parts = (*parent_parts, entry.name)
                        if not _is_allowed_entry(parts, required_files=required_files):
                            raise RuntimeError("model download target contains an unexpected entry")
                        state = os.stat(entry.name, dir_fd=directory_fd, follow_symlinks=False)
                        if stat.S_ISLNK(state.st_mode):
                            raise RuntimeError("model download target contains a symbolic link")
                        if parent_parts[:1] != (".cache",):
                            _require_owner_not_world_writable(state)
                        if stat.S_ISDIR(state.st_mode):
                            flags = (
                                os.O_RDONLY
                                | getattr(os, "O_DIRECTORY", 0)
                                | getattr(os, "O_NOFOLLOW", 0)
                                | getattr(os, "O_CLOEXEC", 0)
                            )
                            child_fd = os.open(entry.name, flags, dir_fd=directory_fd)
                            opened = os.fstat(child_fd)
                            current = os.stat(
                                entry.name, dir_fd=directory_fd, follow_symlinks=False
                            )
                            if (opened.st_dev, opened.st_ino) != (state.st_dev, state.st_ino) or (
                                current.st_dev,
                                current.st_ino,
                            ) != (state.st_dev, state.st_ino):
                                os.close(child_fd)
                                raise RuntimeError(
                                    "model download target changed during validation"
                                )
                            if parts[:1] != (".cache",):
                                _require_protected_directory(
                                    child_fd,
                                    failure_message=(
                                        "model download target contains an unprotected directory"
                                    ),
                                )
                            pending.append((child_fd, parts))
                        elif not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
                            raise RuntimeError("model download target contains an unsafe entry")
                finally:
                    os.close(directory_fd)
        finally:
            for directory_fd, _ in pending:
                os.close(directory_fd)
    finally:
        os.close(root_fd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a protected model download target.")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate_download_target(args.target, args.lock)
    except (OSError, RuntimeError, ValueError):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
