from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

import httpx

from oneload_tts._filesystem import (
    _require_protected_directory,
    commit_open_file,
    open_or_create_bound_directory,
    open_relative_directory,
    private_unlinked_file,
    read_regular_file_bounded,
)

MAX_MODEL_LOCK_BYTES = 256 * 1024
MAX_REDIRECTS = 5
OFFICIAL_HUB_ENDPOINT = "https://huggingface.co"
ALLOWED_DOWNLOAD_HOSTS = ("huggingface.co", ".hf.co", ".huggingface.co")
DOWNLOAD_TIMEOUT = httpx.Timeout(60.0, connect=15.0)
MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def _load_lock(lock_path: Path) -> dict[str, Any]:
    payload = read_regular_file_bounded(
        lock_path,
        maximum_bytes=MAX_MODEL_LOCK_BYTES,
        label="model lock",
    )
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        raise RuntimeError("model lock is not valid JSON") from None
    if not isinstance(raw, dict):
        raise RuntimeError("model lock is not valid JSON")
    model_id = raw.get("model_id")
    revision = raw.get("revision")
    required = raw.get("required_files")
    if not isinstance(model_id, str) or MODEL_ID_PATTERN.fullmatch(model_id) is None:
        raise RuntimeError("model lock contains an unsafe model identifier")
    if not isinstance(revision, str) or REVISION_PATTERN.fullmatch(revision) is None:
        raise RuntimeError("model lock contains an unsafe revision")
    if not isinstance(required, dict) or not required:
        raise RuntimeError("model lock does not list required files")
    for relative, specification in required.items():
        if not isinstance(relative, str):
            raise RuntimeError("model lock contains an unsafe file path")
        parts = Path(relative).parts
        if (
            not parts
            or Path(relative).is_absolute()
            or "\\" in relative
            or any(part in {"", ".", ".."} or "/" in part for part in parts)
        ):
            raise RuntimeError("model lock contains an unsafe file path")
        if not isinstance(specification, dict):
            raise RuntimeError("model lock contains an invalid file specification")
        expected_size = specification.get("size")
        expected_sha256 = specification.get("sha256")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size <= 0
            or not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        ):
            raise RuntimeError("model lock contains an invalid file specification")
    return raw


def _lock_files(lock_path: Path) -> set[str]:
    return set(_load_lock(lock_path)["required_files"])


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
                            _require_protected_directory(
                                child_fd,
                                failure_message=(
                                    "model download target contains an unprotected directory"
                                ),
                            )
                            if parts == (".cache",):
                                os.close(child_fd)
                                continue
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


def _sha256_descriptor_exact(file_fd: int, expected_size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < expected_size:
        block = os.pread(file_fd, min(1024 * 1024, expected_size - offset), offset)
        if not block:
            raise RuntimeError("model file does not match the locked snapshot")
        digest.update(block)
        offset += len(block)
    if os.pread(file_fd, 1, expected_size):
        raise RuntimeError("model file does not match the locked snapshot")
    return digest.hexdigest()


def _existing_file_matches(parent_fd: int, name: str, specification: dict[str, Any]) -> bool:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        expected = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    try:
        file_fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError:
        raise RuntimeError("model file does not match the locked snapshot") from None
    try:
        opened = os.fstat(file_fd)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        identity = (expected.st_dev, expected.st_ino)
        if (
            not stat.S_ISREG(expected.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or opened.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            or (opened.st_dev, opened.st_ino) != identity
            or (current.st_dev, current.st_ino) != identity
            or opened.st_size != specification["size"]
            or _sha256_descriptor_exact(file_fd, specification["size"]) != specification["sha256"]
        ):
            raise RuntimeError("model file does not match the locked snapshot")
        after = os.fstat(file_fd)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ):
            raise RuntimeError("model file changed during validation")
    finally:
        os.close(file_fd)
    return True


def _checked_redirect(current_url: str, location: str) -> str:
    redirected = urljoin(current_url, location)
    parsed = urlsplit(redirected)
    hostname = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username
        or parsed.password
        or not any(
            hostname == allowed or (allowed.startswith(".") and hostname.endswith(allowed))
            for allowed in ALLOWED_DOWNLOAD_HOSTS
        )
    ):
        raise RuntimeError("model download used an unsafe redirect")
    return redirected


def _download_locked_file(
    client: httpx.Client,
    *,
    url: str,
    file_fd: int,
    expected_size: int,
    expected_sha256: str,
) -> None:
    current_url = url
    for _ in range(MAX_REDIRECTS + 1):
        with client.stream("GET", current_url) as response:
            if response.is_redirect:
                location = response.headers.get("location")
                if location is None:
                    raise RuntimeError("model download returned an invalid redirect")
                current_url = _checked_redirect(str(response.url), location)
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                raise RuntimeError("model download failed") from None
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise RuntimeError("model download returned encoded content")
            content_length = response.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) != expected_size:
                        raise ValueError
                except ValueError:
                    raise RuntimeError("model download size does not match the lock") from None
            digest = hashlib.sha256()
            written_total = 0
            for block in response.iter_bytes(chunk_size=1024 * 1024):
                if written_total + len(block) > expected_size:
                    raise RuntimeError("model download exceeded the locked byte budget")
                digest.update(block)
                view = memoryview(block)
                while view:
                    written = os.write(file_fd, view)
                    if written <= 0:
                        raise RuntimeError("model download could not be stored")
                    view = view[written:]
                written_total += len(block)
            if written_total != expected_size or digest.hexdigest() != expected_sha256:
                raise RuntimeError("model download does not match the locked snapshot")
            os.fsync(file_fd)
            return
    raise RuntimeError("model download followed too many redirects")


def download_locked_model(target: Path, lock_path: Path) -> None:
    lock = _load_lock(lock_path)
    validate_download_target(target, lock_path)
    _, root_fd = open_or_create_bound_directory(
        target,
        failure_message="model download target is not protected",
    )
    try:
        with httpx.Client(
            follow_redirects=False,
            timeout=DOWNLOAD_TIMEOUT,
            trust_env=False,
            headers={"Accept-Encoding": "identity", "User-Agent": "OneLoad/0.1"},
        ) as client:
            for relative, specification in lock["required_files"].items():
                parts = Path(relative).parts
                parent_fd = open_relative_directory(
                    root_fd,
                    tuple(parts[:-1]),
                    failure_message="model download target changed during download",
                )
                try:
                    if _existing_file_matches(parent_fd, parts[-1], specification):
                        continue
                    file_url = (
                        f"{OFFICIAL_HUB_ENDPOINT}/"
                        f"{quote(lock['model_id'], safe='/')}/resolve/{lock['revision']}/"
                        f"{quote(relative, safe='/')}?download=true"
                    )
                    with private_unlinked_file(
                        parent_fd,
                        prefix="oneload-model",
                        suffix=".download",
                        failure_message="model download could not be stored safely",
                    ) as temporary_fd:
                        _download_locked_file(
                            client,
                            url=file_url,
                            file_fd=temporary_fd,
                            expected_size=specification["size"],
                            expected_sha256=specification["sha256"],
                        )
                        commit_open_file(
                            temporary_fd,
                            parts[-1],
                            parent_fd,
                            failure_message="model download could not be committed safely",
                        )
                finally:
                    os.close(parent_fd)
    except (httpx.HTTPError, OSError):
        raise RuntimeError("model download failed") from None
    finally:
        os.close(root_fd)

    from oneload_tts.engine import validate_model

    validate_model(target, lock)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a protected model download target.")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    try:
        if args.download:
            download_locked_model(args.target, args.lock)
        else:
            validate_download_target(args.target, args.lock)
    except (OSError, RuntimeError, ValueError, httpx.HTTPError):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
