from __future__ import annotations

import ctypes
import errno
import os
import secrets
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_ACL_EXTENDED_ALLOW = 1
_ACL_MAX_ENTRIES = 128
_ACL_TYPE_EXTENDED = 0x00000100


def _directory_identity(state: os.stat_result) -> tuple[int, int]:
    return state.st_dev, state.st_ino


def _has_macos_extended_allow_acl(directory_fd: int) -> bool:
    """Return whether a directory grants access through a macOS allow ACL."""

    if sys.platform != "darwin":
        return False

    libc = ctypes.CDLL(None, use_errno=True)
    acl_get_fd_np = getattr(libc, "acl_get_fd_np", None)
    acl_get_entry = getattr(libc, "acl_get_entry", None)
    acl_get_tag_type = getattr(libc, "acl_get_tag_type", None)
    acl_free = getattr(libc, "acl_free", None)
    if any(
        function is None for function in (acl_get_fd_np, acl_get_entry, acl_get_tag_type, acl_free)
    ):
        raise OSError(errno.ENOSYS, "macOS ACL inspection is unavailable")

    acl_get_fd_np.argtypes = [ctypes.c_int, ctypes.c_int]
    acl_get_fd_np.restype = ctypes.c_void_p
    acl_get_entry.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    acl_get_entry.restype = ctypes.c_int
    acl_get_tag_type.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    acl_get_tag_type.restype = ctypes.c_int
    acl_free.argtypes = [ctypes.c_void_p]
    acl_free.restype = ctypes.c_int

    ctypes.set_errno(0)
    acl = acl_get_fd_np(directory_fd, _ACL_TYPE_EXTENDED)
    if not acl:
        error = ctypes.get_errno() or errno.EIO
        if error == errno.ENOENT:
            return False
        raise OSError(error, os.strerror(error))
    try:
        for index in range(_ACL_MAX_ENTRIES):
            entry = ctypes.c_void_p()
            ctypes.set_errno(0)
            if acl_get_entry(acl, index, ctypes.byref(entry)) != 0:
                error = ctypes.get_errno()
                if error == errno.EINVAL:
                    return False
                raise OSError(error or errno.EIO, os.strerror(error or errno.EIO))
            tag = ctypes.c_int()
            if acl_get_tag_type(entry, ctypes.byref(tag)) != 0:
                error = ctypes.get_errno() or errno.EIO
                raise OSError(error, os.strerror(error))
            if tag.value == _ACL_EXTENDED_ALLOW:
                return True
        return False
    finally:
        acl_free(acl)


def _require_protected_directory(directory_fd: int, *, failure_message: str) -> None:
    """Require a parent whose entries cannot be replaced by another OS user."""

    state = os.fstat(directory_fd)
    writable_by_others = state.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    sticky = state.st_mode & stat.S_ISVTX
    try:
        has_allow_acl = _has_macos_extended_allow_acl(directory_fd)
    except OSError:
        raise RuntimeError(failure_message) from None
    if (
        not stat.S_ISDIR(state.st_mode)
        or state.st_uid not in {0, os.geteuid()}
        or (writable_by_others and not sticky)
        or has_allow_acl
    ):
        raise RuntimeError(failure_message)


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return _directory_identity(left) == _directory_identity(right)


def open_bound_directory(path: Path, *, failure_message: str) -> tuple[Path, int]:
    """Open the exact directory object resolved at the start of the operation."""

    try:
        resolved = path.expanduser().resolve(strict=True)
        expected = os.stat(resolved, follow_symlinks=False)
        if not stat.S_ISDIR(expected.st_mode):
            raise OSError
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        directory_fd = os.open(resolved, flags)
        opened = os.fstat(directory_fd)
        current = os.stat(resolved, follow_symlinks=False)
    except (OSError, RuntimeError):
        if "directory_fd" in locals():
            os.close(directory_fd)
        raise RuntimeError(failure_message) from None

    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or _directory_identity(opened) != _directory_identity(expected)
        or _directory_identity(current) != _directory_identity(expected)
    ):
        os.close(directory_fd)
        raise RuntimeError(failure_message)
    return resolved, directory_fd


def open_or_create_bound_directory(path: Path, *, failure_message: str) -> tuple[Path, int]:
    """Create and bind a directory without re-resolving mutable path components."""

    try:
        requested = path.expanduser()
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        if requested.is_absolute():
            if requested.anchor != "/":
                raise RuntimeError(failure_message)
            anchor: str | Path = Path("/")
            parts = tuple(requested.parts[1:])
        else:
            anchor = "."
            parts = tuple(requested.parts)
        if any(part in {"", ".", ".."} or "/" in part for part in parts):
            raise RuntimeError(failure_message)

        anchor_fd = os.open(anchor, flags)
        directory_fd = open_relative_directory(
            anchor_fd,
            parts,
            failure_message=failure_message,
        )
        _require_protected_directory(directory_fd, failure_message=failure_message)
    except (OSError, RuntimeError):
        if "directory_fd" in locals():
            os.close(directory_fd)
        raise RuntimeError(failure_message) from None
    finally:
        if "anchor_fd" in locals():
            os.close(anchor_fd)

    return requested, directory_fd


def open_relative_directory(
    root_fd: int,
    parts: tuple[str, ...],
    *,
    failure_message: str,
) -> int:
    """Open or create a relative directory chain without following links."""

    current_fd = os.dup(root_fd)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        for part in parts:
            if part in {"", ".", ".."} or "/" in part:
                raise RuntimeError(failure_message)
            _require_protected_directory(current_fd, failure_message=failure_message)
            created: os.stat_result | None = None
            try:
                os.mkdir(part, mode=0o700, dir_fd=current_fd)
            except FileExistsError:
                pass
            else:
                created = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(created.st_mode)
                    or created.st_uid != os.geteuid()
                    or created.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
                ):
                    raise RuntimeError(failure_message)
            expected = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            next_fd = os.open(part, flags, dir_fd=current_fd)
            opened = os.fstat(next_fd)
            current = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(expected.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or not stat.S_ISDIR(current.st_mode)
                or not _same_identity(expected, opened)
                or not _same_identity(opened, current)
                or (created is not None and not _same_identity(created, opened))
            ):
                os.close(next_fd)
                raise RuntimeError(failure_message)
            os.close(current_fd)
            current_fd = next_fd
    except (OSError, RuntimeError):
        os.close(current_fd)
        raise RuntimeError(failure_message) from None
    return current_fd


def _clone_file_from_descriptor(source_fd: int, parent_fd: int, name: str) -> bool:
    if sys.platform != "darwin":
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    clone = getattr(libc, "fclonefileat", None)
    if clone is None:
        return False
    clone.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    clone.restype = ctypes.c_int
    if clone(source_fd, parent_fd, os.fsencode(name), 0) == 0:
        return True
    error = ctypes.get_errno()
    if error in {errno.ENOTSUP, errno.EXDEV, errno.EINVAL, errno.ENOSYS}:
        return False
    raise OSError(error, os.strerror(error), name)


def copy_regular_file_from_descriptor(
    source_fd: int,
    parent_fd: int,
    name: str,
    *,
    expected_size: int,
    failure_message: str,
) -> int:
    """Freeze an open regular file into a private directory and return its read descriptor."""

    if name in {"", ".", ".."} or "/" in name:
        raise RuntimeError(failure_message)
    target_fd: int | None = None
    created = False
    try:
        _require_protected_directory(parent_fd, failure_message=failure_message)
        source_state = os.fstat(source_fd)
        if not stat.S_ISREG(source_state.st_mode) or source_state.st_size != expected_size:
            raise OSError
        created = _clone_file_from_descriptor(source_fd, parent_fd, name)
        if not created:
            write_flags = (
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            target_fd = os.open(name, write_flags, 0o600, dir_fd=parent_fd)
            created = True
            offset = 0
            remaining = expected_size
            while remaining:
                block = os.pread(source_fd, min(1024 * 1024, remaining), offset)
                if not block:
                    raise OSError
                view = memoryview(block)
                while view:
                    written = os.write(target_fd, view)
                    if written <= 0:
                        raise OSError
                    view = view[written:]
                offset += len(block)
                remaining -= len(block)
            if os.pread(source_fd, 1, expected_size):
                raise OSError
            os.fsync(target_fd)
            os.close(target_fd)
            target_fd = None

        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        target_fd = os.open(name, read_flags, dir_fd=parent_fd)
        target_state = os.fstat(target_fd)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(target_state.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or not _same_identity(target_state, current)
            or target_state.st_ino == source_state.st_ino
            or target_state.st_size != expected_size
        ):
            raise OSError
        os.fchmod(target_fd, 0o400)
        return target_fd
    except (OSError, RuntimeError):
        if target_fd is not None:
            os.close(target_fd)
        if created:
            try:
                os.unlink(name, dir_fd=parent_fd)
            except OSError:
                pass
        raise RuntimeError(failure_message) from None


@contextmanager
def private_staging_directory(
    parent_fd: int,
    *,
    prefix: str,
    failure_message: str,
) -> Iterator[int]:
    """Create a private same-filesystem staging directory and keep it descriptor-bound."""

    staging_name: str | None = None
    staging_fd: int | None = None
    created_identity: tuple[int, int] | None = None
    opened_identity: tuple[int, int] | None = None
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        _require_protected_directory(parent_fd, failure_message=failure_message)
        for _ in range(32):
            candidate = f".{prefix}.{secrets.token_hex(12)}.stage"
            try:
                os.mkdir(candidate, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                continue
            staging_name = candidate
            created = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
            created_identity = _directory_identity(created)
            if (
                not stat.S_ISDIR(created.st_mode)
                or created.st_uid != os.geteuid()
                or created.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
            ):
                raise RuntimeError(failure_message)
            break
        else:
            raise RuntimeError(failure_message)

        staging_fd = os.open(staging_name, flags, dir_fd=parent_fd)
        opened = os.fstat(staging_fd)
        current = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
        opened_identity = _directory_identity(opened)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or opened_identity != created_identity
            or opened_identity != _directory_identity(current)
        ):
            raise RuntimeError(failure_message)
    except (OSError, RuntimeError):
        if staging_fd is not None:
            os.close(staging_fd)
        if staging_name is not None and created_identity is not None:
            try:
                current = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
                if (
                    stat.S_ISDIR(current.st_mode)
                    and _directory_identity(current) == created_identity
                ):
                    os.rmdir(staging_name, dir_fd=parent_fd)
            except OSError:
                pass
        raise RuntimeError(failure_message) from None

    body_failed = False
    try:
        yield staging_fd
    except BaseException:
        body_failed = True
        raise
    finally:
        os.close(staging_fd)
        cleanup_failed = False
        try:
            current = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(current.st_mode) or _directory_identity(current) != opened_identity:
                cleanup_failed = True
            else:
                os.rmdir(staging_name, dir_fd=parent_fd)
        except OSError:
            cleanup_failed = True
        if cleanup_failed and not body_failed:
            raise RuntimeError(failure_message) from None


@contextmanager
def private_temporary_file(
    parent_fd: int,
    *,
    prefix: str,
    suffix: str,
    failure_message: str,
) -> Iterator[tuple[str, int]]:
    """Create an exclusive private file and retain its descriptor through commit."""

    name: str | None = None
    file_fd: int | None = None
    identity: tuple[int, int] | None = None
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        _require_protected_directory(parent_fd, failure_message=failure_message)
        for _ in range(32):
            candidate = f".{prefix}.{secrets.token_hex(12)}{suffix}"
            try:
                file_fd = os.open(candidate, flags, 0o600, dir_fd=parent_fd)
            except FileExistsError:
                continue
            name = candidate
            break
        else:
            raise RuntimeError(failure_message)

        opened = os.fstat(file_fd)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        identity = _directory_identity(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or opened.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
            or not _same_identity(opened, current)
        ):
            raise RuntimeError(failure_message)
    except (OSError, RuntimeError):
        if file_fd is not None:
            os.close(file_fd)
        if name is not None and identity is not None:
            try:
                current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if stat.S_ISREG(current.st_mode) and _directory_identity(current) == identity:
                    os.unlink(name, dir_fd=parent_fd)
            except OSError:
                pass
        raise RuntimeError(failure_message) from None

    body_failed = False
    try:
        yield name, file_fd
    except BaseException:
        body_failed = True
        raise
    finally:
        cleanup_failed = False
        try:
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError:
            cleanup_failed = True
        else:
            if stat.S_ISREG(current.st_mode) and _directory_identity(current) == identity:
                try:
                    os.unlink(name, dir_fd=parent_fd)
                except OSError:
                    cleanup_failed = True
        os.close(file_fd)
        if cleanup_failed and not body_failed:
            raise RuntimeError(failure_message) from None


def commit_open_file(
    file_fd: int,
    temporary_name: str,
    destination_name: str,
    parent_fd: int,
    *,
    failure_message: str,
) -> None:
    """Rename an open temporary file and verify the committed name still binds it."""

    if destination_name in {"", ".", ".."} or "/" in destination_name:
        raise RuntimeError(failure_message)
    try:
        opened = os.fstat(file_fd)
        current = os.stat(temporary_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or opened.st_nlink != 1
            or not _same_identity(opened, current)
        ):
            raise RuntimeError(failure_message)
        os.replace(
            temporary_name,
            destination_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        committed = os.stat(destination_name, dir_fd=parent_fd, follow_symlinks=False)
        after = os.fstat(file_fd)
        if (
            not stat.S_ISREG(committed.st_mode)
            or not _same_identity(opened, committed)
            or not _same_identity(opened, after)
            or committed.st_nlink != 1
        ):
            raise RuntimeError(failure_message)
        os.fsync(parent_fd)
    except OSError:
        raise RuntimeError(failure_message) from None


def read_regular_file_bounded(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    """Read one stable regular file without blocking on FIFOs or device nodes."""

    file_fd: int | None = None
    directory_fd: int | None = None
    requested = path
    try:
        requested = path.expanduser()
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        if requested.is_absolute():
            if requested.anchor != "/":
                raise ValueError(f"{label} path must not use symbolic links")
            anchor: str | Path = Path("/")
            parts = tuple(requested.parts[1:])
        else:
            anchor = "."
            parts = tuple(requested.parts)
        if not parts or any(part in {"", ".", ".."} or "/" in part for part in parts):
            raise ValueError(f"{label} path must name a stable regular file")

        directory_fd = os.open(anchor, directory_flags)
        for part in parts[:-1]:
            expected_directory = os.stat(part, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(expected_directory.st_mode) or not stat.S_ISDIR(
                expected_directory.st_mode
            ):
                raise ValueError(f"{label} path must not use symbolic links")
            try:
                next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
                opened_directory = os.fstat(next_fd)
                current_directory = os.stat(part, dir_fd=directory_fd, follow_symlinks=False)
            except OSError:
                if "next_fd" in locals():
                    os.close(next_fd)
                    del next_fd
                raise ValueError(f"{label} changed while reading") from None
            if (
                not stat.S_ISDIR(opened_directory.st_mode)
                or not stat.S_ISDIR(current_directory.st_mode)
                or not _same_identity(expected_directory, opened_directory)
                or not _same_identity(opened_directory, current_directory)
            ):
                os.close(next_fd)
                del next_fd
                raise ValueError(f"{label} changed while reading")
            os.close(directory_fd)
            directory_fd = next_fd
            del next_fd

        leaf = parts[-1]
        expected = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(expected.st_mode):
            raise ValueError(f"{label} path must not use symbolic links")
        if not stat.S_ISREG(expected.st_mode):
            raise ValueError(f"{label} must be a regular file")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            file_fd = os.open(leaf, flags, dir_fd=directory_fd)
            opened = os.fstat(file_fd)
            current = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        except OSError:
            raise ValueError(f"{label} changed while reading") from None
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or not _same_identity(expected, opened)
            or not _same_identity(opened, current)
        ):
            raise ValueError(f"{label} changed while reading")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            block = os.read(file_fd, min(64 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(file_fd)
        if (
            not _same_identity(opened, after)
            or opened.st_size != after.st_size
            or opened.st_mtime_ns != after.st_mtime_ns
            or opened.st_ctime_ns != after.st_ctime_ns
        ):
            raise ValueError(f"{label} changed while reading")
        return b"".join(chunks)
    except FileNotFoundError:
        raise FileNotFoundError(
            errno.ENOENT,
            f"{label} does not exist",
            os.fspath(requested),
        ) from None
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)
