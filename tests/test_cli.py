from __future__ import annotations

import argparse
import errno
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

import pytest

from oneload_tts import cli


def test_safe_error_message_redacts_paths_and_escapes_controls(tmp_path: Path) -> None:
    private_manifest = tmp_path / "private" / "manifest.json"
    args = argparse.Namespace(
        manifest=private_manifest,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "output",
        result=None,
    )

    message = cli._safe_error_message(
        RuntimeError(f"failed at {private_manifest.resolve()}\x1b]0;spoofed\x07"),
        args,
    )

    assert str(tmp_path) not in message
    assert "\x1b" not in message
    assert "\x07" not in message
    assert "\\u001b" in message
    assert "\\u0007" in message
    assert "<manifest>" in message


def test_safe_error_message_redacts_parent_only_oserror_path(tmp_path: Path) -> None:
    result = tmp_path / "private-results" / "benchmark.json"
    args = argparse.Namespace(
        manifest=None,
        model_path=None,
        output_dir=None,
        result=result,
    )

    message = cli._safe_error_message(
        PermissionError(errno.EACCES, "permission denied", str(result.parent)),
        args,
    )

    assert str(tmp_path) not in message
    assert "private-results" not in message
    assert "<result>" in message


def test_safe_error_message_survives_unavailable_working_directory(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = tmp_path / "private" / "manifest.json"
    args = argparse.Namespace(
        manifest=manifest,
        model_path=None,
        output_dir=None,
        result=None,
    )

    def fail_cwd(cls) -> Path:
        raise OSError("working directory is unavailable")

    monkeypatch.setattr(cli.Path, "cwd", classmethod(fail_cwd))

    message = cli._safe_error_message(RuntimeError(f"failed at {manifest}"), args)

    assert str(tmp_path) not in message
    assert "<manifest>" in message


def test_safe_error_message_fails_closed_for_stale_absolute_alias_target(tmp_path: Path) -> None:
    args = argparse.Namespace(
        manifest=None,
        model_path=None,
        output_dir=None,
        result=tmp_path / "current-alias" / "result.json",
    )
    stale_parent = "/private/tmp/PRIVATE-OLD-RESOLVED-TARGET/blocked-parent"

    message = cli._safe_error_message(
        PermissionError(errno.EACCES, "permission denied", stale_parent),
        args,
    )

    assert message == "operation failed"
    assert stale_parent not in message


@pytest.mark.parametrize(
    "private_uri",
    [
        "file:/private/var/PRIVATE-SENTINEL/secret.wav",
        "file://localhost/private/var/PRIVATE-SENTINEL/secret.wav",
        "FILE:/private/var/PRIVATE-SENTINEL/secret.wav",
        "FiLe://localhost/private/var/PRIVATE-SENTINEL/secret.wav",
        "FILE:///private/var/PRIVATE-SENTINEL/secret.wav",
        "file:/%70rivate/var/PRIVATE-SENTINEL/secret.wav",
        "file:%2Fprivate%2Fvar%2FPRIVATE-SENTINEL%2Fsecret.wav",
    ],
)
def test_safe_error_message_fails_closed_for_file_uri_variants(private_uri: str) -> None:
    args = argparse.Namespace(manifest=None, model_path=None, output_dir=None, result=None)

    message = cli._safe_error_message(RuntimeError(f"loader failed for {private_uri}"), args)

    assert message == "operation failed"
    assert "PRIVATE-SENTINEL" not in message


def test_safe_error_message_checks_the_final_percent_decoding_result() -> None:
    args = argparse.Namespace(manifest=None, model_path=None, output_dir=None, result=None)
    encoded = "file:/private/var/PRIVATE-SENTINEL/secret.wav"
    for _ in range(3):
        encoded = quote(encoded, safe="")

    message = cli._safe_error_message(RuntimeError(f"loader failed for {encoded}"), args)

    assert message == "operation failed"
    assert "PRIVATE-SENTINEL" not in message


def test_safe_error_message_fails_closed_for_excessive_percent_encoding() -> None:
    args = argparse.Namespace(manifest=None, model_path=None, output_dir=None, result=None)
    encoded = "ordinary diagnostic"
    for _ in range(cli.MAX_ERROR_DECODE_PASSES + 1):
        encoded = quote(encoded, safe="")

    assert cli._safe_error_message(RuntimeError(encoded), args) == "operation failed"


@pytest.mark.parametrize(
    "diagnostic",
    [
        "failed at //private/var/PRIVATE-SENTINEL/secret.wav",
        "failed at path:/private/var/PRIVATE-SENTINEL/secret.wav",
        "failed at path:%2F%2Fprivate%2Fvar%2FPRIVATE-SENTINEL%2Fsecret.wav",
    ],
)
def test_safe_error_message_fails_closed_for_repeated_or_colon_paths(
    diagnostic: str,
) -> None:
    args = argparse.Namespace(manifest=None, model_path=None, output_dir=None, result=None)

    message = cli._safe_error_message(RuntimeError(diagnostic), args)

    assert message == "operation failed"
    assert "PRIVATE-SENTINEL" not in message


@pytest.mark.parametrize(
    "diagnostic",
    [
        "loader failed for https://user:PRIVATE-TOKEN@example.com/model",
        "loader failed for https%3A%2F%2Fuser%3APRIVATE-TOKEN%40example.com%2Fmodel",
    ],
)
def test_safe_error_message_fails_closed_for_network_urls(diagnostic: str) -> None:
    args = argparse.Namespace(manifest=None, model_path=None, output_dir=None, result=None)

    message = cli._safe_error_message(RuntimeError(diagnostic), args)

    assert message == "operation failed"
    assert "PRIVATE-TOKEN" not in message


def test_cli_failure_does_not_print_absolute_manifest_path(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    missing = tmp_path / "private" / "missing.json"
    monkeypatch.setattr(cli, "_architecture", lambda: "arm64")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "oneload-tts",
            "render",
            "--manifest",
            str(missing),
            "--model-path",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    assert cli.main() == 1
    error = capsys.readouterr().err
    assert str(tmp_path) not in error
    assert "<manifest>" in error


def test_cli_rejects_control_speaker_without_emitting_control_bytes(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "defaults": {"speaker": "bad\x1b]0;spoofed\x07"},
                "segments": [{"id": "one", "text": "Safe.", "output": "one.wav", "seed": 1}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_architecture", lambda: "arm64")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "oneload-tts",
            "render",
            "--manifest",
            str(manifest),
            "--model-path",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    assert cli.main() == 1
    error = capsys.readouterr().err
    assert "\x1b" not in error
    assert "\x07" not in error
    assert "speaker is not supported" in error


def test_cli_enforces_offline_private_runtime_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    monkeypatch.setattr(cli, "_architecture", lambda: "arm64")
    monkeypatch.setattr(cli, "load_model_lock", lambda: {})
    monkeypatch.setattr(cli, "validate_model", lambda model_path, lock: {"verified_bytes": 1})
    monkeypatch.setattr(
        sys,
        "argv",
        ["oneload-tts", "validate", "--model-path", str(tmp_path / "model")],
    )

    assert cli.main() == 0
    capsys.readouterr()
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
