from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import oneload_tts._filesystem as filesystem
import oneload_tts.captions as captions_module
from oneload_tts.captions import MAX_CAPTION_INPUT_BYTES, main, verify_caption_timing


def _write_fixture(tmp_path: Path, *, second_end: str = "00:00:05,500") -> tuple[Path, Path]:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "receipts": [
                    {"id": "one", "duration_seconds": 2.5},
                    {"id": "two", "duration_seconds": 2.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    captions = tmp_path / "captions.srt"
    captions.write_text(
        f"1\n00:00:00,000 --> 00:00:02,500\nFirst.\n\n2\n00:00:03,500 --> {second_end}\nSecond.\n",
        encoding="utf-8",
    )
    return receipt, captions


def test_verify_caption_timing_accepts_receipt_aligned_cues(tmp_path: Path) -> None:
    receipt, captions = _write_fixture(tmp_path)

    assert verify_caption_timing(receipt, captions) == captions.read_bytes()


def test_verify_caption_timing_rejects_drift(tmp_path: Path) -> None:
    receipt, captions = _write_fixture(tmp_path, second_end="00:00:04,900")

    with pytest.raises(ValueError, match="cue 2 timing"):
        verify_caption_timing(receipt, captions)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"receipts": []},
        {"receipts": [None]},
        {"receipts": [{"duration_seconds": 1.0}] * 33},
    ],
)
def test_verify_caption_timing_rejects_wrong_receipt_shapes(
    tmp_path: Path, payload: object
) -> None:
    receipt, captions = _write_fixture(tmp_path)
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        verify_caption_timing(receipt, captions)


@pytest.mark.parametrize("duration", [-1.0, 0.0, float("nan"), float("inf")])
def test_verify_caption_timing_rejects_invalid_durations(tmp_path: Path, duration: float) -> None:
    receipt, captions = _write_fixture(tmp_path)
    receipt.write_text(
        json.dumps(
            {
                "receipts": [
                    {"duration_seconds": duration},
                    {"duration_seconds": 2.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        verify_caption_timing(receipt, captions)


@pytest.mark.parametrize("gap", [-1.0, float("nan"), float("inf")])
def test_verify_caption_timing_rejects_invalid_gap(tmp_path: Path, gap: float) -> None:
    receipt, captions = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="caption gap"):
        verify_caption_timing(receipt, captions, gap_seconds=gap)


def test_verify_caption_timing_rejects_reversed_cue(tmp_path: Path) -> None:
    receipt, captions = _write_fixture(tmp_path)
    captions.write_text(
        "1\n00:00:02,500 --> 00:00:00,000\nFirst.\n\n2\n00:00:03,500 --> 00:00:05,500\nSecond.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid interval"):
        verify_caption_timing(receipt, captions)


@pytest.mark.parametrize(
    "second_timestamp",
    [
        "00:00:03.500 --> 00:00:05.500",
        "00:00:03,500 --> 00:00:05,500 align:start",
        "00:00:03,500-->00:00:05,500",
    ],
)
def test_verify_caption_timing_rejects_noncanonical_ffmpeg_cues(
    tmp_path: Path, second_timestamp: str
) -> None:
    receipt, captions = _write_fixture(tmp_path)
    captions.write_text(
        f"1\n00:00:00,000 --> 00:00:02,500\nFirst.\n\n2\n{second_timestamp}\nInjected.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical SRT"):
        verify_caption_timing(receipt, captions)


@pytest.mark.parametrize(
    "timestamp",
    [
        "٠٠:٠٠:٠٣,٥٠٠ --> ٠٠:٠٠:٠٥,٥٠٠",
        "００:００:０３,５００ --> ００:００:０５,５００",
    ],
)
def test_verify_caption_timing_rejects_non_ascii_decimal_digits(
    tmp_path: Path, timestamp: str
) -> None:
    receipt, captions = _write_fixture(tmp_path)
    captions.write_text(
        f"1\n00:00:00,000 --> 00:00:02,500\nFirst.\n\n2\n{timestamp}\nSecond.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical SRT"):
        verify_caption_timing(receipt, captions)


@pytest.mark.parametrize("linked_input", ["receipt", "captions"])
def test_verify_caption_timing_rejects_final_symlink(tmp_path: Path, linked_input: str) -> None:
    receipt, captions = _write_fixture(tmp_path)
    source = receipt if linked_input == "receipt" else captions
    link = tmp_path / f"linked-{source.name}"
    link.symlink_to(source)

    with pytest.raises(ValueError, match="symbolic links"):
        verify_caption_timing(
            link if linked_input == "receipt" else receipt,
            link if linked_input == "captions" else captions,
        )


def test_verify_caption_timing_rejects_intermediate_symlink(tmp_path: Path) -> None:
    source_directory = tmp_path / "source"
    source_directory.mkdir()
    receipt, captions = _write_fixture(source_directory)
    link = tmp_path / "linked-directory"
    link.symlink_to(source_directory, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic links"):
        verify_caption_timing(receipt, link / captions.name)


def test_verify_caption_timing_rejects_unaccounted_srt_blocks(tmp_path: Path) -> None:
    receipt, captions = _write_fixture(tmp_path)
    captions.write_text(
        captions.read_text(encoding="utf-8") + "\n3\n00:00:06,500 --> 00:00:07,500\nUnexpected.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        verify_caption_timing(receipt, captions)


@pytest.mark.parametrize("oversized_input", ["receipt", "captions"])
def test_verify_caption_timing_rejects_oversized_inputs(
    tmp_path: Path, oversized_input: str
) -> None:
    receipt, captions = _write_fixture(tmp_path)
    oversized_path = receipt if oversized_input == "receipt" else captions
    oversized_path.write_bytes(b"x" * (MAX_CAPTION_INPUT_BYTES + 1))

    with pytest.raises(ValueError, match="exceeds 256 KiB"):
        verify_caption_timing(receipt, captions)


def test_verify_caption_timing_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    receipt, _ = _write_fixture(tmp_path)
    captions = tmp_path / "captions-fifo.srt"
    os.mkfifo(captions)
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from oneload_tts.captions import verify_caption_timing; import sys; "
                "\ntry: verify_caption_timing(Path(sys.argv[1]), Path(sys.argv[2]))"
                "\nexcept ValueError as error: print(error); raise SystemExit(0)"
                "\nraise SystemExit(2)"
            ),
            str(receipt),
            str(captions),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert completed.returncode == 0
    assert "regular file" in completed.stdout


def test_caption_cli_handles_malformed_receipt_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt, captions = _write_fixture(tmp_path)
    receipt.write_text('{"receipts":[null]}', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["oneload-captions", "--receipt", str(receipt), "--captions", str(captions)],
    )

    assert main() == 1
    output = capsys.readouterr()
    assert "caption verification failed" in output.out
    assert "Traceback" not in output.out
    assert output.err == ""


def test_caption_cli_rejects_huge_integer_duration_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt, captions = _write_fixture(tmp_path)
    receipt.write_text(
        '{"receipts":[{"duration_seconds":' + "9" * 4_000 + '},{"duration_seconds":2.0}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["oneload-captions", "--receipt", str(receipt), "--captions", str(captions)],
    )

    assert main() == 1
    output = capsys.readouterr()
    assert "caption verification failed" in output.out
    assert "invalid duration" in output.out
    assert "Traceback" not in output.out
    assert output.err == ""


def test_caption_cli_writes_exact_private_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt, captions = _write_fixture(tmp_path)
    expected = captions.read_bytes()
    snapshot = tmp_path / "verified" / "captions.srt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "oneload-captions",
            "--receipt",
            str(receipt),
            "--captions",
            str(captions),
            "--snapshot",
            str(snapshot),
        ],
    )

    assert main() == 0
    assert snapshot.read_bytes() == expected
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o400
    assert "matches" in capsys.readouterr().out


def test_caption_snapshot_publishes_from_unlinked_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "captions.srt"
    payload = b"1\n00:00:00,000 --> 00:00:01,000\nSafe.\n"
    original_clone = filesystem._clone_file_from_descriptor
    descriptor_was_unlinked = False

    def checking_clone(source_fd: int, parent_fd: int, name: str) -> bool:
        nonlocal descriptor_was_unlinked
        descriptor_was_unlinked = os.fstat(source_fd).st_nlink == 0
        assert not tuple(tmp_path.glob(".oneload-captions.*.srt"))
        return original_clone(source_fd, parent_fd, name)

    monkeypatch.setattr(filesystem, "_clone_file_from_descriptor", checking_clone)

    captions_module._write_caption_snapshot(destination, payload)

    assert descriptor_was_unlinked
    assert destination.read_bytes() == payload


def test_caption_snapshot_refuses_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "captions.srt"
    destination.write_bytes(b"original")

    with pytest.raises(RuntimeError, match="could not write verified caption snapshot"):
        captions_module._write_caption_snapshot(destination, b"replacement")

    assert destination.read_bytes() == b"original"


def test_caption_cli_snapshot_does_not_reopen_changed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, captions = _write_fixture(tmp_path)
    expected = captions.read_bytes()
    snapshot = tmp_path / "verified" / "captions.srt"
    write_snapshot = captions_module._write_caption_snapshot

    def mutate_then_snapshot(destination: Path, payload: bytes) -> None:
        captions.write_text("attacker replacement", encoding="utf-8")
        write_snapshot(destination, payload)

    monkeypatch.setattr(captions_module, "_write_caption_snapshot", mutate_then_snapshot)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "oneload-captions",
            "--receipt",
            str(receipt),
            "--captions",
            str(captions),
            "--snapshot",
            str(snapshot),
        ],
    )

    assert main() == 0
    assert captions.read_text(encoding="utf-8") == "attacker replacement"
    assert snapshot.read_bytes() == expected


def test_verify_caption_timing_rejects_huge_integer_gap(tmp_path: Path) -> None:
    receipt, captions = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="caption gap"):
        verify_caption_timing(receipt, captions, gap_seconds=10**4_000)
