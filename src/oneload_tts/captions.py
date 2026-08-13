from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

from oneload_tts._filesystem import (
    commit_open_file,
    open_or_create_bound_directory,
    private_unlinked_file,
    read_regular_file_bounded,
)

TIMESTAMP = re.compile(
    r"^(?P<start>[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}) --> "
    r"(?P<end>[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3})$"
)
MAX_CAPTION_INPUT_BYTES = 256 * 1024
MAX_CAPTION_SEGMENTS = 32
MAX_SEGMENT_SECONDS = 10 * 60
MAX_TOTAL_SECONDS = 30 * 60


def _read_bounded(path: Path, *, label: str) -> bytes:
    payload = read_regular_file_bounded(
        path,
        maximum_bytes=MAX_CAPTION_INPUT_BYTES,
        label=label,
    )
    if len(payload) > MAX_CAPTION_INPUT_BYTES:
        raise ValueError(f"{label} exceeds 256 KiB")
    return payload


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"narration receipt contains a non-finite number: {value}")


def _seconds(value: str) -> float:
    hours, minutes, remainder = value.split(":")
    seconds, milliseconds = remainder.split(",")
    if int(minutes) >= 60 or int(seconds) >= 60:
        raise ValueError("caption timestamp is invalid")
    return int(hours) * 3_600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1_000


def _cue_times(captions: str) -> list[tuple[float, float]]:
    normalized = captions.replace("\r\n", "\n")
    if "\r" in normalized or not normalized or normalized.startswith("\n"):
        raise ValueError("captions must use canonical SRT blocks")
    if normalized.endswith("\n"):
        normalized = normalized[:-1]
    if not normalized or normalized.endswith("\n"):
        raise ValueError("captions must use canonical SRT blocks")

    blocks = normalized.split("\n\n")
    if not 1 <= len(blocks) <= MAX_CAPTION_SEGMENTS:
        raise ValueError("captions must contain between 1 and 32 cues")
    times: list[tuple[float, float]] = []
    for expected_index, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        if (
            len(lines) < 3
            or lines[0] != str(expected_index)
            or any(not line.strip() for line in lines[2:])
        ):
            raise ValueError("captions must use canonical SRT blocks")
        match = TIMESTAMP.fullmatch(lines[1])
        if match is None:
            raise ValueError("captions must use canonical SRT timestamps")
        start = _seconds(match["start"])
        end = _seconds(match["end"])
        if start < 0.0 or end <= start:
            raise ValueError("caption cue has an invalid interval")
        times.append((start, end))
    return times


def verify_caption_timing(
    receipt_path: Path, captions_path: Path, *, gap_seconds: float = 1.0
) -> bytes:
    if (
        isinstance(gap_seconds, bool)
        or not isinstance(gap_seconds, int | float)
        or gap_seconds < 0.0
        or gap_seconds > MAX_TOTAL_SECONDS
        or (isinstance(gap_seconds, float) and not math.isfinite(gap_seconds))
    ):
        raise ValueError("caption gap must be a finite non-negative number")
    bounded_gap_seconds = float(gap_seconds)
    receipt = json.loads(
        _read_bounded(receipt_path, label="narration receipt").decode("utf-8"),
        parse_constant=_reject_nonfinite_json,
    )
    if not isinstance(receipt, dict):
        raise ValueError("narration receipt must be an object")
    receipts = receipt.get("receipts")
    if not isinstance(receipts, list) or not 1 <= len(receipts) <= MAX_CAPTION_SEGMENTS:
        raise ValueError("narration receipt must contain between 1 and 32 segments")
    if any(not isinstance(item, dict) for item in receipts):
        raise ValueError("narration receipt segments must be objects")
    caption_payload = _read_bounded(captions_path, label="captions")
    cue_times = _cue_times(caption_payload.decode("utf-8"))
    if len(cue_times) != len(receipts):
        raise ValueError("caption cue count does not match narration receipt")

    expected_start = 0.0
    for index, (cue_start, cue_end) in enumerate(cue_times):
        duration = receipts[index].get("duration_seconds")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, int | float)
            or duration <= 0.0
            or duration > MAX_SEGMENT_SECONDS
            or (isinstance(duration, float) and not math.isfinite(duration))
        ):
            raise ValueError("narration receipt has an invalid duration")
        duration_seconds = float(duration)
        expected_end = expected_start + duration_seconds
        if expected_end > MAX_TOTAL_SECONDS:
            raise ValueError("narration receipt duration exceeds 1,800 seconds")
        if abs(cue_start - expected_start) > 0.02 or abs(cue_end - expected_end) > 0.02:
            raise ValueError(f"caption cue {index + 1} timing does not match narration")
        expected_start = expected_end + bounded_gap_seconds
    return caption_payload


def _write_caption_snapshot(destination: Path, payload: bytes) -> None:
    if destination.name in {"", ".", ".."}:
        raise RuntimeError("could not write verified caption snapshot")
    _, parent_fd = open_or_create_bound_directory(
        destination.parent,
        failure_message="could not write verified caption snapshot",
    )
    try:
        with private_unlinked_file(
            parent_fd,
            prefix="oneload-captions",
            suffix=".srt",
            failure_message="could not write verified caption snapshot",
        ) as file_fd:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(file_fd, remaining)
                if written <= 0:
                    raise OSError
                remaining = remaining[written:]
            os.fchmod(file_fd, 0o400)
            os.fsync(file_fd)
            commit_open_file(
                file_fd,
                destination.name,
                parent_fd,
                failure_message="could not write verified caption snapshot",
            )
    finally:
        os.close(parent_fd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SRT timing against a narration receipt.")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--gap-seconds", type=float, default=1.0)
    args = parser.parse_args()
    try:
        caption_payload = verify_caption_timing(
            args.receipt,
            args.captions,
            gap_seconds=args.gap_seconds,
        )
        if args.snapshot is not None:
            _write_caption_snapshot(args.snapshot, caption_payload)
    except OSError:
        print("caption verification failed: could not read caption inputs")
        return 1
    except json.JSONDecodeError:
        print("caption verification failed: narration receipt is not valid JSON")
        return 1
    except (OverflowError, RecursionError, RuntimeError, ValueError) as error:
        print(f"caption verification failed: {error}")
        return 1
    print("Caption timing matches the narration receipt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
