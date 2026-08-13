from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from oneload_tts.manifest import MAX_MANIFEST_BYTES, load_manifest


def _write_manifest(path: Path, *, segments: list[dict] | None = None) -> Path:
    payload = {
        "schema_version": 1,
        "defaults": {"speaker": "Aiden", "language": "English"},
        "segments": segments
        or [
            {
                "id": "first-scene",
                "text": "  Calm   local narration. ",
                "output": "first.wav",
                "seed": 7,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_manifest_normalizes_and_hashes(tmp_path: Path) -> None:
    manifest = load_manifest(_write_manifest(tmp_path / "manifest.json"))

    assert manifest.settings.speaker == "Aiden"
    assert manifest.segments[0].text == "Calm local narration."
    assert manifest.segments[0].output.as_posix() == "first.wav"
    assert len(manifest.sha256) == 64


@pytest.mark.parametrize(
    "output",
    [
        "../escape.wav",
        "/tmp/escape.wav",
        "not-a-wave.mp3",
        "bad\u202e.wav",
        "caf\u00e9.wav",
        "nested/\u03b1.wav",
        "space name.wav",
    ],
)
def test_load_manifest_rejects_unsafe_outputs(tmp_path: Path, output: str) -> None:
    path = _write_manifest(
        tmp_path / "manifest.json",
        segments=[{"id": "scene", "text": "Safe text.", "output": output, "seed": 1}],
    )

    with pytest.raises(ValueError):
        load_manifest(path)


def test_load_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "manifest.json",
        segments=[
            {"id": "scene", "text": "First.", "output": "one.wav", "seed": 1},
            {"id": "scene", "text": "Second.", "output": "two.wav", "seed": 2},
        ],
    )

    with pytest.raises(ValueError, match="duplicate segment id"):
        load_manifest(path)


def test_load_manifest_rejects_long_text(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "manifest.json",
        segments=[{"id": "scene", "text": "x" * 2_001, "output": "one.wav", "seed": 1}],
    )

    with pytest.raises(ValueError, match="2,000"):
        load_manifest(path)


def test_load_manifest_rejects_oversized_file_before_json_decode(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(b" " * (MAX_MANIFEST_BYTES + 1))

    with pytest.raises(ValueError, match="256 KiB"):
        load_manifest(path)


def test_load_manifest_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "manifest.json"
    os.mkfifo(fifo)
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; from oneload_tts.manifest import load_manifest; "
                "\ntry: load_manifest(Path(__import__('sys').argv[1]))"
                "\nexcept ValueError as error: print(error); raise SystemExit(0)"
                "\nraise SystemExit(2)"
            ),
            str(fifo),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert completed.returncode == 0
    assert "regular file" in completed.stdout


def test_load_manifest_rejects_final_symlink(tmp_path: Path) -> None:
    target = _write_manifest(tmp_path / "target.json")
    link = tmp_path / "manifest.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic links"):
        load_manifest(link)


def test_load_manifest_rejects_intermediate_symlink(tmp_path: Path) -> None:
    target_directory = tmp_path / "target"
    target_directory.mkdir()
    _write_manifest(target_directory / "manifest.json")
    link = tmp_path / "linked-directory"
    link.symlink_to(target_directory, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic links"):
        load_manifest(link / "manifest.json")


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("max_tokens", 4_097, "4,096"),
        ("top_k", 1_001, "1,000"),
        ("repetition_penalty", float("inf"), "finite"),
        ("temperature", True, "number"),
    ],
)
def test_load_manifest_rejects_unbounded_generation_settings(
    tmp_path: Path, name: str, value: object, message: str
) -> None:
    path = _write_manifest(tmp_path / "manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["defaults"][name] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_manifest(path)


def test_load_manifest_rejects_aggregate_generation_budget(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "manifest.json",
        segments=[
            {
                "id": f"scene-{index}",
                "text": "Safe text.",
                "output": f"scene-{index}.wav",
                "seed": index,
            }
            for index in range(5)
        ],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["defaults"]["max_tokens"] = 4_096
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="16,384"):
        load_manifest(path)


def test_load_manifest_rejects_casefold_output_aliases(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "manifest.json",
        segments=[
            {"id": "first", "text": "First.", "output": "Scene.wav", "seed": 1},
            {"id": "second", "text": "Second.", "output": "scene.WAV", "seed": 2},
        ],
    )

    with pytest.raises(ValueError, match="duplicate segment output"):
        load_manifest(path)


def test_load_manifest_accepts_nested_portable_ascii_output(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "manifest.json",
        segments=[
            {
                "id": "first",
                "text": "First.",
                "output": "chapter-01/scene_01.final.wav",
                "seed": 1,
            }
        ],
    )

    manifest = load_manifest(path)

    assert manifest.segments[0].output.as_posix() == "chapter-01/scene_01.final.wav"


def test_load_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path / "manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["ignored_blob"] = "not part of the schema"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported fields"):
        load_manifest(path)
