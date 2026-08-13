from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from oneload_tts._filesystem import read_regular_file_bounded

SEGMENT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
PORTABLE_OUTPUT_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MAX_MANIFEST_BYTES = 256 * 1024
MAX_LANGUAGE_CHARACTERS = 64
MAX_INSTRUCTION_CHARACTERS = 1_000
MAX_OUTPUT_CHARACTERS = 240
MAX_TOP_K = 1_000
MAX_REPETITION_PENALTY = 10.0
MAX_TOKENS_PER_SEGMENT = 4_096
MAX_TOKENS_PER_MANIFEST = 16_384
TOP_LEVEL_FIELDS = {"schema_version", "defaults", "segments"}
DEFAULT_FIELDS = {
    "speaker",
    "language",
    "instruction",
    "temperature",
    "top_k",
    "top_p",
    "repetition_penalty",
    "max_tokens",
}
SEGMENT_FIELDS = {"id", "text", "output", "seed"}
SPEAKERS = {
    "Aiden",
    "Ryan",
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ono_Anna",
    "Sohee",
}


@dataclass(frozen=True)
class GenerationSettings:
    speaker: str
    language: str
    instruction: str | None
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    max_tokens: int


@dataclass(frozen=True)
class Segment:
    segment_id: str
    text: str
    output: PurePosixPath
    seed: int


@dataclass(frozen=True)
class Manifest:
    path: Path
    sha256: str
    settings: GenerationSettings
    segments: tuple[Segment, ...]
    source_bytes: bytes | None = field(default=None, repr=False, compare=False)


def _normalized_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("segment text must be a string")
    text = " ".join(value.split())
    if not text:
        raise ValueError("segment text must not be empty")
    if len(text) > 2_000:
        raise ValueError("segment text exceeds 2,000 characters")
    return text


def _output_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("segment output must be a non-empty string")
    if len(value) > MAX_OUTPUT_CHARACTERS:
        raise ValueError("segment output exceeds 240 characters")
    if "\\" in value or any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError("segment output contains unsupported characters")
    output = PurePosixPath(value)
    if output.is_absolute() or ".." in output.parts:
        raise ValueError("segment output must stay inside the output directory")
    if any(PORTABLE_OUTPUT_COMPONENT.fullmatch(part) is None for part in output.parts):
        raise ValueError("segment output must use portable ASCII file names")
    if output.suffix.lower() != ".wav":
        raise ValueError("segment output must end in .wav")
    return output


def canonical_output_key(output: PurePosixPath) -> str:
    normalized = unicodedata.normalize("NFC", output.as_posix())
    return unicodedata.normalize("NFC", normalized.casefold())


def _number(raw: dict, name: str, default: float) -> float:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(raw: dict, name: str, default: int) -> int:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _settings(raw: object) -> GenerationSettings:
    if not isinstance(raw, dict):
        raise ValueError("defaults must be an object")
    if set(raw).difference(DEFAULT_FIELDS):
        raise ValueError("defaults contains unsupported fields")
    speaker = raw.get("speaker", "Aiden")
    if not isinstance(speaker, str) or speaker not in SPEAKERS:
        raise ValueError("speaker is not supported")
    language = raw.get("language", "English")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must be a non-empty string")
    language = language.strip()
    if len(language) > MAX_LANGUAGE_CHARACTERS:
        raise ValueError("language exceeds 64 characters")
    instruction = raw.get("instruction")
    if instruction is not None and (not isinstance(instruction, str) or not instruction.strip()):
        raise ValueError("instruction must be null or a non-empty string")
    instruction = instruction.strip() if isinstance(instruction, str) else None
    if instruction is not None and len(instruction) > MAX_INSTRUCTION_CHARACTERS:
        raise ValueError("instruction exceeds 1,000 characters")
    temperature = _number(raw, "temperature", 0.9)
    top_k = _integer(raw, "top_k", 50)
    top_p = _number(raw, "top_p", 1.0)
    repetition_penalty = _number(raw, "repetition_penalty", 1.05)
    max_tokens = _integer(raw, "max_tokens", MAX_TOKENS_PER_SEGMENT)
    if not 0.05 <= temperature <= 5.0:
        raise ValueError("temperature must be between 0.05 and 5.0")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError("top_k must be between 1 and 1,000")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be greater than 0 and at most 1")
    if not 0.0 < repetition_penalty <= MAX_REPETITION_PENALTY:
        raise ValueError("repetition_penalty must be greater than 0 and at most 10")
    if not 1 <= max_tokens <= MAX_TOKENS_PER_SEGMENT:
        raise ValueError("max_tokens must be between 1 and 4,096")
    return GenerationSettings(
        speaker=speaker,
        language=language,
        instruction=instruction,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
    )


def load_manifest(path: Path) -> Manifest:
    requested = path.expanduser()
    payload = read_regular_file_bounded(
        requested,
        maximum_bytes=MAX_MANIFEST_BYTES,
        label="manifest",
    )
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError("manifest exceeds 256 KiB")
    raw = json.loads(payload)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    if set(raw).difference(TOP_LEVEL_FIELDS):
        raise ValueError("manifest contains unsupported fields")
    raw_segments = raw.get("segments")
    if not isinstance(raw_segments, list) or not 1 <= len(raw_segments) <= 32:
        raise ValueError("manifest must contain between 1 and 32 segments")
    segments: list[Segment] = []
    identifiers: set[str] = set()
    output_keys: set[str] = set()
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise ValueError(f"segment {index} must be an object")
        if set(raw_segment).difference(SEGMENT_FIELDS):
            raise ValueError(f"segment {index} contains unsupported fields")
        segment_id = raw_segment.get("id")
        if not isinstance(segment_id, str) or SEGMENT_ID.fullmatch(segment_id) is None:
            raise ValueError(f"invalid segment id at index {index}")
        output = _output_path(raw_segment.get("output"))
        if segment_id in identifiers:
            raise ValueError(f"duplicate segment id: {segment_id}")
        output_key = canonical_output_key(output)
        if output_key in output_keys:
            raise ValueError(f"duplicate segment output at index {index}")
        identifiers.add(segment_id)
        output_keys.add(output_key)
        seed = raw_segment.get("seed")
        if not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ValueError(f"segment seed must be a uint32: {segment_id}")
        segments.append(
            Segment(
                segment_id=segment_id,
                text=_normalized_text(raw_segment.get("text")),
                output=output,
                seed=seed,
            )
        )
    settings = _settings(raw.get("defaults", {}))
    if len(segments) * settings.max_tokens > MAX_TOKENS_PER_MANIFEST:
        raise ValueError("manifest generation budget exceeds 16,384 tokens")
    return Manifest(
        path=requested,
        sha256=hashlib.sha256(payload).hexdigest(),
        settings=settings,
        segments=tuple(segments),
        source_bytes=payload,
    )
