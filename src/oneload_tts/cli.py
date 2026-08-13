from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote

from oneload_tts.benchmark import run_benchmark, write_json_atomic
from oneload_tts.engine import load_model_lock, render_manifest, validate_model
from oneload_tts.manifest import load_manifest

ABSOLUTE_PATH_FRAGMENT = re.compile(r"(^|[^A-Za-z0-9])/(?!\s)")
FILE_URI_FRAGMENT = re.compile(r"(?i)\bfile\s*:")
NETWORK_URI_FRAGMENT = re.compile(r"(?i)\bhttps?://[^\s]+")
MAX_ERROR_DECODE_PASSES = 8


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oneload-tts",
        description="Batch Qwen3-TTS narration with one persistent model load on Apple Silicon.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate", help="Validate architecture and model files.")
    validate.add_argument("--model-path", type=Path, required=True)
    render = subcommands.add_parser("render", help="Render a scene manifest.")
    render.add_argument("--manifest", type=Path, required=True)
    render.add_argument("--model-path", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--only", help="Render one segment in a cold process baseline.")
    benchmark = subcommands.add_parser(
        "benchmark", help="Compare cold per-scene processes with persistent batch rendering."
    )
    benchmark.add_argument("--manifest", type=Path, required=True)
    benchmark.add_argument("--model-path", type=Path, required=True)
    benchmark.add_argument("--result", type=Path, required=True)
    benchmark.add_argument("--trials", type=int, default=3)
    return parser


def _architecture() -> str:
    machine = platform.machine().lower()
    if machine not in {"arm64", "aarch64"}:
        raise RuntimeError(f"OneLoad requires an Arm64 host, found {machine}")
    return machine


def _safe_error_message(error: Exception, args: argparse.Namespace) -> str:
    try:
        try:
            message = str(error) or error.__class__.__name__
        except Exception:
            message = "operation failed"
        replacements: dict[str, str] = {}

        def add_private_path(value: Path, label: str) -> None:
            raw = str(value)
            if raw not in {"", ".", "..", "/"}:
                replacements[raw] = label
            try:
                expanded = value.expanduser()
                if expanded.is_absolute():
                    for candidate in (expanded, *expanded.parents):
                        private = str(candidate)
                        if private not in {"", ".", "/"}:
                            replacements[private] = label
            except (OSError, RuntimeError):
                pass
            try:
                resolved = value.expanduser().resolve(strict=False)
            except (OSError, RuntimeError):
                return
            for candidate in (resolved, *resolved.parents):
                private = str(candidate)
                if private not in {"", ".", "/"}:
                    replacements[private] = label

        for name in ("manifest", "model_path", "output_dir", "result"):
            value = getattr(args, name, None)
            if isinstance(value, Path):
                add_private_path(value, f"<{name.replace('_', '-')}>")

        for getter, label in ((Path.cwd, "<cwd>"), (Path.home, "<home>")):
            try:
                private_root = getter()
            except (OSError, RuntimeError):
                continue
            add_private_path(private_root, label)

        for private, label in sorted(
            replacements.items(), key=lambda item: len(item[0]), reverse=True
        ):
            message = message.replace(private, label)
        decoded = message
        for _ in range(MAX_ERROR_DECODE_PASSES):
            if (
                NETWORK_URI_FRAGMENT.search(decoded)
                or FILE_URI_FRAGMENT.search(decoded)
                or ABSOLUTE_PATH_FRAGMENT.search(decoded)
            ):
                message = "operation failed"
                break
            next_decoded = unquote(decoded)
            if next_decoded == decoded:
                break
            decoded = next_decoded
        else:
            message = "operation failed"
        return "".join(
            character
            if not unicodedata.category(character).startswith("C")
            else f"\\u{ord(character):04x}"
            for character in message
        )
    except Exception:
        return "operation failed"


def main() -> int:
    args = _parser().parse_args()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        architecture = _architecture()
        if args.command == "validate":
            validation = validate_model(args.model_path, load_model_lock())
            payload = {"status": "ready", "architecture": architecture, **validation}
        elif args.command == "render":
            payload = render_manifest(
                manifest=load_manifest(args.manifest),
                model_path=args.model_path,
                output_dir=args.output_dir,
                only=args.only,
            )
        else:
            payload = run_benchmark(
                manifest=load_manifest(args.manifest),
                model_path=args.model_path,
                trials=args.trials,
            )
            write_json_atomic(args.result, payload)
        print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(f"oneload-tts: {_safe_error_message(error, args)}", file=sys.stderr)
        return 1
