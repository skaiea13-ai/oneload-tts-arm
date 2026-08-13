from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SCRIPTS = (
    Path("scripts/download-model.sh"),
    Path("scripts/run-benchmark.sh"),
)


@pytest.mark.parametrize("relative_script", PUBLIC_SCRIPTS)
def test_public_script_binds_relative_entrypoint_to_current_repository(
    relative_script: Path,
) -> None:
    source = (PROJECT_ROOT / relative_script).read_text(encoding="utf-8")

    assert source.startswith("#!/bin/bash -p\n")
    assert "unset BASH_ENV ENV" in source
    assert '/usr/sbin/lsof -a -p "$$" -d 255 -FDi' in source
    assert 'source_path="${BASH_SOURCE[0]}"' in source
    assert 'os.open(".", directory_flags)' in source
    assert "os.O_NOFOLLOW" in source
    assert "script_state.st_nlink == 1" in source
    assert "Run this script from the OneLoad repository root." in source
    assert "cd -P" not in source


@pytest.mark.parametrize("relative_script", PUBLIC_SCRIPTS)
def test_public_script_ignores_bash_env_startup_file(tmp_path: Path, relative_script: Path) -> None:
    root = _prepare_root(tmp_path, relative_script)
    marker = tmp_path / "bash-env-was-sourced"
    startup = tmp_path / "startup.sh"
    startup.write_text(f"printf unsafe > {marker}\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["BASH_ENV"] = str(startup)

    subprocess.run(  # noqa: S603
        [f"./{relative_script}"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert not marker.exists()


def _prepare_root(tmp_path: Path, relative_script: Path) -> Path:
    root = tmp_path / "repository"
    destination = root / relative_script
    destination.parent.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / relative_script, destination)
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (root / "model-lock.json").write_text("{}\n", encoding="utf-8")
    return root


def test_download_script_runs_from_documented_repository_root(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path, Path("scripts/download-model.sh"))
    marker = tmp_path / "download-arguments.txt"
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        (
            "#!/bin/sh\nset -eu\n"
            'if [ "$3" = "oneload_tts.download_guard" ]; then exit 0; fi\n'
            'printf \'%s\\n\' "$*" > "$ONELOAD_TEST_MARKER"\n'
        ),
        encoding="utf-8",
    )
    python.chmod(0o700)
    environment = os.environ.copy()
    environment["ONELOAD_TEST_MARKER"] = str(marker)

    subprocess.run(  # noqa: S603
        ["./scripts/download-model.sh"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    arguments = marker.read_text(encoding="utf-8")
    assert arguments.startswith("-I -B -m huggingface_hub.cli.hf download ")
    assert "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit" in arguments
    assert "--revision 1c6c0ff58c43afa8df571facde2efa077efd85e2" in arguments


def test_download_script_rejects_symlinked_custom_target_before_download(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path, Path("scripts/download-model.sh"))
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        (
            "#!/bin/sh\nset -eu\n"
            f'if [ "$4" = "oneload_tts.download_guard" ]; then '
            f"exec {sys.executable} -I -B -m oneload_tts.download_guard "
            '"$5" "$6" "$7" "$8"; fi\n'
            'printf unsafe > "$ONELOAD_TEST_MARKER"\n'
        ),
        encoding="utf-8",
    )
    python.chmod(0o700)
    real_target = tmp_path / "real-model"
    real_target.mkdir()
    linked_target = tmp_path / "linked-model"
    linked_target.symlink_to(real_target, target_is_directory=True)
    marker = tmp_path / "download-started"
    environment = os.environ.copy()
    environment.update(
        {
            "ONELOAD_MODEL_DIR": str(linked_target),
            "ONELOAD_TEST_MARKER": str(marker),
        }
    )

    rejected = subprocess.run(  # noqa: S603
        ["./scripts/download-model.sh"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert rejected.returncode == 65
    assert "unprotected or partially populated" in rejected.stderr
    assert not marker.exists()


def test_benchmark_script_runs_from_documented_repository_root(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path, Path("scripts/run-benchmark.sh"))
    marker = tmp_path / "benchmark-invocation.txt"
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        '#!/bin/sh\nset -eu\nprintf \'%s\\n%s\\n\' "$PWD" "$*" > "$ONELOAD_TEST_MARKER"\n',
        encoding="utf-8",
    )
    python.chmod(0o700)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/bin/sh\nset -eu\nprintf 'unsafe uv lookup\\n' > \"$ONELOAD_UNSAFE_MARKER\"\nexit 99\n",
        encoding="utf-8",
    )
    uv.chmod(0o700)
    unsafe_marker = tmp_path / "unsafe-uv.txt"
    environment = os.environ.copy()
    environment.update(
        {
            "ONELOAD_MODEL_DIR": "/tmp/synthetic-model",
            "ONELOAD_TEST_MARKER": str(marker),
            "ONELOAD_UNSAFE_MARKER": str(unsafe_marker),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
        }
    )

    subprocess.run(  # noqa: S603
        ["./scripts/run-benchmark.sh"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    working_directory, arguments = marker.read_text(encoding="utf-8").splitlines()
    assert working_directory == str(root)
    assert arguments.startswith("-I -B -m oneload_tts benchmark")
    assert not unsafe_marker.exists()


def test_build_backend_is_in_the_frozen_project_lock() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8"))

    assert "hatchling==1.27.0" in project["dependency-groups"]["dev"]
    assert any(
        package["name"] == "hatchling" and package["version"] == "1.27.0"
        for package in lock["package"]
    )


@pytest.mark.parametrize("relative_script", PUBLIC_SCRIPTS)
def test_public_script_rejects_absolute_or_symlink_entrypoint(
    tmp_path: Path, relative_script: Path
) -> None:
    absolute = subprocess.run(  # noqa: S603
        [str(PROJECT_ROOT / relative_script)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert absolute.returncode == 65
    assert "Run this script from the OneLoad repository root." in absolute.stderr

    alias = tmp_path / relative_script.name
    alias.symlink_to(PROJECT_ROOT / relative_script)
    linked = subprocess.run(  # noqa: S603
        [str(alias)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert linked.returncode == 65
    assert "Run this script from the OneLoad repository root." in linked.stderr


def test_public_script_rejects_hard_linked_repository_entrypoint(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    source = tmp_path / "source.sh"
    shutil.copy2(PROJECT_ROOT / "scripts/run-benchmark.sh", source)
    os.link(source, scripts / "run-benchmark.sh")
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (root / "model-lock.json").write_text("{}\n", encoding="utf-8")

    rejected = subprocess.run(  # noqa: S603
        ["./scripts/run-benchmark.sh"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert rejected.returncode == 65
    assert "outside the bound repository root" in rejected.stderr
