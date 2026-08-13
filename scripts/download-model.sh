#!/bin/bash -p
set -euo pipefail

if [[ "${SHELLOPTS:-}" == *privileged* ]]; then
  :
else
  echo "Run this script directly with its privileged-mode Bash shebang." >&2
  exit 65
fi
unset BASH_ENV ENV

oneload_script_lsof="$(/usr/sbin/lsof -a -p "$$" -d 255 -FDi 2>/dev/null || true)"
oneload_script_device="$(
  printf '%s\n' "${oneload_script_lsof}" | /usr/bin/sed -n 's/^D//p'
)"
oneload_script_inode="$(
  printf '%s\n' "${oneload_script_lsof}" | /usr/bin/sed -n 's/^i//p'
)"
if [[ ! "${oneload_script_device}" =~ ^0x[0-9a-fA-F]+$ ||
      ! "${oneload_script_inode}" =~ ^[0-9]+$ ]]; then
  echo "Could not bind the executing script." >&2
  exit 65
fi
bind_repository_root() {
  local source_relative_prefix="$1"
  local source_name="$2"
  local source_path="${BASH_SOURCE[0]}"
  local expected_source="${source_relative_prefix}/${source_name}"
  if [[ "${source_path}" != "${expected_source}" &&
        "${source_path}" != "./${expected_source}" ]]; then
    echo "Run this script from the OneLoad repository root." >&2
    return 65
  fi
  if ! /usr/bin/python3 -I -S - "${source_relative_prefix}" "${source_name}" \
    "$((oneload_script_device))" "${oneload_script_inode}" <<'PY'
import os
import stat
import sys

directory_fds = []
script_fd = None
verified = False
try:
    parts = sys.argv[1].split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    directory_fds.append(os.open(".", directory_flags))
    for part in parts:
        directory_fds.append(os.open(part, directory_flags, dir_fd=directory_fds[-1]))
    script_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    script_fd = os.open(sys.argv[2], script_flags, dir_fd=directory_fds[-1])
    script_state = os.fstat(script_fd)
    verified = (
        stat.S_ISREG(script_state.st_mode)
        and script_state.st_nlink == 1
        and (script_state.st_dev, script_state.st_ino) == (int(sys.argv[3]), int(sys.argv[4]))
    )
except (OSError, ValueError):
    pass
finally:
    if script_fd is not None:
        os.close(script_fd)
    for directory_fd in reversed(directory_fds):
        os.close(directory_fd)
raise SystemExit(0 if verified else 1)
PY
  then
    echo "Refusing a script entry point outside the bound repository root." >&2
    return 65
  fi
}

bind_repository_root "scripts" "download-model.sh"
if [[ ! -f pyproject.toml || ! -f model-lock.json ]]; then
  echo "Could not verify the OneLoad repository root." >&2
  exit 65
fi
model_id="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit"
revision="1c6c0ff58c43afa8df571facde2efa077efd85e2"
model_target="${ONELOAD_MODEL_DIR:-.models/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit}"
python_bin="./.venv/bin/python"

if [[ ! -x "${python_bin}" ]]; then
  echo "Run uv sync --frozen --python 3.12 before downloading the model." >&2
  exit 69
fi

if ! "${python_bin}" -I -B -m oneload_tts.download_guard \
  --target "${model_target}" \
  --lock model-lock.json; then
  echo "Refusing an unprotected or partially populated model download target." >&2
  exit 65
fi

/usr/bin/env -u BASH_ENV -u ENV HF_HUB_DISABLE_TELEMETRY=1 \
  "${python_bin}" -I -B -m huggingface_hub.cli.hf download \
  "${model_id}" \
  --revision "${revision}" \
  --local-dir "${model_target}"

printf '%s\n' "Pinned model is ready under the selected local model directory."
