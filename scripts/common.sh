#!/usr/bin/env bash

resolve_python() {
  local python_bin="${1:-}"
  local code_dir="${2}"

  if [[ -n "${python_bin}" ]]; then
    echo "${python_bin}"
    return
  fi

  if [[ -x "${code_dir}/.venv/bin/python" ]]; then
    echo "${code_dir}/.venv/bin/python"
  elif [[ -x "${code_dir}/.venv/Scripts/python.exe" ]]; then
    echo "${code_dir}/.venv/Scripts/python.exe"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "${CONDA_PREFIX}/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/python.exe" ]]; then
    echo "${CONDA_PREFIX}/python.exe"
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  elif command -v python >/dev/null 2>&1; then
    command -v python
  else
    echo "No Python interpreter found. Please create a .venv or use --python." >&2
    return 1
  fi
}

require_python_modules() {
  local python_bin="$1"
  shift

  local missing=""
  if ! missing="$("${python_bin}" - "$@" <<'PY'
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    print(", ".join(missing))
    raise SystemExit(1)
PY
  )"; then
    if [[ -z "${missing}" ]]; then
      missing="$*"
    fi
    echo "Error: selected Python is missing required module(s): ${missing}" >&2
    echo "Python: ${python_bin}" >&2
    echo "Install project dependencies with:" >&2
    echo "  ${python_bin} -m pip install -e '.[dev,ml]'" >&2
    echo "Or pass a prepared interpreter with --python PATH." >&2
    return 1
  fi
}
