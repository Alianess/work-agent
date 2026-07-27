#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INVOCATION_DIR="$PWD"
VENV_DIR="$WORKSPACE_ROOT/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
BUNDLED_RUNTIME_ROOT="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies"
BOOTSTRAP_PYTHON="/opt/homebrew/opt/python@3.12/bin/python3.12"
MANAGED_NODE="/opt/homebrew/bin/node"
MANAGED_NPM="/opt/homebrew/bin/npm"

export PATH="$VENV_DIR/bin:$BUNDLED_RUNTIME_ROOT/bin/override:$BUNDLED_RUNTIME_ROOT/bin:/opt/homebrew/bin:/usr/local/bin:$BUNDLED_RUNTIME_ROOT/node/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export VIRTUAL_ENV="$VENV_DIR"
export WORK_AGENT_PYTHON="$PYTHON_BIN"
export WORK_AGENT_OFFICE_PYTHON="$PYTHON_BIN"
export WORK_AGENT_NODE="$MANAGED_NODE"
export PIP_REQUIRE_VIRTUALENV=true
export PYTHONNOUSERSITE=1

require_runtime() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    print -u2 "项目唯一运行环境不存在：$VENV_DIR"
    print -u2 "请先运行：scripts/runtime_env.sh bootstrap"
    exit 2
  fi
}

case "${1:-check}" in
  bootstrap)
    if [[ ! -x "$BOOTSTRAP_PYTHON" ]]; then
      print -u2 "缺少 Python 3.12：$BOOTSTRAP_PYTHON"
      exit 2
    fi
    if [[ ! -x "$PYTHON_BIN" ]]; then
      "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
    fi
    "$PYTHON_BIN" -m pip install -r "$WORKSPACE_ROOT/requirements-runtime.txt"
    exec "$0" check
    ;;
  check)
    require_runtime
    cd "$WORKSPACE_ROOT"
    exec "$PYTHON_BIN" scripts/runtime_check.py
    ;;
  python)
    require_runtime
    shift
    cd "$WORKSPACE_ROOT"
    exec "$PYTHON_BIN" "$@"
    ;;
  pip)
    require_runtime
    shift
    cd "$WORKSPACE_ROOT"
    exec "$PYTHON_BIN" -m pip "$@"
    ;;
  node)
    require_runtime
    shift
    cd "$INVOCATION_DIR"
    exec "$MANAGED_NODE" "$@"
    ;;
  npm)
    require_runtime
    shift
    cd "$INVOCATION_DIR"
    exec "$MANAGED_NPM" "$@"
    ;;
  *)
    print -u2 "用法：scripts/runtime_env.sh {bootstrap|check|python|pip|node|npm}"
    exit 2
    ;;
esac
