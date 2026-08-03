#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$WORKSPACE_ROOT/.venv/bin/python"
BUNDLED_RUNTIME_ROOT="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies"
WORK_AGENT_HOST="${WORK_AGENT_HOST:-0.0.0.0}"

# launchd supplies only /usr/bin:/bin:/usr/sbin:/sbin. Keep the service
# deterministic while exposing already-installed Homebrew and bundled tools.
export PATH="$WORKSPACE_ROOT/.venv/bin:$BUNDLED_RUNTIME_ROOT/bin/override:$BUNDLED_RUNTIME_ROOT/bin:/opt/homebrew/bin:/usr/local/bin:$BUNDLED_RUNTIME_ROOT/node/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export VIRTUAL_ENV="$WORKSPACE_ROOT/.venv"
export WORK_AGENT_PYTHON="$PYTHON_BIN"
export WORK_AGENT_OFFICE_PYTHON="$PYTHON_BIN"
export WORK_AGENT_NODE="/opt/homebrew/bin/node"
export PIP_REQUIRE_VIRTUALENV=true
export PYTHONNOUSERSITE=1

if [[ ! -x "$PYTHON_BIN" ]]; then
  print -u2 "项目唯一运行环境不存在：$WORKSPACE_ROOT/.venv"
  print -u2 "请先运行：scripts/runtime_env.sh bootstrap"
  exit 2
fi

cd "$WORKSPACE_ROOT"
exec "$PYTHON_BIN" -u -m work_agent_core.web_server \
  --host "$WORK_AGENT_HOST" \
  --port 8787 \
  --workspace "$WORKSPACE_ROOT" \
  --static-dir web_frontend/dist
