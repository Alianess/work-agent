#!/bin/zsh

set -euo pipefail

LABEL="com.work-agent"
DOMAIN="gui/$(id -u)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_PLIST="$WORKSPACE_ROOT/launchd/$LABEL.plist"
TEMPLATE_PLIST="$WORKSPACE_ROOT/launchd/$LABEL.plist.template"
TARGET_PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

ensure_source_plist() {
  if [[ -f "$SOURCE_PLIST" ]]; then
    return 0
  fi
  if [[ ! -f "$TEMPLATE_PLIST" ]]; then
    print -u2 "Missing launchd plist or template: $SOURCE_PLIST"
    return 2
  fi
  sed "s|__WORKSPACE_ROOT__|$WORKSPACE_ROOT|g" "$TEMPLATE_PLIST" > "$SOURCE_PLIST"
}

wait_for_health() {
  local attempts=20
  local health_url="http://127.0.0.1:8787/api/health"
  local attempt
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if /usr/bin/curl --connect-timeout 1 --silent --fail "$health_url" >/dev/null; then
      print "Work Agent is ready: $health_url"
      return 0
    fi
    sleep 1
  done
  print -u2 "Work Agent did not become healthy within ${attempts}s. Check: $0 errors"
  return 1
}

wait_until_unloaded() {
  local attempts=40
  local attempt
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if ! launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  print -u2 "Work Agent did not finish unloading within 10s."
  return 1
}

ensure_loaded() {
  if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
    return 0
  fi
  ensure_source_plist
  mkdir -p "$HOME/Library/LaunchAgents"
  cp "$SOURCE_PLIST" "$TARGET_PLIST"
  launchctl bootstrap "$DOMAIN" "$TARGET_PLIST"
}

case "${1:-status}" in
  install)
    ensure_source_plist
    mkdir -p "$HOME/Library/LaunchAgents"
    cp "$SOURCE_PLIST" "$TARGET_PLIST"
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    # launchd retires jobs asynchronously. Wait until the old registration is
    # gone before bootstrapping the replacement, otherwise bootstrap can fail
    # with an intermittent Input/output error.
    wait_until_unloaded
    launchctl bootstrap "$DOMAIN" "$TARGET_PLIST"
    wait_for_health
    launchctl print "$DOMAIN/$LABEL"
    ;;
  start)
    ensure_loaded
    launchctl kickstart -k "$DOMAIN/$LABEL"
    wait_for_health
    ;;
  stop)
    launchctl bootout "$DOMAIN/$LABEL"
    ;;
  restart)
    launchctl kickstart -k "$DOMAIN/$LABEL"
    wait_for_health
    ;;
  status)
    launchctl print "$DOMAIN/$LABEL"
    ;;
  logs)
    tail -n "${2:-120}" "$WORKSPACE_ROOT/tmp/work_agent-launchd.out.log"
    ;;
  errors)
    tail -n "${2:-120}" "$WORKSPACE_ROOT/tmp/work_agent-launchd.err.log"
    ;;
  *)
    print "Usage: $0 {install|start|stop|restart|status|logs|errors} [lines]"
    exit 2
    ;;
esac
