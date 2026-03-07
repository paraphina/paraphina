#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

OUTDIR_ROOT=${OUTDIR_ROOT:-/tmp}
OUTDIR=${OUTDIR:-"$OUTDIR_ROOT/paraphina_live_shadow_$(date -u +%Y%m%dT%H%M%SZ)"}
PID_FILE=/tmp/paraphina_shadow_runner.pid
STATE_FILE=/tmp/paraphina_shadow_runner.state
LAST_OUTDIR_FILE=/tmp/paraphina_last_outdir.txt
LATEST_LINK=/tmp/paraphina_shadow_latest
SESSION_NAME=${PARAPHINA_SHADOW_SESSION_NAME:-paraphina_shadow_live}
UNIT_NAME=${PARAPHINA_SHADOW_UNIT_NAME:-paraphina-shadow-live}
BIN=${PARAPHINA_LIVE_BIN:-"$REPO_ROOT/target/release/paraphina_live"}
CONNECTORS=${PARAPHINA_LIVE_CONNECTORS:-extended,hyperliquid,aster,lighter,paradex}

if [[ ! -x "$BIN" && -x "$REPO_ROOT/target/debug/paraphina_live" ]]; then
  BIN="$REPO_ROOT/target/debug/paraphina_live"
fi

if [[ ! -x "$BIN" ]]; then
  echo "error: paraphina_live binary not found or not executable: $BIN" >&2
  exit 1
fi

if ! command -v systemd-run >/dev/null 2>&1 && ! command -v screen >/dev/null 2>&1; then
  echo "error: neither systemd-run nor screen found in PATH." >&2
  exit 1
fi

"$SCRIPT_DIR/stop_all5_shadow_live.sh"

mkdir -p "$OUTDIR"
printf '%s\n' "$OUTDIR" > "$LAST_OUTDIR_FILE"
ln -sfn "$OUTDIR" "$LATEST_LINK"
: > "$OUTDIR/run.log"

launcher=""
if command -v systemd-run >/dev/null 2>&1; then
  systemctl --user stop "$UNIT_NAME" >/dev/null 2>&1 || true
  systemctl --user reset-failed "$UNIT_NAME" >/dev/null 2>&1 || true
  systemd-run \
    --user \
    --unit "$UNIT_NAME" \
    --description "paraphina shadow live ETH" \
    --collect \
    --working-directory "$REPO_ROOT" \
    --setenv=OUTDIR="$OUTDIR" \
    --setenv=PARAPHINA_LIVE_BIN="$BIN" \
    --setenv=PARAPHINA_LIVE_CONNECTORS="$CONNECTORS" \
    --setenv=PARAPHINA_SHADOW_SESSION_NAME="$UNIT_NAME" \
    --setenv=PARAPHINA_SHADOW_GLOBAL_PID_FILE="$PID_FILE" \
    --setenv=PARAPHINA_SHADOW_GLOBAL_STATE_FILE="$STATE_FILE" \
    --setenv=RUST_LOG="${RUST_LOG:-info}" \
    /bin/bash -lc 'exec tools/paraphina_shadow_supervisor.sh' >/dev/null
  launcher="systemd:$UNIT_NAME"
else
  screen -wipe >/dev/null 2>&1 || true
  screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true
  env \
    OUTDIR="$OUTDIR" \
    PARAPHINA_LIVE_BIN="$BIN" \
    PARAPHINA_LIVE_CONNECTORS="$CONNECTORS" \
    PARAPHINA_SHADOW_SESSION_NAME="$SESSION_NAME" \
    PARAPHINA_SHADOW_GLOBAL_PID_FILE="$PID_FILE" \
    PARAPHINA_SHADOW_GLOBAL_STATE_FILE="$STATE_FILE" \
    RUST_LOG="${RUST_LOG:-info}" \
    screen -S "$SESSION_NAME" -dm bash -lc 'cd "$1" && exec tools/paraphina_shadow_supervisor.sh' _ "$REPO_ROOT"
  launcher="screen:$SESSION_NAME"
fi

pid=""
for _ in $(seq 1 60); do
  if [[ -f "$OUTDIR/runner.state" ]]; then
    state=$(awk -F= '/^state=/{print $2}' "$OUTDIR/runner.state" | tail -n 1)
    pid=$(awk -F= '/^runner_pid=/{print $2}' "$OUTDIR/runner.state" | tail -n 1)
    if [[ "$state" == "running" && -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      break
    fi
    if [[ "$state" == "exited" ]]; then
      echo "error: paraphina_live exited during startup." >&2
      tail -n 120 "$OUTDIR/run.log" >&2 || true
      exit 1
    fi
  fi
  sleep 0.25
done

if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
  echo "error: paraphina_live did not reach running state." >&2
  tail -n 120 "$OUTDIR/run.log" >&2 || true
  exit 1
fi

printf 'pid=%s\n' "$pid"
printf 'outdir=%s\n' "$OUTDIR"
printf 'launcher=%s\n' "$launcher"
printf 'watch=./view\n'
