#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

OUTDIR=${OUTDIR:?OUTDIR is required}
BIN=${PARAPHINA_LIVE_BIN:?PARAPHINA_LIVE_BIN is required}
CONNECTORS=${PARAPHINA_LIVE_CONNECTORS:-extended,hyperliquid,aster,lighter,paradex}
SESSION_NAME=${PARAPHINA_SHADOW_SESSION_NAME:-paraphina_shadow_live}
GLOBAL_PID_FILE=${PARAPHINA_SHADOW_GLOBAL_PID_FILE:-/tmp/paraphina_shadow_runner.pid}
GLOBAL_STATE_FILE=${PARAPHINA_SHADOW_GLOBAL_STATE_FILE:-/tmp/paraphina_shadow_runner.state}

RUN_LOG="$OUTDIR/run.log"
COMMAND_FILE="$OUTDIR/command.txt"
LOCAL_STATE_FILE="$OUTDIR/runner.state"
RUNNER_PID_FILE="$OUTDIR/runner.pid"
SUPERVISOR_PID_FILE="$OUTDIR/supervisor.pid"

mkdir -p "$OUTDIR"
cd "$REPO_ROOT"

started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
supervisor_pid=$$
runner_pid=""
state="starting"
stopped_at=""
exit_code=""
signal_num=""

write_state() {
  local tmp
  tmp=$(mktemp "$OUTDIR/.runner.state.XXXXXX")
  {
    printf 'state=%s\n' "$state"
    printf 'supervisor_pid=%s\n' "$supervisor_pid"
    printf 'runner_pid=%s\n' "${runner_pid:-}"
    printf 'session_name=%s\n' "$SESSION_NAME"
    printf 'outdir=%s\n' "$OUTDIR"
    printf 'telemetry_path=%s\n' "$OUTDIR/telemetry.jsonl"
    printf 'started_at=%s\n' "$started_at"
    printf 'stopped_at=%s\n' "$stopped_at"
    printf 'exit_code=%s\n' "$exit_code"
    printf 'signal=%s\n' "$signal_num"
    printf 'bin=%s\n' "$BIN"
    printf 'connectors=%s\n' "$CONNECTORS"
    printf 'run_log=%s\n' "$RUN_LOG"
    printf 'command_file=%s\n' "$COMMAND_FILE"
  } >"$tmp"
  mv "$tmp" "$LOCAL_STATE_FILE"
  cp "$LOCAL_STATE_FILE" "$GLOBAL_STATE_FILE"
}

log_line() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$RUN_LOG"
}

forward_stop() {
  state="stopping"
  write_state
  if [[ -n "${runner_pid:-}" ]] && kill -0 "$runner_pid" 2>/dev/null; then
    kill -TERM "$runner_pid" 2>/dev/null || true
  fi
}

trap forward_stop TERM INT HUP

cat >"$COMMAND_FILE" <<EOF
env \
  HL_COIN=ETH \
  LIGHTER_MARKET=ETH-USD \
  PARADEX_MARKET=ETH-USD-PERP \
  PARAPHINA_PARADEX_PUBLIC_FEED=${PARAPHINA_PARADEX_PUBLIC_FEED:-orderbook} \
  ASTER_MARKET=ETHUSDT \
  EXTENDED_MARKET=ETH-USD \
  EXTENDED_REST_URL=https://api.starknet.extended.exchange \
  EXTENDED_FUNDING_PATH=/api/v1/info/markets/ETH-USD/stats \
  PARAPHINA_TRADE_MODE=shadow \
  PARAPHINA_LIVE_OUT_DIR=$OUTDIR \
  PARAPHINA_TELEMETRY_MODE=jsonl \
  PARAPHINA_TELEMETRY_PATH=$OUTDIR/telemetry.jsonl \
  PARAPHINA_HL_STATE_STALE_MS_OVERRIDE=1500 \
  PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE=1500 \
  PARAPHINA_PARADEX_STATE_STALE_MS_OVERRIDE=${PARAPHINA_PARADEX_STATE_STALE_MS_OVERRIDE:-3000} \
  PARAPHINA_FUNDING_STALE_MS=600000 \
  PARAPHINA_FUNDING_AVOID_WINDOW_MS=120000 \
  PARAPHINA_WS_AUDIT=${PARAPHINA_WS_AUDIT:-1} \
  PARAPHINA_MARKET_RX_STATS=${PARAPHINA_MARKET_RX_STATS:-1} \
  PARAPHINA_MARKET_RX_STATS_EVERY_TICKS=${PARAPHINA_MARKET_RX_STATS_EVERY_TICKS:-20} \
  PARAPHINA_MARKET_RX_STATS_PATH=$OUTDIR/market_rx_stats.log \
  HL_FUNDING_POLL_MS=5000 \
  PARADEX_FUNDING_POLL_MS=5000 \
  LIGHTER_FUNDING_POLL_MS=5000 \
  EXTENDED_FUNDING_POLL_MS=5000 \
  ASTER_FUNDING_POLL_MS=5000 \
  RUST_LOG=${RUST_LOG:-info} \
  $BIN \
  --trade-mode shadow \
  --connectors $CONNECTORS \
  --out-dir $OUTDIR
EOF

printf '%s\n' "$supervisor_pid" >"$SUPERVISOR_PID_FILE"
write_state
log_line "supervisor_start outdir=$OUTDIR session=$SESSION_NAME bin=$BIN connectors=$CONNECTORS"

(
  exec env \
    HL_COIN=ETH \
    LIGHTER_MARKET=ETH-USD \
    PARADEX_MARKET=ETH-USD-PERP \
    PARAPHINA_PARADEX_PUBLIC_FEED="${PARAPHINA_PARADEX_PUBLIC_FEED:-orderbook}" \
    ASTER_MARKET=ETHUSDT \
    EXTENDED_MARKET=ETH-USD \
    EXTENDED_REST_URL=https://api.starknet.extended.exchange \
    EXTENDED_FUNDING_PATH=/api/v1/info/markets/ETH-USD/stats \
    PARAPHINA_TRADE_MODE=shadow \
    PARAPHINA_LIVE_OUT_DIR="$OUTDIR" \
    PARAPHINA_TELEMETRY_MODE=jsonl \
    PARAPHINA_TELEMETRY_PATH="$OUTDIR/telemetry.jsonl" \
    PARAPHINA_HL_STATE_STALE_MS_OVERRIDE=1500 \
    PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE=1500 \
    PARAPHINA_PARADEX_STATE_STALE_MS_OVERRIDE="${PARAPHINA_PARADEX_STATE_STALE_MS_OVERRIDE:-3000}" \
    PARAPHINA_FUNDING_STALE_MS=600000 \
    PARAPHINA_FUNDING_AVOID_WINDOW_MS=120000 \
    PARAPHINA_WS_AUDIT="${PARAPHINA_WS_AUDIT:-1}" \
    PARAPHINA_MARKET_RX_STATS="${PARAPHINA_MARKET_RX_STATS:-1}" \
    PARAPHINA_MARKET_RX_STATS_EVERY_TICKS="${PARAPHINA_MARKET_RX_STATS_EVERY_TICKS:-20}" \
    PARAPHINA_MARKET_RX_STATS_PATH="$OUTDIR/market_rx_stats.log" \
    HL_FUNDING_POLL_MS=5000 \
    PARADEX_FUNDING_POLL_MS=5000 \
    LIGHTER_FUNDING_POLL_MS=5000 \
    EXTENDED_FUNDING_POLL_MS=5000 \
    ASTER_FUNDING_POLL_MS=5000 \
    RUST_LOG="${RUST_LOG:-info}" \
    "$BIN" \
    --trade-mode shadow \
    --connectors "$CONNECTORS" \
    --out-dir "$OUTDIR"
) >>"$RUN_LOG" 2>&1 &

runner_pid=$!
printf '%s\n' "$runner_pid" >"$GLOBAL_PID_FILE"
printf '%s\n' "$runner_pid" >"$RUNNER_PID_FILE"
state="running"
write_state
log_line "runner_start pid=$runner_pid"

set +e
wait "$runner_pid"
rc=$?
set -e

stopped_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if (( rc >= 128 )); then
  signal_num=$((rc - 128))
else
  exit_code=$rc
fi
state="exited"
write_state
log_line "runner_exit pid=$runner_pid rc=$rc exit_code=${exit_code:-none} signal=${signal_num:-none}"

if [[ -f "$GLOBAL_PID_FILE" ]] && [[ "$(cat "$GLOBAL_PID_FILE" 2>/dev/null || true)" == "$runner_pid" ]]; then
  rm -f "$GLOBAL_PID_FILE"
fi
