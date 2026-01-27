#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${OUTDIR:-$PWD/out/$(date -u +%Y%m%dT%H%M%SZ)_all5_live_shadow}"
mkdir -p "$OUTDIR"
printf "%s\n" "$OUTDIR" > /tmp/paraphina_last_outdir.txt

set +e
PARAPHINA_TRADE_MODE=shadow \
PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter,extended,aster,paradex \
PARAPHINA_LIVE_OUT_DIR="$OUTDIR" \
PARAPHINA_TELEMETRY_MODE=jsonl \
PARAPHINA_TELEMETRY_PATH="$OUTDIR/telemetry.jsonl" \
timeout --foreground 60s \
cargo run -p paraphina --bin paraphina_live \
  --features live,live_hyperliquid,live_lighter,live_extended,live_aster,live_paradex \
  2>&1 | tee "$OUTDIR/run.log"
status=${PIPESTATUS[0]}
set -e

if [[ $status -ne 0 && $status -ne 124 ]]; then
  echo "ERROR: all-5 live shadow run failed exit_code=$status" >&2
  exit "$status"
fi
