#!/usr/bin/env bash
set -euo pipefail

PID_FILE=/tmp/paraphina_shadow_runner.pid
UNIT_NAME=${PARAPHINA_SHADOW_UNIT_NAME:-paraphina-shadow-live}

declare -a pids=()

if [[ -f "$PID_FILE" ]]; then
  pid=$(head -n 1 "$PID_FILE" 2>/dev/null || true)
  if [[ -n "${pid:-}" ]]; then
    pids+=("$pid")
  fi
fi

while IFS= read -r pid; do
  if [[ -n "$pid" ]]; then
    pids+=("$pid")
  fi
done < <(
  pgrep -f '(^|/)(paraphina_live)( |$).*--trade-mode shadow' 2>/dev/null || true
)

while IFS= read -r pid; do
  if [[ -n "$pid" ]]; then
    pids+=("$pid")
  fi
done < <(
  pgrep -f 'cargo run .*paraphina_live.*--trade-mode shadow' 2>/dev/null || true
)

systemctl --user stop "$UNIT_NAME" >/dev/null 2>&1 || true
systemctl --user reset-failed "$UNIT_NAME" >/dev/null 2>&1 || true

while IFS= read -r session; do
  if [[ -n "$session" ]]; then
    screen -S "$session" -X quit >/dev/null 2>&1 || true
  fi
done < <(
  screen -ls 2>/dev/null | sed -n 's/^[[:space:]]*\([0-9]\+\.\(paraphina_shadow_live\|paraphina_shadow\)\)[[:space:]].*/\1/p'
)

if [[ ${#pids[@]} -eq 0 ]]; then
  rm -f "$PID_FILE"
  exit 0
fi

mapfile -t unique_pids < <(printf '%s\n' "${pids[@]}" | awk '!seen[$0]++')

kill -TERM "${unique_pids[@]}" 2>/dev/null || true

for _ in $(seq 1 20); do
  still_running=0
  for pid in "${unique_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      still_running=1
      break
    fi
  done
  if [[ "$still_running" -eq 0 ]]; then
    rm -f "$PID_FILE"
    exit 0
  fi
  sleep 0.5
done

kill -KILL "${unique_pids[@]}" 2>/dev/null || true
rm -f "$PID_FILE"
