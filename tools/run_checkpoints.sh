#!/usr/bin/env bash
# Automated checkpoint runner for 100K tick telemetry analysis.
# Polls telemetry.jsonl until each checkpoint threshold is reached, then runs the analyzer.

set -euo pipefail

TELEMETRY="/tmp/shadow_eth_post_fix/telemetry.jsonl"
ANALYZER="/home/developer/code/paraphina/tools/telemetry_analyzer.py"
OUT_DIR="/home/developer/code/paraphina/out"
mkdir -p "$OUT_DIR"

# Checkpoints: 20k, 30k, 40k, 50k, 60k, 70k, 80k, 90k, 100k
# (10k already done)
CHECKPOINTS=(20000 30000 40000 50000 60000 70000 80000 90000 100000)

get_tick_count() {
    wc -l < "$TELEMETRY" 2>/dev/null || echo 0
}

prev_checkpoint="$OUT_DIR/checkpoint_10k.json"

for target in "${CHECKPOINTS[@]}"; do
    label="${target:0:$((${#target}-3))}k"
    echo "$(date '+%H:%M:%S') Waiting for ${target} ticks (checkpoint ${label})..."

    while true; do
        count=$(get_tick_count)
        if [ "$count" -ge "$target" ]; then
            break
        fi
        remaining=$((target - count))
        # At ~4 ticks/sec, estimate wait time
        eta_sec=$((remaining / 4))
        echo "  $(date '+%H:%M:%S') Current: ${count} ticks, need ${target}, ~${eta_sec}s remaining"
        # Sleep adaptively: min(remaining/4, 60) seconds
        sleep_time=$((eta_sec < 60 ? (eta_sec > 5 ? eta_sec : 5) : 60))
        sleep "$sleep_time"
    done

    echo "$(date '+%H:%M:%S') Running checkpoint ${label} (${target} ticks)..."
    cp_json="$OUT_DIR/checkpoint_${label}.json"
    cp_report="$OUT_DIR/report_${label}.txt"

    python3 "$ANALYZER" \
        --telemetry "$TELEMETRY" \
        --max-ticks "$target" \
        --checkpoint-json "$cp_json" \
        --prev-checkpoint "$prev_checkpoint" \
        --output "$cp_report" \
        2>&1

    echo "$(date '+%H:%M:%S') Checkpoint ${label} complete -> ${cp_report}"
    prev_checkpoint="$cp_json"
    echo "---"
done

echo "$(date '+%H:%M:%S') All checkpoints complete!"
