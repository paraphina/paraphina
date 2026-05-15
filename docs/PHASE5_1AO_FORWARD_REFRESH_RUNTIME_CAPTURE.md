# Phase 5.1AO Forward-Refresh Runtime Capture

## Status

This scaffold is passive, disabled by default, and fail-closed. It does not clear the Phase 5.1 blocker because `forward_refresh.remaining.jsonl` still requires future source-owner observations.

## Config

```toml
[phase51_forward_refresh_capture]
enabled = false
output_path = "/home/ubuntu/source_owner_inbox/phase51/forward_refresh.remaining.jsonl"
allow_live = false
append_only = true
max_rows = 5000
```

The default Rust config mirrors this block. Enabling the lane requires an explicit config change plus a non-live/shadow-safe execution mode.

## Files

- `paraphina/src/live/phase51_forward_refresh_capture.rs`: runtime capture scaffold, row sanitizer, fail-closed checks.
- `paraphina/src/live/mod.rs`: exports the scaffold types.
- `paraphina/src/live/runner.rs`: initializes the scaffold and fails closed if an enabled configuration is unsafe.
- `paraphina/src/live/types.rs`: safe audit telemetry shape for capture status reporting.
- `paraphina/src/config.rs`: disabled-by-default config block.
- `tools/phase51ao_forward_refresh_remaining_merge.py`: append-safe merge tool for `forward_refresh.jsonl` and `forward_refresh.remaining.jsonl`.
- `paraphina/tests/phase51_forward_refresh_capture_tests.rs`: unit coverage for disabled, fail-closed, sanitization, target-key, role, and Lighter pressure behavior.

## Runtime Rules

The capture lane only emits a row when all required event-time evidence is already present in memory. It never infers from time, price, size, side, account role, configured caps, documentation, current snapshots, or stale snapshots.

It emits nothing when:

- `enabled=false`
- exact `canonical_group_id` plus `order_key` are absent
- venue-native maker/taker evidence is absent
- Lighter native-limit pressure evidence is incomplete

It fails closed when:

- `allow_live=true`
- capture is enabled outside a non-live/shadow-safe mode
- `append_only=false`
- the output path references `.env` content or a symlink
- `max_rows` would be exceeded
- a row contains raw identifiers, secrets, or unsafe true flags

## Output Rows

All rows include:

- `target_type`
- `venue_id`
- `canonical_group_id`
- `order_key`
- `no_live_flag=true`
- `approved_for_live=false`
- `approved_for_canary=false`
- `approved_for_model_training=false`
- `approved_for_capital_escalation=false`
- `admissible_for_financial_claim=false`
- `admissible_for_ev_admission=false`
- `live_orders_allowed=false`
- `capital_change_allowed=false`
- `risk_limit_relaxation_allowed=false`

`target_type=native_role` may include only already-observed venue-native role fields:

- Aster: `e=ORDER_TRADE_UPDATE`, `m`, `l`
- Extended: `isTaker`
- Hyperliquid: `crossed`
- Lighter: `account_index`, `is_maker_ask`, `ask_account_id`, `bid_account_id`
- Paradex: `liquidity`

`target_type=lighter_native_limit` may include only complete event-time pressure fields:

- `active_order_headroom_account`
- `active_order_sendtx_utilization_account`
- `rest_open_orders_count` and `rest_open_orders_cap`, or `weighted_open_order_slots_used` and `weighted_open_order_slots_cap`
- `native_limit_event_time_status=EVENT_TIME_ALIGNED`

## Merge Command

```bash
python3 tools/phase51ao_forward_refresh_remaining_merge.py \
  --base /home/ubuntu/source_owner_inbox/phase51/forward_refresh.jsonl \
  --remaining /home/ubuntu/source_owner_inbox/phase51/forward_refresh.remaining.jsonl \
  --output /home/ubuntu/source_owner_inbox/phase51/forward_refresh.merged.jsonl
```

The merge tool rejects duplicates, raw fields, unsafe true flags, missing `no_live_flag=true`, unsupported target types, missing target keys, and symlink paths. It writes a separate merged file and summary JSON. It does not mutate the base file unless `--replace-base` is explicitly supplied.

## Validation Sequence After Future Rows Exist

Do not run Phase 5.1 validators while `forward_refresh.remaining.jsonl` is empty. After a future non-live observation window emits real sanitized rows:

1. Run the merge tool above and inspect the summary JSON.
2. Run the existing Phase 5.1 forward-refresh validators against `forward_refresh.merged.jsonl`.
3. Run the downstream Phase 5.1 audit sequence only if the merge and validators accept the new rows.
4. Keep the blocker status on HOLD until Lighter native-limit pressure and remaining maker/taker evidence are observed, validated, and propagated.

## Observation Window Requirement

This requires a future non-live/shadow-safe observation window. Existing artifacts cannot clear the remaining blocker, and this scaffold intentionally does not backfill, infer, or reconstruct missing joins, roles, or pressure fields.
