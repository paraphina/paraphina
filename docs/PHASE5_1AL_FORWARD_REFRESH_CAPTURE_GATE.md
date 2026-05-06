# Phase 5.1al Forward-Refresh Capture Gate

Phase 5.1al is a HOLD-only repo-owned gate for the recommended forward-refresh
lane after retrospective local mining is exhausted.

It materializes a new forward-refresh target pack only from sanitized rows that
already contain deterministic target join keys and the required native source
truth. It does not infer source links, infer maker/taker role, infer Lighter
pressure, connect to venues, read env files, call `sendTx`/`sendTxBatch`, place
or cancel orders, enable live/canary behavior, escalate capital, relax risk
limits, authorize model training, authorize EV admission, or support financial
claims.

## Tool

```text
tools/phase51al_forward_refresh_capture_gate.py
tests/test_phase51al_forward_refresh_capture_gate.py
```

## Input Contract

Input is a local `.jsonl` file of sanitized forward-refresh rows.

Each row must include:

- `target_type`: `native_role` or `lighter_native_limit`.
- `venue_id`: one of `aster`, `extended`, `hyperliquid`, `lighter`, `paradex`.
- `canonical_group_id` or `order_key`.
- No raw identifier fields such as `order_id`, `clientOrderId`, `id`, `i`,
  `trade_id`, or `tx_hash`.
- No secret-shaped fields.
- No unsafe true flags.

`native_role` rows must also include the venue-native role fields accepted by
Phase 5.1v:

- Aster: maker flag `m` or `maker_side`, plus positive fill quantity `l` or
  `lastFilledQty`.
- Extended: `isTaker` or `is_taker`.
- Hyperliquid: `crossed`.
- Lighter: `account_index`, `is_maker_ask`, `ask_account_id`,
  `bid_account_id`.
- Paradex: `liquidity` as `MAKER` or `TAKER`.

`lighter_native_limit` rows must use `venue_id=lighter` and include:

- `active_order_headroom_account`.
- `active_order_headroom_market`.
- `sendtx_per_minute_limit`.
- `sendtx_per_minute_remaining`.
- REST limit/remaining or weighted request limit/remaining.
- `native_limit_event_time_status` in `EVENT_TIME_ALIGNED`,
  `SNAPSHOT_AT_DECISION_TIME`, or `OBSERVED_AT_DECISION_TIME`.

## Output Contract

Each run writes:

```text
runs/phase51al_forward_refresh_capture_gate/<RUN_ID>/
  target_run/
    native_role_capture_targets.jsonl
    lighter_native_limit_capture_targets.jsonl
    capture_bundle_manifest_template.json
    phase51u_forward_capture_target_manifest_summary.json
    phase51u_manifest.json
  source_snapshots/
    phase51al_forward_refresh_native_role_source.jsonl
    phase51al_lighter_forward_native_limit_pressure.jsonl
  source_links.proposed.sanitized.jsonl
  candidate_manifest.forward_refresh.json
  phase51al_request_pack/
    manifest.json
    source_link_request_targets.jsonl
    source_link_request_sources.jsonl
  phase51al_forward_refresh_capture_labels.jsonl
  phase51al_forward_refresh_capture_summary.json
  evidence_pack/artifact_index.json
  manifest.json
```

The `target_run` directory is Phase 5.1u-compatible. The candidate manifest is
Phase 5.1ae/5.1v-compatible. The request pack is included because Phase 5.1ak
requires a request-pack directory even when using a forward-refresh candidate
manifest directly.

## Phase 5.1ak Validation

Use Phase 5.1ak as the final wrapper:

```bash
python3 tools/phase51ak_blocker_resolution_runner.py \
  --target-run runs/phase51al_forward_refresh_capture_gate/<RUN_ID>/target_run \
  --request-pack runs/phase51al_forward_refresh_capture_gate/<RUN_ID>/phase51al_request_pack \
  --no-default-current-manifest \
  --candidate-manifest runs/phase51al_forward_refresh_capture_gate/<RUN_ID>/candidate_manifest.forward_refresh.json \
  --target-pack-mode forward-refresh \
  --run-id <PHASE51AK_FORWARD_REFRESH_RUN_ID>
```

Forward-refresh-ready targets are reported as `READY_FORWARD_REFRESH_PACK`.
Incomplete forward-refresh targets are reported as
`FORWARD_REFRESH_PACK_INCOMPLETE`.

## Fixture Evidence

The deterministic sanitized fixture run proves the gate contract only:

```text
Phase 5.1al:
runs/phase51al_forward_refresh_capture_gate/PHASE51AL-FORWARD-REFRESH-FIXTURE-HOLD-20260506T000000Z

Phase 5.1ak wrapper:
runs/phase51ak_blocker_resolution_runner/PHASE51AK-FORWARD-REFRESH-FIXTURE-HOLD-20260506T000000Z

Phase 5.1ak result:
READY_FORWARD_REFRESH_PACK=2
native-role ready: 1 / 1
Lighter native-limit pressure ready: 1 / 1
```

This fixture is not current-pack source evidence and does not clear the
retained Phase 5.1 blockers. It only proves that a future event-time
forward-refresh pack with complete source truth can be materialized and
validated through Phase 5.1ak without using inference.
