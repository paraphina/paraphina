# Phase 5.1r Forward Native Source Acquisition

Status: `HOLD`, non-live evidence only.

## Objective

Phase 5.1r is the repo-owned adapter that turns local, read-only venue-native
snapshots into the sanitized JSONL source rows consumed by Phase 5.1q. It exists
because Phase 5.1q deliberately rejects raw venue identifiers; 5.1r may ingest
quarantined raw snapshots, hash their source records, and emit only
canonical-group/count/hash evidence.

This gate does not authorize live orders, canary, capital escalation,
risk-limit relaxation, EV admission, model training, or financial claims.

## Venue Role Mapping

The current source mappings are explicit venue-native fields only:

- Lighter: `account_index` plus `is_maker_ask`, `ask_account_id`, and
  `bid_account_id`.
- Hyperliquid: fill `crossed`; `true` maps to taker, `false` maps to maker.
- Paradex: fill `liquidity`.
- Aster: `ORDER_TRADE_UPDATE` payload field `o.m` when a positive last-filled
  quantity is present.
- Extended: trade `isTaker`.
- Generic exact source rows: `native_role` or `native_liquidity_role` when the
  field is itself venue-native and not inferred from strategy intent.

Source checks on 2026-05-03 used official venue docs for the field names:
Hyperliquid Info endpoint, Paradex List Fills, Aster order-update stream,
Extended Get Trades/API docs, and Lighter WebSocket/API docs.

## Tool

Command:

```bash
python3 tools/phase51r_forward_native_source_acquisition.py \
  --observed-pfill-run runs/<canonical_pfill_run> \
  --source-root <local_native_snapshot_dir> \
  --source-json <local_native_snapshot.jsonl> \
  --run-id <run_id>
```

`--source-root` and `--source-json` may be repeated. Source roots are scanned
for `.json` and `.jsonl` files.

Outputs:

- `phase51r_forward_native_source_acquisition_summary.json`
- `native_role_source.jsonl`
- `native_limit_source.jsonl`
- `source_acquisition_labels.jsonl`
- `phase51r_manifest.json`

`native_role_source.jsonl` and `native_limit_source.jsonl` are the only outputs
intended for Phase 5.1q.

## Safety Contract

Phase 5.1r output must not contain raw `order_id`, `client_order_id`,
`venue_order_id`, `raw_order_id`, `raw_client_order_id`, `ask_id`, `bid_id`,
`ask_client_id`, `bid_client_id`, `trade_id`, `fill_id`, `id`, `oid`, `cloid`,
or `tid`. Raw source snapshots may contain those fields only as quarantined
local input; emitted records carry hashes and canonical IDs instead.

All safety booleans must remain false:

- `approved_for_model_training`
- `approved_for_live`
- `approved_for_canary`
- `approved_for_capital_escalation`
- `admissible_for_financial_claim`
- `admissible_for_ev_admission`
- `live_orders_allowed`
- `capital_change_allowed`
- `risk_limit_relaxation_allowed`

Any input source row with an unsafe true flag is rejected.

## Baseline Run

Initial no-source baseline:

```text
runs/phase51r_forward_native_source_acquisition/PHASE51R-FORWARD-NATIVE-SOURCE-ACQUISITION-BASELINE-NO-SOURCES-20260503T000000Z
runs/phase51q_forward_native_evidence/PHASE51R-FORWARD-NATIVE-EVIDENCE-BASELINE-NO-SOURCES-20260503T000000Z
runs/phase51n_maker_taker_attribution_recovery/PHASE51R-MAKER-TAKER-ATTRIBUTION-RECOVERY-BASELINE-NO-SOURCES-20260503T000000Z
runs/phase51h_observed_pfill_feature_audit/PHASE51R-OBSERVED-PFILL-FEATURE-AUDIT-BASELINE-NO-SOURCES-20260503T000000Z
runs/phase51i_pfill_feature_matrix_admissibility/PHASE51R-PFILL-FEATURE-MATRIX-ADMISSIBILITY-BASELINE-NO-SOURCES-20260503T000000Z
```

Phase 5.1r result:

```text
gate_status: HOLD
gate_reason: phase51r_forward_native_source_acquisition_incomplete
observed_pfill_label_count: 6140
native_role_target_count: 287
native_role_source_record_count: 0
native_role_target_recovered_count: 0
lighter_native_limit_target_count: 3132
native_limit_source_record_count: 0
native_limit_complete_source_record_count: 0
lighter_native_limit_target_recovered_count: 0
raw_identifier_redaction_status: PASS
```

Downstream Phase 5.1q/5.1n result remains unchanged:

```text
recovered_forward_native_role_count: 0
native_role_capture_status_counts:
- OBSERVED_PRESERVED: 174
- MISSING_FORWARD_NATIVE_ROLE_SOURCE: 287
- NO_FILL_NOT_APPLICABLE: 5679
native_limit_pressure_status_counts:
- MISSING_NATIVE_LIMIT_PRESSURE_SOURCE: 3132
- NOT_APPLICABLE_NON_LIGHTER: 3008
maker_taker_observed_or_recovered_count: 174
maker_taker_partial_or_missing_count: 287
```

Recovered 5.1h/5.1i remains `HOLD` with the same four blockers:

- `lighter_native_limit_pressure_not_fully_observed`
- `maker_taker_not_fully_observed_for_filled_orders`
- `sparse_pfill_feature_buckets`
- `observed_only_selection_bias_not_resolved`

The baseline proves the acquisition layer is runnable and redaction-safe, but
clears no blocker until real source snapshots are supplied.
