# Phase 5.1q Forward Native Evidence Gate

Status: `HOLD`, non-live evidence only.

## Objective

Phase 5.1q adds a repo-owned gate for forward-captured venue-native maker/taker
role evidence and event-time native-limit pressure evidence. It is designed to
clear the current evidence blockers only when explicit native source data exists
for the same canonical P_fill group.

This gate does not authorize live orders, canary, capital escalation,
risk-limit relaxation, EV admission, model training, or financial claims.

## Venue-Native Role Sources

Allowed all-five venue role sources are:

- Lighter: `LIGHTER_TRADES_JSON`
- Hyperliquid: `HYPERLIQUID_CROSSED`
- Paradex: `PARADEX_LIQUIDITY`
- Aster: `ASTER_ORDER_TRADE_UPDATE_M`
- Extended: `EXTENDED_ISTAKER`
- Generic explicit native fields: `VENUE_NATIVE_FILL_FIELD`,
  `VENUE_NATIVE_TRADE_JOIN`, `VENUE_NATIVE_FEE_ROLE`

The gate rejects role evidence inferred from post-only intent, order purpose,
fee schedules, side, strategy intent, or expected maker/taker behavior.

## Native-Limit Pressure Scope

The current hard blocker is Lighter native-limit pressure, so Phase 5.1q marks
non-Lighter rows `NOT_APPLICABLE_NON_LIGHTER` for this specific gate. That is
not a waiver of future all-venue native-limit evidence requirements.

A Lighter row is `OBSERVED_NATIVE_LIMIT_PRESSURE` only when the source row has:

- canonical group ID alignment;
- active-order headroom at account and market level;
- sendTx limit and remaining headroom;
- REST or weighted-request limit and remaining headroom;
- event-time alignment status.

Missing any of those fields produces `PARTIAL_NATIVE_LIMIT_PRESSURE_SOURCE` or
`MISSING_NATIVE_LIMIT_PRESSURE_SOURCE`.

## Tool

Phase 5.1q expects sanitized source rows. When local source snapshots contain
raw venue identifiers, first run Phase 5.1r:

```bash
python3 tools/phase51r_forward_native_source_acquisition.py \
  --observed-pfill-run runs/<pfill_run> \
  --source-root <local_native_snapshot_dir> \
  --run-id <phase51r_run_id>
```

Command:

```bash
python3 tools/phase51q_forward_native_evidence_capture.py \
  --observed-pfill-run runs/<pfill_run> \
  --native-role-jsonl <sanitized_native_role_source.jsonl> \
  --native-limit-jsonl <sanitized_native_limit_source.jsonl> \
  --run-id <run_id>
```

Outputs:

- `phase51q_forward_native_evidence_summary.json`
- `native_role_evidence.jsonl`
- `forward_native_role_capture_labels.jsonl`
- `native_limit_pressure_labels.jsonl`
- `phase51q_manifest.json`

`native_role_evidence.jsonl` is compatible with
`tools/phase51n_maker_taker_attribution_recovery.py`.

## Safety Contract

Phase 5.1q source rows and outputs must not contain raw `order_id`,
`client_order_id`, `venue_order_id`, `fill_id`, `trade_id`, `ask_id`, `bid_id`,
`ask_client_id`, or `bid_client_id`. Use canonical group IDs, hashes, counts,
and source classifications only.

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

## Board Decision

Phase 5.1q is approved for non-live evidence capture and downstream recovery
reruns only. It cannot change strategy logic, order execution, risk controls,
venue inclusion, or capital allocation.

## Baseline Run

Initial no-source baseline:

```text
runs/phase51q_forward_native_evidence/PHASE51Q-FORWARD-NATIVE-EVIDENCE-BASELINE-NO-SOURCES-20260503T000000Z
runs/phase51n_maker_taker_attribution_recovery/PHASE51Q-MAKER-TAKER-ATTRIBUTION-RECOVERY-BASELINE-NO-SOURCES-20260503T000000Z
```

Result:

```text
gate_status: HOLD
gate_reason: phase51q_forward_native_evidence_incomplete
observed_pfill_label_count: 6140
native_role_evidence_record_count: 0
recovered_forward_native_role_count: 0
native_role_capture_status_counts:
- OBSERVED_PRESERVED: 174
- MISSING_FORWARD_NATIVE_ROLE_SOURCE: 287
- NO_FILL_NOT_APPLICABLE: 5679
native_limit_pressure_status_counts:
- MISSING_NATIVE_LIMIT_PRESSURE_SOURCE: 3132
- NOT_APPLICABLE_NON_LIGHTER: 3008
```

The baseline proves the gate is runnable against real canonical P_fill evidence
and preserves HOLD when no forward native source rows exist.

Phase 5.1r source-acquisition baseline:

```text
runs/phase51r_forward_native_source_acquisition/PHASE51R-FORWARD-NATIVE-SOURCE-ACQUISITION-BASELINE-NO-SOURCES-20260503T000000Z
native_role_target_count: 287
native_role_target_recovered_count: 0
lighter_native_limit_target_count: 3132
lighter_native_limit_target_recovered_count: 0
raw_identifier_redaction_status: PASS
```
