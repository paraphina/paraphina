# Phase 5.1u Forward Capture Target Manifest

Status: `HOLD`, non-live target manifest only.

## Objective

Phase 5.1u emits the exact redacted target set required for the next
forward-native source capture pilot. It consumes canonical observed P-fill
labels and writes:

- all-five venue-native maker/taker role capture targets for filled groups that
  still lack complete native role evidence;
- Lighter event-time native-limit pressure targets for every Lighter P-fill
  label;
- a local capture bundle manifest template for Phase 5.1s staging.

This gate does not capture source truth by itself and does not clear Phase 5.1
blockers. It is a planning/preflight artifact for the read-only forward capture
pilot.

## Tool

Command:

```bash
python3 tools/phase51u_forward_capture_target_manifest.py \
  --observed-pfill-run runs/<canonical_pfill_run> \
  --run-id <phase51u_run_id>
```

Outputs:

- `phase51u_forward_capture_target_manifest_summary.json`
- `native_role_capture_targets.jsonl`
- `lighter_native_limit_capture_targets.jsonl`
- `capture_bundle_manifest_template.json`
- `phase51u_manifest.json`

## Capture Contract

Every forward source row must include one join path:

- `canonical_group_id`;
- `order_key`;
- Phase 5.1t/5.1s source-link sidecar keyed by a redacted source-record hash.

Venue-native role fields:

- Lighter: `account_index`, `is_maker_ask`, `ask_account_id`,
  `bid_account_id`.
- Hyperliquid: `crossed`.
- Paradex: `liquidity`.
- Aster: `ORDER_TRADE_UPDATE` or equivalent envelope, `o.m`/`m`, and positive
  `o.l` or `lastFilledQty`.
- Extended: `isTaker` or `is_taker`.

Lighter native-limit rows must include:

- `active_order_headroom_account`;
- `active_order_headroom_market`;
- `sendtx_per_minute_limit`;
- `sendtx_per_minute_remaining`;
- REST or weighted request limit and remaining fields;
- `native_limit_event_time_status` in `EVENT_TIME_ALIGNED`,
  `SNAPSHOT_AT_DECISION_TIME`, or `OBSERVED_AT_DECISION_TIME`.

## Safety Contract

Phase 5.1u rejects unsafe true authorization flags and raw identifier fields in
canonical P-fill labels. Its outputs carry only canonical groups, order keys,
hashes, target counts, field requirements, and false safety flags.

Prohibited:

- live orders;
- canary/live enablement;
- model training;
- EV admission;
- capital escalation;
- risk-limit relaxation;
- financial/economic claims;
- maker/taker role inference from post-only intent, side, fee schedule, or
  strategy purpose;
- treating official Lighter caps or current snapshots without event-time
  alignment as native-limit pressure.

## Baseline Run

Run:

```text
runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z
```

Input:

```text
runs/phase51i_redacted_canonical_pfill_outcome/PHASE51I-REDACTED-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z
```

Result:

```text
gate_status: HOLD
native_role_capture_target_count: 287
native_role_capture_target_counts_by_venue:
- aster: 113
- extended: 28
- hyperliquid: 6
- lighter: 125
- paradex: 15
lighter_native_limit_capture_target_count: 3132
clears_phase51_blockers: false
raw_identifier_redaction_status: PASS
```

The next move is to capture fresh read-only source bundles satisfying this
target manifest, stage them through Phase 5.1s, then rerun
Phase 5.1r -> 5.1q -> 5.1n -> 5.1h -> 5.1i.
