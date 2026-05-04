# Phase 5.1w Forward Capture Request Pack

Status: `HOLD` for live, canary, capital escalation, risk-limit relaxation,
model training, EV admission, financial claims, and 24/7 readiness.

Phase 5.1w is a repo-owned offline gate that turns the Phase 5.1u forward
capture target manifest into an operator-facing request pack. It exists because
Phase 5.1v can validate a supplied local bundle, but the exact bundle request
should not be inferred manually from docs and validator internals.

## Purpose

Phase 5.1w emits:

- an operator Markdown request pack;
- a machine-readable JSON request pack;
- a `capture_bundle_manifest.skeleton.json` for local path replacement;
- a summary and manifest with artifact hashes.

It performs no network access, reads no secrets, submits no orders, validates no
source data, and does not infer maker/taker role or Lighter native-limit
pressure. Phase 5.1v remains the readiness validator for any supplied bundle.

## Tool

```text
tools/phase51w_forward_capture_request_pack.py
```

Inputs:

- `--target-run`: Phase 5.1u run directory containing:
  - `phase51u_forward_capture_target_manifest_summary.json`
  - `native_role_capture_targets.jsonl`
  - `lighter_native_limit_capture_targets.jsonl`

Outputs:

- `forward_capture_request_pack.md`
- `forward_capture_request_pack.json`
- `capture_bundle_manifest.skeleton.json`
- `phase51w_forward_capture_request_pack_summary.json`
- `phase51w_manifest.json`

## Required Local Files

The canonical Phase 5.1w request pack requires six sanitized local files:

- Aster native role snapshot: `113` targets, requiring `ORDER_TRADE_UPDATE` or
  equivalent, maker flag `o.m` or `m`, and positive fill quantity.
- Extended native role snapshot: `28` targets, requiring `isTaker` or
  `is_taker`.
- Hyperliquid native role snapshot: `6` targets, requiring `crossed`.
- Lighter native role snapshot: `125` targets, requiring `account_index`,
  `is_maker_ask`, `ask_account_id`, and `bid_account_id`.
- Paradex native role snapshot: `15` targets, requiring `liquidity`.
- Lighter native-limit pressure snapshot: `3132` targets, requiring event-time
  active-order headroom, sendTx limit/remaining, REST or weighted request
  limit/remaining, and accepted native-limit event-time alignment status.

Each row must link by `canonical_group_id`, `order_key`, or a source-link
sidecar keyed by redacted source-record hash. Source-link-only manifests are
not sufficient to provide native role or Lighter native-limit evidence; they
only provide joins.

## Safety Contract

The request pack preserves the Phase 5.1 safety boundary:

- no live orders;
- no canary mode;
- no capital escalation;
- no risk-limit relaxation;
- no model training;
- no EV admission;
- no financial claims;
- no network source paths;
- no `.env` source files;
- no symlink source files;
- no secret-shaped fields;
- no raw venue identifiers;
- no maker/taker inference from intent;
- no Lighter limit inference from docs-only caps.

## Baseline Evidence

Run id:

```text
PHASE51W-FORWARD-CAPTURE-REQUEST-PACK-CANONICAL-20260504T000000Z
```

Local run directory:

```text
runs/phase51w_forward_capture_request_pack/PHASE51W-FORWARD-CAPTURE-REQUEST-PACK-CANONICAL-20260504T000000Z
```

Result:

```text
gate_status: HOLD
gate_reason: phase51w_forward_capture_request_pack_emitted_nonlive_hold
native_role_capture_target_count: 287
native_role_capture_target_counts_by_venue:
- aster: 113
- extended: 28
- hyperliquid: 6
- lighter: 125
- paradex: 15
lighter_native_limit_capture_target_count: 3132
required_local_source_file_count: 6
clears_phase51_blockers: false
raw_identifier_redaction_status: PASS
```

## Next Command

Use the skeleton manifest from the run directory, replace placeholder paths
with sanitized local `.json` or `.jsonl` files, and run:

```bash
python3 tools/phase51v_forward_capture_bundle_readiness.py \
  --target-run runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z \
  --candidate-manifest <local_capture_bundle_manifest.json> \
  --run-id <phase51v_run_id>
```

If Phase 5.1v emits `generated_phase51s_manifest_ready=true`, continue:

```bash
python3 tools/phase51s_local_native_source_acquisition.py \
  --manifest runs/phase51v_forward_capture_bundle_readiness/<phase51v_run_id>/phase51s_manifest.generated.json \
  --run-id <phase51s_run_id>
```

Then run Phase 5.1s -> 5.1r -> 5.1q -> 5.1n -> 5.1h -> 5.1i.
