# Phase 5.1v Forward Capture Bundle Readiness

Status: `HOLD` for live, canary, capital escalation, risk-limit relaxation,
model training, EV admission, financial claims, and 24/7 readiness.

Phase 5.1v is a repo-owned offline gate between the Phase 5.1u forward capture
target manifest and the Phase 5.1s local source-staging gate.

## Purpose

Phase 5.1u emits the exact target set still missing from the canonical P-fill
matrix:

- `287` all-five venue-native maker/taker role targets
- `3132` Lighter event-time native-limit pressure targets

Phase 5.1v checks whether a local sanitized capture bundle is structurally
ready to feed Phase 5.1s. It does not capture venue truth, does not call venue
APIs, does not read secrets, does not infer maker/taker role, and does not
clear blockers by itself.

## Tool

```text
tools/phase51v_forward_capture_bundle_readiness.py
```

Inputs:

- `--target-run`: Phase 5.1u run directory containing:
  - `phase51u_forward_capture_target_manifest_summary.json`
  - `native_role_capture_targets.jsonl`
  - `lighter_native_limit_capture_targets.jsonl`
- `--candidate-manifest`: local capture-bundle manifest, normally derived
  from the Phase 5.1u `capture_bundle_manifest_template.json`

Outputs:

- `capture_bundle_readiness_labels.jsonl`
- `missing_native_role_capture_targets.jsonl`
- `missing_lighter_native_limit_capture_targets.jsonl`
- `phase51s_manifest.generated.json`
- `phase51v_forward_capture_bundle_readiness_summary.json`
- `phase51v_manifest.json`

## Safety Contract

The gate rejects:

- network paths
- `.env` files
- symlinks
- unsupported source suffixes
- secret-shaped fields
- unsafe true authorization flags
- nested raw venue identifier fields in source rows

The gate permits placeholder paths but keeps them in `HOLD` as
`PLACEHOLDER_PATH`. Placeholder manifests must never be treated as evidence
that any source-capture blocker is reduced.

## Readiness Semantics

Native role rows are counted ready only when a local source row links to a
Phase 5.1u target by `canonical_group_id`, `order_key`, or a validated
source-link sidecar hash, matches the target venue, and contains the required
venue-native role field:

- Aster: `ORDER_TRADE_UPDATE` or equivalent maker-side field plus positive
  fill quantity
- Extended: `isTaker` or `is_taker`
- Hyperliquid: `crossed`
- Lighter: `account_index`, `is_maker_ask` or `isMakerAsk`,
  `ask_account_id` or `askAccountId`, and `bid_account_id` or `bidAccountId`
- Paradex: `liquidity`

Lighter native-limit rows are counted ready only when a local Lighter source
row links to a Phase 5.1u target by direct join key or validated source-link
sidecar and contains event-time active-order headroom, sendTx remaining/limit,
REST or weighted request remaining/limit, and an accepted event-time alignment
status.

Source-link sidecars are join aids only. A sidecar must map a deterministic
source-row hash to a Phase 5.1u target using `canonical_group_id` or
`order_key`, must contain no raw venue identifiers or secrets, and duplicate
source hashes are rejected. A source-link-only manifest never clears readiness:
the linked source row must also be present in a local source file and must
carry the required venue-native role or Lighter native-limit fields.

When all targets are ready, 5.1v emits a `phase51s_manifest.generated.json`
that can be used as the next Phase 5.1s manifest. Even then the verdict remains
non-live `HOLD`; the downstream chain must still run Phase 5.1s -> 5.1r ->
5.1q -> 5.1n -> 5.1h -> 5.1i before any blocker reduction can be claimed.

## Baseline Evidence

Run id:

```text
PHASE51V-FORWARD-CAPTURE-BUNDLE-READINESS-TEMPLATE-20260503T000000Z
```

Input target run:

```text
runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z
```

Candidate manifest:

```text
runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z/capture_bundle_manifest_template.json
```

Result:

```text
gate_status: HOLD
gate_reason: phase51v_forward_capture_bundle_incomplete_nonlive_hold
native_role_capture_target_ready_count: 0 / 287
lighter_native_limit_capture_target_ready_count: 0 / 3132
source_file_status_counts:
- PLACEHOLDER_PATH: 6
source_link_file_status_counts:
- PLACEHOLDER_PATH: 1
generated_phase51s_manifest_ready: false
downstream_chain_ready: false
clears_phase51_blockers: false
raw_identifier_redaction_status: PASS
```

## Source-Link Readiness Evidence

The repo-owned Phase 5.1v validator now applies validated source-link sidecars
during bundle readiness. Focused non-live tests prove:

```text
source row + valid source-link sidecar: can mark the linked target ready
source-link sidecar only: remains HOLD and marks no target ready
duplicate source-link hash: rejected
```

Validation command:

```bash
python3 -m unittest \
  tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51v_forward_capture_bundle_readiness_accepts_local_bundle \
  tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51v_forward_capture_bundle_readiness_applies_source_link_sidecar \
  tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51v_forward_capture_bundle_readiness_source_link_only_holds \
  tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51v_forward_capture_bundle_readiness_rejects_duplicate_source_link_hash \
  tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51v_forward_capture_bundle_readiness_holds_template_placeholders \
  tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51v_forward_capture_bundle_readiness_rejects_unsafe_manifest
```

## Next Command

After a real sanitized local read-only capture bundle exists:

```bash
python3 tools/phase51v_forward_capture_bundle_readiness.py \
  --target-run runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z \
  --candidate-manifest <local_capture_bundle_manifest.json> \
  --run-id <phase51v_run_id>
```

If `generated_phase51s_manifest_ready` is `true`, run:

```bash
python3 tools/phase51s_local_native_source_acquisition.py \
  --manifest runs/phase51v_forward_capture_bundle_readiness/<phase51v_run_id>/phase51s_manifest.generated.json \
  --run-id <phase51s_run_id>
```

Then continue Phase 5.1s -> 5.1r -> 5.1q -> 5.1n -> 5.1h -> 5.1i.
