# Phase 5.1aj Forward Private Stream Source

Phase 5.1aj is a HOLD-only evidence tool for the remaining native-role blocker.
It converts already-local private/native JSON or JSONL rows into Phase
5.1v-compatible sanitized source rows when the row itself contains both:

- venue-native maker/taker role evidence; and
- raw order/client identifiers whose hashes uniquely match current Phase 5.1u
  targets.

It does not connect to venues, place orders, cancel orders, modify orders,
submit `sendTx`/`sendTxBatch`, enable canary/live mode, escalate capital, relax
risk limits, train models, admit EV candidates, or support financial claims.

## Tool

```text
tools/phase51aj_forward_private_stream_source.py
tests/test_phase51aj_forward_private_stream_source.py
```

Input contract:

```text
--request-pack <Phase 5.1z source-link request pack>
--source-json <venue>=<local .json/.jsonl/.ndjson artifact>
```

Supported venues:

```text
aster
extended
lighter
paradex
```

Output contract:

```text
source_snapshots/phase51aj_forward_private_stream_source_rows.jsonl
source_links.proposed.sanitized.jsonl
phase51aj_candidate_manifest.json
phase51aj_forward_private_stream_source_labels.jsonl
phase51aj_forward_private_stream_source_summary.json
evidence_pack/artifact_index.json
```

The emitted candidate manifest is intended for Phase 5.1ae composition and
Phase 5.1v readiness validation.

## Evidence Rule

Phase 5.1aj may emit a target-linked source row only when all conditions hold:

- the input row has required native role fields for the venue;
- at least one raw or prehashed client/order identifier on that same row
  uniquely matches a current request-pack target hash;
- the output row persists only sanitized identifiers, target keys, role fields,
  source hashes, and no-live guard flags.

Phase 5.1aj may emit a source-link sidecar only when the same row's
`source_record_sha256` is already present in the request pack. Otherwise the
row can still be emitted as a direct target-linked source row, but not as a
source-link sidecar.

Forbidden joins:

- time proximity;
- price/size/account-role collisions;
- current account snapshots standing in for event-time evidence;
- documentation-only limits;
- order-history rows that do not share deterministic source identity;
- raw identifier persistence.

## Current Evidence

The first local recheck is:

```text
runs/phase51aj_forward_private_stream_source/PHASE51AJ-LOCAL-SANITIZED-SOURCE-RECHECK-HOLD-20260506T000000Z
```

Inputs:

```text
request pack:
runs/phase51z_source_link_request_pack/PHASE51Z-CURRENT-TARGET-WIDE-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z

extended source:
runs/phase51ai_non_lighter_order_history_bridge_diagnostic/PHASE51AI-NON-LIGHTER-ORDER-HISTORY-BRIDGE-DIAGNOSTIC-HOLD-20260506T000000Z/source_snapshots/extended_native_source_rows.sanitized.json

paradex source:
runs/phase51ai_non_lighter_order_history_bridge_diagnostic/PHASE51AI-NON-LIGHTER-ORDER-HISTORY-BRIDGE-DIAGNOSTIC-HOLD-20260506T000000Z/source_snapshots/paradex_native_source_rows.sanitized.json

lighter source:
runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-READONLY-BLOCKER-RECHECK-HOLD-20260506T000000Z/source_snapshots/trades.sanitized.json
```

Results:

```text
raw_row_count: 1731
direct_target_linked_source_row_count: 28
direct_target_linked_source_row_count_by_venue: extended=21, paradex=7
request_source_hash_overlap_count: 0
source_link_count: 0
lighter deterministic target overlap: 0
gate_status: HOLD
```

The Phase 5.1ae/5.1v composition is:

```text
runs/phase51ae_candidate_manifest_compose/PHASE51AE-BLOCKER-RECHECK-PLUS-PHASE51AJ-HOLD-20260506T000000Z
runs/phase51v_forward_capture_bundle_readiness/PHASE51V-BLOCKER-RECHECK-PLUS-PHASE51AJ-HOLD-20260506T000000Z
```

Phase 5.1v result:

```text
native-role targets ready: 73 / 287
native-role targets missing: 214 / 287
Lighter native-limit targets ready: 0 / 3132
```

Interpretation: Phase 5.1aj is valid forward infrastructure and recovered
`28` deterministic source rows from existing sanitized Extended/Paradex
artifacts, but those rows overlap target IDs already counted ready by existing
Phase 5.1z evidence. It does not reduce the remaining blocker. It should now be
used only when materially new directly target-linkable private/native rows are
captured.

## Resume Instruction

To use Phase 5.1aj on new evidence:

```bash
python3 tools/phase51aj_forward_private_stream_source.py \
  --request-pack runs/phase51z_source_link_request_pack/PHASE51Z-CURRENT-TARGET-WIDE-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z \
  --source-json extended=<local_extended_private_rows.jsonl> \
  --source-json paradex=<local_paradex_private_rows.jsonl> \
  --source-json aster=<local_aster_private_rows.jsonl> \
  --source-json lighter=<local_lighter_private_rows.jsonl>
```

Then compose and validate:

```bash
python3 tools/phase51ae_candidate_manifest_compose.py \
  --candidate-manifest <existing_current_candidate_manifest> \
  --candidate-manifest <phase51aj_run>/phase51aj_candidate_manifest.json

python3 tools/phase51v_forward_capture_bundle_readiness.py \
  --target-run runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-LINK-HYGIENE-20260505T000000Z \
  --candidate-manifest <phase51ae_run>/candidate_manifest.composed.json
```

Do not treat Phase 5.1aj output as economic evidence or live-readiness
evidence. It is native-role source readiness evidence only.
