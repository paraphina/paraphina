# Phase 5.1t Source-Link Sidecar Builder

Status: `HOLD`, non-live source-link generation only.

## Objective

Phase 5.1t builds redacted source-link sidecars for Phase 5.1s when local
venue-native source snapshots contain raw order/client identifiers that cannot
be emitted downstream. It reads local `.json` / `.jsonl` source snapshots,
matches only by existing redacted `order_id_hash` / `client_order_id_hash`
values from observed P-fill labels, and emits `source_links.sanitized.jsonl`.

This gate does not authorize live orders, canary, capital escalation,
risk-limit relaxation, EV admission, model training, or financial claims.

## Tool

Command:

```bash
python3 tools/phase51t_source_link_sidecar_builder.py \
  --observed-pfill-run runs/<observed_or_canonical_pfill_run> \
  --source-root runs/<local_native_source>/source_snapshots \
  --output-root runs/phase51t_source_link_sidecar_builder \
  --run-id <phase51t_run_id>
```

Outputs:

- `phase51t_source_link_sidecar_builder_summary.json`
- `source_links.sanitized.jsonl`
- `source_link_builder_labels.jsonl`
- `manifest.json`
- `evidence_pack/artifact_index.json`

The intended downstream path is:

```bash
python3 tools/phase51s_local_native_source_acquisition.py \
  --manifest <manifest_with_source_links_sanitized_jsonl>

python3 tools/phase51r_forward_native_source_acquisition.py \
  --observed-pfill-run runs/<observed_or_canonical_pfill_run> \
  --source-json runs/phase51s_local_native_source_acquisition/<run>/local_native_source.jsonl \
  --source-link-jsonl runs/phase51s_local_native_source_acquisition/<run>/local_source_link_sidecar.jsonl
```

## Safety Contract

Phase 5.1t rejects:

- Network paths and `.env` paths.
- Symlinked source paths.
- Secret-shaped fields such as API keys, private keys, JWTs, bearer tokens, or
  passwords.
- Unsafe true authorization flags for live, canary, capital, risk-limit,
  model-training, EV-admission, or financial-claim use.

Phase 5.1t output contains only source-record hashes, `canonical_group_id` /
`order_key` joins, safety flags set to false, and diagnostic labels. It does
not emit raw order/client/fill/trade identifiers.

Negative safety markers such as `not_ev_admission_authorization` are allowed
because they are explicit non-authorization evidence, not credentials.

## Current Evidence

First existing-source run:

```text
runs/phase51t_source_link_sidecar_builder/PHASE51T-SOURCE-LINK-SIDECAR-BUILDER-EXISTING-LIGHTER-SOURCES-20260503T000000Z
gate_status: HOLD
gate_reason: phase51t_source_link_sidecar_builder_complete_nonlive_hold
source_file_count: 31
source_row_count: 1522
source_link_record_count: 363
ambiguous_identity_hash_target_count: 48
source_link_status_counts:
- SOURCE_LINK_EMITTED: 363
- DUPLICATE_SOURCE_HASH_ALREADY_EMITTED: 546
- NO_OBSERVED_IDENTITY_MATCH: 570
- NO_ORDER_IDENTITY_HASH: 17
- AMBIGUOUS_OBSERVED_IDENTITY_MATCH: 26
clears_phase51_blockers: false
```

Downstream result:

```text
Phase 5.1s staged source rows: 1522
Phase 5.1s staged source-link rows: 363
Phase 5.1r source-link applied count: 909
Phase 5.1r native-role source records: 296
Phase 5.1q recovered native-role targets: 0 / 287
Phase 5.1q recovered Lighter native-limit targets: 0 / 3132
Phase 5.1i status: HOLD
Phase 5.1i blockers:
- lighter_native_limit_pressure_not_fully_observed
- maker_taker_not_fully_observed_for_filled_orders
- sparse_pfill_feature_buckets
- observed_only_selection_bias_not_resolved
```

Interpretation: Phase 5.1t is runnable and redaction-safe, but existing local
artifacts still do not contain the exact forward all-five venue-native role
and complete Lighter native-limit evidence needed for calibrated EV review.
