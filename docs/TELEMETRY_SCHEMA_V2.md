# Telemetry Schema v2

Telemetry schema v2 is the Phase 5.1 non-live EV/replay/shadow telemetry
contract. It is additive to v1 and does not change existing v1 producers.

The validator selects the schema for `telemetry.jsonl` from the first record's
`schema_version`:

- `schema_version: 1` -> `schemas/telemetry_schema_v1.json`
- `schema_version: 2` -> `schemas/telemetry_schema_v2.json`

Mixed-version files are invalid. Unknown schema versions are invalid.

## Required Envelope

Every v2 event requires:

- `schema_version`
- `event_type`
- `event_seq`
- `timestamp_local_ns`
- `run_id`
- `baseline_commit`
- `no_live_flag`

`event_seq` must be monotonically increasing.

## Event Families

The v2 contract recognizes the Phase 5.1 event families approved by the EV
revision addenda:

- `V2_RUN_CONTEXT`
- `V2_MARKET_SNAPSHOT`
- `V2_PAIR_EDGE_SNAPSHOT`
- `V2_EV_EVALUATED`
- `V2_ORDER_INTENT`
- `V2_ORDER_LIFECYCLE`
- `V2_FILL_OBSERVED`
- `V2_FAST_HEDGE_DECISION`
- `V2_HEDGE_LIFECYCLE`
- `V2_INVENTORY_SNAPSHOT`
- `V2_BALANCE_SNAPSHOT`
- `V2_REPLAY_LABEL`
- `V2_GUARDRAIL_EVENT`

The Phase 5.1 shadow harness currently emits source-linked
`V2_EV_EVALUATED` and `V2_REPLAY_LABEL` records. These records include
`source_t`, `source_line`, `source_record_sha256`, `model_features_hash`,
`calibration_bucket_id`, and `binding_constraints` so candidates can be traced
back to the v1 telemetry line that produced them.

All Phase 5.1 shadow candidates remain `HOLD` until calibration evidence exists.
`V2_REPLAY_LABEL` records are counterfactual decision labels, not realized fills
or economic proof.

Event-family-specific required fields can be tightened further once emitters are
producing stable artifacts for order lifecycle, fill, hedge, inventory, and
balance events.

## Determinism

Phase 5.1 non-live runs should provide a stable `run_id` and input telemetry
snapshot. The shadow harness derives replay timestamps from the run id and input
hash unless `--replay-timestamp-ns` is provided, and writes a root
`manifest.json` covering the run artifacts.

## Validation

```bash
python3 tools/check_telemetry_contract.py path/to/telemetry.jsonl
python3 tools/check_telemetry_contract.py path/to/run_directory/
```

Existing v1 telemetry remains valid and should continue to emit
`schema_version: 1` until a producer intentionally migrates to v2.
