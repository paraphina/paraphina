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

This initial schema validates the common envelope and common EV evaluation
fields. Event-family-specific required fields can be tightened once emitters
are producing stable artifacts.

## Validation

```bash
python3 tools/check_telemetry_contract.py path/to/telemetry.jsonl
python3 tools/check_telemetry_contract.py path/to/run_directory/
```

Existing v1 telemetry remains valid and should continue to emit
`schema_version: 1` until a producer intentionally migrates to v2.
