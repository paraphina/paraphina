# Phase 5.1af Local Source Retrieval Audit

Phase 5.1af is a HOLD-only exhaustion audit. It answers whether existing local
request packs, bounded Phase 5 telemetry artifacts, and runtime logs can produce
either missing Phase 5.1 artifact without inference:

- a validated redacted source-link mapping for Phase 5.1ad;
- sanitized Lighter event-time native-limit pressure rows for Phase 5.1ab.

It does not place orders, call venue APIs, read env files, infer joins from
time/price/size, emit raw identifiers, clear Phase 5.1 blockers, or authorize
live/canary/model-training/EV-admission promotion.

## Tool

```text
tools/phase51af_local_source_retrieval_audit.py
```

Inputs are local only:

- `--request-pack`: a Phase 5.1z source-link request pack.
- `--bounded-telemetry EXPECTED_SHA256=PATH`: historical bounded telemetry.
- `--log PATH`: runtime logs to scan for field-name patterns only.

The tool rejects network paths, env files, symlinks, unsafe true authorization
flags, and secret-shaped output fields. Runtime log scanning emits pattern
counts only, never log lines or raw values.

## Strict Verdict Semantics

Phase 5.1af must not treat weak field or log hints as blocker clearance.

Accepted statuses:

- `source_link_retrieval_status=MISSING_REQUIRED_LINKAGE`: request source rows
  do not contain enough safe target linkage to form a Phase 5.1ad mapping.
- `source_link_retrieval_status=DISCOVERY_HINT_ONLY`: linkage-like fields are
  present, but downstream Phase 5.1ad/5.1v validation is still required.
- `source_link_retrieval_status=COMPLETE_PHASE51AD_MAPPING_CANDIDATE`: local
  rows have enough safe linkage to attempt Phase 5.1ad materialization; this is
  still not blocker clearance.
- `lighter_pressure_retrieval_status=MISSING_REQUIRED_PRESSURE_FIELDS`: inspected
  local files do not contain event-time active-order/sendTx/REST-or-weighted
  pressure fields.
- `lighter_pressure_retrieval_status=DISCOVERY_HINT_ONLY`: pressure-like fields
  or log patterns exist but must be converted to complete sanitized Phase 5.1ab
  rows before Phase 5.1v can count them.
- `runtime_log_pattern_status=NO_USABLE_PRESSURE_PATTERN`: runtime logs did not
  contain sendTx/remaining/x-ratelimit/weighted pressure patterns.
- `runtime_log_pattern_status=DISCOVERY_HINT_ONLY`: runtime logs contain
  pressure-like patterns, but this is not a complete pressure row.

`clears_phase51_blockers` must remain `false`.

## Reference Evidence

```text
run: runs/phase51af_local_source_retrieval_audit/PHASE51AF-LOCAL-SOURCE-RETRIEVAL-AUDIT-HOLD-20260505T000000Z
request pack: runs/phase51z_source_link_request_pack/PHASE51Z-CURRENT-TARGET-WIDE-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z
source rows scanned: 2819
target rows scanned: 281
source rows by venue: aster=784, extended=1579, lighter=300, paradex=156
target rows by venue: aster=113, extended=28, lighter=125, paradex=15
bounded telemetry SHA256 checks: PASS
bounded telemetry Lighter pressure field counts: 0, 0
runtime log scans complete: true
runtime log pressure status: NO_USABLE_PRESSURE_PATTERN
source_link_retrieval_status: MISSING_REQUIRED_LINKAGE
lighter_pressure_retrieval_status: MISSING_REQUIRED_PRESSURE_FIELDS
local_retrieval_possible_without_inference: false
clears_phase51_blockers: false
```

## Resume Rule

Do not repeat Phase 5.1af unless local artifacts, request-pack scope, source
semantics, or runtime log inputs change materially.

The next move is still to obtain a validated redacted mapping containing only
`source_record_sha256` plus `canonical_group_id` or `order_key`, then run
Phase 5.1ad -> Phase 5.1ae -> Phase 5.1v. Separately, obtain sanitized Lighter
event-time native-limit pressure rows and run Phase 5.1ab -> Phase 5.1ae ->
Phase 5.1v.
