# Phase 5.1ak Blocker Resolution Runner

Phase 5.1ak is a HOLD-only repo-owned runner for the current Phase 5.1 blocker.
It does not create evidence by inference. It composes accepted local evidence
artifacts, runs Phase 5.1v, and emits a target-level decision artifact that
distinguishes recovered current-pack targets from targets that still require a
validated mapping or forward-refresh evidence.

## Scope

Phase 5.1ak may:

- run Phase 5.1aj on already-local directly target-linkable private/native rows;
- run Phase 5.1ab on already-sanitized Lighter event-time pressure rows;
- compose candidate manifests through Phase 5.1ae;
- validate readiness through Phase 5.1v;
- emit `phase51ak_blocker_target_decisions.jsonl` and
  `phase51ak_blocker_resolution_summary.json`.

Phase 5.1ak must not:

- connect to venues;
- read env files;
- place, cancel, modify, replace, or submit orders;
- call `sendTx`, `sendTxBatch`, or any write endpoint;
- infer source links from time, price, size, account role, or proximity;
- infer Lighter pressure from docs-only limits, current snapshots, account
  tiers, empty headers, or runtime config strings;
- authorize model training, EV admission, canary, live trading, capital
  escalation, risk-limit relaxation, or financial claims.

## Current-Pack Recovery

Default current-pack run:

```bash
python3 tools/phase51ak_blocker_resolution_runner.py \
  --run-id PHASE51AK-CURRENT-BLOCKER-RESOLUTION-HOLD-20260506T000000Z \
  --timestamp-ns 1778025600000000000
```

The runner defaults to the current Phase 5.1u target run, the current-target
wide Phase 5.1z request pack, and the latest composed manifest containing
already-ready Hyperliquid plus Phase 5.1z/5.1aj evidence. Additional manifests
or new source inputs can be supplied explicitly:

```bash
python3 tools/phase51ak_blocker_resolution_runner.py \
  --candidate-manifest <phase51ad_or_other_candidate_manifest.json> \
  --phase51aj-source-json <venue>=<local_private_or_native_source.jsonl> \
  --phase51ab-pressure-jsonl <sanitized_lighter_pressure_rows.jsonl>
```

Use `--no-default-current-manifest` only for tests or for a board-documented
forward-refresh target pack.

## Decision Semantics

Each target receives one of these statuses:

- `RECOVERED_CURRENT_PACK`: the final Phase 5.1v validation found a source row
  that satisfies the target.
- `UNRECOVERABLE_FROM_LOCAL_ARTIFACTS`: the supplied local artifacts did not
  satisfy the target, so the next action is `FORWARD_REFRESH_REQUIRED` unless a
  validated redacted current-pack mapping is obtained.

Phase 5.1ak remains `HOLD` even when Phase 5.1v reports
`downstream_chain_ready=true`; downstream promotion still requires the normal
Phase 5.1s -> 5.1r -> 5.1q -> 5.1n -> 5.1h -> 5.1i non-live ladder and board
review.

## Latest Evidence

```text
run: runs/phase51ak_blocker_resolution_runner/PHASE51AK-CURRENT-BLOCKER-RESOLUTION-HOLD-20260506T000000Z
gate_reason: phase51ak_current_pack_incomplete_forward_refresh_required_nonlive_hold
native-role ready: 73 / 287
native-role missing: 214 / 287
native-role missing by venue: aster=74, extended=7, lighter=125, paradex=8
Lighter native-limit pressure ready: 0 / 3132
decision counts: RECOVERED_CURRENT_PACK=73, UNRECOVERABLE_FROM_LOCAL_ARTIFACTS=3346
next required action: obtain_validated_mapping_or_forward_refresh_target_pack_with_event_time_sources
```
