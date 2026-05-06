# Phase 5.1 Board Decision

Date: 2026-05-01

Decision scope: Phase 5.1 non-live EV shadow/replay scaffold, evidence
boundary, Lighter venue-readiness evidence, and safety invariants.

This decision does not authorize live orders, canary promotion, capital
escalation, risk-limit relaxation, or economic claims.

## Decision

Board decision: `PROMOTE_FOR_NEXT_NONLIVE_STEP`

Promotion is limited to the Phase 5.1 non-live evidence scaffold. The repo now
has a reproducible Lighter-only EV shadow harness that can process bounded real
Phase 5 telemetry, emit schema v2 records, preserve no-live/no-capital/no-risk
guards, and document why all candidate EV decisions remain held.

Current EV admission decision: `HOLD`

Current live/canary decision: `HOLD`

Current economic/profitability decision: `HOLD`

## Current Superseding Status

As of the Phase 5.1l filled-horizon source-key recovery gate, the board decision remains
`PROMOTE_FOR_NEXT_NONLIVE_STEP` only. The current blockers are now more
specific:

- Phase 5.1d showed `8611 / 11935` P-fill labels were censored as
  `NO_TERMINAL_EVENT_WITH_SUFFICIENT_WINDOW`.
- Phase 5.1e shows that censoring is primarily a lifecycle
  canonicalization issue, not a true absence of terminal evidence. After
  grouping place intent/ack aliases and order/client-id aliases, the `8611`
  censored labels decompose into `464` canonical filled review rows, `5206`
  canonical not-filled review rows, `288` replace-chain review rows, `2270`
  duplicate place-alias review rows, `375` cancel-all scope review rows, and
  only `8` rows that remain `REMAINS_NO_TERMINAL_EVENT_WITH_SUFFICIENT_WINDOW`.
- Lighter native attribution remains `HOLD`: raw captured telemetry IDs match
  native trades for `96 / 189` Lighter fills (`64` maker, `32` taker), while
  `93` remain `NATIVE_WINDOW_COVERED_NO_MATCH`.
- Phase 5.1f rebuilds one P-fill label per canonical lifecycle group:
  `11935` source rows collapse to `6140` canonical P-fill groups, with `461`
  filled, `4066` terminal not-filled, and `1613` quarantined/censored review
  groups. The canonical readiness rerun remains `HOLD` because
  `1613 / 6140` labels remain censored or review-quarantined.
- Phase 5.1g preserves all `6140` canonical groups, emits an observed-only
  diagnostic pack with `4527` terminal labels (`461` filled, `4066`
  terminal not-filled), and excludes `1613` quarantine/review groups from
  binary P-fill calibration rather than treating them as not-filled outcomes.
- The Phase 5.1g observed-only readiness rerun remains `HOLD` with reason
  `pfill_calibration_sparse_buckets`: it removes censored labels from the
  diagnostic pack but does not satisfy feature-rich model, venue/side bucket,
  maker/taker, or board-review requirements.
- Phase 5.1h audits the `4527` observed-only labels for feature readiness.
  All labels reconcile to queue/churn and markout source context, but the gate
  remains `HOLD`: `4389` inherited raw decision IDs are present in input
  artifacts but not emitted, `4509` labels lack observed horizon timing,
  Lighter native limits are only partial for `2288` labels, and filled-order
  maker/taker status is incomplete for `287` labels.
- Phase 5.1i rebuilds the same feature-matrix path from redacted inputs:
  `raw_identifier_input_present_count=0` and
  `raw_identifier_redaction_status=PASS`, but the matrix remains `HOLD`
  because `4509` observed horizons are still missing.
- Phase 5.1j recovers deterministic terminal source-tick horizons for `4048`
  canonical terminal-not-filled labels, increasing observed horizon coverage
  from `18 / 4527` to `4066 / 4527`. It deliberately leaves `461`
  filled-order horizons unresolved rather than fabricating timing from
  incompatible fill-time fields.
- Phase 5.1k recovers source-tick horizons for `396 / 461` filled-order rows,
  increasing observed horizon coverage from `4066 / 4527` to `4462 / 4527`.
  It leaves `65` filled-order rows as `MISSING_JOIN`, records
  `exchange_ms_only_count=0`, and does not write exchange-millisecond timing
  into source-tick horizon fields.
- Phase 5.1l recovers the remaining `65 / 65` filled-order source-tick
  horizons with source-key/hash evidence, increasing observed horizon coverage
  from `4462 / 4527` to `4527 / 4527`. The matrix still remains `HOLD`,
  now on Lighter native-limit completeness, maker/taker completeness, sparse
  buckets, and observed-only selection bias.
- 5.1 remains `HOLD` for model training, EV admission, live orders, canary,
  capital escalation, risk-limit relaxation, and financial claims until
  observed-only calibration policy, feature completeness, venue-native truth
  gaps, and downstream calibration readiness are accepted.

## Baseline

| Item | Value |
|---|---|
| Phase 5 closeout baseline | `18dd09512288a85e440d3977e32432c3aabc1190` |
| M5 Lighter readiness commit | `705f79c6c647dbb03c8e2a993c6e23b870b4af6c` |
| M6 safety invariant commit | `02f63e30e6d1c877ab11c43d591bbda3f5c28974` |
| Branch | `main` |
| Evidence run | `LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_m6` |

## Accepted

The following Phase 5.1 items are accepted as complete for non-live scaffold
purposes:

| Area | Verdict | Evidence |
|---|---|---|
| EV shadow harness | PASS | `tools/phase51_ev_shadow.py` emits source-linked `V2_EV_EVALUATED` and `V2_REPLAY_LABEL` records from real Phase 5 telemetry. |
| Telemetry schema v2 | PASS | `schemas/telemetry_schema_v2.json` validates v2 events and enforces exact no-live authorization fields. |
| Safety invariants | PASS | Unsafe v2 authorization values and unsafe spec drift fail closed. |
| Lighter venue readiness evidence | PASS for non-live | `docs/PHASE5_1_LIGHTER_VENUE_READINESS.md` documents official-source and connector evidence with explicit flags. |
| Evidence manifesting | PASS | M6 artifacts include deterministic hashes for telemetry, manifest, and artifact index. |
| GitHub baseline | PASS | M5 and M6 commits are pushed to `origin/mm-pnl-harness-clean`. |

## Held

The following items remain explicitly held:

| Area | Hold reason |
|---|---|
| EV admission | All 2,000 M6 candidates remain `HOLD`; no candidate is admitted. |
| P-fill calibration | `missing_pfill_calibration` on all candidates. |
| Markout/adverse-selection calibration | `missing_markout_calibration` on all candidates. |
| Hedge-success calibration | `missing_hedge_success_calibration` on all candidates. |
| Queue/churn/tail costs | Queue reset, churn, and tail-risk calibration are missing. |
| Lighter account-state assumptions | Account tier, fee tier, native limits, and replace post-only preservation remain unresolved for promotion beyond non-live. |
| Pair-conditioned behavior | Fast hedge, residual inventory, and double-action ownership remain non-live placeholders, not execution behavior. |
| PnL/economic claims | Counterfactual EV records are not balance-authoritative financial evidence. |

## Non-Negotiable Blocks

The following are blocked until a separate board decision:

| Blocked item | Reason |
|---|---|
| Live orders | Phase 5.1 scaffold is non-live only. |
| Canary | No fill calibration, no account-state evidence, and no live-risk decision. |
| Capital escalation | Outside Phase 5.1 scope. |
| Risk-limit relaxation | Outside Phase 5.1 scope. |
| Multi-venue launch | V2 scope is all-five, but current Phase 5.1 evidence authorizes only the Lighter-first non-live evidence lane. Non-Lighter venues require separate venue-readiness evidence and a later board decision before any multi-venue V2 launch. |
| Profitability claim | No balance-authoritative live economic evidence. |

## Evidence Summary

M6 source snapshot:

```text
/tmp/phase51_inputs/phase5_tail_1000_20260501T214411Z.telemetry.jsonl
```

M6 input hash:

```text
c2b50d00912b22f877e6e79be0ae16e2342d5ea3eaad22b7be3049f059312b64
```

M6 result:

```text
Input records scanned: 1000
Output telemetry records: 4001
Candidates evaluated: 2000
Replay labels emitted: 2000
Gate status: HOLD
Calibration status: SPARSE
Approved for live: false
Approved for canary: false
Approved for capital escalation: false
Admissible for financial claim: false
```

M6 artifact hashes:

```text
6ddbf774f6fb05f11508866d604d6ac7da89a54ef22e83a9bca6c32eb59090dc  telemetry.jsonl
884e4cd3459ff99d92db8ab25479253fd7f7b75a2f0296da205c4f3b8ac27b54  manifest.json
08ffe38dc4e4b8a0f8e8f5918ddd963815583098f0f994987ab65be6dcd003e8  evidence_pack/artifact_index.json
```

Validation:

```bash
python3 -m py_compile tools/check_telemetry_contract.py tools/phase51_ev_shadow.py
python3 -m unittest tests.test_telemetry_contract_gate
python3 tools/check_telemetry_contract.py \
  runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_m6/telemetry.jsonl
```

Observed result:

```text
Ran 71 tests
OK
OK: 4001 record(s) validated against schema v2
```

## Board Lane Verdicts

| Lane | Verdict | Notes |
|---|---|---|
| Quant EV | HOLD | Objective is represented structurally, but probability/cost calibration is missing. |
| Systems | PASS | Non-live harness, schema, deterministic manifesting, and fail-closed spec validation are in place. |
| Risk | PASS for scaffold, HOLD for execution | No-live/no-canary/no-capital/no-risk flags are enforced; execution-risk states remain placeholders. |
| Execution | PASS for Lighter non-live, HOLD beyond that | Lighter is acceptable as first non-live venue; account/native-limit evidence remains unresolved. |
| Data/Evidence | PASS for counterfactual evidence, HOLD for economics | Replay labels are source-linked and nonfinancial; balance-authoritative PnL is not claimed. |

## Current Evidence Boundary

```text
Phase 5.1k - Filled-horizon timebase recovery and recovered P-fill feature-matrix admissibility
```

Completed non-live scope:

- Added `tools/phase51k_filled_horizon_timebase_recovery.py`, a HOLD-only
  offline evidence gate that recovers filled-order source-tick horizons only
  when both order and fill source ticks are observable.
- Recovered `396 / 461` previously unresolved filled-order source-tick
  horizons and preserved the `4066` existing terminal/source-tick horizons.
- Rebuilt Phase 5.1h and Phase 5.1i with the 5.1j terminal-horizon recovery
  and 5.1k filled-horizon recovery packs.
- Kept `raw_identifier_redaction_status=PASS`, emitted no raw fill/order
  identifiers, and kept every live, canary, capital, risk-relaxation,
  EV-admission, model-training, and financial-claim authorization field false.

Repo-owned artifacts:

- Filled-horizon recovery tool:
  `tools/phase51k_filled_horizon_timebase_recovery.py`
- Recovered feature audit tool:
  `tools/phase51h_observed_pfill_feature_audit.py`
- Recovered matrix tool:
  `tools/phase51i_pfill_feature_matrix_admissibility.py`
- Filled-horizon recovery pack:
  `runs/phase51k_filled_horizon_timebase_recovery/PHASE51K-FILLED-HORIZON-TIMEBASE-RECOVERY-TWO-LANE-20260502T000000Z`
- Recovered feature audit pack:
  `runs/phase51h_observed_pfill_feature_audit/PHASE51K-RECOVERED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z`
- Recovered matrix pack:
  `runs/phase51i_pfill_feature_matrix_admissibility/PHASE51K-PFILL-FEATURE-MATRIX-ADMISSIBILITY-TWO-LANE-20260502T000000Z`

## Next Move

The next optimal move after Phase 5.1k is still not live trading. It is a
non-live evidence-improvement step:

- Resolve or explicitly quarantine the remaining `65` filled-order
  `MISSING_JOIN` rows without fabricating timebase data.
- Improve read-only Lighter native-limit and maker/taker completeness.
- Preserve Lighter official-doc assumptions as evidence metadata: post-only
  orders must rest as maker or cancel if crossing; account tier/fees/latency,
  sendTx/rest limits, and active-order limits are account/profile-sensitive and
  must come from official read endpoints or remain unknown.
- Reconcile excluded quarantine categories only with deterministic venue-native
  evidence; otherwise keep them excluded from training.
- Preserve all Phase 5.1 holds above.

## Phase 5.1i Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for the next non-live feature-completeness evidence
step.

Rationale:

- Redaction hardening passed: the rebuilt Phase 5.1f -> 5.1g -> 5.1h chain
  removes inherited raw `decision_id` fields from emitted labels and the
  Phase 5.1i matrix input reports `raw_identifier_input_present_count=0`.
- Evidence remains non-admissible for training/EV because the current matrix
  still lacks observed horizon timing on `4509` of `4527` labels, has only
  partial Lighter native-limit context for `2288` labels, has incomplete
  maker/taker status on `287` fills, has sparse venue/side buckets, and
  excludes `1613` quarantined/review groups from the observed-only diagnostic
  pack.
- No live, canary, capital, risk-limit, or strategy execution behavior changed.

Current evidence boundary:

```text
Phase 5.1i - Redacted P-fill feature-matrix admissibility
runs/phase51i_pfill_feature_matrix_admissibility/PHASE51I-PFILL-FEATURE-MATRIX-ADMISSIBILITY-REDACTED-TWO-LANE-20260502T000000Z
gate_status: HOLD
gate_reason: phase51i_missing_observed_horizon_features
raw_identifier_redaction_status: PASS
```

Next move:

```text
Implement a non-live observed-horizon recovery audit/tool, then rerun the
redacted 5.1f -> 5.1g -> 5.1h -> 5.1i chain. The target is reducing
observed_horizon_missing_count without weakening redaction, quarantine,
selection-bias, venue-native, or safety holds.
```

## Phase 5.1j Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for the next non-live feature-completeness evidence
step: fill-horizon/source-time recovery plus venue-native and maker/taker
evidence improvement.

Rationale:

- Observed-horizon recovery passed as a non-live evidence improvement. It
  recovered deterministic terminal source-tick horizons for `4048` canonical
  terminal-not-filled labels and preserved `18` existing horizons.
- The matrix still remains non-admissible because the remaining `461` missing
  horizons are filled-order rows that need a separate fill-time/source-time
  treatment; they were not fabricated from `fill_time_ms`.
- The redacted recovered 5.1h -> 5.1i chain keeps
  `raw_identifier_input_present_count=0` and
  `raw_identifier_redaction_status=PASS`.
- Other blockers remain: partial Lighter native-limit context for `2288`
  labels, incomplete maker/taker status on `287` fills, sparse venue/side
  buckets, and `1613` excluded quarantine/review groups.
- No live, canary, capital, risk-limit, or strategy execution behavior changed.

Current evidence boundary:

```text
Phase 5.1j - Observed-horizon recovery
runs/phase51j_observed_horizon_recovery/PHASE51J-OBSERVED-HORIZON-RECOVERY-TWO-LANE-20260502T000000Z
gate_status: HOLD
gate_reason: phase51j_observed_horizon_recovery_partial_horizon_missing
raw_identifier_redaction_status: PASS

Phase 5.1j - Recovered P-fill feature-matrix admissibility
runs/phase51j_pfill_feature_matrix_admissibility/PHASE51J-PFILL-FEATURE-MATRIX-ADMISSIBILITY-RECOVERED-TWO-LANE-20260502T000000Z
gate_status: HOLD
gate_reason: phase51i_missing_observed_horizon_features
observed_horizon_available_count: 4066
observed_horizon_missing_count: 461
```

Next move:

```text
Implement a non-live filled-order horizon/timebase recovery audit that maps
filled rows to source-tick or exchange-time lifecycle/fill evidence without
using model assumptions or financial claims. In parallel, continue
read-only Lighter native-limit and maker/taker evidence enrichment. Rerun the
same recovered 5.1h -> 5.1i chain and keep all safety holds intact.
```

## Phase 5.1k Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for the next non-live feature-completeness evidence
step: resolve or quarantine the remaining filled-horizon join misses and
improve Lighter native-limit/maker-taker evidence.

Rationale:

- Filled-horizon timebase recovery passed as a non-live evidence improvement.
  It recovered `396 / 461` filled-order source-tick horizons and did not
  substitute exchange milliseconds into source-tick fields.
- The recovered 5.1h -> 5.1i chain now reports `4462 / 4527` observed horizons
  available and `65` remaining missing horizons.
- The matrix remains non-admissible with gate reason
  `phase51i_filled_horizon_source_tick_still_missing`.
- Other blockers remain: partial Lighter native-limit context for `2288`
  labels, incomplete maker/taker status on `287` fills, sparse venue/side
  buckets, and `1613` excluded quarantine/review groups.
- No live, canary, capital, risk-limit, or strategy execution behavior changed.

Current evidence boundary:

```text
Phase 5.1k - Filled-horizon timebase recovery
runs/phase51k_filled_horizon_timebase_recovery/PHASE51K-FILLED-HORIZON-TIMEBASE-RECOVERY-TWO-LANE-20260502T000000Z
gate_status: HOLD
gate_reason: phase51k_filled_horizon_timebase_partial
recovered_source_tick_count: 396
still_missing_filled_horizon_count: 65
raw_identifier_redaction_status: PASS

Phase 5.1k - Recovered P-fill feature-matrix admissibility
runs/phase51i_pfill_feature_matrix_admissibility/PHASE51K-PFILL-FEATURE-MATRIX-ADMISSIBILITY-TWO-LANE-20260502T000000Z
gate_status: HOLD
gate_reason: phase51i_filled_horizon_source_tick_still_missing
observed_horizon_available_count: 4462
observed_horizon_missing_count: 65
```

Next move:

```text
Audit the remaining 65 MISSING_JOIN filled-order rows to determine whether they
can be deterministically joined through lifecycle/native evidence or must be
quarantined from calibration. Continue Lighter native-limit and maker/taker
evidence enrichment. Keep all non-live safety holds intact.
```

## Phase 5.1l Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for the next non-live feature-completeness evidence
step: improve Lighter native-limit and maker/taker evidence while preserving
sparse-bucket and observed-only selection-bias holds.

Rationale:

- Phase 5.1l added a source-key/hash fallback recovery gate for the remaining
  Phase 5.1k `MISSING_JOIN` filled-order rows. It emits only redacted source
  keys, hashes, counts, and source-tick horizons; it does not emit raw fill IDs,
  order IDs, client order IDs, venue order IDs, or decision IDs.
- The gate recovered the remaining `65 / 65` filled-order source-tick horizons:
  `43` via source P-fill horizon evidence and `22` via hashed observed-fill
  fallback.
- The recovered 5.1h -> 5.1i chain now reports `4527 / 4527` observed horizons
  available and `0` missing horizons.
- The matrix remains non-admissible with gate reason
  `phase51i_lighter_native_limit_pressure_not_fully_observed`.
- Remaining blockers are partial Lighter native-limit context for `2288`
  labels, incomplete maker/taker status on `287` fills, sparse venue/side
  buckets, and `1613` excluded quarantine/review groups.
- No live, canary, capital, risk-limit, or strategy execution behavior changed.

Current evidence boundary:

```text
Phase 5.1l - Filled-horizon source-key recovery
runs/phase51l_filled_horizon_source_key_recovery/PHASE51L-FILLED-HORIZON-SOURCE-KEY-RECOVERY-TWO-LANE-20260502T000000Z
gate_status: HOLD
gate_reason: phase51l_filled_horizon_source_key_complete_nonlive_hold
target_missing_join_count: 65
source_pfill_horizon_recovered_count: 43
observed_fill_hash_recovered_count: 22
still_missing_filled_horizon_count: 0
raw_identifier_redaction_status: PASS

Phase 5.1l - Recovered P-fill feature-matrix admissibility
runs/phase51i_pfill_feature_matrix_admissibility/PHASE51L-PFILL-FEATURE-MATRIX-ADMISSIBILITY-TWO-LANE-20260502T000000Z
gate_status: HOLD
gate_reason: phase51i_lighter_native_limit_pressure_not_fully_observed
observed_horizon_available_count: 4527
observed_horizon_missing_count: 0
matrix_blocker_ids:
- lighter_native_limit_pressure_not_fully_observed
- maker_taker_not_fully_observed_for_filled_orders
- sparse_pfill_feature_buckets
- observed_only_selection_bias_not_resolved
```

Next move:

```text
Do not start live/canary/model-training/EV-admission work. Improve read-only
Lighter native-limit and maker/taker evidence completeness, rerun 5.1h/5.1i,
and preserve sparse-bucket and observed-only selection-bias holds until there
is a board-approved calibration protocol.
```

## Phase 5.1m Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for the next non-live feature-completeness evidence
step: event-time native-limit pressure and venue-native maker/taker completion
across all filled venues while preserving sparse-bucket and observed-only
selection-bias holds.

Rationale:

- Phase 5.1m added an official Lighter limit-cap snapshot and wired it into the
  read-only account/native-limit evidence path. Official active-order caps are
  used only with observed active-order counts; they are not treated as event-time
  sendTx or REST remaining pressure.
- The Phase 5.1m Lighter evidence pack was accepted only for calibration-label
  ingestion. It keeps live/canary/capital/financial authority blocked and keeps
  sendTx/REST remaining pressure explicitly unobserved.
- The recovered 5.1h -> 5.1i chain keeps the Lighter native-limit rows partial:
  `0 / 2288` observed, `2288 / 2288` partial, and `0` unknown. This is
  intentional because the evidence is a current snapshot, not label-event-time
  native pressure.
- The matrix remains non-admissible with gate reason
  `phase51i_lighter_native_limit_pressure_not_fully_observed`.
- Remaining blockers are Lighter native-limit event-time alignment for `2288`
  labels, incomplete maker/taker status on `287` fills, sparse venue/side
  buckets, and `1613` excluded quarantine/review groups.
- No live, canary, capital, risk-limit, or strategy execution behavior changed.

Current evidence boundary:

```text
Phase 5.1m - Lighter official-doc native-limit enrichment
runs/phase51b_lighter_account_native_limits/PHASE51M-LIGHTER-OFFICIAL-LIMIT-ENRICHMENT-20260503T000000Z
status: PROMOTE_TO_PHASE51C_CALIBRATION_INGESTION
approved_for_live: false
approved_for_canary: false
approved_for_capital_escalation: false
limitations:
- lighter_sendtx_remaining_not_observed
- lighter_rest_request_limit_not_exposed_by_account_limits_payload

Phase 5.1m - Recovered P-fill feature-matrix admissibility
runs/phase51i_pfill_feature_matrix_admissibility/PHASE51M-PFILL-FEATURE-MATRIX-ADMISSIBILITY-NATIVE-DOC-CAP-TWO-LANE-20260503T000000Z
gate_status: HOLD
gate_reason: phase51i_lighter_native_limit_pressure_not_fully_observed
native_limit_observed_count: 0
native_limit_partial_count: 2288
native_limit_unknown_count: 0
observed_horizon_missing_count: 0
filled_horizon_source_key_unrecovered_count: 0
raw_identifier_redaction_status: PASS
matrix_blocker_ids:
- lighter_native_limit_pressure_not_fully_observed
- maker_taker_not_fully_observed_for_filled_orders
- sparse_pfill_feature_buckets
- observed_only_selection_bias_not_resolved
```

Next move:

```text
Do not start live/canary/model-training/EV-admission work. Build the next
non-live gate for event-time native-limit pressure and maker/taker completion
across all filled venues. Use only venue-native fill/trade/limit evidence; do
not infer role from quote intent, post-only flags, or strategy purpose. Rerun
5.1h/5.1i after any evidence recovery and preserve sparse-bucket and
observed-only selection-bias holds.
```

## Phase 5.1n Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for the next non-live evidence-completeness step:
venue-native maker/taker recovery where native trade/fill role sources exist,
plus future event-time native-limit instrumentation that includes sendTx/REST
pressure instead of active-order snapshots alone.

Rationale:

- Phase 5.1n added repo-owned event-time alignment for Lighter account snapshot
  logs. This upgrades the forensic record from current/doc-only capacity
  context to historical active-order context where snapshots align to label
  event time.
- The 025435 lane aligned `1728 / 2194` Lighter rows to event-time active-order
  snapshots and marked `466` as stale. The 073231 lane aligned `3700 / 3954`
  Lighter rows and marked `254` as stale.
- The gate deliberately keeps `native_limit_observed_count: 0` because sendTx
  and REST pressure were not historically observed. Active-order snapshots
  alone do not clear the full native-limit-pressure feature requirement.
- Phase 5.1n also added an all-venue maker/taker recovery gate. It preserved
  `174` already-observed filled rows and classified the remaining `287` filled
  rows as `MISSING_VENUE_NATIVE_ROLE_SOURCE`; no role was inferred from
  post-only flags, order intent, strategy purpose, or fee schedule.
- The recovered 5.1h -> 5.1i chain remains `HOLD` with the same four blockers:
  native-limit pressure incomplete, maker/taker incomplete, sparse buckets, and
  observed-only selection bias.

Current evidence boundary:

```text
Phase 5.1n - Event-time Lighter native-limit alignment
runs/phase51n_lighter_native_limit_time_alignment/PHASE51N-LIGHTER-NATIVE-LIMIT-TIME-ALIGNMENT-TERMINAL-STALE-7200S-20260429T025435Z
runs/phase51n_lighter_native_limit_time_alignment/PHASE51N-LIGHTER-NATIVE-LIMIT-TIME-ALIGNMENT-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z
gate_status: HOLD
native_limit_all_pressure_dimensions_observed_count: 0
raw_identifier_redaction_status: PASS

Phase 5.1n - Maker/taker attribution recovery
runs/phase51n_maker_taker_attribution_recovery/PHASE51N-MAKER-TAKER-ATTRIBUTION-RECOVERY-OBSERVED-ONLY-TWO-LANE-20260503T000000Z
gate_status: HOLD
maker_taker_observed_or_recovered_count: 174
maker_taker_partial_or_missing_count: 287
raw_identifier_redaction_status: PASS

Phase 5.1n - Recovered P-fill feature-matrix admissibility
runs/phase51i_pfill_feature_matrix_admissibility/PHASE51N-PFILL-FEATURE-MATRIX-ADMISSIBILITY-EVENT-TIME-NATIVE-LIMIT-TWO-LANE-20260503T000000Z
gate_status: HOLD
gate_reason: phase51i_lighter_native_limit_pressure_not_fully_observed
native_limit_observed_count: 0
native_limit_partial_count: 2288
maker_taker_observed_count: 174
maker_taker_partial_or_unknown_count: 222
maker_taker_missing_count: 65
raw_identifier_redaction_status: PASS
```

Next move:

```text
Do not start live/canary/model-training/EV-admission work. Continue with
venue-native maker/taker role capture/backfill for all five venues where source
retention allows it, and add forward-looking event-time native-limit telemetry
for sendTx/REST pressure. Rerun 5.1h/5.1i after any recovery; keep sparse
bucket and observed-only selection-bias holds until separately solved.
```

## Phase 5.1o Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for the next non-live evidence-completeness step:
quarantined raw-ID Lighter native trade join where safe, plus forward native
maker/taker capture for Aster, Extended, Paradex, and Hyperliquid and
event-time native-limit capture for sendTx/REST pressure.

Rationale:

- Phase 5.1o added a repo-owned native-role source inventory. It emits
  `native_role_evidence.jsonl` only for exact canonical venue-native role
  evidence and rejects inferred sources.
- The current inventory found `461` filled rows, `174` already-observed
  maker/taker rows, `0` exact canonical recoveries, `125` Lighter
  `SOURCE_AVAILABLE_NO_CANONICAL_JOIN` rows, and `162`
  `MISSING_VENUE_NATIVE_ROLE_SOURCE` rows.
- Venue split for the unresolved rows is Lighter `125` source-available but
  unjoined rows, plus Aster `113`, Extended `28`, Paradex `15`, and
  Hyperliquid `6` rows with no retained native role source in current artifacts.
- Lighter has historical native trade source material that may be recoverable
  through a quarantined raw-ID join. Aster, Extended, Paradex, and Hyperliquid
  lack retained event-time native role fields in current artifacts, so their
  evidence path is forward capture unless separate native fill/trade archives
  are supplied.
- The recovered 5.1h -> 5.1i chain remains `HOLD` with the same four blockers:
  native-limit pressure incomplete, maker/taker incomplete, sparse buckets, and
  observed-only selection bias.

Current evidence boundary:

```text
Phase 5.1o - Native role source inventory
runs/phase51o_native_role_source_inventory/PHASE51O-NATIVE-ROLE-SOURCE-INVENTORY-ALL-VENUE-20260503T120000Z
gate_status: HOLD
gate_reason: phase51o_native_role_sources_incomplete
filled_count: 461
input_observed_preserved_count: 174
recovered_native_role_count: 0
source_available_no_canonical_join_count: 125
missing_native_role_source_count: 162
raw_identifier_redaction_status: PASS

Phase 5.1o - Maker/taker attribution recovery rerun
runs/phase51n_maker_taker_attribution_recovery/PHASE51O-MAKER-TAKER-ATTRIBUTION-RECOVERY-ALL-VENUE-20260503T120000Z
gate_status: HOLD
maker_taker_observed_or_recovered_count: 174
maker_taker_partial_or_missing_count: 287
raw_identifier_redaction_status: PASS

Phase 5.1o - Recovered P-fill feature-matrix admissibility
runs/phase51i_pfill_feature_matrix_admissibility/PHASE51O-PFILL-FEATURE-MATRIX-ADMISSIBILITY-NATIVE-ROLE-RECOVERY-20260503T120000Z
gate_status: HOLD
gate_reason: phase51i_lighter_native_limit_pressure_not_fully_observed
native_limit_observed_count: 0
native_limit_partial_count: 2288
maker_taker_observed_count: 174
maker_taker_partial_or_unknown_count: 222
maker_taker_missing_count: 65
raw_identifier_redaction_status: PASS
```

Next move:

```text
Do not start live/canary/model-training/EV-admission work. Implement the
smallest quarantined Lighter-only native trade join that can emit exact
canonical role evidence without raw ID leakage, and add forward venue-native
role collectors/fields for Aster ORDER_TRADE_UPDATE.m, Extended isTaker,
Paradex liquidity, and Hyperliquid crossed. Preserve sparse-bucket and
observed-only selection-bias holds until separately solved.
```

## Phase 5.1p Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for the next non-live forward-capture evidence step:
capture venue-native maker/taker role fields and event-time native-limit
pressure at order/fill time across all five venues, then rerun the 5.1h/5.1i
matrix.

Rationale:

- Phase 5.1p implemented the quarantined Lighter-only native trade join
  authorized by Phase 5.1o. Raw Lighter side IDs are read only inside the local
  process and board-facing artifacts contain hashes and counts only.
- The all-backfill run indexed `531` unique native Lighter trades with complete
  native role truth (`345` maker, `186` taker, `0` unknown), but recovered
  `0 / 125` source-available canonical Lighter filled rows.
- The downstream maker/taker recovery therefore remains unchanged: `174`
  observed/recovered rows and `287` filled rows still partial or missing native
  role evidence.
- The recovered 5.1h -> 5.1i matrix remains `HOLD` with the same four blockers:
  native-limit pressure incomplete, maker/taker incomplete, sparse buckets, and
  observed-only selection bias.

Current evidence boundary:

```text
Phase 5.1p - Lighter native role canonical join
runs/phase51p_lighter_native_role_canonical_join/PHASE51P-LIGHTER-NATIVE-ROLE-CANONICAL-JOIN-ALL-BACKFILLS-20260503T140000Z
gate_status: HOLD
gate_reason: phase51p_lighter_native_role_join_incomplete
lighter_source_available_target_count: 125
recovered_lighter_native_role_count: 0
unrecovered_lighter_native_role_count: 125
native_role_evidence_record_count: 0
raw_identifier_redaction_status: PASS

Phase 5.1p - Maker/taker attribution recovery rerun
runs/phase51n_maker_taker_attribution_recovery/PHASE51P-MAKER-TAKER-ATTRIBUTION-RECOVERY-LIGHTER-NATIVE-20260503T140000Z
gate_status: HOLD
maker_taker_observed_or_recovered_count: 174
maker_taker_partial_or_missing_count: 287
raw_identifier_redaction_status: PASS

Phase 5.1p - Recovered P-fill feature-matrix admissibility
runs/phase51i_pfill_feature_matrix_admissibility/PHASE51P-PFILL-FEATURE-MATRIX-ADMISSIBILITY-LIGHTER-NATIVE-20260503T140000Z
gate_status: HOLD
gate_reason: phase51i_lighter_native_limit_pressure_not_fully_observed
native_limit_observed_count: 0
native_limit_partial_count: 2288
maker_taker_observed_count: 174
maker_taker_partial_or_unknown_count: 222
maker_taker_missing_count: 65
raw_identifier_redaction_status: PASS
```

Next move:

```text
Do not start live/canary/model-training/EV-admission work. Historical Lighter
native trade backfill recovery is exhausted under exact-hash rules. Add
forward venue-native role and event-time native-limit pressure capture across
all five venues, preserving the no-live, no-capital, no-risk-relaxation
boundary until the matrix blockers clear through observed evidence.
```

## Phase 5.1q Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for the next non-live evidence action: run
`tools/phase51q_forward_native_evidence_capture.py` on real forward-captured,
sanitized venue-native source rows, feed its `native_role_evidence.jsonl` into
Phase 5.1n maker/taker recovery, and rerun the Phase 5.1h/5.1i matrix.

Rationale:

- Phase 5.1q is the repo-owned forward path after historical Lighter native
  trade recovery exhausted exact-hash joins.
- The gate supports all-five venue maker/taker role evidence from explicit
  native fields: Lighter trades, Hyperliquid `crossed`, Paradex `liquidity`,
  Aster `ORDER_TRADE_UPDATE.m`, and Extended `isTaker`.
- The current native-limit pressure blocker is Lighter-specific, so non-Lighter
  rows are not applicable for that blocker while Lighter requires event-time
  active-order, sendTx, and REST/weighted-request headroom.
- Raw order/client/fill/trade identifiers are rejected from source rows and
  outputs. Evidence remains canonical-group/count/hash based.

Current repo-owned gate:

```text
Phase 5.1q - Forward native evidence capture
tool: tools/phase51q_forward_native_evidence_capture.py
spec: docs/PHASE5_1Q_FORWARD_NATIVE_EVIDENCE.md
status: HOLD
authorized outputs:
- native_role_evidence.jsonl
- forward_native_role_capture_labels.jsonl
- native_limit_pressure_labels.jsonl
- phase51q_forward_native_evidence_summary.json
prohibited:
- live orders
- canary
- model training
- EV admission
- capital escalation
- risk-limit relaxation
- financial claims
```

Baseline no-source evidence:

```text
runs/phase51q_forward_native_evidence/PHASE51Q-FORWARD-NATIVE-EVIDENCE-BASELINE-NO-SOURCES-20260503T000000Z
gate_status: HOLD
gate_reason: phase51q_forward_native_evidence_incomplete
native_role_evidence_record_count: 0
recovered_forward_native_role_count: 0
native_role_missing_source_count: 287
native_limit_missing_source_count: 3132
raw_identifier_redaction_status: PASS
```

Next move:

```text
Capture or locate real sanitized forward native source rows for all five role
sources and Lighter event-time native-limit pressure. Run Phase 5.1q, rerun
Phase 5.1n/5.1h/5.1i, and preserve HOLD unless the blockers clear through
observed venue-native evidence.
```

## Phase 5.1r Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for the next non-live evidence action: run
`tools/phase51r_forward_native_source_acquisition.py` on read-only local
venue-native snapshots, feed its sanitized `native_role_source.jsonl` and
`native_limit_source.jsonl` into Phase 5.1q, then rerun
Phase 5.1n/5.1h/5.1i.

Rationale:

- Phase 5.1r is the source-acquisition adapter required before Phase 5.1q when
  raw local snapshots contain venue identifiers that cannot enter 5.1q.
- The gate accepts all-five venue-native role fields only: Lighter
  `is_maker_ask` with account side, Hyperliquid `crossed`, Paradex
  `liquidity`, Aster `ORDER_TRADE_UPDATE.o.m`, and Extended `isTaker`.
- Raw order/client/fill/trade identifiers may exist only in quarantined local
  inputs. Outputs are canonical-group/count/hash based and redaction-checked.
- The baseline no-source run correctly clears no blockers and therefore
  preserves the existing HOLD boundary.

Current repo-owned gate:

```text
Phase 5.1r - Forward native source acquisition
tool: tools/phase51r_forward_native_source_acquisition.py
spec: docs/PHASE5_1R_FORWARD_NATIVE_SOURCE_ACQUISITION.md
status: HOLD
authorized outputs:
- native_role_source.jsonl
- native_limit_source.jsonl
- source_acquisition_labels.jsonl
- phase51r_forward_native_source_acquisition_summary.json
prohibited:
- live orders
- canary
- model training
- EV admission
- capital escalation
- risk-limit relaxation
- financial claims
```

Baseline no-source evidence:

```text
runs/phase51r_forward_native_source_acquisition/PHASE51R-FORWARD-NATIVE-SOURCE-ACQUISITION-BASELINE-NO-SOURCES-20260503T000000Z
gate_status: HOLD
gate_reason: phase51r_forward_native_source_acquisition_incomplete
native_role_target_count: 287
native_role_source_record_count: 0
native_role_target_recovered_count: 0
lighter_native_limit_target_count: 3132
native_limit_source_record_count: 0
native_limit_complete_source_record_count: 0
lighter_native_limit_target_recovered_count: 0
raw_identifier_redaction_status: PASS
```

Downstream rerun evidence:

```text
runs/phase51q_forward_native_evidence/PHASE51R-FORWARD-NATIVE-EVIDENCE-BASELINE-NO-SOURCES-20260503T000000Z
runs/phase51n_maker_taker_attribution_recovery/PHASE51R-MAKER-TAKER-ATTRIBUTION-RECOVERY-BASELINE-NO-SOURCES-20260503T000000Z
runs/phase51h_observed_pfill_feature_audit/PHASE51R-OBSERVED-PFILL-FEATURE-AUDIT-BASELINE-NO-SOURCES-20260503T000000Z
runs/phase51i_pfill_feature_matrix_admissibility/PHASE51R-PFILL-FEATURE-MATRIX-ADMISSIBILITY-BASELINE-NO-SOURCES-20260503T000000Z
```

Next move:

```text
Supply real read-only native snapshots to Phase 5.1r, optionally include
validated source-link sidecars when redacted source hashes rather than direct
group/order keys carry the join, require redacted outputs, then rerun Phase
5.1q -> 5.1n -> 5.1h -> 5.1i. Preserve HOLD unless observed venue-native
evidence materially reduces the native-role/native-limit blockers without
introducing raw IDs or unsafe authorization flags.
```

## Phase 5.1r Source-Link Sidecar Board Decision

Decision: `PROMOTE` only for non-live validated source-hash linkage inside
Phase 5.1r.

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Rationale:

- Existing 5.1s local snapshots can be redaction-safe but unjoined when the
  source rows do not carry direct `canonical_group_id` or `order_key`.
- A source-link sidecar lets future forward captures bind redacted source hashes
  to already observed P-fill labels without preserving raw venue identifiers in
  Phase 5.1r output.
- The sidecar is validated against observed `canonical_group_id` / `order_key`,
  rejects duplicates, rejects conflicts, rejects raw identifier fields, and
  rejects unsafe true authorization flags.
- It does not infer maker/taker roles, Lighter native-limit pressure, fill
  outcomes, EV, PnL, or economic performance.

Current repo-owned change:

```text
tool: tools/phase51r_forward_native_source_acquisition.py
new input: repeated --source-link-jsonl
new label field: canonical_group_link_source
new summary fields:
- source_link_record_count
- source_link_applied_count
- source_link_hash_count
- canonical_group_link_source_counts
- source_link_artifacts
test coverage:
- source-link sidecar recovers joinable staged rows
- ambiguous or raw source-link sidecars are rejected
```

Next move:

```text
Capture forward source snapshots and, when direct group/order keys cannot be
embedded in redacted source rows, capture a redacted source-link sidecar that
maps source hashes to observed canonical groups/order keys. Then run Phase 5.1r
with --source-link-jsonl and feed only sanitized 5.1r outputs into Phase 5.1q.
```

## Phase 5.1s Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for local non-live source staging through
`tools/phase51s_local_native_source_acquisition.py`, followed by the existing
Phase 5.1r -> 5.1q -> 5.1n -> 5.1h -> 5.1i evidence chain.

Rationale:

- Phase 5.1s reduces operational risk by requiring an explicit local manifest
  before any native snapshot is passed to Phase 5.1r.
- It rejects network paths, `.env` files, symlinks, secret-shaped fields, and
  unsafe true authorization flags.
- It strips raw order/client/fill/trade identifiers and emits only redacted
  local source rows plus label/summary/manifest artifacts.
- It stages optional redacted source-link sidecars from the manifest
  `source_links` list so redacted source hashes can be linked to observed
  canonical groups/order keys before Phase 5.1r.
- It is not a blocker-clearing gate by itself. Blocker reduction requires
  downstream exact canonical joins and complete venue-native evidence.

Current repo-owned gate:

```text
Phase 5.1s - Local native source acquisition
tool: tools/phase51s_local_native_source_acquisition.py
spec: docs/PHASE5_1S_LOCAL_NATIVE_SOURCE_ACQUISITION.md
example manifest: configs/phase51s_local_native_source_manifest.example.json
status: HOLD
authorized output:
- local_native_source.jsonl for Phase 5.1r --source-json
- local_source_link_sidecar.jsonl for Phase 5.1r --source-link-jsonl
prohibited:
- live orders
- canary
- model training
- EV admission
- capital escalation
- risk-limit relaxation
- financial claims
```

First local-source evidence:

```text
runs/phase51s_local_native_source_acquisition/PHASE51S-LOCAL-NATIVE-SOURCE-EXISTING-LIGHTER-SOURCES-20260503T000000Z
gate_status: HOLD
gate_reason: phase51s_local_native_source_acquisition_complete_nonlive_hold
source_row_count: 405
join_key_source_row_count: 0
source_row_without_join_key_count: 405
complete_lighter_native_limit_source_row_count: 0
raw_identifier_fields_stripped_count: 3500
raw_identifier_redaction_status: PASS
clears_phase51_blockers: false
```

Current source-link staging support:

```text
manifest input: optional source_links list
accepted sidecar row: source-record hash plus canonical_group_id or order_key
rejected sidecar row: unsupported fields, raw venue IDs, duplicate hashes,
  unsafe true authorization flags, secret-shaped fields, non-string hash/join
  fields, or missing join fields
output: local_source_link_sidecar.jsonl
status: HOLD, non-live join evidence only
sidecar-only run: incomplete_source_links_only
```

Downstream Phase 5.1r evidence from that staged source:

```text
runs/phase51r_forward_native_source_acquisition/PHASE51S-TO-R-EXISTING-LIGHTER-SOURCES-20260503T000000Z
gate_status: HOLD
gate_reason: phase51r_forward_native_source_acquisition_incomplete
native_source_acquisition_status_counts:
- UNJOINED_NO_CANONICAL_GROUP: 405
native_role_target_recovered_count: 0 / 287
lighter_native_limit_target_recovered_count: 0 / 3132
```

Next move:

```text
Use Phase 5.1s as the mandatory local preflight for future native source
snapshots. Capture forward all-five venue-native role rows with canonical
group/order-key linkage and capture complete Lighter event-time active-order,
sendTx, and REST/weighted-request pressure rows. If direct group/order keys
cannot be embedded in redacted source rows, provide a redacted manifest
`source_links` sidecar that maps source hashes to observed labels. Rerun Phase
5.1s -> 5.1r -> 5.1q -> 5.1n -> 5.1h -> 5.1i, preserving HOLD unless observed
evidence clears the blockers without raw IDs or unsafe authorization flags.
```

## Phase 5.1t Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for HOLD-only source-link sidecar generation through
`tools/phase51t_source_link_sidecar_builder.py`, followed by Phase 5.1s ->
5.1r -> 5.1q -> 5.1n -> 5.1h -> 5.1i.

Rationale:

- Existing local source snapshots may contain raw order/client identifiers that
  cannot be emitted downstream.
- Phase 5.1t uses those identifiers only inside a quarantined local process to
  match existing redacted `order_id_hash` / `client_order_id_hash` values from
  observed P-fill labels.
- Phase 5.1t emits only source-record hashes plus `canonical_group_id` /
  `order_key` joins for Phase 5.1s `source_links` staging.
- It does not infer maker/taker role, Lighter native-limit pressure, EV,
  PnL, or financial performance.

Current repo-owned gate:

```text
Phase 5.1t - Source-link sidecar builder
tool: tools/phase51t_source_link_sidecar_builder.py
spec: docs/PHASE5_1T_SOURCE_LINK_SIDECAR_BUILDER.md
status: HOLD
authorized output:
- source_links.sanitized.jsonl for Phase 5.1s manifest source_links
prohibited:
- live orders
- canary
- model training
- EV admission
- capital escalation
- risk-limit relaxation
- financial claims
```

Existing-source evidence:

```text
runs/phase51t_source_link_sidecar_builder/PHASE51T-SOURCE-LINK-SIDECAR-BUILDER-EXISTING-LIGHTER-SOURCES-20260503T000000Z
gate_status: HOLD
source_row_count: 1522
source_link_record_count: 363
source_link_status_counts:
- SOURCE_LINK_EMITTED: 363
- DUPLICATE_SOURCE_HASH_ALREADY_EMITTED: 546
- NO_OBSERVED_IDENTITY_MATCH: 570
- NO_ORDER_IDENTITY_HASH: 17
- AMBIGUOUS_OBSERVED_IDENTITY_MATCH: 26
clears_phase51_blockers: false
```

Downstream evidence:

```text
runs/phase51s_local_native_source_acquisition/PHASE51T-TO-S-EXISTING-LIGHTER-SOURCES-20260503T000000Z
staged_source_row_count: 1522
staged_source_link_row_count: 363

runs/phase51r_forward_native_source_acquisition/PHASE51T-TO-R-EXISTING-LIGHTER-SOURCES-20260503T000000Z
source_link_applied_count: 909
native_role_source_record_count: 296
native_role_target_recovered_count: 0 / 287
lighter_native_limit_target_recovered_count: 0 / 3132

runs/phase51i_pfill_feature_matrix_admissibility/PHASE51T-PFILL-FEATURE-MATRIX-ADMISSIBILITY-EXISTING-LIGHTER-SOURCES-20260503T000000Z
gate_status: HOLD
gate_reason: phase51i_lighter_native_limit_pressure_not_fully_observed
matrix_blocker_ids:
- lighter_native_limit_pressure_not_fully_observed
- maker_taker_not_fully_observed_for_filled_orders
- sparse_pfill_feature_buckets
- observed_only_selection_bias_not_resolved
```

Next move:

```text
Use Phase 5.1t for future forward captures where direct canonical group/order
keys cannot be embedded in redacted source rows. Existing local Lighter
artifacts are now exhausted for blocker reduction: they prove sidecar plumbing
works but do not clear missing all-five venue-native roles or complete Lighter
native-limit pressure. Capture new forward read-only native source rows with
canonical linkage, then rerun 5.1t/5.1s/5.1r/5.1q/5.1n/5.1h/5.1i.
```

## Phase 5.1u Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for HOLD-only forward capture target manifest
generation through `tools/phase51u_forward_capture_target_manifest.py`.

Rationale:

- Existing source plumbing is redaction-safe, but current retained artifacts
  are exhausted and still recover `0 / 287` missing native-role targets and
  `0 / 3132` Lighter native-limit targets.
- Phase 5.1u makes the fresh capture pilot exact and auditable by emitting the
  required canonical groups, order keys, venue-specific native fields, and
  Lighter event-time limit-pressure fields.
- Phase 5.1u does not capture venue truth, does not infer maker/taker role or
  native-limit pressure, and does not clear blockers by itself.
- Phase 5.1r now aggregates distinct source-record hashes for the same
  canonical group so valid multi-fill native source bundles are not falsely
  classified as partial.

Current repo-owned gate:

```text
Phase 5.1u - Forward capture target manifest
tool: tools/phase51u_forward_capture_target_manifest.py
spec: docs/PHASE5_1U_FORWARD_CAPTURE_TARGET_MANIFEST.md
status: HOLD
authorized output:
- native_role_capture_targets.jsonl
- lighter_native_limit_capture_targets.jsonl
- capture_bundle_manifest_template.json
prohibited:
- live orders
- canary
- model training
- EV admission
- capital escalation
- risk-limit relaxation
- financial claims
- source truth inference
```

Baseline target-manifest evidence:

```text
runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z
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

Next move:

```text
Execute a fresh read-only forward source-capture pilot against the Phase 5.1u
target manifest, using direct canonical group/order-key linkage or 5.1t/5.1s
source-link sidecars. Then rerun 5.1s -> 5.1r -> 5.1q -> 5.1n -> 5.1h ->
5.1i and require measurable blocker reduction before any calibration/model
training review.
```

## Phase 5.1v Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for offline forward capture bundle-readiness
validation through `tools/phase51v_forward_capture_bundle_readiness.py`.

Rationale:

- The Phase 5.1u target manifest is exact, but the repo still needs a safe
  handoff between externally captured sanitized private/read-only source files
  and Phase 5.1s.
- Phase 5.1v verifies local bundle readiness without network calls, venue API
  calls, secret reads, live orders, source-truth inference, EV admission, or
  model training.
- The gate rejects unsafe source surfaces and emits a generated Phase 5.1s
  manifest only when all targets are structurally covered by local redacted
  source rows or staged source-link sidecars.
- Source-link sidecars are validated and applied as join aids inside Phase
  5.1v, but sidecars alone do not mark targets ready; a linked source row must
  be present and must carry the required venue-native role or Lighter
  native-limit fields.
- The baseline run against the Phase 5.1u placeholder template correctly
  remains `HOLD` and clears no blocker.

Current repo-owned gate:

```text
Phase 5.1v - Forward capture bundle readiness
tool: tools/phase51v_forward_capture_bundle_readiness.py
spec: docs/PHASE5_1V_FORWARD_CAPTURE_BUNDLE_READINESS.md
status: HOLD
authorized output:
- capture_bundle_readiness_labels.jsonl
- missing_native_role_capture_targets.jsonl
- missing_lighter_native_limit_capture_targets.jsonl
- phase51s_manifest.generated.json
- phase51v_forward_capture_bundle_readiness_summary.json
- phase51v_manifest.json
prohibited:
- live orders
- canary
- network source paths
- secret-shaped fields
- .env source files
- symlink source files
- model training
- EV admission
- capital escalation
- risk-limit relaxation
- financial claims
- source truth inference
```

Baseline bundle-readiness evidence:

```text
runs/phase51v_forward_capture_bundle_readiness/PHASE51V-FORWARD-CAPTURE-BUNDLE-READINESS-TEMPLATE-20260503T000000Z
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

Source-link readiness validation:

```text
source row + valid source-link sidecar: target can become ready
source-link sidecar only: target remains missing
duplicate source-link hash: rejected
summary fields:
- source_link_hash_count
- source_link_applied_row_count
```

Next move:

```text
Acquire or produce a sanitized local read-only all-five forward capture bundle,
run Phase 5.1v against the Phase 5.1u target manifest, and only if 5.1v emits
generated_phase51s_manifest_ready=true run 5.1s -> 5.1r -> 5.1q -> 5.1n ->
5.1h -> 5.1i. Require measurable blocker reduction before any
calibration/model-training review.
```

## Phase 5.1w Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for offline forward capture request-pack generation
through `tools/phase51w_forward_capture_request_pack.py`.

Rationale:

- Phase 5.1v validates supplied local bundles, but the previous handoff still
  required an operator to infer exact bundle requirements from several docs and
  validator internals.
- Phase 5.1w keeps the request pack mechanically tied to the Phase 5.1u target
  manifest and emits the exact local file list, required fields, join strategy,
  prohibitions, manifest skeleton, and next commands.
- Phase 5.1w does not call venue APIs, read secrets, validate source truth,
  infer maker/taker role, infer native-limit pressure, clear blockers, or
  authorize economics.

Current repo-owned gate:

```text
Phase 5.1w - Forward capture request pack
tool: tools/phase51w_forward_capture_request_pack.py
spec: docs/PHASE5_1W_FORWARD_CAPTURE_REQUEST_PACK.md
status: HOLD
authorized output:
- forward_capture_request_pack.md
- forward_capture_request_pack.json
- capture_bundle_manifest.skeleton.json
- phase51w_forward_capture_request_pack_summary.json
- phase51w_manifest.json
prohibited:
- live orders
- canary
- network source paths
- .env source files
- symlink source files
- secret-shaped fields
- raw venue identifiers
- model training
- EV admission
- capital escalation
- risk-limit relaxation
- financial claims
- source truth inference
```

Baseline request-pack evidence:

```text
runs/phase51w_forward_capture_request_pack/PHASE51W-FORWARD-CAPTURE-REQUEST-PACK-CANONICAL-20260504T000000Z
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

Next move:

```text
Use the Phase 5.1w request pack and skeleton manifest to provide six sanitized
local read-only source files, then run 5.1v. If generated_phase51s_manifest_ready=true,
run 5.1s -> 5.1r -> 5.1q -> 5.1n -> 5.1h -> 5.1i. Stop if private
credentials, live/canary/capital/risk authorization, or unredacted source
material would be required.
```

## Phase 5.1x Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for Hyperliquid subset native-role source-readiness
through `tools/phase51x_hyperliquid_native_role_adapter.py`.

Rationale:

- Hyperliquid has only `6` current native-role targets and the official
  Hyperliquid `info` endpoint exposes `userFills`/`userFillsByTime` records
  with boolean `crossed`.
- The adapter is offline-only: it consumes already-local JSON/JSONL source
  snapshots, performs no network calls, reads no secrets or `.env` files,
  rejects URI paths, strips raw `oid`/`cloid`/trade identifiers from output,
  and never infers maker/taker role.
- Phase 5.1x recovered `6 / 6` current Hyperliquid target groups from a
  read-only public-address source snapshot, and Phase 5.1v recognized
  `6 / 287` all-five native-role targets ready.
- Phase 5.1x is not an all-five source-complete bundle because Aster,
  Extended, Lighter, Paradex, and Lighter event-time native-limit pressure
  remain incomplete.

Current repo-owned gate:

```text
Phase 5.1x - Hyperliquid native-role adapter
tool: tools/phase51x_hyperliquid_native_role_adapter.py
status: HOLD
authorized output:
- hyperliquid_forward_native_role_snapshot.jsonl
- hyperliquid_native_role_adapter_labels.jsonl
- phase51x_hyperliquid_native_role_adapter_summary.json
- phase51x_manifest.json
prohibited:
- live orders
- canary
- network source paths
- .env source files
- symlink source files
- secret-shaped fields
- raw venue identifiers in output
- model training
- EV admission
- capital escalation
- risk-limit relaxation
- financial claims
- source truth inference
```

Evidence:

```text
runs/phase51x_hyperliquid_native_role_adapter/PHASE51X-HYPERLIQUID-USERFILLS-NATIVE-ROLE-20260504T000000Z
gate_status: HOLD
hyperliquid_target_recovered_count: 6 / 6
source_row_count: 2000
source_row_emitted_count: 7
raw_identifier_redaction_status: PASS

runs/phase51v_forward_capture_bundle_readiness/PHASE51V-HYPERLIQUID-PARTIAL-SOURCE-HOLD-20260504T000000Z
gate_status: HOLD
native_role_capture_target_ready_count: 6 / 287
lighter_native_limit_capture_target_ready_count: 0 / 3132
generated_phase51s_manifest_ready: false
clears_phase51_blockers: false
```

Next move:

```text
Reuse the Phase 5.1x Hyperliquid source in the next all-five candidate bundle.
Build or capture the remaining Aster, Extended, Paradex, and Lighter native-role
sources plus complete Lighter event-time native-limit pressure, then rerun
Phase 5.1v before Phase 5.1s.
```

## Phase 5.1y Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for offline all-venue native-role source normalization
through `tools/phase51y_all5_native_role_adapter.py`.

Rationale:

- Aster, Extended, Lighter, and Paradex still need fresh native-role source
  rows, but the repo lacked a single all-venue offline normalizer that could
  convert already-local sanitized source rows into Phase 5.1v-ready input.
- Phase 5.1y accepts only explicit venue-native role fields: Aster
  `ORDER_TRADE_UPDATE` / `o.m` with positive fill quantity, Extended
  `isTaker` / `is_taker`, Hyperliquid `crossed`, Lighter `account_index` plus
  `is_maker_ask` and side account IDs, and Paradex `liquidity`.
- Phase 5.1y requires direct canonical group/order-key linkage. Missing,
  non-targeted, or ambiguous joins remain HOLD labels and do not emit source
  truth.
- The adapter is offline-only: it consumes already-local JSON/JSONL source
  snapshots, performs no network calls, reads no secrets or `.env` files,
  rejects URI paths, rejects symlink paths, strips raw venue order/trade
  identifiers from output, and never infers maker/taker role from strategy
  intent, post-only status, fees, or economics.

Current repo-owned gate:

```text
Phase 5.1y - all-venue native-role adapter
tool: tools/phase51y_all5_native_role_adapter.py
status: HOLD
authorized output:
- all5_forward_native_role_snapshot.jsonl
- all5_native_role_adapter_labels.jsonl
- phase51y_all5_native_role_adapter_summary.json
- phase51y_manifest.json
prohibited:
- live orders
- canary
- network source paths
- .env source files
- symlink source files
- secret-shaped fields
- raw venue identifiers in output
- model training
- EV admission
- capital escalation
- risk-limit relaxation
- financial claims
- source truth inference
```

Evidence:

```text
runs/phase51y_all5_native_role_adapter/PHASE51Y-ALL5-STAGED-EMPTY-NATIVE-ROLE-HOLD-20260504T000000Z
gate_status: HOLD
native_role_target_recovered_count: 0 / 287
source_row_count: 0
source_row_emitted_count: 0
raw_identifier_redaction_status: PASS
clears_phase51_blockers: false

runs/phase51y_all5_native_role_adapter/PHASE51Y-HYPERLIQUID-REUSE-NATIVE-ROLE-HOLD-20260504T000000Z
gate_status: HOLD
native_role_target_recovered_count: 6 / 287
native_role_target_recovered_counts_by_venue: hyperliquid=6
source_row_count: 7
source_row_emitted_count: 7
raw_identifier_redaction_status: PASS
clears_phase51_blockers: false
```

Next move:

```text
Use Phase 5.1y as the offline normalization step after fresh read-only native
role capture for Aster, Extended, Paradex, and Lighter. Reuse Phase 5.1x or
Phase 5.1y-normalized Hyperliquid rows for Hyperliquid. Then run Phase 5.1v
against the resulting local capture bundle and proceed to Phase 5.1s only if
5.1v emits generated_phase51s_manifest_ready=true.
```

## Phase 5.1z Board Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only for bounded read-only private-source native-role
capture and sanitized Phase 5.1v bundle input generation through
`tools/phase51z_readonly_native_role_capture.py`.

Rationale:

- Phase 5.1y normalized already-local rows, but Aster, Extended, Paradex, and
  Lighter still required a safe capture/sanitizer step that could use existing
  local read-only credentials without writing raw identifiers or secrets.
- Phase 5.1z is HOLD-only. It never places, edits, cancels, replaces, or
  cancels-all orders; never enables canary/live; never changes capital or risk
  limits; and never emits financial claims.
- Phase 5.1z records only credential presence booleans, redacted hashes,
  sanitized target-linked native fields, artifact hashes, and source counts.
  Raw venue order IDs, client IDs, trade IDs, private keys, tokens, signatures,
  and authorization headers are not persisted.
- Phase 5.1z now emits redaction-safe per-venue diagnostics so the board can
  separate missing native fields from no-target-match source coverage without
  exposing raw identifiers.
- Lighter network capture remains delegated to the existing Phase 5.1b/5.1c
  read-only collectors. Phase 5.1z can consume already-local Lighter source
  rows, but the retained Lighter backfills did not match the current target
  manifest.

Current repo-owned gate:

```text
Phase 5.1z - read-only native-role source capture
tool: tools/phase51z_readonly_native_role_capture.py
status: HOLD
authorized output:
- source_snapshots/phase51z_forward_native_role_rows.jsonl
- phase51z_readonly_native_role_capture_labels.jsonl
- phase51z_candidate_manifest.json
- phase51z_readonly_native_role_capture_summary.json
- manifest.json
diagnostic fields:
- target_count / target_ready_count / target_missing_count
- source_row_count / native_field_ready_count
- target_matched_row_count / duplicate_matched_row_count
- no_target_match_count / rows_with_redacted_hash_candidates
- target_time_windows_by_venue
prohibited:
- live orders
- canary
- order place/edit/cancel/replace/cancel-all
- network source paths for local source inputs
- .env source files as native source inputs
- symlink source files
- secret-shaped fields in source/output records
- raw venue identifiers in output
- model training
- EV admission
- capital escalation
- risk-limit relaxation
- financial claims
```

Evidence:

```text
runs/phase51z_readonly_native_role_capture/PHASE51Z-READONLY-NATIVE-ROLE-CAPTURE-20260504T000000Z
gate_status: HOLD
fetch_readonly_requested: true
fetch_status: aster=268 rows, extended=1601 rows, paradex=30 rows, lighter=SKIPPED
sanitized_source_row_count: 67
sanitized_source_row_counts_by_venue: aster=39, extended=21, paradex=7
raw_identifier_redaction_status: PASS
clears_phase51_blockers: false

runs/phase51v_forward_capture_bundle_readiness/PHASE51V-PHASE51Z-READONLY-NATIVE-ROLE-CAPTURE-HOLD-20260504T000000Z
gate_status: HOLD
native_role_capture_target_ready_count: 67 / 287
lighter_native_limit_capture_target_ready_count: 0 / 3132
generated_phase51s_manifest_ready: false
downstream_chain_ready: false

runs/phase51v_forward_capture_bundle_readiness/PHASE51V-COMBINED-PHASE51Z-HYPERLIQUID-HOLD-20260504T000000Z
gate_status: HOLD
native_role_capture_target_ready_count: 73 / 287
native_role_capture_target_missing_count: 214 / 287
missing native-role targets by venue: aster=74, extended=7, lighter=125, paradex=8
lighter_native_limit_capture_target_ready_count: 0 / 3132
generated_phase51s_manifest_ready: false
downstream_chain_ready: false

runs/phase51z_readonly_native_role_capture/PHASE51Z-LIGHTER-RETAINED-BACKFILLS-HOLD-20260504T000000Z
gate_status: HOLD
sanitized_source_row_count: 0
capture_status_counts: NO_TARGET_MATCH=1400

runs/phase51z_readonly_native_role_capture/PHASE51Z-DIAGNOSTIC-READONLY-NATIVE-ROLE-CAPTURE-20260504T220117Z
gate_status: HOLD
fetch_status: aster=824 rows, extended=1601 rows, paradex=163 rows, lighter=SKIPPED
sanitized_source_row_count: 67
sanitized_source_row_counts_by_venue: aster=39, extended=21, paradex=7
capture_diagnostics:
- aster: target_ready=39/113, native_field_ready=824, no_target_match=784
- extended: target_ready=21/28, native_field_ready=1601, no_target_match=1579
- lighter: target_ready=0/125, native_field_ready=300, no_target_match=300
- paradex: target_ready=7/15, native_field_ready=163, no_target_match=156

runs/phase51v_forward_capture_bundle_readiness/PHASE51V-COMBINED-PHASE51Z-DIAGNOSTIC-HYPERLIQUID-HOLD-20260504T220117Z
gate_status: HOLD
native_role_capture_target_ready_count: 73 / 287
native_role_capture_target_missing_count: 214 / 287
missing native-role targets by venue: aster=74, extended=7, lighter=125, paradex=8
lighter_native_limit_capture_target_ready_count: 0 / 3132
generated_phase51s_manifest_ready: false
downstream_chain_ready: false
```

Next move:

```text
Prioritize a Lighter-only safe observed source with target-linked native role
plus event-time active-order/sendTx/REST-or-weighted request pressure. Continue
Aster/Extended/Paradex recovery only where it can improve target linkage rather
than re-fetching already no-target-matching rows. Do not infer missing Lighter
pressure from caps, account tiers, empty headers, or documentation-only limits.
```

## Phase 5.1n Forward Native-Limit Source Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only the repo-owned non-live plumbing that converts complete
Phase 5.1n Lighter event-time native-limit alignment rows into a Phase 5.1v
local source artifact.

Rationale:

- Phase 5.1v already requires explicit Lighter native-limit fields and cannot
  infer missing pressure from active-order alignment alone.
- Phase 5.1n is the correct owner of event-time Lighter active-order alignment,
  so it should also emit a downstream `lighter_forward_native_limit_pressure_snapshot.jsonl`
  only when active-order headroom, sendTx limit/remaining, and REST-or-weighted
  limit/remaining are all present.
- The generated Phase 5.1v manifest is safe only when
  `forward_native_limit_pressure_source_count > 0`; otherwise it remains an
  empty HOLD artifact and does not reduce the blocker.

Evidence:

```text
runs/phase51n_lighter_native_limit_time_alignment/PHASE51N-LIGHTER-NATIVE-LIMIT-FORWARD-SOURCE-RETEST-20260504T000000Z
gate_status: HOLD
native_limit_event_time_aligned_count: 3700
native_limit_all_pressure_dimensions_observed_count: 0
forward_native_limit_pressure_source_count: 0
phase51v_lighter_native_limit_manifest_ready: false
```

Additional 2026-05-04 read-only header probe:

```text
runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-READONLY-HEADER-PROBE-20260504T000000Z
phase51b_acceptance.status: PROMOTE_TO_PHASE51C_CALIBRATION_INGESTION
safe_nonlive_flags: true
accountLimits response header names: []
sendTx limit/remaining observed: false
REST-or-weighted limit/remaining observed: false

runs/phase51n_lighter_native_limit_time_alignment/PHASE51N-LIGHTER-NATIVE-LIMIT-HEADER-PROBE-HOLD-20260504T000000Z
native_limit_event_time_aligned_count: 3700
native_limit_all_pressure_dimensions_observed_count: 0
forward_native_limit_pressure_source_count: 0
phase51v_lighter_native_limit_manifest_ready: false
```

Interpretation:

```text
The no-order read-only Lighter endpoints currently available to the repo do not
expose the missing sendTx or REST-or-weighted remaining pressure fields in
either body keys or sanitized rate/quota response headers. The board must keep
Phase 5.1v/5.1n on HOLD rather than infer remaining pressure from official
caps. Future payloads that do expose those fields are now preserved by the
Phase 5.1b collector.
```

Next move:

```text
Capture or stage the missing event-time sendTx and REST-or-weighted
limit/remaining pressure fields from a safe observed source if one becomes
available, rerun Phase 5.1n, and feed its generated manifest into Phase 5.1v
only if complete forward rows are emitted. Do not infer pressure from caps. In
parallel, continue remaining all-five native-role source capture where it does
not depend on unavailable Lighter request-pressure fields.
```

## Phase 5.1u/5.1z Target-Link Hygiene Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only the repo-owned target-link hygiene patch and its
non-live replay evidence.

Rationale:

- Phase 5.1u target rows should preserve already-redacted `order_id_hash`,
  `client_order_id_hash`, `decision_id_hash`, `first_fill_time_ms`, and
  `last_fill_time_ms` from canonical P-fill labels so downstream source capture
  can be audited without re-opening raw identifiers.
- Phase 5.1z should consider official Lighter trade side order IDs
  (`ask_id`/`bid_id`) as candidate identity hashes as well as side client IDs.
- The patched replay still recovered `0 / 125` current Lighter native-role
  targets from the retained Lighter trade-backfill sources, so those sources are
  exhausted for current blocker reduction.

Evidence:

```text
runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-LINK-HYGIENE-20260505T000000Z
runs/phase51z_readonly_native_role_capture/PHASE51Z-LIGHTER-TARGET-LINK-HYGIENE-REPLAY-HOLD-20260505T000000Z
runs/phase51v_forward_capture_bundle_readiness/PHASE51V-LIGHTER-TARGET-LINK-HYGIENE-HOLD-20260505T000000Z
gate_status: HOLD
Lighter source rows replayed: 1400
Lighter native-field-ready rows: 1400
Lighter target-linked rows emitted: 0
Lighter native-limit pressure target ready count: 0 / 3132
raw_identifier_redaction_status: PASS
```

Next move:

```text
Do not refetch or replay the same retained Lighter trade snapshots for blocker
reduction. Find a new safe read-only Lighter source or linkage sidecar that can
connect native trade rows to the current canonical target groups. Keep native
limit pressure on HOLD unless event-time sendTx and REST-or-weighted
limit/remaining fields are observed from a non-live-authorized source.
```

## Phase 5.1z Unlinked Source-Row Preservation Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only the repo-owned unlinked sanitized source-row scaffold
and its non-live HOLD evidence.

Rationale:

- Retained Lighter trade-backfill rows contain native role fields but do not
  directly match the current canonical Phase 5.1u target hashes.
- Emitting a sanitized unlinked source row preserves source truth for a later
  validated source-link sidecar without emitting raw order IDs, client IDs, or
  trade IDs.
- Phase 5.1v must keep unlinked rows incomplete unless a redacted sidecar maps
  `source_record_sha256` to the canonical group/order key and the source row has
  the required native fields.

Evidence:

```text
runs/phase51z_readonly_native_role_capture/PHASE51Z-LIGHTER-UNLINKED-NATIVE-ROLE-SOURCE-HOLD-20260505T000000Z
runs/phase51v_forward_capture_bundle_readiness/PHASE51V-LIGHTER-UNLINKED-NATIVE-ROLE-SOURCE-HOLD-20260505T000000Z
gate_status: HOLD
Lighter retained source rows replayed: 1400
Lighter native-field-ready rows: 1400
sanitized unlinked Lighter source rows emitted: 531
target-linked Lighter source rows emitted: 0
Phase 5.1v native-role targets ready without sidecar: 0 / 287
Phase 5.1v Lighter native-limit targets ready: 0 / 3132
generated_phase51s_manifest_ready: false
raw_identifier_redaction_status: PASS
```

Next move:

```text
Build a validated redacted source-link sidecar for the 531 sanitized unlinked
Lighter source rows, or capture a fresh safe target-window Lighter trade source
whose native IDs match current canonical target groups. Keep native-limit
pressure separate and on HOLD until event-time sendTx plus REST-or-weighted
limit/remaining fields are observed from a non-live-authorized source.
```

## Phase 5.1z Source-Link Request Pack Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only the repo-owned source-link request-pack scaffold and
its empty-sidecar HOLD validation.

Rationale:

- Existing Phase 5.1t sidecars have `0` overlap with the `531` preserved
  Lighter source hashes, so no current artifact can safely mark those rows
  target-ready.
- The `531` source rows lack raw identity fields by design; a sidecar cannot be
  derived from them alone without inventing linkage.
- A request pack gives a reviewer the exact redacted source hashes, target
  canonical keys, allowed sidecar schema, and Phase 5.1v validation manifest
  without reopening raw identifiers.

Evidence:

```text
runs/phase51z_source_link_request_pack/PHASE51Z-LIGHTER-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z
runs/phase51v_forward_capture_bundle_readiness/PHASE51V-LIGHTER-SOURCE-LINK-REQUEST-PACK-EMPTY-SIDECAR-HOLD-20260505T000000Z
gate_status: HOLD
source_link_request_source_count: 531
source_link_request_target_count: 125
source_link_sidecar_template_row_count: 0
Phase 5.1v native-role targets ready with empty sidecar: 0 / 287
Phase 5.1v Lighter native-limit targets ready: 0 / 3132
generated_phase51s_manifest_ready: false
raw_identifier_redaction_status: PASS
```

Next move:

```text
Populate the request pack's proposed sidecar with validated redacted links only:
source_record_sha256 plus canonical_group_id or order_key. Rerun Phase 5.1v
against candidate_manifest_with_empty_sidecar.json after replacing the empty
sidecar path with the validated sidecar. If no sidecar can be produced, use a
bounded GET-only target-window Lighter /api/v1/trades capture as a diagnostic,
not as a promotion path.
```

## Phase 5.1 Lighter GET-Only Diagnostic Decision

Decision: `HOLD` for model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, financial claims, and 24/7 readiness.

Decision: `PROMOTE` only the fail-closed redaction hardening and the retained
HOLD-only diagnostic evidence path.

Rationale:

- The bounded GET-only Lighter `/api/v1/trades` diagnostic can be generated
  without live orders and now hashes raw order/trade/client/tx identifiers and
  cursor tokens before artifact write.
- The diagnostic produced native-role fields but no direct target linkage to
  the current Phase 5.1u Lighter targets.
- Repeating the same target-window diagnostic is not expected to reduce the
  blocker unless the target window or source surface changes.

Evidence:

```text
runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TARGET-WINDOW-GETONLY-DIAGNOSTIC-SANITIZED-20260505T143000Z
runs/phase51z_readonly_native_role_capture/PHASE51Z-LIGHTER-GETONLY-SANITIZED-DIAGNOSTIC-20260505T143000Z
runs/phase51v_forward_capture_bundle_readiness/PHASE51V-LIGHTER-GETONLY-SANITIZED-DIAGNOSTIC-20260505T143000Z
gate_status: HOLD
trade_count: 400
raw_identifier_redaction_status: PASS
raw_identifier_key_violation_count: 0
sanitized unlinked Lighter source rows: 400
Lighter target-linked rows: 0 / 125
Phase 5.1v native-role targets ready: 0 / 287
Phase 5.1v Lighter native-limit targets ready: 0 / 3132
generated_phase51s_manifest_ready: false
```

Next move:

```text
Obtain a deterministic redacted source-link sidecar for the existing request
pack, or a different non-live authorized Lighter source that directly links to
the current Phase 5.1u target hashes. Separately obtain event-time sendTx plus
REST-or-weighted limit/remaining pressure before claiming Lighter native-limit
readiness.
```

## Phase 5.1aa Lighter WebSocket Account Snapshot Decision

Decision: `PROMOTE` only the repo-owned non-live read-only source-surface
collector and evidence pack generation; `HOLD` for Phase 5.1 blocker reduction.

Scope:

- `tools/phase51aa_lighter_ws_account_trades_snapshot.py`
- official Lighter `account_all` and `account_all_trades` WebSocket channels
- existing local credentials only
- no order placement, cancellation, modification, sendTx, sendTxBatch, live
  orders, canary, capital escalation, risk-limit relaxation, model training, EV
  admission, or financial claims

Evidence:

```text
account_all run: runs/phase51aa_lighter_ws_account_trades_snapshot/PHASE51AA-LIGHTER-WS-ACCOUNT-ALL-SNAPSHOT-HOLD-20260505T000000Z
account_all_trades run: runs/phase51aa_lighter_ws_account_trades_snapshot/PHASE51AA-LIGHTER-WS-ACCOUNT-TRADES-PARTIALTIMEOUT-HOLD-20260505T000000Z
Phase 5.1z run: runs/phase51z_readonly_native_role_capture/PHASE51Z-LIGHTER-WS-SNAPSHOT-HOLD-20260505T000000Z
Phase 5.1v run: runs/phase51v_forward_capture_bundle_readiness/PHASE51V-LIGHTER-WS-SNAPSHOT-HOLD-20260505T000000Z
account_all message_count: 1
account_all trade_count: 0
account_all_trades message_count: 1
account_all_trades trade_count: 0
raw_identifier_redaction_status: PASS
native-role targets ready: 0 / 287
Lighter native-limit targets ready: 0 / 3132
clears_phase51_blockers: false
```

Board interpretation:

```text
accepted: Phase 5.1aa collector is a safe, tested, read-only account WebSocket
source-surface probe that preserves HOLD boundaries, writes message metadata
rather than private account payloads, and fails closed on raw identifier-like
output keys.

not accepted: any claim that Lighter native-role targets are recovered, any
claim that Lighter native-limit pressure is observed, or any live/canary/model
training/EV-admission promotion.
```

Next move:

```text
Do not repeat the same Lighter account WebSocket snapshot unless account
activity, target window, or source semantics change. Obtain a validated
redacted source-link sidecar for the existing Phase 5.1z Lighter request pack,
or capture a different target-linkable non-live Lighter native-role source.
Separately obtain event-time sendTx plus REST-or-weighted limit/remaining
pressure before claiming Lighter native-limit readiness.
```

## Phase 5.1z All-Venue Source-Link Request Pack Decision

Decision: `PROMOTE` only the venue-neutral source-link request-pack scaffold and
HOLD-only all-venue evidence packaging; `HOLD` for blocker reduction until a
validated redacted sidecar or directly target-linkable source exists.

Evidence:

```text
source run: runs/phase51z_readonly_native_role_capture/PHASE51Z-ALLVENUE-UNLINKED-NATIVE-ROLE-SOURCE-HOLD-20260505T000000Z
request pack: runs/phase51z_source_link_request_pack/PHASE51Z-ALLVENUE-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z
empty-sidecar validation: runs/phase51v_forward_capture_bundle_readiness/PHASE51V-ALLVENUE-SOURCE-LINK-REQUEST-PACK-EMPTY-SIDECAR-HOLD-20260505T000000Z
source_link_request_source_count: 2130
source_link_request_source_counts_by_venue: aster=228, extended=1579, lighter=300, paradex=23
source_link_request_target_count: 281
source_link_request_target_counts_by_venue: aster=113, extended=28, lighter=125, paradex=15
native-role targets ready with empty sidecar: 67 / 287
Lighter native-limit targets ready: 0 / 3132
raw_identifier_redaction_status: PASS
```

Board interpretation:

```text
accepted: the all-venue request pack converts the native-role linkage gap into
a deterministic redacted sidecar request across Aster, Extended, Lighter, and
Paradex without exposing raw identifiers or changing execution behavior.

not accepted: any inference that unlinked source hashes are target-ready, any
model-training/EV-admission promotion, any live/canary/capital/risk change, or
any Lighter native-limit pressure claim.
```

Next move:

```text
Populate a validated redacted source-link sidecar for the all-venue request
pack, or capture a different non-live authorized source that is directly
target-linkable to the current Phase 5.1u hashes. Separately obtain event-time
sendTx plus REST-or-weighted limit/remaining pressure for Lighter.
```

## Phase 5.1ab Lighter Native-Limit Pressure Source Preflight Decision

Decision: `PROMOTE` only the repo-owned HOLD-only local source preflight
scaffold; `HOLD` for blocker reduction until non-live-authorized observed
pressure rows are supplied.

Scope:

```text
tool: tools/phase51ab_lighter_native_limit_pressure_source.py
mode: local sanitized JSONL input only
network access: none
venue write paths: none
sendTx/sendTxBatch/nextNonce: prohibited
outputs: lighter_forward_native_limit_pressure_snapshot.jsonl, labels, Phase 5.1v candidate manifest
```

Board interpretation:

```text
accepted: Phase 5.1ab makes future sanitized Lighter event-time pressure rows
reviewable by Phase 5.1v and rejects raw identifiers, secret-shaped fields,
unsafe authorization flags, network paths, env files, and symlink chains.

not accepted: any claim that Lighter pressure has been observed by this tool,
any inference from GET-only caps, empty headers, docs-only limits, account
tiers, or empty WebSocket snapshots, or any model-training/EV-admission/live
promotion.
```

Validation:

```text
python3 -m py_compile tools/phase51ab_lighter_native_limit_pressure_source.py
python3 -m unittest tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51ab_lighter_native_limit_pressure_source_feeds_phase51v_without_false_clearance
```

Next move:

```text
Supply a non-live-authorized sanitized Lighter pressure JSONL with event-time
active-order headroom, sendTx limit/remaining, REST-or-weighted limit/remaining,
and event-time alignment, then run Phase 5.1ab and the emitted Phase 5.1v
validation command. Keep native-role source-link sidecar acquisition moving in
parallel.
```

## Phase 5.1ac Source-Link Reuse Audit Decision

Decision: `PROMOTE` only the repo-owned HOLD-only reuse audit scaffold and
HOLD evidence pack; `HOLD` for blocker reduction because no reusable sidecar
rows were found.

Evidence:

```text
tool: tools/phase51ac_source_link_reuse_audit.py
run: runs/phase51ac_source_link_reuse_audit/PHASE51AC-ALLVENUE-SOURCE-LINK-REUSE-AUDIT-HOLD-20260505T000000Z
request pack: runs/phase51z_source_link_request_pack/PHASE51Z-ALLVENUE-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z
existing_sidecar_file_count: 2
existing_sidecar_row_count: 574
source_link_request_source_count: 2130
reusable_source_link_count: 0
missing_source_link_count: 2130
missing_source_link_counts_by_venue: aster=228, extended=1579, lighter=300, paradex=23
clears_phase51_blockers: false

current wider request pack: runs/phase51z_source_link_request_pack/PHASE51Z-CURRENT-TARGET-WIDE-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z
current wider reuse audit: runs/phase51ac_source_link_reuse_audit/PHASE51AC-CURRENT-TARGET-WIDE-SOURCE-LINK-REUSE-AUDIT-HOLD-20260505T000000Z
current wider source_link_request_source_count: 2819
current wider reusable_source_link_count: 0
current wider missing_source_link_count: 2819
current wider missing_source_link_counts_by_venue: aster=784, extended=1579, lighter=300, paradex=156
```

Board interpretation:

```text
accepted: existing repo-owned sanitized sidecars do not overlap the all-venue
request-pack source hashes, and the native-role linkage blocker is now
machine-checkably non-derivable from those sidecars.

not accepted: any inference that missing links can be reconstructed without a
new validated redacted sidecar or directly target-linkable source, any
model-training/EV-admission promotion, or any live/canary/capital/risk change.
```

Next move:

```text
Obtain a new validated redacted source-link sidecar for the current-target wide
all-venue request pack, or capture a different non-live authorized
target-linkable native-role source. Keep the Lighter native-limit pressure path
separate through Phase 5.1ab.
```

## Phase 5.1ad Source-Link Sidecar Materialization Decision

Decision: `PROMOTE` only the repo-owned HOLD-only materializer scaffold;
`HOLD` for blocker reduction until a validated redacted mapping exists and
Phase 5.1v validates the materialized candidate manifest.

Evidence:

```text
tool: tools/phase51ad_source_link_sidecar_materialize.py
input contract: Phase 5.1z request pack + redacted mapping file
mapping fields: source_record_sha256 plus canonical_group_id or order_key
output: source_links.sanitized.jsonl and candidate_manifest_with_materialized_sidecar.json
clears_phase51_blockers: false
```

Board interpretation:

```text
accepted: the repo can now materialize and validate externally supplied
redacted source-link mappings without raw identifier persistence, link
inference, or manual manifest surgery.

not accepted: any claim that a source-link sidecar has been obtained for the
all-venue request pack, any blocker clearance without Phase 5.1v validation,
any model-training/EV-admission promotion, or any live/canary/capital/risk
change.
```

Next move:

```text
Use Phase 5.1ad only after a validated redacted mapping is available for the
current-target wide request pack. Until then, continue safe target-linkable
source capture and Lighter native-limit pressure acquisition without repeating
exhausted GET-only/header/empty-WS paths.
```

## Phase 5.1ae Candidate Manifest Composition Decision

Decision: `PROMOTE` only the HOLD-only candidate-manifest composition gate.

Rationale:

- Phase 5.1ad produces a materialized manifest for the current-target wide
  source-link request pack, but all-five validation also needs already accepted
  Hyperliquid native-role source and, later, any Lighter native-limit pressure
  manifest.
- Prior combined manifests existed only as run artifacts. Phase 5.1ae makes
  this composition deterministic, tested, and repo-owned.
- The gate does not infer source links, does not create native-role evidence,
  and does not clear Phase 5.1 blockers by itself.

Evidence:

```text
tool: tools/phase51ae_candidate_manifest_compose.py
input contract: local Phase 5.1v candidate manifest(s), optional local sources, optional local source-link artifacts
output: candidate_manifest.composed.json for Phase 5.1v
real composition run: runs/phase51ae_candidate_manifest_compose/PHASE51AE-CURRENT-TARGET-WIDE-PLUS-HYPERLIQUID-COMPOSE-HOLD-20260505T000000Z
Phase 5.1v validation: runs/phase51v_forward_capture_bundle_readiness/PHASE51V-CURRENT-TARGET-WIDE-PLUS-HYPERLIQUID-COMPOSE-HOLD-20260505T000000Z
native-role targets ready: 73 / 287
Lighter native-limit targets ready: 0 / 3132
clears_phase51_blockers: false
```

Board interpretation:

```text
accepted: post-materialization validation can now compose the current-target
wide materialized manifest with Hyperliquid and future Lighter-pressure
manifests without manual manifest surgery.

not accepted: blocker clearance, source-link inference, native-role evidence
creation, Lighter native-limit pressure claims, model-training/EV-admission
promotion, or any live/canary/capital/risk change.
```

Next move:

```text
When a validated redacted mapping exists, run Phase 5.1ad, compose the
materialized candidate manifest with the Phase 5.1x Hyperliquid source through
Phase 5.1ae, and then rerun Phase 5.1v. Add any future Phase 5.1ab Lighter
pressure manifest through Phase 5.1ae before all-five readiness review.
```

## 2026-05-05 - Phase 5.1af Local Source Retrieval Audit Decision

Decision: `PROMOTE` only the HOLD-only local exhaustion audit as a repo-owned
negative-proof gate.

Rationale:

- The board needed a deterministic answer to whether the existing local request
  pack, bounded telemetry artifacts, and runtime logs could retrieve the two
  remaining artifact classes without source-owner intervention.
- Repeating read-only Lighter GET/WS probes or broad local scans without a
  material source change creates operational churn without reducing blockers.
- The audit preserves the no-live boundary and makes the stop condition
  evidence-owned instead of narrative-owned.

Evidence:

```text
tool: tools/phase51af_local_source_retrieval_audit.py
doc: docs/PHASE5_1AF_LOCAL_SOURCE_RETRIEVAL_AUDIT.md
run: runs/phase51af_local_source_retrieval_audit/PHASE51AF-LOCAL-SOURCE-RETRIEVAL-AUDIT-HOLD-20260505T000000Z
request source rows scanned: 2819
request targets scanned: 281
bounded telemetry hashes match: true
runtime log scans complete: true
source_link_retrieval_status: MISSING_REQUIRED_LINKAGE
lighter_pressure_retrieval_status: MISSING_REQUIRED_PRESSURE_FIELDS
runtime_log_pattern_status: NO_USABLE_PRESSURE_PATTERN
local_retrieval_possible_without_inference: false
clears_phase51_blockers: false
```

Board interpretation:

```text
accepted: existing inspected local files are exhausted for deterministic
source-link mapping and Lighter event-time native-limit pressure retrieval.

not accepted: treating raw identifier field-name presence, native-role field
presence in unlinked request sources, runtime rate_limit strings, or log pattern
hits as source-link or pressure evidence.
```

Next move:

```text
Obtain a validated redacted mapping for the current-target wide request pack
and materialize it through Phase 5.1ad, then compose through Phase 5.1ae and
rerun Phase 5.1v. Separately obtain sanitized Lighter event-time pressure rows
and stage them through Phase 5.1ab before composing and validating.
```

## 2026-05-06 - Phase 5.1ai Non-Lighter Order-History Bridge Decision

Decision: `PROMOTE` only the HOLD-only diagnostic tooling and negative
evidence. `BLOCK` further autonomous local capture loops unless a materially
different source surface or new account activity is available.

Rationale:

- Extended and Paradex official docs expose read-only order-history surfaces
  with client/external identifiers, so one bounded diagnostic was justified.
- The diagnostic preserved the source-link contract: a native trade/fill row
  had to hash to a current request-pack `source_record_sha256` and bridge by a
  deterministic order key to a uniquely target-matched order-history row.
- The real capture emitted zero materializable links; therefore this lane does
  not reduce the Phase 5.1 native-role blocker.

Evidence:

```text
tool: tools/phase51ai_non_lighter_order_history_bridge_diagnostic.py
test: tests/test_phase51ai_non_lighter_order_history_bridge_diagnostic.py
run: runs/phase51ai_non_lighter_order_history_bridge_diagnostic/PHASE51AI-NON-LIGHTER-ORDER-HISTORY-BRIDGE-DIAGNOSTIC-HOLD-20260506T000000Z
extended native rows: 1601
extended order-history rows: 1664
extended request-source overlap: 1579
extended order-history target matches: 21
paradex native rows: 30
paradex order-history rows: 668
paradex request-source overlap: 23
paradex order-history target matches: 15
materializable_source_link_count: 0
raw_identifier_redaction_status: PASS
clears_phase51_blockers: false
```

Next move:

```text
Do not repeat Phase 5.1ai without changed source semantics, target window,
retained source rows, account activity, or auth material. Obtain the validated
redacted source-link mapping or directly target-linkable source required by the
current-target wide request pack, and separately obtain Lighter event-time
pressure rows for Phase 5.1ab.
```
