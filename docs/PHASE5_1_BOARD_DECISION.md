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

As of the Phase 5.1g canonical P-fill quarantine review gate, the board
decision remains `PROMOTE_FOR_NEXT_NONLIVE_STEP` only. The current blockers are
now more specific:

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
- 5.1 remains `HOLD` for model training, EV admission, live orders, canary,
  capital escalation, risk-limit relaxation, and financial claims until the
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
| Multi-venue launch | Phase 5.1 evidence is Lighter-first and venue-local. |
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
Phase 5.1g - Canonical P-fill quarantine review and downstream rerun gate
```

Completed non-live scope:

- Reviewed the `1613` quarantined canonical P-fill groups from Phase 5.1f.
- Kept duplicate-only, cancel-all-scope, replace-chain, and true no-terminal
  cases out of the observed-only diagnostic P-fill pack.
- Re-ran P-fill calibration readiness only on observed terminal outcomes where
  the canonical contract was accepted.
- Emitted only `HOLD` records with `no_live_flag=true` and all live, canary,
  capital, risk-relaxation, EV-admission, model-training, and financial-claim
  authorization fields false.

Repo-owned artifacts:

- Rebuild tool: `tools/phase51f_canonical_pfill_outcome_rebuild.py`
- Quarantine review tool: `tools/phase51g_pfill_quarantine_review.py`
- Canonical pack:
  `runs/phase51f_canonical_pfill_outcome/PHASE51F-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z`
- Quarantine review pack:
  `runs/phase51g_pfill_quarantine_review/PHASE51G-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z`
- Observed-only readiness rerun:
  `runs/phase51g_pfill_calibration_readiness_observed_only/PHASE51G-PFILL-CALIBRATION-READINESS-OBSERVED-ONLY-TWO-LANE-20260502T000000Z`

## Next Move

The next optimal move after Phase 5.1g is still not live trading. It is a
non-live observed-only P-fill calibration review:

- Build feature-rich P-fill diagnostics against the observed-only terminal
  pack without using excluded quarantine rows as negatives.
- Quantify venue/side/layer/regime sparsity, missing observed-horizon fields,
  maker/taker role gaps, and model-feature availability.
- Reconcile excluded quarantine categories only with deterministic
  venue-native evidence; otherwise keep them excluded from training.
- Preserve all Phase 5.1 holds above.
