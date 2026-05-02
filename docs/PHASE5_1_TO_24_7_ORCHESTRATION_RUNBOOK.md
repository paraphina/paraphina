# Phase 5.1 To 24/7 Orchestration Runbook

Date: 2026-05-01

Status: `HOLD` for live/canary/capital. `PROMOTE` only for continuing the
Phase 5.1 non-live evidence workflow.

Branch: `main`

Pre-Phase-5.1b commit: `f047e4ee0adbf2ab8a2e6914c9a3872a8a4e8abb`

This document preserves the autonomous orchestration workflow so another
orchestrator can resume if the active chat is lost. It is an execution-control
document, not live-trading authorization.

## Non-Negotiables

- Do not place live orders from Phase 5.1b evidence.
- Do not start a canary from Phase 5.1b evidence.
- Do not escalate capital from Phase 5.1b evidence.
- Do not relax risk limits from Phase 5.1b evidence.
- Do not treat telemetry PnL as financial authority.
- Do not treat `sendTx` acceptance as execution finality.
- Do not infer maker/taker status from intent; require venue-native evidence.
- Direct venue truth overrides telemetry when they disagree.
- Every stage ends with exactly one verdict: `HOLD`, `PROMOTE`, or `ROLLBACK`.

## Current Critical Path

1. Keep Phase 5.1 non-live. Do not start canary, live orders, capital
   escalation, risk-limit relaxation, model training, or EV admission.
2. Treat Phase 5.1g as the current evidence boundary. It proves the canonical
   P-fill quarantine can be split into observed terminal labels and excluded
   review-only labels without converting censored lifecycle ambiguity into
   false negative outcomes.
3. The current Phase 5.1g quarantine review pack remains `HOLD`: `6140`
   canonical P-fill groups, `4527` observed terminal outcomes, `461` fills,
   `4066` terminal not-filled groups, and `1613` excluded quarantine/review
   groups.
4. The Phase 5.1g observed-only P-fill calibration-readiness rerun remains
   `HOLD` with reason `pfill_calibration_sparse_buckets`: `4527` observed
   outcomes, `461` fills, `4066` terminal not-filled groups, `0` censored
   labels, and sparse venue/side bucket coverage.
5. Next repo-owned move is feature-rich observed-only P-fill calibration
   review plus venue-native reconciliation for excluded quarantine categories.
   Do not train models from 5.1g.
6. Proceed to calibrated EV shadow only after canonical outcomes, maker/taker
   role gaps, holdout splits, feature completeness, venue-native truth gaps,
   and quarantine exclusion policy are accepted.

## Board

| Role | Mandate | Output |
|---|---|---|
| Executive Orchestrator | Sequence gates, enforce scope, integrate findings, update docs, commit only clean state. | Gate verdicts, commits, final reports. |
| Quant/Evidence Lead | EV calibration, P-fill, markout, queue/churn, tail cost, confidence bounds, overfitting controls. | Calibration plan, holdout results, EV admission verdict. |
| Execution/Venue Lead | Lighter and later venue-specific maker-only, finality, replace, latency, fees, native limits. | Venue contract evidence and execution-readiness verdict. |
| Risk/Ops/SRE Lead | Kill switch, residual state, double-action prevention, direct venue truth, rollback-to-flat, alerts, secrets, retention. | Operational readiness verdict and incident-drill evidence. |
| Systems Implementer | Repo patches, schema, tests, evidence tooling, CI hygiene. | Minimal diffs with validation results. |
| Independent Auditor | Challenge scope, evidence, safety, and next-move optimality after each material patch/evidence pack. | `HOLD`/`PROMOTE`/`ROLLBACK` challenge memo. |

## Stage Gates

### Gate 5.1b - Lighter Account/Native-Limit Evidence

Allowed:

- Read-only Lighter account/native-limit collection.
- Captured JSON replay through `tools/phase51b_lighter_account_limits.py`.
- Authenticated read-only HTTP GETs through the same tool.

Required evidence:

- `V2_LIGHTER_ACCOUNT_PROFILE`
- `V2_LIGHTER_ACCOUNT_LIMITS`
- `V2_LIGHTER_ACTIVE_ORDERS`
- Optional `V2_LIGHTER_TRADE_ATTRIBUTION_SAMPLE`
- Sanitized source snapshots.
- `manifest.json`
- `evidence_pack/artifact_index.json`
- `tools/check_telemetry_contract.py` validation output.

Promote condition:

- `phase51b_capture_complete=true`
- Schema v2 validates.
- Account limits and active orders are present.
- No unsafe authorization flags.
- No unredacted secrets in artifacts.

Hold or rollback condition:

- Missing account limits or active orders.
- Unsafe spec flags.
- Any `sendTx`, live-order, canary, capital, or risk-relaxation path.
- Unredacted secrets.
- Attempt to use the pack as financial or live authority.

### Gate 5.1c - Calibration Label Lake

Required evidence:

- Immutable quote-decision labels.
- Order lifecycle labels.
- Fill labels with maker/taker role or explicit unknown state.
- Multi-horizon markout labels.
- Queue age, cancel/replace, churn, and native-limit pressure labels.
- Balance snapshots sufficient to separate financial truth from diagnostics.
- Deterministic train/holdout splits.

Promote condition:

- Quote decision -> order lifecycle -> fill/markout -> balance delta can be
  joined deterministically.
- Sparse buckets are marked `HOLD` or pooled with explicit uncertainty
  inflation.

Current repo-owned tooling:

- `tools/phase51c_label_lake.py` builds quote-decision and order-lifecycle
  labels from non-live EV shadow output.
- `tools/phase51c_observed_labels.py` builds read-only observed fill and
  fair-value markout, and balance-reconciliation labels from existing artifacts
  without copying large source telemetry. It can optionally join Lighter native
  trade snapshots by order/client IDs to upgrade maker/taker attribution from
  `UNKNOWN` to `MAKER`/`TAKER`.
- `tools/phase51c_join_holdout.py` builds a source-aligned deterministic
  quote/order/fill/markout/balance join pack and immutable train/holdout split.
  It requires the label lake and observed-label pack to share the same
  `source_telemetry_sha256`.
- Current 5.1c evidence joins all `356` observed fills to orders, all `356`
  fills to markouts, `189` fills to Lighter quote candidates, and creates a
  deterministic `295` train / `61` holdout split.
- `tools/phase51c_lighter_trade_backfill.py` collects paginated read-only
  Lighter native trades for run-window-aligned maker/taker attribution. It uses
  only the official `trades` read endpoint and remains HOLD-only evidence.
- Current run-window Lighter backfill improved native-role attribution to
  `64` maker and `32` taker labels. The remaining unknowns are `93` Lighter
  fills whose telemetry IDs were absent from native trade history, plus `167`
  non-Lighter fills outside this Lighter-only quote-candidate lane.
- A second balance-backed terminal-stale 7200s artifact adds `117` Lighter
  candidate joins and `80` native-role-attributed complete joins. Across the
  two balance-backed terminal-stale lanes, current Lighter-only 5.1c evidence
  has `306` Lighter candidate joins and `176` native-role-attributed complete
  joins.
- `tools/phase51c_pfill_outcome_labels.py` builds order-level
  `ORDER_PFILL_OUTCOME_LABEL` records from label-lake and join-holdout packs.
  It uses order identity/source/hash keys rather than decision-id-first mapping,
  emits independent order-level train/holdout splits, and labels place orders
  as filled, terminal not-filled, or censored/unobserved.
- Current P_fill outcome evidence across the two balance-backed lanes:
  `11935` place-order labels, `489` filled, `2835` terminal not-filled, and
  `8611` censored/unobserved. This is sufficient as a repo-owned outcome-label
  scaffold but not sufficient for model training because censoring and sparse
  observed fills remain unresolved.
- `tools/phase51c_pfill_calibration_readiness.py` consumes one or more
  P_fill outcome packs and emits HOLD-only calibration-readiness records. It
  preserves existing `order_holdout_split` assignments, writes an order split
  manifest, computes Wilson 95% fill-rate intervals on observed non-censored
  labels, and refuses training/live/canary/economic authorization.
- Current P_fill calibration-readiness evidence across the same two lanes:
  `11935` order labels, `3324` observed outcomes, `489` filled, `2835`
  terminal not-filled, `8611` censored, `669` holdout observed outcomes, and
  `16` venue/side buckets including the global bucket. Gate remains `HOLD`
  because `72.14914118139925%` of labels are censored and feature-rich
  calibration requirements are not met.
- `tools/phase51c_queue_churn_labels.py` emits HOLD-only order-level
  queue/churn proxy labels by joining P_fill order labels back to lifecycle
  events. It records replace/cancel churn, queue-reset proxy counts, terminal
  horizons where available, and can join accepted Phase 5.1b Lighter native
  context while keeping all live/canary/economic approvals false.
- Current queue/churn evidence across the same two lanes: `11935` labels,
  `11935` lifecycle joins, `0` lifecycle misses, `4379` orders with churn,
  `1777` orders with replace/queue-reset proxy events, `3997` orders with
  cancel events, and `2906` orders with terminal horizons. The accepted
  Phase 5.1b Lighter native pack upgrades `6148` Lighter rows to
  `PARTIAL_ACTIVE_ORDER_COUNT_OBSERVED_LIMIT_UNKNOWN`; `5787` non-Lighter rows
  remain venue-native-limit unknown. Gate remains `HOLD`.
- `tools/phase51c_markout_calibration_readiness.py` consumes one or more
  observed-label and deterministic-join/holdout run pairs, validates matching
  `source_telemetry_sha256` and clean-baseline commit provenance, preserves the
  join-provided fill `TRAIN`/`HOLDOUT` split, and emits HOLD-only
  adverse-selection/markout readiness buckets. It refuses training, live,
  canary, capital, risk-relaxation, EV-admission, and financial authority.
- Current markout readiness evidence across the canonical `025435Z` and
  `FROM-BACKFILL-073231Z` lanes: `549` fills, `2196` markout rows, four
  horizons (`100`, `500`, `1000`, `5000` ms), `448` train fills, `101` holdout
  fills, `141` buckets including split-specific buckets, `780` adverse markout
  rows, and mean signed markout PnL `-0.040162968947815805`. Gate remains
  `HOLD` with reason `markout_readiness_sparse_buckets`; unresolved blockers
  include `373` fills with unknown maker/taker role, `243` fills with missing
  candidate joins, fair-value-only future reference prices, and sparse
  venue/side/horizon/split holdout coverage.
- 5.1c remains `HOLD`. No model training, live, canary, capital escalation,
  risk-limit relaxation, or financial claim is authorized from the current
  pack.

### Gate 5.1d - Evidence Quality Triage

Current repo-owned tooling:

- `tools/phase51c_pfill_censoring_audit.py` consumes one or more P-fill
  outcome packs, preserves existing `order_key` and `order_holdout_split`
  assignments, validates source hashes and unsafe flags, and classifies
  censored labels without converting them into observed negatives.
- `tools/phase51c_lighter_attribution_gap_audit.py` compares observed Lighter
  fills to sanitized native trade backfill and explains remaining unknown
  maker/taker labels without inferring role from quote intent.

Current 5.1d evidence:

- P-fill censoring audit across the canonical two lanes: `11935` order labels,
  `3324` observed outcomes, `489` filled, `2835` terminal not-filled, `8611`
  censored, censor rate `0.7214914118139925`, and `141` buckets. All censored
  labels classify as `NO_TERMINAL_EVENT_WITH_SUFFICIENT_WINDOW`, so the blocker
  is not merely end-of-window truncation. Gate remains `HOLD`.
- Lighter attribution gap audit on the canonical `FROM-BACKFILL-073231Z` lane:
  `189` Lighter fills, `96` already native-attributed fills, and `93` unknown
  fills classified as `NO_NATIVE_TRADE_MATCH` against the current 300-trade
  backfill. No stale unknowns are upgradable from existing native identity
  evidence. Gate remains `HOLD`.
- 5.1d promotes only the next non-live evidence step. It does not authorize
  model training, EV admission, live orders, canary, capital escalation,
  risk-limit relaxation, or financial claims.

### Gate 5.1e - Lifecycle/Native Truth Audit

Current repo-owned tooling:

- `tools/phase51e_lifecycle_truth_audit.py` consumes one or more P-fill
  outcome packs, their source label-lake and join-holdout packs, and the
  Lighter attribution gap audit. It canonicalizes order lifecycle identity
  across place intent/ack aliases, order/client hashes, decision IDs, fill
  joins, replace chains, and cancel-all scope without rewriting P-fill labels.
- The same tool re-reads raw captured Lighter telemetry IDs from the observed
  label source and compares them to sanitized native Lighter trades. It emits
  raw-native match status without printing raw IDs or secrets.

Current 5.1e evidence:

- Run:
  `runs/phase51e_lifecycle_truth_audit/PHASE51E-LIFECYCLE-TRUTH-AUDIT-TWO-LANE-20260502T000000Z`
- P-fill rows audited: `11935`.
- Current P-fill state entering 5.1e: `489` filled, `2835` terminal
  not-filled, `8611` censored.
- Canonical lifecycle status counts: `489` `STAYS_FILLED`, `2835`
  `STAYS_NOT_FILLED`, `464` `CENSORED_TO_CANONICAL_FILLED_REVIEW`, `5206`
  `CENSORED_TO_CANONICAL_NOT_FILLED_REVIEW`, `288`
  `CENSORED_TO_REPLACE_CHAIN_REVIEW`, `2270`
  `DUPLICATE_PLACE_ALIAS_COLLAPSE_REVIEW`, `375` `CANCEL_ALL_SCOPE_REVIEW`,
  and `8` `REMAINS_NO_TERMINAL_EVENT_WITH_SUFFICIENT_WINDOW`.
- Raw Lighter native truth: `189` raw Lighter fills audited, `96`
  `MATCHED_NATIVE_ID`, `93` `NATIVE_WINDOW_COVERED_NO_MATCH`; native roles are
  `64` maker, `32` taker, `93` unknown.
- Gate remains `HOLD`. The evidence supports a canonical P-fill outcome
  rebuild/review gate, not model training or EV admission.

5.1e artifacts:

```text
861fd63459957d1b6508c19ccdda020c4a2b4b40a60465457d7ad89d131a8f66  lifecycle_truth_audit_summary.json
82fad77b173825ccca0650368a9d68426c5e6d80c08ef5d05d0203ffbea189ce  order_lifecycle_truth_labels.jsonl
fe099f177ae981c81a41a75a6c757d1960a2331221bdf439bab5cecbdc0903ff  lighter_native_identity_gap_labels.jsonl
3892c024a8e6d60d766eece7d0c7effb9125801c9c18a2bac1b599735c487d11  lighter_raw_native_truth_labels.jsonl
6db64ea422d445cc63d32505577dc71b0e72003b533374ae35a25c1dd70ced50  manifest.json
afcf0fbce4268b96f4ee7a2a848e8950671e570fa7bc74a44f87e065a1ce684c  evidence_pack/artifact_index.json
```

### Gate 5.1f - Canonical P-Fill Outcome Rebuild/Review

Current repo-owned tooling:

- `tools/phase51f_canonical_pfill_outcome_rebuild.py` consumes the Phase 5.1e
  lifecycle truth audit and the source Phase 5.1c P-fill outcome packs. It
  emits one `ORDER_PFILL_OUTCOME_LABEL` per canonical P-fill lifecycle group,
  writes a source-to-canonical manifest, records split conflicts, quarantines
  unresolved review groups, and keeps all live/canary/capital/risk/model
  authorizations false.
- The tool is additive. It does not mutate Phase 5.1c or Phase 5.1e artifacts,
  submit orders, train models, approve EV admission, or make financial claims.

Current 5.1f evidence:

- Run:
  `runs/phase51f_canonical_pfill_outcome/PHASE51F-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z`
- Source P-fill rows accounted for: `11935`.
- Canonical P-fill groups emitted: `6140`.
- Lifecycle-graph canonical groups from 5.1e: `6999`; the `-859` difference is
  expected because 5.1f groups only canonical P-fill source rows.
- Canonical review outcomes: `461` `CANONICAL_OBSERVED_FILLED`, `4066`
  `CANONICAL_OBSERVED_NOT_FILLED`, and `1613`
  `CANONICAL_REVIEW_QUARANTINED`.
- Split conflicts from old per-order splits: `1673`; 5.1f assigns new
  deterministic group-level splits.
- Gate remains `HOLD` with reason
  `phase51f_canonical_pfill_contains_quarantined_review_groups`.

5.1f canonical artifacts:

```text
5df9223e2c617569eb3d8d3c5359558517907856341472f1e225a1046cb49f23  canonical_pfill_order_labels.jsonl
89e1835e6b3a090a99481d46b6287b5e22f7e5b466cff7b7b8eea8e75d1db187  canonical_pfill_outcome_summary.json
cee129458060843872a363cd54a32018781771736bcdc8129f99bce7bd181ac2  source_to_canonical_order_manifest.jsonl
3a73e53e3f0418b16ccaa712f4181c75a1c64d0870b36102be5a165ef2f4772a  split_conflict_manifest.jsonl
69c3881505b812238d7132fda4f7ab09a21600bb42b3d6bdbaae5facf57a5a36  quarantined_review_labels.jsonl
fe67edf1cd28fbce2291d193ce86e9a6c397fe99eadeae472b2127a5f331e137  evidence_pack/artifact_index.json
```

P-fill readiness rerun from canonical labels:

- Run:
  `runs/phase51f_pfill_calibration_readiness_from_canonical/PHASE51F-PFILL-CALIBRATION-READINESS-FROM-CANONICAL-TWO-LANE-20260502T000000Z`
- Order labels: `6140`.
- Observed outcomes: `4527`.
- Filled: `461`.
- Not-filled: `4066`.
- Censored/quarantined: `1613`.
- Censored rate: `0.26270358306188923`.
- Holdout observed outcomes: `902`.
- Gate remains `HOLD` with reason `pfill_calibration_contains_censored_orders`.

### Gate 5.1g - Canonical P-Fill Quarantine Review

Current repo-owned tooling:

- `tools/phase51g_pfill_quarantine_review.py` consumes the Phase 5.1f
  canonical P-fill outcome pack and emits a quarantine-review pack that
  preserves all canonical groups while producing a separate observed-only
  compatibility pack for diagnostic downstream calibration-readiness reruns.
- The tool does not mutate Phase 5.1f artifacts, submit orders, train models,
  approve EV admission, approve live/canary/capital/risk changes, or make
  financial claims.

Current 5.1g evidence:

- Run:
  `runs/phase51g_pfill_quarantine_review/PHASE51G-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z`
- Canonical P-fill groups reviewed: `6140`.
- Observed terminal groups: `4527`.
- Filled groups: `461`.
- Terminal not-filled groups: `4066`.
- Excluded quarantine/review groups: `1613`.
- Exclusion reasons: `1135` `EXCLUDED_DUPLICATE_ALIAS_NO_TERMINAL`, `375`
  `EXCLUDED_CANCEL_ALL_SCOPE_REVIEW`, `95`
  `EXCLUDED_REPLACE_CHAIN_REVIEW`, and `8`
  `RIGHT_CENSORED_NO_TERMINAL`.
- Venue quarantine counts: `lighter=844`, `hyperliquid=339`,
  `extended=236`, `aster=170`, and `paradex=24`.
- Gate remains `HOLD` with reason
  `phase51g_quarantine_review_observed_only_diagnostic_pack`.

5.1g quarantine artifacts:

```text
de3cf674a26ae01a9189d86f15b52617a41e9889215552161a0050217602e8ec  binary_observed_pfill_order_labels.jsonl
de3cf674a26ae01a9189d86f15b52617a41e9889215552161a0050217602e8ec  observed_only_pfill_outcome/pfill_order_labels.jsonl
2121788d7b50ee593dc57a4a92daae668514bf45e52cda214dfa700a96e794fe  observed_only_pfill_outcome/pfill_outcome_summary.json
be33fab71e636c9801e6c083f9b0aee432ba05cf2c59170ccd829aaf09564f8b  quarantine_review_labels.jsonl
bf07596603805c5706e5bbb11c9aebb807d0b71ccb026893f826df177c878694  quarantine_review_summary.json
4a27d7b5772e3d49a58debd269dd1a426b9fb6958a776fc5f6dd4cd6461a2067  source_reconciliation_manifest.jsonl
```

Observed-only P-fill readiness rerun:

- Run:
  `runs/phase51g_pfill_calibration_readiness_observed_only/PHASE51G-PFILL-CALIBRATION-READINESS-OBSERVED-ONLY-TWO-LANE-20260502T000000Z`
- Input observed-only pack:
  `runs/phase51g_pfill_quarantine_review/PHASE51G-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z/observed_only_pfill_outcome`
- Order labels: `4527`.
- Observed outcomes: `4527`.
- Filled: `461`.
- Terminal not-filled: `4066`.
- Censored/quarantined: `0`.
- Train observed outcomes: `3625`.
- Holdout observed outcomes: `902`.
- Buckets: `12`, including global.
- Gate remains `HOLD` with reason `pfill_calibration_sparse_buckets`.

5.1g observed-only readiness artifacts:

```text
19330c451fe84a476ef2a42584e49854bb8e9534540c10b11ed2d34a42eaecfd  pfill_calibration_buckets.jsonl
219f35fee50f764a11aad97d5ce37b5c82d3f7b8d6d2675f77ef71b745cb7774  pfill_calibration_readiness_summary.json
ffecbfa3254c0b940d37a14651f97a0ae711678689fa6bcaf4aebeb7e75d2d5d  pfill_order_split_manifest.jsonl
```

### Gate 5.2 - Calibrated EV Shadow

Required evidence:

- P-fill model with calibrated confidence intervals.
- Markout/adverse-selection model.
- Queue reset and churn cost model.
- Residual/tail cost model.
- Frozen feature/config hash.
- Rolling-origin or regime-separated holdout.

Promote condition:

- Holdout lower-confidence-bound EV is positive after fees, churn, queue loss,
  funding/capital, residual, and tail costs.
- Candidate policy beats baseline under pre-registered metrics.
- No post-hoc threshold tuning on promotion data.

### Gate 5.3 - Native Lighter Execution Proof

Required evidence:

- Testnet/paper non-crossing post-only rests as maker.
- Deliberately crossing post-only cancels/rejects and does not take liquidity.
- IOC cleanup behavior is separate from maker quoting.
- `client_order_index -> order_index -> status/fill/cancel` reconciliation.
- Maker/taker attribution from venue-native fields.
- Cancel+new versus modify latency and queue-reset evidence.

Promote condition:

- `sendTx` success is not treated as finality.
- Every tested lifecycle reaches a reconciled terminal state.
- Replace/modify cannot leak taker behavior under the intended account mode.

### Gate 5.4 - Risk/Ops Hardening

Required evidence:

- Kill switch latches and blocks all new actions.
- Cancel-all and cleanup paths are bounded.
- Residual inventory states are event-sourced.
- Double-action prevention covers MM, fast hedge, periodic hedge, exit, flatten,
  and reconcile-triggered actions.
- Rollback-to-flat produces two clean direct venue audits.
- Log retention, alerts, restart-loop handling, and secrets rotation are
  deployable and tested.

Promote condition:

- Rollback is complete only when venues are flat and open orders are zero, not
  merely when the service returns to shadow.

### Gate 5.5 - Micro-Canary Board Packet

Required evidence:

- Frozen exact commit and config.
- Venue scope, instrument, capital cap, duration, abort rules, rollback plan.
- Pre/post direct venue audits.
- Pre/post balance snapshots.
- Alert and on-call readiness.

Promote condition:

- Board packet accepted separately. No automatic canary promotion from
  non-live evidence.

### Gate 5.6 - Supervised Live Ladder

Required evidence:

- Short canary.
- Longer canary.
- Supervised live windows.
- Repeated long soaks across independent regimes.
- Balance-authoritative nonnegative economics after fees, funding, churn,
  cleanup, and tail reserve.

Promote condition:

- No kill events, no reconcile drift, no dirty direct venue audits, no manual
  cleanup dependence, no alert gaps, and no calibration drift breach.

### Gate 24/7 - Production Certification

Required evidence:

- Repeated clean independent windows.
- Unattended SLOs met.
- Incident drills passed.
- Rollback-to-flat automatic or operationally bounded for every active venue.
- Direct venue truth and balance authority agree within pre-registered
  tolerance.

Promote condition:

- The system can be left running under approved guard policy with no ad hoc
  rescue dependence and positive attributable economics.

## Audit Cadence

Run an independent audit after:

- Every material code/schema/doc patch.
- Every evidence-pack run.
- Every proposed commit.
- Every gate verdict.
- Every attempted promotion.

The audit must answer:

- Did any change touch live/canary/capital/risk behavior?
- Are artifacts reproducible from commit, config, input, and command?
- Did schema/docs/tests pass?
- Is the next move still the highest-leverage blocker?
- Is the current verdict `HOLD`, `PROMOTE`, or `ROLLBACK`?

## Resume Procedure

1. `cd /home/ubuntu/paraphina_mm_pnl_harness`
2. `git status --short`
3. `git rev-parse HEAD`
4. Read this document.
5. Read `ROADMAP.md` Phase 5.1b gate.
6. Read `docs/PHASE5_1_BOARD_DECISION.md` next-move section.
7. Confirm runtime is still shadow before any evidence capture:

```bash
systemctl show paraphina_live --property=ActiveState,SubState,NRestarts
curl -fsS http://127.0.0.1:9898/health/detail
```

8. Do not proceed to live/canary/capital work unless a later committed board
   decision explicitly supersedes this document.

## Next Command Targets

After this document and Phase 5.1b code are committed and pushed, produce the
first real Phase 5.1b pack:

```bash
python3 tools/phase51b_lighter_account_limits.py \
  --env-file /home/ubuntu/paraphina/deploy/env/all5_recover_20260314.env \
  --fetch-readonly \
  --include-trades \
  --allow-sdk-auth \
  --lighter-sdk-path /tmp/lighter_sdk \
  --run-id PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-<utc>
```

If `LIGHTER_AUTH_TOKEN` is not present, either set a short-lived read-only auth
token or use `--allow-sdk-auth` to derive one from existing Lighter API key env.
If the SDK is not installed, provide an explicit `--lighter-sdk-path`; the
collector does not auto-import SDK code from `/tmp`, and the supplied SDK path
and `lighter/` package tree must be owned by root/current user, non-symlinked,
and not group/world writable. The collector loads `--env-file` without shell
execution. Do not print secrets. Do not use funded-main private keys. Do not use
any sendTx path.

Validate the pack:

```bash
python3 tools/check_telemetry_contract.py \
  runs/phase51b_lighter_account_native_limits/<run_id>/telemetry.jsonl
```

Record the result in `docs/PHASE5_1_EVIDENCE_LOG.md`.

## Current Verdict

`HOLD` for live, canary, capital escalation, risk-limit relaxation, and 24/7
production readiness.

`PROMOTE` only for completing the Phase 5.1b non-live evidence workflow.
