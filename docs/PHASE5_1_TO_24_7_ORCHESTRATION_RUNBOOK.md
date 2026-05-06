# Phase 5.1 To 24/7 Orchestration Runbook

Date: 2026-05-01

Status: `HOLD` for live/canary/capital. `PROMOTE` only for continuing the
Phase 5.1 non-live evidence workflow.

Branch: `main`

Pre-Phase-5.1b commit: `f047e4ee0adbf2ab8a2e6914c9a3872a8a4e8abb`

This document preserves the autonomous orchestration workflow so another
orchestrator can resume if the active chat is lost. It is an execution-control
document, not live-trading authorization.

V2 target-spec authority: `docs/V2_SPECIFICATION.md` defines the repo-owned
Phase 5.1/V2 fill-aware, hedge-aware, arbitrage-informed target requirements.
This runbook controls orchestration and evidence sequencing only; it does not
promote V2 behavior beyond the gates in `ROADMAP.md`.

## Non-Negotiables

- Do not place live orders from Phase 5.1 evidence.
- Do not start a canary from Phase 5.1 evidence.
- Do not escalate capital from Phase 5.1 evidence.
- Do not relax risk limits from Phase 5.1 evidence.
- Do not treat telemetry PnL as financial authority.
- Do not treat `sendTx` acceptance as execution finality.
- Do not infer maker/taker status from intent; require venue-native evidence.
- Direct venue truth overrides telemetry when they disagree.
- Every stage ends with exactly one verdict: `HOLD`, `PROMOTE`, or `ROLLBACK`.

## Current Critical Path

1. Keep Phase 5.1 non-live. Do not start canary, live orders, capital
   escalation, risk-limit relaxation, model training, or EV admission.
2. Treat Phase 5.1u as the active forward capture target-manifest gate, Phase
   5.1w as the operator request-pack gate, Phase 5.1v as the offline capture
   bundle-readiness gate, Phase 5.1s as the local native source-staging gate,
   Phase 5.1t as the optional source-link sidecar builder, Phase 5.1r as the
   forward native source-acquisition layer, and Phase 5.1q as the downstream
   forward-evidence gate. Phase 5.1u is repo-owned, non-live, and emits the
   exact target list for fresh all-five native role capture plus Lighter
   event-time native-limit pressure. Phase 5.1w emits the exact operator
   request pack, manifest skeleton, and optional empty local staging bundle for
   the required sanitized local files.
   Phase 5.1v consumes a local candidate capture bundle manifest, rejects
   unsafe source surfaces, and emits a generated Phase 5.1s manifest only when
   all targets are structurally ready. Phase 5.1s requires an explicit local
   manifest, rejects unsafe source surfaces, strips raw identifiers, emits a
   redacted `local_native_source.jsonl` for Phase 5.1r, and stages optional
   manifest `source_links` as `local_source_link_sidecar.jsonl`.
3. The current Phase 5.1p recovered matrix pack remains `HOLD`: `4527`
   observed terminal labels, `461` fills, `4066` terminal not-filled labels,
   `4527` observed horizons available, `0` observed horizons still missing,
   all `4527` joined to queue/churn, all `4527` source-covered by markout
   readiness, and raw identifier redaction status `PASS`.
4. The dominant blockers are feature quality and selection bias, not live
   runtime reachability: Lighter native-limit context remains partial for
   `2288` labels because sendTx/REST pressure was not historically captured,
   filled-order maker/taker status is incomplete for `287` labels, some
   venue/side buckets remain sparse, and `1613` quarantined/review groups remain
   excluded from the observed-only diagnostic pack.
5. Next evidence move is operator/source action, not another live run. The
   2026-05-04 authorized read-only private source attempt succeeded for Lighter
   account/native-limit and trade-backfill artifacts only, at
   `runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS_20260504T110402Z`
   and
   `runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TRADE-BACKFILL-20260504T110416Z`.
   It does not clear the all-five Phase 5.1w blocker. Phase 5.1w now has a
   canonical empty staged-source run at
   `runs/phase51w_forward_capture_request_pack/PHASE51W-LOCAL-STAGED-SOURCE-BUNDLE-CANONICAL-20260504T000000Z`;
   Phase 5.1v validated that manifest as six local files but correctly held
   with `0 / 287` native-role targets and `0 / 3132` Lighter native-limit
   targets ready at
   `runs/phase51v_forward_capture_bundle_readiness/PHASE51V-EMPTY-STAGED-SOURCE-BUNDLE-HOLD-20260504T000000Z`.
   Continue by capturing fresh read-only native-role source rows for Aster,
   Extended, Lighter, and Paradex, then normalize already-local sanitized rows
   through the repo-owned Phase 5.1y all-venue adapter. Reuse the Phase 5.1x
   Hyperliquid `crossed` source, or the Phase 5.1y-normalized Hyperliquid
   output, for the Hyperliquid subset. Complete Lighter event-time
   active-order/sendTx/REST-or-weighted pressure linkage separately. Then
   populate the Phase 5.1w staged files with six sanitized local read-only
   all-five forward capture files and rerun Phase 5.1v against the Phase 5.1u
   target manifest.
   Capture or locate native snapshots with canonical group/order-key linkage,
   or a redacted Phase 5.1t/5.1s `source_links` sidecar that validates those
   joins by source hash. If 5.1v emits
   `generated_phase51s_manifest_ready=true`, stage the generated manifest
   through Phase 5.1s, run Phase 5.1r, feed its sanitized outputs into Phase
   5.1q, feed Phase 5.1q `native_role_evidence.jsonl` into Phase 5.1n, then
   rerun the recovered 5.1h/5.1i matrix. Do not train models from Phase 5.1u,
   Phase 5.1w, Phase 5.1v, Phase 5.1s, Phase 5.1r, or Phase 5.1q.
6. Proceed to calibrated EV shadow only after canonical outcomes, raw-ID
   hygiene, maker/taker role gaps, holdout splits, feature completeness,
   venue-native truth gaps, and quarantine exclusion policy are accepted.
7. Use `docs/PHASE5_1W_FORWARD_CAPTURE_REQUEST_PACK.md` as the standing
   contract for operator request-pack generation,
   `docs/PHASE5_1V_FORWARD_CAPTURE_BUNDLE_READINESS.md` as the standing
   contract for capture bundle readiness,
   `docs/PHASE5_1S_LOCAL_NATIVE_SOURCE_ACQUISITION.md` as the standing
   contract for local source staging,
   `docs/PHASE5_1R_FORWARD_NATIVE_SOURCE_ACQUISITION.md` as the standing
   contract for source acquisition, `docs/PHASE5_1Q_FORWARD_NATIVE_EVIDENCE.md`
   for downstream forward native evidence,
   `tools/phase51ab_lighter_native_limit_pressure_source.py` as the local
   sanitized Lighter native-limit pressure preflight gate,
   `tools/phase51ac_source_link_reuse_audit.py` as the source-link sidecar
   reuse/non-derivability audit, and
   `docs/PHASE5_1P_LIGHTER_NATIVE_ROLE_EVIDENCE.md` for the exhausted
   quarantined Lighter historical join.

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

### Gate 5.1ab - Lighter Native-Limit Pressure Source Preflight

Allowed:

- Local-only staging of externally supplied sanitized Lighter native-limit
  pressure rows through `tools/phase51ab_lighter_native_limit_pressure_source.py`.
- Phase 5.1v validation of the generated candidate manifest.

Required fields in each complete row:

- `active_order_headroom_account`
- `active_order_headroom_market`
- `sendtx_per_minute_limit`
- `sendtx_per_minute_remaining`
- `rest_requests_per_minute_limit` and `rest_requests_per_minute_remaining`, or
  `weighted_requests_per_minute_limit` and `weighted_requests_per_minute_remaining`
- `native_limit_event_time_status=EVENT_TIME_ALIGNED`
- `canonical_group_id`, `order_key`, or a validated redacted source hash/sidecar

Hard boundary:

- Phase 5.1ab performs no network access and must not call `sendTx`,
  `sendTxBatch`, `nextNonce`, or any venue write path.
- It rejects raw identifiers, secret-shaped fields, unsafe true flags, network
  paths, `.env` inputs, and symlink chains.
- It is a preflight path only. It does not observe pressure by itself, does not
  infer missing pressure from official caps or account tiers, and does not clear
  Phase 5.1 blockers.

Resume command pattern:

```bash
python3 tools/phase51ab_lighter_native_limit_pressure_source.py \
  --input-jsonl <sanitized_lighter_pressure_rows.jsonl> \
  --target-run runs/phase51u_forward_capture_target_manifest/<phase51u_run> \
  --output-root runs/phase51ab_lighter_native_limit_pressure_source \
  --run-id <phase51ab_run_id>
```

Then run the `phase51v_validation_command` emitted in the Phase 5.1ab summary.

### Gate 5.1ac - Source-Link Reuse Audit

Allowed:

- Read-only comparison of a Phase 5.1z source-link request pack against local
  `source_links.sanitized.jsonl` files.
- Emission of reusable-link and missing-hash HOLD artifacts.

Hard boundary:

- Phase 5.1ac must not infer missing links.
- Phase 5.1ac must not read raw order IDs, client IDs, trade IDs, secrets, or
  unsafe true authorization flags.
- A zero-overlap result is a non-derivability proof for existing local sidecars,
  not a blocker clearance.

Resume command pattern:

```bash
python3 tools/phase51ac_source_link_reuse_audit.py \
  --request-pack runs/phase51z_source_link_request_pack/<phase51z_request_pack> \
  --sidecar-root runs \
  --output-root runs/phase51ac_source_link_reuse_audit \
  --run-id <phase51ac_run_id>
```

If `reusable_source_link_count` is zero or `candidate_sidecar_complete=false`,
obtain a new validated redacted sidecar or directly target-linkable source.

### Gate 5.1ad - Source-Link Sidecar Materialization

Allowed:

- HOLD-only materialization of a Phase 5.1v-compatible
  `source_links.sanitized.jsonl` from a Phase 5.1z request pack and a redacted
  mapping file.
- Mapping file fields are limited to `source_record_sha256` or equivalent
  accepted source hash field plus `canonical_group_id` or `order_key`.
- Emission of a candidate manifest that points at the materialized sidecar.

Hard boundary:

- Phase 5.1ad must not infer missing links.
- Phase 5.1ad must not read or emit raw order IDs, client IDs, trade IDs,
  secrets, unsafe true authorization flags, or cross-venue mappings.
- Materialization is not blocker clearance; it must be followed by Phase 5.1v
  validation against the materialized candidate manifest.

Resume command pattern:

```bash
python3 tools/phase51ad_source_link_sidecar_materialize.py \
  --request-pack runs/phase51z_source_link_request_pack/<phase51z_request_pack> \
  --mapping <validated_redacted_mapping.jsonl> \
  --output-root runs/phase51ad_source_link_sidecar_materialize \
  --run-id <phase51ad_run_id>
```

Then run the `phase51v_validation_command` emitted in
`phase51ad_source_link_sidecar_materialize_summary.json`.

Current preferred request pack:

```text
runs/phase51z_source_link_request_pack/PHASE51Z-CURRENT-TARGET-WIDE-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z
```

This supersedes the narrower all-venue request pack for source-link handoff
because it preserves `2819` current redacted unlinked source hashes while still
remaining `HOLD` with `0` reusable existing sidecar rows.

### Gate 5.1ae - Candidate Manifest Composition

Allowed:

- HOLD-only composition of already-local Phase 5.1v candidate manifests.
- Optional addition of explicit local source artifacts using
  `SOURCE_ID=VENUE_ID=PATH`.
- Optional addition of explicit local source-link artifacts using
  `SOURCE_LINK_ID=PATH`.
- Emission of `candidate_manifest.composed.json` for Phase 5.1v validation.

Hard boundary:

- Phase 5.1ae must not infer source links or native roles.
- Phase 5.1ae must not read or emit raw order IDs, client IDs, trade IDs,
  secrets, network paths, env files, symlink inputs, unsafe true authorization
  flags, or conflicting duplicate path metadata.
- Composition is not blocker clearance; it must be followed by Phase 5.1v
  validation against the composed candidate manifest.

Resume command pattern:

```bash
python3 tools/phase51ae_candidate_manifest_compose.py \
  --candidate-manifest runs/phase51ad_source_link_sidecar_materialize/<phase51ad_run_id>/candidate_manifest_with_materialized_sidecar.json \
  --source phase51x_hyperliquid_forward_native_role_snapshot=hyperliquid=runs/phase51x_hyperliquid_native_role_adapter/PHASE51X-HYPERLIQUID-USERFILLS-NATIVE-ROLE-20260504T000000Z/hyperliquid_forward_native_role_snapshot.jsonl \
  --target-run runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-LINK-HYGIENE-20260505T000000Z \
  --output-root runs/phase51ae_candidate_manifest_compose \
  --run-id <phase51ae_run_id>
```

Then run:

```bash
python3 tools/phase51v_forward_capture_bundle_readiness.py \
  --target-run runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-LINK-HYGIENE-20260505T000000Z \
  --candidate-manifest runs/phase51ae_candidate_manifest_compose/<phase51ae_run_id>/candidate_manifest.composed.json \
  --output-root runs/phase51v_forward_capture_bundle_readiness \
  --run-id <phase51v_composed_run_id>
```

Current reference composition:

```text
runs/phase51ae_candidate_manifest_compose/PHASE51AE-CURRENT-TARGET-WIDE-PLUS-HYPERLIQUID-COMPOSE-HOLD-20260505T000000Z
runs/phase51v_forward_capture_bundle_readiness/PHASE51V-CURRENT-TARGET-WIDE-PLUS-HYPERLIQUID-COMPOSE-HOLD-20260505T000000Z
```

The reference composition remains `HOLD`: `73 / 287` native-role targets ready
and `0 / 3132` Lighter native-limit targets ready.

### Gate 5.1af - Local Source Retrieval Audit

Allowed:

- HOLD-only scan of explicit local Phase 5.1z request packs, expected-hash
  bounded telemetry artifacts, and explicit runtime logs.
- Emission of redaction-safe summary, labels, and manifest artifacts.
- Strict retrieval statuses that distinguish missing artifacts from discovery
  hints.

Hard boundary:

- Phase 5.1af must not call venue APIs, read env files, follow network paths,
  place orders, infer source links from time/price/size, emit raw identifiers,
  or treat log pattern hits as complete pressure evidence.
- `clears_phase51_blockers` must remain `false`.

Resume command pattern:

```bash
python3 tools/phase51af_local_source_retrieval_audit.py \
  --request-pack runs/phase51z_source_link_request_pack/PHASE51Z-CURRENT-TARGET-WIDE-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z \
  --bounded-telemetry <expected_sha256>=<bounded_telemetry_path> \
  --log <runtime_log_path> \
  --output-root runs/phase51af_local_source_retrieval_audit \
  --run-id <phase51af_run_id>
```

Current reference audit:

```text
runs/phase51af_local_source_retrieval_audit/PHASE51AF-LOCAL-SOURCE-RETRIEVAL-AUDIT-HOLD-20260505T000000Z
```

The reference audit remains `HOLD`:

```text
source_link_retrieval_status: MISSING_REQUIRED_LINKAGE
lighter_pressure_retrieval_status: MISSING_REQUIRED_PRESSURE_FIELDS
runtime_log_pattern_status: NO_USABLE_PRESSURE_PATTERN
local_retrieval_possible_without_inference: false
```

Do not repeat this audit unless the request pack, local telemetry/log inputs, or
source semantics change materially.

### Gate 5.1ag - Lighter Inactive-Order Bridge Audit

Allowed:

- HOLD-only local bridge audit from sanitized Lighter inactive orders to the
  current source-link request pack and retained Lighter native trade sources.
- Consumption of local raw identifier-shaped trade inputs only for hashing and
  matching; output must remain redacted hash-only.
- Emission of a proposed `source_links.proposed.sanitized.jsonl` only when a
  deterministic inactive-order hash uniquely maps a request source hash to a
  current target.

Hard boundary:

- Phase 5.1ag must not call venue APIs, read env files, place orders, infer
  source links from time/price/size, or emit raw identifiers.
- Inactive-order target matches alone are not maker/taker evidence.
- Trade-export role CSV rows without order/client identifiers are not
  target-linkable evidence.

Reference audit:

```text
runs/phase51ag_lighter_inactive_order_bridge_audit/PHASE51AG-LIGHTER-INACTIVE-ORDER-BRIDGE-AUDIT-V2-HOLD-20260506T0145Z
```

Reference result:

```text
inactive-order target matches: 70
materializable source links: 0
```

Do not repeat Lighter inactive-order/export bridge capture unless account
activity, target window, source semantics, auth material, or local artifacts
change materially.

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

### Gate 5.1h - Observed P-Fill Feature Readiness Audit

Current repo-owned tooling:

- `tools/phase51h_observed_pfill_feature_audit.py` consumes the Phase 5.1g
  observed-only P-fill pack, the Phase 5.1g quarantine reconciliation manifest,
  the Phase 5.1f source-to-canonical manifest, queue/churn packs, and markout
  readiness context.
- The tool joins through source order keys, not canonical order keys. It uses
  per-label source telemetry hashes and fails closed on missing queue/churn
  rows, duplicate queue/churn rows, source-hash mismatch, count mismatch,
  baseline mismatch, or unsafe authorization flags.
- It detects inherited raw `decision_id` values as boolean presence only and
  does not emit raw IDs in new Phase 5.1h artifacts.
- Lighter assumptions remain official-doc bounded: post-only limit orders must
  rest as maker or cancel if crossing
  (`https://docs.lighter.xyz/trading/order-types-and-matching`); Standard and
  Premium fee/latency differ
  (`https://docs.lighter.xyz/trading/trading-fees`); REST, WebSocket,
  `sendTx`/`sendTxBatch`, and active-order limits are account/profile
  sensitive (`https://apidocs.lighter.xyz/docs/rate-limits`); account limits
  must come from the official accountLimits read endpoint or remain unknown
  (`https://apidocs.lighter.xyz/reference/accountlimits`).

Current 5.1h evidence:

- Run:
  `runs/phase51h_observed_pfill_feature_audit/PHASE51H-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z`
- Gate remains `HOLD` with reason
  `phase51h_raw_identifier_present_in_input_not_emitted`.
- Observed labels audited: `4527`.
- Filled labels: `461`.
- Terminal not-filled labels: `4066`.
- Queue/churn joined all source keys: `4527`.
- Markout source context available: `4527`.
- Observed horizon available: `18`; missing: `4509`.
- Lighter native limit status: `2288` partial, `0` fully observed.
- Maker/taker status among filled orders: `174` observed, `222`
  partial/unknown, `65` missing.
- Inherited raw decision ID present in input: `4389`, not emitted by 5.1h.

Superseding redacted 5.1i rebuild of the 5.1h feature audit:

- Run:
  `runs/phase51i_redacted_observed_pfill_feature_audit/PHASE51I-REDACTED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z`
- Gate remains `HOLD` with reason
  `phase51h_missing_observed_horizon_features`.
- Observed labels audited: `4527`.
- Filled labels: `461`.
- Terminal not-filled labels: `4066`.
- Queue/churn joined all source keys: `4527`.
- Markout source context available: `4527`.
- Observed horizon available: `18`; missing: `4509`.
- Lighter native limit status: `2288` partial, `0` fully observed.
- Maker/taker status among filled orders: `174` observed, `222`
  partial/unknown, `65` missing.
- Excluded quarantine/review groups carried forward: `1613`.
- Raw identifier input present: `0`.

5.1h artifacts:

```text
74a38fd69b1fea6254c2fc5ad748319ca8cda13e8a0c4a38c681bd462b180404  pfill_feature_audit_summary.json
fad30084482bf284f0c8bc2c759edcee013ff45a170c039aaf2120290268c80e  pfill_feature_bucket_readiness.jsonl
acc8830aff969df032f714bf8d5ee7709b7a17437034d85cc086ea39d0c3c87f  pfill_feature_coverage_labels.jsonl
```

5.1i redacted 5.1h artifacts:

```text
29e7703a8b85945a0725419fb11b3eaf75d67ffed1095daf353a7b0a97989eca  pfill_feature_audit_summary.json
796f58c4198fa58e3bdd720af30bd20360e106e46e16e7b47cfd9fc2b21c2ed1  pfill_feature_bucket_readiness.jsonl
5f14d0d695b9a87a5f0c3daacdf3454a9ca56f39b16f5ac6f18f1c9b2024399d  pfill_feature_coverage_labels.jsonl
```

### Gate 5.1i - Redacted P-Fill Feature-Matrix Admissibility

Current repo-owned tooling:

- `tools/phase51i_pfill_feature_matrix_admissibility.py` consumes a redacted
  Phase 5.1h feature audit pack and emits a HOLD-only matrix admissibility
  summary, bucket records, and blocker records.
- The tool fails closed if the Phase 5.1h input is not redacted, if raw
  identifier fields are present, if counts do not reconcile, if provenance is
  not bound to the clean baseline commit, or if unsafe authorization flags are
  true.
- It does not train a model, submit orders, approve EV admission, approve live
  or canary use, approve capital escalation, relax risk limits, or make
  financial claims.

Current 5.1i evidence:

- Run:
  `runs/phase51i_pfill_feature_matrix_admissibility/PHASE51I-PFILL-FEATURE-MATRIX-ADMISSIBILITY-REDACTED-TWO-LANE-20260502T000000Z`
- Gate remains `HOLD` with reason
  `phase51i_lighter_native_limit_pressure_not_fully_observed`.
- Redaction status: `PASS`; raw identifier input present: `0`.
- Matrix labels: `4527`; filled: `461`; terminal not-filled: `4066`.
- Train/holdout: `3625` / `902`.
- Queue/churn joined all: `4527`; queue misses: `0`.
- Markout source context available: `4527`.
- Observed horizon available: `4527`; missing: `0`.
- Lighter native-limit context: `0` observed, `2288` partial, and `0` unknown.
- Filled-order maker/taker gaps: `222` partial/unknown and `65` missing.
- Excluded quarantine/review groups: `1613` with reasons `1135`
  duplicate-alias/no-terminal, `375` cancel-all scope, `95` replace-chain
  review, and `8` right-censored/no-terminal.
- Matrix blockers: `lighter_native_limit_pressure_not_fully_observed`,
  `maker_taker_not_fully_observed_for_filled_orders`,
  `sparse_pfill_feature_buckets`, and
  `observed_only_selection_bias_not_resolved`.

5.1i artifacts:

```text
428753544c42c7bcaeeb95786cdfa5423b7a619164cbff2c7cf4805b9fbd2962  pfill_feature_matrix_admissibility_summary.json
b9135cdc6e7cdf09ef39d33fdc1bb708ffd67231ef1dd8f4bb822bdb850946ae  pfill_feature_matrix_buckets.jsonl
4d21e552ddc2efc93fa3fa1021de9a067cee0c0d4262b494ab6f8f696ecc847d  pfill_feature_matrix_blockers.jsonl
8e8759afc3f61a6cd709e38c8e5ae7c14fba9c7fa0b455d3561f73b7300aa6ec  manifest.json
```

### Gate 5.1j - Observed-Horizon Recovery

Current repo-owned tooling:

- `tools/phase51j_observed_horizon_recovery.py` consumes a redacted Phase 5.1h
  feature audit pack, the canonical P-fill manifest, and Phase 5.1e lifecycle
  truth. It emits HOLD-only recovery labels, bucket records, and a summary.
- The tool reads lifecycle alias fields internally but does not emit raw order
  or decision identifiers.
- Recovery is limited to deterministic terminal source-tick horizons. Filled
  rows remain unresolved until a separate fill-time/source-time recovery audit
  exists.

Current 5.1j evidence:

- Recovery run:
  `runs/phase51j_observed_horizon_recovery/PHASE51J-OBSERVED-HORIZON-RECOVERY-TWO-LANE-20260502T000000Z`
- Recovered 5.1h run:
  `runs/phase51j_recovered_observed_pfill_feature_audit/PHASE51J-RECOVERED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z`
- Recovered 5.1i run:
  `runs/phase51j_pfill_feature_matrix_admissibility/PHASE51J-PFILL-FEATURE-MATRIX-ADMISSIBILITY-RECOVERED-TWO-LANE-20260502T000000Z`
- Gate remains `HOLD` with recovered matrix reason
  `phase51i_missing_observed_horizon_features`.
- Redaction status: `PASS`; raw identifier input present: `0`.
- Matrix labels: `4527`; filled: `461`; terminal not-filled: `4066`.
- Input observed horizon available/missing: `18` / `4509`.
- Recovered deterministic terminal horizons: `4048`.
- Recovered observed horizon available/missing: `4066` / `461`.
- Remaining missing horizons are filled-order rows requiring a separate
  timebase treatment.
- Matrix blockers remain: `missing_observed_horizon_features`,
  `lighter_native_limit_pressure_not_fully_observed`,
  `maker_taker_not_fully_observed_for_filled_orders`,
  `sparse_pfill_feature_buckets`, and
  `observed_only_selection_bias_not_resolved`.

5.1j artifacts:

```text
2ab20ec4d916e9f61b758ab5fc54d6cc70cb23c75975db56a5b2d4d69f2922bf  observed_horizon_recovery_summary.json
9ba357b41753dbc7cc9f7592497abf81dc5a6474b2b15963fa78627fe6413a2e  observed_horizon_recovery_buckets.jsonl
767da905e574d5c58d73ee96a89da88295e75d1d102bc011b861e172a07f9cec  observed_horizon_recovery_labels.jsonl
904e3d97eccbc33adee9a9c97043254881e79cda1fc71b12bcb4aa884387497f  recovered pfill_feature_audit_summary.json
2dcce34fa99e3257a8ee06d43a6a3dbc2da23b12723c51ea33f3eed7fc182309  recovered pfill_feature_matrix_admissibility_summary.json
775abf60df16e12913995a3834bdc69abac0ce5951e1b4acf5a7ef01d9804377  recovered pfill_feature_matrix_buckets.jsonl
0c94a2b37c7f2e89835c1eb414555e5f521c53dc500658b1ae89c9e5aaec9194  recovered pfill_feature_matrix_blockers.jsonl
```

### Gate 5.1k - Filled-Horizon Timebase Recovery

Current repo-owned tooling:

- `tools/phase51k_filled_horizon_timebase_recovery.py` consumes a redacted
  Phase 5.1h feature audit pack, canonical P-fill outcomes, and Phase 5.1e
  lifecycle truth. It emits HOLD-only recovery labels, bucket records, and a
  summary.
- The tool recovers filled-order horizons only in source-tick timebase when the
  order source tick and fill source tick are both observable. Exchange
  milliseconds are recorded separately if available and are never written into
  `observed_horizon_source_ticks`.
- The tool reads raw join/fill identifiers internally but does not emit raw
  fill IDs, order IDs, client order IDs, venue order IDs, or decision IDs.

Current 5.1k evidence:

- Filled-horizon recovery run:
  `runs/phase51k_filled_horizon_timebase_recovery/PHASE51K-FILLED-HORIZON-TIMEBASE-RECOVERY-TWO-LANE-20260502T000000Z`
- Recovered 5.1h run:
  `runs/phase51h_observed_pfill_feature_audit/PHASE51K-RECOVERED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z`
- Recovered 5.1i run:
  `runs/phase51i_pfill_feature_matrix_admissibility/PHASE51K-PFILL-FEATURE-MATRIX-ADMISSIBILITY-TWO-LANE-20260502T000000Z`
- Gate remains `HOLD` with recovered matrix reason
  `phase51i_filled_horizon_source_tick_still_missing`.
- Redaction status: `PASS`; raw identifier input present: `0`.
- Filled horizons recovered: `396 / 461`.
- Recovered observed horizon available/missing: `4462` / `65`.
- Remaining missing horizons are filled-order `MISSING_JOIN` rows.
- Matrix blockers remain: `filled_horizon_source_tick_still_missing`,
  `missing_observed_horizon_features`,
  `lighter_native_limit_pressure_not_fully_observed`,
  `maker_taker_not_fully_observed_for_filled_orders`,
  `sparse_pfill_feature_buckets`, and
  `observed_only_selection_bias_not_resolved`.

5.1k artifacts:

```text
fb445f9b7c8dc7655f5796c6378c64be5df00349ec4ef93b8ca7ee35ae603ecc  filled_horizon_timebase_recovery_summary.json
cfbb7c1dd86d731448b13b2f1a3f0a632705795c2b256bf69ffff7ddbb85ea06  filled_horizon_timebase_recovery_buckets.jsonl
a1cfb1844604dd643decf68e9c654969a87bba7ef5d551f8143366193802fb51  filled_horizon_timebase_recovery_labels.jsonl
dd79db7f37573a1e6886db36a4fc889704ea406696fbfdd4c122868eae79fb5b  recovered pfill_feature_audit_summary.json
ea85005b6ce7e19f6db889486b5e91e5f8c207b2e2443f08415eef9e44944909  recovered pfill_feature_matrix_admissibility_summary.json
62927129e74983467a75f47cca517d27869dacb899ce29626cec54195bf22dde  recovered pfill_feature_matrix_buckets.jsonl
c6971841da4ca84291aa8a320214dde1d0a945ddf5a76a2b1ad1a04da792fa6f  recovered pfill_feature_matrix_blockers.jsonl
```

### Gate 5.1l - Filled-Horizon Source-Key Recovery

Current repo-owned tooling:

- `tools/phase51l_filled_horizon_source_key_recovery.py` consumes the 5.1k
  filled-horizon recovery pack, canonical P-fill labels, source P-fill labels,
  and observed fill-label packs. It emits HOLD-only recovery labels, bucket
  records, and a summary.
- The tool targets the remaining filled-order `MISSING_JOIN` rows after 5.1k.
  It first reconstructs canonical source-tick horizons from source P-fill
  `observed_horizon_source_ticks`; if unavailable, it falls back to hashed
  order/client identifiers against observed fill labels.
- The tool never emits raw fill IDs, order IDs, client order IDs, venue order
  IDs, or decision IDs. Raw upstream identifiers are not used as output fields.

Current 5.1r / 5.1q / 5.1p evidence:

- Forward native source-acquisition baseline run:
  `runs/phase51r_forward_native_source_acquisition/PHASE51R-FORWARD-NATIVE-SOURCE-ACQUISITION-BASELINE-NO-SOURCES-20260503T000000Z`
- Forward native evidence baseline run:
  `runs/phase51q_forward_native_evidence/PHASE51R-FORWARD-NATIVE-EVIDENCE-BASELINE-NO-SOURCES-20260503T000000Z`
- Phase 5.1r maker/taker recovery rerun:
  `runs/phase51n_maker_taker_attribution_recovery/PHASE51R-MAKER-TAKER-ATTRIBUTION-RECOVERY-BASELINE-NO-SOURCES-20260503T000000Z`
- Phase 5.1r recovered 5.1h run:
  `runs/phase51h_observed_pfill_feature_audit/PHASE51R-OBSERVED-PFILL-FEATURE-AUDIT-BASELINE-NO-SOURCES-20260503T000000Z`
- Phase 5.1r recovered 5.1i run:
  `runs/phase51i_pfill_feature_matrix_admissibility/PHASE51R-PFILL-FEATURE-MATRIX-ADMISSIBILITY-BASELINE-NO-SOURCES-20260503T000000Z`
- Phase 5.1r baseline result: `0` source files, `0` source rows, `0 / 287`
  native-role targets recovered, `0 / 3132` Lighter native-limit targets
  recovered, and raw identifier redaction `PASS`.
- Phase 5.1q downstream result: `0` recovered forward role records, `287`
  filled rows still missing forward native role source, `3132` Lighter rows
  still missing native-limit pressure source, and raw identifier redaction
  `PASS`.

- Filled-horizon source-key recovery run:
  `runs/phase51l_filled_horizon_source_key_recovery/PHASE51L-FILLED-HORIZON-SOURCE-KEY-RECOVERY-TWO-LANE-20260502T000000Z`
- Lighter event-time native-limit alignment runs:
  `runs/phase51n_lighter_native_limit_time_alignment/PHASE51N-LIGHTER-NATIVE-LIMIT-TIME-ALIGNMENT-TERMINAL-STALE-7200S-20260429T025435Z`
  `runs/phase51n_lighter_native_limit_time_alignment/PHASE51N-LIGHTER-NATIVE-LIMIT-TIME-ALIGNMENT-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z`
- Maker/taker recovery run:
  `runs/phase51n_maker_taker_attribution_recovery/PHASE51P-MAKER-TAKER-ATTRIBUTION-RECOVERY-LIGHTER-NATIVE-20260503T140000Z`
- Native role source inventory run:
  `runs/phase51o_native_role_source_inventory/PHASE51O-NATIVE-ROLE-SOURCE-INVENTORY-ALL-VENUE-20260503T120000Z`
- Lighter native role canonical join run:
  `runs/phase51p_lighter_native_role_canonical_join/PHASE51P-LIGHTER-NATIVE-ROLE-CANONICAL-JOIN-ALL-BACKFILLS-20260503T140000Z`
- Recovered 5.1h run:
  `runs/phase51h_observed_pfill_feature_audit/PHASE51P-OBSERVED-PFILL-FEATURE-AUDIT-LIGHTER-NATIVE-20260503T140000Z`
- Recovered 5.1i run:
  `runs/phase51i_pfill_feature_matrix_admissibility/PHASE51P-PFILL-FEATURE-MATRIX-ADMISSIBILITY-LIGHTER-NATIVE-20260503T140000Z`
- 5.1l recovery gate remains `HOLD` with reason
  `phase51l_filled_horizon_source_key_complete_nonlive_hold`.
- Recovered matrix remains `HOLD` with reason
  `phase51i_lighter_native_limit_pressure_not_fully_observed`.
- Redaction status: `PASS`; raw identifier input present: `0`.
- Remaining filled-horizon missing joins recovered: `65 / 65`.
- Recovery path split: `43` source P-fill horizon, `22` observed-fill hash.
- Recovered observed horizon available/missing: `4527` / `0`.
- Recovered Lighter native-limit observed/partial/unknown: `0` / `2288` / `0`.
- Lighter event-time active-order alignment: `1728 / 2194` rows in the
  025435 lane and `3700 / 3954` rows in the 073231 lane, but full native-limit
  pressure remains partial because sendTx/REST pressure was not historically
  observed.
- Maker/taker role recovery: `174` filled rows already complete, the
  quarantined all-backfill Lighter canonical join recovered `0 / 125`
  source-available Lighter rows, and `162`
  Aster/Extended/Paradex/Hyperliquid rows have no retained native-role source
  in current artifacts. Venue split is Lighter `125`, Aster `113`, Extended
  `28`, Paradex `15`, and Hyperliquid `6`.
- Matrix blockers remain: `lighter_native_limit_pressure_not_fully_observed`,
  `maker_taker_not_fully_observed_for_filled_orders`,
  `sparse_pfill_feature_buckets`, and
  `observed_only_selection_bias_not_resolved`.

5.1l artifacts:

```text
d355a58043cb285454a2005725274f6c34ea9fdb19d1a906220b34f41b862790  filled_horizon_source_key_recovery_summary.json
ea0eb037d67a8ce2e36667e3e835331c48a778f0bd200af9826ae2dd2a0106b7  filled_horizon_source_key_recovery_buckets.jsonl
c2befefc86a808bee7674ccd021b5a608d4529dd7489cd3b3878e47899385607  filled_horizon_source_key_recovery_labels.jsonl
e68990f39d1c0d08a4601d2a344ed782b43ae492b24c04dabad76e82a3a014e2  recovered pfill_feature_audit_summary.json
ffb3574e7effe35e1550f1273f3e99cb636b1928393e5d579901e9b43bebc8a2  recovered pfill_feature_bucket_readiness.jsonl
a95e0d7e9e8f6c815821c790f72fe3b56fd6384bbd888bad695b71cfc7a1f993  recovered pfill_feature_coverage_labels.jsonl
4b087d1e44d4af778d5be7fb21c0dbdbbf9cb1f63c9f97b0d4168aa1241794b4  recovered pfill_feature_matrix_admissibility_summary.json
95ca05d6e4f160c9564847c1da427fc761298b2099429e5a445dc4ed0bd7091c  recovered pfill_feature_matrix_buckets.jsonl
f9a7feeacffbe66498f03086e3036e73b146cbfc9f62038e638df50825dc97df  recovered pfill_feature_matrix_blockers.jsonl
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
5. Read `ROADMAP.md` Phase 5.1 / V2 target specification gate.
6. Read `docs/V2_SPECIFICATION.md` current evidence boundary and blockers.
7. Read `docs/PHASE5_1_BOARD_DECISION.md` Phase 5.1v board decision.
8. Confirm runtime is still shadow before any evidence capture:

```bash
systemctl show paraphina_live --property=ActiveState,SubState,NRestarts
curl -fsS http://127.0.0.1:9898/health/detail
```

9. Do not proceed to live/canary/capital work unless a later committed board
   decision explicitly supersedes this document.

## Next Command Targets

After Phase 5.1u, the next target is not another live run. Phase 5.1u emits the
exact target manifest for fresh forward capture:

```text
runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z
native_role_capture_target_count: 287
native_role_capture_target_counts_by_venue:
- aster: 113
- extended: 28
- hyperliquid: 6
- lighter: 125
- paradex: 15
lighter_native_limit_capture_target_count: 3132
clears_phase51_blockers: false
```

Execute a read-only forward source-capture pilot against that target manifest.
Capture all-five venue-native maker/taker fields and complete Lighter
event-time active-order/sendTx/REST-or-weighted-request pressure into a
sanitized local bundle. Use direct canonical group/order-key linkage where
possible; otherwise use Phase 5.1t to build 5.1s-compatible source-link
sidecars. The active request pack is:

```text
runs/phase51w_forward_capture_request_pack/PHASE51W-FORWARD-CAPTURE-REQUEST-PACK-CANONICAL-20260504T000000Z
native_role_capture_target_count: 287
lighter_native_limit_capture_target_count: 3132
required_local_source_file_count: 6
clears_phase51_blockers: false
```

Use its `capture_bundle_manifest.skeleton.json`, replace placeholder paths
with sanitized local files, then run Phase 5.1v:

```bash
python3 tools/phase51v_forward_capture_bundle_readiness.py \
  --target-run runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z \
  --candidate-manifest <local_capture_bundle_manifest.json> \
  --run-id <phase51v_run_id>
```

Only if `generated_phase51s_manifest_ready=true`, run Phase 5.1s using:

```bash
runs/phase51v_forward_capture_bundle_readiness/<phase51v_run_id>/phase51s_manifest.generated.json
```

Then rerun Phase 5.1s -> 5.1r -> 5.1q -> 5.1n -> 5.1h -> 5.1i, and require
measurable blocker reduction without raw identifiers,
live/canary/capital/risk authorization, model-training shortcuts, or
selection-bias shortcuts.

The baseline Phase 5.1v run against the Phase 5.1u placeholder template is:

```text
runs/phase51v_forward_capture_bundle_readiness/PHASE51V-FORWARD-CAPTURE-BUNDLE-READINESS-TEMPLATE-20260503T000000Z
gate_status: HOLD
native_role_capture_target_ready_count: 0 / 287
lighter_native_limit_capture_target_ready_count: 0 / 3132
source_file_status_counts:
- PLACEHOLDER_PATH: 6
generated_phase51s_manifest_ready: false
clears_phase51_blockers: false
```

Phase 5.1x now covers the Hyperliquid subset with a repo-owned offline adapter
for already-local `userFills` snapshots:

```text
runs/phase51x_hyperliquid_native_role_adapter/PHASE51X-HYPERLIQUID-USERFILLS-NATIVE-ROLE-20260504T000000Z
gate_status: HOLD
hyperliquid_target_recovered_count: 6 / 6
source_row_count: 2000
source_row_emitted_count: 7
raw_identifier_redaction_status: PASS
clears_phase51_blockers: false
```

The downstream Phase 5.1v partial-source run is:

```text
runs/phase51v_forward_capture_bundle_readiness/PHASE51V-HYPERLIQUID-PARTIAL-SOURCE-HOLD-20260504T000000Z
gate_status: HOLD
native_role_capture_target_ready_count: 6 / 287
native_role_capture_target_missing_count: 281
lighter_native_limit_capture_target_ready_count: 0 / 3132
generated_phase51s_manifest_ready: false
clears_phase51_blockers: false
```

Use the Phase 5.1x Hyperliquid source as the Hyperliquid native-role file in the
next all-five candidate bundle. Do not rerun Hyperliquid unless the canonical
Phase 5.1u target manifest changes or a newer source snapshot is needed.

Phase 5.1y now covers offline all-venue native-role normalization for
already-local source rows:

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
native_role_target_recovered_counts_by_venue:
- hyperliquid: 6
source_row_count: 7
source_row_emitted_count: 7
raw_identifier_redaction_status: PASS
clears_phase51_blockers: false
```

Use Phase 5.1y after fresh read-only native-role capture produces already-local
JSON/JSONL rows with direct canonical group/order-key linkage. Phase 5.1y
accepts Aster `ORDER_TRADE_UPDATE` / `o.m` with positive fill quantity,
Extended `isTaker` / `is_taker`, Hyperliquid `crossed`, Lighter
`account_index` plus `is_maker_ask` and side account IDs, and Paradex
`liquidity`. It does not fetch sources, infer source truth, use raw
identifiers in output, or clear blockers without Phase 5.1v readiness.

Phase 5.1n now emits a downstream Lighter native-limit source artifact and
Phase 5.1v manifest:

```text
lighter_forward_native_limit_pressure_snapshot.jsonl
phase51v_lighter_native_limit_manifest.generated.json
```

Use the generated manifest only when the Phase 5.1n summary reports
`forward_native_limit_pressure_source_count > 0`. The 2026-05-04 retest
`PHASE51N-LIGHTER-NATIVE-LIMIT-FORWARD-SOURCE-RETEST-20260504T000000Z`
reported `0`, because the supplied native-limit context still lacked sendTx and
REST-or-weighted limit/remaining pressure. Event-time active-order alignment
alone is not complete Lighter native-limit pressure.

The follow-up read-only header probe
`PHASE51B-LIGHTER-READONLY-HEADER-PROBE-20260504T000000Z` used existing local
credentials and GET-only endpoints. It passed schema/secret-scan acceptance for
calibration-label ingestion, but the accountLimits payload still exposed only
tier/fee/LLP-style keys, sanitized rate/quota response header names were empty,
and no sendTx or REST-or-weighted limit/remaining fields were observed. The
downstream rerun
`PHASE51N-LIGHTER-NATIVE-LIMIT-HEADER-PROBE-HOLD-20260504T000000Z` remained
`HOLD` with `3700` event-time-aligned rows, `0` complete pressure rows, and
`phase51v_lighter_native_limit_manifest_ready=false`.

After Phase 5.1t, source-link generation is repo-owned and should be used
before Phase 5.1s whenever a local forward source snapshot contains raw
order/client identifiers that cannot be emitted. The first 5.1t run over
existing local Lighter snapshots emitted `363` redacted source-link rows from
`1522` source rows. The downstream chain staged those sidecars, Phase 5.1r
applied `909` source-link joins and emitted `296` native-role source records,
but Phase 5.1q/5.1n/5.1h/5.1i still recovered `0 / 287` missing native-role
targets and `0 / 3132` Lighter native-limit targets. Existing local Lighter
artifacts are therefore exhausted for blocker reduction. Phase 5.1v now also
applies validated source-link sidecars during bundle-readiness validation, but
only linked source rows with required native role or native-limit fields can
mark targets ready. Source-link-only manifests must remain `HOLD`. The next
command target is a sanitized local forward read-only native source capture
bundle for the remaining Aster, Extended, Paradex, and Lighter native-role
files, plus complete Lighter event-time active-order/sendTx/REST-or-weighted
request pressure. Use canonical linkage or 5.1t-compatible sidecars where raw
order IDs cannot be emitted, then validate the completed bundle by Phase 5.1v
before Phase 5.1s.

Phase 5.1z now owns the bounded read-only private-source capture/sanitizer step
for venue-native role evidence:

```text
tool: tools/phase51z_readonly_native_role_capture.py
run: runs/phase51z_readonly_native_role_capture/PHASE51Z-READONLY-NATIVE-ROLE-CAPTURE-20260504T000000Z
gate_status: HOLD
fetched rows: aster=268, extended=1601, paradex=30
lighter network capture: skipped; use existing Phase 5.1b/5.1c local sources
sanitized target-linked rows emitted: 67
sanitized rows by venue: aster=39, extended=21, paradex=7
raw_identifier_redaction_status: PASS
```

Combined with the existing Phase 5.1x Hyperliquid native-role source:

```text
runs/phase51v_forward_capture_bundle_readiness/PHASE51V-COMBINED-PHASE51Z-HYPERLIQUID-HOLD-20260504T000000Z
gate_status: HOLD
native_role_capture_target_ready_count: 73 / 287
native_role_capture_target_missing_count: 214 / 287
missing native-role targets by venue: aster=74, extended=7, lighter=125, paradex=8
lighter_native_limit_capture_target_ready_count: 0 / 3132
generated_phase51s_manifest_ready: false
downstream_chain_ready: false
```

Existing retained Lighter trade-backfill sources were reprocessed through
Phase 5.1z and recovered `0` additional target-linked rows. Treat that source
path as exhausted for the current target manifest. The next orchestrator should
continue safe read-only recovery for the remaining Aster/Extended/Paradex
target gaps, but should prioritize finding a safe observed Lighter source that
contains both target-linked native role and event-time active-order/sendTx plus
REST-or-weighted request pressure. Do not infer Lighter request pressure from
caps, empty headers, account tiers, or documentation-only limits.

The diagnostics-enabled Phase 5.1z rerun is:

```text
runs/phase51z_readonly_native_role_capture/PHASE51Z-DIAGNOSTIC-READONLY-NATIVE-ROLE-CAPTURE-20260504T220117Z
gate_status: HOLD
sanitized_source_row_count: 67
aster: target_ready=39/113, native_field_ready=824, no_target_match=784
extended: target_ready=21/28, native_field_ready=1601, no_target_match=1579
lighter: target_ready=0/125, native_field_ready=300, no_target_match=300
paradex: target_ready=7/15, native_field_ready=163, no_target_match=156
```

The combined downstream readiness check is:

```text
runs/phase51v_forward_capture_bundle_readiness/PHASE51V-COMBINED-PHASE51Z-DIAGNOSTIC-HYPERLIQUID-HOLD-20260504T220117Z
gate_status: HOLD
native_role_capture_target_ready_count: 73 / 287
native_role_capture_target_missing_count: 214 / 287
lighter_native_limit_capture_target_ready_count: 0 / 3132
generated_phase51s_manifest_ready: false
```

Operational implication: do not spend the next cycle merely refetching the same
read-only Aster/Extended/Paradex/Lighter rows. The next highest-leverage move is
a Lighter-only source path that supplies target-linked native role and complete
event-time request pressure, or a proven new linkage source for remaining
Aster/Extended/Paradex target groups.

The 2026-05-05 target-link hygiene replay is:

```text
tool changes: Phase 5.1u now preserves redacted order/client/decision hashes and
fill-time fields in target rows; Phase 5.1z now considers Lighter ask/bid native
side order IDs as candidate hashes in addition to side client IDs.
target run: runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-LINK-HYGIENE-20260505T000000Z
native-role replay: runs/phase51z_readonly_native_role_capture/PHASE51Z-LIGHTER-TARGET-LINK-HYGIENE-REPLAY-HOLD-20260505T000000Z
readiness run: runs/phase51v_forward_capture_bundle_readiness/PHASE51V-LIGHTER-TARGET-LINK-HYGIENE-HOLD-20260505T000000Z
gate_status: HOLD
lighter source rows replayed: 1400
lighter native-field-ready rows: 1400
lighter rows with redacted hash candidates: 1400
average Lighter redacted hash candidates per row: 8.0
current Lighter native-role targets recovered: 0 / 125
Lighter native-limit pressure recovered: 0 / 3132
raw_identifier_redaction_status: PASS
```

Operational implication: the current retained Lighter trade-backfill sources are
exhausted even after target-link hygiene and native side-order-ID matching. Do
not keep replaying those same six source snapshots as a blocker-reduction move.
The next Lighter-native-role move requires a new target-linkable observed source
or a new safe linkage sidecar. The Lighter-native-limit move remains blocked
unless a read-only source exposes event-time sendTx and REST-or-weighted
limit/remaining fields; official docs indicate remaining volume quota is exposed
by sendTx/sendTxBatch responses, which are write paths and are not authorized in
Phase 5.1 non-live source capture.

The 2026-05-05 unlinked source-row preservation run is:

```text
tool change: Phase 5.1z can emit sanitized unlinked native-role source rows only
when --emit-unlinked-native-role-source-rows is explicitly set.
native-role replay: runs/phase51z_readonly_native_role_capture/PHASE51Z-LIGHTER-UNLINKED-NATIVE-ROLE-SOURCE-HOLD-20260505T000000Z
readiness run: runs/phase51v_forward_capture_bundle_readiness/PHASE51V-LIGHTER-UNLINKED-NATIVE-ROLE-SOURCE-HOLD-20260505T000000Z
gate_status: HOLD
lighter source rows replayed: 1400
lighter native-field-ready rows: 1400
sanitized unlinked lighter source rows emitted: 531
target-linked lighter source rows emitted: 0
native-role targets ready without sidecar: 0 / 287
lighter native-limit pressure recovered: 0 / 3132
raw_identifier_redaction_status: PASS
```

Operational implication: the preserved unlinked rows are now the correct
sidecar-ready artifact for Lighter native-role recovery. They are not target
truth by themselves. The next orchestrator should either build a validated
redacted source-link sidecar for those source hashes or capture a fresh safe
target-window Lighter trade source that links directly to current canonical
targets. Do not treat `sanitized_source_row_count` as target readiness; use
Phase 5.1v target-ready counts as the promotion boundary.

The 2026-05-05 source-link request-pack run is:

```text
tool: tools/phase51z_source_link_request_pack.py
request pack: runs/phase51z_source_link_request_pack/PHASE51Z-LIGHTER-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z
empty-sidecar validation: runs/phase51v_forward_capture_bundle_readiness/PHASE51V-LIGHTER-SOURCE-LINK-REQUEST-PACK-EMPTY-SIDECAR-HOLD-20260505T000000Z
gate_status: HOLD
source_link_request_source_count: 531
source_link_request_target_count: 125
source_link_sidecar_template_row_count: 0
native-role targets ready with empty sidecar: 0 / 287
lighter native-limit pressure ready: 0 / 3132
raw_identifier_redaction_status: PASS
```

Operational implication: use this request pack as the resume point for Lighter
native-role linkage. A valid sidecar row must contain `source_record_sha256`
plus `canonical_group_id` or `order_key`, and must not contain raw order IDs,
client IDs, trade IDs, secrets, or unsafe true flags. After replacing the empty
sidecar placeholder with a validated sidecar path, rerun Phase 5.1v. If Phase
5.1v still reports zero Lighter target readiness, do not run downstream gates
or claim blocker reduction; switch to a bounded GET-only target-window Lighter
trade capture diagnostic.

The all-venue source-link request pack is now the broader resume artifact for
native-role linkage across Aster, Extended, Lighter, and Paradex:

```text
tool: tools/phase51z_source_link_request_pack.py --venue-id all
source run: runs/phase51z_readonly_native_role_capture/PHASE51Z-ALLVENUE-UNLINKED-NATIVE-ROLE-SOURCE-HOLD-20260505T000000Z
request pack: runs/phase51z_source_link_request_pack/PHASE51Z-ALLVENUE-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z
empty-sidecar validation: runs/phase51v_forward_capture_bundle_readiness/PHASE51V-ALLVENUE-SOURCE-LINK-REQUEST-PACK-EMPTY-SIDECAR-HOLD-20260505T000000Z
gate_status: HOLD
source_link_request_source_count: 2130
source_link_request_source_counts_by_venue: aster=228, extended=1579, lighter=300, paradex=23
source_link_request_target_count: 281
source_link_request_target_counts_by_venue: aster=113, extended=28, lighter=125, paradex=15
native-role targets ready with empty sidecar: 67 / 287
lighter native-limit pressure ready: 0 / 3132
raw_identifier_redaction_status: PASS
```

Operational implication: this all-venue pack is retained as historical
evidence, but it is no longer the preferred resume artifact. Use the
current-target wide request pack instead:

```text
runs/phase51z_source_link_request_pack/PHASE51Z-CURRENT-TARGET-WIDE-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z
```

The current-target wide pack preserves the broader `2819` current unlinked
source hashes and supersedes the older all-venue pack for source-link handoff.
Neither pack can clear Phase 5.1 without a validated redacted sidecar or a
directly target-linkable non-live source.

The bounded GET-only Lighter target-window diagnostic was attempted on
2026-05-05 after hardening `tools/phase51c_lighter_trade_backfill.py` to hash
raw order/trade/client/tx identifiers, hash cursor tokens, and fail closed on
raw identifier-like keys before writing sanitized artifacts.

```text
trade diagnostic: runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TARGET-WINDOW-GETONLY-DIAGNOSTIC-SANITIZED-20260505T143000Z
native-role replay: runs/phase51z_readonly_native_role_capture/PHASE51Z-LIGHTER-GETONLY-SANITIZED-DIAGNOSTIC-20260505T143000Z
readiness run: runs/phase51v_forward_capture_bundle_readiness/PHASE51V-LIGHTER-GETONLY-SANITIZED-DIAGNOSTIC-20260505T143000Z
gate_status: HOLD
trade rows fetched: 400
raw_identifier_redaction_status: PASS
raw_identifier_key_violation_count: 0
sanitized unlinked Lighter source rows: 400
current Lighter native-role targets recovered: 0 / 125
native-role targets ready: 0 / 287
Lighter native-limit pressure ready: 0 / 3132
```

Operational implication: the GET-only `/api/v1/trades` surface is now safe as a
diagnostic artifact path, but this target-window sample still does not link to
the current Phase 5.1u Lighter targets. Do not repeat the same diagnostic unless
the target window or source surface changes. The next move remains a validated
redacted source-link sidecar or a different non-live authorized Lighter source
that directly links to the current target hashes. Lighter request-pressure
evidence remains a separate blocker requiring event-time sendTx plus
REST-or-weighted limit/remaining fields.

Phase 5.1aa was added on 2026-05-05 as the next changed Lighter source
surface. It is a HOLD-only read-only WebSocket account snapshot collector for
the official `account_all` and `account_all_trades` channels.

```text
tool: tools/phase51aa_lighter_ws_account_trades_snapshot.py
account_all run: runs/phase51aa_lighter_ws_account_trades_snapshot/PHASE51AA-LIGHTER-WS-ACCOUNT-ALL-SNAPSHOT-HOLD-20260505T000000Z
account_all_trades run: runs/phase51aa_lighter_ws_account_trades_snapshot/PHASE51AA-LIGHTER-WS-ACCOUNT-TRADES-PARTIALTIMEOUT-HOLD-20260505T000000Z
downstream Phase 5.1z run: runs/phase51z_readonly_native_role_capture/PHASE51Z-LIGHTER-WS-SNAPSHOT-HOLD-20260505T000000Z
downstream Phase 5.1v run: runs/phase51v_forward_capture_bundle_readiness/PHASE51V-LIGHTER-WS-SNAPSHOT-HOLD-20260505T000000Z
account_all message_count: 1
account_all trade_count: 0
account_all_trades message_count: 1
account_all_trades trade_count: 0
raw_identifier_redaction_status: PASS
native-role targets ready: 0 / 287
Lighter native-limit pressure ready: 0 / 3132
```

Operational implication: the Lighter account WebSocket source surface is now
repo-owned, tested, reachable, and metadata-only/source-row redacted. It should not be repeated as
the next move unless account activity, target window, or source semantics
change, because the captured snapshots contained zero trade rows and cannot
link current Phase 5.1u targets. The next Lighter native-role move remains a
validated redacted source-link sidecar for the existing request pack, or a
different non-live authorized source that directly links to current target
hashes. Lighter native-limit pressure still requires event-time sendTx plus
REST-or-weighted limit/remaining evidence; do not infer pressure from caps,
empty headers, tiers, docs, or empty WebSocket account snapshots.

## Current Verdict

`HOLD` for live, canary, capital escalation, risk-limit relaxation, and 24/7
production readiness.

`PROMOTE` only for the fail-closed non-live evidence tooling already accepted
through Phase 5.1ae. No model training, EV admission, canary, or live trading
is authorized.

Current resume path:

```text
native-role blocker:
1. Obtain a validated redacted mapping for
   runs/phase51z_source_link_request_pack/PHASE51Z-CURRENT-TARGET-WIDE-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z.
2. Run Phase 5.1ad to materialize source_links.sanitized.jsonl.
3. Run Phase 5.1ae to compose the materialized manifest with the existing
   Hyperliquid source.
4. Run Phase 5.1v against the composed candidate manifest.

Lighter pressure blocker:
1. Obtain sanitized event-time Lighter pressure rows with active-order
   headroom, sendTx limit/remaining, REST-or-weighted limit/remaining, and
   event-time alignment.
2. Run Phase 5.1ab to stage the rows.
3. Include the Phase 5.1ab candidate manifest through Phase 5.1ae before the
   next all-five Phase 5.1v readiness review.
```

Current stop condition: without those source-owner artifacts or a materially
different non-live source surface, there is no safe local implementation move
that can reduce the Phase 5.1 HOLD blocker. Do not infer source links or
Lighter pressure from time/price/size proximity, documentation-only limits,
GET-only caps, empty headers, account tiers, or empty WebSocket snapshots.
