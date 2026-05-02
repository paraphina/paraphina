# Phase 5.1 To 24/7 Orchestration Runbook

Date: 2026-05-01

Status: `HOLD` for live/canary/capital. `PROMOTE` only for continuing the
Phase 5.1 non-live evidence workflow.

Branch: `mm-pnl-harness-clean`

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

1. Freeze and push the Phase 5.1b repo state.
2. Produce the first real read-only Lighter account/native-limit evidence pack.
3. Validate the evidence pack against telemetry schema v2.
4. Accept or hold Phase 5.1b based on whether the pack contains account profile,
   account limits, active-order headroom, fee/market metadata, and optional
   maker/taker trade-role samples.
5. Only after accepted Phase 5.1b evidence, begin calibration-label ingestion
   for P-fill, markout, queue/churn, maker/taker attribution, residual/tail
   costs, and balance reconciliation.

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
- Observed fill and balance labels do not promote 5.1c by themselves. Missing
  native maker/taker role certainty, quote/fill joins, and deterministic
  holdout still force `HOLD`.

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
