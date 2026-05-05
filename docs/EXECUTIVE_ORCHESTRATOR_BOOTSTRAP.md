# Executive Orchestrator Bootstrap

Status: Phase 5.1/V2 is non-live unless the current `ROADMAP.md` and a fresh
board decision explicitly say otherwise.

Purpose: allow a future Executive Orchestrator to resume Paraphina with a short
prompt, while preserving safety boundaries and evidence discipline. This file is
a bootstrap checklist and resume card. It is not a roadmap, evidence log, board
decision, live runbook, or live-trading authorization.

## Short Resume Prompt

```text
Resume Paraphina Executive Orchestrator mode.

Use `AGENTS.md` and `docs/EXECUTIVE_ORCHESTRATOR_BOOTSTRAP.md` as bootstrap
instructions, then verify `ROADMAP.md`, `docs/V2_SPECIFICATION.md`,
`docs/PHASE5_1_BOARD_DECISION.md`,
`docs/PHASE5_1_TO_24_7_ORCHESTRATION_RUNBOOK.md`,
`docs/PHASE5_1_EVIDENCE_LOG.md`, the latest git state, and GitHub CI.

Identify the single highest-leverage safe next move and execute autonomously
until the bounded objective is implemented, validated, documented, committed,
pushed, and CI-green.

Preserve all no-live, no-canary, no-capital-escalation, no-risk-relaxation,
no-secrets, and no-unverified-economic-claims constraints.
```

## Bootstrap Checklist

1. Run `git status --short --branch`.
2. Run `git rev-parse HEAD origin/main`.
3. Read `AGENTS.md`.
4. Read `docs/AGENT_START_HERE.md`.
5. Read the Phase 5.1/V2 gate in `ROADMAP.md`.
6. Read the current relevant section of `docs/V2_SPECIFICATION.md`.
7. Read the latest relevant section of `docs/PHASE5_1_BOARD_DECISION.md`.
8. Read the current resume section of
   `docs/PHASE5_1_TO_24_7_ORCHESTRATION_RUNBOOK.md`.
9. Read the latest relevant section of `docs/PHASE5_1_EVIDENCE_LOG.md`.
10. Verify GitHub Actions for the current head before claiming GitHub is up to
    date.
11. State the active phase, gate status, blocker, next single move, and stop
    conditions before editing files.

## Standing Safety Boundary

- Do not place live orders.
- Do not enable canary or live mode.
- Do not escalate capital.
- Do not relax risk limits.
- Do not expose secrets, raw private identifiers, signed payloads,
  authorization material, private keys, JWTs, or `.env` contents.
- Do not make economic or profitability claims without accepted
  balance-authoritative evidence.
- Do not mutate `phase5/queue.yaml`, `phase5/orchestration.yaml`,
  `phase5/runs/**`, `/etc/paraphina`, `/opt/paraphina`,
  `/var/lib/paraphina`, or `/home/ubuntu/promotion_runs` unless the operator
  explicitly authorizes that exact runtime action.

## Current Resume Card

Verify this card against `ROADMAP.md` before acting; it is a convenience
summary and may become stale.

- Latest known commit at creation: `d1a702b16ea7dd48d3b8c7247038d03fd69cbd1d`
  (`Add Phase 5.1z source-link request pack`).
- Latest known Phase 5.1/V2 status: `specified_holding_nonlive`.
- Latest known verdict: `HOLD` for model training, EV admission, canary, live
  orders, capital escalation, risk-limit relaxation, financial claims, and
  24/7 readiness.
- Latest known blocker: `214 / 287` native-role targets still missing after
  Phase 5.1z plus Hyperliquid, and Lighter event-time native-limit pressure
  remains `0 / 3132`.
- Latest known Lighter source-link resume point:
  `runs/phase51z_source_link_request_pack/PHASE51Z-LIGHTER-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z`.
- Latest known empty-sidecar validation:
  `runs/phase51v_forward_capture_bundle_readiness/PHASE51V-LIGHTER-SOURCE-LINK-REQUEST-PACK-EMPTY-SIDECAR-HOLD-20260505T000000Z`.

## Next Single Move

Verify against `ROADMAP.md` before acting.

Populate the request pack sidecar with validated redacted links only:
`source_record_sha256` plus `canonical_group_id` or `order_key`. The sidecar
must not contain raw order IDs, client IDs, trade IDs, secrets, authorization
material, or unsafe true flags. Rerun Phase 5.1v against the request-pack
candidate manifest after replacing the empty sidecar path with the validated
sidecar.

If no valid sidecar can be produced, the next safe diagnostic is a bounded
GET-only target-window Lighter `/api/v1/trades` capture. That diagnostic is not
a promotion path.

Separately find a non-live-authorized source for event-time sendTx plus
REST-or-weighted limit/remaining pressure. Do not infer missing Lighter
pressure from GET-only caps, empty headers, account tiers, or
documentation-only limits.

## Do Not Do

- Do not rerun retained Lighter trade-backfill snapshots as blocker-reduction
  work; the current roadmap marks them exhausted for direct current target
  recovery.
- Do not infer native role, request pressure, profitability, or economic
  readiness from documentation-only facts.
- Do not run downstream Phase 5.1s, Phase 5.1r, Phase 5.1q, Phase 5.1n,
  Phase 5.1h, or Phase 5.1i unless Phase 5.1v reports target readiness
  improvement and emits the required readiness output.
- Do not train models or admit EV from Phase 5.1u, 5.1w, 5.1v, 5.1s, 5.1r,
  5.1q, or 5.1z artifacts.
- Do not treat source-link-only manifests as native truth; linked source rows
  still need required native role or native-limit fields.

## Board Structure

Use the smallest board that materially advances the active blocker.

- Executive Orchestrator: owns sequencing, repo safety, GitHub state, final
  decisions, and the single next move.
- Quant/Evidence Lead: use for EV validity, calibration, confidence bounds,
  sparse buckets, selection bias, and falsifiability.
- Execution/Venue Lead: use for venue-native source requirements, maker/taker
  semantics, rate limits, account modes, and read-only source feasibility.
- Systems/Data Lead: use for schemas, tools, replay determinism, redaction,
  manifests, tests, and CI compatibility.
- Risk/Ops Lead: use for no-live boundaries, kill/risk invariants, residual
  risk, rollback readiness, and secrets hygiene.
- Independent Auditor: use to challenge whether the proposed next move is still
  highest-leverage and safe.

Deploy subagents only when parallel work is genuinely useful. Reuse or close
agents as soon as their mandate is complete. Do not leave agents running without
a specific active mandate.

## Required Handoff After Each Move

Before declaring a bounded objective complete, record:

- Current phase and active blocker.
- Board/subagent structure used.
- Executive decision.
- Files changed.
- Evidence generated.
- Tests and checks run.
- Commit SHA.
- GitHub/CI status.
- Remaining blocker.
- Next single move.

If any item cannot be verified, say so and keep the gate at `HOLD`.
