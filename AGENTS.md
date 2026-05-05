# Paraphina Agent Contract

This file is the root safety and orientation layer for Codex agents and human
operators. It is not a roadmap, evidence log, board decision, live runbook, or
live-authorization document.

## First Read Order

1. `docs/EXECUTIVE_ORCHESTRATOR_BOOTSTRAP.md`
2. `docs/AGENT_START_HERE.md`
3. `ROADMAP.md`
4. `docs/V2_SPECIFICATION.md`
5. `docs/PHASE5_1_BOARD_DECISION.md`
6. `docs/PHASE5_1_TO_24_7_ORCHESTRATION_RUNBOOK.md`
7. `docs/PHASE5_1_EVIDENCE_LOG.md`

For runtime context only, read `phase5/status.md` and `phase5/queue.yaml`.
Do not mutate Phase 5 runtime state unless a live operator explicitly
authorizes that specific action.

## Authority Order

1. Executable code and tests.
2. Structured generated state, including `phase5/queue.yaml`.
3. Current status boards, including `phase5/status.md`.
4. `ROADMAP.md` for strategic and execution-control authority.
5. `docs/V2_SPECIFICATION.md` for Phase 5.1/V2 target requirements.
6. Phase 5.1 board, runbook, and evidence docs.
7. Historical or generated supporting docs.

When sources disagree, verify against code, tests, structured state, and
`ROADMAP.md` before acting.

## Hard Stops

Never place live orders from Phase 5.1 evidence.

Never enable canary or live mode from Phase 5.1 evidence.

Never escalate capital.

Never relax risk limits.

Never expose secrets, raw private identifiers, API keys, tokens, signatures,
authorization headers, private keys, JWTs, signed payloads, or `.env` contents.

Never make economic, profitability, or production-readiness claims unless they
are tied to accepted balance-authoritative evidence.

Never treat telemetry PnL, replay PnL, model EV, fill-level PnL, or
counterfactual records as financial authority unless reconciled to accepted
balance deltas.

Never restart services, mutate runtime state, or run live/canary commands as
part of documentation, replay, shadow, or evidence-pack work.

## Current Operating Frame

Phase 5 closeout is accepted, but that does not grant unattended production
readiness, capital escalation, or risk-limit relaxation.

Phase 5.1/V2 remains a non-live evidence program unless `ROADMAP.md` and a
fresh board decision explicitly say otherwise. Treat model training, EV
admission, canary, live orders, capital escalation, risk-limit relaxation,
financial claims, and 24/7 readiness as `HOLD` until verified from current
source docs and evidence.

## Executive Orchestrator Resume

Future executive orchestrators should start from
`docs/EXECUTIVE_ORCHESTRATOR_BOOTSTRAP.md`, then verify every volatile status,
blocker count, next move, and evidence path against `ROADMAP.md` and the
current repo/GitHub state before acting.
