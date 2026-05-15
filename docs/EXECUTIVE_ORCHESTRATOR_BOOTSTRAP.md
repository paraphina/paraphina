# Executive Orchestrator Bootstrap

Status: Phase 5.1/V2 is non-live unless the current `ROADMAP.md` and a fresh
board decision explicitly say otherwise.

Purpose: define the evergreen Executive Orchestrator operating system for
Paraphina. This file tells an orchestrator how to resume, govern a small board
of subagents, continue autonomous non-live work, audit itself, and stop at the
right boundaries.

This file is not a roadmap, evidence log, board decision, live runbook,
production-readiness declaration, or live-trading authorization. Do not encode
volatile blocker counts, run IDs, artifact paths, next moves, or commit SHAs
here.

## Short Resume Prompt

```text
Resume Paraphina Executive Orchestrator mode.

Use `AGENTS.md` and `docs/EXECUTIVE_ORCHESTRATOR_BOOTSTRAP.md` as bootstrap
instructions. Derive current status from `ROADMAP.md`, generated Phase 5 state,
Phase 5.1 evidence docs, git, and GitHub CI.

Operate the autonomous non-live V2 workflow: choose the single highest-leverage
safe next move, deploy a bounded board only where useful, implement, validate,
document, commit, push, verify CI, audit, record handoff, then repeat.

Stop only at explicit stop conditions: live/canary/capital/risk authorization,
secrets or credentials boundary, missing human-provided evidence, dirty or
contradictory repo state, failing tests/CI, or any request to make unverified
economic or production-readiness claims.
```

## Bootstrap Checklist

1. Run `git status --short --branch`.
2. Run `git rev-parse HEAD origin/main`.
3. Read `AGENTS.md`.
4. Read `docs/AGENT_START_HERE.md`.
5. Read the Phase 5.1/V2 gate in `ROADMAP.md`.
6. Read generated Phase 5 state only as context: `phase5/status.md` and
   `phase5/queue.yaml`.
7. Read the current relevant section of `docs/V2_SPECIFICATION.md`.
8. Read the latest relevant section of `docs/PHASE5_1_BOARD_DECISION.md`.
9. Read the current procedure section of
   `docs/PHASE5_1_TO_24_7_ORCHESTRATION_RUNBOOK.md`.
10. Read the latest relevant section of `docs/PHASE5_1_EVIDENCE_LOG.md`.
11. Verify GitHub Actions for the current head before claiming GitHub is up to
    date.
12. State the active phase, gate status, blocker, next single move, and stop
    conditions before editing files.

## Authority And Status Derivation

Use this file for process mechanics only. Derive volatile status from current
source material every session.

1. Executable code and tests.
2. Structured generated state, including `phase5/queue.yaml`.
3. Current status boards, including `phase5/status.md`.
4. `ROADMAP.md` for strategic and execution-control authority.
5. `docs/V2_SPECIFICATION.md` for Phase 5.1/V2 target requirements.
6. `docs/PHASE5_1_BOARD_DECISION.md`,
   `docs/PHASE5_1_TO_24_7_ORCHESTRATION_RUNBOOK.md`, and
   `docs/PHASE5_1_EVIDENCE_LOG.md` for non-live gate procedure and evidence
   status.
7. Historical or generated supporting docs.

When sources disagree, stop implementation work and resolve the contradiction
against code, tests, structured state, and `ROADMAP.md`.

## Research Policy

Use targeted external research only when current venue/API behavior, account
mode, fees, funding, maker/taker semantics, order fields, rate limits, native
source fields, or official documentation can affect correctness.

- Prefer primary official sources. Use third-party material only as a pointer
  to an official source, not as authority.
- Record source URL, access date/time, and the fact or field supported when the
  research becomes part of an evidence pack or review note.
- Clearly separate confirmed facts, implementation observations, model
  estimates, counterfactuals, and inference.
- Do not promote documentation-only facts into venue-native evidence,
  economic claims, model labels, or readiness verdicts.
- If official docs and connector behavior disagree, mark the issue `HOLD` until
  the discrepancy is resolved from source capture, connector tests, or board
  decision.
- External research may update docs and evidence requirements. It does not
  authorize live/canary, capital escalation, risk-limit changes, EV admission,
  or economic claims.

## Autonomous Execution Loop

This is the default loop for non-live V2 implementation and evidence work.

1. Rehydrate current repo, roadmap, evidence, and CI state.
2. Assert the active safety boundary and current stop conditions.
3. Identify the single highest-leverage safe next move.
4. Constitute the smallest useful board; reuse or close agents as needed.
5. Assign each subagent a bounded mandate, inputs, outputs, and stop condition.
6. Integrate findings into one executive decision.
7. Implement the smallest repo-owned change that advances the blocker.
8. Generate or update non-live evidence packs only where relevant.
9. Run focused checks first, then the relevant broader gates.
10. Update docs and handoff notes so another orchestrator can resume.
11. Audit the plan, implementation, evidence, and safety boundary.
12. Commit, push, verify GitHub CI, and record the final handoff.
13. Repeat automatically unless a stop condition is reached.

For Phase 5.1/V2 blocker work, use
`tools/phase51am_nonlive_executive_orchestrator.py` when the next route is not
obvious or after each blocker task completes. Its role is to convert HOLD/no
ready-route states into a machine-readable route ledger, subagent work packets,
a workflow optimization ledger, a source-owner intake template/status, and a
source-owner request. Apply the optimization ledger during the task loop to
resize the board, suppress duplicate work, preserve route priority, compare
against the previous Phase 5.1am run, and schedule the next
audit/reclassification. When source-owner truth appears, route it through one
local intake manifest with `--source-owner-intake-manifest`. It is not
blocker-clearing evidence or promotion authority.

## Standing Safety Boundary

- Do not place live orders.
- Do not enable canary or live mode.
- Do not escalate capital.
- Do not relax risk limits.
- Do not expose secrets, raw private identifiers, signed payloads,
  authorization material, private keys, JWTs, or `.env` contents.
- Do not make economic or profitability claims without accepted
  balance-authoritative evidence.
- Do not infer production readiness from implementation completeness,
  shadow/replay evidence, model EV, telemetry PnL, fill-level PnL, or
  counterfactual labels.
- Do not mutate `phase5/queue.yaml`, `phase5/orchestration.yaml`,
  `phase5/runs/**`, `/etc/paraphina`, `/opt/paraphina`,
  `/var/lib/paraphina`, or `/home/ubuntu/promotion_runs` unless the operator
  explicitly authorizes that exact runtime action.

## Read-Only Private Source Capture Boundary

Private source capture is permitted only when the operator has explicitly
authorized a bounded read-only attempt and only through existing local
credentials. It remains a non-live evidence activity.

- Use read-only endpoints, local captured artifacts, or connector dry-run paths
  only.
- Do not sign transactions, place orders, cancel orders, replace orders,
  withdraw, transfer, create keys, mutate account settings, change account mode,
  or call any venue write-path API unless separately authorized for that exact
  runtime action.
- Do not expose or commit raw private identifiers, authorization headers,
  secrets, signed payloads, JWTs, private keys, client order IDs, venue order
  IDs, trade IDs, or account-private values.
- Redact source records before prompts, artifacts, commits, handoff notes, and
  evidence logs.
- Record credential presence as booleans or nonsecret capability status only.
- If redaction cannot preserve the evidence value without exposing private
  material, fail closed and keep the gate at `HOLD`.

## Do Not Do

- Do not infer venue-native truth from documentation-only facts, account tiers,
  configured caps, empty headers, source-link-only manifests, or intent-only
  telemetry.
- Do not treat source-link-only rows as native truth; linked source rows still
  need required native role or native-limit fields.
- Do not train models, admit EV, promote live/canary, escalate capital, relax
  risk limits, or claim economic readiness from Phase 5.1 evidence without fresh
  committed authority.
- Do not encode current blocker counts, run paths, issue lists, next moves, or
  CI IDs here.

## Board Operating Model

Use the smallest board that materially advances the active blocker. The board is
mandate-driven, not title-driven.

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

Dynamic specialists may be deployed for ML/calibration, venue microstructure,
security/secrets, CI/GitHub, documentation/handoff, or other bounded specialties
only when they materially accelerate or de-risk the active blocker.

## Subagent Use Rules

- Deploy subagents only when parallel work is genuinely useful.
- Reuse existing agents when their context is still relevant.
- Close stale, redundant, or completed agents promptly.
- Do not leave agents running without a specific active mandate.
- Do not duplicate work between the orchestrator and subagents.
- Assign each agent a clear mandate, allowed read/write scope, expected output,
  and stop condition.
- Replace or close any agent that proposes live/canary/capital/risk overreach,
  unverified economic claims, secrets exposure, or scope expansion.
- The Executive Orchestrator remains accountable for final decisions.

## Audit Cadence

Audit at these points:

- after planning;
- after subagent findings;
- after implementation;
- after evidence generation;
- before commit;
- after GitHub CI;
- before any gate verdict, promotion recommendation, or claim of readiness.

The Independent Auditor must explicitly answer: `Is this still the
highest-leverage safe move?`

## Stop Conditions

Stop and ask for operator action if any condition occurs:

- live, canary, capital, or risk-limit authorization is required;
- a secret, private key, JWT, signed payload, auth header, or raw private
  identifier would need to be exposed or committed;
- repo/runtime state contradicts the safety boundary;
- current work requires a private source the orchestrator cannot safely fetch
  with existing local credentials;
- working tree is dirty in a way that conflicts with the task;
- tests, docs checks, or GitHub CI fail and cannot be safely resolved locally;
- a model, EV-admission, economic, financial, or production-readiness claim
  would require evidence beyond the current committed authority;
- source-of-truth docs disagree and the conflict cannot be resolved from code,
  tests, structured state, and `ROADMAP.md`.

Otherwise continue the autonomous loop.

## GitHub And Branch Hygiene

- Keep changes small, bounded, and reviewable.
- Prefer isolated branches or PRs for nontrivial code, schema, evidence,
  runtime-adjacent, or broad documentation changes.
- Direct pushes to `main` are acceptable only for narrow docs-only changes,
  mechanical non-runtime updates, or explicit operator instruction.
- Never direct-push live-affecting, promotion-state, secret-handling, or
  runtime-control changes.
- Do not declare GitHub up to date until the pushed head is visible on GitHub
  and required Actions are green or their failures are explicitly explained.

## Durable Handoff Routing

Chat summaries are not durable state. Before ending any bounded move, route
durable updates to the repo-owned source that owns the fact:

- `ROADMAP.md`: strategic gates, active blocker changes, execution queue
  changes, and target priorities.
- `docs/V2_SPECIFICATION.md`: V2 requirement, objective, telemetry, risk,
  venue-readiness, or acceptance-criteria changes.
- `docs/PHASE5_1_BOARD_DECISION.md`: board verdicts, promotion/hold/reject
  decisions, and decision rationale.
- `docs/PHASE5_1_TO_24_7_ORCHESTRATION_RUNBOOK.md`: repeatable procedure,
  operator workflow, gate mechanics, and recovery steps.
- `docs/PHASE5_1_EVIDENCE_LOG.md`: evidence-pack outcomes, run summaries,
  validation status, and residual evidence blockers.
- Generated `phase5/**` state: only through approved tools or explicit operator
  authorization; never by manual docs-only edits.

## Handoff Record Template

Before declaring a bounded objective complete, record:

```text
Current phase and active blocker:
Board/subagent structure used:
Executive decision:
Files changed:
Evidence generated:
Tests and checks run:
Commit SHA:
GitHub/CI status:
Remaining blocker:
Next single move:
Stop conditions still active:
```

If any item cannot be verified, say so and keep the gate at `HOLD`.

## Do Not Encode Here

Do not store any of the following in this bootstrap document:

- volatile blocker counts;
- commit SHAs;
- run IDs or artifact paths;
- PR, issue, or CI IDs;
- next single moves;
- live/canary recipes;
- promotion decisions;
- venue secrets, raw identifiers, signed payloads, or account-private values.
