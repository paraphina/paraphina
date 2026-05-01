# Phase 5 Autonomous Tranche System

This document defines the repo-native workflow used to complete Phase 5 and to
handle any explicitly reopened Phase 5 blocker.

Scope: workflow specification, not current runtime state.

Use this file to understand how Phase 5 should be orchestrated. Use read-only
`phase5/status.md` and `phase5/queue.yaml` to understand the current tranche,
lineage, and topology. Do not mutate Phase 5 runtime state while doing
documentation consolidation.

Closeout note: as of `2026-05-01T01:44:24Z`, the workflow reached accepted
Phase 5 closeout through `phase5_reopened_final_closeout` on surface
`19ac6e21020d39d8`, with active blocker `none`. Future use of this tranche
system should be for a new phase, an explicitly recorded reopened blocker, or a
separate production-readiness gate; do not resume older held branches as if
Phase 5 were still open.

## Purpose

Phase 5 must not continue as ad hoc turn-by-turn orchestration.

The system in `phase5/` and `tools/phase5_*.py` formalizes:

- one serialized live-affecting mainline
- parallel support tracks that do not contaminate live causality
- explicit tranche metadata
- explicit progression gates
- explicit promote / hold / rollback transitions
- deterministic orchestration lanes with isolated worktrees
- automatic closeout recovery, autoscore, and child activation handoff

## Files

- `phase5/queue.yaml`
  - ordered backlog for serialized mainline and parallel support tracks
- `phase5/control_pack.yaml`
  - promoted/control artifacts and automation defaults; use the queue/status
    lineage as newer truth when this control context lags current Phase 5 state
- `phase5/orchestration.yaml`
  - ephemeral orchestration session state for live lane ownership, worktrees, and rung outcomes
- `phase5/status.md`
  - generated human-readable status board
- `phase5/runs/<tranche_id>/`
  - tranche card and latest run manifest for that tranche

## Tooling

- `tools/phase5_prepare_tranche.py`
  - prepare the next or a named tranche and create its repo-side card
- `tools/phase5_run_shadow_ab.py`
  - run control/candidate shadow commands defined for a tranche
- `tools/phase5_run_live_guarded.py`
  - execute a live guarded canary/soak using the tranche system defaults
- `tools/phase5_score_tranche.py`
  - record a tranche verdict and advance the queue/status
- `tools/phase5_tranche.py`
  - shared implementation and lower-level CLI
  - now also owns lane spawning, orchestration state, autoscore, and resumed closeout recovery

## Workflow

1. `python3 tools/phase5_tranche.py orchestrate --tranche-id <id>`
2. tool preflights disk/headroom/health and verifies single live-lane ownership
   plus read-only `audit-state-sync` coherence for the active serialized
   mainline tranche
3. tool spawns deterministic lanes:
   - `live_sentinel`
   - `forensics_gatekeeper`
   - `pass_prep_operator`
   - `fail_prep_operator`
   - optional support-track operators from tranche automation
4. pass/fail/support worktrees are created under `/home/ubuntu/.codex/phase5_worktrees/...`
5. live rung ladder runs in order from the tranche `automation.rung_plan`
6. every rung gets recovered closeout + `live_metrics.json` + `autoscore_bundle.json`
7. final verdict is recorded through the same tranche truth path
8. promotion handoff re-runs `audit-state-sync`; warning, critical, or validator
   errors downgrade promotion to `HOLD`
9. only one child is activated; prep/support worktrees are archived and removed

## Institutional Gate Contract

Before a live-affecting tranche is activated, its queue entry should pass a
read-only gate-contract audit. This audit does not mutate runtime state and does
not replace the live admission gate. It verifies that the tranche is decision
complete before any support gate, canary, soak, or promotion-capable rung is
allowed to carry evidence weight.

The contract must make these fields explicit:

- one serialized-mainline tranche id, objective, hypothesis, branch class, and
  blocker family
- candidate change scope with concrete files
- support gate for promotion-capable live tranches
- explicit rung plan, with promotion-capable final rungs at `7200s` or an
  explicit `final_rung_exception_reason`
- promotion gate criteria and autoscore promotion rules
- rollback criteria
- required evidence artifacts: `closeout`, `metrics`, `autoscore`,
  `direct_venue_audit`, and `cashflow`
- capital budget / unexplained equity-drift limits for economics-relevant
  tranches

Gate-contract failures are structural blockers and must be corrected before the
tranche is trusted for autonomous progression. Institutional-readiness gaps such
as missing machine-readable capital budgets are warnings in the first audit
version, but they remain blockers for any later capital escalation or unattended
readiness claim.

## Automation Model

Defaults live in `phase5/control_pack.yaml` under `automation_defaults`.

Per-tranche overrides live in `phase5/queue.yaml` under `automation`.

Supported fields:

- `support_tracks`
  - allowlist of support-track tranche ids that may be prepared or autorun beside the serialized mainline
- `rung_plan`
  - ordered durations and `continue_on` policy
- `autoscore`
  - machine-readable `clean`, `mechanism`, and `promotion` rule groups
- `autorun_policy` on support tracks
  - `manual`
  - `validate_only`
  - `shadow_smoke`
  - `shadow_ab`

## New Commands

- `python3 tools/phase5_tranche.py audit-gate-contract [--tranche-id <id>]`
- `python3 tools/phase5_tranche.py spawn-lanes --tranche-id <id>`
- `python3 tools/phase5_tranche.py lane-status [--tranche-id <id>]`
- `python3 tools/phase5_tranche.py teardown-lanes --tranche-id <id>`
- `python3 tools/phase5_tranche.py orchestrate --tranche-id <id>`
- `python3 tools/phase5_tranche.py resume-orchestrate --tranche-id <id>`

## Safety Rules

- exactly one serialized mainline may own the live lane at a time
- support lanes may never mutate `/etc/paraphina/*` or `/opt/paraphina/*`
- no final promotion decision is made from markdown scraping; autoscore reads structured closeout + metrics artifacts
- final-rung cleanliness alone is not enough for auto-promotion; promotion requires explicit `automation.autoscore.promotion` rules
- warning or critical `audit-state-sync` findings block preflight and promotion;
  validator exceptions fail closed and leave the tranche in `HOLD`
- `HOLD` with `child_activation_allowed=false` and `activated_child=null` is a
  serialized-mainline stop condition, not permission to reactivate a blocked
  successor
- final topology closeout can only promote when its machine-readable completion
  standard passes; incomplete final evidence records a held closeout

## Parallelism Rules

Safe parallel tracks:

- research
- forensics
- tooling hardening
- `extended` rescue in shadow/local

Unsafe parallelism:

- multiple live-affecting runtime branches at once
- multiple competing live candidate branches on the same baseline
- mixed topology / transport / risk-control changes in one tranche
