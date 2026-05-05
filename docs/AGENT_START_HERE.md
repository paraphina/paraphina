# Agent Start Here

This file is the onboarding path for future Codex chats and human operators who
need to understand Paraphina before changing it.

## Executive Orchestrator Resume

For interrupted or fresh Executive Orchestrator sessions, start with
`AGENTS.md` and `docs/EXECUTIVE_ORCHESTRATOR_BOOTSTRAP.md`, then verify the
current status from the source-of-truth order below before acting.

Do not duplicate current execution queues, blocker counts, run IDs, evidence
paths, or next moves in onboarding docs. The bootstrap document owns process
mechanics, `ROADMAP.md` owns roadmap and gates, and the Phase 5.1 runbook owns
procedure.

## First Read

Read these in order:

1. `README.md`
   - repository overview, documentation hierarchy, safety defaults.
2. `ROADMAP.md`
   - single authoritative roadmap, execution snapshot, target gates, and current
     queue.
3. `docs/ARCHITECTURE.md`
   - current implementation map and status vocabulary.
4. `docs/WHITEPAPER.md`
   - algorithmic narrative and canonical target-spec appendix.
5. `docs/V2_SPECIFICATION.md`
   - active Phase 5.1/V2 target spec for the fill-aware, hedge-aware,
     arbitrage-informed upgrade.
6. `docs/RUNBOOK.md`
   - operational trade-mode controls and live run procedures.
7. `docs/AI_PLAYBOOK.md`
   - repo-specific AI change discipline.

For live topology and Phase 5 status, read `phase5/status.md` and
`phase5/queue.yaml` only as context unless explicitly authorized to change
runtime state.

## Source-Of-Truth Order

When docs disagree, use this order:

1. Executable code and tests.
2. Structured generated state such as `phase5/queue.yaml`.
3. Current status board such as `phase5/status.md`.
4. Authoritative docs: `ROADMAP.md` Execution Snapshot and target gates,
   `docs/ARCHITECTURE.md`,
   `docs/WHITEPAPER.md` Part I,
   `docs/V2_SPECIFICATION.md` for Phase 5.1/V2 target requirements.
5. Supporting docs: `docs/RUNBOOK.md`, `docs/WORLD_CLASS_RUNTIME_GOALS.md`,
   `docs/PHASE5_AUTONOMOUS_TRANCHE_SYSTEM.md`.
6. Historical/generated docs: Roadmap-B readiness reports, WP100 signoff,
   parity matrices, completion audits.

## Safe Work Modes

Docs-only work may edit:

- `README.md`
- `ROADMAP.md`
- `docs/ARCHITECTURE.md`
- `docs/AGENT_START_HERE.md`
- `docs/V2_SPECIFICATION.md`
- current-state notes in `docs/WHITEPAPER.md` before the canonical-spec marker
- scope banners in dated/generated docs
- other documentation files when the change is clearly non-runtime

Docs-only work must not edit:

- `phase5/queue.yaml`
- `phase5/orchestration.yaml`
- `phase5/runs/**`
- `tools/phase5_*.py`
- `tools/paraphina_watch.py`
- `tools/telemetry_analyzer.py`
- `paraphina/src/**`
- `paraphina/tests/**`
- `/etc/paraphina`
- `/opt/paraphina`
- `/var/lib/paraphina`
- `/home/ubuntu/promotion_runs`

Do not run services, restart services, launch live/canary runs, audit venues, or
place/cancel orders as part of documentation consolidation.

## Status Language

Use exact status words:

- `implemented`
- `partially implemented`
- `shadow-only`
- `experimental`
- `planned`
- `historical/generated`

Do not call something implemented unless you have opened the relevant code or
test evidence. Do not call something production-ready because it is implemented
in the whitepaper audit scope.

## Whitepaper Rule

`docs/WHITEPAPER.md` has a hash-locked canonical appendix. Do not edit the
appendix unless the task is explicitly to change the canonical spec and update
the docs integrity artifacts. If current code has moved beyond old appendix
status annotations, add a current-state clarification before the canonical
marker.

The active V2 target spec is separate: use `docs/V2_SPECIFICATION.md` for
Phase 5.1 requirements, and summarize/link it from Part I before the canonical
marker. Do not insert V2 target text into the hash-locked appendix.

## Roadmap Rule

There is one roadmap: `ROADMAP.md`.

Other roadmap-like files are scoped:

- `docs/ROADMAP_B.md` - live venue rollout note.
- `docs/ROADMAP_B_READINESS_REPORT.md` - generated historical snapshot.
- `docs/WORLD_CLASS_RUNTIME_GOALS.md` - runtime objective framework.
- `docs/PHASE5_AUTONOMOUS_TRANCHE_SYSTEM.md` - workflow model.

Do not create another competing roadmap. Add new strategic priorities, target
gates, and execution-queue changes to `ROADMAP.md` and link supporting details
out to dedicated docs.

## Quick Summary For Future Agents

Paraphina is a deterministic market-making research and live-runtime system with
feature-gated live connectors and a Phase 5 evidence workflow. The current docs
architecture deliberately separates strategy (`ROADMAP.md`), implementation map
(`docs/ARCHITECTURE.md`), algorithm/spec narrative (`docs/WHITEPAPER.md`),
V2 target requirements (`docs/V2_SPECIFICATION.md`), and operator procedures
(`docs/RUNBOOK.md`). `ROADMAP.md` now also owns the
execution snapshot, target gates, and lane rules; `phase5/queue.yaml` remains
the structured current queue and `phase5/status.md` is the generated human
status board. Phase 5 is currently frozen as promoted/accepted-closeout on
`phase5_reopened_final_closeout`, surface `19ac6e21020d39d8`, with active
blocker `none`; live-run PnL authority is five-account balance pre/post `bPNL`.
WP100 completion and Phase 5 closeout are not proof of unattended production
readiness. For Phase 5, read status/queue files as context and do not mutate
runtime state unless explicitly authorized.
