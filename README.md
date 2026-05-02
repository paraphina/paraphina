# Paraphina

Paraphina is a research-to-runtime market-making system for venue-agnostic
perpetual futures execution. The repository contains a deterministic simulator,
strategy/risk engines, telemetry contracts, live connector scaffolding and
feature-gated live runners, and the Phase 5 evidence workflow used to qualify
multi-venue live surfaces.

This README is the repository entry point. It should tell a future Codex agent,
developer, or operator where to find the current truth before they make changes.

## Start Here

Read these files in order:

1. `README.md` - repository orientation and documentation hierarchy.
2. `ROADMAP.md` - the single authoritative strategic and execution roadmap;
   read its Roadmap Contract and Execution Snapshot first.
3. `docs/ARCHITECTURE.md` - current architecture, code surfaces, and status
   vocabulary.
4. `docs/AGENT_START_HERE.md` - safe onboarding path for future Codex sessions.
5. `docs/WHITEPAPER.md` - algorithmic whitepaper, current implementation
   narrative, and hash-locked canonical v1 appendix.
6. `docs/V2_SPECIFICATION.md` - repo-owned Phase 5.1/V2 target specification
   for the fill-aware, hedge-aware, arbitrage-informed upgrade.
7. `docs/RUNBOOK.md` - operational controls and trade-mode procedures.

When these disagree, executable code and structured Phase 5 state are stronger
than prose. For live topology, read-only status comes from `phase5/status.md`
and `phase5/queue.yaml`; do not infer it from older generated reports.

## Current System State

The repo currently implements:

- deterministic simulation and batch research contracts under `paraphina/src/`,
  `batch_runs/`, and `tools/`;
- strategy/risk/accounting modules for fair value, volatility, market making,
  exits, hedging, inventory and kill-switch behavior;
- a feature-gated live binary in `paraphina/src/bin/paraphina_live.rs`;
- canonical live venue ordering in `paraphina/src/live/venues.rs`;
- live trade modes where `shadow` is the default safety mode;
- telemetry schemas and replay/check tooling;
- a Phase 5 tranche workflow in `phase5/` and `tools/phase5_*.py`.

The repo does not treat whitepaper completion as final all-5 live topology or
unattended market-making production readiness. WP100 and parity artifacts prove
code-visible implementation coverage for a dated audit scope; live readiness is
separate and must be supported by Phase 5 evidence.

Current Phase 5 closeout: `phase5_reopened_final_closeout` is promoted on
surface `19ac6e21020d39d8` with active blocker `none`. The accepted closeout
uses the final `7200s` five-venue evidence window from
`2026-04-30T22:04:41.427000Z` to `2026-05-01T00:04:47.813000Z`, and live-run
PnL authority is the five-account balance pre/post delta (`bPNL`), not telemetry
`pnl_total`. This closes Phase 5 only; unattended production readiness remains a
future gate.

## Documentation Hierarchy

- `ROADMAP.md` is the strategic source of truth and execution-control document.
- `docs/ARCHITECTURE.md` is the implementation/source map.
- `docs/AGENT_START_HERE.md` is the agent onboarding path.
- `docs/WHITEPAPER.md` is the algorithm/spec reference. Its Part II appendix is
  hash-locked and may contain historical status annotations.
- `docs/V2_SPECIFICATION.md` is the active V2 target spec. It is not live
  authorization and must stay gated by Phase 5.1 evidence and `ROADMAP.md`.
- `docs/ROADMAP_B.md` is a supporting live-venue rollout note, not a second
  roadmap.
- `docs/ROADMAP_B_READINESS_REPORT.md`, `docs/WP_PARITY_MATRIX.md`, and
  `docs/WHITEPAPER_COMPLETION_*` are dated/generated audit artifacts unless
  regenerated.
- `docs/WORLD_CLASS_RUNTIME_GOALS.md` defines runtime objectives and gates, not
  the current topology.
- `docs/PHASE5_AUTONOMOUS_TRANCHE_SYSTEM.md` defines the Phase 5 workflow, not
  the current live tranche state.

## Status Vocabulary

- `implemented` - supported by repository code and tests or reproducible checks.
- `partially implemented` - scaffolding exists but the contract is incomplete.
- `shadow-only` - safe for observation/replay; no live order placement implied.
- `experimental` - research or support-track idea that is not promoted behavior.
- `planned` - roadmap intent with no complete implementation.
- `historical/generated` - snapshot artifact; do not use as current truth unless
  regenerated and reconciled.

## Safety Defaults

Future agents should prefer docs-only changes unless implementation work is
explicitly requested. Do not edit Phase 5 runtime state, live config, promotion
runs, or runtime services as part of documentation consolidation. In particular,
do not mutate `phase5/queue.yaml`, `phase5/orchestration.yaml`,
`phase5/runs/**`, `/etc/paraphina`, `/opt/paraphina`,
`/var/lib/paraphina`, or `/home/ubuntu/promotion_runs` unless a live operator
explicitly authorizes that work.

## Risk Profiles And Starting Inventory

Paraphina exposes three coarse risk profiles via `PARAPHINA_RISK_PROFILE`:

- `aggressive`: larger hedge band, smaller sizing eta, higher loss limit.
- `balanced`: default research profile.
- `conservative`: smaller hedge band and tighter loss limit.

The stress harness `batch_runs/exp03_stress_search.py` sweeps starting inventory
and verifies PnL and kill-switch behavior under deterministic paths. For live or
high-fidelity simulation work, keep starting inventory small enough that the
daily loss limit is not immediately consumed by an adverse opening move.
