# Paraphina Architecture

This document maps the current repository architecture. It is intentionally
separate from `ROADMAP.md`: the roadmap decides priority and direction; this
file explains what the system contains today and how the pieces fit together.

## Status Vocabulary

Use these terms consistently:

- `implemented` - repository code exists and is covered by tests or checks.
- `partially implemented` - scaffolding exists, but the production contract is
  incomplete or not fully validated.
- `shadow-only` - safe observation/replay path; no live order placement implied.
- `experimental` - research/support-track behavior that is not promoted.
- `planned` - roadmap intent without complete implementation.
- `historical/generated` - dated artifact; useful for provenance, not current
  truth unless regenerated.

## High-Level Shape

Paraphina has four major layers:

1. Strategy and risk core
   - deterministic state transitions;
   - fair value, volatility, inventory, PnL, risk regime, kill-switch logic;
   - market-making, cross-venue exits, and hedging.

2. Research and simulation
   - deterministic sim binary;
   - batch experiment harnesses;
   - telemetry and evidence-pack checks.

3. Live runtime
   - feature-gated live binary;
   - connector abstractions;
   - shadow, paper, testnet, and live trade modes;
   - guardrails for live execution.

4. Phase 5 qualification workflow
   - serialized live-affecting tranche mainline;
   - support tracks for research/forensics/tooling;
   - structured queue, status, scoring, and closeout artifacts.

## Core Rust Surfaces

| Area | Current status | Primary files |
|---|---|---|
| Public module map | implemented | `paraphina/src/lib.rs` |
| Simulation entrypoint | implemented | `paraphina/src/main.rs` |
| Config and risk profiles | implemented | `paraphina/src/config.rs` |
| State and accounting | implemented | `paraphina/src/state.rs` |
| Main engine tick | implemented | `paraphina/src/engine.rs` |
| Market-making quotes | implemented | `paraphina/src/mm.rs` |
| Cross-venue exits | implemented | `paraphina/src/exit.rs` |
| Hedging | implemented | `paraphina/src/hedge.rs` |
| Strategy action loop | implemented | `paraphina/src/strategy_action.rs` |
| Order-management planning | implemented, evolving | `paraphina/src/order_management.rs` |
| Telemetry | implemented, contract-sensitive | `paraphina/src/telemetry.rs`, `schemas/` |
| RL runner | shadow/research-oriented | `paraphina/src/rl/runner.rs` |

## Live Runtime Surfaces

| Area | Current status | Primary files |
|---|---|---|
| Live binary | implemented behind feature flags | `paraphina/src/bin/paraphina_live.rs` |
| Trade modes | implemented; default is shadow | `paraphina/src/live/trade_mode.rs` |
| Canonical venue order | implemented | `paraphina/src/live/venues.rs` |
| Connector abstraction | implemented/evolving | `paraphina/src/live/connector.rs`, `paraphina/src/live/gateway.rs` |
| Venue connectors | implemented/evolving | `paraphina/src/live/connectors/**` |
| Live telemetry/replay tests | implemented/evolving | `paraphina/tests/live_*` |

Live execution is operationally gated. The presence of a live connector or live
binary does not imply unattended market-making production readiness.

## Canonical Venues

The live venue registry is code-owned by `paraphina/src/live/venues.rs`.
Roadmap and runbook prose must follow that registry rather than invent a
separate order.

The current canonical set is:

1. Extended
2. Hyperliquid
3. Aster
4. Lighter
5. Paradex

## Research And Telemetry Surfaces

| Area | Current status | Primary files |
|---|---|---|
| Batch orchestration | implemented | `batch_runs/orchestrator.py` |
| Metrics parsing | implemented | `batch_runs/metrics.py` |
| Experiment templates | implemented/historical | `batch_runs/exp*.py` |
| Telemetry contract checks | implemented | `tools/check_telemetry_contract.py`, `schemas/` |
| Docs integrity checks | implemented | `tools/check_docs_integrity.py` |
| Evidence packs | implemented | `docs/EVIDENCE_PACK.md`, `sim_eval` surfaces |

Telemetry fields and stdout parsers are research and audit contracts. Do not
rename or weaken them without updating tests and contract docs.

## Phase 5 Surfaces

`phase5/status.md`, `phase5/queue.yaml`, and `phase5/control_pack.yaml` are
read-only context for documentation work unless a live operator explicitly asks
for runtime-state edits.

Use them this way:

- `phase5/status.md` - current human-readable status board.
- `phase5/queue.yaml` - structured tranche lineage and current queue.
- `phase5/control_pack.yaml` - control artifacts and defaults; reconcile with
  status and queue before treating it as current topology.

Do not use dated WP100 or Roadmap-B reports to infer current Phase 5 topology.
As of the Phase 5 freeze, `phase5_reopened_final_closeout` is promoted on
surface `19ac6e21020d39d8` with active blocker `none`; this is a completed
Phase 5 closeout, not a claim of unattended production readiness.

## Whitepaper Scope

`docs/WHITEPAPER.md` is both a current implementation narrative and a preserved
canonical target-spec appendix. The appendix is protected by the docs integrity
gate. If the current implementation has moved beyond an old appendix status
annotation, correct the current-state note before the canonical marker instead
of casually editing the hash-locked appendix.

Whitepaper completion means implementation parity for the audited whitepaper
scope. It does not mean:

- validated unattended production market making;
- permission to bypass live guardrails.

## Agent Guidance

For new work:

1. Read `README.md`, `ROADMAP.md`, this file, and `docs/AGENT_START_HERE.md`.
2. Locate code evidence with `rg` before making claims.
3. Treat `docs/ROADMAP_B_READINESS_REPORT.md`, `docs/WP_PARITY_MATRIX.md`, and
   `docs/WHITEPAPER_COMPLETION_*` as historical/generated unless regenerated.
4. Keep docs edits separate from runtime/code edits unless explicitly asked.
5. Run `python3 tools/check_docs_integrity.py` after roadmap or whitepaper
   changes.
