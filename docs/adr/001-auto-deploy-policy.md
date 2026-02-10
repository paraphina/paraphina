# ADR-001: Graduated Auto-Deploy Pipeline

**Status:** Accepted  
**Date:** 2026-02-09  
**Deciders:** Engineering

## Context

Paraphina's promotion pipeline (Phase A random-search + Pareto + budget gating,
Phase B bootstrap confidence gate) produces fully validated, evidence-packed
configuration candidates. However, the final step — deploying a promoted `.env`
file to the live VPS and restarting the service — has been entirely manual:
`scp` the file, `systemctl restart`, and watch logs.

This manual last-mile creates several problems:

1. **Latency.** Hours or days can pass between a PROMOTE decision and the config
   being live, even when every quantitative gate has passed.
2. **Human error.** Manual `scp` to the wrong path, forgetting to restart, or
   sourcing the wrong `.env` file are realistic failure modes.
3. **No rollback discipline.** There is no structured mechanism to revert to a
   previous known-good config; operators must remember which file was previously
   active.

At the same time, deploying untested config directly to live trading is
unacceptable. A pathological config could trigger hard risk breaches, exhaust
daily loss limits, or cause reconciliation drift before a human notices.

## Decision

We introduce a **graduated auto-deploy pipeline** that progresses a promoted
config through four stages, each with a configurable soak period and an
automated health gate. Any gate failure triggers immediate rollback to the
previous known-good config.

### Stages

| Stage | Trade Mode | Duration | Health Predicate |
|-------|-----------|----------|------------------|
| 1. Shadow Soak | `shadow` | configurable (default 5 min) | Process alive, telemetry flowing, no kill events |
| 2. Paper Soak | `paper` | configurable (default 10 min) | PnL within expected CI, no kills, drawdown within budget |
| 3. Canary Live | `live` + `CANARY_MODE=1` | configurable (default 15 min) | No canary breaches, reconciliation clean, PnL non-negative |
| 4. Full Live | `live` (normal limits) | continuous | Ongoing Prometheus monitoring |

### Safety Invariants

1. **Rollback SLA:** From anomaly detection to previous config running < 60 seconds.
2. **Symlink rotation:** The active config is always a symlink (`current.env`)
   pointing to a concrete timestamped `.env` file. Rollback swaps the symlink
   to `previous.env`'s target and restarts the service.
3. **Config validation:** Before any deployment, the `paraphina_live --validate-config`
   check must pass for the candidate config. This loads the config, constructs
   Engine + GlobalState, runs one synthetic tick, and verifies no immediate kill
   switch trip or NaN values in volatility precomputed parameters.
4. **Kill switch as safety net:** The existing latching kill switch
   (`KillReason::*`) remains the last line of defense. If the kill switch trips
   during any soak stage, the deploy orchestrator treats it as a gate failure
   and rolls back.
5. **No aggressive auto-deploy:** Initially, only conservative and balanced
   tiers are eligible for auto-deploy. Aggressive tier requires explicit manual
   approval.
6. **No hot-reload:** Config changes always require a process restart, ensuring
   clean state initialization. This is intentional.

### What Remains Manual

- **Escalation from canary to full live** can optionally require human approval
  (configurable via `--require-approval-for-live`).
- **Aggressive tier deployment** always requires manual approval.
- **First-ever deployment** to a new VPS requires manual setup of the symlink
  structure and systemd unit.

### Audit Trail

Every deployment writes a `deploy_state.json` recording:
- Active and previous config IDs
- Deployment timestamp and stage progression
- Gate pass/fail results at each stage
- Rollback events with reasons
- Git commit hash and promotion record reference

This complements the existing `PROMOTION_RECORD.json` evidence chain.

## Consequences

### Positive

- Config changes reach production in ~30 minutes instead of hours/days.
- Structured rollback eliminates "which `.env` was I running?" confusion.
- The graduated soak catches venue-specific edge cases, API drift, and clock
  issues that pure simulation cannot detect.
- Full audit trail satisfies institutional compliance requirements.

### Negative

- More infrastructure to maintain (deploy orchestrator, config manager).
- The soak periods add latency compared to a direct manual deploy by an expert
  who is confident in the config.
- systemd service restarts during soak stages cause brief trading gaps (~5s
  each, 4 restarts total = ~20s of downtime during the rollout).

### Risks

- A bug in the deploy orchestrator itself could cause a bad deploy. Mitigated
  by the kill switch and dry-run mode.
- Network issues between CI and the VPS could leave the pipeline in a partially
  deployed state. Mitigated by the state file and idempotent rollback.

## Phasing

- **Phase 1:** Config lifecycle manager + symlink rotation + validation.
  Manual trigger only — better tooling for the existing manual process.
- **Phase 2:** Deploy orchestrator with shadow + paper soak. No real capital
  at risk. Human approves canary -> live.
- **Phase 3:** Full graduated auto-deploy including canary soak.
- **Phase 4:** CI-triggered auto-deploy on PROMOTE decision.
