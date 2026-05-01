# Phase 5.1 Execution Charter

This charter defines the autonomous Phase 5.1 workflow. It is documentation
only and does not authorize live orders, a canary, capital escalation, risk-limit
relaxation, or runtime service changes.

## Baseline

- Branch: `mm-pnl-harness-clean`
- Phase 5 closeout baseline: `18dd09512288a85e440d3977e32432c3aabc1190`
- Phase 5.1 scaffold baseline: `47c2f02bf3a7099d03a51731eda5cd8c73b8a233`
- Initial evidence run: `runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_47c2f02`
- Initial evidence status: `HOLD`
- Initial evidence scope: Lighter-only non-live EV shadow over real Phase 5
  telemetry tail

## Non-Negotiable Scope

- No live order placement.
- No canary promotion.
- No service restart or runtime cutover.
- No capital escalation.
- No risk-limit relaxation.
- No economic claim unless reconciled to exchange balance deltas under a
  separately authorized live evidence lane.
- `true_edge = max(local_edge, pair_edge)` remains rejected as an admission
  objective.
- `Q_raw = true_edge / eta` remains rejected as a sizing rule.

## Board

- Executive Orchestrator: owns sequencing, scope control, gate decisions,
  integration, commits, and final synthesis.
- Systems Implementation Lead: owns non-live replay/shadow tooling, schema
  integration, deterministic artifacts, and focused CI.
- Quant EV Lead: owns EV objective fidelity, confidence-bounded admission,
  discrete sizing, calibration gaps, and falsifiability.
- Risk/Evidence Auditor: owns no-live guarantees, reproducibility, evidence
  manifests, schema validity, and board-performance audits.
- Execution Microstructure Lead: activated only when venue-specific official
  documentation or connector semantics become the current blocker.

## Operating Loop

Every milestone follows this loop:

1. Capture repo/evidence baseline.
2. Assign bounded read-only audits where parallel review is useful.
3. Implement the smallest non-live patch that advances the current milestone.
4. Run focused validation.
5. Produce or refresh a non-live evidence artifact when relevant.
6. Run the Risk/Evidence Auditor challenge.
7. Commit and push only if the patch is clean and reproducible.
8. Record the next blocker or next milestone.

## Milestones

- M1: Freeze scaffold and initial evidence baseline. Status: `complete`.
- M2: Expand `V2_EV_EVALUATED` and candidate extraction so real Phase 5
  telemetry emits richer replayable candidate context, while calibrated EV
  remains `HOLD`. Status: `complete`.
- M3: Add replay labels and deterministic source references for observed facts,
  model estimates, and counterfactual decisions. Status: `partial`; current
  support emits source-linked counterfactual decision labels only.
- M4: Add sparse-bucket calibration placeholders and explicit `HOLD` reason
  taxonomy for missing P-fill, markout, hedge, queue, churn, and tail evidence.
  Status: `complete`.
- M5: Add Lighter venue-readiness evidence pack covering post-only semantics,
  account/profile assumptions, fees, rate limits, and connector observability.
  Status: `complete_for_nonlive_evidence_pack`; promotion status remains `HOLD`
  for live, canary, capital, or economic claims.
- M6: Add risk/system invariant tests for no-live enforcement, metadata
  propagation, residual-state placeholders, and double-action prevention
  preconditions.
  Status: `complete_for_nonlive_scaffold`; schema v2 and shadow artifacts now
  carry false live/canary/capital/risk authorization fields plus no-action
  residual and double-action precondition states. The validator rejects unsafe
  v2 authorization values.
- M7: Run a Phase 5.1 non-live evidence pack on a bounded real Phase 5 telemetry
  segment and validate schema plus manifest.
- M8: Produce board decision: `PROMOTE_FOR_NEXT_NONLIVE_STEP`, `HOLD`, or
  `REJECT`.

## Audit Cadence

The Risk/Evidence Auditor runs after every material patch, every evidence-pack
run, and every commit candidate. The auditor must explicitly answer:

- Is the current critical path still optimal?
- Is any lane doing speculative or duplicate work?
- Did any change touch live execution, canary, capital, or risk-limit behavior?
- Are artifacts reproducible from commit, config, input telemetry, and command?
- Are schema and manifest checks passing?
- Is the next milestone blocked by missing data, missing code, or missing venue
  evidence?

## Completion Standard

Phase 5.1 is complete only when the repository contains a committed and pushed
non-live implementation that can replay/shadow real Phase 5 telemetry, emit
validated telemetry schema v2 records, preserve no-live/no-capital/no-risk
guards, document calibration and venue-readiness gaps, and produce a board
decision before any Phase 6 or live/canary discussion.
