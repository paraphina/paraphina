# Phase 5 Closeout - 2026-05-01

This note freezes the accepted Phase 5 closeout for future operators and agents.
It is documentation only; it does not authorize a new live run, service restart,
capital escalation, or unattended production mode.

## Final Verdict

- Phase: `5`
- Verdict: `PROMOTE`
- Status: `accepted_closeout`
- Current tranche: `phase5_reopened_final_closeout`
- Evidence tranche: `phase5_reopened_wallet_econ_full_cost_opportunity_adjusted_gate_requal`
- Surface: `19ac6e21020d39d8`
- Promoted at: `2026-05-01T01:44:24Z`
- Frozen at: `2026-05-01T01:44:22Z`
- Active blocker: `none`
- Trade mode after closeout: `shadow`

## Accepted Evidence

- Final run root: `/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_dynamic_size_02_requal_7200s_20260430T220436Z/live_canary`
- Final run window: `2026-04-30T22:04:41.427000Z` to `2026-05-01T00:04:47.813000Z`
- Guard window completed: `true`
- Guard intervened: `false`
- Kill events after closeout: `false`
- Reconcile mismatches after closeout: `0`
- Systemd restarts after closeout: `0`
- Fill evidence: `5` fills / `0.07 ETH`
- Market-making fill evidence venues: `paradex`, `lighter`
- Opportunity-adjusted participation: passed for `hyperliquid`, `aster`, `paradex`, `lighter`, and `extended`
- Completion standard: all five connectors execution-eligible and FV-eligible, no excluded venues, no FV-disabled venues

## Balance PnL Authority

Live-run economics for this closeout are measured by exchange account balance
pre/post comparison, rendered in `./view` as `bPNL`. Telemetry `pnl_total` is
not the economic authority for live runs; it is fallback/context only.

- Repo-local balance comparison file: `/home/ubuntu/paraphina_mm_pnl_harness/phase5/runs/phase5_reopened_final_closeout/balance_snapshot_comparison.json`
- Original balance comparison file: `/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_dynamic_size_02_requal_7200s_20260430T220436Z/live_canary/balance_snapshot_comparison.json`
- Venue count: `5`
- Pre total: `$317.23409775`
- Post total: `$317.24152188`
- Balance delta: `+$0.00742413`
- Absolute balance delta: `$0.00742413`
- Per-venue deltas:
  - `aster`: `$0.00000000`
  - `extended`: `$0.000000`
  - `hyperliquid`: `$0.000000`
  - `lighter`: `+$0.014546`
  - `paradex`: `-$0.00712187`

## Artifact Pointers

- Final topology spec: `/home/ubuntu/paraphina_mm_pnl_harness/phase5/runs/phase5_reopened_final_closeout/final_topology_spec.yaml`
- Stage verdict: `/home/ubuntu/paraphina_mm_pnl_harness/phase5/runs/phase5_reopened_final_closeout/stage_verdict.json`
- Repo-local autoscore bundle: `/home/ubuntu/paraphina_mm_pnl_harness/phase5/runs/phase5_reopened_final_closeout/autoscore_bundle.json`
- Repo-local closeout bundle: `/home/ubuntu/paraphina_mm_pnl_harness/phase5/runs/phase5_reopened_final_closeout/live_closeout_bundle.json`
- Original autoscore bundle: `/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_dynamic_size_02_requal_7200s_20260430T220436Z/live_canary/autoscore_bundle.json`
- Original closeout bundle: `/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_dynamic_size_02_requal_7200s_20260430T220436Z/live_canary/live_closeout_bundle.json`
- Live metrics: `/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_dynamic_size_02_requal_7200s_20260430T220436Z/live_canary/live_metrics.json`
- Balance pre snapshot: `/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_dynamic_size_02_requal_7200s_20260430T220436Z/live_canary/balance_pre_snapshot.json`
- Balance post snapshot: `/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_dynamic_size_02_requal_7200s_20260430T220436Z/live_canary/balance_post_snapshot.json`
- State sync report: `/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_dynamic_size_02_requal_7200s_20260430T220436Z/live_canary/state_sync_report.json`
- Status board: `/home/ubuntu/paraphina_mm_pnl_harness/phase5/status.md`
- Structured queue: `/home/ubuntu/paraphina_mm_pnl_harness/phase5/queue.yaml`

## Operator Constraints

- Do not reopen Phase 5 from older held branches unless a new blocker is
  explicitly recorded in `phase5/status.md` or `phase5/queue.yaml`.
- Do not infer current live topology from Roadmap-B, WP100, or other generated
  historical reports.
- Do not claim unattended production readiness from this closeout. That requires
  a separate future gate.
- Keep runtime in `shadow` unless a live operator explicitly authorizes a new
  guarded workflow.
