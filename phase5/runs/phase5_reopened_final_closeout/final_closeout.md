# Reopened Phase 5 Final Closeout

## Verdict

`ACCEPTED`

The reopened Phase 5 closeout is accepted on surface `19ac6e21020d39d8`.

This is a topology and operational-qualification closeout. It does not claim that future 24/7 economics are solved, only that the live surface is now a valid all5 venue topology for continuous operation under normal sentinel monitoring.

## Frozen Topology

- Runtime binary: `/home/ubuntu/paraphina_mm_pnl_harness/target/release/paraphina_live`
- Runtime SHA256: `7ccdf082ab9629290aad0aec0d1ee177552bff8f7cedd1a4c74fb976560a2c1d`
- Stage overlay: `/home/ubuntu/paraphina_mm_pnl_harness/phase5/runs/phase5_reopened_wallet_econ_full_cost_dynamic_size_02_requal/stage_overlay_live.env`
- Live connectors: `aster,extended,hyperliquid,lighter,paradex`
- FV-disabled venues: `none`
- Excluded venues: `none`

## Frozen Role Matrix

- `hyperliquid`: `fill`
- `aster`: `fill`
- `paradex`: `fill`
- `lighter`: `fill`
- `extended`: `fill`

## Final Long-Soak Evidence

- Tranche: `phase5_reopened_wallet_econ_full_cost_opportunity_adjusted_gate_requal`
- Run root: `/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_dynamic_size_02_requal_7200s_20260430T220436Z/live_canary`
- Segment UTC: `2026-04-30T22:04:41.427000Z -> 2026-05-01T00:04:47.813000Z`
- Ticks analyzed: `29114`
- Guard window completed: `true`
- Guard exit code: `0`
- Guard intervention: `false`
- Kill events post-run: `false`
- Reconcile mismatch post-run: `0`
- Systemd restarts: `0`
- Restored trade mode: `shadow`
- Direct venue audit after restore: `clean across all five venues`
- Account-balance PnL: `0.00742413 USD`
- Absolute balance drift: `0.00742413 USD`

Final 2h executed volume:

- Total fills: `5`
- Total base: `0.07 ETH`
- `hyperliquid`: `0 fills / 0.0 ETH`
- `aster`: `0 fills / 0.0 ETH`
- `paradex`: `2 fills / 0.03 ETH`
- `lighter`: `3 fills / 0.04 ETH`
- `extended`: `0 fills / 0.0 ETH`

Final 2h order activity existed on the accepted surface:

- `hyperliquid`: `25 place intents`, `25 place acks`, `27 cancel intents`, `25 cancel acks`
- `aster`: `26 place intents`, `25 place acks`, `39 cancel intents`, `25 cancel acks`
- `paradex`: `39 place intents`, `38 place acks`, `43 cancel intents`, `40 cancel acks`
- `lighter`: `4 place intents`, `3 place acks`, `3 cancel intents`, `32 cancel acks`
- `extended`: `0 place intents`, `0 place acks`, `0 cancel intents`, `0 cancel acks`

Opportunity-adjusted venue participation:

- `hyperliquid`: `insufficient_active_quote_sample`
- `aster`: `insufficient_active_quote_sample`
- `paradex`: `mm_fill_evidence`
- `lighter`: `mm_fill_evidence`
- `extended`: `no_cost_positive_quote_opportunity`

## Why This Closeout Is Accepted

- No venue remains excluded because of an unresolved connector or platform defect.
- All five venues are execution-eligible on one integrated live surface.
- All five venues are FV-eligible on the integrated surface.
- All five venues have frozen `fill` roles on the accepted surface.
- The final 2h reopened soak was operationally clean.
- Authoritative combined account-balance PnL was `0.00742413 USD` across `5` venue accounts.
- Opportunity-adjusted participation passed all five venues; MM fill evidence venues were `paradex,lighter`.
- Direct venue truth after restore showed zero positions and zero open orders across all five venues.

## Known Caveats

- Final 2h `would_send_zero_pct` remained high at `99.2684`; economics and sizing optimization should continue on top of this clean topology.
- The final 2h sample had `0` fills on `hyperliquid,aster,extended`, so future monitoring should keep checking venue-balance drift rather than assuming every venue fills in every window.

## Evidence Files

- Frozen spec: `/home/ubuntu/paraphina_mm_pnl_harness/phase5/runs/phase5_reopened_final_closeout/final_topology_spec.yaml`
- Final 2h summary: `/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_dynamic_size_02_requal_7200s_20260430T220436Z/live_canary/live_segment_summary.json`
- Final 2h metrics: `/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_dynamic_size_02_requal_7200s_20260430T220436Z/live_canary/live_metrics.json`
- Final 2h report: `/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_dynamic_size_02_requal_7200s_20260430T220436Z/live_canary/telemetry_report_live_segment.md`
- Final 2h guard: `/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_dynamic_size_02_requal_7200s_20260430T220436Z/live_canary/guard.log`
