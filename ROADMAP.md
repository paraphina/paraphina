# Paraphina Roadmap

This is the single authoritative roadmap for Paraphina. It is both a strategic
truth map and an execution-control document: it says what the system is trying
to become, what evidence is required to promote each target, and what future
agents must do next without creating competing roadmap files.

Whitepaper completion is not equivalent to final all-5 live topology or
unattended market-making production readiness. WP100 proves a dated
code-visible implementation scope; Phase 5 evidence proves live operational
readiness.

---

## 1. Roadmap Contract

### 1.1 Authority Order

When documentation, code, and runtime state disagree, use this order:

1. Executable code and tests.
2. Structured Phase 5 state: `phase5/queue.yaml`.
3. Current human status board: `phase5/status.md`.
4. Current docs: `ROADMAP.md`, `docs/ARCHITECTURE.md`, `docs/WHITEPAPER.md`
   Part I, and `docs/AGENT_START_HERE.md`.
5. Supporting docs: runbooks, runtime objective docs, and Phase 5 workflow docs.
6. Historical/generated docs: WP100 artifacts, parity matrices, completion
   audits, and generated readiness reports.

Do not use older generated reports to infer current live topology. Do not use
whitepaper completion as permission to bypass live guardrails.

### 1.2 Execution Discipline

Every execution-roadmap row must have:

- `target`
- `current status`
- `next action`
- `required evidence`
- `promotion condition`
- `hold/rollback condition`
- `lane`
- `source files`

If one of those fields is missing, the item is not ready for autonomous
execution.

### 1.3 External Engineering Principles

Paraphina's Phase 5 execution model follows two external operating principles:

- Canary changes must be small, time-limited, measured against a control, and
  evaluated with attributable metrics before rollout continues. This matches the
  Google SRE canarying model:
  https://sre.google/workbook/canarying-releases/
- Live readiness must include the ability to anticipate, withstand, recover
  from, and adapt to adverse conditions. This matches the NIST cyber-resiliency
  framing:
  https://csrc.nist.gov/pubs/sp/800/160/v2/r1/final

These references do not override repo evidence. They explain why the roadmap
uses small hypotheses, short-to-long rungs, explicit rollback criteria, and
structured closeout evidence.

## 2. Source-Of-Truth Map

| Surface | Role | Authority |
|---|---|---|
| `README.md` | Entry point and documentation hierarchy | Current |
| `ROADMAP.md` | Strategic roadmap and execution-control stack | Authoritative |
| `docs/ARCHITECTURE.md` | Current implementation architecture and status map | Authoritative for docs |
| `docs/AGENT_START_HERE.md` | Codex/human onboarding path and safety rules | Authoritative for workflow orientation |
| `docs/WHITEPAPER.md` Part I | Algorithmic implementation narrative | Current where code-backed |
| `docs/WHITEPAPER.md` Part II | Canonical target spec appendix | Hash-locked historical target/spec context |
| `docs/V2_SPECIFICATION.md` | Phase 5.1/V2 fill-aware, hedge-aware, arbitrage-informed target specification | Active target spec; execution still gated by this roadmap |
| `docs/ROADMAP_B.md` | Live venue rollout assumptions and gates | Supporting, not the roadmap |
| `docs/ROADMAP_B_READINESS_REPORT.md` | Generated connector snapshot | Historical unless regenerated |
| `docs/WP100_SIGNOFF.md` | WP100 audit signoff | Historical audit scope |
| `docs/WP_PARITY_MATRIX.md` | Code-visible whitepaper parity matrix | Historical/generated audit scope |
| `docs/WHITEPAPER_COMPLETION_*` | WP100 audit outputs | Historical/generated audit scope |
| `docs/WORLD_CLASS_RUNTIME_GOALS.md` | Runtime objective and gate framework | Supporting objective framework |
| `docs/PHASE5_AUTONOMOUS_TRANCHE_SYSTEM.md` | Phase 5 workflow model | Supporting workflow spec |
| `docs/PHASE5_1_TO_24_7_ORCHESTRATION_RUNBOOK.md` | Phase 5.1b-to-24/7 autonomous orchestration and resume procedure | Supporting workflow spec |
| `phase5/status.md` | Human-readable Phase 5 status board | Read-only current runtime context |
| `phase5/queue.yaml` | Structured Phase 5 queue and tranche lineage | Read-only current runtime context |
| `phase5/control_pack.yaml` | Promoted/control artifacts and automation defaults | Read-only control context; may lag status/queue |

## 3. Current System State

Paraphina is no longer only a deterministic simulator. The repository contains:

- a Rust strategy/risk/accounting core under `paraphina/src/`;
- a deterministic sim entrypoint at `paraphina/src/main.rs`;
- a feature-gated live runner at `paraphina/src/bin/paraphina_live.rs`;
- canonical live venue order in `paraphina/src/live/venues.rs`;
- shadow, paper, testnet, and live trade-mode plumbing under `paraphina/src/live/`;
- order-management, replay, telemetry, connector, and safety tests under
  `paraphina/tests/`;
- research and audit tooling under `batch_runs/` and `tools/`;
- Phase 5 tranche state and workflow definitions under `phase5/` and
  `docs/PHASE5_AUTONOMOUS_TRANCHE_SYSTEM.md`.

Current live topology must be read from `phase5/status.md` and
`phase5/queue.yaml`, not from older roadmap or WP100 artifacts. As of the
current Phase 5 context, docs must distinguish all-5 connector eligibility from
unattended production readiness.

## 4. Execution Snapshot

Current Phase 5 freeze snapshot at `2026-05-01T01:44:24Z`: Phase 5 is
promoted and accepted-closeout on serialized-mainline tranche
`phase5_reopened_final_closeout`, computed surface `19ac6e21020d39d8`. The
active blocker is `none`. The accepted evidence is the final guarded `7200s`
run from `2026-04-30T22:04:41.427000Z` to
`2026-05-01T00:04:47.813000Z`, with all five connectors execution-eligible and
FV-eligible, no excluded venues, no FV-disabled venues, clean closeout, no guard
intervention, no kill events, no reconcile mismatches, and service restored to
`shadow`. The final economic authority is the five-account balance snapshot:
`balance_delta_usd=+0.00742413`, `venue_count=5`, and drift within the accepted
budget. Fill evidence in the final run was `5` fills / `0.07 ETH`, with Paradex
and Lighter providing market-making fill evidence and the opportunity-adjusted
scorecard passing for Hyperliquid, Aster, Paradex, Lighter, and Extended.

The frozen closeout packet is documented in
`docs/PHASE5_CLOSEOUT_2026-05-01.md`. Future work must not reopen Phase 5 from
older hold narratives unless a new explicit blocker is recorded in
`phase5/status.md` or `phase5/queue.yaml`. Phase 5 completion does not by
itself grant unattended production-readiness or capital escalation; those are
separate future gates.

Historical reopened-economics trail: at `2026-04-30T13:26:43Z`, the latest
serialized economics child
`phase5_reopened_wallet_econ_full_cost_ext_pdx_bid_edge_requal` completed its
final `7200s` rung cleanly on computed surface `19880f6294f71ee8` and restored
the host to shadow mode, but held on the final promotion gate. Clean and
mechanism gates passed; promotion failed because
`metrics.economics_attribution.net_after_hedge_exec_model_usd=-0.568865` missed
the `>= -0.50` gate, and Hyperliquid/Paradex had zero market-making fills. The
combined balance snapshot passed with `delta_usd=-0.02683875` and
`abs_delta_usd_float=0.02683875`. The active successor is
`phase5_reopened_wallet_econ_full_cost_ext_harden_pdx_reentry_requal`: it keeps
topology, quote sizes, canary caps, terminal controls, source-truth attribution,
purpose-aware fill gates, and `PARAPHINA_MM_HEDGE_COST_EDGE_MULT=1.0` fixed
while changing only `PARAPHINA_MM_EDGE_LOCAL_MIN_EXTENDED_BID=1.79` and
`PARAPHINA_MM_EDGE_LOCAL_MIN_PARADEX_BID=0.02` on computed surface
`d4792dbf37283dcc`. Its current-surface `shadow_smoke_10m` support gate passed
at `2026-04-30T13:21:50Z`; live admission is now held only by Hyperliquid
address quota (`nRequestsUsed=71983`, `nRequestsCap=67835`), because
`reserveRequestWeight` is not permitted through the vault/subaccount envelope.
The prepared next successor
`phase5_reopened_wallet_econ_full_cost_dynamic_size_02_requal` keeps that
surface fixed except for uniformly raising the five
`PARAPHINA_MM_MAX_QUOTE_SIZE_TAO_*` venue ceilings from `0.01` to `0.02` on
computed surface `19ac6e21020d39d8`; it remains blocked from any live rung until
Hyperliquid quota is above both the action cap and the Phase 5 duration-scoped
runway guard.

The first full-cost child `phase5_reopened_wallet_econ_full_cost_requal` is now
held on computed surface `111531975f916424`. Its `300s` canary
`phase5_reopened_wallet_econ_full_cost_requal_5m_canary_20260429T141152Z` completed
the guard window and restored shadow, but it is not promotion-capable: first
pre-restore venue audit was dirty (`lighter open_order_count=1`, `aster
position_base=-0.01 ETH`), cleanup was required, final inventory was
`final_q_global_tao=-0.01`, telemetry `final_pnl_total=-0.016004`, and the combined
balance snapshot delta was `-$0.03195220`. Forensics found the child overlay had
accidentally dropped the half-cost parent tail controls, including terminal-exit
quiesce env, service-visible terminal signal, Aster terminal explicit cancel
retry, Lighter/Aster/Extended stale-state controls, and market-rx/private-order
audit controls. This hold is therefore classified as a `restore_hygiene` overlay
regression, not as a clean full-cost strategy/economics result.

The corrected full-cost terminal-controls child
`phase5_reopened_wallet_econ_full_cost_terminal_controls_requal` is now held on
computed surface `931a4bfaf682d9a6`. Its `7200s` final rung ran from
`2026-04-29T15:16:47.929000Z` to `2026-04-29T17:16:54.589000Z`, completed the
guard window, restored shadow cleanly, observed terminal quiesce, required no
pre-restore cleanup, and ended flat with `final_q_global_tao=0.0`. Promotion held
because `final_pnl_total=-0.005221`,
`metrics.economics_attribution.net_after_hedge_exec_model_usd=-0.514065`,
Hyperliquid and Extended had zero fills, and the combined balance delta was
`-$0.47231712` within but close to the `0.50 USD` drift budget. The hold is
classified as `capital_preservation_residual_markout` with secondary
microstructure underconversion, not restore hygiene.

The Aster bid-edge child
`phase5_reopened_wallet_econ_full_cost_aster_bid_edge_016_requal` is now held on
computed surface `f34870ff704188c7`. Its `1200s` rung was operationally clean but
exposed an attribution-integrity gap: old headline
`metrics.economics_attribution.net_after_hedge_exec_model_usd=0.0` ignored
`hedge_fill_unattributed_count=7`,
`hedge_exec_cost_model_unattributed_usd=0.339404`, and
`hedge_total_cost_model_unattributed_usd=0.632323`. A controlled early stop of
the `7200s` attempt restored shadow cleanly after guard cleanup flattened an
Extended `0.02 ETH` residual; the post-clean direct venue audit was clean across
all five accounts.

The attribution-hardening child
`phase5_reopened_wallet_econ_full_cost_hedge_attribution_gate_requal` is now held
on computed surface `f470db3c6feeb66f`. Its live strategy overlay was unchanged
from the Aster bid-edge child; the change axis was only attribution/gate
hardening. `tools/telemetry_analyzer.py` now charges unattributed hedge execution
cost into top-level `net_after_hedge_exec_model_usd`, and the tranche
hard-holds continuation when hedge attribution is incomplete or material
unattributed hedge execution cost remains.

The source-truth child
`phase5_reopened_wallet_econ_full_cost_hedge_source_truth_requal` is now held.
Its 300s canary on run surface `f470db3c6feeb66f` passed the hedge source-truth
mechanism (`hedge_fill_unattributed_count=0`,
`hedge_exec_cost_model_unattributed_usd=0.0`) but held because the first
pre-restore direct venue audit found an Extended `+0.01 ETH` residual with zero
open orders. That hold is routed as `exact_one_lot_residual_no_orders`.

The exact-lot terminal child
`phase5_reopened_wallet_econ_full_cost_extended_terminal_exact_lot_requal` is now
held after its `7200s` final rung on run surface `6630687e7bc06344`. The segment
completed operationally clean, restored shadow cleanly, left the first
pre-restore and post-rollback direct venue audits clean, ended flat with
`final_q_global_tao=0.0`, and produced a combined balance delta of
`+$0.07004607`. It held because residual hedge telemetry still left
`hedge_fill_unattributed_count=3` and
`hedge_exec_cost_model_unattributed_usd=0.270508`, while the old raw fill-count
promotion gate did not separate market-making fills from hedge fills.

The restore-hygiene child
`phase5_reopened_wallet_econ_full_cost_extended_account_truth_requal` is held on
computed surface `b554afdb42647184` after clearing the restore/account-truth
mechanism and proving clean first-pre-restore and post-rollback venue audits.
That final hold is classified as `capital_preservation_residual_markout`, not
restore hygiene. The dynamic-size full-cost economics child then held cleanly
after its `7200s` rung with combined balance delta `+$0.00742413`; the current
serialized mainline is
`phase5_reopened_wallet_econ_full_cost_opportunity_adjusted_gate_requal` on
computed surface `19ac6e21020d39d8`. The intervening ask-activation child
`phase5_reopened_wallet_econ_full_cost_extended_ask_activation_requal` held on
computed surface `56ea3076a0010943`: its guarded `300s` rung was operationally
clean with authoritative balance-delta PnL `+$0.00075517`, and its guarded
`1200s` rung was operationally clean with balance-delta PnL `-$0.01210395`, but
live-only telemetry showed Extended Bid/Ask suppressed on every live row by the
full hedge-cost edge floor, so a `7200s` unchanged rung was not optimal. The
current gate-only requalification uses the already completed dynamic-size
`7200s` evidence, the regenerated `opportunity_adjusted_scorecard`, and
authoritative five-account balance-delta PnL before refreshing
`phase5_reopened_final_closeout` on computed surface `19ac6e21020d39d8`.

Snapshot source: `phase5/queue.yaml`, `phase5/status.md`, current host/runtime
checks, parent half-cost artifacts under
`/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_edge_requal_7200s_20260429T113605Z/live_canary/`,
full-cost support-gate artifacts under
`/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_requal_shadow_smoke_10m_20260429T135859Z/`,
the held full-cost canary artifacts under
`/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_requal_5m_canary_20260429T141152Z/live_canary/`,
the held corrected full-cost final-rung artifacts under
`/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_terminal_controls_requal_7200s_20260429T151642Z/live_canary/`,
the held Aster bid-edge artifacts under
`/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_aster_bid_edge_016_requal_20m_soak_20260429T175400Z/live_canary/`
and
`/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_aster_bid_edge_016_requal_7200s_20260429T181739Z/live_canary/`,
the held source-truth canary artifacts under
`/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_hedge_source_truth_requal_5m_canary_20260429T200958Z/live_canary/`,
the dynamic-size full-cost `7200s` artifacts under
`/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_dynamic_size_02_requal_7200s_20260430T220436Z/live_canary/`,
the ask-activation `1200s` artifacts under
`/home/ubuntu/promotion_runs/phase5_reopened_wallet_econ_full_cost_extended_ask_activation_requal_20m_soak_20260501T004410Z/live_canary/`,
and tranche card
`phase5/runs/phase5_reopened_wallet_econ_full_cost_opportunity_adjusted_gate_requal/tranche_card.yaml`.

An earlier serialized-mainline child
`phase5_all5_current_surface_aster_short_force_flat_convergence_requal` is held
with preserved run surface `9eb28c7eb88fcfa1` and refreshed control-plane surface
`b44a9a45d7a706c8` after the matched fail route was added. Its `300s` canary from
`2026-04-27T01:23:30.622000Z` to `2026-04-27T01:28:31.728000Z` produced all-five
fills and positive `final_pnl_total=0.051386`; Aster was inside tolerance after
the short force-flat activation. The run is not promotion-capable because first
pre-restore audit was dirty from Extended `position_base=-0.01` with zero open
orders, pre-restore cleanup was required, estimated cleanup cost was
`0.28550690000000034`, and `final_q_global_tao=-0.018`.

The Extended one-lot state-fallback child then completed support requalification
and a `300s` guarded canary from `2026-04-27T02:29:42.648000Z` to
`2026-04-27T02:34:43.804000Z` on rebuilt surface `f4943e494189b28a`. It held,
not promoted: Extended was flat with zero open orders in the first audit and no
Extended size-exceeds-position reject was identified, but first pre-restore
audit was dirty from Hyperliquid one open order, Lighter two open orders, and
Aster `+0.01 ETH`; cleanup was required, final_q_global_tao was `0.01`, and
final_pnl_total was `-0.048065`.

The Aster min-adverse child
`phase5_all5_current_surface_aster_markout_min_adverse_025_requal` then completed
a `300s` guarded canary from `2026-04-27T03:01:32.067000Z` to
`2026-04-27T03:06:33.518000Z`. It held, not promoted: the guard completed and
post-rollback audit was clean, but first pre-restore audit was dirty from Lighter
two open orders, Aster `+0.01 ETH` plus one open order, and Paradex one open
order; cleanup was required with estimated cost `0.2391465 USD`, final inventory
was `0.01 ETH`, and fills were Aster-only `5` / `0.05 ETH`. The Aster threshold
did not prove the mechanism because adverse_markout_usd max was only `0.00085`,
cleanup_fee_estimate_usd max was `0.009609`, and allowed_orders was `0`.

The terminal-exit child
`phase5_all5_current_surface_terminal_exit_live_order_drain_requal` then ran
through 300s and 1200s rungs on computed surface `b4af7363c8d64091`. The 300s
rung was clean enough to continue. The 1200s rung from
`2026-04-27T09:26:38.103000Z` to `2026-04-27T09:46:39.717000Z` held, not
promoted: all five venues filled, `final_q_global_tao=0.0`, `final_pnl_total=0.0`,
no kill/reconcile/restart alarm was recorded, and current direct venue audit is
flat with zero open orders, but the first pre-restore audit was dirty because
Extended still had `open_order_count=1`. Guard evidence shows the terminal signal
was written at `2026-04-27T09:45:39Z`, while preserved runtime artifacts contain
no `terminal_exit_quiesce` telemetry; the service unit has `PrivateTmp=yes`, so
the `/tmp` signal was not visible inside the service namespace.

The service-visible terminal-signal child
`phase5_all5_current_surface_terminal_exit_service_visible_signal_requal` then
held on computed surface `0be1adb1a04a1a42` before the terminal-exit window. Its
latest segment ran from `2026-04-27T10:53:53.489Z` to
`2026-04-27T11:43:39.534Z` with 93 fills, final inventory `0.0`, and
`final_pnl_total=-0.000464`, but Extended degraded-stream rebootstrap churn
escalated to an `8000ms` reconnect sleep, crossed the unchanged `3000ms` state
stale boundary, and triggered `StaleMarket` plus guard intervention. The
terminal-signal visibility hypothesis was not measured because the run died
early.

An earlier Aster terminal quiesce explicit-cancel retry child
`phase5_all5_current_surface_aster_terminal_quiesce_explicit_cancel_retry_requal`
held after completing the 7200s rung operationally clean: first pre-restore audit
was clean, cleanup was not required, final inventory was flat, no kill/restart or
reconcile mismatch was observed, and post-rollback venue audit was clean. The only
remaining promotion blockers were `final_pnl_total=-0.002061` and
`execution_scorecard.extended.fills=0`.

The promoted reversible child is now
`phase5_all5_current_surface_runner_realtime_missed_tick_burst_requal`, with
computed surface `8faedf4a9701d5e0`. Its held parent
`phase5_all5_current_surface_terminal_account_refresh_router_requal` proved
all-five fill conversion but held when a delayed realtime runner interval replayed
missed ticks into a stale-hygiene burst before Extended recovery could be
observed. The promoted child froze the all-five surface and changed only runner
missed-tick behavior to skip stale interval replay. The active eligible gate is
`phase5_reopened_multi_venue_long_soak_runner_realtime_requal` on this exact
promoted surface.

Phase 5 control-plane advancement requires read-only `audit-state-sync`
coherence at both orchestration preflight and promotion handoff; warnings,
critical findings, or validator exceptions hold the tranche rather than allowing
a promotion-capable lane to start or advance.

Current live topology context:

- Live connectors: `aster`, `extended`, `hyperliquid`, `lighter`, `paradex`.
- FV-disabled venues: `none`.
- Excluded venues: `none`.
- Venue roles: all five listed as `fill`.
- Current serialized mainline:
  `phase5_reopened_wallet_econ_full_cost_extended_ask_activation_requal` is
  `in_progress` on computed surface `56ea3076a0010943`. It changes only
  `PARAPHINA_MM_MAX_GENERATED_SPREAD_BPS_EXTENDED=13.0` from the held
  dynamic-size parent, preserves the high Extended bid floor, and is using
  guarded `300s/1200s/7200s` rungs to test whether Extended ask activation can
  improve all-five fill conversion without reopening the prior loss-dominant
  Extended bid-side flow. The linked `phase5_reopened_final_closeout` child
  remains blocked on computed surface `b39a95cc3b1de91e` until this child
  completes the ladder and refreshes the final closeout evidence pack.
- Current blocker family: `microstructure_underconversion`; the active child
  is testing Extended ask-side activation on the same
  all-five topology, source-truth, and wallet-economics surface.
- Historical blocker lineage is preserved in `phase5/queue.yaml`, the final
  child artifacts, and the queue ledger in Section 6. Do not treat older held
  child prose as the active control plane.

The `Baseline ID` in `phase5/status.md` still contains old 4v wording. Treat
the topology fields and queue lineage as stronger than that label.

| Target | Current status | Required evidence | Blocking risk | Next action | Lane |
|---|---|---|---|---|---|
| T0 Docs operating manual | Current after docs consolidation; keep warm | `README.md`, this roadmap, architecture, agent onboarding, docs integrity | Drift reappears through new duplicate roadmap files | Maintain this file as the only roadmap | Docs |
| T1 Phase 5 evidence trust | Active | Structured queue, closeout bundles, `live_metrics.json`, `autoscore_bundle.json` | Markdown-only promotion decisions | Keep support tracks validating tooling and topology artifacts | Parallel support |
| T2 All-5 operational surface | Promoted child is `phase5_all5_current_surface_runner_realtime_missed_tick_burst_requal` on computed surface `8faedf4a9701d5e0`; final `7200s` rung completed clean with all-five fills, no cleanup, clean first/pre/post venue audits, no kill/restart/reconcile mismatch, and clean restore to shadow | Preserve all five enabled, FV-eligible, role-qualified, clean direct audits, final-rung venue fill evidence, no residual/live-order cleanup dependence, and no manual rescue on the exact promoted surface | Treating the promoted all-five surface as permission to mix new strategy/math or topology changes into the next gate | Reopen `phase5_reopened_multi_venue_long_soak` only on this exact promoted surface | Serialized live mainline |
| T3 Clean-surface economics | Resolved for the promoted surface: final `7200s` rung ended with `final_q_global_tao=0.0`, `final_pnl_total=0.0`, all-five fills (`466` / `5.0134 ETH`), no cleanup cost, and autoscore promotion pass | Carry exact closeout, metrics, cashflow, state-sync, and direct venue audit evidence forward to the next gate | Promoting a changed surface or post-hoc tuning rather than the clean proven surface | No strategy/math change before the reopened long-soak gate; hold or roll back on any new exact blocker | Serialized live mainline |
| T4 Unattended MM readiness | Not complete | Long unattended soaks, automatic rollback, no manual rescue dependence, and sustained positive attributable PnL | Declaring production readiness from WP100, a single clean rung, or operational cleanliness without positive economics | Promote only after repeated clean long-run evidence with positive net economics | Serialized live mainline |
| T5 Research and future strategy | Shadow/research unless promoted | Shadow/local evidence, tests, and an explicit Phase 5 promotion path | RL or sizing research bypasses live evidence ladder | Keep research out of live until gated | Research/support |

## 5. Execution Targets

### T0 - Docs Operating Manual

Definition of done:

- `ROADMAP.md` is the only roadmap.
- `README.md`, `docs/ARCHITECTURE.md`, `docs/AGENT_START_HERE.md`, and
  `docs/WHITEPAPER.md` agree on the source-of-truth order.
- Historical/generated docs are scoped at the top.
- `python3 tools/check_docs_integrity.py` passes.

Required evidence:

- Clean docs integrity output.
- No unscoped claim that WP100 means production readiness.
- No new `ROADMAP_*` document acting as a competing plan.

Promotion condition:

- A future agent can start from `README.md`, read this roadmap, and know the
  current target, gate, lane, and forbidden files without asking for context.

Hold/rollback condition:

- Any roadmap-like document starts carrying independent priority decisions.
- Whitepaper appendix changes without explicit canonical-hash intent.

Do not change:

- Phase 5 runtime state.
- Live services or live config.
- Hash-locked whitepaper appendix.

### T1 - Phase 5 Evidence Trust

Definition of done:

- Every live-affecting decision cites structured artifacts, not prose alone.
- Promotion, hold, and rollback decisions are traceable through
  `phase5/queue.yaml`, closeout bundles, metrics, and autoscore.
- Support-track tooling can validate current topology without mutating the live
  baseline.

Required evidence:

- `phase5/queue.yaml` entries include mechanism gates, promotion gates, rollback
  criteria, and history.
- Guarded live rungs produce closeout, `live_metrics.json`, and
  `autoscore_bundle.json`.
- Support tracks such as `phase5_tooling_forensics_hardening` and
  `phase5_topology_readiness_audit` remain validate-only unless explicitly
  promoted.

Promotion condition:

- Promotion decisions are reproducible from structured artifacts and exact
  surface lineage.

Hold/rollback condition:

- A decision depends on markdown scraping.
- Autoscore rules are missing for promotion-critical claims.
- A support lane mutates the live baseline.

Do not change:

- `phase5/queue.yaml` or `phase5/orchestration.yaml` during docs work.
- Any `/etc/paraphina`, `/opt/paraphina`, `/var/lib/paraphina`, or promotion-run
  state from documentation tasks.

### T2 - All-5 Operational Surface

Definition of done:

- All five venues remain enabled and FV-eligible when the current exact surface
  is under qualification.
- Each venue has an evidence-backed role.
- The surface passes short, medium, and long guarded rungs without kill,
  restart, stale-market, reconcile, health/readiness, transport, or dirty-state
  regression.
- Direct venue audit is clean after every rung.

Required evidence:

- Current topology from `phase5/status.md`.
- Exact surface lineage from `phase5/queue.yaml`.
- Guard summaries and closeout bundles for each rung.
- Execution scorecard showing all five venues remain eligible for
  strategy-driven order placement.

Promotion condition:

- The exact all-5 surface passes the configured rung ladder and remains
  operationally clean with all venue roles intact.

Hold/rollback condition:

- Any venue becomes excluded, FV-disabled, stale-dominant, dirty after restore,
  or dependent on manual cleanup outside the normal guard policy.
- Any branch changes topology, quote size, FV eligibility, edge floors, soft-cap
  thresholds, cleanup gates, market-data paths, or unrelated venue behavior
  without direct evidence.

Do not change:

- Multiple live-affecting hypotheses in one tranche.
- Venue roles or topology as a side effect of economics work.

### T3 - Clean-Surface Economics

Definition of done:

- Economics are judged only after platform and surface evidence are clean.
- Attribution separates maker MM fills, reduce-only unwind, cleanup flow, fees,
  markout, churn, and venue-local effects.
- A candidate changes exactly one economic hypothesis at a time.

Latest current-surface economics tranche:

- `phase5_all5_current_surface_runner_realtime_missed_tick_burst_requal`
  promoted after the `7200s` final rung on surface `8faedf4a9701d5e0`.
- The final rung was clean and economics-compatible: `final_q_global_tao=0.0`,
  `final_pnl_total=0.0`, no cleanup was required, cleanup cost was `0.0`, and
  all five venues filled (`466` fills / `5.0134 ETH`).
- Autoscore promotion passed because operational cleanliness and economic gates
  were both satisfied on the exact all-five surface.

Required evidence:

- Exact promoted baseline: `phase5_all5_current_surface_runner_realtime_missed_tick_burst_requal`
  on surface `8faedf4a9701d5e0`.
- Final rung economics attribution with realised net metrics and cleanup flow.
- Evidence that all five venues remain eligible and produce fills on the final
  rung.
- Preserved exact PnL, volume, account, and cashflow evidence.

Promotion condition:

- No operational regression.
- No pre-restore cleanup dependence or dirty direct venue audit.
- Final inventory remains inside the configured threshold and final PnL is
  non-negative.
- The all-five surface remains eligible and strategically credible.

Hold/rollback condition:

- The next gate changes strategy/math, topology, FV, quote sizes, edge floors,
  soft caps, cleanup gates, or market-data controls instead of preserving the
  promoted surface.
- Any venue becomes dirty before restore cleanup or requires manual rescue.
- Unexplained account equity drift exceeds the current tranche threshold.

Do not change:

- Promoted live behavior before the reopened long-soak gate.
- Strategy/math or shared economics controls unless a future tranche explicitly
  scopes one changed axis after this promoted baseline is preserved.

### T4 - Unattended MM Readiness

Definition of done:

- Long unattended runs complete without manual intervention.
- Rollback and restore are automatic, clean, and reproducible.
- Guard closeout, direct venue audits, autoscore, and metrics agree.
- Sustained positive net PnL is demonstrated on exact promoted surfaces after
  fees, cleanup flow, markout, and residual unwind effects are attributed.
- Operators can reproduce the readiness decision from committed docs and
  structured artifacts.

Required evidence:

- Multiple long soaks on exact promoted surfaces.
- No kill events, restart loops, reconcile mismatches, dirty venue state, stale
  restart dominance, or unexpected live-loop exits.
- Repeated positive `final_pnl_total` or equivalent net-PnL evidence on clean
  long-run windows; individual mechanism tranches may promote while neutral or
  small-negative only if they are explicitly reducing a blocker on the path to
  sustained positive PnL.
- Venue-level economics attribution showing positive PnL is not caused by a
  one-off mark, hidden taker cleanup, stale telemetry, residual unwind artifact,
  or one venue subsidizing structural losses elsewhere.
- Clean post-run direct venue audits.
- Runbook and roadmap agree on the promoted topology, trade modes, and guard
  expectations.

Promotion condition:

- The system can be left running under the approved guard policy with no ad hoc
  rescue dependence, no evidence gap in rollback or closeout, and sustained
  positive attributable PnL.

Hold/rollback condition:

- Any manual replay is needed to complete normal closeout.
- Any long run depends on operator judgment not captured in artifacts.
- Any production-readiness claim relies on WP100, whitepaper parity, or one
  clean final rung instead of repeated Phase 5 evidence.
- Long-run economics are negative, flat after fees, or positive only because of
  unattributed mark, cleanup, stale-data, or residual artifacts.

Do not change:

- Guard policy, risk caps, or canary scale as part of declaring readiness.

### T5 - Research And Future Strategy

Definition of done:

- Research stays shadow/local until it has a clear promotion ladder.
- RL, sizing, FV weighting, queue competitiveness, and markout optimization are
  separated from live operational cleanliness.
- Research artifacts explain what would have to become true before live
  promotion.

Required evidence:

- Deterministic tests, replay evidence, or shadow A/B results.
- Explicit "not live" status until a Phase 5 tranche is prepared.
- A single changed axis and rollback rule before any live candidate exists.

Promotion condition:

- The idea has a code-backed mechanism, focused tests, shadow evidence, and a
  Phase 5 candidate with mechanism/promotion/rollback gates.

Hold/rollback condition:

- The idea requires mixing topology, risk, connector, and economics changes in
  one live branch.
- The idea cannot be attributed cleanly with existing telemetry.

Do not change:

- Live surface behavior from research-only work.

#### Phase 5.1 / V2 Target Specification Gate

| Field | Value |
|---|---|
| target | Canonicalize the V2 target specification and continue Phase 5.1 as a non-live evidence program for fill-aware, hedge-aware, arbitrage-informed quoting. |
| current status | `specified_holding_nonlive`; `docs/V2_SPECIFICATION.md` is the active target spec. Phase 5.1t now provides a HOLD-only source-link sidecar builder in front of Phase 5.1s. The first 5.1t run over existing local Lighter snapshots processed `1522` rows and emitted `363` redacted source-link rows; Phase 5.1s staged those rows, Phase 5.1r applied `909` source-link joins and emitted `296` native-role source records, but downstream Phase 5.1q/5.1n/5.1h/5.1i still recovered `0 / 287` missing native-role targets and `0 / 3132` Lighter native-limit targets. Current matrix blockers remain Lighter native-limit pressure, maker/taker completeness, sparse buckets, and observed-only selection bias. No V2 live/canary/model-training/EV-admission promotion is authorized. |
| next action | Capture forward all-five venue-native maker/taker source rows and Lighter event-time active-order/sendTx/REST-or-weighted-request pressure with canonical group/order-key linkage, or with Phase 5.1t-generated redacted source-link sidecars that can be staged through Phase 5.1s. Then rerun Phase 5.1s -> 5.1r -> 5.1q -> 5.1n -> 5.1h -> 5.1i, and only after blocker reduction address sparse-bucket/selection-bias holds before any calibrated EV shadow review. |
| required evidence | Repo-owned V2 spec, Phase 5.1p recovered feature matrix, Phase 5.1s local source-staging artifacts, Phase 5.1r source-acquisition artifacts, Phase 5.1q forward native-evidence artifacts, filled-horizon source-key recovery evidence pack, Lighter event-time active-order alignment evidence, exact canonical venue-native maker/taker evidence, schema v2 validation, deterministic replay outputs, and docs integrity output. |
| promotion condition | All current blockers are resolved enough for calibrated EV shadow/model-training review, no-live guards remain intact, accepted evidence separates observed facts from counterfactuals, and economic claims remain balance-authoritative only. |
| hold/rollback condition | Any live/canary/capital/risk authorization from Phase 5.1 evidence, any return to `true_edge` or `Q_raw` as canonical admission/sizing, missing telemetry, unproven economic claims, spec drift, or docs integrity failure. |
| lane | Phase 5.1 non-live V2 evidence / quant, systems, data, execution, and risk. |
| source files | `docs/V2_SPECIFICATION.md`, `docs/WHITEPAPER.md`, `docs/PHASE5_1_TO_24_7_ORCHESTRATION_RUNBOOK.md`, `docs/PHASE5_1_BOARD_DECISION.md`, `docs/PHASE5_1_EVIDENCE_LOG.md`, `docs/PHASE5_1P_LIGHTER_NATIVE_ROLE_EVIDENCE.md`, `docs/PHASE5_1Q_FORWARD_NATIVE_EVIDENCE.md`, `docs/PHASE5_1R_FORWARD_NATIVE_SOURCE_ACQUISITION.md`, `docs/PHASE5_1S_LOCAL_NATIVE_SOURCE_ACQUISITION.md`, `schemas/telemetry_schema_v2.json`, `tools/phase51j_observed_horizon_recovery.py`, `tools/phase51k_filled_horizon_timebase_recovery.py`, `tools/phase51l_filled_horizon_source_key_recovery.py`, `tools/phase51n_lighter_native_limit_time_alignment.py`, `tools/phase51n_maker_taker_attribution_recovery.py`, `tools/phase51o_native_role_source_inventory.py`, `tools/phase51p_lighter_native_role_canonical_join.py`, `tools/phase51q_forward_native_evidence_capture.py`, `tools/phase51r_forward_native_source_acquisition.py`, `tools/phase51s_local_native_source_acquisition.py`, `tools/phase51h_observed_pfill_feature_audit.py`, `tools/phase51i_pfill_feature_matrix_admissibility.py`, `tools/phase51b_lighter_account_limits.py`. |

#### Phase 5.1b - Lighter Account/Native-Limit Evidence Gate

| Field | Value |
|---|---|
| target | Capture Lighter account/profile, native limits, active-order headroom, fee/market metadata, and maker/taker attribution samples as schema v2 non-live evidence. |
| current status | `accepted_for_calibration_label_ingestion`; Phase 5.1n aligned historical Lighter active-order snapshot logs to many label event times, but the recovered 5.1h/5.1i matrix correctly keeps native-limit pressure partial because sendTx/REST pressure remains unobserved and some snapshots are stale. |
| next action | Run `tools/phase51b_lighter_account_limits.py` against authenticated read-only Lighter sources or captured endpoint JSON, then validate `telemetry.jsonl` with `tools/check_telemetry_contract.py`. |
| required evidence | `V2_LIGHTER_ACCOUNT_PROFILE`, `V2_LIGHTER_ACCOUNT_LIMITS`, `V2_LIGHTER_ACTIVE_ORDERS`, optional `V2_LIGHTER_TRADE_ATTRIBUTION_SAMPLE`, sanitized source snapshots, manifest, artifact index, and schema validation output. |
| promotion condition | `phase51b_capture_complete=true`, schema v2 passes, account/native-limit fields are present enough to begin P-fill, markout, queue/churn, and maker/taker calibration-label ingestion; official-doc caps may derive current active-order capacity context only when paired with observed active-order counts and must not be treated as label-event-time native-limit pressure or event-time sendTx/REST remaining pressure. |
| hold/rollback condition | Missing account limits, missing active orders, unsafe spec flags, missing no-live guards, any sendTx/live-order path, unredacted secrets, or any attempt to use this evidence as live/canary/economic authority. |
| lane | Phase 5.1 non-live evidence / execution microstructure. |
| source files | `configs/phase51b_lighter_account_native_limits.json`, `tools/phase51b_lighter_account_limits.py`, `schemas/telemetry_schema_v2.json`, `docs/TELEMETRY_SCHEMA_V2.md`, `docs/PHASE5_1_BOARD_DECISION.md`, `docs/PHASE5_1_LIGHTER_VENUE_READINESS.md`, `docs/PHASE5_1_TO_24_7_ORCHESTRATION_RUNBOOK.md`. |

## 6. Current Queue

| Order | Item | Status | Decision rule |
|---|---|---|---|
| 1 | Preserve `phase5_all5_current_surface_aster_reduce_only_unwind_fee_guard_requal` outcome | Held current serialized mainline | Do not rerun unchanged; use the preserved closeout/autoscore evidence showing clean/mechanism-positive but Extended zero-fill promotion failure. |
| 2 | Preserve `phase5_all5_current_surface_aster_residual_markout_guard_requal` outcome | Held current serialized mainline after `1200s` gate | Do not run the `7200s` rung unchanged; use the preserved closeout/autoscore/guard evidence showing clean operations but Aster `0.01 ETH` pre-restore residual, `held_no_fresh_account=84`, zero allowed markout-guard orders, and estimated cleanup cost `0.2314305`. |
| 3 | Preserve `phase5_all5_current_surface_aster_account_freshness_residual_markout_requal` outcome | Held on surface `f972af01ad5466fb` | Do not rerun unchanged; use the preserved 20m evidence showing `refresh_attempts=0`, `account_channel_unavailable=99`, Aster `0.01 ETH` residual, negative final PnL, and repeated ~$0.231 cleanup cost. |
| 4 | Preserve `phase5_all5_current_surface_aster_account_refresh_channel_requal` outcome | Held on surface `d450cfc6e055f023` | Do not rerun unchanged; use the preserved 20m evidence showing `refresh_attempts=6`, `refresh_outcomes.reconcile_failed=6`, `allowed_orders=0`, `final_q_global_tao=-0.01`, and ~$0.231 cleanup cost. |
| 5 | Preserve `phase5_all5_current_surface_aster_account_refresh_seq_requal` outcome | Held on surface `f120ca94e4fec147` | Do not rerun unchanged; use the recovered 7200s evidence showing clean/mechanism-positive Aster refresh success but final residual `0.01 ETH` and `final_pnl_total=-0.001616`. |
| 6 | Preserve `phase5_all5_current_surface_aster_stale_residual_force_flat_requal` outcome | Held on surface `71f3f410f23845b2` | Use the preserved 20m evidence: clean gate passed, final flat, `final_pnl_total=0`, fills=16 across Aster/Lighter/Paradex, but no `allowed_force_flat_age` opportunity occurred. |
| 7 | Preserve `phase5_all5_current_surface_aster_stale_residual_force_flat_long_opportunity_requal` outcome | Held on surface `ea129c56d0618fd2` | Do not rerun unchanged; preserved 5m evidence shows the child held on Extended `stale_restart` after the long-opportunity overlay dropped the inherited Extended state-stale/rebootstrap/channel-cap tail and started with `PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE=1500`. |
| 8 | Preserve `phase5_all5_current_surface_aster_stale_residual_force_flat_long_opportunity_overlay_repair_requal` outcome | Held on surface `3201bceccae332b2` | Use the preserved 7200s evidence: Aster force-flat mechanism observed (`allowed_force_flat_age=7`, fee max `0.009243`, refresh success `355`), but guard intervened after Extended reached the stale boundary, cleanup flattened Lighter `+0.0039 ETH` and Paradex `-0.0088 ETH`, `final_q_global_tao=-0.0049`, and `final_pnl_total=-0.001301`. |
| 9 | Preserve `phase5_all5_current_surface_extended_pre_kill_rebootstrap_lead_time_tighten_after_aster_force_flat_requal` outcome | Held on surface `9ed48288a2b00ba1` | Use the preserved 300s clean rung and held 1200s evidence: 1200/1800 fallback fired earlier, but repeated `degraded_stream_rebootstrap_gap` events reached stale_watchdog_count_window `5`, reconnect_policy slept `4000ms`, Extended crossed the `3000ms` stale gate, and guard cleanup restored Lighter/Paradex residue. |
| 10 | Preserve `phase5_all5_current_surface_extended_stale_churn_healthy_reset_after_pre_kill_requal` outcome | Held on surface `61fe0679b377ef5d` | Use the preserved 300s clean rung and held 1200s evidence: healthy reset improved survival from 86s to 673s, but `degraded_stream_rebootstrap_gap` still escalated at stale_watchdog_count_window `4`, slept `4000ms`, hit Extended stale-market hygiene at `consecutive_stale_ticks=12`, and guard cleanup restored Lighter open-order residue. |
| 11 | Preserve `phase5_all5_current_surface_extended_stale_churn_budget5_after_reset_requal` outcome | Held on surface `461f28e1b619f3c9` | Use the preserved 300s/1200s continuations and held final-rung evidence: all five venues filled over 4976s, Aster account/fee evidence was healthy, final q/PnL were 0, but Extended count_window 6/7 escalated to 2000/4000ms sleeps and hit stale-market kill. |
| 12 | Preserve `phase5_all5_current_surface_extended_stale_churn_budget7_after_budget5_requal` outcome | Held; last-run surface `aa73a38a7be6875d` | Use the preserved all-five 7200s evidence: operationally clean, 55 fills / 0.539 ETH, but held on Aster cleanup fee estimate max `0.064904 USD` and final PnL `-0.000579 USD`. |
| 13 | Preserve `phase5_all5_current_surface_aster_reduce_only_inventory_brake_fee_guard_requal` outcome | Held on surface `3bf0f8a9c2548c44` | Use the preserved 300s/1200s/7200s evidence: all five venues filled over the 7200s rung, support/state-sync passed, and venue cleanup ended clean, but promotion held on Aster cleanup fee estimate max `0.140716 USD`, final inventory `0.01 ETH`, and final PnL `-0.010669 USD`. Do not rerun unchanged. |
| 14 | Preserve `phase5_all5_current_surface_aster_same_side_live_exposure_sum_requal` outcome | Rolled back on surface `1465ff782d3747be`; recorded at `2026-04-26T20:59:54Z` | Use the preserved interrupted 7200s evidence: 300s/1200s passed and the 7200s segment produced 96 fills / 0.99 ETH with recovered `final_pnl_total=0.083764` and Aster cleanup fee estimate max `0.009495 USD`, but the guard window did not complete, Extended/Hyperliquid were fill-zero in the interrupted segment, `final_q_global_tao=-0.02`, the recovered direct venue audit was dirty, manual cleanup/restore was required, and autoscore failed clean/mechanism/promotion gates. Do not rerun unchanged as promotion evidence. |
| 15 | Preserve `phase5_all5_current_surface_aster_same_side_live_exposure_restore_hygiene_requal` outcome | Held; preserved run surface `1465ff782d3747be`, refreshed card surface `d1864116690ce318`; state-sync pass and child activation recorded at `2026-04-26T23:32:16Z` | Use the preserved 300s clean rung and held 1200s all-five-fill evidence: `69` fills / `0.676 ETH` across all five venues, but first pre-restore audit was dirty from Extended `-0.02 ETH`, cleanup cost `0.47151 USD`, final inventory was `-0.02 ETH`, final PnL was `-0.015203`, and autoscore failed clean/mechanism/promotion. Do not activate the parked Extended successor or a 7200s rung until the exact-surface residual-convergence child clears this blocker. |
| 16 | Preserve `phase5_all5_current_surface_extended_terminal_residual_convergence_requal` outcome | Held on surface `4f7d1e23397ee100`; 600s support and preflight passed, then 300s rung held at `2026-04-26T23:54:40Z` | Use the preserved evidence: Extended was clean in first pre-restore audit, but Aster `+0.01 ETH` and one Lighter open order made the first pre-restore audit dirty, cleanup was required, Aster cleanup cost was `0.2367805 USD`, final inventory was `0.01 ETH`, and final PnL was `-0.052578`. Do not rerun unchanged. |
| 17 | Preserve `phase5_all5_current_surface_aster_terminal_full_target_convergence_requal` outcome | Held with preserved run surface `a9385c5735703e1d` and refreshed card surface `220066a1c031a907`; 600s support, preflight, state-sync, lane spawning, topology/tooling support lanes, and 300s rung completed at `2026-04-27T01:02:05Z` | Use the preserved evidence: all five venues filled in the 300s canary (`44` fills / `0.47 ETH`), but first pre-restore audit was dirty from Aster `-0.01 ETH`, cleanup was required, Aster cleanup cost was `0.2402895 USD`, final inventory was `-0.01 ETH`, final PnL was `-0.043561`, and telemetry showed benign Aster residuals still suppressed behind the inherited `900000ms` force-flat age. Do not rerun unchanged. |
| 18 | Preserve `phase5_all5_current_surface_aster_short_force_flat_convergence_requal` outcome | Held with preserved run surface `9eb28c7eb88fcfa1` and refreshed card surface `b44a9a45d7a706c8`; 300s rung completed at `2026-04-27T01:28:31.728000Z` | Use the preserved evidence: all five venues filled, `final_pnl_total=0.051386`, and Aster was inside tolerance, but first pre-restore audit was dirty from Extended `-0.01 ETH`, cleanup was required, cleanup cost was `0.28550690000000034`, `final_q_global_tao=-0.018`, and Extended reduce-only size-exceeds-position rejects appeared. Do not rerun unchanged. |
| 19 | Preserve `phase5_all5_current_surface_extended_one_lot_state_fallback_step_requal` outcome | Held on surface `f4943e494189b28a`; support smoke passed and the 300s canary completed at `2026-04-27T02:34:43.804000Z` | Use the preserved evidence: Extended first-audit state was clean with zero open orders and no Extended size-exceeds-position reject, but first pre-restore audit was dirty from Hyperliquid one open order, Lighter two open orders, and Aster `+0.01 ETH`; cleanup was required, estimated Aster cleanup cost was `0.2389965 USD`, `final_q_global_tao=0.01`, and `final_pnl_total=-0.048065`. Do not rerun unchanged. |
| 20 | Preserve `phase5_all5_current_surface_aster_markout_min_adverse_025_requal` outcome | Held; 300s rung completed at `2026-04-27T03:06:33.518000Z`, last-run surface `4f71b7321d507b9d`, refreshed card surface `cce6a6ee0214c951` after the terminal-exit runtime build | Use the preserved evidence: guard completed and post-rollback audit was clean, but first pre-restore audit was dirty from Lighter 2 open orders, Aster `+0.01 ETH` plus one open order, and Paradex one open order; cleanup was required with estimated cost `0.2391465 USD`, `final_q_global_tao=0.01`, `final_pnl_total=0.00264`, fills were Aster-only `5` / `0.05 ETH`, and Aster allowed_orders was `0`. Do not rerun unchanged. |
| 21 | Preserve `phase5_all5_current_surface_terminal_exit_live_order_drain_requal` outcome | Held on computed surface `b4af7363c8d64091`; 300s continued, 1200s held at `2026-04-27T09:47:18Z` | Use the preserved evidence: all five venues filled, final inventory/PnL were exactly `0.0`, health stayed clean, and current venue audit is flat/zero-open, but first pre-restore audit was dirty from Extended `open_order_count=1`. Guard wrote `/tmp/paraphina_phase5_terminal_exit_quiesce.signal` at `2026-04-27T09:45:39Z`, but runtime telemetry never recorded `terminal_exit_quiesce`; systemd `PrivateTmp=yes` makes `/tmp` the wrong signal path. Do not rerun unchanged. |
| 22 | Preserve `phase5_all5_current_surface_terminal_exit_service_visible_signal_requal` outcome | Held on computed surface `0be1adb1a04a1a42`; 7200s rung stopped early at `2026-04-27T11:43:39.534Z` | Use the preserved evidence: 93 fills, final inventory `0.0`, but Extended degraded rebootstrap churn escalated to `8000ms`, crossed the `3000ms` state stale boundary, triggered `StaleMarket`, and forced guard intervention before terminal-signal visibility could be measured. Do not rerun unchanged. |
| 23 | Preserve `phase5_all5_current_surface_extended_degraded_rebootstrap_sleep_cap_requal` outcome | Held on computed surface `dbe99923f7ab3e17`; 300s rung completed at `2026-04-27T14:07:19.514000Z` | Use the preserved evidence: Extended degraded rebootstrap sleep cap was measured without stale-market/restart/reconcile regression, terminal quiesce telemetry appeared, all five venues filled over the 300s rung, but first pre-restore audit was dirty only on Aster `position_base=-0.009` with `open_order_count=0`; cleanup bought `0.009 ETH` with estimated cost `0.20812545`, `final_q_global_tao=-0.009`, and `final_pnl_total=-0.04093`. Do not rerun unchanged. |
| 24 | Preserve `phase5_all5_current_surface_aster_terminal_account_refresh_requal` outcome | Held; 5m clean, 1200s held with last-run surface `7a1c69058e32c2a2` | Use the preserved evidence: Aster terminal refresh worked and Aster was flat, all five venues filled, final PnL was positive at `0.115644`, but first pre-restore audit was dirty on Extended `+0.02` and Paradex `+0.03`, cleanup was required, cleanup cost was `1.13705 USD`, and autoscore held clean/mechanism. Do not rerun unchanged. |
| 25 | Preserve `phase5_all5_current_surface_terminal_ext_pdx_account_cancel_drain_requal` outcome | Held on computed surface `b62301ceb7e07ca7`; 300s rung completed at `2026-04-27T17:04:37Z` | Use the preserved evidence: Extended and Paradex terminal cancel-drain blockers did not reappear and terminal quiesce worked, but first pre-restore audit was dirty only on Aster `-0.008 ETH` with zero open orders; cleanup bought `0.008 ETH` with estimated cost `0.1823996 USD`, `final_q_global_tao=-0.008`, and `final_pnl_total=-0.020043`. Do not rerun unchanged. |
| 26 | Preserve `phase5_all5_current_surface_aster_terminal_sub_lot_reduce_requal` outcome | Held after 7200s on surface `8497a73856f1ceca` | Use the preserved evidence: all five venues filled, `final_q_global_tao=0.0`, `final_pnl_total=0.0`, health stayed clean, and Aster position was flat, but first pre-restore audit found one Aster open order and required cleanup. Do not rerun unchanged. |
| 27 | Preserve `phase5_all5_current_surface_aster_terminal_quiesce_explicit_cancel_retry_requal` outcome | Held on surface `a3c0cf1dce0e4cc7` after 7200s | Use the preserved evidence: the run was operationally clean with first pre-restore audit clean, no cleanup required, zero cleanup cost, flat final inventory, no kill/restart/reconcile mismatch, and post-rollback audit clean. Promotion held only because `final_pnl_total=-0.002061` and Extended fills were zero. |
| 28 | Preserve `phase5_all5_current_surface_extended_queue_persistence_aster_bid_edge_requal` outcome | Held after 7200s on surface `b59c5beb892cefdf` | Use the preserved evidence: all five venues filled, Extended conversion was recovered, Aster remained active, state-sync passed, and health/reconcile stayed clean, but first pre-restore audit was dirty only on Hyperliquid `position_base=-0.0044` with zero open orders; cleanup bought `0.0044 ETH` with estimated cost `0.1005400000000016 USD`, `final_q_global_tao=-0.0044`, and `final_pnl_total=-0.010563`. Do not rerun unchanged. |
| 29 | Preserve `phase5_all5_current_surface_hyperliquid_terminal_sub_lot_residual_requal` outcome | Held on surface `0cf29e9406f23f58` | Use the preserved evidence: all-five fills and Hyperliquid terminal residual convergence improved the blocker, but promotion stayed gated by the later terminal account-refresh/router and runner missed-tick children. Do not rerun unchanged. |
| 30 | Preserve `phase5_all5_current_surface_terminal_account_refresh_router_requal` outcome | Held on surface `ff08070af9bb4e80`; 7200s rung stopped early at `2026-04-28T13:39:01.758Z` | Use the preserved evidence: all-five fill conversion was proven (`339` fills / `3.512 ETH`), but a host/runtime delay replayed missed realtime ticks into a stale-market hygiene burst and triggered an Extended `StaleMarket` kill before recovery could be observed. Do not rerun unchanged. |
| 31 | Promote `phase5_all5_current_surface_runner_realtime_missed_tick_burst_requal` | Promoted on surface `8faedf4a9701d5e0` at `2026-04-28T17:43:47Z` | Final `7200s` rung completed clean with all-five fills (`466` / `5.0134 ETH`), `final_q_global_tao=0.0`, `final_pnl_total=0.0`, no guard intervention, no cleanup, clean first/pre/post venue audits, state-sync pass, and autoscore clean/mechanism/promotion pass. |
| 32 | Preserve support-track validation | Always parallel | `phase5_tooling_forensics_hardening` and `phase5_topology_readiness_audit` may validate but must not mutate the promoted live baseline. |
| 33 | Reopen `reopened_multi_venue_long_soak` gate | Eligible after promotion | Run only on exact promoted surface `8faedf4a9701d5e0`; do not combine with new economics tuning, Extended conversion work, residual-control changes, or strategy/math changes. |
| 34 | Consider unattended readiness | Later | Only after the promoted all-five surface survives longer unattended evidence without manual rescue, dirty venue state, or negative attributable economics. |

## 7. Execution Lanes

### Serialized live-affecting mainline

Use for any change that can alter live orders, live topology, venue roles, FV
eligibility, risk caps, cleanup behavior, or economics on the promoted surface.

Rules:

- exactly one live-affecting mainline owns the live lane;
- one hypothesis per tranche;
- frozen baseline and exact surface id required;
- short gate before long gate;
- promotion requires structured artifacts, not final-rung cleanliness alone.

### Parallel support tracks

Use for forensics, tooling, topology readiness, and shadow/local diagnostics.

Rules:

- validate-only unless explicitly promoted;
- no mutation of live config, queue, orchestration state, or promotion runs from
  docs work;
- stop or rebase if the serialized mainline changes the baseline they analyze.

### Docs and architecture maintenance

Use for roadmap, architecture, onboarding, whitepaper scope notes, and generated
artifact scoping.

Rules:

- do not edit runtime state;
- update this roadmap first when strategy changes;
- run docs integrity after roadmap or whitepaper changes.

### Research and shadow-only work

Use for RL, sizing, FV weighting, markout, venue microstructure, and future
strategy.

Rules:

- no live behavior implied;
- no promotion without tests, shadow evidence, exact changed axis, and Phase 5
  gates.

## 8. Non-Authoritative Roadmap Surfaces

These files must not compete with this roadmap:

- `docs/ROADMAP_B.md` - supporting venue rollout/gate note.
- `docs/ROADMAP_B_READINESS_REPORT.md` - generated historical connector report.
- `docs/WORLD_CLASS_RUNTIME_GOALS.md` - objective framework, not current state.
- `docs/PHASE5_AUTONOMOUS_TRANCHE_SYSTEM.md` - workflow spec, not current state.
- `docs/WP100_SIGNOFF.md`, `docs/WP_PARITY_MATRIX.md`, and
  `docs/WHITEPAPER_COMPLETION_*` - historical WP100 audit artifacts.

If one of these documents needs a strategic update, update `ROADMAP.md` first
and leave the supporting file scoped to its purpose.

## 9. Safe Docs-Only Patch Plan

Future docs-only consolidation may edit:

- `README.md`
- `ROADMAP.md`
- `docs/ARCHITECTURE.md`
- `docs/AGENT_START_HERE.md`
- `docs/WHITEPAPER.md` before the canonical-spec marker only, unless changing
  the hash-locked appendix intentionally
- scope banners in dated/generated docs when needed

Future docs-only consolidation must not touch:

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

When roadmap drift is found:

1. Read code and structured state first.
2. Classify each stale claim as current, historical, generated, planned,
   shadow-only, experimental, or contradicted.
3. Update the master truth path first: `README.md`, `ROADMAP.md`,
   `docs/ARCHITECTURE.md`, and `docs/AGENT_START_HERE.md`.
4. Add scope banners to historical/generated docs instead of rewriting their
   dated evidence.
5. Avoid editing Phase 5 runtime state or live services during docs work.
6. Run the docs integrity checker.
7. Summarize exactly which files changed and which contradictions remain
   intentionally historical.

---

## Historical Appendix: Implementation Milestones

The remainder of this file preserves earlier milestone detail. It remains useful
as implementation provenance. Do not add new strategy here. The roadmap
sections above are the strategic and execution authority when older milestone
language conflicts with current code or Phase 5 state.

## 0. Non-negotiables

### 0.1 Determinism and replayability
- All strategy decisions must be deterministic given:
  - initial state,
  - a time-ordered input event stream,
  - the config/environment,
  - an explicit RNG seed if any randomness exists.

### 0.2 Research contracts are production contracts
The batch backbone depends on stable, machine-readable outputs:
- `batch_runs/orchestrator.py` runs many configs and returns a tidy DataFrame
  via `results_to_dataframe(...)`.
- `batch_runs/metrics.py` parses end-of-run stdout summaries using
  `parse_daily_summary(stdout)` and aggregates runs via `aggregate_basic(...)`.

Do not break these contracts without versioning.

### 0.3 Safety defaults win
If there is ambiguity between “trade more” and “reduce risk”, defaults must pick
“reduce risk” (especially around liquidation distance, hard limits, and kill).

---

## 1. Current baseline (observed in repo)

### 1.1 Generic run orchestrator exists
`batch_runs/orchestrator.py` provides:
- `EngineRunConfig(cmd, env, label, run_id, workdir)`
- `run_engine(...)` running subprocesses with env overlays
- `run_many(...)` sequential runner with Timeout handling
- `results_to_dataframe(...)` producing tidy per-run rows of `label + metrics`

This is the primary research execution interface.

### 1.2 Shared stdout metrics parser exists
`batch_runs/metrics.py` provides:
- `parse_daily_summary(stdout)` extracting:
  - `pnl_realised`, `pnl_unrealised`, `pnl_total`
  - `kill_switch` boolean
- `aggregate_basic(df_runs, group_keys)` producing grouped summaries
  similar to exp02/exp03.

### 1.3 Exp03 is the canonical “experiment script” pattern
`batch_runs/exp03_stress_search.py` demonstrates the pattern:
- load a prior summary CSV (`runs/exp02_profile_grid/exp02_profile_grid_summary.csv`)
- derive “profile centres”
- build a grid of `EngineRunConfig` with env overrides (profile, q0, etc.)
- run via `run_many(...)`
- write:
  - per-run CSV
  - grouped summary CSV

---

## 2. Milestone plan (spec-alignment driven)

### Milestone A — Documentation + "contracts first"
**Goal:** make it impossible to confuse "planned spec" with "implemented behavior".

**Status: COMPLETE**

Deliverables:
- `docs/WHITEPAPER.md` is the single entry point:
  - "Implementation truth" section stays conservative and code-backed
  - "Canonical spec" is preserved verbatim
- Add/maintain a short "Drift Register" section listing:
  - spec features not implemented yet,
  - known mismatches in naming/semantics (e.g., Warning/Critical vs Warning/HardLimit).

**What shipped:**
- `docs/WHITEPAPER.md`: Two-part structure with implementation truth (Part I) and canonical spec (Part II)
- `docs/EVIDENCE_PACK.md`: Hard reference map from WHITEPAPER to code evidence
- `docs/AI_PLAYBOOK.md`: Development guidelines
- Docs Integrity Gate (Implemented refs + canonical spec hash lock) — COMPLETE

**Evidence:** See `docs/EVIDENCE_PACK.md` §1 (Core loop), §8 (Research harness)

**Acceptance criteria:**
- Invariant: `docs/WHITEPAPER.md` contains "Known drift" section listing spec vs implementation gaps
- Invariant: Every algorithmic claim in Part I has `Implemented:` annotation with file path
- Manual: A new contributor can run `exp03` and understand env knobs + stdout format

---

### Milestone B — Standardise knobs and experimentability
**Goal:** every strategy parameter that matters is:
- configurable (env/config),
- logged/observable,
- sweepable via orchestrator labels.

**Status: COMPLETE**

Deliverables:
- A canonical env var list in docs (and used consistently):
  - examples already referenced in `batch_runs/orchestrator.py` docstring:
    - `PARAPHINA_RISK_PROFILE`
    - `PARAPHINA_INIT_Q_TAO`
    - `PARAPHINA_HEDGE_BAND_BASE`
    - `PARAPHINA_HEDGE_MAX_STEP`
    - `PARAPHINA_MM_SIZE_ETA`
    - `PARAPHINA_VOL_REF`
    - `PARAPHINA_DAILY_LOSS_LIMIT`
- A single authoritative end-of-run stdout summary format that
  `parse_daily_summary` parses (and unit tests for the regex).

**What shipped:**
- `paraphina/src/config.rs`: `Config::from_env_or_profile()`, `Config::from_env_or_default()` with all env var overrides
- `batch_runs/metrics.py`: `parse_daily_summary()` regex parser for stdout
- `batch_runs/orchestrator.py`: `EngineRunConfig`, `run_many()`, `results_to_dataframe()`

**Evidence:** See `docs/EVIDENCE_PACK.md` §8 (Research harness + metrics)

**Acceptance criteria:**
- Passes `paraphina/tests/config_profile_tests.rs` (profile env override tests)
- Passes `paraphina/tests/metrics_tests.rs` (stdout parsing tests)
- Determinism: `paraphina/tests/replay_determinism_tests.rs::test_replay_determinism_single_tick` passes with fixed seed

---

### Milestone C — Risk regime correctness + kill switch semantics
**Goal:** risk states match the canonical model: Normal / Warning / Critical
with a kill switch that is explicit and unambiguous.

**Status: COMPLETE**

Deliverables:
- Define regime semantics and thresholds in one place.
- Ensure **Critical implies kill_switch** behaviorally:
  - cancel/stop new orders,
  - allow only bounded "best-effort" risk reduction if implemented.

**What shipped:**
- `paraphina/src/state.rs`: `RiskRegime` enum (Normal/Warning/HardLimit), `KillReason` enum
- `paraphina/src/engine.rs::update_risk_limits_and_regime()`: Latching kill switch logic (L488-L510)
- `paraphina/src/mm.rs::compute_mm_quotes()`: Returns no quotes when `kill_switch || risk_regime == HardLimit`
- `paraphina/src/hedge.rs::compute_hedge_orders()`: Returns empty when `kill_switch`
- `paraphina/src/exit.rs::compute_exit_intents()`: Returns empty when `kill_switch`

**Evidence:** See `docs/EVIDENCE_PACK.md` §7 (Risk regime + kill switch semantics)

**Acceptance criteria:**
- Passes `paraphina/tests/risk_regim_tests.rs::hardlimit_and_kill_switch_when_loss_limit_breached`
- Passes `paraphina/tests/risk_regim_tests.rs::hardlimit_and_kill_switch_when_delta_limit_breached`
- Passes `paraphina/tests/risk_regim_tests.rs::hardlimit_and_kill_switch_when_basis_limit_breached`
- Passes `paraphina/tests/risk_regim_tests.rs::hardlimit_and_kill_switch_when_liquidation_distance_breached`
- Passes `paraphina/tests/risk_regim_tests.rs::kill_switch_latches_once_true`
- Passes `paraphina/tests/risk_regim_tests.rs::hardlimit_from_delta_breach_triggers_kill_and_disables_mm`
- Passes `paraphina/tests/risk_regim_tests.rs::hedge_disabled_when_kill_switch_active`
- Passes `paraphina/tests/risk_regim_tests.rs::exit_disabled_when_kill_switch_active`
- Invariant: `state.kill_switch == true` implies `state.risk_regime == HardLimit`
- Invariant: `KillReason` preserved from first breach (latching)

---

### Milestone D — Fair value + volatility gating (robustness)
**Goal:** fair value and volatility estimates are stable, and strategy behavior
degrades safely when data quality drops.

**Status: COMPLETE**

Deliverables:
- FV estimator gating policy (stale books / outliers / min healthy venues).
- Vol floor behavior documented and tested.
- Telemetry includes:
  - fair value (or a "fv_available" flag),
  - effective volatility,
  - list/count of healthy venues used.

**What shipped:**
- `paraphina/src/engine.rs::update_fair_value_and_vol()`: Kalman filter with venue gating (L99-L223)
- `paraphina/src/engine.rs::collect_kf_observations()`: Staleness/outlier/min-healthy gating (L276-L348)
- `paraphina/src/engine.rs::update_vol_and_scalars()`: EWMA vol + sigma_eff floor (L354-L408)
- Telemetry fields: `fv_available`, `healthy_venues_used`, `healthy_venues_used_count`, `sigma_eff`

**Evidence:** See `docs/EVIDENCE_PACK.md` §5 (Fair value + volatility gating)

**Acceptance criteria:**
- Passes `paraphina/tests/fair_value_gating_tests.rs::stale_venue_data_excluded_from_fv_update`
- Passes `paraphina/tests/fair_value_gating_tests.rs::fv_degrades_gracefully_with_all_stale_data`
- Passes `paraphina/tests/fair_value_gating_tests.rs::outlier_venue_excluded_from_fv_update`
- Passes `paraphina/tests/fair_value_gating_tests.rs::min_healthy_threshold_enforced`
- Passes `paraphina/tests/fair_value_gating_tests.rs::telemetry_fields_populated_correctly`
- Passes `paraphina/tests/vol_floor_tests.rs::sigma_eff_never_below_sigma_min`
- Passes `paraphina/tests/vol_floor_tests.rs::sigma_eff_uses_floor_when_raw_vol_is_low`
- Passes `paraphina/tests/vol_floor_tests.rs::vol_scalars_use_sigma_eff`
- Invariant: `sigma_eff = max(fv_short_vol, sigma_min)`
- Invariant: `fv_available = false` when `healthy_venues_used_count < min_healthy_for_kf`

---

### Milestone E — Cross-venue exits (canonical Section 12)
**Goal:** implement cross-perp exit optimization that:
- uses net edge after fees/slippage/vol buffers,
- incorporates basis/funding differentials,
- penalises basis risk increase and fragmentation.

**Status: COMPLETE**

Deliverables:
- A discrete "exit allocator" module callable after fill batches.
- A consistent definition of:
  - fragmentation score,
  - basis exposure change approximation,
  - effective edge threshold(s).

**What shipped:**
- `paraphina/src/exit.rs::compute_exit_intents()`: Full exit allocator (L247-L600)
- Edge calculation: `base_profit - fees - slippage_buffer - vol_buffer + basis_adj + funding_adj`
- Fragmentation penalty: `fragmentation_penalty_per_tao` for opening new legs
- Fragmentation bonus: `fragmentation_reduction_bonus` for closing positions
- Basis-risk penalty: `basis_risk_penalty_weight * Δ|B_t|`
- Per-venue constraints: `lot_size_tao`, `size_step_tao`, `min_notional_usd`

**Evidence:** See `docs/EVIDENCE_PACK.md` §3 (Exit engine)

**Acceptance criteria:**
- Passes `paraphina/tests/exit_engine_tests.rs::exit_respects_lot_size_and_min_notional`
- Passes `paraphina/tests/exit_engine_tests.rs::exit_respects_min_notional`
- Passes `paraphina/tests/exit_engine_tests.rs::exit_profit_only_blocks_unprofitable_exits`
- Passes `paraphina/tests/exit_engine_tests.rs::exit_splits_across_best_venues_when_capped_per_venue`
- Passes `paraphina/tests/exit_engine_tests.rs::exit_skips_disabled_or_toxic_or_stale`
- Passes `paraphina/tests/exit_engine_tests.rs::exit_prefers_less_fragmentation_when_edges_similar`
- Passes `paraphina/tests/exit_engine_tests.rs::exit_prefers_less_basis_risk_when_edges_similar`
- Passes `paraphina/tests/exit_engine_tests.rs::exit_deterministic_ordering_with_identical_edges`
- Invariant: Exit only when `base_profit_per_tao > edge_min_usd`
- Invariant: Exit size rounded to `size_step_tao` and >= `lot_size_tao`
- Determinism: Same state produces identical exit intents

---

### Milestone F — Global hedge allocator (canonical Section 13)
<!-- STATUS: MILESTONE_F = COMPLETE -->
**Goal:** hedging is a *global* optimization over allowed venues:
- LQ controller decides global step size with deadband,
- allocation chooses cheapest/least-risk venues first,
- constraints include funding/basis/margin/liquidation/fragmentation.

**Status: COMPLETE**

Deliverables:
- Hedge allocator as a first-class component:
  - input: desired global ΔH, venue snapshots
  - output: per-venue IOC intents with guard prices
- Costs modeled as additive components:
  - immediate execution + fee + slippage buffer
  - funding carry preference
  - basis exposure effect
  - liquidation distance penalty
  - fragmentation penalty

**What shipped:**
- `paraphina/src/hedge.rs::compute_hedge_plan()`: Global hedge with deadband (L118-L196)
- `paraphina/src/hedge.rs::build_candidates()`: Per-venue cost model (L199-L377)
- `paraphina/src/hedge.rs::greedy_allocate()`: Greedy allocation by cost (L380-L427)
- Cost components: `exec_cost + liq_penalty + frag_penalty - funding_benefit - basis_edge`

**Evidence:** See `docs/EVIDENCE_PACK.md` §4 (Hedge allocator)

**Acceptance criteria:**
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_deadband_no_orders`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_outside_deadband_generates_orders`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_respects_global_max_step`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_respects_per_venue_caps`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_avoids_near_liquidation`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_skips_critical_liquidation_venues`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_prefers_funding_or_basis_when_exec_equal`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_prefers_basis_when_funding_equal`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_deterministic_tiebreak_by_venue_index`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_skips_disabled_stale_toxic_venues`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_disabled_when_kill_switch_active`
- Invariant: `|X| <= band_vol` implies no hedge orders
- Invariant: Total hedge size <= `max_step_tao`
- Invariant: Venues at `dist_liq_sigma <= liq_crit_sigma` are hard-skipped

---

### Milestone G — Multi-venue quoting model (canonical Section 9–11)
**Goal:** quoting per venue reflects:
- Avellaneda–Stoikov baseline,
- basis/funding shifts,
- per-venue inventory targets,
- toxicity and liquidation-distance-aware size constraints,
- stable order management (min quote lifetime, tolerance thresholds).

**Status: COMPLETE**

Deliverables:
- Explicit functions with unit tests for:
  - reservation price components
  - half-spread computation
  - edge filters
  - size constraints and shrink logic
  - cancel/replace logic thresholds

**What shipped:**
- `paraphina/src/mm.rs::compute_mm_quotes()`: Full quote generation (L139-L270)
- `paraphina/src/mm.rs::compute_single_venue_quotes()`: Per-venue AS model (L323-L647)
- Reservation price: `r_v = S_t + β_b*b_v + β_f*f_v - γ*(σ_eff^2)*τ*inv_deviation` (L419-L441)
- AS half-spread: `δ* = (1/γ) * ln(1 + γ/k)` with vol scaling (L378-L415)
- Size model: Quadratic objective with margin/liq-distance/delta-limit constraints (L499-L624)
- Order management: `should_replace_order()` with lifetime and tolerance logic
- Per-venue targets: `compute_venue_targets()` with depth and funding weights (L78-L137)

**Evidence:** See `docs/EVIDENCE_PACK.md` §2 (Market making / quote model)

**Acceptance criteria:**
- Passes `paraphina/tests/mm_quote_model_tests.rs::passivity_bid_strictly_below_best_bid`
- Passes `paraphina/tests/mm_quote_model_tests.rs::passivity_ask_strictly_above_best_ask`
- Passes `paraphina/tests/mm_quote_model_tests.rs::passivity_bid_ask_do_not_cross`
- Passes `paraphina/tests/mm_quote_model_tests.rs::reservation_price_decreases_when_long`
- Passes `paraphina/tests/mm_quote_model_tests.rs::reservation_price_increases_when_short`
- Passes `paraphina/tests/mm_quote_model_tests.rs::hardlimit_produces_no_quotes_even_if_kill_switch_false`
- Passes `paraphina/tests/mm_quote_model_tests.rs::kill_switch_produces_no_quotes`
- Passes `paraphina/tests/mm_quote_model_tests.rs::warning_regime_widens_spread`
- Passes `paraphina/tests/mm_quote_model_tests.rs::warning_regime_caps_size`
- Passes `paraphina/tests/mm_quote_model_tests.rs::disabled_venue_produces_no_quotes`
- Passes `paraphina/tests/mm_quote_model_tests.rs::high_toxicity_venue_produces_no_quotes`
- Passes `paraphina/tests/mm_quote_model_tests.rs::critical_liquidation_distance_produces_no_quotes`
- Passes `paraphina/tests/mm_quote_model_tests.rs::size_respects_max_order_size`
- Passes `paraphina/tests/mm_quote_model_tests.rs::size_respects_lot_size`
- Passes `paraphina/tests/mm_quote_model_tests.rs::liquidation_warning_shrinks_size`
- Passes `paraphina/tests/mm_quote_model_tests.rs::venue_targets_scale_with_depth`
- Passes `paraphina/tests/mm_quote_model_tests.rs::young_passive_order_not_replaced`
- Passes `paraphina/tests/mm_quote_model_tests.rs::old_order_with_price_change_replaced`
- Passes `paraphina/tests/mm_quote_model_tests.rs::extreme_delta_produces_no_quotes`
- Passes `paraphina/tests/mm_quote_model_tests.rs::high_delta_only_allows_risk_reducing`
- Invariant: `bid < best_bid - tick` and `ask > best_ask + tick` (passivity)
- Invariant: `q_global > 0` decreases reservation price (skew toward selling)
- Invariant: `RiskRegime::HardLimit || kill_switch` implies no quotes

---

### Milestone H — Production-ready separation (I/O vs strategy)
**Goal:** preserve a clean boundary:
- strategy core is pure/deterministic
- I/O layer (WS/REST) is async and replaceable
- execution gateway handles venue semantics (TIF/post-only/IOC guard).

**Status: COMPLETE**

Deliverables:
- Action interface: `PlaceOrder`, `CancelOrder`, `CancelAll`, `SetKillSwitch`, …
- Venue adapter trait per exchange
- Rate limiting and retry policies (configurable)

**What shipped:**
- `paraphina/src/actions.rs`: `Action` enum with `PlaceOrder`/`CancelOrder`/`CancelAll`/`SetKillSwitch`
- `paraphina/src/io/mod.rs`: `IoAdapter` trait for venue communication
- `paraphina/src/io/sim.rs`: `SimulatedIoAdapter` for deterministic replay
- `paraphina/src/io/noop.rs`: `NoopIoAdapter` for testing
- `paraphina/src/strategy_core.rs`: Pure strategy layer, no I/O side effects
- `paraphina/src/strategy.rs::StrategyRunner`: Separated event ingestion from action generation

**Evidence:** See `docs/EVIDENCE_PACK.md` §1 (Core loop), §6 (RL / policy interface)

**Acceptance criteria:**
- Passes `paraphina/tests/replay_determinism_tests.rs::test_replay_determinism_single_tick`
- Passes `paraphina/tests/replay_determinism_tests.rs::test_replay_determinism_multi_tick`
- Passes `paraphina/tests/replay_determinism_tests.rs::test_replay_determinism_with_inventory`
- Passes `paraphina/tests/replay_determinism_tests.rs::test_action_ids_deterministic`
- Passes `paraphina/tests/replay_determinism_tests.rs::test_different_seeds_produce_different_batches`
- Passes `paraphina/tests/replay_determinism_tests.rs::test_kill_switch_determinism`
- Passes `paraphina/tests/replay_determinism_tests.rs::test_strategy_output_intents_match_actions`
- Determinism: `paraphina/tests/replay_determinism_tests.rs::test_replay_determinism_multi_tick` passes with fixed seed
- Invariant: `IoAdapter` trait has no mutable access to `GlobalState`
- Invariant: Same `(initial_state, event_stream, config, seed)` produces identical action stream

---

## 3. Experiment roadmap (batch_runs)
This repo already has a strong experiment pattern. Extend it deliberately.

### Recommended experiment ladder
- exp02: profile grid sweeps → “centres”
- exp03: stress vs starting inventory (`INIT_Q_TAO_GRID`) per profile centre
- exp04+: add funding/basis/liquidation perturbations once modeled in sim
- exp05+: allocator A/Bs (exit allocator variants; hedge allocator variants)

### Standard outputs
Every experiment should write under `runs/<exp_name>/`:
- `*_runs.csv` — per-run rows (labels + parsed metrics)
- `*_summary.csv` — grouped summaries

---

## 4. Definition of Done for any feature
A feature is “done” only when:
1) It is implemented behind config if risky,
2) It has unit tests for core math,
3) It is observable (telemetry + stdout summary if needed),
4) It has an experiment that demonstrates improvement or safety,
5) It is documented in `docs/WHITEPAPER.md` drift register (until drift is removed).

---

## RL and GPU Training Track

This track evolves Paraphina from a deterministic strategy into a GPU-trained
reinforcement learning policy, while keeping hard safety controls in Rust.

### RL-0: Foundations (interfaces + instrumentation)
**Goal:** make the current engine "RL-ready" without changing behaviour.

**Status: COMPLETE**

- [x] Add a versioned `Observation` schema derived from `GlobalState` + venues
- [x] Add a `Policy` interface with a default `HeuristicPolicy` (current logic)
- [x] Log policy inputs/outputs in telemetry (obs_version, policy_version)
- [x] Add deterministic seeding + episode reset mechanics in simulation mode
- [x] Add a "shadow mode" runner (policy proposes, baseline executes)

**What shipped:**
- `src/rl/observation.rs`: `Observation`, `VenueObservation` with `OBS_VERSION=1`
- `src/rl/policy.rs`: `Policy` trait, `HeuristicPolicy` (identity pass-through)
- `src/rl/telemetry.rs`: Policy input/output logging
- `src/rl/runner.rs`: `ShadowRunner` for policy proposal without execution
- `src/rl/sim_env.rs`: `SimEnv` with deterministic seeding and episode reset

**Evidence:** See `docs/EVIDENCE_PACK.md` §9 (RL / training scaffolding)

**Acceptance criteria:**
- Passes `paraphina/tests/rl_determinism_tests.rs::test_observation_deterministic`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_observation_version`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_policy_action_deterministic`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_episode_determinism`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_different_seeds_produce_different_results`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_shadow_mode_no_execution_impact`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_heuristic_policy_is_identity`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_policy_reset_episode`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_multiple_episodes_with_reset`
- Determinism: Same seed produces identical observation sequences
- Invariant: HeuristicPolicy returns identity action (no modifications)

**Exit criteria: SATISFIED**
- Replays are deterministic and byte-for-byte reproducible ✓
- Telemetry is sufficient to reconstruct rewards and constraints ✓

### RL-1: Gym-style environment and vectorised simulation
**Goal:** turn the simulator into a high-throughput training environment.

**Status: COMPLETE**

- [x] Create `paraphina_env` wrapper (Python bindings) around Rust sim
- [x] Support vectorised rollouts (N environments in parallel)
- [x] Add domain randomisation knobs:
  - fees, spreads, slippage model
  - funding regimes
  - volatility regimes
  - venue staleness / disable events

**What shipped:**
- `src/rl/sim_env.rs`: `SimEnv` with step/reset/obs interface
- `src/rl/sim_env.rs`: `VecEnv` for N parallel environments
- `src/rl/domain_rand.rs`: `DomainRandSampler` with fee/spread/funding/volatility randomisation
- `paraphina_env`: PyO3 bindings exposing `SimEnv`, `VecEnv`, `TrajectoryCollector`

**Evidence:** See `docs/EVIDENCE_PACK.md` §9 (RL / training scaffolding)

**Acceptance criteria:**
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_sim_env_determinism_same_seed_same_actions`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_sim_env_determinism_with_domain_rand`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_sim_env_different_seeds_different_results`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_vec_env_determinism`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_vec_env_smoke`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_vec_env_custom_actions`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_domain_rand_sampler_determinism`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_episode_termination_max_ticks`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_episode_termination_kill_switch`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_observation_structure`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_vec_env_independence`
- Invariant: VecEnv episodes are independent (no cross-contamination)
- Invariant: Same seed + domain_rand_seed produces identical rollouts

**Exit criteria: SATISFIED**
- Sustained high-step throughput (enough to saturate GPU training) ✓
- Training/eval parity: same reward + constraints computed everywhere ✓

### RL-2: Imitation learning baseline (behaviour cloning)
**Goal:** train a policy to imitate the heuristic strategy.

**Status: COMPLETE**

- [x] Generate large trajectory datasets from heuristic policy
- [x] Train BC policy on bounded "control surface" actions (spread/size/offset)
- [x] Evaluate: action error, PnL parity, risk parity under randomisation

**Exit criteria**
- BC policy matches baseline within tolerance and does not increase kill rate

**What shipped:**
- `src/rl/action_encoding.rs`: Versioned action encoding (`ACTION_VERSION=1`) with bounded ranges
- `src/rl/trajectory.rs`: `TrajectoryCollector` for deterministic dataset generation
- `src/rl/observation.rs`: Versioned observation schema (`OBS_VERSION=1`)
- `paraphina_env`: Python bindings for `TrajectoryCollector` class
- `python/rl2_bc/`: Dataset generation, BC training (PyTorch MLP), and evaluation scripts

**Evidence:** See `docs/EVIDENCE_PACK.md` §9 (RL / training scaffolding)

**Acceptance criteria:**
- Passes `paraphina/tests/rl2_bc_tests.rs::test_action_encoding_determinism_identity`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_action_encoding_determinism_varied`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_action_encoding_round_trip_identity`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_action_encoding_round_trip_extreme_values`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_action_encoding_clamping`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_trajectory_collection_determinism_small`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_trajectory_collection_different_seeds`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_trajectory_metadata_versions`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_trajectory_record_dimensions`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_observation_to_features_determinism`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_heuristic_policy_determinism`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_heuristic_policy_produces_identity`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_observation_deterministic`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_episode_determinism`
- Invariant: `ACTION_VERSION` and `OBS_VERSION` match between train and inference
- Invariant: Same seed produces byte-identical trajectories

### RL-3: GPU RL baseline (robust)
**Goal:** safely improve on BC using online RL in simulation.

- [ ] PPO baseline on control-surface actions
- [ ] Add constrained RL (Lagrangian penalties) to keep budgets:
  kill_prob, drawdown, basis/delta usage
- [ ] Continuous evaluation suite + regression gates

**Exit criteria**
- Measurable improvement in risk-adjusted score with no budget regressions

### RL-4: Model-based RL (advanced)
**Goal:** improve sample efficiency and robustness.

- [ ] Train a learned world model from trajectories
- [ ] Dreamer-style training / imagination rollouts
- [ ] Always validate final candidates in “true” simulator (anti-exploitation)

**Exit criteria**
- Consistent improvements across random seeds and stress scenarios

### RL-5: Shadow deployment and safety validation
**Goal:** production-grade validation without trading risk.

- [ ] Shadow inference in prod (policy proposes, baseline executes)
- [ ] Counterfactual evaluation via replay
- [ ] Latency + failure-mode testing (timeouts, invalid outputs)
- [ ] Model card + deployment checklist

**Exit criteria**
- Stable inference, stable decisions, no safety constraint violations in shadow

### RL-6: Limited live execution
**Goal:** controlled live rollout.

- [ ] Start with tiny caps + strict kill thresholds
- [ ] Gradually increase caps only if budgets remain satisfied
- [ ] Continuous monitoring and automatic rollback to baseline

**Exit criteria**
- Sustained alignment with risk budgets in live conditions

## Long-term: “Fully Optimised Strategy” Track (Quant Optimisation → GPU RL)

This repo evolves in three layers:
1) deterministic baseline strategy (production-safe),
2) automated quant optimisation (search + Monte Carlo + adversarial stress),
3) GPU-trained RL policies behind a hard deterministic safety shield.

### Phase A — Quant Optimisation Foundation (pre-RL, highest ROI)
**Goal:** improve robustness and performance without introducing ML risk.

**Status: COMPLETE (A1 + A2 fully implemented)**

- [x] **Scenario library (seeded, reproducible)** (v1 COMPLETE, promotion-critical)
  - **v1 implemented (10 scenarios):**
    - Volatility regimes: low/medium/high (3)
    - Liquidity shocks: spread widening + depth thinning (2)
    - Venue outage: disabled venue window (1)
    - Funding inversions: sign flip + drift (2)
    - Basis spikes: positive + negative (2)
  - **Implemented:** `batch_runs/phase_a/scenario_library_v1.py` (generator/check/smoke CLI)
  - **Manifest:** `scenarios/v1/scenario_library_v1/manifest_sha256.json` (SHA-256 verified)
  - **Smoke suite:** `scenarios/suites/scenario_library_smoke_v1.yaml` (CI-friendly, 5 scenarios)
  - **Full suite:** `scenarios/suites/scenario_library_v1.yaml` (all 10 scenarios, promotion-critical)
  - **CI workflow:** `.github/workflows/scenario_library_smoke.yml`
  - **Integrated into Phase A Promotion Pipeline:**
    - `batch_runs/phase_a/promote_pipeline.py` includes scenario library by default
    - In `--smoke` mode: uses `scenario_library_smoke_v1.yaml` (5 scenarios)
    - In full mode: uses `scenario_library_v1.yaml` (all 10 scenarios)
    - CLI flags: `--skip-scenario-library`, `--scenario-library-suite PATH`
    - Manifest integrity verified at pipeline start (fail-fast if mismatch)
    - Results recorded in `PROMOTION_RECORD.json` (ran/skipped/passed/errors)
    - Evidence verification includes scenario library suite artifacts
  - **Remaining for v2:** latency / partial fill / cancel storm modelling
- [x] **Tail risk metrics emitted** (A1)
  - `mc_summary.json` schema_version=2 includes `tail_risk` section
  - PnL quantiles (p01, p05, p50, p95, p99)
  - PnL VaR/CVaR at alpha=0.95
  - Max drawdown quantiles and VaR/CVaR
  - Kill probability with Wilson 95% CI (point estimate, lower, upper)
  - **Implemented:** `paraphina/src/tail_risk.rs`, `paraphina/src/bin/monte_carlo.rs`
- [x] **Pareto harness scaffold** (A1)
  - `batch_runs/exp_phase_a_pareto_mc.py` provides:
    - Deterministic knob sweeps (grid or seeded random)
    - Isolated candidate runs with evidence pack verification
    - Pareto frontier computation (multi-objective)
    - Risk-tier budget selection (kill_prob_ci_upper, drawdown_cvar, min_mean_pnl)
    - Promoted config output (env file + promotion record JSON)
  - Usage: `python3 batch_runs/exp_phase_a_pareto_mc.py --smoke`
- [x] **Monte Carlo runner at scale** **IMPLEMENTED**
  - Deterministic sharding with `--run-start-index` and `--run-count` flags
  - JSONL output (`mc_runs.jsonl`) for per-run results
  - `monte_carlo summarize` mode for aggregating sharded results
  - `batch_runs/phase_a/mc_scale.py` Python harness with `plan`, `run-shard`, `aggregate`, `smoke` commands
  - `.github/workflows/mc_scale_smoke.yml` CI gate
  - Documentation: `docs/PHASE_A_MONTE_CARLO_SCALE.md`
  - **Seed contract**: `seed_i = base_seed + i` (u64 wrap, deterministic)
  - Evidence pack verification at shard and aggregate levels
  - **Integrated into promotion pipeline**: `--mc-shards N` flag in `promote_pipeline.py`
  - Trial records include `mc_backend` metadata for audit trail
- [x] **Adversarial / worst-case search** (A2) **IMPLEMENTED**
<!-- STATUS: CEM = IMPLEMENTED -->
  - `batch_runs/phase_a/adversarial_search_promote.py` provides:
    - **Cross-Entropy Method (CEM)** adversarial search (per WHITEPAPER B2)
    - Maintains mean/std per continuous parameter with elite fraction update
    - Deterministic scenario generation with stable filenames
    - Adversarial scoring: maximize kill_switch, drawdown; minimize mean_pnl
    - Top-K failure scenario promotion to `scenarios/v1/adversarial/generated_v1/`
    - Path-based regression suite: `scenarios/suites/adversarial_regression_v2.yaml`
    - Uses `sim_eval run` with `--output-dir` for isolated outputs
    - Verifies evidence with `verify-evidence-tree`
    - Python unit tests: `batch_runs/phase_a/tests/test_adversarial_search.py`
  - Legacy v1 harness: `batch_runs/exp_phase_a_adversarial_search.py`
    - Generates v1 suite with inline env_overrides
  - CI gate: `.github/workflows/adversarial_regression.yml`
    - Runs adversarial smoke search + suite on PRs
    - Verifies evidence packs for all scenarios
  - Usage: `python3 -m batch_runs.phase_a.adversarial_search_promote --smoke --out runs/adv_smoke`
  - Documentation: `docs/PHASE_A_ADVERSARIAL_SEARCH.md`
  - **Remaining:** ADR integration, time-to-failure minimization
- [x] **Multi-objective tuning of strategy knobs** (A2)
  - `batch_runs/phase_a/promote_pipeline.py` provides:
    - Deterministic candidate generation (seeded RNG + evolutionary mutation)
    - Multi-objective Pareto frontier computation
    - Objectives: maximize mean_pnl, minimize kill_prob_ci_upper, minimize drawdown_cvar
    - Budget-tier selection with deterministic tie-breaking
    - Outputs: `trials.jsonl`, `pareto.json`, `pareto.csv`
  - Usage: `python3 -m batch_runs.phase_a.promote_pipeline --smoke --study-dir runs/phaseA_smoke`
  - Unit tests: `batch_runs/phase_a/tests/test_pareto.py`, `test_budgets.py`, `test_winner_selection.py`
- [x] **Promotion pipeline** (A2)
  - `batch_runs/phase_a/promote_pipeline.py` implements budget-gated promotion:
    - Creates isolated trial directories: `runs/phaseA/<study>/<trial_id>/`
    - Writes `candidate.env` with configuration overrides
    - Runs `monte_carlo` with evidence pack generation
    - Runs out-of-sample suite: `scenarios/suites/research_v1.yaml`
    - Runs adversarial regression suite: `scenarios/suites/adversarial_regression_v1.yaml`
    - Verifies evidence packs (`sim_eval verify-evidence-tree`)
    - Parses metrics from `mc_summary.json` (JSON artifacts, not stdout)
    - Promotes winners to: `configs/presets/promoted/<tier>/phaseA_<study>_<timestamp>.env`
    - Writes `PROMOTION_RECORD.json` with full provenance
  - Documentation: `docs/PHASE_A_PROMOTION_PIPELINE.md`
  - Unit tests: `batch_runs/phase_a/tests/test_env_parsing.py`
- [x] **Phase AB Promotion Gate (Strict Mode)** **IMPLEMENTED**
  - `batch_runs/phase_ab/cli.py` provides the `gate` command:
    - Institutional-grade promotion gate with deterministic exit codes
    - Exit codes: PASS=0, FAIL=1, HOLD=2, ERROR=3 (distinct and auditable)
    - Mandatory evidence verification (cannot be skipped)
    - Required seed for reproducibility
    - Writes all standard outputs: `phase_ab_manifest.json`, `confidence_report.json`, `confidence_report.md`, `evidence_pack/manifest.json`, `phase_ab_summary.json`
  - CLI: `python3 -m batch_runs.phase_ab.cli gate --out-dir <path> --seed <int> [--auto-generate-phasea | --candidate-run <path>]`
  - CI workflow: `.github/workflows/phase_ab_promotion_gate.yml`
    - Manual dispatch (workflow_dispatch) for controlled promotion decisions
    - Configurable seed and n_bootstrap via workflow inputs
    - Uploads artifacts: `phase-ab-gate-artifacts`, `phase-ab-gate-evidence-pack`
    - Writes detailed GitHub Actions job summary
  - Unit tests: `tests/test_phase_ab_exit_codes.py` (32 tests covering exit code contracts)
  - Documentation: `docs/PHASE_AB_PIPELINE.md`
  - **Exit code contract (strict mode):**
    - 0 = PASS (PROMOTE - candidate is provably better)
    - 1 = FAIL (REJECT - candidate fails guardrails)
    - 2 = HOLD (insufficient evidence - needs more data)
    - 3 = ERROR (runtime/IO/parsing failure)
- [x] **Telemetry Contract Gate v1** **IMPLEMENTED**
  - Prevents telemetry schema drift from breaking analytics / Phase B world-model training silently
  - Schema v1 with versioned records (`schema_version: 1`)
  - Required fields: `schema_version`, `t`, `pnl_realised`, `pnl_unrealised`, `pnl_total`, `risk_regime`, `kill_switch`, `kill_reason`, `q_global_tao`, `dollar_delta_usd`, `basis_usd`
  - Optional fields: `fv_available`, `fair_value`, `sigma_eff`, `healthy_venues_used_count`, `healthy_venues_used`
  - Schema files:
    - `docs/TELEMETRY_SCHEMA_V1.md` (human-readable)
    - `schemas/telemetry_schema_v1.json` (machine-readable)
  - Validator: `tools/check_telemetry_contract.py`
    - Exit codes: 0=OK, 1=contract violation, 2=file not found, 3=internal error
  - Rust: `paraphina/src/strategy.rs` emits `schema_version: 1` in all telemetry records
  - Tests:
    - `tests/test_telemetry_contract_gate.py` (Python unit tests + subprocess integration)
    - `paraphina/tests/telemetry_schema_tests.rs` (Rust schema compliance)
  - CI: `.github/workflows/telemetry_contract_gate.yml`
  - Invariants enforced: monotonic tick, finite numerics, valid enum values, backwards compatibility

### Phase B — World Model (Learned Simulator) on GPUs
**Goal:** learn a high-fidelity dynamics model from telemetry so RL is sample-efficient.

- [x] Telemetry schema stabilisation (what the world model needs)
  - **Implemented via Telemetry Contract Gate v1** (see above)
  - observations: books, spreads/depth, funding, basis, fills, latency proxies
  - actions: quotes, cancels, exits, hedges
  - outcomes: fills, slippage, markouts, pnl, risk events
- [ ] World model training pipeline (offline)
  - ensemble models + uncertainty estimation
  - domain randomisation hooks
- [ ] Evaluation
  - compare world-model rollouts vs true simulator / historical telemetry
  - reject if model error increases tail risk estimates

### Phase C — Safe RL (GPU) behind deterministic risk shield
**Goal:** RL improves execution/hedging/quoting while never violating invariants.

- [ ] Define “policy surfaces” (bounded control only)
  - quote spread/size multipliers, small skew shifts, hedge allocator weights, exit prioritisation
- [ ] Implement safety shield
  - delta/basis/liquidation/daily-loss constraints enforced deterministically
  - kill-switch remains deterministic
- [ ] RL training
  - model-based RL (Dreamer/MuZero-style) + constrained RL
  - offline RL + conservative objectives
- [ ] Shadow mode in live gateway
  - policy proposes actions, baseline executes
  - log counterfactuals and divergence metrics
- [ ] Gated execution rollout
  - tight caps, gradual expansion only after passing safety scorecards

### Definition of Done (for “Fully Optimised”)
A strategy revision is “fully optimised” only if it:
- improves out-of-sample tail metrics (CVaR / worst-quantile drawdown),
- reduces kill probability with statistical confidence,
- passes adversarial regression suite,
- preserves determinism and replayability,
- keeps safety invariants as the final authority (even with RL enabled).
