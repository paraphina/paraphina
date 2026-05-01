# Phase 5 Status

- Updated UTC: `2026-05-01T01:44:28Z`
- Baseline ID: `phase5_promoted_4v_fvdisabled_lighter`

## Topology

- Topology source: `/etc/paraphina/stage_overlay.env`
- Live connectors: `aster,extended,hyperliquid,lighter,paradex`
- FV-disabled venues: `none`
- Excluded venues: `none`

## Venue Roles

- `hyperliquid`: `fill`
- `aster`: `fill`
- `paradex`: `fill`
- `lighter`: `fill`
- `extended`: `fill`

## Serialized Mainline

- `phase5_live_canary_breach_multiflatten_60m_qual`: `promoted`
  objective: Qualify the multi-venue canary-breach flatten control path through the 60m live gate.
  surface_id: `e48d82243a10937f`
- `phase5_live_final_long_economics_soak`: `rolled_back`
  objective: Run the long-horizon economics soak on the qualified promoted 4-venue surface.
  surface_id: `d562613ea7c7b0f9`
- `phase5_live_long_horizon_breach_forensics`: `promoted`
  objective: Diagnose long-horizon breach behavior after a failed 60m or longer qualification run.
- `phase5_live_hl_breach_deconflict_requal`: `hold`
  objective: Requalify the live surface after deconflicting Hyperliquid canary-breach emergency exits from competing hedge and cancel traffic.
  surface_id: `c5c2e0f8b2170429`
- `phase5_live_hl_clearinghouse_state_truth`: `hold`
  objective: Add Hyperliquid websocket clearinghouseState account truth and requalify the live surface on the unchanged 4-venue topology.
  surface_id: `c5c2e0f8b2170429`
- `phase5_live_pre_restore_cleanup_requal`: `promoted`
  objective: Add bounded pre-restore cleanup on orderly canary exit and requalify rollback cleanliness on the unchanged 4-venue surface.
  surface_id: `c5c2e0f8b2170429`
- `phase5_live_hl_clearinghouse_state_truth_rerun`: `hold`
  objective: Rerun Hyperliquid clearinghouseState truth on the unchanged 4-venue topology after rollback-cleanliness requalification.
  surface_id: `c5c2e0f8b2170429`
- `phase5_live_hl_ws_account_snapshot_normalization`: `hold`
  objective: Normalize Hyperliquid websocket account snapshots so clearinghouseState frames feed usable venue-time account truth into the live controller.
  surface_id: `c5c2e0f8b2170429`
- `phase5_live_canary_breach_response_window_semantics`: `hold`
  objective: Requalify the promoted 4-venue surface after fixing canary-breach response window semantics so an in-flight authoritative exit gets the full configured response budget.
  surface_id: `c5c2e0f8b2170429`
- `phase5_live_breached_venue_exit_scope_deconflict`: `hold`
  objective: Requalify the promoted 4-venue surface by keeping canary-breach emergency control focused on the actually breached venue instead of fanning out secondary venue exits during the same response window.
  surface_id: `c5c2e0f8b2170429`
- `phase5_live_aster_one_lot_residual_unwind_requal`: `promoted`
  objective: Requalify the promoted 4-venue surface by allowing the runtime to actively flatten an exact one-lot Aster residual instead of leaving it stranded until restore cleanup.
  surface_id: `c5c2e0f8b2170429`
- `phase5_live_final_long_economics_soak_rerun`: `promoted`
  objective: Rerun the long-horizon economics soak after the Hyperliquid breach-path deconfliction fix requalifies the control path.
  surface_id: `c5c2e0f8b2170429`
- `phase5_surface_final_closeout`: `rolled_back`
  objective: Freeze the final Phase 5 promoted 4-venue surface and write the final role/spec packet.
  surface_id: `e48d82243a10937f`
- `phase5_reopened_fill_surface_gap_audit`: `promoted`
  objective: Reopen Phase 5 around world-class multi-venue fill qualification instead of merely operational cleanliness.
  surface_id: `e48d82243a10937f`
- `phase5_aster_fill_promotion_program`: `rolled_back`
  objective: Promote Aster from anchor/FV-only behavior toward primary_fill if the blocker audit shows our control or quoting path is suppressing it.
- `phase5_aster_fill_residual_control_requal`: `hold`
  objective: Keep Aster in fill mode while eliminating projected inventory-brake timeouts that currently restart the live loop before the branch can qualify.
- `phase5_aster_fill_budget_partition_requal`: `promoted`
  objective: Keep Aster in fill mode while partitioning the tiny canary quote budget away from concurrent Hyperliquid/Lighter/Paradex MM fills so Aster can qualify cleanly.
  surface_id: `97d2b6d65528d20a`
- `phase5_paradex_fill_promotion_program`: `hold`
  objective: Promote Paradex from anchor/FV-only behavior toward primary_fill if the blocker audit shows our current quoting path is suppressing it.
  surface_id: `d215dec0e10ca9ec`
- `phase5_paradex_fill_residual_control_requal`: `rolled_back`
  objective: Keep Paradex in fill mode while preventing rapid residual accumulation and restart-driven rollback during the isolated Paradex promotion branch.
  surface_id: `1e5767531f7da95f`
- `phase5_paradex_fill_interactive_competitiveness_requal`: `hold`
  objective: Keep Paradex in isolated fill mode while switching its public market-data feed to interactive book visibility so API maker quotes compete against the venue's UI-priority top-of-book more intelligently.
  surface_id: `5751ec45c433d8b9`
- `phase5_paradex_fill_generated_spread_cap_requal`: `promoted`
  objective: Keep Paradex in isolated fill mode on the interactive feed while capping Paradex generated quote spread so API maker quotes actually compete for top-of-book fills.
  surface_id: `8a0995a5a688b777`
- `phase5_lighter_fill_requalification_program`: `hold`
  objective: Keep Lighter FV-disabled but promote it from anchor-only behavior toward isolated primary_fill candidacy on the reopened all-5 program.
  surface_id: `4748803b9212d1d3`
- `phase5_lighter_fill_residual_control_requal`: `hold`
  objective: Keep Lighter in isolated fill mode while preventing fast residual accumulation from tripping canary limits before the branch can qualify.
  surface_id: `2ad560b026c1e81c`
- `phase5_lighter_projected_brake_cancel_only_requal`: `hold`
  objective: Keep Lighter in isolated fill mode while forcing projected-only inventory-brake states to remain cancel-only so the control path cannot create self-inflicted reduce-only shorts before Lighter maker qualification is measured cleanly.
- `phase5_lighter_actual_residual_unwind_requal`: `hold`
  objective: Keep Lighter in isolated fill mode while allowing the inventory-brake path to emit a bounded reduce-only flatten from internal Lighter state when fresh account snapshots lag, so small real Lighter carry can unwind before the hard canary cap kills the run.
- `phase5_lighter_actual_residual_cancel_scope_requal`: `hold`
  objective: Keep Lighter in isolated fill mode while forcing inventory-brake on real Lighter carry to cancel all tracked Lighter live orders before issuing the reduce-only exit, so the flatten path is not competing with stranded maker orders on the same venue.
- `phase5_lighter_pre_restore_cleanup_convergence_requal`: `promoted`
  objective: Preserve the clean fill-positive Lighter cancel-scope branch while making pre-restore cleanup converge before shadow restore, so planned run completion no longer leaves a final lingering Lighter open order that downgrades the short gate.
  surface_id: `e48d82243a10937f`
- `phase5_paradex_runtime_stale_resilience_requal`: `promoted`
  objective: Eliminate repeated non-target Paradex stale-market kills so unrelated venue-promotion branches can survive medium and long live gates on the 4-venue surface.
- `phase5_extended_rescue_execution_program`: `promoted`
  objective: Rescue Extended to the same connector/platform standard as the promoted venues.
  surface_id: `c5c2e0f8b2170429`
- `phase5_extended_anchor_reentry_qual`: `hold`
  objective: Reintroduce Extended onto the integrated 5-venue surface as an anchor venue so the new private account/order truth path is exercised under guarded runtime conditions before Extended is asked to act as a fill venue.
  surface_id: `0df9c3c43bc6ae1e`
- `phase5_extended_anchor_budget_partition_requal`: `hold`
  objective: Keep Extended reintroduced as anchor on the integrated 5-venue surface while partitioning the tiny live canary quote budget so Extended can stay on-surface without stacked multi-venue residuals blowing the canary before fill promotion.
  surface_id: `8754cfb62f8d8143`
- `phase5_extended_anchor_presence_requal`: `hold`
  objective: Keep Extended healthy on the integrated 5-venue anchor surface while restoring nonzero Extended quote eligibility after the prior budget-partition rung over-suppressed all venues into a full quoting gap.
  surface_id: `dc4c8fd2e65fc76d`
- `phase5_integrated_multi_venue_anchor_residual_control_requal`: `blocked`
  objective: Requalify the integrated 5-venue anchor surface if Extended stays healthy but multi-venue residual accumulation still trips canary or restores dirty.
  surface_id: `8754cfb62f8d8143`
- `phase5_extended_fill_promotion_program`: `hold`
  objective: Promote Extended from anchor re-entry to isolated primary_fill candidacy on the reopened all-5 program.
  surface_id: `778897fea2604bf5`
- `phase5_extended_fill_budget_partition_requal`: `promoted`
  objective: Keep Extended in isolated fill mode while partitioning the tiny live canary quote budget away from Hyperliquid and the other non-target venues so Extended can be measured as the primary execution venue without the old Hyperliquid kill path dominating the rung.
  surface_id: `95efe1d2e8ff4d83`
- `phase5_all5_integrated_execution_reentry_qual`: `hold`
  objective: Reintroduce all five venues on one integrated live execution surface so the reopened program measures true multi-venue primary-fill behavior instead of isolated venue branches.
  surface_id: `14eb1a01acf85ad0`
- `phase5_all5_execution_interference_or_residual_requal`: `hold`
  objective: Requalify the all-5 integrated execution surface by correcting the canary open-order budget mismatch that caused the first integrated rung to kill before any fill or inventory signal could develop.
  surface_id: `14eb1a01acf85ad0`
- `phase5_all5_execution_open_order_control_requal`: `hold`
  objective: Requalify the all-5 integrated execution surface with runner-side open-order interference control only if the higher tranche-local open-order budget does not remove the early integrated kill.
  surface_id: `14eb1a01acf85ad0`
- `phase5_all5_execution_competitiveness_requal`: `hold`
  objective: Requalify the all-5 integrated execution surface if it becomes operationally clean but still fails to produce strategically credible multi-venue fill participation.
  surface_id: `99c0a5c23c949328`
- `phase5_all5_projected_quote_budget_requal`: `hold`
  objective: Requalify the all-5 integrated execution surface by fixing projected MM quote allocation so flat symmetric passive quoting no longer self-triggers projected net and gross brake across all five venues.
  surface_id: `14eb1a01acf85ad0`
- `phase5_all5_projected_state_convergence_requal`: `hold`
  objective: Requalify the all-5 integrated execution surface if projected MM budget allocation lands but projected brake or live-order accounting still fails to converge under the integrated surface.
  surface_id: `14eb1a01acf85ad0`
- `phase5_all5_live_order_projection_accounting_requal`: `hold`
  objective: Requalify the integrated all-5 surface if post-plan projected brake convergence is fixed but tracked live orders and intended MM exposure still diverge under guarded live conditions.
  surface_id: `14eb1a01acf85ad0`
- `phase5_all5_native_replace_semantics_requal`: `hold`
  objective: Requalify the integrated all-5 surface if live-order-aware projected accounting is correct but universal cancel-plus-place replace expansion still leaves avoidable overlap risk on venues with native modify semantics.
  surface_id: `14eb1a01acf85ad0`
- `phase5_all5_native_replace_activation_requal`: `hold`
  objective: Requalify the integrated all-5 surface by activating supported native replace traffic on Hyperliquid and Paradex without changing topology, caps, or FV configuration.
  surface_id: `05550367b9ae8c02`
- `phase5_all5_replace_decision_visibility_requal`: `hold`
  objective: Requalify the integrated all-5 surface if native replace transport exists but replace decisions still never surface under guarded live conditions.
  surface_id: `05550367b9ae8c02`
- `phase5_all5_supported_replace_control_layer_requal`: `promoted`
  objective: Requalify the integrated all-5 surface if supported native replace visibility is present but Hyperliquid and Paradex still fail to exercise replace because of one dominant MM control-layer blocker.
  surface_id: `05550367b9ae8c02`
- `phase5_all5_non_hyperliquid_fill_distribution_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if it stays operationally clean after the competitiveness branch but still concentrates fills on Hyperliquid or fewer than two non-Hyperliquid venues.
  surface_id: `d05b1f02f482867f`
- `phase5_all5_non_hyperliquid_residual_control_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if the non-Hyperliquid fill-distribution branch unlocks real non-Hyperliquid fills but still trips canary or restores dirty from residual carry.
  surface_id: `d05b1f02f482867f`
- `phase5_all5_hyperliquid_hedge_reentry_qual`: `hold`
  objective: Requalify the integrated all-5 execution surface if residual-control cleanliness is restored but post-fill inventory relief still starves non-Hyperliquid conversion under soft-governor pressure.
  surface_id: `a1bff74e19f5e204`
- `phase5_all5_non_hyperliquid_hedge_deadband_alignment_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if Hyperliquid hedge re-entry is operationally clean but the hedge controller deadband still sits above the micro-canary soft inventory caps and prevents timely post-fill relief.
  surface_id: `1048cf9f7a93af1a`
- `phase5_all5_non_hyperliquid_soft_cap_size_alignment_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if hedge re-entry is operationally clean but the current Extended quote unit is still too large for the micro-canary soft inventory envelope.
  surface_id: `4307782731740056`
- `phase5_all5_non_hyperliquid_stale_market_hygiene_requal`: `promoted`
  objective: Requalify the integrated all-5 execution surface if size alignment removes soft-governor starvation but medium and long guarded runs still terminate on late stale-market kill boundaries.
  surface_id: `4307782731740056`
- `phase5_all5_non_hyperliquid_stale_market_resilience_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if the current Paradex 500ms lifetime branch still terminates on Extended-driven stale-market boundaries before orderly completion can be measured.
  surface_id: `d882bcebbd0816b0`
- `phase5_all5_extended_ws_session_resilience_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if Extended still terminates the current Paradex 500ms lifetime branch on stale-market boundaries even after switching Extended to the venue-supported depth=1 stream.
  class: `clearance`
  hypothesis blocker: `stale_restart`
  support gate: `shadow_smoke_10m`
  planned credit: `none`
  surface_id: `fdf93c77dafb36b9`
  last observed blocker: `stale_restart`
  last precondition_failed: `false`
  last credit earned: `none`
- `phase5_all5_extended_ws_progress_truth_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if Extended still terminates the current Paradex 500ms lifetime branch on stale-market boundaries even after aligning the connector watchdog and read-timeout settings with the healthy shadow-soak baseline.
  class: `clearance`
  hypothesis blocker: `stale_restart`
  planned credit: `minor`
  surface_id: `fdf93c77dafb36b9`
  last observed blocker: `exact_one_lot_residual_no_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
- `phase5_all5_extended_transport_gap_watchdog_requal`: `hold`
  objective: Re-clear stale_restart on the current higher-conversion all-5 surface if Extended stale episodes are dominated by ping-only/no-data transport gaps rather than runner freeze or future-timestamp deferral.
  class: `clearance`
  hypothesis blocker: `stale_restart`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `1bd39d67f0bf7ff0`
  last observed blocker: `stale_restart`
  last precondition_failed: `false`
  last run surface_id: `822ab570e212c7ac`
- `phase5_all5_extended_bootstrap_no_data_truth_requal`: `hold`
  objective: Localize Extended connect-success to first-book progress if the transport-gap watchdog branch still fails with bootstrap no-data episodes on the same higher-conversion surface.
  class: `localization`
  hypothesis blocker: `stale_restart`
  support gate: `shadow_smoke_10m`
  planned credit: `none`
  surface_id: `1bd39d67f0bf7ff0`
  last observed blocker: `stale_restart`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `e2234fe59f7da8a6`
- `phase5_all5_extended_pre_first_book_bootstrap_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if Extended connect-success sessions still fail before the first publishable book, with zero-frame or frame-no-book bootstrap episodes dominating on the unchanged higher-conversion surface.
  class: `localization`
  hypothesis blocker: `stale_restart`
  support gate: `shadow_smoke_10m`
  planned credit: `none`
  surface_id: `1bd39d67f0bf7ff0`
  last observed blocker: `stale_restart`
  last precondition_failed: `false`
  last credit earned: `none`
  last run surface_id: `9bcd764d975a9386`
- `phase5_all5_extended_first_frame_timeout_alignment_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if Extended official REST seeding works but the connector still times out before the first WS frame on the unchanged higher-conversion surface.
  class: `localization`
  hypothesis blocker: `stale_restart`
  support gate: `shadow_smoke_10m`
  planned credit: `none`
  surface_id: `1bd39d67f0bf7ff0`
  last observed blocker: `stale_restart`
  last precondition_failed: `false`
  last credit earned: `none`
  last run surface_id: `bdfe1443a6a2d20f`
- `phase5_all5_extended_first_frame_transport_progress_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if Extended official REST seeding works and staged bootstrap timeout alignment is in place, but the connector still fails before the first WS frame on the unchanged higher-conversion surface.
  class: `localization`
  hypothesis blocker: `stale_restart`
  support gate: `shadow_smoke_10m`
  planned credit: `none`
  surface_id: `1bd39d67f0bf7ff0`
  last observed blocker: `stale_restart`
  last precondition_failed: `false`
  last credit earned: `none`
  last run surface_id: `2760fe8f66980e45`
- `phase5_all5_extended_control_frame_only_transport_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if Extended REST seeding and seed-bridge preservation work, but connector progress still stalls at control frames before the first WS data frame on the unchanged higher-conversion surface.
  class: `localization`
  hypothesis blocker: `stale_restart`
  support gate: `shadow_smoke_10m`
  planned credit: `none`
  surface_id: `1bd39d67f0bf7ff0`
  last observed blocker: `stale_restart`
  last precondition_failed: `false`
  last credit earned: `none`
  last run surface_id: `dcd66f3d08bd8ebf`
- `phase5_all5_extended_control_frame_only_session_establishment_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if Extended control-frame-only bootstrap failures remain dominant after bounded grace, implying deeper session-establishment transport issues on the unchanged higher-conversion surface.
  class: `localization`
  hypothesis blocker: `stale_restart`
  support gate: `shadow_smoke_10m`
  planned credit: `none`
  surface_id: `1bd39d67f0bf7ff0`
  last observed blocker: `stale_restart`
  last precondition_failed: `false`
  last credit earned: `none`
  last run surface_id: `37a04977cf7c5292`
- `phase5_all5_extended_control_frame_only_socket_stack_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if Extended still reopens on dominant control-frame-only session-establishment failures even after a bounded second-socket hedge is added on the unchanged surface.
  class: `localization`
  hypothesis blocker: `stale_restart`
  support gate: `shadow_smoke_10m`
  planned credit: `none`
  surface_id: `1bd39d67f0bf7ff0`
  last observed blocker: `stale_restart`
  last precondition_failed: `false`
  last credit earned: `none`
  last run surface_id: `a60978bbef3ece18`
- `phase5_all5_extended_control_frame_only_backend_attach_requal`: `promoted`
  objective: Requalify the integrated all-5 execution surface if the tuned Extended socket stack still upgrades cleanly but dominant control-frame-only bootstrap reopen remains on the unchanged higher-conversion surface.
  class: `localization`
  hypothesis blocker: `stale_restart`
  support gate: `shadow_smoke_10m`
  planned credit: `none`
  surface_id: `1bd39d67f0bf7ff0`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `99079bbac178cca6`
- `phase5_all5_extended_post_publish_transport_gap_requal`: `promoted`
  objective: Requalify the integrated all-5 execution surface if Extended reaches first publish successfully on depth=1 but later degrades into ping-alive, no-data transport gaps that reopen stale_restart before downstream residual work can be judged.
  class: `localization`
  hypothesis blocker: `stale_restart`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `1bd39d67f0bf7ff0`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `cd161d43ce9e6b84`
- `phase5_all5_extended_control_frame_only_stream_delivery_requal`: `blocked`
  objective: Requalify the integrated all-5 execution surface if Extended depth1 primary and full-orderbook fallback both upgrade cleanly but still fail to deliver first data before the shared bootstrap deadline.
  class: `localization`
  hypothesis blocker: `stale_restart`
  planned credit: `none`
  surface_id: `1bd39d67f0bf7ff0`
- `phase5_all5_extended_post_publish_stream_delivery_requal`: `blocked`
  objective: Requalify the integrated all-5 execution surface if Extended still reopens stale_restart after first publish because both depth=1 and the bounded post-publish full-orderbook fallback fail to deliver recoverable stream progress before the shared stale deadline.
  class: `localization`
  hypothesis blocker: `stale_restart`
  planned credit: `none`
  surface_id: `1bd39d67f0bf7ff0`
- `phase5_all5_extended_publish_gap_requal`: `blocked`
  objective: Requalify the integrated all-5 execution surface if Extended still receives data but stops emitting publishable book events before the shared stale boundary.
  surface_id: `1bd39d67f0bf7ff0`
- `phase5_all5_extended_freeze_path_requal`: `blocked`
  objective: Requalify the integrated all-5 execution surface if Extended stale episodes are dominated by runner-side freeze/apply gaps on otherwise publishable book flow.
  surface_id: `1bd39d67f0bf7ff0`
- `phase5_all5_extended_future_timestamp_requal`: `blocked`
  objective: Requalify the integrated all-5 execution surface if Extended stale episodes are dominated by future-timestamp deferral rather than transport, publish, or freeze gaps.
  surface_id: `1bd39d67f0bf7ff0`
- `phase5_all5_non_hyperliquid_runtime_soft_cap_alignment_requal`: `blocked`
  objective: Requalify the integrated all-5 execution surface if quote-unit alignment is clean but cumulative one-lot non-Hyperliquid fills still exceed the current micro-canary soft caps.
  surface_id: `83175f986ae15d34`
- `phase5_all5_non_hyperliquid_microstructure_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface by compacting Extended and Paradex quotes toward touch on the now stale-clean, size-aligned surface.
  surface_id: `282159c276ccd8a4`
- `phase5_all5_paradex_interactive_50ms_feed_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if touch-compaction still leaves Paradex under-converting on the interactive book.
  surface_id: `7d40088c7a61e17c`
- `phase5_all5_paradex_lifetime_alignment_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface after the unsupported Paradex interactive 50ms attempt by lowering Paradex minimum quote lifetime on the supported interactive 100ms feed.
  surface_id: `45bd3a8276f7192a`
- `phase5_all5_orderly_exit_residual_hygiene_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if the supported 500ms Paradex lifetime branch produces real non-Hyperliquid fills but still fails promotion because the first audited restore state is dirty before bounded cleanup.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  requires cleared blockers: `stale_restart`
  planned credit: `minor`
  surface_id: `1bd39d67f0bf7ff0`
  last observed blocker: `stale_restart`
  last precondition_failed: `true`
  last credit earned: `none`
- `phase5_all5_non_hyperliquid_exact_one_lot_residual_unwind_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface on the newly stale-clear higher-conversion topology if exact one-lot Extended or Paradex residuals with no live open orders still survive first audit or rollback.
  class: `qualification`
  hypothesis blocker: `exact_one_lot_residual_no_orders`
  requires cleared blockers: `stale_restart`
  planned credit: `minor`
  surface_id: `1bd39d67f0bf7ff0`
  last precondition_failed: `false`
  last run surface_id: `cd161d43ce9e6b84`
- `phase5_all5_non_hyperliquid_actual_residual_cancel_scope_requal`: `blocked`
  objective: Requalify the integrated all-5 execution surface if exact one-lot Extended or Paradex residuals still survive because emergency reduce-only exits race with live orders that are present but not yet canceled on the higher-conversion surface.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  requires cleared blockers: `stale_restart`
  planned credit: `minor`
  surface_id: `1bd39d67f0bf7ff0`
- `phase5_all5_non_hyperliquid_exact_one_lot_execution_truth_requal`: `blocked`
  objective: Requalify the integrated all-5 execution surface if exact one-lot Extended or Paradex residuals still survive even when the runner fallback is used with open_order_count=0 on the stale-clear surface.
  class: `qualification`
  hypothesis blocker: `exact_one_lot_execution_truth`
  requires cleared blockers: `stale_restart`
  planned credit: `minor`
  surface_id: `1bd39d67f0bf7ff0`
- `phase5_all5_paradex_300ms_lifetime_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if Paradex again under-converts after the supported 500ms lifetime branch by lowering only Paradex minimum quote lifetime to 300ms on the same supported interactive 100ms feed.
  class: `qualification`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `29e0db1ea4106acf`
  last precondition_failed: `false`
  last run surface_id: `cfbbfc06652dd021`
- `phase5_all5_paradex_interactive_profile_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if Paradex stays strategically underfilled after the 300ms lifetime branch by switching only the Paradex authenticated trader profile from Pro to interactive on the same stale-clear surface.
  class: `qualification`
  hypothesis blocker: `paradex_underfill_with_interactive_profile`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `bbfdb3b598255ef8`
  last observed blocker: `paradex_underfill_with_interactive_profile`
  last precondition_failed: `false`
  last run surface_id: `932a62b196deb641`
- `phase5_all5_paradex_ui_book_truth_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if Paradex still under-converts after confirmed interactive JWT usage by switching Paradex public pricing back to the documented BBO WS feed and surfacing authoritative UI/API top-of-book truth from the documented REST orderbook endpoints.
  class: `qualification`
  hypothesis blocker: `paradex_underfill_with_ui_book_truth`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `4b98adc8a29f91b5`
  last observed blocker: `paradex_underfill_with_ui_book_truth`
  last precondition_failed: `false`
  last run surface_id: `e111ae7f40a2b704`
- `phase5_all5_paradex_ui_queue_competitiveness_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if Paradex still under-converts after authoritative UI/API book truth is present in standard metrics, implying the next blocker is quote competitiveness against the UI/API queue split rather than missing truth or trader-profile selection.
  class: `qualification`
  hypothesis blocker: `microstructure_underconversion`
  planned credit: `minor`
  surface_id: `40e7991d9f3f4594`
  last observed blocker: `microstructure_underconversion`
  last precondition_failed: `false`
  last run surface_id: `c7bd2299a4dfe6df`
- `phase5_all5_paradex_native_cancel_batch_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if Paradex still under-converts after the faster client-id cancel path is in place, by routing same-tick multi-cancel flow through gateway batching onto native Paradex batch cancel while keeping the inherited current all-5 surface unchanged.
  class: `qualification`
  hypothesis blocker: `microstructure_underconversion`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `40e7991d9f3f4594`
  last observed blocker: `paradex_replace_identity_gap`
  last precondition_failed: `false`
  last run surface_id: `25103f57dd5a86a3`
- `phase5_all5_paradex_native_replace_identity_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if the inherited current all-5 Paradex surface remains replace-heavy and under-converting after native cancel batching is disproven as the dominant mechanism, by resolving client-scoped current MM order identities into Paradex exchange order IDs early enough for native modify to execute without changing topology, feed selection, or trader profile.
  class: `qualification`
  hypothesis blocker: `paradex_replace_identity_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `40e7991d9f3f4594`
  last observed blocker: `paradex_same_side_persistence_gap`
  last precondition_failed: `false`
  last run surface_id: `9b668f4f685736b2`
- `phase5_all5_paradex_same_side_persistence_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if the inherited current all-5 Paradex branch remains under-converting because same-side MM order visibility decays across open-order snapshot gaps before supported replace or suppression-grace can act.
  class: `qualification`
  hypothesis blocker: `paradex_same_side_persistence_gap`
  support gate: `shadow_smoke_10m`
  mechanism gate mode: `live_5m`
  planned credit: `minor`
  surface_id: `40e7991d9f3f4594`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last run surface_id: `9b668f4f685736b2`
- `phase5_all5_paradex_same_side_restore_hygiene_requal`: `hold`
  objective: Requalify the current integrated all-5 Paradex same-side persistence surface if the only remaining failed gate is the first audited restore state going dirty before bounded cleanup, while the inherited same-side control-layer overlay remains unchanged.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  mechanism gate mode: `live_5m`
  planned credit: `minor`
  surface_id: `40e7991d9f3f4594`
  last observed blocker: `paradex_private_order_truth_gap`
  last precondition_failed: `false`
  last run surface_id: `60b92545dbcf792d`
- `phase5_all5_paradex_private_order_truth_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if the current same-side Paradex branch stays operationally clean but still never surfaces bounded gap-grace visibility, implying current-order truth must be refreshed from Paradex private order updates rather than open-order snapshots alone.
  class: `qualification`
  hypothesis blocker: `paradex_private_order_truth_gap`
  support gate: `shadow_smoke_10m`
  mechanism gate mode: `live_5m`
  planned credit: `minor`
  surface_id: `e0baebe04a74fc9a`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last run surface_id: `020ec7bd9f341dff`
- `phase5_all5_paradex_private_truth_restore_hygiene_requal`: `hold`
  objective: Requalify the current integrated all-5 Paradex private-order-truth surface if the only remaining failed gate is the first audited restore state going dirty before bounded cleanup, while the inherited private-order-truth overlay remains unchanged.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `0752749219376f43`
  last observed blocker: `paradex_batch_cancel_request_shape_gap`
  last precondition_failed: `false`
  last run surface_id: `ce73cb323751da1f`
- `phase5_all5_paradex_batch_cancel_canonical_id_requal`: `hold`
  objective: Requalify the integrated all-5 execution surface if the current Paradex private-order-truth surface stays operationally clean but native batch cancel continues to fall back on HTTP 400 because the batch request is still carrying client-scoped identities instead of canonical exchange order IDs.
  class: `qualification`
  hypothesis blocker: `paradex_batch_cancel_request_shape_gap`
  support gate: `shadow_smoke_10m`
  mechanism gate mode: `live_5m`
  planned credit: `minor`
  surface_id: `40f27308a8f1cc05`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last run surface_id: `1d56bc910224dce0`
- `phase5_all5_paradex_batch_cancel_restore_hygiene_requal`: `hold`
  objective: Requalify the current integrated all-5 Paradex batch-cancel canonical-ID surface if the only remaining failed gate is the first audited restore state going dirty before bounded cleanup, while the inherited batch-cancel overlay remains unchanged.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `61f861ba3ee89f50`
  last observed blocker: `no_data_transport_gap`
  last precondition_failed: `false`
  last run surface_id: `3d847524c61b2f20`
- `phase5_all5_current_surface_lighter_post_publish_transport_gap_requal`: `hold`
  objective: Requalify the current integrated all-5 surface if Lighter can bootstrap and publish once, but still trips shared stale-market hygiene before its own connector stale watchdog can recover the session.
  class: `qualification`
  hypothesis blocker: `no_data_transport_gap`
  support gate: `shadow_smoke_30m`
  planned credit: `minor`
  surface_id: `b4199d376a0a74cb`
  last observed blocker: `startup_readiness_gap`
  last precondition_failed: `false`
  last run surface_id: `b6f271de9f8a9de8`
- `phase5_all5_current_surface_startup_readiness_bootstrap_requal`: `hold`
  objective: Requalify the exact current all-5 surface if the new blocker is startup readiness convergence, where shared stale-market hygiene arms before Extended and Paradex can both become market-ready even though the inherited venue-local transport and truth fixes remain in place.
  class: `qualification`
  hypothesis blocker: `startup_readiness_gap`
  support gate: `shadow_smoke_10m`
  mechanism gate mode: `live_5m`
  planned credit: `minor`
  surface_id: `ca9f592dcf7a1a5f`
  last observed blocker: `microstructure_underconversion`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `e3943cecf339e4c3`
- `phase5_all5_current_surface_paradex_queue_preservation_requal`: `hold`
  objective: Requalify the exact current all-5 startup-ready surface if Paradex still under-converts because same-side queue position is lost when the control layer temporarily suppresses a passive quote that could safely remain resting.
  class: `qualification`
  hypothesis blocker: `paradex_queue_position_loss`
  support gate: `shadow_smoke_10m`
  mechanism gate mode: `live_5m`
  planned credit: `minor`
  surface_id: `0c97212bff3ecd7e`
  last observed blocker: `paradex_edge_floor_queue_loss`
  last precondition_failed: `false`
  last run surface_id: `ce16232c0e066256`
- `phase5_all5_current_surface_paradex_edge_floor_queue_hold_requal`: `hold`
  objective: Requalify the exact current startup-ready all-5 surface if Paradex still under-converts because safe same-side orders lose queue position during short, shallow `edge_below_min` dips rather than from startup, truth, or projected-budget suppression.
  class: `qualification`
  hypothesis blocker: `paradex_edge_floor_queue_loss`
  support gate: `shadow_smoke_10m`
  mechanism gate mode: `live_5m`
  planned credit: `minor`
  surface_id: `222b63f0ec8efe7c`
  last observed blocker: `hyperliquid_post_publish_transport_gap`
  last precondition_failed: `false`
  last run surface_id: `10b3f85ac0a5a22d`
- `phase5_all5_current_surface_hyperliquid_post_publish_transport_gap_requal`: `hold`
  objective: Requalify the exact current startup-ready/private-truth/edge-floor all-5 surface if Hyperliquid public book data remains live but runner stale hygiene still kills on delayed published-market freshness.
  class: `qualification`
  hypothesis blocker: `hyperliquid_post_publish_transport_gap`
  support gate: `shadow_smoke_30m`
  planned credit: `minor`
  surface_id: `115410ace49bbdb7`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last run surface_id: `cc82790ca34a3fde`
- `phase5_all5_current_surface_hyperliquid_post_publish_restore_hygiene_requal`: `hold`
  objective: Requalify the exact current startup-ready/private-truth/edge-floor/hyperliquid-aligned all-5 surface if the only remaining failed gate is the first audited restore state going dirty before bounded cleanup, while the inherited live behavior itself remains operationally and strategically qualified.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `9682feef8ee989e4`
  last observed blocker: `microstructure_underconversion`
  last precondition_failed: `false`
  last run surface_id: `f99b7ede87ac2335`
- `phase5_all5_current_surface_paradex_interactive_public_top_requal`: `hold`
  objective: Requalify the exact current restore-clean all-5 surface if Paradex still under-converts because its primary public market snapshots are anchored to API BBO plus delayed UI-touch overlay instead of the interactive public top that actually reflects retail queue competitiveness.
  class: `qualification`
  hypothesis blocker: `paradex_interactive_top_anchor_gap`
  support gate: `shadow_smoke_10m`
  mechanism gate mode: `live_5m`
  planned credit: `minor`
  surface_id: `0cbc6ec16be56e8b`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last run surface_id: `940945bc183ed423`
- `phase5_all5_current_surface_paradex_interactive_public_top_restore_hygiene_requal`: `hold`
  objective: Requalify the exact current interactive-public-top all-5 surface if the only failed gate is the first audited restore state going dirty before bounded cleanup, while the inherited interactive-public-top behavior itself remains operationally clean.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `43741bc9694dcdbd`
  last observed blocker: `paradex_underfill_with_ui_book_truth`
  last precondition_failed: `false`
  last run surface_id: `b0aeffab1b51d580`
- `phase5_all5_current_surface_paradex_ui_book_truth_requal`: `hold`
  objective: Requalify the exact current all-5 surface if Paradex still under-converts after interactive-public-top anchoring by restoring the supported BBO plus authoritative UI/API truth model on the same execution-clean surface.
  class: `qualification`
  hypothesis blocker: `paradex_underfill_with_ui_book_truth`
  support gate: `shadow_smoke_10m`
  mechanism gate mode: `live_5m`
  planned credit: `minor`
  surface_id: `1a6e2f811b8d976c`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last run surface_id: `df98271f29ff5e01`
- `phase5_all5_current_surface_paradex_ui_book_truth_restore_hygiene_requal`: `hold`
  objective: Requalify the exact current Paradex UI-book-truth all-5 surface if the only failed gate is the first audited restore state going dirty before bounded cleanup, while the inherited UI/API-truth behavior itself remains operationally clean.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ced3bc17e7d49505`
  last observed blocker: `lighter_sequence_continuity_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `3fc0efa991845329`
- `phase5_all5_current_surface_lighter_sequence_rebootstrap_requal`: `hold`
  objective: Requalify the exact current all-5 UI-truth surface if Lighter reconnects and re-subscribes, but still lets post-reconnect order-book deltas trip market-cache sequence disorder and stale-market kill before the inherited Paradex competitiveness objective can be measured fairly.
  class: `qualification`
  hypothesis blocker: `lighter_sequence_continuity_gap`
  support gate: `shadow_smoke_30m`
  planned credit: `minor`
  surface_id: `06c3cb53a34f03c6`
  last observed blocker: `extended_post_publish_fallback_rearm_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `0e90d043816c8879`
- `phase5_all5_current_surface_extended_post_publish_fallback_rearm_requal`: `hold`
  objective: Requalify the exact current all-5 UI-truth restore-clean surface if Extended can already recover one post-publish depth1 gap with the existing full-orderbook fallback, but later same-session gaps still reopen stale-market kill because that rescue path never rearms.
  class: `qualification`
  hypothesis blocker: `extended_post_publish_fallback_rearm_gap`
  support gate: `shadow_smoke_30m`
  planned credit: `minor`
  surface_id: `3e0f9290e75ec2e5`
  last observed blocker: `hyperliquid_pre_kill_recovery_alignment_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `2f4767c92ed028ee`
- `phase5_all5_current_surface_hyperliquid_pre_kill_recovery_alignment_requal`: `hold`
  objective: Requalify the exact current all-5 UI-truth restore-clean surface if Hyperliquid still fails late in the rung only because its connector-level reconnect and REST fallback thresholds are slower than the runner stale kill on the same inherited surface.
  class: `qualification`
  hypothesis blocker: `hyperliquid_pre_kill_recovery_alignment_gap`
  support gate: `shadow_smoke_30m`
  planned credit: `minor`
  surface_id: `28252aa74938d88d`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `023f82ec24774df0`
- `phase5_all5_current_surface_hyperliquid_pre_kill_recovery_restore_hygiene_requal`: `hold`
  objective: Requalify the exact Hyperliquid pre-kill-recovery-aligned all-5 surface if the only remaining failed gate is the first audited restore state going dirty before bounded cleanup, while the inherited live behavior itself remains operationally qualified.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `27b6b8555cb6b2ff`
  last observed blocker: `paradex_ui_touch_reference_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `14a335d162463251`
- `phase5_all5_current_surface_paradex_ui_touch_reference_fallback_requal`: `hold`
  objective: Requalify the exact current all-5 restore-clean surface if Paradex still under-converts because the current BBO plus UI-truth path only adjusts from split interactive prices and ignores the more competitive top-level interactive bid/ask already present in authoritative UI truth.
  class: `qualification`
  hypothesis blocker: `paradex_ui_touch_reference_gap`
  support gate: `shadow_smoke_10m`
  mechanism gate mode: `live_5m`
  planned credit: `minor`
  surface_id: `010fe1d0bc5ae101`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `4d39899f0da1b6b0`
- `phase5_all5_current_surface_paradex_ui_touch_reference_fallback_restore_hygiene_requal`: `hold`
  objective: Requalify the exact current Paradex UI touch-reference fallback surface if the only remaining failed gate is the first audited restore state going dirty before bounded cleanup, while the inherited touch-reference behavior itself remains operationally clean.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `2196094e24ed11aa`
  last observed blocker: `topology_fv_reentry_gap`
  last precondition_failed: `false`
  last run surface_id: `b98e89592dd8d838`
- `phase5_all5_paradex_ui_queue_restore_hygiene_requal`: `hold`
  objective: Requalify the current integrated all-5 Paradex competitiveness surface if the only failed gate is the first audited restore state going dirty before bounded cleanup, while the inherited execution surface itself remains operationally clean.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `40e7991d9f3f4594`
  last observed blocker: `runner_freeze_apply_gap`
  last precondition_failed: `false`
  last run surface_id: `7229d7c9edc7fe4b`
- `phase5_all5_current_surface_extended_freeze_path_requal`: `hold`
  objective: Requalify the current integrated all-5 Paradex competitiveness surface if Extended stale/restart failures are driven by runner-side freeze/apply handling on otherwise live post-publish market flow.
  class: `qualification`
  hypothesis blocker: `runner_freeze_apply_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `8ce2e9ed58921ea7`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last run surface_id: `a10bc35f7a2ce131`
- `phase5_all5_current_surface_restore_hygiene_requal`: `hold`
  objective: Requalify the current integrated all-5 surface if the only remaining failed gate is the first audited restore state going dirty before bounded cleanup, while the inherited current-surface Extended freeze-path overlay remains operationally clean.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `8ce2e9ed58921ea7`
  last observed blocker: `transport_gap_watchdog`
  last precondition_failed: `false`
  last run surface_id: `a10bc35f7a2ce131`
- `phase5_all5_current_surface_extended_transport_gap_watchdog_requal`: `hold`
  objective: Requalify the current integrated all-5 surface if Extended stale-watchdog reconnect timing still over-triggers on post-publish transport gaps and collapses the shared live loop before the inherited Paradex competitiveness objective can be measured fairly.
  class: `qualification`
  hypothesis blocker: `transport_gap_watchdog`
  support gate: `shadow_smoke_30m`
  planned credit: `minor`
  surface_id: `2e8c3cdd64e85180`
  last observed blocker: `aster_bridge_wait_timeout`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `e38365d7bad08b3d`
- `phase5_all5_current_surface_aster_bridge_wait_requal`: `hold`
  objective: Requalify the current integrated all-5 surface if Aster loop1 snapshot-to-bridge recovery is exhausting the bridge-wait budget and killing the canary before the inherited all-5 execution surface can be measured fairly.
  class: `qualification`
  hypothesis blocker: `aster_bridge_wait_timeout`
  support gate: `shadow_smoke_30m`
  planned credit: `minor`
  surface_id: `29e0263a595c6760`
  last observed blocker: `microstructure_underconversion`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `e1c73f47c1fea7f1`
- `phase5_all5_fv_reentry_qual`: `hold`
  objective: Re-enable all five venues in fair value on the now all-5 execution-qualified surface so the reopened program reaches the full perfect-topology target before long soak.
  class: `topology`
  hypothesis blocker: `topology_fv_reentry_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `major`
  surface_id: `d9db3659dfd7bc5d`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `8619d85ef9e6ff30`
- `phase5_all5_fv_reentry_restore_hygiene_requal`: `hold`
  objective: Requalify the exact current all-5 FV-reentry surface if the only failed gate is the first audited restore state going dirty before bounded cleanup, while the inherited all-venue fill roles and current venue-local fixes remain operationally clean.
  class: `topology`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `379cd12acb15a12a`
  last observed blocker: `all_venue_market_frontier_backpressure_gap`
  last precondition_failed: `false`
  last run surface_id: `8a64575ab8270f19`
- `phase5_all5_fv_reentry_market_frontier_backpressure_requal`: `hold`
  objective: Requalify the exact current all-5 FV surface if the remaining blocker is a late internal market-frontier backpressure collapse rather than venue-local transport or fill competitiveness.
  class: `topology`
  hypothesis blocker: `all_venue_market_frontier_backpressure_gap`
  support gate: `shadow_smoke_30m`
  planned credit: `major`
  surface_id: `06e8258105825ca6`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last run surface_id: `31912ce8a11160e1`
- `phase5_all5_fv_reentry_market_frontier_backpressure_restore_hygiene_requal`: `hold`
  objective: Requalify the exact current all-5 FV frontier surface if the only failed gate is the first audited restore state going dirty before bounded cleanup, while the widened frontier controls remain operationally clean.
  class: `topology`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `5ae90eafd219e90e`
  last observed blocker: `no_data_transport_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `256f5ae4165da29a`
- `phase5_all5_fv_reentry_lighter_ticker_backstop_requal`: `hold`
  objective: Requalify the exact current all-5 FV frontier-fixed surface if the remaining blocker is Lighter transport intermittency after the sequence-rebootstrap fix, not renewed frontier collapse, hygiene dirt, or Paradex competitiveness.
  class: `topology`
  hypothesis blocker: `no_data_transport_gap`
  support gate: `shadow_smoke_30m`
  planned credit: `major`
  surface_id: `a7d021268a22bc6a`
  last observed blocker: `extended_degraded_stream_rebootstrap_gap`
  last precondition_failed: `false`
  last run surface_id: `7b133e33b9addbf5`
- `phase5_all5_fv_reentry_extended_degraded_stream_rebootstrap_requal`: `hold`
  objective: Requalify the exact current all-5 FV ticker-backstop surface if Extended can recover once into full-orderbook degraded mode but later stalls there with no bounded rebootstrap path before the generic stale kill.
  class: `topology`
  hypothesis blocker: `extended_degraded_stream_rebootstrap_gap`
  support gate: `shadow_smoke_30m`
  planned credit: `major`
  surface_id: `751bc6d3ffec9958`
  last observed blocker: `extended_pre_kill_degraded_rebootstrap_alignment_gap`
  last precondition_failed: `false`
  last run surface_id: `a17d06f175b08c12`
- `phase5_all5_fv_reentry_extended_pre_kill_degraded_rebootstrap_alignment_requal`: `hold`
  objective: Requalify the exact current all-5 FV Extended degraded-stream rebootstrap surface if the reconnect path exists but is still firing too late to outrun the runner stale-market kill.
  class: `topology`
  hypothesis blocker: `extended_pre_kill_degraded_rebootstrap_alignment_gap`
  support gate: `shadow_smoke_30m`
  planned credit: `major`
  surface_id: `5d4174d420a42e7e`
  last observed blocker: `paradex_interactive_top_anchor_gap`
  last precondition_failed: `false`
  last run surface_id: `312c5d736dd5d5eb`
- `phase5_all5_fv_reentry_extended_pre_kill_degraded_rebootstrap_alignment_restore_hygiene_requal`: `blocked`
  objective: Requalify the exact current all-5 FV Extended pre-kill degraded-stream alignment surface if the only failed gate is the first audited restore state going dirty before bounded cleanup.
  class: `topology`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `d4b55fa30bc0df6e`
- `phase5_all5_fv_reentry_lighter_ticker_backstop_restore_hygiene_requal`: `blocked`
  objective: Requalify the exact current all-5 FV ticker-backstop surface if the only failed gate is the first audited restore state going dirty before bounded cleanup, while the new Lighter backstop remains operationally clean.
  class: `topology`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `0b1473cd900779a6`
- `phase5_all5_fv_reentry_extended_degraded_stream_rebootstrap_restore_hygiene_requal`: `blocked`
  objective: Requalify the exact current all-5 FV Extended degraded-stream rebootstrap surface if the only failed gate is the first audited restore state going dirty before bounded cleanup, while the new Extended degraded-stream rebootstrap remains operationally clean.
  class: `topology`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `700280443473fcb6`
- `phase5_extended_private_truth_stability_requal`: `promoted`
  objective: Requalify Extended runtime truth stability if guarded anchor re-entry shows stale, reconnect, unavailable-account, or distorted public-book behavior.
  surface_id: `0df9c3c43bc6ae1e`
- `phase5_extended_fill_competitiveness_or_residual_requal`: `blocked`
  objective: Requalify Extended isolated fill candidacy if the first fill branch stays healthy but lacks fills or restores dirty.
  surface_id: `c5c2e0f8b2170429`
- `phase5_all5_fv_reentry_paradex_interactive_public_top_requal`: `hold`
  objective: Requalify the exact current all-5 FV surface if Paradex still under-converts because the live public top is anchored to raw API BBO plus delayed UI-touch overlay instead of the interactive public top that already reflects retail-priority liquidity.
  class: `topology`
  hypothesis blocker: `paradex_interactive_top_anchor_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ff51710794b8b370`
  last observed blocker: `paradex_open_snapshot_replace_identity_gap`
  last precondition_failed: `false`
  last run surface_id: `b9a1bbfbbe91af50`
- `phase5_all5_fv_reentry_paradex_interactive_public_top_restore_hygiene_requal`: `blocked`
  objective: Requalify the exact current all-5 FV Paradex interactive-public-top surface if the only failed gate is the first audited restore state going dirty before bounded cleanup.
  class: `topology`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `d19ae27af07f7c91`
- `phase5_all5_fv_reentry_paradex_open_snapshot_replace_identity_requal`: `hold`
  objective: Requalify the exact current all-5 FV surface if Paradex queue-preservation and native replace are blocked because open-order snapshots still carry client ids where the current live order state needs canonical exchange ids.
  class: `topology`
  hypothesis blocker: `paradex_open_snapshot_replace_identity_gap`
  support gate: `shadow_smoke_10m`
  mechanism gate mode: `live_5m`
  planned credit: `minor`
  surface_id: `708399e8a96def75`
  last precondition_failed: `false`
  last run surface_id: `1b33fbd85853bb50`
- `phase5_all5_fv_reentry_paradex_open_snapshot_replace_identity_restore_hygiene_requal`: `hold`
  objective: Requalify the exact current all-5 FV Paradex open-snapshot replace-identity surface if the only failed gate is the first audited restore state going dirty before bounded cleanup.
  class: `topology`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `782d656bcda43a3d`
  last observed blocker: `paradex_edge_floor_queue_loss`
  last precondition_failed: `false`
  last run surface_id: `e8eba811c52f219f`
- `phase5_all5_fv_reentry_paradex_edge_floor_queue_hold_exact_surface_requal`: `hold`
  objective: Requalify the exact current all-5 FV Paradex open-snapshot replace-identity/restore-clean surface if Paradex still under-converts because same-side quotes are mostly suppressed on `edge_below_min` rather than converting into real queue-preserving fills.
  class: `topology`
  hypothesis blocker: `paradex_edge_floor_queue_loss`
  support gate: `shadow_smoke_10m`
  mechanism gate mode: `live_5m`
  planned credit: `minor`
  surface_id: `9d76d66936605cd7`
  last observed blocker: `paradex_edge_floor_shadow_mechanism_gate_gap`
  last precondition_failed: `false`
  last run surface_id: `731a30703de7dad5`
- `phase5_all5_fv_reentry_paradex_edge_floor_queue_hold_exact_surface_restore_hygiene_requal`: `blocked`
  objective: Requalify the exact current all-5 FV Paradex edge-floor queue-hold exact surface if the only failed gate is the first audited restore state going dirty before bounded cleanup.
  class: `topology`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `e69a0dfa68cf0cb8`
- `phase5_all5_fv_reentry_paradex_edge_floor_shadow_mechanism_gate_requal`: `hold`
  objective: Requalify the exact current all-5 FV Paradex edge-floor queue-hold surface when the child support gate is blocked by shadow-only mechanism visibility rather than a new live competitiveness or topology defect.
  class: `topology`
  hypothesis blocker: `paradex_edge_floor_shadow_mechanism_gate_gap`
  support gate: `shadow_smoke_10m`
  mechanism gate mode: `live_5m`
  planned credit: `minor`
  surface_id: `9d76d66936605cd7`
  last observed blocker: `paradex_same_side_persistence_gap`
  last precondition_failed: `false`
  last run surface_id: `731a30703de7dad5`
- `phase5_all5_fv_reentry_paradex_edge_floor_shadow_mechanism_gate_restore_hygiene_requal`: `blocked`
  objective: Requalify the exact current all-5 FV Paradex edge-floor surface if the only failed gate after the shadow-mechanism-gated rerun is the first audited restore state going dirty before bounded cleanup.
  class: `topology`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `9d76d66936605cd7`
- `phase5_all5_fv_reentry_paradex_same_side_persistence_rebalance_requal`: `hold`
  objective: Requalify the exact current all-5 FV Paradex edge-floor surface if Paradex still under-converts because same-side resting order presence decays too quickly after bounded under-edge or post-control suppression windows, even though queue-keep is already exercisable.
  class: `topology`
  hypothesis blocker: `paradex_same_side_persistence_gap`
  support gate: `shadow_smoke_10m`
  mechanism gate mode: `live_5m`
  planned credit: `minor`
  surface_id: `d1db43d04d76a5db`
  last observed blocker: `paradex_ui_touch_reference_gap`
  last precondition_failed: `false`
  last run surface_id: `e73c8a41a0da9f10`
- `phase5_all5_fv_reentry_paradex_same_side_persistence_rebalance_restore_hygiene_requal`: `blocked`
  objective: Requalify the exact current all-5 FV Paradex same-side persistence rebalance surface if the only failed gate is the first audited restore state going dirty before bounded cleanup.
  class: `topology`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `38aa66c9470c3f4e`
- `phase5_all5_fv_reentry_paradex_ui_touch_reference_fallback_requal`: `hold`
  objective: Requalify the exact current all-5 FV re-entry surface if Paradex still under-converts because the interactive public-top path never starts authoritative UI truth polling or applies UI touch-reference fallback, even though the surface is now identity-normalized, restore-clean, same-side-preserving, and operationally fill-positive overall.
  class: `topology`
  hypothesis blocker: `paradex_ui_touch_reference_gap`
  support gate: `shadow_smoke_10m`
  mechanism gate mode: `live_5m`
  planned credit: `minor`
  surface_id: `23e3d00dc49b80e7`
  last observed blocker: `extended_pre_kill_degraded_rebootstrap_alignment_gap`
  last precondition_failed: `false`
  last run surface_id: `df8cd5f88bba97d4`
- `phase5_all5_fv_reentry_paradex_ui_touch_reference_fallback_restore_hygiene_requal`: `blocked`
  objective: Requalify the exact current all-5 FV Paradex UI-touch-reference fallback surface if the only failed gate is the first audited restore state going dirty before bounded cleanup.
  class: `topology`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `4519ecfc15002e95`
- `phase5_all5_fv_reentry_extended_pre_kill_degraded_rebootstrap_alignment_exact_surface_requal`: `hold`
  objective: Requalify the exact current all-5 FV UI-touch-enabled surface if Extended degraded-stream recovery still fires too late to outrun stale-market kill.
  class: `topology`
  hypothesis blocker: `extended_pre_kill_degraded_rebootstrap_alignment_gap`
  support gate: `shadow_smoke_30m`
  planned credit: `major`
  surface_id: `350024172d3c525d`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last run surface_id: `8fa4f9f62e8ec63d`
- `phase5_all5_fv_reentry_extended_pre_kill_degraded_rebootstrap_alignment_exact_surface_restore_hygiene_requal`: `hold`
  objective: Requalify the exact current all-5 FV Extended pre-kill degraded-stream alignment surface if the only failed gate is the first audited restore state going dirty before bounded cleanup.
  class: `topology`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `b6a530a96121f456`
  last observed blocker: `all5_projected_mm_budget_distribution_gap`
  last precondition_failed: `false`
  last run surface_id: `9233ab9dac6567b6`
- `phase5_all5_fv_reentry_projected_mm_budget_distribution_exact_surface_requal`: `hold`
  objective: Requalify the exact current all-5 FV surface if operational cleanliness is proven but projected MM quote budget distribution still prevents all five one-lot fill venues from being selected together.
  class: `topology`
  hypothesis blocker: `all5_projected_mm_budget_distribution_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `89e8e7284dcb63fd`
  last observed blocker: `paradex_interactive_top_anchor_gap`
  last precondition_failed: `false`
  last run surface_id: `34ec094f5c1a4eaf`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_exact_surface_requal`: `hold`
  objective: Requalify the exact current all-5 FV surface after correcting the effective runtime canary profile so the live process actually applies the intended all-five projected quote budget.
  class: `topology`
  hypothesis blocker: `all5_projected_mm_budget_distribution_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `9cbfa344b3fed48f`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last run surface_id: `7e50e774a992db4a`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_restore_hygiene_requal`: `hold`
  objective: Requalify the exact current all-5 effective canary-budget profile surface after the first guarded live rung proved the profile mechanism but held on first audited restore cleanliness.
  class: `topology`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `466be8809f86110b`
  last observed blocker: `paradex_edge_floor_queue_loss`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `54b6a3bd128381e5`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_paradex_edge_floor_queue_hold_requal`: `hold`
  objective: Requalify the exact current all-5 effective canary-budget profile surface if Paradex still under-converts because active interactive-top quotes are dominated by `edge_below_min` suppression rather than converting into durable queue-preserving placements.
  class: `topology`
  hypothesis blocker: `paradex_edge_floor_queue_loss`
  support gate: `shadow_smoke_10m`
  mechanism gate mode: `live_5m`
  planned credit: `minor`
  surface_id: `4d4c550e180d8419`
  last observed blocker: `extended_pre_kill_degraded_rebootstrap_alignment_gap`
  last precondition_failed: `false`
  last credit earned: `none`
  last run surface_id: `198c4219ce86268d`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_extended_pre_kill_degraded_rebootstrap_alignment_requal`: `hold`
  objective: Requalify the exact current all-5 effective canary-budget and Paradex edge-floor surface if Extended still becomes the stale-market trigger before the post-publish recovery path can outrun the runner stale boundary.
  class: `topology`
  hypothesis blocker: `extended_pre_kill_degraded_rebootstrap_alignment_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `7f822b3d2b7de0dd`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `0b8e659929af490b`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_canary_response_flatten_retry_requal`: `hold`
  objective: Requalify the exact current all-5 effective canary-budget, Paradex edge-floor, and Extended state-stale-aligned surface after the 60m rung cleared the Extended stale trigger but held on a Hyperliquid actual residual during canary-breach response.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `7f822b3d2b7de0dd`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `0b8e659929af490b`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_lighter_orderly_exit_residual_convergence_requal`: `hold`
  objective: Requalify the exact current all-5 effective canary-budget, Paradex edge-floor, Extended state-stale-aligned, and canary-breach flatten-retry surface after the 60m rung held only on a Lighter sub-lot residual at first restore audit.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `7f822b3d2b7de0dd`
  last observed blocker: `paradex_interactive_top_anchor_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `c54d5b537f7cdc25`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_paradex_interactive_top_anchor_gap_requal`: `hold`
  objective: Requalify the exact current all-5 effective canary-budget profile surface if Paradex still under-converts because the live interactive public-top anchor is classified and scored as API-best even when fresh UI truth supplies a more competitive interactive top-level fallback.
  class: `topology`
  hypothesis blocker: `paradex_interactive_top_anchor_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `7f822b3d2b7de0dd`
  last observed blocker: `paradex_edge_floor_queue_loss`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `c54d5b537f7cdc25`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_paradex_post_anchor_bid_edge_floor_requal`: `hold`
  objective: Requalify the exact current all-5 effective canary-budget profile surface after ParaDex top-level interactive anchoring was classified correctly, if ParaDex still under-converts because the post-anchor surface is almost entirely ask-active while bid-side quotes remain dominated by edge-floor suppression.
  class: `topology`
  hypothesis blocker: `paradex_edge_floor_queue_loss`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `913472deaee75e99`
  last observed blocker: `paradex_edge_floor_queue_loss`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `ed8b00d195cc1eca`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_paradex_queue_persistence_grace_requal`: `hold`
  objective: Requalify the exact current all-5 effective canary-budget profile surface after the ParaDex post-anchor bid-edge branch improved bid-active exposure but still left ParaDex fill-zero with supported-replace misses dominated by desired suppression on the same exact surface.
  class: `topology`
  hypothesis blocker: `paradex_edge_floor_queue_loss`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `d0d764f5b4375c19`
  last observed blocker: `paradex_same_side_persistence_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `18e665604eed3635`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_paradex_same_side_snapshot_gap_grace_requal`: `hold`
  objective: Requalify the exact current all-5 effective canary-budget profile surface after the ParaDex queue-persistence branch improved desired-side suppression behavior but left ParaDex fill-zero with same-side supported-replace visibility gaps on the same exact surface.
  class: `topology`
  hypothesis blocker: `paradex_same_side_persistence_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `6e717a0701c6de91`
  last observed blocker: `paradex_queue_position_loss`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `bc6b6e53caba101d`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_paradex_pending_place_grace_requal`: `hold`
  objective: Requalify the exact current all-5 effective canary-budget profile surface after the ParaDex same-side snapshot-gap grace branch stayed operationally clean but bounded telemetry narrowed the remaining ParaDex fill-zero blocker to pre-ack self-cancels of pending post-only MM orders.
  class: `topology`
  hypothesis blocker: `paradex_queue_position_loss`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `246b81d04e29020b`
  last observed blocker: `all5_projected_mm_budget_distribution_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `58c68025bd416f03`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_pending_intent_guard_requal`: `hold`
  objective: Requalify the exact current all-5 effective canary-budget profile surface after the ParaDex pending-place branch stayed operationally clean but the projected MM budget ledger was dominated by stale same-side pending intent stacks on Lighter and Extended.
  class: `topology`
  hypothesis blocker: `all5_projected_mm_budget_distribution_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `0d9bf7da266f2ed0`
  last observed blocker: `paradex_same_side_persistence_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `c32584664c6cd0ca`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_paradex_cancel_ack_seq_requal`: `hold`
  objective: Requalify the exact current all-5 effective canary-budget profile surface after the pending-intent guard branch stayed operationally clean but narrowed the remaining ParaDex blocker to terminal cancel acknowledgements being ignored when their REST seq_no was lower than prior private-order truth.
  class: `topology`
  hypothesis blocker: `paradex_same_side_persistence_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `14260c26309f4ed5`
  last observed blocker: `paradex_edge_floor_queue_loss`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `c89bcb20fa126c89`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_paradex_client_id_cancel_clear_requal`: `hold`
  objective: Requalify the exact current all-5 effective canary-budget profile surface after the ParaDex cancel-ack sequence branch stayed operationally clean, restored shadow-flat, materially reduced cancel churn, and narrowed the remaining blocker to repeated unresolved ParaDex client-id cancel batches against local same-side MM state residue.
  class: `topology`
  hypothesis blocker: `paradex_edge_floor_queue_loss`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `43bd8c4b963f69bc`
  last observed blocker: `startup_readiness_gap`
  last precondition_failed: `false`
  last credit earned: `none`
  last run surface_id: `18fd83448861d88b`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_paradex_client_id_cancel_clear_startup_arm_min_requal`: `hold`
  objective: Requalify the exact current all-5 effective canary-budget plus ParaDex client-id cancel-clear surface after the first live rung held on an early startup stale restart before the ParaDex mechanism could be observed.
  class: `topology`
  hypothesis blocker: `startup_readiness_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `4037e07cdd991d76`
  last observed blocker: `all5_projected_mm_budget_distribution_gap`
  last precondition_failed: `false`
  last run surface_id: `7f87f53175cc304a`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_aster_lighter_pending_cancel_requal`: `hold`
  objective: Requalify the exact current all-5 effective canary-budget profile surface after the startup-arm child proved operationally clean but the projected MM budget ledger stayed dominated by Aster and Lighter pending-order lineage.
  class: `topology`
  hypothesis blocker: `all5_projected_mm_budget_distribution_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `9fb7fccb1b479d29`
  last observed blocker: `all5_projected_mm_budget_distribution_gap`
  last precondition_failed: `false`
  last run surface_id: `78764f277f8fb824`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_aster_lighter_pending_cancel_pre_restore_stability_requal`: `hold`
  objective: Requalify the exact current all-5 effective canary-budget profile plus Aster/Lighter pending-cancel surface after the first 5m canary improved projected MM budget selection but held because one Lighter order reappeared after shadow restore and required a post-cleanup pass.
  class: `topology`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `9fb7fccb1b479d29`
  last observed blocker: `no_data_transport_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `78764f277f8fb824`
- `phase5_all5_fv_reentry_effective_canary_budget_profile_lighter_readonly_ws_requal`: `promoted`
  objective: Requalify the exact current all-5 effective canary-budget profile plus Aster/Lighter pending-cancel and restore-stability surface if the remaining blocker is Lighter public WebSocket availability in live mode, not topology, restore hygiene, or non-Lighter venue behavior.
  class: `topology`
  hypothesis blocker: `no_data_transport_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `microstructure_underconversion`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `975fc9ec53951d4c`
- `phase5_all5_fv_reentry_paradex_edge_floor_queue_hold_requal`: `hold`
  objective: Requalify the exact current all-5 FV surface if Paradex still under-converts because touch-compressed Paradex quotes are mostly suppressed on `edge_below_min` rather than converting into real queue-preserving placements.
  class: `topology`
  hypothesis blocker: `paradex_edge_floor_queue_loss`
  support gate: `shadow_smoke_10m`
  mechanism gate mode: `live_5m`
  planned credit: `minor`
  surface_id: `1c22a667475b71b8`
- `phase5_all5_fv_reentry_paradex_edge_floor_queue_hold_restore_hygiene_requal`: `blocked`
  objective: Requalify the exact current all-5 FV Paradex edge-floor queue-hold surface if the only failed gate is the first audited restore state going dirty before bounded cleanup.
  class: `topology`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `d181f4b47f1aa23b`
- `phase5_reopened_multi_venue_long_soak`: `promoted`
  objective: Requalify the reopened Phase 5 surface with real multi-venue fill participation over a long live soak.
  class: `qualification`
  hypothesis blocker: `all5_projected_mm_budget_distribution_gap`
  surface_id: `7470314d8cef8831`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `00b280143fc3cee5`
- `phase5_reopened_multi_venue_long_soak_projected_inventory_brake_latch_exact_surface_requal`: `promoted`
  objective: Requalify the exact reopened all-five surface after the 2026-04-22 7200s soak held with measurable multi-venue fills but remained strategically suppressed by order-lane backpressure and incomplete closeout evidence on the current surface.
  class: `qualification`
  hypothesis blocker: `all_venue_market_frontier_backpressure_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `7c0dc3ffd84171fb`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `120a41c0c5cef45c`
- `phase5_reopened_multi_venue_long_soak_paradex_ask_edge_floor_requal`: `promoted`
  objective: Requalify the exact reopened all-five surface after the fresh 2h long soak stayed operationally clean but ParaDex remained fill-zero with bid-heavy active quoting and ask-side edge suppression still dominating on the current surface.
  class: `topology`
  hypothesis blocker: `paradex_edge_floor_queue_loss`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `7c0dc3ffd84171fb`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `120a41c0c5cef45c`
- `phase5_reopened_multi_venue_long_soak_paradex_client_id_cancel_clear_requal`: `hold`
  objective: Requalify the exact reopened all-five surface after the fresh 2h long soak stayed operationally clean and restored shadow-flat, but ParaDex remained fill-zero on the inherited ask-floor surface while repeated unresolved client-id cancel batches and local same-side residue continued to suppress durable queue exposure.
  class: `topology`
  hypothesis blocker: `paradex_queue_position_loss`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `7c0dc3ffd84171fb`
  last observed blocker: `microstructure_underconversion`
  last precondition_failed: `false`
  last run surface_id: `120a41c0c5cef45c`
- `phase5_reopened_multi_venue_long_soak_paradex_cancel_pressure_underconversion_requal`: `promoted`
  objective: Requalify the exact reopened all-five surface after the client-id cancel-clear child restored real ParaDex fills and clean rollback, but the final 60m rung still clean-held because ParaDex remained materially underconverted on the unchanged ask-floor surface with elevated cancel pressure, repeated throttling, and high same-side suppression.
  class: `topology`
  hypothesis blocker: `microstructure_underconversion`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `7c0dc3ffd84171fb`
  last precondition_failed: `false`
  last run surface_id: `120a41c0c5cef45c`
- `phase5_reopened_long_soak_lighter_restore_hygiene_requal`: `hold`
  objective: Requalify the exact current reopened all-five long-soak surface after the 2026-04-20 2h run completed and restored shadow-flat, but failed the clean autoscore gate because Lighter still had two open orders and then a -0.01 ETH residual before bounded cleanup converged.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  planned credit: `minor`
  surface_id: `697789c2975864c2`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `none`
  last run surface_id: `c2df85cacb070ecd`
- `phase5_reopened_lighter_canary_flatten_priority_backlog_requal`: `hold`
  objective: Requalify the exact current reopened all-five surface after the restore-hygiene child failed early because a Lighter reduce-only IOC canary flatten sat behind older priority traffic while real venue exposure worsened into hard CanaryLimitBreach.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `697789c2975864c2`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `35e963295b5d6721`
- `phase5_reopened_canary_ineligible_grace_dispatch_requal`: `hold`
  objective: Requalify the exact current reopened all-five priority-arbitration surface after the 20m priority-backlog child proved multi-venue fills, including ParaDex, but held because canary breach response remained dispatch-empty while direct venue truth exceeded the canary residual limits.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `697789c2975864c2`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last run surface_id: `65544774af0da6b7`
- `phase5_reopened_canary_zero_target_dispatch_guard_requal`: `hold`
  objective: Requalify the exact reopened all-five surface after the ineligible-grace child cleared stale grace suppression but the 60m rung still held on an active canary breach response that persisted with zero target coverage after the breached venue set expanded.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `5cb1be0923472be1`
  last precondition_failed: `false`
  last run surface_id: `c1b54d215960d443`
- `phase5_reopened_canary_zero_target_dispatch_guard_hyperliquid_edge_floor_queue_hold_requal`: `promoted`
  objective: Requalify the exact reopened all-five surface after the zero-target canary-response child cleared residual-control failure but the final rung still clean-held on Hyperliquid zero-fill underconversion and elevated would-send-zero risk.
  class: `topology`
  hypothesis blocker: `microstructure_underconversion`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `0314aa19abeaf8f8`
  last precondition_failed: `false`
  last run surface_id: `e5e19866fd269c99`
- `phase5_all5_extended_post_publish_rebootstrap_lead_time_requal`: `promoted`
  objective: Requalify the exact promoted all-five surface after the reopened 2h long soak held on an Extended-only post-publish degraded-stream rebootstrap arriving too close to the existing 3000ms stale-market kill boundary.
  class: `qualification`
  hypothesis blocker: `extended_pre_kill_degraded_rebootstrap_alignment_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `7470314d8cef8831`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `00b280143fc3cee5`
- `phase5_all5_extended_post_publish_rebootstrap_aster_async_observation_requal`: `hold`
  objective: Requalify the exact current all-five Extended post-publish rebootstrap surface after the same-surface 20m rung cleared the Extended stale/rebootstrap blocker but held on a canary response runner stall during Aster flatten retry.
  class: `qualification`
  hypothesis blocker: `runner_freeze_apply_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `c089c447202177ca`
  last observed blocker: `all_venue_market_frontier_backpressure_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `effcdc92c00c7269`
- `phase5_all5_extended_post_publish_rebootstrap_order_channel_backpressure_latch_requal`: `hold`
  objective: Requalify the exact current all-five Extended post-publish/Aster async-observation surface after the final 60m rung completed operationally clean but held on Extended zero fills while emergency single-flight retries saturated the order lane.
  class: `qualification`
  hypothesis blocker: `all_venue_market_frontier_backpressure_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `c089c447202177ca`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `3fdc1c9ff0a19ca4`
- `phase5_all5_extended_post_publish_rebootstrap_order_channel_backpressure_latch_aster_sync_flatten_requal`: `hold`
  objective: Requalify the exact current all-five order-channel-backpressure-latch surface after the final 60m rung proved ChannelFull pressure improved but held on Aster-dominated residual live exposure and a hard CanaryLimitBreach.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `881d0e40b99f17ea`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `58618f3b406d5573`
- `phase5_all5_extended_post_publish_rebootstrap_order_channel_backpressure_latch_aster_sync_account_sync_inventory_recompute_requal`: `hold`
  objective: Requalify the exact current all-five Aster-sync/order-channel-latch surface after the 60m rung proved Lighter eventually flattened, but the same tick retained stale global inventory in canary-limit evaluation and killed after venue truth had already converged.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `881d0e40b99f17ea`
  last observed blocker: `soft_cap_starvation`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `58618f3b406d5573`
- `phase5_all5_extended_post_publish_rebootstrap_order_channel_backpressure_latch_aster_sync_soft_unwind_cancel_covered_reduce_requal`: `hold`
  objective: Requalify the exact current all-five account-sync inventory-recompute surface after the clean 60m rung held on Extended zero fills and high would_send_zero_pct because small Aster/Hyperliquid residuals pinned soft caps and soft-unwind could not reduce a cancel-covered Aster one-lot inside the live window.
  class: `qualification`
  hypothesis blocker: `soft_cap_starvation`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `881d0e40b99f17ea`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `22d9709714455ba2`
- `phase5_all5_extended_post_publish_rebootstrap_order_channel_backpressure_latch_aster_sync_canary_flatten_observation_bridge_requal`: `hold`
  objective: Requalify the exact current all-five soft-unwind cancel-covered surface after the 20m rung held on an Aster canary-flatten convergence kill: Aster filled four maker sells into a -0.04 ETH venue residual, canary response submitted reduce-only IOC exits while cancel_all cleared live orders, but Aster sync-wait timed out or returned the official -2022 ReduceOnly Order is rejected response before account truth converged, triggering CanaryLimitBreach.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `e195107432c6fff2`
  last observed blocker: `microstructure_underconversion`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `f15c895397113e5d`
- `phase5_all5_extended_empty_position_snapshot_freshness_requal`: `hold`
  objective: Requalify the exact current all-five Aster-observation-bridge surface after the final 60m rung was operationally clean but failed promotion solely because Extended produced zero fills while trapped in a stale one-lot reduce-only missing-position loop.
  class: `qualification`
  hypothesis blocker: `microstructure_underconversion`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `e195107432c6fff2`
  last observed blocker: `soft_cap_starvation`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `cabb94f4056a9baa`
- `phase5_all5_aster_post_cancel_soft_unwind_residual_requal`: `hold`
  objective: Requalify the exact current all-five Extended-freshness surface after the final 60m rung proved all-five fills but held because an Aster +0.02 ETH residual survived a soft-unwind cancel_all acknowledgement and starved quoting under the soft governor until closeout cleanup.
  class: `qualification`
  hypothesis blocker: `soft_cap_starvation`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `e195107432c6fff2`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `762457682b6ff1e2`
- `phase5_all5_extended_canary_flatten_async_observation_requal`: `promoted`
  objective: Requalify the exact current all-five surface after the Aster soft-unwind branch proved all-five fills and would_send recovery but held because an Extended -0.04 ETH hard canary venue-cap breach used a synchronous reduce-only IOC flatten response path that timed out before venue/account convergence could be observed.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `e195107432c6fff2`
  last precondition_failed: `false`
  last run surface_id: `970a7f6a96336612`
- `phase5_all5_paradex_canary_flatten_async_observation_requal`: `hold`
  objective: Requalify the exact all-five surface after the fresh reopened long-soak filled all five venues but held because ParaDex hard-breach reduce-only IOC flatten requests timed out on the synchronous response path before account/order truth convergence could be observed.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `e195107432c6fff2`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `0cc9a1d254386561`
- `phase5_all5_aster_lighter_canary_flatten_async_convergence_requal`: `hold`
  objective: Requalify the exact all-five ParaDex-async surface after the 20m rung proved four-venue fills and positive PnL but held because Aster sync-wait canary flatten stalled the runner while Aster and Lighter residuals widened into a hard canary breach.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `b1d4f3a7a1c4e5f8`
  last observed blocker: `stale_restart`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `9db8c474f0f14c29`
- `phase5_all5_emergency_queue_backpressure_requal`: `promoted`
  objective: Requalify the exact all-five Aster/Lighter async surface after the 20m rung proved all-five fills and positive PnL but failed when emergency Lighter flatten retries met saturated order queues and the live loop exited through CanaryLimitBreach.
  class: `qualification`
  hypothesis blocker: `stale_restart`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `697789c2975864c2`
  last precondition_failed: `false`
  last run surface_id: `c2df85cacb070ecd`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_projected_budget_underselection_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the 2h long soak stayed operationally clean but collapsed projected MM quote selection and filled only Hyperliquid.
  class: `qualification`
  hypothesis blocker: `all5_projected_mm_budget_distribution_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `paradex_same_side_persistence_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `12ff5061e0551599`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_paradex_pending_suppression_grace_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the projected-budget underselection requal improved all-five selection but narrowed the remaining fill-zero blocker to ParaDex same-side pending-place cleanup churn.
  class: `topology`
  hypothesis blocker: `paradex_same_side_persistence_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `all5_projected_mm_budget_distribution_gap`
  last precondition_failed: `false`
  last run surface_id: `bd76d738d3a7f697`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_budget_cleanup_on_skip_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the ParaDex pending-suppression branch proved 20m multi-venue fills but the final 60m qualification regressed to Hyperliquid-only fills under projected-budget cleanup starvation on conditional-quote-skip ticks.
  class: `qualification`
  hypothesis blocker: `all5_projected_mm_budget_distribution_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `soft_cap_starvation`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `a26e3868ce6e0d20`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_paradex_cancelled_residual_fallback_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the cleanup-on-skip branch restored four-venue fills but left the final 60m rung soft-governor starved behind a ParaDex residual that was only flattened at closeout cleanup.
  class: `qualification`
  hypothesis blocker: `soft_cap_starvation`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `all5_projected_mm_budget_distribution_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `ca14088c4567d73f`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_projected_budget_cleanup_credit_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the ParaDex cancelled-residual fallback branch completed clean but selected too few live MM candidates because projected-budget selection used raw unmanaged exposure before same-tick suppression cleanup credit.
  class: `qualification`
  hypothesis blocker: `all5_projected_mm_budget_distribution_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `paradex_same_side_persistence_gap`
  last precondition_failed: `false`
  last run surface_id: `ca14088c4567d73f`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_extended_pending_external_id_cleanup_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the held projected-budget cleanup-credit branch exposed stale projected exposure from unacked Extended MM pending orders whose external client IDs were not cleanup-cancel eligible.
  class: `qualification`
  hypothesis blocker: `all5_projected_mm_budget_distribution_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `all5_projected_mm_budget_distribution_gap`
  last precondition_failed: `false`
  last run surface_id: `61e38f8d37bff0d4`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_canary_flatten_pre_cancel_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the Extended externalId cleanup child restored all-five selection but exposed canary breach-response latency under real Hyperliquid/Aster fills.
  class: `qualification`
  hypothesis blocker: `all_venue_market_frontier_backpressure_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `all_venue_market_frontier_backpressure_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `59c34afd33bf6cb0`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_projected_inventory_brake_latch_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the canary-flatten pre-cancel child completed the 60m guard cleanly but produced zero fills while the priority order lane was saturated by duplicate projected inventory-brake cleanup requests.
  class: `qualification`
  hypothesis blocker: `all_venue_market_frontier_backpressure_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `8680d2de139e4171`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_orderly_exit_stop_cleanup_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the projected inventory-brake latch child restored maker traffic and four-venue fills but held on orderly-exit restore hygiene from residual Hyperliquid/Lighter open orders.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `8680d2de139e4171`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_canary_response_sync_flatten_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the stop-before-cleanup child proved restore hygiene but held on canary-breach response state convergence during real Lighter residual flattening.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `soft_cap_starvation`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `fbf99f685965da91`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_cancel_all_lifecycle_convergence_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the sync-flatten child completed operationally clean but spent the final 60m rung under soft-cap starvation from stale live-order lifecycle state after venue-scoped cancel-all acknowledgements.
  class: `qualification`
  hypothesis blocker: `soft_cap_starvation`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `a0ce4ce8d853e12c`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_canary_response_pre_cancel_ioc_sync_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the cancel_all lifecycle branch restored four-venue fills and ParaDex conversion but the 5m canary held on actual residual convergence during canary-breach response.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `a0ce4ce8d853e12c`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_lighter_canary_flatten_transport_priority_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the pre-cancel sync IOC child proved the flatten mechanism but held on Lighter actual residual convergence under same-tick inventory-brake transport contention.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `471ca1ce79aa7101`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_lighter_canary_flatten_ack_latency_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the transport-priority child proved same-tick canary response precedence but held on Lighter reduce-only IOC acknowledgement/account-state latency before hard CanaryLimitBreach.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `d415abb34825f7cf`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_lighter_canary_flatten_async_observation_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the ack-latency child proved the target venue and flatten side were correct but still held because Lighter canary flatten response waits consumed the active response window before account truth converged.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `4aac83aa79ba0680`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_hyperliquid_canary_flatten_async_observation_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the async-observation child proved all-five fill participation and the Lighter async flatten mechanism, but held because Hyperliquid canary flatten still used sync waits that consumed the response window before account/order truth converged.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `runner_freeze_apply_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `f3ca648e8464b67b`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_aster_canary_flatten_async_observation_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the Hyperliquid async-observation child proved all-five fill topology but held because an Aster canary flatten sync wait created a runner response/apply gap immediately before the next Hyperliquid venue breach.
  class: `qualification`
  hypothesis blocker: `runner_freeze_apply_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `b9ea044e44562ef1`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_canary_cancel_scope_quarantine_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the Aster async-observation child proved the Aster canary flatten dispatch mechanism but held because off-target active MM order traffic was not canceled or quarantined during the actual canary-breach response window.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `hyperliquid_canary_response_sync_timeout`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `b9ea044e44562ef1`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_hyperliquid_canary_response_timeout_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after canary cancel-scope quarantine held because Hyperliquid emergency/canary cancel response blocked on sync-control cancel handling and timed out.
  class: `qualification`
  hypothesis blocker: `hyperliquid_canary_response_sync_timeout`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `soft_cap_starvation`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `351da0e7141aab73`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_extended_multi_lot_residual_unwind_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the Hyperliquid canary-response child proved all-five fill participation but held because the final 60m qualification spent most ticks soft-governor starved behind a three-lot Extended residual that only flattened during closeout cleanup.
  class: `qualification`
  hypothesis blocker: `soft_cap_starvation`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `hyperliquid_canary_response_sync_timeout`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `e63588ae397329ca`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_hyperliquid_canary_response_rearm_requal`: `hold`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the Extended multi-lot residual-unwind branch cleared the soft-governor starvation mechanism but the 20m soak held when Hyperliquid canary flatten made partial account-truth progress and was not rearmed after its bounded observation window.
  class: `qualification`
  hypothesis blocker: `hyperliquid_canary_response_sync_timeout`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `b55fa3a4681b76d6`
- `phase5_all5_fv_reentry_lighter_readonly_long_soak_lighter_emergency_ioc_timeout_requal`: `promoted`
  objective: Requalify the exact current all-5 FV Lighter-readonly surface after the Hyperliquid rearm child proved the no-progress guard is correctly conservative, but the 20m soak held because Lighter reduce-only IOC flatten intents remained unacknowledged long enough for the canary window and kill cleanup to time out.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `b55fa3a4681b76d6`
- `phase5_reopened_final_closeout`: `promoted` <- current
  objective: Close the reopened Phase 5 only after multi-venue fill qualification meets the all-5-primary-fill world-class standard.
  surface_id: `19ac6e21020d39d8`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `19ac6e21020d39d8`
- `phase5_all5_paradex_marketdata_stale_resilience_requal`: `hold`
  objective: Re-clear the promoted all-5 topology after the post-closeout 7h PnL observation reopened ParaDex stale-market risk on the exact reopened final overlay.
  class: `clearance`
  hypothesis blocker: `stale_restart`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `hyperliquid_canary_response_sync_timeout`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `0452962462955f34`
- `phase5_all5_hyperliquid_sync_control_ws_post_requal`: `hold`
  objective: Requalify the exact all-5 ParaDex stale-resilience surface after the 5m live rung cleared ParaDex stale_restart but held on Hyperliquid sync-control cancel fallback timing during kill cleanup.
  class: `qualification`
  hypothesis blocker: `hyperliquid_canary_response_sync_timeout`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `eb00bdae5d5eba38`
- `phase5_all5_hyperliquid_canary_flatten_observation_retry_requal`: `hold`
  objective: Requalify the exact all-5 ParaDex stale-resilience and Hyperliquid sync-control ws_post surface after the 20m live rung cleared the sync-control timeout blocker but held on a Hyperliquid actual residual during canary response.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `47ae72c5188d1099`
- `phase5_all5_lighter_inventory_brake_observation_bridge_requal`: `hold`
  objective: Requalify the exact all-5 ParaDex stale-resilience, Hyperliquid ws_post sync-control, and canary observation-retry surface after the 5m child cleared the Hyperliquid residual but held on a Lighter residual during the same canary response family.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `755cd1b326952cac`
- `phase5_all5_observation_progress_baseline_preservation_requal`: `hold`
  objective: Requalify the exact all-5 canary observation surface after the inventory-brake bridge cleared the short and medium gates but the 60m qualification held when an Aster residual improved without converging before the observation deadline.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `755cd1b326952cac`
- `phase5_all5_lighter_canary_observation_priority_requal`: `hold`
  objective: Requalify the exact all-5 canary observation surface after the baseline-preservation child cleared 5m and 20m but the 60m qualification held when an active Lighter canary flatten observation was preceded by same-window inventory-brake emergency traffic.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ea9f3aa917782cd9`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `df671912698969d4`
- `phase5_all5_hyperliquid_pending_place_grace_requal`: `hold`
  objective: Requalify the exact all-5 surface after the Lighter observation-priority child shifted the blocker to Hyperliquid same-side pending maker accumulation.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `d6bbbc9533b459ed`
  last observed blocker: `actual_residual_live_orders`
- `phase5_all5_extended_async_cancel_gap_grace_requal`: `hold`
  objective: Requalify the exact all-5 surface after the Hyperliquid pending-place child shifted the blocker to Extended asynchronous cancel/order/account truth convergence.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `4ff100bdad56090c`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `7a28b3f8ddf83971`
- `phase5_all5_hyperliquid_batch_same_side_exposure_gate_requal`: `hold`
  objective: Requalify the exact all-5 surface after the Extended async-gap child cleared the short rung but the 20m rung held on a Hyperliquid same-side batch-window maker exposure burst.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `13aecbe5cae8a8bd`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `3d0ed8e5f9910ee0`
- `phase5_all5_aster_canary_flatten_sync_wait_requal`: `hold`
  objective: Requalify the exact all-5 surface after the Hyperliquid batch-window dedup child cleared the 20m rung with all five venues filling but the 60m rung held on Aster canary-flatten convergence.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `0b61e0b862cbf7fc`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `06050168e1d4a72d`
- `phase5_all5_lighter_flatten_observation_window_requal`: `hold`
  objective: Requalify the exact all-five surface after the Aster sync-wait child cleared the Aster -0.04 ETH flatten blocker but the 60m rung held on Lighter canary-flatten observation timing.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `102376130aa695d4`
  last observed blocker: `hyperliquid_canary_response_sync_timeout`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `e306c4b9e95de903`
- `phase5_all5_hyperliquid_cancel_all_ws_post_requal`: `hold`
  objective: Requalify the exact all-five Lighter-observation surface after the 20m rung held on a Hyperliquid cancel-all HTTP fallback timeout path during Extended one-lot residual cleanup.
  class: `qualification`
  hypothesis blocker: `hyperliquid_canary_response_sync_timeout`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ba74194e580591d0`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `9f4a84ff864afbe7`
- `phase5_all5_extended_same_side_live_exposure_sum_requal`: `promoted`
  objective: Requalify the exact all-five Hyperliquid cancel_all ws_post surface after the 60m rung held on Extended same-side accepted maker exposure undercount.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `5fb19ecd859329bc`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `90fa623d1947686e`
- `phase5_all5_current_surface_lighter_adverse_selection_markout_requal`: `promoted`
  objective: Requalify the accepted exact-current all-5 surface after the economics attribution gate identified Lighter as the largest venue-local loss source and the current runtime still lacks a native Lighter MM replace path.
  class: `qualification`
  hypothesis blocker: `microstructure_underconversion`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `7470314d8cef8831`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `6d5a4c238e58432d`
- `phase5_all5_current_surface_lighter_post_replace_2h_pnl_observation_clearance`: `promoted`
  objective: Observe the exact accepted all-5 surface for two hours after Lighter native replace support is promoted, without any further economics or topology changes.
  class: `clearance`
  hypothesis blocker: `microstructure_underconversion`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `7470314d8cef8831`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `6d5a4c238e58432d`
- `phase5_all5_current_surface_aster_reduce_only_unwind_fee_guard_requal`: `hold`
  objective: Requalify the exact accepted all-five surface after the post-Lighter observation shifted the dominant economics blocker to Aster reduce-only unwind and cleanup taker fee leakage.
  class: `qualification`
  hypothesis blocker: `soft_cap_starvation`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `fca79569f649e0cf`
  last observed blocker: `microstructure_underconversion`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `501fe83127eedc53`
- `phase5_all5_current_surface_aster_residual_markout_guard_requal`: `hold`
  objective: Requalify the exact accepted all-five surface after the Aster soft-unwind fee guard removed most taker unwind leakage but still allowed adverse Aster residual carry to mark the 7200s run down before closeout cleanup.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `df2caf050ac1e35c`
  last observed blocker: `capital_preservation_residual_markout`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `40d1d5b533854e22`
- `phase5_all5_current_surface_aster_account_freshness_residual_markout_requal`: `hold`
  objective: Requalify the exact current all-five surface after the Aster residual markout guard held because it could not obtain fresh account truth before deciding whether to flatten the remaining one-lot Aster residual.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ce6a6ec4c397016c`
  last observed blocker: `capital_preservation_residual_markout`
  last precondition_failed: `false`
  last credit earned: `none`
  last run surface_id: `f972af01ad5466fb`
- `phase5_all5_current_surface_aster_account_refresh_channel_requal`: `hold`
  objective: Requalify the exact current all-five surface after proving that the Aster residual markout guard can obtain fresh account truth through an authenticated on-demand Aster account-refresh request channel.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `02451af876662390`
  last observed blocker: `capital_preservation_residual_markout`
  last precondition_failed: `false`
  last run surface_id: `d450cfc6e055f023`
- `phase5_all5_current_surface_aster_account_refresh_seq_requal`: `hold`
  objective: Requalify the exact current all-five surface after proving that Aster REST account-refresh snapshots reconcile monotonically with the live account cache and therefore can produce fresh-account-backed residual markout decisions.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `f9781755e16a86a0`
  last observed blocker: `capital_preservation_residual_markout`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `f120ca94e4fec147`
- `phase5_all5_current_surface_aster_stale_residual_force_flat_requal`: `hold`
  objective: Requalify the exact current all-five surface after the Aster account-refresh sequence child proved fresh account-backed residual decisions, but the final 7200s rung still held because a benign long-lived one-lot Aster residual survived to closeout.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `c3f7ba3455a1e04a`
  last observed blocker: `capital_preservation_residual_markout`
  last precondition_failed: `false`
  last run surface_id: `71f3f410f23845b2`
- `phase5_all5_current_surface_aster_stale_residual_force_flat_long_opportunity_requal`: `hold`
  objective: Requalify the exact current all-five Aster stale-residual force-flat surface after the 20m rung proved clean fill-positive operation but did not produce an eligible stale force-flat opportunity.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `d12feb845dfca5e5`
  last observed blocker: `stale_restart`
  last precondition_failed: `false`
  last run surface_id: `ea129c56d0618fd2`
- `phase5_all5_current_surface_aster_stale_residual_force_flat_long_opportunity_overlay_repair_requal`: `hold`
  objective: Requalify the exact current all-five Aster stale-residual force-flat opportunity surface after the first long-opportunity child held on an overlay inheritance regression that removed Extended stale/rebootstrap controls.
  class: `qualification`
  hypothesis blocker: `stale_restart`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `55427875b7c5b552`
  last observed blocker: `extended_pre_kill_degraded_rebootstrap_alignment_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `3201bceccae332b2`
- `phase5_all5_current_surface_extended_pre_kill_rebootstrap_lead_time_tighten_after_aster_force_flat_requal`: `hold`
  objective: Requalify the exact current all-five Aster force-flat opportunity surface after the repaired overlay still held on an Extended degraded-stream rebootstrap that fired too close to the 3000ms runner stale boundary.
  class: `qualification`
  hypothesis blocker: `extended_pre_kill_degraded_rebootstrap_alignment_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `b6cb8203512454d0`
  last observed blocker: `extended_degraded_stream_rebootstrap_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `9ed48288a2b00ba1`
- `phase5_all5_current_surface_extended_stale_churn_healthy_reset_after_pre_kill_requal`: `hold`
  objective: Requalify the exact current all-five Aster force-flat opportunity surface after the narrower Extended lead-time child proved 1200ms/1800ms fallback timing but held because stale-watchdog churn escalation slept 4000ms inside the 3000ms runner stale boundary.
  class: `qualification`
  hypothesis blocker: `extended_degraded_stream_rebootstrap_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `f2b6d17047345711`
  last observed blocker: `extended_degraded_stream_rebootstrap_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `61fe0679b377ef5d`
- `phase5_all5_current_surface_extended_stale_churn_budget5_after_reset_requal`: `hold`
  objective: Requalify the exact current all-five Aster force-flat opportunity surface after the Extended healthy-reset child materially extended the 1200s rung from 86s to 673s but still held when degraded-stream rebootstrap churn escalated to a 4000ms reconnect sleep inside the unchanged 3000ms runner stale boundary.
  class: `qualification`
  hypothesis blocker: `extended_degraded_stream_rebootstrap_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `f1e19c364c754b5e`
  last observed blocker: `extended_pre_kill_degraded_rebootstrap_alignment_gap`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `461f28e1b619f3c9`
- `phase5_all5_current_surface_extended_stale_churn_budget7_after_budget5_requal`: `hold`
  objective: Requalify the exact current all-five surface after the budget-5 child proved all-five fill participation and long-run Aster account/fee guard health, but still held when Extended degraded-stream rebootstrap churn reached count_window 7 and emitted a 4000ms reconnect sleep inside the unchanged 3000ms stale boundary.
  class: `qualification`
  hypothesis blocker: `extended_degraded_stream_rebootstrap_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `c61c9754a45efabe`
  last observed blocker: `capital_preservation_residual_markout`
  last precondition_failed: `false`
  last run surface_id: `aa73a38a7be6875d`
- `phase5_all5_current_surface_aster_reduce_only_inventory_brake_fee_guard_requal`: `hold`
  objective: Requalify the exact current all-five budget-7 surface after the 7200s rung proved operational cleanliness and all-five fills, but held on residual Aster reduce-only economics because inventory-brake taker flow remained outside the previously promoted soft-unwind fee guard path.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `8c44849d98d247e3`
  last observed blocker: `capital_preservation_residual_markout`
  last precondition_failed: `false`
  last run surface_id: `3bf0f8a9c2548c44`
- `phase5_all5_current_surface_aster_same_side_live_exposure_sum_requal`: `rolled_back`
  objective: Requalify the exact current all-five Aster inventory-brake fee-guard surface after the 7200s rung proved all-five topology and operational cleanliness, but held because stacked same-side Aster live exposure still reached residual states that required fee-bearing reduce-only IOC cleanup.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `2d23a9c952cd6384`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `1465ff782d3747be`
- `phase5_all5_current_surface_aster_same_side_live_exposure_restore_hygiene_requal`: `hold`
  objective: Requalify the exact same all-five Aster same-side live exposure surface after the interrupted 7200s child rolled back on restore hygiene rather than on a cleanly attributable strategy or venue-topology failure.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `2d23a9c952cd6384`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last credit earned: `none`
  last run surface_id: `d1864116690ce318`
- `phase5_all5_current_surface_extended_terminal_residual_convergence_requal`: `hold`
  objective: Requalify the exact all-five current surface with only an Extended terminal-residual convergence guard so the observed -0.02 ETH Extended pre-restore dirty audit can be resolved in-run instead of by post-window cleanup.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `fd983e9efd551adc`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `1ada2456de5cfc28`
- `phase5_all5_current_surface_aster_terminal_full_target_convergence_requal`: `hold`
  objective: Requalify the exact all-five current surface with only an env-gated Aster terminal full-target soft-unwind path so the observed +0.01 ETH Aster terminal residual and cleanup cost can be resolved in-run instead of by post-window cleanup.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `9e2a3c7e9015d5e5`
  last observed blocker: `capital_preservation_residual_markout`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `a9385c5735703e1d`
- `phase5_all5_current_surface_aster_short_force_flat_convergence_requal`: `hold`
  objective: Requalify the exact all-five current surface with only a shorter Aster residual markout force-flat age so benign one-lot Aster terminal residuals flatten inside the guarded run window instead of during post-window cleanup.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `c75c7307c4ee60ef`
  last observed blocker: `capital_preservation_residual_markout`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `b44a9a45d7a706c8`
- `phase5_all5_current_surface_extended_one_lot_state_fallback_step_requal`: `hold`
  objective: Requalify the exact all-five Aster-safe current surface with only an env-gated Extended one-lot state-fallback step so no-fresh-account terminal residual cleanup cannot submit reduce-only IOC size larger than venue-truth position.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `f471adc9893e072e`
  last observed blocker: `capital_preservation_residual_markout`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `f4943e494189b28a`
- `phase5_all5_current_surface_aster_markout_min_adverse_025_requal`: `hold`
  objective: Requalify the exact all-five current surface with only a narrower Aster residual markout adverse threshold so small adverse Aster one-lot residuals can flatten inside the guarded run window instead of waiting for post-window cleanup, while preserving the Extended one-lot fallback clamp.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `977998b82ff22f6b`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `none`
  last run surface_id: `4f71b7321d507b9d`
- `phase5_all5_current_surface_terminal_exit_live_order_drain_requal`: `hold`
  objective: Requalify the exact all-five current surface with an env-gated terminal-exit quiesce/drain window so live MM orders and one-lot residuals converge before the first pre-restore venue audit, instead of relying on post-window cleanup.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `c004cb8a49dc73bc`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last run surface_id: `b4af7363c8d64091`
- `phase5_all5_current_surface_terminal_exit_service_visible_signal_requal`: `hold`
  objective: Requalify the exact all-five terminal-exit surface after the 1200s rung proved all-five fills and flat final inventory but held because the service never observed the `/tmp` terminal-exit signal under systemd `PrivateTmp=yes`, leaving one Extended live open order for first pre-restore audit.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `7079a24b0a474aba`
  last observed blocker: `extended_pre_kill_degraded_rebootstrap_alignment_gap`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `0be1adb1a04a1a42`
- `phase5_all5_current_surface_extended_degraded_rebootstrap_sleep_cap_requal`: `hold`
  objective: Requalify the exact all-five service-visible terminal-exit surface after the latest 7200s attempt held before the terminal window because Extended degraded rebootstrap churn escalated to an 8000ms reconnect sleep inside the unchanged 3000ms runner stale boundary.
  class: `qualification`
  hypothesis blocker: `extended_pre_kill_degraded_rebootstrap_alignment_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `f8bdc673563f219d`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `dbe99923f7ab3e17`
- `phase5_all5_current_surface_aster_terminal_account_refresh_requal`: `hold`
  objective: Requalify the exact all-five sleep-capped terminal-exit surface after the 300s rung proved the Extended degraded rebootstrap cap and service-visible terminal signal, but first pre-restore direct venue audit found a hidden Aster residual that runtime markout telemetry never observed.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `8e23e0467e0d21b6`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last run surface_id: `7a1c69058e32c2a2`
- `phase5_all5_current_surface_terminal_ext_pdx_account_cancel_drain_requal`: `hold`
  objective: Requalify the exact all-five Aster-terminal-refresh surface after the 1200s rung proved Aster terminal account refresh but held because first pre-restore direct venue audit found non-Aster terminal residuals on Extended and Paradex.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `e8c06443f93e2582`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last run surface_id: `b62301ceb7e07ca7`
- `phase5_all5_current_surface_aster_terminal_sub_lot_reduce_requal`: `hold`
  objective: Requalify the exact all-five terminal Extended/Paradex cancel-drain surface after the 300s rung proved the non-Aster terminal path but held because first pre-restore direct venue audit found an Aster sub-lot residual with zero open orders.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `10af34bcc33cc7ab`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last run surface_id: `8497a73856f1ceca`
- `phase5_all5_current_surface_aster_terminal_quiesce_explicit_cancel_retry_requal`: `hold`
  objective: Requalify the exact all-five Aster terminal sub-lot surface after the 7200s rung proved flat inventory and all-five fill evidence but held because first pre-restore direct venue audit found one stale Aster open order despite zero Aster position.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `ee7c1f291ea07777`
  last precondition_failed: `false`
  last run surface_id: `a3c0cf1dce0e4cc7`
- `phase5_all5_current_surface_extended_native_replace_fill_conversion_requal`: `blocked`
  objective: Keep the next exact all-five Extended native replace child blocked behind the active Aster terminal quiesce explicit-cancel retry child so Extended fill conversion can be exercised only after both Aster terminal sub-lot residual convergence and terminal open-order drain are promotion-clean.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `a2993dddb27afd93`
- `phase5_all5_current_surface_extended_queue_persistence_aster_bid_edge_requal`: `hold`
  objective: Requalify the exact all-five current surface after the Aster terminal quiesce explicit-cancel retry child completed operationally clean, clearing the two remaining promotion blockers from the latest 7200s rung: Extended zero fills and the measured Aster bid-side maker-fee drag that made final total PnL negative.
  class: `topology`
  hypothesis blocker: `microstructure_underconversion`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `7744272f8c7d91f3`
  last observed blocker: `microstructure_underconversion`
  last precondition_failed: `false`
  last run surface_id: `b59c5beb892cefdf`
- `phase5_all5_current_surface_hyperliquid_terminal_sub_lot_residual_requal`: `hold`
  objective: Requalify the exact all-five current surface after the Extended queue-persistence plus Aster bid-edge child achieved all-five live fills but held because the first pre-restore direct venue audit found a Hyperliquid sub-lot residual with zero open orders.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `d7e5fb1d74fc3af6`
  last observed blocker: `actual_residual_live_orders`
  last precondition_failed: `false`
  last run surface_id: `0cf29e9406f23f58`
- `phase5_all5_current_surface_terminal_account_refresh_router_requal`: `hold`
  objective: Requalify the exact all-five current surface after the Hyperliquid terminal sub-lot child proved Hyperliquid flat but held because first pre-restore direct venue audit found an Extended +0.01 ETH residual with zero open orders that runtime account state had not observed.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `745541ee13d7899f`
  last observed blocker: `extended_pre_kill_degraded_rebootstrap_alignment_gap`
  last precondition_failed: `false`
  last run surface_id: `ff08070af9bb4e80`
- `phase5_all5_current_surface_runner_realtime_missed_tick_burst_requal`: `promoted`
  objective: Requalify the exact all-five current surface after the terminal account-refresh router child proved all-five fill conversion but held when a delayed realtime runner interval replayed missed ticks as a stale-hygiene burst before Extended recovery could be observed.
  class: `qualification`
  hypothesis blocker: `extended_pre_kill_degraded_rebootstrap_alignment_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `fa6c11da64aacc42`
  last precondition_failed: `false`
  last run surface_id: `8faedf4a9701d5e0`
- `phase5_reopened_multi_venue_long_soak_runner_realtime_requal`: `hold`
  objective: Complete the reopened multi-venue Phase 5 long soak on the exact promoted all-five runner-realtime surface and preserve exact pre/post exchange account balances for final closeout.
  class: `qualification`
  hypothesis blocker: `extended_pre_kill_degraded_rebootstrap_alignment_gap`
  support gate: `shadow_smoke_10m`
  planned credit: `major`
  surface_id: `fa6c11da64aacc42`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last run surface_id: `8faedf4a9701d5e0`
- `phase5_reopened_terminal_stale_order_residual_requal`: `hold`
  objective: Requalify the reopened all-five Phase 5 surface after the 7200s long soak held because terminal quiesce left Hyperliquid and Lighter residuals that required pre-restore cleanup despite a clean guard window.
  class: `qualification`
  hypothesis blocker: `actual_residual_live_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `minor`
  surface_id: `c09211cbe3595829`
  last observed blocker: `capital_preservation_residual_markout`
  last precondition_failed: `false`
- `phase5_reopened_wallet_econ_edge_requal`: `hold`
  objective: Requalify the reopened all-five Phase 5 surface after terminal hygiene was proven clean but wallet balance economics failed the final 7200s promotion gate.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `major`
  surface_id: `76907ff39032f317`
  last observed blocker: `capital_preservation_residual_markout`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `ffb2c470edfcd463`
- `phase5_reopened_wallet_econ_full_cost_requal`: `hold`
  objective: Requalify the reopened all-five Phase 5 surface after the wallet-economics half-cost child stayed operationally clean but failed the 7200s economics and combined balance promotion gates.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `major`
  surface_id: `1ddf3422a3138625`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `111531975f916424`
- `phase5_reopened_wallet_econ_full_cost_terminal_controls_requal`: `hold`
  objective: Requalify the full-cost wallet-economics child with the parent terminal signal, stale-state, and venue recovery controls restored exactly.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `major`
  surface_id: `715c98bbd58ebada`
  last observed blocker: `capital_preservation_residual_markout`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `931a4bfaf682d9a6`
- `phase5_reopened_wallet_econ_full_cost_aster_bid_edge_016_requal`: `hold`
  objective: Requalify the corrected full-cost wallet-economics surface with only Aster bid edge floor tightened after the clean 7200s final rung missed economics because Aster owned the entire modeled hedge-cost loss.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `major`
  surface_id: `3915ab59c7cf11cc`
  last observed blocker: `capital_preservation_residual_markout`
  last precondition_failed: `false`
  last credit earned: `minor`
  last run surface_id: `f34870ff704188c7`
- `phase5_reopened_wallet_econ_full_cost_hedge_attribution_gate_requal`: `hold`
  objective: Requalify the same corrected full-cost Aster-bid-edge surface after hardening the economics scorer so unattributed hedge execution cost cannot be treated as zero-cost promotion economics.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `major`
  surface_id: `3915ab59c7cf11cc`
  last observed blocker: `capital_preservation_residual_markout`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `f34870ff704188c7`
- `phase5_reopened_wallet_econ_full_cost_hedge_source_truth_requal`: `hold`
  objective: Requalify the same conservative wallet-economics surface after repairing runtime hedge-source truth so hedge records and hedge fills preserve the source market-making decision/fill metadata needed for venue-local economics.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `major`
  surface_id: `3915ab59c7cf11cc`
  last observed blocker: `exact_one_lot_residual_no_orders`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `f470db3c6feeb66f`
- `phase5_reopened_wallet_econ_full_cost_extended_terminal_exact_lot_requal`: `hold`
  objective: Requalify the same conservative wallet-economics/source-truth surface after the 300s source-truth canary held because terminal quiesce left an Extended exact one-lot residual with zero venue open orders.
  class: `qualification`
  hypothesis blocker: `exact_one_lot_residual_no_orders`
  support gate: `shadow_smoke_10m`
  planned credit: `major`
  surface_id: `3915ab59c7cf11cc`
  last observed blocker: `microstructure_underconversion`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `6630687e7bc06344`
- `phase5_reopened_wallet_econ_full_cost_residual_source_mm_fill_gate_requal`: `hold`
  objective: Requalify the same conservative wallet-economics surface after separating residual hedge source truth from market-making fill-conversion evidence.
  class: `qualification`
  hypothesis blocker: `microstructure_underconversion`
  support gate: `shadow_smoke_10m`
  planned credit: `major`
  surface_id: `3915ab59c7cf11cc`
  last observed blocker: `restore_hygiene`
  last precondition_failed: `false`
  last run surface_id: `1c448ee709419814`
- `phase5_reopened_wallet_econ_full_cost_extended_account_truth_requal`: `hold`
  objective: Requalify the same wallet-economics surface after the 300s purpose-aware fill gate rung held because Extended venue-side account truth was not observed by terminal quiesce before stop.
  class: `qualification`
  hypothesis blocker: `restore_hygiene`
  support gate: `shadow_smoke_10m`
  planned credit: `major`
  surface_id: `441529d7fac3f18e`
  last observed blocker: `capital_preservation_residual_markout`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `b554afdb42647184`
- `phase5_reopened_wallet_econ_full_cost_ext_pdx_bid_edge_requal`: `hold`
  objective: Requalify the reopened all-five wallet-economics surface after the final 7200s rung proved clean restore and all-five market-making fills but held on capital-preservation economics.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `major`
  surface_id: `0378861c46525433`
  last observed blocker: `capital_preservation_residual_markout`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `19880f6294f71ee8`
- `phase5_reopened_wallet_econ_full_cost_ext_harden_pdx_reentry_requal`: `hold`
  objective: Requalify the reopened all-five wallet-economics surface after the Extended/Paradex bid-edge child restored capital drift but lost Hyperliquid and Paradex market-making fill evidence.
  class: `qualification`
  hypothesis blocker: `capital_preservation_residual_markout`
  support gate: `shadow_smoke_10m`
  planned credit: `major`
  surface_id: `d4792dbf37283dcc`
  last observed blocker: `microstructure_underconversion`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `d4792dbf37283dcc`
- `phase5_reopened_wallet_econ_full_cost_dynamic_size_02_requal`: `hold`
  objective: Requalify the reopened all-five wallet-economics surface with strategy-valid dynamic quote sizing headroom after the prior 7200s rung held on zero Hyperliquid/Paradex MM fill evidence and a small cost-inclusive economics miss.
  class: `qualification`
  hypothesis blocker: `microstructure_underconversion`
  support gate: `shadow_smoke_10m`
  planned credit: `major`
  surface_id: `19ac6e21020d39d8`
  last observed blocker: `microstructure_underconversion`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `19ac6e21020d39d8`
- `phase5_reopened_wallet_econ_full_cost_extended_ask_activation_requal`: `hold`
  objective: Requalify the reopened all-five wallet-economics surface by restoring Extended ask-side quote activation without reopening the losing Extended bid-side fills that drove the previous cost-inclusive economics failure.
  class: `qualification`
  hypothesis blocker: `microstructure_underconversion`
  support gate: `shadow_smoke_10m`
  planned credit: `major`
  surface_id: `56ea3076a0010943`
  last observed blocker: `microstructure_underconversion`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `56ea3076a0010943`
- `phase5_reopened_wallet_econ_full_cost_opportunity_adjusted_gate_requal`: `promoted`
  objective: Requalify the reopened full-cost wallet-economics surface by replacing the raw all-venue market-making fill gate with an opportunity-adjusted participation gate and authoritative five-account balance-delta PnL.
  class: `qualification`
  hypothesis blocker: `microstructure_underconversion`
  support gate: `shadow_smoke_10m`
  planned credit: `major`
  surface_id: `19ac6e21020d39d8`
  last precondition_failed: `false`
  last credit earned: `major`
  last run surface_id: `19ac6e21020d39d8`

## Parallel Support Tracks

- `phase5_extended_rescue_shadow`: `Recover Extended in shadow/local only until it earns re-entry.`
- `phase5_tooling_forensics_hardening`: `Keep analyzer, scorecard, and artifact normalization trustworthy while mainline progresses.`
- `phase5_topology_readiness_audit`: `Compute an explicit all5 topology capability matrix beside the mainline so perfect-topology readiness is visible without scraping markdown.`
- `phase5_blocker_shadow_lab`: `Hold a blocker-local shadow experiment lane that can be attached to the current mainline without opening a second live-affecting hypothesis.`
- `phase5_extended_freeze_shadow_lab`: `Hold a blocker-local shadow experiment lane for the current-surface Extended freeze-path branch without opening a second live-affecting hypothesis.`
- `phase5_extended_transport_gap_shadow_lab`: `Hold a blocker-local shadow experiment lane for the current-surface Extended transport-gap watchdog branch without opening a second live-affecting hypothesis.`
- `phase5_aster_bridge_wait_shadow_lab`: `Hold a blocker-local shadow experiment lane for the current-surface Aster bridge-wait branch without opening a second live-affecting hypothesis.`

## Active Blocker

- `none`: latest serialized-mainline child `phase5_reopened_final_closeout` promoted at `2026-05-01T01:44:24Z`
