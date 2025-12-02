// src/main.rs
//
// Simple driver for the Paraphina MM engine:
//  - builds Config + GlobalState,
//  - runs synthetic ticks with dummy orderbooks,
//  - computes fair value, vols, risk regime, toxicity,
//  - runs the pure strategy policy (MM + hedge),
//  - passes abstract intents to SimGateway for synthetic fills,
//  - recomputes inventory / basis / PnL / risk after fills.

mod config;
mod engine;
mod gateway;
mod hedge;
mod mm;
mod state;
mod strategy;
mod toxicity;
mod types;

use crate::config::Config;
use crate::engine::Engine;
use crate::gateway::SimGateway;
use crate::state::GlobalState;
use crate::strategy::compute_strategy_step;
use crate::types::Side;

fn main() {
    // ---------- Bootstrap config + state ----------
    let cfg = Config::default();

    println!(
        "Paraphina MM starting with config version: {}",
        cfg.version
    );
    println!("Configured venues: {}", cfg.venues.len());

    let mut state = GlobalState::new(&cfg);
    println!("Initial risk regime: {:?}\n", state.risk_regime);

    let engine = Engine::new(&cfg);
    let gateway = SimGateway::new();

    // For now, run a fixed number of synthetic ticks.
    let num_ticks: usize = 50;

    for tick in 0..num_ticks {
        let now_ms: i64 = (tick as i64) * 1_000; // 1 second per tick

        println!("\n================ Tick {} =================", tick);

        // 1) Seed synthetic mids / spreads / depth.
        engine.seed_dummy_mids(&mut state, now_ms);

        // 2) Main engine tick:
        //    - fair value + "Kalman-lite" (Section 5),
        //    - volatility EWMAs + control scalars (Section 6),
        //    - toxicity / venue health (Section 7),
        //    - risk regime & kill switch (Section 14).
        engine.main_tick(&mut state, now_ms);

        // --- TEMP: inject synthetic inventory on tick 0 so hedge engine has work to do ---
        if tick == 0 {
            if let Some(s_t) = state.fair_value {
                // Use the first venue ("extended") for the synthetic position.
                let vcfg0 = &cfg.venues[0];

                // Open a +25 TAO long on extended at ~fair value with taker fees.
                state.apply_perp_fill(
                    0,                  // venue_index for "extended"
                    Side::Buy,          // open a long
                    25.0,               // size in TAO
                    s_t,                // price ≈ fair value
                    vcfg0.taker_fee_bps,
                );

                // Recompute global inventory, delta, basis, risk, etc.
                engine.recompute_after_fills(&mut state);

                println!(
                    "\n[debug] Injected synthetic +25 TAO long on {} -> q_t = {:.4}, delta = {:.2} USD",
                    vcfg0.id, state.q_global_tao, state.dollar_delta_usd
                );
            } else {
                println!(
                    "\n[debug] Skipped synthetic position injection – fair value not available on tick 0"
                );
            }
        }

        // ---------- Global snapshot (post-engine, pre-strategy) ----------
        let s_t = state.fair_value.unwrap_or(0.0);

        println!("Fair value S_t: {:.4}", s_t);
        println!("Sigma_eff: {:.6}", state.sigma_eff);
        println!("Vol ratio (clipped): {:.4}", state.vol_ratio_clipped);
        println!("Spread_mult: {:.4}", state.spread_mult);
        println!("Size_mult: {:.4}", state.size_mult);
        println!("Band_mult: {:.4}", state.band_mult);
        println!("Global q_t (TAO): {:.4}", state.q_global_tao);
        println!("Dollar delta (USD): {:.4}", state.dollar_delta_usd);
        println!("Basis exposure (USD): {:.4}", state.basis_usd);
        println!("Basis gross (USD): {:.4}", state.basis_gross_usd);
        println!("Delta limit (USD): {:.4}", state.delta_limit_usd);
        println!(
            "Basis warn / hard (USD): {:.2} / {:.2}",
            state.basis_limit_warn_usd, state.basis_limit_hard_usd
        );
        println!(
            "Daily PnL (realised): {:.4}",
            state.daily_realised_pnl
        );
        println!(
            "Daily PnL (unrealised): {:.4}",
            state.daily_unrealised_pnl
        );
        println!("Daily PnL total: {:.4}", state.daily_pnl_total);
        println!("Risk regime after tick: {:?}", state.risk_regime);
        println!("Kill switch: {}", state.kill_switch);

        // ---------- Per-venue toxicity ----------
        println!("\nPer-venue toxicity & status:");
        for (idx, v) in state.venues.iter().enumerate() {
            let id = &cfg.venues[idx].id;
            println!(
                "  {:>9}: toxicity={:.3}, status={:?}",
                id, v.toxicity, v.status
            );
        }

        // ---------- Strategy decision (MM + hedge) ----------
        let step = compute_strategy_step(&cfg, &state);

        println!("\nPer-venue quotes:");
        for q in &step.mm_quotes {
            let bid_pair = q.bid.as_ref().map(|b| (b.price, b.size));
            let ask_pair = q.ask.as_ref().map(|a| (a.price, a.size));
            println!(
                "  {:>9}: bid={:?}, ask={:?}",
                q.venue_id, bid_pair, ask_pair
            );
        }

        if let Some(plan) = &step.hedge_plan {
            println!("\nHedge plan:");
            println!("  Desired ΔH (TAO): {:.4}", plan.desired_delta);
            for alloc in &plan.allocations {
                println!(
                    "    -> venue {:>9}: {:?} {:.4} @ ~{:.4}",
                    alloc.venue_id, alloc.side, alloc.size, alloc.est_price
                );
            }
        } else {
            println!("\nHedge plan: none");
        }

        // ---------- Snapshot BEFORE MM + hedge fills ----------
        println!("\nBefore MM + hedge fills:");
        println!("  Global q_t (TAO): {:.4}", state.q_global_tao);
        println!("  Dollar delta (USD): {:.4}", state.dollar_delta_usd);
        println!("  Basis exposure (USD): {:.4}", state.basis_usd);
        println!(
            "  Daily PnL (realised / unrealised / total): {:.4} / {:.4} / {:.4}",
            state.daily_realised_pnl,
            state.daily_unrealised_pnl,
            state.daily_pnl_total,
        );

        // ---------- Combined order intents ----------
        println!("\nOrder intents (abstract):");
        for intent in step
            .mm_intents
            .iter()
            .chain(step.hedge_intents.iter())
        {
            println!(
                "  {:>9} {:?} {:.4} @ {:.4} ({:?})",
                intent.venue_id, intent.side, intent.size, intent.price, intent.purpose
            );
        }

        // Build a single batch for the gateway.
        let mut all_intents = Vec::new();
        all_intents.extend(step.mm_intents.iter().cloned());
        all_intents.extend(step.hedge_intents.iter().cloned());

        // ---------- Apply synthetic fills via SimGateway ----------
        gateway.apply_fills(&cfg, &mut state, &all_intents);

        // Recompute global inventory / delta / basis / PnL / risk from the
        // updated per-venue positions after all synthetic fills.
        engine.recompute_after_fills(&mut state);

        // ---------- Snapshot AFTER MM + hedge fills ----------
        println!("\nAfter MM + hedge fills:");
        println!("  Global q_t (TAO): {:.4}", state.q_global_tao);
        println!("  Dollar delta (USD): {:.4}", state.dollar_delta_usd);
        println!("  Basis exposure (USD): {:.4}", state.basis_usd);
        println!(
            "  Daily PnL (realised / unrealised / total): {:.4} / {:.4} / {:.4}",
            state.daily_realised_pnl,
            state.daily_unrealised_pnl,
            state.daily_pnl_total,
        );
        println!("  Risk regime: {:?}", state.risk_regime);
    }
}
