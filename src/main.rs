// src/main.rs
//
// Simple driver for the Paraphina MM engine:
//  - builds Config + GlobalState,
//  - runs a few synthetic ticks with dummy orderbooks,
//  - computes fair value, vols, quotes, hedge plan,
//  - applies hedge fills into the state,
//  - recomputes inventory / basis / risk as per the whitepaper.

mod config;
mod engine;
mod hedge;
mod mm;
mod state;
mod toxicity;
mod types;

use crate::config::Config;
use crate::engine::Engine;
use crate::hedge::{compute_hedge_plan, hedge_plan_to_order_intents};
use crate::mm::{compute_mm_quotes, mm_quotes_to_order_intents};
use crate::state::GlobalState;

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

    // For now, run a small fixed number of synthetic ticks.
    let num_ticks = 3;

    for tick in 0..num_ticks {
        let now_ms: i64 = (tick as i64) * 1_000; // 1 second per tick

        println!("\n================ Tick {} =================", tick);

        // 1) Seed synthetic mids / spreads / depth.
        engine.seed_dummy_mids(&mut state, now_ms);

        // 2) Main engine tick:
        //    - fair value + Kalman (Section 5),
        //    - volatility EWMAs + control scalars (Section 6),
        //    - inventory & basis (Section 8),
        //    - toxicity + venue status (Section 7),
        //    - risk regime & kill switch (Section 14).
        engine.main_tick(&mut state, now_ms);

        // ---------- Global snapshot BEFORE hedges ----------
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

        // ---------- MM quotes ----------
        let mm_quotes = compute_mm_quotes(&cfg, &state);

        println!("\nPer-venue quotes:");
        for q in &mm_quotes {
            let bid_pair = q.bid.as_ref().map(|b| (b.price, b.size));
            let ask_pair = q.ask.as_ref().map(|a| (a.price, a.size));
            println!(
                "  {:>9}: bid={:?}, ask={:?}",
                q.venue_id, bid_pair, ask_pair
            );
        }

        let mm_intents = mm_quotes_to_order_intents(&mm_quotes);

        // ---------- Hedge plan ----------
        let hedge_plan_opt = compute_hedge_plan(&cfg, &state);

        if let Some(plan) = &hedge_plan_opt {
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

        let hedge_intents = hedge_plan_opt
            .as_ref()
            .map(hedge_plan_to_order_intents)
            .unwrap_or_default();

        // ---------- Combined order intents ----------
        println!("\nOrder intents (abstract):");
        for intent in mm_intents.iter().chain(hedge_intents.iter()) {
            println!(
                "  {:>9} {:?} {:.4} @ {:.4} ({:?})",
                intent.venue_id, intent.side, intent.size, intent.price, intent.purpose
            );
        }

        // ---------- Apply hedge fills into state ----------
        //
        // Toy fill model:
        //   - assume all **hedge** intents are fully filled at their est. price
        //   - use venue taker fee bps
        //   - MM intents are not filled yet (later we can add a Poisson fill model).
        for intent in &hedge_intents {
            let vcfg = &cfg.venues[intent.venue_index];
            let fee_bps = vcfg.taker_fee_bps;

            state.apply_perp_fill(
                intent.venue_index,
                intent.side,
                intent.size,
                intent.price,
                fee_bps,
            );
        }

        // Recompute inventory, basis and risk *after* those hedge fills
        // so q_t, delta, basis and risk regime reflect the new positions.
        engine.recompute_after_fills(&mut state);

        println!("\nAfter hedge fills:");
        println!("  Global q_t (TAO): {:.4}", state.q_global_tao);
        println!("  Dollar delta (USD): {:.4}", state.dollar_delta_usd);
        println!("  Basis exposure (USD): {:.4}", state.basis_usd);
        println!("  Risk regime: {:?}", state.risk_regime);
    }
}
