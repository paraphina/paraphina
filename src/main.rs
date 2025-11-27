// src/main.rs

mod config;
mod state;
mod types;
mod engine;
mod mm;
mod hedge;
mod toxicity;

use crate::config::Config;
use crate::engine::Engine;
use crate::hedge::{compute_hedge_plan, hedge_plan_to_order_intents};
use crate::mm::{compute_mm_quotes, mm_quotes_to_order_intents};
use crate::state::GlobalState;
use crate::types::{OrderIntent, Side};

use std::time::{SystemTime, UNIX_EPOCH};

fn now_ms() -> i64 {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    dur.as_millis() as i64
}

/// Very small demo loop: how often the main loop ticks (fake time).
const MAIN_LOOP_INTERVAL_MS: i64 = 1_000;
/// How many ticks to run before exiting.
const NUM_TICKS: usize = 3;

fn main() {
    let cfg = Config::default();
    let mut state = GlobalState::new(&cfg);

    println!(
        "Paraphina MM starting with config version: {}",
        cfg.version
    );
    println!("Configured venues: {}", cfg.venues.len());
    println!("Initial risk regime: {:?}", state.risk_regime);

    // -----------------------------------------------
    // SIMULATED INITIAL TRADE (perp fill semantics)
    // -----------------------------------------------
    //
    // Instead of manually poking position_tao, we apply a single
    // synthetic perp fill on Extended (venue 0):
    //
    //   Buy 40 TAO @ 250 USDT, paying taker fees.
    //
    // This exercises GlobalState::apply_perp_fill, which updates:
    //   - position_tao
    //   - VWAP
    //   - realised trade PnL (0 here; it's an opening trade)
    //   - fee PnL (negative)
    {
        let venue_index = 0;
        let size_tao = 40.0;
        let price = 250.0;
        let fee_bps = cfg.venues[venue_index].taker_fee_bps;

        state.apply_perp_fill(venue_index, Side::Buy, size_tao, price, fee_bps);

        println!(
            "\n[SIM] Applied synthetic fill on {}: Buy {:.4} TAO @ {:.4} (taker fee_bps = {:.2})",
            cfg.venues[venue_index].id, size_tao, price, fee_bps
        );
    }

    let engine = Engine::new(&cfg);
    let mut t_ms = now_ms();

    for tick in 0..NUM_TICKS {
        // ---- Synthetic market data for this tick (dummy mids) ----
        engine.seed_dummy_mids(&mut state, t_ms);

        // ---- Core engine tick: fair value, vols, inventory, PnL, risk ----
        engine.main_tick(&mut state, t_ms);

        println!("\n================ Tick {} ================", tick);

        if let Some(fv) = state.fair_value {
            // ---- State summary ----
            println!("Fair value S_t: {:.4}", fv);
            println!("Sigma_eff: {:.6}", state.sigma_eff);
            println!("Vol ratio (clipped): {:.4}", state.vol_ratio_clipped);
            println!("Spread_mult: {:.4}", state.spread_mult);
            println!("Size_mult: {:.4}", state.size_mult);
            println!("Band_mult: {:.4}", state.band_mult);
            println!("Global q_t (TAO): {:.4}", state.q_global_tao);
            println!("Dollar delta (USD): {:.4}", state.dollar_delta_usd);
            println!("Basis exposure (USD): {:.4}", state.basis_usd);
            println!("Basis gross (USD): {:.4}", state.basis_gross_usd);
            println!("Delta limit (USD): {:.2}", state.delta_limit_usd);
            println!(
                "Basis warn / hard (USD): {:.2} / {:.2}",
                state.basis_limit_warn_usd, state.basis_limit_hard_usd
            );
            println!("Daily realised PnL: {:.4}", state.daily_realised_pnl);
            println!("Daily unrealised PnL: {:.4}", state.daily_unrealised_pnl);
            println!("Daily PnL total: {:.4}", state.daily_pnl_total);
            println!("Risk regime after tick: {:?}", state.risk_regime);
            println!("Kill switch: {}", state.kill_switch);

            // ---- Per-venue toxicity snapshot ----
            println!("\nPer-venue toxicity & status:");
            for v in &state.venues {
                println!(
                    "  {:>10}: toxicity={:.3}, status={:?}",
                    v.id, v.toxicity, v.status
                );
            }

            // ---- Risk gating: kill switch disables MM + hedging ----
            if state.kill_switch {
                println!("\nKill switch ACTIVE: suppressing quotes and hedge intents.");
            } else {
                // ---- Market-making quotes ----
                let quotes = compute_mm_quotes(&cfg, &state);
                println!("\nPer-venue quotes:");
                for q in &quotes {
                    println!(
                        "  {:>10}: bid={:?}, ask={:?}",
                        q.venue_id,
                        q.bid.as_ref().map(|b| (b.price, b.size)),
                        q.ask.as_ref().map(|a| (a.price, a.size)),
                    );
                }

                // ---- Hedge plan for this tick ----
                let hedge_plan = compute_hedge_plan(&cfg, &state);
                println!("\nHedge plan:");
                match &hedge_plan {
                    None => {
                        println!(
                            "  No hedge needed (inside dead band or no hedge venues)."
                        );
                    }
                    Some(plan) => {
                        println!("  Desired ΔH (TAO): {:+.4}", plan.desired_delta);
                        for alloc in &plan.allocations {
                            println!(
                                "  -> venue {:>10}: {:?} {:.4} @ ~{:.4}",
                                alloc.venue_id,
                                alloc.side,
                                alloc.size,
                                alloc.est_price,
                            );
                        }
                    }
                }

                // ---- Convert quotes + hedge plan into abstract order intents ----
                let mut order_intents: Vec<OrderIntent> =
                    mm_quotes_to_order_intents(&quotes);
                if let Some(plan) = &hedge_plan {
                    order_intents.extend(hedge_plan_to_order_intents(plan));
                }

                println!("\nOrder intents (abstract):");
                for oi in &order_intents {
                    println!(
                        "  {:>10} {:?} {:.4} @ {:.4} ({:?})",
                        oi.venue_id, oi.side, oi.size, oi.price, oi.purpose
                    );
                }
            }
        } else {
            println!("Fair value not initialised yet (no mids).");
        }

        // Advance our fake time for the next tick.
        t_ms += MAIN_LOOP_INTERVAL_MS;
    }
}
