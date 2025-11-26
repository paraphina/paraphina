mod config;
mod state;
mod types;
mod engine;
mod mm;
mod hedge;

use crate::config::Config;
use crate::engine::Engine;
use crate::state::GlobalState;
use crate::mm::{compute_mm_quotes, mm_quotes_to_order_intents};
use crate::hedge::{compute_hedge_plan, hedge_plan_to_order_intents};
use crate::types::OrderIntent;

use std::time::{SystemTime, UNIX_EPOCH};

fn now_ms() -> i64 {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    dur.as_millis() as i64
}

fn main() {
    let cfg = Config::default();
    let mut state = GlobalState::new(&cfg);

    println!(
        "Paraphina MM starting with config version: {}",
        cfg.version
    );
    println!("Configured venues: {}", cfg.venues.len());
    println!("Initial risk regime: {:?}", state.risk_regime);

    let engine = Engine::new(&cfg);

    // TEMP: seed synthetic mids so KF + vol logic can run before we have
    // any real exchange connections wired up.
    let t0 = now_ms();
    engine.seed_dummy_mids(&mut state, t0);

    // ==== TEMP: add some fake positions so the hedge engine has work to do ====
    // extended: long +50 TAO
    if let Some(v) = state.venues.get_mut(0) {
        v.position_tao = 50.0;
    }
    // hyperliquid: short -10 TAO
    if let Some(v) = state.venues.get_mut(1) {
        v.position_tao = -10.0;
    }
    // (aster / lighter / paradex left at 0.0 for now)
    // ==========================================================================

    // Run one "main loop" tick (fair value, vols, inventory, risk).
    engine.main_tick(&mut state, t0);

    if let Some(fv) = state.fair_value {
        println!("--- Post-tick state ---");
        println!("Fair value S_t: {:.4}", fv);
        println!("Sigma_eff: {:.6}", state.sigma_eff);
        println!("Vol ratio (clipped): {:.4}", state.vol_ratio_clipped);
        println!("Spread_mult: {:.4}", state.spread_mult);
        println!("Size_mult: {:.4}", state.size_mult);
        println!("Band_mult: {:.4}", state.band_mult);
        println!("Global q_t (TAO): {:.4}", state.q_global_tao);
        println!("Dollar delta (USD): {:.4}", state.dollar_delta_usd);
        println!("Basis exposure (USD): {:.4}", state.basis_usd);
        println!(
            "Basis gross (USD): {:.4}",
            state.basis_gross_usd
        );
        println!(
            "Delta limit (USD): {:.2}",
            state.delta_limit_usd
        );
        println!(
            "Basis warn / hard (USD): {:.2} / {:.2}",
            state.basis_limit_warn_usd, state.basis_limit_hard_usd
        );
        println!("Daily PnL total: {:.4}", state.daily_pnl_total);
        println!("Risk regime after tick: {:?}", state.risk_regime);
        println!("Kill switch: {}", state.kill_switch);

        // ----- MM quotes -----
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

        // ----- Hedge engine: compute one-step hedge plan -----
        let hedge_plan = compute_hedge_plan(&cfg, &state);
        println!("\nHedge plan:");
        match &hedge_plan {
            None => println!("  No hedge needed (inside dead band or no hedge venues)."),
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

        // ----- Turn MM quotes + hedge plan into abstract order intents -----
        let mut order_intents: Vec<OrderIntent> = mm_quotes_to_order_intents(&quotes);
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
    } else {
        println!("Fair value not initialised yet (no mids).");
    }
}
