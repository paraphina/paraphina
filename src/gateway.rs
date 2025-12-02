// src/gateway.rs
//
// Execution / gateway abstraction for Paraphina.
//
// For now we implement a purely synthetic "SimGateway" that:
//   - takes abstract OrderIntents,
//   - chooses appropriate fees (maker vs taker) per venue,
//   - applies fills into GlobalState via `apply_perp_fill`,
//   - logs each synthetic fill.
//
// Later, real connectors (REST/WebSocket) will implement the
// same interface in terms of OrderIntents.

use crate::config::Config;
use crate::state::GlobalState;
use crate::types::{OrderIntent, OrderPurpose};

/// Synthetic execution backend used in simulation.
pub struct SimGateway;

impl SimGateway {
    /// Create a new simulation gateway.
    pub fn new() -> Self {
        SimGateway
    }

    /// Apply a batch of synthetic fills into the state.
    ///
    /// - MM intents are treated as maker fills:
    ///     fee_bps = maker_fee_bps - maker_rebate_bps
    /// - Hedge / Exit intents are treated as taker fills:
    ///     fee_bps = taker_fee_bps
    pub fn apply_fills(
        &self,
        cfg: &Config,
        state: &mut GlobalState,
        intents: &[OrderIntent],
    ) {
        if intents.is_empty() {
            println!("\nSynthetic fills (SimGateway): none");
            return;
        }

        println!("\nSynthetic fills (SimGateway):");

        for intent in intents {
            let vcfg = &cfg.venues[intent.venue_index];

            // Decide maker vs taker fees based on purpose.
            let fee_bps = match intent.purpose {
                OrderPurpose::Mm => vcfg.maker_fee_bps - vcfg.maker_rebate_bps,
                OrderPurpose::Exit | OrderPurpose::Hedge => vcfg.taker_fee_bps,
            };

            println!(
                "  Synthetic fill: {:>9} {:?} {:.4} @ {:.4} fee_bps={:.2} ({:?})",
                intent.venue_id,
                intent.side,
                intent.size,
                intent.price,
                fee_bps,
                intent.purpose,
            );

            state.apply_perp_fill(
                intent.venue_index,
                intent.side,
                intent.size,
                intent.price,
                fee_bps,
            );
        }
    }
}
