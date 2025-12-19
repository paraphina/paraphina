// src/gateway.rs
//
// Execution gateway abstraction + a simple synthetic gateway used by
// the simulation harness. In production, this will be replaced by
// real exchange connectors implementing `ExecutionGateway`.

use crate::config::Config;
use crate::state::GlobalState;
use crate::types::{FillEvent, OrderIntent, OrderPurpose};

/// Abstract execution gateway.
///
/// Given a set of `OrderIntent`s, the gateway is responsible for turning
/// them into real orders / fills and updating `GlobalState` via
/// `apply_perp_fill`. In the simulation we simply assume that every
/// intent is fully filled at its limit price.
pub trait ExecutionGateway {
    fn process_intents(
        &mut self,
        cfg: &Config,
        state: &mut GlobalState,
        intents: &[OrderIntent],
    ) -> Vec<FillEvent>;
}

/// Synthetic gateway used in the simulation:
///  - every intent is assumed to fill fully at its limit price,
///  - maker / taker fees are taken from `VenueConfig`,
///  - PnL is updated via `GlobalState::apply_perp_fill`.
#[derive(Debug, Default)]
pub struct SimGateway;

impl SimGateway {
    pub fn new() -> Self {
        Self
    }
}

impl ExecutionGateway for SimGateway {
    fn process_intents(
        &mut self,
        cfg: &Config,
        state: &mut GlobalState,
        intents: &[OrderIntent],
    ) -> Vec<FillEvent> {
        let mut fills = Vec::with_capacity(intents.len());

        for intent in intents {
            let venue_index = intent.venue_index;

            let vcfg = match cfg.venues.get(venue_index) {
                Some(v) => v,
                None => {
                    eprintln!(
                        "SimGateway: invalid venue_index {} for intent {:?}",
                        venue_index, intent
                    );
                    continue;
                }
            };

            // Fee model: MM quotes pay maker_fee - maker_rebate,
            // hedge / exit trades pay taker fee.
            let fee_bps = match intent.purpose {
                OrderPurpose::Mm => vcfg.maker_fee_bps - vcfg.maker_rebate_bps,
                OrderPurpose::Exit | OrderPurpose::Hedge => vcfg.taker_fee_bps,
            };

            let size_tao = intent.size;
            if size_tao <= 0.0 {
                continue;
            }

            let price = intent.price;

            println!(
                "Synthetic fill: {:<10} {:?} {:>6.4} @ {:>8.4} fee_bps={:.2} ({:?})",
                intent.venue_id, intent.side, size_tao, price, fee_bps, intent.purpose
            );

            // Apply fill into global state (positions + realised PnL).
            state.apply_perp_fill(venue_index, intent.side, size_tao, price, fee_bps);

            fills.push(FillEvent {
                venue_index,
                venue_id: intent.venue_id.clone(),
                side: intent.side,
                price,
                size: size_tao,
                purpose: intent.purpose,
                fee_bps,
            });
        }

        fills
    }
}
