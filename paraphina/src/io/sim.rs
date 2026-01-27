// src/io/sim.rs
//
// Simulation adapter for Paraphina (Milestone H).
//
// Implements VenueAdapter for simulation mode where:
// - Every PlaceOrder is immediately filled at limit price
// - Fees are applied based on VenueConfig
// - State is updated via apply_perp_fill
//
// This preserves the existing simulation behavior while conforming
// to the new VenueAdapter interface.

use crate::actions::{CancelAllAction, CancelOrderAction, PlaceOrderAction};
use crate::config::Config;
use crate::io::{ActionResult, VenueAdapter};
use crate::state::GlobalState;
use crate::types::{FillEvent, OrderPurpose, Side, TimeInForce};

/// Simulation adapter that fills orders immediately at limit price.
///
/// This adapter replicates the behavior of the original SimGateway
/// but implements the new VenueAdapter trait.
#[derive(Debug, Default)]
pub struct SimAdapter {
    /// Venue index this adapter handles.
    venue_index: usize,
    /// Enable verbose logging (for debugging).
    verbose: bool,
}

impl SimAdapter {
    /// Create a new simulation adapter for a specific venue.
    pub fn new(venue_index: usize) -> Self {
        Self {
            venue_index,
            verbose: false,
        }
    }

    /// Enable verbose logging.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

impl VenueAdapter for SimAdapter {
    fn place_order(
        &mut self,
        cfg: &Config,
        state: &mut GlobalState,
        action: &PlaceOrderAction,
    ) -> ActionResult {
        // Validate venue index
        if action.venue_index != self.venue_index {
            return ActionResult::Error {
                message: format!(
                    "SimAdapter[{}] received action for venue {}",
                    self.venue_index, action.venue_index
                ),
            };
        }

        let vcfg = match cfg.venues.get(action.venue_index) {
            Some(v) => v,
            None => {
                return ActionResult::Error {
                    message: format!("Invalid venue index {}", action.venue_index),
                };
            }
        };

        let (best_bid, best_ask) = state
            .venues
            .get(action.venue_index)
            .map(|venue| {
                (
                    venue.orderbook_l2.best_bid().map(|l| l.price),
                    venue.orderbook_l2.best_ask().map(|l| l.price),
                )
            })
            .unwrap_or((None, None));
        let crosses = crosses_book(action.side, action.price, best_bid, best_ask);

        if action.post_only && crosses {
            return ActionResult::Rejected {
                reason: "post_only_cross".to_string(),
            };
        }

        if action.time_in_force == TimeInForce::Ioc && !crosses {
            return ActionResult::Rejected {
                reason: "ioc_no_fill".to_string(),
            };
        }

        // Determine fee based on order purpose
        // MM orders get maker fee minus rebate
        // Exit/Hedge orders get taker fee
        let fee_bps = match action.purpose {
            OrderPurpose::Mm => vcfg.maker_fee_bps - vcfg.maker_rebate_bps,
            OrderPurpose::Exit | OrderPurpose::Hedge => vcfg.taker_fee_bps,
        };

        let size_tao = action.size;
        if size_tao <= 0.0 {
            return ActionResult::Rejected {
                reason: "Size must be positive".to_string(),
            };
        }

        let price = action.price;
        if !price.is_finite() || price <= 0.0 {
            return ActionResult::Rejected {
                reason: "Price must be positive and finite".to_string(),
            };
        }

        // Log if verbose
        if self.verbose {
            println!(
                "SimAdapter[{}] fill: {:?} {:.4} @ {:.4} fee_bps={:.2} ({:?})",
                self.venue_index, action.side, size_tao, price, fee_bps, action.purpose
            );
        }

        // Return fill event (state application is handled by the caller's fill batcher)
        ActionResult::Filled(FillEvent {
            venue_index: action.venue_index,
            venue_id: action.venue_id.as_str().into(),
            order_id: Some(action.client_order_id.clone()),
            client_order_id: Some(action.client_order_id.clone()),
            seq: None,
            side: action.side,
            price,
            size: size_tao,
            purpose: action.purpose,
            fee_bps,
        })
    }

    fn cancel_order(
        &mut self,
        _cfg: &Config,
        _state: &mut GlobalState,
        action: &CancelOrderAction,
    ) -> ActionResult {
        // In simulation, we don't track resting orders, so cancel is a no-op
        if self.verbose {
            println!(
                "SimAdapter[{}] cancel: order_id={}",
                self.venue_index, action.order_id
            );
        }
        ActionResult::Cancelled {
            order_id: action.order_id.clone(),
        }
    }

    fn cancel_all(
        &mut self,
        _cfg: &Config,
        _state: &mut GlobalState,
        _action: &CancelAllAction,
    ) -> ActionResult {
        // In simulation, we don't track resting orders, so cancel all returns 0
        if self.verbose {
            println!("SimAdapter[{}] cancel_all", self.venue_index);
        }
        ActionResult::CancelledAll { count: 0 }
    }

    fn name(&self) -> &str {
        "SimAdapter"
    }
}

fn crosses_book(side: Side, price: f64, best_bid: Option<f64>, best_ask: Option<f64>) -> bool {
    match side {
        Side::Buy => best_ask.map_or(false, |ask| price >= ask),
        Side::Sell => best_bid.map_or(false, |bid| price <= bid),
    }
}

/// Create simulation adapters for all venues in config.
pub fn create_sim_adapters(cfg: &Config) -> Vec<Box<dyn VenueAdapter>> {
    cfg.venues
        .iter()
        .enumerate()
        .map(|(i, _)| Box::new(SimAdapter::new(i)) as Box<dyn VenueAdapter>)
        .collect()
}

/// Create simulation adapters with verbose logging.
pub fn create_sim_adapters_verbose(cfg: &Config) -> Vec<Box<dyn VenueAdapter>> {
    cfg.venues
        .iter()
        .enumerate()
        .map(|(i, _)| Box::new(SimAdapter::new(i).with_verbose(true)) as Box<dyn VenueAdapter>)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Side;

    fn setup_test() -> (Config, GlobalState) {
        let cfg = Config::default();
        let state = GlobalState::new(&cfg);
        (cfg, state)
    }

    #[test]
    fn test_sim_adapter_fill() {
        let (cfg, mut state) = setup_test();
        let mut adapter = SimAdapter::new(0);

        let action = PlaceOrderAction {
            action_id: "test_001".to_string(),
            venue_index: 0,
            venue_id: "extended".to_string(),
            side: Side::Buy,
            price: 100.0,
            size: 1.0,
            purpose: OrderPurpose::Mm,
            time_in_force: crate::types::TimeInForce::Gtc,
            post_only: false,
            reduce_only: false,
            client_order_id: "co_test_001".to_string(),
        };

        let result = adapter.place_order(&cfg, &mut state, &action);

        match result {
            ActionResult::Filled(fill) => {
                assert_eq!(fill.venue_index, 0);
                assert_eq!(fill.side, Side::Buy);
                assert_eq!(fill.price, 100.0);
                assert_eq!(fill.size, 1.0);
                // Maker fee - rebate = 2.0 - 0.0 = 2.0 for extended
                assert_eq!(fill.fee_bps, 2.0);

                // Apply fill (sim adapter does not mutate state)
                state.apply_fill_event(&fill, 0, &cfg);
            }
            _ => panic!("Expected Filled result"),
        }

        // Check state was updated
        assert_eq!(state.venues[0].position_tao, 1.0);
    }

    #[test]
    fn test_sim_adapter_taker_fee() {
        let (cfg, mut state) = setup_test();
        let mut adapter = SimAdapter::new(0);
        state.venues[0]
            .orderbook_l2
            .apply_snapshot(
                &[crate::orderbook_l2::BookLevel {
                    price: 99.0,
                    size: 1.0,
                }],
                &[crate::orderbook_l2::BookLevel {
                    price: 101.0,
                    size: 1.0,
                }],
                1,
            )
            .expect("seed book");

        let action = PlaceOrderAction {
            action_id: "test_002".to_string(),
            venue_index: 0,
            venue_id: "extended".to_string(),
            side: Side::Sell,
            price: 99.0,
            size: 1.0,
            purpose: OrderPurpose::Hedge,
            time_in_force: crate::types::TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: "co_test_002".to_string(),
        };

        let result = adapter.place_order(&cfg, &mut state, &action);

        match result {
            ActionResult::Filled(fill) => {
                // Taker fee = 5.0 for extended
                assert_eq!(fill.fee_bps, 5.0);
            }
            _ => panic!("Expected Filled result"),
        }
    }

    #[test]
    fn test_sim_adapter_wrong_venue() {
        let (cfg, mut state) = setup_test();
        let mut adapter = SimAdapter::new(0);

        let action = PlaceOrderAction {
            action_id: "test_003".to_string(),
            venue_index: 1, // Wrong venue for this adapter
            venue_id: "hyperliquid".to_string(),
            side: Side::Buy,
            price: 100.0,
            size: 1.0,
            purpose: OrderPurpose::Mm,
            time_in_force: crate::types::TimeInForce::Gtc,
            post_only: false,
            reduce_only: false,
            client_order_id: "co_test_003".to_string(),
        };

        let result = adapter.place_order(&cfg, &mut state, &action);

        match result {
            ActionResult::Error { message } => {
                assert!(message.contains("venue"));
            }
            _ => panic!("Expected Error result"),
        }
    }

    #[test]
    fn test_sim_adapter_invalid_size() {
        let (cfg, mut state) = setup_test();
        let mut adapter = SimAdapter::new(0);

        let action = PlaceOrderAction {
            action_id: "test_004".to_string(),
            venue_index: 0,
            venue_id: "extended".to_string(),
            side: Side::Buy,
            price: 100.0,
            size: 0.0,
            purpose: OrderPurpose::Mm,
            time_in_force: crate::types::TimeInForce::Gtc,
            post_only: false,
            reduce_only: false,
            client_order_id: "co_test_004".to_string(),
        };

        let result = adapter.place_order(&cfg, &mut state, &action);

        match result {
            ActionResult::Rejected { .. } => {}
            _ => panic!("Expected Rejected result"),
        }
    }

    #[test]
    fn test_create_sim_adapters() {
        let cfg = Config::default();
        let adapters = create_sim_adapters(&cfg);

        assert_eq!(adapters.len(), cfg.venues.len());
        for adapter in &adapters {
            assert_eq!(adapter.name(), "SimAdapter");
        }
    }
}
