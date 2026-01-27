//! Deterministic L2 order book engine (core).

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BookSide {
    Bid,
    Ask,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BookLevel {
    pub price: f64,
    pub size: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BookLevelDelta {
    pub side: BookSide,
    pub price: f64,
    pub size: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DepthConfig {
    pub levels: usize,
    pub include_imbalance: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DerivedBookMetrics {
    pub mid: Option<f64>,
    pub spread: Option<f64>,
    pub depth_near_mid: f64,
    pub imbalance: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrderBookError {
    SeqOutOfOrder { last_seq: u64, incoming_seq: u64 },
    SeqGap { last_seq: u64, incoming_seq: u64 },
    InvalidPrice { price: f64 },
    InvalidSize { size: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderBookL2 {
    bids: Vec<BookLevel>,
    asks: Vec<BookLevel>,
    last_seq: u64,
}

impl OrderBookL2 {
    pub fn new() -> Self {
        Self {
            bids: Vec::new(),
            asks: Vec::new(),
            last_seq: 0,
        }
    }

    pub fn last_seq(&self) -> u64 {
        self.last_seq
    }

    pub fn bids(&self) -> &[BookLevel] {
        &self.bids
    }

    pub fn asks(&self) -> &[BookLevel] {
        &self.asks
    }

    pub fn apply_snapshot(
        &mut self,
        bids: &[BookLevel],
        asks: &[BookLevel],
        seq: u64,
    ) -> Result<(), OrderBookError> {
        if seq <= self.last_seq {
            return Err(OrderBookError::SeqOutOfOrder {
                last_seq: self.last_seq,
                incoming_seq: seq,
            });
        }
        self.bids = validate_and_sort(bids, true)?;
        self.asks = validate_and_sort(asks, false)?;
        self.last_seq = seq;
        Ok(())
    }

    pub fn apply_delta(
        &mut self,
        deltas: &[BookLevelDelta],
        seq: u64,
    ) -> Result<(), OrderBookError> {
        if seq <= self.last_seq {
            return Err(OrderBookError::SeqOutOfOrder {
                last_seq: self.last_seq,
                incoming_seq: seq,
            });
        }
        if seq != self.last_seq + 1 {
            return Err(OrderBookError::SeqGap {
                last_seq: self.last_seq,
                incoming_seq: seq,
            });
        }
        for delta in deltas {
            if !delta.price.is_finite() {
                return Err(OrderBookError::InvalidPrice { price: delta.price });
            }
            if !delta.size.is_finite() {
                return Err(OrderBookError::InvalidSize { size: delta.size });
            }
            match delta.side {
                BookSide::Bid => apply_delta_to_levels(&mut self.bids, delta, true),
                BookSide::Ask => apply_delta_to_levels(&mut self.asks, delta, false),
            }
        }
        self.last_seq = seq;
        Ok(())
    }

    pub fn trim_levels(&mut self, max_levels: usize) {
        if max_levels == 0 {
            self.bids.clear();
            self.asks.clear();
            return;
        }
        if self.bids.len() > max_levels {
            self.bids.truncate(max_levels);
        }
        if self.asks.len() > max_levels {
            self.asks.truncate(max_levels);
        }
    }

    pub fn best_bid(&self) -> Option<BookLevel> {
        self.bids.first().copied()
    }

    pub fn best_ask(&self) -> Option<BookLevel> {
        self.asks.first().copied()
    }

    pub fn compute_mid_spread_depth(&self, depth_cfg: DepthConfig) -> DerivedBookMetrics {
        let best_bid = self.best_bid();
        let best_ask = self.best_ask();
        let mid = match (best_bid, best_ask) {
            (Some(bid), Some(ask)) if bid.price.is_finite() && ask.price.is_finite() => {
                Some((bid.price + ask.price) / 2.0)
            }
            _ => None,
        };
        let spread = match (best_bid, best_ask) {
            (Some(bid), Some(ask)) if bid.price.is_finite() && ask.price.is_finite() => {
                Some(ask.price - bid.price)
            }
            _ => None,
        };
        let bid_depth = top_levels_depth(&self.bids, depth_cfg.levels);
        let ask_depth = top_levels_depth(&self.asks, depth_cfg.levels);
        let depth_near_mid = bid_depth + ask_depth;
        let imbalance = if depth_cfg.include_imbalance {
            let total = bid_depth + ask_depth;
            if total > 0.0 {
                Some((bid_depth - ask_depth) / total)
            } else {
                Some(0.0)
            }
        } else {
            None
        };
        DerivedBookMetrics {
            mid,
            spread,
            depth_near_mid,
            imbalance,
        }
    }
}

fn validate_and_sort(levels: &[BookLevel], is_bid: bool) -> Result<Vec<BookLevel>, OrderBookError> {
    let mut out = Vec::with_capacity(levels.len());
    for level in levels {
        if !level.price.is_finite() {
            return Err(OrderBookError::InvalidPrice { price: level.price });
        }
        if !level.size.is_finite() {
            return Err(OrderBookError::InvalidSize { size: level.size });
        }
        if level.size <= 0.0 {
            continue;
        }
        out.push(*level);
    }
    out.sort_by(|a, b| compare_prices(a.price, b.price, is_bid));
    Ok(out)
}

fn apply_delta_to_levels(levels: &mut Vec<BookLevel>, delta: &BookLevelDelta, is_bid: bool) {
    let price = delta.price;
    let size = delta.size;
    if size <= 0.0 {
        if let Some(pos) = levels.iter().position(|l| (l.price - price).abs() < 1e-9) {
            levels.remove(pos);
        }
        return;
    }
    if let Some(level) = levels.iter_mut().find(|l| (l.price - price).abs() < 1e-9) {
        level.size = size;
        return;
    }
    levels.push(BookLevel { price, size });
    levels.sort_by(|a, b| compare_prices(a.price, b.price, is_bid));
}

fn compare_prices(a: f64, b: f64, is_bid: bool) -> Ordering {
    if is_bid {
        b.total_cmp(&a)
    } else {
        a.total_cmp(&b)
    }
}

fn top_levels_depth(levels: &[BookLevel], count: usize) -> f64 {
    levels
        .iter()
        .take(count)
        .map(|l| (l.price * l.size).abs())
        .sum()
}

impl Default for OrderBookL2 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_then_deltas_best_bid_ask() {
        let mut book = OrderBookL2::new();
        let bids = vec![
            BookLevel {
                price: 100.0,
                size: 1.0,
            },
            BookLevel {
                price: 99.5,
                size: 2.0,
            },
        ];
        let asks = vec![
            BookLevel {
                price: 101.0,
                size: 1.0,
            },
            BookLevel {
                price: 102.0,
                size: 3.0,
            },
        ];
        book.apply_snapshot(&bids, &asks, 10).unwrap();
        let delta = vec![
            BookLevelDelta {
                side: BookSide::Bid,
                price: 100.5,
                size: 1.0,
            },
            BookLevelDelta {
                side: BookSide::Ask,
                price: 101.0,
                size: 0.0,
            },
        ];
        book.apply_delta(&delta, 11).unwrap();
        assert_eq!(book.best_bid().unwrap().price, 100.5);
        assert_eq!(book.best_ask().unwrap().price, 102.0);
    }

    #[test]
    fn seq_regression_rejected() {
        let mut book = OrderBookL2::new();
        book.apply_snapshot(&[], &[], 5).unwrap();
        let err = book.apply_delta(&[], 4).unwrap_err();
        match err {
            OrderBookError::SeqOutOfOrder {
                last_seq,
                incoming_seq,
            } => {
                assert_eq!(last_seq, 5);
                assert_eq!(incoming_seq, 4);
            }
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn depth_near_mid_computed() {
        let mut book = OrderBookL2::new();
        book.apply_snapshot(
            &[
                BookLevel {
                    price: 100.0,
                    size: 2.0,
                },
                BookLevel {
                    price: 99.0,
                    size: 3.0,
                },
            ],
            &[
                BookLevel {
                    price: 101.0,
                    size: 3.0,
                },
                BookLevel {
                    price: 102.0,
                    size: 1.0,
                },
            ],
            1,
        )
        .unwrap();
        let metrics = book.compute_mid_spread_depth(DepthConfig {
            levels: 1,
            include_imbalance: true,
        });
        assert_eq!(metrics.depth_near_mid, 503.0);
        assert_eq!(metrics.spread.unwrap(), 1.0);
        assert!(metrics.imbalance.unwrap().abs() > 0.0);
    }

    #[test]
    fn determinism_serialization() {
        let mut book_a = OrderBookL2::new();
        let mut book_b = OrderBookL2::new();
        let snapshot = vec![
            BookLevel {
                price: 100.0,
                size: 1.0,
            },
            BookLevel {
                price: 99.0,
                size: 2.0,
            },
        ];
        let asks = vec![
            BookLevel {
                price: 101.0,
                size: 1.0,
            },
            BookLevel {
                price: 102.0,
                size: 2.0,
            },
        ];
        book_a.apply_snapshot(&snapshot, &asks, 1).unwrap();
        book_b.apply_snapshot(&snapshot, &asks, 1).unwrap();
        book_a
            .apply_delta(
                &[BookLevelDelta {
                    side: BookSide::Bid,
                    price: 100.5,
                    size: 1.0,
                }],
                2,
            )
            .unwrap();
        book_b
            .apply_delta(
                &[BookLevelDelta {
                    side: BookSide::Bid,
                    price: 100.5,
                    size: 1.0,
                }],
                2,
            )
            .unwrap();
        let ser_a = serde_json::to_vec(&book_a).unwrap();
        let ser_b = serde_json::to_vec(&book_b).unwrap();
        assert_eq!(ser_a, ser_b);
    }
}
