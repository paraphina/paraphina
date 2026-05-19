//! Shadow-only V2 arbitrage-aware decision skeleton.
//!
//! This module is intentionally read-only with respect to order flow. It
//! extracts deterministic candidates from baseline MM quotes, records pair-edge
//! features, and emits HOLD-only shadow evidence. It must not construct or
//! mutate order intents.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

use serde::Serialize;

use crate::config::{V2DecisionMode, V2ShadowConfig};
use crate::mm::{MmLevel, MmQuote};
use crate::types::{OrderIntent, OrderPurpose, Side, TimestampMs};

const V2_SHADOW_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum V2ShadowAdmissionStatus {
    Hold,
}

impl V2ShadowAdmissionStatus {
    fn as_str(self) -> &'static str {
        match self {
            V2ShadowAdmissionStatus::Hold => "HOLD",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2ShadowCandidate {
    pub candidate_id: String,
    pub venue_index: usize,
    pub venue_id: String,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub target_linkage_state: &'static str,
    pub admission_status: &'static str,
    pub admission_reason: &'static str,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2PairEdgeSnapshot {
    pub snapshot_id: String,
    pub bid_candidate_id: Option<String>,
    pub ask_candidate_id: Option<String>,
    pub edge_usd: Option<f64>,
    pub edge_bps: Option<f64>,
    pub feature_only: bool,
    pub invalid_reason: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2ShadowDecision {
    pub event_type: &'static str,
    pub schema_version: u32,
    pub telemetry_schema_version: u32,
    pub now_ms: TimestampMs,
    pub decision_mode: &'static str,
    pub admission_status: &'static str,
    pub admission_reason: &'static str,
    pub can_mutate_orders: bool,
    pub order_intent_output_count: usize,
    pub baseline_plan_intent_count: usize,
    pub baseline_mm_order_creating_intent_count: usize,
    pub pair_edge_is_admission: bool,
    pub pressure_complete_claim: bool,
    pub blocker_cleared: bool,
    pub require_phase51_gate: bool,
    pub pair_conditioned_admission_enabled: bool,
    pub fast_hedge_enabled: bool,
    pub order_intent_enabled: bool,
    pub candidates: Vec<V2ShadowCandidate>,
    pub pair_edges: Vec<V2PairEdgeSnapshot>,
}

#[derive(Debug)]
pub struct V2ShadowError {
    reason: String,
}

impl V2ShadowError {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    pub fn sanitized_reason(&self) -> &str {
        &self.reason
    }
}

impl std::fmt::Display for V2ShadowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.sanitized_reason())
    }
}

impl std::error::Error for V2ShadowError {}

pub fn v2_shadow_active(config: &V2ShadowConfig) -> bool {
    config.enabled && matches!(config.decision_mode, V2DecisionMode::Shadow)
}

pub fn evaluate_shadow_decision(
    config: &V2ShadowConfig,
    now_ms: TimestampMs,
    quotes: &[MmQuote],
    baseline_plan_intents: &[OrderIntent],
) -> Option<V2ShadowDecision> {
    if !v2_shadow_active(config) {
        return None;
    }
    let mut candidates = extract_shadow_candidates(quotes);
    if candidates.is_empty() {
        candidates = extract_shadow_candidates_from_intents(baseline_plan_intents);
    }
    let pair_edges = if config.pair_edge_enabled {
        vec![build_pair_edge_snapshot(&candidates)]
    } else {
        Vec::new()
    };
    Some(V2ShadowDecision {
        event_type: "V2_SHADOW_DECISION",
        schema_version: V2_SHADOW_SCHEMA_VERSION,
        telemetry_schema_version: config.telemetry_schema_version,
        now_ms,
        decision_mode: config.decision_mode.as_str(),
        admission_status: V2ShadowAdmissionStatus::Hold.as_str(),
        admission_reason: "shadow_only_no_order_authority",
        can_mutate_orders: false,
        order_intent_output_count: 0,
        baseline_plan_intent_count: baseline_plan_intents.len(),
        baseline_mm_order_creating_intent_count: count_baseline_mm_order_creating_intents(
            baseline_plan_intents,
        ),
        pair_edge_is_admission: false,
        pressure_complete_claim: false,
        blocker_cleared: false,
        require_phase51_gate: config.require_phase51_gate,
        pair_conditioned_admission_enabled: config.pair_conditioned_admission_enabled,
        fast_hedge_enabled: config.fast_hedge_enabled,
        order_intent_enabled: config.order_intent_enabled,
        candidates,
        pair_edges,
    })
}

pub fn emit_shadow_decision(
    config: &V2ShadowConfig,
    now_ms: TimestampMs,
    quotes: &[MmQuote],
    baseline_plan_intents: &[OrderIntent],
) -> Result<Option<usize>, V2ShadowError> {
    let Some(decision) = evaluate_shadow_decision(config, now_ms, quotes, baseline_plan_intents)
    else {
        return Ok(None);
    };
    append_shadow_decision(Path::new(&config.output_path), &decision)?;
    Ok(Some(decision.candidates.len()))
}

fn count_baseline_mm_order_creating_intents(intents: &[OrderIntent]) -> usize {
    intents
        .iter()
        .filter(|intent| match intent {
            OrderIntent::Place(place) => place.purpose == OrderPurpose::Mm,
            OrderIntent::Replace(replace) => replace.purpose == OrderPurpose::Mm,
            OrderIntent::Cancel(_) | OrderIntent::CancelAll(_) => false,
        })
        .count()
}

pub fn extract_shadow_candidates(quotes: &[MmQuote]) -> Vec<V2ShadowCandidate> {
    let mut candidates = Vec::with_capacity(quotes.len() * 2);
    for quote in quotes {
        if let Some(level) = &quote.bid {
            candidates.push(candidate_from_level(quote, Side::Buy, level));
        }
        if let Some(level) = &quote.ask {
            candidates.push(candidate_from_level(quote, Side::Sell, level));
        }
    }
    candidates
}

pub fn extract_shadow_candidates_from_intents(intents: &[OrderIntent]) -> Vec<V2ShadowCandidate> {
    let mut candidates = Vec::new();
    for (idx, intent) in intents.iter().enumerate() {
        match intent {
            OrderIntent::Place(place) if place.purpose == OrderPurpose::Mm => {
                candidates.push(V2ShadowCandidate {
                    candidate_id: format!(
                        "v2_shadow_intent_v1:{}:{}:{}:{}",
                        place.venue_index,
                        place.venue_id,
                        place.side.as_v2_str(),
                        idx
                    ),
                    venue_index: place.venue_index,
                    venue_id: place.venue_id.to_string(),
                    side: place.side,
                    price: place.price,
                    size: place.size,
                    target_linkage_state: if place.phase51_target_key.is_some() {
                        "present_redacted"
                    } else {
                        "missing"
                    },
                    admission_status: V2ShadowAdmissionStatus::Hold.as_str(),
                    admission_reason: "shadow_only_no_order_authority",
                });
            }
            OrderIntent::Replace(replace) if replace.purpose == OrderPurpose::Mm => {
                candidates.push(V2ShadowCandidate {
                    candidate_id: format!(
                        "v2_shadow_intent_v1:{}:{}:{}:{}",
                        replace.venue_index,
                        replace.venue_id,
                        replace.side.as_v2_str(),
                        idx
                    ),
                    venue_index: replace.venue_index,
                    venue_id: replace.venue_id.to_string(),
                    side: replace.side,
                    price: replace.price,
                    size: replace.size,
                    target_linkage_state: if replace.phase51_target_key.is_some() {
                        "present_redacted"
                    } else {
                        "missing"
                    },
                    admission_status: V2ShadowAdmissionStatus::Hold.as_str(),
                    admission_reason: "shadow_only_no_order_authority",
                });
            }
            OrderIntent::Place(_)
            | OrderIntent::Replace(_)
            | OrderIntent::Cancel(_)
            | OrderIntent::CancelAll(_) => {}
        }
    }
    candidates
}

fn candidate_from_level(quote: &MmQuote, side: Side, level: &MmLevel) -> V2ShadowCandidate {
    V2ShadowCandidate {
        candidate_id: format!(
            "v2_shadow_v1:{}:{}:{}",
            quote.venue_index,
            quote.venue_id,
            side.as_v2_str()
        ),
        venue_index: quote.venue_index,
        venue_id: quote.venue_id.to_string(),
        side,
        price: level.price,
        size: level.size,
        target_linkage_state: if level.canonical_target_identity.is_some() {
            "present_redacted"
        } else {
            "missing"
        },
        admission_status: V2ShadowAdmissionStatus::Hold.as_str(),
        admission_reason: "shadow_only_no_order_authority",
    }
}

fn build_pair_edge_snapshot(candidates: &[V2ShadowCandidate]) -> V2PairEdgeSnapshot {
    let best_bid = candidates
        .iter()
        .filter(|candidate| candidate.side == Side::Buy && candidate.price.is_finite())
        .max_by(|lhs, rhs| lhs.price.total_cmp(&rhs.price));
    let best_ask = candidates
        .iter()
        .filter(|candidate| candidate.side == Side::Sell && candidate.price.is_finite())
        .min_by(|lhs, rhs| lhs.price.total_cmp(&rhs.price));

    let Some(bid) = best_bid else {
        return V2PairEdgeSnapshot {
            snapshot_id: "v2_pair_edge_v1:missing_bid".to_string(),
            bid_candidate_id: None,
            ask_candidate_id: best_ask.map(|candidate| candidate.candidate_id.clone()),
            edge_usd: None,
            edge_bps: None,
            feature_only: true,
            invalid_reason: Some("missing_bid"),
        };
    };
    let Some(ask) = best_ask else {
        return V2PairEdgeSnapshot {
            snapshot_id: "v2_pair_edge_v1:missing_ask".to_string(),
            bid_candidate_id: Some(bid.candidate_id.clone()),
            ask_candidate_id: None,
            edge_usd: None,
            edge_bps: None,
            feature_only: true,
            invalid_reason: Some("missing_ask"),
        };
    };

    let edge_usd = bid.price - ask.price;
    let midpoint = (bid.price + ask.price) / 2.0;
    let edge_bps =
        (midpoint.is_finite() && midpoint > 0.0).then_some(edge_usd / midpoint * 10_000.0);
    V2PairEdgeSnapshot {
        snapshot_id: format!("v2_pair_edge_v1:{}:{}", bid.candidate_id, ask.candidate_id),
        bid_candidate_id: Some(bid.candidate_id.clone()),
        ask_candidate_id: Some(ask.candidate_id.clone()),
        edge_usd: Some(edge_usd),
        edge_bps,
        feature_only: true,
        invalid_reason: None,
    }
}

fn append_shadow_decision(path: &Path, decision: &V2ShadowDecision) -> Result<(), V2ShadowError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            V2ShadowError::new(format!("failed_to_create_v2_shadow_parent: {err}"))
        })?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| V2ShadowError::new(format!("failed_to_open_v2_shadow_output: {err}")))?;
    serde_json::to_writer(&mut file, decision)
        .map_err(|err| V2ShadowError::new(format!("failed_to_serialize_v2_shadow: {err}")))?;
    file.write_all(b"\n")
        .map_err(|err| V2ShadowError::new(format!("failed_to_append_v2_shadow_newline: {err}")))?;
    Ok(())
}

trait V2SideExt {
    fn as_v2_str(self) -> &'static str;
}

impl V2SideExt for Side {
    fn as_v2_str(self) -> &'static str {
        match self {
            Side::Buy => "buy",
            Side::Sell => "sell",
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::config::V2ShadowConfig;
    use crate::mm::{mm_quotes_to_order_intents, MmLevel, MmQuote};
    use crate::types::CanonicalTargetIdentity;

    fn quote() -> MmQuote {
        MmQuote {
            venue_index: 1,
            venue_id: Arc::from("hyperliquid"),
            bid: Some(MmLevel {
                price: 99.0,
                size: 0.25,
                canonical_target_identity: CanonicalTargetIdentity::from_explicit(
                    "raw-group-must-not-emit",
                    "raw-order-must-not-emit",
                ),
            }),
            ask: Some(MmLevel {
                price: 101.0,
                size: 0.20,
                canonical_target_identity: None,
            }),
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "quoted",
            ask_terminal_reason: "quoted",
        }
    }

    fn quote_for(
        venue_index: usize,
        venue_id: &'static str,
        bid_price: Option<f64>,
        ask_price: Option<f64>,
    ) -> MmQuote {
        MmQuote {
            venue_index,
            venue_id: Arc::from(venue_id),
            bid: bid_price.map(|price| MmLevel {
                price,
                size: 0.01,
                canonical_target_identity: CanonicalTargetIdentity::from_explicit(
                    "raw-group-must-not-emit",
                    "raw-order-must-not-emit",
                ),
            }),
            ask: ask_price.map(|price| MmLevel {
                price,
                size: 0.01,
                canonical_target_identity: None,
            }),
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "quoted",
            ask_terminal_reason: "quoted",
        }
    }

    fn shadow_config() -> V2ShadowConfig {
        V2ShadowConfig {
            enabled: true,
            decision_mode: V2DecisionMode::Shadow,
            pair_edge_enabled: true,
            ..V2ShadowConfig::default()
        }
    }

    #[test]
    fn v2_shadow_inactive_by_default() {
        let config = V2ShadowConfig::default();
        assert!(!v2_shadow_active(&config));
        assert!(evaluate_shadow_decision(&config, 1_000, &[quote()], &[]).is_none());
    }

    #[test]
    fn v2_shadow_extracts_redacted_hold_only_candidates() {
        let config = shadow_config();
        let decision = evaluate_shadow_decision(&config, 1_000, &[quote()], &[]).expect("decision");
        assert_eq!(decision.admission_status, "HOLD");
        assert_eq!(decision.order_intent_output_count, 0);
        assert!(!decision.can_mutate_orders);
        assert!(!decision.pair_edge_is_admission);
        assert!(!decision.pressure_complete_claim);
        assert!(!decision.blocker_cleared);
        assert_eq!(decision.candidates.len(), 2);
        assert_eq!(
            decision.candidates[0].target_linkage_state,
            "present_redacted"
        );
        assert_eq!(decision.candidates[1].target_linkage_state, "missing");

        let serialized = serde_json::to_string(&decision).expect("serialize");
        assert!(!serialized.contains("raw-group-must-not-emit"));
        assert!(!serialized.contains("raw-order-must-not-emit"));
    }

    #[test]
    fn v2_pair_edge_is_feature_only_not_admission() {
        let config = shadow_config();
        let decision = evaluate_shadow_decision(&config, 1_000, &[quote()], &[]).expect("decision");
        assert_eq!(decision.pair_edges.len(), 1);
        assert!(decision.pair_edges[0].feature_only);
        assert_eq!(decision.pair_edges[0].edge_usd, Some(-2.0));
        assert_eq!(decision.admission_reason, "shadow_only_no_order_authority");
    }

    #[test]
    fn v2_shadow_all_five_quote_matrix_uses_feature_only_pair_edge() {
        let config = shadow_config();
        let quotes = vec![
            quote_for(0, "extended", Some(100.0), Some(105.0)),
            quote_for(1, "hyperliquid", Some(110.0), Some(104.0)),
            quote_for(2, "aster", Some(102.0), Some(99.0)),
            quote_for(3, "lighter", Some(103.0), Some(102.0)),
            quote_for(4, "paradex", Some(104.0), Some(101.0)),
        ];
        let decision = evaluate_shadow_decision(&config, 3_000, &quotes, &[]).expect("decision");

        assert_eq!(decision.candidates.len(), 10);
        assert_eq!(decision.pair_edges.len(), 1);
        let pair_edge = &decision.pair_edges[0];
        assert!(pair_edge.feature_only);
        assert_eq!(
            pair_edge.bid_candidate_id.as_deref(),
            Some("v2_shadow_v1:1:hyperliquid:buy")
        );
        assert_eq!(
            pair_edge.ask_candidate_id.as_deref(),
            Some("v2_shadow_v1:2:aster:sell")
        );
        assert_eq!(pair_edge.edge_usd, Some(11.0));
        assert_eq!(decision.admission_status, "HOLD");
        assert_eq!(decision.admission_reason, "shadow_only_no_order_authority");
        assert!(!decision.can_mutate_orders);
        assert_eq!(decision.order_intent_output_count, 0);
        assert!(!decision.pair_edge_is_admission);
        assert!(!decision.pressure_complete_claim);
        assert!(!decision.blocker_cleared);

        let serialized = serde_json::to_string(&decision).expect("serialize");
        assert!(!serialized.contains("raw-group-must-not-emit"));
        assert!(!serialized.contains("raw-order-must-not-emit"));
    }

    #[test]
    fn v2_shadow_does_not_mutate_baseline_order_intents() {
        let config = shadow_config();
        let quotes = vec![quote()];
        let before = mm_quotes_to_order_intents(&quotes);
        let decision =
            evaluate_shadow_decision(&config, 1_000, &quotes, &before).expect("decision");
        let after = mm_quotes_to_order_intents(&quotes);
        assert_eq!(before, after);
        assert_eq!(decision.order_intent_output_count, 0);
        assert_eq!(decision.baseline_plan_intent_count, before.len());
        assert_eq!(
            decision.baseline_mm_order_creating_intent_count,
            before.len()
        );
    }

    #[test]
    fn v2_shadow_falls_back_to_sanitized_intent_candidates_when_quotes_absent() {
        use std::sync::Arc;

        let config = shadow_config();
        let intents = vec![
            OrderIntent::Place(crate::types::PlaceOrderIntent {
                venue_index: 2,
                venue_id: Arc::from("lighter"),
                side: Side::Buy,
                price: 100.5,
                size: 0.01,
                purpose: OrderPurpose::Mm,
                time_in_force: crate::types::TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: Some("raw-client-id-must-not-emit".to_string()),
                phase51_target_key: Some(crate::types::Phase51ForwardRefreshTargetKey {
                    canonical_group_id: "raw-group-must-not-emit".to_string(),
                    order_key: "raw-order-must-not-emit".to_string(),
                }),
            }),
            OrderIntent::Replace(crate::types::ReplaceOrderIntent {
                venue_index: 3,
                venue_id: Arc::from("paradex"),
                side: Side::Sell,
                price: 101.5,
                size: 0.02,
                purpose: OrderPurpose::Mm,
                time_in_force: crate::types::TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                order_id: "raw-order-id-must-not-emit".to_string(),
                client_order_id: Some("raw-replace-client-id-must-not-emit".to_string()),
                phase51_target_key: None,
            }),
        ];

        let decision = evaluate_shadow_decision(&config, 2_000, &[], &intents).expect("decision");
        assert_eq!(decision.candidates.len(), 2);
        assert_eq!(
            decision.baseline_mm_order_creating_intent_count,
            intents.len()
        );
        assert_eq!(
            decision.candidates[0].target_linkage_state,
            "present_redacted"
        );
        assert_eq!(decision.candidates[1].target_linkage_state, "missing");
        assert_eq!(decision.order_intent_output_count, 0);
        assert!(!decision.can_mutate_orders);

        let serialized = serde_json::to_string(&decision).expect("serialize");
        assert!(!serialized.contains("raw-client-id-must-not-emit"));
        assert!(!serialized.contains("raw-replace-client-id-must-not-emit"));
        assert!(!serialized.contains("raw-order-id-must-not-emit"));
        assert!(!serialized.contains("raw-group-must-not-emit"));
        assert!(!serialized.contains("raw-order-must-not-emit"));
    }
}
