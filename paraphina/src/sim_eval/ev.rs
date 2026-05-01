//! Phase 5.1 non-live EV research primitives.
//!
//! This module is intentionally detached from live execution.  It provides the
//! deterministic data contracts and conservative EV helpers needed for replay
//! and shadow evaluation without changing market-making behaviour.

use serde::{Deserialize, Serialize};

pub const PHASE5_1_BASELINE_COMMIT: &str = "18dd09512288a85e440d3977e32432c3aabc1190";
pub const PHASE5_1_DEFAULT_ALPHA: f64 = 0.05;
pub const PHASE5_1_DEFAULT_Z_ONE_SIDED_95: f64 = 1.644_853_626_951_472_2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CandidateSide {
    Bid,
    Ask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CandidateLayer {
    Touch,
    Working,
    InventoryReducing,
    BaselineCompat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EvDecision {
    Admit,
    Reject,
    Hold,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BindingConstraint {
    HedgeDepthCap,
    EntryMarginCap,
    HedgeMarginCap,
    PairBudget,
    UnpairedDeltaBudget,
    LiquidationSafeSizeEntry,
    LiquidationSafeSizeHedge,
    VenueOrderLimit,
    RateLimitSafeSize,
    ConfiguredNonliveExperimentCap,
    MinLot,
    MinNotional,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderCandidate {
    pub candidate_id: String,
    pub run_id: String,
    pub baseline_commit: String,
    pub config_hash: String,
    pub model_version: String,
    pub instrument_id: String,
    pub entry_venue_id: String,
    pub side: CandidateSide,
    pub layer: CandidateLayer,
    pub passive_price: f64,
    pub candidate_size_q: f64,
    pub candidate_notional_usd: f64,
    pub intended_lifetime_ms: u64,
    pub local_edge_feature: f64,
    pub pair_edge_feature: Option<f64>,
    pub primary_hedge_venue_id: Option<String>,
    pub backup_hedge_venue_id: Option<String>,
    pub no_live_flag: bool,
}

impl OrderCandidate {
    pub fn validate_nonlive_baseline(&self) -> Result<(), String> {
        if self.baseline_commit != PHASE5_1_BASELINE_COMMIT {
            return Err(format!(
                "baseline_commit mismatch: expected {PHASE5_1_BASELINE_COMMIT}, got {}",
                self.baseline_commit
            ));
        }
        if !self.no_live_flag {
            return Err("no_live_flag must be true for Phase 5.1 EV candidates".to_string());
        }
        if self.candidate_id.is_empty() {
            return Err("candidate_id must be non-empty".to_string());
        }
        if self.config_hash.is_empty() {
            return Err("config_hash must be non-empty".to_string());
        }
        if !(self.passive_price.is_finite() && self.passive_price > 0.0) {
            return Err("passive_price must be finite and positive".to_string());
        }
        if !(self.candidate_size_q.is_finite() && self.candidate_size_q >= 0.0) {
            return Err("candidate_size_q must be finite and non-negative".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EvComponents {
    pub p_fill: f64,
    pub p_hedge_success: f64,
    pub p_hedge_partial: f64,
    pub p_hedge_fail: f64,
    pub e_locked_edge: f64,
    pub e_partial_hedge_state: f64,
    pub e_residual_inventory_state: f64,
    pub e_adverse_selection: f64,
    pub e_queue_reset: f64,
    pub e_churn: f64,
    pub e_capital_funding: f64,
    pub e_tail_risk: f64,
}

impl EvComponents {
    pub fn validate(&self) -> Result<(), String> {
        let probabilities = [
            ("p_fill", self.p_fill),
            ("p_hedge_success", self.p_hedge_success),
            ("p_hedge_partial", self.p_hedge_partial),
            ("p_hedge_fail", self.p_hedge_fail),
        ];
        for (name, value) in probabilities {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!("{name} must be finite and in [0, 1]"));
            }
        }
        let hedge_sum = self.p_hedge_success + self.p_hedge_partial + self.p_hedge_fail;
        if (hedge_sum - 1.0).abs() > 1e-9 {
            return Err(format!(
                "hedge state probabilities must sum to 1.0, got {hedge_sum}"
            ));
        }
        let finite_terms = [
            ("e_locked_edge", self.e_locked_edge),
            ("e_partial_hedge_state", self.e_partial_hedge_state),
            (
                "e_residual_inventory_state",
                self.e_residual_inventory_state,
            ),
            ("e_adverse_selection", self.e_adverse_selection),
            ("e_queue_reset", self.e_queue_reset),
            ("e_churn", self.e_churn),
            ("e_capital_funding", self.e_capital_funding),
            ("e_tail_risk", self.e_tail_risk),
        ];
        for (name, value) in finite_terms {
            if !value.is_finite() {
                return Err(format!("{name} must be finite"));
            }
        }
        Ok(())
    }

    pub fn expected_value(&self) -> f64 {
        let conditional_fill_ev = self.p_hedge_success * self.e_locked_edge
            + self.p_hedge_partial * self.e_partial_hedge_state
            + self.p_hedge_fail * self.e_residual_inventory_state;
        self.p_fill * conditional_fill_ev
            - self.e_adverse_selection
            - self.e_queue_reset
            - self.e_churn
            - self.e_capital_funding
            - self.e_tail_risk
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EvConfidence {
    pub alpha: f64,
    pub standard_error: f64,
    pub z_score: f64,
}

impl Default for EvConfidence {
    fn default() -> Self {
        Self {
            alpha: PHASE5_1_DEFAULT_ALPHA,
            standard_error: 0.0,
            z_score: PHASE5_1_DEFAULT_Z_ONE_SIDED_95,
        }
    }
}

impl EvConfidence {
    pub fn lower_confidence_bound(&self, ev_hat: f64) -> Result<f64, String> {
        if !ev_hat.is_finite() {
            return Err("ev_hat must be finite".to_string());
        }
        if !self.alpha.is_finite() || !(0.0..0.5).contains(&self.alpha) {
            return Err("alpha must be finite and in (0, 0.5)".to_string());
        }
        if !self.standard_error.is_finite() || self.standard_error < 0.0 {
            return Err("standard_error must be finite and non-negative".to_string());
        }
        if !self.z_score.is_finite() || self.z_score < 0.0 {
            return Err("z_score must be finite and non-negative".to_string());
        }
        Ok(ev_hat - self.z_score * self.standard_error)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvEvaluation {
    pub candidate_id: String,
    pub selected_size_q: f64,
    pub components: EvComponents,
    pub ev_hat: f64,
    pub ev_lcb_alpha: f64,
    pub alpha: f64,
    pub decision: EvDecision,
    pub decision_reason_primary: String,
    pub binding_constraints: Vec<BindingConstraint>,
}

pub fn evaluate_candidate(
    candidate: &OrderCandidate,
    components: EvComponents,
    confidence: EvConfidence,
    binding_constraints: Vec<BindingConstraint>,
) -> Result<EvEvaluation, String> {
    candidate.validate_nonlive_baseline()?;
    components.validate()?;
    let ev_hat = components.expected_value();
    let ev_lcb_alpha = confidence.lower_confidence_bound(ev_hat)?;
    let (decision, reason) = if ev_lcb_alpha > 0.0 {
        (EvDecision::Admit, "positive_lcb_ev")
    } else {
        (EvDecision::Reject, "non_positive_lcb_ev")
    };
    Ok(EvEvaluation {
        candidate_id: candidate.candidate_id.clone(),
        selected_size_q: candidate.candidate_size_q,
        components,
        ev_hat,
        ev_lcb_alpha,
        alpha: confidence.alpha,
        decision,
        decision_reason_primary: reason.to_string(),
        binding_constraints,
    })
}

pub fn select_discrete_size<F>(
    candidate: &OrderCandidate,
    feasible_sizes: &[f64],
    confidence: EvConfidence,
    mut components_for_size: F,
) -> Result<EvEvaluation, String>
where
    F: FnMut(f64) -> (EvComponents, Vec<BindingConstraint>),
{
    candidate.validate_nonlive_baseline()?;
    if feasible_sizes.is_empty() {
        return Err("feasible_sizes must be non-empty".to_string());
    }

    let mut best: Option<EvEvaluation> = None;
    for &size in feasible_sizes {
        if !size.is_finite() || size < 0.0 {
            return Err("all feasible sizes must be finite and non-negative".to_string());
        }
        let mut sized_candidate = candidate.clone();
        sized_candidate.candidate_size_q = size;
        sized_candidate.candidate_notional_usd = size * candidate.passive_price;
        let (components, constraints) = components_for_size(size);
        let evaluation = evaluate_candidate(&sized_candidate, components, confidence, constraints)?;
        if best
            .as_ref()
            .is_none_or(|current| evaluation.ev_lcb_alpha > current.ev_lcb_alpha)
        {
            best = Some(evaluation);
        }
    }

    let mut selected = best.expect("non-empty feasible size set");
    if selected.ev_lcb_alpha <= 0.0 {
        selected.decision = EvDecision::Reject;
        selected.decision_reason_primary = "best_size_non_positive_lcb_ev".to_string();
    }
    Ok(selected)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReplayLabelType {
    ObservedFact,
    ModelEstimate,
    CounterfactualDecision,
    CounterfactualOutcome,
    SimulatedOutcome,
    PaperOutcome,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayLabel {
    pub event_seq: u64,
    pub candidate_id: String,
    pub label_type: ReplayLabelType,
    pub deterministic_replay_key: String,
    pub admissible_for_financial_claim: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate() -> OrderCandidate {
        OrderCandidate {
            candidate_id: "run:1:lighter:BID:WORKING".to_string(),
            run_id: "run".to_string(),
            baseline_commit: PHASE5_1_BASELINE_COMMIT.to_string(),
            config_hash: "cfg".to_string(),
            model_version: "phase5_1_ev_v0".to_string(),
            instrument_id: "ETH-PERP".to_string(),
            entry_venue_id: "lighter".to_string(),
            side: CandidateSide::Bid,
            layer: CandidateLayer::Working,
            passive_price: 3_000.0,
            candidate_size_q: 0.01,
            candidate_notional_usd: 30.0,
            intended_lifetime_ms: 1_000,
            local_edge_feature: 0.4,
            pair_edge_feature: Some(1.0),
            primary_hedge_venue_id: Some("paradex".to_string()),
            backup_hedge_venue_id: None,
            no_live_flag: true,
        }
    }

    #[test]
    fn positive_pair_edge_can_be_rejected_by_negative_ev() {
        let components = EvComponents {
            p_fill: 0.2,
            p_hedge_success: 0.6,
            p_hedge_partial: 0.0,
            p_hedge_fail: 0.4,
            e_locked_edge: 1.0,
            e_partial_hedge_state: 0.0,
            e_residual_inventory_state: -2.0,
            e_adverse_selection: 0.05,
            e_queue_reset: 0.03,
            e_churn: 0.01,
            e_capital_funding: 0.0,
            e_tail_risk: 0.01,
        };
        let evaluation =
            evaluate_candidate(&candidate(), components, EvConfidence::default(), vec![])
                .expect("evaluation");

        assert_eq!(evaluation.decision, EvDecision::Reject);
        assert_eq!(evaluation.decision_reason_primary, "non_positive_lcb_ev");
        assert!(evaluation.ev_hat < 0.0);
    }

    #[test]
    fn discrete_size_selection_uses_lcb_argmax_not_edge_div_eta() {
        let selected = select_discrete_size(
            &candidate(),
            &[0.01, 0.02, 0.03],
            EvConfidence::default(),
            |size| {
                let edge = match size {
                    x if (x - 0.01).abs() < 1e-12 => 0.03,
                    x if (x - 0.02).abs() < 1e-12 => 0.05,
                    _ => -0.02,
                };
                (
                    EvComponents {
                        p_fill: 1.0,
                        p_hedge_success: 1.0,
                        p_hedge_partial: 0.0,
                        p_hedge_fail: 0.0,
                        e_locked_edge: edge,
                        e_partial_hedge_state: 0.0,
                        e_residual_inventory_state: 0.0,
                        e_adverse_selection: 0.0,
                        e_queue_reset: 0.0,
                        e_churn: 0.0,
                        e_capital_funding: 0.0,
                        e_tail_risk: 0.0,
                    },
                    vec![],
                )
            },
        )
        .expect("selected");

        assert_eq!(selected.selected_size_q, 0.02);
        assert_eq!(selected.decision, EvDecision::Admit);
    }

    #[test]
    fn no_live_flag_is_required() {
        let mut c = candidate();
        c.no_live_flag = false;
        let err = c.validate_nonlive_baseline().unwrap_err();
        assert!(err.contains("no_live_flag"));
    }

    #[test]
    fn hedge_probabilities_must_sum_to_one() {
        let components = EvComponents {
            p_fill: 0.1,
            p_hedge_success: 0.5,
            p_hedge_partial: 0.5,
            p_hedge_fail: 0.5,
            e_locked_edge: 0.0,
            e_partial_hedge_state: 0.0,
            e_residual_inventory_state: 0.0,
            e_adverse_selection: 0.0,
            e_queue_reset: 0.0,
            e_churn: 0.0,
            e_capital_funding: 0.0,
            e_tail_risk: 0.0,
        };
        let err = components.validate().unwrap_err();
        assert!(err.contains("sum to 1.0"));
    }
}
