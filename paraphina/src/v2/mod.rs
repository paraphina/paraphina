//! V2 arbitrage-aware decision evidence and gated admission skeleton.
//!
//! Shadow mode is intentionally read-only with respect to order flow. The
//! paper-admission tranche may only filter existing baseline MM order-creating
//! intents behind explicit gates. The live-canary admission tranche uses the
//! same baseline-intent filter only after live/canary profile gates are present.
//! It must not construct prices, sizes, client IDs, raw order IDs, venue handles,
//! or live transport requests.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

use serde::Serialize;

use crate::config::{V2DecisionMode, V2ShadowConfig};
use crate::mm::{MmLevel, MmQuote};
use crate::types::{OrderIntent, OrderPurpose, Side, TimestampMs};

const V2_SHADOW_SCHEMA_VERSION: u32 = 1;
const V2_ADMISSION_SCHEMA_VERSION: u32 = 1;

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
pub struct V2ShadowCandidateRanking {
    pub rank_index: usize,
    pub candidate_id: String,
    pub rank_status: &'static str,
    pub rank_score_microusd: i64,
    pub pair_edge_feature_usd: Option<f64>,
    pub pair_edge_feature_bps: Option<f64>,
    pub reference_candidate_id: Option<String>,
    pub reference_venue_index: Option<usize>,
    pub reference_venue_id: Option<String>,
    pub rank_tiebreak_key: String,
    pub feature_only: bool,
    pub admission_status: &'static str,
    pub admission_reason: &'static str,
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
    pub ranking_schema_version: u32,
    pub ranking_feature_only: bool,
    pub ranking_is_admission: bool,
    pub candidates: Vec<V2ShadowCandidate>,
    pub candidate_rankings: Vec<V2ShadowCandidateRanking>,
    pub pair_edges: Vec<V2PairEdgeSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2AdmissionGateState {
    pub enabled: bool,
    pub decision_mode_is_paper_admission: bool,
    pub decision_mode_is_live_canary_admission: bool,
    pub execution_mode_is_paper: bool,
    pub execution_mode_is_live: bool,
    pub pair_edge_enabled: bool,
    pub pair_conditioned_admission_enabled: bool,
    pub order_intent_enabled: bool,
    pub fast_hedge_disabled: bool,
    pub require_phase51_gate: bool,
    pub live_canary_admission_approved: bool,
    pub live_canary_mode_enabled: bool,
    pub live_canary_profile_metadata_present: bool,
    pub live_canary_max_position_present: bool,
    pub live_canary_max_gross_position_present: bool,
    pub live_canary_max_abs_venue_position_present: bool,
    pub live_canary_max_open_orders_present: bool,
    pub live_canary_post_only_enforced: bool,
    pub live_canary_reduce_only_not_enforced: bool,
}

impl V2AdmissionGateState {
    pub fn satisfied(&self) -> bool {
        let paper_authority = self.decision_mode_is_paper_admission && self.execution_mode_is_paper;
        let live_canary_authority = self.decision_mode_is_live_canary_admission
            && self.execution_mode_is_live
            && self.live_canary_admission_approved
            && self.live_canary_mode_enabled
            && self.live_canary_profile_metadata_present
            && self.live_canary_max_position_present
            && self.live_canary_max_gross_position_present
            && self.live_canary_max_abs_venue_position_present
            && self.live_canary_max_open_orders_present
            && self.live_canary_post_only_enforced
            && self.live_canary_reduce_only_not_enforced;

        self.enabled
            && (paper_authority || live_canary_authority)
            && self.pair_edge_enabled
            && self.pair_conditioned_admission_enabled
            && self.order_intent_enabled
            && self.fast_hedge_disabled
            && self.require_phase51_gate
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
pub struct V2AdmissionRuntimeContext {
    pub live_canary_mode_enabled: bool,
    pub live_canary_profile_metadata_present: bool,
    pub live_canary_max_position_present: bool,
    pub live_canary_max_gross_position_present: bool,
    pub live_canary_max_abs_venue_position_present: bool,
    pub live_canary_max_open_orders_present: bool,
    pub live_canary_post_only_enforced: bool,
    pub live_canary_reduce_only_not_enforced: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2AdmittedCandidate {
    pub candidate_id: String,
    pub venue_index: usize,
    pub venue_id: String,
    pub side: Side,
    pub rank_index: usize,
    pub rank_score_microusd: i64,
    pub pair_edge_feature_usd: Option<f64>,
    pub pair_edge_feature_bps: Option<f64>,
    pub reference_candidate_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2AdmissionDecision {
    pub event_type: &'static str,
    pub schema_version: u32,
    pub telemetry_schema_version: u32,
    pub now_ms: TimestampMs,
    pub decision_mode: &'static str,
    pub execution_mode: String,
    pub authority_scope: &'static str,
    pub admission_status: &'static str,
    pub admission_reason: &'static str,
    pub can_filter_existing_intents: bool,
    pub can_create_new_intents: bool,
    pub can_mutate_live_orders: bool,
    pub order_intent_output_count: usize,
    pub baseline_plan_intent_count: usize,
    pub baseline_mm_order_creating_intent_count: usize,
    pub suppressed_mm_order_creating_intent_count: usize,
    pub pair_edge_is_admission: bool,
    pub pressure_complete_claim: bool,
    pub blocker_cleared: bool,
    pub gate_state: V2AdmissionGateState,
    pub ranking_schema_version: u32,
    pub ranking_feature_only: bool,
    pub ranking_is_admission: bool,
    pub pair_edges: Vec<V2PairEdgeSnapshot>,
    pub admitted_candidates: Vec<V2AdmittedCandidate>,
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

pub fn v2_paper_admission_mode_requested(config: &V2ShadowConfig) -> bool {
    config.enabled && matches!(config.decision_mode, V2DecisionMode::PaperAdmission)
}

pub fn v2_live_canary_admission_mode_requested(config: &V2ShadowConfig) -> bool {
    config.enabled && matches!(config.decision_mode, V2DecisionMode::LiveCanaryAdmission)
}

pub fn v2_admission_mode_requested(config: &V2ShadowConfig) -> bool {
    v2_paper_admission_mode_requested(config) || v2_live_canary_admission_mode_requested(config)
}

fn env_flag_true(name: &str) -> bool {
    std::env::var(name)
        .map(|raw| {
            matches!(
                raw.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes"
            )
        })
        .unwrap_or(false)
}

fn env_present(name: &str) -> bool {
    std::env::var(name)
        .map(|raw| !raw.trim().is_empty())
        .unwrap_or(false)
}

impl V2AdmissionRuntimeContext {
    pub fn from_env() -> Self {
        let live_canary_profile_metadata_present =
            env_present("PARAPHINA_RUNTIME_CANARY_PROFILE_PATH")
                && env_present("PARAPHINA_RUNTIME_CANARY_PROFILE_SHA256");
        Self {
            live_canary_mode_enabled: env_flag_true("PARAPHINA_CANARY_MODE")
                || live_canary_profile_metadata_present,
            live_canary_profile_metadata_present,
            live_canary_max_position_present: env_present(
                "PARAPHINA_RUNTIME_CANARY_MAX_POSITION_TAO",
            ) || env_present("PARAPHINA_CANARY_MAX_POSITION_TAO"),
            live_canary_max_gross_position_present: env_present(
                "PARAPHINA_RUNTIME_CANARY_MAX_GROSS_POSITION_TAO",
            ) || env_present(
                "PARAPHINA_CANARY_MAX_GROSS_POSITION_TAO",
            ),
            live_canary_max_abs_venue_position_present: env_present(
                "PARAPHINA_RUNTIME_CANARY_MAX_ABS_VENUE_POSITION_TAO",
            ) || env_present(
                "PARAPHINA_CANARY_MAX_ABS_VENUE_POSITION_TAO",
            ),
            live_canary_max_open_orders_present: env_present(
                "PARAPHINA_RUNTIME_CANARY_MAX_OPEN_ORDERS",
            ) || env_present(
                "PARAPHINA_CANARY_MAX_OPEN_ORDERS",
            ),
            live_canary_post_only_enforced: env_flag_true(
                "PARAPHINA_RUNTIME_CANARY_ENFORCE_POST_ONLY",
            ) || env_flag_true(
                "PARAPHINA_CANARY_ENFORCE_POST_ONLY",
            ),
            live_canary_reduce_only_not_enforced: !env_flag_true(
                "PARAPHINA_RUNTIME_CANARY_ENFORCE_REDUCE_ONLY",
            ) && !env_flag_true(
                "PARAPHINA_CANARY_ENFORCE_REDUCE_ONLY",
            ),
        }
    }
}

pub fn admission_gate_state(
    config: &V2ShadowConfig,
    execution_mode: &str,
    runtime_context: &V2AdmissionRuntimeContext,
) -> V2AdmissionGateState {
    V2AdmissionGateState {
        enabled: config.enabled,
        decision_mode_is_paper_admission: matches!(
            config.decision_mode,
            V2DecisionMode::PaperAdmission
        ),
        decision_mode_is_live_canary_admission: matches!(
            config.decision_mode,
            V2DecisionMode::LiveCanaryAdmission
        ),
        execution_mode_is_paper: execution_mode == "paper",
        execution_mode_is_live: execution_mode == "live",
        pair_edge_enabled: config.pair_edge_enabled,
        pair_conditioned_admission_enabled: config.pair_conditioned_admission_enabled,
        order_intent_enabled: config.order_intent_enabled,
        fast_hedge_disabled: !config.fast_hedge_enabled,
        require_phase51_gate: config.require_phase51_gate,
        live_canary_admission_approved: config.live_canary_admission_approved,
        live_canary_mode_enabled: runtime_context.live_canary_mode_enabled,
        live_canary_profile_metadata_present: runtime_context.live_canary_profile_metadata_present,
        live_canary_max_position_present: runtime_context.live_canary_max_position_present,
        live_canary_max_gross_position_present: runtime_context
            .live_canary_max_gross_position_present,
        live_canary_max_abs_venue_position_present: runtime_context
            .live_canary_max_abs_venue_position_present,
        live_canary_max_open_orders_present: runtime_context.live_canary_max_open_orders_present,
        live_canary_post_only_enforced: runtime_context.live_canary_post_only_enforced,
        live_canary_reduce_only_not_enforced: runtime_context.live_canary_reduce_only_not_enforced,
    }
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
    let candidate_rankings = rank_shadow_candidates(&candidates);
    let pair_edges = if config.pair_edge_enabled {
        vec![build_pair_edge_snapshot(&candidates, true)]
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
        ranking_schema_version: 1,
        ranking_feature_only: true,
        ranking_is_admission: false,
        candidates,
        candidate_rankings,
        pair_edges,
    })
}

pub fn evaluate_admission_decision_with_context(
    config: &V2ShadowConfig,
    execution_mode: &str,
    now_ms: TimestampMs,
    baseline_plan_intents: &[OrderIntent],
    runtime_context: &V2AdmissionRuntimeContext,
) -> Option<V2AdmissionDecision> {
    if !v2_admission_mode_requested(config) {
        return None;
    }

    let gate_state = admission_gate_state(config, execution_mode, runtime_context);
    let candidates = extract_shadow_candidates_from_intents(baseline_plan_intents);
    let rankings = rank_shadow_candidates(&candidates);
    let pair_edges = if config.pair_edge_enabled {
        vec![build_pair_edge_snapshot(&candidates, false)]
    } else {
        Vec::new()
    };
    let positive_pair_edge = pair_edges
        .first()
        .and_then(|edge| edge.edge_usd)
        .is_some_and(|edge| edge > 0.0);

    let admitted_candidates = if gate_state.satisfied() {
        rankings
            .iter()
            .filter(|ranking| ranking.rank_status == "scored" && ranking.rank_score_microusd > 0)
            .filter_map(|ranking| {
                let candidate = candidates
                    .iter()
                    .find(|candidate| candidate.candidate_id == ranking.candidate_id)?;
                Some(V2AdmittedCandidate {
                    candidate_id: ranking.candidate_id.clone(),
                    venue_index: candidate.venue_index,
                    venue_id: candidate.venue_id.clone(),
                    side: candidate.side,
                    rank_index: ranking.rank_index,
                    rank_score_microusd: ranking.rank_score_microusd,
                    pair_edge_feature_usd: ranking.pair_edge_feature_usd,
                    pair_edge_feature_bps: ranking.pair_edge_feature_bps,
                    reference_candidate_id: ranking.reference_candidate_id.clone(),
                })
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    let baseline_mm_count = count_baseline_mm_order_creating_intents(baseline_plan_intents);
    let gate_satisfied = gate_state.satisfied();
    let has_admitted_candidates = !admitted_candidates.is_empty();
    let live_canary_mode = matches!(config.decision_mode, V2DecisionMode::LiveCanaryAdmission);
    let admission_reason = match (
        live_canary_mode,
        gate_satisfied,
        has_admitted_candidates,
        positive_pair_edge,
    ) {
        (true, false, _, _) => "live_canary_admission_gate_not_satisfied",
        (false, false, _, _) => "paper_admission_gate_not_satisfied",
        (_, true, false, _) => "no_positive_ranked_candidates",
        (true, true, true, true) => "live_canary_positive_pair_edge_ranked_admission",
        (false, true, true, true) => "paper_positive_pair_edge_ranked_admission",
        (true, true, true, false) => "live_canary_positive_ranked_admission",
        (false, true, true, false) => "paper_positive_ranked_admission",
    };
    let output_count = admitted_candidates.len();
    let admitted = output_count > 0;
    let suppressed_count = if gate_satisfied {
        baseline_mm_count.saturating_sub(output_count)
    } else {
        0
    };
    Some(V2AdmissionDecision {
        event_type: "V2_ADMISSION_DECISION",
        schema_version: V2_ADMISSION_SCHEMA_VERSION,
        telemetry_schema_version: config.telemetry_schema_version,
        now_ms,
        decision_mode: config.decision_mode.as_str(),
        execution_mode: execution_mode.to_string(),
        authority_scope: if live_canary_mode {
            "live_canary_ranked_admission"
        } else {
            "paper_only"
        },
        admission_status: if admitted { "ADMITTED" } else { "HOLD" },
        admission_reason,
        can_filter_existing_intents: gate_satisfied,
        can_create_new_intents: false,
        can_mutate_live_orders: false,
        order_intent_output_count: output_count,
        baseline_plan_intent_count: baseline_plan_intents.len(),
        baseline_mm_order_creating_intent_count: baseline_mm_count,
        suppressed_mm_order_creating_intent_count: suppressed_count,
        pair_edge_is_admission: admitted && positive_pair_edge,
        pressure_complete_claim: false,
        blocker_cleared: false,
        gate_state,
        ranking_schema_version: 1,
        ranking_feature_only: false,
        ranking_is_admission: admitted,
        pair_edges,
        admitted_candidates,
    })
}

pub fn evaluate_paper_admission_decision(
    config: &V2ShadowConfig,
    execution_mode: &str,
    now_ms: TimestampMs,
    baseline_plan_intents: &[OrderIntent],
) -> Option<V2AdmissionDecision> {
    if !v2_paper_admission_mode_requested(config) {
        return None;
    }
    evaluate_admission_decision_with_context(
        config,
        execution_mode,
        now_ms,
        baseline_plan_intents,
        &V2AdmissionRuntimeContext::default(),
    )
}

pub fn evaluate_admission_decision(
    config: &V2ShadowConfig,
    execution_mode: &str,
    now_ms: TimestampMs,
    baseline_plan_intents: &[OrderIntent],
) -> Option<V2AdmissionDecision> {
    evaluate_admission_decision_with_context(
        config,
        execution_mode,
        now_ms,
        baseline_plan_intents,
        &V2AdmissionRuntimeContext::from_env(),
    )
}

pub fn emit_paper_admission_decision(
    config: &V2ShadowConfig,
    execution_mode: &str,
    now_ms: TimestampMs,
    baseline_plan_intents: &[OrderIntent],
) -> Result<Option<V2AdmissionDecision>, V2ShadowError> {
    let Some(decision) =
        evaluate_paper_admission_decision(config, execution_mode, now_ms, baseline_plan_intents)
    else {
        return Ok(None);
    };
    append_json_line(Path::new(&config.output_path), &decision)?;
    Ok(Some(decision))
}

pub fn emit_admission_decision(
    config: &V2ShadowConfig,
    execution_mode: &str,
    now_ms: TimestampMs,
    baseline_plan_intents: &[OrderIntent],
) -> Result<Option<V2AdmissionDecision>, V2ShadowError> {
    emit_admission_decision_with_context(
        config,
        execution_mode,
        now_ms,
        baseline_plan_intents,
        &V2AdmissionRuntimeContext::from_env(),
    )
}

pub fn emit_admission_decision_with_context(
    config: &V2ShadowConfig,
    execution_mode: &str,
    now_ms: TimestampMs,
    baseline_plan_intents: &[OrderIntent],
    runtime_context: &V2AdmissionRuntimeContext,
) -> Result<Option<V2AdmissionDecision>, V2ShadowError> {
    let Some(decision) = evaluate_admission_decision_with_context(
        config,
        execution_mode,
        now_ms,
        baseline_plan_intents,
        runtime_context,
    ) else {
        return Ok(None);
    };
    append_json_line(Path::new(&config.output_path), &decision)?;
    Ok(Some(decision))
}

pub fn apply_admission_filter(
    config: &V2ShadowConfig,
    execution_mode: &str,
    now_ms: TimestampMs,
    intents: &mut Vec<OrderIntent>,
) -> Result<Option<V2AdmissionDecision>, V2ShadowError> {
    apply_admission_filter_with_context(
        config,
        execution_mode,
        now_ms,
        intents,
        &V2AdmissionRuntimeContext::from_env(),
    )
}

pub fn apply_admission_filter_with_context(
    config: &V2ShadowConfig,
    execution_mode: &str,
    now_ms: TimestampMs,
    intents: &mut Vec<OrderIntent>,
    runtime_context: &V2AdmissionRuntimeContext,
) -> Result<Option<V2AdmissionDecision>, V2ShadowError> {
    let Some(decision) = emit_admission_decision_with_context(
        config,
        execution_mode,
        now_ms,
        intents,
        runtime_context,
    )?
    else {
        return Ok(None);
    };
    if !decision.gate_state.satisfied() {
        return Ok(Some(decision));
    }

    let admitted = decision
        .admitted_candidates
        .iter()
        .map(|candidate| candidate.candidate_id.as_str())
        .collect::<std::collections::HashSet<_>>();
    let mut idx = 0usize;
    intents.retain(|intent| {
        let keep = match intent {
            OrderIntent::Place(place) if place.purpose == OrderPurpose::Mm => admitted.contains(
                format!(
                    "v2_shadow_intent_v1:{}:{}:{}:{}",
                    place.venue_index,
                    place.venue_id,
                    place.side.as_v2_str(),
                    idx
                )
                .as_str(),
            ),
            OrderIntent::Replace(replace) if replace.purpose == OrderPurpose::Mm => admitted
                .contains(
                    format!(
                        "v2_shadow_intent_v1:{}:{}:{}:{}",
                        replace.venue_index,
                        replace.venue_id,
                        replace.side.as_v2_str(),
                        idx
                    )
                    .as_str(),
                ),
            OrderIntent::Place(_)
            | OrderIntent::Replace(_)
            | OrderIntent::Cancel(_)
            | OrderIntent::CancelAll(_) => true,
        };
        idx += 1;
        keep
    });
    Ok(Some(decision))
}

pub fn apply_paper_admission_filter(
    config: &V2ShadowConfig,
    execution_mode: &str,
    now_ms: TimestampMs,
    intents: &mut Vec<OrderIntent>,
) -> Result<Option<V2AdmissionDecision>, V2ShadowError> {
    if !v2_paper_admission_mode_requested(config) {
        return Ok(None);
    }
    let Some(decision) = emit_paper_admission_decision(config, execution_mode, now_ms, intents)?
    else {
        return Ok(None);
    };
    if !decision.gate_state.satisfied() {
        return Ok(Some(decision));
    }

    let admitted = decision
        .admitted_candidates
        .iter()
        .map(|candidate| candidate.candidate_id.as_str())
        .collect::<std::collections::HashSet<_>>();
    let mut idx = 0usize;
    intents.retain(|intent| {
        let keep = match intent {
            OrderIntent::Place(place) if place.purpose == OrderPurpose::Mm => admitted.contains(
                format!(
                    "v2_shadow_intent_v1:{}:{}:{}:{}",
                    place.venue_index,
                    place.venue_id,
                    place.side.as_v2_str(),
                    idx
                )
                .as_str(),
            ),
            OrderIntent::Replace(replace) if replace.purpose == OrderPurpose::Mm => admitted
                .contains(
                    format!(
                        "v2_shadow_intent_v1:{}:{}:{}:{}",
                        replace.venue_index,
                        replace.venue_id,
                        replace.side.as_v2_str(),
                        idx
                    )
                    .as_str(),
                ),
            OrderIntent::Place(_)
            | OrderIntent::Replace(_)
            | OrderIntent::Cancel(_)
            | OrderIntent::CancelAll(_) => true,
        };
        idx += 1;
        keep
    });
    Ok(Some(decision))
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

fn rank_shadow_candidates(candidates: &[V2ShadowCandidate]) -> Vec<V2ShadowCandidateRanking> {
    let mut ranked = candidates
        .iter()
        .filter(|candidate| candidate.price.is_finite())
        .map(|candidate| {
            let reference = best_same_side_reference(candidate, candidates);
            let (rank_status, rank_score_microusd, pair_edge_feature_usd, pair_edge_feature_bps) =
                match reference {
                    Some(reference) => {
                        let feature_usd = match candidate.side {
                            Side::Buy => reference.price - candidate.price,
                            Side::Sell => candidate.price - reference.price,
                        };
                        let midpoint = (candidate.price + reference.price) / 2.0;
                        let feature_bps = (midpoint.is_finite() && midpoint > 0.0)
                            .then_some(feature_usd / midpoint * 10_000.0);
                        (
                            "scored",
                            (feature_usd * 1_000_000.0).round() as i64,
                            Some(feature_usd),
                            feature_bps,
                        )
                    }
                    None => ("missing_cross_venue_reference", 0, None, None),
                };
            let reference_candidate_id = reference.map(|reference| reference.candidate_id.clone());
            let reference_venue_index = reference.map(|reference| reference.venue_index);
            let reference_venue_id = reference.map(|reference| reference.venue_id.clone());
            let linkage_tiebreak = if candidate.target_linkage_state == "present_redacted" {
                0
            } else {
                1
            };
            V2ShadowCandidateRanking {
                rank_index: 0,
                candidate_id: candidate.candidate_id.clone(),
                rank_status,
                rank_score_microusd,
                pair_edge_feature_usd,
                pair_edge_feature_bps,
                reference_candidate_id,
                reference_venue_index,
                reference_venue_id,
                rank_tiebreak_key: format!(
                    "{}:{:04}:{}:{}",
                    linkage_tiebreak,
                    candidate.venue_index,
                    candidate.side.as_v2_str(),
                    candidate.candidate_id
                ),
                feature_only: true,
                admission_status: V2ShadowAdmissionStatus::Hold.as_str(),
                admission_reason: "shadow_only_no_order_authority",
            }
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|lhs, rhs| {
        status_sort_key(lhs.rank_status)
            .cmp(&status_sort_key(rhs.rank_status))
            .then_with(|| rhs.rank_score_microusd.cmp(&lhs.rank_score_microusd))
            .then_with(|| lhs.rank_tiebreak_key.cmp(&rhs.rank_tiebreak_key))
    });
    for (idx, ranking) in ranked.iter_mut().enumerate() {
        ranking.rank_index = idx + 1;
    }
    ranked
}

fn best_same_side_reference<'a>(
    candidate: &V2ShadowCandidate,
    candidates: &'a [V2ShadowCandidate],
) -> Option<&'a V2ShadowCandidate> {
    candidates
        .iter()
        .filter(|reference| {
            reference.side == candidate.side
                && reference.venue_id != candidate.venue_id
                && reference.candidate_id != candidate.candidate_id
                && reference.price.is_finite()
        })
        .max_by(|lhs, rhs| match candidate.side {
            Side::Buy => lhs
                .price
                .total_cmp(&rhs.price)
                .then_with(|| rhs.candidate_id.cmp(&lhs.candidate_id)),
            Side::Sell => rhs
                .price
                .total_cmp(&lhs.price)
                .then_with(|| rhs.candidate_id.cmp(&lhs.candidate_id)),
        })
}

fn status_sort_key(status: &str) -> u8 {
    if status == "scored" {
        0
    } else {
        1
    }
}

fn build_pair_edge_snapshot(
    candidates: &[V2ShadowCandidate],
    feature_only: bool,
) -> V2PairEdgeSnapshot {
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
            feature_only,
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
            feature_only,
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
        feature_only,
        invalid_reason: None,
    }
}

fn append_shadow_decision(path: &Path, decision: &V2ShadowDecision) -> Result<(), V2ShadowError> {
    append_json_line(path, decision)
}

fn append_json_line<T: Serialize>(path: &Path, value: &T) -> Result<(), V2ShadowError> {
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
    serde_json::to_writer(&mut file, value)
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

    fn paper_admission_config() -> V2ShadowConfig {
        V2ShadowConfig {
            enabled: true,
            decision_mode: V2DecisionMode::PaperAdmission,
            pair_edge_enabled: true,
            pair_conditioned_admission_enabled: true,
            order_intent_enabled: true,
            require_phase51_gate: true,
            ..V2ShadowConfig::default()
        }
    }

    fn live_canary_admission_config() -> V2ShadowConfig {
        let output_path = std::env::temp_dir().join(format!(
            "paraphina_v2_live_canary_admission_test_{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&output_path);
        V2ShadowConfig {
            enabled: true,
            decision_mode: V2DecisionMode::LiveCanaryAdmission,
            output_path: output_path.display().to_string(),
            pair_edge_enabled: true,
            pair_conditioned_admission_enabled: true,
            live_canary_admission_approved: true,
            order_intent_enabled: true,
            require_phase51_gate: true,
            ..V2ShadowConfig::default()
        }
    }

    fn live_canary_runtime_context() -> V2AdmissionRuntimeContext {
        V2AdmissionRuntimeContext {
            live_canary_mode_enabled: true,
            live_canary_profile_metadata_present: true,
            live_canary_max_position_present: true,
            live_canary_max_gross_position_present: true,
            live_canary_max_abs_venue_position_present: true,
            live_canary_max_open_orders_present: true,
            live_canary_post_only_enforced: true,
            live_canary_reduce_only_not_enforced: true,
        }
    }

    fn mm_place(venue_index: usize, venue_id: &'static str, side: Side, price: f64) -> OrderIntent {
        OrderIntent::Place(crate::types::PlaceOrderIntent {
            venue_index,
            venue_id: Arc::from(venue_id),
            side,
            price,
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
        })
    }

    fn mm_replace(
        venue_index: usize,
        venue_id: &'static str,
        side: Side,
        price: f64,
    ) -> OrderIntent {
        OrderIntent::Replace(crate::types::ReplaceOrderIntent {
            venue_index,
            venue_id: Arc::from(venue_id),
            side,
            price,
            size: 0.01,
            purpose: OrderPurpose::Mm,
            time_in_force: crate::types::TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            order_id: "raw-replace-order-id-must-not-emit".to_string(),
            client_order_id: Some("raw-replace-client-id-must-not-emit".to_string()),
            phase51_target_key: Some(crate::types::Phase51ForwardRefreshTargetKey {
                canonical_group_id: "raw-replace-group-must-not-emit".to_string(),
                order_key: "raw-replace-order-must-not-emit".to_string(),
            }),
        })
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
        assert_eq!(decision.candidate_rankings.len(), 2);
        assert_eq!(decision.ranking_schema_version, 1);
        assert!(decision.ranking_feature_only);
        assert!(!decision.ranking_is_admission);
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
        assert_eq!(decision.candidate_rankings.len(), 10);
        assert_eq!(decision.ranking_schema_version, 1);
        assert!(decision.ranking_feature_only);
        assert!(!decision.ranking_is_admission);
        assert_eq!(
            decision.candidate_rankings[0].candidate_id,
            "v2_shadow_v1:0:extended:buy"
        );
        assert_eq!(decision.candidate_rankings[0].rank_index, 1);
        assert_eq!(decision.candidate_rankings[0].rank_status, "scored");
        assert_eq!(
            decision.candidate_rankings[0].rank_score_microusd,
            10_000_000
        );
        assert_eq!(
            decision.candidate_rankings[0]
                .reference_candidate_id
                .as_deref(),
            Some("v2_shadow_v1:1:hyperliquid:buy")
        );
        assert!(decision.candidate_rankings[0].feature_only);
        assert_eq!(decision.candidate_rankings[0].admission_status, "HOLD");
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
        assert_eq!(decision.candidate_rankings.len(), before.len());
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
        assert_eq!(decision.candidate_rankings.len(), intents.len());
        assert_eq!(
            decision.candidate_rankings[0].candidate_id,
            "v2_shadow_intent_v1:2:lighter:buy:0"
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

    #[test]
    fn v2_paper_admission_filters_existing_mm_intents_only() {
        let config = paper_admission_config();
        let mut intents = vec![
            mm_place(0, "extended", Side::Buy, 99.0),
            mm_place(1, "hyperliquid", Side::Buy, 100.0),
            mm_place(2, "aster", Side::Sell, 98.0),
            mm_place(3, "lighter", Side::Sell, 105.0),
            OrderIntent::Cancel(crate::types::CancelOrderIntent {
                venue_index: 4,
                venue_id: Arc::from("paradex"),
                order_id: "raw-cancel-order-id-must-not-emit".to_string(),
            }),
        ];

        let decision =
            apply_paper_admission_filter(&config, "paper", 4_000, &mut intents).expect("filter");
        let decision = decision.expect("decision");

        assert_eq!(decision.event_type, "V2_ADMISSION_DECISION");
        assert_eq!(decision.authority_scope, "paper_only");
        assert_eq!(decision.admission_status, "ADMITTED");
        assert!(decision.can_filter_existing_intents);
        assert!(!decision.can_create_new_intents);
        assert!(!decision.can_mutate_live_orders);
        assert!(decision.pair_edge_is_admission);
        assert!(decision.ranking_is_admission);
        assert!(!decision.ranking_feature_only);
        assert!(!decision.pressure_complete_claim);
        assert!(!decision.blocker_cleared);
        assert_eq!(decision.baseline_mm_order_creating_intent_count, 4);
        assert_eq!(decision.order_intent_output_count, 2);
        assert_eq!(decision.suppressed_mm_order_creating_intent_count, 2);
        assert_eq!(intents.len(), 3, "two MM intents plus cancel retained");
        assert!(
            matches!(&intents[0], OrderIntent::Place(place) if place.venue_id.as_ref() == "extended" && place.side == Side::Buy)
        );
        assert!(
            matches!(&intents[1], OrderIntent::Place(place) if place.venue_id.as_ref() == "lighter" && place.side == Side::Sell)
        );
        assert!(matches!(&intents[2], OrderIntent::Cancel(_)));

        let serialized = serde_json::to_string(&decision).expect("serialize");
        assert!(!serialized.contains("raw-client-id-must-not-emit"));
        assert!(!serialized.contains("raw-cancel-order-id-must-not-emit"));
        assert!(!serialized.contains("raw-group-must-not-emit"));
        assert!(!serialized.contains("raw-order-must-not-emit"));
    }

    #[test]
    fn v2_live_canary_admission_filters_existing_mm_intents_only_under_all_gates() {
        let config = live_canary_admission_config();
        let context = live_canary_runtime_context();
        let mut intents = vec![
            mm_place(0, "extended", Side::Buy, 99.0),
            mm_place(1, "hyperliquid", Side::Buy, 100.0),
            mm_place(2, "aster", Side::Sell, 98.0),
            mm_place(3, "lighter", Side::Sell, 105.0),
            OrderIntent::Cancel(crate::types::CancelOrderIntent {
                venue_index: 4,
                venue_id: Arc::from("paradex"),
                order_id: "raw-cancel-order-id-must-not-emit".to_string(),
            }),
        ];

        let decision =
            apply_admission_filter_with_context(&config, "live", 4_500, &mut intents, &context)
                .expect("filter")
                .expect("decision");

        assert_eq!(decision.event_type, "V2_ADMISSION_DECISION");
        assert_eq!(decision.decision_mode, "live_canary_admission");
        assert_eq!(decision.execution_mode, "live");
        assert_eq!(decision.authority_scope, "live_canary_ranked_admission");
        assert_eq!(decision.admission_status, "ADMITTED");
        assert!(decision.can_filter_existing_intents);
        assert!(!decision.can_create_new_intents);
        assert!(!decision.can_mutate_live_orders);
        assert!(decision.gate_state.satisfied());
        assert!(decision.gate_state.live_canary_admission_approved);
        assert!(decision.gate_state.live_canary_profile_metadata_present);
        assert!(decision.gate_state.live_canary_post_only_enforced);
        assert!(decision.gate_state.live_canary_reduce_only_not_enforced);
        assert_eq!(decision.baseline_mm_order_creating_intent_count, 4);
        assert_eq!(decision.order_intent_output_count, 2);
        assert_eq!(decision.suppressed_mm_order_creating_intent_count, 2);
        assert!(!decision.pressure_complete_claim);
        assert!(!decision.blocker_cleared);
        assert_eq!(intents.len(), 3, "two MM intents plus cancel retained");
        assert!(
            matches!(&intents[0], OrderIntent::Place(place) if place.venue_id.as_ref() == "extended" && place.side == Side::Buy)
        );
        assert!(
            matches!(&intents[1], OrderIntent::Place(place) if place.venue_id.as_ref() == "lighter" && place.side == Side::Sell)
        );
        assert!(matches!(&intents[2], OrderIntent::Cancel(_)));

        let serialized = serde_json::to_string(&decision).expect("serialize");
        assert!(!serialized.contains("raw-client-id-must-not-emit"));
        assert!(!serialized.contains("raw-cancel-order-id-must-not-emit"));
        assert!(!serialized.contains("raw-group-must-not-emit"));
        assert!(!serialized.contains("raw-order-must-not-emit"));
    }

    #[test]
    fn v2_live_canary_admission_missing_runtime_gate_holds_without_filtering() {
        let config = live_canary_admission_config();
        let mut context = live_canary_runtime_context();
        context.live_canary_profile_metadata_present = false;
        let mut intents = vec![
            mm_place(0, "extended", Side::Buy, 99.0),
            mm_place(1, "hyperliquid", Side::Buy, 100.0),
        ];

        let decision =
            apply_admission_filter_with_context(&config, "live", 4_500, &mut intents, &context)
                .expect("filter")
                .expect("decision");

        assert_eq!(decision.admission_status, "HOLD");
        assert_eq!(
            decision.admission_reason,
            "live_canary_admission_gate_not_satisfied"
        );
        assert!(!decision.gate_state.satisfied());
        assert!(!decision.can_filter_existing_intents);
        assert!(!decision.can_create_new_intents);
        assert!(!decision.can_mutate_live_orders);
        assert_eq!(decision.suppressed_mm_order_creating_intent_count, 0);
        assert_eq!(intents.len(), 2);
    }

    #[test]
    fn v2_live_canary_admission_rejects_fast_hedge_authority() {
        let mut config = live_canary_admission_config();
        config.fast_hedge_enabled = true;
        let context = live_canary_runtime_context();
        let decision = evaluate_admission_decision_with_context(
            &config,
            "live",
            4_500,
            &[mm_place(0, "extended", Side::Buy, 99.0)],
            &context,
        )
        .expect("decision");

        assert_eq!(decision.admission_status, "HOLD");
        assert_eq!(
            decision.admission_reason,
            "live_canary_admission_gate_not_satisfied"
        );
        assert!(!decision.gate_state.fast_hedge_disabled);
        assert!(!decision.gate_state.satisfied());
        assert!(!decision.can_create_new_intents);
        assert!(!decision.blocker_cleared);
    }

    #[test]
    fn v2_live_canary_admission_does_not_filter_outside_live_execution_mode() {
        let config = live_canary_admission_config();
        let context = live_canary_runtime_context();
        let mut intents = vec![
            mm_place(0, "extended", Side::Buy, 99.0),
            mm_place(1, "hyperliquid", Side::Buy, 100.0),
        ];

        let decision =
            apply_admission_filter_with_context(&config, "paper", 4_500, &mut intents, &context)
                .expect("filter")
                .expect("decision");

        assert_eq!(decision.admission_status, "HOLD");
        assert_eq!(
            decision.admission_reason,
            "live_canary_admission_gate_not_satisfied"
        );
        assert!(!decision.gate_state.execution_mode_is_live);
        assert!(!decision.gate_state.satisfied());
        assert_eq!(intents.len(), 2);
    }

    #[test]
    fn v2_paper_admission_missing_gate_holds_without_filtering() {
        let mut config = paper_admission_config();
        config.order_intent_enabled = false;
        let mut intents = vec![
            mm_place(0, "extended", Side::Buy, 99.0),
            mm_place(1, "hyperliquid", Side::Buy, 100.0),
        ];

        let decision =
            apply_paper_admission_filter(&config, "paper", 4_000, &mut intents).expect("filter");
        let decision = decision.expect("decision");

        assert_eq!(decision.admission_status, "HOLD");
        assert_eq!(
            decision.admission_reason,
            "paper_admission_gate_not_satisfied"
        );
        assert_eq!(decision.order_intent_output_count, 0);
        assert_eq!(decision.suppressed_mm_order_creating_intent_count, 0);
        assert!(!decision.gate_state.satisfied());
        assert_eq!(
            intents.len(),
            2,
            "missing gate must not grant filtering authority"
        );
    }

    #[test]
    fn v2_paper_admission_without_positive_ranked_candidates_suppresses_mm_intents() {
        let config = paper_admission_config();
        let mut intents = vec![
            mm_place(0, "extended", Side::Buy, 99.0),
            mm_place(1, "hyperliquid", Side::Sell, 101.0),
        ];

        let decision =
            apply_paper_admission_filter(&config, "paper", 4_000, &mut intents).expect("filter");
        let decision = decision.expect("decision");

        assert!(decision.gate_state.satisfied());
        assert_eq!(decision.admission_status, "HOLD");
        assert_eq!(decision.admission_reason, "no_positive_ranked_candidates");
        assert_eq!(decision.order_intent_output_count, 0);
        assert_eq!(decision.suppressed_mm_order_creating_intent_count, 2);
        assert!(intents.is_empty());
    }

    #[test]
    fn v2_paper_admission_allows_ranked_admission_without_positive_pair_edge() {
        let config = paper_admission_config();
        let mut intents = vec![
            mm_place(0, "extended", Side::Buy, 99.0),
            mm_place(1, "hyperliquid", Side::Buy, 100.0),
        ];

        let decision =
            apply_paper_admission_filter(&config, "paper", 4_000, &mut intents).expect("filter");
        let decision = decision.expect("decision");

        assert!(decision.gate_state.satisfied());
        assert_eq!(decision.admission_status, "ADMITTED");
        assert_eq!(decision.admission_reason, "paper_positive_ranked_admission");
        assert!(!decision.pair_edge_is_admission);
        assert!(decision.ranking_is_admission);
        assert_eq!(decision.order_intent_output_count, 1);
        assert_eq!(decision.suppressed_mm_order_creating_intent_count, 1);
        assert_eq!(intents.len(), 1);
        assert!(
            matches!(&intents[0], OrderIntent::Place(place) if place.venue_id.as_ref() == "extended" && place.side == Side::Buy)
        );
    }

    #[test]
    fn v2_paper_admission_filters_mm_replace_and_leaves_non_mm_intents() {
        let config = paper_admission_config();
        let mut intents = vec![
            mm_replace(0, "extended", Side::Buy, 99.0),
            mm_replace(1, "hyperliquid", Side::Buy, 100.0),
            mm_place(2, "aster", Side::Sell, 98.0),
            mm_replace(3, "lighter", Side::Sell, 105.0),
            OrderIntent::Place(crate::types::PlaceOrderIntent {
                venue_index: 4,
                venue_id: Arc::from("paradex"),
                side: Side::Buy,
                price: 97.0,
                size: 0.01,
                purpose: OrderPurpose::Hedge,
                time_in_force: crate::types::TimeInForce::Ioc,
                post_only: false,
                reduce_only: true,
                client_order_id: Some("raw-hedge-client-id-must-not-emit".to_string()),
                phase51_target_key: None,
            }),
        ];

        let decision =
            apply_paper_admission_filter(&config, "paper", 4_000, &mut intents).expect("filter");
        let decision = decision.expect("decision");

        assert_eq!(decision.admission_status, "ADMITTED");
        assert_eq!(decision.order_intent_output_count, 2);
        assert_eq!(
            intents.len(),
            3,
            "two admitted MM intents plus non-MM retained"
        );
        assert!(
            matches!(&intents[0], OrderIntent::Replace(replace) if replace.venue_id.as_ref() == "extended" && replace.side == Side::Buy)
        );
        assert!(
            matches!(&intents[1], OrderIntent::Replace(replace) if replace.venue_id.as_ref() == "lighter" && replace.side == Side::Sell)
        );
        assert!(
            matches!(&intents[2], OrderIntent::Place(place) if place.purpose == OrderPurpose::Hedge)
        );

        let serialized = serde_json::to_string(&decision).expect("serialize");
        assert!(!serialized.contains("raw-replace-order-id-must-not-emit"));
        assert!(!serialized.contains("raw-replace-client-id-must-not-emit"));
        assert!(!serialized.contains("raw-hedge-client-id-must-not-emit"));
        assert!(!serialized.contains("raw-replace-group-must-not-emit"));
        assert!(!serialized.contains("raw-replace-order-must-not-emit"));
    }

    #[test]
    fn v2_paper_admission_inactive_outside_paper_admission_mode() {
        let config = shadow_config();
        let mut intents = vec![mm_place(0, "extended", Side::Buy, 99.0)];

        let decision =
            apply_paper_admission_filter(&config, "paper", 4_000, &mut intents).expect("filter");

        assert!(decision.is_none());
        assert_eq!(intents.len(), 1);
    }
}
