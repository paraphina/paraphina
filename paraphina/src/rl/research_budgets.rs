// src/rl/research_budgets.rs
//
// Canonical research alignment budgets used by Exp07/Exp11 and RL reward shaping.

use serde::Deserialize;

use crate::config::RiskProfile;

#[derive(Debug, Clone, Deserialize)]
pub struct ResearchAlignmentBudget {
    pub name: String,
    pub max_kill_prob: f64,
    pub max_drawdown_abs: f64,
    pub min_final_pnl_mean: f64,
    pub lambda_std: f64,
    pub lambda_dd: f64,
    pub lambda_kill: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct ResearchAlignmentBudgetFile {
    #[allow(dead_code)]
    schema_version: i64,
    risk_tiers: Vec<ResearchAlignmentBudget>,
}

fn load_budget_file() -> ResearchAlignmentBudgetFile {
    let raw = include_str!("../../../configs/research_alignment_budgets.json");
    serde_json::from_str(raw).expect("invalid research_alignment_budgets.json")
}

pub fn alignment_budget_for_profile(profile: RiskProfile) -> ResearchAlignmentBudget {
    let file = load_budget_file();
    let name = profile.as_str();
    file.risk_tiers
        .into_iter()
        .find(|tier| tier.name == name)
        .unwrap_or_else(|| panic!("Missing alignment budget for profile: {}", name))
}
