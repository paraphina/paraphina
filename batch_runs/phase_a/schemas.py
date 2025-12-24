"""
schemas.py

Typed dataclasses for Phase A promotion pipeline:
- Budgets and risk tier definitions
- Candidate evaluation results
- Optimization objectives
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ===========================================================================
# Budget definitions
# ===========================================================================

@dataclass(frozen=True)
class TierBudget:
    """
    Budget constraints for a risk tier.
    
    A candidate passes the budget if:
    - kill_prob_ci_upper <= max_kill_prob
    - drawdown_cvar <= max_drawdown_cvar
    - mean_pnl >= min_mean_pnl
    """
    tier_name: str
    max_kill_prob: float      # Maximum kill probability (95% CI upper bound)
    max_drawdown_cvar: float  # Maximum drawdown CVaR
    min_mean_pnl: float       # Minimum mean PnL


@dataclass
class BudgetConfig:
    """Collection of tier budgets loaded from budgets.yaml."""
    tiers: Dict[str, TierBudget]
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BudgetConfig":
        """Parse from YAML-loaded dict."""
        tiers = {}
        for tier_name, tier_data in d.get("tiers", {}).items():
            tiers[tier_name] = TierBudget(
                tier_name=tier_name,
                max_kill_prob=float(tier_data.get("max_kill_prob", 0.10)),
                max_drawdown_cvar=float(tier_data.get("max_drawdown_cvar", 1000.0)),
                min_mean_pnl=float(tier_data.get("min_mean_pnl", 10.0)),
            )
        return cls(tiers=tiers)


# ===========================================================================
# Candidate configuration
# ===========================================================================

@dataclass
class CandidateConfig:
    """
    Configuration for a single candidate trial.
    
    Encapsulates all knobs being tuned.
    """
    candidate_id: str
    profile: str
    hedge_band_base: float
    mm_size_eta: float
    vol_ref: float
    daily_loss_limit: float
    
    # Additional optional knobs
    init_q_tao: float = 0.0
    hedge_max_step: float = 10.0
    
    def to_env_overlay(self) -> Dict[str, str]:
        """Convert to environment variable overlay for subprocess."""
        return {
            "PARAPHINA_RISK_PROFILE": self.profile,
            "PARAPHINA_HEDGE_BAND_BASE": str(self.hedge_band_base),
            "PARAPHINA_MM_SIZE_ETA": str(self.mm_size_eta),
            "PARAPHINA_VOL_REF": str(self.vol_ref),
            "PARAPHINA_DAILY_LOSS_LIMIT": str(self.daily_loss_limit),
            "PARAPHINA_INIT_Q_TAO": str(self.init_q_tao),
            "PARAPHINA_HEDGE_MAX_STEP": str(self.hedge_max_step),
        }
    
    def config_hash(self) -> str:
        """Compute deterministic hash of configuration."""
        env = self.to_env_overlay()
        sorted_items = sorted(env.items())
        key_str = "_".join(f"{k}={v}" for k, v in sorted_items)
        return hashlib.sha256(key_str.encode()).hexdigest()[:12]


# ===========================================================================
# Metrics from mc_summary.json
# ===========================================================================

@dataclass
class TailRiskMetrics:
    """Tail risk metrics extracted from mc_summary.json."""
    # PnL VaR/CVaR
    pnl_var: float
    pnl_cvar: float
    
    # Drawdown VaR/CVaR
    drawdown_var: float
    drawdown_cvar: float
    
    # Kill probability with Wilson CI
    kill_prob_point: float
    kill_prob_ci_lower: float
    kill_prob_ci_upper: float
    kill_count: int
    total_runs: int
    
    # Quantiles
    pnl_p01: float = float("nan")
    pnl_p05: float = float("nan")
    pnl_p50: float = float("nan")
    pnl_p95: float = float("nan")
    pnl_p99: float = float("nan")
    
    max_drawdown_p01: float = float("nan")
    max_drawdown_p05: float = float("nan")
    max_drawdown_p50: float = float("nan")
    max_drawdown_p95: float = float("nan")
    max_drawdown_p99: float = float("nan")


@dataclass
class AggregateMetrics:
    """Aggregate metrics from mc_summary.json."""
    mean_pnl: float
    pnl_std: float
    pnl_min: float
    pnl_max: float
    
    mean_drawdown: float
    drawdown_std: float
    drawdown_min: float
    drawdown_max: float
    
    kill_rate: float
    kill_count: int


# ===========================================================================
# Evaluation result
# ===========================================================================

@dataclass
class EvalResult:
    """
    Complete result from evaluating a candidate.
    
    Contains metrics from both research and adversarial suites.
    """
    candidate_id: str
    trial_id: str
    config: CandidateConfig
    trial_dir: Path
    
    # Research suite metrics
    research_mean_pnl: float
    research_kill_prob: float
    research_drawdown_cvar: float
    
    # Adversarial suite metrics  
    adversarial_mean_pnl: float
    adversarial_kill_prob: float
    adversarial_drawdown_cvar: float
    
    # Monte Carlo tail risk (primary metrics for optimization)
    mc_mean_pnl: float
    mc_pnl_cvar: float
    mc_drawdown_cvar: float
    mc_kill_prob_point: float
    mc_kill_prob_ci_upper: float
    mc_total_runs: int
    
    # Evidence verification status (required for promotion eligibility)
    evidence_verified: bool
    evidence_errors: List[str] = field(default_factory=list)
    
    # Suite pass/fail status
    research_suite_passed: bool = True
    adversarial_suite_passed: bool = True
    
    # Execution info
    duration_sec: float = 0.0
    error_message: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if result is valid for optimization."""
        return (
            self.evidence_verified
            and self.research_suite_passed
            and self.adversarial_suite_passed
            and not math.isnan(self.mc_mean_pnl)
            and not math.isnan(self.mc_kill_prob_ci_upper)
            and not math.isnan(self.mc_drawdown_cvar)
        )
    
    def passes_budget(self, budget: TierBudget) -> bool:
        """Check if result passes a tier budget."""
        if not self.is_valid:
            return False
        return (
            self.mc_kill_prob_ci_upper <= budget.max_kill_prob
            and self.mc_drawdown_cvar <= budget.max_drawdown_cvar
            and self.mc_mean_pnl >= budget.min_mean_pnl
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "candidate_id": self.candidate_id,
            "trial_id": self.trial_id,
            "trial_dir": str(self.trial_dir),
            "config": asdict(self.config),
            "research": {
                "mean_pnl": self.research_mean_pnl,
                "kill_prob": self.research_kill_prob,
                "drawdown_cvar": self.research_drawdown_cvar,
                "passed": self.research_suite_passed,
            },
            "adversarial": {
                "mean_pnl": self.adversarial_mean_pnl,
                "kill_prob": self.adversarial_kill_prob,
                "drawdown_cvar": self.adversarial_drawdown_cvar,
                "passed": self.adversarial_suite_passed,
            },
            "monte_carlo": {
                "mean_pnl": self.mc_mean_pnl,
                "pnl_cvar": self.mc_pnl_cvar,
                "drawdown_cvar": self.mc_drawdown_cvar,
                "kill_prob_point": self.mc_kill_prob_point,
                "kill_prob_ci_upper": self.mc_kill_prob_ci_upper,
                "total_runs": self.mc_total_runs,
            },
            "evidence_verified": self.evidence_verified,
            "evidence_errors": self.evidence_errors,
            "duration_sec": self.duration_sec,
        }
        if self.error_message:
            d["error_message"] = self.error_message
        return d


# ===========================================================================
# Pareto front
# ===========================================================================

@dataclass
class ParetoCandidate:
    """
    A candidate on the Pareto frontier.
    
    Objectives (all to be minimized for Pareto):
    - Minimize -mean_pnl (i.e., maximize mean_pnl)
    - Minimize kill_prob_ci_upper
    - Minimize drawdown_cvar
    """
    result: EvalResult
    
    # Objective values (for Pareto comparison)
    obj_neg_pnl: float  # Negated for minimization
    obj_kill_prob: float
    obj_drawdown_cvar: float
    
    @classmethod
    def from_result(cls, result: EvalResult) -> "ParetoCandidate":
        return cls(
            result=result,
            obj_neg_pnl=-result.mc_mean_pnl,
            obj_kill_prob=result.mc_kill_prob_ci_upper,
            obj_drawdown_cvar=result.mc_drawdown_cvar,
        )
    
    def dominates(self, other: "ParetoCandidate") -> bool:
        """
        Check if self dominates other (Pareto dominance).
        
        Self dominates other if:
        - Self is at least as good on all objectives
        - Self is strictly better on at least one objective
        """
        # At least as good on all (minimization)
        at_least_as_good = (
            self.obj_neg_pnl <= other.obj_neg_pnl
            and self.obj_kill_prob <= other.obj_kill_prob
            and self.obj_drawdown_cvar <= other.obj_drawdown_cvar
        )
        
        if not at_least_as_good:
            return False
        
        # Strictly better on at least one
        strictly_better = (
            self.obj_neg_pnl < other.obj_neg_pnl
            or self.obj_kill_prob < other.obj_kill_prob
            or self.obj_drawdown_cvar < other.obj_drawdown_cvar
        )
        
        return strictly_better


# ===========================================================================
# Promotion record
# ===========================================================================

@dataclass
class PromotionRecord:
    """
    Record of a promoted configuration.
    
    Written as PROMOTION_RECORD.json in the promoted config folder.
    """
    promoted_at: str  # ISO timestamp
    study_name: str
    baseline_identifier: str
    candidate_hash: str
    candidate_id: str
    trial_id: str
    
    # Configuration
    config: CandidateConfig
    env_overlay: Dict[str, str]
    
    # Commands executed
    commands_executed: List[str]
    seeds_used: List[int]
    
    # Suite paths
    research_suite_path: str
    adversarial_suite_path: str
    
    # Evidence verification
    evidence_verified: bool
    evidence_verification_results: List[Dict[str, Any]]
    
    # Metrics at promotion
    metrics: Dict[str, float]
    
    # Budget tier passed
    budget_tier: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": 1,
            "promoted_at": self.promoted_at,
            "study_name": self.study_name,
            "baseline_identifier": self.baseline_identifier,
            "candidate_hash": self.candidate_hash,
            "candidate_id": self.candidate_id,
            "trial_id": self.trial_id,
            "config": asdict(self.config),
            "env_overlay": self.env_overlay,
            "commands_executed": self.commands_executed,
            "seeds_used": self.seeds_used,
            "research_suite_path": self.research_suite_path,
            "adversarial_suite_path": self.adversarial_suite_path,
            "evidence_verified": self.evidence_verified,
            "evidence_verification_results": self.evidence_verification_results,
            "metrics": self.metrics,
            "budget_tier": self.budget_tier,
        }


# ===========================================================================
# Helper functions
# ===========================================================================

def is_nan(x: float) -> bool:
    """Check if value is NaN."""
    return math.isnan(x) if isinstance(x, float) else False


def parse_mc_summary(summary_path: Path) -> Dict[str, Any]:
    """
    Parse mc_summary.json into a flat metrics dict.
    
    Returns empty dict on parse error.
    """
    if not summary_path.exists():
        return {}
    
    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}
    
    # Extract from known structure
    tail_risk = summary.get("tail_risk", {})
    aggregate = summary.get("aggregate", {})
    
    pnl_var_cvar = tail_risk.get("pnl_var_cvar", {})
    dd_var_cvar = tail_risk.get("max_drawdown_var_cvar", {})
    kill_prob = tail_risk.get("kill_probability", {})
    pnl_quantiles = tail_risk.get("pnl_quantiles", {})
    dd_quantiles = tail_risk.get("max_drawdown_quantiles", {})
    pnl_agg = aggregate.get("pnl", {})
    dd_agg = aggregate.get("max_drawdown", {})
    
    return {
        "schema_version": tail_risk.get("schema_version", 0),
        "mean_pnl": pnl_agg.get("mean", float("nan")),
        "pnl_std": pnl_agg.get("std_pop", float("nan")),
        "pnl_min": pnl_agg.get("min", float("nan")),
        "pnl_max": pnl_agg.get("max", float("nan")),
        "pnl_var": pnl_var_cvar.get("var", float("nan")),
        "pnl_cvar": pnl_var_cvar.get("cvar", float("nan")),
        "pnl_p01": pnl_quantiles.get("p01", float("nan")),
        "pnl_p05": pnl_quantiles.get("p05", float("nan")),
        "pnl_p50": pnl_quantiles.get("p50", float("nan")),
        "pnl_p95": pnl_quantiles.get("p95", float("nan")),
        "pnl_p99": pnl_quantiles.get("p99", float("nan")),
        "mean_drawdown": dd_agg.get("mean", float("nan")),
        "drawdown_std": dd_agg.get("std_pop", float("nan")),
        "drawdown_min": dd_agg.get("min", float("nan")),
        "drawdown_max": dd_agg.get("max", float("nan")),
        "drawdown_var": dd_var_cvar.get("var", float("nan")),
        "drawdown_cvar": dd_var_cvar.get("cvar", float("nan")),
        "max_drawdown_p01": dd_quantiles.get("p01", float("nan")),
        "max_drawdown_p05": dd_quantiles.get("p05", float("nan")),
        "max_drawdown_p50": dd_quantiles.get("p50", float("nan")),
        "max_drawdown_p95": dd_quantiles.get("p95", float("nan")),
        "max_drawdown_p99": dd_quantiles.get("p99", float("nan")),
        "kill_prob_point": kill_prob.get("point_estimate", float("nan")),
        "kill_prob_ci_lower": kill_prob.get("ci_lower", float("nan")),
        "kill_prob_ci_upper": kill_prob.get("ci_upper", float("nan")),
        "kill_count": kill_prob.get("kill_count", 0),
        "total_runs": kill_prob.get("total_runs", 0),
        "kill_rate": aggregate.get("kill_rate", float("nan")),
    }

