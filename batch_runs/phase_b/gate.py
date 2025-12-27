"""
gate.py

Promotion Gate for Phase B: Confidence-Aware Statistical Gating.

Implements a tri-state decision model:

1. **Guardrails (non-regression / safety constraints)**
   - Pass when candidate is not provably worse than baseline
   - CI overlap is acceptable (candidate may be equal or better)
   - Failure here → REJECT

2. **Promotion Criteria (superiority constraints)**
   - Pass only when candidate is provably better than baseline at alpha
   - CI overlap fails these checks (must demonstrate clear improvement)
   - Failure here (with guardrails passing) → HOLD

Decision outcomes:
- PROMOTE: Guardrails pass AND promotion criteria pass
- HOLD: Guardrails pass BUT promotion criteria fail (needs more data)
- REJECT: Guardrails fail (candidate is provably worse)
- ERROR: Missing/invalid run data, parse errors, etc.

The gate is designed to be conservative: HOLD is the default when evidence
is insufficient to prove superiority, but candidate isn't demonstrably worse.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from batch_runs.phase_b.confidence import (
    ConfidenceInterval,
    RunMetrics,
    compute_run_metrics,
    bootstrap_ci,
    compute_mean,
)
from batch_runs.phase_b.compare import (
    ComparisonResult,
    MetricComparison,
    MetricDirection,
    ComparisonOutcome,
    compare_runs,
    check_dominance,
    check_strict_dominance,
    check_non_inferiority,
)
from batch_runs.phase_b.run_data import (
    load_run_data as load_trials_data,
    LoadedRunData,
    TrialsFileNotFoundError,
    NoUsableObservationsError,
    RunDataError,
)


# =============================================================================
# Enums
# =============================================================================

class PromotionOutcome(Enum):
    """
    Outcome of the promotion gate (tri-state + error).
    
    Decision semantics:
    - PROMOTE: Guardrails pass AND promotion criteria pass
              (candidate is provably better than baseline)
    - HOLD: Guardrails pass BUT promotion criteria fail
            (candidate is not worse, but not provably better either)
    - REJECT: Guardrails fail
              (candidate is provably worse than baseline)
    - ERROR: Missing/invalid run data, parse errors, etc.
    """
    PROMOTE = "promote"      # Guardrails pass + promotion criteria pass
    HOLD = "hold"            # Guardrails pass + promotion criteria fail
    REJECT = "reject"        # Guardrails fail
    ERROR = "error"          # Error during evaluation


# =============================================================================
# Promotion Decision
# =============================================================================

@dataclass
class PromotionDecision:
    """
    Result of the promotion gate evaluation.
    
    Attributes:
        outcome: PROMOTE, HOLD, REJECT, or ERROR
        candidate_path: Path to candidate run
        baseline_path: Path to baseline run (if any)
        candidate_metrics: Computed metrics for candidate
        baseline_metrics: Computed metrics for baseline (if any)
        comparison: Full comparison result
        guardrail_checks: List of guardrail check results (pass when not provably worse)
        promotion_checks: List of promotion check results (pass when provably better)
        guardrails_passed: Whether all guardrails passed
        promotion_passed: Whether all promotion criteria passed
        decision_reason: Short explanation of the decision
        pass_reasons: List of reasons for passing gates (legacy, combines both)
        fail_reasons: List of reasons for failing gates (legacy, combines both)
        timestamp: When the decision was made
        alpha: Significance level used
        n_bootstrap: Number of bootstrap samples used
    """
    outcome: PromotionOutcome
    candidate_path: Optional[Path]
    baseline_path: Optional[Path]
    candidate_metrics: Optional[RunMetrics]
    baseline_metrics: Optional[RunMetrics]
    comparison: Optional[ComparisonResult]
    guardrail_checks: List[Tuple[str, bool, str]] = field(default_factory=list)  # (name, passed, reason)
    promotion_checks: List[Tuple[str, bool, str]] = field(default_factory=list)  # (name, passed, reason)
    guardrails_passed: bool = True
    promotion_passed: bool = True
    decision_reason: str = ""
    pass_reasons: List[str] = field(default_factory=list)
    fail_reasons: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    timestamp: str = ""
    alpha: float = 0.05
    n_bootstrap: int = 1000
    seed: int = 42
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    @property
    def is_promote(self) -> bool:
        """Check if the decision is to promote."""
        return self.outcome == PromotionOutcome.PROMOTE
    
    @property
    def is_hold(self) -> bool:
        """Check if the decision is to hold."""
        return self.outcome == PromotionOutcome.HOLD
    
    @property
    def is_reject(self) -> bool:
        """Check if the decision is to reject."""
        return self.outcome == PromotionOutcome.REJECT
    
    @property
    def exit_code(self) -> int:
        """Get CLI exit code: 0 = promote, 4 = hold, 3 = reject, 1 = error."""
        if self.outcome == PromotionOutcome.PROMOTE:
            return 0
        elif self.outcome == PromotionOutcome.HOLD:
            return 4
        elif self.outcome == PromotionOutcome.REJECT:
            return 3
        else:
            return 1
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = []
        lines.append(f"Promotion Decision: {self.outcome.value.upper()}")
        lines.append(f"Timestamp: {self.timestamp}")
        lines.append(f"Alpha: {self.alpha}")
        lines.append(f"Bootstrap samples: {self.n_bootstrap}")
        lines.append("")
        
        if self.candidate_path:
            lines.append(f"Candidate: {self.candidate_path}")
        if self.baseline_path:
            lines.append(f"Baseline: {self.baseline_path}")
        lines.append("")
        
        if self.candidate_metrics:
            lines.append("Candidate Metrics:")
            lines.append(f"  PnL mean: {self.candidate_metrics.pnl_mean.point:.4f} "
                        f"[{self.candidate_metrics.pnl_mean.lower:.4f}, "
                        f"{self.candidate_metrics.pnl_mean.upper:.4f}]")
            lines.append(f"  Max drawdown: {self.candidate_metrics.max_drawdown.point:.4f} "
                        f"[{self.candidate_metrics.max_drawdown.lower:.4f}, "
                        f"{self.candidate_metrics.max_drawdown.upper:.4f}]")
            lines.append(f"  Kill rate: {self.candidate_metrics.kill_rate:.4f} "
                        f"({self.candidate_metrics.kill_count}/{self.candidate_metrics.n_runs})")
            lines.append("")
        
        if self.baseline_metrics:
            lines.append("Baseline Metrics:")
            lines.append(f"  PnL mean: {self.baseline_metrics.pnl_mean.point:.4f} "
                        f"[{self.baseline_metrics.pnl_mean.lower:.4f}, "
                        f"{self.baseline_metrics.pnl_mean.upper:.4f}]")
            lines.append(f"  Max drawdown: {self.baseline_metrics.max_drawdown.point:.4f} "
                        f"[{self.baseline_metrics.max_drawdown.lower:.4f}, "
                        f"{self.baseline_metrics.max_drawdown.upper:.4f}]")
            lines.append(f"  Kill rate: {self.baseline_metrics.kill_rate:.4f} "
                        f"({self.baseline_metrics.kill_count}/{self.baseline_metrics.n_runs})")
            lines.append("")
        
        # Guardrails section
        lines.append("Guardrails (must pass):")
        if self.guardrail_checks:
            for name, passed, reason in self.guardrail_checks:
                marker = "✓" if passed else "✗"
                lines.append(f"  {marker} {name}: {reason}")
        else:
            lines.append("  (no guardrail checks)")
        lines.append("")
        
        # Promotion criteria section
        lines.append("Promotion criteria (must pass to PROMOTE):")
        if self.promotion_checks:
            for name, passed, reason in self.promotion_checks:
                marker = "✓" if passed else "✗"
                lines.append(f"  {marker} {name}: {reason}")
        else:
            lines.append("  (no promotion criteria checks)")
        lines.append("")
        
        # Decision rationale
        lines.append("Decision rationale:")
        lines.append(f"  {self.decision_reason}")
        lines.append("")
        
        if self.error_message:
            lines.append(f"Error: {self.error_message}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "decision": self.outcome.value.upper(),
            "outcome": self.outcome.value,
            "guardrails_passed": self.guardrails_passed,
            "promotion_passed": self.promotion_passed,
            "decision_reason": self.decision_reason,
            "is_promote": self.is_promote,
            "exit_code": self.exit_code,
            "candidate_path": str(self.candidate_path) if self.candidate_path else None,
            "baseline_path": str(self.baseline_path) if self.baseline_path else None,
            "candidate_metrics": self.candidate_metrics.to_dict() if self.candidate_metrics else None,
            "baseline_metrics": self.baseline_metrics.to_dict() if self.baseline_metrics else None,
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "guardrail_checks": [
                {"name": name, "passed": passed, "reason": reason}
                for name, passed, reason in self.guardrail_checks
            ],
            "promotion_checks": [
                {"name": name, "passed": passed, "reason": reason}
                for name, passed, reason in self.promotion_checks
            ],
            "pass_reasons": self.pass_reasons,
            "fail_reasons": self.fail_reasons,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "alpha": self.alpha,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
        }
    
    def to_markdown(self) -> str:
        """Generate a Markdown report."""
        lines = []
        lines.append("# Phase B Confidence Gate Report")
        lines.append("")
        lines.append(f"**Decision:** {self.outcome.value.upper()}")
        lines.append(f"**Timestamp:** {self.timestamp}")
        lines.append("")
        
        # Decision rationale section
        lines.append("## Decision Rationale")
        lines.append("")
        lines.append(f"> {self.decision_reason}")
        lines.append("")
        lines.append(f"- Guardrails passed: {'✓ Yes' if self.guardrails_passed else '✗ No'}")
        lines.append(f"- Promotion criteria passed: {'✓ Yes' if self.promotion_passed else '✗ No'}")
        lines.append("")
        
        lines.append("## Configuration")
        lines.append("")
        lines.append(f"- Alpha: {self.alpha}")
        lines.append(f"- Bootstrap samples: {self.n_bootstrap}")
        lines.append(f"- Seed: {self.seed}")
        if self.candidate_path:
            lines.append(f"- Candidate path: `{self.candidate_path}`")
        if self.baseline_path:
            lines.append(f"- Baseline path: `{self.baseline_path}`")
        lines.append("")
        
        if self.candidate_metrics:
            lines.append("## Candidate Metrics")
            lines.append("")
            lines.append("| Metric | Point Estimate | Lower CI | Upper CI |")
            lines.append("|--------|----------------|----------|----------|")
            lines.append(f"| PnL Mean | {self.candidate_metrics.pnl_mean.point:.4f} | "
                        f"{self.candidate_metrics.pnl_mean.lower:.4f} | "
                        f"{self.candidate_metrics.pnl_mean.upper:.4f} |")
            lines.append(f"| PnL Median | {self.candidate_metrics.pnl_median.point:.4f} | "
                        f"{self.candidate_metrics.pnl_median.lower:.4f} | "
                        f"{self.candidate_metrics.pnl_median.upper:.4f} |")
            lines.append(f"| PnL CVaR | {self.candidate_metrics.pnl_cvar.point:.4f} | "
                        f"{self.candidate_metrics.pnl_cvar.lower:.4f} | "
                        f"{self.candidate_metrics.pnl_cvar.upper:.4f} |")
            lines.append(f"| Max Drawdown | {self.candidate_metrics.max_drawdown.point:.4f} | "
                        f"{self.candidate_metrics.max_drawdown.lower:.4f} | "
                        f"{self.candidate_metrics.max_drawdown.upper:.4f} |")
            lines.append("")
            lines.append(f"- Kill rate: {self.candidate_metrics.kill_rate:.4f} "
                        f"({self.candidate_metrics.kill_count}/{self.candidate_metrics.n_runs} runs)")
            lines.append(f"- Total runs: {self.candidate_metrics.n_runs}")
            lines.append("")
        
        if self.baseline_metrics:
            lines.append("## Baseline Metrics")
            lines.append("")
            lines.append("| Metric | Point Estimate | Lower CI | Upper CI |")
            lines.append("|--------|----------------|----------|----------|")
            lines.append(f"| PnL Mean | {self.baseline_metrics.pnl_mean.point:.4f} | "
                        f"{self.baseline_metrics.pnl_mean.lower:.4f} | "
                        f"{self.baseline_metrics.pnl_mean.upper:.4f} |")
            lines.append(f"| PnL Median | {self.baseline_metrics.pnl_median.point:.4f} | "
                        f"{self.baseline_metrics.pnl_median.lower:.4f} | "
                        f"{self.baseline_metrics.pnl_median.upper:.4f} |")
            lines.append(f"| PnL CVaR | {self.baseline_metrics.pnl_cvar.point:.4f} | "
                        f"{self.baseline_metrics.pnl_cvar.lower:.4f} | "
                        f"{self.baseline_metrics.pnl_cvar.upper:.4f} |")
            lines.append(f"| Max Drawdown | {self.baseline_metrics.max_drawdown.point:.4f} | "
                        f"{self.baseline_metrics.max_drawdown.lower:.4f} | "
                        f"{self.baseline_metrics.max_drawdown.upper:.4f} |")
            lines.append("")
            lines.append(f"- Kill rate: {self.baseline_metrics.kill_rate:.4f} "
                        f"({self.baseline_metrics.kill_count}/{self.baseline_metrics.n_runs} runs)")
            lines.append(f"- Total runs: {self.baseline_metrics.n_runs}")
            lines.append("")
        
        lines.append("## Guardrails (must pass)")
        lines.append("")
        lines.append("*Non-regression checks: candidate must not be provably worse than baseline.*")
        lines.append("")
        if self.guardrail_checks:
            for name, passed, reason in self.guardrail_checks:
                marker = "✓" if passed else "✗"
                lines.append(f"- {marker} **{name}**: {reason}")
        else:
            lines.append("- (no guardrail checks)")
        lines.append("")
        
        lines.append("## Promotion Criteria (must pass to PROMOTE)")
        lines.append("")
        lines.append("*Superiority checks: candidate must be provably better than baseline.*")
        lines.append("")
        if self.promotion_checks:
            for name, passed, reason in self.promotion_checks:
                marker = "✓" if passed else "✗"
                lines.append(f"- {marker} **{name}**: {reason}")
        else:
            lines.append("- (no promotion criteria checks)")
        lines.append("")
        
        if self.error_message:
            lines.append("## Error")
            lines.append("")
            lines.append(f"```\n{self.error_message}\n```")
            lines.append("")
        
        lines.append("---")
        lines.append(f"*Generated by Phase B Confidence Gate*")
        
        return "\n".join(lines)


# =============================================================================
# Promotion Gate
# =============================================================================

@dataclass
class PromotionGate:
    """
    Configuration for the promotion gate.
    
    Attributes:
        alpha: Significance level for confidence intervals (default 0.05)
        n_bootstrap: Number of bootstrap samples (default 1000)
        block_size: Block size for bootstrap (None = sqrt(n))
        seed: Random seed for reproducibility
        kill_threshold: Maximum allowed kill rate (optional)
        drawdown_threshold: Maximum allowed drawdown upper CI (optional)
        pnl_threshold: Minimum required PnL lower CI (optional)
        require_strict_dominance: Require candidate to strictly dominate baseline
    """
    alpha: float = 0.05
    n_bootstrap: int = 1000
    block_size: Optional[int] = None
    seed: int = 42
    kill_threshold: Optional[float] = None
    drawdown_threshold: Optional[float] = None
    pnl_threshold: Optional[float] = None
    require_strict_dominance: bool = True
    
    def evaluate(
        self,
        candidate_pnl: List[np.ndarray],
        candidate_kills: List[bool],
        baseline_pnl: Optional[List[np.ndarray]] = None,
        baseline_kills: Optional[List[bool]] = None,
        candidate_path: Optional[Path] = None,
        baseline_path: Optional[Path] = None,
    ) -> PromotionDecision:
        """
        Evaluate candidate against baseline using tri-state decision model.
        
        Tri-state decision:
        1. GUARDRAILS (non-regression): Candidate not provably worse than baseline
           - PnL: baseline does NOT dominate candidate (baseline.lower <= candidate.upper)
           - Drawdown: baseline does NOT dominate candidate (baseline.upper >= candidate.lower)
           - Kill threshold: candidate.kill_rate <= threshold (if provided)
           
        2. PROMOTION CRITERIA (superiority): Candidate provably better than baseline
           - PnL: candidate.lower > baseline.upper
           - Drawdown: candidate.upper < baseline.lower
           
        Decision:
        - ERROR: Exception during evaluation
        - REJECT: Any guardrail fails
        - HOLD: All guardrails pass, but promotion criteria fail
        - PROMOTE: All guardrails pass AND all promotion criteria pass
        
        Args:
            candidate_pnl: List of PnL arrays, one per run
            candidate_kills: List of kill flags, one per run
            baseline_pnl: List of PnL arrays for baseline (optional)
            baseline_kills: List of kill flags for baseline (optional)
            candidate_path: Path to candidate run (for reporting)
            baseline_path: Path to baseline run (for reporting)
            
        Returns:
            PromotionDecision with outcome and explanation
        """
        guardrail_checks: List[Tuple[str, bool, str]] = []
        promotion_checks: List[Tuple[str, bool, str]] = []
        pass_reasons = []
        fail_reasons = []
        
        try:
            # Compute candidate metrics
            candidate_metrics = compute_run_metrics(
                candidate_pnl,
                candidate_kills,
                alpha=self.alpha,
                n_bootstrap=self.n_bootstrap,
                block_size=self.block_size,
                seed=self.seed,
            )
            
            # Compute baseline metrics if provided
            baseline_metrics = None
            if baseline_pnl is not None and baseline_kills is not None:
                baseline_metrics = compute_run_metrics(
                    baseline_pnl,
                    baseline_kills,
                    alpha=self.alpha,
                    n_bootstrap=self.n_bootstrap,
                    block_size=self.block_size,
                    seed=self.seed + 1000,  # Different seed for independence
                )
            
            # Perform comparison
            comparison = compare_runs(
                candidate_metrics,
                baseline_metrics,
                kill_threshold=self.kill_threshold,
                drawdown_threshold=self.drawdown_threshold,
                pnl_threshold=self.pnl_threshold,
            )
            
            # === GUARDRAILS: Non-regression checks (candidate not provably worse) ===
            if baseline_metrics is not None:
                # Check non-inferiority (baseline does NOT dominate candidate)
                non_inferior, ni_reasons = check_non_inferiority(
                    candidate_metrics, baseline_metrics
                )
                
                # PnL guardrail: baseline does not dominate on PnL
                pnl_outcome, pnl_reason = check_dominance(
                    candidate_metrics.pnl_mean,
                    baseline_metrics.pnl_mean,
                    MetricDirection.HIGHER_IS_BETTER,
                )
                pnl_guardrail_pass = pnl_outcome != ComparisonOutcome.BASELINE_DOMINATES
                pnl_guardrail_reason = (
                    f"Candidate not worse: {pnl_reason}"
                    if pnl_guardrail_pass
                    else f"Baseline dominates: {pnl_reason}"
                )
                guardrail_checks.append(("PnL non-regression", pnl_guardrail_pass, pnl_guardrail_reason))
                
                # Drawdown guardrail: baseline does not dominate on drawdown
                dd_outcome, dd_reason = check_dominance(
                    candidate_metrics.max_drawdown,
                    baseline_metrics.max_drawdown,
                    MetricDirection.LOWER_IS_BETTER,
                )
                dd_guardrail_pass = dd_outcome != ComparisonOutcome.BASELINE_DOMINATES
                dd_guardrail_reason = (
                    f"Candidate not worse: {dd_reason}"
                    if dd_guardrail_pass
                    else f"Baseline dominates: {dd_reason}"
                )
                guardrail_checks.append(("Drawdown non-regression", dd_guardrail_pass, dd_guardrail_reason))
            
            # Kill threshold guardrail (if provided)
            if self.kill_threshold is not None:
                kill_guardrail_pass = candidate_metrics.kill_rate <= self.kill_threshold
                kill_guardrail_reason = (
                    f"Kill rate {candidate_metrics.kill_rate:.4f} <= threshold {self.kill_threshold:.4f}"
                    if kill_guardrail_pass
                    else f"Kill rate {candidate_metrics.kill_rate:.4f} > threshold {self.kill_threshold:.4f}"
                )
                guardrail_checks.append(("Kill rate threshold", kill_guardrail_pass, kill_guardrail_reason))
            
            # === PROMOTION CRITERIA: Superiority checks (candidate provably better) ===
            if baseline_metrics is not None:
                # Check strict dominance (candidate is provably better)
                dominates, dom_reasons = check_strict_dominance(
                    candidate_metrics, baseline_metrics
                )
                
                # PnL promotion: candidate dominates on PnL
                pnl_outcome, pnl_reason = check_dominance(
                    candidate_metrics.pnl_mean,
                    baseline_metrics.pnl_mean,
                    MetricDirection.HIGHER_IS_BETTER,
                )
                pnl_promo_pass = pnl_outcome == ComparisonOutcome.CANDIDATE_DOMINATES
                pnl_promo_reason = (
                    f"Candidate superior: {pnl_reason}"
                    if pnl_promo_pass
                    else f"Not proven superior: {pnl_reason}"
                )
                promotion_checks.append(("PnL superiority", pnl_promo_pass, pnl_promo_reason))
                
                # Drawdown promotion: candidate dominates on drawdown
                dd_outcome, dd_reason = check_dominance(
                    candidate_metrics.max_drawdown,
                    baseline_metrics.max_drawdown,
                    MetricDirection.LOWER_IS_BETTER,
                )
                dd_promo_pass = dd_outcome == ComparisonOutcome.CANDIDATE_DOMINATES
                dd_promo_reason = (
                    f"Candidate superior: {dd_reason}"
                    if dd_promo_pass
                    else f"Not proven superior: {dd_reason}"
                )
                promotion_checks.append(("Drawdown superiority", dd_promo_pass, dd_promo_reason))
            
            # === COMPUTE FINAL DECISION ===
            guardrails_passed = all(passed for _, passed, _ in guardrail_checks)
            promotion_passed = all(passed for _, passed, _ in promotion_checks) if promotion_checks else True
            
            # Build legacy pass/fail reasons for backwards compatibility
            for name, passed, reason in guardrail_checks:
                if passed:
                    pass_reasons.append(f"[Guardrail] {name}: {reason}")
                else:
                    fail_reasons.append(f"[Guardrail] {name}: {reason}")
            
            for name, passed, reason in promotion_checks:
                if passed:
                    pass_reasons.append(f"[Promotion] {name}: {reason}")
                else:
                    fail_reasons.append(f"[Promotion] {name}: {reason}")
            
            # Determine outcome and decision reason
            if not guardrails_passed:
                outcome = PromotionOutcome.REJECT
                failed_guardrails = [name for name, passed, _ in guardrail_checks if not passed]
                decision_reason = f"REJECT: Guardrails failed ({', '.join(failed_guardrails)}). Candidate is provably worse than baseline."
            elif not promotion_passed:
                outcome = PromotionOutcome.HOLD
                failed_criteria = [name for name, passed, _ in promotion_checks if not passed]
                decision_reason = f"HOLD: Guardrails passed but promotion criteria failed ({', '.join(failed_criteria)}). Candidate is not worse, but not provably better. Consider collecting more data."
            else:
                outcome = PromotionOutcome.PROMOTE
                decision_reason = "PROMOTE: All guardrails passed and candidate is provably better than baseline on all promotion criteria."
            
            return PromotionDecision(
                outcome=outcome,
                candidate_path=candidate_path,
                baseline_path=baseline_path,
                candidate_metrics=candidate_metrics,
                baseline_metrics=baseline_metrics,
                comparison=comparison,
                guardrail_checks=guardrail_checks,
                promotion_checks=promotion_checks,
                guardrails_passed=guardrails_passed,
                promotion_passed=promotion_passed,
                decision_reason=decision_reason,
                pass_reasons=pass_reasons,
                fail_reasons=fail_reasons,
                alpha=self.alpha,
                n_bootstrap=self.n_bootstrap,
                seed=self.seed,
            )
            
        except Exception as e:
            return PromotionDecision(
                outcome=PromotionOutcome.ERROR,
                candidate_path=candidate_path,
                baseline_path=baseline_path,
                candidate_metrics=None,
                baseline_metrics=None,
                comparison=None,
                guardrails_passed=False,
                promotion_passed=False,
                decision_reason=f"ERROR: {str(e)}",
                error_message=str(e),
                alpha=self.alpha,
                n_bootstrap=self.n_bootstrap,
                seed=self.seed,
            )


# =============================================================================
# Loading Metrics from Files
# =============================================================================

def load_run_data(run_dir: Path) -> Tuple[List[np.ndarray], List[bool]]:
    """
    Load run data from a Phase A run directory or JSONL file.
    
    Uses the canonical run_data loader which discovers trials.jsonl with
    the following precedence:
    1. If run_dir is a file ending in .jsonl, treat it as trials JSONL
    2. Else if <run_dir>/trials.jsonl exists, use it
    3. Else search within <run_dir> for **/trials.jsonl and pick the shallowest
       match (fewest path components), breaking ties by newest mtime
    
    Args:
        run_dir: Path to the run directory or JSONL file
        
    Returns:
        Tuple of (pnl_per_run, kill_flags)
        
    Raises:
        TrialsFileNotFoundError: If no trials.jsonl could be discovered
        NoUsableObservationsError: If no valid observations could be extracted
    """
    # Try the new canonical loader first (Phase A trials.jsonl format)
    try:
        loaded = load_trials_data(run_dir)
        return loaded.to_pnl_arrays(), loaded.to_kill_flags()
    except RunDataError:
        # Re-raise run data errors as-is for proper error handling
        raise
    except Exception:
        # Fall through to legacy loading for backwards compatibility
        pass
    
    # Legacy loading for mc_summary.json format (backwards compatibility)
    pnl_per_run = []
    kill_flags = []
    
    # Check for mc_summary.json
    mc_summary = run_dir / "mc_summary.json"
    if mc_summary.exists():
        with open(mc_summary) as f:
            data = json.load(f)
        
        # Extract per-run PnL values if available
        runs_data = data.get("runs", [])
        if runs_data:
            for run in runs_data:
                pnl = run.get("pnl_series", [run.get("final_pnl", 0)])
                pnl_per_run.append(np.array(pnl))
                kill_flags.append(run.get("kill_switch", False))
        else:
            # Fallback: try to get aggregate data
            aggregate = data.get("aggregate", {})
            pnl = aggregate.get("pnl", {})
            n_runs = data.get("tail_risk", {}).get("kill_probability", {}).get("total_runs", 1)
            
            # Create synthetic per-run data from aggregate
            mean_pnl = pnl.get("mean", 0)
            std_pnl = pnl.get("std_pop", 0)
            kill_rate = aggregate.get("kill_rate", 0)
            
            # Generate synthetic runs
            rng = np.random.default_rng(42)
            for _ in range(n_runs):
                final_pnl = rng.normal(mean_pnl, max(std_pnl, 0.01))
                pnl_per_run.append(np.array([final_pnl]))
                kill_flags.append(rng.random() < kill_rate)
    
    # Check for runs directory
    runs_dir = run_dir / "runs"
    if runs_dir.exists():
        for run_file in sorted(runs_dir.glob("run_*.json")):
            with open(run_file) as f:
                run = json.load(f)
            pnl = run.get("pnl_series", [run.get("final_pnl", 0)])
            pnl_per_run.append(np.array(pnl))
            kill_flags.append(run.get("kill_switch", False))
    
    if not pnl_per_run:
        raise ValueError(f"No run data found in {run_dir}")
    
    return pnl_per_run, kill_flags


def load_baseline_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load baseline metrics from a prior run directory.
    
    Args:
        run_dir: Path to the baseline run directory
        
    Returns:
        Dictionary of baseline metrics, or None if not found
    """
    mc_summary = run_dir / "mc_summary.json"
    if not mc_summary.exists():
        return None
    
    with open(mc_summary) as f:
        return json.load(f)


# =============================================================================
# Convenience Function
# =============================================================================

def evaluate_promotion(
    candidate_dir: Path,
    baseline_dir: Optional[Path] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    seed: int = 42,
    kill_threshold: Optional[float] = None,
    drawdown_threshold: Optional[float] = None,
    pnl_threshold: Optional[float] = None,
    require_strict_dominance: bool = True,
) -> PromotionDecision:
    """
    Evaluate a candidate run for promotion.
    
    Args:
        candidate_dir: Path to candidate run directory
        baseline_dir: Path to baseline run directory (optional)
        alpha: Significance level for confidence intervals
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility
        kill_threshold: Maximum allowed kill rate
        drawdown_threshold: Maximum allowed drawdown
        pnl_threshold: Minimum required PnL lower CI
        require_strict_dominance: Require strict dominance over baseline
        
    Returns:
        PromotionDecision with outcome and explanation
    """
    gate = PromotionGate(
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        seed=seed,
        kill_threshold=kill_threshold,
        drawdown_threshold=drawdown_threshold,
        pnl_threshold=pnl_threshold,
        require_strict_dominance=require_strict_dominance,
    )
    
    try:
        candidate_pnl, candidate_kills = load_run_data(candidate_dir)
        
        baseline_pnl = None
        baseline_kills = None
        if baseline_dir is not None:
            baseline_pnl, baseline_kills = load_run_data(baseline_dir)
        
        return gate.evaluate(
            candidate_pnl=candidate_pnl,
            candidate_kills=candidate_kills,
            baseline_pnl=baseline_pnl,
            baseline_kills=baseline_kills,
            candidate_path=candidate_dir,
            baseline_path=baseline_dir,
        )
    except Exception as e:
        return PromotionDecision(
            outcome=PromotionOutcome.ERROR,
            candidate_path=candidate_dir,
            baseline_path=baseline_dir,
            candidate_metrics=None,
            baseline_metrics=None,
            comparison=None,
            error_message=str(e),
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

