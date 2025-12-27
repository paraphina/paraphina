"""
Phase B: Confidence-Aware Statistical Gating for Promotion Decisions.

This phase introduces statistically rigorous promotion criteria based on
confidence intervals and dominance testing against a baseline.

Modules:
- confidence: Bootstrap & CI engine for computing confidence intervals
- compare: Candidate vs baseline comparison with dominance checks
- gate: Promotion gate with statistical tests
- cli: Command-line interface

Usage:
    python3 -m batch_runs.phase_b.cli \
        --candidate-run <path> \
        --baseline-run <path> \
        --out-dir <path> \
        --alpha 0.05
"""

from batch_runs.phase_b.confidence import (
    BlockBootstrap,
    bootstrap_ci,
    compute_mean,
    compute_median,
    compute_cvar,
    compute_max_drawdown,
)

from batch_runs.phase_b.compare import (
    ComparisonResult,
    MetricComparison,
    compare_runs,
    check_dominance,
)

from batch_runs.phase_b.gate import (
    PromotionDecision,
    PromotionGate,
    evaluate_promotion,
)

__all__ = [
    # confidence
    "BlockBootstrap",
    "bootstrap_ci",
    "compute_mean",
    "compute_median",
    "compute_cvar",
    "compute_max_drawdown",
    # compare
    "ComparisonResult",
    "MetricComparison",
    "compare_runs",
    "check_dominance",
    # gate
    "PromotionDecision",
    "PromotionGate",
    "evaluate_promotion",
]

