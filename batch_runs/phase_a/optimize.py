"""
optimize.py

Multi-objective search loop with Pareto front selection.

Implements random search + Pareto front selection using only stdlib.
Supports:
- --trials N
- --mc-runs N
- --study <name>
- --out <runs_dir>
- --seed <int> for determinism

Outputs:
- runs.csv (one row per trial)
- pareto.csv (pareto set)
- report.md (top candidates, budgets pass/fail)
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schemas import (
    BudgetConfig,
    CandidateConfig,
    EvalResult,
    ParetoCandidate,
    TierBudget,
)
from .evaluate import evaluate_candidates


# ===========================================================================
# Knob ranges for random search
# ===========================================================================

KNOB_RANGES = {
    "hedge_band_base": [0.02, 0.03, 0.05, 0.075, 0.10, 0.15],
    "mm_size_eta": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "vol_ref": [0.05, 0.075, 0.10, 0.125, 0.15],
    "daily_loss_limit": [500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0],
    "init_q_tao": [-30.0, -15.0, 0.0, 15.0, 30.0],
    "hedge_max_step": [5.0, 10.0, 15.0, 20.0],
}

PROFILES = ["balanced", "conservative", "aggressive"]


# ===========================================================================
# Candidate generation
# ===========================================================================

def generate_random_candidates(
    n_candidates: int,
    seed: int,
    profiles: Optional[List[str]] = None,
) -> List[CandidateConfig]:
    """
    Generate random candidate configurations using seeded RNG.
    
    Deterministic given seed.
    """
    rng = random.Random(seed)
    if profiles is None:
        profiles = PROFILES
    
    candidates: List[CandidateConfig] = []
    
    for i in range(n_candidates):
        profile = rng.choice(profiles)
        
        config = CandidateConfig(
            candidate_id="",  # Will be set below
            profile=profile,
            hedge_band_base=rng.choice(KNOB_RANGES["hedge_band_base"]),
            mm_size_eta=rng.choice(KNOB_RANGES["mm_size_eta"]),
            vol_ref=rng.choice(KNOB_RANGES["vol_ref"]),
            daily_loss_limit=rng.choice(KNOB_RANGES["daily_loss_limit"]),
            init_q_tao=rng.choice(KNOB_RANGES["init_q_tao"]),
            hedge_max_step=rng.choice(KNOB_RANGES["hedge_max_step"]),
        )
        
        # Set candidate_id from config hash
        config = CandidateConfig(
            candidate_id=config.config_hash(),
            profile=config.profile,
            hedge_band_base=config.hedge_band_base,
            mm_size_eta=config.mm_size_eta,
            vol_ref=config.vol_ref,
            daily_loss_limit=config.daily_loss_limit,
            init_q_tao=config.init_q_tao,
            hedge_max_step=config.hedge_max_step,
        )
        
        candidates.append(config)
    
    return candidates


def generate_smoke_candidates(profiles: Optional[List[str]] = None) -> List[CandidateConfig]:
    """
    Generate minimal smoke test candidates (one per profile).
    
    Fixed, deterministic for quick testing.
    """
    if profiles is None:
        profiles = PROFILES
    
    candidates: List[CandidateConfig] = []
    
    for profile in profiles:
        config = CandidateConfig(
            candidate_id="",
            profile=profile,
            hedge_band_base=0.05,
            mm_size_eta=1.0,
            vol_ref=0.10,
            daily_loss_limit=1000.0,
            init_q_tao=0.0,
            hedge_max_step=10.0,
        )
        config = CandidateConfig(
            candidate_id=config.config_hash(),
            profile=config.profile,
            hedge_band_base=config.hedge_band_base,
            mm_size_eta=config.mm_size_eta,
            vol_ref=config.vol_ref,
            daily_loss_limit=config.daily_loss_limit,
            init_q_tao=config.init_q_tao,
            hedge_max_step=config.hedge_max_step,
        )
        candidates.append(config)
    
    return candidates


# ===========================================================================
# Pareto frontier computation
# ===========================================================================

def compute_pareto_frontier(results: List[EvalResult]) -> List[EvalResult]:
    """
    Compute the Pareto frontier from a list of evaluation results.
    
    Returns candidates that are not dominated by any other valid candidate.
    
    Objectives (all to be minimized after transformation):
    - Minimize -mean_pnl (i.e., maximize mean_pnl)
    - Minimize kill_prob_ci_upper
    - Minimize drawdown_cvar
    """
    # Filter to valid results only
    valid = [r for r in results if r.is_valid]
    
    if not valid:
        return []
    
    # Convert to ParetoCandidate for comparison
    candidates = [ParetoCandidate.from_result(r) for r in valid]
    
    frontier: List[EvalResult] = []
    
    for i, candidate in enumerate(candidates):
        dominated = False
        for j, other in enumerate(candidates):
            if i != j and other.dominates(candidate):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate.result)
    
    # Sort frontier deterministically by candidate_id
    frontier.sort(key=lambda r: r.candidate_id)
    
    return frontier


# ===========================================================================
# Budget filtering
# ===========================================================================

def filter_by_budget(
    results: List[EvalResult],
    budget: TierBudget,
) -> List[EvalResult]:
    """
    Filter results to those that pass the given tier budget.
    """
    return [r for r in results if r.passes_budget(budget)]


def select_winner_deterministic(
    results: List[EvalResult],
    budget: TierBudget,
) -> Optional[EvalResult]:
    """
    Select the single winning candidate from a list of results.
    
    Selection rule (deterministic):
    1. Filter by budget
    2. Best mean_pnl (highest)
    3. Tie-break: lowest drawdown_cvar
    4. Tie-break: lowest kill_prob_ci_upper
    5. Tie-break: candidate_id (alphabetical)
    
    Returns None if no candidate passes budget.
    """
    qualifying = filter_by_budget(results, budget)
    
    if not qualifying:
        return None
    
    # Sort with deterministic tie-breaks
    # Primary: mean_pnl descending (best first)
    # Secondary: drawdown_cvar ascending (lower is better)
    # Tertiary: kill_prob_ci_upper ascending (lower is better)
    # Final: candidate_id alphabetical (for determinism)
    qualifying.sort(
        key=lambda r: (
            -r.mc_mean_pnl,           # Descending (higher is better)
            r.mc_drawdown_cvar,       # Ascending (lower is better)
            r.mc_kill_prob_ci_upper,  # Ascending (lower is better)
            r.candidate_id,           # Alphabetical tie-break
        )
    )
    
    return qualifying[0]


# ===========================================================================
# Output writing (stdlib csv, no pandas)
# ===========================================================================

def result_to_row(result: EvalResult) -> Dict[str, Any]:
    """Convert EvalResult to a flat dict for CSV."""
    config = result.config
    return {
        "candidate_id": result.candidate_id,
        "trial_id": result.trial_id,
        "profile": config.profile,
        "hedge_band_base": config.hedge_band_base,
        "mm_size_eta": config.mm_size_eta,
        "vol_ref": config.vol_ref,
        "daily_loss_limit": config.daily_loss_limit,
        "init_q_tao": config.init_q_tao,
        "hedge_max_step": config.hedge_max_step,
        "mc_mean_pnl": result.mc_mean_pnl,
        "mc_pnl_cvar": result.mc_pnl_cvar,
        "mc_drawdown_cvar": result.mc_drawdown_cvar,
        "mc_kill_prob_point": result.mc_kill_prob_point,
        "mc_kill_prob_ci_upper": result.mc_kill_prob_ci_upper,
        "mc_total_runs": result.mc_total_runs,
        "research_mean_pnl": result.research_mean_pnl,
        "research_kill_prob": result.research_kill_prob,
        "adversarial_mean_pnl": result.adversarial_mean_pnl,
        "adversarial_kill_prob": result.adversarial_kill_prob,
        "evidence_verified": result.evidence_verified,
        "is_valid": result.is_valid,
        "research_passed": result.research_suite_passed,
        "adversarial_passed": result.adversarial_suite_passed,
        "duration_sec": result.duration_sec,
        "trial_dir": str(result.trial_dir),
    }


def write_runs_csv(results: List[EvalResult], path: Path) -> None:
    """Write per-trial results to CSV using stdlib csv module."""
    if not results:
        path.write_text("")
        return
    
    rows = [result_to_row(r) for r in results]
    
    # Get all keys, sorted for determinism
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)
    
    # Put important columns first
    priority_cols = [
        "candidate_id", "trial_id", "profile", "is_valid",
        "mc_mean_pnl", "mc_kill_prob_ci_upper", "mc_drawdown_cvar",
        "evidence_verified",
    ]
    fieldnames = [c for c in priority_cols if c in fieldnames] + \
                 [c for c in fieldnames if c not in priority_cols]
    
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pareto_csv(results: List[EvalResult], path: Path) -> None:
    """Write Pareto frontier to CSV."""
    if not results:
        path.write_text("# Empty Pareto frontier\n")
        return
    
    rows = []
    for i, r in enumerate(results):
        row = result_to_row(r)
        row["pareto_rank"] = i + 1
        rows.append(row)
    
    # Get all keys, with pareto_rank first
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = ["pareto_rank"] + sorted(k for k in all_keys if k != "pareto_rank")
    
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report_md(
    results: List[EvalResult],
    pareto: List[EvalResult],
    winners: Dict[str, Optional[EvalResult]],
    budgets: BudgetConfig,
    study_name: str,
    path: Path,
) -> None:
    """Write report.md with top candidates and budget pass/fail."""
    lines = [
        f"# Phase A Optimization Report: {study_name}",
        f"",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        f"",
        f"## Summary",
        f"",
        f"- Total trials: {len(results)}",
        f"- Valid trials: {sum(1 for r in results if r.is_valid)}",
        f"- Pareto frontier size: {len(pareto)}",
        f"",
    ]
    
    # Budget results
    lines.append("## Budget Evaluation")
    lines.append("")
    lines.append("| Tier | Max Kill Prob | Max DD CVaR | Min PnL | Winner | Pass? |")
    lines.append("|------|--------------|-------------|---------|--------|-------|")
    
    for tier_name, budget in sorted(budgets.tiers.items()):
        winner = winners.get(tier_name)
        if winner:
            winner_str = f"`{winner.candidate_id}`"
            pass_str = "✓ PASS"
        else:
            winner_str = "-"
            pass_str = "✗ FAIL"
        
        lines.append(
            f"| {tier_name} | {budget.max_kill_prob:.2%} | {budget.max_drawdown_cvar:.0f} | "
            f"{budget.min_mean_pnl:.0f} | {winner_str} | {pass_str} |"
        )
    
    lines.append("")
    
    # Pareto frontier
    lines.append("## Pareto Frontier")
    lines.append("")
    
    if pareto:
        lines.append("| Rank | Candidate | Profile | Mean PnL | Kill CI Upper | DD CVaR |")
        lines.append("|------|-----------|---------|----------|---------------|---------|")
        
        for i, r in enumerate(pareto):
            lines.append(
                f"| {i+1} | `{r.candidate_id}` | {r.config.profile} | "
                f"{r.mc_mean_pnl:.2f} | {r.mc_kill_prob_ci_upper:.3f} | {r.mc_drawdown_cvar:.2f} |"
            )
    else:
        lines.append("*No valid candidates on Pareto frontier*")
    
    lines.append("")
    
    # Winner details
    lines.append("## Winner Details")
    lines.append("")
    
    for tier_name in sorted(budgets.tiers.keys()):
        winner = winners.get(tier_name)
        if winner:
            lines.append(f"### {tier_name.capitalize()} Tier Winner")
            lines.append("")
            lines.append(f"- **Candidate ID**: `{winner.candidate_id}`")
            lines.append(f"- **Profile**: {winner.config.profile}")
            lines.append(f"- **Trial Dir**: `{winner.trial_dir}`")
            lines.append("")
            lines.append("**Metrics:**")
            lines.append(f"- Mean PnL: {winner.mc_mean_pnl:.4f}")
            lines.append(f"- Kill Prob CI Upper: {winner.mc_kill_prob_ci_upper:.4f}")
            lines.append(f"- Drawdown CVaR: {winner.mc_drawdown_cvar:.4f}")
            lines.append("")
            lines.append("**Configuration:**")
            lines.append("```")
            for k, v in sorted(winner.config.to_env_overlay().items()):
                lines.append(f"export {k}={v}")
            lines.append("```")
            lines.append("")
    
    # All trials summary
    lines.append("## All Trials")
    lines.append("")
    lines.append("| Trial ID | Candidate | Valid | Mean PnL | Kill CI | DD CVaR |")
    lines.append("|----------|-----------|-------|----------|---------|---------|")
    
    for r in sorted(results, key=lambda x: x.trial_id):
        valid_str = "✓" if r.is_valid else "✗"
        pnl_str = f"{r.mc_mean_pnl:.2f}" if not _is_nan(r.mc_mean_pnl) else "NaN"
        kill_str = f"{r.mc_kill_prob_ci_upper:.3f}" if not _is_nan(r.mc_kill_prob_ci_upper) else "NaN"
        dd_str = f"{r.mc_drawdown_cvar:.2f}" if not _is_nan(r.mc_drawdown_cvar) else "NaN"
        
        lines.append(
            f"| {r.trial_id} | `{r.candidate_id}` | {valid_str} | "
            f"{pnl_str} | {kill_str} | {dd_str} |"
        )
    
    path.write_text("\n".join(lines))


def _is_nan(x: float) -> bool:
    """Check if value is NaN."""
    import math
    return math.isnan(x) if isinstance(x, float) else False


# ===========================================================================
# Main optimization function
# ===========================================================================

def optimize(
    trials: int,
    mc_runs: int,
    study_name: str,
    out_dir: Path,
    seed: int,
    mc_ticks: int = 600,
    budgets: Optional[BudgetConfig] = None,
    verbose: bool = True,
) -> Tuple[List[EvalResult], List[EvalResult], Dict[str, Optional[EvalResult]]]:
    """
    Run multi-objective optimization with Pareto front selection.
    
    Args:
        trials: Number of candidate trials to run
        mc_runs: Number of Monte Carlo runs per candidate
        study_name: Name of the study (used for directory structure)
        out_dir: Base output directory
        seed: Random seed for determinism
        mc_ticks: Ticks per Monte Carlo run
        budgets: Budget configuration (uses default if None)
        verbose: Print progress
    
    Returns:
        (all_results, pareto_frontier, winners_by_tier)
    """
    if budgets is None:
        budgets = load_default_budgets()
    
    if verbose:
        print(f"[optimize] Starting Phase A optimization")
        print(f"  Study: {study_name}")
        print(f"  Trials: {trials}")
        print(f"  MC runs: {mc_runs}")
        print(f"  Seed: {seed}")
        print(f"  Output: {out_dir}")
    
    # Create output directory
    study_dir = out_dir / "phaseA" / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate candidates
    if verbose:
        print(f"\n[optimize] Generating {trials} random candidates (seed={seed})...")
    
    if trials <= len(PROFILES):
        # Use smoke candidates for very small trial counts
        candidates = generate_smoke_candidates()[:trials]
    else:
        candidates = generate_random_candidates(trials, seed)
    
    if verbose:
        print(f"  Generated {len(candidates)} candidates")
    
    # Evaluate all candidates
    if verbose:
        print(f"\n[optimize] Evaluating candidates...")
    
    results = evaluate_candidates(
        configs=candidates,
        study_name=study_name,
        runs_dir=out_dir,
        mc_runs=mc_runs,
        mc_ticks=mc_ticks,
        seed=seed,
        verbose=verbose,
    )
    
    # Compute Pareto frontier
    if verbose:
        print(f"\n[optimize] Computing Pareto frontier...")
    
    pareto = compute_pareto_frontier(results)
    
    if verbose:
        print(f"  Pareto frontier size: {len(pareto)}")
        for i, r in enumerate(pareto):
            print(f"    {i+1}. {r.candidate_id}: pnl={r.mc_mean_pnl:.2f}, "
                  f"kill_ci={r.mc_kill_prob_ci_upper:.3f}, dd_cvar={r.mc_drawdown_cvar:.2f}")
    
    # Select winners per budget tier
    if verbose:
        print(f"\n[optimize] Selecting winners per budget tier...")
    
    winners: Dict[str, Optional[EvalResult]] = {}
    for tier_name, budget in budgets.tiers.items():
        # Consider all valid results (not just Pareto) for budget selection
        valid_results = [r for r in results if r.is_valid]
        winner = select_winner_deterministic(valid_results, budget)
        winners[tier_name] = winner
        
        if verbose:
            if winner:
                print(f"  {tier_name}: {winner.candidate_id} (pnl={winner.mc_mean_pnl:.2f})")
            else:
                print(f"  {tier_name}: No candidate meets budget")
    
    # Write outputs
    if verbose:
        print(f"\n[optimize] Writing output files...")
    
    runs_csv = study_dir / "runs.csv"
    write_runs_csv(results, runs_csv)
    if verbose:
        print(f"  Wrote: {runs_csv}")
    
    pareto_csv = study_dir / "pareto.csv"
    write_pareto_csv(pareto, pareto_csv)
    if verbose:
        print(f"  Wrote: {pareto_csv}")
    
    report_md = study_dir / "report.md"
    write_report_md(results, pareto, winners, budgets, study_name, report_md)
    if verbose:
        print(f"  Wrote: {report_md}")
    
    # Write study metadata
    meta = {
        "schema_version": 1,
        "study_name": study_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "trials": trials,
        "mc_runs": mc_runs,
        "mc_ticks": mc_ticks,
        "valid_count": sum(1 for r in results if r.is_valid),
        "pareto_size": len(pareto),
        "winners": {
            tier: (w.candidate_id if w else None)
            for tier, w in winners.items()
        },
    }
    
    meta_file = study_dir / "study_meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"  Wrote: {meta_file}")
    
    if verbose:
        print(f"\n[optimize] Done!")
        print(f"  Results in: {study_dir}")
    
    return results, pareto, winners


# ===========================================================================
# Default budgets
# ===========================================================================

def load_default_budgets() -> BudgetConfig:
    """Load default budgets (fallback if budgets.yaml not found)."""
    return BudgetConfig(
        tiers={
            "conservative": TierBudget(
                tier_name="conservative",
                max_kill_prob=0.05,
                max_drawdown_cvar=500.0,
                min_mean_pnl=10.0,
            ),
            "balanced": TierBudget(
                tier_name="balanced",
                max_kill_prob=0.10,
                max_drawdown_cvar=1000.0,
                min_mean_pnl=20.0,
            ),
            "aggressive": TierBudget(
                tier_name="aggressive",
                max_kill_prob=0.15,
                max_drawdown_cvar=2000.0,
                min_mean_pnl=30.0,
            ),
            "research": TierBudget(
                tier_name="research",
                max_kill_prob=0.20,
                max_drawdown_cvar=3000.0,
                min_mean_pnl=0.0,
            ),
        }
    )


def load_budgets_from_yaml(path: Path) -> BudgetConfig:
    """
    Load budgets from a YAML file.
    
    Uses stdlib yaml-like parsing (simple key: value format).
    """
    if not path.exists():
        return load_default_budgets()
    
    # Simple YAML parser for our format
    content = path.read_text()
    
    # Parse tiers section
    tiers: Dict[str, TierBudget] = {}
    current_tier: Optional[str] = None
    tier_data: Dict[str, float] = {}
    
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        # Detect tier name (indented under "tiers:")
        if line.startswith("tiers:"):
            continue
        
        # Check for tier name (two-space indent, ends with :)
        if line.endswith(":") and not line.startswith(" "):
            # New top-level tier
            if current_tier and tier_data:
                tiers[current_tier] = TierBudget(
                    tier_name=current_tier,
                    max_kill_prob=tier_data.get("max_kill_prob", 0.10),
                    max_drawdown_cvar=tier_data.get("max_drawdown_cvar", 1000.0),
                    min_mean_pnl=tier_data.get("min_mean_pnl", 10.0),
                )
            current_tier = line[:-1].strip()
            tier_data = {}
        elif ":" in line and current_tier:
            # Key-value pair
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            try:
                tier_data[key] = float(val)
            except ValueError:
                pass
    
    # Don't forget the last tier
    if current_tier and tier_data:
        tiers[current_tier] = TierBudget(
            tier_name=current_tier,
            max_kill_prob=tier_data.get("max_kill_prob", 0.10),
            max_drawdown_cvar=tier_data.get("max_drawdown_cvar", 1000.0),
            min_mean_pnl=tier_data.get("min_mean_pnl", 10.0),
        )
    
    if not tiers:
        return load_default_budgets()
    
    return BudgetConfig(tiers=tiers)

