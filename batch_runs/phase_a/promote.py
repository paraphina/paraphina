"""
promote.py

Implements promotion logic:
- Loads Pareto set from optimization run
- Filters by budget constraints
- Selects single winner via deterministic rule
- Copies winning config to promoted folder
- Writes PROMOTION_RECORD.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schemas import (
    BudgetConfig,
    CandidateConfig,
    EvalResult,
    PromotionRecord,
    TierBudget,
)
from .optimize import load_budgets_from_yaml, load_default_budgets


# ===========================================================================
# Paths
# ===========================================================================

ROOT = Path(__file__).resolve().parents[2]
PROMOTED_DIR = ROOT / "configs" / "presets" / "promoted"

RESEARCH_SUITE = ROOT / "scenarios" / "suites" / "research_v1.yaml"
ADVERSARIAL_SUITE = ROOT / "scenarios" / "suites" / "adversarial_regression_v1.yaml"


# ===========================================================================
# Load optimization results
# ===========================================================================

def load_pareto_results(study_dir: Path) -> List[EvalResult]:
    """
    Load Pareto results from a study directory.
    
    Reads pareto.csv and corresponding eval_result.json files.
    """
    pareto_csv = study_dir / "pareto.csv"
    
    if not pareto_csv.exists():
        return []
    
    results: List[EvalResult] = []
    
    with open(pareto_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trial_dir = Path(row.get("trial_dir", ""))
            if trial_dir.exists():
                result = load_eval_result(trial_dir)
                if result:
                    results.append(result)
    
    return results


def load_all_results(study_dir: Path) -> List[EvalResult]:
    """
    Load all results from a study directory.
    
    Reads runs.csv and corresponding eval_result.json files.
    """
    runs_csv = study_dir / "runs.csv"
    
    if not runs_csv.exists():
        return []
    
    results: List[EvalResult] = []
    
    with open(runs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trial_dir = Path(row.get("trial_dir", ""))
            if trial_dir.exists():
                result = load_eval_result(trial_dir)
                if result:
                    results.append(result)
    
    return results


def load_eval_result(trial_dir: Path) -> Optional[EvalResult]:
    """Load EvalResult from a trial directory's eval_result.json."""
    result_file = trial_dir / "eval_result.json"
    
    if not result_file.exists():
        return None
    
    try:
        with open(result_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
    
    # Reconstruct CandidateConfig
    config_data = data.get("config", {})
    config = CandidateConfig(
        candidate_id=config_data.get("candidate_id", ""),
        profile=config_data.get("profile", "balanced"),
        hedge_band_base=float(config_data.get("hedge_band_base", 0.05)),
        mm_size_eta=float(config_data.get("mm_size_eta", 1.0)),
        vol_ref=float(config_data.get("vol_ref", 0.10)),
        daily_loss_limit=float(config_data.get("daily_loss_limit", 1000.0)),
        init_q_tao=float(config_data.get("init_q_tao", 0.0)),
        hedge_max_step=float(config_data.get("hedge_max_step", 10.0)),
    )
    
    mc = data.get("monte_carlo", {})
    research = data.get("research", {})
    adversarial = data.get("adversarial", {})
    
    return EvalResult(
        candidate_id=data.get("candidate_id", ""),
        trial_id=data.get("trial_id", ""),
        config=config,
        trial_dir=Path(data.get("trial_dir", "")),
        research_mean_pnl=float(research.get("mean_pnl", float("nan"))),
        research_kill_prob=float(research.get("kill_prob", float("nan"))),
        research_drawdown_cvar=float(research.get("drawdown_cvar", float("nan"))),
        adversarial_mean_pnl=float(adversarial.get("mean_pnl", float("nan"))),
        adversarial_kill_prob=float(adversarial.get("kill_prob", float("nan"))),
        adversarial_drawdown_cvar=float(adversarial.get("drawdown_cvar", float("nan"))),
        mc_mean_pnl=float(mc.get("mean_pnl", float("nan"))),
        mc_pnl_cvar=float(mc.get("pnl_cvar", float("nan"))),
        mc_drawdown_cvar=float(mc.get("drawdown_cvar", float("nan"))),
        mc_kill_prob_point=float(mc.get("kill_prob_point", float("nan"))),
        mc_kill_prob_ci_upper=float(mc.get("kill_prob_ci_upper", float("nan"))),
        mc_total_runs=int(mc.get("total_runs", 0)),
        evidence_verified=data.get("evidence_verified", False),
        evidence_errors=data.get("evidence_errors", []),
        research_suite_passed=research.get("passed", True),
        adversarial_suite_passed=adversarial.get("passed", True),
        duration_sec=float(data.get("duration_sec", 0.0)),
        error_message=data.get("error_message"),
    )


# ===========================================================================
# Winner selection
# ===========================================================================

def select_winner(
    results: List[EvalResult],
    budget: TierBudget,
) -> Optional[EvalResult]:
    """
    Select the single winning candidate that passes budget.
    
    Selection rule (deterministic):
    1. Filter by budget
    2. Best mean_pnl (highest)
    3. Tie-break: lowest drawdown_cvar
    4. Tie-break: lowest kill_prob_ci_upper
    5. Tie-break: candidate_id (alphabetical)
    
    Returns None if no candidate passes budget.
    """
    # Filter to valid candidates that pass budget
    qualifying = [r for r in results if r.is_valid and r.passes_budget(budget)]
    
    if not qualifying:
        return None
    
    # Sort with deterministic tie-breaks
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
# Promotion
# ===========================================================================

def promote_winner(
    winner: EvalResult,
    tier_name: str,
    study_name: str,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[Path, Path]:
    """
    Promote a winning candidate configuration.
    
    Copies the winning config to the promoted folder and writes
    PROMOTION_RECORD.json with full provenance.
    
    Args:
        winner: The winning EvalResult
        tier_name: Budget tier name (e.g., "balanced")
        study_name: Study name for provenance
        output_dir: Override output directory (defaults to configs/presets/promoted/)
        verbose: Print progress
    
    Returns:
        (config_file_path, record_file_path)
    """
    if output_dir is None:
        output_dir = PROMOTED_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filenames
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    config_name = f"{tier_name}_{timestamp}_{winner.candidate_id}.env"
    record_name = f"PROMOTION_RECORD_{tier_name}_{timestamp}_{winner.candidate_id}.json"
    
    config_path = output_dir / config_name
    record_path = output_dir / record_name
    
    if verbose:
        print(f"[promote] Promoting {winner.candidate_id} for tier {tier_name}")
        print(f"  Config: {config_path}")
        print(f"  Record: {record_path}")
    
    # Write config file
    env_overlay = winner.config.to_env_overlay()
    
    with open(config_path, "w") as f:
        f.write(f"# Promoted configuration for {tier_name} tier\n")
        f.write(f"# Candidate: {winner.candidate_id}\n")
        f.write(f"# Study: {study_name}\n")
        f.write(f"# Promoted at: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"#\n")
        f.write(f"# Metrics at promotion:\n")
        f.write(f"#   mean_pnl: {winner.mc_mean_pnl:.4f}\n")
        f.write(f"#   kill_prob_ci_upper: {winner.mc_kill_prob_ci_upper:.4f}\n")
        f.write(f"#   drawdown_cvar: {winner.mc_drawdown_cvar:.4f}\n")
        f.write(f"#\n\n")
        for k, v in sorted(env_overlay.items()):
            f.write(f"export {k}={v}\n")
    
    # Compute config hash
    with open(config_path, "rb") as f:
        config_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Build promotion record
    commands_executed = [
        f"monte_carlo --runs {winner.mc_total_runs} --seed 42 --output-dir {winner.trial_dir}/monte_carlo",
    ]
    
    if RESEARCH_SUITE.exists():
        commands_executed.append(
            f"sim_eval suite {RESEARCH_SUITE} --out-dir {winner.trial_dir}/research"
        )
    
    if ADVERSARIAL_SUITE.exists():
        commands_executed.append(
            f"sim_eval suite {ADVERSARIAL_SUITE} --out-dir {winner.trial_dir}/adversarial"
        )
    
    # Collect evidence verification results
    evidence_results: List[Dict[str, Any]] = []
    
    mc_dir = winner.trial_dir / "monte_carlo"
    if mc_dir.exists():
        evidence_results.append({
            "path": str(mc_dir),
            "type": "evidence_pack",
            "verified": winner.evidence_verified,
        })
    
    research_dir = winner.trial_dir / "research"
    if research_dir.exists():
        evidence_results.append({
            "path": str(research_dir),
            "type": "evidence_tree",
            "verified": winner.research_suite_passed,
        })
    
    adversarial_dir = winner.trial_dir / "adversarial"
    if adversarial_dir.exists():
        evidence_results.append({
            "path": str(adversarial_dir),
            "type": "evidence_tree",
            "verified": winner.adversarial_suite_passed,
        })
    
    record = PromotionRecord(
        promoted_at=datetime.utcnow().isoformat() + "Z",
        study_name=study_name,
        baseline_identifier=f"paraphina_phase_a_{study_name}",
        candidate_hash=config_hash,
        candidate_id=winner.candidate_id,
        trial_id=winner.trial_id,
        config=winner.config,
        env_overlay=env_overlay,
        commands_executed=commands_executed,
        seeds_used=[42],  # Default seed used in evaluation
        research_suite_path=str(RESEARCH_SUITE) if RESEARCH_SUITE.exists() else "",
        adversarial_suite_path=str(ADVERSARIAL_SUITE) if ADVERSARIAL_SUITE.exists() else "",
        evidence_verified=winner.evidence_verified,
        evidence_verification_results=evidence_results,
        metrics={
            "mean_pnl": winner.mc_mean_pnl,
            "pnl_cvar": winner.mc_pnl_cvar,
            "drawdown_cvar": winner.mc_drawdown_cvar,
            "kill_prob_point": winner.mc_kill_prob_point,
            "kill_prob_ci_upper": winner.mc_kill_prob_ci_upper,
            "total_runs": winner.mc_total_runs,
        },
        budget_tier=tier_name,
    )
    
    # Write record
    with open(record_path, "w") as f:
        json.dump(record.to_dict(), f, indent=2)
    
    if verbose:
        print(f"  ✓ Promotion complete")
    
    return config_path, record_path


# ===========================================================================
# Main promote function
# ===========================================================================

def promote(
    study_name: str,
    runs_dir: Path,
    tier: Optional[str] = None,
    budgets_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Optional[Tuple[Path, Path]]]:
    """
    Promote winning configurations from an optimization study.
    
    Args:
        study_name: Name of the study to promote from
        runs_dir: Base runs directory
        tier: Specific tier to promote (None = all tiers)
        budgets_file: Path to budgets.yaml (uses defaults if None)
        output_dir: Override output directory
        verbose: Print progress
    
    Returns:
        Dict mapping tier name to (config_path, record_path) or None if no winner
    """
    study_dir = runs_dir / "phaseA" / study_name
    
    if not study_dir.exists():
        raise ValueError(f"Study directory not found: {study_dir}")
    
    # Load budgets
    if budgets_file and budgets_file.exists():
        budgets = load_budgets_from_yaml(budgets_file)
    else:
        budgets = load_default_budgets()
    
    # Load all results (not just Pareto - budget selection considers all valid)
    results = load_all_results(study_dir)
    
    if not results:
        raise ValueError(f"No results found in study: {study_name}")
    
    if verbose:
        print(f"[promote] Loaded {len(results)} results from {study_name}")
        valid_count = sum(1 for r in results if r.is_valid)
        print(f"  Valid results: {valid_count}/{len(results)}")
    
    # Determine which tiers to promote
    tiers_to_promote = [tier] if tier else list(budgets.tiers.keys())
    
    promotions: Dict[str, Optional[Tuple[Path, Path]]] = {}
    
    for tier_name in tiers_to_promote:
        if tier_name not in budgets.tiers:
            if verbose:
                print(f"  Warning: Unknown tier '{tier_name}', skipping")
            continue
        
        budget = budgets.tiers[tier_name]
        
        if verbose:
            print(f"\n[promote] Processing tier: {tier_name}")
            print(f"  Budget: max_kill={budget.max_kill_prob:.2%}, "
                  f"max_dd_cvar={budget.max_drawdown_cvar:.0f}, "
                  f"min_pnl={budget.min_mean_pnl:.0f}")
        
        winner = select_winner(results, budget)
        
        if winner is None:
            if verbose:
                print(f"  ✗ No candidate meets budget for tier {tier_name}")
            promotions[tier_name] = None
            continue
        
        if verbose:
            print(f"  Winner: {winner.candidate_id}")
            print(f"    mean_pnl={winner.mc_mean_pnl:.2f}, "
                  f"kill_ci={winner.mc_kill_prob_ci_upper:.3f}, "
                  f"dd_cvar={winner.mc_drawdown_cvar:.2f}")
        
        # Verify evidence is valid before promotion
        if not winner.evidence_verified:
            if verbose:
                print(f"  ✗ Evidence verification failed - CANNOT PROMOTE")
            promotions[tier_name] = None
            continue
        
        # Promote
        config_path, record_path = promote_winner(
            winner=winner,
            tier_name=tier_name,
            study_name=study_name,
            output_dir=output_dir,
            verbose=verbose,
        )
        
        promotions[tier_name] = (config_path, record_path)
    
    # Summary
    if verbose:
        print(f"\n[promote] Promotion summary:")
        for tier_name, paths in promotions.items():
            if paths:
                print(f"  {tier_name}: ✓ Promoted -> {paths[0].name}")
            else:
                print(f"  {tier_name}: ✗ No promotion")
    
    return promotions


# ===========================================================================
# Validation helpers
# ===========================================================================

def validate_promotion(record_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate a promotion record.
    
    Checks:
    - Record file exists and parses
    - Config file exists and matches hash
    - Evidence was verified
    
    Returns (valid, error_messages)
    """
    errors: List[str] = []
    
    if not record_path.exists():
        return False, [f"Record file not found: {record_path}"]
    
    try:
        with open(record_path) as f:
            record = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return False, [f"Failed to parse record: {e}"]
    
    # Check evidence was verified
    if not record.get("evidence_verified", False):
        errors.append("Evidence was not verified at promotion time")
    
    # Check config file exists
    config_name = record_path.name.replace("PROMOTION_RECORD_", "").replace(".json", ".env")
    config_path = record_path.parent / config_name
    
    if not config_path.exists():
        errors.append(f"Config file not found: {config_path}")
    else:
        # Verify hash
        with open(config_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        
        expected_hash = record.get("candidate_hash", "")
        if actual_hash != expected_hash:
            errors.append(
                f"Config hash mismatch: expected {expected_hash[:12]}..., "
                f"got {actual_hash[:12]}..."
            )
    
    return len(errors) == 0, errors

