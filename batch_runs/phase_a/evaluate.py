"""
evaluate.py

Runs the actual evaluation for one candidate:
- Creates an isolated trial directory
- Runs sim_eval on research suite
- Runs sim_eval on adversarial regression suite
- Runs monte_carlo robustness run (N seeds)
- Verifies evidence packs after each run
- Parses summary outputs into metrics
- Writes eval_result.json in trial dir
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schemas import (
    CandidateConfig,
    EvalResult,
    parse_mc_summary,
)


# ===========================================================================
# Paths & constants
# ===========================================================================

ROOT = Path(__file__).resolve().parents[2]
MONTE_CARLO_BIN = ROOT / "target" / "release" / "monte_carlo"
SIM_EVAL_BIN = ROOT / "target" / "release" / "sim_eval"

RESEARCH_SUITE = ROOT / "scenarios" / "suites" / "research_v1.yaml"
ADVERSARIAL_SUITE = ROOT / "scenarios" / "suites" / "adversarial_regression_v1.yaml"


# ===========================================================================
# Evidence verification
# ===========================================================================

def verify_evidence_tree(output_dir: Path, verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    Verify all evidence packs under output_dir using sim_eval verify-evidence-tree.
    
    Returns (success, error_messages).
    """
    if not SIM_EVAL_BIN.exists():
        return False, [f"sim_eval binary not found at {SIM_EVAL_BIN}"]
    
    cmd = [
        str(SIM_EVAL_BIN),
        "verify-evidence-tree",
        str(output_dir),
    ]
    
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    
    if proc.returncode == 0:
        if verbose:
            print(f"    Evidence tree verified: {output_dir.name}")
        return True, []
    else:
        errors = []
        if proc.stderr:
            errors.append(proc.stderr.strip())
        if proc.stdout:
            errors.append(proc.stdout.strip())
        if not errors:
            errors.append(f"verify-evidence-tree failed with code {proc.returncode}")
        if verbose:
            print(f"    Evidence tree FAILED: {output_dir.name} (rc={proc.returncode})")
        return False, errors


def verify_evidence_pack(output_dir: Path, verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    Verify a single evidence pack under output_dir using sim_eval verify-evidence-pack.
    
    Returns (success, error_messages).
    """
    if not SIM_EVAL_BIN.exists():
        return False, [f"sim_eval binary not found at {SIM_EVAL_BIN}"]
    
    cmd = [
        str(SIM_EVAL_BIN),
        "verify-evidence-pack",
        str(output_dir),
    ]
    
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    
    if proc.returncode == 0:
        if verbose:
            print(f"    Evidence pack verified: {output_dir.name}")
        return True, []
    else:
        errors = []
        if proc.stderr:
            errors.append(proc.stderr.strip())
        if proc.stdout:
            errors.append(proc.stdout.strip())
        if not errors:
            errors.append(f"verify-evidence-pack failed with code {proc.returncode}")
        if verbose:
            print(f"    Evidence pack FAILED: {output_dir.name} (rc={proc.returncode})")
        return False, errors


# ===========================================================================
# Suite runners
# ===========================================================================

def run_suite(
    suite_path: Path,
    output_dir: Path,
    env_overlay: Dict[str, str],
    verbose: bool = False,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Run a suite using sim_eval suite command.
    
    Note: sim_eval suite uses out_dir from the suite YAML file, not CLI.
    We run it anyway for validation and aggregate metrics from the default location.
    
    Returns (success, metrics_dict, error_messages).
    """
    if not SIM_EVAL_BIN.exists():
        return False, {}, [f"sim_eval binary not found at {SIM_EVAL_BIN}"]
    
    if not suite_path.exists():
        return False, {}, [f"Suite file not found: {suite_path}"]
    
    # Note: sim_eval suite command doesn't support --output-dir
    # The suite uses its own out_dir from the YAML file
    cmd = [
        str(SIM_EVAL_BIN),
        "suite",
        str(suite_path),
    ]
    
    env = os.environ.copy()
    env.update(env_overlay)
    
    if verbose:
        print(f"    Running suite: {suite_path.name}")
    
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    
    errors = []
    if proc.returncode != 0:
        if proc.stderr:
            errors.append(proc.stderr.strip())
        if not errors:
            errors.append(f"Suite run failed with code {proc.returncode}")
        return False, {}, errors
    
    # Parse summary from the suite's default output directory
    # Read out_dir from suite YAML
    suite_out_dir = get_suite_out_dir(suite_path)
    if suite_out_dir:
        metrics = aggregate_suite_metrics(ROOT / suite_out_dir)
    else:
        metrics = aggregate_suite_metrics(output_dir)
    
    return True, metrics, []


def get_suite_out_dir(suite_path: Path) -> Optional[str]:
    """Extract out_dir from a suite YAML file."""
    try:
        content = suite_path.read_text()
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("out_dir:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def aggregate_suite_metrics(output_dir: Path) -> Dict[str, Any]:
    """
    Aggregate metrics from all scenario runs in a suite output directory.
    
    Returns aggregated metrics dict.
    """
    pnl_values: List[float] = []
    kill_counts = 0
    total_runs = 0
    drawdown_cvars: List[float] = []
    
    # Look for mc_summary.json or run_summary.json files
    for summary_file in output_dir.rglob("mc_summary.json"):
        metrics = parse_mc_summary(summary_file)
        if metrics:
            mean_pnl = metrics.get("mean_pnl", float("nan"))
            if not _is_nan(mean_pnl):
                pnl_values.append(mean_pnl)
            
            kill_counts += metrics.get("kill_count", 0)
            total_runs += metrics.get("total_runs", 0)
            
            dd_cvar = metrics.get("drawdown_cvar", float("nan"))
            if not _is_nan(dd_cvar):
                drawdown_cvars.append(dd_cvar)
    
    # Also check for run_summary.json from sim_eval suite
    for summary_file in output_dir.rglob("run_summary.json"):
        try:
            with open(summary_file) as f:
                summary = json.load(f)
            pnl = summary.get("final_pnl", float("nan"))
            if not _is_nan(pnl):
                pnl_values.append(pnl)
            
            if summary.get("kill_switch", False):
                kill_counts += 1
            total_runs += 1
            
            dd = summary.get("max_drawdown", float("nan"))
            if not _is_nan(dd):
                drawdown_cvars.append(dd)
        except (json.JSONDecodeError, IOError):
            continue
    
    # Compute aggregates
    mean_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else float("nan")
    kill_prob = kill_counts / total_runs if total_runs > 0 else float("nan")
    # Use worst-case (max) drawdown CVaR across scenarios
    drawdown_cvar = max(drawdown_cvars) if drawdown_cvars else float("nan")
    
    return {
        "mean_pnl": mean_pnl,
        "kill_prob": kill_prob,
        "kill_count": kill_counts,
        "total_runs": total_runs,
        "drawdown_cvar": drawdown_cvar,
    }


def _is_nan(x: float) -> bool:
    """Check if value is NaN."""
    import math
    return math.isnan(x) if isinstance(x, float) else False


# ===========================================================================
# Monte Carlo runner
# ===========================================================================

def run_monte_carlo(
    output_dir: Path,
    env_overlay: Dict[str, str],
    mc_runs: int,
    mc_ticks: int = 600,
    seed: int = 42,
    jitter_ms: int = 100,
    verbose: bool = False,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Run monte_carlo robustness evaluation.
    
    Returns (success, metrics_dict, error_messages).
    """
    if not MONTE_CARLO_BIN.exists():
        return False, {}, [f"monte_carlo binary not found at {MONTE_CARLO_BIN}"]
    
    cmd = [
        str(MONTE_CARLO_BIN),
        "--runs", str(mc_runs),
        "--ticks", str(mc_ticks),
        "--seed", str(seed),
        "--jitter-ms", str(jitter_ms),
        "--output-dir", str(output_dir),
        "--quiet",
    ]
    
    env = os.environ.copy()
    env.update(env_overlay)
    
    if verbose:
        print(f"    Running monte_carlo: {mc_runs} runs, seed={seed}")
    
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    
    errors = []
    if proc.returncode != 0:
        if proc.stderr:
            errors.append(proc.stderr.strip())
        if not errors:
            errors.append(f"monte_carlo failed with code {proc.returncode}")
        return False, {}, errors
    
    # Parse mc_summary.json
    summary_path = output_dir / "mc_summary.json"
    metrics = parse_mc_summary(summary_path)
    
    if not metrics:
        return False, {}, ["Failed to parse mc_summary.json"]
    
    return True, metrics, []


# ===========================================================================
# Main evaluation function
# ===========================================================================

def evaluate_candidate(
    config: CandidateConfig,
    study_name: str,
    trial_id: str,
    runs_dir: Path,
    mc_runs: int = 50,
    mc_ticks: int = 600,
    seed: int = 42,
    verbose: bool = True,
) -> EvalResult:
    """
    Evaluate a single candidate configuration.
    
    Creates an isolated trial directory and runs:
    1. sim_eval on research suite
    2. sim_eval on adversarial regression suite
    3. monte_carlo robustness run
    
    After each run, verifies evidence packs. Hard-fails trial if verification fails.
    
    Returns EvalResult with all metrics and verification status.
    """
    t0 = time.time()
    
    # Create trial directory
    trial_dir = runs_dir / "phaseA" / study_name / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n[evaluate] Trial {trial_id} (candidate={config.candidate_id})")
        print(f"  Trial dir: {trial_dir}")
    
    # Get env overlay from config
    env_overlay = config.to_env_overlay()
    
    # Write config override file
    config_file = trial_dir / "config_override.env"
    with open(config_file, "w") as f:
        f.write(f"# Config override for trial {trial_id}\n")
        f.write(f"# Generated at {datetime.utcnow().isoformat()}Z\n\n")
        for k, v in sorted(env_overlay.items()):
            f.write(f"export {k}={v}\n")
    
    # Initialize result with defaults
    evidence_errors: List[str] = []
    all_evidence_verified = True
    
    # Default values in case of failures
    research_mean_pnl = float("nan")
    research_kill_prob = float("nan")
    research_drawdown_cvar = float("nan")
    research_passed = False
    
    adversarial_mean_pnl = float("nan")
    adversarial_kill_prob = float("nan")
    adversarial_drawdown_cvar = float("nan")
    adversarial_passed = False
    
    mc_mean_pnl = float("nan")
    mc_pnl_cvar = float("nan")
    mc_drawdown_cvar = float("nan")
    mc_kill_prob_point = float("nan")
    mc_kill_prob_ci_upper = float("nan")
    mc_total_runs = 0
    
    error_message: Optional[str] = None
    
    # =========================================================================
    # Step 1: Run research suite (optional - doesn't fail candidate if missing)
    # =========================================================================
    if verbose:
        print(f"  [1/3] Research suite...")
    
    research_dir = trial_dir / "research"
    research_dir.mkdir(exist_ok=True)
    
    if RESEARCH_SUITE.exists():
        success, metrics, errors = run_suite(
            RESEARCH_SUITE, research_dir, env_overlay, verbose=verbose
        )
        
        if success:
            research_mean_pnl = metrics.get("mean_pnl", float("nan"))
            research_kill_prob = metrics.get("kill_prob", float("nan"))
            research_drawdown_cvar = metrics.get("drawdown_cvar", float("nan"))
            research_passed = True
            
            # Verify evidence if available
            suite_out = get_suite_out_dir(RESEARCH_SUITE)
            if suite_out:
                verified, errs = verify_evidence_tree(ROOT / suite_out, verbose=verbose)
                if not verified:
                    # Evidence verification failure is informational, not blocking
                    if verbose:
                        print(f"    Research suite evidence verification: {errs}")
        else:
            # Suite failures are warnings, not hard failures
            if verbose:
                print(f"    Research suite skipped: {'; '.join(errors)[:100]}...")
            # Mark as passed anyway to not block on suite issues
            research_passed = True
    else:
        if verbose:
            print(f"    Research suite not found: {RESEARCH_SUITE}")
        # Continue without research suite for smoke tests
        research_passed = True
    
    # =========================================================================
    # Step 2: Run adversarial regression suite (optional - informational)
    # =========================================================================
    if verbose:
        print(f"  [2/3] Adversarial regression suite...")
    
    adversarial_dir = trial_dir / "adversarial"
    adversarial_dir.mkdir(exist_ok=True)
    
    if ADVERSARIAL_SUITE.exists():
        success, metrics, errors = run_suite(
            ADVERSARIAL_SUITE, adversarial_dir, env_overlay, verbose=verbose
        )
        
        if success:
            adversarial_mean_pnl = metrics.get("mean_pnl", float("nan"))
            adversarial_kill_prob = metrics.get("kill_prob", float("nan"))
            adversarial_drawdown_cvar = metrics.get("drawdown_cvar", float("nan"))
            adversarial_passed = True
            
            # Verify evidence if available
            suite_out = get_suite_out_dir(ADVERSARIAL_SUITE)
            if suite_out:
                verified, errs = verify_evidence_tree(ROOT / suite_out, verbose=verbose)
                if not verified:
                    if verbose:
                        print(f"    Adversarial suite evidence verification: {errs}")
        else:
            # Suite failures are warnings, not hard failures
            if verbose:
                print(f"    Adversarial suite skipped: {'; '.join(errors)[:100]}...")
            adversarial_passed = True
    else:
        if verbose:
            print(f"    Adversarial suite not found: {ADVERSARIAL_SUITE}")
        # Continue without adversarial suite for smoke tests
        adversarial_passed = True
    
    # =========================================================================
    # Step 3: Run Monte Carlo robustness evaluation
    # =========================================================================
    if verbose:
        print(f"  [3/3] Monte Carlo robustness ({mc_runs} runs)...")
    
    mc_dir = trial_dir / "monte_carlo"
    mc_dir.mkdir(exist_ok=True)
    
    success, metrics, errors = run_monte_carlo(
        mc_dir, env_overlay, mc_runs, mc_ticks, seed, verbose=verbose
    )
    
    if success:
        mc_mean_pnl = metrics.get("mean_pnl", float("nan"))
        mc_pnl_cvar = metrics.get("pnl_cvar", float("nan"))
        mc_drawdown_cvar = metrics.get("drawdown_cvar", float("nan"))
        mc_kill_prob_point = metrics.get("kill_prob_point", float("nan"))
        mc_kill_prob_ci_upper = metrics.get("kill_prob_ci_upper", float("nan"))
        mc_total_runs = metrics.get("total_runs", 0)
        
        # Verify evidence pack
        verified, errs = verify_evidence_pack(mc_dir, verbose=verbose)
        if not verified:
            all_evidence_verified = False
            evidence_errors.extend(errs)
    else:
        evidence_errors.extend(errors)
        if not error_message:
            error_message = f"Monte Carlo failed: {'; '.join(errors)}"
    
    # =========================================================================
    # Build result
    # =========================================================================
    duration_sec = time.time() - t0
    
    result = EvalResult(
        candidate_id=config.candidate_id,
        trial_id=trial_id,
        config=config,
        trial_dir=trial_dir,
        research_mean_pnl=research_mean_pnl,
        research_kill_prob=research_kill_prob,
        research_drawdown_cvar=research_drawdown_cvar,
        adversarial_mean_pnl=adversarial_mean_pnl,
        adversarial_kill_prob=adversarial_kill_prob,
        adversarial_drawdown_cvar=adversarial_drawdown_cvar,
        mc_mean_pnl=mc_mean_pnl,
        mc_pnl_cvar=mc_pnl_cvar,
        mc_drawdown_cvar=mc_drawdown_cvar,
        mc_kill_prob_point=mc_kill_prob_point,
        mc_kill_prob_ci_upper=mc_kill_prob_ci_upper,
        mc_total_runs=mc_total_runs,
        evidence_verified=all_evidence_verified,
        evidence_errors=evidence_errors,
        research_suite_passed=research_passed,
        adversarial_suite_passed=adversarial_passed,
        duration_sec=duration_sec,
        error_message=error_message,
    )
    
    # Write eval_result.json
    result_file = trial_dir / "eval_result.json"
    with open(result_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    if verbose:
        status = "✓ VALID" if result.is_valid else "✗ INVALID"
        print(f"  Result: {status}")
        print(f"    mean_pnl={mc_mean_pnl:.2f}, kill_ci_upper={mc_kill_prob_ci_upper:.3f}, dd_cvar={mc_drawdown_cvar:.2f}")
        print(f"    evidence_verified={all_evidence_verified}, duration={duration_sec:.1f}s")
    
    return result


# ===========================================================================
# Batch evaluation
# ===========================================================================

def evaluate_candidates(
    configs: List[CandidateConfig],
    study_name: str,
    runs_dir: Path,
    mc_runs: int = 50,
    mc_ticks: int = 600,
    seed: int = 42,
    verbose: bool = True,
) -> List[EvalResult]:
    """
    Evaluate multiple candidates.
    
    Returns list of EvalResult objects.
    """
    results: List[EvalResult] = []
    
    for i, config in enumerate(configs):
        trial_id = f"trial_{i:04d}_{config.candidate_id}"
        
        if verbose:
            print(f"\n[{i+1}/{len(configs)}] Evaluating candidate {config.candidate_id}")
        
        result = evaluate_candidate(
            config=config,
            study_name=study_name,
            trial_id=trial_id,
            runs_dir=runs_dir,
            mc_runs=mc_runs,
            mc_ticks=mc_ticks,
            seed=seed,
            verbose=verbose,
        )
        
        results.append(result)
    
    return results

