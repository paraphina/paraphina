#!/usr/bin/env python3
"""
exp_phase_a_adversarial_search.py

Phase A-2: Adversarial / worst-case search + failure seed regression suite.

This script implements adversarial search to find failure cases faster than random
Monte Carlo. It samples scenario parameters in a bounded box, runs the monte_carlo
binary for each candidate, and scores based on adversarial objectives.

Goals:
- Maximize kill_rate_upper_ci (or kill_prob upper bound)
- Maximize drawdown_cvar  
- Minimize mean pnl (tertiary)

Outputs:
- runs/<out>/exp_phase_a_adversarial_search_runs.csv (per-run)
- runs/<out>/exp_phase_a_adversarial_search_summary.csv (summary)
- runs/<out>/failure_seeds.json (top K worst cases)
- scenarios/suites/adversarial_regression_v1.yaml (regression suite)

Usage:
    python3 batch_runs/exp_phase_a_adversarial_search.py --smoke --out runs/adv_smoke
    python3 batch_runs/exp_phase_a_adversarial_search.py --full --out runs/adv_full
    python3 batch_runs/exp_phase_a_adversarial_search.py --help

Follows AI_PLAYBOOK exp03 structure pattern.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ===========================================================================
# Paths & constants
# ===========================================================================

ROOT = Path(__file__).resolve().parents[1]
MONTE_CARLO_BIN = ROOT / "target" / "release" / "monte_carlo"
SIM_EVAL_BIN = ROOT / "target" / "release" / "sim_eval"

DEFAULT_OUT_DIR = ROOT / "runs" / "exp_phase_a_adversarial_search"
SUITES_DIR = ROOT / "scenarios" / "suites"

# Default number of top failure seeds to extract
DEFAULT_TOP_K = 10

# Adversarial parameter bounds
# These define the search space for adversarial scenario generation
ADVERSARIAL_PARAM_BOUNDS = {
    # Volatility multiplier: scales vol_ref
    "vol_multiplier": (0.5, 3.0),
    # Jump intensity: scales a shock parameter (simulated via init_q_tao stress)
    "jump_intensity": (0.0, 2.0),
    # Spread multiplier: affects market microstructure
    "spread_multiplier": (0.5, 4.0),
    # Latency multiplier: affects timing jitter
    "latency_multiplier": (0.5, 3.0),
    # Depth collapse factor: reduces available liquidity (simulated via mm_size_eta reduction)
    "depth_collapse_factor": (0.0, 0.9),
    # Venue outage probability: random venue disabling (simulated via reduced runs)
    "venue_outage_prob": (0.0, 0.5),
    # Starting inventory stress
    "init_q_tao": (-60.0, 60.0),
    # Loss limit stress
    "daily_loss_limit": (500.0, 5000.0),
}

# Base configs for adversarial testing
BASE_PROFILES = ["balanced", "conservative", "aggressive"]


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class AdversarialScenario:
    """Configuration for a single adversarial scenario run."""
    scenario_id: str
    seed: int
    profile: str
    vol_multiplier: float
    jump_intensity: float
    spread_multiplier: float
    latency_multiplier: float
    depth_collapse_factor: float
    venue_outage_prob: float
    init_q_tao: float
    daily_loss_limit: float
    mc_runs: int
    mc_ticks: int
    mc_seed: int


@dataclass
class AdversarialResult:
    """Result from running an adversarial scenario."""
    scenario_id: str
    seed: int
    profile: str
    # Scenario parameters
    vol_multiplier: float
    jump_intensity: float
    spread_multiplier: float
    latency_multiplier: float
    depth_collapse_factor: float
    venue_outage_prob: float
    init_q_tao: float
    daily_loss_limit: float
    # Metrics from mc_summary.json
    mean_pnl: float
    pnl_var: float
    pnl_cvar: float
    drawdown_var: float
    drawdown_cvar: float
    kill_prob_point: float
    kill_prob_ci_lower: float
    kill_prob_ci_upper: float
    kill_count: int
    total_runs: int
    max_drawdown_p99: float
    pnl_p01: float
    # Computed adversarial score
    adversarial_score: float
    # Verification status
    evidence_pack_verified: bool
    # Runtime info
    output_dir: str
    duration_sec: float
    returncode: int


# ===========================================================================
# Parsing mc_summary.json
# ===========================================================================

def parse_mc_summary(summary_path: Path) -> Dict[str, Any]:
    """
    Parse mc_summary.json tail metrics into a stable dict.
    
    This is the canonical parser used by the adversarial harness.
    Designed for unit testing.
    
    Returns a flat dict with all relevant metrics.
    Returns empty dict on parse error.
    """
    if not summary_path.exists():
        return {}
    
    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}
    
    # Extract metrics from known structure
    tail_risk = summary.get("tail_risk", {})
    aggregate = summary.get("aggregate", {})
    
    pnl_var_cvar = tail_risk.get("pnl_var_cvar", {})
    dd_var_cvar = tail_risk.get("max_drawdown_var_cvar", {})
    kill_prob = tail_risk.get("kill_probability", {})
    pnl_quantiles = tail_risk.get("pnl_quantiles", {})
    dd_quantiles = tail_risk.get("max_drawdown_quantiles", {})
    pnl_agg = aggregate.get("pnl", {})
    
    return {
        "schema_version": tail_risk.get("schema_version", 0),
        "mean_pnl": pnl_agg.get("mean", float("nan")),
        "pnl_std": pnl_agg.get("std_pop", float("nan")),
        "pnl_var": pnl_var_cvar.get("var", float("nan")),
        "pnl_cvar": pnl_var_cvar.get("cvar", float("nan")),
        "pnl_p01": pnl_quantiles.get("p01", float("nan")),
        "pnl_p05": pnl_quantiles.get("p05", float("nan")),
        "pnl_p50": pnl_quantiles.get("p50", float("nan")),
        "pnl_p95": pnl_quantiles.get("p95", float("nan")),
        "pnl_p99": pnl_quantiles.get("p99", float("nan")),
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


# ===========================================================================
# Adversarial scoring
# ===========================================================================

def compute_adversarial_score(metrics: Dict[str, Any]) -> float:
    """
    Compute adversarial score from metrics.
    
    Higher score = worse/more adversarial scenario.
    
    Objectives (in priority order):
    1. Primary: maximize kill_rate_upper_ci (or kill_prob upper bound)
    2. Secondary: maximize drawdown_cvar
    3. Tertiary: minimize mean pnl
    
    Score formula:
        score = 1000 * kill_prob_ci_upper + 10 * drawdown_cvar - 0.1 * mean_pnl
    
    This ensures kill probability dominates, followed by drawdown, then PnL.
    
    Designed for deterministic ranking given fixed inputs.
    """
    kill_prob_ci_upper = metrics.get("kill_prob_ci_upper", 0.0)
    drawdown_cvar = metrics.get("drawdown_cvar", 0.0)
    mean_pnl = metrics.get("mean_pnl", 0.0)
    
    # Handle NaN values - treat as neutral
    if math.isnan(kill_prob_ci_upper):
        kill_prob_ci_upper = 0.0
    if math.isnan(drawdown_cvar):
        drawdown_cvar = 0.0
    if math.isnan(mean_pnl):
        mean_pnl = 0.0
    
    # Score formula with clear priority ordering
    # Primary: kill_prob_ci_upper (weight 1000)
    # Secondary: drawdown_cvar (weight 10)
    # Tertiary: -mean_pnl (weight 0.1, negative because we minimize PnL)
    score = 1000.0 * kill_prob_ci_upper + 10.0 * drawdown_cvar - 0.1 * mean_pnl
    
    return score


def rank_candidates_deterministic(
    results: List[AdversarialResult],
) -> List[AdversarialResult]:
    """
    Rank candidates deterministically by adversarial score (descending).
    
    Ties are broken by:
    1. scenario_id (alphabetical)
    2. seed (ascending)
    
    This ensures identical rankings given the same inputs.
    """
    return sorted(
        results,
        key=lambda r: (-r.adversarial_score, r.scenario_id, r.seed),
    )


# ===========================================================================
# Scenario generation
# ===========================================================================

def generate_adversarial_scenarios(
    n_scenarios: int,
    seed: int,
    profiles: List[str],
    mc_runs: int,
    mc_ticks: int,
) -> List[AdversarialScenario]:
    """
    Generate adversarial scenarios by sampling from bounded parameter space.
    
    Uses seeded RNG for determinism.
    """
    rng = random.Random(seed)
    scenarios = []
    
    for i in range(n_scenarios):
        profile = rng.choice(profiles)
        
        # Sample each adversarial parameter uniformly from its bounds
        vol_mult = rng.uniform(*ADVERSARIAL_PARAM_BOUNDS["vol_multiplier"])
        jump_int = rng.uniform(*ADVERSARIAL_PARAM_BOUNDS["jump_intensity"])
        spread_mult = rng.uniform(*ADVERSARIAL_PARAM_BOUNDS["spread_multiplier"])
        latency_mult = rng.uniform(*ADVERSARIAL_PARAM_BOUNDS["latency_multiplier"])
        depth_collapse = rng.uniform(*ADVERSARIAL_PARAM_BOUNDS["depth_collapse_factor"])
        venue_outage = rng.uniform(*ADVERSARIAL_PARAM_BOUNDS["venue_outage_prob"])
        init_q = rng.uniform(*ADVERSARIAL_PARAM_BOUNDS["init_q_tao"])
        loss_limit = rng.uniform(*ADVERSARIAL_PARAM_BOUNDS["daily_loss_limit"])
        
        # Create scenario ID from parameters
        scenario_seed = seed + i
        scenario_id = f"adv_{i:04d}_s{scenario_seed}"
        
        scenarios.append(AdversarialScenario(
            scenario_id=scenario_id,
            seed=scenario_seed,
            profile=profile,
            vol_multiplier=vol_mult,
            jump_intensity=jump_int,
            spread_multiplier=spread_mult,
            latency_multiplier=latency_mult,
            depth_collapse_factor=depth_collapse,
            venue_outage_prob=venue_outage,
            init_q_tao=init_q,
            daily_loss_limit=loss_limit,
            mc_runs=mc_runs,
            mc_ticks=mc_ticks,
            mc_seed=scenario_seed,
        ))
    
    return scenarios


def generate_smoke_scenarios(
    profiles: List[str],
    mc_runs: int,
    mc_ticks: int,
) -> List[AdversarialScenario]:
    """
    Generate a minimal smoke test scenario set.
    
    Fixed, deterministic scenarios for quick testing.
    """
    scenarios = []
    base_seed = 42
    
    # One stress scenario per profile
    for i, profile in enumerate(profiles):
        scenarios.append(AdversarialScenario(
            scenario_id=f"smoke_{profile}_{i}",
            seed=base_seed + i,
            profile=profile,
            vol_multiplier=2.0,  # High volatility
            jump_intensity=1.0,
            spread_multiplier=2.0,
            latency_multiplier=1.5,
            depth_collapse_factor=0.5,
            venue_outage_prob=0.0,
            init_q_tao=float(i - 1) * 30.0,  # -30, 0, 30
            daily_loss_limit=1000.0,
            mc_runs=mc_runs,
            mc_ticks=mc_ticks,
            mc_seed=base_seed + i,
        ))
    
    return scenarios


# ===========================================================================
# Monte Carlo runner
# ===========================================================================

def run_adversarial_scenario(
    scenario: AdversarialScenario,
    output_dir: Path,
    verbose: bool = True,
) -> AdversarialResult:
    """
    Run monte_carlo for a single adversarial scenario and return results.
    """
    import time
    
    scenario_output = output_dir / f"scenario_{scenario.scenario_id}"
    scenario_output.mkdir(parents=True, exist_ok=True)
    
    # Compute derived env vars from adversarial parameters
    base_vol_ref = 0.10
    base_mm_size_eta = 1.0
    
    effective_vol_ref = base_vol_ref * scenario.vol_multiplier
    effective_mm_size_eta = base_mm_size_eta * (1.0 - scenario.depth_collapse_factor)
    effective_jitter_ms = int(100 * scenario.latency_multiplier)
    
    # Build command
    cmd = [
        str(MONTE_CARLO_BIN),
        "--runs", str(scenario.mc_runs),
        "--ticks", str(scenario.mc_ticks),
        "--seed", str(scenario.mc_seed),
        "--jitter-ms", str(effective_jitter_ms),
        "--output-dir", str(scenario_output),
        "--quiet",
    ]
    
    # Environment overlay
    env = os.environ.copy()
    env["PARAPHINA_RISK_PROFILE"] = scenario.profile
    env["PARAPHINA_VOL_REF"] = str(effective_vol_ref)
    env["PARAPHINA_MM_SIZE_ETA"] = str(effective_mm_size_eta)
    env["PARAPHINA_INIT_Q_TAO"] = str(scenario.init_q_tao)
    env["PARAPHINA_DAILY_LOSS_LIMIT"] = str(scenario.daily_loss_limit)
    
    if verbose:
        print(f"  Running scenario {scenario.scenario_id} ({scenario.profile})...")
    
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    duration = time.time() - t0
    
    # Parse mc_summary.json
    summary_path = scenario_output / "mc_summary.json"
    metrics = parse_mc_summary(summary_path)
    
    if not metrics:
        if verbose:
            print(f"    ERROR: Failed to parse mc_summary.json for {scenario.scenario_id}")
        return AdversarialResult(
            scenario_id=scenario.scenario_id,
            seed=scenario.seed,
            profile=scenario.profile,
            vol_multiplier=scenario.vol_multiplier,
            jump_intensity=scenario.jump_intensity,
            spread_multiplier=scenario.spread_multiplier,
            latency_multiplier=scenario.latency_multiplier,
            depth_collapse_factor=scenario.depth_collapse_factor,
            venue_outage_prob=scenario.venue_outage_prob,
            init_q_tao=scenario.init_q_tao,
            daily_loss_limit=scenario.daily_loss_limit,
            mean_pnl=float("nan"),
            pnl_var=float("nan"),
            pnl_cvar=float("nan"),
            drawdown_var=float("nan"),
            drawdown_cvar=float("nan"),
            kill_prob_point=float("nan"),
            kill_prob_ci_lower=float("nan"),
            kill_prob_ci_upper=float("nan"),
            kill_count=0,
            total_runs=0,
            max_drawdown_p99=float("nan"),
            pnl_p01=float("nan"),
            adversarial_score=0.0,
            evidence_pack_verified=False,
            output_dir=str(scenario_output),
            duration_sec=duration,
            returncode=proc.returncode,
        )
    
    # Compute adversarial score
    adv_score = compute_adversarial_score(metrics)
    
    # Verify evidence pack
    evidence_verified = verify_evidence_pack(scenario_output, verbose=verbose)
    
    return AdversarialResult(
        scenario_id=scenario.scenario_id,
        seed=scenario.seed,
        profile=scenario.profile,
        vol_multiplier=scenario.vol_multiplier,
        jump_intensity=scenario.jump_intensity,
        spread_multiplier=scenario.spread_multiplier,
        latency_multiplier=scenario.latency_multiplier,
        depth_collapse_factor=scenario.depth_collapse_factor,
        venue_outage_prob=scenario.venue_outage_prob,
        init_q_tao=scenario.init_q_tao,
        daily_loss_limit=scenario.daily_loss_limit,
        mean_pnl=metrics.get("mean_pnl", float("nan")),
        pnl_var=metrics.get("pnl_var", float("nan")),
        pnl_cvar=metrics.get("pnl_cvar", float("nan")),
        drawdown_var=metrics.get("drawdown_var", float("nan")),
        drawdown_cvar=metrics.get("drawdown_cvar", float("nan")),
        kill_prob_point=metrics.get("kill_prob_point", float("nan")),
        kill_prob_ci_lower=metrics.get("kill_prob_ci_lower", float("nan")),
        kill_prob_ci_upper=metrics.get("kill_prob_ci_upper", float("nan")),
        kill_count=metrics.get("kill_count", 0),
        total_runs=metrics.get("total_runs", 0),
        max_drawdown_p99=metrics.get("max_drawdown_p99", float("nan")),
        pnl_p01=metrics.get("pnl_p01", float("nan")),
        adversarial_score=adv_score,
        evidence_pack_verified=evidence_verified,
        output_dir=str(scenario_output),
        duration_sec=duration,
        returncode=proc.returncode,
    )


def verify_evidence_pack(output_dir: Path, verbose: bool = True) -> bool:
    """Verify evidence pack using sim_eval CLI."""
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
        return True
    else:
        if verbose:
            print(f"    Evidence pack FAILED: {output_dir.name} (rc={proc.returncode})")
        return False


# ===========================================================================
# Output writing
# ===========================================================================

def result_to_row(result: AdversarialResult) -> Dict[str, Any]:
    """Convert AdversarialResult to a flat dict for CSV."""
    return {
        "scenario_id": result.scenario_id,
        "seed": result.seed,
        "profile": result.profile,
        "vol_multiplier": result.vol_multiplier,
        "jump_intensity": result.jump_intensity,
        "spread_multiplier": result.spread_multiplier,
        "latency_multiplier": result.latency_multiplier,
        "depth_collapse_factor": result.depth_collapse_factor,
        "venue_outage_prob": result.venue_outage_prob,
        "init_q_tao": result.init_q_tao,
        "daily_loss_limit": result.daily_loss_limit,
        "mean_pnl": result.mean_pnl,
        "pnl_var": result.pnl_var,
        "pnl_cvar": result.pnl_cvar,
        "drawdown_var": result.drawdown_var,
        "drawdown_cvar": result.drawdown_cvar,
        "kill_prob_point": result.kill_prob_point,
        "kill_prob_ci_lower": result.kill_prob_ci_lower,
        "kill_prob_ci_upper": result.kill_prob_ci_upper,
        "kill_count": result.kill_count,
        "total_runs": result.total_runs,
        "max_drawdown_p99": result.max_drawdown_p99,
        "pnl_p01": result.pnl_p01,
        "adversarial_score": result.adversarial_score,
        "evidence_pack_verified": result.evidence_pack_verified,
        "output_dir": result.output_dir,
        "duration_sec": result.duration_sec,
        "returncode": result.returncode,
    }


def write_runs_csv(results: List[AdversarialResult], path: Path) -> None:
    """Write per-scenario results to CSV using stdlib csv module."""
    if not results:
        path.write_text("")
        return
    
    rows = [result_to_row(r) for r in results]
    fieldnames = list(rows[0].keys())  # Preserve order
    
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(
    results: List[AdversarialResult],
    top_k: List[AdversarialResult],
    path: Path,
) -> None:
    """Write summary CSV with top failure seeds."""
    rows = []
    
    for r in top_k:
        row = result_to_row(r)
        row["rank"] = top_k.index(r) + 1
        row["category"] = "top_failure"
        rows.append(row)
    
    if not rows:
        path.write_text("")
        return
    
    # Ensure rank and category are first
    fieldnames = ["rank", "category"] + [k for k in rows[0].keys() if k not in ["rank", "category"]]
    
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_failure_seeds_json(
    top_k: List[AdversarialResult],
    path: Path,
) -> None:
    """Write top K failure seeds to JSON."""
    seeds = []
    for r in top_k:
        seeds.append({
            "rank": top_k.index(r) + 1,
            "scenario_id": r.scenario_id,
            "seed": r.seed,
            "profile": r.profile,
            "adversarial_score": r.adversarial_score,
            "kill_prob_ci_upper": r.kill_prob_ci_upper,
            "drawdown_cvar": r.drawdown_cvar,
            "mean_pnl": r.mean_pnl,
            "scenario_params": {
                "vol_multiplier": r.vol_multiplier,
                "jump_intensity": r.jump_intensity,
                "spread_multiplier": r.spread_multiplier,
                "latency_multiplier": r.latency_multiplier,
                "depth_collapse_factor": r.depth_collapse_factor,
                "venue_outage_prob": r.venue_outage_prob,
                "init_q_tao": r.init_q_tao,
                "daily_loss_limit": r.daily_loss_limit,
            },
        })
    
    with open(path, "w") as f:
        json.dump({
            "schema_version": 1,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "failure_seeds": seeds,
        }, f, indent=2)


def generate_regression_suite_yaml(
    top_k: List[AdversarialResult],
    output_path: Path,
) -> None:
    """
    Generate adversarial_regression_v1.yaml from discovered failure seeds.
    """
    lines = [
        "# Adversarial Regression Suite v1",
        "#",
        "# This suite was auto-generated by exp_phase_a_adversarial_search.py",
        f"# Generated at: {datetime.utcnow().isoformat()}Z",
        "#",
        "# Contains top failure cases discovered by adversarial search.",
        "# These scenarios stress-test edge cases and should be run as regression.",
        "#",
        "# Usage:",
        "#   cargo run --release -p paraphina --bin sim_eval -- suite scenarios/suites/adversarial_regression_v1.yaml",
        "#",
        "",
        "suite_id: adversarial_regression_v1",
        "suite_version: 1",
        "",
        "# Run each scenario twice for determinism verification",
        "repeat_runs: 2",
        "",
        "# Output directory for CI runs",
        "out_dir: runs/adversarial_regression",
        "",
        "# Failure scenarios discovered by adversarial search",
        "scenarios:",
    ]
    
    for i, r in enumerate(top_k):
        lines.extend([
            f"  # Rank {i+1}: score={r.adversarial_score:.2f}, kill_ci_upper={r.kill_prob_ci_upper:.3f}",
            f"  - id: {r.scenario_id}",
            f"    seed: {r.seed}",
            f"    profile: {r.profile}",
            f"    env_overrides:",
            f"      PARAPHINA_RISK_PROFILE: {r.profile}",
            f"      PARAPHINA_VOL_REF: \"{r.vol_multiplier * 0.10:.4f}\"",
            f"      PARAPHINA_MM_SIZE_ETA: \"{1.0 - r.depth_collapse_factor:.4f}\"",
            f"      PARAPHINA_INIT_Q_TAO: \"{r.init_q_tao:.2f}\"",
            f"      PARAPHINA_DAILY_LOSS_LIMIT: \"{r.daily_loss_limit:.2f}\"",
            f"    adversarial_params:",
            f"      vol_multiplier: {r.vol_multiplier:.4f}",
            f"      jump_intensity: {r.jump_intensity:.4f}",
            f"      spread_multiplier: {r.spread_multiplier:.4f}",
            f"      latency_multiplier: {r.latency_multiplier:.4f}",
            f"      depth_collapse_factor: {r.depth_collapse_factor:.4f}",
            f"      venue_outage_prob: {r.venue_outage_prob:.4f}",
            "",
        ])
    
    lines.extend([
        "# Invariants to check after each scenario",
        "invariants:",
        "  # Evidence packs must verify",
        "  evidence_pack_valid: true",
        "  # Allow kill_switch in adversarial scenarios (we're stress-testing)",
        "  expect_kill_switch: allowed",
    ])
    
    output_path.write_text("\n".join(lines))


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase A-2: Adversarial / worst-case search for failure seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick test mode (few scenarios, fast runs)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full search mode (many scenarios, thorough exploration)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: runs/exp_phase_a_adversarial_search)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top failure seeds to extract (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for scenario generation (default: 42)",
    )
    parser.add_argument(
        "--no-suite",
        action="store_true",
        help="Skip generating regression suite YAML",
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.smoke:
        mode = "smoke"
        n_scenarios = 0  # Use fixed smoke scenarios
        mc_runs = 10
        mc_ticks = 50
    elif args.full:
        mode = "full"
        n_scenarios = 50
        mc_runs = 50
        mc_ticks = 300
    else:
        # Default: moderate search
        mode = "default"
        n_scenarios = 20
        mc_runs = 30
        mc_ticks = 150
    
    print(f"[adversarial] Mode: {mode}")
    print(f"[adversarial] Base seed: {args.seed}")
    print(f"[adversarial] Top K: {args.top_k}")
    
    # Check binaries exist
    if not MONTE_CARLO_BIN.exists():
        print(f"ERROR: monte_carlo binary not found at {MONTE_CARLO_BIN}")
        print("Run: cargo build --release -p paraphina --bin monte_carlo")
        sys.exit(1)
    
    if not SIM_EVAL_BIN.exists():
        print(f"ERROR: sim_eval binary not found at {SIM_EVAL_BIN}")
        print("Run: cargo build --release -p paraphina --bin sim_eval")
        sys.exit(1)
    
    # Create output directory
    base_dir = args.out if args.out else DEFAULT_OUT_DIR
    exp_dir = base_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mode}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[adversarial] Output directory: {exp_dir}")
    
    # Generate scenarios
    print(f"\n[adversarial] Generating scenarios...")
    if args.smoke:
        scenarios = generate_smoke_scenarios(BASE_PROFILES, mc_runs, mc_ticks)
    else:
        scenarios = generate_adversarial_scenarios(
            n_scenarios=n_scenarios,
            seed=args.seed,
            profiles=BASE_PROFILES,
            mc_runs=mc_runs,
            mc_ticks=mc_ticks,
        )
    print(f"[adversarial] Generated {len(scenarios)} scenarios")
    
    # Run all scenarios
    print(f"\n[adversarial] Running {len(scenarios)} adversarial scenarios...")
    results: List[AdversarialResult] = []
    for i, scenario in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}] Scenario {scenario.scenario_id}")
        result = run_adversarial_scenario(scenario, exp_dir, verbose=True)
        results.append(result)
    
    # Filter and rank results
    print(f"\n[adversarial] Ranking results by adversarial score...")
    valid_results = [r for r in results if r.evidence_pack_verified]
    ranked = rank_candidates_deterministic(valid_results)
    
    print(f"  Valid results: {len(valid_results)}/{len(results)}")
    if ranked:
        print(f"  Top adversarial score: {ranked[0].adversarial_score:.2f}")
    
    # Extract top K failure seeds
    top_k = ranked[:args.top_k]
    print(f"\n[adversarial] Top {len(top_k)} failure seeds:")
    for i, r in enumerate(top_k):
        print(f"  {i+1}. {r.scenario_id}: score={r.adversarial_score:.2f}, "
              f"kill_ci={r.kill_prob_ci_upper:.3f}, dd_cvar={r.drawdown_cvar:.2f}")
    
    # Write outputs
    print(f"\n[adversarial] Writing output files...")
    
    runs_csv = exp_dir / "exp_phase_a_adversarial_search_runs.csv"
    write_runs_csv(results, runs_csv)
    print(f"  Wrote: {runs_csv}")
    
    summary_csv = exp_dir / "exp_phase_a_adversarial_search_summary.csv"
    write_summary_csv(results, top_k, summary_csv)
    print(f"  Wrote: {summary_csv}")
    
    failure_seeds_json = exp_dir / "failure_seeds.json"
    write_failure_seeds_json(top_k, failure_seeds_json)
    print(f"  Wrote: {failure_seeds_json}")
    
    # Generate regression suite YAML
    if not args.no_suite and top_k:
        SUITES_DIR.mkdir(parents=True, exist_ok=True)
        suite_path = SUITES_DIR / "adversarial_regression_v1.yaml"
        generate_regression_suite_yaml(top_k, suite_path)
        print(f"  Wrote: {suite_path}")
    
    # Create latest symlink
    if args.out is None:
        latest_link = DEFAULT_OUT_DIR / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(exp_dir.name)
        print(f"  Symlink: {latest_link} -> {exp_dir.name}")
    
    print(f"\n[adversarial] Done!")
    print(f"\nResults in: {exp_dir}")
    
    # Summary stats
    if valid_results:
        avg_score = sum(r.adversarial_score for r in valid_results) / len(valid_results)
        max_kill = max(r.kill_prob_ci_upper for r in valid_results if not math.isnan(r.kill_prob_ci_upper))
        print(f"\nSummary:")
        print(f"  Scenarios run: {len(results)}")
        print(f"  Valid (verified): {len(valid_results)}")
        print(f"  Avg adversarial score: {avg_score:.2f}")
        print(f"  Max kill_prob_ci_upper: {max_kill:.3f}")
    
    # Exit with error if no valid results
    if not valid_results:
        print("\nWARNING: No scenarios passed evidence pack verification!")
        sys.exit(1)


if __name__ == "__main__":
    main()

