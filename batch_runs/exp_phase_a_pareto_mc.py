#!/usr/bin/env python3
"""
exp_phase_a_pareto_mc.py

Phase A "A1" vertical slice: Pareto tuning harness for Monte Carlo runs.

This script:
1. Builds a deterministic sweep of strategy knobs via env overlays (small grid or seeded random).
2. For each candidate:
   - Runs monte_carlo into an isolated runs/ directory
   - Verifies evidence packs using sim_eval verify-evidence-pack
   - Reads mc_summary.json to extract tail risk metrics
3. Writes:
   - runs CSV: one row per candidate with all metrics
   - summary CSV: Pareto frontier and "best candidate under budgets"
4. Outputs promoted_config.env + promotion_record.json for winning candidates.

Risk tier budgets:
- kill_prob_ci_upper <= budget (Wilson 95% CI upper bound)
- drawdown_cvar <= budget
- mean_pnl >= budget

Usage:
    python3 batch_runs/exp_phase_a_pareto_mc.py [--smoke] [--full] [--out DIR] [--help]

    --smoke   Quick test mode (fewer candidates, fewer MC runs)
    --full    Full sweep mode (more candidates, more MC runs)
    --out     Output directory (default: runs/exp_phase_a_pareto_mc)
    --help    Show this help

Follows AI_PLAYBOOK exp03 template pattern.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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

EXP_DIR = ROOT / "runs" / "exp_phase_a_pareto_mc"

# Risk tier budgets (configurable)
# These define the acceptance criteria for a candidate to be "promotable"
DEFAULT_BUDGETS = {
    "conservative": {
        "kill_prob_ci_upper": 0.05,   # Max 5% kill probability (95% CI upper)
        "drawdown_cvar": 500.0,       # Max $500 drawdown CVaR
        "min_mean_pnl": 10.0,         # Min $10 mean PnL
    },
    "balanced": {
        "kill_prob_ci_upper": 0.10,   # Max 10% kill probability
        "drawdown_cvar": 1000.0,      # Max $1000 drawdown CVaR
        "min_mean_pnl": 20.0,         # Min $20 mean PnL
    },
    "aggressive": {
        "kill_prob_ci_upper": 0.15,   # Max 15% kill probability
        "drawdown_cvar": 2000.0,      # Max $2000 drawdown CVaR
        "min_mean_pnl": 30.0,         # Min $30 mean PnL
    },
}

# Knob ranges for grid/random sweep
KNOB_RANGES = {
    "PARAPHINA_HEDGE_BAND_BASE": [0.02, 0.05, 0.075, 0.10, 0.15],
    "PARAPHINA_MM_SIZE_ETA": [0.5, 1.0, 1.5, 2.0],
    "PARAPHINA_VOL_REF": [0.05, 0.10, 0.15],
    "PARAPHINA_DAILY_LOSS_LIMIT": [500.0, 1000.0, 2000.0],
}

# Default profiles to test
PROFILES = ["balanced", "conservative", "aggressive"]


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class CandidateConfig:
    """Configuration for a single candidate run."""
    candidate_id: str
    profile: str
    env_overlay: Dict[str, str]
    mc_runs: int
    mc_ticks: int
    mc_seed: int


@dataclass
class CandidateResult:
    """Result from running a candidate."""
    candidate_id: str
    profile: str
    env_overlay: Dict[str, str]
    output_dir: Path
    # From mc_summary.json
    mean_pnl: float
    pnl_var: float
    pnl_cvar: float
    drawdown_var: float
    drawdown_cvar: float
    kill_prob_point: float
    kill_prob_ci_lower: float
    kill_prob_ci_upper: float
    total_runs: int
    # Verification status
    evidence_pack_verified: bool
    # Runtime info
    duration_sec: float
    returncode: int


# ===========================================================================
# Knob sweep generation
# ===========================================================================

def generate_grid_sweep(profiles: List[str]) -> List[Dict[str, Any]]:
    """
    Generate a deterministic grid sweep over knob ranges.
    Returns list of (profile, env_overlay) pairs.
    """
    sweep = []
    for profile in profiles:
        for band_base in KNOB_RANGES["PARAPHINA_HEDGE_BAND_BASE"]:
            for mm_eta in KNOB_RANGES["PARAPHINA_MM_SIZE_ETA"]:
                for vol_ref in KNOB_RANGES["PARAPHINA_VOL_REF"]:
                    for loss_limit in KNOB_RANGES["PARAPHINA_DAILY_LOSS_LIMIT"]:
                        env = {
                            "PARAPHINA_RISK_PROFILE": profile,
                            "PARAPHINA_HEDGE_BAND_BASE": str(band_base),
                            "PARAPHINA_MM_SIZE_ETA": str(mm_eta),
                            "PARAPHINA_VOL_REF": str(vol_ref),
                            "PARAPHINA_DAILY_LOSS_LIMIT": str(loss_limit),
                        }
                        sweep.append({"profile": profile, "env": env})
    return sweep


def generate_smoke_sweep(profiles: List[str]) -> List[Dict[str, Any]]:
    """
    Generate a minimal smoke test sweep (just a few configs per profile).
    """
    sweep = []
    for profile in profiles:
        # Just one config per profile for smoke test
        env = {
            "PARAPHINA_RISK_PROFILE": profile,
            "PARAPHINA_HEDGE_BAND_BASE": "0.05",
            "PARAPHINA_MM_SIZE_ETA": "1.0",
            "PARAPHINA_VOL_REF": "0.10",
            "PARAPHINA_DAILY_LOSS_LIMIT": "1000.0",
        }
        sweep.append({"profile": profile, "env": env})
    return sweep


def generate_seeded_random_sweep(
    profiles: List[str],
    n_samples: int = 20,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate a seeded pseudo-random sweep over knob ranges.
    Deterministic given seed.
    """
    import random
    rng = random.Random(seed)

    sweep = []
    for _ in range(n_samples):
        profile = rng.choice(profiles)
        band_base = rng.choice(KNOB_RANGES["PARAPHINA_HEDGE_BAND_BASE"])
        mm_eta = rng.choice(KNOB_RANGES["PARAPHINA_MM_SIZE_ETA"])
        vol_ref = rng.choice(KNOB_RANGES["PARAPHINA_VOL_REF"])
        loss_limit = rng.choice(KNOB_RANGES["PARAPHINA_DAILY_LOSS_LIMIT"])

        env = {
            "PARAPHINA_RISK_PROFILE": profile,
            "PARAPHINA_HEDGE_BAND_BASE": str(band_base),
            "PARAPHINA_MM_SIZE_ETA": str(mm_eta),
            "PARAPHINA_VOL_REF": str(vol_ref),
            "PARAPHINA_DAILY_LOSS_LIMIT": str(loss_limit),
        }
        sweep.append({"profile": profile, "env": env})

    return sweep


def config_to_id(env: Dict[str, str]) -> str:
    """Generate a deterministic ID from env config."""
    # Sort keys for determinism
    sorted_items = sorted(env.items())
    key_str = "_".join(f"{k}={v}" for k, v in sorted_items)
    return hashlib.sha256(key_str.encode()).hexdigest()[:12]


# ===========================================================================
# Monte Carlo runner
# ===========================================================================

def run_monte_carlo(
    config: CandidateConfig,
    exp_dir: Path,
    verbose: bool = True,
) -> CandidateResult:
    """
    Run monte_carlo for a single candidate and return results.
    """
    import time

    output_dir = exp_dir / f"candidate_{config.candidate_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        str(MONTE_CARLO_BIN),
        "--runs", str(config.mc_runs),
        "--ticks", str(config.mc_ticks),
        "--seed", str(config.mc_seed),
        "--output-dir", str(output_dir),
        "--quiet",
    ]

    # Merge env overlay
    env = os.environ.copy()
    env.update(config.env_overlay)

    if verbose:
        print(f"  Running candidate {config.candidate_id} ({config.profile})...")

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
    summary_path = output_dir / "mc_summary.json"
    if not summary_path.exists():
        if verbose:
            print(f"    ERROR: mc_summary.json not found for {config.candidate_id}")
        return CandidateResult(
            candidate_id=config.candidate_id,
            profile=config.profile,
            env_overlay=config.env_overlay,
            output_dir=output_dir,
            mean_pnl=float("nan"),
            pnl_var=float("nan"),
            pnl_cvar=float("nan"),
            drawdown_var=float("nan"),
            drawdown_cvar=float("nan"),
            kill_prob_point=float("nan"),
            kill_prob_ci_lower=float("nan"),
            kill_prob_ci_upper=float("nan"),
            total_runs=0,
            evidence_pack_verified=False,
            duration_sec=duration,
            returncode=proc.returncode,
        )

    with open(summary_path) as f:
        summary = json.load(f)

    # Extract metrics from tail_risk section
    tail_risk = summary.get("tail_risk", {})
    aggregate = summary.get("aggregate", {})

    pnl_var_cvar = tail_risk.get("pnl_var_cvar", {})
    dd_var_cvar = tail_risk.get("max_drawdown_var_cvar", {})
    kill_prob = tail_risk.get("kill_probability", {})
    pnl_agg = aggregate.get("pnl", {})

    # Verify evidence pack
    evidence_verified = verify_evidence_pack(output_dir, verbose=verbose)

    return CandidateResult(
        candidate_id=config.candidate_id,
        profile=config.profile,
        env_overlay=config.env_overlay,
        output_dir=output_dir,
        mean_pnl=pnl_agg.get("mean", float("nan")),
        pnl_var=pnl_var_cvar.get("var", float("nan")),
        pnl_cvar=pnl_var_cvar.get("cvar", float("nan")),
        drawdown_var=dd_var_cvar.get("var", float("nan")),
        drawdown_cvar=dd_var_cvar.get("cvar", float("nan")),
        kill_prob_point=kill_prob.get("point_estimate", float("nan")),
        kill_prob_ci_lower=kill_prob.get("ci_lower", float("nan")),
        kill_prob_ci_upper=kill_prob.get("ci_upper", float("nan")),
        total_runs=kill_prob.get("total_runs", 0),
        evidence_pack_verified=evidence_verified,
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
# Pareto frontier and budget selection
# ===========================================================================

def is_dominated(a: CandidateResult, b: CandidateResult) -> bool:
    """
    Check if candidate `a` is dominated by candidate `b`.
    Dominated means: b is at least as good on all metrics and strictly better on at least one.

    Objectives:
    - Maximize mean_pnl
    - Minimize drawdown_cvar
    - Minimize kill_prob_ci_upper
    """
    # b at least as good as a on all
    b_pnl_ge = b.mean_pnl >= a.mean_pnl
    b_dd_le = b.drawdown_cvar <= a.drawdown_cvar
    b_kill_le = b.kill_prob_ci_upper <= a.kill_prob_ci_upper

    if not (b_pnl_ge and b_dd_le and b_kill_le):
        return False

    # b strictly better on at least one
    b_pnl_gt = b.mean_pnl > a.mean_pnl
    b_dd_lt = b.drawdown_cvar < a.drawdown_cvar
    b_kill_lt = b.kill_prob_ci_upper < a.kill_prob_ci_upper

    return b_pnl_gt or b_dd_lt or b_kill_lt


def compute_pareto_frontier(results: List[CandidateResult]) -> List[CandidateResult]:
    """
    Compute the Pareto frontier from a list of candidate results.
    Returns candidates that are not dominated by any other.
    """
    # Filter out failed candidates
    valid = [r for r in results if r.evidence_pack_verified and not any(
        is_nan(getattr(r, attr)) for attr in ["mean_pnl", "drawdown_cvar", "kill_prob_ci_upper"]
    )]

    if not valid:
        return []

    frontier = []
    for candidate in valid:
        dominated = False
        for other in valid:
            if other.candidate_id != candidate.candidate_id:
                if is_dominated(candidate, other):
                    dominated = True
                    break
        if not dominated:
            frontier.append(candidate)

    return frontier


def is_nan(x: float) -> bool:
    """Check if value is NaN."""
    return math.isnan(x) if isinstance(x, float) else False


def meets_budget(result: CandidateResult, budget: Dict[str, float]) -> bool:
    """Check if a candidate meets the risk tier budget."""
    if is_nan(result.kill_prob_ci_upper) or is_nan(result.drawdown_cvar) or is_nan(result.mean_pnl):
        return False

    return (
        result.kill_prob_ci_upper <= budget["kill_prob_ci_upper"] and
        result.drawdown_cvar <= budget["drawdown_cvar"] and
        result.mean_pnl >= budget["min_mean_pnl"]
    )


def select_best_under_budget(
    results: List[CandidateResult],
    budget: Dict[str, float],
) -> Optional[CandidateResult]:
    """
    Select the best candidate that meets budget constraints.
    Best = highest mean_pnl among those meeting all constraints.
    """
    qualifying = [r for r in results if meets_budget(r, budget)]
    if not qualifying:
        return None

    # Sort by mean_pnl descending
    qualifying.sort(key=lambda r: r.mean_pnl, reverse=True)
    return qualifying[0]


# ===========================================================================
# Output writing (stdlib csv, no pandas)
# ===========================================================================

def result_to_row(result: CandidateResult) -> Dict[str, Any]:
    """Convert CandidateResult to a flat dict for CSV."""
    row = {
        "candidate_id": result.candidate_id,
        "profile": result.profile,
        "mean_pnl": result.mean_pnl,
        "pnl_var": result.pnl_var,
        "pnl_cvar": result.pnl_cvar,
        "drawdown_var": result.drawdown_var,
        "drawdown_cvar": result.drawdown_cvar,
        "kill_prob_point": result.kill_prob_point,
        "kill_prob_ci_lower": result.kill_prob_ci_lower,
        "kill_prob_ci_upper": result.kill_prob_ci_upper,
        "total_runs": result.total_runs,
        "evidence_pack_verified": result.evidence_pack_verified,
        "duration_sec": result.duration_sec,
        "returncode": result.returncode,
    }
    # Add env overlay keys
    for k, v in result.env_overlay.items():
        row[k] = v
    return row


def write_runs_csv(results: List[CandidateResult], path: Path) -> None:
    """Write per-candidate results to CSV using stdlib csv module."""
    if not results:
        # Write empty CSV with header
        path.write_text("")
        return

    rows = [result_to_row(r) for r in results]
    # Get all unique keys across all rows, sorted for determinism
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(
    results: List[CandidateResult],
    pareto: List[CandidateResult],
    best_by_tier: Dict[str, Optional[CandidateResult]],
    path: Path,
) -> None:
    """Write summary CSV with Pareto frontier and best-per-tier using stdlib csv."""
    rows = []

    # Add Pareto frontier members
    for r in pareto:
        row = result_to_row(r)
        row["category"] = "pareto_frontier"
        row["tier"] = ""
        rows.append(row)

    # Add best per tier
    for tier, result in best_by_tier.items():
        if result is not None:
            row = result_to_row(result)
            row["category"] = "best_under_budget"
            row["tier"] = tier
            rows.append(row)

    if not rows:
        # Write empty CSV
        path.write_text("")
        return

    # Get all unique keys, with category and tier first
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    # Ensure category and tier are first
    other_keys = sorted(k for k in all_keys if k not in ["category", "tier"])
    fieldnames = ["category", "tier"] + other_keys

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_promoted_config(
    result: CandidateResult,
    tier: str,
    exp_dir: Path,
) -> Tuple[Path, Path]:
    """
    Write promoted_config.env and promotion_record.json for a winning candidate.
    Returns (env_path, record_path).
    """
    env_path = exp_dir / f"promoted_config_{tier}.env"
    record_path = exp_dir / f"promotion_record_{tier}.json"

    # Write env file
    with open(env_path, "w") as f:
        f.write(f"# Promoted configuration for {tier} tier\n")
        f.write(f"# Generated by exp_phase_a_pareto_mc.py at {datetime.utcnow().isoformat()}Z\n")
        f.write(f"# Candidate ID: {result.candidate_id}\n")
        f.write("\n")
        for k, v in sorted(result.env_overlay.items()):
            f.write(f"export {k}={v}\n")

    # Compute hash of env file
    with open(env_path, "rb") as f:
        env_hash = hashlib.sha256(f.read()).hexdigest()

    # Write promotion record
    record = {
        "candidate_id": result.candidate_id,
        "tier": tier,
        "promoted_at": datetime.utcnow().isoformat() + "Z",
        "env_file": str(env_path.name),
        "env_file_sha256": env_hash,
        "metrics": {
            "mean_pnl": result.mean_pnl,
            "pnl_var": result.pnl_var,
            "pnl_cvar": result.pnl_cvar,
            "drawdown_var": result.drawdown_var,
            "drawdown_cvar": result.drawdown_cvar,
            "kill_prob_point": result.kill_prob_point,
            "kill_prob_ci_lower": result.kill_prob_ci_lower,
            "kill_prob_ci_upper": result.kill_prob_ci_upper,
            "total_runs": result.total_runs,
        },
        "env_overlay": result.env_overlay,
        "reproduce_command": (
            f"PARAPHINA_RISK_PROFILE={result.profile} "
            + " ".join(f"{k}={v}" for k, v in sorted(result.env_overlay.items()) if k != "PARAPHINA_RISK_PROFILE")
            + f" cargo run --release --bin monte_carlo -- --runs 100 --ticks 600 --seed 42"
        ),
    }

    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)

    return env_path, record_path


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase A Pareto tuning harness for Monte Carlo runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick test mode (fewer candidates, fewer MC runs)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full sweep mode (more candidates, more MC runs)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: runs/exp_phase_a_pareto_mc)",
    )

    args = parser.parse_args()

    # Determine mode
    if args.smoke:
        mode = "smoke"
        mc_runs = 10
        mc_ticks = 50
        sweep = generate_smoke_sweep(PROFILES)
    elif args.full:
        mode = "full"
        mc_runs = 100
        mc_ticks = 600
        sweep = generate_grid_sweep(PROFILES)
    else:
        # Default: small random sweep
        mode = "default"
        mc_runs = 50
        mc_ticks = 200
        sweep = generate_seeded_random_sweep(PROFILES, n_samples=10, seed=42)

    print(f"[phase_a] Mode: {mode}")
    print(f"[phase_a] MC runs per candidate: {mc_runs}")
    print(f"[phase_a] MC ticks per run: {mc_ticks}")
    print(f"[phase_a] Candidates to evaluate: {len(sweep)}")

    # Check binaries exist
    if not MONTE_CARLO_BIN.exists():
        print(f"ERROR: monte_carlo binary not found at {MONTE_CARLO_BIN}")
        print("Run: cargo build --release -p paraphina --bin monte_carlo")
        sys.exit(1)

    if not SIM_EVAL_BIN.exists():
        print(f"ERROR: sim_eval binary not found at {SIM_EVAL_BIN}")
        print("Run: cargo build --release -p paraphina --bin sim_eval")
        sys.exit(1)

    # Use --out if provided, otherwise default EXP_DIR
    base_dir = args.out if args.out else EXP_DIR

    # Create experiment directory
    exp_dir = base_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mode}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase_a] Output directory: {exp_dir}")

    # Build candidate configs
    configs: List[CandidateConfig] = []
    for item in sweep:
        cid = config_to_id(item["env"])
        configs.append(CandidateConfig(
            candidate_id=cid,
            profile=item["profile"],
            env_overlay=item["env"],
            mc_runs=mc_runs,
            mc_ticks=mc_ticks,
            mc_seed=42,  # Fixed seed for determinism
        ))

    # Run all candidates
    print(f"\n[phase_a] Running {len(configs)} candidates...")
    results: List[CandidateResult] = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Candidate {config.candidate_id}")
        result = run_monte_carlo(config, exp_dir, verbose=True)
        results.append(result)

    # Compute Pareto frontier
    print("\n[phase_a] Computing Pareto frontier...")
    pareto = compute_pareto_frontier(results)
    print(f"  Pareto frontier size: {len(pareto)}")
    for r in pareto:
        print(f"    - {r.candidate_id}: pnl={r.mean_pnl:.2f}, dd_cvar={r.drawdown_cvar:.2f}, kill_ci={r.kill_prob_ci_upper:.3f}")

    # Select best per tier
    print("\n[phase_a] Selecting best candidates per budget tier...")
    best_by_tier: Dict[str, Optional[CandidateResult]] = {}
    for tier, budget in DEFAULT_BUDGETS.items():
        # Filter by profile matching tier (or all if not matching)
        tier_results = [r for r in results if r.profile == tier]
        if not tier_results:
            tier_results = results  # Fall back to all

        best = select_best_under_budget(tier_results, budget)
        best_by_tier[tier] = best
        if best:
            print(f"  {tier}: {best.candidate_id} (pnl={best.mean_pnl:.2f}, meets budget)")
        else:
            print(f"  {tier}: No candidate meets budget")

    # Write outputs
    print("\n[phase_a] Writing output files...")

    runs_csv = exp_dir / "pareto_runs.csv"
    write_runs_csv(results, runs_csv)
    print(f"  Wrote: {runs_csv}")

    summary_csv = exp_dir / "pareto_summary.csv"
    write_summary_csv(results, pareto, best_by_tier, summary_csv)
    print(f"  Wrote: {summary_csv}")

    # Write promoted configs for winners
    for tier, result in best_by_tier.items():
        if result is not None:
            env_path, record_path = write_promoted_config(result, tier, exp_dir)
            print(f"  Wrote: {env_path.name}, {record_path.name}")

    # Also create a symlink for easy access (only if using default EXP_DIR)
    if args.out is None:
        latest_link = EXP_DIR / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(exp_dir.name)
        print(f"  Symlink: {latest_link} -> {exp_dir.name}")

    print("\n[phase_a] Done!")
    print(f"\nResults in: {exp_dir}")

    # Exit with error if no candidates passed verification
    verified = [r for r in results if r.evidence_pack_verified]
    if not verified:
        print("\nWARNING: No candidates passed evidence pack verification!")
        sys.exit(1)


if __name__ == "__main__":
    main()
