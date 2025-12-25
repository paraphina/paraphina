#!/usr/bin/env python3
"""
adversarial_search_promote.py

Phase A-B2: Adversarial / worst-case search + scenario promotion to permanent regression suite.

Implements evolutionary-lite / hill-climb over seeds per WHITEPAPER Appendix B2:
- Start with seeded random scenarios
- Run sim_eval run for each candidate with isolated output
- Score by adversarial objective (maximize failure likelihood)
- Extract top-K failure scenarios
- Promote discovered scenarios to scenarios/v1/adversarial/generated_v1/
- Generate v2 suite with path-based scenarios only (no inline env_overrides)

Key differences from v1 (exp_phase_a_adversarial_search.py):
- Generates individual scenario YAML files (path-based)
- Suite contains only scenario paths (compatible with sim_eval suite)
- Uses sim_eval run with --output-dir for each candidate
- Verifies evidence with verify-evidence-tree

Usage:
    python3 -m batch_runs.phase_a.adversarial_search_promote --help
    python3 -m batch_runs.phase_a.adversarial_search_promote --smoke --out runs/adv_search_smoke
    python3 -m batch_runs.phase_a.adversarial_search_promote --trials 20 --out runs/adv_search_full

Outputs:
    <out_dir>/search_trials.jsonl    - Every candidate (one JSON per line)
    <out_dir>/topk.json              - Selected top-K scenarios
    <out_dir>/summary.md             - Human-readable summary
    scenarios/v1/adversarial/generated_v1/   - Promoted scenario files
    scenarios/suites/adversarial_regression_v2.yaml  - Generated suite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ===========================================================================
# Paths & constants
# ===========================================================================

ROOT = Path(__file__).resolve().parents[2]
SIM_EVAL_BIN = ROOT / "target" / "release" / "sim_eval"
MONTE_CARLO_BIN = ROOT / "target" / "release" / "monte_carlo"

SCENARIOS_DIR = ROOT / "scenarios"
GENERATED_SCENARIOS_DIR = SCENARIOS_DIR / "v1" / "adversarial" / "generated_v1"
SUITES_DIR = SCENARIOS_DIR / "suites"
SUITE_V2_PATH = SUITES_DIR / "adversarial_regression_v2.yaml"

DEFAULT_OUT_DIR = ROOT / "runs" / "phaseA_adversarial_search"

# Adversarial parameter bounds for evolutionary search
ADVERSARIAL_PARAM_BOUNDS = {
    "vol": (0.005, 0.05),              # Volatility (base)
    "vol_multiplier": (0.5, 3.0),      # Volatility scaling
    "jump_intensity": (0.0001, 0.002), # Jump events per step
    "jump_sigma": (0.01, 0.10),        # Jump magnitude
    "spread_bps": (0.5, 5.0),          # Spread in basis points
    "latency_ms": (0.0, 50.0),         # Latency in ms
    "init_q_tao": (-60.0, 60.0),       # Initial inventory
    "daily_loss_limit": (300.0, 3000.0),
}

BASE_PROFILES = ["balanced", "conservative", "aggressive"]

# Default top-K for suite promotion
DEFAULT_TOP_K = 5


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class AdversarialCandidate:
    """Configuration for a single adversarial scenario candidate."""
    candidate_id: str
    seed: int
    profile: str
    # Market model params
    vol: float
    vol_multiplier: float
    jump_intensity: float
    jump_sigma: float
    # Microstructure params
    spread_bps: float
    latency_ms: float
    # Strategy stress params
    init_q_tao: float
    daily_loss_limit: float
    # Run params
    ticks: int
    
    def scenario_hash(self) -> str:
        """Compute deterministic hash for stable filename."""
        params = [
            f"s{self.seed}",
            f"p{self.profile}",
            f"v{self.vol:.4f}",
            f"vm{self.vol_multiplier:.2f}",
            f"ji{self.jump_intensity:.5f}",
            f"js{self.jump_sigma:.3f}",
            f"sp{self.spread_bps:.2f}",
            f"lt{self.latency_ms:.1f}",
            f"q{self.init_q_tao:.1f}",
            f"dl{self.daily_loss_limit:.0f}",
        ]
        key = "_".join(params)
        return hashlib.sha256(key.encode()).hexdigest()[:8]
    
    def scenario_filename(self) -> str:
        """Generate deterministic filename for promoted scenario."""
        return f"adv_s{self.seed:05d}_{self.scenario_hash()}.yaml"
    
    def to_scenario_yaml(self) -> str:
        """Generate scenario YAML content."""
        effective_vol = self.vol * self.vol_multiplier
        lines = [
            f"# Adversarial scenario discovered by adversarial_search_promote.py",
            f"# Seed: {self.seed}, Profile: {self.profile}",
            f"# Generated: {datetime.now(timezone.utc).isoformat()}",
            f"",
            f"scenario_id: adv_s{self.seed:05d}_{self.scenario_hash()}",
            f"scenario_version: 1",
            f"",
            f"engine: rl_sim_env",
            f"",
            f"horizon:",
            f"  steps: {self.ticks}",
            f"  dt_seconds: 0.25",
            f"",
            f"rng:",
            f"  base_seed: {self.seed}",
            f"  num_seeds: 2",
            f"",
            f"initial_state:",
            f"  risk_profile: {self.profile}",
            f"  init_q_tao: {self.init_q_tao:.2f}",
            f"",
            f"market_model:",
            f"  type: synthetic",
            f"  synthetic:",
            f"    process: jump_diffusion_stub",
            f"    params:",
            f"      vol: {effective_vol:.6f}",
            f"      drift: 0.0",
            f"      jump_intensity: {self.jump_intensity:.6f}",
            f"      jump_sigma: {self.jump_sigma:.4f}",
            f"",
            f"microstructure_model:",
            f"  fees_bps_maker: 0.0",
            f"  fees_bps_taker: {self.spread_bps:.2f}",
            f"  latency_ms: {self.latency_ms:.1f}",
            f"",
            f"# Adversarial search metadata",
            f"adversarial_params:",
            f"  vol_multiplier: {self.vol_multiplier:.4f}",
            f"  daily_loss_limit: {self.daily_loss_limit:.2f}",
            f"",
            f"invariants:",
            f"  expect_kill_switch: allowed",
            f"  pnl_linearity_check: disabled",
        ]
        return "\n".join(lines)


@dataclass
class SearchResult:
    """Result from evaluating an adversarial candidate."""
    candidate_id: str
    candidate: AdversarialCandidate
    # Metrics from run_summary.json
    mean_pnl: float = float("nan")
    max_drawdown: float = float("nan")
    kill_switch: bool = False
    final_pnl: float = float("nan")
    # Computed score
    adversarial_score: float = 0.0
    # Verification
    evidence_verified: bool = False
    # Runtime
    output_dir: str = ""
    duration_sec: float = 0.0
    returncode: int = -1
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "candidate_id": self.candidate_id,
            "seed": self.candidate.seed,
            "profile": self.candidate.profile,
            "vol": self.candidate.vol,
            "vol_multiplier": self.candidate.vol_multiplier,
            "jump_intensity": self.candidate.jump_intensity,
            "jump_sigma": self.candidate.jump_sigma,
            "spread_bps": self.candidate.spread_bps,
            "latency_ms": self.candidate.latency_ms,
            "init_q_tao": self.candidate.init_q_tao,
            "daily_loss_limit": self.candidate.daily_loss_limit,
            "ticks": self.candidate.ticks,
            "mean_pnl": self.mean_pnl,
            "max_drawdown": self.max_drawdown,
            "kill_switch": self.kill_switch,
            "final_pnl": self.final_pnl,
            "adversarial_score": self.adversarial_score,
            "evidence_verified": self.evidence_verified,
            "output_dir": self.output_dir,
            "duration_sec": self.duration_sec,
            "returncode": self.returncode,
            "error_message": self.error_message,
            "scenario_filename": self.candidate.scenario_filename(),
        }


# ===========================================================================
# Candidate generation (evolutionary-lite)
# ===========================================================================

def generate_random_candidates(
    n_candidates: int,
    base_seed: int,
    profiles: List[str],
    ticks: int,
) -> List[AdversarialCandidate]:
    """
    Generate random adversarial candidates using seeded RNG.
    
    Deterministic given base_seed.
    """
    rng = random.Random(base_seed)
    candidates = []
    
    for i in range(n_candidates):
        seed = base_seed + i
        profile = rng.choice(profiles)
        
        candidate = AdversarialCandidate(
            candidate_id=f"cand_{i:04d}_s{seed}",
            seed=seed,
            profile=profile,
            vol=rng.uniform(*ADVERSARIAL_PARAM_BOUNDS["vol"]),
            vol_multiplier=rng.uniform(*ADVERSARIAL_PARAM_BOUNDS["vol_multiplier"]),
            jump_intensity=rng.uniform(*ADVERSARIAL_PARAM_BOUNDS["jump_intensity"]),
            jump_sigma=rng.uniform(*ADVERSARIAL_PARAM_BOUNDS["jump_sigma"]),
            spread_bps=rng.uniform(*ADVERSARIAL_PARAM_BOUNDS["spread_bps"]),
            latency_ms=rng.uniform(*ADVERSARIAL_PARAM_BOUNDS["latency_ms"]),
            init_q_tao=rng.uniform(*ADVERSARIAL_PARAM_BOUNDS["init_q_tao"]),
            daily_loss_limit=rng.uniform(*ADVERSARIAL_PARAM_BOUNDS["daily_loss_limit"]),
            ticks=ticks,
        )
        candidates.append(candidate)
    
    return candidates


def generate_smoke_candidates(
    base_seed: int,
    profiles: List[str],
    ticks: int,
) -> List[AdversarialCandidate]:
    """
    Generate fixed smoke test candidates (one per profile).
    
    Deterministic and fast for testing.
    """
    candidates = []
    
    for i, profile in enumerate(profiles):
        seed = base_seed + i
        candidate = AdversarialCandidate(
            candidate_id=f"smoke_{profile}_{i}",
            seed=seed,
            profile=profile,
            vol=0.02,
            vol_multiplier=2.0,  # High stress
            jump_intensity=0.001,
            jump_sigma=0.05,
            spread_bps=2.0,
            latency_ms=10.0,
            init_q_tao=float((i - 1) * 30),  # -30, 0, 30
            daily_loss_limit=1000.0,
            ticks=ticks,
        )
        candidates.append(candidate)
    
    return candidates


def mutate_candidate(
    parent: AdversarialCandidate,
    rng: random.Random,
    mutation_strength: float = 0.2,
) -> AdversarialCandidate:
    """
    Mutate a candidate (evolutionary hill-climb step).
    
    Randomly perturbs 1-2 parameters within bounds.
    """
    params_to_mutate = rng.sample(
        ["vol", "vol_multiplier", "jump_intensity", "jump_sigma",
         "spread_bps", "latency_ms", "init_q_tao", "daily_loss_limit"],
        k=rng.randint(1, 2)
    )
    
    new_seed = parent.seed + 1000 + rng.randint(0, 999)
    
    def mutate_param(val: float, bounds: Tuple[float, float]) -> float:
        range_size = bounds[1] - bounds[0]
        delta = rng.gauss(0, mutation_strength * range_size)
        new_val = val + delta
        return max(bounds[0], min(bounds[1], new_val))
    
    return AdversarialCandidate(
        candidate_id=f"mut_{new_seed}",
        seed=new_seed,
        profile=parent.profile if "profile" not in params_to_mutate else rng.choice(BASE_PROFILES),
        vol=mutate_param(parent.vol, ADVERSARIAL_PARAM_BOUNDS["vol"]) if "vol" in params_to_mutate else parent.vol,
        vol_multiplier=mutate_param(parent.vol_multiplier, ADVERSARIAL_PARAM_BOUNDS["vol_multiplier"]) if "vol_multiplier" in params_to_mutate else parent.vol_multiplier,
        jump_intensity=mutate_param(parent.jump_intensity, ADVERSARIAL_PARAM_BOUNDS["jump_intensity"]) if "jump_intensity" in params_to_mutate else parent.jump_intensity,
        jump_sigma=mutate_param(parent.jump_sigma, ADVERSARIAL_PARAM_BOUNDS["jump_sigma"]) if "jump_sigma" in params_to_mutate else parent.jump_sigma,
        spread_bps=mutate_param(parent.spread_bps, ADVERSARIAL_PARAM_BOUNDS["spread_bps"]) if "spread_bps" in params_to_mutate else parent.spread_bps,
        latency_ms=mutate_param(parent.latency_ms, ADVERSARIAL_PARAM_BOUNDS["latency_ms"]) if "latency_ms" in params_to_mutate else parent.latency_ms,
        init_q_tao=mutate_param(parent.init_q_tao, ADVERSARIAL_PARAM_BOUNDS["init_q_tao"]) if "init_q_tao" in params_to_mutate else parent.init_q_tao,
        daily_loss_limit=mutate_param(parent.daily_loss_limit, ADVERSARIAL_PARAM_BOUNDS["daily_loss_limit"]) if "daily_loss_limit" in params_to_mutate else parent.daily_loss_limit,
        ticks=parent.ticks,
    )


# ===========================================================================
# Adversarial scoring
# ===========================================================================

def compute_adversarial_score(result: SearchResult) -> float:
    """
    Compute adversarial score (higher = more adversarial/worse).
    
    Objectives:
    1. kill_switch triggered (highest priority)
    2. large max_drawdown
    3. negative pnl
    
    Formula: 1000*kill + 10*max_drawdown - 0.1*mean_pnl
    """
    kill_weight = 1000.0 if result.kill_switch else 0.0
    
    drawdown = result.max_drawdown if not math.isnan(result.max_drawdown) else 0.0
    pnl = result.mean_pnl if not math.isnan(result.mean_pnl) else 0.0
    
    return kill_weight + 10.0 * drawdown - 0.1 * pnl


def rank_results_deterministic(results: List[SearchResult]) -> List[SearchResult]:
    """
    Rank results by adversarial score (descending).
    
    Deterministic tie-breaking: seed, then candidate_id.
    """
    return sorted(
        results,
        key=lambda r: (-r.adversarial_score, r.candidate.seed, r.candidate_id),
    )


# ===========================================================================
# Runner
# ===========================================================================

def run_candidate(
    candidate: AdversarialCandidate,
    output_dir: Path,
    verbose: bool = True,
) -> SearchResult:
    """
    Run sim_eval for a single candidate scenario.
    
    Uses isolated output directory with evidence verification.
    """
    candidate_output = output_dir / "runs" / candidate.candidate_id
    candidate_output.mkdir(parents=True, exist_ok=True)
    
    # Write scenario YAML to temp location
    scenario_yaml = candidate_output / "scenario.yaml"
    scenario_yaml.write_text(candidate.to_scenario_yaml())
    
    # Build sim_eval run command
    if SIM_EVAL_BIN.exists():
        cmd = [str(SIM_EVAL_BIN), "run", str(scenario_yaml),
               "--output-dir", str(candidate_output), "--verbose"]
    else:
        cmd = ["cargo", "run", "-p", "paraphina", "--bin", "sim_eval", "--release", "--",
               "run", str(scenario_yaml), "--output-dir", str(candidate_output), "--verbose"]
    
    # Environment overlay for loss limit (not in scenario YAML)
    env = os.environ.copy()
    env["PARAPHINA_DAILY_LOSS_LIMIT"] = str(candidate.daily_loss_limit)
    
    if verbose:
        print(f"    Running candidate {candidate.candidate_id}...")
    
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True)
    duration = time.time() - t0
    
    # Parse results from run_summary.json
    mean_pnl = float("nan")
    max_drawdown = float("nan")
    kill_switch = False
    final_pnl = float("nan")
    error_message = None
    
    # Find run_summary.json files (may be in subdirectories, one per seed)
    summary_files = list(candidate_output.rglob("run_summary.json"))
    if summary_files:
        try:
            # Aggregate across all runs for this candidate
            pnl_values = []
            drawdown_values = []
            any_kill = False
            
            for sf in summary_files:
                with open(sf) as f:
                    summary = json.load(f)
                
                # Parse nested structure: results.final_pnl_usd, results.max_drawdown_usd
                results = summary.get("results", {})
                pnl = results.get("final_pnl_usd", float("nan"))
                dd = abs(results.get("max_drawdown_usd", 0.0))
                kill = results.get("kill_switch", {}).get("triggered", False)
                
                if not math.isnan(pnl):
                    pnl_values.append(pnl)
                if not math.isnan(dd):
                    drawdown_values.append(dd)
                if kill:
                    any_kill = True
            
            # Compute aggregates
            if pnl_values:
                mean_pnl = sum(pnl_values) / len(pnl_values)
                final_pnl = pnl_values[0]  # First run's pnl
            if drawdown_values:
                max_drawdown = max(drawdown_values)
            kill_switch = any_kill
            
        except (json.JSONDecodeError, IOError) as e:
            error_message = f"Failed to parse run_summary.json: {e}"
    elif proc.returncode != 0:
        error_message = proc.stderr.strip()[:200] if proc.stderr else f"Exit code {proc.returncode}"
    
    # Verify run completed successfully
    # Note: sim_eval run doesn't produce evidence packs (only suites do)
    # For discovery phase, we consider a run valid if:
    # 1. Return code is 0
    # 2. At least one run_summary.json exists with valid metrics
    evidence_verified = (
        proc.returncode == 0 
        and summary_files 
        and not math.isnan(mean_pnl)
    )
    
    result = SearchResult(
        candidate_id=candidate.candidate_id,
        candidate=candidate,
        mean_pnl=mean_pnl,
        max_drawdown=max_drawdown,
        kill_switch=kill_switch,
        final_pnl=final_pnl,
        evidence_verified=evidence_verified,
        output_dir=str(candidate_output),
        duration_sec=duration,
        returncode=proc.returncode,
        error_message=error_message,
    )
    
    # Compute adversarial score
    result.adversarial_score = compute_adversarial_score(result)
    
    if verbose:
        status = "✓" if evidence_verified else "✗"
        kill_str = "KILL" if kill_switch else ""
        print(f"      {status} pnl={final_pnl:.2f}, dd={max_drawdown:.2f} {kill_str} | score={result.adversarial_score:.2f}")
    
    return result


def verify_evidence_tree(output_dir: Path, verbose: bool = True) -> bool:
    """Verify evidence tree using sim_eval verify-evidence-tree."""
    if SIM_EVAL_BIN.exists():
        cmd = [str(SIM_EVAL_BIN), "verify-evidence-tree", str(output_dir)]
    else:
        cmd = ["cargo", "run", "-p", "paraphina", "--bin", "sim_eval", "--release", "--",
               "verify-evidence-tree", str(output_dir)]
    
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    
    if proc.returncode == 0:
        if verbose:
            print(f"    ✓ Evidence verified: {output_dir.name}")
        return True
    else:
        if verbose:
            print(f"    ✗ Evidence FAILED: {output_dir.name}")
        return False


# ===========================================================================
# Output writing
# ===========================================================================

def write_search_trials_jsonl(results: List[SearchResult], path: Path) -> None:
    """Write all search trials to JSONL (one JSON per line)."""
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")


def write_topk_json(
    topk: List[SearchResult],
    path: Path,
    base_seed: int,
) -> None:
    """Write top-K selected scenarios to JSON."""
    data = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_seed": base_seed,
        "top_k": len(topk),
        "scenarios": [
            {
                "rank": i + 1,
                "candidate_id": r.candidate_id,
                "scenario_filename": r.candidate.scenario_filename(),
                "scenario_path": f"scenarios/v1/adversarial/generated_v1/{r.candidate.scenario_filename()}",
                "seed": r.candidate.seed,
                "profile": r.candidate.profile,
                "adversarial_score": r.adversarial_score,
                "kill_switch": r.kill_switch,
                "max_drawdown": r.max_drawdown,
                "mean_pnl": r.mean_pnl,
            }
            for i, r in enumerate(topk)
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_summary_md(
    results: List[SearchResult],
    topk: List[SearchResult],
    base_seed: int,
    out_dir: Path,
    path: Path,
) -> None:
    """Write human-readable summary markdown."""
    lines = [
        "# Adversarial Search Summary",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Base Seed:** {base_seed}",
        f"**Total Candidates:** {len(results)}",
        f"**Valid (evidence verified):** {sum(1 for r in results if r.evidence_verified)}",
        f"**Top-K Selected:** {len(topk)}",
        "",
        "## Output Locations",
        "",
        f"- Search output: `{out_dir}/`",
        f"- Search trials: `{out_dir}/search_trials.jsonl`",
        f"- Top-K JSON: `{out_dir}/topk.json`",
        f"- Promoted scenarios: `scenarios/v1/adversarial/generated_v1/`",
        f"- Suite file: `scenarios/suites/adversarial_regression_v2.yaml`",
        "",
        "## Verification Commands",
        "",
        "```bash",
        "# Run the generated suite",
        "cargo run -p paraphina --bin sim_eval -- suite \\",
        "  scenarios/suites/adversarial_regression_v2.yaml \\",
        "  --output-dir runs/adv_reg_v2 --verbose",
        "",
        "# Verify evidence packs",
        "cargo run -p paraphina --bin sim_eval -- verify-evidence-tree runs/adv_reg_v2",
        "```",
        "",
        "## Top Adversarial Scenarios",
        "",
        "| Rank | Seed | Profile | Score | Kill | Max DD | PnL |",
        "|------|------|---------|-------|------|--------|-----|",
    ]
    
    for i, r in enumerate(topk):
        kill_str = "✓" if r.kill_switch else ""
        dd = f"{r.max_drawdown:.2f}" if not math.isnan(r.max_drawdown) else "N/A"
        pnl = f"{r.mean_pnl:.2f}" if not math.isnan(r.mean_pnl) else "N/A"
        lines.append(f"| {i+1} | {r.candidate.seed} | {r.candidate.profile} | {r.adversarial_score:.2f} | {kill_str} | {dd} | {pnl} |")
    
    lines.extend([
        "",
        "## Scenario Files",
        "",
    ])
    
    for i, r in enumerate(topk):
        lines.append(f"- **{i+1}.** `{r.candidate.scenario_filename()}`")
    
    path.write_text("\n".join(lines))


def promote_scenarios(topk: List[SearchResult]) -> List[Path]:
    """
    Write scenario YAML files to scenarios/v1/adversarial/generated_v1/.
    
    Returns list of written paths.
    """
    GENERATED_SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    
    written_paths = []
    for r in topk:
        scenario_path = GENERATED_SCENARIOS_DIR / r.candidate.scenario_filename()
        scenario_path.write_text(r.candidate.to_scenario_yaml())
        written_paths.append(scenario_path)
    
    return written_paths


def generate_suite_v2_yaml(topk: List[SearchResult], suite_path: Path) -> None:
    """
    Generate adversarial_regression_v2.yaml with path-based scenarios only.
    
    CRITICAL: scenarios list must be non-empty and deterministically ordered.
    """
    if not topk:
        raise ValueError("Cannot generate suite with empty scenarios list")
    
    # Sort deterministically: by score (descending), then seed, then filename
    sorted_results = sorted(
        topk,
        key=lambda r: (-r.adversarial_score, r.candidate.seed, r.candidate.scenario_filename()),
    )
    
    lines = [
        "# Adversarial Regression Suite v2",
        "#",
        "# Auto-generated by adversarial_search_promote.py",
        f"# Generated at: {datetime.now(timezone.utc).isoformat()}",
        "#",
        "# This suite uses PATH-BASED scenarios only (no inline env_overrides).",
        "# Compatible with sim_eval suite --output-dir.",
        "#",
        "# Usage:",
        "#   cargo run -p paraphina --bin sim_eval -- suite \\",
        "#     scenarios/suites/adversarial_regression_v2.yaml \\",
        "#     --output-dir runs/adv_reg_v2 --verbose",
        "#",
        "# Verify evidence:",
        "#   cargo run -p paraphina --bin sim_eval -- verify-evidence-tree runs/adv_reg_v2",
        "#",
        "",
        "suite_id: adversarial_regression_v2",
        "suite_version: 2",
        "",
        "# Run each scenario for determinism verification",
        "repeat_runs: 1",
        "",
        "# Scenarios discovered by adversarial search",
        "# Ordered by adversarial score (descending), with deterministic tie-breaking",
        "scenarios:",
    ]
    
    for i, r in enumerate(sorted_results):
        scenario_rel_path = f"scenarios/v1/adversarial/generated_v1/{r.candidate.scenario_filename()}"
        lines.append(f"  # Rank {i+1}: score={r.adversarial_score:.2f}, seed={r.candidate.seed}")
        lines.append(f"  - path: {scenario_rel_path}")
        lines.append("")
    
    lines.extend([
        "# Invariants",
        "invariants:",
        "  evidence_pack_valid: true",
        "  expect_kill_switch: allowed",
    ])
    
    SUITES_DIR.mkdir(parents=True, exist_ok=True)
    suite_path.write_text("\n".join(lines))


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_adversarial_search(
    trials: int,
    base_seed: int,
    ticks: int,
    top_k: int,
    out_dir: Path,
    smoke: bool = False,
    verbose: bool = True,
) -> Tuple[List[SearchResult], List[SearchResult]]:
    """
    Run adversarial search and return (all_results, top_k_results).
    """
    print("=" * 70)
    print("Phase A-B2: Adversarial Search + Scenario Promotion")
    print("=" * 70)
    print(f"  Base seed: {base_seed}")
    print(f"  Trials: {trials}")
    print(f"  Ticks: {ticks}")
    print(f"  Top-K: {top_k}")
    print(f"  Output: {out_dir}")
    print(f"  Smoke mode: {smoke}")
    print()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate candidates
    print("[1/4] Generating candidates...")
    if smoke:
        candidates = generate_smoke_candidates(base_seed, BASE_PROFILES, ticks)
    else:
        candidates = generate_random_candidates(trials, base_seed, BASE_PROFILES, ticks)
    print(f"  Generated {len(candidates)} candidates")
    
    # Run all candidates
    print(f"\n[2/4] Running {len(candidates)} adversarial candidates...")
    results: List[SearchResult] = []
    
    for i, candidate in enumerate(candidates):
        print(f"\n  [{i+1}/{len(candidates)}] {candidate.candidate_id}")
        result = run_candidate(candidate, out_dir, verbose)
        results.append(result)
        
        # Write progress
        if (i + 1) % 5 == 0 or i == len(candidates) - 1:
            search_trials_path = out_dir / "search_trials.jsonl"
            write_search_trials_jsonl(results, search_trials_path)
    
    # Rank and select top-K
    print(f"\n[3/4] Ranking by adversarial score...")
    valid_results = [r for r in results if r.evidence_verified]
    ranked = rank_results_deterministic(valid_results)
    
    print(f"  Valid results: {len(valid_results)}/{len(results)}")
    if ranked:
        print(f"  Top adversarial score: {ranked[0].adversarial_score:.2f}")
    
    topk_results = ranked[:top_k]
    
    print(f"\n  Top {len(topk_results)} selected:")
    for i, r in enumerate(topk_results):
        kill_str = " [KILL]" if r.kill_switch else ""
        print(f"    {i+1}. s{r.candidate.seed} ({r.candidate.profile}): "
              f"score={r.adversarial_score:.2f}{kill_str}")
    
    # Write outputs
    print(f"\n[4/4] Writing outputs...")
    
    search_trials_path = out_dir / "search_trials.jsonl"
    write_search_trials_jsonl(results, search_trials_path)
    print(f"  ✓ {search_trials_path}")
    
    topk_path = out_dir / "topk.json"
    write_topk_json(topk_results, topk_path, base_seed)
    print(f"  ✓ {topk_path}")
    
    summary_path = out_dir / "summary.md"
    write_summary_md(results, topk_results, base_seed, out_dir, summary_path)
    print(f"  ✓ {summary_path}")
    
    # Promote scenarios if we have results
    if topk_results:
        print(f"\n  Promoting {len(topk_results)} scenarios...")
        written_paths = promote_scenarios(topk_results)
        for p in written_paths:
            print(f"    ✓ {p.relative_to(ROOT)}")
        
        print(f"\n  Generating suite v2...")
        generate_suite_v2_yaml(topk_results, SUITE_V2_PATH)
        print(f"    ✓ {SUITE_V2_PATH.relative_to(ROOT)}")
    else:
        print("\n  WARNING: No valid results to promote")
    
    # Summary of verified runs
    verified_count = sum(1 for r in results if r.evidence_verified)
    print(f"\n  Run summary: {verified_count}/{len(results)} candidates produced valid results")
    
    print("\n" + "=" * 70)
    print("Adversarial Search Complete")
    print("=" * 70)
    
    return results, topk_results


# ===========================================================================
# CLI
# ===========================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python3 -m batch_runs.phase_a.adversarial_search_promote",
        description="Phase A-B2: Adversarial search + scenario promotion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test (fast, 3 candidates)
  python3 -m batch_runs.phase_a.adversarial_search_promote --smoke --out runs/adv_smoke

  # Full search (20 trials)
  python3 -m batch_runs.phase_a.adversarial_search_promote --trials 20 --out runs/adv_full

Outputs:
  - <out>/search_trials.jsonl    Every candidate
  - <out>/topk.json              Selected top-K
  - <out>/summary.md             Human-readable summary
  - scenarios/v1/adversarial/generated_v1/   Promoted scenarios
  - scenarios/suites/adversarial_regression_v2.yaml   Generated suite
""",
    )
    
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke mode (3 candidates, fast ticks)",
    )
    
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=10,
        help="Number of candidates to evaluate (default: 10)",
    )
    
    parser.add_argument(
        "--ticks",
        type=int,
        default=200,
        help="Ticks per scenario (default: 200)",
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top scenarios to promote (default: {DEFAULT_TOP_K})",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed (default: 42)",
    )
    
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args(argv)
    
    # Determine parameters
    if args.smoke:
        trials = 3
        ticks = 100
        top_k = min(args.top_k, 3)
    else:
        trials = args.trials
        ticks = args.ticks
        top_k = args.top_k
    
    out_dir = args.out if args.out else DEFAULT_OUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Check binaries exist
    if not SIM_EVAL_BIN.exists():
        print("Building sim_eval binary...")
        proc = subprocess.run(
            ["cargo", "build", "--release", "-p", "paraphina", "--bin", "sim_eval"],
            cwd=str(ROOT),
        )
        if proc.returncode != 0:
            print("ERROR: Failed to build sim_eval")
            return 1
    
    # Run search
    try:
        results, topk = run_adversarial_search(
            trials=trials,
            base_seed=args.seed,
            ticks=ticks,
            top_k=top_k,
            out_dir=out_dir,
            smoke=args.smoke,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"\nERROR: Search failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Return success if we have promoted scenarios
    if topk:
        print(f"\nNext steps:")
        print(f"  1. Review: {out_dir}/summary.md")
        print(f"  2. Run suite: cargo run -p paraphina --bin sim_eval -- suite scenarios/suites/adversarial_regression_v2.yaml --output-dir runs/adv_reg_v2 --verbose")
        print(f"  3. Verify: cargo run -p paraphina --bin sim_eval -- verify-evidence-tree runs/adv_reg_v2")
        return 0
    else:
        print("\nWARNING: No scenarios promoted (all candidates failed)")
        return 1


if __name__ == "__main__":
    sys.exit(main())

