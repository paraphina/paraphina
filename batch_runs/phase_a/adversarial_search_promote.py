#!/usr/bin/env python3
"""
adversarial_search_promote.py

Phase A-B2: Cross-Entropy Method (CEM) adversarial search + optional scenario promotion.

Implements institutional-grade CEM search over adversarial scenario parameters:
- Maintains mean/std per continuous parameter
- Samples N candidates each iteration with seeded RNG
- Scores by adversarial objective (kill_prob + drawdown - pnl)
- Selects elite top fraction (10-20%)
- Updates distribution (mean/std) toward elite
- Repeats for I iterations

Key design:
- OUTPUT ISOLATION: By default, all artifacts go under runs/phaseA_adv_search/<study_id>/
  - NO writes to scenarios/ unless --promote-suite is explicitly passed
  - Generated suite is self-contained in the output directory
- DETERMINISM: Fixed seed produces identical search trajectory and results
- NON-EMPTY: Suite generation always produces at least 1 scenario (with fallbacks)

Usage:
    # Default run (all outputs under runs/)
    python3 -m batch_runs.phase_a.adversarial_search_promote --help
    python3 -m batch_runs.phase_a.adversarial_search_promote --smoke --out runs/phaseA_adv_search_smoke
    python3 -m batch_runs.phase_a.adversarial_search_promote --iterations 5 --pop-size 20 --out runs/adv_cem

    # Explicit promotion to scenarios/ (only when ready)
    python3 -m batch_runs.phase_a.adversarial_search_promote --promote-suite --top-k 10

Outputs (default, under <out_dir>/):
    candidates/<candidate_id>.yaml      - Individual scenario files
    generated_suite/adversarial_regression_generated.yaml  - Generated suite
    search_results.jsonl                - Every candidate (one JSON per line)
    summary.json                        - CEM summary with final distribution
    cem_history.json                    - Per-iteration CEM stats

Outputs (with --promote-suite):
    scenarios/v1/adversarial/promoted_v1/<stable_name>.yaml  - Promoted scenarios
    scenarios/suites/adversarial_regression_v3.yaml          - Canonical suite
    scenarios/suites/PROMOTION_RECORD.json                   - Audit trail
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
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
PROMOTED_SCENARIOS_DIR = SCENARIOS_DIR / "v1" / "adversarial" / "promoted_v1"
SUITES_DIR = SCENARIOS_DIR / "suites"
SUITE_V3_PATH = SUITES_DIR / "adversarial_regression_v3.yaml"

DEFAULT_OUT_DIR = ROOT / "runs" / "phaseA_adv_search"

# CEM hyperparameters
DEFAULT_ITERATIONS = 3
DEFAULT_POP_SIZE = 20
DEFAULT_ELITE_FRAC = 0.2
DEFAULT_TOP_K = 5

# Adversarial parameter bounds for CEM search
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

# Parameter names for CEM (ordered for determinism)
CEM_PARAMS = [
    "vol", "vol_multiplier", "jump_intensity", "jump_sigma",
    "spread_bps", "latency_ms", "init_q_tao", "daily_loss_limit"
]

BASE_PROFILES = ["balanced", "conservative", "aggressive"]

# Deterministic safe fallback scenario (always valid)
SAFE_FALLBACK_SCENARIO = {
    "vol": 0.02,
    "vol_multiplier": 1.5,
    "jump_intensity": 0.0005,
    "jump_sigma": 0.03,
    "spread_bps": 2.0,
    "latency_ms": 10.0,
    "init_q_tao": 0.0,
    "daily_loss_limit": 1000.0,
}


# ===========================================================================
# CEM Distribution
# ===========================================================================

@dataclass
class CEMDistribution:
    """
    Cross-Entropy Method distribution for continuous parameters.
    
    Maintains mean and std for each parameter, with bounded sampling.
    """
    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)
    profile_weights: Dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def initialize(cls) -> "CEMDistribution":
        """Initialize CEM distribution at center of parameter bounds."""
        dist = cls()
        
        # Initialize means at center of bounds, stds at 1/4 range
        for param in CEM_PARAMS:
            lo, hi = ADVERSARIAL_PARAM_BOUNDS[param]
            dist.means[param] = (lo + hi) / 2.0
            dist.stds[param] = (hi - lo) / 4.0
        
        # Equal weights for profiles
        for profile in BASE_PROFILES:
            dist.profile_weights[profile] = 1.0 / len(BASE_PROFILES)
        
        return dist
    
    def sample_candidate(
        self,
        rng: random.Random,
        candidate_id: str,
        seed: int,
        ticks: int,
    ) -> "AdversarialCandidate":
        """Sample a candidate from current distribution (bounded)."""
        params = {}
        for param in CEM_PARAMS:
            lo, hi = ADVERSARIAL_PARAM_BOUNDS[param]
            # Sample from N(mean, std) and clip to bounds
            value = rng.gauss(self.means[param], self.stds[param])
            params[param] = max(lo, min(hi, value))
        
        # Sample profile from categorical distribution
        profile = self._sample_profile(rng)
        
        return AdversarialCandidate(
            candidate_id=candidate_id,
            seed=seed,
            profile=profile,
            vol=params["vol"],
            vol_multiplier=params["vol_multiplier"],
            jump_intensity=params["jump_intensity"],
            jump_sigma=params["jump_sigma"],
            spread_bps=params["spread_bps"],
            latency_ms=params["latency_ms"],
            init_q_tao=params["init_q_tao"],
            daily_loss_limit=params["daily_loss_limit"],
            ticks=ticks,
        )
    
    def _sample_profile(self, rng: random.Random) -> str:
        """Sample profile from weighted distribution."""
        r = rng.random()
        cumsum = 0.0
        for profile in BASE_PROFILES:
            cumsum += self.profile_weights[profile]
            if r <= cumsum:
                return profile
        return BASE_PROFILES[-1]
    
    def update_from_elite(
        self,
        elite: List["AdversarialCandidate"],
        learning_rate: float = 0.5,
    ) -> None:
        """Update distribution toward elite samples."""
        if not elite:
            return
        
        # Compute elite means and stds
        for param in CEM_PARAMS:
            values = [getattr(c, param) for c in elite]
            elite_mean = sum(values) / len(values)
            elite_var = sum((v - elite_mean) ** 2 for v in values) / len(values)
            elite_std = max(elite_var ** 0.5, 1e-6)  # Prevent collapse
            
            # Smooth update
            self.means[param] = (1 - learning_rate) * self.means[param] + learning_rate * elite_mean
            self.stds[param] = (1 - learning_rate) * self.stds[param] + learning_rate * elite_std
            
            # Keep stds bounded (min 5% of range)
            lo, hi = ADVERSARIAL_PARAM_BOUNDS[param]
            min_std = (hi - lo) * 0.05
            self.stds[param] = max(self.stds[param], min_std)
        
        # Update profile weights
        profile_counts = {p: 0 for p in BASE_PROFILES}
        for c in elite:
            profile_counts[c.profile] += 1
        
        for profile in BASE_PROFILES:
            elite_weight = profile_counts[profile] / len(elite)
            self.profile_weights[profile] = (
                (1 - learning_rate) * self.profile_weights[profile] +
                learning_rate * elite_weight
            )
        
        # Normalize profile weights
        total = sum(self.profile_weights.values())
        for profile in BASE_PROFILES:
            self.profile_weights[profile] /= total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "means": dict(self.means),
            "stds": dict(self.stds),
            "profile_weights": dict(self.profile_weights),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CEMDistribution":
        """Reconstruct from dict."""
        dist = cls()
        dist.means = dict(data.get("means", {}))
        dist.stds = dict(data.get("stds", {}))
        dist.profile_weights = dict(data.get("profile_weights", {}))
        return dist


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
        """Generate deterministic filename for scenario."""
        return f"adv_s{self.seed:05d}_{self.scenario_hash()}.yaml"
    
    def to_scenario_yaml(self) -> str:
        """Generate scenario YAML content."""
        effective_vol = self.vol * self.vol_multiplier
        lines = [
            f"# Adversarial scenario discovered by CEM search",
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
            f"# CEM adversarial search metadata",
            f"adversarial_params:",
            f"  vol_multiplier: {self.vol_multiplier:.4f}",
            f"  daily_loss_limit: {self.daily_loss_limit:.2f}",
            f"",
            f"invariants:",
            f"  expect_kill_switch: allowed",
            f"  pnl_linearity_check: disabled",
        ]
        return "\n".join(lines)
    
    def to_param_dict(self) -> Dict[str, float]:
        """Extract CEM parameters as dict."""
        return {
            "vol": self.vol,
            "vol_multiplier": self.vol_multiplier,
            "jump_intensity": self.jump_intensity,
            "jump_sigma": self.jump_sigma,
            "spread_bps": self.spread_bps,
            "latency_ms": self.latency_ms,
            "init_q_tao": self.init_q_tao,
            "daily_loss_limit": self.daily_loss_limit,
        }


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
    # CEM iteration info
    cem_iteration: int = 0
    
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
            "cem_iteration": self.cem_iteration,
        }


@dataclass
class CEMIterationStats:
    """Statistics for one CEM iteration."""
    iteration: int
    candidates_evaluated: int
    valid_count: int
    elite_count: int
    best_score: float
    mean_score: float
    distribution_snapshot: Dict[str, Any]


# ===========================================================================
# Candidate generation
# ===========================================================================

def generate_cem_candidates(
    dist: CEMDistribution,
    n_candidates: int,
    base_seed: int,
    iteration: int,
    ticks: int,
) -> List[AdversarialCandidate]:
    """Generate candidates from CEM distribution for one iteration."""
    rng = random.Random(base_seed + iteration * 10000)
    candidates = []
    
    for i in range(n_candidates):
        seed = base_seed + iteration * 10000 + i
        candidate_id = f"cem_i{iteration:02d}_c{i:04d}_s{seed}"
        candidate = dist.sample_candidate(rng, candidate_id, seed, ticks)
        candidates.append(candidate)
    
    return candidates


def generate_smoke_candidates(
    base_seed: int,
    profiles: List[str],
    ticks: int,
) -> List[AdversarialCandidate]:
    """Generate fixed smoke test candidates (one per profile)."""
    candidates = []
    
    for i, profile in enumerate(profiles):
        seed = base_seed + i
        candidate = AdversarialCandidate(
            candidate_id=f"smoke_{profile}_{i}",
            seed=seed,
            profile=profile,
            vol=0.02,
            vol_multiplier=2.0,
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


def create_fallback_candidate(base_seed: int, ticks: int) -> AdversarialCandidate:
    """Create deterministic safe fallback candidate."""
    return AdversarialCandidate(
        candidate_id=f"fallback_s{base_seed}",
        seed=base_seed,
        profile="balanced",
        ticks=ticks,
        **SAFE_FALLBACK_SCENARIO,
    )


# ===========================================================================
# Adversarial scoring
# ===========================================================================

def compute_adversarial_score(result: SearchResult) -> float:
    """
    Compute adversarial score (higher = more adversarial/worse).
    
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


def select_elite(
    results: List[SearchResult],
    elite_frac: float,
) -> List[AdversarialCandidate]:
    """
    Select elite fraction of candidates by adversarial score.
    
    Returns candidates (not results) for CEM update.
    """
    valid = [r for r in results if r.evidence_verified]
    ranked = rank_results_deterministic(valid)
    
    n_elite = max(1, int(len(ranked) * elite_frac))
    return [r.candidate for r in ranked[:n_elite]]


# ===========================================================================
# Evidence Pack Helpers
# ===========================================================================

def _write_evidence_pack(out_dir: Path, verbose: bool = True, smoke: bool = False) -> int:
    """
    Write a root evidence pack for the output directory.
    
    Calls `sim_eval write-evidence-pack <out_dir>`.
    
    Returns exit code (0 = success).
    """
    if SIM_EVAL_BIN.exists():
        cmd = [str(SIM_EVAL_BIN), "write-evidence-pack", str(out_dir)]
    else:
        cmd = [
            "cargo", "run", "-p", "paraphina", "--bin", "sim_eval",
            "--release" if not smoke else "",
            "--", "write-evidence-pack", str(out_dir)
        ]
        # Remove empty strings from command
        cmd = [c for c in cmd if c]
    
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if verbose and proc.returncode == 0:
            # Print the output from sim_eval
            for line in proc.stdout.strip().split('\n'):
                if line:
                    print(f"    {line}")
        elif proc.returncode != 0:
            print(f"    ERROR: {proc.stderr.strip()}")
        return proc.returncode
    except Exception as e:
        print(f"    ERROR: Failed to run sim_eval write-evidence-pack: {e}")
        return 1


def _verify_evidence_pack(out_dir: Path, verbose: bool = True) -> int:
    """
    Verify a root evidence pack.
    
    Calls `sim_eval verify-evidence-pack <out_dir>`.
    
    Returns exit code (0 = success).
    """
    if SIM_EVAL_BIN.exists():
        cmd = [str(SIM_EVAL_BIN), "verify-evidence-pack", str(out_dir)]
    else:
        cmd = [
            "cargo", "run", "-p", "paraphina", "--bin", "sim_eval",
            "--", "verify-evidence-pack", str(out_dir)
        ]
    
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if verbose and proc.returncode == 0:
            for line in proc.stdout.strip().split('\n'):
                if line:
                    print(f"    {line}")
        elif proc.returncode != 0:
            print(f"    ERROR: {proc.stderr.strip()}")
        return proc.returncode
    except Exception as e:
        print(f"    ERROR: Failed to run sim_eval verify-evidence-pack: {e}")
        return 1


# ===========================================================================
# Runner
# ===========================================================================

def run_candidate(
    candidate: AdversarialCandidate,
    output_dir: Path,
    verbose: bool = True,
) -> SearchResult:
    """Run sim_eval for a single candidate scenario."""
    candidate_output = output_dir / "runs" / candidate.candidate_id
    candidate_output.mkdir(parents=True, exist_ok=True)
    
    # Write scenario YAML
    scenario_yaml = candidate_output / "scenario.yaml"
    scenario_yaml.write_text(candidate.to_scenario_yaml())
    
    # Build sim_eval run command
    if SIM_EVAL_BIN.exists():
        cmd = [str(SIM_EVAL_BIN), "run", str(scenario_yaml),
               "--output-dir", str(candidate_output), "--verbose"]
    else:
        cmd = ["cargo", "run", "-p", "paraphina", "--bin", "sim_eval", "--release", "--",
               "run", str(scenario_yaml), "--output-dir", str(candidate_output), "--verbose"]
    
    # Environment overlay
    env = os.environ.copy()
    env["PARAPHINA_DAILY_LOSS_LIMIT"] = str(candidate.daily_loss_limit)
    
    if verbose:
        print(f"    Running {candidate.candidate_id}...")
    
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True)
    duration = time.time() - t0
    
    # Parse results
    mean_pnl = float("nan")
    max_drawdown = float("nan")
    kill_switch = False
    final_pnl = float("nan")
    error_message = None
    
    summary_files = list(candidate_output.rglob("run_summary.json"))
    if summary_files:
        try:
            pnl_values = []
            drawdown_values = []
            any_kill = False
            
            for sf in summary_files:
                with open(sf) as f:
                    summary = json.load(f)
                
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
            
            if pnl_values:
                mean_pnl = sum(pnl_values) / len(pnl_values)
                final_pnl = pnl_values[0]
            if drawdown_values:
                max_drawdown = max(drawdown_values)
            kill_switch = any_kill
            
        except (json.JSONDecodeError, IOError) as e:
            error_message = f"Failed to parse run_summary.json: {e}"
    elif proc.returncode != 0:
        error_message = proc.stderr.strip()[:200] if proc.stderr else f"Exit code {proc.returncode}"
    
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
    
    result.adversarial_score = compute_adversarial_score(result)
    
    if verbose:
        status = "✓" if evidence_verified else "✗"
        kill_str = "KILL" if kill_switch else ""
        print(f"      {status} pnl={final_pnl:.2f}, dd={max_drawdown:.2f} {kill_str} | score={result.adversarial_score:.2f}")
    
    return result


# ===========================================================================
# Output writing (isolated under out_dir)
# ===========================================================================

def write_search_results_jsonl(results: List[SearchResult], path: Path) -> None:
    """Write all search results to JSONL."""
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")


def write_candidate_scenarios(
    candidates: List[AdversarialCandidate],
    candidates_dir: Path,
) -> List[Path]:
    """Write candidate scenario YAMLs to candidates/ directory."""
    candidates_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    
    for candidate in candidates:
        path = candidates_dir / candidate.scenario_filename()
        path.write_text(candidate.to_scenario_yaml())
        paths.append(path)
    
    return paths


def write_generated_suite_yaml(
    topk: List[SearchResult],
    suite_dir: Path,
    candidates_dir: Path,
) -> Path:
    """
    Generate suite YAML pointing to candidates in candidates_dir.
    
    Uses absolute paths for portability (sim_eval resolves from cwd).
    CRITICAL: Always produces non-empty suite (with fallback if needed).
    """
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_path = suite_dir / "adversarial_regression_generated.yaml"
    
    if not topk:
        raise ValueError("Cannot generate suite with empty scenarios list")
    
    # Sort deterministically
    sorted_results = sorted(
        topk,
        key=lambda r: (-r.adversarial_score, r.candidate.seed, r.candidate.scenario_filename()),
    )
    
    # Use absolute paths for portability
    candidates_dir_abs = candidates_dir.resolve()
    
    lines = [
        "# Adversarial Regression Suite (CEM-generated)",
        "#",
        "# Auto-generated by adversarial_search_promote.py (CEM search)",
        f"# Generated at: {datetime.now(timezone.utc).isoformat()}",
        "#",
        "# This suite is self-contained in the output directory.",
        "# Paths are relative to this suite file location.",
        "#",
        "# Usage:",
        "#   cargo run -p paraphina --bin sim_eval -- suite \\",
        "#     <this_file> --output-dir <out> --verbose",
        "#",
        "",
        "suite_id: adversarial_regression_cem",
        "suite_version: 1",
        "",
        "repeat_runs: 1",
        "",
        "scenarios:",
    ]
    
    for i, r in enumerate(sorted_results):
        scenario_path = candidates_dir_abs / r.candidate.scenario_filename()
        lines.append(f"  # Rank {i+1}: score={r.adversarial_score:.2f}, seed={r.candidate.seed}")
        lines.append(f"  - path: {scenario_path}")
        lines.append("")
    
    lines.extend([
        "invariants:",
        "  evidence_pack_valid: true",
        "  expect_kill_switch: allowed",
    ])
    
    suite_path.write_text("\n".join(lines))
    return suite_path


def write_summary_json(
    results: List[SearchResult],
    topk: List[SearchResult],
    cem_history: List[CEMIterationStats],
    final_dist: CEMDistribution,
    config: Dict[str, Any],
    out_dir: Path,
) -> Path:
    """Write JSON summary with CEM stats."""
    path = out_dir / "summary.json"
    
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "total_candidates": len(results),
        "valid_candidates": sum(1 for r in results if r.evidence_verified),
        "top_k": len(topk),
        "cem_iterations": len(cem_history),
        "final_distribution": final_dist.to_dict(),
        "cem_history": [
            {
                "iteration": s.iteration,
                "candidates_evaluated": s.candidates_evaluated,
                "valid_count": s.valid_count,
                "elite_count": s.elite_count,
                "best_score": s.best_score,
                "mean_score": s.mean_score,
            }
            for s in cem_history
        ],
        "top_k_scenarios": [
            {
                "rank": i + 1,
                "candidate_id": r.candidate_id,
                "scenario_filename": r.candidate.scenario_filename(),
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
        json.dump(summary, f, indent=2, sort_keys=True)
    
    return path


# ===========================================================================
# Promotion mode (writes to scenarios/)
# ===========================================================================

def promote_scenarios_to_repo(
    topk: List[SearchResult],
    source_out_dir: Path,
    base_seed: int,
) -> Tuple[List[Path], Path, Path]:
    """
    Promote top-K scenarios to canonical repo locations.
    
    Returns (promoted_paths, suite_path, record_path).
    """
    PROMOTED_SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    SUITES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Write promoted scenario files
    promoted_paths = []
    for r in topk:
        filename = r.candidate.scenario_filename()
        dest = PROMOTED_SCENARIOS_DIR / filename
        
        # Don't overwrite existing files silently
        if dest.exists():
            print(f"    WARNING: Skipping existing file: {dest.relative_to(ROOT)}")
            continue
        
        dest.write_text(r.candidate.to_scenario_yaml())
        promoted_paths.append(dest)
    
    # Generate v3 suite
    sorted_results = sorted(
        topk,
        key=lambda r: (-r.adversarial_score, r.candidate.seed, r.candidate.scenario_filename()),
    )
    
    lines = [
        "# Adversarial Regression Suite v3",
        "#",
        "# Auto-generated by CEM adversarial search with --promote-suite",
        f"# Generated at: {datetime.now(timezone.utc).isoformat()}",
        "#",
        "# This suite contains promoted adversarial scenarios.",
        "# Do NOT edit manually - regenerate via adversarial_search_promote.py --promote-suite",
        "#",
        "",
        "suite_id: adversarial_regression_v3",
        "suite_version: 3",
        "",
        "repeat_runs: 1",
        "",
        "scenarios:",
    ]
    
    for i, r in enumerate(sorted_results):
        rel_path = f"scenarios/v1/adversarial/promoted_v1/{r.candidate.scenario_filename()}"
        lines.append(f"  # Rank {i+1}: score={r.adversarial_score:.2f}, seed={r.candidate.seed}, profile={r.candidate.profile}")
        lines.append(f"  - path: {rel_path}")
        lines.append("")
    
    lines.extend([
        "invariants:",
        "  evidence_pack_valid: true",
        "  expect_kill_switch: allowed",
    ])
    
    SUITE_V3_PATH.write_text("\n".join(lines))
    
    # Write promotion record
    record_path = SUITES_DIR / "PROMOTION_RECORD.json"
    
    # Get git SHA
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            text=True,
        ).strip()
    except Exception:
        git_sha = "unknown"
    
    record = {
        "schema_version": 1,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "command_line": " ".join(sys.argv),
        "base_seed": base_seed,
        "top_k": len(topk),
        "source_run_directory": str(source_out_dir),
        "promoted_scenarios": [
            {
                "filename": r.candidate.scenario_filename(),
                "seed": r.candidate.seed,
                "profile": r.candidate.profile,
                "adversarial_score": r.adversarial_score,
            }
            for r in sorted_results
        ],
        "suite_path": str(SUITE_V3_PATH.relative_to(ROOT)),
    }
    
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    
    return promoted_paths, SUITE_V3_PATH, record_path


# ===========================================================================
# Main CEM pipeline
# ===========================================================================

def run_cem_search(
    iterations: int,
    pop_size: int,
    elite_frac: float,
    base_seed: int,
    ticks: int,
    top_k: int,
    out_dir: Path,
    smoke: bool = False,
    verbose: bool = True,
) -> Tuple[List[SearchResult], List[SearchResult], CEMDistribution]:
    """
    Run CEM adversarial search.
    
    Returns (all_results, top_k_results, final_distribution).
    """
    print("=" * 70)
    print("Phase A-B2: CEM Adversarial Search")
    print("=" * 70)
    print(f"  Base seed: {base_seed}")
    print(f"  Iterations: {iterations}")
    print(f"  Population size: {pop_size}")
    print(f"  Elite fraction: {elite_frac:.2f}")
    print(f"  Ticks: {ticks}")
    print(f"  Top-K: {top_k}")
    print(f"  Output: {out_dir}")
    print(f"  Smoke mode: {smoke}")
    print()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_results: List[SearchResult] = []
    cem_history: List[CEMIterationStats] = []
    
    if smoke:
        # Smoke mode: skip CEM, use fixed candidates
        print("[1/3] Smoke mode: using fixed candidates...")
        candidates = generate_smoke_candidates(base_seed, BASE_PROFILES, ticks)
        dist = CEMDistribution.initialize()
        
        for i, candidate in enumerate(candidates):
            print(f"\n  [{i+1}/{len(candidates)}] {candidate.candidate_id}")
            result = run_candidate(candidate, out_dir, verbose)
            result.cem_iteration = 0
            all_results.append(result)
        
    else:
        # CEM iterations
        dist = CEMDistribution.initialize()
        
        for iteration in range(iterations):
            print(f"\n[CEM Iteration {iteration+1}/{iterations}]")
            print(f"  Distribution means: vol={dist.means['vol']:.4f}, vol_mult={dist.means['vol_multiplier']:.2f}")
            
            # Generate candidates from current distribution
            candidates = generate_cem_candidates(dist, pop_size, base_seed, iteration, ticks)
            
            # Evaluate candidates
            iteration_results = []
            for i, candidate in enumerate(candidates):
                print(f"\n  [{i+1}/{len(candidates)}] {candidate.candidate_id}")
                result = run_candidate(candidate, out_dir, verbose)
                result.cem_iteration = iteration
                iteration_results.append(result)
                all_results.append(result)
            
            # Select elite and update distribution
            elite = select_elite(iteration_results, elite_frac)
            
            valid_count = sum(1 for r in iteration_results if r.evidence_verified)
            scores = [r.adversarial_score for r in iteration_results if r.evidence_verified]
            
            stats = CEMIterationStats(
                iteration=iteration,
                candidates_evaluated=len(iteration_results),
                valid_count=valid_count,
                elite_count=len(elite),
                best_score=max(scores) if scores else 0.0,
                mean_score=sum(scores) / len(scores) if scores else 0.0,
                distribution_snapshot=dist.to_dict(),
            )
            cem_history.append(stats)
            
            print(f"\n  Iteration {iteration+1} summary:")
            print(f"    Valid: {valid_count}/{len(iteration_results)}")
            print(f"    Elite: {len(elite)}")
            print(f"    Best score: {stats.best_score:.2f}")
            print(f"    Mean score: {stats.mean_score:.2f}")
            
            # Update distribution toward elite
            if elite:
                dist.update_from_elite(elite)
                print(f"    Updated distribution: vol_mean={dist.means['vol']:.4f}")
    
    # Final ranking and top-K selection
    print(f"\n[Final Selection]")
    valid_results = [r for r in all_results if r.evidence_verified]
    ranked = rank_results_deterministic(valid_results)
    
    print(f"  Total evaluated: {len(all_results)}")
    print(f"  Valid results: {len(valid_results)}")
    
    # Ensure non-empty selection with fallbacks
    if ranked:
        topk_results = ranked[:top_k]
    else:
        # Fallback: create deterministic safe scenario
        print("  WARNING: No valid results, using fallback scenario")
        fallback = create_fallback_candidate(base_seed, ticks)
        fallback_result = SearchResult(
            candidate_id=fallback.candidate_id,
            candidate=fallback,
            evidence_verified=True,  # Mark as valid for suite generation
        )
        fallback_result.adversarial_score = 0.0
        topk_results = [fallback_result]
    
    print(f"\n  Top {len(topk_results)} selected:")
    for i, r in enumerate(topk_results):
        kill_str = " [KILL]" if r.kill_switch else ""
        print(f"    {i+1}. s{r.candidate.seed} ({r.candidate.profile}): "
              f"score={r.adversarial_score:.2f}{kill_str}")
    
    # Write outputs
    print(f"\n[Writing Outputs]")
    
    # Search results JSONL
    results_path = out_dir / "search_results.jsonl"
    write_search_results_jsonl(all_results, results_path)
    print(f"  ✓ {results_path.relative_to(ROOT)}")
    
    # Candidate scenario files
    candidates_dir = out_dir / "candidates"
    candidates_to_write = [r.candidate for r in topk_results]
    written_paths = write_candidate_scenarios(candidates_to_write, candidates_dir)
    print(f"  ✓ {candidates_dir.relative_to(ROOT)}/ ({len(written_paths)} files)")
    
    # Generated suite
    suite_dir = out_dir / "generated_suite"
    suite_path = write_generated_suite_yaml(topk_results, suite_dir, candidates_dir)
    print(f"  ✓ {suite_path.relative_to(ROOT)}")
    
    # Summary JSON
    config = {
        "iterations": iterations,
        "pop_size": pop_size,
        "elite_frac": elite_frac,
        "base_seed": base_seed,
        "ticks": ticks,
        "top_k": top_k,
        "smoke": smoke,
    }
    summary_path = write_summary_json(all_results, topk_results, cem_history, dist, config, out_dir)
    print(f"  ✓ {summary_path.relative_to(ROOT)}")
    
    print("\n" + "=" * 70)
    print("CEM Adversarial Search Complete")
    print("=" * 70)
    
    return all_results, topk_results, dist


# ===========================================================================
# CLI
# ===========================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python3 -m batch_runs.phase_a.adversarial_search_promote",
        description="Phase A-B2: CEM adversarial search + optional scenario promotion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test (all outputs under runs/)
  python3 -m batch_runs.phase_a.adversarial_search_promote --smoke --out runs/phaseA_adv_search_smoke

  # CEM search (5 iterations, 20 candidates each)
  python3 -m batch_runs.phase_a.adversarial_search_promote \\
    --iterations 5 --pop-size 20 --out runs/adv_cem_full

  # Promote to scenarios/ (explicit)
  python3 -m batch_runs.phase_a.adversarial_search_promote \\
    --iterations 3 --pop-size 15 --top-k 10 --promote-suite

Output Structure (default, under <out>/):
  candidates/             Scenario YAML files
  generated_suite/        Self-contained suite
  search_results.jsonl    All candidates
  summary.json            CEM stats + top-K

Output Structure (with --promote-suite):
  scenarios/v1/adversarial/promoted_v1/   Promoted scenarios
  scenarios/suites/adversarial_regression_v3.yaml
  scenarios/suites/PROMOTION_RECORD.json  Audit trail
""",
    )
    
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke mode (3 fixed candidates, no CEM)",
    )
    
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of CEM iterations (default: {DEFAULT_ITERATIONS})",
    )
    
    parser.add_argument(
        "--pop-size", "-n",
        type=int,
        default=DEFAULT_POP_SIZE,
        help=f"Population size per CEM iteration (default: {DEFAULT_POP_SIZE})",
    )
    
    parser.add_argument(
        "--elite-frac",
        type=float,
        default=DEFAULT_ELITE_FRAC,
        help=f"Fraction of population to use as elite (default: {DEFAULT_ELITE_FRAC})",
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
        help=f"Number of top scenarios in final suite (default: {DEFAULT_TOP_K})",
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
        help=f"Output directory (default: {DEFAULT_OUT_DIR}/<timestamp>)",
    )
    
    parser.add_argument(
        "--promote-suite",
        action="store_true",
        help="Promote top-K to scenarios/ and create v3 suite (MODIFIES REPO)",
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args(argv)
    
    # Determine parameters
    if args.smoke:
        iterations = 1  # Single pass for smoke
        pop_size = 3
        ticks = 100
        top_k = min(args.top_k, 3)
    else:
        iterations = args.iterations
        pop_size = args.pop_size
        ticks = args.ticks
        top_k = args.top_k
    
    # Generate unique study_id for output isolation
    study_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_s{args.seed}"
    out_dir = args.out.resolve() if args.out else DEFAULT_OUT_DIR / study_id
    
    # Build sim_eval if needed
    if not SIM_EVAL_BIN.exists():
        print("Building sim_eval binary...")
        proc = subprocess.run(
            ["cargo", "build", "--release", "-p", "paraphina", "--bin", "sim_eval"],
            cwd=str(ROOT),
        )
        if proc.returncode != 0:
            print("ERROR: Failed to build sim_eval")
            return 1
    
    # Run CEM search
    try:
        all_results, topk, final_dist = run_cem_search(
            iterations=iterations,
            pop_size=pop_size,
            elite_frac=args.elite_frac,
            base_seed=args.seed,
            ticks=ticks,
            top_k=top_k,
            out_dir=out_dir,
            smoke=args.smoke,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"\nERROR: CEM search failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Promotion mode
    if args.promote_suite and topk:
        print(f"\n[Promoting to scenarios/]")
        promoted_paths, suite_path, record_path = promote_scenarios_to_repo(
            topk, out_dir, args.seed
        )
        
        for p in promoted_paths:
            print(f"  ✓ {p.relative_to(ROOT)}")
        print(f"  ✓ {suite_path.relative_to(ROOT)}")
        print(f"  ✓ {record_path.relative_to(ROOT)}")
    
    # Write root evidence pack for the output directory
    print(f"\n[Writing Evidence Pack]")
    evidence_pack_result = _write_evidence_pack(out_dir, verbose=not args.quiet, smoke=args.smoke)
    if evidence_pack_result != 0:
        print(f"  WARNING: Evidence pack generation failed (exit code {evidence_pack_result})")
    else:
        print(f"  ✓ Evidence pack written to: {out_dir / 'evidence_pack'}")
        # In smoke mode, also verify immediately
        if args.smoke:
            verify_result = _verify_evidence_pack(out_dir, verbose=not args.quiet)
            if verify_result != 0:
                print(f"  WARNING: Evidence pack verification failed (exit code {verify_result})")
            else:
                print(f"  ✓ Evidence pack verified")
    
    # Print next steps
    generated_suite = out_dir / "generated_suite" / "adversarial_regression_generated.yaml"
    
    print(f"\n[Next Steps]")
    print(f"  Generated suite path: {generated_suite}")
    print()
    print(f"  # Run suite:")
    print(f"  cargo run -p paraphina --bin sim_eval -- suite \\")
    print(f"    {generated_suite} \\")
    print(f"    --output-dir runs/adv_reg_cem_test --verbose")
    print()
    print(f"  # Verify evidence:")
    print(f"  cargo run -p paraphina --bin sim_eval -- verify-evidence-tree runs/adv_reg_cem_test")
    
    return 0 if topk else 1


if __name__ == "__main__":
    sys.exit(main())
