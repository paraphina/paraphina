#!/usr/bin/env python3
"""
scenario_library_v1.py

Scenario Library v1: Deterministic generation, manifest verification, and smoke testing.

Per ROADMAP.md Phase A requirements:
- Scenario library (seeded, reproducible)
- volatility regimes, spread/depth shocks, venue outages, funding inversions, basis spikes
- All generation must be deterministic given a seed
- Filenames must be stable
- No time-based randomness

CLI entrypoints:
    generate: Writes the scenario library to canonical folder and writes manifest
    check:    Recomputes hashes and validates against manifest (exit 0 OK, exit 2 mismatch)
    smoke:    Runs sim_eval on smoke suite, writes outputs, verifies evidence packs

Usage:
    python3 -m batch_runs.phase_a.scenario_library_v1 generate --seed 20260105
    python3 -m batch_runs.phase_a.scenario_library_v1 check
    python3 -m batch_runs.phase_a.scenario_library_v1 smoke --seed 12345 --out-dir runs/ci/scenario_library_smoke

Design choices:
- Uses stdlib only (no PyYAML) - scenarios are generated as static text
- Deterministic ordering and hashing
- Manifest includes schema_version and generated_by fields
- Evidence pack verification after smoke runs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ===========================================================================
# Module metadata
# ===========================================================================

__version__ = "1.0.0"
MODULE_PATH = "batch_runs.phase_a.scenario_library_v1"

# ===========================================================================
# Paths
# ===========================================================================

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_DIR = ROOT / "scenarios"
LIBRARY_DIR = SCENARIOS_DIR / "v1" / "scenario_library_v1"
SUITES_DIR = SCENARIOS_DIR / "suites"
SUITE_PATH = SUITES_DIR / "scenario_library_smoke_v1.yaml"
MANIFEST_PATH = LIBRARY_DIR / "manifest_sha256.json"

SIM_EVAL_BIN = ROOT / "target" / "release" / "sim_eval"

# ===========================================================================
# Scenario Definitions (deterministic, stable)
# ===========================================================================

# The library includes 10 scenarios covering different shock types:
# - Vol regime: low, medium, high (3)
# - Liquidity shock: spread widening (1)
# - Liquidity shock: depth thinning (1)
# - Venue outage: disabled venue window (1)
# - Funding inversion: sign flip (1)
# - Funding inversion: drift (1)
# - Basis spike: mid divergence positive (1)
# - Basis spike: mid divergence negative (1)

# Each scenario has a stable ID based on type and index

@dataclass
class ScenarioSpec:
    """Specification for a single scenario."""
    scenario_id: str
    category: str
    description: str
    base_seed: int
    num_seeds: int
    steps: int
    dt_seconds: float
    risk_profile: str
    init_q_tao: float
    process: str
    vol: float
    drift: float
    jump_intensity: float
    jump_sigma: float
    fees_bps_maker: float
    fees_bps_taker: float
    latency_ms: float
    # Optional overrides for shock types
    spread_shock_bps: Optional[float] = None
    depth_shock_pct: Optional[float] = None
    venue_outage_start: Optional[int] = None
    venue_outage_end: Optional[int] = None
    funding_sign_flip: bool = False
    funding_drift_bps: Optional[float] = None
    basis_spike_bps: Optional[float] = None
    expect_kill_switch: str = "allowed"

    def to_yaml_content(self) -> str:
        """Generate YAML content for this scenario."""
        lines = [
            f"# Scenario Library v1: {self.category}",
            f"# {self.description}",
            f"# Generated deterministically by {MODULE_PATH} v{__version__}",
            "",
            f"scenario_id: {self.scenario_id}",
            "scenario_version: 1",
            "",
            "engine: rl_sim_env",
            "",
            "horizon:",
            f"  steps: {self.steps}",
            f"  dt_seconds: {self.dt_seconds}",
            "",
            "rng:",
            f"  base_seed: {self.base_seed}",
            f"  num_seeds: {self.num_seeds}",
            "",
            "initial_state:",
            f"  risk_profile: {self.risk_profile}",
            f"  init_q_tao: {self.init_q_tao}",
            "",
            "market_model:",
            "  type: synthetic",
            "  synthetic:",
            f"    process: {self.process}",
            "    params:",
            f"      vol: {self.vol:.6f}",
            f"      drift: {self.drift}",
        ]

        # Add jump parameters only for jump_diffusion_stub
        if self.process == "jump_diffusion_stub":
            lines.extend([
                f"      jump_intensity: {self.jump_intensity:.6f}",
                f"      jump_sigma: {self.jump_sigma:.4f}",
            ])

        lines.extend([
            "",
            "microstructure_model:",
            f"  fees_bps_maker: {self.fees_bps_maker}",
            f"  fees_bps_taker: {self.fees_bps_taker}",
            f"  latency_ms: {self.latency_ms}",
        ])

        # Add shock-specific parameters
        if self.spread_shock_bps is not None:
            lines.extend([
                "",
                "# Liquidity shock: spread widening",
                f"spread_shock_bps: {self.spread_shock_bps}",
            ])

        if self.depth_shock_pct is not None:
            lines.extend([
                "",
                "# Liquidity shock: depth thinning",
                f"depth_shock_pct: {self.depth_shock_pct}",
            ])

        if self.venue_outage_start is not None:
            lines.extend([
                "",
                "# Venue outage event",
                "venue_outage:",
                "  venue_index: 0",
                f"  start_step: {self.venue_outage_start}",
                f"  end_step: {self.venue_outage_end}",
            ])

        if self.funding_sign_flip:
            lines.extend([
                "",
                "# Funding inversion: sign flip",
                "funding_inversion:",
                "  type: sign_flip",
                "  trigger_step: 500",
            ])

        if self.funding_drift_bps is not None:
            lines.extend([
                "",
                "# Funding inversion: gradual drift",
                "funding_drift:",
                f"  drift_bps_per_step: {self.funding_drift_bps}",
                "  start_step: 200",
            ])

        if self.basis_spike_bps is not None:
            lines.extend([
                "",
                "# Basis spike: mid divergence",
                "basis_spike:",
                f"  magnitude_bps: {self.basis_spike_bps}",
                "  venue_index: 1",
                "  trigger_step: 400",
                "  duration_steps: 200",
            ])

        lines.extend([
            "",
            "invariants:",
            f"  expect_kill_switch: {self.expect_kill_switch}",
            "  pnl_linearity_check: disabled",
        ])

        return "\n".join(lines) + "\n"


def get_scenario_library() -> List[ScenarioSpec]:
    """
    Return the canonical list of scenarios for the library.
    
    All parameters are hardcoded for determinism - no randomness.
    """
    scenarios = []

    # ===========================================================================
    # Vol regime scenarios (3)
    # ===========================================================================

    # 1. Low volatility regime
    scenarios.append(ScenarioSpec(
        scenario_id="slib_v1_vol_low",
        category="vol_regime",
        description="Low volatility regime (sigma=0.008)",
        base_seed=1000,
        num_seeds=3,
        steps=1500,
        dt_seconds=0.25,
        risk_profile="balanced",
        init_q_tao=0.0,
        process="gbm",
        vol=0.008,
        drift=0.0,
        jump_intensity=0.0,
        jump_sigma=0.0,
        fees_bps_maker=0.0,
        fees_bps_taker=0.5,
        latency_ms=2.0,
    ))

    # 2. Medium volatility regime
    scenarios.append(ScenarioSpec(
        scenario_id="slib_v1_vol_medium",
        category="vol_regime",
        description="Medium volatility regime (sigma=0.015)",
        base_seed=1001,
        num_seeds=3,
        steps=1500,
        dt_seconds=0.25,
        risk_profile="balanced",
        init_q_tao=0.0,
        process="gbm",
        vol=0.015,
        drift=0.0,
        jump_intensity=0.0,
        jump_sigma=0.0,
        fees_bps_maker=0.0,
        fees_bps_taker=0.5,
        latency_ms=2.0,
    ))

    # 3. High volatility regime
    scenarios.append(ScenarioSpec(
        scenario_id="slib_v1_vol_high",
        category="vol_regime",
        description="High volatility regime (sigma=0.035)",
        base_seed=1002,
        num_seeds=3,
        steps=1500,
        dt_seconds=0.25,
        risk_profile="balanced",
        init_q_tao=0.0,
        process="jump_diffusion_stub",
        vol=0.035,
        drift=0.0,
        jump_intensity=0.0003,
        jump_sigma=0.02,
        fees_bps_maker=0.0,
        fees_bps_taker=1.0,
        latency_ms=5.0,
    ))

    # ===========================================================================
    # Liquidity shock scenarios (2)
    # ===========================================================================

    # 4. Spread widening shock
    scenarios.append(ScenarioSpec(
        scenario_id="slib_v1_liq_spread",
        category="liquidity_shock",
        description="Spread widening shock (+50 bps)",
        base_seed=2000,
        num_seeds=3,
        steps=1500,
        dt_seconds=0.25,
        risk_profile="conservative",
        init_q_tao=10.0,
        process="gbm",
        vol=0.018,
        drift=0.0,
        jump_intensity=0.0,
        jump_sigma=0.0,
        fees_bps_maker=0.0,
        fees_bps_taker=1.5,
        latency_ms=10.0,
        spread_shock_bps=50.0,
    ))

    # 5. Depth thinning shock
    scenarios.append(ScenarioSpec(
        scenario_id="slib_v1_liq_depth",
        category="liquidity_shock",
        description="Depth thinning shock (-60% book depth)",
        base_seed=2001,
        num_seeds=3,
        steps=1500,
        dt_seconds=0.25,
        risk_profile="conservative",
        init_q_tao=-10.0,
        process="gbm",
        vol=0.018,
        drift=0.0,
        jump_intensity=0.0,
        jump_sigma=0.0,
        fees_bps_maker=0.0,
        fees_bps_taker=1.5,
        latency_ms=10.0,
        depth_shock_pct=-60.0,
    ))

    # ===========================================================================
    # Venue outage scenario (1)
    # ===========================================================================

    # 6. Venue outage (venue 0 disabled for 400 steps)
    scenarios.append(ScenarioSpec(
        scenario_id="slib_v1_venue_outage",
        category="venue_outage",
        description="Venue 0 disabled from step 300-700",
        base_seed=3000,
        num_seeds=3,
        steps=1500,
        dt_seconds=0.25,
        risk_profile="balanced",
        init_q_tao=5.0,
        process="gbm",
        vol=0.015,
        drift=0.0,
        jump_intensity=0.0,
        jump_sigma=0.0,
        fees_bps_maker=0.0,
        fees_bps_taker=0.8,
        latency_ms=5.0,
        venue_outage_start=300,
        venue_outage_end=700,
    ))

    # ===========================================================================
    # Funding inversion scenarios (2)
    # ===========================================================================

    # 7. Funding sign flip
    scenarios.append(ScenarioSpec(
        scenario_id="slib_v1_fund_flip",
        category="funding_inversion",
        description="Funding sign flip at step 500",
        base_seed=4000,
        num_seeds=3,
        steps=1500,
        dt_seconds=0.25,
        risk_profile="balanced",
        init_q_tao=0.0,
        process="gbm",
        vol=0.012,
        drift=0.0,
        jump_intensity=0.0,
        jump_sigma=0.0,
        fees_bps_maker=0.0,
        fees_bps_taker=0.5,
        latency_ms=3.0,
        funding_sign_flip=True,
    ))

    # 8. Funding drift
    scenarios.append(ScenarioSpec(
        scenario_id="slib_v1_fund_drift",
        category="funding_inversion",
        description="Gradual funding drift (+0.5 bps/step)",
        base_seed=4001,
        num_seeds=3,
        steps=1500,
        dt_seconds=0.25,
        risk_profile="balanced",
        init_q_tao=0.0,
        process="gbm",
        vol=0.012,
        drift=0.0,
        jump_intensity=0.0,
        jump_sigma=0.0,
        fees_bps_maker=0.0,
        fees_bps_taker=0.5,
        latency_ms=3.0,
        funding_drift_bps=0.5,
    ))

    # ===========================================================================
    # Basis spike scenarios (2)
    # ===========================================================================

    # 9. Positive basis spike
    scenarios.append(ScenarioSpec(
        scenario_id="slib_v1_basis_pos",
        category="basis_spike",
        description="Positive basis spike (+100 bps) on venue 1",
        base_seed=5000,
        num_seeds=3,
        steps=1500,
        dt_seconds=0.25,
        risk_profile="balanced",
        init_q_tao=0.0,
        process="gbm",
        vol=0.015,
        drift=0.0,
        jump_intensity=0.0,
        jump_sigma=0.0,
        fees_bps_maker=0.0,
        fees_bps_taker=0.8,
        latency_ms=5.0,
        basis_spike_bps=100.0,
    ))

    # 10. Negative basis spike
    scenarios.append(ScenarioSpec(
        scenario_id="slib_v1_basis_neg",
        category="basis_spike",
        description="Negative basis spike (-100 bps) on venue 1",
        base_seed=5001,
        num_seeds=3,
        steps=1500,
        dt_seconds=0.25,
        risk_profile="balanced",
        init_q_tao=0.0,
        process="gbm",
        vol=0.015,
        drift=0.0,
        jump_intensity=0.0,
        jump_sigma=0.0,
        fees_bps_maker=0.0,
        fees_bps_taker=0.8,
        latency_ms=5.0,
        basis_spike_bps=-100.0,
    ))

    return scenarios


# ===========================================================================
# Hashing utilities
# ===========================================================================

def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ===========================================================================
# Generation
# ===========================================================================

def generate_library(
    seed: int,
    out_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Generate the scenario library and manifest.
    
    Args:
        seed: RNG seed for deterministic generation (recorded in manifest for provenance)
        out_dir: Output directory (defaults to LIBRARY_DIR)
        verbose: Print progress messages
    
    Returns manifest data for verification.
    """
    target_dir = out_dir if out_dir is not None else LIBRARY_DIR
    manifest_path = target_dir / "manifest_sha256.json"
    
    # Ensure directory exists
    target_dir.mkdir(parents=True, exist_ok=True)
    
    scenarios = get_scenario_library()
    file_hashes: Dict[str, str] = {}
    
    if verbose:
        print(f"Generating {len(scenarios)} scenarios to {target_dir}")
        print(f"Using seed: {seed}")
    
    for spec in scenarios:
        filename = f"{spec.scenario_id}.yaml"
        filepath = target_dir / filename
        content = spec.to_yaml_content()
        
        # Write file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Compute hash
        file_hashes[filename] = compute_sha256(filepath)
        
        if verbose:
            print(f"  + {filename} [{spec.category}]")
    
    # Generate README
    readme_content = generate_readme(seed=seed)
    readme_path = target_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    file_hashes["README.md"] = compute_sha256(readme_path)
    
    if verbose:
        print(f"  + README.md")
    
    # Generate manifest (deterministic ordering by key)
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "generated_by": {
            "module": MODULE_PATH,
            "version": __version__,
        },
        "scenario_count": len(scenarios),
        "files": {k: file_hashes[k] for k in sorted(file_hashes.keys())},
    }
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    
    if verbose:
        print(f"  + manifest_sha256.json ({len(file_hashes)} files)")
        print(f"Library generated successfully.")
    
    return manifest


def generate_readme(seed: int) -> str:
    """Generate README.md content for the scenario library."""
    scenarios = get_scenario_library()
    
    # Group by category
    by_category: Dict[str, List[ScenarioSpec]] = {}
    for s in scenarios:
        by_category.setdefault(s.category, []).append(s)
    
    lines = [
        "# Scenario Library v1",
        "",
        "Canonical scenario library for sim_eval testing and CI verification.",
        "",
        f"**Generated by:** `{MODULE_PATH}` v{__version__}",
        f"**Seed:** {seed}",
        "",
        "## Categories",
        "",
    ]
    
    for category in sorted(by_category.keys()):
        cat_scenarios = by_category[category]
        lines.append(f"### {category.replace('_', ' ').title()}")
        lines.append("")
        for s in cat_scenarios:
            lines.append(f"- **{s.scenario_id}**: {s.description}")
        lines.append("")
    
    lines.extend([
        "## Usage",
        "",
        "Run scenarios via sim_eval:",
        "",
        "```bash",
        "cargo run --release -p paraphina --bin sim_eval -- suite scenarios/suites/scenario_library_smoke_v1.yaml",
        "```",
        "",
        "## Verification",
        "",
        "Verify manifest integrity:",
        "",
        "```bash",
        "python3 -m batch_runs.phase_a.scenario_library_v1 check",
        "```",
        "",
        "## Regeneration",
        "",
        "To regenerate scenarios (should produce identical output with same seed):",
        "",
        "```bash",
        f"python3 -m batch_runs.phase_a.scenario_library_v1 generate --seed {seed}",
        "```",
        "",
    ])
    
    return "\n".join(lines)


def generate_suite(verbose: bool = True) -> None:
    """Generate the smoke suite YAML (static, no PyYAML dependency)."""
    scenarios = get_scenario_library()
    
    # Select a small subset for smoke testing (fast CI)
    # Include one from each category
    smoke_scenarios = [
        "slib_v1_vol_medium",      # vol_regime
        "slib_v1_liq_spread",      # liquidity_shock
        "slib_v1_venue_outage",    # venue_outage
        "slib_v1_fund_flip",       # funding_inversion
        "slib_v1_basis_pos",       # basis_spike
    ]
    
    lines = [
        "# Scenario Library v1 Smoke Suite",
        f"# Generated by {MODULE_PATH} v{__version__}",
        "# Fast CI-friendly subset (5 scenarios)",
        "",
        "suite_id: scenario_library_smoke_v1",
        "suite_version: 1",
        "",
        "# Minimal repeat for speed",
        "repeat_runs: 1",
        "",
        "# Output directory",
        "out_dir: runs/ci/scenario_library_smoke",
        "",
        "# Scenario paths",
        "scenarios:",
    ]
    
    for scenario_id in smoke_scenarios:
        lines.append(f"  - path: scenarios/v1/scenario_library_v1/{scenario_id}.yaml")
    
    content = "\n".join(lines) + "\n"
    
    SUITES_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUITE_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    
    if verbose:
        print(f"Suite generated: {SUITE_PATH}")


# ===========================================================================
# Verification
# ===========================================================================

def check_manifest(verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    Verify manifest against actual files.
    
    Returns:
        (success, list of error messages)
    """
    errors: List[str] = []
    
    if not MANIFEST_PATH.exists():
        errors.append(f"Manifest not found: {MANIFEST_PATH}")
        return False, errors
    
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in manifest: {e}")
        return False, errors
    
    # Check schema version
    if manifest.get("schema_version") != 1:
        errors.append(f"Unknown schema_version: {manifest.get('schema_version')}")
    
    # Check all files
    expected_files = manifest.get("files", {})
    
    for filename, expected_hash in sorted(expected_files.items()):
        filepath = LIBRARY_DIR / filename
        
        if not filepath.exists():
            errors.append(f"Missing file: {filename}")
            continue
        
        actual_hash = compute_sha256(filepath)
        if actual_hash != expected_hash:
            errors.append(
                f"Hash mismatch for {filename}: "
                f"expected {expected_hash[:16]}..., "
                f"got {actual_hash[:16]}..."
            )
    
    # Check for extra files
    actual_files = set()
    for p in LIBRARY_DIR.iterdir():
        if p.is_file() and p.name != "manifest_sha256.json":
            actual_files.add(p.name)
    
    manifest_files = set(expected_files.keys())
    extra = actual_files - manifest_files
    if extra:
        errors.append(f"Extra files not in manifest: {sorted(extra)}")
    
    if verbose:
        if errors:
            print(f"Manifest check FAILED ({len(errors)} errors):")
            for e in errors:
                print(f"  - {e}")
        else:
            print(f"Manifest check PASSED ({len(expected_files)} files verified)")
    
    return len(errors) == 0, errors


def check_suite_references(verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    Verify that suite file references exist.
    
    Returns:
        (success, list of error messages)
    """
    errors: List[str] = []
    
    if not SUITE_PATH.exists():
        errors.append(f"Suite file not found: {SUITE_PATH}")
        return False, errors
    
    # Parse YAML manually (simple line-based parsing, no PyYAML)
    with open(SUITE_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find scenario paths
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("- path:"):
            path_str = line.split(":", 1)[1].strip()
            full_path = ROOT / path_str
            if not full_path.exists():
                errors.append(f"Referenced scenario not found: {path_str}")
    
    if verbose:
        if errors:
            print(f"Suite reference check FAILED ({len(errors)} errors):")
            for e in errors:
                print(f"  - {e}")
        else:
            print("Suite reference check PASSED")
    
    return len(errors) == 0, errors


# ===========================================================================
# Smoke testing
# ===========================================================================

def find_sim_eval() -> Path:
    """Find sim_eval binary, building if needed."""
    if SIM_EVAL_BIN.exists():
        return SIM_EVAL_BIN
    
    # Try debug build
    debug_bin = ROOT / "target" / "debug" / "sim_eval"
    if debug_bin.exists():
        return debug_bin
    
    # Build release binary
    print("Building sim_eval (release)...")
    result = subprocess.run(
        ["cargo", "build", "--release", "-p", "paraphina", "--bin", "sim_eval"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        raise RuntimeError("Failed to build sim_eval")
    
    if not SIM_EVAL_BIN.exists():
        raise RuntimeError(f"sim_eval not found after build: {SIM_EVAL_BIN}")
    
    return SIM_EVAL_BIN


def run_smoke(
    seed: int,
    out_dir: Path,
    verbose: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Run smoke suite and verify evidence packs.
    
    Returns:
        (success, list of error messages)
    """
    errors: List[str] = []
    
    # Ensure suite exists
    if not SUITE_PATH.exists():
        print("Suite not found, generating...")
        generate_suite(verbose=verbose)
    
    # Find or build sim_eval
    sim_eval = find_sim_eval()
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Running smoke suite with seed={seed}")
        print(f"Output directory: {out_dir}")
    
    # Run suite
    cmd = [
        str(sim_eval),
        "suite",
        str(SUITE_PATH),
        "--output-dir", str(out_dir),
        "--verbose",
    ]
    
    if verbose:
        print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PARAPHINA_RNG_SEED": str(seed)},
    )
    
    if verbose and result.stdout:
        print(result.stdout)
    
    if result.returncode != 0:
        errors.append(f"sim_eval suite failed with exit code {result.returncode}")
        if result.stderr:
            errors.append(f"stderr: {result.stderr}")
        return False, errors
    
    if verbose:
        print("Suite run completed successfully")
    
    # Verify evidence packs
    if verbose:
        print("Verifying evidence packs...")
    
    verify_cmd = [
        str(sim_eval),
        "verify-evidence-tree",
        str(out_dir),
    ]
    
    verify_result = subprocess.run(
        verify_cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    
    if verbose and verify_result.stdout:
        print(verify_result.stdout)
    
    if verify_result.returncode != 0:
        errors.append(f"Evidence verification failed with exit code {verify_result.returncode}")
        if verify_result.stderr:
            errors.append(f"stderr: {verify_result.stderr}")
        return False, errors
    
    if verbose:
        print("Evidence verification PASSED")
    
    # Write smoke run summary
    summary = {
        "seed": seed,
        "suite_path": str(SUITE_PATH),
        "output_dir": str(out_dir),
        "sim_eval": str(sim_eval),
        "success": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    
    summary_path = out_dir / "smoke_run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    
    if verbose:
        print(f"Summary written to: {summary_path}")
    
    return True, []


# ===========================================================================
# CLI
# ===========================================================================

def cmd_generate(args: argparse.Namespace) -> int:
    """Generate command handler."""
    print("=" * 60)
    print("Scenario Library v1 Generator")
    print("=" * 60)
    
    generate_library(seed=args.seed, verbose=True)
    generate_suite(verbose=True)
    
    print()
    print("Verifying generated content...")
    success, errors = check_manifest(verbose=True)
    
    if not success:
        return 1
    
    success, errors = check_suite_references(verbose=True)
    if not success:
        return 1
    
    print()
    print("Generation complete!")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Check command handler."""
    success, errors = check_manifest(verbose=True)
    
    if not success:
        return 2  # Exit code 2 for mismatch as specified
    
    success, errors = check_suite_references(verbose=True)
    if not success:
        return 2
    
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Smoke test command handler."""
    print("=" * 60)
    print("Scenario Library v1 Smoke Test")
    print("=" * 60)
    
    out_dir = Path(args.out_dir).resolve()
    success, errors = run_smoke(
        seed=args.seed,
        out_dir=out_dir,
        verbose=True,
    )
    
    if not success:
        print("\nSmoke test FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    
    print("\nSmoke test PASSED!")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="scenario_library_v1",
        description="Scenario Library v1: Deterministic generation and verification",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate scenario library and manifest",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="RNG seed for deterministic generation (required for reproducibility)",
    )
    gen_parser.set_defaults(func=cmd_generate)
    
    # Check command
    check_parser = subparsers.add_parser(
        "check",
        help="Verify manifest hashes (exit 0 OK, exit 2 mismatch)",
    )
    check_parser.set_defaults(func=cmd_check)
    
    # Smoke command
    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run smoke suite and verify evidence packs",
    )
    smoke_parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="RNG seed for deterministic execution (default: 12345)",
    )
    smoke_parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/ci/scenario_library_smoke",
        help="Output directory for smoke run artifacts",
    )
    smoke_parser.set_defaults(func=cmd_smoke)
    
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

