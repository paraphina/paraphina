#!/usr/bin/env python3
"""
batch_runs/phase_a/mc_scale.py

Monte Carlo at Scale harness for Phase A.

This module provides deterministic sharded Monte Carlo execution with
aggregation and evidence pack verification.

Subcommands:
  plan        - Generate mc_scale_plan.json with shard ranges
  run-shard   - Execute a single shard using the monte_carlo binary
  aggregate   - Concatenate shard JSONL files and produce mc_summary.json
  smoke       - Run all steps sequentially for CI validation

Usage:
  python3 -m batch_runs.phase_a.mc_scale plan --out-dir <DIR> --seed 42 --runs 1000 --shards 10 --ticks 600
  python3 -m batch_runs.phase_a.mc_scale run-shard --plan <PATH> --shard-index 0
  python3 -m batch_runs.phase_a.mc_scale aggregate --plan <PATH>
  python3 -m batch_runs.phase_a.mc_scale smoke --seed 12345 --runs 12 --shards 3 --ticks 50 --out-dir runs/ci/mc_scale_smoke

Deterministic seed contract:
  For global run index i, seed_i = base_seed + i (u64 wrap).
  Each shard runs indices [start, end) where the ranges are contiguous and non-overlapping.

No third-party dependencies - stdlib only.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Constants
# =============================================================================

PLAN_SCHEMA_VERSION = 1
DEFAULT_SEED = 12345
DEFAULT_RUNS = 12
DEFAULT_SHARDS = 3
DEFAULT_TICKS = 50


# =============================================================================
# Utility functions
# =============================================================================

def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def find_monte_carlo_binary() -> Path:
    """Find the monte_carlo binary in the expected locations."""
    # Check common release paths
    candidates = [
        Path("target/release/monte_carlo"),
        Path("./target/release/monte_carlo"),
        Path("../target/release/monte_carlo"),
    ]
    
    # Also check PATH
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    
    # Check if it's in PATH
    result = shutil.which("monte_carlo")
    if result:
        return Path(result)
    
    raise FileNotFoundError(
        "monte_carlo binary not found. Build with: cargo build --release -p paraphina --bin monte_carlo"
    )


def find_sim_eval_binary() -> Path:
    """Find the sim_eval binary for evidence verification."""
    candidates = [
        Path("target/release/sim_eval"),
        Path("./target/release/sim_eval"),
        Path("../target/release/sim_eval"),
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    
    result = shutil.which("sim_eval")
    if result:
        return Path(result)
    
    raise FileNotFoundError(
        "sim_eval binary not found. Build with: cargo build --release -p paraphina --bin sim_eval"
    )


def json_dumps_deterministic(obj: Any) -> str:
    """Serialize JSON with deterministic key ordering and formatting."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)


# =============================================================================
# Plan command
# =============================================================================

def compute_shard_ranges(runs: int, shards: int) -> List[Dict[str, int]]:
    """
    Compute contiguous shard ranges.
    
    Returns list of dicts with 'start' and 'end' (exclusive) for each shard.
    """
    if shards <= 0:
        raise ValueError("shards must be >= 1")
    if runs <= 0:
        raise ValueError("runs must be >= 1")
    if shards > runs:
        # More shards than runs - reduce shard count
        shards = runs
    
    base_count = runs // shards
    remainder = runs % shards
    
    ranges = []
    current_start = 0
    
    for i in range(shards):
        # First 'remainder' shards get an extra run
        count = base_count + (1 if i < remainder else 0)
        ranges.append({
            "start": current_start,
            "end": current_start + count,
        })
        current_start += count
    
    return ranges


def cmd_plan(args: argparse.Namespace) -> int:
    """Generate mc_scale_plan.json."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    seed = args.seed
    runs = args.runs
    shards = args.shards
    ticks = args.ticks
    
    # Compute shard ranges
    shard_ranges = compute_shard_ranges(runs, shards)
    
    # Build plan
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "seed": seed,
        "runs": runs,
        "shards": len(shard_ranges),  # May be less than requested if runs < shards
        "ticks": ticks,
        "out_dir": str(out_dir.resolve()),
        "shard_ranges": shard_ranges,
    }
    
    # Write plan (deterministic)
    plan_path = out_dir / "mc_scale_plan.json"
    plan_json = json_dumps_deterministic(plan)
    with open(plan_path, "w") as f:
        f.write(plan_json)
        f.write("\n")  # Trailing newline for POSIX compliance
    
    print(f"✓ Plan written to: {plan_path}")
    print(f"  seed={seed} runs={runs} shards={len(shard_ranges)} ticks={ticks}")
    for i, rng in enumerate(shard_ranges):
        print(f"  shard {i}: run indices [{rng['start']}, {rng['end']})")
    
    return 0


# =============================================================================
# Run-shard command
# =============================================================================

def cmd_run_shard(args: argparse.Namespace) -> int:
    """Execute a single shard using the monte_carlo binary."""
    plan_path = Path(args.plan)
    shard_index = args.shard_index
    
    # Load plan
    with open(plan_path) as f:
        plan = json.load(f)
    
    if shard_index < 0 or shard_index >= len(plan["shard_ranges"]):
        print(f"Error: shard_index {shard_index} out of range [0, {len(plan['shard_ranges'])})")
        return 1
    
    shard = plan["shard_ranges"][shard_index]
    start_idx = shard["start"]
    end_idx = shard["end"]
    run_count = end_idx - start_idx
    
    out_dir = Path(plan["out_dir"])
    shard_dir = out_dir / "shards" / f"shard_{shard_index}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    
    # Find binary
    try:
        mc_binary = find_monte_carlo_binary()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Build command
    cmd = [
        str(mc_binary),
        "--runs", str(plan["runs"]),
        "--run-start-index", str(start_idx),
        "--run-count", str(run_count),
        "--seed", str(plan["seed"]),
        "--ticks", str(plan["ticks"]),
        "--output-dir", str(shard_dir),
        "--quiet",
    ]
    
    print(f"Running shard {shard_index}: indices [{start_idx}, {end_idx})")
    print(f"  Command: {' '.join(cmd)}")
    
    # Execute
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Error: monte_carlo failed with exit code {result.returncode}")
        return result.returncode
    
    # Verify mc_runs.jsonl exists
    jsonl_path = shard_dir / "mc_runs.jsonl"
    if not jsonl_path.exists():
        print(f"Error: Expected output {jsonl_path} not found")
        return 1
    
    # Write shard-level evidence pack (already done by monte_carlo)
    evidence_pack_dir = shard_dir / "evidence_pack"
    if evidence_pack_dir.exists():
        print(f"✓ Shard {shard_index} evidence pack: {evidence_pack_dir}")
    else:
        print(f"Warning: No evidence pack at {evidence_pack_dir}")
    
    print(f"✓ Shard {shard_index} complete: {jsonl_path}")
    return 0


# =============================================================================
# Aggregate command
# =============================================================================

def cmd_aggregate(args: argparse.Namespace) -> int:
    """Concatenate shard JSONL files and produce mc_summary.json."""
    plan_path = Path(args.plan)
    
    # Load plan
    with open(plan_path) as f:
        plan = json.load(f)
    
    out_dir = Path(plan["out_dir"])
    shards_dir = out_dir / "shards"
    
    # Collect all JSONL records
    all_records: List[Dict[str, Any]] = []
    shard_hashes: Dict[str, str] = {}
    
    for i, shard in enumerate(plan["shard_ranges"]):
        shard_dir = shards_dir / f"shard_{i}"
        jsonl_path = shard_dir / "mc_runs.jsonl"
        
        if not jsonl_path.exists():
            print(f"Error: Missing shard JSONL: {jsonl_path}")
            return 1
        
        # Compute hash for manifest
        shard_hashes[str(shard_dir)] = sha256_file(jsonl_path)
        
        # Read records
        with open(jsonl_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    all_records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Error: Malformed JSON at {jsonl_path}:{line_num}: {e}")
                    return 1
    
    # Sort by run_index
    all_records.sort(key=lambda r: r["run_index"])
    
    # Validate
    expected_runs = plan["runs"]
    if len(all_records) != expected_runs:
        print(f"Error: Expected {expected_runs} records, found {len(all_records)}")
        return 1
    
    # Check indices are [0, runs)
    indices = [r["run_index"] for r in all_records]
    expected_indices = list(range(expected_runs))
    if indices != expected_indices:
        # Find first mismatch
        for i, (actual, expected) in enumerate(zip(indices, expected_indices)):
            if actual != expected:
                print(f"Error: Index mismatch at position {i}: got {actual}, expected {expected}")
                return 1
        if len(indices) != len(expected_indices):
            print(f"Error: Index count mismatch: {len(indices)} vs {len(expected_indices)}")
            return 1
    
    # Check for duplicates
    seen: set = set()
    for record in all_records:
        idx = record["run_index"]
        if idx in seen:
            print(f"Error: Duplicate run_index {idx}")
            return 1
        seen.add(idx)
    
    print(f"✓ Validated {len(all_records)} records with indices [0, {expected_runs})")
    
    # Write aggregated JSONL
    aggregated_jsonl_path = out_dir / "mc_runs.jsonl"
    with open(aggregated_jsonl_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")
    
    aggregated_jsonl_hash = sha256_file(aggregated_jsonl_path)
    print(f"✓ Aggregated JSONL: {aggregated_jsonl_path}")
    
    # Call monte_carlo summarize
    try:
        mc_binary = find_monte_carlo_binary()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    cmd = [
        str(mc_binary),
        "summarize",
        "--input", str(aggregated_jsonl_path),
        "--out-dir", str(out_dir),
        "--base-seed", str(plan["seed"]),
    ]
    
    print(f"Running summarize: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Error: summarize failed with exit code {result.returncode}")
        return result.returncode
    
    # Verify mc_summary.json exists
    summary_path = out_dir / "mc_summary.json"
    if not summary_path.exists():
        print(f"Error: Expected output {summary_path} not found")
        return 1
    
    summary_hash = sha256_file(summary_path)
    print(f"✓ Summary: {summary_path}")
    
    # Write mc_scale_manifest.json (root level, before evidence pack generation)
    plan_hash = sha256_file(plan_path)
    mc_scale_manifest = {
        "plan_hash": f"sha256:{plan_hash}",
        "shard_directories": {str(k): f"sha256:{v}" for k, v in shard_hashes.items()},
        "aggregated_mc_runs_jsonl_hash": f"sha256:{aggregated_jsonl_hash}",
        "mc_summary_json_hash": f"sha256:{summary_hash}",
    }
    
    mc_scale_manifest_path = out_dir / "mc_scale_manifest.json"
    with open(mc_scale_manifest_path, "w") as f:
        f.write(json_dumps_deterministic(mc_scale_manifest))
        f.write("\n")
    
    print(f"✓ MC scale manifest: {mc_scale_manifest_path}")
    
    # Generate root evidence pack using sim_eval (canonical format)
    try:
        sim_eval_binary = find_sim_eval_binary()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Use sim_eval write-evidence-pack to generate canonical evidence pack
    write_ep_cmd = [str(sim_eval_binary), "write-evidence-pack", str(out_dir)]
    print(f"Generating evidence pack: {' '.join(write_ep_cmd)}")
    write_result = subprocess.run(write_ep_cmd, capture_output=True, text=True)
    if write_result.returncode != 0:
        print(f"Error: write-evidence-pack failed with exit code {write_result.returncode}")
        if write_result.stderr:
            print(f"  stderr: {write_result.stderr}")
        return 1
    
    evidence_pack_dir = out_dir / "evidence_pack"
    print(f"✓ Evidence pack: {evidence_pack_dir}")
    
    # Verify evidence pack using sim_eval (FATAL if fails)
    verify_cmd = [str(sim_eval_binary), "verify-evidence-pack", str(out_dir)]
    print(f"Verifying evidence pack: {' '.join(verify_cmd)}")
    verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
    if verify_result.returncode != 0:
        print(f"Error: Evidence pack verification failed with exit code {verify_result.returncode}")
        if verify_result.stderr:
            print(f"  stderr: {verify_result.stderr}")
        return 1
    
    print(f"✓ Evidence pack verified")
    
    return 0


# =============================================================================
# Programmatic API for pipeline integration
# =============================================================================

def run_mc_scale(
    out_dir: str,
    seed: int,
    runs: int,
    shards: int,
    ticks: int,
    monte_carlo_bin: Optional[str] = None,
    sim_eval_bin: Optional[str] = None,
    quiet: bool = True,
) -> int:
    """
    Run the full mc_scale pipeline programmatically.
    
    This function is designed to be called from promote_pipeline.py
    to run sharded Monte Carlo within a trial's MC directory.
    
    Args:
        out_dir: Output directory for mc_scale outputs
        seed: Base seed for Monte Carlo runs
        runs: Total number of Monte Carlo runs
        shards: Number of shards to split runs into
        ticks: Number of ticks per run
        monte_carlo_bin: Optional path to monte_carlo binary (auto-detected if None)
        sim_eval_bin: Optional path to sim_eval binary (auto-detected if None)
        quiet: If True, suppress output (default: True for pipeline integration)
    
    Returns:
        0 on success, non-zero on failure
    
    Outputs written to out_dir:
        - mc_scale_plan.json: Plan with shard ranges
        - shards/shard_N/mc_runs.jsonl: Per-shard JSONL files
        - mc_runs.jsonl: Aggregated JSONL
        - mc_summary.json: Summary produced by monte_carlo summarize
        - mc_scale_manifest.json: Manifest with hashes
        - evidence_pack/: Evidence pack directory
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Plan
    if not quiet:
        print(f"mc_scale: planning {runs} runs across {shards} shards")
    
    plan_args = argparse.Namespace(
        out_dir=str(out_path),
        seed=seed,
        runs=runs,
        shards=shards,
        ticks=ticks,
    )
    
    # Suppress plan output in quiet mode
    import io
    import contextlib
    
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            ret = cmd_plan(plan_args)
    else:
        ret = cmd_plan(plan_args)
    
    if ret != 0:
        return ret
    
    # Step 2: Run all shards
    plan_path = out_path / "mc_scale_plan.json"
    with open(plan_path) as f:
        plan = json.load(f)
    
    actual_shards = len(plan["shard_ranges"])
    for i in range(actual_shards):
        if not quiet:
            print(f"mc_scale: running shard {i+1}/{actual_shards}")
        
        shard_args = argparse.Namespace(
            plan=str(plan_path),
            shard_index=i,
        )
        
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                ret = cmd_run_shard(shard_args)
        else:
            ret = cmd_run_shard(shard_args)
        
        if ret != 0:
            return ret
    
    # Step 3: Aggregate
    if not quiet:
        print("mc_scale: aggregating results")
    
    agg_args = argparse.Namespace(plan=str(plan_path))
    
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            ret = cmd_aggregate(agg_args)
    else:
        ret = cmd_aggregate(agg_args)
    
    return ret


# =============================================================================
# Smoke command
# =============================================================================

def cmd_smoke(args: argparse.Namespace) -> int:
    """Run plan -> all shards -> aggregate sequentially."""
    out_dir = Path(args.out_dir)
    seed = args.seed
    runs = args.runs
    shards = args.shards
    ticks = args.ticks
    
    print("=" * 60)
    print("MC SCALE SMOKE TEST")
    print("=" * 60)
    print(f"  out_dir: {out_dir}")
    print(f"  seed: {seed}")
    print(f"  runs: {runs}")
    print(f"  shards: {shards}")
    print(f"  ticks: {ticks}")
    print()
    
    # Step 1: Plan
    print("STEP 1/3: Plan")
    print("-" * 40)
    plan_args = argparse.Namespace(
        out_dir=str(out_dir),
        seed=seed,
        runs=runs,
        shards=shards,
        ticks=ticks,
    )
    ret = cmd_plan(plan_args)
    if ret != 0:
        print(f"FAIL: Plan failed with code {ret}")
        return ret
    print()
    
    # Step 2: Run all shards sequentially
    print("STEP 2/3: Run shards")
    print("-" * 40)
    plan_path = out_dir / "mc_scale_plan.json"
    
    with open(plan_path) as f:
        plan = json.load(f)
    
    actual_shards = len(plan["shard_ranges"])
    for i in range(actual_shards):
        print(f"\nShard {i}/{actual_shards}:")
        shard_args = argparse.Namespace(
            plan=str(plan_path),
            shard_index=i,
        )
        ret = cmd_run_shard(shard_args)
        if ret != 0:
            print(f"FAIL: Shard {i} failed with code {ret}")
            return ret
    print()
    
    # Step 3: Aggregate
    print("STEP 3/3: Aggregate")
    print("-" * 40)
    agg_args = argparse.Namespace(plan=str(plan_path))
    ret = cmd_aggregate(agg_args)
    if ret != 0:
        print(f"FAIL: Aggregate failed with code {ret}")
        return ret
    print()
    
    # Final verification
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # Check required files exist
    required_files = [
        out_dir / "mc_scale_plan.json",
        out_dir / "mc_runs.jsonl",
        out_dir / "mc_summary.json",
        out_dir / "mc_scale_manifest.json",
        out_dir / "evidence_pack" / "SHA256SUMS",
        out_dir / "evidence_pack" / "manifest.json",
    ]
    
    all_exist = True
    for path in required_files:
        if path.exists():
            print(f"✓ {path.relative_to(out_dir)}")
        else:
            print(f"✗ MISSING: {path.relative_to(out_dir)}")
            all_exist = False
    
    # Check mc_summary.json can be parsed
    summary_path = out_dir / "mc_summary.json"
    try:
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"✓ mc_summary.json parsed successfully")
        print(f"  runs: {len(summary.get('runs', []))}")
        print(f"  kill_rate: {summary.get('aggregate', {}).get('kill_rate', 'N/A')}")
    except Exception as e:
        print(f"✗ Failed to parse mc_summary.json: {e}")
        all_exist = False
    
    print()
    if all_exist:
        print("=" * 60)
        print("PASS")
        print("=" * 60)
        print(f"Output directory: {out_dir}")
        return 0
    else:
        print("=" * 60)
        print("FAIL")
        print("=" * 60)
        return 1


# =============================================================================
# Main entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo at Scale harness for Phase A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m batch_runs.phase_a.mc_scale plan --out-dir runs/mc_scale --seed 42 --runs 1000 --shards 10 --ticks 600
  python3 -m batch_runs.phase_a.mc_scale run-shard --plan runs/mc_scale/mc_scale_plan.json --shard-index 0
  python3 -m batch_runs.phase_a.mc_scale aggregate --plan runs/mc_scale/mc_scale_plan.json
  python3 -m batch_runs.phase_a.mc_scale smoke --seed 12345 --runs 12 --shards 3 --ticks 50 --out-dir runs/ci/mc_scale_smoke
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Plan subcommand
    plan_parser = subparsers.add_parser("plan", help="Generate mc_scale_plan.json")
    plan_parser.add_argument("--out-dir", required=True, help="Output directory")
    plan_parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Base seed (default: {DEFAULT_SEED})")
    plan_parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help=f"Total runs (default: {DEFAULT_RUNS})")
    plan_parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS, help=f"Number of shards (default: {DEFAULT_SHARDS})")
    plan_parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS, help=f"Ticks per run (default: {DEFAULT_TICKS})")
    
    # Run-shard subcommand
    shard_parser = subparsers.add_parser("run-shard", help="Execute a single shard")
    shard_parser.add_argument("--plan", required=True, help="Path to mc_scale_plan.json")
    shard_parser.add_argument("--shard-index", type=int, required=True, help="Shard index (0-based)")
    
    # Aggregate subcommand
    agg_parser = subparsers.add_parser("aggregate", help="Aggregate shard results")
    agg_parser.add_argument("--plan", required=True, help="Path to mc_scale_plan.json")
    
    # Smoke subcommand
    smoke_parser = subparsers.add_parser("smoke", help="Run full smoke test (plan + shards + aggregate)")
    smoke_parser.add_argument("--out-dir", required=True, help="Output directory")
    smoke_parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Base seed (default: {DEFAULT_SEED})")
    smoke_parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help=f"Total runs (default: {DEFAULT_RUNS})")
    smoke_parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS, help=f"Number of shards (default: {DEFAULT_SHARDS})")
    smoke_parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS, help=f"Ticks per run (default: {DEFAULT_TICKS})")
    
    args = parser.parse_args()
    
    if args.command == "plan":
        sys.exit(cmd_plan(args))
    elif args.command == "run-shard":
        sys.exit(cmd_run_shard(args))
    elif args.command == "aggregate":
        sys.exit(cmd_aggregate(args))
    elif args.command == "smoke":
        sys.exit(cmd_smoke(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

