#!/usr/bin/env python3
"""
cli.py

CLI for Phase AB: Integrated Phase A → Phase B Orchestrator.

Usage:
    # Run mode (default)
    python3 -m batch_runs.phase_ab.cli run \\
        --candidate-run runs/phaseA_candidate \\
        --baseline-run runs/phaseA_baseline \\
        --out-dir runs/phaseAB_output

    # Smoke test mode (CI-friendly)
    python3 -m batch_runs.phase_ab.cli smoke

    # Smoke with auto-generate (creates Phase A runs if needed)
    python3 -m batch_runs.phase_ab.cli smoke --auto-generate-phasea

Exit codes (institutional CI semantics):
    0 = PROMOTE or HOLD (pipeline succeeded)
    2 = REJECT (candidate fails guardrails)
    3 = ERROR (runtime/IO/parsing failure)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from batch_runs.phase_ab.pipeline import (
    run_phase_ab,
    ROOT,
)


# =============================================================================
# Default Paths for Smoke Mode
# =============================================================================

DEFAULT_SMOKE_OUT_DIR = ROOT / "runs" / "phaseAB_smoke"
DEFAULT_CANDIDATE_SMOKE = ROOT / "runs" / "phaseA_candidate_smoke"
DEFAULT_BASELINE_SMOKE = ROOT / "runs" / "phaseA_baseline_smoke"


# =============================================================================
# Smoke Mode Helpers
# =============================================================================

def _check_phase_a_runs_exist() -> tuple[bool, bool]:
    """
    Check if Phase A smoke runs exist.
    
    Returns:
        (candidate_exists, baseline_exists) tuple
    """
    candidate_trials = DEFAULT_CANDIDATE_SMOKE / "trials.jsonl"
    baseline_trials = DEFAULT_BASELINE_SMOKE / "trials.jsonl"
    
    return (candidate_trials.exists(), baseline_trials.exists())


def _print_generate_commands() -> None:
    """Print commands to generate Phase A smoke runs."""
    print()
    print("Phase A smoke runs not found. Generate them with:")
    print()
    print("  # Generate candidate smoke run")
    print(f"  python3 -m batch_runs.phase_a.promote_pipeline \\")
    print(f"    --smoke \\")
    print(f"    --study-dir {DEFAULT_CANDIDATE_SMOKE} \\")
    print(f"    --seed 42")
    print()
    print("  # Generate baseline smoke run")
    print(f"  python3 -m batch_runs.phase_a.promote_pipeline \\")
    print(f"    --smoke \\")
    print(f"    --study-dir {DEFAULT_BASELINE_SMOKE} \\")
    print(f"    --seed 43")
    print()
    print("Then re-run PhaseAB smoke:")
    print()
    print("  python3 -m batch_runs.phase_ab.cli smoke")
    print()


def _generate_phase_a_runs(verbose: bool = True) -> bool:
    """
    Generate Phase A smoke runs.
    
    Returns:
        True if generation succeeded, False otherwise
    """
    import subprocess
    
    if verbose:
        print("[PhaseAB Smoke] Generating Phase A smoke runs...")
    
    # Generate candidate
    candidate_exists, baseline_exists = _check_phase_a_runs_exist()
    
    if not candidate_exists:
        if verbose:
            print(f"\n  Generating candidate: {DEFAULT_CANDIDATE_SMOKE}")
        
        cmd = [
            sys.executable, "-m", "batch_runs.phase_a.promote_pipeline",
            "--smoke",
            "--study-dir", str(DEFAULT_CANDIDATE_SMOKE),
            "--seed", "42",
        ]
        
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"ERROR: Failed to generate candidate smoke run", file=sys.stderr)
            return False
    else:
        if verbose:
            print(f"  Candidate already exists: {DEFAULT_CANDIDATE_SMOKE}")
    
    # Generate baseline
    if not baseline_exists:
        if verbose:
            print(f"\n  Generating baseline: {DEFAULT_BASELINE_SMOKE}")
        
        cmd = [
            sys.executable, "-m", "batch_runs.phase_a.promote_pipeline",
            "--smoke",
            "--study-dir", str(DEFAULT_BASELINE_SMOKE),
            "--seed", "43",
        ]
        
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"ERROR: Failed to generate baseline smoke run", file=sys.stderr)
            return False
    else:
        if verbose:
            print(f"  Baseline already exists: {DEFAULT_BASELINE_SMOKE}")
    
    return True


# =============================================================================
# Command: run
# =============================================================================

def cmd_run(args: argparse.Namespace) -> int:
    """Run PhaseAB with explicit candidate/baseline paths."""
    result = run_phase_ab(
        candidate_run=args.candidate_run,
        baseline_run=args.baseline_run,
        out_dir=args.out_dir,
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        skip_evidence_verify=args.skip_evidence_verify,
        verbose=not args.quiet,
    )
    
    return result.exit_code


# =============================================================================
# Command: smoke
# =============================================================================

def cmd_smoke(args: argparse.Namespace) -> int:
    """Run PhaseAB smoke test with standard paths."""
    verbose = not args.quiet
    
    if verbose:
        print("=" * 70)
        print("Phase AB Smoke Test")
        print("=" * 70)
    
    # Check if Phase A runs exist
    candidate_exists, baseline_exists = _check_phase_a_runs_exist()
    
    if not candidate_exists or not baseline_exists:
        if args.auto_generate_phasea:
            success = _generate_phase_a_runs(verbose=verbose)
            if not success:
                return 3
        else:
            _print_generate_commands()
            return 3
    
    # Run PhaseAB
    out_dir = args.out_dir if args.out_dir else DEFAULT_SMOKE_OUT_DIR
    
    result = run_phase_ab(
        candidate_run=DEFAULT_CANDIDATE_SMOKE,
        baseline_run=DEFAULT_BASELINE_SMOKE,
        out_dir=out_dir,
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        skip_evidence_verify=args.skip_evidence_verify,
        verbose=verbose,
    )
    
    return result.exit_code


# =============================================================================
# CLI Entry Point
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python3 -m batch_runs.phase_ab.cli",
        description="Phase AB: Integrated Phase A → Phase B Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  run     Run PhaseAB with explicit candidate/baseline paths (default)
  smoke   Run deterministic smoke test with standard paths (CI-friendly)

Examples:
  # Run mode with explicit paths
  python3 -m batch_runs.phase_ab.cli run \\
      --candidate-run runs/phaseA_candidate \\
      --baseline-run runs/phaseA_baseline \\
      --out-dir runs/phaseAB_output

  # Smoke test (uses runs/phaseA_*_smoke directories)
  python3 -m batch_runs.phase_ab.cli smoke

  # Smoke with auto-generate (creates Phase A runs if needed)
  python3 -m batch_runs.phase_ab.cli smoke --auto-generate-phasea

Exit codes (institutional CI semantics):
  0 = PROMOTE or HOLD (pipeline succeeded)
  2 = REJECT (candidate fails guardrails)
  3 = ERROR (runtime/IO/parsing failure)

Outputs:
  - phase_ab_manifest.json: Canonical manifest with all metadata
  - confidence_report.json: Machine-readable Phase B report
  - confidence_report.md: Human-readable Phase B report
""",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Common arguments
    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--alpha",
            type=float,
            default=0.05,
            help="Significance level for confidence intervals (default: 0.05)",
        )
        p.add_argument(
            "--n-bootstrap",
            type=int,
            default=1000,
            help="Number of bootstrap samples (default: 1000)",
        )
        p.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility (default: 42)",
        )
        p.add_argument(
            "--skip-evidence-verify",
            action="store_true",
            help="Skip evidence pack verification",
        )
        p.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="Suppress verbose output",
        )
    
    # Command: run
    run_parser = subparsers.add_parser(
        "run",
        help="Run PhaseAB with explicit paths",
        description="Run Phase AB orchestration with explicit candidate/baseline paths.",
    )
    run_parser.add_argument(
        "--candidate-run",
        type=Path,
        required=True,
        help="Path to candidate Phase A run directory",
    )
    run_parser.add_argument(
        "--baseline-run",
        type=Path,
        default=None,
        help="Path to baseline Phase A run directory (optional)",
    )
    run_parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for Phase AB results",
    )
    add_common_args(run_parser)
    
    # Command: smoke
    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run deterministic smoke test",
        description="Run Phase AB smoke test with standard paths. CI-friendly.",
    )
    smoke_parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {DEFAULT_SMOKE_OUT_DIR})",
    )
    smoke_parser.add_argument(
        "--auto-generate-phasea",
        action="store_true",
        help="Automatically generate Phase A runs if they don't exist",
    )
    add_common_args(smoke_parser)
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    # Default to 'run' if no command specified but --candidate-run is provided
    if args.command is None:
        # Check if we have candidate-run in remaining args (backwards compatibility)
        if argv and ("--candidate-run" in argv or "-h" in argv or "--help" in argv):
            # Re-parse with 'run' command
            args = parser.parse_args(["run"] + (argv or []))
        else:
            parser.print_help()
            return 0
    
    # Dispatch to command
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "smoke":
        return cmd_smoke(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

