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

CI Mode:
    --ci-mode smoke : HOLD is CI pass (exit 0) with clear messaging
    --ci-mode strict : HOLD is CI fail (exit 1) - promotion required
    
    For smoke/integration tests, use --ci-mode smoke (the default for smoke command).
    For promotion gates that require PROMOTE, use --ci-mode strict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from batch_runs.phase_ab.pipeline import (
    run_phase_ab,
    PhaseABResult,
    ROOT,
)


# =============================================================================
# Default Paths for Smoke Mode
# =============================================================================

DEFAULT_SMOKE_OUT_DIR = ROOT / "runs" / "phaseAB_smoke"
DEFAULT_CANDIDATE_SMOKE = ROOT / "runs" / "phaseA_candidate_smoke"
DEFAULT_BASELINE_SMOKE = ROOT / "runs" / "phaseA_baseline_smoke"


# =============================================================================
# CI Exit Code Helper
# =============================================================================

def ci_exit_code_for_decision(decision: str, ci_mode: str = "smoke") -> int:
    """
    Compute CI exit code based on promotion decision and CI mode.
    
    CI Mode semantics:
    - "smoke": HOLD is CI pass (exit 0), REJECT/ERROR are CI fail
              Used for smoke tests where we only verify the pipeline runs correctly.
    - "strict": HOLD is CI fail (exit 1), only PROMOTE passes
                Used for promotion gates that require definitive superiority.
    
    Exit codes:
    - 0: CI pass (PROMOTE always, HOLD in smoke mode)
    - 1: CI fail for HOLD in strict mode
    - 2: CI fail for REJECT
    - 3: CI fail for ERROR
    
    Args:
        decision: One of "PROMOTE", "HOLD", "REJECT", "ERROR"
        ci_mode: "smoke" (default) or "strict"
        
    Returns:
        Exit code for CI
    """
    decision_upper = decision.upper()
    
    if decision_upper == "PROMOTE":
        return 0
    elif decision_upper == "HOLD":
        if ci_mode == "strict":
            return 1  # Strict mode: HOLD is a failure
        else:
            return 0  # Smoke mode: HOLD is success (pipeline worked)
    elif decision_upper == "REJECT":
        return 2
    elif decision_upper == "ERROR":
        return 3
    else:
        return 3  # Unknown decision treated as error


def print_ci_summary(result: PhaseABResult, ci_mode: str = "smoke") -> None:
    """
    Print a clear CI summary message based on decision.
    
    Args:
        result: PhaseABResult from run_phase_ab
        ci_mode: "smoke" or "strict"
    """
    decision = result.manifest.decision
    exit_code = ci_exit_code_for_decision(decision, ci_mode)
    
    print()
    print("=" * 70)
    print("Phase AB CI Summary")
    print("=" * 70)
    print(f"  Decision: {decision}")
    print(f"  CI Mode: {ci_mode}")
    print(f"  Exit Code: {exit_code}")
    print()
    
    if decision == "PROMOTE":
        print("  ✓ CI PASS: Candidate is provably better than baseline.")
        print("  → Candidate may be promoted.")
    elif decision == "HOLD":
        if ci_mode == "smoke":
            print("  ✓ CI PASS: Pipeline succeeded. HOLD means not enough evidence yet.")
            print("  → This is EXPECTED for smoke tests with small sample sizes.")
            print("  → Guardrails passed (candidate not worse), but CIs overlap.")
            print("  → For promotion, collect more data to narrow confidence intervals.")
        else:
            print("  ✗ CI FAIL: HOLD decision in strict mode.")
            print("  → Strict mode requires PROMOTE for CI pass.")
            print("  → Collect more data to achieve statistical significance.")
    elif decision == "REJECT":
        print("  ✗ CI FAIL: Candidate fails guardrails.")
        print("  → Candidate is provably worse than baseline on at least one metric.")
        print("  → Review confidence_report.md for details.")
    elif decision == "ERROR":
        print("  ✗ CI FAIL: Pipeline encountered an error.")
        if result.manifest.errors:
            for err in result.manifest.errors:
                print(f"  → {err}")
    
    print()
    print(f"  Manifest: {result.manifest.phase_b_out_dir}/phase_ab_manifest.json")
    print(f"  Report: {result.manifest.confidence_report_md}")
    print("=" * 70)


def write_github_step_summary(result: PhaseABResult, ci_mode: str = "smoke") -> None:
    """
    Write GitHub Actions step summary if running in GitHub Actions.
    
    Writes to $GITHUB_STEP_SUMMARY if available.
    
    Args:
        result: PhaseABResult from run_phase_ab
        ci_mode: "smoke" or "strict"
    """
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return
    
    decision = result.manifest.decision
    exit_code = ci_exit_code_for_decision(decision, ci_mode)
    
    # Determine status emoji and text
    if decision == "PROMOTE":
        status_emoji = "✅"
        status_text = "CI PASS"
        detail = "Candidate is provably better than baseline."
    elif decision == "HOLD":
        if ci_mode == "smoke":
            status_emoji = "✅"
            status_text = "CI PASS"
            detail = "Pipeline succeeded. HOLD means not enough evidence to prove superiority (expected for smoke tests)."
        else:
            status_emoji = "❌"
            status_text = "CI FAIL"
            detail = "Strict mode requires PROMOTE. Candidate not provably better."
    elif decision == "REJECT":
        status_emoji = "❌"
        status_text = "CI FAIL"
        detail = "Candidate fails guardrails (provably worse)."
    else:
        status_emoji = "❌"
        status_text = "CI FAIL"
        detail = f"Pipeline error: {', '.join(result.manifest.errors) if result.manifest.errors else 'Unknown error'}"
    
    # Generate Markdown summary
    summary = f"""## Phase AB Confidence Gate Result

| Field | Value |
|-------|-------|
| **Status** | {status_emoji} {status_text} |
| **Decision** | `{decision}` |
| **Exit Code** | `{exit_code}` |
| **CI Mode** | `{ci_mode}` |
| **Alpha** | `{result.manifest.alpha}` |
| **Bootstrap Samples** | `{result.manifest.bootstrap_samples}` |
| **Candidate Samples** | `{result.manifest.candidate_samples}` |
| **Baseline Samples** | `{result.manifest.baseline_samples}` |

### Details

{detail}

### Checks Summary

| Check | Status |
|-------|--------|
| **Guardrails** | {'✅ Passed' if decision in ('PROMOTE', 'HOLD') else '❌ Failed'} |
| **Promotion Criteria** | {'✅ Passed' if decision == 'PROMOTE' else ('⏸️ Insufficient evidence' if decision == 'HOLD' else '❌ Failed')} |

### Output Files

- Manifest: `{result.manifest.phase_b_out_dir}/phase_ab_manifest.json`
- Report (JSON): `{result.manifest.confidence_report_json}`
- Report (MD): `{result.manifest.confidence_report_md}`

---
*Generated by Phase AB Confidence Gate*
"""
    
    try:
        with open(summary_file, "a") as f:
            f.write(summary)
    except Exception:
        pass  # Silently fail if we can't write summary


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
    
    # Determine CI mode (default to smoke for backwards compatibility)
    ci_mode = getattr(args, 'ci_mode', 'smoke')
    
    # Print CI summary
    if not args.quiet:
        print_ci_summary(result, ci_mode=ci_mode)
    
    # Write GitHub Actions step summary
    write_github_step_summary(result, ci_mode=ci_mode)
    
    # Return CI-appropriate exit code
    return ci_exit_code_for_decision(result.manifest.decision, ci_mode=ci_mode)


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
    
    # Smoke mode always uses "smoke" CI mode (HOLD is pass)
    ci_mode = getattr(args, 'ci_mode', 'smoke')
    
    # Print CI summary
    if verbose:
        print_ci_summary(result, ci_mode=ci_mode)
    
    # Write GitHub Actions step summary
    write_github_step_summary(result, ci_mode=ci_mode)
    
    # Return CI-appropriate exit code
    return ci_exit_code_for_decision(result.manifest.decision, ci_mode=ci_mode)


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

  # Strict mode for promotion gates (HOLD = CI fail)
  python3 -m batch_runs.phase_ab.cli run --ci-mode strict ...

CI Mode:
  --ci-mode smoke  : HOLD is CI pass (exit 0) - default
  --ci-mode strict : HOLD is CI fail (exit 1) - for promotion gates

Exit codes (with --ci-mode smoke):
  0 = PROMOTE or HOLD (pipeline succeeded)
  2 = REJECT (candidate fails guardrails)
  3 = ERROR (runtime/IO/parsing failure)

Exit codes (with --ci-mode strict):
  0 = PROMOTE only (candidate provably better)
  1 = HOLD (not enough evidence)
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
            "--ci-mode",
            type=str,
            choices=["smoke", "strict"],
            default="smoke",
            help="CI mode: 'smoke' (HOLD=pass, default) or 'strict' (HOLD=fail)",
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
