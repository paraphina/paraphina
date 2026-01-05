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

    # Smoke with deterministic output (CI artifact-friendly)
    python3 -m batch_runs.phase_ab.cli smoke --auto-generate-phasea \\
        --out-dir runs/ci/phase_ab_smoke --seed 12345

    # Promotion gate (strict mode with mandatory evidence verification)
    python3 -m batch_runs.phase_ab.cli gate \\
        --out-dir runs/ci/phase_ab_gate \\
        --seed 24680 \\
        --n-bootstrap 1000 \\
        --auto-generate-phasea

    # Gate with explicit candidate/baseline runs
    python3 -m batch_runs.phase_ab.cli gate \\
        --candidate-run runs/phaseA_candidate \\
        --baseline-run runs/phaseA_baseline \\
        --out-dir runs/ci/phase_ab_gate \\
        --seed 24680

    # Verify an existing evidence pack
    python3 -m batch_runs.phase_ab.cli verify-evidence runs/ci/phase_ab_smoke/evidence_pack

Exit codes (institutional CI semantics):

    Smoke mode (default for smoke command):
        0 = PROMOTE or HOLD (pipeline succeeded)
        1 = (not used in smoke mode)
        2 = REJECT (candidate fails guardrails) or verify-evidence failure
        3 = ERROR (runtime/IO/parsing failure)

    Strict/Gate mode (default for gate command):
        0 = PROMOTE (candidate is provably better - PASS)
        1 = REJECT (candidate fails guardrails - FAIL)
        2 = HOLD (insufficient evidence - needs more data)
        3 = ERROR (runtime/IO/parsing failure) or other unexpected states

CI Mode:
    --ci-mode smoke : HOLD is CI pass (exit 0) with clear messaging
    --ci-mode strict : HOLD is CI fail (exit 2) - promotion required

    For smoke/integration tests, use --ci-mode smoke (the default for smoke command).
    For promotion gates that require PROMOTE, use --ci-mode strict or the gate command.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
# Git Commit Helper
# =============================================================================

def _get_git_commit() -> Optional[str]:
    """Get the current git commit hash, or None if not in a git repo."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# =============================================================================
# Exit Code Constants (Institutional Grade)
# =============================================================================

# Smoke mode exit codes (default for smoke command)
# Used for smoke tests where we only verify the pipeline runs correctly.
SMOKE_EXIT_PASS = 0       # PROMOTE or HOLD - pipeline succeeded
SMOKE_EXIT_REJECT = 2     # REJECT - candidate fails guardrails
SMOKE_EXIT_ERROR = 3      # ERROR - runtime/IO/parsing failure

# Strict/Gate mode exit codes (default for gate command)
# Used for promotion gates that require definitive superiority.
GATE_EXIT_PASS = 0        # PROMOTE - candidate is provably better
GATE_EXIT_FAIL = 1        # REJECT - candidate fails guardrails
GATE_EXIT_HOLD = 2        # HOLD - insufficient evidence (needs more data)
GATE_EXIT_ERROR = 3       # ERROR - runtime/IO/parsing failure


# =============================================================================
# CI Exit Code Helper
# =============================================================================

def ci_exit_code_for_decision(decision: str, ci_mode: str = "smoke") -> int:
    """
    Compute CI exit code based on promotion decision and CI mode.
    
    CI Mode semantics:
    - "smoke": HOLD is CI pass (exit 0), REJECT/ERROR are CI fail
              Used for smoke tests where we only verify the pipeline runs correctly.
    - "strict": Deterministic exit codes for promotion gates:
              PASS (PROMOTE) = 0, FAIL (REJECT) = 1, HOLD = 2, ERROR = 3
              Used for promotion gates that require definitive superiority.
    
    Smoke mode exit codes:
    - 0: CI pass (PROMOTE or HOLD - pipeline succeeded)
    - 2: CI fail for REJECT (candidate fails guardrails)
    - 3: CI fail for ERROR (runtime/IO/parsing failure)
    
    Strict/Gate mode exit codes:
    - 0: PASS (PROMOTE - candidate is provably better)
    - 1: FAIL (REJECT - candidate fails guardrails)
    - 2: HOLD (insufficient evidence - needs more data)
    - 3: ERROR (runtime/IO/parsing failure or unknown state)
    
    Args:
        decision: One of "PROMOTE", "HOLD", "REJECT", "ERROR"
        ci_mode: "smoke" (default) or "strict"
        
    Returns:
        Exit code for CI
    """
    decision_upper = decision.upper()
    
    if ci_mode == "strict":
        # Strict/Gate mode: deterministic institutional exit codes
        if decision_upper == "PROMOTE":
            return GATE_EXIT_PASS   # 0
        elif decision_upper == "REJECT":
            return GATE_EXIT_FAIL   # 1
        elif decision_upper == "HOLD":
            return GATE_EXIT_HOLD   # 2
        elif decision_upper == "ERROR":
            return GATE_EXIT_ERROR  # 3
        else:
            return GATE_EXIT_ERROR  # 3 - Unknown decision treated as error
    else:
        # Smoke mode: HOLD is success (pipeline worked)
        if decision_upper == "PROMOTE":
            return SMOKE_EXIT_PASS   # 0
        elif decision_upper == "HOLD":
            return SMOKE_EXIT_PASS   # 0 - HOLD is success in smoke mode
        elif decision_upper == "REJECT":
            return SMOKE_EXIT_REJECT # 2
        elif decision_upper == "ERROR":
            return SMOKE_EXIT_ERROR  # 3
        else:
            return SMOKE_EXIT_ERROR  # 3 - Unknown decision treated as error


def get_exit_code_description(ci_mode: str = "smoke") -> str:
    """
    Get human-readable exit code documentation for a CI mode.
    
    Args:
        ci_mode: "smoke" or "strict"
        
    Returns:
        Multi-line string describing exit codes
    """
    if ci_mode == "strict":
        return """Exit codes (strict/gate mode):
    0 = PASS (PROMOTE - candidate is provably better)
    1 = FAIL (REJECT - candidate fails guardrails)
    2 = HOLD (insufficient evidence - needs more data)
    3 = ERROR (runtime/IO/parsing failure)"""
    else:
        return """Exit codes (smoke mode):
    0 = PROMOTE or HOLD (pipeline succeeded)
    2 = REJECT (candidate fails guardrails)
    3 = ERROR (runtime/IO/parsing failure)"""


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
            print("  ⏸ CI HOLD: Insufficient evidence for promotion (exit code 2).")
            print("  → Guardrails passed (candidate not worse).")
            print("  → Strict mode requires PROMOTE (exit 0) for CI pass.")
            print("  → Collect more data to achieve statistical significance.")
    elif decision == "REJECT":
        if ci_mode == "strict":
            print("  ✗ CI FAIL: Candidate fails guardrails (exit code 1).")
        else:
            print("  ✗ CI FAIL: Candidate fails guardrails.")
        print("  → Candidate is provably worse than baseline on at least one metric.")
        print("  → Review confidence_report.md for details.")
    elif decision == "ERROR":
        print("  ✗ CI ERROR: Pipeline encountered an error (exit code 3).")
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
    
    # Determine status emoji and text based on mode
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
            status_emoji = "⏸"
            status_text = "CI HOLD"
            detail = "Insufficient evidence for promotion. Guardrails passed but confidence intervals overlap. Collect more data."
    elif decision == "REJECT":
        status_emoji = "❌"
        status_text = "CI FAIL"
        detail = "Candidate fails guardrails (provably worse)."
    else:
        status_emoji = "❌"
        status_text = "CI ERROR"
        detail = f"Pipeline error: {', '.join(result.manifest.errors) if result.manifest.errors else 'Unknown error'}"
    
    # Exit code documentation based on mode
    if ci_mode == "strict":
        exit_code_doc = "0=PASS, 1=FAIL, 2=HOLD, 3=ERROR"
    else:
        exit_code_doc = "0=PASS/HOLD, 2=REJECT, 3=ERROR"
    
    # Generate Markdown summary
    summary = f"""## Phase AB Confidence Gate Result

| Field | Value |
|-------|-------|
| **Status** | {status_emoji} {status_text} |
| **Decision** | `{decision}` |
| **Exit Code** | `{exit_code}` ({exit_code_doc}) |
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
| **Promotion Criteria** | {'✅ Passed' if decision == 'PROMOTE' else ('⏸ Insufficient evidence' if decision == 'HOLD' else '❌ Failed')} |

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
# Evidence Pack Manifest Helpers
# =============================================================================

def _write_evidence_pack_manifest(
    evidence_pack_dir: Path,
    cli_args: List[str],
    seed: Optional[int],
) -> Path:
    """
    Write manifest.json for evidence pack.
    
    Args:
        evidence_pack_dir: Path to evidence_pack directory
        cli_args: CLI arguments used
        seed: Random seed (if provided)
        
    Returns:
        Path to manifest.json
    """
    from batch_runs.evidence_pack import write_manifest, verify_manifest
    
    metadata: Dict[str, Any] = {
        "cli_args": cli_args,
    }
    if seed is not None:
        metadata["seed"] = seed
    
    git_commit = _get_git_commit()
    if git_commit:
        metadata["git_commit"] = git_commit
    
    manifest_path = write_manifest(evidence_pack_dir, metadata)
    
    # Self-check: verify immediately after writing
    verify_manifest(evidence_pack_dir)
    
    return manifest_path


def _write_phase_ab_summary(
    out_dir: Path,
    outcome: str,
    mode: str,
    evidence_pack_dir: Path,
    manifest_path: Path,
    seed: Optional[int],
) -> Path:
    """
    Write phase_ab_summary.json with machine-readable run info.
    
    Args:
        out_dir: Output directory
        outcome: PASS/FAIL/HOLD
        mode: "smoke" or "run"
        evidence_pack_dir: Path to evidence_pack
        manifest_path: Path to manifest.json
        seed: Random seed (if provided)
        
    Returns:
        Path to summary file
    """
    summary = {
        "outcome": outcome,
        "mode": mode,
        "evidence_pack_dir": str(evidence_pack_dir),
        "manifest_path": str(manifest_path),
        "seed": seed,
        "git_commit": _get_git_commit(),
        "python_version": sys.version.split()[0],
    }
    
    summary_path = out_dir / "phase_ab_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    
    return summary_path


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


def _generate_phase_a_runs(verbose: bool = True, seed: Optional[int] = None) -> bool:
    """
    Generate Phase A smoke runs.
    
    Args:
        verbose: Print progress
        seed: Base seed for RNG (candidate uses seed, baseline uses seed+1)
    
    Returns:
        True if generation succeeded, False otherwise
    """
    import subprocess
    
    if verbose:
        print("[PhaseAB Smoke] Generating Phase A smoke runs...")
    
    # Use provided seed or default
    candidate_seed = seed if seed is not None else 42
    baseline_seed = candidate_seed + 1
    
    # Generate candidate
    candidate_exists, baseline_exists = _check_phase_a_runs_exist()
    
    if not candidate_exists:
        if verbose:
            print(f"\n  Generating candidate: {DEFAULT_CANDIDATE_SMOKE}")
        
        cmd = [
            sys.executable, "-m", "batch_runs.phase_a.promote_pipeline",
            "--smoke",
            "--study-dir", str(DEFAULT_CANDIDATE_SMOKE),
            "--seed", str(candidate_seed),
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
            "--seed", str(baseline_seed),
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
    # Seed RNG if provided
    if args.seed is not None:
        random.seed(args.seed)
    
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
    
    # Handle evidence pack manifest (if out_dir has evidence_pack)
    out_dir = Path(args.out_dir).resolve()
    evidence_pack_dir = out_dir / "evidence_pack"
    
    if evidence_pack_dir.exists() and evidence_pack_dir.is_dir():
        try:
            cli_args = sys.argv[1:] if len(sys.argv) > 1 else []
            manifest_path = _write_evidence_pack_manifest(
                evidence_pack_dir,
                cli_args,
                args.seed,
            )
            if not args.quiet:
                print(f"  Evidence pack manifest: {manifest_path}")
            
            # Write summary
            outcome = result.manifest.decision
            summary_path = _write_phase_ab_summary(
                out_dir,
                outcome=outcome,
                mode="run",
                evidence_pack_dir=evidence_pack_dir,
                manifest_path=manifest_path,
                seed=args.seed,
            )
            if not args.quiet:
                print(f"  Phase AB summary: {summary_path}")
        except Exception as e:
            print(f"WARNING: Failed to write evidence pack manifest: {e}", file=sys.stderr)
    
    # Return CI-appropriate exit code
    return ci_exit_code_for_decision(result.manifest.decision, ci_mode=ci_mode)


# =============================================================================
# Command: smoke
# =============================================================================

def cmd_smoke(args: argparse.Namespace) -> int:
    """Run PhaseAB smoke test with standard paths."""
    verbose = not args.quiet
    
    # Seed RNG if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    if verbose:
        print("=" * 70)
        print("Phase AB Smoke Test")
        print("=" * 70)
    
    # Check if Phase A runs exist
    candidate_exists, baseline_exists = _check_phase_a_runs_exist()
    
    if not candidate_exists or not baseline_exists:
        if args.auto_generate_phasea:
            success = _generate_phase_a_runs(verbose=verbose, seed=args.seed)
            if not success:
                return 3
        else:
            _print_generate_commands()
            return 3
    
    # Run PhaseAB
    out_dir = Path(args.out_dir).resolve() if args.out_dir else DEFAULT_SMOKE_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Create evidence_pack directory and write manifest
    evidence_pack_dir = out_dir / "evidence_pack"
    evidence_pack_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy relevant files to evidence_pack
    _populate_evidence_pack(out_dir, evidence_pack_dir, result)
    
    # Write evidence pack manifest
    try:
        cli_args = sys.argv[1:] if len(sys.argv) > 1 else []
        manifest_path = _write_evidence_pack_manifest(
            evidence_pack_dir,
            cli_args,
            args.seed,
        )
        if verbose:
            print()
            print(f"  Evidence pack: {evidence_pack_dir}")
            print(f"  Evidence pack manifest: {manifest_path}")
        
        # Write summary
        outcome = result.manifest.decision
        summary_path = _write_phase_ab_summary(
            out_dir,
            outcome=outcome,
            mode="smoke",
            evidence_pack_dir=evidence_pack_dir,
            manifest_path=manifest_path,
            seed=args.seed,
        )
        if verbose:
            print(f"  Phase AB summary: {summary_path}")
    except Exception as e:
        print(f"WARNING: Failed to write evidence pack manifest: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    
    # Print final outcome
    if verbose:
        print()
        print("=" * 70)
        print(f"  Gating outcome: {result.manifest.decision}")
        print(f"  Evidence pack path: {evidence_pack_dir.absolute()}")
        print("=" * 70)
    
    # Return CI-appropriate exit code
    return ci_exit_code_for_decision(result.manifest.decision, ci_mode=ci_mode)


def _populate_evidence_pack(out_dir: Path, evidence_pack_dir: Path, result: PhaseABResult) -> None:
    """
    Populate evidence_pack directory with relevant files.
    
    Copies key output files to evidence_pack for integrity verification.
    """
    import shutil
    
    # Files to include in evidence pack
    files_to_copy = [
        "phase_ab_manifest.json",
        "confidence_report.json",
        "confidence_report.md",
    ]
    
    for filename in files_to_copy:
        src = out_dir / filename
        if src.exists():
            dst = evidence_pack_dir / filename
            shutil.copy2(src, dst)


# =============================================================================
# Command: gate (Promotion Gate - Strict Mode)
# =============================================================================

def cmd_gate(args: argparse.Namespace) -> int:
    """
    Run Phase AB promotion gate in strict mode with mandatory evidence verification.
    
    This is the institutional-grade promotion gate that:
    1. Runs Phase AB evaluation in STRICT mode
    2. Writes all standard outputs to --out-dir
    3. Immediately runs evidence verification
    4. Enforces deterministic exit codes:
       - PASS (PROMOTE) => 0
       - FAIL (REJECT) => 1
       - HOLD => 2
       - ERROR => 3
    
    Args:
        args: Parsed CLI arguments
        
    Returns:
        Exit code per strict mode contract
    """
    verbose = not args.quiet
    
    # Seed RNG if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    if verbose:
        print("=" * 70)
        print("Phase AB Promotion Gate (Strict Mode)")
        print("=" * 70)
        print()
        print(get_exit_code_description("strict"))
        print()
    
    # Determine candidate/baseline paths
    use_auto_generate = getattr(args, 'auto_generate_phasea', False)
    
    if use_auto_generate:
        # Auto-generate Phase A runs if needed
        candidate_exists, baseline_exists = _check_phase_a_runs_exist()
        
        if not candidate_exists or not baseline_exists:
            success = _generate_phase_a_runs(verbose=verbose, seed=args.seed)
            if not success:
                print("ERROR: Failed to auto-generate Phase A runs", file=sys.stderr)
                return GATE_EXIT_ERROR
        
        candidate_run = DEFAULT_CANDIDATE_SMOKE
        baseline_run = DEFAULT_BASELINE_SMOKE
    else:
        # Use explicit paths
        if not args.candidate_run:
            print("ERROR: --candidate-run is required when not using --auto-generate-phasea", file=sys.stderr)
            return GATE_EXIT_ERROR
        
        candidate_run = Path(args.candidate_run)
        baseline_run = Path(args.baseline_run) if args.baseline_run else None
    
    # Ensure output directory
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"  Candidate: {candidate_run}")
        if baseline_run:
            print(f"  Baseline: {baseline_run}")
        print(f"  Output: {out_dir}")
        print(f"  Seed: {args.seed}")
        print(f"  Bootstrap samples: {args.n_bootstrap}")
        print()
    
    # Run Phase AB (evidence verification is NOT skipped in gate mode)
    result = run_phase_ab(
        candidate_run=candidate_run,
        baseline_run=baseline_run,
        out_dir=out_dir,
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        skip_evidence_verify=False,  # Never skip in gate mode
        verbose=verbose,
    )
    
    # Create evidence_pack directory and populate
    evidence_pack_dir = out_dir / "evidence_pack"
    evidence_pack_dir.mkdir(parents=True, exist_ok=True)
    _populate_evidence_pack(out_dir, evidence_pack_dir, result)
    
    # Write evidence pack manifest
    try:
        cli_args = sys.argv[1:] if len(sys.argv) > 1 else []
        manifest_path = _write_evidence_pack_manifest(
            evidence_pack_dir,
            cli_args,
            args.seed,
        )
        if verbose:
            print()
            print(f"  Evidence pack: {evidence_pack_dir}")
            print(f"  Evidence pack manifest: {manifest_path}")
    except Exception as e:
        print(f"ERROR: Failed to write evidence pack manifest: {e}", file=sys.stderr)
        return GATE_EXIT_ERROR
    
    # MANDATORY: Run evidence verification on the output evidence pack
    if verbose:
        print()
        print("[Gate] Verifying evidence pack integrity...")
    
    from batch_runs.evidence_pack import verify_manifest, ManifestError
    
    try:
        verify_manifest(evidence_pack_dir)
        if verbose:
            print("  ✓ Evidence pack verified successfully")
    except ManifestError as e:
        print(f"ERROR: Evidence verification failed: {e}", file=sys.stderr)
        return GATE_EXIT_ERROR
    except Exception as e:
        print(f"ERROR: Evidence verification error: {e}", file=sys.stderr)
        return GATE_EXIT_ERROR
    
    # Write summary
    try:
        outcome = result.manifest.decision
        summary_path = _write_phase_ab_summary(
            out_dir,
            outcome=outcome,
            mode="gate",
            evidence_pack_dir=evidence_pack_dir,
            manifest_path=manifest_path,
            seed=args.seed,
        )
        if verbose:
            print(f"  Phase AB summary: {summary_path}")
    except Exception as e:
        print(f"WARNING: Failed to write summary: {e}", file=sys.stderr)
    
    # Gate mode always uses strict CI mode
    ci_mode = "strict"
    
    # Print CI summary
    if verbose:
        print_ci_summary(result, ci_mode=ci_mode)
    
    # Write GitHub Actions step summary
    write_github_step_summary(result, ci_mode=ci_mode)
    
    # Print final outcome
    if verbose:
        print()
        print("=" * 70)
        print(f"  Gating outcome: {result.manifest.decision}")
        print(f"  Evidence pack path: {evidence_pack_dir.absolute()}")
        print("=" * 70)
    
    # Return strict mode exit code
    return ci_exit_code_for_decision(result.manifest.decision, ci_mode=ci_mode)


# =============================================================================
# Command: verify-evidence
# =============================================================================

def cmd_verify_evidence(args: argparse.Namespace) -> int:
    """
    Verify an evidence pack directory.
    
    Exit codes:
        0: Verification passed
        2: Verification failed
    """
    from batch_runs.evidence_pack import verify_manifest, ManifestError, count_manifest_files
    
    evidence_dir = Path(args.evidence_dir).resolve()
    
    if not evidence_dir.exists():
        print(f"ERROR: Evidence directory not found: {evidence_dir}", file=sys.stderr)
        return 2
    
    if not evidence_dir.is_dir():
        print(f"ERROR: Path is not a directory: {evidence_dir}", file=sys.stderr)
        return 2
    
    manifest_path = evidence_dir / "manifest.json"
    
    try:
        verify_manifest(evidence_dir, allow_extra=args.allow_extra)
        
        # Count files for summary
        file_count = count_manifest_files(manifest_path)
        
        print(f"OK: Evidence pack verified ({file_count} files)")
        return 0
        
    except ManifestError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


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
  run             Run PhaseAB with explicit candidate/baseline paths (default)
  smoke           Run deterministic smoke test with standard paths (CI-friendly)
  gate            Run promotion gate (strict mode with mandatory evidence verification)
  verify-evidence Verify an evidence pack directory

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

  # Smoke with deterministic output for CI
  python3 -m batch_runs.phase_ab.cli smoke --auto-generate-phasea \\
      --out-dir runs/ci/phase_ab_smoke --seed 12345

  # Promotion gate with auto-generate (strict mode)
  python3 -m batch_runs.phase_ab.cli gate \\
      --auto-generate-phasea \\
      --out-dir runs/ci/phase_ab_gate \\
      --seed 24680 \\
      --n-bootstrap 1000

  # Promotion gate with explicit paths
  python3 -m batch_runs.phase_ab.cli gate \\
      --candidate-run runs/phaseA_candidate \\
      --baseline-run runs/phaseA_baseline \\
      --out-dir runs/ci/phase_ab_gate \\
      --seed 24680

  # Verify an evidence pack
  python3 -m batch_runs.phase_ab.cli verify-evidence runs/ci/phase_ab_smoke/evidence_pack

  # Strict mode for run command (HOLD = CI fail)
  python3 -m batch_runs.phase_ab.cli run --ci-mode strict ...

CI Mode:
  --ci-mode smoke  : HOLD is CI pass (exit 0) - default for smoke command
  --ci-mode strict : HOLD is CI fail (exit 2) - for promotion gates

Exit codes (smoke mode - default for smoke command):
  0 = PROMOTE or HOLD (pipeline succeeded)
  2 = REJECT (candidate fails guardrails)
  3 = ERROR (runtime/IO/parsing failure)

Exit codes (strict/gate mode - default for gate command):
  0 = PASS (PROMOTE - candidate is provably better)
  1 = FAIL (REJECT - candidate fails guardrails)
  2 = HOLD (insufficient evidence - needs more data)
  3 = ERROR (runtime/IO/parsing failure)

Outputs:
  - phase_ab_manifest.json: Canonical manifest with all metadata
  - confidence_report.json: Machine-readable Phase B report
  - confidence_report.md: Human-readable Phase B report
  - evidence_pack/manifest.json: Deterministic evidence pack manifest
  - phase_ab_summary.json: Machine-readable summary for CI
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
    
    # Command: gate (Promotion Gate - Strict Mode)
    gate_parser = subparsers.add_parser(
        "gate",
        help="Run promotion gate (strict mode with mandatory evidence verification)",
        description="""Run Phase AB promotion gate in strict mode.

This is the institutional-grade promotion gate that:
1. Runs Phase AB evaluation in STRICT mode (promotion semantics)
2. Writes all standard outputs to --out-dir
3. Immediately runs evidence verification (mandatory)
4. Enforces deterministic exit codes:
   - PASS (PROMOTE) => 0
   - FAIL (REJECT) => 1
   - HOLD => 2
   - ERROR => 3

Use --auto-generate-phasea to auto-generate Phase A runs (for CI smoke testing).
Otherwise, --candidate-run is required.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    gate_parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for Phase AB results (required)",
    )
    gate_parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for reproducibility (required for determinism)",
    )
    gate_parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples (default: 1000)",
    )
    gate_parser.add_argument(
        "--candidate-run",
        type=Path,
        default=None,
        help="Path to candidate Phase A run directory (required unless --auto-generate-phasea)",
    )
    gate_parser.add_argument(
        "--baseline-run",
        type=Path,
        default=None,
        help="Path to baseline Phase A run directory (optional)",
    )
    gate_parser.add_argument(
        "--auto-generate-phasea",
        action="store_true",
        help="Automatically generate Phase A runs if they don't exist",
    )
    gate_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for confidence intervals (default: 0.05)",
    )
    gate_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    
    # Command: verify-evidence
    verify_parser = subparsers.add_parser(
        "verify-evidence",
        help="Verify an evidence pack directory",
        description="Verify evidence pack integrity against manifest.json.",
    )
    verify_parser.add_argument(
        "evidence_dir",
        type=Path,
        help="Path to evidence pack directory",
    )
    verify_parser.add_argument(
        "--allow-extra",
        action="store_true",
        help="Allow extra files not in manifest",
    )
    
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
    elif args.command == "gate":
        return cmd_gate(args)
    elif args.command == "verify-evidence":
        return cmd_verify_evidence(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
