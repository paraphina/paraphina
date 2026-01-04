"""
pipeline.py

Phase AB: Integrated Phase A → Phase B Orchestrator.

Core logic for:
- Validating and normalizing Phase A run directories
- Verifying evidence packs
- Running Phase B confidence gating
- Writing canonical manifest and outputs

This module is importable and provides the `run_phase_ab()` function.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]  # Repo root (paraphina/)
SIM_EVAL_BIN = ROOT / "target" / "release" / "sim_eval"


# =============================================================================
# Exceptions
# =============================================================================

class RunRootError(Exception):
    """Base exception for run root resolution errors."""
    pass


class TrialsNotFoundError(RunRootError):
    """Raised when trials.jsonl cannot be found."""
    pass


class AdversarialSearchError(RunRootError):
    """
    Raised when directory looks like Phase A adversarial search output.
    
    This is NOT an evaluated run - user needs to run sim_eval suite first.
    """
    pass


class NestedRunError(RunRootError):
    """
    Raised when directory structure indicates nested sim_eval_suite_run.
    
    Provides guidance on which path to pass.
    """
    pass


class EvidenceVerificationError(Exception):
    """Raised when evidence pack verification fails."""
    pass


# =============================================================================
# Run Root Resolution
# =============================================================================

def _looks_like_adversarial_search_output(directory: Path) -> bool:
    """
    Check if directory looks like Phase A adversarial search output.
    
    Adversarial search produces:
    - generated_suite/ directory
    - search_results.jsonl
    - summary.json
    
    These are NOT evaluated runs - they need sim_eval suite to be run first.
    """
    indicators = [
        directory / "generated_suite",
        directory / "search_results.jsonl",
        directory / "summary.json",
    ]
    return any(p.exists() for p in indicators)


def _looks_like_sim_eval_suite_nesting(directory: Path) -> Tuple[bool, Optional[Path]]:
    """
    Check if directory has sim_eval_suite_run nesting.
    
    If the directory contains sim_eval_suite_run/<something>/trials.jsonl,
    provide guidance on which path to pass.
    
    Returns:
        (is_nested, suggested_path) tuple
    """
    sim_eval_run = directory / "sim_eval_suite_run"
    if not sim_eval_run.exists():
        return False, None
    
    # Look for trials.jsonl under sim_eval_suite_run
    for child in sim_eval_run.iterdir():
        if child.is_dir():
            trials = child / "trials.jsonl"
            if trials.exists():
                return True, child
    
    return False, None


def _find_trials_one_level_deep(directory: Path) -> List[Tuple[Path, float]]:
    """
    Find all trials.jsonl files one level deep.
    
    Returns list of (trials_file_path, mtime) tuples.
    """
    results = []
    try:
        for child in directory.iterdir():
            if child.is_dir():
                trials = child / "trials.jsonl"
                if trials.exists():
                    mtime = trials.stat().st_mtime
                    results.append((trials, mtime))
    except (PermissionError, OSError):
        pass
    return results


def resolve_run_root(directory: Path, verbose: bool = False) -> Path:
    """
    Resolve and validate a Phase A run directory to its canonical root.
    
    The canonical run root is a directory containing trials.jsonl.
    
    Resolution rules:
    1. If <dir>/trials.jsonl exists → use <dir>
    2. Else search one level deep for trials.jsonl:
       - If exactly one found → use that parent directory
       - If multiple found → choose newest mtime, print warning
       - If none found → raise with diagnostic error message
    
    Args:
        directory: Input directory path
        verbose: Print resolution steps
        
    Returns:
        Canonical run root path (directory containing trials.jsonl)
        
    Raises:
        TrialsNotFoundError: trials.jsonl not found
        AdversarialSearchError: Directory is adversarial search output
        NestedRunError: Directory has sim_eval nesting issues
    """
    directory = Path(directory).resolve()
    
    if not directory.exists():
        raise TrialsNotFoundError(
            f"Directory does not exist: {directory}"
        )
    
    if not directory.is_dir():
        # If it's a file, check if it's trials.jsonl itself
        if directory.name == "trials.jsonl" and directory.is_file():
            return directory.parent
        raise TrialsNotFoundError(
            f"Path is not a directory: {directory}"
        )
    
    # Rule 1: Check for trials.jsonl at root
    trials_at_root = directory / "trials.jsonl"
    if trials_at_root.exists():
        if verbose:
            print(f"  Found trials.jsonl at root: {directory}")
        return directory
    
    # Check for adversarial search output BEFORE nested search
    if _looks_like_adversarial_search_output(directory):
        raise AdversarialSearchError(
            f"This looks like Phase A adversarial search output, not an evaluated run. "
            f"Phase B gating requires an evaluated run containing trials.jsonl.\n"
            f"\n"
            f"Directory: {directory}\n"
            f"\n"
            f"To create an evaluated run, run sim_eval suite using the generated suite yaml:\n"
            f"\n"
            f"    sim_eval suite {directory}/generated_suite/suite.yaml --output-dir <output_path>\n"
            f"\n"
            f"Then pass <output_path> to PhaseAB instead."
        )
    
    # Check for sim_eval nesting
    is_nested, suggested_path = _looks_like_sim_eval_suite_nesting(directory)
    if is_nested and suggested_path:
        raise NestedRunError(
            f"Directory contains sim_eval_suite_run nesting.\n"
            f"\n"
            f"Expected path: {suggested_path}\n"
            f"You passed: {directory}\n"
            f"\n"
            f"Try passing the nested directory instead:\n"
            f"\n"
            f"    --candidate-run {suggested_path}"
        )
    
    # Rule 2: Search one level deep
    candidates = _find_trials_one_level_deep(directory)
    
    if len(candidates) == 0:
        # No trials.jsonl found - provide helpful error
        raise TrialsNotFoundError(
            f"No trials.jsonl found in {directory} or its immediate subdirectories.\n"
            f"\n"
            f"Phase B gating requires an evaluated Phase A run containing trials.jsonl.\n"
            f"\n"
            f"Common causes:\n"
            f"  1. Directory is not a Phase A run output\n"
            f"  2. Phase A run did not complete successfully\n"
            f"  3. Wrong directory specified\n"
            f"\n"
            f"Expected structure:\n"
            f"  {directory}/\n"
            f"    trials.jsonl      <- this file is required\n"
            f"    pareto.json\n"
            f"    pareto.csv\n"
            f"    trial_XXXX/..."
        )
    
    if len(candidates) == 1:
        trials_path, _ = candidates[0]
        run_root = trials_path.parent
        if verbose:
            print(f"  Found trials.jsonl in subdirectory: {run_root}")
        return run_root
    
    # Multiple candidates - choose newest, warn user
    candidates.sort(key=lambda x: x[1], reverse=True)  # Sort by mtime, newest first
    newest_path, newest_mtime = candidates[0]
    run_root = newest_path.parent
    
    # Print warning
    print(f"WARNING: Multiple trials.jsonl found. Using newest:", file=sys.stderr)
    print(f"  Selected: {run_root}", file=sys.stderr)
    print(f"  Other candidates:", file=sys.stderr)
    for path, mtime in candidates[1:]:
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        print(f"    - {path.parent} (modified: {dt.isoformat()})", file=sys.stderr)
    
    return run_root


# =============================================================================
# Evidence Verification
# =============================================================================

class SimEvalNotFoundError(Exception):
    """Raised when the sim_eval binary cannot be found."""
    pass


def find_sim_eval_bin(verbose: bool = False) -> Path:
    """
    Find the sim_eval binary from the repo's build output.
    
    Priority order:
    1. SIM_EVAL_BIN environment variable (if set and executable)
    2. target/release/sim_eval (preferred)
    3. target/debug/sim_eval (fallback for dev builds)
    4. shutil.which("sim_eval") in PATH (last resort, with warning)
    
    Note: We prefer repo-built binaries to ensure compatibility with
    the verify-evidence-pack command. System-installed sim_eval binaries
    may not support all commands.
    
    Args:
        verbose: Print which binary is being used
        
    Returns:
        Path to sim_eval binary
        
    Raises:
        SimEvalNotFoundError: If no sim_eval binary can be found
    """
    import shutil
    
    # 1. Check environment variable first
    env_bin = os.environ.get("SIM_EVAL_BIN")
    if env_bin:
        env_path = Path(env_bin)
        if env_path.exists() and os.access(env_path, os.X_OK):
            if verbose:
                print(f"  Using sim_eval from SIM_EVAL_BIN: {env_path}")
            return env_path
    
    # 2. Check target/release (preferred)
    release_bin = ROOT / "target" / "release" / "sim_eval"
    if release_bin.exists():
        if verbose:
            print(f"  Using sim_eval from: {release_bin}")
        return release_bin
    
    # 3. Check target/debug (fallback for dev builds)
    debug_bin = ROOT / "target" / "debug" / "sim_eval"
    if debug_bin.exists():
        if verbose:
            print(f"  Using sim_eval from: {debug_bin}")
        return debug_bin
    
    # 4. Last resort: check PATH (but warn, as it may be incompatible)
    which_result = shutil.which("sim_eval")
    if which_result:
        which_path = Path(which_result)
        print(
            f"WARNING: Using sim_eval from PATH: {which_path}\n"
            f"  This may not support all commands. Consider building from source:\n"
            f"  cargo build --release -p paraphina --bins",
            file=sys.stderr
        )
        return which_path
    
    # No binary found - raise helpful error
    raise SimEvalNotFoundError(
        f"sim_eval binary not found.\n"
        f"\n"
        f"Phase AB requires the repo's sim_eval binary for evidence verification.\n"
        f"Build it with:\n"
        f"\n"
        f"    cargo build --release -p paraphina --bins\n"
        f"\n"
        f"Or set the SIM_EVAL_BIN environment variable to point to a compatible binary.\n"
        f"\n"
        f"Checked locations:\n"
        f"  - SIM_EVAL_BIN env var: {env_bin or '(not set)'}\n"
        f"  - {release_bin}\n"
        f"  - {debug_bin}\n"
        f"  - PATH: (not found)"
    )


def verify_evidence_pack(run_root: Path, verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    Verify evidence pack at run_root if it exists.
    
    Calls `sim_eval verify-evidence-pack <run_root>` if evidence_pack/SHA256SUMS exists.
    
    Args:
        run_root: Run directory that may contain evidence_pack/
        verbose: Print verification output
        
    Returns:
        (success, errors) tuple
        - success is True if verification passed or no evidence pack exists
        - errors is list of error messages if verification failed
    """
    evidence_pack = run_root / "evidence_pack"
    sha256sums = evidence_pack / "SHA256SUMS"
    
    if not sha256sums.exists():
        if verbose:
            print(f"  No evidence pack found at {run_root}")
        return True, []  # No evidence pack = nothing to verify
    
    # Find sim_eval binary
    try:
        sim_eval = find_sim_eval_bin(verbose=verbose)
    except SimEvalNotFoundError as e:
        return False, [str(e)]
    
    # Build command: sim_eval verify-evidence-pack <run_root>
    cmd = [str(sim_eval), "verify-evidence-pack", str(run_root)]
    
    if verbose:
        print(f"  Verifying evidence pack: {run_root}")
        print(f"    Command: {' '.join(cmd)}")
    
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        
        if proc.returncode == 0:
            if verbose:
                print(f"  ✓ Evidence verified")
            return True, []
        else:
            errors = []
            if proc.stderr:
                errors.append(f"stderr: {proc.stderr.strip()}")
            if proc.stdout:
                errors.append(f"stdout: {proc.stdout.strip()}")
            if not errors:
                errors.append(f"verify-evidence-pack failed (exit code {proc.returncode})")
            
            # Add helpful context
            errors.append(f"Command: {' '.join(cmd)}")
            errors.append(f"Binary: {sim_eval}")
            return False, errors
            
    except FileNotFoundError as e:
        return False, [
            f"Failed to execute sim_eval: {e}",
            f"Binary path: {sim_eval}",
            "Try rebuilding: cargo build --release -p paraphina --bins"
        ]
    except Exception as e:
        return False, [f"Evidence verification error: {e}"]


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class PhaseABManifest:
    """
    Canonical manifest for Phase AB run.
    
    Contains all metadata about the run, paths to outputs, and decision.
    """
    candidate_run_resolved: str
    baseline_run_resolved: Optional[str]
    phase_b_out_dir: str
    decision: str  # PROMOTE, HOLD, REJECT, ERROR
    alpha: float
    bootstrap_samples: int
    seed: int
    confidence_report_json: str
    confidence_report_md: str
    timestamp: str
    git_commit: Optional[str] = None
    candidate_samples: int = 0
    baseline_samples: int = 0
    evidence_verified_candidate: bool = False
    evidence_verified_baseline: bool = False
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": 1,
            "candidate_run_resolved": self.candidate_run_resolved,
            "baseline_run_resolved": self.baseline_run_resolved,
            "phase_b_out_dir": self.phase_b_out_dir,
            "decision": self.decision,
            "alpha": self.alpha,
            "bootstrap_samples": self.bootstrap_samples,
            "seed": self.seed,
            "confidence_report_json": self.confidence_report_json,
            "confidence_report_md": self.confidence_report_md,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "candidate_samples": self.candidate_samples,
            "baseline_samples": self.baseline_samples,
            "evidence_verified_candidate": self.evidence_verified_candidate,
            "evidence_verified_baseline": self.evidence_verified_baseline,
            "errors": self.errors,
        }


@dataclass
class PhaseABResult:
    """
    Result from Phase AB orchestration.
    
    Contains the manifest, exit code, and any error information.
    """
    manifest: PhaseABManifest
    exit_code: int
    phase_b_decision: Optional[Any] = None  # PromotionDecision when available
    
    @property
    def decision(self) -> str:
        """Get the decision string."""
        return self.manifest.decision


# =============================================================================
# Git Commit Helper
# =============================================================================

def _get_git_commit() -> Optional[str]:
    """Get the current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# =============================================================================
# Main Orchestrator
# =============================================================================

def run_phase_ab(
    candidate_run: Path,
    baseline_run: Optional[Path] = None,
    out_dir: Path = Path("runs/phaseAB_output"),
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    seed: int = 42,
    skip_evidence_verify: bool = False,
    verbose: bool = True,
) -> PhaseABResult:
    """
    Run Phase AB: Integrated Phase A → Phase B orchestration.
    
    Steps:
    1. Resolve candidate run root (validate and normalize)
    2. Resolve baseline run root (if provided)
    3. Verify evidence packs (unless --skip-evidence-verify)
    4. Run Phase B confidence gating
    5. Write manifest and outputs
    
    Args:
        candidate_run: Path to candidate Phase A run directory
        baseline_run: Path to baseline Phase A run directory (optional)
        out_dir: Output directory for Phase AB results
        alpha: Significance level for confidence intervals
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility
        skip_evidence_verify: Skip evidence pack verification
        verbose: Print progress
        
    Returns:
        PhaseABResult with manifest, exit code, and decision
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    errors: List[str] = []
    
    if verbose:
        print("=" * 70)
        print("Phase AB: Integrated Phase A → Phase B Orchestrator")
        print("=" * 70)
        print(f"  Candidate: {candidate_run}")
        if baseline_run:
            print(f"  Baseline: {baseline_run}")
        print(f"  Output: {out_dir}")
        print(f"  Alpha: {alpha}")
        print(f"  Bootstrap samples: {n_bootstrap}")
        print(f"  Seed: {seed}")
        print(f"  Skip evidence verify: {skip_evidence_verify}")
        print()
    
    # Ensure output directory exists
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get git commit
    git_commit = _get_git_commit()
    
    # =========================================================================
    # Step 1: Resolve candidate run root
    # =========================================================================
    if verbose:
        print("[1/4] Resolving candidate run root...")
    
    try:
        candidate_resolved = resolve_run_root(candidate_run, verbose=verbose)
    except RunRootError as e:
        error_msg = str(e)
        print(f"ERROR: {error_msg}", file=sys.stderr)
        
        manifest = PhaseABManifest(
            candidate_run_resolved=str(candidate_run),
            baseline_run_resolved=None,
            phase_b_out_dir=str(out_dir),
            decision="ERROR",
            alpha=alpha,
            bootstrap_samples=n_bootstrap,
            seed=seed,
            confidence_report_json="",
            confidence_report_md="",
            timestamp=timestamp,
            git_commit=git_commit,
            errors=[error_msg],
        )
        _write_manifest(out_dir, manifest)
        return PhaseABResult(manifest=manifest, exit_code=3)
    
    # =========================================================================
    # Step 2: Resolve baseline run root (if provided)
    # =========================================================================
    baseline_resolved: Optional[Path] = None
    if baseline_run:
        if verbose:
            print("\n[2/4] Resolving baseline run root...")
        
        try:
            baseline_resolved = resolve_run_root(baseline_run, verbose=verbose)
        except RunRootError as e:
            error_msg = str(e)
            print(f"ERROR: {error_msg}", file=sys.stderr)
            
            manifest = PhaseABManifest(
                candidate_run_resolved=str(candidate_resolved),
                baseline_run_resolved=str(baseline_run),
                phase_b_out_dir=str(out_dir),
                decision="ERROR",
                alpha=alpha,
                bootstrap_samples=n_bootstrap,
                seed=seed,
                confidence_report_json="",
                confidence_report_md="",
                timestamp=timestamp,
                git_commit=git_commit,
                errors=[error_msg],
            )
            _write_manifest(out_dir, manifest)
            return PhaseABResult(manifest=manifest, exit_code=3)
    else:
        if verbose:
            print("\n[2/4] No baseline provided, skipping...")
    
    # =========================================================================
    # Step 3: Verify evidence packs
    # =========================================================================
    evidence_verified_candidate = False
    evidence_verified_baseline = False
    
    if not skip_evidence_verify:
        if verbose:
            print("\n[3/4] Verifying evidence packs...")
        
        # Verify candidate
        success, errs = verify_evidence_pack(candidate_resolved, verbose=verbose)
        evidence_verified_candidate = success
        if not success:
            errors.extend(errs)
            print(f"ERROR: Candidate evidence verification failed:", file=sys.stderr)
            for err in errs:
                print(f"  {err}", file=sys.stderr)
            
            manifest = PhaseABManifest(
                candidate_run_resolved=str(candidate_resolved),
                baseline_run_resolved=str(baseline_resolved) if baseline_resolved else None,
                phase_b_out_dir=str(out_dir),
                decision="ERROR",
                alpha=alpha,
                bootstrap_samples=n_bootstrap,
                seed=seed,
                confidence_report_json="",
                confidence_report_md="",
                timestamp=timestamp,
                git_commit=git_commit,
                errors=errors,
            )
            _write_manifest(out_dir, manifest)
            return PhaseABResult(manifest=manifest, exit_code=3)
        
        # Verify baseline
        if baseline_resolved:
            success, errs = verify_evidence_pack(baseline_resolved, verbose=verbose)
            evidence_verified_baseline = success
            if not success:
                errors.extend(errs)
                print(f"ERROR: Baseline evidence verification failed:", file=sys.stderr)
                for err in errs:
                    print(f"  {err}", file=sys.stderr)
                
                manifest = PhaseABManifest(
                    candidate_run_resolved=str(candidate_resolved),
                    baseline_run_resolved=str(baseline_resolved),
                    phase_b_out_dir=str(out_dir),
                    decision="ERROR",
                    alpha=alpha,
                    bootstrap_samples=n_bootstrap,
                    seed=seed,
                    confidence_report_json="",
                    confidence_report_md="",
                    timestamp=timestamp,
                    git_commit=git_commit,
                    errors=errors,
                )
                _write_manifest(out_dir, manifest)
                return PhaseABResult(manifest=manifest, exit_code=3)
    else:
        if verbose:
            print("\n[3/4] Skipping evidence verification (--skip-evidence-verify)")
        evidence_verified_candidate = True
        evidence_verified_baseline = baseline_resolved is not None
    
    # =========================================================================
    # Step 4: Run Phase B confidence gating
    # =========================================================================
    if verbose:
        print("\n[4/4] Running Phase B confidence gating...")
    
    # Import Phase B
    try:
        from batch_runs.phase_b.cli import run_gate
        from batch_runs.phase_b.gate import PromotionOutcome
    except ImportError as e:
        error_msg = f"Failed to import Phase B module: {e}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        
        manifest = PhaseABManifest(
            candidate_run_resolved=str(candidate_resolved),
            baseline_run_resolved=str(baseline_resolved) if baseline_resolved else None,
            phase_b_out_dir=str(out_dir),
            decision="ERROR",
            alpha=alpha,
            bootstrap_samples=n_bootstrap,
            seed=seed,
            confidence_report_json="",
            confidence_report_md="",
            timestamp=timestamp,
            git_commit=git_commit,
            evidence_verified_candidate=evidence_verified_candidate,
            evidence_verified_baseline=evidence_verified_baseline,
            errors=[error_msg],
        )
        _write_manifest(out_dir, manifest)
        return PhaseABResult(manifest=manifest, exit_code=3)
    
    # Run Phase B gate
    try:
        decision = run_gate(
            candidate_run=candidate_resolved,
            baseline_run=baseline_resolved,
            out_dir=out_dir,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            seed=seed,
            verbose=verbose,
        )
    except Exception as e:
        error_msg = f"Phase B gating failed: {e}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        
        manifest = PhaseABManifest(
            candidate_run_resolved=str(candidate_resolved),
            baseline_run_resolved=str(baseline_resolved) if baseline_resolved else None,
            phase_b_out_dir=str(out_dir),
            decision="ERROR",
            alpha=alpha,
            bootstrap_samples=n_bootstrap,
            seed=seed,
            confidence_report_json="",
            confidence_report_md="",
            timestamp=timestamp,
            git_commit=git_commit,
            evidence_verified_candidate=evidence_verified_candidate,
            evidence_verified_baseline=evidence_verified_baseline,
            errors=[error_msg],
        )
        _write_manifest(out_dir, manifest)
        return PhaseABResult(manifest=manifest, exit_code=3)
    
    # =========================================================================
    # Build manifest and write outputs
    # =========================================================================
    confidence_json = out_dir / "confidence_report.json"
    confidence_md = out_dir / "confidence_report.md"
    
    candidate_samples = decision.candidate_metrics.n_runs if decision.candidate_metrics else 0
    baseline_samples = decision.baseline_metrics.n_runs if decision.baseline_metrics else 0
    
    manifest = PhaseABManifest(
        candidate_run_resolved=str(candidate_resolved),
        baseline_run_resolved=str(baseline_resolved) if baseline_resolved else None,
        phase_b_out_dir=str(out_dir),
        decision=decision.outcome.value.upper(),
        alpha=alpha,
        bootstrap_samples=n_bootstrap,
        seed=seed,
        confidence_report_json=str(confidence_json),
        confidence_report_md=str(confidence_md),
        timestamp=timestamp,
        git_commit=git_commit,
        candidate_samples=candidate_samples,
        baseline_samples=baseline_samples,
        evidence_verified_candidate=evidence_verified_candidate,
        evidence_verified_baseline=evidence_verified_baseline,
        errors=errors,
    )
    
    _write_manifest(out_dir, manifest)
    
    # Compute exit code from Phase B decision
    exit_code = decision.exit_code
    
    if verbose:
        print()
        print("=" * 70)
        print(f"Phase AB Complete")
        print("=" * 70)
        print(f"  Decision: {decision.outcome.value.upper()}")
        print(f"  Exit code: {exit_code}")
        print(f"  Manifest: {out_dir / 'phase_ab_manifest.json'}")
        print(f"  Reports: {confidence_json.name}, {confidence_md.name}")
    
    return PhaseABResult(
        manifest=manifest,
        exit_code=exit_code,
        phase_b_decision=decision,
    )


def _write_manifest(out_dir: Path, manifest: PhaseABManifest) -> None:
    """Write the Phase AB manifest to the output directory."""
    manifest_path = out_dir / "phase_ab_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)

