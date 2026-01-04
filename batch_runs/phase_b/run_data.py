"""
run_data.py

Canonical run-data discovery and loader for Phase A outputs used by Phase B.

This module provides robust discovery and parsing of Phase A trial outputs:
- Discovers trials.jsonl files with configurable precedence
- Parses JSONL format with resilient metric extraction
- Produces observations suitable for confidence gating
- Provides actionable error messages with detailed diagnostics

Discovery precedence:
1. If run_root is a file ending in .jsonl, treat it as trials JSONL
2. Else if <run_root>/trials.jsonl exists, use it
3. Else search within <run_root> for **/trials.jsonl and pick the shallowest match
   (fewest path components), breaking ties by newest mtime
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Exceptions
# =============================================================================


class RunDataError(Exception):
    """Base exception for run data loading errors."""

    pass


@dataclass
class NoUsableObservationsError(RunDataError):
    """Raised when no usable observations could be extracted from trials data."""

    trials_file: Path
    lines_parsed: int
    rejection_counts: Dict[str, int]
    message: str = ""

    def __post_init__(self):
        if not self.message:
            reasons = ", ".join(
                f"{reason}: {count}" for reason, count in self.rejection_counts.items()
            )
            self.message = (
                f"No usable observations found.\n"
                f"  File: {self.trials_file}\n"
                f"  Lines parsed: {self.lines_parsed}\n"
                f"  Rejections: {reasons if reasons else 'none'}"
            )

    def __str__(self) -> str:
        return self.message


@dataclass
class TrialsFileNotFoundError(RunDataError):
    """Raised when no trials.jsonl file could be discovered."""

    run_root: Path
    message: str = ""

    def __post_init__(self):
        if not self.message:
            self.message = (
                f"No trials.jsonl found in {self.run_root}\n"
                f"  Searched: {self.run_root}/trials.jsonl and **/trials.jsonl"
            )

    def __str__(self) -> str:
        return self.message


# =============================================================================
# Observation dataclass
# =============================================================================


@dataclass
class TrialObservation:
    """
    A single trial observation extracted from Phase A output.

    Contains the numeric metrics needed for Phase B confidence gating.
    """

    pnl: float
    kill_ci: float  # Kill probability upper confidence bound
    dd_cvar: Optional[float] = None  # Drawdown CVaR (optional)
    kill_rate: float = 0.0  # Point estimate of kill rate
    source_line: int = 0  # Line number in source file (1-indexed)

    def to_pnl_list(self) -> List[float]:
        """Return PnL as a single-element list (for compatibility with gate)."""
        return [self.pnl]

    @property
    def is_killed(self) -> bool:
        """Return True if this trial was killed (kill_rate > 0)."""
        return self.kill_rate > 0


# =============================================================================
# Discovery functions
# =============================================================================


def discover_trials_file(run_root: Path) -> Path:
    """
    Discover the trials.jsonl file for a Phase A run.

    Discovery precedence:
    1. If run_root is a file ending in .jsonl, treat it as trials JSONL
    2. Else if <run_root>/trials.jsonl exists, use it
    3. Else search within <run_root> for **/trials.jsonl and pick the shallowest
       match (fewest path components), breaking ties by newest mtime

    Args:
        run_root: Path to run directory or JSONL file

    Returns:
        Path to discovered trials.jsonl file

    Raises:
        TrialsFileNotFoundError: If no trials file could be found
    """
    run_root = Path(run_root)

    # Case 1: run_root is a JSONL file
    if run_root.is_file() and run_root.suffix == ".jsonl":
        return run_root

    # Case 2: Direct trials.jsonl in run_root
    direct_path = run_root / "trials.jsonl"
    if direct_path.exists():
        return direct_path

    # Case 3: Search recursively for **/trials.jsonl
    if not run_root.is_dir():
        raise TrialsFileNotFoundError(run_root)

    matches = list(run_root.rglob("trials.jsonl"))
    if not matches:
        raise TrialsFileNotFoundError(run_root)

    # Sort by: (path depth, -mtime) to get shallowest first, then newest
    def sort_key(p: Path) -> Tuple[int, float]:
        # Count path components relative to run_root
        try:
            relative = p.relative_to(run_root)
            depth = len(relative.parts) - 1  # Subtract 1 for the filename itself
        except ValueError:
            depth = 999
        # Negative mtime so newer files come first
        try:
            mtime = -p.stat().st_mtime
        except OSError:
            mtime = 0
        return (depth, mtime)

    matches.sort(key=sort_key)
    return matches[0]


# =============================================================================
# Metric extraction
# =============================================================================

# Valid status values that indicate an accepted trial
VALID_STATUS_VALUES = frozenset({"OK", "PASS", "VALID", "ok", "pass", "valid", True})


def _extract_metric(
    record: Dict[str, Any], key: str, fallback_keys: Optional[List[str]] = None
) -> Optional[float]:
    """
    Extract a numeric metric from a trial record with resilient key lookup.

    Lookup order:
    1. Top-level key
    2. metrics.<key>
    3. result.<key>
    4. Fallback keys (if provided)

    Args:
        record: The trial record dictionary
        key: Primary key to look for
        fallback_keys: Additional keys to try if primary not found

    Returns:
        The numeric value, or None if not found or not numeric
    """
    # Try top-level
    if key in record:
        val = record[key]
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)

    # Try metrics.<key>
    metrics = record.get("metrics", {})
    if isinstance(metrics, dict) and key in metrics:
        val = metrics[key]
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)

    # Try result.<key>
    result = record.get("result", {})
    if isinstance(result, dict) and key in result:
        val = result[key]
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)

    # Try fallback keys
    if fallback_keys:
        for fk in fallback_keys:
            val = _extract_metric(record, fk)
            if val is not None:
                return val

    return None


def _is_valid_trial(record: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if a trial record is valid for inclusion.

    A trial is valid if:
    - No status field exists (accept by default), OR
    - status field has a valid value (OK, PASS, VALID, True)

    Additionally checks:
    - is_valid field if present (must be True or truthy)

    Args:
        record: The trial record dictionary

    Returns:
        Tuple of (is_valid, rejection_reason)
    """
    # Check is_valid field first
    if "is_valid" in record:
        if not record["is_valid"]:
            return False, "is_valid=False"

    # Check status field
    if "status" in record:
        status = record["status"]
        if status not in VALID_STATUS_VALUES:
            return False, f"status={status}"

    return True, ""


# =============================================================================
# JSONL parsing
# =============================================================================


def parse_trials_jsonl(
    trials_file: Path,
) -> Tuple[List[TrialObservation], Dict[str, int]]:
    """
    Parse a trials.jsonl file into observations.

    Args:
        trials_file: Path to the JSONL file

    Returns:
        Tuple of (observations, rejection_counts)
        rejection_counts maps rejection reason to count

    Raises:
        FileNotFoundError: If trials_file doesn't exist
    """
    observations: List[TrialObservation] = []
    rejection_counts: Dict[str, int] = {}

    with open(trials_file, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                # Skip blank lines
                continue

            # Parse JSON
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                reason = "json_parse_error"
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue

            if not isinstance(record, dict):
                reason = "not_a_dict"
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue

            # Check validity
            is_valid, reject_reason = _is_valid_trial(record)
            if not is_valid:
                rejection_counts[reject_reason] = (
                    rejection_counts.get(reject_reason, 0) + 1
                )
                continue

            # Extract PnL (required)
            pnl = _extract_metric(
                record,
                "pnl",
                fallback_keys=[
                    "pnl_mean",
                    "mc_mean_pnl",
                    "mean_pnl",
                    "final_pnl",
                ],
            )
            if pnl is None:
                reason = "missing_pnl"
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue

            # Extract kill_ci (required) - try multiple keys
            kill_ci = _extract_metric(
                record,
                "kill_ci",
                fallback_keys=[
                    "kill_ucb",
                    "mc_kill_prob_ci_upper",
                    "kill_prob_ci_upper",
                ],
            )
            if kill_ci is None:
                # Try to compute from kill_rate if available
                kill_rate = _extract_metric(
                    record,
                    "kill_rate",
                    fallback_keys=["mc_kill_prob_point", "kill_prob_point"],
                )
                if kill_rate is not None:
                    # Use kill_rate as a fallback for kill_ci
                    kill_ci = kill_rate
                else:
                    reason = "missing_kill_ci"
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    continue

            # Extract kill_rate (optional, default to 0)
            kill_rate = _extract_metric(
                record,
                "kill_rate",
                fallback_keys=["mc_kill_prob_point", "kill_prob_point"],
            )
            if kill_rate is None:
                kill_rate = 0.0

            # Extract dd_cvar (optional)
            dd_cvar = _extract_metric(
                record,
                "dd_cvar",
                fallback_keys=[
                    "mc_drawdown_cvar",
                    "drawdown_cvar",
                ],
            )

            observations.append(
                TrialObservation(
                    pnl=pnl,
                    kill_ci=kill_ci,
                    dd_cvar=dd_cvar,
                    kill_rate=kill_rate,
                    source_line=line_num,
                )
            )

    return observations, rejection_counts


# =============================================================================
# Main loader API
# =============================================================================


@dataclass
class LoadedRunData:
    """
    Container for loaded run data from Phase A trials.

    Attributes:
        observations: List of trial observations
        trials_file: Path to the source trials.jsonl file
        lines_parsed: Total number of non-blank lines parsed
        rejection_counts: Map of rejection reason to count
    """

    observations: List[TrialObservation]
    trials_file: Path
    lines_parsed: int
    rejection_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def n_observations(self) -> int:
        """Number of usable observations."""
        return len(self.observations)

    def to_pnl_arrays(self) -> List[List[float]]:
        """Convert observations to list of PnL lists (for gate compatibility)."""
        return [obs.to_pnl_list() for obs in self.observations]

    def to_kill_flags(self) -> List[bool]:
        """Convert observations to list of kill flags (for gate compatibility)."""
        return [obs.is_killed for obs in self.observations]


def load_run_data(run_root: Path) -> LoadedRunData:
    """
    Load run data from a Phase A run directory or JSONL file.

    This is the main entry point for loading Phase A trial data for Phase B gating.

    Args:
        run_root: Path to run directory or JSONL file

    Returns:
        LoadedRunData with observations and metadata

    Raises:
        TrialsFileNotFoundError: If no trials.jsonl could be discovered
        NoUsableObservationsError: If no valid observations could be extracted
    """
    run_root = Path(run_root)

    # Discover trials file
    trials_file = discover_trials_file(run_root)

    # Parse JSONL
    observations, rejection_counts = parse_trials_jsonl(trials_file)

    # Count total lines parsed
    lines_parsed = sum(rejection_counts.values()) + len(observations)

    # Check for zero observations
    if not observations:
        raise NoUsableObservationsError(
            trials_file=trials_file,
            lines_parsed=lines_parsed,
            rejection_counts=rejection_counts,
        )

    return LoadedRunData(
        observations=observations,
        trials_file=trials_file,
        lines_parsed=lines_parsed,
        rejection_counts=rejection_counts,
    )

