#!/usr/bin/env python3
"""
sim_eval_report.py – Step 6 research reporting + gating.

Scans run root for suite directories matching a prefix, parses run_summary.json
files, computes deltas against a baseline suite, and emits Markdown + CSV reports.

Exit codes:
  0 = success
  1 = kill switch triggered in any suite
  2 = ablation suite missing baseline match for (scenario_id, seed)
  3 = run_summary.json missing required fields
  4 = other error (IO, no suites found, etc.)

Usage:
  python3 tools/sim_eval_report.py \\
      --run-root runs \\
      --baseline-suite research__baseline \\
      --suite-prefix research__ \\
      --out-md _reports/research_report.md \\
      --out-csv _reports/research_report.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple


# =============================================================================
# Data Structures
# =============================================================================

class RunSummary(NamedTuple):
    """Parsed run summary with required fields."""
    scenario_id: str
    seed: int
    ablations: List[str]
    final_pnl_usd: float
    max_drawdown_usd: float
    kill_switch_triggered: bool
    kill_switch_reason: Optional[str]
    checksum: str
    source_path: Path


class SuiteData(NamedTuple):
    """All runs for a single suite."""
    suite_name: str
    runs: List[RunSummary]


class JoinedRun(NamedTuple):
    """A run joined with its baseline counterpart."""
    scenario_id: str
    seed: int
    ablations: List[str]
    pnl: float
    baseline_pnl: float
    delta_pnl: float
    drawdown: float
    baseline_drawdown: float
    delta_drawdown: float
    kill_switch_triggered: bool
    kill_switch_reason: Optional[str]


# =============================================================================
# Parsing
# =============================================================================

REQUIRED_FIELDS = [
    "scenario_id",
    "seed",
    "results.final_pnl_usd",
    "results.max_drawdown_usd",
    "results.kill_switch.triggered",
    "determinism.checksum",
]


def get_nested(d: Dict[str, Any], path: str) -> Any:
    """Get a nested value from a dict using dot notation."""
    keys = path.split(".")
    val = d
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            return None
        val = val[k]
    return val


def parse_run_summary(path: Path) -> Tuple[Optional[RunSummary], Optional[str]]:
    """
    Parse a single run_summary.json file.

    Returns:
        (RunSummary, None) on success
        (None, error_message) on failure
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        return None, f"Failed to parse {path}: {e}"

    # Check required fields
    missing = []
    for field in REQUIRED_FIELDS:
        if get_nested(data, field) is None:
            missing.append(field)

    if missing:
        return None, f"Missing required fields in {path}: {missing}"

    # Extract values
    try:
        scenario_id = data["scenario_id"]
        seed = int(data["seed"])

        # ablations can be at top level or in config; prefer top level
        ablations = data.get("ablations", data.get("config", {}).get("ablations", []))
        if ablations is None:
            ablations = []

        results = data["results"]
        final_pnl = float(results["final_pnl_usd"])
        max_drawdown = float(results["max_drawdown_usd"])

        kill_switch = results.get("kill_switch", {})
        kill_triggered = bool(kill_switch.get("triggered", False))
        kill_reason = kill_switch.get("reason")

        checksum = data["determinism"]["checksum"]

        return RunSummary(
            scenario_id=scenario_id,
            seed=seed,
            ablations=ablations,
            final_pnl_usd=final_pnl,
            max_drawdown_usd=max_drawdown,
            kill_switch_triggered=kill_triggered,
            kill_switch_reason=kill_reason,
            checksum=checksum,
            source_path=path,
        ), None

    except (KeyError, TypeError, ValueError) as e:
        return None, f"Error extracting fields from {path}: {e}"


def discover_run_summaries(suite_dir: Path) -> List[Path]:
    """Recursively find all run_summary.json files under a directory."""
    results = []
    for root, _dirs, files in os.walk(suite_dir):
        for fname in files:
            if fname == "run_summary.json":
                results.append(Path(root) / fname)
    return results


def load_suite(suite_dir: Path) -> Tuple[Optional[SuiteData], List[str]]:
    """
    Load all run summaries from a suite directory.

    Returns:
        (SuiteData, []) on success
        (SuiteData, [errors]) on partial success
        (None, [errors]) on complete failure
    """
    paths = discover_run_summaries(suite_dir)
    if not paths:
        return None, [f"No run_summary.json files found in {suite_dir}"]

    runs = []
    errors = []

    for p in paths:
        run, err = parse_run_summary(p)
        if err:
            errors.append(err)
        elif run:
            runs.append(run)

    if not runs:
        return None, errors

    return SuiteData(suite_name=suite_dir.name, runs=runs), errors


# =============================================================================
# Delta Computation
# =============================================================================

def join_to_baseline(
    ablation_suite: SuiteData,
    baseline_suite: SuiteData,
) -> Tuple[List[JoinedRun], List[str]]:
    """
    Join ablation runs to baseline on (scenario_id, seed).

    Returns:
        (joined_runs, missing_errors) where missing_errors lists unmatched runs
    """
    # Build baseline index
    baseline_idx: Dict[Tuple[str, int], RunSummary] = {}
    for run in baseline_suite.runs:
        key = (run.scenario_id, run.seed)
        baseline_idx[key] = run

    joined = []
    missing = []

    for run in ablation_suite.runs:
        key = (run.scenario_id, run.seed)
        baseline = baseline_idx.get(key)

        if baseline is None:
            missing.append(
                f"Ablation suite '{ablation_suite.suite_name}' run "
                f"(scenario_id={run.scenario_id}, seed={run.seed}) "
                f"has no baseline match"
            )
            continue

        joined.append(JoinedRun(
            scenario_id=run.scenario_id,
            seed=run.seed,
            ablations=run.ablations,
            pnl=run.final_pnl_usd,
            baseline_pnl=baseline.final_pnl_usd,
            delta_pnl=run.final_pnl_usd - baseline.final_pnl_usd,
            drawdown=run.max_drawdown_usd,
            baseline_drawdown=baseline.max_drawdown_usd,
            delta_drawdown=run.max_drawdown_usd - baseline.max_drawdown_usd,
            kill_switch_triggered=run.kill_switch_triggered,
            kill_switch_reason=run.kill_switch_reason,
        ))

    return joined, missing


# =============================================================================
# Statistics
# =============================================================================

def mean(values: List[float]) -> float:
    """Compute mean of a list of floats."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_suite_stats(joined_runs: List[JoinedRun]) -> Dict[str, Any]:
    """Compute aggregate statistics for a suite's joined runs."""
    if not joined_runs:
        return {
            "run_count": 0,
            "mean_delta_pnl": 0.0,
            "min_delta_pnl": 0.0,
            "max_delta_pnl": 0.0,
            "kill_switch_count": 0,
        }

    delta_pnls = [r.delta_pnl for r in joined_runs]
    kill_count = sum(1 for r in joined_runs if r.kill_switch_triggered)

    return {
        "run_count": len(joined_runs),
        "mean_delta_pnl": mean(delta_pnls),
        "min_delta_pnl": min(delta_pnls),
        "max_delta_pnl": max(delta_pnls),
        "kill_switch_count": kill_count,
    }


def compute_per_scenario_stats(
    joined_runs: List[JoinedRun],
) -> List[Dict[str, Any]]:
    """Compute per-scenario statistics."""
    # Group by scenario_id
    by_scenario: Dict[str, List[JoinedRun]] = {}
    for r in joined_runs:
        by_scenario.setdefault(r.scenario_id, []).append(r)

    results = []
    for scenario_id in sorted(by_scenario.keys()):
        runs = by_scenario[scenario_id]
        delta_pnls = [r.delta_pnl for r in runs]
        results.append({
            "scenario_id": scenario_id,
            "n": len(runs),
            "mean_delta_pnl": mean(delta_pnls),
            "min_delta_pnl": min(delta_pnls),
            "max_delta_pnl": max(delta_pnls),
        })

    return results


# =============================================================================
# Report Generation
# =============================================================================

def format_float(v: float, decimals: int = 2) -> str:
    """Format a float with fixed decimal places."""
    return f"{v:.{decimals}f}"


def generate_markdown(
    baseline_suite: SuiteData,
    ablation_results: List[Tuple[SuiteData, List[JoinedRun], Dict[str, Any]]],
) -> str:
    """Generate the full Markdown report."""
    lines = []
    lines.append("# Research Report: Baseline vs Ablations")
    lines.append("")
    lines.append(f"**Baseline suite:** `{baseline_suite.suite_name}`")
    lines.append(f"**Baseline runs:** {len(baseline_suite.runs)}")
    lines.append("")

    # ---------------------------------------------------------------------
    # Overall per-suite table
    # ---------------------------------------------------------------------
    lines.append("## Overall Summary")
    lines.append("")
    lines.append(
        "| Suite | Ablations | Runs | Mean ΔPnL | Min ΔPnL | Max ΔPnL | Kill-Switch |"
    )
    lines.append(
        "|-------|-----------|------|-----------|----------|----------|-------------|"
    )

    for suite, joined, stats in ablation_results:
        ablations_str = ", ".join(suite.runs[0].ablations) if suite.runs else ""
        lines.append(
            f"| {suite.suite_name} "
            f"| {ablations_str} "
            f"| {stats['run_count']} "
            f"| {format_float(stats['mean_delta_pnl'])} "
            f"| {format_float(stats['min_delta_pnl'])} "
            f"| {format_float(stats['max_delta_pnl'])} "
            f"| {stats['kill_switch_count']} |"
        )

    lines.append("")

    # ---------------------------------------------------------------------
    # Per-suite per-scenario tables
    # ---------------------------------------------------------------------
    lines.append("## Per-Suite Breakdown")
    lines.append("")

    for suite, joined, _stats in ablation_results:
        lines.append(f"### {suite.suite_name}")
        lines.append("")

        per_scenario = compute_per_scenario_stats(joined)
        if not per_scenario:
            lines.append("_No matched runs._")
            lines.append("")
            continue

        lines.append("| scenario_id | n | Mean ΔPnL | Min ΔPnL | Max ΔPnL |")
        lines.append("|-------------|---|-----------|----------|----------|")

        for row in per_scenario:
            lines.append(
                f"| {row['scenario_id']} "
                f"| {row['n']} "
                f"| {format_float(row['mean_delta_pnl'])} "
                f"| {format_float(row['min_delta_pnl'])} "
                f"| {format_float(row['max_delta_pnl'])} |"
            )

        lines.append("")

    return "\n".join(lines)


def generate_csv(
    ablation_results: List[Tuple[SuiteData, List[JoinedRun], Dict[str, Any]]],
) -> str:
    """Generate CSV with all joined run data."""
    lines = []
    header = [
        "suite",
        "scenario_id",
        "seed",
        "ablations",
        "pnl",
        "baseline_pnl",
        "delta_pnl",
        "drawdown",
        "baseline_drawdown",
        "delta_drawdown",
        "kill_switch_triggered",
        "kill_switch_reason",
    ]
    lines.append(",".join(header))

    for suite, joined, _stats in ablation_results:
        for r in joined:
            # Quote ablations list and reason to handle commas
            ablations_str = ";".join(r.ablations)
            reason_str = r.kill_switch_reason or ""
            # Escape quotes in reason
            reason_str = reason_str.replace('"', '""')

            row = [
                suite.suite_name,
                r.scenario_id,
                str(r.seed),
                f'"{ablations_str}"',
                format_float(r.pnl),
                format_float(r.baseline_pnl),
                format_float(r.delta_pnl),
                format_float(r.drawdown),
                format_float(r.baseline_drawdown),
                format_float(r.delta_drawdown),
                str(r.kill_switch_triggered).lower(),
                f'"{reason_str}"',
            ]
            lines.append(",".join(row))

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate research report comparing ablation suites to baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-root",
        default="runs",
        help="Root directory containing suite subdirectories (default: runs)",
    )
    parser.add_argument(
        "--baseline-suite",
        default="research__baseline",
        help="Name of the baseline suite directory (default: research__baseline)",
    )
    parser.add_argument(
        "--suite-prefix",
        default="research__",
        help="Prefix to match suite directories (default: research__)",
    )
    parser.add_argument(
        "--out-md",
        default="_reports/research_report.md",
        help="Output path for Markdown report (default: _reports/research_report.md)",
    )
    parser.add_argument(
        "--out-csv",
        default="_reports/research_report.csv",
        help="Output path for CSV report (default: _reports/research_report.csv)",
    )

    args = parser.parse_args()

    run_root = Path(args.run_root)
    baseline_name = args.baseline_suite
    suite_prefix = args.suite_prefix
    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)

    # Validate run root exists
    if not run_root.is_dir():
        print(f"ERROR: Run root directory not found: {run_root}", file=sys.stderr)
        return 4

    # Discover suite directories
    try:
        all_dirs = sorted(
            d for d in run_root.iterdir()
            if d.is_dir() and d.name.startswith(suite_prefix)
        )
    except IOError as e:
        print(f"ERROR: Failed to list run root: {e}", file=sys.stderr)
        return 4

    if not all_dirs:
        print(
            f"ERROR: No suite directories matching '{suite_prefix}*' in {run_root}",
            file=sys.stderr,
        )
        return 4

    print(f"Found {len(all_dirs)} suite(s) matching prefix '{suite_prefix}':")
    for d in all_dirs:
        print(f"  - {d.name}")

    # Load baseline suite
    baseline_dir = run_root / baseline_name
    if not baseline_dir.is_dir():
        print(f"ERROR: Baseline suite not found: {baseline_dir}", file=sys.stderr)
        return 4

    baseline_suite, baseline_errors = load_suite(baseline_dir)
    if baseline_suite is None:
        print(f"ERROR: Failed to load baseline suite:", file=sys.stderr)
        for e in baseline_errors:
            print(f"  {e}", file=sys.stderr)
        return 3

    # Report any parse errors but continue
    for e in baseline_errors:
        print(f"WARNING (baseline): {e}", file=sys.stderr)

    print(f"Loaded baseline: {len(baseline_suite.runs)} runs")

    # Check for kill switches in baseline
    baseline_kill_count = sum(
        1 for r in baseline_suite.runs if r.kill_switch_triggered
    )
    if baseline_kill_count > 0:
        print(
            f"ERROR: {baseline_kill_count} kill switch(es) triggered in baseline!",
            file=sys.stderr,
        )

    # Load and process ablation suites
    ablation_results: List[Tuple[SuiteData, List[JoinedRun], Dict[str, Any]]] = []
    all_errors: List[str] = []
    total_kill_count = baseline_kill_count
    missing_baseline_errors: List[str] = []
    field_errors: List[str] = []

    for suite_dir in all_dirs:
        if suite_dir.name == baseline_name:
            continue  # Skip baseline

        suite, errors = load_suite(suite_dir)
        if suite is None:
            # Treat as fatal field error if no runs could be parsed
            all_errors.extend(errors)
            field_errors.extend(errors)
            continue

        # Report any parse errors
        for e in errors:
            print(f"WARNING ({suite_dir.name}): {e}", file=sys.stderr)
            # If error mentions "Missing required fields", track it
            if "Missing required fields" in e:
                field_errors.append(e)

        # Join to baseline
        joined, missing = join_to_baseline(suite, baseline_suite)
        missing_baseline_errors.extend(missing)

        # Compute stats
        stats = compute_suite_stats(joined)
        total_kill_count += stats["kill_switch_count"]

        ablation_results.append((suite, joined, stats))
        print(
            f"Suite {suite.suite_name}: {len(joined)} matched runs, "
            f"{stats['kill_switch_count']} kills"
        )

    # Generate reports
    print(f"\nGenerating reports...")

    # Ensure output directories exist
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    md_content = generate_markdown(baseline_suite, ablation_results)
    csv_content = generate_csv(ablation_results)

    try:
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Wrote: {out_md}")

        with open(out_csv, "w", encoding="utf-8") as f:
            f.write(csv_content)
        print(f"Wrote: {out_csv}")
    except IOError as e:
        print(f"ERROR: Failed to write output: {e}", file=sys.stderr)
        return 4

    # Determine exit code based on gating rules
    print("\n=== Gating Checks ===")

    # Check 1: Any kill switch triggers
    if total_kill_count > 0:
        print(
            f"FAIL: {total_kill_count} kill switch(es) triggered across all suites",
            file=sys.stderr,
        )
        return 1

    print("PASS: No kill switches triggered")

    # Check 2: Missing baseline matches
    if missing_baseline_errors:
        print("FAIL: Missing baseline matches:", file=sys.stderr)
        for e in missing_baseline_errors:
            print(f"  {e}", file=sys.stderr)
        return 2

    print("PASS: All ablation runs have baseline matches")

    # Check 3: Missing required fields (already caught during parsing)
    if field_errors:
        print("FAIL: Missing required fields in run summaries:", file=sys.stderr)
        for e in field_errors:
            print(f"  {e}", file=sys.stderr)
        return 3

    print("PASS: All run summaries have required fields")

    print("\n=== SUCCESS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

