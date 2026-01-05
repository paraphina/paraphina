#!/usr/bin/env python3
"""
sim_eval_report.py – Quant-grade research reporting with paired delta analysis.

Generates deterministic, audit-friendly research reports comparing ablation
variants to baseline runs at the (scenario_id, seed) pair level.

Features:
  - Paired delta computation: Δ = variant - baseline for matched (scenario_id, seed)
  - Quant-grade summary stats: mean, median, stddev, p05/p95, win/loss/flat rates
  - Deterministic output: sorted lexicographically, rounded to 2 decimals for rendering
  - Missing data handling: graceful handling of missing baseline/variant pairs
  - Duplicate handling: deterministic tie-breaking by checksum then path
  - Machine-readable JSON + human-readable Markdown outputs
  - GitHub Step Summary with delta overview table

Exit codes:
  0 = success
  1 = kill switch triggered in any run
  2 = critical errors (missing run root, no runs found)

Usage:
  python3 tools/sim_eval_report.py \\
      --run-root runs \\
      --out-dir _reports

  Or legacy mode:
  python3 tools/sim_eval_report.py \\
      --run-root runs \\
      --baseline-suite research__baseline \\
      --suite-prefix research__ \\
      --out-md _reports/research_report.md \\
      --out-csv _reports/research_report.csv
"""

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Set


# =============================================================================
# Constants
# =============================================================================

REPORT_SCHEMA_VERSION = 2
ROUNDING_DECIMALS = 2  # For float equality comparisons


# =============================================================================
# Data Structures
# =============================================================================

class RunSummary(NamedTuple):
    """Parsed run summary with all required fields."""
    scenario_id: str
    seed: int
    ablations: Tuple[str, ...]  # Immutable for hashing
    final_pnl_usd: float
    max_drawdown_usd: float
    kill_switch_triggered: bool
    kill_switch_reason: Optional[str]
    checksum: str
    source_path: Path


class PairedDelta(NamedTuple):
    """Delta between variant and baseline for a single (scenario_id, seed) pair."""
    scenario_id: str
    seed: int
    baseline_pnl: float
    variant_pnl: float
    delta_pnl: float
    baseline_drawdown: float
    variant_drawdown: float
    delta_drawdown: float
    baseline_kill_switch: bool
    variant_kill_switch: bool
    delta_kill_switch: int  # 0, +1 (false→true), or -1 (true→false)
    baseline_path: str
    variant_path: str
    baseline_checksum: str
    variant_checksum: str


class SummaryStats(NamedTuple):
    """Quant-grade summary statistics for a variant group."""
    n_pairs: int
    # Delta PnL stats
    mean_delta_pnl: float
    median_delta_pnl: float
    stddev_delta_pnl: float
    p05_delta_pnl: float
    p95_delta_pnl: float
    win_rate: float  # fraction where delta_pnl > 0
    loss_rate: float  # fraction where delta_pnl < 0
    flat_rate: float  # fraction where delta_pnl == 0 (after rounding)
    # Delta drawdown stats
    mean_delta_drawdown: float
    median_delta_drawdown: float
    p05_delta_drawdown: float
    p95_delta_drawdown: float
    # Kill switch delta rates
    kill_switch_false_to_true_rate: float  # % where kill flipped false→true
    kill_switch_true_to_false_rate: float  # % where kill flipped true→false


class VariantReport(NamedTuple):
    """Complete report for a single variant group."""
    variant_key: str
    ablations: Tuple[str, ...]
    stats: SummaryStats
    paired_rows: List[PairedDelta]
    missing_baseline: List[RunSummary]  # Variant runs with no baseline match
    missing_variant: List[Tuple[str, int]]  # Baseline (scenario_id, seed) with no variant


class ResearchReport(NamedTuple):
    """Complete research report with all variant comparisons."""
    metadata: Dict[str, Any]
    baseline_rows: List[RunSummary]
    variant_reports: List[VariantReport]
    all_kill_switches: List[Tuple[str, str, int, str]]  # (variant, scenario_id, seed, reason)


# =============================================================================
# Parsing
# =============================================================================

def get_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get a nested value from a dict using dot notation."""
    keys = path.split(".")
    val = d
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            return default
        val = val[k]
    return val


def parse_run_summary(path: Path) -> Optional[RunSummary]:
    """
    Parse a single run_summary.json file.
    Returns None if parsing fails.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError):
        return None

    try:
        scenario_id = data.get("scenario_id", "")
        seed = int(data.get("seed", 0))

        # ablations: prefer top-level, fallback to config.ablations
        ablations = data.get("ablations", data.get("config", {}).get("ablations", []))
        if ablations is None:
            ablations = []
        ablations = tuple(sorted(ablations))  # Normalize order

        results = data.get("results", {})
        final_pnl = float(results.get("final_pnl_usd", 0.0))
        max_drawdown = float(results.get("max_drawdown_usd", 0.0))

        kill_switch = results.get("kill_switch", {})
        kill_triggered = bool(kill_switch.get("triggered", False))
        kill_reason = kill_switch.get("reason")

        # Checksum: handle gracefully if missing
        checksum = get_nested(data, "determinism.checksum", "missing")

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
        )
    except (KeyError, TypeError, ValueError):
        return None


def discover_run_summaries(root_dir: Path) -> List[Path]:
    """Recursively find all run_summary.json files under a directory."""
    results = []
    for root, _dirs, files in os.walk(root_dir):
        for fname in files:
            if fname == "run_summary.json":
                results.append(Path(root) / fname)
    return sorted(results)  # Deterministic order


def load_all_runs(run_root: Path) -> List[RunSummary]:
    """Load all run summaries from the run root directory."""
    paths = discover_run_summaries(run_root)
    runs = []
    for p in paths:
        run = parse_run_summary(p)
        if run:
            runs.append(run)
    return runs


# =============================================================================
# Baseline/Variant Identification
# =============================================================================

def is_baseline_run(run: RunSummary) -> bool:
    """
    Determine if a run is a baseline run.
    Baseline = empty ablations list OR directory path contains 'baseline'.
    Prefer ablations == () as canonical.
    """
    if len(run.ablations) == 0:
        return True
    # Also check directory path as fallback
    path_str = str(run.source_path).lower()
    return "baseline" in path_str


def make_variant_key(ablations: Tuple[str, ...]) -> str:
    """Create a variant group key from sorted ablations joined by '+'."""
    if not ablations:
        return "baseline"
    return "+".join(ablations)


def group_runs_by_variant(runs: List[RunSummary]) -> Dict[str, List[RunSummary]]:
    """Group runs by their variant key (ablations joined by '+')."""
    groups: Dict[str, List[RunSummary]] = {}
    for run in runs:
        key = make_variant_key(run.ablations)
        if key not in groups:
            groups[key] = []
        groups[key].append(run)
    return groups


# =============================================================================
# Duplicate Handling
# =============================================================================

def select_canonical_run(runs: List[RunSummary]) -> RunSummary:
    """
    Select canonical run when multiple runs exist for same (scenario_id, seed).
    
    Preference:
    1. Lexicographically smallest checksum (for stability)
    2. Lexicographically smallest source path (tie-breaker)
    """
    if len(runs) == 1:
        return runs[0]
    
    # Sort by (checksum, path) lexicographically
    runs_sorted = sorted(runs, key=lambda r: (r.checksum, str(r.source_path)))
    return runs_sorted[0]


def build_run_index(runs: List[RunSummary]) -> Dict[Tuple[str, int], RunSummary]:
    """
    Build an index of runs by (scenario_id, seed).
    Handles duplicates by selecting canonical run.
    """
    # Group by (scenario_id, seed)
    by_key: Dict[Tuple[str, int], List[RunSummary]] = {}
    for run in runs:
        key = (run.scenario_id, run.seed)
        if key not in by_key:
            by_key[key] = []
        by_key[key].append(run)
    
    # Select canonical for each key
    return {key: select_canonical_run(runs_list) for key, runs_list in by_key.items()}


# =============================================================================
# Statistics Computation
# =============================================================================

def mean(values: List[float]) -> float:
    """Compute arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values: List[float]) -> float:
    """Compute median."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    else:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0


def stddev(values: List[float]) -> float:
    """Compute sample standard deviation."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)  # Sample stddev
    return math.sqrt(variance)


def percentile(values: List[float], p: float) -> float:
    """Compute empirical percentile (p in [0, 100])."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx = (p / 100.0) * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    frac = idx - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac


def round_for_comparison(value: float, decimals: int = ROUNDING_DECIMALS) -> float:
    """Round value for equality comparisons."""
    return round(value, decimals)


def compute_paired_deltas(
    baseline_index: Dict[Tuple[str, int], RunSummary],
    variant_runs: List[RunSummary],
) -> Tuple[List[PairedDelta], List[RunSummary], List[Tuple[str, int]]]:
    """
    Compute paired deltas between variant and baseline runs.
    
    Returns:
        - List of paired deltas for matched runs
        - List of variant runs missing baseline (no match)
        - List of baseline keys missing variant
    """
    variant_index = build_run_index(variant_runs)
    
    paired: List[PairedDelta] = []
    missing_baseline: List[RunSummary] = []
    
    # Find variant runs without baseline match
    for key, variant in variant_index.items():
        baseline = baseline_index.get(key)
        if baseline is None:
            missing_baseline.append(variant)
            continue
        
        delta_pnl = variant.final_pnl_usd - baseline.final_pnl_usd
        delta_drawdown = variant.max_drawdown_usd - baseline.max_drawdown_usd
        
        # Kill switch delta: +1 if false→true, -1 if true→false, 0 otherwise
        delta_ks = 0
        if not baseline.kill_switch_triggered and variant.kill_switch_triggered:
            delta_ks = 1
        elif baseline.kill_switch_triggered and not variant.kill_switch_triggered:
            delta_ks = -1
        
        paired.append(PairedDelta(
            scenario_id=key[0],
            seed=key[1],
            baseline_pnl=baseline.final_pnl_usd,
            variant_pnl=variant.final_pnl_usd,
            delta_pnl=delta_pnl,
            baseline_drawdown=baseline.max_drawdown_usd,
            variant_drawdown=variant.max_drawdown_usd,
            delta_drawdown=delta_drawdown,
            baseline_kill_switch=baseline.kill_switch_triggered,
            variant_kill_switch=variant.kill_switch_triggered,
            delta_kill_switch=delta_ks,
            baseline_path=str(baseline.source_path),
            variant_path=str(variant.source_path),
            baseline_checksum=baseline.checksum,
            variant_checksum=variant.checksum,
        ))
    
    # Find baseline keys missing variant
    missing_variant = [key for key in baseline_index if key not in variant_index]
    
    # Sort for determinism
    paired.sort(key=lambda d: (d.scenario_id, d.seed))
    missing_baseline.sort(key=lambda r: (r.scenario_id, r.seed))
    missing_variant.sort()
    
    return paired, missing_baseline, missing_variant


def compute_summary_stats(paired: List[PairedDelta]) -> SummaryStats:
    """Compute quant-grade summary statistics from paired deltas."""
    if not paired:
        return SummaryStats(
            n_pairs=0,
            mean_delta_pnl=0.0, median_delta_pnl=0.0, stddev_delta_pnl=0.0,
            p05_delta_pnl=0.0, p95_delta_pnl=0.0,
            win_rate=0.0, loss_rate=0.0, flat_rate=0.0,
            mean_delta_drawdown=0.0, median_delta_drawdown=0.0,
            p05_delta_drawdown=0.0, p95_delta_drawdown=0.0,
            kill_switch_false_to_true_rate=0.0, kill_switch_true_to_false_rate=0.0,
        )
    
    n = len(paired)
    delta_pnls = [d.delta_pnl for d in paired]
    delta_drawdowns = [d.delta_drawdown for d in paired]
    
    # Win/loss/flat rates (compare after rounding to avoid float noise)
    wins = sum(1 for d in delta_pnls if round_for_comparison(d) > 0)
    losses = sum(1 for d in delta_pnls if round_for_comparison(d) < 0)
    flats = n - wins - losses
    
    # Kill switch transitions
    false_to_true = sum(1 for d in paired if d.delta_kill_switch == 1)
    true_to_false = sum(1 for d in paired if d.delta_kill_switch == -1)
    
    return SummaryStats(
        n_pairs=n,
        mean_delta_pnl=mean(delta_pnls),
        median_delta_pnl=median(delta_pnls),
        stddev_delta_pnl=stddev(delta_pnls),
        p05_delta_pnl=percentile(delta_pnls, 5),
        p95_delta_pnl=percentile(delta_pnls, 95),
        win_rate=wins / n,
        loss_rate=losses / n,
        flat_rate=flats / n,
        mean_delta_drawdown=mean(delta_drawdowns),
        median_delta_drawdown=median(delta_drawdowns),
        p05_delta_drawdown=percentile(delta_drawdowns, 5),
        p95_delta_drawdown=percentile(delta_drawdowns, 95),
        kill_switch_false_to_true_rate=false_to_true / n,
        kill_switch_true_to_false_rate=true_to_false / n,
    )


# =============================================================================
# Report Generation
# =============================================================================

def get_git_sha() -> str:
    """Get current git SHA, or 'unknown' if not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def generate_report(run_root: Path) -> ResearchReport:
    """Generate the complete research report."""
    # Load all runs
    all_runs = load_all_runs(run_root)
    if not all_runs:
        return ResearchReport(
            metadata={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git_sha": get_git_sha(),
                "run_root": str(run_root),
                "schema_version": REPORT_SCHEMA_VERSION,
                "error": "No run_summary.json files found",
            },
            baseline_rows=[],
            variant_reports=[],
            all_kill_switches=[],
        )
    
    # Separate baseline and variant runs
    baseline_runs = [r for r in all_runs if is_baseline_run(r)]
    variant_runs = [r for r in all_runs if not is_baseline_run(r)]
    
    # Build baseline index (handles duplicates)
    baseline_index = build_run_index(baseline_runs)
    
    # Group variants by their ablation key
    variant_groups = group_runs_by_variant(variant_runs)
    
    # Generate report for each variant group
    variant_reports: List[VariantReport] = []
    all_kill_switches: List[Tuple[str, str, int, str]] = []
    
    # Sort variant keys lexicographically for determinism
    for variant_key in sorted(variant_groups.keys()):
        group_runs = variant_groups[variant_key]
        ablations = group_runs[0].ablations if group_runs else ()
        
        # Compute paired deltas
        paired, missing_bl, missing_var = compute_paired_deltas(baseline_index, group_runs)
        
        # Compute summary stats
        stats = compute_summary_stats(paired)
        
        # Track kill switches
        for p in paired:
            if p.variant_kill_switch:
                all_kill_switches.append((variant_key, p.scenario_id, p.seed, "variant"))
        
        variant_reports.append(VariantReport(
            variant_key=variant_key,
            ablations=ablations,
            stats=stats,
            paired_rows=paired,
            missing_baseline=missing_bl,
            missing_variant=missing_var,
        ))
    
    # Track baseline kill switches
    for run in baseline_runs:
        if run.kill_switch_triggered:
            all_kill_switches.append(("baseline", run.scenario_id, run.seed, run.kill_switch_reason or "unknown"))
    
    # Build metadata
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "run_root": str(run_root),
        "schema_version": REPORT_SCHEMA_VERSION,
        "total_runs": len(all_runs),
        "baseline_runs": len(baseline_runs),
        "variant_runs": len(variant_runs),
        "variant_groups": len(variant_reports),
    }
    
    return ResearchReport(
        metadata=metadata,
        baseline_rows=sorted(baseline_runs, key=lambda r: (r.scenario_id, r.seed)),
        variant_reports=variant_reports,
        all_kill_switches=sorted(all_kill_switches),
    )


# =============================================================================
# JSON Output
# =============================================================================

def format_float_json(v: float) -> float:
    """Format float for JSON (full precision, but handle NaN/Inf)."""
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def paired_delta_to_dict(p: PairedDelta) -> Dict[str, Any]:
    """Convert PairedDelta to JSON-serializable dict."""
    return {
        "scenario_id": p.scenario_id,
        "seed": p.seed,
        "baseline_pnl": format_float_json(p.baseline_pnl),
        "variant_pnl": format_float_json(p.variant_pnl),
        "delta_pnl": format_float_json(p.delta_pnl),
        "baseline_drawdown": format_float_json(p.baseline_drawdown),
        "variant_drawdown": format_float_json(p.variant_drawdown),
        "delta_drawdown": format_float_json(p.delta_drawdown),
        "baseline_kill_switch": p.baseline_kill_switch,
        "variant_kill_switch": p.variant_kill_switch,
        "delta_kill_switch": p.delta_kill_switch,
        "baseline_path": p.baseline_path,
        "variant_path": p.variant_path,
        "baseline_checksum": p.baseline_checksum,
        "variant_checksum": p.variant_checksum,
    }


def summary_stats_to_dict(s: SummaryStats) -> Dict[str, Any]:
    """Convert SummaryStats to JSON-serializable dict."""
    return {
        "n_pairs": s.n_pairs,
        "delta_pnl": {
            "mean": format_float_json(s.mean_delta_pnl),
            "median": format_float_json(s.median_delta_pnl),
            "stddev": format_float_json(s.stddev_delta_pnl),
            "p05": format_float_json(s.p05_delta_pnl),
            "p95": format_float_json(s.p95_delta_pnl),
        },
        "rates": {
            "win_rate": format_float_json(s.win_rate),
            "loss_rate": format_float_json(s.loss_rate),
            "flat_rate": format_float_json(s.flat_rate),
        },
        "delta_drawdown": {
            "mean": format_float_json(s.mean_delta_drawdown),
            "median": format_float_json(s.median_delta_drawdown),
            "p05": format_float_json(s.p05_delta_drawdown),
            "p95": format_float_json(s.p95_delta_drawdown),
        },
        "kill_switch_transitions": {
            "false_to_true_rate": format_float_json(s.kill_switch_false_to_true_rate),
            "true_to_false_rate": format_float_json(s.kill_switch_true_to_false_rate),
        },
    }


def baseline_row_to_dict(r: RunSummary) -> Dict[str, Any]:
    """Convert baseline RunSummary to JSON-serializable dict."""
    return {
        "scenario_id": r.scenario_id,
        "seed": r.seed,
        "final_pnl_usd": format_float_json(r.final_pnl_usd),
        "max_drawdown_usd": format_float_json(r.max_drawdown_usd),
        "kill_switch_triggered": r.kill_switch_triggered,
        "checksum": r.checksum,
        "source_path": str(r.source_path),
    }


def variant_report_to_dict(vr: VariantReport) -> Dict[str, Any]:
    """Convert VariantReport to JSON-serializable dict."""
    return {
        "variant_key": vr.variant_key,
        "ablations": list(vr.ablations),
        "summary_stats": summary_stats_to_dict(vr.stats),
        "paired_rows": [paired_delta_to_dict(p) for p in vr.paired_rows],
        "missing_baseline": [
            {"scenario_id": r.scenario_id, "seed": r.seed, "path": str(r.source_path)}
            for r in vr.missing_baseline
        ],
        "missing_variant": [
            {"scenario_id": key[0], "seed": key[1]}
            for key in vr.missing_variant
        ],
    }


def write_json_report(report: ResearchReport, out_path: Path) -> None:
    """Write the JSON research report."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "metadata": report.metadata,
        "baseline_rows": [baseline_row_to_dict(r) for r in report.baseline_rows],
        "variant_reports": [variant_report_to_dict(vr) for vr in report.variant_reports],
        "all_kill_switches": [
            {"variant": ks[0], "scenario_id": ks[1], "seed": ks[2], "reason": ks[3]}
            for ks in report.all_kill_switches
        ],
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


# =============================================================================
# Markdown Output
# =============================================================================

def fmt(v: float, decimals: int = 2) -> str:
    """Format float for Markdown with specified decimals."""
    return f"{v:.{decimals}f}"


def pct(v: float) -> str:
    """Format float as percentage."""
    return f"{v * 100:.1f}%"


def write_markdown_report(report: ResearchReport, out_path: Path) -> None:
    """Write the Markdown research report."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    
    # Header
    lines.append("# Research Report: Baseline vs Ablations")
    lines.append("")
    lines.append(f"**Generated:** {report.metadata.get('generated_at', 'unknown')}")
    lines.append(f"**Git SHA:** `{report.metadata.get('git_sha', 'unknown')[:12]}`")
    lines.append(f"**Run Root:** `{report.metadata.get('run_root', 'unknown')}`")
    lines.append(f"**Schema Version:** {report.metadata.get('schema_version', REPORT_SCHEMA_VERSION)}")
    lines.append("")
    
    # Summary counts
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total Runs:** {report.metadata.get('total_runs', 0)}")
    lines.append(f"- **Baseline Runs:** {report.metadata.get('baseline_runs', 0)}")
    lines.append(f"- **Variant Runs:** {report.metadata.get('variant_runs', 0)}")
    lines.append(f"- **Variant Groups:** {report.metadata.get('variant_groups', 0)}")
    lines.append("")
    
    # Kill switch summary
    if report.all_kill_switches:
        lines.append(f"⚠ **Kill Switches Triggered:** {len(report.all_kill_switches)}")
        lines.append("")
    else:
        lines.append("✓ **No Kill Switches Triggered**")
        lines.append("")
    
    # Executive Summary Table
    if report.variant_reports:
        lines.append("## Executive Summary")
        lines.append("")
        lines.append("| Variant | n | Mean ΔPnL | Median ΔPnL | p05 ΔPnL | p95 ΔPnL | Win% | Mean ΔDD | Median ΔDD |")
        lines.append("|---------|---|-----------|-------------|----------|----------|------|----------|------------|")
        
        for vr in report.variant_reports:
            s = vr.stats
            lines.append(
                f"| `{vr.variant_key}` "
                f"| {s.n_pairs} "
                f"| ${fmt(s.mean_delta_pnl)} "
                f"| ${fmt(s.median_delta_pnl)} "
                f"| ${fmt(s.p05_delta_pnl)} "
                f"| ${fmt(s.p95_delta_pnl)} "
                f"| {pct(s.win_rate)} "
                f"| ${fmt(s.mean_delta_drawdown)} "
                f"| ${fmt(s.median_delta_drawdown)} |"
            )
        lines.append("")
    
    # Detailed per-variant sections
    for vr in report.variant_reports:
        lines.append(f"## Variant: `{vr.variant_key}`")
        lines.append("")
        lines.append(f"**Ablations:** `{', '.join(vr.ablations) or 'none'}`")
        lines.append("")
        
        # Stats summary
        s = vr.stats
        lines.append("### Summary Statistics")
        lines.append("")
        lines.append(f"- **Paired Runs:** {s.n_pairs}")
        lines.append(f"- **ΔPnL:** mean=${fmt(s.mean_delta_pnl)}, median=${fmt(s.median_delta_pnl)}, stddev=${fmt(s.stddev_delta_pnl)}")
        lines.append(f"- **ΔPnL Percentiles:** p05=${fmt(s.p05_delta_pnl)}, p95=${fmt(s.p95_delta_pnl)}")
        lines.append(f"- **Win/Loss/Flat:** {pct(s.win_rate)} / {pct(s.loss_rate)} / {pct(s.flat_rate)}")
        lines.append(f"- **ΔDrawdown:** mean=${fmt(s.mean_delta_drawdown)}, median=${fmt(s.median_delta_drawdown)}")
        lines.append(f"- **Kill-Switch Transitions:** false→true={pct(s.kill_switch_false_to_true_rate)}, true→false={pct(s.kill_switch_true_to_false_rate)}")
        lines.append("")
        
        # Paired rows table
        if vr.paired_rows:
            lines.append("### Paired Results")
            lines.append("")
            lines.append("| Scenario | Seed | Baseline PnL | Variant PnL | ΔPnL | ΔDD | KS→ |")
            lines.append("|----------|------|--------------|-------------|------|-----|-----|")
            
            for p in vr.paired_rows:
                ks_str = "→✓" if p.delta_kill_switch == 1 else ("→✗" if p.delta_kill_switch == -1 else "—")
                lines.append(
                    f"| {p.scenario_id} "
                    f"| {p.seed} "
                    f"| ${fmt(p.baseline_pnl)} "
                    f"| ${fmt(p.variant_pnl)} "
                    f"| ${fmt(p.delta_pnl)} "
                    f"| ${fmt(p.delta_drawdown)} "
                    f"| {ks_str} |"
                )
            lines.append("")
        
        # Missing baseline
        if vr.missing_baseline:
            lines.append("### ⚠ Missing Baseline")
            lines.append("")
            lines.append("The following variant runs have no matching baseline (excluded from delta aggregates):")
            lines.append("")
            for r in vr.missing_baseline:
                lines.append(f"- `{r.scenario_id}` seed={r.seed}")
            lines.append("")
        
        # Missing variant
        if vr.missing_variant:
            lines.append("### ⚠ Missing Variant")
            lines.append("")
            lines.append("The following baseline runs have no matching variant:")
            lines.append("")
            for key in vr.missing_variant:
                lines.append(f"- `{key[0]}` seed={key[1]}")
            lines.append("")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# GitHub Step Summary
# =============================================================================

def generate_github_summary(report: ResearchReport) -> str:
    """Generate a compact GitHub Step Summary string."""
    lines = []
    
    lines.append("## 📊 Research Report: Delta Overview")
    lines.append("")
    
    if report.all_kill_switches:
        lines.append(f"⚠ **{len(report.all_kill_switches)} kill switch(es) triggered**")
        lines.append("")
    
    if report.variant_reports:
        lines.append("| Variant | n | Mean ΔPnL | Median | p05 | p95 | Win% |")
        lines.append("|---------|---|-----------|--------|-----|-----|------|")
        
        for vr in report.variant_reports:
            s = vr.stats
            lines.append(
                f"| `{vr.variant_key}` "
                f"| {s.n_pairs} "
                f"| ${fmt(s.mean_delta_pnl)} "
                f"| ${fmt(s.median_delta_pnl)} "
                f"| ${fmt(s.p05_delta_pnl)} "
                f"| ${fmt(s.p95_delta_pnl)} "
                f"| {pct(s.win_rate)} |"
            )
        lines.append("")
    else:
        lines.append("_No variant comparisons available._")
        lines.append("")
    
    lines.append(f"**Baseline:** {report.metadata.get('baseline_runs', 0)} runs | ")
    lines.append(f"**Variants:** {report.metadata.get('variant_runs', 0)} runs across {report.metadata.get('variant_groups', 0)} groups")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Legacy Mode (backwards compatibility)
# =============================================================================

def run_legacy_mode(args: argparse.Namespace) -> int:
    """Run in legacy mode for backwards compatibility."""
    from pathlib import Path
    
    run_root = Path(args.run_root)
    baseline_name = args.baseline_suite
    suite_prefix = args.suite_prefix
    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)
    
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
        print(f"ERROR: No suite directories matching '{suite_prefix}*' in {run_root}", file=sys.stderr)
        return 4
    
    # Generate report using new infrastructure
    report = generate_report(run_root)
    
    # Write outputs
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    write_markdown_report(report, out_md)
    
    # Generate CSV (simple format for backwards compatibility)
    lines = ["suite,scenario_id,seed,ablations,pnl,baseline_pnl,delta_pnl,drawdown,baseline_drawdown,delta_drawdown,kill_switch_triggered,kill_switch_reason"]
    for vr in report.variant_reports:
        for p in vr.paired_rows:
            ablations_str = ";".join(vr.ablations)
            lines.append(
                f"{vr.variant_key},{p.scenario_id},{p.seed},\"{ablations_str}\","
                f"{fmt(p.variant_pnl)},{fmt(p.baseline_pnl)},{fmt(p.delta_pnl)},"
                f"{fmt(p.variant_drawdown)},{fmt(p.baseline_drawdown)},{fmt(p.delta_drawdown)},"
                f"{str(p.variant_kill_switch).lower()},\"\""
            )
    
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_csv}")
    
    # Check for kill switches
    if report.all_kill_switches:
        print(f"\nWARNING: {len(report.all_kill_switches)} kill switch(es) triggered", file=sys.stderr)
        return 1
    
    return 0


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate quant-grade research report with paired delta analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # New mode arguments
    parser.add_argument(
        "--run-root",
        default="runs",
        help="Root directory containing run outputs (default: runs)",
    )
    parser.add_argument(
        "--out-dir",
        help="Output directory for reports (default: _reports)",
    )
    
    # Legacy mode arguments (for backwards compatibility)
    parser.add_argument(
        "--baseline-suite",
        help="[Legacy] Name of the baseline suite directory",
    )
    parser.add_argument(
        "--suite-prefix",
        help="[Legacy] Prefix to match suite directories",
    )
    parser.add_argument(
        "--out-md",
        help="[Legacy] Output path for Markdown report",
    )
    parser.add_argument(
        "--out-csv",
        help="[Legacy] Output path for CSV report",
    )
    
    args = parser.parse_args()
    
    # Check if running in legacy mode
    if args.baseline_suite or args.suite_prefix or args.out_csv:
        # Legacy mode: set defaults for missing legacy args
        if not args.baseline_suite:
            args.baseline_suite = "research__baseline"
        if not args.suite_prefix:
            args.suite_prefix = "research__"
        if not args.out_md:
            args.out_md = "_reports/research_report.md"
        if not args.out_csv:
            args.out_csv = "_reports/research_report.csv"
        return run_legacy_mode(args)
    
    # New mode
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else Path("_reports")
    
    if not run_root.is_dir():
        print(f"ERROR: Run root directory not found: {run_root}", file=sys.stderr)
        return 2
    
    print(f"Generating research report from: {run_root}")
    
    # Generate report
    report = generate_report(run_root)
    
    if not report.baseline_rows and not report.variant_reports:
        print("WARNING: No runs found", file=sys.stderr)
        if "error" in report.metadata:
            print(f"  {report.metadata['error']}", file=sys.stderr)
    
    # Write outputs
    out_md = out_dir / "research_report.md"
    out_json = out_dir / "research_report.json"
    
    write_json_report(report, out_json)
    write_markdown_report(report, out_md)
    
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    
    # Print GitHub summary
    gh_summary = generate_github_summary(report)
    print("\n--- GitHub Step Summary ---")
    print(gh_summary)
    
    # Write to GITHUB_STEP_SUMMARY if available
    gh_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_summary_path:
        try:
            with open(gh_summary_path, "a", encoding="utf-8") as f:
                f.write("\n")
                f.write(gh_summary)
            print(f"Appended to: {gh_summary_path}")
        except IOError as e:
            print(f"WARNING: Failed to write to GITHUB_STEP_SUMMARY: {e}", file=sys.stderr)
    
    # Final status
    print("\n=== Report Generation Complete ===")
    print(f"Baseline runs: {report.metadata.get('baseline_runs', 0)}")
    print(f"Variant groups: {report.metadata.get('variant_groups', 0)}")
    
    if report.all_kill_switches:
        print(f"\n⚠  {len(report.all_kill_switches)} kill switch(es) triggered:")
        for ks in report.all_kill_switches[:5]:
            print(f"    - {ks[0]}/{ks[1]}@{ks[2]}: {ks[3]}")
        if len(report.all_kill_switches) > 5:
            print(f"    ... and {len(report.all_kill_switches) - 5} more")
        return 1
    
    print("✓ All runs passed (no kill switches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
