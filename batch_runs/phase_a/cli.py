"""
cli.py

CLI entrypoint for Phase A promotion pipeline.

Usage:
    python3 -m batch_runs.phase_a.cli --help
    python3 -m batch_runs.phase_a.cli optimize --trials 10 --mc-runs 30 --seed 42
    python3 -m batch_runs.phase_a.cli promote --study my_study
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .optimize import optimize, load_budgets_from_yaml
from .promote import promote


# ===========================================================================
# Paths
# ===========================================================================

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = ROOT / "runs"
DEFAULT_BUDGETS_FILE = Path(__file__).parent / "budgets.yaml"


# ===========================================================================
# CLI commands
# ===========================================================================

def cmd_optimize(args: argparse.Namespace) -> int:
    """Run multi-objective optimization."""
    print("=" * 60)
    print("Phase A: Multi-objective Optimization")
    print("=" * 60)
    
    # Determine study name
    if args.study:
        study_name = args.study
    else:
        study_name = f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Load budgets
    budgets_file = Path(args.budgets) if args.budgets else DEFAULT_BUDGETS_FILE
    if budgets_file.exists():
        budgets = load_budgets_from_yaml(budgets_file)
        print(f"Loaded budgets from: {budgets_file}")
    else:
        budgets = None
        print("Using default budgets")
    
    # Run optimization
    try:
        results, pareto, winners = optimize(
            trials=args.trials,
            mc_runs=args.mc_runs,
            study_name=study_name,
            out_dir=Path(args.out),
            seed=args.seed,
            mc_ticks=args.mc_ticks,
            budgets=budgets,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"\nERROR: Optimization failed: {e}", file=sys.stderr)
        return 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("Optimization Complete")
    print("=" * 60)
    print(f"  Trials: {len(results)}")
    print(f"  Valid: {sum(1 for r in results if r.is_valid)}")
    print(f"  Pareto frontier: {len(pareto)}")
    print()
    
    # Print winners per tier
    print("Winners by tier:")
    for tier, winner in sorted(winners.items()):
        if winner:
            print(f"  {tier}: {winner.candidate_id} (pnl={winner.mc_mean_pnl:.2f})")
        else:
            print(f"  {tier}: No candidate meets budget")
    
    print(f"\nResults saved to: {Path(args.out) / 'phaseA' / study_name}")
    
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    """Promote winning configurations."""
    print("=" * 60)
    print("Phase A: Promotion")
    print("=" * 60)
    
    # Load budgets
    budgets_file = Path(args.budgets) if args.budgets else DEFAULT_BUDGETS_FILE
    if budgets_file.exists():
        print(f"Loaded budgets from: {budgets_file}")
    else:
        budgets_file = None
        print("Using default budgets")
    
    # Determine output directory
    output_dir = Path(args.promote_dir) if args.promote_dir else None
    
    try:
        promotions = promote(
            study_name=args.study,
            runs_dir=Path(args.out),
            tier=args.tier,
            budgets_file=budgets_file,
            output_dir=output_dir,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"\nERROR: Promotion failed: {e}", file=sys.stderr)
        return 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("Promotion Complete")
    print("=" * 60)
    
    promoted_count = sum(1 for p in promotions.values() if p is not None)
    print(f"  Promoted: {promoted_count}/{len(promotions)} tiers")
    print()
    
    for tier, paths in sorted(promotions.items()):
        if paths:
            config_path, record_path = paths
            print(f"  {tier}:")
            print(f"    Config: {config_path}")
            print(f"    Record: {record_path}")
        else:
            print(f"  {tier}: Not promoted (no qualifying candidate)")
    
    return 0


# ===========================================================================
# Main
# ===========================================================================

def main(argv: Optional[list] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="python3 -m batch_runs.phase_a.cli",
        description="Phase A: Promotion Pipeline + Multi-objective Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run optimization with 10 trials
  python3 -m batch_runs.phase_a.cli optimize --trials 10 --mc-runs 30 --seed 42

  # Run quick smoke test
  python3 -m batch_runs.phase_a.cli optimize --trials 2 --mc-runs 10 --seed 1

  # Promote from a completed study
  python3 -m batch_runs.phase_a.cli promote --study my_study

  # Promote only the balanced tier
  python3 -m batch_runs.phase_a.cli promote --study my_study --tier balanced
""",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # =======================================================================
    # optimize subcommand
    # =======================================================================
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Run multi-objective optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    optimize_parser.add_argument(
        "--trials", "-n",
        type=int,
        default=10,
        help="Number of candidate trials to run (default: 10)",
    )
    
    optimize_parser.add_argument(
        "--mc-runs",
        type=int,
        default=50,
        help="Number of Monte Carlo runs per candidate (default: 50)",
    )
    
    optimize_parser.add_argument(
        "--mc-ticks",
        type=int,
        default=600,
        help="Ticks per Monte Carlo run (default: 600)",
    )
    
    optimize_parser.add_argument(
        "--study",
        type=str,
        default=None,
        help="Study name (default: auto-generated timestamp)",
    )
    
    optimize_parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_RUNS_DIR),
        help=f"Output directory (default: {DEFAULT_RUNS_DIR})",
    )
    
    optimize_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for determinism (default: 42)",
    )
    
    optimize_parser.add_argument(
        "--budgets",
        type=str,
        default=None,
        help=f"Path to budgets.yaml (default: {DEFAULT_BUDGETS_FILE})",
    )
    
    optimize_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    
    # =======================================================================
    # promote subcommand
    # =======================================================================
    promote_parser = subparsers.add_parser(
        "promote",
        help="Promote winning configurations from a study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    promote_parser.add_argument(
        "--study",
        type=str,
        required=True,
        help="Study name to promote from (required)",
    )
    
    promote_parser.add_argument(
        "--tier",
        type=str,
        default=None,
        help="Specific tier to promote (default: all tiers)",
    )
    
    promote_parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_RUNS_DIR),
        help=f"Runs directory containing the study (default: {DEFAULT_RUNS_DIR})",
    )
    
    promote_parser.add_argument(
        "--promote-dir",
        type=str,
        default=None,
        help="Output directory for promoted configs (default: configs/presets/promoted/)",
    )
    
    promote_parser.add_argument(
        "--budgets",
        type=str,
        default=None,
        help=f"Path to budgets.yaml (default: {DEFAULT_BUDGETS_FILE})",
    )
    
    promote_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    
    # =======================================================================
    # Parse and dispatch
    # =======================================================================
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command == "optimize":
        return cmd_optimize(args)
    elif args.command == "promote":
        return cmd_promote(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

