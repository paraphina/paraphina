#!/usr/bin/env python3
"""
Exp06: Full Paraphina research / calibration suite.

Runs, in order:

  1) baseline_profile.py           – baseline q0 = 0, band = 5, loss_limit = 4k
  2) batch_runs.py                 – Exp01 hedge_band_base sweep (q0 = 0)
  3) exp02_hedge_band_heatmap.py   – Exp02 initial_q_tao vs hedge_band heatmap
  4) exp03_size_eta_vol_sweep.py   – Exp03 size_eta vs vol_ref grid
  5) exp04_risk_regime_sweep.py    – Exp04 loss_limit vs initial_q_tao sweep
  6) exp05_calibration_report.py   – Exp05 summary / ranked configs
  7) exp07_optimal_presets.py      – Exp07 optimal profile presets
  8) exp08_profile_validation.py   – Exp08 profile sanity / regression

This is the "big red button" to re-generate all calibration artefacts
from scratch in a reproducible way.

Upgrades vs the simple version:
  - Step registry with IDs (baseline, exp01, ...).
  - CLI to select / skip steps, run subsets, or dry-run.
  - Per-step timing and success/failure tracking.
  - Final summary table of what ran and how it went.

Default behaviour (no CLI flags) is: run ALL steps in order.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """One experiment step in the research suite."""
    step_id: str          # e.g. "exp01"
    label: str            # human-readable label
    script: str           # script filename under tools/
    default_args: List[str]


@dataclass
class StepResult:
    step: Step
    returncode: int
    duration_s: float


STEPS: List[Step] = [
    Step("baseline", "Baseline profile", "baseline_profile.py", []),
    Step("exp01", "Hedge-band sweep (q0 = 0)", "batch_runs.py", []),
    Step("exp02", "Initial_q vs hedge_band heatmap", "exp02_hedge_band_heatmap.py", []),
    Step("exp03", "size_eta vs vol_ref sweep", "exp03_size_eta_vol_sweep.py", []),
    Step("exp04", "Risk regime sweep (loss_limit vs q0)", "exp04_risk_regime_sweep.py", []),
    Step("exp05", "Calibration report / ranked configs", "exp05_calibration_report.py", []),
    # IMPORTANT: wire Exp07 directly to the *per-run* CSV so it doesn't
    # accidentally auto-detect the summary file.
    Step(
        "exp07",
        "Optimal preset selector (profiles)",
        "exp07_optimal_presets.py",
        ["--runs-csv", "runs/exp04_risk_regime/exp04_risk_regime_runs.csv"],
    ),
    Step("exp08", "Profile presets validation", "exp08_profile_validation.py", []),
]

STEP_IDS = [s.step_id for s in STEPS]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full Paraphina research / calibration suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--only",
        metavar="STEP_ID",
        nargs="+",
        choices=STEP_IDS,
        help=f"Run only the specified step IDs (subset of: {', '.join(STEP_IDS)}).",
    )

    p.add_argument(
        "--skip",
        metavar="STEP_ID",
        nargs="+",
        choices=STEP_IDS,
        help="Skip the specified step IDs.",
    )

    p.add_argument(
        "--from-step",
        metavar="STEP_ID",
        choices=STEP_IDS,
        help="Start from this step (inclusive).",
    )

    p.add_argument(
        "--up-to",
        metavar="STEP_ID",
        choices=STEP_IDS,
        help="Run up to this step (inclusive).",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be executed, but do not actually run anything.",
    )

    p.add_argument(
        "--keep-going",
        action="store_true",
        help="Do not abort on first failure; attempt to run all selected steps.",
    )

    p.add_argument(
        "--list-steps",
        action="store_true",
        help="List available steps and exit.",
    )

    return p.parse_args(argv)


def filter_steps(args: argparse.Namespace) -> List[Step]:
    steps: List[Step] = list(STEPS)

    if args.only:
        ids = set(args.only)
        steps = [s for s in steps if s.step_id in ids]

    if args.from_step:
        seen = False
        filtered: List[Step] = []
        for s in steps:
            if s.step_id == args.from_step:
                seen = True
            if seen:
                filtered.append(s)
        steps = filtered

    if args.up_to:
        filtered: List[Step] = []
        for s in steps:
            filtered.append(s)
            if s.step_id == args.up_to:
                break
        steps = filtered

    if args.skip:
        skip_ids = set(args.skip)
        steps = [s for s in steps if s.step_id not in skip_ids]

    return steps


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def run_step(step: Step, dry_run: bool) -> StepResult:
    """Run one Python tool under the current interpreter."""
    cmd = [sys.executable, str(TOOLS / step.script), *step.default_args]

    banner = f"==> [{step.step_id}] python tools/{step.script} {' '.join(step.default_args)}"
    line = "=" * len(banner)
    print(f"\n{line}\n{banner}\n{line}")

    if dry_run:
        print("[DRY-RUN] Not executing, just showing the command above.")
        return StepResult(step=step, returncode=0, duration_s=0.0)

    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=ROOT)
    dt = time.monotonic() - t0

    rc = proc.returncode
    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
    print(f"[{step.step_id}] finished in {dt:.1f}s – {status}")

    return StepResult(step=step, returncode=rc, duration_s=dt)


def print_summary(results: List[StepResult]) -> None:
    if not results:
        print("\nNo steps were executed.")
        return

    print("\n" + "=" * 72)
    print("Calibration suite summary")
    print("=" * 72)

    header = f"{'STEP':<10} {'STATUS':<12} {'DURATION(s)':>12}"
    print(header)
    print("-" * len(header))

    all_ok = True
    for r in results:
        status = "OK" if r.returncode == 0 else f"FAIL({r.returncode})"
        if r.returncode != 0:
            all_ok = False
        print(f"{r.step.step_id:<10} {status:<12} {r.duration_s:>12.1f}")

    print("-" * len(header))
    if all_ok:
        print("All selected steps completed successfully.")
    else:
        print("Some steps failed. Inspect logs / stdout for details.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_steps:
        print("Available steps (in order):")
        for s in STEPS:
            print(f"  {s.step_id:<8} -> {s.script:<28} – {s.label}")
        return

    selected = filter_steps(args)

    if not selected:
        print("No steps selected (after applying --only/--skip/--from-step/--up-to).")
        return

    results: List[StepResult] = []

    for step in selected:
        res = run_step(step, dry_run=args.dry_run)
        results.append(res)

        if not args.keep_going and not args.dry_run and res.returncode != 0:
            print(
                f"\nAborting because step '{step.step_id}' failed and "
                f"--keep-going was not set."
            )
            break

    print_summary(results)


if __name__ == "__main__":
    main()
