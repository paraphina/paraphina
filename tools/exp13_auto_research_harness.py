#!/usr/bin/env python3
"""
exp13_auto_research_harness.py

DARPA-grade auto research harness.

Pipeline
========
For each risk profile (conservative / balanced / aggressive):

  1. Run the Rust engine with JSONL telemetry enabled.
     - PARAPHINA_TELEMETRY_MODE = jsonl
     - PARAPHINA_TELEMETRY_PATH = runs/exp13_auto/<profile>_r<idx>.jsonl

  2. Condense the tick-level telemetry into a single research row via
     tools/research_ticks.py, and append it to a JSONL dataset.

After all runs:

  3. Run tools/research_summary.py on the dataset.
  4. Run tools/exp11_research_alignment.py on the dataset.

Usage examples
--------------
# Minimal: 1 run per profile, 1000 ticks, new dataset v004
python tools/exp13_auto_research_harness.py \
    --ticks 1000 \
    --runs-per-profile 1 \
    --dataset research_dataset_v004.jsonl

# Heavier: 3 runs per profile, 2000 ticks
python tools/exp13_auto_research_harness.py \
    --ticks 2000 \
    --runs-per-profile 3 \
    --dataset research_dataset_v004.jsonl
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNS_DIR = ROOT / "runs" / "exp13_auto"


def run_cmd(cmd: list[str], env: dict | None = None) -> None:
    """Run a subprocess, streaming output, and fail loudly on error."""
    print(f"[exp13] Running: {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def run_single_sim(profile: str, run_idx: int, ticks: int, label_prefix: str, dataset: str) -> None:
    """
    Run a single simulation for a given profile, write telemetry JSONL,
    and append a condensed research row into `dataset`.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Telemetry path relative to repo root (for research_ticks) and absolute (for env).
    telemetry_rel = Path("runs") / "exp13_auto" / f"{profile}_r{run_idx}.jsonl"
    telemetry_abs = ROOT / telemetry_rel

    label = f"{label_prefix}_{profile}_r{run_idx:03d}"

    print(f"\n[exp13] === Run profile={profile} idx={run_idx} ticks={ticks} label={label} ===")
    print(f"[exp13] Telemetry: {telemetry_rel}")

    # 1) Run the Rust engine with telemetry enabled.
    env = os.environ.copy()
    env["PARAPHINA_TELEMETRY_MODE"] = "jsonl"
    env["PARAPHINA_TELEMETRY_PATH"] = str(telemetry_abs)

    run_cmd(
        [
            "cargo",
            "run",
            "--release",
            "--",
            "--ticks",
            str(ticks),
            "--profile",
            profile,
        ],
        env=env,
    )

    # 2) Condense the telemetry JSONL into a single research row.
    run_cmd(
        [
            sys.executable,
            str(TOOLS / "research_ticks.py"),
            "--telemetry",
            str(telemetry_rel),
            "--profile",
            profile,
            "--label",
            label,
            "--out",
            dataset,
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp13: automatic research harness over risk profiles."
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=2000,
        help="Number of ticks per run (default: 2000).",
    )
    parser.add_argument(
        "--runs-per-profile",
        type=int,
        default=2,
        help="Number of independent runs per risk profile (default: 2).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="research_dataset_v004.jsonl",
        help="Path (relative to repo root) of research JSONL dataset to append to.",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        nargs="+",
        default=["conservative", "balanced", "aggressive"],
        help="List of risk profiles to run (default: conservative balanced aggressive).",
    )
    parser.add_argument(
        "--label-prefix",
        type=str,
        default="exp13",
        help="Prefix for run labels stored in the dataset (default: exp13).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[exp13] Root dir:   ", ROOT)
    print("[exp13] Runs dir:   ", RUNS_DIR)
    print("[exp13] Dataset:    ", args.dataset)
    print("[exp13] Profiles:   ", ", ".join(args.profiles))
    print("[exp13] Ticks/run:  ", args.ticks)
    print("[exp13] Runs/profile:", args.runs_per_profile)
    print()

    # 1–2) Run sims + append research rows.
    for profile in args.profiles:
        for run_idx in range(1, args.runs_per_profile + 1):
            run_single_sim(
                profile=profile,
                run_idx=run_idx,
                ticks=args.ticks,
                label_prefix=args.label_prefix,
                dataset=args.dataset,
            )

    # 3) Summarise the dataset.
    print("\n[exp13] === Research summary ===")
    run_cmd(
        [
            sys.executable,
            str(TOOLS / "research_summary.py"),
            "--input",
            args.dataset,
        ]
    )

    # 4) World-model alignment check on the dataset.
    print("\n[exp13] === World-model alignment (Exp11) ===")
    run_cmd(
        [
            sys.executable,
            str(TOOLS / "exp11_research_alignment.py"),
            "--dataset",
            args.dataset,
        ]
    )

    print("\n[exp13] Done.")


if __name__ == "__main__":
    main()
