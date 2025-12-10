#!/usr/bin/env python3
"""
research_summary.py

Summarise a JSONL research dataset produced by tools/research_ticks.py.

Each line is expected to be a *run-level* record with fields like:
  - label, profile, telemetry_path, timestamp_utc
  - band_base, mm_size_eta, vol_ref, daily_loss_limit, init_q_tao, vol_scale
  - num_ticks, final_pnl, max_pnl, min_pnl, max_drawdown
  - frac_normal, frac_warning, frac_hard, kill_switch

The script:
  - loads the JSONL dataset
  - filters out any non-run lines (e.g. stray tick-level JSON)
  - computes global stats
  - computes per-profile stats
  - prints everything in a compact table

Usage
-----

# Basic: summarise the default dataset
python tools/research_summary.py --input research_dataset_v001.jsonl

# If you're still using the earlier file:
python tools/research_summary.py --input research_run_001.jsonl

You can optionally filter by profile or label substring:

python tools/research_summary.py --input research_dataset_v001.jsonl \
    --profile aggressive

python tools/research_summary.py --input research_dataset_v001.jsonl \
    --label-contains manual_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def load_runs(path: Path) -> pd.DataFrame:
    """Load run-level records from a JSONL file into a DataFrame.

    Lines that do not contain the core metrics (e.g. tick-level telemetry)
    are ignored.
    """
    records: List[Dict[str, Any]] = []
    num_lines = 0
    num_skipped = 0

    core_keys = {"final_pnl", "max_drawdown", "num_ticks"}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                num_skipped += 1
                continue

            # Only keep records that look like run-level summaries.
            if not core_keys.issubset(rec.keys()):
                num_skipped += 1
                continue

            records.append(rec)

    if not records:
        raise SystemExit(
            f"No run-level records found in {path} "
            f"(read {num_lines} lines, skipped {num_skipped})."
        )

    df = pd.DataFrame.from_records(records)
    df["_source_path"] = str(path)
    return df


def _safe_bool_series(s: pd.Series) -> pd.Series:
    """Convert a column to bool safely."""
    if s.dtype == bool:
        return s
    # Handle strings like "True"/"False" etc.
    return s.astype(str).str.lower().isin({"1", "true", "yes"})


def summarise_global(df: pd.DataFrame) -> None:
    print("=== Global summary ===")

    n_runs = len(df)
    print(f"Runs:          {n_runs}")

    for col in ["final_pnl", "max_drawdown", "max_pnl", "min_pnl"]:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        mean = float(series.mean())
        med = float(series.median())
        std = float(series.std())
        print(
            f"{col:14s}mean={mean:10.2f}  "
            f"median={med:10.2f}  std={std:10.2f}"
        )

    if "kill_switch" in df.columns:
        ks = _safe_bool_series(df["kill_switch"])
        frac = float(ks.mean())
        print(f"kill_switch:   {ks.sum()} / {len(ks)}  ({frac:.3f} of runs)")

    # Simple "utility" score: reward final PnL, penalise drawdown
    if {"final_pnl", "max_drawdown"} <= set(df.columns):
        pnl = pd.to_numeric(df["final_pnl"], errors="coerce")
        dd = pd.to_numeric(df["max_drawdown"], errors="coerce").clip(lower=0.0)
        utility = pnl - 0.5 * dd
        print(
            f"utility score: mean={float(utility.mean()):.2f}  "
            f"min={float(utility.min()):.2f}  max={float(utility.max()):.2f}"
        )

    print()


def summarise_by_profile(df: pd.DataFrame) -> None:
    if "profile" not in df.columns:
        print("No 'profile' column; skipping per-profile summary.\n")
        return

    print("=== Per-profile summary ===")
    # Ensure numeric
    df = df.copy()
    for col in ["final_pnl", "max_drawdown", "max_pnl", "min_pnl"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "kill_switch" in df.columns:
        df["kill_switch_bool"] = _safe_bool_series(df["kill_switch"])
    else:
        df["kill_switch_bool"] = False

    groups = df.groupby("profile", dropna=False)

    header = (
        f"{'profile':12s} "
        f"{'runs':>4s} "
        f"{'final_pnl_mean':>14s} "
        f"{'final_pnl_med':>13s} "
        f"{'max_dd_mean':>12s} "
        f"{'kill_frac':>9s}"
    )
    print(header)
    print("-" * len(header))

    for profile, g in groups:
        n = len(g)
        final_pnl = g["final_pnl"]
        max_dd = g["max_drawdown"].clip(lower=0.0)
        if n == 0:
            continue

        kill_frac = float(g["kill_switch_bool"].mean())

        print(
            f"{str(profile):12s} "
            f"{n:4d} "
            f"{float(final_pnl.mean()):14.2f} "
            f"{float(final_pnl.median()):13.2f} "
            f"{float(max_dd.mean()):12.2f} "
            f"{kill_frac:9.3f}"
        )

    print()


def list_runs(df: pd.DataFrame, limit: int = 20) -> None:
    """Print a small table of individual runs sorted by utility."""
    if not {"final_pnl", "max_drawdown", "label"} <= set(df.columns):
        print("Not enough columns to list runs with utility; skipping.\n")
        return

    df = df.copy()
    df["final_pnl"] = pd.to_numeric(df["final_pnl"], errors="coerce")
    df["max_drawdown"] = pd.to_numeric(df["max_drawdown"], errors="coerce").clip(
        lower=0.0
    )
    df["utility"] = df["final_pnl"] - 0.5 * df["max_drawdown"]

    df_sorted = df.sort_values("utility", ascending=False)

    print(f"=== Top {min(limit, len(df_sorted))} runs by utility (pnl - 0.5 * dd) ===")
    header = (
        f"{'rank':>4s} "
        f"{'label':24s} "
        f"{'profile':10s} "
        f"{'final_pnl':>10s} "
        f"{'max_dd':>10s} "
        f"{'utility':>10s} "
        f"{'kill':>5s}"
    )
    print(header)
    print("-" * len(header))

    has_profile = "profile" in df_sorted.columns
    has_kill = "kill_switch" in df_sorted.columns

    for i, (_, row) in enumerate(df_sorted.head(limit).iterrows(), start=1):
        label = str(row.get("label", ""))[:24]
        profile = str(row.get("profile", ""))[:10] if has_profile else ""
        final_pnl = float(row["final_pnl"])
        max_dd = float(row["max_drawdown"])
        utility = float(row["utility"])
        kill = ""
        if has_kill:
            kill = "Y" if bool(row["kill_switch"]) else "N"

        print(
            f"{i:4d} "
            f"{label:24s} "
            f"{profile:10s} "
            f"{final_pnl:10.2f} "
            f"{max_dd:10.2f} "
            f"{utility:10.2f} "
            f"{kill:>5s}"
        )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise a JSONL research dataset produced by research_ticks.py"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to research JSONL dataset (e.g. research_dataset_v001.jsonl)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional filter: only include runs with this profile.",
    )
    parser.add_argument(
        "--label-contains",
        type=str,
        default=None,
        help="Optional filter: only include runs whose label contains this substring.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many individual runs to list in the ranking (default: 20).",
    )

    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file does not exist: {args.input}")

    df = load_runs(args.input)

    # Apply optional filters
    if args.profile is not None and "profile" in df.columns:
        df = df[df["profile"] == args.profile]

    if args.label_contains is not None and "label" in df.columns:
        df = df[df["label"].astype(str).str.contains(args.label_contains)]

    if df.empty:
        raise SystemExit("No runs left after filtering.")

    summarise_global(df)
    summarise_by_profile(df)
    list_runs(df, limit=args.limit)


if __name__ == "__main__":
    main()
