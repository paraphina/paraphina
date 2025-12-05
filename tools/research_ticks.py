#!/usr/bin/env python
"""
Research helper for Paraphina tick logs.

Usage:
    python tools/research_ticks.py paraphina_ticks.jsonl --prefix research_run_005

This will:
  - Copy the input JSONL to:  research_run_005.jsonl
  - Print some summary stats to stdout
  - Save three PNG charts in the current directory:
        research_run_005_pnl_curve.png
        research_run_005_delta_curve.png
        research_run_005_basis_curve.png
"""

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for SSH/headless use

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Paraphina JSONL tick logs and plot basic curves."
    )
    parser.add_argument(
        "ticks_path",
        help="Path to the JSONL tick log (produced by --log-jsonl).",
    )
    parser.add_argument(
        "--prefix",
        default="research_run",
        help="Prefix for output files (e.g. research_run_005).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ticks_path = Path(args.ticks_path)
    if not ticks_path.is_file():
        raise SystemExit(f"Input file not found: {ticks_path}")

    # Normalise the prefix (strip any extension if the user typed .jsonl)
    prefix = Path(args.prefix).stem

    # Copy the raw tick log to a research archive file in the CWD
    archived_jsonl = Path(f"{prefix}.jsonl")
    if ticks_path.resolve() != archived_jsonl.resolve():
        shutil.copy2(ticks_path, archived_jsonl)
        print(f"Copied tick log to {archived_jsonl}")
    else:
        print(f"Using existing tick log {archived_jsonl}")

    # Load JSONL into a DataFrame
    print(f"Loading {archived_jsonl} ...")
    df = pd.read_json(archived_jsonl, lines=True)

    if "tick" not in df.columns:
        raise SystemExit("Expected a 'tick' column in the JSONL file.")

    # --- Basic summary -------------------------------------------------------
    n_ticks = len(df)
    last = df.iloc[-1]

    print()
    print(f"Loaded {n_ticks} ticks from {archived_jsonl}")
    print("-" * 72)

    fv_min = df["fair_value"].min()
    fv_max = df["fair_value"].max()
    fv_last = last["fair_value"]
    fv_mean = df["fair_value"].mean()

    print(
        f"Fair value: min={fv_min:.4f}, max={fv_max:.4f}, "
        f"last={fv_last:.4f}, mean={fv_mean:.4f}"
    )

    pnl_last = last["daily_pnl_total"]
    print(f"Total PnL (last tick): {pnl_last:.4f}")

    dd_min = df["dollar_delta_usd"].min()
    dd_max = df["dollar_delta_usd"].max()
    dd_last = last["dollar_delta_usd"]
    dd_mean = df["dollar_delta_usd"].mean()

    print(
        f"Dollar delta USD: min={dd_min:.4f}, max={dd_max:.4f}, "
        f"last={dd_last:.4f}, mean={dd_mean:.4f}"
    )

    basis_min = df["basis_usd"].min()
    basis_max = df["basis_usd"].max()
    basis_last = last["basis_usd"]
    basis_mean = df["basis_usd"].mean()

    print(
        f"Basis exposure USD: min={basis_min:.4f}, max={basis_max:.4f}, "
        f"last={basis_last:.4f}, mean={basis_mean:.4f}"
    )

    # --- Plot helpers --------------------------------------------------------
    def save_curve(y_col: str, ylabel: str, title: str, suffix: str) -> None:
        out_path = Path(f"{prefix}_{suffix}.png")
        plt.figure()
        plt.plot(df["tick"], df[y_col])
        plt.xlabel("tick")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

    # --- Individual curves ---------------------------------------------------
    save_curve(
        y_col="daily_pnl_total",
        ylabel="daily_pnl_total",
        title="Total PnL vs tick",
        suffix="pnl_curve",
    )

    save_curve(
        y_col="dollar_delta_usd",
        ylabel="dollar_delta_usd",
        title="Dollar delta vs tick",
        suffix="delta_curve",
    )

    save_curve(
        y_col="basis_usd",
        ylabel="basis_usd",
        title="Basis exposure vs tick",
        suffix="basis_curve",
    )


if __name__ == "__main__":
    main()
