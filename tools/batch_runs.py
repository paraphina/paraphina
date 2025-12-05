#!/usr/bin/env python3
"""
Hedge-band sweep harness for Paraphina.

- Sweeps hedge_band_base over a small grid of TAO values.
- For each band, runs the Rust sim multiple times with env overrides:
    PARA_INITIAL_Q_TAO
    PARA_HEDGE_BAND_BASE
    PARA_HEDGE_MAX_STEP
- Reads the JSONL logs, computes risk/return metrics, and writes:
    exp01_band_sweep_runs.csv      (per-run metrics)
    exp01_band_sweep_summary.csv   (per-band aggregated metrics)
    exp01_band_sweep_pnl.png       (PnL vs band plot)

This is our first DARPA-grade research harness: fully reproducible and
easy to extend with more knobs later.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Configuration of the experiment
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "batch_runs"
RUNS_DIR.mkdir(exist_ok=True)

# How many synthetic ticks per run.
TICKS = 200

# How many independent runs per hedge band.
RUNS_PER_BAND = 5

# Grid of hedge bands (TAO) to sweep.
HEDGE_BANDS_TAO = [2.5, 5.0, 7.5, 10.0, 15.0]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def run_one_sim(hedge_band_tao: float, run_idx: int) -> pd.DataFrame:
    """
    Launch one Rust simulation with a given hedge_band_base and
    return the per-tick dataframe loaded from the JSONL log.
    """
    env = os.environ.copy()

    # Explicitly pin the research knobs we care about.
    env["PARA_INITIAL_Q_TAO"] = "0.0"

    # These env var names must match the overrides in src/config.rs.
    env["PARA_HEDGE_BAND_BASE"] = str(hedge_band_tao)
    env["PARA_HEDGE_MAX_STEP"] = str(4.0 * hedge_band_tao)

    # Nice to have when debugging.
    env.setdefault("RUST_BACKTRACE", "1")

    run_id = f"band{hedge_band_tao:04.1f}_run{run_idx:02d}".replace(".", "p")
    jsonl_path = RUNS_DIR / f"{run_id}.jsonl"

    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--",
        "--ticks",
        str(TICKS),
        "--log-jsonl",
        str(jsonl_path),
    ]

    print(f"\n=== Running {run_id} (hedge_band_base={hedge_band_tao} TAO) ===")
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)

    # Load the per-tick snapshots.
    df = pd.read_json(jsonl_path, lines=True)
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute risk/return metrics from a per-tick dataframe.
    We deliberately keep this very simple and transparent.
    """
    pnl = df["daily_pnl_total"].to_numpy()
    delta = df["dollar_delta_usd"].to_numpy()
    basis = df["basis_usd"].to_numpy()

    final_pnl = float(pnl[-1])

    # Treat daily_pnl_total as an equity curve for drawdown purposes.
    running_max = np.maximum.accumulate(pnl)
    drawdown = pnl - running_max
    max_drawdown = float(drawdown.min())  # <= 0.0

    max_abs_delta = float(np.abs(delta).max())
    max_abs_basis = float(np.abs(basis).max())

    return {
        "final_pnl": final_pnl,
        "max_drawdown": max_drawdown,
        "max_abs_delta_usd": max_abs_delta,
        "max_abs_basis_usd": max_abs_basis,
    }


# ---------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------

def main() -> None:
    rows = []

    for band in HEDGE_BANDS_TAO:
        for run_idx in range(RUNS_PER_BAND):
            df = run_one_sim(band, run_idx)
            metrics = compute_metrics(df)
            row = {
                "hedge_band_tao": band,
                "run_idx": run_idx,
                **metrics,
            }
            rows.append(row)

    df_runs = pd.DataFrame(rows)
    runs_csv = ROOT / "exp01_band_sweep_runs.csv"
    df_runs.to_csv(runs_csv, index=False)
    print(f"\nSaved per-run metrics to {runs_csv}")

    # Aggregate per band.
    summary = (
        df_runs.groupby("hedge_band_tao")
        .agg(
            final_pnl_mean=("final_pnl", "mean"),
            final_pnl_std=("final_pnl", "std"),
            max_drawdown_mean=("max_drawdown", "mean"),
            max_drawdown_min=("max_drawdown", "min"),  # most negative
            max_abs_delta_mean=("max_abs_delta_usd", "mean"),
            max_abs_basis_mean=("max_abs_basis_usd", "mean"),
        )
        .reset_index()
    )

    summary_csv = ROOT / "exp01_band_sweep_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Saved band-sweep summary to {summary_csv}\n")

    print("=== Band sweep summary (PnL / risk) ===")
    print(summary.to_string(index=False))

    # Simple research plot: mean final PnL vs hedge band (with std error bars).
    fig, ax = plt.subplots()
    ax.errorbar(
        summary["hedge_band_tao"],
        summary["final_pnl_mean"],
        yerr=summary["final_pnl_std"].fillna(0.0),
        fmt="o-",
    )
    ax.set_xlabel("hedge_band_base (TAO)")
    ax.set_ylabel("final PnL (USD)")
    ax.set_title("Experiment 01: hedge_band_base sweep")
    fig.tight_layout()

    png_path = ROOT / "exp01_band_sweep_pnl.png"
    fig.savefig(png_path)
    print(f"Saved PnL vs band plot to {png_path}")


if __name__ == "__main__":
    main()
