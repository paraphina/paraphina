#!/usr/bin/env python3
"""
Experiment 02: q0 vs hedge_band_base heatmap.

For each (initial_q_tao, hedge_band_tao) pair, we:
  - Run the Rust sim via `cargo run -- ... --log-jsonl <path>`
  - Load the JSONL tick log
  - Compute:
      * final_pnl (last tick)
      * max_drawdown (on daily_pnl_total)
      * max_abs_delta_usd
      * max_abs_basis_usd
  - Aggregate to:
      * exp02_hedge_band_heatmap_runs.csv      (one row per run)
      * exp02_hedge_band_heatmap_summary.csv   (one row per grid point)
      * exp02_hedge_band_heatmap_pnl.png       (heatmap of final PnL mean)
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Grid + sim parameters
# ---------------------------------------------------------------------------

TICKS = 1_000

# q0 grid (TAO) – you can tune these later
Q0_GRID = [-50.0, -25.0, 0.0, 25.0, 50.0]

# Hedge band grid (TAO) – consistent with Exp01
BAND_GRID = [2.5, 5.0, 7.5, 10.0, 15.0]

# How many seeds per grid point (currently sim is deterministic, so 1 is fine)
RUNS_PER_CELL = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_single_sim(initial_q_tao: float, hedge_band_tao: float, run_idx: int) -> Path:
    """
    Invoke `cargo run` once with the given parameters and return the JSONL path.
    """
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    log_path = runs_dir / f"exp02_q{initial_q_tao:+.0f}_band{hedge_band_tao:.1f}_r{run_idx}.jsonl"

    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--",
        "--ticks",
        str(TICKS),
        # IMPORTANT: use --flag=value so negative q0 isn’t parsed as a new flag
        f"--initial-q-tao={initial_q_tao}",
        "--hedge-band-base",
        str(hedge_band_tao),
        "--log-jsonl",
        str(log_path),
    ]

    env = dict(os.environ)
    # If the Rust side honours this, it will suppress per-tick printing.
    env.setdefault("PARAPHINA_SILENT", "1")

    print(f"[run] q0={initial_q_tao:+.1f} TAO, band={hedge_band_tao:.1f} TAO → {log_path}")
    subprocess.run(cmd, check=True, env=env)

    return log_path


def load_run_metrics(log_path: Path) -> dict:
    """
    Load a single JSONL tick log and compute basic PnL / risk metrics.
    Assumes the FileSink emits fields named:

        daily_pnl_total
        dollar_delta_usd
        basis_usd

    which matches the Rust state fields used in `print_state_snapshot`.
    """
    with log_path.open() as f:
        rows = [json.loads(line) for line in f]

    if not rows:
        raise ValueError(f"Empty tick log: {log_path}")

    df = pd.DataFrame(rows)

    # Final PnL = last daily_pnl_total
    pnl = df["daily_pnl_total"]
    final_pnl = float(pnl.iloc[-1])

    # Max drawdown on the PnL curve
    cummax = pnl.cummax()
    drawdown = cummax - pnl
    max_drawdown = float(drawdown.max())

    # Max absolute dollar delta and basis over the run
    max_abs_delta = float(df["dollar_delta_usd"].abs().max())
    max_abs_basis = float(df["basis_usd"].abs().max())

    return {
        "final_pnl": final_pnl,
        "max_drawdown": max_drawdown,
        "max_abs_delta_usd": max_abs_delta,
        "max_abs_basis_usd": max_abs_basis,
    }


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------

def main() -> None:
    run_records = []

    for q0 in Q0_GRID:
        for band in BAND_GRID:
            for run_idx in range(1, RUNS_PER_CELL + 1):
                log_path = run_single_sim(q0, band, run_idx)
                metrics = load_run_metrics(log_path)
                run_records.append(
                    {
                        "initial_q_tao": q0,
                        "hedge_band_tao": band,
                        "run_idx": run_idx,
                        **metrics,
                    }
                )

    runs_df = pd.DataFrame(run_records)
    runs_csv = Path("exp02_hedge_band_heatmap_runs.csv")
    runs_df.to_csv(runs_csv, index=False)
    print(f"\nSaved per-run metrics to {runs_csv}")

    # Aggregate to one row per (q0, band)
    summary = (
        runs_df.groupby(["initial_q_tao", "hedge_band_tao"], as_index=False)
        .agg(
            final_pnl_mean=("final_pnl", "mean"),
            final_pnl_std=("final_pnl", "std"),
            max_drawdown_mean=("max_drawdown", "mean"),
            max_drawdown_min=("max_drawdown", "min"),
            max_abs_delta_mean=("max_abs_delta_usd", "mean"),
            max_abs_basis_mean=("max_abs_basis_usd", "mean"),
        )
    )

    summary_csv = Path("exp02_hedge_band_heatmap_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Saved heatmap summary to {summary_csv}")

    # Build a PnL heatmap (rows = q0, cols = band)
    pivot = summary.pivot(
        index="initial_q_tao",
        columns="hedge_band_tao",
        values="final_pnl_mean",
    ).sort_index().sort_index(axis=1)

    fig, ax = plt.subplots()
    im = ax.imshow(pivot.values, origin="lower", aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("hedge_band_tao (TAO)")
    ax.set_ylabel("initial_q_tao (TAO)")
    ax.set_title("Exp02: final PnL mean heatmap")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("final_pnl_mean (USD)")

    fig.tight_layout()
    heatmap_png = Path("exp02_hedge_band_heatmap_pnl.png")
    fig.savefig(heatmap_png, dpi=150)
    print(f"Saved PnL heatmap PNG to {heatmap_png}")

    print("\n=== Exp02 summary table (final_pnl_mean etc.) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
