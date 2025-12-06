#!/usr/bin/env python3
"""
Experiment 03: size_eta / vol_ref sweep.

Goal:
    Map out how PnL and risk respond to:
        - MM size aggressiveness   (mm.size_eta)
        - effective vol ratio via  (vol_ref)

Setup:
    - Underlying config = Reference Regime R0.
    - initial_q_tao = 0 TAO
    - hedge_band_base = 5 TAO
    - TICKS_PER_RUN = 1_000  (short runs for grid scan)
    - RUNS_PER_POINT = 5     (average over seeds)

Outputs:
    - exp03_size_eta_vol_runs.csv   (per-run metrics)
    - exp03_size_eta_vol_summary.csv (grid-aggregated)
    - exp03_size_eta_vol_heatmap_pnl.png (Pnl heatmap)
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"

TICKS_PER_RUN = 1_000
RUNS_PER_POINT = 5

# Grid for size_eta (η) and vol_ref (σ_ref).
SIZE_ETA_GRID = [0.05, 0.10, 0.20, 0.40]
VOL_REF_GRID = [0.01, 0.02, 0.04, 0.08]  # daily vols 1%, 2%, 4%, 8%


def ensure_runs_dir() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def run_single_sim(size_eta: float, vol_ref: float, run_idx: int) -> Path:
    """
    Fire a single Rust simulation under the given (size_eta, vol_ref),
    returning the log path.
    """
    ensure_runs_dir()

    log_name = (
        f"exp03_eta{size_eta:.3f}_vol{vol_ref:.3f}_r{run_idx}.jsonl"
        .replace(".", "p")
    )
    log_path = RUNS_DIR / log_name

    env = os.environ.copy()
    env["PARAPHINA_MM_SIZE_ETA"] = f"{size_eta:.8f}"
    env["PARAPHINA_VOL_REF"] = f"{vol_ref:.8f}"

    # Lock other knobs to the R0 baseline explicitly.
    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--",
        "--ticks",
        str(TICKS_PER_RUN),
        "--initial-q-tao",
        "0",
        "--hedge-band-base",
        "5.0",
        "--log-jsonl",
        str(log_path),
    ]

    print(
        f"[run] η={size_eta:.3f}, vol_ref={vol_ref:.3f}, "
        f"ticks={TICKS_PER_RUN}, log={log_path.name}"
    )
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)

    return log_path


def summarise_log(log_path: Path) -> dict:
    """
    Read a JSONL tick log and compute summary metrics consistent with
    the other research harnesses.
    """
    df = pd.read_json(log_path, lines=True)

    pnl = df["daily_pnl_total"].astype(float)
    delta = df["dollar_delta_usd"].astype(float)
    basis = df["basis_usd"].astype(float)

    final_pnl = float(pnl.iloc[-1])

    # Max drawdown in equity curve.
    running_max = pnl.cummax()
    drawdown = running_max - pnl
    max_drawdown = float(drawdown.max())

    max_abs_delta = float(delta.abs().max())
    max_abs_basis = float(basis.abs().max())

    return {
        "final_pnl": final_pnl,
        "max_drawdown": max_drawdown,
        "max_abs_delta_usd": max_abs_delta,
        "max_abs_basis_usd": max_abs_basis,
    }


def main() -> None:
    ensure_runs_dir()

    rows = []

    for size_eta in SIZE_ETA_GRID:
        for vol_ref in VOL_REF_GRID:
            for run_idx in range(RUNS_PER_POINT):
                log_path = run_single_sim(size_eta, vol_ref, run_idx)
                metrics = summarise_log(log_path)
                row = {
                    "size_eta": size_eta,
                    "vol_ref": vol_ref,
                    "run_idx": run_idx,
                    **metrics,
                }
                rows.append(row)

    runs_df = pd.DataFrame(rows)
    runs_csv = ROOT / "exp03_size_eta_vol_runs.csv"
    runs_df.to_csv(runs_csv, index=False)
    print(f"Saved per-run metrics to {runs_csv}")

    # Aggregate over seeds: one row per (size_eta, vol_ref).
    summary = (
        runs_df
        .groupby(["size_eta", "vol_ref"], as_index=False)
        .agg(
            final_pnl_mean=("final_pnl", "mean"),
            final_pnl_std=("final_pnl", "std"),
            max_drawdown_mean=("max_drawdown", "mean"),
            max_drawdown_min=("max_drawdown", "min"),
            max_abs_delta_mean=("max_abs_delta_usd", "mean"),
            max_abs_basis_mean=("max_abs_basis_usd", "mean"),
        )
    )

    summary_csv = ROOT / "exp03_size_eta_vol_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Saved grid summary to {summary_csv}")

    # ---- Plot heatmap of mean PnL over the grid ----
    pivot = summary.pivot(index="size_eta", columns="vol_ref",
                          values="final_pnl_mean")
    x_vals = pivot.columns.to_numpy()
    y_vals = pivot.index.to_numpy()
    Z = pivot.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[
            x_vals.min(),
            x_vals.max(),
            y_vals.min(),
            y_vals.max(),
        ],
    )

    ax.set_xlabel("vol_ref (σ_ref)")
    ax.set_ylabel("size_eta (η)")
    ax.set_title("Experiment 03: mean final PnL over (η, vol_ref) grid")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean final PnL (USD)")

    heatmap_path = ROOT / "exp03_size_eta_vol_heatmap_pnl.png"
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)

    print(f"Saved PnL heatmap to {heatmap_path}")

    # Also pretty-print the summary table into the terminal.
    print("\n=== Exp03 summary table (final_pnl_mean etc.) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
