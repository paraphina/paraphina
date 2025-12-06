#!/usr/bin/env python3
"""
Experiment 04: Risk regime vs initial position & loss limit.

Sweeps over a grid of:
    - initial_q_tao (TAO),
    - daily loss limit in USD,

runs the `paraphina` binary with JSONL logging, then aggregates:

    - final PnL,
    - max drawdown,
    - final risk regime,
    - kill-switch probability (fraction of runs that ended in kill).

This implementation is robust even if the JSONL log does NOT contain a
`kill_switch` column: in that case we infer kill-switch events from PnL.
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
EXP_DIR = RUNS_DIR / "exp04_risk_regime"


def load_ticks(jsonl_path: Path) -> pd.DataFrame:
    """Load a JSONL tick log into a pandas DataFrame."""
    records = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records in {jsonl_path}")
    return pd.DataFrame.from_records(records)


def summarise_run(jsonl_path: Path, loss_limit_usd: float) -> dict:
    """
    Extract summary risk metrics from a single run log.

    Returns a dict keyed by:
        - final_pnl
        - max_drawdown
        - risk_regime
        - kill_switch (0/1)
    """
    df = load_ticks(jsonl_path)

    # Coerce PnL to float
    if "daily_pnl_total" in df.columns:
        pnl = df["daily_pnl_total"].astype(float)
    else:
        # Failsafe: if somehow missing, treat as zeros
        pnl = pd.Series(np.zeros(len(df), dtype=float))

    final_pnl = float(pnl.iloc[-1])

    # Drawdown = PnL - running max(PnL)
    running_max = pnl.cummax()
    drawdown = pnl - running_max
    max_drawdown = float(drawdown.min())

    # Final risk regime, if logged
    if "risk_regime" in df.columns:
        risk_regime = str(df["risk_regime"].iloc[-1])
    else:
        risk_regime = "Unknown"

    # Kill-switch flag: be robust to missing columns.
    if "kill_switch" in df.columns:
        kill_switch = bool(df["kill_switch"].iloc[-1])
    else:
        # Fallback heuristic: treat a loss beyond the configured limit
        # as a kill-switch event.
        kill_switch = final_pnl <= -abs(loss_limit_usd) - 1e-6

    return {
        "final_pnl": final_pnl,
        "max_drawdown": max_drawdown,
        "risk_regime": risk_regime,
        "kill_switch": int(kill_switch),
    }


def run_single_sim(q0: float, loss_limit_usd: float, run_idx: int) -> dict:
    """
    Run one Paraphina simulation for given (q0, loss_limit) and return metrics.
    """
    RUNS_DIR.mkdir(exist_ok=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    log_name = f"exp04_q{q0:+.0f}_loss{loss_limit_usd:.0f}_r{run_idx}.jsonl"
    log_path = EXP_DIR / log_name

    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--",
        "--ticks",
        "2000",
        "--initial-q-tao",
        str(q0),
        "--hedge-band-base",
        "5.0",
        "--log-jsonl",
        str(log_path),
    ]

    # Pass the loss limit via env so Config can pick it up.
    # Risk engine expects a NEGATIVE daily_loss_limit, and
    # PARAPHINA_DAILY_LOSS_LIMIT_USD is "already negative by convention".
    env = os.environ.copy()
    env["PARAPHINA_DAILY_LOSS_LIMIT_USD"] = str(-abs(loss_limit_usd))

    print(f"[run] q0={q0:+.0f} TAO, loss_limit={loss_limit_usd:.0f} -> {log_name}")
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)

    metrics = summarise_run(log_path, loss_limit_usd)
    metrics.update(
        {
            "initial_q_tao": q0,
            "loss_limit_usd": loss_limit_usd,
            "run_idx": run_idx,
        }
    )
    return metrics


def main() -> None:
    # Grid over initial position and loss limit.
    q0_grid = [-50.0, -25.0, 0.0, 25.0, 50.0]
    loss_limit_grid = [2000.0, 4000.0, 6000.0, 8000.0, 10000.0]
    seeds_per_point = 3  # number of random seeds per grid point

    all_rows = []
    for q0 in q0_grid:
        for loss_limit in loss_limit_grid:
            for run_idx in range(seeds_per_point):
                row = run_single_sim(q0, loss_limit, run_idx)
                all_rows.append(row)

    # Per-run results
    runs_df = pd.DataFrame(all_rows)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    runs_csv = EXP_DIR / "exp04_risk_regime_runs.csv"
    runs_df.to_csv(runs_csv, index=False)
    print(f"\n[exp04] Saved per-run metrics to {runs_csv}")

    # Aggregate across seeds to get mean/std and kill-switch probability.
    summary = (
        runs_df.groupby(["initial_q_tao", "loss_limit_usd"], as_index=False)
        .agg(
            final_pnl_mean=("final_pnl", "mean"),
            final_pnl_std=("final_pnl", "std"),
            max_drawdown_mean=("max_drawdown", "mean"),
            max_drawdown_min=("max_drawdown", "min"),
            kill_prob=("kill_switch", "mean"),
        )
        .sort_values(["initial_q_tao", "loss_limit_usd"])
    )

    summary_csv = EXP_DIR / "exp04_risk_regime_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[exp04] Saved risk-regime summary to {summary_csv}\n")

    print("=== Exp04 summary table (final_pnl_mean, kill_prob, etc.) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
