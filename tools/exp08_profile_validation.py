#!/usr/bin/env python3
"""
Experiment 08: Profile validation / regression.

For each named profile (conservative, balanced, aggressive), this script:

  - runs N simulations via the `paraphina` binary using:
        --profile <PROFILE> --ticks TICKS --hedge-band-base HEDGE_BAND_BASE
  - logs each run to JSONL under runs/exp08_profile_validation/
  - summarises for each run:
        final_pnl, max_drawdown, risk_regime, kill_switch
  - aggregates per profile:
        final_pnl_mean, final_pnl_std,
        max_drawdown_mean,
        kill_prob (fraction of runs with kill_switch = true)
  - writes:
        runs/exp08_profile_validation/exp08_profile_runs.csv
        runs/exp08_profile_validation/exp08_profile_summary.csv

This is a quick sanity / regression harness to see whether the three
profiles behave as intended after any change to the engine / configs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs" / "exp08_profile_validation"

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

PROFILES: List[str] = ["conservative", "balanced", "aggressive"]

TICKS: int = 2000
HEDGE_BAND_BASE: float = 5.0
SEEDS_PER_PROFILE: int = 10

# Fallback loss-limit used only for heuristic kill detection if the JSONL
# log does not contain an explicit `kill_switch` field.
FALLBACK_LOSS_LIMIT_USD: float = 2_000.0


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

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


def summarise_run(jsonl_path: Path, loss_limit_usd: float) -> Dict[str, float]:
    """
    Extract summary risk metrics from a single run log.

    Returns a dict keyed by:
        - final_pnl
        - max_drawdown
        - risk_regime
        - kill_switch (0/1)
    """
    df = load_ticks(jsonl_path)

    # PnL column
    if "daily_pnl_total" in df.columns:
        pnl = df["daily_pnl_total"].astype(float)
    else:
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

    # Kill-switch flag: prefer explicit column, otherwise use heuristic.
    if "kill_switch" in df.columns:
        kill_switch = bool(df["kill_switch"].iloc[-1])
    else:
        kill_switch = final_pnl <= -abs(loss_limit_usd) - 1e-6

    return {
        "final_pnl": final_pnl,
        "max_drawdown": max_drawdown,
        "risk_regime": risk_regime,
        "kill_switch": int(kill_switch),
    }


# ---------------------------------------------------------------------------
# Simulation wrapper
# ---------------------------------------------------------------------------

def run_single(profile: str, run_idx: int) -> Dict[str, object]:
    """
    Run one simulation for a given profile and return per-run metrics.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    log_name = f"exp08_profile-{profile}_r{run_idx}.jsonl"
    log_path = RUNS_DIR / log_name

    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--",
        "--ticks",
        str(TICKS),
        "--hedge-band-base",
        str(HEDGE_BAND_BASE),
        "--profile",
        profile,
        "--log-jsonl",
        str(log_path),
    ]

    print(f"[run] profile={profile:>12}, run_idx={run_idx} -> {log_name}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    metrics = summarise_run(log_path, FALLBACK_LOSS_LIMIT_USD)
    metrics.update(
        {
            "profile": profile,
            "run_idx": run_idx,
        }
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_rows: List[Dict[str, object]] = []

    for profile in PROFILES:
        for run_idx in range(SEEDS_PER_PROFILE):
            row = run_single(profile, run_idx)
            all_rows.append(row)

    runs_df = pd.DataFrame(all_rows)
    runs_csv = RUNS_DIR / "exp08_profile_runs.csv"
    runs_df.to_csv(runs_csv, index=False)
    print(f"\nSaved per-run metrics to {runs_csv}")

    summary = (
        runs_df.groupby("profile", as_index=False)
        .agg(
            final_pnl_mean=("final_pnl", "mean"),
            final_pnl_std=("final_pnl", "std"),
            max_drawdown_mean=("max_drawdown", "mean"),
            max_drawdown_min=("max_drawdown", "min"),
            kill_prob=("kill_switch", "mean"),
        )
        .sort_values("profile")
    )

    summary_csv = RUNS_DIR / "exp08_profile_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Saved profile summary to {summary_csv}\n")

    print("=== Exp08 – Profile validation summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
