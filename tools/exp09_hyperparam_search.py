#!/usr/bin/env python3
"""
Exp09 – Local hyper-parameter search around Exp07/Exp08 presets.

Goal
-----
Take the calibrated profiles (conservative / balanced / aggressive) and
search a small hyper-grid around them in:

    - hedge_band_base       (TAO)
    - size_eta              (size risk parameter)
    - vol_ref               (volatility reference)
    - loss_limit_usd        (daily PnL loss limit, positive number)

For each combination we:
    - run `paraphina` for a fixed number of ticks
    - log JSONL ticks
    - extract final_pnl, max_drawdown, kill_switch

Then we aggregate per config to:
    - final_pnl_mean, final_pnl_std
    - max_drawdown_mean, max_drawdown_min
    - kill_prob

We then reuse the Exp07 risk-adjusted objective to rank configs inside
each profile and write:

    runs/exp09_hypersearch/exp09_runs.csv
    runs/exp09_hypersearch/exp09_summary.csv

You can then:
    - inspect the CSVs in VS Code,
    - look at the top rows printed at the end,
    - optionally promote new configs into your Profile presets.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Reuse risk tiers + scoring from Exp07 so everything is consistent.
from exp07_optimal_presets import RISK_TIERS, compute_score  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs" / "exp09_hypersearch"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Logging helpers (same conventions as Exp04)
# ---------------------------------------------------------------------


def load_ticks(jsonl_path: Path) -> pd.DataFrame:
    """Load a JSONL tick log into a pandas DataFrame."""
    records: List[Dict[str, Any]] = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records in {jsonl_path}")
    return pd.DataFrame.from_records(records)


def summarise_run(jsonl_path: Path, loss_limit_usd: float) -> Dict[str, Any]:
    """
    Extract summary risk metrics from a single run log.

    Returns:
        - final_pnl
        - max_drawdown
        - risk_regime   (if available, else 'Unknown')
        - kill_switch   (0/1)
    """
    df = load_ticks(jsonl_path)

    # Coerce PnL to float; fall back to zeros if missing.
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


# ---------------------------------------------------------------------
# Hyper-grid definition
# ---------------------------------------------------------------------

# Profiles to test (must match --profile CLI)
PROFILES = ["conservative", "balanced", "aggressive"]

# Local grid around your current "good" region.
# Feel free to extend/tighten later; this is a safe starting point.
HEDGE_BAND_BASE_GRID = [2.5, 5.0, 7.5]          # TAO
SIZE_ETA_GRID = [0.05, 0.10, 0.20]              # size risk, J(Q)=eQ-0.5ηQ²
VOL_REF_GRID = [0.01, 0.02, 0.03]               # sigma_ref for vol_ratio
LOSS_LIMIT_USD_GRID = [2000.0, 4000.0]          # positive numbers

TICKS = 2_000                                   # horizon per run
SEEDS_PER_COMBO = 3                             # independent sims per config


@dataclass
class Combo:
    profile: str
    hedge_band_base: float
    size_eta: float
    vol_ref: float
    loss_limit_usd: float


def generate_combos() -> List[Combo]:
    combos: List[Combo] = []
    for profile in PROFILES:
        for band in HEDGE_BAND_BASE_GRID:
            for eta in SIZE_ETA_GRID:
                for vol in VOL_REF_GRID:
                    for ll in LOSS_LIMIT_USD_GRID:
                        combos.append(
                            Combo(
                                profile=profile,
                                hedge_band_base=band,
                                size_eta=eta,
                                vol_ref=vol,
                                loss_limit_usd=ll,
                            )
                        )
    return combos


# ---------------------------------------------------------------------
# Running paraphina
# ---------------------------------------------------------------------


def run_single_sim(combo: Combo, run_idx: int) -> Dict[str, Any]:
    """
    Run one Paraphina simulation for a specific combo and seed index.

    Returns a dict with both config and outcome metrics.
    """
    log_name = (
        f"exp09_{combo.profile}"
        f"_band{combo.hedge_band_base:.1f}"
        f"_eta{combo.size_eta:.2f}"
        f"_vol{combo.vol_ref:.2f}"
        f"_loss{combo.loss_limit_usd:.0f}"
        f"_r{run_idx}.jsonl"
    )
    log_path = RUNS_DIR / log_name

    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--release",
        "--",
        "--ticks",
        str(TICKS),
        "--profile",
        combo.profile,
        "--hedge-band-base",
        f"{combo.hedge_band_base:.4f}",
        "--loss-limit-usd",
        f"{combo.loss_limit_usd:.4f}",
        "--log-jsonl",
        str(log_path),
    ]

    env = os.environ.copy()
    # Size and vol go via env overrides (see main.rs build_config_from_env_and_args).
    env["PARAPHINA_SIZE_ETA"] = str(combo.size_eta)
    env["PARAPHINA_VOL_REF"] = str(combo.vol_ref)

    print(
        f"[run] profile={combo.profile:<12} "
        f"band={combo.hedge_band_base:>4.1f} "
        f"eta={combo.size_eta:>4.2f} "
        f"vol={combo.vol_ref:>4.2f} "
        f"loss={combo.loss_limit_usd:>5.0f} "
        f"r={run_idx}"
    )
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)

    metrics = summarise_run(log_path, combo.loss_limit_usd)
    metrics.update(
        {
            "profile": combo.profile,
            "hedge_band_base": combo.hedge_band_base,
            "size_eta": combo.size_eta,
            "vol_ref": combo.vol_ref,
            "loss_limit_usd": combo.loss_limit_usd,
            "run_idx": run_idx,
            "log_path": str(log_path),
        }
    )
    return metrics


# ---------------------------------------------------------------------
# Aggregation / scoring
# ---------------------------------------------------------------------


def aggregate_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "profile",
        "hedge_band_base",
        "size_eta",
        "vol_ref",
        "loss_limit_usd",
    ]

    grouped = runs_df.groupby(group_cols, as_index=False)

    summary = grouped.agg(
        final_pnl_mean=("final_pnl", "mean"),
        final_pnl_std=("final_pnl", "std"),
        max_drawdown_mean=("max_drawdown", "mean"),
        max_drawdown_min=("max_drawdown", "min"),
        kill_prob=("kill_switch", "mean"),
        num_runs=("run_idx", "count"),
    )

    # Attach risk tiers / scores (reuse Exp07 logic)
    tier_by_name = {tier.name: tier for tier in RISK_TIERS}

    scores: List[float] = []
    for _, row in summary.iterrows():
        profile = str(row["profile"])
        tier = tier_by_name.get(profile)
        if tier is None:
            # Fallback: treat unknown profiles as "balanced"
            tier = tier_by_name["balanced"]
        score = compute_score(row, tier)
        scores.append(score)

    summary["score"] = scores

    # Sort by profile then score desc
    summary = summary.sort_values(
        ["profile", "score"], ascending=[True, False]
    ).reset_index(drop=True)

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    combos = generate_combos()
    all_rows: List[Dict[str, Any]] = []

    print(
        f"Exp09 hyper-parameter search: {len(combos)} configs, "
        f"{SEEDS_PER_COMBO} runs each, ticks={TICKS}"
    )
    for combo in combos:
        for run_idx in range(SEEDS_PER_COMBO):
            row = run_single_sim(combo, run_idx)
            all_rows.append(row)

    runs_df = pd.DataFrame(all_rows)
    runs_csv = RUNS_DIR / "exp09_runs.csv"
    runs_df.to_csv(runs_csv, index=False)
    print(f"\n[exp09] Saved per-run metrics -> {runs_csv}")

    summary_df = aggregate_runs(runs_df)
    summary_csv = RUNS_DIR / "exp09_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[exp09] Saved aggregated summary -> {summary_csv}\n")

    # Pretty-print top rows per profile
    print("=== Exp09 – top configs per profile (by risk-adjusted score) ===\n")
    for profile in PROFILES:
        sub = summary_df[summary_df["profile"] == profile].head(5)
        if sub.empty:
            print(f"[{profile}] No rows.\n")
            continue
        print(f"[{profile}] Top 5:")
        print(
            sub[
                [
                    "hedge_band_base",
                    "size_eta",
                    "vol_ref",
                    "loss_limit_usd",
                    "final_pnl_mean",
                    "final_pnl_std",
                    "max_drawdown_mean",
                    "kill_prob",
                    "score",
                ]
            ].to_string(index=False)
        )
        print("")

    print("Done. Use exp09_summary.csv to decide if presets should move.")


if __name__ == "__main__":
    main()
