#!/usr/bin/env python3
"""
Exp05: Calibration report for Paraphina research runs.

This script reads the summary CSVs produced by Exp01–Exp04 and prints
ranked tables of the "best" parameter settings according to simple,
transparent metrics.

It does NOT change any code or configs; it is strictly a reporting /
decision-support tool for humans.

Expected CSVs (all in repo root):

    - exp01_band_sweep_summary.csv
    - exp02_hedge_band_heatmap_summary.csv
    - exp03_size_eta_vol_summary.csv
    - exp04_risk_regime_summary.csv

You can regenerate any of these by re-running the corresponding tools:

    - tools/batch_runs.py                         (Exp01)
    - tools/exp02_hedge_band_heatmap.py          (Exp02)
    - tools/exp03_size_eta_vol_sweep.py          (Exp03)
    - tools/exp04_risk_regime_sweep.py           (Exp04)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def load_csv(name: str) -> Optional[pd.DataFrame]:
    path = ROOT / name
    if not path.exists():
        print(f"[warn] CSV not found, skipping: {path}")
        return None
    df = pd.read_csv(path)
    print(f"[info] Loaded {name} with shape {df.shape}")
    return df


def show_top(
    df: pd.DataFrame,
    sort_col: str,
    cols: list[str],
    n: int = 10,
    title: str = "",
) -> None:
    if df is None:
        return

    print()
    print("=" * 80)
    if title:
        print(title)
        print("-" * 80)

    df_sorted = df.sort_values(sort_col, ascending=False)
    top = df_sorted[cols].head(n)

    # Pretty print with aligned columns
    print(top.to_string(index=False))
    print("=" * 80)
    print()


def analyse_exp01(df: Optional[pd.DataFrame]) -> None:
    """
    Exp01: hedge_band_base sweep with q0 = 0.
    We care primarily about final_pnl_mean, but keep risk columns visible.
    """
    if df is None:
        return

    cols = [
        "hedge_band_tao",
        "final_pnl_mean",
        "final_pnl_std",
        "max_drawdown_mean",
        "max_abs_delta_mean",
        "max_abs_basis_mean",
    ]
    show_top(
        df,
        sort_col="final_pnl_mean",
        cols=cols,
        n=10,
        title="Exp01 – Hedge band sweep (q0 = 0): top rows by final_pnl_mean",
    )


def analyse_exp02(df: Optional[pd.DataFrame]) -> None:
    """
    Exp02: 2D heatmap over (initial_q_tao, hedge_band_tao).

    We show the best rows by final_pnl_mean but keep q0 and band in view.
    """
    if df is None:
        return

    cols = [
        "initial_q_tao",
        "hedge_band_tao",
        "final_pnl_mean",
        "final_pnl_std",
        "max_drawdown_mean",
        "max_abs_delta_mean",
        "max_abs_basis_mean",
    ]
    show_top(
        df,
        sort_col="final_pnl_mean",
        cols=cols,
        n=15,
        title=(
            "Exp02 – Initial position vs hedge band heatmap: "
            "top rows by final_pnl_mean"
        ),
    )


def analyse_exp03(df: Optional[pd.DataFrame]) -> None:
    """
    Exp03: size_eta vs vol_ref grid.

    Here we’re asking: for different aggressiveness (size_eta) and
    volatility reference levels, which combos look best on PnL and risk?
    """
    if df is None:
        return

    cols = [
        "size_eta",
        "vol_ref",
        "final_pnl_mean",
        "final_pnl_std",
        "max_drawdown_mean",
        "max_abs_delta_mean",
        "max_abs_basis_mean",
    ]
    show_top(
        df,
        sort_col="final_pnl_mean",
        cols=cols,
        n=15,
        title=(
            "Exp03 – Size_eta vs vol_ref grid: "
            "top rows by final_pnl_mean"
        ),
    )


def analyse_exp04(df: Optional[pd.DataFrame]) -> None:
    """
    Exp04: risk regime sweep (initial_q_tao, loss_limit_usd).

    We care about:
      - final_pnl_mean  (higher is better),
      - kill_prob       (must be small),
      - max_drawdown_mean for context.
    """
    if df is None:
        return

    # Make sure columns exist even if CSV schema is slightly different
    expected_cols = [
        "initial_q_tao",
        "loss_limit_usd",
        "final_pnl_mean",
        "final_pnl_std",
        "max_drawdown_mean",
        "kill_prob",
    ]
    cols = [c for c in expected_cols if c in df.columns]

    # Prefer rows with low kill_prob; sort by kill_prob ASC then pnl DESC
    df_sorted = df.sort_values(
        by=["kill_prob", "final_pnl_mean"], ascending=[True, False]
    )

    print()
    print("=" * 80)
    print(
        "Exp04 – Risk regime sweep: rows ranked by (low kill_prob, high final_pnl_mean)"
    )
    print("-" * 80)
    print(df_sorted[cols].head(20).to_string(index=False))
    print("=" * 80)
    print()


def main() -> None:
    print("=== Exp05: Calibration report from existing experiments ===")

    exp01 = load_csv("exp01_band_sweep_summary.csv")
    exp02 = load_csv("exp02_hedge_band_heatmap_summary.csv")
    exp03 = load_csv("exp03_size_eta_vol_summary.csv")
    exp04 = load_csv("exp04_risk_regime_summary.csv")

    analyse_exp01(exp01)
    analyse_exp02(exp02)
    analyse_exp03(exp03)
    analyse_exp04(exp04)

    print("Done. Use these ranked tables to choose candidate configs.\n")


if __name__ == "__main__":
    main()
