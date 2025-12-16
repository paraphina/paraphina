#!/usr/bin/env python3
"""
exp06_safe_region_from_exp04.py

Derive scenario-robust *safe starting inventory regions* from exp04 results.

Input
=====
- runs/exp04_multi_scenario/exp04_summary.csv
    (produced by exp04_multi_scenario_stress.py)
- runs/exp02_profile_grid/exp02_profile_grid_summary.csv
    (used via profile_centres.load_profile_centres_from_exp02 to recover
     per-profile daily_loss_limit)

For each:
    profile   ∈ {aggressive, balanced, conservative}
    scenario  ∈ {baseline, low_vol, high_vol}
    init_q_tao ∈ {-40, -20, 0, 20, 40}

exp04_summary.csv already contains:
    pnl_realised_mean
    pnl_unrealised_mean
    pnl_total_mean
    pnl_total_std
    pnl_total_p5
    pnl_total_p95
    max_drawdown_mean
    max_drawdown_std
    time_to_kill_mean
    regime_frac_normal_mean
    regime_frac_warning_mean
    regime_frac_hard_mean
    kill_switch_frac
    num_runs

We combine those with per-profile daily_loss_limit to evaluate *safety
constraints* and identify:

    - which (profile, init_q_tao) values remain safe across *all* scenarios,
    - a ranked list of preferred q0 for each profile.

Safety constraints (configurable below)
=======================================
By default we require, for every (profile, scenario, init_q_tao) cell:

    kill_switch_frac          <= KILL_SWITCH_FRAC_MAX
    regime_frac_hard_mean     <= REGIME_HARD_FRAC_MAX
    max_drawdown_mean         <= MAX_DRAWDOWN_FRAC_OF_LOSS_LIMIT * daily_loss_limit

We then aggregate across scenarios and keep only (profile, init_q_tao) that
are safe for all scenarios.

Outputs
=======
- runs/exp06_safe_region/exp06_safe_region_per_cell.csv
    One row per (profile, scenario, init_q_tao) with safety flags and
    drawdown fraction of daily_loss_limit.

- runs/exp06_safe_region/exp06_safe_region_by_profile.csv
    One row per (profile, init_q_tao) with:
        - safe_across_all_scenarios (bool)
        - worst_case_pnl_total_mean
        - worst_case_max_drawdown_mean
        - worst_case_dd_frac_of_loss_limit
        - max_kill_switch_frac
        - max_regime_frac_hard_mean
        - score (simple risk-adjusted score)

The script also prints a human-readable summary per profile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from profile_centres import (
    PROFILES,
    load_profile_centres_from_exp02,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Maximum acceptable fraction of runs that ever hit the kill switch
KILL_SWITCH_FRAC_MAX: float = 0.10

# Maximum acceptable fraction of ticks in HardLimit regime
REGIME_HARD_FRAC_MAX: float = 0.01

# Maximum acceptable *average* drawdown as a fraction of daily_loss_limit.
# e.g. 0.30 → we tolerate average max drawdown up to 30% of the daily loss limit.
MAX_DRAWDOWN_FRAC_OF_LOSS_LIMIT: float = 0.30

# Risk-adjusted scoring:
#   score = worst_case_pnl_total_mean - DD_PENALTY * worst_case_max_drawdown_mean
DD_PENALTY: float = 0.5


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

EXP02_DIR = ROOT / "runs" / "exp02_profile_grid"
EXP02_SUMMARY_CSV = EXP02_DIR / "exp02_profile_grid_summary.csv"

EXP04_DIR = ROOT / "runs" / "exp04_multi_scenario"
EXP04_SUMMARY_CSV = EXP04_DIR / "exp04_summary.csv"

EXP06_DIR = ROOT / "runs" / "exp06_safe_region"
EXP06_DIR.mkdir(parents=True, exist_ok=True)

EXP06_PER_CELL_CSV = EXP06_DIR / "exp06_safe_region_per_cell.csv"
EXP06_BY_PROFILE_CSV = EXP06_DIR / "exp06_safe_region_by_profile.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_daily_loss_limits() -> Dict[str, float]:
    """
    Load per-profile daily_loss_limit from exp02 profile centres.
    """
    centres = load_profile_centres_from_exp02(EXP02_SUMMARY_CSV)
    loss_limits: Dict[str, float] = {
        profile: centre.daily_loss_limit for profile, centre in centres.items()
    }
    return loss_limits


def attach_safety_columns(df: pd.DataFrame, loss_limits: Dict[str, float]) -> pd.DataFrame:
    """
    Attach per-row safety diagnostics:

        - daily_loss_limit
        - dd_frac_of_loss_limit = max_drawdown_mean / daily_loss_limit
        - safe_cell: whether this (profile, scenario, init_q_tao) satisfies
          all safety constraints individually.
    """
    df = df.copy()

    # Map daily_loss_limit by profile
    df["daily_loss_limit"] = df["profile"].map(loss_limits)

    # Drawdown fraction; guard against zero/NaN loss_limits
    def _dd_frac(row: pd.Series) -> float:
        dd = row.get("max_drawdown_mean", np.nan)
        ll = row.get("daily_loss_limit", np.nan)
        if ll is None or np.isnan(ll) or ll <= 0:
            return np.nan
        return float(dd) / float(ll)

    df["dd_frac_of_loss_limit"] = df.apply(_dd_frac, axis=1)

    # Safety checks per cell
    df["safe_kill_switch"] = df["kill_switch_frac"] <= KILL_SWITCH_FRAC_MAX
    df["safe_regime_hard"] = df["regime_frac_hard_mean"] <= REGIME_HARD_FRAC_MAX
    df["safe_drawdown"] = df["dd_frac_of_loss_limit"] <= MAX_DRAWDOWN_FRAC_OF_LOSS_LIMIT

    df["safe_cell"] = (
        df["safe_kill_switch"]
        & df["safe_regime_hard"]
        & df["safe_drawdown"]
    )

    return df


def aggregate_by_profile(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-cell safety into per-(profile, init_q_tao) view.

    For each (profile, init_q_tao), compute:

        - safe_across_all_scenarios : all(safe_cell)
        - worst_case_pnl_total_mean : min(pnl_total_mean across scenarios)
        - worst_case_max_drawdown_mean: max(max_drawdown_mean across scenarios)
        - worst_case_dd_frac_of_loss_limit: max(dd_frac_of_loss_limit)
        - max_kill_switch_frac: max(kill_switch_frac)
        - max_regime_frac_hard_mean: max(regime_frac_hard_mean)
        - score: worst_case_pnl_total_mean - DD_PENALTY * worst_case_max_drawdown_mean
    """
    group_keys = ["profile", "init_q_tao"]
    grouped = df_cells.groupby(group_keys, dropna=False)

    def _safe_all(x: pd.Series) -> bool:
        return bool(x.all())

    agg = grouped.agg(
        safe_across_all_scenarios=("safe_cell", _safe_all),
        worst_case_pnl_total_mean=("pnl_total_mean", "min"),
        worst_case_max_drawdown_mean=("max_drawdown_mean", "max"),
        worst_case_dd_frac_of_loss_limit=("dd_frac_of_loss_limit", "max"),
        max_kill_switch_frac=("kill_switch_frac", "max"),
        max_regime_frac_hard_mean=("regime_frac_hard_mean", "max"),
        daily_loss_limit=("daily_loss_limit", "first"),
    ).reset_index()

    # Risk-adjusted score
    agg["score"] = (
        agg["worst_case_pnl_total_mean"]
        - DD_PENALTY * agg["worst_case_max_drawdown_mean"]
    )

    return agg


def print_human_summary(df_by_profile: pd.DataFrame) -> None:
    """
    Print a concise summary of safe regions and recommended q0 per profile.
    """
    print("\n=== exp06: Scenario-robust safe regions by profile ===")
    for profile in PROFILES:
        df_prof = df_by_profile[df_by_profile["profile"] == profile].copy()
        if df_prof.empty:
            print(f"\nProfile={profile}: no rows found.")
            continue

        safe_rows = df_prof[df_prof["safe_across_all_scenarios"]]
        safe_q = sorted(safe_rows["init_q_tao"].unique().tolist())

        print(f"\nProfile={profile}")
        print(f"  Safe across all scenarios? q0 grid: {safe_q if safe_q else 'NONE'}")

        if not safe_q:
            # Still show top-3 by score so we know what's "least bad"
            df_top = df_prof.sort_values("score", ascending=False).head(3)
            print("  Top-3 q0 by risk-adjusted score (even if unsafe):")
            for _, row in df_top.iterrows():
                print(
                    f"    q0={row['init_q_tao']:+.1f}, "
                    f"score={row['score']:.3f}, "
                    f"worst_pnl={row['worst_case_pnl_total_mean']:.3f}, "
                    f"worst_dd={row['worst_case_max_drawdown_mean']:.3f}, "
                    f"dd_frac={row['worst_case_dd_frac_of_loss_limit']:.3f}, "
                    f"max_kill_frac={row['max_kill_switch_frac']:.3f}"
                )
            continue

        # Among safe q0, choose best by score
        df_safe = safe_rows.sort_values("score", ascending=False)
        best = df_safe.iloc[0]

        print("  Recommended q0 (best safe by score):")
        print(
            f"    q0={best['init_q_tao']:+.1f}, "
            f"score={best['score']:.3f}, "
            f"worst_pnl={best['worst_case_pnl_total_mean']:.3f}, "
            f"worst_dd={best['worst_case_max_drawdown_mean']:.3f}, "
            f"dd_frac={best['worst_case_dd_frac_of_loss_limit']:.3f}, "
            f"max_kill_frac={best['max_kill_switch_frac']:.3f}, "
            f"max_regime_hard={best['max_regime_frac_hard_mean']:.3f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[exp06] Loading exp04 summary...")
    if not EXP04_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"exp04 summary not found at {EXP04_SUMMARY_CSV}. "
            "Run exp04_multi_scenario_stress.py first."
        )
    df = pd.read_csv(EXP04_SUMMARY_CSV)

    print("[exp06] Loading per-profile daily loss limits from exp02 centres...")
    loss_limits = load_daily_loss_limits()
    for prof, ll in loss_limits.items():
        print(f"[exp06]  Profile={prof}: daily_loss_limit={ll}")

    print("[exp06] Attaching safety diagnostics per cell...")
    df_cells = attach_safety_columns(df, loss_limits)

    print(f"[exp06] Writing per-cell safety view to {EXP06_PER_CELL_CSV}")
    df_cells.to_csv(EXP06_PER_CELL_CSV, index=False)

    print("[exp06] Aggregating to per-profile safe regions...")
    df_by_profile = aggregate_by_profile(df_cells)

    print(f"[exp06] Writing per-profile safe-region view to {EXP06_BY_PROFILE_CSV}")
    df_by_profile.to_csv(EXP06_BY_PROFILE_CSV, index=False)

    print_human_summary(df_by_profile)
    print("\n[exp06] Done.")


if __name__ == "__main__":
    main()
