#!/usr/bin/env python3
"""
exp12_profile_safety_tuner.py

Exp12 – Profile safety tuner based on research alignment (Exp11).

Goal
====
Given:
  - Exp07 optimal presets (one or more presets per risk_tier).
  - Exp11 research alignment metrics (empirical PnL / drawdown / kill_prob).

Produce:
  - A "tuned presets" CSV with adjusted configs that pull any
    out-of-budget profiles back toward their world-model risk budgets.
  - A human-readable tuning report CSV with before/after comparisons.

Inputs
======
1) Presets CSV from Exp07:
       runs/exp07_optimal_presets/exp07_presets.csv

   This is expected to contain (among others) columns:
       name, risk_tier, rank_within_tier,
       initial_q_tao, loss_limit_usd,
       hedge_band_base, size_eta, vol_ref, delta_limit_usd_base,
       final_pnl_mean, max_drawdown_mean, kill_prob, ...

2) Research alignment CSV from Exp11:
       runs/exp11_research_alignment/exp11_alignment.csv

   One row per risk_tier, with columns:
       risk_tier, n_runs,
       budget_max_kill_prob, budget_max_dd_abs, budget_min_final_pnl_mean,
       emp_final_pnl_mean, emp_max_drawdown_mean, emp_kill_prob,
       emp_kill_ok, emp_dd_ok, emp_pnl_ok, ...

Tuning logic (high-level)
=========================
For each RiskTier in exp07_optimal_presets.RISK_TIERS:

  - Take the best preset for that tier (rank_within_tier == 1 if present).
  - Look up the empirical alignment row from Exp11.

  - Compute risk_scale in (0.1, 1.0]:
        dd_ratio   = emp_dd / dd_budget   (if available)
        kill_ratio = emp_kill / kill_budget (if available)

        If dd_ratio   > 1.05 → risk_scale <= 1 / dd_ratio
        If kill_ratio > 1.05 → risk_scale <= 1 / kill_ratio

        risk_scale is never increased above 1.0; we only reduce risk.

  - Apply risk_scale to key exposure knobs:
        hedge_band_base      *= risk_scale
        size_eta             *= risk_scale
        initial_q_tao        *= risk_scale
        delta_limit_usd_base *= risk_scale

  - Optionally tighten loss_limit_usd:
        if loss_limit_usd > 1.1 * max_drawdown_abs:
            loss_limit_usd = max_drawdown_abs

Outputs
=======
- Tuned presets:
      runs/exp12_profile_safety_tuner/exp12_tuned_presets.csv

  Same columns as exp07_presets.csv, plus:
      base_name_exp12      – original preset name
      risk_scale_exp12     – applied risk scale factor

- Tuning report:
      runs/exp12_profile_safety_tuner/exp12_tuning_report.csv

  One row per risk_tier with:
      budgets, empirical metrics, ratios vs budget,
      before/after key knobs (band, size_eta, delta_limit, loss_limit).

Usage
=====
# Basic usage with defaults:
python tools/exp12_profile_safety_tuner.py

# Explicit paths:
python tools/exp12_profile_safety_tuner.py \
    --presets-csv runs/exp07_optimal_presets/exp07_presets.csv \
    --alignment-csv runs/exp11_research_alignment/exp11_alignment.csv \
    --out-dir runs/exp12_profile_safety_tuner
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from exp07_optimal_presets import (  # type: ignore
    RISK_TIERS,
    RiskTier,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"

DEFAULT_PRESETS_CSV = RUNS_DIR / "exp07_optimal_presets" / "exp07_presets.csv"
DEFAULT_ALIGN_CSV = RUNS_DIR / "exp11_research_alignment" / "exp11_alignment.csv"
DEFAULT_OUT_DIR = RUNS_DIR / "exp12_profile_safety_tuner"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        if np.isnan(f):
            return None
        return f
    except Exception:
        return None


def load_presets(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise SystemExit(f"[exp12] Presets CSV does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    if "risk_tier" not in df.columns:
        raise SystemExit(
            f"[exp12] Presets CSV {csv_path} has no 'risk_tier' column; "
            "did Exp07 run correctly?"
        )
    return df


def load_alignment(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise SystemExit(f"[exp12] Alignment CSV does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    if "risk_tier" not in df.columns:
        raise SystemExit(
            f"[exp12] Alignment CSV {csv_path} has no 'risk_tier' column; "
            "did Exp11 run correctly?"
        )
    return df


# ---------------------------------------------------------------------------
# Core tuning logic
# ---------------------------------------------------------------------------


def build_tuned_presets(
    presets_df: pd.DataFrame, align_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (tuned_presets_df, tuning_report_df).
    """
    tuned_rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    for tier in RISK_TIERS:
        tier_mask = presets_df["risk_tier"] == tier.name
        tier_df = presets_df[tier_mask]
        if tier_df.empty:
            print(f"[exp12] WARNING: No presets found for tier '{tier.name}'.")
            continue

        # Pick the best preset for this tier: rank_within_tier == 1 if present.
        if "rank_within_tier" in tier_df.columns:
            tier_df_sorted = tier_df.sort_values("rank_within_tier")
        else:
            tier_df_sorted = tier_df

        base = tier_df_sorted.iloc[0]
        base_dict = base.to_dict()
        base_name = str(base_dict.get("name", tier.name))

        # Matching alignment row from Exp11.
        align_row = align_df[align_df["risk_tier"] == tier.name]
        if align_row.empty:
            print(
                f"[exp12] WARNING: No alignment row found for tier '{tier.name}' "
                f"in Exp11 alignment CSV."
            )
            pnl_mean = dd_mean = kill_prob = None
        else:
            a = align_row.iloc[0]
            pnl_mean = _safe_float(a.get("emp_final_pnl_mean"))
            dd_mean = _safe_float(a.get("emp_max_drawdown_mean"))
            kill_prob = _safe_float(a.get("emp_kill_prob"))

        # Compute risk scale from drawdown + kill violations.
        risk_scale = 1.0
        dd_ratio: Optional[float] = None
        kill_ratio: Optional[float] = None

        dd_budget = tier.max_drawdown_abs
        if dd_mean is not None and dd_budget and dd_budget > 0:
            # Use absolute magnitude for drawdown.
            dd_abs = abs(dd_mean)
            dd_ratio = dd_abs / dd_budget
            # Only shrink risk if more than 5% over budget.
            if dd_ratio > 1.05:
                risk_scale = min(risk_scale, max(0.1, 1.0 / dd_ratio))

        kill_budget = tier.max_kill_prob
        if kill_prob is not None and kill_budget and kill_budget > 0:
            kill_ratio = kill_prob / kill_budget
            if kill_ratio > 1.05:
                risk_scale = min(risk_scale, max(0.1, 1.0 / kill_ratio))

        risk_scale = float(risk_scale)

        # Build tuned preset row.
        tuned = dict(base_dict)
        tuned_name = f"{base_name}_exp12_tuned"
        tuned["base_name_exp12"] = base_name
        tuned["name"] = tuned_name
        tuned["risk_scale_exp12"] = risk_scale

        def scale_field(col: str) -> None:
            if col in tuned:
                val = _safe_float(tuned[col])
                if val is not None:
                    tuned[col] = val * risk_scale

        # Scale the main exposure knobs.
        for col in ["hedge_band_base", "size_eta", "initial_q_tao", "delta_limit_usd_base"]:
            scale_field(col)

        # loss_limit_usd: if significantly above budgeted drawdown, clamp.
        if "loss_limit_usd" in tuned:
            orig_ll = _safe_float(base_dict.get("loss_limit_usd"))
            if orig_ll is not None and dd_budget and dd_budget > 0:
                if orig_ll > dd_budget * 1.10:
                    tuned["loss_limit_usd"] = dd_budget

        tuned_rows.append(tuned)

        report_rows.append(
            {
                "risk_tier": tier.name,
                "base_name": base_name,
                "tuned_name": tuned_name,
                "risk_scale_exp12": risk_scale,
                "budget_max_dd_abs": dd_budget,
                "emp_max_drawdown_mean": dd_mean,
                "dd_ratio_emp_over_budget": dd_ratio,
                "budget_max_kill_prob": kill_budget,
                "emp_kill_prob": kill_prob,
                "kill_ratio_emp_over_budget": kill_ratio,
                "budget_min_final_pnl_mean": tier.min_final_pnl_mean,
                "emp_final_pnl_mean": pnl_mean,
                "hedge_band_base_orig": base_dict.get("hedge_band_base"),
                "hedge_band_base_tuned": tuned.get("hedge_band_base"),
                "size_eta_orig": base_dict.get("size_eta"),
                "size_eta_tuned": tuned.get("size_eta"),
                "delta_limit_usd_base_orig": base_dict.get("delta_limit_usd_base"),
                "delta_limit_usd_base_tuned": tuned.get("delta_limit_usd_base"),
                "loss_limit_usd_orig": base_dict.get("loss_limit_usd"),
                "loss_limit_usd_tuned": tuned.get("loss_limit_usd"),
            }
        )

    tuned_df = pd.DataFrame(tuned_rows)
    report_df = pd.DataFrame(report_rows)
    return tuned_df, report_df


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Exp12: Profile safety tuner based on Exp11 research alignment."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--presets-csv",
        type=Path,
        default=DEFAULT_PRESETS_CSV,
        help="Path to Exp07 presets CSV.",
    )
    p.add_argument(
        "--alignment-csv",
        type=Path,
        default=DEFAULT_ALIGN_CSV,
        help="Path to Exp11 alignment CSV.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for Exp12 tuned presets + report.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    presets_df = load_presets(args.presets_csv)
    align_df = load_alignment(args.alignment_csv)

    tuned_df, report_df = build_tuned_presets(presets_df, align_df)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tuned_path = out_dir / "exp12_tuned_presets.csv"
    report_path = out_dir / "exp12_tuning_report.csv"

    tuned_df.to_csv(tuned_path, index=False)
    report_df.to_csv(report_path, index=False)

    print(f"[exp12] Wrote tuned presets  -> {tuned_path}")
    print(f"[exp12] Wrote tuning report -> {report_path}")
    print("\n=== Exp12 – Profile safety tuning summary ===\n")

    if report_df.empty:
        print("[exp12] No tiers were tuned (no presets / no alignment rows).")
    else:
        for row in report_df.itertuples(index=False):
            print(f"[{row.risk_tier}] {row.base_name!r} -> {row.tuned_name!r}")
            if row.emp_max_drawdown_mean is not None and row.dd_ratio_emp_over_budget:
                print(
                    "  dd: emp_mean={:.2f}, budget_max={:.2f}, ratio={:.2f}x".format(
                        row.emp_max_drawdown_mean,
                        row.budget_max_dd_abs,
                        row.dd_ratio_emp_over_budget,
                    )
                )
            if row.emp_kill_prob is not None and row.budget_max_kill_prob:
                print(
                    "  kill: emp_prob={:.3f}, budget_max={:.3f}".format(
                        row.emp_kill_prob, row.budget_max_kill_prob
                    )
                )
            print(
                "  knobs: band {:.4g} -> {:.4g}, eta {:.4g} -> {:.4g}, "
                "delta_limit {:.4g} -> {:.4g}, loss_limit {} -> {}".format(
                    row.hedge_band_base_orig,
                    row.hedge_band_base_tuned,
                    row.size_eta_orig,
                    row.size_eta_tuned,
                    row.delta_limit_usd_base_orig,
                    row.delta_limit_usd_base_tuned,
                    row.loss_limit_usd_orig,
                    row.loss_limit_usd_tuned,
                )
            )
            print(f"  risk_scale_exp12 = {row.risk_scale_exp12:.3f}\n")

    print("[exp12] Done.")


if __name__ == "__main__":
    main()
