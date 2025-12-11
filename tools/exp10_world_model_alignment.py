#!/usr/bin/env python3
"""
Exp10 – World-model alignment report for Paraphina profiles.

Purpose
=======
Take the calibrated profile presets (from Exp07) and the empirical
validation results (from Exp08), and check how well each risk tier
(conservative / balanced / aggressive) aligns with the *world-model*
risk budgets encoded in the Exp07 RiskTier definitions.

Inputs
======
1) Presets selected by Exp07:
       runs/exp07_optimal_presets/exp07_presets.csv

   Each row usually contains:
       - name, risk_tier, rank_within_tier
       - initial_q_tao, loss_limit_usd
       - hedge_band_base, size_eta, vol_ref, delta_limit_usd_base
       - final_pnl_mean, final_pnl_std, max_drawdown_mean, kill_prob
   (metrics are from the Exp04 calibration grid used during selection)

2) Profile validation summary from Exp08:
       runs/exp08_profile_validation/exp08_profile_summary.csv

   Typically has one row per profile with:
       profile, final_pnl_mean, final_pnl_std,
       max_drawdown_mean, max_drawdown_min, kill_prob

3) World-model budgets (FROM CODE, NOT FILE):
   Imported from exp07_optimal_presets.RISK_TIERS:

       max_kill_prob      – allowed kill-switch probability
       max_drawdown_abs   – allowed average drawdown magnitude (USD)
       min_final_pnl_mean – minimum acceptable mean PnL (USD)
       lambda_*           – weights for the risk-adjusted score
                            (used here only for an alignment score).

Outputs
=======
By default (no args) this script reads the files above and writes:

    runs/exp10_world_model_alignment/exp10_alignment.csv

Each row represents a risk tier and includes:

    - World-model budgets (max_kill_prob, max_drawdown_abs, min_final_pnl_mean)
    - Design-phase metrics from Exp07 (design_* columns)
    - Validation metrics from Exp08 (val_* columns, if available)
    - Boolean flags:
          design_kill_ok, design_dd_ok, design_pnl_ok
          val_kill_ok,    val_dd_ok,    val_pnl_ok
    - Margins vs budget (positive = safe side, negative = violation)
    - Simple alignment scores for design and validation.

Nothing in Exp10 runs the Rust engine; it is a *pure analysis* harness.

Usage
=====
# Basic (use default CSV locations)
python tools/exp10_world_model_alignment.py

# Explicit input paths + custom output directory
python tools/exp10_world_model_alignment.py \
    --presets-csv runs/exp07_optimal_presets/exp07_presets.csv \
    --profile-summary-csv runs/exp08_profile_validation/exp08_profile_summary.csv \
    --out-dir runs/exp10_world_model_alignment

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from exp07_optimal_presets import (  # type: ignore
    RISK_TIERS,
    RiskTier,
    compute_score,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
DEFAULT_PRESETS_CSV = RUNS_DIR / "exp07_optimal_presets" / "exp07_presets.csv"
DEFAULT_PROFILE_SUMMARY_CSV = (
    RUNS_DIR / "exp08_profile_validation" / "exp08_profile_summary.csv"
)
DEFAULT_OUT_DIR = RUNS_DIR / "exp10_world_model_alignment"


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

PROFILE_COL_CANDIDATES = ["profile", "risk_tier", "tier"]

FINAL_PNL_MEAN_CANDIDATES = ["final_pnl_mean", "pnl_mean", "mean_pnl"]
MAX_DD_MEAN_CANDIDATES = ["max_drawdown_mean", "drawdown_mean", "dd_mean"]
KILL_PROB_CANDIDATES = ["kill_prob", "kill_probability", "p_kill"]


def _pick_col(df: pd.DataFrame, candidates: List[str], what: str) -> str:
    """Pick first matching column (case-insensitive) or raise KeyError."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    raise KeyError(
        f"Could not find column for {what}. "
        f"Tried {candidates}, have {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_presets(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise SystemExit(f"[exp10] Presets CSV does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    if "risk_tier" not in df.columns:
        raise SystemExit(
            f"[exp10] Presets CSV {csv_path} has no 'risk_tier' column; "
            "did Exp07 run correctly?"
        )
    return df


def load_profile_summary(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.is_file():
        print(
            f"[exp10] WARNING: Validation summary CSV not found: {csv_path}\n"
            "        Proceeding without validation metrics."
        )
        return None

    df = pd.read_csv(csv_path)
    return df


# ---------------------------------------------------------------------------
# Alignment logic
# ---------------------------------------------------------------------------


@dataclass
class AlignmentRow:
    risk_tier: str
    preset_name: Optional[str]
    rank_within_tier: Optional[int]

    # Core config
    initial_q_tao: Optional[float]
    loss_limit_usd: Optional[float]
    hedge_band_base: Optional[float]
    size_eta: Optional[float]
    vol_ref: Optional[float]
    delta_limit_usd_base: Optional[float]

    # World-model budgets
    budget_max_kill_prob: float
    budget_max_dd_abs: float
    budget_min_final_pnl_mean: float

    # Design-phase metrics (from Exp07)
    design_final_pnl_mean: Optional[float]
    design_max_drawdown_mean: Optional[float]
    design_kill_prob: Optional[float]
    design_kill_ok: Optional[bool]
    design_dd_ok: Optional[bool]
    design_pnl_ok: Optional[bool]
    design_kill_margin: Optional[float]
    design_dd_margin: Optional[float]
    design_pnl_margin: Optional[float]
    design_score: Optional[float]

    # Validation metrics (from Exp08, if available)
    val_final_pnl_mean: Optional[float]
    val_max_drawdown_mean: Optional[float]
    val_kill_prob: Optional[float]
    val_kill_ok: Optional[bool]
    val_dd_ok: Optional[bool]
    val_pnl_ok: Optional[bool]
    val_kill_margin: Optional[float]
    val_dd_margin: Optional[float]
    val_pnl_margin: Optional[float]
    val_score: Optional[float]


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        if np.isnan(f):
            return None
        return f
    except Exception:
        return None


def _eval_against_budget(
    pnl_mean: Optional[float],
    dd_mean: Optional[float],
    kill_prob: Optional[float],
    tier: RiskTier,
) -> Tuple[
    Optional[bool],
    Optional[bool],
    Optional[bool],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """Compare metrics to tier budgets and compute an alignment score.

    We treat drawdown as a *magnitude* in USD, regardless of sign. If the
    upstream code stores drawdown as negative numbers, abs() fixes that.
    If it stores positive magnitudes, abs() is a no-op.
    """
    if pnl_mean is None or dd_mean is None or kill_prob is None:
        return (None, None, None, None, None, None, None)

    # Always interpret drawdown as an absolute magnitude.
    dd_abs = abs(dd_mean)

    kill_ok = kill_prob <= tier.max_kill_prob
    dd_ok = dd_abs <= tier.max_drawdown_abs
    pnl_ok = pnl_mean >= tier.min_final_pnl_mean

    kill_margin = tier.max_kill_prob - kill_prob
    dd_margin = tier.max_drawdown_abs - dd_abs
    pnl_margin = pnl_mean - tier.min_final_pnl_mean

    row_like = pd.Series(
        {
            "final_pnl_mean": pnl_mean,
            "final_pnl_std": 0.0,  # unknown here; ignore in score
            "max_drawdown_mean": dd_mean,
            "kill_prob": kill_prob,
        }
    )
    score = compute_score(row_like, tier)

    return (
        kill_ok,
        dd_ok,
        pnl_ok,
        kill_margin,
        dd_margin,
        pnl_margin,
        float(score),
    )


def build_alignment_rows(
    presets_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
) -> List[AlignmentRow]:
    rows: List[AlignmentRow] = []

    # Build a small lookup for validation results (profile -> metrics)
    val_lookup: Dict[str, Dict[str, float]] = {}
    if val_df is not None:
        profile_col = _pick_col(val_df, PROFILE_COL_CANDIDATES, "profile")
        pnl_col = _pick_col(
            val_df, FINAL_PNL_MEAN_CANDIDATES, "final_pnl_mean (validation)"
        )
        dd_col = _pick_col(
            val_df, MAX_DD_MEAN_CANDIDATES, "max_drawdown_mean (validation)"
        )
        kill_col = _pick_col(
            val_df, KILL_PROB_CANDIDATES, "kill_prob (validation)"
        )
        for _, r in val_df.iterrows():
            name = str(r[profile_col])
            val_lookup[name] = {
                "final_pnl_mean": _safe_float(r[pnl_col]),
                "max_drawdown_mean": _safe_float(r[dd_col]),
                "kill_prob": _safe_float(r[kill_col]),
            }

    for tier in RISK_TIERS:
        # Pick best preset for this tier: rank_within_tier == 1 if present.
        tier_df = presets_df[presets_df["risk_tier"] == tier.name]
        if tier_df.empty:
            print(f"[exp10] WARNING: No presets found for tier '{tier.name}'.")
            preset_row = None
        else:
            if "rank_within_tier" in tier_df.columns:
                tier_df_sorted = tier_df.sort_values("rank_within_tier")
            else:
                tier_df_sorted = tier_df
            preset_row = tier_df_sorted.iloc[0]

        # Extract config + design metrics.
        if preset_row is not None:
            preset_name = str(preset_row.get("name", tier.name))
            rank = int(preset_row.get("rank_within_tier", 1))

            init_q = _safe_float(preset_row.get("initial_q_tao"))
            loss_limit = _safe_float(preset_row.get("loss_limit_usd"))
            hedge_band_base = _safe_float(preset_row.get("hedge_band_base"))
            size_eta = _safe_float(preset_row.get("size_eta"))
            vol_ref = _safe_float(preset_row.get("vol_ref"))
            delta_limit = _safe_float(
                preset_row.get("delta_limit_usd_base")
            )

            d_pnl = _safe_float(preset_row.get("final_pnl_mean"))
            d_dd = _safe_float(preset_row.get("max_drawdown_mean"))
            d_kill = _safe_float(preset_row.get("kill_prob"))
        else:
            preset_name = None
            rank = None
            init_q = loss_limit = hedge_band_base = size_eta = vol_ref = None
            delta_limit = None
            d_pnl = d_dd = d_kill = None

        (
            d_kill_ok,
            d_dd_ok,
            d_pnl_ok,
            d_kill_margin,
            d_dd_margin,
            d_pnl_margin,
            d_score,
        ) = _eval_against_budget(d_pnl, d_dd, d_kill, tier)

        # Validation metrics, if available.
        v_pnl = v_dd = v_kill = None
        v_kill_ok = v_dd_ok = v_pnl_ok = None
        v_kill_margin = v_dd_margin = v_pnl_margin = None
        v_score = None

        if tier.name in val_lookup:
            metrics = val_lookup[tier.name]
            v_pnl = metrics["final_pnl_mean"]
            v_dd = metrics["max_drawdown_mean"]
            v_kill = metrics["kill_prob"]

            (
                v_kill_ok,
                v_dd_ok,
                v_pnl_ok,
                v_kill_margin,
                v_dd_margin,
                v_pnl_margin,
                v_score,
            ) = _eval_against_budget(v_pnl, v_dd, v_kill, tier)
        elif val_df is not None:
            print(
                f"[exp10] WARNING: No validation row for profile '{tier.name}' "
                "in Exp08 summary."
            )

        rows.append(
            AlignmentRow(
                risk_tier=tier.name,
                preset_name=preset_name,
                rank_within_tier=rank,
                initial_q_tao=init_q,
                loss_limit_usd=loss_limit,
                hedge_band_base=hedge_band_base,
                size_eta=size_eta,
                vol_ref=vol_ref,
                delta_limit_usd_base=delta_limit,
                budget_max_kill_prob=tier.max_kill_prob,
                budget_max_dd_abs=tier.max_drawdown_abs,
                budget_min_final_pnl_mean=tier.min_final_pnl_mean,
                design_final_pnl_mean=d_pnl,
                design_max_drawdown_mean=d_dd,
                design_kill_prob=d_kill,
                design_kill_ok=d_kill_ok,
                design_dd_ok=d_dd_ok,
                design_pnl_ok=d_pnl_ok,
                design_kill_margin=d_kill_margin,
                design_dd_margin=d_dd_margin,
                design_pnl_margin=d_pnl_margin,
                design_score=d_score,
                val_final_pnl_mean=v_pnl,
                val_max_drawdown_mean=v_dd,
                val_kill_prob=v_kill,
                val_kill_ok=v_kill_ok,
                val_dd_ok=v_dd_ok,
                val_pnl_ok=v_pnl_ok,
                val_kill_margin=v_kill_margin,
                val_dd_margin=v_dd_margin,
                val_pnl_margin=v_pnl_margin,
                val_score=v_score,
            )
        )

    return rows


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exp10: World-model alignment report for Paraphina profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--presets-csv",
        type=Path,
        default=DEFAULT_PRESETS_CSV,
        help="Path to Exp07 presets CSV.",
    )
    p.add_argument(
        "--profile-summary-csv",
        type=Path,
        default=DEFAULT_PROFILE_SUMMARY_CSV,
        help="Path to Exp08 profile summary CSV (optional but recommended).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for Exp10 alignment artefacts.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    presets_df = load_presets(args.presets_csv)
    val_df = load_profile_summary(args.profile_summary_csv)

    rows = build_alignment_rows(presets_df, val_df)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame([r.__dict__ for r in rows])
    csv_path = out_dir / "exp10_alignment.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"[exp10] Wrote alignment CSV -> {csv_path}")

    # Pretty console summary
    print("\n=== Exp10 – World-model alignment summary ===\n")
    for r in rows:
        print(f"[{r.risk_tier}] preset={r.preset_name!r}, rank={r.rank_within_tier}")
        print(
            f"  budgets:  max_kill_prob <= {r.budget_max_kill_prob:.3f}, "
            f"max_dd_abs <= {r.budget_max_dd_abs:.0f}, "
            f"mean_pnl >= {r.budget_min_final_pnl_mean:.0f}"
        )

        if r.design_final_pnl_mean is not None:
            print(
                "  design:   pnl_mean={:.2f}, dd_mean={:.2f}, kill_prob={:.3f} "
                "(kill_ok={}, dd_ok={}, pnl_ok={})".format(
                    r.design_final_pnl_mean,
                    r.design_max_drawdown_mean or 0.0,
                    r.design_kill_prob or 0.0,
                    r.design_kill_ok,
                    r.design_dd_ok,
                    r.design_pnl_ok,
                )
            )
        else:
            print("  design:   (no metrics available)")

        if r.val_final_pnl_mean is not None:
            print(
                "  validate: pnl_mean={:.2f}, dd_mean={:.2f}, kill_prob={:.3f} "
                "(kill_ok={}, dd_ok={}, pnl_ok={})".format(
                    r.val_final_pnl_mean,
                    r.val_max_drawdown_mean or 0.0,
                    r.val_kill_prob or 0.0,
                    r.val_kill_ok,
                    r.val_dd_ok,
                    r.val_pnl_ok,
                )
            )
        elif val_df is not None:
            print("  validate: (no validation row found)")
        else:
            print("  validate: (validation CSV missing)")

        print("")

    print("[exp10] Done.")


if __name__ == "__main__":
    main()
