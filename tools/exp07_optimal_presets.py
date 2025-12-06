#!/usr/bin/env python3
"""
Exp07 – Optimal preset selector for Paraphina.

Robust version:

- Can read EITHER:
    * the Exp04 per-run CSV  (e.g. exp04_risk_regime_runs.csv), OR
    * the Exp04 summary CSV  (e.g. exp04_risk_regime_summary.csv).
- If given a per-run CSV, it aggregates by (initial_q_tao, loss_limit_usd):
      final_pnl_mean, final_pnl_std, max_drawdown_mean, kill_prob
- If given a summary CSV, it re-maps columns and uses it directly.
- Selects Conservative / Balanced / Aggressive presets.
- Writes:
      runs/exp07_optimal_presets/exp07_presets.csv
      runs/exp07_optimal_presets/exp07_presets.json
- Prints a Rust snippet you can paste into config.rs / main.rs.

If it cannot find required columns, it tells you exactly which names
it tried and what columns actually exist.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
OUT_DIR = RUNS_DIR / "exp07_optimal_presets"


# ---------------------------------------------------------------------
# Risk tiers
# ---------------------------------------------------------------------

@dataclass
class RiskTier:
    name: str
    max_kill_prob: float        # absolute probability (0.01 = 1%)
    max_drawdown_abs: float     # allowed drawdown magnitude (USD, positive)
    min_final_pnl_mean: float   # minimum acceptable mean PnL (USD)
    lambda_std: float           # penalty weight on volatility
    lambda_dd: float            # penalty weight on drawdown
    lambda_kill: float          # penalty weight on kill probability


RISK_TIERS: List[RiskTier] = [
    RiskTier(
        name="conservative",
        max_kill_prob=0.01,
        max_drawdown_abs=2_000.0,
        min_final_pnl_mean=0.0,
        lambda_std=0.5,
        lambda_dd=0.5,
        lambda_kill=200.0,
    ),
    RiskTier(
        name="balanced",
        max_kill_prob=0.05,
        max_drawdown_abs=4_000.0,
        min_final_pnl_mean=0.0,
        lambda_std=0.3,
        lambda_dd=0.3,
        lambda_kill=100.0,
    ),
    RiskTier(
        name="aggressive",
        max_kill_prob=0.15,
        max_drawdown_abs=8_000.0,
        min_final_pnl_mean=-500.0,
        lambda_std=0.1,
        lambda_dd=0.1,
        lambda_kill=50.0,
    ),
]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select optimal Paraphina presets from Exp04 results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--runs-csv",
        type=Path,
        default=None,
        help=(
            "Path to Exp04 CSV. Can be either per-run (e.g. *_runs.csv) "
            "or summary (e.g. *_summary.csv). If omitted, auto-detect under runs/."
        ),
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for presets (CSV + JSON).",
    )

    p.add_argument(
        "--top-per-tier",
        type=int,
        default=1,
        help="How many top configs to keep per risk tier.",
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------
# Column mapping helpers
# ---------------------------------------------------------------------

# Mandatory dimensions
INIT_Q_CANDIDATES = ["initial_q_tao", "initial_q", "q0", "q_tao_start"]
LOSS_LIMIT_CANDIDATES = ["loss_limit_usd", "loss_limit", "loss_limit_dollars"]

# Raw per-run metrics
FINAL_PNL_CANDIDATES = ["final_pnl", "pnl_final", "final_pnl_quote", "pnl"]
MAX_DD_CANDIDATES = ["max_drawdown", "max_drawdown_quote", "drawdown_max"]
KILLED_CANDIDATES = [
    "killed",
    "kill_flag",
    "hit_loss_limit",
    "stopped_out",
    "kill",
    "kill_switch",  # <- present in your Exp04 per-run CSV
]

# Aggregated (summary) metrics
FINAL_PNL_MEAN_CANDIDATES = ["final_pnl_mean", "pnl_mean", "mean_pnl"]
FINAL_PNL_STD_CANDIDATES = ["final_pnl_std", "pnl_std", "std_pnl"]
MAX_DD_MEAN_CANDIDATES = ["max_drawdown_mean", "dd_mean", "drawdown_mean"]
KILL_PROB_CANDIDATES = ["kill_prob", "kill_probability", "p_kill"]

# Optional extra knobs
EXTRA_COL_CANDIDATES: Dict[str, List[str]] = {
    "hedge_band_base": ["hedge_band_base", "hedge_band"],
    "size_eta": ["size_eta", "eta"],
    "vol_ref": ["vol_ref", "sigma_ref", "volatility_ref"],
    "delta_limit_usd_base": ["delta_limit_usd_base", "delta_limit_usd", "delta_limit"],
}


def pick_col(df: pd.DataFrame, candidates: List[str], what: str) -> str:
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


# ---------------------------------------------------------------------
# Input discovery / aggregation
# ---------------------------------------------------------------------

def auto_find_runs_csv(root: Path) -> Path:
    """
    Search for an Exp04 CSV under runs/.

    Preference:
      1) Per-run files like '*exp04*run*.csv' or '*runs.csv'
      2) Summary files like '*exp04*summary*.csv'
    """
    per_run: List[Path] = []
    summaries: List[Path] = []

    for path in root.rglob("*.csv"):
        s = str(path).lower()
        name = path.name.lower()

        if "exp04" in s and "run" in s:
            per_run.append(path)
        elif "risk_regime" in s and "run" in s:
            per_run.append(path)
        elif name.endswith("runs.csv"):
            per_run.append(path)
        elif "exp04" in s and "summary" in s:
            summaries.append(path)
        elif "risk_regime" in s and "summary" in s:
            summaries.append(path)

    if per_run:
        per_run.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        chosen = per_run[0]
        print(f"[auto-detect] Using per-run CSV: {chosen}")
        return chosen

    if summaries:
        summaries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        chosen = summaries[0]
        print(f"[auto-detect] Using summary CSV: {chosen}")
        return chosen

    raise FileNotFoundError(
        f"Could not auto-detect an Exp04 CSV under {root}. "
        f"Looked for '*exp04*run*.csv', '*runs.csv' or '*summary*.csv'."
    )


def _has_any(df: pd.DataFrame, candidates: List[str]) -> bool:
    cols_lower = {c.lower() for c in df.columns}
    return any(c.lower() in cols_lower for c in candidates)


def load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    """
    Load either a per-run CSV or an aggregated summary CSV.

    Returns a dataframe with canonical columns:
      initial_q_tao, loss_limit_usd,
      final_pnl_mean, final_pnl_std,
      max_drawdown_mean, kill_prob,
      [optional knobs...]
    """
    df = pd.read_csv(csv_path)
    print(f"[load] Loaded CSV with shape {df.shape} from {csv_path}")

    has_raw_pnl = _has_any(df, FINAL_PNL_CANDIDATES)
    has_agg_mean = _has_any(df, FINAL_PNL_MEAN_CANDIDATES)

    # Case 1: Aggregated summary (what exp04_risk_regime_summary.csv looks like)
    if has_agg_mean and not has_raw_pnl:
        print("[load] Detected summary-style CSV; using aggregated metrics as-is.")

        init_col = pick_col(df, INIT_Q_CANDIDATES, "initial_q_tao")
        loss_col = pick_col(df, LOSS_LIMIT_CANDIDATES, "loss_limit_usd")
        pnl_mean_col = pick_col(df, FINAL_PNL_MEAN_CANDIDATES, "final_pnl_mean")
        pnl_std_col = pick_col(df, FINAL_PNL_STD_CANDIDATES, "final_pnl_std")
        dd_mean_col = pick_col(df, MAX_DD_MEAN_CANDIDATES, "max_drawdown_mean")
        kill_prob_col = pick_col(df, KILL_PROB_CANDIDATES, "kill_prob")

        agg = df[[init_col, loss_col, pnl_mean_col, pnl_std_col, dd_mean_col, kill_prob_col]].copy()
        agg = agg.rename(
            columns={
                init_col: "initial_q_tao",
                loss_col: "loss_limit_usd",
                pnl_mean_col: "final_pnl_mean",
                pnl_std_col: "final_pnl_std",
                dd_mean_col: "max_drawdown_mean",
                kill_prob_col: "kill_prob",
            }
        )

        # Optional extra knobs – copy straight through
        for canonical, cand_list in EXTRA_COL_CANDIDATES.items():
            try:
                col = pick_col(df, cand_list, canonical)
            except KeyError:
                continue
            agg[canonical] = df[col].values

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        src_summary = OUT_DIR / "exp07_source_summary.csv"
        agg.to_csv(src_summary, index=False)
        print(f"[aggregate] Wrote aggregated summary -> {src_summary}")
        return agg

    # Case 2: Per-run CSV – aggregate by (initial_q_tao, loss_limit_usd)
    print("[load] Detected per-run CSV; aggregating across seeds.")

    init_col = pick_col(df, INIT_Q_CANDIDATES, "initial_q_tao")
    loss_col = pick_col(df, LOSS_LIMIT_CANDIDATES, "loss_limit_usd")
    pnl_col = pick_col(df, FINAL_PNL_CANDIDATES, "final_pnl")
    dd_col = pick_col(df, MAX_DD_CANDIDATES, "max_drawdown")
    kill_col = pick_col(df, KILLED_CANDIDATES, "kill flag")

    grouped = df.groupby([init_col, loss_col])

    agg = grouped[pnl_col].agg(["mean", "std"]).rename(
        columns={"mean": "final_pnl_mean", "std": "final_pnl_std"}
    )
    dd = grouped[dd_col].mean().rename("max_drawdown_mean")
    kill = grouped[kill_col].mean().rename("kill_prob")

    agg = agg.join(dd).join(kill).reset_index()
    agg = agg.rename(columns={init_col: "initial_q_tao", loss_col: "loss_limit_usd"})

    # Optional knobs: take first value per group
    for canonical, cand_list in EXTRA_COL_CANDIDATES.items():
        try:
            col = pick_col(df, cand_list, canonical)
        except KeyError:
            continue
        vals = grouped[col].first().rename(canonical)
        agg = agg.join(vals)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src_summary = OUT_DIR / "exp07_source_summary.csv"
    agg.to_csv(src_summary, index=False)
    print(f"[aggregate] Wrote aggregated summary -> {src_summary}")

    return agg


# ---------------------------------------------------------------------
# Scoring / selection
# ---------------------------------------------------------------------

def compute_score(row: pd.Series, tier: RiskTier) -> float:
    """Risk-adjusted objective: bigger is better."""
    pnl_mean = float(row["final_pnl_mean"])
    pnl_std = float(row["final_pnl_std"])
    dd_mean = float(row["max_drawdown_mean"])
    kill_prob = float(row["kill_prob"])

    dd_abs = abs(min(0.0, dd_mean))

    penalty_std = tier.lambda_std * pnl_std
    penalty_dd = tier.lambda_dd * dd_abs
    penalty_kill = tier.lambda_kill * kill_prob

    return pnl_mean - (penalty_std + penalty_dd + penalty_kill)


def select_for_tier(df: pd.DataFrame, tier: RiskTier, top_k: int) -> pd.DataFrame:
    df = df.copy()

    dd_abs = df["max_drawdown_mean"].clip(upper=0.0).abs()
    mask = (
        (df["kill_prob"] <= tier.max_kill_prob)
        & (dd_abs <= tier.max_drawdown_abs)
        & (df["final_pnl_mean"] >= tier.min_final_pnl_mean)
    )

    filtered = df[mask].copy()

    if filtered.empty:
        print(
            f"[{tier.name}] No configs passed strict constraints; "
            f"relaxing and ranking everything by score."
        )
        relaxed = df.copy()
        relaxed["__score"] = relaxed.apply(compute_score, axis=1, tier=tier)
        relaxed = relaxed.sort_values("__score", ascending=False)
        return relaxed.head(top_k).drop(columns=["__score"])

    filtered["__score"] = filtered.apply(compute_score, axis=1, tier=tier)
    filtered = filtered.sort_values("__score", ascending=False)
    return filtered.head(top_k).drop(columns=["__score"])


def build_presets(df: pd.DataFrame, top_per_tier: int) -> List[Dict[str, Any]]:
    presets: List[Dict[str, Any]] = []

    for tier in RISK_TIERS:
        tier_df = select_for_tier(df, tier, top_per_tier)

        if tier_df.empty:
            print(f"[{tier.name}] No configs selected even after relaxation.")
            continue

        for rank, (_, row) in enumerate(tier_df.iterrows(), start=1):
            preset_name = tier.name if top_per_tier == 1 else f"{tier.name}_{rank}"

            preset = {
                "name": preset_name,
                "risk_tier": tier.name,
                "rank_within_tier": rank,
                "initial_q_tao": float(row["initial_q_tao"]),
                "loss_limit_usd": float(row["loss_limit_usd"]),
                "hedge_band_base": float(row["hedge_band_base"]) if "hedge_band_base" in row else None,
                "size_eta": float(row["size_eta"]) if "size_eta" in row else None,
                "vol_ref": float(row["vol_ref"]) if "vol_ref" in row else None,
                "delta_limit_usd_base": float(row["delta_limit_usd_base"]) if "delta_limit_usd_base" in row else None,
                "final_pnl_mean": float(row["final_pnl_mean"]),
                "final_pnl_std": float(row["final_pnl_std"]),
                "max_drawdown_mean": float(row["max_drawdown_mean"]),
                "kill_prob": float(row["kill_prob"]),
            }
            presets.append(preset)

    return presets


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

def write_outputs(presets: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not presets:
        print("No presets selected; nothing to write.")
        return

    df = pd.DataFrame(presets)
    csv_path = out_dir / "exp07_presets.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote presets CSV -> {csv_path}")

    json_path = out_dir / "exp07_presets.json"
    with json_path.open("w") as f:
        json.dump(presets, f, indent=2)
    print(f"Wrote presets JSON -> {json_path}")

    print("\nSuggested Rust snippet for config.rs (fill TODOs as needed):\n")
    print("/*")
    print("pub enum Profile { Conservative, Balanced, Aggressive }\n")
    print("pub fn config_for_profile(profile: Profile) -> Config {")
    print("    let mut cfg = Config::default();")
    print("    match profile {")
    for p in presets:
        if p["rank_within_tier"] != 1:
            continue

        variant = {
            "conservative": "Profile::Conservative",
            "balanced": "Profile::Balanced",
            "aggressive": "Profile::Aggressive",
        }.get(p["risk_tier"], f'/* Profile::{p["risk_tier"]} */')

        print(f"        {variant} => {{")
        print(f"            cfg.initial_q_tao = {p['initial_q_tao']:.4};")
        print(f"            cfg.risk.daily_loss_limit = {-abs(p['loss_limit_usd']):.4};  // -loss_limit_usd")
        if p.get("hedge_band_base") is not None:
            print(f"            cfg.hedge.hedge_band_base = {p['hedge_band_base']:.4};")
        if p.get("size_eta") is not None:
            print(f"            cfg.mm.size_eta = {p['size_eta']:.6};")
        if p.get("vol_ref") is not None:
            print(f"            cfg.volatility.vol_ref = {p['vol_ref']:.6};")
        if p.get("delta_limit_usd_base") is not None:
            print(f"            cfg.risk.delta_hard_limit_usd_base = {p['delta_limit_usd_base']:.4};")
        print("            cfg")
        print("        },")
    print("    }")
    print("}")
    print("*/")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.runs_csv is not None:
        csv_path = args.runs_csv
        print(f"[explicit] Using CSV: {csv_path}")
    else:
        csv_path = auto_find_runs_csv(RUNS_DIR)

    summary_df = load_and_aggregate(csv_path)
    presets = build_presets(summary_df, top_per_tier=args.top_per_tier)
    write_outputs(presets, args.out_dir)


if __name__ == "__main__":
    main()
