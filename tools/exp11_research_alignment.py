#!/usr/bin/env python3
"""
exp11_research_alignment.py

Exp11 – Compare empirical research runs vs world-model risk budgets.

This experiment takes a research JSONL dataset produced by tools/research_ticks.py
and checks, for each risk profile (conservative / balanced / aggressive), how
the empirical behaviour lines up with the world-model budgets defined in
exp07_optimal_presets.RISK_TIERS.

Inputs
======
- A research JSONL dataset, e.g.:
      research_dataset_v001.jsonl

  Each line is a JSON object with (at least):
      - "profile": "conservative" | "balanced" | "aggressive"
      - "final_pnl": float (USD)
      - "max_drawdown": float (USD, usually <= 0)
      - "p_kill_switch" or "kill_switch": boolean (did kill-switch fire?)

  This is exactly what tools/research_ticks.py writes.

- World model budgets (imported from exp07_optimal_presets):
      max_kill_prob
      max_drawdown_abs
      min_final_pnl_mean

Outputs
=======
- CSV:
      runs/exp11_research_alignment/exp11_alignment.csv

  One row per risk_tier with:
      - budgets (max_kill_prob, max_dd_abs, min_final_pnl_mean)
      - empirical metrics from the dataset (emp_* columns)
      - boolean flags: emp_kill_ok, emp_dd_ok, emp_pnl_ok
      - margins vs budget (positive = safe side, negative = violation)
      - risk-adjusted alignment score (emp_score)

- Console summary with a human-readable view.

Usage
=====
# Basic (use default dataset path in repo root)
python tools/exp11_research_alignment.py

# Explicit dataset + custom output directory
python tools/exp11_research_alignment.py \
    --dataset research_dataset_v001.jsonl \
    --out-dir runs/exp11_research_alignment
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from exp07_optimal_presets import (  # type: ignore
    RISK_TIERS,
    RiskTier,
    compute_score,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
DEFAULT_DATASET = ROOT / "research_dataset_v001.jsonl"
DEFAULT_OUT_DIR = RUNS_DIR / "exp11_research_alignment"


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

PROFILE_COL_CANDS = ["profile", "risk_profile", "tier"]
FINAL_PNL_COL_CANDS = [
    "final_pnl",
    "metrics.final_pnl",
    "pnl_total",
    "metrics.pnl_total",
]
MAX_DD_COL_CANDS = [
    "max_drawdown",
    "metrics.max_drawdown",
]
KILL_COL_CANDS = [
    "p_kill_switch",
    "kill_switch",
    "kill",
    "kill_prob",
    "metrics.kill_prob",
]


def _pick_col(df: pd.DataFrame, candidates: List[str], what: str) -> str:
    """Pick the first matching column (case-insensitive) or raise."""
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


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"[exp11] Dataset JSONL does not exist: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON on line {i}: {e}") from e
            records.append(rec)
    if not records:
        raise SystemExit(f"[exp11] Dataset {path} is empty.")
    return records


# ---------------------------------------------------------------------------
# Alignment logic
# ---------------------------------------------------------------------------


@dataclass
class ResearchAlignmentRow:
    risk_tier: str
    n_runs: int

    # Budgets
    budget_max_kill_prob: float
    budget_max_dd_abs: float
    budget_min_final_pnl_mean: float

    # Empirical metrics from dataset
    emp_final_pnl_mean: Optional[float]
    emp_max_drawdown_mean: Optional[float]
    emp_kill_prob: Optional[float]

    # Booleans vs budget
    emp_kill_ok: Optional[bool]
    emp_dd_ok: Optional[bool]
    emp_pnl_ok: Optional[bool]

    # Margins vs budget (positive = safe side)
    emp_kill_margin: Optional[float]
    emp_dd_margin: Optional[float]
    emp_pnl_margin: Optional[float]

    # Risk-adjusted alignment score
    emp_score: Optional[float]


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


def build_alignment_rows(df: pd.DataFrame) -> List[ResearchAlignmentRow]:
    rows: List[ResearchAlignmentRow] = []

    profile_col = _pick_col(df, PROFILE_COL_CANDS, "profile")
    pnl_col = _pick_col(df, FINAL_PNL_COL_CANDS, "final_pnl")
    dd_col = _pick_col(df, MAX_DD_COL_CANDS, "max_drawdown")
    kill_col = _pick_col(df, KILL_COL_CANDS, "kill_switch")

    for tier in RISK_TIERS:
        sub = df[df[profile_col] == tier.name]
        n_runs = int(len(sub))

        if n_runs == 0:
            print(
                f"[exp11] WARNING: No research runs found for profile '{tier.name}'."
            )
            pnl_mean = dd_mean = kill_prob = None
        else:
            pnl_mean = _safe_float(sub[pnl_col].mean())
            dd_mean = _safe_float(sub[dd_col].mean())

            # Bool or numeric; mean gives fraction of True
            kill_series = sub[kill_col].astype(float)
            kill_prob = _safe_float(kill_series.mean())

        (
            kill_ok,
            dd_ok,
            pnl_ok,
            kill_margin,
            dd_margin,
            pnl_margin,
            score,
        ) = _eval_against_budget(pnl_mean, dd_mean, kill_prob, tier)

        rows.append(
            ResearchAlignmentRow(
                risk_tier=tier.name,
                n_runs=n_runs,
                budget_max_kill_prob=tier.max_kill_prob,
                budget_max_dd_abs=tier.max_drawdown_abs,
                budget_min_final_pnl_mean=tier.min_final_pnl_mean,
                emp_final_pnl_mean=pnl_mean,
                emp_max_drawdown_mean=dd_mean,
                emp_kill_prob=kill_prob,
                emp_kill_ok=kill_ok,
                emp_dd_ok=dd_ok,
                emp_pnl_ok=pnl_ok,
                emp_kill_margin=kill_margin,
                emp_dd_margin=dd_margin,
                emp_pnl_margin=pnl_margin,
                emp_score=score,
            )
        )

    return rows


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Exp11: Compare empirical research runs vs world-model risk budgets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to research JSONL dataset (output of tools/research_ticks.py).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for Exp11 alignment artefacts.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    records = _load_jsonl(args.dataset)
    df = pd.DataFrame.from_records(records)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_alignment_rows(df)
    df_out = pd.DataFrame([r.__dict__ for r in rows])

    csv_path = out_dir / "exp11_alignment.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"[exp11] Wrote alignment CSV -> {csv_path}")

    # Console summary
    print("\n=== Exp11 – Research vs world-model alignment ===\n")
    for r in rows:
        print(
            f"[{r.risk_tier}] runs={r.n_runs}, "
            f"budgets: max_kill_prob<={r.budget_max_kill_prob:.3f}, "
            f"max_dd_abs<={r.budget_max_dd_abs:.0f}, "
            f"mean_pnl>={r.budget_min_final_pnl_mean:.0f}"
        )
        if r.emp_final_pnl_mean is None:
            print("  empirical: (no runs)")
        else:
            print(
                "  empirical: pnl_mean={:.2f}, dd_mean={:.2f}, kill_prob={:.3f} "
                "(kill_ok={}, dd_ok={}, pnl_ok={})".format(
                    r.emp_final_pnl_mean,
                    r.emp_max_drawdown_mean or 0.0,
                    r.emp_kill_prob or 0.0,
                    r.emp_kill_ok,
                    r.emp_dd_ok,
                    r.emp_pnl_ok,
                )
            )
        print("")

    print("[exp11] Done.")


if __name__ == "__main__":
    main()
