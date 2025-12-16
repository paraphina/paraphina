#!/usr/bin/env python3
"""
exp03_stress_search.py

Stress / search vs starting inventory, refactored onto the generic
orchestrator + metrics backbone.

Pipeline
========
1. Load runs/exp02_profile_grid/exp02_profile_grid_summary.csv.
2. Use profile_centres.load_profile_centres_from_exp02(...) to pick per-profile
   "centre" hyperparameters (band_base, mm_size_eta, vol_ref, daily_loss_limit)
   for profiles:
       aggressive, balanced, conservative.
3. For each profile centre, run a stress grid over starting inventory q0 ∈
   INIT_Q_TAO_GRID, with NUM_REPEATS repeats:

      PARAPHINA_RISK_PROFILE   = profile
      PARAPHINA_INIT_Q_TAO     = q0
      PARAPHINA_HEDGE_BAND_BASE= centre.band_base
      PARAPHINA_MM_SIZE_ETA    = centre.mm_size_eta
      PARAPHINA_VOL_REF        = centre.vol_ref
      PARAPHINA_DAILY_LOSS_LIMIT = centre.daily_loss_limit

4. Parse stdout with metrics.parse_daily_summary (PnL + kill_switch).

5. Write:
      runs/exp03_stress_search/exp03_stress_runs.csv      (per-run)
      runs/exp03_stress_search/exp03_stress_summary.csv   (grouped)

Summary metrics per (profile, init_q_tao):
    pnl_realised_mean
    pnl_unrealised_mean
    pnl_total_mean
    pnl_total_std
    kill_switch_frac
    num_runs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from orchestrator import EngineRunConfig, run_many, results_to_dataframe
from metrics import parse_daily_summary, aggregate_basic
from profile_centres import (
    ProfileCentre,
    PROFILES,
    load_profile_centres_from_exp02,
)

import pandas as pd


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "release" / "paraphina"

EXP02_DIR = ROOT / "runs" / "exp02_profile_grid"
EXP02_SUMMARY_CSV = EXP02_DIR / "exp02_profile_grid_summary.csv"

EXP03_DIR = ROOT / "runs" / "exp03_stress_search"
EXP03_DIR.mkdir(parents=True, exist_ok=True)

# Starting inventory grid (TAO)
INIT_Q_TAO_GRID: List[float] = [-40.0, -20.0, 0.0, 20.0, 40.0]

# Number of independent repeats per (profile, init_q_tao)
NUM_REPEATS: int = 4


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_run_configs(
    centres: Dict[str, ProfileCentre],
) -> List[EngineRunConfig]:
    """
    Build the full grid of EngineRunConfig objects for exp03.

    We do NOT assume any specific CLI flags; we just call the binary directly:
        [BIN]

    If your engine supports e.g. a `--ticks 2000` argument and you want to
    enforce that, you can extend the cmd list here.
    """
    configs: List[EngineRunConfig] = []

    for profile in PROFILES:
        centre = centres[profile]

        for init_q in INIT_Q_TAO_GRID:
            for repeat_idx in range(NUM_REPEATS):
                env = {
                    "PARAPHINA_RISK_PROFILE": profile,
                    "PARAPHINA_INIT_Q_TAO": str(init_q),
                    "PARAPHINA_HEDGE_BAND_BASE": str(centre.band_base),
                    "PARAPHINA_MM_SIZE_ETA": str(centre.mm_size_eta),
                    "PARAPHINA_VOL_REF": str(centre.vol_ref),
                    "PARAPHINA_DAILY_LOSS_LIMIT": str(centre.daily_loss_limit),
                }

                label: Dict[str, Any] = {
                    "experiment": "exp03_stress_search",
                    "profile": profile,
                    "init_q_tao": float(init_q),
                    "band_base": centre.band_base,
                    "mm_size_eta": centre.mm_size_eta,
                    "vol_ref": centre.vol_ref,
                    "daily_loss_limit": centre.daily_loss_limit,
                    "repeat": int(repeat_idx),
                }

                cfg = EngineRunConfig(
                    cmd=[str(BIN)],
                    env=env,
                    label=label,
                    workdir=ROOT,
                )
                configs.append(cfg)

    return configs


def parse_metrics(stdout: str) -> Dict[str, Any]:
    """
    Wrapper around metrics.parse_daily_summary, wired to the orchestrator.
    """
    return parse_daily_summary(stdout)


def summarise_exp03(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Thin wrapper around metrics.aggregate_basic, with exp03's grouping keys.

    Group by (profile, init_q_tao) and compute:

        pnl_realised_mean
        pnl_unrealised_mean
        pnl_total_mean
        pnl_total_std
        kill_switch_frac
        num_runs
    """
    group_keys = ["profile", "init_q_tao"]
    return aggregate_basic(df_runs, group_keys=group_keys)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[exp03] Loading profile centres from exp02 summary...")
    centres = load_profile_centres_from_exp02(EXP02_SUMMARY_CSV)
    for prof, centre in centres.items():
        print(
            f"[exp03] Profile={prof} centre: "
            f"band_base={centre.band_base}, "
            f"mm_size_eta={centre.mm_size_eta}, "
            f"vol_ref={centre.vol_ref}, "
            f"daily_loss_limit={centre.daily_loss_limit}"
        )

    print("[exp03] Building run configs...")
    configs = build_run_configs(centres)
    print(f"[exp03] Total runs to execute: {len(configs)}")

    print("[exp03] Running simulations...")
    results = run_many(
        configs=configs,
        parse_metrics=parse_metrics,
        timeout_sec=None,
        verbose=True,
    )

    print("[exp03] Converting results to DataFrame...")
    df_runs = results_to_dataframe(results)

    runs_csv = EXP03_DIR / "exp03_stress_runs.csv"
    print(f"[exp03] Writing per-run metrics to {runs_csv}")
    df_runs.to_csv(runs_csv, index=False)

    print("[exp03] Computing summary metrics...")
    df_summary = summarise_exp03(df_runs)

    summary_csv = EXP03_DIR / "exp03_stress_summary.csv"
    print(f"[exp03] Writing summary metrics to {summary_csv}")
    df_summary.to_csv(summary_csv, index=False)

    print("[exp03] Done.")


if __name__ == "__main__":
    main()
