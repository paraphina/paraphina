#!/usr/bin/env python3
"""
exp04_multi_scenario_stress.py

Multi-scenario stress harness for Paraphina, with time-series telemetry.

Design
======
- Reuses profile centres from exp02 (via profile_centres.py).
- Stresses effective volatility regime via vol_ref scaling:

    scenario "baseline": vol_scale = 1.0
    scenario "low_vol":  vol_scale = 0.75
    scenario "high_vol": vol_scale = 1.5

- For each:
      profile ∈ {aggressive, balanced, conservative}
      scenario ∈ {baseline, low_vol, high_vol}
      init_q_tao ∈ {-40, -20, 0, 20, 40}
      repeat ∈ {0..NUM_REPEATS-1}

  run the Rust engine with env:

      PARAPHINA_RISK_PROFILE
      PARAPHINA_INIT_Q_TAO
      PARAPHINA_HEDGE_BAND_BASE
      PARAPHINA_MM_SIZE_ETA
      PARAPHINA_VOL_REF           (centre.vol_ref * vol_scale)
      PARAPHINA_DAILY_LOSS_LIMIT
      PARAPHINA_TELEMETRY_MODE    = "jsonl"
      PARAPHINA_TELEMETRY_PATH    = runs/exp04_multi_scenario/telemetry/<run_id>.jsonl

- Parse stdout using metrics.parse_daily_summary (PnL + kill switch).
- Load JSONL telemetry and compute time-series metrics via ts_metrics:

      max_drawdown, time_to_kill, regime occupancy, ...

- Outputs:

    runs/exp04_multi_scenario/exp04_runs.csv          (per-run daily metrics)
    runs/exp04_multi_scenario/exp04_runs_with_ts.csv  (per-run daily + ts metrics)
    runs/exp04_multi_scenario/exp04_summary.csv       (grouped by profile×scenario×init_q_tao)

Summary metrics per (profile, scenario, init_q_tao):
    pnl_realised_mean
    pnl_unrealised_mean
    pnl_total_mean
    pnl_total_std
    pnl_total_p5 (5th percentile)
    pnl_total_p95 (95th percentile)
    max_drawdown_mean
    max_drawdown_std
    time_to_kill_mean
    regime_frac_normal_mean
    regime_frac_warning_mean
    regime_frac_hard_mean
    kill_switch_frac
    num_runs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from orchestrator import EngineRunConfig, run_many, results_to_dataframe
from metrics import parse_daily_summary
from profile_centres import (
    ProfileCentre,
    PROFILES,
    load_profile_centres_from_exp02,
)
from ts_metrics import load_telemetry_jsonl, compute_ts_metrics


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "release" / "paraphina"

EXP02_DIR = ROOT / "runs" / "exp02_profile_grid"
EXP02_SUMMARY_CSV = EXP02_DIR / "exp02_profile_grid_summary.csv"

EXP04_DIR = ROOT / "runs" / "exp04_multi_scenario"
EXP04_DIR.mkdir(parents=True, exist_ok=True)

TELEMETRY_DIR = EXP04_DIR / "telemetry"
TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)

# Starting inventory grid (TAO) – same as exp03/exp05 for comparability
INIT_Q_TAO_GRID: List[float] = [-40.0, -20.0, 0.0, 20.0, 40.0]

# Number of independent repeats per (profile, scenario, init_q_tao)
NUM_REPEATS: int = 4

# "Scenario bank" implemented via vol_ref scaling.
SCENARIOS: List[Dict[str, Any]] = [
    {"name": "baseline", "vol_scale": 1.0},
    {"name": "low_vol",  "vol_scale": 0.75},
    {"name": "high_vol", "vol_scale": 1.5},
]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_run_configs(
    centres: Dict[str, ProfileCentre],
) -> List[EngineRunConfig]:
    """
    Build the full grid of EngineRunConfig objects for exp04.

    For each run we also:
      - set telemetry env vars,
      - assign a unique run_id and telemetry_path (for later ts metrics).
    """
    configs: List[EngineRunConfig] = []

    for profile in PROFILES:
        centre = centres[profile]

        for scenario in SCENARIOS:
            scenario_name = scenario["name"]
            vol_scale = float(scenario["vol_scale"])

            vol_ref_override = centre.vol_ref * vol_scale

            for init_q in INIT_Q_TAO_GRID:
                for repeat_idx in range(NUM_REPEATS):
                    run_id = f"exp04_{profile}_{scenario_name}_q{int(init_q)}_r{repeat_idx}"
                    telem_path = TELEMETRY_DIR / f"{run_id}.jsonl"

                    env = {
                        "PARAPHINA_RISK_PROFILE": profile,
                        "PARAPHINA_INIT_Q_TAO": str(init_q),
                        "PARAPHINA_HEDGE_BAND_BASE": str(centre.band_base),
                        "PARAPHINA_MM_SIZE_ETA": str(centre.mm_size_eta),
                        "PARAPHINA_VOL_REF": str(vol_ref_override),
                        "PARAPHINA_DAILY_LOSS_LIMIT": str(centre.daily_loss_limit),
                        # Telemetry
                        "PARAPHINA_TELEMETRY_MODE": "jsonl",
                        "PARAPHINA_TELEMETRY_PATH": str(telem_path),
                    }

                    label: Dict[str, Any] = {
                        "experiment": "exp04_multi_scenario",
                        "profile": profile,
                        "scenario": scenario_name,
                        "vol_scale": vol_scale,
                        "band_base": centre.band_base,
                        "mm_size_eta": centre.mm_size_eta,
                        "vol_ref_centre": centre.vol_ref,
                        "vol_ref_override": vol_ref_override,
                        "daily_loss_limit": centre.daily_loss_limit,
                        "init_q_tao": float(init_q),
                        "repeat": int(repeat_idx),
                        "telemetry_path": str(telem_path),
                    }

                    cfg = EngineRunConfig(
                        cmd=[str(BIN)],
                        env=env,
                        label=label,
                        workdir=ROOT,
                        run_id=run_id,
                    )
                    configs.append(cfg)

    return configs


def parse_metrics(stdout: str) -> Dict[str, Any]:
    """
    Wrapper around metrics.parse_daily_summary, for use with the orchestrator.
    """
    return parse_daily_summary(stdout)


# ---------------------------------------------------------------------------
# Telemetry → time-series metrics
# ---------------------------------------------------------------------------

def attach_ts_metrics(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    For each run, load its telemetry JSONL and compute time-series metrics.

    Expects df_runs to have:
      - 'run_id' column (from EngineRunConfig.run_id)
      - 'telemetry_path' column (absolute or relative path)

    Returns df_runs merged with:

      - pnl_total_end
      - max_drawdown
      - max_drawdown_time
      - time_to_kill
      - regime_frac_normal
      - regime_frac_warning
      - regime_frac_hard
      - num_ticks

    plus 'ts_error' if any telemetry file fails to load/parse.
    """
    if "telemetry_path" not in df_runs.columns or "run_id" not in df_runs.columns:
        # Nothing to attach; return df unchanged.
        return df_runs

    ts_rows: List[Dict[str, Any]] = []

    for _, row in df_runs.iterrows():
        telem_path = Path(row["telemetry_path"])
        try:
            df_ts = load_telemetry_jsonl(telem_path)
            ts_metrics = compute_ts_metrics(df_ts)
        except Exception as e:  # noqa: BLE001
            ts_metrics = {"ts_error": str(e)}

        ts_metrics["run_id"] = row["run_id"]
        ts_rows.append(ts_metrics)

    if not ts_rows:
        return df_runs

    df_ts_metrics = pd.DataFrame(ts_rows)

    # Merge on run_id
    df_merged = df_runs.merge(df_ts_metrics, on="run_id", how="left")
    return df_merged


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def summarise_exp04(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute extended summary stats for exp04:

        - grouped by (profile, scenario, init_q_tao)

    Metrics:
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
    """
    if df_runs.empty:
        return df_runs

    group_keys = ["profile", "scenario", "init_q_tao"]
    grouped = df_runs.groupby(group_keys, dropna=False)

    # Base PnL stats
    out = grouped.agg(
        pnl_realised_mean=("pnl_realised", "mean"),
        pnl_unrealised_mean=("pnl_unrealised", "mean"),
        pnl_total_mean=("pnl_total", "mean"),
        pnl_total_std=("pnl_total", "std"),
    ).reset_index()

    # Distribution tails for pnl_total
    def _p5(x: pd.Series) -> float:
        x = x.dropna().values
        return float(np.percentile(x, 5)) if len(x) > 0 else float("nan")

    def _p95(x: pd.Series) -> float:
        x = x.dropna().values
        return float(np.percentile(x, 95)) if len(x) > 0 else float("nan")

    tails = grouped["pnl_total"].agg(
        pnl_total_p5=_p5,
        pnl_total_p95=_p95,
    ).reset_index()
    out = out.merge(tails, on=group_keys, how="left")

    # Time-series metrics (drawdown, regimes, etc.) – only if present
    if "max_drawdown" in df_runs.columns:
        ts_agg = grouped.agg(
            max_drawdown_mean=("max_drawdown", "mean"),
            max_drawdown_std=("max_drawdown", "std"),
            time_to_kill_mean=("time_to_kill", "mean"),
            regime_frac_normal_mean=("regime_frac_normal", "mean"),
            regime_frac_warning_mean=("regime_frac_warning", "mean"),
            regime_frac_hard_mean=("regime_frac_hard", "mean"),
        ).reset_index()
        out = out.merge(ts_agg, on=group_keys, how="left")

    # kill_switch_frac from daily summary
    ks_frac = grouped["kill_switch"].mean().rename("kill_switch_frac").reset_index()
    out = out.merge(ks_frac, on=group_keys, how="left")

    # num_runs
    counts = grouped.size().rename("num_runs").reset_index()
    out = out.merge(counts, on=group_keys, how="left")

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[exp04] Loading profile centres from exp02 summary...")
    centres = load_profile_centres_from_exp02(EXP02_SUMMARY_CSV)
    for prof, centre in centres.items():
        print(
            f"[exp04] Profile={prof} centre: "
            f"band_base={centre.band_base}, "
            f"mm_size_eta={centre.mm_size_eta}, "
            f"vol_ref={centre.vol_ref}, "
            f"daily_loss_limit={centre.daily_loss_limit}"
        )

    print("[exp04] Building run configs...")
    configs = build_run_configs(centres)
    print(f"[exp04] Total runs to execute: {len(configs)}")

    print("[exp04] Running simulations with telemetry enabled...")
    results = run_many(
        configs=configs,
        parse_metrics=parse_metrics,
        timeout_sec=None,
        verbose=True,
    )

    print("[exp04] Converting results to DataFrame...")
    df_runs = results_to_dataframe(results)

    runs_csv = EXP04_DIR / "exp04_runs.csv"
    print(f"[exp04] Writing per-run daily metrics to {runs_csv}")
    df_runs.to_csv(runs_csv, index=False)

    print("[exp04] Attaching time-series metrics from telemetry...")
    df_runs_ts = attach_ts_metrics(df_runs)

    runs_ts_csv = EXP04_DIR / "exp04_runs_with_ts.csv"
    print(f"[exp04] Writing per-run daily + ts metrics to {runs_ts_csv}")
    df_runs_ts.to_csv(runs_ts_csv, index=False)

    print("[exp04] Computing summary metrics...")
    df_summary = summarise_exp04(df_runs_ts)

    summary_csv = EXP04_DIR / "exp04_summary.csv"
    print(f"[exp04] Writing summary metrics to {summary_csv}")
    df_summary.to_csv(summary_csv, index=False)

    print("[exp04] Done.")


if __name__ == "__main__":
    main()
