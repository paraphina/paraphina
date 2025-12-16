#!/usr/bin/env python3
"""
exp05_telemetry_validation.py

Prototype harness to validate JSONL telemetry + time-series metrics.

This reuses the same (profile, init_q_tao) grid as exp03, but:
  - sets telemetry env vars so each run writes a JSONL file
  - loads that JSONL and computes path-level metrics:

      max_drawdown, time_to_kill, regime occupancy, ...

NOTE: For this to work, the Rust engine must:
  - respect PARAPHINA_TELEMETRY_MODE = "jsonl"
  - write JSONL to PARAPHINA_TELEMETRY_PATH
  - include at least: t, pnl_total, risk_regime, kill_switch
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from orchestrator import EngineRunConfig, run_many, results_to_dataframe
from metrics import parse_daily_summary
from profile_centres import (
    ProfileCentre,
    PROFILES,
    load_profile_centres_from_exp02,
)
from ts_metrics import load_telemetry_jsonl, compute_ts_metrics


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "release" / "paraphina"

EXP02_DIR = ROOT / "runs" / "exp02_profile_grid"
EXP02_SUMMARY_CSV = EXP02_DIR / "exp02_profile_grid_summary.csv"

EXP05_DIR = ROOT / "runs" / "exp05_telemetry_validation"
EXP05_DIR.mkdir(parents=True, exist_ok=True)

TELEMETRY_DIR = EXP05_DIR / "telemetry"
TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)

INIT_Q_TAO_GRID: List[float] = [-40.0, -20.0, 0.0, 20.0, 40.0]
NUM_REPEATS: int = 2  # keep small initially for quick sanity checks


def build_run_configs(
    centres: Dict[str, ProfileCentre],
) -> List[EngineRunConfig]:
    configs: List[EngineRunConfig] = []

    for profile in PROFILES:
        centre = centres[profile]

        for init_q in INIT_Q_TAO_GRID:
            for repeat_idx in range(NUM_REPEATS):
                run_id = f"exp05_{profile}_q{int(init_q)}_r{repeat_idx}"

                telem_path = TELEMETRY_DIR / f"{run_id}.jsonl"

                env = {
                    "PARAPHINA_RISK_PROFILE": profile,
                    "PARAPHINA_INIT_Q_TAO": str(init_q),
                    "PARAPHINA_HEDGE_BAND_BASE": str(centre.band_base),
                    "PARAPHINA_MM_SIZE_ETA": str(centre.mm_size_eta),
                    "PARAPHINA_VOL_REF": str(centre.vol_ref),
                    "PARAPHINA_DAILY_LOSS_LIMIT": str(centre.daily_loss_limit),
                    # Telemetry
                    "PARAPHINA_TELEMETRY_MODE": "jsonl",
                    "PARAPHINA_TELEMETRY_PATH": str(telem_path),
                }

                label: Dict[str, Any] = {
                    "experiment": "exp05_telemetry_validation",
                    "profile": profile,
                    "init_q_tao": float(init_q),
                    "band_base": centre.band_base,
                    "mm_size_eta": centre.mm_size_eta,
                    "vol_ref": centre.vol_ref,
                    "daily_loss_limit": centre.daily_loss_limit,
                    "repeat": int(repeat_idx),
                    "telemetry_path": str(telem_path),
                }

                cfg = EngineRunConfig(
                    cmd=[str(BIN)],
                    env=env,
                    label=label,
                    run_id=run_id,
                    workdir=ROOT,
                )
                configs.append(cfg)

    return configs


def parse_metrics(stdout: str) -> Dict[str, Any]:
    # Daily summary metrics from stdout
    return parse_daily_summary(stdout)


def attach_ts_metrics(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    For each run, load its telemetry JSONL and compute time-series metrics.

    Expects df_runs to have a 'telemetry_path' column (absolute or relative).
    """
    ts_rows: List[Dict[str, Any]] = []

    for _, row in df_runs.iterrows():
        telem_path = Path(row["telemetry_path"])
        try:
            df_ts = load_telemetry_jsonl(telem_path)
            ts_metrics = compute_ts_metrics(df_ts)
        except Exception as e:  # noqa: BLE001
            ts_metrics = {"ts_error": str(e)}

        # include run_id so we can merge back
        ts_metrics["run_id"] = row["run_id"]
        ts_rows.append(ts_metrics)

    if not ts_rows:
        return df_runs

    df_ts_metrics = pd.DataFrame(ts_rows)

    # Merge on run_id
    df_merged = df_runs.merge(df_ts_metrics, on="run_id", how="left")
    return df_merged


def summarise_exp05(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (profile, init_q_tao) and compute summary of both
    daily-summary metrics and time-series metrics.
    """
    if df_runs.empty:
        return df_runs

    group_keys = ["profile", "init_q_tao"]
    grouped = df_runs.groupby(group_keys, dropna=False)

    out = grouped.agg(
        pnl_total_mean=("pnl_total", "mean"),
        pnl_total_std=("pnl_total", "std"),
        max_drawdown_mean=("max_drawdown", "mean"),
        max_drawdown_std=("max_drawdown", "std"),
        time_to_kill_mean=("time_to_kill", "mean"),
        regime_frac_normal_mean=("regime_frac_normal", "mean"),
        regime_frac_warning_mean=("regime_frac_warning", "mean"),
        regime_frac_hard_mean=("regime_frac_hard", "mean"),
    ).reset_index()

    # kill_switch_frac from daily summary
    ks_frac = grouped["kill_switch"].mean().rename("kill_switch_frac").reset_index()
    out = out.merge(ks_frac, on=group_keys, how="left")

    # num_runs
    counts = grouped.size().rename("num_runs").reset_index()
    out = out.merge(counts, on=group_keys, how="left")

    return out


def main() -> None:
    print("[exp05] Loading profile centres from exp02 summary...")
    centres = load_profile_centres_from_exp02(EXP02_SUMMARY_CSV)
    for prof, centre in centres.items():
        print(
            f"[exp05] Profile={prof} centre: "
            f"band_base={centre.band_base}, "
            f"mm_size_eta={centre.mm_size_eta}, "
            f"vol_ref={centre.vol_ref}, "
            f"daily_loss_limit={centre.daily_loss_limit}"
        )

    print("[exp05] Building run configs...")
    configs = build_run_configs(centres)
    print(f"[exp05] Total runs to execute: {len(configs)}")

    print("[exp05] Running simulations with telemetry enabled...")
    results = run_many(
        configs=configs,
        parse_metrics=parse_metrics,
        timeout_sec=None,
        verbose=True,
    )

    print("[exp05] Converting results to DataFrame...")
    df_runs = results_to_dataframe(results)

    runs_csv = EXP05_DIR / "exp05_runs_raw.csv"
    print(f"[exp05] Writing raw per-run metrics to {runs_csv}")
    df_runs.to_csv(runs_csv, index=False)

    print("[exp05] Attaching time-series metrics from telemetry...")
    df_runs_ts = attach_ts_metrics(df_runs)

    runs_ts_csv = EXP05_DIR / "exp05_runs_with_ts.csv"
    print(f"[exp05] Writing per-run metrics + ts metrics to {runs_ts_csv}")
    df_runs_ts.to_csv(runs_ts_csv, index=False)

    print("[exp05] Computing grouped summary metrics...")
    df_summary = summarise_exp05(df_runs_ts)

    summary_csv = EXP05_DIR / "exp05_summary.csv"
    print(f"[exp05] Writing summary metrics to {summary_csv}")
    df_summary.to_csv(summary_csv, index=False)

    print("[exp05] Done.")


if __name__ == "__main__":
    main()
