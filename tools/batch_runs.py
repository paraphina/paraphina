#!/usr/bin/env python3
"""
research_ticks.py

Log a single telemetry JSONL file as a run-level record into a research
dataset (JSONL).

Usage examples
--------------

# Log the manual test run you just did, tagged as aggressive:
python tools/research_ticks.py \
    --telemetry runs/manual_telemetry/test_run.jsonl \
    --profile aggressive \
    --label manual_test_001 \
    --out research_run_001.jsonl

The script will:
  - load the ticks via batch_runs.ts_metrics.load_telemetry_jsonl
  - compute PnL / drawdown / regime fractions / kill-switch
  - look up the profile centre from batch_runs.profile_centres
  - append a JSON line to the output file
  - print a human-readable summary
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict  # kept for possible future use
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Make repo root importable so we can do `from batch_runs ...`
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batch_runs.ts_metrics import load_telemetry_jsonl
from batch_runs.profile_centres import get_profile_centre


def compute_run_metrics(df: pd.DataFrame) -> dict:
    """Compute basic run-level metrics from a telemetry dataframe."""
    if df.empty:
        raise SystemExit("No rows in telemetry – did the engine actually run?")

    # --- PnL series ---------------------------------------------------------
    pnl = df["pnl_total"].astype(float)

    final_pnl = float(pnl.iloc[-1])
    max_pnl = float(pnl.max())
    min_pnl = float(pnl.min())

    # --- Max drawdown on total PnL -----------------------------------------
    cummx = np.maximum.accumulate(pnl.values)
    dd = cummx - pnl.values
    max_dd = float(dd.max()) if len(dd) else 0.0

    # --- Risk regime fractions ---------------------------------------------
    regime_counts = df["risk_regime"].value_counts(normalize=True)
    frac_normal = float(regime_counts.get("Normal", 0.0))
    frac_warning = float(regime_counts.get("Warning", 0.0))
    frac_hard = float(regime_counts.get("HardLimit", 0.0))

    # --- Kill-switch -------------------------------------------------------
    if "kill_switch" in df.columns:
        kill_fired = bool(df["kill_switch"].any())
    else:
        kill_fired = False

    return {
        "num_ticks": int(len(df)),
        "final_pnl": final_pnl,
        "max_pnl": max_pnl,
        "min_pnl": min_pnl,
        "max_drawdown": max_dd,
        "frac_normal": frac_normal,
        "frac_warning": frac_warning,
        "frac_hard": frac_hard,
        "kill_switch": kill_fired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise a telemetry JSONL file and append it to a "
            "research JSONL dataset."
        ),
    )
    parser.add_argument(
        "--telemetry",
        "-t",
        type=Path,
        required=True,
        help="Path to telemetry JSONL file (e.g. runs/manual_telemetry/test_run.jsonl)",
    )
    parser.add_argument(
        "--profile",
        "-p",
        type=str,
        choices=["aggressive", "balanced", "conservative"],
        required=True,
        help="Risk profile used for the run.",
    )
    parser.add_argument(
        "--label",
        "-l",
        type=str,
        required=True,
        help="Short label for this run (e.g. 'manual_test_001').",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("research_run_001.jsonl"),
        help=(
            "Output JSONL dataset to append to "
            "(default: research_run_001.jsonl)."
        ),
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional free-form notes to store with the run.",
    )

    args = parser.parse_args()

    # Load telemetry
    df = load_telemetry_jsonl(args.telemetry)

    # Compute core metrics
    metrics = compute_run_metrics(df)

    # Look up profile centre knobs
    centre = get_profile_centre(args.profile)

    # Build record
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "profile": args.profile,
        "telemetry_path": str(args.telemetry),
        "notes": args.notes,
        # Profile-centre structural knobs
        "band_base": centre.band_base,
        "mm_size_eta": centre.mm_size_eta,
        "vol_ref": centre.vol_ref,
        "daily_loss_limit": centre.daily_loss_limit,
        "init_q_tao": centre.init_q_tao,
        "vol_scale": centre.vol_scale,
        # Run-level metrics
        **metrics,
    }

    # Ensure output directory exists
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Append JSON line
    with args.out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Pretty-print summary
    print("=== Research run summary ===")
    print(f"Label:         {args.label}")
    print(f"Profile:       {args.profile}")
    print(f"Telemetry:     {args.telemetry}")
    print(f"Saved to:      {args.out}")
    print()
    print(f"Ticks:         {metrics['num_ticks']}")
    print(f"Final PnL:     {metrics['final_pnl']:.2f} USD")
    print(f"Max PnL:       {metrics['max_pnl']:.2f} USD")
    print(f"Max drawdown:  {metrics['max_drawdown']:.2f} USD")
    print()
    print(
        "Risk regime:   "
        f"Normal={metrics['frac_normal']:.3f}, "
        f"Warning={metrics['frac_warning']:.3f}, "
        f"HardLimit={metrics['frac_hard']:.3f}",
    )
    print(f"Kill switch:   {metrics['kill_switch']}")


if __name__ == "__main__":
    main()
