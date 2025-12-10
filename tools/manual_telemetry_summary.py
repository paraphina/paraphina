#!/usr/bin/env python3
"""
manual_telemetry_summary.py

Quick one-off summary for a single telemetry JSONL file.

Default path:
    runs/manual_telemetry/test_run.jsonl

Usage:
    python tools/manual_telemetry_summary.py
    python tools/manual_telemetry_summary.py --path runs/.../some_telemetry.jsonl
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Make sure we can import ts_metrics from the batch_runs package
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]           # ~/code/paraphina
BATCH_RUNS = ROOT / "batch_runs"

if str(BATCH_RUNS) not in sys.path:
    sys.path.insert(0, str(BATCH_RUNS))

from ts_metrics import load_telemetry_jsonl  # now works like in batch_runs scripts


def summarise(path: Path) -> None:
    path = Path(path)

    if not path.exists():
        print(f"[summary] ERROR: telemetry file not found: {path}")
        raise SystemExit(1)

    df = load_telemetry_jsonl(path)

    if df.empty:
        print(f"[summary] No rows in telemetry ({path}) – did the engine actually run?")
        raise SystemExit(1)

    # --- PnL series ---------------------------------------------------------
    pnl = df["pnl_total"].astype(float)

    pnl_real = df.get("pnl_realised", pnl * 0).astype(float)
    pnl_unreal = df.get("pnl_unrealised", pnl * 0).astype(float)

    # Drop NaNs for drawdown calc
    pnl_vals = pnl.values.astype(float)
    mask = np.isfinite(pnl_vals)
    pnl_vals = pnl_vals[mask]

    if pnl_vals.size == 0:
        print("[summary] WARNING: pnl_total is all NaN; skipping drawdown.")
        max_dd = float("nan")
        max_pnl = float("nan")
    else:
        cummax = np.maximum.accumulate(pnl_vals)
        dd = cummax - pnl_vals
        max_dd = float(np.nanmax(dd))
        max_pnl = float(np.nanmax(pnl_vals))

    final_pnl = float(pnl.iloc[-1])

    # --- Risk regime fractions ----------------------------------------------
    if "risk_regime" in df.columns:
        regime_counts = df["risk_regime"].value_counts(normalize=True)
        frac_normal = float(regime_counts.get("Normal", 0.0))
        frac_warning = float(regime_counts.get("Warning", 0.0))
        frac_hard = float(regime_counts.get("HardLimit", 0.0))
    else:
        frac_normal = frac_warning = frac_hard = float("nan")

    # --- Kill switch --------------------------------------------------------
    kill_fired = bool(df.get("kill_switch", False).any())

    # --- Print summary ------------------------------------------------------
    print("=== Manual telemetry summary ===")
    print(f"File:              {path}")
    print(f"Rows (ticks):      {len(df)}")
    print()
    print(f"Final PnL (total): {final_pnl:,.2f} USD")
    print(f"Max PnL:           {max_pnl:,.2f} USD")
    print(f"Max drawdown:      {max_dd:,.2f} USD")
    print()
    print(
        "Risk regime: "
        f"Normal={frac_normal:.3f}, "
        f"Warning={frac_warning:.3f}, "
        f"HardLimit={frac_hard:.3f}"
    )
    print(f"Kill switch fired? {kill_fired}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise a single telemetry JSONL file."
    )
    parser.add_argument(
        "--path",
        "-p",
        type=Path,
        default=Path("runs/manual_telemetry/test_run.jsonl"),
        help="Path to telemetry JSONL "
             "(default: runs/manual_telemetry/test_run.jsonl)",
    )
    args = parser.parse_args()
    summarise(args.path)


if __name__ == "__main__":
    main()
