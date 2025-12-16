#!/usr/bin/env python3
"""
ts_metrics.py

Time-series metrics for Paraphina JSONL telemetry.

Assumed JSONL schema (per line)
-------------------------------
At minimum, we expect the following keys:

- t             : integer tick index
- pnl_total     : float, cumulative total PnL at this tick
- risk_regime   : string, e.g. "Normal" | "Warning" | "HardLimit"
- kill_switch   : bool

Anything else is passed through and can be used later (inventory, basis, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np


def load_telemetry_jsonl(path: Path) -> pd.DataFrame:
    """
    Load a telemetry JSONL file into a pandas DataFrame.

    Parameters
    ----------
    path : Path
        Path to the JSONL file.

    Returns
    -------
    DataFrame
        Columns at least: t, pnl_total, risk_regime, kill_switch
        plus any other fields present in the JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"telemetry file not found: {path}")

    # pandas.read_json with lines=True can read JSONL
    df = pd.read_json(path, lines=True)

    # Ensure expected columns exist (fill with defaults if not)
    if "t" not in df.columns:
        df["t"] = np.arange(len(df))

    if "pnl_total" not in df.columns:
        raise ValueError("telemetry JSONL is missing 'pnl_total' column")

    if "risk_regime" not in df.columns:
        df["risk_regime"] = "Unknown"

    if "kill_switch" not in df.columns:
        df["kill_switch"] = False

    return df


def compute_ts_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute time-series risk metrics from a telemetry DataFrame.

    Metrics
    -------
    - pnl_total_end      : final total PnL
    - max_drawdown       : max peak-to-trough drop in pnl_total
    - max_drawdown_time  : index (t) where max_drawdown trough occurs
    - time_to_kill       : first t where kill_switch == True (None if never)
    - regime_frac_normal : fraction of ticks with risk_regime == "Normal"
    - regime_frac_warning: fraction of ticks with risk_regime == "Warning"
    - regime_frac_hard   : fraction of ticks with risk_regime == "HardLimit"
    - num_ticks          : total number of ticks
    """
    if df.empty:
        return {
            "pnl_total_end": float("nan"),
            "max_drawdown": float("nan"),
            "max_drawdown_time": None,
            "time_to_kill": None,
            "regime_frac_normal": float("nan"),
            "regime_frac_warning": float("nan"),
            "regime_frac_hard": float("nan"),
            "num_ticks": 0,
        }

    # Sort by t just in case
    df = df.sort_values("t").reset_index(drop=True)

    pnl = df["pnl_total"].astype(float).values
    num_ticks = len(pnl)

    # Final PnL
    pnl_total_end = float(pnl[-1])

    # Max drawdown: max over time of (peak - current)
    peaks = np.maximum.accumulate(pnl)
    drawdowns = peaks - pnl
    max_dd = float(drawdowns.max())
    max_dd_idx = int(drawdowns.argmax()) if num_ticks > 0 else 0
    max_dd_time = int(df.loc[max_dd_idx, "t"]) if num_ticks > 0 else None

    # Time to kill switch
    if "kill_switch" in df.columns:
        ks_series = df["kill_switch"].astype(bool)
        if ks_series.any():
            first_ks_idx = int(np.argmax(ks_series.values))
            time_to_kill = int(df.loc[first_ks_idx, "t"])
        else:
            time_to_kill = None
    else:
        time_to_kill = None

    # Regime occupancy fractions
    if "risk_regime" in df.columns:
        regimes = df["risk_regime"].astype(str)
        regime_frac_normal = float((regimes == "Normal").mean())
        regime_frac_warning = float((regimes == "Warning").mean())
        regime_frac_hard = float((regimes == "HardLimit").mean())
    else:
        regime_frac_normal = regime_frac_warning = regime_frac_hard = float("nan")

    return {
        "pnl_total_end": pnl_total_end,
        "max_drawdown": max_dd,
        "max_drawdown_time": max_dd_time,
        "time_to_kill": time_to_kill,
        "regime_frac_normal": regime_frac_normal,
        "regime_frac_warning": regime_frac_warning,
        "regime_frac_hard": regime_frac_hard,
        "num_ticks": num_ticks,
    }
