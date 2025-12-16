#!/usr/bin/env python3
"""
metrics.py

Shared metrics helpers for Paraphina experiments.

- parse_daily_summary(stdout) parses the existing human-readable summary that
  your Rust engine prints at the end of each run.

  It expects lines like:

      Daily PnL (realised / unrealised / total): +123.45 / -67.89 / +55.56
      Kill switch: true

  or:

      Kill switch active: false

- aggregate_basic(...) is a convenience wrapper to produce the same style of
  summary you used in exp02/exp03: mean/std of PnL metrics and kill_switch_frac.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# Parsing from stdout
# ---------------------------------------------------------------------------

# Regex for the "Daily PnL ..." line
_PNL_LINE_RE = re.compile(
    r"Daily PnL \(realised\s*/\s*unrealised\s*/\s*total\):\s*"
    r"(?P<realised>[+-]?\d+(?:\.\d+)?)\s*/\s*"
    r"(?P<unrealised>[+-]?\d+(?:\.\d+)?)\s*/\s*"
    r"(?P<total>[+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Regex for the kill switch line
_KILL_SWITCH_RE = re.compile(
    r"Kill\s*switch(?:\s*active)?\s*[:=]\s*(?P<val>true|false)",
    re.IGNORECASE,
)


def parse_daily_summary(stdout: str) -> Dict[str, Any]:
    """
    Parse the existing daily summary lines from the Paraphina engine stdout.

    Returns
    -------
    dict with keys:
      - pnl_realised (float or NaN)
      - pnl_unrealised (float or NaN)
      - pnl_total (float or NaN)
      - kill_switch (bool or False if not found)
    """
    realised = float("nan")
    unrealised = float("nan")
    total = float("nan")
    kill_switch = False

    # Find last matching PnL line
    pnl_matches = list(_PNL_LINE_RE.finditer(stdout))
    if pnl_matches:
        m = pnl_matches[-1]
        realised = float(m.group("realised"))
        unrealised = float(m.group("unrealised"))
        total = float(m.group("total"))

    # Find last kill switch indication
    ks_matches = list(_KILL_SWITCH_RE.finditer(stdout))
    if ks_matches:
        m = ks_matches[-1]
        val = m.group("val").lower()
        kill_switch = val == "true"

    return {
        "pnl_realised": realised,
        "pnl_unrealised": unrealised,
        "pnl_total": total,
        "kill_switch": kill_switch,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_basic(
    df_runs: pd.DataFrame,
    group_keys: List[str],
) -> pd.DataFrame:
    """
    Aggregate basic PnL + kill-switch metrics in the style of exp02/exp03.

    Input
    -----
    df_runs:
        Per-run DataFrame with columns at least:
            - pnl_realised
            - pnl_unrealised
            - pnl_total
            - kill_switch (bool)
            - plus any grouping keys (profile, init_q_tao, etc.)

    group_keys:
        Keys to group by (e.g. ["experiment", "profile", "init_q_tao"]).

    Output
    ------
    DataFrame with columns:
        - group_keys...
        - pnl_realised_mean
        - pnl_unrealised_mean
        - pnl_total_mean
        - pnl_total_std
        - kill_switch_frac
        - num_runs
    """
    if df_runs.empty:
        return df_runs

    grouped = df_runs.groupby(group_keys, dropna=False)

    out = grouped.agg(
        pnl_realised_mean=("pnl_realised", "mean"),
        pnl_unrealised_mean=("pnl_unrealised", "mean"),
        pnl_total_mean=("pnl_total", "mean"),
        pnl_total_std=("pnl_total", "std"),
    ).reset_index()

    # kill_switch_frac
    ks_frac = grouped["kill_switch"].mean().rename("kill_switch_frac").reset_index()
    out = out.merge(ks_frac, on=group_keys, how="left")

    # num_runs
    counts = grouped.size().rename("num_runs").reset_index()
    out = out.merge(counts, on=group_keys, how="left")

    return out
