#!/usr/bin/env python3
"""
tools/research_ticks.py

Basic research/analysis helper for Paraphina tick logs.

Usage:
    python3 tools/research_ticks.py paraphina_ticks.jsonl
"""

import json
import sys
from pathlib import Path

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_ticks(path: Path):
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tools/research_ticks.py paraphina_ticks.jsonl")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    ticks = load_ticks(path)
    if not ticks:
        print("No ticks found in file.")
        sys.exit(0)

    print(f"Loaded {len(ticks)} ticks from {path}")
    print("-" * 72)

    # Simple manual stats (no dependencies)
    def col(name):
        return [t[name] for t in ticks if name in t]

    pnl = col("daily_pnl_total")
    delta = col("dollar_delta_usd")
    basis = col("basis_usd")
    fv = col("fair_value")

    def basic_stats(xs, label):
        if not xs:
            print(f"{label}: <no data>")
            return
        mn = min(xs)
        mx = max(xs)
        last = xs[-1]
        avg = sum(xs) / len(xs)
        print(f"{label}: min={mn:.4f}, max={mx:.4f}, last={last:.4f}, mean={avg:.4f}")

    basic_stats(fv,    "Fair value")
    basic_stats(pnl,   "Daily PnL total")
    basic_stats(delta, "Dollar delta USD")
    basic_stats(basis, "Basis exposure USD")

    # Optional pandas + chart
    if HAS_PANDAS:
        df = pd.DataFrame(ticks)
        df["tick"] = range(len(df))

        print("\nHead of dataframe:")
        print(df.head())

        # Simple PnL plot
        plt.figure()
        plt.plot(df["tick"], df["daily_pnl_total"])
        plt.xlabel("Tick")
        plt.ylabel("Daily PnL total (USD)")
        plt.title("Paraphina PnL path")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(
            "\n(pandas/matplotlib not installed; "
            "install with `python3 -m pip install --user pandas matplotlib` "
            "to get plots and richer analysis.)"
        )


if __name__ == "__main__":
    main()
