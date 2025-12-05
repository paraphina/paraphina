#!/usr/bin/env python3
"""
Simple inspector for paraphina_ticks.jsonl.

Reads the JSONL file produced by FileSink and prints:
- number of ticks
- first/last fair value
- last total PnL
- min/max/mean dollar delta
- min/max/mean basis exposure
"""

import json
from pathlib import Path
from statistics import mean

PATH = Path("paraphina_ticks.jsonl")


def main() -> None:
    if not PATH.exists():
        print(f"File not found: {PATH}")
        print("Run `cargo run` with use_file_sink = true in src/main.rs first.")
        return

    fair_values = []
    pnl_totals = []
    deltas = []
    basis = []

    with PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            fair_values.append(row["fair_value"])
            pnl_totals.append(row["daily_pnl_total"])
            deltas.append(row["dollar_delta_usd"])
            basis.append(row["basis_usd"])

    n = len(fair_values)
    if n == 0:
        print("No ticks found in log.")
        return

    print(f"Loaded {n} ticks from {PATH}")
    print("-" * 60)
    print(f"Fair value:  first={fair_values[0]:.4f}, last={fair_values[-1]:.4f}")
    print(f"Total PnL:   last tick={pnl_totals[-1]:.4f}")
    print()
    print(
        "Dollar delta USD: "
        f"min={min(deltas):.4f}, max={max(deltas):.4f}, mean={mean(deltas):.4f}"
    )
    print(
        "Basis exposure USD: "
        f"min={min(basis):.4f}, max={max(basis):.4f}, mean={mean(basis):.4f}"
    )


if __name__ == "__main__":
    main()
