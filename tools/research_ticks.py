#!/usr/bin/env python3
"""
tools/research_ticks.py

Lightweight research helper for Paraphina JSONL tick logs.

Usage:
    python3 tools/research_ticks.py paraphina_ticks.jsonl
    python3 tools/research_ticks.py research_run_002.jsonl

It will:
  - load the JSONL into a pandas DataFrame,
  - print basic summary statistics, and
  - save a few PNG charts next to the JSONL file:
        <stem>_pnl_curve.png
        <stem>_delta_curve.png
        <stem>_basis_curve.png
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_ticks(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    if "tick" not in df.columns:
        raise ValueError("Expected a 'tick' column in the JSONL log")
    return df


def print_summary(df: pd.DataFrame, path: Path) -> None:
    print(f"Loaded {len(df)} ticks from {path.name}")
    print("-" * 72)

    # Fair value
    fv = df["fair_value"]
    print(
        "Fair value: "
        f"min={fv.min():.4f}, max={fv.max():.4f}, "
        f"last={fv.iloc[-1]:.4f}, mean={fv.mean():.4f}"
    )

    # Total PnL
    pnl = df["daily_pnl_total"]
    print(
        "Daily PnL total: "
        f"min={pnl.min():.4f}, max={pnl.max():.4f}, "
        f"last={pnl.iloc[-1]:.4f}, mean={pnl.mean():.4f}"
    )

    # Dollar delta
    delta = df["dollar_delta_usd"]
    print(
        "Dollar delta USD: "
        f"min={delta.min():.4f}, max={delta.max():.4f}, "
        f"last={delta.iloc[-1]:.4f}, mean={delta.mean():.4f}"
    )

    # Basis exposure
    basis = df["basis_usd"]
    print(
        "Basis exposure USD: "
        f"min={basis.min():.4f}, max={basis.max():.4f}, "
        f"last={basis.iloc[-1]:.4f}, mean={basis.mean():.4f}"
    )

    print("\nHead of dataframe:")
    # Show a compact head so your terminal doesn’t explode
    print(df.head()[["tick", "fair_value", "dollar_delta_usd", "basis_usd", "daily_pnl_total"]])


def plot_series(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path,
                ylabel: str, title: str) -> None:
    plt.figure()
    plt.plot(df[x_col], df[y_col])
    plt.xlabel("tick")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def make_plots(df: pd.DataFrame, path: Path) -> None:
    stem = path.with_suffix("").name
    base_dir = path.parent

    pnl_png = base_dir / f"{stem}_pnl_curve.png"
    delta_png = base_dir / f"{stem}_delta_curve.png"
    basis_png = base_dir / f"{stem}_basis_curve.png"

    plot_series(
        df,
        x_col="tick",
        y_col="daily_pnl_total",
        out_path=pnl_png,
        ylabel="daily_pnl_total",
        title="Total PnL vs tick",
    )

    plot_series(
        df,
        x_col="tick",
        y_col="dollar_delta_usd",
        out_path=delta_png,
        ylabel="dollar_delta_usd",
        title="Dollar delta vs tick",
    )

    plot_series(
        df,
        x_col="tick",
        y_col="basis_usd",
        out_path=basis_png,
        ylabel="basis_usd",
        title="Basis exposure vs tick",
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python3 tools/research_ticks.py <ticks.jsonl>", file=sys.stderr)
        return 1

    path = Path(argv[1])
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    df = load_ticks(path)
    print_summary(df, path)
    make_plots(df, path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
