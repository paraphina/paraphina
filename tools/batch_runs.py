#!/usr/bin/env python3
"""
tools/batch_runs.py

Hedge-band sweep harness for Paraphina.

For each hedge band in HEDGE_BANDS_TAO, this script:
  - sets PARA_HEDGE_BAND_TAO in the environment,
  - runs the paraphina binary with --ticks / --log-jsonl,
  - loads the JSONL tick log,
  - computes summary metrics (final PnL, max drawdown, max |delta|, max |basis|),
  - writes a CSV summary,
  - and plots final PnL vs hedge band.

Outputs in repo root:
  - exp02_hedge_band_hb*.jsonl
  - exp02_hedge_band_summary.csv
  - exp02_hedge_band_pnl_vs_band.png
"""

import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ----------- Paths & global constants -----------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BINARY = PROJECT_ROOT / "target" / "debug" / "paraphina"

NUM_TICKS = 200
EXPERIMENT_PREFIX = "exp02_hedge_band"

# TAO half-band values we want to test.
HEDGE_BANDS_TAO = [1.0, 2.5, 5.0, 10.0, 20.0]


# ----------- Helpers -----------

def build_binary() -> None:
    """Ensure the Rust binary is built."""
    print(">>> Building paraphina binary (cargo build)...")
    subprocess.run(
        ["cargo", "build", "--quiet"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print(">>> Build complete.\n")


def run_one(hedge_band_tao: float, jsonl_path: Path) -> None:
    """
    Run a single simulation with a given hedge band.

    hedge_band_tao is injected via PARA_HEDGE_BAND_TAO.
    """
    env = os.environ.copy()
    env["PARA_HEDGE_BAND_TAO"] = str(hedge_band_tao)
    env["PARA_RUN_LABEL"] = f"hb_{hedge_band_tao}"

    cmd = [
        str(BINARY),
        "--ticks",
        str(NUM_TICKS),
        "--log-jsonl",
        str(jsonl_path),
    ]

    print(f">>> Running hedge_band_tao={hedge_band_tao}")
    print("    CMD:", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    print(f"    Wrote ticks to {jsonl_path.name}\n")


def load_ticks(jsonl_path: Path) -> pd.DataFrame:
    """Load a JSONL tick file into a pandas DataFrame."""
    records = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    df = pd.DataFrame.from_records(records)
    return df


def summarise_run(df: pd.DataFrame, hedge_band_tao: float) -> dict:
    """
    Compute summary metrics for a single run.

    Uses the standard Paraphina tick schema:
      - daily_pnl_total
      - dollar_delta_usd
      - basis_usd
    """
    pnl = df["daily_pnl_total"].astype(float)

    final_pnl = float(pnl.iloc[-1])
    cummax = pnl.cummax()
    drawdown = cummax - pnl
    max_drawdown = float(drawdown.max())

    max_abs_delta = float(df["dollar_delta_usd"].abs().max())
    max_abs_basis = float(df["basis_usd"].abs().max())

    return {
        "hedge_band_tao": hedge_band_tao,
        "final_pnl": final_pnl,
        "max_drawdown": max_drawdown,
        "max_abs_delta_usd": max_abs_delta,
        "max_abs_basis_usd": max_abs_basis,
    }


def plot_pnl_vs_band(summary_df: pd.DataFrame, png_path: Path) -> None:
    """Plot final PnL vs hedge band."""
    summary_df = summary_df.sort_values("hedge_band_tao")

    plt.figure()
    plt.plot(
        summary_df["hedge_band_tao"],
        summary_df["final_pnl"],
        marker="o",
    )
    plt.xlabel("hedge_band_tao (TAO)")
    plt.ylabel("final_pnl (USD)")
    plt.title("Final PnL vs hedge band")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    print(f"Saved PnL plot to {png_path.name}")


# ----------- Main entrypoint -----------

def main() -> None:
    build_binary()

    summaries = []

    for hb in HEDGE_BANDS_TAO:
        jsonl_name = f"{EXPERIMENT_PREFIX}_hb{hb}.jsonl"
        jsonl_path = PROJECT_ROOT / jsonl_name

        run_one(hb, jsonl_path)
        df = load_ticks(jsonl_path)
        summary = summarise_run(df, hb)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    csv_path = PROJECT_ROOT / f"{EXPERIMENT_PREFIX}_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print("\n=== Hedge-band sweep summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary CSV to {csv_path.name}")

    png_path = PROJECT_ROOT / f"{EXPERIMENT_PREFIX}_pnl_vs_band.png"
    plot_pnl_vs_band(summary_df, png_path)


if __name__ == "__main__":
    main()
