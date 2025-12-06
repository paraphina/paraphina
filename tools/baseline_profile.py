#!/usr/bin/env python3
"""
Baseline profile runner for Paraphina.

Purpose
-------
Run a single, calibrated "baseline" simulation end-to-end:

  * calls `cargo run` with our preferred profile:
      - initial_q_tao      ~ 0 TAO (delta-neutral start)
      - hedge_band_base    ~ 5 TAO
      - loss_limit_usd     ~ 4000 USD (risk guardrail)
      - ticks              (default 10_000)

  * writes the JSONL tick log into `runs/`,
  * calls `tools/research_ticks.py` to generate:
      - *_pnl_curve.png
      - *_delta_curve.png
      - *_basis_curve.png

This becomes our one-command smoke test whenever we change code.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def run_baseline(
    name: str,
    ticks: int,
    initial_q_tao: float,
    hedge_band_base: float,
    loss_limit_usd: float,
) -> None:
    """Run a single baseline simulation + analysis."""
    RUNS.mkdir(exist_ok=True)

    log_path = RUNS / f"{name}.jsonl"

    # 1) Run the Rust engine with calibrated parameters.
    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--",
        "--ticks",
        str(ticks),
        "--initial-q-tao",
        str(initial_q_tao),
        "--hedge-band-base",
        str(hedge_band_base),
        "--loss-limit-usd",
        str(loss_limit_usd),
        "--log-jsonl",
        str(log_path),
    ]

    env = os.environ.copy()
    print(f"[baseline] Running engine: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)

    # 2) Post-process the tick log with research_ticks.py using
    #    the same Python interpreter (so venv is respected).
    prefix = name
    rt_cmd = [
        sys.executable,
        "tools/research_ticks.py",
        str(log_path),
        "--prefix",
        prefix,
    ]

    print(f"[baseline] Running research_ticks: {' '.join(rt_cmd)}")
    subprocess.run(rt_cmd, check=True, cwd=str(ROOT))

    print()
    print("Baseline run complete.")
    print(f"  Log:      {log_path}")
    print(f"  Figures:  {prefix}_pnl_curve.png, {prefix}_delta_curve.png, {prefix}_basis_curve.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a calibrated Paraphina baseline profile (engine + research plots)."
    )

    parser.add_argument(
        "--name",
        type=str,
        default="baseline_q0_band5_loss4k",
        help="Name / prefix for this baseline run (used for log + PNG filenames).",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=10_000,
        help="Number of ticks to simulate.",
    )
    parser.add_argument(
        "--initial-q-tao",
        type=float,
        default=0.0,
        help="Initial global position in TAO for this baseline profile.",
    )
    parser.add_argument(
        "--hedge-band-base",
        type=float,
        default=5.0,
        help="Base hedge band in TAO (see calibration Exp01/Exp02).",
    )
    parser.add_argument(
        "--loss-limit-usd",
        type=float,
        default=4000.0,
        help="Daily loss limit in USD for the hard kill-switch (see Exp04).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_baseline(
        name=args.name,
        ticks=args.ticks,
        initial_q_tao=args.initial_q_tao,
        hedge_band_base=args.hedge_band_base,
        loss_limit_usd=args.loss_limit_usd,
    )


if __name__ == "__main__":
    main()
