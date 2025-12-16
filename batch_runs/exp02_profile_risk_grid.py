#!/usr/bin/env python3
"""
exp02_profile_risk_grid.py

DARPA-grade hypersearch harness for Paraphina.

What it does
============

- Defines three coarse "risk profiles":
    * conservative
    * balanced
    * aggressive

  Each profile is just a *bundle* of the core knobs you exposed in src/config.rs:

    - PARAPHINA_HEDGE_BAND_BASE
    - PARAPHINA_MM_SIZE_ETA
    - PARAPHINA_VOL_REF
    - PARAPHINA_DAILY_LOSS_LIMIT

- Around each base profile, it sweeps MULTIPLICATIVE factors for:
    - hedge_band_base
    - mm.size_eta
    - vol_ref
    - daily_loss_limit

- For every (profile, band_mult, eta_mult, vol_mult, loss_mult, seed) combo:
    - Runs: `cargo run --release -- --ticks TICKS`
    - Sets env overrides for all the knobs above
    - Captures stdout and parses the final summary:

        Daily PnL (realised / unrealised / total)
        Risk regime: ...
        Kill switch: true/false

- Writes:
    - runs/exp02_profile_grid/exp02_profile_grid_runs.csv
      (one row per *simulation run*)

    - runs/exp02_profile_grid/exp02_profile_grid_summary.csv
      (aggregated over seeds for each config point)

    - runs/exp02_profile_grid/exp02_profile_grid_pnl_vs_band.png
      (mean PnL vs hedge band, per profile)

You can shrink or enlarge the grid by editing the lists near the top:
    RISK_PROFILES, HEDGE_BAND_MULTS, MM_SIZE_ETA_MULTS, VOL_REF_MULTS, LOSS_LIMIT_MULTS.
"""

import os
import re
import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths / global constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs" / "exp02_profile_grid"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# How many ticks per simulation
TICKS = 500

# How many "seeds" (= independent runs) per configuration
NUM_SEEDS = 3

# We keep hedge_max_step fixed here; you can also sweep it later.
HEDGE_MAX_STEP = 20.0

# ---------------------------------------------------------------------------
# Base profile definitions (mirror src/config.rs RiskProfile presets)
# ---------------------------------------------------------------------------

# These are the *canonical* base hyperparameters for each coarse profile.
# All sweeps are *multiplicative* around these.
BASE_PROFILES = {
    "conservative": dict(
        hedge_band_base=2.5,
        mm_size_eta=0.20,
        vol_ref=0.015,
        daily_loss_limit=1_500.0,
    ),
    "balanced": dict(
        hedge_band_base=5.0,
        mm_size_eta=0.10,
        vol_ref=0.020,
        daily_loss_limit=10_000.0,
    ),
    "aggressive": dict(
        hedge_band_base=7.5,
        mm_size_eta=0.05,
        vol_ref=0.025,
        daily_loss_limit=4_000.0,
    ),
}

# Dimensions of the search grid.
#
# IMPORTANT:
#   The Cartesian product size is:
#       |profiles| × |band_mults| × |eta_mults| × |vol_mults| × |loss_mults| × NUM_SEEDS
#
#   With the defaults below:
#       3 × 3 × 3 × 3 × 3 × 3  =  729 runs
#
#   If that's too heavy, just shrink these lists.
RISK_PROFILES = ["conservative", "balanced", "aggressive"]

HEDGE_BAND_MULTS = [0.75, 1.0, 1.25]
MM_SIZE_ETA_MULTS = [0.5, 1.0, 2.0]
VOL_REF_MULTS = [0.75, 1.0, 1.25]
LOSS_LIMIT_MULTS = [0.5, 1.0, 2.0]

# Initial global q_t; you can also sweep this if needed.
INIT_Q_TAO = 0.0


# ---------------------------------------------------------------------------
# Helpers: running the sim + parsing stdout
# ---------------------------------------------------------------------------

def run_single_sim(
    profile: str,
    hedge_band_base: float,
    mm_size_eta: float,
    vol_ref: float,
    daily_loss_limit: float,
    seed_index: int,
) -> dict:
    """
    Run a single Paraphina simulation with a given hyperparameter slice.

    Returns a dict of metrics parsed from stdout, plus the return code and raw stdout.
    """

    env = os.environ.copy()

    # Core risk / control knobs wired in src/config.rs
    env["PARAPHINA_INIT_Q_TAO"] = str(INIT_Q_TAO)
    env["PARAPHINA_HEDGE_BAND_BASE"] = f"{hedge_band_base:.6f}"
    env["PARAPHINA_HEDGE_MAX_STEP"] = f"{HEDGE_MAX_STEP:.6f}"
    env["PARAPHINA_MM_SIZE_ETA"] = f"{mm_size_eta:.6f}"
    env["PARAPHINA_VOL_REF"] = f"{vol_ref:.6f}"
    env["PARAPHINA_DAILY_LOSS_LIMIT"] = f"{daily_loss_limit:.6f}"

    # Optional: annotate for logging / debugging
    env["PARAPHINA_EXPERIMENT_TAG"] = (
        f"exp02_profile={profile}"
        f"_band={hedge_band_base:.4f}"
        f"_eta={mm_size_eta:.4f}"
        f"_volref={vol_ref:.4f}"
        f"_loss={daily_loss_limit:.0f}"
        f"_seed={seed_index}"
    )

    cmd = ["cargo", "run", "--release", "--quiet", "--", "--ticks", str(TICKS)]

    result = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout
    stderr = result.stderr

    metrics = parse_stdout_metrics(stdout)
    metrics.update(
        returncode=result.returncode,
        stderr_truncated="\n".join(stderr.splitlines()[-10:]),
        stdout_truncated="\n".join(stdout.splitlines()[-20:]),
    )

    return metrics


_PNL_RE = re.compile(
    r"Daily PnL \(realised / unrealised / total\):\s+([\-0-9.]+)\s*/\s*([\-0-9.]+)\s*/\s*([\-0-9.]+)"
)

_RISK_RE = re.compile(r"Risk regime(?: after tick)?:\s+(\w+)")
_KILL_TRUE_RE = re.compile(r"Kill switch[:\s].*true", re.IGNORECASE)
_KILL_ACTIVE_RE = re.compile(r"Kill switch active", re.IGNORECASE)

_DELTA_RE = re.compile(r"Dollar delta \(USD\):\s+([\-0-9.]+)")
_BASIS_RE = re.compile(r"Basis exposure \(USD\):\s+([\-0-9.]+)")


def parse_stdout_metrics(stdout: str) -> dict:
    """
    Parse the key summary metrics from Paraphina's stdout.

    We only look at the *last* occurrence of each pattern to avoid
    picking up intermediate ticks.
    """
    lines = stdout.splitlines()

    # PnL (realised, unrealised, total)
    pnl_matches = _PNL_RE.findall(stdout)
    if pnl_matches:
        realised_str, unrealised_str, total_str = pnl_matches[-1]
        pnl_realised = float(realised_str)
        pnl_unrealised = float(unrealised_str)
        pnl_total = float(total_str)
    else:
        pnl_realised = np.nan
        pnl_unrealised = np.nan
        pnl_total = np.nan

    # Risk regime
    risk_matches = _RISK_RE.findall(stdout)
    risk_regime = risk_matches[-1] if risk_matches else "Unknown"

    # Kill switch
    kill_switch = bool(
        _KILL_TRUE_RE.search(stdout) or _KILL_ACTIVE_RE.search(stdout)
    )

    # Final delta / basis exposure (from the last "After MM + hedge fills" block)
    delta_matches = _DELTA_RE.findall(stdout)
    basis_matches = _BASIS_RE.findall(stdout)

    dollar_delta_usd = float(delta_matches[-1]) if delta_matches else np.nan
    basis_exposure_usd = float(basis_matches[-1]) if basis_matches else np.nan

    return dict(
        pnl_realised=pnl_realised,
        pnl_unrealised=pnl_unrealised,
        pnl_total=pnl_total,
        risk_regime=risk_regime,
        kill_switch=kill_switch,
        dollar_delta_usd=dollar_delta_usd,
        basis_exposure_usd=basis_exposure_usd,
    )


# ---------------------------------------------------------------------------
# Main hypersearch loop
# ---------------------------------------------------------------------------

def main() -> None:
    total_configs = (
        len(RISK_PROFILES)
        * len(HEDGE_BAND_MULTS)
        * len(MM_SIZE_ETA_MULTS)
        * len(VOL_REF_MULTS)
        * len(LOSS_LIMIT_MULTS)
        * NUM_SEEDS
    )
    print(f"[exp02] Running hypersearch with {total_configs} simulations...")

    rows = []

    for (
        profile,
        band_mult,
        eta_mult,
        vol_mult,
        loss_mult,
    ) in product(
        RISK_PROFILES,
        HEDGE_BAND_MULTS,
        MM_SIZE_ETA_MULTS,
        VOL_REF_MULTS,
        LOSS_LIMIT_MULTS,
    ):
        base = BASE_PROFILES[profile]

        hedge_band = base["hedge_band_base"] * band_mult
        mm_size_eta = base["mm_size_eta"] * eta_mult
        vol_ref = base["vol_ref"] * vol_mult
        daily_loss_limit = base["daily_loss_limit"] * loss_mult

        for seed in range(NUM_SEEDS):
            print(
                f"[exp02] profile={profile:12s} "
                f"band={hedge_band:6.3f} (×{band_mult:.2f}) "
                f"eta={mm_size_eta:7.4f} (×{eta_mult:.2f}) "
                f"vol_ref={vol_ref:7.4f} (×{vol_mult:.2f}) "
                f"loss={daily_loss_limit:8.1f} (×{loss_mult:.2f}) "
                f"seed={seed}"
            )

            metrics = run_single_sim(
                profile=profile,
                hedge_band_base=hedge_band,
                mm_size_eta=mm_size_eta,
                vol_ref=vol_ref,
                daily_loss_limit=daily_loss_limit,
                seed_index=seed,
            )

            row = dict(
                profile=profile,
                band_mult=band_mult,
                band_base=hedge_band,
                mm_size_eta_mult=eta_mult,
                mm_size_eta=mm_size_eta,
                vol_ref_mult=vol_mult,
                vol_ref=vol_ref,
                loss_mult=loss_mult,
                daily_loss_limit=daily_loss_limit,
                seed=seed,
                ticks=TICKS,
            )
            row.update(metrics)
            rows.append(row)

    df_runs = pd.DataFrame(rows)
    runs_csv = RUNS_DIR / "exp02_profile_grid_runs.csv"
    df_runs.to_csv(runs_csv, index=False)
    print(f"[exp02] Wrote per-run metrics to {runs_csv}")

    # -----------------------------------------------------------------------
    # Aggregated summary over seeds
    # -----------------------------------------------------------------------

    group_cols = [
        "profile",
        "band_mult",
        "band_base",
        "mm_size_eta_mult",
        "mm_size_eta",
        "vol_ref_mult",
        "vol_ref",
        "loss_mult",
        "daily_loss_limit",
    ]

    def hardlimit_frac(series: pd.Series) -> float:
        return float((series == "HardLimit").mean())

    summary = (
        df_runs.groupby(group_cols)
        .agg(
            runs=("seed", "count"),
            pnl_total_mean=("pnl_total", "mean"),
            pnl_total_std=("pnl_total", "std"),
            pnl_total_min=("pnl_total", "min"),
            pnl_total_max=("pnl_total", "max"),
            kill_switch_frac=("kill_switch", "mean"),
            hardlimit_frac=("risk_regime", hardlimit_frac),
        )
        .reset_index()
    )

    summary_csv = RUNS_DIR / "exp02_profile_grid_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[exp02] Wrote aggregated summary to {summary_csv}")

    # -----------------------------------------------------------------------
    # Simple diagnostic plot: mean PnL vs hedge band, per profile
    # -----------------------------------------------------------------------

    plt.figure()
    for profile, sub in summary.groupby("profile"):
        sub_sorted = sub.sort_values("band_base")
        plt.plot(
            sub_sorted["band_base"],
            sub_sorted["pnl_total_mean"],
            marker="o",
            label=profile,
        )

    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Hedge band base (TAO)")
    plt.ylabel("Mean total PnL over seeds (USD)")
    plt.title("exp02: Mean PnL vs hedge band (per profile)")
    plt.legend()
    plt.tight_layout()

    plot_path = RUNS_DIR / "exp02_profile_grid_pnl_vs_band.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[exp02] Saved PnL vs band plot to {plot_path}")

    # -----------------------------------------------------------------------
    # Console: top configs by PnL, filtered by risk sanity
    # -----------------------------------------------------------------------

    # Only consider configs where kill switch fired in < 10% of runs
    sane = summary[summary["kill_switch_frac"] < 0.10].copy()

    if not sane.empty:
        top = sane.sort_values("pnl_total_mean", ascending=False).head(15)
        cols = [
            "profile",
            "band_base",
            "mm_size_eta",
            "vol_ref",
            "daily_loss_limit",
            "pnl_total_mean",
            "pnl_total_std",
            "kill_switch_frac",
            "hardlimit_frac",
        ]
        print("\n[exp02] Top configs by mean PnL (with kill_switch_frac < 0.10):")
        print(top[cols].to_string(index=False))
    else:
        print("\n[exp02] No configs passed the kill_switch_frac < 0.10 filter.")


if __name__ == "__main__":
    main()
