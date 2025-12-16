#!/usr/bin/env python3
"""
exp08_profile_validation.py

Stage 8: Package the world-model-derived knobs into profile presets and
sanity-check them against our risk profiles.

This is an *offline* step:

- Reads the world-model constant policy from:
      runs/wm04_constant_policy/optimised_action.npy
      runs/exp07_optimal_presets/optimal_knobs.json

- Uses our standard risk profiles:
      aggressive / balanced / conservative

- Hard-codes the daily loss limits we’ve already been using, and
  provides a sensible init_q_tao per profile (flat by default).

- Writes a CSV + summary text under:
      runs/exp08_profile_validation/

Outputs
=======

1) runs/exp08_profile_validation/exp08_presets.csv

   Columns:
      profile
      band_base
      mm_size_eta
      vol_ref
      daily_loss_limit
      init_q_tao
      vol_scale

2) runs/exp08_profile_validation/summary.txt

   Human-readable description of the presets.

This does *not* re-run the Rust sim; it just packages the world-model
knobs into a clean preset file that later experiments (or production)
can consume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
OUT_DIR = RUNS_DIR / "exp08_profile_validation"


# These are the risk-profile-specific daily loss limits we've been using
# throughout the earlier experiments.
DAILY_LOSS_LIMITS: Dict[str, float] = {
    "aggressive": 2000.0,
    "balanced": 5000.0,
    "conservative": 750.0,
}

# Default starting inventory per profile (TAO).
# Feel free to change these later; for now we keep everything flat.
DEFAULT_Q0: Dict[str, float] = {
    "aggressive": 0.0,
    "balanced": 0.0,
    "conservative": 0.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_world_model_knobs() -> Dict[str, float]:
    """
    Load the decoded TAO MM knobs from exp07_optimal_presets/optimal_knobs.json.

    Expected keys:
        band_base, mm_size_eta, vol_ref, daily_loss_limit, init_q_tao, vol_scale

    We only *use* band_base, mm_size_eta, vol_ref and vol_scale here; the
    per-profile daily_loss_limit and init_q_tao come from our risk profile
    definitions.
    """
    knobs_path = RUNS_DIR / "exp07_optimal_presets" / "optimal_knobs.json"
    if not knobs_path.exists():
        raise FileNotFoundError(
            f"Could not find optimal_knobs.json at {knobs_path}. "
            "Make sure exp07_optimal_presets has been run."
        )

    with knobs_path.open("r") as f:
        knobs = json.load(f)

    required = ["band_base", "mm_size_eta", "vol_ref", "vol_scale"]
    missing = [k for k in required if k not in knobs]
    if missing:
        raise KeyError(
            f"optimal_knobs.json is missing keys: {missing}. "
            f"Found keys: {list(knobs.keys())}"
        )

    return knobs


def build_profile_presets(knobs: Dict[str, float]) -> pd.DataFrame:
    """
    Construct a small DataFrame of presets, one row per risk profile.

    For now we reuse the *same* band_base / mm_size_eta / vol_ref / vol_scale
    across all profiles, and customise only daily_loss_limit and init_q_tao.
    """
    profiles: List[str] = ["aggressive", "balanced", "conservative"]

    rows = []
    for p in profiles:
        band_base = float(knobs["band_base"])
        mm_size_eta = float(knobs["mm_size_eta"])
        vol_ref = float(knobs["vol_ref"])
        vol_scale = float(knobs.get("vol_scale", 1.0))

        daily_loss_limit = float(DAILY_LOSS_LIMITS[p])
        init_q_tao = float(DEFAULT_Q0[p])

        rows.append(
            {
                "profile": p,
                "band_base": band_base,
                "mm_size_eta": mm_size_eta,
                "vol_ref": vol_ref,
                "daily_loss_limit": daily_loss_limit,
                "init_q_tao": init_q_tao,
                "vol_scale": vol_scale,
            }
        )

    df = pd.DataFrame(rows)
    return df


def write_outputs(df_presets: pd.DataFrame, knobs: Dict[str, float]) -> None:
    """Write CSV + human-readable summary to OUT_DIR."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / "exp08_presets.csv"
    df_presets.to_csv(csv_path, index=False)

    summary_path = OUT_DIR / "summary.txt"
    with summary_path.open("w") as f:
        print("exp08: world-model-derived profile presets\n", file=f)
        print("Optimal knobs from world model (exp07):", file=f)
        print(
            f"  band_base (TAO):    {knobs['band_base']:.4f}\n"
            f"  mm_size_eta:        {knobs['mm_size_eta']:.5f}\n"
            f"  vol_ref (level):    {knobs['vol_ref']:.5f}\n"
            f"  vol_scale (factor): {knobs.get('vol_scale', 1.0):.3f}\n",
            file=f,
        )

        print("Per-profile presets:", file=f)
        for _, row in df_presets.iterrows():
            print(
                f"\nProfile = {row['profile']}",
                file=f,
            )
            print(
                f"  band_base (TAO):      {row['band_base']:.4f}\n"
                f"  mm_size_eta:          {row['mm_size_eta']:.5f}\n"
                f"  vol_ref (level):      {row['vol_ref']:.5f}\n"
                f"  daily_loss_limit USD: {row['daily_loss_limit']:.2f}\n"
                f"  init_q_tao (TAO):     {row['init_q_tao']:.2f}\n"
                f"  vol_scale (scenario): {row['vol_scale']:.3f}",
                file=f,
            )

    print(f"[exp08] Wrote presets CSV to {csv_path}")
    print(f"[exp08] Wrote summary   to {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[exp08] Starting profile preset packaging...")
    print(f"[exp08] ROOT  = {ROOT}")
    print(f"[exp08] RUNS  = {RUNS_DIR}")
    print(f"[exp08] OUT   = {OUT_DIR}")

    knobs = load_world_model_knobs()
    print("[exp08] Loaded optimal knobs from exp07_optimal_presets/optimal_knobs.json:")
    for k, v in knobs.items():
        print(f"    {k}: {v}")

    df_presets = build_profile_presets(knobs)
    print("\n[exp08] Resulting presets:")
    print(df_presets.to_string(index=False))

    write_outputs(df_presets, knobs)
    print("[exp08] Done.")


if __name__ == "__main__":
    main()
