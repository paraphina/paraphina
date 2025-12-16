#!/usr/bin/env python3
"""
exp07_optimal_presets.py

Decode the optimised world-model constant policy from wm04 and turn it
into human-readable, clamped TAO MM knobs.

Inputs
------
- runs/wm04_constant_policy/optimised_action.npy  (6-dimensional SA feature)

Outputs
-------
- runs/exp07_optimal_presets/summary.txt          (pretty text summary)
- runs/exp07_optimal_presets/optimal_knobs.json   (machine-readable knobs)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"

WM04_DIR = RUNS_DIR / "wm04_constant_policy"
ACTION_PATH = WM04_DIR / "optimised_action.npy"

OUT_DIR = RUNS_DIR / "exp07_optimal_presets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Action-space clipping (SA feature space)
# These are the min / max values seen in the training dataset for each
# action dimension (from wm04's "Dataset action stats per dim").
#
# dim 0 ~ band_base / 5
# dim 1 ~ mm_size_eta * 40
# dim 2 ~ vol_ref * 100
# dim 3 ~ daily_loss_limit / 5000
# dim 4 ~ init_q_tao / 40
# dim 5 ~ vol_scale / 2
# ---------------------------------------------------------------------------

A_MIN = np.array(
    [0.375, 1.0, 0.84375, -10.0, 0.15, 0.375],
    dtype=np.float32,
)
A_MAX = np.array(
    [1.125, 4.0, 2.8125, 0.0, 1.0, 0.75],
    dtype=np.float32,
)


def decode_action_to_knobs(a_feat: np.ndarray) -> dict:
    """
    Convert a 6-D SA feature action vector into real-world MM knobs.

    We first clip a_feat into the dataset support [A_MIN, A_MAX] to avoid
    wild extrapolation, then invert the normalisation used in
    world_model_dataset._build_action_vector_from_row.
    """
    if a_feat.shape != (6,):
        raise ValueError(f"Expected shape (6,), got {a_feat.shape}")

    # Clip in feature space
    a_clipped = np.clip(a_feat, A_MIN, A_MAX)

    # Inverse normalisation (must match world_model_dataset.py)
    band_base = float(a_clipped[0] * 5.0)          # TAO
    mm_size_eta = float(a_clipped[1] / 40.0)       # dimless
    vol_ref = float(a_clipped[2] / 100.0)          # level
    daily_loss_limit = float(a_clipped[3] * 5000.0)  # USD
    init_q_tao = float(a_clipped[4] * 40.0)        # TAO
    vol_scale = float(a_clipped[5] * 2.0)          # scenario factor

    return {
        "band_base": band_base,
        "mm_size_eta": mm_size_eta,
        "vol_ref": vol_ref,
        "daily_loss_limit": daily_loss_limit,
        "init_q_tao": init_q_tao,
        "vol_scale": vol_scale,
        # keep the raw / clipped feature vectors for debugging
        "a_raw": a_feat.tolist(),
        "a_clipped": a_clipped.tolist(),
    }


def main() -> None:
    if not ACTION_PATH.exists():
        raise SystemExit(
            f"Optimised action not found at {ACTION_PATH}.\n"
            "Run wm04_optimize_constant_policy.py first."
        )

    # 1) Load optimised action vector
    a = np.load(ACTION_PATH).astype(np.float32).reshape(-1)
    if a.shape[0] != 6:
        raise SystemExit(f"Expected 6-D action, got shape {a.shape}")

    knobs = decode_action_to_knobs(a)

    summary_path = OUT_DIR / "summary.txt"
    json_path = OUT_DIR / "optimal_knobs.json"

    # 2) Human-readable summary
    with summary_path.open("w") as f:
        f.write("exp07: decoded constant policy from wm04 (world-model)\n")
        f.write("\nRaw SA action vector a:\n")
        f.write(f"  {knobs['a_raw']}\n")
        f.write("\nClipped SA action vector a':\n")
        f.write(f"  {knobs['a_clipped']}\n")

        f.write("\nDecoded TAO MM knobs (after clipping):\n")
        f.write(f"  band_base       (TAO): {knobs['band_base']:.4f}\n")
        f.write(f"  mm_size_eta          : {knobs['mm_size_eta']:.5f}\n")
        f.write(f"  vol_ref          (lvl): {knobs['vol_ref']:.5f}\n")
        f.write(f"  daily_loss_limit (USD): {knobs['daily_loss_limit']:.2f}\n")
        f.write(f"  init_q_tao       (TAO): {knobs['init_q_tao']:.2f}\n")
        f.write(f"  vol_scale     (factor): {knobs['vol_scale']:.3f}\n")

    # 3) Machine-readable knobs (without the raw/clipped feature vectors)
    dump = {
        k: v
        for k, v in knobs.items()
        if not k.startswith("a_")
    }
    with json_path.open("w") as f:
        json.dump(dump, f, indent=2, sort_keys=True)

    print(f"[exp07] Wrote summary to {summary_path}")
    print(f"[exp07] Wrote optimal knobs JSON to {json_path}")


if __name__ == "__main__":
    main()
