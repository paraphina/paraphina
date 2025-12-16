#!/usr/bin/env python3
"""
wm02_dream_rollouts.py

Evaluate the wm01 MLP world model by generating "dreamed" rollouts and
comparing them to real trajectories from exp04.

- Load DecisionTrajectory objects from exp04_runs_with_ts.csv.
- Load the trained world model checkpoint:
    runs/wm01_world_model/world_model_mlp.pt

- For a bunch of starting states, roll the model forward for H steps:
      s_{t+1} = f_theta(s_t)
  and compute rewards from the predicted state.

- For comparison, take real slices of length H from the telemetry and
  compute the same metrics.

We then write a summary CSV with per-episode stats:

    runs/wm02_dream_eval/wm02_dream_vs_real_summary.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch

from world_model_dataset import build_trajectories_from_runs_csv, DecisionTrajectory
from wm01_train_world_model import MLPWorldModel


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DreamConfig:
    horizon: int = 50              # decision steps per dreamed episode
    num_episodes: int = 200        # how many dreamed episodes to generate
    seed: int = 123
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers: reward / done from state vector
# ---------------------------------------------------------------------------

# State layout (same as world_model_dataset._build_decision_steps):
#
#   0: pnl_end
#   1: pnl_delta
#   2: dd_in_slice
#   3: q_global
#   4: delta_usd
#   5: basis_usd
#   6: sigma_eff   (may be synthetic / 0)
#   7: vol_ratio   (may be synthetic / 0)
#   8: frac_risk_normal
#   9: frac_risk_warning
#  10: frac_risk_hardlimit
#

def reward_from_state(state: np.ndarray) -> float:
    """Same reward definition used in world_model_dataset."""
    pnl_delta = float(state[1])
    dd_in_slice = float(state[2])
    return float(pnl_delta - 0.1 * dd_in_slice)


def done_from_state(state: np.ndarray) -> bool:
    """
    Heuristic "done" condition from predicted state.

    If the predicted HardLimit risk fraction is high, we treat this as an
    episode termination. This approximates a kill-switch event in the
    dreamed environment.
    """
    frac_hard = float(state[10])
    return frac_hard > 0.5


def sanitise_state(state: np.ndarray) -> np.ndarray:
    """Clip / fix NaNs to keep the model's rollouts numerically stable."""
    state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
    state = np.clip(state, -1e6, 1e6)
    return state.astype(np.float32)


# ---------------------------------------------------------------------------
# Dreamed rollouts using the world model
# ---------------------------------------------------------------------------

def dream_rollouts(
    model: MLPWorldModel,
    trajectories: List[DecisionTrajectory],
    cfg: DreamConfig,
) -> pd.DataFrame:
    """
    Generate dreamed rollouts:

    - Sample initial states from real trajectories.
    - Roll forward with the world model for cfg.horizon steps.
    - Compute total reward, final pnl, max drawdown, etc.

    Returns a DataFrame with one row per dreamed episode.
    """
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)
    model.eval()

    rows: List[Dict[str, float]] = []

    # Flatten all states from all real trajectories to sample starting points.
    all_states = np.concatenate([tr.states for tr in trajectories], axis=0)
    num_available = all_states.shape[0]

    for ep in range(cfg.num_episodes):
        # Sample a random starting state from real data.
        idx0 = rng.integers(0, num_available)
        s = all_states[idx0].astype(np.float32)
        s = sanitise_state(s)

        total_reward = 0.0
        max_dd = -np.inf
        final_pnl = float(s[0])
        frac_hard_cum = 0.0

        for t in range(cfg.horizon):
            x = torch.from_numpy(s).unsqueeze(0).to(device)

            with torch.no_grad():
                s_next = model(x).squeeze(0).cpu().numpy()

            s_next = sanitise_state(s_next)

            r = reward_from_state(s_next)
            total_reward += r

            dd = float(s_next[2])
            max_dd = max(max_dd, dd)

            final_pnl = float(s_next[0])

            frac_hard = float(s_next[10])
            frac_hard_cum += frac_hard

            s = s_next

            if done_from_state(s_next):
                break

        avg_frac_hard = frac_hard_cum / float(cfg.horizon)

        rows.append(
            {
                "source": "model",
                "episode": ep,
                "len": t + 1,
                "total_reward": total_reward,
                "final_pnl_end": final_pnl,
                "max_dd_in_episode": max_dd,
                "avg_frac_hard": avg_frac_hard,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Real rollouts from telemetry for comparison
# ---------------------------------------------------------------------------

def real_rollouts(
    trajectories: List[DecisionTrajectory],
    cfg: DreamConfig,
) -> pd.DataFrame:
    """
    Sample real slices of length cfg.horizon from the telemetry-based
    DecisionTrajectories and compute the same metrics as for dreamed
    episodes.
    """
    rng = np.random.default_rng(cfg.seed + 1)

    rows: List[Dict[str, float]] = []

    # Precompute eligible (trajectory, start_index) pairs that have at least
    # cfg.horizon steps remaining.
    eligible: List[tuple[int, int]] = []
    for ti, tr in enumerate(trajectories):
        T = tr.states.shape[0]
        if T >= cfg.horizon:
            for start in range(0, T - cfg.horizon + 1):
                eligible.append((ti, start))

    if not eligible:
        raise ValueError("No trajectories have enough length for the chosen horizon.")

    num_eps = min(cfg.num_episodes, len(eligible))

    chosen_indices = rng.choice(len(eligible), size=num_eps, replace=False)

    for ep_idx, elig_idx in enumerate(chosen_indices):
        tr_i, start = eligible[int(elig_idx)]
        tr = trajectories[tr_i]

        states = tr.states[start : start + cfg.horizon]
        rewards = tr.rewards[start : start + cfg.horizon]
        dones = tr.dones[start : start + cfg.horizon]

        total_reward = float(rewards.sum())
        final_pnl = float(states[-1, 0])
        max_dd = float(states[:, 2].max())
        avg_frac_hard = float(states[:, 10].mean())

        # Effective length if kill-switch fired earlier
        effective_len = cfg.horizon
        if dones.any():
            first_done = int(np.argmax(dones))
            effective_len = first_done + 1

        rows.append(
            {
                "source": "real",
                "episode": ep_idx,
                "len": effective_len,
                "total_reward": total_reward,
                "final_pnl_end": final_pnl,
                "max_dd_in_episode": max_dd,
                "avg_frac_hard": avg_frac_hard,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = DreamConfig()

    runs_csv = ROOT / "runs" / "exp04_multi_scenario" / "exp04_runs_with_ts.csv"
    print(f"[wm02] Loading trajectories from {runs_csv} ...")
    trajectories = build_trajectories_from_runs_csv(
        runs_csv,
        decision_horizon=10,
    )
    print(f"[wm02] Loaded {len(trajectories)} trajectories.")

    ckpt_path = ROOT / "runs" / "wm01_world_model" / "world_model_mlp.pt"
    print(f"[wm02] Loading world model checkpoint from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    state_dim = ckpt["state_dim"]
    hidden_dim = ckpt["hidden_dim"]

    model = MLPWorldModel(state_dim=state_dim, hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[wm02] Loaded model with state_dim={state_dim}, hidden_dim={hidden_dim}")

    print(
        f"[wm02] Generating {cfg.num_episodes} dreamed episodes with horizon={cfg.horizon} ..."
    )
    df_model = dream_rollouts(model, trajectories, cfg)
    print("[wm02] Dreamed episodes head:")
    print(df_model.head())

    print(
        f"[wm02] Sampling {cfg.num_episodes} real episodes of horizon={cfg.horizon} ..."
    )
    df_real = real_rollouts(trajectories, cfg)
    print("[wm02] Real episodes head:")
    print(df_real.head())

    df_all = pd.concat([df_model, df_real], ignore_index=True)

    out_dir = ROOT / "runs" / "wm02_dream_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "wm02_dream_vs_real_summary.csv"
    df_all.to_csv(out_path, index=False)
    print(f"[wm02] Wrote dreamed vs real episode summary to {out_path}")

    # Print overall aggregate stats for quick inspection.
    for src in ["model", "real"]:
        sub = df_all[df_all["source"] == src]
        print(f"\n[wm02] Aggregate stats for {src}:")
        print(
            sub[["total_reward", "final_pnl_end", "max_dd_in_episode", "avg_frac_hard"]]
            .describe()
        )


if __name__ == "__main__":
    main()
