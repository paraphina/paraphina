#!/usr/bin/env python3
"""
wm04_optimize_constant_policy.py

Simple world-model optimisation over a CONSTANT 6-dim action vector.

Interpretation
--------------
- We use the **state+action (SA) world model** trained in wm03:
    runs/wm03_world_model_sa/world_model_sa_mlp.pt

- The action vector 'a' is 6-dimensional and corresponds to the same
  per-run hyperparameters we logged into the SA dataset, e.g.:

    [ band_base, mm_size_eta, vol_ref, daily_loss_limit,
      init_q_tao, vol_scale ]

  (Exact semantics are defined by world_model_dataset; here we just treat
   them as a 6-dim 'control knob' passed into the world model.)

- We optimise a *constant* action vector: same 'a' is used at every step
  and for every episode. This is the simplest possible "policy":

      π(s_t) = a   for all t

- Reward proxy:
    We assume the **first state feature** is a normalised "pnl_total"
    proxy, so we define reward_t = next_state[:, 0].
    Then we maximise the discounted sum of that over a finite horizon.

This is NOT production trading logic. It's a minimal, fully contained
"RL-ish" optimisation to validate the world-model → policy loop.

Outputs
-------
- runs/wm04_constant_policy/optimised_action.npy   (shape [6])
- runs/wm04_constant_policy/summary.txt            (human-readable log)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class RLConfig:
    seed: int = 42

    # World-model rollout horizon (model steps, not real seconds).
    horizon: int = 50

    # How many episodes we sample per optimisation step.
    batch_size: int = 64

    # Number of gradient steps on the constant action vector.
    num_iters: int = 300

    # Optimiser hyperparams.
    lr: float = 0.05
    weight_decay: float = 0.0

    # Discount factor for rewards.
    gamma: float = 0.99

    # Regularisation on action magnitude (keeps us near dataset regime).
    action_l2_reg: float = 0.01


# ---------------------------------------------------------------------
# Minimal SA world model (matches wm03 checkpoint)
# ---------------------------------------------------------------------


class MLPWorldModel(nn.Module):
    """
    Simple MLP world model: f(s_t, a_t) -> s_{t+1}

    The exact layer sizes are inferred from the checkpoint metadata
    (input_dim, hidden_dim, state_dim).
    """

    def __init__(self, input_dim: int, hidden_dim: int, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sa_trajectories(runs_csv: Path) -> List:
    """
    Load state+action trajectories from the exp04 runs CSV.

    We reuse the helper from `world_model_dataset`. Depending on how
    that file evolved, the SA builder may have one of the following
    names; we try both and fail with a clear error if neither exists.
    """
    try:
        from world_model_dataset import (
            build_sa_trajectories_from_runs_csv as build_trajs,
        )
    except ImportError:
        try:
            from world_model_dataset import (
                build_trajectories_from_runs_csv as build_trajs,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Could not import SA trajectory builder from world_model_dataset. "
                "Expected one of:\n"
                "  - build_sa_trajectories_from_runs_csv\n"
                "  - build_trajectories_from_runs_csv\n"
                "Please open world_model_dataset.py and adjust wm04 accordingly."
            ) from exc

    # We assume the builder returns a list[Trajectory] where each Trajectory
    # has at least:
    #   - states:  np.ndarray [T, state_dim]
    #   - actions: np.ndarray [T, action_dim]  (for SA variant)
    #   - rewards, dones, meta: present but unused here.
    trajs = build_trajs(runs_csv, decision_horizon=10, include_actions=True)
    if not trajs:
        raise RuntimeError(f"No trajectories loaded from {runs_csv}")

    first = trajs[0]
    if not hasattr(first, "actions"):
        raise RuntimeError(
            "Loaded trajectories do not have an `.actions` field. "
            "wm04 expects SA trajectories (state+action). Please check "
            "world_model_dataset.py and wm03_train_world_model_sa.py."
        )

    return trajs


def extract_initial_states_and_actions(trajs: List):
    """
    From a list of SA trajectories, build:

        S0: [N, state_dim]  – initial states
        A_data: [M, action_dim] – all actions observed in dataset
    """
    init_states = []
    all_actions = []

    for tr in trajs:
        states = np.asarray(tr.states, dtype=np.float32)
        actions = np.asarray(tr.actions, dtype=np.float32)

        if states.shape[0] == 0:
            continue

        init_states.append(states[0])
        all_actions.append(actions)

    if not init_states:
        raise RuntimeError("No non-empty trajectories found in SA dataset")

    S0 = np.stack(init_states, axis=0)
    A_data = np.concatenate(all_actions, axis=0)

    return S0, A_data


def build_world_model(ckpt_path: Path, device: torch.device) -> MLPWorldModel:
    """
    Load SA world model checkpoint from wm03 and construct the model.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])
    input_dim = int(ckpt["input_dim"])
    hidden_dim = int(ckpt["hidden_dim"])

    model = MLPWorldModel(input_dim=input_dim, hidden_dim=hidden_dim, state_dim=state_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(
        f"[wm04] Loaded SA world model from {ckpt_path}\n"
        f"       state_dim={state_dim}, action_dim={action_dim}, "
        f"input_dim={input_dim}, hidden_dim={hidden_dim}"
    )

    return model, state_dim, action_dim


# ---------------------------------------------------------------------
# Rollout + optimisation
# ---------------------------------------------------------------------


def sample_initial_states(S0: np.ndarray, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Sample a batch of initial states from S0 and move to device.

    S0: [N, state_dim]
    return: [B, state_dim]
    """
    N = S0.shape[0]
    idx = np.random.randint(0, N, size=batch_size)
    batch = S0[idx]
    return torch.from_numpy(batch).to(device)


def evaluate_constant_policy(
    model: MLPWorldModel,
    state_dim: int,
    S0: np.ndarray,
    action_vec: torch.Tensor,
    cfg: RLConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Roll out the world model starting from a batch of initial states,
    using a CONSTANT action at every step.

    Reward proxy:
      r_t = next_state[:, 0]   # assume state[0] ~ pnl_total_norm

    Returns:
      mean discounted return over the batch (scalar tensor).
    """
    B = cfg.batch_size
    gamma = cfg.gamma

    # Sample initial states. Shape: [B, state_dim]
    s = sample_initial_states(S0, batch_size=B, device=device)

    # Expand action to [B, action_dim]
    a = action_vec.unsqueeze(0).expand(B, -1)

    G = torch.zeros((), device=device)
    discount = 1.0

    for t in range(cfg.horizon):
        x = torch.cat([s, a], dim=-1)  # [B, state_dim + action_dim]
        s_next = model(x)              # [B, state_dim]

        # Reward proxy: first state feature of next state.
        r_t = s_next[:, 0]             # [B]
        G = G + discount * r_t.mean()

        s = s_next
        discount *= gamma

    return G


def optimise_constant_action(
    model: MLPWorldModel,
    state_dim: int,
    action_dim: int,
    S0: np.ndarray,
    A_data: np.ndarray,
    cfg: RLConfig,
    device: torch.device,
):
    """
    Gradient-based optimisation of a constant action vector.

    - We initialise from the dataset mean action.
    - At each step we:
        * sample a batch of initial states,
        * roll out horizon H in the world model,
        * compute discounted return G(a),
        * ascend its gradient w.r.t. 'a' (via Adam).

    Returns:
        (best_action_np, history_dict)
    """
    # Compute dataset stats for logging / regularisation.
    a_mean = A_data.mean(axis=0)
    a_std = A_data.std(axis=0) + 1e-6
    a_min = A_data.min(axis=0)
    a_max = A_data.max(axis=0)

    print("[wm04] Dataset action stats per dim:")
    for d in range(action_dim):
        print(
            f"  dim {d}: mean={a_mean[d]: .6f}, std={a_std[d]: .6f}, "
            f"min={a_min[d]: .6f}, max={a_max[d]: .6f}"
        )

    # Parameter: start from mean dataset action.
    a_param = torch.tensor(a_mean, dtype=torch.float32, device=device, requires_grad=True)

    opt = torch.optim.Adam([a_param], lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_G = -math.inf
    best_a = None

    history = {
        "iter": [],
        "G": [],
    }

    for it in range(1, cfg.num_iters + 1):
        opt.zero_grad()

        G = evaluate_constant_policy(
            model=model,
            state_dim=state_dim,
            S0=S0,
            action_vec=a_param,
            cfg=cfg,
            device=device,
        )

        # L2 regularisation in units of dataset std.
        l2_reg = cfg.action_l2_reg * ((a_param - torch.from_numpy(a_mean).to(device)) / torch.from_numpy(a_std).to(device)).pow(2).mean()

        # We *maximise* G, so loss = -(G - l2_reg).
        loss = -(G - l2_reg)

        loss.backward()
        opt.step()

        G_val = G.item()
        history["iter"].append(it)
        history["G"].append(G_val)

        if G_val > best_G:
            best_G = G_val
            best_a = a_param.detach().cpu().numpy().copy()

        if it % 20 == 0 or it == 1:
            print(
                f"[wm04][iter {it:04d}] "
                f"G={G_val: .6f}, "
                f"||a||={a_param.norm().item(): .6f}"
            )

    return best_a, best_G, (a_mean, a_std, a_min, a_max), history


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    ROOT = Path(".").resolve()
    cfg = RLConfig()

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[wm04] Using device: {device}")

    # 1) Load SA trajectories.
    runs_csv = ROOT / "runs" / "exp04_multi_scenario" / "exp04_runs_with_ts.csv"
    print(f"[wm04] Loading SA trajectories from {runs_csv} ...")
    trajs = load_sa_trajectories(runs_csv)
    print(f"[wm04] Loaded {len(trajs)} SA trajectories.")

    S0, A_data = extract_initial_states_and_actions(trajs)
    state_dim = S0.shape[1]
    action_dim = A_data.shape[1]
    print(f"[wm04] state_dim={state_dim}, action_dim={action_dim}")

    # 2) Load SA world model.
    ckpt_path = ROOT / "runs" / "wm03_world_model_sa" / "world_model_sa_mlp.pt"
    model, state_dim_ckpt, action_dim_ckpt = build_world_model(ckpt_path, device=device)

    assert state_dim_ckpt == state_dim, "state_dim mismatch between dataset and checkpoint"
    assert action_dim_ckpt == action_dim, "action_dim mismatch between dataset and checkpoint"

    # 3) Optimise constant action vector.
    best_a, best_G, stats, history = optimise_constant_action(
        model=model,
        state_dim=state_dim,
        action_dim=action_dim,
        S0=S0,
        A_data=A_data,
        cfg=cfg,
        device=device,
    )

    a_mean, a_std, a_min, a_max = stats

    # 4) Save results.
    out_dir = ROOT / "runs" / "wm04_constant_policy"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "optimised_action.npy", best_a)

    summary_path = out_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write("wm04: constant policy optimisation (world-model based)\n")
        f.write(f"device: {device}\n")
        f.write(f"horizon: {cfg.horizon}\n")
        f.write(f"batch_size: {cfg.batch_size}\n")
        f.write(f"num_iters: {cfg.num_iters}\n")
        f.write(f"gamma: {cfg.gamma}\n")
        f.write(f"action_l2_reg: {cfg.action_l2_reg}\n\n")

        f.write("Dataset action stats per dim:\n")
        for d in range(action_dim):
            f.write(
                f"  dim {d}: mean={a_mean[d]: .6f}, std={a_std[d]: .6f}, "
                f"min={a_min[d]: .6f}, max={a_max[d]: .6f}\n"
            )

        f.write("\nBest constant action (in raw SA feature space):\n")
        f.write("  " + " ".join(f"{x:.6f}" for x in best_a) + "\n")
        f.write(f"\nBest discounted return G: {best_G:.6f}\n")

    print(f"[wm04] Saved optimised action to {out_dir / 'optimised_action.npy'}")
    print(f"[wm04] Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
