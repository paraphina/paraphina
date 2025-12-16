#!/usr/bin/env python3
"""
wm03_train_world_model_sa.py

State+action world model for Paraphina.

We train a neural net f_theta(s_t, a_t) -> s_{t+1}, where:

- s_t: decision-level state vector (11 dims from world_model_dataset)
- a_t: normalised action vector derived from run-level config (6 dims)

For exp04 data, actions are *constant within each run* (per-episode config),
but vary across runs. This still lets the model learn how different configs
(band_base, mm_size_eta, vol_ref, etc.) change the dynamics.

This is the final major building block before RL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from world_model_dataset import (
    build_trajectories_sa_from_runs_csv,
    DecisionTrajectorySA,
)


# ---------------------------------------------------------------------------
# Dataset: (s_t, a_t) -> s_{t+1}
# ---------------------------------------------------------------------------

class StateActionTransitionDataset(Dataset):
    """Flatten all SA trajectories into supervised (s_t, a_t, s_{t+1}) pairs."""

    def __init__(self, trajectories: List[DecisionTrajectorySA]) -> None:
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []

        for tr in trajectories:
            s = tr.states.astype(np.float32)   # [T, D_s]
            a = tr.actions.astype(np.float32)  # [T, D_a]

            if s.shape[0] < 2:
                continue

            sa_t = np.concatenate([s[:-1], a[:-1]], axis=1)  # [T-1, D_s + D_a]
            s_next = s[1:]                                   # [T-1, D_s]

            xs.append(sa_t)
            ys.append(s_next)

        if not xs:
            raise ValueError("No usable trajectories (need at least 2 states each).")

        self.x = np.concatenate(xs, axis=0)
        self.y = np.concatenate(ys, axis=0)

        assert self.x.shape[0] == self.y.shape[0]
        self.state_dim = self.y.shape[1]
        self.action_dim = self.x.shape[1] - self.state_dim
        self.input_dim = self.x.shape[1]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x[idx]),  # [D_s + D_a]
            torch.from_numpy(self.y[idx]),  # [D_s]
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPWorldModelSA(nn.Module):
    """2-layer MLP world model: concat(s_t, a_t) -> s_{t+1}."""

    def __init__(self, input_dim: int, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    batch_size: int = 256
    hidden_dim: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 20
    train_frac: float = 0.9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def make_dataloaders(
    trajectories: List[DecisionTrajectorySA],
    cfg: TrainConfig,
) -> Tuple[DataLoader, DataLoader, int, int, int]:
    dataset = StateActionTransitionDataset(trajectories)
    n_total = len(dataset)
    n_train = int(cfg.train_frac * n_total)
    n_val = n_total - n_train

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False
    )

    return (
        train_loader,
        val_loader,
        dataset.state_dim,
        dataset.action_dim,
        dataset.input_dim,
    )


def train_world_model_sa(
    train_loader: DataLoader,
    val_loader: DataLoader,
    state_dim: int,
    input_dim: int,
    cfg: TrainConfig,
) -> MLPWorldModelSA:
    device = torch.device(cfg.device)
    model = MLPWorldModelSA(input_dim=input_dim, state_dim=state_dim, hidden_dim=cfg.hidden_dim).to(device)

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = nn.MSELoss()

    def _run_epoch(loader: DataLoader, train: bool) -> float:
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            with torch.set_grad_enabled(train):
                y_pred = model(x)
                loss = criterion(y_pred, y)

            if train:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            total_loss += float(loss.item())
            n_batches += 1

        return total_loss / max(n_batches, 1)

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = _run_epoch(train_loader, train=True)
        val_loss = _run_epoch(val_loader, train=False)

        print(
            f"[wm03][epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
        )

    return model


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    ROOT = Path(__file__).resolve().parents[1]
    runs_csv = ROOT / "runs" / "exp04_multi_scenario" / "exp04_runs_with_ts.csv"

    print(f"[wm03] Loading SA trajectories from {runs_csv} ...")
    trajectories = build_trajectories_sa_from_runs_csv(
        runs_csv,
        decision_horizon=10,
    )
    print(f"[wm03] Loaded {len(trajectories)} SA trajectories.")

    cfg = TrainConfig()
    print(f"[wm03] Using device: {cfg.device}")

    (
        train_loader,
        val_loader,
        state_dim,
        action_dim,
        input_dim,
    ) = make_dataloaders(trajectories, cfg)

    print(
        f"[wm03] state_dim={state_dim}, action_dim={action_dim}, "
        f"input_dim={input_dim}, train_batches={len(train_loader)}, "
        f"val_batches={len(val_loader)}"
    )

    model = train_world_model_sa(train_loader, val_loader, state_dim, input_dim, cfg)

    # Save model weights
    out_dir = ROOT / "runs" / "wm03_world_model_sa"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "world_model_sa_mlp.pt"
    torch.save(
        {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "input_dim": input_dim,
            "hidden_dim": cfg.hidden_dim,
            "model_state_dict": model.state_dict(),
            "train_config": cfg.__dict__,
        },
        ckpt_path,
    )
    print(f"[wm03] Saved SA world model checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
