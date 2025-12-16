#!/usr/bin/env python3
"""
wm01_train_world_model.py

Phase-0 world model for Paraphina.

Goal
=====
Train a simple neural net to predict the *next decision-level state*
from the current state, using the trajectories built by
world_model_dataset.py from exp04 telemetry.

- Input:  state_t   ∈ R^D
- Target: state_{t+1} ∈ R^D
- Model:  2-layer MLP (ReLU), trained with MSE.

This ignores actions for now (we haven't logged config tweaks yet);
it's a first, clean world-model that learns the endogenous dynamics
of PnL / risk / inventory under a fixed policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from world_model_dataset import build_trajectories_from_runs_csv, DecisionTrajectory


# ---------------------------------------------------------------------------
# Dataset: (state_t, state_{t+1}) pairs
# ---------------------------------------------------------------------------

class StateTransitionDataset(Dataset):
    """Flatten all trajectories into supervised (s_t, s_{t+1}) pairs."""

    def __init__(self, trajectories: List[DecisionTrajectory]) -> None:
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []

        for tr in trajectories:
            states = tr.states.astype(np.float32)  # [T, D]
            if states.shape[0] < 2:
                continue
            xs.append(states[:-1])
            ys.append(states[1:])

        if not xs:
            raise ValueError("No usable trajectories (need at least 2 states each).")

        self.x = np.concatenate(xs, axis=0)  # [N, D]
        self.y = np.concatenate(ys, axis=0)  # [N, D]

        assert self.x.shape == self.y.shape
        self.state_dim = self.x.shape[1]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.y[idx]),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPWorldModel(nn.Module):
    """Simple 2-layer MLP world model: s_t -> s_{t+1}."""

    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
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
    trajectories: List[DecisionTrajectory],
    cfg: TrainConfig,
) -> Tuple[DataLoader, DataLoader, int]:
    dataset = StateTransitionDataset(trajectories)
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

    return train_loader, val_loader, dataset.state_dim


def train_world_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    state_dim: int,
    cfg: TrainConfig,
) -> MLPWorldModel:
    device = torch.device(cfg.device)
    model = MLPWorldModel(state_dim=state_dim, hidden_dim=cfg.hidden_dim).to(device)

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
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
        )

    return model


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    ROOT = Path(__file__).resolve().parents[1]
    runs_csv = ROOT / "runs" / "exp04_multi_scenario" / "exp04_runs_with_ts.csv"

    print(f"[wm01] Loading trajectories from {runs_csv} ...")
    trajectories = build_trajectories_from_runs_csv(
        runs_csv,
        decision_horizon=10,
    )
    print(f"[wm01] Loaded {len(trajectories)} trajectories.")

    cfg = TrainConfig()
    print(f"[wm01] Using device: {cfg.device}")

    train_loader, val_loader, state_dim = make_dataloaders(trajectories, cfg)
    print(f"[wm01] state_dim={state_dim}, "
          f"train_batches={len(train_loader)}, val_batches={len(val_loader)}")

    model = train_world_model(train_loader, val_loader, state_dim, cfg)

    # Save model weights
    out_dir = ROOT / "runs" / "wm01_world_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "world_model_mlp.pt"
    torch.save(
        {
            "state_dim": state_dim,
            "hidden_dim": cfg.hidden_dim,
            "model_state_dict": model.state_dict(),
            "train_config": cfg.__dict__,
        },
        ckpt_path,
    )
    print(f"[wm01] Saved world model checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
