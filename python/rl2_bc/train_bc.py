#!/usr/bin/env python3
"""
RL-2: Behaviour Cloning training script.

Trains an MLP policy to imitate the HeuristicPolicy using supervised learning.
Uses deterministic seeding for reproducibility.

Usage:
    python train_bc.py --data-dir runs/rl2_data --output-dir runs/rl2_model

Requirements:
    - torch
    - numpy
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    print("Error: numpy not found. Please install: pip install numpy")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, random_split
except ImportError:
    print("Error: torch not found. Please install: pip install torch")
    sys.exit(1)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Enable deterministic algorithms (may reduce performance)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class BCDataset(Dataset):
    """Dataset for behaviour cloning."""

    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        self.observations = torch.from_numpy(observations).float()
        self.actions = torch.from_numpy(actions).float()

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.observations[idx], self.actions[idx]


class BCPolicy(nn.Module):
    """MLP policy for behaviour cloning."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        # Build MLP layers
        layers = []
        in_dim = obs_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.01))
            in_dim = h_dim

        # Output layer
        layers.append(nn.Linear(in_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for obs, actions in loader:
        obs = obs.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()
        pred_actions = model(obs)
        loss = nn.functional.mse_loss(pred_actions, actions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model, return (mse_loss, mae)."""
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    with torch.no_grad():
        for obs, actions in loader:
            obs = obs.to(device)
            actions = actions.to(device)

            pred_actions = model(obs)
            mse = nn.functional.mse_loss(pred_actions, actions, reduction="sum")
            mae = torch.abs(pred_actions - actions).sum()

            total_mse += mse.item()
            total_mae += mae.item()
            n_samples += obs.size(0) * actions.size(1)

    return total_mse / n_samples, total_mae / n_samples


def main():
    parser = argparse.ArgumentParser(
        description="Train BC policy to imitate HeuristicPolicy"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing dataset (observations.npy, actions.npy)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/rl2_model",
        help="Output directory for model and metadata (default: runs/rl2_model)",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden layer dimensions (default: 256 256)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "leaky_relu"],
        help="Activation function (default: relu)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split fraction (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save checkpoint every N epochs (0 = only final, default: 0)",
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"RL-2 Behaviour Cloning Training")
    print(f"================================")
    print(f"Data:           {args.data_dir}")
    print(f"Output:         {args.output_dir}")
    print(f"Hidden dims:    {args.hidden_dims}")
    print(f"Activation:     {args.activation}")
    print(f"Learning rate:  {args.lr}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Epochs:         {args.epochs}")
    print(f"Val split:      {args.val_split}")
    print(f"Seed:           {args.seed}")
    print(f"Device:         {device}")
    print()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load data
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    print("Loading data...")
    observations = np.load(data_dir / "observations.npy")
    actions = np.load(data_dir / "actions.npy")

    # Load metadata
    with open(data_dir / "metadata.json", "r") as f:
        data_metadata = json.load(f)

    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]
    n_samples = len(observations)

    print(f"  Samples:      {n_samples}")
    print(f"  Obs dim:      {obs_dim}")
    print(f"  Action dim:   {action_dim}")
    print()

    # Create dataset and split
    dataset = BCDataset(observations, actions)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    # Use generator for reproducible split
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"Train samples: {train_size}")
    print(f"Val samples:   {val_size}")
    print()

    # Create model
    model = BCPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        activation=args.activation,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("Training...")
    train_losses = []
    val_losses = []
    val_maes = []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_mae = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "model_best.pt")

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{args.epochs}: "
                f"train_loss={train_loss:.6f}, "
                f"val_loss={val_loss:.6f}, "
                f"val_mae={val_mae:.6f}"
            )

        # Checkpoint saving
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), output_dir / f"model_epoch{epoch+1}.pt")

    print()
    print(f"Training complete!")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Final val MAE: {val_maes[-1]:.6f}")

    # Save final model
    torch.save(model.state_dict(), output_dir / "model.pt")

    # Save training metadata
    training_metadata = {
        "data_dir": str(args.data_dir),
        "data_metadata": data_metadata,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dims": args.hidden_dims,
        "activation": args.activation,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "val_split": args.val_split,
        "seed": args.seed,
        "device": str(device),
        "n_params": n_params,
        "train_samples": train_size,
        "val_samples": val_size,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "final_val_mae": val_maes[-1],
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_maes": val_maes,
    }

    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(training_metadata, f, indent=2)

    print()
    print(f"Saved to: {output_dir}")
    print(f"  model.pt")
    print(f"  model_best.pt")
    print(f"  training_metadata.json")

    # Try to save training curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        axes[0].plot(train_losses, label="Train")
        axes[0].plot(val_losses, label="Val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE Loss")
        axes[0].set_title("Training Curves")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE curve
        axes[1].plot(val_maes)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Val MAE")
        axes[1].set_title("Validation MAE")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=150)
        plt.close()
        print(f"  training_curves.png")
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

