#!/usr/bin/env python3
"""
RL-2: Behaviour Cloning evaluation script.

Evaluates BC model performance on:
1. Holdout data: action prediction error (MSE, MAE)
2. Rollouts: PnL, risk metrics, kill rate compared to heuristic

Usage:
    python eval_bc.py --model-dir runs/rl2_model --eval-holdout --eval-rollout

Requirements:
    - torch
    - numpy
    - paraphina_env (for rollouts)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    print("Error: numpy not found. Please install: pip install numpy")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
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


class BCDataset(Dataset):
    """Dataset for behaviour cloning evaluation."""

    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        self.observations = torch.from_numpy(observations).float()
        self.actions = torch.from_numpy(actions).float()

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.observations[idx], self.actions[idx]


class BCPolicy(nn.Module):
    """MLP policy for behaviour cloning (must match training)."""

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

        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


def evaluate_holdout(
    model: nn.Module,
    data_dir: Path,
    device: torch.device,
    batch_size: int = 256,
) -> Dict:
    """Evaluate model on holdout data."""
    print("Evaluating on holdout data...")

    # Load data
    observations = np.load(data_dir / "observations.npy")
    actions = np.load(data_dir / "actions.npy")

    # Use last 10% as holdout
    n_holdout = max(1, int(len(observations) * 0.1))
    holdout_obs = observations[-n_holdout:]
    holdout_actions = actions[-n_holdout:]

    dataset = BCDataset(holdout_obs, holdout_actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for obs, target_actions in loader:
            obs = obs.to(device)
            pred_actions = model(obs)
            all_preds.append(pred_actions.cpu().numpy())
            all_targets.append(target_actions.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Compute metrics
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(mse)

    # Per-dimension MAE
    per_dim_mae = np.mean(np.abs(preds - targets), axis=0)

    # Max absolute error
    max_error = np.max(np.abs(preds - targets))

    results = {
        "n_samples": len(holdout_obs),
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "max_error": float(max_error),
        "per_dim_mae": per_dim_mae.tolist(),
    }

    print(f"  Samples:    {len(holdout_obs)}")
    print(f"  MSE:        {mse:.6f}")
    print(f"  MAE:        {mae:.6f}")
    print(f"  RMSE:       {rmse:.6f}")
    print(f"  Max error:  {max_error:.6f}")

    return results


def evaluate_rollouts(
    model: nn.Module,
    num_episodes: int,
    device: torch.device,
    seed: int = 42,
    max_ticks: int = 1000,
) -> Dict:
    """Evaluate model through rollouts compared to heuristic."""
    print(f"Evaluating through rollouts ({num_episodes} episodes)...")

    try:
        import paraphina_env
    except ImportError:
        print("  Warning: paraphina_env not available, skipping rollouts")
        return {"error": "paraphina_env not available"}

    model.eval()

    # Run heuristic baseline
    print("  Running heuristic baseline...")
    heuristic_results = run_heuristic_episodes(
        num_episodes=num_episodes,
        seed=seed,
        max_ticks=max_ticks,
    )

    # Run BC policy
    print("  Running BC policy...")
    bc_results = run_bc_policy_episodes(
        model=model,
        device=device,
        num_episodes=num_episodes,
        seed=seed,
        max_ticks=max_ticks,
    )

    # Compute comparison metrics
    results = {
        "heuristic": heuristic_results,
        "bc_policy": bc_results,
        "comparison": {
            "pnl_ratio": bc_results["mean_pnl"] / max(heuristic_results["mean_pnl"], 1e-6),
            "kill_rate_ratio": bc_results["kill_rate"] / max(heuristic_results["kill_rate"], 1e-6),
            "pnl_diff": bc_results["mean_pnl"] - heuristic_results["mean_pnl"],
            "kill_rate_diff": bc_results["kill_rate"] - heuristic_results["kill_rate"],
        },
    }

    print()
    print("  Results comparison:")
    print(f"                    Heuristic    BC Policy    Ratio")
    print(f"    Mean PnL:       {heuristic_results['mean_pnl']:8.2f}     {bc_results['mean_pnl']:8.2f}     {results['comparison']['pnl_ratio']:.3f}")
    print(f"    Kill rate:      {heuristic_results['kill_rate']:8.2%}     {bc_results['kill_rate']:8.2%}     {results['comparison']['kill_rate_ratio']:.3f}")
    print(f"    Mean length:    {heuristic_results['mean_length']:8.1f}     {bc_results['mean_length']:8.1f}")

    return results


def run_heuristic_episodes(
    num_episodes: int,
    seed: int,
    max_ticks: int,
) -> Dict:
    """Run heuristic policy episodes and collect stats."""
    import paraphina_env

    env = paraphina_env.Env(
        max_ticks=max_ticks,
        apply_domain_rand=False,
        domain_rand_preset="deterministic",
    )

    pnls = []
    kills = 0
    lengths = []

    for ep in range(num_episodes):
        ep_seed = seed + ep * 12345
        env.reset(seed=ep_seed)

        done = False
        step = 0
        while not done:
            action = env.identity_action()
            obs, reward, done, info = env.step(action)
            step += 1

        pnls.append(info["pnl_total"])
        if info["kill_switch"]:
            kills += 1
        lengths.append(step)

    return {
        "mean_pnl": float(np.mean(pnls)),
        "std_pnl": float(np.std(pnls)),
        "min_pnl": float(np.min(pnls)),
        "max_pnl": float(np.max(pnls)),
        "kill_rate": kills / num_episodes,
        "mean_length": float(np.mean(lengths)),
    }


def run_bc_policy_episodes(
    model: nn.Module,
    device: torch.device,
    num_episodes: int,
    seed: int,
    max_ticks: int,
) -> Dict:
    """Run BC policy episodes and collect stats."""
    import paraphina_env

    env = paraphina_env.Env(
        max_ticks=max_ticks,
        apply_domain_rand=False,
        domain_rand_preset="deterministic",
    )

    num_venues = env.num_venues

    pnls = []
    kills = 0
    lengths = []

    model.eval()

    for ep in range(num_episodes):
        ep_seed = seed + ep * 12345
        obs_dict = env.reset(seed=ep_seed)

        done = False
        step = 0
        while not done:
            # Convert observation dict to feature vector
            obs_features = obs_dict_to_features(obs_dict, num_venues)
            obs_tensor = torch.from_numpy(obs_features).float().unsqueeze(0).to(device)

            # Get BC policy action
            with torch.no_grad():
                action_vec = model(obs_tensor).squeeze(0).cpu().numpy()

            # Convert action vector to dict
            action_dict = action_vec_to_dict(action_vec, num_venues)

            obs_dict, reward, done, info = env.step(action_dict)
            step += 1

        pnls.append(info["pnl_total"])
        if info["kill_switch"]:
            kills += 1
        lengths.append(step)

    return {
        "mean_pnl": float(np.mean(pnls)),
        "std_pnl": float(np.std(pnls)),
        "min_pnl": float(np.min(pnls)),
        "max_pnl": float(np.max(pnls)),
        "kill_rate": kills / num_episodes,
        "mean_length": float(np.mean(lengths)),
    }


def obs_dict_to_features(obs_dict: dict, num_venues: int) -> np.ndarray:
    """Convert observation dict to flat feature vector."""
    # Global features (17)
    global_features = [
        obs_dict.get("fair_value") or 0.0,
        obs_dict.get("sigma_eff", 0.0),
        obs_dict.get("vol_ratio_clipped", 1.0),
        obs_dict.get("spread_mult", 1.0),
        obs_dict.get("size_mult", 1.0),
        obs_dict.get("band_mult", 1.0),
        obs_dict.get("q_global_tao", 0.0),
        obs_dict.get("dollar_delta_usd", 0.0),
        obs_dict.get("delta_limit_usd", 0.0),
        obs_dict.get("basis_usd", 0.0),
        obs_dict.get("basis_gross_usd", 0.0),
        obs_dict.get("basis_limit_warn_usd", 0.0),
        obs_dict.get("basis_limit_hard_usd", 0.0),
        obs_dict.get("daily_realised_pnl", 0.0),
        obs_dict.get("daily_unrealised_pnl", 0.0),
        obs_dict.get("daily_pnl_total", 0.0),
        1.0 if obs_dict.get("kill_switch") else 0.0,
    ]

    # Per-venue features (14 per venue)
    venue_features = []
    venues = obs_dict.get("venues", [])

    for i in range(num_venues):
        if i < len(venues):
            v = venues[i]
            mid = v.get("mid") or 0.0
            spread = v.get("spread") or 0.0
            status = v.get("status", "Healthy")

            # One-hot status
            h, w, d = 0.0, 0.0, 0.0
            if "Healthy" in status:
                h = 1.0
            elif "Warning" in status:
                w = 1.0
            elif "Disabled" in status:
                d = 1.0
            else:
                h = 1.0  # Default to healthy

            venue_features.extend([
                mid,
                spread,
                v.get("depth_near_mid", 0.0),
                v.get("staleness_ms") or 0,
                v.get("local_vol_short", 0.0),
                v.get("local_vol_long", 0.0),
                h, w, d,
                v.get("toxicity", 0.0),
                v.get("position_tao", 0.0),
                v.get("avg_entry_price", 0.0),
                v.get("funding_8h", 0.0),
                v.get("margin_available_usd", 0.0),
                v.get("dist_liq_sigma", 10.0),
            ])
        else:
            venue_features.extend([0.0] * 14)

    return np.array(global_features + venue_features, dtype=np.float32)


def action_vec_to_dict(action_vec: np.ndarray, num_venues: int) -> dict:
    """Convert action vector to action dict for environment."""
    # Decode normalized action vector
    # Layout: [spread_scale_0..n, size_scale_0..n, rprice_offset_0..n, hedge_scale, hedge_weights_0..n]

    idx = 0

    # Per-venue: spread_scale (from [0,1] to [0.5, 3.0])
    spread_scale = []
    for _ in range(num_venues):
        val = action_vec[idx]
        spread_scale.append(0.5 + val * 2.5)  # [0.5, 3.0]
        idx += 1

    # Per-venue: size_scale (from [0,1] to [0.0, 2.0])
    size_scale = []
    for _ in range(num_venues):
        val = action_vec[idx]
        size_scale.append(val * 2.0)  # [0.0, 2.0]
        idx += 1

    # Per-venue: rprice_offset (from [-1,1] to [-10, 10])
    rprice_offset = []
    for _ in range(num_venues):
        val = action_vec[idx]
        rprice_offset.append(val * 10.0)  # [-10, 10]
        idx += 1

    # Global: hedge_scale (from [0,1] to [0.0, 2.0])
    hedge_scale = action_vec[idx] * 2.0
    idx += 1

    # Global: hedge_venue_weights (already in [0,1], normalize to sum=1)
    hedge_weights = []
    for _ in range(num_venues):
        hedge_weights.append(max(0.0, action_vec[idx]))
        idx += 1

    weight_sum = sum(hedge_weights)
    if weight_sum > 0:
        hedge_weights = [w / weight_sum for w in hedge_weights]
    else:
        hedge_weights = [1.0 / num_venues] * num_venues

    return {
        "spread_scale": spread_scale,
        "size_scale": size_scale,
        "rprice_offset_usd": rprice_offset,
        "hedge_scale": hedge_scale,
        "hedge_venue_weights": hedge_weights,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC policy")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing trained model",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory for holdout evaluation (default: from training metadata)",
    )
    parser.add_argument(
        "--eval-holdout",
        action="store_true",
        help="Evaluate on holdout data",
    )
    parser.add_argument(
        "--eval-rollout",
        action="store_true",
        help="Evaluate through rollouts",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of rollout episodes (default: 100)",
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

    args = parser.parse_args()

    if not args.eval_holdout and not args.eval_rollout:
        print("Error: specify at least one of --eval-holdout or --eval-rollout")
        sys.exit(1)

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"RL-2 Behaviour Cloning Evaluation")
    print(f"==================================")
    print(f"Model:          {args.model_dir}")
    print(f"Device:         {device}")
    print(f"Seed:           {args.seed}")
    print()

    set_seed(args.seed)

    # Load model
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)

    with open(model_dir / "training_metadata.json", "r") as f:
        metadata = json.load(f)

    obs_dim = metadata["obs_dim"]
    action_dim = metadata["action_dim"]
    hidden_dims = metadata["hidden_dims"]
    activation = metadata.get("activation", "relu")

    print(f"Model config:")
    print(f"  obs_dim:     {obs_dim}")
    print(f"  action_dim:  {action_dim}")
    print(f"  hidden_dims: {hidden_dims}")
    print()

    model = BCPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        activation=activation,
    ).to(device)

    # Load weights (prefer best model if available)
    if (model_dir / "model_best.pt").exists():
        model.load_state_dict(torch.load(model_dir / "model_best.pt", map_location=device))
        print("Loaded model_best.pt")
    else:
        model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
        print("Loaded model.pt")
    print()

    results = {}

    # Holdout evaluation
    if args.eval_holdout:
        data_dir = Path(args.data_dir) if args.data_dir else Path(metadata["data_dir"])
        if not data_dir.exists():
            print(f"Warning: Data directory not found: {data_dir}")
        else:
            results["holdout"] = evaluate_holdout(model, data_dir, device)
            print()

    # Rollout evaluation
    if args.eval_rollout:
        results["rollout"] = evaluate_rollouts(
            model=model,
            num_episodes=args.num_episodes,
            device=device,
            seed=args.seed,
            max_ticks=metadata.get("data_metadata", {}).get("max_ticks", 1000),
        )
        print()

    # Save results
    results_path = model_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")

    # Check exit criteria
    print()
    print("Exit criteria check:")

    passed = True

    if "holdout" in results:
        mae = results["holdout"]["mae"]
        if mae < 0.1:
            print(f"  [PASS] Holdout MAE = {mae:.4f} < 0.1")
        else:
            print(f"  [FAIL] Holdout MAE = {mae:.4f} >= 0.1")
            passed = False

    if "rollout" in results and "comparison" in results["rollout"]:
        comp = results["rollout"]["comparison"]

        # Kill rate check: BC <= heuristic * 1.1
        heur_kr = results["rollout"]["heuristic"]["kill_rate"]
        bc_kr = results["rollout"]["bc_policy"]["kill_rate"]
        if heur_kr == 0:
            kr_ok = bc_kr <= 0.05  # If heuristic has 0 kills, allow up to 5%
        else:
            kr_ok = bc_kr <= heur_kr * 1.1

        if kr_ok:
            print(f"  [PASS] Kill rate ratio = {comp['kill_rate_ratio']:.3f} (BC={bc_kr:.2%}, Heur={heur_kr:.2%})")
        else:
            print(f"  [FAIL] Kill rate ratio = {comp['kill_rate_ratio']:.3f} (BC={bc_kr:.2%}, Heur={heur_kr:.2%})")
            passed = False

        # PnL check: BC >= heuristic * 0.9
        pnl_ok = comp["pnl_ratio"] >= 0.9
        if pnl_ok:
            print(f"  [PASS] PnL ratio = {comp['pnl_ratio']:.3f}")
        else:
            print(f"  [FAIL] PnL ratio = {comp['pnl_ratio']:.3f} < 0.9")
            passed = False

    if passed:
        print()
        print("All exit criteria PASSED!")
        return 0
    else:
        print()
        print("Some exit criteria FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

