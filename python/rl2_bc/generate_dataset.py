#!/usr/bin/env python3
"""
RL-2: Dataset generation for behaviour cloning.

Collects trajectories from the HeuristicPolicy using the paraphina_env
simulation environment. Outputs NPZ-compatible arrays + JSON metadata.

Usage:
    python generate_dataset.py --num-episodes 100 --output-dir runs/rl2_data

Requirements:
    - paraphina_env Rust library must be built and installed
    - numpy for array storage
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Check if paraphina_env is available
try:
    import paraphina_env
except ImportError:
    print("Error: paraphina_env not found.")
    print("Please build and install the Rust library first:")
    print("  cd paraphina_env && maturin develop --release")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: numpy not found. Please install: pip install numpy")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate BC training dataset from HeuristicPolicy"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to collect (default: 100)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=1000,
        help="Maximum ticks per episode (default: 1000)",
    )
    parser.add_argument(
        "--domain-rand",
        type=str,
        default="deterministic",
        choices=["default", "mild", "deterministic"],
        help="Domain randomisation preset (default: deterministic)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/rl2_data",
        help="Output directory (default: runs/rl2_data)",
    )
    parser.add_argument(
        "--no-domain-rand",
        action="store_true",
        help="Disable domain randomisation (equivalent to --domain-rand deterministic)",
    )

    args = parser.parse_args()

    # Handle --no-domain-rand flag
    apply_domain_rand = not args.no_domain_rand
    if args.no_domain_rand:
        domain_rand_preset = "deterministic"
    else:
        domain_rand_preset = args.domain_rand
        if domain_rand_preset == "deterministic":
            apply_domain_rand = False

    print(f"RL-2 Dataset Generation")
    print(f"=======================")
    print(f"Episodes:        {args.num_episodes}")
    print(f"Parallel envs:   {args.num_envs}")
    print(f"Max ticks:       {args.max_ticks}")
    print(f"Domain rand:     {domain_rand_preset} (apply={apply_domain_rand})")
    print(f"Seed:            {args.seed}")
    print(f"Output:          {args.output_dir}")
    print()

    # Print version info
    print(f"Versions:")
    print(f"  obs_version:        {paraphina_env.obs_version()}")
    print(f"  action_version:     {paraphina_env.action_version()}")
    print(f"  trajectory_version: {paraphina_env.trajectory_version()}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create trajectory collector
    print("Creating trajectory collector...")
    collector = paraphina_env.TrajectoryCollector(
        num_envs=args.num_envs,
        max_ticks=args.max_ticks,
        apply_domain_rand=apply_domain_rand,
        domain_rand_preset=domain_rand_preset,
        base_seed=args.seed,
    )

    # Collect trajectories
    print(f"Collecting {args.num_episodes} episodes...")
    data = collector.collect_arrays(args.num_episodes)

    # Extract arrays and metadata
    obs_dim = data["obs_dim"]
    action_dim = data["action_dim"]
    num_samples = data["num_samples"]
    metadata = data["metadata"]

    print(f"Collected {num_samples} transitions")
    print(f"  obs_dim:    {obs_dim}")
    print(f"  action_dim: {action_dim}")
    print()

    # Reshape flat arrays to proper dimensions
    observations = np.array(data["observations"], dtype=np.float32).reshape(
        num_samples, obs_dim
    )
    actions = np.array(data["actions"], dtype=np.float32).reshape(num_samples, action_dim)
    rewards = np.array(data["rewards"], dtype=np.float32)
    terminals = np.array(data["terminals"], dtype=bool)
    episode_indices = np.array(data["episode_indices"], dtype=np.uint32)

    # Save arrays as NPY files
    print("Saving arrays...")
    np.save(output_dir / "observations.npy", observations)
    np.save(output_dir / "actions.npy", actions)
    np.save(output_dir / "rewards.npy", rewards)
    np.save(output_dir / "terminals.npy", terminals)
    np.save(output_dir / "episode_indices.npy", episode_indices)

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary statistics
    print()
    print("Dataset Summary:")
    print(f"  num_episodes:         {metadata['num_episodes']}")
    print(f"  num_transitions:      {metadata['num_transitions']}")
    print(f"  mean_episode_length:  {metadata['mean_episode_length']:.1f}")
    print(f"  kill_rate:            {metadata['kill_rate']:.2%}")
    print(f"  mean_pnl:             {metadata['mean_pnl']:.2f}")
    print()
    print(f"Files saved to: {output_dir}")
    print(f"  observations.npy  {observations.shape}")
    print(f"  actions.npy       {actions.shape}")
    print(f"  rewards.npy       {rewards.shape}")
    print(f"  terminals.npy     {terminals.shape}")
    print(f"  episode_indices.npy {episode_indices.shape}")
    print(f"  metadata.json")

    # Verify determinism by checking first few values
    print()
    print("Determinism check (first 3 obs features of first sample):")
    print(f"  {observations[0, :3]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

